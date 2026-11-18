"""The two-position turnover positioner with three load cells (product module of the final-assembly cell; grader side).

Source: the phase-1 prototype of this task (gate G2), copied and changed (the prototype directory stays read-only):
  -dev/proto/positioner/positioner.py      sha256 916cfede71467a24a6aee6b0d0291009759f2f1820d3b9b5ffd707e7279c4e8e
  -dev/proto/positioner/vehicle_units.py   sha256 9d27e3b657ba6db2d8f0bff393a4d4273d84b4908c50cadcdbccb3a399610aee
                                           (not used any more: the units come from the kit / the vehicle's unit map)
Kept from the prototype: the kinematic chain, the three tendon load cells under a 3-DOF base, the tare of the empty
positioner at each index position, the turn prescribed by a joint equality along a 6 s quintic (record G2 section 4:
a constraint, so the reaction reaches the base and the cells), the scrap rule at the start of every turn.
Changed / new (DESIGN v1.2 section 1.3, rulings L12, L13, M8; dev record P2_P10):
  * every dimension from the frozen cell layout and the fixture facts (`fixtures.Fixtures`), none typed in; the base
    vehicle's mass properties are an input (the kit's / the unit map's), not a classifier over the digest (the
    prototype's `KeyError: 'puck_screw'` on digest v3 came from that classifier);
  * the turn is ALWAYS prescribed (no servo mode); the cell's physics (0.5 ms steps, 10 ms tick) is the default;
  * the request rules of the cell's API (`request_index`): refused while moving, when no turn is left (four per
    episode), or when any part of the robot is inside the rotation column (`robot_in_column`, an exact convex distance
    to the column's cylinder); a request for the current index is a no-op that counts nothing;
  * the scrap rule asks the cell which units are fastened (`is_fastened`) and tells it which it scraps (`on_scrap`, the
    coupling's `scrap`) before it moves them off the bench; the vehicle box is the vehicle with every unit at home
    (fixture facts) + `scrap_margin_m`, in the vehicle's body frame; "on the vehicle" also counts a touch of the
    carrier (it turns with the vehicle);
  * the vehicle's mounting offset on the carrier (pins and indexing; drawn per episode by layout.py) is the vehicle
    body's pose in the carrier frame;
  * `LoadCellNoise`: the per-episode zero offset of each cell and the white noise of each sample, seeded.

Kinematic chain:
  bench (world) -> three load cells (tendon springs, world site on the bench top -> base site 15 mm above it)
  -> open-frame base `pos_base` (3 DOF: heave, roll, pitch about the cells' centroid; the cells' horizontal flexures
     are rigid) with the drive seat, the two outrigger beams and the cross member (their envelopes from the layout,
     lower faces raised to the cells' height so that only the cells carry the base)
  -> hinge `pos_index` (axis = the vehicle's x through its origin) -> carrier `pos_carrier` (fork on the side plates'
     bow edges, shaft to the seat) -> the vehicle body (rigid child of the carrier: one weld group, no contacts).
A cell's reading = its spring force k (L0 - L), compression positive, newton, order [A (under the drive seat),
B (station +y outrigger), C (station -y outrigger)], NOT turned into the vehicle frame (at keel_up the vehicle's y is
the station's -y). Three vertical supports on a 3-DOF base are statically determinate: the static readings are the
exact load split, whatever the stiffness split.

Use (the cell; P14):
  H = build_positioner(spec, fx, vehicle, mount_T=None, cfg=None, add_vehicle_geoms=fn(spec, body))  # into an MjSpec
  P = Positioner(model, data, H, fx, cfg, units=[body names], is_fastened=fn, on_scrap=fn, robot_geoms=[geom ids])
  P.place("keel_up"); every physics step: P.before_step(); mj_step
  P.request_index("deck_up") -> {"ok": True} | {"refused": "moving" | "no_turns_left" | "robot_in_column"}
  P.reading() -> tared newtons (3,);  noise = LoadCellNoise(seed); noise.sample(P.reading())
"""
import copy
import hashlib
import json
import math

import numpy as np

from . import fixtures as FX

__all__ = ["PHYSICS", "DEFAULT_CFG", "INDEX_ANGLE", "merge_cfg", "set_physics", "build_positioner", "Positioner",
           "LoadCellNoise", "compute_tare", "analytic_cells", "invert_cells", "standalone_spec", "quintic"]

INDEX_ANGLE = {"deck_up": 0.0, "keel_up": math.pi}

# the cell's physics (rulings L11 / M8): 0.5 ms implicitfast, 10 ms control tick (20 steps), elliptic cones,
# impratio 100, noslip 5, multi-CCD off; geom defaults condim 4, friction 0.8 / 0.005 / 0.0001, solref 0.004;
# hard welds and the index lock solref 0.002
PHYSICS = {"timestep": 0.0005, "tick_steps": 20, "impratio": 100.0, "noslip_iterations": 5, "multiccd": False,
           "geom_defaults": {"condim": 4, "friction": [0.8, 0.005, 0.0001], "solref": [0.004, 1.0],
                             "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
           "weld_solref": [0.002, 1.0], "weld_solimp": [0.99, 0.999, 0.001, 0.5, 2.0], "gravity": 9.80665,
           "arena_memory_bytes": 512 << 20}

DEFAULT_CFG = {
    "start_index": "keel_up",
    "turn_s": 6.0,                     # the 180 deg turn: a quintic in time (zero speed and acceleration at both ends)
    "settle_s": 0.3,                   # the turn ends (moving -> False) this long after the trajectory
    "max_turns": 4,                    # accepted turns per episode
    "turn_sign": {"keel_up->deck_up": -1, "deck_up->keel_up": 1},
    "scrap_margin_m": 0.03,            # the vehicle box of the scrap rule, grown by this
    "scrap_area": {"xy": [1.62, -0.60], "pitch_m": 0.08, "z_m": -0.90},   # off the bench (x > 1.45 m), below its top
    "column_margin_m": 0.0,            # robot_in_column: inside if the distance to the column is below this
    "noise": {"sigma_N": 0.4, "bias_max_N": 0.02, "rate_hz": 100.0},
    "reading_valid_after_s": 0.5,      # the manual: a reading is valid this long after an outside force is removed
    "names": {"base": "pos_base", "carrier": "pos_carrier", "hinge": "pos_index", "lock": "pos_index_lock",
              "vehicle": "vehicle", "column": "pos_column", "cell": "load_cell_%s"},
}


def merge_cfg(cfg=None):
    out = copy.deepcopy(DEFAULT_CFG)
    for k, v in (cfg or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = dict(out[k])
            out[k].update(v)
        else:
            out[k] = v
    return out


def quintic(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def _quat(R):
    return FX._quat_wxyz(R).tolist()


def set_physics(spec, physics=None):
    """the cell's physics on an MjSpec (the cell may set it itself; the positioner needs only the step loop)"""
    import mujoco
    P = copy.deepcopy(PHYSICS)
    P.update(physics or {})
    spec.memory = int(P["arena_memory_bytes"])
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    spec.option.timestep = float(P["timestep"])
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = int(P["noslip_iterations"])
    spec.option.impratio = float(P["impratio"])
    spec.option.gravity = [0.0, 0.0, -float(P["gravity"])]
    if not P.get("multiccd", False):
        spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    gd = spec.default.geom
    G = P["geom_defaults"]
    gd.condim = int(G["condim"])
    gd.friction = list(G["friction"])
    gd.solref = list(G["solref"])
    gd.solimp = list(G["solimp"])
    return P


class Handles(dict):
    """names of the positioner's elements in the spec / model"""
    __getattr__ = dict.get


def _boxes_with_mass(body, boxes, total_mass, rgba, prefix, lift_bottom_to=None):
    """add boxes (name, centre, half; axis-aligned, body frame) to a body, the total mass split by volume. With
    lift_bottom_to: a box reaching below that height is cut there (its top kept)."""
    import mujoco
    vols, geo = [], []
    for n, c, h in boxes:
        c, h = np.asarray(c, float).copy(), np.asarray(h, float).copy()
        if lift_bottom_to is not None and c[2] - h[2] < lift_bottom_to:
            top = c[2] + h[2]
            c[2], h[2] = (top + lift_bottom_to) / 2, (top - lift_bottom_to) / 2
        geo.append((n, c, h))
        vols.append(8.0 * h[0] * h[1] * h[2])
    V = sum(vols)
    names = []
    for (n, c, h), v in zip(geo, vols):
        g = body.add_geom(name=prefix + n, type=mujoco.mjtGeom.mjGEOM_BOX, size=h.tolist(), pos=c.tolist(),
                          mass=float(total_mass) * v / V, rgba=list(rgba))
        names.append(g.name)
    return names


def build_positioner(spec, fx, vehicle=None, mount_T=None, cfg=None, add_vehicle_geoms=None, physics=None):
    """add the positioner (+ the vehicle body if `vehicle` is given) to `spec`; returns Handles.
    fx: fixtures.Fixtures; vehicle: {"mass_kg", "cg_B_m" (3), "inertia_about_com_B_kgm2" (3x3)} of everything that is
    rigid with the vehicle body (the base vehicle); mount_T: 4x4 (m) of the vehicle frame in the carrier frame (the
    episode's mounting offset; None = exact); add_vehicle_geoms(spec, body): adds the vehicle's geoms (B frame) to it."""
    import mujoco
    cfg = merge_cfg(cfg)
    P = copy.deepcopy(PHYSICS)
    P.update(physics or {})
    N = cfg["names"]
    H = Handles(names=N)
    w = spec.worldbody
    o = fx.vehicle_origin()
    lc = FX.LOAD_CELL
    ch = float(lc["height_m"])
    cells = fx.load_cells_cell_xy()
    C = np.array([[x, y] for _k, x, y in cells])
    # ---- the base: 3 DOF about the cells' centroid; its boxes from the layout, cut at the cells' height
    base = w.add_body(name=N["base"], pos=[0.0, 0.0, 0.0])
    piv = [float(C[:, 0].mean()), float(C[:, 1].mean()), ch]
    base.add_joint(name=N["base"] + "_z", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], pos=piv)
    base.add_joint(name=N["base"] + "_rx", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], pos=piv)
    base.add_joint(name=N["base"] + "_ry", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 1, 0], pos=piv)
    pb = fx.positioner_boxes()
    H.base_geoms = _boxes_with_mass(base, pb["static"], FX.STRUCTURE["base_mass_kg"], FX.STRUCTURE["rgba_base"],
                                    N["base"] + "/", lift_bottom_to=ch)
    # ---- the load cells: one damped spring tendon each, from the bench top to the base
    H.cells = []
    for k, x, y in cells:
        nm = N["cell"] % k
        w.add_site(name=nm + "_w", pos=[x, y, 0.0], size=[0.01, 0, 0], rgba=[0.9, 0.6, 0.1, 1])
        base.add_site(name=nm + "_b", pos=[x, y, ch], size=[0.01, 0, 0], rgba=[0.9, 0.6, 0.1, 1])
        w.add_geom(name=nm + "_vis", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[float(lc["radius_m"]), ch / 2, 0],
                   pos=[x, y, ch / 2], contype=0, conaffinity=0, rgba=[0.9, 0.6, 0.1, 1], group=1)
        t = spec.add_tendon(name=nm)
        t.stiffness = [float(lc["stiffness_N_m"]), 0.0, 0.0]
        t.damping = [float(lc["damping_N_s_m"]), 0.0, 0.0]
        t.springlength = [ch, ch]
        t.wrap_site(nm + "_w")
        t.wrap_site(nm + "_b")
        H.cells.append(nm)
    # ---- the carrier: frame = B at q = 0 (psi 0), hinge about x through the vehicle origin; the turn is prescribed
    car = base.add_body(name=N["carrier"], pos=o.tolist())
    jt = car.add_joint(name=N["hinge"], type=mujoco.mjtJoint.mjJNT_HINGE, axis=[1, 0, 0], pos=[0, 0, 0])
    jt.armature = 0.05
    H.carrier_geoms = _boxes_with_mass(car, pb["turning"], FX.STRUCTURE["carrier_mass_kg"], FX.STRUCTURE["rgba_carrier"],
                                       N["carrier"] + "/")
    lk = spec.add_equality(name=N["lock"], type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT, name1=N["hinge"])
    lk.active = True
    lk.solref = list(P["weld_solref"])
    lk.solimp = list(P["weld_solimp"])
    lk.data = [INDEX_ANGLE[cfg["start_index"]]] + [0.0] * 10
    # ---- the rotation column (no contact; the interlock measures the robot's distance to it)
    col = fx.column()
    x0, x1 = col["x_range_m"]
    w.add_geom(name=N["column"], type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[col["radius_m"], (x1 - x0) / 2, 0],
               pos=[(x0 + x1) / 2, col["axis_point"][1], col["axis_point"][2]], quat=_quat(FX.rotx(0) @ _roty(90.0)),
               contype=0, conaffinity=0, group=5, rgba=[0.9, 0.2, 0.2, 0.0], mass=0.0)
    H.column = N["column"]
    # ---- the vehicle: rigid child of the carrier, frame B (+ the mounting offset)
    H.vehicle = None
    if vehicle is not None:
        Tm = np.eye(4) if mount_T is None else np.asarray(mount_T, float)
        veh = car.add_body(name=N["vehicle"], pos=Tm[:3, 3].tolist(), quat=_quat(Tm[:3, :3]))
        I = np.asarray(vehicle["inertia_about_com_B_kgm2"], float)
        wv, Vv = np.linalg.eigh(0.5 * (I + I.T))
        if np.linalg.det(Vv) < 0:
            Vv[:, 0] = -Vv[:, 0]
        veh.mass = float(vehicle["mass_kg"])
        veh.ipos = [float(v) for v in vehicle["cg_B_m"]]
        veh.iquat = _quat(Vv)
        veh.inertia = [float(v) for v in np.maximum(wv, 1e-9)]
        veh.explicitinertial = True
        if add_vehicle_geoms is not None:
            add_vehicle_geoms(spec, veh)
        H.vehicle = N["vehicle"]
        H.vehicle_mass = float(vehicle["mass_kg"])
        H.vehicle_cg_B = [float(v) for v in vehicle["cg_B_m"]]
    H.mount_T = (np.eye(4) if mount_T is None else np.asarray(mount_T, float)).tolist()
    H.cfg = cfg
    return H


def _roty(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


# ------------------------------------------------------------------ readings
def analytic_cells(fx, mass, cg_B, index, g, mount_T=None):
    """the static split (N) of a payload (mass kg, CG in the vehicle frame B, m) on the three cells at an index"""
    T = fx.T_cell_B(index, mount_T)
    p = T[:3, :3] @ np.asarray(cg_B, float) + T[:3, 3]
    C = np.array([[x, y] for _k, x, y in fx.load_cells_cell_xy()])
    A = np.vstack([np.ones(3), C[:, 0], C[:, 1]])
    W = float(mass) * float(g)
    return np.linalg.solve(A, np.array([W, W * p[0], W * p[1]]))


def invert_cells(F, index, fx, g):
    """what an agent does with three tared readings (the cells' stations from the drawing): (mass kg, x_B m, y_B m);
    keel_up turns the station's y into the vehicle's -y"""
    F = np.asarray(F, float)
    C = np.array([[x, y] for _k, x, y in fx.load_cells_cell_xy()])
    W = float(F.sum())
    xs, ys = float((F * C[:, 0]).sum()) / W, float((F * C[:, 1]).sum()) / W
    o = fx.vehicle_origin()
    return W / float(g), xs - o[0], (ys - o[1]) * (1.0 if index == "deck_up" else -1.0)


class LoadCellNoise(object):
    """the cells' noise of one episode: a zero offset per cell, uniform in +-bias_max_N (or given), drawn once, and a
    white N(0, sigma_N) on every sample; one numpy PCG64 stream seeded by (seed, tag)"""

    def __init__(self, seed, sigma_N=0.4, bias_max_N=0.02, bias_N=None, tag=0x10ADCE11):
        self.rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, int(tag)])
        self.sigma = float(sigma_N)
        self.bias = (np.asarray(bias_N, float) if bias_N is not None
                     else self.rng.uniform(-float(bias_max_N), float(bias_max_N), 3))

    def sample(self, clean):
        return np.asarray(clean, float) + self.bias + self.rng.normal(0.0, self.sigma, 3)


# ------------------------------------------------------------------ the run-time driver
class Positioner(object):
    """index requests, the prescribed turn, the scrap rule, the column interlock, the tare and the readings"""

    def __init__(self, model, data, H, fx, cfg=None, units=(), is_fastened=None, on_scrap=None, robot_geoms=None,
                 tare=None):
        import mujoco
        self.mj = mujoco
        self.m, self.d, self.H, self.fx = model, data, H, fx
        self.cfg = merge_cfg(cfg if cfg is not None else H.get("cfg"))
        N = self.cfg["names"]
        nid = lambda t, n: mujoco.mj_name2id(model, t, n)  # noqa: E731
        self.jid = nid(mujoco.mjtObj.mjOBJ_JOINT, N["hinge"])
        self.qadr, self.dadr = int(model.jnt_qposadr[self.jid]), int(model.jnt_dofadr[self.jid])
        self.lid = nid(mujoco.mjtObj.mjOBJ_EQUALITY, N["lock"])
        self.cids = [nid(mujoco.mjtObj.mjOBJ_TENDON, n) for n in H.cells]
        self.base_id = nid(mujoco.mjtObj.mjOBJ_BODY, N["base"])
        self.car_id = nid(mujoco.mjtObj.mjOBJ_BODY, N["carrier"])
        self.vid = nid(mujoco.mjtObj.mjOBJ_BODY, N["vehicle"])
        self.col_gid = nid(mujoco.mjtObj.mjOBJ_GEOM, N["column"])
        self.base_dofs = [int(model.jnt_dofadr[nid(mujoco.mjtObj.mjOBJ_JOINT, N["base"] + s)]) for s in ("_z", "_rx", "_ry")]
        self.base_qpos = [int(model.jnt_qposadr[nid(mujoco.mjtObj.mjOBJ_JOINT, N["base"] + s)]) for s in ("_z", "_rx", "_ry")]
        if min(self.cids + [self.jid, self.lid, self.car_id, self.col_gid]) < 0:
            raise ValueError("positioner elements missing from the model")
        self.uid = {u: nid(mujoco.mjtObj.mjOBJ_BODY, u) for u in units}
        if any(v < 0 for v in self.uid.values()):
            raise ValueError("unknown unit bodies: %s" % [u for u, v in self.uid.items() if v < 0])
        self.is_fastened = is_fastened or self._own_welds
        self.on_scrap = on_scrap
        self.robot_geoms = list(robot_geoms or [])
        self.tare = tare or {"deck_up": np.zeros(3), "keel_up": np.zeros(3)}
        self.box = fx.vehicle_box_B(float(self.cfg["scrap_margin_m"])) if self.vid >= 0 else None
        self.state = None            # the index position while at rest
        self.target = None           # the target while moving
        self.traj = None
        self.turns_used = 0
        self.scrapped = []           # (time s, unit, why)
        self.requests = []           # (time s, target, answer)
        self.col_tmp = np.zeros(6)

    # ---- the cell's own welds (standalone use): a unit is fastened if its equality "weld_<unit>" is active
    def _own_welds(self, u):
        e = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_EQUALITY, "weld_" + u)
        return e >= 0 and bool(self.d.eq_active[e])

    # ---- pose
    def q(self):
        return float(self.d.qpos[self.qadr])

    def index(self):
        """the index position (the target while moving)"""
        return self.target if self.traj is not None else self.state

    def moving(self):
        return self.traj is not None

    def turns_left(self):
        return int(self.cfg["max_turns"]) - self.turns_used

    def place(self, index=None):
        """the hinge at an index position, the base at rest (the cell resets the rest of qpos itself)"""
        idx = index or self.cfg["start_index"]
        q = INDEX_ANGLE[idx]
        self.d.qpos[self.qadr] = q
        self.d.qvel[self.dadr] = 0.0
        for a in self.base_qpos:
            self.d.qpos[a] = 0.0
        for a in self.base_dofs:
            self.d.qvel[a] = 0.0
        self.m.eq_data[self.lid, 0] = q
        self.d.eq_active[self.lid] = 1
        self.state, self.target, self.traj = idx, None, None
        self.mj.mj_forward(self.m, self.d)

    # ---- requests
    def robot_in_column(self):
        """(inside?, {geom name: distance m}) of the robot's geoms to the rotation column (exact convex distance; a
        negative distance = inside); `column_margin_m` widens the column"""
        mg = float(self.cfg["column_margin_m"])
        near = {}
        inside = False
        for g in self.robot_geoms:
            dist = float(self.mj.mj_geomDistance(self.m, self.d, int(g), self.col_gid, 0.05 + mg, self.col_tmp))
            if dist < 0.05 + mg:
                near[self.m.geom(int(g)).name] = dist
            if dist < mg:
                inside = True
        return inside, near

    def request_index(self, target):
        """the cell API's fixture_index: {"ok": True} or {"refused": why}; why in moving / no_turns_left /
        robot_in_column (refused requests count nothing; a request for the current index is a no-op)"""
        t = float(self.d.time)
        if target not in INDEX_ANGLE:
            raise ValueError("unknown index position %r" % (target,))
        if self.traj is not None:
            ans = {"refused": "moving"}
        elif self.state == target:
            ans = {"ok": True}
        elif self.turns_left() <= 0:
            ans = {"refused": "no_turns_left"}
        elif self.robot_geoms and self.robot_in_column()[0]:
            ans = {"refused": "robot_in_column"}
        else:
            for u in self.unlocked_on_vehicle():
                self.scrap_unit(u, "unfastened on the vehicle at the start of a turn")
            self.mj.mj_forward(self.m, self.d)
            q0 = self.q()
            key = "%s->%s" % (self.state, target)
            sgn = int(self.cfg["turn_sign"].get(key, 1))
            self.traj = (t, q0, q0 + sgn * math.pi)
            self.target, self.state = target, None
            self.turns_used += 1
            ans = {"ok": True}
        self.requests.append((t, target, ans))
        return ans

    def before_step(self):
        """call before every physics step: the prescribed hinge angle; ends the turn after the settle time"""
        if self.traj is None:
            return
        t0, q0, q1 = self.traj
        T = float(self.cfg["turn_s"])
        tt = float(self.d.time) - t0
        self.m.eq_data[self.lid, 0] = q0 + (q1 - q0) * quintic(tt / T)
        if tt >= T + float(self.cfg["settle_s"]):
            self.m.eq_data[self.lid, 0] = q1
            self.traj = None
            self.state, self.target = self.target, None

    # ---- the scrap rule
    def _turning_bodies(self):
        ids = {self.vid, self.car_id}
        for u, b in self.uid.items():
            if self.is_fastened(u):
                ids.add(b)
        return ids

    def unlocked_on_vehicle(self):
        """loose units that touch the vehicle, the carrier or a fastened unit, or whose centre of mass lies in the
        vehicle box (+ margin, the vehicle's body frame)"""
        if self.vid < 0:
            return []
        scr = {x[1] for x in self.scrapped}
        on = self._turning_bodies()
        Rv = self.d.xmat[self.vid].reshape(3, 3)
        pv = self.d.xpos[self.vid]
        lo, hi = self.box
        out = []
        touching = set()
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            b1, b2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
            if b1 in on:
                touching.add(b2)
            if b2 in on:
                touching.add(b1)
        for u, b in self.uid.items():
            if u in scr or self.is_fastened(u):
                continue
            hit = b in touching
            if not hit:
                pB = Rv.T @ (self.d.xipos[b] - pv)
                hit = bool(np.all(pB >= lo) and np.all(pB <= hi))
            if hit:
                out.append(u)
        return out

    def scrap_unit(self, u, why):
        """the cell's scrap first (on_scrap: the coupling releases its capture), then the unit off the bench, at rest"""
        if self.on_scrap is not None:
            self.on_scrap(u, why)
        e = self.mj.mj_name2id(self.m, self.mj.mjtObj.mjOBJ_EQUALITY, "weld_" + u)
        if e >= 0:
            self.d.eq_active[e] = 0
        b = self.uid[u]
        ja = int(self.m.body_jntadr[b])
        qa, da = int(self.m.jnt_qposadr[ja]), int(self.m.jnt_dofadr[ja])
        sa = self.cfg["scrap_area"]
        k = len(self.scrapped)
        self.d.qpos[qa:qa + 3] = [float(sa["xy"][0]), float(sa["xy"][1]) + k * float(sa["pitch_m"]), float(sa["z_m"])]
        self.d.qpos[qa + 3:qa + 7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qvel[da:da + 6] = 0.0
        self.scrapped.append((float(self.d.time), u, why))

    # ---- readings
    def raw(self):
        """the cells' spring forces (N, compression +), station frame order [A, B, C]"""
        m, d = self.m, self.d
        return np.array([m.tendon_stiffness[i] * (m.tendon_lengthspring[i, 0] - d.ten_length[i]) for i in self.cids])

    def reading(self):
        """raw - the tare of the index position (the target's while moving)"""
        return self.raw() - np.asarray(self.tare[self.index() or self.cfg["start_index"]], float)

    def fixture_state(self, noise=None):
        """the cell API's get_state()["fixture"]: {index, moving, load_N[3]} (noise: a LoadCellNoise)"""
        r = self.reading()
        if noise is not None:
            r = noise.sample(r)
        return {"index": self.index(), "moving": self.moving(), "load_N": [float(v) for v in r]}

    def report(self):
        return {"turns_used": self.turns_used, "turns_left": self.turns_left(), "scrapped": list(self.scrapped),
                "requests": [(t, tg, a) for t, tg, a in self.requests]}


# ------------------------------------------------------------------ standalone scenes and the tare
def standalone_spec(fx, vehicle=None, mount_T=None, cfg=None, physics=None, add_vehicle_geoms=None, bench=True):
    """an MjSpec with the cell's physics, the bench and a floor (if bench), and the positioner (+ vehicle):
    (spec, Handles). Used for the tare and by the development tests; the product cell builds its own scene."""
    import mujoco
    spec = mujoco.MjSpec()
    spec.modelname = "rov_positioner"
    set_physics(spec, physics)
    if bench:
        tb = FX.TABLE
        x0, x1 = tb["x_m"]
        y0, y1 = tb["y_m"]
        th = float(tb["thickness_m"])
        spec.worldbody.add_geom(name="bench_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[(x1 - x0) / 2, (y1 - y0) / 2, th / 2],
                                pos=[(x0 + x1) / 2, (y0 + y1) / 2, -th / 2], rgba=[0.80, 0.74, 0.62, 1.0])
        spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5.0, 5.0, 0.1], pos=[0.0, 0.0, -0.994],
                                rgba=[0.3, 0.3, 0.3, 1.0])
    H = build_positioner(spec, fx, vehicle=vehicle, mount_T=mount_T, cfg=cfg, add_vehicle_geoms=add_vehicle_geoms, physics=physics)
    return spec, H


_TARE_CACHE = {}


def compute_tare(fx, cfg=None, physics=None, settle_s=1.5, avg_s=0.5):
    """the empty positioner (no vehicle) at each index position: the mean raw readings {index: (3,)}; cached per
    (layout, config, physics)"""
    import mujoco
    key = hashlib.sha256(json.dumps([fx.L.get("_sha256"), merge_cfg(cfg), physics or {}], sort_keys=True, default=str).encode()).hexdigest()
    if key in _TARE_CACHE:
        return {k: v.copy() for k, v in _TARE_CACHE[key].items()}
    spec, H = standalone_spec(fx, vehicle=None, cfg=cfg, physics=physics, bench=False)
    m = spec.compile()
    d = mujoco.MjData(m)
    P = Positioner(m, d, H, fx, cfg)
    out = {}
    for idx in ("deck_up", "keel_up"):
        mujoco.mj_resetData(m, d)
        P.place(idx)
        n1, n2 = int(round(settle_s / m.opt.timestep)), int(round(avg_s / m.opt.timestep))
        for _ in range(n1):
            P.before_step()
            mujoco.mj_step(m, d)
        acc = np.zeros(3)
        for _ in range(n2):
            P.before_step()
            mujoco.mj_step(m, d)
            acc += P.raw()
        out[idx] = acc / n2
    _TARE_CACHE[key] = {k: v.copy() for k, v in out.items()}
    return out
