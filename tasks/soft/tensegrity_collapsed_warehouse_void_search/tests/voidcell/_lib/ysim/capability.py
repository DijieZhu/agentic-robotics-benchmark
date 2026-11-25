"""Capability measurements of the validated model on blockout features (ysim 0.2.0; extended for 0.3.0;
completed and run on the final model ysim 0.3.1).

STATUS (ysim 0.3.1, CAPABILITY_REV below): complete. The capability sheet of ysim 0.3.1
(status FINAL; not shipped with this copy) was measured with exactly these protocols on the 0.3.1 robot.
The constants STRAIGHT, CLASS_A / CLASS_B, FACE_NEXT_ROLL / FACE_NEXT_TURN are 0.3.1 measurements. This module never
changes the robot: every protocol builds the 0.3.1 model with its defaults and only places it, sets world features /
floor friction, and runs published primitives or set_targets shapes.

These are measurement protocols of the simulation base (what the robot model does on a feature), not ways to solve
a task. Every trial starts from a seeded state (start distance, lateral offset, heading, start face or history,
floor friction where stated) and runs the published primitives. Where a protocol must choose primitives, one of two
measuring devices does it; both are named in every record that uses them:

  privileged_steer_plan   PRIVILEGED MEASURING DEVICE: before every primitive it reads the TRUE pose and face and
                          picks the published primitive whose mean displacement in THIS model's PRIM_TABLE brings the
                          robot closest to a line. It stands in for 'aimed with feedback after every primitive' (the
                          authors replan after every primitive). With-feedback numbers size the sites; they are not
                          difficulty evidence and not a reference solution.
  dead_reckon_plan        OPEN-LOOP SEQUENCE GENERATOR: the same choice rule applied once, in advance, to poses
                          PREDICTED from the PRIM_TABLE means and the measured face transitions, starting from a
                          believed start pose; the resulting fixed sequence then runs with no pose feedback. Its
                          numbers describe the spread of the world for blind legs; not a strategy.

    place(S, xy, roll_dir_deg, z_support)       rigid placement of the robot (before or after stand)
    place_on_face(S, face, xy, roll_dir)        07 start rule: rest shape turned onto a face, dropped 5 cm
    set_floor_friction(S, mu)                   floor friction against the robot (world parameter)
    extent / top_z / radius_xy                  envelope of the robot's collision geometry
    tendon_segments(S)                          straight paths of the 15 tendons (for world.segment_hits)
    Tracker                                     tick-hook sampler (envelope, tendon / rod cuts, world contacts)
    drive(S, plan, ...)                         run primitives until a success test / max count / stuck
    set_targets_mm / primitive_shapes_mm        one target shape in mm through the same PID (07 set_targets)
    trial_*                                     one seeded trial of each protocol -> JSON-ready dict
Protocols and seeds: the trial_* functions below; results: the capability sheet of ysim 0.3.1 (not shipped here).
"""
import math

import mujoco
import numpy as np

from . import gaits
from . import metrics
from . import model as mdl
from . import params as prm
from . import world as wd
from .health import Monitor
from .sim import Simulator

CAPABILITY_REV = "0.3.1-sheet1"   # protocol revision of this module; the robot model is ysim 0.3.1
STUCK_WINDOW = 3          # the debrief's stuck rule (voidscore/acceptance.yaml, block stuck): 3 primitives in a row with COM moving < 0.03 m
STUCK_DIST = 0.03
# ysim 0.3.1: the forward roll with the smallest heading change from the faces the robot lives on after its first
# primitive (035 / 125 / 134): +0.7 +- 1.5 deg (the 0.3.1 validation; re-measured in the
# capability measurements' primitive table). 0.3.0 used roll_100_120 (now -3.3 deg), 0.2.0 roll_120_140 (now -5.9 deg).
STRAIGHT = "roll_100_140"
STRAIGHT_V030 = "roll_100_120"     # the 0.3.0 straightest roll (the 0.3.0 protocols)
STRAIGHT_V020 = "roll_120_140"     # the 0.2.0 protocols' roll, kept for side-by-side re-runs
REV_STRAIGHT = "roll_140_100"      # reverse roll of the 0.2.0 face pre-sequences (trial_face only); 0.3.1 reverse
                                   # table: the reverse-roll table of the capability measurements (not shipped)
# the two face classes of the authors' six rollable faces: two even nodes + one odd (the faces the authors' next_nodes
# names; reached only at a start / drop or by reverse rolls) and one even + two odd (where every primitive lands):
# measured on 0.3.0 and on 0.3.1; unchanged in 0.3.1
CLASS_A = ((0, 2, 5), (0, 3, 4), (1, 2, 4))
CLASS_B = ((0, 3, 5), (1, 2, 5), (1, 3, 4))
# measured face transitions (deterministic in every execution; measured on 0.3.0, re-measured unchanged
# on 0.3.1 in the validation and in every PRIM_TABLE execution of the 0.3.1 capability measurements)
FACE_NEXT_ROLL = {(0, 2, 5): (1, 2, 5), (1, 2, 4): (1, 3, 4), (0, 3, 4): (0, 3, 5),
                  (1, 2, 5): (1, 3, 4), (1, 3, 4): (0, 3, 5), (0, 3, 5): (1, 2, 5)}
FACE_NEXT_TURN = {(0, 2, 5): (0, 3, 5), (1, 2, 4): (1, 2, 5), (0, 3, 4): (1, 3, 4),
                  (0, 3, 5): (0, 3, 5), (1, 2, 5): (1, 2, 5), (1, 3, 4): (1, 3, 4)}
# set_targets_mm uses the normalisation of the 07 cell's set_targets (task 07 executor.py): x = (L - min_length) /
# default_range with the authors' min_length 100 mm and RANGE 100 mm, left = right = RANGE (so the band and the PID
# gains are those of the rest phase). The 0.3.0 protocols had used 140 mm.
SET_TARGETS_MIN_MM = prm.value("min_length") * 1000.0
SET_TARGETS_RANGE_MM = prm.value("default_range") * 1000.0


def _rng(seed):
    return np.random.Generator(np.random.PCG64(seed))


def face_key(face):
    return "".join(str(int(i)) for i in face)


def next_face(face, name):
    """Measured next face after the published primitive `name` from a rollable face (FACE_NEXT_*)."""
    f = tuple(face)
    return (FACE_NEXT_ROLL if gaits.get(name).kind == "roll" else FACE_NEXT_TURN)[f]


# ----------------------------------------------------------------------------- placement and envelope
def roll_dir_deg(nodes):
    """World direction of the primitive table's +y (the rolling direction): principal axis + 90 deg."""
    _, th = metrics.se2(nodes)
    return math.degrees(th) + 90.0


def set_floor_friction(S, mu):
    """Friction of the floor plane against the robot. The floor gets priority 1 so that its own friction is used (as
    every world record does); without it MuJoCo takes the larger of the two (the robot's 0.8). A world parameter."""
    f = S.ids.floor
    if f is None or f < 0:
        raise ValueError("model has no floor")
    S.m.geom_friction[f, 0] = float(mu)
    S.m.geom_priority[f] = 1


def place(S, xy, roll_dir, z_support=0.0, clearance=0.002):
    """Rigidly move the three struts: planar node centroid -> xy, rolling direction -> roll_dir (deg, world),
    lowest collision point -> z_support + clearance; velocities zeroed. Cable rest lengths are untouched (a rigid
    move keeps every tendon length)."""
    m, d = S.m, S.d
    mujoco.mj_forward(m, d)
    n = S.nodes()
    c, _ = metrics.se2(n)
    a = math.radians(roll_dir - roll_dir_deg(n))
    ca, sa = math.cos(a), math.sin(a)
    Rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.0]])
    qz = np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)])
    for k in range(3):
        adr = S.ids.jnt_qpos[k]
        p = d.qpos[adr:adr + 3].copy()
        p[:2] -= c
        p = Rz @ p
        p[:2] += np.asarray(xy, float)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qz, d.qpos[adr + 3:adr + 7].copy())
        d.qpos[adr:adr + 3] = p
        d.qpos[adr + 3:adr + 7] = q
        dof = S.ids.jnt_dof[k]
        d.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(m, d)
    zmin = min(S._geom_zmin(g) for g in S.ids.col_geoms)
    for k in range(3):
        d.qpos[S.ids.jnt_qpos[k] + 2] += z_support + clearance - zmin
    d.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(m, d)


def _rot_between(u, v):
    """Rotation matrix turning unit vector u onto unit vector v."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    ax = np.cross(u, v)
    s, c = float(np.linalg.norm(ax)), float(u @ v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.array([1.0, 0, 0]) if abs(u[0]) < 0.9 else np.array([0, 1.0, 0])
        k = np.cross(u, p)
        k /= np.linalg.norm(k)
        return 2.0 * np.outer(k, k) - np.eye(3)
    k = ax / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def rotate_rigid(S, R, centre):
    """Rotate the three struts rigidly by R about the point centre (velocities zeroed)."""
    m, d = S.m, S.d
    qR = np.zeros(4)
    mujoco.mju_mat2Quat(qR, np.asarray(R, float).reshape(9))
    c = np.asarray(centre, float)
    for k in range(3):
        adr = S.ids.jnt_qpos[k]
        d.qpos[adr:adr + 3] = R @ (d.qpos[adr:adr + 3] - c) + c
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, qR, d.qpos[adr + 3:adr + 7].copy())
        d.qpos[adr + 3:adr + 7] = q / np.linalg.norm(q)
        dof = S.ids.jnt_dof[k]
        d.qvel[dof:dof + 6] = 0.0
    d.qacc_warmstart[:] = 0.0
    mujoco.mj_forward(m, d)


def place_on_face(S, face, xy=(0.0, 0.0), roll_dir=90.0, drop=0.05, z_support=0.0, settle_t=6.0, vtol=5e-4):
    """The 07 start rule ('released in the rest shape 0.05 m above the drop point on one of the six
    rollable faces and left to settle'): the robot's current shape (normally the rest shape after stand()) is turned
    rigidly so that the plane of the three end caps `face` is horizontal and below the robot, moved to xy with its
    rolling direction roll_dir, released with its lowest collision point `drop` above z_support, then left to settle
    with the motors braked. Returns the bottom triple after settling (normally `face`)."""
    face = tuple(sorted(int(i) for i in face))
    if face not in gaits.ROLLABLE_FACES:
        raise gaits.NotRollableFace("%s is not a rollable face %s" % (face, gaits.ROLLABLE_FACES))
    mujoco.mj_forward(S.m, S.d)
    n = S.nodes()
    a, b, c = (n[i] for i in face)
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm)
    cen = n.mean(axis=0)
    if (a - cen) @ nrm < 0:          # outward normal of the face
        nrm = -nrm
    rotate_rigid(S, _rot_between(nrm, (0.0, 0.0, -1.0)), cen)
    place(S, xy, roll_dir, z_support, clearance=drop)
    S.settle(settle_t, vtol=vtol)
    return S.bottom()[0]


def _geom_points(S):
    """Per robot collision geom: ('s', centre, r) | ('c', end_a, end_b, r) | ('m', vertices)."""
    m, d = S.m, S.d
    out = []
    for g in S.ids.col_geoms:
        t = m.geom_type[g]
        p = d.geom_xpos[g]
        R = d.geom_xmat[g].reshape(3, 3)
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            out.append(("s", p.copy(), float(m.geom_size[g][0])))
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            r, h = m.geom_size[g][0], m.geom_size[g][1]
            ax = R[:, 2]
            out.append(("c", p + h * ax, p - h * ax, float(r), ax.copy()))
        elif t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v0, nv = m.mesh_vertadr[mid], m.mesh_vertnum[mid]
            out.append(("m", m.mesh_vert[v0:v0 + nv] @ R.T + p))
    return out


def extent(S, direction, pts=None):
    """(min, max) of the robot's collision geometry projected on a unit direction (exact for spheres, cylinders
    and the convex hulls)."""
    u = np.asarray(direction, float)
    u = u / np.linalg.norm(u)
    lo, hi = 1e9, -1e9
    for g in (pts or _geom_points(S)):
        if g[0] == "s":
            s = g[1] @ u
            lo, hi = min(lo, s - g[2]), max(hi, s + g[2])
        elif g[0] == "c":
            ax = g[4]
            rr = g[3] * math.sqrt(max(0.0, 1.0 - float(ax @ u) ** 2))
            for e in (g[1], g[2]):
                s = e @ u
                lo, hi = min(lo, s - rr), max(hi, s + rr)
        else:
            s = g[1] @ u
            lo, hi = min(lo, float(s.min())), max(hi, float(s.max()))
    return lo, hi


def top_z(S, pts=None):
    return extent(S, (0, 0, 1), pts)[1]


def radius_xy(S, c, pts=None):
    """Largest horizontal distance of the collision geometry from the point c (x, y)."""
    c = np.asarray(c, float)[:2]
    r = 0.0
    for g in (pts or _geom_points(S)):
        if g[0] == "s":
            r = max(r, float(np.linalg.norm(g[1][:2] - c)) + g[2])
        elif g[0] == "c":
            for e in (g[1], g[2]):
                r = max(r, float(np.linalg.norm(e[:2] - c)) + g[3])
        else:
            r = max(r, float(np.linalg.norm(g[1][:, :2] - c, axis=1).max()))
    return r


def tendon_segments(S):
    """(P, Q, tendon index) of every straight piece of the 15 spatial tendons (current state)."""
    d, m = S.d, S.m
    P, Q, T = [], [], []
    for t in range(m.ntendon):
        a, n = int(d.ten_wrapadr[t]), int(d.ten_wrapnum[t])
        pts = np.asarray(d.wrap_xpos).reshape(-1, 3)[a:a + n]       # (nwrap x 6) storage = 2 points per row
        for i in range(n - 1):
            P.append(pts[i].copy())
            Q.append(pts[i + 1].copy())
            T.append(t)
    return np.array(P), np.array(Q), T


class Tracker:
    """Tick hook (every `every` control ticks): envelope in the frame of the last primitive start, tendon and rod
    cuts through world records, contacts with world geoms, health monitor."""

    def __init__(self, S, world=None, every=4):
        self.S, self.world, self.every = S, world, every
        self.k = 0
        self.mon = Monitor(S)
        self.world_geoms = set()
        if world is not None:
            for nm in world.geom_names():
                gid = mujoco.mj_name2id(S.m, mujoco.mjtObj.mjOBJ_GEOM, nm)
                if gid >= 0:
                    self.world_geoms.add(gid)
        self.reset_frame()
        self.tendon_hit_samples = 0
        self.tendon_hit_features = {}
        self.tendon_pen_max = 0.0
        self.rod_pen_max = 0.0
        self.samples = 0
        self.world_contact_geoms = {}
        S.tick_hooks.append(self)

    def reset_frame(self):
        n = self.S.nodes()
        c, th = metrics.se2(n)
        self.c0, self.th0 = c, th
        self.ax = np.array([math.cos(th), math.sin(th), 0.0])      # start principal axis (table x)
        self.fw = np.array([-math.sin(th), math.cos(th), 0.0])     # rolling direction (table y)
        self.w_inst = 0.0
        self.w_caps = 0.0          # end-cap spheres only (W_ROLL of the 07 envelope)
        self.l_inst = 0.0          # span along the current rolling direction (D_ROLL of the 07 envelope)
        self.h_max = 0.0
        self.sw_ax = [1e9, -1e9]
        self.sw_fw = [1e9, -1e9]
        self.r_max = 0.0
        self.r_caps = 0.0          # end caps only (R_TURN of the 07 envelope)

    def __call__(self, S):
        self.k += 1
        if self.k % self.every:
            return
        pts = _geom_points(S)
        n = S.nodes()
        _, th = metrics.se2(n)
        ax_now = np.array([math.cos(th), math.sin(th), 0.0])
        lo, hi = extent(S, ax_now, pts)
        self.w_inst = max(self.w_inst, hi - lo)
        caps = [g for g in pts if g[0] == "s"]
        lo, hi = extent(S, ax_now, caps)
        self.w_caps = max(self.w_caps, hi - lo)
        lo, hi = extent(S, np.array([-ax_now[1], ax_now[0], 0.0]), pts)
        self.l_inst = max(self.l_inst, hi - lo)
        self.h_max = max(self.h_max, top_z(S, pts))
        lo, hi = extent(S, self.ax, pts)
        self.sw_ax = [min(self.sw_ax[0], lo - self.c0 @ self.ax[:2]), max(self.sw_ax[1], hi - self.c0 @ self.ax[:2])]
        lo, hi = extent(S, self.fw, pts)
        self.sw_fw = [min(self.sw_fw[0], lo - self.c0 @ self.fw[:2]), max(self.sw_fw[1], hi - self.c0 @ self.fw[:2])]
        self.r_max = max(self.r_max, radius_xy(S, self.c0, pts))
        self.r_caps = max(self.r_caps, radius_xy(S, self.c0, caps))
        self.samples += 1
        if self.world is not None:
            P, Q, T = tendon_segments(S)
            hits = self.world.segment_hits(P, Q, margin=0.001)
            if hits:
                self.tendon_hit_samples += 1
                for i, nm in hits:
                    self.tendon_hit_features[nm] = self.tendon_hit_features.get(nm, 0) + 1
                idx = sorted({i for i, _ in hits})
                self.tendon_pen_max = max(self.tendon_pen_max, self.world.segment_penetration(P[idx], Q[idx]))
            n6 = S.nodes()                       # the rods (no collision body): node-to-node axis segments
            self.rod_pen_max = max(self.rod_pen_max, self.world.segment_penetration(n6[0::2], n6[1::2]))
            d = S.d
            for i in range(d.ncon):
                c = d.contact[i]
                for g in (c.geom1, c.geom2):
                    if g in self.world_geoms:
                        nm = mujoco.mj_id2name(S.m, mujoco.mjtObj.mjOBJ_GEOM, g)
                        self.world_contact_geoms[nm] = self.world_contact_geoms.get(nm, 0) + 1
        if self.k % (5 * self.every) == 0:
            self.mon.update()

    def envelope(self):
        return {"w_inst_max_m": self.w_inst, "w_caps_max_m": self.w_caps, "l_inst_max_m": self.l_inst, "h_max_m": self.h_max, "swept_width_m": self.sw_ax[1] - self.sw_ax[0],
                "swept_length_m": self.sw_fw[1] - self.sw_fw[0], "swept_ax_m": list(self.sw_ax),
                "swept_fw_m": list(self.sw_fw), "footprint_radius_m": self.r_max, "footprint_radius_caps_m": self.r_caps}

    def n_contacts(self):
        return sum(self.world_contact_geoms.values())

    def detach(self):
        if self in self.S.tick_hooks:
            self.S.tick_hooks.remove(self)


def pose(S):
    """(x, y, principal-axis angle in deg) of the authors' SE(2) (metrics.se2)."""
    c, th = metrics.se2(S.nodes())
    return [float(c[0]), float(c[1]), math.degrees(th)]


def drive(S, plan, success=None, max_prims=10, tracker=None, reverse=False, stop_on_stuck=True):
    """Run primitives (plan: name, list of names or callable(i) -> name or None) until success(S) is true after a
    primitive, max_prims, the plan ends, or stuck (STUCK_WINDOW primitives with the COM moving < STUCK_DIST).
    Returns a summary dict; every row carries the primitive's SE(2) result, the pose after it and whether the robot
    touched a world geom during it."""
    rows = []
    hist = [S.com()[:2].copy()]
    ok, stuck, not_rollable = False, False, None
    for i in range(max_prims):
        if callable(plan):
            name = plan(i)
        elif isinstance(plan, (list, tuple)):
            name = plan[i] if i < len(plan) else None
        else:
            name = plan
        if name is None:
            break
        if tracker is not None:
            tracker.reset_frame()
            nc0 = tracker.n_contacts()
        try:
            r = S.run_primitive(name, reverse=reverse)
        except gaits.NotRollableFace as e:      # ysim 0.3.0: no primitive is defined from this contact set
            not_rollable = {"after_prims": len(rows), "bottom": list(S.bottom()[0]), "error": str(e)}
            break
        pre = r.get("se2_before_rest_phase")
        row = {"name": name, "reverse": reverse, "dist": r["dist"], "dx": r["dx"], "dy": r["dy"],
               "dtheta_deg": r["dtheta_deg"], "t_total": r["t_total"], "t_exec": r["t_exec"],
               "t_rest_phase": r["t_rest_phase"], "tipped": r["bottom_end"] != r["bottom_start"],
               "timeouts": r["timeouts"], "peak_T": r["peak_cable_tension_N"], "energy_J": r["energy_J"],
               "com": S.com().tolist(), "pose": pose(S),
               "bottom_start": list(r["bottom_start"]), "bottom_end": list(r["bottom_end"]),
               "rollable_end": bool(r["rollable_end"]), "end_triangle": bool(r["end_triangle"]),
               "rest_reached": bool(r["rest_reached"]), "top_z_end": float(r["top_z_end"]),
               "short_end_mm": [round(x, 2) for x in r["short_sensors_end_mm"]],
               "pre_rest": None if pre is None else {k: pre[k] for k in ("dx", "dy", "dtheta_deg")}}
        if tracker is not None:
            row["envelope"] = tracker.envelope()
            row["world_contact"] = tracker.n_contacts() > nc0
        rows.append(row)
        hist.append(S.com()[:2].copy())
        if success is not None and success(S):
            ok = True
            break
        if len(hist) > STUCK_WINDOW and np.linalg.norm(hist[-1] - hist[-1 - STUCK_WINDOW]) < STUCK_DIST:
            stuck = True
            if stop_on_stuck:
                break
    return {"success": ok, "stuck": stuck, "not_rollable": not_rollable, "n_prims": len(rows), "prims": rows,
            "t_sim": float(sum(r["t_total"] for r in rows))}


def _health(tr):
    rep = tr.mon.report()
    return {"nan": rep["nan"], "warnings": rep["warnings"], "max_penetration_mm": rep["max_penetration_mm"],
            "worst_pair": rep["worst_pair"], "max_node_speed_m_s": rep["max_node_speed_m_s"],
            "tendon_hit_samples": tr.tendon_hit_samples, "samples": tr.samples, "tendon_pen_max_mm": 1000 * tr.tendon_pen_max, "rod_pen_max_mm": 1000 * tr.rod_pen_max,
            "tendon_hit_features": tr.tendon_hit_features, "world_contacts": tr.world_contact_geoms}


def new_sim(world=None, **cfg):
    cfg.setdefault("visual", False)
    c = world.model_config(**cfg) if world is not None else mdl.ModelConfig(**cfg)
    return Simulator(c)


def start(S, xy, roll_dir, z_support=0.0, pre_rolls=0, pre_name=STRAIGHT):
    """Seeded start: place, stand, optional pre-rolls (to vary the ground face), re-place exactly."""
    place(S, xy, roll_dir, z_support)
    S.stand()
    for _ in range(pre_rolls):
        S.run_primitive(pre_name)
    place(S, xy, roll_dir, z_support)
    S.settle(1.0, vtol=5e-4)


# ----------------------------------------------------------------------------- the two measuring devices
def _table_row(table, face, name):
    """Mean displacement {dx, dy, dtheta_deg} of `name` from `face` in a table: {"by_face": {"125": {name: row}},
    "all": {name: row}} (PRIM_TABLE layout) or a flat {name: row} (0.2.0 layout)."""
    if "by_face" in table:
        row = table["by_face"].get(face_key(face), {}).get(name) if face is not None else None
        return row if row is not None else table.get("all", {}).get(name)
    return table.get(name)


def _best(c, th, face, table, names, p0, goal_deg, sx, sh):
    g = math.radians(goal_deg)
    nrm = np.array([-math.sin(g), math.cos(g)])
    R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    best = None
    for nm in names:
        t = _table_row(table, face, nm)
        if t is None:
            continue
        cp = c + R @ np.array([t["dx"], t["dy"]])
        phi = math.degrees(th) + t["dtheta_deg"] + 90.0
        eh = (phi - goal_deg + 180.0) % 360.0 - 180.0
        cost = (float((cp - p0) @ nrm) / sx) ** 2 + (eh / sh) ** 2
        if best is None or cost < best[0]:
            best = (cost, nm, cp, th + math.radians(t["dtheta_deg"]))
    return best


def privileged_steer_plan(S, table, x_line=0.0, goal_deg=90.0, names=None, sx=0.03, sh=10.0, line_p0=None):
    """PRIVILEGED MEASURING DEVICE (true pose and face; not a task solution, not difficulty evidence): before every
    primitive pick the published primitive whose mean displacement in THIS model's table (PRIM_TABLE by start face,
    or a flat {name: dx, dy, dtheta_deg} table) brings the robot closest to the line through line_p0 (default
    (x_line, 0)) heading goal_deg: cost (e_lateral / sx)^2 + (e_heading / sh)^2 on the predicted pose."""
    names = names or list(gaits.NAMES)
    p0 = np.asarray(line_p0 if line_p0 is not None else (x_line, 0.0), float)

    def plan(i):
        c, th = metrics.se2(S.nodes())
        b, ok = S.bottom()
        return _best(c, th, b if ok else None, table, names, p0, goal_deg, sx, sh)[1]
    return plan


steer_plan = privileged_steer_plan       # 0.2.0 name


def dead_reckon_plan(table, face0, c0, th0, legs, names=None, sx=0.03, sh=10.0, max_prims=30):
    """OPEN-LOOP SEQUENCE GENERATOR (no pose feedback): the device's choice rule applied in advance to poses
    predicted with the table means and the measured face transitions (FACE_NEXT_*), from a believed start pose
    (node centroid c0, principal-axis angle th0 [rad]) and face face0. legs: [{"p0": (x, y), "heading_deg": h,
    "length": L}, ...]; leg k is followed until the predicted progress (cp - p0) . u(h) >= L, then the next leg; the
    plan ends after the last leg or max_prims. Returns {"names": [...], "pred": [[x, y, th_deg, face], ...]}."""
    names = names or list(gaits.NAMES)
    c = np.asarray(c0, float)[:2].copy()
    th = float(th0)
    face = tuple(face0)
    out, pred = [], []
    k = 0
    while len(out) < max_prims and k < len(legs):
        lg = legs[k]
        h = math.radians(lg["heading_deg"])
        u = np.array([math.cos(h), math.sin(h)])
        if float((c - np.asarray(lg["p0"], float)) @ u) >= lg["length"]:
            k += 1
            continue
        b = _best(c, th, face, table, names, np.asarray(lg["p0"], float), lg["heading_deg"], sx, sh)
        _, nm, c, th = b
        face = next_face(face, nm)
        out.append(nm)
        pred.append([float(c[0]), float(c[1]), math.degrees(th), face_key(face)])
    return {"names": out, "pred": pred}


# ----------------------------------------------------------------------------- set_targets (the 07 station's call)
def set_targets_mm(S, lengths_mm, timeout=10.0):
    """One target shape: six short-sensor target lengths in mm (motor order of the authors' code, 100-240 mm),
    executed by the same PID, band and slack rule as a primitive's own states in the default mode, with no face
    check, no symmetry mapping and no rest phase, for at most `timeout` s (the 07 station's set_targets). Normalised
    as the 07 cell does: (L - SET_TARGETS_MIN_MM) / SET_TARGETS_RANGE_MM. Returns (reached_in_band, duration_s)."""
    L = np.asarray(lengths_mm, float)
    if L.shape != (6,) or np.any(L < 100.0 - 1e-9) or np.any(L > 240.0 + 1e-9):
        raise ValueError("out_of_range")
    R = SET_TARGETS_RANGE_MM
    x = ((L - SET_TARGETS_MIN_MM) / R).tolist()
    t_hi, t_lo, _, slack = S.mode_bands()
    t0 = S.time
    S.start_states([x], left=R, right=R, tol=t_hi, low_tol=t_lo, timeout=timeout, raw=True, slack_rule=slack)
    S.run_until_done(timeout + 1.0)
    log = S.exec["log"]
    return bool(log and not log[-1]["timeout"]), S.time - t0


def primitive_shapes_mm(S, name, reverse=False, order="forward"):
    """The target shapes in mm of a published roll / turn for the robot's current face (its own states including
    the two [1]*6 at its ranges): the symmetry-mapped states of the authors' code, optionally re-indexed with the
    authors' reverse gait (reverse=True, rolls only) and optionally played in reversed order (order="reversed")."""
    p = gaits.get(name)
    b = metrics.bottom3(S.nodes())
    if gaits.SYMMETRY.get(b) is None:
        raise gaits.NotRollableFace("bottom triple %s is not a rollable face %s" % (b, gaits.ROLLABLE_FACES))
    mapped = [gaits.map_state(s, b) for s in p.states()]
    if reverse:
        if p.kind != "roll":
            raise ValueError("the authors' code reverses rolls only")
        mapped = [gaits.reverse_state(s, b) for s in mapped]
    rng = [p.left] * 3 + [p.right] * 3
    mm = [[100.0 + x * r for x, r in zip(s, rng)] for s in mapped]
    return mm[::-1] if order == "reversed" else mm


def run_shapes(S, shapes_mm, settle_after=None):
    """A sequence of set_targets_mm shapes, then the rest shape (200 mm each) and settle_after s braked (as a
    primitive ends). Returns a row like drive()'s (SE(2) in the start frame, faces, timeouts)."""
    n0 = S.nodes()
    b0 = metrics.bottom3(n0)
    t0 = S.time
    tout = 0
    for L in list(shapes_mm) + [[200.0] * 6]:
        ok, _ = set_targets_mm(S, L)
        tout += 0 if ok else 1
    S.settle(prm.value("settle_after") if settle_after is None else settle_after)
    dx, dy, dth, dist = metrics.relative(n0, S.nodes())
    b1 = metrics.bottom3(S.nodes())
    return {"dx": dx, "dy": dy, "dtheta_deg": dth, "dist": dist, "t_total": S.time - t0, "timeouts": tout,
            "bottom_start": list(b0), "bottom_end": list(b1), "tipped": b1 != b0,
            "rollable_end": b1 in gaits.SYMMETRY, "end_triangle": b1 in gaits.END_TRIANGLES,
            "com": S.com().tolist(), "pose": pose(S)}


# ----------------------------------------------------------------------------- protocols (one trial each)
def trial_flat(seed, name, reps=3, reverse=False, mu=None):
    """Flat floor (no world): start yaw and face from the seed, then `reps` executions of `name` measured one by one
    (envelope, turning footprint, speed). mu: floor friction (set_floor_friction), default the model's 0.8."""
    g = _rng(seed)
    S = new_sim()
    if mu is not None:
        set_floor_friction(S, mu)
    start(S, (0.0, 0.0), float(g.uniform(0, 360)), pre_rolls=int(seed % 3))
    tr = Tracker(S)
    out = drive(S, name, max_prims=reps, tracker=tr, reverse=reverse, stop_on_stuck=False)
    out.update({"seed": seed, "primitive": name, "mu": mu, "health": _health(tr)})
    return out


def history_to(face, g, max_len=3):
    """A seeded preparation from stand() (face 025) that lands on `face`: for the faces of class B a random sequence
    of 1..max_len published forward primitives (predicted with FACE_NEXT_*); for class A the validation's
    preparations (validate.PREP: stand() for 025, the authors' reverse rolls for 124 / 034)."""
    face = tuple(face)
    if face in CLASS_B:
        for _ in range(500):
            seq = [str(g.choice(gaits.NAMES)) for _ in range(int(g.integers(1, max_len + 1)))]
            f = (0, 2, 5)
            for nm in seq:
                f = next_face(f, nm)
            if f == face:
                return [[nm, False] for nm in seq]
        raise RuntimeError("no history found")
    from . import validate
    opts = validate.PREP[face]
    return [[nm, bool(rev)] for nm, rev in opts[int(g.integers(0, len(opts)))]]


def trial_prim(seed, name, face, start_mode="drop", mu=None, reverse=False):
    """PRIM_TABLE (the 07 envelope's primitive table): one execution of `name` from the rollable face `face` on flat ground.
    start_mode "drop": the rest shape after stand() placed on `face` by place_on_face (the 07 start rule; random
    yaw); "history": a seeded preparation (history_to) of published primitives, then re-placed at the origin with a
    random yaw. Envelope per execution from a Tracker (W_ROLL, W_SWEEP, D_ROLL, H_ROLL, R_TURN). reverse=True (rolls
    only): the authors' reverse gait of `name` (the reverse-roll table by start face; 07 offers it only as set_targets
    shapes)."""
    g = _rng(seed)
    face = tuple(face)
    S = new_sim()
    if mu is not None:
        set_floor_friction(S, mu)
    place(S, (0.0, 0.0), float(g.uniform(0, 360)))
    S.stand()
    hist = []
    if start_mode == "drop":
        place_on_face(S, face, (0.0, 0.0), float(g.uniform(0, 360)))
    elif start_mode == "history":
        hist = history_to(face, g)
        for nm, rev in hist:
            S.run_primitive(nm, reverse=rev)
        place(S, (0.0, 0.0), float(g.uniform(0, 360)))
        S.settle(1.0, vtol=5e-4)
    else:
        raise ValueError(start_mode)
    b = S.bottom()[0]
    out = {"seed": seed, "primitive": name, "face": list(face), "start_mode": start_mode, "history": hist,
           "face_before": list(b), "mu": mu, "reverse": bool(reverse)}
    if b != face:
        out.update({"face_miss": True, "prims": [], "success": False})
        return out
    tr = Tracker(S)
    out.update(drive(S, name, None, 1, tr, reverse=reverse, stop_on_stuck=False))
    out["health"] = _health(tr)
    return out


def trial_sequence(seed, seq, n=8, mu=None, table=None, pre_max=3):
    """Open-loop dispersion (DRIFT5 = n 5 of an identical sequence): a known start = the robot at
    rest on a face of class B after a seeded random history of 1..pre_max published primitives, placed at the origin
    with its rolling direction +y; then n primitives with no pose feedback: `seq` = a published name (n identical
    executions) or "mix_straight" / "mix_left90" / "mix_right90" (dead_reckon_plan from the known start with the
    table: a straight line; 0.55 m straight then a 90 deg turn to the left / right and straight on). mu: floor
    friction, a number or "site" = U(0.76, 0.92) per seed (07 site rule). The pose after every primitive is in
    the rows (start frame = world frame)."""
    g = _rng(seed)
    S = new_sim()
    if mu == "site":
        mu = float(g.uniform(0.76, 0.92))
    if mu is not None:
        set_floor_friction(S, mu)
    place(S, (0.0, 0.0), float(g.uniform(0, 360)))
    S.stand()
    pre = [str(g.choice(gaits.NAMES)) for _ in range(int(g.integers(1, pre_max + 1)))]
    for nm in pre:
        S.run_primitive(nm)
    place(S, (0.0, 0.0), 90.0)
    S.settle(1.0, vtol=5e-4)
    n0 = S.nodes()
    c0, th0 = metrics.se2(n0)
    face0 = S.bottom()[0]
    plan_info = None
    if seq in ("mix_straight", "mix_left90", "mix_right90"):
        legs = [{"p0": c0.tolist(), "heading_deg": 90.0, "length": 99.0 if seq == "mix_straight" else 0.55}]
        if seq != "mix_straight":
            legs.append({"p0": [float(c0[0]), float(c0[1]) + 0.55], "heading_deg": 180.0 if seq == "mix_left90" else 0.0,
                         "length": 99.0})
        plan_info = dead_reckon_plan(table, face0, c0, th0, legs, max_prims=n)
        names = plan_info["names"]
    else:
        names = [seq] * n
    out = drive(S, list(names), None, n, None, stop_on_stuck=False)
    out.update({"seed": seed, "sequence": seq, "mu": mu, "pre": pre, "face0": list(face0), "pose0": [float(c0[0]), float(c0[1]), math.degrees(th0)],
                "plan": plan_info, "device": "dead_reckon_plan (open loop)" if plan_info else "identical primitives (open loop)"})
    return out


def _straight_world(feats, name):
    w = wd.World(name)
    w.add(feats)
    return w


def trial_step_up(seed, h, prim=STRAIGHT, max_prims=8):
    """Step of height h across the path (edge at y = 0, 3 m wide); start 0.25-0.50 m before it, heading +-10 deg."""
    g = _rng(seed)
    D, yaw = float(g.uniform(0.25, 0.50)), float(g.uniform(-10, 10))
    w = _straight_world(wd.step("step", -1.5, 1.5, 0.0, 2.0, h), "step_up_%.3f" % h)
    S = new_sim(w)
    start(S, (0.0, -D - 0.2), 90.0 + yaw, pre_rolls=int(seed % 3))
    tr = Tracker(S, w)
    ok = lambda s: float(np.sort(s.nodes()[:, 2])[:3].max()) > h + 0.005 and s.com()[1] > 0.10
    out = drive(S, prim, ok, max_prims, tr)
    out.update({"seed": seed, "h": h, "D": D, "yaw": yaw, "primitive": prim, "health": _health(tr)})
    return out


def trial_drop(seed, h, prim=STRAIGHT, max_prims=8, after=2):
    """Leave a platform of height h (edge at y = 0) rolling; success = landed on the lower floor AND then `after`
    further forward rolls each tip and travel > 0.10 m (the robot can still roll). Before them the robot is turned
    away from the platform wall with published ccw turns (<= 8) where its rolling direction points back at it
    (0.3.0: no reverse roll; 07 provides none). There is no damage model."""
    g = _rng(seed)
    D, yaw = float(g.uniform(0.20, 0.45)), float(g.uniform(-10, 10))
    w = _straight_world(wd.step("platform", -1.5, 1.5, -2.0, 0.0, h), "drop_%.3f" % h)
    S = new_sim(w)
    start(S, (0.0, -D - 0.2), 90.0 + yaw, z_support=h, pre_rolls=int(seed % 3))
    tr = Tracker(S, w)
    landed = lambda s: s.com()[1] > 0.05 and float(np.sort(s.nodes()[:, 2])[:3].max()) < 0.03   # resting on the floor
    out = drive(S, prim, landed, max_prims, tr)
    ok_after = None
    peak_v = tr.mon.report()["max_node_speed_m_s"]
    if out["success"]:
        turns = 0
        while math.sin(math.radians(roll_dir_deg(S.nodes()))) < 0.3 and turns < 8:
            drive(S, "ccw", None, 1, tr, stop_on_stuck=False)
            turns += 1
        a_rows = drive(S, prim, None, after, tr, stop_on_stuck=False)["prims"]
        ok_after = len(a_rows) == after and all(r["tipped"] and r["dist"] > 0.10 for r in a_rows)
        out["after"] = a_rows
        out["turns_before_after"] = turns
    out.update({"seed": seed, "h": h, "D": D, "yaw": yaw, "primitive": prim, "landed": out["success"],
                "rolls_after": ok_after, "success": bool(out["success"] and ok_after),
                "peak_node_speed_before_after_m_s": peak_v, "health": _health(tr)})
    return out


def trial_slope(seed, angle_deg, direction, prim=STRAIGHT, n=5):
    """Uniform slope by tilting gravity (exact for an infinite incline): uphill = world +y. direction 'up' / 'down'
    / 'across' (rolling toward +y / -y / +x). Measures the slide while standing, progress per primitive along the
    rolling direction, downhill drift, tip rate and run-away (still moving > 5 mm/s after the hold settle_after)."""
    g = _rng(seed)
    yaw = float(g.uniform(-8, 8))
    a = math.radians(angle_deg)
    S = new_sim()
    gmag = float(np.linalg.norm(S.m.opt.gravity))
    S.m.opt.gravity[:] = [0.0, -gmag * math.sin(a), -gmag * math.cos(a)]
    rd = {"up": 90.0, "down": -90.0, "across": 0.0}[direction] + yaw
    place(S, (0, 0), rd)
    S.stand()
    place(S, tuple(S.com()[:2]), rd)          # re-aim in place (the slope plane is the floor plane z = 0)
    c0 = S.com()[:2].copy()
    S.settle(2.0)
    slide_stand = float(np.linalg.norm(S.com()[:2] - c0))
    tr = Tracker(S)
    out = drive(S, prim, None, n, tr, stop_on_stuck=False)
    u = np.array([math.cos(math.radians(rd - yaw)), math.sin(math.radians(rd - yaw))])
    prog, drift, runaway = [], [], 0
    prev = c0
    for r in out["prims"]:
        c = np.array(r["com"][:2])
        prog.append(float((c - prev) @ u))
        drift.append(float(-(c - prev)[1]))      # + = downhill
        prev = c
    v_end = S.max_node_speed()
    out.update({"seed": seed, "angle_deg": angle_deg, "direction": direction, "yaw": yaw, "primitive": prim,
                "slide_while_standing_m": slide_stand, "progress_m": prog, "downhill_drift_m": drift,
                "end_node_speed_m_s": v_end, "tip_rate": float(np.mean([r["tipped"] for r in out["prims"]])) if out["prims"] else 0.0,
                "health": _health(tr)})
    return out


def trial_ramp(seed, angle_deg, run=0.8, plateau=0.8, prim=STRAIGHT, max_prims=10):
    """World ramp from the floor (start line y = 0) to a plateau; success = robot resting on the plateau."""
    g = _rng(seed)
    D, yaw = float(g.uniform(0.25, 0.50)), float(g.uniform(-8, 8))
    w = _straight_world(wd.ramp("ramp", -1.5, 1.5, 0.0, run, angle_deg, plateau=plateau), "ramp_%g" % angle_deg)
    rise = run * math.tan(math.radians(angle_deg))
    S = new_sim(w)
    start(S, (0.0, -D - 0.2), 90.0 + yaw, pre_rolls=int(seed % 3))
    tr = Tracker(S, w)
    ok = lambda s: s.com()[1] > run + 0.12 and float(np.sort(s.nodes()[:, 2])[:3].max()) > rise + 0.005
    out = drive(S, prim, ok, max_prims, tr)
    out.update({"seed": seed, "angle_deg": angle_deg, "rise": rise, "D": D, "yaw": yaw, "primitive": prim, "health": _health(tr)})
    return out


def passage_world(kind, clear_w, clear_h=None, depth=0.55, wall_h=0.80, wall_len=3.0):
    """Feature centred at the origin with its passage along +y: kind 'gap' (open top), 'door' (portal with
    lintel), 'tunnel' (slab on two supports, 0.18 m supports, 0.10 m slab)."""
    if kind == "gap":
        f = wd.portal("gap", (0, 0), 0.0, wall_len, depth, wall_h, clear_w)
    elif kind == "door":
        f = wd.portal("door", (0, 0), 0.0, wall_len, depth, wall_h, clear_w, clear_h)
    elif kind == "tunnel":
        f = wd.tunnel("tunnel", (0, 0), 0.0, clear_w, clear_h, depth)
    else:
        raise ValueError(kind)
    return _straight_world(f, "%s_w%.3f_h%s_d%.2f" % (kind, clear_w, clear_h, depth))


def touching(S, geoms):
    """Names of the given world geoms in contact with the robot now."""
    out = set()
    for i in range(S.d.ncon):
        c = S.d.contact[i]
        for g in (c.geom1, c.geom2):
            if g in geoms:
                out.add(mujoco.mj_id2name(S.m, mujoco.mjtObj.mjOBJ_GEOM, g))
    return sorted(out)


def _stop_state(S, tr, ok):
    tnow = touching(S, tr.world_geoms)
    sides = {nm.split("_")[-1] for nm in tnow if "_jamb_" in nm or "_sup_" in nm or "_side_" in nm}
    return ("passed" if ok else ("wedged" if len(sides) >= 2 else ("blocked" if tnow else "free"))), tnow


def trial_passage(seed, kind, clear_w, clear_h=None, depth=0.55, prim=STRAIGHT, max_extra=4, free_tries=0, table=None,
                  mode=None):
    """Passage along +y through a feature centred at the origin. Start 0.30-0.50 m before the near face, lateral
    offset +-0.03 m, heading +-5 deg (true pose; the believed pose is the nominal one on the axis), 0-2 pre-rolls.
    mode "steered": privileged_steer_plan (PRIVILEGED MEASURING DEVICE, true pose every primitive);
    "open_loop": the fixed sequence of dead_reckon_plan from the believed pose along the axis (no feedback);
    "repeat": `prim` every time (the 0.2.0 open-loop protocol). Default: steered if a table is given, else repeat.
    Success = COM 0.15 m past the far face; stops when stuck or out of budget. If it does not pass: stopped 'wedged'
    (touching both sides) / 'blocked' (touching one part) / 'free' (no contact); then up to free_tries reverse rolls
    of STRAIGHT (the 0.2.0 recovery check; 0.3.0 uses trial_pocket)."""
    g = _rng(seed)
    D, dx, yaw = float(g.uniform(0.30, 0.50)), float(g.uniform(-0.03, 0.03)), float(g.uniform(-5, 5))
    mode = mode or ("steered" if table else "repeat")
    w = passage_world(kind, clear_w, clear_h, depth)
    S = new_sim(w)
    y0 = -0.5 * depth - D - 0.17
    start(S, (dx, y0), 90.0 + yaw, pre_rolls=int(seed % 3))
    tr = Tracker(S, w)
    far = 0.5 * depth + 0.15
    budget = int(math.ceil((D + depth + 0.32) / 0.2)) + max_extra
    plan_info = None
    if mode == "steered":
        plan = privileged_steer_plan(S, table)
    elif mode == "open_loop":
        b, _ = S.bottom()
        plan_info = dead_reckon_plan(table, b, (0.0, y0), 0.0, [{"p0": (0.0, y0), "heading_deg": 90.0, "length": far - y0 + 0.05}],
                                     max_prims=budget)
        plan = plan_info["names"]
    else:
        plan = prim
    out = drive(S, plan, lambda s: s.com()[1] > far, budget, tr)
    y = S.com()[1]
    where = "passed" if out["success"] else ("before" if y < -0.5 * depth - 0.05 else ("inside" if y < 0.5 * depth else "past"))
    state, tnow = _stop_state(S, tr, out["success"])
    res = {"seed": seed, "kind": kind, "clear_w": clear_w, "clear_h": clear_h, "depth": depth, "D": D, "dx": dx, "yaw": yaw,
           "mode": mode, "device": {"steered": "privileged_steer_plan", "open_loop": "dead_reckon_plan",
                                    "repeat": "repeat " + prim}[mode],
           "primitive": prim if mode == "repeat" else None, "where_stopped": where, "plan": plan_info,
           "stop_state": state, "touching_at_stop": tnow, "contacts_feature": bool(tr.world_contact_geoms),
           "names_used": sorted({r["name"] for r in out["prims"]}),
           "max_lateral_m": float(max(abs(r["com"][0]) for r in out["prims"])) if out["prims"] else 0.0}
    if not out["success"] and free_tries:
        y_stop = y
        ok_free = lambda s: (not touching(s, tr.world_geoms)) and s.com()[1] < y_stop - 0.05
        fr = drive(S, STRAIGHT, ok_free, free_tries, tr, reverse=True, stop_on_stuck=False)
        res.update({"freed": bool(fr["success"]), "n_reverse": fr["n_prims"],
                    "free_prims": [{k: r[k] for k in ("dist", "dtheta_deg", "tipped", "com")} for r in fr["prims"]],
                    "touching_after_free": touching(S, tr.world_geoms)})
    out.update(res)
    out["health"] = _health(tr)
    return out


def relief07_patch(name, x0, x1, y0, y1, seed, rms=0.008, peak=0.025, wl=(0.03, 0.08), edge=0.05, cell=0.01,
                   friction=None, iters=30):
    """The 07 site rule's gravel relief ('bumps only, heights >= 0 on the floor; grid 0.01 m, wavelength 0.03-0.08 m,
    rms 8 mm, peak <= 25 mm, 0.05 m taper'; the 07 site generator's rule w2gen/mjcf.py relief_heights: the clipped,
    tapered field is rescaled until its rms over the whole patch, zeros included, is `rms`) built with
    world.heightfield_patch(base=0): the std of the band-limited noise before clipping is found by the same
    rescaling iteration. Same statistics as the 07 rule, not the same random field (seeded here by `seed`)."""
    nx = int(round((x1 - x0) / cell)) + 1
    ny = int(round((y1 - y0) / cell)) + 1
    r = rms
    for _ in range(iters):
        E = wd.relief_grid(nx, ny, cell, seed, r, peak, wl[0], wl[1], 0.0, edge)
        q = float(np.sqrt(np.mean(E ** 2)))
        if q <= 0:
            break
        r *= rms / q
    f = wd.heightfield_patch(name, x0, x1, y0, y1, seed, rms=r, peak=peak, wl=wl, base=0.0, edge=edge, cell=cell,
                             material="gravel", friction=friction)
    E = wd.hfield_grid(f[0])
    f[0].meta.update({"rule": "07 site rule / w2gen relief_heights (bumps only, rms incl. zeros)",
                      "rms_incl_zeros_m": float(np.sqrt(np.mean(E ** 2))), "noise_std_m": r,
                      "bump_cover": float(np.mean(E > 1e-4))})
    return f


def trial_patch(seed, material, name, reps=2, friction=None, size=2.0, relief=True, relief_rule="ysim"):
    """Robot in the middle of a size x size patch: 'dust' / 'concrete' (friction patch; friction overrides) or
    'gravel': relief=True a relief hfield seeded by the trial seed (mu 0.60), relief=False a flat 2 mm friction
    patch with the gravel friction 0.60 (07 site rule: gravel_relief off by default). relief_rule "ysim": the 0.2.0
    layer (std 8 mm about a 20 mm mean thickness, world.heightfield_patch); "07": the 07 bumps-only relief
    (relief07_patch)."""
    g = _rng(seed)
    yaw = float(g.uniform(0, 360))
    h = 0.5 * size
    if material == "gravel" and relief:
        if relief_rule == "07":
            f = relief07_patch("patch", -h, h, -h, h, seed=1000 + seed, friction=friction)
        else:
            f = wd.heightfield_patch("patch", -h, h, -h, h, seed=1000 + seed, friction=friction)
        zs = f[0].meta["ztop"]
    else:
        f = wd.friction_patch("patch", -h, h, -h, h, material, friction)
        zs = f[0].meta["thickness"]
    w = _straight_world(f, "patch_%s_%s_%s" % (material, friction, relief))
    S = new_sim(w)
    start(S, (0.0, 0.0), yaw, z_support=zs, pre_rolls=int(seed % 3))
    tr = Tracker(S, w)
    out = drive(S, name, None, reps, tr, stop_on_stuck=False)
    out.update({"seed": seed, "material": material, "relief": bool(material == "gravel" and relief),
                "relief_rule": relief_rule if (material == "gravel" and relief) else None,
                "friction": friction if friction is not None else wd.MATERIALS[material]["friction"],
                "primitive": name, "health": _health(tr)})
    if material == "gravel" and relief:
        out["relief_stats"] = {k: f[0].meta[k] for k in ("relief_std_m", "mean_thickness_m", "ztop", "rms_incl_zeros_m",
                                                         "bump_cover") if k in f[0].meta}
    return out


def trial_patch_cross(seed, prim=STRAIGHT, surface="relief07", depth=0.60, max_prims=10):
    """Rolling ACROSS a gravel strip from the concrete floor (07 site rule: a 0.60 m deep gravel strip in front of
    the R2 row): strip y 0..depth, x -1..1; surface "relief07" (relief07_patch), "relief_ysim" (the 0.2.0 layer
    with its 20 mm mean thickness) or "flat" (2 mm friction patch, mu 0.60). Start 0.25-0.50 m before the strip,
    heading +-8 deg, 0-2 pre-rolls; `prim` repeated (open loop). Success = node centroid 0.15 m past the far edge.
    Records the progress and tipping of every primitive that started on the strip."""
    g = _rng(seed)
    D, yaw = float(g.uniform(0.25, 0.50)), float(g.uniform(-8, 8))
    if surface == "relief07":
        f = relief07_patch("strip", -1.0, 1.0, 0.0, depth, seed=2000 + seed)
    elif surface == "relief_ysim":
        f = wd.heightfield_patch("strip", -1.0, 1.0, 0.0, depth, seed=2000 + seed)
    elif surface == "flat":
        f = wd.friction_patch("strip", -1.0, 1.0, 0.0, depth, "gravel")
    else:
        raise ValueError(surface)
    w = _straight_world(f, "cross_%s_%.2f" % (surface, depth))
    S = new_sim(w)
    start(S, (0.0, -D - 0.17), 90.0 + yaw, pre_rolls=int(seed % 3))
    tr = Tracker(S, w)
    far = depth + 0.15
    out = drive(S, prim, lambda s: metrics.se2(s.nodes())[0][1] > far, max_prims, tr)
    on = [r for i, r in enumerate(out["prims"]) if (out["prims"][i - 1]["pose"][1] if i else -D - 0.17) > -0.05]
    out.update({"seed": seed, "surface": surface, "depth": depth, "D": D, "yaw": yaw, "primitive": prim,
                "y_end_m": float(metrics.se2(S.nodes())[0][1]),
                "on_strip_rows": len(on), "on_strip_tip_rate": float(np.mean([r["tipped"] for r in on])) if on else None,
                "on_strip_progress_m": [float(r["dy"]) for r in on],
                "relief_stats": {k: f[0].meta[k] for k in ("ztop", "rms_incl_zeros_m", "bump_cover", "mean_thickness_m")
                                 if k in f[0].meta}, "health": _health(tr)})
    return out


def trial_face(seed, name, max_pre=4):
    """0.2.0 protocol (superseded by trial_prim): one execution of `name` from a start face drawn by a seeded
    pre-sequence of 0..max_pre straight rolls, each forward or reverse."""
    g = _rng(seed)
    S = new_sim()
    place(S, (0.0, 0.0), float(g.uniform(0, 360)))
    S.stand()
    for _ in range(int(g.integers(0, max_pre + 1))):
        rev = bool(g.random() < 0.5)
        S.run_primitive(REV_STRAIGHT if rev else STRAIGHT, reverse=rev)
    place(S, (0.0, 0.0), float(g.uniform(0, 360)))
    S.settle(1.0, vtol=5e-4)
    tr = Tracker(S)
    out = drive(S, name, None, 1, tr, stop_on_stuck=False)
    out.update({"seed": seed, "primitive": name, "health": _health(tr)})
    return out


def trial_drift(seed, name, n=5):
    """0.2.0 protocol (superseded by trial_sequence): n identical executions of `name` from a seeded state."""
    g = _rng(seed)
    S = new_sim()
    place(S, (0.0, 0.0), float(g.uniform(0, 360)))
    S.stand()
    for _ in range(int(g.integers(0, 4))):
        rev = bool(g.random() < 0.5)
        S.run_primitive(REV_STRAIGHT if rev else STRAIGHT, reverse=rev)
    S.settle(1.0, vtol=5e-4)
    n0 = S.nodes()
    out = drive(S, name, None, n, None, stop_on_stuck=False)
    dx, dy, dth, dist = metrics.relative(n0, S.nodes())
    out.update({"seed": seed, "primitive": name, "end_dx": dx, "end_dy": dy, "end_dtheta_deg": dth, "end_dist": dist})
    return out


def trial_wall(seed, prim=STRAIGHT, n=7):
    """Roll into a wall (face at y = 0, 3 m wide, 0.8 m high) with the same published roll every time: D_STOP =
    distance of the node centroid from the wall face after the primitives in which the robot touched the wall;
    END_TRI = landing on an end triangle."""
    g = _rng(seed)
    D, yaw = float(g.uniform(0.30, 0.60)), float(g.uniform(-15, 15))
    w = _straight_world([wd.box("wall", -1.5, 1.5, 0.0, 0.2, 0.0, 0.8, "wall", "wall")], "wall")
    S = new_sim(w)
    start(S, (0.0, -D - 0.17), 90.0 + yaw, pre_rolls=int(seed % 3))
    tr = Tracker(S, w)
    out = drive(S, prim, None, n, tr, stop_on_stuck=False)
    rows = out["prims"]
    out.update({"seed": seed, "D": D, "yaw": yaw, "primitive": prim, "health": _health(tr),
                "d_wall_after_each_m": [-r["pose"][1] for r in rows],
                "end_triangle_after_each": [bool(r["end_triangle"]) for r in rows],
                "touching_wall_end": bool(touching(S, tr.world_geoms))})
    return out


def trial_creep(seed, name, hold=30.0):
    """CREEP (a key of the 07 envelope): after one execution of `name` from stand(), hold braked for `hold` s; drift of the node
    centroid and of the nodes, end speed."""
    g = _rng(seed)
    S = new_sim()
    place(S, (0.0, 0.0), float(g.uniform(0, 360)))
    S.stand()
    r = S.run_primitive(name)
    c0, n0 = metrics.se2(S.nodes())[0], S.nodes()
    com0 = S.com().copy()
    S.settle(hold)
    n1 = S.nodes()
    c1 = metrics.se2(n1)[0]
    return {"seed": seed, "primitive": name, "hold_s": hold, "bottom": list(S.bottom()[0]),
            "drift_centroid_xy_mm": 1000 * float(np.linalg.norm(c1 - c0)),
            "drift_com_mm": 1000 * float(np.linalg.norm(S.com() - com0)),
            "node_drift_max_mm": 1000 * float(np.linalg.norm(n1 - n0, axis=1).max()),
            "end_speed_m_s": S.max_node_speed(), "rest_reached": bool(r["rest_reached"])}


def gallery_world(width, length, clear_h, roof_t=0.12):
    """A covered straight gallery along +y from y = 0 to length: two 0.18 m supports, a roof slab (07 site rule:
    slab 0.12 thick)."""
    return _straight_world(wd.tunnel("gal", (0.0, 0.5 * length), 0.0, width, clear_h, length, roof_t=roof_t),
                           "gallery_w%.3f_l%.2f_h%.3f" % (width, length, clear_h))


def trial_blind_gallery(seed, width, length, clear_h, table, start_err=(0.03, 5.0), stops=(1.4, 1.8, 2.1)):
    """Blind leg in a covered gallery: believed start = 0.25 m before the entrance on the axis,
    rolling along +y; true start off by U(+-start_err[0]) m and U(+-start_err[1]) deg, 0-2 pre-rolls. The fixed
    sequence of dead_reckon_plan along the axis (OPEN LOOP, no feedback) runs until it ends, the robot is stuck or
    it leaves the far end. Records the progress along the gallery after every primitive and which thresholds
    (`stops`, m from the entrance) it passed."""
    g = _rng(seed)
    dx, yaw = float(g.uniform(-start_err[0], start_err[0])), float(g.uniform(-start_err[1], start_err[1]))
    w = gallery_world(width, length, clear_h)
    S = new_sim(w)
    y0 = -0.25
    start(S, (dx, y0), 90.0 + yaw, pre_rolls=int(seed % 3))
    tr = Tracker(S, w)
    b, _ = S.bottom()
    target = length + 0.25
    plan_info = dead_reckon_plan(table, b, (0.0, y0), 0.0, [{"p0": (0.0, y0), "heading_deg": 90.0, "length": target - y0}],
                                 max_prims=40)
    out = drive(S, plan_info["names"], lambda s: s.com()[1] > target, len(plan_info["names"]), tr)
    ys = [r["pose"][1] for r in out["prims"]]
    ymax = max(ys) if ys else y0
    state, tnow = _stop_state(S, tr, False)
    out.update({"seed": seed, "width": width, "length": length, "clear_h": clear_h, "dx": dx, "yaw": yaw,
                "device": "dead_reckon_plan (open loop)", "plan": plan_info, "y_max_m": ymax,
                "passed": {"%.1f" % s: bool(ymax >= s) for s in stops}, "stop_state": state, "touching_at_stop": tnow,
                "contacts_feature": bool(tr.world_contact_geoms), "health": _health(tr)})
    return out


def l_route_world(P_w, P_h, gal_h, Lf, gdepth, X=2.4, slab_t=0.12):
    """07 site rule, R2 shape: a front row y 0..Lf with one covered mouth (clear width P_w, clear height P_h)
    at x = 0, behind it a covered transverse gallery y Lf..Lf + gdepth (clear height gal_h) running along x from -X
    to X, closed by a north wall and two end walls; one slab over all of it."""
    b = []
    yN = Lf + gdepth
    b.append(wd.box("fr_side_w", -X - 0.2, -0.5 * P_w, 0.0, Lf, 0.0, gal_h, "wall", "wall"))
    b.append(wd.box("fr_side_e", 0.5 * P_w, X + 0.2, 0.0, Lf, 0.0, gal_h, "wall", "wall"))
    if P_h < gal_h - 1e-6:
        b.append(wd.box("fr_lintel", -0.5 * P_w, 0.5 * P_w, 0.0, Lf, P_h, gal_h, "roof"))
    b.append(wd.box("gal_north", -X - 0.2, X + 0.2, yN, yN + 0.2, 0.0, gal_h, "wall", "wall"))
    b.append(wd.box("gal_end_w", -X - 0.2, -X, Lf, yN, 0.0, gal_h, "wall", "wall"))
    b.append(wd.box("gal_end_e", X, X + 0.2, Lf, yN, 0.0, gal_h, "wall", "wall"))
    b.append(wd.box("slab", -X - 0.2, X + 0.2, -0.04, yN + 0.2, gal_h, gal_h + slab_t, "roof"))
    return _straight_world(b, "lroute_P%.3f_h%.3f_g%.3f" % (P_w, P_h, gdepth))


def trial_l_route(seed, gdepth, turn, s_along, P_w, P_h, gal_h, table, Lf=0.55, start_err=(0.03, 5.0), r_full=0.33, r_zero=0.66):
    """The blind route under a slab (07 site rule): believed start D ~ U(0.30, 0.50) m in front of
    the covered mouth on its axis, rolling north (+y); true start off by U(+-start_err); 0-2 pre-rolls. The fixed
    sequence of dead_reckon_plan (OPEN LOOP): north along the mouth axis to the gallery centreline, then a 90 deg turn
    to `turn` ('left' = west, 'right' = east) and s_along m along the gallery to the point SP = (+-s_along, Lf +
    gdepth / 2). Records the end error to SP and whether it is within r_full / r_zero (the 07 credit radii 0.33 /
    0.66 m)."""
    g = _rng(seed)
    D = float(g.uniform(0.30, 0.50))
    dx, yaw = float(g.uniform(-start_err[0], start_err[0])), float(g.uniform(-start_err[1], start_err[1]))
    w = l_route_world(P_w, P_h, gal_h, Lf, gdepth)
    S = new_sim(w)
    y0 = -D
    start(S, (dx, y0), 90.0 + yaw, pre_rolls=int(seed % 3))
    tr = Tracker(S, w)
    yc = Lf + 0.5 * gdepth
    hd = 180.0 if turn == "left" else 0.0
    sp = np.array([-s_along if turn == "left" else s_along, yc])
    b, _ = S.bottom()
    legs = [{"p0": (0.0, y0), "heading_deg": 90.0, "length": yc - y0}, {"p0": (0.0, yc), "heading_deg": hd, "length": s_along}]
    plan_info = dead_reckon_plan(table, b, (0.0, y0), 0.0, legs, max_prims=40)
    out = drive(S, plan_info["names"], None, len(plan_info["names"]), tr)
    c = metrics.se2(S.nodes())[0]
    err = float(np.linalg.norm(c - sp))
    state, tnow = _stop_state(S, tr, False)
    out.update({"seed": seed, "gdepth": gdepth, "turn": turn, "s_along": s_along, "P_w": P_w, "P_h": P_h, "gal_h": gal_h,
                "Lf": Lf, "D": D, "dx": dx, "yaw": yaw, "device": "dead_reckon_plan (open loop)", "plan": plan_info,
                "sp": sp.tolist(), "end_xy": c.tolist(), "end_err_m": err, "within_full": err <= r_full,
                "within_zero": err <= r_zero, "in_gallery": bool(c[1] > Lf), "planned_end": plan_info["pred"][-1] if plan_info["pred"] else None,
                "stop_state": state, "touching_at_stop": tnow, "health": _health(tr)})
    return out


def pocket_world(width, depth, clear_h, X=1.5, slab_t=0.12):
    """A covered dead-end pocket along +y (07 site class B: looks like a covered P, blocked inside): side
    blocks |x| >= width / 2 from y 0 to depth + 0.2, a plug across the pocket at y = depth, a slab over it."""
    b = [wd.box("pk_side_w", -X, -0.5 * width, 0.0, depth + 0.2, 0.0, clear_h, "wall", "wall"),
         wd.box("pk_side_e", 0.5 * width, X, 0.0, depth + 0.2, 0.0, clear_h, "wall", "wall"),
         wd.box("pk_plug", -0.5 * width, 0.5 * width, depth, depth + 0.2, 0.0, clear_h, "plug" if "plug" in wd.KINDS else "rubble"),
         wd.box("pk_roof", -X, X, -0.04, depth + 0.2, clear_h, clear_h + slab_t, "roof")]
    return _straight_world(b, "pocket_w%.3f_d%.2f_h%.3f" % (width, depth, clear_h))


def trial_pocket(seed, width, depth, clear_h, method, table, rev_name=None, rev_by_face=None, enter_max=10, press=2,
                 out_max=16, shape_tries=5):
    """Dead-end pocket (07 site class B: looks like a covered P, blocked inside). Believed start
    0.30-0.50 m in front on the axis, true start off by +-0.03 m / +-5 deg, 0-2 pre-rolls.
    Phase 1 (in): the robot is driven in by privileged_steer_plan (PRIVILEGED MEASURING DEVICE, true pose, aims along
    the pocket axis as if the pocket went through) until `press` primitives have touched the plug (it reached the
    dead end and pushed against it once more), it is stuck (STUCK_WINDOW), or enter_max primitives; the stop state
    says whether it wedged (touching both sides), is blocked by one part or stopped free.
    Phase 2 (out), by `method`:
      "primitives":     published primitives only (turns and rolls), chosen by privileged_steer_plan toward -y on
                        the pocket axis, at most out_max;
      "reverse_gait":   set_targets shapes (set_targets_mm, as the 07 cell) = the authors' reverse gait of a forward
                        roll, then the rest shape; the roll per current face from rev_by_face {face_key: name} (the
                        sheet's reverse-roll table by start face, by a stated rule of the maintainers, not shipped), else rev_name; at most
                        shape_tries times;
      "reversed_order": set_targets shapes = that roll's own mapped shapes played in reversed order; at most
                        shape_tries times.
    Out = node centroid at y < -0.25 m (outside the front face) and no contact with the pocket."""
    g = _rng(seed)
    D, dx, yaw = float(g.uniform(0.30, 0.50)), float(g.uniform(-0.03, 0.03)), float(g.uniform(-5, 5))
    w = pocket_world(width, depth, clear_h)
    S = new_sim(w)
    start(S, (dx, -D), 90.0 + yaw, pre_rolls=int(seed % 3))
    tr = Tracker(S, w)
    plug = "w_pk_plug"
    st = {"last": 0, "hits": 0}

    def pressed(s):
        c = tr.world_contact_geoms.get(plug, 0)
        if c > st["last"]:
            st["hits"] += 1
        st["last"] = c
        return st["hits"] >= press
    inn = drive(S, privileged_steer_plan(S, table), pressed, enter_max, tr)
    y_in = max([r["pose"][1] for r in inn["prims"]] or [float(metrics.se2(S.nodes())[0][1])])
    st_in, t_in = _stop_state(S, tr, False)
    pose_in = pose(S)
    face_in = list(S.bottom()[0])
    out_ok = lambda s: metrics.se2(s.nodes())[0][1] < -0.25 and not touching(s, tr.world_geoms)
    rows, err, used = [], None, []
    if method == "primitives":
        res = drive(S, privileged_steer_plan(S, table, goal_deg=-90.0), out_ok, out_max, tr, stop_on_stuck=False)
        rows = res["prims"]
        ok = res["success"]
        if res["not_rollable"]:
            err = res["not_rollable"]
    else:
        ok = False
        for _ in range(shape_tries):
            b, rollable = S.bottom()
            nm = (rev_by_face or {}).get(face_key(b)) or rev_name
            try:
                sh = primitive_shapes_mm(S, nm, reverse=(method == "reverse_gait"),
                                         order="reversed" if method == "reversed_order" else "forward")
            except gaits.NotRollableFace as e:
                err = {"bottom": list(b), "error": str(e)}
                break
            tr.reset_frame()
            r = run_shapes(S, sh)
            r["name"] = "%s(%s)" % (method, nm)
            r["touching"] = touching(S, tr.world_geoms)
            rows.append(r)
            used.append(nm)
            if out_ok(S):
                ok = True
                break
    state_out, t_out = _stop_state(S, tr, ok)
    return {"seed": seed, "width": width, "depth": depth, "clear_h": clear_h, "method": method, "rev_name": rev_name,
            "rev_used": used, "D": D, "dx": dx, "yaw": yaw, "device_in": "privileged_steer_plan",
            "device_out": "privileged_steer_plan" if method == "primitives" else "set_targets_mm shapes",
            "enter": {"n_prims": inn["n_prims"], "reached_plug_pressed": bool(inn["success"]), "plug_hits": st["hits"],
                      "y_max_m": y_in, "pose_end": pose_in, "face_end": face_in, "stop_state": st_in, "touching": t_in,
                      "stuck": inn["stuck"], "not_rollable": inn["not_rollable"], "prims": inn["prims"]},
            "out": {"success": bool(ok), "n": len(rows), "rows": rows, "stop_state": state_out, "touching": t_out,
                    "error": err, "y_after_each": [r["pose"][1] for r in rows]},
            "end_triangle_any": any(r.get("end_triangle") for r in inn["prims"] + rows),
            "health": _health(tr)}
