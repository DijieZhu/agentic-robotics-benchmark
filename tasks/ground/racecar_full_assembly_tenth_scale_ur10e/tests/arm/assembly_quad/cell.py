"""The quadruped assembly cell: the robot's body (with its four HAA modules) bolted on a stand on the bench, the sixteen
loose units of the kit resting on the bench, the arm (a UR with a 2F-85, `ur_robot.py`; optional for the floating
check), the coupling state machine (`coupling.py`).

`QuadCell(kit_dir, layout, config=None, robot_hook=None, spec_hook=None)` compiles from the kit (`kit.json`, meshes in
body frames):
  * the stand: two 40 x 40 posts under the body's centre line with a fork on each (the product's own stand; static
    geoms), its pose `layout["stand"]` = {xy, yaw_deg} (the body's x axis turned yaw about z), the body centre
    `config["stand"]["body_z"]` above the bench;
  * the base: a fixed body at the stand's top, its four HAA output pulleys on unpowered hinges (kit "hinge": the belt
    drive's Coulomb friction, its viscous term, the reflected rotor inertia), unlimited until a shoulder fastens;
  * every loose unit a free body (layout["parts"][unit] = (pos, quat)) with its output pulley on its hinge;
  * one weld per (slot, candidate) (inactive until the coupling captures / fastens); contact exclusions only between an
    output pulley and the module it turns in (every interface collides: the keys enter the sockets through the
    pulleys' collision, the supports' bores take the journals through carved clearance);
  * the robot through `robot_hook(spec, cfg) -> dict` (names of its tcp, joints, pads, cameras).
`live_target(slot, unit)` gives a unit's pose for a slot at the design from the carrier as it is now; `tick()` advances
one 10 ms control tick (controllers, jigs, physics, coupling) and records a frame every 4 ticks.
"""
import copy
import json
import math
import os
import time

import numpy as np

from .coupling import CouplingStateMachine, _inv, body_pose, quat_wxyz, weld_name

CELL_CONFIG = {
    # impratio 100: the held part's steady creep in the 2F-85's pads falls from 11 to 2.4 um/s (gripper lane, DESIGN 5);
    # the same contact model for the floating check and the robot (add_ur sets it too)
    "timestep": 0.001, "tick_steps": 10, "impratio": 100.0, "noslip_iterations": 5, "arena_memory_bytes": 768 << 20,
    # contact clusters: no contacts inside a rigidly fastened group (QuadCell.update_clusters)
    "clusters": True,
    # the arm (None: no robot, e.g. the floating check): ur_robot.add_ur's cfg (model, pos, yaw_deg, ...)
    "robot": None,
    "robot_home_q": [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
    "bench": {"size_xy": [2.0, 1.4], "centre_xy": [0.45, 0.0], "thickness": 0.03, "floor_z": -0.994, "friction": 0.8},
    "geom_defaults": {"condim": 4, "friction": [0.8, 0.005, 0.0001], "solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
    "weld_solref": [0.002, 1.0], "weld_solimp": [0.99, 0.999, 0.001, 0.5, 2.0],
    # the stand: two printed forks (121 mm wide, short tabs either side of the body) on 40 x 40 profiles, the body centre
    # `body_z` above the bench (DESIGN.md section 4: 0.50 m). The product's stand puts its forks under the body's end
    # boxes (x +-0.19); ours sit under the HAA modules (x +-0.10, their lowest point -31.5 mm), so that the hip AA
    # supports at the corners can be held from below and from the ends (place study, DESIGN.md 11.3); the tabs are 10 mm
    # (a 30 mm tab reaches into a shoulder's HAA swing circle)
    "stand": {"body_z": 0.50, "post_x": 0.10, "post": 0.040, "fork": [0.040, 0.121, 0.010], "tab": [0.024, 0.010, 0.010],
              "tab_gap": 0.101, "rail_z": 0.06, "foot": [0.040, 0.20, 0.012], "body_bottom_z": -0.0325},
    "materials": {"printed": [0.90, 0.89, 0.85, 1.0], "core": [0.14, 0.14, 0.15, 1.0], "pulley": [0.92, 0.92, 0.88, 1.0],
                  "rubber": [0.05, 0.05, 0.05, 1.0], "stand": [0.70, 0.72, 0.75, 1.0], "fork": [0.12, 0.12, 0.13, 1.0]},
}


def load_kit(kit_dir):
    kit = json.load(open(os.path.join(kit_dir, "kit.json")))
    kit["dir"] = os.path.abspath(kit_dir)          # the saved cell.xml names its meshes by this path
    return kit


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def T_world_robot(stand, cfg):
    """the robot (body) frame in the world: the body on the stand at stand["xy"], turned by stand["yaw_deg"]"""
    T = np.eye(4)
    T[:3, :3] = rotz(math.radians(float(stand.get("yaw_deg", 0.0))))
    T[:3, 3] = [float(stand["xy"][0]), float(stand["xy"][1]), float(cfg["stand"]["body_z"])]
    return T


def stand_boxes(cfg):
    """the stand's boxes in the robot frame: [(name, centre, half sizes)]"""
    st = cfg["stand"]
    zb = -float(st["body_z"])                      # the bench top
    top = float(st["body_bottom_z"])               # the fork plate's top under the body
    fk, tab, foot = st["fork"], st["tab"], st["foot"]
    plate_z = top - fk[2] / 2
    post_top = top - fk[2]
    out = []
    for sx in (1, -1):
        x = sx * float(st["post_x"])
        h = post_top - (zb + foot[2])
        out.append(("post%+d" % sx, [x, 0.0, zb + foot[2] + h / 2], [st["post"] / 2, st["post"] / 2, h / 2]))
        out.append(("fork%+d" % sx, [x, 0.0, plate_z], [fk[0] / 2, fk[1] / 2, fk[2] / 2]))
        for sy in (1, -1):
            y = sy * (float(st["tab_gap"]) / 2 + tab[1] / 2)
            out.append(("tab%+d%+d" % (sx, sy), [x, y, top + tab[2] / 2], [tab[0] / 2, tab[1] / 2, tab[2] / 2]))
        out.append(("foot%+d" % sx, [x, 0.0, zb + foot[2] / 2], [foot[0] / 2, foot[1] / 2, foot[2] / 2]))
    out.append(("rail", [0.0, 0.0, zb + float(st["rail_z"])], [float(st["post_x"]) - st["post"] / 2, st["post"] / 2, st["post"] / 2]))
    return out


def build_spec(kit, layout, cfg, robot_hook=None):
    import mujoco
    spec = mujoco.MjSpec()
    spec.modelname = "quadruped_assembly_cell"
    spec.memory = int(cfg["arena_memory_bytes"])
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    spec.option.timestep = float(cfg["timestep"])
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = int(cfg["noslip_iterations"])
    spec.option.impratio = float(cfg["impratio"])
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    gd = spec.default.geom
    gd.condim = int(cfg["geom_defaults"]["condim"])
    fr = list(cfg["geom_defaults"]["friction"]); fr[0] = float(cfg["bench"]["friction"])
    gd.friction = fr
    gd.solref = cfg["geom_defaults"]["solref"]
    gd.solimp = cfg["geom_defaults"]["solimp"]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.stat.extent, spec.stat.center = 1.8, [0.4, 0.0, 0.3]
    w = spec.worldbody
    w.add_light(name="sun", pos=[0.4, -0.4, 2.5], dir=[-0.15, 0.15, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    w.add_light(name="fill", pos=[-0.6, 0.8, 1.6], dir=[0.4, -0.6, -0.8], diffuse=[0.35, 0.35, 0.35])
    for name, rgba in cfg["materials"].items():
        spec.add_material(name="mat_" + name, rgba=list(rgba))
    spec.add_material(name="mat_bench", rgba=[0.80, 0.74, 0.62, 1.0])
    b = cfg["bench"]
    w.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[b["size_xy"][0] / 2, b["size_xy"][1] / 2, b["thickness"] / 2],
               pos=[b["centre_xy"][0], b["centre_xy"][1], -b["thickness"] / 2], material="mat_bench")
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5.0, 5.0, 0.1], pos=[0.0, 0.0, b["floor_z"]], rgba=[0.3, 0.3, 0.3, 1.0])
    Tr = T_world_robot(layout["stand"], cfg)
    stand = w.add_body(name="stand", pos=Tr[:3, 3].tolist(), quat=quat_wxyz(Tr[:3, :3]).tolist())
    for name, c, hs in stand_boxes(cfg):
        stand.add_geom(name="stand_" + name, type=mujoco.mjtGeom.mjGEOM_BOX, size=hs, pos=c,
                       material="mat_fork" if name.startswith(("fork", "tab")) else "mat_stand")
    kd = kit["dir"]
    bodies = kit["bodies"]
    children = {}
    for bn, bb in bodies.items():
        if bb.get("parent"):
            children.setdefault(bb["parent"], []).append(bn)

    def add_geoms(body, bn):
        bb = bodies[bn]
        for key, f in bb.get("visual", {}).items():
            mat = key.split("#")[0]
            nm = "vis_%s_%s" % (bn, key.replace("#", "_"))
            spec.add_mesh(name=nm, file=os.path.join(kd, f))
            body.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=nm, contype=0, conaffinity=0, group=2,
                          material="mat_" + mat, mass=0.0)
        for i, f in enumerate(bb.get("collision", [])):
            nm = "col_%s_%03d" % (bn, i)
            spec.add_mesh(name=nm, file=os.path.join(kd, f))
            body.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=nm, group=3, rgba=[0.9, 0.3, 0.3, 0.3], mass=0.0)
        if bb.get("mass_kg", 0.0) > 0 and bn != "base":
            body.mass = float(bb["mass_kg"])
            body.ipos = list(bb["com_m"])
            body.iquat = list(bb["inertia_quat"])
            body.inertia = [max(float(v), 1e-9) for v in bb["inertia_diag"]]
            body.explicitinertial = True

    def add_children(parent_body, pn):
        for cn in children.get(pn, []):
            cb = bodies[cn]
            Trel = np.linalg.inv(np.array(bodies[pn]["T_design"])) @ np.array(cb["T_design"])
            body = parent_body.add_body(name=cn, pos=Trel[:3, 3].tolist(), quat=quat_wxyz(Trel[:3, :3]).tolist())
            js = cb.get("joint")
            if js:
                h = kit["hinge"]
                jnt = body.add_joint(name="hinge_" + cn, type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(js["axis"]), pos=list(js["anchor"]),
                                     frictionloss=float(h["frictionloss"]), damping=float(h["damping"]), armature=float(h["armature"]))
                jnt.limited = mujoco.mjtLimited.mjLIMITED_FALSE
            add_geoms(body, cn)
            add_children(body, cn)

    Tb = Tr @ np.array(bodies["base"]["T_design"])
    base = w.add_body(name="base", pos=Tb[:3, 3].tolist(), quat=quat_wxyz(Tb[:3, :3]).tolist())
    add_geoms(base, "base")
    add_children(base, "base")
    for u in kit["loose"]:
        pos, quat = layout["parts"][u]
        body = w.add_body(name=u, pos=list(pos), quat=list(quat))
        body.add_freejoint(name="free_" + u)
        add_geoms(body, u)
        add_children(body, u)
    if robot_hook is None and cfg.get("robot"):
        from .ur_robot import add_ur
        robot_hook = lambda sp, c: add_ur(sp, c["robot"])        # noqa: E731
    robot = robot_hook(spec, cfg) if robot_hook is not None else {}
    # the welds: one per (slot, candidate), inactive
    for s in kit["slots"]:
        for c in s["candidates"]:
            eq = spec.add_equality(name=weld_name(s["id"], c["unit"]), type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                   name1=s["carrier"], name2=c["part"])
            eq.active = False
            eq.solref = list(cfg["weld_solref"])
            eq.solimp = list(cfg["weld_solimp"])
    for b1, b2 in kit.get("excludes", []):
        ex = spec.add_exclude()
        ex.bodyname1, ex.bodyname2 = b1, b2
    # the base and the stand: the body rests on the forks (it is bolted: no contact pair needed)
    ex = spec.add_exclude()
    ex.bodyname1, ex.bodyname2 = "base", "stand"
    return spec, robot


class QuadCell(object):
    def __init__(self, kit_dir, layout, config=None, log=print, robot_hook=None, spec_hook=None, settle_s=1.0, robot_init=None):
        import mujoco
        self.mj, self.np, self.log = mujoco, np, log
        self.kit = load_kit(kit_dir)
        self.cfg = copy.deepcopy(CELL_CONFIG)
        for k, v in (config or {}).items():
            if isinstance(v, dict) and isinstance(self.cfg.get(k), dict):
                self.cfg[k].update(v)
            else:
                self.cfg[k] = v
        self.layout = layout
        t0 = time.time()
        spec, robot = build_spec(self.kit, layout, self.cfg, robot_hook=robot_hook)
        if spec_hook is not None:
            spec_hook(spec)
        m = spec.compile()
        self.spec, self.m = spec, m
        d = mujoco.MjData(m)
        self.d = d
        self.robot = robot
        self.loose = list(self.kit["loose"])
        self.tick_steps = int(self.cfg["tick_steps"])
        self.tick_s = m.opt.timestep * self.tick_steps
        self.T_robot = T_world_robot(layout["stand"], self.cfg)
        for rb, ang in (layout.get("rotor_rad") or {}).items():
            d.qpos[m.jnt_qposadr[m.joint("hinge_" + rb).id]] = float(ang)
        mujoco.mj_forward(m, d)
        if robot_init is None and self.robot:
            robot_init = _ur_init
        self.ctrls = robot_init(self) if robot_init is not None else {}
        self.sm = CouplingStateMachine(m, d, self.kit)
        self.slots = {s["id"]: s for s in self.kit["slots"]}
        unit_of = {bn: bb["unit"] for bn, bb in self.kit["bodies"].items()}
        self.unit_of_body = {b: unit_of[m.body(b).name] for b in range(m.nbody) if m.body(b).name in unit_of}
        self.geoms_of = {u: [] for u in self.loose + ["base"]}
        for g in range(m.ngeom):
            u = self.unit_of_body.get(m.geom_bodyid[g])
            if u in self.geoms_of and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.geoms_of[u].append(g)
        # contact clusters (CELL_CONFIG["clusters"]): every rigidly fastened group of units (the body with what is
        # fastened to it; a knee made on the bench) gets a bit of its own: contype = its bit, conaffinity = every bit
        # but its own, so nothing inside a cluster collides (welded parts cannot move against each other; the hinges
        # inside a cluster are covered by the HAA range set at fastening and by the carve's joint sweeps) while every
        # cluster still collides with the rest (contype 1, conaffinity all: the bench, the stand, the UR, loose units)
        self.col_bits = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(m.ngeom)}
        self.n_fastened = -1
        self.clusters = []
        self.rec_t, self.rec_q, self.rec_s = [], [], []
        self.label = {"step": 0, "text": "start", "part": None, "slot": None}
        self.n_tick = 0
        self.hooks = []
        self.recording = True
        self.jig = {}
        self.ctrl = next(iter(self.ctrls.values()), None)
        rec, self.recording = getattr(self, "recording", True), False
        for _ in range(int(round(settle_s / self.tick_s))):
            self.tick_one_physics()
        self.recording = rec
        mujoco.mj_forward(m, d)
        d.time = 0.0
        self.initial = {u: self.body_T(u) for u in self.loose}
        log("[cell] quadruped cell built and settled in %.1f s (%d bodies, %d geoms)" % (time.time() - t0, m.nbody, m.ngeom))

    # ------------------------------------------------------------------ helpers
    def tick_one_physics(self):
        """one control tick of the controllers and physics only (the settling before t = 0: no coupling, no hooks)"""
        for ct in self.ctrls.values():
            ct.tick()
        stepper = [ct for ct in self.ctrls.values() if hasattr(ct, "step_physics")]
        if stepper:
            stepper[0].step_physics(self.tick_steps)
        else:
            for _s in range(self.tick_steps):
                self.mj.mj_step(self.m, self.d)

    def body_T(self, name):
        return body_pose(self.d, self.m.body(name).id)

    def save_trajectory(self, out_dir):
        """<out_dir>/trajectory.npz: t, q (qpos every 4 ticks), and per frame the label (the step in work) and the
        score, for the review videos"""
        labels = np.array([r["label"].get("text", "") for r in self.rec_s])
        score = np.array([r["score"] for r in self.rec_s], np.float32)
        np.savez_compressed(os.path.join(out_dir, "trajectory.npz"), t=np.array(self.rec_t), q=np.array(self.rec_q),
                            label=labels, score=score)

    def save_model(self, out_dir):
        """<out_dir>/cell.xml from the spec (mj_saveLastXML has no XML to save for a model compiled from a spec): the
        renderers and the true-solid check replay qpos through its kinematics"""
        open(os.path.join(out_dir, "cell.xml"), "w").write(self.spec.to_xml())

    def record(self):
        self.rec_t.append(float(self.d.time))
        self.rec_q.append(np.array(self.d.qpos, np.float32))
        self.rec_s.append({"states": dict(self.sm.state), "score": self.sm.score(), "label": dict(self.label)})

    def tick(self, n=1):
        for _ in range(n):
            self.tick_one()

    def tick_one(self):
        m, d = self.m, self.d
        for ct in self.ctrls.values():
            ct.tick()
        for name, (k, T) in self.jig.items():
            d.mocap_pos[k] = T[:3, 3]
            d.mocap_quat[k] = quat_wxyz(T[:3, :3])
        stepper = [ct for ct in self.ctrls.values() if hasattr(ct, "step_physics")]
        if stepper:
            # the UR's computed torque is recomputed before every physics step (mj_step1 / torque / mj_step2)
            stepper[0].step_physics(self.tick_steps)
        else:
            for _s in range(self.tick_steps):
                self.mj.mj_step(m, d)
        self.mj.mj_forward(m, d)
        self.sm.tick(float(d.time))
        if self.cfg.get("clusters", True):
            nf = sum(1 for x in self.sm.rec.values() if x["fastened"])
            if nf != self.n_fastened:
                self.update_clusters()
        self.n_tick += 1
        for h in self.hooks:
            h()
        if self.recording and self.n_tick % 4 == 0:
            self.record()

    def update_clusters(self):
        """the contact clusters from the fastened slots (see __init__); the geoms' bits in col_bits (what a jig's
        collision toggle restores)"""
        m = self.m
        parent = {u: u for u in self.geoms_of}

        def root(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        for sid, st in self.sm.rec.items():
            if st["fastened"] and st["unit"] is not None:
                cu = self.kit["bodies"][self.slots[sid]["carrier"]]["unit"]
                parent[root(st["unit"])] = root(cu)
        groups = {}
        for u in self.geoms_of:
            groups.setdefault(root(u), []).append(u)
        clusters = sorted([sorted(v) for v in groups.values() if len(v) > 1], key=lambda v: (0 if "base" in v else 1, v))
        allbits = (1 << 31) - 1
        bits = {}
        for i, cl in enumerate(clusters[:30]):
            b = 1 << (i + 1)
            for u in cl:
                for g in self.geoms_of[u]:
                    bits[g] = (b, allbits & ~b)
        for u, gs in self.geoms_of.items():
            for g in gs:
                ct, ca = bits.get(g, (1, allbits))
                self.col_bits[g] = (ct, ca)
                if m.geom_contype[g] or m.geom_conaffinity[g]:          # a jig may have switched it off in flight
                    m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
        self.clusters = clusters
        self.n_fastened = sum(1 for x in self.sm.rec.values() if x["fastened"])

    # ------------------------------------------------------------------ targets
    def candidate(self, sid, unit):
        s = self.slots[sid]
        return [c for c in s["candidates"] if c["unit"] == unit][0]

    def live_target(self, sid, unit):
        """{"T": the unit's root pose for the slot at the design from its carrier as it is now, "carrier", "F" the
        slot's frame in the world (socket face / seat), "insertion" (world, unit)}"""
        s = self.slots[sid]
        c = self.candidate(sid, unit)
        Tc = self.body_T(s["carrier"])
        Tp = Tc @ np.array(c["T_part_in_carrier"], float)
        F = Tc @ np.array(s.get("frame_in_carrier", np.eye(4)), float)
        if s["kind"] == "keyed":
            ins = F[:3, 2]
        else:
            ins = self.T_robot[:3, :3] @ np.array(s.get("insertion_robot_design") or s["insertion_robot"], float)
        return {"T": Tp, "carrier": s["carrier"], "F": F, "insertion": ins / np.linalg.norm(ins)}

    def live_target_moving(self, sid, moving, part=None):
        """the home pose of the unit that MOVES for the slot, from the other side as it is now: `moving` a candidate
        of the slot -> `live_target`; `moving` the carrier's unit (the upper leg lowered onto a lower leg lying on the
        bench) -> its root pose that puts its carrier's socket on `part` (default: the slot's designed candidate) at
        the design, the carrier's hinge as it is now. {"T", "insertion": the moving unit's direction of travel home,
        "part", "F": the socket frame at home (world)}"""
        s = self.slots[sid]
        units = [c["unit"] for c in s["candidates"]]
        if moving in units:
            tg = self.live_target(sid, moving)
            tg["part"] = moving
            return tg
        carrier_unit = self.kit["bodies"][s["carrier"]]["unit"]
        assert moving == carrier_unit, (sid, moving)
        if part is None:
            from .kitpose import designed_candidates
            part = designed_candidates(self.kit)[sid]["unit"]
        c = self.candidate(sid, part)
        T_part = self.body_T(part)
        T_c = T_part @ _inv(np.array(c["T_part_in_carrier"], float))         # the carrier where the part sits home
        T_c_in_u = _inv(self.body_T(moving)) @ self.body_T(s["carrier"])     # the hinge as it is now
        F = T_c @ np.array(s.get("frame_in_carrier", np.eye(4)), float)
        return {"T": T_c @ _inv(T_c_in_u), "carrier": s["carrier"], "F": F, "insertion": -F[:3, 2] / np.linalg.norm(F[:3, 2]),
                "part": part}

    def design_world(self):
        """{body: world pose} of the robot assembled at its design (every unit at its designed slot)"""
        from .kitpose import design_poses
        T, _ = design_poses(self.kit)
        return {b: self.T_robot @ Tb for b, Tb in T.items()}


def _ur_init(cell):
    """the UR's controller at the home posture (joint mode), the gripper open"""
    from .ur_controller import URController
    m, d = cell.m, cell.d
    rb = cell.robot
    q0 = np.array(cell.cfg["robot_home_q"], float)
    for jn, q in zip(rb["joints"], q0):
        d.qpos[m.jnt_qposadr[m.joint(jn).id]] = q
    cell.mj.mj_forward(m, d)
    ct = URController(m, d, tick_s=cell.tick_s, prefix=rb["prefix"], tcp_site=rb["tcp_site"])
    ct.q_cmd = q0.copy()
    ct.q_goal = q0.copy()
    ct.command_width(ct.max_width)
    ct.width_cmd = ct.width_goal = ct.max_width
    ct.write_ctrl()
    return {"ur": ct}
