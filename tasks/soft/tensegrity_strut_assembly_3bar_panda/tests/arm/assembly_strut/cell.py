"""The strut assembly cell (scope default: the three-strut cell, three cradle seats, two trays, 15 loose units; scope
single, the minimum version: one seat, one tray, one strut; the same code): the precedent's bench and Panda (the
SO-101 cell's bytes: the 1.6 x 1.0 m bench top at z = 0, the Panda at (-0.45, 0, 0) with its tcp 0.1034 m from the
hand, the flange F/T site, the pads as the cell's contact, the gripper's own position actuator), the cradle
(fixture.py) with the strut cores (M0) fixed in their seats, the trays (trays.py) with every loose unit a free body
resting in its nest, the coupling (coupling.py).

`StrutCell(gt_dir, panda_dir, layout, config=None, ...)` compiles from the hidden kit (`kit.py`) and the episode's
layout (layout.py):
  * every loose unit a free body with its collision hulls and its explicit inertia (kit.json), named by its unit id;
    the visual meshes (the authors' triangles) are not in the physics model: the render twin adds them (render_twin.py);
  * the strut core a fixed child of the cradle at its seat, its rod's collision geom `rod_geom` (the hull from c022);
  * the coupling's rails and welds (coupling.declare) before compiling, the state machine after;
  * optional mocap jigs (`jigs=True`: the grader's own QC hooks, never in an episode: a weld from a mocap body to a
    unit, inactive until `grab_jig`).
`tick()` advances one 10 ms control tick (controller, jigs, 10 physics steps, coupling, hooks) and records a frame every
4 ticks when `recording`.
"""
import copy
import math
import os
import time

import numpy as np

from . import coupling as cp
from . import fixture as FX
from . import trays as TR
from .kit import StrutKit, add_module

CELL_CONFIG = {
    # the quadruped cell's contact model (the kit's floating check ran on it): 1 ms, 10 steps per tick,
    # implicitfast, elliptic cones, impratio 100, noslip 5, multiccd off
    "timestep": 0.001, "tick_steps": 10, "impratio": 100.0, "noslip_iterations": 5, "arena_memory_bytes": 768 << 20,
    "geom_defaults": {"condim": 4, "friction": [0.8, 0.005, 0.0001], "solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
    # the bench (SO-101's: 1.6 x 1.0 m, 30 mm, top at z = 0, the floor 0.994 m below)
    "bench": {"size_xy": [1.6, 1.0], "thickness": 0.03, "floor_z": -0.994, "friction": 0.8},
    # the Panda (SO-101's cell_builder): base, gravity compensation, tcp, gripper actuator
    "panda_pos": [-0.45, 0.0, 0.0], "panda_gravcomp": 1.0, "tcp_offset_m": 0.1034,
    "gripper": {"kp": 2500.0, "kv": 50.0, "max_force_N": 60.0, "max_width_m": 0.08},
    # the pads: the precedent's contact fix (condim 4, solref 0.002); "flat" = the stock pads, "v_groove" = a 2 mm
    # deep V groove along the hand's x on each pad (workcell equipment)
    "pads": {"profile": "flat", "condim": 4, "solref": [0.002, 1.0], "friction": [1.0, 0.005, 0.0001],
             "v_groove": {"depth_m": 0.002, "half_angle_deg": 60.0}},
    # the nests: stiffer than the parts (workcell plates): contacts with them use the plates' solref (priority 1)
    "nest_contact": {"solref": [0.002, 1.0], "priority": 1},
    "robot_home_q": [0.0, -0.4, 0.0, -2.3, 0.0, 1.9, 0.785],
    "materials": {"cradle": [0.30, 0.32, 0.35, 1.0], "jaw": [0.12, 0.12, 0.13, 1.0], "tray": [0.72, 0.74, 0.76, 1.0],
                  "nest": [0.20, 0.36, 0.62, 1.0], "bench": [0.82, 0.80, 0.76, 1.0]},
    "fixture": {},
    "tray": {},
}


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _T(pos, quat):
    import mujoco
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(quat, float))
    T = np.eye(4)
    T[:3, :3] = R.reshape(3, 3)
    T[:3, 3] = pos
    return T


def add_panda(spec, panda_dir, cfg):
    """the Panda as the SO-101 cell builds it (assembly_six.cell_builder): attached with the prefix "panda/", gravity
    compensated links, the tcp and F/T sites, the force / torque sensors, the pads renamed, the gripper actuator; returns
    {"pad_geoms": [...]}"""
    import mujoco
    panda = mujoco.MjSpec.from_file(os.path.join(panda_dir, "panda.xml"))
    for k in list(panda.keys):
        panda.delete(k)
    frame = spec.worldbody.add_frame(pos=list(cfg["panda_pos"]))
    frame.attach_body(panda.worldbody.first_body(), "panda/", "")
    for b in spec.bodies:
        if b.name.startswith("panda/") and b.name != "panda/link0":
            b.gravcomp = float(cfg["panda_gravcomp"])
    hand = spec.body("panda/hand")
    hand.add_site(name="tcp", pos=[0.0, 0.0, float(cfg["tcp_offset_m"])], size=[0.004, 0.004, 0.004], group=4, rgba=[0.2, 0.9, 0.2, 0.6])
    hand.add_site(name="hand_ft", pos=[0.0, 0.0, 0.0], size=[0.003, 0.003, 0.003], group=4)
    spec.add_sensor(name="hand_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="hand_ft")
    spec.add_sensor(name="hand_torque", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="hand_ft")
    pc = cfg["pads"]
    pad_geoms = []
    for g in spec.geoms:
        b = g.parent
        if b is not None and b.name in ("panda/left_finger", "panda/right_finger") and g.type == mujoco.mjtGeom.mjGEOM_BOX:
            side = "left" if "left" in b.name else "right"
            g.name = "panda/pad_%s_%d" % (side, len([p for p in pad_geoms if side in p]))
            g.friction = list(pc["friction"])
            g.condim = int(pc["condim"])
            g.solref = list(pc["solref"])
            pad_geoms.append(g.name)
    if pc.get("profile") == "v_groove":
        pad_geoms = _v_groove_pads(spec, pad_geoms, pc)
    gr = cfg["gripper"]
    for a in spec.actuators:
        if a.name == "panda/actuator8":
            a.name = "panda/gripper"
            a.gainprm = [gr["kp"], 0, 0, 0, 0, 0, 0, 0, 0, 0]
            a.biasprm = [0, -gr["kp"], -gr["kv"], 0, 0, 0, 0, 0, 0, 0]
            a.ctrlrange = [0.0, gr["max_width_m"] / 2]
            a.forcerange = [-gr["max_force_N"], gr["max_force_N"]]
    return {"pad_geoms": pad_geoms}


def _v_groove_pads(spec, pad_geoms, pc):
    """the shallow V-groove pads (workcell equipment): each finger's pad boxes replaced by two boxes whose
    faces form a V groove `depth_m` deep along the hand's x (the closing axis is the hand's y); the groove's flanks at
    `half_angle_deg` from the closing axis; returns the new pad geom names.
    NOT EXERCISED: the grasp trials passed with the stock flat pads for all three modules, so this fallback was never
    built into a cell or checked; whoever switches `pads.profile` to "v_groove" must check its geometry first."""
    import mujoco
    out = []
    for nm in list(pad_geoms):
        g = [x for x in spec.geoms if x.name == nm][0]
        g.contype, g.conaffinity = 0, 0
        g.name = nm + "_stock"
    for side in ("left", "right"):
        b = spec.body("panda/%s_finger" % side)
        stock = [x for x in spec.geoms if x.name.startswith("panda/pad_%s_" % side) and x.name.endswith("_stock")]
        P = np.array([list(x.pos) for x in stock])
        H = np.array([list(x.size) for x in stock])
        lo, hi = (P - H).min(0), (P + H).max(0)
        c = 0.5 * (lo + hi)
        # the pad face is the stock boxes' face towards the other finger: in the finger frame the closing axis is +-y
        sy = 1.0 if side == "left" else -1.0     # the left finger's pads face its -y? keep the stock faces' side
        face_y = lo[1] if np.mean(P[:, 1]) > 0 else hi[1]
        a = math.radians(float(pc["v_groove"]["half_angle_deg"]))
        dep = float(pc["v_groove"]["depth_m"])
        half_w = 0.5 * (hi[0] - lo[0])
        flank = dep / math.cos(a)
        del sy
        for k, sx in enumerate((-1.0, 1.0)):
            # a flank: tilted about the finger's z by (90 deg - a) so that its face makes the groove
            ang = sx * (math.pi / 2 - a)
            n_in = 1.0 if face_y == lo[1] else -1.0          # into the finger from the face
            t = 0.003
            cx = c[0] + sx * 0.5 * half_w
            cy = face_y + n_in * (0.5 * dep + t / 2)
            gg = b.add_geom(name="panda/pad_%s_v%d" % (side, k), type=mujoco.mjtGeom.mjGEOM_BOX,
                            size=[0.5 * half_w / math.sin(a) + 1e-4, t / 2, 0.5 * (hi[2] - lo[2])],
                            pos=[cx, cy, c[2]], quat=[math.cos(ang / 2), 0.0, 0.0, math.sin(ang / 2)])
            gg.friction = list(pc["friction"])
            gg.condim = int(pc["condim"])
            gg.solref = list(pc["solref"])
            out.append(gg.name)
        del flank
    return out


def scoped_kit(kit, units):
    """the kit as the coupling sees it in this cell: a slot's candidates are only the units the cell holds (the
    coupling's declare / StrutCoupling take every unit of a type from kit.units_of_type)"""
    sk = copy.copy(kit)
    present = set(units)
    sk.units_of_type = lambda t: [u for u in kit.units_of_type(t) if u in present]
    return sk


def build_spec(kit, layout, cfg, scope, units, panda_dir=None, jigs=False, visual=False):
    import mujoco
    spec = mujoco.MjSpec()
    spec.modelname = "strut_assembly_cell"
    spec.memory = int(cfg["arena_memory_bytes"])
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    o = spec.option
    o.timestep = float(cfg["timestep"])
    o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    o.noslip_iterations = int(cfg["noslip_iterations"])
    o.impratio = float(cfg["impratio"])
    o.gravity = [0.0, 0.0, -9.81]
    o.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    gd = spec.default.geom
    gd.condim = int(cfg["geom_defaults"]["condim"])
    fr = list(cfg["geom_defaults"]["friction"])
    fr[0] = float(cfg["bench"]["friction"])
    gd.friction = fr
    gd.solref = cfg["geom_defaults"]["solref"]
    gd.solimp = cfg["geom_defaults"]["solimp"]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.stat.extent, spec.stat.center = 1.0, [0.0, 0.1, 0.1]
    w = spec.worldbody
    w.add_light(name="sun", pos=[0.3, -0.3, 2.0], dir=[-0.15, 0.15, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    w.add_light(name="fill", pos=[-0.5, 0.8, 1.2], dir=[0.4, -0.6, -0.8], diffuse=[0.35, 0.35, 0.35])
    mats = cfg["materials"]
    for name, rgba in mats.items():
        spec.add_material(name="mat_" + name, rgba=list(rgba))
    spec.add_material(name="mat_table", rgba=list(mats["bench"]))
    b = cfg["bench"]
    w.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[b["size_xy"][0] / 2, b["size_xy"][1] / 2, b["thickness"] / 2],
               pos=[0.0, 0.0, -b["thickness"] / 2], material="mat_table")
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.1], pos=[0.0, 0.0, b["floor_z"]], rgba=[0.3, 0.3, 0.3, 1.0])
    # the cradle with the strut cores
    struts = sorted(set(cp.strut_of(m) for m in kit.mates(scope)))
    cxy = layout["fixture"]["xy"]
    cradle = w.add_body(name="cradle", pos=[float(cxy[0]), float(cxy[1]), 0.0])
    for nm, c, hs in FX.cradle_boxes(cfg["fixture"], struts):
        cradle.add_geom(name="cradle_" + nm, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(hs), pos=list(c),
                        material="mat_jaw" if ("jaw" in nm or "saddle" in nm) else "mat_cradle")
    mesh_cache = {}
    carriers = {}
    for s in struts:
        Tc = FX.core_pose_in_cradle(cfg["fixture"], s)
        body = cradle.add_body(name="%s_M0" % s, pos=Tc[:3, 3].tolist(), quat=cp.quat_wxyz(Tc[:3, :3]).tolist())
        add_module(spec, body, kit, "M0", "%s_M0" % s, visual=visual, mesh_cache=mesh_cache, inertial=False)
        carriers[s] = "%s_M0" % s
    # the trays (the scope's set: trays.trays_of)
    nc = cfg["nest_contact"]
    tray_set = TR.trays_of(cfg["tray"], scope)
    for tn, tp in layout["trays"].items():
        a = math.radians(float(tp["yaw_deg"]))
        tdef = tray_set[tn]
        th = float(tdef["plate"][2])
        tb = w.add_body(name=tn, pos=[float(tp["xy"][0]), float(tp["xy"][1]), th], quat=[math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])
        for nm, _kind, c, hs, q in TR.tray_geoms(cfg["tray"], tdef["nests"], tdef["plate"]):
            g = tb.add_geom(name="%s_%s" % (tn, nm), type=mujoco.mjtGeom.mjGEOM_BOX, size=list(hs), pos=list(c), quat=list(q),
                            material="mat_tray" if nm == "plate" else "mat_nest")
            if nm != "plate":
                g.solref = list(nc["solref"])
                g.priority = int(nc["priority"])
    # the loose units
    for u in units:
        pos, quat = layout["parts"][u]
        body = w.add_body(name=u, pos=list(pos), quat=list(quat))
        body.add_freejoint(name="free_" + u)
        add_module(spec, body, kit, kit.units[u]["build"], u, visual=visual, mesh_cache=mesh_cache)
        if jigs:
            jb = w.add_body(name="jig_" + u, mocap=True, pos=list(pos), quat=list(quat))
            jb.add_site(name="jig_site_" + u, size=[0.002] * 3, group=5)
            eq = spec.add_equality(name="jig_weld_" + u, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                   name1="jig_" + u, name2=u)
            eq.solref = [0.005, 1.0]
            eq.active = False
    robot = add_panda(spec, panda_dir, cfg) if panda_dir else {"pad_geoms": []}
    cp.declare(spec, kit, scope, carriers, None)
    return spec, robot, carriers, struts


class StrutCell(object):
    def __init__(self, gt_dir, panda_dir, layout, config=None, scope="single", units=None, log=print, settle_s=0.6, jigs=False,
                 visual=False, coupling_cfg=None, kit=None, robot_q=None):
        import mujoco
        t0 = time.time()
        self.mj, self.log = mujoco, log
        self.cfg = _merge(CELL_CONFIG, config)
        self.layout = layout
        self.loose = list(units or layout["parts"].keys())
        self.kit = scoped_kit(kit or StrutKit(gt_dir), self.loose)
        self.gt_dir, self.panda_dir = gt_dir, panda_dir
        self.scope = scope
        spec, robot, carriers, struts = build_spec(self.kit, layout, self.cfg, scope, self.loose, panda_dir=panda_dir, jigs=jigs, visual=visual)
        self.spec, self.robot, self.carriers, self.struts = spec, robot, carriers, struts
        m = spec.compile()
        d = mujoco.MjData(m)
        self.m, self.d = m, d
        self.tick_steps = int(self.cfg["tick_steps"])
        self.tick_s = m.opt.timestep * self.tick_steps
        rod_idx = [i for i, (_V, pid, _k) in enumerate(self.kit.collision_hulls("M0")) if pid == "c022"][0]
        self.rod_geoms = {s: m.geom("%s_M0_col%03d" % (s, rod_idx)).id for s in struts}
        self.ctrl = None
        if panda_dir:
            from ..assembly_cell.controller import PandaController
            q0 = np.array(robot_q if robot_q is not None else self.cfg["robot_home_q"], float)
            for i in range(7):
                d.qpos[m.jnt_qposadr[m.joint("panda/joint%d" % (i + 1)).id]] = q0[i]
            gr = self.cfg["gripper"]
            for fn in ("finger_joint1", "finger_joint2"):
                d.qpos[m.jnt_qposadr[m.joint("panda/" + fn).id]] = gr["max_width_m"] / 2
            mujoco.mj_forward(m, d)
            self.ctrl = PandaController(m, d, tick_s=self.tick_s)
            self.ctrl.q_cmd = q0.copy()
            self.ctrl.q_goal = q0.copy()
            self.ctrl.command_width(self.ctrl.max_width)
            self.ctrl.width_cmd = self.ctrl.width_goal = self.ctrl.max_width
            self.ctrl.write_ctrl()
        else:
            mujoco.mj_forward(m, d)
        self.sm = cp.StrutCoupling(m, d, self.kit, scope, carriers, coupling_cfg, rod_geoms=self.rod_geoms)
        self.geoms_of = {u: [] for u in self.loose}
        for g in range(m.ngeom):
            nm = m.body(int(m.geom_bodyid[g])).name
            if nm in self.geoms_of and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.geoms_of[nm].append(g)
        self.col_bits = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(m.ngeom)}
        self.jig = {}
        self.jig_mocap, self.jig_eq = {}, {}
        if jigs:
            self.jig_mocap = {u: m.body_mocapid[m.body("jig_" + u).id] for u in self.loose}
            self.jig_eq = {u: m.equality("jig_weld_" + u).id for u in self.loose}
        self.hooks = []
        self.rec_t, self.rec_q, self.rec_s = [], [], []
        self.label = {"step": 0, "text": "start", "part": None, "slot": None}
        self.n_tick = 0
        self.recording = False
        self.planned = {u: _T(*layout["parts"][u]) for u in self.loose}
        for _ in range(int(round(float(settle_s) / self.tick_s))):
            self.tick_one_physics()
        mujoco.mj_forward(m, d)
        d.time = 0.0
        self.initial = {u: self.body_T(u) for u in self.loose}
        self.build_s = time.time() - t0
        log("[cell] strut cell built and settled in %.1f s (%d bodies, %d geoms, %d meshes)" % (self.build_s, m.nbody, m.ngeom, m.nmesh))

    # ------------------------------------------------------------------ helpers
    def body_T(self, name):
        return cp.body_pose(self.d, self.m.body(name).id)

    def tick_one_physics(self):
        if self.ctrl is not None:
            self.ctrl.tick()
        for _s in range(self.tick_steps):
            self.mj.mj_step(self.m, self.d)

    def tick(self, n=1):
        for _ in range(n):
            self.tick_one()

    def tick_one(self):
        m, d = self.m, self.d
        if self.ctrl is not None:
            self.ctrl.tick()
        for u, T in self.jig.items():
            k = self.jig_mocap[u]
            d.mocap_pos[k] = T[:3, 3]
            d.mocap_quat[k] = cp.quat_wxyz(T[:3, :3])
        for _s in range(self.tick_steps):
            self.mj.mj_step(m, d)
        self.mj.mj_forward(m, d)
        self.sm.tick(float(d.time))
        self.n_tick += 1
        for h in self.hooks:
            h()
        if self.recording and self.n_tick % 4 == 0:
            self.record()

    def record(self):
        self.rec_t.append(float(self.d.time))
        self.rec_q.append(np.array(self.d.qpos, np.float32))
        self.rec_s.append({"states": dict(self.sm.state), "score": self.sm.score(), "label": dict(self.label)})

    def save_trajectory(self, out_dir):
        labels = np.array([r["label"].get("text", "") for r in self.rec_s])
        score = np.array([r["score"] for r in self.rec_s], np.float32)
        np.savez_compressed(os.path.join(out_dir, "trajectory.npz"), t=np.array(self.rec_t), q=np.array(self.rec_q), label=labels, score=score)

    # jigs (QC only)
    def grab_jig(self, u, T=None):
        m, d = self.m, self.d
        T = self.body_T(u) if T is None else T
        self.jig[u] = T
        k = self.jig_mocap[u]
        d.mocap_pos[k], d.mocap_quat[k] = T[:3, 3], cp.quat_wxyz(T[:3, :3])
        e = self.jig_eq[u]
        com = np.array(m.body_ipos[m.body(u).id], float)
        m.eq_data[e, 0:3] = com
        m.eq_data[e, 3:6] = com
        m.eq_data[e, 6:10] = [1, 0, 0, 0]
        m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1

    def release_jig(self, u):
        self.jig.pop(u, None)
        if u in self.jig_eq:
            self.d.eq_active[self.jig_eq[u]] = 0

    def collide(self, u, on):
        for g in self.geoms_of.get(u, []):
            self.m.geom_contype[g], self.m.geom_conaffinity[g] = self.col_bits[g] if on else (0, 0)

    # ------------------------------------------------------------------ targets
    def slot(self, sid):
        return self.sm.by_id[sid]

    def design_world(self, sid):
        x = self.slot(sid)
        return self.body_T(self.carriers[x["strut"]]) @ x["Tdes"]

    def sleeve_pose(self, sid, s, roll, flip=False):
        """the unit's world pose on the slot's rod axis: its design-first mouth s (m) in from the rod end, roll (rad)"""
        x = self.slot(sid)
        return self.sm.G_world(x, s, roll) @ self.sm.canonical(x, flip)

    def com_world(self, u):
        b = self.m.body(u).id
        return np.array(self.d.xipos[b], float)

    def lowest_point(self):
        """the lowest point of any loose unit's collision geometry (world z, m)"""
        m, d = self.m, self.d
        z = 1e9
        for u in self.loose:
            for g in self.geoms_of[u]:
                k = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
                z = min(z, float((v @ d.geom_xmat[g].reshape(3, 3)[2]).min() + d.geom_xpos[g][2]))
        return z
