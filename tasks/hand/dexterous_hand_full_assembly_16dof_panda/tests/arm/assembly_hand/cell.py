"""The hand assembly cell: the precedent's bench and Panda (the SO-101 cell's bytes as the tensegrity cell carries them:
the 1.6 x 1.0 m bench top at z = 0, the Panda at (-0.45, 0, 0) with its tcp 0.1034 m from the hand, the flange F/T
site, the pads as the cell's contact, the gripper's own position actuator), the raised fixture (fixture.py) with the
palm's bottom shell fixed on it, the two trays (trays.py) with every loose unit a free, articulated body resting in
its nest, and the coupling (coupling.py).

`HandCell(gt_dir, panda_dir, layout, config=None, ...)` compiles from the hidden kit (`kit.py`) and the episode's
layout (layout.py):
  * every loose unit through `kit.add_unit`: its links, the product joints as simulated (gt/joints.json: the stops of
    the true solids, the free range until the coupling sets the mounted range, the servo's 0.02 N m backdrive
    friction, damping and armature, the passive pins, the four-bar's loop), its collision pieces and inertia;
    the chains start at the start pose with the layout's drawn ab/ad and side angles;
  * the palm base a fixed child of the fixture at the palm frame (palmar face 150 mm above the bench);
  * the coupling's rails, welds and tool capsules (coupling.declare) before compiling, the machine after;
  * optional mocap jigs (`jigs=True`: development and QC hooks only, never in an episode).
`tick()` advances one 10 ms control tick (controller, jigs, 10 physics steps, coupling, hooks).
No position lock on any product joint; nothing holds a loose unit but its nest, the gripper and the coupling.
"""
import copy
import math
import os
import time

import numpy as np

from . import coupling as cp
from . import fixture as FX
from . import layout as LY
from . import trays as TR
from .kit import HandKit, add_unit

CELL_CONFIG = {
    # the contact model of the floating check (after the tensegrity and quadruped cells): 1 ms, 10 steps per tick,
    # implicitfast, elliptic cones, impratio 100, noslip 5, multiccd off
    "timestep": 0.001, "tick_steps": 10, "impratio": 100.0, "noslip_iterations": 5, "arena_memory_bytes": 1536 << 20,
    "geom_defaults": {"condim": 4, "friction": [0.8, 0.005, 0.0001], "solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
    "bench": {"size_xy": [1.6, 1.0], "thickness": 0.03, "floor_z": -0.994, "friction": 0.8},
    "panda_pos": [-0.45, 0.0, 0.0], "panda_gravcomp": 1.0, "tcp_offset_m": 0.1034,
    "gripper": {"kp": 2500.0, "kv": 50.0, "max_force_N": 60.0, "max_width_m": 0.08},
    "pads": {"profile": "flat", "condim": 4, "solref": [0.002, 1.0], "friction": [1.0, 0.005, 0.0001]},
    # the nests: stiffer than the parts (workcell plates; contacts with them use the plates' solref, priority 1) and
    # with the joint friction's impedance: with the parts' default a finger's ab/ad crept 0.15-0.28 deg/min in its nest
    # (MuJoCo's contact friction under a steady sideways load), with this one 0.04-0.16 deg/min
    "nest_contact": {"solref": [0.002, 1.0], "priority": 1, "solimp": [0.99, 0.999, 0.001, 0.5, 2.0]},
    # the fixture's rest deck: stiffer than the parts, as the nests, with the impedance the joints' friction had
    # before it was solved hard (0.99-0.999): with the deck as hard as the joints a fastened finger lying on it never
    # came to rest (its MCP drifted 0.05 deg in 3 s)
    "support_contact": {"solref": [0.002, 1.0], "priority": 1, "solimp": [0.99, 0.999, 0.001, 0.5, 2.0]},
    # the thumb's pad: solved as hard as the joints' friction: MuJoCo's soft friction slips at a speed proportional
    # to its load, and the mounted thumb, its yoke on the pad (a closed loop through its roll joint), turned its side
    # joint by 0.0075 deg/s with the parts' default impedance, 0.0017 deg/s with 0.99-0.999 on the pad and the joints,
    # under 0.00005 deg/s with 0.9999-0.99999 on both
    "pad_contact": {"solref": [0.002, 1.0], "priority": 1, "solimp": [0.9999, 0.99999, 0.001, 0.5, 2.0]},
    "robot_home_q": [0.0, -0.4, 0.0, -2.3, 0.0, 1.9, 0.785],
    # the arm's seven position targets are ramped linearly over the tick's 10 physics steps (a real arm's 1 kHz
    # interpolation of its 100 Hz commands); held for the whole tick, as the precedent cells do, the steps shook a
    # carried unit at 5-13 m/s2 and turned the thumb's friction-held joints in a carry
    "ctrl_interp": True,
    "materials": {"fixture": [0.78, 0.79, 0.80, 1.0], "tray": [0.80, 0.81, 0.82, 1.0], "nest": [0.70, 0.72, 0.74, 1.0],
                  "bench": [0.82, 0.74, 0.62, 1.0]},
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


def add_panda(spec, panda_dir, cfg):
    """the Panda as the precedent cells build it: attached with the prefix "panda/", gravity compensated links, the tcp
    and F/T sites, the force / torque sensors, the pads renamed, the gripper actuator; returns {"pad_geoms": [...]}"""
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
            if pc.get("solimp"):
                g.solimp = list(pc["solimp"])
            pad_geoms.append(g.name)
        elif (b is not None and b.name in ("panda/left_finger", "panda/right_finger") and g.type == mujoco.mjtGeom.mjGEOM_MESH
              and (g.contype or g.conaffinity)):
            # the finger's own collision mesh (every face of the pad bodies is a pad): the pads' contact stiffness.
            # Its tip stands 0.85 mm beyond the pad boxes, so the closed fingertips push a part with it; at the stock
            # time constant (0.02 s) they sank 0.65 mm into the cap's pieces under the fit's 4 N push
            g.solref = list(pc["solref"])
    gr = cfg["gripper"]
    for a in spec.actuators:
        if a.name == "panda/actuator8":
            a.name = "panda/gripper"
            a.gainprm = [gr["kp"], 0, 0, 0, 0, 0, 0, 0, 0, 0]
            a.biasprm = [0, -gr["kp"], -gr["kv"], 0, 0, 0, 0, 0, 0, 0]
            a.ctrlrange = [0.0, gr["max_width_m"] / 2]
            a.forcerange = [-gr["max_force_N"], gr["max_force_N"]]
    return {"pad_geoms": pad_geoms}


def _add_pieces(spec, body, rows, prefix, material, mesh_prefix, contact=None):
    import mujoco
    names = []
    for row in rows:
        nm = row[0]
        if row[1] == "box":
            g = body.add_geom(name=prefix + nm, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(row[3]), pos=list(row[2]), quat=list(row[4]),
                              material=material)
        else:
            mn = mesh_prefix + nm
            ms = spec.add_mesh(name=mn)
            ms.uservert = np.asarray(row[2], float).flatten().tolist()
            g = body.add_geom(name=prefix + nm, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mn, material=material)
        for ct in (contact if isinstance(contact, (list, tuple)) else [contact]):
            if ct and (ct.get("only") is None or any(nm.startswith(p) for p in ct["only"])):
                g.solref = list(ct["solref"])
                g.priority = int(ct["priority"])
                if ct.get("solimp"):
                    g.solimp = list(ct["solimp"])
                break
        names.append(g.name)
    return names


def build_spec(kit, layout, cfg, units, panda_dir=None, jigs=False, cache=None):
    import mujoco
    spec = mujoco.MjSpec()
    spec.modelname = "hand_assembly_cell"
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
    spec.stat.extent, spec.stat.center = 1.0, [0.0, 0.0, 0.15]
    w = spec.worldbody
    w.add_light(name="sun", pos=[0.3, -0.3, 2.0], dir=[-0.15, 0.15, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    w.add_light(name="fill", pos=[-0.5, 0.8, 1.2], dir=[0.4, -0.6, -0.8], diffuse=[0.35, 0.35, 0.35])
    for name, rgba in cfg["materials"].items():
        spec.add_material(name="mat_" + name, rgba=list(rgba))
    b = cfg["bench"]
    w.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[b["size_xy"][0] / 2, b["size_xy"][1] / 2, b["thickness"] / 2],
               pos=[0.0, 0.0, -b["thickness"] / 2], material="mat_bench")
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3.0, 3.0, 0.1], pos=[0.0, 0.0, b["floor_z"]], rgba=[0.3, 0.3, 0.3, 1.0])
    # the fixture with the palm base
    Tf = LY.fixture_T(layout["fixture"])
    fx = w.add_body(name="fixture", pos=Tf[:3, 3].tolist(), quat=cp.quat_wxyz(Tf[:3, :3]).tolist())
    _add_pieces(spec, fx, FX.fixture_geoms(cfg["fixture"]), "fixture_", "mat_fixture", "fixmesh_",
                contact=[dict(cfg["pad_contact"], only=("thumb_pad",)), dict(cfg["support_contact"], only=("deck_",))])
    mesh_cache = {}
    Tp = FX.palm_T_in_fixture(cfg["fixture"])
    pb_unit = [u for u in kit.units if kit.kind_of(u) == "PB"][0]
    pb = add_unit(spec, fx, kit, pb_unit, pose=Tp, free=False, mesh_cache=mesh_cache)
    base_body = pb["links"]["body"]
    # the trays
    trays = TR.trays_of(cfg["tray"])
    turned = layout.get("turned", {})
    for tn, tp in layout["trays"].items():
        Tt = LY.tray_T(tp, trays[tn]["plate"][2])
        tb = w.add_body(name=tn, pos=Tt[:3, 3].tolist(), quat=cp.quat_wxyz(Tt[:3, :3]).tolist())
        rows = TR.tray_geoms(kit, cfg["tray"], tn, cache=cache, turned=turned)
        plate = [r for r in rows if r[0] == "plate"]
        nests = [r for r in rows if r[0] != "plate"]
        _add_pieces(spec, tb, plate, tn + "_", "mat_tray", tn + "_m_")
        _add_pieces(spec, tb, nests, tn + "_", "mat_nest", tn + "_m_", contact=cfg["nest_contact"])
    # the loose units in their nests
    poses, _joints = LY.unit_poses(layout, kit, cfg, cache=cache)
    info = {pb_unit: pb}
    for u in units:
        info[u] = add_unit(spec, w, kit, u, pose=poses[u], free=True, mesh_cache=mesh_cache)
        if jigs:
            jb = w.add_body(name="jig_" + u, mocap=True, pos=poses[u][:3, 3].tolist())
            jb.add_site(name="jig_site_" + u, size=[0.002] * 3, group=5)
    robot = add_panda(spec, panda_dir, cfg) if panda_dir else {"pad_geoms": []}
    unit_bodies = {u: list(info[u]["links"].values())[0] for u in units}
    cp.declare(spec, kit, base_body, unit_bodies)
    if jigs:
        for u in units:
            eq = spec.add_equality(name="jig_weld_" + u, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                   name1="jig_" + u, name2=unit_bodies[u])
            eq.solref = [0.004, 1.0]
            eq.active = False
    return spec, robot, info, base_body, unit_bodies, pb_unit


class HandCell(object):
    def __init__(self, gt_dir, panda_dir, layout, config=None, units=None, log=print, settle_s=1.0, jigs=False, kit=None,
                 robot_q=None, coupling_cfg=None):
        import mujoco
        t0 = time.time()
        self.mj, self.log = mujoco, log
        self.cfg = _merge(CELL_CONFIG, config)
        self.layout = layout
        self.kit = kit or HandKit(gt_dir)
        self.gt_dir, self.panda_dir = gt_dir, panda_dir
        all_loose = [u for u in self.kit.units if self.kit.kind_of(u) != "PB"]
        self.loose = list(units or all_loose)
        self.cache = {k: TR.UnitGeom(self.kit, k) for k in ("FM", "TM", "BS", "PT", "PC", "HB", "CC")}
        spec, robot, info, base_body, unit_bodies, pb_unit = build_spec(self.kit, layout, self.cfg, self.loose, panda_dir=panda_dir,
                                                                        jigs=jigs, cache=self.cache)
        self.spec, self.robot, self.info = spec, robot, info
        self.base_body, self.unit_bodies, self.pb_unit = base_body, unit_bodies, pb_unit
        m = spec.compile()
        d = mujoco.MjData(m)
        self.m, self.d = m, d
        self.tick_steps = int(self.cfg["tick_steps"])
        self.tick_s = m.opt.timestep * self.tick_steps
        # the drawn chain angles (the nest holds the chain where it was laid)
        _poses, joints = LY.unit_poses(layout, self.kit, self.cfg, cache=self.cache)
        for u, js in joints.items():
            if u not in self.loose:
                continue
            for jn, v in js.items():
                d.qpos[m.jnt_qposadr[m.joint(info[u]["joints"][jn]).id]] = v
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
        unit_geoms = {u: list(info[u]["geoms"]) for u in self.loose}
        unit_joints = {u: dict(info[u]["joints"]) for u in self.loose}
        self.sm = cp.HandCoupling(m, d, self.kit, base_body, unit_bodies, unit_geoms, unit_joints, cfg=coupling_cfg)
        self.geoms_of = {u: [m.geom(g).id for g in unit_geoms[u]] for u in self.loose}
        self.bodies_of = {u: [m.body(b).id for b in info[u]["links"].values()] for u in self.loose}
        self.body_unit = {b: u for u, bs in self.bodies_of.items() for b in bs}
        self.joint_ids = {u: {jn: m.joint(n).id for jn, n in info[u]["joints"].items()} for u in self.loose}
        self.free_jnt = {u: m.joint("%s__free" % u).id for u in self.loose}
        self.jig = {}
        self.jig_mocap, self.jig_eq = {}, {}
        if jigs:
            self.jig_mocap = {u: m.body_mocapid[m.body("jig_" + u).id] for u in self.loose}
            self.jig_eq = {u: m.equality("jig_weld_" + u).id for u in self.loose}
        self.hooks = []
        self.n_tick = 0
        self.recording = False
        self.rec_t, self.rec_q = [], []
        self.planned = dict(_poses)
        self.q_start = self.joint_q_all()
        for _ in range(int(round(float(settle_s) / self.tick_s))):
            self.tick_one_physics()
        mujoco.mj_forward(m, d)
        d.time = 0.0
        self.initial = {u: self.unit_T(u) for u in self.loose}
        self.build_s = time.time() - t0
        log("[cell] hand cell built and settled in %.1f s (%d bodies, %d geoms, %d meshes)" % (self.build_s, m.nbody, m.ngeom, m.nmesh))

    # ------------------------------------------------------------------ helpers
    def body_T(self, name_or_id):
        b = name_or_id if isinstance(name_or_id, (int, np.integer)) else self.m.body(name_or_id).id
        return cp.body_pose(self.d, b)

    def unit_T(self, u):
        """the unit's root link pose (world, m); for the palm base its body"""
        if u == self.pb_unit:
            return self.body_T(self.base_body)
        return self.body_T(self.unit_bodies[u])

    def link_T(self, u, link):
        return self.body_T(self.info[u]["links"][link])

    def palm_T(self):
        return self.body_T(self.base_body)

    def joint_q(self, u, jn):
        return float(self.d.qpos[self.m.jnt_qposadr[self.joint_ids[u][jn]]])

    def joint_q_all(self):
        return {u: {jn: self.joint_q(u, jn) for jn in js} for u, js in self.joint_ids.items()}

    def _arm_ctrl_ramp(self):
        """the arm actuators' targets for each physics step of this tick: from the last tick's targets to the
        controller's new ones (None when interpolation is off or there is no arm)"""
        if self.ctrl is None or not self.cfg.get("ctrl_interp", True):
            return None
        acts = list(self.ctrl.acts)
        new = np.array([self.d.ctrl[a] for a in acts], float)
        old = self._ctrl_prev if getattr(self, "_ctrl_prev", None) is not None else new
        self._ctrl_prev = new.copy()
        return acts, old, new

    def _physics_steps(self):
        ramp = self._arm_ctrl_ramp()
        n = self.tick_steps
        for k in range(n):
            if ramp is not None:
                acts, old, new = ramp
                f = (k + 1.0) / n
                for i, a in enumerate(acts):
                    self.d.ctrl[a] = old[i] + f * (new[i] - old[i])
            self.mj.mj_step(self.m, self.d)

    def tick_one_physics(self):
        if self.ctrl is not None:
            self.ctrl.tick()
        self._physics_steps()

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
        self._physics_steps()
        self.mj.mj_forward(m, d)
        self.sm.tick(float(d.time))
        self.n_tick += 1
        for h in self.hooks:
            h()
        if self.recording and self.n_tick % 4 == 0:
            self.rec_t.append(float(d.time))
            self.rec_q.append(np.array(d.qpos, np.float32))

    # jigs (development and QC only)
    def grab_jig(self, u, T=None):
        m, d = self.m, self.d
        T = self.unit_T(u) if T is None else T
        self.jig[u] = T
        k = self.jig_mocap[u]
        d.mocap_pos[k], d.mocap_quat[k] = T[:3, 3], cp.quat_wxyz(T[:3, :3])
        e = self.jig_eq[u]
        m.eq_data[e, :] = 0.0
        m.eq_data[e, 6] = 1.0
        m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1

    def release_jig(self, u):
        self.jig.pop(u, None)
        if u in self.jig_eq:
            self.d.eq_active[self.jig_eq[u]] = 0

    def set_unit_T(self, u, T):
        """teleport (QC setup only): the unit's root link to T (world), zero velocity, joints as they are"""
        m, d = self.m, self.d
        j = self.free_jnt[u]
        qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        d.qpos[qa:qa + 3] = T[:3, 3]
        d.qpos[qa + 3:qa + 7] = cp.quat_wxyz(T[:3, :3])
        d.qvel[va:va + 6] = 0.0
        self.mj.mj_forward(m, d)

    def lowest_point(self, units=None):
        m, d = self.m, self.d
        z = 1e9
        for u in (units or self.loose):
            for g in self.geoms_of[u]:
                k = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
                z = min(z, float((v @ d.geom_xmat[g].reshape(3, 3).T)[:, 2].min() + d.geom_xpos[g][2]))
        return z
