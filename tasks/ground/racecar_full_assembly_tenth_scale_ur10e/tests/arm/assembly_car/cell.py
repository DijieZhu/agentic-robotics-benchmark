"""The race-car assembly cell: the rolling chassis clamped on the work stand, the deck assembly on its shelf, the
compute module and the LiDAR in the parts tray, the four wheels and the stick pack lying on the bench, the arm (a UR
with a 2F-85, `arm.assembly_quad.ur_robot`; optional for the floating check), the coupling state machine
(`coupling.py`) and the cell's watch (scrapping and the safety rules).

`CarCell(kit_dir, layout, config=None, log=print, robot_hook=None, spec_hook=None, settle_s=1.0, robot_init=None)`
compiles from any kit directory in the hidden kit's format (`racecar-asm-kit/1`, its fields in `check_kit`: kit.json and
its meshes, geoms in body frames) and an episode layout (`layout.episode_layout`):
  * the bench, the floor and the fixtures (`fixtures.py`: stand, shelf, tray as static box bodies at the layout's
    places; the datum boards are the render twin's, `fixtures.boards()`);
  * the chassis: a fixed body at the layout's car pose (`layout.T_world_car`), its static children (the four deck
    standoffs) and its four hubs on unpowered hinges (kit "hinges": dry friction with its stiff impedance, damping,
    armature; unlimited), every child placed by its T_design relative to its parent, each hub at the layout's start
    angle;
  * every loose unit a free body at the layout's pose (`free_<unit>`), every geom of a body as the kit describes it
    (box / cylinder / convex mesh, mass 0, the kit's contact overrides: the wheels' stops), explicit inertials;
    visual meshes (gt kits) as non-colliding geoms;
  * one inactive weld per (slot, candidate) (`weld_<slot>__<unit>`), the kit's body-pair exclusions (and the chassis
    against the stand it is bolted on);
  * the robot through `robot_hook(spec, cfg) -> dict` or cfg["robot"] (ur_robot.add_ur's cfg; None: no robot).
Physics as the earlier assembly cell: 1 ms implicitfast, elliptic cones, impratio 100, noslip 5, multi-CCD off; a
10 ms control tick of 10 physics steps (the UR's computed torque recomputed before each step).

Every control tick (`tick_one`): the controllers, the jigs (mocap), the physics, mj_forward, then the coupling's tick
(`sm.tick`), the contact clusters (no contacts inside a rigidly fastened group; `sm.collision_bits` is the clusters'
table, so a wheel stop switched back on gets the bits of the moment), then the watch (`watch_tick`):
  * scrapping: a loose unit whose centre of mass is below the bench top by more than `watch.scrap_below_bench_m` (off
    the bench) or moves faster than `watch.scrap_speed_m_s` is scrapped (`sm.scrap`: its capture released, it leaves
    every candidate list; a scrapped deck voids what is fastened on it); event "scrapped:<unit>:<reason>";
  * safety: a robot link other than the four pads pressing on any body (but the robot) above `watch.robot_force_N`,
    the normal forces of the link on that body summed over their contacts ("robot_contact:<link body>:<other
    body>:<N>N"; the earlier cell compared single contacts); a loose unit pressed on the environment or another unit above
    `watch.part_force_N` (the largest per (unit, other body) sum of the tick) for `watch.part_force_ticks`
    consecutive ticks ("part_force:<unit>:<other body>:<N>N"). `watch_summary()` has the maxima and the violations;
then the hooks and the recording (every 4 ticks).
`live_target(slot, unit)` gives a unit's home pose for a slot from its carrier as it is now; `design_world()` the
car assembled at its design.
"""
import copy
import json
import math
import os
import time

import numpy as np

from . import fixtures as FX
from . import layout as LY
from .coupling import CouplingStateMachine, _inv, _rotz, body_pose, quat_wxyz, weld_name
from .fixtures import BENCH_BOARD, TOOL_PLATE  # noqa: F401  (the surveyed boards: the cell's constants)

KIT_VERSION = "racecar-asm-kit/1"

CELL_CONFIG = {
    # the earlier assembly cell's physics (impratio 100: a held part's creep in the pads; noslip 5)
    "timestep": 0.001, "tick_steps": 10, "impratio": 100.0, "noslip_iterations": 5, "arena_memory_bytes": 512 << 20,
    "clusters": True,
    # the arm (None: no robot, e.g. the floating check): ur_robot.add_ur's cfg (model, pos, yaw_deg, models_dir, ...)
    "robot": None,
    # the arm's ready posture: turned away from the car (q1 = 0 points the arm along the cell's -x), the upper arm up,
    # the forearm level, the tool down: clear of every fixture and out of the overhead camera's view of the boards
    "robot_home_q": [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
    "bench": {"size_xy": [2.0, 1.4], "centre_xy": [0.45, 0.0], "thickness": 0.03, "floor_z": -0.994, "friction": 0.8},
    "geom_defaults": {"condim": 4, "friction": [0.8, 0.005, 0.0001], "solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
    "weld_solref": [0.002, 1.0], "weld_solimp": [0.99, 0.999, 0.001, 0.5, 2.0],
    # the coupling's constructor arguments (its defaults are the frozen values; the stop's clearance 0.1 mm)
    "coupling": {"stop_clear_m": 1e-4},
    # robot_force_N: the robot-contact limit (the grader passes grader_config.json safety.robot_force_N, the same number;
    # this default serves a cell built without the grader's configuration)
    "watch": {"enabled": True, "robot_force_N": 150.0, "part_force_N": 150.0, "part_force_ticks": 20,
              "scrap_speed_m_s": 3.0, "scrap_below_bench_m": 0.05, "scrap": True},
    # the kit's collision geoms in group 3 (hidden); True: group 1 with colours (a kit without visual meshes)
    "show_collision": False,
    "materials": {"bench": [0.80, 0.74, 0.62, 1.0], "stand": FX.MATERIAL["stand"], "shelf": FX.MATERIAL["shelf"],
                  "tray": FX.MATERIAL["tray"]},
}
# the colours of the kit's units where a kit has no visual meshes (show_collision) or names no material
UNIT_RGBA = {"base": [0.45, 0.45, 0.50, 1.0], "WHL": [0.12, 0.12, 0.12, 1.0], "BAT": [0.15, 0.35, 0.85, 1.0],
             "DCK": [0.85, 0.85, 0.80, 1.0], "CMP": [0.20, 0.55, 0.25, 1.0], "LDR": [0.85, 0.45, 0.10, 1.0],
             "stop": [0.9, 0.1, 0.1, 1.0]}
GEOM_TYPES = ("box", "cylinder", "mesh")


def load_kit(kit_dir):
    kit = json.load(open(os.path.join(kit_dir, "kit.json")))
    kit["dir"] = os.path.abspath(kit_dir)          # the saved cell.xml names its meshes by this path
    return kit


def check_kit(kit):
    """the fields the cell reads (kit format racecar-asm-kit/1), or ValueError"""
    if kit.get("kit_version") != KIT_VERSION:
        raise ValueError("kit: kit_version %r, this cell reads %s" % (kit.get("kit_version"), KIT_VERSION))
    B = kit["bodies"]
    for u in kit["loose"]:
        if u not in B or B[u].get("joint") != "free" or B[u].get("parent"):
            raise ValueError("kit: loose unit %s is not a free root body" % u)
    roots = [b for b, bb in B.items() if not bb.get("parent")]
    if "chassis" not in roots:
        raise ValueError("kit: no root body 'chassis'")
    for bn, bb in B.items():
        if bb.get("parent") and bb["parent"] not in B:
            raise ValueError("kit: %s's parent %s is missing" % (bn, bb["parent"]))
        j = bb.get("joint")
        if isinstance(j, dict) and bn not in (kit.get("hinges") or {}):
            raise ValueError("kit: hinged body %s has no kit['hinges'] entry" % bn)
        for g in bb.get("collision", []):
            if g.get("type") not in GEOM_TYPES:
                raise ValueError("kit: geom %s of %s has type %r" % (g.get("name"), bn, g.get("type")))
    return True


def _merge_cfg(config):
    cfg = copy.deepcopy(CELL_CONFIG)
    for k, v in (config or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def _rgba_of(kit, bn, key=None):
    mats = kit.get("materials") or {}
    if key is not None and key in mats:
        m = mats[key]
        return list(m["rgba"] if isinstance(m, dict) else m)
    t = kit["bodies"][bn].get("type", "base")
    return list(UNIT_RGBA.get(t, UNIT_RGBA["base"]))


def build_spec(kit, layout, cfg, robot_hook=None):
    """(MjSpec, robot names) of the cell (module docstring)"""
    import mujoco
    spec = mujoco.MjSpec()
    spec.modelname = "racecar_assembly_cell"
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
    fr = list(cfg["geom_defaults"]["friction"])
    fr[0] = float(cfg["bench"]["friction"])
    gd.friction = fr
    gd.solref = cfg["geom_defaults"]["solref"]
    gd.solimp = cfg["geom_defaults"]["solimp"]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.stat.extent, spec.stat.center = 1.8, [0.45, 0.0, 0.3]
    w = spec.worldbody
    w.add_light(name="sun", pos=[0.4, -0.4, 2.5], dir=[-0.15, 0.15, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    w.add_light(name="fill", pos=[-0.6, 0.8, 1.6], dir=[0.4, -0.6, -0.8], diffuse=[0.35, 0.35, 0.35])
    for name, rgba in cfg["materials"].items():
        spec.add_material(name="mat_" + name, rgba=list(rgba))
    b = cfg["bench"]
    w.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[b["size_xy"][0] / 2, b["size_xy"][1] / 2, b["thickness"] / 2],
               pos=[b["centre_xy"][0], b["centre_xy"][1], -b["thickness"] / 2], material="mat_bench")
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5.0, 5.0, 0.1], pos=[0.0, 0.0, b["floor_z"]], rgba=[0.3, 0.3, 0.3, 1.0])
    # the fixtures
    fixtures = layout.get("fixtures") or {}
    for fxn in FX.FIXTURES:
        pose = fixtures.get(fxn)
        if pose is None:
            nom = LY.DEFAULT_LAYOUT[fxn]
            pose = {"xy": nom["nominal_xy"], "yaw_deg": nom["yaw_deg"]}
        T = LY.T_world_fixture(pose)
        fb = w.add_body(name=fxn, pos=T[:3, 3].tolist(), quat=quat_wxyz(T[:3, :3]).tolist())
        for bn, c, h, _part in FX.fixture_boxes(fxn):
            fb.add_geom(name="%s_%s" % (fxn, bn), type=mujoco.mjtGeom.mjGEOM_BOX, size=h, pos=c, material="mat_" + fxn, group=1)
    # the kit
    kd = kit["dir"]
    B = kit["bodies"]
    show = bool(cfg.get("show_collision"))
    vis_mats = set()
    gtype = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "mesh": mujoco.mjtGeom.mjGEOM_MESH}
    children = {}
    for bn, bb in B.items():
        if bb.get("parent"):
            children.setdefault(bb["parent"], []).append(bn)

    def add_geoms(body, bn):
        bb = B[bn]
        for key, f in sorted((bb.get("visual") or {}).items()):
            mk = key.split("#")[0]
            mat = "mat_vis_" + mk
            if mat not in vis_mats:
                spec.add_material(name=mat, rgba=_rgba_of(kit, bn, mk))
                vis_mats.add(mat)
            nm = "vis_%s_%s" % (bn, key.replace("#", "_"))
            spec.add_mesh(name=nm, file=os.path.join(kd, f))
            body.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=nm, contype=0, conaffinity=0, group=2,
                          material=mat, mass=0.0)
        for g in bb.get("collision", []):
            kw = {"name": g["name"], "type": gtype[g["type"]], "pos": list(g.get("pos", [0, 0, 0])),
                  "quat": list(g.get("quat", [1, 0, 0, 0])), "mass": 0.0, "group": 1 if show else 3}
            if g["type"] == "mesh":
                spec.add_mesh(name="col_" + g["name"], file=os.path.join(kd, g["file"]))
                kw["meshname"] = "col_" + g["name"]
            elif g["type"] == "cylinder":
                kw["size"] = [float(g["size"][0]), float(g["size"][1]), 0.0]
            else:
                kw["size"] = [float(v) for v in g["size"]]
            kw["rgba"] = UNIT_RGBA["stop"] if g.get("role") == "stop" else _rgba_of(kit, bn)
            if not show:
                kw["rgba"] = [kw["rgba"][0], kw["rgba"][1], kw["rgba"][2], 0.4]
            geom = body.add_geom(**kw)
            ct = g.get("contact") or {}
            if "solref" in ct:
                geom.solref = [float(v) for v in ct["solref"]]
            if "solimp" in ct:
                geom.solimp = [float(v) for v in ct["solimp"]]
            if "priority" in ct:
                geom.priority = int(ct["priority"])
            if "friction" in ct:
                geom.friction = [float(v) for v in ct["friction"]]
        if not bb.get("static") and float(bb.get("mass_kg", 0.0)) > 0.0:
            body.mass = float(bb["mass_kg"])
            body.ipos = [float(v) for v in bb["com_m"]]
            body.iquat = [float(v) for v in bb["inertia_quat"]]
            body.inertia = [max(float(v), 1e-10) for v in bb["inertia_diag"]]
            body.explicitinertial = True

    def add_children(parent_body, pn):
        for cn in children.get(pn, []):
            cb = B[cn]
            Trel = _inv(np.asarray(B[pn]["T_design"], float)) @ np.asarray(cb["T_design"], float)
            body = parent_body.add_body(name=cn, pos=Trel[:3, 3].tolist(), quat=quat_wxyz(Trel[:3, :3]).tolist())
            js = cb.get("joint")
            if isinstance(js, dict):
                h = kit["hinges"][cn]
                jnt = body.add_joint(name=js["name"], type=mujoco.mjtJoint.mjJNT_HINGE, axis=[float(v) for v in js["axis"]],
                                     pos=[float(v) for v in js["anchor"]], frictionloss=float(h["frictionloss"]),
                                     damping=float(h["damping"]), armature=float(h["armature"]))
                jnt.limited = mujoco.mjtLimited.mjLIMITED_FALSE
                if h.get("solref_friction"):
                    jnt.solref_friction = [float(v) for v in h["solref_friction"]]
                if h.get("solimp_friction"):
                    jnt.solimp_friction = [float(v) for v in h["solimp_friction"]]
            add_geoms(body, cn)
            add_children(body, cn)

    Tc = LY.T_world_car(layout) @ np.asarray(B["chassis"]["T_design"], float)
    ch = w.add_body(name="chassis", pos=Tc[:3, 3].tolist(), quat=quat_wxyz(Tc[:3, :3]).tolist())
    add_geoms(ch, "chassis")
    add_children(ch, "chassis")
    parts = layout.get("parts") or {}
    for u in kit["loose"]:
        if u not in parts:
            raise ValueError("layout: no pose for the loose unit %s" % u)
        pos, quat = parts[u]
        body = w.add_body(name=u, pos=[float(v) for v in pos], quat=[float(v) for v in quat])
        body.add_freejoint(name="free_" + u)
        add_geoms(body, u)
        add_children(body, u)
    if robot_hook is None and cfg.get("robot"):
        from ..assembly_quad.ur_robot import add_ur
        robot_hook = lambda sp, c: add_ur(sp, c["robot"])        # noqa: E731
    robot = robot_hook(spec, cfg) if robot_hook is not None else {}
    for s in kit["slots"]:
        for c in s["candidates"]:
            eq = spec.add_equality(name=weld_name(s["id"], c["unit"]), type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                   name1=s["carrier"], name2=c["part"])
            eq.active = False
            eq.solref = list(cfg["weld_solref"])
            eq.solimp = list(cfg["weld_solimp"])
    pairs = [tuple(p) for p in kit.get("excludes", [])] + [("chassis", "stand")]
    for b1, b2 in pairs:
        ex = spec.add_exclude()
        ex.bodyname1, ex.bodyname2 = b1, b2
    return spec, robot


class CarCell(object):
    def __init__(self, kit_dir, layout, config=None, log=print, robot_hook=None, spec_hook=None, settle_s=1.0, robot_init=None,
                 kit=None):
        import mujoco
        self.mj, self.np, self.log = mujoco, np, log
        self.kit = kit if kit is not None else load_kit(kit_dir)
        check_kit(self.kit)
        self.kit_dir = self.kit["dir"]
        self.cfg = _merge_cfg(config)
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
        self.T_car = LY.T_world_car(layout)
        self.T_robot = self.T_car                   # the earlier cell's name for the assembled product's frame
        self.hinge_of = {bn: bb["joint"]["name"] for bn, bb in self.kit["bodies"].items() if isinstance(bb.get("joint"), dict)}
        for hb, deg in (layout.get("hub_deg") or {}).items():
            if hb in self.hinge_of:
                d.qpos[m.jnt_qposadr[m.joint(self.hinge_of[hb]).id]] = math.radians(float(deg))
        mujoco.mj_forward(m, d)
        if robot_init is None and self.robot:
            robot_init = _ur_init
        self.ctrls = robot_init(self) if robot_init is not None else {}
        self.sm = CouplingStateMachine(m, d, self.kit, **(self.cfg.get("coupling") or {}))
        self.slots = {s["id"]: s for s in self.kit["slots"]}
        unit_of = {bn: bb["unit"] for bn, bb in self.kit["bodies"].items()}
        self.unit_of_body = {b: unit_of[m.body(b).name] for b in range(m.nbody) if m.body(b).name in unit_of}
        self.geoms_of = {u: [] for u in self.loose + ["base"]}
        for g in range(m.ngeom):
            u = self.unit_of_body.get(int(m.geom_bodyid[g]))
            if u in self.geoms_of and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.geoms_of[u].append(g)
        # contact clusters (cfg["clusters"]): every rigidly fastened group of units gets a bit of its own (contype = the
        # bit, conaffinity = every bit but its own): nothing inside a cluster collides, every cluster still collides
        # with the rest. The coupling restores a wheel's stop with the bits of this table.
        self.col_bits = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in range(m.ngeom)}
        self.sm.collision_bits = self.col_bits
        self.n_fastened = -1
        self.clusters = []
        self.rec_t, self.rec_q, self.rec_s = [], [], []
        self.label = {"step": 0, "text": "start", "part": None, "slot": None}
        self.n_tick = 0
        self.hooks = []
        self.recording = True
        self.jig = {}
        self.ctrl = next(iter(self.ctrls.values()), None)
        self._init_watch()
        self.update_clusters()
        rec, self.recording = self.recording, False
        for _ in range(int(round(float(settle_s) / self.tick_s))):
            self.tick_one_physics()
        self.recording = rec
        mujoco.mj_forward(m, d)
        d.time = 0.0
        self.initial = {u: self.body_T(u) for u in self.loose}
        self.build_s = time.time() - t0
        log("[cell] race-car cell built and settled in %.1f s (%d bodies, %d geoms)" % (self.build_s, m.nbody, m.ngeom))

    # ------------------------------------------------------------------ helpers
    def tick_one_physics(self):
        """one control tick of the controllers and physics only (the settling before t = 0: no coupling, no watch, no hooks)"""
        for ct in self.ctrls.values():
            ct.tick()
        for _name, (k, T) in self.jig.items():
            self.d.mocap_pos[k] = T[:3, 3]
            self.d.mocap_quat[k] = quat_wxyz(T[:3, :3])
        stepper = [ct for ct in self.ctrls.values() if hasattr(ct, "step_physics")]
        if stepper:
            stepper[0].step_physics(self.tick_steps)
        else:
            for _s in range(self.tick_steps):
                self.mj.mj_step(self.m, self.d)

    def body_T(self, name):
        return body_pose(self.d, self.m.body(name).id)

    def hub_deg(self, hub):
        return math.degrees(float(self.d.qpos[self.m.jnt_qposadr[self.m.joint(self.hinge_of[hub]).id]]))

    def place(self, unit, T, forward=True):
        """put a loose unit at the world pose T at rest (a test's or a check's start)"""
        m, d = self.m, self.d
        j = m.joint("free_" + unit).id
        a, v = m.jnt_qposadr[j], m.jnt_dofadr[j]
        d.qpos[a:a + 3] = np.asarray(T, float)[:3, 3]
        d.qpos[a + 3:a + 7] = quat_wxyz(np.asarray(T, float)[:3, :3])
        d.qvel[v:v + 6] = 0.0
        if forward:
            self.mj.mj_forward(m, d)

    def save_trajectory(self, out_dir):
        """<out_dir>/trajectory.npz: t, q (qpos every 4 ticks), and per frame the label (the step in work) and the score"""
        labels = np.array([r["label"].get("text", "") for r in self.rec_s])
        score = np.array([r["score"] for r in self.rec_s], np.float32)
        np.savez_compressed(os.path.join(out_dir, "trajectory.npz"), t=np.array(self.rec_t), q=np.array(self.rec_q),
                            label=labels, score=score)

    def save_model(self, out_dir):
        """<out_dir>/cell.xml from the spec (the renderers and the true-solid check replay qpos through it)"""
        open(os.path.join(out_dir, "cell.xml"), "w").write(self.spec.to_xml())

    def record(self):
        self.rec_t.append(float(self.d.time))
        self.rec_q.append(np.array(self.d.qpos, np.float32))
        self.rec_s.append({"states": dict(self.sm.state), "score": self.sm.score(), "label": dict(self.label)})

    # ------------------------------------------------------------------ the tick
    def tick(self, n=1):
        for _ in range(n):
            self.tick_one()

    def tick_one(self):
        m, d = self.m, self.d
        self.tick_one_physics()
        self.mj.mj_forward(m, d)
        self.sm.tick(float(d.time))
        if self.cfg.get("clusters", True):
            nf = sum(1 for x in self.sm.rec.values() if x["fastened"])
            if nf != self.n_fastened:
                self.update_clusters()
        if self.cfg["watch"].get("enabled", True):
            self.watch_tick()
        self.n_tick += 1
        for h in self.hooks:
            h()
        if self.recording and self.n_tick % 4 == 0:
            self.record()

    def update_clusters(self):
        """the contact clusters from the fastened slots (see __init__); the geoms' bits in col_bits (what the coupling's
        stop restore and a jig's collision toggle use)"""
        m = self.m
        parent = {u: u for u in self.geoms_of}

        def root(u):
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u
        if self.cfg.get("clusters", True):
            for sid, st in self.sm.rec.items():
                if st["fastened"] and st["unit"] is not None:
                    cu = self.kit["bodies"][self.slots[sid]["carrier"]]["unit"]
                    if cu in parent and st["unit"] in parent:
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
                if m.geom_contype[g] or m.geom_conaffinity[g]:          # a stop or a jig may have switched it off
                    m.geom_contype[g], m.geom_conaffinity[g] = ct, ca
        self.clusters = clusters
        self.n_fastened = sum(1 for x in self.sm.rec.values() if x["fastened"])

    # ------------------------------------------------------------------ the watch
    def _init_watch(self):
        m = self.m
        prefix = (self.robot or {}).get("prefix")
        pads = set(m.geom(n).id for n in (self.robot or {}).get("pad_geoms", []))
        loose = set(self.loose)
        # geom classes: 0 environment (world, fixtures, the chassis and its children), 1 robot link, 2 pad, 3 loose unit
        self.gclass = np.zeros(m.ngeom, int)
        self.gunit = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if g in pads:
                self.gclass[g] = 2
            elif prefix and m.body(b).name.startswith(prefix):
                self.gclass[g] = 1
            elif self.unit_of_body.get(b) in loose:
                self.gclass[g] = 3
                self.gunit[g] = self.unit_of_body[b]
        self.unit_body = {u: m.body(u).id for u in self.loose}
        self._f6 = np.zeros(6)
        self._v6 = np.zeros(6)
        self.reset_watch()

    def reset_watch(self):
        self.violations = []
        self.watch_events = []
        self.max_robot_force = 0.0
        self.max_robot_force_what = None
        self.max_robot_contact = 0.0
        self.max_part_force = 0.0
        self.max_part_force_what = None
        self.max_part_sustained = 0.0
        self.max_part_sustained_what = None
        self.max_speed = {}
        self._part_hist = []
        self._part_over = 0

    def watch_tick(self):
        """scrapping and the safety rules for this tick (module docstring); returns the tick's events"""
        m, d, mj = self.m, self.d, self.mj
        W = self.cfg["watch"]
        t = float(d.time)
        events = []
        thr_f = float(W["robot_force_N"])
        part_sum, rob_sum = {}, {}
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            c1, c2 = self.gclass[g1], self.gclass[g2]
            if c1 in (1, 2) and c2 in (1, 2):
                continue
            if c1 == 1 or c2 == 1:
                mj.mj_contactForce(m, d, i, self._f6)
                fn = abs(float(self._f6[0]))
                link, other = (g1, g2) if c1 == 1 else (g2, g1)
                key = (int(m.geom_bodyid[link]), int(m.geom_bodyid[other]))
                rob_sum[key] = rob_sum.get(key, 0.0) + fn
                self.max_robot_contact = max(self.max_robot_contact, fn)
                continue
            if c1 == 2 or c2 == 2:
                continue
            if c1 == 3 or c2 == 3:
                mj.mj_contactForce(m, d, i, self._f6)
                fn = abs(float(self._f6[0]))
                gp, go = (g1, g2) if c1 == 3 else (g2, g1)
                key = (self.gunit[gp], m.body(int(m.geom_bodyid[go])).name)
                part_sum[key] = part_sum.get(key, 0.0) + fn
        # the robot-contact rule (watch.robot_force_N): the normal forces of a robot link (not a pad) on one body, summed over their contacts
        for (lb, ob), fn in rob_sum.items():
            what = "%s:%s" % (m.body(lb).name, m.body(ob).name)
            if fn > self.max_robot_force:
                self.max_robot_force, self.max_robot_force_what = fn, what
            if fn > thr_f:
                v = "robot_contact:%s:%.0fN" % (what, fn)
                last = [x for x in self.violations[-5:] if x["what"].startswith("robot_contact:" + what + ":")]
                if not last or last[-1]["t"] < t - 0.5:
                    self.violations.append({"t": t, "what": v, "force_N": fn})
                    events.append("safety:" + v)
        pm, pk = 0.0, None
        for k_, v_ in part_sum.items():
            if v_ > pm:
                pm, pk = v_, k_
        if pm > self.max_part_force:
            self.max_part_force, self.max_part_force_what = pm, pk
        n = int(W["part_force_ticks"])
        self._part_hist.append(pm)
        if len(self._part_hist) > n:
            del self._part_hist[0]
        if len(self._part_hist) == n:
            s = min(self._part_hist)
            if s > self.max_part_sustained:
                self.max_part_sustained, self.max_part_sustained_what = s, pk
        if pm > float(W["part_force_N"]):
            self._part_over += 1
            if self._part_over >= n:
                v = "part_force:%s:%s:%.0fN" % (pk[0], pk[1], pm)
                last = [x for x in self.violations[-5:] if x["what"].startswith("part_force:%s:" % pk[0])]
                if not last or last[-1]["t"] < t - 0.5:
                    self.violations.append({"t": t, "what": v, "force_N": pm})
                    events.append("safety:" + v)
        else:
            self._part_over = 0
        z_off = -float(W["scrap_below_bench_m"])
        vmax = float(W["scrap_speed_m_s"])
        for u in self.loose:
            b = self.unit_body[u]
            mj.mj_objectVelocity(m, d, mj.mjtObj.mjOBJ_BODY, b, self._v6, 0)          # at the centre of mass
            v = float(np.linalg.norm(self._v6[3:]))
            if v > self.max_speed.get(u, 0.0):
                self.max_speed[u] = v
            if not W.get("scrap", True) or u in self.sm.scrapped:
                continue
            why = None
            if float(d.xipos[b][2]) < z_off:
                why = "off the bench"
            elif v > vmax:
                why = "faster than %.1f m/s (%.2f m/s)" % (vmax, v)
            if why is not None and self.sm.scrap(u, t, why):
                events.append("scrapped:%s:%s" % (u, why))
        for e in events:
            self.watch_events.append((t, e))
        return events

    def watch_summary(self):
        hard = [v for v in self.violations if v["what"].split(":")[0] in ("robot_contact", "part_force")]
        return {"max_robot_link_force_N": round(self.max_robot_force, 3), "max_robot_link_force_pair": self.max_robot_force_what,
                "max_robot_link_contact_N": round(self.max_robot_contact, 3),
                "max_part_force_instant_N": round(self.max_part_force, 3),
                "max_part_force_instant_pair": None if self.max_part_force_what is None else list(self.max_part_force_what),
                "max_part_force_sustained_N": round(self.max_part_sustained, 3),
                "max_part_force_sustained_pair": None if self.max_part_sustained_what is None else list(self.max_part_sustained_what),
                "max_part_speed_m_s": {k: round(v, 4) for k, v in self.max_speed.items()},
                "scrapped": dict(self.sm.scrapped), "violations": list(self.violations), "hard_ok": not hard}

    # ------------------------------------------------------------------ targets
    def candidate(self, sid, unit):
        s = self.slots[sid]
        return [c for c in s["candidates"] if c["unit"] == unit][0]

    def live_target(self, sid, unit, k=None):
        """{"T": the unit's root pose (world) home on the slot from its carrier as it is now, "carrier", "F": the slot's
        frame on the carrier (world; z along the insertion), "insertion": the unit's direction of travel home (world),
        "k": the key orientation used}. keyed: k (default: the design orientation; any_key slots: the nearest one)"""
        s = self.slots[sid]
        c = self.candidate(sid, unit)
        Tc = self.body_T(s["carrier"])
        if s["kind"] == "seat":
            Tp = Tc @ np.asarray(c["T_part_in_carrier"], float)
            ins = Tc[:3, :3] @ np.asarray(s["insertion_carrier"], float)
            Fp = np.asarray(c.get("frame_in_part", np.eye(4)), float)
            return {"T": Tp, "carrier": s["carrier"], "F": Tp @ Fp, "insertion": ins / np.linalg.norm(ins), "k": 0}
        Fc = np.asarray(s["frame_in_carrier"], float)
        Fp = np.asarray(c["frame_in_part"], float)
        pitch = math.radians(float(s.get("pitch_deg", 360.0)))
        F = Tc @ Fc
        if k is None:
            if s.get("any_key"):
                x = self.sm.by_id[sid]
                cc = [q for q in x["cands"] if q["unit"] == unit][0]
                k = int(self.sm.evaluate(x, cc)["k"])
            else:
                k = int(s.get("design_k", 0))
        T = F @ _rotz(k * pitch) @ _inv(Fp)
        return {"T": T, "carrier": s["carrier"], "F": F, "insertion": F[:3, 2] / np.linalg.norm(F[:3, 2]), "k": int(k)}

    def design_world(self):
        """{body: world pose} of the car assembled at its design (every unit at its designed slot, the hubs at q = 0)"""
        return {bn: self.T_car @ np.asarray(bb["T_design"], float) for bn, bb in self.kit["bodies"].items()}

    def fixture_T(self, name):
        return self.body_T(name)


def _ur_init(cell):
    """the UR's controller at the ready posture (joint mode), the gripper open"""
    from ..assembly_quad.gripper_2f85 import set_opening
    from ..assembly_quad.ur_controller import URController
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
    set_opening(m, d, ct.max_width, prefix=rb["prefix"])
    cell.mj.mj_forward(m, d)
    ct.write_ctrl()
    return {"ur": ct}
