"""Appliance tending: prototype robot environment (P1 of docs/dev/20261112-appliance-tending-task-design.md).

Modelled on tasks/arm/arm_hidden_mechanism_box/tests/box_env.py: plain MuJoCo 3.16, a fixed Panda
(robosuite 1.5.2 model files, robosuite itself is not imported), an operational-space controller at
10 Hz, two fixed cameras and a wrist camera, a wrist force/torque sensor. What it loads instead of a
puzzle box is one **unit**: a RoboCasa appliance model (stove, stovetop, microwave or oven, from the
CC-BY-4.0 fixture assets) mounted in front of the robot. No RoboCasa code runs; the appliance
predicates RoboCasa defines in Python are re-implemented here (see `Unit.state`).

Frames: world z up; the robot base at ROBOT_BASE faces +x; a unit's front faces -x (RoboCasa models
face -y; they are turned by -90 degrees about z). Metres, radians, seconds; quaternions (w, x, y, z).

    spec = unit_spec("microwaves/Microwave011")
    env = ApplianceEnv(spec, assets="~/arb-data/robocasa-assets/extracted/fixtures")
    obs = env.reset(seed)
    obs, done, info = env.step([dx, dy, dz, drx, dry, drz, grip, cams])
    env.unit_state()          # every control of the unit, normalized, plus the predicates

Prototype only: no request queue, no credit, no wire. Those come in P2-P4.
"""

import math
import os

import mujoco
import numpy as np

TIMESTEP = 0.002
SUBSTEPS = 50                     # -> 10 Hz control
CONTROL_DT = TIMESTEP * SUBSTEPS
MAX_DPOS = 0.03
MAX_DROT = 0.2
WRENCH_NOISE = 0.3
RGB_NOISE = 2.0
HOME_Q = np.array([0.0, -0.55, 0.0, -2.45, 0.0, 1.95, math.pi / 4])
NULLSPACE_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, math.pi / 4])
HOME_SITE = np.array([0.45, 0.0, 0.55])  # the gripper site at home, relative to the robot base; P2 chose the
                                         # candidate with the largest joint-limit margin over six units
HOME_MIN_DOWN = math.radians(20)  # the home approach points at least this far below horizontal
HOME_STARTS = [HOME_Q, np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, math.pi / 4]),
               np.array([0.0, 0.0, 0.0, -1.6, 0.0, 1.6, math.pi / 4]),
               np.array([0.0, -0.8, 0.0, -2.7, 0.0, 2.6, math.pi / 4])]
HOME_LOOK_BELOW_TOP = 0.15        # it points at the unit's front centre line this far below the unit's top,
                                  # so the wrist camera sees the unit's front
# Control physics. RoboCasa's knobs carry frictionloss 0.75-1.0 N m and armature 1.0 kg m^2, tuned for
# its own controller; this environment's controller (box_env's: the target re-based each step, at most
# 0.03 m / 0.2 rad ahead) delivers about 0.5-0.8 N m at the wrist, and a real cooker knob needs about
# 0.1-0.3 N m. P2 sets knobs to a real knob's values (design doc section 10, P2).
KNOB_FRICTIONLOSS = 0.20          # N m
KNOB_ARMATURE = 0.002             # kg m^2
VISUAL_MASS = 1e-6                # visual meshes carry no mass (as robosuite does for its objects)

# Mounting per kind: where the unit's bottom sits and the robot base height. P1's reach study tunes
# these. FRONT_X: the unit's front face, metres in front of the robot base.
# P1 reach study (docs/dev/20261112-appliance-tending-task-design.md section 10): front 0.55 m and a robot
# base 0.65 m high reach every control of every model but two rear-panel stoves, kinematically.
COUNTER_Z = 0.92
MOUNTS = {
    "stovetop": {"bottom_z": COUNTER_Z, "front_x": 0.55, "robot_z": 0.65},
    "microwave": {"bottom_z": COUNTER_Z, "front_x": 0.55, "robot_z": 0.65},
    "oven": {"bottom_z": 0.55, "front_x": 0.55, "robot_z": 0.65},
    "stove": {"bottom_z": 0.0, "front_x": 0.55, "robot_z": 0.65},
}
KIND_OF_FOLDER = {"stovetops": "stovetop", "microwaves": "microwave", "ovens": "oven", "stoves": "stove"}

CAMERAS = {
    # name: (position, look-at point; both relative to the centre of the unit's front face), fovy, w, h.
    # Beside and above the robot's shoulders, so the arm occludes as little of the front as possible.
    "left": ((-0.45, 0.62, 0.45), (0.12, 0.0, 0.0), 55.0, 320, 240),
    "right": ((-0.45, -0.62, 0.45), (0.12, 0.0, 0.0), 55.0, 320, 240),
}
FLAME_ALPHA = 0.5                 # RoboCasa Stove.update_state shows a burner's flame site while it is on
WRIST = {"fovy": 75.0, "width": 256, "height": 192}

BURNERS = ("rear_left", "rear_center", "rear_right", "front_left", "front_center", "front_right",
           "center", "left", "right")
BURNER_ON_MIN = 0.35              # RoboCasa Stove.is_burner_on: on iff 0.35 <= angle mod 2pi <= 2pi - 0.35
DOOR_OPEN_TH = 0.90               # RoboCasa Fixture.is_open
DOOR_CLOSED_TH = 0.005            # RoboCasa Fixture.is_closed
BUTTON_FAR = 0.15                 # RoboCasa Microwave.gripper_button_far
PREHEAT_TH = 0.25                 # RoboCasa PreheatOven: normalized temperature knob >= 0.25


class EnvError(Exception):
    pass


def robosuite_assets():
    import importlib.util
    spec = importlib.util.find_spec("robosuite")
    if spec is None or not spec.submodule_search_locations:
        raise EnvError("robosuite assets not installed")
    return os.path.join(list(spec.submodule_search_locations)[0], "models", "assets")


def look_at(pos, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.column_stack([r, u, -f])          # MuJoCo camera: looks along -z, y up


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def axis_angle_mat(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def ik(m, d, site, jq, jv, lo, hi, target, approach, iters=300, q_rest=None):
    """Position + approach-axis (the gripper site's z) IK from d.qpos, modified in place.
    Returns (position error m, approach error rad)."""
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    approach = np.asarray(approach, float) / np.linalg.norm(approach)
    for _ in range(iters):
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p = d.site_xpos[site]
        z = d.site_xmat[site].reshape(3, 3)[:, 2]
        ep = target - p
        er = np.cross(z, approach)
        if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
            break
        mujoco.mj_jacSite(m, d, jp, jr, site)
        J = np.vstack([jp[:, jv], 0.5 * jr[:, jv]])
        e = np.concatenate([ep, 0.5 * er])
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-4 * np.eye(6))
        dq = Jp @ e
        if q_rest is not None:                  # stay near a rest pose in the task's null space
            dq += (np.eye(len(jq)) - Jp @ J) @ (0.1 * (q_rest - d.qpos[jq]))
        d.qpos[jq] = np.clip(d.qpos[jq] + np.clip(dq, -0.1, 0.1), lo, hi)
    mujoco.mj_kinematics(m, d)
    z = d.site_xmat[site].reshape(3, 3)[:, 2]
    return float(np.linalg.norm(target - d.site_xpos[site])), math.acos(max(-1.0, min(1.0, float(z @ approach))))


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def unit_spec(model, scale=1.0, kind=None, door_mass=None):
    """A unit = one RoboCasa model plus how it is mounted. `model` is '<folder>/<name>'."""
    folder = model.split("/")[0]
    return {"model": model, "kind": kind or KIND_OF_FOLDER[folder], "scale": float(scale),
            "door_mass": door_mass}


# ----------------------------------------------------------------------------------------------
# Unit loading
# ----------------------------------------------------------------------------------------------
def load_unit_spec(assets, spec):
    """MjSpec of the unit's model.xml, uniformly scaled, visual meshes massless, collision geoms
    hidden from the cameras, mesh and texture files made absolute."""
    folder = os.path.join(os.path.expanduser(assets), spec["model"])
    u = mujoco.MjSpec.from_file(os.path.join(folder, "model.xml"))
    s = spec["scale"]
    for mesh in u.meshes:
        if mesh.file and not os.path.isabs(mesh.file):
            mesh.file = os.path.join(folder, mesh.file)
        mesh.scale = np.asarray(mesh.scale) * s
    for tex in u.textures:
        if tex.file and not os.path.isabs(tex.file):
            tex.file = os.path.join(folder, tex.file)
    for b in u.bodies:
        b.pos = np.asarray(b.pos) * s
    for g in u.geoms:
        g.pos = np.asarray(g.pos) * s
        if g.type != mujoco.mjtGeom.mjGEOM_MESH:
            g.size = np.asarray(g.size) * s
        visual = g.contype == 0 and g.conaffinity == 0
        if visual and g.type == mujoco.mjtGeom.mjGEOM_MESH:
            g.mass = VISUAL_MASS
        elif visual:
            g.mass = VISUAL_MASS        # spawn / region markers
            g.rgba = [0, 0, 0, 0]
        else:
            g.rgba = [0, 0, 0, 0]       # collision geometry: physics only (RoboCasa hides it too)
    for site in u.sites:
        site.pos = np.asarray(site.pos) * s
        site.size = np.asarray(site.size) * s
    for j in u.joints:
        j.pos = np.asarray(j.pos) * s
        if j.type == mujoco.mjtJoint.mjJNT_SLIDE:
            j.range = np.asarray(j.range) * s
        if j.name.startswith("knob_"):
            j.frictionloss = spec.get("knob_frictionloss", KNOB_FRICTIONLOSS)
            j.armature = KNOB_ARMATURE
    if spec.get("door_mass"):
        for b in u.bodies:
            if b.name == "door":
                b.explicitinertial = True
                b.mass = float(spec["door_mass"])
    return u


def unit_bbox(u_spec_compiled):
    """Box over the unit's visual meshes, in its own frame (before mounting)."""
    m = u_spec_compiled
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_contype[g] != 0:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        c = d.geom_xpos[g] + R @ m.geom_aabb[g, :3]
        h = np.abs(R) @ m.geom_aabb[g, 3:]
        lo, hi = np.minimum(lo, c - h), np.maximum(hi, c + h)
    return lo, hi


def mount_pose(kind, lo, hi, jitter=(0.0, 0.0, 0.0), mount=None):
    """World position + yaw of the unit's model frame so that its front face (model -y, turned to
    face -x) is front_x in front of the robot, centred on the robot, its bottom at bottom_z."""
    mt = dict(MOUNTS[kind], **(mount or {}))
    yaw = -math.pi / 2 + jitter[2]
    R = axis_angle_mat([0, 0, yaw])
    # after the turn, model (x, y) -> world (y, -x): the front (model y = lo[1]) faces -x
    centre_model = np.array([(lo[0] + hi[0]) / 2, lo[1], lo[2]])
    target = np.array([mt["front_x"] + jitter[0], jitter[1], mt["bottom_z"]])
    pos = target - R @ centre_model
    return pos, yaw, mt


# ----------------------------------------------------------------------------------------------
# Scene
# ----------------------------------------------------------------------------------------------
def build_model(assets, spec, jitter=(0.0, 0.0, 0.0), mount=None):
    u = load_unit_spec(assets, spec)
    lo, hi = unit_bbox(u.compile())
    u = load_unit_spec(assets, spec)            # compile() froze the spec; start from a fresh copy
    pos, yaw, mt = mount_pose(spec["kind"], lo, hi, jitter, mount)
    robot_base = np.array([0.0, 0.0, mt["robot_z"]])

    w = mujoco.MjSpec()
    w.compiler.degree = False
    w.compiler.autolimits = True
    w.option.timestep = TIMESTEP
    w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    w.option.impratio = 2.0
    w.visual.global_.offwidth = 640
    w.visual.global_.offheight = 480
    wb = w.worldbody
    light = wb.add_light()
    light.pos = [-0.5, 0.0, 2.6]
    light.dir = [0.3, 0.0, -1.0]
    light.diffuse = [0.8, 0.8, 0.8]
    light.castshadow = False
    g = wb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_PLANE
    g.size = [3, 3, 0.1]
    g.rgba = [0.55, 0.52, 0.48, 1]
    # support under the unit: the counter (stovetop, microwave), a cabinet column (oven), nothing (stove)
    depth = hi[1] - lo[1]
    width = hi[0] - lo[0]
    front = mt["front_x"] + jitter[0]
    if mt["bottom_z"] > 0.01:
        g = wb.add_geom()
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [max(depth, 0.6) / 2 + 0.05, max(width, 0.6) / 2 + 0.25, mt["bottom_z"] / 2]
        g.pos = [front + g.size[0] - 0.03, jitter[1], mt["bottom_z"] / 2]
        g.name = "support"
        g.rgba = [0.82, 0.80, 0.76, 1]
    # back wall
    g = wb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [0.02, 1.5, 1.4]
    g.pos = [front + max(depth, 0.6) + 0.15, 0.0, 1.4]
    g.name = "back_wall"
    g.rgba = [0.9, 0.88, 0.84, 1]
    # robot pedestal (visual)
    g = wb.add_geom()
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.pos = [robot_base[0] - 0.05, 0.0, robot_base[2] / 2]
    g.size = [0.14, 0.14, robot_base[2] / 2]
    g.rgba = [0.25, 0.25, 0.27, 1]
    g.contype = 0
    g.conaffinity = 0
    # Cameras and the home pose follow the NOMINAL mount, never the episode's jitter: cameras that moved
    # with the unit would show it in the same place every episode, and a home pose solved for the
    # jittered unit would encode its pose in the initial joint angles.
    front_centre = np.array([mt["front_x"], 0.0, mt["bottom_z"] + (hi[2] - lo[2]) / 2])
    for name, (p, tgt, fovy, _w, _h) in CAMERAS.items():
        cp = front_centre + np.asarray(p)
        ct = front_centre + np.asarray(tgt)
        c = wb.add_camera()
        c.name = name
        c.pos = list(cp)
        c.quat = list(quat_from_mat(look_at(cp, ct)))
        c.fovy = fovy
    # robot
    ra = robosuite_assets()
    robot = mujoco.MjSpec.from_file(os.path.join(ra, "robots", "panda", "robot.xml"))
    grip = mujoco.MjSpec.from_file(os.path.join(ra, "grippers", "panda_gripper.xml"))
    for cam in list(robot.cameras):
        if cam.name == "robotview":
            robot.delete(cam)
    robot.body("right_hand").add_frame().attach_body(grip.body("right_gripper"), "g_", "")
    wb.add_frame(pos=list(robot_base)).attach_body(robot.body("base"), "r_", "")
    # unit
    wb.add_frame(pos=list(pos), quat=list(yaw_quat(yaw))).attach_body(u.body("object"), "u_", "")
    m = w.compile()
    m.site_rgba[:, 3] = 0.0           # sites are markers, never seen (burner flames are switched by the unit)
    top_z = mt["bottom_z"] + (hi[2] - lo[2])
    look = np.array([mt["front_x"], 0.0, top_z - HOME_LOOK_BELOW_TOP])
    return m, {"unit_pos": pos, "unit_yaw": yaw, "robot_base": robot_base, "bbox_lo": lo, "bbox_hi": hi,
               "mount": mt, "look_point": look}


# ----------------------------------------------------------------------------------------------
# Unit state (RoboCasa's predicates, re-implemented)
# ----------------------------------------------------------------------------------------------
def normalized_joint(q, lo, hi):
    """RoboCasa Fixture.get_joint_state: (q - lo) / (hi - lo), flipped when lo < 0."""
    n = (q - lo) / (hi - lo)
    n = min(max(n, 0.0), 1.0)
    return 1.0 - n if lo < 0 else n


class Unit(object):
    """Indexes a compiled unit's controls (joints named as RoboCasa names them, prefixed u_)."""

    def __init__(self, m, kind):
        self.kind = kind
        self.knobs = {}
        for loc in BURNERS:
            try:
                j = m.joint("u_knob_%s_joint" % loc)
            except KeyError:
                continue
            self.knobs[loc] = (int(j.qposadr[0]), int(j.dofadr[0]))
        self.door = None
        for name in ("u_microjoint", "u_door_joint", "u_Door001_joint"):
            try:
                j = m.joint(name)
            except KeyError:
                continue
            self.door = (int(j.qposadr[0]), float(j.range[0]), float(j.range[1]), int(j.dofadr[0]))
            break
        self.temp_knob = self._joint(m, "u_knob_temp_joint")
        self.time_knob = self._joint(m, "u_knob_time_joint")
        self.buttons = {}
        for b in ("start_button", "stop_button"):
            try:
                self.buttons[b] = m.geom("u_" + b).id
            except KeyError:
                pass
        self.handle = None
        try:
            self.handle = m.geom("u_door_handle_main").id
        except KeyError:
            pass
        self.flames = {}
        for loc in self.knobs:
            try:
                self.flames[loc] = m.site("u_burner_on_%s" % loc).id
            except KeyError:
                pass
        self.turned_on = False
        self.dofs = [v[1] for v in self.knobs.values()]
        if self.door:
            self.dofs.append(self.door[3])

    @staticmethod
    def _joint(m, name):
        try:
            j = m.joint(name)
        except KeyError:
            return None
        return (int(j.qposadr[0]), float(j.range[0]), float(j.range[1]))

    def door_fraction(self, d):
        if self.door is None:
            return None
        qa, lo, hi, _ = self.door
        return normalized_joint(float(d.qpos[qa]), lo, hi)

    def burner_on(self, d, loc):
        a = float(d.qpos[self.knobs[loc][0]]) % (2 * math.pi)
        return BURNER_ON_MIN <= a <= 2 * math.pi - BURNER_ON_MIN

    def update(self, m, d, gripper_geoms):
        """RoboCasa Stove.update_state (a burner's flame shows while its knob is on) and
        Microwave.update_state (door open -> off; else a gripper touching start turns it on,
        touching stop turns it off)."""
        for loc, sid in self.flames.items():
            m.site_rgba[sid, 3] = FLAME_ALPHA if self.burner_on(d, loc) else 0.0
        if "start_button" not in self.buttons:
            return
        touching = set()
        for i in range(d.ncon):
            c = d.contact[i]
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in gripper_geoms:
                    touching.add(b)
        if (self.door_fraction(d) or 0.0) >= DOOR_OPEN_TH:
            self.turned_on = False
        elif self.turned_on and self.buttons["stop_button"] in touching:
            self.turned_on = False
        elif not self.turned_on and self.buttons["start_button"] in touching:
            self.turned_on = True

    def state(self, d, grip_pos):
        s = {}
        for loc, (qa, _dof) in self.knobs.items():
            a = float(d.qpos[qa]) % (2 * math.pi)
            s["knob_" + loc] = round(a if a <= math.pi else a - 2 * math.pi, 4)
            s["burner_on_" + loc] = bool(BURNER_ON_MIN <= a <= 2 * math.pi - BURNER_ON_MIN)
        if self.door is not None:
            f = self.door_fraction(d)
            s["door_fraction"] = round(f, 4)
            s["door_open"] = bool(f >= DOOR_OPEN_TH)
            s["door_closed"] = bool(f <= DOOR_CLOSED_TH)
        if self.temp_knob is not None:
            qa, lo, hi = self.temp_knob
            t = normalized_joint(float(d.qpos[qa]), lo, hi)
            s["temperature"] = round(t, 4)
            s["preheated"] = bool(t >= PREHEAT_TH)
        if self.buttons:
            s["turned_on"] = self.turned_on
            for b, gid in self.buttons.items():
                s[b + "_far"] = bool(np.linalg.norm(grip_pos - d.geom_xpos[gid]) > BUTTON_FAR)
        return s


# ----------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------
class ApplianceEnv(object):

    def __init__(self, spec, assets, render=True, mount=None, jitter_scale=(0.03, 0.04, math.radians(10))):
        self.spec = spec
        self.assets = assets
        self.render_enabled = render
        self.mount = mount
        self.jitter_scale = np.asarray(jitter_scale, float)
        self.model = None
        self._renderers = {}

    def reset(self, seed):
        self.seed = int(seed) & (2 ** 64 - 1)
        place = np.random.default_rng([self.seed & 0xFFFFFFFF, self.seed >> 32, 7])
        self.rng = np.random.default_rng([self.seed & 0xFFFFFFFF, self.seed >> 32, 11])
        jitter = place.uniform(-1, 1, 3) * self.jitter_scale
        self.model, self.layout = build_model(self.assets, self.spec, jitter, self.mount)
        self.data = mujoco.MjData(self.model)
        self.unit = Unit(self.model, self.spec["kind"])
        self._index()
        self._init_state()
        self.t = 0
        self.render_next = ("left", "right", "wrist")
        if self.render_enabled:
            self._renderers = {}
        return self.observation()

    def _index(self):
        m = self.model
        self.jq = np.array([m.joint("r_joint%d" % i).qposadr[0] for i in range(1, 8)])
        self.jv = np.array([m.joint("r_joint%d" % i).dofadr[0] for i in range(1, 8)])
        self.act = np.array([m.actuator("r_torq_j%d" % i).id for i in range(1, 8)])
        self.gact = [m.actuator("r_g_gripper_finger_joint1").id, m.actuator("r_g_gripper_finger_joint2").id]
        self.gq = [m.joint("r_g_finger_joint1").qposadr[0], m.joint("r_g_finger_joint2").qposadr[0]]
        self.site = m.site("r_g_grip_site").id
        self.ft_force = m.sensor("r_g_force_ee").adr[0]
        self.ft_torque = m.sensor("r_g_torque_ee").adr[0]
        self.wrist_cam = m.camera("r_eye_in_hand").id
        gripper = m.body("r_g_right_gripper").id      # RoboCasa checks contact with the gripper model only
        self.gripper_geoms = set(g for g in range(m.ngeom) if self._under(m, m.geom_bodyid[g], gripper))
        self.ft_site = m.site("r_g_ft_frame").id
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))

    @staticmethod
    def _under(m, body, root):
        while body > 0:
            if body == root:
                return True
            body = m.body_parentid[body]
        return False

    def home_q(self):
        """Joint pose with the gripper site at HOME_SITE pointing at the unit's look point (so the wrist
        camera sees the unit's front), solved by damped least squares from HOME_Q."""
        m = self.model
        d = mujoco.MjData(m)
        d.qpos[self.jq] = HOME_Q
        target = self.layout["robot_base"] + HOME_SITE
        jid = [m.joint("r_joint%d" % i).id for i in range(1, 8)]
        lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
        v = self.layout["look_point"] - target
        horiz = math.hypot(v[0], v[1])
        v[2] = min(v[2], -math.tan(HOME_MIN_DOWN) * horiz)
        best = None
        for q0 in HOME_STARTS:                  # fixed starts: the result is a function of the unit only
            d.qpos[self.jq] = q0
            pe, ae = ik(m, d, self.site, self.jq, self.jv, lo, hi, target, v, iters=600, q_rest=HOME_Q)
            q = d.qpos[self.jq].copy()
            score = (pe > 0.005 or ae > 0.05, -float(np.min(np.minimum(q - lo, hi - q))))
            if best is None or score < best[0]:
                best = (score, q)
        return best[1]

    def _init_state(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.jq] = self.home_q()
        d.qpos[self.gq[0]], d.qpos[self.gq[1]] = 0.04, -0.04
        mujoco.mj_forward(m, d)
        self.grip_target = 0.04
        self.q_null = d.qpos[self.jq].copy()      # the null space pulls towards the unit-facing home pose
        q_hold = d.qpos[self.jq].copy()
        for _ in range(150):
            d.ctrl[self.act] = 200.0 * (q_hold - d.qpos[self.jq]) - 20.0 * d.qvel[self.jv] + d.qfrc_bias[self.jv]
            d.ctrl[self.gact] = [0.04, -0.04]
            mujoco.mj_step(m, d)
        self.target_pos = d.site_xpos[self.site].copy()
        self.target_R = d.site_xmat[self.site].reshape(3, 3).copy()

    def set_joint(self, name, value):
        """Set a unit joint directly (initial states only; never during an episode)."""
        j = self.model.joint("u_" + name)
        self.data.qpos[j.qposadr[0]] = value
        mujoco.mj_forward(self.model, self.data)
        self.unit.update(self.model, self.data, self.gripper_geoms)

    def _osc_torque(self):
        m, d = self.model, self.data
        mujoco.mj_jacSite(m, d, self._jp, self._jr, self.site)
        J = np.vstack([self._jp, self._jr])[:, self.jv]
        mujoco.mj_fullM(m, d, self._M)
        Mq = self._M[np.ix_(self.jv, self.jv)]
        R = d.site_xmat[self.site].reshape(3, 3)
        ep = self.target_pos - d.site_xpos[self.site]
        Re = self.target_R @ R.T
        eo = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
        v = J @ d.qvel[self.jv]
        kp, ko = 400.0, 200.0
        acc = np.concatenate([kp * ep - 2.0 * math.sqrt(kp) * v[:3], ko * eo - 2.0 * math.sqrt(ko) * v[3:]])
        Minv = np.linalg.inv(Mq)
        Lam = np.linalg.inv(J @ Minv @ J.T + 1e-2 * np.eye(6))
        tau = J.T @ (Lam @ acc) + d.qfrc_bias[self.jv]
        Jbar = Minv @ J.T @ Lam
        N = np.eye(7) - J.T @ Jbar.T
        dq = d.qvel[self.jv]
        tau += N @ (5.0 * (self.q_null - d.qpos[self.jq]) - 6.0 * dq)
        tau -= 0.5 * dq
        over = np.abs(dq) - 1.5
        tau -= np.where(over > 0, 30.0 * over * np.sign(dq), 0.0)
        return tau

    def step(self, action):
        a = np.asarray(action, float)
        d = self.data
        dp, dr = a[:3], a[3:6]
        n = np.linalg.norm(dp)
        if n > MAX_DPOS:
            dp = dp * (MAX_DPOS / n)
        n = np.linalg.norm(dr)
        if n > MAX_DROT:
            dr = dr * (MAX_DROT / n)
        self.target_pos = d.site_xpos[self.site] + dp
        self.target_R = axis_angle_mat(dr) @ d.site_xmat[self.site].reshape(3, 3)
        self.grip_target = 0.04 * (1.0 - float(np.clip(a[6], -1, 1))) / 2.0
        cams = int(a[7]) if len(a) > 7 else 0
        self.render_next = tuple(n for i, n in enumerate(("left", "right", "wrist")) if cams & (1 << i))
        for _ in range(SUBSTEPS):
            d.ctrl[self.act] = self._osc_torque()
            d.ctrl[self.gact] = [self.grip_target, -self.grip_target]
            mujoco.mj_step(self.model, d)
        # RoboCasa updates fixture state once per control step (after the physics substeps).
        self.unit.update(self.model, d, self.gripper_geoms)
        self.t += 1
        return self.observation(), False, {}

    def wrench_world(self):
        """Force (N) and torque (N m) at the wrist sensor, rotated from the sensor frame to the world frame."""
        d = self.data
        R = d.site_xmat[self.ft_site].reshape(3, 3)
        return np.concatenate([R @ d.sensordata[self.ft_force:self.ft_force + 3],
                               R @ d.sensordata[self.ft_torque:self.ft_torque + 3]])

    def unit_state(self):
        return self.unit.state(self.data, self.data.site_xpos[self.site])

    def render(self, cam):
        m, d = self.model, self.data
        if cam == "wrist":
            w, h, cid = WRIST["width"], WRIST["height"], self.wrist_cam
        else:
            _p, _t, _f, w, h = CAMERAS[cam]
            cid = m.camera(cam).id
        key = (w, h)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(m, h, w)
        r = self._renderers[key]
        r.update_scene(d, camera=cid)
        img = r.render().astype(np.float32)
        img += self.rng.normal(0, RGB_NOISE, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def observation(self):
        d = self.data
        R = d.site_xmat[self.site].reshape(3, 3)
        obs = {
            "t": self.t,
            "q": d.qpos[self.jq].copy(),
            "dq": d.qvel[self.jv].copy(),
            "ee_pos": d.site_xpos[self.site].copy(),
            "ee_quat": quat_from_mat(R),
            "gripper": float(d.qpos[self.gq[0]] - d.qpos[self.gq[1]]),
            "wrench": self.wrench_world()
            + self.rng.normal(0, 1, 6) * np.array([WRENCH_NOISE] * 3 + [WRENCH_NOISE / 10] * 3),
        }
        if self.render_enabled:
            for cam in self.render_next:
                obs[cam] = self.render(cam)
        return obs

    def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers = {}
