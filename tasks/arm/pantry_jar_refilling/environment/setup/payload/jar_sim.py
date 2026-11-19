"""Pantry jar refilling: the simulation of the counter (the environment's world, nothing of the reference).

Plain MuJoCo 3.16 on CPU. Two Franka Pandas (robosuite 1.5.2 model files only; robosuite is not imported) stand
side by side on the counter at y = +-0.28 m, facing +x, each with an operational-space controller at
10 Hz (or a joint step). On the counter: three screw-top jars in a row, each an open container (box-segment
rings, a 12 mm floor) with a coloured label band, and a tray of refill pieces of three kinds, one kind per label
colour. Frames: world z up, the counter's top at z = TABLE_Z; metres, radians, seconds; quaternions (w, x, y, z).

The thread: a jar's lid is a child body on a hinge (`lid{j}_turn`) and a slide (`lid{j}_lift`)
coupled by a joint equality (lift = pitch * turn / 2 pi). The hinge's frictionloss is the thread's running torque;
its lower limit is the seated lid, and the limit's force (sensor `seat{j}`) is the closing torque. A lid that has
turned `turns` turns leaves the thread and is swapped for a free body (`lidf{j}`) at the same pose and velocity. A
free lid resting centred and level on its own jar's neck and turned clockwise re-enters the thread at its start; a
lid does not catch on another jar's thread (the lids' diameters differ by 4 mm or more). A lid that has just come free
catches again only after it has left the neck (more than 3 mm off its axis or 4 mm above it).

`draw_unit(seed)` draws a unit (three jars' hidden properties and the label kinds), `draw_episode(seed, unit)` an
episode of it (the jars' order and places, the refill list, the tray's pieces), `World(unit, episode)` builds it.
"""

import math
import os

import mujoco
import numpy as np

TIMESTEP = 0.002
SUBSTEPS = 50                       # -> 10 Hz control
MAX_DPOS = 0.03                     # m per control step (operational-space action)
MAX_DROT = 0.2                      # rad per control step
MAX_DQ = 0.1                        # rad per control step (joint action)
MAX_ARM_SPEED = 8.0                 # rad/s: faster is an unstable simulation
MAX_OBJECT_SPEED = 5.0              # m/s: a jar, lid or piece faster than this is a blow-up, not a throw
IMPRATIO = 10.0
NOSLIP_ITERATIONS = 50
TORSION = 0.003                     # m, torsional friction (condim 4) of the pads and the objects
ARENA_BYTES = 96 * 2 ** 20
TABLE_Z = 0.80
TABLE_MU = 0.1                      # below every jar's mu: a jar's contact with the counter uses the jar's own
HOME_Q = np.array([0.0, -0.35, 0.0, -2.30, 0.0, 1.95, math.pi / 4])
BASES = {"l": np.array([0.0, 0.28]), "r": np.array([0.0, -0.28])}
WRENCH_NOISE = 0.3                  # N (force); torque noise is a tenth, N m
RGB_NOISE = 2.0

# the counter's layout
JAR_X = 0.55
JAR_YS = (0.17, 0.0, -0.17)         # the three places of the row; which jar stands where is drawn per episode
JAR_JITTER = (0.02, 0.01)           # m, each jar's place (x, y): neighbours stay at least 15 cm apart, so a hand turning
                                    # one lid (its body 10.4 cm either side of the fingers) clears a taller neighbour
TRAY_XY = np.array([0.32, -0.42])
TRAY_HALF = (0.075, 0.055)          # the tray's inner half sizes (x, y)
TRAY_RIM = 0.006                    # m, the rim's height (12 mm caught open fingers)
RING_SEGMENTS = 16
PITCH = 0.005                       # m per turn
LID_H = 0.020
LID_MASS = 0.03
LID_MU = 1.0
LID_RGBA = (0.78, 0.22, 0.16, 1)    # every lid looks alike but for its diameter
JAR_RGBA = (0.62, 0.72, 0.80, 1)
THREAD_CATCH = 0.30                 # rad: a free lid re-enters the thread this close to the thread's start
CATCH_RADIAL = 0.003                # m: the free lid's centre off the neck's axis at most this to catch
CATCH_TILT = 8.0                    # deg

KINDS = {                           # the refill pieces; a jar's label has its kind's colour
    "sugar": {"shape": "cube", "half": (0.008, 0.008, 0.008), "mass": 0.004, "rgba": (0.96, 0.95, 0.90, 1)},
    "mint": {"shape": "capsule", "half": (0.012, 0.007, 0.007), "mass": 0.003, "rgba": (0.25, 0.75, 0.40, 1)},
    "berry": {"shape": "capsule", "half": (0.012, 0.007, 0.007), "mass": 0.003, "rgba": (0.75, 0.20, 0.55, 1)},
}
KIND_NAMES = ("sugar", "mint", "berry")

# hidden properties: every unit, practice or scored, is drawn from UNIT_RANGES
LID_RADII = (0.027, 0.029, 0.031, 0.033, 0.035)     # m; a unit's three lids are three different ones
UNIT_RANGES = {"jar_h": (0.10, 0.14), "mass": (0.25, 0.90), "mu": (0.4, 0.8), "tau": (0.20, 0.80),
               "inset": (0.001, 0.003)}             # jar radius = lid radius - inset: the lid overhangs the body (with a
                                                    # wider body the turning fingertips rubbed its shoulder)
TURNS = (1.0, 1.5, 2.0, 2.5)

CAMERAS = {                         # name: (position, look-at, vertical fov deg, width, height)
    "overhead": ((0.62, 0.0, TABLE_Z + 1.10), (0.50, 0.0, TABLE_Z), 55.0, 640, 480),
    "front": ((1.20, 0.0, TABLE_Z + 0.50), (0.50, 0.0, TABLE_Z + 0.08), 45.0, 640, 480),
    "tray": ((TRAY_XY[0] + 0.12, TRAY_XY[1], TABLE_Z + 0.42), (TRAY_XY[0], TRAY_XY[1], TABLE_Z), 40.0, 640, 480),
}
WRIST = {"fovy": 75.0, "width": 320, "height": 240}
CAMERA_BITS = {"overhead": 1, "front": 2, "tray": 4, "wrist_l": 8, "wrist_r": 16}
ALL_CAMS = 31
MJ_CAMERA = {"overhead": "overhead", "front": "front", "tray": "tray", "wrist_l": "l_eye_in_hand",
             "wrist_r": "r_eye_in_hand"}


class RenderError(RuntimeError):
    """The renderer failed (a grader-side fault: the task's error, never the submission's)."""


# ----------------------------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------------------------
def robosuite_assets():
    import importlib.util
    spec = importlib.util.find_spec("robosuite")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("robosuite model files not installed")
    return os.path.join(next(iter(spec.submodule_search_locations)), "models", "assets")


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


def look_at(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r = r / np.linalg.norm(r) if np.linalg.norm(r) > 1e-6 else np.array([0.0, -1.0, 0.0])
    return np.column_stack([r, np.cross(r, f), -f])     # MuJoCo camera: looks along -z, y up


def intrinsics(fovy, w, h):
    f = (h / 2) / math.tan(math.radians(fovy) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def cam_pose(d, cid):
    """World pose of a MuJoCo camera in the OpenCV convention (x right, y down, z forward)."""
    R = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    return R, d.cam_xpos[cid].copy()


# ----------------------------------------------------------------------------------------------
# units and episodes
# ----------------------------------------------------------------------------------------------
def draw_unit(seed, ranges=None):
    """A unit: three jars' hidden properties and which kind each jar's label is."""
    rng = np.random.default_rng([int(seed) % 2 ** 32, int(seed) >> 32, 11])
    rg = dict(UNIT_RANGES, **(ranges or {}))
    radii = rng.choice(LID_RADII, 3, replace=False)
    kinds = [str(k) for k in rng.permutation(KIND_NAMES)]
    jars = []
    for j in range(3):
        lid_r = float(radii[j])
        jars.append({"kind": kinds[j], "lid_r": lid_r,
                     "jar_r": round(lid_r - float(rng.uniform(*rg["inset"])), 4),
                     "jar_h": round(float(rng.uniform(*rg["jar_h"])), 4),
                     "mass": round(float(rng.uniform(*rg["mass"])), 3),
                     "mu": round(float(rng.uniform(*rg["mu"])), 3),
                     "tau": round(float(rng.uniform(*rg["tau"])), 3),
                     "turns": float(rng.choice(TURNS))})
    return {"jars": jars}


def draw_episode(seed, unit):
    """An episode of `unit`: the jars' places (a permutation of the row and a jitter), the refill list (two of
    the three jars, 2-4 pieces each) and the tray's pieces (each listed kind's count plus one or two spare, and one or
    two of the unlisted kind)."""
    rng = np.random.default_rng([int(seed) % 2 ** 32, int(seed) >> 32, 29])
    order = [int(i) for i in rng.permutation(3)]          # order[place] = jar index
    places = {}
    for place, j in enumerate(order):
        places[j] = [JAR_X + float(rng.uniform(-JAR_JITTER[0], JAR_JITTER[0])),
                     JAR_YS[place] + float(rng.uniform(-JAR_JITTER[1], JAR_JITTER[1]))]
    listed = sorted(int(j) for j in rng.choice(3, 2, replace=False))
    refill = {str(unit["jars"][j]["kind"]): int(rng.integers(2, 5)) for j in listed}
    tray = []
    for j in range(3):
        kind = unit["jars"][j]["kind"]
        n = refill[kind] + int(rng.integers(1, 3)) if kind in refill else int(rng.integers(1, 3))
        tray += [kind] * n
    rng.shuffle(tray)
    # the pieces on a jittered grid in the tray (4 x 3 cells), no two in one cell
    cells = [(ix, iy) for ix in range(4) for iy in range(3)]
    pick = rng.permutation(len(cells))[:len(tray)]
    pieces = []
    for k, kind in zip(pick, tray):
        ix, iy = cells[int(k)]
        x = TRAY_XY[0] - TRAY_HALF[0] + 0.022 + ix * 0.036 + float(rng.uniform(-0.004, 0.004))
        y = TRAY_XY[1] - TRAY_HALF[1] + 0.020 + iy * 0.035 + float(rng.uniform(-0.004, 0.004))
        # lying along y (a capsule's long axis): a hand picking one has its fingers, and its long side, along x
        pieces.append({"kind": str(kind), "xy": [x, y], "yaw": float(math.pi / 2 + rng.uniform(-0.3, 0.3))})
    return {"places": {str(j): places[j] for j in range(3)}, "listed": listed, "refill": refill, "pieces": pieces}


# ----------------------------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------------------------
def _geom(body, name, gtype, size, pos, quat=(1, 0, 0, 0), mu=0.6, rgba=JAR_RGBA):
    gg = body.add_geom(name=name, type=gtype, size=list(size), pos=list(pos), quat=list(quat))
    gg.condim = 4
    gg.friction = [mu, TORSION, 0.0001]
    gg.rgba = list(rgba)
    return gg


def _ring(body, name, r_in, r_out, z0, z1, mu, rgba=JAR_RGBA, n=RING_SEGMENTS):
    half_t = r_out * math.tan(math.pi / n) + 0.0005
    rc = (r_in + r_out) / 2
    for k in range(n):
        th = 2 * math.pi * k / n
        _geom(body, f"{name}{k}", mujoco.mjtGeom.mjGEOM_BOX, [(r_out - r_in) / 2, half_t, (z1 - z0) / 2],
              [rc * math.cos(th), rc * math.sin(th), (z0 + z1) / 2], [math.cos(th / 2), 0, 0, math.sin(th / 2)],
              mu=mu, rgba=rgba)


def build(unit, episode):
    w = mujoco.MjSpec()
    w.compiler.degree = False
    w.compiler.autolimits = True
    w.option.timestep = TIMESTEP
    w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    w.option.impratio = IMPRATIO
    w.option.noslip_iterations = NOSLIP_ITERATIONS
    w.memory = ARENA_BYTES
    w.visual.headlight.ambient = [0.35, 0.35, 0.35]
    w.visual.headlight.diffuse = [0.55, 0.55, 0.55]
    w.visual.global_.offwidth, w.visual.global_.offheight = 640, 480
    wb = w.worldbody
    for pos, d in (((0.6, 0.0, TABLE_Z + 1.6), (0, 0, -1)), ((1.4, 0.8, TABLE_Z + 1.2), (-0.6, -0.4, -0.7)),
                   ((1.4, -0.8, TABLE_Z + 1.2), (-0.6, 0.4, -0.7))):
        wb.add_light(pos=list(pos), dir=list(d), castshadow=0, diffuse=[0.45, 0.45, 0.45])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.35, 0.33, 0.30, 1])
    # the counter's own friction is low: a contact takes the larger of its two geoms' frictions, so the jar's own
    # (the unit's `mu`) is what holds it on the counter; the pieces and lids have their own (and a priority)
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.50, 0.80, TABLE_Z / 2],
                pos=[0.35, 0.0, TABLE_Z / 2], rgba=[0.62, 0.55, 0.45, 1], friction=[TABLE_MU, 0.005, 0.0001])
    ra = robosuite_assets()
    for side, xy in BASES.items():
        robot = mujoco.MjSpec.from_file(os.path.join(ra, "robots", "panda", "robot.xml"))
        grip = mujoco.MjSpec.from_file(os.path.join(ra, "grippers", "panda_gripper.xml"))
        for cam in list(robot.cameras):
            if cam.name == "robotview":
                robot.delete(cam)
        robot.body("right_hand").add_frame().attach_body(grip.body("right_gripper"), "g_", "")
        wb.add_frame(pos=[xy[0], xy[1], TABLE_Z]).attach_body(robot.body("base"), f"{side}_", "")
    for name, (pos, tgt, fovy, _w, _h) in CAMERAS.items():
        wb.add_camera(name=name, pos=list(pos), fovy=fovy, quat=list(quat_from_mat(look_at(pos, tgt))))

    # the tray: a floor and a low rim
    tb = wb.add_body(name="tray", pos=[TRAY_XY[0], TRAY_XY[1], TABLE_Z])
    hx, hy = TRAY_HALF
    _geom(tb, "tray_floor", mujoco.mjtGeom.mjGEOM_BOX, [hx + 0.004, hy + 0.004, 0.002], [0, 0, 0.002],
          rgba=(0.30, 0.32, 0.36, 1))
    for k, (sx, sy, px, py) in enumerate(((0.004, hy + 0.008, hx + 0.004, 0), (0.004, hy + 0.008, -hx - 0.004, 0),
                                          (hx, 0.004, 0, hy + 0.004), (hx, 0.004, 0, -hy - 0.004))):
        _geom(tb, f"tray_rim{k}", mujoco.mjtGeom.mjGEOM_BOX, [sx, sy, TRAY_RIM / 2], [px, py, TRAY_RIM / 2],
              rgba=(0.30, 0.32, 0.36, 1))

    for j, jar in enumerate(unit["jars"]):
        xy = episode["places"][str(j)]
        neck_h = jar["turns"] * PITCH
        mouth = jar["lid_r"] - 0.008
        jb = wb.add_body(name=f"jar{j}", pos=[xy[0], xy[1], TABLE_Z + 0.0005])
        jb.add_freejoint(name=f"jar{j}_free")
        _geom(jb, f"jar{j}_floor", mujoco.mjtGeom.mjGEOM_CYLINDER, [jar["jar_r"], 0.006, 0], [0, 0, 0.006], mu=jar["mu"])
        # two crossed boxes over the floor (an octagon): a capsule dropped in sank through the cylinder alone
        hb = 0.707 * (jar["jar_r"] - 0.004)
        for k, yaw in enumerate((0.0, math.pi / 4)):
            _geom(jb, f"jar{j}_floor_box{k}", mujoco.mjtGeom.mjGEOM_BOX, [hb, hb, 0.006], [0, 0, 0.006],
                  [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)], mu=jar["mu"])
        _ring(jb, f"jar{j}_wall", jar["jar_r"] - 0.004, jar["jar_r"], 0.012, jar["jar_h"], jar["mu"])
        _ring(jb, f"jar{j}_shoulder", mouth, jar["jar_r"] - 0.004, jar["jar_h"] - 0.004, jar["jar_h"], jar["mu"])
        _ring(jb, f"jar{j}_neck", mouth, jar["lid_r"] - 0.004, jar["jar_h"], jar["jar_h"] + neck_h, jar["mu"])
        lb = jb.add_geom(name=f"jar{j}_label", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                         size=[jar["jar_r"] + 0.0008, 0.018, 0], pos=[0, 0, 0.55 * jar["jar_h"]],
                         rgba=list(KINDS[jar["kind"]]["rgba"]))
        lb.contype = lb.conaffinity = 0
        lb.mass = 0
        lid = jb.add_body(name=f"lid{j}", pos=[0, 0, jar["jar_h"] + LID_H / 2])
        lid.add_joint(name=f"lid{j}_turn", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                      range=[0, jar["turns"] * 2 * math.pi], limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                      frictionloss=jar["tau"], damping=0.002, armature=0.0002,
                      solref_limit=[0.004, 1], solimp_limit=[0.99, 0.999, 0.001, 0.5, 2],
                      solref_friction=[0.004, 1], solimp_friction=[0.999, 0.9999, 0.001, 0.5, 2])
        lid.add_joint(name=f"lid{j}_lift", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, 1], armature=0.0002)
        lf = wb.add_body(name=f"lidf{j}", pos=[1.5 + 0.2 * j, 1.5, 0.3])
        lf.add_freejoint(name=f"lidf{j}_free")
        lf.gravcomp = 1.0
        for body, name in ((lid, f"lid{j}_geom"), (lf, f"lidf{j}_geom")):
            gg = _geom(body, name, mujoco.mjtGeom.mjGEOM_CYLINDER, [jar["lid_r"], LID_H / 2, 0], [0, 0, 0],
                       mu=LID_MU, rgba=LID_RGBA)
            gg.priority = 1
            gg.mass = LID_MASS
        w.geom(f"lidf{j}_geom").contype = 0
        w.geom(f"lidf{j}_geom").conaffinity = 0
        w.geom(f"lidf{j}_geom").rgba = list(LID_RGBA[:3]) + [0]
        w.add_equality(name=f"thread{j}", type=mujoco.mjtEq.mjEQ_JOINT, name1=f"lid{j}_lift", name2=f"lid{j}_turn",
                       data=[0, PITCH / (2 * math.pi), 0, 0, 0] + [0] * 6,
                       solref=[0.004, 1], solimp=[0.999, 0.9999, 0.001, 0.5, 2])
        w.add_sensor(name=f"seat{j}", type=mujoco.mjtSensor.mjSENS_JOINTLIMITFRC,
                     objtype=mujoco.mjtObj.mjOBJ_JOINT, objname=f"lid{j}_turn")

    for i, pc in enumerate(episode["pieces"]):
        k = KINDS[pc["kind"]]
        pb = wb.add_body(name=f"piece{i}", pos=[pc["xy"][0], pc["xy"][1], TABLE_Z + 0.004 + k["half"][2] + 0.0005],
                         quat=[math.cos(pc["yaw"] / 2), 0, 0, math.sin(pc["yaw"] / 2)])
        pb.add_freejoint(name=f"piece{i}_free")
        if k["shape"] == "cube":
            gg = _geom(pb, f"piece{i}_geom", mujoco.mjtGeom.mjGEOM_BOX, k["half"], [0, 0, 0], mu=0.8, rgba=k["rgba"])
        else:                                   # a capsule lying along x
            gg = _geom(pb, f"piece{i}_geom", mujoco.mjtGeom.mjGEOM_CAPSULE, [k["half"][2], k["half"][0] - k["half"][2], 0],
                       [0, 0, 0], [0.7071068, 0, 0.7071068, 0], mu=0.8, rgba=k["rgba"])
        gg.mass = k["mass"]
        gg.priority = 1

    m = w.compile()
    for j, jar in enumerate(unit["jars"]):
        b = m.body(f"jar{j}").id
        mj, r, h = jar["mass"], jar["jar_r"], jar["jar_h"]
        m.body_mass[b] = mj
        m.body_ipos[b] = [0, 0, 0.4 * h]
        m.body_iquat[b] = [1, 0, 0, 0]
        m.body_inertia[b] = [mj * (3 * r * r + h * h) / 12] * 2 + [mj * r * r / 2]
    for side in BASES:
        for f in ("finger1", "finger2"):
            gid = m.geom(f"{side}_g_{f}_pad_collision").id
            m.geom_condim[gid] = 4
            m.geom_friction[gid, 1] = TORSION
    return m


# ----------------------------------------------------------------------------------------------
# the arms
# ----------------------------------------------------------------------------------------------
class Arm:
    """One Panda: indices, the operational-space controller, the joint step, the wrist sensor."""

    def __init__(self, m, d, p):
        self.m, self.d, self.p = m, d, p
        self.jq = np.array([m.joint(f"{p}_joint{i}").qposadr[0] for i in range(1, 8)])
        self.jv = np.array([m.joint(f"{p}_joint{i}").dofadr[0] for i in range(1, 8)])
        self.act = np.array([m.actuator(f"{p}_torq_j{i}").id for i in range(1, 8)])
        self.gact = [m.actuator(f"{p}_g_gripper_finger_joint1").id, m.actuator(f"{p}_g_gripper_finger_joint2").id]
        self.gq = [m.joint(f"{p}_g_finger_joint1").qposadr[0], m.joint(f"{p}_g_finger_joint2").qposadr[0]]
        self.site = m.site(f"{p}_g_grip_site").id
        self.ft_site = m.site(f"{p}_g_ft_frame").id
        self.f_adr = m.sensor(f"{p}_g_force_ee").adr[0]
        self.t_adr = m.sensor(f"{p}_g_torque_ee").adr[0]
        self.pads = (m.geom(f"{p}_g_finger1_pad_collision").id, m.geom(f"{p}_g_finger2_pad_collision").id)
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))
        self.q_null = HOME_Q.copy()
        self.grip_target = 0.04
        self.mode = "osc"
        self.q_goal = None

    def init_targets(self):
        d = self.d
        self.target_pos = d.site_xpos[self.site].copy()
        self.target_R = d.site_xmat[self.site].reshape(3, 3).copy()

    def pos(self):
        return self.d.site_xpos[self.site].copy()

    def R(self):
        return self.d.site_xmat[self.site].reshape(3, 3).copy()

    def gripper(self):
        return float(self.d.qpos[self.gq[0]] - self.d.qpos[self.gq[1]])

    def wrench_world(self):
        d = self.d
        R = d.site_xmat[self.ft_site].reshape(3, 3)
        return np.concatenate([R @ d.sensordata[self.f_adr:self.f_adr + 3], R @ d.sensordata[self.t_adr:self.t_adr + 3]])

    def command(self, a):
        """Set this control step's command from one arm's action (8, 15 or 9 numbers, already checked)."""
        d = self.d
        if len(a) == 9:
            q = d.qpos[self.jq].copy()
            self.q_goal = q + np.clip(np.asarray(a[:7], float) - q, -MAX_DQ, MAX_DQ)
            self.q_null = np.asarray(a[:7], float).copy()
            self.mode = "joint"
            grip = a[7]
        else:
            if len(a) == 15:
                self.q_null = np.asarray(a[8:15], float).copy()
            dp, dr = np.asarray(a[0:3], float), np.asarray(a[3:6], float)
            n = np.linalg.norm(dp)
            if n > MAX_DPOS:
                dp = dp * (MAX_DPOS / n)
            n = np.linalg.norm(dr)
            if n > MAX_DROT:
                dr = dr * (MAX_DROT / n)
            self.target_pos = d.site_xpos[self.site] + dp
            self.target_R = axis_angle_mat(dr) @ d.site_xmat[self.site].reshape(3, 3)
            self.mode = "osc"
            grip = a[6]
        self.grip_target = 0.04 * (1.0 - float(np.clip(grip, -1, 1))) / 2.0

    def torque(self):
        m, d = self.m, self.d
        if self.mode == "joint":
            return 200.0 * (self.q_goal - d.qpos[self.jq]) - 20.0 * d.qvel[self.jv] + d.qfrc_bias[self.jv]
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


# ----------------------------------------------------------------------------------------------
# the world
# ----------------------------------------------------------------------------------------------
class World:
    def __init__(self, unit, episode):
        self.unit, self.episode = unit, episode
        self.m = m = build(unit, episode)
        self.d = d = mujoco.MjData(m)
        self.arms = {s: Arm(m, d, s) for s in BASES}
        self.nj = len(unit["jars"])
        J = range(self.nj)
        self.jar = [m.body(f"jar{j}").id for j in J]
        self.lid = [m.body(f"lid{j}").id for j in J]
        self.lidf = [m.body(f"lidf{j}").id for j in J]
        self.turn_q = [m.joint(f"lid{j}_turn").qposadr[0] for j in J]
        self.turn_v = [m.joint(f"lid{j}_turn").dofadr[0] for j in J]
        self.lift_q = [m.joint(f"lid{j}_lift").qposadr[0] for j in J]
        self.lift_v = [m.joint(f"lid{j}_lift").dofadr[0] for j in J]
        self.lidf_q = [m.joint(f"lidf{j}_free").qposadr[0] for j in J]
        self.lidf_v = [m.joint(f"lidf{j}_free").dofadr[0] for j in J]
        self.g_lid = [m.geom(f"lid{j}_geom").id for j in J]
        self.g_lidf = [m.geom(f"lidf{j}_geom").id for j in J]
        self.seat_adr = [m.sensor(f"seat{j}").adr[0] for j in J]
        self.turn_max = [jar["turns"] * 2 * math.pi for jar in unit["jars"]]
        self.pieces = [m.body(f"piece{i}").id for i in range(len(episode["pieces"]))]
        self.engaged = [True] * self.nj
        self.armed = [False] * self.nj           # a lid that came free re-enters the thread only after leaving the neck
        self.opened = [False] * self.nj          # the lid has left the thread at least once
        self.seat_peak = [0.0] * self.nj         # the largest closing torque since the lid last entered the thread
        self.events = []
        self.tipped = set()
        self.t = 0
        mujoco.mj_resetData(m, d)
        for a in self.arms.values():
            d.qpos[a.jq] = HOME_Q
            d.qpos[a.gq[0]], d.qpos[a.gq[1]] = 0.04, -0.04
        mujoco.mj_forward(m, d)
        for _ in range(250):                      # settle the arms and let the objects come to rest
            for a in self.arms.values():
                d.ctrl[a.act] = 200.0 * (HOME_Q - d.qpos[a.jq]) - 20.0 * d.qvel[a.jv] + d.qfrc_bias[a.jv]
                d.ctrl[a.gact] = [0.04, -0.04]
            mujoco.mj_step(m, d)
            self._park()
        for a in self.arms.values():
            a.init_targets()
        self.seat_peak = [0.0] * self.nj
        self.jar_start = [d.xpos[b].copy() for b in self.jar]
        self.piece_start = [d.xpos[b].copy() for b in self.pieces]

    # --- the thread ------------------------------------------------------------------------------
    def _park(self):
        d = self.d
        for j in range(self.nj):
            if self.engaged[j]:
                q = self.lidf_q[j]
                d.qpos[q:q + 7] = [1.5 + 0.2 * j, 1.5, 0.3, 1, 0, 0, 0]
                d.qvel[self.lidf_v[j]:self.lidf_v[j] + 6] = 0

    def _show(self, j, engaged):
        m = self.m
        on, off = (self.g_lid[j], self.g_lidf[j]) if engaged else (self.g_lidf[j], self.g_lid[j])
        m.geom_contype[on] = m.geom_conaffinity[on] = 1
        m.geom_contype[off] = m.geom_conaffinity[off] = 0
        m.geom_rgba[on, 3], m.geom_rgba[off, 3] = 1.0, 0.0
        for g, v in ((on, 1), (off, 0)):         # the broadphase reads the body's flags
            b = m.geom_bodyid[g]
            m.body_contype[b] = m.body_conaffinity[b] = v
        m.body_gravcomp[self.lidf[j]] = 1.0 if engaged else 0.0

    def _release(self, j):
        m, d = self.m, self.d
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.lid[j], vel, 0)
        R = d.xmat[self.lid[j]].reshape(3, 3)
        q, v = self.lidf_q[j], self.lidf_v[j]
        d.qpos[q:q + 3] = d.xpos[self.lid[j]]
        d.qpos[q + 3:q + 7] = d.xquat[self.lid[j]]
        d.qvel[v:v + 3] = vel[3:]
        d.qvel[v + 3:v + 6] = R.T @ vel[:3]
        self.engaged[j] = False
        self.armed[j] = False
        self.opened[j] = True
        self._show(j, False)
        mujoco.mj_forward(m, d)
        self.events.append(("released", j, self.t))

    def _try_catch(self, j):
        m, d = self.m, self.d
        jar = self.unit["jars"][j]
        Rj = d.xmat[self.jar[j]].reshape(3, 3)
        Rf = d.xmat[self.lidf[j]].reshape(3, 3)
        p = Rj.T @ (d.xpos[self.lidf[j]] - d.xpos[self.jar[j]])
        top = jar["jar_h"] + jar["turns"] * PITCH
        if np.hypot(p[0], p[1]) > CATCH_RADIAL or not (-0.002 < p[2] - LID_H / 2 - top < 0.004):
            self.armed[j] = True               # the lid has left the neck: it may catch again when set back on
            return
        if not self.armed[j]:
            return
        Rrel = Rj.T @ Rf
        if math.degrees(math.acos(max(-1.0, min(1.0, Rrel[2, 2])))) > CATCH_TILT:
            return
        vel, jv = np.zeros(6), np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.lidf[j], vel, 0)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.jar[j], jv, 0)
        wz = float((Rj.T @ (vel[:3] - jv[:3]))[2])
        if wz > -0.05:                          # the thread catches only while the lid turns clockwise
            return
        yaw = math.atan2(Rrel[1, 0], Rrel[0, 0])
        tm = self.turn_max[j]
        a = tm - ((tm - yaw) % (2 * math.pi))
        if a < tm - THREAD_CATCH:
            return
        d.qpos[self.turn_q[j]] = min(a, tm - 0.03)          # below the release angle: no flip-flop
        d.qpos[self.lift_q[j]] = PITCH * d.qpos[self.turn_q[j]] / (2 * math.pi)
        d.qvel[self.turn_v[j]] = wz
        d.qvel[self.lift_v[j]] = PITCH * wz / (2 * math.pi)
        self.engaged[j] = True
        self.seat_peak[j] = 0.0
        self._show(j, True)
        self._park()
        mujoco.mj_forward(m, d)
        self.events.append(("caught", j, self.t))

    # --- stepping --------------------------------------------------------------------------------
    def step(self, cmds):
        """One control step. `cmds`: {side: action list (8, 15 or 9 numbers)}; a side left out keeps its command."""
        m, d = self.m, self.d
        for s, a in cmds.items():
            self.arms[s].command(a)
        for _ in range(SUBSTEPS):
            for a in self.arms.values():
                d.ctrl[a.act] = a.torque()
                d.ctrl[a.gact] = [a.grip_target, -a.grip_target]
            mujoco.mj_step(m, d)
            self._park()
            for j in range(self.nj):
                if self.engaged[j]:
                    self.seat_peak[j] = max(self.seat_peak[j], abs(float(d.sensordata[self.seat_adr[j]])))
                    if d.qpos[self.turn_q[j]] >= self.turn_max[j] - 0.005 and d.qvel[self.turn_v[j]] > 0:
                        self._release(j)
                else:
                    self._try_catch(j)
        self.t += 1
        for j in range(self.nj):                   # a record for debugging; nothing scores it
            if j not in self.tipped and self.jar_tilt_deg(j) > 15.0:
                self.tipped.add(j)
                self.events.append(("tipped", j, self.t))
        if self.unstable():
            raise FloatingPointError("unstable")

    def unstable(self):
        d = self.d
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return True
        if any(np.max(np.abs(d.qvel[a.jv])) > MAX_ARM_SPEED for a in self.arms.values()):
            return True
        bodies = self.jar + self.pieces + [self.lidf[j] for j in range(self.nj) if not self.engaged[j]]
        v = np.zeros(6)
        for b in bodies:
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            if np.linalg.norm(v[3:]) > MAX_OBJECT_SPEED:
                return True
        return False

    # --- state the grader (and the privileged reference) reads ------------------------------------
    def lid_body(self, j):
        return self.lid[j] if self.engaged[j] else self.lidf[j]

    def lid_pos(self, j):
        return self.d.xpos[self.lid_body(j)].copy()

    def jar_pos(self, j):
        return self.d.xpos[self.jar[j]].copy()

    def jar_R(self, j):
        return self.d.xmat[self.jar[j]].reshape(3, 3).copy()

    def jar_tilt_deg(self, j):
        R = self.d.xmat[self.jar[j]].reshape(3, 3)
        return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))

    def turn(self, j):
        return float(self.d.qpos[self.turn_q[j]]) if self.engaged[j] else None

    def seat_now(self, j):
        """The closing torque this instant (N m): the seated lid's limit force."""
        return abs(float(self.d.sensordata[self.seat_adr[j]])) if self.engaged[j] else 0.0

    def seated(self, j):
        return bool(self.engaged[j] and self.d.qpos[self.turn_q[j]] < 0.02)

    def neck_top(self, j):
        jar = self.unit["jars"][j]
        return self.jar_pos(j) + self.jar_R(j) @ np.array([0, 0, jar["jar_h"] + jar["turns"] * PITCH])

    def piece_in_jar(self, i):
        """The jar piece i is in (its centre inside the wall, between the floor and the shoulder), or None."""
        p = self.d.xpos[self.pieces[i]]
        for j, jar in enumerate(self.unit["jars"]):
            pl = self.jar_R(j).T @ (p - self.jar_pos(j))
            if np.hypot(pl[0], pl[1]) < jar["jar_r"] - 0.004 and 0.006 < pl[2] < jar["jar_h"]:
                return j
        return None

    def piece_in_tray(self, i):
        p = self.d.xpos[self.pieces[i]]
        return bool(abs(p[0] - TRAY_XY[0]) < TRAY_HALF[0] and abs(p[1] - TRAY_XY[1]) < TRAY_HALF[1]
                    and p[2] < TABLE_Z + 0.03)

    def piece_on_counter(self, i):
        p = self.d.xpos[self.pieces[i]]
        return bool(p[2] > TABLE_Z - 0.01 and -0.15 < p[0] < 0.85 and abs(p[1]) < 0.80)


# ----------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------
_SCENE_OPT = None


def scene_option():
    global _SCENE_OPT
    if _SCENE_OPT is None:
        _SCENE_OPT = mujoco.MjvOption()
        _SCENE_OPT.sitegroup[:] = 0            # no site markers in the images
        _SCENE_OPT.geomgroup[:] = 1
    return _SCENE_OPT


def camera_size(cam):
    if cam.startswith("wrist"):
        return WRIST["width"], WRIST["height"], WRIST["fovy"]
    _p, _t, fovy, w, h = CAMERAS[cam]
    return w, h, fovy


def render(m, d, cam, rng):
    w, h, _fovy = camera_size(cam)
    try:
        r = mujoco.Renderer(m, h, w)
        try:
            r.update_scene(d, camera=MJ_CAMERA[cam], scene_option=scene_option())
            img = r.render().astype(np.float32)
        finally:
            r.close()
    except Exception as exc:  # noqa: BLE001 - whatever the renderer raises is reported as one error
        raise RenderError("{} camera ({}x{}): {}: {}".format(cam, w, h, type(exc).__name__, str(exc)[:200])) from exc
    img += rng.normal(0, RGB_NOISE, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)
