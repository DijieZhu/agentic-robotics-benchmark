"""Hotpot serving: the simulation of the counter (the environment's world, nothing of the reference).

Plain MuJoCo 3.16 on CPU. Two Franka Pandas (robosuite 1.5.2 model files only; robosuite is not imported) stand
side by side on the counter at y = +-0.28 m, facing +x, each with an operational-space controller at 10 Hz (or a
joint step). On the counter: a burner with a knob and a braiser of clear broth on it (a grip tab on each side), a
slotted ladle standing in the broth, its handle leaning toward -y; four small dishes of raw ingredients (fish balls,
shrimp, tofu puffs, mushroom caps), each with a tab, on the left; three bowls in a row in front.
Frames: world z up, the counter's top at z = TABLE_Z; metres, radians, seconds; quaternions (w, x, y, z).

The broth is not made of particles. Each vessel (the pot and the bowls) holds a volume of broth, and `Liquid`
computes the rest from the vessel's pose: the horizontal surface that holds that volume, the outflow over the rim
where the surface stands above it (a weir: q = (2/3) Cd sqrt(2 g) d^1.5 per metre of rim), the stream (parcels
launched from the overflowing rim that fly ballistically into a vessel's mouth or onto the counter), the broth's
weight on its vessel (an applied force at the liquid's centroid) and the buoyancy and drag on pieces under a surface.
The surface and the stream are drawn as visual geoms that collide with nothing.

`Heat` keeps the temperatures: the pot's broth (heated by the burner at the knob's setting, cooled to the room and by
the cold pieces in it, at most BOIL) and every piece's core (a first-order lag to the broth it is in, with the kind's
time constant TAU, scaled by the unit's piece size; cooling slowly in the air). A piece is cooked once its core has
reached the kind's T_COOK, and overcooked once it has spent the kind's WINDOW in the pot's broth since. Its colour
follows (raw -> cooked -> overcooked), and a fish ball floats once cooked.

`draw_unit(seed)` draws a unit (the pot's, the broth's and the burner's hidden properties, the pieces' size and
friction), `draw_episode(seed, unit)` an episode of it (which bowl stands where, the order, the pot's place, which
dish holds which kind, the pieces on the dishes), `World(unit, episode)` builds it.
"""

import math
import os

import mujoco
import numpy as np

TIMESTEP = 0.002
SUBSTEPS = 50                       # -> 10 Hz control
LIQUID_EVERY = 5                    # substeps per update of the liquid (100 Hz)
MAX_DPOS = 0.03                     # m per control step (operational-space action)
MAX_DROT = 0.2                      # rad per control step
MAX_DQ = 0.1                        # rad per control step (joint action)
MAX_ARM_SPEED = 8.0                 # rad/s: faster is an unstable simulation
MAX_OBJECT_SPEED = 5.0              # m/s
IMPRATIO = 10.0
NOSLIP_ITERATIONS = 50
TORSION = 0.003                     # m, torsional friction (condim 4) of the pads and the objects
ARENA_BYTES = 96 * 2 ** 20
TABLE_Z = 0.80
TABLE_MU = 0.1
HOME_Q = np.array([0.0, -0.35, 0.0, -2.30, 0.0, 1.95, math.pi / 4])
BASES = {"l": np.array([0.0, 0.28]), "r": np.array([0.0, -0.28])}
WRENCH_NOISE = 0.3                  # N (force); torque noise is a tenth, N m
RGB_NOISE = 2.0
G = 9.81
RHO = 1030.0                        # kg/m^3, the broth
DRAG = 0.6                          # N s/m on a fully submerged piece (linear)

# heat
ROOM = 20.0                         # deg C: the room, the raw pieces
BROTH_T0 = 80.0                     # deg C: the broth at the start
BOIL = 100.0
BROTH_C = 4000.0                    # J/(kg K), the broth's heat capacity; 1 l = 1 kg here
PIECE_C = 3600.0                    # J/(kg K), the pieces'
POT_LOSS = 9.0                      # W/K: the pot's loss to the room
KNOB_MAX = math.pi / 2              # rad: the knob's range, off (0) .. full power (pi/2)
KNOB_OFF = 0.05                     # rad: below this the burner is off
AIR_TAU = 600.0                     # s: a piece's core cools toward the room this slowly outside a broth

# the stove and the pot
STOVE_XY = np.array([0.45, 0.0])
STOVE_R = 0.14
STOVE_H = 0.004
POT_R = 0.125                       # inner radius
POT_WALL = 0.006
POT_FLOOR = 0.008
POT_DEPTH = 0.095                   # inner depth, floor top to rim
POT_SEGMENTS = 24
TAB_HALF = (0.016, 0.030, 0.016)    # a grip tab: 32 mm along x (the fingers close across it), out along y
TAB_Z = 0.065                       # the tabs' centre above the floor's top
POT_MU = 0.6
TAB_MU = 1.0
POT_RGBA = (0.72, 0.73, 0.75, 1)
TAB_RGBA = (0.15, 0.15, 0.16, 1)
SOUP_RGBA = (0.85, 0.62, 0.30, 0.40)          # clear broth: what is under its surface shows through

# the knob (behind the pot, on the right arm's side): a cylinder turning about z with a fin to pinch
KNOB_XY = np.array([0.25, -0.22])
KNOB_R, KNOB_H = 0.018, 0.020
KNOB_FIN = (0.004, 0.016, 0.012)    # half sizes: thin along x at angle 0, across y
KNOB_BASE_RGBA = (0.10, 0.10, 0.11, 1)
KNOB_RGBA = (0.85, 0.20, 0.15, 1)

# the ingredients: kind -> shape, half sizes (at size scale 1), density, cooking (time constant at size 1, the core
# temperature at which it is cooked, how long it may then stay in the pot's broth), colours raw / cooked / overcooked
KINDS = ("fishball", "shrimp", "tofu", "mushroom")
PIECE = {
    "fishball": {"shape": "sphere", "size": (0.012,), "rho": 1080.0, "rho_cooked": 960.0, "tau": 90.0, "t_cook": 72.0,
                 "window": 270.0, "rgba": ((0.80, 0.80, 0.78), (0.97, 0.96, 0.92), (0.80, 0.72, 0.52))},
    "shrimp": {"shape": "capsule", "size": (0.0075, 0.008), "rho": 1070.0, "tau": 50.0, "t_cook": 63.0,
               "window": 240.0, "rgba": ((0.55, 0.58, 0.60), (0.96, 0.52, 0.36), (0.70, 0.33, 0.22))},
    "tofu": {"shape": "box", "size": (0.012, 0.012, 0.012), "rho": 550.0, "tau": 150.0, "t_cook": 70.0,
             "window": 360.0, "rgba": ((0.96, 0.90, 0.70), (0.90, 0.72, 0.38), (0.62, 0.42, 0.20))},
    "mushroom": {"shape": "ellipsoid", "size": (0.014, 0.014, 0.008), "rho": 700.0, "tau": 60.0, "t_cook": 65.0,
                 "window": 240.0, "rgba": ((0.86, 0.80, 0.68), (0.55, 0.42, 0.30), (0.30, 0.22, 0.15))},
}

# the dishes of raw ingredients (left of the pot): a shallow plate with a low rim and a grip tab toward -x
DISH_XY = ((0.36, 0.34), (0.52, 0.34), (0.36, 0.49), (0.52, 0.49))
DISH_R = 0.050                      # inner radius
DISH_FLOOR = 0.004
DISH_RIM_H = 0.002
DISH_WALL = 0.003
DISH_SEGMENTS = 16
DISH_TAB = (0.022, 0.012, 0.004)    # half sizes: out along -x from the rim, flat
DISH_TAB_Z = 0.022                  # the tab's centre above the dish's underside (a raised handle)
DISH_MASS = 0.12
DISH_MU = 0.5
DISH_RGBA = (0.90, 0.90, 0.88, 1)

# the bowls
BOWL_X = 0.72
BOWL_YS = (0.15, 0.0, -0.15)
BOWL_JITTER = 0.01
BOWL_R0, BOWL_R1 = 0.040, 0.065     # inner radius at the floor and at the rim
BOWL_DEPTH = 0.060
BOWL_FLOOR = 0.006
BOWL_WALL = 0.004
BOWL_SEGMENTS = 16
BOWL_MASS = 0.35
BOWL_MU = 0.7
BOWL_COLOURS = {"white": (0.93, 0.93, 0.90, 1), "blue": (0.20, 0.40, 0.80, 1), "yellow": (0.95, 0.80, 0.20, 1)}
BOWL_NAMES = ("white", "blue", "yellow")

# the ladle, a slotted skimmer (body frame: the dish's floor centred at the origin, level; the handle leaves the dish's
# rim at -x and rises toward -x at LADLE_BETA from the dish's plane; the rim runs round the back and the sides, the
# front, +x, is open over 127 deg). It starts standing in the pot, its front toward +y and its handle leaning on the
# rim toward -y.
LADLE_DISH_R = 0.045
LADLE_RIM_H = 0.015
LADLE_BETA = math.radians(70.0)
LADLE_HANDLE = (0.125, 0.009, 0.005)          # half sizes (length, width, thickness)
LADLE_IN_POT = (0.0, -0.035)                  # the dish's start, relative to the pot's axis; its front toward +y,
LADLE_YAW = math.pi / 2                       # the handle leaning toward -y (the right arm)
LADLE_MASS = 0.15
LADLE_RGBA = (0.80, 0.80, 0.82, 1)
GRIP_KP = 4000.0                    # the fingers' position gain, N/m, and their force limit, N (the Franka Hand's
GRIP_FORCE = 70.0                   # continuous grasp force; robosuite's model has 1000 and 20)
PAD_TORSION = 0.010                 # m, the finger pads' torsional friction (a 1 cm contact patch)
GRIP_PAST = 0.03                    # m: grip +1 aims each finger this far past closed, so a closed hand squeezes
                                    # what it holds with GRIP_FORCE whatever its size

UNIT_RANGES = {"pot_mass": (1.4, 2.4), "soup_l": (2.2, 3.0), "cd": (0.35, 0.75), "burner_kw": (2.0, 3.5),
               "piece_scale": (0.85, 1.15), "piece_mu": (0.4, 0.8)}
PIECES_PER_BOWL = (2, 3)             # an order asks for 2-3 pieces per bowl (6-9 in all), each of a kind at random
PIECES_PER_KIND_MAX = 4              # no kind listed more than this in all
PIECE_EXTRA = 1                      # each dish holds one more of its kind than the order asks for

CAMERAS = {                         # name: (position, look-at, vertical fov deg, width, height)
    "overhead": ((0.40, 0.0, TABLE_Z + 1.00), (0.55, 0.0, TABLE_Z), 60.0, 640, 480),
    "front": ((1.30, -0.10, TABLE_Z + 0.50), (0.52, -0.05, TABLE_Z + 0.05), 50.0, 640, 480),
}
WRIST = {"fovy": 75.0, "width": 320, "height": 240}
CAMERA_BITS = {"overhead": 1, "front": 2, "wrist_l": 4, "wrist_r": 8}
ALL_CAMS = 15
HIDDEN_GROUP = 4
MJ_CAMERA = {"overhead": "overhead", "front": "front", "wrist_l": "l_eye_in_hand", "wrist_r": "r_eye_in_hand"}
STREAM_GEOMS = 10


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


def _rng(seed, tag):
    return np.random.default_rng([int(seed) % 2 ** 32, int(seed) >> 32, tag])


# ----------------------------------------------------------------------------------------------
# units and episodes
# ----------------------------------------------------------------------------------------------
def draw_unit(seed, ranges=None):
    """A unit: the pot's empty mass, the broth's volume and discharge coefficient, the burner's full power, the pieces'
    size scale and friction."""
    rng = _rng(seed, 11)
    rg = dict(UNIT_RANGES, **(ranges or {}))
    return {"pot_mass": round(float(rng.uniform(*rg["pot_mass"])), 3),
            "soup_l": round(float(rng.uniform(*rg["soup_l"])), 3),
            "cd": round(float(rng.uniform(*rg["cd"])), 3),
            "burner_kw": round(float(rng.uniform(*rg["burner_kw"])), 3),
            "piece_scale": round(float(rng.uniform(*rg["piece_scale"])), 3),
            "piece_mu": round(float(rng.uniform(*rg["piece_mu"])), 3)}


def draw_episode(seed, unit):
    """An episode of `unit`: which bowl stands where (and a jitter), the order (each bowl's pieces, 2-3,
    PIECES_PER_BOWL, each of a kind at random, no kind more than PIECES_PER_KIND_MAX in all; and broth 100-250 ml in
    10 ml steps), the pot's place, which dish holds which kind, and the pieces on each dish (the order's count of its
    kind and PIECE_EXTRA more)."""
    rng = _rng(seed, 29)
    perm = [int(i) for i in rng.permutation(3)]
    bowls = {}
    for place, b in enumerate(perm):
        bowls[BOWL_NAMES[b]] = [BOWL_X + float(rng.uniform(-BOWL_JITTER, BOWL_JITTER)),
                                BOWL_YS[place] + float(rng.uniform(-BOWL_JITTER, BOWL_JITTER))]
    while True:
        order = {}
        for b in range(3):
            n = int(rng.integers(PIECES_PER_BOWL[0], PIECES_PER_BOWL[1] + 1))
            k = [int(x) for x in rng.integers(0, len(KINDS), n)]
            order[BOWL_NAMES[b]] = {"items": {kd: k.count(j) for j, kd in enumerate(KINDS)},
                                    "soup_ml": int(rng.integers(10, 26)) * 10}
        if all(sum(o["items"][kd] for o in order.values()) <= PIECES_PER_KIND_MAX for kd in KINDS):
            break
    pot_xy = [float(STOVE_XY[0] + rng.uniform(-0.01, 0.01)), float(STOVE_XY[1] + rng.uniform(-0.01, 0.01))]
    pot_yaw = float(rng.uniform(-0.05, 0.05))
    dish_of = [KINDS[int(j)] for j in rng.permutation(len(KINDS))]          # dish k holds kind dish_of[k]
    spots = [(0.0, 0.0), (0.024, 0.0), (-0.024, 0.0), (0.0, 0.024), (0.0, -0.024)]
    pieces, kinds = [], []
    for k, kd in enumerate(dish_of):
        n = sum(o["items"][kd] for o in order.values()) + PIECE_EXTRA
        for j in [int(x) for x in rng.permutation(len(spots))[:n]]:
            pieces.append([k, spots[j][0] + float(rng.uniform(-0.003, 0.003)),
                           spots[j][1] + float(rng.uniform(-0.003, 0.003)), float(rng.uniform(0, 2 * math.pi))])
            kinds.append(kd)
    dishes = [[x + float(rng.uniform(-0.01, 0.01)), y + float(rng.uniform(-0.01, 0.01))] for x, y in DISH_XY]
    return {"bowls": bowls, "order": order, "pot": pot_xy + [pot_yaw], "dishes": dishes, "dish_of": dish_of,
            "pieces": pieces, "kinds": kinds}


def piece_size(kind, scale):
    return tuple(x * scale for x in PIECE[kind]["size"])


def piece_volume(kind, scale):
    sh, sz = PIECE[kind]["shape"], piece_size(kind, scale)
    if sh == "sphere":
        return 4.0 / 3.0 * math.pi * sz[0] ** 3
    if sh == "capsule":
        return math.pi * sz[0] ** 2 * (2 * sz[1] + 4.0 / 3.0 * sz[0])
    if sh == "box":
        return 8.0 * sz[0] * sz[1] * sz[2]
    return 4.0 / 3.0 * math.pi * sz[0] * sz[1] * sz[2]


def piece_rest_half_height(kind, scale):
    """Half its height lying on a plate (a capsule and a cylinder lie flat / on its round face)."""
    sh, sz = PIECE[kind]["shape"], piece_size(kind, scale)
    if sh in ("sphere", "capsule"):
        return sz[0]
    return sz[2]


# ----------------------------------------------------------------------------------------------
# the model
# ----------------------------------------------------------------------------------------------
def _geom(body, name, gtype, size, pos, quat=(1, 0, 0, 0), mu=0.6, rgba=POT_RGBA):
    gg = body.add_geom(name=name, type=gtype, size=list(size), pos=list(pos), quat=list(quat))
    gg.condim = 4
    gg.friction = [mu, TORSION, 0.0001]
    gg.rgba = list(rgba)
    return gg


def _yaw_quat(th):
    return [math.cos(th / 2), 0, 0, math.sin(th / 2)]


def _visual(body, name, gtype, size, rgba):
    gg = body.add_geom(name=name, type=gtype, size=list(size), rgba=list(rgba))
    gg.contype = gg.conaffinity = 0
    gg.mass = 0
    gg.group = 2
    return gg


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
    w.visual.headlight.diffuse = [0.45, 0.45, 0.45]
    w.visual.global_.offwidth, w.visual.global_.offheight = 640, 480
    wb = w.worldbody
    for pos, d in (((0.6, 0.0, TABLE_Z + 1.6), (0, 0, -1)), ((1.4, 0.8, TABLE_Z + 1.2), (-0.6, -0.4, -0.7)),
                   ((1.4, -0.8, TABLE_Z + 1.2), (-0.6, 0.4, -0.7))):
        wb.add_light(pos=list(pos), dir=list(d), castshadow=0, diffuse=[0.45, 0.45, 0.45])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.35, 0.33, 0.30, 1])
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.50, 0.80, TABLE_Z / 2],
                pos=[0.35, 0.0, TABLE_Z / 2], rgba=[0.62, 0.55, 0.45, 1], friction=[TABLE_MU, 0.005, 0.0001])
    _geom(wb, "stove", mujoco.mjtGeom.mjGEOM_CYLINDER, [STOVE_R, STOVE_H / 2, 0],
          [STOVE_XY[0], STOVE_XY[1], TABLE_Z + STOVE_H / 2], mu=0.3, rgba=(0.08, 0.08, 0.09, 1))
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

    # the pot: a floor, a wall of box segments, two grip tabs (body origin: the floor's underside, centred)
    px, py, pyaw = episode["pot"]
    pb = wb.add_body(name="pot", pos=[px, py, TABLE_Z + STOVE_H + 0.0005], quat=_yaw_quat(pyaw))
    pb.add_freejoint(name="pot_free")
    _geom(pb, "pot_floor", mujoco.mjtGeom.mjGEOM_CYLINDER, [POT_R + POT_WALL, POT_FLOOR / 2, 0],
          [0, 0, POT_FLOOR / 2], mu=POT_MU)
    n = POT_SEGMENTS
    half_t = (POT_R + POT_WALL) * math.tan(math.pi / n) + 0.0005
    for k in range(n):
        th = 2 * math.pi * k / n
        rc = POT_R + POT_WALL / 2
        _geom(pb, f"pot_wall{k}", mujoco.mjtGeom.mjGEOM_BOX, [POT_WALL / 2, half_t, POT_DEPTH / 2],
              [rc * math.cos(th), rc * math.sin(th), POT_FLOOR + POT_DEPTH / 2], _yaw_quat(th), mu=POT_MU)
    for k, sgn in enumerate((1, -1)):
        _geom(pb, f"pot_tab{k}", mujoco.mjtGeom.mjGEOM_BOX, TAB_HALF,
              [0, sgn * (POT_R + POT_WALL + TAB_HALF[1] - 0.002), POT_FLOOR + TAB_Z], mu=TAB_MU, rgba=TAB_RGBA)

    # the bowls (body origin: the underside, centred): a floor and a conical wall of tilted boxes
    for name in BOWL_NAMES:
        bx, by = episode["bowls"][name]
        bb = wb.add_body(name=f"bowl_{name}", pos=[bx, by, TABLE_Z + 0.0005])
        bb.add_freejoint(name=f"bowl_{name}_free")
        rgba = BOWL_COLOURS[name]
        _geom(bb, f"bowl_{name}_floor", mujoco.mjtGeom.mjGEOM_CYLINDER, [BOWL_R0 + BOWL_WALL, BOWL_FLOOR / 2, 0],
              [0, 0, BOWL_FLOOR / 2], mu=BOWL_MU, rgba=rgba)
        slope = math.atan2(BOWL_R1 - BOWL_R0, BOWL_DEPTH)
        slant = math.hypot(BOWL_R1 - BOWL_R0, BOWL_DEPTH)
        rm = (BOWL_R0 + BOWL_R1) / 2 + BOWL_WALL / 2
        half_t = (BOWL_R1 + BOWL_WALL) * math.tan(math.pi / BOWL_SEGMENTS) + 0.0005
        for k in range(BOWL_SEGMENTS):
            th = 2 * math.pi * k / BOWL_SEGMENTS
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, np.array(_yaw_quat(th)), np.array([math.cos(slope / 2), 0, math.sin(slope / 2), 0]))
            _geom(bb, f"bowl_{name}_wall{k}", mujoco.mjtGeom.mjGEOM_BOX, [BOWL_WALL / 2, half_t, slant / 2],
                  [rm * math.cos(th), rm * math.sin(th), BOWL_FLOOR + BOWL_DEPTH / 2], q, mu=BOWL_MU, rgba=rgba)

    # the ladle: a slotted dish with a low rim and a handle rising from its rim, standing in the pot
    c, s_ = math.cos(pyaw), math.sin(pyaw)
    lx, ly = LADLE_IN_POT
    lb = wb.add_body(name="ladle", pos=[px + c * lx - s_ * ly, py + s_ * lx + c * ly,
                                         TABLE_Z + STOVE_H + POT_FLOOR + 0.002], quat=_yaw_quat(pyaw + LADLE_YAW))
    lb.add_freejoint(name="ladle_free")
    for k, yy in enumerate((-0.036, -0.018, 0.0, 0.018, 0.036)):  # the slotted floor: five bars, 4 mm slots, the
        hy = 0.007                                                 # outer ones against the rim
        hx = math.sqrt(max(LADLE_DISH_R ** 2 - (abs(yy) + hy) ** 2, 1e-6))
        _geom(lb, f"ladle_bar{k}", mujoco.mjtGeom.mjGEOM_BOX, [hx, hy, 0.002], [0, yy, 0.002], mu=0.5, rgba=LADLE_RGBA)
    ns = 12
    half_t = (LADLE_DISH_R + 0.002) * math.tan(math.pi / ns) + 0.0005
    for k in range(ns):
        th = 2 * math.pi * k / ns
        if math.cos(th) > 0.45:                    # the front (+x, away from the handle) is open (127 deg)
            continue
        _geom(lb, f"ladle_rim{k}", mujoco.mjtGeom.mjGEOM_BOX, [0.002, half_t, LADLE_RIM_H / 2],
              [(LADLE_DISH_R + 0.002) * math.cos(th), (LADLE_DISH_R + 0.002) * math.sin(th), LADLE_RIM_H / 2],
              _yaw_quat(th), mu=0.5, rgba=LADLE_RGBA)
    neck = np.array([-(LADLE_DISH_R + 0.004), 0.0, LADLE_RIM_H])
    u = np.array([-math.cos(LADLE_BETA), 0.0, math.sin(LADLE_BETA)])
    _geom(lb, "ladle_handle", mujoco.mjtGeom.mjGEOM_BOX, LADLE_HANDLE, neck + u * LADLE_HANDLE[0],
          [math.cos(LADLE_BETA / 2), 0, math.sin(LADLE_BETA / 2), 0], mu=1.0, rgba=LADLE_RGBA)

    # the knob: a fixed base and a cap turning about z (0 off .. KNOB_MAX full) with a fin to pinch
    _geom(wb, "knob_base", mujoco.mjtGeom.mjGEOM_CYLINDER, [KNOB_R + 0.008, 0.003, 0],
          [KNOB_XY[0], KNOB_XY[1], TABLE_Z + 0.003], mu=0.5, rgba=KNOB_BASE_RGBA)
    kb = wb.add_body(name="knob", pos=[KNOB_XY[0], KNOB_XY[1], TABLE_Z + 0.006])
    kj = kb.add_joint(name="knob_hinge", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], range=[0.0, KNOB_MAX])
    kj.damping = [0.05, 0, 0]
    kj.frictionloss = 0.05
    _geom(kb, "knob_cap", mujoco.mjtGeom.mjGEOM_CYLINDER, [KNOB_R, KNOB_H / 2, 0], [0, 0, KNOB_H / 2], mu=1.0,
          rgba=KNOB_BASE_RGBA)
    _geom(kb, "knob_fin", mujoco.mjtGeom.mjGEOM_BOX, KNOB_FIN, [0, 0, KNOB_H + KNOB_FIN[2]], mu=1.0, rgba=KNOB_RGBA)

    # the dishes of raw ingredients (body origin: the underside, centred), each with a flat grip tab toward -x
    for k, (dx, dy) in enumerate(episode["dishes"]):
        db = wb.add_body(name=f"dish{k}", pos=[dx, dy, TABLE_Z + 0.0005])
        db.add_freejoint(name=f"dish{k}_free")
        _geom(db, f"dish{k}_floor", mujoco.mjtGeom.mjGEOM_CYLINDER, [DISH_R + DISH_WALL, DISH_FLOOR / 2, 0],
              [0, 0, DISH_FLOOR / 2], mu=DISH_MU, rgba=DISH_RGBA)
        half_t = (DISH_R + DISH_WALL) * math.tan(math.pi / DISH_SEGMENTS) + 0.0005
        for q in range(DISH_SEGMENTS):
            th = 2 * math.pi * q / DISH_SEGMENTS
            rc = DISH_R + DISH_WALL / 2
            _geom(db, f"dish{k}_rim{q}", mujoco.mjtGeom.mjGEOM_BOX, [DISH_WALL / 2, half_t, DISH_RIM_H / 2],
                  [rc * math.cos(th), rc * math.sin(th), DISH_FLOOR + DISH_RIM_H / 2], _yaw_quat(th), mu=DISH_MU,
                  rgba=DISH_RGBA)
        _geom(db, f"dish{k}_tab", mujoco.mjtGeom.mjGEOM_BOX, DISH_TAB,
              [-(DISH_R + DISH_WALL + DISH_TAB[0] - 0.002), 0, DISH_TAB_Z], mu=1.0,
              rgba=DISH_RGBA)

    # the pieces, on their dishes
    sc = unit["piece_scale"]
    for i, ((k, ox, oy, yaw), kd) in enumerate(zip(episode["pieces"], episode["kinds"])):
        dx, dy = episode["dishes"][k]
        spec = PIECE[kd]
        sz = piece_size(kd, sc)
        hb = wb.add_body(name=f"piece{i}", pos=[dx + ox, dy + oy, TABLE_Z + DISH_FLOOR + piece_rest_half_height(kd, sc)
                                                + 0.002], quat=_yaw_quat(yaw))
        hb.add_freejoint(name=f"piece{i}_free")
        if spec["shape"] == "sphere":
            gg = _geom(hb, f"piece{i}_geom", mujoco.mjtGeom.mjGEOM_SPHERE, [sz[0], 0, 0], [0, 0, 0],
                       mu=unit["piece_mu"], rgba=spec["rgba"][0] + (1,))
        elif spec["shape"] == "capsule":                    # lying along x
            gg = _geom(hb, f"piece{i}_geom", mujoco.mjtGeom.mjGEOM_CAPSULE, [sz[0], sz[1], 0], [0, 0, 0],
                       [math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0], mu=unit["piece_mu"],
                       rgba=spec["rgba"][0] + (1,))
        elif spec["shape"] == "box":
            gg = _geom(hb, f"piece{i}_geom", mujoco.mjtGeom.mjGEOM_BOX, list(sz), [0, 0, 0], mu=unit["piece_mu"],
                       rgba=spec["rgba"][0] + (1,))
        else:
            gg = _geom(hb, f"piece{i}_geom", mujoco.mjtGeom.mjGEOM_ELLIPSOID, list(sz), [0, 0, 0],
                       mu=unit["piece_mu"], rgba=spec["rgba"][0] + (1,))
        gg.mass = spec["rho"] * piece_volume(kd, sc)
        gg.priority = 1

    # visual-only parts on mocap bodies: each vessel's soup surface (a flat ellipsoid) and the stream's segments
    for v in ["pot"] + [f"bowl_{nm}" for nm in BOWL_NAMES]:
        sb_ = wb.add_body(name=f"{v}_surface", mocap=True, pos=[0, 0, -1])
        _visual(sb_, f"{v}_surface_geom", mujoco.mjtGeom.mjGEOM_ELLIPSOID, [0.05, 0.05, 0.0015], SOUP_RGBA)
    for k in range(STREAM_GEOMS):
        st = wb.add_body(name=f"stream{k}", mocap=True, pos=[0, 0, -1])
        _visual(st, f"stream{k}_geom", mujoco.mjtGeom.mjGEOM_CAPSULE, [0.004, 0.01, 0], SOUP_RGBA)

    m = w.compile()
    # the pot's own mass and inertia (a thin-walled cylinder), the bowls' and the ladle's
    b = m.body("pot").id
    mp, r, h = unit["pot_mass"], POT_R + POT_WALL / 2, POT_DEPTH + POT_FLOOR
    m.body_mass[b] = mp
    m.body_ipos[b] = [0, 0, 0.4 * h]
    m.body_iquat[b] = [1, 0, 0, 0]
    m.body_inertia[b] = [mp * (6 * r * r + 2 * h * h) / 12] * 2 + [mp * r * r]
    for name in BOWL_NAMES:
        b = m.body(f"bowl_{name}").id
        m.body_mass[b] = BOWL_MASS
        m.body_ipos[b] = [0, 0, 0.025]
        m.body_iquat[b] = [1, 0, 0, 0]
        m.body_inertia[b] = [BOWL_MASS * 0.0025] * 2 + [BOWL_MASS * 0.0035]
    for k in range(len(episode["dishes"])):
        b = m.body(f"dish{k}").id
        m.body_mass[b] = DISH_MASS
        m.body_ipos[b] = [0, 0, 0.004]
        m.body_iquat[b] = [1, 0, 0, 0]
        m.body_inertia[b] = [DISH_MASS * 0.0008] * 2 + [DISH_MASS * 0.0014]
    b = m.body("ladle").id
    m.body_mass[b] = LADLE_MASS
    m.body_ipos[b] = [-0.06, 0, 0.07]
    m.body_iquat[b] = [1, 0, 0, 0]
    m.body_inertia[b] = [LADLE_MASS * 0.004, LADLE_MASS * 0.005, LADLE_MASS * 0.002]
    for side in BASES:
        for j in ("1", "2"):
            aid = m.actuator(f"{side}_g_gripper_finger_joint{j}").id
            m.actuator_gainprm[aid, 0] = GRIP_KP
            m.actuator_biasprm[aid, 1] = -GRIP_KP
            m.actuator_forcerange[aid] = [-GRIP_FORCE, GRIP_FORCE]
            m.actuator_ctrlrange[aid] = [-GRIP_PAST, 0.04] if j == "1" else [-0.04, GRIP_PAST]
        for f in ("finger1", "finger2"):
            gid = m.geom(f"{side}_g_{f}_pad_collision").id
            m.geom_condim[gid] = 4
            m.geom_friction[gid, 1] = PAD_TORSION
    for g in range(m.ngeom):
        if m.geom_group[g] == 0 and m.body(m.geom_bodyid[g]).name.startswith(tuple(f"{side}_" for side in BASES)):
            m.geom_group[g] = HIDDEN_GROUP
    mujoco.mj_setConst(m, mujoco.MjData(m))     # the constants that depend on the masses set above (invweight0):
    return m                                    # stale ones made the light ladle blow up on contact


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
        jid = [m.joint(f"{p}_joint{i}").id for i in range(1, 8)]
        self.q_lo, self.q_hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
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
        """Set this control step's command from one arm's action (8, 15 or 9 numbers, already checked). Joint angles
        in an action are clipped to the joints' limits."""
        d = self.d
        if len(a) == 9:
            q = d.qpos[self.jq].copy()
            qt = np.clip(np.asarray(a[:7], float), self.q_lo, self.q_hi)
            self.q_goal = q + np.clip(qt - q, -MAX_DQ, MAX_DQ)
            self.q_null = qt.copy()
            self.mode = "joint"
            grip = a[7]
        else:
            if len(a) == 15:
                self.q_null = np.clip(np.asarray(a[8:15], float), self.q_lo, self.q_hi)
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
        self.grip_target = 0.04 - (0.04 + GRIP_PAST) * (float(np.clip(grip, -1, 1)) + 1.0) / 2.0

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
        if not np.all(np.isfinite(tau)):
            raise FloatingPointError("unstable")
        return tau


# ----------------------------------------------------------------------------------------------
# the soup
# ----------------------------------------------------------------------------------------------
class Vessel:
    """A container of revolution in its body's frame: the inside from the floor's top (z0, radius r0) to the rim
    (z1, radius r1), the radius linear in z. Holds `vol` m^3 of soup."""

    SLICES = 40
    RIM_POINTS = 48

    def __init__(self, name, body, z0, z1, r0, r1, vol=0.0):
        self.name, self.body = name, body
        self.z0, self.z1, self.r0, self.r1 = z0, z1, r0, r1
        dz = (z1 - z0) / self.SLICES
        self.dz = dz
        self.zk = z0 + dz * (np.arange(self.SLICES) + 0.5)
        self.rk = r0 + (r1 - r0) * (self.zk - z0) / (z1 - z0)
        th = 2 * math.pi * np.arange(self.RIM_POINTS) / self.RIM_POINTS
        self.rim = np.column_stack([r1 * np.cos(th), r1 * np.sin(th), np.full_like(th, z1)])
        self.rim_ds = 2 * math.pi * r1 / self.RIM_POINTS
        self.vol = float(vol)
        hs = np.linspace(0.0, z1 - z0, 401)                 # upright: volume below the height z0 + h, tabulated
        rs = r0 + (r1 - r0) * hs / (z1 - z0)
        self.up_h = hs
        self.up_v = math.pi * hs * (r0 * r0 + r0 * rs + rs * rs) / 3.0
        self.c = None                 # the surface: world height above the body's origin (along world z)
        self.n = np.array([0, 0, 1.0])
        self.capacity = self.volume(1.0, np.array([0, 0, 1.0]))[0]

    def volume(self, c, n):
        """Volume below the plane {x : n . x = c} (body frame, n unit) inside the vessel, and its centroid."""
        nz = n[2]
        nh = math.hypot(n[0], n[1])
        e = c - nz * self.zk                       # per slice: the region n_h . (x, y) <= e
        r = self.rk
        if nh < 1e-9:
            A = np.where(e >= 0, math.pi * r * r, 0.0)
            ubar = np.zeros_like(A)
        else:
            s = np.clip(e / nh, -r, r)
            root = np.sqrt(np.maximum(r * r - s * s, 0.0))
            A = r * r * np.arccos(-s / r) + s * root
            ubar = np.where(A > 1e-12, -(2.0 / 3.0) * root ** 3 / np.maximum(A, 1e-12), 0.0)
        V = float(np.sum(A) * self.dz)
        if V <= 1e-12:
            return 0.0, np.array([0, 0, self.z0])
        zc = float(np.sum(A * self.zk) / np.sum(A))
        uc = float(np.sum(A * ubar) / np.sum(A))
        u = np.array([n[0], n[1], 0.0]) / nh if nh >= 1e-9 else np.zeros(3)
        return V, u * uc + np.array([0, 0, zc])

    def solve(self, n):
        """The surface height c that holds self.vol at the orientation n (world up in the body frame)."""
        if n[2] > 1.0 - 1e-5:                                # upright (within 0.26 deg): the table
            h = float(np.interp(self.vol, self.up_v, self.up_h)) if self.vol < self.up_v[-1] else self.up_h[-1]
            rz = self.r0 + (self.r1 - self.r0) * h / (self.z1 - self.z0)
            zc = self.z0 + h * (self.r0 ** 2 + 2 * self.r0 * rz + 3 * rz ** 2) / (4 * (self.r0 ** 2 + self.r0 * rz + rz ** 2) + 1e-18)
            return (self.z0 + h) * n[2], np.array([0, 0, zc])
        pts = np.array([[0, 0, self.z0], [0, 0, self.z1]])
        ext = np.abs(n[:2]).sum() * self.r1
        lo = min(n @ pts[0], n @ pts[1]) - ext - 1e-4
        hi = max(n @ pts[0], n @ pts[1]) + ext + 1e-4
        if self.vol <= 1e-12:
            return lo, np.array([0, 0, self.z0])
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            if self.volume(mid, n)[0] < self.vol:
                lo = mid
            else:
                hi = mid
        c = 0.5 * (lo + hi)
        return c, self.volume(c, n)[1]


class Liquid:
    """The soup in every vessel, the stream between them and the spill, updated every LIQUID_EVERY substeps."""

    def __init__(self, world, cd):
        self.w = world
        m = world.m
        self.cd = cd
        self.vessels = [Vessel("pot", m.body("pot").id, POT_FLOOR, POT_FLOOR + POT_DEPTH, POT_R, POT_R,
                               world.unit["soup_l"] / 1000.0)]
        for nm in BOWL_NAMES:
            self.vessels.append(Vessel(nm, m.body(f"bowl_{nm}").id, BOWL_FLOOR, BOWL_FLOOR + BOWL_DEPTH, BOWL_R0, BOWL_R1))
        self.surf_mocap = [m.body_mocapid[m.body(f"{'pot' if v.name == 'pot' else 'bowl_' + v.name}_surface").id]
                           for v in self.vessels]
        self.surf_geom = [m.geom(f"{'pot' if v.name == 'pot' else 'bowl_' + v.name}_surface_geom").id
                          for v in self.vessels]
        self.stream_mocap = [m.body_mocapid[m.body(f"stream{k}").id] for k in range(STREAM_GEOMS)]
        self.stream_geom = [m.geom(f"stream{k}_geom").id for k in range(STREAM_GEOMS)]
        self.parcels = []                  # [pos(3), vel(3), vol, source index, age]
        self.spilled = 0.0
        self.received = [0.0] * len(self.vessels)
        self.flow = [0.0] * len(self.vessels)   # m^3/s out of each vessel at the last update
        self.dt = TIMESTEP * LIQUID_EVERY
        self._v6 = np.zeros(6)
        # the pieces and their heat
        sc = world.unit["piece_scale"]
        self.pieces = list(world.pieces)
        self.kinds = list(world.kinds)
        self.pv = np.array([piece_volume(kd, sc) for kd in self.kinds])
        self.pa = (3.0 * self.pv / (4.0 * math.pi)) ** (1.0 / 3.0)          # the radius of a sphere of its volume
        self.pm = np.array([PIECE[kd]["rho"] for kd in self.kinds]) * self.pv
        self.lift = np.array([(PIECE[kd]["rho"] - PIECE[kd].get("rho_cooked", PIECE[kd]["rho"])) * G
                              for kd in self.kinds]) * self.pv               # a cooked fish ball's extra lift
        self.tau = np.array([PIECE[kd]["tau"] for kd in self.kinds]) * sc * sc
        self.t_cook = np.array([PIECE[kd]["t_cook"] for kd in self.kinds])
        self.window = np.array([PIECE[kd]["window"] for kd in self.kinds])
        self.pgeom = [m.body_geomadr[b] for b in self.pieces]
        self.burner_w = 1000.0 * world.unit["burner_kw"]
        self.T_broth = BROTH_T0
        self.T = np.full(len(self.pieces), ROOM)
        self.T_max = np.full(len(self.pieces), ROOM)
        self.over = np.zeros(len(self.pieces))                 # s in the pot's broth since cooked
        self.where = np.full(len(self.pieces), -1)            # the vessel a piece is under the surface of, or -1

    def _pose(self, v):
        d = self.w.d
        return d.xpos[v.body].copy(), d.xmat[v.body].reshape(3, 3).copy()

    def update(self):
        d = self.w.d
        dt = self.dt
        d.xfrc_applied[:] = 0.0
        for i, v in enumerate(self.vessels):
            p, R = self._pose(v)
            n = R.T @ np.array([0, 0, 1.0])
            c, cen = v.solve(n)
            v.c, v.n = c, n
            # outflow over the rim
            dh = c - v.rim @ n
            over = dh > 0
            Q = 0.0
            if v.vol > 1e-9 and np.any(over):
                q = (2.0 / 3.0) * self.cd * math.sqrt(2 * G) * dh[over] ** 1.5 * v.rim_ds
                Q = float(np.sum(q))
                dv = min(v.vol, Q * dt)
                if dv > 0:
                    pl = (v.rim[over] * q[:, None]).sum(axis=0) / Q
                    dmean = float(np.sum(dh[over] * q) / Q)
                    out = np.array([pl[0], pl[1], 0.0])
                    out = R @ (out / max(np.linalg.norm(out), 1e-9))
                    out[2] = 0.0
                    out /= max(np.linalg.norm(out), 1e-9)
                    pw = p + R @ pl
                    mujoco.mj_objectVelocity(self.w.m, d, mujoco.mjtObj.mjOBJ_BODY, v.body, self._v6, 0)
                    vel = self._v6[3:] + np.cross(self._v6[:3], pw - d.xipos[v.body])   # the CoM's velocity + w x r
                    vel = vel + out * 0.7 * math.sqrt(2 * G * dmean)
                    self.parcels.append([pw + out * 0.004, vel, dv, i, 0.0])
                    v.vol -= dv
            self.flow[i] = Q
            # weight at the centroid
            if v.vol > 1e-9:
                F = np.array([0, 0, -RHO * G * v.vol])
                cw = p + R @ cen
                d.xfrc_applied[v.body, :3] += F
                d.xfrc_applied[v.body, 3:] += np.cross(cw - d.xipos[v.body], F)
        self._fly(dt)
        self._pieces()
        self._heat(dt)
        self._draw()

    def _fly(self, dt):
        keep = []
        for pc in self.parcels:
            p0 = pc[0].copy()
            pc[1] = pc[1] + np.array([0, 0, -G * dt])
            pc[0] = pc[0] + pc[1] * dt
            pc[4] += dt
            landed = False
            for i, v in enumerate(self.vessels):
                if i == pc[3] and pc[4] < 0.05:
                    continue
                p, R = self._pose(v)
                if R[2, 2] < 0.8:
                    continue
                rim_c = p + R @ np.array([0, 0, v.z1])
                if p0[2] >= rim_c[2] > pc[0][2]:
                    f = (p0[2] - rim_c[2]) / max(p0[2] - pc[0][2], 1e-12)
                    x = p0 + f * (pc[0] - p0)
                    if math.hypot(x[0] - rim_c[0], x[1] - rim_c[1]) < v.r1:
                        v.vol += pc[2]
                        self.received[i] += pc[2]
                        landed = True
                        break
            if landed:
                continue
            if pc[0][2] < TABLE_Z + 0.002:
                self.spilled += pc[2]
                continue
            keep.append(pc)
        self.parcels = keep

    def _pieces(self):
        """Buoyancy and drag on every piece under a surface (its submerged volume taken as that of a sphere of the same
        volume, by the depth of its centre); a cooked fish ball gets its extra lift."""
        d = self.w.d
        for i, b in enumerate(self.pieces):
            x = d.xpos[b]
            a = self.pa[i]
            self.where[i] = -1
            for vi, v in enumerate(self.vessels):
                if v.vol <= 1e-9:
                    continue
                p, R = self._pose(v)
                xl = R.T @ (x - p)
                rz = v.r0 + (v.r1 - v.r0) * (xl[2] - v.z0) / (v.z1 - v.z0)
                if not (v.z0 - a < xl[2] < v.z1 + a and math.hypot(xl[0], xl[1]) < rz):
                    continue
                depth = v.c - v.n @ xl            # how far the centre is below the surface
                hcap = min(max(depth + a, 0.0), 2 * a)
                if hcap <= 0:
                    break
                vsub = math.pi * hcap * hcap * (3 * a - hcap) / 3.0
                frac = vsub / (4.0 / 3.0 * math.pi * a ** 3)
                if frac > 0.3:
                    self.where[i] = vi
                lift = RHO * G * frac * self.pv[i] + self.lift[i] * frac * self.done_frac(i)
                mujoco.mj_objectVelocity(self.w.m, d, mujoco.mjtObj.mjOBJ_BODY, b, self._v6, 0)
                d.xfrc_applied[b, :3] += np.array([0, 0, lift]) - DRAG * frac * self._v6[3:]
                break

    def burner_power(self):
        """W into the pot's broth: the knob's share of the unit's full power, while the pot stands on the ring."""
        w = self.w
        a = float(w.d.qpos[w.knob_q])
        if a < KNOB_OFF or not w.pot_on_ring(0.03, 10.0):
            return 0.0
        return self.burner_w * min(a / KNOB_MAX, 1.0)

    def _heat(self, dt):
        pot = self.vessels[0]
        inpot = self.where == 0
        if pot.vol > 1e-6:
            q_in = self.burner_power() - POT_LOSS * (self.T_broth - ROOM)
            q_in -= float(np.sum((self.pm * PIECE_C * (self.T_broth - self.T) / self.tau)[inpot]))
            self.T_broth = min(BOIL, self.T_broth + q_in * dt / (pot.vol * 1000.0 * BROTH_C))
        self.T = np.where(inpot, self.T + (self.T_broth - self.T) * dt / self.tau, self.T + (ROOM - self.T) * dt / AIR_TAU)
        self.T_max = np.maximum(self.T_max, self.T)
        cooked = self.T_max >= self.t_cook
        self.over = self.over + np.where(cooked & inpot, dt, 0.0)

    def done_frac(self, i):
        """0 raw .. 1 cooked, by the hottest its core has been."""
        return float(min(max((self.T_max[i] - ROOM) / (self.t_cook[i] - ROOM), 0.0), 1.0))

    def cooked(self, i):
        return bool(self.T_max[i] >= self.t_cook[i])

    def overcooked(self, i):
        return bool(self.over[i] > self.window[i])

    def piece_rgba(self, i):
        c = PIECE[self.kinds[i]]["rgba"]
        if not self.cooked(i):
            f = self.done_frac(i)
            return tuple(c[0][k] + (c[1][k] - c[0][k]) * f for k in range(3))
        f = min(self.over[i] / self.window[i], 1.0)
        return tuple(c[1][k] + (c[2][k] - c[1][k]) * f for k in range(3))

    def _draw(self):
        m, d = self.w.m, self.w.d
        for i, g in enumerate(self.pgeom):
            m.geom_rgba[g, :3] = self.piece_rgba(i)
        for i, v in enumerate(self.vessels):
            mid, g = self.surf_mocap[i], self.surf_geom[i]
            if v.vol <= 2e-6:
                m.geom_rgba[g, 3] = 0.0
                d.mocap_pos[mid] = [0, 0, -1]
                continue
            p, R = self._pose(v)
            n = v.n
            nz = max(n[2], 0.2)
            zs = min(max(v.c / nz, v.z0 + 0.001), v.z1)       # where the surface crosses the vessel's axis
            rz = v.r0 + (v.r1 - v.r0) * (zs - v.z0) / (v.z1 - v.z0)
            centre = p + R @ np.array([0, 0, zs])
            axis = R[:, 2]
            tilt = np.array([axis[0], axis[1], 0.0])
            if np.linalg.norm(tilt) < 1e-6:
                ex = np.array([1.0, 0, 0])
            else:
                ex = tilt / np.linalg.norm(tilt)
            ey = np.cross([0, 0, 1.0], ex)
            Rm = np.column_stack([ex, ey, [0, 0, 1.0]])
            d.mocap_pos[mid] = centre
            d.mocap_quat[mid] = quat_from_mat(Rm)
            m.geom_size[g] = [rz / max(axis[2], 0.2) * 0.995, rz * 0.995, 0.0015]
            m.geom_rgba[g, 3] = SOUP_RGBA[3]
        # the stream: segments between parcels spread over the flight, oldest to newest
        pts = [pc[0] for pc in sorted(self.parcels, key=lambda pc: -pc[4])]
        k = 0
        if len(pts) >= 2:
            idx = np.unique(np.linspace(0, len(pts) - 1, min(len(pts), STREAM_GEOMS + 1)).astype(int))
            for a, b in zip(idx[:-1], idx[1:]):
                p0, p1 = pts[a], pts[b]
                L = np.linalg.norm(p1 - p0)
                if L < 1e-4:
                    continue
                mid = self.stream_mocap[k]
                z = (p1 - p0) / L
                x = np.cross(z, [1.0, 0, 0]) if abs(z[0]) < 0.9 else np.cross(z, [0, 1.0, 0])
                x /= np.linalg.norm(x)
                d.mocap_pos[mid] = (p0 + p1) / 2
                d.mocap_quat[mid] = quat_from_mat(np.column_stack([x, np.cross(z, x), z]))
                m.geom_size[self.stream_geom[k]] = [0.005, L / 2, 0]
                m.geom_rgba[self.stream_geom[k], 3] = 0.8
                k += 1
                if k >= STREAM_GEOMS:
                    break
        for j in range(k, STREAM_GEOMS):
            d.mocap_pos[self.stream_mocap[j]] = [0, 0, -1]
            m.geom_rgba[self.stream_geom[j], 3] = 0.0

    def in_flight(self):
        return float(sum(pc[2] for pc in self.parcels))

    def state(self):
        """Everything of the soup that is not in MuJoCo's state (for the reference's shadow)."""
        return {"vol": [v.vol for v in self.vessels], "spilled": self.spilled, "received": list(self.received),
                "flow": list(self.flow), "T_broth": self.T_broth, "T": self.T.tolist(), "T_max": self.T_max.tolist(),
                "over": self.over.tolist(),
                "parcels": [[list(map(float, p[0])), list(map(float, p[1])), float(p[2]), int(p[3]), float(p[4])]
                            for p in self.parcels]}

    def set_state(self, st):
        for v, x in zip(self.vessels, st["vol"]):
            v.vol = float(x)
        self.spilled = float(st["spilled"])
        self.received = [float(x) for x in st["received"]]
        self.flow = [float(x) for x in st["flow"]]
        self.T_broth = float(st["T_broth"])
        self.T, self.T_max, self.over = np.array(st["T"]), np.array(st["T_max"]), np.array(st["over"])
        self.parcels = [[np.array(p[0]), np.array(p[1]), p[2], p[3], p[4]] for p in st["parcels"]]


# ----------------------------------------------------------------------------------------------
# the world
# ----------------------------------------------------------------------------------------------
class World:
    def __init__(self, unit, episode):
        self.unit, self.episode = unit, episode
        self.m = m = build(unit, episode)
        self.d = d = mujoco.MjData(m)
        self.arms = {s: Arm(m, d, s) for s in BASES}
        self.pot = m.body("pot").id
        self.bowls = {nm: m.body(f"bowl_{nm}").id for nm in BOWL_NAMES}
        self.ladle = m.body("ladle").id
        self.pieces = [m.body(f"piece{i}").id for i in range(len(episode["pieces"]))]
        self.kinds = list(episode["kinds"])
        self.dishes = [m.body(f"dish{k}").id for k in range(len(episode["dishes"]))]
        self.knob_q = m.joint("knob_hinge").qposadr[0]
        # which pieces a robot (any geom of an arm, its hand or its fingers) has touched, checked every substep
        robot = tuple(f"{side}_" for side in BASES)
        self._robot_geom = np.array([m.body(int(m.geom_bodyid[g])).name.startswith(robot) for g in range(m.ngeom)])
        self._piece_of = np.full(m.ngeom, -1, dtype=int)
        for i, b in enumerate(self.pieces):
            self._piece_of[m.body_geomadr[b]:m.body_geomadr[b] + m.body_geomnum[b]] = i
        self.touched = [False] * len(self.pieces)
        self.t = 0
        mujoco.mj_resetData(m, d)
        for a in self.arms.values():
            d.qpos[a.jq] = HOME_Q
            d.qpos[a.gq[0]], d.qpos[a.gq[1]] = 0.04, -0.04
        mujoco.mj_forward(m, d)
        self.ring = np.array([STOVE_XY[0], STOVE_XY[1], TABLE_Z + STOVE_H])
        self.pot_rest_z = float(d.xpos[self.pot][2]) - 0.0005
        self.liquid = Liquid(self, unit["cd"])
        for k in range(1000):                     # settle the arms and let the objects come to rest in the soup
            for a in self.arms.values():
                d.ctrl[a.act] = 200.0 * (HOME_Q - d.qpos[a.jq]) - 20.0 * d.qvel[a.jv] + d.qfrc_bias[a.jv]
                d.ctrl[a.gact] = [0.04, -0.04]
            if k % LIQUID_EVERY == 0:
                self.liquid.update()
            mujoco.mj_step(m, d)
        for a in self.arms.values():
            a.init_targets()
        self.pot_start = d.xpos[self.pot].copy()
        self.pot_rest_z = float(d.xpos[self.pot][2])
        self.pot_lifted = False

    def step(self, cmds):
        """One control step. `cmds`: {side: action list (8, 15 or 9 numbers)}; a side left out keeps its command."""
        m, d = self.m, self.d
        for s, a in cmds.items():
            self.arms[s].command(a)
        for k in range(SUBSTEPS):
            for a in self.arms.values():
                d.ctrl[a.act] = a.torque()
                d.ctrl[a.gact] = [a.grip_target, -a.grip_target]
            if k % LIQUID_EVERY == 0:
                self.liquid.update()
            mujoco.mj_step(m, d)
            self._touch()
        self.t += 1
        if not self.pot_lifted and self.d.xpos[self.pot][2] > self.pot_rest_z + 0.02:
            self.pot_lifted = True
        if self.unstable():
            raise FloatingPointError("unstable")

    def _touch(self):
        """Mark the pieces a robot geom is in contact with."""
        n = self.d.ncon
        if not n:
            return
        g1, g2 = self.d.contact.geom1[:n], self.d.contact.geom2[:n]
        hit = np.concatenate([self._piece_of[g2][self._robot_geom[g1]], self._piece_of[g1][self._robot_geom[g2]]])
        for i in hit[hit >= 0]:
            self.touched[int(i)] = True

    def unstable(self):
        d = self.d
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return True
        if any(np.max(np.abs(d.qvel[a.jv])) > MAX_ARM_SPEED for a in self.arms.values()):
            return True
        v = np.zeros(6)
        for b in [self.pot, self.ladle] + list(self.bowls.values()) + self.dishes + self.pieces:
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            if np.linalg.norm(v[3:]) > MAX_OBJECT_SPEED:
                return True
        return False

    # --- state the grader (and the privileged reference) reads ------------------------------------
    def tilt_deg(self, b):
        R = self.d.xmat[b].reshape(3, 3)
        return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))

    def bowl_soup_ml(self, nm):
        return 1e6 * self.liquid.vessels[1 + BOWL_NAMES.index(nm)].vol

    def pot_soup_ml(self):
        return 1e6 * self.liquid.vessels[0].vol

    def spilled_ml(self):
        return 1e6 * self.liquid.spilled

    def pot_on_ring(self, tol, tilt):
        """The pot stands on its ring: its centre within `tol` m of the ring's centre, tilted at most `tilt` deg, its
        bottom at most 1 cm above the ring."""
        p = self.d.xpos[self.pot]
        return bool(math.hypot(p[0] - self.ring[0], p[1] - self.ring[1]) <= tol and self.tilt_deg(self.pot) <= tilt
                    and p[2] <= self.pot_rest_z + 0.01)

    def distance_to_rim(self, x):
        """The distance from the point x to the pot's rim (the circle of its wall's top, inside radius)."""
        p, R = self.d.xpos[self.pot], self.d.xmat[self.pot].reshape(3, 3)
        xl = R.T @ (np.asarray(x, float) - p)
        rho = math.hypot(xl[0], xl[1])
        return math.hypot(rho - POT_R, xl[2] - (POT_FLOOR + POT_DEPTH))

    def knob(self):
        return float(self.d.qpos[self.knob_q])

    def burner_on(self):
        return self.knob() >= KNOB_OFF

    def piece_place(self, i):
        """'pot', a bowl's name, 'dish' (on one of the dishes), or None (elsewhere): for a vessel, the centre inside
        its wall, above its floor, below its rim; for a dish, within its rim and less than 4 cm above its floor."""
        x = self.d.xpos[self.pieces[i]]
        for v in self.liquid.vessels:
            p = self.d.xpos[v.body]
            R = self.d.xmat[v.body].reshape(3, 3)
            xl = R.T @ (x - p)
            rz = v.r0 + (v.r1 - v.r0) * (xl[2] - v.z0) / (v.z1 - v.z0)
            if v.z0 < xl[2] < v.z1 and math.hypot(xl[0], xl[1]) < rz:
                return v.name
        for b in self.dishes:
            p = self.d.xpos[b]
            R = self.d.xmat[b].reshape(3, 3)
            xl = R.T @ (x - p)
            if math.hypot(xl[0], xl[1]) < DISH_R + 0.002 and DISH_FLOOR - 0.005 < xl[2] < DISH_FLOOR + 0.04:
                return "dish"
        return None

    def cooked(self, i):
        return self.liquid.cooked(i)

    def overcooked(self, i):
        return self.liquid.overcooked(i)

    def broth_c(self):
        return self.liquid.T_broth


# ----------------------------------------------------------------------------------------------
# rendering (MuJoCo's own renderer, OSMesa)
# ----------------------------------------------------------------------------------------------
_SCENE_OPT = None


def scene_option():
    global _SCENE_OPT
    if _SCENE_OPT is None:
        _SCENE_OPT = mujoco.MjvOption()
        _SCENE_OPT.sitegroup[:] = 0
        _SCENE_OPT.geomgroup[:] = 1
        _SCENE_OPT.geomgroup[HIDDEN_GROUP] = 0
    return _SCENE_OPT


def camera_size(cam):
    if cam.startswith("wrist"):
        return WRIST["width"], WRIST["height"], WRIST["fovy"]
    _p, _t, fovy, w, h = CAMERAS[cam]
    return w, h, fovy


def render(m, d, cam, rng, cache=None):
    """One camera's image of the state `d`, with the pixel noise drawn from `rng`. `cache` (a dict the caller keeps
    for this model) holds one mujoco.Renderer per image size."""
    w, h, _fovy = camera_size(cam)
    try:
        key = (w, h)
        r = cache.get(key) if cache is not None else None
        if r is None:
            r = mujoco.Renderer(m, h, w)
            if cache is not None:
                cache[key] = r
        r.update_scene(d, camera=MJ_CAMERA[cam], scene_option=scene_option())
        img = r.render().astype(np.float32)
        if cache is None:
            r.close()
    except Exception as exc:  # noqa: BLE001 - whatever the renderer raises is reported as one error
        raise RenderError("{} camera ({}x{}): {}: {}".format(cam, w, h, type(exc).__name__, str(exc)[:200])) from exc
    img += rng.normal(0, RGB_NOISE, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def close_renderers(cache):
    for r in list(cache.values()):
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
    cache.clear()
