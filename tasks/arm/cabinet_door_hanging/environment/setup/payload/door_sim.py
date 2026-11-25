"""Cabinet door hanging: the simulation of the cell (two Pandas, a wall cabinet with its old door hung, the new door's
rack, an empty parking rack, six pin stands).

Plain MuJoCo 3.16 on CPU, robosuite 1.5.2 model files only (robosuite is not imported). Frames: world z up, metres,
radians, seconds; quaternions (w, x, y, z). The robots stand on a table (top at TABLE_Z, edge at x = TABLE_EDGE_X)
facing +x; the cabinet hangs on a wall beyond the table's edge, so nothing is below its door.

- Left arm (y = +0.28): the Panda parallel gripper.
- Right arm (y = -0.28): an L-shaped vacuum cup tool (with the hand pointing down the cup faces forward, along the
  hand's x axis). Its `grip` > 0 asks for vacuum: the cup seals on the door's front face when its face touches the
  door square (within CUP_SEAL_GAP and CUP_SEAL_DEG, fully on the leaf) and then holds the door through a soft weld
  (the cup's compliance). The seal breaks when vacuum is switched off, or when the held load (the force, or the
  moment about the cup's face centre) stays above CUP_MAX_FORCE / CUP_MAX_TORQUE for CUP_OVERLOAD_SUBSTEPS
  substeps (20 ms): the door is then let go.
- The cabinet (the "carcass", fixed) has one opening with two hinges on its left (+y) side. A hinge is a HINGE_LEN
  stack of n knuckles (3 or 5) on one vertical axis: carcass knuckles at both ends, door knuckles between them, each
  door knuckle AX_PLAY shorter than its slot at each end. A knuckle is a ring of 16 boxes around its bore; rings do
  not collide with rings, and thin square end plates carry the axial contact between knuckles. Every knuckle of the
  series has the outer radius KN_RO whatever its bore, so the bore does not show from the side.
- A pin is a free T-shaped body: a capsule shaft (SHAFT long below its head), a head that seats on the hinge's top
  knuckle, an arm and a square grip post POST in front of its axis. Inside a bore of its own class a pin slides only
  under a push of about PIN_DRAG (a detent: a frictionloss on its translational dofs, switched per substep); within
  PIN_SNAP of its seat it clicks in: its head centre is held there (a point constraint, so the seated pin still turns
  about its axis) until it is pulled up harder than PIN_PULL. A pin of a smaller class slides through a bore freely
  (no detent, no click) and comes to rest on its head; a pin of a larger class does not enter the bore.
- The old door (body `old`, the same model as the new door `door`) starts hung in the opening on two old pins
  (radius OLD_PIN_R, class "old": thinner than every class, so they slide freely and never count). It must be taken
  off and must not fall: `old_fell` turns true when its centre of mass comes below TABLE_Z - 0.05.
- A unit is one cabinet of the series (`draw_unit`): knuckle count, pin class, radial clearance, hinge heights, door
  mass and handle mass (both doors). An episode (`draw_episode`) places the carcass, the new door in its rack, the
  empty parking rack, and six pins (two of each class, in an order drawn per episode) in the six stands.
"""

import math
import os

import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import door_parts  # noqa: E402

TIMESTEP = 0.002
SUBSTEPS = 50                       # 10 Hz control
MAX_DPOS = 0.03
MAX_DROT = 0.2
MAX_DQ = 0.1
MAX_ARM_SPEED = 8.0
MAX_OBJECT_SPEED = 5.0
TABLE_Z = 0.80
TABLE_EDGE_X = 0.55
HOME_Q = np.array([0.0, -0.35, 0.0, -2.30, 0.0, 1.95, math.pi / 4])
BASES = {"l": np.array([0.0, 0.28]), "r": np.array([0.0, -0.28])}
TOOLS = {"l": "gripper", "r": "cup"}

# hinges and pins
PIN_CLASSES = {"S": 0.0025, "M": 0.00325, "L": 0.0040}    # pin radius by class (5, 6.5 and 8 mm pins)
HINGE_LEN = 0.075                   # a hinge's knuckle stack
KN_RO = 0.0090                      # every knuckle's outer radius (the largest bore, L + 0.5 mm, + a 4.5 mm wall)
AX_PLAY = 0.001                     # axial play each side of a door knuckle in its slot
SHAFT = HINGE_LEN - 0.004           # a pin's shaft below its head
N_SEG = 16
PIN_DRAG = 6.0                      # N, the detent's sliding resistance inside a bore
PIN_SNAP = 0.0007                   # m: a pin whose head comes this close to its seat snaps into its detent
PIN_PULL = 40.0                     # N: a snapped pin comes out only when pulled up harder than this
POST = np.array([-0.050, 0.0, 0.027])   # grip post centre in a pin's frame
STAND_H = 0.125
STAND_CLEAR = 0.0015
PAD_TORSION = 0.01                  # m, the finger pads' torsional friction

# the cabinet (carcass frame: origin on the hinge axis at the hung door's bottom edge, x into the cabinet)
CARCASS_NOM = np.array([0.589, 0.180, 0.86])
DOOR_T, DOOR_W, DOOR_H = 0.018, 0.40, 0.45
EDGE_GAP = 0.014                    # axis to the door's hinge edge
DEPTH = 0.30

# units: every unit, practice or scored, is drawn from these
N_KN_SET = (3, 5)
HB_SET = (0.05, 0.06, 0.07)         # bottom hinge centre above the door's bottom edge
HT_SET = (0.38, 0.39, 0.40)         # top hinge centre
UNIT_RANGES = {"clear": (0.0002, 0.0005), "door_mass": (1.5, 3.5), "handle_mass": (0.1, 0.4)}
# episodes
CARCASS_JITTER = (0.008, 0.008, 0.008, math.radians(2.0))   # x, y, z (m), yaw (rad)
RACK = np.array([0.38, 0.10])       # the new door's rack: the racked door's frame origin (x, y)
PARK = np.array([0.50, -0.36])      # the empty parking rack the old door goes to (the same frame)
PARK_GAP = 0.012                    # the parking rack's channel: 8 mm free each side of a door (the rack's: 1 mm)
RACK_JITTER = 0.008                 # each rack's place, x and y
OLD_PIN_R = 0.0020                  # the old door's wrong pins: 4 mm, smaller than every class
STANDS = [np.array([0.45, y]) for y in (0.30, 0.38, 0.46, 0.54, 0.62, 0.70)]
BAG_AT = (0.05, -1.12, math.radians(20.0))   # the tool bag on the floor: x, y, yaw

CUP_R = 0.030
CUP_DROP = 0.090                    # the L tool: flange to the cup's centre line
CUP_REACH = 0.045                   # post axis to the cup's face
CUP_MAX_FORCE = 150.0               # N, the held load the seal survives (sustained)
CUP_MAX_TORQUE = 15.0               # N m, about the cup's face centre
CUP_OVERLOAD_SUBSTEPS = 10          # 20 ms over a limit breaks the seal
CUP_SEAL_GAP = 0.003                # m, the cup's face to the door's face for the seal to form
CUP_SEAL_DEG = 10.0                 # the cup's axis to the door's normal for the seal to form
CUP_SOLREF = (0.02, 1.0)
WRENCH_NOISE = 0.3                  # N (force); torque noise is a tenth, N m
RGB_NOISE = 2.0

CAMERAS = {                         # name: (position, look-at, vertical fov deg, width, height)
    "overhead": ((0.25, 0.0, TABLE_Z + 1.25), (0.42, 0.0, TABLE_Z), 70.0, 640, 480),
    "front": ((-0.30, 0.0, TABLE_Z + 0.85), (0.60, 0.0, TABLE_Z + 0.25), 50.0, 640, 480),
    "side": ((0.22, 0.58, TABLE_Z + 0.42), (0.59, 0.18, TABLE_Z + 0.27), 42.0, 640, 480),
}
WRIST = {"fovy": 75.0, "width": 320, "height": 240}
MJ_CAMERA = {"overhead": "overhead", "front": "front", "side": "side", "wrist_l": "l_eye_in_hand",
             "wrist_r": "r_g_cup_cam"}
CAMERA_BITS = {"overhead": 1, "front": 2, "side": 4, "wrist_l": 8, "wrist_r": 16}
ALL_CAMS = 31


class RenderError(RuntimeError):
    """The renderer failed (a grader-side fault: the task's error, never the submission's)."""


def intrinsics(fovy, w, h):
    f = (h / 2) / math.tan(math.radians(fovy) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])


def camera_size(cam):
    """(width, height, vertical fov deg) of a camera."""
    if cam.startswith("wrist"):
        return WRIST["width"], WRIST["height"], WRIST["fovy"]
    _p, _t, fovy, w, h = CAMERAS[cam]
    return w, h, fovy


def cam_pose(d, cid):
    """World pose of a MuJoCo camera in the OpenCV convention (x right, y down, z forward)."""
    R = d.cam_xmat[cid].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])
    return R, d.cam_xpos[cid].copy()


def render(world, cam, rng, cache):
    """One camera's image (uint8 H x W x 3) of the world's state, with sensor noise. The cameras are drawn by
    door_render.py (beside this file) in a process of its own per episode, started at the episode's first image;
    `cache` keeps that process. Raises RenderError when the renderer fails."""
    import door_render
    w, h, _fovy = camera_size(cam)
    try:
        view = cache.get("view")
        if view is None:
            view = cache["view"] = door_render.Worker(world.unit, world.episode)
        img = view.render(MJ_CAMERA[cam], world.d.qpos, w, h).astype(np.float32)
    except Exception as exc:  # noqa: BLE001 - whatever the renderer raises is reported as one error
        raise RenderError("{} camera ({}x{}): {}: {}".format(cam, w, h, type(exc).__name__, str(exc)[:200])) from exc
    img += rng.normal(0, RGB_NOISE, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def close_renderers(cache):
    for r in cache.values():
        try:
            r.close()
        except Exception:  # noqa: BLE001
            pass
    cache.clear()


def draw_unit(seed):
    """A unit: the cabinet's hinges and door."""
    rng = np.random.default_rng([int(seed) % 2 ** 32, int(seed) >> 32, 101])
    rg = UNIT_RANGES
    return {"n_kn": int(rng.choice(N_KN_SET)), "pin_class": str(rng.choice(sorted(PIN_CLASSES))),
            "clear": round(float(rng.uniform(*rg["clear"])), 5),
            "hb": float(rng.choice(HB_SET)), "ht": float(rng.choice(HT_SET)),
            "door_mass": round(float(rng.uniform(*rg["door_mass"])), 3),
            "handle_mass": round(float(rng.uniform(*rg["handle_mass"])), 3)}


def draw_episode(seed, unit):
    """An episode of `unit`: the carcass's pose, the new door's rack's jitter, the six stands' pin classes (two of each
    class, in a drawn order; the same for every unit, so the stands say nothing about the unit's class) and the
    parking rack's jitter."""
    rng = np.random.default_rng([int(seed) % 2 ** 32, int(seed) >> 32, 202])
    jx, jy, jz, jyaw = CARCASS_JITTER
    stands = sorted(PIN_CLASSES) * 2
    stands = [stands[i] for i in rng.permutation(len(STANDS))]
    return {"carcass": [float(rng.uniform(-jx, jx)), float(rng.uniform(-jy, jy)), float(rng.uniform(-jz, jz)),
                        float(rng.uniform(-jyaw, jyaw))],
            "rack": [float(rng.uniform(-RACK_JITTER, RACK_JITTER)), float(rng.uniform(-RACK_JITTER, RACK_JITTER))],
            "stands": stands,
            "park": [float(rng.uniform(-RACK_JITTER, RACK_JITTER)), float(rng.uniform(-RACK_JITTER, RACK_JITTER))]}


def carcass_pose(episode):
    dx, dy, dz, yaw = episode["carcass"]
    p = CARCASS_NOM + np.array([dx, dy, dz])
    c, s = math.cos(yaw), math.sin(yaw)
    return p, np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def knuckle_zs(hz, n):
    """Centres of a hinge's n knuckles, its stack centred at hz (heights in the door's / carcass's frame)."""
    kl = HINGE_LEN / n
    return [hz - HINGE_LEN / 2 + kl * (i + 0.5) for i in range(n)], kl


def robosuite_assets():
    import importlib.util
    spec = importlib.util.find_spec("robosuite")
    return os.path.join(next(iter(spec.submodule_search_locations)), "models", "assets")


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def look_at(pos, target):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r = r / np.linalg.norm(r)
    return np.column_stack([r, np.cross(r, f), -f])


def axis_angle_mat(v):
    v = np.asarray(v, float)
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _box(body, name, size, pos, rgba, quat=(1, 0, 0, 0), mu=0.4, mass=None, contype=1):
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=list(pos), quat=list(quat))
    g.rgba = list(rgba)
    g.friction = [mu, 0.005, 0.0001]
    g.contype = g.conaffinity = contype
    if mass is not None:
        g.mass = mass
    return g


# collision groups: rings meet pins and the general world but not other rings; knuckle end discs meet only discs
CT_GENERAL, CT_RING, CT_PIN, CT_DISC = 1, 2, 4, 8


def _ring(body, name, cx, cy, z0, z1, r_in, r_out, rgba, mu=0.3, mass=None, discs=False):
    t = r_out - r_in
    rc = r_in + t / 2
    hw = r_out * math.tan(math.pi / N_SEG) * 1.02
    for k in range(N_SEG):
        a = 2 * math.pi * k / N_SEG
        g = _box(body, f"{name}_s{k}", (t / 2, hw, (z1 - z0) / 2), (cx + rc * math.cos(a), cy + rc * math.sin(a),
                 (z0 + z1) / 2), rgba, quat=(math.cos(a / 2), 0, 0, math.sin(a / 2)), mu=mu,
                 mass=None if mass is None else mass / N_SEG)
        g.contype, g.conaffinity = CT_RING, CT_GENERAL | CT_PIN
    if discs:       # the knuckle's two end faces: thin square plates (a stable face contact), meeting only other plates
        for j, z in enumerate((z0 + 0.0005, z1 - 0.0005)):
            g = body.add_geom(name=f"{name}_d{j}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[r_out * 0.85, r_out * 0.85, 0.0005],
                              pos=[cx, cy, z], rgba=list(rgba[:3]) + [0])
            g.contype, g.conaffinity = CT_DISC, CT_DISC
            g.friction = [mu, 0.005, 0.0001]
            g.mass = 0.0 if mass is None else 0.0001


def _cup_tool():
    """An L-shaped vacuum cup tool: a post down from the flange, then the cup facing along the flange's x axis
    (forward when the hand points down). The gripper's frame and sensor names (ft_frame, eef/grip_site, force_ee)."""
    s = mujoco.MjSpec()
    s.compiler.degree = False
    b = s.worldbody.add_body(name="right_gripper")
    b.add_site(name="ft_frame", pos=[0, 0, 0], size=[0.01, 0, 0])
    g = b.add_geom(name="cup_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.018, CUP_DROP / 2, 0],
                   pos=[0, 0, CUP_DROP / 2], rgba=[0.25, 0.25, 0.28, 1])
    g.mass = 0.35
    g = b.add_geom(name="cup_arm", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUP_REACH / 2, 0.015, 0.015],
                   pos=[CUP_REACH / 2 - 0.01, 0, CUP_DROP], rgba=[0.25, 0.25, 0.28, 1])
    g.mass = 0.1
    e = b.add_body(name="eef", pos=[CUP_REACH, 0, CUP_DROP], quat=[math.cos(math.pi / 4), 0, math.sin(math.pi / 4), 0])
    e.add_site(name="grip_site", pos=[0, 0, 0], size=[0.01, 0, 0])
    p = e.add_geom(name="cup_pad", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[CUP_R, 0.005, 0], pos=[0, 0, -0.005],
                   rgba=[0.15, 0.15, 0.15, 1])
    p.mass = 0.05
    p.friction = [1.0, 0.005, 0.0001]
    # a camera on the post, above the cup, looking where the cup faces (the tool's +x)
    b.add_camera(name="cup_cam", pos=[0.03, 0.0, 0.03], fovy=WRIST["fovy"],
                 quat=list(quat_from_mat(np.column_stack([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))))
    s.add_sensor(name="force_ee", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                 objname="ft_frame")
    s.add_sensor(name="torque_ee", type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE,
                 objname="ft_frame")
    return s


def _door(wb, name, pos, quat, spec, pin_r, clear, rgba, brass=(0.80, 0.62, 0.30, 1)):
    """A door (free body; frame origin on its hinge axis at its bottom edge, the leaf toward -y) at pos / quat, with
    its two hinges' door knuckles."""
    db = wb.add_body(name=name, pos=list(pos), quat=list(quat))
    db.add_freejoint(name=f"{name}_free")
    _box(db, f"{name}_leaf", (DOOR_T / 2, DOOR_W / 2, DOOR_H / 2), (0, -EDGE_GAP - DOOR_W / 2, DOOR_H / 2), rgba,
         mass=spec["door_mass"])
    _box(db, f"{name}_handle", (0.012, 0.008, 0.06), (-DOOR_T / 2 - 0.012, -EDGE_GAP - DOOR_W + 0.04, DOOR_H / 2),
         (0.3, 0.3, 0.32, 1), mass=spec["handle_mass"])
    bore = pin_r + clear
    for h, hz in enumerate((spec["hb"], spec["ht"])):
        zs, kl = knuckle_zs(hz, spec["n_kn"])
        for i, z in enumerate(zs):
            if i % 2 == 1:
                half = kl / 2 - AX_PLAY
                _ring(db, f"{name}_k{h}_{i}", 0, 0, z - half, z + half, bore, KN_RO, brass, mass=0.01, discs=True)
                _box(db, f"{name}_kleaf{h}_{i}", (0.001, (EDGE_GAP - KN_RO) / 2 + 0.001, half - 0.001),
                     (0, -(EDGE_GAP + KN_RO) / 2, z), brass)
    return db


def build(unit, episode):
    w = mujoco.MjSpec()
    w.compiler.degree = False
    w.compiler.autolimits = True
    w.option.timestep = TIMESTEP
    w.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    w.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    w.option.impratio = 10.0
    w.option.noslip_iterations = 30
    w.memory = 128 * 2 ** 20
    w.default.geom.solref = [0.004, 1]
    w.default.geom.solimp = [0.95, 0.99, 0.0005, 0.5, 2]
    w.visual.global_.offwidth, w.visual.global_.offheight = 640, 480
    w.visual.headlight.ambient = [0.35, 0.35, 0.35]
    wb = w.worldbody
    wb.add_light(pos=[0.5, 0, TABLE_Z + 1.8], dir=[0, 0, -1], castshadow=0, diffuse=[0.5, 0.5, 0.5])
    wb.add_light(pos=[-0.5, 0.8, TABLE_Z + 1.2], dir=[0.6, -0.4, -0.5], castshadow=0, diffuse=[0.4, 0.4, 0.4])
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], rgba=[0.35, 0.33, 0.30, 1])
    _box(wb, "table", ((TABLE_EDGE_X + 0.28) / 2, 0.80, TABLE_Z / 2), ((TABLE_EDGE_X - 0.28) / 2, 0.0, TABLE_Z / 2),
         (0.62, 0.55, 0.45, 1))
    _box(wb, "wall", (0.01, 1.2, 1.2), (0.95, 0.0, 1.2), (0.85, 0.84, 0.80, 1))

    ra = robosuite_assets()
    for side, xy in BASES.items():
        robot = mujoco.MjSpec.from_file(os.path.join(ra, "robots", "panda", "robot.xml"))
        for cam in list(robot.cameras):
            if cam.name == "robotview":
                robot.delete(cam)
        tool = mujoco.MjSpec.from_file(os.path.join(ra, "grippers", "panda_gripper.xml")) if TOOLS[side] == "gripper" \
            else _cup_tool()
        robot.body("right_hand").add_frame().attach_body(tool.body("right_gripper"), "g_", "")
        wb.add_frame(pos=[xy[0], xy[1], TABLE_Z]).attach_body(robot.body("base"), f"{side}_", "")

    # the carcass: a fixed body at its episode pose
    cpos, cR = carcass_pose(episode)
    cb = wb.add_body(name="carcass", pos=list(cpos), quat=list(quat_from_mat(cR)))
    wood = (0.70, 0.60, 0.48, 1)
    yl, yr = -0.015, -EDGE_GAP - DOOR_W + 0.030        # the door overlays the right side panel: its stop
    zb, zt = -0.005, DOOR_H + 0.005
    cx = 0.015 + DEPTH / 2                              # the panels' front edge 15 mm behind the axis
    _box(cb, "c_side_l", (DEPTH / 2, 0.009, (zt - zb) / 2), (cx, yl + 0.009, (zb + zt) / 2), wood)
    _box(cb, "c_side_r", (DEPTH / 2, 0.009, (zt - zb) / 2), (cx, yr - 0.009, (zb + zt) / 2), wood)
    _box(cb, "c_top", (DEPTH / 2, (yl - yr) / 2 + 0.018, 0.009), (cx, (yl + yr) / 2, zt + 0.009), wood)
    _box(cb, "c_bottom", (DEPTH / 2, (yl - yr) / 2 + 0.018, 0.009), (cx, (yl + yr) / 2, zb - 0.009), wood)
    _box(cb, "c_shelf", (DEPTH / 2 - 0.02, (yl - yr) / 2, 0.009), (cx + 0.02, (yl + yr) / 2, (zb + zt) / 2), wood)
    steel = (0.55, 0.55, 0.58, 1)
    pin_r = PIN_CLASSES[unit["pin_class"]]
    bore = pin_r + unit["clear"]
    for h, hz in enumerate((unit["hb"], unit["ht"])):
        zs, kl = knuckle_zs(hz, unit["n_kn"])
        for i, z in enumerate(zs):
            if i % 2 == 0:
                _ring(cb, f"ck{h}_{i}", 0, 0, z - kl / 2, z + kl / 2, bore, KN_RO, steel, discs=True)
                _box(cb, f"cleaf{h}_{i}", (0.008, 0.001, kl / 2 - 0.001), (KN_RO + 0.0065, 0, z), steel)

    # the new door in its rack's channel; the old door hung in the opening; the empty parking rack
    sxy = RACK + np.array(episode["rack"])
    _door(wb, "door", (sxy[0], sxy[1], TABLE_Z + 0.0005), (1, 0, 0, 0), unit, pin_r, unit["clear"],
          (0.88, 0.84, 0.74, 1))
    _door(wb, "old", cpos, quat_from_mat(cR), unit, pin_r, unit["clear"], (0.45, 0.33, 0.24, 1),
          brass=(0.42, 0.36, 0.26, 1))
    pxy = PARK + np.array(episode["park"])
    for name, xy, gap in (("rack", sxy, 0.005), ("park", pxy, PARK_GAP)):
        for k, dx in enumerate((-1, 1)):
            _box(wb, f"{name}_rail{k}", (0.004, 0.19, 0.020), (xy[0] + dx * (DOOR_T / 2 + gap),
                 xy[1] - EDGE_GAP - DOOR_W / 2, TABLE_Z + 0.020), (0.3, 0.3, 0.32, 1))

    # pins and their stands; then the old door's two pins, each home in its hinge (on its head)
    old_tops = [knuckle_zs(hz, unit["n_kn"])[0][-1] + knuckle_zs(hz, unit["n_kn"])[1] / 2
                for hz in (unit["hb"], unit["ht"])]
    places = [((sxy[0], sxy[1], TABLE_Z + STAND_H + 0.0005), (1, 0, 0, 0), PIN_CLASSES[cls])
              for sxy, cls in zip(STANDS, episode["stands"])]
    places += [(cpos + cR @ np.array([0, 0, z + 0.0003]), quat_from_mat(cR), OLD_PIN_R) for z in old_tops]
    for k, (sxy, cls) in enumerate(zip(STANDS, episode["stands"])):
        r = PIN_CLASSES[cls]
        _ring(wb, f"stand{k}", sxy[0], sxy[1], TABLE_Z, TABLE_Z + STAND_H, r + STAND_CLEAR, r + 0.008,
              (0.3, 0.3, 0.32, 1))
    for k, (ppos, pquat, r) in enumerate(places):
        pb = wb.add_body(name=f"pin{k}", pos=list(ppos), quat=list(pquat))
        pb.add_freejoint(name=f"pin{k}_free")
        g = pb.add_geom(name=f"pin{k}_shaft", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                        size=[r, SHAFT / 2 - r, 0], pos=[0, 0, -SHAFT / 2], rgba=list(steel))
        g.mass, g.friction = 0.03, [0.3, 0.005, 0.0001]
        g.contype, g.conaffinity = CT_PIN, CT_GENERAL | CT_RING
        g = pb.add_geom(name=f"pin{k}_head", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r + 0.003, 0.0015, 0],
                        pos=[0, 0, 0.0015], rgba=list(steel))
        g.mass = 0.01
        g.contype, g.conaffinity = CT_PIN, CT_GENERAL | CT_RING
        _box(pb, f"pin{k}_arm", (-POST[0] / 2 + 0.001, 0.004, 0.003), (POST[0] / 2, 0, 0.006), steel, mass=0.015)
        _box(pb, f"pin{k}_post", (0.006, 0.006, 0.018), tuple(POST), steel, mass=0.025, mu=1.0)
        w.add_equality(name=f"seat{k}", type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY,
                       name1=f"pin{k}", name2="world", active=False, solref=[0.004, 1])
    for name in ("door", "old"):
        w.add_equality(name=f"vac_{name}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                       name1="r_g_eef", name2=name, active=False, solref=list(CUP_SOLREF))
    # the contact shapes of the drawn parts a hand or a carried door can reach (door_parts.contact_shapes: the fitter's
    # things on the bench, the tool bag, the rail behind the cabinet, the vacuum tool's
    # bellows, fitting and hose); they are not drawn themselves (door_scene.py draws the parts)
    for name, frame, kind, a, b, r in door_parts.contact_shapes(unit, TABLE_Z, DEPTH, DOOR_W + EDGE_GAP, DOOR_H + 0.02, BAG_AT):
        body = {"world": wb, "carcass": cb, "cup": w.body("r_g_right_gripper")}[frame]
        a, b = np.asarray(a, float), np.asarray(b, float)
        if kind == "cyl":
            g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, 0, 0], fromto=list(a) + list(b))
        else:
            pos, quat = (a + b) / 2, (1, 0, 0, 0)
            if kind == "bag":
                c, s_ = math.cos(BAG_AT[2]), math.sin(BAG_AT[2])
                pos = np.array([BAG_AT[0] + c * pos[0] - s_ * pos[1], BAG_AT[1] + s_ * pos[0] + c * pos[1], pos[2]])
                quat = (math.cos(BAG_AT[2] / 2), 0, 0, math.sin(BAG_AT[2] / 2))
            g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list((b - a) / 2), pos=list(pos), quat=list(quat))
        g.rgba = [0.5, 0.5, 0.5, 0.0]
        g.group = 3
        g.mass = 0.0
        g.friction = [0.4, 0.005, 0.0001]
    for name, (pos, tgt, fovy, _w, _h) in CAMERAS.items():
        wb.add_camera(name=name, pos=list(pos), fovy=fovy, quat=list(quat_from_mat(look_at(pos, tgt))))
    m = w.compile()
    for side in BASES:
        if TOOLS[side] == "gripper":
            for f in ("finger1", "finger2"):
                gid = m.geom(f"{side}_g_{f}_pad_collision").id
                m.geom_condim[gid] = 4
                m.geom_friction[gid, 1] = PAD_TORSION
    return m


class Arm:
    """One Panda: the operational-space controller and joint step of pantry_jar_refilling's jar_sim.Arm."""

    def __init__(self, m, d, p):
        self.m, self.d, self.p = m, d, p
        self.tool = TOOLS[p]
        self.jq = np.array([m.joint(f"{p}_joint{i}").qposadr[0] for i in range(1, 8)])
        self.jv = np.array([m.joint(f"{p}_joint{i}").dofadr[0] for i in range(1, 8)])
        self.act = np.array([m.actuator(f"{p}_torq_j{i}").id for i in range(1, 8)])
        if self.tool == "gripper":
            self.gact = [m.actuator(f"{p}_g_gripper_finger_joint1").id, m.actuator(f"{p}_g_gripper_finger_joint2").id]
            self.gq = [m.joint(f"{p}_g_finger_joint1").qposadr[0], m.joint(f"{p}_g_finger_joint2").qposadr[0]]
        self.site = m.site(f"{p}_g_grip_site").id
        self.ft_site = m.site(f"{p}_g_ft_frame").id
        self.f_adr = m.sensor(f"{p}_g_force_ee").adr[0]
        self.t_adr = m.sensor(f"{p}_g_torque_ee").adr[0]
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))
        self.q_null = HOME_Q.copy()
        self.grip_cmd = -1.0
        self.grip_target = 0.04
        self.mode = "osc"
        self.q_goal = None
        self.peak_tau = np.zeros(7)

    def init_targets(self):
        self.target_pos = self.d.site_xpos[self.site].copy()
        self.target_R = self.d.site_xmat[self.site].reshape(3, 3).copy()

    def pos(self):
        return self.d.site_xpos[self.site].copy()

    def R(self):
        return self.d.site_xmat[self.site].reshape(3, 3).copy()

    def gripper(self):
        if self.tool != "gripper":
            return 0.0
        return float(self.d.qpos[self.gq[0]] - self.d.qpos[self.gq[1]])

    def wrench_world(self):
        d = self.d
        R = d.site_xmat[self.ft_site].reshape(3, 3)
        return np.concatenate([R @ d.sensordata[self.f_adr:self.f_adr + 3], R @ d.sensordata[self.t_adr:self.t_adr + 3]])

    def command(self, a):
        d = self.d
        if len(a) == 8:
            q = d.qpos[self.jq].copy()
            self.q_goal = q + np.clip(np.asarray(a[:7], float) - q, -MAX_DQ, MAX_DQ)
            self.q_null = np.asarray(a[:7], float).copy()
            self.mode = "joint"
            grip = a[7]
        else:
            if len(a) == 14:
                self.q_null = np.asarray(a[7:14], float).copy()
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
        self.grip_cmd = float(np.clip(grip, -1, 1))
        self.grip_target = 0.04 * (1.0 - self.grip_cmd) / 2.0

    def torque(self):
        m, d = self.m, self.d
        if self.mode == "joint":
            tau = 200.0 * (self.q_goal - d.qpos[self.jq]) - 20.0 * d.qvel[self.jv] + d.qfrc_bias[self.jv]
        else:
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
        lim = self.m.actuator_ctrlrange[self.act, 1]
        self.peak_tau = np.maximum(self.peak_tau, np.abs(tau) / lim)
        return tau


class World:
    def __init__(self, unit, episode):
        self.unit, self.episode = unit, episode
        self.m = m = build(unit, episode)
        self.d = d = mujoco.MjData(m)
        self.arms = {s: Arm(m, d, s) for s in BASES}
        self.cpos, self.cR = carcass_pose(episode)
        self.doors = {"door": m.body("door").id, "old": m.body("old").id}
        self.door = self.doors["door"]
        self.npin = len(STANDS) + 2
        self.pins = [m.body(f"pin{k}").id for k in range(self.npin)]
        self.pin_v = [m.joint(f"pin{k}_free").dofadr[0] for k in range(self.npin)]
        self.pin_class = list(episode["stands"]) + ["old", "old"]
        self.old_fell = False                  # the old door came down below the table top: the episode is lost
        self.seat_eq = [m.equality(f"seat{k}").id for k in range(self.npin)]
        self.snapped = [False] * self.npin
        self.vac = {n: m.equality(f"vac_{n}").id for n in self.doors}
        self.cup_pad = m.geom("r_g_cup_pad").id
        self.cup_body = m.body("r_g_eef").id
        self.held = None                       # the door the cup holds
        self.seal_events = []
        self.cup_peak = [0.0, 0.0]
        self.tops = [knuckle_zs(hz, unit["n_kn"])[0][-1] + knuckle_zs(hz, unit["n_kn"])[1] / 2
                     for hz in (unit["hb"], unit["ht"])]          # each hinge's top face (carcass frame)
        self.bottoms = [hz - HINGE_LEN / 2 for hz in (unit["hb"], unit["ht"])]
        self.t = 0
        mujoco.mj_resetData(m, d)
        for a in self.arms.values():
            d.qpos[a.jq] = HOME_Q
            if a.tool == "gripper":
                d.qpos[a.gq[0]], d.qpos[a.gq[1]] = 0.04, -0.04
        mujoco.mj_forward(m, d)
        for _ in range(250):
            for a in self.arms.values():
                d.ctrl[a.act] = 200.0 * (HOME_Q - d.qpos[a.jq]) - 20.0 * d.qvel[a.jv] + d.qfrc_bias[a.jv]
                if a.tool == "gripper":
                    d.ctrl[a.gact] = [0.04, -0.04]
            mujoco.mj_step(m, d)
            self._detent()
        for a in self.arms.values():
            a.init_targets()

    @property
    def sealed(self):
        return self.held is not None

    # --- frames ---------------------------------------------------------------------------------
    def to_carcass(self, p):
        return self.cR.T @ (np.asarray(p) - self.cpos)

    def from_carcass(self, p):
        return self.cpos + self.cR @ np.asarray(p)

    def axis_world(self, z_local=0.0):
        return self.from_carcass([0, 0, z_local])

    def door_pose(self, name="door"):
        b = self.doors[name]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    # --- pins -----------------------------------------------------------------------------------
    def pin_local(self, k):
        return self.to_carcass(self.d.xpos[self.pins[k]])

    def pin_tilt_deg(self, k):
        R = self.d.xmat[self.pins[k]].reshape(3, 3)
        return math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))

    def pin_home(self, k):
        """The hinge pin k sits in (its head at most 3 mm above that hinge's top face, on the axis), or None."""
        p = self.pin_local(k)
        if np.hypot(p[0], p[1]) > 0.0015 or self.pin_tilt_deg(k) > 3.0:
            return None
        for h in range(2):
            if -0.0015 < p[2] - self.tops[h] < 0.003:
                return h
        return None

    def door_hung(self, name="door"):
        """The door hangs on the opening: its frame within 2 mm of the carcass frame (the hinge axis at the hung
        height), its leaf within 2 deg of upright and 3 deg of closed."""
        pd, Rd = self.door_pose(name)
        pl = self.to_carcass(pd)
        if np.hypot(pl[0], pl[1]) > 0.002 or abs(pl[2]) > 0.002:
            return False
        Rl = self.cR.T @ Rd
        return bool(Rl[2, 2] > math.cos(math.radians(2)) and Rl[0, 0] > math.cos(math.radians(3)))

    def hinges_pinned(self):
        """For each hinge: is a pin of the unit's class home in it."""
        out = [False, False]
        for k in range(self.npin):
            h = self.pin_home(k)
            if h is not None and self.pin_class[k] == self.unit["pin_class"]:
                out[h] = True
        return out

    def _detent(self):
        m, d = self.m, self.d
        for k, b in enumerate(self.pins):
            p = self.to_carcass(d.xpos[b])
            tip = p[2] - SHAFT
            drag = 0.0
            if np.hypot(p[0], p[1]) < 0.002 and self.pin_class[k] == self.unit["pin_class"]:
                for h in range(2):
                    if tip < self.tops[h] - 0.001 and p[2] > self.bottoms[h]:
                        drag = PIN_DRAG
                        if not self.snapped[k] and p[2] - self.tops[h] < PIN_SNAP:
                            self._snap(k)
            v = self.pin_v[k]
            m.dof_frictionloss[v:v + 3] = drag
            if self.snapped[k]:
                rows = np.where((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                                & (d.efc_id[:d.nefc] == self.seat_eq[k]))[0]
                if len(rows) >= 3 and -d.efc_force[rows[2]] > PIN_PULL:   # the seat's pull on the pin, up
                    d.eq_active[self.seat_eq[k]] = 0
                    self.snapped[k] = False

    def _snap(self, k):
        """Pin k's detent has clicked: its head centre is held where it is (a point constraint, so the seated pin
        still turns about its axis: the opening door pushes its arm round)."""
        m, d = self.m, self.d
        e = self.seat_eq[k]
        m.eq_data[e, 0:3] = 0
        m.eq_data[e, 3:6] = d.xpos[self.pins[k]]
        d.eq_active[e] = 1
        self.snapped[k] = True

    # --- the vacuum cup -------------------------------------------------------------------------
    def _cup_face_on(self):
        """The door whose front face the cup touches square (within 3 mm and 10 deg, fully on the leaf), or None."""
        d = self.d
        Rc = d.xmat[self.cup_body].reshape(3, 3)
        for name, b in self.doors.items():
            Rd = d.xmat[b].reshape(3, 3)
            pc = Rd.T @ (d.xpos[self.cup_body] - d.xpos[b])
            if abs(pc[0] + DOOR_T / 2) > CUP_SEAL_GAP:
                continue
            if not (-EDGE_GAP - DOOR_W + CUP_R < pc[1] < -EDGE_GAP - CUP_R and CUP_R < pc[2] < DOOR_H - CUP_R):
                continue
            if float(Rc[:, 2] @ Rd[:, 0]) > math.cos(math.radians(CUP_SEAL_DEG)):
                return name
        return None

    def _seal(self, name):
        """Weld the door to the cup as they touch now, the cup's face put exactly on the door's front face; the
        weld's reference point is the cup's face centre, so its torque is the moment at the cup. The pad stops
        colliding."""
        m, d = self.m, self.d
        b, e = self.doors[name], self.vac[name]
        R1 = d.xmat[self.cup_body].reshape(3, 3)
        p2, R2 = d.xpos[b], d.xmat[b].reshape(3, 3)
        pc = R2.T @ (d.xpos[self.cup_body] - p2)
        pc[0] = -DOOR_T / 2
        q1c, qrel = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q1c, d.xquat[self.cup_body])
        mujoco.mju_mulQuat(qrel, q1c, d.xquat[b])
        m.eq_data[e, 0:3] = pc
        m.eq_data[e, 3:6] = R1.T @ (p2 + R2 @ pc - d.xpos[self.cup_body])
        m.eq_data[e, 6:10] = qrel
        m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1
        m.geom_contype[self.cup_pad] = m.geom_conaffinity[self.cup_pad] = 0
        self.held = name
        self.over = 0
        self.win = []
        self.seal_events.append(("seal", name, self.t))

    def _unseal(self, why):
        self.d.eq_active[self.vac[self.held]] = 0
        self.m.geom_contype[self.cup_pad] = self.m.geom_conaffinity[self.cup_pad] = 1
        self.seal_events.append((why, self.held, self.t))
        self.held = None

    def cup_load(self):
        """The cup's held load now: |force| (N) and |moment at the cup| (N m); zeros when nothing is held."""
        d = self.d
        if self.held is None:
            return 0.0, 0.0
        rows = np.where((d.efc_type[:d.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
                        & (d.efc_id[:d.nefc] == self.vac[self.held]))[0]
        if len(rows) < 6:
            return 0.0, 0.0
        f = d.efc_force[rows]
        return float(np.linalg.norm(f[:3])), float(np.linalg.norm(f[3:6]))

    def _vacuum(self):
        want = self.arms["r"].grip_cmd > 0
        if self.held is not None:
            if not want:
                self._unseal("release")
                return
            fl, tl = self.cup_load()
            self.over = self.over + 1 if (fl > CUP_MAX_FORCE or tl > CUP_MAX_TORQUE) else 0
            self.win = (self.win + [(fl, tl)])[-CUP_OVERLOAD_SUBSTEPS:]   # the sustained load: the least over 20 ms
            if len(self.win) == CUP_OVERLOAD_SUBSTEPS:
                sf, st = min(x[0] for x in self.win), min(x[1] for x in self.win)
                self.cup_peak = [max(self.cup_peak[0], sf), max(self.cup_peak[1], st)]
            if self.over >= CUP_OVERLOAD_SUBSTEPS:
                self._unseal("overload")
        elif want:
            name = self._cup_face_on()
            if name is not None:
                self._seal(name)

    # --- stepping -------------------------------------------------------------------------------
    def step(self, cmds):
        m, d = self.m, self.d
        for s, a in cmds.items():
            self.arms[s].command(a)
        for _ in range(SUBSTEPS):
            for a in self.arms.values():
                d.ctrl[a.act] = a.torque()
                if a.tool == "gripper":
                    d.ctrl[a.gact] = [a.grip_target, -a.grip_target]
            mujoco.mj_step(m, d)
            self._detent()
            self._vacuum()
        self.t += 1
        if d.xipos[self.doors["old"]][2] < TABLE_Z - 0.05:
            self.old_fell = True
        if self.unstable():
            raise FloatingPointError("unstable")

    def unstable(self):
        d = self.d
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return True
        if any(np.max(np.abs(d.qvel[a.jv])) > MAX_ARM_SPEED for a in self.arms.values()):
            return True
        v = np.zeros(6)
        for b in list(self.doors.values()) + self.pins:
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            if np.linalg.norm(v[3:]) > MAX_OBJECT_SPEED:
                return True
        return False

    # --- the oracle's shadow: the state beyond qpos / qvel ----------------------------------------
    def state(self):
        """Everything that, with the unit and the episode, sets this world exactly: the simulation state, the
        equality constraints (the pins' detents, the cup's seal), the seal's bookkeeping and the step."""
        m, d = self.m, self.d
        return {"t": int(self.t), "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(), "act": d.act.tolist(),
                "ctrl": d.ctrl.tolist(), "eq_active": [int(x) for x in d.eq_active], "eq_data": m.eq_data.tolist(),
                "held": self.held, "old_fell": bool(self.old_fell), "snapped": [bool(x) for x in self.snapped], "over": int(getattr(self, "over", 0)),
                "win": [list(map(float, x)) for x in getattr(self, "win", [])],
                "grip_cmd": {s: float(a.grip_cmd) for s, a in self.arms.items()}}

    def set_state(self, st):
        m, d = self.m, self.d
        d.qpos[:] = np.asarray(st["qpos"], float)
        d.qvel[:] = np.asarray(st["qvel"], float)
        if d.act.size:
            d.act[:] = np.asarray(st["act"], float)
        d.ctrl[:] = np.asarray(st["ctrl"], float)
        m.eq_data[:] = np.asarray(st["eq_data"], float)
        d.eq_active[:] = np.asarray(st["eq_active"], dtype=d.eq_active.dtype)
        self.held = st["held"]
        self.old_fell = bool(st["old_fell"])
        on = 0 if self.held is not None else 1
        m.geom_contype[self.cup_pad] = m.geom_conaffinity[self.cup_pad] = on
        self.snapped = [bool(x) for x in st["snapped"]]
        self.over = int(st["over"])
        self.win = [tuple(x) for x in st["win"]]
        for s, g in st["grip_cmd"].items():
            self.arms[s].grip_cmd = float(g)
        self.t = int(st["t"])
        mujoco.mj_forward(m, d)
        self._detent()

    # --- the test after the episode ------------------------------------------------------------
    def swing_test(self, torque=0.3, open_deg=70.0, seconds=2.0):
        """The arms parked out of the way (no collisions). A test hand turns the door about the hinge axis with at
        most `torque` (a damped PD toward `open_deg`) for `seconds`, then back to closed; the door is let go and
        settles. Reports the opening reached, the closed yaw, the free corner's sag, the hinges pinned."""
        m, d = self.m, self.d
        if self.held is not None:
            self._unseal("test")
        for gid in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            if name.startswith(("l_", "r_")):
                m.geom_contype[gid] = m.geom_conaffinity[gid] = 0
        for a in self.arms.values():
            a.mode = "joint"
            a.q_goal = HOME_Q.copy()
        b = self.door
        axis_xy = self.cpos[:2]

        def yaw():
            Rl = self.cR.T @ d.xmat[b].reshape(3, 3)
            return -math.atan2(Rl[1, 0], Rl[0, 0])               # opening (free edge toward -x) is positive

        def free_corner():
            return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0, -EDGE_GAP - DOOR_W, 0])

        z_corner0 = free_corner()[2]
        out = {"opened_deg": 0.0}
        v = np.zeros(6)
        n = int(seconds / TIMESTEP)
        for i in range(3 * n):
            target = math.radians(open_deg) if i < n else 0.0
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            tq = float(np.clip(2.0 * (target - yaw()) - 0.4 * (-v[2]), -torque, torque))
            arm = d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([0, -EDGE_GAP - DOOR_W / 2, DOOR_H / 2])
            Rn = d.xmat[b].reshape(3, 3)[:, 0]
            r = np.hypot(*(arm[:2] - axis_xy))
            f = -Rn * (tq / max(r, 1e-3))                        # a push normal to the leaf at its centre
            d.xfrc_applied[b, :3] = f
            d.xfrc_applied[b, 3:] = np.cross(arm - d.xipos[b], f)
            for a in self.arms.values():
                d.ctrl[a.act] = a.torque()
            mujoco.mj_step(m, d)
            self._detent()
            if i < n:
                out["opened_deg"] = max(out["opened_deg"], math.degrees(yaw()))
        d.xfrc_applied[b] = 0
        for _ in range(500):
            mujoco.mj_step(m, d)
            self._detent()
        out["opened_deg"] = round(out["opened_deg"], 1)
        out["closed_deg"] = round(math.degrees(yaw()), 2)
        out["corner_sag_mm"] = round(float(z_corner0 - free_corner()[2]) * 1000, 2)
        out["hinges_pinned"] = self.hinges_pinned()
        out["door_hung"] = self.door_hung()
        return out
