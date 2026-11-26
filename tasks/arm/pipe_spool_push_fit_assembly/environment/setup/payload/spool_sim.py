"""Pipe spool push-fit assembly: the simulation of the cell (two Pandas at a bench, two loose copper pipes and two
push-fit elbows on the bench).

Plain MuJoCo 3.16 on CPU, robosuite 1.5.2 model files only (robosuite is not imported). Frames: world z up, metres,
radians, seconds; quaternions (w, x, y, z). The robots stand on the bench (top at TABLE_Z) facing +x.

- Both arms have the Panda parallel gripper, driven as the Franka Hand grasps: each finger's position servo
  (GRIP_KP) is limited to GRIP_FORCE, and grip +1 aims each finger GRIP_PAST beyond closed, so a closed hand squeezes
  whatever it holds with GRIP_FORCE.
- Pipe A and pipe B: 22 mm copper pipes (a convex mesh: a cylinder with a CHAMFER at each end), each lying loose on
  the bench (nothing holds them; a pipe that is pushed rolls), roughly along y. Pipe A carries a blue band near one of
  its ends (its "banded end").
- Two elbows (identical 90 deg push-fit elbows) on the bench, each lying flat or standing on one socket's mouth (that
  socket pointing down, the other one level). An elbow's frame: origin at the corner's
  centre, z normal to the elbow's plane, socket 0 along +x and socket 1 along +y, each socket's mouth SOCK_L from the
  origin. A socket is a bore (a ring of 16 boxes, radius R_PIPE + the unit's clearance) and, OR_DEPTH inside the
  mouth, an O-ring: a plate that stops a pipe end until the joint engages. Inside every socket a kinematic chain
  without geoms carries a joined pipe: a `slider` body on a slide joint along the socket's axis (inward; range 0 to
  the unit's stop depth; its frictionloss is the unit's insertion force) and on it a `rotor` body on a hinge about the
  axis (its frictionloss is the unit's turning torque).
- The push-fit joint (World._joints, every physics substep):
  * engage: a pipe end pressed on a socket's O-ring with an axial contact force above the insertion force, its axis
    within TILT_TOL_DEG and LAT_TOL of the socket's, engages: the slider is set to the end's depth, a weld between
    the rotor and the pipe is switched on (the pipe put on the socket's axis, its spin kept), and the O-ring plate
    stops colliding. From then on the pipe moves in the socket only along the axis (against the insertion force) up
    to the stop, and turns about it (against the turning torque). A skewed pipe rests on the O-ring however hard it
    is pushed;
  * come apart: a joined pipe pulled back until its end is less than OUT_DEPTH inside disengages (the weld off, the
    O-ring back);
  * home: the slider within HOME_TOL of the stop. A home joint locks (World._lock: the push-fit's grip ring): its
    pipe can no longer be drawn back (the slide's lower limit is raised to where it is, at most LOCK_BACK short of
    the stop), and it still turns against the turning torque;
  * made in the air: whenever a joint engages, and every substep in which a joined pipe advances in its socket
    (faster than ADVANCE_V), the joint is marked as made on a support if any part of its connected spool touches
    anything other than the two grippers and the other spool parts (the bench, a robot's arm links). The
    mark goes when the joint comes apart.
- The cameras' pictures are drawn by spool_render.py from spool_scene.py's parts (render, below); the drawn camera
  rig has contact boxes in the model (spool_parts.contact_shapes; group 3, not drawn).
- A unit is one batch of fittings and pipe (`draw_unit`): insertion force, stop depth, turning torque, bore
  clearance, elbow mass and pipe mass per metre. An episode (`draw_episode`) draws the pipes' lengths and places,
  which end of A is banded, the elbows' places, poses (flat or standing) and turns on the bench, and the work
  order's twist.
"""

import math
import os

import mujoco
import numpy as np

TIMESTEP = 0.002
SUBSTEPS = 50                       # 10 Hz control
MAX_DPOS = 0.03
MAX_DROT = 0.2
MAX_DQ = 0.1
MAX_ARM_SPEED = 8.0
MAX_OBJECT_SPEED = 5.0
TABLE_Z = 0.80
TABLE_EDGE_X = 0.80
TABLE_HALF_Y = 0.80
HOME_Q = np.array([0.0, -0.35, 0.0, -2.30, 0.0, 1.95, math.pi / 4])
BASES = {"l": np.array([0.0, 0.28]), "r": np.array([0.0, -0.28])}

GRIP_KP = 4000.0                    # the fingers' position gain, N/m, and their force limit, N (the Franka Hand's
GRIP_FORCE = 70.0                   # continuous grasp force; robosuite's model has 1000 and 20)
GRIP_PAST = 0.03                    # m: grip +1 aims each finger this far past closed
PAD_TORSION = 0.01                  # m, the finger pads' torsional friction
PAD_MU = 1.0

# the pipes
R_PIPE = 0.011                      # 22 mm outside diameter
CHAMFER = 0.0015
PIPE_MU = 0.4
LEN_A = (0.22, 0.30)
LEN_B = (0.19, 0.23)
BAND_FROM_END = 0.025               # the band's centre from the banded end
BAND_HALF = 0.006

# the elbows
SOCK_L = 0.050                      # the corner's centre to a socket's mouth
BORE_D = 0.030                      # the bore's length from the mouth
R_OUT = 0.015                       # a socket's outside radius
CORNER_HALF = (0.019, 0.019, 0.015)
OR_DEPTH = 0.004                    # the O-ring's face, inside the mouth
OR_HALF = 0.0015                    # half the O-ring plate's thickness
OUT_DEPTH = 0.0035                  # a joined pipe whose end comes out to less than this inside disengages
LOCK_BACK = 0.0005                  # a home joint's pipe is held this close to the stop (it cannot be drawn back)
HOME_TOL = 0.001
TILT_TOL_DEG = 1.0
LAT_TOL = 0.0005
ADVANCE_V = 0.001                   # m/s: a joined pipe advancing faster than this is being pushed in
INNER_MASS = 0.01                   # each slider and rotor
N_SEG = 16
ELBOW_MU = 0.5
JOIN_SOLREF = (0.004, 1.0)
STIFF_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2)

UNIT_RANGES = {"f_ins": (10.0, 25.0), "stop": (0.020, 0.028), "torque": (0.50, 1.20), "clear": (0.0002, 0.0004),
               "elbow_mass": (0.08, 0.16), "pipe_kg_m": (0.45, 0.75)}

# nominal places (x, y) on the bench; each episode moves each by up to PLACE_JITTER
PIPE_A_XY = np.array([0.40, 0.24])
PIPE_B_XY = np.array([0.58, 0.18])
ELBOW_XY = [np.array([0.42, -0.24]), np.array([0.62, 0.44])]
PLACE_JITTER = 0.015
PIPE_YAW_JITTER = math.radians(30.0)
TWISTS = (0, 180)

WRENCH_NOISE = 0.3                  # N (force); torque noise is a tenth, N m
RGB_NOISE = 2.0

CAMERAS = {                         # name: (position, look-at, vertical fov deg, width, height)
    "overhead": ((0.30, 0.0, TABLE_Z + 1.30), (0.48, 0.0, TABLE_Z), 62.0, 640, 480),
    "front": ((-0.35, 0.0, TABLE_Z + 0.80), (0.55, 0.0, TABLE_Z + 0.10), 50.0, 640, 480),
    "side": ((0.45, -0.95, TABLE_Z + 0.45), (0.47, 0.0, TABLE_Z + 0.15), 50.0, 640, 480),
}
WRIST = {"fovy": 75.0, "width": 320, "height": 240}
MJ_CAMERA = {"overhead": "overhead", "front": "front", "side": "side", "wrist_l": "l_eye_in_hand",
             "wrist_r": "r_eye_in_hand"}
CAMERA_BITS = {"overhead": 1, "front": 2, "side": 4, "wrist_l": 8, "wrist_r": 16}
ALL_CAMS = 31

# collision classes: pipes meet everything; O-ring plates meet pipes only
CT_GENERAL, CT_PLATE = 1, 2

COPPER = (0.78, 0.46, 0.26, 1)
BAND = (0.10, 0.30, 0.85, 1)
PLASTIC = (0.88, 0.88, 0.84, 1)
RUBBER = (0.08, 0.08, 0.08, 1)


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
    spool_render.py (beside this file: MuJoCo's Filament renderer on the CPU) in a process of its own per episode,
    started at the episode's first image; `cache` keeps that process. Raises RenderError when the renderer fails."""
    import spool_render
    w, h, _fovy = camera_size(cam)
    try:
        view = cache.get("view")
        if view is None:
            view = cache["view"] = spool_render.Worker(world.unit, world.episode)
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
    """A unit: one batch of fittings and pipe."""
    rng = np.random.default_rng([int(seed) % 2 ** 32, int(seed) >> 32, 101])
    rg = UNIT_RANGES
    return {"f_ins": round(float(rng.uniform(*rg["f_ins"])), 2),
            "stop": round(float(rng.uniform(*rg["stop"])), 5),
            "torque": round(float(rng.uniform(*rg["torque"])), 3),
            "clear": round(float(rng.uniform(*rg["clear"])), 5),
            "elbow_mass": round(float(rng.uniform(*rg["elbow_mass"])), 3),
            "pipe_kg_m": round(float(rng.uniform(*rg["pipe_kg_m"])), 3)}


def draw_episode(seed, unit):
    """An episode of `unit`: the pipes' lengths, places and yaws (both lying roughly along y), which end of A is banded
    (+1: the end toward the pipe's local +z), the elbows' places, yaws and poses, the work order's twist (deg).
    `e_down[e]` is None for an elbow lying flat, else the socket (0 or 1) it stands on: at least one elbow stands in
    every episode, both in half of them."""
    rng = np.random.default_rng([int(seed) % 2 ** 32, int(seed) >> 32, 202])
    j = PLACE_JITTER

    def jit():
        return [float(rng.uniform(-j, j)), float(rng.uniform(-j, j))]

    return {"len_a": round(float(rng.uniform(*LEN_A)), 4), "len_b": round(float(rng.uniform(*LEN_B)), 4),
            "a_xy": jit(), "a_yaw": float(rng.uniform(-PIPE_YAW_JITTER, PIPE_YAW_JITTER)),
            "b_xy": jit(), "b_yaw": float(rng.uniform(-PIPE_YAW_JITTER, PIPE_YAW_JITTER)),
            "band": int(rng.choice([-1, 1])),
            "e_xy": [jit(), jit()], "e_yaw": [float(rng.uniform(-math.pi, math.pi)) for _ in range(2)],
            "twist_deg": int(rng.choice(TWISTS)), "e_down": _draw_poses(rng)}


def _draw_poses(rng):
    """Which elbows stand, and on which socket: both (probability 1/2) or exactly one, either one alike."""
    both = bool(rng.random() < 0.5)
    lone = int(rng.integers(2))
    downs = [int(rng.integers(2)) for _ in range(2)]
    return [downs[e] if (both or e == lone) else None for e in (0, 1)]


def robosuite_assets():
    import importlib.util
    spec = importlib.util.find_spec("robosuite")
    return os.path.join(next(iter(spec.submodule_search_locations)), "models", "assets")


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def mat_from_quat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


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


def rot_between(a, b):
    """The smallest rotation taking unit vector a to unit vector b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    v = np.cross(a, b)
    s, c = np.linalg.norm(v), float(a @ b)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.cross(a, [1.0, 0, 0])
        if np.linalg.norm(p) < 1e-6:
            p = np.cross(a, [0, 1.0, 0])
        return axis_angle_mat(math.pi * p / np.linalg.norm(p))
    return axis_angle_mat(v / s * math.atan2(s, c))


def yaw_mat(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def chamfered_cylinder(r, L, ch, n=24):
    v = []
    for z, rr in ((-L / 2, r - ch), (-L / 2 + ch, r), (L / 2 - ch, r), (L / 2, r - ch)):
        for k in range(n):
            a = 2 * math.pi * k / n
            v += [rr * math.cos(a), rr * math.sin(a), z]
    return v


def _box(body, name, size, pos, rgba, quat=(1, 0, 0, 0), mu=0.4, mass=None, contype=CT_GENERAL,
         conaffinity=CT_GENERAL):
    g = body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size), pos=list(pos), quat=list(quat))
    g.rgba = list(rgba)
    g.friction = [mu, 0.005, 0.0001]
    g.contype, g.conaffinity = contype, conaffinity
    if mass is not None:
        g.mass = mass
    return g


def _socket_frame(j):
    """Socket j's axis (outward) and its frame's rotation (local z along the axis) in the elbow's frame."""
    u = np.array([1.0, 0, 0]) if j == 0 else np.array([0, 1.0, 0])
    return u, rot_between([0, 0, 1.0], u)


def _elbow(wb, name, pos, R, unit):
    """An elbow lying with its frame at (pos, R): the corner, two sockets (bore rings, O-ring plates) and per socket
    the slider/rotor chain."""
    eb = wb.add_body(name=name, pos=list(pos), quat=list(quat_from_mat(R)))
    eb.add_freejoint(name=f"{name}_free")
    ring_mass = 0.0005
    inner = 4 * INNER_MASS
    corner_mass = unit["elbow_mass"] - 2 * N_SEG * ring_mass - 2 * 0.001 - inner
    _box(eb, f"{name}_corner", CORNER_HALF, (0, 0, 0), PLASTIC, mu=ELBOW_MU, mass=corner_mass)
    r_in = R_PIPE + unit["clear"]
    t = R_OUT - r_in
    rc = r_in + t / 2
    hw = R_OUT * math.tan(math.pi / N_SEG) * 1.02
    for j in (0, 1):
        u, Rs = _socket_frame(j)
        qs = quat_from_mat(Rs)
        mouth = SOCK_L * u
        for k in range(N_SEG):
            a = 2 * math.pi * k / N_SEG
            lp = np.array([rc * math.cos(a), rc * math.sin(a), -BORE_D / 2])
            q = np.zeros(4)
            mujoco.mju_mulQuat(q, qs, np.array([math.cos(a / 2), 0, 0, math.sin(a / 2)]))
            _box(eb, f"{name}_s{j}_b{k}", (t / 2, hw, BORE_D / 2), mouth + Rs @ lp, PLASTIC, quat=q, mu=ELBOW_MU,
                 mass=ring_mass)
        _box(eb, f"{name}_s{j}_oring", (0.0095, 0.0095, OR_HALF), mouth - (OR_DEPTH + OR_HALF) * u, RUBBER, quat=qs,
             mu=0.8, mass=0.001, contype=CT_PLATE, conaffinity=CT_PLATE)
        sl = eb.add_body(name=f"{name}_s{j}_slider", pos=list(mouth), quat=list(qs))
        # the chain's mass sits at the corner (on the socket's axis), so that an elbow stands on a mouth
        sl.explicitinertial = True
        sl.mass, sl.inertia, sl.ipos = INNER_MASS, [1e-5, 1e-5, 1e-5], [0, 0, -SOCK_L]
        sl.add_joint(name=f"{name}_s{j}_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 0, -1],
                     range=[0, unit["stop"]], limited=True, frictionloss=unit["f_ins"], armature=0.01,
                     solref_limit=list(JOIN_SOLREF), solimp_limit=list(STIFF_SOLIMP))
        ro = sl.add_body(name=f"{name}_s{j}_rotor")
        ro.explicitinertial = True
        ro.mass, ro.inertia, ro.ipos = INNER_MASS, [1e-5, 1e-5, 1e-5], [0, 0, -SOCK_L]
        ro.add_joint(name=f"{name}_s{j}_turn", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                     frictionloss=unit["torque"], armature=1e-4)


def _pipe(w, wb, name, length, center, axis, kg_m, band=0):
    w.add_mesh(name=f"{name}_mesh", uservert=chamfered_cylinder(R_PIPE, length, CHAMFER))
    R = rot_between([0, 0, 1.0], axis)
    pb = wb.add_body(name=name, pos=list(center), quat=list(quat_from_mat(R)))
    pb.add_freejoint(name=f"{name}_free")
    g = pb.add_geom(name=f"{name}_g", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"{name}_mesh", rgba=list(COPPER))
    g.mass = kg_m * length
    g.friction = [PIPE_MU, 0.005, 0.0001]
    g.contype, g.conaffinity = CT_GENERAL | CT_PLATE, CT_GENERAL | CT_PLATE
    if band:
        z = band * (length / 2 - BAND_FROM_END)
        g = pb.add_geom(name=f"{name}_band", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[R_PIPE + 0.0004, BAND_HALF, 0],
                        pos=[0, 0, z], rgba=list(BAND))
        g.contype = g.conaffinity = 0
        g.mass = 0.0


def pipe_center(episode, which):
    """A pipe's centre and axis at the episode's start (lying on the bench)."""
    xy = (PIPE_A_XY if which == "A" else PIPE_B_XY) + np.array(episode["a_xy" if which == "A" else "b_xy"])
    yaw = math.pi / 2 + episode["a_yaw" if which == "A" else "b_yaw"]
    return np.array([xy[0], xy[1], TABLE_Z + R_PIPE + 0.0002]), np.array([math.cos(yaw), math.sin(yaw), 0.0])


def elbow_start(episode, e):
    """Elbow e's frame at the start: lying flat (its normal up) turned by its yaw, or standing on socket
    e_down[e]'s mouth (that socket's axis down, the corner above the mouth) with the other socket level along the
    yaw's direction."""
    xy = ELBOW_XY[e] + np.array(episode["e_xy"][e])
    yaw = episode["e_yaw"][e]
    down = episode["e_down"][e]
    if down is None:
        return np.array([xy[0], xy[1], TABLE_Z + R_OUT + 0.0003]), yaw_mat(yaw)
    h = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    R = np.column_stack([z, h, np.cross(z, h)]) if down == 0 else np.column_stack([h, z, np.cross(h, z)])
    return np.array([xy[0], xy[1], TABLE_Z + SOCK_L + 0.0003]), R


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
    _box(wb, "table", ((TABLE_EDGE_X + 0.28) / 2, TABLE_HALF_Y, TABLE_Z / 2), ((TABLE_EDGE_X - 0.28) / 2, 0.0,
         TABLE_Z / 2), (0.62, 0.58, 0.52, 1), mu=0.4)

    # the contact shapes of the drawn camera rig an arm can reach (spool_parts.contact_shapes: the rear mast and its
    # boom, the three fixed cameras, the side camera's stand); they are not drawn themselves (spool_scene.py draws
    # the parts). No part and no reference motion comes near them.
    import spool_parts
    for name, lo, hi in spool_parts.contact_shapes(CAMERAS, TABLE_Z):
        lo, hi = np.asarray(lo, float), np.asarray(hi, float)
        g = _box(wb, name, (hi - lo) / 2, (lo + hi) / 2, (0.2, 0.2, 0.2, 1), mu=0.4)
        g.group = 3

    ra = robosuite_assets()
    for side, xy in BASES.items():
        robot = mujoco.MjSpec.from_file(os.path.join(ra, "robots", "panda", "robot.xml"))
        for cam in list(robot.cameras):
            if cam.name == "robotview":
                robot.delete(cam)
        grip = mujoco.MjSpec.from_file(os.path.join(ra, "grippers", "panda_gripper.xml"))
        robot.body("right_hand").add_frame().attach_body(grip.body("right_gripper"), "g_", "")
        # the wrist camera's drawn block (spool_parts.WRIST_BOX at the model's eye_in_hand camera) as a massless
        # contact box on the hand
        cam = [c for c in robot.cameras if c.name == "eye_in_hand"][0]
        Rc = mat_from_quat(cam.quat)
        Rb = np.column_stack([-Rc[:, 2], Rc[:, 0], Rc[:, 1]])
        lo, hi = np.asarray(spool_parts.WRIST_BOX[0]), np.asarray(spool_parts.WRIST_BOX[1])
        gw = _box(robot.body("right_hand"), "wcam", (hi - lo) / 2, np.asarray(cam.pos, float) + Rb @ ((lo + hi) / 2),
                  (0.1, 0.1, 0.1, 1), quat=quat_from_mat(Rb), mu=0.4, mass=0.0)
        gw.group = 3
        wb.add_frame(pos=[xy[0], xy[1], TABLE_Z]).attach_body(robot.body("base"), f"{side}_", "")

    for which, length in (("A", episode["len_a"]), ("B", episode["len_b"])):
        c, a = pipe_center(episode, which)
        _pipe(w, wb, which, length, c, a, unit["pipe_kg_m"], band=episode["band"] if which == "A" else 0)
    for e in (0, 1):
        p, R = elbow_start(episode, e)
        _elbow(wb, f"E{e}", p, R, unit)
    welds = []
    for e in (0, 1):
        for j in (0, 1):
            for pipe in ("A", "B"):
                n = f"join_E{e}s{j}_{pipe}"
                w.add_equality(name=n, type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                               name1=f"E{e}_s{j}_rotor", name2=pipe, active=False, solref=list(JOIN_SOLREF),
                               solimp=list(STIFF_SOLIMP))
                welds.append(n)
    for name, (pos, tgt, fovy, _w, _h) in CAMERAS.items():
        wb.add_camera(name=name, pos=list(pos), fovy=fovy, quat=list(quat_from_mat(look_at(pos, tgt))))
    m = w.compile()
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
            m.geom_friction[gid, 0] = PAD_MU
            m.geom_friction[gid, 1] = PAD_TORSION
    return m


class Arm:
    """One Panda: the operational-space controller and joint step of pantry_jar_refilling's jar_sim.Arm, with the
    Franka Hand's force grasp."""

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
        jid = [m.joint(f"{p}_joint{i}").id for i in range(1, 8)]
        self.q_lo, self.q_hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))
        self.q_null = HOME_Q.copy()
        self.grip_cmd = -1.0
        self.grip_target = 0.04
        self.mode = "osc"
        self.q_goal = None

    def init_targets(self):
        self.target_pos = self.d.site_xpos[self.site].copy()
        self.target_R = self.d.site_xmat[self.site].reshape(3, 3).copy()

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
        """This control step's command from one arm's action (7, 14 or 8 numbers, already checked). Joint angles in
        an action are clipped to the joints' limits."""
        d = self.d
        if len(a) == 8:
            q = d.qpos[self.jq].copy()
            qt = np.clip(np.asarray(a[:7], float), self.q_lo, self.q_hi)
            self.q_goal = q + np.clip(qt - q, -MAX_DQ, MAX_DQ)
            self.q_null = qt.copy()
            self.mode = "joint"
            grip = a[7]
        else:
            if len(a) == 14:
                self.q_null = np.clip(np.asarray(a[7:14], float), self.q_lo, self.q_hi)
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
        self.grip_target = 0.04 - (0.04 + GRIP_PAST) * (self.grip_cmd + 1.0) / 2.0

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
        return tau


class World:
    PIPES = ("A", "B")

    def __init__(self, unit, episode):
        self.unit, self.episode = unit, episode
        self.m = m = build(unit, episode)
        self.d = d = mujoco.MjData(m)
        self.arms = {s: Arm(m, d, s) for s in BASES}
        self.len = {"A": float(episode["len_a"]), "B": float(episode["len_b"])}
        self.pipe = {p: m.body(p).id for p in self.PIPES}
        self.pipe_geom = {m.geom(f"{p}_g").id: p for p in self.PIPES}
        self.elbow = [m.body(f"E{e}").id for e in (0, 1)]
        self.sockets = [(e, j) for e in (0, 1) for j in (0, 1)]           # socket k = 2 e + j
        self.slide_q = [m.joint(f"E{e}_s{j}_slide").qposadr[0] for e, j in self.sockets]
        self.slide_v = [m.joint(f"E{e}_s{j}_slide").dofadr[0] for e, j in self.sockets]
        self.slide_j = [m.joint(f"E{e}_s{j}_slide").id for e, j in self.sockets]
        self.turn_q = [m.joint(f"E{e}_s{j}_turn").qposadr[0] for e, j in self.sockets]
        self.turn_v = [m.joint(f"E{e}_s{j}_turn").dofadr[0] for e, j in self.sockets]
        self.rotor = [m.body(f"E{e}_s{j}_rotor").id for e, j in self.sockets]
        self.plate = [m.geom(f"E{e}_s{j}_oring").id for e, j in self.sockets]
        self.plate_sock = {g: k for k, g in enumerate(self.plate)}
        self.weld = {(k, p): m.equality(f"join_E{e}s{j}_{p}").id for k, (e, j) in enumerate(self.sockets)
                     for p in self.PIPES}
        # which spool part a geom belongs to ("A", "B", 0, 1), and the geoms a spool part may touch while a joint is
        # made: the other spool parts and the two grippers (the gripper's bodies, prefixed l_g_ / r_g_)
        self.part_of = {}
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            for p in self.PIPES:
                if b == self.pipe[p]:
                    self.part_of[g] = p
            for e in (0, 1):
                if b == self.elbow[e]:
                    self.part_of[g] = e
        self.hand_geom = np.zeros(m.ngeom, bool)
        for g in range(m.ngeom):
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
            if bname.startswith(("l_g_", "r_g_")):
                self.hand_geom[g] = True
        self.engaged = [None] * 4             # socket k: (pipe, end sign) or None
        self.dirty = [False] * 4              # socket k's joint was made on a support
        self.locked = [False] * 4             # socket k's joint is home and held by its grip ring: its lock depth
        self.peak_push = [0.0] * 4            # the largest axial push on each O-ring seen (N), for the record
        self.t = 0
        mujoco.mj_resetData(m, d)
        for a in self.arms.values():
            d.qpos[a.jq] = HOME_Q
            d.qpos[a.gq[0]], d.qpos[a.gq[1]] = 0.04, -0.04
        mujoco.mj_forward(m, d)
        for _ in range(250):
            for a in self.arms.values():
                d.ctrl[a.act] = 200.0 * (HOME_Q - d.qpos[a.jq]) - 20.0 * d.qvel[a.jv] + d.qfrc_bias[a.jv]
                d.ctrl[a.gact] = [0.04, -0.04]
            mujoco.mj_step(m, d)
        for a in self.arms.values():
            a.init_targets()
        self.q_prev = {s: d.qpos[a.jq].copy() for s, a in self.arms.items()}

    # --- geometry --------------------------------------------------------------------------------
    def socket_pose(self, k):
        """Socket k's mouth (world) and outward axis (world unit vector)."""
        e, j = self.sockets[k]
        b = self.elbow[e]
        R = self.d.xmat[b].reshape(3, 3)
        u = R[:, 0] if j == 0 else R[:, 1]
        return self.d.xpos[b] + SOCK_L * u, u.copy()

    def pipe_end(self, p, sign):
        """Pipe p's end `sign` (+1: toward its local +z): the end's centre and the pipe's axis pointing out of it."""
        b = self.pipe[p]
        z = self.d.xmat[b].reshape(3, 3)[:, 2]
        return self.d.xpos[b] + sign * self.len[p] / 2 * z, sign * z

    def depth(self, k):
        return float(self.d.qpos[self.slide_q[k]])

    def home(self, k):
        return self.engaged[k] is not None and self.depth(k) >= self.unit["stop"] - HOME_TOL

    def joints(self):
        """The joints now: [(socket k, pipe, end sign, home, made on a support)]."""
        return [(k, self.engaged[k][0], self.engaged[k][1], self.home(k), self.dirty[k]) for k in range(4)
                if self.engaged[k] is not None]

    def groups(self):
        """The connected spool parts: a list of sets of "A", "B", 0, 1, joined by engaged joints."""
        parent = {x: x for x in ("A", "B", 0, 1)}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x

        for k, en in enumerate(self.engaged):
            if en is not None:
                a, b = find(en[0]), find(self.sockets[k][0])
                parent[a] = b
        out = {}
        for x in parent:
            out.setdefault(find(x), set()).add(x)
        return list(out.values())

    def group_of(self, part):
        for g in self.groups():
            if part in g:
                return g
        return {part}

    # --- the push-fit joints -----------------------------------------------------------------------
    def _touching(self):
        """The spool parts touching anything other than a gripper or another spool part now."""
        d = self.d
        n = d.ncon
        out = set()
        if not n:
            return out
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        for a, b in zip(g1, g2):
            pa, pb = self.part_of.get(int(a)), self.part_of.get(int(b))
            if pa is not None and pb is None and not self.hand_geom[b]:
                out.add(pa)
            elif pb is not None and pa is None and not self.hand_geom[a]:
                out.add(pb)
        return out

    def _engage(self, k, p, sign, s):
        """Join pipe p's end `sign` to socket k at depth s. The rotor's pose is computed here rather than by MuJoCo's
        kinematics, so the step's observation keeps the poses MuJoCo computed at the start of its last substep."""
        m, d = self.m, self.d
        e = self.weld[(k, p)]
        s = min(max(s, 0.0), self.unit["stop"])
        d.qpos[self.slide_q[k]] = s
        d.qvel[self.slide_v[k]] = 0.0
        d.qpos[self.turn_q[k]] = 0.0
        d.qvel[self.turn_v[k]] = 0.0
        eb, j = self.sockets[k]
        Re = d.xmat[self.elbow[eb]].reshape(3, 3)
        R1 = Re @ _socket_frame(j)[1]                        # the rotor's orientation at turn 0
        u = R1[:, 2]
        Rp = d.xmat[self.pipe[p]].reshape(3, 3)
        # put the pipe on the socket's axis, its spin kept: the pipe's axis out of the joined end is -u
        R2 = rot_between(sign * Rp[:, 2], -u) @ Rp
        q1c, qrel = np.zeros(4), np.zeros(4)
        mujoco.mju_negQuat(q1c, quat_from_mat(R1))
        mujoco.mju_mulQuat(qrel, q1c, quat_from_mat(R2))
        m.eq_data[e, 0:3] = [0.0, 0.0, sign * self.len[p] / 2]
        m.eq_data[e, 3:6] = 0.0
        m.eq_data[e, 6:10] = qrel
        m.eq_data[e, 10] = 1.0
        d.eq_active[e] = 1
        g = self.plate[k]
        m.geom_contype[g] = m.geom_conaffinity[g] = 0
        self.engaged[k] = (p, int(sign))
        self.dirty[k] = bool(self._touching() & self.group_of(p))      # the joined group, this joint included

    def _lock(self, k):
        """Socket k's joint is home: its grip ring holds the pipe, which cannot be drawn back any more (it still
        turns). The slide joint's lower limit is raised to where the pipe is (at most LOCK_BACK short of the stop)."""
        lo = min(self.depth(k), self.unit["stop"] - LOCK_BACK)
        self.m.jnt_range[self.slide_j[k], 0] = lo
        self.locked[k] = float(lo)

    def _disengage(self, k):
        p, _sign = self.engaged[k]
        self.d.eq_active[self.weld[(k, p)]] = 0
        g = self.plate[k]
        self.m.geom_contype[g] = self.m.geom_conaffinity[g] = CT_PLATE
        self.engaged[k] = None
        self.dirty[k] = False

    def _joints(self):
        m, d = self.m, self.d
        for k in range(4):
            if self.engaged[k] is not None and self.depth(k) < OUT_DEPTH:
                self._disengage(k)
            elif self.engaged[k] is not None and self.locked[k] is False and self.home(k):
                self._lock(k)
        n = d.ncon
        push = {}
        if n:
            g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
            f6 = np.zeros(6)
            for i in range(n):
                a, b = int(g1[i]), int(g2[i])
                if a in self.plate_sock and b in self.pipe_geom:
                    k, p, sgn = self.plate_sock[a], self.pipe_geom[b], 1.0
                elif b in self.plate_sock and a in self.pipe_geom:
                    k, p, sgn = self.plate_sock[b], self.pipe_geom[a], -1.0
                else:
                    continue
                mujoco.mj_contactForce(m, d, i, f6)
                fw = d.contact.frame[i].reshape(3, 3).T @ f6[:3]       # on geom2, from geom1
                _mouth, u = self.socket_pose(k)
                push[(k, p)] = push.get((k, p), 0.0) + sgn * float(fw @ u)   # the O-ring's push on the pipe, outward
        for (k, p), f in push.items():
            self.peak_push[k] = max(self.peak_push[k], f)
            if self.engaged[k] is not None or f <= self.unit["f_ins"]:
                continue
            mouth, u = self.socket_pose(k)
            best = None
            for sign in (1, -1):
                pe, out = self.pipe_end(p, sign)
                if any(en == (p, sign) for en in self.engaged):
                    continue
                s = float((mouth - pe) @ u)
                lat = float(np.linalg.norm((pe - mouth) - ((pe - mouth) @ u) * u))
                tilt = math.degrees(math.acos(float(np.clip(-out @ u, -1.0, 1.0))))
                if 0.0 < s < SOCK_L and lat < LAT_TOL and tilt < TILT_TOL_DEG:
                    best = (sign, s)
            if best is not None:
                self._engage(k, p, best[0], best[1])
        moving = [k for k in range(4) if self.engaged[k] is not None and d.qvel[self.slide_v[k]] > ADVANCE_V]
        if moving:
            touch = self._touching()
            if touch:
                for k in moving:
                    if self.group_of(self.engaged[k][0]) & touch:
                        self.dirty[k] = True

    # --- stepping -------------------------------------------------------------------------------
    def step(self, cmds):
        m, d = self.m, self.d
        for s, a in cmds.items():
            self.arms[s].command(a)
        for i in range(SUBSTEPS):
            if i == SUBSTEPS - 1:
                self.q_prev = {s: d.qpos[a.jq].copy() for s, a in self.arms.items()}
            for a in self.arms.values():
                d.ctrl[a.act] = a.torque()
                d.ctrl[a.gact] = [a.grip_target, -a.grip_target]
            mujoco.mj_step(m, d)
            self._joints()
        self.t += 1
        if self.unstable():
            raise FloatingPointError("unstable")

    def part_bodies(self):
        return [self.pipe["A"], self.pipe["B"]] + self.elbow

    def unstable(self):
        d = self.d
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            return True
        if any(np.max(np.abs(d.qvel[a.jv])) > MAX_ARM_SPEED for a in self.arms.values()):
            return True
        v = np.zeros(6)
        for b in self.part_bodies():
            mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            if np.linalg.norm(v[3:]) > MAX_OBJECT_SPEED:
                return True
        return False

    def part_speed(self):
        v = np.zeros(6)
        out = 0.0
        for b in self.part_bodies():
            mujoco.mj_objectVelocity(self.m, self.d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            out = max(out, float(np.linalg.norm(v[3:])))
        return out

    def hand_clearance(self, s):
        """The distance from arm s's grip site to the nearest spool part's surface (pipes as segments, elbows as
        spheres of their reach)."""
        p = self.arms[s].pos()
        best = 1e9
        for q in self.PIPES:
            c = self.d.xpos[self.pipe[q]]
            z = self.d.xmat[self.pipe[q]].reshape(3, 3)[:, 2]
            t = float(np.clip((p - c) @ z, -self.len[q] / 2, self.len[q] / 2))
            best = min(best, float(np.linalg.norm(p - c - t * z)) - R_PIPE)
        for b in self.elbow:
            best = min(best, float(np.linalg.norm(p - self.d.xpos[b])) - (SOCK_L + R_OUT))
        return best

    def on_bench(self):
        """Every spool part's centre of mass above the bench top."""
        return all(self.d.xipos[b][2] > TABLE_Z for b in self.part_bodies())

    def twist(self):
        """The spool's geometry, when its joints make the drawing's shape (A's two ends in the two elbows, B's one end
        in one of them): {"b_elbow", "banded_in_b_elbow", "twist_deg"}; otherwise None. The twist is the angle from the
        B-elbow's B-socket axis to the other elbow's free socket axis, about A's axis pointing from the B-elbow to the
        other elbow (right hand), in [0, 360)."""
        a_in = {self.sockets[k][0]: (k, en[1]) for k, en in enumerate(self.engaged) if en is not None and en[0] == "A"}
        b_in = [(k, en[1]) for k, en in enumerate(self.engaged) if en is not None and en[0] == "B"]
        if len(a_in) != 2 or len(b_in) != 1:
            return None
        kb, _sb = b_in[0]
        eb = self.sockets[kb][0]
        eo = 1 - eb
        ka_b, sa_b = a_in[eb]
        ka_o, sa_o = a_in[eo]
        free = [k for k in range(4) if self.sockets[k][0] == eo and k != ka_o]
        if len(free) != 1 or self.engaged[free[0]] is not None:
            return None
        za = self.d.xmat[self.pipe["A"]].reshape(3, 3)[:, 2]
        ax = sa_o * za                                       # from the B-elbow's end of A toward the other's
        o1 = self.socket_pose(kb)[1]
        o2 = self.socket_pose(free[0])[1]
        o1 = o1 - (o1 @ ax) * ax
        o2 = o2 - (o2 @ ax) * ax
        ang = math.degrees(math.atan2(float(np.cross(o1, o2) @ ax), float(o1 @ o2))) % 360.0
        return {"b_elbow": int(eb), "banded_in_b_elbow": bool(sa_b == self.episode["band"]), "twist_deg": ang}

    # --- the oracle's shadow: the state beyond qpos / qvel ----------------------------------------
    def state(self):
        """Everything that, with the unit and the episode, sets this world exactly: the simulation state, the joints'
        welds and bookkeeping, and the step."""
        m, d = self.m, self.d
        return {"t": int(self.t), "qpos": d.qpos.tolist(), "qvel": d.qvel.tolist(), "act": d.act.tolist(),
                "ctrl": d.ctrl.tolist(), "eq_active": [int(x) for x in d.eq_active], "eq_data": m.eq_data.tolist(),
                "engaged": [None if en is None else [en[0], int(en[1])] for en in self.engaged],
                "dirty": [bool(x) for x in self.dirty],
                "locked": [float(x) if x is not False else False for x in self.locked],
                "peak_push": [float(x) for x in self.peak_push],
                "grip_cmd": {s: float(a.grip_cmd) for s, a in self.arms.items()},
                "q_prev": {s: [float(x) for x in v] for s, v in self.q_prev.items()}}

    def set_state(self, st):
        m, d = self.m, self.d
        d.qpos[:] = np.asarray(st["qpos"], float)
        d.qvel[:] = np.asarray(st["qvel"], float)
        if d.act.size:
            d.act[:] = np.asarray(st["act"], float)
        d.ctrl[:] = np.asarray(st["ctrl"], float)
        m.eq_data[:] = np.asarray(st["eq_data"], float)
        d.eq_active[:] = np.asarray(st["eq_active"], dtype=d.eq_active.dtype)
        self.engaged = [None if en is None else (str(en[0]), int(en[1])) for en in st["engaged"]]
        self.dirty = [bool(x) for x in st["dirty"]]
        self.locked = [False if x is False else float(x) for x in st["locked"]]
        for k, jid in enumerate(self.slide_j):
            m.jnt_range[jid, 0] = 0.0 if self.locked[k] is False else self.locked[k]
        self.peak_push = [float(x) for x in st["peak_push"]]
        for k, g in enumerate(self.plate):
            on = 0 if self.engaged[k] is not None else CT_PLATE
            m.geom_contype[g] = m.geom_conaffinity[g] = on
        for s, g in st["grip_cmd"].items():
            self.arms[s].grip_cmd = float(g)
        self.q_prev = {s: np.asarray(v, float) for s, v in st["q_prev"].items()}
        self.t = int(st["t"])
        mujoco.mj_forward(m, d)
