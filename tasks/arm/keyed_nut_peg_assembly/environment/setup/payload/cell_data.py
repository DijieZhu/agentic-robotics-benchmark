"""Bayonet nut kit cell: the data sheet's numbers, the camera calibration and a few geometry helpers.

The companion of `datasheet.md`: every constant of the kit's drawing, the cell's layout, the robot's controller
interface, the sensors and the cameras, with the helpers that follow from them (camera matrices, the cells a groove
occupies, the image codec of the wire), and the TOLERANCES within which every real kit and the cell on the day
differ from these nominal values (`TOL`; data sheet, section 11). Plain Python and NumPy; it contains no simulator.
If your controller uses it, copy it into your submission folder (nothing else of your home is available when the
controller is graded).

Frames and units: world frame, metres, radians, seconds; the table top is the plane z = TABLE_Z and spans x, y in
[-TABLE_HALF, TABLE_HALF]; the Panda base stands at ROBOT_BASE facing +x. Quaternions are (w, x, y, z).
"""

import math

import numpy as np

# ----------------------------------------------------------------------------------------------
# Control and takt
# ----------------------------------------------------------------------------------------------
CONTROL_DT = 0.1                 # one action every 0.1 s (10 Hz); the arm's torque loop runs at 500 Hz
NUTS = 3                         # nuts in a kit (one per peg)
MAX_STEPS = 850                  # control steps per episode: the takt time for a kit, 85 s simulated
MAX_DPOS = 0.02
MAX_DROT = 0.15
TABLE_Z = 0.80
TABLE_HALF = 0.40
ROBOT_BASE = np.array([-0.56, 0.0, TABLE_Z])
WORKSPACE_LO = np.array([-0.45, -0.45, TABLE_Z + 0.002])
WORKSPACE_HI = np.array([0.45, 0.45, TABLE_Z + 0.55])

# ----------------------------------------------------------------------------------------------
# Kit drawing (m)
# ----------------------------------------------------------------------------------------------
NSEC = 24             # angular sectors of the nut ring (15 degrees each)
LAYER = 0.005         # layer height of the nut ring
NLAYER = 9            # nut height 45 mm
R_HOLE = 0.0128       # hole radius: the flats of the 24-sided hole from the axis (shaft 12.3 mm)
R_GROOVE = 0.019      # groove floor radius
R_NUT = 0.028         # nut outer radius
NUT_H = NLAYER * LAYER
HANDLE_LEN = 0.070
HANDLE_W = 0.008      # half width
HANDLE_Z = (0.025, 0.041)
SHAFT_R = 0.0123      # a locating fit in the 25.6 mm hole: 1 mm on the diameter (the fine series of clearance holes)
PEG_H = 0.065
COLLAR = 0.025        # collar half width
COLLAR_H = 0.010
PIN_Z = 0.0375        # pin centre above the collar top
PIN_R = 0.0172        # pin tip radius
PIN_W = 0.0015        # pin half width (tangential)
PIN_HH = 0.002        # pin half height
# groove schedule (layers): entrance channel 0-1, first turn 1-2, riser 2-4, second turn 4-5, riser 5-7
TURN_LAYERS = ((1, 2), (4, 5))
RISERS = ((0, 1), (2, 4), (5, 7))
NUT_COM = (0.015, 0.0, 0.018)    # the nut's centre of mass in its own frame (towards the handle, above the bottom face)
BAND_HALF_H = 0.0035             # a peg band: a dark ring 7 mm tall around the shaft
BAND_FIRST = 0.010               # the first band's centre below the shaft top
BAND_PITCH = 0.013               # the next bands' centres, each this much lower

# ----------------------------------------------------------------------------------------------
# Layout tolerances (uniform per episode)
# ----------------------------------------------------------------------------------------------
PEG_REGION = ((-0.02, 0.15), (-0.21, 0.21))
PEG_MIN_SPACING = 0.13
NUT_REGION = ((-0.24, -0.06), (-0.24, 0.24))
NUT_PEG_MIN_DIST = 0.13
NUT_MIN_SPACING = 0.12       # between nut centres
HANDLE_CLEAR = 0.07          # handle centre from any other nut centre
NUT_YAW_CENTER = math.radians(135.0)   # nut yaw = azimuth of the nut from the robot base + this +- spread
NUT_YAW_SPREAD = math.radians(60.0)

# ----------------------------------------------------------------------------------------------
# Safety stops
# ----------------------------------------------------------------------------------------------
ENGAGE_XY = 0.005            # a nut is on its peg: hole axis within this distance of that peg's shaft axis
GROOVE_TOP = 0.0395          # nut bottom above the collar when it rests on the pin (pin top)
FORCE_ANGLE_TOL = 5.0        # deg: pin centre this far inside a solid wall cell (sideways) breaks the nut
FORCE_Z_TOL = 0.0015         # m: ... or this far (up or down)
FORCE_STEPS = 3              # consecutive control steps
INSTAB_ROBOT_QVEL = 8.0      # rad/s: an episode with faster robot joints ends as a fault
INSTAB_NUT_VEL = 5.0         # m/s

# ----------------------------------------------------------------------------------------------
# Robot, gripper, sensors
# ----------------------------------------------------------------------------------------------
# Tolerances: what a real kit and the cell on the day differ in from the drawing. Every value is drawn uniformly
# within its range, independently, per nut, per peg and per episode (data sheet, section 11). A range (lo, hi) is
# absolute; a single number t means nominal - t .. nominal + t.
TOL = {
    # every nut
    "wall_deg": 1.0,            # each of the 24 sector boundaries of the hole wall (the channel walls)
    "level": 0.0002,            # each of the 8 layer boundaries inside the wall (the channel floors and ceilings)
    "hole_r": 0.0001,           # hole flats from the axis
    "groove_r": 0.0002,         # groove floor radius
    "handle_deg": 0.5,          # the handle's direction against the wall's sector grid
    "colour": 0.04,             # each RGB component of the nut's colour
    # every peg
    "shaft_r": 0.00005,
    "pin_r": 0.0002,            # pin tip radius
    "pin_w": 0.00005,           # pin half width
    "pin_hh": 0.00005,          # pin half height
    "pin_z": 0.0003,            # pin centre above the collar top
    # the cell on the day (every episode)
    "pad_friction": (0.6, 1.2),             # sliding friction of the finger pads on a nut (the robot model says 2)
    "joint_friction": (0.6, 1.4),           # scale of each arm joint's friction curve (JOINT_FRICTION)
    "wrench_drift": (0.3 / 60.0, 0.012 / 60.0),   # zero drift of the wrench: N per s, N m per s, per axis
    "cam_fixed_pos": 0.002,     # fixed cameras: position, per axis
    "cam_fixed_rot_deg": 0.2,   # fixed cameras: orientation, about each axis
    "cam_wrist_pos": 0.0015,    # wrist camera on the hand: position, per axis
    "cam_wrist_rot_deg": 0.3,   # wrist camera on the hand: orientation, about each axis
    "cam_focal": 0.005,         # focal length, relative, every camera
    "cam_principal_px": 1.0,    # principal point, pixels per axis, every camera
    "light": 0.15,              # brightness of the lamps, relative
}
NUT_FRICTION = (0.3, 0.7)    # sliding friction between a nut and a peg (the nut spec's `friction`)
NUT_MASS = (0.05, 0.15)      # kg
PAD_FRICTION_MODEL = 2.0     # the robot model's pad friction, which TOL["pad_friction"] replaces
PAD_TORSION_MODEL = 0.05     # ... and its torsional friction (m), scaled with the pad friction
PEG_FRICTION = 0.1           # the pegs' own coefficient: below every nut's, so that a nut-peg pair has the nut's
# The arm joints' friction torque (N m) at joint speed w (rad/s): phi1 / (1 + exp(-phi2 (w + phi3))) -
# phi1 / (1 + exp(-phi2 phi3)), the curve identified on the Franka Emika Panda (Gaz et al., IEEE RA-L 2019,
# supplementary Table I). The cell's tracking law does not compensate it.
JOINT_FRICTION = ((0.54615, 0.87224, 0.64068, 1.2794, 0.83904, 0.30301, 0.56489),
                  (5.1181, 9.0657, 10.136, 5.5903, 8.3469, 17.133, 10.336),
                  (0.039533, 0.025882, -0.04607, 0.036194, 0.026226, -0.021047, 0.0035526))

GRIP_KP = 3000.0             # finger position gain (N/m); each finger's force is capped at 20 N
RGB_NOISE = 2.0              # 8-bit sigma
WRENCH_NOISE = 0.3           # N (torque: 0.03 N m)
WRENCH_BIAS = 0.5            # N per axis, fixed per episode (torque: 0.05 N m)
HOME_Q = np.array([0.0, -0.35, 0.0, -2.35, 0.0, 2.0, math.pi / 4])
NULLSPACE_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, math.pi / 4])
OSC_KP_POS = 400.0           # 1/s^2: task-space position gain (natural frequency 20 rad/s, critically damped)
OSC_KP_ROT = 200.0           # 1/s^2: task-space orientation gain (14.1 rad/s, critically damped)

# ----------------------------------------------------------------------------------------------
# Cameras
# ----------------------------------------------------------------------------------------------
CAMERAS = {
    # name: (position, look-at target, fovy deg, width, height); fixed cameras never move
    "left": ((-0.30, 0.55, 1.45), (0.05, 0.0, 0.80), 50.0, 320, 240),
    "right": ((-0.30, -0.55, 1.45), (0.05, 0.0, 0.80), 50.0, 320, 240),
}
WRIST = {"fovy": 75.0, "width": 256, "height": 192}
CAMERA_NAMES = ("left", "right", "wrist")
OBS_KEYS = ("t", "order", "q", "dq", "ee_pos", "ee_quat", "gripper", "wrench", "T_world_wrist",
            "left", "right", "wrist")

NUT_COLOURS = {"blue": [0.16, 0.36, 0.78, 1], "green": [0.20, 0.62, 0.30, 1], "red": [0.78, 0.18, 0.16, 1],
               "yellow": [0.86, 0.74, 0.16, 1], "orange": [0.85, 0.45, 0.12, 1], "purple": [0.55, 0.25, 0.70, 1],
               "teal": [0.15, 0.65, 0.68, 1], "pink": [0.80, 0.35, 0.55, 1]}
PEG_COLOURS = {"collar": [0.30, 0.31, 0.34, 1], "shaft_and_pin": [0.62, 0.64, 0.67, 1], "band": [0.07, 0.07, 0.09, 1]}


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def look_at(pos, target, up=(0, 0, 1)):
    """Rotation whose columns are the camera axes x right, y up, z backwards (the renderer's convention)."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f = f / np.linalg.norm(f)
    z = -f
    x = np.cross(np.array(up, dtype=float), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def camera_intrinsics(fovy, width, height):
    """Pinhole K for a vertical field of view `fovy` (degrees); the principal point is at (W/2, H/2), so the
    centre of pixel (u, v) is at (u + 0.5, v + 0.5)."""
    f = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def fixed_camera_matrices(name):
    """(K, T_world_cam) of a fixed camera ("left" or "right"), OpenCV convention (x right, y down, z forward)."""
    pos, tgt, fovy, w, h = CAMERAS[name]
    R = look_at(pos, tgt) @ np.diag([1.0, -1.0, -1.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return camera_intrinsics(fovy, w, h), T


def wrist_camera_intrinsics():
    """K of the wrist camera; its pose is every observation's `T_world_wrist` (OpenCV convention)."""
    return camera_intrinsics(WRIST["fovy"], WRIST["width"], WRIST["height"])


def decode_image(d):
    """A wire image {"dtype": "uint8", "shape": [H, W, 3], "b64": ...} as an array (H, W, 3), RGB, row-major."""
    import base64
    return np.frombuffer(base64.b64decode(d["b64"]), np.uint8).reshape(d["shape"])


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def quat_to_mat(q):
    """Rotation matrix of a quaternion (w, x, y, z)."""
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def groove_cells(groove):
    """The (sector, layer) cells cut out of the ring's inner part, for a groove {"entrance": 0..23, "turns":
    [k1, k2], "dead_ends": [n1, n2], "false_entrances": [sectors]} of the family (data sheet, section 3).
    Channels are 2 sectors wide; sector s spans [15 s, 15 (s + 1)] degrees counter-clockwise from the handle
    (nut +x); layer 0 is the bottom one."""
    cells = set()

    def chan(sec, layers):
        for lay in layers:
            cells.update({(sec % NSEC, lay), ((sec + 1) % NSEC, lay)})

    s = int(groove["entrance"])
    for i, (lo, hi) in enumerate(RISERS):
        chan(s, range(lo, hi + 1))
        if i < 2:
            k = int(groove["turns"][i])
            step = 1 if k > 0 else -1
            for j in range(0, k + step, step):
                chan(s + j, TURN_LAYERS[i])
            for j in range(1, int(groove.get("dead_ends", [0, 0])[i]) + 1):    # dead end, opposite way
                chan(s - step * j, TURN_LAYERS[i])
            s = s + k
    for f in groove.get("false_entrances", []):                                # blind risers
        chan(int(f), range(0, TURN_LAYERS[0][1] + 1))
    return cells


def wall_boundaries(u=None):
    """Angles (degrees, counter-clockwise from the sector grid's zero) of the NSEC sector boundaries of a nut's hole
    wall: nominally 15 s; a real nut's are each off by its `wall` tolerance. Sector s lies between boundary s and
    boundary s + 1 (boundary NSEC is boundary 0 plus 360)."""
    d = 360.0 / NSEC
    off = u["wall"] if u else [0.0] * NSEC
    return [d * k + float(off[k]) for k in range(NSEC)]


def level_boundaries(u=None):
    """Heights (m above the nut's bottom face) of the NLAYER + 1 layer boundaries of a nut's hole wall: nominally
    LAYER * l; a real nut's NLAYER - 1 inner ones are each off by its `level` tolerance."""
    off = u["level"] if u else [0.0] * (NLAYER - 1)
    return [0.0] + [LAYER * (k + 1) + float(off[k]) for k in range(NLAYER - 1)] + [NUT_H]


def sample_unit(rng, nuts=NUTS):
    """The actual values of one episode's kit and cell within the tolerances TOL: per nut, per peg (peg i has i + 1
    bands) and for the cell on the day. Every value is uniform within its range and independent of the others.
    Angles in `wall`, `handle` are degrees; camera `rot` is a rotation vector in radians. JSON-ready."""
    def sym(t, n=None):
        v = rng.uniform(-t, t, n)
        return float(v) if n is None else [float(x) for x in v]

    unit = {"nuts": [], "pegs": [], "cameras": {}}
    for _ in range(nuts):
        unit["nuts"].append({"wall": sym(TOL["wall_deg"], NSEC), "level": sym(TOL["level"], NLAYER - 1),
                             "hole_r": sym(TOL["hole_r"]), "groove_r": sym(TOL["groove_r"]),
                             "handle": sym(TOL["handle_deg"]), "rgb": sym(TOL["colour"], 3)})
    for _ in range(3):
        unit["pegs"].append({"shaft_r": sym(TOL["shaft_r"]), "pin_r": sym(TOL["pin_r"]), "pin_w": sym(TOL["pin_w"]),
                             "pin_hh": sym(TOL["pin_hh"]), "pin_z": sym(TOL["pin_z"])})
    unit["pad_friction"] = float(rng.uniform(*TOL["pad_friction"]))
    unit["joint_friction"] = [float(x) for x in rng.uniform(TOL["joint_friction"][0], TOL["joint_friction"][1], 7)]
    unit["wrench_drift"] = sym(TOL["wrench_drift"][0], 3) + sym(TOL["wrench_drift"][1], 3)
    for name in CAMERA_NAMES:
        fixed = name in CAMERAS
        unit["cameras"][name] = {
            "pos": sym(TOL["cam_fixed_pos"] if fixed else TOL["cam_wrist_pos"], 3),
            "rot": sym(math.radians(TOL["cam_fixed_rot_deg"] if fixed else TOL["cam_wrist_rot_deg"]), 3),
            "focal": sym(TOL["cam_focal"]), "principal": sym(TOL["cam_principal_px"], 2)}
    unit["light"] = sym(TOL["light"])
    return unit


def joint_friction_torque(w, scale=None):
    """The friction torque (N m) at the seven arm joints for joint speeds `w` (rad/s); it opposes the motion.
    `scale`: each joint's factor within TOL["joint_friction"] (default 1)."""
    phi1, phi2, phi3 = (np.asarray(v, float) for v in JOINT_FRICTION)
    if scale is not None:
        phi1 = phi1 * np.asarray(scale, float)
    w = np.asarray(w, float)
    return phi1 / (1.0 + np.exp(-phi2 * (w + phi3))) - phi1 / (1.0 + np.exp(-phi2 * phi3))


def band_heights(bands):
    """Heights above the table of the centres of a peg's `bands` dark bands (1, 2 or 3)."""
    top = COLLAR_H + PEG_H
    return [top - BAND_FIRST - BAND_PITCH * k for k in range(bands)]
