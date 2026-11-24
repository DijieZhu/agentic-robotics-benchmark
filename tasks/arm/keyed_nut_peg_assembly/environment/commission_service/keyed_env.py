"""Bayonet nut kit: the task's robot cell (scene, robot, controller, sensors, safety stops).

Where this file runs: in the grader (tests/keyed_env.py) and, as a byte copy, in the commissioning service of
the agent's sandbox (environment/commission_service/keyed_env.py, readable by the service's account only).
The agent never reads it. What the agent has is the cell's data sheet (datasheet.md and cell_data.py in its
home: the kit's drawing data and tolerances, the robot and its controller, the sensors, the cameras, the
safety stops, the wire) and a window of commissioning episodes on the service's development kits. What
differs between the two places is only the KIT that is loaded: three nut specs (see `NUT_SPEC_KEYS`) and the
work order. Every graded episode loads its own kit, drawn with `sample_kit` from a seed; the service loads its
development kits, none of them a graded kit or a candidate of the graded set's selection. This file has no
ground-truth accessor and judges no assembly: the grader judges every nut by the rule the task instruction
states, with its own code (keyed_credit.py), on the state this simulation reaches.

Frames and units
----------------
World frame: the table top is the plane z = 0.80 and spans x, y in [-0.40, 0.40]. The Panda base
stands at (-0.56, 0, 0.80) facing +x. Metres, radians, seconds. Quaternions are (w, x, y, z).

Main entry points
-----------------
    kit = sample_kit(numpy.random.default_rng(seed))    # or load_kit(path)
    env = NutEnv(kit["nuts"])               # three nut specs
    obs = env.reset(seed, kit["order"])     # any int seed: layout + sensor noise; order[i] = bands of nut i's peg
    obs, done, info = env.step(action)      # 10 Hz; info = {"t", "fault", "ruined"} (see Safety stops)

Action (a list of 8 numbers):
    [dx, dy, dz, drx, dry, drz, grip, cams]
    dx..dz:   target displacement of the gripper site in the world frame (m; the vector is clipped
              to length MAX_DPOS = 0.02);
    drx..drz: a world-frame rotation vector (rad; clipped to length MAX_DROT = 0.15) applied to the
              current gripper orientation; an operational-space controller tracks the target;
    grip:     -1 = fully open (80 mm) ... +1 = fully closed; linear in between. The two fingers are one
              mechanism (as on the real hand): they move symmetrically about the hand's axis, each driven
              with at most 20 N, so a part is centred between them when it is gripped and a held part
              cannot push them aside together;
    cams:     integer bit mask of the cameras whose RGB image the NEXT observation carries
              (1 = left, 2 = right, 4 = wrist; 0 = none). Images cost wall time (and, when graded,
              controller budget: see the task instruction).

Scene
-----
Three pegs stand on the table. Each peg: a 50 x 50 x 10 mm collar (base), on it a round shaft of
radius 12.3 mm, 65 mm tall, with a square end (no lead-in chamfer), carrying one pin: a small block
on the shaft surface (3 mm wide, 4 mm tall, reaching 17.2 mm from the shaft axis) centred 37.5 mm
above the collar top, on the peg's local +x side. The pegs look alike except for 1, 2 or 3 dark bands near the shaft top. Every
episode draws the peg positions and yaws anew.

The nut: a ring 45 mm tall (outer radius 28 mm, a 24-sided polygon) whose hole (inscribed radius
12.8 mm, square edges) fits over the shaft with a locating fit (1 mm on the diameter: the fine series
of clearance holes; 0.5 mm of radial clearance on the drawing), with a handle bar (16 mm wide, 16 mm tall, from 25 to 41 mm above the
nut's bottom face) reaching 98 mm from the hole axis along the nut's local +x axis. The ring is
built from 24 sectors (15 degrees each; sector s spans [15 s, 15 (s + 1)] degrees counter-clockwise
from the handle) x 9 layers (5 mm each, layer 0 at the bottom). In each cell the inner part of the
wall (radius 12.8 to 19 mm) is either solid or cut away; the cut cells form a GROOVE in the hole
wall, through which the pin must travel for the nut to come down. The top layer is always solid,
so the outside of a nut does not show its groove, but the groove is real geometry and can be seen
where the hole is open to view: looking down into the hole at a steep angle shows the far side of
the hole wall below the top layer, where every cut cell is a recess about 6 mm deep (a step at its
bottom edge), and the entrances are openings in the bottom face. The groove family (identical for
every nut):

  * channels are 2 sectors (30 degrees) wide;
  * an ENTRANCE riser: layers 0-1 at sectors (e, e + 1);
  * a first TURN: layers 1-2, running k1 sectors from sector e (k1 = 2..4 sectors either way:
    counter-clockwise for k1 > 0), then a riser in layers 2-4 at the new sector;
  * a second TURN: layers 4-5, k2 sectors (2..4, either way), then a riser in layers 5-7;
  * optional DEAD ENDS: a turn channel can also continue 1-3 sectors the other way and stop;
  * optional FALSE ENTRANCES (0 to 2): risers in layers 0-2 at other sectors that lead nowhere.

The pin enters the groove from below. With the nut resting on the pin, the nut's bottom is 39.5
mm above the collar. The nut comes down to 24.5 mm above the collar once the pin is in the
entrance riser (or a false entrance), to 9.5 mm once it has passed the first turn and riser, and
to the collar once it has passed the second. Passing a turn means rotating the nut about the shaft
while the pin is in the turn channel; the nut turns by -15 degrees per sector the pin advances
counter-clockwise in the nut frame. Which entrance, which turns and which decoys a nut has differs
between nuts (the nut spec's `groove`). A kit has NUTS = 3 nuts. Every episode places them flat on
the table at fresh positions (NUT_REGION, at least NUT_MIN_SPACING apart, handles clear of the other
nuts) and yaws: each yaw is the nut's azimuth seen from the robot base (atan2(y - 0, x + 0.56)) plus
135 degrees plus a uniform offset within +-60 degrees. The work order assigns each nut a different
peg; nut i goes on the peg with order[i] bands (`work_order` names the nuts by colour, `colour_name`).

Nut spec (a dict): {"id": str, "groove": {"entrance": 0..23, "turns": [k1, k2], "dead_ends":
[n1, n2] (sectors, 0 = none), "false_entrances": [sectors]}, "mass": kg (0.05-0.15), "friction":
sliding friction of the nut surfaces (0.6-1.0), "rgba": [r, g, b, 1]}.

Safety stops (they end the episode, in the commissioning service as in the grader)
----------------------------------------------------------------------------------
  * an unstable simulation: any arm joint faster than INSTAB_ROBOT_QVEL = 8 rad/s, a nut faster than
    INSTAB_NUT_VEL = 5 m/s, or a non-finite state (`info["fault"]` names it);
  * a ruined nut: while a nut is on its own peg (its hole axis within ENGAGE_XY = 5 mm of that peg's
    shaft axis, below the shaft top), the centre of the pin lies inside a solid cell of the nut's
    wall, more than FORCE_ANGLE_TOL = 5 degrees sideways or FORCE_Z_TOL = 1.5 mm up or down from
    every cut cell, for FORCE_STEPS = 3 consecutive steps (`pin_in_wall`); the groove walls are solid,
    so this only happens when the pin is forced through one. `info["ruined"]` lists the ruined nuts.
    It is judged on the real nut and the real pin (below).
With `early_end=False` (a development option) a ruined nut is recorded but the episode goes on.

Tolerances: the real kit and the cell on the day
------------------------------------------------
Everything above is the DRAWING. Every episode (`reset`) is a real kit on the cell as it is that day:
`sample_unit` draws, from the episode's seed, each nut's own sector and layer boundaries, hole and groove
radii, handle direction and colour shade, each peg's own shaft and pin, the finger pads' friction, each arm
joint's friction (JOINT_FRICTION, which the tracking law does not compensate), the wrench zero's drift, each
camera's real mount and intrinsics (the published calibration and `T_world_wrist` stay the drawing's) and the
lamps' brightness, each uniformly within its range in TOL. The observations never carry these values; the data
sheet states the ranges. `NutEnv(..., tolerances=False)` builds the drawing's kit on an ideal cell (a
maintainer's check).

The cameras' images
-------------------
`NutEnv.render` draws through keyed_render.py (beside this file): MuJoCo 3.16's Filament renderer on the CPU. It
draws this model's robot (its visual meshes), pegs and nuts in the simulator's state through this model's cameras
(their real mounts and intrinsics of the day), in the laboratory the cell stands in (a render bundle of meshes,
textures and an image light, read from KEYED_RENDER_DIR; nothing of it is simulated), under the gantry's lamps at
the brightness of the day (`lamp_gain`). The scenery of SCENE_XML (floor, wall, table block, pedestal block, lights)
is the simulation's and is not drawn. The sensor noise is added here, from the episode's generator.
"""

import bisect
import json
import math
import os

import numpy as np

import mujoco

try:                                # the cameras' renderer; an environment built with render=False never needs it
    import keyed_render as _KR
    RenderError = _KR.RenderError
    _KR_ERROR = None
except ImportError as _exc:
    _KR, _KR_ERROR = None, "keyed_render.py cannot be imported beside keyed_env.py: {}".format(_exc)

    class RenderError(RuntimeError):
        """The cameras' renderer failed: an error of the environment, not of a controller."""

# ----------------------------------------------------------------------------------------------
# Public constants
# ----------------------------------------------------------------------------------------------
TIMESTEP = 0.002
SUBSTEPS = 50                    # -> 10 Hz control
CONTROL_DT = TIMESTEP * SUBSTEPS
NUTS = 3                         # nuts in a kit (one per peg)
MAX_STEPS = 850                  # control steps per episode: the takt time for a kit, 85 s simulated
MAX_DPOS = 0.02
MAX_DROT = 0.15
TABLE_Z = 0.80
TABLE_HALF = 0.40
ROBOT_BASE = np.array([-0.56, 0.0, TABLE_Z])
WORKSPACE_LO = np.array([-0.45, -0.45, TABLE_Z + 0.002])
WORKSPACE_HI = np.array([0.45, 0.45, TABLE_Z + 0.55])

# geometry (m)
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

# layout ranges (uniform per episode)
PEG_REGION = ((-0.02, 0.15), (-0.21, 0.21))
PEG_MIN_SPACING = 0.13
NUT_REGION = ((-0.24, -0.06), (-0.24, 0.24))
NUT_PEG_MIN_DIST = 0.13
NUT_MIN_SPACING = 0.12       # between nut centres
HANDLE_CLEAR = 0.07          # handle centre from any other nut centre
NUT_YAW_CENTER = math.radians(135.0)   # nut yaw = azimuth of the nut from the robot base + this +- spread
NUT_YAW_SPREAD = math.radians(60.0)

# safety stops
ENGAGE_XY = 0.005            # a nut is on its peg: hole axis within this distance of that peg's shaft axis
GROOVE_TOP = 0.0395          # nut bottom above the collar when it rests on the pin (pin top)
FORCE_ANGLE_TOL = 5.0        # deg: pin centre this far inside a solid wall cell (sideways) breaks the nut
FORCE_Z_TOL = 0.0015         # m: ... or this far (up or down)
FORCE_STEPS = 3              # consecutive control steps

# Tolerances: what a real kit and the cell on the day differ in from the drawing. Every value is drawn uniformly
# within its range, independently, per nut, per peg and per episode (`sample_unit`); where each number comes from:
# tests/review/tolerances.md. A range (lo, hi) is absolute; a single number t means nominal - t .. nominal + t.
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
FINGER_COUPLING = (0.005, 1.0)   # solref of the constraint that makes the two fingers one mechanism
RGB_NOISE = 2.0              # 8-bit sigma
WRENCH_NOISE = 0.3           # N (torque: 0.03 N m)
WRENCH_BIAS = 0.5            # N per axis, fixed per episode (torque: 0.05 N m)
INSTAB_ROBOT_QVEL = 8.0      # rad/s: an episode with faster robot joints ends as a fault
INSTAB_NUT_VEL = 5.0         # m/s
HOME_Q = np.array([0.0, -0.35, 0.0, -2.35, 0.0, 2.0, math.pi / 4])
NULLSPACE_Q = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 1.9, math.pi / 4])

CAMERAS = {
    # name: (position, look-at target, fovy deg, width, height); fixed cameras never move
    "left": ((-0.30, 0.55, 1.45), (0.05, 0.0, 0.80), 50.0, 320, 240),
    "right": ((-0.30, -0.55, 1.45), (0.05, 0.0, 0.80), 50.0, 320, 240),
}
WRIST = {"fovy": 75.0, "width": 256, "height": 192}
CAMERA_NAMES = ("left", "right", "wrist")
NUT_SPEC_KEYS = ("id", "groove", "mass", "friction", "rgba")
OBS_KEYS = ("t", "order", "q", "dq", "ee_pos", "ee_quat", "gripper", "wrench", "T_world_wrist",
            "left", "right", "wrist")


class EnvError(Exception):
    pass


class ActionError(ValueError):
    pass


# ----------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------
def look_at(pos, target, up=(0, 0, 1)):
    """Rotation whose columns are the MuJoCo camera axes (x right, y up, z backwards)."""
    pos, target = np.asarray(pos, float), np.asarray(target, float)
    f = target - pos
    f = f / np.linalg.norm(f)
    z = -f
    x = np.cross(np.array(up, dtype=float), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def quat_from_mat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def mat_from_quat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


def axis_angle_mat(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def wrap_angle(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def robosuite_assets():
    """robosuite's model assets (Panda + gripper) installed in the image; robosuite is not imported."""
    import importlib.util
    spec = importlib.util.find_spec("robosuite")
    if spec is None or not spec.submodule_search_locations:
        raise EnvError("robosuite assets not installed")
    return os.path.join(list(spec.submodule_search_locations)[0], "models", "assets")


def camera_intrinsics(fovy, width, height):
    f = (height / 2.0) / math.tan(math.radians(fovy) / 2.0)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1.0]])


def fixed_camera_matrices(name):
    """(K, T_world_cam) of a fixed camera, OpenCV convention (x right, y down, z forward)."""
    pos, tgt, fovy, w, h = CAMERAS[name]
    R = look_at(pos, tgt) @ np.diag([1.0, -1.0, -1.0])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = pos
    return camera_intrinsics(fovy, w, h), T


def parse_action(a):
    """Validate an action against the public format; returns (move[6], grip, render tuple)."""
    if not isinstance(a, (list, tuple)) or len(a) != 8:
        raise ActionError("action must be a list of 8 numbers")
    out = []
    for v in a:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or abs(v) > 1e3:
            raise ActionError("action must be a list of 8 numbers")
        out.append(float(v))
    cams = out[7]
    if cams != int(cams) or not 0 <= cams <= 7:
        raise ActionError("action must be a list of 8 numbers")
    cams = int(cams)
    render = tuple(n for i, n in enumerate(CAMERA_NAMES) if cams & (1 << i))
    return np.array(out[:6]), float(np.clip(out[6], -1.0, 1.0)), render


def encode_image(img):
    """The JSON form of an image: raw RGB bytes, row-major."""
    import base64
    img = np.ascontiguousarray(img, dtype=np.uint8)
    return {"dtype": "uint8", "shape": list(img.shape), "b64": base64.b64encode(img.tobytes()).decode("ascii")}


def decode_image(d):
    import base64
    return np.frombuffer(base64.b64decode(d["b64"]), np.uint8).reshape(d["shape"])


def to_wire(obs):
    """Observation with numpy arrays -> JSON-ready dict (images via `encode_image`)."""
    out = {}
    for k, v in obs.items():
        if k in CAMERA_NAMES:
            out[k] = encode_image(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out


def check_nut_spec(spec):
    if not isinstance(spec, dict) or any(k not in spec for k in NUT_SPEC_KEYS):
        raise EnvError("nut spec needs keys " + ", ".join(NUT_SPEC_KEYS))
    g = spec["groove"]
    if (not isinstance(g, dict) or int(g["entrance"]) not in range(NSEC) or len(g["turns"]) != 2
            or any(not 1 <= abs(int(k)) <= 6 for k in g["turns"])):
        raise EnvError("groove needs entrance 0-23 and two turns of 1-6 sectors")
    return spec


def groove_cells(groove):
    """The (sector, layer) cells cut out of the ring's inner part. Channels are 2 sectors wide;
    sector s spans [15 s, 15 (s + 1)] degrees counter-clockwise from the handle (nut +x)."""
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


# ----------------------------------------------------------------------------------------------
# Model assembly
# ----------------------------------------------------------------------------------------------
def _box(x0, x1, y0, y1, z0, z1, attrs):
    return '<geom type="box" pos="{:.6f} {:.6f} {:.6f}" size="{:.6f} {:.6f} {:.6f}" {}/>'.format(
        (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2, attrs)


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


def nut_body_xml(spec, u=None):
    """The nut as an MJCF body (origin: hole axis on the bottom face; handle along local +x). The ring is
    NSEC x NLAYER cells, each an inner block (R_HOLE..R_GROOVE) and an outer block (R_GROOVE..R_NUT);
    groove cells have no inner block. `u` (a nut of `sample_unit`) makes it a real nut: the inner blocks stand on
    the nut's own sector and layer boundaries, the hole and the groove floor have their own radii, the handle its
    own direction, the colour its own shade. Without `u` it is the drawing's nut."""
    cells = groove_cells(spec["groove"])
    rgb = [float(c) for c in spec["rgba"][:3]]
    if u:
        rgb = [min(1.0, max(0.0, c + float(dc))) for c, dc in zip(rgb, u["rgb"])]
    rgba = " ".join("{:.3f}".format(c) for c in rgb + [1.0])
    attrs = 'rgba="{}" condim="4" solref="0.002 1" friction="{:.4f} 0.01 0.0001"'.format(rgba, float(spec["friction"]))
    da = 2 * math.pi / NSEC
    r_hole = R_HOLE + (float(u["hole_r"]) if u else 0.0)
    r_groove = R_GROOVE + (float(u["groove_r"]) if u else 0.0)
    wb = wall_boundaries(u)
    wb.append(wb[0] + 360.0)
    zb = level_boundaries(u)
    box = '<geom type="box" pos="{:.6f} {:.6f} {:.6f}" quat="{:.6f} 0 0 {:.6f}" size="{:.6f} {:.6f} {:.6f}" {}/>'
    g = []
    for sec in range(NSEC):
        a = (sec + 0.5) * da                                   # the outer block stands on the drawing's sector
        ai = math.radians((wb[sec] + wb[sec + 1]) / 2.0) if u else a   # the inner block on the nut's own
        wi = math.radians(wb[sec + 1] - wb[sec]) if u else da
        for lay in range(NLAYER):
            for (r0, r1, inner) in ((r_hole, r_groove, True), (r_groove, R_NUT, False)):
                if inner and (sec, lay) in cells:
                    continue
                rc = (r0 + r1) / 2
                if inner:
                    ang, ht = ai, r1 * math.tan(wi / 2) + 0.0002
                    z0, z1 = (zb[lay], zb[lay + 1]) if u else (lay * LAYER, (lay + 1) * LAYER)
                else:
                    ang, ht = a, r1 * math.tan(da / 2) + 0.0002
                    z0, z1 = lay * LAYER, (lay + 1) * LAYER
                g.append(box.format(rc * math.cos(ang), rc * math.sin(ang), (z0 + z1) / 2, math.cos(ang / 2),
                                    math.sin(ang / 2), (r1 - r0) / 2, ht, (z1 - z0) / 2, attrs))
    hd = math.radians(float(u["handle"])) if u else 0.0       # the handle's direction against the sector grid
    hx = (R_NUT - 0.002 + R_NUT + HANDLE_LEN) / 2
    g.append(box.format(hx * math.cos(hd), hx * math.sin(hd), (HANDLE_Z[0] + HANDLE_Z[1]) / 2, math.cos(hd / 2),
                        math.sin(hd / 2), (HANDLE_LEN + 0.002) / 2, HANDLE_W, (HANDLE_Z[1] - HANDLE_Z[0]) / 2, attrs))
    m = float(spec["mass"])
    return ('<body name="nut"><freejoint name="nut_free"/>'
            '<inertial pos="0.015 0 0.018" mass="{m:.5f}" diaginertia="{a:.3e} {b:.3e} {c:.3e}"/>{g}</body>').format(
        m=m, a=m * 6.0e-4, b=m * 9.0e-4, c=m * 1.1e-3, g="".join(g))


def peg_body_xml(i, bands, u=None):
    """Peg i (i + 1 bands). `u` (a peg of `sample_unit`) makes it a real peg: its own shaft radius and pin, and the
    pegs' own friction coefficient (below every nut's, so that a nut on a peg slides with the nut's coefficient)."""
    shaft_r = SHAFT_R + (float(u["shaft_r"]) if u else 0.0)
    pin_r = PIN_R + (float(u["pin_r"]) if u else 0.0)
    pin_w = PIN_W + (float(u["pin_w"]) if u else 0.0)
    pin_hh = PIN_HH + (float(u["pin_hh"]) if u else 0.0)
    pin_z = PIN_Z + (float(u["pin_z"]) if u else 0.0)
    fr = ' friction="{:.3f} 0.005 0.0001"'.format(PEG_FRICTION) if u else ""
    shaft = 'rgba="0.62 0.64 0.67 1"' + fr
    g = _box(-COLLAR, COLLAR, -COLLAR, COLLAR, 0, COLLAR_H, 'rgba="0.30 0.31 0.34 1"' + fr)
    g += '<geom type="cylinder" pos="0 0 {:.4f}" size="{:.5f} {:.4f}" {}/>'.format(
        COLLAR_H + PEG_H / 2, shaft_r, PEG_H / 2, shaft)
    g += _box(shaft_r - 0.0005, pin_r, -pin_w, pin_w, COLLAR_H + pin_z - pin_hh, COLLAR_H + pin_z + pin_hh, shaft)
    top = COLLAR_H + PEG_H
    for k in range(bands):
        z = top - 0.010 - 0.013 * k
        g += ('<geom type="cylinder" pos="0 0 {:.4f}" size="{:.5f} 0.0035" rgba="0.07 0.07 0.09 1" contype="0" '
              'conaffinity="0" group="1" mass="0"/>').format(z, shaft_r + 0.0004)
    return '<body name="peg{}">{}</body>'.format(i, g)


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


def apply_cell_state(m, unit):
    """Put the cell's state of the day (`sample_unit`) into a compiled model: the pads' friction, the cameras' real
    poses and intrinsics (the published calibration stays the drawing's). The lamps' brightness is not in the model:
    NutEnv.lamp_gain carries it to the cameras' renderer, which multiplies its lamps and its image light by it."""
    scale = float(unit["pad_friction"]) / PAD_FRICTION_MODEL
    for name in ("r_g_finger1_pad_collision", "r_g_finger2_pad_collision"):
        gid = m.geom(name).id
        m.geom_friction[gid, 0] = PAD_FRICTION_MODEL * scale
        m.geom_friction[gid, 1] = PAD_TORSION_MODEL * scale
    pix = 1e-5                                               # an arbitrary pixel size (m) for MuJoCo's intrinsics
    for name in CAMERA_NAMES:
        err = unit["cameras"][name]
        if name in CAMERAS:
            cid, (_p, _t, fovy, w, h) = m.camera(name).id, CAMERAS[name]
        else:
            cid, fovy, w, h = m.camera("r_eye_in_hand").id, WRIST["fovy"], WRIST["width"], WRIST["height"]
        m.cam_pos[cid] = m.cam_pos[cid] + np.asarray(err["pos"], float)       # in the parent's frame
        R = axis_angle_mat(err["rot"]) @ mat_from_quat(m.cam_quat[cid])
        m.cam_quat[cid] = quat_from_mat(R)
        f = (h / 2.0) / math.tan(math.radians(fovy) / 2.0) * (1.0 + float(err["focal"]))
        m.cam_resolution[cid] = [w, h]
        m.cam_sensorsize[cid] = [w * pix, h * pix]
        # MuJoCo's principal offset is opposite to the image's: the real principal point is at
        # (w / 2 + principal[0], h / 2 + principal[1]) in pixels (u right, v down)
        m.cam_intrinsic[cid] = [f * pix, f * pix, -float(err["principal"][0]) * pix, -float(err["principal"][1]) * pix]


SCENE_XML = """<mujoco model="keyed_nut">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{dt}" integrator="implicitfast" cone="elliptic" impratio="10"/>
  <visual><global offwidth="320" offheight="240"/><quality shadowsize="2048"/>
    <headlight ambient="0.25 0.25 0.25" diffuse="0.45 0.45 0.45"/></visual>
  <asset>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.42 0.40 0.38" rgb2="0.36 0.34 0.33" width="256" height="256"/>
    <material name="floormat" texture="floor" texrepeat="6 6"/>
    <texture name="wood" type="2d" builtin="flat" mark="random" random="0.08" rgb1="0.58 0.46 0.33" rgb2="0.5 0.4 0.3" width="512" height="512"/>
    <material name="tablemat" texture="wood" texrepeat="3 3" specular="0.1"/>
    <texture name="wall" type="2d" builtin="gradient" rgb1="0.82 0.84 0.86" rgb2="0.65 0.67 0.70" width="128" height="128"/>
    <material name="wallmat" texture="wall"/>
  </asset>
  <default><geom solref="0.004 1" solimp="0.95 0.99 0.001"/></default>
  <worldbody>
    <light pos="0.2 0.5 2.4" dir="-0.1 -0.2 -1" diffuse="0.55 0.55 0.52" castshadow="true"/>
    <light pos="-0.6 -0.6 2.2" dir="0.3 0.3 -1" diffuse="0.25 0.25 0.27" castshadow="false"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floormat"/>
    <geom name="wall_back" type="box" pos="1.3 0 1.2" size="0.02 2.5 1.2" material="wallmat" contype="0" conaffinity="0"/>
    <geom name="table" type="box" pos="0 0 {tz2}" size="{th} {th} {tz2}" material="tablemat" friction="0.7 0.005 0.0001"/>
    <geom name="pedestal" type="box" pos="{rx} 0 {tz2}" size="0.12 0.12 {tz2}" rgba="0.25 0.25 0.27 1" contype="0" conaffinity="0" group="1"/>
  </worldbody>
</mujoco>"""


def build_model(specs, layout, unit=None):
    """Compile table + Panda + 3 pegs + the nuts for a layout {pegs: [(x, y, yaw)]*3, nuts: [(x, y, yaw)]}.
    `unit` (`sample_unit`) gives every nut and peg its real dimensions; without it they are the drawing's. The
    cell's state of the day is applied afterwards (`apply_cell_state`)."""
    root = mujoco.MjSpec.from_string(SCENE_XML.format(dt=TIMESTEP, tz2=TABLE_Z / 2, th=TABLE_HALF, rx=ROBOT_BASE[0]))
    wb = root.worldbody
    for name, (pos, tgt, fovy, _w, _h) in CAMERAS.items():
        c = wb.add_camera()
        c.name = name
        c.pos = list(pos)
        c.quat = list(quat_from_mat(look_at(pos, tgt)))
        c.fovy = fovy
    assets = robosuite_assets()
    robot = mujoco.MjSpec.from_file(os.path.join(assets, "robots", "panda", "robot.xml"))
    grip = mujoco.MjSpec.from_file(os.path.join(assets, "grippers", "panda_gripper.xml"))
    for cam in list(robot.cameras):
        if cam.name == "robotview":
            robot.delete(cam)
    robot.body("right_hand").add_frame().attach_body(grip.body("right_gripper"), "g_", "")
    wb.add_frame(pos=list(ROBOT_BASE)).attach_body(robot.body("base"), "r_", "")
    objs = mujoco.MjSpec.from_string(
        '<mujoco><compiler angle="radian" autolimits="true"/>'
        '<worldbody>{}{}</worldbody></mujoco>'.format(
            "".join(peg_body_xml(i, i + 1, unit["pegs"][i] if unit else None) for i in range(3)),
            "".join(nut_body_xml(sp, unit["nuts"][i] if unit else None)
                    .replace('<body name="nut">', '<body name="nut%d">' % i, 1)
                    .replace('name="nut_free"', 'name="nut%d_free"' % i, 1) for i, sp in enumerate(specs))))
    for i, (x, y, yaw) in enumerate(layout["pegs"]):
        wb.add_frame(pos=[x, y, TABLE_Z], quat=list(yaw_quat(yaw))).attach_body(objs.body("peg%d" % i), "o_", "")
    for i, (x, y, yaw) in enumerate(layout["nuts"]):
        wb.add_frame(pos=[x, y, TABLE_Z + 0.0005], quat=list(yaw_quat(yaw))).attach_body(objs.body("nut%d" % i), "o_", "")
    for act in root.actuators:
        if act.name.startswith("r_g_gripper_finger_joint"):
            act.gainprm[0] = GRIP_KP
            act.biasprm[1] = -GRIP_KP
    # the hand's two fingers are one mechanism (one drive, as on the real hand): they open and close symmetrically
    # about the hand's axis, so a part is centred when gripped and a held part cannot push the fingers aside together
    eq = root.add_equality()
    eq.type = mujoco.mjtEq.mjEQ_JOINT
    eq.name = "finger_coupling"
    eq.name1 = "r_g_finger_joint1"
    eq.name2 = "r_g_finger_joint2"
    eq.data[:5] = [0.0, -1.0, 0.0, 0.0, 0.0]
    eq.solref = list(FINGER_COUPLING)
    # materials for the cameras' renderer only (keyed_render.py gives each nut, each kind of peg part and the
    # gripper's parts one; the nuts and the pegs keep their geoms' own colours); no geom of this model uses them
    for name in ["look_nut%d" % i for i in range(len(specs))] + ["look_collar", "look_shaft", "look_pin", "look_band",
                                                                 "look_hand", "look_finger", "look_pad"]:
        root.add_material(name=name)
    m = root.compile()
    m.cam_fovy[m.camera("r_eye_in_hand").id] = WRIST["fovy"]
    return m


def sample_layout(rng, nuts=NUTS):
    """Peg positions/yaws (peg i has i + 1 bands) and the nut placements; rejection-sampled."""
    for _ in range(20000):
        pegs = []
        while len(pegs) < 3:
            x = rng.uniform(*PEG_REGION[0])
            y = rng.uniform(*PEG_REGION[1])
            if all(math.hypot(x - p[0], y - p[1]) >= PEG_MIN_SPACING for p in pegs):
                pegs.append((float(x), float(y), float(rng.uniform(-math.pi, math.pi))))
        placed = []
        for _k in range(200):
            if len(placed) == nuts:
                break
            nx, ny = rng.uniform(*NUT_REGION[0]), rng.uniform(*NUT_REGION[1])
            az = math.atan2(ny - ROBOT_BASE[1], nx - ROBOT_BASE[0])
            nyaw = float(wrap_angle(az + NUT_YAW_CENTER + rng.uniform(-NUT_YAW_SPREAD, NUT_YAW_SPREAD)))
            hx, hy = nx + 0.063 * math.cos(nyaw), ny + 0.063 * math.sin(nyaw)    # handle centre
            ok = all(math.hypot(nx - p[0], ny - p[1]) >= NUT_PEG_MIN_DIST and math.hypot(hx - p[0], hy - p[1]) >= 0.09
                     for p in pegs) and abs(hy) < 0.3 and hx > -0.28
            for (ox, oy, oyaw) in placed:
                ohx, ohy = ox + 0.063 * math.cos(oyaw), oy + 0.063 * math.sin(oyaw)
                ok = ok and math.hypot(nx - ox, ny - oy) >= NUT_MIN_SPACING \
                    and math.hypot(hx - ox, hy - oy) >= HANDLE_CLEAR and math.hypot(ohx - nx, ohy - ny) >= HANDLE_CLEAR \
                    and math.hypot(hx - ohx, hy - ohy) >= 0.05
            if ok:
                placed.append((float(nx), float(ny), nyaw))
        if len(placed) == nuts:
            order = rng.permutation(3)
            return {"pegs": [pegs[i] for i in order], "nuts": placed}
    raise EnvError("layout sampling failed")


# ----------------------------------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------------------------------
class NutEnv(object):
    """One kit: NUTS nuts on the table, each to be assembled on its own peg (the work order)."""

    def __init__(self, nut_specs, render=True, max_steps=MAX_STEPS, early_end=True, tolerances=True):
        """early_end: end the episode at the first ruined nut (a fault always ends it). tolerances: every episode
        is a real kit on the cell of the day (`sample_unit`, drawn from the episode's seed); False builds the
        drawing's kit on an ideal cell (a maintainer's check; neither the grader nor the service uses it)."""
        if isinstance(nut_specs, dict):
            nut_specs = [nut_specs]
        self.specs = [check_nut_spec(sp) for sp in nut_specs]
        self.n = len(self.specs)
        self.early_end = early_end
        self.tolerances = bool(tolerances)
        self.render_enabled = render
        self.max_steps = int(max_steps)
        self.model = None
        self._view = None
        self.lamp_gain = 1.0

    # ------------------------------------------------------------------ episode
    def reset(self, seed, order):
        """order[i] = band count (1-3) of the peg nut i goes on; with three nuts, a permutation of 1, 2, 3."""
        order = [int(b) for b in order]
        if len(order) != self.n or any(b not in (1, 2, 3) for b in order) or len(set(order)) != self.n:
            raise EnvError("order must give each nut a different peg (band count 1-3)")
        self.order = order
        self.seed = int(seed) & (2 ** 64 - 1)
        place = np.random.default_rng([self.seed & 0xFFFFFFFF, self.seed >> 32, 7])
        self.rng = np.random.default_rng([self.seed & 0xFFFFFFFF, self.seed >> 32, 11])
        self.layout = sample_layout(place, self.n)
        self.unit = (sample_unit(np.random.default_rng([self.seed & 0xFFFFFFFF, self.seed >> 32, 13]), self.n)
                     if self.tolerances else None)
        self.close()
        self.model = build_model(self.specs, self.layout, self.unit)
        self.data = mujoco.MjData(self.model)
        self.fk = mujoco.MjData(self.model)
        self._index()
        if self.unit is not None:
            apply_cell_state(self.model, self.unit)
        self.lamp_gain = 1.0 + float(self.unit["light"]) if self.unit is not None else 1.0   # the lamps' brightness
        self._init_state()
        self.wrench_bias = np.concatenate([self.rng.uniform(-WRENCH_BIAS, WRENCH_BIAS, 3),
                                           self.rng.uniform(-0.1 * WRENCH_BIAS, 0.1 * WRENCH_BIAS, 3)])
        self.wrench_drift = np.asarray(self.unit["wrench_drift"], float) if self.unit is not None else np.zeros(6)
        self.t = 0
        self.done = False
        self.fault = None
        self.broken = [False] * self.n
        self.force_count = [0] * self.n
        self.render_next = CAMERA_NAMES
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
        self.ft_site = m.site("r_g_ft_frame").id
        self.wrist_cam = m.camera("r_eye_in_hand").id
        # the published wrist-camera pose is the drawing's mount on the hand (taken before `apply_cell_state` moves
        # the real camera within its mounting tolerance)
        self.wrist_body = int(m.cam_bodyid[self.wrist_cam])
        self.wrist_pos0 = m.cam_pos[self.wrist_cam].copy()
        self.wrist_mat0 = mat_from_quat(m.cam_quat[self.wrist_cam]) @ np.diag([1.0, -1.0, -1.0])
        self.nut_body, self.nut_qadr, self.nut_dadr, self.nut_geoms = [], [], [], []
        for i in range(self.n):
            b = m.body("o_nut%d" % i).id
            j = m.joint("o_nut%d_free" % i)
            self.nut_body.append(b)
            self.nut_qadr.append(int(j.qposadr[0]))
            self.nut_dadr.append(int(j.dofadr[0]))
            self.nut_geoms.append(set(int(g) for g in np.where(m.geom_bodyid == b)[0]))
        fingers = set()
        for b in range(m.nbody):
            nm = m.body(b).name
            if nm.startswith("r_g_") and ("finger" in nm):
                fingers.add(b)
        self.finger_geoms = set(int(i) for i in np.where(np.isin(m.geom_bodyid, list(fingers)))[0])
        self.target_peg = [b - 1 for b in self.order]           # peg body i carries i + 1 bands
        self.peg_poses = [tuple(p) for p in self.layout["pegs"]]
        self.cells = [groove_cells(sp["groove"]) for sp in self.specs]
        u = self.unit
        # every nut's own wall grid (sector boundaries relative to boundary 0, layer boundaries) and groove floor
        self.wall0 = [wall_boundaries(u["nuts"][i] if u else None)[0] for i in range(self.n)]
        self.wall_rel = [[b - self.wall0[i] for b in wall_boundaries(u["nuts"][i] if u else None)] for i in range(self.n)]
        self.levels = [level_boundaries(u["nuts"][i] if u else None) for i in range(self.n)]
        self.groove_r = [R_GROOVE + (float(u["nuts"][i]["groove_r"]) if u else 0.0) for i in range(self.n)]
        # every peg's own pin: radius of its middle from the shaft axis, height of its centre above the table
        self.pin_mid = [((SHAFT_R + (float(u["pegs"][k]["shaft_r"]) if u else 0.0))
                         + (PIN_R + (float(u["pegs"][k]["pin_r"]) if u else 0.0))) / 2 for k in range(3)]
        self.pin_height = [TABLE_Z + COLLAR_H + PIN_Z + (float(u["pegs"][k]["pin_z"]) if u else 0.0) for k in range(3)]
        if u is not None:
            f = np.array(JOINT_FRICTION, float)
            phi1 = f[0] * np.asarray(u["joint_friction"], float)
            self.joint_friction = (phi1, f[1], f[2], phi1 / (1.0 + np.exp(-f[1] * f[2])))
        else:
            self.joint_friction = None
        self._jp, self._jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        self._M = np.zeros((m.nv, m.nv))

    def _init_state(self):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.jq] = HOME_Q
        d.qpos[self.gq[0]], d.qpos[self.gq[1]] = 0.04, -0.04
        mujoco.mj_forward(m, d)
        self.q_stale = d.qpos.copy()
        self.grip_target = 0.04
        self.q_null = NULLSPACE_Q.copy()
        q_hold = d.qpos[self.jq].copy()
        for _ in range(150):
            tau = 200.0 * (q_hold - d.qpos[self.jq]) - 20.0 * d.qvel[self.jv] + d.qfrc_bias[self.jv]
            d.ctrl[self.act] = tau
            d.ctrl[self.gact] = [0.04, -0.04]
            self.q_stale[:] = d.qpos          # the positions this step's poses are computed from
            mujoco.mj_step(m, d)
        self.target_pos = d.site_xpos[self.site].copy()
        self.target_R = d.site_xmat[self.site].reshape(3, 3).copy()

    # ------------------------------------------------------------------ control
    def _osc_torque(self):
        m, d = self.model, self.data
        jp, jr = self._jp, self._jr
        mujoco.mj_jacSite(m, d, jp, jr, self.site)
        J = np.vstack([jp, jr])[:, self.jv]
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
        qn = d.qpos[self.jq]
        dq = d.qvel[self.jv]
        tau += N @ (5.0 * (self.q_null - qn) - 6.0 * dq)
        tau -= 0.5 * dq
        over = np.abs(dq) - 1.5
        tau -= np.where(over > 0, 30.0 * over * np.sign(dq), 0.0)
        return tau

    def step(self, action):
        """Apply one 10 Hz action. Returns (obs, done, info). Raises ActionError on a malformed action."""
        if self.done:
            raise EnvError("episode already finished")
        mv, g, render = parse_action(action)
        d = self.data
        dp = mv[:3]
        n = np.linalg.norm(dp)
        if n > MAX_DPOS:
            dp = dp * (MAX_DPOS / n)
        dr = mv[3:]
        n = np.linalg.norm(dr)
        if n > MAX_DROT:
            dr = dr * (MAX_DROT / n)
        cur_p = d.site_xpos[self.site].copy()
        cur_R = d.site_xmat[self.site].reshape(3, 3).copy()
        self.target_pos = np.clip(cur_p + dp, WORKSPACE_LO, WORKSPACE_HI)
        self.target_R = axis_angle_mat(dr) @ cur_R
        self.grip_target = 0.04 * (1.0 - g) / 2.0
        self.render_next = render
        self._physics_step()
        self.t += 1
        self._post_step()
        return self.observation(), self.done, self.info()

    def _physics_step(self):
        m, d = self.model, self.data
        gw = [self.grip_target, -self.grip_target]
        jf = self.joint_friction
        for _ in range(SUBSTEPS):
            d.ctrl[self.act] = self._osc_torque()
            d.ctrl[self.gact] = gw
            if jf is not None:            # the joints' friction (JOINT_FRICTION), which the tracking law does not know
                d.qfrc_applied[self.jv] = jf[3] - jf[0] / (1.0 + np.exp(-jf[1] * (d.qvel[self.jv] + jf[2])))
            self.q_stale[:] = d.qpos          # the positions this step's poses are computed from
            mujoco.mj_step(m, d)
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            self.fault = "non-finite state"

    # ------------------------------------------------------------------ credit
    def nut_state(self, i=0):
        """(position of nut i's origin = hole axis on its bottom face, rotation matrix)."""
        d = self.data
        b = self.nut_body[i]
        return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()

    def _finger_touch(self, i):
        """(finger geoms, finger bodies) touching nut i."""
        d, m = self.data, self.model
        geoms, bodies = set(), set()
        ng = self.nut_geoms[i]
        for k in range(d.ncon):
            c = d.contact[k]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in ng and g2 in self.finger_geoms:
                geoms.add(g2)
                bodies.add(int(m.geom_bodyid[g2]))
            elif g2 in ng and g1 in self.finger_geoms:
                geoms.add(g1)
                bodies.add(int(m.geom_bodyid[g1]))
        return geoms, bodies

    def pin_in_wall(self, i=0):
        """True if the centre of nut i's target peg's pin lies inside a solid cell of the nut's wall, more than
        FORCE_ANGLE_TOL sideways or FORCE_Z_TOL up/down from every groove cell of the same layer/sector."""
        p, R = self.nut_state(i)
        k = self.target_peg[i]
        px, py, pyaw = self.peg_poses[k]
        rc = self.pin_mid[k]
        pin = np.array([px + rc * math.cos(pyaw), py + rc * math.sin(pyaw), self.pin_height[k]])
        q = R.T @ (pin - p)
        z = float(q[2])
        if not 0.0 < z < NUT_H or math.hypot(q[0], q[1]) > self.groove_r[i]:
            return False
        ang = math.degrees(math.atan2(q[1], q[0]))
        for dz in (0.0, FORCE_Z_TOL, -FORCE_Z_TOL):
            lay = bisect.bisect_right(self.levels[i], z + dz) - 1          # the nut's own layers
            for dang in (0.0, FORCE_ANGLE_TOL, -FORCE_ANGLE_TOL):
                sec = bisect.bisect_right(self.wall_rel[i], (ang + dang - self.wall0[i]) % 360.0) - 1   # ... and sectors
                if (sec, lay) in self.cells[i] or lay < 0:
                    return False
        return True

    def on_peg(self, i=0):
        """True while nut i is on its own peg: its hole axis within ENGAGE_XY of that peg's shaft axis, the nut
        below the shaft top and above the table (the condition under which `pin_in_wall` can ruin it)."""
        p, _R = self.nut_state(i)
        px, py, _pyaw = self.peg_poses[self.target_peg[i]]
        top = TABLE_Z + COLLAR_H + PEG_H
        return math.hypot(p[0] - px, p[1] - py) <= ENGAGE_XY and p[2] < top - 0.005 and p[2] > TABLE_Z

    def _post_step(self):
        d = self.data
        if self.fault is None:
            if np.max(np.abs(d.qvel[self.jv])) > INSTAB_ROBOT_QVEL:
                self.fault = "robot joint speed"
            elif any(np.linalg.norm(d.qvel[a:a + 3]) > INSTAB_NUT_VEL for a in self.nut_dadr):
                self.fault = "nut speed"
        if self.fault is not None:
            self.done = True
            return
        for i in range(self.n):
            if not self.broken[i] and self.on_peg(i):
                self.force_count[i] = self.force_count[i] + 1 if self.pin_in_wall(i) else 0
                if self.force_count[i] >= FORCE_STEPS:
                    self.broken[i] = True
        if (self.early_end and any(self.broken)) or self.t >= self.max_steps:
            self.done = True

    def info(self):
        return {"t": self.t, "fault": self.fault, "ruined": [i for i in range(self.n) if self.broken[i]]}

    # ------------------------------------------------------------------ sensing
    def close(self):
        """Free the episode's renderer resources."""
        if self._view is not None:
            try:
                self._view.close()
            except Exception:  # noqa: BLE001 - a renderer that failed to free must not stop the caller
                pass
        self._view = None

    def render(self, cam):
        """`cam`'s 8-bit RGB image now (keyed_render.py: the episode's drawn model is built at the first image),
        with the sensor noise. Raises RenderError."""
        if self._view is None:
            if _KR is None:
                raise RenderError(_KR_ERROR)
            self._view = _KR.open_view(self)
        rgb = self._view.render(cam)
        noise = self.rng.normal(0.0, RGB_NOISE, rgb.shape)
        return np.clip(rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    def _wrist_pose(self, d):
        """T_world_cam of the wrist camera as published: the drawing's mount on the hand, OpenCV convention."""
        Rb = d.xmat[self.wrist_body].reshape(3, 3)
        T = np.eye(4)
        T[:3, :3] = Rb @ self.wrist_mat0
        T[:3, 3] = d.xpos[self.wrist_body] + Rb @ self.wrist_pos0
        return T

    def wrist_camera_matrices(self):
        """(K, T_world_cam) of the wrist camera now as published (the drawing's intrinsics and mount), OpenCV
        convention. The real camera sits within its tolerances of that."""
        return camera_intrinsics(WRIST["fovy"], WRIST["width"], WRIST["height"]), self._wrist_pose(self.data)

    def work_order(self):
        """[{"colour": name, "peg_bands": b}] per nut (nut i first)."""
        return [{"colour": colour_name(sp["rgba"]), "peg_bands": b} for sp, b in zip(self.specs, self.order)]

    def observation(self):
        d = self.data
        R = d.site_xmat[self.site].reshape(3, 3)
        Rft = d.site_xmat[self.ft_site].reshape(3, 3)
        zero = self.wrench_bias + self.wrench_drift * (self.t * CONTROL_DT)      # the sensor's zero now
        f = Rft @ d.sensordata[self.ft_force:self.ft_force + 3] + zero[:3] + self.rng.normal(0, WRENCH_NOISE, 3)
        tq = (Rft @ d.sensordata[self.ft_torque:self.ft_torque + 3] + zero[3:]
              + self.rng.normal(0, WRENCH_NOISE * 0.1, 3))
        obs = {"t": int(self.t),
               "order": self.work_order(),
               "q": np.round(d.qpos[self.jq], 6),
               "dq": np.round(d.qvel[self.jv], 6),
               "ee_pos": np.round(d.site_xpos[self.site], 6),
               "ee_quat": np.round(quat_from_mat(R), 6),
               "gripper": round(float(d.qpos[self.gq[0]] - d.qpos[self.gq[1]]), 6),
               "wrench": np.round(np.concatenate([f, tq]), 4),
               "T_world_wrist": np.round(self.wrist_camera_matrices()[1], 6)}
        empty = np.zeros((0, 0, 3), np.uint8)
        for cam in CAMERA_NAMES:
            obs[cam] = self.render(cam) if (self.render_enabled and cam in self.render_next) else empty
        return obs

    def perturbed_observation(self, obs, rng):
        """A probe copy for the grader's lock: small joint-reading offsets (up to 0.001 rad) with the
        gripper pose and wrist-camera pose moved by the same offsets, and a small wrench offset.

        The observed poses are the simulation's own, computed from the joint positions before the last
        physics step, so they are not exactly the forward kinematics of the observed `q`. The copy is built
        the same way: its poses are the kinematics of those older positions plus the copy's offsets. (A copy
        whose poses were the exact kinematics of its `q` could be told from the real observation by anyone
        holding the model: its residual would be exactly zero.)"""
        q = np.asarray(obs["q"], float) + np.array([rng.uniform(-0.001, 0.001) for _ in range(7)])
        f = self.fk
        f.qpos[:] = self.q_stale
        f.qpos[self.jq] = self.q_stale[self.jq] + (q - np.asarray(obs["q"], float))
        mujoco.mj_kinematics(self.model, f)
        out = dict(obs)
        out["q"] = np.round(q, 6)
        out["ee_pos"] = np.round(f.site_xpos[self.site], 6)
        out["ee_quat"] = np.round(quat_from_mat(f.site_xmat[self.site].reshape(3, 3)), 6)
        out["T_world_wrist"] = np.round(self._wrist_pose(f), 6)
        w = np.asarray(obs["wrench"], float).copy()
        w[:3] += np.array([rng.uniform(-0.2, 0.2) for _ in range(3)])
        out["wrench"] = np.round(w, 4)
        return out


NUT_COLOURS = {"blue": [0.16, 0.36, 0.78, 1], "green": [0.20, 0.62, 0.30, 1], "red": [0.78, 0.18, 0.16, 1],
               "yellow": [0.86, 0.74, 0.16, 1], "orange": [0.85, 0.45, 0.12, 1], "purple": [0.55, 0.25, 0.70, 1],
               "teal": [0.15, 0.65, 0.68, 1], "pink": [0.80, 0.35, 0.55, 1]}


def colour_name(rgba):
    """The NUT_COLOURS name of an rgba (nearest)."""
    return min(NUT_COLOURS, key=lambda k: sum((a - b) ** 2 for a, b in zip(NUT_COLOURS[k][:3], rgba[:3])))


def sample_nut_spec(rng, nut_id="N", colour=None):
    """A random nut from the task's nut family (the hidden graded nuts are drawn with this exact function):
    entrance uniform over the 24 sectors; each turn 2, 3 or 4 sectors, either way; each turn channel continues
    into a dead end of 0-3 sectors the other way; 0, 1 or 2 false entrances at sectors at least 2 sectors clear
    of every cell of the true path and of each other; mass uniform over NUT_MASS; friction (the sliding
    friction between this nut and a peg) uniform over NUT_FRICTION;
    colour from NUT_COLOURS (uniform, unless `colour` names one). `rng` is a numpy Generator."""
    g = {"entrance": int(rng.integers(NSEC)),
         "turns": [int(rng.choice([2, 3, 4])) * int(rng.choice([-1, 1])) for _ in range(2)],
         "dead_ends": [int(rng.integers(4)) for _ in range(2)],
         "false_entrances": []}
    # sectors the true path and its first dead end occupy in the bottom layers (entrance riser and first turn)
    busy = {sec for (sec, lay) in groove_cells(g) if lay <= TURN_LAYERS[0][1]}
    for _ in range(int(rng.integers(3))):
        free = [f for f in range(NSEC)
                if all(min((f + d - b) % NSEC, (b - f - d) % NSEC) >= 2 for b in busy for d in (0, 1))]
        if not free:
            break
        f = int(rng.choice(free))
        g["false_entrances"].append(f)
        busy.update({f % NSEC, (f + 1) % NSEC})
    return check_nut_spec({"id": nut_id, "groove": g, "mass": round(float(rng.uniform(*NUT_MASS)), 4),
                           "friction": round(float(rng.uniform(*NUT_FRICTION)), 4),
                           "rgba": list(NUT_COLOURS[colour] if colour else
                                        NUT_COLOURS[sorted(NUT_COLOURS)[int(rng.integers(len(NUT_COLOURS)))]])})


def sample_kit(rng, kit_id="K"):
    """A random kit as graded (the hidden kits are drawn with this exact function): NUTS nuts drawn with
    `sample_nut_spec`, of NUTS different colours chosen uniformly, and the work order, a uniformly random
    assignment of the nuts to the pegs with 1, 2 and 3 bands. Returns {"nuts": [specs], "order": [bands]}."""
    names = sorted(NUT_COLOURS)
    cols = [names[int(i)] for i in rng.choice(len(names), NUTS, replace=False)]
    nuts = [sample_nut_spec(rng, "%s-%d" % (kit_id, i), colour=c) for i, c in enumerate(cols)]
    order = [int(b) + 1 for b in rng.permutation(3)][:NUTS]
    return {"nuts": nuts, "order": order}


def load_spec(path):
    with open(path) as fh:
        return check_nut_spec(json.load(fh))


def load_kit(path):
    """A kit file: {"nuts": [3 nut specs], "order": [band count per nut]}."""
    with open(path) as fh:
        kit = json.load(fh)
    return {"nuts": [check_nut_spec(sp) for sp in kit["nuts"]], "order": [int(b) for b in kit["order"]]}
