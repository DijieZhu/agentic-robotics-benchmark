#!/usr/bin/env python3
"""Reference controller for object_finding_and_organizing_on_messy_desk.

Works from the observation only (three static RGB-D cameras, the hand camera, proprioception, the
request text), with reference views of every model alone on the desk (refs.json: as it arrives
and knocked over, from the practice environment).

Perception: depth back-projection of the static cameras -> containers located from their rim band
(pose varies per episode) -> desk / container walls / floors / robot removed -> voxel clusters ->
colour histogram + sorted extents vs the references; the hand camera from above for a closer look
(dense cloud, its own references) and for the grasp: fingers across a side under the 8 cm opening,
the 0.2 m wide hand clear of every container wall and the fingers landing on nothing, the pads
centred on the object's height above what it rests on.

Belief: what the controller put into each container itself (model, slot, where it came to rest),
which containers may hold objects it never identified (the initial piles until taken apart, a
container the world dropped something into: its height map is compared with the one the controller
left), what it set down on the desk, which models it has identified alone on the desk and not put
away since, and where a carried object it had named slipped out of the fingers. Objects are sorted
into slots near the containers' centre lines, where a straight-down hand can take them out again; in
idle time the initial piles are taken apart one object at a time (identified alone on the desk, then
put back in a slot). A tall object standing on the desk that a pinch does not hold (a top-down
pinch reaches only its top ~6 cm, a bottle's sloped shoulder) is pushed over first.

Requests: loose candidates, then the containers' entries for the model, then whatever was never
identified that could be it (dug out onto the desk and identified alone), then a look into every
container. An object taken out of a container goes straight to red only when the view there is sure
of it (or the controller put it there itself and the view agrees); otherwise it is set down and
looked at alone first. Delivered is marked after the hand camera confirms the object in red, and not
when the target is seen lying loose after all (a long clamp can bounce out over the rim). No such
object only when nothing could be it: no entry, loose note or view inside a container names the
model, nothing unidentified is left where it could be (a container the world dropped something into
counts as clear once every unexplained object in it was seen to be clearly something else), and a
last look covers the blind ends of the strip in front of the base. A wrong mark costs a point and an
expired request nothing, so the controller never guesses. Before a mark, while time allows, the desk
is put in order: what the search took out goes back and loose or misfiled objects are sorted (the
world leaves the requested object alone while the request is open).

Every call recomputes its action from the observation it is given (closed-loop servo), so a
repeated query of the same step with perturbed proprioception answers differently.
"""
import base64
import io
import json
import math
import os
import sys
from collections import deque

# Single-threaded BLAS before numpy loads: the controller's arrays are small, and OpenBLAS's
# worker pool (one thread per host core, inside a container allowed fewer) once dead-locked a
# matrix-vector product, which froze the controller for the rest of the episode.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------------------
# Task geometry (task_card.md)
# ----------------------------------------------------------------------------------------
# nominal centres (task card): red in front of the base, the class containers behind it, the middle
# one set back behind red and the outer two forward (all of their floor within the arm's reach)
CONTAINER_NOMINAL = {"gray": (0.44, -0.31), "lightblue": (0.575, 0.0), "blue": (0.44, 0.31), "red": (0.25, 0.0)}
CONTAINER_OUTER = (0.326, 0.272, 0.120)          # class containers: outer footprint x (long), y (short), wall height
RED_OUTER = (0.237, 0.272, 0.120)                # the delivery container is shorter along x
CONTAINER_OUTERS = {n: (RED_OUTER if n == "red" else CONTAINER_OUTER) for n in CONTAINER_NOMINAL}
CONTAINER_INNER_HALF = (0.140, 0.120)
CONTAINER_INNER_HALVES = {n: ((0.102, 0.120) if n == "red" else CONTAINER_INNER_HALF) for n in CONTAINER_NOMINAL}
CONTAINER_FLOOR_Z, CONTAINER_RIM_Z = 0.0024, 0.1206
NEAR_STRIP_X = 0.27            # in front of this x the hand camera's views add to the static ones (the strip at the base)
CONTAINER_OF_CLASS = {"A": "gray", "B": "lightblue", "C": "blue"}
CLASS_OF_CONTAINER = {v: k for k, v in CONTAINER_OF_CLASS.items()}
CLASSES = {
    "A": ["cracker box", "sugar box", "pudding box", "gelatin box", "wood block", "foam brick"],
    "B": ["master chef can", "tomato soup can", "tuna fish can", "potted meat can", "mustard bottle", "bleach cleanser"],
    "C": ["banana", "bowl", "mug", "power drill", "scissors", "large marker", "large clamp", "extra large clamp", "pitcher base"],
}
CLASS_OF = {m: c for c, ms in CLASSES.items() for m in ms}
# look-alike pairs the classifiers confuse (same colours, a size apart): either may be the other
# Look-alikes for the hand camera that the classifier separates too poorly to stake a mark on: from
# straight above a standing can is its lid, and the master chef can (6.3 x 8.7 cm) and the tomato soup
# can (6.3 x 9.4 cm) differ by 7 mm of height (one was named the other three times running, in a
# container, in red and alone on the desk). An entry, note or view of either keeps the other from being
# declared absent, and a delivery of either is claimed only when the static cameras, which see the
# labels, do not name the other one in red.
CAN_TWIN = {"master chef can": "tomato soup can", "tomato soup can": "master chef can"}
# the same static check before claiming a delivery of either twin box (pudding / gelatin)
CLAIM_TWIN = dict(CAN_TWIN, **{"pudding box": "gelatin box", "gelatin box": "pudding box"})
TWIN = {"large clamp": "extra large clamp", "extra large clamp": "large clamp", "pudding box": "gelatin box", "gelatin box": "pudding box"}


CLAMPS = ("large clamp", "extra large clamp")
CLAMP_SPLIT = 0.14      # longest extent (m) seen by the hand camera: the large clamp reads about 0.125, the extra large 0.155-0.19


def is_model(info, model, slack=0.25):
    """The classification `info` names `model`, or its look-alike twin with `model` close behind
    (not for the clamps: those are told apart by size in clamp_by_size)."""
    if info["model"] == model:
        return True
    if info["model"] in CLAMPS:
        return False
    return TWIN.get(info["model"]) == model and info.get("all", {}).get(model, 9.0) - info["d"] < slack


def clamp_by_size(info, dims):
    """The two clamps are one shape at two sizes and their per-model distances barely differ, so
    when the nearest model is a clamp the hand camera's longest extent names it, and the model
    margin is taken against the nearest non-clamp model. (An extra large clamp delivered for a
    requested large one was a wrong claim.)"""
    if not info or info.get("model") not in CLAMPS:
        return info
    info = dict(info)
    info["model"] = "extra large clamp" if max(dims) > CLAMP_SPLIT else "large clamp"
    others = [d for m, d in info.get("all", {}).items() if m not in CLAMPS]
    if others:
        info["model_margin"] = min(others) - info["d"]
    return info
DESK_X, DESK_Y = (0.15, 0.78), (-0.62, 0.62)
REACH, BASE_CLEAR = 0.80, 0.36
IMG = 320                                   # task cameras are IMG x IMG; focal scales with the width
F_PX, C_PX = 171.03317 * IMG / 224.0, IMG / 2.0
CAMS = {
    "agentview": {"f": F_PX, "c": C_PX,
                  "pose": [1.35, 0.0, 0.95, 0.0, -0.375099, 0.0, 0.926985]},
    "left_view": {"f": F_PX, "c": C_PX,
                  "pose": [0.45, -1.15, 0.75, 0.6808448119669411, -0.19091972663321244, 0.19091972663321244, 0.6808448119669411]},
    "oblique": {"f": F_PX, "c": C_PX,          # front-left corner of the desk, looking at the containers
                "pose": [1.25, -0.95, 0.85, 0.3981, -0.2464, 0.1119, 0.8765]},
}


def intrinsics(w):
    """Focal and principal point for a w x w frame of the task cameras."""
    return F_PX * w / IMG, w / 2.0
POS_STEP_MAX, ROT_STEP_MAX = 0.05, 0.25
GRIPPER_OPEN_W = 0.08

# hand geometry (Franka hand, fingers open): half width across the fingers, half thickness
HAND_HALF_W, HAND_HALF_T = 0.1025, 0.0315      # measured on the Franka USD (body 0.205 x 0.063)
GRASP_MAX_WIDTH = 0.080     # the hand's opening
WIDTH_MARGIN = 0.014        # the hand-camera cloud loses ~7 mm per side at an object's silhouette
GRIP_SETTLE_MAX = 12        # steps to wait for the fingers to stop after the close command

# controller tunables
OBSERVE_POS = (0.30, 0.0, 0.55)
TRAVEL_Z = 0.38               # carry height: well above the walls and whatever stands in a pile
VOXEL = 0.012
Z_ROBOT = 0.30
GRASP_BELOW_TOP = 0.03
RIM_GRASP_BELOW_TOP = 0.022     # wide hollow objects (bowl, mug, pitcher): straddle the wall at the rim
RIM_MIN_RADIUS, RIM_MIN_DEPTH = 0.035, 0.012
REFINE_RADIUS, REFINE_MAX_SHIFT = 0.07, 0.04    # hand-camera re-centring just before the descent
REFINE_NEAR = 0.05           # the re-centring looks at the top around the anchor only, not a touching neighbour's
TOP_UNDER_RADIUS = 0.025     # the top that sets the finger height: right under the grasp point
TOP_DROP_MIN = 0.015         # ... used when it lies this far below the planned top
GRASP_MIN_ABOVE_FLOOR = 0.032
HOVER_ABOVE_TOP = 0.16
PATROL_EVERY = 600           # idle steps between looks into the class containers (world misplacements)
RETRY_MAX, RETRY_JITTER, RETRY_YAW = 1, 0.012, 0.15   # grasp attempts per object (a shifted retry rescued 0 of 23 failed grasps in the lab and the episode logs, so one), shift across the fingers (m), turn (rad)
MARK_LAST = 20                # steps before a request's deadline at which the mark goes out whatever the arm is doing
DESCENT_SHORT = 0.015         # a final descent that stops this far from the grasp point does not close
STUCK_ERR, STUCK_AFTER = 0.10, 4   # moves ending this far off, this many in a row: the arm is folded up, recover (a descent that stops 3-4 cm short at the edge of reach is not that)
HOLLOW = ("bowl", "mug", "pitcher base")
RED_REACH_X = 0.27            # an object in the red container nearer the base than this is left alone (the hand stops above it)
TIDY_R = (0.36, 0.76)         # loose objects nearer the base or farther out than this are left alone: beyond 0.76 a straight-down hand lifted under half and every attempt cost a stuck-arm recovery (one episode spent its second half in that loop over a clamp pushed to r = 0.80); at 0.36-0.40 it lifts about half (11 of 21 in the lab), and an object pushed nearer the base is set back on the desk by the world
DIG_FAILS_MAX = 5             # failed grasps in one container before a request stops digging there
SPOT_WALL_CLEAR = 0.075       # a desk spot for setting an object down keeps this far from every container
UNPACK_FAILS_MAX = 2          # failed grasps at one spot of a container within one request before the dig moves on
STALL_ROUNDS = 2              # rounds of "nothing in reach" in a container before the other containers are dug as well
# belief-driven behaviour (v4)
QUICK_OBSERVE_Z = 0.34        # with the hand above this, a cycle looks from where the hand is
FULL_OBSERVE_EVERY = 150      # steps between looks from the observe pose (the strip in front of the base)
EXPLORE_PILES = False         # take the initial piles apart in idle time (identified on the desk, back in slots)
EXPLORE_LITE = True           # ... or one object at a time in idle time, each put back right after it was identified
SWEEP_EVERY = 1000            # steps between looks at the blind ends of the strip in front of the base
# slots in a class container (local x along its long side, y across, finger-axis direction in the container
# frame): the middle, both ends on the centre line (fingers across the container: a straight-down hand
# fits there with about 1.3 cm of slack), both sides of the middle (fingers along it); anything placed
# there can be taken out again by a straight-down hand
CLASS_SLOTS = [(0.0, 0.0, 0.0), (0.085, 0.0, math.pi / 2), (-0.085, 0.0, math.pi / 2), (0.0, 0.068, 0.0), (0.0, -0.068, 0.0)]
RED_SLOTS = [(0.0, 0.0, math.pi / 2)]
SLOT_CLEAR = 0.035            # radius around a slot centre that must be empty for the slot to count as free
PLACE_GAP = 0.015             # the held object's bottom this far above the surface under it at release
RESTORE_MARGIN = 200          # steps left before a deadline that still allow putting one dug-out object back
ENTRY_TRIES = 2               # visits per request to one container entry of the requested model
ENTRY_RADIUS = 0.07           # an entry's object is looked for within this distance of where it was put
CANDIDATE_SLACK = 0.6         # static-camera gate for loose objects that may be the requested model
IN_CONTAINER_SURE_D = 0.5     # hand-camera classification inside a container this close to a reference: sure
IN_CONTAINER_UNSURE = 0.8     # ... this far from every reference: an object never identified
DESK_SURE_D = 0.6             # hand-camera classification of an object alone on the desk: sure
DESK_SURE_MARGIN = 0.2        # ... and this far ahead of the next model
FINGER_PAD_LOW = 0.028        # the open fingertips this far under the tool centre point (they touch the desk at z 0.028)
FINGER_PAD_HIGH = 0.010       # the upper edge of the finger pads under the tool centre point
CARRY_STEP = 0.035            # metres per step while carrying (5 cm steps shook thin objects out of the fingers)
CHANGE_CELL = 0.012           # height-map cell for change detection in a container
CHANGE_RISE = 0.03            # a cell this much higher than the reference is new
CHANGE_MIN_CELLS = 20         # ... and this many such cells (~30 cm2) are an object dropped in
FINGER_HALF_WIDTH = 0.012     # half the width of a finger (across the closing direction)
FINGER_THICK = 0.014          # a finger's thickness outward from its pad
RESTORE_STEPS = 110           # steps one object takes to go back into its container
LOOK_SLACK = 0.6             # an unidentified object in a container is dug out for a request when the model is this close
HELD_MIN_WIDTH = 0.002        # finger opening above which something is still between the fingers
PAD_OVERLAP = 0.012           # least overlap of the finger pads with the object across the fingers
FAILED_YAW_PENALTY = 0.04     # per earlier failure of about the same grasp at about the same spot
FAILED_MEMORY = 3000          # steps a failed grasp is remembered
TILT_PENALTY = 0.03           # a tilted grasp costs this much more (in metres of width) than an upright one
SKIP_AFTER_FAIL = 300         # steps a loose object whose grasp failed is left alone (x (1 + earlier fails)^2; the next try takes another grasp)
PLACE_SETTLE = 8              # steps to let a placed object settle before the container is looked at
TOPPLE_MIN_H = 0.105          # a loose object standing this tall whose grasp failed is pushed over: a top-down pinch
                              # reaches only its top ~6 cm (the palm stops at the top), on a standing bottle the sloped
                              # shoulder it slips from (bleach 0.146, mustard 0.135 standing)
TOPPLE_MAX_FOOT = 0.075       # ... when the short side of its footprint is under this
TOPPLE_MAX = 2                # pushes at about one spot within FAILED_MEMORY steps
TARGET_TRIES = 6              # grasp attempts per request at the spot where the target was identified
PALM_CLEAR = 0.004            # the palm stays this far above everything under its footprint
LOG = os.environ.get("MESSY_DESK_SOLVER_LOG", "")


def log(*a):
    print("[solver]", *a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------
# Small maths
# ----------------------------------------------------------------------------------------
def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_half(a):
    """Wrap to (-pi/2, pi/2]: a gripper yaw is only defined modulo pi."""
    return (a + math.pi / 2) % math.pi - math.pi / 2


def yaw_branch(xy, yaw, tilt=None):
    """Of the two hand yaws that are the same grasp for the symmetric hand (yaw and yaw + pi),
    the one that keeps the wrist's twist small relative to the arm's azimuth toward (x, y). On
    the +y side of the desk a hand turned to -90 degrees ran a joint into its limit (the descent
    stopped short or the arm folded up in 19 of 34 grasps in the episode logs; turned to +90
    degrees, 1 of 7). The lean, defined in the hand's frame, flips with the half turn."""
    phi = math.atan2(xy[1], xy[0])
    y2 = wrap(phi + wrap_half(yaw - phi))
    if is_tilted(tilt) and abs(wrap(y2 - yaw)) > 1.0:
        tilt = (-tilt[0], -tilt[1])
    return y2, tilt


def rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-6:
        return np.zeros(3)
    if ang > math.pi - 1e-3:
        d = np.sqrt(np.maximum(np.diag(R) + 1.0, 0.0) / 2.0)
        return d / max(np.linalg.norm(d), 1e-9) * ang
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * math.sin(ang))
    return v * ang


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ----------------------------------------------------------------------------------------
# Containers (pose varies per episode: centre x, y and yaw of the long side)
# ----------------------------------------------------------------------------------------
def nominal_containers():
    return {n: (x, y, 0.0) for n, (x, y) in CONTAINER_NOMINAL.items()}


def outer_half(name):
    o = CONTAINER_OUTERS.get(name, CONTAINER_OUTER)
    return o[0] / 2, o[1] / 2


def inner_half(name):
    return CONTAINER_INNER_HALVES.get(name, CONTAINER_INNER_HALF)


def container_local(b, x, y):
    """Desk xy -> coordinates in container b's frame (x along its long side). Vectorised."""
    bx, by, yaw = b
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x - bx, y - by
    return c * dx + s * dy, -s * dx + c * dy


def container_point(b, lx, ly):
    bx, by, yaw = b
    c, s = math.cos(yaw), math.sin(yaw)
    return bx + c * lx - s * ly, by + s * lx + c * ly


def container_of_xy(containers, x, y, shrink=0.0):
    for name, b in containers.items():
        lx, ly = container_local(b, x, y)
        ix, iy = inner_half(name)
        if abs(lx) <= ix - shrink and abs(ly) <= iy - shrink:
            return name
    return None


def footprint_dist(containers, name, x, y):
    """Distance from (x, y) to container `name`'s outer footprint (0 inside it)."""
    lx, ly = container_local(containers[name], x, y)
    ox, oy = outer_half(name)
    return math.hypot(max(abs(lx) - ox, 0.0), max(abs(ly) - oy, 0.0))


def hand_corners(x, y, yaw, margin=0.0):
    """Corners + edge midpoints of the open hand's footprint (finger axis = home +y turned by yaw)."""
    u = np.array([-math.sin(yaw), math.cos(yaw)])        # across the fingers
    v = np.array([math.cos(yaw), math.sin(yaw)])          # hand thickness
    hw, ht = HAND_HALF_W + margin, HAND_HALF_T + margin
    pts = []
    for a in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for bb in (-1.0, 0.0, 1.0):
            pts.append((x + a * hw * u[0] + bb * ht * v[0], y + a * hw * u[1] + bb * ht * v[1]))
    return pts


def hand_fits(containers, x, y, yaw, inside=None):
    """Open hand lowered at (x, y) with `yaw`: outside every container's outer footprint, or (when
    `inside` names a container) entirely within that container's interior."""
    pts = hand_corners(x, y, yaw, 0.005)
    for name, b in containers.items():
        ox, oy = outer_half(name)
        ix, iy = inner_half(name)
        for px, py in pts:
            lx, ly = container_local(b, px, py)
            if name == inside:
                # inside a container the hand body (0.205 x 0.063) fits only near the centre line:
                # every millimetre of margin counts (the containers are kinematic and do not yield)
                if abs(lx) > ix - 0.004 or abs(ly) > iy - 0.004:
                    return False
            elif abs(lx) <= ox + 0.004 and abs(ly) <= oy + 0.004:
                return False
    return True


HAND_BODY_ABOVE_TCP = 0.034     # the hand body's underside above the tool centre point (measured)
FINGER_BELOW_TCP = 0.009        # open fingertips below the tool centre point (contact probe: a can's top meets the
                                # finger with the tool centre 0.9 cm above it; the earlier 0.028 was wrong)
RIM_CLEAR = 0.012               # a hand point this far above a rim passes over that container
# hand tilts tried when the straight-down hand does not fit: (lean along the finger axis, lean
# along the thickness axis), radians; the wrist leans the opposite way, away from the near wall
# (Compound leans, 0.35 or 0.55 along the finger axis with 0.30 across, lifted 7 of 67 in the
# development episodes, and each cost 40-60 steps of a request's 1200: they are not tried. A single
# lean of 0.30 lifted about a third of the time, 0.45 along the fingers about a sixth.)
TILTS = [(0.0, 0.0)] + [(sg * a, 0.0) for a in (0.30, 0.45) for sg in (1, -1)] \
    + [(0.0, sg * b) for b in (0.30, 0.45) for sg in (1, -1)]


def _hand_points():
    """Sample points of the open hand in its own frame (finger axis, thickness axis, up), metres
    from the tool centre point: the body, the two fingers, the wrist above."""
    pts = []
    for su in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for sv in (-1.0, 0.0, 1.0):
            for h in (HAND_BODY_ABOVE_TCP, HAND_BODY_ABOVE_TCP + 0.04, HAND_BODY_ABOVE_TCP + 0.08):
                pts.append((su * HAND_HALF_W, sv * HAND_HALF_T, h))
    for su in (-1.0, 1.0):
        for sv in (-1.0, 1.0):
            for h in (-FINGER_BELOW_TCP, 0.0, HAND_BODY_ABOVE_TCP):
                pts.append((su * (GRIPPER_OPEN_W / 2 + 0.008), sv * 0.011, h))
    for su in (-1.0, 0.0, 1.0):          # the wrist, last three points (hand_fits3d relies on that)
        pts.append((su * 0.045, 0.0, 0.17))
    return np.array(pts)


HAND_PTS = _hand_points()


def axis_rot(axis, a):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * K @ K


def tilt_rot(yaw, tilt):
    """World rotation on top of the straight-down hand at `yaw`: tilt[0] > 0 leans the fingertips
    toward +finger axis, tilt[1] > 0 toward -thickness axis (the wrist leans the other way)."""
    u = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    v = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return axis_rot(v, tilt[0]) @ axis_rot(u, tilt[1])


def is_tilted(tilt):
    return tilt is not None and (abs(tilt[0]) > 1e-3 or abs(tilt[1]) > 1e-3)


def hand_fits3d(containers, p, yaw, tilt=(0.0, 0.0), inside=None):
    """The open hand with its tool centre point at p (x, y, z), `yaw` and `tilt`: every hand point
    below a container's rim must be clear of that container's walls (inside its interior when it
    is the container `inside`, outside its footprint otherwise). Points above the rims are free,
    which is what lets a hand reach an object high in a pile, or near a wall when tilted."""
    u = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    v = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    B = np.stack([u, v, np.array([0.0, 0.0, 1.0])], 1)
    R = tilt_rot(yaw, tilt) if is_tilted(tilt) else np.eye(3)
    W = np.asarray(p, float) + (R @ B @ HAND_PTS.T).T
    below = W[:, 2] < CONTAINER_RIM_Z + RIM_CLEAR
    for name, b in containers.items():
        ox, oy = outer_half(name)
        ix, iy = inner_half(name)
        lx, ly = container_local(b, W[:, 0], W[:, 1])
        in_outer = (np.abs(lx) <= ox + 0.004) & (np.abs(ly) <= oy + 0.004)
        if name == inside:
            # the hand comes in from above: below the rim, all of it is inside the interior
            in_inner = (np.abs(lx) <= ix - 0.004) & (np.abs(ly) <= iy - 0.004)
            if np.any(below & ~in_inner):
                return False
            # and it may only lean toward the container's middle: the wrist (and the forearm
            # above it, which the point model does not carry) stays over the interior. A hand
            # leaning out over a wall clears the point model above the rim but lands its wrist
            # on the rim, where it hangs 6-8 cm short of the grasp
            wl = lx[-3:]
            wy = ly[-3:]
            if np.any(np.abs(wl) > ix) or np.any(np.abs(wy) > iy):
                return False
        elif np.any(below & in_outer):
            return False
    return bool(np.all(W[:, 2] > -0.004))


def fmt_tilt(tilt):
    return "%+.2f/%+.2f" % (tilt[0], tilt[1]) if is_tilted(tilt) else "0"


def fmt_plans(g):
    return [(round(x[1], 2), round(x[2], 3), fmt_tilt(x[3] if len(x) > 3 else None)) for x in g]


def approach_dir(yaw, tilt):
    """Unit vector from the wrist toward the fingertips."""
    R = tilt_rot(yaw, tilt) if is_tilted(tilt) else np.eye(3)
    return -(R @ np.array([0.0, 0.0, 1.0]))


def fits_with_approach(containers, p, yaw, tilt, inside):
    """The final grasp pose fits, and so does the pose 4 cm back along the approach."""
    if not hand_fits3d(containers, p, yaw, tilt, inside):
        return False
    return hand_fits3d(containers, np.asarray(p, float) - 0.04 * approach_dir(yaw, tilt), yaw, tilt, inside)


# ----------------------------------------------------------------------------------------
# Perception
# ----------------------------------------------------------------------------------------
def decode_png(b64, mode=None):
    im = Image.open(io.BytesIO(base64.b64decode(b64)))
    if mode:
        im = im.convert(mode)
    return np.array(im)


EDGE_TAU = 0.012     # depth jump to a 4-neighbour (m) that marks an anti-aliased silhouette pixel
EDGE_FILTER_STATIC = False   # the renderer runs without anti-aliasing: depth is exact, the filter only lost thin objects
EDGE_FILTER_EIH = True


def depth_edges(z, tau=EDGE_TAU):
    """Silhouette pixels: the renderer anti-aliases depth, so pixels on an object's outline carry
    a depth between the object and what lies behind it and back-project into a 'skirt' of flying
    points hanging from the rim down to the desk (2-4 cm around every object, and long strips
    under the fingers). Drop every pixel whose depth differs from any 4-neighbour by more than `tau`."""
    e = np.zeros(z.shape, dtype=bool)
    d = np.abs(np.diff(z, axis=0)) > tau
    e[:-1, :] |= d
    e[1:, :] |= d
    d = np.abs(np.diff(z, axis=1)) > tau
    e[:, :-1] |= d
    e[:, 1:] |= d
    return e


def backproject(depth_mm, rgb, cam, max_depth=3.0):
    z = depth_mm.astype(np.float32) / 1000.0
    h, w = z.shape
    f, c = intrinsics(w)
    v, u = np.mgrid[0:h, 0:w]
    valid = (z > 0.05) & (z < max_depth)
    if EDGE_FILTER_STATIC:
        valid &= ~depth_edges(z)
    pts = np.stack([z[valid], -(u[valid] - c) / f * z[valid], -(v[valid] - c) / f * z[valid]], 1)
    pose = cam["pose"]
    world = pts @ quat_to_mat(pose[3:7]).T + np.asarray(pose[:3])
    return world, rgb[valid].astype(np.float32) / 255.0


def backproject_rt(depth_mm, rgb, f, c, R, t, max_depth=1.5, edge_filter=None):
    z = depth_mm.astype(np.float32) / 1000.0
    h, w = z.shape
    f, c = intrinsics(w)
    v, u = np.mgrid[0:h, 0:w]
    valid = (z > 0.02) & (z < max_depth)
    if EDGE_FILTER_EIH if edge_filter is None else edge_filter:
        valid &= ~depth_edges(z)
    pts = np.stack([z[valid], -(u[valid] - c) / f * z[valid], -(v[valid] - c) / f * z[valid]], 1)
    return pts @ R.T + np.asarray(t), rgb[valid].astype(np.float32) / 255.0


def label_components(idx):
    """26-connectivity labels for integer voxel coordinates (N,3) -> labels (N,) 0-based, count."""
    keys = {}
    for i, v in enumerate(map(tuple, idx)):
        keys.setdefault(v, []).append(i)
    labels = np.full(len(idx), -1, int)
    nb = [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) != (0, 0, 0)]
    comp = 0
    for v in list(keys):
        if labels[keys[v][0]] >= 0:
            continue
        dq = deque([v])
        labels[keys[v]] = comp
        while dq:
            cur = dq.popleft()
            for d in nb:
                n = (cur[0] + d[0], cur[1] + d[1], cur[2] + d[2])
                lst = keys.get(n)
                if lst is not None and labels[lst[0]] < 0:
                    labels[lst] = comp
                    dq.append(n)
        comp += 1
    return labels, comp


def fit_container(p, cx, cy, yaw0=0.0, name=None):
    """Refine one container's pose from rim-band points near its nominal centre: coarse-to-fine grid
    search over (x, y, yaw) maximising the number of points on the outline of the known outer
    footprint. Robust to partial rings (a container seen from one side only)."""
    ox, oy = outer_half(name)
    q = p[np.hypot(p[:, 0] - cx, p[:, 1] - cy) < 0.32]
    if len(q) < 30:
        return None
    qx, qy = q[:, 0], q[:, 1]

    def score(x, y, yaw):
        c, sn = math.cos(yaw), math.sin(yaw)
        lx = (qx - x) * c + (qy - y) * sn
        ly = -(qx - x) * sn + (qy - y) * c
        on_long = (np.abs(np.abs(ly) - oy) < 0.02) & (np.abs(lx) < ox + 0.02)
        on_short = (np.abs(np.abs(lx) - ox) < 0.02) & (np.abs(ly) < oy + 0.02)
        # rim-height points well inside the outline belong to a wall the outline has swallowed
        # (a neighbour's, 3 cm away), or to a tall object: a mild penalty breaks the tie
        inside = (np.abs(lx) < ox - 0.035) & (np.abs(ly) < oy - 0.035)
        return float((on_long | on_short).sum()) - 0.5 * float(inside.sum())

    # the search stays within the poses the task card allows (jitter x +-0.02, y +-0.008, yaw +-5
    # degrees, plus a little slack): a wider search snapped the outline onto the next container's
    # wall and left that wall standing inside this container as a phantom object
    best = (-1e9, cx, cy, yaw0)
    for yaw in yaw0 + np.radians(np.arange(-7, 7.1, 1.0)):
        for dx in np.arange(-0.03, 0.031, 0.01):
            for dy in np.arange(-0.012, 0.0121, 0.004):
                sc = score(cx + dx, cy + dy, yaw)
                if sc > best[0]:
                    best = (sc, cx + dx, cy + dy, yaw)
    _, bx, by, byaw = best
    for yaw in byaw + np.radians(np.arange(-1.0, 1.01, 0.25)):
        for dx in np.arange(-0.006, 0.0061, 0.002):
            for dy in np.arange(-0.004, 0.0041, 0.002):
                sc = score(bx + dx, by + dy, yaw)
                if sc > best[0]:
                    best = (sc, bx + dx, by + dy, yaw)
    return best


def detect_containers(pts, col):
    """Container poses from the rim band of the fused cloud (the upper walls, above most objects):
    each container is fitted around its nominal pose with the known footprint. A container with too few
    band points keeps its nominal pose."""
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    band = (z > CONTAINER_RIM_Z - 0.05) & (z < CONTAINER_RIM_Z + 0.02) & (x > 0.10) & (x < 0.80) & (np.abs(y) < 0.65)
    containers = nominal_containers()
    fits = []
    if band.sum() < 40:
        return containers, fits
    p = pts[band][:, :2]
    for name, (bx, by, _) in list(containers.items()):
        r = fit_container(p, bx, by, name=name)
        if r is None or r[0] < 40:
            continue
        sc, fx, fy, fyaw = r
        containers[name] = (float(fx), float(fy), float(fyaw))
        fits.append({"name": name, "score": sc, "centre": [float(fx), float(fy)], "yaw": float(fyaw)})
    return containers, fits


# The bare desk top (the stock table): flat at z = -0.0014 out to x = 0.809, then its rounded rim
# rises 1.2 cm (z = +0.0104 at x = 0.839-0.854) and falls away past x = 0.90 (task card).
DESK_PROFILE_X = (0.0, 0.809, 0.839, 0.854, 0.872, 0.884, 0.896, 0.907, 0.918, 0.929, 0.940, 0.950)
DESK_PROFILE_Z = (-0.0014, -0.0014, 0.0104, 0.0104, 0.0094, 0.0075, 0.0047, 0.0011, -0.0035, -0.0089, -0.0152, -0.0223)


def desk_top_z(x):
    return np.interp(np.abs(x), DESK_PROFILE_X, DESK_PROFILE_Z)


def scene_mask(pts, containers):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # the whole desk top except the robot's base column; objects can slide past the arrival area.
    # Heights are taken above the desk's own profile, so its raised rim is not an object.
    h = z - desk_top_z(x)
    keep = (x > 0.08) & (x < 0.80) & (np.abs(y) < 0.70) & (np.hypot(x, y) > 0.20) & (h > 0.012) & (z < Z_ROBOT)
    oz = CONTAINER_OUTER[2]
    for name, b in containers.items():
        ox, oy = outer_half(name)
        ix, iy = inner_half(name)
        lx, ly = container_local(b, x, y)
        in_outer = (np.abs(lx) <= ox + 0.014) & (np.abs(ly) <= oy + 0.014)
        in_inner = (np.abs(lx) <= ix - 0.010) & (np.abs(ly) <= iy - 0.010)
        keep &= ~(in_outer & ~in_inner & (z < oz + 0.025))
        keep &= ~(in_inner & (z < CONTAINER_FLOOR_Z + 0.015))
    return keep


def rgb_to_hsv(c):
    r, g, b = c[:, 0], c[:, 1], c[:, 2]
    mx, mn = c.max(1), c.min(1)
    d = mx - mn
    nz = d > 1e-6
    dd = np.where(nz, d, 1)
    h = np.where(mx == r, ((g - b) / dd) % 6, np.where(mx == g, (b - r) / dd + 2, (r - g) / dd + 4))
    h = np.where(nz, h / 6.0, 0.0)
    s = np.where(mx > 1e-6, d / np.where(mx > 1e-6, mx, 1), 0)
    return np.stack([h, s, mx], 1)


def colour_hist(hsv, nh=12, ns=3, nv=4):
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    sat = s > 0.25
    hs = np.zeros((nh, ns))
    if sat.any():
        hi = np.minimum((h[sat] * nh).astype(int), nh - 1)
        si = np.minimum(((s[sat] - 0.25) / 0.75 * ns).astype(int), ns - 1)
        np.add.at(hs, (hi, si), 1)
    gv = np.zeros(nv)
    if (~sat).any():
        vi = np.minimum((v[~sat] * nv).astype(int), nv - 1)
        np.add.at(gv, vi, 1)
    hist = np.concatenate([hs.ravel(), gv])
    return hist / max(hist.sum(), 1)


def hist_dist(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(0.5 * np.sum((a - b) ** 2 / (a + b + 1e-6)))



def cluster_features(p, c, containers):
    cen = p.mean(0)
    top = float(np.percentile(p[:, 2], 97))
    xy = p[:, :2] - cen[:2]
    cov = xy.T @ xy / max(len(p), 1)
    evals, evecs = np.linalg.eigh(cov)
    major = evecs[:, 1]
    yaw = float(np.arctan2(major[1], major[0]))
    proj = xy @ evecs
    ext = np.percentile(proj, 97, axis=0) - np.percentile(proj, 3, axis=0)
    b = container_of_xy(containers, cen[0], cen[1])
    floor = CONTAINER_FLOOR_Z if b else float(desk_top_z(cen[0]))
    slab = p[p[:, 2] > top - 0.03]
    gxy = slab[:, :2].mean(0) if len(slab) >= 5 else cen[:2]
    slab_yaw = yaw
    if len(slab) >= 12:
        sxy = slab[:, :2] - slab[:, :2].mean(0)
        sev, svec = np.linalg.eigh(sxy.T @ sxy / len(slab))
        slab_yaw = float(np.arctan2(svec[1, 1], svec[0, 1]))
    return {"n": int(len(p)), "centroid": cen.tolist(), "grasp_xy": [float(gxy[0]), float(gxy[1])],
            "top_z": top, "floor": floor, "container": b, "yaw": yaw, "slab_yaw": slab_yaw,
            "len_major": float(ext[1]), "len_minor": float(ext[0]),
            "dims": sorted([float(ext[1]), float(ext[0]), float(top - floor)]),
            "mean_rgb": c.mean(0).tolist(), "hist": colour_hist(rgb_to_hsv(c)).tolist(), "pts": p}


def perceive(images, containers=None, voxel=VOXEL, min_pts=12, extra=()):
    """images: {cam: (rgb uint8, depth uint16)} for the static cameras, plus optional extra
    views as (rgb, depth, R_cam, t_cam) -> (clusters, containers)."""
    P, C = [], []
    for cam in CAMS:
        rgb, d = images[cam]
        pts, col = backproject(d, rgb, CAMS[cam])
        P.append(pts)
        C.append(col)
    allp, allc = np.concatenate(P), np.concatenate(C)
    if containers is None:
        # container poses from the static cameras only: they see all four rims completely,
        # while a hand-camera view clips them at its edge and biases the fit
        containers, _ = detect_containers(allp, allc)
    for rgb, d, R, t in extra:
        pts, col = backproject_rt(d, rgb, None, None, R, t)
        # extra (hand-camera) views only fill the strip between the robot and the containers,
        # the one place the static cameras cannot see; elsewhere they add nothing but noise
        keep = pts[:, 0] < NEAR_STRIP_X
        P.append(pts[keep])
        C.append(col[keep])
    allp, allc = np.concatenate(P), np.concatenate(C)
    m = scene_mask(allp, containers)
    pts, col = allp[m], allc[m]
    if len(pts) == 0:
        return [], containers
    lo = pts.min(0) - voxel
    idx = np.floor((pts - lo) / voxel).astype(int)
    labels, n = label_components(idx)
    out = []
    for k in range(n):
        mm = labels == k
        if mm.sum() < min_pts:
            continue
        c = cluster_features(pts[mm], col[mm], containers)
        if is_wall_sliver(c, containers):
            continue
        out.append(c)
    return out, containers


def is_wall_sliver(c, containers):
    """A thin vertical plate as tall as the container walls, standing on a container's outline: a
    wall left over by a container pose that is off by a centimetre or two, not an object."""
    if c["top_z"] < CONTAINER_RIM_Z - 0.03:
        return False
    x, y = c["centroid"][0], c["centroid"][1]
    for name, b in containers.items():
        ox, oy = outer_half(name)
        lx, ly = container_local(b, x, y)
        on_line = abs(lx) < ox + 0.04 and abs(ly) < oy + 0.04 and (abs(abs(lx) - ox) < 0.04 or abs(abs(ly) - oy) < 0.04)
        if on_line and c["len_minor"] <= 0.02:
            return True
        # a somewhat wider blob whose top is the rim itself, centred within 3 cm of the wall line
        # and still thin: a wall segment merged with its rim, not an object (with 12 cm walls many
        # objects reach rim height, so only a thin blob counts; an object standing against the
        # inner wall has its centre at least 5 cm from the outer wall line)
        if c["top_z"] >= CONTAINER_RIM_Z - 0.015 and c["len_minor"] <= 0.035 and abs(lx) < ox + 0.03 and abs(ly) < oy + 0.03 and (abs(abs(lx) - ox) < 0.03 or abs(abs(ly) - oy) < 0.03):
            return True
    return False


def perceive_eih(rgb, depth, R_cam, t_cam, expected_xy, floor, top_hint, containers, radius=0.13, voxel=0.006,
                 edge_filter=None):
    """Dense cluster of the object under the hand camera."""
    pts, col = backproject_rt(depth, rgb, CAMS["agentview"]["f"], CAMS["agentview"]["c"], R_cam, t_cam,
                              edge_filter=edge_filter)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # height above whatever the point stands on: the container floor inside a container, else the desk profile
    inside = np.zeros(len(pts), bool)
    for name, b in containers.items():
        ix, iy = inner_half(name)
        lx, ly = container_local(b, x, y)
        inside |= (np.abs(lx) <= ix) & (np.abs(ly) <= iy)
    base = np.where(inside, CONTAINER_FLOOR_Z, desk_top_z(x))
    keep = (z - base > 0.008) & (z < top_hint + 0.06) & (np.hypot(x - expected_xy[0], y - expected_xy[1]) < radius)
    oz = CONTAINER_OUTER[2]
    for name, b in containers.items():
        ox, oy = outer_half(name)
        ix, iy = inner_half(name)
        lx, ly = container_local(b, x, y)
        in_outer = (np.abs(lx) <= ox + 0.014) & (np.abs(ly) <= oy + 0.014)
        in_inner = (np.abs(lx) <= ix - 0.008) & (np.abs(ly) <= iy - 0.008)
        keep &= ~(in_outer & ~in_inner & (z < oz + 0.025))
    pts, col = pts[keep], col[keep]
    if len(pts) < 30:
        return None
    scene = pts
    lo = pts.min(0) - voxel
    idx = np.floor((pts - lo) / voxel).astype(int)
    labels, n = label_components(idx)
    # the object under the camera: a large component near the expected position (a neighbour
    # half inside the radius is large too, but far off)
    best, best_score = None, 0.0
    for k in range(n):
        m = labels == k
        cnt = int(m.sum())
        if cnt < 30:
            continue
        d = math.hypot(float(pts[m, 0].mean()) - expected_xy[0], float(pts[m, 1].mean()) - expected_xy[1])
        sc = cnt * math.exp(-(d / 0.06) ** 2)
        if sc > best_score:
            best, best_score = m, sc
    if best is None:
        return None
    p, pc = pts[best], col[best]
    blocker, note = None, None
    sp = stack_split(p, expected_xy)
    support = None
    if sp is not None:
        own, tidx, note, support = sp
        if tidx is not None and len(tidx) >= 30:
            blocker = cluster_features(p[tidx], pc[tidx], containers)
            blocker["floor"] = floor if blocker["container"] else float(desk_top_z(blocker["centroid"][0]))
            blocker["dims"] = sorted([blocker["len_major"], blocker["len_minor"], blocker["top_z"] - blocker["floor"]])
        if len(own) >= 30:
            p, pc = p[own], pc[own]
    c = cluster_features(p, pc, containers)
    c["blocker"], c["stack_note"] = blocker, note
    # two objects that touch merge into one component; the grasp point comes from the top
    # surface around the expected position only, so a neighbour does not drag it aside
    near = p[np.hypot(p[:, 0] - expected_xy[0], p[:, 1] - expected_xy[1]) < 0.06]
    if c["len_major"] > 0.12 and len(near) >= 30:
        top_near = float(np.percentile(near[:, 2], 97))
        slab = near[near[:, 2] > top_near - 0.03]
        if len(slab) >= 10:
            c["grasp_xy"] = [float(slab[:, 0].mean()), float(slab[:, 1].mean())]
    c["floor"] = floor if c["container"] else float(desk_top_z(c["centroid"][0]))
    c["dims"] = sorted([c["len_major"], c["len_minor"], c["top_z"] - c["floor"]])
    # what it rests on: only where the plateau analysis found it lying on another object (a ring of
    # lower points around a curved object, a banana's flank or a can's side, is the object itself)
    c["support_z"] = support if (support is not None and support < c["top_z"] - 0.012) else None
    c["scene_pts"] = scene
    return c



def desk_sure(info):
    """A classification of an object seen alone sure enough to name it: close to a reference and clear
    of the next model (the confusions of the references' leave-one-out all had margins under 0.3)."""
    return info["d"] < DESK_SURE_D and info.get("model_margin", 0.0) >= DESK_SURE_MARGIN


def fingers_clear(scene, hx, hy, yaw, z_g, palm=False):
    """The two open fingers, lowered to tool-centre height z_g at (hx, hy) and `yaw`, land on nothing:
    no point of the hand camera's cloud inside either finger's footprint above the fingertip height.
    A finger coming down on a neighbour stops the descent or pushes the pile apart. With `palm`,
    nothing under the hand body's footprint stands above the palm either (a tall neighbour)."""
    if scene is None or len(scene) == 0:
        return True
    u = np.array([-math.sin(yaw), math.cos(yaw)])
    v = np.array([math.cos(yaw), math.sin(yaw)])
    rel = scene[:, :2] - (hx, hy)
    pu, pv = np.abs(rel @ u), np.abs(rel @ v)
    tip = z_g - FINGER_PAD_LOW
    hit = (pv < FINGER_HALF_WIDTH) & (pu > GRASP_MAX_WIDTH / 2 - 0.002) & (pu < GRASP_MAX_WIDTH / 2 + FINGER_THICK) \
        & (scene[:, 2] > tip + 0.006)
    if int(hit.sum()) >= 4:
        return False
    if palm:
        over = (pu < HAND_HALF_W) & (pv < HAND_HALF_T) & (scene[:, 2] > z_g + HAND_BODY_ABOVE_TCP + 0.004)
        if int(over.sum()) >= 15:
            return False
    return True


def grasp_depth(c):
    """How far under the object's top the tool centre point goes. The finger pads reach from
    FINGER_PAD_LOW to FINGER_PAD_HIGH under the tool centre point; they are centred on the object's
    height above what it rests on (the support under a flat object lying on another one, else the
    floor), but never deeper than GRASP_BELOW_TOP under its top (the hand body would land on it), never
    more than 4 mm above its top, and with the fingertips clear of the support and the floor."""
    top, floor = c["top_z"], c["floor"]
    s = c.get("support_z")
    base = s if (s is not None and s > floor) else floor
    h = max(0.0, top - base)
    centred = base + h / 2.0 + (FINGER_PAD_LOW + FINGER_PAD_HIGH) / 2.0
    z = min(centred, top + 0.004)
    z = max(z, top - GRASP_BELOW_TOP, base + FINGER_PAD_LOW + 0.004, floor + GRASP_MIN_ABOVE_FLOOR)
    return max(-0.008, top - z)


# ----------------------------------------------------------------------------------------
# Stacking: what rests on what, inferred from the hand camera's cloud. Objects that touch
# merge into one connected component; the component is cut into plateaus (cells of a 1 cm
# height map that connect where the top height changes by less than STACK_STEP). A plateau
# whose boundary mostly drops onto another plateau sits on it; a tilted plateau (a leaning
# object) whose high end borders a lower plateau leans on it. Whatever rests on the object the
# controller wants comes off first.
# ----------------------------------------------------------------------------------------
STACK_CELL = 0.01
STACK_STEP = 0.02
STACK_MIN_CELLS = 15      # smaller plateaus (a cap, a handle, noise) never count as a resting object
STACK_DROP = 0.015        # a neighbouring cell this much lower is "below" the boundary
STACK_ON_FRAC = 0.4       # share of the boundary that must drop onto one plateau to rest on it
STACK_RAMP = 0.04         # height spread that makes a plateau a tilted object
STACK_ENCLOSED = 0.85     # a lower plateau whose outside is this much one other plateau is that object's hollow (a mug's inside), not something under it
STACK_ON_HULL = 0.4       # share of a resting plateau's cells that must lie over the footprint (convex hull) of the plateau it rests on: a plateau merely beside a lower one (a lying mug's ridge next to its own flank, a tall can next to a short one) is not on it
STACK_LEAN_HULL = 0.15    # the same for a leaning plateau: its high end must reach over the supporting footprint


def plateaus(p, cell=STACK_CELL, step=STACK_STEP, min_cells=STACK_MIN_CELLS):
    """-> (cells {(i,j): [point idx]}, hmax {(i,j): top z}, label {(i,j): plateau}, members [[cells]])
    Plateaus smaller than `min_cells` are merged into the neighbour they touch most."""
    cells = {}
    ij = np.floor(p[:, :2] / cell).astype(int)
    for k in range(len(p)):
        cells.setdefault((int(ij[k, 0]), int(ij[k, 1])), []).append(k)
    hmax = {c: float(p[idx, 2].max()) for c, idx in cells.items()}
    label, members = {}, []
    for c0 in sorted(cells, key=lambda c: -hmax[c]):
        if c0 in label:
            continue
        lab = len(members)
        label[c0] = lab
        stack, mem = [c0], [c0]
        while stack:
            c = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (c[0] + di, c[1] + dj)
                    if nb in cells and nb not in label and abs(hmax[nb] - hmax[c]) < step:
                        label[nb] = lab
                        stack.append(nb)
                        mem.append(nb)
        members.append(mem)
    for lab in sorted(range(len(members)), key=lambda l: len(members[l])):
        mem = members[lab]
        if not mem or len(mem) >= min_cells:
            continue
        touch = {}
        for c in mem:
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (c[0] + di, c[1] + dj)
                    ol = label.get(nb)
                    if ol is not None and ol != lab:
                        touch[ol] = touch.get(ol, 0) + 1
        if touch:
            to = max(touch, key=touch.get)
            for c in mem:
                label[c] = to
            members[to].extend(mem)
            members[lab] = []
    return cells, hmax, label, members


def _hull(points):
    """Convex hull of 2-D points (monotone chain) -> vertices counter-clockwise."""
    pts = sorted(set(tuple(p) for p in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for q in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    upper = []
    for q in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return lower[:-1] + upper[:-1]


def inside_hull_frac(cells_a, cells_b, margin=1.0):
    """Share of the cells `cells_a` that lie within the convex hull of `cells_b` grown by `margin`
    cells (all in cell coordinates); a degenerate hull (a line) counts as its bounding box."""
    if not cells_a or not cells_b:
        return 0.0
    hull = _hull(cells_b)
    if len(hull) < 3:
        xs, ys = [c[0] for c in cells_b], [c[1] for c in cells_b]
        x0, x1, y0, y1 = min(xs) - margin, max(xs) + margin, min(ys) - margin, max(ys) + margin
        return sum(1 for c in cells_a if x0 <= c[0] <= x1 and y0 <= c[1] <= y1) / float(len(cells_a))
    n = 0
    for c in cells_a:
        ok = True
        for i in range(len(hull)):
            a, b = hull[i], hull[(i + 1) % len(hull)]
            ex, ey = b[0] - a[0], b[1] - a[1]
            if (ex * (c[1] - a[1]) - ey * (c[0] - a[0])) / max(math.hypot(ex, ey), 1e-9) < -margin:
                ok = False
                break
        n += ok
    return n / float(len(cells_a))


def stack_relations(p, cells, hmax, label, members):
    """-> {plateau: {"on": {other: share}, "leans": {other: cells}, "top", "bottom", "n_cells"}}
    A plateau enclosed by the one above it (its outside is almost only that plateau) is the hollow
    of one object (a mug's or bowl's inside seen from above), never something under it; and a
    plateau rests or leans on another only when it reaches over that plateau's footprint, so the
    two sides of one lying object (its ridge and its lower flank) or two objects side by side are
    not read as a stack."""
    stats = {}
    for lab, mem in enumerate(members):
        if not mem:
            continue
        zs = p[[k for c in mem for k in cells[c]], 2]
        top = float(np.percentile(zs, 97))
        border, over, high_over, outside = 0, {}, {}, {}
        for c in mem:
            lower, is_border = set(), False
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    nb = (c[0] + di, c[1] + dj)
                    if nb == c:
                        continue
                    ol = label.get(nb)
                    if ol == lab:
                        continue
                    is_border = True
                    outside[ol] = outside.get(ol, 0) + 1      # ol None: empty cell (desk, unseen)
                    if ol is not None and hmax[nb] < hmax[c] - STACK_DROP:
                        lower.add(ol)
            if is_border:
                border += 1
            for ol in lower:
                over[ol] = over.get(ol, 0) + 1
                if hmax[c] > top - 0.03:
                    high_over[ol] = high_over.get(ol, 0) + 1
        stats[lab] = (top, float(np.percentile(zs, 3)), border, over, high_over, outside)

    def enclosed_by(low, high):
        """Is the lower plateau the hollow of the higher one? Either its outside is almost only
        that plateau, or it lies within that plateau's footprint (a mug's inner wall seen from
        above borders the rim on one side and, where the floor is too low to count, nothing on
        the other)."""
        outside = stats[low][5]
        total = sum(outside.values())
        if total > 0 and outside.get(high, 0) >= STACK_ENCLOSED * total:
            return True
        return inside_hull_frac(members[low], members[high], margin=0.5) >= STACK_ENCLOSED

    out = {}
    for lab, (top, bottom, border, over, high_over, outside) in stats.items():
        rel = {"top": top, "bottom": bottom, "n_cells": len(members[lab]), "on": {}, "leans": {}}
        if len(members[lab]) >= STACK_MIN_CELLS and border:
            for ol, cnt in over.items():
                if cnt >= STACK_ON_FRAC * border and not enclosed_by(ol, lab) and inside_hull_frac(members[lab], members[ol]) >= STACK_ON_HULL:
                    rel["on"][ol] = cnt / float(border)
            if top - bottom >= STACK_RAMP:
                for ol, cnt in high_over.items():
                    if cnt >= 3 and ol not in rel["on"] and not enclosed_by(ol, lab) and inside_hull_frac(members[lab], members[ol]) >= STACK_LEAN_HULL:
                        rel["leans"][ol] = cnt
        out[lab] = rel
    return out


def stack_split(p, expected_xy, cell=STACK_CELL):
    """The object at `expected_xy` inside the merged cloud `p` and whatever rests on it.
    -> None (nothing rests on anything here: keep the cloud whole), or
       (own point idx, topper point idx or None, note) where the topper is the largest plateau
       resting or leaning on the object's plateau; own = the object's plateau alone when it is
       itself a topper or carries one."""
    if len(p) < 60:
        return None
    cells, hmax, label, members = plateaus(p, cell)
    live = [lab for lab, mem in enumerate(members) if mem]
    if len(live) < 2:
        return None
    rels = stack_relations(p, cells, hmax, label, members)
    # the plateau under the expected position: the nearest cell, the higher plateau on a tie
    ci, cj = int(math.floor(expected_xy[0] / cell)), int(math.floor(expected_xy[1] / cell))
    target = min(cells, key=lambda c: (max(abs(c[0] - ci), abs(c[1] - cj)), -hmax[c]))
    mine = label[target]
    toppers = [lab for lab in live if lab != mine and (mine in rels[lab]["on"] or mine in rels[lab]["leans"])]
    carried = [lab for lab in live if lab != mine and (lab in rels[mine]["on"] or lab in rels[mine]["leans"])]
    if not toppers and not carried:
        return None
    own = [k for c in members[mine] for k in cells[c]]
    if toppers:
        tl = max(toppers, key=lambda lab: rels[lab]["n_cells"])
        tidx = [k for c in members[tl] for k in cells[c]]
        how = "on" if mine in rels[tl]["on"] else "leaning on"
        note = "%d-cell plateau (top %.3f) %s the %d-cell one (top %.3f)" % (
            rels[tl]["n_cells"], rels[tl]["top"], how, rels[mine]["n_cells"], rels[mine]["top"])
        return own, tidx, note, None
    note = "it is the topper: %d cells (top %.3f) resting on %s" % (
        rels[mine]["n_cells"], rels[mine]["top"], ["%d cells" % rels[l]["n_cells"] for l in carried])
    # what it rests on: the highest top of the plateaus it lies on (a flat object on another one is
    # pinched by its own sides, above that height)
    support = max(rels[l]["top"] for l in carried)
    return own, None, note, support


class Classifier(object):
    def __init__(self, refs, w_dims=8.0):
        self.refs = refs
        self.w_dims = w_dims

    def dist(self, cl, r):
        return hist_dist(cl["hist"], r["hist"]) + self.w_dims * float(np.abs(np.array(cl["dims"]) - np.array(r["dims"])).sum())

    def __call__(self, cl):
        best = {}
        for r in self.refs:
            d = self.dist(cl, r)
            if r["model"] not in best or d < best[r["model"]]:
                best[r["model"]] = d
        return decide(best)


def decide(best):
    """Per-model distances {model: d} -> the decision: nearest model, its class, and the margin
    to the nearest model of another class."""
    if not best:
        return {"model": "?", "cls": "C", "d": 9.0, "class_margin": -1.0, "ranked": [], "all": {}}
    ranked = sorted(best.items(), key=lambda t: t[1])
    model, d0 = ranked[0]
    cls = CLASS_OF[model]
    other = [d for m, d in ranked if CLASS_OF[m] != cls]
    margin = (other[0] - d0) if other else 1.0
    model_margin = (ranked[1][1] - d0) if len(ranked) > 1 else 1.0
    return {"model": model, "cls": cls, "d": d0, "class_margin": margin, "model_margin": model_margin,
            "ranked": ranked[:3], "all": dict(best)}


STATIC_WEIGHT = 0.7     # hand-camera + 0.7 x static distance: either view alone confuses a few look-alikes


def combine(eih_info, static_info, w=STATIC_WEIGHT):
    """Fuse the hand camera's per-model distances with the static cameras' for the same object."""
    if not static_info or not static_info.get("all"):
        return eih_info
    e, s = eih_info["all"], static_info["all"]
    return decide({m: e.get(m, 9.0) + w * s.get(m, 9.0) for m in set(e) | set(s)})


# ----------------------------------------------------------------------------------------
# Controller (generator based: `yield policy` where policy(obs) -> action)
# ----------------------------------------------------------------------------------------
class Controller(object):
    def __init__(self, refs):
        if isinstance(refs, list):
            refs = {"static": refs}
        self.classify = Classifier(refs.get("static", []))
        self.classify_eih = Classifier(refs.get("eih", []), w_dims=12.0)
        rel = refs.get("eih_rel")
        self.eih_R = np.array(rel["R"], float) if rel else np.eye(3)
        self.eih_t = np.array(rel["t"], float) if rel else np.zeros(3)
        self.reset()

    def reset(self):
        self.obs = None
        self.q_home = None
        self.gen = None
        self.last_t, self.policy = None, None
        self.containers = nominal_containers()
        self.request_text = None
        self.request_state = None
        self.drop_slot = {b: 0 for b in CONTAINER_NOMINAL}
        self.skip_list = []
        self.last_perceive_t = -999
        self.last_clusters = []
        self.not_empty_t = {}
        self.carry_hang = 0.0
        self.last_descent_short = False
        self.last_grasp_unreachable = False
        self.stuck_strikes = 0
        self.stuck_count, self.recovering, self.stuck_recovered = 0, False, False
        # what the controller knows about the class containers' contents: the models it put in
        # itself (bottom -> top), and whether an initial pile of unseen objects may still lie
        # under them (the instruction: A objects start in gray, B objects in light-blue)
        self.inventory = {b: [] for b in CLASS_OF_CONTAINER}      # entries {model, slot, xy, t, name[, stuck]}
        # containers that may hold objects never identified: "pile" (an initial pile not yet taken apart),
        # "stuck" (taken apart, some of what no grasp reaches unidentified), "seen" (something unexplained)
        self.unknown = {"gray": "pile", "lightblue": "pile", "blue": None}
        self.seen_t = {}              # when each container was last flagged "seen"
        self.carrying = None          # the sure name of what the fingers hold while carrying it (or None)
        self.loose_seen = []          # (model, x, y, t): identified alone on the desk and not put away since
        self.vanished = []            # (model, container, t): an entry no longer seen where it was put
        self.exploring = {"gray": False, "lightblue": False}
        self.explore_failed = []
        self.desk_items = []          # objects the controller set down on the desk itself
        self.failed_grasps = []       # (x, y, yaw, t) of grasps that came up empty
        self.toppled = []             # (x, y, t) of pushes that put a tall object over
        self.last_comps = {}          # container -> (t, components) of the hand camera's last look into it
        self.ref_map = {}             # container -> (t, height grid) as the controller last left it
        self.acted_t = {}             # container -> step of the controller's last change there
        self.new_in = {}              # container -> (x, y, t) of something new seen there
        self.check_soon = set()       # containers to look into at the next chance (something fell in)
        self.dropped_at = None
        self.last_patrol_t = -10 ** 9
        self.last_full_observe_t = -10 ** 9

    # -- wire entry -------------------------------------------------------------------------
    def act(self, t, obs):
        if self.q_home is None:
            self.set_home(obs)
            self.gen = self.run()
            self.policy = next(self.gen)
        elif t != self.last_t:
            self.obs = obs
            try:
                self.policy = self.gen.send(obs)
            except Exception as exc:  # noqa: BLE001 - one failed behaviour must not end the episode
                log("t=%d behaviour failed (%r); starting over from the observe pose" % (t, exc))
                self.carry_hang = 0.0
                self.gen = self.run()
                try:
                    self.policy = next(self.gen)
                except Exception as exc2:  # noqa: BLE001
                    log("t=%d restart failed too (%r); holding still" % (t, exc2))
                    self.q_home = None          # rebuild everything on the next step
                    self.policy = lambda o: [0.0] * 8
        self.last_t = t
        return self.mark_before_deadline([float(x) for x in self.policy(obs)], obs)

    def mark_before_deadline(self, a, obs):
        """The mark goes out before the deadline whatever the behaviour is in the middle of (a tidy
        grasp the arm got stuck in once carried a delivered request past its deadline): with
        MARK_LAST steps left, delivered once the target was confirmed in red, absent once absence was
        settled, and nothing otherwise: an expired request costs nothing, a wrong mark a point."""
        rs, req = self.request_state, obs.get("request")
        if rs is None or req is None or rs.get("marked") or a[7] != 0 or req["text"] != self.request_text:
            return a
        if req["deadline_t"] - int(obs["t"]) > MARK_LAST:
            return a
        if rs.get("in_red") or (rs.get("placed_red") and rs.get("trust_red")):
            a[7] = 1.0             # confirmed in red, or set down there after a sure identification
        elif rs.get("absent_ready"):
            a[7] = 2.0
        else:
            return a
        rs["marked"] = True
        log("t=%d deadline in %d steps; marking %s now" % (int(obs["t"]), req["deadline_t"] - int(obs["t"]), "delivered" if a[7] == 1.0 else "absent"))
        return a

    def set_home(self, obs):
        """The first observation of an episode defines the home orientation (tool straight
        down, fingers along +y) that every later orientation target is expressed against."""
        self.obs = obs
        self.q_home = np.array(obs["ee_pose"][3:7], float)
        self.R_home = quat_to_mat(self.q_home)

    # -- proprio helpers ------------------------------------------------------------------
    def ee(self):
        return np.array(self.obs["ee_pose"][:3], float)

    def yaw(self, obs=None):
        R = quat_to_mat(np.array((obs or self.obs)["ee_pose"][3:7], float))
        y_axis = R[:, 1]
        return wrap(math.atan2(y_axis[1], y_axis[0]) - math.pi / 2)

    def orient_error(self, yaw, obs=None, tilt=None):
        R_cur = quat_to_mat(np.array((obs or self.obs)["ee_pose"][3:7], float))
        R_des = rot_z(yaw) @ self.R_home
        if is_tilted(tilt):
            R_des = tilt_rot(yaw, tilt) @ R_des
        return rotvec(R_des @ R_cur.T)

    def width(self):
        return float(self.obs["gripper_width"])

    def t(self):
        return int(self.obs["t"])

    def request(self):
        return self.obs.get("request")

    # -- motion primitives --------------------------------------------------------------------
    def servo(self, pos, yaw, grip, mark=0, step_max=POS_STEP_MAX, tilt=None):
        pos = np.asarray(pos, float)

        def policy(o):
            err = pos - np.array(o["ee_pose"][:3], float)
            d = np.clip(err, -step_max, step_max)
            r = np.clip(self.orient_error(yaw, o, tilt), -ROT_STEP_MAX, ROT_STEP_MAX)
            return [d[0], d[1], d[2], r[0], r[1], r[2], grip, mark]
        return policy

    def goto(self, pos, yaw=0.0, grip=0.0, tol=0.01, yaw_tol=0.04, max_steps=60, mark=0, step_max=POS_STEP_MAX, tilt=None):
        pos = np.asarray(pos, float)
        for i in range(max_steps):
            err = pos - self.ee()
            if np.linalg.norm(err) < tol and np.linalg.norm(self.orient_error(yaw, tilt=tilt)) < yaw_tol:
                self.stuck_count = 0
                return True
            yield self.servo(pos, yaw, grip, mark, step_max, tilt)
            mark = 0
        # An arm that keeps ending far from where it was sent has folded into a configuration the
        # Cartesian servo cannot leave (a joint at its limit): a few straight lifts to the
        # observe height unfold it. Without this, every later move of the episode failed too.
        if np.linalg.norm(pos - self.ee()) > STUCK_ERR and not self.recovering:
            self.stuck_count += 1
            if self.stuck_count >= STUCK_AFTER:
                log("t=%d arm stuck (%d moves ended far off); recovering through the observe pose" % (self.t(), self.stuck_count))
                self.recovering = True
                try:
                    cur = self.ee()
                    yield from self.goto((cur[0], cur[1], 0.45), 0.0, grip=grip, tol=0.03, yaw_tol=0.3, max_steps=40)
                    yield from self.goto(OBSERVE_POS, 0.0, grip=grip, tol=0.02, yaw_tol=0.1, max_steps=80)
                finally:
                    self.recovering = False
                    self.stuck_count = 0
                    self.stuck_recovered = True       # the current grasp gives up (try_grasps, pick)
        return False

    def hold(self, steps, grip, mark=0, yaw=None, tilt=None):
        pos = self.ee()
        yaw = self.yaw() if yaw is None else yaw
        for i in range(steps):
            yield self.servo(pos, yaw, grip, mark, tilt=tilt)
            mark = 0

    def gripper_closed_on_object(self):
        # fully shut reads 0.0000-0.0005; a power drill held by its trigger reads 4 mm; fully open
        # reads 0.0800, and a large clamp pinched across its jaws reads 0.077-0.078 (four lifts of
        # one clamp were dropped again at the 0.077 bound)
        return 0.002 < self.width() < 0.0792

    def travel_z(self):
        """Carry height: the held object's bottom clears the container rims."""
        return max(TRAVEL_Z, CONTAINER_RIM_Z + 0.04 + self.carry_hang)

    def refine_xy(self, xy, anchor_xy, floor, top_z, yaw=0.0):
        """Hovering 10 cm above the object with the hand camera: re-measure the top centroid
        around the planned anchor and shift the grasp point by how far it moved; also measure
        the top within the strip the fingers sweep (the finger span along the closing axis, a
        finger's width across). Two touching objects merge into one component whose top is the
        taller one's, so the component's top drags the grasp point toward the neighbour and sets
        the fingers above the object; the strip, not the point under the tool centre, is what
        the fingers meet, so a hollow object's rim (the pitcher's, under a grasp planned over its
        opening) or an object the hand sits beside along the finger axis keeps its real top.
        -> (xy, local top z or None)"""
        try:
            R_cam, t_cam = self.cam_pose()
            rgb = decode_png(self.obs["rgb"]["eye_in_hand"], "RGB")
            dep = decode_png(self.obs["depth"]["eye_in_hand"])
            d = perceive_eih(rgb, dep, R_cam, t_cam, anchor_xy, floor, top_z, self.containers, radius=REFINE_RADIUS)
        except Exception as exc:  # noqa: BLE001
            log("refine failed: %r" % exc)
            return xy, None
        if d is None:
            return xy, None
        p = d["pts"]
        near = p[np.hypot(p[:, 0] - anchor_xy[0], p[:, 1] - anchor_xy[1]) < REFINE_NEAR]
        if len(near) < 15:
            return xy, None
        slab = near[near[:, 2] > float(np.percentile(near[:, 2], 97)) - 0.02]
        if len(slab) < 15:
            return xy, None
        cx, cy = slab[:, :2].mean(0)
        dx, dy = float(cx - anchor_xy[0]), float(cy - anchor_xy[1])
        if math.hypot(dx, dy) > REFINE_MAX_SHIFT:
            log("t=%d refine: shift %.3f too large, ignored" % (self.t(), math.hypot(dx, dy)))
            dx = dy = 0.0
        g = (xy[0] + dx, xy[1] + dy)
        u, v = np.array([-math.sin(yaw), math.cos(yaw)]), np.array([math.cos(yaw), math.sin(yaw)])
        rel = p[:, :2] - g
        under = p[(np.abs(rel @ u) < GRASP_MAX_WIDTH / 2) & (np.abs(rel @ v) < TOP_UNDER_RADIUS)]
        # the highest few points, not a percentile: the strip over a hollow pitcher or a thin tall
        # box is mostly its inside or its sides, and the 97th percentile sat 5 cm under the rim
        top_local = float(np.percentile(under[:, 2], 99.5)) if len(under) >= 10 else None
        log("t=%d refine: shift (%+.3f,%+.3f) from %d top points, top under the grasp %s (planned %.3f)%s" % (
            self.t(), dx, dy, len(slab), "%.3f" % top_local if top_local is not None else "-", top_z,
            "; stack: " + d["stack_note"] if d.get("stack_note") else ""))
        return g, top_local

    def palm_top(self, xy, yaw):
        """Highest point under the hand body's footprint at (xy, yaw) in the current hand-camera view
        (container walls left out: the hand-fit checks keep the hand off them). -> z or None"""
        try:
            R_cam, t_cam = self.cam_pose()
            rgb = decode_png(self.obs["rgb"]["eye_in_hand"], "RGB")
            dep = decode_png(self.obs["depth"]["eye_in_hand"])
            pts, _ = backproject_rt(dep, rgb, None, None, R_cam, t_cam)
        except Exception as exc:  # noqa: BLE001
            log("palm check failed: %r" % exc)
            return None
        u, v = np.array([-math.sin(yaw), math.cos(yaw)]), np.array([math.cos(yaw), math.sin(yaw)])
        rel = pts[:, :2] - np.asarray(xy)
        m = (np.abs(rel @ u) < HAND_HALF_W + 0.005) & (np.abs(rel @ v) < HAND_HALF_T + 0.005)
        for name, b in self.containers.items():
            ox, oy = outer_half(name)
            ix, iy = inner_half(name)
            lx, ly = container_local(b, pts[:, 0], pts[:, 1])
            wall = (np.abs(lx) <= ox + 0.014) & (np.abs(ly) <= oy + 0.014) & ~((np.abs(lx) <= ix - 0.008) & (np.abs(ly) <= iy - 0.008))
            m &= ~(wall & (pts[:, 2] < CONTAINER_OUTER[2] + 0.025))
        # the hand camera sees its own open fingers around the tool centre point: only what lies
        # well below them counts
        m &= pts[:, 2] < float(self.ee()[2]) - 0.045
        z = pts[m, 2]
        if len(z) < 20:
            return None
        return float(np.percentile(z, 98))

    def pick(self, xy, yaw, top_z, floor, below_top=GRASP_BELOW_TOP, anchor_xy=None, tilt=None, keep_top=False):
        """Top-down pinch grasp at `yaw`, fingers `below_top` under the object's top (a rim grasp
        uses a shallow depth). The grasp point is refined from the hand camera 10 cm above the
        object before the final descent. Returns True when something is held after lifting."""
        x, y = xy
        yaw, tilt = yaw_branch((x, y), wrap_half(yaw), tilt)
        z_pre = max(top_z + 0.10, floor + 0.12)
        self.carry_hang = max(0.0, (top_z - floor) - below_top)     # how far the object hangs below the fingertips
        z_travel = max(self.travel_z(), z_pre)
        yield from self.goto((x, y, z_travel), yaw, grip=0.0, tol=0.015)
        yield from self.goto((x, y, z_pre), yaw, grip=0.0, tol=0.008, max_steps=30)
        if self.stuck_recovered:
            return False            # the arm folded up on the way and is back at the observe pose
        yield from self.hold(1, grip=0.0)
        (x, y), top_local = self.refine_xy((x, y), anchor_xy or (x, y), floor, top_z, yaw=yaw)
        if top_local is not None and top_local < top_z - TOP_DROP_MIN and not keep_top:
            top_z = top_local       # the planned top was a taller neighbour's
            self.carry_hang = max(0.0, (top_z - floor) - below_top)
        z_g = max(top_z - below_top, floor + GRASP_MIN_ABOVE_FLOOR)
        if not is_tilted(tilt) and container_of_xy(self.containers, x, y, shrink=-0.01) is None:
            palm = self.palm_top((x, y), yaw)
            if palm is not None and z_g + HAND_BODY_ABOVE_TCP < palm + PALM_CLEAR:
                # the palm would come down on something higher under its footprint (a bottle's cap
                # beside a grasp planned over its shoulder, a taller neighbour): the hand stops there
                log("t=%d palm clearance: grasp z %.3f -> %.3f (%.3f under the palm)" % (
                    self.t(), z_g, palm + PALM_CLEAR - HAND_BODY_ABOVE_TCP, palm))
                z_g = palm + PALM_CLEAR - HAND_BODY_ABOVE_TCP
                if z_g - FINGER_PAD_LOW > top_z - 0.008:
                    # the fingertips would stay above the object: something taller stands under the palm
                    log("t=%d palm clearance leaves the fingers above the object (top %.3f): not going down" % (self.t(), top_z))
                    self.last_descent_short = True
                    yield from self.goto((x, y, z_travel), yaw, grip=0.0, tol=0.015, max_steps=30, step_max=0.03)
                    return False
        self.last_grasp = (x, y, z_g, yaw)
        if is_tilted(tilt):
            # lean the hand away from the near wall above the rim, then go in along the fingers.
            # The tool centre point sits FINGER_BELOW_TCP above the fingertips, so a tilted hand
            # puts its fingertips 0.8-1.6 cm to the side of the TCP: the TCP is sent that much the
            # other way, so that the fingertips, not the TCP, land on the object (uncompensated,
            # tilted grasps lifted 2 of 16 in the episode logs: one finger came down on the object)
            a = approach_dir(yaw, tilt)
            log("t=%d tilted grasp: plan (%.3f,%.3f) anchor (%.3f,%.3f) yaw %.2f tilt %s approach (%.2f,%.2f,%.2f) -> tcp (%.3f,%.3f) z %.3f" % (
                self.t(), x, y, (anchor_xy or (x, y))[0], (anchor_xy or (x, y))[1], yaw, fmt_tilt(tilt), a[0], a[1], a[2],
                x - FINGER_BELOW_TCP * a[0], y - FINGER_BELOW_TCP * a[1], z_g))
            x, y = x - FINGER_BELOW_TCP * a[0], y - FINGER_BELOW_TCP * a[1]
            self.last_grasp = (x, y, z_g, yaw)
            back = max(0.06, (CONTAINER_RIM_Z + 0.06 - z_g) / max(-a[2], 0.3))
            p_pre = np.array([x, y, z_g]) - a * back
            yield from self.goto(p_pre, yaw, grip=0.0, tol=0.01, yaw_tol=0.06, max_steps=40, tilt=tilt)
            yield from self.goto((x, y, z_g), yaw, grip=0.0, tol=0.006, yaw_tol=0.06, max_steps=30, step_max=0.02, tilt=tilt)
            self.last_grasp_actual = [float(v) for v in self.ee()]
            self.last_descent_short = float(np.linalg.norm(self.ee() - np.array([x, y, z_g]))) > DESCENT_SHORT
            if self.last_descent_short:
                log("t=%d descent stopped %.3f m short of the grasp point; not closing" % (self.t(), float(np.linalg.norm(self.ee() - np.array([x, y, z_g])))))
                yield from self.goto(p_pre, yaw, grip=0.0, tol=0.015, yaw_tol=0.08, max_steps=30, step_max=0.03, tilt=tilt)
                yield from self.goto((p_pre[0], p_pre[1], z_travel), yaw, grip=0.0, tol=0.015, max_steps=30, step_max=0.03)
                return False
            yield from self.close_fingers(yaw=yaw, tilt=tilt)
            yield from self.goto(p_pre, yaw, grip=1.0, tol=0.015, yaw_tol=0.08, max_steps=30, step_max=0.015, tilt=tilt)
            yield from self.goto((p_pre[0], p_pre[1], z_travel), yaw, grip=1.0, tol=0.015, max_steps=30, step_max=0.03)
        else:
            yield from self.goto((x, y, z_g), yaw, grip=0.0, tol=0.006, max_steps=25, step_max=0.02)
            self.last_grasp_actual = [float(v) for v in self.ee()]
            self.last_descent_short = float(np.linalg.norm(self.ee() - np.array([x, y, z_g]))) > DESCENT_SHORT
            if self.last_descent_short:
                log("t=%d descent stopped %.3f m short of the grasp point; not closing" % (self.t(), float(np.linalg.norm(self.ee() - np.array([x, y, z_g])))))
                b_in = container_of_xy(self.containers, x, y, shrink=-0.01)
                if b_in is not None:
                    self.acted_t[b_in] = self.t()
                yield from self.goto((x, y, z_travel), yaw, grip=0.0, tol=0.015, max_steps=30, step_max=0.03)
                return False
            yield from self.close_fingers()
            # out of the pile slowly first: neighbours pressing on it pull a shallow pinch apart
            ee = self.ee()
            yield from self.goto((x, y, float(ee[2]) + 0.05), yaw, grip=1.0, tol=0.006, max_steps=8, step_max=0.012)
            yield from self.goto((x, y, z_travel), yaw, grip=1.0, tol=0.015, max_steps=30, step_max=0.03)
        yield from self.hold(2, grip=1.0)
        b_in = container_of_xy(self.containers, x, y, shrink=-0.01)
        if b_in is not None:
            self.acted_t[b_in] = self.t()        # the controller changed what is in there
        # judged after the lift only: a width read right after the close command, while the
        # fingers were still travelling to a wide object, called a pitcher in the hand "nothing",
        # and the object went back down from the carry height
        return self.gripper_closed_on_object()

    def close_fingers(self, yaw=None, tilt=None):
        """Command the fingers shut and wait until they stop moving (at least three, at most
        GRIP_SETTLE_MAX steps): the drive takes several steps to meet a wide object."""
        last = self.width()
        for i in range(GRIP_SETTLE_MAX):
            yield from self.hold(1, grip=1.0, yaw=yaw, tilt=tilt)
            w = self.width()
            if i >= 2 and abs(w - last) < 0.0015:
                break
            last = w


    def carry_to(self, pos, yaw, tol=0.015, max_steps=60, step_max=CARRY_STEP):
        """Move with the object in the fingers, watching them: when they shut (the object slipped
        out), stop there and note where it fell. -> True while it is still held."""
        pos = np.asarray(pos, float)
        shut = 0
        for i in range(max_steps):
            if self.width() < HELD_MIN_WIDTH:
                shut += 1
                if shut >= 2:
                    ee = self.ee()
                    self.dropped_at = (float(ee[0]), float(ee[1]), self.t())
                    b = container_of_xy(self.containers, ee[0], ee[1], shrink=-0.03)
                    log("t=%d the object slipped out of the fingers over %s at (%.2f,%.2f)" % (self.t(), b or "the desk", ee[0], ee[1]))
                    if self.carrying is not None:
                        # what fell is known: it stays on the books where it fell (a dropped pair of
                        # scissors in blue, unrecorded, once let "no such object" go out wrongly)
                        if b in self.inventory:
                            self.inventory[b].append({"model": self.carrying, "slot": None, "xy": (float(ee[0]), float(ee[1])),
                                                      "t": self.t(), "name": b, "dropped": True})
                        elif b is None:
                            self.loose_seen.append((self.carrying, float(ee[0]), float(ee[1]), self.t()))
                    if b in self.unknown and not self.unknown[b]:
                        self.unknown[b] = "seen"
                        self.seen_t[b] = self.t()
                    if b in CLASS_OF_CONTAINER or b == "red":
                        self.check_soon.add(b)
                    return False
            else:
                shut = 0
            if np.linalg.norm(pos - self.ee()) < tol and np.linalg.norm(self.orient_error(yaw)) < 0.04:
                return True
            # start gently: a full-speed first step jerks a light object out of a shallow pinch
            yield self.servo(pos, yaw, 1.0, 0, min(step_max, 0.015 + 0.01 * i))
        return self.width() >= HELD_MIN_WIDTH

    def place_on_desk(self, xy, yaw=0.0):
        """Set the held object down on the desk, its bottom ~2 cm above the top (tall objects
        dropped from higher topple)."""
        x, y = xy
        yaw = yaw_branch((x, y), yaw)[0]
        ok = yield from self.carry_to((x, y, self.travel_z()), yaw)
        if ok:
            yield from self.goto((x, y, 0.02 + self.carry_hang), yaw, grip=1.0, tol=0.01, max_steps=30, step_max=0.03)
        if not ok:
            self.carry_hang = 0.0
            yield from self.goto((self.ee()[0], self.ee()[1], TRAVEL_Z), yaw, grip=0.0, tol=0.03, max_steps=20)
            return False
        still = self.width() > HELD_MIN_WIDTH
        yield from self.hold(4, grip=0.0)
        self.carry_hang = 0.0
        yield from self.goto((x, y, TRAVEL_Z), yaw, grip=0.0, tol=0.02, max_steps=30)
        if not still:
            log("t=%d the object was lost on the way (fingers shut before the release)" % self.t())
        return still

    def observe(self, settle=2, sweep=False):
        """Fused perception from the observe pose. The hand camera there looks straight down on the
        strip between the robot and the containers, which the container walls hide from the static
        cameras, except its two ends; `sweep` also looks at those from above (a low object at the
        +y end is invisible to every camera otherwise)."""
        extra = []
        if not sweep and float(self.ee()[2]) > QUICK_OBSERVE_Z and self.t() - self.last_full_observe_t < FULL_OBSERVE_EVERY:
            # the hand is up out of the way: look with the static cameras from where it is (going to the
            # observe pose every cycle cost a fifth of the episode); the strip in front of the base, which
            # only the hand camera sees, is looked at every FULL_OBSERVE_EVERY steps
            yield from self.hold(settle, grip=0.0)
            imgs = {cam: (decode_png(self.obs["rgb"][cam], "RGB"), decode_png(self.obs["depth"][cam])) for cam in CAMS}
            clusters, containers = perceive(imgs, containers=self.containers)
            near = [c for c in self.last_clusters if c["container"] is None and c["centroid"][0] < NEAR_STRIP_X]
            clusters += [c for c in near if not any(math.hypot(c["centroid"][0] - d["centroid"][0], c["centroid"][1] - d["centroid"][1]) < 0.05 for d in clusters)]
            for c in clusters:
                if "cls_info" not in c:
                    c["cls_info"] = self.classify(c) if (c["container"] is None or c["n"] < 600) else None
            self.last_perceive_t = self.t()
            self.last_clusters = clusters
            self.maybe_empty = [b for b in self.inventory if sum(c["n"] for c in clusters if c["container"] == b) < 15
                                and (self.inventory[b] or self.unknown.get(b))
                                and self.t() - self.not_empty_t.get(b, -10 ** 9) > 300]
            return clusters
        self.last_full_observe_t = self.t()
        if sweep:
            for y_end in (0.47, -0.47):
                yield from self.goto((0.21, y_end, 0.42), 0.0, grip=0.0, tol=0.02, max_steps=40)
                yield from self.hold(2, grip=0.0)
                extra.append((decode_png(self.obs["rgb"]["eye_in_hand"], "RGB"), decode_png(self.obs["depth"]["eye_in_hand"])) + self.cam_pose())
            self.last_sweep_t = self.t()
        yield from self.goto(OBSERVE_POS, 0.0, grip=0.0, tol=0.02)
        yield from self.hold(settle, grip=0.0)
        imgs = {cam: (decode_png(self.obs["rgb"][cam], "RGB"), decode_png(self.obs["depth"][cam])) for cam in CAMS}
        eih = (decode_png(self.obs["rgb"]["eye_in_hand"], "RGB"), decode_png(self.obs["depth"]["eye_in_hand"]))
        clusters, containers = perceive(imgs, extra=[(eih[0], eih[1]) + self.cam_pose()] + extra)
        self.containers = containers
        for c in clusters:
            c["cls_info"] = self.classify(c) if (c["container"] is None or c["n"] < 600) else None
        self.last_perceive_t = self.t()
        self.last_clusters = clusters
        # a container that looks empty to the static cameras may only hide a low object behind
        # its 12 cm walls: the hand camera checks from above before the inventory is dropped
        self.maybe_empty = [b for b in self.inventory if sum(c["n"] for c in clusters if c["container"] == b) < 15
                            and (self.inventory[b] or self.unknown.get(b))
                            and self.t() - self.not_empty_t.get(b, -10 ** 9) > 300]
        self.dump("observe", imgs["agentview"][0], clusters)
        return clusters

    def container_cloud(self, b):
        """Points inside container b's interior above its floor, from the current hand-camera frame."""
        cb = self.containers[b]
        R_cam, t_cam = self.cam_pose()
        rgb = decode_png(self.obs["rgb"]["eye_in_hand"], "RGB")
        dep = decode_png(self.obs["depth"]["eye_in_hand"])
        pts, _ = backproject_rt(dep, rgb, None, None, R_cam, t_cam)
        lx, ly = container_local(cb, pts[:, 0], pts[:, 1])
        ix, iy = inner_half(b)
        keep = (np.abs(lx) < ix - 0.008) & (np.abs(ly) < iy - 0.008) \
            & (pts[:, 2] > CONTAINER_FLOOR_Z + 0.012) & (pts[:, 2] < CONTAINER_RIM_Z + 0.1)
        return pts[keep], np.stack([lx[keep], ly[keep]], 1)

    @staticmethod
    def height_grid(local_xy, z, cell=CHANGE_CELL):
        g = {}
        ij = np.floor(local_xy / cell).astype(int)
        for (i, j), zz in zip(map(tuple, ij), z):
            if zz > g.get((i, j), -1.0):
                g[(i, j)] = float(zz)
        return g


    def changed_cells(self, b, grid):
        """Cells of container b higher than in the reference by CHANGE_RISE, when the reference is
        newer than the controller's own last action there -> list of (i, j) or None (no reference)."""
        ref = self.ref_map.get(b)
        if ref is None or ref[0] < self.acted_t.get(b, -1):
            return None
        rg = ref[1]
        out = []
        for c, z in grid.items():
            z0 = rg.get(c, CONTAINER_FLOOR_Z)
            if z - z0 > CHANGE_RISE:
                # a cell next to one that was this high before is the same old edge seen a little shifted
                nb = max([rg.get((c[0] + di, c[1] + dj), CONTAINER_FLOOR_Z) for di in (-1, 0, 1) for dj in (-1, 0, 1)])
                if z - nb > CHANGE_RISE:
                    out.append(c)
        return out

    def container_components(self, b):
        """Everything standing on container b's floor, seen from straight above with the hand
        camera (the static cameras miss low objects behind the walls): pseudo-clusters to inspect."""
        cb = self.containers[b]
        yield from self.goto((cb[0], cb[1], CONTAINER_RIM_Z + 0.24), 0.0, grip=0.0, tol=0.01, max_steps=40)
        yield from self.hold(2, grip=0.0)
        R_cam, t_cam = self.cam_pose()
        rgb = decode_png(self.obs["rgb"]["eye_in_hand"], "RGB")
        dep = decode_png(self.obs["depth"]["eye_in_hand"])
        pts, _ = backproject_rt(dep, rgb, None, None, R_cam, t_cam)
        lx, ly = container_local(cb, pts[:, 0], pts[:, 1])
        ix, iy = inner_half(b)
        keep = (np.abs(lx) < ix - 0.008) & (np.abs(ly) < iy - 0.008) \
            & (pts[:, 2] > CONTAINER_FLOOR_Z + 0.012) & (pts[:, 2] < CONTAINER_RIM_Z + 0.1)
        q = pts[keep]
        out = []
        if len(q) < 40:
            self.last_comps[b] = (self.t(), out)
            self.ref_map[b] = (self.t(), {})
            return out
        lo = q.min(0) - 0.008
        labels, n = label_components(np.floor((q - lo) / 0.008).astype(int))
        for k in range(n):
            m = labels == k
            if m.sum() < 40:
                continue
            top = float(np.percentile(q[m, 2], 97))
            slab = q[m][q[m][:, 2] > top - 0.02]
            gx, gy = (slab[:, :2].mean(0) if len(slab) >= 5 else q[m, :2].mean(0))
            out.append({"grasp_xy": [float(gx), float(gy)], "top_z": top, "floor": CONTAINER_FLOOR_Z, "container": b,
                        "n": int(m.sum()), "centroid": [float(gx), float(gy), float(q[m, 2].mean())], "cls_info": None,
                        "pts": q[m]})
        out = sorted(out, key=lambda c: -c["n"])
        self.last_comps[b] = (self.t(), out)
        # what changed since the controller last left this container: the world drops objects in
        try:
            loc = np.stack(container_local(cb, q[:, 0], q[:, 1]), 1)
            grid = self.height_grid(loc, q[:, 2])
            cells = self.changed_cells(b, grid)
            if cells is not None and len(cells) >= CHANGE_MIN_CELLS:
                cx = float(np.mean([c[0] for c in cells]) + 0.5) * CHANGE_CELL
                cy = float(np.mean([c[1] for c in cells]) + 0.5) * CHANGE_CELL
                wx, wy = container_point(cb, cx, cy)
                fresh = [e for e in self.inventory.get(b, []) if self.t() - e["t"] < 800
                         and math.hypot(e["xy"][0] - wx, e["xy"][1] - wy) < 0.13]
                if fresh:
                    e = min(fresh, key=lambda e: math.hypot(e["xy"][0] - wx, e["xy"][1] - wy))
                    log("t=%d %s: %d cells higher around (%.2f,%.2f): the %s placed there fell over; entry moved" % (
                        self.t(), b, len(cells), wx, wy, e["model"]))
                    e["xy"] = (float(wx), float(wy))
                else:
                    self.new_in[b] = (float(wx), float(wy), self.t())
                    if not self.unknown.get(b):
                        self.unknown[b] = "seen"
                        self.seen_t[b] = self.t()
                    log("t=%d %s: %d cells higher than when last left (around (%.2f,%.2f)): something new" % (self.t(), b, len(cells), wx, wy))
            self.ref_map[b] = (self.t(), grid)
        except Exception as exc:  # noqa: BLE001
            log("change check of %s failed: %r" % (b, exc))
        return out

    def verify_empty(self, b):
        """Look into container b from above with the hand camera; True (and the inventory
        dropped) when nothing stands on its floor."""
        cb = self.containers[b]
        yield from self.goto((cb[0], cb[1], CONTAINER_RIM_Z + 0.24), 0.0, grip=0.0, tol=0.01, max_steps=40)
        yield from self.hold(2, grip=0.0)
        R_cam, t_cam = self.cam_pose()
        rgb = decode_png(self.obs["rgb"]["eye_in_hand"], "RGB")
        dep = decode_png(self.obs["depth"]["eye_in_hand"])
        pts, _ = backproject_rt(dep, rgb, None, None, R_cam, t_cam)
        lx, ly = container_local(cb, pts[:, 0], pts[:, 1])
        ix, iy = inner_half(b)
        inside = (np.abs(lx) < ix - 0.01) & (np.abs(ly) < iy - 0.01) \
            & (pts[:, 2] > CONTAINER_FLOOR_Z + 0.012) & (pts[:, 2] < CONTAINER_RIM_Z)
        n = int(inside.sum())
        if n < 40:
            log("t=%d %s container is empty" % (self.t(), b))
            self.inventory[b] = []
            self.vanished = [v for v in self.vanished if v[1] != b]     # emptied by the worker with the rest
            self.unknown[b] = None
            if b in self.exploring:
                self.exploring[b] = False
            return True
        log("t=%d %s container only looked empty (%d points on its floor)" % (self.t(), b, n))
        self.not_empty_t[b] = self.t()
        return False

    def dump(self, tag, rgb, clusters):
        if not LOG:
            return
        try:
            os.makedirs(LOG, exist_ok=True)
            Image.fromarray(rgb).save(os.path.join(LOG, "t%06d_%s.png" % (self.t(), tag)))
            rows = []
            for c in clusters:
                info = c.get("cls_info")
                rows.append({"n": c["n"], "centroid": [round(v, 3) for v in c["centroid"]], "top_z": round(c["top_z"], 3),
                             "container": c["container"], "dims": [round(v, 3) for v in c["dims"]], "yaw": round(c["yaw"], 2),
                             "model": info["model"] if info else None, "d": round(info["d"], 3) if info else None,
                             "margin": round(info["class_margin"], 3) if info else None})
            with open(os.path.join(LOG, "t%06d_%s.json" % (self.t(), tag)), "w") as f:
                json.dump({"t": self.t(), "request": self.request(), "containers": {k: [round(v, 3) for v in b] for k, b in self.containers.items()}, "clusters": rows}, f)
        except Exception as exc:  # noqa: BLE001
            log("dump failed: %r" % exc)

    def cam_pose(self):
        q = np.array(self.obs["ee_pose"][3:7], float)
        R_e = quat_to_mat(q)
        t_e = self.ee()
        return R_e @ self.eih_R, t_e + R_e @ self.eih_t

    def inspect(self, c, radius=0.13, hover=HOVER_ABOVE_TOP, offset=(0.0, 0.0)):
        """Hover over a cluster and classify it from the hand camera (dense cloud)."""
        x, y = c["grasp_xy"]
        hx, hy = x + offset[0], y + offset[1]
        yield from self.goto((hx, hy, max(TRAVEL_Z, c["top_z"] + hover)), 0.0, grip=0.0, tol=0.015)
        yield from self.goto((hx, hy, c["top_z"] + hover), 0.0, grip=0.0, tol=0.006, yaw_tol=0.03, max_steps=40)
        yield from self.hold(3, grip=0.0)
        R_cam, t_cam = self.cam_pose()
        rgb = decode_png(self.obs["rgb"]["eye_in_hand"], "RGB")
        dep = decode_png(self.obs["depth"]["eye_in_hand"])
        d = perceive_eih(rgb, dep, R_cam, t_cam, (x, y), c["floor"], c["top_z"], self.containers, radius=radius)
        if d is None and EDGE_FILTER_EIH:
            # a thin object (scissors, a marker, a flat clamp) is mostly silhouette: the depth-edge
            # filter that keeps grasp geometry clean leaves too little of it
            d = perceive_eih(rgb, dep, R_cam, t_cam, (x, y), c["floor"], c["top_z"], self.containers, radius=radius,
                             edge_filter=False)
            if d is not None:
                log("t=%d inspect: found only without the depth-edge filter" % self.t())
        if d is None:
            log("t=%d inspect: nothing under the hand camera" % self.t())
            self.dump("inspect_none", rgb, [])
            return None
        d["cls_info"] = clamp_by_size(combine(self.classify_eih(d), c.get("cls_info")), d["dims"])
        d["container"] = c["container"]
        self.note_sighting(d)
        if d.get("stack_note"):
            log("t=%d stack: %s" % (self.t(), d["stack_note"]))
        if d.get("blocker"):
            blk = d["blocker"]
            blk["cls_info"] = clamp_by_size(self.classify_eih(blk), blk["dims"])
            blk["container"] = c["container"]
            log("t=%d stack: %s (d=%.2f) rests on the inspected %s at (%.2f,%.2f) top=%.3f; it comes off first" % (
                self.t(), blk["cls_info"]["model"], blk["cls_info"]["d"], d["cls_info"]["model"],
                blk["grasp_xy"][0], blk["grasp_xy"][1], blk["top_z"]))
        self.dump("inspect", rgb, [d] + ([d["blocker"]] if d.get("blocker") else []))
        return d

    def note_sighting(self, d):
        """A hand-camera view inside a container that names the requested model, however poor the
        view: while it stands, the model is not declared absent (one such view, d=0.66 in a pile,
        whose grasp then failed, was followed by a wrong "no such object"). It is withdrawn only when
        the object there is taken out and identified alone as something else."""
        rs = self.request_state
        if rs is None or d is None or d.get("container") is None or rs.get("in_red"):
            return
        if is_model(d["cls_info"], rs["model"], 0.3):
            rs.setdefault("sightings", []).append((d["container"], float(d["grasp_xy"][0]), float(d["grasp_xy"][1])))
            log("t=%d sighting: %s in %s at (%.2f,%.2f) (d=%.2f)" % (self.t(), d["cls_info"]["model"], d["container"],
                d["grasp_xy"][0], d["grasp_xy"][1], d["cls_info"]["d"]))

    def withdraw_sighting(self, b, xy, info):
        rs = self.request_state
        if rs is None or not rs.get("sightings") or xy is None or info is None:
            return
        if desk_sure(info) and not is_model(info, rs["model"]):
            n = len(rs["sightings"])
            rs["sightings"] = [s for s in rs["sightings"] if not (s[0] == b and math.hypot(s[1] - xy[0], s[2] - xy[1]) < 0.07)]
            if len(rs["sightings"]) < n:
                log("t=%d sighting in %s withdrawn: it is a %s" % (self.t(), b, info["model"]))

    # -- grasp planning -------------------------------------------------------------------------
    def failed_here(self, x, y, yaw):
        """How many grasps at about this spot and yaw failed in the last FAILED_MEMORY steps."""
        n = 0
        for (fx, fy, fyaw, ft) in self.failed_grasps:
            if self.t() - ft < FAILED_MEMORY and math.hypot(fx - x, fy - y) < 0.05 and abs(wrap_half(fyaw - yaw)) < 0.35:
                n += 1
        return n

    def choose_yaw(self, c, inside=None, band=None, below_top=GRASP_BELOW_TOP):
        """-> (xy, yaw, tilt) for a pinch of cluster c, or None: fingers across its narrowest side, the
        grasped width under the opening, and the hand clear of every container wall at the grasp
        height (a hand point above a rim is free; where the straight-down hand hits a wall the hand
        leans away from it, TILTS). The hand need not be centred on the object along the finger
        axis: the open fingers are 8 cm apart and the first one to touch pushes the object to the
        middle as they close, so the hand may sit up to ~3 cm off the object. `band` = the points
        the fingers will close on (default: the top rim)."""
        x, y = c["grasp_xy"]
        z_g = max(c["top_z"] - below_top, c["floor"] + GRASP_MIN_ABOVE_FLOOR)
        p = c.get("pts")
        pref = wrap_half(c.get("slab_yaw", c["yaw"]))
        cands = [pref] + [wrap_half(pref + s * d) for d in (0.26, 0.52, 0.79, 1.05, 1.31) for s in (1, -1)] + [wrap_half(pref + 1.57)]

        def fit(hx, hy, yaw):
            return next((t for t in TILTS if fits_with_approach(self.containers, (hx, hy, z_g), yaw, t, inside)), None)
        if band is None and p is not None:
            band = p[p[:, 2] > c["top_z"] - 0.015]      # the top rim only: side walls seen obliquely inflate deeper slabs
        if band is None or len(band) < 10:
            for yaw in cands:
                tilt = fit(x, y, yaw)
                if tilt is not None:
                    return (x, y), yaw, tilt
            return None
        half_open = GRASP_MAX_WIDTH / 2 - WIDTH_MARGIN / 2
        best = None
        for k, yaw in enumerate(cands):
            u = np.array([-math.sin(yaw), math.cos(yaw)])
            v = np.array([math.cos(yaw), math.sin(yaw)])
            pv = (band[:, :2] - (x, y)) @ v
            vlo, vhi = float(np.percentile(pv, 3)), float(np.percentile(pv, 97))
            # the hand may also sit off the object's middle across the fingers (along v): the pads,
            # 2 cm wide, still close on the object's two faces as long as they overlap it by a
            # centimetre; near a container wall that is what lets a straight-down hand in
            voffs = [0.0]
            if vhi - vlo > 2 * PAD_OVERLAP:
                lo_v, hi_v = vlo + PAD_OVERLAP, vhi - PAD_OVERLAP
                voffs = sorted(set([0.0] + [float(o) for o in np.arange(-0.03, 0.0301, 0.006) if lo_v <= o <= hi_v]), key=abs)
            for voff in voffs:
                sel = np.abs(pv - voff) < (0.02 if voff == 0.0 else 0.015)
                q = band[sel] if sel.sum() >= 8 else (band if voff == 0.0 else None)
                if q is None:
                    continue
                proj = (q[:, :2] - (x, y)) @ u
                lo, hi = float(np.percentile(proj, 3)), float(np.percentile(proj, 97))
                w = hi - lo
                if w + WIDTH_MARGIN > GRASP_MAX_WIDTH:
                    continue
                # hand offsets along the finger axis that keep the object between the open fingers,
                # nearest to centred first
                centre = (lo + hi) / 2
                span = half_open - w / 2
                for off in sorted(np.arange(-span, span + 1e-9, 0.005), key=lambda o: abs(o)):
                    hx = x + (centre + off) * u[0] + voff * v[0]
                    hy = y + (centre + off) * u[1] + voff * v[1]
                    tilt = fit(hx, hy, yaw)
                    if tilt is not None and not is_tilted(tilt) and not fingers_clear(c.get("scene_pts"), hx, hy, yaw, z_g,
                                                                                      palm=inside is None and c.get("container") is None):
                        continue
                    if tilt is not None:
                        # upright first (a tilted hand lifts about half as often), then narrow and centred,
                        # and not the way it already failed here
                        cost = w + 0.2 * abs(off) + 0.25 * abs(voff) + 0.0005 * k \
                            + (TILT_PENALTY + 0.02 * (abs(tilt[0]) + abs(tilt[1])) if is_tilted(tilt) else 0.0) \
                            + FAILED_YAW_PENALTY * self.failed_here(x, y, yaw)
                        if best is None or cost < best[0]:
                            best = (cost, (float(hx), float(hy)), yaw, tilt)
                        break
        return None if best is None else (best[1], best[2], best[3])

    def search_pinch(self, c, inside=None, band=None):
        """When the pinch at the centroid is too wide (a clamp across its jaw, scissors across the
        blades), look for one elsewhere on the object: candidate centres on a 1.2 cm grid over the
        top slab, every yaw; feasible = narrower than the opening with margin and enough of the
        object between the fingers; cost = width + 0.5 x distance from the centroid (torque).
        -> ((x, y), yaw) or None."""
        p = c.get("pts")
        if p is None or len(p) < 40:
            return None
        if band is None:
            band = p[p[:, 2] > c["top_z"] - 0.015]
        if len(band) < 20:
            return None
        cx, cy = c["grasp_xy"]
        z_g = max(c["top_z"] - GRASP_BELOW_TOP, c["floor"] + GRASP_MIN_ABOVE_FLOOR)
        lo, hi = band[:, :2].min(0), band[:, :2].max(0)
        best = None
        yaws = [k * math.pi / 12 - math.pi / 2 for k in range(12)]
        for gx in np.arange(lo[0], hi[0] + 1e-6, 0.012):
            for gy in np.arange(lo[1], hi[1] + 1e-6, 0.012):
                near = np.hypot(band[:, 0] - gx, band[:, 1] - gy)
                if near.min() > 0.008:                 # not on the object
                    continue
                dist = math.hypot(gx - cx, gy - cy)
                if dist > 0.045:                       # a pinch far out on a handle swings the object off
                    continue
                for yaw in yaws:
                    v = np.array([math.cos(yaw), math.sin(yaw)])
                    sel = np.abs((band[:, :2] - (gx, gy)) @ v) < 0.02
                    if sel.sum() < 15:
                        continue
                    u = np.array([-math.sin(yaw), math.cos(yaw)])
                    proj = (band[sel, :2] - (gx, gy)) @ u
                    lo_u, hi_u = np.percentile(proj, 3), np.percentile(proj, 97)
                    width = float(hi_u - lo_u)
                    if width + WIDTH_MARGIN > GRASP_MAX_WIDTH or abs(lo_u + hi_u) / 2 > 0.012:
                        continue                         # too wide, or the fingers would close off-centre
                    cost = width + 0.5 * dist
                    if best is not None and cost >= best[0]:
                        continue
                    tilt = next((t for t in TILTS if fits_with_approach(self.containers, (gx, gy, z_g), yaw, t, inside)), None)
                    if tilt is None:
                        continue
                    if not is_tilted(tilt) and not fingers_clear(c.get("scene_pts"), gx, gy, yaw, z_g,
                                                                  palm=inside is None and c.get("container") is None):
                        continue
                    best = (cost + 0.02 * (abs(tilt[0]) + abs(tilt[1])), (float(gx), float(gy)), wrap_half(yaw), tilt)
        return None if best is None else (best[1], best[2], best[3])


    def rim_grasp(self, c, inside=None):
        """Wide hollow object (bowl, mug, pitcher): one finger inside, one outside, straddling the wall
        at the rim. Hollow = the cloud dips at least RIM_MIN_DEPTH below the rim around its centre.
        -> ((x, y), yaw) with the finger axis along the radius, or None."""
        p = c.get("pts")
        top = c["top_z"]
        if p is None or len(p) < 40:
            return None
        slab = p[p[:, 2] > top - 0.03]
        if len(slab) < 20:
            return None
        cen = slab[:, :2].mean(0)
        radius = float(np.percentile(np.hypot(slab[:, 0] - cen[0], slab[:, 1] - cen[1]), 90))
        inner = p[np.hypot(p[:, 0] - cen[0], p[:, 1] - cen[1]) < 0.4 * radius]
        depth = top - (float(np.median(inner[:, 2])) if len(inner) >= 5 else top)
        if radius < RIM_MIN_RADIUS or depth < RIM_MIN_DEPTH:
            return None
        for k in range(12):
            th = k * math.pi / 6
            wx, wy = cen[0] + radius * math.cos(th), cen[1] + radius * math.sin(th)
            yaw = wrap_half(th - math.pi / 2)             # finger axis (-sin yaw, cos yaw) = radial
            if math.hypot(wx, wy) > REACH - 0.04 or math.hypot(wx, wy) < BASE_CLEAR + 0.02:
                continue
            if not hand_fits(self.containers, wx, wy, yaw, inside=inside):
                continue
            return (float(wx), float(wy)), yaw
        return None

    def plan_grasps(self, c, inside=None):
        """-> [(xy, yaw, below_top), ...] for cluster c, best first: a pinch across its narrow
        side when it fits the hand, then a rim grasp when the object is hollow. Empty = skip."""
        out = []
        xy = (c["grasp_xy"][0], c["grasp_xy"][1])
        height = c["top_z"] - c["floor"]
        rim = self.rim_grasp(c, inside=inside)
        info = c.get("cls_info") or {}
        if rim is not None and inside is not None and (info.get("model") not in HOLLOW or info.get("d", 9.0) > IN_CONTAINER_SURE_D):
            rim = None      # inside a container two objects with a gap between them look hollow
        below = grasp_depth(c)
        pick = self.choose_yaw(c, inside=inside, below_top=below)
        shallow = [(pick[0], pick[1], below, pick[2])] if pick is not None else []
        if rim is not None and (not shallow or c["len_minor"] > 0.055):
            # hollow and wide (bowl, mug, pitcher): the rim grasp, one finger inside, is the plan.
            # A pinch across it only "fits" when the hand camera saw half of it, and then a finger
            # lands on the rim and pushes it away
            return [(rim[0], rim[1], RIM_GRASP_BELOW_TOP, None)]
        if not shallow:
            alt = self.search_pinch(c, inside=inside)
            if alt is not None:
                shallow = [(alt[0], alt[1], below, alt[2])]
        # No grasp deeper than the hand body allows: the body's underside is HAND_BODY_ABOVE_TCP
        # (3.4 cm) above the tool centre point, so fingers more than ~3 cm below an object's top
        # put the body onto the object and push it over instead of closing on it. Grasps planned
        # 4-6.5 cm below the top lifted 16 of 110 tries in the episode logs; 3 cm lifted 58%.
        out += shallow
        if rim is not None:
            out.append((rim[0], rim[1], RIM_GRASP_BELOW_TOP, None))
        return out

    def try_grasps(self, plans, top_z, floor, anchor_xy, keep_top=False):
        """Attempt the planned grasps in order; stop at the first that holds. A plan that closed
        on nothing is tried once more shifted RETRY_JITTER across the fingers (the direction a
        1-2 cm perception error turns into a fingertip landing on the object) and turned a
        little; at most RETRY_MAX attempts in all, so a hopeless object does not eat the deadline."""
        attempts = 0
        for plan in plans:
            xy, yaw, below = plan[0], plan[1], plan[2]
            tilt = plan[3] if len(plan) > 3 else None
            for k in range(2):
                if attempts >= RETRY_MAX:
                    return False
                if k == 1:
                    v = (-math.sin(yaw), math.cos(yaw))        # across the fingers
                    sg = 1.0 if (attempts % 2 == 0) else -1.0
                    xy = (plan[0][0] + sg * RETRY_JITTER * v[0], plan[0][1] + sg * RETRY_JITTER * v[1])
                    yaw = wrap_half(plan[1] + sg * RETRY_YAW)
                attempts += 1
                self.last_grasp_unreachable = False
                held = yield from self.pick(xy, yaw, top_z, floor, below_top=below, anchor_xy=anchor_xy, tilt=tilt, keep_top=keep_top)
                if held:
                    return True
                self.failed_grasps.append((float(anchor_xy[0]), float(anchor_xy[1]), float(wrap_half(yaw)), self.t()))
                log("t=%d grasp (yaw %.2f, depth %.3f, tilt %s%s) failed" % (self.t(), yaw, below, fmt_tilt(tilt), ", retry" if k else ""))
                if self.stuck_recovered:
                    # the arm folded up going there: leave the object for now. Only a second
                    # recovery on the way to the same request's target writes it off (three
                    # recoveries in a row once ran a request past its deadline; one, with the
                    # posture-regularised IK, is a hiccup, and giving the target up on it lost
                    # a request whose object was in plain sight)
                    self.stuck_recovered = False
                    self.stuck_strikes += 1
                    self.last_grasp_unreachable = self.stuck_strikes >= 2
                    log("t=%d leaving this object after the stuck-arm recovery%s" % (
                        self.t(), " (second time: written off)" if self.last_grasp_unreachable else ""))
                    return False
                if self.last_descent_short:
                    self.last_grasp_unreachable = True
                    break           # the hand never got there: a shifted retry will not either
        return False

    def free_spot(self, clusters, near=None):
        """A free desk spot for setting an object down: reachable, clear of the robot base and
        of other objects, and one where the open hand fits (to put it down and to take it back).
        The spot closest to `near` (x, y) wins."""
        occ = [c["centroid"][:2] for c in clusters]
        best = None
        for x in np.arange(0.17, 0.761, 0.02):
            for y in np.arange(-0.62, 0.621, 0.03):
                r = math.hypot(x, y)
                if r > REACH - 0.06 or r < BASE_CLEAR + 0.04:
                    continue
                if any(math.hypot(x - ox, y - oy) < 0.11 for ox, oy in occ):
                    continue
                if min(footprint_dist(self.containers, n, x, y) for n in self.containers) < SPOT_WALL_CLEAR:
                    continue      # right against a wall the hand cannot get round the object again
                if not (hand_fits(self.containers, x, y, 0.0) or hand_fits(self.containers, x, y, math.pi / 2)):
                    continue
                d = math.hypot(x - near[0], y - near[1]) if near is not None else r
                if best is None or d < best[0]:
                    best = (d, (float(x), float(y)))
        return best[1] if best else None

    def spot_yaw(self, xy):
        return 0.0 if hand_fits(self.containers, xy[0], xy[1], 0.0) else math.pi / 2

    def loose(self, clusters):
        return [c for c in clusters if c["container"] is None and c["cls_info"] is not None]

    @staticmethod
    def within_reach(c):
        """A loose object a straight-down hand can get to (TIDY_R): nearer the base the hand stops
        above the desk, farther out the arm is at full stretch and folds up."""
        return TIDY_R[0] <= math.hypot(c["grasp_xy"][0], c["grasp_xy"][1]) <= TIDY_R[1]

    @staticmethod
    def _near_any(c, points, r=0.05):
        return any(math.hypot(c["centroid"][0] - x, c["centroid"][1] - y) < r for x, y in points)

    def skipped(self, c):
        for (x, y, t, dur) in self.skip_list:
            if math.hypot(x - c["centroid"][0], y - c["centroid"][1]) < 0.06 and self.t() - t < dur:
                return True
        return False

    def skip(self, c, dur=600):
        """Leave this spot alone for `dur` steps; a spot that keeps failing is left alone for longer
        (an object wedged between two containers is not worth a visit every few hundred steps)."""
        x, y = c["centroid"][0], c["centroid"][1]
        fails = sum(1 for (px, py, t, d) in self.skip_list if math.hypot(px - x, py - y) < 0.08 and self.t() - t < 4000)
        self.skip_list.append((x, y, self.t(), dur * (1 + fails) ** 2))

    # -- main behaviour ---------------------------------------------------------------------------
    #
    # The controller keeps a belief of where every object is: what it put into each container
    # itself (model, slot, time), which containers may still hold objects it never identified (the
    # initial piles until they have been taken apart, any container the world may have dropped an
    # object into), and what it set down on the desk itself. A request is served from that belief:
    # loose candidates first, then the container entries for the named model, then the containers
    # with unidentified contents (dug out one object at a time onto the desk, where each object is
    # identified alone). "No such object" is marked only when nothing unidentified is left that
    # could be it; otherwise the request is left to expire, which costs nothing, while a wrong
    # mark costs a point.
    def run(self):
        self.last_sweep_t = -10 ** 9
        while True:
            clusters = yield from self.observe(sweep=self.t() - self.last_sweep_t > SWEEP_EVERY)
            for b in list(getattr(self, "maybe_empty", [])):
                yield from self.verify_empty(b)
            self.update_request()
            if self.request_state is not None:
                done = yield from self.serve_request(clusters)
                if done:
                    continue
            did = yield from self.tidy(clusters)
            if did:
                continue
            if self.request_state is None and (EXPLORE_PILES or EXPLORE_LITE):
                did = yield from self.explore(clusters)
                if did:
                    continue
                if self.t() - self.last_patrol_t > PATROL_EVERY:
                    yield from self.patrol()
                    continue
            yield from self.hold(10, grip=0.0)

    def update_request(self):
        req = self.request()
        if req is None:
            if self.request_state is not None and not self.request_state.get("marked"):
                log("t=%d request %r closed without a mark" % (self.t(), self.request_text))
            self.request_state = None
        elif req["text"] != self.request_text or self.request_state is None:
            self.request_text = req["text"]
            self.request_state = {"model": req["text"].replace("the ", "", 1), "inspected": [], "deadline": req["deadline_t"],
                                  "opened": self.t(), "failed": [], "checked": {}, "patrolled": set(), "identified": set()}
            self.stuck_strikes = 0
            log("t=%d request %r deadline %d; unknown %s; entries %s" % (
                self.t(), self.request_text, req["deadline_t"], {b: v for b, v in self.unknown.items() if v},
                [(e["name"], e["model"], e.get("slot")) for e in self.entries(self.request_state["model"])]))

    # -- belief helpers ---------------------------------------------------------------------------
    def slots_of(self, name):
        return RED_SLOTS if name == "red" else CLASS_SLOTS

    def slot_xy(self, name, k):
        lx, ly, _ = self.slots_of(name)[k]
        return container_point(self.containers[name], lx, ly)

    def entries(self, model=None, name=None):
        out = []
        for b, lst in self.inventory.items():
            if name is not None and b != name:
                continue
            for e in lst:
                if model is None or e["model"] == model or (model in TWIN and TWIN[model] == e["model"]):
                    out.append(e)
        return out

    def remove_entry(self, e):
        for b, lst in self.inventory.items():
            if e in lst:
                lst.remove(e)
                return


    def entry_near(self, name, xy, r=0.05):
        best = None
        for e in self.inventory.get(name, []):
            ex, ey = e["xy"]
            d = math.hypot(ex - xy[0], ey - xy[1])
            if d < r and (best is None or d < best[0]):
                best = (d, e)
        return best[1] if best else None

    def surface_at(self, name, xy, clusters):
        """Top of whatever the static cameras see standing within SLOT_CLEAR of xy in container
        `name` (the floor when nothing)."""
        top = CONTAINER_FLOOR_Z
        n = 0
        for c in clusters:
            if c["container"] != name or c.get("pts") is None:
                continue
            p = c["pts"]
            near = np.hypot(p[:, 0] - xy[0], p[:, 1] - xy[1]) < SLOT_CLEAR
            if near.sum() >= 6:
                n += int(near.sum())
                top = max(top, float(np.percentile(p[near, 2], 95)))
        return top, n

    def strict_free_slot(self, name):
        """A slot with no entry and nothing in the hand camera's last look into the container within
        SLOT_CLEAR + 2 cm of it: placing there buries nothing never identified. -> slot or None."""
        seen = self.last_comps.get(name)
        if seen is None or self.t() - seen[0] > 600:
            return None
        for k in range(len(self.slots_of(name))):
            if any(e.get("slot") == k for e in self.inventory.get(name, [])):
                continue
            sx, sy = self.slot_xy(name, k)
            busy = False
            for c in seen[1]:
                p = c.get("pts")
                if p is None:
                    continue
                near = np.hypot(p[:, 0] - sx, p[:, 1] - sy) < SLOT_CLEAR + 0.02
                if int(near.sum()) >= 6 and float(np.percentile(p[near, 2], 90)) > CONTAINER_FLOOR_Z + 0.015:
                    busy = True
                    break
            if not busy:
                return k
        return None

    def choose_slot(self, name, clusters):
        """The first slot with no entry and nothing seen standing in it; else the slot whose stack is
        lowest. -> (slot, surface z under it)"""
        n = len(self.slots_of(name))
        best = None
        for k in range(n):
            xy = self.slot_xy(name, k)
            top, cnt = self.surface_at(name, xy, clusters)
            used = any(e.get("slot") == k for e in self.inventory.get(name, []))
            free = not used and top < CONTAINER_FLOOR_Z + 0.015
            if free:
                return k, top
            if best is None or top < best[0]:
                best = (top, k)
        return best[1], best[0]

    def place_in_container(self, name, model=None, clusters=None, slot=None):
        """Carry the held object over container `name` and set it down in a slot (CLASS_SLOTS: the
        middle, both ends on the centre line, both sides of the middle), its bottom just above what
        is under it there, the hand turned the way that slot lets a straight-down hand in again. The
        model (when known) goes on the container's inventory with its slot."""
        clusters = self.last_clusters if clusters is None else clusters
        b = self.containers[name]
        if slot is None:
            k, under = self.choose_slot(name, clusters)
        else:
            k, under = slot, CONTAINER_FLOOR_Z
        lx, ly, phi = self.slots_of(name)[k]
        x, y = container_point(b, lx, ly)
        # finger axis (-sin yaw, cos yaw) along the slot's direction phi in the container frame
        yaw = yaw_branch((x, y), wrap_half(b[2] + phi - math.pi / 2))[0]
        z_rel = under + PLACE_GAP + self.carry_hang
        if name == "red" or not hand_fits3d(self.containers, (x, y, z_rel), yaw, (0.0, 0.0), inside=name):
            # red stands too near the base for a hand to reach low, and elsewhere the hand may not fit
            # down to the surface: release from just above the rim instead
            z_rel = max(z_rel, CONTAINER_RIM_Z + 0.02 + self.carry_hang)
        self.carrying = model
        ok = yield from self.carry_to((x, y, self.travel_z()), yaw)
        self.carrying = None
        if ok:
            yield from self.goto((x, y, z_rel), yaw, grip=1.0, tol=0.008, max_steps=40, step_max=0.03)
        if not ok:
            self.carry_hang = 0.0
            yield from self.goto((self.ee()[0], self.ee()[1], TRAVEL_Z), yaw, grip=0.0, tol=0.03, max_steps=20)
            return False
        still = True       # held all the way over the slot: if it slipped on the way down, it fell in right there
        yield from self.hold(4, grip=0.0)
        self.carry_hang = 0.0
        yield from self.goto((x, y, TRAVEL_Z), yaw, grip=0.0, tol=0.02, max_steps=30)
        if not still:
            log("t=%d the %s was lost on the way to %s (fingers shut before the release)" % (self.t(), model, name))
            return False
        entry = None
        if name in self.inventory:
            entry = {"model": model, "slot": k, "xy": (float(x), float(y)), "t": self.t(), "name": name}
            self.inventory[name].append(entry)
        self.forget_loose(model)
        self.vanished = [v for v in self.vanished if v[0] != model]
        log("t=%d placed %s in %s slot %d (%.2f,%.2f) from z %.3f" % (self.t(), model, name, k, x, y, z_rel))
        if name in CLASS_OF_CONTAINER:
            # the container as the controller left it, from the pose every later look into it takes:
            # the reference a later look compares with (what the world drops in shows up as new), and
            # where the object came to rest (it may roll a few centimetres off the slot)
            yield from self.hold(PLACE_SETTLE, grip=0.0)
            self.acted_t[name] = self.t()
            comps = yield from self.container_components(name)
            near = [c for c in comps if c["n"] >= 40 and math.hypot(c["grasp_xy"][0] - x, c["grasp_xy"][1] - y) < 0.09
                    and not any(e is not entry and math.hypot(e["xy"][0] - c["grasp_xy"][0], e["xy"][1] - c["grasp_xy"][1]) < 0.03
                                for e in self.inventory.get(name, []))]
            if entry is not None and near:
                c = min(near, key=lambda c: math.hypot(c["grasp_xy"][0] - x, c["grasp_xy"][1] - y))
                entry["xy"] = (float(c["grasp_xy"][0]), float(c["grasp_xy"][1]))
        return True

    def note_loose(self, model, xy):
        """A model identified alone on the desk (desk-sure): it is on the desk until the controller puts
        it away under that name (absence of the model is never declared meanwhile)."""
        if model is None:
            return
        self.vanished = [v for v in self.vanished if v[0] != model]
        self.loose_seen = [e for e in self.loose_seen if not (e[0] == model and math.hypot(e[1] - xy[0], e[2] - xy[1]) < 0.08)]
        self.loose_seen.append((model, float(xy[0]), float(xy[1]), self.t()))

    def forget_loose(self, model):
        if model is None:
            return
        near = [e for e in self.loose_seen if e[0] == model]
        if near:
            gx, gy = (self.last_grasp[0], self.last_grasp[1]) if getattr(self, "last_grasp", None) else (near[0][1], near[0][2])
            drop = min(near, key=lambda e: math.hypot(e[1] - gx, e[2] - gy))
            self.loose_seen.remove(drop)

    def desk_note(self, xy, model, src):
        """Something the controller set down on the desk itself: what it is (identified alone there)
        and where it came from (what a request dug out goes back before the mark)."""
        self.desk_items.append({"xy": (float(xy[0]), float(xy[1])), "model": model, "src": src, "t": self.t()})

    def desk_item_at(self, c, r=0.06):
        best = None
        for it in self.desk_items:
            d = math.hypot(it["xy"][0] - c["centroid"][0], it["xy"][1] - c["centroid"][1])
            if d < r and (best is None or d < best[0]):
                best = (d, it)
        return best[1] if best else None

    def prune_desk_items(self, clusters):
        """Drop desk notes with no loose cluster near them any more (sorted, or moved by the world)."""
        loose = [c for c in clusters if c["container"] is None]
        self.desk_items = [it for it in self.desk_items
                           if any(math.hypot(it["xy"][0] - c["centroid"][0], it["xy"][1] - c["centroid"][1]) < 0.07 for c in loose)]

    # -- requests -----------------------------------------------------------------------------------
    def serve_request(self, clusters):
        """One action for the open request; False when there is nothing to do for it right now (the
        caller tidies meanwhile)."""
        rs = self.request_state
        model = rs["model"]
        if rs.get("marked"):
            self.request_state = None
            return False
        left = rs["deadline"] - self.t()
        if rs.get("in_red") or rs.get("absent_ready"):
            # judged with the desk at the mark: while time allows, put back what was taken out and sort
            # whatever else lies loose or misfiled first (the world never moves the requested object
            # and never brings its model while the request is open, so waiting costs the mark nothing)
            if left > RESTORE_MARGIN:
                did = yield from self.tidy(clusters, restore_only=True)
                if not did:
                    did = yield from self.tidy(clusters)
                if did:
                    return True
            twin = CLAIM_TWIN.get(model)
            if rs.get("in_red") and twin is not None and not rs.get("can_check"):
                rs["can_check"] = True
                for c in clusters:
                    info = c.get("cls_info") or {}
                    if c.get("container") == "red" and info.get("model") == twin and info.get("model_margin", 0.0) >= 0.1:
                        log("t=%d the static cameras see a %s in red, not the %s: no claim" % (self.t(), twin, model))
                        rs["in_red"] = False
                        rs["trust_red"] = False
                        return True
            if rs.get("in_red") and not rs.get("red_recheck"):
                # before claiming delivery: a loose object that looks like the target is looked at
                # once (the target may have fallen out of red after the release)
                rs["red_recheck"] = True
                for c in self.loose(clusters):
                    if not self.within_reach(c) or not self.could_be(c, model, 0.25):
                        continue
                    d = yield from self.inspect(c)
                    if d is not None and is_model(d["cls_info"], model):
                        log("t=%d the %s lies loose at (%.2f,%.2f): not in red after all" % (
                            self.t(), model, d["grasp_xy"][0], d["grasp_xy"][1]))
                        rs["in_red"] = False
                        rs["trust_red"] = False
                        rs["seen_target"] = True
                        rs["target_at"] = (float(d["grasp_xy"][0]), float(d["grasp_xy"][1]))
                        return True
                return True
            mark = 1 if rs.get("in_red") else 2
            yield from self.hold(1, grip=0.0, mark=mark)
            rs["marked"] = True
            log("t=%d marked %s" % (self.t(), "delivered" if mark == 1 else "absent"))
            self.request_state = None
            return True
        if rs.get("placed_red"):
            rs["placed_red"] = False
            verdict = yield from self.confirm_red(model)
            log("t=%d red check after delivery: %s" % (self.t(), verdict))
            if verdict == "yes" or (verdict == "unsure" and rs.get("trust_red")):
                rs["in_red"] = True
            return True
        # the desk is judged when the request closes, also when it expires: with the deadline near,
        # what the search took out goes back first (about RESTORE_STEPS each)
        out = [c for c in self.loose(clusters) if self.desk_item_at(c) is not None and self.within_reach(c)]
        if out and left < RESTORE_MARGIN + RESTORE_STEPS * len(out):
            if not rs.get("restoring"):
                rs["restoring"] = True
                log("t=%d %d steps left: putting %d objects back before the deadline" % (self.t(), left, len(out)))
            did = yield from self.tidy(clusters, restore_only=True)
            if did:
                return True
        # loose objects that may be it
        cands = self.loose_candidates(clusters, model)
        if cands:
            done = yield from self.check_candidate(cands[0])
            return done
        # what the controller put away itself
        for e in sorted(self.entries(model), key=lambda e: -e["t"]):
            if rs["checked"].get(id(e), 0) >= ENTRY_TRIES:
                continue
            rs["checked"][id(e)] = rs["checked"].get(id(e), 0) + 1
            done = yield from self.retrieve_entry(e)
            if done:
                return True
        # containers with contents never identified
        for b in self.search_containers(model):
            if rs.get("stalled", {}).get(b, 0) >= STALL_ROUNDS or self.t() - rs.get("blocked", {}).get(b, -999) < 150:
                continue
            if sum(1 for f in rs["failed"] if f[0] == b) >= DIG_FAILS_MAX:
                continue            # this pile has refused enough grasps for one request
            res = yield from self.dig_one(b, clusters)
            if res:
                return True
        # a look into every container from above (the world may have dropped it into one)
        for b in ("red", "gray", "lightblue", "blue"):
            if b in rs["patrolled"]:
                continue
            rs["patrolled"].add(b)
            done = yield from self.look_for(b, model)
            if done:
                return True
        # what went into its class container without a sure name
        home = CONTAINER_OF_CLASS.get(CLASS_OF.get(model))
        for e in sorted([e for e in self.inventory.get(home, []) if e["model"] is None], key=lambda e: -e["t"]):
            if rs["checked"].get(id(e), 0) >= ENTRY_TRIES:
                continue
            rs["checked"][id(e)] = rs["checked"].get(id(e), 0) + 1
            done = yield from self.retrieve_entry(e, identify=True)
            if done:
                return True
        # any loose object not yet looked at
        cands = [c for c in self.loose(clusters) if not self._near_any(c, rs["inspected"]) and self.within_reach(c)]
        if cands:
            done = yield from self.check_candidate(cands[0])
            return done
        if self.absent_certain(model):
            if not rs.get("swept"):
                rs["swept"] = True          # one more look first, the strip's blind ends included
                self.last_sweep_t = -10 ** 9
                return True
            log("t=%d absent: %s (nothing unidentified could be it); tidying before the mark" % (self.t(), model))
            rs["absent_ready"] = True
            return True
        # once more, from the top: what moved since may be reachable now
        if not rs.get("second_pass") and rs["deadline"] - self.t() > 400:
            rs["second_pass"] = True
            rs["patrolled"] = set()
            rs["identified"] = set()
            rs["stalled"] = {}
            rs["blocked"] = {}
            rs["checked"] = {}
            rs["inspected"] = []
            rs["clear_of"] = {}
            log("t=%d %s not found yet; a second pass" % (self.t(), model))
            return True
        for b in [b for b, v in self.unknown.items() if v in ("seen", "lost")]:
            if ("account", b) in rs["patrolled"]:
                continue
            rs["patrolled"].add(("account", b))
            moved = yield from self.account_for(b)
            if moved:
                return True
        if self.absent_certain(model):
            if not rs.get("swept"):
                rs["swept"] = True          # one more look first, the strip's blind ends included
                self.last_sweep_t = -10 ** 9
                return True
            log("t=%d absent: %s (nothing unidentified could be it); tidying before the mark" % (self.t(), model))
            rs["absent_ready"] = True
            return True
        if not rs.get("logged_stuck"):
            rs["logged_stuck"] = True
            log("t=%d %s not found, absence not certain (unknown %s, entries %s); the request stays open" % (
                self.t(), model, {b: v for b, v in self.unknown.items() if v},
                [(e["name"], e["model"], e.get("stuck", False)) for e in self.entries(model)]))
        return False

    def loose_candidates(self, clusters, model):
        rs = self.request_state
        out = []
        for c in self.loose(clusters):
            if self._near_any(c, rs["inspected"]) or not self.within_reach(c):
                continue
            ta = rs.get("target_at")
            if ta is not None and rs.get("target_tries", 0) < TARGET_TRIES and \
                    math.hypot(c["centroid"][0] - ta[0], c["centroid"][1] - ta[1]) < 0.10:
                out.insert(0, c)           # identified there by the hand camera: tried again first
                continue
            it = self.desk_item_at(c)
            if it is not None and it["model"] is not None:
                if not (it["model"] == model or TWIN.get(it["model"]) == model):
                    continue             # identified alone on the desk when it was set down there
            elif not self.could_be(c, model, CANDIDATE_SLACK):
                continue
            out.append(c)
        ta = rs.get("target_at")
        return sorted(out, key=lambda c: (0 if (ta is not None and math.hypot(c["centroid"][0] - ta[0], c["centroid"][1] - ta[1]) < 0.10) else 1,
                                          (c["cls_info"].get("all") or {}).get(model, 9.0)))

    def could_be(self, c, model, slack):
        info = c.get("cls_info") or {}
        if not info.get("all"):
            return True
        d = info["all"].get(model, 9.0)
        if TWIN.get(model) in info["all"]:
            d = min(d, info["all"][TWIN[model]])
        return d - info["d"] < slack

    def search_containers(self, model):
        """Containers that may hold the model among objects never identified: the initial pile of its
        class until it has been taken apart (and what could not be identified there), and any
        container the world may have dropped something into."""
        cls = CLASS_OF.get(model)
        out = []
        for b in ("gray", "lightblue", "blue"):
            u = self.unknown.get(b)
            if u in ("pile", "stuck") and CLASS_OF_CONTAINER.get(b) == cls:
                out.append(b)
        return out     # (something new in a container is looked at by look_for, not dug blindly)

    def absent_certain(self, model):
        """No loose object could be it, no entry names it, and no container holds anything never
        identified that could be it."""
        if self.entries(model) or self.search_containers(model) or self.request_state.get("seen_target") \
                or self.request_state.get("sightings"):
            return False
        twin = CAN_TWIN.get(model)
        if twin is not None and (self.entries(twin) or any(e[0] == twin for e in self.loose_seen)
                                 or any(v[0] == twin for v in self.vanished)):
            return False             # a look-alike can is on the books: it may be this one
        if any(e[0] == model or e[0] == TWIN.get(model) for e in self.loose_seen):
            return False             # identified alone on the desk earlier and not put away since
        if any(v[0] == model or v[0] == TWIN.get(model) for v in self.vanished):
            return False             # put away earlier and no longer seen where it was put
        home = CONTAINER_OF_CLASS.get(CLASS_OF.get(model))
        if any(e["model"] is None for e in self.inventory.get(home, [])):
            return False
        rs = self.request_state
        for b, v in self.unknown.items():
            if v == "seen" and not (rs.get("clear_of", {}).get(b, -1) >= self.seen_t.get(b, 0)):
                return False
        for c in self.loose(self.last_clusters):
            it = self.desk_item_at(c)
            if it is not None and it["model"] is not None:
                if it["model"] == model or TWIN.get(it["model"]) == model:
                    return False
                continue
            if self._near_any(c, rs["inspected"]):
                continue
            if self.could_be(c, model, 2 * CANDIDATE_SLACK):
                return False
        return True

    def check_candidate(self, c):
        """Inspect a loose cluster from the hand camera; deliver it when it is the target."""
        rs = self.request_state
        model = rs["model"]
        key = (c["centroid"][0], c["centroid"][1])
        rs["inspected"].append(key)
        d = yield from self.inspect(c)
        if d is None:
            return True
        info = d["cls_info"]
        log("t=%d request check: %s (d=%.2f, margin %.2f) vs wanted %s" % (self.t(), info["model"], info["d"],
            info.get("model_margin", 0.0), model))
        if desk_sure(info):
            self.note_loose(info["model"], d["grasp_xy"])
        if not is_model(info, model):
            it = self.desk_item_at(c)
            if it is not None and it["model"] is None and desk_sure(info):
                it["model"] = info["model"]
            return True
        rs["seen_target"] = True           # it is here: absence is never claimed for this request
        rs["target_at"] = (float(d["grasp_xy"][0]), float(d["grasp_xy"][1]))
        g = self.plan_grasps(d)
        if d.get("blocker"):
            acted = yield from self.clear_blocker(d, None, self.last_clusters)
            if acted:
                rs["inspected"].remove(key)
                return True
        if not g:
            log("t=%d target not graspable where it lies" % self.t())
            pushed = yield from self.topple_if_tall(d)
            if pushed:
                rs["inspected"].remove(key)
            return True
        rs["target_tries"] = rs.get("target_tries", 0) + 1
        held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"]))
        if not held:
            log("t=%d grasp of the target failed%s" % (self.t(), "" if self.last_grasp_unreachable else "; retrying"))
            if not self.last_grasp_unreachable:
                rs["inspected"].remove(key)
            yield from self.topple_if_tall(d)
            return True
        it = self.desk_item_at(c)
        if it is not None:
            self.desk_items.remove(it)
        rs["trust_red"] = desk_sure(info)
        yield from self.deliver()
        return True

    def topple_if_tall(self, d):
        """After a failed grasp of a tall object standing loose on the desk: push it over along the
        short side of its footprint, toward free desk, with the closed hand at ~0.6 of its height.
        A top-down pinch reaches only an object's top ~6 cm, which on a standing bottle is the sloped
        shoulder; lying, it is a 4-5 cm tall object the fingers close on at mid-height.
        -> True when the arm pushed."""
        if d is None or d.get("container") is not None:
            return False
        h = d["top_z"] - d["floor"]
        if h < TOPPLE_MIN_H or d["len_minor"] > TOPPLE_MAX_FOOT:
            return False
        cx, cy = float(d["centroid"][0]), float(d["centroid"][1])
        recent = [p for p in self.toppled if self.t() - p[2] < FAILED_MEMORY and math.hypot(p[0] - cx, p[1] - cy) < 0.10]
        if len(recent) >= TOPPLE_MAX:
            return False
        half = d["len_minor"] / 2.0 + WIDTH_MARGIN / 2.0
        occ = [c["centroid"][:2] for c in self.last_clusters
               if math.hypot(c["centroid"][0] - cx, c["centroid"][1] - cy) > 0.05]
        best = None
        for dyaw in (0.0, 0.35, -0.35):
            for flip in (0.0, math.pi):
                a = d["yaw"] + math.pi / 2 + dyaw + flip
                ux, uy = math.cos(a), math.sin(a)
                body = [(cx + ux * s, cy + uy * s) for s in (0.03, h / 2.0, h + 0.01)]
                ok = all(TIDY_R[0] + 0.02 <= math.hypot(px, py) <= TIDY_R[1] for px, py in body[:2]) \
                    and math.hypot(*body[2]) <= REACH - 0.02
                ok = ok and all(footprint_dist(self.containers, n, px, py) > 0.03 for n in self.containers for px, py in body)
                ok = ok and all(math.hypot(px - ox, py - oy) > 0.08 for px, py in body for ox, oy in occ)
                if not ok:
                    continue
                # the hand's broad side leading (its thickness axis along the push), or, beside a
                # wall, its narrow end (its long axis along the push)
                for k, (yaw_h, back) in enumerate(((wrap_half(a), HAND_HALF_T), (wrap_half(a + math.pi / 2), HAND_HALF_W))):
                    start = (cx - ux * (half + back + 0.02), cy - uy * (half + back + 0.02))
                    if not hand_fits(self.containers, start[0], start[1], yaw_h):
                        continue
                    if not all(math.hypot(px - ox, py - oy) > 0.06 for px, py in hand_corners(start[0], start[1], yaw_h) for ox, oy in occ):
                        continue
                    if math.hypot(*start) > REACH - 0.03 or math.hypot(*start) < BASE_CLEAR - 0.06:
                        continue
                    score = abs(dyaw) + abs(math.hypot(*body[1]) - 0.56) + 0.1 * k
                    if best is None or score < best[0]:
                        best = (score, a, start, yaw_h)
        if best is None:
            log("t=%d topple: no free direction for the %.2f m tall object at (%.2f,%.2f)" % (self.t(), h, cx, cy))
            return False
        _, a, start, yaw = best
        ux, uy = math.cos(a), math.sin(a)
        # the palm's underside at ~0.6 of the height, the closed fingertips clear of the desk
        z_push = d["floor"] + max(0.6 * h - HAND_BODY_ABOVE_TCP, 0.035)
        back = HAND_HALF_T if abs(wrap_half(yaw - wrap_half(a))) < 0.1 else HAND_HALF_W
        end = (cx + ux * (half + 0.06 - back), cy + uy * (half + 0.06 - back))   # the leading face 6 cm past the far side
        log("t=%d topple: pushing the %.2f m tall object at (%.2f,%.2f) over toward (%.2f,%.2f)" % (
            self.t(), h, cx, cy, cx + ux * h / 2, cy + uy * h / 2))
        self.toppled.append((cx, cy, self.t()))
        z_up = max(self.travel_z(), d["top_z"] + 0.10)
        yield from self.goto((start[0], start[1], z_up), yaw, grip=1.0, tol=0.015)
        yield from self.goto((start[0], start[1], z_push), yaw, grip=1.0, tol=0.01, max_steps=40, step_max=0.03)
        yield from self.goto((end[0], end[1], z_push), yaw, grip=1.0, tol=0.012, max_steps=40, step_max=0.012)
        yield from self.goto((end[0], end[1], z_up), yaw, grip=0.0, tol=0.02, max_steps=30)
        rs = self.request_state
        if rs is not None and rs.get("target_at") is not None and math.hypot(rs["target_at"][0] - cx, rs["target_at"][1] - cy) < 0.10:
            rs["target_at"] = (cx + ux * h / 2.0, cy + uy * h / 2.0)
        return True

    def deliver(self):
        """The held object is the target: into red."""
        rs = self.request_state
        ok = yield from self.place_in_container("red", rs["model"])
        if ok:
            rs["placed_red"] = True
            yield from self.hold(6, grip=0.0)

    def retrieve_entry(self, e, identify=False):
        """The model went into container e["name"] at slot e["slot"]: take off what was put on top of
        it later (onto the desk), then take it (with identify: onto the desk too, to be named there
        alone). -> True when the arm did something."""
        rs = self.request_state
        b = e["name"]
        above = sorted([o for o in self.inventory.get(b, []) if o is not e and o.get("slot") is not None
                        and o.get("slot") == e.get("slot") and o["t"] > e["t"]], key=lambda o: o["t"])
        comps = yield from self.container_components(b)
        x, y = e["xy"]
        near = sorted([c for c in comps if math.hypot(c["grasp_xy"][0] - x, c["grasp_xy"][1] - y) < ENTRY_RADIUS],
                      key=lambda c: -c["top_z"])
        if not near:
            # not where it was put: it may have been pushed; any component no other entry explains
            others = [c for c in comps if c["n"] >= 40 and not any(o is not e and math.hypot(o["xy"][0] - c["grasp_xy"][0], o["xy"][1] - c["grasp_xy"][1]) < 0.04
                                                              for o in self.inventory.get(b, []))]
            if not others or rs["checked"].get(("moved", id(e)), 0) >= 1:
                log("t=%d entry %s in %s slot %s: nothing there any more" % (self.t(), e["model"], b, e.get("slot")))
                self.remove_entry(e)
                if e["model"] is not None:
                    # not seen is not gone (a pair of scissors lying on another object in blue was missed
                    # this way, and "no such object" went out): the model stays present for absence
                    self.vanished.append((e["model"], b, self.t()))
                return True
            rs["checked"][("moved", id(e))] = 1
            near = sorted(others, key=lambda c: math.hypot(c["grasp_xy"][0] - x, c["grasp_xy"][1] - y))
            log("t=%d entry %s in %s: not at its slot; trying the component at (%.2f,%.2f)" % (
                self.t(), e["model"], b, near[0]["grasp_xy"][0], near[0]["grasp_xy"][1]))
        c = near[0]
        d = yield from self.inspect(c)
        if d is None:
            return True
        info = d["cls_info"]
        log("t=%d entry %s in %s slot %s: top is %s (d=%.2f), %d entries on top" % (
            self.t(), e["model"], b, e.get("slot"), info["model"], info["d"], len(above)))
        g = self.plan_grasps(d, inside=b)
        if not g:
            rs.setdefault("blocked", {})[b] = self.t()
            return True
        held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"]))
        if not held:
            return True
        if above:
            self.remove_entry(above[-1])
            yield from self.set_down_and_identify(b, info)
            return True
        self.remove_entry(e)
        if identify or not is_model(info, rs["model"], 0.35):
            # straight to red only when the view in the container agrees: it may name something else
            # (in a pile a tomato soup can read as the master chef can at d=0.20), or be too poor to
            # name anything (the world dropped a banana onto an entry, read at d=1.21, and it went to
            # red as the tomato soup can): look at it alone on the desk first
            yield from self.set_down_and_identify(b, info)
            return True
        rs["trust_red"] = True       # where the controller put the model itself, and the view agrees
        yield from self.deliver()
        return True

    def set_down_and_identify(self, b, info=None):
        """The held object came out of container b: set it down on a free desk spot, look at it there
        alone (the hand camera from above), and deliver it right away when it is the target."""
        rs = self.request_state
        origin = (float(self.last_grasp[0]), float(self.last_grasp[1])) if getattr(self, "last_grasp", None) else None
        spot = self.free_spot(self.last_clusters, near=tuple(self.ee()[:2]))
        if spot is None:
            log("t=%d no free desk spot; back into %s" % (self.t(), b))
            yield from self.place_in_container(b, None)
            if b in self.unknown and not self.unknown[b]:
                self.unknown[b] = "seen"
                self.seen_t[b] = self.t()
            return
        still = yield from self.place_on_desk(spot, self.spot_yaw(spot))
        if not still:
            return
        c = {"grasp_xy": [spot[0], spot[1]], "top_z": 0.12, "floor": float(desk_top_z(spot[0])), "container": None,
             "centroid": [spot[0], spot[1], 0.04], "cls_info": None, "n": 0}
        d = yield from self.inspect(c, radius=0.08)
        if d is not None and math.hypot(d["centroid"][0] - spot[0], d["centroid"][1] - spot[1]) > 0.06:
            d = None                      # a neighbour, not what was set down
        model = d["cls_info"]["model"] if (d is not None and desk_sure(d["cls_info"])) else None
        if d is not None:
            self.withdraw_sighting(b, origin, d["cls_info"])
            if model is not None:
                self.note_loose(model, d["grasp_xy"])
        where = (d["grasp_xy"][0], d["grasp_xy"][1]) if d is not None else spot
        self.desk_note(where, model, b)
        log("t=%d out of %s onto the desk at (%.2f,%.2f): %s" % (self.t(), b, where[0], where[1],
            "%s (d=%.2f)" % (d["cls_info"]["model"], d["cls_info"]["d"]) if d else "nothing seen"))
        if rs is not None and not rs.get("in_red") and d is not None and is_model(d["cls_info"], rs["model"]):
            rs["seen_target"] = True
            rs["target_at"] = (float(d["grasp_xy"][0]), float(d["grasp_xy"][1]))
            g = self.plan_grasps(d)
            if g:
                held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"]))
                if held:
                    self.desk_items = [it for it in self.desk_items if it["xy"] != (float(where[0]), float(where[1]))]
                    rs["trust_red"] = desk_sure(d["cls_info"])
                    yield from self.deliver()

    def dig_one(self, b, clusters, explore=False):
        """Take one object out of container b's unidentified contents: the highest component with a
        grasp the hand fits in the container (the target, when the view from above is sure of it,
        straight to red; anything else onto the desk, identified alone there).
        -> True when the arm did something, False when nothing in b can be taken now (and, exploring,
        "done" once what is left has been identified where it lies)."""
        rs = self.request_state
        comps = yield from self.container_components(b)
        comps = [c for c in comps if c["n"] >= 40]
        failed = rs["failed"] if rs is not None else self.explore_failed
        tried = 0
        for c in sorted(comps, key=lambda c: -c["top_z"]):
            if tried >= 3:
                break
            if self.entry_near(b, c["grasp_xy"], r=0.035):
                continue                   # something the controller put there itself
            fails = sum(1 for (fb, fx, fy) in failed if fb == b and math.hypot(fx - c["grasp_xy"][0], fy - c["grasp_xy"][1]) < 0.05)
            if fails >= UNPACK_FAILS_MAX:
                continue
            tried += 1
            d = yield from self.inspect(c)
            if d is None:
                continue
            g = self.plan_grasps(d, inside=b)
            info = d["cls_info"]
            log("t=%d dig %s: %s (d=%.2f, margin %.2f) at (%.2f,%.2f) top=%.3f grasps=%s" % (
                self.t(), b, info["model"], info["d"], info.get("model_margin", 0.0), d["grasp_xy"][0], d["grasp_xy"][1],
                d["top_z"], fmt_plans(g)))
            if d.get("blocker"):
                acted = yield from self.clear_blocker(d, b, clusters)
                if acted:
                    return True
            if not g:
                continue
            held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"]))
            if not held:
                failed.extend([(b, d["grasp_xy"][0], d["grasp_xy"][1])] * (2 if self.last_grasp_unreachable else 1))
                return True
            sure = rs is not None and info["model"] == rs["model"] and info.get("model_margin", 0.0) > 0.3 \
                and info["d"] < IN_CONTAINER_SURE_D
            if sure:
                rs["trust_red"] = True
                yield from self.deliver()
                return True
            yield from self.set_down_and_identify(b, info)
            return True
        if not [c for c in comps if not self.entry_near(b, c["grasp_xy"], r=0.035)]:
            log("t=%d %s: nothing unidentified left inside" % (self.t(), b))
            self.unknown[b] = None
            return "done" if explore else False
        if rs is not None and self.unknown.get(b) in ("pile", "stuck") and not rs.setdefault("dug_entries", {}).get(b):
            # an entry the controller put on top of an unidentified pile may hide the object: take the
            # highest graspable entry out onto the desk once per request and container
            rs["dug_entries"][b] = True
            for c in sorted([c for c in comps if self.entry_near(b, c["grasp_xy"], r=0.035)], key=lambda c: -c["top_z"])[:2]:
                d = yield from self.inspect(c)
                if d is None:
                    continue
                g = self.plan_grasps(d, inside=b)
                if not g:
                    continue
                held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"]))
                if held:
                    e = self.entry_near(b, d["grasp_xy"], r=0.05)
                    if e is not None:
                        self.remove_entry(e)
                    yield from self.set_down_and_identify(b, d["cls_info"])
                return True
        # what is left cannot be taken now: identify it from above where it lies (once per request)
        if rs is not None:
            rs.setdefault("blocked", {})[b] = self.t()
            rs.setdefault("stalled", {})[b] = rs.setdefault("stalled", {}).get(b, 0) + 1
            if b in rs["identified"]:
                return False
            rs["identified"].add(b)
        yield from self.identify_stuck(b, comps)
        return "done" if explore else True

    def identify_stuck(self, b, comps):
        """What is left in b has no grasp: identify each component from above and note it as a stuck
        entry; b counts as known when every one of them is identified for sure."""
        ids, sure = [], True
        for c in sorted(comps, key=lambda c: -c["n"])[:6]:
            if self.entry_near(b, c["grasp_xy"], r=0.035):
                continue
            d = yield from self.inspect(c)
            if d is None:
                continue
            info = d["cls_info"]
            ok = info["d"] < IN_CONTAINER_SURE_D and info.get("model_margin", 0.0) > 0.25
            ids.append((info["model"], round(info["d"], 2), ok))
            if ok:
                self.inventory[b].append({"model": info["model"], "slot": None, "xy": tuple(d["grasp_xy"]), "t": self.t(),
                                          "name": b, "stuck": True})
            else:
                sure = False
        log("t=%d %s: left out of reach %s -> %s" % (self.t(), b, ids, "all identified" if sure else "some unidentified"))
        self.unknown[b] = None if sure else ("stuck" if self.unknown.get(b) in ("pile", "stuck") else self.unknown.get(b))

    def look_for(self, b, model):
        """A look into container b from above for the requested model (the world may have dropped it
        there). -> True when the arm did something useful."""
        comps = yield from self.container_components(b)
        rs = self.request_state
        todo = sorted([c for c in comps if c["n"] >= 40 and (b == "red" or not self.entry_near(b, c["grasp_xy"], r=0.035))],
                      key=lambda c: -c["top_z"])
        clear = len(todo) <= 4
        for c in todo[:4] if b != "red" else todo[:2]:
            d = yield from self.inspect(c)
            if d is None:
                clear = False
                continue
            info = d["cls_info"]
            log("t=%d look into %s: %s (d=%.2f)" % (self.t(), b, info["model"], info["d"]))
            alld = info.get("all") or {}
            near = min(alld.get(model, 9.0), alld.get(TWIN.get(model), 9.0)) - info["d"]
            if b != "red" and (is_model(info, model, 0.3) or near < LOOK_SLACK or not alld):
                clear = False
            if b == "red":
                if is_model(info, model, 0.3) and info["d"] < DESK_SURE_D and info.get("model_margin", 0.0) >= DESK_SURE_MARGIN:
                    self.request_state["in_red"] = True
                    log("t=%d the target is already in red" % self.t())
                    return True
                continue
            new = self.new_in.get(b)
            is_new = new is not None and math.hypot(d["grasp_xy"][0] - new[0], d["grasp_xy"][1] - new[1]) < 0.08
            could = (info.get("all") or {}).get(model, 9.0) - info["d"] < LOOK_SLACK
            if is_model(info, model, 0.3) and info["d"] < IN_CONTAINER_SURE_D:
                self.request_state["seen_target"] = True
            if is_model(info, model, 0.3) or is_new or (info["d"] > IN_CONTAINER_UNSURE and could):
                # it, something new, or something unidentified that may be it: out onto the desk, looked at alone
                g = self.plan_grasps(d, inside=b)
                if not g:
                    self.unknown[b] = self.unknown.get(b) or "seen"
                    self.seen_t[b] = self.t()
                    continue
                held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"]))
                if held:
                    e = self.entry_near(b, d["grasp_xy"], r=0.05)
                    if e is not None:
                        self.remove_entry(e)
                    yield from self.set_down_and_identify(b, info)
                return True
        if b != "red" and clear and rs is not None:
            # everything unexplained in there is clearly something else: an unexplained drop into this
            # container does not keep the requested model from being declared absent
            rs.setdefault("clear_of", {})[b] = self.t()
            if self.unknown.get(b) == "seen":
                log("t=%d %s: nothing there could be the %s" % (self.t(), b, model))
        return False

    def confirm_red(self, model):
        """Look into red from above: is the object there the target? -> "yes", "no", "unsure", "empty"."""
        b = self.containers["red"]
        x, y = b[0], b[1]
        yield from self.goto((x, y, CONTAINER_RIM_Z + 0.16), 0.0, grip=0.0, tol=0.01, max_steps=30)
        yield from self.hold(2, grip=0.0)
        R_cam, t_cam = self.cam_pose()
        rgb = decode_png(self.obs["rgb"]["eye_in_hand"], "RGB")
        dep = decode_png(self.obs["depth"]["eye_in_hand"])
        d = perceive_eih(rgb, dep, R_cam, t_cam, (x, y), CONTAINER_FLOOR_Z, CONTAINER_RIM_Z, self.containers, radius=0.14)
        self.dump("confirm_red", rgb, [d] if d else [])
        if d is None or container_of_xy(self.containers, d["centroid"][0], d["centroid"][1]) != "red":
            return "empty"
        info = clamp_by_size(self.classify_eih(d), d["dims"])
        log("t=%d red holds %s (d=%.2f, margin %.2f) for %s" % (self.t(), info["model"], info["d"], info.get("model_margin", 0.0), model))
        if is_model(info, model, 0.35):
            return "yes"
        if info["d"] < DESK_SURE_D and info.get("model_margin", 0.0) >= 0.15:
            return "no"           # a clear view of something else (a wrong mark costs a point)
        return "unsure"

    def clear_blocker(self, d, b, clusters):
        """Something rests on the object the controller wants (d["blocker"]): take it off first.
        -> True when the arm did something."""
        blk = d["blocker"]
        g = self.plan_grasps(blk, inside=b)
        log("t=%d stack: taking %s off %s in %s first; grasps=%s" % (
            self.t(), blk["cls_info"]["model"], d["cls_info"]["model"], b or "the desk", fmt_plans(g)))
        if not g:
            return False
        rs = self.request_state
        failed = rs["failed"] if rs is not None else self.explore_failed
        fails = sum(1 for (fb, fx, fy) in failed if fb == b and math.hypot(fx - blk["grasp_xy"][0], fy - blk["grasp_xy"][1]) < 0.06)
        if fails >= UNPACK_FAILS_MAX:
            return False
        held = yield from self.try_grasps(g, blk["top_z"], blk["floor"], tuple(blk["grasp_xy"]), keep_top=True)
        if not held:
            failed.extend([(b, blk["grasp_xy"][0], blk["grasp_xy"][1])] * (2 if self.last_grasp_unreachable else 1))
            return True
        if b is not None:
            e = self.entry_near(b, blk["grasp_xy"], r=0.05)
            if e is not None:
                self.remove_entry(e)
            yield from self.set_down_and_identify(b, blk["cls_info"])
        else:
            spot = self.free_spot(clusters, near=(blk["centroid"][0], blk["centroid"][1]))
            if spot is None:
                yield from self.place_in_container(CONTAINER_OF_CLASS[blk["cls_info"]["cls"]], blk["cls_info"]["model"])
            else:
                yield from self.place_on_desk(spot, self.spot_yaw(spot))
        return True

    # -- tidying ------------------------------------------------------------------------------------
    def tidy(self, clusters, restore_only=False):
        """Sort one loose object into its class container (with restore_only: one the controller set
        down on the desk itself). Objects whose class container is being taken apart wait on the desk,
        and so does what an open request dug out, until its target is found or its absence settled.
        -> True when the arm did something."""
        self.prune_desk_items(clusters)
        rs = self.request_state
        loose = []
        for c in self.loose(clusters):
            if self.skipped(c) or not self.within_reach(c):
                continue
            it = self.desk_item_at(c)
            cls = CLASS_OF.get(it["model"]) if (it is not None and it["model"]) else c["cls_info"]["cls"]
            home = CONTAINER_OF_CLASS[cls]
            if EXPLORE_PILES and not restore_only and (self.exploring.get(home) or self.unknown.get(home) == "pile") \
                    and self.strict_free_slot(home) is None:
                continue           # its container's pile is not taken apart yet and no slot there is free: it waits
            # (before a mark everything goes back: the desk is judged with the request)
            if rs is not None and not (rs.get("in_red") or rs.get("absent_ready") or rs.get("restoring")) \
                    and it is not None and it["t"] >= rs["opened"]:
                continue
            loose.append(c)
        loose.sort(key=lambda c: -c["cls_info"]["class_margin"])
        for c in loose:
            d = yield from self.inspect(c)
            if d is None:
                self.skip(c)
                return True
            info = d["cls_info"]
            it = self.desk_item_at(c)
            if it is not None and it["model"] and info["d"] > DESK_SURE_D:
                # identified alone earlier and seen worse now (a neighbour in view): keep the earlier name
                info = dict(info, model=it["model"], cls=CLASS_OF[it["model"]], class_margin=max(info["class_margin"], 0.5))
            if desk_sure(info):
                self.note_loose(info["model"], d["grasp_xy"])
            g = self.plan_grasps(d)
            log("t=%d loose %s (%s d=%.2f margin=%.2f; static %s%s) at (%.2f,%.2f) top=%.3f grasps=%s" % (
                self.t(), info["model"], info["cls"], info["d"], info["class_margin"], c["cls_info"]["model"],
                "; noted %s" % it["model"] if it else "", d["grasp_xy"][0], d["grasp_xy"][1], d["top_z"], fmt_plans(g)))
            if not desk_sure(info) and not (it is not None and it["model"]) and info["class_margin"] >= 0.08:
                # a second, closer look from beside the first before the object goes in unnamed: an
                # unnamed entry keeps every model of its class from being declared absent
                d2 = yield from self.inspect(c, hover=0.11, offset=(0.03, 0.0))
                if d2 is not None and math.hypot(d2["centroid"][0] - d["centroid"][0], d2["centroid"][1] - d["centroid"][1]) < 0.04:
                    i2 = d2["cls_info"]
                    log("t=%d second look: %s (d=%.2f, margin %.2f)" % (self.t(), i2["model"], i2["d"], i2.get("model_margin", 0.0)))
                    if desk_sure(i2) or (i2["model"] == info["model"] and max(i2["d"], info["d"]) < DESK_SURE_D
                                         and min(i2.get("model_margin", 0.0), info.get("model_margin", 0.0)) >= 0.1):
                        info = dict(i2, model_margin=max(i2.get("model_margin", 0.0), DESK_SURE_MARGIN), d=min(i2["d"], info["d"]))
                        d = dict(d, cls_info=info)
            if rs is not None and rs.get("in_red") and not rs.get("marked") and is_model(info, rs["model"]):
                # the target lies here, not in red: it fell out after the release (a long clamp bounced
                # over the rim once, and the unsure red check was trusted). Deliver it again instead
                # of marking.
                log("t=%d the %s lies loose at (%.2f,%.2f): not in red after all" % (
                    self.t(), rs["model"], d["grasp_xy"][0], d["grasp_xy"][1]))
                rs["in_red"] = False
                rs["trust_red"] = False
                rs["seen_target"] = True
                rs["target_at"] = (float(d["grasp_xy"][0]), float(d["grasp_xy"][1]))
                rs["inspected"] = [k for k in rs["inspected"] if math.hypot(k[0] - d["centroid"][0], k[1] - d["centroid"][1]) > 0.05]
                return True
            if rs is not None and not rs.get("in_red") and not rs.get("absent_ready") and is_model(info, rs["model"]):
                rs["seen_target"] = True
                rs["target_at"] = (float(d["grasp_xy"][0]), float(d["grasp_xy"][1]))
                held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"])) if g else False
                if not held:
                    yield from self.topple_if_tall(d)
                if held:
                    if it is not None and it in self.desk_items:
                        self.desk_items.remove(it)
                    rs["trust_red"] = desk_sure(info)
                    yield from self.deliver()
                return True
            if d.get("blocker") and info["class_margin"] >= 0.08:
                acted = yield from self.clear_blocker(d, None, clusters)
                if acted:
                    return True
            if info["class_margin"] < 0.08 or not g:
                pushed = False
                if not g and info["class_margin"] >= 0.08:
                    pushed = yield from self.topple_if_tall(d)
                if not pushed:
                    self.skip(c)
                return True
            held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"]))
            if held:
                if it is not None and it in self.desk_items:
                    self.desk_items.remove(it)
                sure = (it is not None and it["model"] is not None) or desk_sure(info)
                home = CONTAINER_OF_CLASS[info["cls"]]
                slot = self.strict_free_slot(home) if (EXPLORE_PILES and (self.exploring.get(home) or self.unknown.get(home) == "pile")) else None
                if slot is None and self.unknown.get(home) == "pile" and self.exploring.get(home):
                    self.exploring[home] = False       # put back onto the pile before a mark: the dig starts over later
                ok = yield from self.place_in_container(home, info["model"] if sure else None, clusters, slot=slot)
                if ok and not sure:
                    e = self.inventory[CONTAINER_OF_CLASS[info["cls"]]][-1]
                    e["guess"] = info["model"]
            else:
                log("t=%d grasp failed" % self.t())
                pushed = yield from self.topple_if_tall(d)
                if not pushed:
                    self.skip(c, dur=SKIP_AFTER_FAIL)
            return True
        if restore_only:
            return False
        # misfiled objects seen by the static cameras (the world drops objects into wrong containers)
        for c in clusters:
            if c["cls_info"] is None or self.skipped(c) or c["container"] not in CLASS_OF_CONTAINER:
                continue
            if self.entry_near(c["container"], c["grasp_xy"], r=0.04):
                continue
            if not (c["cls_info"]["cls"] != CLASS_OF_CONTAINER[c["container"]] and c["cls_info"]["class_margin"] > 0.4):
                continue
            d = yield from self.inspect(c)
            if d is None or d["cls_info"]["cls"] == CLASS_OF_CONTAINER[c["container"]] or d["cls_info"]["class_margin"] < 0.4 \
                    or d["cls_info"]["d"] > IN_CONTAINER_SURE_D:
                self.skip(c)
                return True
            g = self.plan_grasps(d, inside=c["container"])
            log("t=%d misplaced %s in %s grasps=%s" % (self.t(), d["cls_info"]["model"], c["container"], fmt_plans(g)))
            if not g:
                self.skip(c)
                return True
            held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"]))
            if held:
                yield from self.set_down_and_identify(c["container"], d["cls_info"])
            else:
                self.skip(c)
            return True
        # an object in red that no request put there (dropped on the way over it)
        for c in clusters:
            if c["container"] != "red" or c["n"] < 15 or self.skipped(c) or c["grasp_xy"][0] < RED_REACH_X:
                continue
            if rs is not None and (rs.get("in_red") or rs.get("placed_red")):
                continue
            d = yield from self.inspect(c)
            if d is None:
                self.skip(c)
                return True
            g = self.plan_grasps(d, inside="red")
            if not g:
                self.skip(c)
                return True
            held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"]))
            if held:
                yield from self.set_down_and_identify("red", d["cls_info"])
            else:
                self.skip(c)
            return True
        return False

    # -- taking the initial piles apart -------------------------------------------------------------
    def explore(self, clusters):
        """Idle time: take an unidentified initial pile apart, one container at a time: every object a
        grasp reaches onto the desk (identified there, alone; the tidying puts them back slot by slot
        once the pile is done), what no grasp reaches identified from above where it lies.
        -> True when the arm did something."""
        if EXPLORE_LITE and not EXPLORE_PILES and any(it["src"] in ("gray", "lightblue") for it in self.desk_items):
            return False            # one object out at a time: the tidying puts it back first
        for b in ("gray", "lightblue"):
            if self.unknown.get(b) != "pile":
                self.exploring[b] = False
                continue
            if not self.exploring.get(b):
                if any(self.exploring.values()):
                    continue
                self.exploring[b] = True
                log("t=%d taking the %s pile apart" % (self.t(), b))
            res = yield from self.dig_one(b, clusters, explore=True)
            if res == "done" or res is False:
                self.exploring[b] = False
                log("t=%d %s pile done (%s)" % (self.t(), b, self.unknown.get(b)))
                if self.unknown.get(b) == "pile":
                    self.unknown[b] = "stuck"
                return True
            return bool(res)
        return False

    def patrol(self):
        """Look into each class container from above and account for what is there (account_for).
        At most one object is moved per patrol."""
        self.last_patrol_t = self.t()
        order = sorted(b for b in self.check_soon if b in CLASS_OF_CONTAINER) + \
            [b for b in ("gray", "lightblue", "blue") if b not in self.check_soon]
        self.check_soon = set()
        for b in order:
            moved = yield from self.account_for(b)
            if moved:
                self.last_patrol_t = -10 ** 9          # look again soon: more may have changed
                return

    def account_for(self, b):
        """Look into container b from above and account for what is there: a component no entry
        explains is identified from above when that is sure (and noted as an entry, or taken out when
        it is misfiled), otherwise it is taken out onto the desk and identified there (the tidying puts
        it where it belongs). What is higher than when the controller last left the container (the
        world drops objects into wrong containers) is always taken out. An initial pile is left to the
        exploration; only its top is checked for a misfiled object. -> True when an object was moved."""
        comps = yield from self.container_components(b)
        comps = [c for c in comps if c["n"] >= 40]
        new = self.new_in.pop(b, None)
        pile = self.unknown.get(b) in ("pile", "stuck")
        unexplained = [c for c in comps if not self.entry_near(b, c["grasp_xy"], r=0.04)]
        if new is not None:
            cands = sorted(unexplained or comps, key=lambda c: math.hypot(c["grasp_xy"][0] - new[0], c["grasp_xy"][1] - new[1]))[:1]
        elif pile:
            cands = sorted(unexplained, key=lambda c: -c["top_z"])[:1]
        else:
            cands = sorted(unexplained, key=lambda c: -c["top_z"])
        settled = True
        for c in cands:
            if self.skipped(c):
                settled = False
                continue
            d = yield from self.inspect(c)
            if d is None:
                continue
            info = d["cls_info"]
            sure = info["d"] < IN_CONTAINER_SURE_D and info.get("model_margin", 0.0) > 0.25
            misfiled = sure and info["cls"] != CLASS_OF_CONTAINER[b]
            if sure and not misfiled and new is None:
                if not pile and not self.entry_near(b, d["grasp_xy"], r=0.04):
                    self.inventory[b].append({"model": info["model"], "slot": None, "xy": tuple(d["grasp_xy"]), "t": self.t(), "name": b})
                continue
            if pile and new is None and not misfiled:
                continue
            g = self.plan_grasps(d, inside=b)
            log("t=%d account for %s: %s %s (d=%.2f) grasps=%s" % (self.t(), b, "new" if new is not None else ("misfiled" if misfiled else "unidentified"),
                info["model"], info["d"], fmt_plans(g)))
            if not g:
                settled = False
                continue
            held = yield from self.try_grasps(g, d["top_z"], d["floor"], tuple(d["grasp_xy"]))
            if held:
                e = self.entry_near(b, d["grasp_xy"], r=0.05)
                if e is not None:
                    self.remove_entry(e)
                yield from self.set_down_and_identify(b, info)
            else:
                self.skip(c, dur=SKIP_AFTER_FAIL)
            return True
        if not pile and settled and self.unknown.get(b) in ("seen", "lost"):
            log("t=%d %s: everything there accounted for" % (self.t(), b))
            self.unknown[b] = None
        return False


# ----------------------------------------------------------------------------------------
# Wire
# ----------------------------------------------------------------------------------------
def load_refs():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "refs.json")) as f:
        return json.load(f)


def main():
    ctl = Controller(load_refs())
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        kind = msg.get("type")
        if kind == "reset":
            ctl.reset()
            print(json.dumps({"ok": True}), flush=True)
        elif kind == "act":
            try:
                a = ctl.act(int(msg.get("t", 0)), msg["obs"])
            except Exception as exc:  # noqa: BLE001 - never break the wire
                log("error at t=%s: %r" % (msg.get("t"), exc))
                a = [0.0] * 8
            print(json.dumps({"action": a}), flush=True)
        elif kind == "close":
            print(json.dumps({"ok": True}), flush=True)
            return
        else:
            print(json.dumps({"error": "unknown request"}), flush=True)


if __name__ == "__main__":
    main()
