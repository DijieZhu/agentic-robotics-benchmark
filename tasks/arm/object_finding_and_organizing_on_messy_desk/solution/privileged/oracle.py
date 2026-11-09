#!/usr/bin/env python3
"""Privileged reference controller for object_finding_and_organizing_on_messy_desk.

This program exists to show that the hidden episodes can be done. It plans from the ground truth
the grader adds to every observation of a submission that carries the reference token
(`obs["privileged"]`: every object's model, pose and state as the simulator scores it, the
containers' poses, the open request's object, the true proprioception). An agent never gets
that block; it works from the cameras. Without the block this controller holds still.

What it does, every decision from the truth:
- a request for an absent model: the desk is put in order, then "no such object";
- a request for a present object: whatever lies on it or in the way of the hand comes off first
  (into another place in its own container, or onto the desk), then the object goes into red;
  the desk is put in order (what the dig moved goes back), then "delivered";
- otherwise: loose and misfiled objects into their class containers, misfiled first, nearest
  first; a container spot is chosen on the truth's height map, where a straight-down hand can
  take the object out again;
- grasps from the object's own mesh (geometry.json: the stock YCB meshes, thinned) at its true
  pose: fingers across a side narrower than the opening at the pads' height, the open fingers
  and the palm landing on nothing, the hand clear of the container walls (leaning when a
  straight-down hand does not fit); a tall bottle standing on the desk is pushed over first.

The motion primitives (servo, guarded descent, the slow first lift, carrying, release just
above the surface) and the hand geometry follow the task's perception reference (oracle v17,
solution/reference_v17/), which was tuned on this simulator.

Every reply is computed from the observation it answers: the servo reads the observation's
proprioception, the decisions read the privileged block (the same in both queries of a
repeated `t`), so a repeated query never changes what the controller decides.
"""
import json
import math
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------------------
# Task geometry (task_card.md, env.py)
# ----------------------------------------------------------------------------------------
CLASSES = {
    "A": ["003_cracker_box", "004_sugar_box", "008_pudding_box", "009_gelatin_box", "036_wood_block", "061_foam_brick"],
    "B": ["002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can", "010_potted_meat_can",
          "006_mustard_bottle", "021_bleach_cleanser"],
    "C": ["011_banana", "024_bowl", "025_mug", "035_power_drill", "037_scissors", "040_large_marker",
          "051_large_clamp", "052_extra_large_clamp", "019_pitcher_base"],
}
CLASS_OF = {m: c for c, ms in CLASSES.items() for m in ms}
CONTAINER_OF_CLASS = {"A": "gray", "B": "lightblue", "C": "blue"}
CLASS_CONTAINERS = ("gray", "lightblue", "blue")
HOLLOW = ("024_bowl", "025_mug", "019_pitcher_base")
# round cross-sections, where an off-centre pinch is a chord the closing fingers push the object out of
ROUND = ("002_master_chef_can", "005_tomato_soup_can", "007_tuna_fish_can", "006_mustard_bottle", "021_bleach_cleanser",
         "024_bowl", "025_mug", "019_pitcher_base", "040_large_marker", "011_banana")
WIDE_HOLLOW = ("024_bowl", "025_mug")      # wider than the opening: only a rim grasp holds them
OUTER_HALF = {"gray": (0.1632, 0.1360), "lightblue": (0.1632, 0.1360), "blue": (0.1632, 0.1360), "red": (0.1187, 0.1360)}
INNER_HALF = {"gray": (0.1397, 0.1203), "lightblue": (0.1397, 0.1203), "blue": (0.1397, 0.1203), "red": (0.1016, 0.1203)}
RIM_Z = 0.1206
# The physical support everywhere: the table's collider top is at z = 0.0153 (a raycast; the task card's
# visual desk height is -0.0014), and the containers stand sunk into it, so their effective floor is the same
# table top (a ray down the middle of a container hits the table). Objects rest there.
DESK_Z = FLOOR_Z = 0.0153
REACH, BASE_CLEAR = 0.80, 0.36
BASE_RESCUE = 0.26
WS_X, WS_Y, TCP_Z_MIN = (-0.1, 0.85), (-0.75, 0.75), 0.028
POS_STEP_MAX, ROT_STEP_MAX = 0.05, 0.25

# hand (Franka, fingers open), measured on the stage in the tool-centre frame (the frame ee_pose reports; the
# collision meshes of panda_hand and the two fingers): the body 0.205 x 0.063 with its underside 0.034 above the
# tool centre; each finger 0.021 wide (along the thickness axis), its inner face 0.040 off the centre when open,
# 0.026 thick outward, its tip 0.0123 under the tool centre (the env's floor TCP z 0.028 = the desk 0.0153 + 0.0123)
HAND_HALF_W, HAND_HALF_T = 0.1025, 0.0315
HAND_BODY_ABOVE_TCP = 0.034
FINGER_TIP_BELOW_TCP = 0.0123
FINGER_BELOW_TCP = FINGER_TIP_BELOW_TCP   # (tilted grasps: the TCP is shifted against the lean by this much)
FINGER_HALF_WIDTH, FINGER_THICK = 0.0105, 0.0264
# the finger's profile (its collision mesh in franka.usd): 1.57 cm thick from the inner face over its lower 2 cm
# (the tip up to 6.7 mm above the tool centre), 2.64 cm only at its base near the hand
FINGER_TIP_THICK, FINGER_TIP_TOP = 0.0157, 0.0067
OPEN_W = 0.080
WIDTH_MARGIN = 0.010                 # the grasped width stays this far under the opening
GRASP_BELOW_TOP = 0.030              # the tool centre never deeper than this under the top (the palm lands on it)
TIP_CLEAR = 0.003                    # the fingertips this far above what the object rests on
RIM_BELOW_TOP = 0.022
RIM_CLEAR = 0.012
GRIP_SETTLE_MAX = 12
TRAVEL_Z = 0.38
CARRY_STEP = 0.035
PLACE_GAP = 0.012                    # the held object's bottom this far above what is under it at release
HOME_POS = (0.46, 0.0, 0.39)
IDLE_POS = (0.42, 0.0, 0.40)
STUCK_ERR, STUCK_AFTER = 0.10, 4
TILTS = [(0.0, 0.0)] + [(sg * a, 0.0) for a in (0.30, 0.45) for sg in (1, -1)] \
    + [(0.0, sg * b) for b in (0.30, 0.45) for sg in (1, -1)]
TOPPLE_MIN_H = 0.105                 # a bottle standing this tall is pushed over before the grasp
TOPPLE_MODELS = ("006_mustard_bottle", "021_bleach_cleanser")
MARK_MARGIN = 30                     # steps before the deadline at which the mark goes out whatever is going on
TIDY_BEFORE_MARK = True
FAIL_COOLDOWN = 250                  # steps a failed object is left alone (x the number of failures)
PREPARE_EVERY = 300                  # steps between idle looks for objects no hand could take out
# Idle-time work on the containers (off): sliding wall-bound objects inward and digging every blocked object
# free before any request names it. Measured on the practice seeds' initial piles, neither made a blocked pile
# object retrievable (the slides do not move an object wedged at a wall) and both spent hundreds of steps.
IDLE_DRAG, IDLE_RESTRUCTURE = False, False
LOG = os.environ.get("MESSY_DESK_ORACLE_LOG", "1") != "0"


def log(*a):
    if LOG:
        print("[oracle]", *a, file=sys.stderr, flush=True)


# ----------------------------------------------------------------------------------------
# Small maths (as in the perception reference)
# ----------------------------------------------------------------------------------------
def quat_to_mat(q):
    w, x, y, z = [float(v) for v in q]
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def wrap_half(a):
    return (a + math.pi / 2) % math.pi - math.pi / 2


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


def axis_rot(axis, a):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * K @ K


def is_tilted(tilt):
    return tilt is not None and (abs(tilt[0]) > 1e-3 or abs(tilt[1]) > 1e-3)


def tilt_rot(yaw, tilt):
    u = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    v = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return axis_rot(v, tilt[0]) @ axis_rot(u, tilt[1])


def approach_dir(yaw, tilt):
    R = tilt_rot(yaw, tilt) if is_tilted(tilt) else np.eye(3)
    return -(R @ np.array([0.0, 0.0, 1.0]))


def yaw_branch(xy, yaw, tilt=None):
    """Of the two equivalent hand yaws (yaw, yaw + pi), the one that keeps the wrist's twist small
    relative to the arm's azimuth (the perception reference's rule: the other ran a joint into its limit)."""
    phi = math.atan2(xy[1], xy[0])
    y2 = wrap(phi + wrap_half(yaw - phi))
    if is_tilted(tilt) and abs(wrap(y2 - yaw)) > 1.0:
        tilt = (-tilt[0], -tilt[1])
    return y2, tilt


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def yaw_quat(a):
    return [math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)]


def container_local(b, x, y):
    bx, by, yaw = b
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = x - bx, y - by
    return c * dx + s * dy, -s * dx + c * dy


def container_point(b, lx, ly):
    bx, by, yaw = b
    c, s = math.cos(yaw), math.sin(yaw)
    return bx + c * lx - s * ly, by + s * lx + c * ly


def footprint_dist(containers, name, x, y):
    lx, ly = container_local(containers[name], x, y)
    ox, oy = OUTER_HALF[name]
    return math.hypot(max(abs(lx) - ox, 0.0), max(abs(ly) - oy, 0.0))


def _hand_points():
    """Sample points of the open hand in its own frame (finger axis, thickness axis, up), metres from the
    tool centre point: the body, the two fingers, the wrist above (the perception reference's model)."""
    pts = []
    for su in (-1.0, -0.5, 0.0, 0.5, 1.0):
        for sv in (-1.0, 0.0, 1.0):
            for h in (HAND_BODY_ABOVE_TCP, HAND_BODY_ABOVE_TCP + 0.04, HAND_BODY_ABOVE_TCP + 0.08):
                pts.append((su * HAND_HALF_W, sv * HAND_HALF_T, h))
    for su in (-1.0, 1.0):
        for sv in (-1.0, 1.0):
            for h in (-FINGER_TIP_BELOW_TCP, -0.003, FINGER_TIP_TOP):
                pts.append((su * (OPEN_W / 2 + FINGER_TIP_THICK), sv * FINGER_HALF_WIDTH, h))
            for h in (0.02, HAND_BODY_ABOVE_TCP):
                pts.append((su * (OPEN_W / 2 + FINGER_THICK), sv * FINGER_HALF_WIDTH, h))
    for su in (-1.0, 0.0, 1.0):
        pts.append((su * 0.045, 0.0, 0.17))
    return np.array(pts)


HAND_PTS = _hand_points()


def hand_fits3d(containers, p, yaw, tilt=(0.0, 0.0), inside=None):
    """The open hand at p with `yaw` and `tilt` clears every container wall below its rim (inside the
    interior of `inside`, outside every other footprint); the wrist stays over `inside`'s interior."""
    u = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    v = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    B = np.stack([u, v, np.array([0.0, 0.0, 1.0])], 1)
    R = tilt_rot(yaw, tilt) if is_tilted(tilt) else np.eye(3)
    W = np.asarray(p, float) + (R @ B @ HAND_PTS.T).T
    below = W[:, 2] < RIM_Z + RIM_CLEAR
    for name, b in containers.items():
        ox, oy = OUTER_HALF[name]
        ix, iy = INNER_HALF[name]
        lx, ly = container_local(b, W[:, 0], W[:, 1])
        in_outer = (np.abs(lx) <= ox + 0.004) & (np.abs(ly) <= oy + 0.004)
        if name == inside:
            in_inner = (np.abs(lx) <= ix - 0.004) & (np.abs(ly) <= iy - 0.004)
            if np.any(below & ~in_inner):
                return False
            if np.any(np.abs(lx[-3:]) > ix) or np.any(np.abs(ly[-3:]) > iy):
                return False
        elif np.any(below & in_outer):
            return False
    return bool(np.all(W[:, 2] > -0.004))


def fits_with_approach(containers, p, yaw, tilt, inside):
    if not hand_fits3d(containers, p, yaw, tilt, inside):
        return False
    return hand_fits3d(containers, np.asarray(p, float) - 0.04 * approach_dir(yaw, tilt), yaw, tilt, inside)


def reachable_xy(x, y, margin=0.0):
    r = math.hypot(x, y)
    return (BASE_CLEAR - 0.04 + margin <= r <= REACH + 0.01 - margin and WS_X[0] < x < WS_X[1] - margin
            and WS_Y[0] + margin < y < WS_Y[1] - margin)


def load_geometry():
    with open(os.path.join(HERE, "geometry.json")) as f:
        g = json.load(f)
    return {m: np.array(d["points"], float) for m, d in g.items()}


GEOM = None


# ----------------------------------------------------------------------------------------
# The world as the privileged block states it
# ----------------------------------------------------------------------------------------
class World(object):
    def __init__(self, blk):
        self.t = int(blk["t"])
        self.objs = {int(o["id"]): o for o in blk["objects"] if o.get("pos") is not None and o.get("quat") is not None}
        self.request = blk.get("request")
        self.containers = {n: (float(c["xy"][0]), float(c["xy"][1]), float(c["yaw"])) for n, c in blk["containers"].items()}
        self.ee = np.array(blk["ee_pose"][:3], float)
        self.ee_q = np.array(blk["ee_pose"][3:7], float)
        self.width = float(blk["gripper_width"])
        self._pts = {}

    def target_id(self):
        r = self.request
        return None if r is None or r.get("object") is None else int(r["object"])

    def points(self, oid):
        p = self._pts.get(oid)
        if p is None:
            o = self.objs[oid]
            R = quat_to_mat(o["quat"])
            p = (R @ (GEOM[o["model"]] * float(o["scale"])).T).T + np.asarray(o["pos"], float)
            self._pts[oid] = p
        return p

    def home_of(self, oid):
        return "red" if oid == self.target_id() else CONTAINER_OF_CLASS[self.objs[oid]["cls"]]

    def done(self, oid):
        o = self.objs[oid]
        return o.get("state") == "in_container" and o.get("in_container") == self.home_of(oid)

    def container_at(self, x, y, shrink=0.0):
        for name, b in self.containers.items():
            lx, ly = container_local(b, x, y)
            ix, iy = INNER_HALF[name]
            if abs(lx) <= ix - shrink and abs(ly) <= iy - shrink:
                return name
        return None

    def where(self, oid):
        """The container the object lies in (by its centre, as the simulator judges), else None."""
        o = self.objs[oid]
        if o.get("in_container"):
            return o["in_container"]
        p = o["pos"]
        return self.container_at(p[0], p[1], shrink=-0.01) if p[2] < RIM_Z + 0.12 else None

    def others(self, oid, near_xy=None, radius=0.30, exclude=()):
        pts = []
        for k in self.objs:
            if (oid is not None and k == oid) or k in exclude:
                continue
            p = self.objs[k]["pos"]
            if near_xy is not None and math.hypot(p[0] - near_xy[0], p[1] - near_xy[1]) > radius:
                continue
            pts.append((k, self.points(k)))
        return pts


class FakeWorld(World):
    """A hypothetical world for the planner: the given objects at the given points, nothing else."""

    def __init__(self, containers, objs, pts):
        self.t = 0
        self.objs = objs
        self.request = None
        self.containers = containers
        self._pts = dict(pts)


# ----------------------------------------------------------------------------------------
# Grasp planning on the true geometry
# ----------------------------------------------------------------------------------------
def _span(P, z_g):
    """The object's points between the fingertips and the palm: what the closing fingers meet."""
    lo, hi = z_g - FINGER_TIP_BELOW_TCP - 0.002, z_g + HAND_BODY_ABOVE_TCP
    return P[(P[:, 2] >= lo) & (P[:, 2] <= hi)]


def hand_hits(Q, hx, hy, yaw, z_g):
    """Which of the points Q the open hand would touch coming down to (hx, hy, z_g): the two finger
    columns down to the fingertips, the gap between the fingers (another object would be grasped with
    it), and the palm's footprint above its underside. -> boolean mask"""
    u = np.array([-math.sin(yaw), math.cos(yaw)])
    v = np.array([math.cos(yaw), math.sin(yaw)])
    rel = Q[:, :2] - (hx, hy)
    pu, pv = rel @ u, rel @ v
    au, av = np.abs(pu), np.abs(pv)
    tip = z_g - FINGER_TIP_BELOW_TCP
    thick = np.where(Q[:, 2] < z_g + FINGER_TIP_TOP + 0.005, FINGER_TIP_THICK, FINGER_THICK)
    fingers = (av < FINGER_HALF_WIDTH + 0.004) & (au > OPEN_W / 2 - 0.004) & (au < OPEN_W / 2 + thick + 0.004) \
        & (Q[:, 2] > tip - 0.002)
    palm = (au < HAND_HALF_W + 0.004) & (av < HAND_HALF_T + 0.004) & (Q[:, 2] > z_g + HAND_BODY_ABOVE_TCP - 0.004)
    gap = (au <= OPEN_W / 2 - 0.004) & (av < FINGER_HALF_WIDTH + 0.004) & (Q[:, 2] > tip - 0.002) \
        & (Q[:, 2] < z_g + HAND_BODY_ABOVE_TCP)
    return fingers | palm | gap


def grasp_height(top, bottom, ground):
    """Tool-centre height for a pinch: the fingertips just above what the object rests on, but the palm
    never lower than 4 mm above its top (for a tall object the fingers hold its top 4 cm)."""
    return max(top - GRASP_BELOW_TOP, bottom + FINGER_TIP_BELOW_TCP + TIP_CLEAR,
               ground + FINGER_TIP_BELOW_TCP + TIP_CLEAR, TCP_Z_MIN)


class Plan(object):
    __slots__ = ("xy", "yaw", "z", "tilt", "cost", "kind", "width", "blockers")

    def __init__(self, xy, yaw, z, tilt, cost, kind="pinch", width=0.0, blockers=()):
        self.xy, self.yaw, self.z, self.tilt, self.cost, self.kind, self.width = xy, yaw, z, tilt, cost, kind, width
        self.blockers = tuple(blockers)

    def __repr__(self):
        return "Plan(%s (%.3f,%.3f,%.3f) yaw %.2f tilt %s w %.3f cost %.3f%s)" % (
            self.kind, self.xy[0], self.xy[1], self.z, self.yaw, "0" if not is_tilted(self.tilt) else
            "%+.2f/%+.2f" % tuple(self.tilt), self.width, self.cost, (" blockers %s" % (self.blockers,)) if self.blockers else "")


def area_centroid(P, cell=0.005):
    """The centre of the object's footprint (its points' 5 mm cells, each counted once): a proxy for the centre
    of mass that the uneven density of a scanned mesh's vertices does not pull toward its detailed parts."""
    c = np.unique(np.floor(P[:, :2] / cell).astype(np.int64), axis=0)
    return (c.mean(0) + 0.5) * cell


def _width_at(S, cen, u, v, voff, half=FINGER_HALF_WIDTH):
    sel = np.abs((S[:, :2] - cen) @ v - voff) < half
    if sel.sum() < 4:
        return None
    bu = (S[sel, :2] - cen) @ u
    return float(bu.min()), float(bu.max())


def _count(why, key):
    if why is not None:
        why[key] = why.get(key, 0) + 1


def plan_grasps(w, oid, fails=0, ignore_others=False, why=None, avoid_yaws=()):
    """Feasible pinch grasps of object `oid`, best first: the fingers across a side that is narrower than
    the opening over the span between the fingertips and the palm, on parallel faces (for a round object
    not on a chord, where the closing fingers push it out), the open fingers and the palm landing on nothing
    (the fingertips stop above what lies under them, as long as enough of the object stays between them),
    the hand clear of the container walls (leaning if it must). A shallower grasp (the object's top 2.4 or
    3.2 cm) is tried only when the full one finds nothing. With ignore_others, the plans that only other
    objects rule out are kept, each with the objects in its way (`blockers`)."""
    P = w.points(oid)
    o = w.objs[oid]
    inside = w.where(oid)
    ground = FLOOR_Z if inside else DESK_Z
    top, bottom = float(P[:, 2].max()), float(P[:, 2].min())
    cen = area_centroid(P)
    others = w.others(oid, near_xy=cen, radius=0.30)
    Q = np.concatenate([q for _, q in others], 0) if others else np.zeros((0, 3))
    owner = np.concatenate([np.full(len(q), k) for k, q in others]) if others else np.zeros(0, int)
    round_ = o["model"] in ROUND
    upright_hollow = o["model"] in HOLLOW and (quat_to_mat(o["quat"]) @ np.array([0.0, -1.0, 0.0]))[2] > 0.85
    wide_hollow = upright_hollow and o["model"] in WIDE_HOLLOW
    z0 = grasp_height(top, bottom, ground)
    heights = [z0] + [z for z in (top - 0.020, top - 0.012) if z > z0 + 0.004]
    # the yaws: the object's own footprint axes first (a box's edges, exactly: 10 degrees off, a 6.3 x 8.3 cm box is
    # 7.6 cm across the fingers), then a 10-degree grid
    C = np.cov((P[:, :2] - P[:, :2].mean(0)).T)
    evals, evecs = np.linalg.eigh(C)
    a_major = math.atan2(evecs[1, 1], evecs[0, 1])
    yaws = [wrap_half(a_major), wrap_half(a_major + math.pi / 2)]
    yaws += [y for y in (k * math.pi / 18 - math.pi / 2 for k in range(18))
             if all(abs(wrap_half(y - a)) > 0.03 for a in yaws)]
    plans = []
    for shallow, z_g in enumerate(heights):
        if shallow and plans:
            break
        span = _span(P, z_g)
        if len(span) < 6:
            span = P
        for k, yaw in enumerate(yaws):
            if any(abs(wrap_half(yaw - a)) < 0.30 for a in avoid_yaws):
                continue                 # the arm could not get the hand down this way here
            plan = _plan_at_yaw(w, P, top, bottom, cen, z_g, span, Q, owner, inside, yaw, k, round_, wide_hollow,
                                fails, ignore_others, why, shallow)
            if plan is not None:
                plans.append(plan)
    if upright_hollow:
        plans += rim_plans(w, oid, P, top, inside, Q, owner, ignore_others)
    plans.sort(key=lambda p: (len(p.blockers), p.cost))
    return plans


def _plan_at_yaw(w, P, top, bottom, cen, z_g, span, Q, owner, inside, yaw, k, round_, wide_hollow, fails,
                 ignore_others, why, shallow):
    u = np.array([-math.sin(yaw), math.cos(yaw)])
    v = np.array([math.cos(yaw), math.sin(yaw)])
    sv = (span[:, :2] - cen) @ v
    vlo, vhi = float(sv.min()), float(sv.max())
    # where along the thickness axis the fingers go: as near the centroid as the object allows
    if vhi - vlo > 0.024:
        voffs = sorted(set([0.0] + [float(x) for x in np.arange(-0.04, 0.0401, 0.01)
                                     if vlo + 0.010 <= x <= vhi - 0.010]), key=abs)
    else:
        voffs = [(vlo + vhi) / 2.0]
    for voff in voffs:
        ext = _width_at(span, cen, u, v, voff)
        if ext is None:
            continue
        ulo, uhi = ext
        width = uhi - ulo
        if width + WIDTH_MARGIN > OPEN_W:
            _count(why, "wide")
            continue
        # parallel faces: on a round object the width barely changes across the finger's width (a chord's does)
        if round_:
            side = [_width_at(span, cen, u, v, voff + d, half=0.004) for d in (-0.008, 0.008)]
            if any(e is not None and abs((e[1] - e[0]) - width) > 0.012 for e in side):
                _count(why, "chord")
                continue
        span_off = max(0.0, (OPEN_W / 2 - 0.003) - width / 2.0)     # how far the hand may sit off the middle
        offs = [0.0] + [sg * d for d in np.arange(0.005, span_off + 1e-9, 0.005) for sg in (1.0, -1.0)]
        for off in offs:
            plan = _plan_at_offset(w, P, top, bottom, cen, z_g, Q, owner, inside, yaw, k, u, v, ulo, uhi, voff, off,
                                   width, wide_hollow, fails, ignore_others, why, shallow)
            if plan is not None:
                return plan
    return None


def _plan_at_offset(w, P, top, bottom, cen, z_g, Q, owner, inside, yaw, k, u, v, ulo, uhi, voff, off, width,
                    wide_hollow, fails, ignore_others, why, shallow):
    if True:
        uc = (ulo + uhi) / 2.0 + off
        hx, hy = cen + uc * u + voff * v
        hx, hy = float(hx), float(hy)
        if not reachable_xy(hx, hy):
            _count(why, "reach")
            return None
        # what lies under the finger columns (the object below in a stack, a neighbour beside it) sets how
        # low the fingertips may go; enough of the object's side must stay between the fingers
        zg = z_g
        if len(Q):
            rel = Q[:, :2] - (hx, hy)
            au, av = np.abs(rel @ u), np.abs(rel @ v)
            col = (av < FINGER_HALF_WIDTH + 0.004) & (au > OPEN_W / 2 - 0.004) & (au < OPEN_W / 2 + FINGER_TIP_THICK + 0.004)
            if col.any():
                zr = float(Q[col, 2].max()) + FINGER_TIP_BELOW_TCP + TIP_CLEAR
                if zr > zg + 1e-4:
                    grip = top - (zr - FINGER_TIP_BELOW_TCP)
                    if grip >= min(0.012, 0.6 * (top - bottom)) and zr <= top + 0.004:
                        ext2 = _width_at(_span(P, zr), cen, u, v, voff)
                        if ext2 is not None and ext2[1] - ext2[0] + WIDTH_MARGIN <= OPEN_W:
                            zg = zr
        own = hand_hits(P, hx, hy, yaw, zg)
        own &= ~(np.abs((P[:, :2] - (hx, hy)) @ u) <= OPEN_W / 2 - 0.004)    # what lies between the fingers is the grasp
        if own.sum() > 2:
            _count(why, "own")
            return None
        tilt = next((t for t in TILTS if fits_with_approach(w.containers, (hx, hy, zg), yaw, t, inside)), None)
        if tilt is None:
            _count(why, "walls")
            return None
        blockers = ()
        if len(Q):
            hit = hand_hits(Q, hx, hy, yaw, zg)
            if hit.sum() > 2:
                if not ignore_others:
                    _count(why, "others")
                    return None
                blockers = tuple(sorted(set(int(x) for x in owner[hit])))
        cost = width + 1.5 * math.hypot(hx - cen[0], hy - cen[1]) + 0.002 * k \
            + (0.05 + 0.03 * (abs(tilt[0]) + abs(tilt[1])) if is_tilted(tilt) else 0.0) \
            + (0.10 if wide_hollow else 0.0) + 2.0 * max(0.0, zg - z_g) + 0.03 * shallow
        if fails and (k + fails) % 3 == 0:
            cost -= 0.01                # a retry after a failure prefers another yaw
        cost += 0.3 * abs(off)
        return Plan((hx, hy), yaw, zg, tilt, cost, "pinch", width, blockers)


def rim_plans(w, oid, P, top, inside, Q, owner, ignore_others):
    """Hollow object standing open side up (bowl, mug, pitcher): one finger inside, one outside,
    straddling the wall at the rim, the finger axis radial."""
    out = []
    slab = P[P[:, 2] > top - 0.012]
    if len(slab) < 10:
        return out
    cen = P[:, :2].mean(0)
    rad = np.hypot(slab[:, 0] - cen[0], slab[:, 1] - cen[1])
    radius = float(np.percentile(rad, 80))
    z_g = max(top - RIM_BELOW_TOP, FLOOR_Z + FINGER_TIP_BELOW_TCP + TIP_CLEAR, TCP_Z_MIN)
    for k in range(16):
        th = k * math.pi / 8
        wx, wy = cen[0] + radius * math.cos(th), cen[1] + radius * math.sin(th)
        yaw = wrap_half(th - math.pi / 2)
        if not reachable_xy(wx, wy):
            continue
        if not fits_with_approach(w.containers, (wx, wy, z_g), yaw, (0.0, 0.0), inside):
            continue
        blockers = ()
        if len(Q):
            hit = hand_hits(Q, wx, wy, yaw, z_g)
            if hit.sum() > 2:
                if not ignore_others:
                    continue
                blockers = tuple(sorted(set(int(x) for x in owner[hit])))
        out.append(Plan((float(wx), float(wy)), yaw, z_g, (0.0, 0.0), 0.08 + 0.002 * k, "rim", 0.0, blockers))
    return out


# ----------------------------------------------------------------------------------------
# The controller
# ----------------------------------------------------------------------------------------
class Abort(Exception):
    pass


class Controller(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.gen = None
        self.policy = None
        self.last_t = None
        self.obs = None
        self.w = None
        self.q_home = None
        self.R_home = None
        self.fails = {}            # object id -> (count, last t)
        self.marked = set()        # request ids marked
        self.dug = {}              # object id -> container it was taken out of for a request
        self.carry_hang = 0.0
        self.held_id = None
        self.stuck_count = 0
        self.recovering = False
        self.stuck_recovered = False
        self.toppled = {}          # object id -> times pushed
        self.moved_before_grasp = False
        self.guard = None
        self.dragged = {}          # object id -> times slid off a container wall
        self.prepared_t = -10 ** 9
        self.bad_yaws = {}         # object id -> plan yaws the arm could not bring the hand down at
        self.restructured_t = -10 ** 9
        self.restructure_tries = {}

    # -- wire entry -----------------------------------------------------------------------------
    def act(self, t, obs):
        blk = obs.get("privileged")
        if blk is None:
            return [0.0] * 8
        if self.gen is None:
            self.obs, self.w = obs, World(blk)
            self.q_home = np.array(obs["ee_pose"][3:7], float) if t == 0 else self.q_home_default()
            self.R_home = quat_to_mat(self.q_home)
            self.gen = self.run()
            self.policy = next(self.gen)
        elif t != self.last_t:
            self.obs, self.w = obs, World(blk)
            try:
                self.policy = self.gen.send(obs)
            except Exception as exc:  # noqa: BLE001 - one failed behaviour must not end the episode
                log("t=%d behaviour failed (%r); starting over" % (t, exc))
                import traceback
                traceback.print_exc(file=sys.stderr)
                self.carry_hang = 0.0
                self.gen = self.run()
                self.policy = next(self.gen)
        self.last_t = t
        if LOG and t % 500 == 0 and t != getattr(self, "status_t", None):
            self.status_t = t
            todo = self.todo()
            log("t=%d status: request %s; todo %s" % (t, (self.w.request or {}).get("text"), [
                (k, self.w.objs[k]["model"][4:12], self.w.objs[k].get("state"), self.w.where(k),
                 "cool" if self.cooling(k) else "", "moving" if self.moving(k) else "") for k in todo[:8]]))
        return self.mark_before_deadline([float(x) for x in self.policy(obs)])

    def mark_before_deadline(self, a):
        """The mark goes out before the deadline whatever the arm is in the middle of: with MARK_MARGIN steps
        left, "no such object" for an absent model, "delivered" once the object rests in red."""
        if self.guard is not None and self.guard[0] == self.t():
            a[7] = self.guard[1]           # a repeated query of this step gets the same mark
            return a
        r = self.w.request
        if r is None or r["id"] in self.marked or a[7] != 0:
            return a
        if int(r["deadline_t"]) - self.t() > MARK_MARGIN:
            return a
        tgt = self.w.target_id()
        if tgt is None:
            a[7] = 2.0
        elif tgt in self.w.objs and self.w.done(tgt):
            a[7] = 1.0
        else:
            return a
        self.marked.add(r["id"])
        self.guard = (self.t(), a[7])
        log("t=%d deadline in %d steps: mark %d" % (self.t(), int(r["deadline_t"]) - self.t(), int(a[7])))
        return a

    def q_home_default(self):
        # straight down, fingers along +y (the episode's start pose; used only if the first step was missed)
        return np.array([0.0, 1.0, 0.0, 0.0])

    # -- proprioception (decisions from the truth) ------------------------------------------------
    def ee(self):
        return self.w.ee

    def t(self):
        return self.w.t

    def orient_error(self, yaw, q, tilt=None):
        R_cur = quat_to_mat(q)
        R_des = rot_z(yaw) @ self.R_home
        if is_tilted(tilt):
            R_des = tilt_rot(yaw, tilt) @ R_des
        return rotvec(R_des @ R_cur.T)

    # -- motion primitives -------------------------------------------------------------------------
    def servo(self, pos, yaw, grip, mark=0, step_max=POS_STEP_MAX, tilt=None):
        pos = np.asarray(pos, float)

        def policy(o):
            err = pos - np.array(o["ee_pose"][:3], float)
            d = np.clip(err, -step_max, step_max)
            r = np.clip(self.orient_error(yaw, np.array(o["ee_pose"][3:7], float), tilt), -ROT_STEP_MAX, ROT_STEP_MAX)
            return [d[0], d[1], d[2], r[0], r[1], r[2], grip, mark]
        return policy

    def goto(self, pos, yaw=0.0, grip=0.0, tol=0.01, yaw_tol=0.04, max_steps=60, step_max=POS_STEP_MAX, tilt=None,
             watch=None):
        pos = np.array(pos, float)
        pos[2] = max(pos[2], TCP_Z_MIN)
        for i in range(max_steps):
            err = pos - self.ee()
            if np.linalg.norm(err) < tol and np.linalg.norm(self.orient_error(yaw, self.w.ee_q, tilt)) < yaw_tol:
                self.stuck_count = 0
                return True
            if watch is not None and not watch():
                return False
            yield self.servo(pos, yaw, grip, 0, step_max, tilt)
        if np.linalg.norm(pos - self.ee()) > STUCK_ERR and not self.recovering:
            self.stuck_count += 1
            if self.stuck_count >= STUCK_AFTER:
                log("t=%d arm stuck (%d moves ended far off); recovering" % (self.t(), self.stuck_count))
                self.recovering = True
                try:
                    cur = self.ee()
                    yield from self.goto((cur[0], cur[1], 0.45), 0.0, grip=grip, tol=0.03, yaw_tol=0.3, max_steps=40)
                    yield from self.goto(IDLE_POS, 0.0, grip=grip, tol=0.02, yaw_tol=0.1, max_steps=80)
                finally:
                    self.recovering = False
                    self.stuck_count = 0
                    self.stuck_recovered = True
        return False

    def hold(self, steps, grip, yaw=None, tilt=None, mark=0):
        pos = self.ee().copy()
        yaw = self.hand_yaw() if yaw is None else yaw
        for i in range(steps):
            yield self.servo(pos, yaw, grip, mark if i == 0 else 0, tilt=tilt)

    def hand_yaw(self):
        R = quat_to_mat(self.w.ee_q)
        y_axis = R[:, 1]
        return wrap(math.atan2(y_axis[1], y_axis[0]) - math.pi / 2)

    def mark(self, kind):
        """One step with the mark, the arm where it is, the fingers open."""
        yield from self.hold(1, grip=0.0, mark=kind)
        yield from self.hold(1, grip=0.0)

    def close_fingers(self, yaw, tilt=None):
        last = self.w.width
        for i in range(GRIP_SETTLE_MAX):
            yield from self.hold(1, grip=1.0, yaw=yaw, tilt=tilt)
            wd = self.w.width
            if i >= 2 and abs(wd - last) < 0.0015:
                break
            last = wd

    def travel_z(self):
        return max(TRAVEL_Z, RIM_Z + 0.05 + self.carry_hang)

    def is_held(self, oid):
        o = self.w.objs.get(oid)
        if o is None:
            return False
        if o.get("held"):
            return True
        # still between the fingers: near the tool centre, both fingers touching, the fingers not shut
        p = np.asarray(o["pos"], float)
        return bool(o.get("contact")) and self.w.width > 0.002 and np.linalg.norm(p[:2] - self.ee()[:2]) < 0.08 \
            and p[2] > self.ee()[2] - 0.25

    # -- pick and place --------------------------------------------------------------------------
    def pick(self, oid, plan):
        """Top-down (or leaning) pinch of object `oid` by `plan`. -> True when it is held after the lift."""
        x, y = plan.xy
        yaw, tilt = yaw_branch((x, y), wrap_half(plan.yaw), plan.tilt)
        z_g = plan.z
        P = self.w.points(oid)
        top = float(P[:, 2].max())
        z_pre = max(top + 0.10, z_g + 0.12)
        self.stuck_recovered = False
        z_travel = max(self.travel_z(), z_pre)
        yield from self.goto((x, y, z_travel), yaw, grip=0.0, tol=0.015)
        yield from self.goto((x, y, z_pre), yaw, grip=0.0, tol=0.008, max_steps=30)
        if self.stuck_recovered:
            return False
        # the object may have moved meanwhile (a neighbour settled): re-plan once on the spot
        if oid not in self.w.objs:
            return False
        if np.linalg.norm(self.w.points(oid)[:, :2].mean(0) - P[:, :2].mean(0)) > 0.015:
            log("t=%d object %d moved before the grasp; re-planning" % (self.t(), oid))
            self.moved_before_grasp = True
            return False
        if is_tilted(tilt):
            a = approach_dir(yaw, tilt)
            x, y = x - FINGER_BELOW_TCP * a[0], y - FINGER_BELOW_TCP * a[1]
            back = max(0.06, (RIM_Z + 0.06 - z_g) / max(-a[2], 0.3))
            p_pre = np.array([x, y, z_g]) - a * back
            yield from self.goto(p_pre, yaw, grip=0.0, tol=0.01, yaw_tol=0.06, max_steps=40, tilt=tilt)
            yield from self.goto((x, y, z_g), yaw, grip=0.0, tol=0.006, yaw_tol=0.06, max_steps=30, step_max=0.02, tilt=tilt)
            short = float(np.linalg.norm(self.ee() - np.array([x, y, z_g])))
            if short > 0.015:
                log("t=%d leaning descent stopped %.3f short; not closing" % (self.t(), short))
                if short > 0.05:
                    self.bad_yaws.setdefault(oid, []).append(plan.yaw)
                yield from self.goto(p_pre, yaw, grip=0.0, tol=0.015, yaw_tol=0.08, max_steps=30, step_max=0.03, tilt=tilt)
                yield from self.goto((p_pre[0], p_pre[1], z_travel), yaw, grip=0.0, tol=0.015, max_steps=30, step_max=0.03)
                return False
            yield from self.close_fingers(yaw, tilt)
            yield from self.goto(p_pre, yaw, grip=1.0, tol=0.015, yaw_tol=0.08, max_steps=30, step_max=0.015, tilt=tilt)
            yield from self.goto((p_pre[0], p_pre[1], z_travel), yaw, grip=1.0, tol=0.015, max_steps=30, step_max=0.03)
        else:
            yield from self.goto((x, y, z_g + 0.02), yaw, grip=0.0, tol=0.006, max_steps=25, step_max=0.02)
            yield from self.goto((x, y, z_g), yaw, grip=0.0, tol=0.0025, max_steps=20, step_max=0.01)
            short = float(np.linalg.norm(self.ee() - np.array([x, y, z_g])))
            if short > 0.015:
                log("t=%d descent stopped %.3f short; not closing" % (self.t(), short))
                if short > 0.05:
                    self.bad_yaws.setdefault(oid, []).append(plan.yaw)
                yield from self.goto((x, y, z_travel), yaw, grip=0.0, tol=0.015, max_steps=30, step_max=0.03)
                return False
            reached = self.ee().copy()
            yield from self.close_fingers(yaw)
            yield from self.hold(2, grip=1.0, yaw=yaw)
            closed_w = self.w.width
            ee = self.ee()
            yield from self.goto((x, y, float(ee[2]) + 0.05), yaw, grip=1.0, tol=0.006, max_steps=10, step_max=0.008)
            o_now = self.w.objs.get(oid, {})
            log("t=%d grasp diag %d: tcp target (%.3f,%.3f,%.3f) reached (%.3f,%.3f,%.3f) width closed %.4f now %.4f; "
                "object z %.3f -> %.3f contact %s held %s" % (
                    self.t(), oid, x, y, z_g, reached[0], reached[1], reached[2], closed_w, self.w.width,
                    float(P[:, 2].min()), float(self.w.points(oid)[:, 2].min()) if oid in self.w.objs else float("nan"),
                    o_now.get("contact"), o_now.get("held")))
            if self.is_held(oid):
                Pn = self.w.points(oid)
                self.carry_hang = max(0.0, float(self.ee()[2] - Pn[:, 2].min()))
            yield from self.goto((x, y, max(z_travel, self.travel_z())), yaw, grip=1.0, tol=0.015, max_steps=40, step_max=0.02)
        yield from self.hold(2, grip=1.0)
        ok = self.is_held(oid)
        if ok:
            Pn = self.w.points(oid)
            self.carry_hang = max(0.0, float(self.ee()[2] - Pn[:, 2].min()))
            self.held_id = oid
        return ok

    def carry_to(self, pos, yaw, tol=0.015, max_steps=70, step_max=CARRY_STEP):
        """Move with the object in the fingers; stop as soon as it is no longer held. -> True while held."""
        pos = np.asarray(pos, float)
        oid = self.held_id
        for i in range(max_steps):
            if oid is not None and not self.is_held(oid):
                log("t=%d object %d slipped out on the way" % (self.t(), oid))
                return False
            if np.linalg.norm(pos - self.ee()) < tol and np.linalg.norm(self.orient_error(yaw, self.w.ee_q)) < 0.04:
                return True
            yield self.servo(pos, yaw, 1.0, 0, min(step_max, 0.015 + 0.01 * i))
        return oid is None or self.is_held(oid)

    def release_at(self, x, y, z_rel, yaw):
        """Carry the held object over (x, y), lower it to z_rel (tool centre) and let go."""
        ok = yield from self.carry_to((x, y, self.travel_z()), yaw)
        if not ok:
            yield from self.retreat(yaw)
            return False
        yield from self.goto((x, y, z_rel), yaw, grip=1.0, tol=0.008, max_steps=40, step_max=0.03)
        yield from self.hold(4, grip=0.0)
        self.carry_hang = 0.0
        self.held_id = None
        yield from self.goto((x, y, TRAVEL_Z), yaw, grip=0.0, tol=0.02, max_steps=30)
        return True

    def retreat(self, yaw=None):
        self.carry_hang = 0.0
        self.held_id = None
        ee = self.ee()
        yield from self.goto((ee[0], ee[1], TRAVEL_Z), self.hand_yaw() if yaw is None else yaw, grip=0.0, tol=0.03,
                             max_steps=20)

    # -- where to put things -------------------------------------------------------------------------
    def held_footprint(self, oid):
        """The held object's points relative to the tool centre, in the world frame (it moves with the
        hand, turned by the hand's yaw change)."""
        return self.w.points(oid) - self.ee()

    def height_map(self, name, exclude=(), cell=0.01):
        """The truth's height map of container `name`'s interior (1 cm cells, the floor where nothing
        lies), from every other object's points, grown by one cell."""
        b = self.w.containers[name]
        ix, iy = INNER_HALF[name]
        nx, ny = int(2 * ix / cell) + 1, int(2 * iy / cell) + 1
        H = np.full((nx, ny), FLOOR_Z)
        for k, q in self.w.others(None, near_xy=b[:2], radius=0.30, exclude=exclude):
            lx, ly = container_local(b, q[:, 0], q[:, 1])
            i = np.floor((lx + ix) / cell).astype(int)
            j = np.floor((ly + iy) / cell).astype(int)
            m = (i >= 0) & (i < nx) & (j >= 0) & (j < ny) & (q[:, 2] < RIM_Z + 0.10)
            if m.any():
                np.maximum.at(H, (i[m], j[m]), q[m, 2])
        G = H.copy()
        G[1:, :] = np.maximum(G[1:, :], H[:-1, :])
        G[:-1, :] = np.maximum(G[:-1, :], H[1:, :])
        H2 = G.copy()
        H2[:, 1:] = np.maximum(H2[:, 1:], G[:, :-1])
        H2[:, :-1] = np.maximum(H2[:, :-1], G[:, 1:])
        return H2, cell

    def place_spot(self, name, oid, exclude=(), avoid=None):
        """(x, y, z_rel, yaw) to set the held object down in container `name`: its footprint inside the
        interior, as low as the truth's height map allows there, where the hand fits for the release
        (red: from just above the rim, the hand cannot reach low that near the base), and, in a class
        container, where a hand can take it out again (the planner finds a grasp of it at the predicted
        resting pose, among the container's walls). The hand keeps its yaw or turns a quarter (the object
        turns with it). `avoid`: an object whose footprint the held one must not land on (what a dig is for)."""
        b = self.w.containers[name]
        rel = self.held_footprint(oid)
        yaw0 = self.hand_yaw()
        hang = -float(rel[:, 2].min())
        ix, iy = INNER_HALF[name]
        H, cell = self.height_map(name, exclude=tuple(exclude) + (oid,))
        A = None
        if avoid is not None and avoid in self.w.objs:
            A = self.w.points(avoid)[:, :2]
        red = name == "red"
        cands = []
        for dyaw in (0.0, math.pi / 2, -math.pi / 2):
            yaw = wrap(yaw0 + dyaw)
            F = (rot_z(dyaw) @ rel.T).T
            lxs = np.arange(0.0, 0.0501, 0.025) if red else np.arange(-0.10, 0.1001, 0.02)
            lys = np.arange(-0.04, 0.0401, 0.04) if red else np.arange(-0.08, 0.0801, 0.02)
            for lx in lxs:
                for ly in lys:
                    x, y = container_point(b, lx, ly)
                    W = F[:, :2] + (x, y)
                    flx, fly = container_local(b, W[:, 0], W[:, 1])
                    if np.any(np.abs(flx) > ix - 0.012) or np.any(np.abs(fly) > iy - 0.012):
                        continue
                    if A is not None and np.min(((A[:, None, :] - W[None, ::3, :]) ** 2).sum(-1)) < 0.03 ** 2:
                        continue
                    i = np.clip(np.floor((flx + ix) / cell).astype(int), 0, H.shape[0] - 1)
                    j = np.clip(np.floor((fly + iy) / cell).astype(int), 0, H.shape[1] - 1)
                    under = float(H[i, j].max())
                    z_rel = under + PLACE_GAP + hang
                    fits_low = (not red) and reachable_xy(x, y) and \
                        hand_fits3d(self.w.containers, (x, y, z_rel), yaw, (0.0, 0.0), inside=name)
                    if not fits_low:
                        z_rel = max(z_rel, RIM_Z + 0.02 + hang)
                    cost = (under - FLOOR_Z) * 4.0 + (0.0 if fits_low or red else 0.15) + 0.05 * abs(dyaw) \
                        + 0.1 * math.hypot(lx, ly) + (0.02 if abs(ly) > 0.05 else 0.0)
                    cands.append((cost, (float(x), float(y), float(z_rel), float(yaw)), F, under, dyaw))
        if not cands:
            if avoid is not None:
                return None
            return float(b[0]), float(b[1]), RIM_Z + 0.03 + hang, yaw0
        cands.sort(key=lambda c: c[0])
        if red:
            return cands[0][1]
        for c in cands[:12]:
            if self.retrievable(oid, name, c):
                return c[1]
        log("t=%d no spot in %s from which object %d could be taken out again; the best one" % (self.t(), name, oid))
        return cands[0][1]

    def retrievable(self, oid, name, cand):
        """Would a hand find a grasp of the held object resting at this candidate spot (alone there, among the
        container walls)?"""
        _, (x, y, z_rel, yaw), F, under, dyaw = cand
        P = F + np.array([x, y, z_rel])
        P[:, 2] += under - float(P[:, 2].min())               # it settles onto the surface under it
        o = dict(self.w.objs[oid])
        o["pos"] = list(P.mean(0))
        o["quat"] = quat_mul(yaw_quat(dyaw), o["quat"])
        o["in_container"] = name
        fw = FakeWorld(self.w.containers, {oid: o}, {oid: P})
        return bool(plan_grasps(fw, oid))

    def desk_spot(self, oid, near=None):
        """A free desk spot for setting the held object down for a while."""
        rel = self.held_footprint(oid)
        occ = [self.w.objs[k]["pos"][:2] for k in self.w.objs if k != oid]
        best = None
        for x in np.arange(0.40, 0.721, 0.02):
            for y in np.arange(-0.60, 0.601, 0.03):
                if not reachable_xy(x, y, margin=0.03):
                    continue
                W = rel[:, :2] + (x, y)
                if any(min(footprint_dist(self.w.containers, n, px, py) for n in self.w.containers) < 0.04
                       for px, py in W[::5]):
                    continue
                if any(math.hypot(x - ox, y - oy) < 0.12 for ox, oy in occ):
                    continue
                d = math.hypot(x - near[0], y - near[1]) if near is not None else math.hypot(x - 0.55, y)
                if best is None or d < best[0]:
                    best = (d, (float(x), float(y)))
        return best[1] if best else None

    def put_away(self, oid, dest):
        """The held object into container `dest` (or onto the desk when dest is None)."""
        if oid == self.w.target_id():
            dest = "red"
        if dest is None:
            spot = self.desk_spot(oid, near=self.ee()[:2])
            if spot is None:
                dest = CONTAINER_OF_CLASS[self.w.objs[oid]["cls"]]
            else:
                hang = -float(self.held_footprint(oid)[:, 2].min())
                ok = yield from self.release_at(spot[0], spot[1], DESK_Z + PLACE_GAP + hang + 0.005, self.hand_yaw())
                return ok
        x, y, z_rel, yaw = self.place_spot(dest, oid)
        ok = yield from self.release_at(x, y, z_rel, yaw)
        return ok

    # -- the jobs ------------------------------------------------------------------------------------
    def fail(self, oid, cooldown=None):
        n, _, _ = self.fails.get(oid, (0, 0, 0))
        self.fails[oid] = (n + 1, self.t(), FAIL_COOLDOWN * (n + 1) if cooldown is None else cooldown)

    def nfails(self, oid):
        return self.fails.get(oid, (0, 0, 0))[0]

    def cooling(self, oid):
        n, t0, d = self.fails.get(oid, (0, -10 ** 9, 0))
        return n > 0 and self.t() - t0 < d

    def moving(self, oid):
        o = self.w.objs[oid]
        return not o.get("resting") or o.get("state") == "transit"

    def todo(self, exclude=()):
        """Objects not where they belong, best first: misfiled (in a wrong container), then loose."""
        out = []
        tgt = self.w.target_id()
        for oid, o in self.w.objs.items():
            if oid in exclude or oid == tgt or self.w.done(oid):
                continue
            p = o["pos"]
            if p[2] < -0.1:
                continue
            if math.hypot(p[0], p[1]) < BASE_RESCUE + 0.01:
                continue                    # the world sets it back on the desk itself
            inside = self.w.where(oid)
            gain = 2.0 if inside is not None else 1.0
            d = float(np.linalg.norm(np.asarray(p[:2]) - self.ee()[:2]))
            out.append((-gain + 0.5 * d + (0.6 if self.cooling(oid) else 0.0), oid))
        out.sort()
        return [oid for _, oid in out]

    def desk_in_order(self, exclude=()):
        """Everything except `exclude` (and what keeps failing) in its home container and at rest."""
        for oid in self.todo(exclude=exclude):
            n = self.nfails(oid)
            if n >= 3:
                continue
            return False
        return True

    def first_blocker(self, oid, seen=None):
        """The object to take away first so that `oid` can be grasped: of the plans only other objects rule out,
        the best one's highest obstacle (not one already on the chain), itself cleared first when it has no free
        grasp (up to three deep). -> object id, or None when no plan of `oid` would be free even without the others."""
        seen = set() if seen is None else seen
        seen.add(oid)
        n = self.nfails(oid)
        for plan in [p for p in plan_grasps(self.w, oid, fails=n, ignore_others=True) if p.blockers]:
            cands = [k for k in plan.blockers if k not in seen and k in self.w.objs and not self.cooling(k)]
            if not cands:
                continue
            bid = max(cands, key=lambda k: float(self.w.points(k)[:, 2].max()))
            if len(seen) < 4 and not plan_grasps(self.w, bid, fails=self.nfails(bid)):
                deeper = self.first_blocker(bid, seen)
                if deeper is not None:
                    return deeper
            return bid
        return None

    def move_object(self, oid, dest, avoid=None):
        """Pick object `oid` and put it into `dest` (a container name, or None for the desk); when the hand
        cannot get at it, what is in the way goes first (see first_blocker; a blocker goes home, or elsewhere
        in its container away from `avoid`, or onto the desk). -> True when `oid` itself was put away."""
        if oid not in self.w.objs:
            return False
        o = self.w.objs[oid]
        if self.moving(oid):
            yield from self.hold(3, grip=0.0)
            return False
        if o["model"] in TOPPLE_MODELS and self.w.where(oid) is None and self.toppled.get(oid, 0) < 2:
            P = self.w.points(oid)
            if float(P[:, 2].max() - P[:, 2].min()) > TOPPLE_MIN_H:
                did = yield from self.topple(oid)
                if did:
                    return False
        n = self.nfails(oid)
        plans = plan_grasps(self.w, oid, fails=n, avoid_yaws=self.bad_yaws.get(oid, ()))
        if not plans:
            bid = self.first_blocker(oid)
            if bid is not None and self.moving(bid):
                yield from self.hold(3, grip=0.0)
                return False
            if bid is not None:
                log("t=%d object %d (%s): no free grasp; moving %d (%s) out of the way first" % (
                    self.t(), oid, o["model"], bid, self.w.objs[bid]["model"]))
                # for a request's object, what is in the way goes out onto the desk (put back in the same container
                # it can end up in the way again); the tidy loop brings it home before the mark
                yield from self.move_aside(bid, avoid=oid if avoid is None else avoid,
                                           to_desk=oid == self.w.target_id())
                if oid != self.w.target_id():
                    self.fail(oid)          # a tidy job digs one layer at a time, between the easy ones
                return False
            if bid is None and self.wall_side(oid) is not None:
                did = yield from self.drag_from_wall(oid)
                if did:
                    return False
            log("t=%d object %d (%s) at (%.2f,%.2f,%.2f) in %s: no grasp" % (
                self.t(), oid, o["model"], o["pos"][0], o["pos"][1], o["pos"][2], self.w.where(oid)))
            self.fail(oid)
            yield from self.hold(2, grip=0.0)
            return False
        plan = plans[0]
        log("t=%d pick %d (%s) from %s -> %s: %s" % (self.t(), oid, o["model"], self.w.where(oid), dest, plan))
        self.moved_before_grasp = False
        held = yield from self.pick(oid, plan)
        if not held:
            if self.moved_before_grasp:
                return False
            log("t=%d grasp of %d (%s) failed" % (self.t(), oid, o["model"]))
            self.fail(oid)
            yield from self.retreat()
            return False
        if oid == self.w.target_id():
            dest = "red"            # a request named what the hand already holds
        ok = yield from self.put_away(oid, dest)
        if not ok:
            self.fail(oid)
            return False
        self.fails.pop(oid, None)
        return True

    def move_aside(self, bid, avoid, desk_ok=True, to_desk=False):
        """Take object `bid` out of the way of `avoid`: into its home container when that is not where it lies
        (or the home is red's target), else to another place in the same container off `avoid`'s footprint, else
        onto the desk (the tidy loop puts it back later; with desk_ok False the best place in the container)."""
        n = self.nfails(bid)
        plans = plan_grasps(self.w, bid, fails=n, avoid_yaws=self.bad_yaws.get(bid, ()))
        if not plans:
            self.fail(bid)
            yield from self.hold(2, grip=0.0)
            return False
        self.moved_before_grasp = False
        held = yield from self.pick(bid, plans[0])
        if not held:
            if not self.moved_before_grasp:
                self.fail(bid)
            yield from self.retreat()
            return False
        home = self.w.home_of(bid)
        here = self.w.where(avoid) if avoid in self.w.objs else None
        if to_desk and self.desk_spot(bid, near=self.ee()[:2]) is not None:
            ok = yield from self.put_away(bid, None)
            return ok
        if home != here:
            ok = yield from self.put_away(bid, home)
            return ok
        spot = self.place_spot(here, bid, avoid=avoid)
        if spot is None and not desk_ok:
            spot = self.place_spot(here, bid)
        if spot is not None:
            ok = yield from self.release_at(*spot)
            return ok
        ok = yield from self.put_away(bid, None)
        return ok

    def wall_side(self, oid):
        """(inward direction (dx, dy), gap to the wall) for an object in a container lying within 2 cm of one of
        its walls, else None."""
        name = self.w.where(oid)
        if name is None:
            return None
        b = self.w.containers[name]
        P = self.w.points(oid)
        lx, ly = container_local(b, P[:, 0], P[:, 1])
        ix, iy = INNER_HALF[name]
        gaps = {(-1.0, 0.0): ix - float(lx.max()), (1.0, 0.0): float(lx.min()) + ix,
                (0.0, -1.0): iy - float(ly.max()), (0.0, 1.0): float(ly.min()) + iy}
        d_loc, gap = min(gaps.items(), key=lambda kv: kv[1])
        if gap > 0.02:
            return None
        c, s_ = math.cos(b[2]), math.sin(b[2])
        return (c * d_loc[0] - s_ * d_loc[1], s_ * d_loc[0] + c * d_loc[1]), gap

    def drag_from_wall(self, oid):
        """Slide an object that lies against a container wall (no pinch fits there) toward the middle: the
        closed fingertips press on its top and the hand moves 5 cm inward. -> True when the arm moved."""
        ws = self.wall_side(oid)
        if ws is None or self.dragged.get(oid, 0) >= 2:
            return False
        (dx, dy), gap = ws
        name = self.w.where(oid)
        P = self.w.points(oid)
        top = float(P[:, 2].max())
        topP = P[P[:, 2] > top - 0.008]
        d = np.array([dx, dy])
        best = None
        yaw0 = wrap_half(math.atan2(dy, dx))             # the hand's long side along the wall
        for tilt in [(0.0, 0.0)] + [(0.0, sg * b) for b in (0.30, 0.45) for sg in (1, -1)] \
                + [(sg * a, 0.0) for a in (0.30, 0.45) for sg in (1, -1)]:
            for shift in (0.0, 0.01, 0.02, 0.03):
                px, py = topP[:, :2].mean(0) + shift * d
                # the press point must stay on the object's top
                if np.min(np.hypot(topP[:, 0] - px, topP[:, 1] - py)) > 0.012:
                    continue
                a = approach_dir(yaw0, tilt)
                tip = np.array([px, py, top - 0.006])
                tcp = tip - FINGER_TIP_BELOW_TCP * a          # the tool centre sits back along the approach
                ok = all(hand_fits3d(self.w.containers, tcp + k * np.array([dx, dy, 0.0]), yaw0, tilt, inside=name)
                         for k in (0.0, 0.05))
                if ok and reachable_xy(float(tcp[0]), float(tcp[1])):
                    best = (tcp, tilt)
                    break
            if best is not None:
                break
        if best is None:
            return False
        tcp, tilt = best
        yaw, tilt = yaw_branch((tcp[0], tcp[1]), yaw0, tilt)
        a = approach_dir(yaw, tilt)
        self.dragged[oid] = self.dragged.get(oid, 0) + 1
        log("t=%d drag %d (%s) %.3f off the wall in %s inward (%.2f,%.2f) tilt %s" % (
            self.t(), oid, self.w.objs[oid]["model"], gap, name, dx, dy, tilt))
        above = tcp - 0.03 * a
        end = tcp + 0.05 * np.array([dx, dy, 0.0])
        yield from self.goto((above[0], above[1], self.travel_z()), yaw, grip=1.0, tol=0.015, tilt=tilt, yaw_tol=0.08)
        yield from self.goto(above, yaw, grip=1.0, tol=0.006, max_steps=30, tilt=tilt, yaw_tol=0.08)
        yield from self.goto(tcp, yaw, grip=1.0, tol=0.004, max_steps=15, step_max=0.008, tilt=tilt, yaw_tol=0.08)
        yield from self.goto(end, yaw, grip=1.0, tol=0.006, max_steps=25, step_max=0.006, tilt=tilt, yaw_tol=0.08)
        yield from self.goto((end[0], end[1], TRAVEL_Z), yaw, grip=0.0, tol=0.02, max_steps=30)
        yield from self.hold(4, grip=0.0)
        return True

    def restructure_one(self):
        """Idle time: an object in a class container that no hand can take out now (others on it or around it)
        is dug free the way a request would dig for it, the blockers going to places in the same container from
        which they can be taken out again. The initial piles come apart this way before most requests open.
        -> True when the arm did something."""
        if self.t() - self.restructured_t < PREPARE_EVERY:
            return False
        for name in CLASS_CONTAINERS:
            inside = [k for k in self.w.objs if self.w.where(k) == name and self.w.done(k) and not self.moving(k)]
            for oid in sorted(inside, key=lambda k: float(self.w.points(k)[:, 2].min())):
                if self.restructure_tries.get(oid, 0) >= 3 or self.cooling(oid):
                    continue
                if plan_grasps(self.w, oid, avoid_yaws=self.bad_yaws.get(oid, ())):
                    continue
                bid = self.first_blocker(oid)
                if bid is None or self.w.where(bid) != name or self.moving(bid):
                    continue
                self.restructure_tries[oid] = self.restructure_tries.get(oid, 0) + 1
                log("t=%d restructure %s: %d (%s) cannot be taken out; moving %d (%s) aside" % (
                    self.t(), name, oid, self.w.objs[oid]["model"], bid, self.w.objs[bid]["model"]))
                yield from self.move_aside(bid, avoid=oid, desk_ok=False)
                return True
        self.restructured_t = self.t()
        return False

    def prepare_one(self):
        """Idle time: an object in a class container that no hand could take out (against a wall) is slid
        toward the middle, so that a later request for it can be served. -> True when the arm did something.
        Looked at once every PREPARE_EVERY steps (planning every container object is not free)."""
        if self.t() - self.prepared_t < PREPARE_EVERY:
            return False
        self.prepared_t = self.t()
        for oid, o in self.w.objs.items():
            name = self.w.where(oid)
            if name not in CLASS_CONTAINERS or self.moving(oid) or self.dragged.get(oid, 0) >= 2:
                continue
            if self.wall_side(oid) is None:
                continue
            if plan_grasps(self.w, oid, ignore_others=True):
                continue
            did = yield from self.drag_from_wall(oid)
            if did:
                self.prepared_t = -10 ** 9
                return True
        return False

    def topple(self, oid):
        """Push a tall standing bottle over along the short side of its footprint toward free desk."""
        P = self.w.points(oid)
        cen = P[:, :2].mean(0)
        h = float(P[:, 2].max() - P[:, 2].min())
        bottom = float(P[:, 2].min())
        # the footprint's short axis
        C = np.cov((P[:, :2] - cen).T)
        evals, evecs = np.linalg.eigh(C)
        short = evecs[:, 0]
        occ = [np.asarray(self.w.objs[k]["pos"][:2]) for k in self.w.objs if k != oid]
        best = None
        for sgn in (1.0, -1.0):
            for dang in (0.0, 0.4, -0.4):
                a = math.atan2(short[1], short[0]) + (0.0 if sgn > 0 else math.pi) + dang
                ux, uy = math.cos(a), math.sin(a)
                body = [cen + np.array([ux, uy]) * s for s in (0.03, h / 2.0, h + 0.01)]
                if not all(reachable_xy(float(px), float(py)) for px, py in body):
                    continue
                if not all(footprint_dist(self.w.containers, n, float(px), float(py)) > 0.03
                           for n in self.w.containers for px, py in body):
                    continue
                if not all(np.linalg.norm(bp - op) > 0.08 for bp in body for op in occ):
                    continue
                proj = (P[:, :2] - cen) @ np.array([ux, uy])
                half = float(proj.max())
                yaw_h = wrap_half(a)                     # the hand's broad side leading
                start = cen - np.array([ux, uy]) * (half + HAND_HALF_T + 0.025)
                if not hand_fits3d(self.w.containers, (start[0], start[1], 0.1), yaw_h):
                    continue
                if not reachable_xy(float(start[0]), float(start[1])):
                    continue
                score = abs(dang) + abs(np.linalg.norm(body[1]) - 0.56)
                if best is None or score < best[0]:
                    best = (score, a, start, yaw_h, half)
        if best is None:
            return False
        _, a, start, yaw, half = best
        ux, uy = math.cos(a), math.sin(a)
        z_push = bottom + max(0.6 * h - HAND_BODY_ABOVE_TCP, 0.035)
        end = cen + np.array([ux, uy]) * (half + 0.06 - HAND_HALF_T)
        log("t=%d topple %d (%s, %.2f m tall) toward %.2f rad" % (self.t(), oid, self.w.objs[oid]["model"], h, a))
        self.toppled[oid] = self.toppled.get(oid, 0) + 1
        z_up = max(self.travel_z(), bottom + h + 0.10)
        yield from self.goto((start[0], start[1], z_up), yaw, grip=1.0, tol=0.015)
        yield from self.goto((start[0], start[1], z_push), yaw, grip=1.0, tol=0.01, max_steps=40, step_max=0.03)
        yield from self.goto((end[0], end[1], z_push), yaw, grip=1.0, tol=0.012, max_steps=40, step_max=0.012)
        yield from self.goto((end[0], end[1], z_up), yaw, grip=0.0, tol=0.02, max_steps=30)
        yield from self.hold(8, grip=0.0)
        return True

    def tidy_one(self, exclude=(), avoid_home=None, easy_only=False):
        """Put one object where it belongs (not into `avoid_home`; with easy_only, only one that a hand can take
        right now, no dig). -> True when the arm did something."""
        for oid in self.todo(exclude=exclude):
            if self.cooling(oid):
                continue
            if self.moving(oid):
                continue
            if avoid_home is not None and CONTAINER_OF_CLASS[self.w.objs[oid]["cls"]] == avoid_home:
                continue
            if easy_only and not plan_grasps(self.w, oid, fails=self.nfails(oid), avoid_yaws=self.bad_yaws.get(oid, ())):
                continue
            dest = CONTAINER_OF_CLASS[self.w.objs[oid]["cls"]]
            yield from self.move_object(oid, dest)
            return True
        return False

    def serve(self):
        """One job for the open request."""
        r = self.w.request
        rid = r["id"]
        left = int(r["deadline_t"]) - self.t()
        tgt = self.w.target_id()
        if tgt is None:
            # the desk in order first (what a hand can take right now: a dig would only delay the mark), then the mark
            did = False
            if left >= MARK_MARGIN and TIDY_BEFORE_MARK:
                did = yield from self.tidy_one(easy_only=True)
            if not did:
                log("t=%d request %d (%s): no such object (desk %s)" % (self.t(), rid, r.get("text"),
                                                                       "in order" if self.desk_in_order() else "not in order"))
                self.marked.add(rid)
                yield from self.mark(2)
            return
        if tgt not in self.w.objs:
            yield from self.hold(2, grip=0.0)
            return
        if self.w.done(tgt):
            # in red and at rest: what the dig took out goes back, the desk in order (what a hand can take right
            # now), then the mark
            did = False
            if left >= 150:
                did = yield from self.tidy_one(exclude=(tgt,), easy_only=True)
            if not did:
                log("t=%d request %d (%s): delivered" % (self.t(), rid, r.get("text")))
                self.marked.add(rid)
                yield from self.mark(1)
            return
        o = self.w.objs[tgt]
        if o.get("in_container") == "red" or self.moving(tgt):
            yield from self.hold(3, grip=0.0)      # settling in red (or anywhere)
            return
        # while the object is still being dug for, nothing goes back into its container (it would bury it again);
        # after three failures it is given up and the desk is tidied as usual
        bury = self.w.where(tgt) if self.nfails(tgt) < 3 else None
        if self.cooling(tgt):
            did = yield from self.tidy_one(exclude=(tgt,), avoid_home=bury)
            if not did:
                yield from self.hold(5, grip=0.0)
            return
        ok = yield from self.move_object(tgt, "red")
        if not ok and self.nfails(tgt) and tgt in self.w.objs and not self.w.done(tgt):
            n, t0, _ = self.fails[tgt]
            self.fails[tgt] = (n, t0, 40 * n)       # a request's object is tried again soon, not after 250 x n steps

    def run(self):
        # a first step at the start pose (the probe lock compares replies; nothing to decide yet)
        yield from self.hold(1, grip=0.0)
        while True:
            r = self.w.request
            if r is not None and r["id"] not in self.marked:
                yield from self.serve()
                continue
            did = yield from self.tidy_one()
            if not did and IDLE_DRAG:
                did = yield from self.prepare_one()
            if not did and IDLE_RESTRUCTURE:
                did = yield from self.restructure_one()
            if not did:
                yield from self.goto(IDLE_POS, 0.0, grip=0.0, tol=0.02, max_steps=20)
                yield from self.hold(5, grip=0.0)


def main():
    global GEOM
    GEOM = load_geometry()
    ctl = Controller()
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
