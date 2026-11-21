"""Worlds and models for the duct model's tests and gates. Not part of the contract.

    bare_world(ducts)              a world dict with only what the ducts and the car need (no centreline)
    line_duct(did, start, yaw)     one straight duct of 15 pieces
    impact_rig(angle_deg, ...)     one straight duct and the drive model's car aimed at it, compiled (model, data, car, ducts)
    stadium_world(n_pieces=902)    a long stadium track walled with full ducts, for the lapping cost gate
    venue_spec(world, variant, pushable, mode)   the contract's greybox venue with the duct model's ducts: the stand-in for the venue generator's
                                   `build_spec` until the venue generator exists (the cell can pass it as a provider override)
"""

import copy
import math

import mujoco
import numpy as np

from sketch.examples import check
from sketch.examples import constants as K
from sketch.examples import greybox

from .build import add_ducts
from .params import load_params

FLOOR_RGBA = "0.55 0.55 0.57 1"


def line_duct(did, start, yaw, n=K.DUCT_PIECES_FULL, c=K.PIECE_LENGTH_NOMINAL_M, wall_class="orange_ribbed",
              wall="W0"):
    """A straight duct whose tail end is at `start`, laid along `yaw`."""
    ux, uy = math.cos(yaw), math.sin(yaw)
    pieces = [[start[0] + (k + 0.5) * c * ux, start[1] + (k + 0.5) * c * uy, yaw] for k in range(n)]
    return {"id": did, "wall": wall, "wall_class": wall_class, "diameter_m": K.DUCT_DIAMETER_M, "length_m": n * c,
            "piece_length_m": c, "pieces": pieces}


def bare_world(ducts, base_grip=0.7, putback_seed=1116001, world_id="s4_bare"):
    """A world dict with the fields the ducts, the tracker and the car read: truth.ducts / walls / floor / seeds.
    It has no centreline and does not pass check_world."""
    walls = {}
    for d in ducts:
        walls.setdefault(d["wall"], []).append(d["id"])
    return {"schema": K.WORLD_SCHEMA, "world_id": world_id, "generator": {"name": "s4-scenes", "version": "1", "seed_hex": None},
            "layout_digest": "", "venue_sheet": {},
            "truth": {"cell": {"grip_tier": "lab", "layout_class": "example"},
                      "ducts": ducts,
                      "walls": [{"id": w, "side": "left", "closed": False, "ducts": ids} for w, ids in walls.items()],
                      "gaps": [], "open_walls": [], "clutter": [],
                      "floor": {"kind": "polished_concrete", "base_grip": float(base_grip), "patches": []},
                      "avoidance": {"boxes": [], "slow_cars": []},
                      "seeds": {"relocation": 1, "putback": int(putback_seed), "sensors": 2, "others": 3}}}


def floor_spec(timestep=K.PHYSICS_DT_S):
    return mujoco.MjSpec.from_string(
        '<mujoco model="s4_floor"><compiler angle="radian" autolimits="true"/>'
        '<option timestep="%r" cone="elliptic"/><worldbody>'
        '<light pos="0 0 6" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>'
        '<geom name="floor" type="plane" size="0 0 0.05" group="%d" contype="%d" conaffinity="%d" rgba="%s"/>'
        '</worldbody></mujoco>' % (timestep, K.GROUP_FLOOR, K.CONTACT["floor"][0], K.CONTACT["floor"][1], FLOOR_RGBA))


def compile_with_car(world, car_pose, params=None, mode=None, pushable=True, tyre_model=None, timestep=K.PHYSICS_DT_S,
                     grip=None):
    """Floor + ducts + the drive model's car at `car_pose`: (model, data, car runtime, tracker)."""
    from sketch.car import CarRuntime, add_car
    from .tracker import DuctTracker
    p = params if params is not None else load_params()
    spec = floor_spec(timestep)
    add_ducts(spec, world, pushable=pushable, params=p, mode=mode)
    add_car(spec, prefix=K.EGO_PREFIX, pose=tuple(car_pose), tyre_model=tyre_model)
    model = spec.compile()
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    g = float(world["truth"]["floor"]["base_grip"] if grip is None else grip)
    car = CarRuntime(model, data, prefix=K.EGO_PREFIX, grip=g)
    car.place(car_pose)
    ducts = DuctTracker(model, data, world, params=p) if pushable else None
    mujoco.mj_forward(model, data)
    return model, data, car, ducts


# ------------------------------------------------------------------------------------------------ impact rig
CAR_FRONT_X = K.X_FRONT_M                      # foremost point of the car in its frame C


def impact_world(angle_deg, hit_piece=7.0, n=K.DUCT_PIECES_FULL, c=K.PIECE_LENGTH_NOMINAL_M, base_grip=0.7,
                 target_x=0.0):
    """One straight duct whose axis makes `angle_deg` with the +x axis and crosses y = 0 at x = target_x at the
    station `hit_piece` x c from its tail (7.0 = the start of piece 7; 7.5 = the centre of piece 7)."""
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    start = (target_x - hit_piece * c * ux, -hit_piece * c * uy)
    return bare_world([line_duct("D0", start, a, n=n, c=c)], base_grip=base_grip, world_id="s4_impact")


def impact_car_pose(angle_deg, v, target_x=0.0, run_up_m=None):
    """Car pose on y = 0 heading +x, its front `run_up_m` before the duct's skin along the path."""
    a = math.radians(angle_deg)
    skin = target_x - K.DUCT_RADIUS_M / max(math.sin(a), 1e-6)          # where the car's centreline meets the skin
    ru = (0.6 + 0.35 * v * v) if run_up_m is None else run_up_m
    return [skin - ru - CAR_FRONT_X, 0.0, 0.0]


# ------------------------------------------------------------------------------------------------ stadium track
def _stadium_point(u, ls, rc):
    """Point and heading at arc length u of a stadium centred at the origin, driven counter-clockwise: straights of
    length ls along x at y = -rc (left to right) and y = +rc, semicircles of radius rc."""
    per = 2.0 * ls + 2.0 * math.pi * rc
    u = u % per
    if u < ls:
        return -0.5 * ls + u, -rc, 0.0
    u -= ls
    if u < math.pi * rc:
        phi = -0.5 * math.pi + u / rc
        return 0.5 * ls + rc * math.cos(phi), rc * math.sin(phi), phi + 0.5 * math.pi
    u -= math.pi * rc
    if u < ls:
        return 0.5 * ls - u, rc, math.pi
    u -= ls
    phi = 0.5 * math.pi + u / rc
    return -0.5 * ls + rc * math.cos(phi), rc * math.sin(phi), phi + 0.5 * math.pi


def _wall_ducts(ls, rc, off, prefix, wall, c=K.PIECE_LENGTH_NOMINAL_M, gap=0.03, max_pieces=None):
    """Full ducts laid along the stadium line moved `off` to the left, chords of length c, `gap` between ducts."""
    r = rc - off
    per = 2.0 * ls + 2.0 * math.pi * r
    ducts, u, pieces, total = [], 0.0, [], 0

    def pt(uu):
        x, y, h = _stadium_point(uu, ls, r)
        return x, y

    def advance(u0, chord):
        x0, y0 = pt(u0)
        lo, hi = u0 + chord - 1e-9, u0 + 1.3 * chord
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            x, y = pt(mid)
            if math.hypot(x - x0, y - y0) < chord:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    while True:
        u2 = advance(u, c)
        if u2 > per - 0.05 or (max_pieces is not None and total >= max_pieces):
            break
        (x0, y0), (x1, y1) = pt(u), pt(u2)
        pieces.append([0.5 * (x0 + x1), 0.5 * (y0 + y1), math.atan2(y1 - y0, x1 - x0)])
        total += 1
        u = u2
        if len(pieces) == K.DUCT_PIECES_FULL:
            ducts.append({"id": "%s%d" % (prefix, len(ducts)), "wall": wall, "wall_class": "orange_ribbed",
                          "diameter_m": K.DUCT_DIAMETER_M, "length_m": len(pieces) * c, "piece_length_m": c, "pieces": pieces})
            pieces = []
            u = advance(u, gap)
    if pieces:
        ducts.append({"id": "%s%d" % (prefix, len(ducts)), "wall": wall, "wall_class": "orange_ribbed",
                      "diameter_m": K.DUCT_DIAMETER_M, "length_m": len(pieces) * c, "piece_length_m": c, "pieces": pieces})
    return ducts


def stadium_world(n_pieces=902, rc=6.0, half_width=0.8, base_grip=0.7):
    """A stadium track (counter-clockwise, half straights ls / 2 chosen so the two walls hold about n_pieces pieces,
    then trimmed to exactly n_pieces). Fields: truth.ducts / walls / centreline (samples, 0.1 m apart) / floor /
    start_pose / seeds; not a check_world-valid venue (no timing line, gaps, open walls)."""
    off = half_width + K.DUCT_RADIUS_M
    wall_len = n_pieces * (K.PIECE_LENGTH_NOMINAL_M + 0.03 / 15.0)
    ls = max(1.0, (wall_len - 4.0 * math.pi * rc) / 4.0)
    for _ in range(40):
        outer = _wall_ducts(ls, rc, -off, "OUT", "W_OUT")
        inner = _wall_ducts(ls, rc, off, "IN", "W_IN")
        total = sum(len(d["pieces"]) for d in outer + inner)
        if total >= n_pieces:
            break
        ls += 0.25 * (n_pieces - total) * K.PIECE_LENGTH_NOMINAL_M / 2.0 + 0.1
    # trim to exactly n_pieces (from the last inner duct backwards)
    excess = total - n_pieces
    while excess > 0:
        d = inner[-1]
        k = min(excess, len(d["pieces"]))
        d["pieces"] = d["pieces"][:len(d["pieces"]) - k]
        excess -= k
        if not d["pieces"]:
            inner.pop()
        else:
            d["length_m"] = len(d["pieces"]) * d["piece_length_m"]
    ducts = outer + inner
    per = 2.0 * ls + 2.0 * math.pi * rc
    n_s = int(round(per / 0.1))
    samples = [[*_stadium_point(per * i / n_s, ls, rc)[:2], half_width, half_width] for i in range(n_s)]
    w = bare_world(ducts, base_grip=base_grip, world_id="s4_stadium_%d" % n_pieces)
    t = w["truth"]
    t["walls"] = [{"id": "W_OUT", "side": "right", "closed": True, "ducts": [d["id"] for d in outer]},
                  {"id": "W_IN", "side": "left", "closed": True, "ducts": [d["id"] for d in inner]}]
    t["centreline"] = {"closed": True, "direction": "ccw", "length_m": per, "samples": samples}
    x0, y0, h0 = _stadium_point(1.0, ls, rc)
    t["start_pose"] = {"pose": [x0, y0, h0], "s_m": 1.0}
    t["bounds"] = {"xmin": -0.5 * ls - rc - off - K.DUCT_RADIUS_M, "xmax": 0.5 * ls + rc + off + K.DUCT_RADIUS_M,
                   "ymin": -rc - off - K.DUCT_RADIUS_M, "ymax": rc + off + K.DUCT_RADIUS_M}
    t["stadium"] = {"ls": ls, "rc": rc}
    return w


class Pursuit(object):
    """Pure pursuit on a world's centreline with the true pose (privileged; for gates only). `offset_m` shifts the
    line to the left."""

    def __init__(self, world, speed_mps=4.5, lookahead_m=1.2, offset_m=0.0):
        arr = np.asarray(world["truth"]["centreline"]["samples"], dtype=float)[:, :2]
        t = np.roll(arr, -1, axis=0) - np.roll(arr, 1, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        self.p = arr + offset_m * np.stack([-t[:, 1], t[:, 0]], axis=1)
        self.ds = float(np.mean(np.linalg.norm(np.roll(arr, -1, axis=0) - arr, axis=1)))
        self.v = speed_mps
        self.ld = lookahead_m
        self.i = None

    def command(self, pose):
        x, y, yaw = pose
        if self.i is None:
            self.i = int(np.argmin(np.hypot(self.p[:, 0] - x, self.p[:, 1] - y)))
        n = len(self.p)
        win = (self.i + np.arange(-5, 40)) % n
        self.i = int(win[np.argmin(np.hypot(self.p[win, 0] - x, self.p[win, 1] - y))])
        tx, ty = self.p[(self.i + int(round(self.ld / self.ds))) % n]
        alpha = math.atan2(ty - y, tx - x) - yaw
        steer = math.atan2(2.0 * K.WHEELBASE_M * math.sin(alpha), self.ld)
        return max(-K.STEER_MAX_RAD, min(K.STEER_MAX_RAD, steer)), self.v


# ------------------------------------------------------------------------------------------------ venue stand-in
def venue_spec(world, variant="track", pushable=True, mode=None, params=None):
    """The contract's greybox venue (floor, zone slabs and decals, tape, timing line, clutter; no ducts) plus the duct model's ducts:
    the stand-in for the venue generator's `build_spec(world, variant, pushable)` until the venue generator exists. The `avoid` variant's boxes are
    the obstacle model's (`add_boxes`), as in the venue generator's contract."""
    bare = copy.deepcopy(world)
    bare["truth"]["ducts"] = []
    spec = mujoco.MjSpec.from_string(greybox.build_xml(bare, variant="track", cars=None))
    add_ducts(spec, world, pushable=pushable, params=params, mode=mode)
    return spec


def check_errors(model, world, mode):
    """The contract's check_model on a model built here: pushable=True in every pushable mode (the contract recognises the
    active_window layout), False for static ducts."""
    pushable = mode != "static"
    return check.check_model(model, world, pushable=pushable)
