"""World-file additions of a head-to-head venue: the grid, the three field cars of each heat, and the views of a
head-to-head world that the shared venue, car and opponent code read.

    w, rec = generate(base_world, "fast")        # the generator: a draft world whose two heat fields each passed the
                                                 # field-alone run, drawn again with an attempt index where one failed;
                                                 # rec = per heat and per attempt made: passed or the first contact
    w = add_head_to_head(base_world, "fast", attempts=[0, 2])   # the same draw for given attempt indices, no run
    w = add_head_to_head(base_world, "fast")     # attempt index 0 in both heats; nothing of the base is changed
    w = with_heat_laps(w, 9)                     # the final world: N on the venue sheet, the layout digest renewed
    base = base_view(w)                          # the base world exactly as the venue generator wrote it
    view = heat_view(w, 1)                       # the base world whose truth.avoidance holds heat 1's field, no boxes
    text = dumps(w)                              # the bytes of a world file (the venue generator's writing rule)

Draws (numpy default_rng over [truth.seeds.others, key, ...]; keys in `field_params.json`, block `streams`):
  [seed, 71]        the side of the car's grid slot: +1 left, -1 right of the centreline, the same in both heats,
                    drawn once per venue;
  [seed, 72, h, k]  the line of heat h (1, 2), grid slot k (1 to 3): the opponents' path rule with no boxes, a blend
                    of the curvature-minimising line and the centreline with weight 10^U(-6, -3), drawn again from the
                    same stream (at most 20 draws) until the path check passes (the car's footprint 0.06 m from the
                    walls, smallest radius at least the opponents' minimum);
  [seed, 73, h, k]  the pace of that car: the tier's nominal for the slot plus 0.001 n, n uniform on -20 .. 20.
These are the streams of attempt index a = 0. Attempt a = 1 to 4 (block `redraw`) of heat h uses
[seed, 72, h, k, a] and [seed, 73, h, k, a]: all three lines and all three paces of that heat are new, the other heat
and the car's side are not touched.

The redraw (`generate`): each heat h has an attempt index a_h. Its field is drawn with a = 0 and its three cars run
from the grid without the car for 300 s (`alone.run_alone`); if that run has a contact with a wall, between two field
cars or with anything else, or a staff reset, the heat is drawn again with a = 1, and so on up to a = 4. a_h is the
first index whose field passes; when all five fail the heat keeps a = 4 and the record says so (the venue then fails
the field-alone predicate). The two heats are independent. `truth.head_to_head.field_attempts` = [a_1, a_2].

Grid: s0 = the station of the reference point of the venue's start pose; slot k's reference point stands at station
s0 - 1.5 (k - 1) m. Slots 1 to 3 hold the field cars, fastest first, each at the point of its own line whose station
is the slot's, heading along its line. Slot 4 holds the car, 0.15 m to its side of the centreline, heading along the
centreline's tangent. A slot record is {slot, pose (car frame [x, y, yaw], as the start pose), s_m (station of the
reference point)}. The lines are independent of the tier, so the three tiers of a venue drawn with the same attempt
indices differ only in the paces; the field-alone run depends on the paces, so a_h can differ between the tiers, and
then the lines of that heat differ too.
"""

import copy
import hashlib
import json
import math
import os

import numpy as np

from sketch.examples import check
from sketch.others import placement as PL
from sketch.others.params import load_params as load_others_params
from sketch.others.path import REF_X, Path

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "field_params.json")
DRAW_BLOCKS = ("world_schema", "base_world_schema", "world_id_prefix", "grid", "field", "tiers", "streams", "redraw")


class FieldError(RuntimeError):
    pass


def load_params(source=None):
    """The field parameters as a dict: from `source` (a path or a dict) or from the file next to this module."""
    if isinstance(source, dict):
        return copy.deepcopy(source)
    with open(source or PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def draw_params_sha256(params=None):
    """sha256 of the canonical text of the blocks that decide the additions."""
    P = load_params(params)
    body = {k: P[k] for k in DRAW_BLOCKS}
    return hashlib.sha256(check.canonical_json(body).encode("ascii")).hexdigest()


def dumps(world):
    from sketch.venue.gen_venue import dumps as _dumps
    return _dumps(world)


def write_world(world, path):
    text = dumps(world)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)
    return path


def car_id(k, params=None):
    """The id of the field car in grid slot k."""
    return load_params(params)["field"]["id_format"] % int(k)


def others_seed(world):
    return int(world["truth"]["seeds"]["others"])


def stream(seed, key, *more):
    return np.random.default_rng([int(seed), int(key)] + [int(m) for m in more])


def attempts_max(params=None):
    """The largest attempt index of a heat's draw."""
    return int(load_params(params)["redraw"]["attempts_max"])


def check_attempt(attempt, params=None):
    """`attempt` as an int; ValueError unless it is an integer from 0 to `attempts_max`."""
    amax = attempts_max(params)
    if isinstance(attempt, bool) or not isinstance(attempt, (int, np.integer)) or not 0 <= int(attempt) <= amax:
        raise ValueError("an attempt index is an integer from 0 to %d, not %r" % (amax, attempt))
    return int(attempt)


def check_attempts(attempts, params=None):
    """The two attempt indices [a_1, a_2] (None: [0, 0]); ValueError on anything else."""
    if attempts is None:
        return [0, 0]
    try:
        n = len(attempts)
    except TypeError:
        n = -1
    if isinstance(attempts, (str, bytes, dict)) or n != 2:
        raise ValueError("attempts are two indices, one per heat, not %r" % (attempts,))
    return [check_attempt(a, params) for a in attempts]


def attempt_keys(attempt, params=None):
    """The keys a stream of a heat's draw gets after [seed, key, h, k]: none for attempt 0, [a] for attempt a >= 1."""
    a = check_attempt(attempt, params)
    return [] if a == 0 else [a]


# ---------------------------------------------------------------------------------------------- draws
def draw_car_side(seed, params=None):
    P = load_params(params)
    return 1 if int(stream(seed, P["streams"]["car_side"]).integers(0, 2)) == 1 else -1


def draw_pace(seed, h, k, tier, params=None, attempt=0):
    """(nominal, speed factor, n) of the car in slot k of heat h for a tier, at an attempt index."""
    P = load_params(params)
    if tier not in P["tiers"]:
        raise ValueError("tier must be one of %s" % sorted(P["tiers"]))
    half = int(P["field"]["pace_half_steps"])
    n = int(stream(seed, P["streams"]["pace"], h, k, *attempt_keys(attempt, P)).integers(-half, half + 1))
    nominal = float(P["tiers"][tier][k - 1])
    return nominal, round(nominal + float(P["field"]["pace_step"]) * n, 3), n


def draw_line(base, seed, h, k, params=None, others=None, attempt=0):
    """(path samples [x, y, reference speed] rounded as in a world file, line weight, draws used) of slot k, heat h,
    at an attempt index."""
    P = load_params(params)
    op = load_others_params(others)
    fp = P["field"]
    lo, hi = fp["line_weight_log10_range"]
    r = stream(seed, P["streams"]["line_weight"], h, k, *attempt_keys(attempt, P))
    for attempt in range(int(fp["line_attempts"])):
        weight = 10.0 ** float(r.uniform(lo, hi))
        cand = check.round_floats(PL.plan_path(base, [], weight, op))
        wall, _bclear, rmin = PL.path_check(base, cand, [], op)
        if wall >= float(fp["line_wall_clear_min_m"]) and rmin >= float(op["path"]["min_radius_m"]):
            return cand, weight, attempt + 1
    raise FieldError("heat %d slot %d attempt %d: no line kept the clearances in %d draws"
                     % (h, k, int(attempt), fp["line_attempts"]))


# ---------------------------------------------------------------------------------------------- grid
def ref_point(pose):
    x, y, yaw = (float(v) for v in pose)
    return x + REF_X * math.cos(yaw), y + REF_X * math.sin(yaw)


def start_station(world, track=None):
    tr = track or check.Track(world)
    return tr.project(*ref_point(world["truth"]["start_pose"]["pose"]))[0]


def slot_station(s0, k, spacing, length):
    return (float(s0) - float(spacing) * (k - 1)) % float(length)


def signed_station_gap(a, b, length):
    """a - b wrapped into (-length / 2, length / 2]."""
    d = (a - b + 0.5 * length) % length - 0.5 * length
    return 0.5 * length if d == -0.5 * length else d


def line_station_at(path, track, s_target, tol=1e-9, max_iter=60):
    """The station on `path` of its point whose centreline station is `s_target` (Newton steps on a numerical slope,
    halved while they do not reduce the error). Where the path passes inside a corner of the centreline polyline the
    stations of its points skip a few millimetres; the point returned is then the one nearest to `s_target`."""
    L = track.length

    def err_at(sp):
        px, py = path.at(sp)[:2]
        return signed_station_gap(track.project(px, py)[0], s_target, L)

    x, y = track.at(s_target)[:2]
    sp = path.project(x, y)[0]
    e = err_at(sp)
    for _ in range(max_iter):
        if abs(e) < tol:
            break
        h = 1e-4
        slope = (err_at(sp + h) - err_at(sp - h)) / (2.0 * h)
        if not math.isfinite(slope) or slope < 0.05:
            slope = 1.0
        step = -e / slope
        for _k in range(30):
            cand = (sp + step) % path.length
            ec = err_at(cand)
            if abs(ec) < abs(e):
                sp, e = cand, ec
                break
            step *= 0.5
        else:
            break
    return sp


def pose_on_line(path, sp):
    """Car-frame pose [x, y, yaw] whose reference point is the path point at station sp, heading along the path."""
    x, y, h, _ = path.at(sp)
    return [x - REF_X * math.cos(h), y - REF_X * math.sin(h), check.wrap_angle(h)]


def car_slot_pose(track, s_m, side, lateral):
    x, y, h, _wl, _wr = track.at(s_m)
    rx, ry = x - side * lateral * math.sin(h), y + side * lateral * math.cos(h)
    return [rx - REF_X * math.cos(h), ry - REF_X * math.sin(h), check.wrap_angle(h)]


def grid_slots(base, field_samples, side, params=None):
    """The four slot records of one heat; `field_samples` = the three lines in slot order."""
    P = load_params(params)
    g = P["grid"]
    tr = check.Track(base)
    s0 = start_station(base, tr)
    out = []
    for k in range(1, int(g["slots"]) + 1):
        s_k = slot_station(s0, k, g["spacing_m"], tr.length)
        if k == int(g["car_slot"]):
            pose = car_slot_pose(tr, s_k, side, float(g["car_lateral_m"]))
        else:
            path = Path(field_samples[k - 1])
            pose = pose_on_line(path, line_station_at(path, tr, s_k))
        out.append({"slot": k, "pose": pose, "s_m": s_k})
    return out


# ---------------------------------------------------------------------------------------------- the world
def _check_base(base, P):
    if base.get("schema") != P["base_world_schema"]:
        raise FieldError("the base world's schema is %r, not %r" % (base.get("schema"), P["base_world_schema"]))
    if check.world_digest(base) != base["layout_digest"]:
        raise FieldError("the base world's layout digest does not match its content")
    if "head_to_head" in base["truth"] or "heat_laps" in base["venue_sheet"]:
        raise FieldError("the base world carries head-to-head additions already")


def additions(base, tier, params=None, others=None, info=None, attempts=None):
    """The `truth.head_to_head` block of a base world for a tier and the attempt indices of its two heats (floats as
    drawn; the world is rounded later)."""
    P = load_params(params)
    if tier not in P["tiers"]:
        raise ValueError("tier must be one of %s" % sorted(P["tiers"]))
    attempts = check_attempts(attempts, P)
    op = load_others_params(others)
    fp, sc = P["field"], op["slow_cars"]
    seed = others_seed(base)
    side = draw_car_side(seed, P)
    b = base["truth"]["bounds"]
    heats, slots = [], []
    lines_info = []
    for h, session in enumerate(fp["sessions"], start=1):
        field, samples = [], []
        a = attempts[h - 1]
        for k in range(1, int(fp["cars"]) + 1):
            smp, weight, used = draw_line(base, seed, h, k, P, op, attempt=a)
            nominal, factor, n = draw_pace(seed, h, k, tier, P, attempt=a)
            samples.append(smp)
            lines_info.append({"heat": h, "slot": k, "attempt": a, "line_weight": weight, "line_draws": used,
                               "pace_steps": n})
            field.append({"id": car_id(k, P), "slot": k, "nominal_factor": nominal, "speed_factor": factor,
                          "gap_keep_m": float(fp["gap_keep_m"]), "entry_s_m": None,
                          "staging_pose": [b["xmin"] - float(sc["staging_offset_m"]) - float(sc["staging_spacing_m"]) * (k - 1),
                                           b["ymin"] - float(sc["staging_offset_m"]), 0.0],
                          "path": {"closed": True, "samples": smp}})
        sl = grid_slots(base, samples, side, P)
        for car in field:
            car["entry_s_m"] = sl[car["slot"] - 1]["s_m"]
        heats.append({"session": session, "field": field})
        slots.append(sl)
    if info is not None:
        info["lines"] = lines_info
        info["car_side"] = side
    return {"tier": tier,
            "grid": {"spacing_m": float(P["grid"]["spacing_m"]), "car_lateral_m": float(P["grid"]["car_lateral_m"]),
                     "car_side": side, "slots": slots},
            "heats": heats,
            P["redraw"]["attempts_key"]: list(attempts)}


def add_head_to_head(base, tier, heat_laps=None, params=None, others=None, info=None, attempts=None):
    """A head-to-head world of a base world (see the module text) drawn with the given attempt indices [a_1, a_2]
    (None: [0, 0]) without any run; `heat_laps` None gives a draft world."""
    P = load_params(params)
    _check_base(base, P)
    hh = additions(base, tier, P, others, info=info, attempts=attempts)
    vs = copy.deepcopy(base["venue_sheet"])
    vs["heat_laps"] = None if heat_laps is None else int(heat_laps)
    truth = copy.deepcopy(base["truth"])
    truth["head_to_head"] = hh
    w = {"schema": P["world_schema"], "world_id": P["world_id_prefix"] + base["world_id"][2:],
         "base_world_id": base["world_id"], "base_layout_digest": base["layout_digest"],
         "generator": copy.deepcopy(base["generator"]), "layout_digest": None, "venue_sheet": vs, "truth": truth}
    w = check.round_floats(w)
    w["layout_digest"] = check.world_digest(w)
    return w


def _attempt_entry(a, rec):
    """What the generator's record keeps of one attempt: passed or not, the first contact, the trains, the whole
    field-alone record (its wall-clock seconds apart, so that two records of one draw compare equal)."""
    rec = dict(rec)
    wall = rec.pop("wall_s", None)
    return {"attempt": a, "pass": bool(rec["pass"]), "first_contact": rec.get("first_contact"),
            "why_not": list(rec.get("why_not", [])), "first_train_t_s": rec.get("first_train_t_s"),
            "alone": rec, "wall_s": wall}


def generate(base, tier, heat_laps=None, params=None, others=None, alone=None, run_s=None, info=None):
    """The generator of a head-to-head world with the redraw of a heat whose field fails alone (module text).

    `alone(world, h)` -> a field-alone record with a boolean `pass` (default: `alone.run_alone` for `run_s`, the field
    parameters' 300 s when None). Returns (world, record); record = {schema, world_id, base_world_id, tier,
    layout_digest, field_attempts, pass, heats: [{heat, attempt, pass, all_failed, attempts: [one per attempt made]}]};
    `pass` is true when both heats found a field that runs alone."""
    P = load_params(params)
    key = P["redraw"]["attempts_key"]
    if alone is None:
        from .alone import run_alone

        def alone(w, h):
            return run_alone(w, h, run_s=run_s, params=P, others=others)
    amax = attempts_max(P)
    attempts = [0, 0]
    heats = []
    for h in (1, 2):
        tried = []
        for a in range(amax + 1):
            attempts[h - 1] = a
            w = add_head_to_head(base, tier, params=P, others=others, attempts=attempts)
            tried.append(_attempt_entry(a, alone(w, h)))
            if tried[-1]["pass"]:
                break
        ok = tried[-1]["pass"]
        heats.append({"heat": h, "attempt": attempts[h - 1], "pass": ok, "all_failed": not ok, "attempts": tried})
    world = add_head_to_head(base, tier, heat_laps=heat_laps, params=P, others=others, info=info, attempts=attempts)
    record = {"schema": "racecar-h2h-field-draw/1", "world_id": world["world_id"], "base_world_id": base["world_id"],
              "tier": tier, "layout_digest": world["layout_digest"], key: list(attempts),
              "pass": all(x["pass"] for x in heats), "heats": heats}
    return world, record


def with_heat_laps(world, n, params=None):
    """A copy of a head-to-head world with N on its venue sheet and its layout digest renewed."""
    P = load_params(params)
    pr = P["predicates"]
    if isinstance(n, bool) or not isinstance(n, int) or not pr["laps_min"] <= n <= pr["laps_max"]:
        raise ValueError("heat_laps must be an integer from %d to %d" % (pr["laps_min"], pr["laps_max"]))
    w = copy.deepcopy(world)
    w["venue_sheet"]["heat_laps"] = n
    w["layout_digest"] = check.world_digest(w)
    return w


def base_view(world):
    """The base world inside a head-to-head world: additions removed, the base schema, id and digest restored."""
    P = load_params()
    vs = {k: copy.deepcopy(v) for k, v in world["venue_sheet"].items() if k != "heat_laps"}
    truth = {k: copy.deepcopy(v) for k, v in world["truth"].items() if k != "head_to_head"}
    return {"schema": P["base_world_schema"], "world_id": world["base_world_id"],
            "generator": copy.deepcopy(world["generator"]), "layout_digest": world["base_layout_digest"],
            "venue_sheet": vs, "truth": truth}


def heat_index(h):
    if isinstance(h, str):
        h = {"heat1": 1, "heat2": 2}[h]
    if h not in (1, 2):
        raise ValueError("a heat is 1 or 2 (heat1, heat2)")
    return int(h)


def field_of(world, h):
    return world["truth"]["head_to_head"]["heats"][heat_index(h) - 1]["field"]


def slots_of(world, h):
    return world["truth"]["head_to_head"]["grid"]["slots"][heat_index(h) - 1]


def heat_view(world, h):
    """The base world with truth.avoidance = {boxes: [], slow_cars: that heat's field} (the record the shared
    opponent code reads), under the head-to-head world id, its digest renewed. Never written as a world file."""
    v = base_view(world)
    v["world_id"] = world["world_id"]
    v["truth"]["avoidance"] = {"boxes": [], "slow_cars": copy.deepcopy(field_of(world, h))}
    v["layout_digest"] = check.world_digest(v)
    return v


def slot_path_station(world, h, k):
    """The station on the line of the field car in slot k of heat h at which its grid slot's reference point lies."""
    car = field_of(world, h)[k - 1]
    path = Path(car["path"]["samples"])
    pose = slots_of(world, h)[k - 1]["pose"]
    return path.project(*ref_point(pose))[0]
