"""Checks of a head-to-head world: the schema of the additions, their consistency with the base world and the
rules of the draw, the grid predicates and the pace predicates.

    err = check_world_h2h(world)                 # [] when the file is a valid head-to-head world
    err = check_world_h2h(world, deep=True)      # also: the additions are what the draw gives for the base world
                                                 # with the attempt indices the world records
    gp = grid_predicates(world)                  # per heat: wheel margin of the car's slot, slot gaps, raised edges
    pp = pace_predicates(world)                  # per heat: paces within the tolerance of their nominals, order

Grid predicates (each heat): every wheel ground point of the car's slot at least `wheel_margin_m` inside the track
corridor; the footprints of the four slots (field cars with their marker box) at least `slot_clearance_m` from each
other; no slot footprint across the boundary of a floor patch with a raised edge.
"""

import json
import math
import os

import numpy as np

from sketch.examples import check
from sketch.examples import constants as K
from sketch.others import placement as PL
from sketch.others.params import load_params as load_others_params
from sketch.others.path import Path, car_extent, transform

from . import world as W

SCHEMA_PATH = os.path.join(W.HERE, "world_h2h.schema.json")


def _schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------------- geometry
def footprint(pose, tail_box=False, others=None):
    """Corners (4, 2) of a car's footprint at a car-frame pose; with the marker box for a field car."""
    op = load_others_params(others)
    x0, x1, hw = car_extent(op["tail_box_centre_m"] if tail_box else None)
    loc = np.array([[x0, -hw], [x1, -hw], [x1, hw], [x0, hw]], dtype=float)
    return transform(loc, [float(v) for v in pose])


def _segments_cross(a, b, c, d):
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on(p, q, r):
        return min(p[0], q[0]) - 1e-12 <= r[0] <= max(p[0], q[0]) + 1e-12 and \
            min(p[1], q[1]) - 1e-12 <= r[1] <= max(p[1], q[1]) + 1e-12

    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)) and o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0:
        return True
    return (o1 == 0 and on(a, b, c)) or (o2 == 0 and on(a, b, d)) or (o3 == 0 and on(c, d, a)) or (o4 == 0 and on(c, d, b))


def _point_segment(p, a, b):
    ab = (b[0] - a[0], b[1] - a[1])
    den = ab[0] * ab[0] + ab[1] * ab[1]
    u = 0.0 if den <= 0.0 else max(0.0, min(1.0, ((p[0] - a[0]) * ab[0] + (p[1] - a[1]) * ab[1]) / den))
    return math.hypot(p[0] - (a[0] + u * ab[0]), p[1] - (a[1] + u * ab[1]))


def _edges(poly):
    n = len(poly)
    return [(tuple(poly[i]), tuple(poly[(i + 1) % n])) for i in range(n)]


def polygons_cross(p, q):
    """True when the boundaries of two polygons meet."""
    return any(_segments_cross(a, b, c, d) for a, b in _edges(p) for c, d in _edges(q))


def polygon_distance(p, q):
    """Smallest distance between two simple polygons (0 when they overlap)."""
    if polygons_cross(p, q):
        return 0.0
    if any(check.point_in_polygon(float(x), float(y), [list(v) for v in q]) for x, y in p) or \
            any(check.point_in_polygon(float(x), float(y), [list(v) for v in p]) for x, y in q):
        return 0.0
    best = math.inf
    for pt in p:
        for a, b in _edges(q):
            best = min(best, _point_segment(pt, a, b))
    for pt in q:
        for a, b in _edges(p):
            best = min(best, _point_segment(pt, a, b))
    return best


def corridor_margin(track, x, y):
    """Distance of a point from the nearer corridor boundary (negative outside)."""
    s, d = track.project(x, y)
    _, _, _, wl, wr = track.at(s)
    return min(wl - d, d + wr)


def on_raised_edge(poly, world):
    """Ids of the floor patches with a raised edge whose boundary passes under the polygon."""
    out = []
    for pt in world["truth"]["floor"]["patches"]:
        if float(pt.get("edge_height_m", 0.0)) <= 0.0:
            continue
        patch = [list(v) for v in pt["polygon"]]
        if polygons_cross(poly, patch) or any(check.point_in_polygon(float(x), float(y), [list(v) for v in poly])
                                              for x, y in patch):
            out.append(pt["id"])
    return out


# ---------------------------------------------------------------------------------------------- predicates
def grid_predicates(world, params=None, others=None):
    P = W.load_params(params)
    pr = P["predicates"]
    car_slot = int(P["grid"]["car_slot"])
    tr = check.Track(world)
    heats = []
    for h in (1, 2):
        slots = W.slots_of(world, h)
        polys, margins = [], []
        for sl in slots:
            is_car = sl["slot"] == car_slot
            polys.append(footprint(sl["pose"], tail_box=not is_car, others=others))
            if is_car:
                for w_ in K.WHEELS:
                    gx, gy, _ = K.WHEEL_GROUND_C[w_]
                    x, y = transform(np.array([[gx, gy]]), sl["pose"])[0]
                    margins.append(corridor_margin(tr, float(x), float(y)))
        gaps = []
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                gaps.append(polygon_distance(polys[i], polys[j]))
        raised = {sl["slot"]: on_raised_edge(poly, world) for sl, poly in zip(slots, polys)}
        raised = {k: v for k, v in raised.items() if v}
        wm, sg = min(margins), min(gaps)
        heats.append({"heat": h, "car_wheel_margin_m": round(wm, 6), "min_slot_gap_m": round(sg, 6),
                      "raised_edge_slots": raised,
                      "wheel_ok": wm >= float(pr["wheel_margin_m"]) - 1e-9,
                      "gap_ok": sg >= float(pr["slot_clearance_m"]) - 1e-9, "edge_ok": not raised})
        heats[-1]["pass"] = heats[-1]["wheel_ok"] and heats[-1]["gap_ok"] and heats[-1]["edge_ok"]
    return {"heats": heats, "pass": all(x["pass"] for x in heats)}


def pace_predicates(world, params=None):
    P = W.load_params(params)
    hh = world["truth"]["head_to_head"]
    nominal = P["tiers"][hh["tier"]]
    tol = float(P["predicates"]["pace_tolerance"])
    heats = []
    for h in (1, 2):
        f = W.field_of(world, h)
        dev = [abs(c["speed_factor"] - nominal[c["slot"] - 1]) for c in f]
        order = all(f[i]["speed_factor"] > f[i + 1]["speed_factor"] for i in range(len(f) - 1))
        heats.append({"heat": h, "paces": [c["speed_factor"] for c in f], "max_deviation": round(max(dev), 6),
                      "within": max(dev) <= tol + 1e-9, "fastest_first": order})
        heats[-1]["pass"] = heats[-1]["within"] and heats[-1]["fastest_first"]
    return {"heats": heats, "pass": all(x["pass"] for x in heats)}


# ---------------------------------------------------------------------------------------------- the checker
def check_world_h2h(world, deep=False, params=None, others=None, base_check=True):
    """Errors of a head-to-head world (empty list when it is valid). `deep` also redraws the additions from the
    base view and compares them; `base_check` runs the shared world checker on the base view."""
    P = W.load_params(params)
    op = load_others_params(others)
    err = check.schema_errors(world, _schema())
    if err:
        return err
    pr = P["predicates"]
    if world["world_id"] != P["world_id_prefix"] + world["base_world_id"][2:]:
        err.append("world_id is not %s + the base world id after its prefix" % P["world_id_prefix"])
    if world["layout_digest"] != check.world_digest(world):
        err.append("layout_digest does not match the content")
    base = W.base_view(world)
    if check.world_digest(base) != world["base_layout_digest"]:
        err.append("the base view's digest is not base_layout_digest")
    if base_check:
        err += ["base view: " + e for e in check.check_world(base)]
    hh = world["truth"]["head_to_head"]
    g = hh["grid"]
    if abs(g["spacing_m"] - float(P["grid"]["spacing_m"])) > 1e-9 or abs(g["car_lateral_m"] - float(P["grid"]["car_lateral_m"])) > 1e-9:
        err.append("grid spacing or the car's lateral offset differ from the field parameters")
    if g["car_side"] != W.draw_car_side(W.others_seed(world), P):
        err.append("grid.car_side is not the draw of the venue's seed")
    tr = check.Track(world)
    L = tr.length
    s0 = W.start_station(world, tr)
    nominal = P["tiers"][hh["tier"]]
    fp = P["field"]
    st_tol, ps_tol = float(pr["station_tolerance_m"]), float(pr["pose_tolerance_m"])
    car_slot = int(P["grid"]["car_slot"])
    for h in (1, 2):
        heat = hh["heats"][h - 1]
        tag = "heat %d" % h
        if heat["session"] != fp["sessions"][h - 1]:
            err.append("%s: session is %r" % (tag, heat["session"]))
        field = heat["field"]
        slots = g["slots"][h - 1]
        if [s["slot"] for s in slots] != list(range(1, int(P["grid"]["slots"]) + 1)):
            err.append("%s: slots are not 1 to 4 in order" % tag)
            continue
        for k, c in enumerate(field, start=1):
            ct = "%s %s" % (tag, c["id"])
            if c["id"] != W.car_id(k, P) or c["slot"] != k:
                err.append("%s: id or slot out of order" % ct)
            if abs(c["nominal_factor"] - nominal[k - 1]) > 1e-9:
                err.append("%s: nominal_factor %s is not the tier's %s" % (ct, c["nominal_factor"], nominal[k - 1]))
            steps = (c["speed_factor"] - nominal[k - 1]) / float(fp["pace_step"])
            if abs(steps - round(steps)) > 1e-6 or abs(round(steps)) > int(fp["pace_half_steps"]):
                err.append("%s: speed_factor %s is not the nominal plus a whole number of steps within the draw" % (ct, c["speed_factor"]))
            if abs(c["gap_keep_m"] - float(fp["gap_keep_m"])) > 1e-9:
                err.append("%s: gap_keep_m differs from the field parameters" % ct)
            if abs(W.signed_station_gap(c["entry_s_m"], slots[k - 1]["s_m"], L)) > 1e-6:
                err.append("%s: entry_s_m is not its slot's station" % ct)
            arr = np.asarray(c["path"]["samples"], dtype=float)
            if not np.isfinite(arr).all():
                err.append("%s: path has a non-finite number" % ct)
                continue
            wall, _bc, rmin = PL.path_check(world, c["path"]["samples"], [], op)
            if wall < float(fp["line_wall_clear_min_m"]) - 1e-9 or rmin < float(op["path"]["min_radius_m"]) - 1e-9:
                err.append("%s: path check fails (wall clearance %.4f m, smallest radius %.3f m)" % (ct, wall, rmin))
            # the field car's slot: on its own line at the slot's station, heading along the line
            path = Path(c["path"]["samples"])
            pose = slots[k - 1]["pose"]
            rx, ry = W.ref_point(pose)
            sp, lat, d2 = path.project(rx, ry)
            if math.sqrt(d2) > ps_tol:
                err.append("%s: slot pose is %.4f m off its line" % (ct, math.sqrt(d2)))
            if abs(W.signed_station_gap(tr.project(rx, ry)[0], slots[k - 1]["s_m"], L)) > st_tol:
                err.append("%s: slot pose's station differs from s_m" % ct)
            if abs(check.wrap_angle(pose[2] - path.at(sp)[2])) > 1e-3:
                err.append("%s: slot pose does not head along its line" % ct)
        if [c["speed_factor"] for c in field] != sorted([c["speed_factor"] for c in field], reverse=True) or \
                len(set(c["speed_factor"] for c in field)) != len(field):
            err.append("%s: the field is not fastest first" % tag)
        for sl in slots:
            want = W.slot_station(s0, sl["slot"], g["spacing_m"], L)
            if abs(W.signed_station_gap(sl["s_m"], want, L)) > st_tol:
                err.append("%s slot %d: s_m %.6f is not the start station less %.1f m per slot" % (tag, sl["slot"], sl["s_m"], g["spacing_m"]))
        cs = slots[car_slot - 1]
        want = W.car_slot_pose(tr, cs["s_m"], g["car_side"], g["car_lateral_m"])
        if math.hypot(cs["pose"][0] - want[0], cs["pose"][1] - want[1]) > ps_tol or abs(check.wrap_angle(cs["pose"][2] - want[2])) > 1e-4:
            err.append("%s: the car's slot is not %.2f m to its side of the centreline at its station" % (tag, g["car_lateral_m"]))
    if g["slots"][0][car_slot - 1] != g["slots"][1][car_slot - 1]:
        err.append("the car's slot differs between the heats")
    key = P["redraw"]["attempts_key"]
    try:
        attempts = W.check_attempts(hh.get(key), P) if key in hh else None
    except ValueError as e:
        attempts = None
        err.append("%s: %s" % (key, e))
    if attempts is None and key not in hh:
        err.append("%s is missing" % key)
    if deep and not err:
        again = check.round_floats(W.additions(base, hh["tier"], P, op, attempts=attempts))
        if check.canonical_json(again) != check.canonical_json(hh):
            for h in (1, 2):
                if check.canonical_json(again["heats"][h - 1]) != check.canonical_json(hh["heats"][h - 1]) or \
                        check.canonical_json(again["grid"]["slots"][h - 1]) != check.canonical_json(g["slots"][h - 1]):
                    err.append("heat %d: the field and grid are not what the draw gives for the base world, tier %s "
                               "and attempt index %d" % (h, hh["tier"], attempts[h - 1]))
            if not err:
                err.append("the additions are not what the draw gives for the base world and tier %s" % hh["tier"])
    return err
