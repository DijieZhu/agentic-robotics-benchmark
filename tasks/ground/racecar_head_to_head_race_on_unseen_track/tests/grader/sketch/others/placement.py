"""Placement rule of the avoidance set: three boxes and one or two slow cars per venue.

    from sketch.others import place_avoidance, with_avoidance
    av = place_avoidance(world, rng)              # the `truth.avoidance` block for a world whose track is finished
    world2 = with_avoidance(world, rng)           # a copy with it, floats rounded, layout_digest renewed

The venue generator calls it last, with a generator of its own seed (suggested: default_rng([seed, 5])); it needs
the centreline, the floor patches, the timing line and checkpoints, the open walls, the start pose and the bounds.
Everything below is a function of the world and the draws of `rng`, in this order: boxes (per try: station, three
sizes, mass, yaw, lateral offset), number of slow cars, then per car: speed factor, line weight, entry fraction.

Boxes: `box.count` boxes of size within `box.size_range_m` (1 cm steps) and mass
within `box.mass_range_kg` (0.05 kg steps), turned by up to +-90 deg against the driving direction, wholly inside the
track at least `wall_gap_min_m` from the walls, leaving a passage of at least 0.8 m + `passage_margin_m` between box
and wall (measured as the checker does). Not within `start_clear_ahead_m` ahead of / `start_clear_behind_m` behind the
start pose, not within `line_clear_m` of the timing line or a checkpoint, not within 1 m of an open wall, not on a
raised floor patch, and at least `station_gap_frac` of the lap from each other.

Slow-car path: the lateral offset d(s) from the centreline (sampled every `solve_spacing_m`) minimising
    sum |second difference of the path points|^2 + w sum d^2,     w = 10^U(smooth_weight_log10_range)
under bounds that keep the car's side `wall_clear_m` from the wall faces and, over the stations a box covers (plus the
car's length), `box_clear_m` from the box on the side of its wider passage (scipy lsq_linear). w small: a
curvature-minimising line; w large: the centreline. Resampled every `sample_m`; its column 3 is the reference speed
(path.reference_speed) at the true grip under it (the minimum within `grip_window_m`). Checked: the car's footprint
(tail box included) along the whole path keeps 0.06 m from the walls and 0.1 m from the boxes, the path's smallest
radius is at least `min_radius_m`; else the car's line weight is drawn again (up to 20 times).
"""

import copy
import math

import numpy as np

from sketch.car import load_params as load_drive_params
from sketch.examples import check
from sketch.examples import constants as K

from .params import load_params
from .path import car_extent, outline, reference_speed, signed_curvature, transform


class PlacementError(RuntimeError):
    pass


def _step_draw(rng, lo, hi, step):
    n = int(round((hi - lo) / step))
    return round(lo + step * int(rng.integers(0, n + 1)), 6)


def _station_dist(a, b, L):
    d = abs(a - b) % L
    return min(d, L - d)


def box_passage(tr, pose, size):
    """(free passage, inside) of a box exactly as the contract and check.check_world measure it."""
    corners = check.box_corners(pose, size[0], size[1])
    inside = all(tr.inside(x, y) for x, y in corners)
    s, _ = tr.project(pose[0], pose[1])
    offs = [tr.project(x, y)[1] for x, y in corners]
    _, _, _, wl, wr = tr.at(s)
    return max(wl - max(offs), min(offs) + wr), inside, s, offs


def _forbidden(world, tr, p):
    """Station intervals (lo, hi) where no box may stand."""
    t = world["truth"]
    bp = p["box"]
    L = tr.length
    out = []
    s0 = t["start_pose"]["s_m"]
    out.append((s0 - float(bp["start_clear_behind_m"]), s0 + K.X_FRONT_M + float(bp["start_clear_ahead_m"])))
    for ln in [t["timing_line"]] + list(t["checkpoints"]):
        out.append((ln["s_m"] - float(bp["line_clear_m"]), ln["s_m"] + float(bp["line_clear_m"])))
    for o in t["open_walls"]:
        out.append((o["s_m"] - 0.5 * o["width_m"] - 1.0, o["s_m"] + 0.5 * o["width_m"] + 1.0))
    return [((a % L), (a % L) + (b - a)) for a, b in out]


def _in_intervals(s, iv, L):
    for a, b in iv:
        if a <= s <= b or a <= s + L <= b:
            return True
    return False


def _on_raised_patch(world, corners):
    for pt in world["truth"]["floor"]["patches"]:
        if pt["edge_height_m"] <= 0.0:
            continue
        poly = pt["polygon"]
        if any(check.point_in_polygon(x, y, poly) for x, y in corners):
            return True
        if any(check.point_in_polygon(x, y, corners) for x, y in poly):
            return True
    return False


def place_boxes(world, rng, params=None):
    """The `avoidance.boxes` list (see the module text)."""
    p = load_params(params)
    bp = p["box"]
    tr = check.Track(world)
    L = tr.length
    forb = _forbidden(world, tr, p)
    need = K.BOX_FREE_PASSAGE_M + float(bp["passage_margin_m"])
    gap_min = float(bp["wall_gap_min_m"])
    boxes = []
    tries = 0
    while len(boxes) < int(bp["count"]):
        tries += 1
        if tries > int(bp["max_tries"]):
            raise PlacementError("could not place %d boxes in %d tries (track too narrow?)" % (bp["count"], tries - 1))
        s = float(rng.uniform(0.0, L))
        size = [_step_draw(rng, lo, hi, float(bp["size_step_m"])) for lo, hi in bp["size_range_m"]]
        mass = _step_draw(rng, bp["mass_range_kg"][0], bp["mass_range_kg"][1], float(bp["mass_step_kg"]))
        dyaw = float(rng.uniform(-0.5 * math.pi, 0.5 * math.pi))
        u = float(rng.random())
        if _in_intervals(s, forb, L) or any(_station_dist(s, b["s_m"], L) < float(bp["station_gap_frac"]) * L for b in boxes):
            continue
        x, y, h, wl, wr = tr.at(s)
        yaw = check.wrap_angle(h + dyaw)
        # lateral extent of the box about its centre, across the track
        c, sn = math.cos(dyaw), math.sin(dyaw)
        e = 0.5 * (abs(sn) * size[0] + abs(c) * size[1])
        lo, hi = -wr + e + gap_min, wl - e - gap_min
        # sub-intervals where the left or the right passage is wide enough
        ivs = []
        left_ok_hi = wl - e - need                  # left passage = wl - (d + e) >= need
        right_ok_lo = -wr + e + need                # right passage = (d - e) + wr >= need
        if left_ok_hi >= lo:
            ivs.append((lo, min(hi, left_ok_hi)))
        if right_ok_lo <= hi:
            ivs.append((max(lo, right_ok_lo), hi))
        ivs = [(a, b) for a, b in ivs if b >= a]
        if not ivs:
            continue
        tot = sum(b - a for a, b in ivs)
        r = u * tot
        d = ivs[-1][1]
        for a, b in ivs:
            if r <= b - a:
                d = a + r
                break
            r -= b - a
        nx, ny = -math.sin(h), math.cos(h)
        pose = [x + d * nx, y + d * ny, yaw]
        passage, inside, s_c, offs = box_passage(tr, pose, size)
        corners = check.box_corners(pose, size[0], size[1])
        if not inside or passage < need - 1e-6:
            continue
        if not all(tr.inside(cx, cy, margin=gap_min - 1e-6) for cx, cy in corners):
            continue
        if _on_raised_patch(world, corners) or _in_intervals(s_c, forb, L):
            continue
        boxes.append({"id": "B%d" % len(boxes), "size_m": size, "mass_kg": mass, "pose": pose, "s_m": s_c,
                      "free_passage_m": passage})
    return boxes


# ---------------------------------------------------------------------------------------------- slow-car path
def _centre_samples(tr, h):
    n = int(math.ceil(tr.length / h))
    st = np.arange(n) * (tr.length / n)
    rows = [tr.at(float(s)) for s in st]
    c = np.array([[r[0], r[1]] for r in rows])
    hd = np.array([r[2] for r in rows])
    nrm = np.stack([-np.sin(hd), np.cos(hd)], axis=1)
    wl = np.array([r[3] for r in rows])
    wr = np.array([r[4] for r in rows])
    return st, c, nrm, wl, wr


def lateral_bounds(world, boxes, params=None, h=None):
    """Stations, centre points, normals and the bounds (lo, hi) of the path's lateral offset (see the module text)."""
    p = load_params(params)
    pp = p["path"]
    tr = check.Track(world)
    h = float(pp["solve_spacing_m"]) if h is None else h
    st, c, nrm, wl, wr = _centre_samples(tr, h)
    x0, x1, hw = car_extent(p["tail_box_centre_m"])
    side = hw + float(pp["wall_clear_m"])
    lo, hi = -wr + side, wl - side
    clear = hw + float(pp["box_clear_m"])
    L = tr.length
    for bx in boxes:
        corners = check.box_corners(bx["pose"], bx["size_m"][0], bx["size_m"][1])
        pr = [tr.project(x, y) for x, y in corners]
        ss = np.array([q[0] for q in pr])
        dd = np.array([q[1] for q in pr])
        # stations of the corners relative to the box centre's station (no wrap inside a box)
        sc = bx["s_m"]
        rel = (ss - sc + 0.5 * L) % L - 0.5 * L
        a = sc + rel.min() - (K.X_FRONT_M - K.REF_POINT_C[0]) - float(pp["box_clear_m"])
        b = sc + rel.max() + (K.REF_POINT_C[0] - x0) + float(pp["box_clear_m"])
        _, _, _, bwl, bwr = tr.at(sc)
        left_room = bwl - dd.max()
        right_room = dd.min() + bwr
        relst = (st - a) % L
        m = relst <= (b - a)
        if left_room >= right_room:
            lo[m] = np.maximum(lo[m], dd.max() + clear)
        else:
            hi[m] = np.minimum(hi[m], dd.min() - clear)
    if np.any(lo > hi + 1e-9):
        raise PlacementError("no lateral room for a slow car at %d stations" % int(np.sum(lo > hi + 1e-9)))
    return st, c, nrm, lo, hi


def solve_line(c, nrm, lo, hi, weight):
    """Lateral offsets d (bounded) minimising |second difference of c + d n|^2 + weight |d|^2 on a closed polyline."""
    from scipy.optimize import lsq_linear
    from scipy.sparse import coo_matrix, vstack, identity
    n = len(c)
    rows, cols, vals = [], [], []
    for i in range(n):
        for k, coef in ((i - 1, 1.0), (i, -2.0), (i + 1, 1.0)):
            j = k % n
            for ax in (0, 1):
                rows.append(2 * i + ax)
                cols.append(j)
                vals.append(coef * nrm[j, ax])
    A = coo_matrix((vals, (rows, cols)), shape=(2 * n, n)).tocsr()
    b = (np.roll(c, 1, axis=0) - 2.0 * c + np.roll(c, -1, axis=0)).reshape(-1)
    if weight > 0.0:
        A = vstack([A, math.sqrt(weight) * identity(n, format="csr")]).tocsr()
        b = np.concatenate([b, np.zeros(n)])
    lo2, hi2 = lo.copy(), hi.copy()
    eq = hi2 - lo2 < 1e-9
    hi2[eq] = lo2[eq] + 1e-9
    res = lsq_linear(A, -b, bounds=(lo2, hi2), method="trf", lsmr_tol="auto", tol=1e-10, max_iter=2000)
    return np.clip(res.x, lo, hi)


def resample_closed(pts, step):
    """Points of a closed polyline resampled at equal arc length close to `step`."""
    seg = np.roll(pts, -1, axis=0) - pts
    ln = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    total = float(cum[-1])
    n = max(16, int(math.ceil(total / step)))
    t = np.arange(n) * (total / n)
    i = np.searchsorted(cum, t, side="right") - 1
    u = (t - cum[i]) / ln[i]
    return pts[i] + seg[i] * u[:, None]


def path_check(world, samples, boxes, params=None):
    """(smallest wall clearance of the car's footprint along the path, smallest box clearance, smallest radius)."""
    p = load_params(params)
    tr = check.Track(world)
    x0, x1, hw = car_extent(p["tail_box_centre_m"])
    pts = np.asarray(samples, dtype=float)[:, :2]
    nxt, prv = np.roll(pts, -1, axis=0), np.roll(pts, 1, axis=0)
    hd = np.arctan2(nxt[:, 1] - prv[:, 1], nxt[:, 0] - prv[:, 0])
    ref = K.REF_POINT_C[0]
    loc = np.array([[x0, -hw], [x0, hw], [x1, -hw], [x1, hw], [0.5 * (x0 + x1), -hw], [0.5 * (x0 + x1), hw]])
    wall = math.inf
    for (px, py), h in zip(pts, hd):
        pose = [px - ref * math.cos(h), py - ref * math.sin(h), h]
        for qx, qy in transform(loc, pose):
            s, d = tr.project(qx, qy)
            _, _, _, wl, wr = tr.at(s)
            wall = min(wall, wl - d, d + wr)
    bclear = math.inf
    if boxes:
        dense = outline(x0, x1, hw, 0.05)
        polys = [np.asarray(check.box_corners(b["pose"], b["size_m"][0], b["size_m"][1])) for b in boxes]
        for (px, py), h in zip(pts, hd):
            cx, cy = px, py
            for poly in polys:
                if math.hypot(poly[:, 0].mean() - cx, poly[:, 1].mean() - cy) > 1.2:
                    continue
                pose = [px - ref * math.cos(h), py - ref * math.sin(h), h]
                car = transform(dense, pose)
                bclear = min(bclear, _poly_dist(car, poly))
    kap = np.abs(signed_curvature(np.roll(pts, 2, axis=0), pts, np.roll(pts, -2, axis=0)))
    rmin = float(1.0 / max(kap.max(), 1e-9))
    return wall, bclear, rmin


def _poly_dist(points, poly):
    """Smallest distance from points to a convex polygon (0 when a point is inside)."""
    best = math.inf
    n = len(poly)
    inside = np.ones(len(points), dtype=bool)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ab = b - a
        t = np.clip(((points - a) @ ab) / float(ab @ ab), 0.0, 1.0)
        d = np.linalg.norm(points - (a + t[:, None] * ab), axis=1)
        best = min(best, float(d.min()))
        cross = ab[0] * (points[:, 1] - a[1]) - ab[1] * (points[:, 0] - a[0])
        inside &= cross >= 0.0
    return 0.0 if inside.any() else best


def plan_path(world, boxes, weight, params=None, drive_params=None):
    """Path samples [x, y, reference speed] of a slow car (see the module text)."""
    p = load_params(params)
    pp, sp = p["path"], p["speed"]
    st, c, nrm, lo, hi = lateral_bounds(world, boxes, p)
    d = solve_line(c, nrm, lo, hi, weight)
    pts = resample_closed(c + d[:, None] * nrm, float(pp["sample_m"]))
    fm = check.FloorMap(world)
    g = np.array([fm.grip_at(float(x), float(y)) for x, y in pts])
    w = max(1, int(round(float(sp["grip_window_m"]) / float(pp["sample_m"]))))
    gmin = np.min(np.stack([np.roll(g, k) for k in range(-w, w + 1)]), axis=0)
    dp = drive_params or load_drive_params()
    v = reference_speed(pts, gmin, p, float(dp["top_speed_mps"]))
    return [[float(x), float(y), round(float(vv), 3)] for (x, y), vv in zip(pts, v)]


def place_slow_cars(world, boxes, rng, n_slow=None, params=None, drive_params=None):
    p = load_params(params)
    scp, pp = p["slow_cars"], p["path"]
    tr = check.Track(world)
    L = tr.length
    choices = list(scp["count_choices"])
    n = int(choices[int(rng.integers(0, len(choices)))]) if n_slow is None else int(n_slow)
    b = world["truth"]["bounds"]
    out = []
    frac0 = None
    for i in range(n):
        factor = _step_draw(rng, scp["speed_factor_range"][0], scp["speed_factor_range"][1], float(scp["speed_factor_step"]))
        lo_w, hi_w = pp["smooth_weight_log10_range"]
        frac = float(rng.uniform(*scp["entry_frac_range"]))
        if frac0 is None:
            frac0 = frac
        samples = None
        for attempt in range(20):
            weight = 10.0 ** float(rng.uniform(lo_w, hi_w))
            cand = plan_path(world, boxes, weight, p, drive_params)
            wall, bclear, rmin = path_check(world, cand, boxes, p)
            if wall >= 0.06 and bclear >= 0.1 and rmin >= float(pp["min_radius_m"]):
                samples = cand
                break
        if samples is None:
            raise PlacementError("slow car %d: no line kept the clearances in 20 draws" % i)
        entry = (world["truth"]["start_pose"]["s_m"] + L * (frac0 + i / float(n))) % L
        out.append({"id": "S%d" % i, "speed_factor": factor, "gap_keep_m": float(scp["gap_keep_m"]), "entry_s_m": entry,
                    "staging_pose": [b["xmin"] - float(scp["staging_offset_m"]) - float(scp["staging_spacing_m"]) * i,
                                     b["ymin"] - float(scp["staging_offset_m"]), 0.0],
                    "path": {"closed": True, "samples": samples}})
    return out


def place_avoidance(world, rng, n_slow=None, params=None, drive_params=None):
    """The `truth.avoidance` block: {"boxes": [...], "slow_cars": [...]} (see the module text)."""
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    boxes = place_boxes(world, rng, params)
    rounded = check.round_floats(boxes)
    cars = place_slow_cars(world, rounded, rng, n_slow=n_slow, params=params, drive_params=drive_params)
    return check.round_floats({"boxes": rounded, "slow_cars": cars})


def with_avoidance(world, rng, n_slow=None, params=None):
    """A copy of `world` with a new avoidance block, floats rounded and the layout digest renewed."""
    w = copy.deepcopy(world)
    w["truth"]["avoidance"] = {"boxes": [], "slow_cars": []}
    w["truth"]["avoidance"] = place_avoidance(w, rng, n_slow=n_slow, params=params)
    w = check.round_floats(w)
    w["layout_digest"] = check.world_digest(w)
    return w
