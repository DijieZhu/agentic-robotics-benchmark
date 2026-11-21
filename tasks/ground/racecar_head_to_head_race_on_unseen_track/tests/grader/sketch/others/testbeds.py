"""Test venues for the obstacle model's gates and tests while the venue generator does not exist.

    from sketch.others.testbeds import make_testbed, TESTBED_SEEDS, DEV_SEEDS
    world = make_testbed(seed, grip_tier="mid", layout="classic")      # a world file that passes check.check_world

NOT the venue generator and not part of any hidden set: a star-shaped closed centreline (12 to 18 control points with
random radii, a periodic cubic spline, stretches with a radius below 1.0 m relaxed by local Laplacian smoothing, scaled
to 62-78 m or 92-108 m), a smooth width profile of 1.0 to 2.2 m that narrows in bends (in `featured` with a 1.0 m
section), the ducts of each wall all with the same number of pieces (at most 15) and one piece length solved so that
the wall closes, end gaps of 0 to 12 cm, floor patches (a dust patch on the tightest bend, on the driving line; in
`featured` also a raised carpet tile with a 5 mm edge across the track), timing line, three checkpoints, start pose,
and the avoidance set of `placement.place_avoidance`. Layout class `example`: the footprints (about 22-30 x 16-25 m,
33-44 x 25-32 m) are larger than the graded classes allow; no open walls, no shared walls, no clutter. A layout that
`check.check_world` rejects, or whose walls come within 0.3 m of another stretch of the track, is skipped (the next
sub-seed is tried). `DEV_SEEDS` were used while the follower was written; `TESTBED_SEEDS` only by the gates.
"""

import math

import numpy as np

from sketch.examples import check
from sketch.examples import constants as K

from .placement import place_avoidance

DEV_SEEDS = (101, 102, 103)
TESTBED_SEEDS = (1, 2, 3, 4, 5, 6, 7, 8)
TIERS = {"low": ("sealed_concrete", 0.50), "mid": ("polished_concrete", 0.675), "high": ("exhibition_carpet", 0.875)}
R = K.DUCT_RADIUS_M


def _centreline(rng, layout):
    """Dense closed centreline (m, 2), counter-clockwise, scaled to a length of the class range, or None."""
    from scipy.interpolate import CubicSpline
    k = int(rng.integers(12, 19))
    ang = np.sort((np.arange(k) + rng.uniform(-0.3, 0.3, k)) * 2.0 * math.pi / k)
    rad = 1.0 + rng.uniform(-0.45, 0.45, k)
    ex = float(rng.uniform(1.1, 1.5))
    pts = np.stack([ex * rad * np.cos(ang), rad * np.sin(ang)], axis=1)
    closed = np.vstack([pts, pts[:1]])
    t = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    cs = CubicSpline(t, closed, bc_type="periodic")
    tt = np.linspace(0.0, t[-1], 4000, endpoint=False)
    dense = cs(tt)
    length = float(np.linalg.norm(np.roll(dense, -1, axis=0) - dense, axis=1).sum())
    lo, hi = (62.0, 78.0) if layout == "classic" else (92.0, 108.0)
    target = float(rng.uniform(lo, hi))
    dense = dense * (target / length)
    x, y = dense[:, 0], dense[:, 1]
    if 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) < 0.0:
        dense = dense[::-1].copy()
    return dense


def _resample(pts, step):
    seg = np.roll(pts, -1, axis=0) - pts
    ln = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(ln)])
    total = float(cum[-1])
    n = int(math.ceil(total / step))
    tt = np.arange(n) * (total / n)
    i = np.searchsorted(cum, tt, side="right") - 1
    u = (tt - cum[i]) / ln[i]
    return pts[i] + seg[i] * u[:, None]


def _curv(p, w=3):
    from .path import signed_curvature
    return signed_curvature(np.roll(p, w, axis=0), p, np.roll(p, -w, axis=0))


def _widths(rng, p, layout):
    """Total width per sample: smooth, 1.0 .. 2.2, narrower in bends, a 1.0 m section in `featured`."""
    n = len(p)
    s = np.arange(n) / n
    w = 1.75 + 0.3 * np.sin(2 * math.pi * (2 * s + rng.uniform())) + 0.15 * np.sin(2 * math.pi * (5 * s + rng.uniform()))
    if layout == "featured":
        c = rng.uniform()
        dist = np.minimum(np.abs(s - c), 1.0 - np.abs(s - c)) * len(p) * 0.1
        w = np.minimum(w, 1.0 + 0.08 * np.maximum(dist - 2.0, 0.0) ** 1.5)
    kap = np.abs(_curv(p))
    # the inner duct axis keeps 0.4 m of radius: half width + R <= 1/kappa - 0.4
    cap = 2.0 * (1.0 / np.maximum(kap, 1e-6) - 0.4 - R)
    w = np.minimum(w, cap)
    # smooth (running minimum then mean over +-1.5 m) so that the cap does not make steps
    k = 15
    wm = np.min(np.stack([np.roll(w, j) for j in range(-k, k + 1)]), axis=0)
    w = np.mean(np.stack([np.roll(wm, j) for j in range(-k, k + 1)]), axis=0)
    return np.clip(w, 1.0, 2.2)


def _relax(p, length, rmin=1.0, iters=400):
    """Smooth the stretches whose radius is below `rmin` (Laplacian steps on those samples and their neighbours),
    keep the length, resample at 0.1 m; None when it does not get there."""
    for _ in range(iters):
        kap = np.abs(_curv(p))
        bad = kap > 1.0 / rmin
        if not bad.any():
            return _resample(p * (length / float(np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1).sum())), 0.1)
        m = bad.copy()
        for j in range(1, 6):
            m |= np.roll(bad, j) | np.roll(bad, -j)
        lap = np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0) - 2.0 * p
        p = p + 0.4 * lap * m[:, None]
    return None


def _proximity_ok(p, w):
    """No wall comes within 0.3 m of another stretch of the track (more than 4 m away along it)."""
    n = len(p)
    step = 4
    idx = np.arange(0, n, step)
    q = p[idx]
    d = np.linalg.norm(q[:, None, :] - q[None, :, :], axis=2)
    sep = np.abs(idx[:, None] - idx[None, :])
    sep = np.minimum(sep, n - sep) * 0.1
    need = 0.5 * (w[idx][:, None] + w[idx][None, :]) + 4 * R + 0.3
    return not bool(np.any((sep > 4.0) & (d < need)))


class _Walker(object):
    """Walks chords along a closed polyline: from the present point, the next point of the polyline that is a given
    straight distance away (exact on the segment where it lies)."""

    def __init__(self, line):
        self.v = [tuple(map(float, q)) for q in line]
        self.n = len(self.v)
        self.len = [math.dist(self.v[i], self.v[(i + 1) % self.n]) for i in range(self.n)]
        self.total = sum(self.len)

    def start(self):
        self.i, self.t, self.lap = 0, 0.0, 0
        return self.point()

    def point(self):
        a, b = self.v[self.i], self.v[(self.i + 1) % self.n]
        return (a[0] + self.t * (b[0] - a[0]), a[1] + self.t * (b[1] - a[1]))

    def u(self):
        return self.lap * self.total + sum(self.len[:self.i]) + self.t * self.len[self.i]

    def advance(self, chord):
        p0 = self.point()
        i, t0 = self.i, self.t
        for _ in range(4 * self.n):
            a, b = self.v[i], self.v[(i + 1) % self.n]
            if math.dist(b, p0) >= chord:
                dx, dy = b[0] - a[0], b[1] - a[1]
                fx, fy = a[0] - p0[0], a[1] - p0[1]
                qa = dx * dx + dy * dy
                qb = 2.0 * (fx * dx + fy * dy)
                qc = fx * fx + fy * fy - chord * chord
                t = (-qb + math.sqrt(max(qb * qb - 4.0 * qa * qc, 0.0))) / (2.0 * qa)
                self.i, self.t = i, max(t, t0)
                return self.point()
            t0 = 0.0
            i += 1
            if i == self.n:
                i = 0
                self.lap += 1
        raise RuntimeError("chord walk did not converge")


def _lay_wall(walker, plan, c_last):
    """Walk the plan (('duct', n, c|None) and ('gap', width) items) along a closed axis polyline."""
    a = walker.start()
    ducts = []
    for item in plan:
        if item[0] == "duct":
            n, c = item[1], (c_last if item[2] is None else item[2])
            pieces = []
            for _ in range(n):
                b = walker.advance(c)
                pieces.append([0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]), math.atan2(b[1] - a[1], b[0] - a[0])])
                a = b
            ducts.append({"c": c, "pieces": pieces})
        elif item[1] > 0.0:
            a = walker.advance(item[1])
    return ducts, None, walker.u(), walker.total


def _wall(line, rng, wall_id, side, prefix, classes):
    """The ducts and gaps of one closed wall along its axis polyline."""
    wk = _Walker(line)
    total = wk.total
    n_d = int(math.ceil(total / K.DUCT_LENGTH_M))
    widths = [float(rng.choice([0.0, 0.0, rng.uniform(0.02, 0.12)])) for _ in range(n_d)]
    base = total - sum(widths)
    n_p = min(K.DUCT_PIECES_FULL, int(round(base / (n_d * K.PIECE_LENGTH_NOMINAL_M))))
    for _ in range(3):
        plan = []
        for k in range(n_d):
            plan += [("duct", n_p, None), ("gap", widths[k])]
        lo, hi = K.PIECE_LENGTH_RANGE_M
        for _ in range(60):                      # one piece length for the whole wall, solved so that it closes
            mid = 0.5 * (lo + hi)
            if _lay_wall(wk, plan, mid)[2] < total:
                lo = mid
            else:
                hi = mid
        c_last = 0.5 * (lo + hi)
        if n_p * c_last <= K.DUCT_LENGTH_M:
            break
        n_p -= 1
    ducts, gaps, u_end, _ = _lay_wall(wk, plan, c_last)
    if (abs(u_end - total) > 1e-6 or not K.PIECE_LENGTH_RANGE_M[0] <= c_last <= K.PIECE_LENGTH_RANGE_M[1]
            or n_p * c_last > K.DUCT_LENGTH_M):
        return None
    out_d, out_g = [], []
    ids = ["%s%d" % (prefix, k) for k in range(len(ducts))]
    for k, d in enumerate(ducts):
        out_d.append({"id": ids[k], "wall": wall_id, "wall_class": classes[k % len(classes)], "diameter_m": K.DUCT_DIAMETER_M,
                      "length_m": len(d["pieces"]) * d["c"], "piece_length_m": d["c"], "pieces": d["pieces"]})
    for k, d in enumerate(ducts):
        nxt = ducts[(k + 1) % len(ducts)]
        head = check.piece_ends(d["pieces"][-1], d["c"])[1]
        tail = check.piece_ends(nxt["pieces"][0], nxt["c"])[0]
        out_g.append({"id": "G%s%d" % (prefix, k), "wall": wall_id, "ducts": [ids[k], ids[(k + 1) % len(ids)]],
                      "ends": [list(head), list(tail)], "width_m": math.dist(head, tail)})
    return {"id": wall_id, "side": side, "closed": True, "ducts": ids}, out_d, out_g


def _rect(cx, cy, yaw, lx, ly):
    return [list(v) for v in check.box_corners([cx, cy, yaw], lx, ly)]


def make_testbed(seed, grip_tier="mid", layout="classic", n_slow=2, with_avoid=True):
    """A test venue (see the module text) for an integer seed; deterministic."""
    for sub in range(80):
        rng = np.random.default_rng([int(seed), 0x5E5, sub, 0 if layout == "classic" else 1])
        dense = _centreline(rng, layout)
        p = _relax(_resample(dense, 0.1), float(np.linalg.norm(np.roll(dense, -1, axis=0) - dense, axis=1).sum()))
        if p is None:
            continue
        w = _widths(rng, p, layout)
        if not _proximity_ok(p, w):
            continue
        world = _assemble(rng, seed, sub, p, w, grip_tier, layout)
        if world is None:
            continue
        world = check.round_floats(world)
        world["layout_digest"] = check.world_digest(world)
        if check.check_world(world):
            continue
        if with_avoid:
            world["truth"]["avoidance"] = place_avoidance(world, np.random.default_rng([int(seed), 0xA70]), n_slow=n_slow)
        world = check.round_floats(world)
        world["layout_digest"] = check.world_digest(world)
        return world
    raise RuntimeError("testbed seed %d: no layout in 80 sub-seeds" % seed)


def _assemble(rng, seed, sub, p, w, grip_tier, layout):
    # frame: centre the footprint of the walls on the origin, longer side along x
    t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nl = np.stack([-t[:, 1], t[:, 0]], axis=1)
    outer = np.concatenate([p + (0.5 * w + 2 * R)[:, None] * nl, p - (0.5 * w + 2 * R)[:, None] * nl])
    span = outer.max(axis=0) - outer.min(axis=0)
    if span[1] > span[0]:
        rot = np.array([[0.0, -1.0], [1.0, 0.0]])
        p, outer = p @ rot.T, outer @ rot.T
    centre = 0.5 * (outer.max(axis=0) + outer.min(axis=0))
    p = p - centre
    t = np.roll(p, -1, axis=0) - np.roll(p, 1, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    nl = np.stack([-t[:, 1], t[:, 0]], axis=1)
    half = 0.5 * w
    classes_l = ["orange_ribbed", "yellow_ribbed"]
    classes_r = ["yellow_ribbed", "orange_ribbed"] + (["black_plain"] if layout == "featured" else [])
    wl = _wall(p + (half + R)[:, None] * nl, rng, "W_IN", "left", "IN", classes_l)
    wr = _wall(p - (half + R)[:, None] * nl, rng, "W_OUT", "right", "OUT", classes_r)
    if wl is None or wr is None:
        return None
    ducts = wl[1] + wr[1]
    allp = np.array([e for d in ducts for pc in d["pieces"] for e in check.piece_ends(pc, d["piece_length_m"])])
    xmax = float(max(abs(allp[:, 0].min()), abs(allp[:, 0].max()))) + R
    ymax = float(max(abs(allp[:, 1].min()), abs(allp[:, 1].max()))) + R
    kind, base = TIERS[grip_tier]
    samples = [[float(x), float(y), float(h), float(h)] for (x, y), h in zip(p, half)]
    world = {
        "schema": K.WORLD_SCHEMA, "world_id": "tb_%s_%s_%d" % (layout, grip_tier, seed),
        "generator": {"name": "s5_testbed", "version": "S5-1", "seed_hex": None}, "layout_digest": "",
        "venue_sheet": {"venue_label": "Test hall %d" % seed, "floor": {"kind": kind, "side_pull": None},
                        "walls": {"kinds": list(K.WALL_CLASSES), "height_m": K.DUCT_DIAMETER_M},
                        "track": {"min_width_m": math.floor(float(w.min()) * 10.0) / 10.0,
                                  "max_footprint_m": [math.ceil(2 * xmax), math.ceil(2 * ymax)]},
                        "features": []},
        "truth": {"cell": {"grip_tier": grip_tier, "layout_class": "example"},
                  "bounds": {"xmin": -xmax, "ymin": -ymax, "xmax": xmax, "ymax": ymax}, "stats": {},
                  "centreline": {"closed": True, "direction": "ccw", "length_m": 0.0, "samples": samples},
                  "walls": [wl[0], wr[0]], "ducts": ducts, "gaps": wl[2] + wr[2], "open_walls": [],
                  "floor": {"kind": kind, "base_grip": base, "patches": []}, "clutter": [],
                  "timing_line": None, "checkpoints": [], "start_pose": None,
                  "avoidance": {"boxes": [], "slow_cars": []},
                  "seeds": {"relocation": 1000 + seed, "putback": 2000 + seed, "sensors": 3000 + seed, "others": 4000 + seed}}}
    world = check.round_floats(world)
    tw = world["truth"]
    tr = check.Track(world)
    L = tr.length
    tw["centreline"]["length_m"] = L

    def cross(s):
        a, b = tr.boundary(s, "left"), tr.boundary(s, "right")
        h = tr.at(s)[2]
        return {"s_m": s, "a": list(a), "b": list(b), "forward": [math.cos(h), math.sin(h)]}

    tw["timing_line"] = cross(0.0)
    tw["checkpoints"] = [dict(id="CP%d" % (i + 1), **cross(L * f)) for i, f in enumerate((0.25, 0.5, 0.75))]
    sx, sy, sh, _, _ = tr.at(L - 1.0)
    tw["start_pose"] = {"pose": [sx, sy, sh], "s_m": L - 1.0}
    # floor patches: dust on the tightest bend (on the driving line), a raised carpet tile in `featured`
    kap = np.abs(_curv(np.asarray([r[:2] for r in tw["centreline"]["samples"]])))
    i_b = int(np.argmax(kap))
    s_b = float(tr.s[i_b])
    bx, by, bh, bwl, bwr = tr.at(s_b)
    dust = {"id": "P0", "kind": "dust", "polygon": _rect(bx, by, bh, 1.4, 0.8 * (bwl + bwr)),
            "grip": round(max(0.3, base - 0.15), 3), "edge_height_m": 0.0, "on_racing_line": True, "s_m": s_b}
    patches = [dust]
    if layout == "featured":
        s_c = (s_b + 0.5 * L) % L
        cx, cy, ch, cwl, cwr = tr.at(s_c)
        patches.append({"id": "P1", "kind": "carpet_tile", "polygon": _rect(cx, cy, ch, 2.0, cwl + cwr + 0.2),
                        "grip": round(min(0.95, base + 0.15), 3), "edge_height_m": 0.005, "on_racing_line": True, "s_m": s_c})
    tw["floor"]["patches"] = patches
    world = check.round_floats(world)
    tw = world["truth"]
    for pt in tw["floor"]["patches"]:
        v = np.asarray(pt["polygon"])
        pt["s_m"] = tr.project(float(v[:, 0].mean()), float(v[:, 1].mean()))[0]
    for ln in [tw["timing_line"]] + tw["checkpoints"]:
        ln["s_m"] = tr.project(0.5 * (ln["a"][0] + ln["b"][0]), 0.5 * (ln["a"][1] + ln["b"][1]))[0]
    tw["start_pose"]["s_m"] = tr.project(*tw["start_pose"]["pose"][:2])[0]
    width = tr.left + tr.right
    b = tw["bounds"]
    tw["stats"] = {"length_m": tr.length, "min_width_m": float(width.min()),
                   "width_ge_1p5_frac": float(tr.seg_len[width >= 1.5 - 1e-9].sum() / tr.length),
                   "min_radius_m": tr.min_radius(), "footprint_m": [b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]],
                   "n_ducts": len(tw["ducts"]), "n_pieces": sum(len(d["pieces"]) for d in tw["ducts"])}
    world["venue_sheet"]["features"] = sorted(check.expected_features(world), key=K.FEATURES.index)
    return world


def make_stadium(straight_m=24.0, radius_m=3.0, width_m=1.8, grip_tier="mid", seed=0, n_slow=1, with_avoid=False):
    """A stadium (two straights of `straight_m`, two half circles of `radius_m`, constant width) for the contact
    scenarios of gate G3; the same assembly and checks as make_testbed. Its avoidance block is empty unless
    `with_avoid`."""
    a, r = 0.5 * straight_m, radius_m
    pts = []
    n_line, n_arc = int(round(straight_m / 0.02)), int(round(math.pi * r / 0.02))
    for k in range(n_line):
        pts.append((-a + straight_m * k / n_line, -r))
    for k in range(n_arc):
        t = -0.5 * math.pi + math.pi * k / n_arc
        pts.append((a + r * math.cos(t), r * math.sin(t)))
    for k in range(n_line):
        pts.append((a - straight_m * k / n_line, r))
    for k in range(n_arc):
        t = 0.5 * math.pi + math.pi * k / n_arc
        pts.append((-a + r * math.cos(t), r * math.sin(t)))
    p = _resample(np.asarray(pts), 0.1)
    w = np.full(len(p), float(width_m))
    rng = np.random.default_rng([int(seed), 0x57AD])
    world = _assemble(rng, seed, 0, p, w, grip_tier, "classic")
    if world is None:
        raise RuntimeError("stadium: the walls do not close")
    world["world_id"] = "stadium_%s_%d" % (grip_tier, seed)
    world = check.round_floats(world)
    if with_avoid:
        world["truth"]["avoidance"] = place_avoidance(world, np.random.default_rng([int(seed), 0xA70]), n_slow=n_slow)
        world = check.round_floats(world)
    world["layout_digest"] = check.world_digest(world)
    return world
