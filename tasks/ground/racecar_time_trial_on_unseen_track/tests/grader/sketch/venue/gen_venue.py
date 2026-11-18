"""The seeded venue generator of racecar_time_trial_on_unseen_track.

    from sketch.venue import generate
    world = generate(seed_hex, grip_tier, layout_class)       # a world file (dict) that passes check.check_world

    python -B -m sketch.venue.gen_venue --seed 9f3a... --tier mid --cls featured --out DIR [--png] [--mjcf]

Pipeline (one attempt; an attempt that fails any rule is dropped and the next one is drawn from the same seed, so a
seed always gives the same world):

  1. shape     skeleton polygon: class rectangle, corner edits, notches (Featured: one spine notch); mirrored, the
               direction drawn; fillets at every vertex (layout.make_polygon, transform_polygon, fillet_radii)
  2. widths    half widths with the bend, separation, spine and narrow-section rules; validated (layout.Layout)
  3. lines     timing line on a straight of at least 6 m, start pose 0.5-2.0 m (front) before it, checkpoints about
               every 7 m in driving order
  4. walls     wall chains along the duct axis curves (the shared row laid once), open walls on straights, ducts of
               7.62 m with end gaps, compressed last ducts, wall classes (walls.*)
  5. frame     the polygon is turned by 90 degrees when the footprint's long side is along y, then moved so that the
               footprint (outer faces of the outer ducts) is centred on the origin; steps 2-4 are rebuilt each time
               with the same random streams, so the result is a rigid motion of the first build
  6. floor     base grip in the tier, floor kind, 1-3 patches (one on the line of a bend), Featured a carpet tile
  7. clutter   pillars, crates, banners, tables, bins at least 0.55 m from every duct, behind every open wall and
               around and inside the track
  8. file      every float rounded to 6 decimals; stations, lines, stats recomputed on the rounded centreline
               (check.Track); the obstacle model's placement rule adds the avoidance set (default_rng([seed words, 5, k]));
               seeds, venue sheet, layout digest; check.check_world must return no error
Random streams: numpy default_rng(seed words + [stage, attempt]); nothing else is random.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys

import numpy as np

from sketch.examples import check
from sketch.examples import constants as K

from . import geometry as G
from . import layout as LY
from . import walls as WL

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS_PATH = os.path.join(HERE, "venue_params.json")
R = K.DUCT_RADIUS_M
STAGES = {"shape": 1, "transform": 2, "radii": 3, "widths": 4, "avoid": 5, "floor": 6, "clutter": 7, "lines": 8,
          "openings": 9, "walls": 10, "classes": 11, "sheet": 12}


class GenerationError(RuntimeError):
    pass


def load_params(path=PARAMS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def params_digest(P):
    return hashlib.sha256(check.canonical_json(P).encode("ascii")).hexdigest()


def seed_words(seed_hex):
    if not isinstance(seed_hex, str) or not 16 <= len(seed_hex) <= 64 or any(c not in "0123456789abcdef" for c in seed_hex):
        raise ValueError("seed_hex must be 16-64 lower-case hex digits")
    h = seed_hex + "0" * ((-len(seed_hex)) % 8)
    return [int(h[i:i + 8], 16) for i in range(0, len(h), 8)] + [len(seed_hex)]


def stream(words, stage, attempt, *more):
    return np.random.default_rng(list(words) + [STAGES[stage], int(attempt)] + [int(m) for m in more])


def world_id(seed_hex, grip_tier, layout_class):
    return "s3%s_%s_%s" % (layout_class[0], grip_tier, seed_hex[:12])


def _hash_int(seed_hex, name, bits=52):
    return int(hashlib.sha256(("%s:%s" % (seed_hex, name)).encode("ascii")).hexdigest(), 16) % (1 << bits)


# ------------------------------------------------------------------------------------------------ lines
def _point(lay, s, d=0.0):
    """(x, y, heading) of the centreline at station s of the layout's own parametrisation, moved d to the left."""
    cl = lay.cl
    s = s % cl.length
    i = int(np.searchsorted(np.asarray(cl.prim_start), s, side="right") - 1)
    while lay.prims[i].length <= 1e-12:
        i -= 1
    t = s - cl.prim_start[i]
    x, y, h, _ = lay.prims[i].at(min(t, lay.prims[i].length))
    return x - d * math.sin(h), y + d * math.cos(h), h


def place_lines(rng, lay, P):
    Lp = P["lines"]
    cl = lay.cl
    L = cl.length
    zone = set(lay.spine["prims"]) if lay.spine else set()
    narrow = lay.narrow["prim"] if lay.narrow else None
    cands = [p for p in lay.straights(Lp["timing_straight_min_m"]) if p[0] not in zone and p[0] != narrow]
    u = float(rng.random())
    f = float(rng.uniform(*Lp["timing_frac"]))
    g = float(rng.uniform(*Lp["start_front_gap_m"]))
    dl = float(rng.uniform(-Lp["start_lateral_m"], Lp["start_lateral_m"]))
    if not cands:
        raise LY.LayoutError("no straight for the timing line")
    pi, s0, plen = cands[min(int(u * len(cands)), len(cands) - 1)]
    s_t = s0 + f * plen
    s_c = s_t - g - K.X_FRONT_M
    if s_c + K.X_REAR_M < s0 + 0.05:
        s_c = s0 + 0.05 - K.X_REAR_M
        g = s_t - s_c - K.X_FRONT_M
        if g < 0.3:
            raise LY.LayoutError("timing straight too short for the start pose")
    n_cp = max(Lp["checkpoints_min"], int(round(L / Lp["checkpoint_spacing_m"])))
    jit = [float(rng.uniform(-Lp["checkpoint_jitter"], Lp["checkpoint_jitter"])) for _ in range(n_cp)]
    s_cp = [s_t + L * (k + 1 + jit[k]) / (n_cp + 1) for k in range(n_cp)]
    return {"s_t": s_t % L, "s_start": s_c % L, "d_start": dl, "s_cp": [s % L for s in s_cp], "timing_prim": pi}


# ------------------------------------------------------------------------------------------------ one build
def build_core(words, attempt, poly, direction, P, cls):
    """Steps 2-4 for one polygon (fresh random streams each call): layout, lines, chains, walls, bounds."""
    radii = LY.fillet_radii(stream(words, "radii", attempt), poly, P["shape"])
    lay = LY.Layout(poly, radii, direction, P, cls, stream(words, "widths", attempt))
    lines = place_lines(stream(words, "lines", attempt), lay, P)
    chain_list = WL.chains(lay)
    avoid_s = [lines["s_t"], lines["s_start"]]
    opens = WL.place_open_walls(stream(words, "openings", attempt), lay, chain_list, P["classes"][cls]["open_walls"],
                                avoid_s, P)
    laid = WL.lay_walls(stream(words, "walls", attempt), lay, chain_list, opens, P)
    bounds = WL.duct_bounds(laid["ducts"])
    return {"radii": radii, "lay": lay, "lines": lines, "chains": chain_list, "openings": opens, "laid": laid,
            "bounds": bounds}


def framed_core(words, attempt, P, cls):
    poly, info = LY.make_polygon(stream(words, "shape", attempt), P, cls)
    poly, direction = LY.transform_polygon(stream(words, "transform", attempt), poly, cls)
    core = build_core(words, attempt, poly, direction, P, cls)
    x0, y0, x1, y1 = core["bounds"]
    if (y1 - y0) > (x1 - x0):
        poly = LY.rotate_polygon(poly)
        core = build_core(words, attempt, poly, direction, P, cls)
        x0, y0, x1, y1 = core["bounds"]
    poly = LY.translate_polygon(poly, -0.5 * (x0 + x1), -0.5 * (y0 + y1))
    core = build_core(words, attempt, poly, direction, P, cls)
    core["poly"], core["direction"], core["shape_info"] = poly, direction, info
    return core


# ------------------------------------------------------------------------------------------------ floor
def _grip(rng, kind, base, tier, F):
    a = float(rng.uniform(*F["dust_factor"]))
    b = float(rng.uniform(*F["metal_factor"]))
    c = float(rng.uniform(*F["carpet_on_hard_floor"]))
    d = float(rng.uniform(*F["carpet_on_carpet_drop"]))
    if kind == "dust":
        g = base * a
    elif kind == "metal_cover":
        g = base * b
    elif tier == "high":
        g = base - d
    else:
        g = max(c, base + F["carpet_above_hard_min"])
    return round(max(F["patch_grip_min"], g), 3)


def _corridor_track(lay):
    w = {"truth": {"centreline": {"samples": lay.samples().tolist()}}}
    return check.Track(w)


def _station_dist(a, b, L):
    d = abs(a - b) % L
    return min(d, L - d)


def place_floor(rng, lay, lines, P, cls, tier):
    F, C = P["floor"], P["classes"][cls]
    lo, hi = F["tiers"][tier]
    base = round(float(rng.uniform(lo, hi)), 3)
    kinds = F["kinds"][tier]
    kind = kinds[int(rng.integers(0, len(kinds)))]
    n = int(rng.integers(C["patch_count"][0], C["patch_count"][1] + 1))
    pk = [C["patch_kinds"][int(rng.integers(0, len(C["patch_kinds"])))] for _ in range(n)]
    if C["carpet_tile"]:
        pk[int(rng.integers(0, n))] = "carpet_tile"
    cl = lay.cl
    L = cl.length
    tr = _corridor_track(lay)
    avoid = [lines["s_t"], lines["s_start"]]
    bends = [(i, s0, p) for i, s0, p in lay.bends()
             if abs(p.sweep) >= F["bend_min_turn_rad"] and p.r <= F["bend_max_radius_m"]
             and all(_station_dist(s0 + 0.5 * p.length, a, L) > F["start_clear_m"] + 0.5 * p.length for a in avoid)]
    if not bends:
        raise LY.LayoutError("no bend for the patch on the racing line")
    patches = []
    used_s = []
    for j in range(n):
        u = float(rng.random())
        jit = float(rng.uniform(-0.2, 0.2))
        dr = [float(x) for x in rng.random(16)]
        grip = _grip(rng, pk[j], base, tier, F)
        if j == 0:
            i, s0, p = bends[min(int(u * len(bends)), len(bends) - 1)]
            s_c = s0 + (0.5 + jit) * p.length
            on_line = True
        else:
            ok = False
            for t in range(40):
                s_c = float(L * ((u + 0.618034 * t) % 1.0))
                if all(_station_dist(s_c, a, L) > F["start_clear_m"] for a in avoid) and \
                        all(_station_dist(s_c, q, L) > F["patch_separation_m"] for q in used_s):
                    ok = True
                    break
            if not ok:
                raise LY.LayoutError("no room for floor patch %d" % j)
            q = int(np.argmin(np.abs(((cl.s - s_c) + 0.5 * L) % L - 0.5 * L)))
            pr = lay.prims[int(cl.prim[q])]
            on_line = pr.kind == "arc" and abs(pr.sweep) >= 0.5
        poly, edge = _patch_polygon(lay, tr, pk[j], s_c, dr, F)
        if poly is None:
            raise LY.LayoutError("patch %d does not fit" % j)
        if check.polygon_area(poly) <= 0:
            poly = poly[::-1]
        used_s.append(s_c)
        patches.append({"id": "P%d" % j, "kind": pk[j], "polygon": [list(v) for v in poly], "grip": grip,
                        "edge_height_m": edge, "on_racing_line": bool(on_line), "s_m": s_c})
    return {"kind": kind, "base_grip": base, "patches": patches}


def _patch_polygon(lay, tr, kind, s_c, dr, F):
    cl = lay.cl
    L = cl.length
    q = int(np.argmin(np.abs(((cl.s - s_c) + 0.5 * L) % L - 0.5 * L)))
    wl, wr = float(lay.wl[q]), float(lay.wr[q])
    kq = float(cl.k[q])
    W = wl + wr
    if kind == "dust":
        a = F["dust_along_m"][0] + dr[0] * (F["dust_along_m"][1] - F["dust_along_m"][0])
        fb = F["dust_across_frac"][0] + dr[1] * (F["dust_across_frac"][1] - F["dust_across_frac"][0])
        b = 0.5 * fb * W
        inner = 1.0 if kq > 0 else (-1.0 if kq < 0 else 0.0)
        w_in = wl if kq > 0 else wr
        d_c = 0.5 * (wl - wr) + inner * 0.25 * w_in
        lim = 0.8 / abs(kq) if abs(kq) > 1e-9 else math.inf
        pts = []
        nv = int(F["dust_vertices"])
        for k in range(nv):
            ph = 2.0 * math.pi * k / nv
            rn = 1.0 + F["dust_noise"] * (2.0 * dr[2 + k % 14] - 1.0)
            ss = s_c + a * rn * math.cos(ph)
            dd = d_c + b * rn * math.sin(ph)
            dd = max(-lim, min(lim, dd))
            x, y, _ = _point(lay, ss, dd)
            pts.append((x, y))
        return pts, 0.0
    x, y, h = _point(lay, s_c, 0.0)
    if kind == "metal_cover":
        along = F["metal_along_m"][0] + dr[0] * (F["metal_along_m"][1] - F["metal_along_m"][0])
        across = W + 2.0 * F["metal_overhang_m"]
        dm = 0.5 * (wl - wr)
        cx, cy = x - dm * math.sin(h), y + dm * math.cos(h)
        return G.rect_corners(cx, cy, h, along, across), 0.0
    along = F["carpet_along_m"][0] + dr[0] * (F["carpet_along_m"][1] - F["carpet_along_m"][0])
    across = F["carpet_across_m"][0] + dr[1] * (F["carpet_across_m"][1] - F["carpet_across_m"][0])
    edge = round(F["carpet_edge_m"][0] + dr[2] * (F["carpet_edge_m"][1] - F["carpet_edge_m"][0]), 3)
    m = F["carpet_wall_margin_m"]
    for _ in range(8):
        across = min(across, W - 2.0 * m - 0.02)
        lo_d, hi_d = -wr + 0.5 * across + m, wl - 0.5 * across - m
        if hi_d >= lo_d:
            d_c = lo_d + dr[3] * (hi_d - lo_d)
            cx, cy = x - d_c * math.sin(h), y + d_c * math.cos(h)
            pts = G.rect_corners(cx, cy, h, along, across)
            probe = pts + [((pts[i][0] + pts[(i + 1) % 4][0]) * 0.5, (pts[i][1] + pts[(i + 1) % 4][1]) * 0.5) for i in range(4)]
            if all(tr.inside(px, py, margin=m) for px, py in probe):
                return pts, edge
        along *= 0.85
        across *= 0.9
    return None, 0.0


# ------------------------------------------------------------------------------------------------ clutter
def _segments(ducts):
    a, b = [], []
    for d in ducts:
        c = d["piece_length_m"]
        for x, y, yaw in d["pieces"]:
            dx, dy = 0.5 * c * math.cos(yaw), 0.5 * c * math.sin(yaw)
            a.append((x - dx, y - dy))
            b.append((x + dx, y + dy))
    return np.asarray(a), np.asarray(b)


def _outline(cx, cy, yaw, lx, ly, step=0.25):
    corners = G.rect_corners(cx, cy, yaw, lx, ly)
    pts = [(cx, cy)]
    for i in range(4):
        p, q = corners[i], corners[(i + 1) % 4]
        m = max(1, int(math.ceil(math.dist(p, q) / step)))
        for k in range(m):
            pts.append((p[0] + (q[0] - p[0]) * k / m, p[1] + (q[1] - p[1]) * k / m))
    return pts, corners


def place_clutter(rng, lay, laid, opens, P, outer_side):
    Cp = P["clutter"]
    tr = _corridor_track(lay)
    sa, sb = _segments(laid["ducts"])
    cl = lay.cl
    L = cl.length
    kinds = Cp["kinds"]
    names = list(kinds)
    probs = [kinds[k]["prob"] for k in names]
    n_items = int(rng.integers(Cp["count"][0], Cp["count"][1] + 1))
    placed, entries = [], []

    def fits(shape_pts):
        if any(tr.inside(px, py) for px, py in shape_pts):
            return False
        dmin = float(G.points_segs_dist(shape_pts, sa, sb).min()) - R
        if dmin < Cp["wall_clear_m"]:
            return False
        for other in placed:
            if float(np.min(np.hypot(np.asarray(shape_pts)[:, None, 0] - np.asarray(other)[None, :, 0],
                                     np.asarray(shape_pts)[:, None, 1] - np.asarray(other)[None, :, 1]))) < Cp["item_gap_m"]:
                return False
        return True

    def draw_item(rng):
        u = float(rng.random())
        acc, kind = 0.0, names[-1]
        for k, p in zip(names, probs):
            acc += p
            if u < acc:
                kind = k
                break
        spec = kinds[kind]
        vals = [float(v) for v in rng.random(6)]
        dark = vals[5] < spec["dark_prob"]
        return kind, spec, vals, dark

    def make(kind, spec, vals, dark, cx, cy, yaw):
        """(entries without ids, outline points)"""
        if kind == "table_leg":
            (tx0, tx1), (ty0, ty1) = spec["table_m"]
            lx, ly = tx0 + vals[0] * (tx1 - tx0), ty0 + vals[1] * (ty1 - ty0)
            dia = spec["leg_m"][0] + vals[2] * (spec["leg_m"][1] - spec["leg_m"][0])
            pts, corners = _outline(cx, cy, yaw, lx, ly)
            legs = G.rect_corners(cx, cy, yaw, lx - 0.1, ly - 0.1)
            ents = [{"kind": "table_leg", "shape": "cylinder", "pose": [px, py, 0.0], "size_m": [dia, dia, spec["height_m"]],
                     "reflectivity": "dark" if dark else "normal"} for px, py in legs]
            return ents, pts
        if kind == "other":
            (d0, d1), (h0, h1) = spec["size_m"]
            dia, hh = d0 + vals[0] * (d1 - d0), h0 + vals[1] * (h1 - h0)
            pts, _ = _outline(cx, cy, 0.0, dia, dia)
            return [{"kind": "other", "shape": "cylinder", "pose": [cx, cy, 0.0], "size_m": [dia, dia, hh],
                     "reflectivity": "dark" if dark else "normal"}], pts
        (a0, a1), (b0, b1), (c0, c1) = spec["size_m"]
        lx, ly, lz = a0 + vals[0] * (a1 - a0), b0 + vals[1] * (b1 - b0), c0 + vals[2] * (c1 - c0)
        if kind == "pillar":
            yaw, ly = 0.0, lx
        pts, _ = _outline(cx, cy, yaw, lx, ly)
        return [{"kind": kind, "shape": "box", "pose": [cx, cy, G.wrap(yaw)], "size_m": [lx, ly, lz],
                 "reflectivity": "dark" if dark else "normal"}], pts

    # 1. behind every open wall: something the scan finds through the opening
    for o in opens:
        for t in range(Cp["tries"] // 10):
            kind, spec, vals, dark = draw_item(rng)
            if kind == "pillar":
                kind, spec = "crate", kinds["crate"]
            dist = Cp["behind_open_wall_m"][0] + vals[3] * (Cp["behind_open_wall_m"][1] - Cp["behind_open_wall_m"][0])
            lat = (vals[4] - 0.5) * 1.0
            (ax, ay), (bx, by) = o["tape"]
            mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
            tx, ty = G.unit((bx - ax, by - ay))
            ox, oy = o["outward"]
            cx, cy = mx + dist * ox + lat * tx, my + dist * oy + lat * ty
            ents, pts = make(kind, spec, vals, dark, cx, cy, math.atan2(ty, tx))
            if fits(pts):
                placed.append(pts)
                entries += ents
                break
    # 2. around the track and in the infield
    tries = 0
    while len(placed) < n_items + len(opens) and tries < Cp["tries"]:
        tries += 1
        kind, spec, vals, dark = draw_item(rng)
        s = float(rng.random()) * L
        infield = float(rng.random()) < Cp["infield_prob"]
        q = int(np.argmin(np.abs(((cl.s - s) + 0.5 * L) % L - 0.5 * L)))
        side = outer_side if not infield else ("left" if outer_side == "right" else "right")
        rng_m = Cp["infield_m"] if infield else Cp["outside_m"]
        off = (lay.wl[q] if side == "left" else lay.wr[q]) + 2.0 * R + rng_m[0] + vals[3] * (rng_m[1] - rng_m[0])
        sg = 1.0 if side == "left" else -1.0
        cx, cy = cl.x[q] + sg * off * cl.nx[q], cl.y[q] + sg * off * cl.ny[q]
        yaw = float(cl.h[q]) + (vals[4] - 0.5) * 0.6
        ents, pts = make(kind, spec, vals, dark, float(cx), float(cy), yaw)
        if fits(pts):
            placed.append(pts)
            entries += ents
    for k, e in enumerate(entries):
        e_id = {"id": "C%d" % k}
        e_id.update(e)
        entries[k] = e_id
    return entries


# ------------------------------------------------------------------------------------------------ the file
def _dist_station(a, b, L):
    d = abs(a - b) % L
    return min(d, L - d)


def assemble(seed_hex, grip_tier, layout_class, core, floor, clutter, P):
    lay, lines, laid = core["lay"], core["lines"], core["laid"]
    words = seed_words(seed_hex)
    t = {
        "cell": {"grip_tier": grip_tier, "layout_class": layout_class},
        "bounds": {},
        "stats": {},
        "centreline": {"closed": True, "direction": lay.direction, "length_m": 0.0, "samples": lay.samples().tolist()},
        "walls": laid["walls"],
        "ducts": laid["ducts"],
        "gaps": laid["gaps"],
        "open_walls": [WL.finish_open_wall(o, laid["ducts"]) for o in laid["open_walls"]],
        "floor": floor,
        "clutter": clutter,
        "timing_line": {},
        "checkpoints": [],
        "start_pose": {},
        "avoidance": {"boxes": [], "slow_cars": []},
        "seeds": {name: _hash_int(seed_hex, name) for name in ("relocation", "putback", "sensors", "others")},
    }
    world = {"schema": K.WORLD_SCHEMA, "world_id": world_id(seed_hex, grip_tier, layout_class),
             "generator": {"name": P["generator"]["name"], "version": P["generator"]["version"], "seed_hex": seed_hex},
             "layout_digest": "", "venue_sheet": {}, "truth": t}
    world = check.round_floats(world)
    t = world["truth"]
    tr = check.Track(world)
    L = tr.length
    t["centreline"]["length_m"] = L

    def line_at(s_layout):
        x, y, _ = _point(lay, s_layout, 0.0)
        s, _ = tr.project(x, y)
        h = tr.at(s)[2]
        return {"s_m": s, "a": list(tr.boundary(s, "left")), "b": list(tr.boundary(s, "right")),
                "forward": [math.cos(h), math.sin(h)]}

    t["timing_line"] = line_at(lines["s_t"])
    t["checkpoints"] = [dict(id="CP%d" % (k + 1), **line_at(s)) for k, s in enumerate(lines["s_cp"])]
    x, y, h = _point(lay, lines["s_start"], lines["d_start"])
    s, _ = tr.project(x, y)
    t["start_pose"] = {"pose": [x, y, G.wrap(h)], "s_m": s}
    for o in t["open_walls"]:
        (ax, ay), (bx, by) = o["tape"]
        s, _ = tr.project(0.5 * (ax + bx), 0.5 * (ay + by))
        _, _, _, wl, wr = tr.at(s)
        o["s_m"], o["track_width_m"] = s, wl + wr
    for p in t["floor"]["patches"]:
        v = np.asarray(p["polygon"], dtype=float)
        p["s_m"] = tr.project(float(np.mean(v[:, 0])), float(np.mean(v[:, 1])))[0]
    # bounds from the rounded pieces, outward to 6 decimals
    xs, ys = [], []
    for d in t["ducts"]:
        for k, piece in enumerate(d["pieces"]):
            for px, py in check.piece_ends(piece, d["piece_length_m"]):
                xs.append(px)
                ys.append(py)
    xmin, xmax = min(xs) - R - 2e-6, max(xs) + R + 2e-6
    ymin, ymax = min(ys) - R - 2e-6, max(ys) + R + 2e-6
    t["bounds"] = {"xmin": math.floor(xmin * 1e6) / 1e6, "ymin": math.floor(ymin * 1e6) / 1e6,
                   "xmax": math.ceil(xmax * 1e6) / 1e6, "ymax": math.ceil(ymax * 1e6) / 1e6}
    b = t["bounds"]
    width = tr.left + tr.right
    t["stats"] = {"length_m": L, "min_width_m": float(width.min()),
                  "width_ge_1p5_frac": float(tr.seg_len[width >= 1.5 - 1e-9].sum() / L),
                  "min_radius_m": tr.min_radius(), "footprint_m": [b["xmax"] - b["xmin"], b["ymax"] - b["ymin"]],
                  "n_ducts": len(t["ducts"]), "n_pieces": sum(len(d["pieces"]) for d in t["ducts"])}
    world = check.round_floats(world)
    world["venue_sheet"] = venue_sheet(stream(words, "sheet", 0), world, P)
    return world


def venue_sheet(rng, world, P):
    t = world["truth"]
    F = P["floor"]
    seed_hex = world["generator"]["seed_hex"]
    f = float(rng.uniform(*F["side_pull_factor"]))
    m = round(float(rng.uniform(*F["side_pull_mass_kg"])), 1)
    mu = t["floor"]["base_grip"] * f
    force = round(2.0 * mu * m * 9.81) / 2.0
    h = hashlib.sha256(("%s:label" % seed_hex).encode("ascii")).hexdigest()
    letters = "ABCDEFGHJKLMNPRSTUVWXYZ"
    label = "Hall %s%d, track %s" % (letters[int(h[0:2], 16) % len(letters)], int(h[2:4], 16) % 9 + 1,
                                     letters[int(h[4:6], 16) % len(letters)])
    fp = t["stats"]["footprint_m"]
    feats = check.expected_features(world)
    return {"venue_label": label,
            "floor": {"kind": t["floor"]["kind"], "side_pull": {"peak_force_n": force, "test_mass_kg": m}},
            "walls": {"kinds": list(K.WALL_CLASSES), "height_m": K.DUCT_DIAMETER_M},
            "track": {"min_width_m": max(1.0, math.floor(t["stats"]["min_width_m"] * 10.0 + 1e-9) / 10.0),
                      "max_footprint_m": [float(math.ceil(fp[0] + 0.2)), float(math.ceil(fp[1] + 0.2))]},
            "features": [x for x in K.FEATURES if x in feats]}


def _with_avoidance(world, words, attempt):
    from sketch.others import PlacementError, place_avoidance
    last = None
    for k in range(4):
        try:
            av = place_avoidance(world, np.random.default_rng(list(words) + [STAGES["avoid"], int(attempt), k]))
            world["truth"]["avoidance"] = av
            return world
        except PlacementError as e:
            last = e
    raise GenerationError("avoidance set: %s" % last)


# ------------------------------------------------------------------------------------------------ entry point
def generate(seed_hex, grip_tier, layout_class, params=None, report=None, with_avoidance=True):
    """A world file (dict) for one candidate seed. `report` (a list) receives one line per dropped
    attempt."""
    P = params or load_params()
    if layout_class not in ("classic", "featured"):
        raise ValueError("layout_class is classic or featured")
    if grip_tier not in P["floor"]["tiers"]:
        raise ValueError("grip_tier is low, mid or high")
    words = seed_words(seed_hex)
    C = P["classes"][layout_class]
    for attempt in range(P["shape"]["max_attempts"]):
        try:
            core = framed_core(words, attempt, P, layout_class)
            lay = core["lay"]
            outer = "right" if lay.direction == "ccw" else "left"
            WL.assign_classes(stream(words, "classes", attempt), core["laid"]["walls"], core["laid"]["ducts"], outer, C, P)
            floor = place_floor(stream(words, "floor", attempt), lay, core["lines"], P, layout_class, grip_tier)
            opens_tmp = [WL.finish_open_wall(copy.deepcopy(o), core["laid"]["ducts"]) for o in core["laid"]["open_walls"]]
            clutter = place_clutter(stream(words, "clutter", attempt), lay, core["laid"], opens_tmp, P, outer)
            world = assemble(seed_hex, grip_tier, layout_class, core, floor, clutter, P)
            if with_avoidance:
                world = _with_avoidance(world, words, attempt)
            world = check.round_floats(world)
            world["layout_digest"] = check.world_digest(world)
            if with_avoidance:
                err = check.check_world(world)
                if err:
                    raise GenerationError("check_world: " + "; ".join(err[:3]))
            return world
        except (LY.LayoutError, WL.WallError, GenerationError) as e:
            if report is not None:
                report.append("attempt %d: %s: %s" % (attempt, type(e).__name__, e))
            continue
    raise GenerationError("seed %s: no valid venue in %d attempts" % (seed_hex, P["shape"]["max_attempts"]))


def generate_with_info(seed_hex, grip_tier, layout_class, params=None):
    """(world, info): info has the attempts dropped before the accepted one and their reasons."""
    rep = []
    w = generate(seed_hex, grip_tier, layout_class, params=params, report=rep)
    return w, {"dropped": len(rep), "reasons": rep}


# ------------------------------------------------------------------------------------------------ writing
def dumps(world):
    """The bytes of a world file: one key per line, arrays of plain numbers (and short rows of them) on one line."""
    def scalar_row(v):
        return isinstance(v, list) and all(not isinstance(x, (list, dict)) for x in v)

    def enc(v, ind):
        pad = " " * ind
        if isinstance(v, dict):
            if not v:
                return "{}"
            if all(not isinstance(x, (list, dict)) for x in v.values()) and len(json.dumps(v)) <= 110:
                return json.dumps(v, ensure_ascii=True)
            body = ",\n".join("%s %s: %s" % (pad, json.dumps(k), enc(x, ind + 1)) for k, x in v.items())
            return "{\n" + body + "\n" + pad + "}"
        if isinstance(v, list):
            if scalar_row(v) or (all(scalar_row(x) for x in v) and len(json.dumps(v)) <= 110):
                return json.dumps(v, ensure_ascii=True)
            return "[\n" + ",\n".join("%s %s" % (pad, enc(x, ind + 1)) for x in v) + "\n" + pad + "]"
        return json.dumps(v, ensure_ascii=True, allow_nan=False)

    return enc(world, 0) + "\n"


def write_world(world, path):
    text = dumps(world)
    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write(text)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="generate one venue (S3)")
    ap.add_argument("--seed", required=True, help="16-64 lower-case hex digits")
    ap.add_argument("--tier", required=True, choices=("low", "mid", "high"))
    ap.add_argument("--cls", required=True, choices=("classic", "featured"))
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--png", action="store_true", help="also the greybox top view")
    ap.add_argument("--mjcf", action="store_true", help="also the MJCF and the other exports")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    w, info = generate_with_info(a.seed, a.tier, a.cls)
    if a.mjcf or a.png:
        from .export import export_all
        paths = export_all(w, a.out, png=a.png, mjcf=a.mjcf)
    else:
        paths = {"world": write_world(w, os.path.join(a.out, w["world_id"] + ".json"))}
    print(json.dumps({"world_id": w["world_id"], "layout_digest": w["layout_digest"], "dropped_attempts": info["dropped"],
                      "files": paths}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
