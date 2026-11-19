"""Walls of a venue: wall chains along the duct axis curves, open walls, ducts, gaps, classes.

Wall chains. The duct axis of a side is the centreline moved (half width + 0.165 m) to that side. A side without a
spine is one closed chain. On the spine's side the axis runs from the fork T down the shared row to its free end E,
stays at E around the hairpin (the axis offset equals the hairpin radius) and runs back up the same row: the way
back is the same line and is laid only once. So that side gives two chains: the outer chain from T round the venue
back to T (open, both ends at the fork) and the shared chain T -> E (side `shared`).

Open walls sit on straights of a wall that is not shared, and never where another section of the track lies within
3 m behind the opening (a wheel there would count as beyond the tape).

Ducts. A chain is cut into runs by its open walls; a closed chain without an open wall starts at a random point and
closes with a gap. A run is laid from its start: full ducts (15 pieces of 0.508 m, every piece a chord of the axis
polyline, the pieces chained end to end) with a junction after each (butted with probability `butt_prob`, else an
end gap of U(gap_m)); the last duct of a run is compressed (n pieces of 0.35-0.53 m, solved by bisection) so that
it ends exactly at the run's end; when a full duct would leave less than `last_duct_min_m`, the rest is split into
two ducts of about half each.
"""

import math

import numpy as np

from sketch.examples import check
from sketch.examples import constants as K

from . import geometry as G

R = K.DUCT_RADIUS_M


class WallError(RuntimeError):
    pass


class Chain(object):
    def __init__(self, name, side, pts, stations, closed):
        self.name, self.side, self.closed = name, side, closed
        pts = np.asarray(pts, dtype=float)
        self.stations = np.asarray(stations, dtype=float)
        self.pts = pts
        self.poly = G.Polyline(np.vstack([pts, pts[:1]]) if closed else pts)
        # centreline station of every polyline vertex (the closing vertex repeats the first one)
        st = np.concatenate([self.stations, self.stations[:1]]) if closed else self.stations
        self.vertex_s = st[self.poly.src]

    def u_of_station(self, s, L):
        """Arc position on this chain's polyline of the axis point at centreline station s (None if not on it)."""
        vs = self.vertex_s
        n = len(vs)
        for i in range(n - 1):
            a, b = vs[i], vs[i + 1]
            span = (b - a) % L
            rel = (s - a) % L
            if span > 0 and rel <= span + 1e-12:
                f = rel / span
                return float(self.poly.u[i] + f * (self.poly.u[i + 1] - self.poly.u[i]))
        return None


def chains(lay):
    """The wall chains of a layout: list of Chain (outer / inner order of the sides, then the shared chain)."""
    cl = lay.cl
    out = []
    for side in ("left", "right"):
        A = lay.axis[side]
        if lay.spine is None or lay.spine["side"] != side:
            out.append(Chain("W_" + side[0].upper(), side, A, cl.s, True))
            continue
        sp = lay.spine
        idx_leg = np.flatnonzero(cl.prim == sp["leg_in"])
        idx_tip = np.flatnonzero(cl.prim == sp["tip"][0])
        idx_top_out = np.flatnonzero(cl.prim == sp["top"][1])
        if not len(idx_leg) or not len(idx_tip) or not len(idx_top_out):
            raise WallError("spine samples missing")
        spine_idx = list(idx_leg) + [int(idx_tip[0])]
        out_start, out_end = int(idx_top_out[0]), int(idx_leg[0])
        n = cl.n
        if out_start <= out_end:
            outer_idx = list(range(out_start, out_end + 1))
        else:
            outer_idx = list(range(out_start, n)) + list(range(0, out_end + 1))
        out.append(Chain("W_" + side[0].upper(), side, A[outer_idx], cl.s[outer_idx], False))
        out.append(Chain("W_S", "shared", A[spine_idx], cl.s[spine_idx], False))
    return out


# ------------------------------------------------------------------------------------------------ open walls
def _sdist(a, b, L):
    d = abs(a - b) % L
    return min(d, L - d)


def place_open_walls(rng, lay, chain_list, count, s_avoid, P):
    """`count` open walls: (chain index, centre arc position u, width, station). Raises WallError if no place."""
    Ow = P["open_walls"]
    cl = lay.cl
    L = cl.length
    cand = [i for i, c in enumerate(chain_list) if c.side != "shared"]
    zone = set(lay.spine["prims"]) if lay.spine else set()
    narrow = lay.narrow
    lines = [p for p in lay.straights(2.0) if p[0] not in zone]
    tr = check.Track({"truth": {"centreline": {"samples": lay.samples().tolist()}}})
    out = []
    for _ in range(Ow["tries"]):
        if len(out) >= count:
            break
        ci = cand[int(rng.integers(0, len(cand)))]
        li = int(rng.integers(0, max(1, len(lines)))) if lines else 0
        W = float(rng.uniform(*Ow["width_m"]))
        f = float(rng.random())
        if not lines:
            break
        pi, s0, plen = lines[li]
        pad = 0.5 * W + Ow["straight_pad_m"]
        if plen < 2.0 * pad:
            continue
        s = s0 + pad + f * (plen - 2.0 * pad)
        q = int(np.argmin(np.abs(((cl.s - s) + 0.5 * L) % L - 0.5 * L)))
        tw = float(lay.wl[q] + lay.wr[q])
        if W > tw - Ow["track_margin_m"]:
            continue
        if any(min(abs(s - a) % L, L - abs(s - a) % L) < Ow["line_clear_m"] for a in s_avoid):
            continue
        if any(min(abs(s - o[3]) % L, L - abs(s - o[3]) % L) < Ow["separation_m"] for o in out):
            continue
        if narrow is not None and _sdist(s, narrow["s_m"], L) < 0.5 * narrow["length_m"] + 2.0:
            continue
        ch = chain_list[ci]
        u = ch.u_of_station(s, L)
        if u is None:
            continue
        if _track_behind(lay, tr, ch.side, s, W, Ow["behind_clear_m"]):
            continue
        if not ch.closed and (u - 0.5 * W < Ow["chain_end_clear_m"] or u + 0.5 * W > ch.poly.length - Ow["chain_end_clear_m"]):
            continue
        out.append((ci, u, W, s))
    if len(out) < count:
        raise WallError("could not place %d open walls" % count)
    return out


def _track_behind(lay, tr, side, s, W, depth):
    """True when another part of the track lies behind the opening: a point of the strip beyond the opening (its
    width, up to `depth` metres out from the wall face) inside the corridor of a section more than 1 m of station
    away. A wheel there would count as beyond the tape (the cell tests every tape near the car)."""
    L = lay.cl.length
    sg = 1.0 if side == "left" else -1.0
    for f in np.linspace(-0.5, 0.5, 7):
        ss = s + f * W
        x, y, h = _layout_point(lay, ss)
        q = int(np.argmin(np.abs(((lay.cl.s - ss) + 0.5 * L) % L - 0.5 * L)))
        w = float(lay.wl[q] if side == "left" else lay.wr[q])
        nx, ny = -math.sin(h), math.cos(h)
        for d in np.arange(0.05, depth + 1e-9, 0.1):
            px, py = x + sg * (w + d) * nx, y + sg * (w + d) * ny
            if tr.inside(px, py):
                s2, _ = tr.project(px, py)
                if _sdist(s2, s, tr.length) > 1.0:
                    return True
    return False


def _layout_point(lay, s):
    cl = lay.cl
    s = s % cl.length
    i = int(np.searchsorted(np.asarray(cl.prim_start), s, side="right") - 1)
    while lay.prims[i].length <= 1e-12:
        i -= 1
    x, y, h, _ = lay.prims[i].at(min(s - cl.prim_start[i], lay.prims[i].length))
    return x, y, h


# ------------------------------------------------------------------------------------------------ ducts
def _pieces(poly, u, n, c):
    """n chord pieces of length c from arc position u: (poses, end arc position) or (None, None)."""
    poses = []
    p0 = poly.point(u)
    for _ in range(n):
        u2 = poly.advance(u, c)
        if u2 is None:
            return None, None
        p1 = poly.point(u2)
        poses.append([0.5 * (p0[0] + p1[0]), 0.5 * (p0[1] + p1[1]), math.atan2(p1[1] - p0[1], p1[0] - p0[0])])
        u, p0 = u2, p1
    return poses, u


def _end_after(poly, u, n, c):
    pos, e = _pieces(poly, u, n, c)
    return math.inf if pos is None else e


def _final(poly, u, ub, W):
    """The compressed last duct of a run, ending exactly at ub: (poses, c)."""
    lo_c, hi_c = W["piece_final_m"]
    rem = ub - u
    n = max(1, int(round(rem / W["piece_full_m"])))
    for _ in range(40):
        a, b = 0.2, 0.8
        fa, fb = _end_after(poly, u, n, a) - ub, _end_after(poly, u, n, b) - ub
        if fa > 0:
            n -= 1
            if n < 1:
                raise WallError("run too short for one piece")
            continue
        if fb < 0:
            n += 1
            continue
        for _ in range(60):
            m = 0.5 * (a + b)
            if _end_after(poly, u, n, m) - ub < 0:
                a = m
            else:
                b = m
        c = a                                   # the largest bracket end that still ends inside the run
        if c > hi_c + 1e-12:
            n += 1
            continue
        if c < lo_c - 1e-12:
            if n == 1:
                raise WallError("run too short for one piece")
            n -= 1
            continue
        if n * c > K.DUCT_LENGTH_M + 1e-9:
            raise WallError("compressed duct longer than a duct")
        poses, e = _pieces(poly, u, n, c)
        return poses, c
    raise WallError("no piece length for the last duct")


def _draw_gap(rng, W):
    b = float(rng.random())
    w = float(rng.uniform(*W["gap_m"]))
    return 0.0 if b < W["butt_prob"] else w


def lay_run(rng, poly, ua, ub, W):
    """Ducts of one run: (list of (poses, piece length), list of the gap widths between consecutive ducts)."""
    full_n, full_c = W["pieces_full"], W["piece_full_m"]
    full_len = full_n * full_c
    ducts, gaps = [], []
    u = ua
    while True:
        rem = ub - u
        if rem < W["piece_final_m"][0]:
            raise WallError("run remainder %.3f m below one piece" % rem)
        if rem <= 0.98 * full_len:
            poses, c = _final(poly, u, ub, W)
            ducts.append((poses, c))
            return ducts, gaps
        g = _draw_gap(rng, W)
        if rem - full_len - g >= W["last_duct_min_m"]:
            n1 = full_n
        else:
            # a full duct would leave less than `last_duct_min_m`: the rest becomes two ducts of about half each
            n1 = int(min(full_n, max(1, math.ceil(0.5 * (rem - g) / full_c))))
        poses, u2 = _pieces(poly, u, n1, full_c)
        if poses is None:
            raise WallError("run ended inside a duct")
        ducts.append((poses, full_c))
        gaps.append(g)
        u = poly.advance(u2, g) if g > 0 else u2
        if u is None:
            raise WallError("gap beyond the run")


def lay_walls(rng, lay, chain_list, openings, P):
    """Every chain laid with ducts. Returns dict with walls, ducts, gaps, open walls (open walls without station
    fields; the generator adds s_m and track_width_m on the rounded track)."""
    W = P["walls"]
    walls, ducts, gaps, opens = [], [], [], []
    by_chain = {}
    for k, (ci, u, w, s) in enumerate(openings):
        by_chain.setdefault(ci, []).append((u, w, k))
    dcount = [0]

    def new_duct(wall_id, poses, c):
        did = "D%d" % dcount[0]
        dcount[0] += 1
        ducts.append({"id": did, "wall": wall_id, "wall_class": None, "diameter_m": K.DUCT_DIAMETER_M,
                      "length_m": len(poses) * c, "piece_length_m": c, "pieces": poses})
        return did

    for ci, ch in enumerate(chain_list):
        U = ch.poly.length
        ops = sorted(by_chain.get(ci, []))
        runs, joins = [], []                    # runs: (ua, ub); joins between run k and k+1: ("open", w, k) / None
        if ch.closed:
            u0 = float(rng.random()) * U
            if ops:
                # rotate the loop so that it starts right after the first opening (which then closes the loop)
                u0 = (ops[0][0] + 0.5 * ops[0][1]) % U
            pl = _rotated(ch.poly, u0)
            if ops:
                ops_r = sorted((((u - u0) % U), w, k) for (u, w, k) in ops)
                cur = 0.0
                for j, (c, w, k) in enumerate(ops_r):
                    end = c - 0.5 * w if j < len(ops_r) - 1 else pl.retreat(U, w)
                    runs.append((cur, end))
                    joins.append(("open", w, k))
                    cur = c + 0.5 * w
            else:
                g = _draw_gap(rng, W)
                end = pl.retreat(U, g) if g > 0 else U
                runs.append((0.0, end))
                joins.append(("gap", g, None))                     # closes the loop
        else:
            pl = ch.poly
            cur = 0.0
            for (u, w, k) in ops:
                runs.append((cur, u - 0.5 * w))
                joins.append(("open", w, k))
                cur = u + 0.5 * w
            runs.append((cur, U))
        wall_id = ch.name
        ids = []
        run_first = []
        run_last = []
        for r_i, (ua, ub) in enumerate(runs):
            if r_i > 0 and joins[r_i - 1][0] == "open":
                # the run starts exactly one opening width (chord) after the previous run's end
                ua = pl.advance(runs[r_i - 1][1], joins[r_i - 1][1])
                runs[r_i] = (ua, ub)
            rd, rg = lay_run(rng, pl, ua, ub, W)
            first = None
            for j, (poses, c) in enumerate(rd):
                did = new_duct(wall_id, poses, c)
                if first is None:
                    first = did
                if ids and j > 0:
                    gaps.append(_junction("gap", wall_id, ids[-1], did, rg[j - 1], ducts))
                ids.append(did)
            run_first.append(first)
            run_last.append(ids[-1])
        # junctions between runs (and the closing one)
        for r_i in range(len(runs)):
            nxt = r_i + 1
            if nxt >= len(runs):
                if not ch.closed:
                    break
                nxt = 0
            kind, w, k = joins[r_i]
            a, b = run_last[r_i], run_first[nxt]
            if kind == "open":
                opens.append((k, _junction("open", wall_id, a, b, w, ducts, side=ch.side)))
            else:
                gaps.append(_junction("gap", wall_id, a, b, w, ducts))
        walls.append({"id": wall_id, "side": ch.side, "closed": bool(ch.closed), "ducts": ids})
    opens.sort(key=lambda x: x[0])
    for k, (_, o) in enumerate(opens):
        o["id"] = "OW%d" % k
    for k, g in enumerate(gaps):
        g["id"] = "G%d" % k
    return {"walls": walls, "ducts": ducts, "gaps": gaps, "open_walls": [o for _, o in opens]}


def _rotated(poly, u0):
    """The closed polyline started at arc position u0 (same loop, same direction)."""
    U = poly.length
    p = poly.p
    u = poly.u
    i = int(np.searchsorted(u, u0, side="right") - 1)
    i = min(max(i, 0), len(p) - 2)
    start = poly.point(u0)
    pts = [start] + [tuple(q) for q in p[i + 1:-1]] + [tuple(q) for q in p[:i + 1]] + [start]
    out = G.Polyline(np.asarray(pts))
    if abs(out.length - U) > 1e-6:
        raise WallError("loop rotation changed its length")
    return out


def _ends(duct):
    c = duct["piece_length_m"]
    x0, y0, a0 = duct["pieces"][0]
    x1, y1, a1 = duct["pieces"][-1]
    tail = (x0 - 0.5 * c * math.cos(a0), y0 - 0.5 * c * math.sin(a0))
    head = (x1 + 0.5 * c * math.cos(a1), y1 + 0.5 * c * math.sin(a1))
    return tail, head


def _junction(kind, wall_id, a, b, w, ducts, side=None):
    da = next(d for d in ducts if d["id"] == a)
    db = next(d for d in ducts if d["id"] == b)
    head = _ends(da)[1]
    tail = _ends(db)[0]
    out = {"id": None, "wall": wall_id, "ducts": [a, b], "ends": [list(head), list(tail)],
           "width_m": math.hypot(tail[0] - head[0], tail[1] - head[1])}
    if kind == "open":
        out["_side"] = side
    return out


def finish_open_wall(o, ducts):
    """Tape and outward of an open wall from its two end pieces."""
    da = next(d for d in ducts if d["id"] == o["ducts"][0])
    db = next(d for d in ducts if d["id"] == o["ducts"][1])
    side = o.pop("_side")
    sgn = 1.0 if side == "right" else -1.0             # the track is on the left of a right wall
    ya, yb = da["pieces"][-1][2], db["pieces"][0][2]
    ha, tb = o["ends"]
    ta = [ha[0] - sgn * R * math.sin(ya), ha[1] + sgn * R * math.cos(ya)]
    tb2 = [tb[0] - sgn * R * math.sin(yb), tb[1] + sgn * R * math.cos(yb)]
    hm = math.atan2(tb2[1] - ta[1], tb2[0] - ta[0])
    o["tape"] = [ta, tb2]
    o["outward"] = [sgn * math.sin(hm), -sgn * math.cos(hm)]
    return o


# ------------------------------------------------------------------------------------------------ classes
def assign_classes(rng, walls, ducts, outer_side, cls_params, P):
    W = P["walls"]
    by_id = {d["id"]: d for d in ducts}
    pal = _palette(rng, W["palette_probs"])
    for w in walls:
        cur = "orange_ribbed" if pal != "yellow" else "yellow_ribbed"
        for did in w["ducts"]:
            u = float(rng.random())
            if pal == "mixed" and u < W["mixed_switch_prob"]:
                cur = "yellow_ribbed" if cur == "orange_ribbed" else "orange_ribbed"
            by_id[did]["wall_class"] = cur
    if cls_params["black_ducts"]:
        outer = [d for w in walls if w["side"] == outer_side for d in w["ducts"]]
        n = len(outer)
        frac = float(rng.uniform(*W["black_frac"]))
        n_black = max(1, min(int(math.floor(0.30 * n)), int(round(frac * n))))
        n_runs = int(rng.integers(W["black_runs"][0], W["black_runs"][1] + 1))
        n_runs = max(1, min(n_runs, n_black))
        sizes = [n_black // n_runs + (1 if k < n_black % n_runs else 0) for k in range(n_runs)]
        starts = [int(rng.integers(0, n)) for _ in range(n_runs)]
        black = set()
        for st, sz in zip(starts, sizes):
            k = st
            placed = 0
            guard = 0
            while placed < sz and guard < 2 * n:
                if outer[k % n] not in black:
                    black.add(outer[k % n])
                    placed += 1
                k += 1
                guard += 1
        for did in outer:
            if did in black:
                by_id[did]["wall_class"] = "black_plain"
    return pal


def _palette(rng, probs):
    u = float(rng.random())
    acc = 0.0
    for k, p in probs.items():
        acc += p
        if u < acc:
            return k
    return list(probs)[-1]


def duct_bounds(ducts):
    """(xmin, ymin, xmax, ymax): every piece end point -/+ the duct radius (check.check_world's containment test)."""
    xs, ys = [], []
    for d in ducts:
        c = d["piece_length_m"]
        for x, y, a in d["pieces"]:
            dx, dy = 0.5 * c * math.cos(a), 0.5 * c * math.sin(a)
            xs += [x - dx, x + dx]
            ys += [y - dy, y + dy]
    return min(xs) - R, min(ys) - R, max(xs) + R, max(ys) + R
