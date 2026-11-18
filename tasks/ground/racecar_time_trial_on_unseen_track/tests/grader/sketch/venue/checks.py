"""The measures behind the venue generator's gates: geometry of a world, a kinematic drive around it, statistics.

    geometry_report(world)     closed, self-crossing, widths, radius, footprint (check.Track on the file as written)
    kinematic_lap(world)       a kinematic bicycle car with the owner's envelope, steering lock and servo rate follows
                               the centreline with pure pursuit once round the lap from the start pose: its envelope
                               and wheel points must stay inside the corridor and clear of every duct piece, and it
                               must cross the timing line and every checkpoint in order; no wheel point may be
                               beyond a tape line (within 3 m of the tape) at any time
    stats_row(world)           the numbers of the statistics table (length, widths, footprint, ducts, walls)
"""

import math

import numpy as np

from sketch.examples import check
from sketch.examples import constants as K

from . import geometry as G

R = K.DUCT_RADIUS_M


def geometry_report(world):
    t = world["truth"]
    tr = check.Track(world)
    w = tr.left + tr.right
    p = np.asarray(t["centreline"]["samples"], dtype=float)
    closed = bool(t["centreline"]["closed"]) and float(np.max(tr.seg_len)) <= 0.25 + 1e-9
    return {"closed": closed, "self_intersects": bool(tr.self_intersects()), "min_width_m": float(w.min()),
            "max_width_m": float(w.max()), "min_radius_m": float(tr.min_radius()), "length_m": float(tr.length),
            "n_samples": int(len(p))}


class _Corridor(object):
    """Windowed projection on the centreline (the semantics of check.Track.project near a known station)."""

    def __init__(self, world):
        tr = check.Track(world)
        self.tr = tr
        self.p, self.seg, self.seg_len, self.s = tr.p, tr.seg, tr.seg_len, tr.s
        self.left, self.right, self.n, self.L = tr.left, tr.right, tr.n, tr.length

    def project(self, x, y, hint, win=60):
        idx = (np.arange(hint - win, hint + win + 1)) % self.n
        a = self.p[idx]
        sg = self.seg[idx]
        L2 = self.seg_len[idx] ** 2
        u = np.clip(((x - a[:, 0]) * sg[:, 0] + (y - a[:, 1]) * sg[:, 1]) / L2, 0.0, 1.0)
        fx, fy = a[:, 0] + u * sg[:, 0], a[:, 1] + u * sg[:, 1]
        d = np.hypot(fx - x, fy - y)
        k = int(np.argmin(d))
        i = int(idx[k])
        cross = sg[k, 0] * (y - fy[k]) - sg[k, 1] * (x - fx[k])
        off = float(d[k]) if cross >= 0 else -float(d[k])
        s = (self.s[i] + u[k] * self.seg_len[i]) % self.L
        j = (i + 1) % self.n
        wl = (1 - u[k]) * self.left[i] + u[k] * self.left[j]
        wr = (1 - u[k]) * self.right[i] + u[k] * self.right[j]
        return s, off, wl, wr, i


def kinematic_lap(world, speed=1.0, dt=0.02, lookahead=0.6, wall_clear=0.02):
    """One lap of a kinematic car from the start pose. Returns a dict with ok, the smallest corridor margin of the
    envelope corners and wheel points, the smallest clearance of the envelope to a duct piece, the steering use,
    and the order of the lines crossed."""
    t = world["truth"]
    cor = _Corridor(world)
    tr = cor.tr
    sa, sb = [], []
    for d in t["ducts"]:
        for piece in d["pieces"]:
            a, b = check.piece_ends(piece, d["piece_length_m"])
            sa.append(a)
            sb.append(b)
    sa, sb = np.asarray(sa), np.asarray(sb)
    mid = 0.5 * (sa + sb)
    x, y, yaw = t["start_pose"]["pose"]
    steer = 0.0
    Lw = K.WHEELBASE_M
    s0, _, _, _, hint = cor.project(x, y, int(np.argmin(np.hypot(tr.p[:, 0] - x, tr.p[:, 1] - y))), win=cor.n // 2)
    travelled = 0.0
    margin_min, clear_min, steer_max = math.inf, math.inf, 0.0
    env = [(K.X_FRONT_M, 0.5 * K.ENVELOPE_M[1]), (K.X_FRONT_M, -0.5 * K.ENVELOPE_M[1]),
           (K.X_REAR_M, 0.5 * K.ENVELOPE_M[1]), (K.X_REAR_M, -0.5 * K.ENVELOPE_M[1]),
           (0.5 * (K.X_FRONT_M + K.X_REAR_M), 0.5 * K.ENVELOPE_M[1]), (0.5 * (K.X_FRONT_M + K.X_REAR_M), -0.5 * K.ENVELOPE_M[1])]
    wheels = [(v[0], v[1]) for v in K.WHEEL_GROUND_C.values()]
    lines = [("T", t["timing_line"])] + [(c["id"], c) for c in t["checkpoints"]]
    tapes = [(o["tape"][0], o["tape"][1], o["outward"]) for o in t["open_walls"]]
    tape_max = 0
    crossed = []
    ref_prev = None
    L = tr.length
    steps = int(math.ceil((L + 5.0) / (speed * dt)))         # the start pose is up to 3.2 m before the line
    s_ref = s0
    for _ in range(steps):
        rx, ry = x + K.REF_POINT_C[0] * math.cos(yaw), y + K.REF_POINT_C[0] * math.sin(yaw)
        s_ref, _, _, _, hint = cor.project(rx, ry, hint)
        # pure pursuit from the rear axle to the centreline point `lookahead` ahead of the reference point
        tx, ty, _, _, _ = tr.at(s_ref + lookahead)
        dx, dy = tx - x, ty - y
        alpha = math.atan2(dy, dx) - yaw
        ld = max(0.3, math.hypot(dx, dy))
        target = math.atan2(2.0 * Lw * math.sin(alpha), ld)
        target = max(-K.STEER_MAX_RAD, min(K.STEER_MAX_RAD, target))
        steer += max(-K.STEER_RATE_MAX_RADPS * dt, min(K.STEER_RATE_MAX_RADPS * dt, target - steer))
        steer_max = max(steer_max, abs(steer))
        x += speed * dt * math.cos(yaw)
        y += speed * dt * math.sin(yaw)
        yaw = G.wrap(yaw + speed * dt * math.tan(steer) / Lw)
        travelled += speed * dt
        c, s = math.cos(yaw), math.sin(yaw)
        pts = [(x + c * px - s * py, y + s * px + c * py) for px, py in env + wheels]
        for px, py in pts:
            _, off, wl, wr, _ = cor.project(px, py, hint)
            margin_min = min(margin_min, wl - off, off + wr)
        for a, b, o in tapes:
            tape_max = max(tape_max, sum(1 for px, py in pts[6:] if beyond_tape(px, py, a, b, o)))
        near = np.hypot(mid[:, 0] - x, mid[:, 1] - y) < 1.5
        if np.any(near):
            dd = float(G.points_segs_dist(pts[:6] + [(x, y)], sa[near], sb[near]).min()) - R
            clear_min = min(clear_min, dd)
        ref = (x + K.REF_POINT_C[0] * c, y + K.REF_POINT_C[0] * s)
        if ref_prev is not None:
            for name, ln in lines:
                if _crosses(ref_prev, ref, ln):
                    crossed.append(name)
        ref_prev = ref
    want = ["T"] + [c["id"] for c in t["checkpoints"]] + ["T"]
    order_ok = _contains_in_order(crossed, want)
    ok = margin_min >= 0.0 and clear_min >= wall_clear and order_ok and tape_max == 0
    return {"ok": bool(ok), "corridor_margin_min_m": float(margin_min), "duct_clearance_min_m": float(clear_min),
            "tape_wheels_max": int(tape_max),
            "steer_max_rad": float(steer_max), "lines_in_order": bool(order_ok), "crossed": len(crossed),
            "distance_m": float(travelled)}


def beyond_tape(x, y, a, b, outward, depth=3.0):
    """The contract: on the outward side of the tape and projecting between its ends (here within `depth`)."""
    mx, my = 0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])
    h = (x - mx) * outward[0] + (y - my) * outward[1]
    if h <= 0.0 or h > depth:
        return False
    tx, ty = b[0] - a[0], b[1] - a[1]
    u = ((x - a[0]) * tx + (y - a[1]) * ty) / (tx * tx + ty * ty)
    return 0.0 <= u <= 1.0


def _crosses(p, q, ln):
    a, b = ln["a"], ln["b"]
    fwd = ln["forward"]
    d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    d2 = (b[0] - a[0]) * (q[1] - a[1]) - (b[1] - a[1]) * (q[0] - a[0])
    if (d1 > 0) == (d2 > 0) or d1 == d2:
        return False
    d3 = (q[0] - p[0]) * (a[1] - p[1]) - (q[1] - p[1]) * (a[0] - p[0])
    d4 = (q[0] - p[0]) * (b[1] - p[1]) - (q[1] - p[1]) * (b[0] - p[0])
    if (d3 > 0) == (d4 > 0):
        return False
    return (q[0] - p[0]) * fwd[0] + (q[1] - p[1]) * fwd[1] > 0


def _contains_in_order(seq, want):
    k = 0
    for x in seq:
        if k < len(want) and x == want[k]:
            k += 1
    return k == len(want)


# ------------------------------------------------------------------------------------------------ statistics
def stats_row(world):
    t = world["truth"]
    tr = check.Track(world)
    w = tr.left + tr.right
    wts = tr.seg_len / tr.length
    order = np.argsort(w)
    cw = np.cumsum(wts[order])
    p50 = float(w[order][min(np.searchsorted(cw, 0.5), len(w) - 1)])
    full = sum(1 for d in t["ducts"]
               if len(d["pieces"]) == K.DUCT_PIECES_FULL and abs(d["piece_length_m"] - K.PIECE_LENGTH_NOMINAL_M) < 1e-6)
    wall_len = sum(d["length_m"] for d in t["ducts"])
    shared_ids = {wl["id"] for wl in t["walls"] if wl["side"] == "shared"}
    shared_len = sum(d["length_m"] for d in t["ducts"] if d["wall"] in shared_ids)
    black = sum(1 for d in t["ducts"] if d["wall_class"] == "black_plain")
    outer_side = "right" if t["centreline"]["direction"] == "ccw" else "left"
    outer = sum(1 for d in t["ducts"] if any(wl["id"] == d["wall"] and wl["side"] == outer_side for wl in t["walls"]))
    pos_gaps = [g["width_m"] for g in t["gaps"] if g["width_m"] > 1e-6]
    behind = 0
    for o in t["open_walls"]:
        m = np.mean(np.asarray(o["tape"], dtype=float), axis=0)
        if any(math.dist(m, c["pose"][:2]) < 3.5 for c in t["clutter"]):
            behind += 1
    return {"world_id": world["world_id"], "class": t["cell"]["layout_class"], "tier": t["cell"]["grip_tier"],
            "length_m": tr.length, "footprint_x_m": t["stats"]["footprint_m"][0], "footprint_y_m": t["stats"]["footprint_m"][1],
            "width_min_m": float(w.min()), "width_p50_m": p50, "width_max_m": float(w.max()),
            "width_ge_1p5_frac": t["stats"]["width_ge_1p5_frac"], "min_radius_m": t["stats"]["min_radius_m"],
            "n_ducts": len(t["ducts"]), "n_full_ducts": full, "n_pieces": t["stats"]["n_pieces"],
            "wall_length_m": wall_len, "wall_per_track_m": wall_len / tr.length, "ducts_per_track_m": len(t["ducts"]) / tr.length,
            "shared_wall_m": shared_len, "n_gaps_open": len(pos_gaps), "n_junctions": len(t["gaps"]) + len(t["open_walls"]),
            "gap_mean_m": float(np.mean(pos_gaps)) if pos_gaps else 0.0, "n_open_walls": len(t["open_walls"]),
            "open_wall_m": [o["width_m"] for o in t["open_walls"]], "black_frac_outer": (black / outer) if outer else 0.0,
            "n_patches": len(t["floor"]["patches"]), "patch_kinds": [p["kind"] for p in t["floor"]["patches"]],
            "base_grip": t["floor"]["base_grip"], "patch_grip": [p["grip"] for p in t["floor"]["patches"]],
            "n_clutter": len(t["clutter"]), "open_walls_with_clutter_behind": behind, "n_checkpoints": len(t["checkpoints"]),
            "n_slow_cars": len(t["avoidance"]["slow_cars"]),
            "features": list(world["venue_sheet"]["features"]), "direction": t["centreline"]["direction"]}


# the real numbers the statistics are compared with (the 2026 competitions' published maps and the IROS 2026 tube layout script)
REAL = {
    "ICRA 2026 Classic map (B0 2.2)": {"footprint_m": [19.0, 18.0], "length_m": "about 100 (estimate in the competition rules)",
                                       "width_m": "at least 1.0, most of it at least 1.5"},
    "ICRA 2026 Master map (B0 2.2)": {"footprint_m": [38.0, 18.0], "length_m": "about 240 (estimate)"},
    "IROS 2026 design (B0 2.2, R2 3.2)": {"length_m": 210.0, "width_p50_m": 2.24, "track_area_max_m": [43.0, 23.0]},
    "IROS 2026 tube layout (tube_layout.png title)": {"tubes": 60, "wall_m": 432.0, "saved_by_sharing_m": 68.0,
                                                       "wall_per_track_m": round(432.0 / 210.0, 3),
                                                       "tubes_per_track_m": round(60 / 210.0, 3),
                                                       "shared_frac_of_both_walls": round(68.0 / 500.0, 3)},
    "DESIGN 4.1 starting values": {"classic": {"length_m": [60, 80], "footprint_m": [20, 20]},
                                   "featured": {"length_m": [90, 110], "footprint_m": [30, 20]},
                                   "width_m": [1.0, 2.2], "width_ge_1p5_frac_min": 0.70, "min_radius_m": 0.9,
                                   "duct_m": 7.62, "gap_m": [0.0, 0.15], "open_wall_m": [0.6, 1.2]},
}


def summarise(rows):
    """Per class: min / median / max of the main numbers."""
    out = {}
    keys = ["length_m", "footprint_x_m", "footprint_y_m", "width_min_m", "width_p50_m", "width_max_m",
            "width_ge_1p5_frac", "min_radius_m", "n_ducts", "n_full_ducts", "n_pieces", "wall_per_track_m",
            "ducts_per_track_m", "shared_wall_m", "n_gaps_open", "gap_mean_m", "n_open_walls", "black_frac_outer",
            "n_patches", "n_clutter", "n_checkpoints", "n_slow_cars"]
    for cls in sorted({r["class"] for r in rows}):
        rr = [r for r in rows if r["class"] == cls]
        out[cls] = {"n": len(rr)}
        for k in keys:
            v = np.asarray([float(r[k]) for r in rr])
            out[cls][k] = {"min": round(float(v.min()), 3), "p50": round(float(np.median(v)), 3),
                           "max": round(float(v.max()), 3), "mean": round(float(v.mean()), 3)}
    return out
