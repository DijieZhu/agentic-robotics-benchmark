"""Site family W3 of version 1.1 (W3+C in the documents): seed -> the three counterfactual sites and the site.

W3 = W2's open ground (bands A, R1, B drawn exactly by W2's rules) + one whole slab over the north part of the site +
three look-alike covered P mouths in the front row (one leads in, two are plugged a short way inside) + a deepened,
structured void behind them + rubble boxes in its straight runs:

    true mouth -> fore-gallery G1 (east-west, with a dead-end pocket beyond the mouth) -> connector N (north) ->
    [back corridor K (east-west, optional second pocket)] -> chamber C in a back corner -> search point 3

Random streams (each a numpy default_rng):
  [seed, 10]       visible stage: bands, R1, the front row (three mouth footprints), search point 3, pad and points
                   1-2, ground patches, then (after the layout is kept) camera, lighting, friction, free blocks. No
                   draw and no rejection of it depends on which mouth is the true one.
  [seed, 11, tau]  hidden stage, tau = 0, 1, 2, at most 400 tries each: the void that connects mouth tau to the
                   chamber. A visible stage is kept only when all three mouths admit a void; else the visible stage
                   is redrawn from its own stream.
  [seed, 12]       the true mouth tau* = integers(0, 3).
  [seed, 2]        colour chips (W2's rule; route pad -> 1 -> 2 of the tau = 0 counterfactual, never in the void).
  [seed, 13]       rubble (module rubble, rule A), a fresh stream for each of the three counterfactual voids.
Free blocks are placed against the union of the solids of the three counterfactual sites and dropped (from the
last) while G9 fails on ANY of the three (judged before the rubble), so whether a block stays does not depend on
tau*. The rubble is the last step of the assembly: every counterfactual site gets the boxes the rule gives ITS void.
Everything a camera ray or mission() can see is fixed before stream 13 is read.

Selection-level rules that read the counterfactual sites (G9 and G17-G19 judged on all three) live in predicates.py.
"""
import math
import os

import numpy as np

from w2gen import envelope as envmod
from w2gen import family as w2
from w2gen import grid as gridmod
from w2gen.family import (BL, BOX_MU, CAMERA_MODEL, CAP_RGB, CRATE_RGB, FLOOR_RGB, LEAN_RISE, PATCH_MU, PATCH_RGB,  # noqa: F401
                          PIXEL_CONVENTION, R1_ROOF_T, ROOF_OVERHANG, SLAB_T, SUP, TONE, WALL_H, WALL_T, XL_CLEAR,
                          _box, _clip_split, _lean_slab, _rect_gap, _rect_overlap, _row, _split_long, _subtract,
                          cam_axes, corner_margin, draw_camera, draw_lighting, place_chips, principalpixel, project, r6,
                          spacing_ok)

from . import GENERATOR_VERSION
from . import rubble as rubblemod

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_DIR = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
ACCEPTANCE = os.path.join(DEV_DIR[:-len("-dev")], "tests", "voidscore", "acceptance.yaml")


def load_acceptance(path=ACCEPTANCE):
    """Weights and credit radii from the task's acceptance file (the single source of the scoring numbers)."""
    import yaml
    with open(path, "rb") as fh:
        acc = yaml.safe_load(fh.read().decode("utf-8"))
    weights = {int(k): float(v) for k, v in acc["weights"].items()}
    full, zero = float(acc["credit_radius_m"]["full"]), float(acc["credit_radius_m"]["zero"])
    return weights, (full, zero)


FAMILY = "W3"
LX_DEFAULT = 4.4
LY_CAM = 4.0                   # W2's site depth: the camera rule and the open-ground bands use it
LY = 5.8                       # the site rectangle's depth (mission().site.size_m)
SP3_CLEAR = 0.62               # = zero credit radius + max D_STOP + 0.01, rounded up to 0.01 (checked in classes())
D1_MIN = 0.90
POCKET = (0.45, 0.90)
WP = (0.30, 0.70)
ROUTE_WIN = (4.0, 6.0)
WEIGHTS, CREDIT_RADIUS = load_acceptance()
HIDDEN_TRIES = 400
MAX_ATTEMPTS = 2000
STEP = 0.02


def sp3_clear_rule(env):
    """Point 3's clearance: the zero credit radius + the largest stop distance + 0.01, rounded up to 0.01 m, so that
    no pose against any solid or site wall earns any credit on point 3."""
    return math.ceil(round((CREDIT_RADIUS[1] + max(env["values"]["D_STOP"]) + 0.01) * 100.0, 6)) / 100.0


def classes(env):
    c = w2.classes(env)
    v = env["values"]
    if abs(sp3_clear_rule(env) - SP3_CLEAR) > 1e-9:
        raise ValueError("point 3 clearance rule gives %.2f m, the family is built for %.2f m"
                         % (sp3_clear_rule(env), SP3_CLEAR))
    c["V"] = dict(w_range=[r6(v["W_SWEEP"] + 0.36), r6(v["W_SWEEP"] + 0.50)], clear_h=c["gallery"]["clear_h"],
                  rule="straight runs under the slab: w = W_SWEEP + 2 U(0.18, 0.25), clear height H_ROLL + 0.14")
    c["mouth"] = dict(w=c["P"]["w"], clear_h=c["gallery"]["clear_h"],
                      rule="three front-row mouths: P width W_SWEEP + 0.14, clear height = slab underside H_ROLL + 0.14")
    return c


# ------------------------------------------------------------------------------------------------ visible stage
def _draw_visible(rng, env, cls, lx):
    U = lambda lo, hi: float(rng.uniform(lo, hi))
    v = env["values"]
    a, t1, Lf = U(0.92, 1.05), U(0.50, 0.60), U(0.50, 0.60)
    gstar = v["W_SWEEP"] + 2 * U(0.18, 0.25)
    b = LY_CAM - a - t1 - Lf - gstar
    if not 0.88 <= b <= 1.10:
        return None, "band B depth outside [0.88, 1.10]: %.3f" % b
    y1b = a + t1
    y2 = y1b + b
    yg = y2 + Lf
    P_w, P_h, T_w, XN_w = cls["P"]["w"], cls["P"]["clear_h"], cls["T"]["w"], cls["XN"]["w"]
    hz = cls["gallery"]["clear_h"]
    boxes = []
    # ---------------- R1: W2's rule, byte for byte in its draws
    third = ("gap_T", "gap_XN", "slot_XL")[int(rng.integers(0, 3))]
    order = ["gap_P", third, "tunnel_P"] if rng.random() < 0.5 else ["gap_P", "tunnel_P", third]
    if rng.random() < 0.5:
        order = order[::-1]
    fw = dict(gap_P=P_w, tunnel_P=P_w + 2 * SUP, gap_T=T_w, gap_XN=XN_w, slot_XL=P_w + 2 * SUP)
    r = _row(rng, [fw[t] for t in order], [0.45] * 2, lx)
    if r is None:
        return None, "R1 does not fit"
    feet1, solids1, slack1 = r
    roof_kind = "lean" if rng.random() < 0.5 else "flat"
    r1 = []
    for k, ((f0, f1), t) in enumerate(zip(feet1, order)):
        covered = t in ("tunnel_P", "slot_XL")
        x0, x1 = (f0 + SUP, f1 - SUP) if covered else (f0, f1)
        gid = "R1_%d" % k
        op = dict(id=gid, row="R1", type=t, cls={"gap_P": "P", "tunnel_P": "P", "gap_T": "T", "gap_XN": "XN", "slot_XL": "XL"}[t],
                  x0=r6(x0), x1=r6(x1), y0=r6(a), y1=r6(y1b), foot=[r6(f0), r6(f1)], covered=covered,
                  depth=r6(t1), role="opening")
        if t == "tunnel_P":
            h = P_h
            rise = LEAN_RISE if roof_kind == "lean" else 0.0
            boxes.append(_box("sup_w_" + gid, "support", f0, x0, a, y1b, 0, h))
            boxes.append(_box("sup_e_" + gid, "support", x1, f1, a, y1b, 0, h + rise))
            if rise:
                boxes.append(_lean_slab("roof_" + gid, f0, f1, a - ROOF_OVERHANG, y1b + ROOF_OVERHANG, h, h + rise, R1_ROOF_T))
                clear_min = h + SUP * rise / (f1 - f0)
            else:
                boxes.append(_box("roof_" + gid, "roof", f0, f1, a - ROOF_OVERHANG, y1b + ROOF_OVERHANG, h, h + R1_ROOF_T))
                clear_min = h
            op.update(roof=roof_kind, clear_h=r6(h), clear_h_min=r6(clear_min), roof_t=R1_ROOF_T)
        elif t == "slot_XL":
            boxes.append(_box("sup_w_" + gid, "support", f0, x0, a, y1b, 0, XL_CLEAR))
            boxes.append(_box("sup_e_" + gid, "support", x1, f1, a, y1b, 0, XL_CLEAR))
            boxes.append(_box("roof_" + gid, "roof", f0, f1, a - ROOF_OVERHANG, y1b + ROOF_OVERHANG, XL_CLEAR, XL_CLEAR + R1_ROOF_T))
            op.update(roof="flat", clear_h=XL_CLEAR, clear_h_min=XL_CLEAR, roof_t=R1_ROOF_T)
        r1.append(op)
    for k, seg in enumerate(solids1):
        for j, (s0, s1) in enumerate(_split_long(*seg)):
            boxes.append(_box("R1_blk_%d_%d" % (k, j), "rubble", s0, s1, a, y1b, 0, U(0.50, 0.88)))
    # ---------------- front row: three look-alike P mouths, solid segments (all under the slab: support, height hz)
    foot = P_w + 2 * SUP
    r = _row(rng, [foot] * 3, [0.0, 0.0], lx)
    if r is None:
        return None, "front row does not fit"
    feet2, solids2, slack2 = r
    mouths = []
    for k, (f0, f1) in enumerate(feet2):
        mid = "m%d" % (k + 1)
        boxes.append(_box("sup_w_" + mid, "support", f0, f0 + SUP, y2, yg, 0, hz))
        boxes.append(_box("sup_e_" + mid, "support", f1 - SUP, f1, y2, yg, 0, hz))
        mouths.append(dict(id=mid, x0=f0 + SUP, x1=f1 - SUP, foot=(f0, f1)))
    for k, (s0, s1) in enumerate(solids2):
        if s1 - s0 > 1e-6:
            boxes.append(_box("R2_fill_%d" % k, "support", s0, s1, y2, yg, 0, hz))
    # ---------------- search point 3 in a back corner (0.64-0.72 m from the back wall and the corner's side wall)
    corner = "W" if rng.random() < 0.5 else "E"
    dx = SP3_CLEAR + U(0.02, 0.10)
    x3 = dx if corner == "W" else lx - dx
    y3 = LY - SP3_CLEAR - U(0.02, 0.10)
    # ---------------- pad, search points 1 and 2 (W2's rule: they depend on R1 only)
    gp = [q for q in r1 if q["type"] == "gap_P"][0]
    tp = [q for q in r1 if q["type"] == "tunnel_P"][0]
    gpx, tpx = 0.5 * (gp["x0"] + gp["x1"]), 0.5 * (tp["x0"] + tp["x1"])
    pad = (float(np.clip(tpx + U(-0.5, 0.5), 0.55, lx - 0.55)), U(0.40, 0.50))
    sp1x = float(np.clip(gpx + (0.9 if gpx < lx / 2 else -0.9) + U(-0.2, 0.2), 0.5, lx - 0.5))
    if abs(sp1x - pad[0]) < 1.5:
        sp1x = float(np.clip(pad[0] + (1.7 if pad[0] < lx / 2 else -1.7), 0.5, lx - 0.5))
    sp1 = (sp1x, U(0.45, a - 0.40))
    sp2x = float(np.clip(tpx + (0.75 if gpx < tpx else -0.75) + U(-0.15, 0.15), 0.5, lx - 0.5))
    sp2 = (sp2x, y1b + b / 2 + U(-0.08, 0.08))
    # ---------------- ground patches: dust as W2, gravel in front of all three mouths
    gx0 = max(0.0, mouths[0]["x0"] - 0.15)
    gx1 = min(lx, mouths[2]["x1"] + 0.15)
    dx0 = float(np.clip(0.5 * (pad[0] + sp1[0]) - 0.45, 0.05, lx - 0.95))
    patches = [dict(name="dust", x0=r6(dx0), x1=r6(dx0 + 0.90), y0=0.25, y1=0.85, mu=PATCH_MU["dust"], thickness=0.002,
                    rgb=list(PATCH_RGB["dust"])),
               dict(name="gravel", x0=r6(gx0), x1=r6(gx1), y0=r6(y2 - 0.60), y1=r6(y2), mu=PATCH_MU["gravel"],
                    thickness=0.002, rgb=list(PATCH_RGB["gravel"]))]
    slab = dict(x0=0.0, x1=r6(lx), y0=r6(y2 - ROOF_OVERHANG), y1=r6(LY), z0=r6(hz), z1=r6(hz + SLAB_T))
    boxes.append(_box("void_slab", "roof", 0.0, lx, y2 - ROOF_OVERHANG, LY, hz, hz + SLAB_T))
    vis = dict(a=a, t1=t1, Lf=Lf, gstar=gstar, b=b, y1b=y1b, y2=y2, yg=yg, hz=hz, slack1=slack1, slack2=slack2,
               order=order, third=third, roof_kind=roof_kind, r1=r1, mouths=mouths, corner=corner, sp3=(x3, y3),
               pad=pad, sp1=sp1, sp2=sp2, patches=patches, slab=slab, boxes=boxes)
    return vis, None


# ------------------------------------------------------------------------------------------------ hidden stage
def _clearance_at(rects_solid, p, lx, ly):
    d = min(p[0], lx - p[0], p[1], ly - p[1])
    for x0, x1, y0, y1 in rects_solid:
        dx = max(x0 - p[0], 0.0, p[0] - x1)
        dy = max(y0 - p[1], 0.0, p[1] - y1)
        d = min(d, math.hypot(dx, dy))
    return d


def _fill_boxes(free, x_lo, x_hi, y_lo, y_hi, prefix, hz):
    """Support boxes (to the slab underside) filling [x_lo, x_hi] x [y_lo, y_hi] minus the free rectangles: the cells
    of the breakpoint grid, merged along x per row, then rows of identical x spans merged along y."""
    xs = sorted({x_lo, x_hi} | {min(max(r[0], x_lo), x_hi) for r in free} | {min(max(r[1], x_lo), x_hi) for r in free})
    ys = sorted({y_lo, y_hi} | {min(max(r[2], y_lo), y_hi) for r in free} | {min(max(r[3], y_lo), y_hi) for r in free})
    xs = [x for i, x in enumerate(xs) if i == 0 or x - xs[i - 1] > 1e-7]
    ys = [y for i, y in enumerate(ys) if i == 0 or y - ys[i - 1] > 1e-7]
    rows = []
    for j in range(len(ys) - 1):
        yc = 0.5 * (ys[j] + ys[j + 1])
        spans, cur = [], None
        for i in range(len(xs) - 1):
            xc = 0.5 * (xs[i] + xs[i + 1])
            solid = not any(r[0] - 1e-9 <= xc <= r[1] + 1e-9 and r[2] - 1e-9 <= yc <= r[3] + 1e-9 for r in free)
            if solid:
                cur = [xs[i], xs[i + 1]] if cur is None else [cur[0], xs[i + 1]]
            elif cur is not None:
                spans.append(tuple(cur))
                cur = None
        if cur is not None:
            spans.append(tuple(cur))
        rows.append((ys[j], ys[j + 1], spans))
    rects = []
    open_ = {}
    for y0, y1, spans in rows:
        nxt = {}
        for s in spans:
            if s in open_:
                nxt[s] = (open_[s][0], y1)
            else:
                nxt[s] = (y0, y1)
        for s, (a, b) in open_.items():
            if s not in nxt:
                rects.append((s[0], s[1], a, b))
        open_ = nxt
    for s, (a, b) in open_.items():
        rects.append((s[0], s[1], a, b))
    rects.sort(key=lambda r: (r[2], r[0]))
    return [_box("%s_%d" % (prefix, k), "support", x0, x1, y0, y1, 0.0, hz) for k, (x0, x1, y0, y1) in enumerate(rects)
            if x1 - x0 > 1e-6 and y1 - y0 > 1e-6]


def _route(solid_rects, lx, ly, start, goal, r_plan, y_min):
    """Grid shortest path (0.02 m, disc radius r_plan, exact clearance to the solid footprints and the site edges) from
    start to goal over y >= y_min; (length, path points) or (None, None)."""
    xs = np.arange(STEP / 2, lx, STEP)
    ys = np.arange(y_min + STEP / 2, ly, STEP)
    X, Y = np.meshgrid(xs, ys)
    d = np.minimum.reduce([X, lx - X, ly - Y])
    for x0, x1, y0, y1 in solid_rects:
        if y1 < y_min - 0.5:
            continue
        ddx = np.maximum(np.maximum(x0 - X, 0.0), X - x1)
        ddy = np.maximum(np.maximum(y0 - Y, 0.0), Y - y1)
        d = np.minimum(d, np.hypot(ddx, ddy))
    ok = d >= r_plan

    class _G:
        pass
    g = gridmod.Grid.__new__(gridmod.Grid)
    g.step, g.lx, g.ly, g.xs, g.ys, g.X, g.Y = STEP, lx, ly, xs, ys, X, Y
    legs = g.legs(ok, {"s": start, "g": goal}, ("s", "g"))
    if legs[0]["length_m"] is None:
        return None, None, d
    return float(legs[0]["length_m"]), legs[0]["path"], d


def _draw_hidden(seed, vis, tau, cls, env, lx):
    rng = np.random.default_rng([int(seed), 11, int(tau)])
    U = lambda lo, hi: float(rng.uniform(lo, hi))
    v = env["values"]
    vlo, vhi = v["W_SWEEP"] + 0.36, v["W_SWEEP"] + 0.50
    r_plan = cls["plan_radius"]
    y2, yg, hz, Lf = vis["y2"], vis["yg"], vis["hz"], vis["Lf"]
    M = vis["mouths"][tau]
    m0, m1 = M["x0"], M["x1"]
    xM = 0.5 * (m0 + m1)
    x3, y3 = vis["sp3"]
    why = {}

    def no(k):
        why[k] = why.get(k, 0) + 1

    corner_sign = -1.0 if vis["corner"] == "W" else 1.0
    first = corner_sign if rng.random() < 0.5 else -corner_sign
    for t in range(1, HIDDEN_TRIES + 1):
        want = first if t <= HIDDEN_TRIES // 2 else -first
        g1, wN, wK = U(vlo, vhi), U(vlo, vhi), U(vlo, vhi)
        ex, ey = U(0.02, 0.35), U(0.02, 0.25)
        if vis["corner"] == "W":
            cx0, cx1 = 0.0, x3 + SP3_CLEAR + ex
        else:
            cx0, cx1 = x3 - SP3_CLEAR - ex, lx
        yc0 = y3 - SP3_CLEAR - ey
        wp = yc0 - (yg + g1)
        if not WP[0] <= wp <= WP[1]:
            no("partition thickness")
            continue
        lo, hi = (xM + D1_MIN, lx - wN / 2) if want > 0 else (wN / 2, xM - D1_MIN)
        if hi <= lo:
            no("no room for the connector on that side")
            continue
        xN = U(lo, hi)
        nx0, nx1 = xN - wN / 2, xN + wN / 2
        inside = nx0 >= cx0 - 1e-9 and nx1 <= cx1 + 1e-9
        outside = nx1 <= cx0 + 1e-9 or nx0 >= cx1 - 1e-9
        if not (inside or outside):
            no("connector straddles the chamber wall")
            continue
        lp = U(*POCKET)
        if xN > xM:
            gx0, gx1 = m0 - lp, nx1
        else:
            gx0, gx1 = nx0, m1 + lp
        if gx0 < 0.0 or gx1 > lx:
            gx0, gx1 = max(gx0, 0.0), min(gx1, lx)
            lp = (m0 - gx0) if xN > xM else (gx1 - m1)
            if lp < POCKET[0]:
                no("pocket cut by the site wall")
                continue
        free = [("mouth", m0, m1, y2, yg), ("G1", gx0, gx1, yg, yg + g1), ("N", nx0, nx1, yg + g1, yc0)]
        pockets = [dict(name="P1", run="G1", rect=[r6(m0 - lp), r6(m0), r6(yg), r6(yg + g1)] if xN > xM
                        else [r6(m1), r6(m1 + lp), r6(yg), r6(yg + g1)], length=r6(lp))]
        shape = "L"
        pocket2 = 0.0
        K = None
        if outside:
            if rng.random() < 0.5:
                pocket2 = U(*POCKET)
            if nx0 >= cx1:
                kx0, kx1 = cx1, min(lx, nx1 + pocket2)
                pocket2 = kx1 - nx1
            else:
                kx0, kx1 = max(0.0, nx0 - pocket2), cx0
                pocket2 = nx0 - kx0
            if 0.0 < pocket2 < POCKET[0]:
                no("second pocket cut by the site wall")
                continue
            K = (kx0, kx1, yc0, yc0 + wK)
            free.append(("K",) + K)
            if pocket2 > 0:
                pockets.append(dict(name="P2", run="K", rect=[r6(nx1), r6(kx1), r6(yc0), r6(yc0 + wK)] if nx0 >= cx1
                                    else [r6(kx0), r6(nx0), r6(yc0), r6(yc0 + wK)], length=r6(pocket2)))
            sense_g1 = 1.0 if xN > xM else -1.0
            sense_k = 1.0 if (0.5 * (cx0 + cx1) > xN) else -1.0
            shape = "Z" if sense_g1 == sense_k else "U"
        free.append(("C", cx0, cx1, yc0, LY))
        plug_p = {}
        for k, mo in enumerate(vis["mouths"]):
            if k != tau:
                plug_p[mo["id"]] = U(0.25, Lf - 0.20)
        # boxes of this void: fill of the band [yg, LY], plugs of the two other mouths
        fr = [(q[1], q[2], q[3], q[4]) for q in free if q[0] != "mouth"]
        hboxes = _fill_boxes(fr, 0.0, lx, yg, LY, "void_fill", hz)
        for mid, p in plug_p.items():
            mo = [q for q in vis["mouths"] if q["id"] == mid][0]
            hboxes.append(_box(mid, "plug", mo["x0"], mo["x1"], y2 + p, yg, 0.0, hz))
        solids = [tuple(b["aabb"][:4]) for b in vis["boxes"] if b["kind"] in gridmod.SOLID_KINDS] + \
                 [tuple(b["aabb"][:4]) for b in hboxes]
        start = (xM, y2 + 0.10)
        blind, path, _d = _route(solids, lx, LY, start, (x3, y3), r_plan, y2 - 0.30)
        if blind is None:
            no("no route")
            continue
        if not ROUTE_WIN[0] <= blind <= ROUTE_WIN[1]:
            no("route %s" % ("short" if blind < ROUTE_WIN[0] else "long"))
            continue
        clr = _clearance_at(solids, (x3, y3), lx, LY)
        if clr < SP3_CLEAR - 1e-9:
            no("point 3 clearance")
            continue
        return dict(ok=True, tau=int(tau), tries=t, rejections=why, g1=g1, wN=wN, wK=wK if K else None, wp=wp, xN=xN,
                    d1=abs(xN - xM), shape=shape, g1_dir=("toward" if (xN - xM) * corner_sign > 0 else "away"),
                    chamber=(cx0, cx1, yc0, LY), free=free, K=K, pockets=pockets, plug_p=plug_p, boxes=hboxes,
                    blind_m=blind, route=path, sp3_clear=clr, G1=(gx0, gx1, yg, yg + g1), N=(nx0, nx1, yg + g1, yc0))
    return dict(ok=False, tau=int(tau), rejections=why)


# ------------------------------------------------------------------------------------------------ assembly
def _assemble(seed, vis, hid, tau, env, cls, lx, attempts, reasons):
    hz = vis["hz"]
    mouths = []
    for k, mo in enumerate(vis["mouths"]):
        true = k == tau
        p = None if true else hid["plug_p"][mo["id"]]
        mouths.append(dict(id=mo["id"], row="R2", type="mouth_true" if true else "mouth_blind", cls="P" if true else "B",
                           role="true mouth" if true else "blind mouth", x0=r6(mo["x0"]), x1=r6(mo["x1"]),
                           y0=r6(vis["y2"]), y1=r6(vis["yg"]), foot=[r6(mo["foot"][0]), r6(mo["foot"][1])],
                           covered=True, clear_h=r6(hz), clear_h_min=r6(hz), roof="void_slab", roof_t=SLAB_T,
                           depth=r6(vis["Lf"] if true else p), **({} if true else {"plug_depth_m": r6(p)})))
    x3, y3 = vis["sp3"]
    order = np.argsort([abs(0.5 * (m["x0"] + m["x1"]) - x3) for m in vis["mouths"]])
    near_rank = int(np.where(order == tau)[0][0])
    rect = lambda q: dict(x0=r6(q[0]), x1=r6(q[1]), y0=r6(q[2]), y1=r6(q[3]))
    runs = {"G1": dict(rect(hid["G1"]), width=r6(hid["g1"]), axis="x"),
            "N": dict(rect(hid["N"]), width=r6(hid["wN"]), axis="y")}
    if hid["K"] is not None:
        runs["K"] = dict(rect(hid["K"]), width=r6(hid["wK"]), axis="x")
    plugs = [dict(mouth=mid, x0=r6([m for m in vis["mouths"] if m["id"] == mid][0]["x0"]),
                  x1=r6([m for m in vis["mouths"] if m["id"] == mid][0]["x1"]), y0=r6(vis["y2"] + p), y1=r6(vis["yg"]),
                  z1=r6(hz), depth_m=r6(p)) for mid, p in sorted(hid["plug_p"].items())]
    void = dict(true_mouth=vis["mouths"][tau]["id"], blind_mouths=[m["id"] for k, m in enumerate(vis["mouths"]) if k != tau],
                runs=runs, chamber=dict(rect(hid["chamber"])), pockets=hid["pockets"], plugs=plugs,
                slab=dict(vis["slab"]), gallery=dict(rect(hid["G1"]), clear_h=r6(hz), note="= run G1 (legacy key)"),
                shape=hid["shape"], g1_dir=hid["g1_dir"], d1_m=r6(hid["d1"]), partition_m=r6(hid["wp"]),
                blind_route_m=r6(hid["blind_m"]), blind_route_start=[r6(0.5 * (vis["mouths"][tau]["x0"] + vis["mouths"][tau]["x1"])), r6(vis["y2"] + 0.10)],
                sp3_clearance_m=r6(hid["sp3_clear"]), hidden_tries=hid["tries"],
                nearest_mouth_is_true=bool(near_rank == 0))
    stratum = dict(corner=vis["corner"], true_index=int(tau), near_rank=near_rank, shape=hid["shape"],
                   g1_dir=hid["g1_dir"], rule="corner from the visible stream; true mouth default_rng([seed, 12]); "
                                               "near_rank = rank of the true mouth by |x - x3| (0 = nearest)")
    site = dict(family=FAMILY, generator=GENERATOR_VERSION, seed=int(seed), draw_attempts=attempts,
                draw_rejections=dict(sorted(reasons.items())), stratum=stratum,
                frame="x east, y north, z up, origin = south-west floor corner; metres",
                size_m=[r6(lx), r6(LY)], camera_depth_m=LY_CAM, body_length_m=BL, envelope=env, classes=cls,
                params=dict(a=r6(vis["a"]), t1=r6(vis["t1"]), Lf=r6(vis["Lf"]), g_band=r6(vis["gstar"]), b=r6(vis["b"]),
                            y2=r6(vis["y2"]), slack_R1_m=r6(vis["slack1"]), slack_R2_m=r6(vis["slack2"])),
                bands=dict(A=[0.0, r6(vis["a"])], R1=[r6(vis["a"]), r6(vis["y1b"])], B=[r6(vis["y1b"]), r6(vis["y2"])],
                           R2_front=[r6(vis["y2"]), r6(vis["yg"])], under_slab=[r6(vis["yg"]), r6(LY)]),
                r1=dict(order=vis["order"], third=vis["third"], tunnel_roof=vis["roof_kind"], openings=vis["r1"]),
                r2=dict(order=[m["id"] for m in vis["mouths"]], decoy_type=None, openings=mouths),
                void=void,
                pad=dict(xy=[r6(vis["pad"][0]), r6(vis["pad"][1])], retrieval_radius_m=0.60, setdown_error_m=0.10),
                search_points=[dict(id=1, xy=[r6(vis["sp1"][0]), r6(vis["sp1"][1])], weight=WEIGHTS[1], where="band A, open floor"),
                               dict(id=2, xy=[r6(vis["sp2"][0]), r6(vis["sp2"][1])], weight=WEIGHTS[2], where="band B, open floor"),
                               dict(id=3, xy=[r6(x3), r6(y3)], weight=WEIGHTS[3], where="chamber in the %s back corner" % vis["corner"])],
                patches=[dict(p) for p in vis["patches"]],
                boxes=[dict(b) for b in vis["boxes"]] + [dict(b) for b in hid["boxes"]])
    site["walls"] = []
    for nm, (x0, x1, y0, y1) in dict(s=(-WALL_T, lx + WALL_T, -WALL_T, 0.0), n=(-WALL_T, lx + WALL_T, LY, LY + WALL_T),
                                     w=(-WALL_T, 0.0, 0.0, LY), e=(lx, lx + WALL_T, 0.0, LY)).items():
        site["boxes"].insert(len(site["walls"]), _box(nm, "wall", x0, x1, y0, y1, 0.0, WALL_H))
        site["walls"].append("wall__" + nm)
    return site


def make_layouts(seed, env=None, lx=LX_DEFAULT):
    env = env or envmod.load()
    cls = classes(env)
    rng = np.random.default_rng([int(seed), 10])
    reasons = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        vis, why = _draw_visible(rng, env, cls, lx)
        if vis is None:
            key = why.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
            continue
        hid = [_draw_hidden(seed, vis, tau, cls, env, lx) for tau in range(3)]
        if all(h["ok"] for h in hid):
            break
        reasons["a mouth admits no void"] = reasons.get("a mouth admits no void", 0) + 1
    else:
        raise RuntimeError("seed %d: no valid draw in %d attempts" % (seed, MAX_ATTEMPTS))
    return env, cls, rng, vis, hid, attempt, reasons


def add_rubble(seed, site, vis, hid, cls):
    """Last step of the assembly: the rubble boxes rule A gives the void of this site's mouth (kind rubble, from the
    floor to the slab underside) and the hidden record void.rubble."""
    hz = vis["hz"]
    P = cls["P"]["w"]
    placed, rec = rubblemod.place(seed, vis, hid, P)
    boxes = [_box(name, "rubble", bx[0], bx[1], bx[2], bx[3], 0.0, hz) for name, bx in placed]
    site["boxes"] += boxes
    for b, r in zip(boxes, rec["boxes"]):
        r["name"] = b["name"]
        r["aabb"] = list(b["aabb"])
        for k in ("length_m", "across_m", "gap_m", "run_width_m"):
            r[k] = r6(r[k])
    rec["note"] = ("rubble in the straight runs under the slab: 1-2 boxes per run, each against one wall of its run, "
                   "from the floor to the slab underside, leaving a P-wide gap beside it; not in the chamber, a pocket "
                   "or a junction square; >= 0.5 m from the mouth's inner end and every corner point; >= 0.8 m free "
                   "between two boxes of a run; hidden stream, placed after everything visible")
    site["void"]["rubble"] = rec
    return boxes


def build_sites(seed, env=None, lx=LX_DEFAULT, with_world_checks=True, rubble=True):
    """dict(site = the site of tau*, sites = the three counterfactual sites (index = mouth), xmls, g9s, info).

    Camera, lighting, friction, free blocks and chips come from the visible streams after the layout is kept and are
    identical in the three counterfactual sites; each counterfactual site then gets the rubble of its own void.
    rubble=False builds the sites without the last step (regression against the probe only; not a version 1.1
    site, and the generator string says so)."""
    from . import mjcf
    from w2gen import predicates as w2p
    env, cls, rng, vis, hid, attempts, reasons = make_layouts(seed, env, lx)
    tau_star = int(np.random.default_rng([int(seed), 12]).integers(0, 3))
    sites = [_assemble(seed, vis, hid[t], t, env, cls, lx, attempts, reasons) for t in range(3)]
    camera = draw_camera(rng, lx, LY_CAM)
    lighting = draw_lighting(rng)
    physics = dict(floor_mu=r6(rng.uniform(0.76, 0.92)), box_mu=BOX_MU, patch_mu=dict(PATCH_MU),
                   floor_mu_rule="U(0.76, 0.92) per site (authors' sysID range)")
    for s in sites:
        s["camera"], s["lighting"], s["physics"] = camera, lighting, physics
        s["camera_rule_note"] = "W2's camera rule with the camera rectangle [0, LX] x [0, LY_CAM = 4.0]"
        s["gravel_relief"] = dict(enabled=False, grid_m=0.01, wavelength_m=[0.03, 0.08], rms_m=0.008, peak_max_m=0.025,
                                  edge_taper_m=0.05, heights="bumps only, >= 0 on top of the floor",
                                  rng_stream="default_rng([seed, 1])")
    # free blocks against the union of the three counterfactual sites' solids
    union = dict(sites[0])
    seen = set()
    ub = []
    for s in sites:
        for b in s["boxes"]:
            key = (b["name"], tuple(b["aabb"]))
            if key not in seen:
                seen.add(key)
                ub.append(b)
    union["boxes"] = ub
    free = w2.place_free_blocks(rng, union, cls)
    for s in sites:
        s["boxes"] += [dict(b) for b in free]
    dropped = 0
    g9s = None
    while True:
        xmls = [mjcf.site_xml(s) for s in sites]
        g9s = [w2p.g9_routes(s, x) for s, x in zip(sites, xmls)] if with_world_checks else [None] * 3
        if g9s[0] is None or all(g["pass"] for g in g9s):
            break
        fr = [k for k, b in enumerate(sites[0]["boxes"]) if b.get("free_block")]
        if not fr:
            break
        for s in sites:
            s["boxes"].pop(fr[-1])
        dropped += 1
    for s in sites:
        s["free_blocks"] = dict(placed=len(free), dropped_for_G9=dropped,
                                kept=[b["name"] for b in s["boxes"] if b.get("free_block")],
                                rule="W2's rule (A <= 3, B <= 2, keep-outs, spacing) against the union of the three "
                                     "counterfactual sites' solids; dropped from the last one while G9 fails on any of them")
    route_pts = []
    if g9s[0] is not None and g9s[0]["a"]["legs"]:
        for leg in g9s[0]["a"]["legs"][:2]:
            if leg["path"]:
                route_pts += leg["path"][::5]
    chips = place_chips(np.random.default_rng([int(seed), 2]), sites[0], route_pts)
    for s in sites:
        s["chips"] = [dict(c) for c in chips]
        s["chip_rule"] = ("W2's chip rule, default_rng([seed, 2]); route = pad -> 1 -> 2 of the tau = 0 counterfactual "
                          "(never enters the void)")
    if rubble:
        for t, s in enumerate(sites):
            add_rubble(seed, s, vis, hid[t], cls)
            s["generator"] = GENERATOR_VERSION
    else:
        for s in sites:
            s["generator"] = GENERATOR_VERSION + " without rubble (regression mode, not a version 1.1 site)"
    xmls = [mjcf.site_xml(s) for s in sites]
    g9s = [w2p.g9_routes(s, x) for s, x in zip(sites, xmls)] if with_world_checks else [None] * 3
    return dict(site=sites[tau_star], xml=xmls[tau_star], g9=g9s[tau_star], sites=sites, xmls=xmls, g9s=g9s,
                tau_star=tau_star, vis=vis, hid=hid, cls=cls, env=env)
