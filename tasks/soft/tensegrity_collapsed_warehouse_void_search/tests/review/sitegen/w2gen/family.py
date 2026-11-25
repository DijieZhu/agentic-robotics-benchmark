"""Site family W2 (the task's first site family): seed -> site dict. Static boxes only, no robot geometry.

Coordinates: origin = south-west floor corner, x east 0..LX, y north 0..LY, z up, metres. Bands along x:
A [0, a] (set-down pad, search point 1), R1 [a, a + t1] (rubble ridge with three openings), B [a + t1, y2]
(search point 2), R2 front row [y2, y2 + Lf] (true mouth M, blind mouth Bm, one decoy X), gallery
[y2 + Lf, LY] (the void under one slab; search point 3). The north perimeter wall is the gallery's north wall.

Rule-level rejections (a band depth out of range, a row that does not fit, search point 3 too near / far from
the true mouth, ...) are redrawn inside the seed's own random stream, so one seed always gives one site; the
predicates G1-G11 (predicates.py) are judged afterwards on that site.
"""
import math

import numpy as np

from . import GENERATOR_VERSION
from . import envelope as envmod

FAMILY = "W2"
BL = 0.33                      # body length = strut length end to end [R]
LX_DEFAULT = 4.4
WALL_T, WALL_H = 0.20, 0.90
SUP = 0.18                     # width of a support block that carries a roof
R1_ROOF_T = 0.10               # roof of the R1 tunnel and of every low slot
SLAB_T = 0.12                  # the void's slab
XL_CLEAR = 0.15                # clear height of a low slot (authors' low-clearance beam height)
X_NOTCH_DEPTH = 0.35           # depth of the R2 decoy before solid rubble
LEAN_RISE = 0.20               # east support of a leaning R1 tunnel roof is this much higher
ROOF_OVERHANG = 0.04
MAX_ATTEMPTS = 20000

CAMERA_MODEL = dict(width=640, height=480, fx=607.686, fy=607.987, cx=326.550, cy=250.662, rate_hz=7,
                    depth_unit_m=0.00025, depth_range_m=[0.25, 9.0])
# MuJoCo 3.14 principal point offset, measured when the family was built (a probe render): with pixel centres at
# integer coordinates (OpenCV / RealSense convention) the render matches the given K when
# principalpixel = (W/2 - 0.5 - cx, H/2 - 0.5 - cy). Re-measured on every site by predicate G3.
PIXEL_CONVENTION = "opencv: pixel (u, v) with integer u, v is the centre of column u, row v"


def principalpixel(cam=CAMERA_MODEL, sign=-1.0):
    return [r6(sign * (cam["cx"] - (cam["width"] / 2 - 0.5))), r6(sign * (cam["cy"] - (cam["height"] / 2 - 0.5)))]


# end-cap colours of the robot model (ysim/model.py SOCK_RGBA) - the colour chips copy them, lightness +-10 %
CAP_RGB = {"red": (0.80, 0.10, 0.10), "green": (0.10, 0.62, 0.20), "blue": (0.12, 0.25, 0.85)}
CRATE_RGB = [(0.78, 0.16, 0.14), (0.14, 0.30, 0.74)]       # first two free blocks: red and blue crates
TONE = dict(wall=(0.40, 0.40, 0.42), rubble=(0.62, 0.62, 0.60), support=(0.56, 0.56, 0.55),
            roof=(0.70, 0.69, 0.66), plug=(0.56, 0.56, 0.55), crate=(0.5, 0.5, 0.5))
FLOOR_RGB = (0.52, 0.52, 0.50)
PATCH_RGB = dict(dust=(0.68, 0.65, 0.58), gravel=(0.44, 0.40, 0.34))
BOX_MU = 0.80
PATCH_MU = dict(dust=0.35, gravel=0.60)


def r6(x):
    return round(float(x), 6)


# ------------------------------------------------------------------------------------------------ rules
def classes(env):
    v = env["values"]
    P_w, P_h = v["W_SWEEP"] + 0.14, v["H_ROLL"] + 0.12
    out = dict(
        P=dict(w=r6(P_w), clear_h=r6(P_h), rule="w = W_SWEEP + 0.14; covered: clear height = H_ROLL + 0.12"),
        T=dict(w=r6(v["W_SWEEP"] + 0.04), rule="w = W_SWEEP + 0.04, open top only"),
        XN=dict(w=r6(v["W_ROLL"] - 0.08), rule="w = W_ROLL - 0.08 (decoy: shorter than a strut)"),
        XL=dict(w=r6(P_w), clear_h=XL_CLEAR, rule="w as P, clear height 0.15 (must be < H_REST - 0.05)"),
        B=dict(w=r6(P_w), rule="blind mouth: looks like the true mouth, solid plug at depth p ~ U(0.25, Lf - 0.20)"),
        gallery=dict(clear_h=r6(v["H_ROLL"] + 0.14), depth_rule="g = W_SWEEP + 2 U(0.18, 0.25)"),
        free_block_clearance=r6(v["W_SWEEP"] + 0.18),
        plan_radius=r6(v["W_SWEEP"] / 2 + 0.04),
    )
    if not XL_CLEAR < v["H_REST"] - 0.05:
        raise ValueError("class XL: 0.15 m is not below H_REST - 0.05 = %.3f" % (v["H_REST"] - 0.05))
    return out


def site_depth(env):
    """LY = sum of the band medians rounded up to 0.1 m."""
    v = env["values"]
    med = 0.5 * (0.92 + 1.05) + 0.55 + 0.55 + (v["W_SWEEP"] + 2 * 0.215) + 0.5 * (0.88 + 1.10)
    return math.ceil(round(med * 10.0, 6)) / 10.0


def _box(name, kind, x0, x1, y0, y1, z0, z1, rgb=None, **extra):
    b = dict(name="%s__%s" % (kind, name), kind=kind, pos=[r6((x0 + x1) / 2), r6((y0 + y1) / 2), r6((z0 + z1) / 2)],
             size=[r6((x1 - x0) / 2), r6((y1 - y0) / 2), r6((z1 - z0) / 2)], quat=None,
             aabb=[r6(x0), r6(x1), r6(y0), r6(y1), r6(z0), r6(z1)], rgb=[r6(c) for c in (rgb or TONE[kind])])
    b.update(extra)
    return b


def _lean_slab(name, x0, x1, y0, y1, z_w, z_e, t):
    """A slab whose underside runs from (x0, z_w) to (x1, z_e), thickness t, rotated about the y axis."""
    L = math.hypot(x1 - x0, z_e - z_w)
    th = math.atan2(z_e - z_w, x1 - x0)
    nx, nz = -math.sin(th), math.cos(th)
    cx, cz = 0.5 * (x0 + x1) + nx * t / 2, 0.5 * (z_w + z_e) + nz * t / 2
    phi = -th                                    # rotation about +y that maps the box x axis onto (cos th, 0, sin th)
    quat = [r6(math.cos(phi / 2)), 0.0, r6(math.sin(phi / 2)), 0.0]
    corners_x, corners_z = [], []
    for sx in (-1, 1):
        for sz in (-1, 1):
            corners_x.append(cx + sx * L / 2 * math.cos(th) + sz * t / 2 * nx)
            corners_z.append(cz + sx * L / 2 * math.sin(th) + sz * t / 2 * nz)
    b = _box(name, "roof", x0, x1, y0, y1, 0, 0)
    b.update(pos=[r6(cx), r6(0.5 * (y0 + y1)), r6(cz)], size=[r6(L / 2), r6((y1 - y0) / 2), r6(t / 2)], quat=quat,
             aabb=[r6(min(corners_x)), r6(max(corners_x)), r6(y0), r6(y1), r6(min(corners_z)), r6(max(corners_z))],
             slope_deg=r6(math.degrees(th)))
    return b


def _row(rng, foot, mins_between, lx):
    """Openings left to right with solid segments between (ends >= 0.30, between >= the given minimum);
    the slack is shared by Dirichlet(2, ...). Returns (opening footprints, solid segments) or None."""
    n = len(foot)
    mins = [0.30] + list(mins_between) + [0.30]
    left = lx - sum(foot) - sum(mins)
    if left < 0:
        return None
    w = rng.dirichlet(np.ones(n + 1) * 2.0)
    segs = [m + left * float(q) for m, q in zip(mins, w)]
    x, feet, solids = 0.0, [], []
    for k in range(n):
        solids.append((x, x + segs[k]))
        x += segs[k]
        feet.append((x, x + foot[k]))
        x += foot[k]
    solids.append((x, lx))
    return feet, solids, left


def _split_long(x0, x1, limit=0.9):
    n = 2 if x1 - x0 > limit else 1
    xs = np.linspace(x0, x1, n + 1)
    return [(float(xs[i]), float(xs[i + 1])) for i in range(n)]


def _subtract(interval, cuts):
    """Parts of [a, b] outside every (c0, c1) in cuts."""
    a, b = interval
    parts = [(a, b)]
    for c0, c1 in cuts:
        nxt = []
        for p0, p1 in parts:
            if c1 <= p0 or c0 >= p1:
                nxt.append((p0, p1))
                continue
            if c0 > p0:
                nxt.append((p0, c0))
            if c1 < p1:
                nxt.append((c1, p1))
        parts = nxt
    return [(p0, p1) for p0, p1 in parts if p1 - p0 > 1e-6]


def _clip_split(interval, x0, x1):
    """[a, b] split into (inside [x0, x1], outside) lists."""
    a, b = interval
    inside = [(max(a, x0), min(b, x1))] if min(b, x1) - max(a, x0) > 1e-6 else []
    return inside, _subtract(interval, [(x0, x1)])


# ------------------------------------------------------------------------------------------------ one draw
def draw_stratum(seed):
    """Per-seed stratum, own stream default_rng([seed, 3]), drawn once: the SP3 mode and whether the R2 decoy lies
    under the slab. Redrawing them with the geometry would let the rejections couple them (first batch: every
    near_true site had its decoy under the slab, 24 of 24), and the decoy is in plain view."""
    r = np.random.default_rng([int(seed), 3])
    return dict(sp3_mode="near_true" if r.random() < 0.5 else "near_blind", r2_decoy="gap_XN",
                r2_decoy_under_slab=bool(r.random() < 0.5),
                rule="default_rng([seed, 3]): mode 50 / 50, decoy under the slab 50 / 50; R2 decoy always gap_XN")


def _draw(rng, env, cls, lx, ly, stratum):
    U = lambda lo, hi: float(rng.uniform(lo, hi))
    v = env["values"]
    a, t1, Lf = U(0.92, 1.05), U(0.50, 0.60), U(0.50, 0.60)
    g = v["W_SWEEP"] + 2 * U(0.18, 0.25)
    b = ly - a - t1 - Lf - g
    if not 0.88 <= b <= 1.10:
        return None, "band B depth outside [0.88, 1.10]: %.3f" % b
    y1b = a + t1
    y2 = y1b + b
    yg = y2 + Lf
    P_w, P_h, T_w, XN_w = cls["P"]["w"], cls["P"]["clear_h"], cls["T"]["w"], cls["XN"]["w"]
    boxes = []

    # ---------------- R1: gap_P at one end, tunnel_P and a third opening (T / XN / XL), whole row maybe flipped
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

    # ---------------- R2 front row: true mouth M, blind mouth Bm, decoy X = gap_XN (slot_XL is not used in R2,
    #                  because (near_true, slot_XL outside the slab) cannot be built at LX 4.4: an XL decoy would mark
    #                  the near_blind mode with certainty - 41 of 41 in the first batch)
    x_type = stratum["r2_decoy"]
    order2 = ["M", "Bm", "X"]
    rng.shuffle(order2)
    foot2 = dict(M=P_w + 2 * SUP, Bm=P_w + 2 * SUP, X=(XN_w if x_type == "gap_XN" else P_w + 2 * SUP))
    mins2 = [0.0 if {order2[i], order2[i + 1]} == {"M", "Bm"} else 0.45 for i in range(2)]
    r = _row(rng, [foot2[t] for t in order2], mins2, lx)
    if r is None:
        return None, "R2 front row does not fit"
    feet2, solids2, slack2 = r
    F = dict(zip(order2, feet2))
    cxM, cxB = 0.5 * sum(F["M"]), 0.5 * sum(F["Bm"])
    Mx0, Mx1 = F["M"][0] + SUP, F["M"][1] - SUP
    Bx0, Bx1 = F["Bm"][0] + SUP, F["Bm"][1] - SUP

    # ---------------- search point 3; the mode is the seed's stratum (drawn once, not redrawn on rejection)
    mode = stratum["sp3_mode"]
    if mode == "near_blind":
        sp3x = cxB + U(-0.10, 0.10)
        dM = abs(sp3x - cxM)
        if not 0.70 <= dM <= 1.30:
            return None, "near_blind |SP3.x - M| outside [0.70, 1.30]: %.3f" % dM
    else:
        side = -1.0 if cxB > cxM else 1.0
        dM = U(0.70, 1.00)
        sp3x = cxM + side * dM
    if not 0.80 <= sp3x <= lx - 0.80:
        return None, "SP3.x outside [0.80, LX - 0.80]: %.3f" % sp3x
    sp3y = yg + g / 2 + U(-0.05, 0.05)

    # ---------------- gallery, slab, plug
    e = U(0.80, 1.00)
    gal0 = min(Mx0 - U(0.05, 0.50), sp3x - e)
    gal1 = max(Mx1 + U(0.05, 0.50), sp3x + e)
    gal0, gal1 = max(gal0, 0.0), min(gal1, lx)
    p = U(0.25, Lf - 0.20)
    slab0 = max(0.0, min(gal0 - SUP, F["M"][0], F["Bm"][0]))
    slab1 = min(lx, max(gal1 + SUP, F["M"][1], F["Bm"][1]))
    if x_type == "slot_XL" and F["X"][1] > slab0 and F["X"][0] < slab1:
        return None, "slot_XL decoy under the slab"
    if (F["X"][1] > slab0 and F["X"][0] < slab1) != stratum["r2_decoy_under_slab"]:
        return None, "decoy cover differs from the stratum"
    hz = cls["gallery"]["clear_h"]          # slab underside = clear height of the gallery and of both mouths

    # front row boxes
    boxes.append(_box("sup_w_M", "support", F["M"][0], Mx0, y2, yg, 0, hz))
    boxes.append(_box("sup_e_M", "support", Mx1, F["M"][1], y2, yg, 0, hz))
    boxes.append(_box("sup_w_Bm", "support", F["Bm"][0], Bx0, y2, yg, 0, hz))
    boxes.append(_box("sup_e_Bm", "support", Bx1, F["Bm"][1], y2, yg, 0, hz))
    boxes.append(_box("Bm", "plug", Bx0, Bx1, y2 + p, yg, 0, hz))
    X0, X1 = F["X"]
    under_x = X1 > slab0 and X0 < slab1
    if x_type == "gap_XN":
        xh = hz if under_x else U(0.50, 0.88)
        boxes.append(_box("X_back", "support" if under_x else "rubble", X0, X1, y2 + X_NOTCH_DEPTH, yg, 0, xh))
        xop = dict(x0=r6(X0), x1=r6(X1), covered=bool(under_x), clear_h=r6(hz) if under_x else None,
                   clear_h_min=r6(hz) if under_x else 9.0)
    else:
        xi0, xi1 = X0 + SUP, X1 - SUP
        boxes.append(_box("sup_w_X", "support", X0, xi0, y2, y2 + X_NOTCH_DEPTH, 0, XL_CLEAR))
        boxes.append(_box("sup_e_X", "support", xi1, X1, y2, y2 + X_NOTCH_DEPTH, 0, XL_CLEAR))
        boxes.append(_box("roof_X", "roof", X0, X1, y2 - ROOF_OVERHANG, y2 + X_NOTCH_DEPTH, XL_CLEAR, XL_CLEAR + R1_ROOF_T))
        boxes.append(_box("X_back", "rubble", X0, X1, y2 + X_NOTCH_DEPTH, yg, 0, U(0.50, 0.88)))
        xop = dict(x0=r6(xi0), x1=r6(xi1), covered=True, clear_h=XL_CLEAR, clear_h_min=XL_CLEAR, roof="flat",
                   roof_t=R1_ROOF_T)
    for k, seg in enumerate(solids2):
        ins, outs = _clip_split(seg, slab0, slab1)
        for j, (s0, s1) in enumerate(ins):
            boxes.append(_box("R2_fill_%d_%d" % (k, j), "support", s0, s1, y2, yg, 0, hz))
        for j, part in enumerate(outs):
            for i, (s0, s1) in enumerate(_split_long(*part)):
                boxes.append(_box("R2_blk_%d_%d_%d" % (k, j, i), "rubble", s0, s1, y2, yg, 0, U(0.50, 0.88)))
    # gallery band: filled to the slab underside under the slab, rubble outside it
    for j, (s0, s1) in enumerate(_subtract((slab0, slab1), [(gal0, gal1)])):
        boxes.append(_box("void_fill_%d" % j, "support", s0, s1, yg, ly, 0, hz))
    for j, part in enumerate(_subtract((0.0, lx), [(slab0, slab1)])):
        for i, (s0, s1) in enumerate(_split_long(*part)):
            boxes.append(_box("back_blk_%d_%d" % (j, i), "rubble", s0, s1, yg, ly, 0, U(0.60, 0.90)))
    boxes.append(_box("void_slab", "roof", slab0, slab1, y2 - ROOF_OVERHANG, ly, hz, hz + SLAB_T))

    r2 = [dict(id="M", row="R2", type="mouth_true", cls="P", role="true mouth", x0=r6(Mx0), x1=r6(Mx1), y0=r6(y2),
               y1=r6(yg), foot=[r6(F["M"][0]), r6(F["M"][1])], covered=True, clear_h=r6(hz), clear_h_min=r6(hz),
               roof="void_slab", roof_t=SLAB_T, depth=r6(Lf)),
          dict(id="Bm", row="R2", type="mouth_blind", cls="B", role="blind mouth", x0=r6(Bx0), x1=r6(Bx1), y0=r6(y2),
               y1=r6(yg), foot=[r6(F["Bm"][0]), r6(F["Bm"][1])], covered=True, clear_h=r6(hz), clear_h_min=r6(hz),
               roof="void_slab", roof_t=SLAB_T, depth=r6(p), plug_depth_m=r6(p)),
          dict(id="X", row="R2", type=x_type, cls=("XN" if x_type == "gap_XN" else "XL"), role="decoy", y0=r6(y2),
               y1=r6(y2 + X_NOTCH_DEPTH), foot=[r6(X0), r6(X1)], depth=X_NOTCH_DEPTH, **xop)]
    r2.sort(key=lambda q: q["foot"][0])

    # ---------------- pad, search points 1 and 2
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

    # ---------------- ground patches
    gx0 = max(0.0, min(Mx0, Bx0) - 0.15)
    gx1 = min(lx, max(Mx1, Bx1) + 0.15)
    dx0 = float(np.clip(0.5 * (pad[0] + sp1[0]) - 0.45, 0.05, lx - 0.95))
    patches = [dict(name="dust", x0=r6(dx0), x1=r6(dx0 + 0.90), y0=0.25, y1=0.85, mu=PATCH_MU["dust"], thickness=0.002,
                    rgb=list(PATCH_RGB["dust"])),
               dict(name="gravel", x0=r6(gx0), x1=r6(gx1), y0=r6(y2 - 0.60), y1=r6(y2), mu=PATCH_MU["gravel"],
                    thickness=0.002, rgb=list(PATCH_RGB["gravel"]))]

    layout = dict(
        params=dict(a=r6(a), t1=r6(t1), Lf=r6(Lf), g=r6(g), b=r6(b), y2=r6(y2), plug_depth_p=r6(p), e=r6(e),
                    slack_R1_m=r6(slack1), slack_R2_m=r6(slack2)),
        bands=dict(A=[0.0, r6(a)], R1=[r6(a), r6(y1b)], B=[r6(y1b), r6(y2)], R2_front=[r6(y2), r6(yg)],
                   gallery=[r6(yg), r6(ly)]),
        r1=dict(order=order, third=third, tunnel_roof=roof_kind, openings=r1),
        r2=dict(order=order2, decoy_type=x_type, openings=r2),
        void=dict(sp3_mode=mode, true_mouth="M", blind_mouth="Bm", sp3_minus_true_mouth_x=r6(sp3x - cxM),
                  nearest_P_mouth_is_true=bool(abs(sp3x - cxM) < abs(sp3x - cxB)),
                  gallery=dict(x0=r6(gal0), x1=r6(gal1), y0=r6(yg), y1=r6(ly), clear_h=r6(hz)),
                  slab=dict(x0=r6(slab0), x1=r6(slab1), y0=r6(y2 - ROOF_OVERHANG), y1=r6(ly), z0=r6(hz), z1=r6(hz + SLAB_T)),
                  plug=dict(x0=r6(Bx0), x1=r6(Bx1), y0=r6(y2 + p), y1=r6(yg), z1=r6(hz))),
        pad=dict(xy=[r6(pad[0]), r6(pad[1])], retrieval_radius_m=0.60, setdown_error_m=0.10),
        search_points=[dict(id=1, xy=[r6(sp1[0]), r6(sp1[1])], weight=0.15, where="band A, open floor"),
                       dict(id=2, xy=[r6(sp2[0]), r6(sp2[1])], weight=0.25, where="band B, open floor"),
                       dict(id=3, xy=[r6(sp3x), r6(sp3y)], weight=0.60, where="gallery under the slab", mode=mode)],
        patches=patches,
        boxes=boxes,
    )
    return layout, None


# ------------------------------------------------------------------------------------------------ camera
def cam_axes(tilt_deg, yaw_deg):
    """Columns: camera right (+u), image down (+v), optical axis (forward) in world coordinates."""
    t, y = math.radians(tilt_deg), math.radians(yaw_deg)
    fwd = np.array([math.sin(t) * math.sin(y), math.sin(t) * math.cos(y), -math.cos(t)])
    right = np.array([math.cos(y), -math.sin(y), 0.0])
    down = np.cross(fwd, right)
    return np.stack([right, down, fwd], axis=1)


def project(pos, R, pts, cam=CAMERA_MODEL):
    pc = (np.asarray(pts, float) - np.asarray(pos, float)) @ np.asarray(R, float)
    u = cam["fx"] * pc[:, 0] / pc[:, 2] + cam["cx"]
    v = cam["fy"] * pc[:, 1] / pc[:, 2] + cam["cy"]
    return u, v, pc[:, 2]


def corner_margin(pos, R, lx, ly, cam=CAMERA_MODEL):
    corners = np.array([[0, 0, 0], [lx, 0, 0], [lx, ly, 0], [0, ly, 0]], float)
    u, v, _ = project(pos, R, corners, cam)
    W, H = cam["width"], cam["height"]
    return float(min((u + 0.5).min(), (v + 0.5).min(), (W - 0.5 - u).min(), (H - 0.5 - v).min())), u, v


def _rotvec(axis, ang):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def _unit(rng):
    while True:
        q = rng.normal(size=3)
        n = float(np.linalg.norm(q))
        if n > 1e-6:
            return q / n


def draw_camera(rng, lx, ly, cam=CAMERA_MODEL):
    U = lambda lo, hi: float(rng.uniform(lo, hi))
    pos = [lx / 2 + U(-0.15, 0.15), U(0.15, 0.35), U(4.75, 4.95)]
    yaw = U(-1.5, 1.5)
    inset = U(2.0, 2.8)
    margin = None
    for _ in range(200):
        tilt = math.degrees(math.atan((cam["height"] - cam["cy"]) / cam["fy"])) - math.degrees(math.atan(pos[1] / pos[2])) - inset
        R = cam_axes(tilt, yaw)
        margin, _u, _v = corner_margin(pos, R, lx, ly, cam)
        if margin >= 12.0:
            break
        pos[2] += 0.05
    # calibration error given to the agent (translation <= 5 mm, rotation <= 0.2 deg), drawn once per site
    t_err = _unit(rng) * U(0.0, 0.005)
    r_axis, r_ang = _unit(rng), math.radians(U(0.0, 0.2))
    R_err = _rotvec(r_axis, r_ang)
    R_true = cam_axes(tilt, yaw)
    T_true = np.eye(4)
    T_true[:3, :3], T_true[:3, 3] = R_true, pos
    T_given = np.eye(4)
    T_given[:3, :3], T_given[:3, 3] = R_err @ R_true, np.asarray(pos) + t_err
    rnd = lambda M: [[r6(q) for q in row] for row in np.asarray(M)]
    return dict(
        model=dict(cam, principalpixel_mujoco=principalpixel(cam), pixel_convention=PIXEL_CONVENTION,
                   sensorsize_m=[r6(cam["width"] * 1e-5), r6(cam["height"] * 1e-5)]),
        rule=dict(x="LX/2 + U(-0.15, 0.15)", y="U(0.15, 0.35)", z="U(4.75, 4.95) + 0.05 k until corners >= 12 px",
                  yaw_deg="U(-1.5, 1.5)", tilt_from_nadir_deg="atan((H - cy) / fy) - atan(y / z) - U(2.0, 2.8)"),
        true=dict(pos=[r6(q) for q in pos], tilt_deg=r6(tilt), yaw_deg=r6(yaw), inset_deg=r6(inset),
                  corner_margin_px=r6(margin), T_world_camera=rnd(T_true),
                  frame="camera axes: x right (+u), y down (+v), z forward (optical axis)"),
        given=dict(T_world_camera=rnd(T_given)),
        calibration_error=dict(translation_m=[r6(q) for q in t_err], translation_norm_m=r6(np.linalg.norm(t_err)),
                               rotation_axis=[r6(q) for q in r_axis], rotation_deg=r6(math.degrees(r_ang)),
                               rule="T_given = (R_err R_true, t_true + dt); |dt| ~ U(0, 5 mm), angle ~ U(0, 0.2 deg)"),
    )


def draw_lighting(rng):
    U = lambda lo, hi: float(rng.uniform(lo, hi))
    n0 = np.array([0.15, 0.25, -1.0])
    n0 /= np.linalg.norm(n0)
    ang = math.radians(U(0.0, 25.0))
    az = U(0.0, 2 * math.pi)
    e1 = np.cross(n0, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n0, e1)
    d = math.cos(ang) * n0 + math.sin(ang) * (math.cos(az) * e1 + math.sin(az) * e2)
    key = 0.75 * U(0.7, 1.3)
    amb = 0.35 * U(0.7, 1.3)
    return dict(key_dir=[r6(q) for q in d], key_off_nominal_deg=r6(math.degrees(ang)), key_diffuse=r6(key),
                ambient=r6(amb), nominal=dict(key_dir=[r6(q) for q in n0], key_diffuse=0.75, ambient=0.35),
                rule="key direction within 25 deg of nominal, key and ambient x U(0.7, 1.3); no direct sunlight")


# ------------------------------------------------------------------------------------------------ free blocks
def _rect_gap(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[1], b[1]))
    dy = max(0.0, max(a[2], b[2]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def _rect_overlap(a, b):
    return (min(a[1], b[1]) - max(a[0], b[0]) > 1e-9) and (min(a[3], b[3]) - max(a[2], b[2]) > 1e-9)


def spacing_ok(rect, others, clr):
    """Free-block spacing (the family's rule as built): to every other solid the gap is >= clr, or the block
    stands flush against it (gap exactly 0, no overlap). A literal '>= clr to everything' admits no block at all
    in bands 0.88-1.10 m deep (measured when the family was built)."""
    for o in others:
        if _rect_overlap(rect, o):
            return False
        gp = _rect_gap(rect, o)
        if 0.0 < gp < clr:
            return False
    return True


def place_free_blocks(rng, site, cls):
    U = lambda lo, hi: float(rng.uniform(lo, hi))
    lx, ly = site["size_m"]
    clr = cls["free_block_clearance"]
    pad = site["pad"]["xy"]
    sp = {s["id"]: s["xy"] for s in site["search_points"]}
    keep = [(pad, 0.45), (sp[1], 0.40), (sp[2], 0.40)]
    for row in ("r1", "r2"):
        for gq in site[row]["openings"]:
            gx = 0.5 * (gq["x0"] + gq["x1"])
            keep += [((gx, gq["y0"] - 0.25), 0.40), ((gx, gq["y1"] + 0.25), 0.40)]
    walls = [(-WALL_T, 0.0, -WALL_T, ly + WALL_T), (lx, lx + WALL_T, -WALL_T, ly + WALL_T),
             (-WALL_T, lx + WALL_T, -WALL_T, 0.0), (-WALL_T, lx + WALL_T, ly, ly + WALL_T)]
    solids = walls + [tuple(b["aabb"][:4]) for b in site["boxes"] if b["kind"] in ("rubble", "support", "plug")]
    boxes3d = [b["aabb"] for b in site["boxes"]]
    patches = [(p["x0"], p["x1"], p["y0"], p["y1"]) for p in site["patches"]]
    A0, A1 = site["bands"]["A"]
    B0, B1 = site["bands"]["B"]
    placed = []
    for band, (ya, yb, n) in (("A", (A0, A1, 3)), ("B", (B0, B1, 2))):
        k = 0
        for _ in range(400):
            if k >= n:
                break
            w, d, h = U(0.20, 0.38), U(0.20, 0.34), U(0.14, 0.50)
            my, mx = int(rng.integers(0, 3)), float(rng.random())
            if my == 0:
                y0 = ya
            elif my == 1:
                y0 = yb - d
            else:
                y0 = U(ya, yb - d)
            if mx < 0.15:
                x0 = 0.0
            elif mx < 0.30:
                x0 = lx - w
            else:
                x0 = U(0.0, lx - w)
            rect = (x0, x0 + w, y0, y0 + d)
            cx, cy = x0 + w / 2, y0 + d / 2
            if not all(math.hypot(cx - p[0], cy - p[1]) > r + 0.5 * math.hypot(w, d) for p, r in keep):
                continue
            if any(_rect_overlap(rect, q) for q in patches):
                continue
            if not spacing_ok(rect, solids + [q["rect"] for q in placed], clr):
                continue
            if any(_rect_overlap(rect, b[:4]) and h > b[4] + 1e-9 and b[5] > 1e-9 for b in boxes3d):
                continue
            col = CRATE_RGB[len(placed)] if len(placed) < len(CRATE_RGB) else None
            name = "free_%s%d" % (band, k)
            box = _box(name, "crate" if col else "rubble", rect[0], rect[1], rect[2], rect[3], 0.0, h, rgb=col,
                       free_block=True, placement=dict(y=("flush_south", "flush_north", "free")[my],
                                                       x=("flush_west" if mx < 0.15 else "flush_east" if mx < 0.30 else "free")))
            placed.append(dict(rect=rect, box=box))
            k += 1
    return [q["box"] for q in placed]


# ------------------------------------------------------------------------------------------------ colour chips
def place_chips(rng, site, route_pts):
    """8-14 flat 30-60 mm chips in the three end-cap colours, visual only; >= 0.15 m from the pad and the search
    points; at least 3 within 0.4 m of the reference route (here: the G9a shortest route, open part)."""
    U = lambda lo, hi: float(rng.uniform(lo, hi))
    lx, ly = site["size_m"]
    n = int(rng.integers(8, 15))
    pad = site["pad"]["xy"]
    pts = [pad] + [s["xy"] for s in site["search_points"]]
    solid = [b["aabb"][:4] for b in site["boxes"] if b["kind"] in ("rubble", "support", "crate", "plug", "roof")]
    patch_top = {p["name"]: (p["x0"], p["x1"], p["y0"], p["y1"], p["thickness"]) for p in site["patches"]}
    chips = []
    route_pts = np.asarray(route_pts, float) if len(route_pts) else np.zeros((0, 2))
    for k in range(n):
        for _ in range(500):
            s = U(0.03, 0.06)
            shape = "disc" if rng.random() < 0.5 else "square"
            if k < 3 and len(route_pts):
                c = route_pts[int(rng.integers(0, len(route_pts)))]
                rr, aa = U(0.05, 0.35), U(0, 2 * math.pi)
                x, y = c[0] + rr * math.cos(aa), c[1] + rr * math.sin(aa)
            else:
                x, y = U(0.08, lx - 0.08), U(0.08, ly - 0.08)
            half = s / 2 * (1.0 if shape == "disc" else math.sqrt(2.0))
            if not (half < x < lx - half and half < y < ly - half):
                continue
            if any(math.hypot(x - p[0], y - p[1]) < 0.15 + half for p in pts):
                continue
            rect = (x - half, x + half, y - half, y + half)
            if any(_rect_overlap(rect, q) or _rect_gap(rect, q) < 0.01 for q in solid):
                continue
            if any(math.hypot(x - q["xy"][0], y - q["xy"][1]) < 0.08 for q in chips):
                continue
            z = 0.0
            for (px0, px1, py0, py1, t) in patch_top.values():
                if px0 - half <= x <= px1 + half and py0 - half <= y <= py1 + half:
                    z = t
            colour = ("red", "green", "blue")[int(rng.integers(0, 3))]
            light = U(0.9, 1.1)
            rgb = [r6(min(1.0, c_ * light)) for c_ in CAP_RGB[colour]]
            dmin = float(np.min(np.hypot(route_pts[:, 0] - x, route_pts[:, 1] - y))) if len(route_pts) else None
            chips.append(dict(name="chip__%d" % k, shape=shape, size_m=r6(s), thickness_m=0.002, xy=[r6(x), r6(y)],
                              z_m=r6(z), yaw_deg=r6(U(0, 90)), colour=colour, lightness=r6(light), rgb=rgb,
                              dist_to_route_m=None if dmin is None else r6(dmin)))
            break
    return chips


# ------------------------------------------------------------------------------------------------ the site
def make_site(seed, env=None, lx=LX_DEFAULT, gravel_relief=False):
    """Rule-level layout of one seed (free blocks and chips are added by build_site)."""
    env = env or envmod.load()
    cls = classes(env)
    ly = site_depth(env)
    rng = np.random.default_rng(seed)
    stratum = draw_stratum(seed)
    reasons = {}
    for attempt in range(1, MAX_ATTEMPTS + 1):
        layout, why = _draw(rng, env, cls, lx, ly, stratum)
        if layout:
            break
        key = why.split(":")[0] if ":" in why else " ".join(why.split()[:3])
        reasons[key] = reasons.get(key, 0) + 1
    else:
        raise RuntimeError("seed %d: no valid draw in %d attempts" % (seed, MAX_ATTEMPTS))
    site = dict(
        family=FAMILY, generator=GENERATOR_VERSION, seed=int(seed), draw_attempts=attempt,
        draw_rejections=dict(sorted(reasons.items())), stratum=stratum,
        frame="x east, y north, z up, origin = south-west floor corner; metres",
        size_m=[r6(lx), r6(ly)], body_length_m=BL,
        envelope=env, classes=cls,
    )
    site.update(layout)
    site["walls"] = []
    for nm, (x0, x1, y0, y1) in dict(s=(-WALL_T, lx + WALL_T, -WALL_T, 0.0), n=(-WALL_T, lx + WALL_T, ly, ly + WALL_T),
                                     w=(-WALL_T, 0.0, 0.0, ly), e=(lx, lx + WALL_T, 0.0, ly)).items():
        site["boxes"].insert(len(site["walls"]), _box(nm, "wall", x0, x1, y0, y1, 0.0, WALL_H))
        site["walls"].append("wall__" + nm)
    site["camera"] = draw_camera(rng, lx, ly)
    site["lighting"] = draw_lighting(rng)
    site["physics"] = dict(floor_mu=r6(rng.uniform(0.76, 0.92)), box_mu=BOX_MU, patch_mu=dict(PATCH_MU),
                           floor_mu_rule="U(0.76, 0.92) per site (authors' sysID range)")
    site["gravel_relief"] = dict(enabled=bool(gravel_relief), grid_m=0.01, wavelength_m=[0.03, 0.08], rms_m=0.008,
                                 peak_max_m=0.025, edge_taper_m=0.05, heights="bumps only, >= 0 on top of the floor",
                                 rng_stream="default_rng([seed, 1])")
    site["_rng_state_after_layout"] = None
    return site, rng, cls


def build_site(seed, env=None, lx=LX_DEFAULT, gravel_relief=False, with_world_checks=True):
    """Full site: layout, free blocks (dropped from the last one while G9 fails), colour chips.
    Returns (site, xml, check) where check holds the predicate results computed on the way (G9)."""
    from . import mjcf, predicates
    site, rng, cls = make_site(seed, env, lx, gravel_relief)
    site.pop("_rng_state_after_layout")
    free = place_free_blocks(rng, site, cls)
    site["boxes"] += free
    dropped = 0
    g9 = None
    while True:
        xml = mjcf.site_xml(site)
        g9 = predicates.g9_routes(site, xml) if with_world_checks else None
        if g9 is None or g9["pass"]:
            break
        fr = [k for k, b in enumerate(site["boxes"]) if b.get("free_block")]
        if not fr:
            break
        site["boxes"].pop(fr[-1])
        dropped += 1
    site["free_blocks"] = dict(placed=len(free), dropped_for_G9=dropped,
                               kept=[b["name"] for b in site["boxes"] if b.get("free_block")],
                               rule="A <= 3, B <= 2; keep-outs pad 0.45, search points 0.40, 0.40 around a point 0.25 m "
                                    "in front of / behind each opening; gap to every other solid >= W_SWEEP + 0.18 or flush; "
                                    "off the ground patches; dropped from the last one while G9 fails")
    route_pts = []
    if g9 is not None and g9["a"]["legs"]:
        for leg in g9["a"]["legs"][:2]:          # pad -> 1 -> 2: the open, visible part of the reference route
            if leg["path"]:
                route_pts += leg["path"][::5]
    site["chips"] = place_chips(np.random.default_rng([int(seed), 2]), site, route_pts)
    site["chip_rule"] = ("8-14 chips, 30-60 mm discs or squares, 2 mm, visual only (contype = conaffinity = 0), "
                         "end-cap colours x U(0.9, 1.1); >= 0.15 m from pad and search points; first 3 within 0.4 m "
                         "of the G9a route (pad -> 1 -> 2); rng default_rng([seed, 2])")
    xml = mjcf.site_xml(site)
    return site, xml, g9
