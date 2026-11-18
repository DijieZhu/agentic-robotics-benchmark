"""Reference reader of the dial gauge (bricks.targets.gauge).

Moved here from bricks.targets.gauge on 2026-11-14, unchanged. Reader: find the bright elliptical
dial, rectify it affinely, find the needle as the darkest ray between 18 % and 55 % of the dial radius, find the
scale's end ticks around the 90 degree gap, and interpolate the needle angle between them (only the hint's range and
the 270 degree convention).
"""

from __future__ import annotations

import math

import numpy as np

from .. import vision
from ..targets.gauge import SWEEP_DEG  # noqa: F401  (geometry of the instrument, catalogue values)

__all__ = ["read", "GAUGE_HI_FRACS"]

GAUGE_HI_FRACS = (0.80, 0.86, 0.92)   # extra thresholds of the second try (fractions of the bright percentile)
# 2026-11-16, the dials on the frozen look: (1) a face clipped white (the camera's exposure set by a
# dark pipe behind the dial) keeps too little tick contrast for the 0.03 floor of the tick annulus: the floor is lower
# there; (2) the second try also fits the islands of bright regions (a grey face inside its bezel on a brighter wall),
# and it runs whenever the first candidates gave no reading (it ran only when there was no candidate at all)
TICK_LEVEL_MIN = 0.03
TICK_LEVEL_MIN_CLIPPED = 0.018
CLIPPED_WHITE = 0.985


# ----------------------------------------------------------------------------------------------- reader
def _needle_line(Dk, rho_px, t0, r_lo=0.12, r_hi=0.62, band=0.07, iters=3):
    """Refine the needle direction on the rectified darkness image Dk (circle of radius rho_px at the image centre)
    by an iterative weighted line fit of the dark pixels near the current line (the line need not pass through the
    centre: the needle floats above the dial, so parallax shifts it sideways). Returns (angle rad, linearity) or None."""
    N = Dk.shape[0]
    c = (N - 1) / 2.0
    yy, xx = np.mgrid[0:N, 0:N]
    X = (xx - c) / rho_px
    Y = -(yy - c) / rho_px
    r = np.hypot(X, Y)
    ring = (r >= r_lo) & (r <= r_hi)
    p0 = np.zeros(2)
    d = np.array([math.cos(t0), math.sin(t0)])
    ang, lin = None, 0.0
    for _ in range(iters):
        nrm = np.array([-d[1], d[0]])
        dist = (X - p0[0]) * nrm[0] + (Y - p0[1]) * nrm[1]
        along = (X - p0[0]) * d[0] + (Y - p0[1]) * d[1]
        sel = ring & (np.abs(dist) <= band) & (along > 0)
        w = (Dk * sel) ** 2
        ws = float(w.sum())
        if ws < 1e-6:
            return None
        mx, my = float((w * X).sum() / ws), float((w * Y).sum() / ws)
        cxx = float((w * (X - mx) ** 2).sum() / ws)
        cyy = float((w * (Y - my) ** 2).sum() / ws)
        cxy = float((w * (X - mx) * (Y - my)).sum() / ws)
        ev, evec = np.linalg.eigh(np.array([[cxx, cxy], [cxy, cyy]]))
        dn = evec[:, 1]
        if dn @ d < 0:
            dn = -dn
        d = dn
        # anchor the line at the weighted centroid, then step back along the needle towards the hub
        p0 = np.array([mx, my]) - d * 0.35
        lin = float(ev[1] / max(ev[0], 1e-12))
        ang = math.atan2(d[1], d[0])
    return ang, lin


def _rectify(gray, ell, N):
    """Affine warp of the ellipse to a centred circle of radius 0.48 N in an N x N image (y up in dial terms)."""
    import cv2
    cx, cy, a, b, ang = ell
    u = np.array([math.cos(ang), math.sin(ang)])
    v = np.array([math.sin(ang), -math.cos(ang)])
    A = np.column_stack([a * u, b * v])
    rho = 0.48 * N
    S = np.array([[1.0 / rho, 0.0], [0.0, -1.0 / rho]])
    M = A @ S
    c = (N - 1) / 2.0
    t = np.array([cx, cy]) - M @ np.array([c, c])
    src = gray
    scale = min(a, b) / rho
    if scale > 1.3:
        src = cv2.GaussianBlur(gray, (0, 0), 0.5 * scale)
    out = cv2.warpAffine(src.astype(np.float32), np.hstack([M, t[:, None]]).astype(np.float64), (N, N),
                         flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
    return out, rho


def _read_on_ellipse(gray, ell, lo, hi, nominal_sweep):
    N = int(np.clip(2.2 * ell[2], 120, 640))
    Rimg, rho_px = _rectify(gray, ell, N)
    c = (N - 1) / 2.0
    ell_r = (c, c, rho_px, rho_px, 0.0)
    P, th, rho = vision.polar_sample(Rimg, ell_r, n_theta=720, rho_max=1.0, n_rho=100)
    nt = len(th)
    deg_per_bin = 360.0 / nt
    inner = P[:, rho < 0.97]
    white = float(np.percentile(inner, 92))
    black = float(np.percentile(inner, 1.0))
    if white - black < 0.04:
        return None
    # v3: the dial's white level per angle (a large dial under a near light is brighter on one side; against one
    # global white level its thin ticks vanished there and an arc of them read as the 90 degree gap: 160 mm dials
    # seen from below were unreadable in 26 % of in-region frames)
    face = (rho >= 0.30) & (rho < 0.97)
    w_ang = vision.circ_smooth(np.percentile(P[:, face], 92, axis=1), int(round(8.0 / (360.0 / len(th)))))
    w_ang = np.maximum(w_ang, black + 0.04)[:, None]
    Dk = np.clip((w_ang - P) / (w_ang - black), 0.0, 1.0)
    Dimg = np.clip((white - Rimg) / (white - black), 0.0, 1.0)
    # needle: the darkest long ray between 18 % and 55 % of the radius
    band = (rho >= 0.18) & (rho <= 0.55)
    prof = vision.circ_smooth(Dk[:, band].mean(1), 2)
    k = int(np.argmax(prof))
    contrast = float(prof[k] - np.median(prof))
    t_ray = math.radians(vision.subpixel_peak(prof, k, circular=True) * deg_per_bin)
    t_needle, lin, used_fit = t_ray, 0.0, False
    fit = _needle_line(Dimg, rho_px, t_ray)
    if fit is not None:
        t_fit, lin = fit
        if lin > 6.0 and abs(((t_fit - t_ray + math.pi) % (2 * math.pi)) - math.pi) < math.radians(15):
            t_needle, used_fit = t_fit, True
    # scale ticks: angular darkness profile in the tick annulus, smoothed over ~20 deg; the gap is its longest low arc
    tb = (rho >= 0.74) & (rho <= 0.95)
    tp = Dk[:, tb].mean(1)
    tps = vision.circ_smooth(tp, int(round(10.0 / deg_per_bin)))
    level = float(np.percentile(tps, 70))
    if level < (TICK_LEVEL_MIN_CLIPPED if white >= CLIPPED_WHITE else TICK_LEVEL_MIN):
        return None
    s0, ln = vision.longest_run(tps < 0.45 * level)
    if s0 is None or ln * deg_per_bin < 30:
        return None
    gap_w = ln * deg_per_bin
    thr = 0.5 * float(np.percentile(tp, 95))

    def end_tick(edge_bin, direction):
        # walk from just inside the gap outwards to the first strong tick (the scale's end tick)
        for j in range(-int(4 / deg_per_bin), int(14 / deg_per_bin)):
            i = (edge_bin + direction * j) % nt
            if tp[i] >= thr and tp[i] >= tp[(i - 1) % nt] and tp[i] >= tp[(i + 1) % nt]:
                return vision.subpixel_peak(tp, i, circular=True) * deg_per_bin
        return edge_bin * deg_per_bin

    # the gap runs counter-clockwise from the low-end tick to the high-end tick
    t_lo = end_tick(s0, -1)
    t_hi = end_tick((s0 + ln - 1) % nt, +1)
    sweep = (t_lo - t_hi) % 360.0
    tn = math.degrees(t_needle) % 360.0
    u = ((t_lo - tn) % 360.0) / sweep
    if u > 1.0 + (360.0 - sweep) / sweep / 2.0:
        u -= 360.0 / sweep
    value = lo + u * (hi - lo)
    q_sweep = math.exp(-((sweep - nominal_sweep) / 20.0) ** 2)
    q_needle = float(np.clip(contrast / 0.25, 0.0, 1.0))
    conf = q_sweep * q_needle
    dbg = {"ellipse": [float(v) for v in ell], "needle_deg": tn, "ray_deg": math.degrees(t_ray) % 360.0,
           "line_fit": used_fit, "linearity": lin, "lo_tick_deg": t_lo, "hi_tick_deg": t_hi, "sweep_deg": sweep,
           "gap_deg": gap_w, "u": u, "needle_contrast": contrast, "white": white, "black": black}
    return value, conf, dbg


def read(rgb: np.ndarray, hint: dict):
    """Read a dial gauge. hint: {'range': (lo, hi), optional 'sweep_deg' (270)}. Returns (value, confidence, debug);
    value is None when no dial is found."""
    lo, hi = hint["range"]
    sweep = float(hint.get("sweep_deg", SWEEP_DEG))
    gray = vision.gray01(rgb)
    best = None
    tried = []

    def _try(cands, skip=()):
        nonlocal best
        for ell, sc in cands[:5]:
            if any(math.hypot(ell[0] - e[0], ell[1] - e[1]) < 0.1 * e[2] and abs(ell[2] - e[2]) < 0.1 * e[2]
                   for e in skip):
                continue
            r = _read_on_ellipse(gray, ell, lo, hi, sweep)
            tried.append({"ellipse": [float(v) for v in ell], "score": sc, "ok": r is not None,
                          "conf": None if r is None else r[1]})
            if r is None:
                continue
            key = r[1] * (0.5 + 0.5 * sc / max(cands[0][1], 1e-9))
            if best is None or key > best[0]:
                best = (key, r)

    cands = vision.find_bright_ellipses(gray, min_semi_axis_px=6.0)
    _try(cands)
    if best is None:
        # 2026-11-14: no dial found; the faces whose contour the needle and the digits notch (a review found
        # 100 mm dials seen 30-40 deg from below), fitted on the contour's convex hull, and faces inside a dark bezel
        # on a light wall (a hole of the wall's region at the plain thresholds), with thresholds near the face's white;
        # 2026-11-16: also the islands of bright regions, and whenever the first candidates gave no reading
        cands2 = vision.find_bright_ellipses(gray, min_semi_axis_px=6.0, hull=True, hi_fracs=GAUGE_HI_FRACS,
                                             islands=True)
        _try(cands2, skip=[c[0] for c in cands[:5]])
    if best is None:
        return None, 0.0, {"candidates": tried}
    value, conf, dbg = best[1]
    dbg["candidates"] = tried
    return float(value), float(conf), dbg
