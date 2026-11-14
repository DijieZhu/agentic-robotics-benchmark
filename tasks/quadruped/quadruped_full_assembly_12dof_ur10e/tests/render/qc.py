"""Measurement routines (numpy only, no OpenCV) that check a photo is USABLE: its scale references are visible and
measurable.  Used by the tests, by photo_set.py (visibility stats) and by task authors as a QC gate on a photo set.

    gray(rgb)                                   -> float luminance 0..1
    saddle_response(g, d)                       -> checkerboard X-corner response (signed)
    detect_x_corners(g, d, rel_threshold, min_dist, mask) -> (N, 2) sub-pixel corner candidates (u, v), no ground truth
    match_points(detected, expected, tol_px)    -> (matched expected indices, errors, unmatched detections)
    profile(g, p0, p1, n)                       -> intensities sampled along a segment (bilinear)
    dark_runs(values, rel)                      -> centres (in samples) of dark runs, e.g. ruler ticks
    ref_pixels(seg, labels)                     -> {scale_ref name: visible pixel count}
    measure_checkerboard(rgb, camera, ref)      -> {n_expected, n_detected_matched, rms_px, max_px}  (grader-side check)
    measure_ruler(rgb, camera, ref)             -> {n_ticks_expected, n_ticks_found, rms_mm, rms_px, spacing_px}
    measure_grid_lines(rgb, camera, ref)        -> cutting-mat grid lines crossed by the projected mid-row: found, rms (px)
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def gray(rgb) -> np.ndarray:
    a = np.asarray(rgb, np.float64)
    if a.ndim == 3:
        a = a @ np.array([0.299, 0.587, 0.114])
    return a / 255.0 if np.asarray(rgb).dtype == np.uint8 else a


def _shift(g, dy, dx):
    H, W = g.shape
    p = max(abs(dy), abs(dx))
    gp = np.pad(g, p, mode="edge")
    return gp[p + dy:p + dy + H, p + dx:p + dx + W]


def saddle_response(g, d=2) -> np.ndarray:
    """ChESS-style X-corner response at radius d: |a1 + a2 - b1 - b2| - |a1 - a2| - |b1 - b2| for the diagonal sample
    pairs (a, b) and for the axis pairs at radius d sqrt 2 (maximum of the two: rotation-tolerant).  High at
    checkerboard X-corners, ~0 on edges, L-junctions (board margin) and flat regions."""
    def resp(p1, p2, q1, q2):
        a1, a2 = _shift(g, *p1), _shift(g, *p2)
        b1, b2 = _shift(g, *q1), _shift(g, *q2)
        return np.abs(a1 + a2 - b1 - b2) - np.abs(a1 - a2) - np.abs(b1 - b2)
    dd = max(1, int(round(d * 1.41421356)))
    r1 = resp((-d, -d), (d, d), (-d, d), (d, -d))
    r2 = resp((-dd, 0), (dd, 0), (0, -dd), (0, dd))
    return np.maximum(np.maximum(r1, r2), 0.0)


def _box(x, r):
    if r <= 0:
        return x
    c = np.cumsum(np.cumsum(np.pad(x, ((r + 1, r), (r + 1, r))), 0), 1)
    return (c[2 * r + 1:, 2 * r + 1:] - c[:-2 * r - 1, 2 * r + 1:] - c[2 * r + 1:, :-2 * r - 1] + c[:-2 * r - 1, :-2 * r - 1])


def _gauss(g, sigma=1.0):
    r = max(1, int(3 * sigma))
    x = np.arange(-r, r + 1)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= k.sum()
    gp = np.pad(g, r, mode="edge")
    h = sum(k[i] * gp[:, i:i + g.shape[1]] for i in range(2 * r + 1))
    return sum(k[i] * h[i:i + g.shape[0], :] for i in range(2 * r + 1))


def refine_corners(g, pts, win=4, iterations=12, return_converged=False):
    """cornerSubPix-style refinement: the corner q satisfies sum_p w (grad grad^T)(p - q) = 0 over a window.
    A point converges when an update moves it by < 0.005 px without leaving its window."""
    gs = _gauss(g, 0.8)
    gy, gx = np.gradient(gs)
    H, W = g.shape
    out, conv = [], []
    yy, xx = np.mgrid[-win:win + 1, -win:win + 1]
    wgt = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * (0.6 * win) ** 2))
    for u, v in np.asarray(pts, float).reshape(-1, 2):
        ok = False
        for _ in range(iterations):
            u0, v0 = int(round(u)), int(round(v))
            if u0 - win < 1 or v0 - win < 1 or u0 + win >= W - 1 or v0 + win >= H - 1:
                break
            Gx = gx[v0 - win:v0 + win + 1, u0 - win:u0 + win + 1] * np.sqrt(wgt)
            Gy = gy[v0 - win:v0 + win + 1, u0 - win:u0 + win + 1] * np.sqrt(wgt)
            X, Y = xx + u0, yy + v0
            A = np.array([[np.sum(Gx * Gx), np.sum(Gx * Gy)], [np.sum(Gx * Gy), np.sum(Gy * Gy)]])
            b = np.array([np.sum(Gx * Gx * X + Gx * Gy * Y), np.sum(Gx * Gy * X + Gy * Gy * Y)])
            if abs(np.linalg.det(A)) < 1e-12:
                break
            q = np.linalg.solve(A, b)
            moved = np.hypot(q[0] - u, q[1] - v)
            if moved > win:
                break
            u, v = float(q[0]), float(q[1])
            if moved < 0.005:
                ok = True
                break
        out.append((u, v))
        conv.append(ok)
    pts_out = np.array(out, float).reshape(-1, 2)
    return (pts_out, np.array(conv, bool)) if return_converged else pts_out


def detect_x_corners(g, d=2, rel_threshold=0.3, min_dist=None, mask=None, refine=True) -> np.ndarray:
    """Checkerboard X-corner candidates WITHOUT ground truth: local maxima of the (box-smoothed) ChESS response above
    rel_threshold x its maximum, non-maximum suppressed within min_dist, then refined to sub-pixel (refine_corners)."""
    R = _box(saddle_response(g, d), 1) / 9.0
    if mask is not None:
        R = R * mask
    thr = rel_threshold * R.max(initial=0.0)
    if thr <= 0:
        return np.zeros((0, 2))
    min_dist = max(2, 2 * d) if min_dist is None else min_dist
    H, W = R.shape
    cand = np.argwhere(R >= thr)
    order = np.argsort(-R[cand[:, 0], cand[:, 1]], kind="stable")
    taken = np.zeros_like(R, bool)
    pts = []
    for v, u in cand[order]:
        if taken[v, u]:
            continue
        taken[max(0, v - min_dist):v + min_dist + 1, max(0, u - min_dist):u + min_dist + 1] = True
        pts.append((float(u), float(v)))
    pts = np.array(pts, float).reshape(-1, 2)
    if not (refine and len(pts)):
        return pts
    ref, conv = refine_corners(g, pts, win=max(2, d), return_converged=True)
    keep = []                                   # drop non-converged candidates; merge those that converged onto one corner
    for p_ in ref[conv]:
        if all(np.hypot(*(p_ - q)) > 1.0 for q in keep):
            keep.append(p_)
    return np.array(keep, float).reshape(-1, 2)


def match_points(detected, expected, tol_px=1.5):
    det = np.asarray(detected, float).reshape(-1, 2)
    exp = np.asarray(expected, float).reshape(-1, 2)
    if len(det) == 0:
        return np.zeros(0, int), np.zeros(0), np.arange(0)
    D = np.linalg.norm(exp[:, None, :] - det[None, :, :], axis=2)
    j = D.argmin(axis=1)
    e = D[np.arange(len(exp)), j]
    ok = e <= tol_px
    used = set(j[ok].tolist())
    return np.nonzero(ok)[0], e[ok], np.array([k for k in range(len(det)) if k not in used], int)


def profile(g, p0, p1, n):
    t = np.linspace(0.0, 1.0, n)
    u = p0[0] + (p1[0] - p0[0]) * t
    v = p0[1] + (p1[1] - p0[1]) * t
    H, W = g.shape
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - u0, v - v0
    return (g[v0, u0] * (1 - fu) * (1 - fv) + g[v0, u0 + 1] * fu * (1 - fv) + g[v0 + 1, u0] * (1 - fu) * fv + g[v0 + 1, u0 + 1] * fu * fv)


def dark_runs(values, rel=0.5):
    """Centres (sample index, sub-sample) of runs darker than lo + rel (hi - lo), darkness-weighted."""
    x = np.asarray(values, float)
    lo, hi = np.percentile(x, 2), np.percentile(x, 98)
    thr = lo + rel * (hi - lo)
    dark = x < thr
    out, i, n = [], 0, len(x)
    while i < n:
        if dark[i]:
            j = i
            while j + 1 < n and dark[j + 1]:
                j += 1
            w = thr - x[i:j + 1]
            out.append(float((np.arange(i, j + 1) * w).sum() / w.sum()))
            i = j + 1
        else:
            i += 1
    return np.array(out)


def ref_pixels(seg, labels) -> Dict[str, int]:
    seg = np.asarray(seg)
    ids, counts = np.unique(seg[seg >= 0], return_counts=True)
    out: Dict[str, int] = {}
    for i, c in zip(ids.tolist(), counts.tolist()):
        ref = (labels.get(i) or labels.get(str(i)) or {}).get("scale_ref")
        if ref:
            out[ref] = out.get(ref, 0) + int(c)
    return out


def measure_checkerboard(rgb, camera, ref, tol_px=1.5, d=None) -> Dict[str, float]:
    """Detect X-corners WITHOUT ground truth, then compare with the exact projection of the board's inner corners."""
    from .scale_refs import features
    f = features(ref)
    uv, z = camera.project(f["inner_corners"])
    inside = (z > 0) & (uv[:, 0] >= 3) & (uv[:, 1] >= 3) & (uv[:, 0] <= camera.width - 4) & (uv[:, 1] <= camera.height - 4)
    bc, _ = camera.project(f["board_corners"])
    g = gray(rgb)
    if d is None:   # window ~ a third of the projected square
        grid = uv.reshape(f["inner_corners_shape"][0], f["inner_corners_shape"][1], 2)   # foreshortening: use the smaller spacing
        sq = min(np.median(np.linalg.norm(np.diff(grid, axis=1), axis=2)), np.median(np.linalg.norm(np.diff(grid, axis=0), axis=2)))
        d = int(np.clip(round(sq / 3.5), 2, 8))
    mask = _polygon_mask(bc, g.shape)
    det = detect_x_corners(g, d=d, mask=mask)
    idx, err, extra = match_points(det, uv[inside], tol_px)
    return {"n_expected": int(inside.sum()), "n_matched": int(len(idx)), "n_extra_detections": int(len(extra)),
            "rms_px": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
            "max_px": float(err.max()) if len(err) else float("nan"), "window_d": int(d)}


def _polygon_mask(poly, shape):
    """1.0 inside the convex polygon `poly` (K, 2) in pixel coordinates (pixel centres), else 0.0."""
    H, W = shape
    vv, uu = np.mgrid[0:H, 0:W].astype(np.float64)
    P = np.asarray(poly, float)
    area2 = float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))
    s = 1.0 if area2 >= 0 else -1.0
    inside = np.ones((H, W), bool)
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        c = (b[0] - a[0]) * (vv - a[1]) - (b[1] - a[1]) * (uu - a[0])
        inside &= (c * s) >= 0
    return inside.astype(float)


def _sample_uv(g, uv):
    H, W = g.shape
    u = np.clip(uv[:, 0], 0, W - 1.001)
    v = np.clip(uv[:, 1], 0, H - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = u - u0, v - v0
    return (g[v0, u0] * (1 - fu) * (1 - fv) + g[v0, u0 + 1] * fu * (1 - fv) + g[v0 + 1, u0] * (1 - fu) * fv + g[v0 + 1, u0 + 1] * fu * fv)


def measure_ruler(rgb, camera, ref, samples_per_mm=24, tol_mm=0.3) -> Dict[str, float]:
    """Sample the photo along the PROJECTED 1 mm tick band (a straight 3D line through the tick midpoints, projected with
    the camera incl. distortion, so the path follows the curved image of the ruler), locate the dark ticks and compare
    their positions (in mm along the scale) with the integer millimetres."""
    from .scale_refs import features
    f = features(ref)
    mids = np.asarray(f["tick_midpoints_1mm"], float)
    n_mm = len(mids) - 1
    t = np.arange(n_mm * samples_per_mm + 1) / float(samples_per_mm)          # mm along the scale
    k = np.minimum(np.floor(t).astype(int), n_mm - 1)
    P = mids[k] + (t - k)[:, None] * (mids[k + 1] - mids[k])
    uv, z = camera.project(P)
    ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] <= camera.width - 1) & (uv[:, 1] <= camera.height - 1)
    g = gray(rgb)
    vals = _sample_uv(g, uv)
    vals[~ok] = np.nan
    good = np.nan_to_num(vals, nan=float(np.nanmax(vals)) if np.isfinite(vals).any() else 1.0)
    runs_mm = dark_runs(good, rel=0.5) / float(samples_per_mm)
    expected = np.arange(n_mm + 1, dtype=float)
    uvt, zt = camera.project(mids)
    vis = (zt > 0) & (uvt[:, 0] >= 0) & (uvt[:, 1] >= 0) & (uvt[:, 0] <= camera.width - 1) & (uvt[:, 1] <= camera.height - 1)
    idx, err, extra = match_points(np.c_[runs_mm, np.zeros_like(runs_mm)], np.c_[expected[vis], np.zeros(int(vis.sum()))], tol_mm)
    spacing_px = float(np.median(np.linalg.norm(np.diff(uvt, axis=0), axis=1)))
    return {"n_ticks_expected": int(vis.sum()), "n_ticks_found": int(len(idx)), "n_extra_runs": int(len(extra)),
            "rms_mm": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
            "rms_px": float(np.sqrt(np.mean(err ** 2)) * spacing_px) if len(err) else float("nan"), "spacing_px": spacing_px}


def edge_agreement(seg, rgb, rel_threshold=0.08, radius=1) -> float:
    """Fraction of segmentation boundary pixels (between two different ids) that have a strong image gradient within
    `radius` px: the geometry-preservation measure a neural pass must keep relative to its base image
    (neural_backend.py); boundaries between two geoms of the same colour have no edge, so compare, don't threshold."""
    s = np.asarray(seg)
    b = np.zeros(s.shape, bool)
    b[:, 1:] |= s[:, 1:] != s[:, :-1]
    b[1:, :] |= s[1:, :] != s[:-1, :]
    g = gray(rgb)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    strong = np.hypot(gx, gy) > rel_threshold
    if radius > 0:
        acc = np.zeros_like(strong)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                acc |= _shift(strong.astype(float), dy, dx) > 0
        strong = acc
    n = int(b.sum())
    return float((b & strong).sum() / n) if n else 1.0


def measure_grid_lines(rgb, camera, ref, samples_per_mm=24, tol_mm=None) -> Dict[str, float]:
    """Cutting mat: sample the photo along the projected line half-way between the two central horizontal grid lines and
    locate the (bright) vertical lines it crosses; compare with the exact grid positions."""
    from .scale_refs import _params, features, to_world
    p = _params(ref)
    f = features(ref)
    xs = np.asarray(f["grid_x_mm"], float)
    ys = np.asarray(f["grid_y_mm"], float)
    yc = 0.5 * (ys[len(ys) // 2] + ys[len(ys) // 2 - 1]) if len(ys) > 1 else ys[0] + p["grid_mm"] / 2
    z = (p["thickness_mm"] + 0.05) * 1e-3
    t = np.arange(int(round((xs[-1] - xs[0]) * samples_per_mm)) + 1) / float(samples_per_mm) + xs[0]
    P = to_world(ref, np.c_[t * 1e-3, np.full_like(t, yc * 1e-3), np.full_like(t, z)])
    uv, zz = camera.project(P)
    ok = (zz > 0) & (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] <= camera.width - 1) & (uv[:, 1] <= camera.height - 1)
    g = gray(rgb)
    vals = _sample_uv(g, uv)
    vals[~ok] = np.nan
    inv = -np.nan_to_num(vals, nan=float(np.nanmin(vals)) if np.isfinite(vals).any() else 0.0)   # bright lines -> dark runs
    runs_mm = dark_runs(inv, rel=0.5) / float(samples_per_mm) + xs[0]
    Pl = to_world(ref, np.c_[xs * 1e-3, np.full_like(xs, yc * 1e-3), np.full_like(xs, z)])
    uvl, zl = camera.project(Pl)
    vis = (zl > 0) & (uvl[:, 0] >= 0) & (uvl[:, 1] >= 0) & (uvl[:, 0] <= camera.width - 1) & (uvl[:, 1] <= camera.height - 1)
    tol = tol_mm if tol_mm is not None else 0.2 * p["grid_mm"]
    idx, err, extra = match_points(np.c_[runs_mm, np.zeros_like(runs_mm)], np.c_[xs[vis], np.zeros(int(vis.sum()))], tol)
    spacing_px = float(np.median(np.linalg.norm(np.diff(uvl, axis=0), axis=1)))
    return {"n_lines_expected": int(vis.sum()), "n_lines_found": int(len(idx)), "n_extra_runs": int(len(extra)),
            "rms_mm": float(np.sqrt(np.mean(err ** 2))) if len(err) else float("nan"),
            "rms_px": float(np.sqrt(np.mean(err ** 2)) * spacing_px / p["grid_mm"]) if len(err) else float("nan"), "spacing_px": spacing_px}
