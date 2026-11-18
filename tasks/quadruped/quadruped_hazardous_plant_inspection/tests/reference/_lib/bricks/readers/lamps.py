"""Reference reader of the indicator-lamp panel (bricks.targets.lamps).

Moved here from bricks.targets.lamps on 2026-11-14, unchanged: find the row of equal, collinear dark
bezels and judge each lens ON / OFF by its brightness against the panel around it.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .. import vision

__all__ = ["read"]


def _dark_circles(gray):
    """Dark round blobs (bezels) as ellipses [(ell, score)]."""
    H, W = gray.shape
    g = cv2.GaussianBlur(gray, (0, 0), 0.8)
    loc = cv2.blur(g, (max(15, W // 12), max(15, W // 12)))
    out = []
    for f in (0.45, 0.6):
        m = (g < f * np.maximum(loc, 0.02)).astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in cnts:
            if len(c) < 8:
                continue
            area = cv2.contourArea(c)
            if area < 12 or area > 0.05 * H * W:
                continue
            (ex, ey), (ew, eh), ang = cv2.fitEllipse(c)
            a, b = max(ew, eh) / 2, min(ew, eh) / 2
            if b < 2.0 or b / a < 0.25:
                continue
            fill = area / (math.pi * a * b)
            if not 0.75 < fill < 1.12:
                continue
            out.append(((ex, ey, a, b, math.radians(ang) + (0 if ew >= eh else math.pi / 2)), fill))
    # de-duplicate
    res = []
    for e, s in sorted(out, key=lambda t: -t[1]):
        if all(math.hypot(e[0] - f[0], e[1] - f[1]) > 0.5 * f[3] for f, _ in res):
            res.append((e, s))
    return res


def _pick_row(cands, n, W, H):
    """The n candidates of similar size on one line nearest the image centre, ordered left to right."""
    if len(cands) < n:
        return None
    best, best_s = None, -1.0
    # keep the 28 candidates nearest the image centre (bounded cost)
    E = sorted((c[0] for c in cands), key=lambda e: math.hypot(e[0] - W / 2, e[1] - H / 2))[:28]
    for i in range(len(E)):
        for j in range(len(E)):
            if i == j:
                continue
            a, b = E[i], E[j]
            if abs(a[2] - b[2]) > 0.3 * max(a[2], b[2]):
                continue
            p0 = np.array(a[:2])
            d = np.array(b[:2]) - p0
            L = np.linalg.norm(d)
            if L < 1e-6:
                continue
            d /= L
            nrm = np.array([-d[1], d[0]])
            inl = []
            for k, e in enumerate(E):
                if abs(e[2] - a[2]) > 0.3 * a[2]:
                    continue
                if abs((np.array(e[:2]) - p0) @ nrm) < 0.5 * a[3]:
                    inl.append(k)
            if len(inl) < n:
                continue
            t = np.array([(np.array(E[k][:2]) - p0) @ d for k in inl])
            order = np.argsort(t)
            inl = [inl[k] for k in order]
            t = t[order]
            # the window of n consecutive lamps closest to the image centre
            for s in range(0, len(inl) - n + 1):
                win = inl[s:s + n]
                cx = np.mean([E[k][0] for k in win])
                cy = np.mean([E[k][1] for k in win])
                gaps = np.diff(t[s:s + n])
                reg = 1.0 if n < 3 else float(np.exp(-(np.std(gaps) / max(np.mean(gaps), 1e-6) / 0.15) ** 2))
                sc = reg * vision.center_weight(cx, cy, W, H, 0.4)
                if sc > best_s:
                    best_s, best = sc, win
    if best is None:
        return None
    row = [E[k] for k in best]
    row.sort(key=lambda e: e[0])
    return row


def read(rgb: np.ndarray, hint: dict):
    """States of the n lamps left to right. hint: {'n_lamps': n}. Returns (list of bool, confidence, debug)."""
    n = int(hint["n_lamps"])
    gray = vision.gray01(rgb)
    _, s, v = vision.hsv01(rgb)
    H, W = gray.shape
    row = _pick_row(_dark_circles(gray), n, W, H)
    if row is None:
        return None, 0.0, {"reason": "lamps not found"}
    yy, xx = np.mgrid[0:H, 0:W]
    states, feats = [], []
    for (cx, cy, a, b, ang) in row:
        ca, sa = math.cos(ang), math.sin(ang)
        u = ((xx - cx) * ca + (yy - cy) * sa) / a
        w = (-(xx - cx) * sa + (yy - cy) * ca) / b
        r = np.sqrt(u * u + w * w)
        lens = r < 0.8
        ring = (r > 1.5) & (r < 2.1)
        if lens.sum() < 2:
            return None, 0.0, {"reason": "lamps too small"}
        # the lens is the brighter part of the dark bezel blob (at oblique views the blob includes the bezel wall)
        vl = float(np.percentile(v[lens], 75))
        vp = float(np.median(v[ring])) if ring.any() else 0.5
        feats.append({"v_lens": vl, "v_panel": vp, "s_lens": float(np.median(s[lens])), "px": int(lens.sum())})
    for f in feats:
        on = f["v_lens"] > 0.88 or f["v_lens"] > 1.25 * f["v_panel"]
        states.append(bool(on))
    # 2026-11-14: the margin of the decision rule above. A lamp is ON when either test passes,
    # so the decision's margin is that of the stronger test, max(s_abs, s_rel), on either side; before, the margin
    # was the smaller of the two |margins|, which is about 0 for every lit lens on a bright panel (v_lens 1.0 against
    # a panel of 0.8 is exactly the ratio 1.25): right readings of the hall's low panel had confidence 0-0.36
    margins = [abs(max((f["v_lens"] - 0.88) / 0.1, (f["v_lens"] / max(f["v_panel"], 1e-3) - 1.25) / 0.3))
               for f in feats]
    conf = float(np.clip(min(margins), 0, 1)) * float(np.clip(min(f["px"] for f in feats) / 12.0, 0, 1))
    return states, conf, {"lamps": [list(map(float, e)) for e in row], "features": feats}
