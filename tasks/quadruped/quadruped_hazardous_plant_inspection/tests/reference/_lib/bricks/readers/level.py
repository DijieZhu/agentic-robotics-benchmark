"""Reference reader of the tubular level gauge (bricks.targets.level), v3 of 2026-11-14.

Find the bright scale plate (plain thresholds and an illumination-flattened copy of the image), check every candidate
against the catalogue's scale (the whole 21-tick comb, majors every 10 %, the scale span over the plate width), take
the 0 % and 100 % rows from the comb, and scan the tube strip to the plate's left for the liquid / empty step.
"""

from __future__ import annotations

import math

import numpy as np

from .. import vision
from ..targets.level import PLATE_GAP, PLATE_W, TUBE_R  # noqa: F401  (geometry of the instrument, catalogue values)

__all__ = ["N_TICKS", "MAJOR_EVERY", "TICK_BAND", "MAJOR_BAND", "LEVEL_HI_FRACS", "V_MARGIN", "SPAN_RATIO_LIMITS",
           "MIN_COMB_MATCHED", "REFINE_MAX", "read"]


# --------------------------------------------------------------------------------------------- reader (v3)
# The scale plate as the catalogue prints it: 21 ticks every 5 % (0 % at the bottom), majors every 10 % reaching
# 0.45 of the plate width (minors 0.28), both end ticks majors, 25 mm of plate beyond each end tick.
N_TICKS = 21
MAJOR_EVERY = 2
TICK_BAND = (0.03, 0.26)          # plate-width fractions: the tick roots (minor and major ticks)
MAJOR_BAND = (0.32, 0.43)         # only the majors reach here
LEVEL_HI_FRACS = (0.86, 0.92)     # extra plate thresholds (fractions of the bright percentile): a plate under a
                                  # lighting gradient is cut by one threshold and merges with the wall at another
V_MARGIN = 0.8                    # the rectified search extends this far (in quad heights) above and below a quad:
                                  # a plate found only in part (a threshold through a lighting gradient) still shows
                                  # its whole scale
SPAN_RATIO_LIMITS = (0.55, 1.8)   # measured / catalogue ratio of (scale span / plate width) in the image
MIN_COMB_MATCHED = 0.80
REFINE_MAX = 8                    # candidates re-tried with their long sides re-found (vision.refine_plate_sides)
MAJOR_LEN_FRAC = 0.45             # the catalogue's major ticks: from the plate's left edge to 0.45 of its width
MAX_COMB_RESID = 0.25             # rms residual of the comb's fit, in pitches (comb quality is 0 beyond it)


def _plate_comb(rect_g, x_pl, RW, y0, RHq, window_m, plate_h_m):
    """The scale's tick comb in a rectified plate frame (plate columns x_pl .. x_pl + RW, plate rows of the quad
    y0 .. y0 + RHq, rows beyond them included). Returns (comb or None, reason)."""
    t0, t1 = x_pl + max(1, int(round(TICK_BAND[0] * RW))), x_pl + int(round(TICK_BAND[1] * RW))
    m0, m1 = x_pl + int(round(MAJOR_BAND[0] * RW)), x_pl + int(round(MAJOR_BAND[1] * RW))
    pitch_nom = RHq * (window_m / plate_h_m) / (N_TICKS - 1)
    win = max(5, int(round(0.45 * pitch_nom)))
    tick = vision.highpass_1d(rect_g[:, t0:t1].mean(1), win)
    major = vision.highpass_1d(rect_g[:, m0:m1].mean(1), win)
    comb = vision.scale_comb(tick, N_TICKS, MAJOR_EVERY, major, min_pitch=max(3.0, 0.55 * pitch_nom),
                             max_pitch=2.4 * pitch_nom)
    if comb is None:
        return None, "no complete scale (21 ticks)"
    if not comb["major_ok"]:
        return None, "tick pattern is not the catalogue's (majors every 10 %)"
    if comb["matched"] < MIN_COMB_MATCHED:
        return None, "too few ticks matched"
    if comb["resid"] > MAX_COMB_RESID:
        # 2026-11-14: the ticks found are not on one straight scale (genuine combs fit within 0.01 pitch;
        # the three confident misreads of a fresh sample fitted at 0.7-1.3 pitches)
        return None, "ticks off a regular scale (residual %.2f pitch)" % comb["resid"]
    return comb, ""


def _span_ratio(quad, comb, ml, mv, RW, RHq, window_m, plate_w_m):
    """Catalogue check in the image: (scale span / plate width) measured at the plate, over its catalogue value."""
    Hinv = np.linalg.inv(_rect_homography(quad, ml, mv, RW, RHq))
    x_pl = RW * ml
    xc = x_pl + 0.5 * RW

    def img(x, y):
        p = Hinv @ np.array([x, y, 1.0])
        return p[:2] / p[2]

    r0, r1 = comb["rows"][0], comb["rows"][-1]
    rm = 0.5 * (r0 + r1)
    span = float(np.linalg.norm(img(xc, r0) - img(xc, r1)))
    width = float(np.linalg.norm(img(x_pl, rm) - img(x_pl + RW, rm)))
    return (span / max(width, 1e-6)) / (window_m / plate_w_m)


def _rect_homography(quad, ml, mv, RW, RHq):
    """The homography of vision.warp_quad for a quad warped with margins (ml left, mv top and bottom)."""
    import cv2
    dst = np.float32([[ml * RW, mv * RHq], [ml * RW + RW, mv * RHq], [ml * RW + RW, mv * RHq + RHq],
                      [ml * RW, mv * RHq + RHq]])
    return cv2.getPerspectiveTransform(np.float32(quad), dst)


def read(rgb: np.ndarray, hint: dict):
    """Level in [0, 1]. Returns (level, confidence, debug).

    v3 (2026-11-14): every plate candidate is checked against the catalogue's scale before it is read: the whole
    comb of 21 ticks (majors every 10 %, both end ticks majors) must be found in a search that extends beyond the
    candidate (a plate found in part still shows its whole scale), and the scale span over the plate width in the
    image must be the catalogue's within the perspective of the allowed views. The 0 % and 100 % rows are the
    comb's end ticks (not the extreme peaks found), and the liquid row is mapped through the comb's projective fit.
    A candidate that fails a check is not read (debug 'rejected'). Before v3 the highest tick found was taken for
    100 %: a plate cut at 80 % by a threshold through a lighting gradient read 88 % for a true 71 % (ladder site 32)."""
    gray = vision.gray01(rgb)
    quads = vision.find_bright_quads(gray, min_area_px=60, aspect=(0.03, 0.4), max_candidates=10,
                                     hi_fracs=LEVEL_HI_FRACS)
    # more candidates from the image with its light flattened (a plate under a gradient merges with the tube's
    # backing at one end at every plain threshold); every candidate is checked against the catalogue below
    quads = quads + vision.find_bright_quads(vision.flatten_plate_image(gray), min_area_px=60, aspect=(0.03, 0.4),
                                             max_candidates=6, hi_fracs=LEVEL_HI_FRACS)
    if not quads:
        return None, 0.0, {"reason": "no scale plate"}
    hh, ss, vv = vision.hsv01(rgb)
    best, rejected = _read_quads(gray, ss, vv, quads, hint)
    if best is None:
        # 2026-11-14: no candidate passed; the same candidates with their long sides re-found in their
        # rectified frame (a quad that took the tube's light-grey backing into the plate, or a side tilted by
        # something touching the plate's edge, puts the tick band beside the tick roots); same checks
        refined = []
        for quad, sc in quads[:REFINE_MAX]:
            rq = vision.refine_plate_sides(gray, quad)
            if rq is not None and float(np.max(np.linalg.norm(rq - np.asarray(quad, np.float32), axis=1))) > 1.5:
                refined.append((rq, sc))
        if refined:
            best, rej2 = _read_quads(gray, ss, vv, refined, hint, strict=True)
            rejected = rejected + [dict(r, refined=True) for r in rej2]
            if best is not None:
                best[3]["refined"] = True
    if best is None:
        # 2026-11-14: a plate as bright as the wall behind it gives no bright quad at all; candidates from
        # the printed ticks (vision.tick_plate_quads: a line of tick roots, the plate's width from the majors'
        # length), same checks
        tq = vision.tick_plate_quads(gray, major_frac=MAJOR_LEN_FRAC, end_margin=0.025 / L_of(hint))
        if tq:
            best, rej3 = _read_quads(gray, ss, vv, tq, hint, strict=True)
            rejected = rejected + [dict(r, from_ticks=True) for r in rej3]
            if best is not None:
                best[3]["from_ticks"] = True
    if best is None:
        return None, 0.0, {"reason": "no readable plate", "rejected": rejected}
    best[3]["rejected"] = rejected
    return float(np.clip(best[1], 0.0, 1.0)), best[2], best[3]


def L_of(hint):
    return float(hint.get("window_m", 0.5))


def _read_quads(gray, ss, vv, quads, hint, strict=False):
    """Best (score, level, confidence, debug) over the plate candidates that pass the catalogue checks, or None; and
    the rejected candidates with the reason. strict (the refined and tick-root candidates, whose width is
    an estimate, so the tube's search band may miss the tube): the liquid must be seen and its top must lie inside
    the scale (an empty-tube or a 0 % / 100 % reading from such a candidate is refused)."""
    best = None
    rejected = []
    Himg, Wimg = gray.shape
    L = float(hint.get("window_m", 0.5))
    pw_m, ph_m = hint.get("plate_size_m", (PLATE_W, L + 0.05))
    a_exp = pw_m / ph_m
    RHq, RW, ml, mv = 400, 40, 1.6, V_MARGIN
    W_tot, H_tot = int(RW * (1 + ml)), int(round(RHq * (1 + 2 * mv)))
    y0 = mv * RHq
    for quad, sc in quads:
        qd = np.asarray(quad)
        wq = 0.5 * (np.linalg.norm(qd[1] - qd[0]) + np.linalg.norm(qd[2] - qd[3]))
        hq = 0.5 * (np.linalg.norm(qd[3] - qd[0]) + np.linalg.norm(qd[2] - qd[1]))
        r = (wq / max(hq, 1e-6)) / a_exp
        if (qd[:, 1].min() < 2 or qd[:, 1].max() > Himg - 3 or qd[:, 0].min() < 2 or qd[:, 0].max() > Wimg - 3):
            rejected.append({"quad": qd.tolist(), "why": "cut by the image border"})
            continue                       # the plate is cut by the image border: both scale ends are needed
        rect, Hm = vision.warp_quad(np.dstack([gray, ss, vv]).astype(np.float32), quad, W_tot, H_tot,
                                    margin=(ml, mv, 0.0, mv))
        g, s = rect[..., 0], rect[..., 1]
        x_pl = int(RW * ml)
        comb, why = _plate_comb(g, x_pl, RW, y0, RHq, L, float(ph_m))
        if comb is None:
            rejected.append({"quad": qd.tolist(), "why": why})
            continue
        ratio = _span_ratio(qd, comb, ml, mv, RW, RHq, L, float(pw_m))
        if not SPAN_RATIO_LIMITS[0] <= ratio <= SPAN_RATIO_LIMITS[1]:
            rejected.append({"quad": qd.tolist(), "why": "scale span / plate width %.2f x the catalogue's" % ratio})
            continue
        r_top, r_bot = float(comb["rows"][0]), float(comb["rows"][-1])
        if not vision.rect_points_in_image(Hm, [(x_pl, r_top), (x_pl, r_bot)], gray.shape):
            rejected.append({"quad": qd.tolist(), "why": "a scale end lies outside the image"})
            continue
        # liquid = saturated or much darker than the backing; the tube lies about tube_offset_m left of the plate's
        # left edge (datasheet); at oblique views it shifts sideways in the plate's rectified frame (it stands in
        # front of the plate plane), so search a band of columns for the strongest liquid / empty step
        off = hint.get("tube_offset_m", TUBE_R + PLATE_GAP) / pw_m * RW
        tw = max(2, int(round(0.5 * hint.get("tube_od_m", 2 * TUBE_R) / pw_m * RW)))
        a, b = int(round(r_top)), int(round(r_bot))
        n = b - a + 1
        if n < 10:
            rejected.append({"quad": qd.tolist(), "why": "scale too short"})
            continue
        x_lo, x_hi = max(0, int(x_pl - off - 2.2 * tw)), max(1, int(x_pl - off + 2.2 * tw))
        reg_s = s[a:b + 1, x_lo:x_hi]
        reg_g = g[a:b + 1, x_lo:x_hi]
        bk = float(np.percentile(g[a:b + 1, x_lo:x_pl], 97))
        # +1 where the tube shows liquid (saturated colour), -1 where it shows the bright backing (empty glass),
        # about 0 elsewhere (the wall, the gap): the tube column is the one with the strongest step
        sig = np.clip((reg_s - 0.12) / 0.18, 0, 1) - np.clip((reg_g / max(bk, 1e-3) - 0.75) / 0.15, 0, 1)
        best_col, empty_col = None, None
        for c0 in range(0, max(1, sig.shape[1] - tw + 1)):
            seg = sig[:, c0:c0 + tw].mean(1)
            cs = np.concatenate([[0.0], np.cumsum(seg)])
            score = (cs[-1] - cs) - cs          # liquid below row k, empty above it
            k = int(np.argmax(score))
            step = float(score[k]) / n
            # the bare backing beside the tube looks like an empty tube: a column only counts when it shows liquid
            if (seg > 0.4).mean() >= 0.02:
                if best_col is None or step > best_col[0]:
                    best_col = (step, k, seg, c0)
            elif empty_col is None or step > empty_col[0]:
                empty_col = (step, n, seg, c0)
        if best_col is None:
            if empty_col is None:
                rejected.append({"quad": qd.tolist(), "why": "no tube"})
                continue
            if strict:
                rejected.append({"quad": qd.tolist(), "why": "no liquid seen (estimated plate width)"})
                continue
            best_col = (0.5 * empty_col[0],) + empty_col[1:]      # no liquid anywhere: empty, lower confidence
        contrast, best_k, seg, c0 = best_col
        row = a + best_k - 0.5
        k_idx = vision.comb_index(comb, row)                  # 0 at the 100 % tick, 20 at the 0 % tick
        lev = (N_TICKS - 1 - k_idx) / (N_TICKS - 1)
        if strict and not 0.0 < lev < 1.0:
            rejected.append({"quad": qd.tolist(), "why": "liquid top at a scale end (estimated plate width)"})
            continue
        conf = float(np.clip(contrast / 0.6, 0, 1)) * (0.6 + 0.4 * comb["quality"])
        # the plate's apparent aspect: narrower is possible (oblique views), much wider is not
        prior = math.exp(-(math.log(r / 1.3) / 0.25) ** 2) if r > 1.3 else (1.0 if r > 0.3 else 0.3)
        if best is None or conf * sc * prior > best[0]:
            best = (conf * sc * prior, lev, conf,
                    {"quad": quad.tolist(), "ticks": [r_top, r_bot], "n_ticks": int(round(comb["matched"] * N_TICKS)),
                     "contrast": contrast, "row": row, "aspect_ratio": r, "tube_col": x_lo + c0,
                     "comb": {k: comb[k] for k in ("pitch", "matched", "resid", "major_sep", "quality")},
                     "span_ratio": ratio})
    return best, rejected
