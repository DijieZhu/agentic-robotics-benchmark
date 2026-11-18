"""Reference readers of the valves (bricks.targets.valve): the quarter-turn lever and the rising-stem indicator.

The lever reader moved here from bricks.targets.valve on 2026-11-14 unchanged; the indicator reader is v3 of that day
plate candidates from plain thresholds and an illumination-flattened copy of the image, each checked
against the catalogue's indicator (the 21-tick comb with majors at 0 / 25 / 50 / 75 / 100 %, the scale span over the
plate width, a pointer-sized pointer), the opening from the pointer row through the comb's projective fit.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .. import vision
from ..targets.valve import PLATE_H, PLATE_W, SCALE_SPAN  # noqa: F401  (geometry of the instrument, catalogue values)

__all__ = ["LEVER_UNVERIFIED_CONF", "LEVER_OBLIQUE_CONF", "LEVER_MAX_PIPE_TILT", "LEVER_IMAGE_MAX_CONF",
           "read_lever", "GATE_HI_FRACS", "GATE_MAX_QUADS", "GATE_RETRY_CONF", "GATE_N_TICKS", "GATE_MAJOR_EVERY",
           "GATE_TICK_BAND", "GATE_MAJOR_BAND", "GATE_ALL_FRACS", "GATE_V_MARGIN", "GATE_SPAN_RATIO_LIMITS",
           "GATE_MIN_MATCHED", "GATE_PTR_ROWS", "GATE_MAX_GAP", "GATE_END_SLACK", "read_gate", "read"]


def _lever_masks(rgb, k_max: int = 4):
    """Candidate lever-grip components (saturated, elongated), best first: [mask]."""
    h, s, v = vision.hsv01(rgb)
    m = (s > 0.42) & (v > 0.12)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    H, W = m.shape
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    cands = []
    for k in range(1, n):
        area = stats[k, cv2.CC_STAT_AREA]
        if area < 12:
            continue
        ys, xs = np.nonzero(lab == k)
        P = np.column_stack([xs, ys]).astype(np.float64)
        C = np.cov(P.T) if len(P) > 2 else np.eye(2)
        ev = np.linalg.eigvalsh(C)
        elong = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
        if elong < 2.0:
            continue
        d = math.hypot(cent[k, 0] - W / 2, cent[k, 1] - H / 2) / (0.5 * math.hypot(W, H))
        sc = area * min(elong, 12.0) * math.exp(-(d / 0.4) ** 2)
        cands.append((sc, k))
    cands.sort(key=lambda t: -t[0])
    return [lab == k for _, k in cands[:k_max]]


def _lever_mask(rgb):
    masks = _lever_masks(rgb, 1)
    return masks[0] if masks else None


LEVER_GRIP_FRAC = 0.60          # catalogue lever: the red grip covers 0.36 .. 0.96 of the lever length
LEVER_SIZE_LIMITS = (0.85, 1.40)  # measured / expected grip length in the image (v3; genuine grips 1.00-1.21)
LEVER_MAX_FROM_TAG_M = 0.40     # the grip's centre lies within this of the tag plate (in the face plane)


LEVER_SPLIT_SAT = (0.88, 0.82)   # saturations at which a merged red region is split, the highest first


def _is_cut(mask) -> bool:
    """The candidate touches the frame border (as read_lever's 'cut')."""
    ys, xs = np.nonzero(mask)
    Hh, Ww = mask.shape
    return bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= Ww - 2 or ys.max() >= Hh - 2)


def _split_grip_candidates(rgb, k_max: int = 4):
    """Lever-grip candidates at higher saturation thresholds: [(threshold, mask)]. A grip merged with a red surface
    behind it (into one region, often not elongated any more, so not a candidate of _lever_masks) separates from it
    at a threshold between their saturations (catalogue grip 0.80 / 0.08 / 0.06: saturation 0.93; a fire door's
    leaf 0.66 / 0.20 / 0.16: 0.76). Per threshold the candidates are ranked as in _lever_masks."""
    h, s, v = vision.hsv01(rgb)
    H, W = s.shape
    out = []
    for t in LEVER_SPLIT_SAT:
        m = cv2.morphologyEx(((s > t) & (v > 0.12)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        cands = []
        for k in range(1, n):
            area = stats[k, cv2.CC_STAT_AREA]
            if area < 12:
                continue
            ys, xs = np.nonzero(lab == k)
            P = np.column_stack([xs, ys]).astype(np.float64)
            ev = np.linalg.eigvalsh(np.cov(P.T)) if len(P) > 2 else np.ones(2)
            elong = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
            if elong < 2.0:
                continue
            d = math.hypot(cent[k, 0] - W / 2, cent[k, 1] - H / 2) / (0.5 * math.hypot(W, H))
            cands.append((area * min(elong, 12.0) * math.exp(-(d / 0.4) ** 2), k))
        cands.sort(key=lambda e: -e[0])
        out.extend((t, lab == k) for _, k in cands[:k_max])
    return out


def _grip_check(mask, hint, K, Rf, rgb=None, hub_s_min=0.42):
    """v3 catalogue check of a grip candidate in plane mode (needs the tag's position): the grip's length along its
    axis in the image against the catalogue's grip (LEVER_GRIP_FRAC x lever length) at the tag's distance, foreshortened
    by the lifted 3D direction; its distance from the tag; and (with rgb) the lever's hub on its axis. Returns
    (ratio or None, dist_m or None, ok)."""
    tag = hint.get("tag")
    p_tag = hint.get("tag_pos_cam")
    if p_tag is None and isinstance(tag, dict):
        p_tag = tag.get("pos_cam")
    if p_tag is None:
        return None, None, None
    p_tag = np.asarray(p_tag, float)
    ys, xs = np.nonzero(mask)
    P = np.column_stack([xs, ys]).astype(np.float64)
    m = P.mean(0)
    ev, evec = np.linalg.eigh(np.cov(P.T))
    e = evec[:, 1]
    t = (P - m) @ e
    meas = float(t.max() - t.min())
    Ki = np.linalg.inv(K)
    r1 = Ki @ np.array([m[0], m[1], 1.0])
    r2 = Ki @ np.array([m[0] + 20 * e[0], m[1] + 20 * e[1], 1.0])
    n_lev = Rf[:, 0]
    ell = np.cross(n_lev, np.cross(r1, r2))
    if np.linalg.norm(ell) < 1e-9:
        return None, None, False
    ell /= np.linalg.norm(ell)
    # the grip centre in 3D: the view ray through the centroid met with the tag's plane
    ray = r1 / np.linalg.norm(r1)
    den = float(n_lev @ ray)
    if abs(den) < 1e-6:
        return None, None, False
    X = ray * float(n_lev @ p_tag) / den
    Lg = LEVER_GRIP_FRAC * float(hint.get("lever_len_m", 0.18))
    a, b = X - 0.5 * Lg * ell, X + 0.5 * Lg * ell
    if a[2] <= 1e-3 or b[2] <= 1e-3:
        return None, None, False
    ua, ub = K @ (a / a[2]), K @ (b / b[2])
    expect = float(np.hypot(ua[0] - ub[0], ua[1] - ub[1]))
    ratio = meas / max(expect, 1e-6)
    dist = float(np.linalg.norm(X - p_tag))
    ok = bool(LEVER_SIZE_LIMITS[0] <= ratio <= LEVER_SIZE_LIMITS[1] and dist <= LEVER_MAX_FROM_TAG_M)
    if ok and rgb is not None:
        # the lever's hub (same colour, at the pivot, beyond the grip's inner end on its axis): a red band on a
        # red-painted pipe has the grip's size but no hub
        hub, _ = _hub_pivot(rgb, mask, m, e, float(t.min()), float(t.max()), s_min=hub_s_min)
        ok = hub is not None
    return ratio, dist, ok


def _pipe_direction(gray, exclude_mask):
    """Dominant long straight edge direction (radians, image coordinates) away from the lever."""
    g8 = np.clip(gray * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(g8, (0, 0), 1.2), 30, 90)
    if exclude_mask is not None:
        edges[cv2.dilate(exclude_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0] = 0
    H, W = g8.shape
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=40, minLineLength=int(0.12 * min(H, W)), maxLineGap=6)
    if lines is None:
        return None
    acc = np.zeros(180)
    for x0, y0, x1, y1 in np.asarray(lines).reshape(-1, 4):
        ln = math.hypot(x1 - x0, y1 - y0)
        a = int(round(math.degrees(math.atan2(y1 - y0, x1 - x0)))) % 180
        acc[a] += ln
    acc = vision.circ_smooth(acc, 2)
    return math.radians(float(np.argmax(acc)))


LEVER_UNVERIFIED_CONF = 0.2     # confidence cap of an image-only reading whose pipe was not found on both sides
LEVER_OBLIQUE_CONF = 0.3        # cap when the view looks oblique (image-only) or the lever is cut by the frame
LEVER_MAX_PIPE_TILT = 20.0      # deg: an upright camera sees a 'y' pipe within this of horizontal ('z': vertical)
LEVER_IMAGE_MAX_CONF = 0.4      # cap of every image-only reading: foreshortening is not observable without the plate


def _hub_pivot(rgb, mask, m, e, t_lo, t_hi, s_min=0.42):
    """The lever's pivot in the image: the hub (a smaller blob of the grip's colour on the lever line, beyond one
    end of the grip, about 0.6 grip lengths away). Returns (pivot xy or None, candidates [xy, xy])."""
    ext = t_hi - t_lo
    cands = [m + e * (t_lo - 0.6 * ext), m + e * (t_hi + 0.6 * ext)]
    h, s, v = vision.hsv01(rgb)
    hue0 = float(np.median(h[mask]))
    dh = np.abs(((h - hue0) + 180.0) % 360.0 - 180.0)
    sat = ((s > s_min) & (v > 0.12) & (dh < 20.0) & ~mask).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(sat, connectivity=8)
    nrm = np.array([-e[1], e[0]])
    best, best_d = None, None
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < max(4, 0.03 * mask.sum()):
            continue
        c = cent[k]
        for cc in cands:
            dd = float(np.linalg.norm(c - cc))
            if dd < 0.35 * ext and abs(float((c - m) @ nrm)) < 0.35 * ext and (best_d is None or dd < best_d):
                best, best_d = np.asarray(c, float), dd
    return best, cands


def _pipe_both_sides(rgb, mask, m, e, hint):
    """Find the pipe near the lever: straight edges of one direction in a band through the lever's pivot that extend
    beyond the valve body on BOTH sides along that direction (the pipe enters and leaves the valve). Returns
    (pipe direction in radians (image), info dict) or (None, info)."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    P = np.column_stack([xs, ys]).astype(np.float64)
    t = (P - m) @ e
    t_lo, t_hi = float(t.min()), float(t.max())
    ext = max(t_hi - t_lo, 2.0)
    Lpx = ext / 0.6                                  # the grip spans 0.36 .. 0.96 of the lever length
    od_rel = float(hint.get("pipe_od_m", 0.048)) / max(float(hint.get("lever_len_m", 0.18)), 1e-3)
    gap = max(0.25, 1.1 * od_rel) * Lpx              # valve body + flanges hide the pipe within this of the pivot
    band = 0.6 * Lpx                                 # pipe radius + parallax of the lever plane in front of the pipe
    need = 0.3 * Lpx                                 # visible pipe edge length required on each side
    info = {"L_px": Lpx}
    pivot, cands = _hub_pivot(rgb, mask, m, e, t_lo, t_hi)
    pivots = [pivot] if pivot is not None else cands
    info["hub"] = pivot is not None
    # edges in a region around the lever (the lever itself masked out)
    r = 2.6 * Lpx
    cx0 = int(max(0, min(p[0] for p in pivots + [m]) - r))
    cx1 = int(min(W, max(p[0] for p in pivots + [m]) + r))
    cy0 = int(max(0, min(p[1] for p in pivots + [m]) - r))
    cy1 = int(min(H, max(p[1] for p in pivots + [m]) + r))
    if cx1 - cx0 < 8 or cy1 - cy0 < 8:
        return None, info
    g8 = np.clip(vision.gray01(rgb[cy0:cy1, cx0:cx1]) * 255, 0, 255).astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(g8, (0, 0), 1.2), 30, 90)
    k = max(5, int(round(0.04 * Lpx)) | 1)
    edges[cv2.dilate(mask[cy0:cy1, cx0:cx1].astype(np.uint8), np.ones((k, k), np.uint8)) > 0] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, threshold=max(8, int(0.15 * Lpx)),
                            minLineLength=max(8, int(0.2 * Lpx)), maxLineGap=max(3, int(0.03 * Lpx)))
    if lines is None:
        return None, info
    seg = np.asarray(lines, np.float64).reshape(-1, 4) + np.array([cx0, cy0, cx0, cy0], np.float64)
    ang = np.degrees(np.arctan2(seg[:, 3] - seg[:, 1], seg[:, 2] - seg[:, 0])) % 180.0
    seg_len = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    best = (0.0, None, None)
    for c in pivots:
        mid = 0.5 * (seg[:, :2] + seg[:, 2:])
        near = np.linalg.norm(mid - c, axis=1) < r
        # the lever's steel bar runs from the pivot to the grip: its edges are not the pipe (with a strap or a
        # bracket on the other side of the pivot they would fake a pipe across the real one)
        far_end = m + e * (t_hi if np.linalg.norm(m + e * t_hi - c) > np.linalg.norm(m + e * t_lo - c) else t_lo)
        ab = far_end - c
        lab = max(float(np.linalg.norm(ab)), 1e-6)
        ua = ab / lab
        na = np.array([-ua[1], ua[0]])
        on_bar = np.ones(len(seg), bool)
        for pts in (seg[:, :2], seg[:, 2:]):
            tt = (pts - c) @ ua
            on_bar &= (tt > -0.05 * Lpx) & (tt < lab + 0.05 * Lpx) & (np.abs((pts - c) @ na) < 0.1 * Lpx)
        near &= ~on_bar
        if not near.any():
            continue
        acc = np.zeros(180)
        np.add.at(acc, np.round(ang[near]).astype(int) % 180, seg_len[near])
        acc = vision.circ_smooth(acc, 2)
        peaks = []
        for _ in range(3):
            j = int(np.argmax(acc))
            if acc[j] <= 0:
                break
            peaks.append(j)
            for dj in range(-8, 9):
                acc[(j + dj) % 180] = 0.0
        for q in peaks:
            da = np.abs(((ang - q) + 90.0) % 180.0 - 90.0)
            sel = near & (da <= 5.0)
            if not sel.any():
                continue
            u = np.array([math.cos(math.radians(q)), math.sin(math.radians(q))])
            nv = np.array([-u[1], u[0]])
            p0, p1 = seg[sel, :2] - c, seg[sel, 2:] - c
            off = np.maximum(np.abs(p0 @ nv), np.abs(p1 @ nv))
            a0, a1 = p0 @ u, p1 @ u
            lo, hi = np.minimum(a0, a1), np.maximum(a0, a1)
            inb = off <= band
            cov_p = np.clip(hi - np.maximum(lo, gap), 0, None) * inb
            cov_n = np.clip(np.minimum(hi, -gap) - lo, 0, None) * inb
            pos, neg = float(cov_p.sum()), float(cov_n.sum())
            score = min(pos, neg)
            if score > best[0]:
                best = (score, math.radians(q), {"pos_px": pos, "neg_px": neg, "pivot": c.tolist(), "q_deg": q})
    if best[1] is None or best[0] < need:
        info.update(best[2] or {})
        return None, info
    info.update(best[2])
    return best[1], info


def read_lever(rgb: np.ndarray, hint: dict):
    """Lever angle from the pipe in degrees (0 open .. 90 closed). hint: lever_hint(...) plus optionally 'K' and
    'R_cam_face' (3x3, face axes in the camera's optical frame) and, v3, 'tag' (the dict tag.read returned for the
    valve's plate, or 'tag_pos_cam'). Returns (angle_deg, confidence, debug).

    Version 2, text plates: 'plate' = the value `bricks.readers.tagtext.read` returned for the valve's
    plate (in the same or any frame of the same camera pose): its border pose gives 'R_cam_face' (the plate is
    parallel to the valve's face) and its centre 'tag_pos_cam' unless the hint already has them
    (`tagtext.lever_hint(value)` makes the same keys). The plate's centre lies 0.08-0.25 m from the grip's centre
    in the face plane (the marker's was 0.15-0.30 m), inside LEVER_MAX_FROM_TAG_M.

    v3 (2026-11-14), plane mode with the tag's position: the grip candidates are checked against the catalogue (the
    grip's image length at the tag's distance, LEVER_SIZE_LIMITS, and its distance from the tag, LEVER_MAX_FROM_TAG_M);
    and the lever's hub must lie on its axis beyond its inner end; the first that fits is read, with a confidence
    from its pixel count and the view's obliqueness only; if none fits,
    the reading's confidence is capped at LEVER_UNVERIFIED_CONF. Without the tag's position plane mode is as before.

    Without 'R_cam_face' (image-only mode) the angle is measured against the pipe found in the image, and the
    reading is never confident (< 0.5): with the pipe found on both sides of the lever (straight edges of one
    direction in a band through the lever's pivot, beyond the valve body on each side, the lever's own bar excluded)
    the confidence is at most LEVER_IMAGE_MAX_CONF (0.4); a pipe that an upright camera sees tilted by more than
    LEVER_MAX_PIPE_TILT from its nominal direction (strong perspective) caps it at LEVER_OBLIQUE_CONF (0.3); with no
    such pipe the angle against the strongest straight edge in the image is returned with at most
    LEVER_UNVERIFIED_CONF (0.2) (those readings were wrong by tens of degrees when the strongest edge was not the
    pipe, 2026-11-13 review). Even with the pipe verified, foreshortening is not observable in the image alone:
    readings were 13-80 deg off at >= 60 deg off axis or under steep elevation. Both modes: a lever grip cut by the
    frame border caps the confidence at 0.3."""
    plate = hint.get("plate")
    if isinstance(plate, dict) and plate.get("R_cam_face") is not None:
        hint = dict(hint)
        if hint.get("R_cam_face") is None:
            hint["R_cam_face"] = np.asarray(plate["R_cam_face"], float)
        if hint.get("tag_pos_cam") is None and plate.get("pos_cam") is not None:
            hint["tag_pos_cam"] = np.asarray(plate["pos_cam"], float)
    masks = _lever_masks(rgb)
    if not masks:
        return None, 0.0, {"reason": "no lever"}
    mask = masks[0]
    plane = hint.get("R_cam_face") is not None and hint.get("K") is not None
    size = None
    if plane:
        # v3: with the tag's position, take the first candidate whose size and place fit the catalogue's grip
        K_ = np.asarray(hint["K"], float)
        Rf_ = np.asarray(hint["R_cam_face"], float)
        checks = [(mk, _grip_check(mk, hint, K_, Rf_, rgb)) for mk in masks]
        fit = [(mk, c) for mk, c in checks if c[2]]
        # 2026-11-14: a candidate cut by the frame border is no grip seen whole (its length is not the
        # grip's): an uncut fit comes first
        fit.sort(key=lambda f: _is_cut(f[0]))
        if not fit or _is_cut(fit[0][0]):
            # 2026-11-14: a grip in front of a red-painted surface (a fire door's leaf or frame) merges with
            # it into one saturated region; split at higher saturations (the grip's paint is more saturated than a
            # fire door's), the highest first, and check the uncut parts, the hub at the same saturation
            for s_min, mk in _split_grip_candidates(rgb):
                if _is_cut(mk):
                    continue
                c = _grip_check(mk, hint, K_, Rf_, rgb, hub_s_min=s_min)
                if c[2]:
                    fit = [(mk, c)] + fit
                    break
        if fit:
            mask, size = fit[0]
        elif checks[0][1][2] is not None:
            size = checks[0][1]
    ys, xs = np.nonzero(mask)
    P = np.column_stack([xs, ys]).astype(np.float64)
    m = P.mean(0)
    ev, evec = np.linalg.eigh(np.cov(P.T))
    e = evec[:, 1]
    elong = math.sqrt(ev[1] / max(ev[0], 1e-9))
    Hh, Ww = mask.shape
    # a grip cut by the frame border: its visible shape (and direction) is not the lever's
    cut = bool(xs.min() <= 1 or ys.min() <= 1 or xs.max() >= Ww - 2 or ys.max() >= Hh - 2)
    dbg = {"centroid": m.tolist(), "dir_img_deg": math.degrees(math.atan2(e[1], e[0])), "elong": elong,
           "area": int(len(P)), "cut": cut}
    conf = float(np.clip((elong - 2.0) / 3.0, 0, 1) * np.clip(len(P) / 150.0, 0, 1))
    pipe_local = np.array([0.0, 1.0, 0.0]) if hint.get("pipe_axis", "y") == "y" else np.array([0.0, 0.0, 1.0])
    if hint.get("R_cam_face") is not None and hint.get("K") is not None:
        K = np.asarray(hint["K"], float)
        Ki = np.linalg.inv(K)
        r1 = Ki @ np.array([m[0], m[1], 1.0])
        r2 = Ki @ np.array([m[0] + 20 * e[0], m[1] + 20 * e[1], 1.0])
        n_img = np.cross(r1, r2)
        Rf = np.asarray(hint["R_cam_face"], float)
        n_lev = Rf[:, 0]
        ell = np.cross(n_lev, n_img)
        if np.linalg.norm(ell) < 1e-9:
            return None, 0.0, dbg
        ell /= np.linalg.norm(ell)
        p = Rf @ pipe_local
        ang = math.degrees(math.acos(min(1.0, abs(float(ell @ p)))))
        dbg["mode"] = "plane"
        # grazing views make the lift ill-conditioned: |n_lev . view ray|
        cosv = abs(float(n_lev @ (r1 / np.linalg.norm(r1))))
        if size is not None:
            # v3: the grip's size and place were checked against the catalogue: the elongation proxy goes (a grip
            # of the catalogue's shape seen in perspective measures 2.9-3.5, which capped right readings below 0.5)
            dbg.update(size_ratio=size[0], from_tag_m=size[1], size_ok=size[2])
            if size[2]:
                conf = float(np.clip(len(P) / 150.0, 0, 1))
            else:
                conf = min(conf, LEVER_UNVERIFIED_CONF)       # a blob of the wrong size or place is not the grip
        conf *= float(np.clip(cosv / 0.3, 0, 1))
        if cut:
            conf = min(conf, LEVER_OBLIQUE_CONF)
        return ang, conf, dbg
    q, pinfo = _pipe_both_sides(rgb, mask, m, e, hint)
    dbg["pipe"] = pinfo
    dbg["mode"] = "image"
    if q is not None:
        dbg["pipe_both_sides"] = True
        c_out = conf * 0.8
    else:
        dbg["pipe_both_sides"] = False
        q = _pipe_direction(vision.gray01(rgb), mask)
        if q is None:
            return None, 0.0, dbg
        c_out = min(conf * 0.8, LEVER_UNVERIFIED_CONF)
    a = math.atan2(e[1], e[0])
    d = abs(((a - q) + math.pi / 2) % math.pi - math.pi / 2)
    dbg["pipe_img_deg"] = math.degrees(q)
    if dbg["pipe_both_sides"]:
        # the image angle is only the lever's angle when the view is near frontal: an upright camera sees a 'y' pipe
        # roughly horizontal and a 'z' pipe roughly vertical; strong perspective (close, oblique, wide lens) tilts it
        expect = 0.0 if hint.get("pipe_axis", "y") == "y" else 90.0
        tilt = abs(((math.degrees(q) - expect) + 90.0) % 180.0 - 90.0)
        dbg["pipe_tilt_deg"] = tilt
        if tilt > LEVER_MAX_PIPE_TILT:
            c_out = min(c_out, LEVER_OBLIQUE_CONF)
    if dbg.get("cut"):
        c_out = min(c_out, LEVER_OBLIQUE_CONF)
    # the image angle is the lever's angle only near a frontal view, and nothing in the image alone tells how
    # oblique the view is (measured 2026-11-13: with the pipe found on both sides and upright, readings were still
    # 13-80 deg off at >= 60 deg off axis or under steep elevation): image-only readings are never confident
    c_out = min(c_out, LEVER_IMAGE_MAX_CONF)
    return math.degrees(d), c_out, dbg


GATE_HI_FRACS = (0.78, 0.86)     # extra plate thresholds (fractions of the bright percentile), see read_gate
GATE_MAX_QUADS = 6               # plate candidates tried in the second pass
GATE_RETRY_CONF = 0.5            # a first-pass reading below this confidence triggers the second pass
def _plate_ticks(rect_gray, x_frac=(0.04, 0.30)):
    """Rows of the horizontal tick marks in a rectified plate (darkness profile along the tick roots)."""
    h, w = rect_gray.shape
    x0, x1 = int(w * x_frac[0]), max(int(w * x_frac[0]) + 1, int(w * x_frac[1]))
    strip = rect_gray[:, x0:x1]
    white = float(np.percentile(rect_gray, 90))
    dk = np.clip((white - strip) / max(white, 1e-3), 0, 1).mean(1)
    dk = dk - np.median(dk)
    return dk


# The indicator plate as the catalogue prints it: 21 ticks every 5 % over SCALE_SPAN, centred on the plate (0 % at
# the bottom), majors at 0 / 25 / 50 / 75 / 100 % reaching 0.62 of the plate width (minors 0.38), the pointer (18 x
# 7 mm, orange) in front of the plate's left half.
GATE_N_TICKS = 21
GATE_MAJOR_EVERY = 5
GATE_TICK_BAND = (0.05, 0.30)        # plate-width fractions: tick roots
GATE_MAJOR_BAND = (0.44, 0.58)       # only the majors reach here
GATE_ALL_FRACS = (0.78, 0.86, 0.90, 0.94)   # v3: every candidate is checked, so all thresholds run in one pass
GATE_V_MARGIN = 0.35                 # rectified search beyond a candidate, in candidate heights (a plate found in part)
GATE_SPAN_RATIO_LIMITS = (0.55, 1.8)  # measured / catalogue ratio of (scale span / plate width) in the image
GATE_MIN_MATCHED = 0.65             # at 4-5 m the ticks are 4-5 px apart and some alias away; the pattern still holds
GATE_PTR_ROWS = (0.4, 3.0)           # pointer blob height in tick pitches (7 mm = 1.4 pitches; parallax smears it)
GATE_MAX_GAP = 4                     # ticks the pointer and its arm may hide inside the scale
GATE_POINTER_HUE = (10.0, 50.0)      # degrees: the catalogue's orange pointer (rendered about 28); red wheels, green pipes out
GATE_END_SLACK = 3                   # end ticks the pointer may hide at 0 % / 100 % (the majors anchor the scale)
GATE_REFINE_MAX = 8                  # candidates re-tried with their long sides re-found
GATE_MAJOR_LEN_FRAC = 0.62           # the catalogue's major ticks reach 0.62 of the plate width from its left edge
GATE_PTR_BEYOND = 0.75               # ticks a pointer reading may lie beyond an end tick (else refused)


def read_gate(rgb: np.ndarray, hint: dict):
    """Opening of a rising-stem gate valve in [0, 1] from its indicator. Returns (opening, confidence, debug).

    v3 (2026-11-14): plate candidates come from the default thresholds and thresholds near the plate's white
    (GATE_ALL_FRACS) in one pass, and every candidate is checked against the catalogue's indicator before it is read:
    the whole comb of 21 ticks with majors at 0 / 25 / 50 / 75 / 100 % must be found (in a search that extends beyond
    the candidate), the scale span over the plate width in the image must be the catalogue's within the perspective
    of the allowed views, and the pointer blob must be pointer-sized. The opening is the pointer row mapped through the
    comb's projective fit. A candidate that fails is not read (debug 'rejected'). Before v3 any bright quad with three
    tick-like rows was read: a wall region beside the plate (its rows from the tag's marker) read 95 % for a true 43 %
    (ladder site 22), and the plate itself, against a wall of nearly its brightness, was not found at all."""
    gray = vision.gray01(rgb)
    h_img, s_img, v_img = vision.hsv01(rgb)
    quads = vision.find_bright_quads(gray, min_area_px=60, aspect=(0.12, 0.6), max_candidates=10,
                                     hi_fracs=GATE_ALL_FRACS)
    quads = quads + vision.find_bright_quads(vision.flatten_plate_image(gray), min_area_px=60, aspect=(0.12, 0.6),
                                             max_candidates=6, hi_fracs=GATE_ALL_FRACS)
    if not quads:
        return None, 0.0, {"reason": "no plate"}
    pw_m, ph_m = hint.get("plate_size_m", (PLATE_W, PLATE_H))
    # v3: only pixels of the pointer's catalogue colour (orange) count as the pointer (a green pipe crossing below the
    # plate read as a pointer at -5 % for a true 91 %, fresh site 7001); saturation stays the old test otherwise
    hue_ok = (h_img >= GATE_POINTER_HUE[0]) & (h_img <= GATE_POINTER_HUE[1])
    best, rejected = _gate_from_quads(gray, s_img * hue_ok, v_img, quads, float(pw_m), float(ph_m))
    if best is None:
        # 2026-11-14: no candidate passed; the same candidates with their long sides re-found in their
        # rectified frame (the pointer arm crossing the plate's left edge tilts the contour's side, and the tick
        # band then leaves the tick roots at one end: views 30-45 deg off axis); same checks
        refined = []
        for quad, sc in quads[:GATE_REFINE_MAX]:
            rq = vision.refine_plate_sides(gray, quad)
            if rq is not None and float(np.max(np.linalg.norm(rq - np.asarray(quad, np.float32), axis=1))) > 1.5:
                refined.append((rq, sc))
        if refined:
            best, rej2 = _gate_from_quads(gray, s_img * hue_ok, v_img, refined, float(pw_m), float(ph_m))
            rejected = rejected + [dict(r, refined=True) for r in rej2]
            if best is not None:
                best[3]["refined"] = True
    if best is None:
        # 2026-11-14: a plate as bright as the wall behind it (seen 40-45 deg off axis) gives no bright quad;
        # candidates from the printed ticks (vision.tick_plate_quads), same checks
        tq = vision.tick_plate_quads(gray, major_frac=GATE_MAJOR_LEN_FRAC,
                                     end_margin=0.5 * (float(ph_m) - SCALE_SPAN) / SCALE_SPAN)
        if tq:
            best, rej3 = _gate_from_quads(gray, s_img * hue_ok, v_img, tq, float(pw_m), float(ph_m))
            rejected = rejected + [dict(r, from_ticks=True) for r in rej3]
            if best is not None:
                best[3]["from_ticks"] = True
    if best is None:
        return None, 0.0, {"reason": "no readable plate", "rejected": rejected}
    best[3]["rejected"] = rejected
    return float(np.clip(best[1], -0.05, 1.05)), best[2], best[3]


def _gate_from_quads(gray, s_img, v_img, quads, pw_m=PLATE_W, ph_m=PLATE_H):
    """Best (score, opening, confidence, debug) over plate candidates that pass the catalogue checks, or None; and
    the rejected candidates with the reason."""
    best = None
    rejected = []
    Himg, Wimg = gray.shape
    PW, PH, ml, mv = 52, 240, 0.4, GATE_V_MARGIN
    W_tot, H_tot = int(round(PW * (1 + ml))), int(round(PH * (1 + 2 * mv)))
    l_px = PW * ml
    span_frac = SCALE_SPAN / ph_m
    pitch_nom = PH * span_frac / (GATE_N_TICKS - 1)
    for quad, sc in quads:
        qd = np.asarray(quad, np.float64)
        # sample the plate plus a margin to the left (the pointer overlaps the plate's left edge) and above / below
        rect, Hm = vision.warp_quad(np.dstack([gray, s_img, v_img]).astype(np.float32), quad, W_tot, H_tot,
                                    margin=(ml, mv, 0.0, mv))
        g, s = rect[..., 0], rect[..., 1]
        x0 = int(round(l_px))
        t0, t1 = x0 + max(1, int(round(GATE_TICK_BAND[0] * PW))), x0 + int(round(GATE_TICK_BAND[1] * PW))
        m0, m1 = x0 + int(round(GATE_MAJOR_BAND[0] * PW)), x0 + int(round(GATE_MAJOR_BAND[1] * PW))
        win = max(5, int(round(0.45 * pitch_nom)))
        tick = vision.highpass_1d(g[:, t0:t1].mean(1), win)
        major = vision.highpass_1d(g[:, m0:m1].mean(1), win)
        comb = vision.scale_comb(tick, GATE_N_TICKS, GATE_MAJOR_EVERY, major, min_pitch=max(3.0, 0.55 * pitch_nom),
                                 max_pitch=2.4 * pitch_nom, max_gap=GATE_MAX_GAP, end_slack=GATE_END_SLACK,
                                 peak_frac=0.18)
        if comb is None:
            rejected.append({"quad": qd.tolist(), "why": "no complete scale (21 ticks)"})
            continue
        if not comb["major_ok"]:
            rejected.append({"quad": qd.tolist(), "why": "tick pattern is not the catalogue's"})
            continue
        if comb["matched"] < GATE_MIN_MATCHED:
            rejected.append({"quad": qd.tolist(), "why": "too few ticks matched"})
            continue
        r_top, r_bot = float(comb["rows"][0]), float(comb["rows"][-1])
        if not vision.rect_points_in_image(Hm, [(l_px, r_top), (l_px, r_bot)], gray.shape):
            rejected.append({"quad": qd.tolist(), "why": "a scale end lies outside the image"})
            continue
        span = r_bot - r_top
        # catalogue check in the image: scale span over plate width
        Hinv = np.linalg.inv(Hm)

        def img(x, y):
            p = Hinv @ np.array([x, y, 1.0])
            return p[:2] / p[2]

        xc, rm = l_px + 0.5 * PW, 0.5 * (r_top + r_bot)
        span_px = float(np.linalg.norm(img(xc, r_top) - img(xc, r_bot)))
        width_px = float(np.linalg.norm(img(l_px, rm) - img(l_px + PW, rm)))
        ratio = (span_px / max(width_px, 1e-6)) / (SCALE_SPAN / pw_m)
        if not GATE_SPAN_RATIO_LIMITS[0] <= ratio <= GATE_SPAN_RATIO_LIMITS[1]:
            rejected.append({"quad": qd.tolist(), "why": "scale span / plate width %.2f x the catalogue's" % ratio})
            continue
        # the pointer: the saturated blob with most pixels over the plate's left half, near the scale rows
        sat = (s > 0.45).astype(np.uint8)
        sat[:, int(l_px + 0.55 * PW):] = 0
        sat[:max(0, int(r_top - 0.08 * span))] = 0
        sat[int(r_bot + 0.08 * span) + 1:] = 0
        ncomp, lab, stats, cent = cv2.connectedComponentsWithStats(sat, connectivity=8)
        bestc, best_n = None, 0
        for kc in range(1, ncomp):
            on_plate = int((lab[:, int(l_px):int(l_px + 0.55 * PW)] == kc).sum())
            if on_plate > best_n:
                bestc, best_n = kc, on_plate
        if bestc is None or best_n < 3:
            rejected.append({"quad": qd.tolist(), "why": "no pointer"})
            continue
        ys, xs = np.nonzero(lab[:, int(l_px):int(l_px + 0.55 * PW)] == bestc)
        ptr_h = (float(ys.max()) - float(ys.min()) + 1.0) / max(comb["pitch"], 1e-6)
        if not GATE_PTR_ROWS[0] <= ptr_h <= GATE_PTR_ROWS[1]:
            rejected.append({"quad": qd.tolist(), "why": "pointer blob %.1f pitches tall" % ptr_h})
            continue
        row = float(np.mean(ys))
        # an end tick that was not seen must be under the pointer
        hidden = [k for k in (0, 1, GATE_N_TICKS - 2, GATE_N_TICKS - 1) if not comb["present"][k]]
        if any(not (ys.min() - 1.5 * comb["pitch"] <= comb["rows"][k] <= ys.max() + 1.5 * comb["pitch"])
               for k in hidden if k in (0, GATE_N_TICKS - 1)):
            rejected.append({"quad": qd.tolist(), "why": "an end tick is missing and the pointer is not over it"})
            continue
        k_idx = vision.comb_index(comb, row)                 # 0 at the 100 % tick, 20 at the 0 % tick
        opening = (GATE_N_TICKS - 1 - k_idx) / (GATE_N_TICKS - 1)
        if not -GATE_PTR_BEYOND <= k_idx <= GATE_N_TICKS - 1 + GATE_PTR_BEYOND:
            # 2026-11-14: the pointer lies on the scale; an orange blob beyond an end tick is something else
            # (an orange pipe edge just above the plate read 105 % for a true 82 %, fresh site 9422)
            rejected.append({"quad": qd.tolist(), "why": "pointer %.1f ticks beyond the scale" % (
                -k_idx if k_idx < 0 else k_idx - GATE_N_TICKS + 1)})
            continue
        conf = float(np.clip(len(ys) / 40.0, 0, 1)) * (0.6 + 0.4 * comb["quality"])
        if best is None or conf * sc > best[0]:
            best = (conf * sc, opening, conf, {"quad": quad.tolist(), "ticks": [r_top, r_bot], "row": row,
                                               "n_ticks": int(round(comb["matched"] * GATE_N_TICKS)),
                                               "n_ptr": int(len(ys)), "ptr_pitches": ptr_h,
                                               "comb": {k: comb[k] for k in ("pitch", "matched", "resid",
                                                                             "major_sep", "quality")},
                                               "span_ratio": ratio})
    return best, rejected


def read(rgb, hint):
    if hint.get("type") == "lever_valve":
        return read_lever(rgb, hint)
    return read_gate(rgb, hint)
