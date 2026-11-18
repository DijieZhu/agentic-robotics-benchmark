"""Reader of version 2's text tag plates, for the privileged reference program only (never in
the cell's vendored closure or the practice ground: `bricks.readers` is on the assembly tool's deny list).

Written from the catalogue's numbers alone (`bricks.lettering`: the glyph table, the plate's size, its border line,
cap height, pitch and colours), as a program of the agent's could be:

1. the plate: a signal-yellow region (any size) whose outline gives a first quad; the printed border line then
   gives the plate's corners: across each side, intensity profiles through the expected line, the dark valley
   between the plate's yellow on both sides, a straight line fitted to each side's valley points, the four lines
   intersected (two passes). Refused: a plate cut by the frame border, too few valley points.
2. the size gate: the legibility rule's sizes from the plate's pose (cap = f x 25 mm / distance, pitch = f x 20 mm x
   cos(angle off the normal) / distance, the generator's measure), the image's own sizes at the plate centre as a
   floor (IMG_FLOOR), and the angle: below CAP_MIN_PX / PITCH_MIN_PX or beyond OFF_AXIS_MAX_DEG the reader returns
   nothing (why 'small' / 'oblique'): a first prototype returned confident wrong tags below the legible size,
   and a score cannot tell them.
3. the text: the plate rectified to a canonical image (cap height CANON_CAP_PX) and divided by its local yellow
   field (a light gradient or a shading band across the plate is not ink); for every character count 4..8 the
   fixed-pitch cells centred on the plate are matched against the glyph templates (normalised correlation, small
   shifts; templates blurred to the frame's resolution); ink outside the cells disqualifies a count; the grammar
   (1-3 letters, a hyphen, 2-5 digits) picks letters, hyphen and digits; refused below a cell score, a per-cell
   margin over the class's second-best glyph or a margin over the second-best reading; and render-and-compare: the
   whole reading drawn at its cells must correlate with the rectified plate (VERIFY_MIN).
4. the pose: planar PnP (IPPE) of the border line's centre-line rectangle (181.2 x 69.6 mm); of the two solutions the
   one nearer a facing hint (`facing_cam`: the plate's outward normal in the camera's optical frame, e.g. from the
   recorded facing), else the one with the smaller reprojection error.

Public calls (frames: the camera's optical frame, x right, y down, z forward; face frame of the plate as every bricks
target: x out of the plate toward the viewer, y the viewer's right, z up):

    read(rgb, hint) -> (value | None, confidence, debug)
        hint: 'K' (3x3, needed for the pose and the angle gate), optional 'quad' (4 x 2 px, a predicted outline TL,
        TR, BR, BL), 'roi' (x0, y0, x1, y1), 'facing_cam' (3,), 'expected' (a tag: only chooses which of several
        plates to return, never changes a reading), 'sensor_blur_px' (default 0.8); studies only: 'gates': 'off'
        (the score gates off), 'normalise': 'off' (the first, global normalisation of the plate),
        'blur_guard': 'off' (no refusal by the plate edge's width)
        value: {'text', 'score', 'pos_cam' (plate centre), 'R_cam_face', 'corners' (border centre line, 4 x 2 px),
        'outline' (4 x 2 px), 'cap_px', 'pitch_px', 'off_axis_deg', 'dist_m', 'reproj_px', 'mirror_gap_px'}
    read_all(rgb, hint) -> (values, debug)            every plate read in the frame
    find_plates(rgb, roi=None) -> [{'outline', 'area_px', 'bbox'}]
    refine_border(gray, outline) -> (border quad 4 x 2 | None, info)
    plate_pose(quad, K, facing_cam=None) -> dict | None
    find_plates_wide(rgb, depth_m, K, T_world_cam=None) -> [candidate]   the wide camera's plate finder
    lever_hint(value) -> {'R_cam_face', 'tag_pos_cam', 'plate'}   the quarter-turn valve reader's plane
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from .. import lettering as LT

__all__ = ["read", "read_all", "find_plates", "refine_border", "plate_pose", "find_plates_wide", "lever_hint",
           "CAP_MIN_PX", "PITCH_MIN_PX", "OFF_AXIS_MAX_DEG", "RULE_CAP_PX", "RULE_PITCH_PX", "RULE_OFF_AXIS_DEG"]

# ------------------------------------------------------------------------------------------------ the legibility gate
# The legible region (acceptance_v1.yaml plate_readable: cap 14 px, pitch 8 px at the plate's centre, at
# most 60 deg off the plate) and the reader's own gate around it: 10 % below the cap height, 5 % below the pitch and to
# 65 deg. A gate exactly at the region's sizes refuses about half the plates at its edge (the pose's distance has a few
# per cent of noise there), which would push the measured region above the rule; below the gate it refuses (a first
# prototype read confident wrong tags at 9-12 px). Reading beyond the region (0.9 x the cap, 0.95 x the pitch, 60-65 deg)
# is a deviation from the legibility rule that is accepted as long as the studies show no wrong tag there; nothing is
# guaranteed beyond the region.
RULE_CAP_PX, RULE_PITCH_PX = 14.0, 8.0
RULE_OFF_AXIS_DEG = 60.0             # the legible region's angle; 65 in an earlier text, which was only the gate
CAP_MIN_PX = 0.9 * RULE_CAP_PX       # 12.6
# 0.95 x the region's pitch since 2026-11-16 07:35 (was 0.9 x = 7.2): the legibility study's board fill
# returned V-7145 as N-7145 at pitch 7.3 px (cap 17 px, 57 deg), the only wrong tag of the
# studies at the camera's own blur; reading beyond the region is accepted only without a wrong tag there. Set
# after seeing that result; the study was repeated on a fresh random draw with this gate.
PITCH_MIN_PX = 0.95 * RULE_PITCH_PX  # 7.6
OFF_AXIS_MAX_DEG = 65.0              # the reader's gate (not the region)
# ------------------------------------------------------------------------------------------------ text decision
# the gates below were set from a development study of the plates (9,825 plate
# views with the gates off: 4,126 right reads, 1 wrong): right reads had cell scores >= 0.48 (0.1 % quantile 0.77),
# means >= 0.80 (0.90), cell margins >= 0.049, read margins >= 0.116, whole-plate correlations >= 0.67 (0.81); the
# wrong read (DX-94337 read DX-94333 at 16 px, 49 deg) had a cell at 0.33, a cell margin of 0.003, a mean of 0.81
IMG_FLOOR = 0.85            # the image's own cap / pitch at the plate centre: at least this x the rule's minimum
CELL_MIN = 0.60             # every cell's chosen glyph correlates at least this
MEAN_MIN = 0.80             # mean over the cells
CELL_MARGIN_MIN = 0.04      # chosen glyph over the class's second-best, every cell
READ_MARGIN_MIN = 0.03      # best reading's mean over the second-best reading's (another text)
VERIFY_MIN = 0.70           # whole-plate render-and-compare correlation
SUPPORT_MIN = 0.0           # the border line's valley found on this share of the long sides' samples
OUTSIDE_INK_MAX = 0.03      # strong ink outside the cells (fraction of the cells' strong ink) for a count to count
MARGIN_CLEAR_MM = 9.0       # ink within this of the plate's edge is the border's, not the lettering's
READER_V_MIN = 30           # the reader's yellow mask: value floor (dim plates under a room light's edge)
EDGE_ABS_MIN = 0.03         # the plate's outer edge: yellowness step at least this (0..1)
EDGE_REL_MIN = 0.30         # and this fraction of the plate's own yellowness (dim plates; was 0.08 absolute)
VALLEY_ABS_MIN = 0.04       # the border line's valley below the plate on both sides: at least this (0..1 grey)
VALLEY_REL_MIN = 0.18       # and this fraction of the plate's own brightness beside it
# ------------------------------------------------------------------------------------------------ geometry
CANON_CAP_PX = 32.0
_C = CANON_CAP_PX / (LT.TAG_CAP_M * 1000.0)          # canonical px per mm
_PW_MM, _PH_MM = LT.TAG_PLATE_W_M * 1000.0, LT.TAG_PLATE_H_M * 1000.0
_B_MM = (LT.BORDER_INSET_M + LT.BORDER_LINE_M / 2.0) * 1000.0    # border centre line from the plate's edge (4.8)
_BW_M = LT.TAG_PLATE_W_M - 2 * (LT.BORDER_INSET_M + LT.BORDER_LINE_M / 2.0)   # 0.1812
_BH_M = LT.TAG_PLATE_H_M - 2 * (LT.BORDER_INSET_M + LT.BORDER_LINE_M / 2.0)   # 0.0696
_CW, _CH = int(math.ceil(_PW_MM * _C)), int(math.ceil(_PH_MM * _C))
_UNIT_MM = LT.TAG_CAP_M * 1000.0 / LT.GRID_H
_PAD = 2.0                  # template pad (canonical px)
_SHIFT = 3                  # matching slack (canonical px) around each cell's nominal place
# face frame <- plate PnP frame (x right, y down, z into the plate)
_R_P_FACE = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])


def _plate_gray(rgb) -> np.ndarray:
    """0..1 'plate channel' 0.6 R + 0.4 G: yellow 0.85 against ink 0.06 (luma gives 0.74 against 0.06)."""
    x = np.asarray(rgb[..., :3], np.float32)
    return (0.6 * x[..., 0] + 0.4 * x[..., 1]) / 255.0


def order_quad(p) -> np.ndarray:
    """TL, TR, BR, BL for an upright view (by angle around the centroid, starting at the smallest x + y)."""
    p = np.asarray(p, np.float64).reshape(4, 2)
    c = p.mean(0)
    p = p[np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))]
    k = int(np.argmin(p[:, 0] + p[:, 1]))
    return np.roll(p, -k, axis=0)


# ------------------------------------------------------------------------------------------------ finding plates
def _yellow_mask(rgb, v_min=50) -> np.ndarray:
    """Signal-yellow pixels (OpenCV HSV: hue 12-36, saturation >= 100, value >= v_min). The reader takes dim plates
    too (v_min 30); the wide finder keeps 50 (fewer dark yellowish regions)."""
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return ((h >= 12) & (h <= 36) & (s >= 100) & (v >= v_min)).astype(np.uint8)


def find_plates(rgb, roi=None, min_area=40, max_n=24) -> list:
    """Yellow regions shaped like a plate: [{'outline' (4 x 2 px, TL TR BR BL), 'area_px', 'bbox'}], largest first."""
    m = _yellow_mask(rgb, v_min=READER_V_MIN)
    if roi is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in roi)
        keep = np.zeros_like(m)
        keep[max(0, y0):y1, max(0, x0):x1] = 1
        m &= keep
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < min_area:
            continue
        hull = cv2.convexHull(c)
        peri = cv2.arcLength(hull, True)
        q = None
        for eps in (0.02, 0.035, 0.05, 0.08):
            ap = cv2.approxPolyDP(hull, eps * peri, True)
            if len(ap) == 4:
                q = ap[:, 0, :].astype(np.float64)
                break
        if q is None:
            q = cv2.boxPoints(cv2.minAreaRect(c)).astype(np.float64)
        q = order_quad(q)
        w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
        hgt = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
        if hgt < 4 or w < 6 or not 0.6 <= w / hgt <= 6.0:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        out.append({"outline": q, "area_px": a, "bbox": (x, y, x + bw, y + bh)})
    out.sort(key=lambda e: -e["area_px"])
    return out[:max_n]


def _border_candidates(rgb, gray, roi=None, max_n=24) -> list:
    """Plates from their border line: closed dark quadrilaterals (adaptive threshold, as a fiducial detector finds
    its candidates) whose inside is the plate's yellow; adjacent plates whose yellow merges into one region are
    separated by their lines. The line's centre is taken for the contour (the outer or inner edge: +-0.8 mm), and the
    plate's outline 4.8 mm outside it, through the quad's homography."""
    g8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
    ym = _yellow_mask(rgb, v_min=READER_V_MIN)
    if roi is not None:
        x0, y0, x1, y1 = (int(round(v)) for v in roi)
        keep = np.zeros_like(ym)
        keep[max(0, y0):y1, max(0, x0):x1] = 1
        ym &= keep
    if not ym.any():
        return []
    out, seen = [], []
    for bs in (15, 31, 61):
        th = cv2.adaptiveThreshold(g8, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, bs, 5)
        cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = float(cv2.contourArea(c))
            if a < 60:
                continue
            peri = cv2.arcLength(c, True)
            ap = cv2.approxPolyDP(c, 0.04 * peri, True)
            if len(ap) != 4 or not cv2.isContourConvex(ap):
                continue
            q = order_quad(ap[:, 0, :])
            w = 0.5 * (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3]))
            h = 0.5 * (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1]))
            if h < 4 or w < 8 or not 0.7 <= w / h <= 6.0:
                continue
            mask = np.zeros(ym.shape, np.uint8)
            cv2.fillConvexPoly(mask, np.round(q).astype(np.int32), 1)
            inside = mask.astype(bool)
            if inside.sum() < 30 or ym[inside].mean() < 0.45:
                continue
            ctr = q.mean(0)
            if any(np.linalg.norm(ctr - s0) < 4 and abs(a - s1) < 0.3 * s1 for s0, s1 in seen):
                continue
            seen.append((ctr, a))
            Hq = _homography(q, (_B_MM, _B_MM, _PW_MM - _B_MM, _PH_MM - _B_MM))
            outline = _apply_h(Hq, [(0, 0), (_PW_MM, 0), (_PW_MM, _PH_MM), (0, _PH_MM)])
            out.append({"outline": outline, "area_px": a, "bbox": None, "src": "border"})
            # the same contour may be the plate's own edge (a plate on a narrow dark mount: the mount is the dark
            # ring): the outline is then the contour itself
            out.append({"outline": q.astype(np.float64), "area_px": a * 0.999, "bbox": None, "src": "edge"})
    out.sort(key=lambda e: -e["area_px"])
    return out[:max_n]


def _homography(quad_img, rect_mm) -> np.ndarray:
    """Plate mm (x right from the left edge, y down from the top) -> image px, from 4 corners."""
    x0, y0, x1, y1 = rect_mm
    src = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return cv2.getPerspectiveTransform(src, np.float32(quad_img))


def _apply_h(H, pts) -> np.ndarray:
    p = np.asarray(pts, np.float64).reshape(-1, 2)
    q = (H @ np.column_stack([p, np.ones(len(p))]).T).T
    return q[:, :2] / q[:, 2:3]


def _sample(img, pts) -> np.ndarray:
    """Bilinear samples of a float image at (N, 2) px; NaN outside."""
    p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    v = cv2.remap(img, p[:, :, 0], p[:, :, 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)
    return v.reshape(-1)


def _fit_line(pts):
    """Total least squares line through (N, 2) points with two rounds of outlier removal: (point, direction)."""
    p = np.asarray(pts, np.float64)
    keep = np.ones(len(p), bool)
    for _ in range(3):
        q = p[keep]
        if len(q) < 4:
            return None
        c = q.mean(0)
        _, _, vt = np.linalg.svd(q - c)
        d = vt[0]
        n = np.array([-d[1], d[0]])
        r = np.abs((p - c) @ n)
        s = max(1.4826 * float(np.median(r[keep])), 0.15)
        keep = r <= 3.0 * s
    return c, d


def _intersect(l1, l2):
    (c1, d1), (c2, d2) = l1, l2
    A = np.array([d1, -d2]).T
    if abs(np.linalg.det(A)) < 1e-9:
        return None
    t = np.linalg.solve(A, c2 - c1)
    return c1 + t[0] * d1


def _yellowness(rgb) -> np.ndarray:
    """0..1 '(R + G) / 2 - B': the plate's yellow about 0.86 x its light, white or grey walls, dark steel and the
    ink near 0. The plate's outer edge is a step in it on any backing."""
    x = np.asarray(rgb[..., :3], np.float32)
    return np.clip((0.5 * (x[..., 0] + x[..., 1]) - x[..., 2]) / 255.0, 0.0, 1.0)


def _side_points(prof_fn, H, side, s, along):
    """[(a, None, profile)] for one side: profiles across the side at the positions `along` (mm), samples at insets
    `s` (mm) from the plate's edge; one sampling call for the whole side."""
    A, S = np.meshgrid(np.asarray(along, float), np.asarray(s, float), indexing="ij")
    if side == 0:
        pm = np.stack([A, S], -1)
    elif side == 2:
        pm = np.stack([A, _PH_MM - S], -1)
    elif side == 1:
        pm = np.stack([_PW_MM - S, A], -1)
    else:
        pm = np.stack([S, A], -1)
    prof = prof_fn(_apply_h(H, pm.reshape(-1, 2))).reshape(len(along), len(s))
    return [(a, None, prof[i]) for i, a in enumerate(along)]


def _mm_at(side, a, inset):
    if side == 0:
        return (a, inset)
    if side == 2:
        return (a, _PH_MM - inset)
    if side == 1:
        return (_PW_MM - inset, a)
    return (inset, a)


def _alongs(side, n_along):
    if side in (0, 2):
        return np.linspace(_B_MM + 12.0, _PW_MM - _B_MM - 12.0, n_along)
    return np.linspace(_B_MM + 8.0, _PH_MM - _B_MM - 8.0, max(8, n_along // 2))


def _edge_pass(yel, H, n_along):
    """The plate's outer edge per side: the 50 % crossing of the yellowness between the outside (-4..-1.5 mm) and
    the 4 mm strip (0.6..3.2 mm inside). Returns 4 lines (image) or None per side."""
    step = 0.2
    s = np.arange(-4.0, 3.4 + 1e-9, step)
    lines = []
    for side in range(4):
        pts = []
        for a, pm, prof in _side_points(lambda p: _sample(yel, p), H, side, s, _alongs(side, n_along)):
            if not np.all(np.isfinite(prof)):
                continue
            out_lv = float(np.median(prof[s <= -1.5]))
            in_lv = float(prof[(s >= 0.6) & (s <= 3.2)].max())
            if in_lv - out_lv < max(EDGE_ABS_MIN, EDGE_REL_MIN * in_lv):
                continue
            half = 0.5 * (in_lv + out_lv)
            # the first crossing from outside going in
            idx = np.nonzero((prof[:-1] < half) & (prof[1:] >= half) & (s[:-1] > -3.0) & (s[1:] < 2.4))[0]
            if len(idx) == 0:
                continue
            i = int(idx[-1])
            t = (half - prof[i]) / max(prof[i + 1] - prof[i], 1e-9)
            pts.append(_mm_at(side, a, s[i] + t * step))
        ln = _fit_line(_apply_h(H, pts)) if len(pts) >= max(5, len(_alongs(side, n_along)) // 3) else None
        lines.append((ln, len(pts)))
    return lines


def _valley_pass(gray, H, n_along, win):
    """The border line's centre per side: the dark valley within `win` mm of 4.8 mm inside the edge, the plate's
    inner field bright beside it and the outer side rising a little (the strip, then the wall or a dark backing)."""
    step = 0.25
    s = np.arange(0.3, 10.0 + 1e-9, step)
    w = int(round(3.5 / step))
    kk = np.nonzero(np.abs(s - _B_MM) <= win)[0]
    k_lo, k_hi = int(kk[0]), int(kk[-1])
    lines = []
    for side in range(4):
        pts = []
        for a, pm, prof in _side_points(lambda p: _sample(gray, p), H, side, s, _alongs(side, n_along)):
            if not np.all(np.isfinite(prof)):
                continue
            prof = np.convolve(prof, np.ones(3) / 3.0, mode="same")
            prof[0], prof[-1] = prof[1], prof[-2]
            k = k_lo + int(np.argmin(prof[k_lo:k_hi + 1]))
            if k <= k_lo or k >= k_hi:
                continue
            lo = prof[k]
            in_max = float(prof[k + 1:k + 1 + w].max())
            out_max = float(prof[max(0, k - w):k].max())
            need = max(VALLEY_ABS_MIN, VALLEY_REL_MIN * max(in_max, 1e-6))
            if in_max - lo < need or out_max - lo < 0.35 * need:
                continue
            a0, a1, a2 = prof[k - 1], prof[k], prof[k + 1]
            den = a0 - 2 * a1 + a2
            off = 0.5 * (a0 - a2) / den if den > 1e-9 else 0.0
            pts.append(_mm_at(side, a, s[k] + float(np.clip(off, -1, 1)) * step))
        ln = _fit_line(_apply_h(H, pts)) if len(pts) >= max(5, len(_alongs(side, n_along)) // 3) else None
        lines.append((ln, len(pts)))
    return lines


def _quad_from_lines(lines):
    corners = [_intersect(lines[3], lines[0]), _intersect(lines[0], lines[1]), _intersect(lines[1], lines[2]),
               _intersect(lines[2], lines[3])]
    if any(c is None for c in corners):
        return None
    return np.array(corners)


def refine_border(gray, outline, rgb=None, n_along=28) -> tuple:
    """The border line's centre-line quad (TL, TR, BR, BL px) from a first outline of the plate, or (None, info).

    Pass 1 (needs rgb): the plate's outer edges as steps of the yellowness; pass 2 and 3: the border line's valley
    (wide, then narrow window). A side whose line is not resolved (oblique, far) keeps the outer edge moved 4.8 mm
    inward through the plate's homography. info: 'sides' (valley points per side), 'edge_sides', 'from_edge' (the
    sides taken from the outer edge)."""
    info = {}
    H = _homography(order_quad(outline), (0.0, 0.0, _PW_MM, _PH_MM))
    edge_lines = None
    if rgb is not None:
        el = _edge_pass(_yellowness(rgb), H, n_along)
        info["edge_sides"] = [n for _, n in el]
        if all(ln is not None for ln, _ in el):
            q = _quad_from_lines([ln for ln, _ in el])
            if q is not None:
                H = _homography(q, (0.0, 0.0, _PW_MM, _PH_MM))
                edge_lines = el
    quad = None
    for win in (3.6, 1.4):
        vl = _valley_pass(gray, H, n_along, win)
        info["sides"] = [n for _, n in vl]
        use, from_edge = [], []
        for side, (ln, n) in enumerate(vl):
            if ln is not None:
                use.append(ln)
                continue
            if edge_lines is None:
                info["why"] = "border"
                return None, info
            # the outer edge moved inward to the line's centre (4.8 mm), through the current homography
            al = _alongs(side, n_along)
            pts = [_mm_at(side, a, _B_MM) for a in (al[0], al[-1])]
            p = _apply_h(H, pts)
            use.append((p[0], (p[1] - p[0]) / max(np.linalg.norm(p[1] - p[0]), 1e-9)))
            from_edge.append(side)
        q = _quad_from_lines(use)
        if q is None:
            info["why"] = "border lines parallel"
            return None, info
        quad = q
        info["from_edge"] = from_edge
        H = _homography(quad, (_B_MM, _B_MM, _PW_MM - _B_MM, _PH_MM - _B_MM))
    return quad, info


# ------------------------------------------------------------------------------------------------ pose
def plate_pose(quad, K, facing_cam=None) -> dict | None:
    """The plate's pose from its border line's centre-line quad (TL, TR, BR, BL px) by IPPE with the catalogue's
    size. Returns {'pos_cam', 'R_cam_face', 'reproj_px', 'off_axis_deg', 'dist_m', 'mirror_gap_px',
    'chosen': 'facing'|'reprojection', 'alt': {...}} or None."""
    q = np.asarray(quad, np.float64).reshape(4, 1, 2)
    w, h = _BW_M / 2.0, _BH_M / 2.0
    obj = np.array([[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]], np.float64)
    try:
        n, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, q, np.asarray(K, np.float64), None,
                                                    flags=cv2.SOLVEPNP_IPPE)
    except cv2.error:
        return None
    sols = []
    for rv, tv, e in zip(rvecs, tvecs, np.asarray(errs).ravel()):
        Rp, _ = cv2.Rodrigues(rv)
        R = Rp @ _R_P_FACE                  # R_cam_face: its columns are the face axes in the camera frame
        t = tv.ravel()
        if t[2] <= 0:
            continue
        nrm = R[:, 0]
        los = -t / np.linalg.norm(t)
        off = math.degrees(math.acos(float(np.clip(nrm @ los, -1.0, 1.0))))
        sols.append({"pos_cam": t, "R_cam_face": R, "reproj_px": float(e), "off_axis_deg": off,
                     "dist_m": float(np.linalg.norm(t))})
    sols = [s for s in sols if s["off_axis_deg"] < 90.0]
    if not sols:
        return None
    chosen = "reprojection"
    sols.sort(key=lambda s: s["reproj_px"])
    if facing_cam is not None and len(sols) > 1:
        f = np.asarray(facing_cam, float)
        f = f / max(np.linalg.norm(f), 1e-12)
        sols.sort(key=lambda s: -float(s["R_cam_face"][:, 0] @ f))
        chosen = "facing"
    best = dict(sols[0])
    best["chosen"] = chosen
    if len(sols) > 1:
        best["alt"] = {k: sols[1][k] for k in ("pos_cam", "R_cam_face", "reproj_px", "off_axis_deg")}
        best["mirror_gap_px"] = abs(sols[1]["reproj_px"] - sols[0]["reproj_px"])
    else:
        best["mirror_gap_px"] = float("inf")
    return best


# ------------------------------------------------------------------------------------------------ reading
_TPL_CACHE = {}


def _templates(sig_x, sig_y):
    key = (round(sig_x * 4) / 4.0, round(sig_y * 4) / 4.0)
    t = _TPL_CACHE.get(key)
    if t is None:
        base = LT.glyph_templates(CANON_CAP_PX, pad_px=_PAD)
        t = {}
        for ch, a in base.items():
            if key[0] > 0.15 or key[1] > 0.15:
                a = cv2.GaussianBlur(a, (0, 0), sigmaX=max(key[0], 0.15), sigmaY=max(key[1], 0.15))
            t[ch] = np.ascontiguousarray(a, np.float32)
        _TPL_CACHE[key] = t
    return t


def _cell_origins(n):
    """Canonical px of each cell's template window top-left for n characters (the template's glyph grid lower left
    sits at (pad + 1.2 units, pad + pen/2 + cap) inside it)."""
    u = _UNIT_MM * _C
    x_first = (_PW_MM * _C - LT.ink_width_units(n) * u) / 2.0         # first glyph grid's left edge
    yb = (_PH_MM * _C + CANON_CAP_PX) / 2.0                           # baseline
    tx = _PAD + 0.5 * (LT.ADVANCE - LT.GRID_W) * u
    ty = _PAD + 0.5 * LT.PEN * u + CANON_CAP_PX
    return [(x_first + k * LT.ADVANCE * u - tx, yb - ty) for k in range(n)]


def _centre_sizes(Hc2i):
    """Cap height and character pitch (image px) at the plate's centre: the measure of the legibility rule
    (cap = f x 25 mm / distance, pitch = f x 20 mm x cos(angle) / distance, at the plate's centre)."""
    c = np.array([_PW_MM * _C / 2.0, _PH_MM * _C / 2.0])
    pitch = LT.ADVANCE / LT.GRID_H * CANON_CAP_PX
    p = _apply_h(Hc2i, [c + (0, -CANON_CAP_PX / 2), c + (0, CANON_CAP_PX / 2), c + (-pitch / 2, 0), c + (pitch / 2, 0)])
    return float(np.linalg.norm(p[1] - p[0])), float(np.linalg.norm(p[3] - p[2]))


def _local_sizes(Hc2i, n):
    """Cap height and pitch (image px) at the text's cells, the smallest over the cells."""
    u = _UNIT_MM * _C
    x_first = (_PW_MM * _C - LT.ink_width_units(n) * u) / 2.0
    yb = (_PH_MM * _C + CANON_CAP_PX) / 2.0
    caps, pitches = [], []
    for k in range(n):
        x = x_first + k * LT.ADVANCE * u + 0.5 * LT.GRID_W * u
        p = _apply_h(Hc2i, [(x, yb), (x, yb - CANON_CAP_PX), (x + LT.ADVANCE * u, yb - 0.5 * CANON_CAP_PX),
                            (x, yb - 0.5 * CANON_CAP_PX)])
        caps.append(float(np.linalg.norm(p[1] - p[0])))
        pitches.append(float(np.linalg.norm(p[2] - p[3])))
    return min(caps), min(pitches)


def _read_cells(ink, tpl, n):
    """[{glyph: score}] for n cells, and the fraction of ink outside the cells."""
    H, W = ink.shape
    out = []
    any_t = next(iter(tpl.values()))
    th, tw = any_t.shape
    mask = np.zeros_like(ink, bool)
    for (ox, oy) in _cell_origins(n):
        x0, y0 = int(round(ox)) - _SHIFT, int(round(oy)) - _SHIFT
        x1, y1 = x0 + tw + 2 * _SHIFT, y0 + th + 2 * _SHIFT
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            out.append(None)
            continue
        patch = np.ascontiguousarray(ink[y0:y1, x0:x1])
        mask[y0 + _SHIFT:y1 - _SHIFT, x0 + _SHIFT:x1 - _SHIFT] = True
        if float(patch.std()) < 1e-4:
            out.append({c: 0.0 for c in tpl})
            continue
        sc = {}
        for ch, t in tpl.items():
            r = cv2.matchTemplate(patch, t, cv2.TM_CCOEFF_NORMED)
            v = float(np.nanmax(r)) if np.isfinite(r).any() else -1.0
            sc[ch] = v
        out.append(sc)
    # strong ink (> 0.5) outside the cells, against the strong ink inside them (sensor noise stays below 0.5)
    strong = ink > 0.5
    outside = float(np.count_nonzero(strong & ~mask)) / max(float(np.count_nonzero(strong & mask)), 1.0)
    return out, outside


def _decide(cells):
    """Best tag under the grammar: (text, mean, min cell score, min cell margin) or None; also the runner-up's mean."""
    n = len(cells)
    if any(c is None for c in cells):
        return None, None
    cands = []
    for h in (1, 2, 3):
        nd = n - h - 1
        if not 2 <= nd <= 5:
            continue
        txt, vals, margins = [], [], []
        for k in range(n):
            sc = cells[k]
            cls = LT.LETTERS if k < h else ("-" if k == h else LT.DIGITS)
            if cls == "-":
                ch = "-"
                others = max(v for c, v in sc.items() if c != "-")
            else:
                ranked = sorted(cls, key=lambda c: -sc[c])
                ch = ranked[0]
                others = sc[ranked[1]]
            txt.append(ch)
            vals.append(sc[ch])
            margins.append(sc[ch] - others)
        cands.append(("".join(txt), float(np.mean(vals)), float(np.min(vals)), float(np.min(margins))))
    if not cands:
        return None, None
    cands.sort(key=lambda c: -c[1])
    return cands[0], (cands[1][1] if len(cands) > 1 else None)


def edge_blur_px(rgb, border) -> tuple:
    """The frame's blur at a plate, from its outer edge (the blur guard study): across each side of the plate, the
    yellowness profile from the backing to the plate's 4 mm strip; the distance between its 25 % and 75 % crossings,
    in frame px along the edge's normal, divided by 1.349 (a Gaussian edge's sigma). Per side the median over its
    profiles; the estimate is the smallest side's (a side whose edge face shows obliquely reads wider). Returns
    (sigma px or None, per side [median or None])."""
    Hb = _homography(border, (_B_MM, _B_MM, _PW_MM - _B_MM, _PH_MM - _B_MM))
    yel = _yellowness(rgb)
    step = 0.2
    s = np.arange(-4.0, 3.4 + 1e-9, step)
    per_side = []
    for side in range(4):
        ws = []
        for a, _pm, prof in _side_points(lambda p: _sample(yel, p), Hb, side, s, _alongs(side, 28)):
            if not np.all(np.isfinite(prof)):
                continue
            out_lv = float(np.median(prof[s <= -1.5]))
            in_lv = float(prof[(s >= 0.6) & (s <= 3.2)].max())
            if in_lv - out_lv < max(EDGE_ABS_MIN, EDGE_REL_MIN * in_lv):
                continue
            q = (prof - out_lv) / (in_lv - out_lv)
            xs = []
            for lev in (0.25, 0.75):
                idx = np.nonzero((q[:-1] < lev) & (q[1:] >= lev) & (s[:-1] > -3.5) & (s[1:] < 3.0))[0]
                if len(idx) == 0:
                    break
                i = int(idx[0]) if lev == 0.25 else int(idx[-1])
                xs.append(s[i] + (lev - q[i]) / max(q[i + 1] - q[i], 1e-9) * step)
            if len(xs) != 2 or xs[1] < xs[0]:
                continue
            p0, p1 = _apply_h(Hb, [_mm_at(side, a, xs[0]), _mm_at(side, a, xs[1])])
            ws.append(float(np.linalg.norm(p1 - p0)) / 1.349)
        per_side.append(float(np.median(ws)) if len(ws) >= 5 else None)
    ok = [w for w in per_side if w is not None]
    return (min(ok) if ok else None), per_side


# The blur guard (2026-11-16): the plate edge's width (edge_blur_px) above what the cell's inspection camera gives
# (lens blur 0.6 px, no motion blur) refuses the plate (why 'blur'): a frame much blurrier than the camera reads
# confusable glyphs wrong (V as W, 3 as 8) with confidence. The limit by cap height (px at the plate's centre) is the
# largest edge width of the cell's own right reads within +-30 % of that cap (study boards:
# 9,631 reads, 8 walls, 4 lights) plus 0.10 px. Measured on those boards: no read
# inside the legible region refused; with an extra 2.0 px Gaussian blur on the same frames 20 of 20 wrong tags refused,
# with 1.2 px 0 of 13 (the measure cannot tell 1.2 px from the camera's own at small plates: a change of the camera
# model needs the text study again).
BLUR_LIMIT_PX = ((12.6, 0.85), (16.0, 0.90), (20.0, 0.98), (26.0, 1.06), (30.0, 1.12), (37.0, 1.15), (45.0, 1.19),
                 (120.0, 1.21))


def blur_limit_px(cap_px) -> float:
    """BLUR_LIMIT_PX interpolated (linear in cap; held at the ends)."""
    c = [p[0] for p in BLUR_LIMIT_PX]
    v = [p[1] for p in BLUR_LIMIT_PX]
    return float(np.interp(float(cap_px), c, v))


BG_KEEP = 0.85              # a rectified pixel is the plate's field (not ink) when >= this x the background so far
BG_SIGMA_PX = 5.0           # the field's smoothing scale (canonical px: a stroke is 4.6 px wide, a glyph cell 25.6)
BG_SEED_PCT = 40.0          # the first field pixels: the inner part's pixels above this percentile (ink covers less)


def _solve_small(A, b):
    """Gaussian elimination with partial pivoting in Python floats (a 6 x 6 system; no BLAS / LAPACK, whose kernels
    differ in their last bits between CPUs)."""
    n = len(b)
    M = [[float(A[i][j]) for j in range(n)] + [float(b[i])] for i in range(n)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(c + 1, n):
            f = M[r][c] / M[c][c]
            for j in range(c, n + 1):
                M[r][j] -= f * M[c][j]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))) / M[i][i]
    return x


def _quad_field(sm, inner, iters=3):
    """A quadratic surface in the plate's coordinates fitted to its field pixels (the inner part's brighter pixels,
    then those within BG_KEEP of the fit), or None."""
    H, W = sm.shape
    yy, xx = np.mgrid[0:H, 0:W]
    X = (xx - 0.5 * (W - 1)) / (0.5 * W)
    Y = (yy - 0.5 * (H - 1)) / (0.5 * H)
    basis = [np.ones_like(X), X, Y, X * X, X * Y, Y * Y]
    v = sm[inner]
    if v.size < 50:
        return None
    M = inner & (sm >= float(np.percentile(v, BG_SEED_PCT)))
    fit = None
    for _ in range(iters):
        if int(M.sum()) < 30:
            return fit
        B = [f[M].astype(np.float64) for f in basis]
        t = sm[M].astype(np.float64)
        A = [[float(np.sum(B[i] * B[j])) for j in range(6)] for i in range(6)]
        rhs = [float(np.sum(B[i] * t)) for i in range(6)]
        co = _solve_small(A, rhs)
        if co is None:
            return fit
        fit = sum(cf * f for cf, f in zip(co, basis)).astype(np.float32)
        M = inner & (sm >= BG_KEEP * fit)
    return fit


def _flatten(canon) -> tuple:
    """(the rectified plate divided by its local yellow field, info). The field is estimated from the plate's own
    pixels inside the border line (the strip outside it and the plate's edge face, seen obliquely, may be lit
    differently): a smooth quadratic surface through the inner part's field pixels first, then the pixels at least
    BG_KEEP of it taken as field and a normalised convolution (Gaussian, BG_SIGMA_PX) of their values, which fills the
    strokes from the field around them; twice. On an evenly lit plate the field is flat and the ink map is the one the
    global normalisation gave; a gradient or a shading band (a plate high under a spot light, a plate lit from one
    side) is divided out. info: the field's max / min over the inner part (1.0 on an even plate)."""
    c = np.asarray(canon, np.float32)
    sm = cv2.blur(c, (3, 3))
    a = int(round((LT.BORDER_INSET_M * 1000 + LT.BORDER_LINE_M * 1000 + 2.0) * _C))
    inner = np.zeros(c.shape, bool)
    inner[a:_CH - a, a:_CW - a] = True
    rough = _quad_field(sm, inner)
    if rough is None:
        rough = np.full(c.shape, float(np.percentile(sm[inner], 92)), np.float32)
    rough = np.maximum(rough, 1e-3)
    bg = rough
    for _ in range(2):
        mask = (inner & (sm >= BG_KEEP * bg)).astype(np.float32)
        num = cv2.GaussianBlur(c * mask, (0, 0), BG_SIGMA_PX)
        den = cv2.GaussianBlur(mask, (0, 0), BG_SIGMA_PX)
        bg = np.where(den > 0.05, num / np.maximum(den, 1e-6), rough).astype(np.float32)
    bg = np.maximum(bg, 1e-3)
    ib = bg[a:_CH - a, a:_CW - a]
    span = float(ib.max() / max(float(ib.min()), 1e-3))
    return c / bg, round(span, 3)


def _read_quad(img_rgb, gray, border, K, hint, dbg):
    """Read one plate from its border quad. Returns a value dict, or None with dbg['why']."""
    Himg, Wimg = gray.shape
    # the canonical plate image <- the frame
    Hb = _homography(border, (_B_MM, _B_MM, _PW_MM - _B_MM, _PH_MM - _B_MM))     # mm -> image
    S = np.diag([1.0 / _C, 1.0 / _C, 1.0])
    Hc2i = Hb @ S                                                                   # canonical px -> image
    outline = _apply_h(Hb, [(0, 0), (_PW_MM, 0), (_PW_MM, _PH_MM), (0, _PH_MM)])
    dbg["outline"] = outline.round(2).tolist()
    if outline[:, 0].min() < 1 or outline[:, 1].min() < 1 or outline[:, 0].max() > Wimg - 2 or \
            outline[:, 1].max() > Himg - 2:
        dbg["why"] = "cut"
        return None
    pose = plate_pose(border, K, hint.get("facing_cam")) if K is not None else None
    if K is not None and pose is None:
        dbg["why"] = "pose"
        return None
    if pose is not None:
        dbg["off_axis_deg"] = round(pose["off_axis_deg"], 2)
        if pose["off_axis_deg"] > OFF_AXIS_MAX_DEG:
            dbg["why"] = "oblique"
            return None
    Hi2c = np.linalg.inv(Hc2i)
    best = None
    sizes = {}
    for n in (4, 5, 6, 7, 8):
        cap_px, pitch_px = _local_sizes(Hc2i, n)
        sizes[n] = (cap_px, pitch_px)
    cap8, pitch8 = sizes[8]
    cap_i, pitch_i = _centre_sizes(Hc2i)
    if pose is not None:
        # the legibility rule's measure (the generator's): f x 25 mm / distance and f x 20 mm x cos(angle) /
        # distance, from the plate's pose; the image's own sizes at the centre are kept as a floor
        f = float(K[0, 0])
        cap_c = f * LT.TAG_CAP_M / pose["dist_m"]
        pitch_c = f * LT.TAG_PITCH_M * math.cos(math.radians(pose["off_axis_deg"])) / pose["dist_m"]
    else:
        cap_c, pitch_c = cap_i, pitch_i
    dbg["cap_px"], dbg["pitch_px"] = round(cap_c, 2), round(pitch_c, 2)
    dbg["cap_px_img"], dbg["pitch_px_img"] = round(cap_i, 2), round(pitch_i, 2)
    if cap_c < CAP_MIN_PX or pitch_c < PITCH_MIN_PX or cap_i < IMG_FLOOR * CAP_MIN_PX or \
            pitch_i < IMG_FLOOR * PITCH_MIN_PX:
        dbg["why"] = "small"
        return None
    up = cap8 / CANON_CAP_PX
    interp = cv2.INTER_LINEAR if up < 1.5 else cv2.INTER_AREA
    canon = cv2.warpPerspective(gray, Hi2c, (_CW, _CH), flags=interp | cv2.WARP_FILL_OUTLIERS,
                                borderMode=cv2.BORDER_REPLICATE)
    a = int(round((LT.BORDER_INSET_M * 1000 + LT.BORDER_LINE_M * 1000 + 2.0) * _C))
    inner = canon[a:_CH - a, a:_CW - a]
    hi, lo = float(np.percentile(inner, 92)), float(np.percentile(inner, 3))
    dbg["contrast"] = round(hi - lo, 3)
    if hi - lo < 0.12:                       # the absolute contrast of the plate as seen (dim plates are refused)
        dbg["why"] = "flat"
        return None
    if hint.get("normalise", "local") != "off":
        # 2026-11-16: the plate against its local background, so that a light gradient or a shading band across
        # the plate (a plate high under a room's spot light: its upper half at half the lower half's brightness) is
        # not taken for ink outside the cells
        flat_img, bg_info = _flatten(canon)
        dbg["bg_span"] = bg_info
        inner = flat_img[a:_CH - a, a:_CW - a]
        hi, lo = float(np.percentile(inner, 92)), float(np.percentile(inner, 3))
        canon_n = flat_img
    else:
        canon_n = canon
    ink = np.clip((hi - canon_n) / max(hi - lo, 1e-6), 0.0, 1.0).astype(np.float32)
    # the border line and the strip outside it are not lettering: clear 9 mm from the edge (the lettering starts
    # 17.35 mm from the side edges and 25.3 mm from the top and bottom; 9 mm tolerates a side found from the outer
    # edge, which a plate's own edge face seen obliquely shifts by up to 2 mm)
    b = int(round(MARGIN_CLEAR_MM * _C))
    ink[:b, :] = 0
    ink[_CH - b:, :] = 0
    ink[:, :b] = 0
    ink[:, _CW - b:] = 0
    blur = float(hint.get("sensor_blur_px", 0.8))
    # the frame's blur in the canonical image: horizontally by the pitch's magnification, vertically by the cap's
    pitch_canon = LT.ADVANCE / LT.GRID_H * CANON_CAP_PX
    tpl = _templates(blur * pitch_canon / max(pitch8, 1e-6), blur * CANON_CAP_PX / max(cap8, 1e-6))
    reads = []
    for n in (4, 5, 6, 7, 8):
        cells, outside = _read_cells(ink, tpl, n)
        if outside > OUTSIDE_INK_MAX:
            continue
        d, runner = _decide(cells)
        if d is None:
            continue
        reads.append((d, runner, n, outside))
    if not reads:
        dbg["why"] = "no count fits"
        return None
    reads.sort(key=lambda r: -r[0][1])
    (txt, mean, mn, margin), runner, n, outside = reads[0]
    second = max([r[0][1] for r in reads[1:]] + ([runner] if runner is not None else []), default=None)
    verify = _verify(ink, txt, b, pitch_canon, blur * pitch_canon / max(pitch8, 1e-6),
                     blur * CANON_CAP_PX / max(cap8, 1e-6))
    sides = list(dbg.get("sides") or [])
    support = min(sides[0], sides[2]) / 28.0 if len(sides) == 4 else 0.0
    eb, _eb_sides = edge_blur_px(img_rgb, border)
    dbg.update(n=n, mean=round(mean, 4), min=round(mn, 4), cell_margin=round(margin, 4), outside=round(outside, 4),
               second=None if second is None else round(second, 4), text_seen=txt, verify=round(verify, 4),
               support=round(support, 3), edge_blur=None if eb is None else round(eb, 3))
    gates = hint.get("gates", "on") != "off"
    if gates and (mn < CELL_MIN or mean < MEAN_MIN):
        dbg["why"] = "score"
        return None
    if gates and margin < CELL_MARGIN_MIN:
        dbg["why"] = "cell margin"
        return None
    if gates and second is not None and mean - second < READ_MARGIN_MIN:
        dbg["why"] = "read margin"
        return None
    if gates and verify < VERIFY_MIN:
        dbg["why"] = "verify"
        return None
    if gates and support < SUPPORT_MIN:
        dbg["why"] = "support"
        return None
    if gates and hint.get("blur_guard", "on") != "off" and eb is not None and eb > blur_limit_px(cap_c):
        dbg["why"] = "blur"
        return None
    if not LT.tag_ok(txt):
        dbg["why"] = "grammar"
        return None
    val = {"text": txt, "score": mean, "corners": np.asarray(border, float), "outline": outline,
           "cap_px": cap_c, "pitch_px": pitch_c, "cap_px_img": cap_i, "pitch_px_img": pitch_i,
           "cap_px_min": sizes[n][0], "pitch_px_min": sizes[n][1],
           "min_cell": mn, "cell_margin": margin, "read_margin": None if second is None else mean - second,
           "verify": verify, "support": support, "from_edge": list(dbg.get("from_edge") or []),
           "edge_blur_px": eb, "bg_span": dbg.get("bg_span")}
    if pose is not None:
        val.update(pos_cam=pose["pos_cam"], R_cam_face=pose["R_cam_face"], off_axis_deg=pose["off_axis_deg"],
                   dist_m=pose["dist_m"], reproj_px=pose["reproj_px"], mirror_gap_px=pose["mirror_gap_px"],
                   pose_chosen=pose["chosen"])
    return val


_FULL_CACHE = {}


def _verify(ink, text, clear_px, pitch_canon, sig_x, sig_y) -> float:
    """Render-and-compare: the normalised correlation, over the plate's inner field, between the rectified ink and
    the whole reading drawn at its cells (the lettering's own rasteriser, blurred as the frame). A quad that is not
    the plate's (two plates merged, a side found wrong) or a misplaced cell grid scores low even when every cell
    found a good glyph."""
    key = (text, round(sig_x * 4) / 4.0, round(sig_y * 4) / 4.0)
    ref = _FULL_CACHE.get(key)
    if ref is None:
        u = _UNIT_MM * _C
        x_first = (_PW_MM * _C - LT.ink_width_units(len(text)) * u) / 2.0
        yb = (_PH_MM * _C + CANON_CAP_PX) / 2.0
        cov = LT.coverage(LT.text_segments(text, u, x_first, yb), _CW, _CH, 1.0, 1.0, 4).astype(np.float32) / 16.0
        if key[1] > 0.15 or key[2] > 0.15:
            cov = cv2.GaussianBlur(cov, (0, 0), sigmaX=max(key[1], 0.15), sigmaY=max(key[2], 0.15))
        ref = cov
        if len(_FULL_CACHE) > 4000:
            _FULL_CACHE.clear()
        _FULL_CACHE[key] = ref
    a = ink[clear_px:_CH - clear_px, clear_px:_CW - clear_px].astype(np.float64)
    r = ref[clear_px:_CH - clear_px, clear_px:_CW - clear_px].astype(np.float64)
    a = a - a.mean()
    r = r - r.mean()
    den = math.sqrt(float((a * a).sum()) * float((r * r).sum()))
    return float((a * r).sum() / den) if den > 1e-12 else 0.0


def _confidence(val) -> float:
    """A returned reading passed every gate: 0.5 .. 1.0 by its mean score (the decision is the gates, not this)."""
    return float(0.5 + 0.5 * np.clip((val["score"] - MEAN_MIN) / 0.25, 0.0, 1.0))


def read_all(rgb, hint=None) -> tuple:
    """Every plate read in the frame: ([value], debug {'candidates': [...]})."""
    hint = hint or {}
    K = hint.get("K")
    K = None if K is None else np.asarray(K, np.float64)
    gray = _plate_gray(rgb)
    cands = _border_candidates(rgb, gray, roi=hint.get("roi")) + find_plates(rgb, roi=hint.get("roi"))
    if hint.get("quad") is not None:
        cands = [{"outline": order_quad(hint["quad"]), "area_px": 0.0, "bbox": None, "given": True}] + cands
    vals, dbg = [], {"candidates": []}
    done = []                               # plates read: (border centre, outline width)
    tried = []                              # quads read (whether or not a reading came out)
    for c in cands:
        q0 = np.asarray(c["outline"], float)
        c0, w0 = q0.mean(0), float(np.linalg.norm(q0[1] - q0[0]))
        # the same plate from another threshold or source, already refined: skip (a failed one does not block)
        if any(np.linalg.norm(c0 - e[0]) < 0.12 * max(w0, 1.0) and abs(w0 - e[1]) < 0.15 * max(w0, 1.0)
               for e in done):
            continue
        d = {"outline0": q0.round(1).tolist(), "src": c.get("src", "yellow")}
        border, info = refine_border(gray, c["outline"], rgb=rgb)
        d.update(info)
        if border is None:
            dbg["candidates"].append(d)
            continue
        ctr = border.mean(0)
        Hb = _homography(border, (_B_MM, _B_MM, _PW_MM - _B_MM, _PH_MM - _B_MM))
        ol = _apply_h(Hb, [(0, 0), (_PW_MM, 0), (_PW_MM, _PH_MM), (0, _PH_MM)])
        key = (ctr, float(np.linalg.norm(ol[1] - ol[0])))
        if any(np.linalg.norm(ctr - e[0]) < 1.5 and abs(key[1] - e[1]) < 0.02 * max(key[1], 1.0) for e in tried):
            continue                        # this very quad was read already
        tried.append(key)
        v = _read_quad(rgb, gray, border, K, hint, d)
        dbg["candidates"].append(d)
        if v is None:
            continue
        done.append(key)                    # a reading covers this plate: other sources of it are skipped
        v["confidence"] = _confidence(v)
        # one reading per plate: overlapping quads (a merged region's quad over a plate's own) keep the better fit
        clash = [k for k, w in enumerate(vals) if _overlap(v["outline"], w["outline"])]
        if clash:
            if all(v["verify"] > vals[k]["verify"] for k in clash):
                for k in sorted(clash, reverse=True):
                    vals.pop(k)
            else:
                continue
        vals.append(v)
    return vals, dbg


def _overlap(q1, q2) -> bool:
    """True when either quad's centre lies inside the other."""
    for a, b in ((q1, q2), (q2, q1)):
        c = np.asarray(a, np.float32).mean(0)
        if cv2.pointPolygonTest(np.asarray(b, np.float32).reshape(-1, 1, 2), (float(c[0]), float(c[1])), False) >= 0:
            return True
    return False


def read(rgb, hint=None):
    """One plate: the one whose text equals hint['expected'] when it is read, else the one nearest the predicted
    quad, else the one nearest the image centre. (value | None, confidence, debug)."""
    hint = hint or {}
    vals, dbg = read_all(rgb, hint)
    if not vals:
        whys = [c.get("why") for c in dbg["candidates"]]
        dbg["why"] = whys[0] if whys else "no plate"
        return None, 0.0, dbg
    H, W = rgb.shape[:2]
    ref = np.asarray(hint["quad"], float).reshape(4, 2).mean(0) if hint.get("quad") is not None else \
        np.array([W / 2.0, H / 2.0])
    exp = hint.get("expected")
    vals.sort(key=lambda v: (0 if (exp is not None and v["text"] == exp) else 1,
                             float(np.linalg.norm(np.asarray(v["corners"]).mean(0) - ref))))
    v = vals[0]
    dbg["n_read"] = len(vals)
    return v, v["confidence"], dbg


def lever_hint(value) -> dict:
    """The quarter-turn valve reader's plane from a read plate: the plate is parallel to the valve's face,
    so its face rotation is the valve's; its centre replaces the marker's centre as the grip check's anchor."""
    return {"R_cam_face": np.asarray(value["R_cam_face"], float), "tag_pos_cam": np.asarray(value["pos_cam"], float),
            "plate": value}


# ------------------------------------------------------------------------------------------------ wide camera
PLATE_AREA_M2 = LT.TAG_PLATE_W_M * LT.TAG_PLATE_H_M


WIDE_HUE = (18, 32)          # OpenCV hue (0-180) of a candidate's median: signal yellow renders at 23-27; orange
                             # paint (guards, actuators, shelving) at 13-17
WIDE_MIN_HEIGHT_M = 0.15     # above the floor (with T_world_cam): hazard stripes on sills and kerbs are lower
STRIPE_GAP = 1.6             # a candidate with >= 2 similar yellow regions within this x its size is a stripe row


def find_plates_wide(rgb, depth_m, K, T_world_cam=None, min_px=4, area_m2=(0.0035, 0.024),
                     height_m=(0.025, 0.20), width_m=(0.035, 0.30), hue=WIDE_HUE,
                     min_height_m=WIDE_MIN_HEIGHT_M) -> list:
    """Yellow plate candidates in a wide (navigation) frame with a metric size check from the depth image.

    rgb (H, W, 3) uint8; depth_m (H, W) float metres (NaN or 0 invalid) or None (then no size check, 'range_m'
    None); K 3x3. A candidate: a connected yellow region of >= min_px pixels whose metric area (pixels x depth^2 /
    (fx fy)) fits a plate seen at up to about 70 degrees (0.0151 m^2 head-on), and whose bounding box (pixel
    extent x depth / f; an oblique plate's box is taller than the plate) stays within plate-like bounds. Returns
    [{'centroid_px', 'bbox', 'area_px', 'range_m', 'area_m2', 'size_m', 'pos_cam', 'pos_world', 'ok_size', 'hue',
    'sat', 'fill', 'height_m'}], the checked ones first (largest first). 'ok_size' also needs the region's median
    hue inside WIDE_HUE and, with T_world_cam, its centre at least WIDE_MIN_HEIGHT_M above the floor (z = 0), and
    not two similar yellow regions within STRIPE_GAP x its size (a hazard stripe's row; key 'stripe_neighbours')."""
    m = _yellow_mask(rgb)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2HSV)
    K = np.asarray(K, float)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    out = []
    for k in range(1, n):
        a = int(stats[k, cv2.CC_STAT_AREA])
        if a < min_px:
            continue
        x, y, w, h = (int(stats[k, i]) for i in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH,
                                                  cv2.CC_STAT_HEIGHT))
        u, v = float(cent[k, 0]), float(cent[k, 1])
        sel = lab[y:y + h, x:x + w] == k
        hh = hsv[y:y + h, x:x + w][sel]
        rec = {"centroid_px": (u, v), "bbox": (x, y, x + w, y + h), "area_px": a, "range_m": None, "area_m2": None,
               "size_m": None, "pos_cam": None, "pos_world": None, "ok_size": None,
               "hue": float(np.median(hh[:, 0])), "sat": float(np.median(hh[:, 1])), "fill": a / float(w * h),
               "height_m": None}
        if depth_m is not None:
            d = np.asarray(depth_m, float)[y:y + h, x:x + w][lab[y:y + h, x:x + w] == k]
            d = d[np.isfinite(d) & (d > 0.05)]
            if len(d):
                z = float(np.median(d))
                rec["range_m"] = z
                am = a * z * z / (fx * fy)
                hm, wm = (h + 0.5) * z / fy, (w + 0.5) * z / fx
                rec["area_m2"], rec["size_m"] = am, (wm, hm)
                rec["ok_size"] = bool(area_m2[0] <= am <= area_m2[1] and height_m[0] <= hm <= height_m[1] and
                                      width_m[0] <= wm <= width_m[1])
                p = np.array([(u - cx) / fx * z, (v - cy) / fy * z, z])
                rec["pos_cam"] = p
                if not hue[0] <= rec["hue"] <= hue[1]:
                    rec["ok_size"] = False
                if T_world_cam is not None:
                    T = np.asarray(T_world_cam, float)
                    rec["pos_world"] = T[:3, :3] @ p + T[:3, 3]
                    rec["height_m"] = float(rec["pos_world"][2])
                    if rec["height_m"] < min_height_m:
                        rec["ok_size"] = False
        out.append(rec)
    # hazard stripes (yellow and black bands on beams, hatch heads, sills): a row of similar yellow segments about
    # one segment apart; a plate stands alone (plates on a board are about two plate widths apart, or more)
    nbs = []
    for r in out:
        x0, y0, x1, y1 = r["bbox"]
        size = max(x1 - x0, y1 - y0, 1)
        nb = []
        for j, q in enumerate(out):
            if q is r:
                continue
            dd = np.hypot(q["centroid_px"][0] - r["centroid_px"][0], q["centroid_px"][1] - r["centroid_px"][1])
            if dd < STRIPE_GAP * size and 0.3 <= q["area_px"] / max(r["area_px"], 1) <= 3.0:
                nb.append(j)
        nbs.append(nb)
    for i, r in enumerate(out):
        # in a row: two similar neighbours, or the end of a row (a neighbour that has two)
        row = len(nbs[i]) >= 2 or any(len(nbs[j]) >= 2 for j in nbs[i])
        r["stripe_neighbours"] = len(nbs[i])
        if row and r["ok_size"]:
            r["ok_size"] = False
    out.sort(key=lambda r: (0 if r["ok_size"] else 1, -r["area_px"]))
    return out
