"""Painted-tag reader (numpy + scipy only).  Knows the published font, tag format and plate layout; finds the plate from
the image alone (bright quadrilateral), so it works when the square code is covered by fouling.

    read_tags(frame_rgb) -> [(tag, plate_corners_px)]        plate corners: TL, TR, BR, BL (plate upright)
    read_tag_with_homography(gray, Hm) -> tag or None         Hm: panel-local (X, Y) of one valve -> image (u, v)

Assumes the camera roll relative to the panel is within +-45 deg (the plate's long side is horizontal).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from . import imgproc as IP  # P14: package-relative import (G4: sys.path insert + top-level import)
from . import panel_std as PS

_TEMPL = {}
for _ch in PS.READ_ALPHABET:
    _g = PS.glyph(_ch)
    _TEMPL[_ch] = ((_g - _g.mean()) / (_g.std() + 1e-9)).reshape(-1)
_TCH = list(_TEMPL)
_TM = np.stack([_TEMPL[c] for c in _TCH])            # (n_glyphs, 35)
_SUB = np.array([-0.25, 0.0, 0.25])

# font-cell centres of the 3 characters, panel-local (X, Y)
_CELLS = np.array([[x0 + PS.FONT_CELL / 2, y0 + PS.FONT_CELL / 2] for _, _, _, x0, y0 in PS.text_cell_rects()])


def font_cell_px(Hm: np.ndarray) -> float:
    """Smallest projected font-cell edge [px] over the text block (from the plate homography)."""
    x0, x1 = PS.TEXT_LEFT, PS.TEXT_LEFT + PS.TEXT_W
    y0, y1 = PS.TEXT_CENTER_Y - PS.TEXT_H / 2, PS.TEXT_CENTER_Y + PS.TEXT_H / 2
    q = IP.apply_h(Hm, [[x0, y1], [x1, y1], [x1, y0], [x0, y0]])
    ncol = PS.N_CHARS * PS.FONT_COLS + (PS.N_CHARS - 1) * PS.CHAR_GAP_CELLS
    w = min(np.hypot(*(q[1] - q[0])), np.hypot(*(q[2] - q[3]))) / ncol
    h = min(np.hypot(*(q[3] - q[0])), np.hypot(*(q[2] - q[1]))) / PS.FONT_ROWS
    return float(min(w, h))


def read_tag_with_homography(gray: np.ndarray, Hm: np.ndarray, min_score: float = 0.35, min_margin: float = 0.08,
                             details: bool = False, min_font_px: float = 0.0):
    if min_font_px > 0 and font_cell_px(Hm) < min_font_px:
        return (None, None) if details else None
    oy, ox = np.meshgrid(_SUB, _SUB, indexing="ij")
    off = np.stack([ox.reshape(-1), -oy.reshape(-1)], 1) * PS.FONT_CELL          # (9, 2), Y up
    P = (_CELLS[:, None, :] + off[None]).reshape(-1, 2)
    uv = IP.apply_h(Hm, P)
    s = IP.bilinear(gray, uv[:, 0], uv[:, 1]).reshape(len(_CELLS), 9)
    if np.isnan(s).mean() > 0.02:
        return (None, None) if details else None
    v = np.nanmean(s, 1).reshape(PS.N_CHARS, PS.FONT_ROWS * PS.FONT_COLS)
    out, info = "", []
    for k in range(PS.N_CHARS):
        x = v[k]
        z = -(x - x.mean()) / (x.std() + 1e-9)               # ink is dark
        sc = _TM @ z / z.size
        order = np.argsort(-sc)
        b, r = float(sc[order[0]]), float(sc[order[1]])
        info.append((_TCH[order[0]], round(b, 3), round(b - r, 3)))
        out += _TCH[order[0]] if (b > min_score and b - r > min_margin) else "?"
    tag = out if PS.tag_is_valid(out) else None
    return (tag, info) if details else tag


def _plate_candidates(gray: np.ndarray, min_plate_px: float):
    Hh, Ww = gray.shape
    masks = []
    for win in (Ww // 4 | 1, Ww // 2 | 1):
        masks.append(IP.adaptive_mask(gray, win, 0.03, dark=False))
    t = IP.otsu_threshold(gray[::2, ::2])
    masks.append(gray > t)
    min_area = int(0.25 * min_plate_px * min_plate_px * PS.PLATE_H / PS.PLATE_W)
    tried = []
    for mk in masks:
        for sub, org, area in IP.component_boxes(mk, min_area, int(min_plate_px), max_frac=0.98):
            h, w = sub.shape
            if w < 0.6 * h:
                continue
            pts = IP.region_hull_points(sub, org)
            quad, ratio = IP.hull_to_quad(pts)
            if quad is None or ratio < 0.85:
                continue
            quad = IP.order_quad(quad)
            size = float(np.hypot(*(quad[2] - quad[0])))
            if any(float(np.mean(np.hypot(*(quad - t).T))) < max(1.0, 0.02 * size) for t in tried):
                continue
            tried.append(quad)
            yield quad


def read_tags(frame_rgb: np.ndarray, min_plate_px: float = 24.0, min_font_px: float = 0.0,
              min_margin: float = 0.08) -> list:
    """v1 = the defaults (reader of record in the G4 sweep).  'strict' = min_font_px 2.5, min_margin 0.15."""
    gray = IP.to_gray(frame_rgb)
    src = PS.plate_corners_local()
    out = []
    for quad in _plate_candidates(gray, min_plate_px):
        # long side should be the top/bottom edge
        top = np.hypot(*(quad[1] - quad[0])) + np.hypot(*(quad[2] - quad[3]))
        side = np.hypot(*(quad[3] - quad[0])) + np.hypot(*(quad[2] - quad[1]))
        if top < 0.8 * side:
            continue
        quiet_px = 0.5 * top / PS.PLATE_W * PS.QUIET
        qr = IP.refine_quad(gray, quad, band_px=max(1.0, 0.8 * quiet_px), inside_dark=False)
        tag = None
        for q in ((qr, quad) if qr is not quad else (quad,)):
            Hm = IP.homography(src, q)
            tag = read_tag_with_homography(gray, Hm, min_margin=min_margin, min_font_px=min_font_px)
            if tag is not None:
                break
        if tag is not None:
            c = quad.mean(0)
            if not any(np.hypot(*(c - o[1].mean(0))) < 0.3 * top / 2 for o in out):
                out.append((tag, q))
    return out


# ------------------------------------------------------------------ v2: content registration (opt-in, after the G4 sweep)
_POS_IDX = [[_TCH.index(ch) for ch in alph] for alph in PS.TAG_ALPHABETS]


def _format_score(gray, Hm) -> float:
    """Sum over the 3 characters of the best template score inside that position's alphabet (tag format)."""
    oy, ox = np.meshgrid(_SUB, _SUB, indexing="ij")
    off = np.stack([ox.reshape(-1), -oy.reshape(-1)], 1) * PS.FONT_CELL
    P = (_CELLS[:, None, :] + off[None]).reshape(-1, 2)
    uv = IP.apply_h(Hm, P)
    s = IP.bilinear(gray, uv[:, 0], uv[:, 1]).reshape(len(_CELLS), 9)
    if np.isnan(s).mean() > 0.02:
        return -9.0
    v = np.nanmean(s, 1).reshape(PS.N_CHARS, -1)
    tot = 0.0
    for k in range(PS.N_CHARS):
        x = v[k]
        z = -(x - x.mean()) / (x.std() + 1e-9)
        sc = _TM @ z / z.size
        tot += float(sc[_POS_IDX[k]].max())
    return tot


def register_quad(gray, quad, steps=(4.0, 2.0, 1.0, 0.5), max_move: float = None):
    """Coordinate ascent of the 8 corner coordinates on the format score (fixes plate edges hidden by fouling).
    Each corner may move at most max_move px from its start (default 6 % of the plate's top edge)."""
    src = PS.plate_corners_local()
    q0 = np.asarray(quad, float).copy()
    q = q0.copy()
    if max_move is None:
        max_move = max(3.0, 0.06 * float(np.hypot(*(q0[1] - q0[0]))))
    best = _format_score(gray, IP.homography(src, q))
    for st in steps:
        improved = True
        it = 0
        while improved and it < 6:
            improved, it = False, it + 1
            for i in range(4):
                for j in range(2):
                    for sgn in (1, -1):
                        q2 = q.copy()
                        q2[i, j] += sgn * st
                        if np.hypot(*(q2[i] - q0[i])) > max_move:
                            continue
                        s2 = _format_score(gray, IP.homography(src, q2))
                        if s2 > best + 1e-4:
                            q, best, improved = q2, s2, True
    return q, best


def read_tags_v2(frame_rgb: np.ndarray, min_plate_px: float = 24.0, min_font_px: float = 0.0,
                 min_margin: float = 0.08) -> list:
    """read_tags + content registration of the plate quad when the first read fails or is ambiguous."""
    gray = IP.to_gray(frame_rgb)
    src = PS.plate_corners_local()
    out = []
    for quad in _plate_candidates(gray, min_plate_px):
        top = np.hypot(*(quad[1] - quad[0])) + np.hypot(*(quad[2] - quad[3]))
        side = np.hypot(*(quad[3] - quad[0])) + np.hypot(*(quad[2] - quad[1]))
        if top < 0.8 * side:
            continue
        quiet_px = 0.5 * top / PS.PLATE_W * PS.QUIET
        qr = IP.refine_quad(gray, quad, band_px=max(1.0, 0.8 * quiet_px), inside_dark=False)
        tag, q_used = None, qr
        for q in ((qr, quad) if qr is not quad else (quad,)):
            tag = read_tag_with_homography(gray, IP.homography(src, q), min_margin=min_margin, min_font_px=min_font_px)
            if tag is not None:
                q_used = q
                break
        if tag is None and min(top, side) / 2 >= 12.0 and _format_score(gray, IP.homography(src, qr)) >= 1.5:
            q_reg, sc = register_quad(gray, qr)
            t2, info = read_tag_with_homography(gray, IP.homography(src, q_reg), details=True, min_margin=min_margin,
                                                min_font_px=min_font_px)
            if t2 is not None and all(b >= 0.6 for _, b, _ in info):
                tag, q_used = t2, q_reg
        if tag is not None:
            c = quad.mean(0)
            if not any(np.hypot(*(c - o[1].mean(0))) < 0.3 * top / 2 for o in out):
                out.append((tag, q_used))
    return out


# ----------------------------------------------------------------------------------------------- v3 (package P14)
def _plate_candidates_v3(gray: np.ndarray, min_plate_px: float):
    """v1 / v2's bright-quad candidates plus masks for a LIGHT panel face (P14: the finished skid's lit blue-grey face
    is 0.4-0.55 grey next to a 0.75-0.9 plate, so the v1 masks join the plate to the lamp's hot spot on the face):
    adaptive thresholds with larger offsets (0.08, 0.15) and the upper split of the bright half of the histogram."""
    Hh, Ww = gray.shape
    masks = []
    for win in (Ww // 4 | 1, Ww // 2 | 1):
        masks.append(IP.adaptive_mask(gray, win, 0.03, dark=False))
    t = IP.otsu_threshold(gray[::2, ::2])
    masks.append(gray > t)
    for win in (Ww // 4 | 1, Ww // 2 | 1):
        for c in (0.08, 0.15):
            masks.append(IP.adaptive_mask(gray, win, c, dark=False))
    hi = gray[::2, ::2][gray[::2, ::2] > t]
    if hi.size > 100:
        masks.append(gray > IP.otsu_threshold(hi))
    min_area = int(0.25 * min_plate_px * min_plate_px * PS.PLATE_H / PS.PLATE_W)
    tried = []
    for mk in masks:
        for sub, org, area in IP.component_boxes(mk, min_area, int(min_plate_px), max_frac=0.98):
            h, w = sub.shape
            if w < 0.6 * h:
                continue
            pts = IP.region_hull_points(sub, org)
            quad, ratio = IP.hull_to_quad(pts)
            if quad is None or ratio < 0.85:
                continue
            quad = IP.order_quad(quad)
            size = float(np.hypot(*(quad[2] - quad[0])))
            if any(float(np.mean(np.hypot(*(quad - t_).T))) < max(1.0, 0.02 * size) for t_ in tried):
                continue
            tried.append(quad)
            yield quad


def read_tags_v3(frame_rgb: np.ndarray, min_plate_px: float = 24.0, min_font_px: float = 0.0,
                 min_margin: float = 0.08, reg_min_score: float = 0.8, border_px: float = 2.0) -> list:
    """v2 (refinement, content registration) over the v3 candidates, the registration tried from a format score of
    reg_min_score (v2: 1.5; on the finished world's fouled plates the bumps reaching the plate's left edge leave the
    outline 2-4 px off and the score at 0.9-1.3, from where the registration reaches 3.0); candidates touching the
    frame border (within border_px) are skipped: a plate cut by the frame has no trustworthy outline, and the one
    false read of the finished-world A/B was such a cut quad around a code (P14; v1 / v2 unchanged)."""
    gray = IP.to_gray(frame_rgb)
    Hh, Ww = gray.shape
    src = PS.plate_corners_local()
    out = []
    for quad in _plate_candidates_v3(gray, min_plate_px):
        if quad[:, 0].min() < border_px or quad[:, 1].min() < border_px or quad[:, 0].max() > Ww - border_px \
                or quad[:, 1].max() > Hh - border_px:
            continue
        top = np.hypot(*(quad[1] - quad[0])) + np.hypot(*(quad[2] - quad[3]))
        side = np.hypot(*(quad[3] - quad[0])) + np.hypot(*(quad[2] - quad[1]))
        if top < 0.8 * side:
            continue
        quiet_px = 0.5 * top / PS.PLATE_W * PS.QUIET
        qr = IP.refine_quad(gray, quad, band_px=max(1.0, 0.8 * quiet_px), inside_dark=False)
        tag, q_used = None, qr
        for q in ((qr, quad) if qr is not quad else (quad,)):
            tag = read_tag_with_homography(gray, IP.homography(src, q), min_margin=min_margin, min_font_px=min_font_px)
            if tag is not None:
                q_used = q
                break
        if tag is None and min(top, side) / 2 >= 12.0 and _format_score(gray, IP.homography(src, qr)) >= reg_min_score:
            q_reg, sc = register_quad(gray, qr)
            t2, info = read_tag_with_homography(gray, IP.homography(src, q_reg), details=True, min_margin=min_margin,
                                                min_font_px=min_font_px)
            if t2 is not None and all(b >= 0.6 for _, b, _ in info):
                tag, q_used = t2, q_reg
        if tag is not None:
            c = quad.mean(0)
            if not any(np.hypot(*(c - o[1].mean(0))) < 0.3 * top / 2 for o in out):
                out.append((tag, q_used))
    return out
