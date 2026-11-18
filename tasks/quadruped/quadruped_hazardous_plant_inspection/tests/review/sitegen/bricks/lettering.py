"""The plant's standard lettering (version 2): one glyph table for the tag plates, the room plates, the survey mark's
floor disc, the catalogue's lettering sheet and a reader's templates.

The alphabet is a single-stroke engraving face: capitals A-Z, digits 0-9 and the hyphen (37 glyphs; a space only
advances). Every glyph is a set of polylines on a grid GRID_W = 4 units wide and GRID_H = 8 units high (cap height =
8 units; y up, origin at the glyph cell's lower left), drawn with a round pen PEN = 1.15 units wide, characters
advancing by ADVANCE = 6.4 units (fixed pitch). For a cap height H: unit = H / 8, pen = 1.15 H / 8, pitch = 0.8 H.
The table is the first version 2 study's except D (larger corner cuts on its right) and
Q (a longer tail): the two closest letter pairs, O / Q and D / O, correlated 0.966 and 0.942 at 14 px (blurred by
0.8 px), now 0.935 and 0.923; neither letter was in that study (its tags used C G H I L P S T V only).

Why a rasteriser of our own (numpy only, no font file, no Pillow, no OpenCV drawing): version 1's plate text came
from Pillow's bundled font, whose shapes and hinting change with the Pillow / FreeType version, and OpenCV's
anti-aliased polylines are not specified bit for bit either. Here every pixel's ink is an exact count of SS x SS
sub-samples whose centres lie inside the pen's capsule around a stroke (or inside a ring / disc); the arithmetic is
IEEE-754 float64 +, -, *, / and comparisons on numpy arrays, then integer blending, so the same call gives the same
bytes on every machine and for every library version (guarded by golden sha256 values in
the lettering's own test, checked with numpy 2.5.4 and with numpy 2.5.3)).

Coordinates of the drawing functions: design units (millimetres for plates), image orientation (x right, y DOWN);
pixel (i, j) covers x in [j / sx, (j + 1) / sx), y in [i / sy, (i + 1) / sy). A plate texture's left edge is the
viewer's left (face frame -y), its top edge the plate's top (+z): the same mapping as version 1's tag plates on the
plate box's +x face.

Public interface (sizes in metres unless the name says mm or px; colours sRGB 0-255):

    GLYPHS, LETTERS, DIGITS, CHARSET, GRID_W, GRID_H, PEN, ADVANCE
    TAG_PLATE_W_M 0.1908, TAG_PLATE_H_M 0.0792, TAG_PLATE_T_M 0.002, BORDER_INSET_M 0.004, BORDER_LINE_M 0.0016,
    TAG_CAP_M 0.025 (pen 0.00359375, pitch 0.020), TAG_YELLOW (245, 196, 0), INK (15, 15, 15),
    PLATE_WHITE (238, 238, 230), ROOM_PLATE_W_M 0.40, ROOM_PLATE_H_M 0.10, ROOM_NUMBER_CAP_M 0.045,
    ROOM_NAME_CAP_M 0.020, TAG_MAX_CHARS 8
    tag_ok(text) -> bool; check_tag(text) -> text (ValueError if not the grammar)
    tag_layout(text) -> dict (the tag's cells and ink box in the plate's face frame, metres)
    tag_plate_image(text, px_per_mm=4.0) -> (317, 763, 3) uint8 at 4 px/mm
    room_plate_image(number, name, px_per_mm=2.0) -> (200, 800, 3) uint8 at 2 px/mm
    survey_mark_image(text, n=512) -> (n, n, 3) uint8 (a 0.40 m square: disc, ring, the tag)
    glyph_sheet(cap_px=160) -> uint8 RGB (the catalogue's lettering.png)
    lettering_block() -> dict (the `lettering` block of catalog.json)
    glyph_templates(cap_px, ss=8, pad_px=2) -> {glyph: float32 (h, w) ink coverage 0..1}, one fixed-pitch cell each
    text_coverage(text, cap_px, ...) -> float32 ink coverage of a whole line
    coverage(shapes, width_px, height_px, sx, sy, ss=4) -> int32 counts 0..ss*ss (the rasteriser)
"""

from __future__ import annotations

import math
import re

import numpy as np

__all__ = [
    "GLYPHS", "LETTERS", "DIGITS", "CHARSET", "GRID_W", "GRID_H", "PEN", "ADVANCE",
    "TAG_PLATE_W_M", "TAG_PLATE_H_M", "TAG_PLATE_T_M", "BORDER_INSET_M", "BORDER_LINE_M", "TAG_CAP_M", "TAG_PEN_M",
    "TAG_PITCH_M", "TAG_YELLOW", "INK", "PLATE_WHITE", "ROOM_PLATE_W_M", "ROOM_PLATE_H_M", "ROOM_NUMBER_CAP_M",
    "ROOM_NAME_CAP_M", "TAG_MAX_CHARS", "TAG_PATTERN", "SURVEY_RING_RGB",
    "tag_ok", "check_tag", "tag_layout", "tag_plate_image", "room_plate_image", "room_plate_layout",
    "survey_mark_image", "glyph_sheet", "lettering_block", "glyph_templates", "text_coverage", "coverage",
    "text_segments", "ink_width_units",
]

# ------------------------------------------------------------------------------------------------ the glyph table
GRID_W, GRID_H = 4.0, 8.0
PEN = 1.15            # pen diameter in grid units (cap height / 6.96)
ADVANCE = 6.4         # character pitch in grid units (0.8 x cap height)

_O0 = ((1, 0), (3, 0), (4, 1), (4, 7), (3, 8), (1, 8), (0, 7), (0, 1), (1, 0))
_OO = ((1.5, 0), (2.5, 0), (4, 1.5), (4, 6.5), (2.5, 8), (1.5, 8), (0, 6.5), (0, 1.5), (1.5, 0))

GLYPHS = {
    "0": (_O0,),
    "1": (((0.8, 6.2), (2.4, 8), (2.4, 0)), ((0.8, 0), (4, 0))),
    "2": (((0, 7), (1, 8), (3, 8), (4, 7), (4, 5.2), (0, 0), (4, 0)),),
    "3": (((0, 7), (1, 8), (3, 8), (4, 7), (4, 5), (3, 4), (1.6, 4)), ((3, 4), (4, 3), (4, 1), (3, 0), (1, 0), (0, 1))),
    "4": (((3.1, 0), (3.1, 8), (0, 2.6), (4, 2.6)),),
    "5": (((4, 8), (0, 8), (0, 4.6), (3, 4.6), (4, 3.6), (4, 1), (3, 0), (1, 0), (0, 1)),),
    "6": (((4, 7), (3, 8), (1, 8), (0, 7), (0, 1), (1, 0), (3, 0), (4, 1), (4, 3.6), (3, 4.6), (0, 4.6)),),
    "7": (((0, 8), (4, 8), (1.4, 0)),),
    "8": (((1, 4), (0, 5), (0, 7), (1, 8), (3, 8), (4, 7), (4, 5), (3, 4), (1, 4), (0, 3), (0, 1), (1, 0), (3, 0),
           (4, 1), (4, 3), (3, 4)),),
    "9": (((0, 1), (1, 0), (3, 0), (4, 1), (4, 7), (3, 8), (1, 8), (0, 7), (0, 4.4), (1, 3.4), (4, 3.4)),),
    "-": (((0.7, 4), (3.3, 4)),),
    "A": (((0, 0), (2, 8), (4, 0)), ((0.8, 3), (3.2, 3))),
    "B": (((0, 0), (0, 8), (3, 8), (4, 7), (4, 5), (3, 4), (0, 4)), ((3, 4), (4, 3), (4, 1), (3, 0), (0, 0))),
    "C": (((4, 7), (3, 8), (1, 8), (0, 7), (0, 1), (1, 0), (3, 0), (4, 1)),),
    "D": (((0, 0), (0, 8), (2, 8), (4, 6), (4, 2), (2, 0), (0, 0)),),
    "E": (((4, 8), (0, 8), (0, 0), (4, 0)), ((0, 4), (3, 4))),
    "F": (((4, 8), (0, 8), (0, 0)), ((0, 4), (3, 4))),
    "G": (((4, 7), (3, 8), (1, 8), (0, 7), (0, 1), (1, 0), (3, 0), (4, 1), (4, 3.6), (2.2, 3.6)),),
    "H": (((0, 0), (0, 8)), ((4, 0), (4, 8)), ((0, 4), (4, 4))),
    "I": (((2, 0), (2, 8)), ((0.8, 0), (3.2, 0)), ((0.8, 8), (3.2, 8))),
    "J": (((4, 8), (4, 1), (3, 0), (1, 0), (0, 1), (0, 2.6)),),
    "K": (((0, 0), (0, 8)), ((4, 8), (0, 3.4)), ((1.5, 4.8), (4, 0))),
    "L": (((0, 8), (0, 0), (4, 0)),),
    "M": (((0, 0), (0, 8), (2, 3.6), (4, 8), (4, 0)),),
    "N": (((0, 0), (0, 8), (4, 0), (4, 8)),),
    "O": (_OO,),
    "P": (((0, 0), (0, 8), (3, 8), (4, 7), (4, 5), (3, 4), (0, 4)),),
    "Q": (_OO, ((2, 3), (4, 0))),
    "R": (((0, 0), (0, 8), (3, 8), (4, 7), (4, 5), (3, 4), (0, 4)), ((2, 4), (4, 0))),
    "S": (((4, 7), (3, 8), (1, 8), (0, 7), (0, 5), (1, 4), (3, 4), (4, 3), (4, 1), (3, 0), (1, 0), (0, 1)),),
    "T": (((0, 8), (4, 8)), ((2, 8), (2, 0))),
    "U": (((0, 8), (0, 1), (1, 0), (3, 0), (4, 1), (4, 8)),),
    "V": (((0, 8), (2, 0), (4, 8)),),
    "W": (((0, 8), (1, 0), (2, 5), (3, 0), (4, 8)),),
    "X": (((0, 0), (4, 8)), ((0, 8), (4, 0))),
    "Y": (((0, 8), (2, 4), (4, 8)), ((2, 4), (2, 0))),
    "Z": (((0, 8), (4, 8), (0, 0), (4, 0)),),
}
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DIGITS = "0123456789"
CHARSET = LETTERS + DIGITS + "-"

# ------------------------------------------------------------------------------------------------ the tag plate
TAG_PLATE_W_M, TAG_PLATE_H_M, TAG_PLATE_T_M = 0.1908, 0.0792, 0.002   # version 1's plate outline and thickness
BORDER_INSET_M = 0.004          # the border line's outer edge, inside the plate's edge
BORDER_LINE_M = 0.0016          # the border line's width
TAG_CAP_M = 0.025               # cap height on a tag plate
TAG_PEN_M = PEN * TAG_CAP_M / GRID_H        # 3.594 mm
TAG_PITCH_M = round(ADVANCE * TAG_CAP_M / GRID_H, 9)  # 20 mm
TAG_YELLOW = (245, 196, 0)      # signal yellow (tag plates only)
INK = (15, 15, 15)              # lettering and border line
PLATE_WHITE = (238, 238, 230)   # room plates (and version 1's / grid marker plates)
TAG_MAX_CHARS = 8
TAG_PATTERN = r"^[A-Z]{1,3}-[0-9]{2,5}$"
_TAG_RE = re.compile(TAG_PATTERN)

# ------------------------------------------------------------------------------------------------ room plates, survey mark
ROOM_PLATE_W_M, ROOM_PLATE_H_M = 0.40, 0.10
ROOM_NUMBER_CAP_M, ROOM_NAME_CAP_M = 0.045, 0.020
ROOM_MARGIN_M = 0.014           # from the plate's edge to the first glyph cell
SURVEY_DISC_M = 0.40            # the floor disc's square texture side
SURVEY_RING_RGB = (200, 30, 160)
SURVEY_DISC_RGB = (240, 240, 240)


def tag_ok(text: str) -> bool:
    """The tag grammar: one to three letters, a hyphen, two to five digits, at most TAG_MAX_CHARS characters."""
    return isinstance(text, str) and len(text) <= TAG_MAX_CHARS and _TAG_RE.match(text) is not None


def check_tag(text: str) -> str:
    if not tag_ok(text):
        raise ValueError("not a plant tag (1-3 letters, a hyphen, 2-5 digits, at most %d characters): %r"
                         % (TAG_MAX_CHARS, text))
    return text


def ink_width_units(n_chars: int) -> float:
    """Width of n characters' glyph grids in grid units (first cell's left edge to the last cell's right edge,
    without the pen's half-width on either side)."""
    return 0.0 if n_chars <= 0 else (n_chars - 1) * ADVANCE + GRID_W


# ------------------------------------------------------------------------------------------------ the rasteriser
def coverage(shapes, width_px: int, height_px: int, sx: float, sy: float, ss: int = 4) -> np.ndarray:
    """Exact sub-sample coverage counts (int32, 0..ss*ss) of a set of shapes on a (height_px, width_px) image.

    shapes: a list of tuples in design units (x right, y down); pixel (i, j) covers x in [j / sx, (j + 1) / sx) and
    y in [i / sy, (i + 1) / sy); sub-sample (a, b) of it sits at ((j ss + b + 0.5) / (ss sx), (i ss + a + 0.5) /
    (ss sy)). Kinds:
      ("capsule", ax, ay, bx, by, r)   points within r of the segment a-b (a round pen stroke)
      ("ring", x0, y0, x1, y1, w)      an axis-aligned rectangle outline, outer edge x0..x1 / y0..y1, width w
      ("rect", x0, y0, x1, y1)         a filled axis-aligned rectangle
      ("annulus", cx, cy, r_out, r_in) points with r_in <= distance <= r_out (r_in 0: a disc)
    """
    W, H, ss = int(width_px), int(height_px), int(ss)
    sxs, sys_ = float(sx) * ss, float(sy) * ss
    inside = np.zeros((H * ss, W * ss), bool)

    def window(x_lo, x_hi, y_lo, y_hi):
        J0 = max(0, int(math.floor(x_lo * sxs)) - 1)
        J1 = min(W * ss, int(math.ceil(x_hi * sxs)) + 1)
        I0 = max(0, int(math.floor(y_lo * sys_)) - 1)
        I1 = min(H * ss, int(math.ceil(y_hi * sys_)) + 1)
        if J0 >= J1 or I0 >= I1:
            return None
        xs = (np.arange(J0, J1, dtype=np.float64) + 0.5) / sxs
        ys = (np.arange(I0, I1, dtype=np.float64) + 0.5) / sys_
        return I0, I1, J0, J1, xs[None, :], ys[:, None]

    for sh in shapes:
        kind = sh[0]
        if kind == "capsule":
            ax, ay, bx, by, r = (float(v) for v in sh[1:6])
            w = window(min(ax, bx) - r, max(ax, bx) + r, min(ay, by) - r, max(ay, by) + r)
            if w is None:
                continue
            I0, I1, J0, J1, xs, ys = w
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            px = xs - ax
            py = ys - ay
            if L2 > 0.0:
                t = np.clip((px * dx + py * dy) / L2, 0.0, 1.0)
                ex = px - t * dx
                ey = py - t * dy
            else:
                ex = px + 0.0 * py
                ey = py + 0.0 * px
            inside[I0:I1, J0:J1] |= (ex * ex + ey * ey) <= r * r
        elif kind in ("ring", "rect"):
            x0, y0, x1, y1 = (float(v) for v in sh[1:5])
            w = window(x0, x1, y0, y1)
            if w is None:
                continue
            I0, I1, J0, J1, xs, ys = w
            outer = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
            if kind == "ring":
                lw = float(sh[5])
                inner = (xs > x0 + lw) & (xs < x1 - lw) & (ys > y0 + lw) & (ys < y1 - lw)
                outer = outer & ~inner
            inside[I0:I1, J0:J1] |= outer
        elif kind == "annulus":
            cx, cy, ro, ri = (float(v) for v in sh[1:5])
            w = window(cx - ro, cx + ro, cy - ro, cy + ro)
            if w is None:
                continue
            I0, I1, J0, J1, xs, ys = w
            ex = xs - cx
            ey = ys - cy
            d2 = ex * ex + ey * ey
            inside[I0:I1, J0:J1] |= (d2 <= ro * ro) & (d2 >= ri * ri)
        else:
            raise ValueError("unknown shape %r" % (kind,))
    return inside.reshape(H, ss, W, ss).sum(axis=(1, 3), dtype=np.int32)


def _blend(cov: np.ndarray, n: int, bg, fg, out: np.ndarray | None = None) -> np.ndarray:
    """Integer blend: round((bg (n - c) + fg c) / n) per channel; `out` (uint8 RGB) is painted where c > 0."""
    c = cov.astype(np.int64)
    if out is None:
        out = np.empty(cov.shape + (3,), np.uint8)
        base = np.broadcast_to(np.asarray(bg, np.int64), cov.shape + (3,))
    else:
        base = out.astype(np.int64)
    res = (base * (n - c[..., None]) + np.asarray(fg, np.int64) * c[..., None] + n // 2) // n
    out[...] = res.astype(np.uint8)
    return out


def text_segments(text: str, unit: float, x0: float, baseline: float, advance: float | None = None) -> list:
    """Capsule shapes of `text`: the first glyph cell's lower-left corner at (x0, baseline) in design units (y down),
    `unit` design units per grid unit; characters advance by ADVANCE units (or `advance` design units)."""
    adv = ADVANCE * unit if advance is None else float(advance)
    r = 0.5 * PEN * unit
    out = []
    for k, ch in enumerate(text):
        if ch == " ":
            continue
        g = GLYPHS.get(ch)
        if g is None:
            raise ValueError("no glyph for %r (the lettering has A-Z, 0-9 and the hyphen)" % ch)
        ox = x0 + k * adv
        for stroke in g:
            for (ax, ay), (bx, by) in zip(stroke[:-1], stroke[1:]):
                out.append(("capsule", ox + ax * unit, baseline - ay * unit, ox + bx * unit, baseline - by * unit, r))
    return out


# ------------------------------------------------------------------------------------------------ tag plates
def tag_layout(text: str) -> dict:
    """Where the tag's lettering lies on its plate, in the plate's face frame (metres; y the viewer's right, z up,
    origin at the plate centre). The glyph cells (not the ink) are centred on the plate:
    first cell's left edge y0 = -(ink_width_units(n) * unit) / 2, baseline z = -TAG_CAP_M / 2.

    Returns {'text', 'n', 'unit', 'pen', 'pitch', 'cap', 'cells': [(y_left, y_right, z_bottom, z_top) per glyph cell],
    'ink_box': (y_lo, y_hi, z_lo, z_hi) with the pen, 'border_outer': (half_w, half_h), 'border_inner': (...),
    'border_centre': (half_w, half_h), 'clearance': smallest gap ink to border line (m)}."""
    n = len(text)
    u = TAG_CAP_M / GRID_H
    wt = ink_width_units(n) * u
    y0 = -wt / 2.0
    cells = [(y0 + k * ADVANCE * u, y0 + k * ADVANCE * u + GRID_W * u, -TAG_CAP_M / 2.0, TAG_CAP_M / 2.0)
             for k in range(n)]
    r = TAG_PEN_M / 2.0
    ink = (y0 - r, -y0 + r, -TAG_CAP_M / 2.0 - r, TAG_CAP_M / 2.0 + r)
    bo = (TAG_PLATE_W_M / 2.0 - BORDER_INSET_M, TAG_PLATE_H_M / 2.0 - BORDER_INSET_M)
    bi = (bo[0] - BORDER_LINE_M, bo[1] - BORDER_LINE_M)
    bc = (bo[0] - BORDER_LINE_M / 2.0, bo[1] - BORDER_LINE_M / 2.0)
    clearance = min(bi[0] - ink[1], bi[1] - ink[3])
    return {"text": text, "n": n, "unit": u, "pen": TAG_PEN_M, "pitch": TAG_PITCH_M, "cap": TAG_CAP_M,
            "cells": cells, "ink_box": ink, "border_outer": bo, "border_inner": bi, "border_centre": bc,
            "clearance": clearance}


def _tag_shapes(text: str) -> list:
    """The tag plate's ink in millimetres (x right from the plate's left edge, y down from its top edge)."""
    W, H = TAG_PLATE_W_M * 1000.0, TAG_PLATE_H_M * 1000.0
    a, lw = BORDER_INSET_M * 1000.0, BORDER_LINE_M * 1000.0
    u = TAG_CAP_M * 1000.0 / GRID_H
    x0 = (W - ink_width_units(len(text)) * u) / 2.0
    yb = H / 2.0 + GRID_H * u / 2.0
    return [("ring", a, a, W - a, H - a, lw)] + text_segments(text, u, x0, yb)


def tag_plate_image(text: str, px_per_mm: float = 4.0, ss: int = 4, check: bool = True,
                    plate_rgb=TAG_YELLOW, ink_rgb=INK) -> np.ndarray:
    """RGB uint8 texture of a text tag plate: signal yellow, a black border line 1.6 mm wide 4.0 mm inside the edge,
    the tag in the plant lettering (25 mm cap height, 20 mm pitch) centred on one line. At 4 px/mm: 763 x 317 px
    (columns = the plate's 190.8 mm, rows = its 79.2 mm; each axis scaled exactly to the plate). `check`: refuse a
    text outside the tag grammar (False lets a study draw any string of the charset)."""
    if check:
        check_tag(text)
    W = int(round(TAG_PLATE_W_M * 1000.0 * px_per_mm))
    H = int(round(TAG_PLATE_H_M * 1000.0 * px_per_mm))
    sx, sy = W / (TAG_PLATE_W_M * 1000.0), H / (TAG_PLATE_H_M * 1000.0)
    cov = coverage(_tag_shapes(text), W, H, sx, sy, ss)
    return _blend(cov, ss * ss, plate_rgb, ink_rgb)


# ------------------------------------------------------------------------------------------------ room plates
def room_plate_layout(number: str, name: str) -> dict:
    """The room plate's lettering in millimetres (x from the plate's left edge, y down from its top): the room number
    at 45 mm cap height from the left margin, the name at 20 mm to its right, both centred vertically; a name that
    does not fit on one line is split at the space nearest its middle onto two lines (line pitch 1.5 cap heights).
    Without a number the name is centred. Returns {'number': (x0, baseline, unit), 'name_lines': [(text, x0,
    baseline, unit)], 'fits': bool}."""
    W, H = ROOM_PLATE_W_M * 1000.0, ROOM_PLATE_H_M * 1000.0
    m = ROOM_MARGIN_M * 1000.0
    un = ROOM_NUMBER_CAP_M * 1000.0 / GRID_H
    um = ROOM_NAME_CAP_M * 1000.0 / GRID_H
    out = {"number": None, "name_lines": [], "fits": True}
    x_name = m
    if number:
        out["number"] = (m, H / 2.0 + GRID_H * un / 2.0, un)
        x_name = m + ink_width_units(len(number)) * un + 0.5 * ADVANCE * un
    avail = W - m - x_name
    name = " ".join(str(name).split())
    if not name:
        return out
    lines = [name]
    if ink_width_units(len(name)) * um > avail and " " in name:
        cut = min((i for i, ch in enumerate(name) if ch == " "), key=lambda i: abs(i - len(name) / 2.0))
        lines = [name[:cut], name[cut + 1:]]
    pitch_y = 1.5 * GRID_H * um
    yb0 = H / 2.0 + GRID_H * um / 2.0 - (len(lines) - 1) * pitch_y / 2.0
    for k, ln in enumerate(lines):
        w = ink_width_units(len(ln)) * um
        if w > avail:
            out["fits"] = False
        x0 = x_name if number else (W - w) / 2.0
        out["name_lines"].append((ln, x0, yb0 + k * pitch_y, um))
    return out


def room_plate_image(number: str, name: str, px_per_mm: float = 2.0, ss: int = 4) -> np.ndarray:
    """RGB uint8 texture of a room plate (0.40 x 0.10 m, white, black border line and lettering): the number at
    45 mm cap height and the name at 20 mm. At 2 px/mm: 800 x 200 px. Raises ValueError if the text does
    not fit or uses a character outside the lettering (letters, digits, hyphen, space)."""
    lay = room_plate_layout(str(number or ""), str(name or ""))
    if not lay["fits"]:
        raise ValueError("room plate text does not fit: %r %r" % (number, name))
    Wmm, Hmm = ROOM_PLATE_W_M * 1000.0, ROOM_PLATE_H_M * 1000.0
    W, H = int(round(Wmm * px_per_mm)), int(round(Hmm * px_per_mm))
    a, lw = BORDER_INSET_M * 1000.0, BORDER_LINE_M * 1000.0
    shapes = [("ring", a, a, Wmm - a, Hmm - a, lw)]
    if lay["number"] is not None:
        x0, yb, u = lay["number"]
        shapes += text_segments(str(number), u, x0, yb)
    for ln, x0, yb, u in lay["name_lines"]:
        shapes += text_segments(ln, u, x0, yb)
    cov = coverage(shapes, W, H, W / Wmm, H / Hmm, ss)
    return _blend(cov, ss * ss, PLATE_WHITE, INK)


# ------------------------------------------------------------------------------------------------ the survey mark
def survey_mark_image(text: str, n: int = 512, ss: int = 4, corner_rgb=(0, 0, 0)) -> np.ndarray:
    """RGB uint8 texture (n x n) of the survey mark's floor disc on a SURVEY_DISC_M square: a light disc of radius
    0.194 m with a ring (0.166-0.194 m, SURVEY_RING_RGB, as version 1), the mark's tag in the plant lettering in the
    middle at up to 48 mm cap height (smaller if a longer text needs it). Corners `corner_rgb`."""
    S = SURVEY_DISC_M * 1000.0
    c = S / 2.0
    r_out, r_ring = 0.4845 * S, 0.415 * S
    sx = n / S
    disc = coverage([("annulus", c, c, r_out, 0.0)], n, n, sx, sx, ss)
    img = _blend(disc, ss * ss, corner_rgb, SURVEY_DISC_RGB)
    ring = coverage([("annulus", c, c, r_out, r_ring)], n, n, sx, sx, ss)
    img = _blend(ring, ss * ss, None, SURVEY_RING_RGB, out=img)
    if text:
        cap = 48.0
        clear = r_ring - 12.0
        while cap > 10.0:
            u = cap / GRID_H
            w = ink_width_units(len(text)) * u + PEN * u
            h = cap + PEN * u
            if math.hypot(w / 2.0, h / 2.0) <= clear:
                break
            cap -= 1.0
        u = cap / GRID_H
        x0 = c - ink_width_units(len(text)) * u / 2.0
        ink = coverage(text_segments(text, u, x0, c + cap / 2.0), n, n, sx, sx, ss)
        img = _blend(ink, ss * ss, None, INK, out=img)
    return img


# ------------------------------------------------------------------------------------------------ catalogue
def glyph_sheet(cap_px: float = 160.0, ss: int = 4) -> np.ndarray:
    """The catalogue's lettering sheet: every glyph at `cap_px` of cap height, black on white, each in its 4 x 8
    grid cell outlined in light grey (rows: digits and the hyphen; A-M; N-Z)."""
    u = cap_px / GRID_H
    rows = [DIGITS + "-", LETTERS[:13], LETTERS[13:]]
    step = ADVANCE * u * 1.25
    marg = 3.0 * u
    W = int(math.ceil(2 * marg + 12 * step + GRID_W * u))
    rh = cap_px * 1.6
    H = int(math.ceil(2 * marg + len(rows) * rh - (rh - cap_px)))
    img = np.full((H, W, 3), 255, np.uint8)
    grid = []
    for r, row in enumerate(rows):
        yb = marg + r * rh + cap_px
        for k in range(len(row)):
            x0 = marg + k * step
            grid.append(("ring", x0, yb - cap_px, x0 + GRID_W * u, yb, max(1.0, u * 0.08)))
    img = _blend(coverage(grid, W, H, 1.0, 1.0, ss), ss * ss, None, (200, 200, 200), out=img)
    ink = []
    for r, row in enumerate(rows):
        ink += text_segments(row, u, marg, marg + r * rh + cap_px, advance=step)
    return _blend(coverage(ink, W, H, 1.0, 1.0, ss), ss * ss, None, (0, 0, 0), out=img)


def lettering_block() -> dict:
    """The `lettering` block of catalog.json: everything a program needs to draw the plant's lettering and a tag
    plate at any size (numbers only; units stated in each key)."""
    def r6(v):
        return float(round(float(v), 6))

    glyphs = {ch: [[[r6(x), r6(y)] for x, y in stroke] for stroke in GLYPHS[ch]] for ch in CHARSET}
    return {
        "grid_units": {"width": GRID_W, "cap_height": GRID_H,
                       "origin": "lower left corner of the glyph cell, x to the right, y up"},
        "pen_units": PEN,
        "pen_shape": "round (a stroke is every point within half the pen of the polyline)",
        "pitch_units": ADVANCE,
        "space": "advances one pitch, no ink",
        "glyphs": glyphs,
        "tag_grammar": {"pattern": TAG_PATTERN, "max_chars": TAG_MAX_CHARS,
                        "words": "one to three capital letters, a hyphen, two to five digits"},
        "tag_plate": {
            "size_mm": [r6(TAG_PLATE_W_M * 1000), r6(TAG_PLATE_H_M * 1000)],
            "thickness_mm": r6(TAG_PLATE_T_M * 1000),
            "plate_rgb": list(TAG_YELLOW), "ink_rgb": list(INK),
            "border_line": {"outer_edge_inset_mm": r6(BORDER_INSET_M * 1000), "width_mm": r6(BORDER_LINE_M * 1000)},
            "cap_height_mm": r6(TAG_CAP_M * 1000), "pen_mm": r6(TAG_PEN_M * 1000),
            "pitch_mm": r6(TAG_PITCH_M * 1000),
            "text_placement": "one line; the glyph cells (not the ink) are centred on the plate: the first cell's "
                              "left edge is half of ((n - 1) x pitch + 4 units) left of the plate's centre and the "
                              "cells' bottom edge half the cap height below it",
        },
        "room_plate": {
            "size_mm": [r6(ROOM_PLATE_W_M * 1000), r6(ROOM_PLATE_H_M * 1000)],
            "plate_rgb": list(PLATE_WHITE), "ink_rgb": list(INK),
            "border_line": {"outer_edge_inset_mm": r6(BORDER_INSET_M * 1000), "width_mm": r6(BORDER_LINE_M * 1000)},
            "number_cap_height_mm": r6(ROOM_NUMBER_CAP_M * 1000), "name_cap_height_mm": r6(ROOM_NAME_CAP_M * 1000),
            "text_placement": "the room number from %.0f mm inside the left edge, the name half a number pitch after "
                              "it (two lines when it is long), both centred vertically" % (ROOM_MARGIN_M * 1000),
        },
    }


# ------------------------------------------------------------------------------------------------ reader templates
def text_coverage(text: str, cap_px: float, pad_px: float = 2.0, ss: int = 8, advance_px: float | None = None):
    """Ink coverage (float32 0..1) of one line of text at `cap_px` cap height: the first glyph cell's left edge
    `pad_px + pen/2` from the image's left, the cells' bottom `pad_px + pen/2` above the image's bottom.
    Returns (image, (x0, baseline)) in pixel coordinates (continuous, y down)."""
    u = cap_px / GRID_H
    r = 0.5 * PEN * u
    adv = ADVANCE * u if advance_px is None else float(advance_px)
    n = len(text)
    w = (n - 1) * adv + GRID_W * u if n else 0.0
    W = int(math.ceil(w + 2 * (pad_px + r)))
    H = int(math.ceil(cap_px + 2 * (pad_px + r)))
    x0 = pad_px + r
    yb = H - pad_px - r
    cov = coverage(text_segments(text, u, x0, yb, advance=adv), W, H, 1.0, 1.0, ss)
    return cov.astype(np.float32) / float(ss * ss), (x0, yb)


def glyph_templates(cap_px: float, ss: int = 8, pad_px: float = 2.0) -> dict:
    """{glyph: float32 ink coverage 0..1} for every glyph of CHARSET at `cap_px` cap height, each on the same cell:
    width = pitch (ADVANCE units) + 2 pad, height = cap + pen + 2 pad, the glyph's 4-unit grid centred in the pitch.
    Template (row, col) <-> cell position: the glyph grid's lower-left corner is at
    (x = pad + (ADVANCE - GRID_W) / 2 units, y = pad + pen/2 + cap) from the cell's top-left corner."""
    u = cap_px / GRID_H
    r = 0.5 * PEN * u
    Wc = int(math.ceil(ADVANCE * u + 2 * pad_px))
    Hc = int(math.ceil(cap_px + 2 * r + 2 * pad_px))
    x0 = pad_px + 0.5 * (ADVANCE - GRID_W) * u
    yb = pad_px + r + cap_px
    out = {}
    for ch in CHARSET:
        cov = coverage(text_segments(ch, u, x0, yb), Wc, Hc, 1.0, 1.0, ss)
        out[ch] = cov.astype(np.float32) / float(ss * ss)
    return out
