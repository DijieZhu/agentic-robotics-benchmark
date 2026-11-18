"""Panel standard used by the G4 optics prototype (pure python / numpy; importable by pilots and reference programs).

Everything here is information the agent-facing panel standard would publish: plate layout, code grid, painted-text font
and tag format, handle geometry and paint, camera intrinsics convention.  No site-specific quantity lives here.

Panel-local frame of one valve position (metres):
  origin = the valve-stem axis on the panel face; X = to the right as seen by a viewer facing the panel; Y = up;
  Z = out of the panel, towards the viewer.

Plate (one per valve / port), centre 0.22 m above the stem, flush on the panel face:
  [quiet 20 mm][ marker 160 mm ][gap 30 mm][ painted tag, 3 characters, 80 mm high ][quiet 20 mm]
  marker = 8 x 8 cells of 20 mm: a one-cell black border around 6 x 6 data cells (bit 1 = black).
  The 6 x 6 data grid is the "120 mm, 6 x 6 cells of 20 mm" of the design; the border adds one cell per side.

Image convention: continuous pixel coordinates, pixel (col i, row j) covers [i, i+1) x [j, j+1), centre (i+0.5, j+0.5).
Dome port: rectilinear pinhole, f = (W/2) / tan(hfov/2), principal point (W/2, H/2), square pixels.
"""
from __future__ import annotations

import math
import os

import numpy as np

# ------------------------------------------------------------------------------------------------ code / plate layout
# P14: fixed 20 mm (G4 read the cell size from the environment variable G4_CELL_MM for its 25 mm fallback, unused)
CELL = 0.020                     # code cell edge [m]
DATA_N = 6                       # data cells per side
BORDER = 1                       # black border, in cells
MARKER_N = DATA_N + 2 * BORDER   # 8 cells across
MARKER = MARKER_N * CELL         # 0.160 m outer black square
QUIET = CELL                     # white margin around the marker and at the plate edges (one cell)
GAP_CODE_TEXT = 0.030

TEXT_H = 0.080                   # painted character height
FONT_ROWS, FONT_COLS = 7, 5
FONT_CELL = TEXT_H / FONT_ROWS
CHAR_GAP_CELLS = 1
N_CHARS = 3
TEXT_W = N_CHARS * FONT_COLS * FONT_CELL + (N_CHARS - 1) * CHAR_GAP_CELLS * FONT_CELL

PLATE_W = QUIET + MARKER + GAP_CODE_TEXT + TEXT_W + QUIET
PLATE_H = QUIET + MARKER + QUIET
PLATE_THICK = 0.004
PLATE_ABOVE_STEM = 0.22          # plate centre above the stem axis

PLATE_CENTER = np.array([0.0, PLATE_ABOVE_STEM])
CODE_CENTER = np.array([-PLATE_W / 2 + QUIET + MARKER / 2, PLATE_ABOVE_STEM])
TEXT_LEFT = -PLATE_W / 2 + QUIET + MARKER + GAP_CODE_TEXT        # X of the text block's left edge
TEXT_CENTER_Y = PLATE_ABOVE_STEM

# ------------------------------------------------------------------------------------------------ handle
STEM_OUT = 0.15                  # blade centre out from the panel face
BLADE_L, BLADE_T, BLADE_D = 0.160, 0.012, 0.060   # length, thickness, depth along the stem
STEM_R = 0.015
# two-colour paint: the half that points to the viewer's right when open (blade horizontal) is orange; turning clockwise
# (as seen facing the panel) by 90 deg closes the valve and brings the orange half to the bottom (blade vertical).
PAINT_ORANGE = (1.00, 0.45, 0.05)
PAINT_WHITE = (0.92, 0.92, 0.90)

# ------------------------------------------------------------------------------------------------ panel grid
COL_PITCH = 0.60
ROW_PITCH = 0.60
ROW_HEIGHTS = (1.50, 2.10, 2.70)          # stem height above the mud mat for rows BV, PV, TV
ROW_LETTERS = ("B", "P", "T")
N_COLUMNS = 6

# ------------------------------------------------------------------------------------------------ font and tags
_GLYPHS = {
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
}
# tag format: <row letter> 'V' <column digit 1..6>, e.g. "PV3"
TAG_ALPHABETS = (("B", "P", "T"), ("V",), tuple(str(i) for i in range(1, N_COLUMNS + 1)))
READ_ALPHABET = tuple(sorted(_GLYPHS))     # the reader matches every glyph, then checks the format


def glyph(ch: str) -> np.ndarray:
    """(7, 5) float array, 1 = ink."""
    return np.array([[int(c) for c in row] for row in _GLYPHS[ch]], dtype=np.float64)


def valid_tags() -> list[str]:
    return [r + "V" + str(c) for r in ROW_LETTERS for c in range(1, N_COLUMNS + 1)]


def tag_is_valid(tag: str) -> bool:
    return len(tag) == 3 and all(ch in alph for ch, alph in zip(tag, TAG_ALPHABETS))


def text_cell_rects(tag_len: int = N_CHARS):
    """Yield (char_index, row, col, x0, y0) of each font cell's lower-left corner in panel-local X/Y."""
    top = TEXT_CENTER_Y + TEXT_H / 2
    for k in range(tag_len):
        x_char = TEXT_LEFT + k * (FONT_COLS + CHAR_GAP_CELLS) * FONT_CELL
        for r in range(FONT_ROWS):
            for c in range(FONT_COLS):
                yield k, r, c, x_char + c * FONT_CELL, top - (r + 1) * FONT_CELL


def code_cell_center(i: int, j: int) -> np.ndarray:
    """Centre (X, Y) of marker cell (row i from the top, column j from the left), i, j in -1 .. MARKER_N."""
    x0 = CODE_CENTER[0] - MARKER / 2
    y0 = CODE_CENTER[1] + MARKER / 2
    return np.array([x0 + (j + 0.5) * CELL, y0 - (i + 0.5) * CELL])


def marker_corners_local() -> np.ndarray:
    """Marker outer corners (TL, TR, BR, BL) in panel-local X/Y."""
    cx, cy = CODE_CENTER
    h = MARKER / 2
    return np.array([[cx - h, cy + h], [cx + h, cy + h], [cx + h, cy - h], [cx - h, cy - h]])


def plate_corners_local() -> np.ndarray:
    cx, cy = PLATE_CENTER
    w, h = PLATE_W / 2, PLATE_H / 2
    return np.array([[cx - w, cy + h], [cx + w, cy + h], [cx + w, cy - h], [cx - w, cy - h]])


# ------------------------------------------------------------------------------------------------ camera intrinsics
def intrinsics(width: int, height: int, hfov_deg: float) -> np.ndarray:
    """3x3 K of the dome-port camera in the continuous pixel convention above (v grows downwards)."""
    f = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])
