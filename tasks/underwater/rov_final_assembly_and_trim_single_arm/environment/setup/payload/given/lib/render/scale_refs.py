"""Scale references placed as PHYSICAL objects in the scene (never 2D overlays), with procedural textures.

Constructors (return a spec.ScaleRef; sizes are the NOMINAL everyday sizes a human knows):

    ruler(name, pos, yaw_deg, length_mm=300)                      300 mm steel rule, 1 / 5 / 10 mm ticks, numbered per cm
    checkerboard(name, pos, yaw_deg, cols=10, rows=7, square_mm=20)   printed target on a 3 mm board, white margin
    drink_can(name, pos, yaw_deg)                                 66 mm diameter x 122 mm tall beverage can
    id_card(name, pos, yaw_deg)                                   ISO/IEC 7810 ID-1 card 85.60 x 53.98 x 0.76 mm, r 3.18 mm
    a4_sheet(name, pos, yaw_deg)                                  ISO 216 A4 sheet 210 x 297 x 0.1 mm
    cutting_mat(name, pos, yaw_deg, size_mm=(450, 300), grid_mm=10, bold_every=5)   self-healing mat, printed grid, 2 mm

Backend-agnostic geometry and facts:

    parts(ref)        -> [RefPart]  meshes (with uv + texture array) and primitives in the REF frame; every backend
                         (MuJoCo here; USD for a ray-traced renderer, isaac_backend.py) builds the same parts
    nominal(ref)      -> the PUBLIC description (kind, nominal dimensions, what is printed on it): photos.json
    features(ref)     -> exact feature points in WORLD coordinates (checkerboard inner corners, ruler ticks, card corners,
                         can rims, sheet corners): GRADER-ONLY (photos_gt.json) and used by qc.py / the tests
    add_to_spec(mjspec, ref) -> {body, geoms, materials, textures, meshes}   the MuJoCo builder

Ref frame: origin at the centre of the face the object rests on, +z up out of it; the ruler's scale runs along +x
(0 mm at x = -length/2), its ticks hang from the +y edge; the checkerboard's columns run along +x and rows along +y
with the (-x, +y) corner square black; the card's chip is on the -x half; the cutting mat's grid is centred, its cm
numbers run along the +y and -x margins.  Textures are generated here from numbers only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from . import meshes as M
from . import textures as T
from .spec import ScaleRef, rng as _rng

MM = 1e-3


def yaw_quat(yaw_deg: float) -> List[float]:
    a = math.radians(yaw_deg) / 2.0
    return [math.cos(a), 0.0, 0.0, math.sin(a)]


def quat_to_R(q) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def pose_matrix(ref: ScaleRef) -> np.ndarray:
    Tm = np.eye(4)
    Tm[:3, :3] = quat_to_R(ref.quat)
    Tm[:3, 3] = ref.pos
    return Tm


def to_world(ref: ScaleRef, pts_local) -> np.ndarray:
    Tm = pose_matrix(ref)
    P = np.atleast_2d(np.asarray(pts_local, float))
    return P @ Tm[:3, :3].T + Tm[:3, 3]


# ----------------------------------------------------------------------------- parts
@dataclass
class RefPart:
    """One renderable piece of a scale reference, in the ref frame (metres)."""
    name: str
    kind: str                                   # "mesh" | "box" | "cylinder"
    mesh: Optional[M.Mesh] = None
    size: Optional[List[float]] = None          # box: full extents (x y z); cylinder: (radius, full height)
    pos: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rgba: List[float] = field(default_factory=lambda: [0.8, 0.8, 0.8, 1.0])
    texture: Optional[np.ndarray] = None        # uint8 H x W x 3 (mesh parts with uv only)
    specular: float = 0.3
    shininess: float = 0.3
    reflectance: float = 0.0


DEFAULTS = {
    "ruler": dict(length_mm=300.0, end_mm=8.0, width_mm=25.0, thickness_mm=0.9, px_per_mm=8, tick_mm=(3.0, 4.5, 6.5),
                  tick_width_mm=0.22, numeral_height_mm=3.0),
    "checkerboard": dict(cols=10, rows=7, square_mm=20.0, margin_mm=None, thickness_mm=3.0, px_per_square=48),
    "can": dict(diameter_mm=66.0, height_mm=122.0, palette=0),
    "card": dict(width_mm=85.60, height_mm=53.98, thickness_mm=0.76, corner_radius_mm=3.18, palette=0),
    "a4": dict(width_mm=210.0, height_mm=297.0, thickness_mm=0.1),
    "cutting_mat": dict(width_mm=450.0, height_mm=300.0, grid_mm=10.0, bold_every=5, margin_mm=15.0, thickness_mm=2.0,
                        colour="green", line_mm=0.35, bold_line_mm=0.7, px_per_mm=4),
}


def _params(ref: ScaleRef) -> Dict[str, Any]:
    if ref.kind not in DEFAULTS:
        raise ValueError("unknown scale reference kind %r (have %s)" % (ref.kind, sorted(DEFAULTS)))
    p = dict(DEFAULTS[ref.kind])
    unknown = set(ref.params) - set(p)
    if unknown:
        raise ValueError("%s: unknown params %s" % (ref.kind, sorted(unknown)))
    p.update(ref.params)
    if ref.kind == "checkerboard" and p["margin_mm"] is None:
        p["margin_mm"] = p["square_mm"]
    return p


# ---- ruler
def _ruler_parts(ref, p):
    L = p["length_mm"] + 2 * p["end_mm"]
    Wd, th = p["width_mm"], p["thickness_mm"]
    k = int(p["px_per_mm"])
    H, W = int(round(Wd * k)), int(round(L * k))
    r = _rng(ref.seed, "ruler", ref.name)
    img = T.brushed(H, W, rgb1=(0.74, 0.75, 0.77), rng=r, strength=0.05)
    ink = (0.04, 0.04, 0.05)
    tw = p["tick_width_mm"]
    t1, t5, t10 = p["tick_mm"]
    for s in range(int(round(p["length_mm"])) + 1):
        x = (p["end_mm"] + s) * k
        ln = t10 if s % 10 == 0 else (t5 if s % 5 == 0 else t1)
        T.fill_rect(img, x - tw * k / 2, 0, x + tw * k / 2, ln * k, ink)          # row 0 = the +y edge (v = 0)
        if s % 10 == 0:
            T.draw_text(img, str(s // 10), x, (t10 + 1.0) * k, p["numeral_height_mm"] * k, ink, align="center")
    T.draw_text(img, "mm", (p["end_mm"] + p["length_mm"] - 14) * k, (Wd - 5.5) * k, 2.5 * k, ink)
    body = RefPart("body", "box", size=[L * MM, Wd * MM, th * MM], pos=[0, 0, th * MM / 2], rgba=[0.62, 0.63, 0.65, 1],
                   specular=0.6, shininess=0.6)
    top = RefPart("face", "mesh", mesh=M.quad(L * MM, Wd * MM, z=th * MM + 0.05 * MM), texture=T.to_uint8(img),
                  rgba=[1, 1, 1, 1], specular=0.5, shininess=0.5)
    return [body, top]


def _ruler_features(ref, p):
    Wd, th = p["width_mm"], p["thickness_mm"]
    s = np.arange(int(round(p["length_mm"])) + 1, dtype=float)
    x = (-p["length_mm"] / 2 + s) * MM
    z = (th + 0.05) * MM
    edge = np.c_[x, np.full_like(x, Wd / 2 * MM), np.full_like(x, z)]
    mid = np.c_[x, np.full_like(x, (Wd / 2 - p["tick_mm"][0] / 2) * MM), np.full_like(x, z)]
    return {"tick_roots_mm_0_to_%d" % int(p["length_mm"]): to_world(ref, edge),
            "tick_midpoints_1mm": to_world(ref, mid),
            "top_face_corners": to_world(ref, [[-(p["length_mm"] / 2 + p["end_mm"]) * MM, sgn * Wd / 2 * MM, z]
                                               for sgn in (1, -1)] + [[(p["length_mm"] / 2 + p["end_mm"]) * MM, sgn * Wd / 2 * MM, z] for sgn in (-1, 1)])}


# ---- checkerboard
def _board_dims(p):
    return p["cols"] * p["square_mm"] + 2 * p["margin_mm"], p["rows"] * p["square_mm"] + 2 * p["margin_mm"]


def _checker_parts(ref, p):
    bx, by = _board_dims(p)
    th = p["thickness_mm"]
    k = int(p["px_per_square"]) / p["square_mm"]            # texels per mm
    W, H = int(round(bx * k)), int(round(by * k))
    white, black = (0.93, 0.93, 0.92), (0.07, 0.07, 0.08)
    img = T.flat(H, W, white)
    m = p["margin_mm"] * k
    sq = p["square_mm"] * k
    for j in range(p["rows"]):            # row 0 at the +y edge (v = 0)
        for i in range(p["cols"]):
            if (i + j) % 2 == 0:
                T.fill_rect(img, m + i * sq, m + j * sq, m + (i + 1) * sq, m + (j + 1) * sq, black)
    board = RefPart("board", "box", size=[bx * MM, by * MM, th * MM], pos=[0, 0, th * MM / 2], rgba=[0.9, 0.9, 0.88, 1],
                    specular=0.1, shininess=0.1)
    face = RefPart("print", "mesh", mesh=M.quad(bx * MM, by * MM, z=th * MM + 0.05 * MM), texture=T.to_uint8(img),
                   rgba=[1, 1, 1, 1], specular=0.05, shininess=0.1)
    return [board, face]


def _checker_features(ref, p):
    sq, th = p["square_mm"], p["thickness_mm"]
    z = (th + 0.05) * MM
    pts = []
    for j in range(1, p["rows"]):            # row-major from the +y edge, then along +x (OpenCV-like ordering)
        for i in range(1, p["cols"]):
            pts.append([(-p["cols"] / 2 + i) * sq * MM, (p["rows"] / 2 - j) * sq * MM, z])
    bx, by = _board_dims(p)
    corners = [[-bx / 2 * MM, by / 2 * MM, z], [bx / 2 * MM, by / 2 * MM, z], [bx / 2 * MM, -by / 2 * MM, z], [-bx / 2 * MM, -by / 2 * MM, z]]
    return {"inner_corners": to_world(ref, pts), "inner_corners_shape": [p["rows"] - 1, p["cols"] - 1],
            "board_corners": to_world(ref, corners)}


# ---- can
_CAN_PALETTES = [((0.78, 0.10, 0.12), (0.95, 0.95, 0.95), (0.10, 0.10, 0.12)),
                 ((0.10, 0.35, 0.70), (0.95, 0.80, 0.20), (0.95, 0.95, 0.95)),
                 ((0.15, 0.55, 0.30), (0.95, 0.95, 0.90), (0.95, 0.75, 0.10))]


def _can_profile(p):
    R, Hh = p["diameter_mm"] / 2, p["height_mm"]
    # (r, z) mm bottom to top; max radius R on the wall, rim top at Hh
    return [(0.72 * R, 1.6), (0.80 * R, 0.0), (0.94 * R, 3.5), (R, 8.0), (R, Hh - 18.0), (0.955 * R, Hh - 12.0),
            (0.86 * R, Hh - 5.5), (0.835 * R, Hh - 2.0), (0.835 * R, Hh), (0.80 * R, Hh), (0.79 * R, Hh - 1.6)]


def _can_parts(ref, p):
    R, Hh = p["diameter_mm"] / 2, p["height_mm"]
    prof = _can_profile(p)
    c1, c2, c3 = _CAN_PALETTES[int(p["palette"]) % len(_CAN_PALETTES)]
    Wt, Ht = 1024, 512
    img = T.flat(Ht, Wt, (0.78, 0.79, 0.81))
    z_to_row = lambda z: (Hh - z) / Hh * Ht  # noqa: E731
    lab_top, lab_bot = z_to_row(Hh - 18.0), z_to_row(8.0)
    T.fill_rect(img, 0, lab_top, Wt, lab_bot, c1)
    r = _rng(ref.seed, "can", ref.name)
    band_y = lab_top + (lab_bot - lab_top) * (0.30 + 0.1 * r.random())
    xs = np.arange(Wt)
    wave = band_y + 18 * np.sin(2 * np.pi * xs / Wt * 2 + r.random() * 6.28)
    for x in xs:
        T.fill_rect(img, x, wave[x], x + 1, wave[x] + 46, c2)
    for k in range(3):
        cx = Wt * (0.15 + 0.33 * k)
        T.fill_rect(img, cx - 60, lab_bot - 120, cx + 60, lab_bot - 60, c3)
    wall = RefPart("wall", "mesh", mesh=M.revolve([(a * MM, b * MM) for a, b in prof], n_seg=72, z_top=Hh * MM, z_bottom=0.0),
                   texture=T.to_uint8(img), rgba=[1, 1, 1, 1], specular=0.7, shininess=0.7)
    lid = RefPart("lid", "mesh", mesh=M.disc(0.79 * R * MM, (Hh - 1.6) * MM, n_seg=72), rgba=[0.70, 0.71, 0.73, 1],
                  specular=0.8, shininess=0.8)
    tab = RefPart("tab", "box", size=[16 * MM, 9 * MM, 0.6 * MM], pos=[4 * MM, 0, (Hh - 1.3) * MM], rgba=[0.62, 0.63, 0.65, 1],
                  specular=0.8, shininess=0.8)
    bottom = RefPart("bottom", "mesh", mesh=M.disc(0.72 * R * MM, 1.6 * MM, n_seg=72, up=False), rgba=[0.70, 0.71, 0.73, 1])
    return [wall, lid, tab, bottom]


def _can_features(ref, p):
    R, Hh = p["diameter_mm"] / 2, p["height_mm"]
    a = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    rim = np.c_[0.835 * R * np.cos(a), 0.835 * R * np.sin(a), np.full_like(a, Hh)] * MM
    wall_top = np.c_[R * np.cos(a), R * np.sin(a), np.full_like(a, Hh - 18.0)] * MM
    wall_bot = np.c_[R * np.cos(a), R * np.sin(a), np.full_like(a, 8.0)] * MM
    return {"top_rim": to_world(ref, rim), "wall_top_circle": to_world(ref, wall_top), "wall_bottom_circle": to_world(ref, wall_bot),
            "axis_bottom_top": to_world(ref, [[0, 0, 0], [0, 0, Hh * MM]])}


# ---- card
_CARD_PALETTES = [((0.05, 0.30, 0.45), (0.10, 0.55, 0.60)), ((0.35, 0.08, 0.30), (0.75, 0.20, 0.35)),
                  ((0.12, 0.12, 0.14), (0.35, 0.36, 0.40))]


def _card_parts(ref, p):
    w, h, th, rc = p["width_mm"], p["height_mm"], p["thickness_mm"], p["corner_radius_mm"]
    k = 10
    W, H = int(round(w * k)), int(round(h * k))
    c1, c2 = _CARD_PALETTES[int(p["palette"]) % len(_CARD_PALETTES)]
    yy, xx = np.mgrid[0:H, 0:W]
    t = ((xx / W) * 0.6 + (yy / H) * 0.4)[..., None]
    img = np.asarray(c1)[None, None, :] * (1 - t) + np.asarray(c2)[None, None, :] * t
    gold, line = (0.83, 0.68, 0.33), (0.55, 0.43, 0.18)
    x0, y0 = 10.25 * k, 19.23 * k                       # ISO 7816-2 contact area position (from the left / top edges)
    T.fill_rect(img, x0, y0, x0 + 11.4 * k, y0 + 9.0 * k, gold)
    for yl in (3.0, 6.0):
        T.fill_rect(img, x0, y0 + yl * k - 1, x0 + 11.4 * k, y0 + yl * k + 1, line)
    T.fill_rect(img, x0 + 5.5 * k, y0, x0 + 5.9 * k, y0 + 9.0 * k, line)
    for g in range(4):                                  # four groups of blank "digit" plates (no characters)
        gx = 8.0 * k + g * 18.0 * k
        for d in range(4):
            T.fill_rect(img, gx + d * 3.6 * k, 36.0 * k, gx + d * 3.6 * k + 2.8 * k, 40.0 * k, (0.85, 0.86, 0.88), alpha=0.8)
    T.fill_rect(img, 8.0 * k, 45.0 * k, 45.0 * k, 47.0 * k, (0.85, 0.86, 0.88), alpha=0.6)
    outline = M.rounded_rect(w * MM, h * MM, rc * MM, n_arc=6)
    core = RefPart("core", "mesh", mesh=M.prism(outline, 0.0, th * MM, top=False), rgba=list(c1) + [1.0], specular=0.3, shininess=0.4)
    face = RefPart("face", "mesh", mesh=M.polygon_face(outline, th * MM, bbox=(-w / 2 * MM, -h / 2 * MM, w / 2 * MM, h / 2 * MM)),
                   texture=T.to_uint8(img), rgba=[1, 1, 1, 1], specular=0.4, shininess=0.5)
    return [core, face]


def _card_features(ref, p):
    w, h, th, rc = p["width_mm"], p["height_mm"], p["thickness_mm"], p["corner_radius_mm"]
    z = th * MM
    corners = [[-w / 2, h / 2], [w / 2, h / 2], [w / 2, -h / 2], [-w / 2, -h / 2]]   # of the circumscribed rectangle
    return {"rect_corners": to_world(ref, [[a * MM, b * MM, z] for a, b in corners]),
            "edge_midpoints": to_world(ref, [[0, h / 2 * MM, z], [w / 2 * MM, 0, z], [0, -h / 2 * MM, z], [-w / 2 * MM, 0, z]]),
            "corner_radius_m": rc * MM}


# ---- A4
def _a4_parts(ref, p):
    w, h, th = p["width_mm"], p["height_mm"], p["thickness_mm"]
    return [RefPart("sheet", "box", size=[w * MM, h * MM, th * MM], pos=[0, 0, th * MM / 2], rgba=[0.95, 0.95, 0.93, 1],
                    specular=0.05, shininess=0.05)]


def _a4_features(ref, p):
    w, h, th = p["width_mm"], p["height_mm"], p["thickness_mm"]
    z = th * MM
    return {"corners": to_world(ref, [[-w / 2 * MM, h / 2 * MM, z], [w / 2 * MM, h / 2 * MM, z], [w / 2 * MM, -h / 2 * MM, z], [-w / 2 * MM, -h / 2 * MM, z]])}


# ---- cutting mat
_MAT_COLOURS = {"green": ((0.12, 0.38, 0.27), (0.80, 0.88, 0.82)), "grey": ((0.30, 0.31, 0.33), (0.86, 0.87, 0.88)),
                "blue": ((0.12, 0.25, 0.45), (0.82, 0.86, 0.92))}


def _mat_lines(p):
    """Grid line offsets (mm) from the mat centre along x and y (ascending): the grid is centred and stops at the margin;
    bold lines are every `bold_every`-th line counted from the first (-x / -y) line."""
    g = p["grid_mm"]
    out = []
    for half in (p["width_mm"] / 2 - p["margin_mm"], p["height_mm"] / 2 - p["margin_mm"]):
        n = int(np.floor(half / g + 1e-9))
        out.append(np.arange(-n, n + 1) * g)
    return out


def _mat_parts(ref, p):
    w, h, th = p["width_mm"], p["height_mm"], p["thickness_mm"]
    if p["colour"] not in _MAT_COLOURS:
        raise ValueError("cutting_mat colour must be one of %s" % sorted(_MAT_COLOURS))
    base, ink = _MAT_COLOURS[p["colour"]]
    k = float(p["px_per_mm"])
    W, H = int(round(w * k)), int(round(h * k))
    r = _rng(ref.seed, "mat", ref.name)
    img = np.clip(T.flat(H, W, base) * (0.97 + 0.06 * T.value_noise(H, W, 12, r, octaves=2)[..., None]), 0, 1)
    xs, ys = _mat_lines(p)
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    for i, x in enumerate(xs):                      # texel u = (x + w/2) k; texel v = (h/2 - y) k (v = 0 at +y)
        lw = p["bold_line_mm"] if i % p["bold_every"] == 0 else p["line_mm"]          # counted from the -x end, like a real mat
        T.fill_rect(img, (x + w / 2 - lw / 2) * k, (h / 2 - y1) * k, (x + w / 2 + lw / 2) * k, (h / 2 - y0) * k, ink)
    for j, y in enumerate(ys):
        lw = p["bold_line_mm"] if j % p["bold_every"] == 0 else p["line_mm"]          # counted from the -y end
        T.fill_rect(img, (x0 + w / 2) * k, (h / 2 - y - lw / 2) * k, (x1 + w / 2) * k, (h / 2 - y + lw / 2) * k, ink)
    for i, x in enumerate(xs):                      # cm numbers along the +y margin at the bold lines, from the -x end
        if i % p["bold_every"] == 0 and i > 0:
            T.draw_text(img, "%d" % round((x - x0) / 10.0), (x + w / 2) * k, (h / 2 - y1 - 0.75 * p["margin_mm"]) * k, 4.0 * k, ink, align="center")
    for j, y in enumerate(ys):                      # and along the -x margin, from the -y end
        if j % p["bold_every"] == 0:
            T.draw_text(img, "%d" % round((y - y0) / 10.0), (x0 + w / 2 - 0.55 * p["margin_mm"]) * k, (h / 2 - y - 2.0) * k, 4.0 * k, ink, align="center")
    slab = RefPart("slab", "box", size=[w * MM, h * MM, th * MM], pos=[0, 0, th * MM / 2], rgba=list(base) + [1.0], specular=0.05, shininess=0.1)
    face = RefPart("print", "mesh", mesh=M.quad(w * MM, h * MM, z=th * MM + 0.05 * MM), texture=T.to_uint8(img), rgba=[1, 1, 1, 1],
                   specular=0.05, shininess=0.1)
    return [slab, face]


def _mat_features(ref, p):
    xs, ys = _mat_lines(p)
    z = (p["thickness_mm"] + 0.05) * MM
    inter = [[x * MM, y * MM, z] for y in ys[::-1] for x in xs]          # rows from +y, then along +x
    w, h = p["width_mm"], p["height_mm"]
    return {"grid_intersections": to_world(ref, inter), "grid_shape": [len(ys), len(xs)],
            "grid_x_mm": xs.tolist(), "grid_y_mm": ys.tolist(),
            "mat_corners": to_world(ref, [[-w / 2 * MM, h / 2 * MM, z], [w / 2 * MM, h / 2 * MM, z], [w / 2 * MM, -h / 2 * MM, z], [-w / 2 * MM, -h / 2 * MM, z]])}


_PARTS = {"ruler": _ruler_parts, "checkerboard": _checker_parts, "can": _can_parts, "card": _card_parts, "a4": _a4_parts,
          "cutting_mat": _mat_parts}
_FEATURES = {"ruler": _ruler_features, "checkerboard": _checker_features, "can": _can_features, "card": _card_features, "a4": _a4_features,
             "cutting_mat": _mat_features}
KINDS = tuple(sorted(_PARTS))


def parts(ref: ScaleRef) -> List[RefPart]:
    return _PARTS[ref.kind](ref, _params(ref))


def features(ref: ScaleRef) -> Dict[str, Any]:
    return _FEATURES[ref.kind](ref, _params(ref))


def nominal(ref: ScaleRef) -> Dict[str, Any]:
    """The PUBLIC facts of a reference: what a human holding it knows (never its pose)."""
    p = _params(ref)
    if ref.kind == "ruler":
        return {"name": ref.name, "kind": "steel ruler", "scale_length_mm": p["length_mm"],
                "graduation": "1 mm ticks along one long edge; longer ticks every 5 mm and every 10 mm; numbered every 10 mm "
                              "(0 .. %d = centimetres)" % int(p["length_mm"] // 10),
                "overall_mm": [p["length_mm"] + 2 * p["end_mm"], p["width_mm"], p["thickness_mm"]]}
    if ref.kind == "checkerboard":
        bx, by = _board_dims(p)
        return {"name": ref.name, "kind": "checkerboard calibration target printed on a flat rigid board",
                "squares": [p["cols"], p["rows"]], "square_mm": p["square_mm"], "inner_corners": [p["cols"] - 1, p["rows"] - 1],
                "board_mm": [bx, by, p["thickness_mm"]], "white_margin_mm": p["margin_mm"]}
    if ref.kind == "can":
        return {"name": ref.name, "kind": "beverage can (standard 355 ml size)", "diameter_mm": p["diameter_mm"], "height_mm": p["height_mm"]}
    if ref.kind == "card":
        return {"name": ref.name, "kind": "ID-1 plastic card (bank-card size, ISO/IEC 7810)",
                "size_mm": [p["width_mm"], p["height_mm"]], "thickness_mm": p["thickness_mm"], "corner_radius_mm": p["corner_radius_mm"]}
    if ref.kind == "a4":
        return {"name": ref.name, "kind": "A4 sheet of paper (ISO 216)", "size_mm": [p["width_mm"], p["height_mm"]]}
    if ref.kind == "cutting_mat":
        return {"name": ref.name, "kind": "%s self-healing cutting mat with a printed grid" % p["colour"], "size_mm": [p["width_mm"], p["height_mm"]],
                "thickness_mm": p["thickness_mm"], "grid_mm": p["grid_mm"],
                "bold_every": p["bold_every"], "grid": "lines every %g mm; counting from one corner, every %d lines one bold line; bold lines numbered in cm along two margins"
                        % (p["grid_mm"], p["bold_every"])}
    raise ValueError(ref.kind)


# ----------------------------------------------------------------------------- constructors
def _ref(kind, name, pos, yaw_deg, quat, seed, collide, **params):
    q = list(quat) if quat is not None else yaw_quat(yaw_deg)
    r = ScaleRef(kind=kind, name=name, pos=[float(v) for v in pos], quat=[float(v) for v in q],
                 params={k: v for k, v in params.items() if v is not None}, collide=collide, seed=seed)
    _params(r)
    return r


def ruler(name="ruler", pos=(0, 0, 0), yaw_deg=0.0, quat=None, seed=0, collide=False, **params) -> ScaleRef:
    return _ref("ruler", name, pos, yaw_deg, quat, seed, collide, **params)


def checkerboard(name="checkerboard", pos=(0, 0, 0), yaw_deg=0.0, quat=None, seed=0, collide=False, **params) -> ScaleRef:
    return _ref("checkerboard", name, pos, yaw_deg, quat, seed, collide, **params)


def drink_can(name="can", pos=(0, 0, 0), yaw_deg=0.0, quat=None, seed=0, collide=False, **params) -> ScaleRef:
    return _ref("can", name, pos, yaw_deg, quat, seed, collide, **params)


def id_card(name="card", pos=(0, 0, 0), yaw_deg=0.0, quat=None, seed=0, collide=False, **params) -> ScaleRef:
    return _ref("card", name, pos, yaw_deg, quat, seed, collide, **params)


def a4_sheet(name="a4", pos=(0, 0, 0), yaw_deg=0.0, quat=None, seed=0, collide=False, **params) -> ScaleRef:
    return _ref("a4", name, pos, yaw_deg, quat, seed, collide, **params)


def cutting_mat(name="cutting_mat", pos=(0, 0, 0), yaw_deg=0.0, quat=None, seed=0, collide=False, size_mm=None, **params) -> ScaleRef:
    """A3-class self-healing mat (default 450 x 300 x 2 mm, green, 10 mm grid, every 5th line bold, cm numbers)."""
    if size_mm is not None:
        params["width_mm"], params["height_mm"] = float(size_mm[0]), float(size_mm[1])
    return _ref("cutting_mat", name, pos, yaw_deg, quat, seed, collide, **params)


# ----------------------------------------------------------------------------- MuJoCo builder
def add_to_spec(spec, ref: ScaleRef) -> Dict[str, Any]:
    """Add `ref` to a mujoco.MjSpec as a static body `ref.name` (visual geoms in group 2, no contacts unless
    ref.collide, which adds a collision primitive in group 3).  Returns the created element names."""
    import mujoco
    pre = ref.name + "__"
    body = spec.worldbody.add_body(name=ref.name, pos=list(ref.pos), quat=list(ref.quat))
    out = {"body": ref.name, "geoms": [], "materials": [], "textures": [], "meshes": []}
    for part in parts(ref):
        mat_name = pre + part.name
        mat = spec.add_material(name=mat_name, rgba=list(part.rgba), specular=part.specular, shininess=part.shininess,
                                reflectance=part.reflectance)
        out["materials"].append(mat_name)
        if part.texture is not None:
            h, w = part.texture.shape[:2]
            tex = spec.add_texture(name=pre + part.name + "_tex", type=mujoco.mjtTexture.mjTEXTURE_2D, width=w, height=h, nchannel=3)
            tex.data = np.ascontiguousarray(part.texture, np.uint8).tobytes()
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            out["textures"].append(tex.name)
        gname = pre + part.name
        kw = dict(name=gname, material=mat_name, contype=0, conaffinity=0, group=2, pos=list(part.pos))
        if part.kind == "mesh":
            m = part.mesh
            mesh = spec.add_mesh(name=pre + part.name + "_mesh", uservert=m.vertices.reshape(-1).tolist(),
                                 userface=m.faces.reshape(-1).tolist())
            if m.uv is not None:
                mesh.usertexcoord = m.uv.reshape(-1).tolist()
                mesh.userfacetexcoord = m.faces.reshape(-1).tolist()
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            out["meshes"].append(mesh.name)
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name, **kw)
        elif part.kind == "box":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2 for s in part.size], **kw)
        elif part.kind == "cylinder":
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[part.size[0], part.size[1] / 2, 0], **kw)
        else:
            raise ValueError(part.kind)
        out["geoms"].append(gname)
    if ref.collide:
        lo, hi = _local_aabb(ref)
        c = (lo + hi) / 2
        body.add_geom(name=pre + "collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=((hi - lo) / 2).tolist(), pos=c.tolist(),
                      group=3, contype=1, conaffinity=1, rgba=[0, 0, 0, 0])
        out["geoms"].append(pre + "collision")
    return out


def _local_aabb(ref):
    pts = []
    for part in parts(ref):
        if part.kind == "mesh":
            pts.append(part.mesh.vertices + np.asarray(part.pos))
        elif part.kind == "box":
            s = np.asarray(part.size) / 2
            pts.append(np.asarray(part.pos) + np.array([-s, s]))
        else:
            r, h = part.size
            pts.append(np.asarray(part.pos) + np.array([[-r, -r, -h / 2], [r, r, h / 2]]))
    P = np.vstack(pts)
    return P.min(axis=0), P.max(axis=0)
