"""Procedural textures (numpy uint8 H x W x 3), deterministic in their seed; no image asset of any third party.

    make(TextureSpec, base_dir=None)             -> array   (kinds: flat checker speckle gradient wood brushed grid paper file)
    value_noise(h, w, cells, rng, octaves)       -> float array in [0, 1]
    fill_rect(img, x0, y0, x1, y1, rgb)          anti-aliased (exact area coverage) axis-aligned rectangle, texel units
    draw_text(img, text, x, y, height, rgb, anchor)   5 x 7 bitmap glyphs (digits, a few letters) drawn with fill_rect

Texel convention (shared with meshes.py): texel (row r, col c) covers [c, c + 1] x [r, r + 1]; a feature at texel
x maps to u = x / W; row 0 is v = 0 (meshes.py puts v = 0 at the top edge of a textured face).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .spec import TextureSpec, rng as _rng


def _rgb(c):
    c = np.asarray(c, float)
    return c / 255.0 if c.max() > 1.0 else c


def to_uint8(img_f):
    return np.clip(np.rint(np.asarray(img_f) * 255.0), 0, 255).astype(np.uint8)


def value_noise(h, w, cells, rng, octaves=3, persistence=0.5):
    """Smooth value noise: bilinear upsampling of random grids (cells x cells at the coarsest octave)."""
    out = np.zeros((h, w))
    amp, total = 1.0, 0.0
    for o in range(octaves):
        n = max(2, int(cells * (2 ** o)))
        g = rng.random((n + 1, n + 1))
        ys = np.linspace(0, n, h, endpoint=False)
        xs = np.linspace(0, n, w, endpoint=False)
        y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
        fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
        fy, fx = fy * fy * (3 - 2 * fy), fx * fx * (3 - 2 * fx)
        a = g[y0][:, x0]
        b = g[y0][:, x0 + 1]
        c = g[y0 + 1][:, x0]
        d = g[y0 + 1][:, x0 + 1]
        out += amp * ((a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy)
        total += amp
        amp *= persistence
    return out / total


def fill_rect(img, x0, y0, x1, y1, rgb, alpha=1.0):
    """Blend `rgb` over img (float H x W x 3 in 0..1) with the exact area coverage of [x0, x1] x [y0, y1] (texels)."""
    H, W = img.shape[:2]
    x0, x1 = max(0.0, min(x0, x1)), min(float(W), max(x0, x1))
    y0, y1 = max(0.0, min(y0, y1)), min(float(H), max(y0, y1))
    if x1 <= x0 or y1 <= y0:
        return img
    c0, c1 = int(np.floor(x0)), int(np.ceil(x1))
    r0, r1 = int(np.floor(y0)), int(np.ceil(y1))
    cols = np.arange(c0, c1)
    rows = np.arange(r0, r1)
    cx = np.clip(np.minimum(cols + 1, x1) - np.maximum(cols, x0), 0, 1)
    cy = np.clip(np.minimum(rows + 1, y1) - np.maximum(rows, y0), 0, 1)
    a = (cy[:, None] * cx[None, :] * alpha)[..., None]
    img[r0:r1, c0:c1] = img[r0:r1, c0:c1] * (1 - a) + a * _rgb(rgb)[None, None, :]
    return img


# 5 x 7 glyphs, rows top to bottom, '#' = ink.  Drawn for this module (public-domain style pixel font).
_GLYPHS = {
    "0": [".###.", "#...#", "#..##", "#.#.#", "##..#", "#...#", ".###."],
    "1": ["..#..", ".##..", "..#..", "..#..", "..#..", "..#..", ".###."],
    "2": [".###.", "#...#", "....#", "...#.", "..#..", ".#...", "#####"],
    "3": ["####.", "....#", "....#", ".###.", "....#", "....#", "####."],
    "4": ["...#.", "..##.", ".#.#.", "#..#.", "#####", "...#.", "...#."],
    "5": ["#####", "#....", "####.", "....#", "....#", "#...#", ".###."],
    "6": ["..##.", ".#...", "#....", "####.", "#...#", "#...#", ".###."],
    "7": ["#####", "....#", "...#.", "..#..", ".#...", ".#...", ".#..."],
    "8": [".###.", "#...#", "#...#", ".###.", "#...#", "#...#", ".###."],
    "9": [".###.", "#...#", "#...#", ".####", "....#", "...#.", ".##.."],
    "m": [".....", ".....", "##.#.", "#.#.#", "#.#.#", "#.#.#", "#.#.#"],
    "c": [".....", ".....", ".###.", "#....", "#....", "#....", ".###."],
    "A": [".###.", "#...#", "#...#", "#####", "#...#", "#...#", "#...#"],
    " ": [".....", ".....", ".....", ".....", ".....", ".....", "....."],
}


def text_width(text, height):
    s = height / 7.0
    return len(text) * 6 * s - s


def draw_text(img, text, x, y, height, rgb, anchor="left"):
    """Bitmap text with its top-left at (x, y) texels (anchor 'center' centres it on x)."""
    s = height / 7.0
    if anchor == "center":
        x = x - text_width(text, height) / 2.0
    for k, ch in enumerate(text):
        g = _GLYPHS[ch]
        gx = x + k * 6 * s
        for r, row in enumerate(g):
            for c, px in enumerate(row):
                if px == "#":
                    fill_rect(img, gx + c * s, y + r * s, gx + (c + 1) * s, y + (r + 1) * s, rgb)
    return img


# ----------------------------------------------------------------------------- kinds
def flat(h, w, rgb1, **_):
    return np.ones((h, w, 3)) * _rgb(rgb1)[None, None, :]


def checker(h, w, rgb1, rgb2, squares=8, **_):
    yy, xx = np.mgrid[0:h, 0:w]
    m = ((xx * squares // w + yy * squares // h) % 2).astype(float)[..., None]
    return _rgb(rgb1) * (1 - m) + _rgb(rgb2) * m


def speckle(h, w, rgb1, rgb2, rng, density=0.08, cells=6, contrast=0.12, **_):
    base = value_noise(h, w, cells, rng, octaves=4)
    img = _rgb(rgb1)[None, None, :] * (1 - contrast + 2 * contrast * base[..., None])
    spots = rng.random((h, w)) < density
    img[spots] = img[spots] * 0.4 + _rgb(rgb2) * 0.6
    return img


def gradient(h, w, rgb1, rgb2, **_):
    t = np.linspace(0, 1, h)[:, None, None]
    return _rgb(rgb1) * (1 - t) + _rgb(rgb2) * t * np.ones((1, w, 1))


def wood(h, w, rgb1, rgb2, rng, rings=9.0, warp=0.35, **_):
    """Grain running along x (texture u)."""
    n = value_noise(h, w, 4, rng, octaves=4)
    yy = np.linspace(0, 1, h)[:, None]
    phase = rings * (yy + warp * n) * 2 * np.pi
    t = 0.5 + 0.5 * np.sin(phase)
    t = t ** 2.2
    fine = value_noise(h, w, 40, rng, octaves=1)
    t = np.clip(t * 0.85 + 0.15 * fine, 0, 1)[..., None]
    return _rgb(rgb1) * (1 - t) + _rgb(rgb2) * t


def brushed(h, w, rgb1, rng, strength=0.06, **_):
    """Brushed metal: streaks along x."""
    streak = rng.normal(0, 1, (h, 1)) * np.ones((1, w))
    k = max(3, w // 64)
    kern = np.ones(k) / k
    streak = np.apply_along_axis(lambda r: np.convolve(r, kern, mode="same"), 1, streak + 0.3 * rng.normal(0, 1, (h, w)))
    streak = (streak - streak.mean()) / (streak.std() + 1e-9)
    return np.clip(_rgb(rgb1)[None, None, :] * (1 + strength * streak[..., None]), 0, 1)


def grid(h, w, rgb1, rgb2, rng, tiles_x=8, tiles_y=8, grout=0.06, jitter=0.04, **_):
    """Tiles of colour rgb1 (per-tile jitter) separated by grout rgb2."""
    img = flat(h, w, rgb2)
    tw, th = w / float(tiles_x), h / float(tiles_y)
    for j in range(tiles_y):
        for i in range(tiles_x):
            c = np.clip(_rgb(rgb1) * (1 + jitter * rng.normal()), 0, 1)
            g = grout * min(tw, th) / 2.0
            fill_rect(img, i * tw + g, j * th + g, (i + 1) * tw - g, (j + 1) * th - g, c)
    return img


def paper(h, w, rgb1, rng, **_):
    n = value_noise(h, w, 30, rng, octaves=2)
    return np.clip(_rgb(rgb1)[None, None, :] * (0.985 + 0.03 * n[..., None]), 0, 1)


def make(tex: TextureSpec, base_dir: Optional[str] = None) -> np.ndarray:
    """The uint8 H x W x 3 array of a TextureSpec."""
    h, w = int(tex.height), int(tex.width)
    r = _rng(tex.seed, "texture", tex.name, tex.kind)
    kw = dict(tex.params)
    if tex.kind == "file":
        from PIL import Image
        p = tex.file if os.path.isabs(tex.file) else os.path.join(base_dir or os.getcwd(), tex.file)
        with Image.open(p) as im:
            return np.asarray(im.convert("RGB"), np.uint8).copy()
    fns = {"flat": flat, "checker": checker, "speckle": speckle, "gradient": gradient, "wood": wood,
           "brushed": brushed, "grid": grid, "paper": paper}
    if tex.kind not in fns:
        raise ValueError("unknown texture kind %r (have %s)" % (tex.kind, sorted(fns) + ["file"]))
    img = fns[tex.kind](h, w, rgb1=tex.rgb1, rgb2=tex.rgb2, rng=r, **kw)
    return to_uint8(img)
