"""The crushed look of a damaged parcel: an overlay on the parcel's own base colour map, made when the cell counts the
parcel as damaged (never at a shift's start).

    new_rgb = crush_rgb(rgb, seed, kind)        # uint8 H x W x C (C >= 3), the same shape back; deterministic

What it draws, over the whole map (the waybill's sheet included): the board or film darkened to ``TONE`` of its colour
and greyed a little, then dark irregular creases: ``N_CREASES`` polylines of 2-5 segments each across the map, a few
pixels wide (for a 512 px map), at ``CREASE_DARK`` of the darkened colour, each with a pale fold line beside it
(cartons), or the same creases a little softer and more of them with pale scuffs between them (soft parcels). No text,
no symbol, nothing on the geometry or the depth.

The measure the documents give (``measure``): on a parcel's faces in a camera image, the mean grey level drops and the
share of edge pixels (Canny 50 / 150 on the grey image) rises; measured on 20 parcels in the filament backend: the grey
level to 0.45-0.78 of the intact one, the edge share 1.5-24 times (cameras.md / measured.json "damage_look").
"""

from __future__ import annotations

import numpy as np

__all__ = ["crush_rgb", "measure", "TONE", "CREASE_DARK", "N_CREASES"]

TONE = 0.5
GREY = 0.18
CREASE_DARK = 0.30
N_CREASES = (70, 100)


def crush_rgb(img, seed, kind="rigid"):
    import cv2
    img = np.asarray(img, np.uint8)
    h, w = img.shape[:2]
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, (int(seed) >> 32) & 0xFFFFFFFF, 0xC5A5])
    rgb = img[..., :3].astype(np.float32)
    lum = rgb.mean(axis=2, keepdims=True)
    base = (rgb * (1.0 - GREY) + lum * GREY) * TONE
    dark = np.zeros((h, w), np.uint8)
    pale = np.zeros((h, w), np.uint8)
    soft = str(kind) in ("soft", "mailer")
    n = int(rng.integers(N_CREASES[0], N_CREASES[1] + 1)) + (25 if soft else 0)
    s = max(1.0, w / 512.0)
    for _ in range(n):
        x, y = float(rng.uniform(0, w)), float(rng.uniform(0, h))
        ang = float(rng.uniform(0, np.pi))
        seg = float(rng.uniform(0.04, 0.14)) * max(w, h)
        pts = [(x, y)]
        for _k in range(int(rng.integers(2, 6))):
            ang += float(rng.normal(0.0, 0.6))
            x += seg * np.cos(ang)
            y += seg * np.sin(ang)
            pts.append((x, y))
        p = np.round(np.array(pts)).astype(np.int32).reshape(-1, 1, 2)
        th = int(round(float(rng.uniform(2.0, 4.5)) * s))
        cv2.polylines(dark, [p], False, 255, thickness=max(1, th), lineType=cv2.LINE_8)
        off = np.round(np.array([np.sin(ang), -np.cos(ang)]) * (th + 1)).astype(np.int32)
        cv2.polylines(pale, [p + off.reshape(1, 1, 2)], False, 255, thickness=max(1, th // 2), lineType=cv2.LINE_8)
    if soft:                                       # scuffs between the creases: short pale strokes
        for _ in range(int(rng.integers(80, 120))):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            a = float(rng.uniform(0, np.pi))
            L = int(rng.integers(4, 14) * s)
            cv2.line(pale, (x, y), (int(x + L * np.cos(a)), int(y + L * np.sin(a))), 255, max(1, int(s)),
                     cv2.LINE_8)
    d = (dark > 0)[..., None]
    pl = ((pale > 0) & ~(dark > 0))[..., None]
    out = np.where(d, base * (CREASE_DARK if not soft else CREASE_DARK + 0.08), base)
    out = np.where(pl, np.minimum(255.0, base * (1.45 if not soft else 1.35)), out)
    res = img.copy()
    res[..., :3] = np.clip(np.rint(out), 0, 255).astype(np.uint8)
    return res


def measure(before, after, mask=None):
    """(mean grey ratio after / before, edge share after / before) of two images of the same parcel (uint8 H x W x 3),
    over ``mask`` (bool H x W; None: the whole image); an edge share below 1 % counts as 1 %."""
    import cv2
    out = []
    for im in (before, after):
        g = cv2.cvtColor(np.asarray(im, np.uint8), cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(g, 50, 150) > 0
        m = np.ones(g.shape, bool) if mask is None else mask
        out.append((float(g[m].mean()), float(e[m].mean())))
    (g0, e0), (g1, e1) = out
    return g1 / max(g0, 1e-9), e1 / max(e0, 0.01)          # an edge share below 1 % counts as 1 %
