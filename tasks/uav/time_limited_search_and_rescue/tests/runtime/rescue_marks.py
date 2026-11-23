"""Paints of the wayfinding parts: the model's signal colours as paint (tone), and a door apron's picture (worn paint
over the floor's concrete). One module for every renderer of the scene: the cameras' renderer draws with it, and the
scene that people's pictures are rendered from is built with it. Table lookups and exact arithmetic only (no pow),
so that a picture is the same on every host."""
import hashlib

import numpy as np


def tone(rgb, sat=0.78, val=0.86):
    """a signal colour a little greyer and darker: paint, not a display colour (sRGB values in and out)"""
    l_ = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    return tuple(min(1.0, max(0.0, val * (l_ + sat * (c - l_)))) for c in rgb[:3])


TONES = {"metal": (0.78, 0.86), "wall": (0.78, 0.86), "floor": (0.75, 0.70), "mat": (0.80, 0.80), "apron": (0.75, 0.72)}


def hex_of(rgb):
    return "%02x%02x%02x" % tuple(int(255 * min(1.0, max(0.0, float(x))) + 0.5) for x in rgb[:3])


def _tables():
    v = np.arange(256, dtype=np.float64) / 255.0
    fwd = np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)        # computed once per process from 256 values
    fwd = np.round(fwd * 65535.0).astype(np.int64)
    lin = np.arange(65536, dtype=np.float64) / 65535.0
    back = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    return fwd, np.clip(np.round(back * 255.0), 0, 255).astype(np.uint8)


def apron_picture(toned_rgb, seed, base, n=256):
    """(n, n, 3) uint8: the paint in the toned colour over the worn concrete `base` (an sRGB picture, uint8), rubbed
    through in blotches and along the edges. The pattern follows from the site's seed and the colour."""
    from PIL import Image
    fwd, back = _tables()
    key = "%d_%s" % (int(seed), hex_of(toned_rgb))
    rg = np.random.default_rng(int(hashlib.md5(key.encode()).hexdigest()[:8], 16))
    b8 = np.asarray(Image.fromarray(np.ascontiguousarray(base)).convert("RGB").resize((n, n), Image.BILINEAR))
    lin = fwd[b8]                                                           # 0..65535
    lum = lin.sum(-1, keepdims=True) // 3
    lum_mean = int(lum.sum() // lum.size)
    noise = np.zeros((n, n), np.int64)
    for oct_, amp in ((4, 50), (9, 30), (23, 20), (61, 12)):
        g = (rg.integers(0, 256, (oct_ + 1, oct_ + 1))).astype(np.uint8)
        noise += amp * np.asarray(Image.fromarray(g).resize((n, n), Image.BICUBIC)).astype(np.int64)
    lo, hi = int(noise.min()), int(noise.max())
    noise = (noise - lo) * 1000 // max(hi - lo, 1)                          # 0..1000
    ii = np.arange(n, dtype=np.int64)
    ex = np.minimum(ii, n - 1 - ii) * 1000 // (n - 1)
    edge = np.minimum(ex[None, :], ex[:, None])                             # 0..500
    keep = np.clip((noise - 160) * 9, 0, 1000) * np.clip(edge * 45 - (1000 - noise) * 16 // 10, 0, 1000) // 1000
    keep = (860 * keep + 140 * np.clip((noise - 500) * 4, 0, 1000)) // 1000  # 0..1000: the share of paint
    t = np.array([int(fwd[int(255 * c + 0.5)]) for c in toned_rgb[:3]], np.int64)
    paint = t[None, None, :] * (800 + 400 * lum // max(lum_mean, 1)) // 1000
    floor = lin * np.array([47, 46, 44], np.int64)[None, None, :] // 100    # the concrete, darker than the swept floor
    img = (paint * keep[..., None] + floor * (1000 - keep[..., None])) // 1000
    return back[np.clip(img, 0, 65535)]
