"""Texture images of the skid (cell internals; the agent's catalog is drawn from them): tag plates, dial faces, instrument scales and panel labels.

Every image is drawn with Pillow (its built-in scalable font, `ImageFont.load_default(size)`) into an RGB uint8 array
and handed to MuJoCo as in-memory texture data (no files). A texture on a box geom covers each face of the box; every
textured plate of the skid faces -y (towards the robot), and the image's left edge lies at -x, its top edge at +z.

Dial convention (every dial gauge of the skid): the scale sweeps 270 degrees clockwise, from the low end at the 7:30
position (225 degrees counter-clockwise from 3 o'clock) to the high end at the 4:30 position (-45 degrees); see
`dial_angle`. The same function places the needle, so the drawn scale and the needle agree exactly.

Runtime dependencies: numpy + Pillow only.
"""

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["font", "tag_plate", "label_plate", "dial_face", "dial_angle", "scale_strip", "GAUGE_KINDS",
           "DIAL_SWEEP_DEG", "DIAL_START_DEG", "nice_ticks"]

DIAL_START_DEG = 225.0       # the low end of every dial, counter-clockwise from 3 o'clock
DIAL_SWEEP_DEG = 270.0       # swept clockwise

# The gauge types the skid uses (the variant decides which one a gauge position gets). `lo`/`hi` in `unit`;
# `major`/`minor`: tick spacing; `scale`: value of one `unit` in the process quantity's base unit
# (pressures: bar; flows: m3/h; levels: %).
GAUGE_KINDS = {
    "p_0_6_bar": {"lo": 0.0, "hi": 6.0, "unit": "bar", "major": 1.0, "minor": 0.2, "scale": 1.0, "quantity": "pressure"},
    "p_0_10_bar": {"lo": 0.0, "hi": 10.0, "unit": "bar", "major": 2.0, "minor": 0.5, "scale": 1.0, "quantity": "pressure"},
    "p_0_100_psi": {"lo": 0.0, "hi": 100.0, "unit": "psi", "major": 20.0, "minor": 5.0, "scale": 1.0 / 14.5037738, "quantity": "pressure"},
    "p_0_600_kpa": {"lo": 0.0, "hi": 600.0, "unit": "kPa", "major": 100.0, "minor": 20.0, "scale": 0.01, "quantity": "pressure"},
    "c_m1_3_bar": {"lo": -1.0, "hi": 3.0, "unit": "bar", "major": 1.0, "minor": 0.2, "scale": 1.0, "quantity": "pressure"},
    "c_m100_300_kpa": {"lo": -100.0, "hi": 300.0, "unit": "kPa", "major": 100.0, "minor": 20.0, "scale": 0.01, "quantity": "pressure"},
    "d_0_1_bar": {"lo": 0.0, "hi": 1.0, "unit": "bar", "major": 0.2, "minor": 0.05, "scale": 1.0, "quantity": "differential pressure"},
    "d_0_100_kpa": {"lo": 0.0, "hi": 100.0, "unit": "kPa", "major": 20.0, "minor": 5.0, "scale": 0.01, "quantity": "differential pressure"},
    "f_05_5_m3h": {"lo": 0.5, "hi": 5.0, "unit": "m3/h", "major": 0.5, "minor": 0.1, "scale": 1.0, "quantity": "flow"},
    "f_10_80_lpm": {"lo": 10.0, "hi": 80.0, "unit": "L/min", "major": 10.0, "minor": 2.0, "scale": 0.06, "quantity": "flow"},
    "l_0_100_pct": {"lo": 0.0, "hi": 100.0, "unit": "%", "major": 10.0, "minor": 2.0, "scale": 1.0, "quantity": "level"},
}

_FONTS = {}


def font(size):
    """Pillow's built-in scalable font at `size` px (cached)."""
    size = int(size)
    f = _FONTS.get(size)
    if f is None:
        f = ImageFont.load_default(size=size)
        _FONTS[size] = f
    return f


def _text_size(draw, text, fnt):
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=fnt)
    return x1 - x0, y1 - y0, x0, y0


def _centered_text(draw, cx, cy, text, fnt, fill):
    w, h, ox, oy = _text_size(draw, text, fnt)
    draw.text((cx - w / 2 - ox, cy - h / 2 - oy), text, font=fnt, fill=fill)


def _fit_font(draw, text, max_w, max_h, start):
    size = int(start)
    while size > 8:
        w, h, _, _ = _text_size(draw, text, font(size))
        if w <= max_w and h <= max_h:
            break
        size -= 2
    return font(size)


def tag_plate(text, w_px=320, h_px=128, bg=(236, 236, 228), fg=(18, 18, 18), border=(30, 30, 30)):
    """A stainless-style tag plate: the tag text centred, a thin border (RGB uint8 array h_px x w_px)."""
    img = Image.new("RGB", (w_px, h_px), bg)
    d = ImageDraw.Draw(img)
    bw = max(2, h_px // 32)
    d.rectangle([bw, bw, w_px - 1 - bw, h_px - 1 - bw], outline=border, width=bw)
    fnt = _fit_font(d, text, w_px * 0.88, h_px * 0.66, h_px * 0.62)
    _centered_text(d, w_px / 2, h_px / 2, text, fnt, fg)
    return np.asarray(img, dtype=np.uint8)


def label_plate(lines, w_px=256, h_px=96, bg=(245, 245, 240), fg=(15, 15, 15)):
    """A panel label: one or more lines of text centred on a plain plate."""
    if isinstance(lines, str):
        lines = [lines]
    img = Image.new("RGB", (w_px, h_px), bg)
    d = ImageDraw.Draw(img)
    n = len(lines)
    for i, t in enumerate(lines):
        fnt = _fit_font(d, t, w_px * 0.9, h_px / n * 0.78, h_px / n * 0.72)
        _centered_text(d, w_px / 2, h_px * (i + 0.5) / n, t, fnt, fg)
    return np.asarray(img, dtype=np.uint8)


def dial_angle(value, lo, hi):
    """Needle / tick angle (radians, counter-clockwise from 3 o'clock as seen from the front) of `value` on a dial from
    `lo` to `hi`; values outside the range stop at the pegs 4 degrees beyond the ends."""
    u = (float(value) - lo) / (hi - lo)
    u = min(1.0 + 4.0 / DIAL_SWEEP_DEG, max(-4.0 / DIAL_SWEEP_DEG, u))
    return math.radians(DIAL_START_DEG - DIAL_SWEEP_DEG * u)


def nice_ticks(lo, hi, step):
    n = int(round((hi - lo) / step))
    return [lo + i * step for i in range(n + 1)]


def _fmt(v, step):
    if abs(step - round(step)) < 1e-9 and abs(v - round(v)) < 1e-9:
        return "%d" % int(round(v))
    dec = 1 if step >= 0.1 - 1e-9 else 2
    return ("%." + str(dec) + "f") % v


def dial_face(kind, title="", px=512, case_rgb=(52, 54, 58)):
    """A round dial (white face, black scale) on a square of the case colour (RGB uint8 px x px). `kind` is a key of
    GAUGE_KINDS. The unit is printed under the centre, `title` (e.g. the measured quantity) above it."""
    k = GAUGE_KINDS[kind]
    img = Image.new("RGB", (px, px), case_rgb)
    d = ImageDraw.Draw(img)
    c = px / 2.0
    r = px * 0.47
    d.ellipse([c - r, c - r, c + r, c + r], fill=(250, 250, 246), outline=(20, 20, 20), width=max(2, px // 128))
    lo, hi = k["lo"], k["hi"]
    r_out = r * 0.92
    for v in nice_ticks(lo, hi, k["minor"]):
        a = dial_angle(v, lo, hi)
        major = abs((v - lo) / k["major"] - round((v - lo) / k["major"])) < 1e-6
        r_in = r_out - (r * 0.13 if major else r * 0.065)
        wdt = max(2, px // (90 if major else 200))
        d.line([(c + r_in * math.cos(a), c - r_in * math.sin(a)), (c + r_out * math.cos(a), c - r_out * math.sin(a))],
               fill=(10, 10, 10), width=wdt)
    fnt = font(int(px * 0.075))
    r_lab = r * 0.62
    for v in nice_ticks(lo, hi, k["major"]):
        a = dial_angle(v, lo, hi)
        _centered_text(d, c + r_lab * math.cos(a), c - r_lab * math.sin(a), _fmt(v, k["major"]), fnt, (10, 10, 10))
    _centered_text(d, c, c + r * 0.42, k["unit"], font(int(px * 0.085)), (10, 10, 10))
    if title:
        _centered_text(d, c, c - r * 0.33, title, font(int(px * 0.05)), (60, 60, 60))
    # the centre boss (the needle's hub is drawn by the geometry; this is its shadow ring)
    rb = r * 0.05
    d.ellipse([c - rb, c - rb, c + rb, c + rb], fill=(90, 90, 90))
    return np.asarray(img, dtype=np.uint8)


def scale_strip(kind, w_px=128, h_px=640, bg=(248, 248, 244), fg=(12, 12, 12), margin_frac=0.06, title=""):
    """A vertical scale (low at the bottom, high at the top) for a rotameter or a level gauge: ticks along the left
    edge, numbers to their right, the unit at the top. `margin_frac` of the height is left blank at both ends; the
    scale's low and high ends lie exactly at those margins (see `scale_span`)."""
    k = GAUGE_KINDS[kind]
    img = Image.new("RGB", (w_px, h_px), bg)
    d = ImageDraw.Draw(img)
    lo, hi = k["lo"], k["hi"]
    top_y = h_px * (margin_frac + 0.06)          # leave room for the unit at the very top
    bot_y = h_px * (1.0 - margin_frac)

    def ypix(v):
        return bot_y - (v - lo) / (hi - lo) * (bot_y - top_y)

    for v in nice_ticks(lo, hi, k["minor"]):
        major = abs((v - lo) / k["major"] - round((v - lo) / k["major"])) < 1e-6
        x1 = w_px * (0.42 if major else 0.25)
        d.line([(0, ypix(v)), (x1, ypix(v))], fill=fg, width=max(2, h_px // (200 if major else 400)))
    fnt = font(int(min(w_px * 0.26, h_px * 0.04)))
    for v in nice_ticks(lo, hi, k["major"]):
        w, h, ox, oy = _text_size(d, _fmt(v, k["major"]), fnt)
        d.text((w_px * 0.47 - ox, ypix(v) - h / 2 - oy), _fmt(v, k["major"]), font=fnt, fill=fg)
    _centered_text(d, w_px / 2, h_px * 0.04, k["unit"] if not title else title, font(int(min(w_px * 0.22, h_px * 0.035))), fg)
    return np.asarray(img, dtype=np.uint8)


def scale_span(margin_frac=0.06):
    """(low, high) fractions of a scale strip's height, measured from the BOTTOM, at which the scale's ends lie."""
    return margin_frac, 1.0 - (margin_frac + 0.06)
