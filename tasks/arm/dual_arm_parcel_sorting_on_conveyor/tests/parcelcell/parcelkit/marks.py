"""Other marks on a parcel: printed handling pictograms, a board-maker's stamp, small stickers, a mailer's resin mark.

Every mark is drawn as an image in its own millimetre frame (row 0 = the mark's top edge) and returned with its size.
None of them carries the waybill's print: no barcode, no 2-D code, no routing code, no print bands; the plain paper
sticker has a waybill-like size and a paper's tone but no print at all (see ``describe``).
"""
from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")


def _font(kind, px):
    name = {"bold": "RobotoCondensed-Bold.ttf", "medium": "RobotoCondensed-Medium.ttf",
            "regular": "RobotoCondensed-Regular.ttf"}[kind]
    return ImageFont.truetype(os.path.join(_FONT_DIR, name), max(1, int(round(px))))


def _canvas(w_mm, h_mm, ppm, ss=2):
    W, H = max(2, int(round(w_mm * ppm * ss))), max(2, int(round(h_mm * ppm * ss)))
    im = Image.new("L", (W, H), 0)
    return im, ImageDraw.Draw(im), ppm * ss


def _finish(im, ss=2):
    a = np.asarray(im, np.float32) / 255.0
    h, w = a.shape
    h2, w2 = h // ss, w // ss
    return a[:h2 * ss, :w2 * ss].reshape(h2, ss, w2, ss).mean(axis=(1, 3))


def _arrow_up(d, cx, top, bot, half_w, k, stroke):
    head = (bot - top) * 0.42
    d.polygon([(cx, top), (cx + half_w, top + head), (cx - half_w, top + head)], fill=255)
    d.rectangle([cx - stroke / 2, top + head - 1, cx + stroke / 2, bot], fill=255)


def pictograms(ppm, which=("up", "fragile", "dry")):
    """A row of framed handling pictograms (ink mask). Cell 20 x 20 mm, 2 mm gap."""
    n = len(which)
    w_mm, h_mm = n * 20 + (n - 1) * 2, 20
    im, d, k = _canvas(w_mm, h_mm, ppm)
    lw = max(1, int(round(0.9 * k)))
    for i, kind in enumerate(which):
        x0 = (i * 22) * k
        x1, y0, y1 = x0 + 20 * k - 1, 0, 20 * k - 1
        d.rectangle([x0, y0, x1, y1], outline=255, width=lw)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        if kind == "up":
            for dx in (-4.0, 4.0):
                _arrow_up(d, cx + dx * k, 3.5 * k, 14.5 * k, 2.8 * k, k, 1.4 * k)
            d.rectangle([cx - 7.5 * k, 15.8 * k, cx + 7.5 * k, 17.0 * k], fill=255)
        elif kind == "fragile":
            # wine glass: bowl, stem, foot; a crack in the bowl
            d.polygon([(cx - 5 * k, 3.5 * k), (cx + 5 * k, 3.5 * k), (cx + 4.2 * k, 8 * k), (cx + 1.0 * k, 10.5 * k),
                       (cx - 1.0 * k, 10.5 * k), (cx - 4.2 * k, 8 * k)], fill=255)
            d.line([(cx - 1.8 * k, 3.5 * k), (cx - 0.4 * k, 5.6 * k), (cx - 2.0 * k, 7.0 * k), (cx - 0.6 * k, 9.0 * k)],
                   fill=0, width=max(1, int(0.7 * k)))
            d.rectangle([cx - 0.6 * k, 10.5 * k, cx + 0.6 * k, 15.2 * k], fill=255)
            d.rectangle([cx - 4.0 * k, 15.2 * k, cx + 4.0 * k, 16.6 * k], fill=255)
        elif kind == "dry":
            # umbrella with drops
            d.pieslice([cx - 7 * k, 5 * k, cx + 7 * k, 17 * k], 180, 360, fill=255)
            d.rectangle([cx - 0.5 * k, 10.8 * k, cx + 0.5 * k, 16.0 * k], fill=255)
            d.arc([cx - 0.5 * k, 14.6 * k, cx + 2.6 * k, 17.4 * k], 0, 180, fill=255, width=max(1, int(0.9 * k)))
            for dx, dy in ((-5, 2.6), (0, 1.6), (5, 2.6)):
                d.ellipse([cx + (dx - 0.7) * k, (dy - 0.9) * k, cx + (dx + 0.7) * k, (dy + 0.9) * k], fill=255)
        elif kind == "recycle":
            r = 6.5 * k
            for j in range(3):
                a0 = j * 120 + 15
                d.arc([cx - r, cy - r, cx + r, cy + r], a0, a0 + 85, fill=255, width=max(1, int(1.6 * k)))
                a = math.radians(a0 + 85)
                px, py = cx + r * math.cos(a), cy + r * math.sin(a)
                t = a + math.pi / 2
                d.polygon([(px + 2.4 * k * math.cos(t), py + 2.4 * k * math.sin(t)),
                           (px + 1.8 * k * math.cos(t - 2.2), py + 1.8 * k * math.sin(t - 2.2)),
                           (px + 1.8 * k * math.cos(t + 2.2), py + 1.8 * k * math.sin(t + 2.2))], fill=255)
    return {"ink": _finish(im), "size_mm": (w_mm, h_mm), "kind": "pictograms", "which": list(which)}


def maker_stamp(ppm, rng):
    """Round board-maker's stamp (generic wording only)."""
    D = 34.0
    im, d, k = _canvas(D, D, ppm)
    lw = max(1, int(round(0.8 * k)))
    d.ellipse([0, 0, D * k - 1, D * k - 1], outline=255, width=lw)
    d.ellipse([2.2 * k, 2.2 * k, (D - 2.2) * k, (D - 2.2) * k], outline=255, width=max(1, lw // 2))
    lines = (("CORRUGATED", 3.0, "bold"), ("SINGLE WALL", 2.6, "medium"),
             ("ECT %d" % int(rng.choice([26, 29, 32, 44])), 4.2, "bold"), ("BOARD TEST", 2.4, "medium"))
    y = 9.2
    for t, cap, kind in lines:
        f = _font(kind, cap / 0.711 * k)
        d.text((D / 2 * k, (y + cap) * k), t, font=f, fill=255, anchor="ms")
        y += cap + 1.9
    return {"ink": _finish(im), "size_mm": (D, D), "kind": "maker_stamp"}


def fragile_sticker(ppm):
    """Red sticker 64 x 26 mm, white glass icon and the word FRAGILE (colour sticker, no white field)."""
    w, h = 64.0, 26.0
    im, d, k = _canvas(w, h, ppm)            # mask of the white print on the red sticker
    cx = 11 * k
    d.polygon([(cx - 5 * k, 4 * k), (cx + 5 * k, 4 * k), (cx + 4.2 * k, 9 * k), (cx + 1 * k, 12 * k),
               (cx - 1 * k, 12 * k), (cx - 4.2 * k, 9 * k)], fill=255)
    d.rectangle([cx - 0.6 * k, 12 * k, cx + 0.6 * k, 19 * k], fill=255)
    d.rectangle([cx - 4 * k, 19 * k, cx + 4 * k, 20.6 * k], fill=255)
    f = _font("bold", 11.0 / 0.711 * k)
    d.text((40 * k, 18.5 * k), "FRAGILE", font=f, fill=255, anchor="ms")
    return {"print": _finish(im), "size_mm": (w, h), "kind": "fragile_sticker",
            "sticker_srgb": (0.80, 0.13, 0.11), "print_srgb": (0.95, 0.94, 0.92), "corner_mm": 2.0}


def lot_sticker(ppm, rng):
    """Small plain sticker 36 x 20 mm with two text lines (no barcode, no 2-D code)."""
    w, h = 36.0, 20.0
    im, d, k = _canvas(w, h, ppm)
    f1 = _font("bold", 5.0 / 0.711 * k)
    f2 = _font("medium", 2.6 / 0.711 * k)
    d.text((3 * k, 8.6 * k), "LOT %04d" % int(rng.integers(0, 10000)), font=f1, fill=255, anchor="ls")
    d.text((3 * k, 15.6 * k), "%02d/%02d · %s" % (int(rng.integers(1, 13)), int(rng.integers(20, 27)),
                                                 "ABCDEFGH"[int(rng.integers(8))]), font=f2, fill=255, anchor="ls")
    tint = (0.90, 0.86, 0.62) if rng.uniform() < 0.5 else (0.86, 0.90, 0.93)   # pale yellow or pale blue paper
    return {"print": _finish(im), "size_mm": (w, h), "kind": "lot_sticker",
            "sticker_srgb": tint, "print_srgb": (0.10, 0.10, 0.12), "corner_mm": 1.5}


BLANK_SIZES_MM = ((100.0, 150.0), (76.0, 102.0), (90.0, 125.0), (70.0, 110.0), (100.0, 170.0))
BLANK_TINTS = (((0.930, 0.930, 0.912), 0.55), ((0.940, 0.910, 0.820), 0.25), ((0.860, 0.862, 0.850), 0.20))


def blank_sticker(ppm, rng):
    """Plain paper sticker of a waybill-like size (a cover-up or a spare label): white, cream or pale grey paper, no
    print at all (no barcode, no 2-D code, no text)."""
    w, h = BLANK_SIZES_MM[int(rng.integers(len(BLANK_SIZES_MM)))]
    tint = BLANK_TINTS[int(rng.choice(len(BLANK_TINTS), p=[t[1] for t in BLANK_TINTS]))][0]
    im, d, k = _canvas(w, h, ppm)
    return {"print": _finish(im), "size_mm": (w, h), "kind": "blank_sticker",
            "sticker_srgb": tint, "print_srgb": (0.10, 0.10, 0.12), "corner_mm": 2.5}


def arrow_sticker(ppm):
    """Orange sticker 40 x 40 mm with two up arrows."""
    w, h = 40.0, 40.0
    im, d, k = _canvas(w, h, ppm)
    for dx in (-7.0, 7.0):
        _arrow_up(d, (20 + dx) * k, 6 * k, 30 * k, 6 * k, k, 3.4 * k)
    d.rectangle([5 * k, 32.5 * k, 35 * k, 35 * k], fill=255)
    return {"print": _finish(im), "size_mm": (w, h), "kind": "arrow_sticker",
            "sticker_srgb": (0.93, 0.47, 0.10), "print_srgb": (0.08, 0.07, 0.07), "corner_mm": 2.0}


def resin_mark(ppm):
    """Mailer film print: chasing-arrow triangle with 4 and LDPE under it (the generic resin code)."""
    w, h = 16.0, 19.0
    im, d, k = _canvas(w, h, ppm)
    lw = max(1, int(round(1.1 * k)))
    pts = [(8 * k, 1.2 * k), (15 * k, 13 * k), (1 * k, 13 * k)]
    for i in range(3):
        a, b = pts[i], pts[(i + 1) % 3]
        a2 = (a[0] + 0.12 * (b[0] - a[0]), a[1] + 0.12 * (b[1] - a[1]))
        b2 = (a[0] + 0.80 * (b[0] - a[0]), a[1] + 0.80 * (b[1] - a[1]))
        d.line([a2, b2], fill=255, width=lw)
    d.text((8 * k, 10.6 * k), "4", font=_font("bold", 5.0 / 0.711 * k), fill=255, anchor="ms")
    d.text((8 * k, 18.4 * k), "LDPE", font=_font("bold", 3.0 / 0.711 * k), fill=255, anchor="ms")
    return {"ink": _finish(im), "size_mm": (w, h), "kind": "resin_mark"}


def describe() -> dict:
    return {
        "pictograms": "printed straight on the board in dark ink, a row of framed 20 x 20 mm symbols (this way up, "
                      "fragile, keep dry); no white field",
        "maker_stamp": "round 34 mm ring stamp in dark ink with generic board wording; no white field",
        "fragile_sticker": "red 64 x 26 mm sticker, white glass icon and the word FRAGILE",
        "lot_sticker": "pale yellow or pale blue 36 x 20 mm sticker with two text lines; no barcode, no 2-D code",
        "arrow_sticker": "orange 40 x 40 mm sticker with two black up arrows",
        "resin_mark": "16 x 19 mm chasing-arrow triangle with 4 / LDPE printed on mailer film",
        "blank_sticker": "plain paper sticker of a waybill-like size (70 x 110 to 100 x 170 mm), white, cream or pale "
                         "grey, with no print at all",
        "tape": "packing tape strips 38 to 72 mm wide: clear, brown, white, or matt kraft paper tape",
        "film": "mailer film white, grey or black, matt or glossy",
    }
