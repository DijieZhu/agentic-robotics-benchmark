#!/usr/bin/env python3
"""The pictures the cameras' scene draws itself (PIL / numpy; no text, no brand, no mark).

- every SKU's print as one atlas (box_atlas, wrap_atlas): the task's body colour, its band round the unit at the
  task's height and its logo (shape, colour, size and place from shelf_world.add_look) carry the identity; fine
  print, a panel of lines, a bar block and a table grid are drawn in tints of those colours beside them. A box: the
  four sides in a row (front = the unit's -x face, then going to the viewer's right), top and bottom under them. A
  can or bottle: the wrap (u = the angle from +x, v = height).
- code_label: the task's best-before code as a label: two seven-segment digits in the task's proportions (7 x 4 mm,
  0.8 mm strokes, 2 mm apart) and ink colour on the task's 11.5 x 10 mm light patch.
- dress_skus: the products of the neighbouring bays, from a palette of their own.
The shelf's ticket strip, the floor, the ceiling tiles, the back panel and the aisle sign are in the render bundle
(the functions that drew them are kept here).
"""
import argparse
import json
import math
import os
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

PPM = 8000.0            # pixels per metre on a product's print (0.125 mm a pixel)
CODE_PPM = 60000.0      # the code label: 0.0167 mm a pixel


def s8(c):
    return tuple(int(round(max(0.0, min(1.0, float(x))) * 255)) for x in c[:3])


def tint(c, k):
    """k > 0 towards white, k < 0 towards black"""
    c = np.asarray(c[:3], float)
    return tuple((c + (1 - c) * k) if k >= 0 else (c * (1 + k)))


def lum(c):
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def noise(shape, rng, scale=1.0, blur=0.0):
    n = rng.normal(0, 1, shape).astype(np.float32)
    if blur > 0:
        im = Image.fromarray(np.clip(n * 40 + 128, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(blur))
        n = (np.asarray(im).astype(np.float32) - 128) / 40.0
        n /= (n.std() + 1e-6)
    return n * scale


def fine_print(d, x0, y0, x1, y1, col, rng, pitch, h):
    """rows of short dashes: reads as small print from a distance, says nothing"""
    y = y0
    while y + h <= y1:
        x = x0
        while x < x1 - 4:
            w = float(rng.uniform(3, 14)) * h
            if rng.random() < 0.88:
                d.rectangle([x, y, min(x + w, x1), y + h], fill=col)
            x += w + float(rng.uniform(1.2, 2.5)) * h
        y += pitch


def bar_block(d, x0, y0, x1, y1, rng, ink=(20, 20, 24), paper=(244, 244, 240)):
    d.rectangle([x0, y0, x1, y1], fill=paper)
    x = x0 + 0.06 * (x1 - x0)
    unit = max(1.0, (x1 - x0) / 70.0)
    while x < x1 - 0.06 * (x1 - x0):
        w = unit * float(rng.integers(1, 4))
        d.rectangle([x, y0 + 0.08 * (y1 - y0), x + w, y1 - 0.22 * (y1 - y0)], fill=ink)
        x += w + unit * float(rng.integers(1, 3))


def table_grid(d, x0, y0, x1, y1, col, rows, lw):
    d.rectangle([x0, y0, x1, y1], outline=col, width=lw)
    for i in range(1, rows):
        y = y0 + (y1 - y0) * i / rows
        d.line([x0, y, x1, y], fill=col, width=max(1, lw - 1) if i > 1 else lw + 1)
    d.line([x0 + 0.62 * (x1 - x0), y0 + (y1 - y0) / rows, x0 + 0.62 * (x1 - x0), y1], fill=col, width=max(1, lw - 1))


def logo(d, cx, cy, s, shape, col, ppm):
    s *= ppm
    edge = s8(tint(col, -0.35))
    lw = max(1, int(0.0003 * ppm))
    if shape == "square":
        d.rectangle([cx - s, cy - s, cx + s, cy + s], fill=s8(col), outline=edge, width=lw)
    elif shape == "disc":
        d.ellipse([cx - s, cy - s, cx + s, cy + s], fill=s8(col), outline=edge, width=lw)
    else:
        for dy in (-0.5 * s, 0.5 * s):
            d.rectangle([cx + dy - 0.22 * s, cy - s, cx + dy + 0.22 * s, cy + s], fill=s8(col), outline=edge, width=lw)


def finish_print(im, rng, grain=2.2, ao=None):
    a = np.asarray(im).astype(np.float32)
    a += noise(a.shape[:2], rng, grain, 0.6)[..., None]
    a += noise(a.shape[:2], rng, grain * 0.8, 6.0)[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def logo_half(sku, hh):
    look = sku["look"]
    hy = sku["size"][1] if sku["kind"] == "box" else sku["size"][0]
    return 0.45 * min(hy, look["band_h"] * hh + 0.01), look["logo_z"] * hh


def box_atlas(sku, seed):
    """-> (image, roughness image, layout): a box's print. Atlas in metres: sides in a row (front, the -y side, back,
    the +y side), each 2 hz tall; under them the top (2 hy x 2 hx) and the bottom."""
    rng = np.random.default_rng(seed)
    hx, hy, hz = sku["size"]
    look = sku["look"]
    widths = [2 * hy, 2 * hx, 2 * hy, 2 * hx]
    Wm, Hm = sum(widths), 2 * hz + 2 * hx + 0.004
    W, H = int(round(Wm * PPM)), int(round(Hm * PPM))
    body = sku["rgba"][:3]
    im = Image.new("RGB", (W, H), s8(body))
    d = ImageDraw.Draw(im)
    px = lambda m: m * PPM                                                    # noqa: E731
    zy = lambda z: px(hz - z)                                                 # noqa: E731  unit z -> atlas y (top = hz)
    zl = look["logo_z"] * hz
    bh = look["band_h"] * hz
    dark = lum(body) > 0.45
    ink = s8(tint(body, -0.62) if dark else tint(body, 0.72))
    soft = s8(tint(body, -0.16) if dark else tint(body, 0.16))
    # the band round all four sides, with two keylines
    d.rectangle([0, zy(zl + bh), W, zy(zl - bh)], fill=s8(look["band"]))
    kl = s8(tint(look["band"], 0.55 if lum(look["band"]) < 0.5 else -0.45))
    for zz in (zl + bh - 0.0012, zl - bh + 0.0012):
        d.line([0, zy(zz), W, zy(zz)], fill=kl, width=max(1, int(px(0.0004))))
    x0s = np.concatenate([[0], np.cumsum(widths)])
    s, _ = logo_half(sku, hz)
    for k in range(4):
        x0, x1 = px(x0s[k]), px(x0s[k + 1])
        m = px(0.004)
        top_free = (zy(hz) + m, zy(zl + bh) - m)
        bot_free = (zy(zl - bh) + m, zy(-hz) - m)
        big = top_free if top_free[1] - top_free[0] > bot_free[1] - bot_free[0] else bot_free
        small = bot_free if big is top_free else top_free
        if k == 0:                                                            # the front: the logo, a title block
            logo(d, (x0 + x1) / 2, zy(zl), s, look["logo_shape"], look["logo"], PPM)
            if big[1] - big[0] > px(0.02):
                y = big[0] + 0.18 * (big[1] - big[0])
                for i, wk in enumerate((0.72, 0.5)):
                    hgt = px(0.0052 if i == 0 else 0.0026)
                    d.rounded_rectangle([x0 + m, y, x0 + m + wk * (x1 - x0 - 2 * m), y + hgt], radius=hgt / 2, fill=ink if i == 0 else soft)
                    y += hgt + px(0.0028)
                fine_print(d, x0 + m, y + px(0.002), x1 - m, min(big[1], y + px(0.012)), soft, rng, px(0.0017), px(0.0007))
            if small[1] - small[0] > px(0.008):
                r_ = min(px(0.0045), 0.4 * (small[1] - small[0]))
                d.rounded_rectangle([x1 - m - 3.2 * r_, small[1] - 2 * r_, x1 - m, small[1]], radius=r_, outline=ink, width=max(1, int(px(0.00035))))
        elif k == 2:                                                          # the back: a panel of lines, the bar block
            if big[1] - big[0] > px(0.03):
                d.rounded_rectangle([x0 + m, big[0], x1 - m, big[1]], radius=px(0.0015), fill=s8(tint(body, 0.86)))
                fine_print(d, x0 + m + px(0.002), big[0] + px(0.003), x1 - m - px(0.002), big[1] - px(0.016), (70, 70, 74), rng, px(0.0016), px(0.0006))
                bar_block(d, x0 + m + px(0.003), big[1] - px(0.014), x0 + m + px(0.003) + min(px(0.026), x1 - x0 - 2 * m - px(0.006)), big[1] - px(0.002), rng)
            if small[1] - small[0] > px(0.008):
                fine_print(d, x0 + m, small[0], x1 - m, small[1], soft, rng, px(0.0017), px(0.0007))
        else:                                                                 # the sides: a table, a line block
            if big[1] - big[0] > px(0.03):
                gh = min(big[1] - big[0], px(0.05))
                table_grid(d, x0 + m, big[0], x1 - m, big[0] + gh, ink, 7, max(1, int(px(0.0003))))
                fine_print(d, x0 + m + px(0.0015), big[0] + px(0.0012), x1 - m - px(0.0015), big[0] + gh, soft, rng, gh / 7.0, px(0.0006))
            if small[1] - small[0] > px(0.006):
                fine_print(d, x0 + m, small[0], x1 - m, small[1], soft, rng, px(0.0018), px(0.0007))
        d.line([x1 - 1, 0, x1 - 1, px(2 * hz)], fill=s8(tint(body, -0.08)), width=1)       # the fold
    # top and bottom: the body colour, a glue flap line
    ty = px(2 * hz + 0.004)
    d.rectangle([0, px(2 * hz), W, H], fill=s8(tint(body, -0.04)))
    d.line([0, ty + px(hx), px(2 * hy), ty + px(hx)], fill=s8(tint(body, -0.22)), width=max(1, int(px(0.0004))))
    d.line([px(2 * hy + 0.004), ty + px(hx), px(4 * hy + 0.004), ty + px(hx)], fill=s8(tint(body, -0.22)), width=max(1, int(px(0.0004))))
    im = finish_print(im, rng)
    rgh = Image.fromarray(np.clip(92 + noise((H, W), rng, 9.0, 3.0), 40, 200).astype(np.uint8))   # 0.36 satin varnish
    lay = {"kind": "box", "Wm": Wm, "Hm": Hm, "x0": [float(v) for v in x0s[:4]], "top_y": 2 * hz + 0.004,
           "top_x": 0.0, "bot_x": 2 * hy + 0.004}
    return im, rgh, lay


def wrap_atlas(sku, seed, bottle=False):
    """a can's or bottle's printed wrap: u = the angle from the unit's +x axis (0..2 pi), v = height over the unit;
    the logos on the -x and +x sides (pi and 0)"""
    rng = np.random.default_rng(seed)
    r, hh = sku["size"]
    look = sku["look"]
    Wm, Hm = 2 * math.pi * r, 2 * hh
    W, H = int(round(Wm * PPM)), int(round(Hm * PPM))
    body = sku["rgba"][:3]
    im = Image.new("RGB", (W, H), s8(body))
    d = ImageDraw.Draw(im)
    px = lambda m: m * PPM                                                    # noqa: E731
    zy = lambda z: px(hh - z)                                                 # noqa: E731
    zl, bh = look["logo_z"] * hh, look["band_h"] * hh
    dark = lum(body) > 0.45
    ink = s8(tint(body, -0.62) if dark else tint(body, 0.72))
    soft = s8(tint(body, -0.16) if dark else tint(body, 0.16))
    d.rectangle([0, zy(zl + bh), W, zy(zl - bh)], fill=s8(look["band"]))
    kl = s8(tint(look["band"], 0.55 if lum(look["band"]) < 0.5 else -0.45))
    for zz in (zl + bh - 0.0012, zl - bh + 0.0012):
        d.line([0, zy(zz), W, zy(zz)], fill=kl, width=max(1, int(px(0.0004))))
    s, _ = logo_half(sku, hh)
    for ang in (0.0, math.pi, 2 * math.pi):
        logo(d, px(ang * r), zy(zl), s, look["logo_shape"], look["logo"], PPM)
    m = px(0.004)
    top_free = (zy(hh) + px(0.008), zy(zl + bh) - m)
    bot_free = (zy(zl - bh) + m, zy(-hh) - px(0.008))
    big = top_free if top_free[1] - top_free[0] > bot_free[1] - bot_free[0] else bot_free
    for q in (0.5 * math.pi, 1.5 * math.pi):                                  # between the logos: a panel of lines
        cx = px(q * r)
        hw = px(0.021)
        if big[1] - big[0] > px(0.02):
            d.rounded_rectangle([cx - hw, big[0], cx + hw, big[1]], radius=px(0.0015), fill=s8(tint(body, 0.84)))
            fine_print(d, cx - hw + px(0.002), big[0] + px(0.003), cx + hw - px(0.002), big[1] - (px(0.014) if q < 3 else px(0.002)),
                       (70, 70, 74), rng, px(0.0016), px(0.0006))
            if q < 3:
                bar_block(d, cx - hw + px(0.003), big[1] - px(0.012), cx + hw - px(0.01), big[1] - px(0.002), rng)
    for ang in (0.0, math.pi, 2 * math.pi):                                   # a title block over or under each logo
        cx = px(ang * r)
        if big[1] - big[0] > px(0.012):
            y = big[0] + 0.25 * (big[1] - big[0])
            hgt = px(0.0045)
            d.rounded_rectangle([cx - px(0.018), y, cx + px(0.018), y + hgt], radius=hgt / 2, fill=ink)
            d.rounded_rectangle([cx - px(0.012), y + hgt + px(0.002), cx + px(0.012), y + hgt + px(0.0045)], radius=px(0.001), fill=soft)
    im = finish_print(im, rng)
    rgh = Image.fromarray(np.clip((70 if not bottle else 105) + noise((H, W), rng, 8.0, 3.0), 30, 200).astype(np.uint8))
    return im, rgh, {"kind": "wrap", "Wm": Wm, "Hm": Hm}


def code_label(code, paper, ink, sizes):
    """the task's code on its patch (add_code): digits h x w with stroke t, their centres (w + gap) apart, on a patch
    of half sizes pu x pv; a label's paper grain, slightly soft ink edges"""
    pu, pv = sizes["patch"]
    h, w, t, gap = sizes["h"], sizes["w"], sizes["stroke"], sizes["gap"]
    W, H = int(round(2 * pu * CODE_PPM)), int(round(2 * pv * CODE_PPM))
    im = Image.new("RGB", (W, H), s8(paper))
    d = ImageDraw.Draw(im)
    seg = {"a": (0, h / 2, w / 2, t / 2), "d": (0, -h / 2, w / 2, t / 2), "g": (0, 0, w / 2, t / 2),
           "b": (w / 2, h / 4, t / 2, h / 4), "c": (w / 2, -h / 4, t / 2, h / 4),
           "e": (-w / 2, -h / 4, t / 2, h / 4), "f": (-w / 2, h / 4, t / 2, h / 4)}
    S = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc", 5: "afgcd", 6: "afgedc", 7: "abc", 8: "abcdefg", 9: "abcdfg"}
    for i, digit in enumerate((code // 10 % 10, code % 10)):
        u0 = (i - 0.5) * (w + gap)
        for name in S[digit]:
            du, dv, su, sv = seg[name]
            cx, cy = W / 2 + (u0 + du) * CODE_PPM, H / 2 - dv * CODE_PPM
            d.rectangle([cx - su * CODE_PPM, cy - sv * CODE_PPM, cx + su * CODE_PPM, cy + sv * CODE_PPM], fill=s8(ink))
    rng = np.random.default_rng(1000 + code)
    im = im.filter(ImageFilter.GaussianBlur(1.1))
    a = np.asarray(im).astype(np.float32) + noise((H, W), rng, 1.6, 0.8)[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


DRESS_PALETTE = [(0.80, 0.72, 0.52), (0.62, 0.20, 0.18), (0.24, 0.36, 0.56), (0.36, 0.52, 0.34), (0.55, 0.38, 0.52),
                 (0.86, 0.56, 0.24), (0.24, 0.50, 0.52), (0.48, 0.36, 0.26), (0.84, 0.80, 0.50), (0.34, 0.34, 0.60),
                 (0.72, 0.38, 0.46), (0.22, 0.25, 0.30), (0.90, 0.89, 0.85), (0.50, 0.58, 0.28), (0.16, 0.30, 0.24),
                 (0.70, 0.16, 0.30), (0.93, 0.76, 0.70), (0.38, 0.62, 0.72)]
DRESS_KINDS = {"box": (0.025, 0.035, 0.07), "tallbox": (0.030, 0.030, 0.10), "can": (0.033, 0.061), "bottle": (0.032, 0.095),
               "flatbox": (0.022, 0.060, 0.085), "tin": (0.040, 0.030), "jar": (0.036, 0.070), "bigbox": (0.035, 0.095, 0.125)}


def dress_skus(n=28, seed=158):
    """SKUs for the neighbouring bays (people's media only): the task's kinds and a few more sizes, a palette of
    their own. The same dictionary shape as a task SKU."""
    rng = np.random.default_rng(seed)
    out = {}
    kinds = list(DRESS_KINDS)
    for i in range(n):
        kind = kinds[i % len(kinds)]
        base = DRESS_KINDS[kind]
        sc = rng.uniform(0.93, 1.06, 3)
        if len(base) == 3:
            size = [base[0] * sc[0], base[1] * sc[1], base[2] * sc[2]]
            k = "box"
        else:
            size = [base[0] * sc[0], base[1] * sc[2]]
            k = "cyl"
        body, band, lg, cap = (DRESS_PALETTE[int(j)] for j in rng.choice(len(DRESS_PALETTE), 4, replace=False))
        out["d%02d_%s" % (i, kind)] = {"kind": k, "size": [round(float(v), 4) for v in size], "rgba": list(body) + [1.0], "dress_kind": kind,
                                       "look": {"band": list(band), "logo": list(lg), "logo_shape": ("square", "disc", "bars")[int(rng.integers(3))],
                                                "logo_z": round(float(rng.uniform(-0.3, 0.2)), 3), "band_h": round(float(rng.uniform(0.16, 0.36)), 3),
                                                "cap": list(cap)}}
    return out


def ticket_strip(out, rng):
    """the shelf edge's ticket rail, 0.50 m of it (tiles along the shelf): a dark channel with a clear cover, paper
    tickets at a 0.125 m pitch with bars and lines (no figures)"""
    ppm = 6000.0
    W, H = int(0.5 * ppm), int(0.03 * ppm)
    im = Image.new("RGB", (W, H), (46, 49, 54))
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(0.0035 * ppm), W, int(0.0265 * ppm)], fill=(58, 61, 66))
    for k in range(4):
        cx = (k + 0.5) * 0.125 * ppm + float(rng.uniform(-0.004, 0.004)) * ppm
        hw, y0, y1 = 0.024 * ppm, 0.005 * ppm, 0.025 * ppm
        d.rectangle([cx - hw, y0, cx + hw, y1], fill=(243, 242, 236) if k % 3 else (250, 226, 90))
        d.rectangle([cx - hw + 0.002 * ppm, y0 + 0.0025 * ppm, cx - hw + 0.019 * ppm, y0 + 0.0105 * ppm], fill=(28, 28, 32))
        fine_print(d, cx - hw + 0.002 * ppm, y0 + 0.0125 * ppm, cx + 0.004 * ppm, y1 - 0.0015 * ppm, (90, 90, 94), rng, 0.0022 * ppm, 0.0009 * ppm)
        bar_block(d, cx + 0.006 * ppm, y0 + 0.008 * ppm, cx + hw - 0.002 * ppm, y1 - 0.002 * ppm, rng, paper=(243, 242, 236))
    finish_print(im, rng, 1.5).save(os.path.join(out, "ticket_col.png"))


def floor_maps(out, rng, n=4096, tiles=4, tile_m=0.6):
    """a store floor, 2.4 x 2.4 m: large vinyl tiles (speckled warm light grey, tile to tile a slightly different
    tone), thin joints; waxed (roughness about 0.22) with duller traffic, scuffs and trolley lines; a normal map of
    the joints and a faint waviness"""
    col = np.zeros((n, n, 3), np.float32)
    rg = np.zeros((n, n), np.float32)
    hgt = np.zeros((n, n), np.float32)
    ts = n // tiles
    speck = noise((n, n), rng, 1.0, 0.7)
    cloud = noise((n, n), rng, 1.0, 40.0)
    base = np.array([198, 194, 184], np.float32)
    for i in range(tiles):
        for j in range(tiles):
            t = base + rng.normal(0, 3.0) + rng.normal(0, 1.2, 3)
            col[i * ts:(i + 1) * ts, j * ts:(j + 1) * ts] = t
    col += speck[..., None] * 7.0 + cloud[..., None] * 3.0
    chips = (noise((n, n), rng, 1.0, 1.6) > 1.9)
    col[chips] -= 26.0
    chips2 = (noise((n, n), rng, 1.0, 1.2) > 2.1)
    col[chips2] += 16.0
    j_w = max(2, int(0.002 / (tiles * tile_m) * n))
    for k in range(tiles):
        for a in (col, None):
            pass
        s = k * ts
        col[s:s + j_w, :] = (118, 114, 106)
        col[:, s:s + j_w] = (118, 114, 106)
        hgt[s:s + j_w, :] -= 1.0
        hgt[:, s:s + j_w] -= 1.0
    wear = np.clip(0.5 + 0.5 * noise((n, n), rng, 1.0, 90.0), 0, 1)
    rg[:] = 0.20 + 0.16 * wear + 0.03 * speck
    im = Image.new("L", (n, n), 0)
    d = ImageDraw.Draw(im)
    for _ in range(70):                                                       # scuffs and trolley lines
        x, y = rng.uniform(0, n, 2)
        ang = rng.normal(0, 0.25) + (0 if rng.random() < 0.8 else math.pi / 2)
        le = rng.uniform(0.05, 0.6) * n
        d.line([x, y, x + le * math.cos(ang), y + le * math.sin(ang)], fill=int(rng.uniform(40, 130)), width=int(rng.integers(1, 5)))
    sc = np.asarray(im.filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32) / 255.0
    rg += 0.35 * sc
    col -= 14.0 * sc[..., None]
    rg[hgt < 0] = 0.7
    Image.fromarray(np.clip(col, 0, 255).astype(np.uint8)).save(os.path.join(out, "floor_col.png"))
    v = np.clip(rg * 255, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([v, v, v, v], -1), "RGBA").save(os.path.join(out, "floor_rgh.png"))
    hgt = hgt * 0.5 + 0.12 * noise((n, n), rng, 1.0, 60.0)
    hb = np.asarray(Image.fromarray(np.clip(hgt * 60 + 128, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.5))).astype(np.float32)
    gx = np.roll(hb, -1, 1) - np.roll(hb, 1, 1)
    gy = np.roll(hb, -1, 0) - np.roll(hb, 1, 0)
    nrm = np.stack([-gx, -gy, np.full_like(gx, 14.0)], -1)                    # DirectX style green (flipped by the writer)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    Image.fromarray(np.clip(nrm * 127.5 + 127.5, 0, 255).astype(np.uint8)).save(os.path.join(out, "floor_nrm.png"))


def ceiling_maps(out, rng, n=2048):
    """suspended ceiling, 1.2 x 1.2 m of it: four 0.6 m mineral-fibre tiles (fissured), white T-bars"""
    col = np.full((n, n), 226.0, np.float32) + noise((n, n), rng, 3.0, 1.0) + noise((n, n), rng, 2.0, 30.0)
    pits = noise((n, n), rng, 1.0, 1.1) > 1.75
    col[pits] -= 34.0
    hgt = np.zeros((n, n), np.float32)
    hgt[pits] = -1.0
    t = int(0.024 / 1.2 * n)
    for s in (0, n // 2):
        for sl in (np.s_[max(0, s - t // 2):s + t // 2 + 1, :], np.s_[:, max(0, s - t // 2):s + t // 2 + 1]):
            col[sl] = 238.0
            hgt[sl] = 1.2
    col[-t // 2:, :] = 238.0
    col[:, -t // 2:] = 238.0
    hgt[-t // 2:, :] = 1.2
    hgt[:, -t // 2:] = 1.2
    v = np.clip(col, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([v, v, v], -1)).save(os.path.join(out, "ceil_col.png"))
    hb = np.asarray(Image.fromarray(np.clip(hgt * 50 + 128, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))).astype(np.float32)
    gx = np.roll(hb, -1, 1) - np.roll(hb, 1, 1)
    gy = np.roll(hb, -1, 0) - np.roll(hb, 1, 0)
    nrm = np.stack([-gx, -gy, np.full_like(gx, 10.0)], -1)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    Image.fromarray(np.clip(nrm * 127.5 + 127.5, 0, 255).astype(np.uint8)).save(os.path.join(out, "ceil_nrm.png"))


def perforated(out, rng, n=1024):
    """a gondola's back panel, 0.1 x 0.1 m: a 25 mm grid of 5 mm holes (albedo darkening + normal)"""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    p = n / 4.0
    dx, dy = (xx % p) - p / 2, (yy % p) - p / 2
    hole = (dx * dx + dy * dy) < (0.0025 / 0.1 * n) ** 2
    col = np.full((n, n), 236.0, np.float32) + noise((n, n), rng, 1.5, 2.0)
    col[hole] = 30.0
    v = np.clip(col, 0, 255).astype(np.uint8)
    Image.fromarray(np.stack([v, v, v], -1)).save(os.path.join(out, "perf_col.png"))
    hb = np.asarray(Image.fromarray(np.where(hole, 60, 128).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.2))).astype(np.float32)
    gx = np.roll(hb, -1, 1) - np.roll(hb, 1, 1)
    gy = np.roll(hb, -1, 0) - np.roll(hb, 1, 0)
    nrm = np.stack([-gx, -gy, np.full_like(gx, 12.0)], -1)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    Image.fromarray(np.clip(nrm * 127.5 + 127.5, 0, 255).astype(np.uint8)).save(os.path.join(out, "perf_nrm.png"))


def aisle_sign(out, rng):
    """a hanging aisle marker, 1.2 x 0.36 m: a deep blue-grey board with a white disc and pictogram bars (no words,
    no figures)"""
    ppm = 2500.0
    W, H = int(1.2 * ppm), int(0.36 * ppm)
    im = Image.new("RGB", (W, H), (38, 58, 78))
    d = ImageDraw.Draw(im)
    d.ellipse([0.05 * ppm, 0.05 * ppm, 0.31 * ppm, 0.31 * ppm], fill=(240, 240, 236))
    d.rounded_rectangle([0.135 * ppm, 0.10 * ppm, 0.225 * ppm, 0.26 * ppm], radius=0.012 * ppm, fill=(38, 58, 78))
    y = 0.075 * ppm
    for wk in (0.62, 0.48, 0.55):
        d.rounded_rectangle([0.38 * ppm, y, (0.38 + wk * 0.74) * ppm, y + 0.045 * ppm], radius=0.022 * ppm, fill=(240, 240, 236))
        y += 0.08 * ppm
    finish_print(im, rng, 1.2).save(os.path.join(out, "sign_col.png"))


def cardboard(out, rng, n=1024):
    col = np.array([168, 132, 92], np.float32)[None, None, :] + noise((n, n), rng, 5.0, 0.8)[..., None] + noise((n, n), rng, 6.0, 25.0)[..., None]
    flute = 4.0 * np.sin(np.arange(n) / n * 2 * math.pi * 60)[None, :, None]
    Image.fromarray(np.clip(col + flute, 0, 255).astype(np.uint8)).save(os.path.join(out, "carton_col.png"))



# ------------------------------------------------------------------------------------------- round 2: a more real print
from PIL import ImageFont  # noqa: E402

FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_C = "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf"
FONT_M = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
_SYL = ["ra", "no", "vel", "ti", "ma", "so", "len", "ka", "do", "mi", "ren", "ta", "lo", "vi", "sen", "pa", "ru", "ne", "ko", "da", "li", "mo"]
_fonts = {}


def _font(path, px):
    px = max(5, int(round(px)))
    if (path, px) not in _fonts:
        _fonts[(path, px)] = ImageFont.truetype(path, px)
    return _fonts[(path, px)]


def pseudo_word(rng, lo=2, hi=4):
    return "".join(_SYL[int(i)] for i in rng.integers(len(_SYL), size=int(rng.integers(lo, hi + 1))))


def text_block(d, x0, y0, x1, y1, col, rng, mm, ppm, font=FONT_C, lead=1.32):
    """lines of meaningless words set in a real face, `mm` tall: fine print"""
    px = mm * ppm / 1000.0
    f = _font(font, px)
    y = y0
    while y + px <= y1:
        x = x0
        last = rng.random() < 0.25
        while True:
            w = pseudo_word(rng)
            tw = d.textlength(w + " ", font=f)
            if x + tw > x1 or (last and x > x0 + 0.55 * (x1 - x0)):
                break
            d.text((x, y), w, fill=col, font=f)
            x += tw
        y += px * lead


def barcode(d, x0, y0, x1, y1, rng, ppm, digits=True):
    """an EAN-like block: quiet zone, guard bars, 12 digits' worth of bars, the figures under them"""
    d.rectangle([x0, y0, x1, y1], fill=(246, 246, 242))
    n = 95
    unit = (x1 - x0) * 0.86 / n
    xs = x0 + (x1 - x0) * 0.07
    fh = (y1 - y0) * 0.20 if digits else 0
    bits = [1, 0, 1] + [int(b) for b in rng.integers(0, 2, 42)] + [0, 1, 0, 1, 0] + [int(b) for b in rng.integers(0, 2, 42)] + [1, 0, 1]
    for i, b in enumerate(bits):
        if b:
            guard = i < 3 or 45 <= i < 50 or i >= 92
            d.rectangle([xs + i * unit, y0 + (y1 - y0) * 0.07, xs + (i + 1) * unit - 0.2, y1 - (fh * (0.55 if guard else 1.15)) - (y1 - y0) * 0.05], fill=(18, 18, 22))
    if digits:
        f = _font(FONT_M, fh)
        d.text((xs, y1 - fh * 1.12), "".join(str(int(v)) for v in rng.integers(0, 10, 13)), fill=(18, 18, 22), font=f)


def nutrition(d, x0, y0, x1, y1, ink, rng, ppm):
    lw = max(1, int(0.00025 * ppm))
    d.rectangle([x0, y0, x1, y1], outline=ink, width=lw + 1)
    rows = max(4, int((y1 - y0) / (0.0031 * ppm)))
    rh = (y1 - y0) / rows
    f = _font(FONT_C, rh * 0.62)
    fb = _font(FONT_B, rh * 0.70)
    for i in range(rows):
        y = y0 + i * rh
        if i:
            d.line([x0, y, x1, y], fill=ink, width=lw + (1 if i in (1, 2) else 0))
        if i == 0:
            d.text((x0 + rh * 0.3, y + rh * 0.12), pseudo_word(rng, 3, 4).capitalize(), fill=ink, font=fb)
        else:
            d.text((x0 + rh * 0.3, y + rh * 0.16), pseudo_word(rng, 2, 3), fill=ink, font=f)
            v = "%d %s" % (int(rng.integers(1, 480)), ("g", "mg", "kJ", "%")[int(rng.integers(4))])
            d.text((x1 - rh * 0.3 - d.textlength(v, font=f), y + rh * 0.16), v, fill=ink, font=f)


def batch_print(d, x, y, rng, ppm, col=(70, 70, 74)):
    f = _font(FONT_M, 0.0019 * ppm)
    d.text((x, y), "L%05d %s%d" % (int(rng.integers(10000, 99999)), "ABCDEF"[int(rng.integers(6))], int(rng.integers(1, 9))), fill=col, font=f)


def edge_wear(im, lines_x, lines_y, rng, ppm, body):
    """worn folds: along the given vertical and horizontal lines the print is rubbed towards the board's colour in
    a ragged band of about 0.5 mm"""
    a = np.asarray(im).astype(np.float32)
    H, W = a.shape[:2]
    board = np.array([200, 190, 172], np.float32) if lum(body) < 0.6 else np.array([150, 140, 128], np.float32)
    mask = np.zeros((H, W), np.float32)
    bw = max(2, int(0.0005 * ppm))
    for x in lines_x:
        x = int(round(x))
        mask[:, max(0, x - bw):min(W, x + bw)] = 1.0
    for y in lines_y:
        y = int(round(y))
        mask[max(0, y - bw):min(H, y + bw), :] = 1.0
    rag = np.clip(0.45 + 0.9 * noise((H, W), rng, 1.0, 2.5), 0, 1)
    spots = np.clip(noise((H, W), rng, 1.0, 5.0) - 2.4, 0, 1) * 0.5
    m = np.clip(mask * rag * 0.75 + spots, 0, 1)[..., None]
    return Image.fromarray(np.clip(a * (1 - m) + board * m, 0, 255).astype(np.uint8))


def box_atlas(sku, seed):     # noqa: F811  (round 2 replaces round 1's)
    rng = np.random.default_rng(seed)
    hx, hy, hz = sku["size"]
    look = sku["look"]
    widths = [2 * hy, 2 * hx, 2 * hy, 2 * hx]
    Wm, Hm = sum(widths), 2 * hz + 2 * hx + 0.004
    W, H = int(round(Wm * PPM)), int(round(Hm * PPM))
    body = sku["rgba"][:3]
    im = Image.new("RGB", (W, H), s8(body))
    d = ImageDraw.Draw(im)
    px = lambda m: m * PPM                                                    # noqa: E731
    zy = lambda z: px(hz - z)                                                 # noqa: E731
    zl = look["logo_z"] * hz
    bh = look["band_h"] * hz
    dark = lum(body) > 0.45
    ink = s8(tint(body, -0.66) if dark else tint(body, 0.78))
    soft = s8(tint(body, -0.30) if dark else tint(body, 0.34))
    d.rectangle([0, zy(zl + bh), W, zy(zl - bh)], fill=s8(look["band"]))
    kl = s8(tint(look["band"], 0.55 if lum(look["band"]) < 0.5 else -0.45))
    for zz in (zl + bh - 0.0012, zl - bh + 0.0012):
        d.line([0, zy(zz), W, zy(zz)], fill=kl, width=max(1, int(px(0.0004))))
    x0s = np.concatenate([[0], np.cumsum(widths)])
    s, _ = logo_half(sku, hz)
    name = pseudo_word(rng, 2, 3).capitalize()
    for k in range(4):
        x0, x1 = px(x0s[k]), px(x0s[k + 1])
        m = px(0.0035)
        top_free = (zy(hz) + m, zy(zl + bh) - m)
        bot_free = (zy(zl - bh) + m, zy(-hz) - m)
        big = top_free if top_free[1] - top_free[0] > bot_free[1] - bot_free[0] else bot_free
        small = bot_free if big is top_free else top_free
        bw_ = x1 - x0 - 2 * m
        if k == 0:
            logo(d, (x0 + x1) / 2, zy(zl), s, look["logo_shape"], look["logo"], PPM)
            if big[1] - big[0] > px(0.018):
                fs = min(px(0.0085), bw_ / max(3.2, 0.62 * len(name)))
                f = _font(FONT_B, fs)
                y = big[0] + 0.10 * (big[1] - big[0])
                d.text((x0 + m, y), name, fill=ink, font=f)
                y += fs * 1.25
                d.text((x0 + m, y), pseudo_word(rng, 3, 4), fill=soft, font=_font(FONT_R, fs * 0.42))
                y += fs * 0.75
                text_block(d, x0 + m, y, x1 - m, min(big[1], y + px(0.010)), soft, rng, 1.15, PPM)
            if small[1] - small[0] > px(0.007):
                f = _font(FONT_B, min(px(0.0042), 0.6 * (small[1] - small[0])))
                wt = "%d g" % (int(rng.integers(8, 90)) * 10)
                tw = d.textlength(wt, font=f)
                d.rounded_rectangle([x1 - m - tw - px(0.003), small[1] - f.size * 1.5, x1 - m, small[1]], radius=f.size * 0.5, outline=ink, width=max(1, int(px(0.0003))))
                d.text((x1 - m - tw - px(0.0015), small[1] - f.size * 1.32), wt, fill=ink, font=f)
        elif k == 2:
            if big[1] - big[0] > px(0.028):
                d.rounded_rectangle([x0 + m, big[0], x1 - m, big[1]], radius=px(0.0012), fill=s8(tint(body, 0.88)))
                bc_h = px(0.013)
                text_block(d, x0 + m + px(0.0018), big[0] + px(0.002), x1 - m - px(0.0018), big[1] - bc_h - px(0.003), (60, 60, 64), rng, 1.05, PPM)
                barcode(d, x0 + m + px(0.002), big[1] - bc_h - px(0.0015), x0 + m + px(0.002) + min(px(0.030), bw_ - px(0.004)), big[1] - px(0.0015), rng, PPM)
            if small[1] - small[0] > px(0.006):
                text_block(d, x0 + m, small[0], x1 - m, small[1] - px(0.003), soft, rng, 1.0, PPM)
                batch_print(d, x0 + m, small[1] - px(0.0026), rng, PPM, col=soft)
        else:
            if big[1] - big[0] > px(0.03):
                gh = min(big[1] - big[0], px(0.052))
                if k == 1:
                    nutrition(d, x0 + m, big[0], x1 - m, big[0] + gh, ink, rng, PPM)
                else:
                    text_block(d, x0 + m, big[0], x1 - m, big[0] + gh, soft, rng, 1.1, PPM)
            if small[1] - small[0] > px(0.006):
                text_block(d, x0 + m, small[0], x1 - m, small[1], soft, rng, 1.0, PPM)
    ty = px(2 * hz + 0.004)
    d.rectangle([0, px(2 * hz), W, H], fill=s8(tint(body, -0.04)))
    for xx in (0.0, px(2 * hy + 0.004)):
        d.line([xx, ty + px(hx), xx + px(2 * hy), ty + px(hx)], fill=s8(tint(body, -0.25)), width=max(1, int(px(0.0005))))
    im = edge_wear(im, [px(v) for v in x0s], [0, px(2 * hz) - 1, ty + 1, H - 1], rng, PPM, body)
    im = finish_print(im, rng)
    rgh = Image.fromarray(np.clip(92 + noise((H, W), rng, 9.0, 3.0), 40, 200).astype(np.uint8))
    lay = {"kind": "box", "Wm": Wm, "Hm": Hm, "x0": [float(v) for v in x0s[:4]], "top_y": 2 * hz + 0.004,
           "top_x": 0.0, "bot_x": 2 * hy + 0.004}
    return im, rgh, lay


def wrap_atlas(sku, seed, bottle=False):     # noqa: F811
    rng = np.random.default_rng(seed)
    r, hh = sku["size"]
    look = sku["look"]
    Wm, Hm = 2 * math.pi * r, 2 * hh
    W, H = int(round(Wm * PPM)), int(round(Hm * PPM))
    body = sku["rgba"][:3]
    im = Image.new("RGB", (W, H), s8(body))
    d = ImageDraw.Draw(im)
    px = lambda m: m * PPM                                                    # noqa: E731
    zy = lambda z: px(hh - z)                                                 # noqa: E731
    zl, bh = look["logo_z"] * hh, look["band_h"] * hh
    dark = lum(body) > 0.45
    ink = s8(tint(body, -0.66) if dark else tint(body, 0.78))
    soft = s8(tint(body, -0.30) if dark else tint(body, 0.34))
    d.rectangle([0, zy(zl + bh), W, zy(zl - bh)], fill=s8(look["band"]))
    kl = s8(tint(look["band"], 0.55 if lum(look["band"]) < 0.5 else -0.45))
    for zz in (zl + bh - 0.0012, zl - bh + 0.0012):
        d.line([0, zy(zz), W, zy(zz)], fill=kl, width=max(1, int(px(0.0004))))
    s, _ = logo_half(sku, hh)
    for ang in (0.0, math.pi, 2 * math.pi):
        logo(d, px(ang * r), zy(zl), s, look["logo_shape"], look["logo"], PPM)
    m = px(0.004)
    top_free = (zy(hh) + px(0.008), zy(zl + bh) - m)
    bot_free = (zy(zl - bh) + m, zy(-hh) - px(0.008))
    big = top_free if top_free[1] - top_free[0] > bot_free[1] - bot_free[0] else bot_free
    name = pseudo_word(rng, 2, 3).capitalize()
    for q in (0.5 * math.pi, 1.5 * math.pi):
        cx = px(q * r)
        hw = px(0.021)
        if big[1] - big[0] > px(0.02):
            d.rounded_rectangle([cx - hw, big[0], cx + hw, big[1]], radius=px(0.0012), fill=s8(tint(body, 0.86)))
            if q < 3:
                text_block(d, cx - hw + px(0.002), big[0] + px(0.002), cx + hw - px(0.002), big[1] - px(0.015), (60, 60, 64), rng, 1.05, PPM)
                barcode(d, cx - hw + px(0.003), big[1] - px(0.014), cx + hw - px(0.008), big[1] - px(0.0015), rng, PPM)
            else:
                nutrition(d, cx - hw + px(0.002), big[0] + px(0.002), cx + hw - px(0.002), big[1] - px(0.002), (60, 60, 64), rng, PPM)
    for ang in (0.0, math.pi, 2 * math.pi):
        cx = px(ang * r)
        if big[1] - big[0] > px(0.012):
            fs = min(px(0.0075), 0.45 * (big[1] - big[0]))
            f = _font(FONT_B, fs)
            tw = d.textlength(name, font=f)
            y = big[0] + 0.18 * (big[1] - big[0])
            d.text((cx - tw / 2, y), name, fill=ink, font=f)
            f2 = _font(FONT_R, fs * 0.42)
            w2 = pseudo_word(rng, 3, 4)
            d.text((cx - d.textlength(w2, font=f2) / 2, y + fs * 1.25), w2, fill=soft, font=f2)
    # the wrap's lap seam: a thin step at 3/4 of the way round
    xs = px(1.75 * math.pi * r)
    d.line([xs, 0, xs, H], fill=s8(tint(body, -0.22)), width=max(1, int(px(0.0003))))
    d.line([xs + px(0.0006), 0, xs + px(0.0006), H], fill=s8(tint(body, 0.18)), width=max(1, int(px(0.0002))))
    im = finish_print(im, rng)
    rgh = Image.fromarray(np.clip((70 if not bottle else 105) + noise((H, W), rng, 8.0, 3.0), 30, 200).astype(np.uint8))
    return im, rgh, {"kind": "wrap", "Wm": Wm, "Hm": Hm}


def ticket_strip(out, rng):     # noqa: F811
    """the shelf edge's ticket rail, 0.50 m of it: tickets at a 0.125 m pitch with a price in figures, a line of
    small print, a barcode strip"""
    ppm = 8000.0
    W, H = int(0.5 * ppm), int(0.03 * ppm)
    im = Image.new("RGB", (W, H), (46, 49, 54))
    d = ImageDraw.Draw(im)
    d.rectangle([0, int(0.0035 * ppm), W, int(0.0265 * ppm)], fill=(58, 61, 66))
    for k in range(4):
        cx = (k + 0.5) * 0.125 * ppm + float(rng.uniform(-0.004, 0.004)) * ppm
        hw, y0, y1 = 0.026 * ppm, 0.005 * ppm, 0.025 * ppm
        promo = (k == int(rng.integers(4)))
        paper = (250, 226, 90) if promo else (243, 242, 236)
        d.rectangle([cx - hw, y0, cx + hw, y1], fill=paper)
        price = "%d.%d9" % (int(rng.integers(0, 9)), int(rng.integers(0, 10)))
        fb = _font(FONT_B, 0.0115 * ppm)
        d.text((cx - hw + 0.0015 * ppm, y0 + 0.0004 * ppm), price, fill=(24, 24, 28), font=fb)
        fs = _font(FONT_C, 0.0026 * ppm)
        d.text((cx - hw + 0.0018 * ppm, y1 - 0.0064 * ppm), pseudo_word(rng, 2, 3) + " " + pseudo_word(rng, 2, 2), fill=(70, 70, 74), font=fs)
        d.text((cx - hw + 0.0018 * ppm, y1 - 0.0034 * ppm), "%d g  %d.%02d/kg" % (int(rng.integers(8, 90)) * 10, int(rng.integers(1, 30)), int(rng.integers(100))), fill=(70, 70, 74), font=fs)
        barcode(d, cx + 0.006 * ppm, y0 + 0.010 * ppm, cx + hw - 0.0015 * ppm, y1 - 0.0015 * ppm, rng, ppm, digits=False)
    finish_print(im, rng, 1.5).save(os.path.join(out, "ticket_col.png"))
