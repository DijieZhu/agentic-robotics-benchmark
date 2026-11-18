"""Procedural textures (no downloaded art): concrete, block work, hazard stripes, zone tape, grating, checker
plate, the deployment pad, signs with text, trefoils, grid-marker fiducials. Deterministic (fixed seeds)."""

from __future__ import annotations

import math
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_FONT_CACHE = {}


def font(px: int):
    px = max(6, int(px))
    f = _FONT_CACHE.get(px)
    if f is None:
        try:
            f = ImageFont.load_default(size=px)
        except TypeError:  # old Pillow
            f = ImageFont.load_default()
        _FONT_CACHE[px] = f
    return f


def _noise(n, scale, seed):
    rs = np.random.RandomState(seed)
    acc = np.zeros((n, n))
    amp = 1.0
    k = max(2, n // scale)
    while k <= n:
        base = rs.rand(k, k)
        img = np.array(Image.fromarray((base * 255).astype(np.uint8)).resize((n, n), Image.BICUBIC)) / 255.0
        acc += amp * img
        amp *= 0.5
        k *= 2
    acc -= acc.min()
    acc /= max(acc.max(), 1e-9)
    return acc


def concrete(base=(150, 150, 146), n=512, seed=1, var=26, speckle=0.02):
    nz = _noise(n, 64, seed)
    rs = np.random.RandomState(seed + 7)
    img = np.array(base, float)[None, None, :] + (nz[..., None] - 0.5) * var
    sp = rs.rand(n, n) < speckle
    img[sp] -= 30
    return np.clip(img, 0, 255).astype(np.uint8)


def blockwork(base=(214, 214, 206), n=512, seed=2, rows=6, cols=3):
    img = concrete(base, n, seed, var=10, speckle=0.0).astype(float)
    h = n // rows
    w = n // cols
    for r in range(rows):
        y = r * h
        img[y:y + 2, :] -= 28
        off = (w // 2) if r % 2 else 0
        for c in range(cols + 1):
            x = (c * w + off) % n
            img[y:y + h, x:x + 2] -= 28
    return np.clip(img, 0, 255).astype(np.uint8)


def shield_concrete(n=512, seed=3):
    img = concrete((150, 148, 140), n, seed, var=34, speckle=0.03).astype(float)
    # formwork joints and tie holes
    for k in range(1, 4):
        img[k * n // 4 - 1:k * n // 4 + 1, :] -= 25
    for i in range(4):
        for j in range(4):
            cx, cy = int((i + 0.5) * n / 4), int((j + 0.5) * n / 4)
            yy, xx = np.ogrid[:n, :n]
            m = (xx - cx) ** 2 + (yy - cy) ** 2 < 36
            img[m] -= 60
    return np.clip(img, 0, 255).astype(np.uint8)


def stripes(c1=(235, 190, 20), c2=(25, 25, 25), n=256, period=64, angle=45):
    yy, xx = np.mgrid[:n, :n]
    a = math.radians(angle)
    t = (xx * math.cos(a) + yy * math.sin(a)) % period
    m = t < period / 2
    img = np.where(m[..., None], np.array(c1, np.uint8), np.array(c2, np.uint8))
    return img.astype(np.uint8)


def grating(n=256, pitch=16, bar=4, base=(60, 62, 64), hole=(18, 18, 20)):
    img = np.full((n, n, 3), hole, np.uint8)
    for k in range(0, n, pitch):
        img[k:k + bar, :] = base
        img[:, k:k + bar // 2] = base
    return img


def checker_plate(n=256, base=(120, 122, 124)):
    img = concrete(base, n, 9, var=12, speckle=0.0).astype(float)
    for i in range(0, n, 32):
        for j in range(0, n, 32):
            for (dx, dy, ang) in ((8, 8, 1), (24, 24, -1)):
                for t in range(-6, 7):
                    x = i + dx + t
                    y = j + dy + ang * t // 2
                    if 0 <= x < n and 0 <= y < n:
                        img[max(0, y - 1):y + 2, x] += 40
    return np.clip(img, 0, 255).astype(np.uint8)


def pad(n=512):
    img = Image.new("RGB", (n, n), (225, 185, 25))
    d = ImageDraw.Draw(img)
    b = n // 14
    d.rectangle([0, 0, n - 1, n - 1], outline=(20, 20, 20), width=b)
    c = n // 2
    d.line([c, b * 2, c, n - 2 * b], fill=(20, 20, 20), width=b // 2)
    d.line([b * 2, c, n - 2 * b, c], fill=(20, 20, 20), width=b // 2)
    d.ellipse([c - n // 6, c - n // 6, c + n // 6, c + n // 6], outline=(20, 20, 20), width=b // 3)
    d.text((c + n // 10, c - n // 3), "DATUM", font=font(n // 12), fill=(20, 20, 20))
    # an arrow pointing +x (into the building): image right = +x when the pad is seen from above
    d.polygon([(n - 3 * b, c), (n - 5 * b, c - b), (n - 5 * b, c + b)], fill=(20, 20, 20))
    return np.array(img)


def trefoil(n=256, text="RADIATION AREA", bg=(240, 205, 25), fg=(120, 20, 140)):
    img = Image.new("RGB", (n, n), bg)
    d = ImageDraw.Draw(img)
    c = n // 2
    cy = int(n * 0.42)
    R = n * 0.3
    r0 = n * 0.06
    for k in range(3):
        a0 = math.degrees(-math.pi / 2 + k * 2 * math.pi / 3) - 30
        d.pieslice([c - R, cy - R, c + R, cy + R], a0, a0 + 60, fill=fg)
    d.ellipse([c - r0 * 1.6, cy - r0 * 1.6, c + r0 * 1.6, cy + r0 * 1.6], fill=bg)
    d.ellipse([c - r0, cy - r0, c + r0, cy + r0], fill=fg)
    d.rectangle([2, 2, n - 3, n - 3], outline=(20, 20, 20), width=4)
    d.text((c, int(n * 0.86)), text, font=font(n // 12), fill=(20, 20, 20), anchor="mm")
    return np.array(img)


def notice(n=256, text="NOTICE", bg=(30, 90, 170), fg=(255, 255, 255)):
    img = Image.new("RGB", (n, n), (245, 245, 245))
    d = ImageDraw.Draw(img)
    d.ellipse([n * 0.2, n * 0.08, n * 0.8, n * 0.68], fill=bg)
    d.text((n // 2, int(n * 0.38)), "!", font=font(n // 3), fill=fg, anchor="mm")
    words = text.split(" ")
    d.text((n // 2, int(n * 0.78)), " ".join(words[:2]), font=font(n // 11), fill=(20, 20, 20), anchor="mm")
    if len(words) > 2:
        d.text((n // 2, int(n * 0.9)), " ".join(words[2:]), font=font(n // 11), fill=(20, 20, 20), anchor="mm")
    return np.array(img)


def room_sign(text, w=512, h=128, bg=(245, 245, 240), fg=(20, 30, 60)):
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([3, 3, w - 4, h - 4], outline=fg, width=5)
    s = int(h * 0.48)
    while s > 10:
        f = font(s)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
        if x1 - x0 < w - 30:
            break
        s -= 2
    d.text((w // 2, h // 2), text, font=font(s), fill=fg, anchor="mm")
    return np.array(img)


def room_plate(number, name):
    """Version 2: the room plate (0.40 x 0.10 m, white, a border line) in the plant lettering: the
    number at 45 mm cap height, the name at 20 mm (bricks.lettering: no font file, the same bytes on
    every machine)."""
    from bricks.lettering import room_plate_image
    return room_plate_image(str(number), str(name))


def survey_mark(text, n=512):
    """Version 2: the survey mark's floor disc with its tag in the plant lettering."""
    from bricks.lettering import survey_mark_image
    return survey_mark_image(str(text), n=n)


def tag_plate(text, w=256, h=128, bg=(250, 250, 250), fg=(10, 10, 10), border=(200, 30, 160)):
    """Placeholder tag plate (marked: magenta border)."""
    img = Image.new("RGB", (w, h), bg)
    d = ImageDraw.Draw(img)
    d.rectangle([2, 2, w - 3, h - 3], outline=border, width=6)
    s = int(h * 0.42)
    while s > 10:
        f = font(s)
        x0, y0, x1, y1 = d.textbbox((0, 0), text, font=f)
        if x1 - x0 < w - 24:
            break
        s -= 2
    d.text((w // 2, h // 2), text, font=font(s), fill=fg, anchor="mm")
    return np.array(img)


def fiducial(marker_id: int, label: str = "", n: int = 256, bits: int = 5):
    """A square binary fiducial (border + 5x5 code + label strip). Uses OpenCV's ArUco 5x5 dictionary when
    available so standard detectors can read it; else a hashed pattern."""
    code = None
    try:
        import cv2
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        code = cv2.aruco.generateImageMarker(d, int(marker_id) % 250, 200, borderBits=1)
    except Exception:  # noqa: BLE001
        rs = np.random.RandomState(1000 + int(marker_id))
        m = rs.rand(bits, bits) < 0.5
        code = np.zeros((bits + 2, bits + 2), np.uint8)
        code[1:-1, 1:-1] = np.where(m, 255, 0)
        code = np.array(Image.fromarray(code).resize((200, 200), Image.NEAREST))
    img = Image.new("RGB", (n, n), (255, 255, 255))
    img.paste(Image.fromarray(code).convert("RGB"), ((n - 200) // 2, 10))
    d = ImageDraw.Draw(img)
    d.text((n // 2, n - 24), label, font=font(30), fill=(0, 0, 0), anchor="mm")
    return np.array(img)


def floor_marker(text, n=256):
    img = Image.new("RGBA", (n, n), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse([4, 4, n - 5, n - 5], fill=(240, 240, 240, 255), outline=(200, 30, 160, 255), width=18)
    d.text((n // 2, n // 2), text, font=font(n // 5), fill=(20, 20, 20, 255), anchor="mm")
    return np.array(img.convert("RGB"))


def dial_placeholder(lo, hi, unit, value, n=256):
    """A placeholder gauge face (magenta ring marks it as a stand-in for the bricks gauge)."""
    img = Image.new("RGB", (n, n), (250, 250, 248))
    d = ImageDraw.Draw(img)
    c = n / 2
    d.ellipse([3, 3, n - 4, n - 4], outline=(200, 30, 160), width=10)
    for k in range(11):
        a = math.radians(225 - 27 * k)
        r0, r1 = n * 0.36, n * 0.43
        d.line([c + r0 * math.cos(a), c - r0 * math.sin(a), c + r1 * math.cos(a), c - r1 * math.sin(a)],
               fill=(20, 20, 20), width=4)
    d.text((c, c + n * 0.22), "%g-%g %s" % (lo, hi, unit), font=font(n // 11), fill=(20, 20, 20), anchor="mm")
    d.text((c, c - n * 0.18), "STAND-IN", font=font(n // 14), fill=(200, 30, 160), anchor="mm")
    return np.array(img)


def write_png(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        Image.fromarray(np.ascontiguousarray(img, np.uint8)).save(path, format="PNG", optimize=False)
    return path
