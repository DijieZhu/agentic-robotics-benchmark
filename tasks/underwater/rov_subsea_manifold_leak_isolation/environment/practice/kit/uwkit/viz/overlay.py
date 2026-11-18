"""2-D overlays for the demo videos (PIL): action caption, per-thruster rotor-speed bars, a top-view thrust diagram,
and per-tile labels with the expected spin direction (computed from the model, so a viewer can compare it with the
rendered rotor)."""
from __future__ import annotations

import math
import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FWD_RGB = (90, 205, 120)        # forward thrust (positive rotor speed)
REV_RGB = (245, 150, 60)        # reverse thrust
TXT_RGB = (235, 240, 245)
DIM_RGB = (170, 180, 190)
PANEL_RGBA = (8, 16, 24, 170)

_FONT_CANDIDATES = (
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "SC"),
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", "SC"),
)


@lru_cache(maxsize=32)
def font(size: int, bold: bool = False):
    """A CJK-capable font when the system has one (Noto Sans CJK SC), else DejaVu Sans, else PIL's default."""
    for path, want in _FONT_CANDIDATES[(1 if bold else 0):] + _FONT_CANDIDATES[:(1 if bold else 0)]:
        if os.path.exists(path):
            for idx in range(10):
                try:
                    f = ImageFont.truetype(path, size, index=idx)
                except OSError:
                    break
                if want in " ".join(f.getname()):
                    return f
    try:
        import matplotlib
        p = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf",
                         "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        return ImageFont.truetype(p, size)
    except Exception:
        return ImageFont.load_default()


def _panel(draw, box, rgba=PANEL_RGBA, radius=8):
    draw.rounded_rectangle(box, radius=radius, fill=rgba)


def spin_glyph(draw, centre, radius, sense, rgb, width=3):
    """Circular arrow: sense +1 = counter-clockwise on screen, -1 = clockwise, 0 = a dot."""
    cx, cy = centre
    if sense == 0:
        draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=rgb)
        return
    a0, a1 = 40, 320
    draw.arc([cx - radius, cy - radius, cx + radius, cy + radius], start=a0, end=a1, fill=rgb, width=width)
    # arrow head at the end of the arc in the direction of motion (PIL angles are clockwise from +x)
    end = a1 if sense < 0 else a0
    t = math.radians(end)
    px, py = cx + radius * math.cos(t), cy + radius * math.sin(t)
    d = 1.0 if sense < 0 else -1.0                                   # clockwise on screen = increasing PIL angle
    tx, ty = -math.sin(t) * d, math.cos(t) * d                       # tangent of motion
    nx, ny = math.cos(t), math.sin(t)
    s = max(5.0, radius * 0.75)
    pts = [(px + tx * s, py + ty * s), (px + nx * s * 0.65, py + ny * s * 0.65), (px - nx * s * 0.65, py - ny * s * 0.65)]
    draw.polygon(pts, fill=rgb)


def caption(img: Image.Image, lines, xy=(24, 20), sizes=(30, 19), max_w=None):
    """Top-left caption: first line large, the rest small."""
    d = ImageDraw.Draw(img, "RGBA")
    fonts = [font(sizes[0], True)] + [font(sizes[1])] * (len(lines) - 1)
    w = max(d.textlength(t, font=f) for t, f in zip(lines, fonts)) + 28
    if max_w:
        w = min(w, max_w)
    h = sum(f.size + 8 for f in fonts) + 16
    _panel(d, [xy[0] - 12, xy[1] - 10, xy[0] - 12 + w, xy[1] - 10 + h])
    y = xy[1]
    for t, f in zip(lines, fonts):
        d.text((xy[0], y), t, font=f, fill=TXT_RGB if f is fonts[0] else DIM_RGB)
        y += f.size + 8
    return img


def rpm_bars(img: Image.Image, rpm, rotors, xy, rpm_ref=3800.0, width=520, row_h=26, thrust=None, title=None):
    """One signed bar per thruster: length ~ |rpm| / rpm_ref, green = forward, orange = reverse, plus the visible spin
    sense seen from the nose cone (circular arrow) - the differential use of the thrusters at a glance."""
    d = ImageDraw.Draw(img, "RGBA")
    n = len(rpm)
    f, fs = font(17), font(15)
    title = title or "推进器转速 rpm（绿 = 正推 / 橙 = 反推；箭头 = 从鼻锥看的转向）"
    h = 40 + n * row_h + 10
    x0, y0 = xy
    _panel(d, [x0, y0, x0 + width, y0 + h])
    d.text((x0 + 12, y0 + 8), title, font=fs, fill=DIM_RGB)
    lab_w, val_w = 120, 78
    bx0 = x0 + lab_w + 34
    bx1 = x0 + width - val_w - 10
    mid = 0.5 * (bx0 + bx1)
    half = 0.5 * (bx1 - bx0)
    for k in range(n):
        y = y0 + 38 + k * row_h
        r = rotors.rotors[k]
        kind = "H" if abs(r.axis_B[2]) < 0.7 else "V"
        d.text((x0 + 12, y), f"{r.site[4:]:>1} {kind} {r.propeller:<3}", font=f, fill=TXT_RGB)
        sense = int(np.sign(r.spin_sign_forward * rpm[k])) if abs(rpm[k]) > 1e-6 else 0
        spin_glyph(d, (x0 + lab_w + 14, y + 12), 9, sense, TXT_RGB if sense else DIM_RGB, width=3)
        d.line([mid, y + 1, mid, y + row_h - 5], fill=DIM_RGB, width=1)
        L = half * min(1.0, abs(rpm[k]) / rpm_ref)
        col = FWD_RGB if rpm[k] > 0 else REV_RGB
        if L > 0.5:
            d.rectangle([mid, y + 4, mid + L, y + row_h - 8] if rpm[k] > 0 else [mid - L, y + 4, mid, y + row_h - 8],
                        fill=col)
        d.text((bx1 + 8, y), f"{rpm[k]:+5.0f}", font=f, fill=TXT_RGB)
    return h


def thrust_diagram(img: Image.Image, thrust, rotors, xy, size=260, thrust_ref=40.0, title=None):
    """Top view (x forward = up on screen, y port = left): horizontal thrusters as arrows of the force on the vehicle,
    vertical thrusters as rings (green = pushes the vehicle up, orange = down); length / size ~ sqrt(|T| / thrust_ref)."""
    d = ImageDraw.Draw(img, "RGBA")
    x0, y0 = xy
    _panel(d, [x0, y0, x0 + size, y0 + size + 30])
    fs = font(15)
    d.text((x0 + 10, y0 + 6), title or "俯视：推力方向（作用在车上）", font=fs, fill=DIM_RGB)
    cx, cy = x0 + size / 2, y0 + 30 + size / 2
    s = size / 0.62                                                       # px per metre

    def P(p):
        return (cx - p[1] * s, cy - p[0] * s)

    hx, hy = 0.229, 0.17
    d.rectangle([P((hx, hy))[0], P((hx, hy))[1], P((-hx, -hy))[0], P((-hx, -hy))[1]], outline=DIM_RGB, width=2)
    d.polygon([P((hx + 0.03, 0)), P((hx - 0.01, 0.03)), P((hx - 0.01, -0.03))], fill=DIM_RGB)
    f = font(15)
    for k, r in enumerate(rotors.rotors):
        c = r.centre_B
        T = float(thrust[k])
        col = FWD_RGB if T > 0 else REV_RGB
        q = P(c)
        if abs(r.axis_B[2]) < 0.7:
            L = 0.11 * min(1.3, math.sqrt(abs(T) / thrust_ref))
            dvec = r.axis_B * np.sign(T)
            tip = P(c + dvec * L)
            if L > 0.004:
                d.line([q, tip], fill=col, width=5)
                ang = math.atan2(tip[1] - q[1], tip[0] - q[0])
                hs = 11
                d.polygon([tip, (tip[0] - hs * math.cos(ang - 0.45), tip[1] - hs * math.sin(ang - 0.45)),
                           (tip[0] - hs * math.cos(ang + 0.45), tip[1] - hs * math.sin(ang + 0.45))], fill=col)
            d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], fill=TXT_RGB)
        else:
            up = -float(r.axis_B[2]) * T                                   # force on the body along +z
            rad = 4 + 18 * min(1.3, math.sqrt(abs(T) / thrust_ref))
            colv = FWD_RGB if up > 0 else REV_RGB
            if abs(T) > 0.05:
                d.ellipse([q[0] - rad, q[1] - rad, q[0] + rad, q[1] + rad], outline=colv, width=4)
                d.text((q[0] - 5, q[1] - 9), "↑" if up > 0 else "↓", font=f, fill=colv)
            else:
                d.ellipse([q[0] - 4, q[1] - 4, q[0] + 4, q[1] + 4], outline=DIM_RGB, width=2)
        d.text((q[0] + 8, q[1] + 4), r.site[4:], font=f, fill=TXT_RGB)
    return size + 30


def tile_label(img: Image.Image, rotor, rpm, thrust, cmd, mode_note=""):
    d = ImageDraw.Draw(img, "RGBA")
    w = img.size[0]
    _panel(d, [6, 6, w - 6, 92], radius=6)
    f, fs = font(19, True), font(15)
    sense = int(np.sign(rotor.spin_sign_forward * rpm)) if abs(rpm) > 1e-6 else 0
    word = {1: "逆时针", -1: "顺时针", 0: "停"}[sense]
    d.text((16, 12), f"{rotor.site}  {rotor.propeller} 桨  {'水平' if abs(rotor.axis_B[2]) < 0.7 else '垂直'}", font=f,
           fill=TXT_RGB)
    d.text((16, 40), f"指令 {cmd:+.2f}   {rpm:+5.0f} rpm   推力 {thrust:+5.1f} N", font=fs, fill=TXT_RGB)
    d.text((16, 62), f"从鼻锥看应为：{word}  {mode_note}", font=fs, fill=FWD_RGB if rpm > 0 else (REV_RGB if rpm < 0 else DIM_RGB))
    spin_glyph(d, (w - 40, 50), 20, sense, FWD_RGB if rpm > 0 else (REV_RGB if rpm < 0 else DIM_RGB), width=4)


def contact_sheet(frames, captions, cols=3, scale=0.5, title=None, pad=10):
    """Grid of keyframes with captions -> PIL image."""
    ims = [Image.fromarray(f).resize((int(f.shape[1] * scale), int(f.shape[0] * scale)), Image.LANCZOS) for f in frames]
    w, h = ims[0].size
    rows = int(math.ceil(len(ims) / cols))
    cap_h = 34
    top = 50 if title else 0
    sheet = Image.new("RGB", (cols * (w + pad) + pad, top + rows * (h + cap_h + pad) + pad), (18, 22, 28))
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((pad, 12), title, font=font(24, True), fill=TXT_RGB)
    for i, (im, c) in enumerate(zip(ims, captions)):
        r_, c_ = divmod(i, cols)
        x, y = pad + c_ * (w + pad), top + pad + r_ * (h + cap_h + pad)
        d.text((x, y + 4), c, font=font(18), fill=TXT_RGB)
        sheet.paste(im, (x, y + cap_h))
    return sheet
