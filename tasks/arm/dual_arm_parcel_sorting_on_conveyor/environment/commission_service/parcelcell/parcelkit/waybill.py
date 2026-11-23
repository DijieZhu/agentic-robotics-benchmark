"""The fictional carrier's waybill: one design, two paper sizes (100 x 180 mm and 76 x 130 mm).

All dimensions in millimetres, origin at the label's top-left corner, x to the right (reading direction), y down.
The design is drawn for the 100 x 180 mm sheet; the 76 x 130 mm sheet uses the same design with x positions scaled by
76/100, y positions by 130/180, and type, codes and emblem scaled by the smaller factor (130/180).

render(identity, paper, px_per_mm) -> dict(rgb, ink, paper_mask): the sheet as an sRGB image (row 0 = the label's top
edge) plus the ink coverage (0 paper .. 1 ink), drawn at 2x and averaged; the barcode and the 2-D code are drawn with
exact per-pixel coverage.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import code128
from .identity import CARRIER, WaybillIdentity

PAPERS = {"100x180": (100.0, 180.0), "76x130": (76.0, 130.0)}
PAPER_SRGB = (0.935, 0.937, 0.925)       # white thermal paper
INK_SRGB = (0.075, 0.075, 0.085)         # thermal black
CAP = 0.711                              # Roboto cap height / em
_FONT_DIR = os.path.join(os.path.dirname(__file__), "fonts")
FONTS = {"bold": "RobotoCondensed-Bold.ttf", "medium": "RobotoCondensed-Medium.ttf",
         "regular": "RobotoCondensed-Regular.ttf"}

# zones of the 100 x 180 design: (name, y_top, y_bottom, rule thickness under the zone)
ZONES = (
    ("header", 3.0, 24.0, 0.8),
    ("sender", 24.0, 44.0, 0.45),
    ("recipient", 44.0, 82.0, 0.8),
    ("routing", 82.0, 118.0, 0.8),
    ("barcode", 118.0, 160.0, 0.45),
    ("footer", 160.0, 177.0, 0.0),
)
QR_MODULE_MM = 1.15          # 21 modules (version 1, level M) -> 24.2 mm, + 2 modules quiet zone each side
BAR_HEIGHT_MM = 30.0
BAR_QUIET_MODULES = 10


def design(paper: str = "100x180") -> dict:
    """Machine-readable description of the design at one paper size (mm, label frame: x right, y down)."""
    W, H = PAPERS[paper]
    sx, sy = W / 100.0, H / 180.0
    s = min(sx, sy)
    z = {}
    for name, y0, y1, rule in ZONES:
        z[name] = {"x": [0.0, W], "y": [round(y0 * sy, 2), round(y1 * sy, 2)], "rule_below_mm": round(rule * s, 2)}
    qr = _qr_box(W, H)
    bc = _barcode_box(W, H, "KV0000000000")
    return {
        "paper": paper, "size_mm": [W, H],
        "paper_colour_srgb": list(PAPER_SRGB), "ink_colour_srgb": list(INK_SRGB),
        "print": "black only (thermal); no colour",
        "zones": z,
        "elements": {
            "emblem": {"box_mm": _r([4 * sx, 6 * sy, 19 * sx, 21 * sy]),
                       "shape": "black disc with a white diamond and a black centre dot"},
            "wordmark": {"text": CARRIER["wordmark"], "cap_height_mm": round(7.2 * s, 2),
                         "tagline": CARRIER["tagline"]},
            "service_box": {"box_mm": _r([72 * sx, 5 * sy, 97 * sx, 22 * sy]),
                            "style": "solid black box, white two-character service code"},
            "routing_code": {"cap_height_mm": round(13.0 * s, 2), "box_mm": _r([4 * sx, 85 * sy, 66 * sx, 115 * sy]),
                             "style": "very large bold black characters, e.g. NV27-4C"},
            "qr_code": {"box_mm": _r(qr), "symbol": "QR version 1, error correction M, quiet zone 2 modules",
                        "module_mm": round(QR_MODULE_MM * s, 3), "content": "the tracking number"},
            "linear_barcode": {"box_mm": _r(bc["box"]), "symbology": "Code 128 (set B for KV, set C for the digits)",
                               "module_mm": round(bc["module"], 3), "bar_height_mm": round(BAR_HEIGHT_MM * sy, 2),
                               "quiet_zone_modules": BAR_QUIET_MODULES, "content": "the tracking number",
                               "human_readable_below": True},
            "sender_lines": 3, "recipient_lines": 4,
        },
        "rules": "black horizontal rules between the zones",
    }


def _r(v):
    return [round(float(x), 2) for x in v]


def _qr_box(W, H):
    sx, sy = W / 100.0, H / 180.0
    s = min(sx, sy)
    side = (21 + 4) * QR_MODULE_MM * s
    x1 = 97.0 * sx
    y0 = 85.0 * sy + (30.0 * sy - side) / 2 + 0.5 * s
    return [x1 - side, y0, x1, y0 + side]


def _barcode_box(W, H, text):
    sy = H / 180.0
    n = len(code128.modules(text, quiet=BAR_QUIET_MODULES))
    avail = W - 2 * 2.0
    module = min(0.6 * W / 100.0, avail / n)
    width = n * module
    x0 = (W - width) / 2
    y0 = 121.0 * sy
    return {"box": [x0, y0, x0 + width, y0 + BAR_HEIGHT_MM * sy], "module": module, "n_modules": n}


def _font(kind, px):
    return ImageFont.truetype(os.path.join(_FONT_DIR, FONTS[kind]), max(1, int(round(px))))


def _cov_1d(edges_px, n):
    """Coverage of [a, b) intervals over pixels 0..n-1 (exact area anti-aliasing)."""
    out = np.zeros(n)
    xs = np.arange(n + 1, dtype=float)
    for a, b in edges_px:
        out += np.clip(np.minimum(xs[1:], b) - np.maximum(xs[:-1], a), 0, None)
    return np.clip(out, 0, 1)


def render(identity: WaybillIdentity, paper: str, px_per_mm: float, ss: int = 2, size_px=None) -> dict:
    """Render the waybill. ``size_px`` (w, h) forces the output size (else paper mm x px_per_mm)."""
    W, H = PAPERS[paper]
    if size_px is None:
        size_px = (int(round(W * px_per_mm)), int(round(H * px_per_mm)))
    wpx, hpx = size_px
    kx, ky = wpx / W, hpx / H                     # px per mm (output)
    Kx, Ky = kx * ss, ky * ss                     # px per mm (drawing)
    sx, sy = W / 100.0, H / 180.0
    s = min(sx, sy)
    img = Image.new("L", (wpx * ss, hpx * ss), 0)  # ink coverage, 0 = paper
    d = ImageDraw.Draw(img)
    K = 0.5 * (Kx + Ky)

    def X(x):
        return x * sx * Kx

    def Y(y):
        return y * sy * Ky

    def rect(x0, y0, x1, y1, fill=255):        # design mm (canonical)
        d.rectangle([X(x0), Y(y0), X(x1) - 1, Y(y1) - 1], fill=fill)

    def text(x, y_top, cap_mm, string, kind="regular", fill=255, anchor="la", max_w_mm=None):
        """``y_top`` = top of the capitals (canonical mm); ``cap_mm`` canonical cap height."""
        em = cap_mm * s / CAP * K
        f = _font(kind, em)
        if max_w_mm is not None:
            w = d.textlength(string, font=f)
            lim = max_w_mm * sx * Kx
            if w > lim:
                em *= lim / w
                f = _font(kind, em)
        base = Y(y_top) + em * CAP                 # baseline = top of the capitals + cap height
        ax = {"la": "ls", "ra": "rs", "ma": "ms"}[anchor]
        d.text((X(x), base), string, font=f, fill=fill, anchor=ax)

    # ---- header: emblem, wordmark, service box
    ex0, ey0, ex1, ey1 = X(4), Y(6), X(19), Y(21)
    side = min(ex1 - ex0, ey1 - ey0)
    cx, cy = 0.5 * (ex0 + ex1), 0.5 * (ey0 + ey1)
    r = side / 2
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    q = r * 0.62
    d.polygon([(cx, cy - q), (cx + q, cy), (cx, cy + q), (cx - q, cy)], fill=0)
    q2 = r * 0.2
    d.ellipse([cx - q2, cy - q2, cx + q2, cy + q2], fill=255)
    text(22, 7.8, 7.2, CARRIER["wordmark"], "bold", max_w_mm=47)
    text(22.3, 17.6, 2.4, CARRIER["tagline"], "medium", max_w_mm=45)
    rect(72, 5, 97, 22)
    text(84.5, 8.5, 10.0, identity.service, "bold", fill=0, anchor="ma")

    # ---- zone rules
    for name, y0, y1, rule in ZONES:
        if rule > 0:
            t = rule * s / sy
            rect(3, y1 - t / 2, 97, y1 + t / 2)

    # ---- sender
    text(4, 26.6, 2.0, "FROM", "bold")
    for i, line in enumerate(identity.sender):
        text(4, 30.2 + 4.5 * i, 2.6, line, "regular", max_w_mm=92)
    # ---- recipient
    text(4, 46.6, 2.4, "TO", "bold")
    caps = (3.6, 3.6, 4.8, 3.2)
    ys = (50.8, 57.0, 63.6, 72.6)
    kinds = ("medium", "medium", "bold", "medium")
    for line, c, y, k in zip(identity.recipient, caps, ys, kinds):
        text(4, y, c, line, k, max_w_mm=92)
    # ---- routing
    text(4, 85.4, 2.0, "ROUTE", "bold")
    text(3.6, 90.0, 13.0, identity.routing, "bold", max_w_mm=62)
    text(4, 109.6, 2.2, "SORT " + identity.routing.split("-")[0][:2] + " · LANE " + identity.routing.split("-")[1],
         "medium", max_w_mm=60)
    # ---- footer
    text(4, 163.4, 2.6, "WT " + identity.weight, "medium")
    text(30, 163.4, 2.6, "PCS 1/1", "medium")
    text(52, 163.4, 2.6, "SHIP " + identity.date, "medium")
    text(4, 169.6, 2.6, "REF " + identity.ref, "medium")
    text(96, 169.6, 2.6, CARRIER["wordmark"] + " " + identity.service, "bold", anchor="ra")
    # ---- human-readable tracking number
    text(50, 153.6, 3.4, identity.tracking_print, "medium", anchor="ma")

    ink = np.asarray(img, dtype=np.float64) / 255.0
    # ---- linear barcode (exact coverage, drawn at the drawing resolution)
    bc = _barcode_box(W, H, identity.tracking)
    mods = code128.modules(identity.tracking, quiet=BAR_QUIET_MODULES)
    x0 = bc["box"][0]
    edges, k = [], 0
    while k < len(mods):
        if mods[k]:
            j = k
            while j < len(mods) and mods[j]:
                j += 1
            edges.append(((x0 + k * bc["module"]) * Kx, (x0 + j * bc["module"]) * Kx))
            k = j
        else:
            k += 1
    colcov = _cov_1d(edges, ink.shape[1])
    rowcov = _cov_1d([(bc["box"][1] * Ky, bc["box"][3] * Ky)], ink.shape[0])
    ink = np.maximum(ink, rowcov[:, None] * colcov[None, :])
    # ---- 2-D code
    qm = qr_modules(identity.tracking)
    qx0, qy0, qx1, qy1 = _qr_box(W, H)
    mod = (qx1 - qx0) / (qm.shape[0] + 4)
    n = qm.shape[0]
    cx_edges = [((qx0 + (2 + i) * mod) * Kx, (qx0 + (3 + i) * mod) * Kx) for i in range(n)]
    cy_edges = [((qy0 + (2 + i) * mod) * Ky, (qy0 + (3 + i) * mod) * Ky) for i in range(n)]
    cols = np.stack([_cov_1d([e], ink.shape[1]) for e in cx_edges])        # n x Wd
    rows = np.stack([_cov_1d([e], ink.shape[0]) for e in cy_edges])        # n x Hd
    qcov = np.clip(rows.T @ qm.astype(float) @ cols, 0, 1)                  # Hd x Wd
    ink = np.maximum(ink, qcov)

    # 2x -> 1x by area averaging
    ink = ink.reshape(hpx, ss, wpx, ss).mean(axis=(1, 3))
    paper_c = np.asarray(PAPER_SRGB)
    ink_c = np.asarray(INK_SRGB)
    lin_p, lin_i = _lin(paper_c), _lin(ink_c)
    lin = lin_p[None, None, :] * (1 - ink[..., None]) + lin_i[None, None, :] * ink[..., None]
    rgb = np.clip(np.round(_srgb(lin) * 255.0), 0, 255).astype(np.uint8)
    return {"rgb": rgb, "ink": ink, "px_per_mm": (kx, ky), "barcode": bc, "qr_box_mm": [qx0, qy0, qx1, qy1]}


def qr_modules(text: str) -> np.ndarray:
    """QR modules (1 = dark) without the quiet zone: OpenCV's encoder, level M."""
    import cv2
    p = cv2.QRCodeEncoder_Params()
    p.correction_level = cv2.QRCODE_ENCODER_CORRECT_LEVEL_M
    q = cv2.QRCodeEncoder.create(p).encode(text)
    m = (q < 128).astype(np.uint8)
    ys, xs = np.nonzero(m)
    return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _srgb(x):
    x = np.clip(np.asarray(x, float), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
