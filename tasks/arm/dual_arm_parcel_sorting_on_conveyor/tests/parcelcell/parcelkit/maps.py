"""Texture atlas of one parcel: packing, and the carton / mailer / waybill maps.

One square atlas per parcel holds every face island and the waybill's own island (drawn at ``WAYBILL_DENSITY`` times
the board's texel density). Maps: base colour (sRGB), normal (OpenGL, +v = up), ORM (R occlusion, G roughness,
B metallic). Islands carry an 8 px gutter at 2048 px, filled by evaluating the same procedural surface beyond the
island's edge, so smaller mip levels and the downsampled agent textures do not bleed.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import marks as mk
from . import noise as nz
from .geometry import face_axes, face_half, waybill_points

FULL = 2048
GUTTER = 8                      # px at FULL
ALIGN = 8                       # island corners on multiples of 8 px at FULL (exact at 1024 / 512 / 256)
WAYBILL_DENSITY = 2.4           # waybill texel density / board texel density

BOARDS = {   # sRGB base, fibre contrast
    "kraft_light": ((0.745, 0.585, 0.405), 1.0),
    "kraft_mid": ((0.665, 0.495, 0.325), 1.0),
    "kraft_dark": ((0.565, 0.415, 0.275), 1.1),
    "kraft_red": ((0.685, 0.470, 0.315), 1.0),
    "white": ((0.770, 0.760, 0.730), 0.6),   # greyer than the waybill paper: luminance 0.54, 0.79 of a printed sheet's mean
}
FILMS = {     # sRGB base, roughness
    "grey": ((0.575, 0.585, 0.600), 0.42),
    "white": ((0.780, 0.785, 0.785), 0.60),  # milky, greyer: luminance 0.58, 0.85 of a printed sheet's mean
    "black": ((0.065, 0.066, 0.072), 0.27),
}
TAPES = {
    "clear": {"tint": (0.93, 0.91, 0.86), "rough": 0.22, "opacity": 0.0},
    "brown": {"srgb": (0.560, 0.370, 0.185), "rough": 0.30, "opacity": 0.93},
    "white": {"srgb": (0.880, 0.878, 0.860), "rough": 0.45, "opacity": 0.96},    # white poly tape
    "kraft": {"srgb": (0.640, 0.480, 0.315), "rough": 0.82, "opacity": 1.0},     # gummed kraft paper tape (matt)
}
# the kit's mix (the shares a hub's inbound sees: a third of the board white, every tape colour, stickers often)
BOARD_SHARES = (0.18, 0.20, 0.14, 0.14, 0.34)          # kraft_light, kraft_mid, kraft_dark, kraft_red, white
TAPE_SHARES = (("clear", 0.40), ("brown", 0.25), ("white", 0.20), ("kraft", 0.15))
TAPE_WIDTHS_M = ((0.038, 0.20), (0.048, 0.35), (0.050, 0.25), (0.072, 0.20))
TAPE_PATTERNS = (("I", 0.40), ("H", 0.35), ("I2", 0.25))
STICKER_CHANCES = (("fragile", 0.30), ("lot", 0.40), ("arrow", 0.20), ("blank", 0.40), ("blank", 0.15))
GLOSSY_FILM_SHARE = 0.35                                # mailers in a glossy film
GLOSSY_ROUGH_FACTOR = 0.45


# ======================================================================================================== packing
def _skyline(items, N, gutter, align):
    """Bottom-left skyline packing of (name, w, h) px items (gutter around each). None if they do not fit."""
    sky = [[0, 0, N]]                              # segments: x, y, width
    out = {}
    for name, pw, ph in items:
        W, H = pw + 2 * gutter, ph + 2 * gutter
        best = None
        for i in range(len(sky)):
            x = sky[i][0]
            if x + W > N:
                break
            y, j, rem = 0, i, W
            while rem > 0 and j < len(sky):
                y = max(y, sky[j][1])
                rem -= sky[j][2]
                j += 1
            if rem > 0:
                continue
            y = int(math.ceil(y / align)) * align
            if y + H > N:
                continue
            if best is None or (y + H, x) < (best[1] + H, best[0]):
                best = (x, y)
        if best is None:
            return None
        x, y = best
        out[name] = (x + gutter, y + gutter, pw, ph)
        # update the skyline over [x, x + W)
        new, x1 = [], x + W
        for sx, sy, sw in sky:
            ex = sx + sw
            if ex <= x or sx >= x1:
                new.append([sx, sy, sw])
                continue
            if sx < x:
                new.append([sx, sy, x - sx])
            if ex > x1:
                new.append([x1, sy, ex - x1])
        new.append([x, y + H, W])
        new.sort()
        merged = []
        for seg in new:
            if merged and merged[-1][1] == seg[1] and merged[-1][0] + merged[-1][2] == seg[0]:
                merged[-1][2] += seg[2]
            else:
                merged.append(seg)
        sky = merged
    return out


def pack(rects, N=FULL, gutter=GUTTER, align=ALIGN):
    """rects: list of (name, w_m, h_m, density multiplier). Largest uniform density that fits (skyline packing,
    several orderings). Returns {name: (x0, y0, w, h)} in px at N, and the density (px per m)."""
    def try_pack(dens):
        items = []
        for name, w, h, k in rects:
            pw = int(math.ceil(w * dens * k / align)) * align
            ph = int(math.ceil(h * dens * k / align)) * align
            items.append((name, pw, ph))
        for key in (lambda t: -t[2], lambda t: -t[1] * t[2], lambda t: -t[1], lambda t: -max(t[1], t[2])):
            r = _skyline(sorted(items, key=key), N, gutter, align)
            if r is not None:
                return r
        return None
    lo, hi = 50.0, 40000.0
    best = None
    for _ in range(36):
        mid = math.sqrt(lo * hi)
        r = try_pack(mid)
        if r is None:
            hi = mid
        else:
            lo, best = mid, r
    return best, lo


class Island:
    def __init__(self, name, rect, face=None, ha=None, hb=None):
        self.name, self.rect, self.face, self.ha, self.hb = name, rect, face, ha, hb

    def coords(self, g=GUTTER):
        """(a, b) face-plane coordinates of the island's texels incl. a gutter of g px; and atlas slices."""
        x0, y0, w, h = self.rect
        cols = np.arange(x0 - g, x0 + w + g) + 0.5
        rows = np.arange(y0 - g, y0 + h + g) + 0.5
        a = -self.ha + (cols - x0) / w * 2 * self.ha
        b = self.hb - (rows - y0) / h * 2 * self.hb
        A, B = np.meshgrid(a, b)
        return A, B, (slice(y0 - g, y0 + h + g), slice(x0 - g, x0 + w + g))

    def texel(self):
        x0, y0, w, h = self.rect
        return 2 * self.ha / w, 2 * self.hb / h

    def uv(self, a, b, N=FULL):
        x0, y0, w, h = self.rect
        col = x0 + (np.asarray(a) + self.ha) / (2 * self.ha) * w
        row = y0 + (self.hb - np.asarray(b)) / (2 * self.hb) * h
        return np.stack([col / N, 1.0 - row / N], -1)


class Buffers:
    def __init__(self, N=FULL, fill_srgb=(0.5, 0.45, 0.4)):
        self.N = N
        self.base = np.empty((N, N, 3), np.float32)
        self.base[:] = nz.lin(fill_srgb)
        self.normal = np.zeros((N, N, 3), np.float32)
        self.normal[..., 2] = 1
        self.rough = np.full((N, N), 0.8, np.float32)
        self.ao = np.ones((N, N), np.float32)
        self.metal = np.zeros((N, N), np.float32)
        self.cc = np.zeros((N, N), np.float32)        # clearcoat mask (the tape's glossy layer)

    def put(self, sl, base, normal, rough, ao, cc=None):
        self.base[sl] = base
        self.normal[sl] = normal
        self.rough[sl] = rough
        self.ao[sl] = ao
        if cc is not None:
            self.cc[sl] = cc


def _remap(img, X, Y, border=0.0):
    """Sample a 2-D float image at pixel coordinates (X, Y) (bilinear, outside -> border)."""
    return cv2.remap(img.astype(np.float32), X.astype(np.float32), Y.astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _mark_local(A, B, ca, cb, theta, size_mm, ppm, img_shape):
    """Texel (a, b) -> pixel coordinates in a mark image centred at (ca, cb) and turned by theta."""
    c, s = math.cos(theta), math.sin(theta)
    da, db = A - ca, B - cb
    xl = (c * da + s * db) * 1000.0          # mm, mark frame (x right, y up)
    yl = (-s * da + c * db) * 1000.0
    h, w = img_shape
    X = (xl + size_mm[0] / 2) / size_mm[0] * w - 0.5
    Y = (size_mm[1] / 2 - yl) / size_mm[1] * h - 0.5
    return X, Y, xl, yl


def _rounded_rect_sd(xl, yl, W, H, r):
    qx, qy = np.abs(xl) - (W / 2 - r), np.abs(yl) - (H / 2 - r)
    out = np.sqrt(np.maximum(qx, 0) ** 2 + np.maximum(qy, 0) ** 2) + np.minimum(np.maximum(qx, qy), 0) - r
    return out                                 # mm, negative inside


# ======================================================================================================== waybill island
def waybill_maps(isl, render_out, rng, on_film=False):
    """Base / height / rough / ao of the waybill island from a rendered sheet (render_out of waybill.render)."""
    x0, y0, w, h = isl.rect
    g = GUTTER
    rgb = render_out["rgb"].astype(np.float32) / 255.0
    ink = render_out["ink"].astype(np.float32)
    assert rgb.shape[:2] == (h, w), (rgb.shape, (h, w))
    pad = ((g, g), (g, g))
    rgb = np.stack([np.pad(rgb[..., c], pad, mode="edge") for c in range(3)], -1)
    ink = np.pad(ink, pad, mode="edge")
    base = nz.lin(rgb)
    H, W = ink.shape
    tex_a, tex_b = isl.texel()
    t = 0.5 * (tex_a + tex_b)
    # paper grain, faint smudges near one edge
    grain = nz.fbm2(rng, H, W, max(2.0, 0.0006 / t), 3) * 1.5e-6 + nz.fbm2(rng, H, W, max(2.0, 0.004 / t), 2) * 4e-6
    hgt = grain - ink * 1.0e-6
    if on_film:
        hgt = hgt + nz.fbm2(rng, H, W, max(3.0, 0.012 / t), 3) * 3.5e-5
    smudge = np.clip(nz.fbm2(rng, H, W, max(3.0, 0.02 / t), 3) * 1.6 - 0.9, 0, 1) * float(rng.uniform(0, 0.12))
    base = base * (1 - smudge[..., None] * np.array([0.35, 0.36, 0.40], np.float32))
    rough = 0.50 + 0.10 * ink + 0.05 * smudge
    ao = np.ones_like(ink)
    nrm = nz.height_to_normal(hgt, t)
    return base, nrm, rough.astype(np.float32), ao


# ======================================================================================================== carton
def carton_params(rng, board=None, tape=None, pattern=None):
    board = board or str(rng.choice(list(BOARDS), p=list(BOARD_SHARES)))
    tape = tape or str(rng.choice([k for k, _ in TAPE_SHARES], p=[v for _, v in TAPE_SHARES]))
    pattern = pattern or str(rng.choice([k for k, _ in TAPE_PATTERNS], p=[v for _, v in TAPE_PATTERNS]))
    width = float(rng.choice([k for k, _ in TAPE_WIDTHS_M], p=[v for _, v in TAPE_WIDTHS_M]))
    return {"board": board, "tape": tape, "tape_pattern": pattern, "tape_width": width,
            "seam_gap": float(rng.uniform(0.0005, 0.0035)), "seam_offset": float(rng.uniform(-0.002, 0.002)),
            "wear": float(rng.uniform(0.45, 1.0)), "n_dents": int(rng.poisson(2.6)),
            "joint_face": int(rng.choice([2, 3])), "joint_side": int(rng.choice([-1, 1]))}


def _tape_sd(P, f, h, p):
    """Signed distance (m, positive inside) of the face-plane point P to the union of tape strips."""
    hx, hy, hz = h
    w = p["tape_width"] / 2
    ys = p["seam_offset"]
    L_end = p["_L_end"]
    sd = np.full(P.shape[:-1], -1.0)
    X, Y, Z = P[..., 0], P[..., 1], P[..., 2]
    jag = 0.0008 * np.abs(((Y * 1000 / 3.0) % 1.0) - 0.5) * 2      # serrated cut, 3 mm pitch
    for top in (1, -1):                       # +z then -z
        if top == -1 and p["tape_pattern"] == "I" and p.get("_bottom_untaped"):
            continue
        band = w - np.abs(Y - ys)
        if f == (0 if top == 1 else 1):
            sd = np.maximum(sd, band)
        if f in (2, 3):
            ends = (Z * top) - (hz - L_end[0 if top == 1 else 1]) - jag
            sd = np.maximum(sd, np.minimum(band, ends))
        if p["tape_pattern"] == "H" and top == 1:
            for sx in (1, -1):
                edge = w - (hx - X * sx)                           # strip centred on the top edge at x = +-hx
                if f == 0:
                    sd = np.maximum(sd, edge)
                if f == (2 if sx == 1 else 3):
                    sd = np.maximum(sd, w - (hz - Z))
                if f in (4, 5):                                    # its ends run 35 mm down the long panels
                    sd = np.maximum(sd, np.minimum(edge, Z - (hz - 0.035) - jag))
        if p["tape_pattern"] == "I2" and top == 1:
            # a second strip across the first (a cross on the top)
            band2 = w - np.abs(X)
            if f == 0:
                sd = np.maximum(sd, band2)
            if f in (4, 5):
                sd = np.maximum(sd, np.minimum(band2, Z - (hz - 0.05) - jag))
    return sd


def carton_maps(shape, isls, wb_isl, wb_render, pl, p, rng, marks_out):
    """Fill the atlas buffers of a carton. ``marks_out`` collects the placed marks (record)."""
    buf = Buffers(fill_srgb=BOARDS[p["board"]][0])
    h = shape.h
    r = shape.r
    board_srgb, fib = BOARDS[p["board"]]
    board_lin = nz.lin(board_srgb)
    kraft_lin = nz.lin(BOARDS["kraft_mid"][0])
    seed = int(rng.integers(0, 2 ** 20))
    p["_L_end"] = (float(rng.uniform(0.05, 0.075)), float(rng.uniform(0.05, 0.075)))
    p["_bottom_untaped"] = False
    tp = TAPES[p["tape"]]
    # dents (3-D, on the box surface), scuff patches, a crushed corner
    dents = []
    for _ in range(p["n_dents"]):
        f = int(rng.integers(6))
        ha, hb, hn = face_half(h, f)
        eu, ev, n = face_axes(f)
        c = n * hn + eu * rng.uniform(-0.8, 0.8) * ha + ev * rng.uniform(-0.8, 0.8) * hb
        dents.append((c, float(rng.uniform(0.012, 0.030)), float(rng.uniform(0.0006, 0.0020))))
    creases = []
    for _ in range(int(rng.poisson(1.0))):
        f = int(rng.integers(6))
        ha, hb, _ = face_half(h, f)
        creases.append((f, rng.uniform(-0.7, 0.7) * ha, rng.uniform(-0.7, 0.7) * hb, rng.uniform(0, np.pi),
                        float(rng.uniform(0.03, 0.09)), float(rng.uniform(0.0003, 0.0008))))
    corner = None
    if rng.uniform() < 0.35:
        corner = np.sign(rng.uniform(-1, 1, 3)) * h
    # marks: printed pictograms on a side panel, maker's stamp, stickers (never over the waybill)
    wb_face = pl["face"] if pl else -1
    placed = []

    def free_spot(f, size_mm, tries=40, theta=None):
        ha, hb, _ = face_half(h, f)
        wm, hm = size_mm[0] / 1000, size_mm[1] / 1000
        for _ in range(tries):
            th = (float(rng.choice([0, 0, math.pi / 2, math.pi])) if theta is None else theta) + \
                float(rng.uniform(-0.04, 0.04))
            ex, ey = abs(math.cos(th)) * wm + abs(math.sin(th)) * hm, abs(math.sin(th)) * wm + abs(math.cos(th)) * hm
            if ex / 2 > ha - 0.008 or ey / 2 > hb - 0.008:
                continue
            ca = float(rng.uniform(-(ha - 0.008 - ex / 2), ha - 0.008 - ex / 2))
            cb = float(rng.uniform(-(hb - 0.008 - ey / 2), hb - 0.008 - ey / 2))
            rad = 0.5 * math.hypot(ex, ey)
            ok = True
            for (f2, a2, b2, r2) in placed:
                if f2 == f and math.hypot(ca - a2, cb - b2) < rad + r2 + 0.006:
                    ok = False
            if ok:
                placed.append((f, ca, cb, rad))
                return ca, cb, th
        return None
    if pl:
        W, H = pl["size_m"]
        placed.append((wb_face, pl["offset_ab_m"][0], pl["offset_ab_m"][1], 0.5 * math.hypot(W, H) + 0.004))
    ppm = 12.0
    mark_list = []
    if rng.uniform() < 0.75:
        which = tuple(rng.choice(["up", "fragile", "dry"], size=int(rng.integers(2, 4)), replace=False))
        if "up" not in which:
            which = ("up",) + which[:2]
        m = mk.pictograms(ppm, which)
        for f in [int(x) for x in rng.permutation([4, 5, 2, 3])]:
            spot = free_spot(f, m["size_mm"], theta=0.0)          # upright on a side panel
            if spot:
                mark_list.append((f, spot[0], spot[1], spot[2], m, "ink"))
                break
    if rng.uniform() < 0.6:
        m = mk.maker_stamp(ppm, rng)
        for f in [1, 2, 3, 4, 5]:
            spot = free_spot(f, m["size_mm"])
            if spot:
                mark_list.append((f, spot[0], spot[1], spot[2], m, "ink"))
                break
    for kind, prob in STICKER_CHANCES:
        if rng.uniform() < prob:
            m = {"fragile": lambda: mk.fragile_sticker(ppm), "lot": lambda: mk.lot_sticker(ppm, rng),
                 "arrow": lambda: mk.arrow_sticker(ppm), "blank": lambda: mk.blank_sticker(ppm, rng)}[kind]()
            if kind == "blank":          # a paper label goes where a waybill would: the larger faces more often
                area = np.array([4.0 * face_half(h, f)[0] * face_half(h, f)[1] for f in range(6)])
                faces = [int(x) for x in rng.choice(6, size=6, replace=False, p=area / area.sum())]
            else:
                faces = [int(x) for x in rng.permutation(6)]
            for f in faces:
                spot = free_spot(f, m["size_mm"])
                if spot:
                    mark_list.append((f, spot[0], spot[1], spot[2], m, "sticker"))
                    break
    for f, ca, cb, th, m, how in mark_list:
        marks_out.append({"kind": m["kind"], "face": f, "centre_ab_m": [ca, cb], "theta_rad": th,
                          "size_mm": list(m["size_mm"])})

    for isl in isls:
        f = isl.face
        A, B, sl = isl.coords()
        eu, ev, n = face_axes(f)
        ha, hb, hn = face_half(h, f)
        P = n * hn + eu * A[..., None] + ev * B[..., None]
        ta, tb = isl.texel()
        t = 0.5 * (ta + tb)
        Hh, Ww = A.shape
        # ---- board colour
        low = nz.fbm3(P, 0.14, seed, 3)
        mid = nz.fbm2(rng, Hh, Ww, max(2.0, 0.006 / t), 3)
        fine = nz.fbm2(rng, Hh, Ww, max(1.0, 0.0012 / t), 2)
        val = 1 + 0.045 * low + 0.025 * mid * fib + 0.03 * fine * fib
        base = board_lin[None, None, :] * val[..., None]
        base[..., 0] *= 1 + 0.015 * low
        fleck = np.clip((nz.fbm2(rng, Hh, Ww, max(1.0, 0.0008 / t), 1) - 0.55) * 4, 0, 1) * 0.10 * fib
        base *= (1 - fleck)[..., None]
        if p["board"] == "white":
            mottle = np.clip(nz.fbm2(rng, Hh, Ww, max(2.0, 0.01 / t), 3) * 1.5, -1, 1)
            base *= (1 - 0.025 * mottle)[..., None]
        rough = 0.84 + 0.04 * mid + 0.02 * fine
        ao = np.ones_like(A, np.float32)
        hgt = 1.2e-5 * fine + 2.5e-5 * mid
        # ---- flutes: ripple across the panel's horizontal (u) axis, pitch ~8 mm
        pitch = 0.0079
        # visible washboarding (about 0.1 mm), strongest in the middle of a panel
        wash = 0.5 + 0.5 * np.clip(1 - (np.abs(A) / max(ha, 1e-3)) ** 2, 0, 1)
        hgt += 3.2e-5 * wash * np.cos(2 * np.pi * A / pitch + 0.3 * mid)      # (preview: 9e-5 read as brushed foil)
        base *= (1 + 0.008 * np.cos(2 * np.pi * A / pitch))[..., None]
        # ---- dents and creases
        for c, sig, dep in dents:
            d2 = np.sum((P - c) ** 2, axis=-1)
            hgt -= dep * np.exp(-0.5 * d2 / sig ** 2)
        for (cf, ca, cb, ang, ln, dep) in creases:
            if cf != f:
                continue
            ux, uy = math.cos(ang), math.sin(ang)
            s_ = (A - ca) * ux + (B - cb) * uy
            q_ = -(A - ca) * uy + (B - cb) * ux
            along = np.clip(1 - (2 * np.abs(s_) / ln) ** 2, 0, 1)
            hgt -= dep * along * np.exp(-0.5 * (q_ / 0.0012) ** 2)
            base *= (1 - 0.05 * along * np.exp(-0.5 * (q_ / 0.0015) ** 2))[..., None]
        if corner is not None:
            dc = np.linalg.norm(P - corner, axis=-1)
            crush = nz.smoothstep(0.05, 0.0, dc)
            # a soft crush (broad, low relief) instead of a bumpy patch that glinted like foil; matt, darker
            hgt -= crush * 3.5e-4 * (0.6 + 0.4 * nz.fbm2(rng, Hh, Ww, max(2.0, 0.012 / t), 2))
            base *= (1 - 0.12 * crush)[..., None]
            rough = np.clip(rough + 0.08 * crush, 0, 1)
        # ---- top / bottom flap seam, flap edges
        if f in (0, 1):
            Yw = P[..., 1]
            gap = p["seam_gap"] / 2
            dseam = np.abs(Yw - p["seam_offset"])
            in_gap = nz.smoothstep(gap + 0.5 * t, gap - 0.5 * t, dseam)
            # the tape bridges the flap gap: under the tape the gap is a faint 0.08 mm dip seen through the film
            # (a deep groove or a flap step there reads as a ledge at mid-height on a carton lying on its side); the
            # flap edges' light line is kept only where no tape covers them (nowhere on faces 0 / 1 for the kit's tape
            # patterns, which always cover the seam).
            under = np.clip(_tape_sd(P, f, h, p) / t + 0.5, 0, 1)
            base *= (1 - in_gap * (0.72 * (1 - under) + (0.22 if p["tape"] == "clear" else 0.08) * under))[..., None]
            hgt -= in_gap * (0.0018 * (1 - under) + 0.00008 * under)
            ao *= 1 - in_gap * (0.65 * (1 - under) + 0.06 * under)
            edge_line = np.exp(-0.5 * ((dseam - gap - 0.0005) / 0.0004) ** 2) * (1 - under)
            base *= (1 + 0.06 * edge_line)[..., None]
            step = 0.00025 * (Yw > p["seam_offset"]) * (1 - under)      # one flap a little proud of the other
            hgt += step
            ao *= 1 - 0.12 * np.exp(-np.maximum(dseam - gap, 0) / 0.0012) * (1 - under)
        # ---- manufacturer's joint (glued tab) on one end panel
        if f == p["joint_face"]:
            jx = (ha - 0.03) * p["joint_side"]
            dj = (A - jx) * p["joint_side"]
            tab = nz.smoothstep(-0.5 * t, 0.5 * t, dj)
            hgt += 0.0003 * tab
            base *= (1 - 0.03 * tab)[..., None]
            base *= (1 - 0.10 * np.exp(-0.5 * (dj / 0.0005) ** 2))[..., None]
        # ---- printed marks (ink) and stickers
        for (mf, ca, cb, th, m, how) in mark_list:
            if mf != f:
                continue
            img = m["ink"] if how == "ink" else m["print"]
            X, Y, xl, yl = _mark_local(A, B, ca, cb, th, m["size_mm"], ppm, img.shape)
            v = _remap(img, X, Y)
            if how == "ink":
                ink_lin = nz.lin((0.13, 0.105, 0.09))
                cover = v * (0.86 + 0.08 * mid)
                base = base * (1 - cover[..., None]) + ink_lin[None, None, :] * cover[..., None]
                rough = rough * (1 - cover) + 0.80 * cover
            else:
                sdm = _rounded_rect_sd(xl, yl, m["size_mm"][0], m["size_mm"][1], m["corner_mm"])
                inside = np.clip(0.5 - sdm / (t * 1000), 0, 1)
                st_lin = nz.lin(m["sticker_srgb"]) * (1 + 0.01 * fine[..., None])
                pr_lin = nz.lin(m["print_srgb"])
                col = st_lin * (1 - v[..., None]) + pr_lin[None, None, :] * v[..., None]
                base = base * (1 - inside[..., None]) + col * inside[..., None]
                rough = rough * (1 - inside) + 0.40 * inside
                hgt += 0.00008 * inside
                edge = np.exp(-0.5 * (sdm / 0.25) ** 2)
                ao *= 1 - 0.10 * edge * (sdm > 0)
        # ---- tape
        sd = _tape_sd(P, f, h, p)
        tm = np.clip(sd / t + 0.5, 0, 1)
        if tm.max() > 0:
            lift = np.zeros_like(tm)
            lift_noise = nz.fbm2(rng, Hh, Ww, max(2.0, 0.03 / t), 2)
            near_edge = np.exp(-np.maximum(sd, 0) / 0.0015) * (sd > -t)
            lift = near_edge * np.clip(lift_noise * 3 - 0.6, 0, 1)
            bubbles = np.clip(nz.fbm2(rng, Hh, Ww, max(2.0, 0.005 / t), 2) * 2.5 - 1.3, 0, 1)
            if p["tape"] == "clear":
                tint = np.asarray(tp["tint"], np.float32)
                tb_ = base * tint[None, None, :] * (1 + 0.25 * lift + 0.10 * bubbles)[..., None]
            else:
                tcol = nz.lin(tp["srgb"]) * (1 + 0.04 * mid)[..., None]
                tb_ = tcol * tp["opacity"] + base * (1 - tp["opacity"]) * 0.8
                tb_ = tb_ * (1 + 0.15 * lift)[..., None]
            base = base * (1 - tm[..., None]) + tb_ * tm[..., None]
            rough = rough * (1 - tm) + (tp["rough"] + 0.08 * lift + 0.05 * bubbles) * tm
            # the tape as its own glossy film: a raised edge (0.15 mm), lifted runs near the free edges,
            # wrinkles across the strip, a hairline shadow under the edge, and a clearcoat mask (layered gloss)
            wrk = np.clip(nz.fbm2(rng, Hh, Ww, max(2.0, 0.008 / t), 2) * 2.0 - 0.6, 0, 1)
            hgt += tm * (1.5e-4 + 3.5e-4 * lift + 4e-5 * bubbles + 3e-5 * wrk)
            ao *= 1 - 0.18 * np.exp(-0.5 * ((sd + 0.0003) / 0.0004) ** 2) * (sd < 0.0003)
            cc_ = tm * (0.85 + 0.15 * (1 - lift)) * (0.1 if p["tape"] == "kraft" else 1.0)   # paper tape: matt
        else:
            cc_ = np.zeros_like(tm)
        # ---- edge wear on the bevels and scuffs
        dedge = np.minimum(ha - np.abs(A), hb - np.abs(B))
        on_bevel = nz.smoothstep(0.9 * r, 0.1 * r, dedge)
        wn = nz.fbm3(P, 0.025, seed + 3, 3)
        patch = np.clip(wn * 2.4 - 0.5, 0, 1)                      # wear only in patches along the edges
        wear = np.clip(on_bevel * patch * p["wear"], 0, 1) * (1 - 0.7 * tm)
        wear *= np.clip(0.6 + 0.8 * nz.fbm2(rng, Hh, Ww, max(1.0, 0.0015 / t), 2), 0, 1)
        lighter = board_lin * 1.15 if p["board"] != "white" else kraft_lin * 1.25
        k_w = 0.45 if p["board"] != "white" else 0.35
        base = base * (1 - k_w * wear[..., None]) + lighter[None, None, :] * k_w * wear[..., None]
        rough = np.clip(rough + 0.06 * wear, 0, 1)
        hgt += 4e-5 * wear * nz.fbm2(rng, Hh, Ww, max(1.0, 0.001 / t), 1)
        sc = np.clip(nz.fbm3(P, 0.05, seed + 7, 2) * 2.2 - 1.1, 0, 1) * p["wear"] * (1 - tm)
        base *= (1 + 0.15 * sc)[..., None]                       # scuffs (lighter, rougher fibre)
        rough = np.clip(rough + 0.06 * sc, 0, 1)
        smudge = np.clip(nz.fbm3(P, 0.035, seed + 9, 3) * 2.6 - 1.5, 0, 1) * (1 - tm)
        base *= (1 - 0.13 * smudge)[..., None]                    # handling dirt
        if f == 1:                                     # belt grime on the bottom
            grime = np.clip(0.4 + 0.6 * nz.fbm2(rng, Hh, Ww, max(2.0, 0.03 / t), 3), 0, 1)
            base *= (1 - 0.08 * grime)[..., None]
        # ---- contact shadow around the waybill sheet
        if pl and f == wb_face:
            th = pl["theta_rad"]
            c_, s_ = math.cos(th), math.sin(th)
            oa, ob = pl["offset_ab_m"]
            xl = c_ * (A - oa) + s_ * (B - ob)
            yl = -s_ * (A - oa) + c_ * (B - ob)
            W, H = pl["size_m"]
            sdl = np.maximum(np.abs(xl) - W / 2, np.abs(yl) - H / 2)
            ao *= 1 - 0.28 * np.exp(-np.maximum(sdl, 0) / 0.0009) * (sdl > -0.002)
        nrm = nz.height_to_normal(hgt, t)
        buf.put(sl, base, nrm, np.clip(rough, 0.75 * (1 - tm) + 0.15 * tm, 1.0), ao, cc=cc_)   # board >= 0.75
    if wb_isl is not None:
        b_, n_, r_, a_ = waybill_maps(wb_isl, wb_render, rng)
        x0, y0, w, hh = wb_isl.rect
        g = GUTTER
        buf.put((slice(y0 - g, y0 + hh + g), slice(x0 - g, x0 + w + g)), b_, n_, r_, a_)
    return buf


# ======================================================================================================== mailer
def mailer_params(rng, film=None):
    film = film or str(rng.choice(list(FILMS), p=[0.45, 0.30, 0.25]))
    out = {"film": film, "flap_face": int(rng.choice([0, 1])), "flap_end": int(rng.choice([-1, 1])),
           "flap_len": float(rng.uniform(0.035, 0.05)), "resin_mark": bool(rng.uniform() < 0.7)}
    out["glossy"] = bool(rng.uniform() < GLOSSY_FILM_SHARE)
    return out


def mailer_maps(shape, isls, wb_isl, wb_render, pl, p, rng, marks_out):
    buf = Buffers(fill_srgb=FILMS[p["film"]][0])
    film_srgb, film_rough = FILMS[p["film"]]
    film_lin = nz.lin(film_srgb)
    hx, hy, hz = shape.h
    seed = int(rng.integers(0, 2 ** 20))
    ppm = 12.0
    resin = mk.resin_mark(ppm) if p["resin_mark"] else None
    resin_pos = None
    if resin is not None:
        f = 1 - p["flap_face"] if rng.uniform() < 0.5 else p["flap_face"]
        ca = float(np.sign(rng.uniform(-1, 1)) * (hx - shape.fin - 0.03))
        cb = float(np.sign(rng.uniform(-1, 1)) * (hy - shape.fin - 0.03))
        if pl and pl["face"] == f:
            oa, ob = pl["offset_ab_m"]
            if math.hypot(ca - oa, cb - ob) < 0.5 * math.hypot(*pl["size_m"]) + 0.015:
                ca, cb = -ca, -cb
        resin_pos = (f, ca, cb)
        marks_out.append({"kind": "resin_mark", "face": f, "centre_ab_m": [ca, cb], "theta_rad": 0.0,
                          "size_mm": list(resin["size_mm"])})
    for isl in isls:
        f = isl.face
        side = 0 if f == 0 else 1
        A, B, sl = isl.coords()
        eu, ev, n = face_axes(f)
        P2 = eu * A[..., None] + ev * B[..., None]
        X, Y = P2[..., 0], P2[..., 1]
        ta, tb = isl.texel()
        t = 0.5 * (ta + tb)
        Hh, Ww = A.shape
        P = np.stack([X, Y, np.zeros_like(X)], -1)
        low = nz.fbm3(P, 0.12, seed + side, 2)
        fine = nz.fbm2(rng, Hh, Ww, max(1.0, 0.002 / t), 2)
        base = film_lin[None, None, :] * (1 + 0.025 * low + 0.008 * fine)[..., None]
        rough = film_rough + 0.03 * low + 0.02 * fine
        ao = np.ones_like(A, np.float32)
        # fine creases (normal map) + crinkle
        hgt = 2.0e-5 * nz.fbm2(rng, Hh, Ww, max(2.0, 0.003 / t), 3)
        for _ in range(int(rng.integers(16, 26))):              # more, crisper creases (limp film)
            # creases radiate from the seam near the perimeter, run freely in the middle
            c = rng.uniform(-1, 1, 2) * [hx, hy]
            dxe, dye = hx - abs(c[0]), hy - abs(c[1])
            if min(dxe, dye) < 0.35 * min(hx, hy):
                ang = (0.0 if dxe < dye else np.pi / 2) + rng.normal(0, 0.35)
            else:
                ang = rng.uniform(0, np.pi)
            ln = rng.uniform(0.03, 0.11)
            dep = rng.uniform(1.5e-4, 4.5e-4) * (1 if rng.uniform() < 0.6 else -1)   # deeper creases
            wd = rng.uniform(0.0008, 0.0028)
            ux, uy = math.cos(ang), math.sin(ang)
            s_ = (X - c[0]) * ux + (Y - c[1]) * uy
            q_ = -(X - c[0]) * uy + (Y - c[1]) * ux
            along = np.clip(1 - (2 * np.abs(s_) / ln) ** 2, 0, 1) ** 1.5
            prof = np.exp(-0.5 * (q_ / wd) ** 2)
            hgt += dep * along * prof
            rough = rough + 0.03 * along * prof
        for sx_ in (-1, 1):                          # corner ears: a diagonal fold in from some corners
            for sy_ in (-1, 1):
                if rng.uniform() < 0.5:
                    continue
                ang = math.atan2(sy_ * hy, sx_ * hx) + rng.normal(0, 0.15)
                ux, uy = math.cos(ang), math.sin(ang)
                cx_, cy_ = sx_ * hx, sy_ * hy
                s_ = -((X - cx_) * ux + (Y - cy_) * uy)
                q_ = -(X - cx_) * uy + (Y - cy_) * ux
                along = np.clip(s_ / 0.05, 0, 1) * np.clip(1 - s_ / 0.07, 0, 1)
                hgt += 4.0e-4 * along * np.exp(-0.5 * (q_ / 0.004) ** 2)
        # geometric wrinkles of the mesh deepen the colour a touch in the folds
        wr = shape._wrinkle(X, Y, side)
        base *= (1 - 0.08 * wr)[..., None]
        # perimeter fin: heat-seal ribs, a lighter seal band
        d = shape._dist(X, Y)
        in_fin = nz.smoothstep(shape.fin + 0.5 * t, shape.fin - 0.5 * t, d)
        hgt += in_fin * 6.0e-5 * np.cos(2 * np.pi * d / 0.0011)          # embossed heat-seal ribs, visible
        seal_tint = 1.35 if p["film"] == "black" else 0.94      # the welded seam reads (milky on black film)
        base *= (1 + (seal_tint - 1) * in_fin)[..., None]
        rough = rough + 0.25 * in_fin
        ao *= 1 - 0.20 * np.exp(-0.5 * ((d - shape.fin + 0.0004) / 0.0006) ** 2)
        ao *= 1 - 0.12 * np.exp(-0.5 * ((d - shape.fin) / 0.0015) ** 2)
        # flap on one face at one end: double film, its free edge a step
        if f == p["flap_face"]:
            xe = p["flap_end"] * X
            edge_pos = hx - p["flap_len"]
            flap = nz.smoothstep(edge_pos - 0.5 * t, edge_pos + 0.5 * t, xe)
            hgt += 2.2e-4 * flap
            base *= (1 + (0.04 if p["film"] == "black" else -0.03) * flap)[..., None]
            rough = rough - 0.03 * flap
            ao *= 1 - 0.18 * np.exp(-0.5 * ((xe - edge_pos + 0.0006) / 0.0006) ** 2)
            # adhesive strip seen through the flap (a band just inside the free edge)
            band = nz.smoothstep(edge_pos + 0.004, edge_pos + 0.006, xe) * nz.smoothstep(
                edge_pos + 0.026, edge_pos + 0.024, xe)
            base *= (1 + (0.05 if p["film"] == "black" else -0.04) * band)[..., None]
        if resin_pos is not None and resin_pos[0] == f:
            _, ca, cb = resin_pos
            Xm, Ym, _, _ = _mark_local(A, B, ca, cb, 0.0, resin["size_mm"], ppm, resin["ink"].shape)
            v = _remap(resin["ink"], Xm, Ym) * 0.85
            ink = nz.lin((0.85, 0.85, 0.84)) if p["film"] != "white" else nz.lin((0.45, 0.46, 0.48))
            base = base * (1 - v[..., None]) + ink[None, None, :] * v[..., None]
        if pl and f == pl["face"]:
            th = pl["theta_rad"]
            c_, s_ = math.cos(th), math.sin(th)
            oa, ob = pl["offset_ab_m"]
            xl = c_ * (A - oa) + s_ * (B - ob)
            yl = -s_ * (A - oa) + c_ * (B - ob)
            W, H = pl["size_m"]
            sdl = np.maximum(np.abs(xl) - W / 2, np.abs(yl) - H / 2)
            ao *= 1 - 0.22 * np.exp(-np.maximum(sdl, 0) / 0.0009) * (sdl > -0.002)
        nrm = nz.height_to_normal(hgt, t)
        if p.get("glossy"):                                      # a glossy film: lower roughness, the seal stays matt
            rough = np.where(rough < 0.6, rough * GLOSSY_ROUGH_FACTOR, rough)
        buf.put(sl, base, nrm, np.clip(rough, 0.08 if p.get("glossy") else 0.25, 0.95), ao)   # the matt seal may reach 0.95
    if wb_isl is not None:
        b_, n_, r_, a_ = waybill_maps(wb_isl, wb_render, rng, on_film=True)
        x0, y0, w, hh = wb_isl.rect
        g = GUTTER
        buf.put((slice(y0 - g, y0 + hh + g), slice(x0 - g, x0 + w + g)), b_, n_, r_, a_)
    return buf


# ======================================================================================================== outputs
def finish(buf, size):
    """8-bit maps at ``size`` px (area average from FULL): base sRGB, normal, ORM."""
    f = FULL // size
    base = nz.downsample(buf.base, f)
    nrm = nz.downsample(buf.normal, f)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-6)
    rough = nz.downsample(buf.rough, f)
    ao = nz.downsample(buf.ao, f)
    metal = nz.downsample(buf.metal, f)
    base8 = np.clip(np.round(nz.srgb(base) * 255), 0, 255).astype(np.uint8)
    nrm8 = nz.encode_normal(nrm)
    orm8 = np.clip(np.round(np.stack([ao, rough, metal], -1) * 255), 0, 255).astype(np.uint8)
    out = {"basecolor": base8, "normal": nrm8, "orm": orm8}
    if float(buf.cc.max()) > 0:
        cc = nz.downsample(buf.cc, f)
        out["cc"] = np.clip(np.round(np.repeat(cc[..., None], 3, -1) * 255), 0, 255).astype(np.uint8)
    return out


def waybill_uv(wb_isl, pl, xl, yl, N=FULL):
    """Label-frame coordinates (m) -> atlas uv of the waybill island."""
    W, H = pl["size_m"]
    x0, y0, w, h = wb_isl.rect
    col = x0 + (np.asarray(xl) + W / 2) / W * w
    row = y0 + (H / 2 - np.asarray(yl)) / H * h
    return np.stack([col / N, 1.0 - row / N], -1)


__all__ = ["pack", "Island", "carton_maps", "mailer_maps", "finish", "waybill_uv", "carton_params",
           "mailer_params", "BOARDS", "FILMS", "TAPES", "FULL", "WAYBILL_DENSITY", "waybill_points"]
