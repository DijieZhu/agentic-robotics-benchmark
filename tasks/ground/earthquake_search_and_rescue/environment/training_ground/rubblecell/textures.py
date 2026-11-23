"""Procedural textures of the rubble site (cell internals): concrete, broken concrete, brick, plaster in a few paints,
floor tiles, timber, asphalt, paving, fabric, painted metal, rubble and dust, all made with numpy value noise from a
fixed seed (the same pixels in every run), plus the painter of the site's ground image (the heightfield's texture).

Every texture is an (H, W, 3) uint8 array meant to repeat (`TILE_M` metres per repeat for the uniform-mapped ones).
"""

import math

import numpy as np

__all__ = ["make_textures", "TILE_M", "fbm", "value_noise"]

# metres covered by one repeat of each uniform texture
TILE_M = {"concrete": 1.6, "concrete_dark": 1.6, "rubble": 1.2, "brick": 1.0, "plaster_white": 2.0,
          "plaster_cream": 2.0, "plaster_blue": 2.0, "plaster_green": 2.0, "plaster_pink": 2.0, "tiles": 1.2,
          "wood": 1.0, "wood_dark": 1.0, "asphalt": 2.0, "paving": 1.6, "fabric_red": 0.6, "fabric_blue": 0.6,
          "fabric_grey": 0.6, "fabric_green": 0.6, "metal_white": 1.0, "metal_red": 1.0, "metal_blue": 1.0,
          "metal_grey": 1.0, "rust": 1.0, "dust": 1.5, "rebar": 0.5, "track": 0.4, "glass": 1.0}


def value_noise(shape, cells, rng, periodic=True):
    """Smooth value noise in [0, 1] on an (h, w) grid with `cells` lattice cells across (tileable when periodic)."""
    h, w = shape
    cy, cx = (cells, cells) if np.isscalar(cells) else cells
    lat = rng.random((cy + 1, cx + 1))
    if periodic:
        lat[-1, :] = lat[0, :]
        lat[:, -1] = lat[:, 0]
    ys = np.linspace(0, cy, h, endpoint=not periodic)
    xs = np.linspace(0, cx, w, endpoint=not periodic)
    y0 = np.clip(np.floor(ys).astype(int), 0, cy - 1)
    x0 = np.clip(np.floor(xs).astype(int), 0, cx - 1)
    fy = ys - y0
    fx = xs - x0
    fy = fy * fy * (3 - 2 * fy)
    fx = fx * fx * (3 - 2 * fx)
    a = lat[y0][:, x0]
    b = lat[y0][:, x0 + 1]
    c = lat[y0 + 1][:, x0]
    d = lat[y0 + 1][:, x0 + 1]
    top = a + (b - a) * fx[None, :]
    bot = c + (d - c) * fx[None, :]
    return top + (bot - top) * fy[:, None]


def fbm(shape, base_cells, octaves, rng, gain=0.5, periodic=True):
    out = np.zeros(shape)
    amp, tot = 1.0, 0.0
    cells = base_cells
    for _ in range(octaves):
        out += amp * value_noise(shape, cells, rng, periodic)
        tot += amp
        amp *= gain
        cells = cells * 2 if np.isscalar(cells) else (cells[0] * 2, cells[1] * 2)
    return out / tot


def _rgb(base, field, spread):
    """base colour (0-1 rgb) modulated by a field in [0, 1] with `spread` (+-)."""
    f = (field - 0.5)[..., None] * 2.0 * np.asarray(spread)[None, None, :]
    return np.clip(np.asarray(base)[None, None, :] + f, 0, 1)


def _u8(x):
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def _speckle(shape, rng, density, size_px=1):
    m = rng.random(shape) < density
    if size_px > 1:
        k = size_px
        m = np.maximum.reduce([np.roll(np.roll(m, dy, 0), dx, 1) for dy in range(k) for dx in range(k)])
    return m


def concrete(n, rng, base=(0.58, 0.57, 0.54), dark=False):
    f = fbm((n, n), 4, 6, rng)
    g = fbm((n, n), 16, 3, rng)
    img = _rgb(base, f, (0.07, 0.07, 0.07))
    img *= (0.92 + 0.16 * g)[..., None]
    # aggregate pores and speckles
    pores = _speckle((n, n), rng, 0.004, 2)
    img[pores] *= 0.55
    light = _speckle((n, n), rng, 0.01, 1)
    img[light] = np.clip(img[light] * 1.15, 0, 1)
    # water stains
    st = fbm((n, n), 3, 4, rng)
    img *= (1.0 - 0.18 * np.clip((st - 0.6) * 4, 0, 1))[..., None]
    if dark:
        img *= 0.78
    return _u8(img)


def rubble(n, rng):
    """Broken concrete and masonry fragments with dust: a mosaic of chips."""
    img = np.zeros((n, n, 3))
    f = fbm((n, n), 6, 5, rng)
    # Voronoi-ish chips: nearest of random seeds (tileable through wrapping)
    k = 140
    pts = rng.random((k, 2)) * n
    cols = rng.choice(4, k, p=[0.55, 0.2, 0.15, 0.1])
    palette = np.array([[0.62, 0.61, 0.58], [0.66, 0.42, 0.32], [0.82, 0.80, 0.74], [0.45, 0.44, 0.42]])
    yy, xx = np.mgrid[0:n, 0:n]
    best = np.full((n, n), 1e9)
    second = np.full((n, n), 1e9)
    lab = np.zeros((n, n), int)
    for i, (py, px) in enumerate(pts):
        dy = np.abs(yy - py)
        dx = np.abs(xx - px)
        dy = np.minimum(dy, n - dy)
        dx = np.minimum(dx, n - dx)
        dd = dy * dy + dx * dx
        upd = dd < best
        second = np.where(upd, best, np.minimum(second, dd))
        lab = np.where(upd, i, lab)
        best = np.where(upd, dd, best)
    base = palette[cols[lab]] * (0.85 + 0.3 * rng.random(k)[lab])[..., None]
    edge = np.sqrt(second) - np.sqrt(best)
    shade = np.clip(edge / 6.0, 0, 1)
    img = base * (0.55 + 0.45 * shade)[..., None]
    img *= (0.9 + 0.2 * f)[..., None]
    dust = fbm((n, n), 4, 4, rng)
    img = img * (1 - 0.35 * dust[..., None]) + np.array([0.70, 0.67, 0.62]) * 0.35 * dust[..., None]
    return _u8(img)


def brick(n, rng):
    img = np.zeros((n, n, 3))
    rows = 15                         # 1.0 m / 15 = 6.7 cm courses
    bh = n / rows
    bw = bh * 3.3
    mortar = 0.12
    f = fbm((n, n), 8, 4, rng)
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    r = np.floor(yy / bh).astype(int)
    off = (r % 2) * bw / 2.0
    c = np.floor((xx + off) / bw).astype(int)
    nbx = int(math.ceil(n / bw)) + 2
    tint = rng.random((rows + 1, nbx + 1, 3))
    base = np.array([0.55, 0.27, 0.18])
    col = base[None, None, :] * (0.8 + 0.4 * tint[r % (rows + 1), c % (nbx + 1)])
    col[..., 1] *= 0.95 + 0.1 * tint[r % (rows + 1), c % (nbx + 1), 0]
    fy = (yy / bh) - r
    fx = ((xx + off) / bw) - c
    m = (fy < mortar) | (fx < mortar * bh / bw)
    col[m] = np.array([0.66, 0.64, 0.6])
    col *= (0.85 + 0.3 * f)[..., None]
    # soot / dust
    dust = fbm((n, n), 3, 4, rng)
    col = col * (1 - 0.3 * dust[..., None]) + np.array([0.6, 0.58, 0.55]) * 0.3 * dust[..., None]
    return _u8(col)


def plaster(n, rng, paint):
    f = fbm((n, n), 5, 6, rng)
    img = _rgb(paint, f, (0.04, 0.04, 0.04))
    # cracks: thin dark random walks
    crack = np.zeros((n, n), bool)
    for _ in range(5):
        y, x = rng.integers(0, n, 2)
        ang = rng.uniform(0, 2 * math.pi)
        for _s in range(int(n * 0.8)):
            ang += rng.normal(0, 0.25)
            y = (y + math.sin(ang)) % n
            x = (x + math.cos(ang)) % n
            crack[int(y), int(x)] = True
    img[crack] *= 0.45
    dirt = fbm((n, n), 3, 4, rng)
    img *= (1 - 0.22 * np.clip(dirt - 0.45, 0, 1) * 2)[..., None]
    dust = fbm((n, n), 6, 3, rng)
    img = img * (1 - 0.15 * dust[..., None]) + np.array([0.72, 0.69, 0.64]) * 0.15 * dust[..., None]
    return _u8(img)


def tiles(n, rng, base=(0.78, 0.74, 0.66)):
    k = 4                      # 1.2 m / 4 = 30 cm tiles
    t = n / k
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    ty, tx = np.floor(yy / t).astype(int), np.floor(xx / t).astype(int)
    tint = rng.random((k, k))
    img = np.asarray(base)[None, None, :] * (0.93 + 0.12 * tint[ty % k, tx % k])[..., None]
    f = fbm((n, n), 8, 4, rng)
    img *= (0.95 + 0.1 * f)[..., None]
    fy, fx = yy / t - ty, xx / t - tx
    grout = (fy < 0.025) | (fx < 0.025)
    img[grout] = np.array([0.5, 0.48, 0.45])
    dust = fbm((n, n), 4, 4, rng)
    img = img * (1 - 0.4 * dust[..., None]) + np.array([0.66, 0.63, 0.58]) * 0.4 * dust[..., None]
    return _u8(img)


def wood(n, rng, base=(0.55, 0.38, 0.22)):
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    warp = fbm((n, n), (4, 2), 4, rng)
    rings = np.sin((yy / n * 40.0 + warp * 6.0) * math.pi)
    f = fbm((n, n), (32, 2), 3, rng)
    img = np.asarray(base)[None, None, :] * (0.85 + 0.1 * rings + 0.15 * f)[..., None]
    # plank seams every 1/5
    seam = (xx % (n / 5)) < 1.5
    img[seam] *= 0.5
    return _u8(img)


def asphalt(n, rng):
    f = fbm((n, n), 6, 5, rng)
    img = _rgb((0.24, 0.24, 0.25), f, (0.04, 0.04, 0.04))
    sp = _speckle((n, n), rng, 0.03, 1)
    img[sp] *= 1.6
    sp2 = _speckle((n, n), rng, 0.02, 1)
    img[sp2] *= 0.6
    dust = fbm((n, n), 3, 4, rng)
    img = img * (1 - 0.35 * dust[..., None]) + np.array([0.6, 0.58, 0.54]) * 0.35 * dust[..., None]
    return _u8(img)


def paving(n, rng):
    k = 8
    t = n / k
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    ty = np.floor(yy / t).astype(int)
    tx = np.floor((xx + (ty % 2) * t / 2) / t).astype(int)
    tint = rng.random((k + 1, k + 2))
    img = np.array([0.62, 0.6, 0.56])[None, None, :] * (0.88 + 0.2 * tint[ty % (k + 1), tx % (k + 2)])[..., None]
    fy = yy / t - ty
    fx = (xx + (ty % 2) * t / 2) / t - tx
    m = (fy < 0.04) | (fx < 0.04)
    img[m] *= 0.6
    f = fbm((n, n), 6, 4, rng)
    img *= (0.9 + 0.2 * f)[..., None]
    return _u8(img)


def fabric(n, rng, base):
    yy, xx = np.mgrid[0:n, 0:n]
    weave = ((yy % 4 < 2) ^ (xx % 4 < 2)).astype(float)
    f = fbm((n, n), 4, 4, rng)
    img = np.asarray(base)[None, None, :] * (0.88 + 0.06 * weave + 0.12 * f)[..., None]
    dust = fbm((n, n), 3, 3, rng)
    img = img * (1 - 0.3 * dust[..., None]) + np.array([0.7, 0.67, 0.62]) * 0.3 * dust[..., None]
    return _u8(img)


def metal(n, rng, base):
    f = fbm((n, n), 3, 5, rng)
    img = _rgb(base, f, (0.03, 0.03, 0.03))
    # scratches
    for _ in range(40):
        y, x = rng.integers(0, n, 2)
        L = rng.integers(10, n // 3)
        ang = rng.uniform(0, math.pi)
        for t in range(L):
            yy = int(y + t * math.sin(ang)) % n
            xx = int(x + t * math.cos(ang)) % n
            img[yy, xx] = img[yy, xx] * 0.7 + 0.3 * np.array([0.5, 0.5, 0.5])
    rust = fbm((n, n), 4, 5, rng)
    rm = np.clip((rust - 0.62) * 5, 0, 1)[..., None]
    img = img * (1 - rm) + np.array([0.45, 0.25, 0.12]) * rm
    dust = fbm((n, n), 3, 3, rng)
    img = img * (1 - 0.25 * dust[..., None]) + np.array([0.68, 0.65, 0.6]) * 0.25 * dust[..., None]
    return _u8(img)


def dust(n, rng):
    f = fbm((n, n), 4, 6, rng)
    img = _rgb((0.66, 0.63, 0.58), f, (0.06, 0.06, 0.06))
    sp = _speckle((n, n), rng, 0.02, 2)
    img[sp] *= 0.75
    return _u8(img)


def track(n, rng):
    yy, xx = np.mgrid[0:n, 0:n]
    grouser = ((xx % (n // 8)) < (n // 32)).astype(float)
    f = fbm((n, n), 4, 3, rng)
    img = np.array([0.09, 0.09, 0.09])[None, None, :] * (1.0 + 0.6 * grouser + 0.3 * f)[..., None]
    return _u8(img)


def make_textures(n=256, seed=20261112):
    """Every texture of the site, {name: uint8 (n, n, 3)} (the ground painter takes them too)."""
    rng = np.random.default_rng(seed)
    T = {}
    T["concrete"] = concrete(n, rng)
    T["concrete_dark"] = concrete(n, rng, dark=True)
    T["rubble"] = rubble(n, rng)
    T["brick"] = brick(n, rng)
    for nm, paint in (("plaster_white", (0.86, 0.85, 0.81)), ("plaster_cream", (0.88, 0.81, 0.66)),
                      ("plaster_blue", (0.62, 0.72, 0.80)), ("plaster_green", (0.66, 0.76, 0.64)),
                      ("plaster_pink", (0.86, 0.70, 0.66))):
        T[nm] = plaster(n, rng, paint)
    T["tiles"] = tiles(n, rng)
    T["wood"] = wood(n, rng)
    T["wood_dark"] = wood(n, rng, base=(0.36, 0.24, 0.15))
    T["asphalt"] = asphalt(n, rng)
    T["paving"] = paving(n, rng)
    for nm, c in (("fabric_red", (0.55, 0.16, 0.14)), ("fabric_blue", (0.18, 0.28, 0.52)),
                  ("fabric_grey", (0.45, 0.45, 0.47)), ("fabric_green", (0.25, 0.42, 0.28))):
        T[nm] = fabric(n, rng, c)
    for nm, c in (("metal_white", (0.85, 0.85, 0.83)), ("metal_red", (0.62, 0.12, 0.10)),
                  ("metal_blue", (0.16, 0.30, 0.55)), ("metal_grey", (0.42, 0.44, 0.46))):
        T[nm] = metal(n, rng, c)
    T["rust"] = metal(n, rng, (0.42, 0.26, 0.15))
    T["dust"] = dust(n, rng)
    T["rebar"] = metal(n, rng, (0.32, 0.24, 0.18))
    T["track"] = track(n, rng)
    T["glass"] = _u8(_rgb((0.55, 0.62, 0.66), fbm((n, n), 2, 3, rng), (0.05, 0.05, 0.05)))
    return T
