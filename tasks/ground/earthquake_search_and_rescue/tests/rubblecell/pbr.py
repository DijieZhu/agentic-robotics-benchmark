"""Physically based texture sets of the rubble site's render twin (cell internals): albedo (sRGB), tangent-space normal
map and ORM (ambient occlusion, roughness, metallic) for every surface class, made procedurally with numpy, scipy and
OpenCV from fixed seeds (the same pixels on every machine), tileable, and cached as PNG files.

Nothing here touches the physics model: the sets dress the render twin only (look.py). Every set repeats seamlessly
(spectral noise, wrapped Voronoi cells, cracks drawn with their wrapped copies), so a texture tiled across a wall shows
no seams. `TILE_M` is the size in metres one repeat covers.

Usage: `python -m rubblecell.pbr DIR` writes every set into DIR (the task image does this at build time, see the
Dockerfile); `texture_sets(dir)` loads them (and makes the missing ones).
"""

import math
import os

import numpy as np

__all__ = ["VERSION", "SETS", "TILE_M", "texture_sets", "make_set", "cache_dir"]

VERSION = "pbr-12"
SEED_TAG = "pbr-10"          # the sets' random seeds derive from this tag, not from VERSION: a set no change touched
                             # keeps its pixels from one version to the next

# set name: (resolution in pixels, metres covered by one repeat)
SETS = {
    "concrete": (1024, 1.8), "rubble": (1024, 1.4), "brick": (1024, 0.9), "plaster": (1024, 2.0),
    "tiles": (1024, 1.2), "wood": (1024, 1.2), "asphalt": (1024, 2.4), "paving": (1024, 1.2),
    "fabric": (512, 0.5), "paint_metal": (512, 1.2), "rust": (512, 1.0), "rebar": (256, 0.3), "dust": (512, 1.5),
    "track": (512, 0.36), "skin": (512, 0.4), "hair": (256, 0.3), "robot_paint": (512, 0.6), "robot_dark": (512, 0.6),
    "alu": (512, 0.5), "road_paint": (512, 1.0), "bag": (512, 0.4),
    # the background's buildings: two storeys (3.2 m each) by two window bays (3.2 m) per repeat; a shop storey
    "facade_plaster": (1024, 6.4), "facade_brick": (1024, 6.4), "facade_panel": (1024, 6.4), "shopfront": (1024, 6.4),
    "foliage": (512, 1.2), "roof": (512, 2.0),
}
# Sets that are not made here: photographed surfaces (scans and an art pack's maps: environment/assets, the look pack),
# shipped as files in the cache folder beside the made ones. set name: (resolution, metres covered by one repeat when a
# material names no repeat of its own). A missing file is an error: nothing stands in for a photograph.
SHIPPED = {
    "concrete": (1024, 2.71), "concrete_dark": (1024, 3.5), "slab": (1024, 3.0), "slab_broken": (1024, 1.6),
    "ground_concrete": (1024, 3.0), "brick": (1024, 1.5), "tiles": (1024, 2.0), "asphalt": (1024, 4.0),
    "paving": (1024, 1.8), "rubble": (1024, 2.0), "rubble_dark": (1024, 2.0), "chunk_brick": (512, 1.8),
    "chunk_plaster": (512, 1.0), "plaster": (1024, 1.85), "plaster_ext": (1024, 4.6), "wall_cream": (1024, 3.0), "wall_green": (1024, 3.0),
    "wall_pink": (1024, 3.0), "wall_blue": (1024, 3.0), "wall_white": (1024, 3.0), "wood": (1024, 2.0),
    "wood_dark": (1024, 2.0), "beam": (512, 0.5), "rust": (512, 1.1), "fabric": (512, 0.6), "paint_metal": (512, 1.0),
    "bag": (512, 0.26),
}
for _k in SHIPPED:
    SETS.pop(_k, None)
TILE_M = {k: v[1] for k, v in SETS.items()}
TILE_M.update({k: v[1] for k, v in SHIPPED.items()})


def cache_dir():
    d = os.environ.get("RUBBLECELL_LOOK_DIR")
    if d:
        return d
    if os.path.isdir("/opt/rubblecell-look"):
        return "/opt/rubblecell-look"
    return os.path.join(os.path.expanduser("~"), ".cache", "rubblecell-look")


# ------------------------------------------------------------------------------------------------ noise primitives
def _spectral(n, rng, beta=2.0, lo=1.0, hi=None):
    """Tileable noise with a power spectrum ~ f^-beta between `lo` and `hi` cycles per tile: zero mean, unit std."""
    fx = np.fft.fftfreq(n) * n
    f = np.hypot(fx[None, :], fx[:, None])
    keep = f >= lo
    if hi is not None:
        keep &= f <= hi
    amp = np.zeros_like(f)
    amp[keep] = f[keep] ** (-beta / 2.0)
    ph = rng.uniform(0.0, 2.0 * math.pi, (n, n))
    z = np.fft.ifft2(amp * np.exp(1j * ph)).real
    z -= z.mean()
    s = z.std()
    return z / (s if s > 0 else 1.0)


def _u01(z, k=1.0):
    """A zero-mean field squashed into (0, 1)."""
    return 0.5 + 0.5 * np.tanh(k * z)


def _voronoi(n, k, rng, offsets=False):
    """Wrapped Voronoi cells of k random sites: (cell index, distance to the nearest site, to the second) in tile units;
    with `offsets` also each pixel's offset from its site (x, y; in pixels, continuous across the tile's edges)."""
    from scipy.spatial import cKDTree
    pts = rng.random((k, 2))
    offs = np.array([(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)], float)
    allp = (pts[None, :, :] + offs[:, None, :]).reshape(-1, 2)
    ids = np.tile(np.arange(k), 9)
    g = (np.arange(n) + 0.5) / n
    X, Y = np.meshgrid(g, g)
    d, i = cKDTree(allp).query(np.stack([X.ravel(), Y.ravel()], 1), k=2)
    out = (ids[i[:, 0]].reshape(n, n), d[:, 0].reshape(n, n), d[:, 1].reshape(n, n))
    if offsets:
        site = allp[i[:, 0]]
        out += ((X - site[:, 0].reshape(n, n)) * n, (Y - site[:, 1].reshape(n, n)) * n)
    return out


def _cracks(n, rng, count, length, width=1.4, wander=0.32, branch=0.25):
    """Tileable anti-aliased crack network in [0, 1] (`length` in pixels)."""
    import cv2
    canvas = np.zeros((n, n), np.float32)

    def walk(p, ang, L, w):
        pts = [p.copy()]
        step = 3.0
        for _ in range(max(2, int(L / step))):
            ang += rng.normal(0.0, wander)
            p = p + step * np.array([math.cos(ang), math.sin(ang)])
            pts.append(p.copy())
            if rng.random() < branch * step / L and w > 0.8:
                walk(p.copy(), ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.1), L * rng.uniform(0.2, 0.45), w * 0.6)
        pts = np.array(pts)
        for ox in (-n, 0, n):
            for oy in (-n, 0, n):
                q = np.round((pts + [ox, oy]) * 16).astype(np.int32)
                cv2.polylines(canvas, [q], False, 1.0, thickness=max(1, int(round(w))), lineType=cv2.LINE_AA, shift=4)

    for _ in range(count):
        walk(rng.random(2) * n, rng.uniform(0, 2 * math.pi), length * rng.uniform(0.5, 1.0), width * rng.uniform(0.7, 1.3))
    return np.clip(canvas, 0.0, 1.0)


def _blur(a, sigma):
    from scipy.ndimage import gaussian_filter
    if sigma <= 0:
        return a
    return gaussian_filter(np.asarray(a, np.float32), sigma, mode="wrap")


def _dots(n, rng, count, r0, r1):
    """Tileable soft round dots (pores, pebbles) in [0, 1]."""
    import cv2
    c = np.zeros((n, n), np.float32)
    for _ in range(count):
        x, y = rng.random(2) * n
        r = rng.uniform(r0, r1)
        for ox in (-n, 0, n):
            for oy in (-n, 0, n):
                cv2.circle(c, (int((x + ox) * 16), int((y + oy) * 16)), max(1, int(r * 16)), 1.0, -1, cv2.LINE_AA, 4)
    return np.clip(c, 0, 1)


def _normal(h, strength):
    """Tangent-space normal map (OpenGL convention, +y up the texture) of a height field (wrapped differences)."""
    dx = 0.5 * (np.roll(h, -1, 1) - np.roll(h, 1, 1))
    dy = 0.5 * (np.roll(h, -1, 0) - np.roll(h, 1, 0))
    nx, ny, nz = -dx * strength, dy * strength, np.ones_like(h)
    l = np.sqrt(nx * nx + ny * ny + nz * nz)
    return np.stack([nx / l, ny / l, nz / l], -1)


def _streaks(n, rng, density=40, length=0.6):
    """Tileable vertical runs (rain and dirt down a wall: the texture's v is up on wall faces), thin and long, in [0, 1]."""
    cols = _spectral(n, rng, 0.8, density * 2, None)
    cols = np.repeat(np.mean(cols, axis=0, keepdims=True), n, axis=0)
    along = _u01(_spectral(n, rng, 2.6, 1, 3), 1.2)
    lines = np.clip((_u01(cols, 1.0) - 0.62) * 4.0, 0, 1)
    return lines * np.clip(along * (1.0 / max(length, 0.1)) - 0.2, 0, 1)


def _cavity(h, sigma):
    """How far each pixel lies below its neighbourhood (for ambient occlusion), >= 0."""
    return np.clip(_blur(h, sigma) - h, 0, None)


def _pack(albedo, h, nstrength, rough, metal=0.0, ao=None):
    alb = np.clip(albedo, 0, 1)
    nrm = _normal(h, nstrength)
    if ao is None:
        ao = np.ones(h.shape)
    rough = np.broadcast_to(np.clip(rough, 0.04, 1.0), h.shape)
    metal = np.broadcast_to(np.clip(metal, 0.0, 1.0), h.shape)
    orm = np.stack([np.clip(ao, 0, 1), rough, metal], -1)
    u8 = lambda a: (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)  # noqa: E731
    return {"albedo": u8(alb), "normal": u8(nrm * 0.5 + 0.5), "orm": u8(orm)}


def _mix(a, b, t):
    t = t[..., None] if np.ndim(t) == 2 else t
    return a * (1 - t) + b * t


def _col(rgb, shape):
    return np.broadcast_to(np.asarray(rgb, float), shape + (3,)).copy()


# ------------------------------------------------------------------------------------------------ the materials
def _concrete(n, rng):
    sh = (n, n)
    big = _spectral(n, rng, 2.4, 1, 12)
    mid = _spectral(n, rng, 1.6, 6, 80)
    fine = _spectral(n, rng, 0.9, 60, None)
    pores = _dots(n, rng, int(n * n / 2600), 0.5, 1.6)
    cracks = _blur(_cracks(n, rng, 3, n * 0.8, 1.1), 0.6)
    stains = np.clip(_u01(_spectral(n, rng, 2.6, 1, 8), 1.2) - 0.62, 0, None) * 1.6
    v = (np.arange(n) + 0.5) / n
    board = np.clip(1 - np.abs(((v * 6) % 1.0) - 0.5) * 2 * 90, 0, 1)[:, None] * np.ones((1, n))   # formwork joints
    runs = _streaks(n, rng, 50, 0.7)
    base = np.array([0.64, 0.63, 0.60])
    alb = _col(base, sh) * (1 + 0.035 * big[..., None] + 0.025 * mid[..., None] + 0.03 * fine[..., None])
    alb = _mix(alb, alb * np.array([0.86, 0.85, 0.83]), np.clip(stains, 0, 0.7))
    alb = _mix(alb, alb * np.array([0.80, 0.79, 0.77]), runs * 0.6)
    rust = np.clip(_u01(_spectral(n, rng, 2.2, 3, 30), 1.5) - 0.93, 0, None) * 8.0
    alb = _mix(alb, np.array([0.50, 0.38, 0.28]), np.clip(rust * runs + rust * 0.3, 0, 0.35))
    alb = _mix(alb, alb * 0.85, board * 0.6)
    alb = _mix(alb, alb * 0.45, pores * 0.6)
    alb = _mix(alb, alb * 0.38, cracks * 0.85)
    h = 0.25 * mid + 0.15 * fine - 2.5 * pores - 3.0 * cracks - 0.6 * board
    rough = 0.86 + 0.05 * _u01(mid) + 0.06 * cracks
    ao = 1 - 0.55 * pores - 0.6 * cracks - 0.2 * board
    return _pack(alb, h, 0.9, rough, 0.0, ao)


def _rubble(n, rng):
    """The debris of a collapse under a coat of dust: broken slabs and blocks lying every which way (each piece a tilted
    fracture face with chipped edges), smaller lumps between them, gravel down in the crevices, the crevices in deep
    shadow; mostly concrete, some brick, plaster, tile and wood. Heights are in pixels (the normal map's slope is the
    pieces' tilt)."""
    sh = (n, n)
    pal = np.array([[0.58, 0.57, 0.54], [0.52, 0.30, 0.22], [0.78, 0.76, 0.71], [0.30, 0.29, 0.28],
                    [0.64, 0.42, 0.30], [0.44, 0.33, 0.22]])    # concrete, brick, plaster, dark, terracotta, wood
    p_kind = [0.56, 0.17, 0.10, 0.08, 0.05, 0.04]
    wob = _spectral(n, rng, 1.2, 8, 140)                        # ragged fracture lines
    fine = _spectral(n, rng, 1.0, 50, None)                     # the fracture faces' grain
    layer = []
    # slabs of ~25 cm (concrete, plaster, dark: no brick that big) over lumps of ~7 cm
    for sites, p_take, lift, pk in ((34, 0.5, 9.0, [0.72, 0, 0.14, 0.14, 0, 0]), (430, 1.0, 0.0, p_kind)):
        c, d1, d2, ox, oy = _voronoi(n, sites, rng, offsets=True)
        k = c.max() + 1
        take = rng.random(k) < p_take
        tilt = rng.normal(0.0, 0.42, (k, 2))
        kind = rng.choice(len(pal), size=k, p=pk)
        tint = pal[kind] * (0.84 + 0.26 * rng.random((k, 1)))
        fresh = rng.random(k)                                   # a freshly broken piece carries less dust
        e = np.clip((d2 - d1) * n * 0.5 + 1.6 * wob, 0, None)  # pixels in from the piece's border
        layer.append({"take": take[c], "face": tilt[c, 0] * ox + tilt[c, 1] * oy + lift, "e": e, "tint": tint[c],
                      "flat": 1.0 - np.clip(np.hypot(tilt[c, 0], tilt[c, 1]) / 0.8, 0, 1), "fresh": fresh[c]})
    slab, lump = layer
    on = slab["take"]
    face = np.where(on, slab["face"], lump["face"])
    e = np.where(on, slab["e"], lump["e"])
    tint = np.where(on[..., None], slab["tint"], lump["tint"])
    flat = np.where(on, slab["flat"], lump["flat"])
    fresh = np.where(on, slab["fresh"], lump["fresh"])
    mottle = _spectral(n, rng, 2.0, 6, 60)
    width = 2.2 + 2.6 * _u01(_spectral(n, rng, 1.6, 4, 40), 1.2)    # crevice half-widths of 3-6 mm
    crev = np.clip(1.0 - e / width, 0, 1)
    bevel = np.clip(1.0 - e / (width + 7.0), 0, 1) ** 2
    # gravel down in the crevices
    gc, g1, g2 = _voronoi(n, 2600, rng)
    gk = gc.max() + 1
    gtint = (0.5 * pal[rng.choice(len(pal), size=gk, p=p_kind)] + 0.5 * np.array([0.62, 0.60, 0.56])) \
        * (0.78 + 0.25 * rng.random((gk, 1)))
    gdome = np.sqrt(np.clip((g2 - g1) * n * 0.5 / 4.0, 0, 1))
    h = face + 1.4 * fine - 6.0 * bevel
    h = h * (1 - crev) + (-12.0 + 3.0 * gdome) * crev
    alb = tint * (1 + 0.07 * fine[..., None] + 0.06 * mottle[..., None])
    alb = _mix(alb, gtint[gc] * (0.7 + 0.3 * gdome[..., None]), crev)
    dustcol = np.array([0.70, 0.67, 0.61])
    patch = _u01(_spectral(n, rng, 2.2, 1, 12), 1.3)
    dust = np.clip((0.14 + 0.55 * patch * (0.45 + 0.55 * flat)) * (0.45 + 0.75 * fresh) - 0.25 * bevel, 0, 0.8)
    alb = _mix(alb, dustcol * (1 + 0.05 * fine[..., None]), dust * (1 - 0.6 * crev))
    alb = _mix(alb, alb * 0.30, np.clip(crev ** 1.3, 0, 1) * 0.85)
    ao = 1 - 0.7 * crev ** 1.2 - 0.12 * bevel
    rough = 0.88 + 0.08 * dust
    return _pack(alb, h, 1.0, rough, 0.0, ao)


def _brick(n, rng):
    sh = (n, n)
    rows, cols = 12, 4                                          # 0.9 m tile: 12 courses of 75 mm, 4 bricks of 225 mm
    v = (np.arange(n) + 0.5) / n * rows
    u = (np.arange(n) + 0.5) / n * cols
    V, U = np.meshgrid(v, u, indexing="ij")
    row = np.floor(V).astype(int)
    Us = U + 0.5 * (row % 2)
    col = np.floor(Us).astype(int) % cols
    fu, fv = Us - np.floor(Us), V - np.floor(V)
    mort_u, mort_v = 0.010 / 0.225, 0.010 / 0.075
    du = np.minimum(fu, 1 - fu) / mort_u
    dv = np.minimum(fv, 1 - fv) / mort_v
    jitter = _spectral(n, rng, 2.0, 8, 60) * 0.25
    inside = np.clip(np.minimum(du, dv) - 0.5 + jitter, 0, 1)
    bid = row * cols + col
    nb = rows * cols
    hue = rng.random(nb)
    reds = np.array([[0.56, 0.24, 0.16], [0.62, 0.30, 0.19], [0.47, 0.20, 0.14], [0.66, 0.38, 0.26], [0.36, 0.17, 0.13]])
    pick = np.minimum((hue * len(reds)).astype(int), len(reds) - 1)
    btint = reds[pick] * (0.85 + 0.25 * rng.random((nb, 1)))
    fine = _spectral(n, rng, 1.1, 30, None)
    mid = _spectral(n, rng, 1.8, 4, 40)
    alb = btint[bid] * (1 + 0.10 * fine[..., None] + 0.06 * mid[..., None])
    mortar = _col([0.60, 0.58, 0.54], sh) * (1 + 0.08 * fine[..., None])
    chips = np.clip(_u01(_spectral(n, rng, 1.4, 20, 200), 2.0) - 0.78, 0, None) * 5 * (inside < 0.999)
    alb = _mix(mortar, alb, np.clip(inside, 0, 1))
    soot = np.clip(_u01(_spectral(n, rng, 2.4, 1, 10), 1.6) - 0.5, 0, None) * 1.4
    alb = _mix(alb, alb * np.array([0.45, 0.43, 0.42]), np.clip(soot, 0, 0.7))
    dust = _u01(_spectral(n, rng, 2.0, 2, 30), 1.2)
    alb = _mix(alb, np.array([0.72, 0.69, 0.64]), 0.10 + 0.25 * dust * (1 - inside * 0.6))
    h = 1.0 * np.clip(inside * 2.0, 0, 1) + 0.18 * fine - 0.8 * np.clip(chips, 0, 1)
    ao = 1 - 0.45 * (1 - np.clip(inside * 2.5, 0, 1))
    rough = 0.86 + 0.08 * (1 - inside)
    return _pack(alb, h, 2.6, rough, 0.0, ao)


def _plaster(n, rng):
    """White paint over grey plaster (tinted per paint by the material colour): a few small peeled patches (a large
    wall repeats this tile many times: blotches would read as a pattern), cracks, faint grime and dirt runs."""
    sh = (n, n)
    fine = _spectral(n, rng, 1.0, 40, None)
    mid = _spectral(n, rng, 1.8, 4, 60)
    peel = _spectral(n, rng, 2.8, 1, 10) + 0.25 * _spectral(n, rng, 1.6, 10, 120)
    pm = np.clip((peel - 1.9) * 3.2, 0, 1)                     # 1 where the paint is gone (about 3 % of a wall)
    paint = _col([0.86, 0.85, 0.82], sh) * (1 + 0.018 * mid[..., None] + 0.012 * fine[..., None])
    plast = _col([0.78, 0.76, 0.72], sh) * (1 + 0.06 * fine[..., None] + 0.04 * mid[..., None])
    alb = _mix(paint, plast, pm)
    rim = np.clip(1 - np.abs(peel - 1.9) * 5.4, 0, 1)
    cracks = _blur(_cracks(n, rng, 3, n * 0.7, 1.1, wander=0.45), 0.5)
    grime = np.clip(_u01(_spectral(n, rng, 2.8, 1, 4), 1.0) - 0.5, 0, None) * 0.5
    runs = _streaks(n, rng, 60, 0.8)
    alb = _mix(alb, alb * np.array([0.88, 0.87, 0.85]), np.clip(grime, 0, 0.15))
    alb = _mix(alb, alb * np.array([0.86, 0.85, 0.83]), runs * 0.6)
    alb = _mix(alb, alb * 0.40, cracks * 0.8)
    h = -0.35 * pm + 0.3 * rim + 0.10 * fine + 0.06 * mid - 2.0 * cracks
    rough = 0.74 + 0.16 * pm + 0.04 * grime
    ao = 1 - 0.5 * cracks - 0.15 * pm
    return _pack(alb, h, 1.4, rough, 0.0, ao)


def _tiles(n, rng):
    sh = (n, n)
    k = 4                                                        # 0.3 m tiles in a 1.2 m repeat
    v = (np.arange(n) + 0.5) / n * k
    V, U = np.meshgrid(v, v, indexing="ij")
    fu, fv = U - np.floor(U), V - np.floor(V)
    tid = np.floor(V).astype(int) * k + np.floor(U).astype(int)
    g = 0.006 / 0.3
    dd = np.minimum(np.minimum(fu, 1 - fu), np.minimum(fv, 1 - fv)) / g
    inside = np.clip(dd - 0.5, 0, 1)
    nt = k * k
    base = np.array([0.80, 0.77, 0.71])
    ttint = base * (0.92 + 0.12 * rng.random((nt, 1)))
    fine = _spectral(n, rng, 1.0, 50, None)
    alb = ttint[tid] * (1 + 0.03 * fine[..., None])
    speck = _dots(n, rng, int(n * n / 900), 0.5, 1.4)
    alb = _mix(alb, alb * 0.55, speck * 0.5)
    grout = _col([0.42, 0.40, 0.37], sh) * (1 + 0.1 * fine[..., None])
    alb = _mix(grout, alb, inside)
    broken = rng.random(nt) < 0.35
    cr = _cracks(n, rng, 10, n * 0.25, 1.1, wander=0.25, branch=0.4)
    crk = np.where(broken[tid], cr, 0.0) * inside
    alb = _mix(alb, alb * 0.25, crk)
    dust = _u01(_spectral(n, rng, 1.8, 3, 60), 1.5)
    dmask = np.clip(dust - 0.45, 0, 1) * 1.1
    alb = _mix(alb, np.array([0.74, 0.71, 0.66]), np.clip(dmask, 0, 0.6))
    h = 0.9 * inside + 0.05 * fine - 1.0 * crk
    rough = np.where(inside > 0.5, 0.32, 0.88) * (1 - dmask * 0.6) + dmask * 0.6 * 0.95
    ao = 1 - 0.4 * (1 - inside) - 0.4 * crk
    return _pack(alb, h, 2.0, rough, 0.0, ao)


def _wood(n, rng):
    sh = (n, n)
    v = (np.arange(n) + 0.5) / n
    V, U = np.meshgrid(v, v, indexing="ij")
    warp = _spectral(n, rng, 2.4, 1, 12) * 0.025 + _spectral(n, rng, 1.8, 4, 40) * 0.004
    rings = np.sin((V + warp) * 2 * math.pi * 46)
    grain = _u01(rings, 1.6)
    fine = _spectral(n, rng, 1.0, 40, None)
    streak = _spectral(n, rng, 1.2, 4, 200)
    streak = np.repeat(np.mean(streak, axis=1, keepdims=True), n, axis=1)   # long streaks along u
    alb = _mix(_col([0.62, 0.45, 0.28], sh), _col([0.42, 0.28, 0.16], sh), grain * 0.55 + 0.15 * _u01(streak))
    alb = alb * (1 + 0.05 * fine[..., None])
    knots = _dots(n, rng, 3, 6, 14)
    alb = _mix(alb, np.array([0.26, 0.16, 0.09]), _blur(knots, 3) * 0.8)
    seams = np.clip(1 - np.abs(((V * 6) % 1.0) - 0.5) * 2 * 60, 0, 1)        # plank joints every 0.2 m
    alb = _mix(alb, alb * 0.3, seams)
    dust = _u01(_spectral(n, rng, 2.0, 2, 30), 1.3)
    alb = _mix(alb, np.array([0.70, 0.67, 0.61]), 0.12 + 0.25 * dust)
    h = 0.35 * grain + 0.25 * _u01(streak) + 0.1 * fine - 2.0 * seams
    rough = 0.66 + 0.2 * dust
    ao = 1 - 0.5 * seams
    return _pack(alb, h, 1.4, rough, 0.0, ao)


def _asphalt(n, rng):
    sh = (n, n)
    fine = _spectral(n, rng, 0.6, 120, None)
    mid = _spectral(n, rng, 1.2, 20, 160)
    big = _spectral(n, rng, 2.4, 1, 8)
    stones = _dots(n, rng, int(n * n / 140), 0.5, 1.3)
    darkst = _dots(n, rng, int(n * n / 260), 0.5, 1.2)
    lum = 0.20 * (1 + 0.10 * fine + 0.03 * mid + 0.01 * big)
    lum = lum + 0.16 * stones - 0.06 * darkst
    alb = _col([1.0, 1.0, 1.02], sh) * lum[..., None]
    patch = np.clip(_u01(_spectral(n, rng, 2.6, 1, 6), 2.0) - 0.93, 0, None) * 6
    alb = _mix(alb, alb * 0.9, np.clip(patch, 0, 1))
    cracks = _blur(_cracks(n, rng, 4, n * 0.6, 1.4, wander=0.3), 0.7)
    alb = _mix(alb, _col([0.06, 0.06, 0.065], sh), cracks * 0.9)
    oil = np.clip(_u01(_spectral(n, rng, 2.6, 2, 12), 1.8) - 0.9, 0, None) * 5
    alb = _mix(alb, alb * 0.8, np.clip(oil, 0, 0.5))
    dust = _u01(_spectral(n, rng, 2.2, 1, 12), 1.2)
    alb = _mix(alb, np.array([0.52, 0.50, 0.47]), np.clip((dust - 0.5) * 0.35, 0, 0.15))
    h = 0.8 * stones + 0.25 * fine + 0.1 * mid - 1.5 * cracks
    rough = np.clip(0.88 - 0.35 * cracks - 0.25 * np.clip(oil, 0, 1) - 0.1 * stones, 0.3, 1.0)
    ao = 1 - 0.25 * (1 - stones) * 0.5 - 0.4 * cracks
    return _pack(alb, h, 1.6, rough, 0.0, ao)


def _paving(n, rng):
    sh = (n, n)
    k = 4
    v = (np.arange(n) + 0.5) / n * k
    V, U = np.meshgrid(v, v, indexing="ij")
    fu, fv = U - np.floor(U), V - np.floor(V)
    pid = np.floor(V).astype(int) * k + np.floor(U).astype(int)
    g = 0.008 / 0.3
    dd = np.minimum(np.minimum(fu, 1 - fu), np.minimum(fv, 1 - fv)) / g
    jit = _spectral(n, rng, 1.8, 10, 80) * 0.6
    inside = np.clip(dd - 0.6 + jit, 0, 1)
    npv = k * k
    ptint = np.array([0.60, 0.58, 0.55]) * (0.86 + 0.2 * rng.random((npv, 1)))
    fine = _spectral(n, rng, 1.0, 50, None)
    mid = _spectral(n, rng, 1.8, 6, 60)
    alb = ptint[pid] * (1 + 0.06 * fine[..., None] + 0.05 * mid[..., None])
    alb = _mix(_col([0.20, 0.18, 0.15], sh), alb, inside)
    stains = np.clip(_u01(_spectral(n, rng, 2.4, 1, 14), 1.4) - 0.55, 0, None) * 1.8
    alb = _mix(alb, alb * 0.6, np.clip(stains, 0, 0.6))
    dust = _u01(_spectral(n, rng, 1.8, 3, 60), 1.3)
    alb = _mix(alb, np.array([0.70, 0.67, 0.62]), 0.08 + 0.18 * dust)
    tilt = (rng.random(npv) - 0.5)[pid] * 0.3
    h = 0.9 * inside + 0.08 * fine + tilt * inside
    rough = 0.86 + 0.08 * (1 - inside)
    ao = 1 - 0.55 * (1 - inside)
    return _pack(alb, h, 1.8, rough, 0.0, ao)


def _fabric(n, rng):
    sh = (n, n)
    v = (np.arange(n) + 0.5) / n
    V, U = np.meshgrid(v, v, indexing="ij")
    f = 140
    twill = 0.5 + 0.5 * np.sin(((U + V) * f) * 2 * math.pi) * np.sin((U * f * 0.5) * 2 * math.pi)
    fine = _spectral(n, rng, 1.0, 30, None)
    folds = _spectral(n, rng, 2.4, 1, 8)
    alb = _col([0.82, 0.82, 0.82], sh) * (0.9 + 0.1 * twill[..., None]) * (1 + 0.05 * fine[..., None])
    alb = alb * (1 + 0.08 * folds[..., None])
    dirt = np.clip(_u01(_spectral(n, rng, 2.2, 1, 20), 1.4) - 0.4, 0, None) * 1.2
    alb = _mix(alb, np.array([0.62, 0.59, 0.54]), np.clip(dirt, 0, 0.55))
    h = 0.4 * twill + 0.6 * folds * 0.3
    return _pack(alb, h, 1.2, 0.94, 0.0, 1 - 0.15 * (1 - twill))


def _paint_metal(n, rng):
    """Painted sheet metal (cars, the team's van, appliances, signs; tinted per paint): an even coat with a faint orange
    peel under a light film of dust, a few fine scratches, rare small chips."""
    sh = (n, n)
    peel = _spectral(n, rng, 1.0, 60, None)
    film = _u01(_spectral(n, rng, 2.4, 1, 10), 1.0)
    scr = _blur(_cracks(n, rng, 9, n * 0.14, 0.8, wander=0.04, branch=0.0), 0.3)
    chip = np.clip((_spectral(n, rng, 1.6, 10, 160) - 2.7) * 3.0, 0, 1)
    alb = _col([0.86, 0.86, 0.86], sh) * (1 + 0.012 * peel[..., None])
    alb = _mix(alb, np.array([0.72, 0.70, 0.66]), 0.05 + 0.13 * film)
    alb = _mix(alb, np.array([0.66, 0.66, 0.66]), scr * 0.30)
    alb = _mix(alb, np.array([0.42, 0.27, 0.17]), chip * 0.8)
    h = 0.04 * peel - 0.35 * scr - 0.6 * chip
    rough = 0.30 + 0.25 * film + 0.35 * chip
    metal = 0.5 * scr
    return _pack(alb, h, 1.0, rough, metal, 1 - 0.15 * chip)


def _rust(n, rng):
    sh = (n, n)
    a = _u01(_spectral(n, rng, 1.6, 2, None), 1.4)
    b = _u01(_spectral(n, rng, 1.0, 30, None), 1.4)
    alb = _mix(_col([0.42, 0.20, 0.09], sh), _col([0.62, 0.36, 0.18], sh), a)
    alb = _mix(alb, _col([0.22, 0.12, 0.07], sh), np.clip(b - 0.6, 0, 1) * 1.5)
    h = 0.6 * a + 0.5 * b
    return _pack(alb, h, 1.6, 0.88, 0.15, 1 - 0.2 * (1 - b))


def _rebar(n, rng):
    sh = (n, n)
    v = (np.arange(n) + 0.5) / n
    V, U = np.meshgrid(v, v, indexing="ij")
    ribs = np.clip(np.sin(U * 2 * math.pi * 24) * 2.0, 0, 1)
    r = _u01(_spectral(n, rng, 1.4, 2, None), 1.5)
    alb = _mix(_col([0.36, 0.33, 0.31], sh), _col([0.50, 0.27, 0.13], sh), r)
    h = 0.8 * ribs + 0.2 * r
    return _pack(alb, h, 2.0, 0.62 + 0.2 * r, 0.75 - 0.5 * r, 1 - 0.3 * (1 - ribs))


def _dust(n, rng):
    sh = (n, n)
    a = _spectral(n, rng, 1.6, 2, None)
    b = _dots(n, rng, int(n * n / 600), 0.5, 1.6)
    alb = _col([0.70, 0.67, 0.61], sh) * (1 + 0.06 * a[..., None])
    alb = _mix(alb, np.array([0.40, 0.38, 0.35]), b * 0.5)
    return _pack(alb, 0.3 * a + 0.6 * b, 1.2, 0.97, 0.0, 1 - 0.25 * b)


def _track(n, rng):
    sh = (n, n)
    v = (np.arange(n) + 0.5) / n
    V, U = np.meshgrid(v, v, indexing="ij")
    bars = np.clip((np.abs(((U * 8) % 1.0) - 0.5) - 0.18) * 12, 0, 1)
    fine = _spectral(n, rng, 1.0, 30, None)
    alb = _col([0.10, 0.10, 0.10], sh) * (1 + 0.15 * fine[..., None])
    dust = _u01(_spectral(n, rng, 2.0, 1, 20), 1.2)
    alb = _mix(alb, np.array([0.45, 0.43, 0.40]), 0.12 + 0.25 * dust * (1 - bars))
    return _pack(alb, 1.2 * bars + 0.1 * fine, 2.2, 0.86, 0.0, 1 - 0.4 * (1 - bars))


def _skin(n, rng):
    sh = (n, n)
    a = _spectral(n, rng, 1.8, 2, 60)
    b = _spectral(n, rng, 1.0, 40, None)
    alb = _col([0.92, 0.90, 0.88], sh) * (1 + 0.04 * a[..., None] + 0.02 * b[..., None])
    dirt = np.clip(_u01(_spectral(n, rng, 2.4, 1, 6), 1.2) - 0.5, 0, None) * 0.8
    alb = _mix(alb, np.array([0.70, 0.66, 0.60]), np.clip(dirt, 0, 0.18))
    return _pack(alb, 0.2 * b, 0.6, 0.56 + 0.15 * dirt, 0.0, None)


def _hair(n, rng):
    sh = (n, n)
    s = _spectral(n, rng, 1.0, 20, None)
    s = np.repeat(np.mean(s, axis=0, keepdims=True), n, axis=0)
    alb = _col([0.85, 0.85, 0.85], sh) * (1 + 0.2 * s[..., None])
    dust = _u01(_spectral(n, rng, 2.0, 1, 10), 1.4)
    alb = _mix(alb, np.array([0.70, 0.67, 0.61]), 0.25 * dust)
    return _pack(alb, 0.5 * s, 1.0, 0.75, 0.0, None)


def _robot_paint(n, rng):
    sh = (n, n)
    peel = _spectral(n, rng, 1.0, 60, None)
    scr = _blur(_cracks(n, rng, 18, n * 0.12, 1.0, wander=0.08, branch=0.0), 0.4)
    alb = _col([0.97, 0.97, 0.97], sh) * (1 + 0.012 * peel[..., None])
    alb = _mix(alb, np.array([0.30, 0.30, 0.31]), scr * 0.6)
    dust = _u01(_spectral(n, rng, 2.0, 1, 16), 1.3)
    alb = _mix(alb, np.array([0.74, 0.71, 0.65]), 0.05 + 0.22 * dust)
    rough = 0.32 + 0.35 * dust * 0.6 + 0.2 * scr
    return _pack(alb, 0.03 * peel - 0.5 * scr, 1.0, rough, 0.6 * scr, None)


def _robot_dark(n, rng):
    sh = (n, n)
    f = _spectral(n, rng, 1.0, 40, None)
    alb = _col([0.95, 0.95, 0.95], sh) * (1 + 0.03 * f[..., None])
    dust = _u01(_spectral(n, rng, 2.0, 1, 16), 1.3)
    alb = _mix(alb, np.array([0.72, 0.69, 0.64]) / 0.25, 0.04 + 0.1 * dust)
    return _pack(np.clip(alb, 0, 1), 0.05 * f, 0.8, 0.48 + 0.3 * dust, 0.0, None)


def _alu(n, rng):
    sh = (n, n)
    s = _spectral(n, rng, 1.0, 20, None)
    s = 0.7 * np.repeat(np.mean(s, axis=1, keepdims=True), n, axis=1) + 0.3 * s
    alb = _col([0.91, 0.91, 0.92], sh) * (1 + 0.03 * s[..., None])
    return _pack(alb, 0.1 * s, 0.6, 0.34 + 0.06 * _u01(s), 1.0, None)


def _road_paint(n, rng):
    sh = (n, n)
    wear = _u01(_spectral(n, rng, 1.6, 2, 120), 1.8)
    alb = _mix(_col([0.88, 0.88, 0.85], sh), _col([0.30, 0.30, 0.31], sh), np.clip((wear - 0.62) * 4, 0, 1))
    dust = _u01(_spectral(n, rng, 2.0, 1, 20), 1.2)
    alb = _mix(alb, np.array([0.66, 0.64, 0.59]), 0.2 * dust)
    return _pack(alb, 0.3 * (1 - wear), 0.8, 0.62, 0.0, None)


def _bag(n, rng):
    sh = (n, n)
    v = (np.arange(n) + 0.5) / n
    V, U = np.meshgrid(v, v, indexing="ij")
    grid = np.maximum(np.clip(1 - np.abs(((U * 20) % 1.0) - 0.5) * 2 * 8, 0, 1),
                      np.clip(1 - np.abs(((V * 20) % 1.0) - 0.5) * 2 * 8, 0, 1))
    fine = _spectral(n, rng, 1.0, 40, None)
    alb = _col([0.95, 0.95, 0.95], sh) * (1 - 0.08 * grid[..., None]) * (1 + 0.03 * fine[..., None])
    dust = _u01(_spectral(n, rng, 2.0, 1, 12), 1.2)
    alb = _mix(alb, np.array([0.74, 0.71, 0.66]), 0.06 + 0.2 * dust)
    return _pack(alb, 0.3 * grid + 0.1 * fine, 1.0, 0.7, 0.0, None)


def _facade(n, rng, kind):
    """A street facade of two storeys by two window bays (6.4 m square): wall, windows (dark glass in frames, sills),
    some panes broken, grime and runs below the sills, cracks."""
    sh = (n, n)
    px = n / 6.4                                                # pixels per metre
    v = (np.arange(n) + 0.5) / px                               # metres up (row 0 is the top: flipped below)
    u = (np.arange(n) + 0.5) / px
    V, U = np.meshgrid(v[::-1], u, indexing="ij")              # V: metres above the repeat's bottom
    fine = _spectral(n, rng, 1.0, 60, None)
    mid = _spectral(n, rng, 1.8, 4, 60)
    if kind == "brick":
        wall = _col([0.55, 0.30, 0.22], sh) * (1 + 0.10 * fine[..., None] + 0.08 * mid[..., None])
        course = np.clip(1 - np.abs(((V / 0.075) % 1.0) - 0.5) * 2 * 6, 0, 1)
        wall = _mix(wall, _col([0.62, 0.60, 0.56], sh), course * 0.7)
        hwall = -0.5 * course
    elif kind == "panel":
        wall = _col([0.66, 0.65, 0.62], sh) * (1 + 0.05 * fine[..., None] + 0.05 * mid[..., None])
        joint = np.maximum(np.clip(1 - np.abs(((V / 3.2) % 1.0)) * 3.2 * 40, 0, 1),
                           np.clip(1 - np.abs(((U / 3.2) % 1.0)) * 3.2 * 40, 0, 1))
        wall = _mix(wall, wall * 0.5, joint)
        hwall = -1.0 * joint
    else:
        tint = np.array([[0.86, 0.80, 0.68], [0.80, 0.78, 0.74], [0.84, 0.70, 0.58], [0.76, 0.74, 0.66]])[rng.integers(4)]
        wall = _col(tint, sh) * (1 + 0.03 * fine[..., None] + 0.04 * mid[..., None])
        hwall = 0.05 * fine
    alb = wall
    h = hwall.copy()
    rough = np.full(sh, 0.85)
    metal = np.zeros(sh)
    stain = np.zeros(sh)
    # windows: 1.3 m wide, 1.5 m tall, sills at 0.9 m above each floor, centred in each 3.2 m bay
    frame_col = np.array([[0.90, 0.90, 0.88], [0.35, 0.24, 0.16], [0.22, 0.22, 0.24]])[rng.integers(3)]
    for st in range(2):
        for b in range(2):
            x0, x1 = b * 3.2 + 0.95, b * 3.2 + 2.25
            z0, z1 = st * 3.2 + 0.9, st * 3.2 + 2.4
            win = (U > x0) & (U < x1) & (V > z0) & (V < z1)
            inner = (U > x0 + 0.06) & (U < x1 - 0.06) & (V > z0 + 0.06) & (V < z1 - 0.06)
            mull = inner & (np.abs(U - (x0 + x1) / 2) < 0.025)
            broken = rng.random() < 0.25
            glass = np.array([0.05, 0.06, 0.07]) if not broken else np.array([0.015, 0.014, 0.013])
            alb = np.where(win[..., None], frame_col, alb)
            alb = np.where((inner & ~mull)[..., None], glass, alb)
            rough = np.where(inner & ~mull, 0.06 if not broken else 0.95, rough)
            h = np.where(win, h - 1.2, h)
            h = np.where(inner & ~mull, h - 0.6, h)
            sill = (U > x0 - 0.08) & (U < x1 + 0.08) & (V > z0 - 0.07) & (V < z0)
            alb = np.where(sill[..., None], _col([0.70, 0.69, 0.66], sh), alb)
            h = np.where(sill, h + 1.0, h)
            run = (U > x0) & (U < x1) & (V < z0 - 0.07) & (V > z0 - 1.2)
            fall = np.clip((V - (z0 - 1.2)) / 1.1, 0, 1)
            stain = np.where(run, np.maximum(stain, fall), stain)
    cracks = _blur(_cracks(n, rng, 3, n * 0.5, 1.2, wander=0.4), 0.6)
    soot = np.clip(_u01(_spectral(n, rng, 2.6, 1, 6), 1.4) - 0.6, 0, None) * 1.2
    drips = _streaks(n, rng, 110, 1.0)                          # dirt washed down from each sill, in runs
    rain = _streaks(n, rng, 40, 0.9)                            # faint rain marks down the whole wall
    alb = alb * (1 - stain * (0.05 + 0.25 * drips))[..., None]
    alb = alb * (1 - 0.07 * rain)[..., None]
    alb = _mix(alb, alb * 0.8, np.clip(soot * 0.3, 0, 0.1))     # a faint unevenness of the weathering, no blotches
    alb = _mix(alb, alb * 0.4, cracks * 0.8)
    h = h - 1.5 * cracks
    return _pack(alb, h, 1.2, rough, metal, 1 - 0.3 * np.clip(-h, 0, 1))


def _shopfront(n, rng):
    """A shop storey (the repeat's lower half: 3.2 m) under the facade's first floor: glass, shutters, signs."""
    sh = (n, n)
    px = n / 6.4
    v = (np.arange(n) + 0.5) / px
    V, U = np.meshgrid(v[::-1], v, indexing="ij")
    fine = _spectral(n, rng, 1.0, 60, None)
    wall = _col([0.30, 0.29, 0.28], sh) * (1 + 0.06 * fine[..., None])
    alb = wall.copy()
    rough = np.full(sh, 0.8)
    h = np.zeros(sh)
    for b in range(2):
        x0, x1 = b * 3.2 + 0.25, b * 3.2 + 2.95
        kind = rng.integers(3)
        win = (U > x0) & (U < x1) & (V > 0.15) & (V < 2.55)
        if kind == 0:                                           # glass and its frame
            alb = np.where(win[..., None], np.array([0.05, 0.06, 0.07]), alb)
            rough = np.where(win, 0.06, rough)
        elif kind == 1:                                         # a roller shutter
            slats = 0.5 + 0.5 * np.sin(V / 0.06 * 2 * math.pi)
            alb = np.where(win[..., None], _col([0.52, 0.53, 0.54], sh) * (0.85 + 0.15 * slats[..., None]), alb)
            h = np.where(win, h + 0.5 * slats, h)
            rough = np.where(win, 0.5, rough)
        else:                                                   # boarded up
            alb = np.where(win[..., None], _col([0.50, 0.36, 0.22], sh) * (1 + 0.1 * fine[..., None]), alb)
            rough = np.where(win, 0.8, rough)
        h = np.where(win, h - 1.0, h)
        sign = (U > x0) & (U < x1) & (V > 2.65) & (V < 3.1)
        sc = np.array([[0.70, 0.12, 0.10], [0.12, 0.30, 0.55], [0.85, 0.70, 0.15], [0.15, 0.45, 0.25]])[rng.integers(4)]
        alb = np.where(sign[..., None], sc * (1 + 0.05 * fine[..., None]), alb)
        rough = np.where(sign, 0.45, rough)
    upper = V > 3.2                                             # the repeat's upper half: a plain band (cropped by size)
    alb = np.where(upper[..., None], _col([0.62, 0.60, 0.56], sh) * (1 + 0.05 * fine[..., None]), alb)
    dirt = np.clip(1 - V / 0.8, 0, 1) * 0.3
    alb = alb * (1 - dirt[..., None])
    return _pack(alb, h, 1.0, rough, 0.0, None)


def _foliage(n, rng):
    sh = (n, n)
    cell, d1, d2 = _voronoi(n, 900, rng)
    k = cell.max() + 1
    g = np.array([0.20, 0.30, 0.12]) * (0.6 + 0.6 * rng.random((k, 1)))
    alb = g[cell]
    edge = np.clip((d2 - d1) * n / 3.0, 0, 1)
    alb = alb * (0.5 + 0.5 * edge[..., None])
    dust = _u01(_spectral(n, rng, 2.0, 1, 20), 1.2)
    alb = _mix(alb, np.array([0.55, 0.52, 0.45]), 0.25 * dust)
    return _pack(alb, 1.5 * edge, 2.0, 0.75, 0.0, 0.6 + 0.4 * edge)


def _roof(n, rng):
    sh = (n, n)
    a = _spectral(n, rng, 1.0, 80, None)
    b = _spectral(n, rng, 2.2, 1, 10)
    alb = _col([0.30, 0.29, 0.28], sh) * (1 + 0.15 * a[..., None] + 0.08 * b[..., None])
    return _pack(alb, 0.5 * a, 1.0, 0.9, 0.0, None)


_MAKERS = {"concrete": _concrete, "rubble": _rubble, "brick": _brick, "plaster": _plaster, "tiles": _tiles,
           "wood": _wood, "asphalt": _asphalt, "paving": _paving, "fabric": _fabric, "paint_metal": _paint_metal,
           "rust": _rust, "rebar": _rebar, "dust": _dust, "track": _track, "skin": _skin, "hair": _hair,
           "robot_paint": _robot_paint, "robot_dark": _robot_dark, "alu": _alu, "road_paint": _road_paint,
           "bag": _bag, "facade_plaster": lambda n, r: _facade(n, r, "plaster"),
           "facade_brick": lambda n, r: _facade(n, r, "brick"), "facade_panel": lambda n, r: _facade(n, r, "panel"),
           "shopfront": _shopfront, "foliage": _foliage, "roof": _roof}


def make_set(name):
    n = SETS[name][0]
    seed = int.from_bytes(("%s/%s" % (SEED_TAG, name)).encode(), "little") % (2 ** 63)
    return _MAKERS[name](n, np.random.default_rng(seed))


def texture_sets(directory=None, names=None):
    """{set: {"albedo", "normal", "orm": uint8 (n, n, 3)}}, loaded from `directory` (made and saved when missing)."""
    from PIL import Image
    directory = os.path.join(directory or cache_dir(), VERSION)
    out = {}
    for name in names or (list(SETS) + list(SHIPPED)):
        paths = {k: os.path.join(directory, "%s_%s.png" % (name, k)) for k in ("albedo", "normal", "orm")}
        if all(os.path.isfile(p) for p in paths.values()):
            out[name] = {k: np.asarray(Image.open(p).convert("RGB")) for k, p in paths.items()}
            continue
        if name in SHIPPED:
            raise RuntimeError("the look pack's texture set %r is missing in %s" % (name, directory))
        s = make_set(name)
        try:
            os.makedirs(directory, exist_ok=True)
            for k, p in paths.items():
                tmp = p + ".tmp%d" % os.getpid()
                Image.fromarray(s[k]).save(tmp, format="PNG", compress_level=6)
                os.replace(tmp, p)
        except OSError:
            pass                                                 # read-only cache: keep the set in memory only
        out[name] = s
    return out


if __name__ == "__main__":
    import sys
    import time
    d = sys.argv[1] if len(sys.argv) > 1 else cache_dir()
    t0 = time.time()
    sets = texture_sets(d)
    print("[rubblecell.pbr] %d sets (%s) in %s, %.1f s" % (len(sets), VERSION, os.path.join(d, VERSION), time.time() - t0))
