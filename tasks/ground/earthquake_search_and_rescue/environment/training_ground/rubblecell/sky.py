"""The sky over the rubble site (cell internals): a procedural high-dynamic-range sky (a hazy, dusty late-day sky with
the sun's glow, broken cloud and smoke columns rising from fires in the city) turned into the two cube maps Filament
draws with, written as KTX1 files: the image light (prefiltered specular levels plus the irradiance's spherical
harmonics, as Filament's cmgen writes them) and the skybox the cameras see.

MuJoCo's Filament renderer samples the cube map in a y-up frame: a world direction (x, y, z) (z up) reads the cube map
at (x, z, -y), so world up is the cube's +y face and north (+y) its -z face (found with a test sky whose six faces had
distinct colours; `_to_world` is the inverse). The irradiance harmonics are computed in the cube map's frame.
"""

import math
import struct

import numpy as np

__all__ = ["make_sky", "cube_dirs", "write_ktx", "radiance"]

GL_R11F_G11F_B10F = 0x8C3A
SKYBOX_GAIN = 0.5           # the sky the cameras see, relative to the image light it casts (keeps it from blowing out)
GL_RGB = 0x1907


# --------------------------------------------------------------------------------------------- cube maps and KTX
def cube_dirs(n):
    """(6, n, n, 3) unit directions of the texel centres of an OpenGL cube map (faces +x, -x, +y, -y, +z, -z; row 0 is
    the face's top), and (n, n) solid angles of one face's texels."""
    c = (np.arange(n) + 0.5) / n * 2.0 - 1.0
    sc, tc = np.meshgrid(c, c)                                 # sc along a row (u), tc down the rows (v)
    one = np.ones_like(sc)
    faces = [np.stack([one, -tc, -sc], -1), np.stack([-one, -tc, sc], -1), np.stack([sc, one, tc], -1),
             np.stack([sc, -one, -tc], -1), np.stack([sc, -tc, one], -1), np.stack([-sc, -tc, -one], -1)]
    d = np.stack(faces, 0)
    l2 = (d * d).sum(-1, keepdims=True)
    d = d / np.sqrt(l2)
    dw = (2.0 / n) ** 2 / (l2[0, ..., 0] ** 1.5)
    return d, dw


def _pack_r11g11b10(rgb):
    """float32 (..., 3) >= 0 -> uint32 packed R11F_G11F_B10F (truncating the mantissas, via float16's 5-bit exponent)."""
    x = np.clip(np.asarray(rgb, np.float32), 0.0, 64000.0)
    h = x.astype(np.float16).view(np.uint16).astype(np.uint32)
    r = (h[..., 0] >> 4) & 0x7FF
    g = (h[..., 1] >> 4) & 0x7FF
    b = (h[..., 2] >> 5) & 0x3FF
    return (r | (g << 11) | (b << 22)).astype("<u4")


def write_ktx(levels, sh=None):
    """KTX1 bytes of a cube map: `levels` = [(6, n, n, 3) float32, ...] from the largest; `sh` = (9, 3) or None."""
    n0 = levels[0].shape[1]
    kv = b""
    if sh is not None:
        txt = "".join("%g %g %g\n" % tuple(float(v) for v in row) for row in sh).encode()
        body = b"sh\x00" + txt
        kv = struct.pack("<I", len(body)) + body
        kv += b"\x00" * ((4 - len(kv) % 4) % 4)
    hdr = struct.pack("<13I", 0x04030201, GL_R11F_G11F_B10F, 1, GL_RGB, GL_R11F_G11F_B10F, GL_R11F_G11F_B10F,
                      n0, n0, 0, 0, 6, len(levels), len(kv))
    out = [b"\xabKTX 11\xbb\r\n\x1a\n", hdr, kv]
    for lv in levels:
        n = lv.shape[1]
        out.append(struct.pack("<I", n * n * 4))
        for f in range(6):
            out.append(_pack_r11g11b10(lv[f]).tobytes())
    return b"".join(out)


def _to_world(c):
    """World directions (z up) of cube-map directions."""
    return np.stack([c[..., 0], -c[..., 2], c[..., 1]], -1)


def irradiance_sh(rad, dw, d):
    """Filament's shader-ready irradiance SH (9 x rgb) of a cube map of radiance (6, n, n, 3)."""
    x, y, z = d[..., 0], d[..., 1], d[..., 2]
    polys = [np.ones_like(x), y, z, x, x * y, y * z, 3 * z * z - 1, x * z, x * x - y * y]
    sp = math.sqrt(math.pi)
    K = [1 / (2 * sp), math.sqrt(3) / (2 * sp), math.sqrt(3) / (2 * sp), math.sqrt(3) / (2 * sp),
         math.sqrt(15) / (2 * sp), math.sqrt(15) / (2 * sp), math.sqrt(5) / (4 * sp), math.sqrt(15) / (2 * sp),
         math.sqrt(15) / (4 * sp)]
    Ahat = [math.pi, 2 * math.pi / 3, 2 * math.pi / 3, 2 * math.pi / 3] + [math.pi / 4] * 5
    w = dw[None, ..., None]
    out = np.zeros((9, 3))
    for i in range(9):
        proj = (rad * polys[i][..., None] * w).sum(axis=(0, 1, 2))
        out[i] = Ahat[i] * K[i] * K[i] * proj / math.pi
    return out


# --------------------------------------------------------------------------------------------- the sky
def _noise2(u, v, seed, octaves=6, base=1.0):
    """Smooth 2-D value-noise fbm at arbitrary points (u, v arrays), in [0, 1]."""
    rng = np.random.default_rng(seed)
    out = np.zeros_like(u)
    amp, tot, f = 1.0, 0.0, base
    for o in range(octaves):
        g = rng.random((64, 64))
        x, y = u * f, v * f
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
        i0, j0 = x0.astype(int) % 64, y0.astype(int) % 64
        i1, j1 = (i0 + 1) % 64, (j0 + 1) % 64
        a = g[j0, i0] + (g[j0, i1] - g[j0, i0]) * fx
        b = g[j1, i0] + (g[j1, i1] - g[j1, i0]) * fx
        out += amp * (a + (b - a) * fy)
        tot += amp
        amp *= 0.5
        f *= 2.03
    return out / tot


_HDRI = {}
HDRI_GAIN = 0.65            # the photographed sky's mean radiance over the upper hemisphere, in the made sky's units


def _hdri_mips():
    """{width: (h, w, 3) float32} of the look pack's photographed sky (equirect, rows from the zenith down, the sun's
    azimuth in the middle column, the sun's disc clipped, mean of the upper hemisphere 1), or None without a pack."""
    if "m" not in _HDRI:
        import os
        from . import pbr
        d = os.path.join(pbr.cache_dir(), "sky")
        m = {}
        for w in (8, 16, 32, 64, 128, 256, 512, 1024):
            p = os.path.join(d, "sky_%d.npy" % w)
            if os.path.isfile(p):
                m[w] = np.load(p).astype(np.float32)
        _HDRI["m"] = m or None
    return _HDRI["m"]


def _hdri_radiance(d, P, blur, sun_disk):
    M = _hdri_mips()
    want = 2.0 * math.pi / max(blur, 1e-4)
    w = max([k for k in M if k <= max(8.0, want)] or [min(M)])
    img = M[w]
    h = img.shape[0]
    sun = np.asarray(P["sun_to"], float)
    az = np.arctan2(d[..., 1], d[..., 0]) - math.atan2(sun[1], sun[0])
    el = np.arcsin(np.clip(d[..., 2], -1.0, 1.0))
    u = (az / (2.0 * math.pi) + 0.5) % 1.0 * w - 0.5
    v = np.clip((0.5 - el / math.pi) * h - 0.5, 0.0, h - 1.0)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fu, fv = (u - u0)[..., None], (v - v0)[..., None]
    v1 = np.minimum(v0 + 1, h - 1)
    a = img[v0, u0 % w] * (1 - fu) + img[v0, (u0 + 1) % w] * fu
    b = img[v1, u0 % w] * (1 - fu) + img[v1, (u0 + 1) % w] * fu
    sky = (a * (1 - fv) + b * fv).astype(np.float64) * HDRI_GAIN
    if sun_disk:
        gam = np.arccos(np.clip((d * sun).sum(-1), -1.0, 1.0))
        disk = np.clip((math.radians(0.55) - gam) / math.radians(0.12), 0.0, 1.0)
        sky = sky + disk[..., None] * np.asarray(P["sun_rgb"]) * P["disk"]
    return np.maximum(sky, 0.0)


def radiance(d, P, blur=0.0, sun_disk=False):
    """Linear HDR radiance of the sky in world directions d (..., 3), z up. `blur` (radians) widens every feature (for
    the image light's rough levels). With the look pack's photographed sky in the cache folder that sky is used,
    turned so that its sun stands at the site's sun azimuth; else the made sky below."""
    if _hdri_mips() is not None:
        return _hdri_radiance(d, P, blur, sun_disk)
    up = np.clip(d[..., 2], -1.0, 1.0)
    el = np.arcsin(up)
    sun = np.asarray(P["sun_to"], float)
    cosg = np.clip((d * sun).sum(-1), -1.0, 1.0)
    gam = np.arccos(cosg)
    # base gradient: dusty warm horizon to a hazy blue zenith
    t = np.clip(el / (math.pi / 2), 0.0, 1.0) ** 0.55
    hz, zn = np.asarray(P["horizon"]), np.asarray(P["zenith"])
    sky = hz[None] * (1 - t[..., None]) + zn[None] * t[..., None]
    sky = sky.reshape(d.shape)
    # the sun's glow (forward scattering in dust), widened by the blur
    s1, s2 = math.hypot(0.09, blur), math.hypot(0.45, blur)
    glow = P["glow1"] * np.exp(-(gam / s1) ** 2) + P["glow2"] * np.exp(-(gam / s2) ** 2)
    sky = sky + glow[..., None] * np.asarray(P["sun_rgb"])
    # clouds on a dome: broken cumulus lit from the sun's side, fading into the haze low down
    k = np.maximum(up, 0.0) + 0.12
    cu, cv = d[..., 0] / k, d[..., 1] / k
    oct_ = 6 if blur < 0.02 else (4 if blur < 0.1 else (2 if blur < 0.3 else 1))
    cl = _noise2(cu * 0.9 + 7.0, cv * 0.9 + 3.0, P["seed"], octaves=oct_)
    cover = np.clip((cl - P["cloud_thr"]) / 0.22, 0.0, 1.0) * np.clip(up * 5.0, 0.0, 1.0)
    lit = 0.55 + 0.45 * np.clip(cosg, 0, 1)
    cloud_rgb = np.asarray(P["cloud"])[None] * lit.reshape(-1, 1)
    cloud_rgb = cloud_rgb.reshape(d.shape)
    sky = sky * (1 - cover[..., None] * 0.85) + cloud_rgb * cover[..., None] * 0.85
    # smoke columns rising from fires in the city
    az = np.arctan2(d[..., 1], d[..., 0])
    for (a0, w0, lean, dark) in P["plumes"]:
        h = np.clip(el, 0.0, 1.2)
        center = a0 + lean * h
        width = w0 * (0.5 + 2.8 * h) + blur
        da = np.angle(np.exp(1j * (az - center)))
        prof = np.exp(-(da / width) ** 2) * np.clip(1.0 - h / 0.85, 0.0, 1.0) ** 0.7 * np.clip(el * 20 + 0.5, 0, 1)
        n = _noise2(da * 6.0 + 11.0, el * 9.0 + a0 * 3, P["seed"] + 17, octaves=max(1, oct_ - 1))
        dens = np.clip(prof * (0.75 + 1.0 * n), 0.0, 0.95)
        smoke = np.asarray([0.26, 0.24, 0.23]) * dark
        sky = sky * (1 - dens[..., None]) + smoke[None] * dens[..., None]
    # below the horizon: the city's ground and walls in the haze
    gnd = np.asarray(P["ground"])
    below = np.clip(-el / 0.06 + 0.15, 0.0, 1.0)
    sky = sky * (1 - below[..., None]) + gnd[None] * below[..., None]
    if sun_disk:
        disk = np.clip((math.radians(0.55) - gam) / math.radians(0.12), 0.0, 1.0)
        sky = sky + disk[..., None] * np.asarray(P["sun_rgb"]) * P["disk"]
    return np.maximum(sky, 0.0)


def render_sun(env):
    """The render's sun (elevation, azimuth in degrees): the plan's sun lowered to a late-afternoon light and turned to
    rake along the street. Only the colour images use it: the physics model's sun is not involved, and the thermal
    model reads the sun's strength only."""
    el = 18.0 + 0.45 * (float(env["sun_elev"]) - 25.0)
    a = float(env["sun_az"])
    az = math.copysign(65.0 + 0.4 * abs(a), a if a != 0 else 1.0)
    return el, az


def _params(env, seed):
    el_d, az_d = render_sun(env)
    el, az = math.radians(el_d), math.radians(az_d)
    sun_to = (math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el))   # towards the sun
    haze = float(env.get("haze", 0.5))
    rng = np.random.default_rng(seed)
    low = max(0.0, 1.0 - el_d / 45.0)                           # a low sun: a warmer, more orange light
    sun_rgb = np.array([1.0, 0.86 - 0.12 * low, 0.70 - 0.22 * low])
    plumes = []
    for _ in range(int(rng.integers(2, 4))):
        plumes.append((float(rng.uniform(-math.pi, math.pi)), float(rng.uniform(0.04, 0.09)),
                       float(rng.uniform(-0.6, 0.6)), float(rng.uniform(0.6, 1.0))))
    return {"sun_to": sun_to, "seed": int(seed) % (2 ** 31),
            "horizon": [0.95 + 0.2 * haze, 0.84 + 0.12 * haze, 0.70 + 0.04 * haze],
            "zenith": [0.24 + 0.16 * haze, 0.36 + 0.14 * haze, 0.62 + 0.06 * haze],
            "glow1": 2.2, "glow2": 0.55 + 0.4 * haze, "sun_rgb": sun_rgb.tolist(), "disk": 600.0,
            "cloud": [1.05, 1.00, 0.95], "cloud_thr": 0.46 + 0.08 * (1 - haze), "plumes": plumes,
            "ground": [0.20, 0.19, 0.17], "sun_color": (sun_rgb / sun_rgb.max()).tolist(), "low": low}


def make_sky(env, seed, ibl_size=128, sky_size=512, sun_lux=95000.0, ibl_intensity=26000.0):
    """The look's sky: {"ibl_ktx", "skybox_ktx", "sun_dir" (the light's travel direction), "sun_color", "sun_lux",
    "ibl_intensity"}."""
    P = _params(env, seed)
    levels = []
    n = ibl_size
    nlev = int(math.log2(ibl_size)) + 1
    for i in range(nlev):
        d, _ = cube_dirs(n)
        t = i / max(1, nlev - 1)
        pr = 1.0 - math.sqrt(max(0.0, 1.0 - t))                # perceptual roughness of this level
        blur = max(2.0 / n, (pr * pr) * 1.1)
        levels.append(radiance(_to_world(d).reshape(-1, 3), P, blur=blur).reshape(6, n, n, 3).astype(np.float32))
        n = max(1, n // 2)
    d, dw = cube_dirs(32)
    rad = radiance(_to_world(d).reshape(-1, 3), P, blur=0.15).reshape(6, 32, 32, 3)
    sh = irradiance_sh(rad, dw, d)
    ibl = write_ktx(levels, sh)
    d, _ = cube_dirs(sky_size)
    box = radiance(_to_world(d).reshape(-1, 3), P, blur=0.0, sun_disk=True).reshape(6, sky_size, sky_size, 3)
    box = (box * SKYBOX_GAIN).astype(np.float32)
    skybox = write_ktx([box])
    k = 1.0 - 0.35 * P["low"]
    return {"ibl_ktx": ibl, "skybox_ktx": skybox, "sun_dir": [-v for v in P["sun_to"]], "sun_color": P["sun_color"],
            "sun_lux": float(sun_lux) * k, "ibl_intensity": float(ibl_intensity), "params": P}
