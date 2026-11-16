"""Burner visuals of the kitchen task (PR #81): the single source of the gas flames and of the radiant glass-ceramic
zones for both renderers of the project (the Filament agent cameras, kitchen_render.py, and the path-traced USD demo).

Pure numpy, no renderer imports, no random numbers: every output is a deterministic function of the arguments.

What the simulator shows (appliance_env.Unit.update, read only): a burner's flame site u_burner_on_<loc> is visible
(alpha > 0) while the burner is on; its radius is FLAME_SCALE[0] ... FLAME_SCALE[1] (0.6 ... 1.4) times its base
radius from the lowest to the highest setting; its colour is RoboCasa's (gas (0.3, 0.4, 1.0), electric (0.5, 0, 0)).
From that:

    kind  = burner_kind(site_rgb)                        "gas" | "radiant"
    level = burner_level(radius, base_radius)            0 (just on) ... 1 (highest)
    key   = burner_key(loc)                              a stable integer per burner (its name)

Frames and units: every mesh is in the burner frame = the flame site's frame (origin at the site's centre, z along the
site's axis, i.e. up, x / y as the site's), metres.  A mesh is a dict part -> (V float (n, 3), N float (n, 3),
UV float (n, 2), I uint32 (m,) triangles, counter-clockwise seen from outside).  Times are simulator seconds.
Colours are linear RGB unless named *_srgb.  Textures are uint8 arrays whose row i holds v = (i + 0.5) / height and
column j u = (j + 0.5) / width (row 0 is v = 0: Filament's convention for uploaded rows; a PNG written from the array
shows v = 0 at the top, so a renderer with v = 0 at the image's bottom, USD's, flips it or uses 1 - v).  Emission
textures are sRGB-encoded (decode to linear); alpha textures are linear.

GAS (gas_mesh): a ring of separate flame jets around the burner head (the head's rim is at about the site's base
radius on RoboCasa's burners): per jet an opaque bright inner cone ("gas_inner") and a longer translucent emissive
sheath ("gas_sheath", blue, violet towards the tip), leaning outwards; 16 ... 36 jets by radius; a second inner ring
on large burners (base radius >= INNER_RING_MIN_R); a faint blue glow over the head and around the jets'
roots ("gas_haze").  Length and lean grow with the level; each jet flickers (length and lean, FLICKER_AMP percent)
as a smooth function of (t, key, jet), periodic with FLICKER_PERIOD, so that a renderer may precompute
FLICKER_PHASES variants (t = k * FLICKER_STEP) and switch them per frame (gas_phase(t)).

RADIANT (radiant_mesh, radiant_texture): a disk at the glass top carrying the heating zone's glow (coil pattern, soft
halo fading into the glass) as an emission texture and an alpha texture; the disk's radius is RADIANT_HALO times the
site's current radius (the zone follows the site); the colour goes from dull dark red (level 0) to bright orange-red
(level 1).  glass_coat_mesh: a clear glossy layer over the whole glass top, drawn after the glow, so that the room's
reflections lie over the glow (the glow reads as under the glass).

materials() is the material table (per part: shading, colours, emission, alpha, which textures); textures() returns
the textures it names.  python kitchen_fx.py <dir> writes preview OBJ / PNG / JSON files for checking another
renderer against this one.
"""
import json
import math
import os
import sys

import numpy as np

FLAME_SCALE = (0.6, 1.4)          # appliance_env.FLAME_SCALE

# -- gas -------------------------------------------------------------------------------------------------------------
JETS_MIN, JETS_MAX = 16, 36       # jets of the outer ring, by base radius:
JETS_R = (0.025, 0.070)           # JETS_MIN at or below JETS_R[0] m, JETS_MAX at or above JETS_R[1] m
PORT_R = 0.93                     # outer ring of ports: radius / base radius (just under the head's rim)
PORT_Z = 0.3                      # port height: fraction of the site's half-height above the site's centre
INNER_RING_MIN_R = 1.0            # m: burners at least this large get an inner ring (dual crown); off: RoboCasa's
                                  # large caps are mostly solid (u6), and the site does not say which have one
INNER_RING_R = 0.45               # its radius / base radius
INNER_RING_FRACTION = 0.5         # its jets / the outer ring's
INNER_RING_LENGTH = 0.55          # its jet length / the outer ring's
JET_LENGTH = (0.20, 0.60)         # jet length / base radius at level 0 ... 1
JET_LEAN_DEG = (36.0, 56.0)       # jet axis from the vertical (outwards) at the root, level 0 ... 1
JET_BEND_DEG = 22.0               # the axis turns up by this much from root to tip (buoyancy)
JET_WIDTH = (0.0014, 0.0030)      # m, sheath half-width at its widest, smallest ... largest burner
JET_WIDTH_LEVEL = (0.9, 1.55)     # width factor at level 0 ... 1
INNER_LENGTH = 0.38               # inner cone length / jet length
FLICKER_PERIOD = 1.6              # s: the flicker repeats exactly after this
FLICKER_STEP = 0.1                # s: the control step (10 Hz cameras); FLICKER_PHASES = PERIOD / STEP variants
FLICKER_PHASES = 16
FLICKER_HARMONICS = (2, 3, 5, 7)  # cycles per period: 1.25, 1.875, 3.125, 4.375 Hz
FLICKER_AMP = (0.03, 0.13)        # relative length flicker at level 0 ... 1 (lean flickers by 30 % of it, in rad)

# rings of the sheath (fraction of the length) and its half-width profile; the inner cone's
SHEATH_V = np.array([0.0, 0.10, 0.30, 0.55, 0.78, 0.92, 1.0])
SHEATH_W = np.array([0.55, 0.95, 1.10, 1.00, 0.78, 0.48, 0.0])
INNER_V = np.array([0.0, 0.45, 0.80, 1.0])
INNER_W = np.array([0.62, 0.55, 0.32, 0.0])
SHEATH_SIDES, INNER_SIDES = 6, 6
HAZE_SEG = 48

# -- radiant ---------------------------------------------------------------------------------------------------------
RADIANT_HALO = 1.30               # the glow disk's radius / the zone's (site's current) radius
RADIANT_LEVELS = 6                # textures / materials precomputed for levels 0, 0.2, ..., 1 (radiant_bucket)
RADIANT_TURNS = 5.0               # turns of each arm of the double spiral across the zone's radius
RADIANT_LIFT = 0.0004             # m above the site's centre plane (the glass top)
COAT_LIFT = 0.0008                # m: the glass coat, above the glow
RADIANT_SEG = 64

# What the Filament agent cameras show (sRGB, close-ups at 960 x 720, filmic tone mapping, exposure_scale 50), for a
# renderer that matches its emission scale by eye: gas inner cones ~ (120, 165, 245), sheaths ~ (80, 80, 170) over the
# dark hob, the haze a blue tint ~ (90, 100, 180) at the port ring; radiant on black glass (u3): low ~ (184, 12, 9),
# high ~ (241, 92, 23), brightest coil ~ (235, 129, 84); on p0's brown-grey glass low ~ (130, 53, 42).
MATERIALS = {
    # Filament path in kitchen_render: "unlit" = MuJoCo decor material (unlit, opaque), colour may exceed 1 (HDR);
    # "emissive_translucent" = PBR transparent, base colour black, alpha from the alpha texture, emission = emission
    # texture x emission x alpha;  "clear_coat" = PBR transparent, alpha ~ 0, glossy (reflections only).
    "gas_inner": {"shading": "unlit", "color": [0.36, 0.95, 2.9], "alpha": 1.0,
                  "note": "the bright inner cone: cyan-white; opaque, unlit (an emitter); HDR colour (> 1)"},
    "gas_sheath": {"shading": "emissive_translucent", "emission": 0.30, "texture": "gas_sheath",
                   "roughness": 1.0,
                   "note": "outer flame: blue near the root, violet towards the tip; alpha falls to 0 at the tip"},
    "gas_haze": {"shading": "emissive_translucent", "emission": 0.60, "texture": "gas_haze", "roughness": 1.0,
                 "note": "faint blue glow over the head and around the jets' roots"},
    "radiant_glow": {"shading": "emissive_translucent", "emission": 0.42, "texture": "radiant_<bucket>",
                     "roughness": 1.0,
                     "note": "the heating zone seen through dark glass; one texture / material per level bucket"},
    "glass_coat": {"shading": "clear_coat", "color": [0.0, 0.0, 0.0], "alpha": 0.30, "roughness": 0.12,
                   "metallic": 0.0, "note": "dark glossy glass over the whole top (alpha 0.3 darkens what is under it, "
                                            "the glow included): the room's reflections lie over the glow"},
}


# ---------------------------------------------------------------------------------------------------------------------
# What the simulator shows
# ---------------------------------------------------------------------------------------------------------------------
def burner_kind(site_rgb):
    """'gas' when the site's blue exceeds its red (RoboCasa gas burners: (0.3, 0.4, 1.0)), else 'radiant' (electric
    glass-ceramic zones: (0.5, 0, 0))."""
    r, _g, b = (float(x) for x in site_rgb[:3])
    return "gas" if b > r else "radiant"


def burner_level(radius, base_radius):
    """The setting from the site's current radius: (radius / base - 0.6) / 0.8, clipped to 0 ... 1."""
    s = float(radius) / max(float(base_radius), 1e-9)
    return float(min(max((s - FLAME_SCALE[0]) / (FLAME_SCALE[1] - FLAME_SCALE[0]), 0.0), 1.0))


def burner_key(name):
    """A stable 32-bit integer of a burner's name (FNV-1a; Python's hash() is salted per process)."""
    h = 2166136261
    for ch in str(name).encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def _hash01(*ints):
    """Deterministic uniform numbers in [0, 1) from integers (vectorised, splitmix-style)."""
    x = np.uint64(0x9E3779B97F4A7C15)
    acc = np.zeros(np.broadcast(*[np.asarray(i) for i in ints]).shape, np.uint64)
    with np.errstate(over="ignore"):
        for i in ints:
            acc = (acc ^ np.asarray(i, np.uint64)) * np.uint64(0xBF58476D1CE4E5B9) + x
            acc ^= acc >> np.uint64(31)
            acc *= np.uint64(0x94D049BB133111EB)
            acc ^= acc >> np.uint64(29)
    return (acc >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def flicker(t, key, jet, channel=0):
    """Smooth flicker in about -1 ... 1 of jet `jet` of burner `key` at time t (s); channel 0 = length, 1 = lean.
    A sum of FLICKER_HARMONICS sines (whole cycles per FLICKER_PERIOD) with phases from (key, jet, channel, k)."""
    t = np.asarray(t, float)
    jet = np.asarray(jet)
    out = 0.0
    amps = (0.55, 0.45, 0.30, 0.20)
    for k, (m, a) in enumerate(zip(FLICKER_HARMONICS, amps)):
        ph = 2 * math.pi * _hash01(key, jet, channel, k)
        out = out + a * np.sin(2 * math.pi * m * t / FLICKER_PERIOD + ph)
    return out / 1.1


def gas_phase(t):
    """The precomputed flicker variant for simulator time t: round(t / FLICKER_STEP) mod FLICKER_PHASES."""
    return int(math.floor(float(t) / FLICKER_STEP + 0.5)) % FLICKER_PHASES


def gas_phase_time(k):
    return k * FLICKER_STEP


def gas_bucket(level, n=5):
    """Level bucket for precomputed gas meshes (renderer cache): round(level * (n - 1))."""
    return int(math.floor(float(level) * (n - 1) + 0.5))


def radiant_bucket(level):
    return int(math.floor(float(level) * (RADIANT_LEVELS - 1) + 0.5))


# ---------------------------------------------------------------------------------------------------------------------
# Gas
# ---------------------------------------------------------------------------------------------------------------------
def gas_layout(base_radius, half_height):
    """The burner's rings: list of dicts (n jets, port radius, port height, length factor, width, angle offset)."""
    r0 = float(base_radius)
    f = min(max((r0 - JETS_R[0]) / (JETS_R[1] - JETS_R[0]), 0.0), 1.0)
    n = int(round(JETS_MIN + f * (JETS_MAX - JETS_MIN)))
    width = JET_WIDTH[0] + f * (JET_WIDTH[1] - JET_WIDTH[0])
    zp = PORT_Z * float(half_height)
    rings = [{"n": n, "r": PORT_R * r0, "z": zp, "length": 1.0, "width": width, "offset": 0.0, "ring": 0}]
    if r0 >= INNER_RING_MIN_R:
        rings.append({"n": max(8, int(round(n * INNER_RING_FRACTION))), "r": INNER_RING_R * r0, "z": zp + 0.5 * float(half_height),
                      "length": INNER_RING_LENGTH, "width": 0.8 * width, "offset": 0.5, "ring": 1})
    return rings


def _tube(C, T, B1, B2, W, Vs, sides, aspect):
    """Rings of a tapered tube. C, T, B1, B2: (J, R, 3) centre, tangent, cross-section axes per jet and ring;
    W: (J, R) half-width; Vs: (R,) v coordinate.  Returns V, N, UV, I of all J jets (last ring = tip point ok)."""
    J, R = W.shape
    a = 2 * math.pi * np.arange(sides) / sides
    ca, sa = np.cos(a), np.sin(a)
    # (J, R, S, 3)
    dirs = ca[None, None, :, None] * B1[:, :, None, :] + sa[None, None, :, None] * B2[:, :, None, :]
    P = C[:, :, None, :] + (W[:, :, None, None] * (ca[None, None, :, None] * B1[:, :, None, :]
                                                    + aspect * sa[None, None, :, None] * B2[:, :, None, :]))
    N = dirs
    uv = np.stack(np.broadcast_arrays((np.arange(sides) / sides)[None, :], Vs[:, None]), -1)   # (R, S, 2)
    UV = np.broadcast_to(uv[None], (J, R, sides, 2))
    s0 = np.arange(sides)
    s1 = (s0 + 1) % sides
    tri = []
    for r in range(R - 1):
        a0, a1 = r * sides + s0, r * sides + s1
        b0, b1 = (r + 1) * sides + s0, (r + 1) * sides + s1
        tri.append(np.stack([a0, a1, b1], -1))
        tri.append(np.stack([a0, b1, b0], -1))
    tri = np.concatenate(tri, 0)                                        # (F, 3) per jet
    I = (tri[None, :, :] + (np.arange(J) * R * sides)[:, None, None]).reshape(-1)
    return P.reshape(-1, 3), N.reshape(-1, 3), UV.reshape(-1, 2), I.astype(np.uint32)


def _fix_winding(V, N, I):
    """Make triangles counter-clockwise seen from outside (along the vertex normals).  Every mesh here is built with
    one consistent orientation, so the first non-degenerate triangles decide for all of them."""
    t = I.reshape(-1, 3)
    k = t[:64]
    g = np.cross(V[k[:, 1]] - V[k[:, 0]], V[k[:, 2]] - V[k[:, 0]])
    d = np.einsum("ij,ij->i", g, N[k[:, 0]] + N[k[:, 1]] + N[k[:, 2]])
    if np.sum(d[np.abs(d) > 1e-18] < 0) > np.sum(d[np.abs(d) > 1e-18] > 0):
        t = t[:, ::-1]
    return np.ascontiguousarray(t).reshape(-1).astype(np.uint32)


def _jets(ring, r0, level, ts, key, steady=False):
    """Jets of one ring at the times ts (P,): arrays over P * n jets, time-major (jet k of time p at p * n + k)."""
    n = ring["n"]
    ts = np.atleast_1d(np.asarray(ts, float))
    P = len(ts)
    j = np.arange(n)
    phi = np.tile(2 * math.pi * (j + ring["offset"]) / n, P)
    amp = 0.0 if steady else FLICKER_AMP[0] + level * (FLICKER_AMP[1] - FLICKER_AMP[0])
    jid = j + 1000 * ring["ring"]
    fl_len = (1.0 + amp * flicker(ts[:, None], key, jid[None, :], 0)).reshape(-1)
    fl_lean = (0.3 * amp * flicker(ts[:, None], key, jid[None, :], 1)).reshape(-1)
    L = r0 * (JET_LENGTH[0] + level * (JET_LENGTH[1] - JET_LENGTH[0])) * ring["length"] * fl_len      # (P n,)
    lean = math.radians(JET_LEAN_DEG[0] + level * (JET_LEAN_DEG[1] - JET_LEAN_DEG[0])) + fl_lean        # (P n,)
    w = ring["width"] * (JET_WIDTH_LEVEL[0] + level * (JET_WIDTH_LEVEL[1] - JET_WIDTH_LEVEL[0]))
    er = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], -1)
    et = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], -1)
    ez = np.array([0.0, 0.0, 1.0])
    root = er * ring["r"] + ez * ring["z"]
    return n, P, L, lean, w, er, et, ez, root


def _centreline(Vs, L, lean, er, ez, root):
    """Centre, tangent at fractions Vs of the jet (J, R, 3): the axis turns up by JET_BEND_DEG from root to tip."""
    bend = math.radians(JET_BEND_DEG)
    # sample finely and integrate
    fine = np.linspace(0, 1, 33)
    th = lean[:, None] - bend * fine[None, :]                                   # (J, F)
    d = np.sin(th)[..., None] * er[:, None, :] + np.cos(th)[..., None] * ez[None, None, :]
    seg = (d[:, 1:] + d[:, :-1]) / 2 * (L[:, None, None] / (len(fine) - 1))
    pos = np.concatenate([np.zeros_like(seg[:, :1]), np.cumsum(seg, 1)], 1) + root[:, None, :]
    x = np.clip(np.asarray(Vs, float), 0.0, 1.0) * (len(fine) - 1)
    i0 = np.minimum(np.floor(x).astype(int), len(fine) - 2)
    fr = (x - i0)[None, :, None]
    C = pos[:, i0] * (1 - fr) + pos[:, i0 + 1] * fr
    th_v = lean[:, None] - bend * Vs[None, :]
    T = np.sin(th_v)[..., None] * er[:, None, :] + np.cos(th_v)[..., None] * ez[None, None, :]
    return C, T


def gas_meshes(base_radius, half_height, level, ts, key):
    """Meshes of a lit gas burner at each time of ts, in the burner frame: a list over ts of
    {"gas_inner", "gas_sheath", "gas_haze": (V, N, UV, I)}.  base_radius, half_height: the flame site's base size (m);
    level 0 ... 1 (burner_level); ts: simulator times (s); key: burner_key(loc).  The haze does not flicker."""
    r0 = float(base_radius)
    level = float(min(max(level, 0.0), 1.0))
    ts = np.atleast_1d(np.asarray(ts, float))
    per = [{"gas_inner": [], "gas_sheath": []} for _ in ts]
    extent = 0.0
    for ring in gas_layout(r0, half_height):
        n, P, L, lean, w, er, et, ez, root = _jets(ring, r0, level, ts, key)
        for part, Vs, Ws, sides, frac in (("gas_sheath", SHEATH_V, SHEATH_W, SHEATH_SIDES, 1.0),
                                          ("gas_inner", INNER_V, INNER_W, INNER_SIDES, INNER_LENGTH)):
            C, T = _centreline(Vs * frac, L, lean, er, ez, root)
            B1 = np.broadcast_to(et[:, None, :], C.shape)
            B2 = np.cross(T, B1)
            W = w * Ws[None, :] * np.ones((len(L), 1))
            V, N, UV, I = _tube(C, T, B1, B2, W, Vs, sides, 0.75)
            I = _fix_winding(V, N, I)
            nv, ni = len(V) // P, len(I) // P
            for k in range(P):
                per[k][part].append((V[k * nv:(k + 1) * nv], N[k * nv:(k + 1) * nv], UV[k * nv:(k + 1) * nv],
                                     I[k * ni:(k + 1) * ni] - np.uint32(k * nv)))
        if ring["ring"] == 0:
            _n, _P, L0, lean0, _w, er0, _et, ez0, root0 = _jets(ring, r0, level, [0.0], key, steady=True)
            C0, _T = _centreline(SHEATH_V, L0, lean0, er0, ez0, root0)
            extent = float(np.max(np.hypot(C0[..., 0], C0[..., 1])))
    haze = _haze(r0, half_height, extent)
    return [dict({part: _merge(lst) for part, lst in d.items()}, gas_haze=haze) for d in per]


def gas_mesh(base_radius, half_height, level, t, key):
    """gas_meshes at one time t (s)."""
    return gas_meshes(base_radius, half_height, level, [t], key)[0]


def _haze(r0, hl, extent):
    """A flat disk from the centre to beyond the flame ring at the head's top: u = 0 at the centre, 0.5 at the outer
    port ring, 1 at the outer edge (gas_haze texture profile along u)."""
    rp = PORT_R * r0
    r_out = max(extent * 1.02, rp * 1.15)
    radii = np.concatenate([np.linspace(0, rp, 4), np.linspace(rp, r_out, 4)[1:]])
    us = np.concatenate([np.linspace(0, 0.5, 4), np.linspace(0.5, 1.0, 4)[1:]])
    a = 2 * math.pi * np.arange(HAZE_SEG) / HAZE_SEG
    z = float(hl)
    V = np.concatenate([np.stack([rr * np.cos(a), rr * np.sin(a), np.full_like(a, z)], -1) for rr in radii])
    UV = np.concatenate([np.stack([np.full_like(a, u), np.full_like(a, 0.5)], -1) for u in us])
    N = np.tile([0.0, 0.0, 1.0], (len(V), 1))
    I = []
    S = HAZE_SEG
    for k in range(len(radii) - 1):
        for s in range(S):
            a0, a1 = k * S + s, k * S + (s + 1) % S
            b0, b1 = (k + 1) * S + s, (k + 1) * S + (s + 1) % S
            I += [a0, b0, b1, a0, b1, a1]
    I = np.array(I, np.uint32)
    return V, N, UV, _fix_winding(V, N, I)


def _merge(lst):
    V, N, UV, I = [], [], [], []
    n0 = 0
    for v, nn, uv, ii in lst:
        V.append(v); N.append(nn); UV.append(uv); I.append(ii + n0)
        n0 += len(v)
    return (np.concatenate(V).astype(np.float64), np.concatenate(N).astype(np.float64),
            np.concatenate(UV).astype(np.float64), np.concatenate(I).astype(np.uint32))


def gas_variants(base_radius, half_height, level, key):
    """All FLICKER_PHASES variants at one level: list over phases k of gas_mesh(..., t = k * FLICKER_STEP, ...)
    (the flicker is periodic: gas_mesh(..., t, ...) == variant gas_phase(t) whenever t is a multiple of the step)."""
    return gas_meshes(base_radius, half_height, level, [gas_phase_time(k) for k in range(FLICKER_PHASES)], key)


# ---------------------------------------------------------------------------------------------------------------------
# Radiant
# ---------------------------------------------------------------------------------------------------------------------
def radiant_mesh():
    """The glow disk of unit radius at z = 0 (burner frame, scale x / y by RADIANT_HALO * the site's current radius
    and lift by RADIANT_LIFT): {"radiant_glow": (V, N, UV, I)}; uv maps the disk onto the texture square."""
    S = RADIANT_SEG
    a = 2 * math.pi * np.arange(S) / S
    radii = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    V = np.concatenate([np.stack([r * np.cos(a), r * np.sin(a), np.zeros(S)], -1) for r in radii])
    UV = np.stack([(V[:, 0] + 1) / 2, (V[:, 1] + 1) / 2], -1)
    N = np.tile([0.0, 0.0, 1.0], (len(V), 1))
    I = []
    for k in range(len(radii) - 1):
        for s in range(S):
            a0, a1 = k * S + s, k * S + (s + 1) % S
            b0, b1 = (k + 1) * S + s, (k + 1) * S + (s + 1) % S
            I += [a0, b0, b1, a0, b1, a1]
    I = np.array(I, np.uint32)
    return {"radiant_glow": (V, N, UV, _fix_winding(V, N, I))}


def radiant_colour(level):
    """Linear RGB of the hottest coil at a level: dull dark red (0) ... bright orange-red (1)."""
    lo = np.array([0.32, 0.014, 0.006])
    hi = np.array([1.00, 0.085, 0.014])
    s = float(min(max(level, 0.0), 1.0))
    w = np.array([s ** 0.8, s ** 1.8, s ** 1.8])        # orange comes in late: medium is still red
    return lo + (hi - lo) * w


def radiant_texture(level, size=128):
    """{"emission": (size, size, 3) uint8 sRGB, "alpha": (size, size) uint8} over the glow disk's square
    (texel (i, j) <-> uv ((j + 0.5) / size, (i + 0.5) / size)); r = 1 at RADIANT_HALO, the zone's edge at
    1 / RADIANT_HALO.  The coil: a double spiral of soft bands, a darker centre; the halo fades into the glass."""
    c = (np.arange(size) + 0.5) / size * 2 - 1
    x, y = np.meshgrid(c, c)
    r = np.hypot(x, y) * RADIANT_HALO                   # 1 = the zone's edge
    th = np.arctan2(y, x)
    s = float(min(max(level, 0.0), 1.0))
    # double spiral: arm k at radius (theta / 2 pi + k / 2) / turns (mod 1 / turns)
    pitch = 1.0 / RADIANT_TURNS
    q = (r / pitch - th / (2 * math.pi)) * 2.0          # two arms: bands at integer q
    band = 0.66 + 0.34 * (0.5 + 0.5 * np.cos(2 * math.pi * q))   # soft bands: a smooth ripple, no aliasing
    inside = np.clip((0.97 - r) / 0.06, 0.0, 1.0) * np.clip((r - 0.10) / 0.10, 0.0, 1.0)
    # the element is hotter in the middle ring, cooler at the centre and the rim
    radial = 0.75 + 0.25 * np.cos((r - 0.55) * math.pi * 1.1)
    coil = band * inside * radial
    halo = np.exp(-np.maximum(r - 0.92, 0.0) / 0.12) * np.clip((1.30 - r) / 0.30, 0.0, 1.0)
    halo = halo * (1.0 - inside) * 0.45 + inside * 0.25
    e_lin = radiant_colour(s)[None, None, :] * (coil + halo)[..., None]
    # a dim zone glows deeper red (cooler), a bright one a little more orange in the bands' centres
    e_lin = np.clip(e_lin, 0.0, 1.0)
    alpha = np.clip(0.95 * coil + 0.5 * halo, 0.0, 1.0)
    alpha = np.maximum(alpha, 0.45 * inside)            # the glass over the zone is dark between the coils
    return {"emission": _srgb8(e_lin), "alpha": np.round(alpha * 255).astype(np.uint8)}


def glass_coat_mesh(lo_xy, hi_xy, z=COAT_LIFT):
    """A rectangle over the glass top (burner frame of any zone, or the unit's frame; lo_xy / hi_xy its corners)."""
    (x0, y0), (x1, y1) = lo_xy, hi_xy
    V = np.array([[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], float)
    N = np.tile([0.0, 0.0, 1.0], (4, 1))
    UV = np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]])
    I = np.array([0, 1, 2, 0, 2, 3], np.uint32)
    return {"glass_coat": (V, N, UV, _fix_winding(V, N, I))}


# ---------------------------------------------------------------------------------------------------------------------
# Textures and materials
# ---------------------------------------------------------------------------------------------------------------------
def _srgb8(lin):
    lin = np.clip(np.asarray(lin, float), 0.0, 1.0)
    s = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return np.round(s * 255).astype(np.uint8)


def _profile(xs, knots, values):
    return np.interp(xs, knots, values)


def gas_sheath_texture(size=64):
    """Along v (rows: row i <-> v = (i + 0.5) / size; constant along u): emission sRGB and alpha of the sheath."""
    v = (np.arange(size) + 0.5) / size
    col = np.stack([_profile(v, [0, 0.35, 0.7, 1.0], [0.05, 0.06, 0.12, 0.22]),
                    _profile(v, [0, 0.35, 0.7, 1.0], [0.30, 0.22, 0.14, 0.10]),
                    _profile(v, [0, 0.35, 0.7, 1.0], [1.00, 1.00, 1.00, 0.90])], -1)
    a = _profile(v, [0, 0.08, 0.3, 0.55, 0.8, 1.0], [0.50, 0.82, 0.78, 0.52, 0.20, 0.0])
    em = np.repeat(_srgb8(col)[:, None, :], 4, 1)
    al = np.repeat(np.round(a * 255).astype(np.uint8)[:, None], 4, 1)
    return {"emission": em, "alpha": al}


def gas_haze_texture(size=64):
    """Along u (columns; constant along v): 0 = the burner's centre, 0.5 = the outer port ring, 1 = the outer edge."""
    u = (np.arange(size) + 0.5) / size
    a = _profile(u, [0, 0.30, 0.42, 0.5, 0.60, 0.76, 1.0], [0.012, 0.015, 0.12, 0.24, 0.13, 0.03, 0.0])
    col = np.stack([np.full_like(u, 0.08), np.full_like(u, 0.22), np.full_like(u, 1.0)], -1)
    em = np.repeat(_srgb8(col)[None, :, :], 4, 0)
    al = np.repeat(np.round(a * 255).astype(np.uint8)[None, :], 4, 0)
    return {"emission": em, "alpha": al}


def texture_v(size):
    """v coordinate of texture rows (row i <-> v = (i + 0.5) / size): rows are stored in increasing v."""
    return (np.arange(size) + 0.5) / size


def textures():
    """Every texture materials() names: {name: {"emission": uint8 sRGB (h, w, 3), "alpha": uint8 (h, w)}}."""
    out = {"gas_sheath": gas_sheath_texture(), "gas_haze": gas_haze_texture()}
    for b in range(RADIANT_LEVELS):
        out["radiant_%d" % b] = radiant_texture(b / (RADIANT_LEVELS - 1))
    return out


def materials():
    return json.loads(json.dumps(MATERIALS))


# ---------------------------------------------------------------------------------------------------------------------
# Preview files for the other renderer
# ---------------------------------------------------------------------------------------------------------------------
def write_obj(path, parts):
    with open(path, "w") as fh:
        fh.write("# kitchen_fx preview: burner frame, metres, z up\n")
        n0 = 0
        for name, (V, N, UV, I) in sorted(parts.items()):
            fh.write("o %s\nusemtl %s\n" % (name, name))
            for p in V:
                fh.write("v %.6f %.6f %.6f\n" % tuple(p))
            for uv in UV:
                fh.write("vt %.5f %.5f\n" % tuple(uv))
            for nn in N:
                fh.write("vn %.5f %.5f %.5f\n" % tuple(nn))
            for t in I.reshape(-1, 3) + 1 + n0:
                fh.write("f %d/%d/%d %d/%d/%d %d/%d/%d\n" % (t[0], t[0], t[0], t[1], t[1], t[1], t[2], t[2], t[2]))
            n0 += len(V)


def _write_png(path, img):
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:            # numpy-only fallback: a binary PPM / PGM beside it
        with open(os.path.splitext(path)[0] + (".ppm" if img.ndim == 3 else ".pgm"), "wb") as fh:
            fh.write(b"P6\n" if img.ndim == 3 else b"P5\n")
            fh.write(b"%d %d\n255\n" % (img.shape[1], img.shape[0]))
            fh.write(np.ascontiguousarray(img).tobytes())


def preview(out_dir):
    """OBJ meshes of a large (0.068 m) and a small (0.030 m) gas burner at levels 0 / 0.5 / 1 and t = 0, 0.1;
    a radiant glow disk and a glass coat; PNG textures; a JSON of the parameters and the material table."""
    os.makedirs(out_dir, exist_ok=True)
    for r0, hl, name in ((0.0684, 0.0046, "gas_large"), (0.0301, 0.0025, "gas_small")):
        for lv in (0.0, 0.5, 1.0):
            for t in (0.0, 0.1):
                write_obj(os.path.join(out_dir, "%s_level%.1f_t%.1f.obj" % (name, lv, t)),
                          gas_mesh(r0, hl, lv, t, burner_key("left")))
    write_obj(os.path.join(out_dir, "radiant_disk_unit.obj"), radiant_mesh())
    write_obj(os.path.join(out_dir, "glass_coat_example.obj"), glass_coat_mesh((-0.3, -0.25), (0.3, 0.25)))
    for name, tex in textures().items():
        _write_png(os.path.join(out_dir, "tex_%s_emission.png" % name), tex["emission"])
        _write_png(os.path.join(out_dir, "tex_%s_alpha.png" % name), tex["alpha"])
    params = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple, str))}
    params.update({k: v.tolist() for k, v in globals().items() if k.isupper() and isinstance(v, np.ndarray)})
    with open(os.path.join(out_dir, "kitchen_fx_preview.json"), "w") as fh:
        json.dump({"materials": materials(), "parameters": params,
                   "flicker_examples": {"t": [round(k * 0.1, 1) for k in range(17)],
                                        "jet0_length_channel": [round(float(flicker(k * 0.1, burner_key("left"), 0, 0)), 6) for k in range(17)]},
                   "keys": {n: burner_key(n) for n in ("left", "center", "rear_left", "front_right")}}, fh, indent=1)


if __name__ == "__main__":
    preview(sys.argv[1] if len(sys.argv) > 1 else "fx_preview")
