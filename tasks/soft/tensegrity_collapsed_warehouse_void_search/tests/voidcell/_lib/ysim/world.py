"""Parametric, seeded blockout worlds around the ysim robot (ysim 0.2.0).

A world is a list of small feature RECORDS (``Feature``: name, type, art kind, pose, size, material, friction,
priority, colour, group, meta). Records are plain data (JSON round trip, ``World.records()`` /
``World.from_records()``), exported as static MuJoCo geoms (``World.to_mjcf()`` -> world_xml / asset_xml for
``model.ModelConfig``). Art replaces a record's *visual* with an asset of the same bounding box later (the art
class is ``kind``: wall | rubble | support | roof | crate | step | ramp | bump | patch | terrain | marker); the
collision primitive, its pose, its friction and therefore the physics stay as recorded.

Builders (each returns a list of records; composite builders also write the clear opening into ``meta``):
    box / box_xyz           static box (axis-aligned by extents, or centre + half sizes + rotation)
    portal                  a wall (or slab stood on edge) with a portal of given clear width and height
    tunnel                  a slab on two supports (clear width, clear height, length; flat or lean roof)
    step                    a raised block (step up from the floor; step down / drop when leaving it)
    ramp                    an inclined slab from the floor up to a given rise, optional plateau
    friction_patch          a thin floor patch whose friction overrides the robot's (priority 1)
    bumps                   seeded fixed small bumps (buried sphere caps) for rough ground
    heightfield_patch       seeded band-limited relief layer (MuJoCo hfield) with its own friction
World helpers: ``plot_top`` (top-view figure for documents), ``segment_hits`` (which records a straight tendon
passes through; used to detect lines cutting obstacles), ``from_concept07`` (a site JSON record of an earlier family).

Physics conventions: every world geom has priority 1, so its own friction is the contact friction against the
robot (MuJoCo otherwise takes the max of the pair; the robot's geoms carry the floor default 0.8). Contact
solref / condim are the model defaults (params: Rule S). World geoms never collide with each other (static).
Materials table: ``MATERIALS`` (friction provenance per material). Frame: x east, y north, z up, metres.
Nothing of the robot is built here.
"""
import hashlib
import json
import math
import os
from dataclasses import dataclass, field, asdict

import numpy as np

from . import paths

# ----------------------------------------------------------------------------- materials
# friction: contact friction against the robot. Provenance per value. rgba: blockout colour only (art replaces it).
MATERIALS = {
    "concrete": {"rgba": (0.62, 0.62, 0.60, 1.0), "friction": 0.8,
                 "source": "params 'friction' 0.8 (authors' MJCF default geom friction; RA-L sysID 0.76-0.92 on their "
                           "lab floor) - used as the floor and generic rubble value; ASSUMPTION that concrete ~ their floor"},
    "wall": {"rgba": (0.40, 0.40, 0.42, 1.0), "friction": 0.8, "source": "as concrete"},
    "rubble": {"rgba": (0.62, 0.62, 0.60, 1.0), "friction": 0.8, "source": "as concrete (the site family's material rule)"},
    "support": {"rgba": (0.56, 0.56, 0.55, 1.0), "friction": 0.8, "source": "as concrete (the site family's material rule)"},
    "slab": {"rgba": (0.70, 0.69, 0.66, 1.0), "friction": 0.8, "source": "as concrete (the site family's material rule)"},
    "crate_red": {"rgba": (0.78, 0.16, 0.14, 1.0), "friction": 0.8, "source": "as concrete; colour of the site family's crates"},
    "crate_blue": {"rgba": (0.14, 0.30, 0.74, 1.0), "friction": 0.8, "source": "as concrete; colour of the site family's crates"},
    "dust": {"rgba": (0.66, 0.63, 0.56, 1.0), "friction": 0.35,
             "source": "ASSUMPTION of the site family's rule (dust film, mu 0.35); no measurement"},
    "gravel": {"rgba": (0.45, 0.40, 0.33, 1.0), "friction": 0.60,
               "source": "ASSUMPTION of the site family's rule (gravel patch, mu 0.60, relief rms 8 mm); no measurement"},
    "wood": {"rgba": (0.66, 0.52, 0.36, 1.0), "friction": 0.8,
             "source": "RA-L: 8 deg wooden incline (platform not stated); friction as the floor (ASSUMPTION)"},
    "marker": {"rgba": (0.10, 0.69, 0.48, 1.0), "friction": 0.0, "source": "no collision (group 3)"},
}
KINDS = ("wall", "rubble", "support", "roof", "crate", "step", "ramp", "bump", "patch", "terrain", "marker")


def _rot(euler_deg):
    """Rotation matrix of intrinsic x-y-z Euler angles in degrees (MuJoCo's default eulerseq 'xyz')."""
    a, b, c = (math.radians(v) for v in euler_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
    Rz = np.array([[math.cos(c), -math.sin(c), 0], [math.sin(c), math.cos(c), 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def _quat(R):
    """(w, x, y, z) of a rotation matrix (Shepperd)."""
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q) * (1 if q[0] >= 0 else -1)


# ----------------------------------------------------------------------------- the record
@dataclass
class Feature:
    """One static primitive. type: box (size = half extents) | sphere (size = (r,)) | hfield (size = MuJoCo hfield
    size (rx, ry, ztop, zbase); the elevation grid is regenerated from meta['hfield'] parameters and its seed)."""
    name: str
    type: str
    kind: str
    pos: tuple
    size: tuple
    euler_deg: tuple = (0.0, 0.0, 0.0)
    material: str = "concrete"
    friction: float = None          # None -> MATERIALS[material]["friction"]
    collide: bool = True
    group: int = 0                  # 0 = visible + collides; 3 = marker (no collision)
    rgba: tuple = None              # None -> MATERIALS[material]["rgba"] (blockout colour)
    meta: dict = field(default_factory=dict)

    def mu(self):
        return MATERIALS[self.material]["friction"] if self.friction is None else float(self.friction)

    def colour(self):
        return tuple(self.rgba) if self.rgba is not None else MATERIALS[self.material]["rgba"]

    def R(self):
        return _rot(self.euler_deg)

    def to_dict(self):
        d = asdict(self)
        d["pos"], d["size"], d["euler_deg"] = list(self.pos), list(self.size), list(self.euler_deg)
        if d["rgba"] is not None:
            d["rgba"] = list(d["rgba"])
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        for k in ("pos", "size", "euler_deg"):
            d[k] = tuple(d[k])
        if d.get("rgba") is not None:
            d["rgba"] = tuple(d["rgba"])
        return Feature(**d)

    def aabb(self):
        """Axis-aligned bounds (lo, hi) in world (boxes and spheres exactly; hfield = its patch box)."""
        p = np.asarray(self.pos, float)
        if self.type == "sphere":
            r = self.size[0]
            return p - r, p + r
        if self.type == "hfield":
            rx, ry, zt, zb = self.size
            return p + np.array([-rx, -ry, -zb]), p + np.array([rx, ry, zt])
        h = np.asarray(self.size, float)
        R = self.R()
        ext = np.abs(R) @ h
        return p - ext, p + ext

    def footprint(self):
        """Top-view outline (closed polygon, 4 corners for boxes / hfields, None for spheres)."""
        p = np.asarray(self.pos, float)
        if self.type == "sphere":
            return None
        if self.type == "hfield":
            rx, ry = self.size[0], self.size[1]
            return np.array([[p[0] - rx, p[1] - ry], [p[0] + rx, p[1] - ry], [p[0] + rx, p[1] + ry], [p[0] - rx, p[1] + ry]])
        hx, hy, hz = self.size
        R = self.R()
        C = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) @ R.T + p
        xy = C[:, :2]
        # convex hull of the projected corners (monotone chain)
        pts = sorted(set(map(tuple, np.round(xy, 9))))
        if len(pts) < 3:
            return np.array(pts)

        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lo, hi = [], []
        for q in pts:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], q) <= 0:
                lo.pop()
            lo.append(q)
        for q in reversed(pts):
            while len(hi) >= 2 and cross(hi[-2], hi[-1], q) <= 0:
                hi.pop()
            hi.append(q)
        return np.array(lo[:-1] + hi[:-1])

    def top_z(self):
        return float(self.aabb()[1][2])


# ----------------------------------------------------------------------------- builders
def box(name, x0, x1, y0, y1, z0, z1, kind="rubble", material=None, friction=None, euler_deg=(0, 0, 0), rgba=None,
        meta=None, group=0):
    """Box given by its unrotated extents; euler_deg rotates it about its centre."""
    material = material or {"wall": "wall", "support": "support", "roof": "slab", "crate": "concrete"}.get(kind, "concrete")
    return Feature(name, "box", kind, (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.5 * (z0 + z1)),
                   (0.5 * abs(x1 - x0), 0.5 * abs(y1 - y0), 0.5 * abs(z1 - z0)), tuple(euler_deg), material, friction,
                   group == 0, group, rgba, dict(meta or {}))


def box_xyz(name, centre, half, yaw_deg=0.0, kind="rubble", material="concrete", friction=None, euler_deg=None, meta=None):
    e = tuple(euler_deg) if euler_deg is not None else (0.0, 0.0, float(yaw_deg))
    return Feature(name, "box", kind, tuple(map(float, centre)), tuple(map(float, half)), e, material, friction, True, 0,
                   None, dict(meta or {}))


def _local(cx, cy, yaw_deg):
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    return lambda u, v: (cx + c * u - s * v, cy + s * u + c * v)


def portal(name, centre_xy, yaw_deg, wall_length, thickness, height, clear_w, clear_h=None, kind="wall",
           material="wall", friction=None):
    """A wall with one portal. The wall runs along local x (length wall_length), its thickness along local y (the
    passage axis); the portal (clear width clear_w, clear height clear_h; None or >= height = open to the top) is
    centred. yaw_deg turns the whole wall about z. Returns jambs (+ lintel)."""
    f = _local(centre_xy[0], centre_xy[1], yaw_deg)
    side = 0.5 * (wall_length - clear_w)
    assert side > 0, "wall shorter than the portal"
    out = []
    meta = {"composite": name, "role": "jamb", "clear_w": clear_w, "clear_h": clear_h if (clear_h and clear_h < height) else None,
            "passage_axis_deg": yaw_deg + 90.0, "depth": thickness}
    for k, sgn in (("w", -1), ("e", 1)):
        u = sgn * (0.5 * clear_w + 0.5 * side)
        out.append(box_xyz("%s_jamb_%s" % (name, k), (*f(u, 0.0), 0.5 * height), (0.5 * side, 0.5 * thickness, 0.5 * height),
                           yaw_deg, kind, material, friction, meta=dict(meta)))
    if clear_h is not None and clear_h < height:
        lh = height - clear_h
        out.append(box_xyz("%s_lintel" % name, (*f(0.0, 0.0), clear_h + 0.5 * lh), (0.5 * clear_w + 1e-4, 0.5 * thickness, 0.5 * lh),
                           yaw_deg, "roof" if kind == "wall" else kind, "slab" if material == "wall" else material, friction,
                           meta=dict(meta, role="lintel")))
    return out


def tunnel(name, centre_xy, yaw_deg, clear_w, clear_h, length, support_w=0.18, roof_t=0.10, overhang=0.04,
           lean_rise=0.0, kind_support="support", kind_roof="roof", friction=None):
    """A slab on two supports (the site family's covered passage 'tunnel_P'): passage along local y (length), clear width clear_w
    between the supports, clear height clear_h under the slab (at the low side for a lean roof: the east support
    is lean_rise higher and the slab is tilted about the passage axis). Slab thickness roof_t, overhang front/back."""
    f = _local(centre_xy[0], centre_xy[1], yaw_deg)
    meta = {"composite": name, "clear_w": clear_w, "clear_h": clear_h, "length": length, "passage_axis_deg": yaw_deg + 90.0,
            "lean_rise": lean_rise}
    out = []
    for k, sgn, rise in (("w", -1, 0.0), ("e", 1, lean_rise)):
        u = sgn * (0.5 * clear_w + 0.5 * support_w)
        h = clear_h + rise
        out.append(box_xyz("%s_sup_%s" % (name, k), (*f(u, 0.0), 0.5 * h), (0.5 * support_w, 0.5 * length, 0.5 * h), yaw_deg,
                           kind_support, "support", friction, meta=dict(meta, role="support")))
    span = clear_w + 2 * support_w
    tilt = math.degrees(math.atan2(lean_rise, span)) if lean_rise else 0.0
    # slab rests on both supports: centre height = clear_h + rise/2 + roof_t/2 (the site family's roof_over rule)
    out.append(box_xyz("%s_roof" % name, (*f(0.0, 0.0), clear_h + 0.5 * lean_rise + 0.5 * roof_t),
                       (0.5 * span, 0.5 * length + overhang, 0.5 * roof_t), kind=kind_roof, material="slab", friction=friction,
                       euler_deg=(0.0, -tilt, yaw_deg) if tilt else (0.0, 0.0, yaw_deg), meta=dict(meta, role="roof")))
    return out


def step(name, x0, x1, y0, y1, height, material="concrete", friction=None):
    """A raised block from the floor (z 0..height): a step up when rolled onto, a step down / drop when left."""
    return [box(name, x0, x1, y0, y1, 0.0, height, "step", material, friction, meta={"height": height})]


def ramp(name, x0, x1, y_start, run, angle_deg, thickness=0.05, plateau=0.0, material="wood", friction=None):
    """An inclined slab rising toward +y from the floor line y = y_start for horizontal run `run` at angle_deg,
    then an optional plateau of depth `plateau` at the top height run * tan(angle)."""
    th = math.radians(angle_deg)
    L = run / math.cos(th)                       # slab length along the slope
    hl, ht = 0.5 * L, 0.5 * thickness
    cy = y_start + hl * math.cos(th) + ht * math.sin(th)
    cz = hl * math.sin(th) - ht * math.cos(th)
    rise = run * math.tan(th)
    out = [Feature(name, "box", "ramp", (0.5 * (x0 + x1), cy, cz), (0.5 * (x1 - x0), hl, ht), (angle_deg, 0.0, 0.0),
                   material, friction, True, 0, None, {"angle_deg": angle_deg, "rise": rise, "y_start": y_start, "run": run})]
    if plateau > 0:
        out += [box(name + "_plateau", x0, x1, y_start + run, y_start + run + plateau, 0.0, rise, "step", material, friction,
                    meta={"height": rise})]
    return out


def friction_patch(name, x0, x1, y0, y1, material="dust", friction=None, thickness=0.002):
    """A thin floor patch (default 2 mm) whose friction is the contact friction against the robot (priority 1)."""
    return [box(name, x0, x1, y0, y1, 0.0, thickness, "patch", material, friction, meta={"thickness": thickness})]


def bumps(name, x0, x1, y0, y1, seed, density=20.0, h_range=(0.005, 0.015), r_range=(0.02, 0.04), material="rubble",
          friction=None, min_gap=0.0):
    """Seeded fixed small bumps for rough ground: buried spheres whose caps stand h above the floor
    (count = density per m^2; caps do not overlap when min_gap >= 0)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    n = int(round(density * (x1 - x0) * (y1 - y0)))
    out, placed = [], []
    tries = 0
    while len(out) < n and tries < 50 * max(n, 1):
        tries += 1
        r = rng.uniform(*r_range)
        h = rng.uniform(*h_range)
        a = math.sqrt(max(h * (2 * r - h), 0.0))          # cap footprint radius
        x, y = rng.uniform(x0 + a, x1 - a), rng.uniform(y0 + a, y1 - a)
        if any(math.hypot(x - px, y - py) < a + pa + min_gap for px, py, pa in placed):
            continue
        placed.append((x, y, a))
        out.append(Feature("%s_%d" % (name, len(out)), "sphere", "bump", (x, y, h - r), (r,), (0, 0, 0), material, friction,
                           True, 0, None, {"cap_h": h, "seed": seed}))
    return out


def relief_grid(nx, ny, cell, seed, rms, peak, wl_min, wl_max, base, edge):
    """Elevation grid E[iy, ix] (m, >= 0) of a relief layer: band-limited Gaussian noise (wavelengths wl_min..wl_max,
    isotropic FFT band-pass of white noise from PCG64(seed)), scaled to std `rms`, clipped to [-base, peak], lifted
    by `base` (mean thickness), tapered linearly to 0 over `edge` at the patch border."""
    rng = np.random.Generator(np.random.PCG64(seed))
    w = rng.standard_normal((ny, nx))
    F = np.fft.rfft2(w)
    ky = np.fft.fftfreq(ny, d=cell)[:, None]
    kx = np.fft.rfftfreq(nx, d=cell)[None, :]
    k = np.sqrt(kx ** 2 + ky ** 2)
    band = (k >= 1.0 / wl_max) & (k <= 1.0 / wl_min)
    n = np.fft.irfft2(F * band, s=(ny, nx))
    n = (n - n.mean()) / max(n.std(), 1e-12)
    rel = np.clip(n * rms, -base, peak)
    E = base + rel
    yy, xx = np.meshgrid(np.arange(ny) * cell, np.arange(nx) * cell, indexing="ij")
    Lx, Ly = (nx - 1) * cell, (ny - 1) * cell
    d = np.minimum.reduce([xx, Lx - xx, yy, Ly - yy])
    taper = np.clip(d / max(edge, 1e-9), 0.0, 1.0)
    return E * taper


def heightfield_patch(name, x0, x1, y0, y1, seed, rms=0.008, peak=0.025, wl=(0.03, 0.08), base=None, edge=0.05,
                      cell=0.01, material="gravel", friction=None, zbase=0.005):
    """Seeded relief layer (the site family's gravel patch) as a MuJoCo hfield on the floor: relief std `rms` about a
    mean thickness `base` (default 2.5 rms), peaks <= `peak` above it, band-limited to wavelengths wl, tapered to the
    floor over `edge`. ASSUMPTION: there is no measurement of real debris relief."""
    base = 2.5 * rms if base is None else base
    nx = int(round((x1 - x0) / cell)) + 1
    ny = int(round((y1 - y0) / cell)) + 1
    par = {"nx": nx, "ny": ny, "cell": cell, "seed": int(seed), "rms": rms, "peak": peak, "wl_min": wl[0], "wl_max": wl[1],
           "base": base, "edge": edge}
    E = relief_grid(nx, ny, cell, seed, rms, peak, wl[0], wl[1], base, edge)
    ztop = float(E.max())
    meta = {"hfield": par, "ztop": ztop, "relief_std_m": float(E[E > 0].std()), "mean_thickness_m": float(E.mean()),
            "sha256": hashlib.sha256(np.round(E, 9).tobytes()).hexdigest()}
    f = Feature(name, "hfield", "terrain", (0.5 * (x0 + x1), 0.5 * (y0 + y1), 0.0),
                (0.5 * (nx - 1) * cell, 0.5 * (ny - 1) * cell, max(ztop, 1e-4), zbase), (0, 0, 0), material, friction, True, 0,
                None, meta)
    return [f]


def hfield_grid(f):
    """Regenerate the elevation grid E[iy, ix] of an hfield record from its parameters (deterministic)."""
    p = f.meta["hfield"]
    return relief_grid(p["nx"], p["ny"], p["cell"], p["seed"], p["rms"], p["peak"], p["wl_min"], p["wl_max"], p["base"], p["edge"])


# ----------------------------------------------------------------------------- the world
class World:
    def __init__(self, name="world", seed=None, bounds=None, meta=None):
        self.name = name
        self.seed = seed
        self.bounds = bounds              # (x0, x1, y0, y1) of the site, for plots
        self.features = []
        self.meta = dict(meta or {})

    def add(self, *items):
        for it in items:
            if isinstance(it, Feature):
                self.features.append(it)
            else:
                self.features.extend(it)
        names = [f.name for f in self.features]
        assert len(names) == len(set(names)), "duplicate feature names"
        return self

    # ---- records
    def records(self):
        return {"world": self.name, "seed": self.seed, "bounds": self.bounds, "meta": self.meta,
                "features": [f.to_dict() for f in self.features], "materials": {k: dict(v) for k, v in MATERIALS.items()}}

    def to_json(self, path):
        with open(path, "w") as fh:
            json.dump(self.records(), fh, indent=1)

    @staticmethod
    def from_records(rec):
        w = World(rec["world"], rec.get("seed"), rec.get("bounds"), rec.get("meta"))
        w.add([Feature.from_dict(d) for d in rec["features"]])
        return w

    def digest(self):
        """SHA-256 of the canonical records (feature order kept)."""
        s = json.dumps([f.to_dict() for f in self.features], sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()

    # ---- MJCF
    def to_mjcf(self):
        """(world_xml, asset_xml) strings for model.ModelConfig(world_xml=..., asset_xml=...)."""
        W, A = [], []
        for f in self.features:
            mu = f.mu()
            rgba = "%.4g %.4g %.4g %.4g" % f.colour()
            col = 'contype="1" conaffinity="1" priority="1"' if (f.collide and f.group == 0) else 'contype="0" conaffinity="0"'
            q = _quat(f.R())
            if f.type == "box":
                W.append('<geom name="w_%s" type="box" pos="%.9g %.9g %.9g" quat="%.12g %.12g %.12g %.12g" size="%.9g %.9g %.9g" '
                         'friction="%.6g 0.005 0.0001" rgba="%s" group="%d" %s/>'
                         % ((f.name,) + tuple(f.pos) + tuple(q) + tuple(f.size) + (mu, rgba, f.group, col)))
            elif f.type == "sphere":
                W.append('<geom name="w_%s" type="sphere" pos="%.9g %.9g %.9g" size="%.9g" friction="%.6g 0.005 0.0001" '
                         'rgba="%s" group="%d" %s/>' % ((f.name,) + tuple(f.pos) + (f.size[0], mu, rgba, f.group, col)))
            elif f.type == "hfield":
                E = hfield_grid(f)
                zt = f.size[2]
                En = np.clip(E / zt, 0.0, 1.0)
                # MJCF elevation rows run from +y to -y (image order): flip the y-increasing grid
                el = " ".join("%.6g" % v for v in np.flipud(En).ravel())
                ny, nx = En.shape
                A.append('<hfield name="hf_%s" nrow="%d" ncol="%d" size="%.9g %.9g %.9g %.9g" elevation="%s"/>'
                         % (f.name, ny, nx, f.size[0], f.size[1], f.size[2], f.size[3], el))
                W.append('<geom name="w_%s" type="hfield" hfield="hf_%s" pos="%.9g %.9g %.9g" friction="%.6g 0.005 0.0001" '
                         'rgba="%s" group="%d" %s/>' % ((f.name, f.name) + tuple(f.pos) + (mu, rgba, f.group, col)))
            else:
                raise ValueError(f.type)
        return "\n".join(W), "\n".join(A)

    def model_config(self, **kw):
        from . import model as mdl
        wx, ax = self.to_mjcf()
        kw["world_xml"] = kw.get("world_xml", "") + wx
        kw["asset_xml"] = kw.get("asset_xml", "") + ax
        return mdl.ModelConfig(**kw)

    def geom_names(self):
        return ["w_" + f.name for f in self.features]

    # ---- geometry queries
    def segment_hits(self, P, Q, margin=0.0, kinds=None):
        """Names of collision records that the straight segments P[i]-Q[i] (n x 3) pass through (boxes: slab test on
        the oriented box shrunk by `margin`; spheres: distance < r - margin; hfields: ignored). Returns a list of
        (segment index, feature name)."""
        P = np.atleast_2d(np.asarray(P, float))
        Q = np.atleast_2d(np.asarray(Q, float))
        hits = []
        for f in self.features:
            if not f.collide or f.group != 0 or f.type == "hfield" or (kinds and f.kind not in kinds):
                continue
            c = np.asarray(f.pos, float)
            if f.type == "sphere":
                r = f.size[0] - margin
                d = Q - P
                t = np.clip(np.einsum("ij,ij->i", c - P, d) / np.maximum(np.einsum("ij,ij->i", d, d), 1e-18), 0, 1)
                dist = np.linalg.norm(P + t[:, None] * d - c, axis=1)
                hits += [(int(i), f.name) for i in np.nonzero(dist < r)[0]]
                continue
            R = f.R()
            h = np.asarray(f.size, float) - margin
            p = (P - c) @ R
            q = (Q - c) @ R
            d = q - p
            with np.errstate(divide="ignore", invalid="ignore"):
                t1 = (-h - p) / d
                t2 = (h - p) / d
            tmin = np.where(np.abs(d) < 1e-15, np.where(np.abs(p) <= h, -np.inf, np.inf), np.minimum(t1, t2))
            tmax = np.where(np.abs(d) < 1e-15, np.where(np.abs(p) <= h, np.inf, -np.inf), np.maximum(t1, t2))
            lo = np.maximum(tmin.max(axis=1), 0.0)
            hi = np.minimum(tmax.min(axis=1), 1.0)
            hits += [(int(i), f.name) for i in np.nonzero(lo < hi)[0]]
        return hits

    def segment_penetration(self, P, Q, n=41):
        """Largest depth (m) by which the straight segments P[i]-Q[i] reach inside any collision box (sampled at n
        points per segment; depth = distance from the sample to the nearest box face, 0 outside)."""
        P = np.atleast_2d(np.asarray(P, float))
        Q = np.atleast_2d(np.asarray(Q, float))
        t = np.linspace(0.0, 1.0, n)[None, :, None]
        X = (P[:, None, :] + t * (Q - P)[:, None, :]).reshape(-1, 3)
        best = 0.0
        for f in self.features:
            if not f.collide or f.group != 0 or f.type != "box":
                continue
            loc = (X - np.asarray(f.pos, float)) @ f.R()
            d = np.min(np.asarray(f.size, float) - np.abs(loc), axis=1)
            if d.size:
                best = max(best, float(d.max()))
        return best

    def surface_z(self, x, y):
        """Highest collision surface below the open sky at (x, y) (boxes / spheres / hfields; floor = 0)."""
        import mujoco
        z = 0.0
        for f in self.features:
            if not f.collide or f.group != 0:
                continue
            lo, hi = f.aabb()
            if not (lo[0] <= x <= hi[0] and lo[1] <= y <= hi[1]):
                continue
            if f.type == "hfield":
                E = hfield_grid(f)
                c = f.meta["hfield"]["cell"]
                ix = int(round((x - (f.pos[0] - f.size[0])) / c))
                iy = int(round((y - (f.pos[1] - f.size[1])) / c))
                if 0 <= iy < E.shape[0] and 0 <= ix < E.shape[1]:
                    z = max(z, float(E[iy, ix]))
                continue
            z = max(z, float(hi[2]))
        return z

    # ---- top-view figure
    def plot_top(self, path, robot_xy=None, paths_xy=None, title=None, annotate=True, extra=None, dpi=150):
        """Top-view figure of the records (colour by art kind, roofs hatched, patches hatched, openings labelled
        with their clear width / height). robot_xy: list of (6x3 node arrays) to draw struts; paths_xy: list of
        (N x 2) arrays."""
        os.environ.setdefault("MPLCONFIGDIR", os.path.join(paths.tmp_dir(), "mplconfig"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Circle
        fig, ax = plt.subplots(figsize=(8.0, 6.6))
        order = {"patch": 0, "terrain": 0, "bump": 1, "step": 1, "ramp": 1, "rubble": 2, "wall": 2, "support": 3, "crate": 3,
                 "roof": 4, "marker": 5}
        for f in sorted(self.features, key=lambda f: order.get(f.kind, 2)):
            col = f.colour()[:3]
            if f.type == "sphere":
                a = math.sqrt(max(f.meta.get("cap_h", f.size[0]) * (2 * f.size[0] - f.meta.get("cap_h", f.size[0])), 0))
                ax.add_patch(Circle(f.pos[:2], a, facecolor=col, edgecolor="0.25", lw=0.3, zorder=2))
                continue
            poly = f.footprint()
            if f.kind == "marker":
                ax.add_patch(Polygon(poly, closed=True, fill=False, edgecolor="0.35", lw=0.6, ls="--", zorder=5))
            elif f.kind == "roof":
                ax.add_patch(Polygon(poly, closed=True, facecolor=(*col, 0.35), edgecolor="0.15", lw=0.9, hatch="xx", zorder=4))
            elif f.kind in ("patch", "terrain"):
                ax.add_patch(Polygon(poly, closed=True, facecolor=(*col, 0.45), edgecolor="0.3", lw=0.6,
                                     hatch="...." if f.kind == "terrain" else "////", zorder=1))
                if annotate:
                    ax.text(poly[:, 0].mean(), poly[:, 1].mean(), "%s mu %.2f" % (f.material, f.mu()), fontsize=6.5, ha="center",
                            va="center", zorder=6)
            elif f.kind == "ramp":
                ax.add_patch(Polygon(poly, closed=True, facecolor=(*col, 0.8), edgecolor="0.2", lw=0.6, zorder=2))
                if annotate:
                    ax.text(poly[:, 0].mean(), poly[:, 1].mean(), "ramp %.0f deg" % f.meta.get("angle_deg", 0), fontsize=6.5,
                            ha="center", zorder=6)
            else:
                ax.add_patch(Polygon(poly, closed=True, facecolor=col, edgecolor="0.2", lw=0.6, zorder=order.get(f.kind, 2)))
                if annotate and f.kind == "step":
                    ax.text(poly[:, 0].mean(), poly[:, 1].mean(), "h %.0f mm" % (1000 * f.top_z()), fontsize=6.5, ha="center",
                            zorder=6)
        if annotate:
            done = set()
            for f in self.features:
                comp = f.meta.get("composite")
                if not comp or comp in done:
                    continue
                done.add(comp)
                parts = [g for g in self.features if g.meta.get("composite") == comp]
                c = np.mean([g.pos[:2] for g in parts], axis=0)
                cw, ch = f.meta.get("clear_w"), f.meta.get("clear_h")
                txt = "%s\nw %.2f" % (comp, cw) + ("  h %.2f" % ch if ch else "")
                ax.text(c[0], c[1], txt, fontsize=6.5, ha="center", va="center", zorder=7,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="0.5", lw=0.4, alpha=0.85))
        for P in (paths_xy or []):
            P = np.asarray(P)
            ax.plot(P[:, 0], P[:, 1], "-", color=(0.85, 0.35, 0.05), lw=1.4, zorder=8)
        for N in (robot_xy or []):
            N = np.asarray(N)
            for k, cc in enumerate(((0.8, 0.1, 0.1), (0.1, 0.62, 0.2), (0.12, 0.25, 0.85))):
                ax.plot(N[[2 * k, 2 * k + 1], 0], N[[2 * k, 2 * k + 1], 1], "-", color=cc, lw=2.0, zorder=9)
        if extra:
            extra(ax)
        if self.bounds:
            x0, x1, y0, y1 = self.bounds
            ax.set_xlim(x0 - 0.25, x1 + 0.25)
            ax.set_ylim(y0 - 0.25, y1 + 0.25)
        else:
            ax.autoscale()
        ax.set_aspect("equal")
        ax.set_xlabel("x east [m]")
        ax.set_ylabel("y north [m]")
        ax.grid(True, lw=0.3, color="0.85", zorder=0)
        ax.set_title(title or self.name, fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        return path


# ----------------------------------------------------------------------------- earlier site-record import
_C07_MATERIAL = {"wall": "wall", "rubble": "rubble", "support": "support", "roof": "slab", "crate": "concrete"}


def from_concept07(site, gravel_seed=None):
    """Build a World from a site record of an earlier family (dict or path of a
    site JSON file). Every box keeps its
    name, art kind and pose; red / blue crates keep their colour; the dust patch becomes a friction patch (mu 0.35),
    the gravel patch a relief hfield (rms 8 mm, mu 0.60; seeded by the site seed unless gravel_seed is given).
    The camera, pad, search points and chamber go to World.meta."""
    if isinstance(site, str):
        with open(site) as fh:
            site = json.load(fh)
    w = World("concept07_%s_seed%d" % (site["family"], site["seed"]), site["seed"],
              (0.0, site["size"]["Lx"], 0.0, site["size"]["Ly"]))
    for b in site["boxes"]:
        mat = _C07_MATERIAL[b["kind"]]
        if b["kind"] == "crate" and b.get("rgba"):
            mat = "crate_red" if b["rgba"][0] > b["rgba"][2] else "crate_blue"
        f = box(b["name"], b["x0"], b["x1"], b["y0"], b["y1"], b["z0"], b["z1"], b["kind"], mat,
                euler_deg=tuple(b["euler"]) if b.get("euler") else (0, 0, 0), group=b.get("group", 0),
                meta={"source": "concept07 box"})
        w.add(f)
    for p in site["patches"]:
        if p["kind"] == "dust":
            w.add(friction_patch("patch_" + p["name"], p["x0"], p["x1"], p["y0"], p["y1"], "dust", p.get("mu")))
        elif p["kind"] == "gravel":
            w.add(heightfield_patch("patch_" + p["name"], p["x0"], p["x1"], p["y0"], p["y1"],
                                    seed=site["seed"] if gravel_seed is None else gravel_seed,
                                    rms=p.get("relief_rms_m", 0.008), material="gravel", friction=p.get("mu")))
    for g in site.get("gates", []):
        comp = "%s %s" % (g["id"], g["type"])
        hit = False
        for f in w.features:
            if f.name in ("sup_w_" + g["id"], "sup_e_" + g["id"], "roof_" + g["id"]) or (
                    g.get("id_role") == "void mouth" and f.name in ("roof_mouth",)):
                f.meta.update({"composite": comp, "clear_w": round(g["x1"] - g["x0"], 3), "clear_h": g.get("clear_h")})
                hit = True
        if not hit:      # open gap: a no-collision marker carries the opening for plots and records
            w.add(box("gate_" + g["id"], g["x0"], g["x1"], g["y0"], g["y1"], 0.0, 0.001, "marker", "marker", group=3,
                      meta={"composite": comp, "clear_w": round(g["x1"] - g["x0"], 3), "clear_h": None}))
    w.meta.update({k: site[k] for k in ("pad", "search_points", "chamber", "camera", "camera_model", "gates", "bands")
                   if k in site})
    return w
