"""Scanning-sonar pings -> world-anchored horizontal maps, plume-section components, structure masking from the
manifold general-arrangement drawing (numpy + scipy only).  Port of proto/g3_plume/detect.py (sonar part, sha256
5d88279a...) with the 8-bit scale, the drawing and the sea-bed height made explicit.

A ping is a dict: t, angle (head angle, rad, the device convention: bearing = -angle in the head frame), data
(uint8 samples), range_m, gain (index), o (head position, world / L), R (head frame -> world / L).  A program builds
o / R from its navigation pose and the manual's sonar mount; the grader-side checks from the true pose.

    g = Grid(cx, cy, half, res);  S, N = accumulate(g, pings, codec);  M = smooth_map(g, S, N)
    valid = ~structure_mask(g, ga, o)
    best, comps, bg_db = single_sweep_best(g, M, valid)                  # one 360 deg scan
    best, comps, info = persistent_section(g, sweep_maps(g, sweeps), valid, prior_xy=..., prior_r=...)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi

TAN_EDGE = math.tan(math.radians(20.0))      # lowest rays of the sampled vertical fan


@dataclass(frozen=True)
class SonarCodec:
    """8-bit sample -> dB: value 0 at floor_db (normal gain), 255 at floor_db + dyn_db, each gain index step shifts
    the floor by gain_step_db.  Only dyn_db matters for the relative thresholds used here (the floor cancels)."""
    floor_db: float = -88.0
    dyn_db: float = 60.0
    gain_step_db: float = 6.0

    def to_db(self, code, gain):
        floor = self.floor_db - (np.asarray(gain, float) - 1) * self.gain_step_db
        return floor + np.asarray(code, float) / 255.0 * self.dyn_db


CODEC = SonarCodec()


# ----------------------------------------------------------------------------------------------------- the drawing
@dataclass
class ManifoldGA:
    """The manifold general-arrangement drawing + its as-built pose: structure frame S (origin on the mudmat top at
    the structure centre, x along lx, z up) at origin_w with heading psi; frame footprint lx x ly; roof (grating top)
    roof_z above the mudmat top; mudmat thickness; the six slot axes (k = 2 * column + row, branch = k + 1) and the
    two flanges of each branch (S, horizontal); the leak (flange) height range above the mudmat; the sea-bed height
    z_bed_w (world) -- a program measures it (depth + altitude)."""
    origin_w: np.ndarray
    psi: float
    lx: float = 12.0
    ly: float = 8.0
    roof_z: float = 3.4
    mudmat_thk: float = 0.3
    slots: np.ndarray = None
    flanges: np.ndarray = None
    src_z_range: tuple = (1.0, 1.6)
    z_bed_w: float = None
    pitch: float = 4.0

    def __post_init__(self):
        self.origin_w = np.asarray(self.origin_w, float).reshape(3)
        c, s = math.cos(self.psi), math.sin(self.psi)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        if self.slots is None:
            p = self.pitch
            self.slots = np.array([(cx * p, ry * p / 2.0) for cx in (-1, 0, 1) for ry in (-1, 1)], float)
        self.slots = np.asarray(self.slots, float)
        if self.flanges is None:
            self.flanges = np.array([[s_ + np.array([0.0, -math.copysign(off, s_[1])]) for off in (0.30, 0.55)]
                                     for s_ in self.slots])
        self.flanges = np.asarray(self.flanges, float)[..., :2]
        if self.z_bed_w is None:
            self.z_bed_w = float(self.origin_w[2] - self.mudmat_thk)

    @classmethod
    def from_site(cls, site: dict, manifold: dict) -> "ManifoldGA":
        """From a site package's site.json + manifold.json (the as-built pose and the type drawing)."""
        st = manifold["structure"]
        fl = np.array([[f[:2] for f in fs] for fs in manifold["flanges"]], float)
        sl = np.array([s["xy"] for s in manifold["slots"]], float)
        zr = manifold.get("params", {}).get("source_height_m", [1.0, 1.6])
        return cls(origin_w=site["pose_o"], psi=float(site["pose_psi"]), lx=float(st["lx"]), ly=float(st["ly"]),
                   roof_z=float(st["roof_z"]), mudmat_thk=float(st["mudmat_thk"]), slots=sl, flanges=fl,
                   src_z_range=(float(zr[0]), float(zr[1])), z_bed_w=-float(site["seabed_depth_m"]),
                   pitch=float(manifold.get("params", {}).get("slot_pitch", 4.0)))

    @property
    def roof_w(self) -> float:
        return float(self.origin_w[2] + self.roof_z)

    def to_w(self, p_s):
        p = np.asarray(p_s, float)
        if p.shape[-1] == 2:
            p = np.concatenate([p, np.zeros(p.shape[:-1] + (1,))], -1)
        return p @ self.R.T + self.origin_w

    def to_s(self, p_w):
        return (np.asarray(p_w, float) - self.origin_w) @ self.R

    def vec_to_w(self, v_s):
        v = np.asarray(v_s, float)
        if v.shape[-1] == 2:
            v = np.concatenate([v, np.zeros(v.shape[:-1] + (1,))], -1)
        return v @ self.R.T

    def vec_to_s(self, v_w):
        return np.asarray(v_w, float) @ self.R

    def nearest_slot(self, p_s_xy) -> int:
        d = ((self.slots - np.asarray(p_s_xy, float)[None, :2]) ** 2).sum(1)
        return int(np.argmin(d))


# ----------------------------------------------------------------------------------------------------- maps
class Grid:
    def __init__(self, cx, cy, half, res=0.1):
        self.res = float(res)
        self.nx = self.ny = int(math.ceil(2 * half / res))
        self.x0 = cx - half
        self.y0 = cy - half

    @property
    def shape(self):
        return (self.ny, self.nx)

    def idx(self, x, y):
        ix = np.floor((x - self.x0) / self.res).astype(np.int64)
        iy = np.floor((y - self.y0) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return np.where(ok, iy * self.nx + ix, -1)

    def centres(self):
        x = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        y = self.y0 + (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(x, y)


def accumulate(grid: Grid, pings, codec: SonarCodec = CODEC, rmin=0.8, stride=1):
    """Sum of linear intensity and sample count per cell (horizontal projection of the slant range)."""
    n = grid.nx * grid.ny
    S = np.zeros(n)
    N = np.zeros(n)
    if not pings:
        return S, N
    data = np.array([p["data"][::stride] for p in pings], float)
    ns = pings[0]["data"].shape[0]
    rng_m = np.array([p["range_m"] for p in pings])
    r = (np.arange(ns)[::stride] + 0.5)[None, :] * (rng_m / ns)[:, None]
    dirs = np.array([np.asarray(p["R"]) @ np.array([math.cos(-p["angle"]), math.sin(-p["angle"]), 0.0]) for p in pings])
    dxy = dirs[:, :2] / np.maximum(np.linalg.norm(dirs[:, :2], axis=1, keepdims=True), 1e-9)
    o = np.array([np.asarray(p["o"])[:2] for p in pings])
    x = o[:, 0:1] + r * dxy[:, 0:1]
    y = o[:, 1:2] + r * dxy[:, 1:2]
    gain = np.array([p["gain"] for p in pings])[:, None]
    lin = 10.0 ** (codec.to_db(data, gain) / 10.0)
    k = grid.idx(x.ravel(), y.ravel())
    ok = (k >= 0) & (np.broadcast_to(r, x.shape).ravel() > rmin)
    S += np.bincount(k[ok], lin.ravel()[ok], minlength=n)
    N += np.bincount(k[ok], minlength=n)
    return S, N


def smooth_map(grid: Grid, S, N, sigma_cells=1.5, min_cov=0.3):
    Ss = ndi.gaussian_filter(S.reshape(grid.shape), sigma_cells, mode="constant")
    Ns = ndi.gaussian_filter(N.reshape(grid.shape), sigma_cells, mode="constant")
    return np.where(Ns > min_cov, Ss / np.maximum(Ns, 1e-12), np.nan)


def structure_mask(grid: Grid, ga: ManifoldGA, sonar_xyz, margin=0.5):
    """True where the lower edge of the fan may already reach the roof / frame (inside the footprint) or the bed."""
    X, Y = grid.centres()
    q = ga.to_s(np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], 1))
    inside = (np.abs(q[:, 0]) < ga.lx / 2 + margin) & (np.abs(q[:, 1]) < ga.ly / 2 + margin)
    d = np.hypot(X.ravel() - sonar_xyz[0], Y.ravel() - sonar_xyz[1])
    hr = sonar_xyz[2] - ga.roof_w
    hb = sonar_xyz[2] - ga.z_bed_w
    m = (inside & (d > hr / TAN_EDGE - 0.3)) | (d > hb / TAN_EDGE - 0.3) | (hr < 0.3) & inside
    return m.reshape(grid.shape)


def components(grid: Grid, M, valid, thr, min_area_m2=0.04, bg=None):
    """Connected components of (M > thr) & valid; features in linear units."""
    above = (M > thr) & valid
    lab, n = ndi.label(above)
    X, Y = grid.centres()
    out = []
    cell = grid.res ** 2
    for j in range(1, n + 1):
        sel = lab == j
        area = sel.sum() * cell
        if area < min_area_m2:
            continue
        ex = np.maximum(M[sel] - (bg if bg is not None else thr), 0.0)
        wsum = ex.sum()
        cx = float((X[sel] * ex).sum() / max(wsum, 1e-30))
        cy = float((Y[sel] * ex).sum() / max(wsum, 1e-30))
        out.append(dict(label=j, area=float(area), energy=float(wsum * cell), peak_db=float(10 * np.log10(M[sel].max())),
                        mean_db=float(10 * np.log10(M[sel].mean())), cx=cx, cy=cy, mask=sel))
    return out


def sweep_maps(grid: Grid, sweeps, codec: SonarCodec = CODEC, sigma_cells=1.5):
    maps = []
    for sw in sweeps:
        S, N = accumulate(grid, sw, codec)
        maps.append(smooth_map(grid, S, N, sigma_cells))
    return maps


def persistent_section(grid: Grid, maps, valid, delta_db=6.0, frac_min=0.5, prior_xy=None, prior_r=None,
                       min_area_m2=0.1):
    """Average and persistence over sweeps; the plume = the persistent component with the largest integrated
    excess (optionally within prior_r of prior_xy).  Background per sweep = median over valid covered cells."""
    st = np.stack(maps)
    cov = np.isfinite(st)
    vmask = valid[None] & cov
    bgs = np.array([np.nanmedian(np.where(vm, m, np.nan)) for m, vm in zip(st, vmask)])
    above = np.where(vmask, st > bgs[:, None, None] * 10 ** (delta_db / 10.0), False)
    nvis = vmask.sum(0)
    frac = np.where(nvis > 0, above.sum(0) / np.maximum(nvis, 1), 0.0)
    mean = np.where(nvis > 0, np.nansum(np.where(vmask, st, 0.0), 0) / np.maximum(nvis, 1), np.nan)
    bg = float(np.nanmedian(bgs))
    cand = (frac >= frac_min) & (nvis >= max(1, len(maps) // 2))
    comps = components(grid, np.where(cand, mean, 0.0), cand & valid, thr=bg, min_area_m2=min_area_m2, bg=bg)
    if prior_xy is not None and prior_r is not None:
        comps = [c for c in comps if math.hypot(c["cx"] - prior_xy[0], c["cy"] - prior_xy[1]) <= prior_r]
    comps.sort(key=lambda c: -c["energy"])
    return (comps[0] if comps else None), comps, dict(bg_db=10 * math.log10(bg), frac=frac, mean=mean)


def single_sweep_best(grid: Grid, M, valid, delta_db=6.0, min_area_m2=0.1):
    """One sweep: components above median + delta; the plume = the one with the largest integrated excess."""
    bg = float(np.nanmedian(np.where(valid & np.isfinite(M), M, np.nan)))
    comps = components(grid, np.nan_to_num(M, nan=0.0), valid & np.isfinite(M), thr=bg * 10 ** (delta_db / 10.0),
                       min_area_m2=min_area_m2, bg=bg)
    comps.sort(key=lambda c: -c["energy"])
    return (comps[0] if comps else None), comps, 10 * math.log10(bg)
