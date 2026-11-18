"""Gamma dose-rate field, shielding and a dosimeter model (numpy only, independent of MuJoCo).

Field: H(p) = background + sum_j S_j * exp(-sum_k mu_k * L_k(p, s_j)) / max(|p - s_j|, r_min)^2, in uSv/h, where S_j is
a point source's dose rate at 1 m (unshielded) and L_k the length of the straight path source -> p inside shield box k
(axis-aligned or oriented boxes, slab method, exact). Line sources are split into `n_seg` equal point sources.
Narrow-beam attenuation without build-up (conservative for the shielding: real build-up raises the dose behind thick
shields by a factor of a few); `MU` gives linear attenuation coefficients at 662 keV (Cs-137).

Dosimeter: counts in a dwell time ~ Poisson(sensitivity * H * t) with sensitivity in counts per second per uSv/h
(e.g. ~3 cps per uSv/h for a small Geiger tube, ~50-500 for a CsI scintillator); optional dead time (non-paralysable
model, inverted by `estimate`).

Minimum-count rule (counting statistics only): the relative 1-sigma error of a reading of N counts is 1 / sqrt(N), so
a reading lies within a relative tolerance tol of the true rate with probability ~P (two-sided, normal approximation,
z = 1.96 for 95 %, 1.645 for 90 %) when N >= (z / tol)^2: tol 25 % -> 62 counts, tol 10 % -> 385 counts at 95 %
(271 at 90 %). The dwell time that gives N counts at a dose rate H is t = N / (sensitivity * H): for tol 10 % at
95 %, 50 cps per uSv/h needs 7.7 s at 1 uSv/h and 77 s at 0.1 uSv/h; a 3 cps Geiger tube 128 s at 1 uSv/h.
`counts_needed(tol, p)` and `dwell_needed(tol, rate, sensitivity, p)` compute these.

`Dosimeter.measure` returns only what the instrument shows (counts, rate estimate, its 1-sigma, dwell); the true dose
rate at the point is a separate grader-side value: `Dosimeter.measure_with_truth` returns (reading, true_rate).
`DoseAccumulator` integrates the robot's own dose along its path.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = ["MU", "PointSource", "LineSource", "ShieldBox", "RadiationField", "Dosimeter", "DoseAccumulator",
           "rasterise", "shields_from_boxes", "counts_needed", "dwell_needed"]

MU = {"concrete": 18.0, "steel": 57.0, "lead": 123.0, "water": 8.6, "air": 0.0}   # 1/m at 662 keV


def _z_two_sided(p: float) -> float:
    from statistics import NormalDist
    return float(NormalDist().inv_cdf(0.5 + 0.5 * float(p)))


def counts_needed(tol: float, p: float = 0.95) -> int:
    """Smallest count N with z_p / sqrt(N) <= tol (relative tolerance, two-sided probability p; normal approximation
    of the Poisson count, good from about 30 counts): N = ceil((z / tol)^2). tol 0.25 -> 62, tol 0.10 -> 385."""
    z = _z_two_sided(p)
    return int(math.ceil((z / float(tol)) ** 2 - 1e-9))


def dwell_needed(tol: float, rate_usvh: float, sensitivity_cps: float, p: float = 0.95) -> float:
    """Dwell time (s) for counts_needed(tol, p) at a dose rate (uSv/h) with a detector of `sensitivity_cps` counts per
    second per uSv/h (dead time ignored)."""
    return counts_needed(tol, p) / max(float(sensitivity_cps) * float(rate_usvh), 1e-12)


@dataclass
class PointSource:
    pos: tuple
    strength: float                  # uSv/h at 1 m
    name: str = "src"

    def points(self):
        return np.asarray([self.pos], float), np.asarray([self.strength], float)


@dataclass
class LineSource:
    p0: tuple
    p1: tuple
    strength_per_m: float            # uSv/h at 1 m from each metre of line (summed over the segments)
    name: str = "line"
    n_seg: int = 32

    def points(self):
        p0, p1 = np.asarray(self.p0, float), np.asarray(self.p1, float)
        L = float(np.linalg.norm(p1 - p0))
        t = (np.arange(self.n_seg) + 0.5) / self.n_seg
        return p0 + t[:, None] * (p1 - p0), np.full(self.n_seg, self.strength_per_m * L / self.n_seg)


@dataclass
class ShieldBox:
    center: tuple
    half: tuple
    mu: float = MU["concrete"]       # 1/m
    R: np.ndarray = field(default_factory=lambda: np.eye(3))   # box axes in the world (columns)
    name: str = "shield"


def shields_from_boxes(centers, halves, rotations, mu) -> list:
    """ShieldBox list from arrays (e.g. MuJoCo's geom_xpos, geom_size, geom_xmat of box geoms)."""
    return [ShieldBox(tuple(c), tuple(h), float(mu), np.asarray(R, float).reshape(3, 3))
            for c, h, R in zip(centers, halves, rotations)]


def _path_in_box(src: np.ndarray, pts: np.ndarray, box: ShieldBox) -> np.ndarray:
    """Length (m) of each segment src -> pts[i] inside the box (vectorised slab method)."""
    R = np.asarray(box.R, float)
    c = np.asarray(box.center, float)
    h = np.asarray(box.half, float)
    o = R.T @ (src - c)                         # (3,)
    v = (pts - src) @ R                         # (N, 3) segment vectors in box axes
    tmin = np.zeros(len(pts))
    tmax = np.ones(len(pts))
    for a in range(3):
        va = v[:, a]
        par = np.abs(va) < 1e-12
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h[a] - o[a]) / va
            t2 = (h[a] - o[a]) / va
        lo = np.where(par, -np.inf, np.minimum(t1, t2))
        hi = np.where(par, np.inf, np.maximum(t1, t2))
        outside = par & (abs(o[a]) > h[a])
        lo = np.where(outside, np.inf, lo)
        tmin = np.maximum(tmin, lo)
        tmax = np.minimum(tmax, hi)
    seg = np.linalg.norm(v, axis=1)
    return np.clip(tmax - tmin, 0.0, None) * seg


class RadiationField:
    def __init__(self, sources=(), shields=(), background: float = 0.08, r_min: float = 0.05):
        self.sources = list(sources)
        self.shields = list(shields)
        self.background = float(background)
        self.r_min = float(r_min)

    def _src_points(self):
        P, S = [], []
        for s in self.sources:
            p, w = s.points()
            P.append(p)
            S.append(w)
        if not P:
            return np.zeros((0, 3)), np.zeros(0)
        return np.concatenate(P), np.concatenate(S)

    def dose_rate(self, pts, shielding: bool = True) -> np.ndarray:
        """Dose rate (uSv/h) at (N, 3) points (or one point)."""
        pts = np.atleast_2d(np.asarray(pts, float))
        out = np.full(len(pts), self.background)
        SP, SS = self._src_points()
        for sp, ss in zip(SP, SS):
            r2 = np.maximum(np.sum((pts - sp) ** 2, axis=1), self.r_min ** 2)
            att = np.zeros(len(pts))
            if shielding:
                for b in self.shields:
                    att += b.mu * _path_in_box(sp, pts, b)
            out += ss * np.exp(-att) / r2
        return out

    def hottest_source(self):
        SP, SS = self._src_points()
        k = int(np.argmax(SS))
        return SP[k], SS[k]


class Dosimeter:
    def __init__(self, sensitivity_cps: float = 50.0, dead_time_s: float = 0.0, rng=None):
        self.sens = float(sensitivity_cps)
        self.tau = float(dead_time_s)
        self.rng = rng or np.random.default_rng(0)

    def counts(self, rate_usvh: float, dwell_s: float) -> int:
        n_true = self.sens * float(rate_usvh)
        n_obs = n_true / (1.0 + n_true * self.tau) if self.tau > 0 else n_true   # non-paralysable approximation
        return int(self.rng.poisson(max(n_obs, 0.0) * float(dwell_s)))

    def estimate(self, counts: int, dwell_s: float):
        """(dose-rate estimate, 1-sigma) in uSv/h from a count."""
        cps = counts / max(dwell_s, 1e-9)
        if self.tau > 0:
            cps = cps / max(1.0 - cps * self.tau, 1e-6)
        rate = cps / self.sens
        sd = math.sqrt(max(counts, 1)) / max(dwell_s, 1e-9) / self.sens
        return rate, sd

    def measure_with_truth(self, fld: RadiationField, pos, dwell_s: float):
        """(reading, true_rate): the reading as `measure` returns it, and the true dose rate (uSv/h) at `pos` for the
        grader. Keep `true_rate` on the grader side; a mission program gets the reading only."""
        true = float(fld.dose_rate(pos)[0])
        c = self.counts(true, dwell_s)
        est, sd = self.estimate(c, dwell_s)
        return {"counts": c, "rate": est, "sd": sd, "dwell_s": float(dwell_s)}, true

    def measure(self, fld: RadiationField, pos, dwell_s: float) -> dict:
        """What the instrument shows after a dwell at `pos`: {'counts', 'rate' (uSv/h estimate), 'sd' (its 1-sigma),
        'dwell_s'}. The true dose rate is not part of it (since 2026-11-13; use measure_with_truth on the
        grader side)."""
        return self.measure_with_truth(fld, pos, dwell_s)[0]


class DoseAccumulator:
    """The robot's accumulated dose (uSv)."""

    def __init__(self):
        self.total_usv = 0.0
        self.time_s = 0.0

    def add(self, rate_usvh: float, dt_s: float):
        self.total_usv += float(rate_usvh) * float(dt_s) / 3600.0
        self.time_s += float(dt_s)
        return self.total_usv

    def add_path(self, fld: RadiationField, points, speed_mps: float):
        """Dose along a polyline (N, 3) travelled at a constant speed (midpoint rule per segment)."""
        P = np.asarray(points, float)
        if len(P) < 2:
            return self.total_usv
        mid = 0.5 * (P[1:] + P[:-1])
        dt = np.linalg.norm(P[1:] - P[:-1], axis=1) / max(speed_mps, 1e-6)
        rates = fld.dose_rate(mid)
        for r, t in zip(rates, dt):
            self.add(r, t)
        return self.total_usv


def rasterise(fld: RadiationField, x_range, y_range, res: float = 0.1, z: float = 0.3, shielding: bool = True):
    """Dose-rate map on a floor grid at height z: (grid (ny, nx) uSv/h, xs, ys)."""
    xs = np.arange(x_range[0], x_range[1] + 1e-9, res)
    ys = np.arange(y_range[0], y_range[1] + 1e-9, res)
    X, Y = np.meshgrid(xs, ys)
    P = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    return fld.dose_rate(P, shielding).reshape(len(ys), len(xs)), xs, ys
