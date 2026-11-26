"""The single-frame hazard check of straight moves (the maker's documented occupancy parameters, ported as
parameters, not as its code).

For one depth frame (ranges along the ray table's unit rays, in the depth camera frame):
  1. points: pixels used by the caller (mask) whose range r satisfies min_range <= r < max_range; point = r x ray;
  2. occupied voxels: edge `voxel`, key = floor(point / voxel), the grid's origin at the depth camera centre (a
     grid shift, if given, is added to the points and to the path samples alike);
  3. inflation: every occupied key plus every integer offset o with |o| x voxel <= inflation radius, |o_i| <=
     round(radius / voxel) (the maker's sphere list; the configuration states how many offsets that is, and the
     check refuses to run if the count differs);
  4. a straight move: the robot's centre starts at `start` (depth frame) and goes along the unit direction u;
     samples s = k x path_sample, k = 0, 1, ...; a sample collides when it, or any of its 26 neighbours on the
     path_sample grid (the sample plus every combination of -1, 0, +1 path_sample per axis), falls in an
     inflated voxel;
  5. the move's stop index is the first colliding k (its stop distance k x path_sample); a move of length L is
     clear when no sample with k <= round(L / path_sample) collides, else it is blocked at the stop distance.
Directions of the move family: u = (sin az cos el, sin el, cos az cos el) in the depth camera frame, for every
elevation (outer loop) and azimuth (inner loop) of the configuration.
"""
import math

import numpy as np

from . import need


def inflation_offsets(voxel, radius):
    """Integer offsets o, |o_i| <= round(radius / voxel), with |o| x voxel <= radius; (K, 3) int64 in x, y, z
    loop order."""
    mx = int(round(radius / voxel))
    g = np.arange(-mx, mx + 1)
    r2 = radius * radius
    out = [(x, y, z) for x in g for y in g for z in g
           if (x * voxel) ** 2 + (y * voxel) ** 2 + (z * voxel) ** 2 <= r2]
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def directions(azimuth_deg, elevation_deg):
    """Unit directions (n, 3) of the move family, elevation-major (numpy's scalar functions, so that every user of
    the library gets the same bits)."""
    out = []
    for el in elevation_deg:
        for az in azimuth_deg:
            a, e = np.radians(float(az)), np.radians(float(el))
            out.append([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    return np.array(out, dtype=float)


def _encode(K, lo, span):
    K = K - lo
    return (K[..., 0] * span[1] + K[..., 1]) * span[2] + K[..., 2]


class HazardCheck:
    def __init__(self, voxel, min_range, max_range, inflation_radius, n_offsets, path_sample, max_length, start):
        self.voxel = float(voxel)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.step = float(path_sample)
        self.max_length = float(max_length)
        self.start = np.asarray(start, float).reshape(3)
        self.offsets = inflation_offsets(self.voxel, float(inflation_radius))
        if len(self.offsets) != int(n_offsets):
            raise ValueError("the inflation radius gives %d offsets, the configuration states %d"
                             % (len(self.offsets), int(n_offsets)))
        self.n_samples = int(round(self.max_length / self.step)) + 1
        self.s = np.arange(self.n_samples) * self.step
        self.nb = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)],
                           dtype=float) * self.step

    @classmethod
    def from_config(cls, cfg, start=None):
        """The check with the configuration's numbers; `start` overrides hazard.move_start (for a regression)."""
        if not math.isclose(need(cfg, "hazard.inflation_radius"),
                            need(cfg, "hazard.robot_radius") + need(cfg, "hazard.collision_distance")):
            raise ValueError("hazard.inflation_radius is not hazard.robot_radius + hazard.collision_distance")
        return cls(need(cfg, "hazard.voxel"), need(cfg, "hazard.min_range"), need(cfg, "hazard.max_range"),
                   need(cfg, "hazard.inflation_radius"), need(cfg, "hazard.inflation_offsets"),
                   need(cfg, "hazard.path_sample"), need(cfg, "hazard.max_length"),
                   need(cfg, "hazard.move_start") if start is None else start)

    def family(self, cfg):
        """The configuration's move directions (n, 3)."""
        return directions(need(cfg, "hazard.directions_azimuth"), need(cfg, "hazard.directions_elevation"))

    def length_index(self, L):
        """The last sample index of a move of length L."""
        k = int(round(float(L) / self.step))
        if k >= self.n_samples:
            raise ValueError("a move of %g m is longer than the check's %g m" % (L, self.max_length))
        return k

    def stop_indices(self, ranges, rays, mask, dirs, grid_shift=None):
        """First colliding sample index per direction (-1: no sample up to max_length collides).
        ranges (n,), rays (n, 3) unit rays in the depth frame, mask (n,) pixels to use, dirs (m, 3)."""
        r = np.asarray(ranges, float).reshape(-1)
        Rr = np.asarray(rays, float).reshape(-1, 3)
        ok = np.asarray(mask, bool).reshape(-1) & np.isfinite(r) & (r >= self.min_range) & (r < self.max_range)
        D = np.asarray(dirs, float).reshape(-1, 3)
        out = np.full(len(D), -1, dtype=np.int64)
        if not ok.any():
            return out
        pts = Rr[ok] * r[ok, None]
        path = self.start[None, None, None, :] + self.s[None, :, None, None] * D[:, None, None, :] + self.nb[None, None, :, :]
        if grid_shift is not None:
            sh = np.asarray(grid_shift, float).reshape(3)
            pts = pts + sh
            path = path + sh
        keys = np.unique(np.floor(pts / self.voxel).astype(np.int64), axis=0)
        pk = np.floor(path / self.voxel).astype(np.int64)
        infl = (keys[:, None, :] + self.offsets[None, :, :]).reshape(-1, 3)
        lo = np.minimum(infl.min(axis=0), pk.reshape(-1, 3).min(axis=0))
        hi = np.maximum(infl.max(axis=0), pk.reshape(-1, 3).max(axis=0))
        span = hi - lo + 1
        inf_code = np.unique(_encode(infl, lo, span))
        hit = np.isin(_encode(pk, lo, span), inf_code).any(axis=2)           # (m, n_samples)
        h = hit.any(axis=1)
        out[h] = np.argmax(hit[h], axis=1)
        return out

    def stops(self, ranges, rays, mask, dirs, grid_shift=None):
        """Stop distance per direction (inf: clear over max_length)."""
        k = self.stop_indices(ranges, rays, mask, dirs, grid_shift)
        return np.where(k >= 0, self.s[np.maximum(k, 0)], np.inf)

    def is_clear(self, stop_index, L):
        """A move of length L along a direction whose first colliding sample is stop_index (-1 none)."""
        k = int(stop_index)
        return k < 0 or k > self.length_index(L)
