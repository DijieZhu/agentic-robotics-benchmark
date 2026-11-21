"""Keep-in / keep-out validation of planned segments, a port of the reference firmware's validator.

The firmware (choreographer validator) builds a voxel map at the mapper resolution (0.08 m) over the keep-in
bounds (+2 voxels): (1) the surfaces of every keep-in zone, one voxel outside it, are occupied; (2) the volume of
every keep-in zone is set free (this removes surfaces between overlapping keep-ins); (3) the surfaces of every
keep-out zone are occupied; (4) the occupied cells are dilated by the inflation = robot radius 0.28 m + collision
distance 0.02 m (a cylinder of radius ceil(r / res) voxels in x-y and +-ceil(h / res) voxels in z, as the planner
library's map utility).  A segment, resampled at 10 Hz, violates KEEP_IN if its last point or any point is in an
unknown cell or outside the map, and KEEP_OUT if any point is in an occupied cell.
Index of a point: round((p - origin) / res - 0.5) (planner library).  The dock and perch behaviours of the
firmware switch validation off for their final moves; ``spacekit.primitives`` does the same.
[R firmware choreographer validator, mapper config; the dilation shape and the index rounding are recalled from the
public planner-library source, which is not in the downloaded tree -- G]
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["ZONE_RES", "ZONE_INFLATION", "UNKNOWN", "FREE", "OCC", "ZoneMap", "resample_positions"]

ZONE_RES = 0.08            # m  [R mapper map_resolution / validator map_res_]
ZONE_INFLATION = 0.30      # m  [R robot_radius 0.28 + collision_distance 0.02]
EPS = 0.001                # [R validator.h]
UNKNOWN, FREE, OCC = -1, 0, 100


class ZoneMap:
    """Voxel validator for a zone list: dicts with 'type' ('keepin' | 'keepout'), 'min' (3,), 'max' (3,).

    ``check_point(p)`` -> 'ok' | 'keep_in' | 'keep_out'; ``check_positions(P)`` the first violation of a sampled
    path (end point first, as the firmware); ``check_segment(seg)`` for spacekit Setpoint lists (10 Hz resample);
    ``clearance_ok(p)`` boolean.  The map is built once (numpy, a few seconds for a station's zone file).
    inflation_arm (v0.4; None = v0.3): a second map dilated by this radius (``inflation_arm_z`` vertically,
    default the same) for a robot with its arm deployed; the query methods take ``arm=True`` to use it.
    approach_tol (v0.4): the slack of the band rule's "never closer" test [m] (``check_positions``).
    """

    def __init__(self, zones, res: float = ZONE_RES, inflation: float = ZONE_INFLATION, inflation_z: float | None = None,
                 inflation_arm: float | None = None, inflation_arm_z: float | None = None, approach_tol: float = 0.01):
        self.res = float(res)
        self.approach_tol = float(approach_tol)
        self.zones = [dict(type=z["type"], min=np.minimum(z["min"], z["max"]), max=np.maximum(z["min"], z["max"]),
                           id=z.get("id")) for z in zones]
        kin = [z for z in self.zones if z["type"] == "keepin"]
        if not kin:
            raise ValueError("zero keep-in zones")
        lo = np.min([z["min"] for z in kin], axis=0)
        hi = np.max([z["max"] for z in kin], axis=0)
        r = self.res
        self.origin = np.floor(lo / r) * r - 2.0 * r
        top = np.ceil(hi / r) * r + 2.0 * r
        self.dim = np.ceil((top - self.origin) / r).astype(int)
        self.map = np.full(tuple(self.dim), UNKNOWN, np.int8)
        for z in kin:
            self._process(z, OCC, surface=True)
        for z in kin:
            self._process(z, FREE, surface=False)
        for z in self.zones:
            if z["type"] == "keepout":
                self._process(z, OCC, surface=True)
        self.inflation = float(inflation)
        self.inflation_z = float(inflation if inflation_z is None else inflation_z)
        self.raw = self.map.copy()          # v0.4: the occupancy before the dilation (the band-exit rule measures to it)
        self._dilate()
        self.inflation_arm = None if inflation_arm is None else float(inflation_arm)
        self.inflation_arm_z = None if inflation_arm is None else float(inflation_arm if inflation_arm_z is None else inflation_arm_z)
        self.map_arm = None
        if self.inflation_arm is not None:
            self.map_arm = self.raw.copy()
            self.map_arm[self._occupied_after_dilation(self.raw, self.inflation_arm, self.inflation_arm_z)] = OCC

    def index(self, p) -> np.ndarray:
        """Voxel index of points p (N, 3) or (3,)."""
        return np.round((np.asarray(p, float) - self.origin) / self.res - 0.5).astype(int)

    def _set(self, pts, value):
        idx = self.index(pts)
        ok = np.all((idx >= 0) & (idx < self.dim), axis=1)
        idx = idx[ok]
        self.map[idx[:, 0], idx[:, 1], idx[:, 2]] = value

    def _process(self, z, value, surface: bool):
        r = self.res
        zmin = z["min"] + r * EPS
        zmax = z["max"] - r * EPS
        if surface:
            gap = r if z["type"] == "keepin" else 0.0
            zmin = zmin - gap
            zmax = zmax + gap
            for i in range(3):
                j, k = (i + 1) % 3, (i + 2) % 3
                a = np.arange(zmin[j], zmax[j] + 1e-12, r)
                b = np.arange(zmin[k], zmax[k] + 1e-12, r)
                A, B = np.meshgrid(a, b, indexing="ij")
                P = np.zeros((A.size, 3))
                P[:, j], P[:, k] = A.ravel(), B.ravel()
                P[:, i] = zmin[i]
                self._set(P, value)
                P[:, i] = zmax[i]
                self._set(P, value)
        else:
            axes = [np.arange(zmin[i], zmax[i] + 1e-12, r) for i in range(3)]
            G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
            self._set(G, value)

    def _dilate(self):
        self.map[self._occupied_after_dilation(self.map, self.inflation, self.inflation_z)] = OCC

    def _occupied_after_dilation(self, base, inflation: float, inflation_z: float) -> np.ndarray:
        rn = int(math.ceil(inflation / self.res))
        hn = int(math.ceil(inflation_z / self.res))
        occ = base == OCC
        # z first (+-hn), then the x-y disk (hypot <= rn): separable for this cylinder
        oz = occ.copy()
        for dz in range(1, hn + 1):
            oz[:, :, dz:] |= occ[:, :, :-dz]
            oz[:, :, :-dz] |= occ[:, :, dz:]
        out = oz.copy()
        for dx in range(-rn, rn + 1):
            for dy in range(-rn, rn + 1):
                if (dx == 0 and dy == 0) or math.hypot(dx, dy) > rn:
                    continue
                sx = slice(max(dx, 0), self.dim[0] + min(dx, 0))
                tx = slice(max(-dx, 0), self.dim[0] + min(-dx, 0))
                sy = slice(max(dy, 0), self.dim[1] + min(dy, 0))
                ty = slice(max(-dy, 0), self.dim[1] + min(-dy, 0))
                out[sx, sy, :] |= oz[tx, ty, :]
        return out

    def _map(self, arm: bool):
        if arm and self.map_arm is None:
            raise ValueError("arm=True needs a ZoneMap built with inflation_arm")
        return self.map_arm if arm else self.map

    def value(self, p, arm: bool = False) -> np.ndarray:
        """Map values at points (UNKNOWN outside the map); arm=True: the arm-deployed map."""
        P = np.atleast_2d(np.asarray(p, float))
        idx = self.index(P)
        ok = np.all((idx >= 0) & (idx < self.dim), axis=1)
        v = np.full(len(P), UNKNOWN, int)
        v[ok] = self._map(arm)[idx[ok, 0], idx[ok, 1], idx[ok, 2]]
        return v

    def check_point(self, p, arm: bool = False) -> str:
        """'ok', 'keep_out' (occupied) or 'keep_in' (unknown / outside)."""
        v = int(self.value(p, arm)[0])
        return "ok" if v == FREE else ("keep_out" if v == OCC else "keep_in")

    def clearance_ok(self, p) -> bool:
        """True when the point is in a free cell."""
        return self.check_point(p) == "ok"

    def check_positions(self, P, band_exit: bool | str = False, arm: bool = False) -> tuple:
        """(result, index of the first violating sample or -1): end point first, then every sample in order.
        band_exit (v0.4) for a path that starts inside the inflated band (an occupied cell that is not occupied before
        the dilation) -- False: v0.3 (refused);
          'exit' (the v0.4 defect fix): accepted if its leading samples inside the band move strictly away from the
            occupied cells (``clearance`` strictly increases between consecutive samples that move, stays equal
            between samples that do not -- a rotation in place) and every sample from the first free one on is free;
          True (default of the firmware layer: the exit rule merged with the "no approach" rule): accepted by the 'exit'
            rule, or if no sample is unknown or inside an occupied cell, the samples inside the band come first (it
            never re-enters the band) and none of them is closer to the occupied cells than any sample before it by
            more than ``approach_tol`` (the clearance never decreases): a robot in the band may leave it, move along
            the wall, or stay in the band, but never get closer; the end point may be in the band.
        arm: use the arm-deployed map (and its inflation for the clearance)."""
        P = np.atleast_2d(np.asarray(P, float))
        if band_exit and band_exit != "exit" and len(P):
            v0 = self.value(P, arm)
            if v0[0] == OCC and self._no_approach(P, v0, arm):
                return "ok", -1
        v_end = self.check_point(P[-1], arm)
        if v_end != "ok":
            return ("keep_out" if v_end == "keep_out" else "keep_in"), len(P) - 1
        v = self.value(P, arm)
        bad = np.nonzero(v != FREE)[0]
        if len(bad) == 0:
            return "ok", -1
        k = int(bad[0])
        if band_exit and k == 0 and self._leaves_band(P, v, arm):
            return "ok", -1
        return ("keep_out" if v[k] == OCC else "keep_in"), k

    def _leaves_band(self, P, v, arm: bool = False) -> bool:
        free = np.nonzero(v == FREE)[0]
        if len(free) == 0:
            return False
        k0 = int(free[0])
        if np.any(v[k0:] != FREE) or np.any(v[:k0] != OCC):
            return False
        lead = P[:k0 + 1]
        idx = self.index(lead[:k0])
        if np.any(self.raw[idx[:, 0], idx[:, 1], idx[:, 2]] != FREE):
            return False                      # inside an occupied cell itself, not in the band around it
        c = self.clearance(lead, arm)
        moved = np.linalg.norm(np.diff(lead, axis=0), axis=1) > 1e-6
        dc = np.diff(c)
        return bool(np.all(np.where(moved, dc > 1e-9, np.abs(dc) <= 1e-12)))

    def _no_approach(self, P, v, arm: bool = False) -> bool:
        """The "no approach" rule: a path that starts in the band never gets closer to the occupied cells (see check_positions)"""
        if np.any(v == UNKNOWN):
            return False
        free = np.nonzero(v == FREE)[0]
        k0 = int(free[0]) if len(free) else len(P)
        if np.any(v[k0:] != FREE):
            return False                      # back into the band (or an occupied cell) after leaving it
        lead = P[:k0]
        idx = self.index(lead)
        if np.any(self.raw[idx[:, 0], idx[:, 1], idx[:, 2]] != FREE):
            return False                      # inside an occupied cell itself, not in the band around it
        inf, inf_z = (self.inflation_arm, self.inflation_arm_z) if arm else (self.inflation, self.inflation_z)
        c = self.clearance(lead, arm)
        tol = self.approach_tol / min(inf, inf_z)
        return bool(np.all(c >= np.maximum.accumulate(c) - tol))

    def clearance(self, P, arm: bool = False) -> np.ndarray:
        """Normalised clearance of points (N, 3) to the occupied cells before the dilation, in the dilation's metric:
        min over occupied cells of max(horizontal distance / inflation, vertical distance / inflation_z), distances to
        the cell boxes; >= 1 means outside the band (searched in a window of the inflation around the points); arm:
        measured in the arm-deployed inflation."""
        P = np.atleast_2d(np.asarray(P, float))
        r = self.res
        inf, inf_z = (self.inflation_arm, self.inflation_arm_z) if arm else (self.inflation, self.inflation_z)
        if inf is None:
            raise ValueError("arm=True needs a ZoneMap built with inflation_arm")
        rn = int(math.ceil(inf / r)) + 1
        hn = int(math.ceil(inf_z / r)) + 1
        idx = self.index(P)
        lo = np.maximum(idx.min(axis=0) - [rn, rn, hn], 0)
        hi = np.minimum(idx.max(axis=0) + [rn, rn, hn] + 1, self.dim)
        sub = self.raw[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        occ = np.argwhere(sub == OCC)
        if len(occ) == 0:
            return np.full(len(P), np.inf)
        centre = self.origin + (occ + lo + 0.5) * r
        d = np.abs(P[:, None, :] - centre[None, :, :]) - 0.5 * r
        d = np.maximum(d, 0.0)
        dxy = np.hypot(d[..., 0], d[..., 1]) / inf
        dz = d[..., 2] / inf_z
        return np.min(np.maximum(dxy, dz), axis=1)

    def check_segment(self, seg, rate_hz: float = 10.0, band_exit: bool | str = False, arm: bool = False) -> tuple:
        """Validate a list of spacekit Setpoints (resampled at rate_hz); band_exit, arm as ``check_positions``."""
        return self.check_positions(resample_positions(seg, rate_hz), band_exit=band_exit, arm=arm)

    def free_fraction(self) -> float:
        """Share of free voxels in the map (diagnostic)."""
        return float(np.mean(self.map == FREE))


def resample_positions(seg, rate_hz: float = 10.0) -> np.ndarray:
    """Positions of a trapezoidal segment (Setpoint list) at rate_hz, using each sample's constant-acceleration
    extrapolation until the next sample (as the firmware's resampler)."""
    if not seg:
        return np.zeros((0, 3))
    t0, t1 = seg[0].t, seg[-1].t
    n = max(1, int(math.floor((t1 - t0) * rate_hz)) + 1)
    ts = t0 + np.arange(n) / rate_hz
    out = []
    k = 0
    for t in ts:
        while k + 1 < len(seg) and seg[k + 1].t <= t + 1e-12:
            k += 1
        s = seg[k]
        dt = t - s.t
        out.append(np.asarray(s.pos) + np.asarray(s.vel) * dt + 0.5 * np.asarray(s.acc) * dt * dt)
    out.append(np.asarray(seg[-1].pos))
    return np.array(out)
