"""Doppler velocity log, 4-beam convex Janus, bottom tracking only.

Geometry (site `dvl`; reference-vehicle convention: x forward, y starboard, z down = transducer normal, i.e. the
site +z axis is the boresight): beam k points 22.5 deg away from the boresight (site +z, or -z with
DVLConfig.boresight = -1 for a site aligned with the body), at azimuth 45 + 90 k deg in the site xy-plane.  Each beam is sampled with a centre ray plus 6 rays on
the half-power cone (beam width 4.4 deg) and ray-cast with mj_multiRay.

Per beam: range = beam-pattern- and backscatter-weighted mean of the hit ranges; SNR from the sonar equation

    SNR = FOM_0 - 20 log10(r) - 2 alpha r + 10 log10(sigma(i))                 (area target: 40 log r - 20 log r)

with FOM_0 chosen so that a sand bottom at the rated 50 m altitude sits at the detection threshold (soft mud ->
~33 m).  A beam is valid when it hits, the range is inside [min, max], SNR >= threshold and the ray spread inside the
beam is small (no straddling of an edge).  Radial velocity of beam k: m_k = (v_target - v_dvl) . b_k + noise, where
v_target is the velocity of whatever the beam hits (moving objects bias the solution).

Velocity solution: least squares over the valid beams (>= 3 needed).  With 4 beams the redundant beam gives the
error velocity; if it exceeds a threshold the report is flagged invalid.  The covariance of the solution is reported.
Altitude: plane fit through the beam hit points, distance along the boresight.

Lock is lost (valid = False) when: fewer than 3 valid beams; altitude outside 0.05 .. 50 m (soft bottoms less);
steep slopes (grazing beams lose SNR, high incidence); the beams straddle structure edges; a moving object makes the
error velocity large; speed > 3.75 m/s; the unit is out of the water.  Ping rate adapts to altitude: 15 Hz at 1 m
down to 4 Hz at 50 m (4 Hz while searching).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import mujoco

from ._util import RayCaster, site_id, site_pose, site_velocity_world, SpecItem
from .materials import MaterialMap
from .acoustics import WaterColumn, SEAWATER_TEMPERATE


@dataclass
class DVLConfig:
    frequency_hz: float = 1.0e6
    beam_angle_deg: float = 22.5
    beam_width_deg: float = 4.4
    beam_azimuth0_deg: float = 45.0
    min_altitude_m: float = 0.05
    max_altitude_m: float = 50.0
    max_speed_m_s: float = 3.75
    rate_max_hz: float = 15.0
    rate_min_hz: float = 4.0
    rate_alt_lo_m: float = 1.0
    rate_alt_hi_m: float = 50.0
    speed_of_sound_cfg: float = 1500.0
    boresight: int = 1                     # +1: beams around site +z (transducer normal, z down); -1: site -z


@dataclass
class DVLHidden:
    snr_threshold_db: float = 10.0
    fom0_db: Optional[float] = None        # None -> calibrated so that sand at max altitude = threshold
    sigma_radial_m_s: float = 0.0015       # per-beam radial velocity noise (white)
    sigma_radial_frac: float = 0.004       # + fraction of |radial velocity|
    scale_error: float = 0.0               # unit scale-factor error (drawn per unit if None-initialised)
    bias_m_s: tuple = (0.0, 0.0, 0.0)      # constant velocity bias in the DVL frame
    mount_yaw_err_rad: float = 0.0
    error_velocity_max: float = 0.05       # m/s (+2 % of speed) before the 4-beam report is rejected
    spread_max_frac: float = 0.15          # ray range spread inside one beam (edge straddling)
    dropout_prob: float = 0.005            # random ping loss (bubbles, thruster wash)

    @staticmethod
    def sample(rng: np.random.Generator) -> "DVLHidden":
        """Draw unit-specific internals (scale error within the +-1 % long-term accuracy class, mm/s bias,
        small mounting yaw error)."""
        return DVLHidden(scale_error=float(rng.uniform(-0.006, 0.006)),
                         bias_m_s=tuple((rng.normal(0, 0.0007, 3)).tolist()),
                         mount_yaw_err_rad=float(rng.normal(0, math.radians(0.3))))


@dataclass
class DVLReading:
    t: float
    valid: bool                            # velocity valid (bottom lock)
    velocity: np.ndarray                   # (3,) m/s in the DVL (site) frame, NaN if invalid
    fom: float                             # figure of merit: 1-sigma of the velocity estimate (m/s), NaN if invalid
    covariance: np.ndarray                 # (3, 3)
    altitude: float                        # m, NaN if fewer than 3 beams
    beam_range: np.ndarray                 # (4,) m along each beam, NaN if not valid
    beam_valid: np.ndarray                 # (4,) bool
    beam_velocity: np.ndarray              # (4,) m/s radial
    status: str                            # "lock", "search", "out_of_water", "overspeed", "dropout", "inconsistent"


def dvl_public_spec(cfg: DVLConfig) -> list[SpecItem]:
    return [
        SpecItem("type", "Doppler velocity log, 4-beam convex Janus, bottom track"),
        SpecItem("frequency", cfg.frequency_hz / 1e6, "MHz"),
        SpecItem("beam angle / beam width", f"{cfg.beam_angle_deg} / {cfg.beam_width_deg}", "deg"),
        SpecItem("altitude range", f"{cfg.min_altitude_m} .. {cfg.max_altitude_m}", "m",
                 "upper end depends on bottom type"),
        SpecItem("max velocity", cfg.max_speed_m_s, "m/s"),
        SpecItem("ping rate", f"{cfg.rate_min_hz:.0f} .. {cfg.rate_max_hz:.0f}", "Hz", "adapts to altitude"),
        SpecItem("long-term accuracy", "+-1.01 %", "", "plus a few mm/s"),
        SpecItem("output", "velocity (DVL frame), validity, figure of merit, covariance, altitude, per-beam data"),
        SpecItem("installation", "keep a 65 deg cone below the transducers clear"),
        SpecItem("near structures", "outer beams that graze or hit a nearby wall, panel or rail lose bottom lock or "
                 "bias the velocity; within about 0.3 m of a vertical structure expect no lock"),
        SpecItem("attitude", "bottom lock is not guaranteed beyond about 30 deg of roll or pitch (outer beams graze the "
                 "bottom); typically lost between 35 and 50 deg"),
    ]


class DVL:
    def __init__(self, m: mujoco.MjModel, materials: MaterialMap, rng: np.random.Generator,
                 water: WaterColumn = SEAWATER_TEMPERATE, site: str = "dvl",
                 config: Optional[DVLConfig] = None, hidden: Optional[DVLHidden] = None,
                 geomgroup: Optional[Sequence[int]] = None, exclude_body: Optional[int] = None, t0: float = 0.0):
        self.m = m
        self.cfg = config or DVLConfig()
        self.hid = hidden or DVLHidden()
        self.rng = rng
        self.mat = materials
        self.water = water
        self.sid = site_id(m, site)
        root = int(m.body_rootid[m.site_bodyid[self.sid]]) if exclude_body is None else int(exclude_body)
        self.caster = RayCaster(m, geomgroup=geomgroup, bodyexclude=root)
        c = self.cfg
        ba = math.radians(c.beam_angle_deg)
        self.beams = np.array([[math.sin(ba) * math.cos(math.radians(c.beam_azimuth0_deg + 90 * k)),
                                math.sin(ba) * math.sin(math.radians(c.beam_azimuth0_deg + 90 * k)),
                                float(c.boresight) * math.cos(ba)] for k in range(4)])
        # sub-rays: centre + 6 on the half-power cone
        hw = math.radians(c.beam_width_deg) / 2.0
        sub = []
        wts = []
        for b in self.beams:
            u = np.cross(b, [0, 0, 1.0]); u /= np.linalg.norm(u)
            v = np.cross(b, u)
            sub.append(b); wts.append(1.0)
            for j in range(6):
                ph = j * math.pi / 3
                dvec = b * math.cos(hw) + math.sin(hw) * (math.cos(ph) * u + math.sin(ph) * v)
                sub.append(dvec / np.linalg.norm(dvec)); wts.append(0.25)   # two-way gain at the -3 dB cone
        self.sub_dirs = np.array(sub)                    # (28, 3) in the DVL frame
        self.sub_w = np.array(wts)
        self.next_t = float(t0)
        self.last: Optional[DVLReading] = None
        self.alpha = water.absorption_db_per_m(c.frequency_hz, 10.0)
        if self.hid.fom0_db is None:
            # sand (lambert -25 dB) at max altitude on a flat bottom (incidence = beam angle) sits at threshold
            r = c.max_altitude_m / math.cos(ba)
            sig = materials.table["sand"].mu * math.cos(ba) ** 2
            self.fom0 = self.hid.snr_threshold_db + 20 * math.log10(r) + 2 * self.alpha * r - 10 * math.log10(sig)
        else:
            self.fom0 = self.hid.fom0_db
        cy, sy = math.cos(self.hid.mount_yaw_err_rad), math.sin(self.hid.mount_yaw_err_rad)
        self.R_mis = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1.0]])   # true beams = R_mis @ nominal

    def rate_for_altitude(self, alt: Optional[float]) -> float:
        c = self.cfg
        if alt is None or not np.isfinite(alt):
            return c.rate_min_hz
        a = min(max(alt, c.rate_alt_lo_m), c.rate_alt_hi_m)
        k = math.log(c.rate_min_hz / c.rate_max_hz) / math.log(c.rate_alt_hi_m / c.rate_alt_lo_m)
        return c.rate_max_hz * (a / c.rate_alt_lo_m) ** k

    def update(self, t: float, m: mujoco.MjModel, d: mujoco.MjData) -> list[DVLReading]:
        out = []
        while self.next_t <= t + 1e-12:
            r = self._ping(self.next_t, m, d)
            out.append(r)
            self.last = r
            self.next_t += 1.0 / self.rate_for_altitude(r.altitude if r.valid else None)
        return out

    def _point_velocity(self, m, d, gid: int, p: np.ndarray) -> np.ndarray:
        res = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, gid, res, 0)
        return res[3:] + np.cross(res[:3], p - d.geom_xpos[gid])

    def _ping(self, t: float, m: mujoco.MjModel, d: mujoco.MjData) -> DVLReading:
        c, hid = self.cfg, self.hid
        nan3 = np.full(3, np.nan)
        origin, R = site_pose(d, self.sid)
        if origin[2] >= 0.0:
            return DVLReading(t, False, nan3, np.nan, np.full((3, 3), np.nan), np.nan, np.full(4, np.nan),
                              np.zeros(4, bool), np.full(4, np.nan), "out_of_water")
        if self.rng.random() < hid.dropout_prob:
            return DVLReading(t, False, nan3, np.nan, np.full((3, 3), np.nan), np.nan, np.full(4, np.nan),
                              np.zeros(4, bool), np.full(4, np.nan), "dropout")
        Rt = R @ self.R_mis
        dirs_w = self.sub_dirs @ Rt.T
        rmax = c.max_altitude_m / math.cos(math.radians(c.beam_angle_deg)) * 1.3
        gid, dist, nrm = self.caster.cast(d, origin, dirs_w, rmax)
        v_dvl_w, _ = site_velocity_world(m, d, self.sid)
        ranges = np.full(4, np.nan)
        bvalid = np.zeros(4, bool)
        bvel = np.full(4, np.nan)
        pts = []
        c_true = self.water.sound_speed(-origin[2])
        scale = c.speed_of_sound_cfg / c_true
        for k in range(4):
            sl = slice(7 * k, 7 * k + 7)
            g, r, n = gid[sl], dist[sl], nrm[sl]
            hit = (g >= 0) & (r > 0)
            if hit.sum() < 4:
                continue
            cosi = np.abs(np.einsum("ij,ij->i", dirs_w[sl][hit], n[hit]))
            sig = self.mat.sigma(g[hit], cosi)
            w = self.sub_w[sl][hit] * sig
            if w.sum() <= 0:
                continue
            rk = float((w * r[hit]).sum() / w.sum())
            spread = float((r[hit].max() - r[hit].min()) / max(rk, 1e-6))
            sig_c = float(sig[0]) if hit[0] else float(sig.mean())
            snr = self.fom0 - 20 * math.log10(max(rk, 0.05)) - 2 * self.alpha * rk + 10 * math.log10(max(sig_c, 1e-12))
            alt_k = rk * math.cos(math.radians(c.beam_angle_deg))
            if snr < hid.snr_threshold_db or spread > hid.spread_max_frac or rk < c.min_altitude_m or \
                    alt_k > c.max_altitude_m * 1.05:
                continue
            # radial velocity: relative motion of the hit surface along the beam
            g0 = int(g[hit][np.argmax(w)])
            p_hit = origin + dirs_w[7 * k] * rk
            v_t = self._point_velocity(m, d, g0, p_hit)
            vr_true = float((v_t - v_dvl_w) @ dirs_w[7 * k])
            sd = hid.sigma_radial_m_s + hid.sigma_radial_frac * abs(vr_true)
            # Doppler -> velocity uses the configured sound speed: v_meas = v_true * c_cfg / c_true
            bvel[k] = vr_true * scale * (1.0 + hid.scale_error) + sd * self.rng.standard_normal()
            ranges[k] = rk * scale
            bvalid[k] = True
            pts.append(self.sub_dirs[7 * k] * rk * scale)     # in the (nominal) DVL frame
        nval = int(bvalid.sum())
        alt = np.nan
        if nval >= 3:
            P = np.array(pts)
            cen = P.mean(0)
            if nval == 3:
                nrm_ = np.cross(P[1] - P[0], P[2] - P[0])
            else:
                _, _, vt = np.linalg.svd(P - cen)
                nrm_ = vt[-1]
            nrm_ = nrm_ / np.linalg.norm(nrm_)
            if abs(nrm_[2]) > 1e-3:
                alt = float(abs(cen @ nrm_) / abs(nrm_[2]))      # distance along the boresight to the fitted plane
        if nval < 3:
            return DVLReading(t, False, nan3, np.nan, np.full((3, 3), np.nan), alt, ranges, bvalid, bvel, "search")
        A = -self.beams[bvalid]                                  # m_k = -(v_dvl . b_k) (static bottom)
        y = bvel[bvalid]
        sd = hid.sigma_radial_m_s + hid.sigma_radial_frac * np.abs(y)
        Wm = np.diag(1.0 / sd ** 2)
        AtWA = A.T @ Wm @ A
        cov = np.linalg.inv(AtWA)
        v = cov @ A.T @ Wm @ y + np.asarray(hid.bias_m_s)
        status = "lock"
        valid = True
        if nval == 4:
            resid = y - A @ (v - np.asarray(hid.bias_m_s))
            ev = float(np.sqrt(np.mean(resid ** 2)) * 2.0)
            if ev > hid.error_velocity_max + 0.02 * float(np.linalg.norm(v)):
                valid, status = False, "inconsistent"
        if float(np.linalg.norm(v)) > c.max_speed_m_s:
            valid, status = False, "overspeed"
        fom = float(math.sqrt(np.trace(cov) / 3.0))
        if not valid:
            return DVLReading(t, False, nan3, np.nan, np.full((3, 3), np.nan), alt, ranges, bvalid, bvel, status)
        return DVLReading(t, True, v, fom, cov, alt, ranges, bvalid, bvel, status)
