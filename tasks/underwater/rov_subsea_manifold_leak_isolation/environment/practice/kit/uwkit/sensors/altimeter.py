"""Single-beam echo sounder / altimeter (115 kHz class, 25 deg conical beam).

Rays fill the beam cone (centre + 3 rings).  Their returns (beam pattern x backscatter, area-TVG as for the imaging
sonar) are deposited into a range envelope; the reported distance is the leading edge of the strongest echo (first
sample above 50 % of the peak after smoothing), converted with the configured speed of sound.  Confidence (0-100 %)
from the peak SNR.  The beam axis is the site +z axis (transducer normal, same convention as the `dvl` site of the reference
vehicle; boresight = -1 for a site whose -z looks down); site `altimeter` if the vehicle has one, else `dvl`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import mujoco

from ._util import RayCaster, RateScheduler, site_pose, SpecItem
from .materials import MaterialMap
from .acoustics import WaterColumn, SEAWATER_TEMPERATE
from .sonar import _deposit, _gauss_kernel


@dataclass
class AltimeterConfig:
    frequency_hz: float = 115e3
    beam_deg: float = 25.0
    min_range_m: float = 0.3
    max_range_m: float = 100.0
    scan_length_m: float = 30.0          # user setting (auto mode in the device; fixed here)
    n_bins: int = 600
    rate_hz: float = 10.0
    speed_of_sound_cfg: float = 1500.0
    resolution_frac: float = 0.005
    boresight: int = 1                   # +1: beam around site +z (transducer normal); -1: site -z


@dataclass
class AltimeterHidden:
    snr0_db: float = 95.0                # SNR of a sand bottom at 1 m, normal incidence (area TVG)
    noise_sigma_frac: float = 0.002


@dataclass
class AltimeterReading:
    t: float
    valid: bool
    distance_m: float
    confidence_pct: float


def altimeter_public_spec(cfg: AltimeterConfig) -> list[SpecItem]:
    return [
        SpecItem("type", "single-beam echo sounder / altimeter"),
        SpecItem("frequency", cfg.frequency_hz / 1e3, "kHz"),
        SpecItem("beam width", cfg.beam_deg, "deg (conical)"),
        SpecItem("range", f"{cfg.min_range_m} .. {cfg.max_range_m}", "m"),
        SpecItem("resolution", "0.5 % of range"),
        SpecItem("output", "distance to the first strong echo and a confidence 0-100 %"),
    ]


class Altimeter:
    def __init__(self, m: mujoco.MjModel, materials: MaterialMap, rng: np.random.Generator,
                 water: WaterColumn = SEAWATER_TEMPERATE, site: Optional[str] = None,
                 config: Optional[AltimeterConfig] = None, hidden: Optional[AltimeterHidden] = None,
                 geomgroup: Optional[Sequence[int]] = None, t0: float = 0.0):
        self.cfg = config or AltimeterConfig()
        self.hid = hidden or AltimeterHidden()
        self.rng = rng
        self.mat = materials
        self.water = water
        if site is None:
            site = "altimeter" if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "altimeter") >= 0 else "dvl"
        self.sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.sid < 0:
            raise KeyError(site)
        root = int(m.body_rootid[m.site_bodyid[self.sid]])
        self.caster = RayCaster(m, geomgroup=geomgroup, bodyexclude=root)
        self.sched = RateScheduler(self.cfg.rate_hz, t0)
        hw = math.radians(self.cfg.beam_deg) * 0.75
        dirs, gains, dom = [], [], []
        s = math.radians(self.cfg.beam_deg) / 2.355
        rings = [(0.0, 1)] + [(hw * f, n) for f, n in ((0.33, 8), (0.66, 16), (1.0, 24))]
        for th, n in rings:
            for j in range(n):
                ph = 2 * math.pi * j / n
                dirs.append([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph),
                             float(self.cfg.boresight) * math.cos(th)])
                gains.append(math.exp(-(th / s) ** 2))
                dom.append(1.0 / n * max(th, hw * 0.2))
        self.dirs = np.array(dirs)
        self.g2 = np.array(gains)
        self.dom = np.array(dom) * (hw / 3)
        self.alpha = water.absorption_db_per_m(self.cfg.frequency_hz, 10.0)
        self.last: Optional[AltimeterReading] = None

    def update(self, t: float, d: mujoco.MjData) -> list[AltimeterReading]:
        out = []
        while self.sched.due(t):
            tf = self.sched.fire()
            out.append(self._ping(tf, d))
        if out:
            self.last = out[-1]
        return out

    def _ping(self, t: float, d: mujoco.MjData) -> AltimeterReading:
        c, h = self.cfg, self.hid
        o, R = site_pose(d, self.sid)
        if o[2] >= 0:
            return AltimeterReading(t, False, float("nan"), 0.0)
        scale = c.speed_of_sound_cfg / self.water.sound_speed(-o[2])
        dw = self.dirs @ R.T
        gid, dist, nrm = self.caster.cast(d, o, dw, c.scan_length_m / scale)
        hit = (gid >= 0) & (dist > 0)
        if not np.any(hit):
            return AltimeterReading(t, False, float("nan"), 0.0)
        r = dist[hit]
        cosi = np.abs(np.einsum("ij,ij->i", dw[hit], nrm[hit]))
        sig = self.mat.sigma(gid[hit], cosi)
        I = self.g2[hit] * sig * self.dom[hit] / np.maximum(cosi, 0.05)
        tan_i = np.sqrt(np.maximum(1 - cosi ** 2, 0)) / np.maximum(cosi, 0.05)
        dr = np.clip(r * math.radians(c.beam_deg) / 6 * tan_i, 0, 0.5 * r)
        prof = np.zeros(c.n_bins)
        _deposit(prof, r * scale, dr * scale, I, c.scan_length_m)
        k = _gauss_kernel(max(c.resolution_frac * float(np.median(r)) / (c.scan_length_m / c.n_bins), 0.5))
        prof = np.convolve(prof, k, mode="same")
        if prof.max() <= 0:
            return AltimeterReading(t, False, float("nan"), 0.0)
        ipk = int(np.argmax(prof))
        above = np.nonzero(prof[:ipk + 1] >= 0.5 * prof[ipk])[0]
        i0 = int(above[0])
        dist_m = (i0 + 0.5) * c.scan_length_m / c.n_bins
        rpk = max(dist_m / scale, 0.05)
        # echo energy relative to a sand bottom at normal incidence filling the beam; area target -> 20 log r
        e_ref = self.mat.table["sand"].mu * float((self.g2 * self.dom).sum())
        snr = h.snr0_db + 10 * math.log10(max(float(I.sum()) / e_ref, 1e-12)) \
            - 20 * math.log10(rpk) - 2 * self.alpha * rpk
        conf = float(np.clip((snr - 10.0) / 30.0 * 100.0, 0, 100))
        dist_m += h.noise_sigma_frac * dist_m * self.rng.standard_normal()
        valid = conf > 0 and c.min_range_m <= dist_m <= c.max_range_m
        return AltimeterReading(t, bool(valid), float(dist_m) if valid else float("nan"), conf)
