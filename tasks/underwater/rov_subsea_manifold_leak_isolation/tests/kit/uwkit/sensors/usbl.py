"""Acoustic positioning (USBL-type: surface transceiver measures slant range + bearing to a vehicle locator).

Each interrogation at time t_p (period ~1 s with small jitter):
  * line-of-sight check transceiver -> locator with mj_ray; blocked paths are mostly lost, sometimes received via a
    reflected path (outlier: extra path length, wrong bearing);
  * slant range from travel time with the configured sound speed: rho_m = rho * c_cfg / c_mean + noise
    (0.2 % of range + 5 cm, 1 sigma);
  * azimuth / elevation noise ~1 deg (1 sigma) plus a fixed transceiver heading misalignment (hidden);
  * random multipath outliers (probability higher when the locator is near the surface or near structures);
  * the fix is delivered at t_p + rho / c + processing latency (1-2 s total), time-stamped t_p.
Output is the *raw* fix (world frame position, range, bearings, a quality std).  Fusing it with dead reckoning is left
to the agent's program.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import mujoco

from ._util import RayCaster, DelayLine, site_id, site_pose, SpecItem
from .acoustics import WaterColumn, SEAWATER_TEMPERATE


@dataclass
class USBLConfig:
    transceiver_pos: tuple = (0.0, 0.0, -1.0)   # world position of the surface transceiver
    period_s: float = 1.0
    max_range_m: float = 300.0
    speed_of_sound_cfg: float = 1500.0


@dataclass
class USBLHidden:
    range_sigma_frac: float = 0.002
    range_sigma_m: float = 0.05
    angle_sigma_rad: float = math.radians(1.0)
    heading_misalign_rad: float = 0.0
    latency_s: float = 1.0
    period_jitter_frac: float = 0.05
    outlier_prob: float = 0.02
    outlier_prob_shallow: float = 0.10          # when the locator is within 2 m of the surface
    blocked_receive_prob: float = 0.3           # blocked line of sight -> reflected-path outlier instead of loss
    outlier_extra_path_m: tuple = (0.5, 6.0)
    outlier_angle_rad: tuple = (math.radians(3.0), math.radians(25.0))

    @staticmethod
    def sample(rng: np.random.Generator) -> "USBLHidden":
        return USBLHidden(heading_misalign_rad=float(rng.normal(0, math.radians(0.5))),
                          latency_s=float(rng.uniform(0.8, 1.4)),
                          angle_sigma_rad=float(math.radians(rng.uniform(0.6, 1.2))))


@dataclass
class USBLFix:
    t: float                                   # measurement time
    t_delivered: float
    valid: bool
    position: np.ndarray                       # (3,) world, NaN if invalid
    range_m: float
    azimuth_rad: float                         # CCW from world +x (east), at the transceiver
    elevation_rad: float                       # up-positive
    std_m: float                               # nominal 1-sigma horizontal position error for this geometry


def usbl_public_spec(cfg: USBLConfig) -> list[SpecItem]:
    return [
        SpecItem("type", "acoustic positioning, surface transceiver + vehicle locator"),
        SpecItem("update interval", cfg.period_s, "s"),
        SpecItem("latency", "1-2", "s", "fix time-stamped at the interrogation"),
        SpecItem("range accuracy", "0.2 % of slant range + 5 cm (1 sigma)"),
        SpecItem("bearing accuracy", "about 1", "deg (1 sigma)"),
        SpecItem("max range", cfg.max_range_m, "m"),
        SpecItem("limitations", "needs line of sight; multipath near the surface and structures gives outliers"),
    ]


class USBL:
    def __init__(self, m: mujoco.MjModel, rng: np.random.Generator, water: WaterColumn = SEAWATER_TEMPERATE,
                 site: str = "imu", config: Optional[USBLConfig] = None, hidden: Optional[USBLHidden] = None,
                 geomgroup: Optional[Sequence[int]] = None, t0: float = 0.0):
        self.cfg = config or USBLConfig()
        self.hid = hidden or USBLHidden()
        self.rng = rng
        self.water = water
        self.sid = site_id(m, site)
        root = int(m.body_rootid[m.site_bodyid[self.sid]])
        self.caster = RayCaster(m, geomgroup=geomgroup, bodyexclude=root)
        self.next_t = float(t0)
        self.queue = DelayLine()

    def update(self, t: float, d: mujoco.MjData) -> list[USBLFix]:
        while self.next_t <= t + 1e-12:
            fx = self._interrogate(self.next_t, d)
            if fx is not None:
                self.queue.push(fx.t_delivered, fx)
            jit = 1.0 + self.hid.period_jitter_frac * (2 * self.rng.random() - 1)
            self.next_t += self.cfg.period_s * jit
        return self.queue.pop_ready(t)

    def _interrogate(self, t: float, d: mujoco.MjData) -> Optional[USBLFix]:
        c, h = self.cfg, self.hid
        X = np.asarray(c.transceiver_pos, float)
        p, _ = site_pose(d, self.sid)
        v = p - X
        rho = float(np.linalg.norm(v))
        if p[2] >= 0 or rho > c.max_range_m or rho < 0.1:
            return None
        u = v / rho
        gid, dist, _ = self.caster.cast_one(d, X, u)
        blocked = gid >= 0 and 0 < dist < rho - 0.05
        outlier = False
        if blocked:
            if self.rng.random() > h.blocked_receive_prob:
                return None
            outlier = True
        else:
            pout = h.outlier_prob_shallow if p[2] > -2.0 else h.outlier_prob
            outlier = self.rng.random() < pout
        c_mean = self.water.mean_sound_speed(-X[2], -p[2])
        rng_m = rho * c.speed_of_sound_cfg / c_mean
        az = math.atan2(v[1], v[0]) + h.heading_misalign_rad
        el = math.asin(max(-1.0, min(1.0, v[2] / rho)))
        if outlier:
            rng_m += self.rng.uniform(*h.outlier_extra_path_m)
            az += self.rng.choice([-1, 1]) * self.rng.uniform(*h.outlier_angle_rad)
            el += self.rng.choice([-1, 1]) * self.rng.uniform(*h.outlier_angle_rad)
        rng_m += (h.range_sigma_frac * rho + h.range_sigma_m) * self.rng.standard_normal()
        az += h.angle_sigma_rad * self.rng.standard_normal()
        el += h.angle_sigma_rad * self.rng.standard_normal()
        pos = X + rng_m * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
        std = math.hypot(h.range_sigma_frac * rho + h.range_sigma_m, rho * math.radians(1.0))
        t_del = t + rho / c_mean + h.latency_s
        return USBLFix(t=t, t_delivered=t_del, valid=True, position=pos, range_m=rng_m, azimuth_rad=az,
                       elevation_rad=el, std_m=std)
