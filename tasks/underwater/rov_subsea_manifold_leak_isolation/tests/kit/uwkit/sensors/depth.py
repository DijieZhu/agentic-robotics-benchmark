"""Pressure / depth sensor (piezo-resistive, 30 bar class) and the firmware's depth conversion.

Truth: p = p_atm + rho_true g depth at site `depth` (WaterColumn).  Sensor: scale and offset error, temperature
coefficient acting on a sensor temperature that follows the water with a first-order lag, white noise, quantisation
to the 0.2 mbar resolution.  Firmware: depth = (p_meas - p_surface_cal) / (rho_cfg g), where p_surface_cal is the
pressure recorded at power-on at the surface (with its own small error) and rho_cfg the configured water density
(1000 fresh / 1025 salt): a wrong rho_cfg is a 2.5 % depth scale error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import mujoco

from ._util import RateScheduler, site_id, site_pose, SpecItem
from .acoustics import WaterColumn, SEAWATER_TEMPERATE


@dataclass
class DepthConfig:
    rate_hz: float = 20.0
    resolution_pa: float = 20.0           # 0.2 mbar
    rho_cfg: float = 1025.0               # firmware water-density setting
    g_cfg: float = 9.80665
    range_pa: float = 30e5


@dataclass
class DepthHidden:
    scale_error: float = 0.0
    offset_pa: float = 0.0                # residual after surface calibration (absolute error mostly cancels)
    noise_pa: float = 18.0                # 1 sigma (~1.8 mm of water)
    temp_coeff_pa_per_c: float = 3.0      # residual temperature sensitivity
    temp_tau_s: float = 25.0
    surface_cal_err_pa: float = 0.0
    temp_offset_c: float = 0.0
    temp_noise_c: float = 0.01

    @staticmethod
    def sample(rng: np.random.Generator) -> "DepthHidden":
        return DepthHidden(scale_error=float(rng.normal(0, 0.0007)), offset_pa=float(rng.normal(0, 15.0)),
                           temp_coeff_pa_per_c=float(rng.normal(0, 4.0)),
                           surface_cal_err_pa=float(rng.normal(0, 10.0)),
                           temp_offset_c=float(rng.normal(0, 0.5)))


@dataclass
class DepthReading:
    t: float
    valid: bool
    pressure_pa: float
    temperature_c: float
    depth_m: float                         # firmware conversion with rho_cfg


def depth_public_spec(cfg: DepthConfig) -> list[SpecItem]:
    return [
        SpecItem("type", "absolute pressure sensor, 30 bar class, with temperature"),
        SpecItem("range", f"0 .. {cfg.range_pa / 1e5:.0f}", "bar"),
        SpecItem("resolution", cfg.resolution_pa / 100, "mbar", "about 2 mm of water"),
        SpecItem("update rate", cfg.rate_hz, "Hz"),
        SpecItem("depth conversion", f"(p - p_surface) / (rho_cfg g), rho_cfg = {cfg.rho_cfg} kg/m3 (setting)"),
        SpecItem("absolute accuracy", "about +-0.15 % of depth + 5 mm", "",
                 "unit-to-unit (2 sigma) after the surface zero; a density setting that differs from the actual water "
                 "density adds the same relative error (1 % density = 1 % depth)"),
        SpecItem("temperature", "+-1 degC class, slow response"),
    ]


class DepthSensor:
    def __init__(self, m: mujoco.MjModel, rng: np.random.Generator, water: WaterColumn = SEAWATER_TEMPERATE,
                 site: str = "depth", config: Optional[DepthConfig] = None, hidden: Optional[DepthHidden] = None,
                 t0: float = 0.0):
        self.cfg = config or DepthConfig()
        self.hid = hidden or DepthHidden()
        self.rng = rng
        self.water = water
        self.sid = site_id(m, site)
        self.sched = RateScheduler(self.cfg.rate_hz, t0)
        self.T_sensor: Optional[float] = None
        self._tT: Optional[float] = None
        self.p_surface_cal = water.p_atm_pa + self.hid.surface_cal_err_pa
        self.last: Optional[DepthReading] = None

    def update(self, t: float, d: mujoco.MjData) -> list[DepthReading]:
        out = []
        while self.sched.due(t):
            tf = self.sched.fire()
            out.append(self._read(tf, d))
        if out:
            self.last = out[-1]
        return out

    def _read(self, t: float, d: mujoco.MjData) -> DepthReading:
        c, h = self.cfg, self.hid
        pos, _ = site_pose(d, self.sid)
        depth_true = max(-float(pos[2]), 0.0)
        Tw = self.water.temperature(depth_true)
        if self.T_sensor is None:
            self.T_sensor, self._tT = Tw, t
        dt = max(0.0, t - self._tT)
        self.T_sensor += (Tw - self.T_sensor) * (1.0 - math.exp(-dt / h.temp_tau_s))
        self._tT = t
        p_true = self.water.pressure_pa(depth_true) if pos[2] < 0 else self.water.p_atm_pa
        p = p_true * (1.0 + h.scale_error) + h.offset_pa + h.temp_coeff_pa_per_c * (self.T_sensor - 20.0)
        p += h.noise_pa * self.rng.standard_normal()
        p = round(p / c.resolution_pa) * c.resolution_pa
        p = min(max(p, 0.0), c.range_pa)
        T_out = self.T_sensor + h.temp_offset_c + h.temp_noise_c * self.rng.standard_normal()
        depth = (p - self.p_surface_cal) / (c.rho_cfg * c.g_cfg)
        return DepthReading(t=t, valid=True, pressure_pa=float(p), temperature_c=float(round(T_out, 2)),
                            depth_m=float(depth))
