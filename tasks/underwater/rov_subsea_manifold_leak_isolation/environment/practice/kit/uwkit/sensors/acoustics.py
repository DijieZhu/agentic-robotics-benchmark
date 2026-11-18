"""Water acoustics and density helpers (numpy only).

* sound_speed: Coppens (1981), valid 0-35 degC, 0-45 PSU, 0-4000 m.  Check values: fresh 20 degC 1482.4 m/s,
  fresh 5 degC 1426.1 m/s, sea (35 PSU) 5 degC 1470.6 m/s (TEOS-10 via gsw gives 1482.3 / 1426.3 / 1470.7).
* absorption_db_per_m: Ainslie & McColm (1998) (boric acid + MgSO4 + pure water).  750 kHz, 10 degC, 35 PSU,
  pH 8, surface: 0.238 dB/m.
* density_kg_m3: EOS-80 one-atmosphere equation of state (UNESCO 1981); compressibility ignored (+0.45 kg/m3 per
  100 m, i.e. 0.04 %).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def sound_speed(T_c: float, S_psu: float, depth_m: float = 0.0) -> float:
    t = T_c / 10.0
    D = max(depth_m, 0.0) / 1000.0
    S = S_psu
    c0 = 1449.05 + 45.7 * t - 5.21 * t ** 2 + 0.23 * t ** 3 + (1.333 - 0.126 * t + 0.009 * t ** 2) * (S - 35.0)
    dD = (16.23 + 0.253 * t) * D + (0.213 - 0.1 * t) * D ** 2 + (0.016 + 0.0002 * (S - 35.0)) * (S - 35.0) * t * D
    return float(c0 + dD)


def absorption_db_per_m(f_hz: float, T_c: float = 10.0, S_psu: float = 35.0, depth_m: float = 0.0,
                        pH: float = 8.0) -> float:
    f = f_hz / 1000.0          # kHz
    z = max(depth_m, 0.0) / 1000.0  # km
    f1 = 0.78 * math.sqrt(max(S_psu, 0.0) / 35.0) * math.exp(T_c / 26.0)
    f2 = 42.0 * math.exp(T_c / 17.0)
    boric = 0.106 * f1 * f * f / (f1 * f1 + f * f) * math.exp((pH - 8.0) / 0.56)
    mgso4 = 0.52 * (1.0 + T_c / 43.0) * (S_psu / 35.0) * f2 * f * f / (f2 * f2 + f * f) * math.exp(-z / 6.0)
    water = 0.00049 * f * f * math.exp(-(T_c / 27.0 + z / 17.0))
    return float((boric + mgso4 + water) / 1000.0)   # dB/km -> dB/m


def density_kg_m3(T_c: float, S_psu: float) -> float:
    T = T_c
    S = max(S_psu, 0.0)
    rw = (999.842594 + 6.793952e-2 * T - 9.095290e-3 * T ** 2 + 1.001685e-4 * T ** 3 - 1.120083e-6 * T ** 4
          + 6.536332e-9 * T ** 5)
    A = 0.824493 - 4.0899e-3 * T + 7.6438e-5 * T ** 2 - 8.2467e-7 * T ** 3 + 5.3875e-9 * T ** 4
    B = -5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T ** 2
    return float(rw + A * S + B * S ** 1.5 + 4.8314e-4 * S * S)


@dataclass(frozen=True)
class WaterColumn:
    """Bulk water properties of a (hidden) site.  Homogeneous; a stratified site can subclass and override the
    methods (they take depth as an argument for that reason)."""
    temperature_c: float = 12.0
    salinity_psu: float = 35.0
    p_atm_pa: float = 101325.0
    g: float = 9.80665

    def density(self, depth_m: float = 0.0) -> float:
        return density_kg_m3(self.temperature_c, self.salinity_psu)

    def temperature(self, depth_m: float = 0.0) -> float:
        return self.temperature_c

    def sound_speed(self, depth_m: float = 0.0) -> float:
        return sound_speed(self.temperature_c, self.salinity_psu, depth_m)

    def mean_sound_speed(self, depth_a: float, depth_b: float) -> float:
        return 0.5 * (self.sound_speed(depth_a) + self.sound_speed(depth_b))

    def absorption_db_per_m(self, f_hz: float, depth_m: float = 0.0) -> float:
        return absorption_db_per_m(f_hz, self.temperature_c, self.salinity_psu, depth_m)

    def pressure_pa(self, depth_m: float) -> float:
        """Absolute pressure at depth (homogeneous density)."""
        return self.p_atm_pa + self.density() * self.g * max(depth_m, 0.0)


SEAWATER_TEMPERATE = WaterColumn(12.0, 35.0)
FRESHWATER_POOL = WaterColumn(20.0, 0.0)
