"""Battery pack: OCV(SoC) + series resistance + one RC polarisation branch, energy integral,
continuous / burst current budget.  Pure numpy-free scalar model (one pack per vehicle).

Terminal voltage  V = n_s * OCV_cell(SoC) - I * R0 - v1,     dv1/dt = (I * R1 - v1) / tau1
Charge            dQ/dt = I   (Ah used; SoC = 1 - Q / C)
Energy            E_out = integral V * I dt  [J]
Current budget    I_allow = i_burst while the burst budget b > 0, else i_continuous;
                  db/dt = -1 while I > i_continuous, +0.5 otherwise, b in [0, burst_s].
The pack itself never cuts out (the reference packs have no BMS); flags record abuse.
"""
from __future__ import annotations

import math

import numpy as np

from ..params import BatterySpec


class Battery:
    def __init__(self, spec: BatterySpec, soc0: float = 1.0):
        self.spec = spec
        self._soc_tab = np.asarray(spec.ocv_soc, float)
        self._ocv_tab = np.asarray(spec.ocv_cell_V, float) * spec.series_cells
        self.reset(soc0)

    def reset(self, soc0=1.0):
        self.q_used_Ah = (1.0 - float(soc0)) * self.spec.capacity_Ah
        self.v1 = 0.0
        self.current = 0.0
        self.energy_out_J = 0.0
        self.charge_out_C = 0.0
        self.burst_budget_s = self.spec.burst_s
        self.time_over_continuous_s = 0.0
        self.time_over_burst_s = 0.0
        self.time_below_cutoff_s = 0.0
        self.peak_current = 0.0
        self.voltage = self.ocv()

    @property
    def soc(self):
        return max(0.0, 1.0 - self.q_used_Ah / self.spec.capacity_Ah)

    def ocv(self):
        return float(np.interp(self.soc, self._soc_tab, self._ocv_tab))

    def terminal_voltage(self, current):
        return self.ocv() - current * self.spec.r0_ohm - self.v1

    def allowed_current(self):
        return self.spec.i_burst_A if self.burst_budget_s > 0.0 else self.spec.i_continuous_A

    def step(self, current, dt):
        """Draw `current` [A] for dt [s]; returns the terminal voltage at the end of the step."""
        s = self.spec
        I = float(current)
        V = self.terminal_voltage(I)
        self.energy_out_J += V * I * dt
        self.charge_out_C += I * dt
        self.q_used_Ah += I * dt / 3600.0
        a = math.exp(-dt / s.tau1_s)
        self.v1 = a * self.v1 + (1 - a) * I * s.r1_ohm
        if I > s.i_continuous_A:
            self.burst_budget_s = max(0.0, self.burst_budget_s - dt)
            self.time_over_continuous_s += dt
        else:
            self.burst_budget_s = min(s.burst_s, self.burst_budget_s + 0.5 * dt)
        if I > s.i_burst_A:
            self.time_over_burst_s += dt
        self.current = I
        self.peak_current = max(self.peak_current, I)
        self.voltage = self.terminal_voltage(I)
        if self.voltage < s.v_cutoff_V:
            self.time_below_cutoff_s += dt
        return self.voltage

    def status(self):
        return dict(voltage_V=self.voltage, current_A=self.current, soc=self.soc,
                    energy_out_Wh=self.energy_out_J / 3600.0, q_used_Ah=self.q_used_Ah,
                    burst_budget_s=self.burst_budget_s, time_over_continuous_s=self.time_over_continuous_s,
                    time_over_burst_s=self.time_over_burst_s, time_below_cutoff_s=self.time_below_cutoff_s,
                    peak_current_A=self.peak_current, below_recover=self.voltage < self.spec.v_recover_V)
