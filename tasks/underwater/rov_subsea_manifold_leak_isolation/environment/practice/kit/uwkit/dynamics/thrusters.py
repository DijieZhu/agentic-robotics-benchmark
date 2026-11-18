"""Thruster model: tabulated steady-state curve + first-order lag + saturation + current + inflow loss.

Command interface: normalised command u in [-1, 1]  <->  PWM = pwm_center + u * pwm_half_span
(1100-1900 us).  The steady-state thrust, rotor speed and current come from the manufacturer's
bollard table (models/thruster_curve.csv, 6 supply voltages x 201 PWM points); between table
voltages the columns are blended linearly, outside 10-20 V they are clamped.  Deadband and the
forward / reverse asymmetry are those of the table (no extra parameters).

Dynamics: the effective command u_eff follows u_cmd with a first-order lag (tau_up while the rotor
speeds up, tau_down while it slows down or reverses).  Because the table rotor speed is close to
linear in PWM outside the deadband, this is effectively a first-order rotor-speed lag; thrust
(~ n^2) therefore rises more slowly than 63 % at t = tau.

Optional advance-ratio loss (default on): T = T_bollard(u_eff, V) * clip(1 - s*u_a / (r * P_g * n),
0, f_max), u_a = axial water-relative speed of the thruster along its positive direction, s = sign
of the thrust, n = rotor speed [rev/s], P_g = geometric pitch, r = inflow_zero_ratio.
"""
from __future__ import annotations

import csv
import math
import os

import numpy as np

from ..params import MODELS_DIR, ThrusterModelSpec

KGF = 9.80665


class ThrusterCurve:
    """Bilinear (PWM x voltage) lookup of steady-state thrust [N], signed rpm and current [A]."""

    def __init__(self, path=None):
        path = path or os.path.join(MODELS_DIR, "thruster_curve.csv")
        rows = []
        with open(path) as f:
            for r in csv.DictReader(line for line in f if not line.startswith("#")):
                rows.append((float(r["voltage_V"]), float(r["pwm_us"]), float(r["rpm"]),
                             float(r["current_A"]), float(r["force_kgf"])))
        arr = np.array(rows)
        self.voltages = np.unique(arr[:, 0])
        self.pwm = np.unique(arr[:, 1])
        nv, npwm = len(self.voltages), len(self.pwm)
        if arr.shape[0] != nv * npwm:
            raise ValueError("thruster table is not a full PWM x voltage grid")
        arr = arr[np.lexsort((arr[:, 1], arr[:, 0]))]
        F = arr[:, 4].reshape(nv, npwm) * KGF
        rpm = arr[:, 2].reshape(nv, npwm)
        cur = arr[:, 3].reshape(nv, npwm)
        mid = int(np.argmin(np.abs(self.pwm - 1500.0)))
        rpm = np.where(self.pwm[None, :] < self.pwm[mid], -rpm, rpm)
        # isotonic clean-up (the raw table has a few 0.01-0.5 % non-monotone points at the ends):
        # going outward from the stop point, |value| may only grow.
        for X in (F, rpm):
            X[:, mid:] = np.maximum.accumulate(X[:, mid:], axis=1)
            X[:, :mid + 1] = np.minimum.accumulate(X[:, :mid + 1][:, ::-1], axis=1)[:, ::-1]
        self.F, self.rpm, self.current = F, rpm, cur
        self.mid = mid
        self.pwm0 = float(self.pwm[0])
        self.dpwm = float(self.pwm[1] - self.pwm[0])
        if not np.allclose(np.diff(self.pwm), self.dpwm):
            raise ValueError("PWM grid must be uniform")
        self.v0 = float(self.voltages[0])
        self.dv = np.diff(self.voltages)
        self._flat = np.stack([F, rpm, cur])           # (3, nv, npwm)
        self.npwm = npwm
        self.pwm_max = float(self.pwm[-1])
        self._cV = None
        self._col = None
        self._inv_V = None
        self._vlo, self._vhi = float(self.voltages[0]), float(self.voltages[-1])
        self._vstep = float(self.voltages[1] - self.voltages[0])
        self._vuniform = bool(np.allclose(np.diff(self.voltages), self._vstep))
        u = np.abs(self.pwm - self.pwm[mid]) / (self.pwm[-1] - self.pwm[mid])
        u[mid] = 1.0
        self._k_current = np.max(cur / u[None, :], axis=1)      # max I/|u| per table voltage

    def current_slope_bound(self, V):
        """k(V) with steady current <= k(V) * |u| for every command u (exact on this curve)."""
        j, w = self._vweights(V)
        return float((1 - w) * self._k_current[j] + w * self._k_current[j + 1])

    # ------------------------------------------------------------------ lookups
    def _vweights(self, V):
        vs = self.voltages
        V = min(max(float(V), self._vlo), self._vhi)
        if self._vuniform:
            x = (V - self._vlo) / self._vstep
            j = min(int(x), len(vs) - 2)
            return j, x - j
        j = min(int(np.searchsorted(vs, V, side="right") - 1), len(vs) - 2)
        return j, (V - vs[j]) / (vs[j + 1] - vs[j])

    def column(self, V):
        """(F, rpm, current) columns [3, npwm] at supply voltage V."""
        j, w = self._vweights(V)
        return (1 - w) * self._flat[:, j] + w * self._flat[:, j + 1]

    def steady(self, pwm_us, V):
        """Steady thrust [N], signed rpm, current [A] for PWM array at voltage V (vectorised)."""
        if V != self._cV:
            self._cV = V
            self._col = self.column(V)
        x = (np.minimum(np.maximum(np.asarray(pwm_us, float), self.pwm0), self.pwm_max) - self.pwm0) / self.dpwm
        i = np.minimum(x.astype(np.intp), self.npwm - 2)
        a = x - i
        c = self._col
        lo = c[:, i]
        return tuple(lo + a * (c[:, i + 1] - lo))

    def limits(self, V):
        """(max forward thrust, max reverse thrust magnitude) [N] at voltage V."""
        F = self.column(V)[0]
        return float(F[-1]), float(-F[0])

    def _inverse_tables(self, V):
        if self._inv_V != V:
            F = self.column(V)[0]
            mid = self.mid
            up, pw_up = F[mid:], self.pwm[mid:]
            k = int(np.argmax(up > 0.0))                      # first strictly positive point
            xu = np.concatenate([[0.0], up[k:]])
            yu = np.concatenate([[pw_up[max(k - 1, 0)]], pw_up[k:]])
            dn, pw_dn = F[:mid + 1][::-1], self.pwm[:mid + 1][::-1]
            kd = int(np.argmax(dn < 0.0))
            xd = np.concatenate([[0.0], -dn[kd:]])
            yd = np.concatenate([[pw_dn[max(kd - 1, 0)]], pw_dn[kd:]])
            self._inv_V, self._inv = V, (xu, yu, xd, yd)
        return self._inv

    def inverse(self, force_N, V):
        """PWM [us] that produces the requested steady thrust at voltage V (firmware linearisation).

        Zero force -> stop PWM; tiny forces map to the deadband edge; out-of-range forces saturate."""
        xu, yu, xd, yd = self._inverse_tables(V)
        f = np.asarray(force_N, float)
        pu = np.interp(np.maximum(f, 0.0), xu, yu)
        pd = np.interp(np.maximum(-f, 0.0), xd, yd)
        return np.where(f > 0, pu, np.where(f < 0, pd, float(self.pwm[self.mid])))


class ThrusterBank:
    """n thrusters sharing one curve.  State: effective normalised command u_eff."""

    def __init__(self, n, model: ThrusterModelSpec, curve: ThrusterCurve | None = None):
        self.n = int(n)
        self.spec = model
        self.curve = curve or ThrusterCurve()
        self.u_eff = np.zeros(self.n)
        self.u_cmd = np.zeros(self.n)
        self.thrust = np.zeros(self.n)
        self.rpm = np.zeros(self.n)
        self.current = np.zeros(self.n)
        self._dt = None
        self.enabled = np.ones(self.n, bool)          # fault injection hook (False = dead thruster)

    def u_to_pwm(self, u):
        return self.spec.pwm_center_us + np.minimum(np.maximum(u, -1.0), 1.0) * self.spec.pwm_half_span_us

    def pwm_to_u(self, pwm):
        return np.clip((np.asarray(pwm, float) - self.spec.pwm_center_us) / self.spec.pwm_half_span_us, -1, 1)

    def _gains(self, dt):
        if dt != self._dt:
            self._dt = dt
            self._k_up = 1.0 - math.exp(-dt / self.spec.tau_up_s)
            self._k_dn = 1.0 - math.exp(-dt / self.spec.tau_down_s)
        return self._k_up, self._k_dn

    def steady_current(self, u, V):
        return self.curve.steady(self.u_to_pwm(u), V)[2]

    def update(self, u_cmd, V, dt, u_axial=None):
        """Advance the lag by dt and return the thrust [N] of every thruster (along its dir)."""
        u = np.minimum(np.maximum(np.asarray(u_cmd, float), -1.0), 1.0) * self.enabled
        self.u_cmd = u
        k_up, k_dn = self._gains(dt)
        spin_up = (np.abs(u) > np.abs(self.u_eff)) & (u * self.u_eff >= 0.0)
        self.u_eff = self.u_eff + np.where(spin_up, k_up, k_dn) * (u - self.u_eff)
        T, rpm, cur = self.curve.steady(self.u_to_pwm(self.u_eff), V)
        if self.spec.inflow_loss and u_axial is not None:
            n_rps = np.abs(rpm) / 60.0
            ref = self.spec.inflow_zero_ratio * self.spec.geometric_pitch_m * np.maximum(n_rps, 1e-6)
            fac = np.minimum(np.maximum(1.0 - np.sign(T) * u_axial / ref, 0.0), self.spec.inflow_factor_max)
            T = np.where(n_rps > 1.0, T * fac, T)
        self.thrust, self.rpm, self.current = T, rpm, cur
        return T

    def reset(self):
        self.u_eff[:] = 0.0
        self.u_cmd[:] = 0.0
        self.thrust[:] = 0.0
        self.rpm[:] = 0.0
        self.current[:] = 0.0
