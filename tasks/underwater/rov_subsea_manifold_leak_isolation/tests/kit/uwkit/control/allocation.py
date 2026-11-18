"""Thrust allocation with saturation handling (prioritised, direction-preserving group scaling).

B (6 x n) maps per-thruster force [N] (along each thruster's positive direction) to the body
wrench [Fx Fy Fz Mx My Mz] about the reference point (CG).  P = B^+ (min-norm pseudo-inverse).

allocate(groups, V): groups = wrench vectors in priority order, e.g.
    [tau_attitude, tau_depth, tau_translation]
f = 0; for each group g: f_g = P tau_g; alpha_g = largest value in [0, 1] with
lo <= f + alpha_g f_g <= hi (per thruster, lo = -T_rev(V), hi = T_fwd(V)); f += alpha_g f_g.
Each group keeps its direction (no cross-axis distortion inside a group); a lower-priority group
only uses the thrust margin left by the higher ones.  This mirrors the usual open-source ROV mixer
(roll / pitch / yaw first, then linear terms scaled into the remaining headroom) but works in
newtons with the asymmetric forward / reverse limits of the curve at the measured voltage.
Thrust -> command uses the inverse of the steady-state curve at the measured voltage.
"""
from __future__ import annotations

import numpy as np

from ..dynamics.thrusters import ThrusterCurve
from ..params import ThrusterModelSpec, VehicleSpec


def allocation_matrix(pos, dirs, ref_point):
    P, D = np.asarray(pos, float), np.asarray(dirs, float)
    return np.vstack([D.T, np.cross(P - np.asarray(ref_point, float), D).T])


class Allocator:
    def __init__(self, B, curve: ThrusterCurve | None = None, thruster_model: ThrusterModelSpec | None = None,
                 limit_margin=1.0):
        self.B = np.asarray(B, float)
        self.n = self.B.shape[1]
        self.P = np.linalg.pinv(self.B)
        self.rank = int(np.linalg.matrix_rank(self.B))
        self.curve = curve or ThrusterCurve()
        self.tm = thruster_model or ThrusterModelSpec()
        self.margin = float(limit_margin)
        self._V = None

    @classmethod
    def from_vehicle(cls, v: VehicleSpec, **kw):
        return cls(v.allocation_matrix(), thruster_model=v.thruster_model, **kw)

    def limits(self, V):
        if V != self._V:
            fwd, rev = self.curve.limits(V)
            self._lim = (-rev * self.margin, fwd * self.margin)
            self._V = V
        return self._lim

    def scale_groups(self, groups, V):
        lo, hi = self.limits(V)
        FG = self.P @ np.asarray(groups, float).T             # (n, n_groups)
        f = np.zeros(self.n)
        alphas = []
        for k in range(FG.shape[1]):
            fg = FG[:, k]
            room = np.where(fg > 0.0, hi - f, lo - f)
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(np.abs(fg) > 1e-12, room / fg, np.inf)
            a = min(1.0, float(r.min()))
            a = a if a > 0.0 else 0.0
            f = f + a * fg
            alphas.append(a)
        return f, alphas

    def thrust_to_command(self, f, V):
        pwm = self.curve.inverse(f, V)
        u = (pwm - self.tm.pwm_center_us) / self.tm.pwm_half_span_us
        return np.minimum(np.maximum(u, -1.0), 1.0)

    def allocate(self, groups, V):
        """Returns (u_cmd (n,), info) with info = dict(f=thrusts [N], alpha=[...], wrench=B f)."""
        f, alphas = self.scale_groups(groups, V)
        u = self.thrust_to_command(f, V)
        return u, dict(f=f, alpha=alphas, wrench=self.B @ f)

    def axis_capacity(self, V):
        """Direction-preserving maximum of each pure axis wrench (+ and -), shape (6, 2)."""
        out = np.zeros((6, 2))
        for i in range(6):
            for k, s in enumerate((1.0, -1.0)):
                e = np.zeros(6)
                e[i] = s * 1e4
                f, a = self.scale_groups([e], V)
                out[i, k] = a[0] * 1e4
        return out
