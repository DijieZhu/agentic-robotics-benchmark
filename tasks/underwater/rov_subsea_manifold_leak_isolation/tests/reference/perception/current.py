"""Current (water velocity) estimates from the vehicle's own public state (numpy only; DESIGN 2.2 H3, 2.4 (e)).

Two methods, both on lists of `state()` replies (keys: t, quat_wxyz, nav_velocity_m_s, wrench_groups, dvl_lock):

  drift_fit(states)
      the vehicle is let go horizontally (wrench mode with zero horizontal force; depth and attitude held by the
      firmware): its velocity over ground (DVL navigation) relaxes to the water velocity.  Fits
      v(t) = v_inf + (v_0 - v_inf) exp(-(t - t_0) / tau) to both horizontal components with one tau (grid search,
      linear least squares inside).  Needs DVL lock and room to drift (0.3 m/s x 20 s = 6 m: away from the panel).
  hold_force(states, cal)
      the vehicle holds station (set_hold): the mean commanded horizontal force (depth + translation groups, body
      -> world with the attitude) balances the drag.  `cal` (DragCalibration) inverts force -> relative speed per
      body axis; it is fitted from training-tank runs at known current speeds (same vehicle, same water: decision U1)
      with `DragCalibration.fit`.  The current points opposite to the holding force.

    v_w, info = drift_fit(states)                    # (2,) world horizontal current, dict(tau, rms, n)
    cal = DragCalibration.fit(samples)               # samples: [(v_rel_body (2,), F_body (2,))]
    v_w, info = hold_force(states, cal)
"""
from __future__ import annotations

import math

import numpy as np


def _quat_mat(q):
    w, x, y, z = (float(v) for v in q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _series(states, need_lock=True):
    t, v = [], []
    for s in states:
        vel = s.get("nav_velocity_m_s")
        if vel is None or (need_lock and not s.get("dvl_lock", False)):
            continue
        t.append(float(s["t"]))
        v.append([float(vel[0]), float(vel[1])])
    return np.asarray(t, float), np.asarray(v, float).reshape(-1, 2)


def drift_fit(states, taus=None, need_lock=True) -> tuple:
    """Free-drift fit (see the module docstring).  Returns (v_inf (2,) or None, info)."""
    t, v = _series(states, need_lock)
    if len(t) < 10:
        return None, dict(n=int(len(t)), reason="too few locked samples")
    t = t - t[0]
    taus = np.geomspace(0.5, 30.0, 60) if taus is None else np.asarray(taus, float)
    best = None
    for tau in taus:
        e = np.exp(-t / tau)
        A = np.column_stack([1.0 - e, e])            # v = v_inf (1 - e) + v_0 e
        sol, *_ = np.linalg.lstsq(A, v, rcond=None)
        r = v - A @ sol
        rms = float(np.sqrt(np.mean(r ** 2)))
        if best is None or rms < best[0]:
            best = (rms, float(tau), sol)
    rms, tau, sol = best
    return sol[0].copy(), dict(n=int(len(t)), tau_s=tau, rms_m_s=rms, v0=sol[1].tolist(),
                               settled=bool(t[-1] > 2.5 * tau))


def mean_hold_force_w(states, groups=("depth", "translation")) -> tuple:
    """Mean commanded force in world axes over a hold (the firmware's group wrenches, body axes, rotated by the
    attitude).  Returns (F_w (3,), F_b_mean (3,), n)."""
    acc_w, acc_b, n = np.zeros(3), np.zeros(3), 0
    for s in states:
        wg = s.get("wrench_groups") or {}
        q = s.get("quat_wxyz")
        if q is None or not all(g in wg and wg[g] is not None for g in groups):
            continue
        Fb = sum(np.asarray(wg[g][:3], float) for g in groups)
        acc_b += Fb
        acc_w += _quat_mat(q) @ Fb
        n += 1
    if n == 0:
        return None, None, 0
    return acc_w / n, acc_b / n, n


class DragCalibration:
    """Per body axis (surge x, sway y): F = c1 v + c2 v |v| (relative water speed v along that axis), fitted from
    tank holds at known current speeds and directions."""

    def __init__(self, c1=(0.0, 0.0), c2=(0.0, 0.0)):
        self.c1 = np.asarray(c1, float)
        self.c2 = np.asarray(c2, float)

    @classmethod
    def fit(cls, samples) -> "DragCalibration":
        """samples: [(v_rel_body (2,) = water velocity relative to the vehicle in body axes, F_body (2,) = the mean
        holding force)].  The drag on the vehicle is along v_rel; the holding force balances it: F = -drag."""
        c1, c2 = [], []
        V = np.asarray([s[0] for s in samples], float)
        F = np.asarray([s[1] for s in samples], float)
        for k in range(2):
            v = V[:, k]
            A = np.column_stack([v, v * np.abs(v)])
            sol, *_ = np.linalg.lstsq(A, -F[:, k], rcond=None)
            c1.append(sol[0])
            c2.append(sol[1])
        return cls(c1, c2)

    def drag(self, v_rel_b) -> np.ndarray:
        v = np.asarray(v_rel_b, float)[:2]
        return self.c1 * v + self.c2 * v * np.abs(v)

    def invert(self, F_hold_b) -> np.ndarray:
        """Relative water velocity (body x, y) whose drag the holding force F_hold_b balances."""
        D = -np.asarray(F_hold_b, float)[:2]
        out = np.zeros(2)
        for k in range(2):
            a, b, d = self.c2[k], self.c1[k], abs(D[k])
            if a > 1e-9:
                s = (-b + math.sqrt(b * b + 4 * a * d)) / (2 * a)
            elif b > 1e-9:
                s = d / b
            else:
                s = 0.0
            out[k] = math.copysign(s, D[k])
        return out


def hold_force(states, cal: DragCalibration) -> tuple:
    """Current from a station hold (see the module docstring).  Returns (v_w (2,) or None, info)."""
    F_w, F_b, n = mean_hold_force_w(states)
    if n == 0:
        return None, dict(n=0, reason="no state with wrench groups")
    q = [s for s in states if s.get("quat_wxyz") is not None][-1]["quat_wxyz"]
    R = _quat_mat(q)
    v_b = cal.invert(F_b)
    v_w = (R @ np.r_[v_b, 0.0])[:2]
    return v_w, dict(n=n, F_w=F_w.tolist(), F_b=F_b.tolist(), v_rel_b=v_b.tolist())
