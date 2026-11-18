"""On-board navigation estimate (vehicle firmware, given to the agent as is): DVL + AHRS + depth dead reckoning.

  * Velocity: the DVL bottom-track velocity (DVL frame) is rotated to the body frame with the DVL mounting rotation,
    the lever-arm term w x r_dvl is removed (w from the AHRS), and the result is rotated to the world frame with the
    AHRS attitude.
  * Position: x, y integrate the world velocity; z = -depth from the depth sensor (firmware conversion).
  * Lock loss: the last world velocity is held (status HOLD, then STALE after `stale_after_s`), and the velocity
    uncertainty grows as accel_unc * tau, i.e. the position variance grows like (accel_unc tau^2 / 2)^2.
  * Covariance (x, y): a random-walk part (DVL noise, correlated over one ping interval) plus a systematic part
    (assumed heading-bias + scale error times the distance travelled since the last reset).  This is a *nominal*
    figure from the firmware's assumptions, not the true error: near steel the real heading error can be far larger.
  * USBL fixes are not fused here (they are provided raw); the agent may call `reset()` with its own fix.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import mujoco

from ..sensors._util import quat_to_mat, mat_to_quat, euler_zyx_from_mat, site_id, site_pose, SpecItem


@dataclass
class NavConfig:
    rate_hz: float = 10.0
    stale_after_s: float = 5.0
    accel_unc_m_s2: float = 0.05
    heading_bias_sigma_rad: float = math.radians(1.0)
    scale_sigma: float = 0.005
    depth_sigma_m: float = 0.02
    depth_sigma_frac: float = 0.005
    initial_sigma_m: float = 0.0


@dataclass
class NavSolution:
    t: float
    valid: bool
    position: np.ndarray            # (3,) world (x, y from dead reckoning, z = -depth)
    velocity: np.ndarray            # (3,) world
    quat: np.ndarray                # AHRS attitude (w, x, y, z)
    heading_deg: float
    altitude: float                 # DVL altitude (NaN if unknown)
    cov_pos: np.ndarray             # (3, 3)
    dvl_lock: bool
    status: str                     # "LOCK", "HOLD", "STALE", "INIT"
    time_since_lock: float
    distance_since_reset: float
    omega_b: np.ndarray = None      # AHRS bias-corrected body rate (rad/s)

    def to_nav_estimate(self):
        """Adapter to the autopilot's NavEstimate (uwkit.control.autopilot): attitude, body rate, depth, body-axes
        velocity over ground (None before the first DVL lock), world position."""
        from ..control.autopilot import NavEstimate
        R = quat_to_mat(self.quat)
        vel_b = None if self.status == "INIT" else R.T @ self.velocity
        return NavEstimate(t=self.t, quat_wb=self.quat.copy(), omega_b=np.asarray(self.omega_b).copy(),
                           depth=float(-self.position[2]), vel_b=vel_b, pos_w=self.position.copy())


def nav_public_spec(cfg: NavConfig) -> list[SpecItem]:
    return [
        SpecItem("type", "dead-reckoning navigation estimate (DVL + AHRS + depth), no external fixes"),
        SpecItem("output rate", cfg.rate_hz, "Hz"),
        SpecItem("on DVL lock loss", f"holds the last velocity; status HOLD, STALE after {cfg.stale_after_s} s"),
        SpecItem("covariance", "nominal, from DVL noise, assumed heading/scale errors and hold time"),
        SpecItem("reset", "position can be reset by the user program"),
    ]


class DeadReckoning:
    def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, config: Optional[NavConfig] = None,
                 dvl_site: str = "dvl", body: Optional[str] = None, t0: float = 0.0,
                 initial_position: Optional[Sequence[float]] = None):
        """The firmware knows its own geometry: DVL lever arm and mounting rotation relative to the vehicle body
        (taken from the model at construction).  initial_position: the surface launch fix (x, y)."""
        self.cfg = config or NavConfig()
        sid = site_id(m, dvl_site)
        bid = int(m.body_rootid[m.site_bodyid[sid]]) if body is None else m.body(body).id
        pb, Rb = d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy()
        ps, Rs = site_pose(d, sid)
        self.r_dvl_b = Rb.T @ (ps - pb)
        self.R_bd = Rb.T @ Rs
        self.body_id = bid
        self.t = float(t0)
        self.next_out = float(t0)
        p0 = np.zeros(2) if initial_position is None else np.asarray(initial_position[:2], float)
        self.xy = p0.copy()
        self.z = 0.0
        self.v_w = np.zeros(3)
        self.Sv = np.eye(3) * 0.05 ** 2
        self.P_rw = np.eye(2) * self.cfg.initial_sigma_m ** 2
        self.dist = 0.0
        self.R_wb = np.eye(3)
        self.omega_b = np.zeros(3)
        self.heading = 0.0
        self.alt = float("nan")
        self.have_att = False
        self.have_lock = False
        self.t_lock = None
        self.t_lost: Optional[float] = None
        self.dvl_period = 0.1
        self._t_last_dvl = None

    # ----------------------------------------------------------------------------------------------------
    def reset(self, position_xy: Sequence[float], sigma_m: float = 0.0) -> None:
        self.xy = np.asarray(position_xy[:2], float).copy()
        self.P_rw = np.eye(2) * float(sigma_m) ** 2
        self.dist = 0.0

    def _hold_var(self, t: float) -> float:
        if self.t_lost is None:
            return 0.0
        tau = max(0.0, t - self.t_lost)
        return (0.5 * self.cfg.accel_unc_m_s2 * tau * tau) ** 2

    def _integrate(self, t: float) -> None:
        dt = t - self.t
        if dt <= 0:
            return
        self.xy = self.xy + self.v_w[:2] * dt
        self.dist += float(np.linalg.norm(self.v_w[:2])) * dt
        if self.have_lock:
            self.P_rw = self.P_rw + self.Sv[:2, :2] * dt * self.dvl_period
        self.t = t

    def _on_dvl(self, r) -> None:
        if r.valid and self.have_att:
            if self.t_lost is not None:                     # bake the accumulated hold error into P
                self.P_rw = self.P_rw + np.eye(2) * self._hold_var(r.t)
                self.t_lost = None
            v_b = self.R_bd @ np.asarray(r.velocity) - np.cross(self.omega_b, self.r_dvl_b)
            self.v_w = self.R_wb @ v_b
            C = self.R_wb @ self.R_bd
            S = C @ np.asarray(r.covariance) @ C.T
            J = np.array([-self.v_w[1], self.v_w[0], 0.0])   # d(Rz(psi) v)/d psi
            S = S + np.outer(J, J) * math.radians(0.5) ** 2
            self.Sv = S
            if self._t_last_dvl is not None:
                self.dvl_period = min(max(r.t - self._t_last_dvl, 0.02), 1.0)
            self._t_last_dvl = r.t
            self.have_lock = True
            self.t_lock = r.t
            if np.isfinite(r.altitude):
                self.alt = float(r.altitude)
        elif not r.valid:
            if self.have_lock or self.t_lost is None:
                if self.t_lost is None:
                    self.t_lost = r.t
            self.have_lock = False
            self.alt = float("nan") if not np.isfinite(r.altitude) else float(r.altitude)

    def _on_ahrs(self, a) -> None:
        self.R_wb = quat_to_mat(a.quat)
        self.omega_b = np.asarray(a.angular_velocity)
        self.heading = float(a.heading_deg)
        self.have_att = True
        self._quat = np.asarray(a.quat)

    def _on_depth(self, dr) -> None:
        self.z = -float(dr.depth_m)

    def update(self, t: float, dvl=(), ahrs=(), depth=()) -> list[NavSolution]:
        ev = [(r.t, 0, r) for r in ahrs] + [(r.t, 1, r) for r in depth] + [(r.t, 2, r) for r in dvl]
        ev.sort(key=lambda e: (e[0], e[1]))
        out = []
        for te, kind, r in ev:
            while self.next_out <= te:
                self._integrate(self.next_out)
                out.append(self._solution(self.next_out))
                self.next_out += 1.0 / self.cfg.rate_hz
            self._integrate(max(te, self.t))
            (self._on_ahrs, self._on_depth, self._on_dvl)[kind](r)
        while self.next_out <= t + 1e-12:
            self._integrate(self.next_out)
            out.append(self._solution(self.next_out))
            self.next_out += 1.0 / self.cfg.rate_hz
        return out

    def _solution(self, t: float) -> NavSolution:
        c = self.cfg
        sys_sd = (c.heading_bias_sigma_rad + c.scale_sigma) * self.dist
        P2 = self.P_rw + np.eye(2) * (sys_sd ** 2 + self._hold_var(t))
        cov = np.zeros((3, 3))
        cov[:2, :2] = P2
        cov[2, 2] = (c.depth_sigma_m + c.depth_sigma_frac * abs(self.z)) ** 2
        if self.have_lock:
            status = "LOCK"
        elif self.t_lost is not None:
            status = "HOLD" if t - self.t_lost < c.stale_after_s else "STALE"
        else:
            status = "INIT"
        tsl = 0.0 if self.have_lock else (t - self.t_lock if self.t_lock is not None else float("inf"))
        q = getattr(self, "_quat", np.array([1.0, 0, 0, 0]))
        return NavSolution(t=t, valid=self.have_att, position=np.array([self.xy[0], self.xy[1], self.z]),
                           velocity=self.v_w.copy(), quat=q.copy(), heading_deg=self.heading, altitude=self.alt,
                           cov_pos=cov, dvl_lock=self.have_lock, status=status, time_since_lock=tsl,
                           distance_since_reset=self.dist, omega_b=self.omega_b.copy())
