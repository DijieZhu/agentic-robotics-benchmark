"""On-board autopilot of the reference vehicle (the 'firmware' given to behaviour tasks).

Modelled on the flight modes of common open-source ROV autopilots; uses ONLY a navigation estimate (NavEstimate) and the
measured battery voltage - never simulator truth.  Output: normalised thruster commands.

Modes
  RAW           thruster commands passed through (setpoint.raw)
  MANUAL        normalised body wrench (setpoint.manual, 6 values in [-1, 1]) x rated axis capacity,
                no stabilisation
  STABILIZE     roll / pitch held at setpoint.roll / .pitch; yaw held (yaw_rate input slews the
                heading setpoint); surge / sway / heave from setpoint.manual[0:3]
  DEPTH_HOLD    STABILIZE + depth hold (setpoint.depth, or climb_rate slews it)
  HEADING_HOLD  DEPTH_HOLD + absolute heading setpoint (setpoint.yaw, rate limited)
  VEL           DEPTH_HOLD + heading + body-velocity setpoint (setpoint.vel_b = (u, v)); needs vel_b
  POS_HOLD      DEPTH_HOLD + heading + horizontal station keeping at setpoint.pos_w (x, y); needs
                pos_w and vel_b (station keeping with a positioning system)
  FORCE         POS_HOLD on every axis except those in setpoint.force_mask (Fx Fy Fz Mx My Mz,
                body axes), which take the force / torque setpoints (contact work); the torque
                setpoints are allocated with the attitude group, the forces with the translation group

Structure (per control period, default 100 Hz):
  attitude: quaternion error -> body-rate setpoint (P, rate-limited) -> rate PI -> torque
  depth   : depth error -> depth-rate setpoint (P, limited up / down) -> vertical-velocity PI ->
            world vertical force, rotated into the body frame
  velocity: setpoint slew (acceleration limit) -> body-velocity PI -> body Fx, Fy
  position: world position error -> velocity setpoint (P, speed-limited) -> velocity loop
  allocation: prioritised groups [attitude, depth, translation] (Allocator), anti-windup =
            conditional integration in saturated groups + integrator clamps, output slew limits.
Gains: AutopilotGains.tuned(vehicle) - loop shaping on a nominal plant (rigid inertia x a nominal
added-mass factor, no knowledge of the true hydrodynamics), see UWKIT_DYNAMICS.md section 4.5.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from ..frames import euler_to_quat, quat_conj, quat_mul, quat_to_euler, quat_to_mat, wrap_pi
from ..params import VehicleSpec
from .allocation import Allocator

MODES = ("RAW", "MANUAL", "STABILIZE", "DEPTH_HOLD", "HEADING_HOLD", "VEL", "POS_HOLD", "FORCE")
V_NOMINAL = 14.8


@dataclass
class NavEstimate:
    t: float
    quat_wb: np.ndarray                    # attitude estimate (w, x, y, z), B -> W
    omega_b: np.ndarray                    # body angular rate (gyro) [rad/s]
    depth: float                           # depth of the body origin [m], positive down
    vel_b: Optional[np.ndarray] = None     # velocity over ground of the body origin, body axes
    pos_w: Optional[np.ndarray] = None     # world position of the body origin (x, y[, z])
    depth_rate: Optional[float] = None     # [m/s], positive down


@dataclass
class Setpoint:
    roll: float = 0.0
    pitch: float = 0.0
    yaw: Optional[float] = None            # None: hold the yaw at mode entry
    yaw_rate: float = 0.0                  # pilot yaw-rate input (slews the yaw setpoint) [rad/s]
    depth: Optional[float] = None          # None: hold the depth at mode entry
    climb_rate: float = 0.0                # pilot climb input, positive up [m/s]
    vel_b: tuple = (0.0, 0.0)              # body-velocity setpoint (u, v) [m/s]
    pos_w: Optional[tuple] = None          # None: hold the position at mode entry
    manual: tuple = (0.0,) * 6             # normalised body wrench [-1, 1]
    raw: Optional[tuple] = None            # raw thruster commands [-1, 1]
    force_b: tuple = (0.0, 0.0, 0.0)       # contact force setpoint, body axes [N]
    torque_b: tuple = (0.0, 0.0, 0.0)      # contact torque setpoint [N m]
    force_mask: tuple = (False,) * 6       # which of Fx Fy Fz Mx My Mz are force-controlled


@dataclass
class AutopilotGains:
    att_p: tuple = (3.0, 3.0, 1.5)         # angle error -> rate setpoint [1/s]
    rate_max: tuple = (1.0, 1.0, 0.8)      # [rad/s]
    rate_p: tuple = (3.0, 3.0, 2.8)        # [N m / (rad/s)]
    rate_i: tuple = (4.8, 4.8, 3.4)        # [N m / rad]
    depth_p: float = 1.0                   # [1/s]
    depth_rate_down: float = 0.5           # max descent rate [m/s]
    depth_rate_up: float = 0.5             # max ascent rate [m/s]
    vz_p: float = 112.0                    # [N / (m/s)]
    vz_i: float = 90.0                     # [N / m]
    vel_p: tuple = (57.0, 57.0)            # [N / (m/s)]
    vel_i: tuple = (115.0, 115.0)          # [N / m]
    accel_max: float = 0.3                 # velocity-setpoint slew [m/s^2]
    pos_p: float = 0.6                     # [1/s]
    vel_max: float = 0.5                   # station-keeping speed limit [m/s]
    i_frac: float = 0.9                    # integrator clamp, fraction of the axis capacity
    depth_i_zone: float = 0.25             # depth integrator runs only within this error band [m]
    slew_force: float = 400.0              # output slew limit [N/s]
    slew_torque: float = 40.0              # [N m / s]
    yaw_lead_max: float = 0.5              # max yaw-setpoint lead in STABILIZE [rad]
    depth_ab: tuple = (0.2, 0.022)         # alpha-beta depth-rate filter gains (per 10 ms update)
    notes: dict = field(default_factory=dict)

    @classmethod
    def tuned(cls, v: VehicleSpec, k_rot=1.5, k_surge=1.5, k_heave=2.2, w_rate=(8.0, 8.0, 6.0),
              w_vz=4.0, w_vel=3.0):
        """Loop shaping on a nominal plant (the firmware does not know the true hydrodynamics):
        J_nom = k_rot * I_rigid, m_xy = k_surge * m, m_z = k_heave * m (nominal added-mass factors);
        rate PI: crossover w_rate, integral zero at w/5; angle P = w_rate / (2.7, 2.7, 4);
        heave-velocity PI: crossover w_vz, zero at w/5; depth P = w_vz / 4;
        surge / sway PI: crossover w_vel, integral gain m_xy w^2 / 1.5 (high, so that the integrator
        builds the steady force against a cross current quickly even with heavy damping);
        position P = w_vel / 5."""
        I = np.diag(v.inertia_com_kgm2) * k_rot
        wr = np.asarray(w_rate, float)
        rate_p = I * wr
        rate_i = rate_p * wr / 5.0
        att_p = wr / np.array([2.7, 2.7, 4.0])
        mz, mxy = k_heave * v.mass_kg, k_surge * v.mass_kg
        def tup(x):
            return tuple(float(v) for v in np.round(x, 3))
        g = cls(att_p=tup(att_p), rate_p=tup(rate_p), rate_i=tup(rate_i), depth_p=round(w_vz / 4.0, 3),
                vz_p=round(mz * w_vz, 2), vz_i=round(mz * w_vz * w_vz / 5.0, 2),
                vel_p=(round(mxy * w_vel, 2),) * 2, vel_i=(round(mxy * w_vel * w_vel / 1.5, 2),) * 2,
                pos_p=round(w_vel / 5.0, 3))
        g.notes = dict(k_rot=k_rot, k_surge=k_surge, k_heave=k_heave, w_rate=[float(x) for x in wr], w_vz=w_vz,
                       w_vel=w_vel)
        return g


def _rot_error(q, q_sp):
    """Body-frame rotation vector taking the current attitude to the setpoint."""
    qe = quat_mul(quat_conj(q), q_sp)
    if qe[0] < 0:
        qe = -qe
    v = qe[1:]
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        return 2.0 * v
    return 2.0 * math.atan2(s, qe[0]) * v / s


def _slew(prev, new, max_step):
    return prev + np.minimum(np.maximum(new - prev, -max_step), max_step)


class Autopilot:
    def __init__(self, vehicle: VehicleSpec, gains: Optional[AutopilotGains] = None,
                 control_dt: float = 0.01, allocator: Optional[Allocator] = None):
        self.v = vehicle
        self.g = gains or AutopilotGains.tuned(vehicle)
        self.dt = float(control_dt)
        self.alloc = allocator or Allocator.from_vehicle(vehicle)
        self.n = self.alloc.n
        cap = self.alloc.axis_capacity(V_NOMINAL)
        self.capacity = np.minimum(cap[:, 0], cap[:, 1])        # symmetric rated capacity per axis
        self.mode = "DEPTH_HOLD"
        self.sp = Setpoint()
        self._entry_pending = True
        self._reset_state()
        self.status = {}
        self.set_gains(self.g)

    def set_gains(self, gains: AutopilotGains):
        """Install gains (vectors are cached as arrays for the 100 Hz loop)."""
        self.g = gains
        g = gains
        self._att_p = np.asarray(g.att_p, float)
        self._rate_max = np.asarray(g.rate_max, float)
        self._rate_p = np.asarray(g.rate_p, float)
        self._rate_i = np.asarray(g.rate_i, float)
        self._vel_p = np.asarray(g.vel_p, float)
        self._vel_i = np.asarray(g.vel_i, float)
        self._slew_lim = np.concatenate([np.full(3, g.slew_force), np.full(3, g.slew_torque)]) * self.dt
        cap = self.capacity
        self._i_lim_rot = g.i_frac * cap[3:] / np.maximum(self._rate_i, 1e-9)
        self._i_lim_z = g.i_frac * cap[2] / max(g.vz_i, 1e-9)
        self._i_lim_v = g.i_frac * cap[:2] / np.maximum(self._vel_i, 1e-9)

    # ------------------------------------------------------------------ mode handling
    def _reset_state(self):
        self.i_rate = np.zeros(3)
        self.i_vz = 0.0
        self.i_vel = np.zeros(2)
        self.yaw_sp = None
        self.depth_sp = None
        self.pos_sp = None
        self.vel_sp = np.zeros(2)
        self.tau_prev = [np.zeros(6), np.zeros(6), np.zeros(6)]
        self._ab = None                       # alpha-beta depth filter state (depth, rate)
        self.u = np.zeros(self.n)

    def set_mode(self, mode, **setpoints):
        mode = mode.upper()
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}; one of {MODES}")
        if mode != self.mode:
            used_old, used_new = self._uses(self.mode), self._uses(mode)
            if "vel" not in used_new or "vel" not in used_old:
                self.i_vel[:] = 0.0
                self.vel_sp[:] = 0.0
            if "depth" not in used_new or "depth" not in used_old:
                self.i_vz = 0.0
            if "pos" not in used_new:
                self.pos_sp = None
            self._entry_pending = True
        self.mode = mode
        self.sp = replace(Setpoint(), **setpoints) if setpoints else Setpoint()
        if self.sp.depth is not None:
            self.depth_sp = float(self.sp.depth)
        if self.sp.yaw is not None:
            self.yaw_sp = float(self.sp.yaw)
        if self.sp.pos_w is not None:
            self.pos_sp = np.asarray(self.sp.pos_w, float)[:2].copy()

    def set_setpoint(self, **kw):
        """Update setpoints without a mode change (bumpless)."""
        self.sp = replace(self.sp, **kw)
        if "depth" in kw and kw["depth"] is not None:
            self.depth_sp = float(kw["depth"])
        if "yaw" in kw and kw["yaw"] is not None:
            self.yaw_sp = float(kw["yaw"])
        if "pos_w" in kw and kw["pos_w"] is not None:
            self.pos_sp = np.asarray(kw["pos_w"], float)[:2].copy()

    @staticmethod
    def _uses(mode):
        return {"RAW": set(), "MANUAL": set(), "STABILIZE": {"att"}, "DEPTH_HOLD": {"att", "depth"},
                "HEADING_HOLD": {"att", "depth"}, "VEL": {"att", "depth", "vel"},
                "POS_HOLD": {"att", "depth", "vel", "pos"}, "FORCE": {"att", "depth", "vel", "pos"}}[mode]

    # ------------------------------------------------------------------ estimators
    def _depth_rate(self, nav: NavEstimate, R):
        if nav.depth_rate is not None:
            return float(nav.depth_rate)
        if nav.vel_b is not None:
            return -float((R @ np.asarray(nav.vel_b, float))[2])
        a, b = self.g.depth_ab
        if self._ab is None:
            self._ab = [float(nav.depth), 0.0]
        x, r = self._ab
        x += r * self.dt
        res = float(nav.depth) - x
        x += a * res
        r += b * res / self.dt
        self._ab = [x, r]
        return r

    # ------------------------------------------------------------------ main update
    def update(self, nav: NavEstimate, v_batt: Optional[float] = None):
        """One control period: returns normalised thruster commands (n,)."""
        g, sp, dt = self.g, self.sp, self.dt
        V = round(float(v_batt), 2) if v_batt is not None else V_NOMINAL     # 10 mV ADC resolution
        mode = self.mode
        if mode == "RAW":
            self.u = np.clip(np.asarray(sp.raw if sp.raw is not None else np.zeros(self.n), float), -1, 1)
            self.status = dict(mode=mode)
            return self.u
        q = np.asarray(nav.quat_wb, float)
        R = quat_to_mat(q)
        rpy = quat_to_euler(q)
        w = np.asarray(nav.omega_b, float)
        uses = self._uses(mode)
        if self._entry_pending:
            if self.yaw_sp is None or (sp.yaw is None and mode in ("STABILIZE", "DEPTH_HOLD")):
                self.yaw_sp = float(rpy[2]) if sp.yaw is None else float(sp.yaw)
            if self.depth_sp is None:
                self.depth_sp = float(nav.depth)
            if "pos" in uses and self.pos_sp is None:
                if nav.pos_w is None:
                    raise ValueError(f"{mode} needs a position estimate (nav.pos_w)")
                self.pos_sp = np.asarray(nav.pos_w, float)[:2].copy()
            self._entry_pending = False
        fm = np.asarray(sp.force_mask, bool) if mode == "FORCE" else np.zeros(6, bool)
        man = np.clip(np.asarray(sp.manual, float), -1, 1) * self.capacity
        tau_att, tau_dep, tau_tr = np.zeros(6), np.zeros(6), np.zeros(6)
        info = dict(mode=mode)

        if mode == "MANUAL":
            tau_att[3:] = man[3:]
            tau_tr[:3] = man[:3]
        else:
            # ---------------- attitude
            if mode in ("STABILIZE", "DEPTH_HOLD"):
                if sp.yaw is not None:
                    self.yaw_sp = float(sp.yaw)
                elif sp.yaw_rate:
                    lead = wrap_pi(self.yaw_sp + sp.yaw_rate * dt - rpy[2])
                    self.yaw_sp = float(wrap_pi(rpy[2] + np.clip(lead, -g.yaw_lead_max, g.yaw_lead_max)))
            elif sp.yaw is not None:
                self.yaw_sp = float(sp.yaw)
            q_sp = euler_to_quat(sp.roll, sp.pitch, self.yaw_sp)
            e_att = _rot_error(q, q_sp)
            w_sp = np.minimum(np.maximum(self._att_p * e_att, -self._rate_max), self._rate_max)
            e_w = w_sp - w
            tau_rot = self._rate_p * e_w + self._rate_i * self.i_rate
            tau_att[3:] = np.where(fm[3:], 0.0, tau_rot)
            info.update(att_err=e_att, rate_sp=w_sp)
            # ---------------- depth
            if "depth" in uses and not fm[2]:
                if sp.climb_rate:
                    self.depth_sp = max(0.0, self.depth_sp - sp.climb_rate * dt)
                d_rate = self._depth_rate(nav, R)
                e_d = self.depth_sp - float(nav.depth)
                wz_sp = min(max(g.depth_p * e_d, -g.depth_rate_up), g.depth_rate_down)
                e_vz = wz_sp - d_rate
                F_down = g.vz_p * e_vz + g.vz_i * self.i_vz
                tau_dep[:3] = R.T @ np.array([0.0, 0.0, -F_down])
                info.update(depth_err=e_d, depth_rate=d_rate, depth_rate_sp=wz_sp)
            else:
                e_vz = 0.0
                if mode == "STABILIZE":
                    tau_tr[2] = man[2]
            # ---------------- horizontal
            e_v = np.zeros(2)
            if "vel" in uses:
                if nav.vel_b is None:
                    raise ValueError(f"{mode} needs a velocity estimate (nav.vel_b)")
                vb = np.asarray(nav.vel_b, float)
                if "pos" in uses:
                    if nav.pos_w is None:
                        raise ValueError(f"{mode} needs a position estimate (nav.pos_w)")
                    p = np.asarray(nav.pos_w, float)
                    if np.any(fm[:2]):                          # free axes follow the vehicle
                        for ax in (0, 1):
                            if fm[ax]:
                                e_ax = R[:2, ax] / max(np.linalg.norm(R[:2, ax]), 1e-9)
                                self.pos_sp = self.pos_sp + e_ax * float(e_ax @ (p[:2] - self.pos_sp))
                    e_p = self.pos_sp - p[:2]
                    v_w = g.pos_p * e_p
                    s = float(np.linalg.norm(v_w))
                    if s > g.vel_max:
                        v_w *= g.vel_max / s
                    target = (R.T @ np.array([v_w[0], v_w[1], 0.0]))[:2]
                    info.update(pos_err=e_p)
                else:
                    target = np.asarray(sp.vel_b, float)
                self.vel_sp = _slew(self.vel_sp, target, g.accel_max * dt)
                e_v = self.vel_sp - vb[:2]
                F_xy = self._vel_p * e_v + self._vel_i * self.i_vel
                tau_tr[:2] = np.where(fm[:2], 0.0, F_xy)
                info.update(vel_sp=self.vel_sp.copy(), vel_err=e_v)
            elif mode != "MANUAL":
                tau_tr[:2] = man[:2]
            if mode == "FORCE":
                # 0.3.0: a released rotation axis's torque setpoint takes the attitude group's priority (moments are
                # allocated first, as the attitude controller's torque it replaces); forces stay in the translation
                # group.  0.2.0 put both into the translation group (torque scaled with the forces on saturation).
                f_b, t_b = np.asarray(sp.force_b, float), np.asarray(sp.torque_b, float)
                tau_tr[:3] = tau_tr[:3] + np.where(fm[:3], f_b, 0.0)
                tau_att[3:] = tau_att[3:] + np.where(fm[3:], t_b, 0.0)
        # ---------------- output slew + allocation
        lim = self._slew_lim
        groups = []
        for k, tau in enumerate((tau_att, tau_dep, tau_tr)):
            t = _slew(self.tau_prev[k], tau, lim) if mode != "MANUAL" else tau
            self.tau_prev[k] = t
            groups.append(t)
        u, ainfo = self.alloc.allocate(groups, V)
        a_att, a_dep, a_tr = ainfo["alpha"]
        # ---------------- integrators with anti-windup (conditional integration + clamps)
        if mode != "MANUAL":
            lim_r = self._i_lim_rot
            if a_att > 0.999 and not fm[3:].any():
                ir = self.i_rate + e_w * dt
            else:
                grow = np.sign(e_w) == np.sign(self.i_rate)
                ok = (a_att > 0.999) | ~grow
                ir = np.where(ok & ~fm[3:], self.i_rate + e_w * dt, self.i_rate)
            self.i_rate = np.minimum(np.maximum(ir, -lim_r), lim_r)
            if "depth" in uses and not fm[2]:
                in_zone = abs(self.depth_sp - float(nav.depth)) < g.depth_i_zone
                if in_zone and (a_dep > 0.999 or (e_vz > 0) != (self.i_vz > 0)):
                    self.i_vz += e_vz * dt
                self.i_vz = min(max(self.i_vz, -self._i_lim_z), self._i_lim_z)
            if "vel" in uses:
                if a_tr > 0.999 and not fm[:2].any():
                    iv = self.i_vel + e_v * dt
                else:
                    grow = np.sign(e_v) == np.sign(self.i_vel)
                    ok = (a_tr > 0.999) | ~grow
                    iv = np.where(ok & ~fm[:2], self.i_vel + e_v * dt, self.i_vel)
                self.i_vel = np.minimum(np.maximum(iv, -self._i_lim_v), self._i_lim_v)
        info.update(alpha=ainfo["alpha"], thrust=ainfo["f"], wrench=ainfo["wrench"],
                    saturated=bool(min(ainfo["alpha"]) < 0.999), yaw_sp=self.yaw_sp, depth_sp=self.depth_sp,
                    pos_sp=None if self.pos_sp is None else self.pos_sp.copy())
        self.status = info
        self.u = u
        return u


def run_closed_loop(sim, ap: Autopilot, duration, nav_fn=None, log_every=None, on_control=None):
    """Drive `sim` with `ap` for `duration` seconds.  nav_fn(sim) -> NavEstimate (default: truth,
    noise-free - tests only).  The autopilot runs every round(ap.dt / sim.dt) physics steps.
    Returns a list of (t, VehicleState, autopilot status) samples every `log_every` seconds."""
    nav_fn = nav_fn or (lambda s: s.truth_nav())
    k_ctrl = max(1, int(round(ap.dt / sim.dt)))
    n = int(round(duration / sim.dt))
    k_log = None if log_every is None else max(1, int(round(log_every / sim.dt)))
    log = []
    for k in range(n):
        if k % k_ctrl == 0:
            if on_control is not None:
                on_control(sim, ap)
            sim.set_thrusters(ap.update(nav_fn(sim), sim.battery.voltage))
        sim.step()
        if k_log and (k + 1) % k_log == 0:
            log.append((sim.time, sim.state(), dict(ap.status)))
    return log
