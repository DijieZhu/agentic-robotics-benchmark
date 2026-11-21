"""Flight firmware of the reference free-flyer: 6-DOF controller, flight modes, trapezoidal planner and a
small executive (move / hold / stop / idle, tolerance monitoring, speed-level preparation).

``FlightController`` is a step-for-step port of the reference firmware's controller (62.5 Hz):
  command shaper (forward integration of the current setpoint by time_delta, quaternion propagated with
  the 4x4 rate matrix incl. its third-order commutator term), mode logic (idle / stopping / nominal and
  the latched 'stopped' mode entered after 4 stopping ticks with Butterworth-filtered |v|, |w| below
  threshold), PID position loop through a velocity target
      F_body = sat( m * [ R^T( vel_kd (vel_gain (v_t - v) + kp/kd e + int) ) + R^T(accel_gain a_t) ], 100 N),
  PID attitude loop with gyroscopic compensation
      tau = J (alpha_gain alpha_t) + kd diag(J) (w_t + int + kp/kd q_err,vec - w) - (J w) x w.
Quaternions at the API are (w, x, y, z) with q rotating body vectors into the world frame; positions and
linear velocities in world axes, angular velocities and the produced wrench in body axes.

``plan_trapezoid`` ports the firmware's trapezoidal planner (greedy then fair ramps, rotation and
translation stretched to the same duration, optional face-forward look-translate-rotate split, samples
every half control period of the planner rate).  ``Firmware`` glues controller, planner and allocator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ._fast import cross
from .propulsion import FlightAllocator, NNLSAllocator, PropulsionParams, load_firmware_reference
from .rotations import quat_conj, quat_from_axis_angle, quat_mul, quat_to_mat, mat_to_quat, rotvec_from_quat

__all__ = [
    "MODE_IDLE", "MODE_STOP", "MODE_NOMINAL", "MODE_STOPPED", "FlightMode", "load_flight_modes",
    "ControlState", "Setpoint", "FlightController", "plan_trapezoid", "segment_setpoint", "Firmware",
]

MODE_IDLE, MODE_STOP, MODE_NOMINAL, MODE_STOPPED = 0, 1, 2, 3


@dataclass
class FlightMode:
    """A firmware flight mode: gains, impeller speed level (0 off, 1-3), tolerances and planner limits."""

    name: str
    control_enabled: bool
    att_kp: np.ndarray
    att_ki: np.ndarray
    omega_kd: np.ndarray
    pos_kp: np.ndarray
    pos_ki: np.ndarray
    vel_kd: np.ndarray
    speed: int
    tolerance_pos_endpoint: float
    tolerance_pos: float
    tolerance_vel: float
    tolerance_att: float
    tolerance_omega: float
    tolerance_time: float
    hard_limit_vel: float
    hard_limit_accel: float
    hard_limit_omega: float
    hard_limit_alpha: float

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "FlightMode":
        """Build from one entry of the firmware reference 'flight_modes' block."""
        v = {k: np.asarray(d[k], float) for k in ("att_kp", "att_ki", "omega_kd", "pos_kp", "pos_ki", "vel_kd")}
        f = {k: float(d[k]) for k in ("tolerance_pos_endpoint", "tolerance_pos", "tolerance_vel", "tolerance_att",
                                      "tolerance_omega", "tolerance_time", "hard_limit_vel", "hard_limit_accel",
                                      "hard_limit_omega", "hard_limit_alpha")}
        return cls(name=name, control_enabled=bool(d["control_enabled"]), speed=int(d["speed"]), **v, **f)


def load_flight_modes(ref: dict | None = None) -> dict:
    """{name: FlightMode} from the firmware reference (off, standard, agile, hush, fine)."""
    ref = load_firmware_reference() if ref is None else ref
    return {k: FlightMode.from_dict(k, v) for k, v in ref["flight_modes"].items()}


@dataclass
class ControlState:
    """Estimated state handed to the controller (what the localiser reports)."""

    pos: np.ndarray            # world position of the body origin [m]
    quat: np.ndarray           # (w, x, y, z), body -> world
    vel: np.ndarray            # world-axes linear velocity [m/s]
    omega: np.ndarray          # body-axes angular velocity [rad/s]
    confidence: int = 0        # 0 good, 1 poor, 2 lost (controller idles unless 0)


@dataclass
class Setpoint:
    """Trajectory sample: time [s], position, velocity, acceleration (world), attitude (w,x,y,z),
    angular velocity and acceleration (body axes, as the planner produces them)."""

    t: float
    pos: np.ndarray
    vel: np.ndarray
    acc: np.ndarray
    quat: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray


def _omega_matrix(v):
    """The firmware's 4x4 rate matrix acting on (x, y, z, w) quaternion coefficients."""
    o = np.zeros((4, 4))
    o[1, 0], o[2, 0], o[3, 0] = -v[2], v[1], -v[0]
    o[0, 1], o[2, 1], o[3, 1] = v[2], -v[0], -v[1]
    o[0, 2], o[1, 2], o[3, 2] = -v[1], v[0], -v[2]
    o[0, 3], o[1, 3], o[2, 3] = v[0], v[1], v[2]
    return o


def _xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], float)


def _wxyz(c):
    return np.array([c[3], c[0], c[1], c[2]], float)


def _normalize_fw(q):
    """Firmware NormalizeQuaternion: enforce w >= 0, normalise if |q| > 1e-7 ((w, x, y, z) in/out)."""
    q = np.asarray(q, float).copy()
    if q[0] < 0:
        q = -q
    m = math.sqrt(float(q @ q))
    if m > 1e-7:
        q /= m
    return q


def _quat_error(cmd, actual) -> float:
    """Firmware QuatError: 2 |acos(w)| of actual^-1 * cmd (normalised, w >= 0)."""
    out = _normalize_fw(quat_mul(quat_conj(actual), cmd))
    return abs(math.acos(max(-1.0, min(1.0, float(out[0]))))) * 2.0


def _safe_divide(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    out = np.zeros(3)
    nz = b != 0
    out[nz] = a[nz] / b[nz]
    return out


class FlightController:
    """Port of the firmware controller (see module docstring).  One instance per robot; call ``step`` at
    the control rate (62.5 Hz).  ``ctl`` = the 'control' block of the firmware reference."""

    def __init__(self, ctl: dict | None = None, mass: float | None = None, inertia=None):
        ref = load_firmware_reference()
        self.c = ref["control"] if ctl is None else ctl
        mp = ref["mass_properties_configured"]
        self.mass = float(mp["mass_kg"] if mass is None else mass)
        self.inertia = np.asarray(mp["inertia_kgm2"] if inertia is None else inertia, float)
        self.rate = float(self.c["rate_hz"])
        self.reset()

    def reset(self) -> None:
        """Firmware Initialize(): clear modes, filters, latched pose and integrators."""
        self.mode_cmd = 0
        self.stopped_mode = False
        self.prev_filter_vel = np.zeros(3)
        self.prev_filter_omega = np.zeros(3)
        self.prev_mode_cmd = [0, 0, 0, 0, 0]
        self.prev_position = np.zeros(3)
        self.prev_att = np.zeros(4)        # the firmware starts with an all-zero quaternion
        self.linear_integrator = np.zeros(3)
        self.rotational_integrator = np.zeros(3)
        self.out = {}
        self._ratio_cache = {}
        self._Jinv = np.linalg.inv(self.inertia)

    # -- helpers ported from the firmware -----------------------------------------------------------
    def _butter(self, x, delay, acc):
        g1, g2 = self.c["butterworth_gain_1"], self.c["butterworth_gain_2"]
        tmp = x * g1 - delay * g2
        out = tmp + delay
        acc[0] += out * out
        return tmp

    def _filter_threshold(self, vec, thresh, prev) -> bool:
        acc = [0.0]
        for i in range(3):
            prev[i] = self._butter(float(vec[i]), float(prev[i]), acc)
        return acc[0] < thresh

    def _dti(self, inp, accum, status, upper, lower):
        if status <= 1:
            accum[:] = 0.0
            return np.zeros(3)
        accum += inp / self.rate
        return np.clip(accum, lower, upper)

    def _ratios(self, g: FlightMode):
        key = id(g)
        r = self._ratio_cache.get(key)
        if r is None or r[0] is not g:
            r = (g, _safe_divide(g.pos_ki, g.vel_kd), _safe_divide(g.pos_kp, g.vel_kd), _safe_divide(g.att_ki, g.omega_kd),
                 _safe_divide(g.att_kp, g.omega_kd), g.omega_kd * np.diag(self.inertia))
            self._ratio_cache[key] = r
        return r[1:]

    def step(self, time_delta: float, est: ControlState, cmd: Setpoint | None, mode: int, gains: FlightMode) -> dict:
        """One controller tick.  cmd: the active setpoint (may be None for idle / stop with no history);
        time_delta: now - cmd.t; mode: MODE_IDLE / MODE_STOP / MODE_NOMINAL.  Returns a dict with
        force / torque (body), accel / alpha cmds, errors, the forward-integrated trajectory, ctl_status."""
        c = self.c
        if cmd is None:
            z = np.zeros(3)
            cmd = Setpoint(0.0, est.pos.copy(), z, z, est.quat.copy(), z, z)
        o = {}
        dt = float(time_delta)
        # ForwardTrajectory
        o["traj_pos"] = cmd.pos + cmd.vel * dt + 0.5 * cmd.acc * dt * dt
        o["traj_vel"] = cmd.vel + cmd.acc * dt
        o["traj_accel"] = np.asarray(cmd.acc, float).copy()
        o["traj_alpha"] = np.asarray(cmd.alpha, float).copy()
        o["traj_omega"] = cmd.omega + cmd.alpha * dt
        if not (np.any(cmd.omega) or np.any(cmd.alpha)):
            o["traj_quat"] = _normalize_fw(cmd.quat)
        else:
            Ow, Oa = _omega_matrix(cmd.omega), _omega_matrix(cmd.alpha)
            v = 0.5 * dt * (0.5 * dt * np.asarray(cmd.alpha, float) + np.asarray(cmd.omega, float))
            nv = float(np.linalg.norm(v))
            E = math.cos(nv) * np.eye(4) + (math.sin(nv) / nv if nv > 1e-12 else 1.0) * _omega_matrix(v)
            E += (1.0 / 48.0) * dt ** 3 * (Oa @ Ow - Ow @ Oa)
            o["traj_quat"] = _normalize_fw(_wxyz(E @ _xyzw(cmd.quat)))
        o["traj_error_pos"] = float(np.linalg.norm(o["traj_pos"] - est.pos))
        o["traj_error_vel"] = float(np.linalg.norm(o["traj_vel"] - est.vel))
        o["traj_error_omega"] = float(np.linalg.norm(o["traj_omega"] - est.omega))
        o["traj_error_att"] = _quat_error(o["traj_quat"], est.quat)
        # UpdateMode
        self.mode_cmd = MODE_IDLE if est.confidence != 0 else int(mode)
        for i in range(3, -1, -1):
            self.prev_mode_cmd[i + 1] = self.prev_mode_cmd[i]
        self.prev_mode_cmd[0] = self.mode_cmd
        vel_below = self._filter_threshold(est.vel, c["stopping_vel_thresh_sq"], self.prev_filter_vel)
        om_below = self._filter_threshold(est.omega, c["stopping_omega_thresh_sq"], self.prev_filter_omega)
        self.stopped_mode = False
        pm = self.prev_mode_cmd
        if vel_below and om_below and pm[4] == MODE_STOP and pm[4] == pm[3] == pm[2] == pm[1]:
            self.stopped_mode = True
        # UpdateCtlStatus
        pos_err2 = float(np.sum((self.prev_position - est.pos) ** 2))
        q_err = _quat_error(est.quat, self.prev_att) if np.any(self.prev_att) else math.pi
        if (pos_err2 > c["stopped_pos_thresh"] or abs(q_err) > c["stopped_quat_thresh_rad"]) and self.mode_cmd == MODE_STOPPED:
            status = MODE_STOP
        else:
            status = MODE_STOPPED if self.stopped_mode else self.mode_cmd
        o["ctl_status"] = status
        R = quat_to_mat(est.quat)
        # FindPosErr
        target = self.prev_position if self.stopped_mode else o["traj_pos"]
        pos_err = target - est.pos
        ki_lin, kp_lin, ki_rot, kp_rot, kd_rot = self._ratios(gains)
        lo, hi = c["pos_int_limits"]
        pos_int = self._dti(ki_lin * pos_err, self.linear_integrator, status, hi, lo)
        o["pos_err"], o["pos_err_int"] = pos_err, pos_int
        # FindBodyForceCmd
        if status == MODE_IDLE:
            o["force"] = np.zeros(3)
            o["accel_cmd"] = np.zeros(3)
        else:
            tv = np.zeros(3) if self.stopped_mode else o["traj_vel"]
            ta = np.zeros(3) if self.stopped_mode else o["traj_accel"]
            vv = -est.vel
            if status > 1:
                vv = c["vel_gain"] * (tv - est.vel) + kp_lin * pos_err + pos_int
            a = (R.T @ vv) * gains.vel_kd
            b = R.T @ (ta * np.asarray(c["accel_gain"], float))
            f = self.mass * (a + b)
            mag = float(np.linalg.norm(f))
            lim = c["linear_force_limit_n"]
            o["force"] = f if mag < lim else lim / mag * f
            o["accel_cmd"] = o["force"] / self.mass
        # FindAttErr
        q_cmd = self.prev_att if self.stopped_mode else o["traj_quat"]
        q_out = _normalize_fw(quat_mul(quat_conj(est.quat), q_cmd))
        o["att_err_mag"] = 2.0 * math.acos(max(-1.0, min(1.0, float(q_out[0]))))
        att_err = q_out[1:].copy()
        lo, hi = c["att_int_limits"]
        att_int = self._dti(att_err * ki_rot, self.rotational_integrator, status, hi, lo)
        o["att_err"], o["att_err_int"] = att_err, att_int
        # FindBodyAlphaTorqueCmd
        cmd_om = np.zeros(3) if self.stopped_mode else o["traj_omega"]
        cmd_al = np.zeros(3) if self.stopped_mode else o["traj_alpha"]
        rate_err = -est.omega
        if status > 1:
            rate_err = cmd_om + att_int + kp_rot * att_err - est.omega
        rate_err = rate_err * kd_rot
        o["alpha_cmd"] = self._Jinv @ rate_err
        if status == MODE_IDLE:
            o["torque"] = np.zeros(3)
        else:
            al = np.asarray(c["alpha_gain"], float) * cmd_al
            Jw = self.inertia @ est.omega
            o["torque"] = self.inertia @ al + rate_err - cross(Jw, est.omega)
        # UpdatePrevious
        if not self.stopped_mode:
            self.prev_position = np.asarray(est.pos, float).copy()
            self.prev_att = np.asarray(est.quat, float).copy()
        self.out = o
        return o


# ------------------------------------------------------------------------------------------------ planner
def _greedy_ramp(d, v, a, eps):
    """Firmware GreedyRamp: returns (t_total, r, c, h) for distance d, max speed v, max accel a."""
    if d < eps:
        return 0.0, 0.0, 0.0, 0.0
    h = math.sqrt(a * d)
    if h > v:
        h = v
        r = h / a
        c = (d - h * h / a) / h
    else:
        r = d / h
        c = 0.0
    return r * 2.0 + c, r, c, h


def _fair_ramp(t, d, v, a, eps, r, c, h):
    """Firmware FairRamp: re-time a ramp over the dominating duration t (returns r, c, h)."""
    if t < eps or d < eps:
        return r, c, h
    if abs(t * t * a / 4.0 - d) < eps:
        return t / 2.0, 0.0, math.sqrt(a * d)
    h = (a * t - math.sqrt(a * a * t * t - 4.0 * d * a)) / 2.0
    r = h / a
    return r, (d - r * h) / h, h


def _rot_angle_axis(q0, q1):
    """Angle in [0, pi] and unit axis (in the frame of q0) of q0^-1 q1, as the firmware normalises it."""
    rv = rotvec_from_quat(quat_mul(quat_conj(q0), q1))
    ang = float(np.linalg.norm(rv))
    if ang < 1e-15:
        return 0.0, np.array([1.0, 0.0, 0.0])
    return ang, rv / ang


def _insert_trapezoid(seg, t0, dt_req, p0, q0, p1, q1, lin_v, rot_v, lin_a, rot_a, min_period, eps):
    """Append the samples of one trapezoidal move to ``seg``; returns the new time offset."""
    rot_m, rot_d = _rot_angle_axis(q0, q1)
    err = np.asarray(p1, float) - np.asarray(p0, float)
    lin_m = float(np.linalg.norm(err))
    lin_d = err / lin_m if lin_m > 0 else np.zeros(3)
    if lin_m < eps and rot_m < eps:
        return t0
    lin_t, lin_r, lin_c, lin_h = _greedy_ramp(lin_m, lin_v, lin_a, eps)
    rot_t, rot_r, rot_c, rot_h = _greedy_ramp(rot_m, rot_v, rot_a, eps)
    tmin = max(lin_t, rot_t)
    if dt_req > 0 and tmin < dt_req:
        tmin = dt_req
    lin_r, lin_c, lin_h = _fair_ramp(tmin, lin_m, lin_v, lin_a, eps, lin_r, lin_c, lin_h)
    rot_r, rot_c, rot_h = _fair_ramp(tmin, rot_m, rot_v, rot_a, eps, rot_r, rot_c, rot_h)
    ts = []
    if lin_m > eps:
        ts.append(lin_r)
        if lin_c > eps:
            ts.append(lin_r + lin_c)
    if rot_m > eps:
        ts.append(rot_r)
        if rot_c > eps:
            ts.append(rot_r + rot_c)
    t = 0.0
    while t < tmin:
        ts.append(t)
        t += min_period / 2.0
    ts.append(tmin)
    ts.sort()
    z = np.zeros(3)
    for t in ts:
        p, v, a, q, w, b = np.asarray(p0, float).copy(), z.copy(), z.copy(), np.asarray(q0, float).copy(), z.copy(), z.copy()
        if lin_m > eps:
            if t >= tmin:
                p = p0 + lin_d * (lin_a * lin_r * lin_r + lin_h * lin_c)
            elif t >= lin_r + lin_c:
                tt = t - lin_r - lin_c
                a = -lin_a * lin_d
                v = lin_h * lin_d - lin_a * lin_d * tt
                p = p0 + lin_d * (0.5 * lin_a * lin_r * lin_r + lin_h * lin_c + lin_h * tt - 0.5 * lin_a * tt * tt)
            elif t >= lin_r and lin_c > eps:
                tt = t - lin_r
                v = lin_h * lin_d
                p = p0 + lin_d * (0.5 * lin_a * lin_r * lin_r + lin_h * tt)
            elif t >= 0.0:
                a = lin_a * lin_d
                v = lin_a * lin_d * t
                p = p0 + lin_d * (0.5 * lin_a * t * t)
        if rot_m > eps:
            if t >= tmin:
                ang = rot_a * rot_r * rot_r + rot_h * rot_c
            elif t >= rot_r + rot_c:
                tt = t - rot_r - rot_c
                b = -rot_a * rot_d
                w = rot_h * rot_d - rot_a * rot_d * tt
                ang = 0.5 * rot_a * rot_r * rot_r + rot_h * rot_c + rot_h * tt - 0.5 * rot_a * tt * tt
            elif t >= rot_r and rot_c > eps:
                tt = t - rot_r
                w = rot_h * rot_d
                ang = 0.5 * rot_a * rot_r * rot_r + rot_h * tt
            else:
                b = rot_a * rot_d
                w = rot_a * rot_d * t
                ang = 0.5 * rot_a * t * t
            q = quat_mul(q0, quat_from_axis_angle(rot_d, ang))
        seg.append(Setpoint(t0 + t, p, v, a, q, w, b))
    return t0 + tmin


def _faceforward_quat(delta, eps):
    """Firmware face-forward attitude: body x along the motion, body y in the world x-y plane."""
    vfwd = delta / np.linalg.norm(delta)
    vdown = np.array([0.0, 0.0, 1.0])
    vright = np.array([0.0, 1.0, 0.0])
    if abs(float(vdown @ vfwd)) < 1.0 - eps:
        vright = np.cross(vdown, vfwd)
        vdown = np.cross(vfwd, vright)
        if vdown[2] < 0:
            vright = -vright
            vdown = np.cross(vfwd, vright)
    else:
        vdown = np.cross(vfwd, vright)
        vright = np.cross(vdown, vfwd)
        if vright[1] < 0:
            vdown = -vdown
            vright = np.cross(vdown, vfwd)
    dcm = np.stack([vfwd / np.linalg.norm(vfwd), vright / np.linalg.norm(vright), vdown / np.linalg.norm(vdown)], 1)
    return mat_to_quat(dcm)


def plan_trapezoid(poses, mode: FlightMode, faceforward: bool = True, t0: float = 0.0, times=None,
                   planner: dict | None = None, desired: dict | None = None):
    """Trapezoidal segment through poses [(pos, quat), ...] (the first is the start pose).

    Limits: the mode's hard limits (accelerations divided by 2 in holonomic mode, as the firmware's
    choreographer does), optionally lowered by ``desired`` = dict(vel, accel, omega, alpha).  times:
    optional requested times per pose (the move is stretched to them if slower).  Returns a list of
    Setpoint samples (times start at t0).  Raises ValueError when a desired value exceeds a hard limit.
    """
    pl = load_firmware_reference()["planner"] if planner is None else planner
    eps, shim, rate = pl["epsilon"], pl["faceforward_shim_m"], pl["desired_rate_hz"]
    div = 1.0 if faceforward else pl["holonomic_divider"]
    lim = dict(vel=mode.hard_limit_vel, accel=mode.hard_limit_accel / div, omega=mode.hard_limit_omega,
               alpha=mode.hard_limit_alpha / div)
    if desired:
        for k, v in desired.items():
            if v is None or v < 0:
                continue
            if v > lim[k] + 1e-12:
                raise ValueError(f"desired {k} {v} exceeds the flight-mode limit {lim[k]}")
            lim[k] = v
    seg = []
    t = t0
    for i in range(1, len(poses)):
        p0, q0 = np.asarray(poses[i - 1][0], float), np.asarray(poses[i - 1][1], float)
        p1, q1 = np.asarray(poses[i][0], float), np.asarray(poses[i][1], float)
        dt_req = 0.0 if times is None else float(times[i] - times[i - 1])
        delta = p1 - p0
        if faceforward and np.linalg.norm(delta) > shim:
            qf = _faceforward_quat(delta, eps)
            t = _insert_trapezoid(seg, t, dt_req, p0, q0, p0, qf, lim["vel"], lim["omega"], lim["accel"], lim["alpha"], 1.0 / rate, eps)
            t = _insert_trapezoid(seg, t, dt_req, p0, qf, p1, qf, lim["vel"], lim["omega"], lim["accel"], lim["alpha"], 1.0 / rate, eps)
            t = _insert_trapezoid(seg, t, dt_req, p1, qf, p1, q1, lim["vel"], lim["omega"], lim["accel"], lim["alpha"], 1.0 / rate, eps)
        else:
            t = _insert_trapezoid(seg, t, dt_req, p0, q0, p1, q1, lim["vel"], lim["omega"], lim["accel"], lim["alpha"], 1.0 / rate, eps)
    return seg


def segment_setpoint(seg, t):
    """(setpoint, index) active at time t: the latest sample with sample.t <= t (None before the start)."""
    if not seg or t < seg[0].t:
        return None, -1
    lo, hi = 0, len(seg) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if seg[mid].t <= t + 1e-12:
            lo = mid
        else:
            hi = mid - 1
    return seg[lo], lo


class Firmware:
    """Executive around the ported controller and an allocator.

    Commands: set_mode(name), move(poses, faceforward), hold(), stop(), idle().  Call tick(t, est,
    impeller_speed) at the control rate; it returns dict(impeller_cmd, servo_cmd, force, torque, ...).
    Status: ``state`` in {'idle', 'preparing', 'moving', 'stopping', 'stopped'}, ``result`` of the last
    move ('success', 'tolerance_pos', 'tolerance_att', 'tolerance_pos_endpoint', 'cancelled',
    'localization_lost', 'overspeed_stop', 'overspeed_cutoff', 'speed_timeout' or None while running).
    allocator: 'flight' (port) or 'nnls'.  Over-speed response (two tiers, design review package):
    |v| > overspeed_stop [m/s] commands a stop, |v| > overspeed_cutoff idles the propulsion (None = off).
    The controller idles by itself whenever est.confidence != 0 (as the firmware does).
    stop_feedforward (v0.4): what the stop mode keeps of the setpoint that was active when a move was stopped early
    (``stop()`` / ``hold()`` while moving, a tolerance failure, an over-speed stop, a localisation loss): 'zero'
    (default, a defect fix) keeps its pose and zeroes its velocity, acceleration, rate and angular acceleration;
    'hold' (v0.3, the flight software's controller) keeps the setpoint as it was, so the stop mode adds the stale
    acceleration feedforward -- after a stop inside a rotation ramp the rate damping then balances J alpha at
    w = alpha / omega_kd (0.0545 rad/s in STANDARD's face-forward turns), above the 0.02 rad/s stopping threshold, and
    the executive never reports 'stopped' (observed on a hub site at t = 628 s).  A move that reaches the
    end of its segment is not affected (its last sample has zero rates).
    """

    def __init__(self, propulsion: PropulsionParams | None = None, allocator: str = "flight", mode: str = "standard",
                 ref: dict | None = None, overspeed_stop: float | None = 0.5, overspeed_cutoff: float | None = 0.75,
                 ready_tol: float = 0.1, stop_feedforward: str = "zero"):
        if stop_feedforward not in ("zero", "hold"):
            raise ValueError(f"stop_feedforward {stop_feedforward!r} not in ('zero', 'hold')")
        self.stop_feedforward = stop_feedforward
        self.ref = load_firmware_reference() if ref is None else ref
        self.modes = load_flight_modes(self.ref)
        mp = self.ref["mass_properties_configured"]
        self.ctl = FlightController(self.ref["control"], mp["mass_kg"], mp["inertia_kgm2"])
        p = propulsion if propulsion is not None else PropulsionParams.from_dict(self.ref["propulsion"])
        cls = FlightAllocator if allocator == "flight" else NNLSAllocator
        self.alloc = cls(p, mp["com_m"])
        self.p = p
        self.exec = self.ref["executive"]
        self.overspeed_stop, self.overspeed_cutoff = overspeed_stop, overspeed_cutoff
        self.ready_tol = ready_tol
        self.mode = self.modes[mode]
        self.ctl_mode = MODE_STOP
        self.segment = []
        self.last_cmd = None
        self.state = "stopping"
        self.result = None
        self.faults = []
        self._tol_timer = {"pos": None, "att": None}
        self._prep_start = None
        self._pending = None
        self.last = {}

    # -- commands --------------------------------------------------------------------------------------
    def set_mode(self, name: str) -> None:
        """Switch flight mode (gains, speed level, limits); a speed change waits for the impellers."""
        new = self.modes[name]
        if new.speed != self.mode.speed:
            self._prep_start = None
            self.state = "preparing"
        self.mode = new

    def move(self, poses, faceforward: bool = True, t_now: float = 0.0, est=None, desired: dict | None = None):
        """Plan from the current estimated pose through ``poses`` [(pos, quat), ...] and execute."""
        start = (est.pos, est.quat) if est is not None else poses[0]
        plist = [start] + list(poses) if est is not None else list(poses)
        self.segment = plan_trapezoid(plist, self.mode, faceforward, t0=t_now, desired=desired)
        self.result = None
        self._tol_timer = {"pos": None, "att": None}
        if self.state == "preparing":
            self._pending = "move"
            return self.segment
        self.state = "moving" if self.segment else "stopping"
        self.ctl_mode = MODE_NOMINAL if self.segment else MODE_STOP
        if not self.segment:
            self.result = "already_there"
        return self.segment

    def stop(self) -> None:
        """Stop (zero velocity, then latch the pose); the last setpoint's pose is kept, its feedforward zeroed (or
        kept as the firmware does with stop_feedforward='hold')."""
        if self.state == "moving":
            self.result = "cancelled"
            self._stop_feedforward()
        self.state = "stopping"
        self.ctl_mode = MODE_STOP

    def _stop_feedforward(self) -> None:
        c = self.last_cmd
        if self.stop_feedforward == "zero" and c is not None:
            z = np.zeros(3)
            self.last_cmd = Setpoint(c.t, np.asarray(c.pos, float).copy(), z, z.copy(), np.asarray(c.quat, float).copy(),
                                     z.copy(), z.copy())

    def hold(self) -> None:
        """Station keeping = stop and let the controller latch the pose."""
        self.stop()

    def idle(self) -> None:
        """Zero force and torque (the robot drifts); impellers keep the flight-mode speed."""
        self.state = "idle"
        self.ctl_mode = MODE_IDLE

    # -- tick ------------------------------------------------------------------------------------------
    def speed_level(self) -> int:
        """Impeller speed level of the current flight mode (0..3)."""
        return self.mode.speed

    def tick(self, t: float, est: ControlState, impeller_speed=None) -> dict:
        """One control period: executive logic, controller, allocation."""
        lvl = self.speed_level()
        if self.state == "preparing":
            ok = impeller_speed is not None and all(
                abs(int(self.p.imp_cmd_per_rad_s * self.p.imp_levels[lvl - 1] + 0.5) / self.p.imp_cmd_per_rad_s - w) < self.ready_tol
                if lvl > 0 else abs(w) < self.ready_tol for w in impeller_speed)
            if self._prep_start is None:
                self._prep_start = t
            if ok:
                self.state = "stopping"
                if self._pending == "move" and self.segment:
                    shift = t - self.segment[0].t
                    for sp in self.segment:
                        sp.t += shift
                    self.state, self.ctl_mode = "moving", MODE_NOMINAL
                self._pending = None
            elif t - self._prep_start > self.exec["speed_ramp_timeout_s"]:
                self.state, self.result = "stopping", "speed_timeout"
            if self.state == "preparing":
                self.ctl_mode = MODE_STOP
        speed = float(np.linalg.norm(est.vel))
        if self.overspeed_cutoff is not None and speed > self.overspeed_cutoff and self.state != "idle":
            self.result = "overspeed_cutoff"
            self.idle()
        elif self.overspeed_stop is not None and speed > self.overspeed_stop and self.state == "moving":
            self.stop()
            self.result = "overspeed_stop"
        if est.confidence != 0 and self.state == "moving":
            self.result = "localization_lost"
            self._stop_feedforward()
            self.state = "stopping"
            self.ctl_mode = MODE_STOP
        cmd, tdelta = self.last_cmd, 0.0
        if self.state == "moving":
            sp, k = segment_setpoint(self.segment, t)
            if sp is None:
                self.ctl_mode = MODE_STOP
            else:
                cmd = sp
                self.last_cmd = sp
                self.ctl_mode = MODE_NOMINAL
                if t >= self.segment[-1].t:
                    self.ctl_mode = MODE_STOP
                    self.state = "stopping"
                    if self.mode.tolerance_pos_endpoint > 0 and self.ctl.out.get("traj_error_pos", 0.0) > self.mode.tolerance_pos_endpoint:
                        self.result = "tolerance_pos_endpoint"
        if cmd is not None:
            tdelta = t - cmd.t
        out = self.ctl.step(tdelta, est, cmd, self.ctl_mode if self.mode.control_enabled else MODE_IDLE, self.mode)
        if self.state == "moving":
            for key, tol, err in (("pos", self.mode.tolerance_pos, out["traj_error_pos"]),
                                  ("att", self.mode.tolerance_att, out["traj_error_att"])):
                if tol > 0 and err > tol:
                    if self._tol_timer[key] is None:
                        self._tol_timer[key] = t
                    if t - self._tol_timer[key] > self.exec["tolerance_max_time_s"]:
                        self.result = "tolerance_" + key
                        self.stop()
                        self.result = "tolerance_" + key
                else:
                    self._tol_timer[key] = None
        if self.state == "stopping" and out["ctl_status"] in (MODE_STOP, MODE_STOPPED):
            thr = math.sqrt(self.ref["control"]["stopping_vel_thresh_sq"])
            if np.linalg.norm(est.vel) <= thr and np.linalg.norm(est.omega) <= thr:
                self.state = "stopped"
                if self.result is None:
                    self.result = "success"
        al = self.alloc.allocate(out["force"], out["torque"], lvl)
        self.last = dict(out, **al, state=self.state, result=self.result, mode=self.mode.name)
        return self.last
