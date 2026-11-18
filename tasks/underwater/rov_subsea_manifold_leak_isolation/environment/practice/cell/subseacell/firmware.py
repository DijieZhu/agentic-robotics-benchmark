"""The vehicle firmware of task 07 (cell side; DESIGN v1.2 section 1.4; decisions J4, K3, U2, U3, U8; package P04).

The autopilot is given to the program, it is not what the task tests.  This module is the product version of the
gate-G1 prototype (proto/g1_valve_turn/firmware.py, the code under every phase 1 valve-turn and grasp run); the control
law is unchanged (tests/test_firmware.py replays a scripted sequence against the prototype bit for bit).  New here: the
protective behaviours are executed (protect.py), state() has the public key table STATE_SCHEMA, the public calls check
their arguments and answer {"accepted": ...}, the rated capability table of the manual (rated_capability()).

It is built on the shared simulation kit: its autopilot supplies the gains (loop shaping on a nominal plant), the
integrator clamps, rate and output-slew limits; its allocator the prioritised, direction-preserving group scaling.
The firmware reads ONLY the on-board sensors (AHRS 50 Hz, depth 20 Hz, DVL + dead-reckoning navigation 10 Hz), the
battery's terminal voltage and the tilt servo's reading; no simulator truth enters the control law.

Modes (the only names that reach the program; the kit's own mode names never appear in state(), refusals or docs):

  call                                                         mode       needs the DVL bottom lock
  idle()                                                       idle       no      thrusters stopped
  set_attitude(roll=0, pitch=0, heading=None, depth=None,      attitude   no      quaternion attitude setpoint (any
               thrust=(0, 0, 0), quat=None)                                       attitude); optional depth keeping;
                                                                                  thrust = normalised body thrust x
                                                                                  the rated axis capacity (thrust z
                                                                                  is ignored while depth is held)
  set_velocity(u, v, depth, heading)                           velocity   yes     body velocity over ground, depth,
                                                                                  heading; level attitude
  set_hold(x, y, depth, heading)                               station    yes     station keeping at (x, y) of the
                                                                                  navigation frame; level attitude
  set_wrench(force=(0, 0, 0), torque=(0, 0, 0), free=all six,  wrench     only the held horizontal axes
             frame="body", quat=None, depth=None)
      `free` selects BODY axes Fx Fy Fz Mx My Mz that take the open-loop force / torque; the others are held:
      held rotation axes hold `quat` (default: the attitude at the call; a repeated call with the same `free` keeps
      the previous one); held Fz = depth keeping at `depth` (default: the depth at the call) - the depth loop's thrust is
      a world-vertical thrust expressed in body axes, so it holds depth at any attitude, but the program's own thrust
      on body z is dropped while Fz is held (rolled 90 deg, body z is horizontal: the current feed-forward on that axis
      is lost; free all three translations and close depth yourself through a world-frame vertical thrust);
      held Fx / Fy = station keeping on that body axis at the position of the call (needs the bottom lock; without it
      a held horizontal axis gets zero thrust).  frame="world": the force is given in world axes and re-expressed in
      body axes every control period, then masked by `free`; the torque is always body axes.
      Torques (released rotation axes) are allocated with the attitude group (first), forces with the translation
      group (last) - decision J4.

Angles in rad; heading = yaw about +z (up) from +x (east), counter-clockwise; depth positive down, of the body origin
(depth sensor reading corrected for its lever arm with the attitude); positions in the navigation frame.

Rules
  * degradation: bottom lock lost in station / velocity -> depth + attitude hold continue; the horizontal thrust is
    held at its last value for degrade_hold_s from the loss, then ramped to zero over degrade_ramp_s; `degraded` is
    set once the gap has lasted lock_debounce_s and stays set until the next motion call; no automatic return to the
    commanded mode after re-lock.  A gap shorter than lock_debounce_s (dropped pings) is ridden out on the held
    thrust and is not a loss.  A station / velocity call made without bottom lock waits for the first lock (zero
    horizontal thrust, not degraded).  In wrench mode the held horizontal axes ride out short gaps the same way and
    get zero thrust after a loss (no `degraded` flag).
  * saturation: groups [rotation (attitude + released torques), depth, translation] are allocated in this order, each
    scaled as a whole (direction kept), each lower group using only the thrust left by the higher ones.
  * integrators: clamp = integrator_frac x the rated capacity of the axis (the 14.8 V column of rated_capability(),
    symmetric: the smaller of + and -), cleared at every mode change (a new `free` set counts as a mode change); a
    repeated call of the same mode only updates the setpoints (bumpless).
  * supply current: the battery's continuous / burst current budget is enforced on every physics step by scaling all
    eight thruster commands by one factor (the simulator's limiter, `sim.current_limit` must be on; U2); state()
    reports the factor.
  * protective behaviours (protect.py): collision stop = thrusters off, then depth + attitude hold at the attitude and
    depth of that moment, then the hold stays until the next motion call; motion calls are refused while a stop is
    active; disabled / out of bounds / undervoltage end the episode (the cell reads protect.ended).

Cell loop (per physics step; the order of the phase 1 worlds):
    if k % k_ctrl == 0: sim.set_thrusters(fw.control(sim.time))      # every control_dt
    <devices tick, grasp pre>; sim.step(); fw.ingest(suite.step(sim.time)); <grasp post>
    fw.physics_tick(sim.time)                                        # protective behaviours
No number of the reference vehicle is written here (everything comes from the built model and its VehicleSpec).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from uwkit.control.autopilot import Autopilot, AutopilotGains, _rot_error, _slew, V_NOMINAL
from uwkit.control.autopilot import MODES as _KIT_MODES
from uwkit.frames import euler_to_quat, quat_to_mat, quat_to_euler

from .protect import Protection

MODES = ("idle", "attitude", "velocity", "station", "wrench")
CALL_MODE = {"idle": "idle", "set_attitude": "attitude", "set_velocity": "velocity", "set_hold": "station",
             "set_wrench": "wrench"}
MOTION_CALLS = tuple(CALL_MODE)
GROUPS = ("rotation", "depth", "translation")
AXES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
NAV_STATUS = ("LOCK", "HOLD", "STALE")
GRIPPER_STATES = ("open", "closed_empty", "stalled_on_object")
ALL = (True,) * 6
FRAMES = ("body", "world")

# the shared kit's own autopilot mode names: never in anything the program receives (DESIGN section 5).  Taken from the
# kit at import time, so this file does not spell them (the practice service's copy is scanned for them, P27).
KIT_MODE_NAMES = tuple(_KIT_MODES)


@dataclass(frozen=True)
class FirmwareConfig:
    control_dt: float = 0.01               # firmware.control_period_s
    degrade_hold_s: float = 2.0            # firmware.degrade.hold_last_s
    degrade_ramp_s: float = 1.0            # firmware.degrade.ramp_down_s
    integrator_frac: float = 0.9           # firmware.integrator_limit_frac
    lock_debounce_s: float = 1.0           # a bottom-lock gap shorter than this is not a lock loss (P04; dropped pings)

    VALUE_KEYS = {"firmware.control_period_s": "control_dt", "firmware.degrade.hold_last_s": "degrade_hold_s",
                  "firmware.degrade.ramp_down_s": "degrade_ramp_s",
                  "firmware.integrator_limit_frac": "integrator_frac",
                  "firmware.lock_debounce_s": "lock_debounce_s"}          # P24a (decision P21-1)

    @classmethod
    def from_values(cls, values: dict) -> "FirmwareConfig":
        return cls(**{f: float(values[k]) for k, f in cls.VALUE_KEYS.items() if k in values})


# --------------------------------------------------------------------------------------------------- state() table
# (key, type, unit, description, filled by).  This is the table of the API document (P21 renders it); state() returns
# exactly these keys.  Descriptions are agent-visible text.
STATE_SCHEMA = (
    ("t", "float", "s", "simulation clock", "firmware"),
    ("nav_position_m", "float[3]", "m", "navigation estimate of the vehicle position: x (east) and y (north) by dead "
     "reckoning on the bottom-track velocity (reset with set_nav_fix), z = minus the depth-sensor depth", "firmware"),
    ("nav_velocity_m_s", "float[3]", "m/s", "velocity over ground in world axes used by the estimate; held at the last "
     "value while the bottom lock is lost", "firmware"),
    ("nav_cov_m2", "float[2][2]", "m^2", "nominal covariance of the horizontal position estimate (from the "
     "estimator's own assumptions, not the true error)", "firmware"),
    ("nav_status", "str", "-", "LOCK (bottom lock) | HOLD (lock lost, last velocity held) | STALE (no valid "
     "velocity for longer than the hold time, or none since power-up)", "firmware"),
    ("nav_since_lock_s", "float|null", "s", "time since the last valid bottom-track velocity (0 while locked, null "
     "if there was none)", "firmware"),
    ("depth_m", "float|null", "m", "depth of the body origin, positive down (depth sensor, lever arm corrected with "
     "the attitude); the quantity of every depth setpoint", "firmware"),
    ("altitude_m", "float|null", "m", "DVL altitude above the bottom (null without three valid beams)", "firmware"),
    ("dvl_lock", "bool", "-", "the last DVL velocity was valid (bottom lock)", "firmware"),
    ("dvl_beam_range_m", "(float|null)[4]", "m", "slant range along each DVL beam (null: no return on that beam)",
     "firmware"),
    ("dvl_beam_valid", "bool[4]", "-", "per-beam validity of the last DVL report", "firmware"),
    ("quat_wxyz", "float[4]|null", "-", "attitude estimate (AHRS): body (x forward, y left, z up) to world (x east, y "
     "north, z up)", "firmware"),
    ("heading_rad", "float|null", "rad", "yaw of the attitude estimate about +z from +x (east), counter-clockwise; the "
     "convention of every heading argument", "firmware"),
    ("rates_b_rad_s", "float[3]|null", "rad/s", "body angular rates (gyro, bias corrected)", "firmware"),
    ("velocity_b_m_s", "float[3]|null", "m/s", "velocity over ground in body axes from the navigation estimate (null "
     "before the first bottom lock)", "firmware"),
    ("battery_V", "float", "V", "battery terminal voltage", "firmware"),
    ("battery_A", "float", "A", "battery current (thrusters + hotel load)", "firmware"),
    ("battery_soc", "float", "-", "state of charge, 0..1", "firmware"),
    ("mode", "str", "-", "idle | attitude | velocity | station | wrench", "firmware"),
    ("degraded", "bool", "-", "station / velocity lost the bottom lock since the last motion call (depth and attitude "
     "still held, horizontal thrust held, then ramped to zero)", "firmware"),
    ("wrench_groups", "{rotation, depth, translation: float[6]}", "N, N m", "body wrench [Fx, Fy, Fz, Mx, My, Mz] "
     "delivered by each allocation group after scaling: rotation = attitude loop and released-axis torques, depth = "
     "depth loop, translation = horizontal loops, normalised thrust and released-axis forces", "firmware"),
    ("group_scale", "{rotation, depth, translation: float}", "-", "scale factor applied to each group (1 = delivered "
     "in full; a lower group only gets the thrust left by the higher ones)", "firmware"),
    ("saturated", "bool", "-", "at least one group was scaled below 1", "firmware"),
    ("current_limit_scale", "float", "-", "factor applied to all eight thruster commands by the supply-current "
     "limit (1 = not limiting)", "firmware"),
    ("thrusters", "float[8]", "-", "normalised thruster commands of the autopilot, -1..1 (before the supply-current "
     "factor)", "firmware"),
    ("camera_tilt_deg", "float|null", "deg", "camera tilt servo reading, optical axis up positive", "firmware"),
    ("gripper", "str", "-", "open | closed_empty | stalled_on_object (from the jaw travel and stall; no object "
     "identity)", "cell"),
    ("protect_stops", "int", "-", "collision protective stops so far", "firmware"),
    ("protect_phase", "str|null", "-", "null | stop (thrusters off) | hold (depth and attitude hold) | surfacing | "
     "disabled; motion calls are refused while it is not null", "firmware"),
    ("protect_remaining_s", "float", "s", "time left in the current stop and hold (0 when none)", "firmware"),
    ("budget", "object", "-", "budget usage of this run (filled by the cell)", "cell"),
)
STATE_KEYS = tuple(r[0] for r in STATE_SCHEMA)


class CallError(ValueError):
    """A public call with invalid arguments (the message is neutral and safe to show to the program)."""


@dataclass
class _Sensed:
    t: float = 0.0
    q: np.ndarray = None
    w: np.ndarray = None
    depth: float = None          # body origin depth [m], positive down
    depth_t: float = None
    nav: object = None           # last NavSolution
    dvl: object = None           # last DVLReading


# --------------------------------------------------------------------------------------------------- argument checks
def _num(x, name):
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise CallError(f"{name} must be a number") from None
    if not math.isfinite(v):
        raise CallError(f"{name} must be finite")
    return v


def _vec(x, n, name):
    try:
        a = np.asarray(x, float).reshape(-1)
    except (TypeError, ValueError):
        raise CallError(f"{name} must be {n} numbers") from None
    if a.size != n:
        raise CallError(f"{name} must be {n} numbers")
    if not np.all(np.isfinite(a)):
        raise CallError(f"{name} must be finite")
    return a


def _free_mask(free):
    if isinstance(free, str):
        free = [free]
    try:
        items = list(free)
    except TypeError:
        raise CallError("free must be six booleans or a list of axis names") from None
    if items and all(isinstance(f, str) for f in items):
        bad = [f for f in items if f not in AXES]
        if bad:
            raise CallError(f"free: unknown axis name; use {', '.join(AXES)}")
        return tuple(a in items for a in AXES)
    if len(items) != 6:
        raise CallError("free must be six booleans (Fx, Fy, Fz, Mx, My, Mz) or a list of axis names")
    out = []
    for f in items:
        if isinstance(f, (bool, np.bool_)):
            out.append(bool(f))
        elif isinstance(f, (int, np.integer)) and int(f) in (0, 1):
            out.append(bool(f))
        else:
            raise CallError("free must be six booleans (Fx, Fy, Fz, Mx, My, Mz) or a list of axis names")
    return tuple(out)


def _quat(q, name="quat", normalise=True):
    a = _vec(q, 4, name)
    n = float(np.linalg.norm(a))
    if n < 1e-6:
        raise CallError(f"{name} must be a non-zero quaternion (w, x, y, z)")
    return a / np.linalg.norm(a) if normalise else a


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _f(x):
    """JSON-safe float (NaN / inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def _fl(a):
    return None if a is None else [_f(v) for v in np.asarray(a, float).reshape(-1)]


# --------------------------------------------------------------------------------------------------- firmware
class Firmware:
    def __init__(self, sim, suite, vehicle_spec, *, config: FirmwareConfig = None, protect: Protection = None,
                 gains: AutopilotGains = None):
        self.cfg = config or FirmwareConfig()
        g = gains or AutopilotGains.tuned(vehicle_spec)
        if g.i_frac != self.cfg.integrator_frac:
            g = replace(g, i_frac=self.cfg.integrator_frac)
        self.sim, self.suite = sim, suite
        self.ap = Autopilot(vehicle_spec, gains=g, control_dt=self.cfg.control_dt)
        self.g = self.ap.g
        self.alloc = self.ap.alloc
        self.dt = float(self.cfg.control_dt)
        self.cap = self.ap.capacity
        if not getattr(sim, "current_limit", False):
            raise ValueError("the supply-current limit is part of the firmware: build the simulation with "
                             "current_limit=True")
        m = sim.model
        sid = m.site("depth").id
        self.r_depth_b = np.array(m.site_pos[sid], float) if m.site_bodyid[sid] == sim.hull else np.zeros(3)
        self.protect = protect
        self.s = _Sensed()
        self.calls = []                    # public call log (name, t, accepted)
        self.mode = "idle"
        self.degraded = False
        self._had_lock = False
        self._reset_ctrl()
        self.sp = {}
        self.status = dict(alpha=[1.0, 1.0, 1.0], groups=[np.zeros(6)] * 3, wrench=np.zeros(6), mode="idle",
                           saturated=False)
        self.u = np.zeros(self.alloc.n)
        self._ab = None
        self._t_lock_lost = None
        self._t_gap = None
        self._F_last = np.zeros(2)
        self._prot_phase = None

    # ======================================================================================= public motion calls
    def _refusal(self):
        """None if motion calls are accepted now, else the refusal answer."""
        P = self.protect
        if P is None:
            return None
        t = self.sim.time
        if P.ended:
            return dict(accepted=False, reason=P.end_reason, protect_phase=P.phase(t))
        ph = P.phase(t)
        if ph is not None:
            return dict(accepted=False, reason="protective_stop", protect_phase=ph,
                        remaining_s=round(P.remaining_s(t), 3))
        return None

    def _accept(self, name):
        r = self._refusal()
        self.calls.append((name, round(self.sim.time, 3), r is None))
        return r

    def idle(self):
        r = self._accept("idle")
        if r is not None:
            return r
        self._enter("idle")
        return dict(accepted=True, mode=self.mode)

    def set_attitude(self, roll=0.0, pitch=0.0, heading=None, depth=None, thrust=(0.0, 0.0, 0.0), quat=None):
        if quat is not None:
            q = _quat(quat, normalise=False)              # normalised once, in _set_attitude
        else:
            r_, p_ = _num(roll, "roll"), _num(pitch, "pitch")
            h_ = None if heading is None else _num(heading, "heading")
            q = None
        d_ = None if depth is None else _num(depth, "depth")
        th = np.clip(_vec(thrust, 3, "thrust"), -1, 1)
        r = self._accept("set_attitude")
        if r is not None:
            return r
        if q is None:
            q = euler_to_quat(r_, p_, self._yaw_now() if h_ is None else h_)
        self._set_attitude(q, d_, th)
        return dict(accepted=True, mode=self.mode)

    def _set_attitude(self, q, depth, thrust):
        self._enter("attitude", quat=np.asarray(q, float) / np.linalg.norm(q), depth=depth, thrust=thrust)
        if depth is not None:
            self.depth_sp = float(depth)

    def set_velocity(self, u, v, depth, heading):
        u_, v_, d_, h_ = _num(u, "u"), _num(v, "v"), _num(depth, "depth"), _num(heading, "heading")
        r = self._accept("set_velocity")
        if r is not None:
            return r
        self._enter("velocity", vel=np.array([u_, v_], float), depth=d_, heading=h_)
        self.depth_sp = d_
        return dict(accepted=True, mode=self.mode)

    def set_hold(self, x, y, depth, heading):
        x_, y_, d_, h_ = _num(x, "x"), _num(y, "y"), _num(depth, "depth"), _num(heading, "heading")
        r = self._accept("set_hold")
        if r is not None:
            return r
        self._enter("station", pos=np.array([x_, y_], float), depth=d_, heading=h_)
        self.depth_sp = d_
        self.pos_sp = np.array([x_, y_], float)
        return dict(accepted=True, mode=self.mode)

    def set_wrench(self, force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0), free=ALL, frame="body", quat=None,
                   depth=None):
        fv, tv = _vec(force, 3, "force"), _vec(torque, 3, "torque")
        free = _free_mask(free)
        if frame not in FRAMES:
            raise CallError("frame must be 'body' or 'world'")
        qv = None if quat is None else _quat(quat)
        d_ = None if depth is None else _num(depth, "depth")
        r = self._accept("set_wrench")
        if r is not None:
            return r
        same = self.mode == "wrench" and self.sp.get("free") == free
        if qv is not None:
            q_hold = qv
        elif same and self.sp.get("quat") is not None:
            q_hold = self.sp["quat"]
        elif self.s.q is not None:
            q_hold = self.s.q.copy()
        else:
            q_hold = None
        self._enter("wrench", force=fv, torque=tv, free=free, frame=frame, quat=q_hold)
        if not same:
            self._reset_ctrl()                     # a new free set counts as a mode change
            self._had_lock = False
            self.depth_sp = self.s.depth if d_ is None else d_
            nav = self.s.nav
            self.pos_sp = None if nav is None else np.array(nav.position[:2], float)
        elif d_ is not None:
            self.depth_sp = d_
        return dict(accepted=True, mode=self.mode)

    def set_nav_fix(self, x, y):
        """Reset the navigation estimate's horizontal position (the program's own fix)."""
        x_, y_ = _num(x, "x"), _num(y, "y")
        self.suite.nav.reset((x_, y_))
        n = self.s.nav
        if n is not None:
            pos = np.array(n.position, float)
            pos[0], pos[1] = x_, y_
            self.s.nav = replace(n, position=pos)
        return dict(accepted=True)

    # ======================================================================================= mode bookkeeping
    def _enter(self, mode, **sp):
        if mode != self.mode or self.degraded:
            self._had_lock = False
        if mode != self.mode:
            self._reset_ctrl()
            self.mode = mode
        self.degraded = False
        self._t_lock_lost = None
        self._t_gap = None
        self.sp = sp

    def _reset_ctrl(self):
        """Mode change: integrators and setpoint memories cleared; the output slew state is kept (no thrust step)."""
        self.i_rate = np.zeros(3)
        self.i_vz = 0.0
        self.i_vel = np.zeros(2)
        self.vel_sp = np.zeros(2)
        if not hasattr(self, "tau_prev"):
            self.tau_prev = [np.zeros(6), np.zeros(6), np.zeros(6)]
        self.depth_sp = None
        self.pos_sp = None

    # ======================================================================================= sensing
    def ingest(self, out):
        """Feed the readings of one SensorSuite.step() call."""
        a = out.get("ahrs") or []
        if a:
            r = a[-1]
            self.s.q = np.asarray(r.quat, float)
            self.s.w = np.asarray(r.angular_velocity, float)
            self.s.t = r.t
        dp = out.get("depth") or []
        if dp and self.s.q is not None:
            r = dp[-1]
            R = quat_to_mat(self.s.q)
            dep = float(r.depth_m) + float((R @ self.r_depth_b)[2])
            if self._ab is None:
                self._ab = [dep, 0.0, r.t]
            else:
                x, v, t0 = self._ab
                h = max(r.t - t0, 1e-3)
                x += v * h
                res = dep - x
                x += 0.4 * res
                v += 0.08 * res / h
                self._ab = [x, v, r.t]
            self.s.depth, self.s.depth_t = dep, r.t
        nv = out.get("nav") or []
        if nv:
            self.s.nav = nv[-1]
        dv = out.get("dvl") or []
        if dv:
            self.s.dvl = dv[-1]

    def locked(self):
        n = self.s.nav
        return bool(n is not None and n.dvl_lock and n.status == "LOCK")

    def _yaw_now(self):
        return float(quat_to_euler(self.s.q)[2]) if self.s.q is not None else 0.0

    # ======================================================================================= protective behaviours
    def physics_tick(self, t):
        """Run the protective behaviours (call after every physics step, after ingest).  Returns the new events."""
        P = self.protect
        if P is None:
            return []
        ev = P.tick(t, self.sim.battery.voltage)
        ph = P.phase(t)
        if ph != self._prot_phase or P.new_stop:
            self._on_protect_phase(self._prot_phase, ph)
            self._prot_phase = ph
        return ev

    def _on_protect_phase(self, old, new):
        if new in ("stop", "disabled"):
            self._enter("idle")
            self.u = np.zeros(self.alloc.n)
            self.sim.set_thrusters(self.u)          # thrusters cut at once, not at the next control period
        elif new == "hold":
            q = self.s.q.copy() if self.s.q is not None else np.array([1.0, 0.0, 0.0, 0.0])
            self._set_attitude(q, self.s.depth, np.zeros(3))
        elif new == "surfacing":
            self._set_attitude(euler_to_quat(0.0, 0.0, self._yaw_now()), 0.0, np.zeros(3))
        # new is None after "hold": the depth + attitude hold stays until the next motion call

    # ======================================================================================= control
    def control(self, t):
        """One control period: returns the normalised thruster commands."""
        g, dt = self.g, self.dt
        V = round(float(self.sim.battery.voltage), 2)
        mode = self.mode
        if mode == "idle" or self.s.q is None or self.s.depth is None:
            self.u = np.zeros(self.alloc.n)
            self.status = dict(alpha=[1.0, 1.0, 1.0], groups=[np.zeros(6)] * 3, wrench=np.zeros(6), mode=mode,
                               saturated=False)
            return self.u
        sp = self.sp
        q, w = self.s.q, self.s.w
        R = quat_to_mat(q)
        lock = self.locked()
        nav = self.s.nav
        fm = np.array(sp["free"], bool) if mode == "wrench" else np.zeros(6, bool)
        tau_att, tau_dep, tau_tr = np.zeros(6), np.zeros(6), np.zeros(6)
        # ---------------- attitude setpoint
        if mode == "attitude":
            q_sp = sp["quat"]
        elif mode in ("velocity", "station"):
            q_sp = euler_to_quat(0.0, 0.0, sp["heading"])
        else:
            q_sp = sp["quat"] if sp.get("quat") is not None else q
        e_att = _rot_error(q, q_sp)
        w_sp = np.clip(self.ap._att_p * e_att, -self.ap._rate_max, self.ap._rate_max)
        e_w = w_sp - w
        tau_rot = self.ap._rate_p * e_w + self.ap._rate_i * self.i_rate
        if mode == "wrench":
            tau_att[3:] = np.where(fm[3:], sp["torque"], tau_rot)
        else:
            tau_att[3:] = tau_rot
        # ---------------- depth
        depth_on = (mode in ("velocity", "station") or (mode == "attitude" and sp.get("depth") is not None) or
                    (mode == "wrench" and not fm[2]))
        e_vz = 0.0
        if depth_on:
            if self.depth_sp is None:
                self.depth_sp = self.s.depth
            if lock and nav is not None:
                d_rate = -float(nav.velocity[2])
            else:
                d_rate = float(self._ab[1]) if self._ab else 0.0
            e_d = self.depth_sp - self.s.depth
            wz_sp = min(max(g.depth_p * e_d, -g.depth_rate_up), g.depth_rate_down)
            e_vz = wz_sp - d_rate
            F_down = g.vz_p * e_vz + g.vz_i * self.i_vz
            tau_dep[:3] = R.T @ np.array([0.0, 0.0, -F_down])
        # ---------------- horizontal
        e_v = np.zeros(2)
        horiz_loop = False
        if mode == "attitude":
            tau_tr[:3] = sp["thrust"] * self.cap[:3]
            if sp.get("depth") is not None:
                tau_tr[2] = 0.0
        elif mode in ("velocity", "station") or (mode == "wrench" and not (fm[0] and fm[1])):
            if lock and not self.degraded:
                horiz_loop = True
                self._had_lock = True
                self._t_gap = None
                vb = R.T @ np.asarray(nav.velocity, float)
                if mode == "velocity":
                    target = sp["vel"]
                else:
                    p = np.asarray(nav.position[:2], float) + np.asarray(nav.velocity[:2], float) * max(0.0, t - nav.t)
                    if self.pos_sp is None:
                        self.pos_sp = p.copy()
                    if mode == "wrench":
                        for ax in (0, 1):
                            if fm[ax]:
                                e_ax = R[:2, ax] / max(np.linalg.norm(R[:2, ax]), 1e-9)
                                self.pos_sp = self.pos_sp + e_ax * float(e_ax @ (p - self.pos_sp))
                    e_p = self.pos_sp - p
                    v_w = g.pos_p * e_p
                    s_ = float(np.linalg.norm(v_w))
                    if s_ > g.vel_max:
                        v_w *= g.vel_max / s_
                    target = (R.T @ np.array([v_w[0], v_w[1], 0.0]))[:2]
                self.vel_sp = _slew(self.vel_sp, target, g.accel_max * dt)
                e_v = self.vel_sp - vb[:2]
                F_xy = self.ap._vel_p * e_v + self.ap._vel_i * self.i_vel
                tau_tr[:2] = np.where(fm[:2], 0.0, F_xy)
                self._F_last = tau_tr[:2].copy()
            elif mode in ("velocity", "station") and not self._had_lock:
                tau_tr[:2] = 0.0              # no bottom lock yet since the call: wait for it (not a degradation)
            else:
                # no bottom lock: the horizontal thrust is held at its last value from the first control period
                # without lock (t_gap).  A gap shorter than lock_debounce_s (single dropped pings) is ridden out;
                # a longer one is a lock loss: station / velocity degrade (depth + attitude hold continue, thrust held
                # until t_gap + degrade_hold_s, ramped to zero by t_gap + degrade_hold_s + degrade_ramp_s), wrench
                # gives the held horizontal axes zero thrust.
                if self._t_gap is None:
                    self._t_gap = t
                el = t - self._t_gap
                lost = self.degraded or el >= self.cfg.lock_debounce_s - 1e-9
                if mode in ("velocity", "station"):
                    if lost and not self.degraded:
                        self.degraded = True
                        self._t_lock_lost = self._t_gap
                        self.i_vel[:] = 0.0
                        self.vel_sp[:] = 0.0
                    hold, ramp = self.cfg.degrade_hold_s, self.cfg.degrade_ramp_s
                    k = 1.0 if el < hold else max(0.0, 1.0 - (el - hold) / ramp)
                    tau_tr[:2] = self._F_last * k
                elif not lost and self._had_lock:
                    tau_tr[:2] = np.where(fm[:2], 0.0, self._F_last)
                else:
                    tau_tr[:2] = 0.0
        if mode == "wrench":
            f = sp["force"]
            fb = R.T @ f if sp["frame"] == "world" else f
            tau_tr[:3] = tau_tr[:3] + np.where(fm[:3], fb, 0.0)
        # ---------------- slew + allocation
        lim = self.ap._slew_lim
        groups = []
        for k, tau in enumerate((tau_att, tau_dep, tau_tr)):
            tt = _slew(self.tau_prev[k], tau, lim)
            self.tau_prev[k] = tt
            groups.append(tt)
        u, ainfo = self.alloc.allocate(groups, V)
        a_att, a_dep, a_tr = ainfo["alpha"]
        # ---------------- integrators (conditional integration + clamps)
        lim_r = self.ap._i_lim_rot
        held_rot = ~fm[3:]
        grow = np.sign(e_w) == np.sign(self.i_rate)
        ok = (a_att > 0.999) | ~grow
        self.i_rate = np.clip(np.where(ok & held_rot, self.i_rate + e_w * dt, self.i_rate), -lim_r, lim_r)
        if depth_on:
            in_zone = abs(self.depth_sp - self.s.depth) < g.depth_i_zone
            if in_zone and (a_dep > 0.999 or (e_vz > 0) != (self.i_vz > 0)):
                self.i_vz += e_vz * dt
            self.i_vz = min(max(self.i_vz, -self.ap._i_lim_z), self.ap._i_lim_z)
        if horiz_loop:
            growv = np.sign(e_v) == np.sign(self.i_vel)
            okv = (a_tr > 0.999) | ~growv
            self.i_vel = np.clip(np.where(okv & ~fm[:2], self.i_vel + e_v * dt, self.i_vel),
                                 -self.ap._i_lim_v, self.ap._i_lim_v)
        delivered = [a * gg for a, gg in zip(ainfo["alpha"], groups)]
        self.status = dict(mode=mode, alpha=list(ainfo["alpha"]), groups=delivered, wrench=ainfo["wrench"],
                           saturated=bool(min(ainfo["alpha"]) < 0.999), att_err=e_att, V=V)
        self.u = u
        return u

    # ======================================================================================= introspection (cell side)
    def integrator_terms(self):
        """Present integrator contributions [N or N m] and their clamps (tests / grader-side diagnostics only)."""
        return dict(rotation=self.ap._rate_i * self.i_rate, rotation_clamp=self.ap._rate_i * self.ap._i_lim_rot,
                    depth=self.g.vz_i * self.i_vz, depth_clamp=self.g.vz_i * self.ap._i_lim_z,
                    horizontal=self.ap._vel_i * self.i_vel, horizontal_clamp=self.ap._vel_i * self.ap._i_lim_v)

    def rated_capability(self, voltages=(12.0, 13.5, 14.8, 16.0), gripper_axis_b=None):
        return rated_capability(self.alloc, voltages, gripper_axis_b=gripper_axis_b, cap=self.cap,
                                integrator_frac=self.g.i_frac)

    # ======================================================================================= readout
    def state(self, gripper="open", budget=None):
        """The program's state() reply (keys = STATE_KEYS).  `gripper` and `budget` are filled by the cell."""
        if gripper not in GRIPPER_STATES:
            raise ValueError(f"gripper state must be one of {GRIPPER_STATES}")
        t = self.sim.time
        n, q, dv = self.s.nav, self.s.q, self.s.dvl
        bat = self.sim.battery.status()
        if n is not None:
            status = n.status if n.status in NAV_STATUS else "STALE"
            since = 0.0 if n.dvl_lock else (n.time_since_lock if math.isfinite(n.time_since_lock) else None)
            vb = None if n.status == "INIT" else quat_to_mat(n.quat).T @ np.asarray(n.velocity, float)
            nav = dict(nav_position_m=_fl(n.position), nav_velocity_m_s=_fl(n.velocity),
                       nav_cov_m2=[_fl(n.cov_pos[0, :2]), _fl(n.cov_pos[1, :2])], nav_status=status,
                       nav_since_lock_s=_f(since), velocity_b_m_s=_fl(vb))
        else:
            nav = dict(nav_position_m=None, nav_velocity_m_s=None, nav_cov_m2=None, nav_status="STALE",
                       nav_since_lock_s=None, velocity_b_m_s=None)
        if dv is not None:
            beams = dict(altitude_m=_f(dv.altitude), dvl_lock=bool(dv.valid),
                         dvl_beam_range_m=[_f(r) if bool(v) else None for r, v in zip(dv.beam_range, dv.beam_valid)],
                         dvl_beam_valid=[bool(v) for v in dv.beam_valid])
        else:
            beams = dict(altitude_m=None, dvl_lock=False, dvl_beam_range_m=[None] * 4, dvl_beam_valid=[False] * 4)
        try:
            tilt = round(math.degrees(self.sim.camera_tilt()), 2)
        except (KeyError, AttributeError):
            tilt = None
        P = self.protect
        if P is not None:
            pst = dict(protect_stops=int(P.count), protect_phase=P.phase(t),
                       protect_remaining_s=_f(P.remaining_s(t)) if math.isfinite(P.remaining_s(t)) else 0.0)
        else:
            pst = dict(protect_stops=0, protect_phase=None, protect_remaining_s=0.0)
        st = self.status
        out = dict(
            t=float(t), **nav, depth_m=_f(self.s.depth), **beams,
            quat_wxyz=_fl(q), heading_rad=None if q is None else _f(quat_to_euler(q)[2]),
            rates_b_rad_s=_fl(self.s.w),
            battery_V=_f(bat["voltage_V"]), battery_A=_f(bat["current_A"]), battery_soc=_f(bat["soc"]),
            mode=self.mode, degraded=bool(self.degraded),
            wrench_groups={k: _fl(v) for k, v in zip(GROUPS, st["groups"])},
            group_scale={k: _f(a) for k, a in zip(GROUPS, st["alpha"])},
            saturated=bool(st.get("saturated", False)),
            current_limit_scale=_f(getattr(self.sim, "current_scale", 1.0)),
            thrusters=_fl(self.u), camera_tilt_deg=tilt, gripper=gripper, **pst,
            budget=dict(budget) if budget is not None else {})
        return {k: out[k] for k in STATE_KEYS}


# --------------------------------------------------------------------------------------------------- rated capability
def rated_capability(alloc, voltages=(12.0, 13.5, 14.8, 16.0), gripper_axis_b=None, cap=None, integrator_frac=None):
    """What this firmware delivers along each pure body axis (+ and -) and about the gripper axis, from the thruster
    curve and the geometry only (no hydrodynamics): the allocator's direction-preserving maximum (one group, all
    other wrench components zero).  Body axes x forward, y left, z up; moments about the vehicle's centre of mass."""
    out = dict(voltages_V=[float(v) for v in voltages], axes={}, thruster_limits_N={})
    for V in voltages:
        c = alloc.axis_capacity(float(V))
        row = {}
        for i, nm in enumerate(AXES):
            row[nm + "+"] = float(c[i, 0])
            row[nm + "-"] = float(c[i, 1])
        key = "%.1f" % V
        out["axes"][key] = row
        lo, hi = alloc.limits(float(V))
        out["thruster_limits_N"][key] = dict(forward=float(np.max(hi)), reverse=float(-np.min(lo)))
        if gripper_axis_b is not None:
            a = np.asarray(gripper_axis_b, float)
            a = a / np.linalg.norm(a)
            ga = {}
            for sgn, nm in ((1.0, "torque+"), (-1.0, "torque-")):
                e = np.concatenate([np.zeros(3), sgn * a]) * 1e4
                _, al = alloc.scale_groups([e], float(V))
                ga[nm] = float(al[0] * 1e4)
            out.setdefault("gripper_axis", {})[key] = ga
    if cap is not None:
        frac = float(integrator_frac)
        out["integrator_limit"] = dict(basis_V=V_NOMINAL, fraction=frac,
                                       axes={nm: frac * float(cap[i]) for i, nm in enumerate(AXES)})
    return out
