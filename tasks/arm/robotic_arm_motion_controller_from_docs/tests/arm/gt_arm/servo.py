"""Firmware layer of the tier servo BUS-SERVO-12V-A for one unit (PLAN 3.2 U dynamic subset; W1-CTL).

What a bus command goes through before it moves the arm, and what a reading goes through before it reaches the bus:

  command  q_bus (rad) -> quantised to ticks (4096 / rev) -> clamped to the EEPROM angle-limit ticks -> a FIFO that
           releases it after the unit's command latency (U[15, 40] ms) -> the goal of a trapezoidal speed profile
           (max speed = the datasheet no-load speed x the unit's factor U[0.85, 1.0]; acceleration 40 rad/s^2; a
           `duration_s` request synchronises the joints to that time) -> the setpoint, converted to the design angle by
           the unit's TRUE calibration, is the target of a MuJoCo position actuator on the OUTPUT hinge (kp = 998.22 x the
           unit's gain U[0.85, 1.15], kv 2.731, force range +-2.94 N m = stall)
  encoder  the output hinge angle (the servo's own magnetic encoder sits on its output shaft, before the passive series
           compliance of the link: unit.py `<joint>_slack`) -> bus angle by the true calibration -> quantised to ticks
  load     the actuator force / stall x the unit's load-reading scale U[0.9, 1.1], in per mille of max torque, positive
           along +bus, + Gaussian noise of 1 % of stall; the servo refreshes the register every 20 ms, so repeated
           reads at one instant return the same value (a fresh sample costs bench time)

Stepping is the caller's (the bench simulator `ctl_info.sim`): `ServoBus.before_step(t)` sets the actuator targets for
the physics step that starts at time t. numpy only; names no source.
"""

import math

import numpy as np

TICKS = 4096
MID = 2048
TICK_RAD = 2.0 * math.pi / TICKS
NO_LOAD_SPEED_RAD_S = 6.49           # datasheet, 12 V (docs_synth tier_servo.FACTS)
STALL_NM = 2.94
ACCEL_RAD_S2 = 40.0
LOAD_REFRESH_S = 0.020
LOAD_NOISE_PERMILLE = 10.0
MOVING_SPEED_RAD_S = 0.01


def bus_to_ticks(b):
    return np.rint(np.asarray(b, float) / TICK_RAD + MID).astype(int)


def ticks_to_bus(t):
    return (np.asarray(t, float) - MID) * TICK_RAD


class ServoBus(object):
    """The six servos of one unit on one bus. `joints` = the six joint names (bus order = servo id 1..6).

    `cal` = {joint: {"sign", "zero_offset_rad", "min_ticks", "max_ticks"}} (the unit's TRUE calibration and the EEPROM
    limit registers); `dyn` = {"latency_s", "speed_factor" {j}, "load_scale" {j}}; `qadr` / `dadr` = qpos / qvel
    addresses of the output hinges; `act` = actuator ids in joint order; `noise_seed` = the load-noise stream's seed."""

    def __init__(self, model, joints, cal, dyn, qadr, dadr, act, noise_seed, t0=0.0, q0=None):
        self.m = model
        self.joints = list(joints)
        n = len(self.joints)
        self.sign = np.array([float(cal[j]["sign"]) for j in self.joints])
        self.zero = np.array([float(cal[j]["zero_offset_rad"]) for j in self.joints])
        self.tmin = np.array([int(cal[j]["min_ticks"]) for j in self.joints])
        self.tmax = np.array([int(cal[j]["max_ticks"]) for j in self.joints])
        self.latency = float(dyn["latency_s"])
        self.vmax = np.array([NO_LOAD_SPEED_RAD_S * float(dyn["speed_factor"].get(j, 1.0)) for j in self.joints])
        self.lscale = np.array([float(dyn["load_scale"].get(j, 1.0)) for j in self.joints])
        self.qadr = np.asarray(qadr, int)
        self.dadr = np.asarray(dadr, int)
        self.act = np.asarray(act, int)
        self.noise_seed = int(noise_seed) & ((1 << 63) - 1)
        self.fifo = []                                   # (t_due, goal_ticks[n], speed_limit[n] or None)
        q0 = np.zeros(n) if q0 is None else np.asarray(q0, float)
        b0 = self.q_to_bus(q0)
        self.sp = b0.copy()                              # profile setpoint (bus rad, continuous)
        self.sv = np.zeros(n)                            # setpoint velocity
        self.goal = b0.copy()
        self.vlim = self.vmax.copy()
        self._load_cache = {}

    # conversions (true calibration)
    def q_to_bus(self, q):
        return self.sign * (np.asarray(q, float) - self.zero)

    def bus_to_q(self, b):
        return np.asarray(b, float) * self.sign + self.zero

    def clamp_ticks(self, t):
        return np.minimum(np.maximum(np.asarray(t, int), self.tmin), self.tmax)

    # commands
    def command(self, t_now, goal_bus, duration_s=None):
        """Queue a goal (bus rad, quantised and clamped here). Returns the goal ticks. With `duration_s` the profile speed of
        each joint is |goal - setpoint at release| / duration (capped at the joint's max speed)."""
        gt = self.clamp_ticks(bus_to_ticks(goal_bus))
        self.fifo.append((float(t_now) + self.latency, gt, None if duration_s is None else float(duration_s)))
        return gt

    def _release(self, t):
        while self.fifo and self.fifo[0][0] <= t + 1e-12:
            _, gt, dur = self.fifo.pop(0)
            g = ticks_to_bus(gt)
            if dur is not None and dur > 0:
                self.vlim = np.minimum(self.vmax, np.maximum(np.abs(g - self.sp) / dur, 1e-3))
            else:
                self.vlim = self.vmax.copy()
            self.goal = g

    def before_step(self, t, dt):
        """Release due commands, advance the speed profile by dt, set the actuator targets (design angles)."""
        self._release(t)
        d = self.goal - self.sp
        vstop = np.sqrt(2.0 * ACCEL_RAD_S2 * np.abs(d))
        vdes = np.sign(d) * np.minimum(self.vlim, vstop)
        dv = np.clip(vdes - self.sv, -ACCEL_RAD_S2 * dt, ACCEL_RAD_S2 * dt)
        self.sv = self.sv + dv
        step = self.sv * dt
        arrive = (np.sign(step) == np.sign(d)) & (np.abs(step) >= np.abs(d))
        self.sp = np.where(arrive, self.goal, self.sp + step)
        self.sv = np.where(arrive | (np.abs(self.goal - self.sp) < 1e-12), 0.0, self.sv)
        return self.bus_to_q(self.sp)

    def profile_done(self):
        return (not self.fifo) and bool(np.all(np.abs(self.goal - self.sp) < 1e-12))

    # readings
    def encoder_ticks(self, data):
        return bus_to_ticks(self.q_to_bus(data.qpos[self.qadr]))

    def encoder_bus(self, data):
        return ticks_to_bus(self.encoder_ticks(data))

    def moving(self, data):
        return (not self.profile_done()) or bool(np.max(np.abs(data.qvel[self.dadr])) > MOVING_SPEED_RAD_S)

    def _noise(self, t):
        k = int(math.floor(t / LOAD_REFRESH_S + 1e-9))
        if k not in self._load_cache:
            r = np.random.default_rng(np.random.SeedSequence([self.noise_seed & 0xFFFFFFFF, self.noise_seed >> 32, k & 0xFFFFFFFF, 7]))
            self._load_cache = {k: r.normal(0.0, LOAD_NOISE_PERMILLE, len(self.joints))}
        return self._load_cache[k]

    def loads(self, data, t):
        """Per mille of max torque, positive along +bus, integers (0.1 % steps)."""
        tau = data.actuator_force[self.act]
        raw = 1000.0 * self.sign * tau / STALL_NM * self.lscale + self._noise(t)
        return np.clip(np.rint(raw), -1000, 1000).astype(int)
