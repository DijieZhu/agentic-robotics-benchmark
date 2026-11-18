"""On-board gait firmware of the quadruped (MuJoCo, numpy only).

A learned locomotion policy runs at 50 Hz and sets joint targets; a joint PD loop runs every physics step (1 ms)
like the robot's own joint loop. Around it: mode presets, a state machine with the protective behaviours, a posture
controller for standing poses the policy does not cover (low / pitched stance, sit).

Interface (frozen):
    ctl = RLQuadrupedController(model, data, prefix="", seed=0)
    ctl.reset(pos=(x, y), yaw=0.0)            # places the robot standing (simulation set-up), resets the estimate
    ctl.set_command(vx, vy, yaw_rate, body_height=None, mode=None)
    ctl.set_head(pan, tilt)
    ctl.set_body_pitch(pitch)                 # rad, + = nose down, while standing
    ctl.step()                                # one 1 ms control tick; the policy is called every 20 ticks
    ctl.get_state()                           # what the robot knows (no ground truth); 'status' reports
                                              #   'ok', 'blocked', 'protective_sit', 'fallen'
    ctl.truth()                               # ground truth, for logging only
Modes: stand, walk, high_step, crouch_walk, sit.

Protective behaviours: `blocked` (stop, back off about 8 cm, report; reasons in get_state()["blocked"]["reason"]:
'step' a foot came down on something higher than the mode's rating, 'strikes' / 'strike_pitch' a swinging foot met
an obstacle, 'stall' no progress by the leg odometry (or no rotation by the IMU when turning), 'slip' no progress by
the inertial velocity while the legs step; forward, sideways and turning commands alike); cleared by a mode change or
a new walking command. `protective_sit` (roll or pitch beyond 35 degrees). `fallen`.

Signals used: IMU orientation / gyro / accelerometer, joint encoders, foot switches (all through RobotIO.read(),
with the sensor-noise model) and the leg-odometry estimate built from them. Ground-truth base pose or velocity is
never read. Deterministic for a given seed.
"""

from __future__ import annotations

import collections
import json
import math
import os
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from rl_support.estimator import Rz, StateEstimator, G
from rl_support.kinematics import LegKinematics, LEGS
from rl_support.plant import RobotIO, SensorNoise

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_POLICY = os.environ.get("QUAD_RL_POLICY", os.path.join(HERE, "policy", "quadruped_policy.npz"))
FOOT_RADIUS = 0.017
L_LINK = 0.16
_G_W = np.array([0.0, 0.0, -G])


def nominal_pose(h):
    """Joint targets (12,) of a stand with the feet under the hip-flexion axes at base height h (knees back)."""
    c = min(max((h - FOOT_RADIUS) / (2 * L_LINK), 0.2), 0.999)
    q2 = math.acos(c)
    return np.tile(np.array([0.0, q2, -2.0 * q2]), 4)


@dataclass(frozen=True)
class Mode:
    name: str
    walking: bool
    height: float = 0.24              # default body height (base origin above the ground, m)
    height_range: tuple = (0.20, 0.26)
    clearance: float = 0.06           # swing-foot clearance command (m)
    period: float = 0.34              # gait period command (s)
    v_max: tuple = (0.45, 0.20, 1.0)  # |vx|, |vy| (m/s), |yaw rate| (rad/s)
    accel: tuple = (0.8, 0.6, 2.5)    # command rate limits (m/s^2, m/s^2, rad/s^2)
    rating: float = 0.03              # obstacle height this mode is rated for (m): stumble threshold reference


MODES = {
    "stand": Mode("stand", False, 0.24, (0.15, 0.26), v_max=(0.0, 0.0, 0.0)),
    # walk's body height starts at its default: lower, the robot squeezes under round pipes of 0.29-0.30 m that the
    # needs_crouch class leaves to crouch_walk (build stage: 25/40 and 40/40 passed at 0.22 m, 0/40 at 0.24 m)
    "walk": Mode("walk", True, 0.24, (0.24, 0.26), 0.06, 0.34, (0.45, 0.20, 1.0), (0.8, 0.6, 2.5), 0.03),
    "high_step": Mode("high_step", True, 0.255, (0.24, 0.27), 0.13, 0.46, (0.25, 0.10, 0.6), (0.5, 0.4, 1.5), 0.08),
    # crouch_walk's body height ends at 0.19 m: higher, it no longer passes the needs_crouch class's lowest
    # clearance (build stage, 0.283 m chase: 40/40 crossed at 0.19, 24/40 at 0.195, 0/40 at 0.20 and 0.21)
    "crouch_walk": Mode("crouch_walk", True, 0.18, (0.17, 0.19), 0.065, 0.38, (0.20, 0.10, 0.6), (0.5, 0.4, 1.5),
                        0.03),
    "sit": Mode("sit", False, 0.075, (0.06, 0.10), v_max=(0.0, 0.0, 0.0)),
}


class NumpyPolicy:
    """The exported actor: (x - mean) / std -> Dense+swish ... Dense -> tanh."""

    def __init__(self, path: str = DEFAULT_POLICY):
        z = np.load(path, allow_pickle=False)
        self.meta = json.loads(str(z["meta"]))
        n = self.meta["n_layers"]
        self.mean = z["obs_mean"].astype(np.float64)
        self.inv_std = 1.0 / z["obs_std"].astype(np.float64)
        self.W = [z[f"w{i}"].astype(np.float64) for i in range(n)]
        self.b = [z[f"b{i}"].astype(np.float64) for i in range(n)]
        self.path = path

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = (obs - self.mean) * self.inv_std
        last = len(self.W) - 1
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = h @ W + b
            if i < last:
                h = h / (1.0 + np.exp(-h))
        return np.tanh(h)


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


class RLQuadrupedController:
    PLANNER_DIV = 2                    # estimator every 2 ms

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "", seed: int = 0,
                 noise: SensorNoise | None = None, policy: NumpyPolicy | str | None = None,
                 modes: dict | None = None, stumble_stop: bool = True, protective_sit: bool = True, **_ignored):
        self.m, self.d = model, data
        self.dt = float(model.opt.timestep)
        self.io = RobotIO(model, data, prefix, seed=seed, noise=noise)
        self.kin = LegKinematics(model, prefix)
        self.est = StateEstimator(self.kin, self.dt * self.PLANNER_DIV, FOOT_RADIUS, robust_plane=False)
        self.policy = policy if isinstance(policy, NumpyPolicy) else NumpyPolicy(policy or DEFAULT_POLICY)
        meta = self.policy.meta
        self.kp = float(meta["kp"])
        self.kd = float(meta["kd"])
        self.action_scale = np.tile(np.array(meta["action_scale"], float), 4)
        self.hist_len = int(meta["history_len"])
        self.frame_dim = int(meta["frame_dim"])
        self.duty = float(meta["duty"])
        self.offsets = np.array(meta["leg_offsets"], float)
        self.ctrl_div = int(round(float(meta["ctrl_dt"]) / self.dt))
        self.modes = dict(MODES if modes is None or not isinstance(next(iter(modes.values())), Mode) else modes)
        self.stumble_stop_enabled = stumble_stop
        self.protective_enabled = protective_sit
        self.prefix = prefix
        self.q_stand = nominal_pose(0.24)
        self.knee_sign = -np.ones(4)
        self._mass_props()
        try:
            self.r_imu = np.array(model.site_pos[model.site(prefix + "imu").id], float)   # IMU in the base frame
        except KeyError:
            self.r_imu = np.zeros(3)
        self.reset_internal()

    # ------------------------------------------------------------------------------------------ model constants
    def _mass_props(self):
        m, P = self.m, self.prefix
        d = mujoco.MjData(m)
        qa = self.io.root_qadr
        d.qpos[qa:qa + 7] = [0, 0, 0, 1, 0, 0, 0]
        d.qpos[self.io.qadr] = self.q_stand
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        leg_bodies = {m.body(P + f"{l}_{k}").id for l in LEGS for k in ("hip", "thigh", "shank")}
        msum, mt = np.zeros(3), 0.0
        for b in self.io.robot_bodies:
            if b in leg_bodies:
                continue
            msum += m.body_mass[b] * d.xipos[b]
            mt += m.body_mass[b]
        self.trunk_mass, self.trunk_com = mt, msum / mt
        self.mass = self.io.mass

    # ------------------------------------------------------------------------------------------ public API
    def reset(self, pos=(0.0, 0.0), yaw: float = 0.0, z: float | None = None, mode: str = "stand",
              phase: float = 0.0, settle: float = 0.3, ground_z: float = 0.0, roll: float = 0.0):
        """Place the robot standing at (x, y, yaw) on ground of height ground_z (simulation set-up only) and reset
        the controller and the onboard estimate (the start pose is known). roll: start on a side slope."""
        m, d = self.m, self.d
        qa = self.io.root_qadr
        h = 0.24 if z is None else z
        h_w = h / math.cos(roll)
        d.qpos[qa:qa + 3] = [pos[0], pos[1], ground_z + h_w + 0.002]
        cz, sz = math.cos(yaw / 2), math.sin(yaw / 2)
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        d.qpos[qa + 3:qa + 7] = [cz * cr, cz * sr, sz * sr, sz * cr]
        d.qpos[self.io.qadr] = nominal_pose(h)
        d.qvel[self.io.root_dadr:self.io.root_dadr + 6] = 0
        d.qvel[self.io.dadr] = 0
        d.ctrl[:] = 0
        if self.io.head_q is not None:
            d.qpos[list(self.io.head_q)] = 0
        mujoco.mj_forward(m, d)
        self.io.reset_sensors()
        self.reset_internal(phase)
        meas = self.io.read()
        self.meas = meas
        self.est.reset(np.array([pos[0], pos[1], ground_z + h_w + 0.002]), yaw, meas)
        self.q_target = nominal_pose(h)
        self.mode = self.modes["stand"]
        self.set_command(0.0, 0.0, 0.0, mode=mode)
        for _ in range(int(round(settle / self.dt))):
            self.step()
        self.io.energy = dict(mech_pos=0.0, mech_neg=0.0, joule=0.0, electronics=0.0, time=0.0)

    def reset_internal(self, phase: float = 0.0):
        self.t = 0.0
        self._k = 0
        self.phase = float(phase) % 1.0
        self.cmd_user = np.zeros(3)
        self.v_cmd = np.zeros(3)
        self.h_user = None
        self.pitch_user = 0.0
        self.mode = self.modes["stand"]
        self.state = "stand"              # stand | walk | stopping | posture | sit_lower | seated | prepare | rise
        #                                   | backoff | protective
        self.state_t = 0.0
        self.status = "ok"
        self.last_act = np.zeros(12)
        self.hist = None
        self.q_target = nominal_pose(0.24)
        self.h_cmd = 0.24
        self.c_cmd = 0.06
        self.T_cmd = 0.34
        self.stand_flag = 1.0
        self.head_cmd = (0.0, 0.0)
        self.fallen = False
        self.still_t = 0.0
        self.strike_times = []
        self.strike_count = 0
        self.hit_prev = np.zeros(4, bool)
        self.air_t = np.zeros(4)
        self.stall_t = 0.0
        self.prog_hist = []
        self.prog_last_t = {0: -math.inf, 1: -math.inf, -1: -math.inf}
        self.prog_s = 0.0
        self.prog_last = None
        self.debug_log = None
        self.prev_contact = np.zeros(4, bool)
        self.touchdown = np.zeros(4, bool)
        self.post = None                  # posture-controller targets
        self.protect_t = None
        self.blocked_info = None
        self.events = []
        self.meas = None
        # foothold memory of the step guard (heights in the odometry frame minus the drift seen on feet in stance)
        self.fh_sup = None                # (4,) height of each foot's last foothold that held STEP_HOLD
        self.fh_pend = [None, None, None, None]
        self.fh_npend = 0
        self.fh_drift = 0.0
        self.fh_zprev = None
        self.fh_st = np.zeros(4)
        self.fh_prevc = np.zeros(4, bool)
        self.step_hit = None              # (t, foot, rise) of the last foothold above the mode's step limit
        # inertial velocity (world frame, at the IMU) for the slip-aware stall check
        self.v_in = None
        self.acc_sum = np.zeros(3)
        self.acc_n = 0
        self.prog_i = 0.0
        # sideways and turning stall checks: progress to the left by the leg odometry and by the inertial velocity
        # (each tick's displacement on that tick's left axis), unwrapped heading; cumulative, differenced over the
        # stall window; the last time of each command kind (as prog_last_t)
        self.lat_s = 0.0
        self.lat_i = 0.0
        self.yaw_u = 0.0
        self.yaw_u_last = None
        self.c_lat = 0.0                  # integral of the sideways command (m)
        self.c_yaw = 0.0                  # integral of the yaw-rate command (rad)
        self.lat_last_t = {0: -math.inf, 1: -math.inf, -1: -math.inf}
        self.turn_last_t = {0: -math.inf, 1: -math.inf, -1: -math.inf}
        # stall checks after a change of command (the forward stall window): the integrals of the command the
        # gait was given (v_cmd, rate limited at the mode's acceleration: forward, sideways, yaw rate), the last
        # RESPONSE_DELAY of them (given_q), and for each axis whether the user's command changed since the walk state
        # began (cmd_seen: its value at the last check)
        self.c_given = np.zeros(3)
        self.given_q = collections.deque()
        self.cmd_seen = None
        self.cmd_changed = np.zeros(3, bool)

    def set_command(self, vx: float, vy: float, yaw_rate: float, body_height: float | None = None,
                    mode: str | None = None):
        """Velocity command in the body heading frame (m/s, m/s, rad/s). mode None keeps the current mode, except
        that a moving command in 'stand' (or 'sit') switches to 'walk'. body_height None = the mode's default."""
        if mode is not None and mode not in self.modes:
            raise ValueError(f"unknown mode {mode!r}; modes: {sorted(self.modes)}")
        cmd = np.array([vx, vy, yaw_rate], float)
        moving = bool(np.any(np.abs(cmd) > 1e-4))
        if mode is None:
            mode = self.mode.name
            if mode in ("stand", "sit") and moving:
                mode = "walk"
        new = self.modes[mode]
        if new.name != self.mode.name and self.status == "blocked":
            self.status = "ok"                    # a mode change clears the 'blocked' report
        self.mode = new
        self.cmd_user = np.clip(cmd, -np.array(new.v_max), np.array(new.v_max)) if new.walking else np.zeros(3)
        self.h_user = body_height

    def set_head(self, pan: float, tilt: float):
        self.head_cmd = (float(pan), float(tilt))

    def set_body_pitch(self, pitch: float):
        """Body pitch (rad, + = nose down) while standing, e.g. to aim the head; +-20 degrees."""
        self.pitch_user = float(np.clip(pitch, -math.radians(20.0), math.radians(20.0)))

    # ------------------------------------------------------------------------------------------ the 1 kHz tick
    def step(self):
        meas = self.io.read()
        self.meas = meas
        self.acc_sum += meas["acc"]
        self.acc_n += 1
        if self._k % self.PLANNER_DIV == 0:
            contact = meas["contact"]
            self.touchdown |= contact & ~self.prev_contact
            self.prev_contact = contact.copy()
            # leg odometry: a foot whose switch is on is taken as fixed on the ground
            self.est.update(meas, contact, self.touchdown)
            self.touchdown[:] = False
            self._footholds(contact)
            self._inertial()
        if self._k % self.ctrl_div == 0:
            self._control_tick(meas)
        self._k += 1
        q, dq = meas["q"], meas["dq"]
        if self.post is not None:
            tau = self.post["ff"] + self.post["kp"] * (self.post["q"] - q) - self.post["kd"] * dq
        else:
            tau = self.kp * (self.q_target - q) - self.kd * dq
        self.io.apply(tau)
        self.io.set_head(*self.head_cmd)
        mujoco.mj_step(self.m, self.d)
        self.io.account_energy(self.dt)
        self.t += self.dt

    # ------------------------------------------------------------------------------------------ 50 Hz logic
    def _control_tick(self, meas):
        dt = self.ctrl_div * self.dt
        self.state_t += dt
        roll, pitch, _ = self.est.rpy
        tilt = max(abs(roll), abs(pitch))
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            self.fallen = True
            self.status = "fallen"
        # protective sit: roll or pitch beyond 35 degrees
        if (self.protective_enabled and tilt > self.PROTECT_TILT and self.state not in
                ("protective", "seated", "sit_lower") and not self.fallen):
            self._begin_posture()              # capture the feet where they are, then lower the body
            q_now = self.est.q.ravel().copy()
            q_low = nominal_pose(0.10)
            q_low[0::3] = q_now[0::3]          # keep the ab/ad angles: crouch along the body's own z axis
            self.post.update(q_from=q_now, q_low=q_low, T=0.8)
            self._enter("protective")
            self.status = "protective_sit"
            self.protect_t = 0.0
            self.events.append((self.t, "protective_sit"))
        mode = self.mode
        moving = bool(np.any(np.abs(self.cmd_user) > 1e-4))
        st = self.state
        # ---------------------------------------------------------------- state transitions
        if st == "protective":
            self.protect_t += dt
            if self.protect_t >= 5.0 and tilt < math.radians(25.0):
                if moving or mode.name in ("stand",):
                    self._begin_rise()
        elif st == "backoff":
            if self.state_t >= 0.8:
                self._enter("stopping")
        elif st in ("sit_lower", "seated"):
            if mode.name != "sit" and (moving or mode.name == "stand"):
                self._begin_prepare()
        elif st == "prepare":
            if self.state_t >= self.post["T"] + 0.1:
                self._begin_rise()
        elif st == "rise":
            if self.state_t >= self.post["T"] + 0.2:
                self._handover_to_policy()
        elif st == "posture":
            want_posture = (mode.name == "stand" and not moving and self._needs_posture())
            if mode.name == "sit":
                self._begin_sit()
            elif not want_posture and self._posture_at_neutral():
                self._handover_to_policy()
        elif st in ("stand", "walk", "stopping"):
            if mode.name == "sit":
                if st == "stand" and self.state_t > 0.3:
                    self._begin_sit()
                else:
                    self._enter("stopping")
            elif moving and mode.walking:
                if st != "walk":
                    self._enter("walk")
                    if self.status == "blocked" and self.CLEAR_BLOCKED_ON_WALK:
                        self.status = "ok"         # a new walking command after the stop clears it
            elif st == "walk":
                self._enter("stopping")
            elif st == "stopping":
                v = np.linalg.norm(self.est.v[:2])
                self.still_t = self.still_t + dt if (v < 0.06 and np.all(np.abs(self.v_cmd) < 0.02)) else 0.0
                if self.still_t > 0.25:
                    self._enter("stand")
            elif st == "stand" and self.state_t > 0.4 and mode.name == "stand" and self._needs_posture():
                self._begin_posture()
        # ---------------------------------------------------------------- outputs
        st = self.state
        if st in ("walk", "stopping", "stand", "backoff"):
            self.post = None
            self._policy_tick(meas, dt)
        else:
            self._posture_tick(meas, dt)

    def _enter(self, st):
        self.state = st
        self.state_t = 0.0
        self.still_t = 0.0
        self.prog_hist = []
        self.given_q = collections.deque()
        self.cmd_seen = None
        self.cmd_changed = np.zeros(3, bool)

    # ------------------------------------------------------------------------------------------ policy
    def _needs_posture(self):
        h = self.h_user
        return abs(self.pitch_user) > math.radians(1.0) or (h is not None and abs(h - 0.24) > 0.004)

    def _policy_tick(self, meas, dt):
        mode = self.mode if self.mode.walking else self.modes["walk"]
        st = self.state
        # velocity command (rate limited); stopping and standing ramp to zero
        if st == "walk":
            target = self.cmd_user.copy()
        elif st == "backoff":
            target = np.array([-0.10, 0.0, 0.0])
        else:
            target = np.zeros(3)
        # speeding up at the mode's rate, slowing down fast (the policy was trained with step changes)
        up = np.array(mode.accel) * dt
        down = np.array([3.0, 3.0, 6.0]) * dt
        delta = target - self.v_cmd
        slowing = np.abs(target) < np.abs(self.v_cmd) - 1e-9
        lim = np.where(slowing, down, up)
        self.v_cmd += np.clip(delta, -lim, lim)
        # gait parameters (smoothed), stand flag
        lo, hi = mode.height_range
        if self.mode.name == "stand":
            lo, hi = 0.20, 0.26
            h_goal = 0.24 if self.h_user is None else float(np.clip(self.h_user, lo, hi))
        else:
            h_goal = mode.height if self.h_user is None else float(np.clip(self.h_user, lo, hi))
        # pre-emptive crouch when the body tilts a lot (the static tipping angle of the stand is only ~34 deg)
        roll, pitch, _ = self.est.rpy
        if max(abs(roll), abs(pitch)) > self.BRACE_TILT:
            h_goal = min(h_goal, 0.20)
        self.h_cmd += float(np.clip(h_goal - self.h_cmd, -0.15 * dt, 0.15 * dt))
        self.c_cmd += float(np.clip(mode.clearance - self.c_cmd, -0.2 * dt, 0.2 * dt))
        self.T_cmd += float(np.clip(mode.period - self.T_cmd, -0.5 * dt, 0.5 * dt))
        stepping = st in ("walk", "backoff") or (st == "stopping" and (self.state_t < 0.4 or
                                                                        np.any(np.abs(self.v_cmd) > 0.02)))
        self.stand_flag = 0.0 if stepping else 1.0
        if not stepping:
            pass                                   # the clock is frozen while standing
        frame = self._frame(meas)
        if self.hist is None:
            self.hist = np.tile(frame, (self.hist_len, 1))
        else:
            self.hist = np.vstack([self.hist[1:], frame[None, :]])
        act = self.policy(self.hist.reshape(-1))
        self.last_act = act
        self.q_target = nominal_pose(self.h_cmd) + act * self.action_scale
        # the clock advances after the action (as in training: the phase of the next observation)
        if self.stand_flag < 0.5:
            self.phase = (self.phase + dt / self.T_cmd) % 1.0
        if st == "walk" and self.stumble_stop_enabled:
            self._stumble_check(meas, dt)

    def _frame(self, meas):
        R = _quat_to_R(meas["quat"])
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        cmd = np.array([self.v_cmd[0], self.v_cmd[1], self.v_cmd[2], (self.h_cmd - 0.24) * 10.0,
                        (self.c_cmd - 0.07) * 10.0, (self.T_cmd - 0.4) * 10.0, self.stand_flag])
        ph = 2 * math.pi * self.phase
        clock = np.array([math.sin(ph), math.cos(ph)]) * (1.0 - self.stand_flag)
        return np.concatenate([meas["gyro"] * 0.25, grav, meas["q"] - self.q_stand, meas["dq"] * 0.05,
                               self.last_act, cmd, clock])

    def swing_progress(self):
        psi = (self.phase + self.offsets) % 1.0
        return np.where(psi >= self.duty, (psi - self.duty) / (1.0 - self.duty), -1.0)

    # ------------------------------------------------------------------------------------------ stumble stop
    BRACE_TILT = math.radians(22.0)   # above this roll / pitch the policy is told to crouch to 0.20 m
    PROTECT_TILT = math.radians(35.0)  # protective sit

    # stumble-stop parameters
    STRIKE_WINDOW = (0.12, 0.62)      # swing progress in which a re-contact counts as a strike
    STRIKE_AIR_MIN = 0.04             # s airborne before the re-contact
    STRIKE_HEIGHT = 0.012             # m above the stance plane (a touchdown on the floor is no strike)
    STRIKES_TO_BLOCK = 3              # within STRIKE_MEMORY
    STRIKE_MEMORY = 1.5               # s
    STALL_RATIO = 0.25                # progress over the last STALL_TIME below this fraction of the commanded
    STALL_TIME = 1.5                  # progress (odometry, along the commanded direction) = stalled
    PITCH_UP = math.radians(10.0)     # a strike with the body pitched up this much = climbing into it
    # high_step is expected to labour over obstacles near its rating (the body pitches up on a kerb, progress
    # slows): it gets a longer, looser stall window, more strikes, and no pitch-up rule
    # (a blind robot in high_step often sets a foot on top of a pipe or kerb edge: strikes are not evidence here)
    HIGH_STEP_STUMBLE = dict(STALL_RATIO=0.15, STALL_TIME=2.5, STRIKES_TO_BLOCK=10 ** 6, PITCH_UP=math.radians(90.0))

    # step guard: a foot that comes down more than STEP_RISE[mode] above its own last foothold has met an obstacle
    # above the mode's rating (heights from the leg kinematics and the IMU, corrected for the vertical drift of the
    # odometry seen on the feet in stance). The rise of a foot between two footholds is bounded on a slope by
    # stride x sin(slope), so the guard tells a step from a ramp by its size only.
    STEP_RISE = {"walk": 0.030, "crouch_walk": 0.030, "high_step": None}
    STEP_HOLD = 0.06                  # s a foothold must hold before it becomes the foot's reference
    # slip-aware stall check: progress measured with the inertial velocity (accelerometer integrated in the world
    # frame, pulled towards the leg odometry with INERTIAL_TAU while moving and ZUPT_TAU while standing still), so
    # that treading in place with slipping feet shows as no progress although the leg odometry advances
    INERTIAL_TAU = 20.0
    ZUPT_TAU = 0.05
    SLIP_CHECK = True
    SLIP_FACTOR = 1.0                 # slip = inertial progress below SLIP_FACTOR x the stall threshold
    # sideways and turning stall checks, for the stall windows in which the forward check above cannot run (the
    # forward command was not held above 0.05 m/s with one sign): the same test as the forward check, on the
    # sideways command (held above PLANAR_MIN_SPEED with one sign: progress to the left by the leg odometry ->
    # 'stall', by the inertial velocity -> 'slip') and on the yaw-rate command (held above PLANAR_MIN_YAW_RATE with
    # one sign: rotation by the IMU -> 'stall'); the same window and fraction. A window with a held forward command
    # is the forward check's only, so a walk with a steady forward command behaves exactly as before.
    PLANAR_CHECK = True
    PLANAR_MIN_SPEED = 0.05           # m/s
    PLANAR_MIN_YAW_RATE = 0.10        # rad/s
    # after a change of command within one walk (the forward stall window): the progress required over the
    # window is STALL_RATIO x the command the gait was actually given (the rate-limited command v_cmd, which speeds
    # up at the mode's acceleration and slows down fast), delayed by the gait's response time RESPONSE_DELAY and
    # integrated over the window, and a check runs only when that commanded amount averages at least its
    # held-command threshold over the window (FWD_MIN_SPEED, PLANAR_MIN_SPEED, PLANAR_MIN_YAW_RATE). Per axis
    # (forward, sideways, turning): until that axis's command changes during the walk, the requirement is computed
    # as before (forward: the present command x the window; sideways / turning: the command integrated), the start
    # of a walk being covered by the arming delay STALL_TIME + 0.5 s; so a walk with one command behaves bit for bit
    # as before. RESPONSE_DELAY: the robot's velocity follows v_cmd with a lag of 0.02-0.14 s on level floor (largest
    # in high_step), and ran on about 0.08 s in the old direction after a reversal under a full turning command.
    FWD_MIN_SPEED = 0.05              # m/s
    RESPONSE_DELAY = 0.14             # s (a multiple of the 20 ms control tick)
    CLEAR_BLOCKED_ON_WALK = True

    CROUCH_STUMBLE = {}               # crouch_walk overrides (none: walk's parameters)

    def _stumble_param(self, name):
        if self.mode.name == "high_step" and name in self.HIGH_STEP_STUMBLE:
            return self.HIGH_STEP_STUMBLE[name]
        if self.mode.name == "crouch_walk" and name in self.CROUCH_STUMBLE:
            return self.CROUCH_STUMBLE[name]
        return getattr(self, name)

    def _footholds(self, contact):
        """Foothold memory of the step guard (estimator rate)."""
        fz = self.est.foot_w[:, 2]
        prevc = self.fh_prevc
        if self.fh_zprev is not None:
            held = contact & prevc & (self.fh_st >= 0.03)
            if held.any():
                self.fh_drift += float(np.mean(fz[held] - self.fh_zprev[held]))
        dt = self.dt * self.PLANNER_DIV
        self.fh_st = np.where(contact, self.fh_st + dt, 0.0)
        self.fh_zprev = fz.copy()
        zc = fz - self.fh_drift
        if self.fh_sup is None:
            self.fh_sup = zc.copy()
        lim = self.STEP_RISE.get(self.mode.name) if self.state == "walk" else None
        t = self.t
        new_c = contact & ~prevc
        if new_c.any() or self.fh_npend:
            for i in range(4):
                if new_c[i]:
                    # a new contact: its height (at touchdown) against the foot's last held foothold
                    if self.fh_pend[i] is None:
                        self.fh_npend += 1
                    self.fh_pend[i] = (t, float(zc[i]), float(zc[i] - self.fh_sup[i]))
                    continue
                p = self.fh_pend[i]
                if p is None:
                    continue
                if not contact[i]:
                    self.fh_pend[i] = None
                    self.fh_npend -= 1
                elif t - p[0] >= self.STEP_HOLD - 1e-9:
                    # the contact held: it is a foothold (a brush against a wall or a jamb does not count)
                    if lim is not None and p[2] > lim:
                        self.step_hit = (t, i, p[2])
                    self.fh_sup[i] = p[1]
                    self.fh_pend[i] = None
                    self.fh_npend -= 1
        self.fh_prevc = contact.copy()

    def _inertial(self):
        """Inertial velocity at the IMU (estimator rate): the mean accelerometer reading since the last update,
        rotated into the world frame, gravity removed, integrated; pulled slowly towards the leg odometry."""
        n = self.acc_n
        if n == 0:
            return
        est = self.est
        R = est.R
        dt = n * self.dt
        a_w = R @ (self.acc_sum / n) + _G_W
        self.acc_sum = np.zeros(3)
        self.acc_n = 0
        w, r = est.omega_b, self.r_imu
        lever = R @ np.array([w[1] * r[2] - w[2] * r[1], w[2] * r[0] - w[0] * r[2], w[0] * r[1] - w[1] * r[0]])
        v_leg = est.v + lever
        if self.v_in is None:
            self.v_in = v_leg.copy()
        else:
            v = self.v_in + a_w * dt
            still = (self.state in ("stand", "posture", "seated") and bool(np.all(est.contact)) and
                     abs(float(est.v[0])) < 0.03 and abs(float(est.v[1])) < 0.03)
            k = min(1.0, dt / (self.ZUPT_TAU if still else self.INERTIAL_TAU))
            self.v_in = v + k * (v_leg - v)
        self.v_in_base = self.v_in - lever

    def _stumble_check(self, meas, dt):
        """Stumble stop: if the obstacle ahead is above the mode's clearance, back off, stop and report 'blocked'.
        A strike = a foot switch that comes on while the gait clock still has that foot early / mid swing, after
        it had been airborne, with the foot above the stance plane (it met an obstacle's face or top on its way
        forward). Blocked = STRIKES_TO_BLOCK strikes within STRIKE_MEMORY, or a strike with the body pitched up by
        PITCH_UP, or a stall (forward progress over STALL_TIME below STALL_RATIO of the commanded displacement while
        stepping: the present command x the window while the forward command has not changed during this walk, the
        rate-limited command delayed by RESPONSE_DELAY and integrated over the window once it has (FWD_MIN_SPEED); for
        a window without a held forward command, the planar check on the commanded displacement and rotation)."""
        stall_time = self._stumble_param("STALL_TIME")
        stall_ratio = self._stumble_param("STALL_RATIO")
        strikes_to_block = self._stumble_param("STRIKES_TO_BLOCK")
        pitch_up_lim = self._stumble_param("PITCH_UP")
        s = self.swing_progress()
        contact = meas["contact"]
        in_swing = s >= 0.0
        self.air_t = np.where(in_swing & ~contact, self.air_t + dt, np.where(in_swing, self.air_t, 0.0))
        lo, hi = self.STRIKE_WINDOW
        touch = contact & ~self.hit_prev
        self.hit_prev = contact.copy()
        fw = self.est.foot_w
        # the estimator's plane runs through the foot-sphere centres at touchdown: this is the height above it
        h_above = fw[:, 2] - np.array([self.est.plane_z(x, y) for x, y in fw[:, :2]])
        new_hit = touch & (s > lo) & (s < hi) & (self.air_t >= self.STRIKE_AIR_MIN) & (h_above > self.STRIKE_HEIGHT)
        t = self.t
        warm = self.state_t > 1.0
        if warm:
            for i in np.nonzero(new_hit)[0]:
                self.strike_times.append(t)
                self.strike_count += 1
        self.strike_times = [x for x in self.strike_times if t - x < self.STRIKE_MEMORY]
        v_cmd = self.cmd_user[0]
        v_est = float((Rz(self.est.rpy[2]).T @ self.est.v)[0])
        # stall: odometry progress along the (turning) heading over a sliding window vs the commanded speed;
        # progress accumulates each tick's displacement projected on that tick's heading
        yaw = self.est.rpy[2]
        cy, sy = math.cos(yaw), math.sin(yaw)
        p_xy = self.est.p[:2].copy()
        if self.prog_last is not None:
            dp = p_xy - self.prog_last
            self.prog_s += float(dp[0] * cy + dp[1] * sy)
        self.prog_last = p_xy
        vb = getattr(self, "v_in_base", None)
        if vb is not None:
            self.prog_i += float(vb[0] * cy + vb[1] * sy) * dt
        # sideways / turning checks' sums (they do not enter the forward check)
        vy_cmd, wz_cmd = float(self.cmd_user[1]), float(self.cmd_user[2])
        if self.prog_hist:
            dp0 = p_xy - self.prog_hist[-1][9]
            self.lat_s += float(-dp0[0] * sy + dp0[1] * cy)
        if vb is not None:
            self.lat_i += float(-vb[0] * sy + vb[1] * cy) * dt
        if self.yaw_u_last is not None:
            self.yaw_u += (yaw - self.yaw_u_last + math.pi) % (2.0 * math.pi) - math.pi
        self.yaw_u_last = yaw
        self.c_lat += vy_cmd * dt
        self.c_yaw += wz_cmd * dt
        # the command given to the gait: entry k holds the integral of v_cmd over the ticks before it as it was
        # RESPONSE_DELAY earlier (the oldest kept in this walk state if it is younger), so that two entries differ by
        # the command that drove the motion between them, delayed by the gait's response time
        if self.cmd_seen is not None:
            self.cmd_changed |= self.cmd_user != self.cmd_seen
        self.cmd_seen = self.cmd_user.copy()
        self.given_q.append(self.c_given.copy())
        if len(self.given_q) > int(round(self.RESPONSE_DELAY / dt)) + 1:
            self.given_q.popleft()
        self.prog_hist.append((t, self.prog_s, v_cmd, self.prog_i, self.lat_s, self.lat_i, self.yaw_u, self.c_lat,
                               self.c_yaw, p_xy, self.given_q[0]))
        self.c_given += self.v_cmd * dt
        while len(self.prog_hist) > 1 and t - self.prog_hist[1][0] >= stall_time - 1e-9:
            self.prog_hist.pop(0)
        h0 = self.prog_hist[0]
        t0, x0, i0 = h0[0], h0[1], h0[3]
        # steady: every entry in the window had |command| > 0.05 with the sign of the present one (kept in O(1):
        # the time of the last entry of each kind; an entry is in the window when its time is >= t0)
        key = (1 if v_cmd > 0 else -1) if abs(v_cmd) > self.FWD_MIN_SPEED else 0
        last = self.prog_last_t
        last[key] = t
        steady = key != 0 and max(last[0], last[-key]) < t0
        armed = warm and steady and t - t0 >= stall_time - 1e-9 and self.state_t > stall_time + 0.5
        if self.cmd_changed[0]:
            # the forward command changed during this walk: require a fraction of the displacement the gait was
            # commanded over the window (rate-limited command, delayed, integrated), not of the present command x
            # the window
            e_fwd = float(self.prog_hist[-1][10][0] - h0[10][0]) * np.sign(v_cmd)
            armed = armed and e_fwd >= self.FWD_MIN_SPEED * (t - t0)
            need = stall_ratio * e_fwd
        else:
            need = stall_ratio * abs(v_cmd) * (t - t0)
        if armed and (self.prog_hist[-1][1] - x0) * np.sign(v_cmd) < need:
            self.stall_t = t - t0
        else:
            self.stall_t = 0.0
        slip = bool(self.SLIP_CHECK and armed and vb is not None and
                    (self.prog_hist[-1][3] - i0) * np.sign(v_cmd) < self.SLIP_FACTOR * need)
        planar = None
        if self.PLANAR_CHECK:
            ky = (1 if vy_cmd > 0 else -1) if abs(vy_cmd) > self.PLANAR_MIN_SPEED else 0
            kw = (1 if wz_cmd > 0 else -1) if abs(wz_cmd) > self.PLANAR_MIN_YAW_RATE else 0
            self.lat_last_t[ky] = t
            self.turn_last_t[kw] = t
            if warm and not steady and t - t0 >= stall_time - 1e-9 and self.state_t > stall_time + 0.5:
                planar = self._planar_check(h0, self.prog_hist[-1], ky, kw, stall_ratio, vb is not None)
        pitch_up = self.est.rpy[1] < -pitch_up_lim
        reason = None
        hit = self.step_hit
        self.step_hit = None
        if hit is not None and t - hit[0] < 0.05:
            reason = "step"
        elif warm and len(self.strike_times) >= strikes_to_block:
            reason = "strikes"
        elif warm and np.any(new_hit) and pitch_up:
            reason = "strike_pitch"
        elif self.stall_t > 0.0:
            reason = "stall"
        elif slip:
            reason = "slip"
        elif planar is not None:
            reason = planar
        if self.debug_log is not None:
            self.debug_log.append(dict(t=t, s=s.copy(), contact=contact.copy(), hit=new_hit.copy(), h=h_above.copy(),
                                       air=self.air_t.copy(), pitch=float(self.est.rpy[1]), v_est=v_est,
                                       stall=self.stall_t, state=self.state))
        if reason is not None:
            self.status = "blocked"
            self.blocked_info = dict(t=t, reason=reason, mode=self.mode.name)
            self.events.append((t, "blocked"))
            self._enter("backoff")
            self.cmd_user = np.zeros(3)
            self.strike_times = []
            self.stall_t = 0.0
            self.prog_hist = []

    def _planar_check(self, h0, h1, ky, kw, ratio, have_inertial):
        """Sideways and turning stall tests over one window (prog_hist entries h0 .. h1), for a sideways command
        (ky = its sign) or a yaw-rate command (kw) held with one sign over the whole window: 'stall' when the
        progress to that side by the leg odometry, or the rotation by the IMU, is below `ratio` of the commanded
        amount (the command integrated over the window; once that axis's command has changed during this walk, the
        rate-limited command delayed by RESPONSE_DELAY and integrated, and the test runs only when that averages at
        least PLANAR_MIN_SPEED / PLANAR_MIN_YAW_RATE over the window); 'slip' when the progress by the inertial
        velocity is.
        None = progress (or nothing held)."""
        t0 = h0[0]
        if ky != 0 and max(self.lat_last_t[0], self.lat_last_t[-ky]) < t0:
            if self.cmd_changed[1]:
                # the sideways command changed during this walk: the rate-limited command, delayed, integrated
                e = float(h1[10][1] - h0[10][1]) * ky
                need = ratio * e if e >= self.PLANAR_MIN_SPEED * (h1[0] - t0) else None
            else:
                need = ratio * abs(h1[7] - h0[7])
            if need is not None and (h1[4] - h0[4]) * ky < need:
                return "stall"
            if (need is not None and self.SLIP_CHECK and have_inertial and
                    (h1[5] - h0[5]) * ky < self.SLIP_FACTOR * need):
                return "slip"
        if kw != 0 and max(self.turn_last_t[0], self.turn_last_t[-kw]) < t0:
            if self.cmd_changed[2]:
                e = float(h1[10][2] - h0[10][2]) * kw
                need = ratio * e if e >= self.PLANAR_MIN_YAW_RATE * (h1[0] - t0) else None
            else:
                need = ratio * abs(h1[8] - h0[8])
            if need is not None and (h1[6] - h0[6]) * kw < need:
                return "stall"
        return None

    # ------------------------------------------------------------------------------------------ posture control
    def _stance_forces(self, foot_w, com_w):
        """Vertical ground forces (4,) that hold the robot statically (min-norm, non-negative)."""
        A = np.vstack([np.ones(4), foot_w[:, 1] - com_w[1], foot_w[:, 0] - com_w[0]])
        b = np.array([self.mass * G, 0.0, 0.0])
        f = A.T @ np.linalg.solve(A @ A.T + 1e-9 * np.eye(3), b)
        for _ in range(3):
            if np.all(f >= 0.5):
                break
            f = np.maximum(f, 0.5)
            f *= self.mass * G / f.sum()
        return f

    def _begin_posture(self, target_h=None, target_pitch=None):
        est = self.est
        R0 = est.R
        p0 = est.p.copy()
        feet_w = (R0 @ est.foot_b.T).T + p0
        yaw = est.rpy[2]
        ground = float(np.mean(feet_w[:, 2])) - FOOT_RADIUS
        self.post = dict(feet_w=feet_w, base_xy=p0[:2].copy(), yaw=yaw, ground=ground,
                         h=est.height, pitch=est.rpy[1], q=est.q.ravel().copy(), ff=np.zeros(12), kp=8.0, kd=0.3,
                         T=0.0, h_goal=target_h, pitch_goal=target_pitch)
        self._enter("posture")

    def _posture_targets(self, h, pitch):
        P = self.post
        yaw = P["yaw"]
        cp, sp = math.cos(pitch), math.sin(pitch)
        Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
        R = Rz(yaw) @ Ry
        base = np.array([P["base_xy"][0], P["base_xy"][1], P["ground"] + h])
        q = np.zeros(12)
        feet_b = (R.T @ (P["feet_w"] - base).T).T
        for i in range(4):
            q[3 * i:3 * i + 3] = self.kin.ik_leg(i, feet_b[i], self.knee_sign[i])
        return q, R, base, feet_b

    def _posture_ff(self, q, R, feet_b):
        g_b = R.T @ np.array([0.0, 0.0, -G])
        com_b, tau_g = self.kin.com_and_gravity(q.reshape(4, 3), g_b, self.trunk_mass, self.trunk_com)
        _, J = self.kin.fk(q.reshape(4, 3))
        f_w = self._stance_forces((R @ feet_b.T).T, R @ com_b)
        tau = np.zeros(12)
        for i in range(4):
            f_b = R.T @ np.array([0.0, 0.0, f_w[i]])
            tau[3 * i:3 * i + 3] = -J[i].T @ f_b + tau_g[i]
        return tau

    def _posture_tick(self, meas, dt):
        st = self.state
        P = self.post
        if st == "posture":
            h_goal = 0.24 if self.h_user is None else float(np.clip(self.h_user, 0.15, 0.26))
            p_goal = self.pitch_user
            if self.mode.name != "stand" or np.any(np.abs(self.cmd_user) > 1e-4):
                h_goal, p_goal = 0.24, 0.0              # leaving: back to the neutral pose first
            P["h"] += float(np.clip(h_goal - P["h"], -0.08 * dt, 0.08 * dt))
            P["pitch"] += float(np.clip(p_goal - P["pitch"], -0.35 * dt, 0.35 * dt))
            q, R, base, feet_b = self._posture_targets(P["h"], P["pitch"])
            P["q"], P["ff"], P["kp"], P["kd"] = q, self._posture_ff(q, R, feet_b), 8.0, 0.3
        elif st == "protective":
            # crouch in joint space towards the support (whatever its slope): lowers the CoM along the body z axis
            sp = _smooth(self.state_t / P["T"])
            q = P["q_from"] + (P["q_low"] - P["q_from"]) * sp
            g_b = self.est.R.T @ np.array([0.0, 0.0, -G])
            _, tau_g = self.kin.com_and_gravity(q.reshape(4, 3), g_b, self.trunk_mass, self.trunk_com)
            P["q"], P["ff"], P["kp"], P["kd"] = q, tau_g.ravel(), 8.0, 0.3
            P["h"] = float(self.est.height)
        elif st == "sit_lower":
            P["h"] += float(np.clip(self.modes["sit"].height - P["h"], -0.10 * dt, 0.10 * dt))
            P["pitch"] += float(np.clip(0.0 - P["pitch"], -0.35 * dt, 0.35 * dt))
            q, R, base, feet_b = self._posture_targets(P["h"], P["pitch"])
            low = P["h"] <= self.modes["sit"].height + 1e-3
            if low and st == "sit_lower" and self.state_t > 1.0:
                self._enter("seated")
                P["q_hold"] = self.est.q.ravel().copy()
            # as the body nears the floor, release the support so that it settles on its belly
            w = float(np.clip((P["h"] - 0.075) / 0.06, 0.0, 1.0))
            P["q"], P["ff"], P["kp"], P["kd"] = q, w * self._posture_ff(q, R, feet_b), 3.0 + 5.0 * w, 0.2
        elif st == "seated":
            P["q"], P["ff"], P["kp"], P["kd"] = P.get("q_hold", self.est.q.ravel()), np.zeros(12), 1.0, 0.05
        elif st in ("prepare", "rise"):
            s = _smooth(self.state_t / max(P["T"], 1e-3))
            if st == "prepare":
                q = P["q0"] + (P["q1"] - P["q0"]) * s
                P["q"], P["ff"], P["kp"], P["kd"] = q, np.zeros(12), 8.0, 0.25
            else:
                h = P["h0"] + (0.24 - P["h0"]) * s
                P["h"] = h
                q, R, base, feet_b = self._posture_targets(h, 0.0)
                P["q"], P["ff"], P["kp"], P["kd"] = q, self._posture_ff(q, R, feet_b), 8.0, 0.3
        self.q_target = P["q"].copy()
        self.stand_flag = 1.0

    def _posture_at_neutral(self):
        P = self.post
        return P is not None and abs(P["h"] - 0.24) < 0.004 and abs(P["pitch"]) < math.radians(0.5)

    def _begin_sit(self):
        if self.post is None:
            self._begin_posture()
        self._enter("sit_lower")

    def _begin_prepare(self):
        """Seated (or protective) -> feet placed under the hips at the low crouch, by a joint move."""
        if self.post is None:
            self._begin_posture()
        P = self.post
        P["q0"] = self.est.q.ravel().copy()
        P["q1"] = nominal_pose(0.085)
        P["T"] = 0.8
        self._enter("prepare")

    def _begin_rise(self):
        est = self.est
        R0 = est.R
        p0 = est.p.copy()
        if self.post is None:
            self._begin_posture()
        P = self.post
        P["feet_w"] = (R0 @ est.foot_b.T).T + p0
        P["base_xy"] = p0[:2].copy()
        P["yaw"] = est.rpy[2]
        P["ground"] = float(np.mean(P["feet_w"][:, 2])) - FOOT_RADIUS
        P["h0"] = max(0.07, float(p0[2] - P["ground"]))
        P["T"] = 1.5
        self.status = "ok" if self.status in ("protective_sit",) else self.status
        self._enter("rise")

    def _handover_to_policy(self):
        self.post = None
        self.hist = None
        self.last_act = np.zeros(12)
        self.h_cmd = float(np.clip(self.est.height, 0.20, 0.26))
        self.v_cmd[:] = 0.0
        self._enter("stand")
        if self.mode.name == "sit":
            self.mode = self.modes["stand"]

    # ------------------------------------------------------------------------------------------ state
    def get_state(self) -> dict:
        est = self.est
        roll, pitch, yaw = est.rpy
        e = self.io.energy
        mode_name = self.mode.name
        if self.state in ("sit_lower", "seated"):
            mode_name = "sit"
        return dict(
            t=self.t,
            mode=mode_name,
            state=self.state,
            status=self.status,
            sit_state={"sit_lower": "lower", "seated": "seated", "prepare": "prepare", "rise": "rise",
                       "protective": "protective"}.get(self.state),
            stepping=bool(self.stand_flag < 0.5),
            gait_phase=float(self.phase),
            joint_pos=est.q.ravel().copy(),
            joint_vel=est.dq.ravel().copy(),
            imu_quat=self.meas["quat"].copy() if self.meas is not None else None,
            imu_rpy=(roll, pitch, yaw),
            imu_gyro=est.omega_b.copy(),
            foot_contact=est.contact.copy(),
            odom_pos=est.p.copy(),
            odom_yaw=yaw,
            odom_vel_world=est.v.copy(),
            odom_vel_body=(Rz(yaw).T @ est.v),
            body_height=float(est.height),
            ground_slope_rp=tuple(float(x) for x in est.slope_rp()),
            odom_distance=float(est.distance),
            head=self.io.head_angles(),
            energy_J=dict(e),
            fallen=bool(self.fallen),
            command=self.cmd_user.copy(),
            blocked=self.blocked_info,
            events=list(self.events),
        )

    @property
    def stepping(self):
        return bool(self.stand_flag < 0.5) or self.state in ("prepare", "rise")

    def truth(self) -> dict:
        """Ground truth for logging and evaluation (never used by the controller)."""
        return self.io.truth()


# an alias: older test code constructs `QuadrupedController(model, data, seed=..., modes=..., ...)`
QuadrupedController = RLQuadrupedController
