"""Locomotion controller of the quadruped: a gait clock, Raibert-style footsteps, swing trajectories with a settable
apex, a quasi-static force distribution for the stance feet, joint PD with feed-forward for the swing legs.

Signals used (all onboard, through RobotIO.read()): IMU orientation and angular rate, accelerometer, joint encoders,
foot switches; the leg-odometry estimate built from them (estimator.py). The controller never reads the simulator's
base pose; RobotIO.truth() exists only for logging.

Rates: the physics step is 1 ms. The joint loop (torque = feed-forward + Kp (q_des - q) + Kd (dq_des - dq), limited
to +-2.7 N m) runs every step (1 kHz, the robot's joint-torque loop); the planner (estimator, modes, gait, footholds,
force distribution) runs every planner_div = 2 steps (500 Hz). step() advances the simulation by one 1 ms period.

Gait: a trot clock with lift-off gating (a pair lifts only when the other diagonal pair is down), late-touchdown
search, early touchdown, a trip reflex. Stance posture: all knees backward by default (knees="back").

Public interface (mission code):
    ctl = QuadrupedController(model, data, prefix="")
    ctl.reset(pos=(x, y), yaw=0.0)            # places the robot standing (simulation set-up), resets the estimate
    ctl.set_command(vx, vy, yaw_rate, body_height=None, mode=None)
    ctl.set_head(pan, tilt)
    ctl.step()                                 # one 1 ms control period
    ctl.get_state()                            # what the robot knows (no ground truth)
Modes: stand, walk (default gait for moving), high_step, crouch_walk, sit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .balance import distribute
from .fastmath import cross3, cross_rows, solve_damped_batch
from .estimator import StateEstimator, Rz, G
from .kinematics import LegKinematics, LEGS
from .plant import RobotIO, SensorNoise

FOOT_RADIUS = 0.017


@dataclass(frozen=True)
class Mode:
    name: str
    gait: str | None                 # 'trot' or None (standing)
    period: float = 0.32             # gait cycle (s)
    duty: float = 0.55               # stance fraction per leg
    swing_height: float = 0.05       # apex above the higher of lift-off and touchdown (m)
    profile: str = "bell"            # 'bell' (smooth arc) or 'box' (lift, move, lower: for obstacles)
    height: float = 0.24             # default body height (base origin above the ground, m)
    height_range: tuple = (0.20, 0.28)
    v_max: tuple = (0.5, 0.25, 1.2)  # |vx|, |vy| (m/s), |yaw rate| (rad/s)
    accel: tuple = (0.8, 0.5, 2.5)   # reference rate limits
    stance_width: float = 0.0        # extra half-width of the stance (m)
    kv_step: float = 0.10            # Raibert velocity-feedback gain (s)
    step_max: float = 0.11           # max foothold offset from the neutral point (m)


MODES = {
    "stand": Mode("stand", None, height=0.24, height_range=(0.12, 0.30), v_max=(0.0, 0.0, 0.0)),
    "walk": Mode("walk", "trot", 0.32, 0.55, 0.05, "bell", 0.24, (0.20, 0.28), (0.50, 0.25, 1.2)),
    "high_step": Mode("high_step", "trot", 0.48, 0.58, 0.11, "box", 0.25, (0.22, 0.28), (0.30, 0.10, 0.6),
                      accel=(0.4, 0.3, 1.5)),
    "crouch_walk": Mode("crouch_walk", "trot", 0.36, 0.60, 0.035, "bell", 0.17, (0.13, 0.21), (0.20, 0.10, 0.6),
                        accel=(0.4, 0.3, 1.5)),
    "sit": Mode("sit", None, height=0.045, height_range=(0.04, 0.06), v_max=(0.0, 0.0, 0.0), stance_width=0.05),
}


@dataclass
class Gains:
    kp_z: float = 500.0
    kd_z: float = 45.0
    kp_xy: float = 25.0
    kd_xy: float = 10.0
    kp_rp: float = 350.0
    kd_rp: float = 30.0
    kp_yaw: float = 120.0
    kd_yaw: float = 18.0
    w_wrench: tuple = (1.0, 1.0, 2.0, 12.0, 12.0, 6.0)
    mu: float = 0.45                 # design friction coefficient of the force distribution
    f_min: float = 1.0
    reg: float = 2e-4
    kp_swing: tuple = (8.0, 8.0, 6.0)
    kd_swing: tuple = (0.25, 0.25, 0.18)
    kd_stance: tuple = (0.04, 0.04, 0.03)
    kp_hold: tuple = (5.0, 5.0, 5.0)
    kd_hold: tuple = (0.15, 0.15, 0.12)
    fric_comp: float = 0.08          # N m, compensated part of the joint's Coulomb friction (swing legs)
    pos_err_max: float = 0.04
    yaw_err_max: float = 0.25
    hurry: bool = False              # speed up the swing when the support is lost (tested: hurts step climbing)
    overstep: bool = False           # high_step: a foot landing on top of something steps on 8 cm further
                                     # (tested: harmful on pipes and sills; kept off)
    robust_plane: bool = False       # down-weight a single foot off the ground plane (hurts step climbing)
    support_debounce: float = 0.0    # s of contact before a foot counts as support (tested 0.01: hurts step-ups)


_UP = np.array([0.0, 0.0, 1.0])


def _jsolve(J, v, lam=1e-4):
    """Damped least-squares solution of J x = v (3 x 3)."""
    return np.linalg.solve(J.T @ J + lam * np.eye(3), J.T @ v)


def _minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


class QuadrupedController:
    TROT_OFFSETS = np.array([0.0, 0.5, 0.5, 0.0])        # FL, FR, HL, HR

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "", seed: int = 0,
                 noise: SensorNoise | None = None, gains: Gains | None = None, modes: dict | None = None,
                 knees: str = "back", planner_div: int = 2):
        """knees: 'x' = the robot's default stance (front knees point backward, hind knees forward);
        'back' = all knees point backward (hind legs mirrored), which keeps the hind shanks behind the feet."""
        self.m, self.d = model, data
        self.dt = float(model.opt.timestep)               # physics step = joint-loop period (1 ms)
        self.hl_div = int(planner_div)                    # estimator / planner / force distribution every N steps
        self.hl_dt = self.dt * self.hl_div
        self.io = RobotIO(model, data, prefix, seed=seed, noise=noise)
        self.kin = LegKinematics(model, prefix)
        self.g = gains or Gains()
        self.est = StateEstimator(self.kin, self.hl_dt, FOOT_RADIUS, robust_plane=self.g.robust_plane)
        self.modes = dict(MODES if modes is None else modes)
        self.knees = knees
        self.knee_sign = np.array([-1.0, -1.0, 1.0, 1.0]) if knees == "x" else -np.ones(4)
        self.prefix = prefix
        self._kd_stance = np.array(self.g.kd_stance)
        self._kp_hold = np.array(self.g.kp_hold)
        self._kd_hold = np.array(self.g.kd_hold)
        self._w_wrench = np.array(self.g.w_wrench)
        self._mass_props()
        self.mode = self.modes["stand"]
        self.reset_internal()

    # ------------------------------------------------------------------ model-derived constants
    def _mass_props(self):
        m, P = self.m, self.prefix
        d = mujoco.MjData(m)
        qa = self.io.root_qadr
        d.qpos[qa:qa + 7] = [0, 0, 0, 1, 0, 0, 0]
        hs = 1.0 if self.knees == "x" else -1.0
        d.qpos[self.io.qadr] = np.array([0, .8, -1.6, 0, .8, -1.6, 0, -.8 * hs, 1.6 * hs, 0, -.8 * hs, 1.6 * hs])
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        leg_bodies = {m.body(P + f"{l}_{k}").id for l in LEGS for k in ("hip", "thigh", "shank")}
        msum, mt = np.zeros(3), 0.0
        I = np.zeros((3, 3))
        for b in self.io.robot_bodies:
            if b in leg_bodies:
                continue
            msum += m.body_mass[b] * d.xipos[b]
            mt += m.body_mass[b]
        self.trunk_mass = mt
        self.trunk_com = msum / mt
        self.mass = self.io.mass
        com = d.subtree_com[self.io.base_id]
        for b in self.io.robot_bodies:
            R = d.ximat[b].reshape(3, 3)
            Ib = R @ np.diag(m.body_inertia[b]) @ R.T
            r = d.xipos[b] - com
            I += Ib + m.body_mass[b] * (r @ r * np.eye(3) - np.outer(r, r))
        self.I_body = I
        self.stand_q = d.qpos[self.io.qadr].reshape(4, 3).copy()

    # ------------------------------------------------------------------ public API
    def reset(self, pos=(0.0, 0.0), yaw: float = 0.0, z: float | None = None, mode: str = "stand",
              phase: float = 0.0, settle: float = 0.3, ground_z: float = 0.0, roll: float = 0.0):
        """Place the robot standing at (x, y, yaw) on ground of height ground_z (simulation set-up only) and reset
        the controller and the onboard estimate (the start pose is known). roll: start on a side slope of that
        angle, body parallel to it."""
        m, d = self.m, self.d
        qa, da = self.io.root_qadr, self.io.dadr
        h = self.modes["stand"].height if z is None else z
        h_w = h / math.cos(roll)
        d.qpos[qa:qa + 3] = [pos[0], pos[1], ground_z + h_w + 0.002]
        qz = np.array([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
        qx = np.array([math.cos(roll / 2), math.sin(roll / 2), 0, 0])
        w1, x1, y1, z1 = qz
        w2, x2, y2, z2 = qx
        d.qpos[qa + 3:qa + 7] = [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                                 w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]
        d.qpos[self.io.qadr] = self.stand_q.ravel()
        d.qvel[self.io.root_dadr:self.io.root_dadr + 6] = 0
        d.qvel[self.io.dadr] = 0
        d.ctrl[:] = 0
        if self.io.head_q is not None:
            d.qpos[list(self.io.head_q)] = 0
        mujoco.mj_forward(m, d)
        self.io.reset_sensors()
        self.reset_internal(phase)
        meas = self.io.read()
        self.est.reset(np.array([pos[0], pos[1], ground_z + h_w + 0.002]), yaw, meas)
        self.yaw_ref = self.est.rpy[2]
        self.p_ref = self.est.p[:2].copy()
        self.h_ref = self.est.height
        self.slope_rp = np.array(self.est.slope_rp()) * np.array([0.6, 0.9])
        self.mode = self.modes["stand"]
        self.set_command(0, 0, 0, mode=mode)
        n = int(round(settle / self.dt))
        for _ in range(n):
            self.step()
        self.io.energy = dict(mech_pos=0.0, mech_neg=0.0, joule=0.0, electronics=0.0, time=0.0)

    def reset_internal(self, phase=0.0):
        self.t = 0.0
        self._k = 0
        self._ff = np.zeros(12)
        self._kp = np.zeros(12)
        self._kd = np.zeros(12)
        self._qd = np.zeros(12)
        self._dqd = np.zeros(12)
        self.stepping = False
        self.phase = float(phase)
        self.phase0 = float(phase)
        self.swing = np.zeros(4, bool)
        self.swing_t = np.zeros(4)
        self.late_t = np.zeros(4)
        self.lift_pos = np.zeros((4, 3))
        self.target = np.zeros((4, 3))
        self.stance_age = np.ones(4)
        self.touchdown = np.zeros(4, bool)
        self.lift_pending = np.zeros(4, bool)
        self.pending_t = np.zeros(4)
        self.swing_extra = np.zeros(4)
        self.trips = 0
        self.hurries = 0
        self.oversteps = np.zeros(4, int)
        self.overstep_count = 0
        self.contact_age = np.ones(4)
        self.stop_t = None
        self.allow_lift = True
        self.cmd = np.zeros(3)
        self.v_ref = np.zeros(3)                  # body frame vx, vy, yaw rate
        self.h_cmd = None
        self.pitch_cmd = 0.0
        self.pending_mode = None
        self.slope_rp = np.zeros(2)
        self.sit_state = None                     # None, 'widen', 'lower', 'seated', 'rise', 'narrow'
        self.sit_t = 0.0
        self.width_extra = 0.0
        self.width_target = 0.0
        self.hold_q = None
        self.fallen = False
        self.last_forces = np.zeros((4, 3))
        self.head_cmd = (0.0, 0.0)
        self.yaw_ref = 0.0
        self.p_ref = np.zeros(2)
        self.h_ref = 0.24

    def set_command(self, vx: float, vy: float, yaw_rate: float, body_height: float | None = None,
                    mode: str | None = None):
        """Velocity command in the body heading frame (m/s, m/s, rad/s). mode None keeps the current mode, except
        that a moving command in 'stand' (or 'sit') switches to 'walk'. body_height None = the mode's default."""
        self.cmd = np.array([vx, vy, yaw_rate], float)
        moving = np.any(np.abs(self.cmd) > 1e-4)
        if mode is None:
            cur = self.pending_mode or self.mode.name
            if cur in ("stand", "sit") and moving:
                mode = "walk"
        if mode is not None:
            if mode not in self.modes:
                raise ValueError(f"unknown mode {mode!r}; modes: {sorted(self.modes)}")
            if mode != self.mode.name or self.pending_mode is not None:
                self.pending_mode = mode if mode != self.mode.name else None
        self.h_cmd = body_height

    def set_head(self, pan: float, tilt: float):
        self.head_cmd = (float(pan), float(tilt))

    def set_body_pitch(self, pitch: float):
        """Extra body pitch (rad, + = nose down) while standing, e.g. to look at something low."""
        self.pitch_cmd = float(np.clip(pitch, -0.25, 0.25))

    # ------------------------------------------------------------------ the control step
    def step(self):
        """One control period (1 ms): sensors -> (every planner_div steps: estimator, modes, gait, footsteps, force
        distribution) -> joint torque = feed-forward + Kp (q_des - q) + Kd (dq_des - dq) -> physics step."""
        dt = self.dt
        meas = self.io.read()
        self.meas = meas
        if self._k % self.hl_div == 0:
            self._planner(meas)
        self._k += 1
        tau = self._ff + self._kp * (self._qd - meas["q"]) + self._kd * (self._dqd - meas["dq"])
        self.io.apply(tau)
        self.io.set_head(*self.head_cmd)
        mujoco.mj_step(self.m, self.d)
        self.io.account_energy(dt)
        self.t += dt

    def _planner(self, meas):
        dt = self.hl_dt
        stance_mask = (~self.swing) & meas["contact"]
        self.contact_age = np.where(meas["contact"], self.contact_age + dt, 0.0)
        self.est.update(meas, stance_mask, self.touchdown)
        self.touchdown[:] = False
        self._mode_logic()
        self._update_refs()
        self.stance_age[~self.swing] += dt
        if self.stepping:
            self._gait(meas)
        self._commands(meas)
        roll, pitch, _ = self.est.rpy
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            self.fallen = True

    # ------------------------------------------------------------------ modes
    def _mode_logic(self):
        """Mode switching. Gait changes happen only with all feet down (stop, switch, restart). Sitting:
        widen -> lower -> seated (body on its belly, weak joint hold); standing up: prepare (feet placed on the
        ground by a joint move) -> rise (balance controller) -> narrow (one cycle of steps) -> stand."""
        dt = self.hl_dt
        want = self.pending_mode
        if want is not None:
            new = self.modes[want]
            if self.stepping:
                self._request_stop()
            elif self.sit_state in ("seated", "lower"):
                self._begin_prepare()
            elif self.sit_state in ("prepare", "rise", "narrow", "widen"):
                pass
            else:
                self.mode = new
                self.pending_mode = None
                if want == "sit":
                    self.sit_state = "widen"
                    self.sit_t = 0.0
                    self.width_target = new.stance_width
                    self.width_extra = new.stance_width
                    self._start_stepping(width_only=True)
        m = self.mode
        moving = np.any(np.abs(self.cmd) > 1e-4)
        # sitting sequence
        if self.sit_state is not None:
            self.sit_t += dt
            st = self.sit_state
            if st == "widen" and not self.stepping:
                self.sit_state, self.sit_t = "lower", 0.0
            elif st == "lower" and self.est.height < self.modes["sit"].height + 0.008:
                self.sit_state, self.sit_t = "seated", 0.0
                self.hold_q = self.est.q.copy()
            elif st == "prepare" and self.sit_t > self.prepare_T + 0.1 and (
                    self.est.contact.all() or self.sit_t > self.prepare_T + 1.0):
                self.sit_state, self.sit_t = "rise", 0.0
                self.h_ref = self.est.height
            elif st == "rise" and self.est.height > self.modes["stand"].height - 0.02:
                self.sit_state, self.sit_t = "narrow", 0.0
                self.width_target = 0.0
                self.width_extra = 0.0
                self._start_stepping(width_only=True)
            elif st == "narrow" and not self.stepping:
                self.sit_state = None
                self.mode = self.modes["stand"]
            return
        if m.gait is not None:
            if moving and not self.stepping and self.pending_mode is None:
                self._start_stepping()
            if self.stepping and not moving and self.stop_t is None:
                if np.linalg.norm(self.est.v[:2]) < 0.10:
                    self._request_stop()
            if self.stepping and moving and self.stop_t is not None and self.pending_mode is None:
                self.stop_t = None
                self.allow_lift = True

    def _begin_prepare(self):
        """Seated -> place every foot on the ground under its hip at the sitting width (joint-space move)."""
        est, kin = self.est, self.kin
        h = est.height
        targets = np.zeros((4, 3))
        for i in range(4):
            n = self._neutral(i)
            targets[i] = [n[0], n[1], -h + FOOT_RADIUS - 0.025]
        self.prep_q0 = est.q.copy()
        self.prep_q1 = np.array([kin.ik_leg(i, targets[i], self.knee_sign[i]) for i in range(4)])
        self.prepare_T = 0.8
        self.sit_state, self.sit_t = "prepare", 0.0
        self.hold_q = None

    def _start_stepping(self, width_only=False):
        gm = self.mode if self.mode.gait is not None else self.modes["walk"]
        self.gait_mode = gm
        self.stepping = True
        self.allow_lift = True
        self.stop_t = None
        # first lift-off of the (FL, HR) pair immediately
        self.phase = (gm.duty - 1e-3 + self.phase0) % 1.0
        self.width_only = width_only
        if width_only:
            self.stop_t = 1.0 * gm.period    # step in place one cycle, then stop

    def _request_stop(self):
        """Stop stepping once the velocity reference is at rest: one more cycle of steps re-centres the feet."""
        if self.stop_t is None and np.all(np.abs(self.v_ref) < 0.02):
            self.stop_t = 1.0 * self.gait_mode.period

    # ------------------------------------------------------------------ references
    def _update_refs(self):
        dt, m, est = self.hl_dt, self.mode, self.est
        vmax = np.array(m.v_max)
        if self.sit_state is not None or self.pending_mode is not None and self.stepping:
            cmd = np.zeros(3)
        else:
            cmd = np.clip(self.cmd, -vmax, vmax)
        if not self.stepping:
            cmd = np.zeros(3)
        acc = np.array(m.accel if m.gait else (1.0, 1.0, 2.0))
        self.v_ref += np.clip(cmd - self.v_ref, -acc * dt, acc * dt)
        yaw = est.rpy[2]
        self.yaw_ref += self.v_ref[2] * dt
        e = (self.yaw_ref - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(e) > self.g.yaw_err_max:
            self.yaw_ref = yaw + math.copysign(self.g.yaw_err_max, e)
        c, s = math.cos(yaw), math.sin(yaw)
        vw = np.array([c * self.v_ref[0] - s * self.v_ref[1], s * self.v_ref[0] + c * self.v_ref[1]])
        self.v_ref_w = vw
        self.p_ref += vw * dt
        err = self.p_ref - est.p[:2]
        n = np.linalg.norm(err)
        if n > self.g.pos_err_max:
            self.p_ref = est.p[:2] + err * (self.g.pos_err_max / n)
        # body height
        if self.sit_state in ("lower", "seated", "prepare"):
            h_goal = self.modes["sit"].height
            rate = 0.10
        elif self.sit_state in ("rise",):
            h_goal = self.modes["stand"].height
            rate = 0.08
        elif self.sit_state in ("widen", "narrow"):
            h_goal = self.modes["stand"].height
            rate = 0.10
        else:
            gm = m
            if self.pending_mode is not None and not self.stepping:
                gm = self.modes[self.pending_mode]
            lo, hi = gm.height_range
            h_goal = gm.height if self.h_cmd is None else float(np.clip(self.h_cmd, lo, hi))
            rate = 0.12
        self.h_ref += float(np.clip(h_goal - self.h_ref, -rate * dt, rate * dt))
        # slope adaptation (pitch / roll follow the ground plane, partly)
        sr, sp = est.slope_rp()
        k = 1.0 - math.exp(-dt / 0.15)
        self.slope_rp += k * (np.array([0.6 * sr, 0.9 * sp]) - self.slope_rp)

    # ------------------------------------------------------------------ gait
    def _gait(self, meas):
        dt = self.hl_dt
        gm = self.gait_mode
        T, duty = gm.period, gm.duty
        Tsw = (1.0 - duty) * T
        prev = self.phase
        self.phase = (self.phase + dt / T) % 1.0
        if self.stop_t is not None:
            self.stop_t -= dt
            if self.stop_t <= 0.0:
                self.allow_lift = False
        if self.width_only:
            # while stepping in place to change the stance width, move the width with the steps
            self.width_extra += float(np.clip(self.width_target - self.width_extra, -0.25 * dt, 0.25 * dt))
        contact = meas["contact"]
        # support check: a stance foot that lost contact (slid off an obstacle) while the other pair is in the
        # air -> the swinging legs finish their swing three times faster (get a foot down)
        support = (~self.swing) & contact
        diag_ok = (support[0] and support[3]) or (support[1] and support[2]) or support.sum() >= 3
        hurry = self.g.hurry and (not diag_ok) and self.swing.any()
        if hurry:
            self.hurries += 1
        for i in range(4):
            off = self.TROT_OFFSETS[i]
            a = (prev - off) % 1.0
            b = (self.phase - off) % 1.0
            crossed = (a < duty <= b) if b >= a else (a < duty or duty <= b)
            if self.swing[i]:
                self.swing_t[i] += dt * (3.0 if hurry and self.swing_t[i] > 0.3 * Tsw else 1.0)
                s = self.swing_t[i] / Tsw
                # trip reflex: a foot that hits something early in its swing lifts higher for the rest of it
                if contact[i] and 0.06 < s < 0.55 and self.swing_extra[i] < 0.05:
                    self.swing_extra[i] += 0.025
                    self.trips += 1
                td = contact[i] and s > 0.55
                if td and self.g.overstep and gm.profile == "box" and self.oversteps[i] == 0:
                    fw = self.est.foot_w[i]
                    z_exp = self.est.plane_z(fw[0], fw[1])
                    if fw[2] - z_exp > 0.025:
                        # landed on top of an obstacle: lift again and set the foot down 8 cm further on
                        yaw = self.est.rpy[2]
                        self.lift_pos[i] = fw.copy()
                        self.target[i] = fw + np.array([0.08 * math.cos(yaw), 0.08 * math.sin(yaw), 0.0])
                        self.target[i][2] = z_exp
                        self.swing_t[i] = 0.0
                        self.late_t[i] = 0.0
                        self.swing_extra[i] = 0.05 - gm.swing_height   # apex 5 cm above the obstacle top
                        self.oversteps[i] = 1
                        self.overstep_count += 1
                        continue
                if s >= 1.0:
                    self.late_t[i] += dt
                if td or self.late_t[i] > 0.25:
                    self.swing[i] = False
                    self.stance_age[i] = 0.0
                    self.touchdown[i] = True
                    self.late_t[i] = 0.0
                    self.oversteps[i] = 0
            else:
                if crossed and self.allow_lift and self.stance_age[i] > 0.05:
                    self.lift_pending[i] = True
                    self.pending_t[i] = 0.0
        # lift-off gating: a pending pair lifts only if the legs left on the ground support the robot
        if self.lift_pending.any():
            self.pending_t[self.lift_pending] += dt
            lifting = self.lift_pending.copy()
            support = (~self.swing) & (~lifting) & contact & (self.contact_age >= self.g.support_debounce)
            ok = (support[0] and support[3]) or (support[1] and support[2]) or support.sum() >= 3
            if ok:
                for i in np.nonzero(lifting)[0]:
                    self.swing[i] = True
                    self.swing_t[i] = 0.0
                    self.late_t[i] = 0.0
                    self.swing_extra[i] = 0.0
                    self.lift_pos[i] = self.est.foot_w[i].copy()
                    self.target[i] = self._foothold(i, 0.0)
                self.lift_pending[:] = False
            else:
                # give up this cycle after a whole swing time (the legs stay down; the clock goes on)
                drop = self.pending_t > Tsw
                self.lift_pending[drop] = False
        if not self.allow_lift and not np.any(self.swing) and not self.lift_pending.any():
            self.stepping = False
            self.stop_t = None
            self.allow_lift = True

    def _neutral(self, i):
        sy = 1.0 if i in (0, 2) else -1.0
        n = self.kin.neutral[i].copy()
        n[0] += self.trunk_com_xy()[0]
        n[1] += sy * self.width_extra
        return n

    def trunk_com_xy(self):
        return self.com_b[:2] if hasattr(self, "com_b") else self.trunk_com[:2]

    def _foothold(self, i, s):
        gm, est = self.gait_mode, self.est
        T, duty = gm.period, gm.duty
        Tsw, Tst = (1.0 - duty) * T, duty * T
        t_rem = max(0.0, (1.0 - s) * Tsw)
        yaw = est.rpy[2]
        wz = self.v_ref[2]
        Ryaw = Rz(yaw)
        vb = np.array([self.v_ref[0], self.v_ref[1], 0.0])
        p_td = est.p + Ryaw @ (vb * t_rem)
        yaw_td = yaw + wz * t_rem
        n = self._neutral(i)
        n[2] = 0.0
        hip_td = p_td + Rz(yaw_td) @ n
        # Raibert: half the stance travel ahead, plus velocity feedback, plus the turn during half a stance
        v_ref_w = Ryaw @ vb
        fb = gm.kv_step * (est.v - v_ref_w)
        fb[2] = 0.0
        turn = (Rz(wz * Tst / 2) - np.eye(3)) @ (Rz(yaw_td) @ n)
        off = Ryaw @ (vb * Tst / 2) + fb + turn
        off[2] = 0.0
        r = np.linalg.norm(off[:2])
        if r > gm.step_max:
            off *= gm.step_max / r
        tgt = hip_td + off
        tgt[2] = est.plane_z(tgt[0], tgt[1])
        return tgt

    def _swing_point(self, i, s):
        gm = self.gait_mode
        p0, p1 = self.lift_pos[i], self.target[i]
        H = gm.swing_height + self.swing_extra[i]
        top = max(p0[2], p1[2]) + H
        if gm.profile == "box":
            sx = _minjerk((s - 0.20) / 0.52)
            if s < 0.32:
                z = p0[2] + (top - p0[2]) * _minjerk(s / 0.32)
            elif s < 0.66:
                z = top
            else:
                z = top + (p1[2] - top) * _minjerk((s - 0.66) / 0.34)
        else:
            sx = _minjerk((s - 0.08) / 0.78)
            if s < 0.45:
                z = p0[2] + (top - p0[2]) * _minjerk(s / 0.45)
            else:
                z = top + (p1[2] - top) * _minjerk((s - 0.45) / 0.55)
        xy = p0[:2] + (p1[:2] - p0[:2]) * sx
        return np.array([xy[0], xy[1], z])

    # ------------------------------------------------------------------ torques
    def _set_leg(self, i, ff, kp, kd, qd, dqd):
        sl = slice(3 * i, 3 * i + 3)
        self._ff[sl], self._kp[sl], self._kd[sl], self._qd[sl], self._dqd[sl] = ff, kp, kd, qd, dqd

    def _commands(self, meas):
        """Joint-level commands (feed-forward, gains, references) for the 1 kHz loop."""
        g, est, kin = self.g, self.est, self.kin
        R = est.R
        q, dq = est.q, est.dq
        foot_b, J = est.foot_b, est.J
        # whole-robot CoM in the base frame (trunk + legs from the encoders)
        com_b, tau_g = kin.com_and_gravity(q, R.T @ np.array([0.0, 0.0, -G]), self.trunk_mass, self.trunk_com)
        self.com_b = com_b
        com_w = R @ com_b

        if self.sit_state == "seated" and self.hold_q is not None:
            # low-power hold: weak joint PD on the posture the robot sat down in (the body rests on its belly)
            self._ff[:], self._kp[:], self._kd[:] = 0.0, 1.0, 0.05
            self._qd[:], self._dqd[:] = self.hold_q.ravel(), 0.0
            return
        if self.sit_state == "prepare":
            sp = _minjerk(self.sit_t / self.prepare_T)
            q_des = self.prep_q0 + (self.prep_q1 - self.prep_q0) * sp
            self._ff[:], self._kp[:], self._kd[:] = tau_g.ravel(), 8.0, 0.2
            self._qd[:], self._dqd[:] = q_des.ravel(), 0.0
            return

        contact = meas["contact"]
        in_dist = (~self.swing) & contact
        # stance legs that have just touched down / are about to lift: soft entry and exit
        reg = np.full(4, g.reg)
        for i in range(4):
            if in_dist[i]:
                ramp = min(1.0, self.stance_age[i] / 0.02)
                reg[i] = g.reg / max(ramp, 0.02) ** 2
        # desired wrench
        yaw = est.rpy[2]
        p, v = est.p, est.v
        m = self.mass
        h = est.height
        F = np.zeros(3)
        F[:2] = m * (g.kp_xy * (self.p_ref - p[:2]) + g.kd_xy * (self.v_ref_w - v[:2]))
        F[2] = m * (g.kp_z * (self.h_ref - h) + g.kd_z * (0.0 - v[2]) + G)
        roll, pitch, _ = est.rpy
        r_ref = self.slope_rp[0]
        p_ref = self.slope_rp[1] + (self.pitch_cmd if not self.stepping else 0.0)
        e_yaw = (self.yaw_ref - yaw + math.pi) % (2 * math.pi) - math.pi
        c, s = math.cos(yaw), math.sin(yaw)
        e_rp_h = np.array([r_ref - roll, p_ref - pitch])
        e_w = np.array([c * e_rp_h[0] - s * e_rp_h[1], s * e_rp_h[0] + c * e_rp_h[1], e_yaw])
        w_w = R @ est.omega_b
        w_ref = np.array([0.0, 0.0, self.v_ref[2]])
        kp_ang = np.array([g.kp_rp, g.kp_rp, g.kp_yaw])
        kd_ang = np.array([g.kd_rp, g.kd_rp, g.kd_yaw])
        Iw = R @ self.I_body @ R.T
        Mw = Iw @ (kp_ang * e_w + kd_ang * (w_ref - w_w))
        W = np.concatenate([F, Mw])
        idx = np.nonzero(in_dist)[0]
        forces = np.zeros((4, 3))
        if len(idx):
            r = (R @ foot_b[idx].T).T - com_w
            # friction pyramids about the gravity axis: the local surface normal under each foot is unknown to a
            # blind robot (a fitted plane is wrong at every slope transition and on steps)
            f = distribute(r, W, _UP, g.mu, g.f_min, self._w_wrench, reg[idx])
            forces[idx] = f
        self.last_forces = forces
        # stance legs: tau = -J^T f + damping toward the joint velocity the body motion implies
        wb_ref = R.T @ w_ref
        vb_ref = R.T @ np.array([self.v_ref_w[0], self.v_ref_w[1], 0.0])
        if len(idx):
            fb = forces[idx] @ R                                   # rows: R^T f_i
            t_st = -np.einsum("nji,nj->ni", J[idx], fb)            # -J^T f
            v_foot = -(vb_ref[None, :] + cross_rows(-foot_b[idx], wb_ref))   # -(v + w x p)
            dq_ref = solve_damped_batch(J[idx], v_foot)
            for n_, i in enumerate(idx):
                self._set_leg(i, t_st[n_], 0.0, self._kd_stance, q[i], dq_ref[n_])
        # swing legs and legs without contact
        kp_sw, kd_sw = np.array(g.kp_swing), np.array(g.kd_swing)
        base_w = est.p
        for i in range(4):
            if in_dist[i]:
                continue
            if self.swing[i]:
                gm = self.gait_mode
                Tsw = (1.0 - gm.duty) * gm.period
                s = self.swing_t[i] / Tsw
                if s < 0.8 and self.oversteps[i] == 0:
                    self.target[i] = self._foothold(i, s)
                if s <= 1.0:
                    pw = self._swing_point(i, s)
                    pw2 = self._swing_point(i, min(s + 0.01, 1.0))
                    vw = (pw2 - pw) / (0.01 * Tsw) if s + 0.01 <= 1.0 else np.zeros(3)
                else:
                    # late: keep reaching down slowly
                    pw = self.target[i].copy()
                    pw[2] -= 0.45 * self.late_t[i]
                    vw = np.array([0.0, 0.0, -0.45])
                pb = R.T @ (pw - base_w)
                vb = R.T @ (vw - est.v) - cross3(est.omega_b, pb)
                q_des = kin.ik_leg(i, pb, self.knee_sign[i])
                dq_des = np.clip(_jsolve(J[i], vb), -25, 25)
                self._set_leg(i, g.fric_comp * np.tanh(dq_des / 0.3) + tau_g[i], kp_sw, kd_sw, q_des, dq_des)
            else:
                # scheduled stance but no contact (stepped into a hole, slipping): extend towards the ground
                pb = foot_b[i] + R.T @ np.array([0.0, 0.0, -0.012])
                q_des = kin.ik_leg(i, pb, self.knee_sign[i])
                self._set_leg(i, tau_g[i], self._kp_hold, self._kd_hold, q_des, 0.0)

    # ------------------------------------------------------------------ state for the mission code
    def get_state(self) -> dict:
        est = self.est
        meas_q = est.q.ravel().copy()
        roll, pitch, yaw = est.rpy
        head = self.io.head_angles()
        e = self.io.energy
        return dict(
            t=self.t,
            mode=self.mode.name if self.sit_state is None else "sit",
            pending_mode=self.pending_mode,
            sit_state=self.sit_state,
            stepping=bool(self.stepping),
            gait_phase=float(self.phase),
            joint_pos=meas_q,
            joint_vel=est.dq.ravel().copy(),
            imu_quat=self.meas["quat"].copy() if getattr(self, "meas", None) is not None else None,
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
            head=head,
            energy_J=dict(e),
            fallen=bool(self.fallen),
            command=self.cmd.copy(),
        )

    def truth(self) -> dict:
        """Ground truth for logging and evaluation (never used by the controller)."""
        return self.io.truth()
