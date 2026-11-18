"""The robot's hardware boundary in simulation: onboard sensors in, joint torques out.

Everything the controller knows about the robot passes through `RobotIO.read()` (what the real robot's IMU, joint
encoders and foot switches deliver, with their noise) and everything it does passes through `RobotIO.apply()`
(motor torques, limited like the real actuators). Ground truth is available through `RobotIO.truth()` for logging
and evaluation only; the controller never calls it.

Sensor model (defaults, all seeded and deterministic):
  IMU (AHRS)   orientation = truth x small error: roll / pitch white noise 0.1 deg, heading drift from a gyro bias
               of 8 deg/h (random sign per seed) integrated over time; gyro white noise 0.003 rad/s and bias
               3.9e-5 rad/s; accelerometer white noise 0.02 m/s^2.
  encoders     5000 CPR x 4 x 9 = 180 000 counts per joint turn (3.5e-5 rad steps); velocity from the drive with
               white noise 0.01 rad/s.
  foot switch  touch force > 1.5 N switches on, < 0.6 N switches off (hysteresis), sampled at the control rate.
Actuator model: torque command clipped to +-2.7 N m (12 A x 0.225 N m/A) and to the drive's voltage limit above
  65 rad/s (linear fall to 0 at the no-load 83.8 rad/s at 24 V).
Energy model (`energy` dict): mechanical joint power tau * qdot (positive part counted as drawn, negative part as
  dissipated in the drive, no regeneration), copper (Joule) loss of the field-oriented drive 3/2 I_q^2 R_phase =
  0.75 R_ll I_q^2 with I_q = tau / (0.025 N m/A x 9) and the motor's line-to-line resistance R_ll = 0.452 Ohm
  (manufacturer value, the line-to-line reading is an assumption), plus a constant electronics / compute / sensor
  load (default 12 W) for battery estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

LEGS = ("FL", "FR", "HL", "HR")
JOINTS = ("hip_abad", "hip_flex", "knee")

TORQUE_LIMIT = 2.7
KT_JOINT = 0.025 * 9.0          # N m per A at the joint
R_MOTOR = 0.452                 # Ohm, line to line
JOULE_K = 0.75 * R_MOTOR        # W per A^2 of q-axis current (FOC, wye winding)
OMEGA_FULL = 65.0               # rad/s at the joint with full torque available
OMEGA_NOLOAD = 83.8             # rad/s at the joint, no load, 24 V
ENC_STEP = 2.0 * math.pi / 180000.0


@dataclass
class SensorNoise:
    enabled: bool = True
    rp_sd: float = math.radians(0.1)
    yaw_drift_rate: float = math.radians(8.0) / 3600.0
    gyro_sd: float = 0.003
    gyro_bias: float = 3.9e-5
    acc_sd: float = 0.02
    qvel_sd: float = 0.01
    quantize: bool = True
    touch_on: float = 1.5
    touch_off: float = 0.6


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_small(rx, ry, rz):
    a = math.sqrt(rx * rx + ry * ry + rz * rz)
    if a < 1e-12:
        return np.array([1.0, 0.5 * rx, 0.5 * ry, 0.5 * rz])
    s = math.sin(0.5 * a) / a
    return np.array([math.cos(0.5 * a), rx * s, ry * s, rz * s])


class RobotIO:
    """Binds to the robot inside any MjModel (names optionally prefixed, e.g. after MjSpec.attach)."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str = "", seed: int = 0,
                 noise: SensorNoise | None = None, electronics_power: float = 12.0):
        self.m, self.d, self.prefix = model, data, prefix
        self.noise = noise or SensorNoise()
        self.rng = np.random.default_rng(seed)
        self.electronics_power = electronics_power
        P = lambda n: prefix + n
        self.base_id = model.body(P("base")).id
        self.root_qadr = model.jnt_qposadr[model.joint(P("root")).id]
        self.root_dadr = model.jnt_dofadr[model.joint(P("root")).id]
        self.jnames = [f"{l}_{j}" for l in LEGS for j in JOINTS]
        jid = [model.joint(P(n)).id for n in self.jnames]
        self.qadr = np.array([model.jnt_qposadr[i] for i in jid])
        self.dadr = np.array([model.jnt_dofadr[i] for i in jid])
        self.act = np.array([model.actuator(P(n)).id for n in self.jnames])

        def sadr(n):
            s = model.sensor(P(n))
            return int(model.sensor_adr[s.id]), int(model.sensor_dim[s.id])
        self.s_quat = sadr("imu_quat")[0]
        self.s_gyro = sadr("imu_gyro")[0]
        self.s_acc = sadr("imu_acc")[0]
        self.s_q = np.array([sadr(n + "_pos")[0] for n in self.jnames])
        self.s_dq = np.array([sadr(n + "_vel")[0] for n in self.jnames])
        self.s_touch = np.array([sadr(l + "_foot_touch")[0] for l in LEGS])
        self.foot_site = np.array([model.site(P(l + "_foot")).id for l in LEGS])
        self.foot_geom = np.array([model.geom(P(l + "_foot")).id for l in LEGS])
        try:
            self.head_act = (model.actuator(P("head_pan")).id, model.actuator(P("head_tilt")).id)
            self.head_q = (model.jnt_qposadr[model.joint(P("head_pan")).id],
                           model.jnt_qposadr[model.joint(P("head_tilt")).id])
        except KeyError:
            self.head_act = None
            self.head_q = None
        self.robot_bodies = {b for b in range(model.nbody) if self._is_robot_body(b)}
        self.mass = float(model.body_subtreemass[self.base_id])
        self.reset_sensors()

    def _is_robot_body(self, b):
        while b != 0:
            if b == self.base_id:
                return True
            b = self.m.body_parentid[b]
        return False

    # ------------------------------------------------------------------ state of the sensor models
    def reset_sensors(self):
        self.contact = np.zeros(4, bool)
        sgn = 1.0 if self.rng.random() < 0.5 else -1.0
        self.yaw_drift_rate = sgn * self.noise.yaw_drift_rate
        self.gyro_bias = self.rng.normal(0.0, self.noise.gyro_bias, 3)
        self.t0 = float(self.d.time)
        self.energy = dict(mech_pos=0.0, mech_neg=0.0, joule=0.0, electronics=0.0, time=0.0)
        self.last_tau = np.zeros(12)

    # ------------------------------------------------------------------ sensors
    def read(self) -> dict:
        d, n = self.d, self.noise
        sd = d.sensordata
        quat = sd[self.s_quat:self.s_quat + 4].copy()
        gyro = sd[self.s_gyro:self.s_gyro + 3].copy()
        acc = sd[self.s_acc:self.s_acc + 3].copy()
        q = sd[self.s_q].copy()
        dq = sd[self.s_dq].copy()
        touch = sd[self.s_touch].copy()
        if n.enabled:
            r = self.rng
            yaw_err = self.yaw_drift_rate * (d.time - self.t0)
            quat = _quat_mul(_quat_small(0.0, 0.0, yaw_err), quat)          # heading drift (world frame)
            quat = _quat_mul(quat, _quat_small(r.normal(0, n.rp_sd), r.normal(0, n.rp_sd), 0.0))
            gyro = gyro + self.gyro_bias + r.normal(0, n.gyro_sd, 3)
            acc = acc + r.normal(0, n.acc_sd, 3)
            dq = dq + r.normal(0, n.qvel_sd, 12)
            if n.quantize:
                q = np.round(q / ENC_STEP) * ENC_STEP
        on = touch > n.touch_on
        off = touch < n.touch_off
        self.contact = np.where(on, True, np.where(off, False, self.contact))
        return dict(t=float(d.time), q=q, dq=dq, quat=quat / np.linalg.norm(quat), gyro=gyro, acc=acc,
                    contact=self.contact.copy(), touch=touch)

    # ------------------------------------------------------------------ actuators
    def torque_limit(self, dq):
        w = np.abs(dq)
        lim = np.full(12, TORQUE_LIMIT)
        over = w > OMEGA_FULL
        lim[over] = TORQUE_LIMIT * np.clip((OMEGA_NOLOAD - w[over]) / (OMEGA_NOLOAD - OMEGA_FULL), 0.0, 1.0)
        return lim

    def apply(self, tau: np.ndarray):
        dq_true = self.d.qvel[self.dadr]
        lim = self.torque_limit(dq_true)
        tau = np.clip(tau, -lim, lim)
        self.d.ctrl[self.act] = tau
        self.last_tau = tau
        return tau

    def account_energy(self, dt: float):
        """Call once per physics step after mj_step."""
        tau = self.last_tau
        dq = self.d.qvel[self.dadr]
        p = tau * dq
        e = self.energy
        e["mech_pos"] += float(np.sum(np.maximum(p, 0.0))) * dt
        e["mech_neg"] += float(np.sum(np.minimum(p, 0.0))) * dt
        e["joule"] += float(np.sum(JOULE_K * (tau / KT_JOINT) ** 2)) * dt
        e["electronics"] += self.electronics_power * dt
        e["time"] += dt

    def set_head(self, pan: float, tilt: float):
        if self.head_act is not None:
            self.d.ctrl[self.head_act[0]] = pan
            self.d.ctrl[self.head_act[1]] = tilt

    def head_angles(self):
        if self.head_q is None:
            return None
        return float(self.d.qpos[self.head_q[0]]), float(self.d.qpos[self.head_q[1]])

    # ------------------------------------------------------------------ ground truth (logging / evaluation only)
    def truth(self) -> dict:
        d = self.d
        a = self.root_qadr
        pos = d.qpos[a:a + 3].copy()
        quat = d.qpos[a + 3:a + 7].copy()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        R = R.reshape(3, 3)
        v_body = d.qvel[self.root_dadr:self.root_dadr + 3].copy()      # free joint: world-frame linear velocity
        w_body = d.qvel[self.root_dadr + 3:self.root_dadr + 6].copy()  # body-frame angular velocity
        feet = d.site_xpos[self.foot_site].copy()
        return dict(t=float(d.time), pos=pos, quat=quat, R=R, vel_world=v_body, omega_body=w_body,
                    com=d.subtree_com[self.base_id].copy(), feet=feet)

    def robot_contacts(self):
        """List of (robot geom name, other geom name, normal force) for contacts touching the robot."""
        out = []
        m, d = self.m, self.d
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            r1, r2 = b1 in self.robot_bodies, b2 in self.robot_bodies
            if r1 == r2:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            g_rob, g_oth = (c.geom1, c.geom2) if r1 else (c.geom2, c.geom1)
            out.append((m.geom(g_rob).name, m.geom(g_oth).name, float(f6[0])))
        return out
