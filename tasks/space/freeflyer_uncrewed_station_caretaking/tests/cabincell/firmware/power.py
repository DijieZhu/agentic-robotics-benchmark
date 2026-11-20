"""Energy account and status() telemetry of the task firmware (robot_api.md, status()).

* ``TaskBattery`` -- the simulation package's ``sensors.Battery`` on the 2 h basis (``other_w`` =
  firmware.battery.other_w) that also books the two lamps (firmware.lamp.power_w for each lamp that is on), as the
  payload load.
* ``Telemetry`` -- a periodic task on the rig's scheduler that samples what status() reports at the rate of the
  sensor behind each field, sample-and-hold in between: battery charge at spec pwr.eps.telemetry_rate (resolution
  firmware.battery.pct_resolution), the two arm joint encoders (spec arm.joint_servo resolution) and the two arm servo
  currents at spec arm.joint_servo telemetry_Hz (current = actuator torque / torque constant, quantised; no
  gripper motor current), the IMU at the firmware rate (the rig's ``Imu``; gyro and accelerometer in the body frame).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from spacekit.gripper import q_to_tilt
from spacekit.sensors import Battery


class TaskBattery(Battery):
    """Battery that books the lamps (``lamp_w`` each) on top of the package model."""

    def __init__(self, n_packs: int, capacity_wh: float, soc0: float, other_w: float, lamp_w: float):
        super().__init__(n_packs, capacity_wh, soc0=soc0, other_w=other_w)
        self.lamp_w = float(lamp_w)
        self.lamps = dict(forward=False, aft=False)
        self.used["lamps"] = 0.0

    def set_lamp(self, which: str, on: bool) -> None:
        if which not in self.lamps:
            raise ValueError(which)
        self.lamps[which] = bool(on)

    @property
    def lamps_w(self) -> float:
        return self.lamp_w * sum(1 for v in self.lamps.values() if v)

    def update(self, dt, impeller_w=0.0, arm_mech_w=0.0, payload_w=0.0):
        lw = self.lamps_w
        self.used["lamps"] += lw * dt / 3600.0
        p = super().update(dt, impeller_w, arm_mech_w, payload_w + lw)
        self.used["payload"] -= lw * dt / 3600.0          # booked once, under 'lamps'
        return p

    def pct(self) -> float:
        return 100.0 * self.soc


def quantise(x: float, lsb: float) -> float:
    return float(round(float(x) / lsb) * lsb)


class Telemetry:
    """Sensor samples behind status() (module doc).  ``rows`` keeps nothing; ``snapshot()`` returns the held values."""

    def __init__(self, rig, layer, spec):
        self.rig, self.layer, self.spec = rig, layer, spec
        m = rig.model
        self.act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in layer.names["actuators"]]
        self.jq = [m.jnt_qposadr[j] for j in layer.arm_joints]
        self.enc_rad = math.radians(spec.encoder_deg)
        self.battery_pct = None
        self.arm_deg = (None, None)
        self.arm_a = (0.0, 0.0)
        self.imu = None
        self._t_bat = -1e9
        self._t_servo = -1e9
        self.sample(None)
        rig.sim.add_periodic(self.sample, rig.period, "firmware_telemetry")

    def sample(self, sim) -> None:
        t = float(self.rig.data.time)
        d = self.rig.data
        if t - self._t_bat >= 1.0 / self.spec.battery_hz - 1e-9:
            self._t_bat = t
            self.battery_pct = quantise(100.0 * self.rig.battery.soc, self.spec.battery_pct_resolution)
        if t - self._t_servo >= 1.0 / self.spec.servo_hz - 1e-9:
            self._t_servo = t
            q = [float(d.qpos[a]) for a in self.jq]
            qq = [quantise(v, self.enc_rad) for v in q]
            self.arm_deg = (float(q_to_tilt(qq[0])), float(math.degrees(qq[1])))
            tau = [float(d.actuator_force[a]) for a in self.act]
            self.arm_a = tuple(quantise(abs(x) / self.spec.torque_per_amp_nm, self.spec.current_lsb_a) for x in tau)
        imu = getattr(self.rig, "imu_out", None)
        if imu is not None:
            self.imu = (np.asarray(imu[0], float).copy(), np.asarray(imu[1], float).copy())

    def snapshot(self) -> dict:
        g, a = self.imu if self.imu is not None else (np.zeros(3), np.zeros(3))
        return dict(battery_pct=self.battery_pct, arm_tilt_deg=round(self.arm_deg[0], 3), arm_pan_deg=round(self.arm_deg[1], 3),
                    arm_current_a=[round(self.arm_a[0], 5), round(self.arm_a[1], 5)],
                    imu=dict(accel_m_s2=[float(x) for x in a], gyro_rad_s=[float(x) for x in g]))
