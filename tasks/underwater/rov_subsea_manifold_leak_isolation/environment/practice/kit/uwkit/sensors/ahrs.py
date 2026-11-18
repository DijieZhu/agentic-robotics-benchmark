"""IMU (gyro, accelerometer, magnetometer) and the on-board AHRS (complementary filter).

World W is ENU for magnetics: x east, y north, z up.  Earth field (total F, inclination I down-positive, declination
D east-positive): B = F (cos I sin D, cos I cos D, -sin I).

Magnetic disturbances (properties of the site): steel structures as point dipoles
    B_dip(r) = mu0 / (4 pi) * (3 (m . r_hat) r_hat - m) / |r|^3
(`induced_dipole` gives the moment of a magnetically soft body of volume V in the Earth field, m = k V B / mu0 with
k = 3 for a high-permeability sphere), residual hard / soft iron of the vehicle, and an optional field proportional to
the total thruster current.

AHRS: Mahony-type complementary filter at the IMU rate: gyro integration corrected towards the accelerometer's gravity
direction (gated when |a| differs from g) and towards the tilt-compensated magnetometer (horizontal component only),
with integral gyro-bias estimation.  Heading uses the configured declination D_cfg.  Errors therefore come out of the
physics: acceleration-induced tilt errors, slow gyro-bias transients, and heading errors near steel.

Outputs: attitude quaternion (w, x, y, z) of FLU body in ENU W, roll / pitch / yaw (FLU/ENU convention, see
_util.euler_zyx_from_mat), compass heading (deg, clockwise from north), bias-corrected angular rate and specific
force (body frame), raw magnetic field (body frame, uT).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np
import mujoco

from ._util import (RateScheduler, GaussMarkov, site_id, site_pose, site_velocity_world, mat_to_quat, quat_to_mat,
                    euler_zyx_from_mat, so3_exp, orthonormalize, SpecItem)

MU0_4PI = 1e-7
G0 = 9.80665


def earth_field_enu(total_uT: float = 50.0, incl_deg: float = 60.0, decl_deg: float = 0.0) -> np.ndarray:
    I, D = math.radians(incl_deg), math.radians(decl_deg)
    return total_uT * np.array([math.cos(I) * math.sin(D), math.cos(I) * math.cos(D), -math.sin(I)])


def induced_dipole(volume_m3: float, b_earth_uT: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Dipole moment [A m^2] induced in a soft-iron body of volume V by the Earth field (k = 3: sphere, mu >> 1)."""
    return k * volume_m3 * np.asarray(b_earth_uT) * 1e-6 / (4 * math.pi * 1e-7)


@dataclass
class MagneticWorld:
    """Magnetic environment of the site."""
    b_earth_uT: np.ndarray = field(default_factory=lambda: earth_field_enu(50.0, 60.0, 0.0))
    dipoles: list = field(default_factory=list)          # [(pos (3,), moment (3,) A m^2), ...]

    def field_uT(self, p: np.ndarray) -> np.ndarray:
        B = np.array(self.b_earth_uT, dtype=float)
        for pos, mom in self.dipoles:
            r = np.asarray(p, float) - np.asarray(pos, float)
            dist = float(np.linalg.norm(r))
            if dist < 0.05:
                dist = 0.05
            rh = r / dist
            mm = np.asarray(mom, float)
            B = B + MU0_4PI * (3.0 * (mm @ rh) * rh - mm) / dist ** 3 * 1e6
        return B


@dataclass
class AHRSConfig:
    imu_rate_hz: float = 100.0
    output_decimation: int = 2              # AHRS output at imu_rate / decimation
    declination_cfg_deg: float = 0.0        # firmware declination setting
    kp: float = 0.8
    ki: float = 0.01
    k_mag: float = 0.6
    accel_gate: float = 0.15                # fraction of g beyond which the accel correction is faded out


@dataclass
class AHRSHidden:
    gyro_noise_rad_s: float = 4.9e-4        # 0.004 deg/s/sqrt(Hz) at 100 Hz
    gyro_bias0_rad_s: tuple = (0.0, 0.0, 0.0)
    gyro_bias_rw: float = 2.0e-5            # rad/s/sqrt(s)
    accel_noise: float = 0.0104             # m/s^2 (150 ug/sqrt(Hz) at 100 Hz)
    accel_bias: tuple = (0.0, 0.0, 0.0)
    mag_noise_uT: float = 0.3
    hard_iron_uT: tuple = (0.0, 0.0, 0.0)
    soft_iron: tuple = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    thruster_current_uT_per_A: tuple = (0.0, 0.0, 0.0)

    @staticmethod
    def sample(rng: np.random.Generator) -> "AHRSHidden":
        S = np.eye(3) + rng.normal(0, 0.01, (3, 3))
        return AHRSHidden(gyro_bias0_rad_s=tuple(rng.normal(0, math.radians(0.05), 3).tolist()),
                          accel_bias=tuple(rng.normal(0, 0.03, 3).tolist()),
                          hard_iron_uT=tuple(rng.normal(0, 1.0, 3).tolist()),
                          soft_iron=tuple(map(tuple, ((S + S.T) / 2).tolist())),
                          thruster_current_uT_per_A=tuple(rng.normal(0, 0.05, 3).tolist()))


@dataclass
class AHRSReading:
    t: float
    valid: bool
    quat: np.ndarray                        # (w, x, y, z), FLU body -> ENU world (estimate)
    roll: float
    pitch: float
    yaw: float                              # rad, CCW from east (ENU)
    heading_deg: float                      # clockwise from (true, if declination_cfg is right) north
    angular_velocity: np.ndarray            # body, rad/s (bias corrected)
    specific_force: np.ndarray              # body, m/s^2 (includes +g up when static)
    mag_uT: np.ndarray                      # body, raw


def ahrs_public_spec(cfg: AHRSConfig) -> list[SpecItem]:
    return [
        SpecItem("type", "MEMS IMU (3-axis gyro, accelerometer) + 3-axis magnetometer, on-board AHRS filter"),
        SpecItem("IMU rate / AHRS output", f"{cfg.imu_rate_hz:.0f} / {cfg.imu_rate_hz / cfg.output_decimation:.0f}", "Hz"),
        SpecItem("roll / pitch accuracy (static)", "about 0.5", "deg"),
        SpecItem("roll / pitch accuracy (dynamic)", "about 1 - 2", "deg",
                 "during sustained accelerations (the filter trusts the accelerometer less above 0.15 g, then drifts "
                 "with the gyro); settles back within seconds"),
        SpecItem("heading", "magnetic, plus the configured declination; affected by nearby steel and motor currents"),
        SpecItem("gyro noise density", 0.004, "deg/s/sqrt(Hz)"),
        SpecItem("accelerometer noise density", 150, "ug/sqrt(Hz)"),
    ]


class AHRS:
    def __init__(self, m: mujoco.MjModel, rng: np.random.Generator, magworld: Optional[MagneticWorld] = None,
                 site: str = "imu", config: Optional[AHRSConfig] = None, hidden: Optional[AHRSHidden] = None,
                 t0: float = 0.0, thruster_current_fn: Optional[Callable[[], float]] = None):
        self.cfg = config or AHRSConfig()
        self.hid = hidden or AHRSHidden()
        self.rng = rng
        self.mag = magworld or MagneticWorld()
        self.sid = site_id(m, site)
        self.m = m
        self.sched = RateScheduler(self.cfg.imu_rate_hz, t0)
        self.gyro_bias = GaussMarkov(rng, 3, sigma_rw=self.hid.gyro_bias_rw, x0=np.array(self.hid.gyro_bias0_rad_s))
        self.thr_fn = thruster_current_fn
        self._prev_v: Optional[np.ndarray] = None
        self._prev_t: Optional[float] = None
        self.R_est: Optional[np.ndarray] = None
        self.b_est = np.zeros(3)
        self._k = 0
        self.last: Optional[AHRSReading] = None

    def _measure(self, t: float, d: mujoco.MjData):
        h = self.hid
        p, R = site_pose(d, self.sid)
        v_w, w_w = site_velocity_world(self.m, d, self.sid)
        if self._prev_v is None or t <= self._prev_t:
            a_w = np.zeros(3)
        else:
            a_w = (v_w - self._prev_v) / (t - self._prev_t)
        dt = 0.0 if self._prev_t is None else t - self._prev_t
        self._prev_v, self._prev_t = v_w, t
        w_b = R.T @ w_w
        bg = self.gyro_bias.step(dt)
        gyro = w_b + bg + h.gyro_noise_rad_s * self.rng.standard_normal(3)
        f_b = R.T @ (a_w - np.array([0.0, 0.0, -G0]))
        acc = f_b + np.asarray(h.accel_bias) + h.accel_noise * self.rng.standard_normal(3)
        B_w = self.mag.field_uT(p)
        B_b = np.asarray(h.soft_iron) @ (R.T @ B_w) + np.asarray(h.hard_iron_uT)
        if self.thr_fn is not None:
            B_b = B_b + np.asarray(h.thruster_current_uT_per_A) * float(self.thr_fn())
        mag = B_b + h.mag_noise_uT * self.rng.standard_normal(3)
        return gyro, acc, mag, R

    def _init_from(self, acc, mag):
        """Static alignment: body triad (east, north, up) from accel + horizontal mag, matched to the world triad
        built with the configured declination."""
        up_b = acc / np.linalg.norm(acc)
        h = mag - (mag @ up_b) * up_b
        north_b = h / np.linalg.norm(h)
        east_b = np.cross(north_b, up_b)
        Dc = math.radians(self.cfg.declination_cfg_deg)
        north_w = np.array([math.sin(Dc), math.cos(Dc), 0.0])
        up_w = np.array([0.0, 0.0, 1.0])
        east_w = np.cross(north_w, up_w)
        self.R_est = orthonormalize(np.stack([east_w, north_w, up_w], 1) @ np.stack([east_b, north_b, up_b], 1).T)

    def update(self, t: float, d: mujoco.MjData) -> list[AHRSReading]:
        out = []
        cfg = self.cfg
        while self.sched.due(t):
            tf = self.sched.fire()
            gyro, acc, mag, R_true = self._measure(tf, d)
            dt = self.sched.period
            if self.R_est is None:
                self._init_from(acc, mag)
            Rb = self.R_est
            # accelerometer correction (gravity direction), gated by |a| ~ g
            an = float(np.linalg.norm(acc))
            e = np.zeros(3)
            gate = max(0.0, 1.0 - abs(an - G0) / (cfg.accel_gate * G0))
            if an > 1e-6 and gate > 0:
                v_up = Rb.T @ np.array([0.0, 0.0, 1.0])
                e += gate * np.cross(acc / an, v_up)
            # magnetometer correction, horizontal only
            Dc = math.radians(cfg.declination_cfg_deg)
            hw = Rb @ mag
            hh = np.array([hw[0], hw[1], 0.0])
            if np.linalg.norm(hh) > 1e-6:
                ref = np.array([math.sin(Dc), math.cos(Dc), 0.0]) * np.linalg.norm(hh)
                ref_b = Rb.T @ ref
                meas_b = Rb.T @ hh
                e += cfg.k_mag * np.cross(meas_b, ref_b) / (np.linalg.norm(hh) ** 2)
            # Mahony: b_hat' = -ki e ; omega = gyro - b_hat + kp e
            self.b_est -= cfg.ki * e * dt
            w = gyro - self.b_est + cfg.kp * e
            self.R_est = orthonormalize(Rb @ so3_exp(w * dt))
            self._k += 1
            if self._k % cfg.output_decimation == 0:
                Re = self.R_est
                roll, pitch, yaw = euler_zyx_from_mat(Re)
                hdg = (90.0 - math.degrees(yaw)) % 360.0
                out.append(AHRSReading(t=tf, valid=True, quat=mat_to_quat(Re), roll=roll, pitch=pitch, yaw=yaw,
                                       heading_deg=hdg, angular_velocity=gyro - self.b_est, specific_force=acc,
                                       mag_uT=mag))
        if out:
            self.last = out[-1]
        return out
