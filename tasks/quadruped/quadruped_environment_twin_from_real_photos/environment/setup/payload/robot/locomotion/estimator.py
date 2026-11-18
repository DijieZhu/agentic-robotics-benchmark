"""Onboard state estimate from the IMU (AHRS orientation, gyro, accelerometer), the joint encoders and the foot
switches only. Nothing here reads the simulator's base pose.

  attitude      the AHRS quaternion (its heading drifts slowly)
  velocity      leg odometry: each foot in stance is assumed fixed on the ground, so the base velocity is
                -(J qdot + omega x p_foot) rotated into the odometry frame; averaged over the stance feet and fused
                with the accelerometer (complementary filter)
  position      integral of the velocity (drifts; it is an odometry, not a localisation)
  ground        the last touchdown point of every foot (odometry frame) -> least-squares plane z = a x + b y + c;
                gives the body height above the local ground and the slope under the robot
"""

from __future__ import annotations

import math

import numpy as np

from .fastmath import cross3, cross_rows

G = 9.81


def quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_rpy(R):
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = math.asin(max(-1.0, min(1.0, -R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class StateEstimator:
    def __init__(self, kin, dt: float, foot_radius: float = 0.017, vel_tc: float = 0.033, height_tc: float = 0.005,
                 robust_plane: bool = True):
        self.kin, self.dt, self.r_foot = kin, dt, foot_radius
        self.robust_plane = robust_plane
        self.vel_blend = dt / (vel_tc + dt)          # complementary filter: leg odometry vs accelerometer
        self.height_blend = dt / (height_tc + dt)
        self.reset(np.array([0.0, 0.0, 0.24]), 0.0, None)

    def reset(self, pos, yaw, meas=None, height=None):
        self.p = np.array(pos, float)
        self.v = np.zeros(3)
        self.yaw0_offset = 0.0
        self.R = Rz(yaw)
        self.rpy = (0.0, 0.0, yaw)
        self.omega_b = np.zeros(3)
        self.q = np.zeros((4, 3))
        self.dq = np.zeros((4, 3))
        self.foot_b = np.zeros((4, 3))
        self.J = np.zeros((4, 3, 3))
        self.contact = np.zeros(4, bool)
        self.stance_time = np.zeros(4)
        self.td_points = None            # (4, 3) last touchdown point (foot-sphere centre) per foot, odometry frame
        self.plane = np.array([0.0, 0.0, 0.0])   # z = a x + b y + c of the foot-centre plane
        self.height = 0.24 if height is None else height
        self.distance = 0.0
        self.v_legs = np.zeros(3)
        if meas is not None:
            self._attitude(meas)
            self._joints(meas)
            # place the odometry origin so that the given pose holds and the ground plane is under the feet
            fw = self.p[:, None] + self.R @ self.foot_b.T
            self.td_points = fw.T.copy()
            self.foot_w = fw.T.copy()
            self._fit_plane()
            self.height = self.p[2] - self.plane_z(self.p[0], self.p[1]) + self.r_foot

    def _attitude(self, meas):
        R_imu = quat_to_R(meas["quat"])
        self.R = R_imu
        self.rpy = R_to_rpy(R_imu)
        self.omega_b = meas["gyro"]

    def _joints(self, meas):
        self.q = meas["q"].reshape(4, 3)
        self.dq = meas["dq"].reshape(4, 3)
        self.foot_b, self.J = self.kin.fk(self.q)

    def plane_z(self, x, y):
        a, b, c = self.plane
        return a * x + b * y + c

    def _fit_plane(self):
        """Least-squares plane through the four touchdown points, with one reweighting pass so that a single foot
        resting on an obstacle (a pipe, a sill) does not tilt the plane: points off the fit by more than 15 mm get
        a weight 0.05. A small ridge term keeps the slope defined when the points are nearly collinear."""
        P = self.td_points
        A = np.c_[P[:, 0], P[:, 1], np.ones(len(P))]
        z = P[:, 2]
        reg = np.diag([1e-4, 1e-4, 0.0])
        w = np.ones(len(P))
        for _ in range(2 if self.robust_plane else 1):
            Aw = A * w[:, None]
            sol = np.linalg.solve(A.T @ Aw + reg, Aw.T @ z)
            res = np.abs(A @ sol - z)
            w = np.where(res > 0.015, 0.05, 1.0)
            if w.min() == 1.0:
                break
        self.plane = sol

    def slope_rp(self):
        """Roll / pitch of the ground plane seen in the heading frame (radians)."""
        a, b, _ = self.plane
        yaw = self.rpy[2]
        c, s = math.cos(yaw), math.sin(yaw)
        gx = c * a + s * b        # slope along the heading
        gy = -s * a + c * b       # slope to the left
        return math.atan(gy), -math.atan(gx)

    def update(self, meas, stance: np.ndarray, touchdown: np.ndarray, acc_ok: bool = True):
        """stance: feet the controller treats as being on the ground; touchdown: feet that just touched down."""
        dt = self.dt
        self._attitude(meas)
        self._joints(meas)
        R, w = self.R, self.omega_b
        # leg odometry
        self.stance_time = np.where(stance, self.stance_time + dt, 0.0)
        use = stance & (self.stance_time > 0.008)
        vel, wts = [], []
        if use.any():
            ii = np.nonzero(use)[0]
            Jdq = np.einsum("nij,nj->ni", self.J[ii], self.dq[ii])
            v_b = -(Jdq + cross_rows(self.foot_b[ii], -w))       # -(J dq + w x p)
            # the foot sphere rolls: its centre moves by omega_shank x (r up) over the fixed contact point
            q1 = self.q[ii, 0]
            rate = self.dq[ii, 1] + self.dq[ii, 2]
            w_sh = np.empty((len(ii), 3))
            w_sh[:, 0] = w[0] + self.dq[ii, 0]
            w_sh[:, 1] = w[1] + rate * np.cos(q1)
            w_sh[:, 2] = w[2] + rate * np.sin(q1)
            w_sh_w = w_sh @ R.T
            v_roll = np.stack([w_sh_w[:, 1] * self.r_foot, -w_sh_w[:, 0] * self.r_foot, np.zeros(len(ii))], 1)
            vel = v_b @ R.T + v_roll
            wts = np.minimum(1.0, self.stance_time[ii] / 0.03)
        a_w = R @ meas["acc"] + np.array([0.0, 0.0, -G])
        v_pred = self.v + (a_w * dt if acc_ok else 0.0)
        if len(vel):
            v_leg = (vel * wts[:, None]).sum(0) / wts.sum()
            self.v_legs = v_leg
            k = self.vel_blend
            self.v = v_pred + k * (v_leg - v_pred)
        else:
            self.v = v_pred
        self.p = self.p + self.v * dt
        self.distance += float(np.linalg.norm(self.v[:2])) * dt
        # ground plane from the touchdown points
        fw = self.p[:, None] + R @ self.foot_b.T
        fw = fw.T
        if self.td_points is None:
            self.td_points = fw.copy()
        self._nfit = getattr(self, "_nfit", 0) + 1
        refit = touchdown.any() or self._nfit >= 10
        if touchdown.any():
            self.td_points[touchdown] = fw[touchdown]
        # stance feet track their current estimate (keeps the plane consistent with the drifting odometry)
        tr = stance & (self.stance_time > 0.02)
        if tr.any():
            self.td_points[tr] = 0.9 * self.td_points[tr] + 0.1 * fw[tr]
        if refit:
            self._nfit = 0
            self._fit_plane()
        h_new = self.p[2] - self.plane_z(self.p[0], self.p[1]) + self.r_foot
        self.height += self.height_blend * (h_new - self.height)
        self.contact = meas["contact"].copy()
        self.foot_w = fw

    def snapshot(self):
        return dict(pos=self.p.copy(), vel=self.v.copy(), rpy=tuple(self.rpy), R=self.R.copy(),
                    omega_body=self.omega_b.copy(), height=self.height, plane=self.plane.copy(),
                    distance=self.distance)
