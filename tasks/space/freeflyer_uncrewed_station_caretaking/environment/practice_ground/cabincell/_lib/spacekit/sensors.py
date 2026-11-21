"""On-board sensors of the free-flyer (no rendering here; see render.py for images).

Imu            gyro (inertial rate incl. the station's orbital rotation) and accelerometer (specific force:
               the frame's fictitious / tidal accelerations are removed), white noise + bias random walk.
Localizer      pose / velocity estimate = truth + correlated error whose size depends on a feature-
               richness map (mapped features in the forward camera's view); covariance output; a
               confidence flag (0 good / 1 poor / 2 lost) from log-det thresholds as the firmware's
               localiser does; the controller idles whenever confidence != 0.
DepthCamera    per-pixel ray casting with mj_multiRay (single origin), ToF-like noise, range gating.
camera helpers pinhole intrinsics <-> MuJoCo fovy, optical-frame <-> MuJoCo camera frame, projection,
               FOV ('w') and radial-tangential distortion models.
Battery        packs of Li-ion energy; impeller (propulsion.impeller_power), avionics, arm, payload loads.
JointSensors   arm joint positions / velocities with encoder quantisation and noise.
ContactSensor  contact flag and normal force between the gripper pads and anything else.
All randomness comes from an explicit numpy Generator.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from ._fast import cross
from .frames import body_com_velocity
from .rotations import mat_to_quat, quat_from_rotvec, quat_mul, quat_to_mat

__all__ = [
    "Imu", "FeaturePointMap", "GridRichnessMap", "Localizer", "DepthCamera", "fovy_from_intrinsics",
    "intrinsics_from_fovy", "optical_to_mujoco_camera", "mujoco_camera_to_optical", "camera_xml",
    "project_points", "distort_fov", "undistort_fov", "distort_radtan", "camera_model", "Battery", "JointSensors",
    "ContactSensor", "ContactEvents", "true_state",
]


def true_state(model, data, body: int):
    """(pos, quat (w,x,y,z), vel_world, omega_body) of a body's frame origin (truth, for graders)."""
    R = data.xmat[body].reshape(3, 3)
    w_world = data.cvel[body, 0:3]
    root = model.body_rootid[body]
    v_origin = data.cvel[body, 3:6] + cross(w_world, data.xpos[body] - data.subtree_com[root])
    return data.xpos[body].copy(), data.xquat[body].copy(), v_origin, R.T @ w_world


class Imu:
    """IMU at a site.  read(model, data) -> (gyro (3,), accel (3,)) in the site frame.

    frame: optional RotatingStationFrame; its rate is added to the gyro (a real gyro sees the station's
    orbital rotation) and its acceleration on the site body is removed from the accelerometer.
    Noise: densities (per sqrt(Hz)) and bias random walks, sampled at rate_hz.  Defaults are typical
    industrial MEMS values (gyro 0.1 deg/sqrt(h), accel 1 mg/sqrt(Hz)) -- an estimate, see the note.
    """

    def __init__(self, model, site: str, rng: np.random.Generator, rate_hz: float = 62.5, frame=None,
                 gyro_noise: float = 2.9e-5, gyro_bias_rw: float = 1e-6, gyro_bias0: float = 1e-4,
                 accel_noise: float = 1e-3, accel_bias_rw: float = 1e-4, accel_bias0: float = 2e-3):
        self.sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site)
        if self.sid < 0:
            raise ValueError(f"site {site} not found")
        self.body = int(model.site_bodyid[self.sid])
        self.rng = rng
        self.dt = 1.0 / rate_hz
        self.frame = frame
        self.gn, self.gbrw, self.an, self.abrw = gyro_noise, gyro_bias_rw, accel_noise, accel_bias_rw
        self.gbias = rng.normal(0.0, gyro_bias0, 3)
        self.abias = rng.normal(0.0, accel_bias0, 3)
        self._acc = np.zeros(6)

    def true(self, model, data):
        """Noise-free (gyro, specific force) in the site frame."""
        R = data.site_xmat[self.sid].reshape(3, 3)
        w = data.cvel[self.body, 0:3].copy()
        mujoco.mj_rnePostConstraint(model, data)
        mujoco.mj_objectAcceleration(model, data, mujoco.mjtObj.mjOBJ_SITE, self.sid, self._acc, 0)
        a = self._acc[3:6].copy()
        if self.frame is not None and getattr(self.frame, "enabled", True):
            w = w + self.frame.Om
            Wf = np.zeros((model.nbody, 6))
            self.frame.accumulate(model, data, Wf)
            m = model.body_subtreemass[model.body_rootid[self.body]]
            root = model.body_rootid[self.body]
            sub = np.nonzero(model.body_rootid == root)[0]
            a = a - Wf[sub, 0:3].sum(0) / m
        return R.T @ w, R.T @ a

    def read(self, model, data):
        """Noisy sample (call at rate_hz)."""
        g, a = self.true(model, data)
        sq = math.sqrt(self.dt)
        self.gbias += self.rng.normal(0.0, self.gbrw * sq, 3)
        self.abias += self.rng.normal(0.0, self.abrw * sq, 3)
        g = g + self.gbias + self.rng.normal(0.0, self.gn / sq, 3)
        a = a + self.abias + self.rng.normal(0.0, self.an / sq, 3)
        return g, a


class FeaturePointMap:
    """Feature richness = mapped feature points inside the forward camera's view cone / n_good (<= 1).

    points: (N, 3) world positions of mapped features; cam_axis_body: viewing axis in the body frame;
    fov_deg: full cone angle; range (min, max) [m].  occlusion (optional, slower): (model, data,
    exclude_body) to discard features hidden behind geometry with mj_ray.
    """

    def __init__(self, points, cam_axis_body=(1.0, 0.0, 0.0), fov_deg: float = 120.0, range_m=(0.3, 6.0),
                 n_good: int = 40):
        self.P = np.asarray(points, float).reshape(-1, 3)
        a = np.asarray(cam_axis_body, float)
        self.axis = a / np.linalg.norm(a)
        self.cos_half = math.cos(math.radians(fov_deg) / 2.0)
        self.rmin, self.rmax = range_m
        self.n_good = n_good

    def count(self, pos, R, occlusion=None) -> int:
        """Number of features in view from (pos, R body->world)."""
        if len(self.P) == 0:
            return 0
        d = self.P - np.asarray(pos, float)
        r = np.linalg.norm(d, axis=1)
        ax = np.asarray(R) @ self.axis
        with np.errstate(invalid="ignore", divide="ignore"):
            c = (d @ ax) / r
        vis = (r >= self.rmin) & (r <= self.rmax) & (c >= self.cos_half)
        if occlusion is not None and np.any(vis):
            model, data, excl = occlusion
            gid = np.zeros(1, np.int32)
            for i in np.nonzero(vis)[0]:
                hit = mujoco.mj_ray(model, data, np.asarray(pos, float), d[i] / r[i], None, 1, excl, gid)
                if 0 <= hit < r[i] - 0.02:
                    vis[i] = False
        return int(np.sum(vis))

    def __call__(self, pos, R, occlusion=None) -> float:
        return min(1.0, self.count(pos, R, occlusion) / float(self.n_good))


class GridRichnessMap:
    """Richness (0..1) on a regular grid, trilinear interpolation, clamped outside."""

    def __init__(self, origin, spacing, values):
        from .cabin import GridFlowField
        v = np.asarray(values, float)
        self._g = GridFlowField(origin, spacing, np.stack([v, v, v], -1))

    def __call__(self, pos, R=None, occlusion=None) -> float:
        return float(np.clip(self._g(np.asarray(pos, float)[None, :])[0, 0], 0.0, 1.0))


class Localizer:
    """Localisation output model (see module docstring).

    sigma_pos / sigma_att: stationary per-axis 1-sigma error with full richness (map-based mode); the
    defaults read the published map-based RMSE (0.0491 m, 0.0275 rad) as an RMSE of the error norm, so
    per axis it is divided by sqrt(3); drift_pos / drift_att: per-axis diffusion [m^2/s, rad^2/s] without
    features (visual-inertial dead reckoning; defaults make the time-averaged error-norm RMSE of a 600 s
    activity equal the published 0.2777 m / 0.0696 rad, q = 2 RMSE^2 / (3 * 600)); richness_min: below it the
    estimator is dead reckoning; tau_reloc: time constant of re-convergence when features return;
    vel_sigma / omega_sigma: velocity and rate estimate noise independent of the pose error.  Confidence
    thresholds as 1-sigma radii (pos_sigma_lost [m], att_sigma_lost [rad]) converted to log-det thresholds of the
    3x3 covariances.  The reported covariance (and so the confidence flag) follows the same book-keeping in both
    error models: relaxation with ``tau`` (``tau_reloc`` while re-converging) toward the richness-dependent level
    in map mode, growth by drift * dt in dead reckoning.

    error_model (v0.4):
      'consistent' (default) -- the estimate is a smooth trajectory and its velocity / rate are its own derivative
        (what an IMU-propagated estimator reports): per axis, the map-mode pose error is a critically damped
        second-order Gauss-Markov process (stationary std sigma / sqrt(richness)) in an error clock s that runs at
        ds/dt = 1 / corr_time + |v| / corr_length + |w| / corr_angle (the error is nearly constant while the robot
        holds a pose and decorrelates as the view changes; corr_length / corr_angle None drop the term; at least
        2 / tau_reloc while re-converging); stated in clock units the process is stationary whatever the motion, so
        a change of speed never pumps the error out of its range.  In dead reckoning the error integrates an
        Ornstein-Uhlenbeck velocity error of correlation time vio_tau whose long-run diffusion equals drift_pos /
        drift_att (the reported covariance grows as before).
        Estimated velocity = true velocity + d(position error)/dt + the independent noise (vel_sigma); estimated
        body rate likewise.  corr_time, corr_length, corr_angle and vio_tau are [G].
      'legacy' -- v0.3 (bit-identical): first-order Gauss-Markov pose error with correlation time ``tau`` and
        velocity / rate errors independent of it.  With the faithful firmware this made a hovering robot follow
        the fast estimate error (2-3 cm/s), the stopped logic (|v| < 2 cm/s, Butterworth-filtered) re-latch the
        hold point on the current estimate, and the hold point random-walk 0.2-0.7 m in 10 min.
    update(t, model, data, body) -> dict(pos, quat, vel, omega, cov_pos, cov_att, confidence, richness,
    pos_error, att_error) (errors are the true ones, for graders / tests only).
    """

    ERROR_MODELS = ("consistent", "legacy")

    def __init__(self, rng: np.random.Generator, richness=None, rate_hz: float = 62.5,
                 sigma_pos: float = 0.0491 / math.sqrt(3.0), sigma_att: float = 0.0275 / math.sqrt(3.0),
                 tau: float = 2.0, drift_pos: float = 2.0 * 0.2777 ** 2 / 1800.0,
                 drift_att: float = 2.0 * 0.0696 ** 2 / 1800.0,
                 richness_min: float = 0.2, tau_reloc: float = 5.0, vel_sigma: float = 0.003,
                 omega_sigma: float = 2e-4, pos_sigma_lost: float = 0.20, att_sigma_lost: float = math.radians(20.0),
                 error_model: str = "consistent", corr_time: float = 60.0, corr_length: float | None = 1.0,
                 corr_angle: float | None = 1.0, vio_tau: float = 10.0):
        if error_model not in self.ERROR_MODELS:
            raise ValueError(f"error_model {error_model!r} not in {self.ERROR_MODELS}")
        self.rng = rng
        self.richness = richness if richness is not None else (lambda pos, R, occlusion=None: 1.0)
        self.dt = 1.0 / rate_hz
        self.sp, self.sa, self.tau = sigma_pos, sigma_att, tau
        self.dp, self.da = drift_pos, drift_att
        self.rmin, self.tau_reloc = richness_min, tau_reloc
        self.vs, self.ws = vel_sigma, omega_sigma
        self.thr_pos = 3.0 * math.log(pos_sigma_lost ** 2)
        self.thr_att = 3.0 * math.log(att_sigma_lost ** 2)
        self.error_model = error_model
        self.corr_time, self.corr_length, self.corr_angle, self.vio_tau = corr_time, corr_length, corr_angle, vio_tau
        self.e_pos = rng.normal(0.0, sigma_pos, 3)
        self.e_att = rng.normal(0.0, sigma_att, 3)
        self.var_pos = sigma_pos ** 2
        self.var_att = sigma_att ** 2
        self.e_vel = np.zeros(3)
        self.e_om = np.zeros(3)
        # consistent model: ds_* = de/ds (error clock s), de_* = de/dt (what the velocity / rate estimates carry);
        # stationary start
        self.ds_pos = np.zeros(3)
        self.ds_att = np.zeros(3)
        self.de_pos = np.zeros(3)
        self.de_att = np.zeros(3)
        self._dr = False
        self.last_clock = None
        if error_model == "consistent":
            self.ds_pos = rng.normal(0.0, sigma_pos, 3)
            self.ds_att = rng.normal(0.0, sigma_att, 3)
            self.de_pos = self.ds_pos / corr_time
            self.de_att = self.ds_att / corr_time
        self.forced_lost = False

    @staticmethod
    def confidence_from_cov(cov_pos, cov_att, thr_pos: float, thr_att: float) -> int:
        """Firmware PoseConfidence: good if both log-dets below threshold, lost if both above, else poor."""
        lp = float(np.linalg.slogdet(cov_pos)[1])
        la = float(np.linalg.slogdet(cov_att)[1])
        gp, ga = lp < thr_pos, la < thr_att
        if gp and ga:
            return 0
        if not gp and not ga:
            return 2
        return 1

    def _propagate(self, r):
        dt = self.dt
        if r >= self.rmin:
            # map-based correction: Gauss-Markov toward a richness-dependent stationary error
            s_p = self.sp / math.sqrt(r)
            s_a = self.sa / math.sqrt(r)
            tau = self.tau if self.var_pos <= 4 * s_p ** 2 else self.tau_reloc
            a = math.exp(-dt / tau)
            q = math.sqrt(max(0.0, 1.0 - a * a))
            self.e_pos = a * self.e_pos + q * self.rng.normal(0.0, s_p, 3)
            self.e_att = a * self.e_att + q * self.rng.normal(0.0, s_a, 3)
            self.var_pos = a * a * self.var_pos + (1 - a * a) * s_p ** 2
            self.var_att = a * a * self.var_att + (1 - a * a) * s_a ** 2
        else:
            self.e_pos = self.e_pos + self.rng.normal(0.0, math.sqrt(self.dp * dt), 3)
            self.e_att = self.e_att + self.rng.normal(0.0, math.sqrt(self.da * dt), 3)
            self.var_pos += self.dp * dt
            self.var_att += self.da * dt
        a = math.exp(-dt / 1.0)
        q = math.sqrt(1 - a * a)
        self.e_vel = a * self.e_vel + q * self.rng.normal(0.0, self.vs, 3)
        self.e_om = a * self.e_om + q * self.rng.normal(0.0, self.ws, 3)

    @staticmethod
    def _gm2_step(h: float):
        """Exact one-step map over a step h of the error clock of a critically damped second-order Gauss-Markov state
        (e, de/ds) with unit natural frequency and unit stationary std: (Phi (2, 2), lower Cholesky factor (2, 2) of
        the step noise Q = I - Phi Phi^T; the stationary covariance in clock units is the identity)."""
        ex = math.exp(-h)
        Phi = ex * np.array([[1.0 + h, h], [-h, 1.0 - h]])
        Q = np.eye(2) - Phi @ Phi.T
        l11 = math.sqrt(max(Q[0, 0], 0.0))
        l21 = Q[1, 0] / l11 if l11 > 0 else 0.0
        l22 = math.sqrt(max(Q[1, 1] - l21 * l21, 0.0))
        return Phi, np.array([[l11, 0.0], [l21, l22]])

    def clock_rate(self, vel, om) -> float:
        """ds/dt of the error clock (consistent model): 1 / corr_time + |v| / corr_length + |w| / corr_angle [1/s]."""
        w0 = 1.0 / self.corr_time
        if self.corr_length:
            w0 += float(np.linalg.norm(vel)) / self.corr_length
        if self.corr_angle:
            w0 += float(np.linalg.norm(om)) / self.corr_angle
        return w0

    def _propagate_consistent(self, r, vel, om):
        dt = self.dt
        if r >= self.rmin:
            s_p = self.sp / math.sqrt(r)
            s_a = self.sa / math.sqrt(r)
            reloc = self.var_pos > 4 * s_p ** 2
            tau = self.tau_reloc if reloc else self.tau
            sd = self.clock_rate(vel, om)
            if reloc:
                sd = max(sd, 2.0 / self.tau_reloc)
            self.last_clock = sd
            # the error is the reported std g (per axis) times a unit-variance state z that runs in the error clock,
            # so the true error always matches the reported covariance (also while it changes with the richness or
            # re-converges after dead reckoning); z is taken from the current error (an error set from outside is
            # kept)
            g0 = np.repeat([math.sqrt(self.var_pos), math.sqrt(self.var_att)], 3)    # reported std before the step
            z = np.empty((2, 6))
            z[0] = np.concatenate([self.e_pos, self.e_att]) / g0
            if self._dr:
                # back from dead reckoning: the velocity error carried over, bounded to 3 in clock units
                z[1] = np.clip(np.concatenate([self.de_pos, self.de_att]) / (sd * g0), -3.0, 3.0)
                self._dr = False
            else:
                z[1] = np.concatenate([self.ds_pos, self.ds_att]) / g0
            a = math.exp(-dt / tau)                                                   # v0.3 bookkeeping
            self.var_pos = a * a * self.var_pos + (1 - a * a) * s_p ** 2
            self.var_att = a * a * self.var_att + (1 - a * a) * s_a ** 2
            g1 = np.repeat([math.sqrt(self.var_pos), math.sqrt(self.var_att)], 3)
            Phi, Lc = self._gm2_step(sd * dt)
            n = self.rng.normal(0.0, 1.0, (2, 6))
            z = Phi @ z + Lc @ n
            e, ds = g1 * z[0], g1 * z[1]
            de = ds * sd + z[0] * (g1 - g0) / dt                                      # d(g z)/dt
            self.e_pos, self.e_att = e[0:3], e[3:6]
            self.ds_pos, self.ds_att = ds[0:3], ds[3:6]
            self.de_pos, self.de_att = de[0:3], de[3:6]
        else:
            # dead reckoning: the error integrates an OU velocity error whose long-run diffusion is drift
            self.last_clock = None
            a = math.exp(-dt / self.vio_tau)
            q = math.sqrt(max(0.0, 1.0 - a * a))
            su_p = math.sqrt(self.dp / (2.0 * self.vio_tau))
            su_a = math.sqrt(self.da / (2.0 * self.vio_tau))
            if not self._dr:
                # entering dead reckoning: the odometry's velocity error starts from its stationary distribution (the
                # mapped error's rate of change is a map-correction rate, not an odometry velocity error; carrying
                # it over integrated it into a drift the reported covariance does not have)
                n0 = self.rng.normal(0.0, 1.0, (2, 3))
                self.de_pos, self.de_att = su_p * n0[0], su_a * n0[1]
                self._dr = True
            n = self.rng.normal(0.0, 1.0, (2, 3))
            u_p = a * self.de_pos + q * su_p * n[0]
            u_a = a * self.de_att + q * su_a * n[1]
            self.e_pos = self.e_pos + 0.5 * dt * (self.de_pos + u_p)
            self.e_att = self.e_att + 0.5 * dt * (self.de_att + u_a)
            self.de_pos, self.de_att = u_p, u_a
            self.var_pos += self.dp * dt
            self.var_att += self.da * dt
        a = math.exp(-dt / 1.0)
        q = math.sqrt(1 - a * a)
        self.e_vel = a * self.e_vel + q * self.rng.normal(0.0, self.vs, 3)
        self.e_om = a * self.e_om + q * self.rng.normal(0.0, self.ws, 3)

    def update(self, t, model, data, body: int, occlusion=None) -> dict:
        """One estimator tick for the body (call at rate_hz)."""
        pos, quat, vel, om = true_state(model, data, body)
        R = quat_to_mat(quat)
        r = float(self.richness(pos, R, occlusion))
        if self.error_model == "legacy":
            self._propagate(r)
            v_err, w_err = self.e_vel, self.e_om
        else:
            self._propagate_consistent(r, vel, om)
            # body rate of q_est = q (x) exp(e_att): w + w x e_att + d(e_att)/dt (small-angle)
            v_err, w_err = self.de_pos + self.e_vel, self.de_att + cross(om, self.e_att) + self.e_om
        q_est = quat_mul(quat, quat_from_rotvec(self.e_att))
        cov_pos = np.eye(3) * self.var_pos
        cov_att = np.eye(3) * self.var_att
        conf = 2 if self.forced_lost else self.confidence_from_cov(cov_pos, cov_att, self.thr_pos, self.thr_att)
        return dict(pos=pos + self.e_pos, quat=q_est, vel=vel + v_err, omega=om + w_err, cov_pos=cov_pos,
                    cov_att=cov_att, confidence=conf, richness=r, pos_error=float(np.linalg.norm(self.e_pos)),
                    att_error=float(np.linalg.norm(self.e_att)))


# ------------------------------------------------------------------------------------------ cameras
def fovy_from_intrinsics(fy: float, height: int) -> float:
    """MuJoCo camera fovy [deg] of a pinhole camera with focal length fy [px] and image height [px]."""
    return math.degrees(2.0 * math.atan(height / (2.0 * fy)))


def intrinsics_from_fovy(fovy_deg: float, width: int, height: int) -> np.ndarray:
    """3x3 K of a MuJoCo camera (square pixels, principal point at the image centre)."""
    f = (height / 2.0) / math.tan(math.radians(fovy_deg) / 2.0)
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


_OPT2MJ = np.diag([1.0, -1.0, -1.0])


def optical_to_mujoco_camera(R_parent_optical) -> np.ndarray:
    """Rotation parent <- MuJoCo camera frame (looks along -z, y up) from parent <- optical frame
    (x right, y down, z forward)."""
    return np.asarray(R_parent_optical, float) @ _OPT2MJ


def mujoco_camera_to_optical(R_parent_mjcam) -> np.ndarray:
    """Inverse of optical_to_mujoco_camera."""
    return np.asarray(R_parent_mjcam, float) @ _OPT2MJ


def camera_xml(name: str, pos, quat_parent_optical, fy: float, height: int) -> str:
    """MJCF <camera> element for an optical-frame pose (quat (w,x,y,z), parent <- optical) and intrinsics."""
    q = mat_to_quat(optical_to_mujoco_camera(quat_to_mat(quat_parent_optical)))
    p = " ".join(f"{v:.6g}" for v in pos)
    qs = " ".join(f"{v:.9g}" for v in q)
    return f'<camera name="{name}" pos="{p}" quat="{qs}" fovy="{fovy_from_intrinsics(fy, height):.6g}"/>'


def project_points(K, R_world_optical, t_world_optical, points):
    """Pinhole projection of world points (N, 3): returns (uv (N, 2), depth (N,)) in the optical frame."""
    P = (np.asarray(points, float) - np.asarray(t_world_optical, float)) @ np.asarray(R_world_optical, float)
    z = P[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([K[0, 0] * P[:, 0] / z + K[0, 2], K[1, 1] * P[:, 1] / z + K[1, 2]], 1)
    return uv, z


def distort_fov(xn, w: float):
    """FOV (Devernay-Faugeras) distortion of normalised image points (N, 2): r_d = atan(2 r tan(w/2)) / w."""
    xn = np.atleast_2d(np.asarray(xn, float))
    r = np.linalg.norm(xn, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(r > 1e-12, np.arctan(2.0 * r * math.tan(w / 2.0)) / (w * r), 1.0)
    return xn * s[:, None]


def undistort_fov(xd, w: float):
    """Inverse FOV model: r_u = tan(r_d w) / (2 tan(w/2))."""
    xd = np.atleast_2d(np.asarray(xd, float))
    r = np.linalg.norm(xd, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(r > 1e-12, np.tan(r * w) / (2.0 * math.tan(w / 2.0) * r), 1.0)
    return xd * s[:, None]


def distort_radtan(xn, k1: float, k2: float, p1: float, p2: float):
    """Radial-tangential (Brown-Conrady, 4 coefficients) distortion of normalised points (N, 2)."""
    xn = np.atleast_2d(np.asarray(xn, float))
    x, y = xn[:, 0], xn[:, 1]
    r2 = x * x + y * y
    rad = 1.0 + k1 * r2 + k2 * r2 * r2
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([xd, yd], 1)


def camera_model(role: str, ref: dict | None = None) -> dict:
    """Configured camera of the reference free-flyer (roles: forward_wide, aft_wide, forward_depth, aft_depth,
    forward_imaging): dict(K (3,3), distortion, distortion_model ('fov' | 'radtan'), R_body_optical (3,3),
    t_body (3,), q_body_optical (w,x,y,z), placeholder_in_source, width, height).  Image size = 2 * (cx, cy)
    rounded (the configuration gives no size)."""
    if ref is None:
        from .propulsion import load_firmware_reference
        ref = load_firmware_reference()
    c = ref["cameras"][role]
    K = np.asarray(c["K"], float)
    q = np.asarray(c["q_body_optical_wxyz"], float)
    return dict(K=K, distortion=list(c["distortion"]), distortion_model=c["distortion_model"],
                R_body_optical=quat_to_mat(q), t_body=np.asarray(c["t_body_m"], float), q_body_optical=q,
                placeholder_in_source=bool(c["placeholder_in_source"]),
                width=int(round(2 * K[0, 2])), height=int(round(2 * K[1, 2])))


class DepthCamera:
    """Ray-cast depth camera at a MuJoCo site or camera (optical axis = MuJoCo camera -z; for a site, +x).

    width, height, fx, fy, cx, cy: intrinsics of the depth image; stride: subsample pixels;
    range_m: (min, max) valid range; noise: (a [m], b [1/m]) for sigma = a + b d^2; exclude_body: body
    not seen by the rays (the robot itself); exclude_bodies: further bodies not seen by the rays (v0.3: the
    robot's whole rigid structure -- on the reference build v1 the forward depth camera sits inside the forward
    sensor module's collision proxy, a child body of the core, so excluding the core alone blinds it).  The
    rays exclude the camera's mount body when it is in the set (else ``exclude_body``) and continue past any
    other excluded geom they meet first.  read() -> z-depth image [m], NaN where invalid.
    """

    def __init__(self, model, name: str, width: int, height: int, fx: float, fy: float, cx: float, cy: float,
                 kind: str = "camera", stride: int = 1, range_m=(0.1, 4.0), noise=(0.002, 0.002),
                 exclude_body: int = -1, rng: np.random.Generator | None = None, geomgroup=None, exclude_bodies=None):
        obj = mujoco.mjtObj.mjOBJ_CAMERA if kind == "camera" else mujoco.mjtObj.mjOBJ_SITE
        self.id = mujoco.mj_name2id(model, obj, name)
        if self.id < 0:
            raise ValueError(f"{kind} {name} not found")
        self.kind = kind
        excl = np.zeros(model.nbody, bool)
        if exclude_bodies is not None:
            excl[np.asarray(list(exclude_bodies), int)] = True
        if exclude_body >= 0:
            excl[exclude_body] = True
        mount = int(model.cam_bodyid[self.id] if kind == "camera" else model.site_bodyid[self.id])
        self._ray_excl = mount if excl[mount] else exclude_body       # the one body mj_multiRay can skip
        self._geom_excl = excl[model.geom_bodyid] if excl.any() else None   # per geom: part of the excluded set
        self.n_continued = 0                                          # rays continued past excluded geoms (last read)
        us = np.arange(0, width, stride) + 0.5
        vs = np.arange(0, height, stride) + 0.5
        U, V = np.meshgrid(us, vs)
        x = (U - cx) / fx
        y = (V - cy) / fy
        d = np.stack([x, y, np.ones_like(x)], -1).reshape(-1, 3)        # optical frame (z forward)
        self.dz = 1.0 / np.linalg.norm(d, axis=1)
        self.dirs_opt = d * self.dz[:, None]
        self.shape = U.shape
        self.range, self.noise, self.excl = range_m, noise, exclude_body
        self.rng = rng
        self.geomgroup = None if geomgroup is None else np.asarray(geomgroup, np.uint8)
        n = len(self.dirs_opt)
        self._gid = np.zeros(n, np.int32)
        self._dist = np.zeros(n)

    def _pose(self, data):
        if self.kind == "camera":
            return data.cam_xpos[self.id].copy(), mujoco_camera_to_optical(data.cam_xmat[self.id].reshape(3, 3))
        R = data.site_xmat[self.id].reshape(3, 3)
        # site convention: +x forward, +y left, +z up  ->  optical x = -y_site, y = -z_site, z = x_site
        R_site_opt = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        return data.site_xpos[self.id].copy(), R @ R_site_opt

    def read(self, model, data) -> np.ndarray:
        """Depth image (z along the optical axis) [m], NaN outside the valid range or no hit."""
        p, R = self._pose(data)
        dirs = self.dirs_opt @ R.T
        mujoco.mj_multiRay(model, data, p, dirs.reshape(-1).copy(), self.geomgroup, 1, self._ray_excl, self._gid,
                           self._dist, None, len(dirs), self.range[1] + 1.0)
        dist = self._dist.copy()
        self.n_continued = 0
        if self._geom_excl is not None:
            again = np.nonzero((self._gid >= 0) & self._geom_excl[np.maximum(self._gid, 0)])[0]
            self.n_continued = len(again)
            g1 = np.zeros(1, np.int32)
            for i in again:                       # first hit on an excluded body: continue the ray past it
                v, s = dirs[i], float(dist[i])
                for _ in range(8):
                    s += 1e-5
                    s2 = mujoco.mj_ray(model, data, p + v * s, v, self.geomgroup, 1, self._ray_excl, g1)
                    if s2 < 0:
                        s = -1.0
                        break
                    s += s2
                    if not self._geom_excl[int(g1[0])]:
                        break
                else:
                    s = -1.0
                dist[i] = s
        z = dist * self.dz
        bad = (dist < 0) | (z < self.range[0]) | (z > self.range[1])
        if self.rng is not None:
            sig = self.noise[0] + self.noise[1] * z * z
            z = z + self.rng.normal(0.0, 1.0, z.shape) * sig
        z[bad] = np.nan
        return z.reshape(self.shape)


# ------------------------------------------------------------------------------------------ power
class Battery:
    """Battery packs (energy book-keeping) of the free-flyer.

    n_packs x capacity_wh at nominal voltage; avionics_w: always-on load (computers, boards, lights);
    other_w: further always-on loads; arm_efficiency: mechanical -> electrical for the arm joints;
    arm_idle_w: arm electronics when powered.  update(dt, impeller_w, arm_mech_w, payload_w) integrates.
    voltage(): 4S Li-ion open-circuit curve (12.0 V empty .. 16.8 V full, piecewise linear) -- estimate.
    """

    def __init__(self, n_packs: int = 4, capacity_wh: float = 49.0, soc0: float = 1.0, avionics_w: float = 35.2,
                 other_w: float = 6.0, arm_efficiency: float = 0.5, arm_idle_w: float = 1.0, arm_powered: bool = True):
        self.n, self.cap = n_packs, capacity_wh
        self.energy_wh = n_packs * capacity_wh * soc0
        self.avionics_w, self.other_w = avionics_w, other_w
        self.arm_eff, self.arm_idle_w, self.arm_powered = arm_efficiency, arm_idle_w, arm_powered
        self.used = dict(impellers=0.0, avionics=0.0, arm=0.0, payload=0.0)
        self.last_power_w = 0.0

    @property
    def soc(self) -> float:
        """State of charge 0..1."""
        return max(0.0, self.energy_wh / (self.n * self.cap))

    def voltage(self) -> float:
        """Open-circuit pack voltage [V] at the current state of charge."""
        s = self.soc
        pts = [(0.0, 12.0), (0.1, 13.6), (0.5, 14.6), (0.9, 16.0), (1.0, 16.8)]
        for (s0, v0), (s1, v1) in zip(pts[:-1], pts[1:]):
            if s <= s1:
                return v0 + (v1 - v0) * (s - s0) / (s1 - s0)
        return pts[-1][1]

    def update(self, dt: float, impeller_w: float = 0.0, arm_mech_w: float = 0.0, payload_w: float = 0.0) -> float:
        """Integrate dt seconds of load; returns the total power [W]."""
        arm_w = (self.arm_idle_w + max(arm_mech_w, 0.0) / self.arm_eff) if self.arm_powered else 0.0
        base = self.avionics_w + self.other_w
        p = impeller_w + base + arm_w + payload_w
        h = dt / 3600.0
        self.used["impellers"] += impeller_w * h
        self.used["avionics"] += base * h
        self.used["arm"] += arm_w * h
        self.used["payload"] += payload_w * h
        self.energy_wh -= p * h
        self.last_power_w = p
        return p

    def endurance_h(self, power_w: float) -> float:
        """Hours left at a constant load."""
        return self.energy_wh / power_w if power_w > 0 else math.inf


class JointSensors:
    """Encoders of named joints: read(data) -> (positions, velocities) with quantisation and noise."""

    def __init__(self, model, joints, rng: np.random.Generator | None = None, resolution: float = 2 * math.pi / 4096,
                 pos_noise: float = 0.0, vel_noise: float = 0.002):
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joints]
        if min(ids) < 0:
            raise ValueError(f"joint not found in {joints}")
        self.qadr = np.array([model.jnt_qposadr[i] for i in ids])
        self.dadr = np.array([model.jnt_dofadr[i] for i in ids])
        self.rng, self.res, self.pn, self.vn = rng, resolution, pos_noise, vel_noise

    def read(self, data):
        """(positions [rad or m], velocities)."""
        q = data.qpos[self.qadr].copy()
        v = data.qvel[self.dadr].copy()
        if self.res:
            q = np.round(q / self.res) * self.res
        if self.rng is not None:
            q = q + self.rng.normal(0.0, self.pn, q.shape) if self.pn else q
            v = v + self.rng.normal(0.0, self.vn, v.shape) if self.vn else v
        return q, v


class ContactSensor:
    """Contact between a set of geoms (e.g. gripper pads) and anything else.

    read(model, data) -> dict(in_contact, normal_force [N] (sum), n_contacts, partners (geom ids)).
    threshold: normal force below which no contact is reported.
    """

    def __init__(self, model, geoms, threshold: float = 0.05):
        ids = [g if isinstance(g, (int, np.integer)) else mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g)
               for g in geoms]
        if min(ids) < 0:
            raise ValueError(f"geom not found in {geoms}")
        self.geoms = set(int(i) for i in ids)
        self.threshold = threshold
        self._f = np.zeros(6)

    def read(self, model, data) -> dict:
        """Current contact summary."""
        total, n, partners = 0.0, 0, []
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 in self.geoms) == (g2 in self.geoms):
                continue
            mujoco.mj_contactForce(model, data, i, self._f)
            total += abs(float(self._f[0]))
            n += 1
            partners.append(g2 if g1 in self.geoms else g1)
        return dict(in_contact=total >= self.threshold, normal_force=total, n_contacts=n, partners=sorted(set(partners)))


class ContactEvents:
    """Per-physics-step contact events between two body groups (v0.4; e.g. the robot and the loose items, for graders
    and logs -- the robot program does not see them).

    bodies_a / bodies_b: body ids (an id in both is treated as group a).  ``attach(sim)`` registers ``update`` as a
    periodic task of every physics step of a ``stepping.Simulator`` (it runs before the step, so it sees the contacts
    and constraint forces of the step just taken).  An event opens on the first step with a contact between a body of
    group a and one of group b whose normal force is >= min_force, and closes on the first step without one.
    ``events``: dicts (a, b, t0, t1 (None while open), steps, peak_N, impulse_Ns, speed_ms -- relative speed along the
    normal at the onset); ``active`` {(a, b): event}; ``touching(b)``; ``reset()``.  Reads only."""

    def __init__(self, model, bodies_a, bodies_b, min_force: float = 0.0):
        self.m = model
        self.A = set(int(b) for b in bodies_a)
        self.B = set(int(b) for b in bodies_b) - self.A
        self.min_force = float(min_force)
        self._gA = np.isin(model.geom_bodyid, list(self.A))
        self._gB = np.isin(model.geom_bodyid, list(self.B))
        self._f = np.zeros(6)
        self._v = np.zeros(6)
        self.reset()

    def reset(self) -> None:
        self.events = []
        self.active = {}

    def attach(self, sim, name: str = "contact_events") -> None:
        sim.add_periodic(lambda s: self.update(s.data), self.m.opt.timestep, name)

    def touching(self, body: int) -> bool:
        return any(k[1] == int(body) for k in self.active)

    def update(self, data) -> None:
        m = self.m
        seen = {}
        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self._gA[g1] and self._gB[g2]:
                a, b, sgn = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2]), 1.0
            elif self._gA[g2] and self._gB[g1]:
                a, b, sgn = int(m.geom_bodyid[g2]), int(m.geom_bodyid[g1]), -1.0
            else:
                continue
            mujoco.mj_contactForce(m, data, i, self._f)
            fn = abs(float(self._f[0]))
            if fn < self.min_force:
                continue
            key = (a, b)
            if key not in seen:
                n = np.asarray(c.frame[0:3], float) * sgn                    # normal from a toward b
                mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, b, self._v, 0)
                vb = self._v[3:6].copy()
                mujoco.mj_objectVelocity(m, data, mujoco.mjtObj.mjOBJ_BODY, a, self._v, 0)
                seen[key] = [0.0, float(np.dot(self._v[3:6] - vb, n))]
            seen[key][0] += fn
        t = float(data.time)
        dt = float(m.opt.timestep)
        for key, (fn, vn) in seen.items():
            ev = self.active.get(key)
            if ev is None:
                ev = dict(a=key[0], b=key[1], t0=t, t1=None, steps=0, peak_N=0.0, impulse_Ns=0.0, speed_ms=vn)
                self.active[key] = ev
                self.events.append(ev)
            ev["steps"] += 1
            ev["peak_N"] = max(ev["peak_N"], fn)
            ev["impulse_Ns"] += fn * dt
        for key in [k for k in self.active if k not in seen]:
            self.active.pop(key)["t1"] = t
