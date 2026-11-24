"""rigsim: deterministic generator of multi-sensor robot logs for rig calibration (version 3).

One *rig* is a wheeled robot (a Clearpath Husky A200 with a sensor arch) carrying a spinning LiDAR (the rig's
reference frame; a Velodyne VLP-16), an RGB camera (a FLIR Blackfly S BFS-U3-63S4C: rolling shutter, binned 4x4, 6 mm
lens, re-focused: its intrinsics are not the nominal ones) and an IMU (a Bosch BMI088), driving through a warehouse
hall with props, vehicles that drive and people who walk. Some rigs have no wheel odometry; on some the camera
bracket is knocked once during the session. A rig has a set of true parameters (camera and IMU
extrinsics, sensor clock offsets and drifts, IMU biases), a CAD prior that is off by a few degrees and centimetres,
several recorded sequences that may contain injected faults, and one held-out sequence that the grader keeps to
itself.

Everything is a pure function of the rig seed (a list of integers fed to `numpy.random.SeedSequence`) and of the
site's asset files. The logs are generated ONCE, offline (gen_rigs.py: LiDAR by ray casting on the site's triangles,
camera frames rasterised from the same asset files); nothing here runs at verify.

Frames and conventions (also stated in instruction.md):

* world W: z up, gravity (0, 0, -9.80665) m/s^2, floor z = 0.
* body B: x forward, y left, z up, origin on the floor between the wheels.
* LiDAR L: x forward, y left, z up. T_body_lidar is known exactly (manifest.json).
* camera C: OpenCV optical frame (x right, y down, z forward), pinhole with Brown-Conrady distortion.
* IMU I: an arbitrary orientation on the board; the CAD prior says roughly which.
* T_a_b is the pose of frame b in frame a: p_a = R_a_b p_b + t_a_b.
* reference clock: the base controller's clock, which stamps the IMU and the wheel odometry.
  The camera and the LiDAR run free on their own clocks: a sensor with clock offset o and drift d reads
  s = t + o + d * t at reference time t (d is dimensionless; ppm = 1e-6). Its frames are evenly spaced on ITS
  clock (plus a trigger jitter), so stamp spacing carries no drift.
"""

import hashlib
import json
import math
import os

import numpy as np

G = 9.80665
GRAVITY_W = np.array([0.0, 0.0, -G])

IMG_W, IMG_H = 768, 512
F_NOMINAL = 625.0            # 6 mm lens on 9.6 um binned pixels
RENDER_SS = 1.5               # the pinhole source image is rendered at this multiple of the focal length
JPEG_QUALITY = 92
LIDAR_RINGS = 16
LIDAR_COLS = 1800
LIDAR_ELEV_DEG = np.linspace(-15.0, 15.0, LIDAR_RINGS)
LIDAR_AZ_DEG = np.arange(LIDAR_COLS) * (360.0 / LIDAR_COLS)
LIDAR_MIN_RANGE = 0.5
LIDAR_MAX_RANGE = 60.0
CAMERA_RATE = 10.0
LIDAR_RATE = 10.0
IMU_RATE = 200.0
ODOM_RATE = 50.0
FINE_DT = 0.001
PERIOD = 0.1                  # camera and LiDAR frame period, on the sensor's own clock
SWEEP = 0.1                   # LiDAR: column j is taken j / LIDAR_COLS * SWEEP (sensor time) after the scan's stamp
CAM_TRIGGER_JITTER = 0.0003
LIDAR_TRIGGER_JITTER = 0.00005
STAMP_JITTER = 0.0002
EXTRA_EDGE = 2.0              # and up to this much more (frame_trim)
LOG_EDGE = (0.2, 1.0)         # a sensor's log starts this long after a sequence's start and ends this long before its end

KNOCK_TYPE = "camera_bracket_shift"
BRACKET_STATES = (0.0, 90.0, -90.0, 180.0)   # as designed; a quarter turn left; a quarter turn right; back to front
WALK_SPEED = 1.3
FAULT_TYPES = ("camera_frozen", "camera_time_jump", "lidar_degraded", "imu_axis_swap",
               "odom_glitch")
FAULT_SENSOR = {"camera_frozen": "camera", "camera_time_jump": "camera",
                "lidar_degraded": "lidar", "imu_axis_swap": "imu", "odom_glitch": "wheel_odom"}

# Nominal camera orientation in the LiDAR frame: camera z (optical axis) along LiDAR x,
# camera x (right) along LiDAR -y, camera y (down) along LiDAR -z.
R_LIDAR_CAM_NOMINAL = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


# ----------------------------------------------------------------------------- SO(3) helpers

def hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def exp_so3(v):
    v = np.asarray(v, dtype=float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3) + hat(v)
    k = v / th
    K = hat(k)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def log_so3(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = min(1.0, max(-1.0, c))
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * w
    if th > math.pi - 1e-6:
        # near pi: axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(k))
        k = A[:, i] / max(k[i], 1e-12)
        k = k / np.linalg.norm(k)
        return th * k
    return th / (2.0 * math.sin(th)) * w


def rot_angle_deg(R):
    return math.degrees(float(np.linalg.norm(log_so3(R))))


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_zyx_to_R(yaw, pitch, roll):
    """Vectorised R = Rz(yaw) Ry(pitch) Rx(roll) for arrays of angles; returns (..., 3, 3)."""
    yaw, pitch, roll = np.broadcast_arrays(np.asarray(yaw, float), np.asarray(pitch, float),
                                           np.asarray(roll, float))
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.empty(yaw.shape + (3, 3))
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R, t = T[:3, :3], T[:3, 3]
    return make_T(R.T, -R.T @ t)


def random_unit(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def axis_aligned_rotations():
    """The 24 proper rotations whose matrices are signed permutations."""
    out = []
    import itertools
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for r, (c, s) in enumerate(zip(perm, signs)):
                R[r, c] = s
            if np.linalg.det(R) > 0:
                out.append(R)
    return out


# ----------------------------------------------------------------------------- texture hash

_M1 = np.uint64(0x9E3779B97F4A7C15)
_M2 = np.uint64(0xBF58476D1CE4E5B9)
_M3 = np.uint64(0x94D049BB133111EB)




def _hann_smooth(x, width_s, dt):
    n = max(3, int(round(width_s / dt)) | 1)
    w = np.hanning(n)
    w /= w.sum()
    pad = n // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(xp, w, mode="valid")



# ----------------------------------------------------------------------------- motion

def gen_controls(rng, style, duration):
    """Forward speed and yaw-rate commands on the fine grid, exactly zero in the rests."""
    dt = FINE_DT
    n = int(round(duration / dt)) + 1
    t = np.arange(n) * dt
    s0 = rng.uniform(2.2, 3.0)
    s1 = rng.uniform(2.2, 3.0)
    v = np.zeros(n)
    w = np.zeros(n)
    rests = [(0.0, s0), (duration - s1, duration)]
    tc = s0 + 0.75
    end = duration - s1 - 0.75
    while tc < end:
        seg = rng.uniform(1.2, 3.2)
        seg = min(seg, end - tc)
        r = rng.uniform()
        if style == "turns":
            if r < 0.22:
                vv, ww = 0.0, rng.choice([-1, 1]) * rng.uniform(0.5, 1.0)       # spin in place
            elif r < 0.30:
                vv, ww = 0.0, 0.0                                              # stop
            else:
                vv, ww = rng.uniform(0.4, 1.0), rng.choice([-1, 1]) * rng.uniform(0.3, 0.9)
        elif style == "stopgo":
            if r < 0.30:
                vv, ww = 0.0, 0.0
            elif r < 0.40:
                vv, ww = 0.0, rng.choice([-1, 1]) * rng.uniform(0.4, 0.8)
            else:
                vv, ww = rng.uniform(0.6, 1.4), rng.uniform(-0.25, 0.25)
        else:
            if r < 0.15:
                vv, ww = 0.0, 0.0
            elif r < 0.30:
                vv, ww = 0.0, rng.choice([-1, 1]) * rng.uniform(0.4, 0.9)
            else:
                vv, ww = rng.uniform(0.3, 1.2), rng.uniform(-0.7, 0.7)
        m = (t >= tc) & (t < tc + seg)
        v[m] = vv
        w[m] = ww
        if vv == 0.0 and ww == 0.0 and seg > 1.8:
            rests.append((tc + 0.75, tc + seg - 0.75))
        tc += seg
    # the smoothing's half width is 0.7 s and the commands are zero for 0.75 s around the first and last rest: a
    # rest is exactly zero without cutting the commands afterwards (the cut left a step in acceleration, which the
    # accelerometer showed as single-sample spikes)
    v = _hann_smooth(_hann_smooth(v, 1.0, dt), 0.4, dt)
    w = _hann_smooth(_hann_smooth(w, 1.0, dt), 0.4, dt)
    for a, b in rests:
        m = (t >= a) & (t <= b)
        assert not np.any(v[m]) and not np.any(w[m]), "a rest is not at rest"
    rests = [(a, b) for a, b in rests if b - a > 0.3]
    return t, v, w, rests


class Trajectory(object):
    """Body trajectory on the fine grid of one sequence (local time tau from 0)."""

    def __init__(self, tau, pos, ypr, v_cmd, w_cmd):
        self.tau = tau
        self.pos = pos            # (N,3) body origin in W
        self.ypr = ypr            # (N,3) yaw (unwrapped), pitch, roll
        self.v = v_cmd
        self.w = w_cmd
        dt = FINE_DT
        rates = np.gradient(ypr, dt, axis=0)
        yaw, pitch, roll = ypr[:, 0], ypr[:, 1], ypr[:, 2]
        yd, pd, rd = rates[:, 0], rates[:, 1], rates[:, 2]
        self.omega_b = np.stack([
            rd - yd * np.sin(pitch),
            pd * np.cos(roll) + yd * np.cos(pitch) * np.sin(roll),
            -pd * np.sin(roll) + yd * np.cos(pitch) * np.cos(roll)], axis=1)
        self.R = euler_zyx_to_R(yaw, pitch, roll)

    @property
    def duration(self):
        return float(self.tau[-1])

    def interp(self, tau):
        """Body pose (R (M,3,3), p (M,3)) at local times tau (clamped to the sequence)."""
        tau = np.clip(np.asarray(tau, float), 0.0, self.tau[-1])
        p = np.stack([np.interp(tau, self.tau, self.pos[:, i]) for i in range(3)], axis=1)
        ang = [np.interp(tau, self.tau, self.ypr[:, i]) for i in range(3)]
        return euler_zyx_to_R(ang[0], ang[1], ang[2]), p

    def interp_omega(self, tau):
        tau = np.clip(np.asarray(tau, float), 0.0, self.tau[-1])
        return np.stack([np.interp(tau, self.tau, self.omega_b[:, i]) for i in range(3)], axis=1)

    def point_kinematics(self, p_b, tau):
        """World position, velocity and acceleration of a body-fixed point at times tau."""
        pw = self.pos + np.einsum("nij,j->ni", self.R, p_b)
        vel = np.gradient(pw, FINE_DT, axis=0)
        acc = np.gradient(vel, FINE_DT, axis=0)
        tau = np.clip(np.asarray(tau, float), 0.0, self.tau[-1])
        out = []
        for arr in (pw, vel, acc):
            out.append(np.stack([np.interp(tau, self.tau, arr[:, i]) for i in range(3)], axis=1))
        return out


def _vibration(rng, t, rms, n=48):
    """Band-limited (5..60 Hz) signal with the given rms of its SECOND derivative (rms > 0) as a displacement:
    returns x(t) with rms(x'') = rms."""
    f = rng.uniform(5.0, 60.0, size=n)
    ph = rng.uniform(0.0, 2 * math.pi, size=n)
    w = 2 * math.pi * f
    a = rms * math.sqrt(2.0 / n)
    x = np.zeros_like(t)
    for i in range(n):
        x -= a / (w[i] * w[i]) * np.sin(w[i] * t + ph[i])
    return x


def _vibration_rate(rng, t, rms, n=48):
    """An angle whose FIRST derivative has the given rms, band 5..60 Hz."""
    f = rng.uniform(5.0, 60.0, size=n)
    ph = rng.uniform(0.0, 2 * math.pi, size=n)
    w = 2 * math.pi * f
    a = rms * math.sqrt(2.0 / n)
    x = np.zeros_like(t)
    for i in range(n):
        x -= a / w[i] * np.cos(w[i] * t + ph[i])
    return x


def gen_trajectory(rng, style, duration, zone, clear=None, cell=0.05, half=(30.0, 20.0), min_clear=0.9):
    """A drive inside the zone (x0, x1, y0, y1), at least min_clear from the hall's floor-level things (`clear`: the
    hall's clearance map in metres on `cell` cells; None = no check)."""
    for _attempt in range(400):
        tau, v, w, rests = gen_controls(rng, style, duration)
        dt = FINE_DT
        yaw = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * dt)])
        x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] * np.cos(yaw[1:]) + v[:-1] * np.cos(yaw[:-1])) * dt)])
        y = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] * np.sin(yaw[1:]) + v[:-1] * np.sin(yaw[:-1])) * dt)])
        placed = None
        for _k in range(40):
            a = rng.uniform(-math.pi, math.pi)
            ca, sa = math.cos(a), math.sin(a)
            xr = ca * x - sa * y
            yr = sa * x + ca * y
            if xr.max() - xr.min() > zone[1] - zone[0] - 2.0 or yr.max() - yr.min() > zone[3] - zone[2] - 2.0:
                continue
            ox = rng.uniform(zone[0] + 1.0 - xr.min(), zone[1] - 1.0 - xr.max())
            oy = rng.uniform(zone[2] + 1.0 - yr.min(), zone[3] - 1.0 - yr.max())
            if clear is not None:
                ix = np.floor((xr[::50] + ox + half[0]) / cell).astype(int)
                iy = np.floor((yr[::50] + oy + half[1]) / cell).astype(int)
                if clear[ix, iy].min() < min_clear:
                    continue
            placed = (xr + ox, yr + oy, yaw + a)
            break
        if placed is not None:
            break
    else:
        raise RuntimeError("could not fit a trajectory into the zone")
    xw, yw, yaw = placed
    # suspension: roll and pitch from lateral and longitudinal acceleration, plus a body oscillation that builds
    # up with motion and decays after a stop
    acc_long = np.gradient(v, dt)
    acc_lat = v * w
    env = np.tanh((np.abs(v) + 0.5 * np.abs(w)) / 0.35)
    decay = math.exp(-dt / rng.uniform(0.4, 0.8))
    envd = env.copy()
    for i in range(1, envd.size):
        if envd[i] < envd[i - 1] * decay:
            envd[i] = envd[i - 1] * decay
    c_roll = rng.uniform(0.012, 0.025)
    c_pitch = rng.uniform(0.012, 0.025)
    t = tau

    def osc(amp_deg):
        f1, f2 = rng.uniform(1.1, 2.0), rng.uniform(2.3, 3.4)
        p1, p2 = rng.uniform(0, 2 * math.pi, size=2)
        a = math.radians(amp_deg)
        return a * (np.sin(2 * math.pi * f1 * t + p1) + 0.45 * np.sin(2 * math.pi * f2 * t + p2))

    roll = c_roll * acc_lat + envd * osc(rng.uniform(0.7, 1.5))
    pitch = -c_pitch * acc_long + envd * osc(rng.uniform(0.7, 1.5))
    zb = envd * 0.004 * np.sin(2 * math.pi * rng.uniform(1.5, 2.5) * t + rng.uniform(0, 6.28))
    # drive-train and floor vibration while moving: real motion of the body, the same for every sensor
    lvl = np.clip(np.abs(v) + 0.4 * np.abs(w), 0.0, 1.2) * env
    a_rms = rng.uniform(0.15, 0.35)
    g_rms = rng.uniform(0.004, 0.010)
    vib = np.stack([_vibration(rng, t, a_rms) for _ in range(3)], axis=1) * lvl[:, None]
    ang = np.stack([_vibration_rate(rng, t, g_rms) for _ in range(3)], axis=1) * lvl[:, None]
    pos = np.stack([xw, yw, zb], axis=1) + vib
    ypr = np.stack([yaw, pitch, roll], axis=1) + ang
    return Trajectory(tau, pos, ypr, v, w), rests


# ----------------------------------------------------------------------------- rig

def _rng(seed_words, *path):
    ss = np.random.SeedSequence(entropy=list(seed_words), spawn_key=tuple(path))
    return np.random.default_rng(ss)


def rig_seed(set_name, index):
    """The pre-registered seed of rig `index` (an int, or "i.j" for the j-th replacement of a rejected rig i)."""
    h = hashlib.sha256(("ale-robotics/multisensor_rig_calibration_from_logs/v3/%s/%s" % (set_name, index)).encode()).digest()
    return [int.from_bytes(h[4 * i:4 * i + 4], "big") for i in range(4)]


class Rig(object):
    pass


def make_rig(seed_words, site, n_sequences=3, styles=None, has_odom=True, knock=False):
    """Sample every parameter of one rig (no rendering). `site`: rigsite.Site (the hall's occupancy and the props'
    footprints), or None for a rig without a site (unit tests of the motion and clock models)."""
    rig = Rig()
    rig.seed_words = [int(s) for s in seed_words]
    rp = _rng(rig.seed_words, 1)
    # camera intrinsics (given exactly in the manifest)
    rig.has_odom = bool(has_odom)
    f = F_NOMINAL * (1.0 + rp.uniform(-0.06, 0.06))       # a re-focused lens: NOT given, to be calibrated
    rig.K = np.array([[f * rp.uniform(0.9985, 1.0015), 0.0, IMG_W / 2.0 + rp.uniform(-20, 20)],
                      [0.0, f, IMG_H / 2.0 + rp.uniform(-20, 20)], [0.0, 0.0, 1.0]])
    rig.dist = np.array([rp.uniform(-0.32, -0.08), rp.uniform(-0.02, 0.14), rp.uniform(-0.0015, 0.0015),
                         rp.uniform(-0.0015, 0.0015), rp.uniform(-0.02, 0.02)])
    rig.readout = float(rp.uniform(0.010, 0.030))         # rolling shutter: first row to last row
    # LiDAR mount, known exactly (the arch's top bar carries it; risers of 0..60 mm under the arch's feet)
    rig.T_body_lidar = make_T(rz(math.radians(rp.uniform(-1.0, 1.0))),
                              np.array([rp.uniform(0.05, 0.20), rp.uniform(-0.03, 0.03),
                                        rp.uniform(0.94, 1.00)]))
    # camera: CAD nominal, then the true mount differs
    tilt = math.radians(rp.uniform(2.0, 10.0))          # designed downward tilt
    yaw_d = math.radians(rp.uniform(-4.0, 4.0))
    R_nom = rz(yaw_d) @ exp_so3(np.array([0.0, tilt, 0.0])) @ R_LIDAR_CAM_NOMINAL
    t_nom = np.array([rp.uniform(0.06, 0.16), rp.uniform(-0.08, 0.08), rp.uniform(-0.25, -0.12)])
    rig.T_lidar_camera_prior = make_T(R_nom, t_nom)
    dR = exp_so3(random_unit(rp) * math.radians(rp.uniform(2.0, 10.0)))
    dt_ = random_unit(rp) * rp.uniform(0.02, 0.08)
    # the bracket's state: as designed, or refitted a quarter or half turn about the mast's vertical axis (the LiDAR's z)
    rig.bracket_deg = float(BRACKET_STATES[int(rp.integers(0, 4))])
    Rq = rz(math.radians(rig.bracket_deg))
    rig.T_lidar_camera = make_T(Rq @ R_nom @ dR, Rq @ (t_nom + dt_))
    # IMU: axis-aligned CAD orientation, true mount differs
    rots = axis_aligned_rotations()
    R_inom = rots[int(rp.integers(0, len(rots)))]
    t_inom = np.array([rp.uniform(-0.15, 0.05), rp.uniform(-0.08, 0.08), rp.uniform(-0.45, -0.30)])
    rig.T_lidar_imu_prior = make_T(R_inom, t_inom)
    dRi = exp_so3(random_unit(rp) * math.radians(rp.uniform(1.0, 5.0)))
    dti = random_unit(rp) * rp.uniform(0.01, 0.04)
    rig.T_lidar_imu = make_T(R_inom @ dRi, t_inom + dti)
    # clocks
    rig.cam_offset = float(rp.choice([-1, 1]) * rp.uniform(0.02, 0.5))
    rig.cam_drift = float(rp.uniform(-200e-6, 200e-6))
    rig.lidar_offset = float(rp.choice([-1, 1]) * rp.uniform(0.02, 0.5))
    rig.lidar_drift = float(rp.uniform(-100e-6, 100e-6))
    rig.cam_phase = float(rp.uniform(0.0, PERIOD))        # sensor time of the camera's frame 0, modulo the period
    rig.lidar_phase = float(rp.uniform(0.0, PERIOD))
    # IMU errors (Bosch BMI088: inside the datasheet's offsets; scale and cross-axis after the factory trim)
    rig.gyro_bias = rp.uniform(-0.015, 0.015, size=3)
    rig.accel_bias = rp.uniform(-0.15, 0.15, size=3)
    rig.gyro_noise = 0.0035
    rig.accel_noise = 0.024
    rig.gyro_rw = 2e-5
    rig.accel_rw = 2e-4
    rig.gyro_S = np.diag(rp.uniform(-0.001, 0.001, size=3)) + rp.uniform(-0.001, 0.001, size=(3, 3)) * (1 - np.eye(3))
    rig.accel_S = np.diag(rp.uniform(-0.001, 0.001, size=3)) + rp.uniform(-0.001, 0.001, size=(3, 3)) * (1 - np.eye(3))
    # wheel odometry (skid steer)
    rig.odom_v_scale = float(rp.uniform(0.97, 1.03))
    rig.odom_w_scale = float(rp.uniform(0.80, 1.00))
    # camera exposure and noise
    rig.exposure = float(rp.uniform(0.85, 1.15))
    rig.cam_noise = float(rp.uniform(1.2, 2.0))
    # the working zone in the hall
    rs = _rng(rig.seed_words, 2)
    zx, zy = rs.uniform(24.0, 34.0), rs.uniform(16.0, 24.0)
    cx = rs.uniform(-30.0 + 2.5 + zx / 2, 30.0 - 2.5 - zx / 2)
    cy = rs.uniform(-20.0 + 2.5 + zy / 2, 20.0 - 2.5 - zy / 2)
    rig.zone = (cx - zx / 2, cx + zx / 2, cy - zy / 2, cy + zy / 2)
    rig.n_small = int(rs.integers(30, 46))
    rig.n_large = int(rs.integers(3, 7))
    # sequences: visible ones, then the held-out one
    if styles is None:
        base = ["turns", "stopgo", "mixed"]
        order = list(_rng(rig.seed_words, 3).permutation(3))
        styles = [base[i] for i in order][:n_sequences]
        while len(styles) < n_sequences:
            styles.append("mixed")
    rt = _rng(rig.seed_words, 4)
    # the session, in time: held-out drive A, the given sequences, held-out drive B (index c = 0 .. n + 1)
    starts = []
    tcur = rt.uniform(10.0, 30.0)
    durations = []
    for c in range(n_sequences + 2):
        d = float(rt.uniform(24.0, 28.0))
        starts.append(float(tcur))
        durations.append(d)
        tcur += d + rt.uniform(70.0, 130.0)
    rig.seq_styles = ["mixed"] + styles + ["mixed"]
    rig.seq_starts = starts
    rig.seq_durations = durations
    rig.n_visible = n_sequences
    rig.given = list(range(1, n_sequences + 1))
    rig.held = {"A": 0, "B": n_sequences + 1}
    rig.session_end = float(tcur)
    clear = cell = None
    if site is not None:
        clear, cell = site.clearance_map()
    rig.trajs = []
    rig.rests = []
    for k in range(n_sequences + 2):
        tr, rests = gen_trajectory(_rng(rig.seed_words, 5, k), rig.seq_styles[k], durations[k], rig.zone,
                                   clear, cell if cell else 0.05)
        rig.trajs.append(tr)
        rig.rests.append(rests)
    rig.props = []
    if site is not None:
        import rigsite
        tracks = [np.stack([tr.pos[::100, 0], tr.pos[::100, 1]], axis=1) for tr in rig.trajs]
        rig.props = rigsite.place_props(_rng(rig.seed_words, 6), site, rig.zone, tracks, rig.n_small, rig.n_large)
    # the IMU's slow bias walk over the session (1 s knots)
    rw = _rng(rig.seed_words, 12)
    nk = int(math.ceil(rig.session_end)) + 2
    rig.rw_t = np.arange(nk, dtype=float)
    rig.gyro_walk = np.cumsum(rw.normal(0.0, rig.gyro_rw, size=(nk, 3)), axis=0)
    rig.accel_walk = np.cumsum(rw.normal(0.0, rig.accel_rw, size=(nk, 3)), axis=0)
    # the scored biases: the mean over the given sequences
    tt = np.concatenate([starts[k] + np.arange(0.0, durations[k], 0.05) for k in rig.given])
    rig.gyro_bias_scored = rig.gyro_bias + np.array([np.interp(tt, rig.rw_t, rig.gyro_walk[:, i]).mean() for i in range(3)])
    rig.accel_bias_scored = rig.accel_bias + np.array([np.interp(tt, rig.rw_t, rig.accel_walk[:, i]).mean() for i in range(3)])
    # the knock: the camera bracket shifts once, during the second given sequence, while the robot moves
    rig.knock = None
    rig.T_lidar_camera_before = rig.T_lidar_camera.copy()
    if knock:
        rk = _rng(rig.seed_words, 13)
        c = rig.given[1]; tr = rig.trajs[c]
        for _ in range(200):
            tk = float(rk.uniform(8.0, 18.0))
            if abs(np.interp(tk, tr.tau, tr.v)) + 0.5 * abs(np.interp(tk, tr.tau, tr.w)) > 0.2:
                break
        dRk = exp_so3(random_unit(rk) * math.radians(rk.uniform(1.5, 4.0)))
        dtk = random_unit(rk) * rk.uniform(0.002, 0.008)
        after = rig.T_lidar_camera
        rig.T_lidar_camera_before = make_T(after[:3, :3] @ dRk, after[:3, 3] + dtk)
        rig.knock = {"c": c, "tau": tk, "t": starts[c] + tk}
    # movers: vehicles that drive and people who walk, per drive
    rig.movers = [[] for _ in rig.trajs]
    if site is not None:
        import rigsite
        for c, tr in enumerate(rig.trajs):
            rig.movers[c] = rigsite.sample_movers(_rng(rig.seed_words, 14, c), site, rig.zone, rig.props, tr, durations[c])
    # faults (given sequences only)
    rig.faults = sample_faults(_rng(rig.seed_words, 8), rig)
    return rig


def T_lidar_camera_at(rig, c, tau):
    """The camera's mount at local time tau of drive c (the knock changes it once)."""
    k = rig.knock
    if k is None:
        return rig.T_lidar_camera
    before = (c < k["c"]) or (c == k["c"] and tau < k["tau"])
    return rig.T_lidar_camera_before if before else rig.T_lidar_camera


def mover_state(m, tau):
    """Position (x, y), heading and pose index of a mover at local time tau: it runs its straight leg back and
    forth (a vehicle reverses, a person turns round)."""
    L, v = m["length"], m["speed"]
    s = (m["s0"] + v * np.asarray(tau, float)) % (2 * L)
    fwd = s <= L
    d = np.where(fwd, s, 2 * L - s)
    x = m["p0"][0] + d * math.cos(m["heading"]); y = m["p0"][1] + d * math.sin(m["heading"])
    if m["kind"] == "person":
        yaw = np.where(fwd, m["heading"], m["heading"] + math.pi)
        pose = (np.floor((m["phase"] + np.asarray(tau, float) * 25.0)) % 20).astype(int)
    else:
        yaw = np.full(np.shape(d), m["heading"] + m.get("yaw0", 0.0)); pose = np.zeros(np.shape(d), int)
    return x, y, yaw, pose


def sample_faults(rng, rig):
    n = int(rng.choice([1, 2, 2, 3]))
    types = [t for t in rng.permutation(len(FAULT_TYPES)) if rig.has_odom or FAULT_TYPES[int(t)] != "odom_glitch"][:n]
    faults = []
    for ti in types:
        ftype = FAULT_TYPES[int(ti)]
        for _ in range(50):
            k = int(rig.given[int(rng.integers(0, rig.n_visible))])
            dur = rig.seq_durations[k]
            length = float(rng.uniform(1.5, 3.5))
            a = float(rng.uniform(4.0, dur - 4.0 - length))
            clash = any(f["seq"] == k and not (a + length + 1.0 < f["tau0"] or a > f["tau1"] + 1.0)
                        for f in faults)
            tr = rig.trajs[k]
            m = (tr.tau >= a) & (tr.tau <= a + length)
            moving = np.mean((np.abs(tr.v[m]) + 0.5 * np.abs(tr.w[m])) > 0.15)
            if not clash and moving >= 0.85:
                break
        f = {"type": ftype, "sensor": FAULT_SENSOR[ftype], "seq": k, "tau0": a, "tau1": a + length}
        if ftype == "camera_time_jump":
            f["jump"] = float(rng.choice([-1, 1]) * rng.uniform(0.035, 0.080))
        elif ftype == "imu_axis_swap":
            f["pair"] = [(0, 1), (0, 2), (1, 2)][int(rng.choice([0, 0, 1, 2]))]
        elif ftype == "lidar_degraded":
            f["fraction"] = float(rng.uniform(0.35, 0.65))
        elif ftype == "odom_glitch":
            mode = str(rng.choice(["stall", "spike"]))
            tr = rig.trajs[k]
            m = (tr.tau >= a) & (tr.tau <= a + length)
            if mode == "stall" and np.mean(np.abs(tr.v[m]) > 0.3) < 0.85:
                mode = "spike"
            f["mode"] = mode
        faults.append(f)
    return faults


# ----------------------------------------------------------------------------- clocks

def sensor_stamp(t_ref, offset, drift):
    return t_ref + offset + drift * t_ref


def sensor_to_ref(stamp, offset, drift):
    return (stamp - offset) / (1.0 + drift)


def _clock(rig, sensor):
    if sensor == "camera":
        return rig.cam_offset, rig.cam_drift, rig.cam_phase, CAM_TRIGGER_JITTER
    return rig.lidar_offset, rig.lidar_drift, rig.lidar_phase, LIDAR_TRIGGER_JITTER


def frame_times(rig, k, sensor):
    """The frames of `sensor` ("camera" | "lidar") logged in sequence k (visible or held out).

    The sensor runs free for the whole session: frame n happens at sensor time phase + n * PERIOD + w_n (w_n: the
    trigger's jitter) and is stamped with that time plus a stamping jitter. The log holds the frames whose events
    fall between the sequence's start + e0 and its end - e1 (e0, e1 drawn per sensor and sequence).
    Returns (tau: reference time of each event minus the sequence's start, stamp: what the log says)."""
    o, d, phase, wj = _clock(rig, sensor)
    r = _rng(rig.seed_words, 9, k, 1 if sensor == "camera" else 2)
    e0, e1 = r.uniform(*LOG_EDGE), r.uniform(*LOG_EDGE)
    T0, dur = rig.seq_starts[k], rig.seq_durations[k]
    s_lo, s_hi = sensor_stamp(T0 + e0, o, d), sensor_stamp(T0 + dur - e1, o, d)
    n0 = int(math.ceil((s_lo - phase) / PERIOD)); n1 = int(math.floor((s_hi - phase) / PERIOD))
    n = np.arange(n0, n1 + 1)
    s_event = phase + n * PERIOD + r.normal(0.0, wj, size=n.size)
    tau = sensor_to_ref(s_event, o, d) - T0
    stamp = s_event + r.normal(0.0, STAMP_JITTER, size=n.size)
    a, b = frame_trim(rig, k, sensor)
    return tau[a:tau.size - b], stamp[a:stamp.size - b]


def frame_trim(rig, k, sensor):
    """Frames dropped from the start and the end of a sensor's log of a drive, beyond LOG_EDGE: 0..2 s more at each
    end, drawn per sensor and drive from a stream of their own (so the frames that stay are the same events). With
    clock offsets of up to half a second, edges of 0.2..1 s alone would bound an offset to about 0.1 s from where the
    log starts and ends against the drive's reference times."""
    r = _rng(rig.seed_words, 15, k, 1 if sensor == "camera" else 2)
    return int(r.uniform(0.0, EXTRA_EDGE) / PERIOD), int(r.uniform(0.0, EXTRA_EDGE) / PERIOD)


# ----------------------------------------------------------------------------- LiDAR

def lidar_ray_dirs():
    el = np.radians(LIDAR_ELEV_DEG)[:, None]
    az = np.radians(LIDAR_AZ_DEG)[None, :]
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                  np.sin(el) * np.ones_like(az)], axis=-1)
    return d.reshape(-1, 3)


def column_dt(rig):
    """Reference-time delay of each LiDAR column after column 0."""
    return (np.arange(LIDAR_COLS) / LIDAR_COLS * SWEEP) / (1.0 + rig.lidar_drift)


def sensor_pose_w(rig, traj, tau, T_body_sensor):
    Rb, pb = traj.interp(tau)
    R = Rb @ T_body_sensor[:3, :3]
    p = pb + Rb @ T_body_sensor[:3, 3]
    return R, p


def render_lidar(rig, site, traj, tau, rng, degrade=None, noise=True, movers=()):
    """One sweep whose column 0 is taken at local reference time tau: range (rings, cols) in metres (0 = no
    return) and reflectivity (0..1). Every column is cast from the LiDAR's pose at its own time (the driver does not
    motion-compensate). Hits: the site's triangles or the robot's own body, whichever is nearer."""
    tj = tau + column_dt(rig)
    R, p = sensor_pose_w(rig, traj, tj, rig.T_body_lidar)                       # (C,3,3), (C,3)
    d_l = lidar_ray_dirs().reshape(LIDAR_RINGS, LIDAR_COLS, 3)
    d_w = np.einsum("cij,rcj->rci", R, d_l)
    o_w = np.broadcast_to(p[None], d_w.shape)
    t, cosi, mat = site.caster.cast(o_w.reshape(-1, 3), d_w.reshape(-1, 3))
    nir = np.where(mat >= 0, site.nir[np.maximum(mat, 0)], np.nan)
    if getattr(rig, "self_range", None) is not None:                            # the robot's own body, in the LiDAR's frame
        own = rig.self_range.reshape(-1) < t
        t = np.where(own, rig.self_range.reshape(-1), t)
        cosi = np.where(own, rig.self_cos.reshape(-1), cosi)
        nir = np.where(own, rig.self_nir.reshape(-1), nir)
    for m in movers:                                                            # vehicles and people, each at the column's time
        mx, my, myaw, mpose = mover_state(m, tj)
        cy_, sy_ = np.cos(myaw), np.sin(myaw)
        ox = np.broadcast_to((p[:, 0] - mx)[None], d_w.shape[:2]); oy = np.broadcast_to((p[:, 1] - my)[None], d_w.shape[:2])
        o_m = np.stack([cy_ * ox + sy_ * oy, -sy_ * ox + cy_ * oy, np.broadcast_to(p[:, 2][None], d_w.shape[:2])], -1)
        d_m = np.stack([cy_ * d_w[..., 0] + sy_ * d_w[..., 1], -sy_ * d_w[..., 0] + cy_ * d_w[..., 1], d_w[..., 2]], -1)
        caster, mnir = site.mover_caster(m, int(mpose[LIDAR_COLS // 2]))
        tm, cm, _ = caster.cast(o_m.reshape(-1, 3), d_m.reshape(-1, 3))
        own = tm < t
        t = np.where(own, tm, t); cosi = np.where(own, cm, cosi); nir = np.where(own, mnir, nir)
    t = t.reshape(LIDAR_RINGS, LIDAR_COLS); cosi = cosi.reshape(t.shape); nir = nir.reshape(t.shape)
    true_range = t.copy()
    if not noise:
        return np.where(np.isfinite(t), t, 0.0), np.nan_to_num(nir)
    # mixed pixels at depth edges between neighbouring columns
    tf = np.where(np.isfinite(t), t, LIDAR_MAX_RANGE)
    for sh in (1, -1):
        nb = np.roll(tf, sh, axis=1)
        edge = np.abs(nb - tf) > 0.3
        mix = edge & (rng.uniform(size=t.shape) < 0.065)
        t = np.where(mix, tf + rng.uniform(size=t.shape) * (nb - tf), t)
    sigma = 0.02 + 0.01 * np.clip((tf - 20.0) / 40.0, 0.0, 1.0)
    rng_ = t + rng.normal(0.0, 1.0, size=t.shape) * sigma
    refl = nir * np.clip(cosi, 0.0, 1.0) ** 0.5
    refl = np.clip(np.where(cosi > 0, np.maximum(refl, 0.35 * nir), refl) + rng.normal(0.0, 0.02, size=t.shape), 0.0, 1.0)
    valid = np.isfinite(true_range) & np.isfinite(nir) & (rng_ > LIDAR_MIN_RANGE) & (rng_ < LIDAR_MAX_RANGE)
    valid &= cosi > 0.06                                                        # grazing
    valid &= np.nan_to_num(nir) * cosi >= 0.10 * (tf / LIDAR_MAX_RANGE) ** 2    # weaker than a 10 % target at 60 m
    valid &= rng.uniform(size=t.shape) >= 0.005
    if degrade is not None:
        m = rng.uniform(size=t.shape) < degrade
        short = rng.uniform(0.5, 1.0, size=t.shape) * np.minimum(np.where(np.isfinite(rng_), rng_, 12.0), 12.0)
        rng_ = np.where(m, short, rng_)
        refl = np.where(m, rng.uniform(0.0, 0.15, size=t.shape), refl)
        valid |= m & (rng_ > LIDAR_MIN_RANGE)
    rng_ = np.where(valid, rng_, 0.0)
    return rng_, np.where(valid, np.nan_to_num(refl), 0.0)


def lidar_points(range_img):
    """Points of a sweep in the LiDAR's frame AS LOGGED (each column at its own time), with their column index."""
    d = lidar_ray_dirs()
    r = range_img.reshape(-1)
    m = r > 0
    col = np.tile(np.arange(LIDAR_COLS), LIDAR_RINGS)[m]
    return d[m] * r[m, None], col


# ----------------------------------------------------------------------------- camera

def camera_source(rig):
    """The pinhole image the renderer draws for a frame, and the map from the logged (distorted) image into it.
    Returns dict(w, h, fx, fy, cx, cy, map_x, map_y)."""
    import cv2
    K, D = rig.K, rig.dist
    u, v = np.meshgrid(np.arange(IMG_W, dtype=np.float64), np.arange(IMG_H, dtype=np.float64))
    pts = np.stack([u.ravel(), v.ravel()], axis=1).reshape(-1, 1, 2)
    xy = cv2.undistortPoints(pts, K, D).reshape(-1, 2)                          # normalised pinhole coordinates
    fx, fy = K[0, 0] * RENDER_SS, K[1, 1] * RENDER_SS
    x0, x1, y0, y1 = xy[:, 0].min(), xy[:, 0].max(), xy[:, 1].min(), xy[:, 1].max()
    pad = 40.0                                                                  # room for the rolling shutter's shift
    cx, cy = -x0 * fx + pad, -y0 * fy + pad
    w = int(math.ceil((x1 - x0) * fx + 2 * pad)); h = int(math.ceil((y1 - y0) * fy + 2 * pad))
    w += w % 2; h += h % 2
    mx = (xy[:, 0] * fx + cx).reshape(IMG_H, IMG_W).astype(np.float32)
    my = (xy[:, 1] * fy + cy).reshape(IMG_H, IMG_W).astype(np.float32)
    return {"w": w, "h": h, "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy), "map_x": mx, "map_y": my}


def row_dt(rig):
    """Reference-time offset of each image row from the frame's stamp (rolling shutter, top row first)."""
    return (np.arange(IMG_H) - (IMG_H - 1) / 2.0) / IMG_H * rig.readout / (1.0 + rig.cam_drift)


def camera_shots(rig, c):
    """The frames of drive c for the renderer: [{name, eye, R (world from optical frame), movers}] at the middle of
    each frame's readout, with the plan (tau, stamps, which frames are frozen copies) for finish_camera."""
    traj = rig.trajs[c]
    name = "seq_%02d" % (c - 1)
    faults = [f for f in rig.faults if f["seq"] == c and f["sensor"] == "camera"]
    tau, stamps = frame_times(rig, c, "camera")
    frozen = np.zeros(tau.size, bool)
    recs = []
    for f in faults:
        m = (tau >= f["tau0"]) & (tau <= f["tau1"])
        idx = np.nonzero(m)[0]
        if f["type"] == "camera_time_jump":
            stamps[m] += f["jump"]
        else:
            frozen[idx] = True
        recs.append({"sequence": name, "sensor": "camera", "type": f["type"],
                     "start_s": float(stamps[idx[0]]), "end_s": float(stamps[idx[-1]])})
    if rig.knock is not None and rig.knock["c"] == c:
        sk = float(sensor_stamp(rig.knock["t"], rig.cam_offset, rig.cam_drift))
        recs.append({"sequence": name, "sensor": "camera", "type": KNOCK_TYPE, "start_s": sk, "end_s": sk})
    shots = []
    for i in range(tau.size):
        if frozen[i]:
            continue
        T_bc = rig.T_body_lidar @ T_lidar_camera_at(rig, c, tau[i])
        R, p = sensor_pose_w(rig, traj, tau[i:i + 1], T_bc)
        mv = []
        for k, m in enumerate(rig.movers[c]):
            x, y, yaw, pose = mover_state(m, tau[i])
            mv.append([k, int(pose), float(x), float(y), float(yaw)])
        shots.append({"name": "%06d" % i, "eye": [float(x) for x in p[0]], "R": [[float(x) for x in row] for row in R[0]], "movers": mv})
    return {"tau": tau, "stamps": stamps, "frozen": frozen, "shots": shots, "records": recs, "c": c}


def frame_depth(rig, site, c, tau, src, sub=2):
    """Depth along the optical axis of the renderer's pinhole image at local time tau (every sub-th pixel), by ray
    casting on the site and the movers: what the rolling shutter's row-by-row re-projection needs."""
    traj = rig.trajs[c]
    T_bc = rig.T_body_lidar @ T_lidar_camera_at(rig, c, tau)
    R, p = sensor_pose_w(rig, traj, np.array([tau]), T_bc); R, p = R[0], p[0]
    us = (np.arange(0, src["w"], sub) + 0.5 * (sub - 1)); vs = (np.arange(0, src["h"], sub) + 0.5 * (sub - 1))
    uu, vv = np.meshgrid(us, vs)
    d_c = np.stack([(uu - src["cx"]) / src["fx"], (vv - src["cy"]) / src["fy"], np.ones_like(uu)], -1).reshape(-1, 3)
    nrm = np.linalg.norm(d_c, axis=1)
    d_w = (d_c / nrm[:, None]) @ R.T
    o_w = np.broadcast_to(p, d_w.shape)
    t, _, _ = site.caster.cast(o_w, d_w)
    for m in rig.movers[c]:
        mx, my, myaw, mpose = mover_state(m, tau)
        cy_, sy_ = math.cos(float(myaw)), math.sin(float(myaw))
        ox, oy = o_w[:, 0] - float(mx), o_w[:, 1] - float(my)
        o_m = np.stack([cy_ * ox + sy_ * oy, -sy_ * ox + cy_ * oy, o_w[:, 2]], -1)
        d_m = np.stack([cy_ * d_w[:, 0] + sy_ * d_w[:, 1], -sy_ * d_w[:, 0] + cy_ * d_w[:, 1], d_w[:, 2]], -1)
        tm, _, _ = site.mover_caster(m, int(mpose))[0].cast(o_m, d_m)
        t = np.minimum(t, tm)
    z = np.where(np.isfinite(t), t, 80.0) / nrm
    return z.reshape(uu.shape).astype(np.float32)


def rolling_shutter_maps(rig, c, tau, src, depth):
    """For every pixel of the logged (distorted) image: where its light is in the renderer's mid-readout pinhole
    image. Row v is taken row_dt[v] from the frame's stamp, from the camera's pose at that time; the scene point is
    found with the mid-readout depth (looked up twice), so a static scene is exact to first order in the motion."""
    import cv2
    traj = rig.trajs[c]
    T_bc = rig.T_body_lidar @ T_lidar_camera_at(rig, c, tau)
    R0, p0 = sensor_pose_w(rig, traj, np.array([tau]), T_bc); R0, p0 = R0[0], p0[0]
    Rv, pv = sensor_pose_w(rig, traj, tau + row_dt(rig), T_bc)                  # (H,3,3), (H,3)
    Rrel = np.einsum("ji,vjk->vik", R0, Rv)                                     # row camera -> mid camera
    trel = np.einsum("ji,vj->vi", R0, pv - p0)
    x = (src["map_x"] - src["cx"]) / src["fx"]; y = (src["map_y"] - src["cy"]) / src["fy"]   # the pixel's ray in its row's camera
    ray = np.stack([x, y, np.ones_like(x)], -1)                                 # (H,W,3)
    dh, dw = depth.shape
    sx, sy = dw / float(src["w"]), dh / float(src["h"])
    mx, my = src["map_x"], src["map_y"]
    for _ in range(2):
        z = cv2.remap(depth, (mx * sx).astype(np.float32), (my * sy).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        X = np.einsum("vij,vwj->vwi", Rrel, ray * z[..., None]) + trel[:, None, :]
        zz = np.maximum(X[..., 2], 0.05)
        mx = (X[..., 0] / zz * src["fx"] + src["cx"]).astype(np.float32); my = (X[..., 1] / zz * src["fy"] + src["cy"]).astype(np.float32)
    return mx, my


def finish_frame(rig, src_bgr, src, rng, maps=None):
    """The logged frame from the renderer's pinhole image: lens distortion, vignetting, exposure, sensor noise,
    8 bits, JPEG. Returns the file's bytes."""
    import cv2
    mx, my = (src["map_x"], src["map_y"]) if maps is None else maps
    img = cv2.remap(src_bgr, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    lin = (img.astype(np.float32) / 255.0) ** 2.2
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)
    r2 = ((xx - rig.K[0, 2]) ** 2 + (yy - rig.K[1, 2]) ** 2) / float((IMG_W / 2.0) ** 2 + (IMG_H / 2.0) ** 2)
    lin *= (rig.exposure * (1.0 - 0.25 * r2))[..., None]
    out = np.clip(lin, 0.0, 1.0) ** (1.0 / 2.2) * 255.0
    out = out + rng.normal(0.0, rig.cam_noise, size=out.shape).astype(np.float32) * np.sqrt(0.25 + 0.75 * out / 255.0)
    out = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    assert ok
    return buf.tobytes()


# ----------------------------------------------------------------------------- IMU and odometry

def sim_imu(rig, traj, tau, T0, rng, faults=()):
    """Gyro (rad/s) and accelerometer (m/s^2) samples in the IMU frame at local times tau."""
    T_body_imu = rig.T_body_lidar @ rig.T_lidar_imu
    R_bi = T_body_imu[:3, :3]
    _, _, acc_w = traj.point_kinematics(T_body_imu[:3, 3], tau)
    Rb, _ = traj.interp(tau)
    om_b = traj.interp_omega(tau)
    gyro = om_b @ R_bi                                     # R_bi^T om_b
    f_w = acc_w - GRAVITY_W
    f_b = np.einsum("nji,nj->ni", Rb, f_w)                 # Rb^T f_w
    accel = f_b @ R_bi
    t = T0 + tau
    gw = np.stack([np.interp(t, rig.rw_t, rig.gyro_walk[:, i]) for i in range(3)], axis=1)
    aw = np.stack([np.interp(t, rig.rw_t, rig.accel_walk[:, i]) for i in range(3)], axis=1)
    gyro = gyro @ (np.eye(3) + rig.gyro_S).T + rig.gyro_bias + gw + rng.normal(0.0, rig.gyro_noise, size=gyro.shape)
    accel = accel @ (np.eye(3) + rig.accel_S).T + rig.accel_bias + aw + rng.normal(0.0, rig.accel_noise, size=accel.shape)
    for f in faults:
        if f["type"] != "imu_axis_swap":
            continue
        m = (tau >= f["tau0"]) & (tau <= f["tau1"])
        i, j = f["pair"]
        for arr in (gyro, accel):
            tmp = arr[m, i].copy()
            arr[m, i] = arr[m, j]
            arr[m, j] = tmp
    return gyro, accel


def sim_odom(rig, traj, tau, rng, faults=()):
    v = np.interp(tau, traj.tau, traj.v)
    w = np.interp(tau, traj.tau, traj.w)
    # wheel slip while accelerating and while turning (skid steer): the wheels over-read a little
    acc = np.abs(np.interp(tau, traj.tau, np.gradient(traj.v, FINE_DT)))
    slip = 1.0 + 0.04 * np.clip(acc / 0.8, 0.0, 1.0) + 0.03 * np.clip(np.abs(w), 0.0, 1.0)
    vm = rig.odom_v_scale * v * slip + rng.normal(0.0, 0.01, size=tau.shape)
    wm = rig.odom_w_scale * w + rng.normal(0.0, 0.01, size=tau.shape)
    for f in faults:
        if f["type"] != "odom_glitch":
            continue
        m = (tau >= f["tau0"]) & (tau <= f["tau1"])
        if f["mode"] == "stall":
            vm[m] = rng.normal(0.0, 0.003, size=int(m.sum()))
        else:
            vm[m] = vm[m] + 0.8 + 0.6 * np.sin(np.linspace(0, 6.0, int(m.sum())))
    return vm, wm


# ----------------------------------------------------------------------------- writing

def _fmt(x):
    return "%.6f" % x


def savez(path, **arrays):
    """np.savez_compressed with fixed zip dates: numpy stamps every member with the wall clock, which would make a
    rig's files differ from run to run and carry the time they were written."""
    import io
    import zipfile
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in arrays:
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asanyarray(arrays[name]), allow_pickle=False)
            zi = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue(), compresslevel=6)


def write_sequence(rig, site, c, seq_dir):
    """Write given drive c (chronological index) but the camera's frames (finish_camera does those after the
    renderer has drawn them). Returns the fault records in stamp terms and the camera plan."""
    os.makedirs(os.path.join(seq_dir, "camera", "frames"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "lidar", "scans"), exist_ok=True)
    traj = rig.trajs[c]
    T0 = rig.seq_starts[c]
    name = "seq_%02d" % (c - 1)
    faults = [f for f in rig.faults if f["seq"] == c]
    plan = camera_shots(rig, c)
    records = list(plan["records"])
    with open(os.path.join(seq_dir, "camera", "timestamps.csv"), "w") as fh:
        fh.write("frame,stamp_s\n")
        for i in range(plan["tau"].size):
            fh.write("%06d.jpg,%s\n" % (i, _fmt(plan["stamps"][i])))

    # lidar
    noise = _rng(rig.seed_words, 10, c, 2)
    tau_l, lst = frame_times(rig, c, "lidar")
    with open(os.path.join(seq_dir, "lidar", "timestamps.csv"), "w") as fh:
        fh.write("scan,stamp_s\n")
        for i, tl in enumerate(tau_l):
            deg = None
            for f in faults:
                if f["type"] == "lidar_degraded" and f["tau0"] <= tl <= f["tau1"]:
                    deg = f["fraction"]
            rimg, refl = render_lidar(rig, site, traj, tl, noise, degrade=deg, movers=rig.movers[c])
            fn = "%06d.npz" % i
            savez(os.path.join(seq_dir, "lidar", "scans", fn),
                  range_mm=np.round(rimg * 1000.0).astype(np.uint16),
                  reflectivity=np.round(refl * 255.0).astype(np.uint8))
            fh.write("%s,%s\n" % (fn, _fmt(lst[i])))
    for f in faults:
        if f["type"] == "lidar_degraded":
            m = (tau_l >= f["tau0"]) & (tau_l <= f["tau1"])
            idx = np.nonzero(m)[0]
            records.append({"sequence": name, "sensor": "lidar", "type": f["type"],
                            "start_s": float(lst[idx[0]]), "end_s": float(lst[idx[-1]])})

    # imu
    noise = _rng(rig.seed_words, 10, c, 3)
    tau_i = np.arange(0.0, rig.seq_durations[c], 1.0 / IMU_RATE)
    gyro, accel = sim_imu(rig, traj, tau_i, T0, noise, faults)
    with open(os.path.join(seq_dir, "imu.csv"), "w") as fh:
        fh.write("stamp_s,gx,gy,gz,ax,ay,az\n")
        for i in range(tau_i.size):
            fh.write("%s,%.6f,%.6f,%.6f,%.5f,%.5f,%.5f\n" % (
                _fmt(T0 + tau_i[i]), gyro[i, 0], gyro[i, 1], gyro[i, 2],
                accel[i, 0], accel[i, 1], accel[i, 2]))
    for f in faults:
        if f["type"] == "imu_axis_swap":
            m = (tau_i >= f["tau0"]) & (tau_i <= f["tau1"])
            idx = np.nonzero(m)[0]
            records.append({"sequence": name, "sensor": "imu", "type": f["type"],
                            "start_s": float(T0 + tau_i[idx[0]]), "end_s": float(T0 + tau_i[idx[-1]])})

    # wheel odometry (not on every rig)
    if rig.has_odom:
        noise = _rng(rig.seed_words, 10, c, 4)
        tau_o = np.arange(0.0, rig.seq_durations[c], 1.0 / ODOM_RATE)
        vm, wm = sim_odom(rig, traj, tau_o, noise, faults)
        with open(os.path.join(seq_dir, "wheel_odom.csv"), "w") as fh:
            fh.write("stamp_s,v_mps,yaw_rate_rps\n")
            for i in range(tau_o.size):
                fh.write("%s,%.4f,%.4f\n" % (_fmt(T0 + tau_o[i]), vm[i], wm[i]))
        for f in faults:
            if f["type"] == "odom_glitch":
                m = (tau_o >= f["tau0"]) & (tau_o <= f["tau1"])
                idx = np.nonzero(m)[0]
                records.append({"sequence": name, "sensor": "wheel_odom", "type": f["type"],
                                "start_s": float(T0 + tau_o[idx[0]]), "end_s": float(T0 + tau_o[idx[-1]])})
    return records, plan


def finish_camera(rig, site, c, seq_dir, plan, src_dir):
    """The camera's log of drive c from the renderer's pinhole images in src_dir (<name>.png): rolling shutter, lens,
    vignetting, noise, JPEG."""
    import cv2
    src = camera_source(rig)
    noise = _rng(rig.seed_words, 10, c, 1)
    last = None
    for i in range(plan["tau"].size):
        fn = "%06d.jpg" % i
        if plan["frozen"][i] and last is not None:
            data = last                                                         # the driver re-sends the last good image
        else:
            img = cv2.imread(os.path.join(src_dir, "%06d.png" % i), cv2.IMREAD_COLOR)
            assert img is not None and img.shape[1] == src["w"] and img.shape[0] == src["h"], (src_dir, i)
            depth = frame_depth(rig, site, c, plan["tau"][i], src)
            data = finish_frame(rig, img, src, noise, rolling_shutter_maps(rig, c, plan["tau"][i], src, depth))
        last = data
        with open(os.path.join(seq_dir, "camera", "frames", fn), "wb") as fo:
            fo.write(data)


def _mat_list(T):
    return [[float("%.9f" % x) for x in row] for row in np.asarray(T)]


def manifest(rig, rig_id):
    return {
        "rig_id": rig_id,
        "format_version": 3,
        "reference_clock": "base controller clock; stamps imu.csv and wheel_odom.csv",
        "sequences": [{"name": "seq_%02d" % (c - 1), "approx_start_s": round(rig.seq_starts[c], 1),
                       "approx_duration_s": round(rig.seq_durations[c], 1)} for c in rig.given],
        "camera": {"sensor": "FLIR Blackfly S BFS-U3-63S4C (Sony IMX178, rolling shutter), 4x4 binned, 6 mm lens (re-focused after the repair)",
                   "width": IMG_W, "height": IMG_H, "nominal_rate_hz": CAMERA_RATE,
                   "model": "pinhole with Brown-Conrady distortion (OpenCV: k1, k2, p1, p2, k3), optical frame x right, y down, z forward",
                   "nominal_focal_px": F_NOMINAL, "nominal_principal_point_px": [IMG_W / 2.0, IMG_H / 2.0],
                   "intrinsics": "NOT calibrated: focal length within 6 % of nominal (fx / fy within 0.3 %), principal point within 20 px of nominal, k1 -0.32..-0.08, k2 -0.02..0.14, p1 and p2 within 0.0015, k3 within 0.02",
                   "rolling_shutter": "rows are read top to bottom; the readout of a frame takes between 0.010 and 0.030 s (not calibrated); the stamp is the middle of the readout, on the camera's clock",
                   "exposure_ms": 2.0, "file_format": "JPEG, quality %d" % JPEG_QUALITY},
        "lidar": {"sensor": "Velodyne VLP-16, 600 rpm",
                  "rings": LIDAR_RINGS, "columns": LIDAR_COLS, "nominal_rate_hz": LIDAR_RATE,
                  "elevation_deg": [round(float(e), 6) for e in LIDAR_ELEV_DEG],
                  "azimuth_deg": "column j points at azimuth 360*j/columns deg, counter-clockwise from +x",
                  "range_unit": "mm (uint16), 0 = no return",
                  "reflectivity_unit": "0..255 (uint8)",
                  "min_range_m": LIDAR_MIN_RANGE, "max_range_m": LIDAR_MAX_RANGE,
                  "sweep_s": SWEEP,
                  "scan_model": "not motion-compensated: the stamp is the time of column 0 on the LiDAR's clock, column j is taken j/columns*sweep_s later"},
        "imu": {"sensor": "Bosch BMI088", "nominal_rate_hz": IMU_RATE, "gyro_unit": "rad/s", "accel_unit": "m/s^2",
                "accel_measures": "specific force (reads +9.81 along up when at rest)"},
        "wheel_odom": ({"present": True, "nominal_rate_hz": ODOM_RATE,
                        "fields": "v_mps: forward speed of the body; yaw_rate_rps: body yaw rate"} if rig.has_odom
                       else {"present": False, "note": "this rig's base controller logs no wheel odometry"}),
        "T_body_lidar": _mat_list(rig.T_body_lidar),
        "gravity_mps2": G,
    }


def prior_dict(rig):
    Tp = rig.T_lidar_imu_prior.copy(); Tp[:3, :3] = np.eye(3)
    return {
        "T_lidar_camera": _mat_list(rig.T_lidar_camera_prior),
        "T_lidar_imu": _mat_list(Tp),
        "error_bounds": {
            "T_lidar_camera": {"rotation_deg": 10.0, "translation_m": 0.08,
                               "bracket": "the bracket may have been refitted on another face of the mast: the true mount is the CAD mount, or the CAD mount turned by a quarter turn (either way) or a half turn about the LiDAR's z axis, and then off by up to rotation_deg and translation_m"},
            "T_lidar_imu": {"translation_m": 0.04,
                            "rotation": "NOT documented (the rotation block above is a placeholder): the board sits in one of the 24 axis-aligned orientations, off by up to 5 deg"},
        },
    }


def truth_dict(rig):
    out = {
        "T_lidar_camera": _mat_list(rig.T_lidar_camera),
        "T_lidar_imu": _mat_list(rig.T_lidar_imu),
        "camera": {"fx": float(rig.K[0, 0]), "fy": float(rig.K[1, 1]), "cx": float(rig.K[0, 2]), "cy": float(rig.K[1, 2]),
                   "distortion": [float(x) for x in rig.dist], "readout_s": float(rig.readout)},
        "time": {"camera": {"offset_s": rig.cam_offset, "drift_ppm": rig.cam_drift * 1e6},
                 "lidar": {"offset_s": rig.lidar_offset, "drift_ppm": rig.lidar_drift * 1e6}},
        "imu": {"gyro_bias_rps": [float(x) for x in rig.gyro_bias_scored],
                "accel_bias_mps2": [float(x) for x in rig.accel_bias_scored]},
    }
    if rig.knock is not None:
        out["changes"] = [{"type": KNOCK_TYPE, "T_lidar_camera_before": _mat_list(rig.T_lidar_camera_before)}]
    return out


def _dump_yaml(obj, path):
    import yaml
    with open(path, "w") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False, default_flow_style=None, width=120)


def write_rig(rig, site, rig_id, out_dir):
    """Write the agent-visible rig folder but the camera frames. Returns (fault records, {drive: camera plan})."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest(rig, rig_id), fh, indent=1)
    _dump_yaml(prior_dict(rig), os.path.join(out_dir, "rough_prior.yaml"))
    records, plans = [], {}
    for c in rig.given:
        r, p = write_sequence(rig, site, c, os.path.join(out_dir, "seq_%02d" % (c - 1)))
        records += r
        plans[c] = p
    return records, plans


# ----------------------------------------------------------------------------- held-out data

def heldout_drive(rig, site, c, n_pairs=36, pts_per_pair=600):
    """Evaluation data of held-out drive c: the fine body trajectory, IMU samples (true parameters, no faults), the
    camera and LiDAR stamps, and LiDAR points (as logged: each in the LiDAR's frame at its own column's time) paired
    with camera frames for the projection check (points the camera sees, by the true lens and readout)."""
    import cv2
    traj = rig.trajs[c]
    T0 = rig.seq_starts[c]
    noise = _rng(rig.seed_words, 11, c)
    tau_i = np.arange(0.0, rig.seq_durations[c], 1.0 / IMU_RATE)
    gyro, accel = sim_imu(rig, traj, tau_i, T0, noise)
    tau_c, cam_st = frame_times(rig, c, "camera")
    tau_l, lid_st = frame_times(rig, c, "lidar")
    cdt = column_dt(rig)
    # frames taken while the robot moves (at rest neither a clock nor the readout time shows)
    mv = np.abs(np.interp(tau_c, traj.tau, traj.v)) + 0.5 * np.abs(np.interp(tau_c, traj.tau, traj.w)) > 0.3
    cand = np.nonzero(mv & (np.arange(tau_c.size) >= 5) & (np.arange(tau_c.size) < tau_c.size - 5))[0]
    sel = cand[np.unique(np.linspace(0, cand.size - 1, n_pairs).round().astype(int))]
    pairs = []
    for ci in sel:
        T_body_cam = rig.T_body_lidar @ T_lidar_camera_at(rig, c, tau_c[ci])
        li = int(np.argmin(np.abs(tau_l + 0.5 * SWEEP - tau_c[ci])))
        rimg, _ = render_lidar(rig, site, traj, tau_l[li], noise, movers=())      # the static site only: a mover's point has no fixed place
        pts, col = lidar_points(rimg)
        Rl, pl = sensor_pose_w(rig, traj, tau_l[li] + cdt[col], rig.T_body_lidar)
        Rc, pc = sensor_pose_w(rig, traj, np.array([tau_c[ci]]), T_body_cam)
        pw = np.einsum("nij,nj->ni", Rl, pts) + pl
        pcam = (pw - pc[0]) @ Rc[0]
        z = pcam[:, 2]
        front = z > 0.5
        xn = np.where(front[:, None], pcam[:, :2] / np.where(front, z, 1.0)[:, None], 9.0)
        near = front & (np.abs(xn[:, 0]) < 0.75) & (np.abs(xn[:, 1]) < 0.5)        # inside the lens model's range
        idx = np.nonzero(near)[0]
        uv = cv2.projectPoints(pcam[idx].reshape(-1, 1, 3), np.zeros(3), np.zeros(3), rig.K, rig.dist)[0].reshape(-1, 2)
        ok = (uv[:, 0] >= 8) & (uv[:, 0] < IMG_W - 8) & (uv[:, 1] >= 8) & (uv[:, 1] < IMG_H - 8)
        idx = idx[ok]
        if idx.size > pts_per_pair:
            idx = np.sort(noise.choice(idx, size=pts_per_pair, replace=False))
        pairs.append({"cam_idx": int(ci), "lidar_idx": li, "points": pts[idx], "cols": col[idx]})
    return {"seq_start": T0, "traj": traj, "imu_tau": tau_i, "gyro": gyro, "accel": accel, "cam_tau": tau_c,
            "cam_stamp": cam_st, "lidar_tau": tau_l, "lidar_stamp": lid_st, "pairs": pairs}


def save_heldout(rig, site, path):
    """Both held-out drives (A before the given sequences, B after) in one npz."""
    arrays = {}
    for tag, c in rig.held.items():
        ho = heldout_drive(rig, site, c)
        tr = ho["traj"]
        d = {"seq_start": np.array(ho["seq_start"]),
             "traj_tau": tr.tau, "traj_pos": tr.pos, "traj_ypr": tr.ypr,
             "imu_tau": ho["imu_tau"], "gyro": ho["gyro"], "accel": ho["accel"],
             "cam_tau": ho["cam_tau"], "cam_stamp": ho["cam_stamp"],
             "lidar_tau": ho["lidar_tau"], "lidar_stamp": ho["lidar_stamp"],
             "pair_cam_idx": np.array([p["cam_idx"] for p in ho["pairs"]]),
             "pair_lidar_idx": np.array([p["lidar_idx"] for p in ho["pairs"]]),
             "pair_npts": np.array([len(p["points"]) for p in ho["pairs"]]),
             "pair_points": np.concatenate([p["points"] for p in ho["pairs"]], axis=0),
             "pair_cols": np.concatenate([p["cols"] for p in ho["pairs"]], axis=0).astype(np.int32)}
        for k, v in d.items():
            arrays["%s_%s" % (tag, k)] = v
    savez(path, **arrays)
