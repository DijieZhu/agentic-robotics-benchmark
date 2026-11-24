"""rigsim: deterministic generator of multi-sensor robot logs for rig calibration.

One *rig* is a wheeled robot carrying a spinning LiDAR (the rig's reference frame), an RGB
camera and an IMU, driving through one indoor scene. A rig has a set of true parameters
(camera and IMU extrinsics, sensor clock offsets and drifts, IMU biases, noise levels), a CAD
prior that is off by a few degrees and centimetres, several recorded sequences that may
contain injected faults, and one held-out sequence that the grader keeps to itself.

Everything is a pure function of the rig seed (a list of integers fed to
`numpy.random.SeedSequence`), so the grader can regenerate a rig bit for bit.

Frames and conventions (also stated in instruction.md):

* world W: z up, gravity (0, 0, -9.80665) m/s^2.
* body B: x forward, y left, z up, origin on the floor between the wheels.
* LiDAR L: x forward, y left, z up. T_body_lidar is known exactly (manifest.json).
* camera C: OpenCV optical frame (x right, y down, z forward), pinhole, no distortion.
* IMU I: an arbitrary orientation on the board; the CAD prior says roughly which.
* T_a_b is the pose of frame b in frame a: p_a = R_a_b p_b + t_a_b.
* reference clock: the base controller's clock, which stamps the IMU and the wheel odometry.
  A sensor with clock offset o and drift d stamps an event that happens at reference time t
  with s = t + o + d * t (d is dimensionless; ppm = 1e-6).
"""

import json
import math
import os

import numpy as np

G = 9.80665
GRAVITY_W = np.array([0.0, 0.0, -G])

IMG_W, IMG_H = 320, 240
SUPERSAMPLE = 2
LIDAR_RINGS = 32
LIDAR_COLS = 1024
LIDAR_ELEV_DEG = np.linspace(-16.0, 16.0, LIDAR_RINGS)
LIDAR_AZ_DEG = np.arange(LIDAR_COLS) * (360.0 / LIDAR_COLS)
LIDAR_MIN_RANGE = 0.4
LIDAR_MAX_RANGE = 60.0
CAMERA_RATE = 10.0
LIDAR_RATE = 10.0
IMU_RATE = 200.0
ODOM_RATE = 50.0
FINE_DT = 0.001

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


def _mix(h):
    h = h ^ (h >> np.uint64(30))
    h = h * _M2
    h = h ^ (h >> np.uint64(27))
    h = h * _M3
    h = h ^ (h >> np.uint64(31))
    return h


def hash01(ix, iy, iz, salt):
    """Deterministic hash of integer cell coordinates to [0, 1)."""
    with np.errstate(over="ignore"):
        h = (ix.astype(np.int64).astype(np.uint64) * np.uint64(0x8DA6B343)
             + iy.astype(np.int64).astype(np.uint64) * np.uint64(0xD8163841)
             + iz.astype(np.int64).astype(np.uint64) * np.uint64(0xCB1AB31F)
             + np.uint64(salt) * _M1)
        h = _mix(h)
    return (h >> np.uint64(11)).astype(np.float64) / float(1 << 53)


# ----------------------------------------------------------------------------- scene

class Scene(object):
    """A rectangular room (floor z=0, four walls, ceiling) with yaw-rotated boxes inside.

    Primitive 0 is the room (hit from inside); primitives 1.. are boxes. Every primitive has
    a procedural texture: 3-D cells of a per-primitive size in the primitive's local frame,
    each with a hashed colour, and a LiDAR reflectivity that follows the colour's luminance
    only in part.
    """

    def __init__(self, room_half, boxes, textures, light_dir):
        self.room_half = np.asarray(room_half, float)      # (hx, hy, height)
        self.boxes = boxes                                   # list of (cx, cy, cz, yaw, hx, hy, hz)
        self.textures = textures                             # per primitive dict
        self.light_dir = np.asarray(light_dir, float)
        b = np.array(boxes, float).reshape(-1, 7)
        self._c = b[:, :3]
        self._yaw = b[:, 3]
        self._h = b[:, 4:7]
        self._cos = np.cos(self._yaw)
        self._sin = np.sin(self._yaw)
        self._r = np.linalg.norm(self._h, axis=1)

    # -- ray casting -------------------------------------------------------------------
    def cast(self, origin, dirs):
        """Cast rays from one origin. dirs (N,3) unit. Returns t, prim, normal_w, local."""
        o = np.asarray(origin, float)
        d = dirs
        n = d.shape[0]
        hx, hy, hz = self.room_half
        lo = np.array([-hx, -hy, 0.0])
        hi = np.array([hx, hy, hz])
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            ta = (lo - o) * inv
            tb = (hi - o) * inv
        tfar = np.maximum(ta, tb)
        axis = np.argmin(tfar, axis=1)
        t = tfar[np.arange(n), axis]
        prim = np.zeros(n, dtype=np.int32)
        normal = np.zeros((n, 3))
        normal[np.arange(n), axis] = -np.sign(d[np.arange(n), axis])
        local = o + d * t[:, None]

        for k in range(self._c.shape[0]):
            # bounding-sphere prefilter
            oc = self._c[k] - o
            proj = d @ oc
            dist2 = float(oc @ oc) - proj * proj
            cand = np.nonzero((dist2 <= self._r[k] ** 2) & (proj > -self._r[k]))[0]
            if cand.size == 0:
                continue
            c, s = self._cos[k], self._sin[k]
            ol = np.array([c * (o[0] - self._c[k, 0]) + s * (o[1] - self._c[k, 1]),
                           -s * (o[0] - self._c[k, 0]) + c * (o[1] - self._c[k, 1]),
                           o[2] - self._c[k, 2]])
            dc = d[cand]
            dl = np.stack([c * dc[:, 0] + s * dc[:, 1], -s * dc[:, 0] + c * dc[:, 1], dc[:, 2]],
                          axis=1)
            h = self._h[k]
            with np.errstate(divide="ignore", invalid="ignore"):
                inv = 1.0 / dl
                t1 = (-h - ol) * inv
                t2 = (h - ol) * inv
            tmin_ax = np.minimum(t1, t2)
            tmax_ax = np.maximum(t1, t2)
            tmin = np.max(tmin_ax, axis=1)
            tmax = np.min(tmax_ax, axis=1)
            hit = (tmax >= tmin) & (tmin > 1e-6) & (tmin < t[cand])
            if not np.any(hit):
                continue
            idx = cand[hit]
            th = tmin[hit]
            ax = np.argmax(tmin_ax[hit], axis=1)
            nl = np.zeros((idx.size, 3))
            nl[np.arange(idx.size), ax] = -np.sign(dl[hit][np.arange(idx.size), ax])
            t[idx] = th
            prim[idx] = k + 1
            normal[idx] = np.stack([c * nl[:, 0] - s * nl[:, 1], s * nl[:, 0] + c * nl[:, 1],
                                    nl[:, 2]], axis=1)
            local[idx] = ol + dl[hit] * th[:, None]
        # room primitives use world coordinates as their local frame
        return t, prim, normal, local

    # -- appearance --------------------------------------------------------------------
    def albedo(self, prim, local):
        """RGB albedo (N,3) and LiDAR reflectivity (N,) of hit points."""
        n = prim.shape[0]
        rgb = np.zeros((n, 3))
        refl = np.zeros(n)
        for p in np.unique(prim):
            m = prim == p
            tex = self.textures[int(p)]
            q = local[m]
            cell = tex["cell"]
            ix = np.floor((q[:, 0] + tex["off"][0]) / cell[0])
            iy = np.floor((q[:, 1] + tex["off"][1]) / cell[1])
            iz = np.floor((q[:, 2] + tex["off"][2]) / cell[2])
            h1 = hash01(ix, iy, iz, tex["salt"])
            h2 = hash01(ix, iy, iz, tex["salt"] + 7919)
            h3 = hash01(ix, iy, iz, tex["salt"] + 104729)
            pal = tex["palette"]                       # (P, 3)
            pick = np.minimum((h2 * pal.shape[0]).astype(int), pal.shape[0] - 1)
            shade = tex["lo"] + (tex["hi"] - tex["lo"]) * h1
            col = pal[pick] * shade[:, None]
            rgb[m] = np.clip(col, 0.02, 1.0)
            lum = 0.299 * col[:, 0] + 0.587 * col[:, 1] + 0.114 * col[:, 2]
            refl[m] = np.clip(0.65 * lum + 0.35 * h3 * tex["refl_gain"], 0.02, 1.0)
        return rgb, refl


def sample_scene(rng, room_half, trajectories):
    """Boxes placed in free space around the given (x, y) trajectory samples."""
    hx, hy, hz = room_half
    pts = np.concatenate([tr[:, :2] for tr in trajectories], axis=0)
    boxes = []
    target = int(rng.integers(38, 52))
    tries = 0
    while len(boxes) < target and tries < 4000:
        tries += 1
        kind = rng.choice(["crate", "crate", "shelf", "pillar", "cabinet"])
        if kind == "crate":
            half = np.array([rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.6), rng.uniform(0.2, 0.75)])
        elif kind == "shelf":
            half = np.array([rng.uniform(0.7, 1.5), rng.uniform(0.2, 0.35), rng.uniform(0.75, 1.25)])
        elif kind == "cabinet":
            half = np.array([rng.uniform(0.3, 0.6), rng.uniform(0.25, 0.4), rng.uniform(0.5, 1.0)])
        else:
            half = np.array([rng.uniform(0.15, 0.3), rng.uniform(0.15, 0.3), hz / 2.0])
        yaw = rng.uniform(0.0, math.pi)
        rxy = math.hypot(half[0], half[1])
        cx = rng.uniform(-hx + rxy + 0.05, hx - rxy - 0.05)
        cy = rng.uniform(-hy + rxy + 0.05, hy - rxy - 0.05)
        d = np.hypot(pts[:, 0] - cx, pts[:, 1] - cy)
        if d.min() < rxy + 0.85:
            continue
        ok = True
        for b in boxes:
            if math.hypot(b[0] - cx, b[1] - cy) < rxy + math.hypot(b[4], b[5]) + 0.1:
                ok = False
                break
        if not ok:
            continue
        boxes.append((cx, cy, half[2], yaw, half[0], half[1], half[2]))
    return boxes


def sample_textures(rng, n_prims):
    textures = []
    for p in range(n_prims):
        if p == 0:
            cell = rng.uniform(0.5, 1.2, size=3)
            palette = np.array([[0.75, 0.74, 0.70], [0.55, 0.57, 0.60], [0.80, 0.70, 0.55],
                                [0.35, 0.40, 0.48], [0.62, 0.30, 0.25], [0.25, 0.45, 0.30]])
            palette = palette[rng.permutation(len(palette))[: int(rng.integers(3, 6))]]
            lo, hi = 0.35, 1.05
        else:
            cell = rng.uniform(0.12, 0.45, size=3)
            k = int(rng.integers(2, 5))
            palette = rng.uniform(0.15, 0.95, size=(k, 3))
            lo, hi = rng.uniform(0.3, 0.6), rng.uniform(0.95, 1.15)
        textures.append({
            "cell": cell,
            "off": rng.uniform(0.0, 1.0, size=3),
            "salt": int(rng.integers(1, 2 ** 31 - 1)),
            "palette": palette,
            "lo": float(lo),
            "hi": float(hi),
            "refl_gain": float(rng.uniform(0.6, 1.4)),
        })
    return textures


# ----------------------------------------------------------------------------- trajectories

def _hann_smooth(x, width_s, dt):
    n = max(3, int(round(width_s / dt)) | 1)
    w = np.hanning(n)
    w /= w.sum()
    pad = n // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(xp, w, mode="valid")


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
    tc = s0 + 0.6
    end = duration - s1 - 0.6
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
        if vv == 0.0 and ww == 0.0:
            rests.append((tc + 0.6, tc + seg - 0.6))
        tc += seg
    v = _hann_smooth(_hann_smooth(v, 1.0, dt), 0.4, dt)
    w = _hann_smooth(_hann_smooth(w, 1.0, dt), 0.4, dt)
    for a, b in rests[:2]:
        m = (t >= a) & (t <= b)
        v[m] = 0.0
        w[m] = 0.0
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


def gen_trajectory(rng, style, duration, room_half, margin=2.6):
    hx, hy, _ = room_half
    for _attempt in range(200):
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
            span_x = xr.max() - xr.min()
            span_y = yr.max() - yr.min()
            if span_x > 2 * (hx - margin) or span_y > 2 * (hy - margin):
                continue
            ox = rng.uniform(-hx + margin - xr.min(), hx - margin - xr.max())
            oy = rng.uniform(-hy + margin - yr.min(), hy - margin - yr.max())
            placed = (xr + ox, yr + oy, yaw + a)
            break
        if placed is not None:
            break
    else:
        raise RuntimeError("could not fit a trajectory into the room")
    xw, yw, yaw = placed
    # suspension: roll and pitch from lateral and longitudinal acceleration, plus vibration
    acc_long = np.gradient(v, dt)
    acc_lat = v * w
    env = np.tanh((np.abs(v) + 0.5 * np.abs(w)) / 0.35)
    c_roll = rng.uniform(0.012, 0.025)
    c_pitch = rng.uniform(0.012, 0.025)
    t = tau

    def osc(amp_deg):
        f1, f2 = rng.uniform(1.1, 2.0), rng.uniform(2.3, 3.4)
        p1, p2 = rng.uniform(0, 2 * math.pi, size=2)
        a = math.radians(amp_deg)
        return a * (np.sin(2 * math.pi * f1 * t + p1) + 0.45 * np.sin(2 * math.pi * f2 * t + p2))

    roll = c_roll * acc_lat + env * osc(rng.uniform(0.7, 1.5))
    pitch = -c_pitch * acc_long + env * osc(rng.uniform(0.7, 1.5))
    zb = env * 0.004 * np.sin(2 * math.pi * rng.uniform(1.5, 2.5) * t + rng.uniform(0, 6.28))
    pos = np.stack([xw, yw, zb], axis=1)
    ypr = np.stack([yaw, pitch, roll], axis=1)
    return Trajectory(tau, pos, ypr, v, w), rests


# ----------------------------------------------------------------------------- rig

def _rng(seed_words, *path):
    ss = np.random.SeedSequence(entropy=list(seed_words), spawn_key=tuple(path))
    return np.random.default_rng(ss)


class Rig(object):
    pass


def make_rig(seed_words, n_sequences=3, styles=None):
    """Sample every parameter of one rig (cheap; no rendering)."""
    rig = Rig()
    rig.seed_words = [int(s) for s in seed_words]
    rp = _rng(rig.seed_words, 1)
    # intrinsics
    f = rp.uniform(235.0, 290.0)
    rig.K = np.array([[f * rp.uniform(0.995, 1.005), 0.0, 160.0 + rp.uniform(-5, 5)],
                      [0.0, f, 120.0 + rp.uniform(-5, 5)], [0.0, 0.0, 1.0]])
    # LiDAR mount, known exactly
    rig.T_body_lidar = make_T(rz(math.radians(rp.uniform(-1.0, 1.0))),
                              np.array([rp.uniform(0.05, 0.20), rp.uniform(-0.03, 0.03),
                                        rp.uniform(0.70, 0.95)]))
    # camera: CAD nominal, then the true mount differs
    tilt = math.radians(rp.uniform(2.0, 10.0))          # designed downward tilt
    yaw_d = math.radians(rp.uniform(-4.0, 4.0))
    R_nom = rz(yaw_d) @ exp_so3(np.array([0.0, tilt, 0.0])) @ R_LIDAR_CAM_NOMINAL
    t_nom = np.array([rp.uniform(0.06, 0.16), rp.uniform(-0.08, 0.08), rp.uniform(-0.25, -0.12)])
    rig.T_lidar_camera_prior = make_T(R_nom, t_nom)
    dR = exp_so3(random_unit(rp) * math.radians(rp.uniform(2.0, 10.0)))
    dt_ = random_unit(rp) * rp.uniform(0.02, 0.08)
    rig.T_lidar_camera = make_T(R_nom @ dR, t_nom + dt_)
    # IMU: axis-aligned CAD orientation, true mount differs
    rots = axis_aligned_rotations()
    R_inom = rots[int(rp.integers(0, len(rots)))]
    t_inom = np.array([rp.uniform(-0.15, 0.05), rp.uniform(-0.08, 0.08), rp.uniform(-0.45, -0.30)])
    rig.T_lidar_imu_prior = make_T(R_inom, t_inom)
    dRi = exp_so3(random_unit(rp) * math.radians(rp.uniform(1.0, 5.0)))
    dti = random_unit(rp) * rp.uniform(0.01, 0.04)
    rig.T_lidar_imu = make_T(R_inom @ dRi, t_inom + dti)
    # clocks
    rig.cam_offset = float(rp.choice([-1, 1]) * rp.uniform(0.015, 0.100))
    rig.cam_drift = float(rp.uniform(-400e-6, 400e-6))
    rig.lidar_offset = float(rp.choice([-1, 1]) * rp.uniform(0.005, 0.060))
    rig.lidar_drift = float(rp.uniform(-150e-6, 150e-6))
    rig.stamp_jitter = 0.0002
    # IMU errors
    rig.gyro_bias = rp.uniform(-0.015, 0.015, size=3)
    rig.accel_bias = rp.uniform(-0.15, 0.15, size=3)
    rig.gyro_noise = 0.0015
    rig.accel_noise = 0.015
    # wheel odometry
    rig.odom_v_scale = float(rp.uniform(0.97, 1.03))
    rig.odom_w_scale = float(rp.uniform(0.95, 1.05))
    # camera exposure / light
    rig.exposure = float(rp.uniform(0.85, 1.1))
    rig.lidar_range_noise = 0.008
    # scene
    rs = _rng(rig.seed_words, 2)
    rig.room_half = np.array([rs.uniform(13.0, 18.0), rs.uniform(9.0, 13.0), rs.uniform(4.0, 6.0)])
    ld = np.array([rs.uniform(-0.5, 0.5), rs.uniform(-0.5, 0.5), 1.0])
    rig.light_dir = ld / np.linalg.norm(ld)
    # sequences: visible ones, then the held-out one
    if styles is None:
        base = ["turns", "stopgo", "mixed"]
        order = list(_rng(rig.seed_words, 3).permutation(3))
        styles = [base[i] for i in order][:n_sequences]
        while len(styles) < n_sequences:
            styles.append("mixed")
    rt = _rng(rig.seed_words, 4)
    starts = []
    tcur = rt.uniform(10.0, 30.0)
    durations = []
    for k in range(n_sequences + 1):
        d = float(rt.uniform(24.0, 28.0))
        starts.append(float(tcur))
        durations.append(d)
        tcur += d + rt.uniform(70.0, 130.0)
    rig.seq_styles = styles + ["mixed"]
    rig.seq_starts = starts
    rig.seq_durations = durations
    rig.n_visible = n_sequences
    rig.trajs = []
    rig.rests = []
    for k in range(n_sequences + 1):
        tr, rests = gen_trajectory(_rng(rig.seed_words, 5, k), rig.seq_styles[k], durations[k],
                                   rig.room_half)
        rig.trajs.append(tr)
        rig.rests.append(rests)
    tracks = [np.stack([tr.pos[::250, 0], tr.pos[::250, 1]], axis=1) for tr in rig.trajs]
    boxes = sample_scene(_rng(rig.seed_words, 6), rig.room_half, tracks)
    textures = sample_textures(_rng(rig.seed_words, 7), len(boxes) + 1)
    rig.scene = Scene(rig.room_half, boxes, textures, rig.light_dir)
    # faults (visible sequences only)
    rig.faults = sample_faults(_rng(rig.seed_words, 8), rig)
    return rig


def sample_faults(rng, rig):
    n = int(rng.choice([1, 2, 2, 3]))
    types = list(rng.permutation(len(FAULT_TYPES)))[:n]
    faults = []
    for ti in types:
        ftype = FAULT_TYPES[int(ti)]
        for _ in range(50):
            k = int(rng.integers(0, rig.n_visible))
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


# ----------------------------------------------------------------------------- sensor models

def sensor_stamp(t_ref, offset, drift):
    return t_ref + offset + drift * t_ref


def sensor_to_ref(stamp, offset, drift):
    return (stamp - offset) / (1.0 + drift)


def lidar_ray_dirs():
    el = np.radians(LIDAR_ELEV_DEG)[:, None]
    az = np.radians(LIDAR_AZ_DEG)[None, :]
    d = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                  np.sin(el) * np.ones_like(az)], axis=-1)
    return d.reshape(-1, 3)


def camera_ray_dirs(K, ss=SUPERSAMPLE):
    u = (np.arange(IMG_W * ss) + 0.5) / ss - 0.5
    v = (np.arange(IMG_H * ss) + 0.5) / ss - 0.5
    uu, vv = np.meshgrid(u, v)
    x = (uu - K[0, 2]) / K[0, 0]
    y = (vv - K[1, 2]) / K[1, 1]
    d = np.stack([x, y, np.ones_like(x)], axis=-1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def sensor_pose_w(rig, traj, tau, T_body_sensor):
    Rb, pb = traj.interp(tau)
    R = Rb @ T_body_sensor[:3, :3]
    p = pb + Rb @ T_body_sensor[:3, 3]
    return R, p


def render_lidar(rig, traj, tau, rng, degrade=None):
    """One instantaneous scan at local time tau: range (rings, cols) metres and reflectivity."""
    R, p = sensor_pose_w(rig, traj, np.array([tau]), rig.T_body_lidar)
    R, p = R[0], p[0]
    d_l = lidar_ray_dirs()
    d_w = d_l @ R.T
    t, prim, normal, local = rig.scene.cast(p, d_w)
    _, refl = rig.scene.albedo(prim, local)
    cosi = np.abs(np.sum(normal * d_w, axis=1))
    refl = refl * (0.35 + 0.65 * cosi ** 0.5)
    rng_ = t + rng.normal(0.0, rig.lidar_range_noise, size=t.shape)
    refl = np.clip(refl + rng.normal(0.0, 0.02, size=t.shape), 0.0, 1.0)
    valid = (rng_ > LIDAR_MIN_RANGE) & (rng_ < LIDAR_MAX_RANGE)
    # grazing returns drop out
    valid &= cosi > 0.06
    if degrade is not None:
        m = rng.uniform(size=t.shape) < degrade
        rng_ = np.where(m, rng.uniform(0.5, 1.0, size=t.shape) * np.minimum(rng_, 12.0), rng_)
        refl = np.where(m, rng.uniform(0.0, 0.15, size=t.shape), refl)
        valid |= m & (rng_ > LIDAR_MIN_RANGE)
    rng_ = np.where(valid, rng_, 0.0)
    return rng_.reshape(LIDAR_RINGS, LIDAR_COLS), refl.reshape(LIDAR_RINGS, LIDAR_COLS)


def lidar_points(range_img, refl_img=None):
    d = lidar_ray_dirs()
    r = range_img.reshape(-1)
    m = r > 0
    pts = d[m] * r[m, None]
    if refl_img is None:
        return pts
    return pts, refl_img.reshape(-1)[m]


def render_camera(rig, traj, tau, rng, cam_dirs=None):
    T_body_cam = rig.T_body_lidar @ rig.T_lidar_camera
    R, p = sensor_pose_w(rig, traj, np.array([tau]), T_body_cam)
    R, p = R[0], p[0]
    if cam_dirs is None:
        cam_dirs = camera_ray_dirs(rig.K)
    d_w = cam_dirs @ R.T
    t, prim, normal, local = rig.scene.cast(p, d_w)
    rgb, _ = rig.scene.albedo(prim, local)
    lam = np.clip(normal @ rig.light_dir, 0.0, None)
    # a soft fill light from the camera side keeps faces that face away readable
    fill = np.clip(-np.sum(normal * d_w, axis=1), 0.0, None)
    shade = 0.30 + 0.55 * lam + 0.20 * fill
    img = rgb * shade[:, None] * rig.exposure
    ss = SUPERSAMPLE
    img = img.reshape(IMG_H, ss, IMG_W, ss, 3).mean(axis=(1, 3))
    img = img + rng.normal(0.0, 1.5 / 255.0, size=img.shape)
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)


def sim_imu(rig, traj, tau, rng, faults=()):
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
    gyro = gyro + rig.gyro_bias + rng.normal(0.0, rig.gyro_noise, size=gyro.shape)
    accel = accel + rig.accel_bias + rng.normal(0.0, rig.accel_noise, size=accel.shape)
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
    # ordinary wheel slip while accelerating hard: the wheels over-read a little
    acc = np.abs(np.interp(tau, traj.tau, np.gradient(traj.v, FINE_DT)))
    slip = 1.0 + 0.04 * np.clip(acc / 0.8, 0.0, 1.0)
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


def frame_times(rig, k, rate, phase_salt):
    """True reference times of a sensor's frames in visible/held-out sequence k."""
    r = _rng(rig.seed_words, 9, k, phase_salt)
    phase = r.uniform(0.0, 1.0 / rate)
    dur = rig.seq_durations[k]
    tau = np.arange(0.05 + phase, dur - 0.05, 1.0 / rate)
    return tau


# ----------------------------------------------------------------------------- writing

def _fmt(x):
    return "%.6f" % x


def write_sequence(rig, k, seq_dir):
    """Render and write visible sequence k. Returns the fault records in stamp terms."""
    import cv2
    os.makedirs(os.path.join(seq_dir, "camera", "frames"), exist_ok=True)
    os.makedirs(os.path.join(seq_dir, "lidar", "scans"), exist_ok=True)
    traj = rig.trajs[k]
    T0 = rig.seq_starts[k]
    faults = [f for f in rig.faults if f["seq"] == k]
    noise = _rng(rig.seed_words, 10, k)
    records = []

    # camera
    tau_c = frame_times(rig, k, CAMERA_RATE, 1)
    t_ref = T0 + tau_c
    stamps = sensor_stamp(t_ref, rig.cam_offset, rig.cam_drift)
    stamps = stamps + noise.normal(0.0, rig.stamp_jitter, size=stamps.shape)
    cam_dirs = camera_ray_dirs(rig.K)
    last_img = None
    frozen_img = None
    affected = {}
    for f in faults:
        if f["sensor"] == "camera":
            m = (tau_c >= f["tau0"]) & (tau_c <= f["tau1"])
            affected[id(f)] = np.nonzero(m)[0]
            if f["type"] == "camera_time_jump":
                stamps[m] += f["jump"]
    with open(os.path.join(seq_dir, "camera", "timestamps.csv"), "w") as fh:
        fh.write("frame,stamp_s\n")
        for i, tc in enumerate(tau_c):
            frozen = any(f["type"] == "camera_frozen" and i in affected[id(f)] for f in faults
                         if f["sensor"] == "camera")
            if frozen:
                if frozen_img is None:
                    frozen_img = last_img
                img = frozen_img
            else:
                frozen_img = None
                img = render_camera(rig, traj, tc, noise, cam_dirs)
            last_img = img
            name = "%06d.png" % i
            cv2.imwrite(os.path.join(seq_dir, "camera", "frames", name),
                        np.ascontiguousarray(img[:, :, ::-1]))
            fh.write("%s,%s\n" % (name, _fmt(stamps[i])))
    for f in faults:
        if f["sensor"] == "camera":
            idx = affected[id(f)]
            records.append({"sequence": "seq_%02d" % k, "sensor": "camera", "type": f["type"],
                            "start_s": float(stamps[idx[0]]), "end_s": float(stamps[idx[-1]])})

    # lidar
    tau_l = frame_times(rig, k, LIDAR_RATE, 2)
    t_ref = T0 + tau_l
    lst = sensor_stamp(t_ref, rig.lidar_offset, rig.lidar_drift)
    lst = lst + noise.normal(0.0, rig.stamp_jitter, size=lst.shape)
    with open(os.path.join(seq_dir, "lidar", "timestamps.csv"), "w") as fh:
        fh.write("scan,stamp_s\n")
        for i, tl in enumerate(tau_l):
            deg = None
            for f in faults:
                if f["type"] == "lidar_degraded" and f["tau0"] <= tl <= f["tau1"]:
                    deg = f["fraction"]
            rimg, refl = render_lidar(rig, traj, tl, noise, degrade=deg)
            name = "%06d.npz" % i
            np.savez(os.path.join(seq_dir, "lidar", "scans", name),
                     range_mm=np.round(rimg * 1000.0).astype(np.uint16),
                     reflectivity=np.round(refl * 255.0).astype(np.uint8))
            fh.write("%s,%s\n" % (name, _fmt(lst[i])))
    for f in faults:
        if f["type"] == "lidar_degraded":
            m = (tau_l >= f["tau0"]) & (tau_l <= f["tau1"])
            idx = np.nonzero(m)[0]
            records.append({"sequence": "seq_%02d" % k, "sensor": "lidar", "type": f["type"],
                            "start_s": float(lst[idx[0]]), "end_s": float(lst[idx[-1]])})

    # imu
    tau_i = np.arange(0.0, rig.seq_durations[k], 1.0 / IMU_RATE)
    gyro, accel = sim_imu(rig, traj, tau_i, noise, faults)
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
            records.append({"sequence": "seq_%02d" % k, "sensor": "imu", "type": f["type"],
                            "start_s": float(T0 + tau_i[idx[0]]), "end_s": float(T0 + tau_i[idx[-1]])})

    # wheel odometry
    tau_o = np.arange(0.0, rig.seq_durations[k], 1.0 / ODOM_RATE)
    vm, wm = sim_odom(rig, traj, tau_o, noise, faults)
    with open(os.path.join(seq_dir, "wheel_odom.csv"), "w") as fh:
        fh.write("stamp_s,v_mps,yaw_rate_rps\n")
        for i in range(tau_o.size):
            fh.write("%s,%.4f,%.4f\n" % (_fmt(T0 + tau_o[i]), vm[i], wm[i]))
    for f in faults:
        if f["type"] == "odom_glitch":
            m = (tau_o >= f["tau0"]) & (tau_o <= f["tau1"])
            idx = np.nonzero(m)[0]
            records.append({"sequence": "seq_%02d" % k, "sensor": "wheel_odom", "type": f["type"],
                            "start_s": float(T0 + tau_o[idx[0]]), "end_s": float(T0 + tau_o[idx[-1]])})
    return records


def _mat_list(T):
    return [[float("%.9f" % x) for x in row] for row in np.asarray(T)]


def manifest(rig, rig_id):
    return {
        "rig_id": rig_id,
        "format_version": 1,
        "reference_clock": "base controller clock; stamps imu.csv and wheel_odom.csv",
        "sequences": [{"name": "seq_%02d" % k, "approx_start_s": round(rig.seq_starts[k], 1),
                       "approx_duration_s": round(rig.seq_durations[k], 1)}
                      for k in range(rig.n_visible)],
        "camera": {"width": IMG_W, "height": IMG_H, "nominal_rate_hz": CAMERA_RATE,
                   "model": "pinhole, no distortion, OpenCV optical frame (x right, y down, z forward)",
                   "K": _mat_list(rig.K), "global_shutter": True},
        "lidar": {"rings": LIDAR_RINGS, "columns": LIDAR_COLS, "nominal_rate_hz": LIDAR_RATE,
                  "elevation_deg": [round(float(e), 6) for e in LIDAR_ELEV_DEG],
                  "azimuth_deg": "column j points at azimuth 360*j/columns deg, counter-clockwise from +x",
                  "range_unit": "mm (uint16), 0 = no return",
                  "reflectivity_unit": "0..255 (uint8)",
                  "min_range_m": LIDAR_MIN_RANGE, "max_range_m": LIDAR_MAX_RANGE,
                  "scan_model": "each scan is motion-compensated by the driver to its stamp"},
        "imu": {"nominal_rate_hz": IMU_RATE, "gyro_unit": "rad/s", "accel_unit": "m/s^2",
                "accel_measures": "specific force (reads +9.81 along up when at rest)"},
        "wheel_odom": {"nominal_rate_hz": ODOM_RATE,
                       "fields": "v_mps: forward speed of the body; yaw_rate_rps: body yaw rate"},
        "T_body_lidar": _mat_list(rig.T_body_lidar),
        "gravity_mps2": G,
    }


def prior_dict(rig):
    return {
        "T_lidar_camera": _mat_list(rig.T_lidar_camera_prior),
        "T_lidar_imu": _mat_list(rig.T_lidar_imu_prior),
        "error_bounds": {
            "T_lidar_camera": {"rotation_deg": 10.0, "translation_m": 0.08},
            "T_lidar_imu": {"rotation_deg": 5.0, "translation_m": 0.04},
        },
    }


def truth_dict(rig):
    return {
        "T_lidar_camera": _mat_list(rig.T_lidar_camera),
        "T_lidar_imu": _mat_list(rig.T_lidar_imu),
        "time": {"camera": {"offset_s": rig.cam_offset, "drift_ppm": rig.cam_drift * 1e6},
                 "lidar": {"offset_s": rig.lidar_offset, "drift_ppm": rig.lidar_drift * 1e6}},
        "imu": {"gyro_bias_rps": [float(x) for x in rig.gyro_bias],
                "accel_bias_mps2": [float(x) for x in rig.accel_bias]},
    }


def _dump_yaml(obj, path):
    import yaml
    with open(path, "w") as fh:
        yaml.safe_dump(obj, fh, sort_keys=False, default_flow_style=None, width=120)


def write_rig(rig, rig_id, out_dir):
    """Write the agent-visible rig folder. Returns the fault records (stamp terms)."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest(rig, rig_id), fh, indent=1)
    _dump_yaml(prior_dict(rig), os.path.join(out_dir, "rough_prior.yaml"))
    records = []
    for k in range(rig.n_visible):
        records += write_sequence(rig, k, os.path.join(out_dir, "seq_%02d" % k))
    return records


# ----------------------------------------------------------------------------- held-out data

def heldout(rig, n_pairs=36, pts_per_pair=600):
    """Evaluation data from the held-out sequence; never written into an agent-visible rig.

    Returns a dict of arrays: the fine body trajectory, IMU samples (true parameters, no
    faults), the camera and LiDAR stamps of the sequence, and LiDAR points paired with
    camera frames for the projection check.
    """
    k = rig.n_visible
    traj = rig.trajs[k]
    T0 = rig.seq_starts[k]
    noise = _rng(rig.seed_words, 11)
    tau_i = np.arange(0.0, rig.seq_durations[k], 1.0 / IMU_RATE)
    gyro, accel = sim_imu(rig, traj, tau_i, noise)
    tau_c = frame_times(rig, k, CAMERA_RATE, 1)
    tau_l = frame_times(rig, k, LIDAR_RATE, 2)
    cam_st = sensor_stamp(T0 + tau_c, rig.cam_offset, rig.cam_drift) + noise.normal(0, rig.stamp_jitter, tau_c.shape)
    lid_st = sensor_stamp(T0 + tau_l, rig.lidar_offset, rig.lidar_drift) + noise.normal(0, rig.stamp_jitter, tau_l.shape)
    # pairs: camera frames spread over the sequence, each with the nearest LiDAR scan
    sel = np.linspace(5, tau_c.size - 6, n_pairs).round().astype(int)
    pairs = []
    T_body_cam = rig.T_body_lidar @ rig.T_lidar_camera
    for ci in sel:
        li = int(np.argmin(np.abs(tau_l - tau_c[ci])))
        rimg, refl = render_lidar(rig, traj, tau_l[li], noise)
        pts = lidar_points(rimg)
        # keep points that the camera actually sees at its own capture time
        Rl, pl = sensor_pose_w(rig, traj, np.array([tau_l[li]]), rig.T_body_lidar)
        Rc, pc = sensor_pose_w(rig, traj, np.array([tau_c[ci]]), T_body_cam)
        pw = pts @ Rl[0].T + pl[0]
        pcam = (pw - pc[0]) @ Rc[0]
        z = pcam[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = rig.K[0, 0] * pcam[:, 0] / z + rig.K[0, 2]
            v = rig.K[1, 1] * pcam[:, 1] / z + rig.K[1, 2]
        m = (z > 0.5) & (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        idx = np.nonzero(m)[0]
        if idx.size > pts_per_pair:
            idx = noise.choice(idx, size=pts_per_pair, replace=False)
        pairs.append({"cam_idx": int(ci), "lidar_idx": li, "points": pts[idx]})
    return {
        "seq_start": T0,
        "traj": traj,
        "imu_tau": tau_i,
        "gyro": gyro,
        "accel": accel,
        "cam_tau": tau_c,
        "cam_stamp": cam_st,
        "lidar_tau": tau_l,
        "lidar_stamp": lid_st,
        "pairs": pairs,
    }


def save_heldout(ho, path):
    """Compact npz form of `heldout()` (used for the practice rigs' truth folders)."""
    tr = ho["traj"]
    sub = slice(None)
    arrays = {
        "seq_start": np.array(ho["seq_start"]),
        "traj_tau": tr.tau[sub], "traj_pos": tr.pos[sub], "traj_ypr": tr.ypr[sub],
        "traj_v": tr.v[sub], "traj_w": tr.w[sub],
        "imu_tau": ho["imu_tau"], "gyro": ho["gyro"], "accel": ho["accel"],
        "cam_tau": ho["cam_tau"], "cam_stamp": ho["cam_stamp"],
        "lidar_tau": ho["lidar_tau"], "lidar_stamp": ho["lidar_stamp"],
        "pair_cam_idx": np.array([p["cam_idx"] for p in ho["pairs"]]),
        "pair_lidar_idx": np.array([p["lidar_idx"] for p in ho["pairs"]]),
        "pair_npts": np.array([len(p["points"]) for p in ho["pairs"]]),
        "pair_points": np.concatenate([p["points"] for p in ho["pairs"]], axis=0),
    }
    np.savez_compressed(path, **arrays)
