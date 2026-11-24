"""Reference calibrator for multisensor_rig_calibration_from_logs.

    python3 calibrate.py --input <rig_dir> --output <out_dir>

Pipeline (all from the rig's own logs; nothing else is read):

1. Load every sequence. Rest intervals come from the gyro and the wheel odometry.
2. Gyro bias from the rests (median over rests).
3. LiDAR odometry per sequence: scan-to-scan point-to-plane ICP; per-scan outlier
   fractions flag degraded LiDAR intervals.
4. LiDAR clock (offset + drift) and the IMU rotation: coarse lag from cross-correlating
   angular speeds, then a joint robust least-squares fit of the LiDAR's relative rotations
   against integrated gyro. Gyro residuals flag IMU axis faults.
5. Accelerometer bias from the rests, with gravity taken from the LiDAR's floor plane.
6. Camera faults from the stamps (time jumps) and the pixels (frozen frames).
7. Camera-LiDAR extrinsic: LiDAR reflectivity against image intensity (correlation),
   coarse-to-fine, on frames taken at rest; then the camera clock from frames taken while
   moving (LiDAR odometry carries points to the camera time); then a joint refinement.
8. Wheel-odometry glitches from forward speed against the LiDAR's.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree

G = 9.80665


# ----------------------------------------------------------------------------- SO(3)

def hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def expm(v):
    v = np.asarray(v, float)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3) + hat(v)
    K = hat(v / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


def logm(R):
    c = min(1.0, max(-1.0, (np.trace(R) - 1.0) / 2.0))
    th = math.acos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-9:
        return 0.5 * w
    if th > math.pi - 1e-6:
        A = (R + np.eye(3)) / 2.0
        k = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(k))
        k = A[:, i] / max(k[i], 1e-12)
        return th * k / np.linalg.norm(k)
    return th / (2.0 * math.sin(th)) * w


def exp_batch(w):
    th = np.linalg.norm(w, axis=1)
    ths = np.where(th < 1e-12, 1.0, th)
    k = w / ths[:, None]
    K = np.zeros((w.shape[0], 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s = np.sin(th)[:, None, None]
    c = (1.0 - np.cos(th))[:, None, None]
    return np.eye(3)[None] + s * K + c * (K @ K)


def log_batch(R):
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    th = np.arccos(c)
    w = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1)
    s = np.sin(th)
    f = np.where(th < 1e-6, 0.5, th / (2.0 * np.where(np.abs(s) < 1e-12, 1.0, s)))
    return w * f[:, None]


def quat_mul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], axis=-1)


def quat_from_rotvec(w):
    th = np.linalg.norm(w, axis=-1, keepdims=True)
    half = 0.5 * th
    k = np.where(th < 1e-12, 0.5, np.sin(half) / np.where(th < 1e-12, 1.0, th))
    return np.concatenate([np.cos(half), w * k], axis=-1)


def quat_to_R(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z)
    R[..., 0, 1] = 2 * (x * y - z * w)
    R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w)
    R[..., 1, 1] = 1 - 2 * (x * x + z * z)
    R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w)
    R[..., 2, 1] = 2 * (y * z + x * w)
    R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def prefix_quat(q):
    """Inclusive prefix product q0*q1*...*qi (Hillis-Steele scan)."""
    out = q.copy()
    shift = 1
    n = q.shape[0]
    while shift < n:
        nxt = out.copy()
        nxt[shift:] = quat_mul(out[:-shift], out[shift:])
        out = nxt
        shift *= 2
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


# ----------------------------------------------------------------------------- loading

def read_csv(path):
    with open(path) as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


class Sequence(object):
    pass


def load_rig(rig_dir):
    with open(os.path.join(rig_dir, "manifest.json")) as fh:
        man = json.load(fh)
    import yaml
    with open(os.path.join(rig_dir, "rough_prior.yaml")) as fh:
        prior = yaml.safe_load(fh)
    el = np.radians(np.array(man["lidar"]["elevation_deg"], float))[:, None]
    ncol = int(man["lidar"]["columns"])
    az = np.radians(np.arange(ncol) * 360.0 / ncol)[None, :]
    dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az),
                     np.sin(el) * np.ones_like(az)], axis=-1)
    seqs = []
    for s in man["sequences"]:
        d = os.path.join(rig_dir, s["name"])
        q = Sequence()
        q.name = s["name"]
        q.dir = d
        _, rows = read_csv(os.path.join(d, "camera", "timestamps.csv"))
        q.cam_files = [os.path.join(d, "camera", "frames", r[0]) for r in rows]
        q.cam_stamp = np.array([float(r[1]) for r in rows])
        _, rows = read_csv(os.path.join(d, "lidar", "timestamps.csv"))
        q.lidar_files = [os.path.join(d, "lidar", "scans", r[0]) for r in rows]
        q.lidar_stamp = np.array([float(r[1]) for r in rows])
        imu = np.loadtxt(os.path.join(d, "imu.csv"), delimiter=",", skiprows=1)
        q.imu_t, q.gyro, q.acc = imu[:, 0], imu[:, 1:4], imu[:, 4:7]
        od = np.loadtxt(os.path.join(d, "wheel_odom.csv"), delimiter=",", skiprows=1)
        q.odom_t, q.odom_v, q.odom_w = od[:, 0], od[:, 1], od[:, 2]
        seqs.append(q)
    rig = {"manifest": man, "prior": prior, "dirs": dirs, "seqs": seqs,
           "K": np.array(man["camera"]["K"], float),
           "T_body_lidar": np.array(man["T_body_lidar"], float),
           "W": int(man["camera"]["width"]), "H": int(man["camera"]["height"])}
    return rig


def load_scan(rig, path, despeckle=True):
    z = np.load(path)
    r = z["range_mm"].astype(np.float64) / 1000.0
    refl = z["reflectivity"].astype(np.float64) / 255.0
    m = r > 0
    if despeckle:
        m &= supported(r)
    return rig["dirs"][m] * r[m][:, None], refl[m], m


def supported(r):
    """Returns whose range agrees with at least one 4-neighbour of the range image."""
    ok = np.zeros(r.shape, bool)
    tol = 0.05 + 0.04 * r
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb = np.roll(np.roll(r, dr, axis=0), dc, axis=1)
        if dr == 1:
            nb[0, :] = 0
        if dr == -1:
            nb[-1, :] = 0
        ok |= (nb > 0) & (np.abs(nb - r) < tol)
    return ok & (r > 0)


def speckle_fraction(path):
    z = np.load(path)
    r = z["range_mm"].astype(np.float64) / 1000.0
    valid = r > 0
    return float(np.mean(~supported(r)[valid]))


def scan_edges(rig, path):
    """Edge points of one scan: reflectivity steps and depth discontinuities along each ring.

    A reflectivity edge sits between two neighbouring returns on the same surface (at their
    midpoint); a depth edge sits on the near side, along the direction halfway between the
    two beams. Returns points (N,3) in the LiDAR frame and weights (N,).
    """
    z = np.load(path)
    r = z["range_mm"].astype(np.float64) / 1000.0
    refl = z["reflectivity"].astype(np.float64) / 255.0
    r = np.where(supported(r), r, 0.0)
    dirs = rig["dirs"]
    rn = np.roll(r, -1, axis=1)
    fn = np.roll(refl, -1, axis=1)
    dn = np.roll(dirs, -1, axis=1)
    both = (r > 0.8) & (rn > 0.8) & (r < 30.0) & (rn < 30.0)
    same = both & (np.abs(r - rn) < 0.05 + 0.03 * np.minimum(r, rn))
    dstep = both & ~same
    mid = dirs + dn
    mid /= np.linalg.norm(mid, axis=-1, keepdims=True)
    wr = np.where(same, np.abs(refl - fn), 0.0)
    wr = np.where(wr > 0.06, wr, 0.0)
    pr = mid * (0.5 * (r + rn))[..., None]
    near = np.minimum(r, rn)
    wd = np.where(dstep, np.clip(np.abs(r - rn) / 0.5, 0.0, 1.0) * 0.5, 0.0)
    pd = mid * near[..., None]
    m1 = wr > 0
    m2 = wd > 0
    pts = np.concatenate([pr[m1], pd[m2]], axis=0)
    w = np.concatenate([wr[m1], wd[m2]])
    return pts, w


def load_gray(path):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0


# ----------------------------------------------------------------------------- rests

def rest_intervals(q, min_len=0.6):
    """Reference-time intervals where the robot stands still (gyro quiet, wheels at zero)."""
    t = q.imu_t
    win = 40
    sd = np.zeros(t.size)
    for ax in range(3):
        x = np.pad(q.gyro[:, ax], win, mode="edge")
        c1 = np.cumsum(np.concatenate([[0.0], x]))
        c2 = np.cumsum(np.concatenate([[0.0], x * x]))
        n = 2 * win + 1
        mu = (c1[n:] - c1[:-n]) / n
        var = (c2[n:] - c2[:-n]) / n - mu * mu
        sd += np.maximum(var[: t.size], 0.0)
    still = np.sqrt(sd) < 0.006
    k = 25
    ker = np.ones(k) / k
    va = np.abs(np.convolve(q.odom_v, ker, mode="same"))
    wa = np.abs(np.convolve(q.odom_w, ker, mode="same"))
    still &= (np.interp(t, q.odom_t, va) < 0.012) & (np.interp(t, q.odom_t, wa) < 0.012)
    out = []
    i = 0
    n = t.size
    while i < n:
        if still[i]:
            j = i
            while j + 1 < n and still[j + 1]:
                j += 1
            a, b = t[i] + 0.25, t[j] - 0.25
            if b - a >= min_len:
                out.append((a, b))
            i = j + 1
        else:
            i += 1
    return out


# ----------------------------------------------------------------------------- LiDAR odometry

def voxel_down(p, size):
    k = np.floor(p / size).astype(np.int64)
    _, idx = np.unique(k, axis=0, return_index=True)
    return p[np.sort(idx)]


def normals_knn(p, tree, k=12):
    _, nn = tree.query(p, k=k)
    nb = p[nn]
    mu = nb.mean(axis=1, keepdims=True)
    C = np.einsum("nki,nkj->nij", nb - mu, nb - mu) / k
    w, v = np.linalg.eigh(C)
    n = v[:, :, 0]
    planarity = (w[:, 1] - w[:, 0]) / np.maximum(w[:, 2], 1e-12)
    return n, planarity


def icp_p2pl(src, dst, dst_tree, dst_n, T0, iters=20):
    T = T0.copy()
    for it in range(iters):
        maxd = 1.0 if it < 5 else (0.5 if it < 10 else 0.25)
        ps = src @ T[:3, :3].T + T[:3, 3]
        d, j = dst_tree.query(ps, k=1, distance_upper_bound=maxd)
        m = np.isfinite(d)
        if m.sum() < 50:
            break
        ps, q, n = ps[m], dst[j[m]], dst_n[j[m]]
        r = np.sum((ps - q) * n, axis=1)
        J = np.concatenate([np.cross(ps, n), n], axis=1)
        s = 0.05
        w = np.minimum(1.0, s / np.maximum(np.abs(r), 1e-12))
        H = J.T @ (J * w[:, None])
        g = J.T @ (w * r)
        dx = -np.linalg.solve(H + 1e-9 * np.eye(6), g)
        dT = make_T(expm(dx[:3]), dx[3:])
        T = dT @ T
        if np.linalg.norm(dx) < 1e-7:
            break
    return T


def lidar_odometry(rig, q):
    """Poses T_W_L(i) of every scan (W = first scan's frame) and per-scan outlier fractions."""
    n = len(q.lidar_files)
    poses = [np.eye(4)]
    outlier = np.zeros(n)
    prev = None
    vel = np.eye(4)
    for i in range(n):
        pts, _, _ = load_scan(rig, q.lidar_files[i])
        outlier[i] = speckle_fraction(q.lidar_files[i])
        pts = pts[np.linalg.norm(pts, axis=1) > 1.0]
        pd = voxel_down(pts, 0.15)
        if prev is not None:
            ptree, pn, ppts = prev
            T0 = vel
            T = icp_p2pl(pd, ppts, ptree, pn, T0)
            poses.append(poses[-1] @ T)
            vel = T
        tree = cKDTree(pd)
        nrm, _ = normals_knn(pd, tree)
        prev = (tree, nrm, pd)
    return np.array(poses), outlier


def flag_runs(times, bad, min_count=2, gap=0.35):
    """Group flagged samples into [start, end] intervals."""
    idx = np.nonzero(bad)[0]
    out = []
    if idx.size == 0:
        return out
    start = prev = idx[0]
    for i in idx[1:]:
        if times[i] - times[prev] > gap:
            if prev - start + 1 >= min_count:
                out.append((times[start], times[prev]))
            start = i
        prev = i
    if prev - start + 1 >= min_count:
        out.append((times[start], times[prev]))
    return out


# ----------------------------------------------------------------------------- gyro <-> LiDAR

class GyroIntegrator(object):
    """Orientation of the IMU relative to its first sample, from bias-corrected gyro."""

    def __init__(self, t, gyro, bias):
        self.t = t
        w = gyro - bias
        dt = np.diff(t)
        wm = 0.5 * (w[1:] + w[:-1])
        dq = quat_from_rotvec(wm * dt[:, None])
        q = np.concatenate([[[1.0, 0.0, 0.0, 0.0]], dq], axis=0)
        self.q = prefix_quat(q)

    def R(self, tt):
        tt = np.clip(tt, self.t[0], self.t[-1])
        i = np.clip(np.searchsorted(self.t, tt) - 1, 0, self.t.size - 2)
        a = (tt - self.t[i]) / (self.t[i + 1] - self.t[i])
        q0, q1 = self.q[i], self.q[i + 1]
        dot = np.sum(q0 * q1, axis=1)
        q1 = np.where(dot[:, None] < 0, -q1, q1)
        qq = q0 * (1 - a)[:, None] + q1 * a[:, None]
        qq /= np.linalg.norm(qq, axis=1, keepdims=True)
        return quat_to_R(qq)


def lidar_rates(poses, stamps):
    Rr = np.einsum("nji,njk->nik", poses[:-1, :3, :3], poses[1:, :3, :3])
    w = log_batch(Rr) / np.diff(stamps)[:, None]
    return 0.5 * (stamps[1:] + stamps[:-1]), w


def coarse_lag(t_l, w_l, t_i, w_i, max_lag=0.3, step=0.001):
    """Lag (s) such that stamp_lidar = t_ref + lag maximises angular-speed correlation."""
    sl = np.linalg.norm(w_l, axis=1)
    si = np.linalg.norm(w_i, axis=1)
    # average the gyro over each LiDAR interval: box filter of 0.1 s
    k = 20
    c = np.cumsum(np.concatenate([[0.0], si]))
    box = (c[k:] - c[:-k]) / k
    tb = t_i[k // 2: k // 2 + box.size]
    lags = np.arange(-max_lag, max_lag + 1e-9, step)
    best, best_l = -2.0, 0.0
    for lag in lags:
        g = np.interp(t_l - lag, tb, box)
        cc = np.corrcoef(sl, g)[0, 1]
        if cc > best:
            best, best_l = cc, lag
    return best_l, best


def kabsch(a, b, w=None):
    """R minimising sum w |a_i - R b_i|^2."""
    if w is None:
        w = np.ones(a.shape[0])
    H = (b * w[:, None]).T @ a
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


# ----------------------------------------------------------------------------- main stages

def stage_lidar_imu(rig, log):
    seqs = rig["seqs"]
    # rests and gyro bias
    rest_bias = []
    for q in seqs:
        q.rests = rest_intervals(q)
        for a, b in q.rests:
            m = (q.imu_t >= a) & (q.imu_t <= b)
            if m.sum() > 50:
                rest_bias.append(np.median(q.gyro[m], axis=0))
    bg = np.median(np.array(rest_bias), axis=0)
    log("gyro bias %s from %d rests" % (np.round(bg, 5), len(rest_bias)))
    # LiDAR odometry
    for q in seqs:
        q.poses, q.outlier = lidar_odometry(rig, q)
        log("%s: odometry done, outlier median %.3f max %.3f" % (q.name, np.median(q.outlier[1:]),
                                                                q.outlier.max()))
    # coarse lag per sequence from angular speed
    lags = []
    for q in seqs:
        t_l, w_l = lidar_rates(q.poses, q.lidar_stamp)
        lag, cc = coarse_lag(t_l, w_l, q.imu_t, q.gyro - bg)
        lags.append(lag)
        q.lag0 = lag
        log("%s: coarse lidar lag %.4f s (corr %.3f)" % (q.name, lag, cc))
    # initial rotation by Kabsch on angular velocities
    A, B = [], []
    for q in seqs:
        t_l, w_l = lidar_rates(q.poses, q.lidar_stamp)
        gi = np.stack([np.interp(t_l - q.lag0, q.imu_t, q.gyro[:, i] - bg[i]) for i in range(3)], axis=1)
        good = (q.outlier[1:] < np.median(q.outlier) + 0.08) & (q.outlier[:-1] < np.median(q.outlier) + 0.08)
        A.append(w_l[good])
        B.append(gi[good])
    A, B = np.concatenate(A), np.concatenate(B)
    R_li = kabsch(A, B)
    for _ in range(3):
        r = np.linalg.norm(A - B @ R_li.T, axis=1)
        w = 1.0 / np.maximum(r / 0.02, 1.0)
        R_li = kabsch(A, B, w)
    prior_R = np.array(rig["prior"]["T_lidar_imu"], float)[:3, :3]
    log("initial R_li differs from prior by %.2f deg" % math.degrees(np.linalg.norm(logm(prior_R.T @ R_li))))
    # joint fit: R_li, lidar offset, drift on relative rotations over short spans
    integ = [GyroIntegrator(q.imu_t, q.gyro, bg) for q in seqs]
    pairs = []
    for qi, q in enumerate(seqs):
        n = len(q.lidar_stamp)
        for span in (1, 3):
            i = np.arange(0, n - span)
            j = i + span
            ok = (q.outlier[i] < np.median(q.outlier) + 0.08) & (q.outlier[j] < np.median(q.outlier) + 0.08)
            if span > 1:
                for s in range(1, span):
                    ok &= q.outlier[i + s] < np.median(q.outlier) + 0.08
            Rr = np.einsum("nji,njk->nik", q.poses[i, :3, :3], q.poses[j, :3, :3])
            pairs.append((qi, q.lidar_stamp[i[ok]], q.lidar_stamp[j[ok]], Rr[ok]))
    tmid = np.mean([np.mean(q.imu_t) for q in seqs])
    off0 = float(np.median(lags))

    def resid(x, keep=None):
        R = R_li @ expm(x[:3])
        off, drift = off0 + x[3], x[4] * 1e-6
        out = []
        for qi, sa, sb, Rr in pairs:
            ta = (sa - off) / (1 + drift)
            tb = (sb - off) / (1 + drift)
            Ri = np.einsum("nji,njk->nik", integ[qi].R(ta), integ[qi].R(tb))
            pred = R @ Ri @ R.T
            e = log_batch(np.einsum("nji,njk->nik", Rr, pred))
            out.append(e)
        return np.concatenate(out)

    def fun(x, mask):
        e = resid(x)
        return (e[mask]).ravel()

    x = np.zeros(5)
    mask = np.ones(resid(x).shape[0], bool)
    for rnd in range(3):
        sol = least_squares(fun, x, args=(mask,), loss="huber", f_scale=0.002, x_scale=[0.01, 0.01, 0.01, 0.005, 50.0])
        x = sol.x
        e = np.linalg.norm(resid(x), axis=1)
        thr = max(5.0 * np.median(e), 0.004)
        mask = e < thr
        log("lidar/imu fit round %d: median rot resid %.5f rad, kept %d/%d" % (rnd, np.median(e), mask.sum(), mask.size))
    R_li = R_li @ expm(x[:3])
    # the offset model here is stamp = t + off + drift*t' with t' = t; recover offset at t=0
    lidar_off = off0 + x[3]
    lidar_drift = x[4] * 1e-6
    rig["R_li"] = R_li
    rig["gyro_bias"] = bg
    rig["lidar_offset"] = lidar_off
    rig["lidar_drift"] = lidar_drift
    rig["integ"] = integ
    rig["pair_resid_mask"] = mask
    log("R_li vs prior %.3f deg; lidar offset %.5f s drift %.2f ppm" % (
        math.degrees(np.linalg.norm(logm(prior_R.T @ R_li))), lidar_off, lidar_drift * 1e6))
    return rig


def lidar_to_ref(rig, stamp):
    return (stamp - rig["lidar_offset"]) / (1.0 + rig["lidar_drift"])


def stage_accel(rig, log):
    """Accelerometer bias from rests; gravity direction from the LiDAR's floor plane."""
    R_li = rig["R_li"]
    ests = []
    for q in rig["seqs"]:
        tl = lidar_to_ref(rig, q.lidar_stamp)
        for a, b in q.rests:
            m = (q.imu_t >= a) & (q.imu_t <= b)
            li = np.nonzero((tl >= a) & (tl <= b))[0]
            if m.sum() < 50 or li.size == 0:
                continue
            pts, _, _ = load_scan(rig, q.lidar_files[li[li.size // 2]])
            n = floor_normal(pts)
            if n is None:
                continue
            f = np.mean(q.acc[m], axis=0)
            ests.append(f - G * (R_li.T @ n))
    if not ests:
        rig["accel_bias"] = np.zeros(3)
        log("no rest with a floor fit: accel bias 0")
        return
    E = np.array(ests)
    ba = np.median(E, axis=0)
    rig["accel_bias"] = ba
    log("accel bias %s from %d rests (spread %s)" % (np.round(ba, 4), len(ests), np.round(E.std(axis=0), 4)))


def floor_normal(pts):
    """Upward unit normal of the floor plane in the LiDAR frame (RANSAC + refit)."""
    cand = pts[(pts[:, 2] < -0.3) & (np.linalg.norm(pts[:, :2], axis=1) < 15.0)]
    if cand.shape[0] < 200:
        return None
    rs = np.random.default_rng(0)
    best, best_n, best_d = 0, None, None
    for _ in range(200):
        s3 = cand[rs.choice(cand.shape[0], 3, replace=False)]
        n = np.cross(s3[1] - s3[0], s3[2] - s3[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[2]) < 0.9:
            continue
        d = -n @ s3[0]
        inl = np.abs(cand @ n + d) < 0.03
        if inl.sum() > best:
            best, best_n, best_d = inl.sum(), n, d
    if best_n is None:
        return None
    inl = cand[np.abs(cand @ best_n + best_d) < 0.03]
    mu = inl.mean(axis=0)
    _, _, Vt = np.linalg.svd(inl - mu, full_matrices=False)
    n = Vt[2]
    if n[2] < 0:
        n = -n
    return n


# ----------------------------------------------------------------------------- faults

def camera_faults(rig, log):
    faults = []
    for q in rig["seqs"]:
        st = q.cam_stamp
        d = np.diff(st)
        d0 = np.median(d)
        q.cam_bad = np.zeros(st.size, bool)
        jumps = np.nonzero(np.abs(d - d0) > 0.012)[0]       # d[i] = st[i+1]-st[i]
        used = set()
        for a in jumps:
            if a in used:
                continue
            ja = d[a] - d0
            # the matching jump back has the opposite sign
            for b in jumps:
                if b > a and b not in used and abs((d[b] - d0) + ja) < 0.012:
                    used.update([a, b])
                    i0, i1 = a + 1, b
                    q.cam_bad[i0:i1 + 1] = True
                    faults.append({"sequence": q.name, "sensor": "camera", "type": "camera_time_jump",
                                   "start_s": float(st[i0]), "end_s": float(st[i1])})
                    break
        # frozen frames: bit-identical to the previous frame
        q.gray = [load_gray(f) for f in q.cam_files]
        same = np.zeros(st.size, bool)
        for i in range(1, st.size):
            same[i] = np.array_equal(q.gray[i], q.gray[i - 1])
        for a, b in runs(same):
            q.cam_bad[a:b + 1] = True
            faults.append({"sequence": q.name, "sensor": "camera", "type": "camera_frozen",
                           "start_s": float(st[a]), "end_s": float(st[b])})
    log("camera faults: %s" % [(f["sequence"], f["type"]) for f in faults])
    return faults


def runs(mask, min_len=1):
    out = []
    i, n = 0, mask.size
    while i < n:
        if mask[i]:
            j = i
            while j + 1 < n and mask[j + 1]:
                j += 1
            if j - i + 1 >= min_len:
                out.append((i, j))
            i = j + 1
        else:
            i += 1
    return out


def lidar_faults(rig, log):
    faults = []
    for q in rig["seqs"]:
        o = q.outlier.copy()
        base = np.median(o)
        bad = o > base + 0.08
        q.lidar_bad = bad
        for a, b in runs(bad):
            faults.append({"sequence": q.name, "sensor": "lidar", "type": "lidar_degraded",
                           "start_s": float(q.lidar_stamp[a]), "end_s": float(q.lidar_stamp[b])})
    log("lidar faults: %s" % [(f["sequence"], round(f["end_s"] - f["start_s"], 2)) for f in faults])
    return faults


def imu_faults(rig, log):
    """IMU samples whose gyro disagrees with the LiDAR's angular velocity."""
    faults = []
    R_li, bg = rig["R_li"], rig["gyro_bias"]
    for q in rig["seqs"]:
        tl = lidar_to_ref(rig, q.lidar_stamp)
        Rr = np.einsum("nji,njk->nik", q.poses[:-1, :3, :3], q.poses[1:, :3, :3])
        w_l = log_batch(Rr) / np.diff(tl)[:, None]
        tm = 0.5 * (tl[1:] + tl[:-1])
        ok = ~(q.lidar_bad[1:] | q.lidar_bad[:-1])
        # gyro averaged over each LiDAR interval, rotated into the LiDAR frame
        c = np.cumsum(np.concatenate([[np.zeros(3)], q.gyro - bg]), axis=0)
        k = 20
        box = (c[k:] - c[:-k]) / k
        tb = q.imu_t[k // 2: k // 2 + box.shape[0]]
        g = np.stack([np.interp(tm, tb, box[:, i]) for i in range(3)], axis=1) @ R_li.T
        r = np.linalg.norm(g - w_l, axis=1)
        # accelerometer direction against gravity seen by the LiDAR is checked too
        ca = np.cumsum(np.concatenate([[np.zeros(3)], q.acc - rig["accel_bias"]]), axis=0)
        ka = 200                                       # 1 s average: suspension vibration cancels
        abox = (ca[ka:] - ca[:-ka]) / ka
        ta = q.imu_t[ka // 2: ka // 2 + abox.shape[0]]
        acc_l = np.stack([np.interp(tm, ta, abox[:, i]) for i in range(3)], axis=1) @ R_li.T
        up = acc_l / np.maximum(np.linalg.norm(acc_l, axis=1, keepdims=True), 1e-9)
        tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
        bad = ok & ((r > 0.06) | (tilt > 25.0))
        # fill single-sample holes, drop isolated samples
        bad = smooth_mask(bad, 5)
        for a, b in runs(bad, min_len=3):
            if tm[b] - tm[a] < 0.8:
                continue
            ta, tb_ = tm[a] - 0.05, tm[b] + 0.05
            ia = int(np.searchsorted(q.imu_t, ta))
            ib = int(np.searchsorted(q.imu_t, tb_)) - 1
            ia, ib = max(ia, 0), min(ib, q.imu_t.size - 1)
            faults.append({"sequence": q.name, "sensor": "imu", "type": "imu_axis_swap",
                           "start_s": float(q.imu_t[ia]), "end_s": float(q.imu_t[ib])})
    log("imu faults: %s" % [(f["sequence"], round(f["start_s"], 2), round(f["end_s"], 2)) for f in faults])
    rig["imu_fault_iv"] = [(f["sequence"], f["start_s"], f["end_s"]) for f in faults]
    return faults


def smooth_mask(m, k):
    m = m.copy()
    idx = np.nonzero(m)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= k + 1:
            m[a:b] = True
    return m


def body_speed_from_lidar(rig, q):
    """Forward speed of the body origin at the LiDAR's reference times."""
    T_bl = rig["T_body_lidar"]
    R_bl, t_bl = T_bl[:3, :3], T_bl[:3, 3]
    t_lb = -R_bl.T @ t_bl                              # body origin in the LiDAR frame
    tl = lidar_to_ref(rig, q.lidar_stamp)
    P = q.poses[:, :3, 3] + np.einsum("nij,j->ni", q.poses[:, :3, :3], t_lb)
    V = np.gradient(P, tl, axis=0)
    xb = np.einsum("nij,j->ni", q.poses[:, :3, :3], R_bl.T @ np.array([1.0, 0.0, 0.0]))
    return tl, np.sum(V * xb, axis=1)


def odom_faults(rig, log):
    faults = []
    for q in rig["seqs"]:
        tl, vl = body_speed_from_lidar(rig, q)
        ok = ~q.lidar_bad
        vi = np.interp(q.odom_t, tl[ok], vl[ok])
        r = np.abs(q.odom_v - vi)
        bad = r > 0.25
        # where the LiDAR itself is degraded its speed is no reference
        tbad = tl[q.lidar_bad]
        if tbad.size:
            near = np.min(np.abs(q.odom_t[:, None] - tbad[None, :]), axis=1) < 0.4
            bad &= ~near
        bad = smooth_mask(bad, 30)
        for a, b in runs(bad, min_len=25):
            faults.append({"sequence": q.name, "sensor": "wheel_odom", "type": "odom_glitch",
                           "start_s": float(q.odom_t[a]), "end_s": float(q.odom_t[b])})
    log("odom faults: %s" % [(f["sequence"], round(f["start_s"], 2), round(f["end_s"], 2)) for f in faults])
    return faults


# ----------------------------------------------------------------------------- camera

def blur(img, sigma):
    import cv2
    if sigma <= 0:
        return img
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(img, (k, k), sigma)


def bilinear(img, u, v):
    H, W = img.shape
    u = np.clip(u, 0, W - 1.001)
    v = np.clip(v, 0, H - 1.001)
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    a = u - u0
    b = v - v0
    return ((1 - a) * (1 - b) * img[v0, u0] + a * (1 - b) * img[v0, u0 + 1]
            + (1 - a) * b * img[v0 + 1, u0] + a * b * img[v0 + 1, u0 + 1])


class EdgeImage(object):
    """Gradient-magnitude edge maps of one camera image at several blurs (cached)."""

    def __init__(self, gray):
        import cv2
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        g = np.sqrt(gx * gx + gy * gy)
        self.g = g / max(np.percentile(g, 99), 1e-6)
        self.maps = {}

    def img(self, sigma):
        if sigma not in self.maps:
            m = blur(self.g, sigma)
            self.maps[sigma] = m / max(m.max(), 1e-9)
        return self.maps[sigma]


class AlignFrame(object):
    """A camera edge image and LiDAR edge points (LiDAR frame at the camera's time)."""

    def __init__(self, edge, pts, w):
        self.edge = edge
        self.pts = pts
        self.w = w

    def img(self, sigma):
        return self.edge.img(sigma)


def ncc_score(frames, T_lc, K, W, H, sigma, margin=3):
    """Mean over frames of the weighted mean edge strength at projected LiDAR edge points."""
    R, t = T_lc[:3, :3], T_lc[:3, 3]
    tot, cnt = 0.0, 0
    for fr in frames:
        pc = (fr.pts - t) @ R
        z = pc[:, 2]
        ok = z > 0.5
        zs = np.where(ok, z, 1.0)
        u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
        v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
        ok &= (u > margin) & (u < W - 1 - margin) & (v > margin) & (v < H - 1 - margin)
        if ok.sum() < 50:
            continue
        g = bilinear(fr.img(sigma), u[ok], v[ok])
        w = fr.w[ok]
        tot += float(g @ w) / float(w.sum())
        cnt += 1
    return tot / max(cnt, 1), cnt


def local_max(f, x0, steps, penalty=None, maxfev=3000):
    """Maximise f near x0 with Powell, line searches starting at the given step sizes."""
    steps = np.asarray(steps, float)

    def g(z):
        x = x0 + z * steps
        v = -f(x)
        if penalty is not None:
            v += penalty(x)
        return v

    res = minimize(g, np.zeros(len(steps)), method="Powell",
                   options={"xtol": 1e-3, "ftol": 1e-7, "maxfev": maxfev, "direc": np.eye(len(steps))})
    return x0 + res.x * steps, -res.fun


def trans_penalty(x, bound=0.10):
    t = np.abs(np.asarray(x[3:6]))
    return float(np.sum(np.clip(t - bound, 0.0, None) ** 2)) * 100.0


def perturb(T, x):
    """Left-perturb rotation (x[:3], rad) and translation (x[3:], m) of T_lidar_camera."""
    return make_T(expm(x[:3]) @ T[:3, :3], T[:3, 3] + x[3:6])


def camera_frame_at(rig, q, ci, t_cam_ref, max_dt=0.06):
    """AlignFrame for camera frame ci at reference time t_cam_ref, or None."""
    tl = lidar_to_ref(rig, q.lidar_stamp)
    li = int(np.argmin(np.abs(tl - t_cam_ref)))
    if abs(tl[li] - t_cam_ref) > max_dt or q.lidar_bad[li]:
        return None
    pts, refl, _ = load_scan_cached(rig, q, li)
    edge = edge_cached(rig, q, ci)
    # carry the scan from its time to the camera's: rotation from the gyro, translation from odometry
    qi = rig["seqs"].index(q)
    integ = rig["integ"][qi]
    R_li = rig["R_li"]
    Ri = (integ.R(np.array([t_cam_ref]))[0]).T @ integ.R(np.array([tl[li]]))[0]
    R_rel = R_li @ Ri @ R_li.T                           # R_{L(tc) L(tl)}
    pos = q.poses[:, :3, 3]
    p_c = np.array([np.interp(t_cam_ref, tl, pos[:, k]) for k in range(3)])
    p_l = pos[li]
    # odometry orientation at the camera time (nearest scan is fine for this term)
    R_wc = q.poses[li, :3, :3] @ R_rel.T
    t_rel = R_wc.T @ (p_l - p_c)
    return AlignFrame(edge, pts @ R_rel.T + t_rel, refl)


def load_scan_cached(rig, q, li):
    cache = rig.setdefault("_scan_cache", {})
    key = (q.name, li)
    if key not in cache:
        pts, w = scan_edges(rig, q.lidar_files[li])
        cache[key] = (pts, w, None)
    return cache[key]


def edge_cached(rig, q, ci):
    cache = rig.setdefault("_edge_cache", {})
    key = (q.name, ci)
    if key not in cache:
        cache[key] = EdgeImage(q.gray[ci])
    return cache[key]


def stage_camera(rig, log):
    K, W, H = rig["K"], rig["W"], rig["H"]
    T0 = np.array(rig["prior"]["T_lidar_camera"], float)
    bound = 0.25
    # frames at rest: camera time does not matter
    rest_frames = []
    for q in rig["seqs"]:
        for a, b in q.rests:
            lo, hi = a + bound, b - bound
            if hi <= lo:
                continue
            cand = np.nonzero((q.cam_stamp >= lo) & (q.cam_stamp <= hi) & ~q.cam_bad)[0]
            if cand.size == 0:
                continue
            ci = cand[cand.size // 2]
            fr = camera_frame_at(rig, q, ci, 0.5 * (a + b), max_dt=0.5 * (b - a))
            if fr is not None:
                rest_frames.append(fr)
    log("camera: %d rest frames" % len(rest_frames))
    # coarse rotation grid at heavy blur; keep the best few distinct candidates
    grid = np.radians(np.arange(-12.0, 12.01, 2.0))
    cands = []
    for ax in grid:
        for ay in grid:
            for az in grid:
                x = np.array([ax, ay, az, 0.0, 0.0, 0.0])
                sc, _ = ncc_score(rest_frames, perturb(T0, x), K, W, H, 6.0)
                cands.append((sc, x))
    cands.sort(key=lambda c: -c[0])
    starts = []
    for sc, x in cands:
        if all(np.linalg.norm(x[:3] - y[:3]) > math.radians(3.0) for _, y in starts):
            starts.append((sc, x))
        if len(starts) >= 5:
            break
    log("camera coarse: best %s" % [(round(sc, 3), list(np.round(np.degrees(x[:3]), 0))) for sc, x in starts])
    # rotation only while the blur is heavy: translation trades against rotation there
    coarse = []
    for sc0, x in starts:
        for sigma, st in ((6.0, 1.0), (3.0, 0.5)):
            f = lambda r: ncc_score(rest_frames, perturb(T0, np.concatenate([r, x[3:]])), K, W, H, sigma)[0]
            r, _ = local_max(f, x[:3], [math.radians(st)] * 3, maxfev=1500)
            x = np.concatenate([r, x[3:]])
        coarse.append(x)
    # heavy blur can pull the rotation a few degrees off: search a wide rotation grid at 1.5 px
    seeds = []
    for x in coarse:
        if any(np.linalg.norm(x[:3] - y[:3]) < math.radians(1.0) for y in seeds):
            continue
        seeds.append(x)
    grid_pts = []
    for dx in np.radians(np.arange(-3.0, 3.01, 1.0)):
        for dy in np.radians(np.arange(-4.0, 4.01, 0.5)):
            for dz in np.radians(np.arange(-4.0, 4.01, 0.5)):
                grid_pts.append(np.array([dx, dy, dz]))
    fine_starts = []
    for x in seeds[:2]:
        for dr in grid_pts:
            y = np.concatenate([x[:3] + dr, x[3:]])
            fine_starts.append((ncc_score(rest_frames, perturb(T0, y), K, W, H, 1.5)[0], y))
    fine_starts.sort(key=lambda c: -c[0])
    picked = []
    for sc, y in fine_starts:
        if all(np.linalg.norm(y[:3] - z[:3]) > math.radians(1.0) for _, z in picked):
            picked.append((sc, y))
        if len(picked) >= 3:
            break
    log("camera 1.5 px grid: %s" % [(round(sc, 4), list(np.round(np.degrees(y[:3]), 1))) for sc, y in picked])
    best = None
    for sc0, x in picked:
        for sigma, st, tt in ((1.5, 0.2, 0.01), (0.8, 0.1, 0.005)):
            f = lambda y: ncc_score(rest_frames, perturb(T0, y), K, W, H, sigma)[0]
            x, sc = local_max(f, x, [math.radians(st)] * 3 + [tt] * 3, penalty=trans_penalty, maxfev=3000)
        log("camera start (coarse %.3f) -> score %.4f at %s deg, %s m" % (
            sc0, sc, np.round(np.degrees(x[:3]), 2), np.round(x[3:], 3)))
        if best is None or sc > best[0]:
            best = (sc, x)
    x = best[1]
    T_lc = perturb(T0, x)
    rig["T_lc_rest"] = T_lc
    # camera clock from moving frames: per-window offset scans, then a line fit
    windows = []
    for q in rig["seqs"]:
        n = q.cam_stamp.size
        for w0 in range(0, n - 20, 20):
            idx = np.arange(w0, min(w0 + 20, n), 2)
            idx = idx[~q.cam_bad[idx]]
            if idx.size < 5:
                continue
            windows.append((q, idx))
    est = []
    for q, idx in windows:
        def score(off):
            frs = []
            for ci in idx:
                fr = camera_frame_at(rig, q, ci, q.cam_stamp[ci] - off)
                if fr is not None:
                    frs.append(fr)
            return ncc_score(frs, T_lc, K, W, H, 1.5)[0] if len(frs) >= 3 else np.nan
        offs = np.arange(-0.25, 0.2501, 0.01)
        sc = np.array([score(o) for o in offs])
        if np.mean(np.isfinite(sc)) < 0.7:
            continue
        o1 = offs[int(np.nanargmax(sc))]
        offs2 = np.arange(o1 - 0.012, o1 + 0.0121, 0.002)
        sc2 = np.array([score(o) for o in offs2])
        if not np.all(np.isfinite(sc2)):
            continue
        k = int(np.argmax(sc2))
        o2 = offs2[k]
        if 0 < k < len(offs2) - 1:
            a_, b_, c_ = sc2[k - 1], sc2[k], sc2[k + 1]
            den = a_ - 2 * b_ + c_
            if den < 0:
                o2 = o2 + 0.002 * 0.5 * (a_ - c_) / den
        sharp = float(sc2.max() - np.nanmedian(sc))
        est.append((float(np.mean(q.cam_stamp[idx])), o2, sharp))
    E = np.array(est)
    log("camera windows (t, offset, sharpness): %s" % [(round(a, 1), round(b, 4), round(c, 3)) for a, b, c in est])
    # robust line through the window estimates: stamp - t_ref = off + drift * t_ref (t_ref ~ stamp).
    # RANSAC over pairs of windows, then a weighted fit on the inliers.
    w = np.clip(E[:, 2], 0.0, 0.15)
    usable = np.nonzero(w > 0.02)[0]
    best_in = None
    for a_ in usable:
        for b_ in usable:
            if E[b_, 0] - E[a_, 0] < 20.0:
                continue
            d_ = (E[b_, 1] - E[a_, 1]) / (E[b_, 0] - E[a_, 0])
            if abs(d_) > 1e-3:
                continue
            o_ = E[a_, 1] - d_ * E[a_, 0]
            inl = usable[np.abs(E[usable, 1] - (o_ + d_ * E[usable, 0])) < 0.004]
            if best_in is None or w[inl].sum() > w[best_in].sum():
                best_in = inl
    if best_in is None or best_in.size < 2:
        best_in = usable
    keep = np.zeros(len(E), bool)
    keep[best_in] = True
    for _ in range(3):
        A = np.stack([np.ones(keep.sum()), E[keep, 0]], axis=1)
        coef, *_ = np.linalg.lstsq(A * w[keep, None], E[keep, 1] * w[keep], rcond=None)
        r = E[:, 1] - (coef[0] + coef[1] * E[:, 0])
        keep = (w > 0.02) & (np.abs(r) < 0.004)
    rig["cam_offset"] = float(coef[0])
    rig["cam_drift"] = float(coef[1])
    log("camera clock: offset %.5f s drift %.1f ppm from %d/%d windows" % (coef[0], coef[1] * 1e6, keep.sum(), len(est)))
    # joint refinement: extrinsic and clock on rest + moving frames
    sel = []
    for q in rig["seqs"]:
        idx = np.arange(0, q.cam_stamp.size, 4)
        sel += [(q, ci) for ci in idx if not q.cam_bad[ci]]

    def frames_for(off, drift):
        frs = list(rest_frames)
        for q, ci in sel:
            fr = camera_frame_at(rig, q, ci, (q.cam_stamp[ci] - off) / (1 + drift))
            if fr is not None:
                frs.append(fr)
        return frs

    # with the clock known, moving frames join the rest frames: search the rotation grid again
    # on many more views, which a wrong optimum of the rest frames alone rarely survives
    co, cdr = rig["cam_offset"], rig["cam_drift"]
    many = list(rest_frames)
    for q, ci in sel[::2]:
        fr = camera_frame_at(rig, q, ci, (q.cam_stamp[ci] - co) / (1 + cdr))
        if fr is not None:
            many.append(fr)
    grid2 = []
    for dr in grid_pts:
        y0 = np.concatenate([x[:3] + dr, x[3:]])
        grid2.append((ncc_score(many, perturb(T0, y0), K, W, H, 1.5)[0], y0))
    grid2.sort(key=lambda c: -c[0])
    picked2 = []
    for sc, y0 in grid2:
        if all(np.linalg.norm(y0[:3] - z[:3]) > math.radians(1.0) for _, z in picked2):
            picked2.append((sc, y0))
        if len(picked2) >= 3:
            break
    best2 = None
    for _, y0 in picked2:
        for sigma, st, tt in ((1.5, 0.2, 0.01), (0.8, 0.1, 0.005)):
            f = lambda yy: ncc_score(many, perturb(T0, yy), K, W, H, sigma)[0]
            y0, sc = local_max(f, y0, [math.radians(st)] * 3 + [tt] * 3, penalty=trans_penalty, maxfev=2000)
        if best2 is None or sc > best2[0]:
            best2 = (sc, y0)
    log("camera many-frame search (%d frames): %s -> score %.4f at %s deg" % (
        len(many), [(round(sc, 4), list(np.round(np.degrees(y0[:3]), 1))) for sc, y0 in picked2],
        best2[0], np.round(np.degrees(best2[1][:3]), 2)))
    x = best2[1]

    y = np.concatenate([x, [rig["cam_offset"], rig["cam_drift"] * 1e3]])

    def fj(y):
        frs = frames_for(y[6], y[7] * 1e-3)
        return ncc_score(frs, perturb(T0, y[:6]), K, W, H, 0.8)[0]

    y, best_j = local_max(fj, y, [math.radians(0.05)] * 3 + [0.003] * 3 + [0.001, 0.01],
                          penalty=trans_penalty, maxfev=2500)

    class _R(object):
        pass
    res = _R()
    res.fun = -best_j
    rig["T_lidar_camera"] = perturb(T0, y[:6])
    rig["cam_offset"] = float(y[6])
    rig["cam_drift"] = float(y[7] * 1e-3)
    log("camera joint: ncc %.4f, offset %.5f drift %.1f ppm" % (-res.fun, y[6], y[7] * 1e3))


def write_outputs(rig, out_dir, faults):
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    T_li = np.array(rig["prior"]["T_lidar_imu"], float)
    T_li[:3, :3] = rig["R_li"]
    doc = {
        "T_lidar_camera": [[float(v) for v in row] for row in rig["T_lidar_camera"]],
        "T_lidar_imu": [[float(v) for v in row] for row in T_li],
        "time": {"camera": {"offset_s": float(rig["cam_offset"]), "drift_ppm": float(rig["cam_drift"] * 1e6)},
                 "lidar": {"offset_s": float(rig["lidar_offset"]), "drift_ppm": float(rig["lidar_drift"] * 1e6)}},
        "imu": {"gyro_bias_rps": [float(v) for v in rig["gyro_bias"]],
                "accel_bias_mps2": [float(v) for v in rig["accel_bias"]]},
    }
    with open(os.path.join(out_dir, "calibration.yaml"), "w") as fh:
        yaml.safe_dump(doc, fh, sort_keys=False)
    with open(os.path.join(out_dir, "fault_report.json"), "w") as fh:
        json.dump({"faults": faults}, fh, indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)
    log = lambda s: print("[calibrate] " + s, file=sys.stderr, flush=True)
    rig = load_rig(args.input)
    stage_lidar_imu(rig, log)
    faults = []
    faults += lidar_faults(rig, log)
    stage_accel(rig, log)
    faults += imu_faults(rig, log)
    faults += camera_faults(rig, log)
    faults += odom_faults(rig, log)
    stage_camera(rig, log)
    write_outputs(rig, args.output, faults)
    log("done")


if __name__ == "__main__":
    main()
