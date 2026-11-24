"""Calibration of camera/LiDAR/IMU extrinsics, clocks and IMU biases from driving logs."""
import argparse
import hashlib
import json
import os
import pickle
import sys
import time

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rig_io  # noqa: E402
import lo  # noqa: E402
import faults as flt  # noqa: E402
import edges  # noqa: E402

G = 9.80665
T_START = time.process_time()


def log(*a):
    print("[%6.1fs]" % (time.process_time() - T_START), *a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- statics / gyro bias
def static_mask(seq, relax=1.0):
    """IMU samples during which the robot stands still."""
    t = seq.imu_stamps
    v = np.interp(t, seq.odom_stamps, np.abs(seq.odom_v))
    w = np.interp(t, seq.odom_stamps, np.abs(seq.odom_yaw_rate))
    g = seq.gyro
    # moving std of gyro norm over 0.25 s
    n = 50
    k = np.ones(n) / n
    gn = g - np.array([np.convolve(g[:, i], k, 'same') for i in range(3)]).T
    sd = np.sqrt(np.convolve((gn ** 2).sum(1), k, 'same'))
    m = (v < 0.05 * relax) & (w < 0.05 * relax) & (sd < 0.012 * relax)
    # erode by 0.25 s
    me = np.convolve(m.astype(float), np.ones(101) / 101, 'same') > 0.999
    return me


def cumint(t, x):
    c = np.zeros_like(x)
    c[1:] = np.cumsum(0.5 * (x[1:] + x[:-1]) * np.diff(t)[:, None], axis=0)
    return c


def interval_integrals(t, C, a, b):
    """integral over [a_k, b_k] of the signal with cumulative integral C sampled at t."""
    ia = np.array([np.interp(a, t, C[:, i]) for i in range(C.shape[1])]).T
    ib = np.array([np.interp(b, t, C[:, i]) for i in range(C.shape[1])]).T
    return ib - ia


def fit_offset(t_imu, C, s0, s1, y, lo_, hi_, step=0.0005, shift=0.0):
    """find d (s - t) such that y_k ~ W^T int_{s0-d}^{s1-d} gyro (y: n or n x m); returns d, W, curve."""
    y2 = y.reshape(len(y), -1)
    ds = np.arange(lo_, hi_ + step / 2, step)
    sse = []
    ws = []
    for d in ds:
        X = interval_integrals(t_imu, C, s0 - d + shift, s1 - d + shift)
        w, res, _, _ = np.linalg.lstsq(X, y2, rcond=None)
        r = np.linalg.norm(y2 - X @ w, axis=1)
        sc = 1.4826 * np.median(r) + 1e-6
        rr = np.minimum(r, 3 * sc)
        sse.append(np.sum(rr ** 2))
        ws.append(w)
    sse = np.array(sse)
    i = int(np.argmin(sse))
    d = ds[i]
    if 0 < i < len(ds) - 1:
        a, b, c = sse[i - 1], sse[i], sse[i + 1]
        den = a - 2 * b + c
        if den > 0:
            d = ds[i] + 0.5 * step * (a - c) / den
    return d, ws[i], sse


def refine_offset(t_imu, C, s0, s1, y, d0, half=0.004, step=0.0001, shift=0.0):
    """fine search with robust weights fixed from the coarse solution."""
    y2 = y.reshape(len(y), -1)
    X = interval_integrals(t_imu, C, s0 - d0 + shift, s1 - d0 + shift)
    w, _, _, _ = np.linalg.lstsq(X, y2, rcond=None)
    r = np.linalg.norm(y2 - X @ w, axis=1)
    sc = 1.4826 * np.median(r) + 1e-6
    keep = r < 3 * sc
    return fit_offset(t_imu, C, s0[keep], s1[keep], y[keep], d0 - half, d0 + half, step, shift) + (keep,)


def kabsch(X, Y):
    """R minimising sum |Y_k - R X_k|^2."""
    H = X.T @ Y
    U, S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


# ---------------------------------------------------------------- camera rotations
def camera_rotations(rig, seq, bad):
    import cv2
    n = len(seq.cam_stamps)
    K, D = rig.K, rig.dist
    out = {}
    prev = None
    prev_pts = None
    for k in range(n):
        img = cv2.imread(os.path.join(seq.path, "camera", "frames", seq.cam_files[k]), cv2.IMREAD_GRAYSCALE)
        if prev is not None and not bad[k] and not bad[k - 1]:
            p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 12)
            if p0 is not None and len(p0) > 30:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, img, p0, None, winSize=(21, 21), maxLevel=3)
                st = st.ravel() == 1
                if st.sum() > 30:
                    a = cv2.undistortPoints(p0[st], K, D)
                    b = cv2.undistortPoints(p1[st], K, D)
                    flow = np.linalg.norm((b - a).reshape(-1, 2), axis=1)
                    if np.median(flow) * K[0, 0] < 0.3:
                        out[k - 1] = np.zeros(3)
                    else:
                        E, msk = cv2.findEssentialMat(a, b, np.eye(3), cv2.RANSAC, 0.999, 1.0 / K[0, 0])
                        if E is not None and E.shape == (3, 3):
                            _, R, tt, msk2 = cv2.recoverPose(E, a, b, np.eye(3), mask=msk)
                            # R maps frame k-1 coordinates to frame k: x_k = R x_{k-1} + t
                            out[k - 1] = lo.logSO3(R.T)   # rotation of frame k in frame k-1
        prev = img
    return out


def frame_hashes(seq):
    hs = []
    for f in seq.cam_files:
        with open(os.path.join(seq.path, "camera", "frames", f), "rb") as fh:
            hs.append(hashlib.md5(fh.read()).hexdigest())
    return hs


def swap_free(s, sw, a, b):
    """intervals [a, b] (reference time) touching no flagged IMU sample."""
    c = np.concatenate([[0], np.cumsum(sw)])
    ia = np.searchsorted(s.imu_stamps, a - 0.1)
    ib = np.searchsorted(s.imu_stamps, b + 0.1)
    return (c[ib] - c[ia]) == 0


def lidar_offsets(seqs, rels, gb, up_l, shift, lid_bad, swap):
    ds, ts, ws = [], [], []
    for si, s in enumerate(seqs):
        rel = rels[si]
        ks = np.array([k for k in sorted(rel) if not (lid_bad[si][k] or lid_bad[si][k + 1])])
        ks = ks[swap_free(s, swap[si], s.scan_stamps[ks] - 0.1, s.scan_stamps[ks + 1] + 0.1)]
        y = np.array([lo.logSO3(rel[k][:3, :3]) @ up_l for k in ks])
        s0 = s.scan_stamps[ks]
        s1 = s.scan_stamps[ks + 1]
        C = cumint(s.imu_stamps, s.gyro - gb)
        mv = np.abs(y) > 0.002 if len(ks) else np.zeros(0, bool)
        if mv.sum() < 20:
            log("lidar seq", si, "too little rotation for the clock")
            ds.append(np.nan)
            ts.append(np.mean(s.scan_stamps))
            ws.append(np.full(3, np.nan))
            continue
        d, w, sse = fit_offset(s.imu_stamps, C, s0[mv], s1[mv], y[mv], -0.4, 0.4, 0.001, shift)
        d, w, sse, keep = refine_offset(s.imu_stamps, C, s0[mv], s1[mv], y[mv], d, shift=shift)
        ds.append(d)
        ts.append(np.mean(s.scan_stamps))
        ws.append(w.ravel())
    return np.array(ds), np.array(ts), ws


def camera_fit(seqs, crs, cam_fix, gb, swap):
    cam_d, cam_t, cam_X, cam_Y, cam_iv = [], [], [], [], []
    for si, s in enumerate(seqs):
        cr = crs[si]
        ks = np.array(sorted(cr), int)
        Y = np.array([cr[k] for k in ks]).reshape(-1, 3)
        ok = np.linalg.norm(Y, axis=1) < 0.15
        ks, Y = ks[ok], Y[ok]
        s0 = cam_fix[si][ks]
        s1 = cam_fix[si][ks + 1]
        C = cumint(s.imu_stamps, s.gyro - gb)
        mv = (np.linalg.norm(Y, axis=1) > 0.002) & swap_free(s, swap[si], s0 - 0.25, s1 + 0.25)
        if mv.sum() < 20:
            log("camera seq", si, "too little rotation for the clock")
            cam_iv.append((np.zeros(0), np.zeros(0), np.zeros((0, 3))))
            continue
        d, w, sse = fit_offset(s.imu_stamps, C, s0[mv], s1[mv], Y[mv], -0.4, 0.4, 0.001)
        d, w, sse, keep = refine_offset(s.imu_stamps, C, s0[mv], s1[mv], Y[mv], d)
        X = interval_integrals(s.imu_stamps, C, s0[mv][keep] - d, s1[mv][keep] - d)
        cam_X.append(X)
        cam_Y.append(Y[mv][keep])
        cam_iv.append((s0 - d, s1 - d, Y))
        log("camera seq", si, "offset", d, "n", keep.sum())
        cam_d.append(d)
        cam_t.append(np.mean(s.cam_stamps))
    return np.array(cam_d), np.array(cam_t), cam_X, cam_Y, cam_iv


def clock_fit(d, t):
    """offset and drift of d = o + drift * t; with a single sequence the drift is taken as zero."""
    if len(d) < 2 or np.ptp(t) < 1.0:
        return float(np.mean(d)), 0.0
    A = np.stack([np.ones_like(t), t], 1)
    (o, dr), _, _, _ = np.linalg.lstsq(A, d, rcond=None)
    return o, dr


def align_up(R_li, up_i, up_l):
    """smallest rotation of R_li (IMU in LiDAR) such that R_li @ up_i = up_l."""
    a = R_li @ up_i
    v = np.cross(a, up_l)
    ang = np.arcsin(np.clip(np.linalg.norm(v), 0, 1))
    if np.linalg.norm(v) < 1e-12:
        return R_li
    return lo.expSO3(v / np.linalg.norm(v) * ang) @ R_li


def write_calib(out, T_lc, T_li, co_off, co_dr, lo_off, lo_dr, gb, ab):
    cal = {
        "T_lidar_camera": np.asarray(T_lc).tolist(),
        "T_lidar_imu": np.asarray(T_li).tolist(),
        "time": {"camera": {"offset_s": float(co_off), "drift_ppm": float(co_dr * 1e6)},
                 "lidar": {"offset_s": float(lo_off), "drift_ppm": float(lo_dr * 1e6)}},
        "imu": {"gyro_bias_rps": np.asarray(gb).tolist(), "accel_bias_mps2": np.asarray(ab).tolist()},
    }
    with open(os.path.join(out, "calibration.yaml"), "w") as fh:
        yaml.safe_dump(cal, fh, sort_keys=False)


def write_faults(out, faults):
    with open(os.path.join(out, "fault_report.json"), "w") as fh:
        json.dump({"faults": flt.best_per_type(faults)[:50]}, fh, indent=1)


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cache", default=None)
    a = ap.parse_args()
    rig = rig_io.Rig(a.input)
    prior = rig.prior
    T_lc0 = np.array(prior["T_lidar_camera"], float)
    T_li0 = np.array(prior["T_lidar_imu"], float)
    names = rig.sequence_names
    seqs = [rig.sequence(n) for n in names]
    faults = []
    R_bl = rig.T_body_lidar[:3, :3]
    up_l = R_bl.T @ np.array([0, 0, 1.0])

    # ---- statics, gyro bias
    # IMU axis swaps seen by the accelerometer (gravity on a wrong axis) are kept out of the statics
    up_i0 = T_li0[:3, :3].T @ up_l
    swap = [flt.imu_axis_swap(names[si], s, ([], [], []), np.eye(3), np.zeros(3), np.zeros(3), up_i0)[1]
            for si, s in enumerate(seqs)]
    nsw = [np.convolve(sw.astype(float), np.ones(101), 'same') == 0 for sw in swap]
    stat = [static_mask(s) & ok for s, ok in zip(seqs, nsw)]
    for relax in (2.0, 4.0):
        if sum(m.sum() for m in stat) >= 400:
            break
        stat = [static_mask(s, relax) & ok for s, ok in zip(seqs, nsw)]
    if sum(m.sum() for m in stat) >= 50:
        gb = np.concatenate([s.gyro[m] for s, m in zip(seqs, stat)]).mean(0)
    else:
        gb = np.median(np.concatenate([s.gyro for s in seqs]), axis=0)
    log("gyro bias", gb, "static samples", sum(m.sum() for m in stat))

    # ---- camera bookkeeping: frozen frames, time jumps; LiDAR degradation
    cam_bad = []
    cam_fix = []
    lid_bad = []
    for si, s in enumerate(seqs):
        hs = frame_hashes(s)
        bad = np.zeros(len(hs), bool)
        for k in range(1, len(hs)):
            if hs[k] == hs[k - 1]:
                bad[k] = True
        faults += flt.camera_frozen(names[si], s.cam_stamps, hs)
        fj, badj, fixj = flt.camera_time_jump(names[si], s.cam_stamps)
        faults += fj
        cam_bad.append(bad | badj)
        cam_fix.append(fixj)
        fl, badl, _ = flt.lidar_degraded(names[si], rig, s)
        faults += fl
        lid_bad.append(badl)
        faults += flt.odom_steps(names[si], s)
    log("faults so far", faults)
    write_faults(a.output, faults)

    # ---- IMU axis swaps (gravity on a wrong axis), camera rotations and clock
    crs = [camera_rotations(rig, s, cam_bad[si]) for si, s in enumerate(seqs)]
    log("camera rotations done")
    for it in range(2):
        cam_d, cam_t, cam_X, cam_Y, cam_iv = camera_fit(seqs, crs, cam_fix, gb, swap)
        co_off, co_dr = clock_fit(cam_d, cam_t)
        log("camera offsets", cam_d, "clock", co_off, co_dr * 1e6)
        R_ci = kabsch(np.concatenate(cam_X), np.concatenate(cam_Y))
        log("R_cam_imu", lo.logSO3(R_ci))
        if it == 0:
            swap = [flt.imu_axis_swap(names[si], s, cam_iv[si], R_ci, gb, np.zeros(3), up_i0)[1] | swap[si]
                    for si, s in enumerate(seqs)]
            log("imu swap samples", [int(m.sum()) for m in swap])

    # ---- LiDAR odometry, pass 1 (constant-velocity deskew)
    cache = a.cache
    if cache and os.path.exists(cache + ".1"):
        rels1 = pickle.load(open(cache + ".1", "rb"))
    else:
        rels1 = [lo.odometry(rig, s, voxel=0.4) for s in seqs]
        if cache:
            pickle.dump(rels1, open(cache + ".1", "wb"))
    log("lidar odometry pass 1 done")
    lid_d, lid_t, lid_w = lidar_offsets(seqs, rels1, gb, up_l, shift=0.5 * rig.manifest["lidar"]["sweep_s"], lid_bad=lid_bad, swap=swap)
    log("lidar pass-1 offsets", lid_d)
    up_i = np.nanmean([w / np.linalg.norm(w) for w in lid_w], axis=0)
    lid_d = np.where(np.isfinite(lid_d), lid_d, np.nanmedian(lid_d))
    R_li1 = align_up(T_li0[:3, :3], up_i / np.linalg.norm(up_i), up_l)

    # ---- pass 2: gyro deskew
    if cache and os.path.exists(cache + ".2"):
        rels = pickle.load(open(cache + ".2", "rb"))
    else:
        rels = []
        for si, s in enumerate(seqs):
            C = cumint(s.imu_stamps, (s.gyro - gb) @ R_li1.T)
            colt = rig.column_delay_s
            d0 = lid_d[si]

            sw = swap[si]

            def rotfun(k, s=s, C=C, d0=d0, sw=sw):
                t0 = s.scan_stamps[k] - d0
                if not swap_free(s, sw, np.array([t0]), np.array([t0 + rig.manifest["lidar"]["sweep_s"]]))[0]:
                    return None
                c0 = np.array([np.interp(t0, s.imu_stamps, C[:, i]) for i in range(3)])
                return np.array([np.interp(t0 + colt, s.imu_stamps, C[:, i]) for i in range(3)]).T - c0
            rels.append(lo.odometry(rig, s, rotfun=rotfun))
        if cache:
            pickle.dump(rels, open(cache + ".2", "wb"))
    log("lidar odometry pass 2 done")
    lid_d, lid_t, lid_w = lidar_offsets(seqs, rels, gb, up_l, shift=0.0, lid_bad=lid_bad, swap=swap)
    fin = np.isfinite(lid_d)
    lo_off, lo_dr = clock_fit(lid_d[fin], lid_t[fin])
    log("lidar offsets", lid_d, "clock", lo_off, lo_dr * 1e6)
    up_i = np.nanmean([w / np.linalg.norm(w) for w in lid_w], axis=0)
    up_i /= np.linalg.norm(up_i)
    log("imu up axis", up_i, [w.ravel() for w in lid_w])
    if sum(m.sum() for m in stat) >= 50:
        acc_st = np.concatenate([s.accel[m] for s, m in zip(seqs, stat)]).mean(0)
    else:
        acc_st = np.median(np.concatenate([s.accel for s in seqs]), axis=0)
    T_li = T_li0.copy()
    T_li[:3, :3] = align_up(T_li0[:3, :3], up_i, up_l)
    T_lc = T_lc0.copy()
    T_lc[:3, :3] = T_li[:3, :3] @ R_ci.T
    write_calib(a.output, T_lc, T_li, co_off, co_dr, lo_off, lo_dr, gb, acc_st - G * up_i)

    # ---- camera-LiDAR extrinsic from edges at rest
    cam_ref = [(s.cam_stamps - co_off) / (1 + co_dr) for s in seqs]
    lid_ref = [(s.scan_stamps - lo_off) / (1 + lo_dr) for s in seqs]
    try:
        views = edges.build_views(rig, seqs, stat, cam_ref, lid_ref, T_lc0, log=log,
                                  cam_bad=cam_bad, lid_bad=lid_bad)
        R_ic = R_ci.T
        mode = os.environ.get("EDGE_MODE", "axis")
        if mode == "full":
            T_lc = edges.solve(rig, views, T_lc0, log=log)
        else:
            R_li_t = align_up(T_li0[:3, :3], up_i, up_l)
            Tb = T_lc0.copy()
            Tb[:3, :3] = R_li_t @ R_ic
            T_lc = edges.solve(rig, views, Tb, log=log, axis=up_l, grid_deg=12.0,
                                fix_t=os.environ.get("FIX_T", "0") == "1")
        if np.linalg.norm(T_lc[:3, 3] - T_lc0[:3, 3]) > 0.2:
            log("static solution moved too far, keeping the prior translation")
            T_lc[:3, 3] = T_lc0[:3, 3]
        log("T_lidar_camera (static)", T_lc)
    except Exception as e:  # keep the IMU-based rotation and the prior translation
        log("static edge alignment failed", repr(e))
    try:
        if os.environ.get("MOVING", "1") == "1":
            mviews = []
            for si, s in enumerate(seqs):
                C_l = cumint(s.imu_stamps, (s.gyro - gb) @ (T_lc[:3, :3] @ R_ci).T)
                traj = edges.Traj(lid_ref[si], rels[si], s.imu_stamps, C_l)
                cam_ok = ~cam_bad[si]
                cam_ok &= swap_free(s, swap[si], cam_ref[si] - 0.15, cam_ref[si] + 0.15)
                lid_ok = ~lid_bad[si] & swap_free(s, swap[si], lid_ref[si] - 0.1, lid_ref[si] + 0.2)
                mviews += edges.moving_views(rig, s, traj, cam_ref[si], cam_ok, lid_ok, lo_dr, T_lc)
            log("moving views", len(mviews), sum(len(v[0]) for v in mviews))
            T_s = T_lc
            for rnd in range(int(os.environ.get("REFINE_ROUNDS", "1"))):
                T_lc = edges.refine(rig, mviews, T_lc, log=log)
            if np.linalg.norm(T_lc[:3, 3] - T_lc0[:3, 3]) > 0.2:
                log("moving refine moved too far, keeping the static solution")
                T_lc = T_s
    except Exception as e:  # keep the IMU-based rotation and the prior translation
        log("moving edge alignment failed", repr(e))
    R_li = T_lc[:3, :3] @ R_ci
    T_li = T_li0.copy()
    T_li[:3, :3] = R_li
    log("T_lidar_camera", T_lc)

    # ---- accelerometer bias: at rest the body is level
    up_imu = R_li.T @ up_l
    ab = acc_st - G * up_imu
    log("accel bias", ab)

    # ---- IMU axis swaps, odometry glitches
    for si, s in enumerate(seqs):
        fs, _ = flt.imu_axis_swap(names[si], s, cam_iv[si], R_ci, gb, ab, up_imu)
        faults += fs
    log("faults", flt.best_per_type(faults))

    write_calib(a.output, T_lc, T_li, co_off, co_dr, lo_off, lo_dr, gb, ab)
    write_faults(a.output, faults)


if __name__ == "__main__":
    main()
