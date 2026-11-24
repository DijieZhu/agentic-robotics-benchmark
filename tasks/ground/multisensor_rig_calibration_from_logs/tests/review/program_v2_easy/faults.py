"""Detection of faulty stretches in the logs."""
import numpy as np


def runs(mask):
    """list of (i0, i1) inclusive index runs where mask is True."""
    m = np.concatenate([[False], np.asarray(mask, bool), [False]])
    d = np.diff(m.astype(int))
    st = np.nonzero(d == 1)[0]
    en = np.nonzero(d == -1)[0] - 1
    return list(zip(st, en))


def close_gaps(mask, gap):
    m = np.asarray(mask, bool).copy()
    rs = runs(m)
    for (a0, a1), (b0, b1) in zip(rs[:-1], rs[1:]):
        if b0 - a1 - 1 <= gap:
            m[a1:b0] = True
    return m


def camera_frozen(name, stamps, hashes):
    dup = np.zeros(len(hashes), bool)
    for k in range(1, len(hashes)):
        dup[k] = hashes[k] == hashes[k - 1]
    out = []
    for i0, i1 in runs(dup):
        if i1 - i0 + 1 >= 3:
            out.append({"sequence": name, "sensor": "camera", "type": "camera_frozen",
                        "start_s": float(stamps[i0]), "end_s": float(stamps[i1])})
    return out


def camera_time_jump(name, stamps):
    """returns faults, a boolean mask of frames to skip and corrected stamps."""
    d = np.diff(stamps) - np.median(np.diff(stamps))
    bad = np.zeros(len(stamps), bool)
    out = []
    ev = np.nonzero(np.abs(d) > 0.02)[0]
    used = set()
    for i in ev:
        if i in used:
            continue
        # look for the opposite jump later on
        cand = [j for j in ev if j > i and j not in used and np.sign(d[j]) != np.sign(d[i])
                and 0.25 < stamps[j] - stamps[i] < 10.0
                and abs(abs(d[j]) - abs(d[i])) < max(0.02, 0.3 * abs(d[i]))]
        if cand:
            j = cand[0]
            used.add(i); used.add(j)
            bad[i + 1:j + 1] = True
            out.append({"sequence": name, "sensor": "camera", "type": "camera_time_jump",
                        "start_s": float(stamps[i + 1]), "end_s": float(stamps[j]), "score": 2.0})
    # a single jump that is never undone: shifted until the end of the sequence
    rest = [i for i in ev if i not in used]
    if not out and len(rest) == 1:
        i = rest[0]
        # frames next to the jump are dropped, the rest are corrected by the jump (in the returned fix)
        bad[max(i - 1, 0):i + 3] = True
        fix = stamps.copy()
        fix[i + 1:] -= d[i]
        out.append({"sequence": name, "sensor": "camera", "type": "camera_time_jump",
                    "start_s": float(stamps[i + 1]), "end_s": float(stamps[-1]), "score": 1.0})
        return out, bad, fix
    return out, bad, stamps.copy()


def lidar_degraded(name, rig, seq):
    """scans with many short, dim returns, a collapsed reflectivity or an odd return count."""
    from scipy.ndimage import median_filter
    n = len(seq.scan_stamps)
    f = np.zeros(n)
    F = np.zeros((n, 2))
    for k in range(n):
        r, refl = seq.scan(k)
        v = r > 0
        f[k] = np.mean(v & (r < 3.0) & (refl < 0.08))
        F[k] = [v.mean(), np.median(refl[v]) if v.any() else 0.0]
    med = np.median(f)
    mad = 1.4826 * np.median(np.abs(f - med)) + 1e-4
    thr = max(0.012, med + 10 * mad, 3 * np.percentile(f, 75))
    rm = median_filter(F, size=(min(101, 2 * (n // 2) + 1), 1), mode="nearest")
    rel_refl = F[:, 1] / np.maximum(rm[:, 1], 1e-6)
    dvalid = np.abs(F[:, 0] - rm[:, 0])
    bad = close_gaps((f > thr) | (rel_refl < 0.7) | (dvalid > 0.06), 2)
    out = []
    for i0, i1 in runs(bad):
        if i1 - i0 + 1 >= 5:
            sc = max(np.mean(f[i0:i1 + 1]) / thr, np.mean(0.7 / np.maximum(rel_refl[i0:i1 + 1], 0.05)),
                     np.mean(dvalid[i0:i1 + 1]) / 0.06)
            out.append({"sequence": name, "sensor": "lidar", "type": "lidar_degraded",
                        "start_s": float(seq.scan_stamps[i0]), "end_s": float(seq.scan_stamps[i1]),
                        "score": float(sc)})
    return out, bad, f


def best_per_type(faults):
    """at most one fault of each type per rig: keep the highest-scoring candidate."""
    best = {}
    for f in faults:
        t = f["type"]
        if t not in best or f.get("score", 1.0) > best[t].get("score", 1.0):
            best[t] = f
    out = []
    for f in best.values():
        g = dict(f)
        g.pop("score", None)
        out.append(g)
    return out


SWAPS = [(0, 1), (0, 2), (1, 2)]


def perm(i, j):
    P = np.eye(3)
    P[[i, j]] = P[[j, i]]
    return P


def imu_axis_swap(name, seq, cam_iv, R_ci, gb, ab, up_imu, G=9.80665):
    """cam_iv: (t0, t1, Y) camera rotation vectors over reference-time intervals."""
    t = seq.imu_stamps
    n = len(t)
    flag = np.zeros(n, bool)
    which = np.zeros(n, int) - 1
    # accelerometer: gravity on a wrong axis
    k = np.ones(40) / 40
    alp = np.array([np.convolve(seq.accel[:, i] - ab[i], k, 'same') for i in range(3)]).T
    gexp = G * up_imu
    e0 = np.linalg.norm(alp - gexp, axis=1)
    for si, (i, j) in enumerate(SWAPS):
        ep = np.linalg.norm(alp @ perm(i, j).T - gexp, axis=1)
        m = (e0 > 4.0) & (ep < 2.5)
        flag |= m
        which[m] = si
    # gyro against camera rotations
    t0, t1, Y = cam_iv
    if len(t0):
        from calib import cumint, interval_integrals
        C = cumint(t, seq.gyro - gb)
        X = interval_integrals(t, C, t0, t1)
        r0 = np.linalg.norm(Y - X @ R_ci.T, axis=1)
        noise = 1.4826 * np.median(r0) + 1e-4
        for si, (i, j) in enumerate(SWAPS):
            rp = np.linalg.norm(Y - X @ perm(i, j).T @ R_ci.T, axis=1)
            m = (r0 > 4 * noise) & (rp < 0.5 * r0) & (r0 - rp > 3 * noise)
            for a, b in zip(t0[m], t1[m]):
                sel = (t >= a) & (t <= b)
                flag[sel] = True
                which[sel] = si
    flag = close_gaps(flag, 80)
    out = []
    for i0, i1 in runs(flag):
        if t[i1] - t[i0] >= 0.8:
            out.append({"sequence": name, "sensor": "imu", "type": "imu_axis_swap",
                        "start_s": float(t[i0]), "end_s": float(t[i1]), "score": float(t[i1] - t[i0])})
    return out, flag


def odom_steps(name, seq):
    """wheel-speed glitches: a jump far beyond any physical acceleration, undone by the opposite jump later."""
    t = seq.odom_stamps
    v = seq.odom_v
    dv = np.diff(v)
    mad = 1.4826 * np.median(np.abs(dv - np.median(dv))) + 1e-4
    thr = max(0.15, 10 * mad)
    ev = np.nonzero(np.abs(dv) > thr)[0]
    out = []
    for i in ev:
        for j in ev:
            if j <= i or np.sign(dv[j]) == np.sign(dv[i]):
                continue
            dur = t[j] - t[i + 1]
            if not (0.3 < dur < 8.0):
                continue
            a, b = abs(dv[i]), abs(dv[j])
            if min(a, b) / max(a, b) < 0.25:
                continue
            out.append({"sequence": name, "sensor": "wheel_odom", "type": "odom_glitch",
                        "start_s": float(t[i + 1]), "end_s": float(t[j]),
                        "score": 100.0 + float(min(a, b) / thr)})
            break
    return out


def odom_glitch(name, seq, t_mid_ref, v_lidar, valid, thr=0.25):
    """compare the wheel speed with the LiDAR's forward speed."""
    t = seq.odom_stamps
    tm = t_mid_ref[valid]
    vl = np.interp(t, tm, v_lidar[valid])
    inside = (t > tm[0]) & (t < tm[-1])
    # no interpolation across gaps in the LiDAR speeds
    j = np.clip(np.searchsorted(tm, t), 1, len(tm) - 1)
    near = np.minimum(np.abs(t - tm[j - 1]), np.abs(tm[j] - t)) < 0.15
    d = np.abs(seq.odom_v - vl)
    flag = inside & near & (d > thr)
    flag = close_gaps(flag, 10)
    out = []
    for i0, i1 in runs(flag):
        if t[i1] - t[i0] >= 0.7:
            out.append({"sequence": name, "sensor": "wheel_odom", "type": "odom_glitch",
                        "start_s": float(t[i0]), "end_s": float(t[i1]),
                        "score": float(np.mean(d[i0:i1 + 1]) * (t[i1] - t[i0]))})
    return out, flag
