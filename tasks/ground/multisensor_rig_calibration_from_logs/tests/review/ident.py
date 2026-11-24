#!/usr/bin/env python3
"""Identifiability of every scored parameter on a rig's GIVEN sequences: a privileged, truth-initialised calibration.

    <gen venv python> -I -B ident.py --root <rigs root> --set practice --index 0 --name rig_a --hall .. --props .. --husky .. --cache ..
        --tests <task tests dir> --out <json> [--no-camera]

Privileged = it may use the truth to associate data and to start, never as the answer: each estimate below is the
minimiser of a residual made of the LOGGED observations, started at the truth.
  A. LiDAR-IMU rotation, gyro bias, LiDAR clock offset per sequence (then offset + drift by a line through the
     sequences): relative rotations of consecutive sweeps (point-to-plane ICP on sweeps motion-corrected with the true
     motion, started at the true relative pose) against the integrated gyro.
  B. Accelerometer bias: the rests' mean specific force with the estimated rotation (flat floor). IMU position: least
     squares on the accelerometer's samples while moving, the LiDAR's own acceleration taken from the truth (so this
     one says whether the drives excite the lever arm, not how well LiDAR odometry would give it).
  C. Camera-LiDAR extrinsic: maximum of the edge-alignment score (the reference calibrator's: LiDAR depth and
     reflectivity edges against the image's gradient) over the six parameters, started at the truth, on frames at rest
     and in motion. Camera clock offset per sequence: the score's maximum over a time shift with the true extrinsic;
     then offset + drift by a line.
Faulty stretches are left out by the true fault list. The result: where each scored parameter settles, and the
task's metrics of that calibration on the held-out sequence."""
import argparse, csv, json, math, os, sys, time
import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gen"))
import rigsim, rigsite
ap = argparse.ArgumentParser()
for k in ("root", "set", "index", "name", "hall", "props", "husky", "people", "cache", "tests", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--no-camera", action="store_true"); ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--odom", type=int, default=1); ap.add_argument("--knock", type=int, default=0)
a = ap.parse_args()
sys.path.insert(0, a.tests); import rigmetrics as RM
site = rigsite.Site(a.hall, a.props, a.cache)
rigsite.PEOPLE_DIR = os.path.abspath(a.people)
rig = rigsim.make_rig(rigsim.rig_seed(a.set, a.index), site, has_odom=bool(a.odom), knock=bool(a.knock))
rd, td = os.path.join(a.root, a.set, a.name), os.path.join(a.root, a.set + "_truth", a.name)
faults = json.load(open(os.path.join(td, "faults.json")))["faults"]
DIRS = rigsim.lidar_ray_dirs().reshape(rigsim.LIDAR_RINGS, rigsim.LIDAR_COLS, 3)
CDT = rigsim.column_dt(rig)
T_bl = rig.T_body_lidar; T_li = rig.T_lidar_imu; R_li = T_li[:3, :3]
t0 = time.time()


def col(path, k):
    with open(path) as fh:
        return np.array([float(r[k]) for r in csv.DictReader(fh)])


def bad(seq, sensor, s0, s1):
    return any(f["sequence"] == seq and f["sensor"] == sensor and not (s1 < f["start_s"] - 0.3 or s0 > f["end_s"] + 0.3) for f in faults)


def lidar_pose(traj, tau):
    R, p = rigsim.sensor_pose_w(rig, traj, np.atleast_1d(tau), T_bl)
    return R, p


def deskewed(traj, path, tau0):
    """the sweep's points in the LiDAR frame at column 0's time (true motion), with ring, column"""
    z = np.load(path); r = z["range_mm"].astype(float) / 1000.0
    m = r > 0
    ring, cc = np.nonzero(m)
    p = DIRS[ring, cc] * r[ring, cc][:, None]
    Rj, pj = lidar_pose(traj, tau0 + CDT[cc]); R0, p0 = lidar_pose(traj, tau0)
    pw = np.einsum("nij,nj->ni", Rj, p) + pj
    return (pw - p0[0]) @ R0[0], r, z["reflectivity"].astype(float) / 255.0, ring, cc


def voxel(p, s):
    k = np.floor(p / s).astype(np.int64); _, i = np.unique(k, axis=0, return_index=True); return p[np.sort(i)]


def normals(p, tree, k=10):
    _, nn = tree.query(p, k=k); nb = p[nn]; mu = nb.mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", nb - mu, nb - mu) / k
    w, v = np.linalg.eigh(C); return v[:, :, 0]


def icp(src, dst, tree, dn, T, iters=12):
    for it in range(iters):
        ps = src @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(ps, k=1, distance_upper_bound=0.3)
        m = np.isfinite(d)
        if m.sum() < 100:
            return None
        ps, q, n = ps[m], dst[j[m]], dn[j[m]]
        r = np.sum((ps - q) * n, axis=1); J = np.concatenate([np.cross(ps, n), n], axis=1)
        w = np.minimum(1.0, 0.03 / np.maximum(np.abs(r), 1e-12))
        dx = -np.linalg.solve(J.T @ (J * w[:, None]) + 1e-9 * np.eye(6), J.T @ (w * r))
        dT = np.eye(4); dT[:3, :3] = rigsim.exp_so3(dx[:3]); dT[:3, 3] = dx[3:]; T = dT @ T
        if np.linalg.norm(dx) < 1e-7:
            break
    return T


# ------------------------------------------------------------------ A: sweeps against the gyro
# A 16-beam sweep registers poorly to the next one; each sweep is registered to the site's own surface instead (points
# sampled on the triangles the LiDAR was cast on: privileged association, as a map built from the logs would give),
# started at its true pose. The relative rotations of consecutive sweeps then go against the gyro.
site.build(rig.props)
_tr = site.caster.tris.astype(np.float64); _n = site.caster.n
_ar = 0.5 * np.linalg.norm(np.cross(_tr[:, 1] - _tr[:, 0], _tr[:, 2] - _tr[:, 0]), axis=1)
_rs = np.random.default_rng(0)
_cnt = _rs.poisson(_ar * 150.0)
_idx = np.repeat(np.arange(len(_tr)), _cnt)
_u = _rs.uniform(size=(len(_idx), 2)); _fl = _u.sum(1) > 1; _u[_fl] = 1 - _u[_fl]
MAP_P = _tr[_idx, 0] + _u[:, :1] * (_tr[_idx, 1] - _tr[_idx, 0]) + _u[:, 1:] * (_tr[_idx, 2] - _tr[_idx, 0]); MAP_N = _n[_idx]
MAP_T = cKDTree(MAP_P)
print("[ident] map: %d points on %d triangles, %.0f s" % (len(MAP_P), len(_tr), time.time() - t0))
SEQ = []
for k in rig.given:
    name = "seq_%02d" % (k - 1); sd = os.path.join(rd, name); traj = rig.trajs[k]; T0 = rig.seq_starts[k]
    lst = col(os.path.join(sd, "lidar", "timestamps.csv"), "stamp_s")
    tau_true = rigsim.sensor_to_ref(lst, rig.lidar_offset, rig.lidar_drift) - T0          # privileged: for association and deskew only
    imu = np.loadtxt(os.path.join(sd, "imu.csv"), delimiter=",", skiprows=1)
    rel = []
    prev = None
    for i in range(0, lst.size, a.stride):
        p, _, _, _, _ = deskewed(traj, os.path.join(sd, "lidar", "scans", "%06d.npz" % i), tau_true[i])
        p = voxel(p[(np.linalg.norm(p, axis=1) > 1.0) & (np.linalg.norm(p, axis=1) < 40.0)], 0.1)
        Ra, pa = lidar_pose(traj, tau_true[i])
        Tt = np.eye(4); Tt[:3, :3] = Ra[0]; Tt[:3, 3] = pa[0]                               # the sweep's true pose in the world
        T = icp(p, MAP_P, MAP_T, MAP_N, Tt.copy())
        ok = T is not None and not bad(name, "lidar", lst[i] - 0.05, lst[i] + 0.15)
        if ok and prev is not None and not bad(name, "imu", T0 + tau_true[prev[0]], T0 + tau_true[i]):
            rel.append((lst[prev[0]], lst[i], rigsim.log_so3(prev[1][:3, :3].T @ T[:3, :3]), rigsim.log_so3(prev[2][:3, :3].T @ Tt[:3, :3])))
        prev = (i, T, Tt) if ok else None
    SEQ.append({"name": name, "T0": T0, "imu": imu, "rel": rel, "lst": lst, "c": k})
    e = np.array([np.linalg.norm(r[2] - r[3]) for r in rel])
    print("[ident] %s: %d sweep pairs, registration rotation error median %.4f deg, %.0f s" % (name, len(rel), math.degrees(np.median(e)), time.time() - t0))


def gyro_rot(imu, ta, tb, bias):
    """rotation vector of the integrated gyro between reference times ta and tb (IMU frame, b in a)"""
    t = imu[:, 0]; i0 = np.searchsorted(t, ta); i1 = np.searchsorted(t, tb)
    ts = np.concatenate([[ta], t[i0:i1], [tb]])
    g = np.stack([np.interp(ts, t, imu[:, 1 + j]) for j in range(3)], 1) - bias
    R = np.eye(3)
    for j in range(len(ts) - 1):
        R = R @ rigsim.exp_so3(0.5 * (g[j] + g[j + 1]) * (ts[j + 1] - ts[j]))
    return R


def resid_A(x):
    R = rigsim.exp_so3(x[:3]) @ R_li; b = rig.gyro_bias_scored + x[3:6]
    out = []
    for k, S in enumerate(SEQ):
        o = rig.lidar_offset + rig.lidar_drift * (S["T0"] + 13.0) + x[6 + k]              # the clock's reading error at this sequence
        for sa, sb, rv, _ in S["rel"]:
            Rg = gyro_rot(S["imu"], sa - o, sb - o, b)
            out.append(rigsim.log_so3((R @ Rg @ R.T).T @ rigsim.exp_so3(rv)) / 0.0005)
    return np.concatenate(out)


sol = least_squares(resid_A, np.zeros(6 + len(SEQ)), x_scale=[1e-3] * 3 + [1e-4] * 3 + [1e-3] * len(SEQ), loss="huber", f_scale=3.0, max_nfev=40)
xA = sol.x
R_li_hat = rigsim.exp_so3(xA[:3]) @ R_li; bg_hat = rig.gyro_bias_scored + xA[3:6]
tm = np.array([S["T0"] + 13.0 for S in SEQ]); eo = xA[6:]                                  # clock error at each sequence's middle
cA = np.polyfit(tm, eo, 1) if len(SEQ) > 1 else np.array([0.0, eo[0]])
th = rig.seq_starts[rig.held["B"]] + 13.0
RES = {"A": {"imu_rot_deg": math.degrees(np.linalg.norm(xA[:3])), "gyro_bias_err_rps": float(np.linalg.norm(xA[3:6])),
             "lidar_clock_err_per_sequence_ms": (eo * 1e3).round(3).tolist(), "lidar_drift_err_ppm": float(cA[0] * 1e6),
             "lidar_clock_err_at_heldout_ms": float(np.polyval(cA, th) * 1e3), "sweep_pairs": sum(len(S["rel"]) for S in SEQ)}}
print("[ident] A:", json.dumps(RES["A"]), "%.0f s" % (time.time() - t0))
lid_o = rig.lidar_offset + cA[1] - cA[0] * 0.0 * 0; lid_d = rig.lidar_drift + cA[0]
lid_o = rig.lidar_offset + (cA[1])                                                         # e(t) = c0 t + c1 -> offset + c1, drift + c0

# ------------------------------------------------------------------ B: accelerometer
acc_rest = []
for k, S in enumerate(SEQ):
    for ra, rb in rig.rests[S["c"]]:
        m = (S["imu"][:, 0] > S["T0"] + ra + 0.6) & (S["imu"][:, 0] < S["T0"] + rb - 0.2)
        if m.sum() > 50 and not bad(S["name"], "imu", S["T0"] + ra, S["T0"] + rb):
            acc_rest.append(S["imu"][m, 4:7])
acc_rest = np.concatenate(acc_rest)
R_bi_hat = T_bl[:3, :3] @ R_li_hat
ba_hat = acc_rest.mean(0) - R_bi_hat.T @ np.array([0.0, 0.0, rigsim.G])
# lever arm: accel_imu - b - R_il (a_L - g)_L = R_il (alpha x t + w x (w x t)), linear in t (LiDAR frame); true LiDAR acceleration
Arows, yrows = [], []
for k, S in enumerate(SEQ):
    traj = rig.trajs[S["c"]]; imu = S["imu"]; tau = imu[:, 0] - S["T0"]
    mv = np.abs(np.interp(tau, traj.tau, traj.w)) > 0.25
    for f in faults:
        if f["sequence"] == S["name"] and f["sensor"] == "imu":
            mv &= ~((imu[:, 0] > f["start_s"] - 0.3) & (imu[:, 0] < f["end_s"] + 0.3))
    idx = np.nonzero(mv)[0][::2]
    _, _, aL = traj.point_kinematics(T_bl[:3, 3], tau[idx]); Rb, _ = traj.interp(tau[idx]); Rwl = Rb @ T_bl[:3, :3]
    fL = np.einsum("nji,nj->ni", Rwl, aL - rigsim.GRAVITY_W)                              # specific force at the LiDAR origin, LiDAR frame
    g = (imu[:, 1:4] - bg_hat) @ R_li_hat.T                                               # angular rate in the LiDAR frame
    al = np.gradient(g, 0.005, axis=0)
    # smooth the angular acceleration (the gyro's noise differentiated)
    ker = np.hanning(21); ker /= ker.sum(); al = np.stack([np.convolve(al[:, j], ker, mode="same") for j in range(3)], 1)
    fi = (imu[idx, 4:7] - ba_hat) @ R_li_hat.T
    for n_, i in enumerate(idx):
        Arows.append(RM._hat(al[i]) + RM._hat(g[i]) @ RM._hat(g[i])); yrows.append(fi[n_] - fL[n_])
A_ = np.concatenate(Arows); y_ = np.concatenate(yrows)
t_li_hat, *_ = np.linalg.lstsq(A_, y_, rcond=None)
RES["B"] = {"accel_bias_err_mps2": float(np.linalg.norm(ba_hat - rig.accel_bias_scored)), "imu_position_err_m": float(np.linalg.norm(t_li_hat - T_li[:3, 3])),
            "rest_samples": int(len(acc_rest)), "moving_samples": int(len(y_) // 3)}
print("[ident] B:", json.dumps(RES["B"]), "%.0f s" % (time.time() - t0))
T_li_hat = np.eye(4); T_li_hat[:3, :3] = R_li_hat; T_li_hat[:3, 3] = t_li_hat

# ------------------------------------------------------------------ C: camera (mount, intrinsics, distortion, readout, clock)
T_lc_hat = rig.T_lidar_camera.copy(); T_lcb_hat = rig.T_lidar_camera_before.copy(); cam_o, cam_d = rig.cam_offset, rig.cam_drift
intr_hat = [rig.K[0, 0], rig.K[1, 1], rig.K[0, 2], rig.K[1, 2], rig.dist.copy(), rig.readout]
if not a.no_camera:
    import cv2
    W, H = rigsim.IMG_W, rigsim.IMG_H
    def edges(path):
        """LiDAR edge points of a sweep: depth steps (the near side) and reflectivity steps along each ring; column kept"""
        z = np.load(path); r = z["range_mm"].astype(float) / 1000.0; refl = z["reflectivity"].astype(float) / 255.0
        rn = np.roll(r, -1, axis=1); fn = np.roll(refl, -1, axis=1)
        both = (r > 0.8) & (rn > 0.8) & (r < 30) & (rn < 30)
        same = both & (np.abs(r - rn) < 0.05 + 0.03 * np.minimum(r, rn)); dstep = both & ~same
        wr = np.where(same & (np.abs(refl - fn) > 0.08), np.abs(refl - fn), 0.0)
        wd = np.where(dstep, np.clip(np.abs(r - rn) / 0.5, 0, 1) * 0.5, 0.0)
        out = []
        for wgt, rng_ in ((wr, 0.5 * (r + rn)), (wd, np.minimum(r, rn))):
            ring, cc = np.nonzero(wgt > 0)
            mid = DIRS[ring, cc] + DIRS[ring, (cc + 1) % rigsim.LIDAR_COLS]; mid /= np.linalg.norm(mid, axis=1, keepdims=True)
            out.append((mid * rng_[ring, cc][:, None], wgt[ring, cc], cc))
        return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out]), np.concatenate([o[2] for o in out])
    def edge_img(path):
        g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY).astype(np.float64) / 255.0       # the frame as logged (distorted)
        gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3); e = np.sqrt(gx * gx + gy * gy)
        e = cv2.GaussianBlur(e / max(np.percentile(e, 99), 1e-6), (0, 0), 1.5); return e / max(e.max(), 1e-9)
    def bil(img, u, v):
        u0 = np.floor(u).astype(int); v0 = np.floor(v).astype(int); x = u - u0; y = v - v0
        return (1 - x) * (1 - y) * img[v0, u0] + x * (1 - y) * img[v0, u0 + 1] + (1 - x) * y * img[v0 + 1, u0] + x * y * img[v0 + 1, u0 + 1]
    FR = []
    for S in SEQ:
        k = S["c"]; sd = os.path.join(rd, S["name"]); traj = rig.trajs[k]; T0 = S["T0"]
        cst = col(os.path.join(sd, "camera", "timestamps.csv"), "stamp_s")
        tl = rigsim.sensor_to_ref(S["lst"], rig.lidar_offset, rig.lidar_drift) - T0
        for ci in range(8, cst.size - 8, 2):
            if bad(S["name"], "camera", cst[ci] - 0.2, cst[ci] + 0.2):
                continue
            tc = rigsim.sensor_to_ref(cst[ci], rig.cam_offset, rig.cam_drift) - T0
            if rig.knock is not None and k == rig.knock["c"] and abs(tc - rig.knock["tau"]) < 0.15:
                continue
            sp = abs(float(np.interp(tc, traj.tau, traj.v))) + abs(float(np.interp(tc, traj.tau, traj.w)))
            if ci % 6 and sp <= 0.2:
                continue                                                                   # at rest: every sixth frame is enough
            li = int(np.argmin(np.abs(tl + 0.05 - tc)))
            if bad(S["name"], "lidar", S["lst"][li] - 0.2, S["lst"][li] + 0.2):
                continue
            pts, w, cc = edges(os.path.join(sd, "lidar", "scans", "%06d.npz" % li))
            Rj, pj = lidar_pose(traj, tl[li] + CDT[cc]); pw = np.einsum("nij,nj->ni", Rj, pts) + pj      # true motion: each edge point in the world
            after = rig.knock is None or k > rig.knock["c"] or (k == rig.knock["c"] and tc >= rig.knock["tau"])
            FR.append({"k": k, "tc": tc, "pw": pw, "w": w, "img": edge_img(os.path.join(sd, "camera", "frames", "%06d.jpg" % ci)), "traj": traj, "moving": sp > 0.2, "after": after})
    def score(frames, T_lc, intr, dts=0.0):
        fx, fy, cx, cy, dist, ro = intr
        tot, cnt = 0.0, 0
        for fr in frames:
            v = np.full(len(fr["pw"]), (H - 1) / 2.0)
            for _ in range(2):                                                             # the row a point falls in sets its time
                tau = fr["tc"] + dts + (v - (H - 1) / 2.0) / H * ro / (1.0 + rig.cam_drift)
                Rl, pl = lidar_pose(fr["traj"], tau)
                pc = np.einsum("nji,nj->ni", Rl, fr["pw"] - pl); pc = (pc - T_lc[:3, 3]) @ T_lc[:3, :3]
                z = pc[:, 2]; ok = z > 0.5; zs = np.where(ok, z, 1.0)
                x = np.clip(pc[:, 0] / zs, -2, 2); y = np.clip(pc[:, 1] / zs, -2, 2)
                xd, yd = RM._distort(x, y, dist)
                u = fx * xd + cx; v = np.clip(fy * yd + cy, -H, 2 * H)
            ok &= (u > 3) & (u < W - 4) & (v > 3) & (v < H - 4) & (np.abs(x) < 0.8) & (np.abs(y) < 0.55)
            if ok.sum() < 50:
                continue
            tot += float(bil(fr["img"], u[ok], v[ok]) @ fr["w"][ok]) / float(fr["w"][ok].sum()); cnt += 1
        return tot / max(cnt, 1)
    TRUE_INTR = [rig.K[0, 0], rig.K[1, 1], rig.K[0, 2], rig.K[1, 2], rig.dist.copy(), rig.readout]
    def pert(T0_, x):
        T = T0_.copy(); T[:3, :3] = rigsim.exp_so3(x[:3]) @ T[:3, :3]; T[:3, 3] = T[:3, 3] + x[3:6]; return T
    def intr_of(x):
        d = rig.dist.copy(); d[0] += x[3]; d[1] += x[4]
        return [rig.K[0, 0] + x[0], rig.K[1, 1] + x[0], rig.K[0, 2] + x[1], rig.K[1, 2] + x[2], d, rig.readout + x[5]]
    FA = [f for f in FR if f["after"]][::2]; FB = [f for f in FR if not f["after"]][::2]
    st = np.array([0.002] * 3 + [0.01] * 3 + [4.0, 3.0, 3.0, 0.01, 0.01, 0.003])
    res = minimize(lambda z: -score(FA, pert(rig.T_lidar_camera, (z * st)[:6]), intr_of((z * st)[6:])), np.zeros(12), method="Powell", options={"xtol": 2e-2, "ftol": 1e-6, "maxfev": 1500})
    xC = res.x * st; T_lc_hat = pert(rig.T_lidar_camera, xC[:6]); intr_hat = intr_of(xC[6:])
    RES["C"] = {"camera_rot_err_deg": math.degrees(np.linalg.norm(xC[:3])), "camera_trans_err_m": float(np.linalg.norm(xC[3:6])),
                "focal_err_px": float(xC[6]), "pp_err_px": [float(xC[7]), float(xC[8])], "k1_err": float(xC[9]), "k2_err": float(xC[10]), "readout_err_ms": float(xC[11] * 1e3),
                "score_at_truth": score(FA, rig.T_lidar_camera, TRUE_INTR), "score_at_optimum": score(FA, T_lc_hat, intr_hat),
                "score_at_cad_and_nominal": score(FA, rig.T_lidar_camera_prior, [rigsim.F_NOMINAL, rigsim.F_NOMINAL, W / 2.0, H / 2.0, np.zeros(5), 0.02]),
                "frames_after": len(FA), "frames_before": len(FB), "evals": int(res.nfev)}
    T_lcb_hat = T_lc_hat.copy()
    if rig.knock is not None:
        st6 = st[:6]
        rb = minimize(lambda z: -score(FB, pert(rig.T_lidar_camera_before, z * st6), intr_hat), np.zeros(6), method="Powell", options={"xtol": 2e-2, "ftol": 1e-6, "maxfev": 600})
        T_lcb_hat = pert(rig.T_lidar_camera_before, rb.x * st6)
        RES["C"]["before_rot_err_deg"] = math.degrees(np.linalg.norm((rb.x * st6)[:3])); RES["C"]["before_trans_err_m"] = float(np.linalg.norm((rb.x * st6)[3:]))
        RES["C"]["knock_score_before_frames_with_after_mount"] = score(FB, rig.T_lidar_camera, TRUE_INTR); RES["C"]["knock_score_before_frames_with_before_mount"] = score(FB, rig.T_lidar_camera_before, TRUE_INTR)
    # camera clock per sequence: the score over a time shift, true mount and intrinsics, moving frames
    eo = []
    for S in SEQ:
        fr = [f for f in FR if f["k"] == S["c"] and f["moving"]]
        fa = [f for f in fr if f["after"]]; fb = [f for f in fr if not f["after"]]
        ds = np.arange(-0.012, 0.0121, 0.001)
        sc = np.array([(score(fa, rig.T_lidar_camera, TRUE_INTR, d) * len(fa) + score(fb, rig.T_lidar_camera_before, TRUE_INTR, d) * len(fb)) / max(len(fr), 1) for d in ds])
        j = int(np.argmax(sc)); d = ds[j]
        if 0 < j < len(ds) - 1:
            den = sc[j - 1] - 2 * sc[j] + sc[j + 1]; d = ds[j] + (0.0005 * (sc[j - 1] - sc[j + 1]) / den if den < 0 else 0.0)
        eo.append(-d)
    eo = np.array(eo); cC = np.polyfit(tm, eo, 1) if len(SEQ) > 1 else np.array([0.0, eo[0]])
    cam_o = rig.cam_offset + cC[1]; cam_d = rig.cam_drift + cC[0]
    RES["C"]["camera_clock_err_per_sequence_ms"] = (eo * 1e3).round(3).tolist(); RES["C"]["camera_drift_err_ppm"] = float(cC[0] * 1e6)
    print("[ident] C:", json.dumps(RES["C"]), "%.0f s" % (time.time() - t0))

# ------------------------------------------------------------------ the task's metrics of the settled calibration
truth, ho = RM.load_truth_files(rd, td)
cal = {"T_lidar_camera": T_lc_hat, "T_lidar_camera_before": T_lcb_hat if rig.knock is not None else None, "T_lidar_imu": T_li_hat,
       "cam_fx": intr_hat[0], "cam_fy": intr_hat[1], "cam_cx": intr_hat[2], "cam_cy": intr_hat[3], "cam_dist": intr_hat[4], "cam_readout_s": intr_hat[5],
       "camera_offset": cam_o, "camera_drift": cam_d,
       "lidar_offset": rig.lidar_offset + cA[1], "lidar_drift": rig.lidar_drift + cA[0], "gyro_bias": bg_hat, "accel_bias": ba_hat}
RES["metrics_settled"] = RM.calibration_metrics(truth, ho, cal)
calT = dict(cal, T_lidar_camera=rig.T_lidar_camera, T_lidar_camera_before=rig.T_lidar_camera_before if rig.knock is not None else None, T_lidar_imu=T_li,
            cam_fx=rig.K[0, 0], cam_fy=rig.K[1, 1], cam_cx=rig.K[0, 2], cam_cy=rig.K[1, 2], cam_dist=rig.dist, cam_readout_s=rig.readout,
            camera_offset=rig.cam_offset, camera_drift=rig.cam_drift, lidar_offset=rig.lidar_offset,
            lidar_drift=rig.lidar_drift, gyro_bias=rig.gyro_bias_scored, accel_bias=rig.accel_bias_scored)
RES["metrics_truth"] = RM.calibration_metrics(truth, ho, calT)
print("[ident] settled:", {k: float("%.4g" % v) for k, v in RES["metrics_settled"].items()})
print("[ident] truth:  ", {k: float("%.4g" % v) for k, v in RES["metrics_truth"].items()})
json.dump(RES, open(a.out, "w"), indent=1)
