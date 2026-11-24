"""Camera-LiDAR extrinsic from depth edges of static sweeps against image edges."""
import numpy as np
import cv2
from scipy.optimize import minimize

import lo


def lidar_edge_points(rig, r, with_cols=False):
    """3D points on foreground depth edges (half a column / ring towards the background)."""
    R, C = r.shape
    el = np.radians(np.array(rig.manifest["lidar"]["elevation_deg"], float))
    out = []
    cols = []
    # along columns (within a ring)
    for sgn in (1, -1):
        rn = np.roll(r, -sgn, axis=1)      # neighbour on the background side
        rp = np.roll(r, sgn, axis=1)       # neighbour on the other side
        thr = np.maximum(0.3, 0.15 * r)
        m = (r > 0.8) & (r < 30) & ((rn - r > thr) | (rn == 0)) & (rp > 0) & (np.abs(rp - r) < 0.05 * r + 0.05)
        ii, jj = np.nonzero(m)
        az = 2 * np.pi * (jj + 0.5 * sgn) / C
        e = el[ii]
        rr = r[ii, jj]
        out.append(np.stack([rr * np.cos(e) * np.cos(az), rr * np.cos(e) * np.sin(az), rr * np.sin(e)], 1))
        cols.append(jj)
    # along rings (top / bottom edges)
    for sgn in (1, -1):
        m = np.zeros_like(r, bool)
        if sgn == 1:
            rn = np.zeros_like(r); rn[:-1] = r[1:]
            rp = np.zeros_like(r); rp[1:] = r[:-1]
        else:
            rn = np.zeros_like(r); rn[1:] = r[:-1]
            rp = np.zeros_like(r); rp[:-1] = r[1:]
        thr = np.maximum(0.3, 0.15 * r)
        m = (r > 0.8) & (r < 30) & (rn - r > thr) & (rn > 0) & (rp > 0) & (np.abs(rp - r) < 0.05 * r + 0.05)
        ii, jj = np.nonzero(m)
        ii2 = np.clip(ii + sgn, 0, R - 1)
        e = 0.5 * (el[ii] + el[ii2])
        az = 2 * np.pi * jj / C
        rr = r[ii, jj]
        out.append(np.stack([rr * np.cos(e) * np.cos(az), rr * np.cos(e) * np.sin(az), rr * np.sin(e)], 1))
        cols.append(jj)
    if with_cols:
        return np.concatenate(out), np.concatenate(cols)
    return np.concatenate(out)


def image_dt(rig, img_gray):
    und = cv2.undistort(img_gray, rig.K, rig.dist)
    blur = cv2.GaussianBlur(und, (3, 3), 0)
    ed = cv2.Canny(blur, 40, 100)
    # mask out the border introduced by undistortion
    valid = cv2.undistort(np.full_like(img_gray, 255), rig.K, rig.dist) > 128
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    dt = cv2.distanceTransform((ed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    return dt.astype(np.float32), valid


class EdgeProblem(object):
    def __init__(self, rig, views):
        """views: list of (edge points in LiDAR frame (N,3), dt image, valid mask)."""
        self.K = rig.K
        self.views = views
        self.cap = 20.0

    def cost(self, x, T0):
        R = lo.expSO3(x[:3]) @ T0[:3, :3]
        t = T0[:3, 3] + x[3:]
        tot = 0.0
        cnt = 0
        H, W = self.views[0][1].shape
        for P, dt, valid in self.views:
            q = (P - t) @ R
            z = np.maximum(q[:, 2], 1e-3)
            u = self.K[0, 0] * q[:, 0] / z + self.K[0, 2]
            v = self.K[1, 1] * q[:, 1] / z + self.K[1, 2]
            ui = np.round(u).astype(int)
            vi = np.round(v).astype(int)
            inside = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (q[:, 2] > 0.5)
            ui, vi = ui[inside], vi[inside]
            ok = valid[vi, ui]
            d = np.minimum(dt[vi[ok], ui[ok]], self.cap)
            tot += d.sum() + self.cap * (len(P) - ok.sum())
            cnt += len(P)
        pen = 200.0 * (max(0.0, np.linalg.norm(x[3:]) - 0.12) + max(0.0, np.linalg.norm(x[:3]) - np.radians(14)))
        return tot / max(cnt, 1) + pen


def preselect(rig, P, T0, margin=80):
    R, t = T0[:3, :3], T0[:3, 3]
    q = (P - t) @ R
    z = q[:, 2]
    m = z > 1.0
    zs = np.where(m, z, 1)
    u = rig.K[0, 0] * q[:, 0] / zs + rig.K[0, 2]
    v = rig.K[1, 1] * q[:, 1] / zs + rig.K[1, 2]
    W, H = rig.manifest["camera"]["width"], rig.manifest["camera"]["height"]
    # keep points that are inside for the prior
    m2 = m & (u > 20) & (u < W - 20) & (v > 20) & (v < H - 20)
    return P[m2]


def solve(rig, views, T0, log=print, axis=None, grid_deg=10.0, fix_t=False):
    """6-DoF edge alignment from T0; with axis given, only the rotation about that axis (LiDAR frame) is free."""
    prob = EdgeProblem(rig, views)

    def full(y):
        if axis is None:
            return y
        if fix_t:
            return np.concatenate([axis * y[0], np.zeros(3)])
        return np.concatenate([axis * y[0], y[1:]])
    # coarse rotation grid with a wide cap
    prob.cap = 40.0
    best = (np.inf, None)
    g = np.radians(np.arange(-grid_deg, grid_deg + 0.1, 2.5))
    if axis is None:
        cands = [np.array([a, b, c, 0, 0, 0]) for a in g for b in g for c in g]
    else:
        cands = [np.array([a] if fix_t else [a, 0, 0, 0]) for a in np.radians(np.arange(-grid_deg, grid_deg + 0.1, 0.5))]
    for x in cands:
        f = prob.cost(full(x), T0)
        if f < best[0]:
            best = (f, x)
    x = best[1]
    log("edge grid", x, best[0])
    nd = len(x)
    for cap, scale in ((25.0, 0.03), (12.0, 0.01), (6.0, 0.004), (4.0, 0.002), (3.0, 0.001)):
        prob.cap = cap
        S = np.full(nd, scale)
        if nd > 1:
            S[-3:] *= 2
        fun = lambda y: prob.cost(full(x + y * S), T0)
        res = minimize(fun, np.zeros(nd), method="Powell", options={"xtol": 1e-2, "ftol": 1e-7, "maxfev": 4000})
        x = x + res.x * S
        log("edge cap", cap, x, res.fun)
    xf = full(x)
    T = np.eye(4)
    T[:3, :3] = lo.expSO3(xf[:3]) @ T0[:3, :3]
    T[:3, 3] = T0[:3, 3] + xf[3:]
    return T


def build_views(rig, seqs, stat, cam_ref, lid_ref, T0, log=print, cam_bad=None, lid_bad=None):
    """one view per static stretch: edge points of a sweep and the DT of a frame taken then."""
    views = []
    for si, (s, m, tc, tl) in enumerate(zip(seqs, stat, cam_ref, lid_ref)):
        mc = np.interp(tc, s.imu_stamps, m.astype(float)) > 0.99
        ml = np.interp(tl, s.imu_stamps, m.astype(float)) > 0.99
        if cam_bad is not None:
            mc &= ~cam_bad[si]
        if lid_bad is not None:
            ml &= ~lid_bad[si]
        idx = np.nonzero(mc)[0]
        if len(idx) == 0:
            continue
        segs = np.split(idx, np.nonzero(np.diff(idx) > 1)[0] + 1)
        for sg in segs:
            if len(sg) < 3:
                continue
            k = sg[len(sg) // 2]
            li = int(np.argmin(np.abs(tl - tc[k]) + 1e3 * (~ml)))
            if abs(tl[li] - tc[k]) > 0.5:
                continue
            r, _ = s.scan(li)
            P = lidar_edge_points(rig, r)
            P = preselect(rig, P, T0)
            img = cv2.imread(s.path + "/camera/frames/" + s.cam_files[k], cv2.IMREAD_GRAYSCALE)
            dt, valid = image_dt(rig, img)
            views.append((P, dt, valid))
    log("edge views", len(views), [len(v[0]) for v in views])
    return views


class Traj(object):
    """LiDAR poses of one sequence from chained scan-to-scan motion, rotation within a sweep from the gyro."""
    def __init__(self, taus, rel, t_imu, C_l):
        n = len(taus)
        self.taus = taus
        Rs = np.zeros((n, 3, 3))
        ps = np.zeros((n, 3))
        T = np.eye(4)
        last = np.eye(4)
        Rs[0] = np.eye(3)
        self.ok = np.zeros(n, bool)
        self.ok[0] = True
        for k in range(n - 1):
            if k in rel:
                last = rel[k]
                self.ok[k + 1] = self.ok[k]
            else:
                self.ok[k + 1] = False
            T = T @ last
            Rs[k + 1] = T[:3, :3]
            ps[k + 1] = T[:3, 3]
        self.Rs, self.ps = Rs, ps
        self.t_imu, self.C = t_imu, C_l

    def pose(self, tau):
        tau = np.atleast_1d(tau)
        k = np.clip(np.searchsorted(self.taus, tau) - 1, 0, len(self.taus) - 2)
        f = (tau - self.taus[k]) / (self.taus[k + 1] - self.taus[k])
        c1 = np.array([np.interp(tau, self.t_imu, self.C[:, i]) for i in range(3)]).T
        c0 = np.array([np.interp(self.taus[k], self.t_imu, self.C[:, i]) for i in range(3)]).T
        R = np.einsum('nij,njk->nik', self.Rs[k], lo.exp_batch(c1 - c0))
        p = self.ps[k] + f[:, None] * (self.ps[k + 1] - self.ps[k])
        return R, p, k


def moving_views(rig, seq, traj, cam_tau, cam_ok, lid_ok, lid_dr, T0, step=2):
    views = []
    sweep = rig.manifest["lidar"]["sweep_s"] / (1 + lid_dr)
    ncol = rig.manifest["lidar"]["columns"]
    cache = {}

    def edges_of(k):
        if k not in cache:
            r, _ = seq.scan(k)
            cache[k] = lidar_edge_points(rig, r, with_cols=True)
        return cache[k]
    taus = traj.taus
    for c in range(0, len(cam_tau), step):
        if not cam_ok[c]:
            continue
        k = int(np.argmin(np.abs(taus - cam_tau[c])))
        if k < 1 or k >= len(taus) - 1 or not (lid_ok[k] and lid_ok[k - 1] and traj.ok[k] and traj.ok[k - 1]):
            continue
        Pa, ca = edges_of(k)
        Pb, cb = edges_of(k - 1)
        ma = ca < ncol // 2
        mb = cb >= ncol // 2
        P = np.concatenate([Pa[ma], Pb[mb]])
        tp = np.concatenate([taus[k] + ca[ma] / ncol * sweep, taus[k - 1] + cb[mb] / ncol * sweep])
        Rp, pp, _ = traj.pose(tp)
        Rc, pc, _ = traj.pose(np.array([cam_tau[c]]))
        w = np.einsum('nij,nj->ni', Rp, P) + pp
        Pl = (w - pc[0]) @ Rc[0]
        Pl = preselect(rig, Pl, T0)
        if len(Pl) < 30:
            continue
        img = cv2.imread(seq.path + "/camera/frames/" + seq.cam_files[c], cv2.IMREAD_GRAYSCALE)
        dt, valid = image_dt(rig, img)
        views.append((Pl, np.minimum(dt, 50).astype(np.float32), valid))
    return views


def refine(rig, views, T0, log=print):
    prob = EdgeProblem(rig, views)
    x = np.zeros(6)
    for cap, scale in ((10.0, 0.004), (6.0, 0.002), (4.0, 0.001), (3.0, 0.0005)):
        prob.cap = cap
        S = np.full(6, scale)
        S[3:] *= 2
        fun = lambda y: prob.cost(x + y * S, T0)
        res = minimize(fun, np.zeros(6), method="Powell", options={"xtol": 1e-2, "ftol": 1e-7, "maxfev": 3000})
        x = x + res.x * S
        log("moving edge cap", cap, np.degrees(x[:3]), x[3:], res.fun, res.nfev)
    T = np.eye(4)
    T[:3, :3] = lo.expSO3(x[:3]) @ T0[:3, :3]
    T[:3, 3] = T0[:3, 3] + x[3:]
    return T
