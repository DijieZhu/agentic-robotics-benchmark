"""LiDAR scan-to-scan odometry with constant-velocity deskew (point-to-plane ICP)."""
import numpy as np
from scipy.spatial import cKDTree


def hat(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def expSO3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3) + hat(w)
    k = w / th
    K = hat(k)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def logSO3(R):
    c = np.clip((np.trace(R) - 1) / 2, -1, 1)
    th = np.arccos(c)
    if th < 1e-9:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    return th / (2 * np.sin(th)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def exp_batch(W):
    th = np.linalg.norm(W, axis=1)
    ths = np.where(th < 1e-12, 1.0, th)
    k = W / ths[:, None]
    K = np.zeros((len(W), 3, 3))
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]; K[:, 1, 0] = k[:, 2]
    K[:, 1, 2] = -k[:, 0]; K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    s = np.sin(th)[:, None, None]; c = (1 - np.cos(th))[:, None, None]
    return np.eye(3)[None] + s * K + c * K @ K


def range_normals(rig, r):
    """points (rings, cols, 3), valid mask, normals (rings, cols, 3) from the range image."""
    P = rig.ray_dirs * r[..., None]
    valid = r > 0
    # neighbours: +-2 columns, +-1 ring
    dc = np.roll(P, -2, axis=1) - np.roll(P, 2, axis=1)
    vc = valid & np.roll(valid, -2, axis=1) & np.roll(valid, 2, axis=1)
    dr = np.zeros_like(P)
    dr[1:-1] = P[2:] - P[:-2]
    vr = np.zeros_like(valid)
    vr[1:-1] = valid[2:] & valid[:-2]
    n = np.cross(dc, dr)
    nn = np.linalg.norm(n, axis=-1)
    ok = valid & vc & vr & (nn > 1e-9)
    n = n / np.where(nn > 1e-9, nn, 1)[..., None]
    # reject depth discontinuities
    rr = r
    jump = (np.abs(np.roll(rr, -2, 1) - rr) > 0.08 * rr + 0.1) | (np.abs(np.roll(rr, 2, 1) - rr) > 0.08 * rr + 0.1)
    ok &= ~jump
    return P, valid, n, ok


class Scan(object):
    def __init__(self, rig, r, refl, max_range=45.0, voxel=0.25):
        P, valid, n, ok = range_normals(rig, r)
        cols = np.broadcast_to(np.arange(r.shape[1])[None, :], r.shape)
        m = ok & (r < max_range) & (r > 0.8)
        self.pts = P[m]
        self.nrm = n[m]
        self.col = cols[m]
        self.frac = cols[m] / float(r.shape[1])
        self.rotcols = None
        # voxel subsample for the source
        key = np.floor(self.pts / voxel).astype(np.int64)
        key = key[:, 0] * 73856093 ^ key[:, 1] * 19349663 ^ key[:, 2] * 83492791
        _, idx = np.unique(key, return_index=True)
        self.sub = idx


def deskew(pts, frac, xi, rot=None):
    """points taken at fraction frac of the sweep -> frame at fraction 0, with twist xi=(w,v) per sweep.
    rot: optional (N,3) rotation vectors of each point's sensor frame in the column-0 frame (from the gyro)."""
    R = exp_batch(frac[:, None] * xi[None, :3] if rot is None else rot)
    return np.einsum('nij,nj->ni', R, pts) + frac[:, None] * xi[None, 3:]


def icp(src, tgt, T0, xi, tree_cache=None, iters=12):
    """T (4x4): pose of src (column-0 frame) in tgt frame. xi: twist per sweep for deskew."""
    tr = None if tgt.rotcols is None else tgt.rotcols[tgt.col]
    sr = None if src.rotcols is None else src.rotcols[src.col[src.sub]]
    tp = deskew(tgt.pts, tgt.frac, xi, tr)
    tn = np.einsum('nij,nj->ni', exp_batch(tgt.frac[:, None] * xi[None, :3] if tr is None else tr), tgt.nrm)
    tree = cKDTree(tp)
    sp = deskew(src.pts[src.sub], src.frac[src.sub], xi, sr)
    T = T0.copy()
    dmax = 1.0
    for it in range(iters):
        q = sp @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(q, distance_upper_bound=dmax)
        m = np.isfinite(d)
        qm = q[m]; nm = tn[j[m]]; pm = tp[j[m]]
        res = np.einsum('ij,ij->i', qm - pm, nm)
        s = max(np.median(np.abs(res)) * 3, 0.02)
        w = 1.0 / (1.0 + (res / s) ** 2)
        A = np.hstack([np.cross(qm, nm), nm])
        H = (A * w[:, None]).T @ A
        g = (A * w[:, None]).T @ res
        dx = -np.linalg.solve(H + 1e-6 * np.eye(6), g)
        dT = np.eye(4)
        dT[:3, :3] = expSO3(dx[:3]); dT[:3, 3] = dx[3:]
        T = dT @ T
        dmax = max(0.3, dmax * 0.7)
        if np.linalg.norm(dx) < 1e-6:
            break
    return T, H


def twist_of(T):
    return np.concatenate([logSO3(T[:3, :3]), T[:3, 3]])


def odometry(rig, seq, rotfun=None, voxel=0.25):
    """relative motion of consecutive scans: dict k -> T_k_k+1, the pose of scan k+1 in scan k.
    rotfun(k) -> (columns, 3) rotation vectors of each column's frame in the column-0 frame, or None."""
    n = len(seq.scan_stamps)
    scans = {}
    rel = {}
    T = np.eye(4)
    prev = None
    for k in range(n):
        r, refl = seq.scan(k)
        sc = Scan(rig, r, refl, voxel=voxel)
        if rotfun is not None:
            sc.rotcols = rotfun(k)
        if prev is not None and len(sc.pts) > 500 and len(prev.pts) > 500:
            xi = twist_of(T)
            for rep in range(2):
                T, H = icp(sc, prev, T, xi)
                xi_new = twist_of(T)
                if np.linalg.norm(xi_new - xi) < 1e-4:
                    break
                xi = xi_new
            rel[k - 1] = T.copy()
        prev = sc
    return rel
