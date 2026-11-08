"""Single-view geometric grasp candidate generator with per-candidate features.

Candidates are parallel-jaw grasps in the GraspNet convention (R columns: approach, closing, z;
t = gripper origin, fingers span x in (-0.02, depth), contact line at x = depth). Each candidate is
centred on the observed cross-section of the object inside the finger slab, opened just wider than
it, and checked against the observed cloud with the evaluator's own gripper box model.
"""
import time
import warnings
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore', category=RuntimeWarning)

FW, DB, HH = 0.01, 0.02, 0.01        # finger width, base depth, half finger height
VOX = 0.005
KNN = 1024
YB = 0.005                            # occupancy bin along closing axis
NYB = 30                              # bins over [-0.075, 0.075]


def backproject(depth, K, scale):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth.astype(np.float64) / scale
    m = (z > 0.1) & (z < 1.5)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x[m], y[m], z[m]], -1)


def frames(a, theta):
    """a (S,3) unit approaches, theta (T,) -> R (S,T,3,3) with column0=a."""
    ref = np.where(np.abs(a[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    b = np.cross(a, ref); b /= np.linalg.norm(b, axis=1, keepdims=True)
    c = np.cross(a, b)
    y = np.cos(theta)[None, :, None] * b[:, None] + np.sin(theta)[None, :, None] * c[:, None]
    A = np.broadcast_to(a[:, None], y.shape)
    z = np.cross(A, y)
    return np.stack([A, y, z], -1)


def preprocess(depth, K, scale):
    pts = backproject(depth, K, scale)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts)).voxel_down_sample(VOX)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.015, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.zeros(3))
    P = np.asarray(pcd.points); N = np.asarray(pcd.normals)
    o3d.utility.random.seed(0)
    plane, inl = pcd.segment_plane(0.006, 3, 500)
    nt, off = np.asarray(plane[:3]), plane[3]
    if off < 0:
        nt, off = -nt, -off
    H = P @ nt + off
    obj = H > 0.008
    # clusters of object points
    lab = -np.ones(len(P), int)
    if obj.sum() > 0:
        sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[obj]))
        lab[obj] = np.asarray(sub.cluster_dbscan(eps=0.012, min_points=5))
    return dict(P=P, N=N, H=H, nt=nt, off=off, obj=obj, lab=lab)


def sample_seeds(pp, n_seeds, rng):
    P, obj, lab = pp['P'], pp['obj'], pp['lab']
    cand = np.nonzero(obj & (lab >= 0))[0]
    if len(cand) == 0:
        cand = np.nonzero(obj)[0]
    if len(cand) == 0:
        cand = np.arange(len(P))
    # spread seeds: voxelise at 1cm, one random point per voxel, then subsample
    key = np.floor(P[cand] / 0.012).astype(np.int64)
    _, first = np.unique(key, axis=0, return_index=True)
    cand = cand[first]
    if len(cand) > n_seeds:
        cand = rng.choice(cand, n_seeds, replace=False)
    else:
        cand = cand[rng.permutation(len(cand))]
    return cand


def _interval(occ, c0):
    """occ (M,NYB) bool occupancy; find contiguous occupied interval (gaps of >=2 empty bins end it)
    around bin c0. returns lo, hi bin indices (inclusive) and ok flag."""
    M = occ.shape[0]
    lo = np.full(M, c0); hi = np.full(M, c0)
    alive = np.ones(M, bool)
    for k in range(1, c0 + 1):            # leftwards
        i = c0 - k
        gap = ~occ[:, i] & ((i - 1 < 0) | ~occ[:, max(i - 1, 0)])
        alive &= ~gap
        lo = np.where(alive & occ[:, i], i, lo)
    alive = np.ones(M, bool)
    for k in range(1, NYB - c0):
        i = c0 + k
        gap = ~occ[:, i] & ((i + 1 >= NYB) | ~occ[:, min(i + 1, NYB - 1)])
        alive &= ~gap
        hi = np.where(alive & occ[:, i], i, hi)
    # touching the ends -> object too wide
    ok = (lo > 0) & (hi < NYB - 1)
    return lo, hi, ok


def project(X, K):
    z = np.maximum(X[..., 2], 1e-6)
    u = K[0, 0] * X[..., 0] / z + K[0, 2]
    v = K[1, 1] * X[..., 1] / z + K[1, 2]
    return u, v, z


def generate(depth, K, scale, n_seeds=400, n_rot=12, depths=(0.01, 0.02, 0.03, 0.04),
             margins=(0.006, 0.014, 0.024), seed=0, with_feats=True, deadline=None):
    rng = np.random.default_rng(seed)
    pp = preprocess(depth, K, scale)
    P, Nrm, H, nt = pp['P'], pp['N'], pp['H'], pp['nt']
    tree = pp['tree'] = cKDTree(P)
    seeds = sample_seeds(pp, n_seeds, rng)
    S = len(seeds)
    ps = P[seeds]; ns = Nrm[seeds]
    ray = ps / np.linalg.norm(ps, axis=1, keepdims=True)
    appr = [-ns, np.broadcast_to(-nt, ps.shape).copy(), ray]
    a4 = -ns - nt[None]; a4 /= np.linalg.norm(a4, axis=1, keepdims=True) + 1e-9
    appr.append(a4)
    NA = len(appr)
    theta = np.arange(n_rot) * np.pi / n_rot
    Rs = np.stack([frames(a, theta) for a in appr], 1)            # (S,NA,T,3,3)
    Rs = Rs.reshape(S, NA * n_rot, 3, 3)
    aid = np.repeat(np.arange(NA), n_rot)
    dist, nb = tree.query(ps, k=KNN, distance_upper_bound=0.1)
    valid = np.isfinite(dist)
    nb = np.where(valid, nb, 0)

    out = {k: [] for k in ['R', 't', 'w', 'd', 'seed', 'aid']}
    chunk = 24
    c0 = NYB // 2
    for s0 in range(0, S, chunk):
        if deadline is not None and time.time() > deadline:
            break
        sl = slice(s0, min(S, s0 + chunk))
        Q = P[nb[sl]] - ps[sl, None]                               # (s,K,3)
        Q = np.where(valid[sl, :, None], Q, 10.0)
        L = np.einsum('skj,scjm->sckm', Q, Rs[sl])                 # (s,C,K,3)
        x, y, z = L[..., 0], L[..., 1], L[..., 2]
        inz = np.abs(z) < HH
        ss, cc = x.shape[:2]
        yb0 = np.floor(y / YB).astype(np.int64) + c0
        inr = inz & (yb0 >= 0) & (yb0 < NYB)
        for d in depths:
            slab = inz & (x > -DB) & (x < d)
            yb = np.where(slab & inr, yb0, NYB)
            occ = np.zeros((ss * cc, NYB + 1), bool)
            occ[np.repeat(np.arange(ss * cc), yb.shape[-1]), yb.reshape(-1)] = True
            occ = occ[:, :NYB]
            occ[:, c0] = True                                      # seed bin always counted
            lo, hi, ok0 = _interval(occ, c0)
            ylo = (lo - c0) * YB; yhi = (hi - c0 + 1) * YB
            ymid = ((ylo + yhi) / 2).reshape(ss, cc)
            yc = y - ymid[..., None]
            ay = np.abs(yc)
            for margin in margins:
                w = ((yhi - ylo) + 2 * margin).reshape(ss, cc)
                ok = ok0.reshape(ss, cc) & (w <= 0.1)
                hw = w[..., None] / 2
                band = (ay > hw) & (ay < hw + FW)
                coll = (slab & band).any(-1) | (inz & (x > -DB - FW) & (x <= -DB) & (ay < hw + FW)).any(-1)
                si, ci = np.nonzero(ok & ~coll)
                if len(si) == 0:
                    continue
                R = Rs[sl][si, ci]
                out['R'].append(R); out['t'].append(ps[sl][si] + ymid[si, ci, None] * R[:, :, 1])
                out['w'].append(w[si, ci]); out['d'].append(np.full(len(si), d))
                out['seed'].append(seeds[sl][si]); out['aid'].append(aid[ci])
    if not out['R']:
        return None, pp
    cand = {k: np.concatenate(v) for k, v in out.items()}
    if with_feats:
        cand['feat'] = featurize(cand['R'], cand['t'], cand['w'], cand['d'], pp, depth, K, scale)
    return cand, pp



def global_feats(cand, pp, depth, K, scale):
    R, t, w, d = cand['R'], cand['t'], cand['w'], cand['d']
    a, yv, zv = R[:, :, 0], R[:, :, 1], R[:, :, 2]
    nt, off = pp['nt'], pp['off']
    P, Nrm, lab = pp['P'], pp['N'], pp['lab']
    s = cand['seed']
    ray = t / np.linalg.norm(t, axis=1, keepdims=True)
    c = t + d[:, None] * a
    f = [a @ ray.T.diagonal() if False else np.einsum('ij,ij->i', a, ray),
         -(a @ nt), np.einsum('ij,ij->i', a, -Nrm[s]), np.abs(yv @ nt),
         np.einsum('ij,ij->i', Nrm[s], -ray),
         t @ nt + off, c @ nt + off]
    # fingertip heights (lowest corner of finger boxes at the tip)
    tips = []
    for sy in (-1, 1):
        for sz in (-1, 1):
            p = c + (sy * (w / 2 + FW / 2))[:, None] * yv + sz * HH * zv
            tips.append(p @ nt + off)
    tips = np.stack(tips, 1)
    f += [tips.min(1)]
    base = t - (DB + FW / 2) * a
    f += [base @ nt + off]
    # cluster features
    cl = lab[s]
    cnt = np.bincount(lab[lab >= 0] if (lab >= 0).any() else np.zeros(0, int), minlength=lab.max() + 2)
    f += [np.where(cl >= 0, cnt[np.maximum(cl, 0)], 0)]
    # depth-image probes (positive = probe is behind the observed surface)
    zimg = depth.astype(np.float64) / scale
    Hh, Ww = zimg.shape

    def probe(X):
        u, v, zz = project(X, K)
        ui = np.clip(np.round(u).astype(int), 0, Ww - 1)
        vi = np.clip(np.round(v).astype(int), 0, Hh - 1)
        zo = zimg[vi, ui]
        inside = (u >= 0) & (u < Ww) & (v >= 0) & (v < Hh)
        zo = np.where((zo > 0) & inside, zo, 0.0)
        return np.where(zo > 0, zz - zo, 0.05)
    xs = np.linspace(-DB, 1, 4)                          # fraction placeholders
    fprobe = []
    for side in (-1, 1):
        pts = []
        for fx in (0.0, 0.33, 0.66, 1.0):
            for sz in (-0.7, 0.7):
                xx = -DB + fx * (d + DB)
                pts.append(t + xx[:, None] * a + (side * (w / 2 + FW / 2))[:, None] * yv + sz * HH * zv)
        pr = probe(np.stack(pts, 1))
        fprobe += [pr.max(1), pr.mean(1), (pr > 0.01).mean(1)]
    bp = []
    for fy in np.linspace(-1, 1, 5):
        for sz in (-0.7, 0.7):
            bp.append(t - (DB + FW / 2) * a + (fy * (w / 2 + FW))[:, None] * yv + sz * HH * zv)
    pr = probe(np.stack(bp, 1))
    fprobe += [pr.max(1), pr.mean(1)]
    # contact line profile
    lp = []
    for fy in np.linspace(-1, 1, 9):
        lp.append(c + (fy * w / 2)[:, None] * yv)
    pr = np.clip(probe(np.stack(lp, 1)), -0.05, 0.05)
    fprobe += list(pr.T)
    # beyond fingertips
    pb = []
    for side in (-1, 1):
        pb.append(c + 0.01 * a + (side * (w / 2 + FW / 2))[:, None] * yv)
    pr = probe(np.stack(pb, 1))
    fprobe += [pr.max(1)]
    f += fprobe
    return np.stack(f, 1)


def to_rows(cand, score, idx):
    n = len(idx)
    rows = np.zeros((n, 17))
    rows[:, 0] = score[idx]
    rows[:, 1] = cand['w'][idx]
    rows[:, 2] = 0.02
    rows[:, 3] = cand['d'][idx]
    rows[:, 4:13] = cand['R'][idx].reshape(n, 9)
    rows[:, 13:16] = cand['t'][idx]
    rows[:, 16] = -1
    return rows


# ---------------------------------------------------------------------------------------------
# v2 features: recomputed from (R, t, w, d) alone, using the gripper-frame neighbourhood.
GX = np.array([-0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
GYF = np.array([-1e9, -1.0, -0.5, 0.0, 0.5, 1.0, 1e9])       # relative to half width (inner)


def featurize(R, t, w, d, pp, depth, K, scale, chunk=2048):
    P, Nrm = pp['P'], pp['N']
    tree = pp.get('tree')
    if tree is None:
        tree = pp['tree'] = cKDTree(P)
    n = len(t)
    a = R[:, :, 0]
    c = t + (d / 2)[:, None] * a
    feats = []
    for s0 in range(0, n, chunk):
        sl = slice(s0, min(n, s0 + chunk))
        m = sl.stop - sl.start
        dist, nb = tree.query(c[sl], k=384, distance_upper_bound=0.09)
        valid = np.isfinite(dist)
        nb = np.where(valid, nb, 0)
        Q = np.where(valid[..., None], P[nb] - t[sl, None], 10.0)
        L = np.einsum('mkj,mjl->mkl', Q, R[sl])
        Ln = np.einsum('mkj,mjl->mkl', Nrm[nb], R[sl])
        x, y, z = L[..., 0], L[..., 1], L[..., 2]
        hw = (w[sl] / 2)[:, None]; dd = d[sl][:, None]
        ay = np.abs(y)
        # signed lateral coordinate: inside (|y|<hw) maps to [-1,1], fingers just outside
        yr = y / hw
        # occupancy grid: x bins x |y| bands x z bands
        xi = np.digitize(x, GX) - 1                                    # -1..len(GX)-1
        ybands = np.array([0, 0.5, 1.0, 1.0 + FW / 0.05, 1e9])
        yi = np.digitize(np.abs(yr), np.array([0.5, 1.0])) + 0  # 0: |yr|<.5, 1: <1, 2: >=1
        yo = ay - hw
        yi = np.where(yo > FW, 3, np.where(yo > 0, 2, yi))             # 2: finger band, 3: beyond
        yi = np.where(yo > FW + 0.02, 4, yi)
        zi = np.digitize(np.abs(z), np.array([HH, 2 * HH]))            # 0 in-slab, 1 near, 2 far
        okb = (xi >= 0) & (xi < len(GX) - 1) & (zi < 2) & valid
        nx, ny, nz = len(GX) - 1, 5, 2
        flat = ((xi * ny + yi) * nz + zi)
        flat = np.where(okb, flat, nx * ny * nz) + (np.arange(m) * (nx * ny * nz + 1))[:, None]
        grid = np.bincount(flat.ravel(), minlength=m * (nx * ny * nz + 1)).reshape(m, -1)[:, :-1]
        inz = np.abs(z) < HH
        slab = inz & (x > -DB) & (x < dd)
        inner = slab & (ay <= hw)
        n_inner = inner.sum(1)
        fing = slab & (ay > hw) & (ay < hw + FW)
        # clearance from fingers to nearest observed point in slab outside
        with np.errstate(all='ignore'):
            outside = slab & (ay > hw)
            clear = np.nan_to_num(np.nanmin(np.where(outside, ay - hw, np.nan), 1), nan=0.1)
            inside_y = np.where(inner, y, np.nan)
            ymax_in = np.nan_to_num(np.nanmax(inside_y, 1), nan=0.0)
            ymin_in = np.nan_to_num(np.nanmin(inside_y, 1), nan=0.0)
            gap_l = hw[:, 0] + ymin_in; gap_r = hw[:, 0] - ymax_in
            lm = inner & (y < ymin_in[:, None] + 0.006)
            rm = inner & (y > ymax_in[:, None] - 0.006)
            nl = np.maximum(lm.sum(1), 1); nr = np.maximum(rm.sum(1), 1)
            lny = (Ln[..., 1] * lm).sum(1) / nl
            rny = (Ln[..., 1] * rm).sum(1) / nr
            lnx = (Ln[..., 0] * lm).sum(1) / nl
            rnx = (Ln[..., 0] * rm).sum(1) / nr
            # normals of inner points: how much they face along closing axis
            ny_abs = (np.abs(Ln[..., 1]) * inner).sum(1) / np.maximum(n_inner, 1)
            nx_in = (Ln[..., 0] * inner).sum(1) / np.maximum(n_inner, 1)
            xin = np.where(inner, x, np.nan)
            xmin_in = np.nan_to_num(np.nanmin(xin, 1), nan=0.0)
            xmean_in = np.nan_to_num(np.nanmean(xin, 1), nan=0.0)
        # contact-line band at the fingertips: occupancy and mean closing-axis normal by y/hw bin
        band = (np.abs(z) < HH) & (np.abs(x - dd) < 0.006) & valid
        yq = np.clip(np.floor((yr + 1.5) / 0.3).astype(np.int64), 0, 9)
        fl = np.where(band, yq, 10) + (np.arange(m) * 11)[:, None]
        cnt = np.bincount(fl.ravel(), minlength=m * 11).reshape(m, 11)[:, :10]
        sny = np.bincount(fl.ravel(), weights=Ln[..., 1].ravel(), minlength=m * 11).reshape(m, 11)[:, :10]
        sny = sny / np.maximum(cnt, 1)
        f = np.concatenate([grid.astype(np.float64), cnt.astype(np.float64), sny,
                            np.stack([n_inner, fing.sum(1), clear, gap_l, gap_r, ymax_in - ymin_in,
                                      lny, rny, lnx, rnx, lm.sum(1), rm.sum(1), ny_abs, nx_in,
                                      xmin_in, xmean_in], 1)], 1)
        feats.append(f)
    F = np.concatenate(feats)
    cand = dict(R=R, t=t, w=w, d=d, seed=pp['tree'].query(t)[1])
    G = global_feats(cand, pp, depth, K, scale)
    return np.concatenate([np.stack([w, d], 1), F, G], 1)
