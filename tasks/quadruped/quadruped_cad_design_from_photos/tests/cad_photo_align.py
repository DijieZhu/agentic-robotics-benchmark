"""cad_photo.align: per-part rigid alignment of a submitted part onto the ground-truth part (hardware_design.md 6.4).

Shipped twice, byte-identical: inside tests/ (as cad_photo_align.py) and in the agent's practice checker. Pure geometry
(numpy, scipy, trimesh); no hidden data. Lengths mm.

    align_part(agent_mesh_world, gt_mesh_world, seed) -> {"T": 4x4 (agent world -> gt world), "rms_mm", "multistart",
                                                          "offset_mm", "rotation_deg", "gt_pts", "gt_tree", ...}

Initial pose = identity (both parts placed at q = 0 by their own assembly.json). Point-to-plane ICP on surface samples
(20 000 of the submitted part against 200 000 of the ground truth, trimmed to the best 90 %, 40 iterations), every step
restricted so that the total correction stays within 15 mm / 15 deg; if the final trimmed RMS exceeds 3 mm, a multistart
from the 24 proper rotations that map the submitted part's principal axes onto the ground truth's (no restriction; each
start on a 1:10 subsample for 15 iterations, the two best refined in full); the lowest RMS wins. The alignment's own offset is recorded, never scored (placement is graded by the kinematic criterion).
"""

import math

import numpy as np

N_SRC = 20000
N_DST = 200000
TRIM = 0.90
ITERS = 40
MAX_T_MM = 15.0
MAX_R_DEG = 15.0
MULTISTART_RMS_MM = 3.0
QUERY_BOUND_MM = 40.0        # nearest-neighbour search radius (far points are clamped: a KD-tree query of far points is slow)
STEP_ROT = 0.15              # per-iteration step limits (rad, mm) of the unrestricted ICP
STEP_T_MM = 15.0


def sample_surface(mesh, n, seed):
    import trimesh
    pts, fid = trimesh.sample.sample_surface(mesh, int(n), seed=int(seed) % (2 ** 32))
    return np.asarray(pts, float), np.asarray(mesh.face_normals[fid], float)


def surface_distances(query, samples, normals, tree, disk_mm=0.4):
    """Distance of each query point to the surface represented by dense samples with normals: the distance to the
    tangent disk (radius disk_mm) of the nearest sample (point-to-plane inside the disk, Euclidean beyond its rim), so
    points ON the sampled surface read ~0 whatever the sample spacing."""
    d, idx = tree.query(query)
    v = query - samples[idx]
    dn = np.abs(np.einsum("ij,ij->i", v, normals[idx]))
    dl = np.sqrt(np.maximum(0.0, d * d - dn * dn))
    return np.sqrt(dn * dn + np.maximum(0.0, dl - disk_mm) ** 2)


def _rodrigues(w):
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def _rot_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))))


def _apply(T, P):
    return P @ T[:3, :3].T + T[:3, 3]


def _query(tree, P):
    d, idx = tree.query(P, distance_upper_bound=QUERY_BOUND_MM)
    far = ~np.isfinite(d)
    d = np.where(far, QUERY_BOUND_MM, d)
    idx = np.where(far, 0, idx)
    return d, idx


def trimmed_rms(T, src, tree, trim=TRIM):
    d, _ = _query(tree, _apply(T, src))
    k = max(1, int(len(d) * trim))
    return float(math.sqrt(np.mean(np.sort(d)[:k] ** 2)))


def icp(src, dst, dst_n, tree, T0, iters=ITERS, trim=TRIM, restrict=True, pivot=None):
    """Point-to-plane ICP of `src` onto (dst, dst_n); returns (T, trimmed RMS). `restrict`: the total correction
    relative to T0 is clamped to MAX_T_MM / MAX_R_DEG (rotation about `pivot`, default the source centroid)."""
    T = np.array(T0, float)
    piv = np.asarray(src.mean(axis=0) if pivot is None else pivot, float)
    for _ in range(int(iters)):
        P = _apply(T, src)
        if not np.all(np.isfinite(P)):
            break
        d, idx = _query(tree, P)
        k = max(6, int(len(d) * trim))
        keep = np.argsort(d)[:k]
        p, q, n = P[keep], dst[idx[keep]], dst_n[idx[keep]]
        c = _apply(T, piv[None, :])[0]
        A = np.c_[np.cross(p - c, n), n]
        b = np.einsum("ij,ij->i", q - p, n)
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(x)):
            break
        w = float(np.linalg.norm(x[:3]))
        if w > STEP_ROT:
            x[:3] *= STEP_ROT / w
        tn = float(np.linalg.norm(x[3:]))
        if tn > STEP_T_MM:
            x[3:] *= STEP_T_MM / tn
        R = _rodrigues(x[:3])
        dT = np.eye(4)
        dT[:3, :3] = R
        dT[:3, 3] = c - R @ c + x[3:]
        Tn = dT @ T
        if restrict:
            D = Tn @ np.linalg.inv(T0)
            r = _rot_deg(D[:3, :3])
            t = float(np.linalg.norm(_apply(D, piv[None, :])[0] - piv))
            if r > MAX_R_DEG or t > MAX_T_MM:
                break
        T = Tn
        if float(np.linalg.norm(x)) < 1e-7:
            break
    return T, trimmed_rms(T, src, tree, trim)


def _principal(P):
    c = P.mean(axis=0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    return c, V[:, np.argsort(w)[::-1]]


def pca_starts(src, dst):
    """The 24 proper rotations mapping the source principal axes onto the destination's (axis permutations x sign flips
    with det +1), each with the centroids matched."""
    cs, Vs = _principal(src)
    cd, Vd = _principal(dst)
    out = []
    import itertools
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[:, i] = s * Vd[:, p]
            R = M @ Vs.T
            if np.linalg.det(R) < 0:
                continue
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = cd - R @ cs
            out.append(T)
    return out


def align_part(agent_mesh, gt_mesh, seed, gt_samples=None):
    """Align the submitted part (world frame at q = 0) onto the ground-truth part (world frame at q = 0)."""
    from scipy.spatial import cKDTree
    src, _src_n = sample_surface(agent_mesh, N_SRC, seed * 2 + 1)
    if gt_samples is None:
        dst, dst_n = sample_surface(gt_mesh, N_DST, seed * 2 + 2)
    else:
        dst, dst_n = gt_samples
    tree = cKDTree(dst)
    T, rms = icp(src, dst, dst_n, tree, np.eye(4), restrict=True)
    multistart = False
    if rms > MULTISTART_RMS_MM:
        # the 24 starts on a coarse subsample (2 000 against 20 000 points, 15 iterations), the two best refined in full
        multistart = True
        sc, dc, dcn = src[::10], dst[::10], dst_n[::10]
        tc = cKDTree(dc)
        coarse = []
        for T0 in pca_starts(src, dst):
            Tk, rk = icp(sc, dc, dcn, tc, T0, iters=15, restrict=False)
            coarse.append((rk, Tk))
        coarse.sort(key=lambda x: x[0])
        best = (rms, T)
        for _rk, Tk in coarse[:2]:
            Tf, rf = icp(src, dst, dst_n, tree, Tk, iters=ITERS, restrict=False)
            if rf < best[0]:
                best = (rf, Tf)
        rms, T = best
    piv = src.mean(axis=0)
    return {"T": T, "rms_mm": float(rms), "multistart": multistart,
            "offset_mm": float(np.linalg.norm(_apply(T, piv[None, :])[0] - piv)), "rotation_deg": float(_rot_deg(T[:3, :3])),
            "gt_pts": dst, "gt_normals": dst_n, "gt_tree": tree, "src_pts": src}
