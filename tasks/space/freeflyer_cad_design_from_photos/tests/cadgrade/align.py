"""The rigid alignment of the whole unit (no scale, no mirror) onto the standard.

Source: the query samples of every tagged part that holds a usable solid, block by block in an order that does not
depend on how a pair's two tags are written (the pair's two blocks are ordered by their own surface centroids).
Target: the dense samples of the standard places those tags can take (both places of a pair when either of its tags
is present). Starts: the 24 proper rotations that map the axes onto the axes and the 24 that map the source's
principal axes onto the target's, each with the centroids matched. Three stages of trimmed point-to-plane ICP: every
start runs a short pass on a small subsample against a sparse subset of the target; the best distinct poses run a longer
pass on a larger subsample against the same sparse subset; the fits within the tie band of the best are refined on the
largest subsample against the dense target (points with no target point within the bound take no part in a fit). Among
the refined fits within the tie band of the best, the one that puts the tagged parts' surface centroids nearest their
places wins (a pair's tag counts its nearer place, so the labels inside a pair never decide).
"""
import itertools
import math

import numpy as np

from .util import apply, axis_angle_R, make_T, orthonormal, rot_angle_deg


def _principal_frame(P):
    c = P.mean(axis=0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    return c, V[:, np.argsort(w)[::-1]]


def _proper(M):
    return np.linalg.det(M) > 0


def axis_starts(src, dst):
    """The 24 proper rotations of the cube group, about the source centroid onto the target centroid."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            R = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                R[p, i] = s
            if not _proper(R):
                continue
            out.append(make_T(R, cd - R @ cs))
    return out


def pca_starts(src, dst):
    """The 24 proper rotations mapping the source cloud's principal axes onto the target's, centroids matched (the
    bimanual task's cad_tree.checks)."""
    cs, Vs = _principal_frame(src)
    cd, Vd = _principal_frame(dst)
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[:, i] = s * Vd[:, p]
            R = M @ Vs.T
            if np.linalg.det(R) < 0:
                continue
            out.append(make_T(R, cd - R @ cs))
    return out


def icp(src, dst, dst_n, tree, T0, iters, trim, bound_mm, tol=1e-9):
    """Point-to-plane ICP of src onto (dst, dst_n), trimmed; returns (T, trimmed RMS). (UAV grade.icp; stops when the
    step's norm, rotation in rad and translation in mm together, falls below tol)"""
    T = np.array(T0, float)
    for _ in range(int(iters)):
        P = apply(T, src)
        d, idx = tree.query(P, distance_upper_bound=bound_mm)
        far = ~np.isfinite(d)
        d = np.where(far, bound_mm, d)
        idx = np.where(far, 0, idx)
        k = max(6, int(len(d) * trim))
        keep = np.argsort(d, kind="stable")[:k]
        keep = keep[~far[keep]]                 # a point with no target point within the bound takes no part
        if len(keep) < 6:
            break
        p, q, n = P[keep], dst[idx[keep]], dst_n[idx[keep]]
        c = p.mean(axis=0)
        A = np.c_[np.cross(p - c, n), n]
        b = np.einsum("ij,ij->i", q - p, n)
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(x)):
            break
        w = float(np.linalg.norm(x[:3]))
        if w > 0.15:
            x[:3] *= 0.15 / w
        R = axis_angle_R(x[:3], float(np.linalg.norm(x[:3]))) if np.linalg.norm(x[:3]) > 1e-15 else np.eye(3)
        dT = make_T(R, c - R @ c + x[3:])
        T = dT @ T
        if float(np.linalg.norm(x)) < tol:
            break
    d, _ = tree.query(apply(T, src), distance_upper_bound=bound_mm)
    d = np.where(np.isfinite(d), d, bound_mm)
    k = max(1, int(len(d) * trim))
    return make_T(orthonormal(T[:3, :3]), T[:3, 3]), float(math.sqrt(np.mean(np.sort(d)[:k] ** 2)))


def _subsample(P, n):
    n = min(len(P), int(n))
    return P[np.linspace(0, len(P) - 1, n).astype(np.int64)] if n > 0 else P[:0]


def align(blocks, coarse_target, fine_target, choices, cfg, stats):
    """T (document -> standard) of the whole unit.

    blocks         [(query samples (n, 3), surface centroid (3,), tag)] in the label-independent order
    coarse_target  (points, normals, KD-tree) for the starts and the coarse passes
    fine_target    (points, normals, KD-tree) for the refinement
    choices        {tag: [standard place centroids the tag may take]}
    """
    A = cfg["alignment"]
    src_all = np.vstack([b[0] for b in blocks])
    sub = _subsample(src_all, A["n_coarse"])
    mid = _subsample(src_all, A["n_mid"])
    fine = _subsample(src_all, A["n_fine"])
    Dc, Nc, Tc = coarse_target
    Df, Nf, Tf = fine_target
    starts = axis_starts(src_all, Dc) + pca_starts(src_all, Dc)
    bound = float(A["bound_mm"])
    tol = float(A["icp_tol"])
    tie_abs, tie_rel = float(A["tie_abs_mm"]), float(A["tie_rel"])

    def distinct(cands, deg, mm, limit=None):
        out = []
        for c in sorted(cands, key=lambda x: (round(x[0], 9), x[1])):
            if all(rot_angle_deg(c[2][:3, :3] @ o[2][:3, :3].T) > deg or np.linalg.norm(c[2][:3, 3] - o[2][:3, 3]) > mm
                   for o in out):
                out.append(c)
            if limit is not None and len(out) >= limit:
                break
        return out

    def within_band(cands):
        best = min(c[0] for c in cands)
        return [c for c in cands if c[0] <= best + max(tie_abs, tie_rel * best)]

    # stage 1: every start, a short pass on a small subsample against the sparse target
    coarse = []
    for i, T0 in enumerate(starts):
        Tk, rk = icp(sub, Dc, Nc, Tc, T0, A["coarse_iters"], A["trim"], bound, tol)
        coarse.append((rk, i, Tk))
    # stage 2: the best distinct poses, a longer pass on a larger subsample, still against the sparse target
    stage2 = []
    for _rk, i, Tk in distinct(coarse, 2.0, 2.0, int(A["refine_candidates"])):
        Tr, rr = icp(mid, Dc, Nc, Tc, Tk, A["mid_iters"], A["trim"], bound, tol)
        stage2.append((rr, i, Tr))
    stage2 = distinct(stage2, 0.5, 0.5)
    # stage 3: the fits within the tie band, refined on the full subsample against the dense target
    stage3 = []
    for _rr, i, Tr in within_band(stage2):
        Tf3, rf = icp(fine, Df, Nf, Tf, Tr, A["icp_iters"], A["trim"], float(A["fine_bound_mm"]), tol)
        stage3.append((rf, i, Tf3))
    stage3 = distinct(stage3, 0.5, 0.5)
    tied = within_band(stage3)
    stats["alignment_coarse_distinct"] = len(stage2)
    cap = float(A["assembly_cap_mm"])

    def placement_error(T):
        d = []
        for _q, c, tag in blocks:
            p = apply(T, c)
            d.append(min(cap, min(float(np.linalg.norm(p - q)) for q in choices[tag])))
        return float(np.mean(d)) if d else 0.0

    scored = sorted(((round(placement_error(T), 6), round(r, 9), i, T, r) for r, i, T in tied),
                    key=lambda x: (x[0], x[1], x[2]))
    pe, _rr, i, T, r = scored[0]
    stats["alignment_residual_mm"] = float(r)
    stats["alignment_placement_error_mm"] = float(pe)
    stats["alignment_candidates_tied"] = len(tied)
    stats["alignment_start"] = int(i)
    stats["alignment_samples_fine"] = int(len(fine))
    return T
