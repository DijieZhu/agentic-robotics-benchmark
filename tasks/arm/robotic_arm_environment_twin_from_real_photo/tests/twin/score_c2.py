"""C2 pose and shape: per matched pair q = b_pos x b_rot x b_iou (pose.combine "product"; "mean" and "geomean" are the
alternatives); C2 = sum q / max(n_true, n_submitted), per
work area when the areas are balanced (twin/regions.py: the mean over the areas of the area's sum of q over
max(n_true(area), n_submitted(area))).

- b_pos = band(|c_S - c_T|, 5, 25 mm), centroids of the solids in the true camera frame (agent at s*);
- b_rot = band(angle, 3, 20 deg), the angle of a restricted point-to-plane ICP rotation of the agent surface onto
  the true surface about the common centroid (the agent solid translated onto the true centroid first), from the
  identity, at most 30 deg, with a uniform scale fitted alongside so a size error is not read as a rotation; the ICP
  solves only for turns about the true shape's stiff axes (an axis is soft when the true solid turned 30 deg about
  it keeps IoU >= 0.90 with itself), so the spin of a can costs nothing whatever shape the agent used;
- b_iou = linear in the capped IoU at s*, 0 at 0.30 and 1 at 0.80 (the matching's capped IoU, 1 mm cells for
  items thinner than 40 mm).

The ICP rotation is the matcher's (match.Matcher.icp, the same rotation its oversize cap uses), so a pair is turned
onto the truth once and that one rotation is both charged here and used to measure the cap.

The per-pair q (true body -> agent body, q) is exported for the physics criteria.
"""
import math

import numpy as np

from .align import rodrigues
from .config import band_down, band_up
from .regions import mean_over_areas


def rot_angle_deg(R):
    c = (float(np.trace(R)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def icp_rotation(src, dst, dst_n, c, cfg, stiff=None):
    """Rotation about c taking src onto the surface (dst, dst_n): (R, angle_deg, iterations, scale).

    Point-to-plane ICP on the best icp_trim share of the correspondences, from the identity, rotation steps of at
    most icp_step_rad, stopped before the total rotation exceeds icp_max_deg. Only rotations about the `stiff` axes
    (columns, an orthonormal basis; default all three) are solved for: a turn about an axis the true shape is nearly
    symmetric about is invisible and is not charged. A uniform scale about c is fitted alongside (a nuisance
    parameter): without it a solid that is only too large or too small drifts into a spurious rotation (about 10 deg
    on average for a 10 % size error, measured on the ground-truth hulls)."""
    from scipy.spatial import cKDTree
    pc = cfg["pose"]
    B = np.eye(3) if stiff is None else np.asarray(stiff, float).reshape(3, -1)
    tree = cKDTree(dst)
    R = np.eye(3)
    sig = 1.0
    trim = float(pc["icp_trim"])
    step = float(pc["icp_step_rad"])
    lim = float(pc["icp_max_deg"])
    fit_scale = bool(pc.get("icp_fit_scale", True))
    lo_s, hi_s = pc.get("icp_scale_range", [0.5, 2.0])
    nb = B.shape[1]
    it = 0
    for it in range(1, int(pc["icp_iterations"]) + 1):
        P = c + sig * ((src - c) @ R.T)
        d, idx = tree.query(P)
        k = max(6, int(len(d) * trim))
        keep = np.argsort(d, kind="stable")[:k]
        p, q, n = P[keep], dst[idx[keep]], dst_n[idx[keep]]
        b = np.einsum("ij,ij->i", q - p, n)
        if not np.any(b):
            break
        A = np.cross(p - c, n) @ B
        if fit_scale:
            A = np.c_[A, np.einsum("ij,ij->i", p - c, n)]
        if A.shape[1] == 0:
            break
        x = np.linalg.lstsq(A, b, rcond=None)[0]
        if not np.all(np.isfinite(x)):
            break
        om = B @ x[:nb] if nb else np.zeros(3)
        w = float(np.linalg.norm(om))
        eps = float(np.clip(x[nb], -0.1, 0.1)) if fit_scale else 0.0
        if w < 1e-12 and abs(eps) < 1e-12:
            break
        if w > step:
            om *= step / w
            w = step
        Rn = (rodrigues(om / w, w) if w >= 1e-12 else np.eye(3)) @ R
        if rot_angle_deg(Rn) > lim:
            break
        R = Rn
        sig = float(np.clip(sig * (1.0 + eps), lo_s, hi_s))
    return R, rot_angle_deg(R), it, sig


def true_surfaces(true_solids, cfg):
    pc = cfg["pose"]
    return [T.surface(int(pc["icp_samples"]), int(pc["seed"])) for T in true_solids]


def symmetry_axes(T, surface, keys, h, cfg):
    """(stiff basis (3, n), info) of a true solid. Candidate axes: the eigenvectors of the point-to-plane rotation
    Hessian sum (r x n)(r x n)^T over the surface samples (r from the centroid). An axis is soft when the solid
    turned by icp_max_deg about it keeps IoU >= sym_iou with itself (the spin of a can: 0.955-0.971 on the
    ground-truth cans; the soft-looking long axis of a long box stays below 0.72)."""
    from .align import rodrigues as rod
    pc = cfg["pose"]
    P, N = surface
    c = T.volume_centroid()[1]
    J = np.cross(P - c, N)
    w, U = np.linalg.eigh(J.T @ J / max(1, len(J)))
    ang = math.radians(float(pc["icp_max_deg"]))
    stiff, rows = [], []
    for a in range(3):
        R = rod(U[:, a], ang)
        ks = T.transformed(R, c - R @ c).keys(h)
        ni = int(np.intersect1d(keys, ks, assume_unique=True).size)
        iou = ni / float(keys.size + ks.size - ni) if (keys.size + ks.size - ni) > 0 else 0.0
        soft = iou >= float(pc["sym_iou"])
        rows.append({"axis": [float(v) for v in U[:, a]], "self_iou": iou, "soft": soft,
                     "normal_motion_mm_per_rad": 1000.0 * math.sqrt(max(0.0, float(w[a])))})
        if not soft:
            stiff.append(U[:, a])
    B = np.array(stiff).T if stiff else np.zeros((3, 0))
    return B, {"axes": rows, "n_soft": 3 - len(stiff)}


def score(pairs, true_solids, agent_solids, s_star, details, n_true, n_sub, cfg, surfaces=None, stiff=None,
          areas=None):
    """details: {(i, j): pair dict of the matcher at s*} (with its ICP rotation); stiff: per true item the stiff
    rotation basis (default all axes), used only when a pair's details carry no ICP result. Returns (C2, info with
    per-true-item rows)."""
    pc = cfg["pose"]
    per = {}
    total = 0.0
    for i, j, capped in pairs:
        T, S = true_solids[i], agent_solids[j]
        cT = T.volume_centroid()[1]
        cS = s_star * S.volume_centroid()[1]
        pos_mm = 1000.0 * float(np.linalg.norm(cS - cT))
        d0 = details.get((i, j), {})
        if "rot_deg" in d0:
            ang, iters, sig = d0["rot_deg"], d0["icp_iterations"], d0["icp_scale"]
        else:
            if surfaces is None:
                surfaces = true_surfaces(true_solids, cfg)
            Ss = S.scaled(s_star)
            P, _N = Ss.surface(int(pc["icp_samples"]), int(pc["seed"]))
            P = P + (cT - cS)
            dst, dst_n = surfaces[i]
            _R, ang, iters, sig = icp_rotation(P, dst, dst_n, cT, cfg, None if stiff is None else stiff[i])
        b_pos = band_down(pos_mm, *pc["pos_mm"])
        b_rot = band_down(ang, *pc["rot_deg"])
        b_iou = band_up(capped, *pc["iou"])
        comb = pc.get("combine", "mean")
        if comb == "product":                    # all three must hold (a grasp needs the place, the turn and the shape)
            q = b_pos * b_rot * b_iou
        elif comb == "geomean":
            q = (b_pos * b_rot * b_iou) ** (1.0 / 3.0)
        else:
            q = (b_pos + b_rot + b_iou) / 3.0
        total += q
        d = details.get((i, j), {})
        per[T.name] = {"agent": S.name, "q": q, "b_pos": b_pos, "b_rot": b_rot, "b_iou": b_iou,
                       "pos_err_mm": pos_mm, "rot_deg": ang, "icp_iterations": iters, "icp_scale": sig,
                       "capped_iou": capped,
                       "iou": d.get("iou"), "cap": d.get("cap"), "grid_mm": d.get("grid_mm"),
                       "cap_axes": d.get("cap_axes"), "extent_agent_mm": d.get("extent_agent_mm"),
                       "extent_true_mm": d.get("extent_true_mm")}
    den = max(n_true, n_sub)
    plain = total / den if den > 0 else 0.0
    info = {"per_true": per, "n_true": n_true, "n_submitted": n_sub, "denominator": den, "unbalanced": plain}
    if not (areas and areas.get("balanced")):
        return plain, info
    pa = {}
    for r in areas["areas"]:
        nt, ns = areas["n_true"][r], areas["n_sub"][r]
        sq = sum(v["q"] for t, v in per.items() if areas["of_true"].get(t) == r)
        d = max(nt, ns)
        pa[r] = {"sum_q": sq, "n_true": nt, "n_submitted": ns, "denominator": d, "score": (sq / d) if d > 0 else None}
    info["per_area"] = pa
    return mean_over_areas({r: v["score"] for r, v in pa.items()}), info
