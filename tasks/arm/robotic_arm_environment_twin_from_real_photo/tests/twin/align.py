"""Alignment of a twin into the true camera frame.

1. The twin's own camera frame: x_cam = R_cv^T (x_world - p), from its photo_cam (twin_camera.camera_frame).
2. R_pp (frame_alignment "principal_ray_then_scale"): the rotation taking e_z onto unit(K_t^-1 [cx_a, cy_a, 1]),
   i.e. the agent's principal ray onto the true ray through the pixel the agent declared as its principal point;
   clamped at 2 deg. "scale_only" leaves R_pp = I.
3. One global scale s* about the camera centre, fitted on matched items and iterated with the matching:
   s0 = median |c_t| / |c_a| over a bearing Hungarian (pairs under 3 deg); then repeatedly match on the capped IoU
   at s, drop pairs whose centroid residual |c_t - s c_a| exceeds 30 mm, refit s = sum c_t . c_a / sum |c_a|^2,
   until the pair set is stable (at most 4 refits). With fewer than 3 pairs s* = s0 (1 without a bearing pair).
"""
import math

import numpy as np


def rodrigues(axis, angle):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def principal_ray_rotation(K_t, K_a, clamp_deg=2.0):
    """(R_pp, info): R_pp e_z = unit(K_t^-1 [cx_a, cy_a, 1]), the angle clamped at clamp_deg."""
    K_t = np.asarray(K_t, float)
    K_a = np.asarray(K_a, float)
    vx = (K_a[0, 2] - K_t[0, 2]) / K_t[0, 0]
    vy = (K_a[1, 2] - K_t[1, 2]) / K_t[1, 1]
    info = {"cx_a": float(K_a[0, 2]), "cy_a": float(K_a[1, 2]), "cx_t": float(K_t[0, 2]), "cy_t": float(K_t[1, 2])}
    if vx == 0.0 and vy == 0.0:
        info.update({"angle_deg": 0.0, "clamped": False, "raw_angle_deg": 0.0})
        return np.eye(3), info
    v = np.array([vx, vy, 1.0])
    v /= np.linalg.norm(v)
    axis = np.array([-v[1], v[0], 0.0])                  # e_z x v
    ang = math.atan2(float(np.linalg.norm(axis)), float(v[2]))
    lim = math.radians(float(clamp_deg))
    info.update({"raw_angle_deg": math.degrees(ang), "clamped": ang > lim})
    ang = min(ang, lim)
    info["angle_deg"] = math.degrees(ang)
    return rodrigues(axis, ang), info


def frame_map(twin, R_pp=None):
    """(M, t) with x' = M x + t taking the twin's world into its (aligned) camera frame."""
    M = twin.R_cv.T if R_pp is None else R_pp @ twin.R_cv.T
    return M, -(M @ twin.p)


def bearing_pairs(ct, ca, max_deg):
    """Hungarian on the angle between the centroid bearings; pairs under max_deg."""
    from scipy.optimize import linear_sum_assignment
    if len(ct) == 0 or len(ca) == 0:
        return []
    ut = ct / np.linalg.norm(ct, axis=1)[:, None]
    ua = ca / np.linalg.norm(ca, axis=1)[:, None]
    ang = np.degrees(np.arccos(np.clip(ut @ ua.T, -1.0, 1.0)))
    cost = np.where(ang < max_deg, ang, 1e6)
    ri, cj = linear_sum_assignment(cost)
    return [(int(i), int(j)) for i, j in zip(ri, cj) if ang[i, j] < max_deg]


def ls_scale(ct, ca):
    num = float(np.sum(ct * ca))
    den = float(np.sum(ca * ca))
    return num / den if den > 0 else 1.0


def fit_scale(ct, ca, match_at, cfg):
    """(s*, pairs matched at s*, info). ct (n_t, 3), ca (n_a, 3): centroids in the true and the aligned agent camera
    frame at scale 1; match_at(s) -> [(i, j, capped_iou)] at scale s."""
    sc = cfg["scale"]
    trim_mm = float(sc["trim_mm"])
    hist = []
    bp = bearing_pairs(ct, ca, float(sc["bearing_deg"])) if len(ct) and len(ca) else []
    s0 = float(np.median([np.linalg.norm(ct[i]) / np.linalg.norm(ca[j]) for i, j in bp])) if bp else 1.0
    hist.append({"step": "bearing", "pairs": len(bp), "s": s0})
    s = s0
    pairs = match_at(s)
    hist.append({"step": "match", "s": s, "pairs": len(pairs)})
    for it in range(int(sc["iterations"])):
        kept = [(i, j) for i, j, _ in pairs if 1000.0 * float(np.linalg.norm(ct[i] - s * ca[j])) <= trim_mm]
        if len(kept) >= int(sc["min_pairs"]):
            s = ls_scale(ct[[i for i, _ in kept]], ca[[j for _, j in kept]])
        else:
            s = s0
        new = match_at(s)
        stable = [(i, j) for i, j, _ in new] == [(i, j) for i, j, _ in pairs]
        hist.append({"step": "refit", "iteration": it + 1, "kept_pairs": len(kept), "s": s, "pairs": len(new),
                     "stable": stable})
        pairs = new
        if stable:
            break
    return s, pairs, {"s0": s0, "bearing_pairs": len(bp), "history": hist}
