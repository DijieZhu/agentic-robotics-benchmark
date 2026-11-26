"""The one-flight fit (rules Z3 and Z6): are the declared camera poses of a flight one place?

Inputs, for the key frames with a valid declared pose: the declared wide-camera pose (p_k, R_k) in the twin frame
(cameras.json) and the on-board estimate's wide-camera pose (p^_k, R^_k) for the same frame, read at the frame's
stamp minus the configured offset and multiplied by the camera mounting (posechain.estimate_wide_pose).

  1. start with every such frame as an inlier;
  2. on the current inliers: R_g = the rotation nearest (in the Frobenius sense) to sum_k R^_k R_k^T; with R_g
     fixed, x_k = R_g p_k, least squares for p^_k ~ s x_k + t: s = sum (x_k - x_m) . (p^_k - p^_m) /
     sum |x_k - x_m|^2 clamped to the configured interval (s = 1 when that denominator is below the configured
     smallest spread), t = p^_m - s x_m (x_m, p^_m the inliers' means);
  3. residuals of every frame: e_k = |s R_g p_k + t - p^_k|, a_k = the angle between R_g R_k and R^_k; a frame is
     an inlier when e_k <= the position threshold and a_k <= the angle threshold;
  4. repeat 2 and 3 until the inlier set no longer changes, at most the configured number of fits; an empty inlier
     set ends the iteration.
Fewer inliers than the configured minimum: rule Z3 (the flight is zero). Otherwise every frame that is not an
inlier is an outlier: rule Z6 (that frame is zero).
"""
import numpy as np

from . import need
from .posechain import rotation_angle_deg


def proj_so3(M):
    """The rotation nearest to M (Frobenius norm)."""
    U, _, Vt = np.linalg.svd(np.asarray(M, float))
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    return U @ D @ Vt


def params_from_config(cfg):
    return {"position_threshold": float(need(cfg, "gate.one_flight.position_threshold")),
            "angle_threshold_deg": float(need(cfg, "gate.one_flight.angle_threshold")),
            "scale_clamp": tuple(float(x) for x in need(cfg, "gate.one_flight.scale_clamp")),
            "min_inliers": int(need(cfg, "gate.one_flight.min_inliers")),
            "max_iterations": int(need(cfg, "gate.one_flight.max_iterations")),
            "scale_min_spread": float(need(cfg, "gate.one_flight.scale_min_spread"))}


def _fit(P, R, Ph, Rh, inl, clamp, min_spread):
    Rg = proj_so3(sum(Rh[k] @ R[k].T for k in np.nonzero(inl)[0]))
    X = P @ Rg.T
    xm, ym = X[inl].mean(axis=0), Ph[inl].mean(axis=0)
    den = float(((X[inl] - xm) ** 2).sum())
    if den < min_spread:
        s = 1.0
    else:
        s = float(np.clip(((X[inl] - xm) * (Ph[inl] - ym)).sum() / den, clamp[0], clamp[1]))
    t = ym - s * xm
    e = np.linalg.norm(s * X + t - Ph, axis=1)
    a = np.array([rotation_angle_deg(Rg @ R[k], Rh[k]) for k in range(len(P))])
    return Rg, s, t, e, a


def fit_one_flight(declared, estimate, params):
    """declared, estimate: lists of 4 x 4 wide-camera poses of the same frames (declared in the twin frame, estimate
    in the estimate's frame). Returns a dict: rotation, scale, translation (the map twin -> estimate frame),
    e_m, a_deg (per frame), inlier (per frame, bool), n_inliers, fits, converged, z3 (bool)."""
    n = len(declared)
    if n != len(estimate):
        raise ValueError("declared and estimate poses must pair up")
    out = {"rotation": np.eye(3), "scale": 1.0, "translation": np.zeros(3), "e_m": np.full(n, np.inf),
           "a_deg": np.full(n, np.inf), "inlier": np.zeros(n, bool), "n_inliers": 0, "fits": 0, "converged": False}
    if n == 0:
        out["z3"] = 0 < params["min_inliers"]
        return out
    P = np.array([np.asarray(T, float)[:3, 3] for T in declared])
    R = np.array([np.asarray(T, float)[:3, :3] for T in declared])
    Ph = np.array([np.asarray(T, float)[:3, 3] for T in estimate])
    Rh = np.array([np.asarray(T, float)[:3, :3] for T in estimate])
    inl = np.ones(n, bool)
    for it in range(params["max_iterations"]):
        Rg, s, t, e, a = _fit(P, R, Ph, Rh, inl, params["scale_clamp"], params["scale_min_spread"])
        new = (e <= params["position_threshold"]) & (a <= params["angle_threshold_deg"])
        out.update(rotation=Rg, scale=s, translation=t, e_m=e, a_deg=a, fits=it + 1)
        if np.array_equal(new, inl):
            out["converged"] = True
            inl = new
            break
        inl = new
        if not inl.any():
            break
    out["inlier"] = inl
    out["n_inliers"] = int(inl.sum())
    out["z3"] = out["n_inliers"] < params["min_inliers"]
    return out


def residual_summary(res):
    """A JSON-ready summary of a fit (finite numbers only)."""
    e = np.asarray(res["e_m"], float)
    a = np.asarray(res["a_deg"], float)
    fin = np.isfinite(e) & np.isfinite(a)
    return {"n_inliers": int(res["n_inliers"]), "z3": bool(res["z3"]), "scale": float(res["scale"]),
            "fits": int(res["fits"]), "converged": bool(res["converged"]),
            "e_m_max": float(e[fin].max()) if fin.any() else None,
            "a_deg_max": float(a[fin].max()) if fin.any() else None,
            "rotation_angle_deg": float(rotation_angle_deg(np.eye(3), res["rotation"]))}
