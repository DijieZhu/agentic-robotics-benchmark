"""Path planning helpers: continuous IK along Cartesian pose paths of one or
both arms, time scaling to a joint-speed limit with a minimum-jerk profile.
"""

from __future__ import annotations

import math

import numpy as np

from .kin import ArmKin, rot_axis_angle, rot_err
from .motion import DT, min_jerk

SIDES = ("Left", "Right")
SL = {"Left": slice(0, 7), "Right": slice(7, 14)}
SG = {"Left": 1.0, "Right": -1.0}

# natural posture of each arm for palm work (two-palm squeeze, hands
# pointing at each other along y): arms abducted, elbows out
Q_PALM_NOM = {
    "Left": np.array([-0.72, 0.78, 0.68, 1.08, -1.56, -1.46, -1.0]),
    "Right": np.array([0.72, -0.78, -0.68, 1.08, 1.56, 1.46, 1.0]),
}


class PlanFail(Exception):
    pass


def lerp_pose(p0, R0, p1, R1, s):
    p = p0 + (p1 - p0) * s
    v = rot_err(R0, R1)
    ang = float(np.linalg.norm(v))
    if ang < 1e-9:
        return p, R0.copy()
    return p, rot_axis_angle(v / ang, ang * s) @ R0


def solve_path(kin: ArmKin, poses, q_start, iters=60, tol=4e-3,
               rot_tol=0.04, max_jump=0.3, margin=0.005, tol_mid=9e-3,
               rot_mid=0.09):
    """Continuous IK along poses [(pos, R)]; raises PlanFail."""
    q = np.asarray(q_start, float).copy()
    out = []
    for i, (p, R) in enumerate(poses):
        qn, ok, ep, er = kin.ik(p, R=R, q0=q, iters=iters, pos_tol=tol,
                                rot_tol=rot_tol, null_gain=0.0,
                                margin=margin)
        if not ok:
            # one retry with more iterations
            qn, ok, ep, er = kin.ik(p, R=R, q0=q, iters=250, pos_tol=tol,
                                    rot_tol=rot_tol, null_gain=0.0,
                                    margin=margin)
        last = i == len(poses) - 1
        if not ok and ((not last and ep < tol_mid and er < rot_mid) or
                       (last and ep < 1.6 * tol and er < 1.6 * rot_tol)):
            ok = True
        if not ok:
            raise PlanFail(f"{kin.side}: IK fail at waypoint {i}/{len(poses)}"
                           f" (pos err {ep:.4f}, rot err {er:.3f})")
        if np.max(np.abs(qn - q)) > max_jump:
            raise PlanFail(f"{kin.side}: joint jump {np.max(np.abs(qn - q)):.2f}"
                           f" at waypoint {i}")
        q = qn
        out.append(q.copy())
    return np.array(out)


def time_scale(Q, vmax=3.0, tmin=0.2, amax=12.0):
    """Q: (n, d) joint waypoints of a path (the first row is the start).
    Returns per-tick samples (m, d) along the path with a minimum-jerk
    timing whose peak joint speed is <= vmax."""
    Q = np.asarray(Q, float)
    if len(Q) < 2:
        return Q[-1:].copy()
    seg = np.abs(np.diff(Q, axis=0))           # (n-1, d)
    s_knots = np.concatenate([[0.0], np.cumsum(seg.max(axis=1))])
    L = s_knots[-1]
    # per-joint path length matters for the speed: peak speed of joint j
    # = 1.875 * (dq_j/ds)_max * L / T; bound by the max ratio
    if L < 1e-9:
        n = max(1, int(round(tmin / DT)))
        return np.repeat(Q[-1:], n, axis=0)
    T = max(tmin, 1.875 * L / vmax, math.sqrt(5.77 * L / amax))
    n = max(1, int(math.ceil(T / DT)))
    s = min_jerk(np.arange(1, n + 1) / n) * L
    out = np.empty((n, Q.shape[1]))
    for j in range(Q.shape[1]):
        out[:, j] = np.interp(s, s_knots, Q[:, j])
    return out


def two_arm_path(kins, pose_fns, q14_start, n=40, vmax=3.0, tmin=0.2):
    """pose_fns: {side: f(u)->(pos, R)} for u in [0, 1] (sides absent keep
    their joints). Both arms share one time scaling. Returns (m, 14)."""
    Q = np.repeat(np.asarray(q14_start, float)[None], n + 1, axis=0)
    for s, f in pose_fns.items():
        poses = [f(u) for u in np.linspace(0, 1, n + 1)[1:]]
        Qs = solve_path(kins[s], poses, q14_start[SL[s]])
        Q[1:, SL[s]] = Qs
    return time_scale(Q, vmax=vmax, tmin=tmin)


def palm_R(side: str, psi: float, yaw: float = 0.0, tilt: float = 0.0):
    """Palm frame: tool z along -SG*y (toward the middle), rolled by psi
    about the tool axis, the whole frame yawed by `yaw` about world z and
    tilted down by `tilt` (about the frame's x axis at psi = 0)."""
    z = np.array([0.0, -SG[side], 0.0])
    x = np.array([1.0, 0.0, 0.0])
    if tilt:
        Rt = rot_axis_angle((1, 0, 0), SG[side] * tilt)
        z, x = Rt @ z, Rt @ x
    x = rot_axis_angle(z, psi) @ x
    R = np.column_stack([x, np.cross(z, x), z])
    if yaw:
        R = rot_axis_angle((0, 0, 1), yaw) @ R
    return R
