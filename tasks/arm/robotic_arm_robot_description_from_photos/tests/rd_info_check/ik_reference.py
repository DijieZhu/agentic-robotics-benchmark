"""ik_reference.py: a standalone Levenberg-Marquardt / damped-least-squares IK on MuJoCo forward kinematics
(mj_kinematics + mj_comPos + mj_jacSite): analytic Jacobians, position + approach-axis targets, optional full-orientation
(quaternion / roll) targets, joint-limit clamping, random restarts. The grader's controller in `downstream_reach`.

Stable API (announced here; consumers import `from arm import ik_reference` or vendor the file)
----------------------------------------------------------------------------------------------

    solve(model, data, target, q0=None, seed=0, max_iters=100, restarts=24,
          tol_pos=0.002, tol_rot_deg=1.0, polish=True, joints=ARM_JOINTS, site="tcp",
          rot_weight=0.1, collision_free=False) -> np.ndarray[5] | None

        `target`: {"pos": [3] m world, "approach": [3] unit (desired +z of the site),
                   "roll_rad"?: float (roll about the approach axis, see `target_rotation`),
                   "quat"?: [4] wxyz (full orientation; overrides `roll_rad`)}
        Returns the joint vector (joint order `joints`, inside `model.jnt_range`) that places the
        site within `tol_pos` / `tol_rot_deg` of the target, or None when no restart converges.
        Deterministic for a given (model, target, q0, seed): restart 0 starts from `q0` (clipped;
        the model's `home` keyframe or zeros when q0 is None), restart 1 from the home keyframe,
        restarts 2-5 from informed yaw candidates (target azimuth, both branches, mid-range and
        folded poses), the rest from uniform draws of `numpy.random.Generator(PCG64(seed))`.
        `data` is scratch: its qpos is overwritten (jaw and any other DoF are left as they are).
        With `polish=True` the solver keeps iterating after reaching the tolerances until the
        residual stops improving (typically 1e-9 m), so a solved target's residual is far inside
        the 0.5 mm / 0.25 deg full-credit band of the grader.
        `collision_free=True` (rung 3 preparation) rejects converged configurations with any
        active contact (`mj_forward`, `data.ncon > 0`) and keeps restarting.

    solve_batch(model, targets, q0s=None, seed=0, **kw) -> list[np.ndarray | None]
        One `MjData`, one solve per target; `seed + i` per target (or `seeds[i]`).

    residual(model, data, q, target, joints=ARM_JOINTS, site="tcp") -> (pos_err_m, rot_err_deg)
        The grader's own measure: position error and approach-axis angle (or geodesic angle when
        the target carries `quat`).

    target_rotation(target) -> np.ndarray[3,3] | None
        The full target rotation when `quat` (or `roll_rad`) is given, else None. Roll convention:
        roll_rad = 0 puts the site's +x (jaw closing direction) along the projection of world +x
        onto the plane normal to `approach` (world -y when approach is +-x), then rotates about
        `approach` by roll_rad (right-hand).

    reach_bounds(model, joints=ARM_JOINTS, site="tcp") -> dict
        Certified kinematic bounds read from the model: `shoulder_point` (world position of the
        second joint's anchor at q = 0), `r_max` (sum of the segment lengths from that point to the
        site: an upper bound of the reach by the triangle inequality), `z_min` (lower bound of the
        site height from the shoulder-pitch range and the remaining chain length). A target with
        |pos - shoulder_point| > r_max or pos_z < z_min is certifiably unreachable.

    jacobian_condition(model, data, q, joints=ARM_JOINTS, site="tcp") -> float
        sigma_min / sigma_max of the 3 x n position Jacobian at q (near-singular when small).

Everything is single-threaded numpy + mujoco; no file I/O, no randomness outside `seed`.
"""

import math

import numpy as np

ARM_JOINTS = ["base_yaw", "shoulder_pitch", "elbow_pitch", "wrist_pitch", "wrist_roll"]
TCP_SITE = "tcp"
IK_REFERENCE_VERSION = "arm-chain-ik-reference/1.0.0"

__all__ = ["solve", "solve_batch", "residual", "target_rotation", "reach_bounds", "jacobian_condition",
           "ARM_JOINTS", "TCP_SITE", "IK_REFERENCE_VERSION", "ChainIndex"]


def _mujoco():
    import mujoco
    return mujoco


# ----------------------------------------------------------------------------- model indexing
class ChainIndex(object):
    """Joint / dof / qpos indices, limits and the site id of the IK chain in a model (cached per call)."""

    def __init__(self, model, joints=ARM_JOINTS, site=TCP_SITE):
        mujoco = _mujoco()
        self.joints = list(joints)
        self.jid = []
        for j in self.joints:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if jid < 0:
                raise ValueError("model has no joint {!r}".format(j))
            self.jid.append(int(jid))
        self.qadr = np.array([int(model.jnt_qposadr[j]) for j in self.jid])
        self.dadr = np.array([int(model.jnt_dofadr[j]) for j in self.jid])
        self.lo = np.array([float(model.jnt_range[j][0]) for j in self.jid])
        self.hi = np.array([float(model.jnt_range[j][1]) for j in self.jid])
        limited = np.array([bool(model.jnt_limited[j]) for j in self.jid])
        self.lo = np.where(limited, self.lo, -np.pi)
        self.hi = np.where(limited, self.hi, np.pi)
        self.sid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site))
        if self.sid < 0:
            raise ValueError("model has no site {!r}".format(site))
        self.n = len(self.joints)
        self.home = None
        if model.nkey > 0:
            kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
            kid = 0 if kid < 0 else int(kid)
            self.home = np.asarray(model.key_qpos[kid], float)[self.qadr].copy()
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def clip(self, q):
        return np.minimum(np.maximum(np.asarray(q, float), self.lo), self.hi)

    def inside(self, q, slack=1e-6):
        q = np.asarray(q, float)
        return bool(np.all(q >= self.lo - slack) and np.all(q <= self.hi + slack))

    def fk(self, model, data, q):
        mujoco = _mujoco()
        data.qpos[self.qadr] = q
        mujoco.mj_kinematics(model, data)
        return data.site_xpos[self.sid], data.site_xmat[self.sid].reshape(3, 3)

    def fk_jac(self, model, data, q):
        """pos, R, jacp (3 x n), jacr (3 x n) of the site at q."""
        mujoco = _mujoco()
        data.qpos[self.qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self.sid)
        return (data.site_xpos[self.sid], data.site_xmat[self.sid].reshape(3, 3),
                self._jacp[:, self.dadr], self._jacr[:, self.dadr])


# ----------------------------------------------------------------------------- rotations
def _unit(v):
    v = np.asarray(v, float).ravel()
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n <= 0:
        raise ValueError("zero / non-finite vector")
    return v / n


def _matrix_from_quat(q):
    w, x, y, z = _unit(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rotvec(R):
    """Rotation vector (axis * angle) of a rotation matrix (angle in [0, pi])."""
    c = max(-1.0, min(1.0, (np.trace(R) - 1.0) / 2.0))
    ang = math.acos(c)
    if ang < 1e-12:
        return np.zeros(3)
    if ang > math.pi - 1e-6:
        # axis from the symmetric part
        A = (R + np.eye(3)) / 2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k]
        axis = axis / max(np.linalg.norm(axis), 1e-12)
        return axis * ang
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (ang / (2.0 * math.sin(ang)))


def _rot_from_rotvec(r):
    ang = np.linalg.norm(r)
    if ang < 1e-12:
        return np.eye(3)
    k = r / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * (K @ K)


def _axis_rotvec(a, b):
    """Rotation vector of the minimal rotation taking unit vector a onto unit vector b."""
    c = float(np.clip(a @ b, -1.0, 1.0))
    ax = np.cross(a, b)
    s = np.linalg.norm(ax)
    if s < 1e-12:
        if c > 0:
            return np.zeros(3)
        # antipodal: any axis normal to a
        h = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        n = np.cross(a, h)
        return n / np.linalg.norm(n) * math.pi
    return ax / s * math.atan2(s, c)


def _angle_deg(a, b):
    return math.degrees(math.acos(max(-1.0, min(1.0, float(_unit(a) @ _unit(b))))))


def target_rotation(target):
    """Full target rotation matrix from `quat` (wxyz) or `approach` + `roll_rad`; None if neither."""
    if target.get("quat") is not None:
        return _matrix_from_quat(target["quat"])
    if target.get("roll_rad") is None:
        return None
    a = _unit(target["approach"])
    hint = np.array([1.0, 0.0, 0.0])
    if abs(float(a @ hint)) > 0.999:
        hint = np.array([0.0, -1.0, 0.0])
    x = hint - (hint @ a) * a
    x = x / np.linalg.norm(x)
    y = np.cross(a, x)
    R0 = np.column_stack([x, y, a])
    return _rot_from_rotvec(a * float(target["roll_rad"])) @ R0


# ----------------------------------------------------------------------------- residual
def residual(model, data, q, target, joints=ARM_JOINTS, site=TCP_SITE, index=None):
    """(position error [m], orientation error [deg]) of configuration q against the target.

    The orientation error is the angle between the site's +z and `approach`, or the geodesic
    angle to the full rotation when the target carries `quat`.
    """
    idx = index or ChainIndex(model, joints, site)
    p, R = idx.fk(model, data, np.asarray(q, float))
    perr = float(np.linalg.norm(p - np.asarray(target["pos"], float)))
    Rt = _matrix_from_quat(target["quat"]) if target.get("quat") is not None else None
    if Rt is not None:
        c = max(-1.0, min(1.0, (np.trace(R.T @ Rt) - 1.0) / 2.0))
        return perr, math.degrees(math.acos(c))
    return perr, _angle_deg(R[:, 2], target["approach"])


def jacobian_condition(model, data, q, joints=ARM_JOINTS, site=TCP_SITE, index=None):
    """sigma_min / sigma_max of the site's position Jacobian restricted to the chain's dofs."""
    idx = index or ChainIndex(model, joints, site)
    _, _, jacp, _ = idx.fk_jac(model, data, np.asarray(q, float))
    sv = np.linalg.svd(jacp, compute_uv=False)
    return float(sv[-1] / max(sv[0], 1e-12))


# ----------------------------------------------------------------------------- solver
def _error_and_jac(idx, model, data, q, p_t, a_t, R_t, rot_weight, want_jac=True):
    """e (6,) and J (6 x n) with e(q + dq) ~ e(q) - J dq. Rows 0-2 position, 3-5 rot_weight x rotation."""
    if want_jac:
        p, R, jacp, jacr = idx.fk_jac(model, data, q)
    else:
        p, R = idx.fk(model, data, q)
        jacp = jacr = None
    e = np.empty(6)
    e[:3] = p_t - p
    a = R[:, 2]
    if R_t is not None:
        r = _rotvec(R_t @ R.T)          # rotation taking R onto R_t, world frame
        P = None
    else:
        r = _axis_rotvec(a, a_t)
        P = np.eye(3) - np.outer(a, a)  # roll about the current approach axis is free
    e[3:] = rot_weight * r
    if not want_jac:
        return e, None, p, R
    J = np.empty((6, idx.n))
    J[:3] = jacp
    J[3:] = rot_weight * (jacr if P is None else P @ jacr)
    return e, J, p, R


def _converged(e, rot_weight, tol_pos, tol_rot_rad):
    return np.linalg.norm(e[:3]) <= tol_pos and np.linalg.norm(e[3:]) / rot_weight <= tol_rot_rad


def _informed_starts(idx, model, data, p_t):
    """Yaw candidates for a yaw-then-pitch chain: put a nominal pose's tcp azimuth onto the target
    azimuth (both branches, +-pi) for two nominal poses (mid-range; folded = elbow at its bound
    nearest the fold, wrist mid). Pure FK, deterministic, harmless for other chains."""
    mid = idx.clip(0.5 * (idx.lo + idx.hi))
    poses = [mid.copy()]
    if idx.n >= 3:
        folded = mid.copy()
        folded[2] = idx.lo[2] if abs(idx.lo[2]) > abs(idx.hi[2]) else idx.hi[2]
        poses.append(folded)
    out = []
    for q in poses:
        q[0] = 0.0 if idx.lo[0] <= 0.0 <= idx.hi[0] else mid[0]
        p, _ = idx.fk(model, data, q)
        base = np.asarray(data.xanchor[idx.jid[0]], float)
        v = p - base
        w = p_t - base
        if math.hypot(v[0], v[1]) < 1e-6 or math.hypot(w[0], w[1]) < 1e-6:
            out.append(idx.clip(q))
            continue
        daz = math.atan2(w[1], w[0]) - math.atan2(v[1], v[0])
        for branch in (0.0, math.pi):
            yaw = q[0] + daz + branch
            yaw = (yaw + math.pi) % (2 * math.pi) - math.pi
            qs = q.copy()
            qs[0] = yaw
            out.append(idx.clip(qs))
    return out


def solve(model, data, target, q0=None, seed=0, max_iters=100, restarts=24, tol_pos=0.002, tol_rot_deg=1.0,
          polish=True, joints=ARM_JOINTS, site=TCP_SITE, rot_weight=0.1, collision_free=False,
          index=None, lam0=1.0, lam_min=1e-4, lam_max=1e4, polish_iters=25, tight_pos=1e-7, tight_rot_deg=1e-5):
    """LM-DLS inverse kinematics; see the module docstring for the contract. Returns q (n,) or None.

    Restart schedule: q0 (clipped; home / zeros when None), the home keyframe, two informed yaw
    starts (target azimuth, both branches), then uniform draws from PCG64(seed). A restart that
    converges inside the tolerances is polished; when it reaches `tight_pos` / `tight_rot_deg` it
    is returned at once, otherwise the tightest converged configuration over all restarts is.
    """
    idx = index or ChainIndex(model, joints, site)
    p_t = np.asarray(target["pos"], float).ravel()
    if p_t.shape != (3,) or not np.all(np.isfinite(p_t)):
        raise ValueError("target.pos must be a finite 3-vector")
    R_t = target_rotation(target)
    a_t = _unit(target["approach"]) if target.get("approach") is not None else (R_t[:, 2] if R_t is not None else None)
    if a_t is None:
        raise ValueError("target needs `approach` or `quat`")
    tol_rot = math.radians(tol_rot_deg)
    tight_rot = math.radians(tight_rot_deg)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    starts = []
    if q0 is not None:
        starts.append(idx.clip(np.asarray(q0, float).ravel()))
    elif idx.home is not None:
        starts.append(idx.clip(idx.home))
    else:
        starts.append(idx.clip(np.zeros(idx.n)))
    if idx.home is not None:
        starts.append(idx.clip(idx.home))
    starts.extend(_informed_starts(idx, model, data, p_t))
    mujoco = _mujoco() if collision_free else None
    best = None  # (residual norm, q) over converged restarts
    for r in range(int(restarts)):
        q = starts[r] if r < len(starts) else idx.lo + rng.random(idx.n) * (idx.hi - idx.lo)
        lam = float(lam0)
        e, J, _, _ = _error_and_jac(idx, model, data, q, p_t, a_t, R_t, rot_weight)
        ok = _converged(e, rot_weight, tol_pos, tol_rot)
        its = 0
        polishing = 0
        while its < int(max_iters) or (ok and polishing < polish_iters):
            its += 1
            if ok:
                if not polish or _converged(e, rot_weight, tight_pos, tight_rot):
                    break
                polishing += 1
            H = J.T @ J
            g = J.T @ e
            try:
                dq = np.linalg.solve(H + (lam * lam) * np.eye(idx.n), g)
            except np.linalg.LinAlgError:
                lam = min(lam * 3.0, lam_max)
                continue
            q_new = idx.clip(q + dq)
            e_new, J_new, _, _ = _error_and_jac(idx, model, data, q_new, p_t, a_t, R_t, rot_weight)
            if np.linalg.norm(e_new) < np.linalg.norm(e):
                q, e, J = q_new, e_new, J_new
                lam = max(lam * 0.5, lam_min)
                ok = _converged(e, rot_weight, tol_pos, tol_rot)
            else:
                lam = min(lam * 3.0, lam_max)
                if lam >= lam_max:
                    break
        if not ok:
            continue
        if collision_free:
            data.qpos[idx.qadr] = q
            mujoco.mj_forward(model, data)
            if data.ncon > 0:
                continue
        if _converged(e, rot_weight, tight_pos, tight_rot):
            return np.asarray(q, float)
        n = float(np.linalg.norm(e))
        if best is None or n < best[0]:
            best = (n, np.asarray(q, float).copy())
    return None if best is None else best[1]


def solve_batch(model, targets, q0s=None, seed=0, seeds=None, index=None, **kw):
    """Solve every target with one scratch MjData; per-target seed = seeds[i] or seed + i."""
    mujoco = _mujoco()
    data = mujoco.MjData(model)
    idx = index or ChainIndex(model, kw.get("joints", ARM_JOINTS), kw.get("site", TCP_SITE))
    out = []
    for i, t in enumerate(targets):
        q0 = None if q0s is None else q0s[i]
        s = seeds[i] if seeds is not None else int(seed) + i
        out.append(solve(model, data, t, q0=q0, seed=s, index=idx, **kw))
    return out


# ----------------------------------------------------------------------------- certified bounds
def reach_bounds(model, joints=ARM_JOINTS, site=TCP_SITE, margin_m=0.0):
    """Certified kinematic bounds of the chain, read from the model at q = 0 (see docstring)."""
    mujoco = _mujoco()
    idx = ChainIndex(model, joints, site)
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    if idx.home is not None:
        pass  # bounds are configuration independent; q = 0 is only used to read anchor positions
    mujoco.mj_kinematics(model, data)
    anchors = [np.array(data.xanchor[idx.jid[k]], float) for k in range(idx.n)]
    site_p = np.array(data.site_xpos[idx.sid], float)
    # segment lengths along the chain from each joint anchor to the next, then to the site
    pts = anchors + [site_p]
    seg = [float(np.linalg.norm(pts[k + 1] - pts[k])) for k in range(len(pts) - 1)]
    shoulder = anchors[1] if idx.n > 1 else anchors[0]
    r_max = float(sum(seg[1:])) + margin_m
    # z lower bound: the second joint (shoulder pitch, horizontal axis) tilts the upper arm by at
    # most max(|range|) from its q = 0 direction; everything after the elbow can hang straight down.
    up_dir = (pts[2] - pts[1]) / max(np.linalg.norm(pts[2] - pts[1]), 1e-12) if idx.n > 2 else np.array([0, 0, 1.0])
    max_tilt = max(abs(idx.lo[1]), abs(idx.hi[1])) if idx.n > 1 else math.pi
    z_elbow_min = shoulder[2] + seg[1] * (math.cos(max_tilt) if max_tilt < math.pi / 2 else -1.0) if idx.n > 2 else shoulder[2]
    if up_dir[2] < 0.999:  # upper arm not vertical at q = 0: fall back to the conservative bound
        z_elbow_min = shoulder[2] - seg[1]
    z_min = float(z_elbow_min - sum(seg[2:])) - margin_m
    return {"shoulder_point": [float(v) for v in shoulder], "r_max": r_max, "z_min": z_min,
            "segment_lengths": seg, "site_at_zero": [float(v) for v in site_p],
            "anchors_at_zero": [[float(v) for v in a] for a in anchors],
            "rule": "unreachable if |pos - shoulder_point| > r_max or pos[2] < z_min (triangle inequality; joint 2 range)"}
