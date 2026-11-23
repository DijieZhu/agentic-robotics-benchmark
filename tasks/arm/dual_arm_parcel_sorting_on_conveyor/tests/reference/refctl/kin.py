"""Kinematics for the reference controller: forward kinematics and a damped
least-squares inverse kinematics of one 7-DOF arm (TCP site), on a private
MjData copy of the cell's model (never touches the simulation state).

Tool frame (TCP site): z = approach axis (out of the gripper), x = finger
opening direction, y = z cross x.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from parcelcell.robot import ARM_JOINTS, JOINT_NAMES

SIDES = ("Left", "Right")


def rot_axis_angle(axis, ang) -> np.ndarray:
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(ang), math.sin(ang)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def frame_from_zx(z, x_hint) -> np.ndarray:
    """Rotation matrix with columns (x, y, z): z given, x as close as possible
    to x_hint."""
    z = np.asarray(z, float)
    z = z / np.linalg.norm(z)
    x = np.asarray(x_hint, float)
    x = x - z * (x @ z)
    n = np.linalg.norm(x)
    if n < 1e-6:
        x = np.array([0.0, 0.0, 1.0]) - z * z[2]
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0.0, 0.0]) - z * z[0]
        n = np.linalg.norm(x)
    x = x / n
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def rot_err(R_cur, R_tgt) -> np.ndarray:
    """World-frame rotation vector taking R_cur to R_tgt."""
    Re = R_tgt @ R_cur.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Re.reshape(-1))
    if q[0] < 0:
        q = -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.zeros(3)
    ang = 2 * math.atan2(s, q[0])
    return v / s * ang


class ArmKin:
    """FK / IK of one arm on a private MjData."""

    def __init__(self, model: mujoco.MjModel, side: str):
        self.m = model
        self.d = mujoco.MjData(model)
        self.side = side
        names = ARM_JOINTS[side]
        self.jid = np.array([model.joint(n).id for n in names])
        self.qadr = model.jnt_qposadr[self.jid].copy()
        self.dadr = model.jnt_dofadr[self.jid].copy()
        self.lo = model.jnt_range[self.jid, 0].copy()
        self.hi = model.jnt_range[self.jid, 1].copy()
        self.site = model.site(f"{side}_tcp").id
        self.idx14 = np.array([JOINT_NAMES.index(n) for n in names])
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return (self.d.site_xpos[self.site].copy(),
                self.d.site_xmat[self.site].reshape(3, 3).copy())

    def jac(self, q):
        self.fk(q)
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
        return self._jp[:, self.dadr].copy(), self._jr[:, self.dadr].copy()

    def ik(self, pos, R=None, axis=None, q0=None, q_rest=None,
           iters=150, pos_tol=2e-3, rot_tol=0.02, w_rot=0.35,
           damping=0.03, null_gain=0.05, step_max=0.25, margin=0.0,
           limit_gain=0.0):
        """Damped least squares. Orientation: R (full), or axis (tool z only),
        or neither (position only). Returns (q, ok, pos_err, rot_err)."""
        pos = np.asarray(pos, float)
        q = np.array(q0 if q0 is not None else (self.lo + self.hi) / 2, float)
        lo, hi = self.lo + margin, self.hi - margin
        q = np.clip(q, lo, hi)
        qr = None if q_rest is None else np.asarray(q_rest, float)
        if axis is not None:
            axis = np.asarray(axis, float)
            axis = axis / np.linalg.norm(axis)
        ep = er = np.inf
        for it in range(iters):
            p, Rc = self.fk(q)
            e_p = pos - p
            if R is not None:
                e_r = rot_err(Rc, R)
            elif axis is not None:
                zc = Rc[:, 2]
                c = np.cross(zc, axis)
                s = np.linalg.norm(c)
                ang = math.atan2(s, float(zc @ axis))
                e_r = c / s * ang if s > 1e-12 else (
                    np.zeros(3) if zc @ axis > 0 else np.array([1.0, 0, 0]) * math.pi)
            else:
                e_r = np.zeros(3)
            ep, er = float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
            if ep < pos_tol and er < rot_tol:
                return q, True, ep, er
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacSite(self.m, self.d, self._jp, self._jr, self.site)
            Jp = self._jp[:, self.dadr]
            Jr = self._jr[:, self.dadr]
            if R is None and axis is None:
                J, e = Jp, e_p
            else:
                if axis is not None and R is None:
                    # remove the free rotation about the tool axis
                    zc = Rc[:, 2]
                    P = np.eye(3) - np.outer(zc, zc)
                    Jr = P @ Jr
                J = np.vstack([Jp, w_rot * Jr])
                e = np.concatenate([e_p, w_rot * e_r])
            JT = J.T
            A = J @ JT + damping ** 2 * np.eye(J.shape[0])
            dq = JT @ np.linalg.solve(A, e)
            if qr is not None and null_gain > 0:
                g = null_gain * max(0.0, 1.0 - it / (0.7 * iters))
                dn = g * (qr - q)
                dq = dq + dn - JT @ np.linalg.solve(A, J @ dn)
            if limit_gain > 0:
                # push away from joint limits inside the null space
                mid = (self.lo + self.hi) / 2
                span = (self.hi - self.lo) / 2
                xn = (q - mid) / span
                dn = -limit_gain * (xn ** 7) / span
                dq = dq + dn - JT @ np.linalg.solve(A, J @ dn)
            n = np.max(np.abs(dq))
            if n > step_max:
                dq *= step_max / n
            q = np.clip(q + dq, lo, hi)
        return q, False, ep, er


class SeedLib:
    """A library of random arm configurations with their TCP poses; the
    nearest entries to a target pose seed the IK (robust global IK)."""

    def __init__(self, kin: ArmKin, n=6000, seed=0, margin=0.05):
        rng = np.random.default_rng(seed)
        self.kin = kin
        lo, hi = kin.lo + margin, kin.hi - margin
        self.Q = rng.uniform(lo, hi, size=(n, 7))
        P = np.zeros((n, 3))
        Z = np.zeros((n, 3))
        X = np.zeros((n, 3))
        for i, q in enumerate(self.Q):
            p, R = kin.fk(q)
            P[i], Z[i], X[i] = p, R[:, 2], R[:, 0]
        self.P, self.Z, self.X = P, Z, X

    def nearest(self, pos, axis=None, xdir=None, k=6, w_ang=0.15):
        d = np.sum((self.P - np.asarray(pos)) ** 2, axis=1)
        if axis is not None:
            d = d + w_ang * (1.0 - self.Z @ np.asarray(axis)) ** 1
        if xdir is not None:
            d = d + 0.5 * w_ang * (1.0 - np.abs(self.X @ np.asarray(xdir)))
        idx = np.argsort(d)[:k]
        return self.Q[idx]


def ik_global(kin: ArmKin, lib: SeedLib, pos, R=None, axis=None, k=8,
              q_rest=None, extra_seeds=(), margin=0.02, iters=150,
              cost=None):
    """IK from the library's nearest seeds (+ extra seeds); returns the
    feasible solution with the lowest cost (default: distance to q_rest, or
    joint-limit closeness)."""
    ax = axis if axis is not None else (None if R is None else R[:, 2])
    xd = None if R is None else R[:, 0]
    seeds = list(extra_seeds) + list(lib.nearest(pos, ax, xd, k=k))
    best, bc = None, np.inf
    mid = (kin.lo + kin.hi) / 2
    span = (kin.hi - kin.lo) / 2
    for s in seeds:
        q, ok, ep, er = kin.ik(pos, R=R, axis=axis, q0=s, q_rest=q_rest,
                               iters=iters, margin=margin)
        if not ok:
            continue
        if cost is not None:
            c = cost(q)
        else:
            c = float(np.sum(((q - mid) / span) ** 6))
            if q_rest is not None:
                c += 0.3 * float(np.sum((q - q_rest) ** 2))
        if c < bc:
            best, bc = q, c
    return best
