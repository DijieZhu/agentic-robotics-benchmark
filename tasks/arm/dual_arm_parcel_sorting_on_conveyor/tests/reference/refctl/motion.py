"""Motion layer of the reference controller.

* ``Dyn``: inverse-dynamics feedforward through the position servos. The
  robot's drives are position servos (tau = kp (u - q) - kv qd, minus joint
  damping and friction loss); to make the joints follow a planned
  (q, qd, qdd) the command is u = q + (M qdd + c(q, qd) + g(q) + (kv + d) qd
  + J^T F_ext) / kp, with F_ext an intended contact force. Everything is
  computed on a private MjData of the robot model (no simulator state is
  read for it).
* Trajectory helpers: minimum-jerk joint moves with a speed limit, Cartesian
  paths solved by continuous IK.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from parcelcell.robot import JOINT_NAMES

from .kin import ArmKin, rot_axis_angle, rot_err

DT = 0.02


def min_jerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


def min_jerk_d(s):
    s = np.clip(s, 0.0, 1.0)
    return 30 * s * s * (1 - s) ** 2


def min_jerk_dd(s):
    s = np.clip(s, 0.0, 1.0)
    return 60 * s * (1 - s) * (1 - 2 * s)


class Dyn:
    """Feedforward command for the 14 arm joints."""

    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        self.jid = np.array([model.joint(n).id for n in JOINT_NAMES])
        self.qadr = model.jnt_qposadr[self.jid]
        self.dadr = model.jnt_dofadr[self.jid]
        aid = np.array([model.actuator(n).id for n in JOINT_NAMES])
        self.kp = model.actuator_gainprm[aid, 0].copy()
        self.kv = -model.actuator_biasprm[aid, 2].copy()
        self.damp = model.dof_damping[self.dadr].copy()
        self.lo = model.actuator_ctrlrange[aid, 0].copy()
        self.hi = model.actuator_ctrlrange[aid, 1].copy()
        self.tlim = model.actuator_forcerange[aid, 1].copy()
        self._res = np.zeros(model.nv)
        self._jp = np.zeros((3, model.nv))
        self._jr = np.zeros((3, model.nv))
        # zero everything that is not the arms (parcels parked far away in
        # the private copy so they never matter)
        self.d.qpos[:] = model.qpos0

    def tau_ff(self, q, qd=None, qdd=None):
        d = self.d
        d.qpos[self.qadr] = q
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        if qd is not None:
            d.qvel[self.dadr] = qd
        if qdd is not None:
            d.qacc[self.dadr] = qdd
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        mujoco.mj_comVel(self.m, d)
        mujoco.mj_rne(self.m, d, 1, self._res)
        tau = self._res[self.dadr].copy()
        if qd is not None:
            tau += self.damp * qd
        return tau

    def jac_point(self, q, body_name: str, point_world=None, site=None):
        """Translational Jacobian (3 x 14 in JOINT_NAMES order) of a point
        fixed to a body (or a site)."""
        d = self.d
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        if site is not None:
            sid = self.m.site(site).id
            mujoco.mj_jacSite(self.m, d, self._jp, self._jr, sid)
        else:
            bid = self.m.body(body_name).id
            mujoco.mj_jac(self.m, d, self._jp, self._jr,
                          np.asarray(point_world, float), bid)
        return self._jp[:, self.dadr].copy()

    def command(self, q, qd=None, qdd=None, tau_extra=None):
        """Position command for the servos so the joints follow (q, qd, qdd)
        while applying tau_extra (e.g. J^T F for a contact force)."""
        tau = self.tau_ff(q, qd, qdd)
        if tau_extra is not None:
            tau = tau + tau_extra
        tau = np.clip(tau, -self.tlim, self.tlim)
        u = q + tau / self.kp
        if qd is not None:
            u = u + self.kv * qd / self.kp
        return np.clip(u, self.lo, self.hi)


# ----------------------------------------------------------------------
def joint_move(q0, q1, vmax=2.0, tmin=0.3, amax=8.0):
    """Minimum-jerk move between two 14-vectors; returns arrays (q, qd, qdd)
    per tick (the first sample is one tick after q0, the last equals q1)."""
    q0 = np.asarray(q0, float)
    q1 = np.asarray(q1, float)
    dq = q1 - q0
    D = float(np.max(np.abs(dq)))
    T = max(tmin, 1.875 * D / vmax, math.sqrt(5.77 * D / amax))
    n = max(1, int(math.ceil(T / DT)))
    T = n * DT
    s = np.arange(1, n + 1) / n
    q = q0 + np.outer(min_jerk(s), dq)
    qd = np.outer(min_jerk_d(s), dq) / T
    qdd = np.outer(min_jerk_dd(s), dq) / (T * T)
    return q, qd, qdd


def finite_diff(qs, q_prev=None):
    """Velocities / accelerations of a sampled path (central differences)."""
    qs = np.asarray(qs, float)
    if q_prev is not None:
        full = np.vstack([q_prev[None], qs])
    else:
        full = np.vstack([qs[:1], qs])
    qd = np.gradient(full, DT, axis=0)[1:]
    qdd = np.gradient(np.gradient(full, DT, axis=0), DT, axis=0)[1:]
    return qd, qdd


def interp_pose(p0, R0, p1, R1, s):
    """Pose interpolation: linear position, geodesic rotation."""
    p = p0 + (p1 - p0) * s
    v = rot_err(R0, R1)
    ang = float(np.linalg.norm(v))
    if ang < 1e-9:
        return p, R0.copy()
    return p, rot_axis_angle(v / ang, ang * s) @ R0


class CartPath:
    """Continuous IK along a sampled Cartesian path for one arm."""

    def __init__(self, kin: ArmKin):
        self.kin = kin

    def solve(self, poses, q_start, q_rest=None, iters=60, tol=3e-3,
              rot_tol=0.03, max_jump=0.35):
        """poses: list of (pos, R or None, axis or None). Returns (qs (n,7),
        ok, worst position error)."""
        q = np.asarray(q_start, float).copy()
        out = []
        worst = 0.0
        ok_all = True
        for (p, R, ax) in poses:
            qn, ok, ep, er = self.kin.ik(p, R=R, axis=ax, q0=q, q_rest=q_rest,
                                         iters=iters, pos_tol=tol,
                                         rot_tol=rot_tol, null_gain=0.0)
            if not ok or np.max(np.abs(qn - q)) > max_jump:
                ok_all = False
            worst = max(worst, ep)
            q = qn
            out.append(q.copy())
        return np.array(out), ok_all, worst
