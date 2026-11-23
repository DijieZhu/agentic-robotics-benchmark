"""The station controller's tool-pose command level.

Beside the joint level (14 joint position targets), a program may command, per arm, a target pose of the arm's
tool centre point (TCP) in the world frame: a position and a unit quaternion (w, x, y, z). The controller then
computes that arm's joint targets itself, every tick:

* Inverse kinematics by damped least squares (Levenberg-Marquardt with a residual-dependent damping, which stays
  well behaved at the edge of the workspace), started from the arm's present solution (its commanded posture), a
  few iterations per tick, each accepted only when it brings the tool closer to the target (backtracking). The step
  is the smallest joint change that reduces the pose error, so the posture stays near the present one and the
  arm's redundant joint does not drift.
* Joint ranges: no solution leaves a joint's range less a small margin; a joint slows down as it nears that bound
  (per tick it covers at most a share of the distance left) and, once there, stays for the rest of the tick's solve.
* Speed: the solution moves by at most the command speed limit per tick (the whole joint step is scaled, so the
  path's direction is kept), so the cell's own limiter never has to cut a tool-pose command.
* Out of reach: the orientation weighs heavily in the solve (``rot_length_m``: one radian of orientation error
  weighs like that many metres of position error), so an unreachable target ends short in position with the
  tool's orientation kept close to the target's (it is not turned away to gain position); the solve reports so
  (``out_of_reach`` and the remaining error); it does not jump, it does not stop the robot. The flag latches: once
  set it stays set until the target changes or the solution's position error falls clearly below the error at
  which it was set (``unlatch_m`` / ``unlatch_rel``), or the target is reached.
* Static gravity correction: the joint targets sent to the servos are the solution plus the servos' static sag
  under the arm's own weight (gravity torque / servo stiffness), so that the measured tool pose, not only the
  commanded one, settles at the target. A load held in the gripper adds its own sag; the correction does not know
  it.
* No obstacle avoidance: the controller moves the tool towards the target whatever lies on the way.

Everything here is a pure function of the requests and the robot's own state, computed with MuJoCo's own C math
(no BLAS): the same requests give the same joint targets in any process.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .robot import JOINT_NAMES, SIDES, build_robot_spec

__all__ = ["ToolKinematics", "ToolPoseLevel", "TOOL_DEFAULTS", "quat_angle_deg"]


def _dot(a, b) -> float:
    """Exactly rounded dot product of two short vectors (the same bits on any CPU)."""
    return math.fsum(float(x) * float(y) for x, y in zip(a, b))


def _norm(a) -> float:
    return math.sqrt(_dot(a, a))


def _rot(R, v) -> np.ndarray:
    """R @ v for a 3 x 3 matrix, exactly rounded per component."""
    return np.array([_dot(R[i], v) for i in range(3)])

TOOL_DEFAULTS = {
    "iterations": 6,             # damped least-squares iterations per tick (at most)
    "damping": 1.0e-4,           # constant part of the damping
    "residual_damping": 0.1,     # the residual part: this x 1/2 |e|^2 (weighted error, m^2)
    "rot_length_m": 2.0,         # 1 rad of orientation error weighs like this much position error
    "max_step_error_m": 0.5,     # the (weighted) error one iteration aims at is at most this long
    "joint_margin_rad": 0.01,    # solutions stay this far inside each joint's range
    "speed_share": 0.95,         # share of the command speed limit the solution may use per tick
    "bound_approach": 0.3,       # per tick a joint covers at most this share of its distance to a bound
    "reached_m": 0.001,          # "reached": position error at most this
    "reached_deg": 0.5,          # and orientation error at most this
    "stall_rel": 0.002,          # out of reach: a tick improves the error by less than this share
    "stall_ticks": 5,            # for this many ticks in a row
    "unlatch_m": 0.005,          # a set flag clears when the position error falls this much below its value at
    "unlatch_rel": 0.05,         # the flag's setting (or by this share of it, whichever is larger)
    "gravity_compensation": True,
}

_SIDE_KEY = {"left": "Left", "right": "Right"}


def quat_angle_deg(qa, qb) -> float:
    """The angle (degrees) between two orientations given as unit quaternions (w, x, y, z)."""
    c = abs(_dot(qa, qb))
    return math.degrees(2.0 * math.acos(min(1.0, c)))


class ToolKinematics:
    """A robot-only copy of the cell's robot (the same arms, grippers and mounts, nothing else): forward
    kinematics of the tool centre points, their Jacobians and the arms' static gravity torques. It never touches
    the cell's world."""

    def __init__(self, lock_couplers=None, mounts=None):
        spec = build_robot_spec(lock_couplers, mounts)
        self.m = m = spec.compile()
        self.d = mujoco.MjData(m)
        J = mujoco.mjtObj.mjOBJ_JOINT
        jid = [mujoco.mj_name2id(m, J, n) for n in JOINT_NAMES]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.dadr = np.array([m.jnt_dofadr[j] for j in jid])
        self.tcp = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "%s_tcp" % s) for s in SIDES}
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self._bias = np.zeros(m.nv)
        self._q = None

    def set(self, q14) -> None:
        q14 = np.asarray(q14, float)
        if self._q is not None and np.array_equal(self._q, q14):
            return
        d = self.d
        d.qpos[:] = self.m.qpos0
        d.qpos[self.qadr] = q14
        d.qvel[:] = 0.0
        mujoco.mj_kinematics(self.m, d)
        mujoco.mj_comPos(self.m, d)
        self._q = q14.copy()

    def pose(self, side: str):
        """(position, rotation matrix 3x3) of the side's TCP at the last ``set``."""
        s = self.tcp[side]
        return self.d.site_xpos[s].copy(), self.d.site_xmat[s].reshape(3, 3).copy()

    def pose_quat(self, side: str):
        p, R = self.pose(side)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.reshape(9))
        if q[0] < 0:
            q = -q
        return p, q

    def jacobian(self, side: str, cols) -> np.ndarray:
        """6 x 7 Jacobian (linear rows, then angular rows; world frame) of the side's TCP over the arm's joints."""
        mujoco.mj_jacSite(self.m, self.d, self._jacp, self._jacr, self.tcp[side])
        return np.ascontiguousarray(np.vstack([self._jacp[:, cols], self._jacr[:, cols]]))

    def gravity(self) -> np.ndarray:
        """The 14 arm joints' static gravity torques (N m) at the last ``set``."""
        mujoco.mj_comVel(self.m, self.d)
        mujoco.mj_rne(self.m, self.d, 0, self._bias)
        return self._bias[self.dadr].copy()


class ToolPoseLevel:
    """The tool-pose level of one cell (one per shift). ``world``: the ShiftWorld; ``cfg``: the configuration's
    ``world.tool_pose`` block (over ``TOOL_DEFAULTS``); ``speed_limit``: rad/s; ``tick_s``: s."""

    def __init__(self, world, cfg: dict | None, speed_limit: float, tick_s: float = 0.02):
        c = dict(TOOL_DEFAULTS)
        c.update(cfg or {})
        self.cfg = c
        self.kin = ToolKinematics(world.cfg.lock_couplers, world.cfg.robot_mounts)
        m = world.model
        self.lo = np.asarray(world._ctrl_lo, float).copy()
        self.hi = np.asarray(world._ctrl_hi, float).copy()
        self.kp = np.array([float(m.actuator_gainprm[a, 0]) for a in world._arm_aid])
        self.step_max = float(speed_limit) * float(tick_s)
        self.cols = {s: self.kin.dadr[i * 7:(i + 1) * 7] for i, s in enumerate(SIDES)}
        self.idx = {s: np.arange(i * 7, (i + 1) * 7) for i, s in enumerate(SIDES)}
        self.L = float(c["rot_length_m"])
        self.margin = float(c["joint_margin_rad"])
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        self.target = {s: None for s in SIDES}      # (pos 3, quat 4)
        self.q_ik = {s: None for s in SIDES}        # the arm's kinematic solution (7)
        self.status = {s: None for s in SIDES}
        self.ticks = {s: 0 for s in SIDES}
        self.out_of_reach_ticks = {s: 0 for s in SIDES}
        self.commands = {s: 0 for s in SIDES}
        self.latch = {s: None for s in SIDES}        # position error when the out-of-reach flag was set
        self.stall_run = {s: 0 for s in SIDES}       # consecutive ticks without the stall_rel gain

    @property
    def active(self) -> bool:
        return any(self.target[s] is not None for s in SIDES)

    def level(self, side: str) -> str:
        return "tool_pose" if self.target[side] is not None else "joint"

    def set_target(self, side: str, pos, quat, q_applied14) -> None:
        q = np.asarray(quat, float).copy()
        q /= _norm(q)
        if q[0] < 0:
            q = -q
        pos = np.asarray(pos, float).copy()
        old = self.target[side]
        if old is None:
            self.q_ik[side] = np.asarray(q_applied14, float)[self.idx[side]].copy()
        elif not (np.array_equal(old[0], pos) and np.array_equal(old[1], q)) and self.status[side] is not None:
            self.status[side] = dict(self.status[side], out_of_reach=False)   # a new target: solve afresh
            self.latch[side] = None
        self.target[side] = (pos, q)
        self.commands[side] += 1

    def clear(self, side: str | None = None) -> None:
        for s in (SIDES if side is None else (side,)):
            self.target[s] = None
            self.q_ik[s] = None
            self.status[s] = None
            self.latch[s] = None
            self.stall_run[s] = 0

    # ------------------------------------------------------------------ solve
    def _error(self, side, q14, tp, tq):
        """Weighted 6-vector error (position m; orientation rad x rot_length) and the unweighted parts."""
        self.kin.set(q14)
        p, R = self.kin.pose(side)
        cq = np.zeros(4)
        mujoco.mju_mat2Quat(cq, R.reshape(9))
        r_local = np.zeros(3)
        mujoco.mju_subQuat(r_local, tq, cq)             # cq * exp(r_local) = tq
        r = _rot(R, r_local)                            # world frame
        ep = tp - p
        e = np.concatenate([ep, self.L * r])
        return e, _norm(ep), math.degrees(_norm(r))

    def _lm_step(self, side, q14, e, locked):
        J = self.kin.jacobian(side, self.cols[side])
        J[3:, :] *= self.L
        J[:, locked] = 0.0
        en = _norm(e)
        cap = float(self.cfg["max_step_error_m"])
        eu = e * (cap / en) if en > cap else e
        A = np.zeros((7, 7))
        mujoco.mju_mulMatTMat(A, J, J)
        lam = float(self.cfg["residual_damping"]) * 0.5 * _dot(eu, eu) + float(self.cfg["damping"])
        A[np.diag_indices(7)] += lam
        g = np.zeros(7)
        mujoco.mju_mulMatTVec(g, J, np.ascontiguousarray(eu))
        mujoco.mju_cholFactor(A, 0.0)
        dq = np.zeros(7)
        mujoco.mju_cholSolve(dq, A, g)
        dq[locked] = 0.0
        return dq

    def solve(self, side: str, q_applied14) -> np.ndarray:
        """One tick of the side's arm: returns its 7 joint targets (gravity correction included) and sets
        ``status[side]``."""
        c = self.cfg
        tp, tq = self.target[side]
        ix = self.idx[side]
        q14 = np.asarray(q_applied14, float).copy()
        q0 = self.q_ik[side].copy()
        lo, hi = self.lo[ix] + self.margin, self.hi[ix] - self.margin
        q0 = np.clip(q0, lo, hi)
        budget = float(c["speed_share"]) * self.step_max
        approach = float(c.get("bound_approach", 0.3))
        q = q0.copy()
        q14[ix] = q
        e, ep, er = self._error(side, q14, tp, tq)
        ep0, er0 = ep, er
        f0 = f = _dot(e, e)
        locked = np.zeros(7, bool)
        speed_limited = False
        tol_p, tol_r = float(c["reached_m"]), float(c["reached_deg"])
        range_limited = False
        for _ in range(int(c["iterations"])):
            if ep <= 0.05 * tol_p and er <= 0.05 * tol_r:
                break
            dq = self._lm_step(side, q14, e, locked)
            # joint ranges: a joint at its bound that the step pushes outward is held there for the rest of this
            # tick and the step is solved again without it (the other joints take over)
            for _p in range(3):
                push = ((q <= lo + 1e-4) & (dq < 0)) | ((q >= hi - 1e-4) & (dq > 0))
                if not np.any(push & ~locked):
                    break
                locked |= push
                range_limited = True
                dq = self._lm_step(side, q14, e, locked)
            # scale the whole step (its direction kept): every joint within the tick's speed budget from q0 and
            # within its range
            s = 1.0
            s_b = 1.0
            for i in range(7):
                if abs(dq[i]) < 1e-15:
                    continue
                cur = q[i] - q0[i]
                room = budget - cur if dq[i] > 0 else budget + cur
                s = min(s, max(0.0, room) / abs(dq[i]))
                # towards a bound a joint slows down: per tick it covers at most a share of what is left
                rb = (approach * (hi[i] - q0[i]) - cur) if dq[i] > 0 else (approach * (q0[i] - lo[i]) + cur)
                s_b = min(s_b, max(0.0, rb) / abs(dq[i]))
            if s < 1.0 and s <= s_b:
                speed_limited = True
            if s_b < s:
                s = s_b
                range_limited = True
            if s <= 0.0:
                break
            # backtracking: accept only a step that brings the tool closer
            acc = False
            for _k in range(6):
                qt = np.clip(q + s * dq, lo, hi)
                q14[ix] = qt
                e2, ep2, er2 = self._error(side, q14, tp, tq)
                f2 = _dot(e2, e2)
                if f2 < f:
                    acc = True
                    break
                s *= 0.5
            if not acc:
                q14[ix] = q
                break
            q, e, ep, er, f = qt, e2, ep2, er2, f2
            if speed_limited:
                break
        gain = f0 - f
        prev = (self.status[side] or {}).get("out_of_reach", False)
        q14[ix] = q
        self.kin.set(q14)                                # the pose of the kept solution
        self.q_ik[side] = q.copy()
        out = q.copy()
        if c.get("gravity_compensation", True):
            tau = self.kin.gravity()
            out = q + tau[ix] / self.kp[ix]
        # never beyond the cell's own speed limit from the applied target, never beyond the joint ranges
        qa = np.asarray(q_applied14, float)[ix]
        lim = self.step_max * (1.0 - 1e-9)
        out = np.clip(np.clip(out, qa - lim, qa + lim), self.lo[ix], self.hi[ix])
        reached = ep <= tol_p and er <= tol_r
        # out of reach: a tick no longer brings the solution closer by stall_rel; then the flag latches until the
        # target changes (set_target), the target is reached, or the position error falls clearly below its value
        # when the flag was set
        lat = self.latch[side]
        if reached:
            oor = False
        elif lat is not None:
            oor = ep > lat - max(float(c.get("unlatch_m", 0.005)), float(c.get("unlatch_rel", 0.05)) * lat)
        else:
            stalled = (not speed_limited) and gain <= float(c["stall_rel"]) * f0 + 1e-14
            self.stall_run[side] = self.stall_run[side] + 1 if stalled else 0
            oor = self.stall_run[side] >= int(c.get("stall_ticks", 1))
        if lat is not None and not oor:
            self.stall_run[side] = 0                     # the flag cleared by progress: count the stall afresh
        self.latch[side] = ep if (oor and lat is None) else (lat if oor else None)
        self.ticks[side] += 1
        if oor:
            self.out_of_reach_ticks[side] += 1
        self.status[side] = {"reached": bool(reached), "out_of_reach": bool(oor),
                             "error_m": ep, "error_deg": er, "speed_limited": bool(speed_limited),
                             "range_limited": bool(range_limited)}
        return out

    def tick(self, q_hold14, q_applied14) -> np.ndarray:
        """The 14 targets of this tick: every arm at the tool-pose level gets its solution, the others keep
        ``q_hold14``."""
        q = np.asarray(q_hold14, float).copy()
        for s in SIDES:
            if self.target[s] is not None:
                q[self.idx[s]] = self.solve(s, q_applied14)
        return q

    # ------------------------------------------------------------------ the wire
    def measured(self, q14) -> dict:
        """{"left": {"pos", "quat"}, "right": ...}: each tool's pose at the measured joint positions."""
        self.kin.set(q14)
        out = {}
        for s in SIDES:
            p, qq = self.kin.pose_quat(s)
            out[s.lower()] = {"pos": [round(float(v), 6) for v in p], "quat": [round(float(v), 6) for v in qq]}
        return out

    def report(self) -> dict:
        """{"left": null | {...}, "right": ...}: each arm's tool-pose target and how far the solution is from it."""
        out = {}
        for s in SIDES:
            t = self.target[s]
            if t is None:
                out[s.lower()] = None
                continue
            st = self.status[s] or {}
            out[s.lower()] = {"pos": [round(float(v), 6) for v in t[0]],
                              "quat": [round(float(v), 6) for v in t[1]],
                              "reached": bool(st.get("reached", False)),
                              "out_of_reach": bool(st.get("out_of_reach", False)),
                              "error_m": round(float(st["error_m"]), 6) if "error_m" in st else None,
                              "error_deg": round(float(st["error_deg"]), 4) if "error_deg" in st else None}
        return out

    def summary(self) -> dict:
        return {"commands": {s.lower(): int(self.commands[s]) for s in SIDES},
                "ticks": {s.lower(): int(self.ticks[s]) for s in SIDES},
                "out_of_reach_ticks": {s.lower(): int(self.out_of_reach_ticks[s]) for s in SIDES}}
