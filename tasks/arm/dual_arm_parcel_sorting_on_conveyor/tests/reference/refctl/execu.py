"""Executor: plays planned joint trajectories on the cell with the
inverse-dynamics feedforward, one 20 ms tick at a time.

The executor owns the planned state (q, qd of the 14 arm joints and the two
gripper commands). Contact forces wanted by a primitive are given as
per-arm world forces at a body point and are added through J^T F.
"""

from __future__ import annotations

import numpy as np

from parcelcell.robot import JOINT_NAMES

from .kin import ArmKin
from .motion import DT, Dyn, finite_diff, joint_move

SIDES = ("Left", "Right")
SL = {"Left": slice(0, 7), "Right": slice(7, 14)}


class Executor:
    def __init__(self, cell, log=None):
        self.cell = cell
        self.m = cell.model
        self.dyn = Dyn(cell.model)
        self.kin = {s: ArmKin(cell.model, s) for s in SIDES}
        self.q = cell.joint_state().q.copy()
        self.qd = np.zeros(14)
        self.grip = np.zeros(2)
        self.force = {s: None for s in SIDES}     # (body, point_local, F_world)
        self.hooks = []                           # per-tick callbacks
        self.log = log or (lambda *a: None)
        self.peak_cmd_speed = np.zeros(14)
        self.peak_qd = np.zeros(14)
        self.peak_tau = np.zeros(14)
        self._u_prev = None
        self.abort = False
        self.ticks = 0

    # ------------------------------------------------------------------
    def sync(self):
        """Take the planned state from the robot's measured state."""
        js = self.cell.joint_state()
        self.q = js.q.copy()
        self.qd = np.zeros(14)

    def tick(self, q=None, qd=None, qdd=None, grip=None):
        if q is not None:
            self.q = np.asarray(q, float)
        self.qd = np.zeros(14) if qd is None else np.asarray(qd, float)
        if grip is not None:
            self.grip = np.asarray(grip, float)
        tau_x = None
        for s in SIDES:
            f = self.force[s]
            if f is None:
                continue
            body, p_local, F = f
            if body == "site":
                J = self.dyn.jac_point(self.q, None, site=p_local)
            else:
                self.dyn.jac_point(self.q, body, np.zeros(3))
                J = self.dyn.jac_point(self.q, body,
                                       self._world_point(body, p_local))
            t = J.T @ np.asarray(F, float)
            tau_x = t if tau_x is None else tau_x + t
        u = self.dyn.command(self.q, self.qd, qdd, tau_x)
        if self._u_prev is not None:
            self.peak_cmd_speed = np.maximum(self.peak_cmd_speed,
                                             np.abs(u - self._u_prev) / DT)
        self._u_prev = u
        self.cell.step(u, self.grip)
        js = self.cell.joint_state()
        self.peak_qd = np.maximum(self.peak_qd, np.abs(js.qd))
        self.peak_tau = np.maximum(self.peak_tau, np.abs(js.tau))
        self.ticks += 1
        for h in self.hooks:
            h(self)

    def _world_point(self, body, p_local):
        d = self.dyn.d
        bid = self.m.body(body).id
        return d.xpos[bid] + d.xmat[bid].reshape(3, 3) @ np.asarray(p_local)

    # ------------------------------------------------------------------
    def play(self, qs, qds=None, qdds=None, grips=None):
        n = len(qs)
        if qds is None:
            qds, qdds = finite_diff(qs, self.q)
        for i in range(n):
            g = None if grips is None else grips[i]
            self.tick(qs[i], qds[i], None if qdds is None else qdds[i], g)
            if self.abort:
                return False
        return True

    def move_to(self, q_goal, vmax=2.0, tmin=0.3, grip=None):
        q, qd, qdd = joint_move(self.q, q_goal, vmax=vmax, tmin=tmin)
        grips = None if grip is None else [grip] * len(q)
        return self.play(q, qd, qdd, grips)

    def hold(self, seconds, grip=None):
        n = max(1, int(round(seconds / DT)))
        for _ in range(n):
            self.tick(self.q, None, None, grip)
            if self.abort:
                return False
        return True

    def arm_q(self, side):
        return self.q[SL[side]].copy()

    def with_arm(self, side, q7, base=None):
        q = (self.q if base is None else base).copy()
        q[SL[side]] = q7
        return q

    def measured_tcp_err(self):
        js = self.cell.joint_state()
        out = {}
        for s in SIDES:
            p_plan, _ = self.kin[s].fk(self.q[SL[s]])
            p_meas, _ = self.kin[s].fk(js.q[SL[s]])
            out[s] = float(np.linalg.norm(p_plan - p_meas))
        return out
