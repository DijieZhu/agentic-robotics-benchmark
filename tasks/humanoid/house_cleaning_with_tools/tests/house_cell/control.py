"""The robots' on-board controllers, as the cell exposes them (the cell manual documents every number here).

Per robot, each 10 ms control tick:
  * locomotion: a velocity command (vx, vy forward / left in the robot's heading frame, wz) or a goal pose (x, y, yaw in
    the room, straight-line approach, no obstacle avoidance) moves the pelvis targets with speed and acceleration
    limits; the stand height moves the base_z target and the leg posture;
  * waist: joint targets (yaw, roll, pitch), rate-limited;
  * arms: per arm either joint targets or a grasp-site pose target (room frame); a pose target goes through the cell's
    6-DoF damped-least-squares IK every tick from the measured state (a target near the previous one keeps the arm's
    configuration; a far one gets a seeded global solve); joint targets are rate-limited;
  * hands: open / close synergies (a hand that holds something closes on it: physics.Grips.hand_shape).
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from .robot import (ARM_HOME, HAND_CLOSED, HAND_OPEN, PELVIS_HEIGHT, SIDES, STAND_RANGE, ArmIK, LegPosture,
                    RobotIndex)

TICK = 0.01
V_MAX = (0.5, 0.3, 0.8)            # vx, vy (m/s), wz (rad/s)
A_MAX = (0.8, 0.8, 1.5)            # their accelerations
STAND_RATE = 0.15                  # m/s
WAIST_RATE = 0.8                   # rad/s
ARM_RATE = 1.5                     # rad/s per joint
GOAL_GAIN = 1.2                    # 1/s, goal-pose approach (P on the remaining distance)
GOAL_TOL = (0.005, 0.01)           # m, rad: a goal counts as reached
ARM_ACC = 6.0                      # rad/s^2 per joint
CART_SPEED, CART_ACC = 0.25, 1.0   # grasp-site pose targets: m/s, peak m/s^2 ...
ROT_SPEED, ROT_ACC = 1.0, 4.0      # ... and rad/s, peak rad/s^2
MJ_PEAK = 5.7735                   # peak acceleration of a minimum-jerk move = MJ_PEAK * distance / duration^2
NEAR_POS, NEAR_ROT = 0.12, 0.6     # m, rad: a new pose target this close to the previous one keeps the arm's branch


def approach(cur: np.ndarray, goal: np.ndarray, vel: np.ndarray, vmax, amax, per_axis: bool = False):
    """One tick of a speed- and acceleration-limited approach that stops on the goal: (new value, new velocity)."""
    delta = goal - cur
    if per_axis:
        vstop = np.sqrt(2.0 * amax * np.abs(delta))
        vdes = np.sign(delta) * np.minimum(vmax, vstop)
        vel = vel + np.clip(vdes - vel, -amax * TICK, amax * TICK)
        nxt = cur + vel * TICK
        over = np.sign(goal - nxt) != np.sign(delta)
        nxt[over] = goal[over]
        vel[over] = 0.0
        return nxt, vel
    dist = float(np.linalg.norm(delta))
    if dist < 1e-9:
        return goal.copy(), np.zeros_like(vel)
    vdes = min(vmax, math.sqrt(2.0 * amax * dist)) * delta / dist
    dv = vdes - vel
    n = float(np.linalg.norm(dv))
    if n > amax * TICK:
        dv *= amax * TICK / n
    vel = vel + dv
    step = vel * TICK
    if float(np.linalg.norm(step)) >= dist:
        return goal.copy(), np.zeros_like(vel)
    return cur + step, vel


def slerp_toward(q: np.ndarray, goal: np.ndarray, max_angle: float) -> np.ndarray:
    if float(q @ goal) < 0:
        goal = -goal
    dot = float(np.clip(q @ goal, -1.0, 1.0))
    ang = 2.0 * math.acos(dot)
    if ang <= max_angle or ang < 1e-9:
        return goal / np.linalg.norm(goal)
    t = max_angle / ang
    th = math.acos(dot)
    out = (math.sin((1 - t) * th) * q + math.sin(t * th) * goal) / math.sin(th)
    return out / np.linalg.norm(out)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class RobotController:
    def __init__(self, m: mujoco.MjModel, r: RobotIndex):
        self.m, self.r = m, r
        self.legs = LegPosture(m, r)
        self.ik = ArmIK(m, r)
        self.reset_targets(None)

    def reset_targets(self, d) -> None:
        r = self.r
        self.vel_cmd = np.zeros(3)
        self.vel = np.zeros(3)            # current commanded base velocity (heading frame), after acceleration limits
        self.goal = None                  # (x, y, yaw) or None
        self.base_target = np.zeros(3) if d is None else d.qpos[r.base_q[:3]].copy()
        self.stand_cmd = PELVIS_HEIGHT
        self.stand = PELVIS_HEIGHT
        self.waist_cmd = np.zeros(3)
        self.waist = np.zeros(3)
        self.arm_mode = {s: "joint" for s in SIDES}
        self.arm_cmd = {s: np.array(ARM_HOME[s], float) for s in SIDES}
        self.arm_pose = {s: None for s in SIDES}
        self.arm = {s: np.array(ARM_HOME[s], float) for s in SIDES}
        self.arm_vel = {s: np.zeros(7) for s in SIDES}
        self.pose_cur = {s: None for s in SIDES}       # smoothed grasp-site target (pos, quat, frame)
        self.pose_vel = {s: np.zeros(3) for s in SIDES}
        self.q_goal = {s: np.array(ARM_HOME[s], float) for s in SIDES}
        self.seg = {s: None for s in SIDES}            # line segment: (start pos, start quat, duration, elapsed)
        self.seg_group = {s: 0 for s in SIDES}
        self.seg_end = {}
        self._group = 0
        self.arm_ik_err = {s: (0.0, 0.0) for s in SIDES}
        self.hand_cmd = {s: "open" for s in SIDES}
        self.hand_shape = {s: None for s in SIDES}     # a holding hand's finger targets (physics.Grips.hand_shape)
        if d is not None:
            for s in SIDES:
                self.arm[s] = d.qpos[r.arm_q[s]].copy()
                self.arm_cmd[s] = self.arm[s].copy()
            self.waist = d.qpos[r.waist_q].copy()
            self.waist_cmd = self.waist.copy()
            self.stand = PELVIS_HEIGHT + float(d.qpos[r.base_q[3]])
            self.stand_cmd = self.stand

    # ---------------------------------------------------------------------------------------------- commands
    def command(self, cmd: dict) -> list[str]:
        """Apply one command dict (already validated for types); returns notes on clamped values."""
        notes = []
        if "base_velocity" in cmd:
            v = np.array(cmd["base_velocity"], float)
            c = np.clip(v, -np.array(V_MAX), np.array(V_MAX))
            if np.any(np.abs(c - v) > 1e-9):
                notes.append("base_velocity clamped")
            self.vel_cmd = c
            self.goal = None
        if "base_goal" in cmd:
            g = cmd["base_goal"]
            self.goal = None if g is None else np.array(g, float)
            self.vel_cmd = np.zeros(3)
        if "stand_height" in cmd:
            h = float(cmd["stand_height"])
            self.stand_cmd = float(np.clip(h, STAND_RANGE[0], STAND_RANGE[1]))
            if abs(self.stand_cmd - h) > 1e-9:
                notes.append("stand_height clamped")
        if "waist" in cmd:
            w = np.array(cmd["waist"], float)
            r, m = self.r, self.m
            lo = np.array([m.jnt_range[m.dof_jntid[v], 0] for v in r.waist_v])
            hi = np.array([m.jnt_range[m.dof_jntid[v], 1] for v in r.waist_v])
            self.waist_cmd = np.clip(w, lo, hi)
        self._group += 1
        for s in SIDES:
            a = cmd.get(f"{s}_arm")
            if a is None:
                continue
            if "q" in a:
                q = np.array(a["q"], float)
                self.arm_cmd[s] = np.clip(q, self.r.arm_lo[s], self.r.arm_hi[s])
                self.arm_mode[s] = "joint"
                self.arm_pose[s] = None
                self.pose_cur[s] = None
                self.seg_end.pop(s, None)
            elif "pose" in a or "pos" in a:
                if "pose" in a:
                    p = np.array(a["pose"], float)
                    pos, quat = p[:3].copy(), p[3:7] / np.linalg.norm(p[3:7])
                else:                                       # a position-only target: the wrist is free
                    pos, quat = np.array(a["pos"], float)[:3].copy(), None
                frame = "room" if a.get("frame", "base") == "room" else "base"
                path = "joint" if a.get("path") == "joint" else "line"
                self.arm_pose[s] = (pos, quat, frame, path)
                self.pose_cur[s] = None                     # (re)started on the next tick
                self.seg_group[s] = self._group             # arms commanded together share one timeline
                self.arm_mode[s] = "pose"
            h = cmd.get(f"{s}_hand")
        for s in SIDES:
            h = cmd.get(f"{s}_hand")
            if h in ("open", "close"):
                self.hand_cmd[s] = h
        return notes

    # ---------------------------------------------------------------------------------------------- per tick
    def tick(self, d: mujoco.MjData) -> None:
        r = self.r
        x, y, yaw = (float(v) for v in d.qpos[r.base_q[:3]])
        # locomotion: desired heading-frame velocity from the command or the goal
        if self.goal is not None:
            gx, gy, gyaw = self.goal
            tx, ty, tyaw = self.base_target
            dx, dy = gx - tx, gy - ty
            c, s_ = math.cos(tyaw), math.sin(tyaw)
            fx, fy = c * dx + s_ * dy, -s_ * dx + c * dy
            want = np.array([GOAL_GAIN * fx, GOAL_GAIN * fy, GOAL_GAIN * wrap(gyaw - tyaw)])
            if math.hypot(dx, dy) < GOAL_TOL[0] and abs(wrap(gyaw - tyaw)) < GOAL_TOL[1]:
                want[:] = 0.0
        else:
            want = self.vel_cmd
        want = np.clip(want, -np.array(V_MAX), np.array(V_MAX))
        dv = np.clip(want - self.vel, -np.array(A_MAX) * TICK, np.array(A_MAX) * TICK)
        self.vel = self.vel + dv
        tyaw = self.base_target[2]
        c, s_ = math.cos(tyaw), math.sin(tyaw)
        self.base_target[0] += (c * self.vel[0] - s_ * self.vel[1]) * TICK
        self.base_target[1] += (s_ * self.vel[0] + c * self.vel[1]) * TICK
        self.base_target[2] = tyaw + self.vel[2] * TICK
        if self.goal is not None:
            gx, gy, gyaw = self.goal
            if math.hypot(gx - self.base_target[0], gy - self.base_target[1]) < GOAL_TOL[0] and \
                    abs(wrap(gyaw - self.base_target[2])) < GOAL_TOL[1]:
                self.base_target[:] = (gx, gy, self.base_target[2] + wrap(gyaw - self.base_target[2]))
                self.vel[:] = 0.0
        # the pelvis target may run ahead of an obstructed robot by at most 5 cm / 0.1 rad (no wind-up)
        lag = self.base_target[:2] - np.array([x, y])
        n = float(np.linalg.norm(lag))
        if n > 0.05:
            self.base_target[:2] = np.array([x, y]) + lag * (0.05 / n)
            self.vel[:2] *= 0.5
        e = wrap(self.base_target[2] - yaw)
        if abs(e) > 0.1:
            self.base_target[2] = yaw + math.copysign(0.1, e)
        d.ctrl[r.base_act[0]] = self.base_target[0]
        d.ctrl[r.base_act[1]] = self.base_target[1]
        d.ctrl[r.base_act[2]] = self.base_target[2]
        # stand height and legs
        self.stand += float(np.clip(self.stand_cmd - self.stand, -STAND_RATE * TICK, STAND_RATE * TICK))
        d.ctrl[r.base_act[3]] = self.stand - PELVIS_HEIGHT
        legs = self.legs.targets(self.stand)
        for s in SIDES:
            d.ctrl[r.leg_act[s]] = legs
        # waist
        self.waist += np.clip(self.waist_cmd - self.waist, -WAIST_RATE * TICK, WAIST_RATE * TICK)
        d.ctrl[r.waist_act] = self.waist
        # arms
        cz, sz = math.cos(yaw), math.sin(yaw)
        Rb = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        qb = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
        for s in SIDES:
            if self.arm_mode[s] == "pose" and self.arm_pose[s] is not None:
                gpos, gquat, frame, path = self.arm_pose[s]

                def to_room(p_, q_):
                    if frame != "base":
                        return p_, q_
                    if q_ is None:
                        return Rb @ p_ + np.array([x, y, 0.0]), None
                    qw = np.zeros(4)
                    mujoco.mju_mulQuat(qw, qb, q_)
                    return Rb @ p_ + np.array([x, y, 0.0]), qw

                if self.pose_cur[s] is None:
                    # start of a new target: its IK branch. A target near the previous one (within NEAR_POS and
                    # NEAR_ROT, same frame) stays on the arm's present configuration (a local solve from the present
                    # joint targets); a far one, or a near one the local solve cannot reach, gets a seeded global
                    # solve. A "line" path then interpolates the grasp-site pose from where the last target left it
                    # (or from the measured pose)
                    gp, gq = to_room(gpos, gquat)
                    prev = self.seg_end.get(s)
                    near = (prev is not None and prev[2] == frame and float(np.linalg.norm(gpos - prev[0])) < NEAR_POS
                            and ((gquat is None and prev[1] is None) or (gquat is not None and prev[1] is not None and
                                 2.0 * math.acos(min(1.0, abs(float(np.dot(prev[1], gquat))))) < NEAR_ROT)))
                    qg = None
                    if near:
                        qg, ep_l, er_l = self.ik.solve(d.qpos, s, gp, gq, self.arm_cmd[s], iters=40)
                        if ep_l > 0.01 or er_l > 0.05:
                            qg = None
                    if qg is None:
                        qg, _, _ = self.ik.solve_global(d.qpos, s, gp, gq, self.arm_cmd[s])
                    self.q_goal[s] = qg
                    if prev is not None and prev[2] == frame and prev[1] is not None and gquat is not None:
                        p0, q0 = prev[0].copy(), prev[1].copy()
                    elif prev is not None and prev[2] == frame and gquat is None:
                        p0, q0 = prev[0].copy(), None
                    else:
                        sid = r.grasp_site[s]
                        wq = np.zeros(4)
                        mujoco.mju_mat2Quat(wq, d.site_xmat[sid])
                        if frame == "base":
                            p0 = Rb.T @ (d.site_xpos[sid] - np.array([x, y, 0.0]))
                            q0 = np.zeros(4)
                            mujoco.mju_mulQuat(q0, np.array([qb[0], -qb[1], -qb[2], -qb[3]]), wq)
                        else:
                            p0, q0 = d.site_xpos[sid].copy(), wq
                    self.pose_cur[s] = (p0, q0)
                    dist = float(np.linalg.norm(gpos - p0))
                    ang = 0.0 if (gquat is None or q0 is None) else 2.0 * math.acos(min(1.0, abs(float(np.dot(q0,
                                                                                                         gquat)))))
                    # the duration: the mean speeds, and the peak accelerations (a short move is not a fast jolt)
                    self.seg[s] = [p0, q0, max(0.25, 1.6 * dist / CART_SPEED, 1.6 * ang / ROT_SPEED,
                                               math.sqrt(MJ_PEAK * dist / CART_ACC), math.sqrt(MJ_PEAK * ang / ROT_ACC)),
                                   0.0]
                    # arms commanded together: one duration (the longer), so a board held by both is not torn
                    for o in SIDES:
                        if o != s and self.seg.get(o) is not None and self.seg_group[o] == self.seg_group[s] \
                                and self.seg[o][3] == 0.0:
                            dur = max(self.seg[o][2], self.seg[s][2])
                            self.seg[o][2] = self.seg[s][2] = dur
                    if path == "joint":
                        self.pose_cur[s] = (gpos.copy(), None if gquat is None else gquat.copy())
                    self.seg_end[s] = (gpos.copy(), None if gquat is None else gquat.copy(), frame)
                if path == "line":
                    p0, q0, dur, el = self.seg[s]
                    el = min(dur, el + TICK)
                    self.seg[s][3] = el
                    u = el / dur
                    w = u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)        # minimum-jerk progress
                    cpos = p0 + (gpos - p0) * w
                    if gquat is None or q0 is None:
                        cquat = None if gquat is None else gquat
                    else:
                        cquat = slerp_toward(q0, gquat, w * 2.0 * math.acos(min(1.0, abs(float(np.dot(q0, gquat)))))
                                             + 1e-12)
                    self.pose_cur[s] = (cpos, cquat)
                    pos, quat = to_room(cpos, cquat)
                    q, ep, er = self.ik.solve(d.qpos, s, pos, quat, self.arm_cmd[s], iters=6)
                    if ep > 0.02:                  # off the reachable path: fall back on the goal's branch
                        q2, ep2, er2 = self.ik.solve(d.qpos, s, pos, quat, self.q_goal[s], iters=10)
                        if ep2 < ep:
                            q, ep, er = q2, ep2, er2
                else:
                    pos, quat = to_room(gpos, gquat)
                    q, ep, er = self.ik.solve(d.qpos, s, pos, quat, self.q_goal[s], iters=4)
                    self.q_goal[s] = q
                self.arm_cmd[s] = q
                self.arm_ik_err[s] = (ep, er)
            self.arm[s], self.arm_vel[s] = approach(self.arm[s], self.arm_cmd[s], self.arm_vel[s], ARM_RATE, ARM_ACC,
                                                    per_axis=True)
            d.ctrl[r.arm_act[s]] = self.arm[s]
            if self.hand_cmd[s] == "close":
                d.ctrl[r.hand_act[s]] = HAND_CLOSED[s] if self.hand_shape[s] is None else self.hand_shape[s]
            else:
                d.ctrl[r.hand_act[s]] = HAND_OPEN[s]
