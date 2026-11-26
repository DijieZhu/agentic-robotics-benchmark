"""Pipe spool push-fit assembly: the privileged scripted two-arm reference (reads the simulator's truth).

`Reference(ep).run()` drives an episode object whose `step(action)` applies one wire action and whose `env.w` is a
`spool_sim.World` holding the true state (over the wire: wire_world.ShadowEpisode, synced from `truth` every step).

0. An elbow that stands on a socket's mouth is tipped over first (`upright`): the right hand tips E1, the left hand E2.
1. The left hand takes pipe A from above, 6 cm from its plain end, lifts it, turns it so that its banded end points
   to -y, and holds it along y at the work station.
2. The right hand takes the nearer elbow (E1) from above by one socket's barrel (the outlet), its fingers closing along
   the other socket's axis, lifts it and turns it so that that socket faces A's banded end, with the outlet along x.
   It brings the socket onto A's end (the error read from the true poses of the socket and the pipe end), then pushes
   along the axis until the joint is home, while the left hand holds A against the push.
3. The left hand lets go of A (A now hangs from E1 in the right hand), takes pipe B from above and pushes one of its
   ends into E1's outlet while the right hand holds E1.
4. The left hand lets go of B, takes the other elbow (E2) from above by its outlet barrel, turns it so that its free
   socket faces A's plain end and its outlet points along the order's twist, and pushes it onto A while the right
   hand holds E1.
5. The left hand lets go, the right hand sets the spool down on the bench and lets go; both hands back off.
"""

import math

import mujoco
import numpy as np

import spool_sim as S

HOLD = {"l": [0.0] * 6 + [-1.0], "r": [0.0] * 6 + [-1.0], "cams": 0}
Z = np.array([0.0, 0.0, 1.0])
E1_CORNER = np.array([0.38, -0.13, S.TABLE_Z + 0.20])   # E1's corner while B and A go in
E2_CORNER = np.array([0.50, -0.13, S.TABLE_Z + 0.20])   # E1's corner while E2 goes on
GRASP_FROM_END = 0.035
B_GRASP_FROM_END = 0.022
PICK_ABOVE = 0.003                                     # the grip site above a picked elbow barrel's axis
PIPE_PICK_ABOVE = 0.003                                 # ... and above a pipe's axis (the pipe lies on the bench)
TIP_START = 0.045                                      # a standing elbow is pushed over: the hand starts this far
TIP_ABOVE = 0.015                                      # from the corner, this high above it, and pushes it
TIP_PUSH = 0.10                                        # this far
LIFT = 0.12
SAFE_Z = S.TABLE_Z + 0.36                              # travel height between picks
ELBOW_GRASP = 0.036                                    # the grasp on a barrel, from the corner
OPEN, CLOSE = -1.0, 1.0
IK_FORWARD = np.array([0.0, 0.3, 0.0, -1.9, 0.0, 2.2, math.pi / 4])   # a start for hands pointing forward


def rotvec(Rm):
    """The rotation vector of rotation matrix Rm."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(Rm, float).reshape(9))
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def rot_err(R_des, R):
    return rotvec(R_des @ R.T)


def hand_R(close, approach):
    """The grip site's rotation whose fingers close along `close` and which approaches along `approach`."""
    x = np.asarray(close, float)
    z = np.asarray(approach, float)
    x = x - (x @ z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def yaw_toward(R_from, R_to, max_step):
    """A rotation from R_from toward R_to by at most max_step rad."""
    v = rotvec(R_to @ R_from.T)
    n = np.linalg.norm(v)
    if n <= max_step:
        return R_to
    return S.axis_angle_mat(v * (max_step / n)) @ R_from


def yaw_angle(a, b):
    """The rotation about z (rad, in (-pi, pi]) taking a's horizontal direction to b's."""
    return math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1])


class Failed(RuntimeError):
    pass


class Reference:

    def __init__(self, ep):
        self.ep = ep
        self.w = ep.env.w
        self.d = self.w.d
        self.integ = {"l": np.zeros(3), "r": np.zeros(3)}
        self.integ_r = {"l": np.zeros(3), "r": np.zeros(3)}
        self.grip = {"l": OPEN, "r": OPEN}
        self.events = []
        self.t = 0
        self.hold_pose = {"l": None, "r": None}
        self.debug = False
        self.pick_yaw = None

    # --- low-level -------------------------------------------------------------------------------

    def servo(self, s, p_des, R_des, grip=None, ki=0.0, ilim=0.04, vmax=None):
        """A Cartesian step of arm s toward (p_des, R_des); with ki, an integral term on both (against a held load);
        with vmax, the position step capped at vmax."""
        arm = self.w.arms[s]
        e = np.asarray(p_des, float) - arm.pos()
        if vmax is not None:
            n = np.linalg.norm(e)
            if n > vmax:
                e = e * (vmax / n)
        er = rot_err(R_des, arm.R())
        if ki:
            self.integ[s] = np.clip(self.integ[s] + ki * (np.asarray(p_des, float) - arm.pos()), -ilim, ilim)
            self.integ_r[s] = np.clip(self.integ_r[s] + ki * er, -0.15, 0.15)
        g = self.grip[s] if grip is None else grip
        self.grip[s] = g
        return [float(x) for x in e + self.integ[s]] + [float(x) for x in er + self.integ_r[s]] + [float(g)]

    def reset_integ(self, s):
        self.integ[s] = np.zeros(3)
        self.integ_r[s] = np.zeros(3)

    def holding(self, s):
        """Arm s's command that keeps its stored hold pose (or holds still where it is)."""
        hp = self.hold_pose[s]
        if hp is None:
            return [0.0] * 6 + [float(self.grip[s])]
        return self.servo(s, hp[0], hp[1], ki=hp[2])

    def step(self, l=None, r=None):
        act = {"l": l if l is not None else self.holding("l"), "r": r if r is not None else self.holding("r"),
               "cams": 0}
        _obs, done, _ = self.ep.step(act)
        self.t += 1
        if done:
            raise Failed("episode over")

    def run_until(self, cmd_fn, done_fn, max_steps, name):
        for _ in range(max_steps):
            if done_fn():
                self.events.append((name, self.t))
                return True
            cmd_fn()
        raise Failed("{} did not finish in {} steps".format(name, max_steps))

    def move(self, s, p_des, R_des, grip=None, tol=0.004, rtol=0.03, max_steps=150, vmax=0.03, name="move",
             rstep=0.2):
        """Arm s to (p_des, R_des) at most vmax per step, turning at most rstep per step."""
        arm = self.w.arms[s]

        def cmd():
            Rt = yaw_toward(arm.R(), R_des, rstep)
            if s == "l":
                self.step(l=self.servo(s, p_des, Rt, grip, vmax=vmax))
            else:
                self.step(r=self.servo(s, p_des, Rt, grip, vmax=vmax))

        def done():
            return (np.linalg.norm(arm.pos() - p_des) < tol
                    and np.linalg.norm(rot_err(R_des, arm.R())) < rtol)

        self.run_until(cmd, done, max_steps, name)

    def yaw_path(self, s, p0, R0, p1, delta, grip, vmax=0.02, name="yaw_path", tol=0.003, rtol=0.02):
        """Arm s from (p0, R0) to p1 while turning about z by delta (rad), along a straight line raised in the middle
        by 5 cm; then settles on (p1, yaw(delta) R0)."""
        n = max(int(np.linalg.norm(p1 - p0) / vmax), int(abs(delta) / 0.08), 1)
        for i in range(1, n + 1):
            f = i / n
            p = p0 + f * (p1 - p0) + np.array([0, 0, 0.05 * math.sin(math.pi * f)])
            Rt = S.yaw_mat(f * delta) @ R0
            if s == "l":
                self.step(l=self.servo(s, p, Rt, grip))
            else:
                self.step(r=self.servo(s, p, Rt, grip))
        self.move(s, p1, S.yaw_mat(delta) @ R0, grip, tol=tol, rtol=rtol, max_steps=100, name=name)

    def set_grip(self, s, g, steps=8):
        for _ in range(steps):
            if s == "l":
                self.step(l=self.servo(s, self.w.arms[s].pos(), self.w.arms[s].R(), g))
            else:
                self.step(r=self.servo(s, self.w.arms[s].pos(), self.w.arms[s].R(), g))

    def keep(self, s, ki=0.02):
        """Store arm s's present pose as its hold pose (servoed with an integral term)."""
        arm = self.w.arms[s]
        self.reset_integ(s)
        self.hold_pose[s] = (arm.pos(), arm.R(), ki)

    def release(self, s):
        self.hold_pose[s] = None
        self.reset_integ(s)

    # --- truth --------------------------------------------------------------------------------------

    def pipe_axis(self, p):
        b = self.w.pipe[p]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3)[:, 2].copy()

    def elbow_frame(self, e):
        b = self.w.elbow[e]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    # --- picking ---------------------------------------------------------------------------------------

    def close_sign(self, s, c, later_yaw):
        """The closing direction (+c or -c, horizontal) and the turn about world z (rad, possibly beyond pi) that take
        arm s's hand from its present closing axis to it for a top grasp, so that its wrist joint (q7) stays inside its
        limits at the grasp and after a later turn about z by later_yaw, turning as little as possible. With the hand
        pointing down, turning it about world z by y changes q7 by -y."""
        arm = self.w.arms[s]
        q7 = float(self.w.d.qpos[arm.jq[6]])
        lo, hi = float(arm.q_lo[6]) + 0.15, float(arm.q_hi[6]) - 0.15
        cur = arm.R()[:, 0]
        best = None
        for opt in (np.asarray(c, float), -np.asarray(c, float)):
            th0 = yaw_angle(cur, opt)
            for th in (th0, th0 - 2 * math.pi, th0 + 2 * math.pi):
                q_g, q_f = q7 - th, q7 - th - later_yaw
                inside = lo <= q_g <= hi and lo <= q_f <= hi
                cost = abs(th) + (0.0 if inside else 100.0 + max(0.0, lo - min(q_g, q_f), max(q_g, q_f) - hi))
                if best is None or cost < best[0]:
                    best = (cost, opt, th)
        self.pick_yaw = best[2]
        return best[1]

    def pick(self, s, p_grasp, close_dir, name, exact=False, above_axis=PICK_ABOVE):
        """Arm s takes a part from above at p_grasp (the part's axis point; the grip site above_axis higher), fingers
        closing along close_dir (unless `exact`, the sign nearest the hand's present closing axis)."""
        arm = self.w.arms[s]
        c = np.asarray(close_dir, float)
        c = c - (c @ Z) * Z
        c /= np.linalg.norm(c)
        if not exact and c @ arm.R()[:, 0] < 0:
            c = -c
        Rg = hand_R(c, -Z)
        above = p_grasp + np.array([0, 0, LIFT])
        at = p_grasp + np.array([0, 0, above_axis])
        z_safe = max(arm.pos()[2], above[2], SAFE_Z)
        self.move(s, np.array([arm.pos()[0], arm.pos()[1], z_safe]), arm.R(), OPEN, tol=0.02, rtol=0.2,
                  max_steps=80, name=name + ":rise")
        # the turn about z: the planned one (close_sign) when the grasp's sign was chosen with it, else the shortest
        R_down = hand_R(arm.R()[:, 0], -Z)
        th = self.pick_yaw if (exact and self.pick_yaw is not None) else yaw_angle(R_down[:, 0], c)
        self.pick_yaw = None
        self.yaw_path(s, arm.pos(), R_down, np.array([above[0], above[1], z_safe]), th, OPEN, tol=0.02, rtol=0.1,
                      name=name + ":over")
        self.move(s, above, Rg, OPEN, tol=0.01, max_steps=200, name=name + ":above")
        self.move(s, at, Rg, OPEN, tol=0.002, max_steps=120, vmax=0.01, name=name + ":down")
        self.set_grip(s, CLOSE, steps=10)
        self.move(s, above, Rg, CLOSE, tol=0.01, max_steps=120, vmax=0.015, name=name + ":lift")

    def run(self):
        try:
            self.job()
        except Failed as exc:
            self.events.append(("failed", str(exc), self.t))
        while True:
            self.step(l=[0.0] * 6 + [OPEN], r=[0.0] * 6 + [OPEN])

    def let_go(self, s, up=0.08):
        """Arm s opens and moves straight up (away from what it held)."""
        self.release(s)
        self.set_grip(s, OPEN, steps=8)
        arm = self.w.arms[s]
        self.move(s, arm.pos() + np.array([0, 0, up]), arm.R(), OPEN, tol=0.01, max_steps=60, name=s + ":let_go")
        self.keep(s, ki=0.0)

    def socket_axes(self, e, j):
        """Socket (e, j)'s outward axis, world."""
        _c, Re = self.elbow_frame(e)
        return Re[:, j].copy()

    def frames(self, axis, side):
        """An orthonormal frame [axis, side (made normal to axis), axis x side]."""
        a = np.asarray(axis, float)
        a = a / np.linalg.norm(a)
        b = np.asarray(side, float) - (np.asarray(side, float) @ a) * a
        if np.linalg.norm(b) < 1e-6:
            b = np.cross(a, [1.0, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1.0, 0])
        b /= np.linalg.norm(b)
        return np.column_stack([a, b, np.cross(a, b)])

    def joint_cmd(self, s, part_pt, part_ax, part_side, want_pt, want_ax, want_side, vmax, rmax=0.08, ki=0.0):
        """Arm s's step that moves the part it holds (a point, an axis and a side direction on it, now) toward the
        wanted point, axis and side: the hand moves rigidly with the part."""
        arm = self.w.arms[s]
        F0 = self.frames(part_ax, part_side)
        F1 = self.frames(want_ax, want_side)
        dR = F1 @ F0.T
        v = rotvec(dR)
        n = np.linalg.norm(v)
        if n > rmax:
            dR = S.axis_angle_mat(v * (rmax / n))
        p_hand = arm.pos()
        p_new = np.asarray(want_pt, float) + dR @ (p_hand - part_pt)
        return self.servo(s, p_new, dR @ arm.R(), vmax=vmax, ki=ki)

    def fit(self, mover, s, k, p, sign, side_want=None, gap=0.02, far=0.06, max_steps=900, name="fit"):
        """Join pipe p's end `sign` into socket k. `mover` is "elbow" (arm s holds the socket's elbow) or "pipe" (arm s
        holds the pipe). The other part is held by the other arm (its hold pose). With side_want, the elbow's other
        socket (mover "elbow") is turned toward that direction about the joint's axis. Brings the part `far` before
        the mouth, aligns `gap` before it, closes in, then pushes along the axis until the joint is home."""
        w = self.w
        e, j = w.sockets[k]
        other = 1 - j
        st = {"s_des": -far, "phase": "bring", "n": 0}
        stop = w.unit["stop"]

        def cmd():
            mouth, u = w.socket_pose(k)
            end, out = w.pipe_end(p, sign)
            depth = float((mouth - end) @ u)
            lat = np.linalg.norm((end - mouth) - ((end - mouth) @ u) * u)
            ang = math.degrees(math.acos(float(np.clip(-out @ u, -1, 1))))
            ki = 0.0
            if st["phase"] == "bring":
                vmax = 0.02
                if lat < 0.004 and ang < 2.0 and abs(depth + far) < 0.006:
                    st["phase"], st["s_des"] = "align", -gap
                    self.reset_integ(s)
            elif st["phase"] == "align":
                ki = 0.1
                st["n"] = st["n"] + 1 if (lat < 0.4 * S.LAT_TOL and ang < 0.3 * S.TILT_TOL_DEG) else 0
                if st["n"] >= 3:
                    st["s_des"] = min(st["s_des"] + 0.003, 0.0045)
                    if st["s_des"] >= 0.0045:
                        st["phase"] = "push"
                vmax = 0.01
            elif w.engaged[k] is None:
                # pressing on the O-ring: the push turns the parts in the grasps, so build it slowly and ease off
                # while the alignment is restored
                ki = 0.1
                if lat > 0.8 * S.LAT_TOL or ang > 0.8 * S.TILT_TOL_DEG:
                    st["s_des"] = max(st["s_des"] - 0.0005, depth - 0.001)
                elif lat < 0.5 * S.LAT_TOL and ang < 0.5 * S.TILT_TOL_DEG:
                    st["s_des"] = min(st["s_des"] + 0.0005, stop + 0.02)
                vmax = 0.01
            else:
                ki = 0.1
                st["s_des"] = min(max(st["s_des"], depth) + 0.001, stop + 0.02)
                vmax = 0.02
            sd = st["s_des"]
            if mover == "elbow":
                side = self.socket_axes(e, other)
                sw = side_want if side_want is not None else side
                # the socket's mouth where the pipe end is sd inside it; its axis opposite the pipe's out-direction
                cmd_ = self.joint_cmd(s, mouth, u, side, end + sd * (-out), -out, sw, vmax, ki=ki)
            else:
                side = w.arms[s].R()[:, 0]
                cmd_ = self.joint_cmd(s, end, out, side, mouth - sd * u, -u, side, vmax, ki=ki)
            if s == "l":
                self.step(l=cmd_)
            else:
                self.step(r=cmd_)
            if self.debug and self.t % 20 == 0:
                print("   fit", name, self.t, st["phase"], "s_des %.4f depth %.4f lat %.4f ang %.2f" % (
                    st["s_des"], depth, lat, ang), "eng", w.engaged[k], "hand", w.arms[s].pos().round(3))

        self.run_until(cmd, lambda: w.home(k), max_steps, name)
        self.reset_integ(s)
        for _ in range(5):                      # settle on the stop, the push eased
            self.step(**{s: self.servo(s, w.arms[s].pos(), w.arms[s].R())})

    def ik(self, s, p_des, R_des, seeds=None, iters=200):
        """Joint angles of arm s that put its grip site at (p_des, R_des): damped least squares on a scratch copy of
        the model, from several starting postures, within the joints' limits. Returns (q, position error, rotation
        error) of the best."""
        w = self.w
        m = w.m
        arm = w.arms[s]
        d = mujoco.MjData(m)
        best = None
        q_now = w.d.qpos[arm.jq].copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        rng = np.random.default_rng(12345)
        starts = [q_now] + list(seeds or []) + [S.HOME_Q, IK_FORWARD]
        starts += [arm.q_lo + rng.uniform(0.1, 0.9, 7) * (arm.q_hi - arm.q_lo) for _ in range(24)]
        for n_try, q0 in enumerate(starts):
            if best is not None and n_try >= 3 and best[1] < 0.002 and best[2] < 0.01:
                break
            d.qpos[:] = w.d.qpos
            q = np.clip(np.asarray(q0, float).copy(), arm.q_lo + 0.02, arm.q_hi - 0.02)
            for _ in range(iters):
                d.qpos[arm.jq] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                pe = p_des - d.site_xpos[arm.site]
                re = rot_err(R_des, d.site_xmat[arm.site].reshape(3, 3))
                mujoco.mj_jacSite(m, d, jp, jr, arm.site)
                J = np.vstack([jp, jr])[:, arm.jv]
                e = np.concatenate([pe, re])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
                q = np.clip(q + np.clip(dq, -0.1, 0.1), arm.q_lo + 0.02, arm.q_hi - 0.02)
            d.qpos[arm.jq] = q
            mujoco.mj_kinematics(m, d)
            perr = float(np.linalg.norm(p_des - d.site_xpos[arm.site]))
            rerr = float(np.linalg.norm(rot_err(R_des, d.site_xmat[arm.site].reshape(3, 3))))
            score = perr + 0.1 * rerr + 0.02 * float(np.linalg.norm(q - q_now))
            if best is None or score < best[3]:
                best = (q, perr, rerr, score)
        return best[0], best[1], best[2]

    def joint_move(self, s, q_goal, grip, name, tol=0.01, max_steps=200):
        """Arm s by joint steps to q_goal (which also becomes its null-space posture)."""
        arm = self.w.arms[s]
        q_goal = [float(x) for x in q_goal]

        def cmd():
            a = q_goal + [float(grip)]
            self.grip[s] = grip
            if s == "l":
                self.step(l=a)
            else:
                self.step(r=a)

        def done():
            return float(np.max(np.abs(self.w.d.qpos[arm.jq] - q_goal))) < tol

        self.run_until(cmd, done, max_steps, name)

    def ik_path(self, s, p_goal, R_goal, rstep=0.05, pstep=0.01):
        """A joint path of arm s from its present pose to (p_goal, R_goal): the straight line and the shortest turn
        cut into small steps, each solved by damped least squares from the previous step's joint angles. Returns
        the joint angles of every step; raises Failed when a step cannot be reached within 5 mm / 0.05 rad."""
        w = self.w
        m = w.m
        arm = w.arms[s]
        d = mujoco.MjData(m)
        d.qpos[:] = w.d.qpos
        p0, R0 = arm.pos(), arm.R()
        v = rotvec(R_goal @ R0.T)
        n = max(1, int(math.ceil(max(np.linalg.norm(v) / rstep, np.linalg.norm(p_goal - p0) / pstep))))
        q = w.d.qpos[arm.jq].copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        path = []
        for i in range(1, n + 1):
            f = i / n
            pt = p0 + f * (p_goal - p0)
            Rt = S.axis_angle_mat(v * f) @ R0
            for _ in range(30):
                d.qpos[arm.jq] = q
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                e = np.concatenate([pt - d.site_xpos[arm.site], rot_err(Rt, d.site_xmat[arm.site].reshape(3, 3))])
                if np.linalg.norm(e[:3]) < 2e-4 and np.linalg.norm(e[3:]) < 2e-3:
                    break
                mujoco.mj_jacSite(m, d, jp, jr, arm.site)
                J = np.vstack([jp, jr])[:, arm.jv]
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
                dq += 0.2 * (np.eye(7) - np.linalg.pinv(J) @ J) @ (0.5 * (arm.q_lo + arm.q_hi) - q) * 0.05
                q = np.clip(q + np.clip(dq, -0.1, 0.1), arm.q_lo + 0.01, arm.q_hi - 0.01)
            d.qpos[arm.jq] = q
            mujoco.mj_kinematics(m, d)
            pe = float(np.linalg.norm(pt - d.site_xpos[arm.site]))
            re = float(np.linalg.norm(rot_err(Rt, d.site_xmat[arm.site].reshape(3, 3))))
            if pe > 0.005 or re > 0.08:
                raise Failed("no path for arm {} at step {}/{} ({:.3f} m, {:.3f} rad)".format(s, i, n, pe, re))
            path.append(q.copy())
        return path

    def turn_hand(self, s, R_goal, grip, name, p_goal=None, max_steps=400):
        """Arm s turns its hand to R_goal (and moves to p_goal, else stays) along a joint path of small Cartesian
        steps (ik_path), followed by joint steps; then settles on the pose in Cartesian steps."""
        arm = self.w.arms[s]
        p = arm.pos() if p_goal is None else np.asarray(p_goal, float)
        try:
            path = self.ik_path(s, p, R_goal)
        except Failed as exc:
            raise Failed("{}: {}".format(name, exc))
        steps = 0
        for i, q in enumerate(path):
            last = i == len(path) - 1
            tol = 0.01 if last else 0.04
            while float(np.max(np.abs(self.w.d.qpos[arm.jq] - q))) > tol:
                a = [float(x) for x in q] + [float(grip)]
                self.grip[s] = grip
                self.step(**{s: a})
                steps += 1
                if steps > max_steps:
                    raise Failed("{} did not finish in {} steps".format(name, max_steps))
        self.events.append((name, self.t))
        self.move(s, p, R_goal, grip, tol=0.004, rtol=0.03, max_steps=60, name=name + ":settle")

    def upright(self, s, e):
        """If elbow e stands on a socket's mouth, arm s tips it over so that it lies flat: the closed hand, pointing
        down, pushes the corner's upper half level, at right angles to the level socket, until the elbow falls about
        its mouth's rim (its level socket stays level, the other one comes down level). The push points toward the
        arm's base unless the elbow would land close to a pipe. Then the hand rises."""
        w = self.w
        c, Re = self.elbow_frame(e)
        downs = [j for j in (0, 1) if Re[2, j] < -0.7]
        if not downs:
            return
        h = Re[:, 1 - downs[0]].copy()
        h[2] = 0.0
        h /= np.linalg.norm(h)
        a = np.cross(Z, h)
        base = np.array([S.BASES[s][0], S.BASES[s][1], c[2]])

        def room(sign):
            """The landed elbow's room from the pipes (m, 2D): the corner about 5 cm along the push."""
            land = (c + 0.05 * sign * a)[:2]
            gap = 9.0
            for p in ("A", "B"):
                pc, pa = self.pipe_axis(p)
                L = w.len[p]
                t = float(np.clip((land - pc[:2]) @ pa[:2] / max(float(pa[:2] @ pa[:2]), 1e-9), -L / 2, L / 2))
                gap = min(gap, float(np.linalg.norm(land - (pc[:2] + t * pa[:2]))) - S.R_PIPE - S.SOCK_L - S.R_OUT)
            return gap

        # toward the arm's base (nearer, easier to take later) unless the landed elbow would crowd a pipe
        signs = sorted((1.0, -1.0), key=lambda sg: float(np.linalg.norm(c + 0.05 * sg * a - base)))
        a = a * (signs[0] if room(signs[0]) > 0.01 else signs[1])
        arm = w.arms[s]
        close = self.close_sign(s, a, 0.0)
        Rp = hand_R(close, -Z)
        start = c - TIP_START * a
        start[2] = c[2] + TIP_ABOVE
        above = start + np.array([0, 0, LIFT])
        z_safe = max(arm.pos()[2], above[2], SAFE_Z)
        self.move(s, np.array([arm.pos()[0], arm.pos()[1], z_safe]), arm.R(), CLOSE, tol=0.02, rtol=0.2,
                  max_steps=80, name=f"up{e}:rise")
        R_down = hand_R(arm.R()[:, 0], -Z)
        self.yaw_path(s, arm.pos(), R_down, np.array([above[0], above[1], z_safe]), self.pick_yaw, CLOSE, tol=0.02,
                      rtol=0.1, name=f"up{e}:over")
        self.pick_yaw = None
        self.move(s, above, Rp, CLOSE, tol=0.01, max_steps=200, name=f"up{e}:above")
        self.move(s, start, Rp, CLOSE, tol=0.003, max_steps=120, vmax=0.01, name=f"up{e}:down")
        end = start + TIP_PUSH * a

        def fallen():
            _c, R = self.elbow_frame(e)
            return abs(np.cross(R[:, 0], R[:, 1])[2]) > 0.9

        def push():
            self.step(**{s: self.servo(s, end, Rp, CLOSE, vmax=0.008)})

        self.run_until(push, lambda: fallen() or np.linalg.norm(arm.pos() - end) < 0.003, 150, f"up{e}:push")
        self.move(s, arm.pos() + np.array([0, 0, 0.10]), Rp, OPEN, tol=0.01, max_steps=80, name=f"up{e}:up")
        for _ in range(15):
            self.step(**{s: self.servo(s, arm.pos(), arm.R(), OPEN)})
        if not fallen():
            raise Failed(f"up{e}: the elbow did not land flat")

    def job(self):
        self.upright("r", 0)
        self.upright("l", 1)
        self.take_e1()
        self.keep("r")
        self.fit_b()
        self.let_go("l")
        self.fit_a()
        self.let_go("l")
        self.fit_e2()
        self.let_go("l")
        self.set_down()

    def take_e1(self):
        """The right hand takes E1 from above by the barrel of the socket that will take A (ELBOW_GRASP from the
        corner, fingers closing along the other socket's axis) and carries it to the station: that socket facing +y,
        the outlet along +x."""
        w = self.w
        e = 0
        self.e1 = e
        c, Re = self.elbow_frame(e)
        # the socket onto A: the one for which (that socket x the other) points down, so the outlet ends along +x
        j_a = 1 if np.cross(Re[:, 1], Re[:, 0])[2] < 0 else 0
        self.e1_ja = j_a
        ua, uo = Re[:, j_a], Re[:, 1 - j_a]
        delta = yaw_angle(ua, np.array([0, 1.0, 0]))
        close = self.close_sign("r", uo, delta)
        self.pick("r", c + ELBOW_GRASP * ua, close, "take_e1", exact=True)
        arm = w.arms["r"]
        c, _Re = self.elbow_frame(e)
        p_goal = arm.pos() - c
        p_goal = S.yaw_mat(delta) @ p_goal + E1_CORNER
        self.yaw_path("r", arm.pos(), arm.R(), p_goal, delta, CLOSE, tol=0.004, rtol=0.03, name="e1:station")

    def fit_b(self):
        """Pipe B into E1's outlet (along +x): the left hand takes B from above 2.2 cm from one end, turns it along x
        and pushes the other end in along -x while the right hand holds E1."""
        w = self.w
        k = 2 * self.e1 + 1 - self.e1_ja
        mouth, o1 = w.socket_pose(k)
        c, a = self.pipe_axis("B")
        L = w.len["B"]
        best = None
        for sign in (1, -1):                         # B's axis out of the end that goes in must become -o1
            ang = yaw_angle(sign * a, -o1)
            if best is None or abs(ang) < abs(best[1]):
                best = (sign, ang)
        sign, delta = best
        g = c - sign * (L / 2 - B_GRASP_FROM_END) * a
        self.pick("l", g, self.close_sign("l", np.cross(Z, a), delta), "take_b", exact=True,
                  above_axis=PIPE_PICK_ABOVE)
        arm = w.arms["l"]
        c, a = self.pipe_axis("B")
        end = c + sign * L / 2 * a
        mouth, o1 = w.socket_pose(k)
        p_goal = mouth + 0.05 * o1 + S.yaw_mat(delta) @ (arm.pos() - end)
        self.yaw_path("l", arm.pos(), arm.R(), p_goal, delta, CLOSE, tol=0.01, rtol=0.05, name="b:bring")
        self.fit("pipe", "l", k, "B", sign, far=0.05, name="b:fit")

    def fit_a(self):
        """Pipe A's banded end into E1's socket facing +y: the left hand takes A from above 3.5 cm from its plain end,
        turns it along y with the banded end toward -y and pushes it in."""
        w = self.w
        k = 2 * self.e1 + self.e1_ja
        c, a = self.pipe_axis("A")
        band = w.episode["band"]
        L = w.len["A"]
        g = c - band * (L / 2 - GRASP_FROM_END) * a
        delta = yaw_angle(band * a, np.array([0, -1.0, 0]))
        self.pick("l", g, self.close_sign("l", np.cross(Z, a), delta), "take_a", exact=True,
                  above_axis=PIPE_PICK_ABOVE)
        arm = w.arms["l"]
        c, a = self.pipe_axis("A")
        delta = yaw_angle(band * a, np.array([0, -1.0, 0]))
        end = c + band * L / 2 * a
        mouth, u = w.socket_pose(k)
        p_goal = mouth + 0.06 * u + S.yaw_mat(delta) @ (arm.pos() - end)
        self.yaw_path("l", arm.pos(), arm.R(), p_goal, delta, CLOSE, tol=0.01, rtol=0.05, name="a:bring")
        self.fit("pipe", "l", k, "A", band, name="a:fit")

    def fit_e2(self):
        """The other elbow onto A's plain end, its outlet parallel to E1's outlet (twist 0) or opposite to it (180).
        The right hand first carries the spool forward, where A's plain end is within the left arm's easy reach; the
        left hand takes E2 from above by its outlet's barrel (fingers closing along the other socket's axis), turns it
        about z so that the other socket faces A's plain end and pushes it on, the hand pointing down throughout."""
        w = self.w
        e = 1 - self.e1
        plain = -w.episode["band"]
        phi = float(w.episode["twist_deg"])
        armR = w.arms["r"]
        c1, _R1 = self.elbow_frame(self.e1)
        self.release("r")
        self.move("r", armR.pos() + (E2_CORNER - c1), armR.R(), CLOSE, tol=0.005, rtol=0.03, max_steps=150,
                  vmax=0.015, name="e2:carry")
        self.keep("r")
        _end, out = w.pipe_end("A", plain)
        o1 = w.socket_pose(2 * self.e1 + 1 - self.e1_ja)[1]
        o2 = S.axis_angle_mat(out * math.radians(phi)) @ o1
        o2 = o2 - (o2 @ out) * out
        o2 /= np.linalg.norm(o2)
        self.o2 = o2
        c, Re = self.elbow_frame(e)
        # turned about z so that socket j_a faces -out, the other socket points along (ua x uo) x (-out)... choose the
        # socket onto A whose outlet then points along o2 (an elbow lying flat keeps its plane horizontal)
        n = np.cross(Re[:, 0], Re[:, 1])                    # the elbow's normal (up or down)
        j_a = None
        for j in (0, 1):
            ua, uo = Re[:, j], Re[:, 1 - j]
            sgn = 1.0 if j == 0 else -1.0                   # ua x uo = sgn * n
            # after a turn about z taking ua to -out: uo' = (sgn * n) x (-out) / |...|
            uo_new = np.cross(sgn * n, -out)
            if uo_new @ o2 > 0:
                j_a = j
        if j_a is None:
            raise Failed("e2: no socket gives the outlet")
        ua, uo = Re[:, j_a], Re[:, 1 - j_a]
        delta = yaw_angle(ua, -out)
        self.pick("l", c + ELBOW_GRASP * uo, self.close_sign("l", ua, delta), "take_e2", exact=True)
        arm = w.arms["l"]
        c, Re = self.elbow_frame(e)
        ua = Re[:, j_a]
        delta = yaw_angle(ua, -out)
        end, out = w.pipe_end("A", plain)
        p_goal = end + 0.08 * out + S.yaw_mat(delta) @ (arm.pos() - (c + S.SOCK_L * ua))
        p_goal[2] = max(p_goal[2], end[2])
        self.yaw_path("l", arm.pos(), arm.R(), p_goal, delta, CLOSE, tol=0.01, rtol=0.05, name="e2:bring")
        self.fit("elbow", "l", 2 * e + j_a, "A", plain, side_want=o2, name="e2:fit")

    def set_down(self):
        """The left hand moves away; the right hand lowers the spool straight down onto the bench and lets go, then
        backs off."""
        w = self.w
        armL = w.arms["l"]
        self.move("l", np.array([0.25, 0.40, S.TABLE_Z + 0.45]), armL.R(), OPEN, tol=0.02, rtol=0.2, max_steps=120,
                  name="l:away")
        self.keep("l", ki=0.0)
        arm = w.arms["r"]
        self.release("r")
        R0 = arm.R()

        def lower():
            self.step(r=self.servo("r", arm.pos() - np.array([0, 0, 0.005]), R0, CLOSE))

        self.run_until(lower, lambda: bool(w._touching()), 200, "down:lower")
        for _ in range(4):
            self.step(r=self.servo("r", arm.pos() - np.array([0, 0, 0.002]), R0, CLOSE))
        self.set_grip("r", OPEN, steps=10)
        self.move("r", arm.pos() + np.array([0, 0, 0.12]), arm.R(), OPEN, tol=0.01, max_steps=80, name="down:up")
        self.keep("r", ki=0.0)
