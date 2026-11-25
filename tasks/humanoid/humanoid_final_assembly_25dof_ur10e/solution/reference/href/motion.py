"""Motion through the API's own calls (the cell's controller does the rest): joint moves and straight tcp lines on a
minimum-jerk profile streamed one control tick at a time, inside the configured limits (the joint rate limit and the
Cartesian limits of describe()['control'], the carry limits for a held module), guarded moves that stop on the wrist
force reading, the gripper's width and force, and holds. Every move starts from the controller's COMMANDED state (no
jump) and ends when the arm has arrived (or reports that it did not)."""
import math

import numpy as np

from . import tf
from .errors import ReferenceFailure


class Motion(object):
    def __init__(self, session, kin, log, info, stiffness=1.0):
        self.s, self.kin, self.log = session, kin, log
        c = info["control"]
        self.tick = float(c["tick_s"])
        self.rate = float(c["joint_rate_limit_rad_s"])
        self.v_cart = float(c["cartesian_speed_limit_m_s"])
        self.w_cart = float(c["cartesian_angular_limit_rad_s"])
        cl = c.get("carry_limits") or {}
        self.v_carry = float(cl.get("tcp_speed_m_s", 0.25))
        self.a_carry = float(cl.get("tcp_acc_m_s2", 0.5))
        self.v_free = 0.8 * self.v_cart          # free moves: 80 % of the controller's Cartesian limit
        self.a_free = 1.0
        self.rate_use = 0.8 * self.rate            # 80 % of the joint rate limit
        g = info["robot"]["gripper"]
        self.g_rate = float(g["rate_m_s"])
        self.max_width = float(g["max_width_m"])
        self.mode = None
        self.stiffness = float(stiffness)
        self.ticks_moving = 0

    # ---------------------------------------------------------------- small helpers
    def _mode(self, mode):
        if self.mode != mode:
            self.s.set_control(mode, self.stiffness)
            self.mode = mode

    def state(self):
        return self.s.state()

    def cmd_q(self, st=None):
        st = st or self.s.state()
        return np.array(st["robot"]["cmd_q"], float)

    def tcp_actual(self, st=None):
        st = st or self.s.state()
        return tf.T_of(st["robot"]["tcp_pos"], st["robot"]["tcp_quat"])

    def hold(self, seconds):
        n = max(1, int(round(float(seconds) / self.tick)))
        while n > 0:
            k = min(n, 500)
            self.s.step(k)
            n -= k

    # ---------------------------------------------------------------- joint moves
    def plan_joint(self, q0, q1, v, a, offsets=((0.0, 0.0, 0.0),)):
        """the duration of a minimum-jerk joint move q0 -> q1 that keeps every joint under the used rate and every
        given tcp-frame point under v (m/s) and a (m/s^2): (T, peak tcp-point speed factor dp/ds, joint peak)"""
        dq = np.asarray(q1, float) - np.asarray(q0, float)
        T_j = tf.QUINTIC_V * float(np.max(np.abs(dq))) / self.rate_use
        P = self.kin.path_points(np.asarray(q0, float), np.asarray(q1, float), [np.asarray(o, float) for o in offsets])
        seg = np.linalg.norm(np.diff(P, axis=0), axis=2) * (P.shape[0] - 1)     # |dp/ds| per segment and point
        dpds = float(np.max(seg)) if seg.size else 0.0
        T_v = tf.QUINTIC_V * dpds / v
        T_a = math.sqrt(tf.QUINTIC_A * dpds / a)
        return max(T_j, T_v, T_a, 0.2), dpds, float(np.max(np.abs(dq)))

    def move_joint(self, q1, v=None, a=None, offsets=((0.0, 0.0, 0.0),), tol=0.02, settle_max_s=1.5, what="joint move"):
        v = self.v_free if v is None else float(v)
        a = self.a_free if a is None else float(a)
        st = self.s.state()
        q0 = self.cmd_q(st)
        q1 = np.asarray(q1, float)
        T, dpds, dqmax = self.plan_joint(q0, q1, v, a, offsets)
        n = int(math.ceil(T / self.tick))
        self._mode("joint")
        for i in range(1, n + 1):
            q = q0 + tf.quintic(i / float(n)) * (q1 - q0)
            self.s.command(q=[float(x) for x in q])
            self.s.step(1)
        self.ticks_moving += n
        arrived = self._settle_joint(q1, tol, settle_max_s)
        self.log.write("move", what=what, move="joint", T_s=round(n * self.tick, 3), ticks=n, dq_max_rad=dqmax,
                       peak_point_speed_m_s=round(tf.QUINTIC_V * dpds / (n * self.tick), 4), arrived=arrived)
        if not arrived["ok"]:
            raise ReferenceFailure(self.s._phase(), "%s: the arm did not arrive" % what, arrived)
        return arrived

    def _settle_joint(self, q1, tol, settle_max_s):
        """the arm has arrived when it is at rest (every joint slower than 0.02 rad/s) with the commanded configuration
        at the target; the actual joints then differ from it by the static error of the load, reported, and that error
        must stay under `tol` (a larger one means something holds the arm back)"""
        n = int(round(settle_max_s / self.tick))
        err = None
        for i in range(n + 1):
            st = self.s.state()
            q = np.array(st["robot"]["q"], float)
            dq = np.array(st["robot"]["dq"], float)
            cmd = np.array(st["robot"]["cmd_q"], float)
            err = float(np.max(np.abs(q - q1)))
            if float(np.max(np.abs(cmd - q1))) < 1e-6 and float(np.max(np.abs(dq))) < 0.02:
                return {"ok": err <= tol, "err_rad": err, "settle_ticks": i}
            self.s.step(1)
        return {"ok": False, "err_rad": err, "settle_ticks": n, "why": "not at rest"}

    # ---------------------------------------------------------------- straight tcp lines
    def move_linear(self, T1, v=None, a=None, tol_p=3e-3, tol_r=0.02, settle_max_s=1.5, what="line", T0=None):
        """the tcp from its commanded pose (or T0) to T1 on a straight line (orientation by the shortest rotation),
        minimum-jerk; then waits until the ACTUAL tcp is within tol_p / tol_r of T1"""
        v = self.v_free if v is None else float(v)
        a = self.a_free if a is None else float(a)
        st = self.s.state()
        if T0 is None:
            T0 = self.kin.fk_T(self.cmd_q(st))
        p0, R0 = T0[:3, 3], T0[:3, :3]
        p1, R1 = np.asarray(T1[:3, 3], float), np.asarray(T1[:3, :3], float)
        L = float(np.linalg.norm(p1 - p0))
        rv = tf.rotvec(R0.T @ R1)
        th = float(np.linalg.norm(rv))
        T = max(tf.QUINTIC_V * L / v, math.sqrt(tf.QUINTIC_A * L / a), tf.QUINTIC_V * th / (0.8 * self.w_cart), 0.1)
        n = int(math.ceil(T / self.tick))
        self._mode("cartesian")
        for i in range(1, n + 1):
            s = tf.quintic(i / float(n))
            p = p0 + s * (p1 - p0)
            R = R0 @ tf.exp_rot(s * rv)
            self.s.command(tcp_pos=[float(x) for x in p], tcp_quat=[float(x) for x in tf.mat_to_quat(R)])
            self.s.step(1)
        self.ticks_moving += n
        arrived = self._settle_tcp(T1, tol_p, tol_r, settle_max_s)
        self.log.write("move", what=what, move="line", T_s=round(n * self.tick, 3), ticks=n, length_m=L, turn_rad=th,
                       arrived=arrived)
        if not arrived["ok"]:
            raise ReferenceFailure(self.s._phase(), "%s: the tcp did not arrive" % what, arrived)
        return arrived

    def _settle_tcp(self, T1, tol_p, tol_r, settle_max_s):
        """the tcp has arrived when the arm is at rest (every joint slower than 0.01 rad/s) with the COMMANDED
        configuration's tcp at the target (0.05 mm, 0.5 mrad: the controller's own convergence); the actual tcp then
        differs from it by the static sag of the load, reported, and that difference must stay under tol_p / tol_r"""
        n = int(round(settle_max_s / self.tick))
        out = None
        for i in range(n + 1):
            st = self.s.state()
            Ta = self.tcp_actual(st)
            Tc = self.kin.fk_T(np.array(st["robot"]["cmd_q"], float))
            ep = float(np.linalg.norm(Ta[:3, 3] - T1[:3, 3]))
            er = tf.angle(Ta[:3, :3].T @ T1[:3, :3])
            cp = float(np.linalg.norm(Tc[:3, 3] - T1[:3, 3]))
            cr = tf.angle(Tc[:3, :3].T @ T1[:3, :3])
            sp = float(np.max(np.abs(np.array(st["robot"]["dq"], float))))
            out = {"err_m": ep, "err_rad": er, "cmd_err_m": cp, "cmd_err_rad": cr, "settle_ticks": i}
            if cp <= 5e-5 and cr <= 5e-4 and sp < 0.01:
                out["ok"] = ep <= tol_p and er <= tol_r
                return out
            self.s.step(1)
        out["ok"] = False
        out["why"] = "not at rest"
        return out

    def command_tcp(self, T):
        self._mode("cartesian")
        self.s.command(tcp_pos=[float(x) for x in T[:3, 3]], tcp_quat=[float(x) for x in tf.mat_to_quat(T[:3, :3])])

    # ---------------------------------------------------------------- a guarded move on the wrist force
    def wrench_mean(self, n=10):
        F = []
        for _ in range(n):
            st = self.s.state()
            F.append(np.array(st["robot"]["wrench"][:3], float))
            self.s.step(1)
        return np.mean(F, axis=0)

    def guarded_line(self, direction, dist_max, speed, f_change_N, what="guarded line", stop_when=None, check_every=5):
        """the tcp moves along `direction` (world, unit) at `speed` from its commanded pose until the wrist force has
        changed by f_change_N from its reading at rest before the move (the stop) or dist_max is travelled; the tcp is
        then held where it was commanded. `stop_when` (optional) is a callable checked every `check_every` ticks;
        True stops the line there ("stop": "condition"). Returns {"stop": "force" | "distance" | "condition",
        "travel_m", "dF_N", "ticks"}"""
        F0 = self.wrench_mean(10)
        st = self.s.state()
        T0 = self.kin.fk_T(self.cmd_q(st))
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        self._mode("cartesian")
        n_max = int(math.ceil(dist_max / (speed * self.tick)))
        out = {"stop": "distance", "travel_m": dist_max, "dF_N": None, "ticks": n_max, "F0_N": [float(x) for x in F0]}
        for i in range(1, n_max + 1):
            trav = min(dist_max, speed * self.tick * i)
            T = T0.copy()
            T[:3, 3] = T0[:3, 3] + d * trav
            self.command_tcp(T)
            self.s.step(1)
            st = self.s.state()
            dF = float(np.linalg.norm(np.array(st["robot"]["wrench"][:3], float) - F0))
            if dF >= f_change_N:
                out.update({"stop": "force", "travel_m": trav, "dF_N": dF, "ticks": i})
                break
            out["dF_N"] = dF
            if stop_when is not None and i % int(check_every) == 0 and stop_when():
                out.update({"stop": "condition", "travel_m": trav, "ticks": i})
                break
        self.ticks_moving += out["ticks"]
        self.log.write("move", what=what, move="guarded", **out)
        return out

    # ---------------------------------------------------------------- a held part set on its true pose, a servoed line
    @staticmethod
    def part_correction(T_cmd, T_part_now, T_part_want, F, axis, axial=False, cap_m=None, cap_rad=None, gain=1.0, rot_free=None):
        """the commanded tcp moved by the world correction that takes a held part from its true pose to its wanted
        pose: the rotation about the part's interface point (T_part @ F's origin) and the translation of that point;
        the component along `axis` (world, unit) only when `axial`; the rotation's component about `rot_free` (world,
        unit; None: none) left out; scaled by `gain`, then the translation capped at cap_m and the angle at cap_rad
        (None: not capped). Returns (the new commanded tcp pose, {"m": translation, "rad": angle} applied)."""
        c_now = (np.asarray(T_part_now, float) @ np.asarray(F, float))[:3, 3]
        c_want = (np.asarray(T_part_want, float) @ np.asarray(F, float))[:3, 3]
        t = c_want - c_now
        ax = np.asarray(axis, float) / np.linalg.norm(axis)
        if not axial:
            t = t - float(t @ ax) * ax
        rv = tf.rotvec(np.asarray(T_part_want, float)[:3, :3] @ np.asarray(T_part_now, float)[:3, :3].T)
        if rot_free is not None:
            u = np.asarray(rot_free, float) / np.linalg.norm(rot_free)
            rv = rv - float(rv @ u) * u
        t, rv = t * float(gain), rv * float(gain)
        n_t, n_r = float(np.linalg.norm(t)), float(np.linalg.norm(rv))
        if cap_m is not None and n_t > cap_m:
            t = t * (cap_m / n_t)
        if cap_rad is not None and n_r > cap_rad:
            rv = rv * (cap_rad / n_r)
        R = tf.exp_rot(rv)
        T = np.array(T_cmd, float)
        out = np.eye(4)
        out[:3, :3] = R @ T[:3, :3]
        out[:3, 3] = R @ (T[:3, 3] - c_now) + c_now + t
        return out, {"m": float(np.linalg.norm(t)), "rad": float(np.linalg.norm(rv))}

    def servo_line(self, direction, dist, speed, part_now, part_want, F, T0=None, every=2, gain=0.5, cap_m=0.0005,
                   cap_rad=math.radians(0.1), axial=False, F0=None, guard_df=None, stop_when=None, stop_every=5,
                   rot_free=None, what="servo line"):
        """the tcp along `direction` (world, unit) by `dist` at a constant `speed` from T0 (default: the commanded
        pose), one command per control tick (no settle at the end); every `every` ticks the held part's true pose
        part_now() is compared with part_want(s) (s = the travel commanded so far) and `gain` x the world correction
        (part_correction: lateral and angular, axial only when asked, the turn about rot_free left out; capped per
        correction) is added to the streamed
        pose (the servo rule: every 2 ticks, 0.5 x the error, at most 0.5 mm / 0.1 deg). Optional guard: the wrist force's
        change from F0 (|F - F0|, N) reaching guard_df stops the line where it is commanded; optional stop_when()
        (checked every `stop_every` ticks) stops it. Returns {"stop": "distance" | "force" | "condition", "ticks",
        "travel_m", "dF_N_max", "n_corr", "corr_m_max", "corr_rad_max", "T_end"}"""
        d = np.asarray(direction, float) / np.linalg.norm(direction)
        if T0 is None:
            T0 = self.kin.fk_T(self.cmd_q())
        Tb = np.array(T0, float)
        n = max(1, int(math.ceil(float(dist) / (float(speed) * self.tick) - 1e-9)))
        self._mode("cartesian")
        out = {"stop": "distance", "ticks": n, "travel_m": float(dist), "dF_N_max": None, "n_corr": 0, "corr_m_max": 0.0,
               "corr_rad_max": 0.0}
        T = Tb
        for i in range(1, n + 1):
            s = min(float(dist), float(speed) * self.tick * i)
            if (i - 1) % int(every) == 0:
                s_prev = min(float(dist), float(speed) * self.tick * (i - 1))
                Tb, size = self.part_correction(Tb, part_now(), part_want(s_prev), F, d, axial=axial, cap_m=cap_m,
                                                cap_rad=cap_rad, gain=gain, rot_free=rot_free)
                out["n_corr"] += 1
                out["corr_m_max"] = max(out["corr_m_max"], size["m"])
                out["corr_rad_max"] = max(out["corr_rad_max"], size["rad"])
            T = Tb.copy()
            T[:3, 3] = Tb[:3, 3] + d * s
            self.command_tcp(T)
            self.s.step(1)
            if guard_df is not None:
                st = self.s.state()
                dF = float(np.linalg.norm(np.array(st["robot"]["wrench"][:3], float) - np.asarray(F0, float)))
                out["dF_N_max"] = dF if out["dF_N_max"] is None else max(out["dF_N_max"], dF)
                if dF >= guard_df:
                    out.update({"stop": "force", "ticks": i, "travel_m": s})
                    break
            if stop_when is not None and i % int(stop_every) == 0 and stop_when():
                out.update({"stop": "condition", "ticks": i, "travel_m": s})
                break
        self.ticks_moving += out["ticks"]
        out["T_end"] = T
        self.log.write("move", what=what, move="servo", **{k: v for k, v in out.items() if k != "T_end"})
        return out

    # ---------------------------------------------------------------- the gripper
    def gripper(self, width, force_N, what="gripper", max_s=2.0, still_m=5e-5, still_ticks=5):
        """command the width and the force register (N per pad); step until the measured width is still (no more than
        still_m over still_ticks ticks) and the commanded width has been reached by the ramp"""
        self.s.command(width=float(width), force=float(force_N))
        ramp = abs(float(width) - float(self.s.state()["robot"]["gripper_width"])) / self.g_rate
        n_min = int(math.ceil(ramp / self.tick)) + 2
        hist = []
        n = int(round(max_s / self.tick))
        for i in range(n):
            self.s.step(1)
            st = self.s.state()
            hist.append(float(st["robot"]["gripper_width"]))
            if i >= n_min and len(hist) > still_ticks and max(hist[-still_ticks:]) - min(hist[-still_ticks:]) <= still_m:
                break
        out = {"width_cmd_m": float(width), "width_m": hist[-1], "force_N": float(force_N), "ticks": len(hist)}
        self.log.write("gripper", what=what, **out)
        return out
