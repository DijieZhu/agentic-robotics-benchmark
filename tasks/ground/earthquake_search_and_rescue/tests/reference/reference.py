"""The reference program of earthquake_search_and_rescue (privileged: it runs inside the cell through reference_run and
reads the site's global map: the planned route, the obstacles on it, the survivor's position, the robot's true pose).

It drives the route the site was planned with: through the facade spill, over the step (front flippers raised), across
the floor gap (flippers straight out), through the narrow breach (narrow gauge), under the dropped slabs (mast down,
arm stowed), lifts the beam out of the doorway it blocks and lays it beside the path, then at the survivor looks at
them with the mast camera and the thermal camera, reports their position, places the survival kit next to their head
or hand, turns round and drives the route back to the deployment point.
"""

import math

import numpy as np

V_CRUISE = 0.32
FOLD = math.radians(170.0)
DRIVE_FLIP = math.radians(10.0)
GAP_FLIP = math.radians(-8.0)
V_SLOW = 0.16
STOW = [0.0, math.radians(184.0), math.radians(-184.0), math.radians(-55.0), 0.0]


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Ref(object):
    def __init__(self, srv):
        self.s = srv
        self.log = []
        self.t_limit = srv.cfg["sim_cap_s"] - 5.0

    # ---------------------------------------------------------------- low level
    def note(self, what):
        x, y, yaw = self.pose()
        self.log.append("t=%.1f %s at (%.2f, %.2f, %.0f deg)" % (self.s.d.time, what, x, y, math.degrees(yaw)))

    def pose(self):
        p, q = self.s.robot.pose()
        w, x, y, z = q
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return float(p[0]), float(p[1]), yaw

    def step(self, n=10):
        r = self.s.step(n)
        return not r.get("done")

    def wait(self, sec):
        n = int(sec / 0.01)
        while n > 0:
            k = min(n, 100)
            if not self.step(k):
                return False
            n -= k
        return True

    def cfg_drive(self):
        s = self.s
        s.set_mast(0.0)
        s.set_head(0.0, 0.0)
        s.set_arm(STOW)

    def settle_joints(self, sec=1.0):
        self.s.drive(0.0, 0.0)
        return self.wait(sec)

    def turn_to(self, yaw_goal, tol=0.06, tmax=25.0):
        s = self.s
        x, y, yaw = self.pose()
        if abs(_wrap(yaw_goal - yaw)) < tol:
            return True
        # turn short: flippers folded over the tracks, then back to what they were
        prev = (s.robot.cmd["flip_F"], s.robot.cmd["flip_B"])
        s.drive(0.0, 0.0)
        s.set_flippers(FOLD, FOLD)
        self.wait(2.8)
        ok = self._turn(yaw_goal, tol, tmax)
        s.set_flippers(prev[0], prev[1])
        self.wait(2.8)
        return ok

    def _turn(self, yaw_goal, tol=0.06, tmax=25.0):
        s = self.s
        t0 = s.d.time
        while s.d.time - t0 < tmax:
            x, y, yaw = self.pose()
            e = _wrap(yaw_goal - yaw)
            if abs(e) < tol:
                break
            s.drive(0.0, float(np.clip(2.0 * e, -0.9, 0.9)))
            if not self.step(5):
                return False
        s.drive(0.0, 0.0)
        return self.step(10)

    def go_to(self, gx, gy, v=V_CRUISE, tol=0.12, tmax=90.0, reverse=False, turn_first=True):
        """Pure-pursuit-like drive to (gx, gy): turn in place when the heading error is large."""
        s = self.s
        t0 = s.d.time
        x, y, yaw = self.pose()
        hd = math.atan2(gy - y, gx - x) + (math.pi if reverse else 0.0)
        if turn_first and abs(_wrap(hd - yaw)) > 0.35:
            if not self.turn_to(hd):
                return False
        stuck_t, last = s.d.time, (x, y)
        while s.d.time - t0 < tmax:
            x, y, yaw = self.pose()
            dx, dy = gx - x, gy - y
            dist = math.hypot(dx, dy)
            if dist < tol:
                break
            hd = math.atan2(dy, dx) + (math.pi if reverse else 0.0)
            e = _wrap(hd - yaw)
            if abs(e) > 0.6:
                s.drive(0.0, float(np.clip(2.0 * e, -0.9, 0.9)))
            else:
                vv = min(v, 0.6 * dist + 0.05) * (-1.0 if reverse else 1.0)
                s.drive(vv * max(0.2, math.cos(e)), float(np.clip(2.2 * e, -0.8, 0.8)))
            if not self.step(5):
                return False
            if self.tilted(math.radians(30.0)):
                self.log.append("tilt guard near (%.2f, %.2f)" % (x, y))
                s.drive(-0.15 if not reverse else 0.15, 0.0)
                self.wait(1.5)
                s.drive(0.0, 0.0)
                self.wait(1.0)
            if s.d.time - stuck_t > 6.0:
                if math.hypot(x - last[0], y - last[1]) < 0.05:
                    self.log.append("stuck near (%.2f, %.2f)" % (x, y))
                    # back off a little and retry
                    s.drive(-0.15 if not reverse else 0.15, 0.0)
                    self.wait(1.2)
                stuck_t, last = s.d.time, (x, y)
        s.drive(0.0, 0.0)
        return self.step(5)

    def go_line(self, p0, n, s_goal, v=V_SLOW, tol=0.03, tmax=60.0, tilt_lim=math.radians(24.0)):
        """Facing along unit direction n, drive forward or backward along the line through p0 until the along-line
        coordinate reaches s_goal, steering out the lateral error and the heading error."""
        s = self.s
        p0 = np.asarray(p0, float)
        n = np.asarray(n, float) / np.linalg.norm(n)
        lat_dir = np.array([-n[1], n[0]])
        hd = math.atan2(n[1], n[0])
        t0 = s.d.time
        x, y, yaw = self.pose()
        if abs(_wrap(hd - yaw)) > 0.25:
            if not self.turn_to(hd, tol=0.05):
                return False
        stuck_t, last = s.d.time, (x, y)
        while s.d.time - t0 < tmax:
            x, y, yaw = self.pose()
            rel = np.array([x, y]) - p0
            sc = float(rel @ n)
            el = float(rel @ lat_dir)
            ds = s_goal - sc
            if abs(ds) < tol:
                break
            fwd = 1.0 if ds > 0 else -1.0
            he = _wrap(hd - yaw)
            w = 2.0 * he - fwd * 2.5 * el
            vv = fwd * min(v, 0.8 * abs(ds) + 0.03)
            s.drive(vv, float(np.clip(w, -0.7, 0.7)))
            if not self.step(5):
                return False
            if self.tilted(tilt_lim):
                self.log.append("tilt guard on the line at s=%.2f" % sc)
                s.drive(-0.6 * vv, 0.0)
                self.wait(1.5)
                s.drive(0.0, 0.0)
                self.wait(1.0)
            if s.d.time - stuck_t > 8.0:
                if math.hypot(x - last[0], y - last[1]) < 0.04:
                    self.log.append("line stuck at s=%.2f (goal %.2f), lat %.2f" % (sc, s_goal, el))
                    break
                stuck_t, last = s.d.time, (x, y)
        s.drive(0.0, 0.0)
        return self.step(5)

    def tilted(self, lim=math.radians(24.0)):
        w, x, y, z = self.s.robot.pose()[1]
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        return abs(roll) > lim or abs(self.pitch()) > lim

    def pitch(self):
        w, x, y, z = self.s.robot.pose()[1]
        return math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))     # + = nose down

    def climb_step(self, p0, n, s_goal, wp=None):
        """Climb a step along the line: front flippers raised steeply until their tips are over the step's edge, then
        lower them onto the top (they lever the nose up) while the rear flippers press the tail up; level on top."""
        s = self.s
        p0 = np.asarray(p0, float)
        n = np.asarray(n, float)
        lat_dir = np.array([-n[1], n[0]])
        hd = math.atan2(n[1], n[0])
        s_edge = None
        if wp is not None and wp.get("p") is not None and wp.get("step_near") is not None:
            s_edge = float((np.array(wp["p"]) - p0) @ n) - float(wp["step_near"])
        s.set_flippers(math.radians(65.0), 0.0)
        self.settle_joints(2.0)
        phase = 0
        t0 = s.d.time
        t_phase = s.d.time
        last_sc, t_move = None, s.d.time
        max_up = 0.0
        while s.d.time - t0 < 90.0:
            x, y, yaw = self.pose()
            rel = np.array([x, y]) - p0
            sc, el = float(rel @ n), float(rel @ lat_dir)
            if sc >= s_goal:
                break
            if last_sc is None or sc > last_sc + 0.02:
                last_sc, t_move = sc, s.d.time
            stalled = s.d.time - t_move > 1.2
            pt = -self.pitch()                       # + = nose up
            max_up = max(max_up, pt)
            if phase == 0 and ((s_edge is not None and sc >= s_edge - 0.42) or stalled):
                phase, t_phase = 1, s.d.time
                s.set_flippers(math.radians(20.0), math.radians(-40.0))
            elif phase == 1 and max_up > math.radians(10.0) and pt < math.radians(4.0) and (s_edge is None or sc > s_edge + 0.1):
                phase, t_phase = 2, s.d.time
                s.set_flippers(math.radians(5.0), math.radians(-5.0))
            elif phase == 1 and s.d.time - t_phase > 10.0 and stalled:
                # no progress: back off and try again
                self.log.append("step retry at s=%.2f" % sc)
                s.drive(-0.15, 0.0)
                self.wait(2.0)
                s.set_flippers(math.radians(70.0), 0.0)
                self.settle_joints(2.0)
                phase, t_phase, max_up = 0, s.d.time, 0.0
                last_sc, t_move = None, s.d.time
                continue
            if pt > math.radians(42.0):
                s.drive(-0.15, 0.0)
                self.wait(1.5)
                s.set_flippers(math.radians(70.0), 0.0)
                phase, t_phase, max_up = 0, s.d.time, 0.0
                continue
            he = _wrap(hd - yaw)
            s.drive(0.16, float(np.clip(2.0 * he - 2.5 * el, -0.5, 0.5)))
            if not self.step(5):
                return False
        s.drive(0.0, 0.0)
        s.set_flippers(DRIVE_FLIP, DRIVE_FLIP)
        return self.wait(1.0)

    def cross_gap(self, p0, n, wp, s_end, back=False):
        """Stop short of the floor gap, square up, flippers straight out, cross slowly and straight."""
        s = self.s
        p = np.array(wp["p"])
        n = np.asarray(n, float)
        # the gap's near edge along the line (forward: d0 past the opening; back: the far edge)
        if not back:
            s_edge = float((p - p0) @ n) + float(wp["gap_d0"])
        else:
            s_edge = float((p - p0) @ n) - float(wp["gap_d0"]) - float(wp["gap_w"])
        s.set_flippers(0.0, 0.0)
        if not self.go_line(p0, n, s_edge - 0.75, v=0.12, tol=0.04):
            return False
        x, y, yaw = self.pose()
        self.turn_to(math.atan2(n[1], n[0]), tol=0.02)
        # flippers a little below straight: their belts carry the robot across, the drums stay clear of the edges
        s.set_flippers(GAP_FLIP, GAP_FLIP)
        self.settle_joints(2.5)
        return self.go_line(p0, n, s_end, v=0.12, tol=0.08, tmax=90.0, tilt_lim=math.radians(80.0))

    # ---------------------------------------------------------------- obstacles
    def through(self, wp, back=False):
        """Pass an opening from its approach point to its far point."""
        s = self.s
        kinds = wp.get("kinds", [])
        crawl = wp["cfg"] == "crawl"
        s.set_gauge(0.30)                     # every opening narrow: the most room either side
        if "gap" in kinds:
            s.set_flippers(0.0, 0.0)
        elif "step" in kinds and not back:
            s.set_flippers(math.radians(38.0), math.radians(-12.0))
        elif "step" in kinds and back:
            s.set_flippers(math.radians(-8.0), math.radians(25.0))
        elif "spill" in kinds:
            s.set_flippers(math.radians(25.0), 0.0)
        else:
            s.set_flippers(DRIVE_FLIP, DRIVE_FLIP)
        self.settle_joints(1.5)

    def clear_beam(self, wp):
        """Lift the beam out of the opening's notches and drop it beside the robot."""
        s = self.s
        m, d = s.m, s.d
        name = wp.get("beam")
        if not name:
            return True
        import mujoco
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        bp = d.xpos[bid].copy()
        n = np.array(wp["n"])
        # flippers folded back over the tracks so the robot's nose can come close; arm base 0.42 m from the beam
        s.set_flippers(math.radians(170.0), math.radians(10.0))
        self.settle_joints(3.0)
        if not self.go_line(bp[:2], n, -(0.42 + 0.165), v=V_SLOW, tol=0.02):
            return False
        if not self.turn_to(math.atan2(n[1], n[0]), tol=0.02):
            return False
        self.settle_joints(0.5)
        bp = d.xpos[bid].copy()

        def to_ch(pw):
            x, y, yaw = self.pose()
            c, sn = math.cos(yaw), math.sin(yaw)
            dx, dy = pw[0] - x, pw[1] - y
            zc = d.xpos[s.chassis][2]
            return [c * dx + sn * dy, -sn * dx + c * dy, pw[2] - zc]

        pitch = -math.radians(70.0)
        u = np.array([math.cos(pitch), 0.0, math.sin(pitch)])      # tool axis in the arm's plane (forward-down)
        s.gripper(0.15)
        tgt = np.array(to_ch(bp))
        pre = tgt - u * 0.12
        r = s.arm_to(list(pre), pitch=pitch, roll=math.pi / 2)
        if not r.get("ok"):
            self.log.append("beam pre-grasp unreachable %s" % (np.round(pre, 3).tolist(),))
            return False
        if not self.wait(5.0):
            return False
        r = s.arm_to(list(tgt + np.array([0, 0, 0.004])), pitch=pitch, roll=math.pi / 2)
        if not self.wait(2.0):
            return False
        s.gripper(0.0)
        if not self.wait(1.5):
            return False
        self.note("beam grasp: held=%s" % (s.held == bid))
        s.arm_to(list(tgt + np.array([0.0, 0, 0.30])), pitch=pitch, roll=math.pi / 2, speed=0.10)
        if not self.wait(4.0):
            return False
        # back out with the beam held high, then swing it to the left and set it down
        x, y, yaw = self.pose()
        if not self.go_line(np.array([x, y]), n, -0.8, v=0.12, tol=0.04):
            return False
        ang = math.radians(70.0)
        side = [0.165 + 0.50 * math.cos(ang), 0.50 * math.sin(ang), 0.28]
        s.arm_to(side, pitch=-math.radians(40.0), roll=math.pi / 2, speed=0.12)
        if not self.wait(6.0):
            return False
        s.arm_to([side[0], side[1], -0.02], pitch=-math.radians(40.0), roll=math.pi / 2, speed=0.10)
        if not self.wait(4.0):
            return False
        s.gripper(0.15)
        if not self.wait(1.5):
            return False
        s.arm_to([side[0] * 0.7 + 0.05, side[1] * 0.7, 0.30], pitch=-math.radians(40.0), roll=0.0)
        if not self.wait(3.0):
            return False
        s.set_arm(STOW)
        s.set_flippers(DRIVE_FLIP, DRIVE_FLIP)
        ok = self.wait(6.0)
        self.note("beam dropped at (%.2f, %.2f)" % tuple(d.xpos[bid][:2]))
        return ok

    def wait_arm(self, tmax=12.0, tol=0.03):
        s = self.s
        t0 = s.d.time
        while s.d.time - t0 < tmax:
            q = np.array(s.robot.arm_q())
            c = np.array(s.robot.cmd["arm"])
            if s._arm_lin is None and np.max(np.abs(q - c)) < tol:
                break
            if not self.step(10):
                return False
        return self.step(20)

    def to_chassis(self, pw):
        d = self.s.d
        x, y, yaw = self.pose()
        c, sn = math.cos(yaw), math.sin(yaw)
        dx, dy = pw[0] - x, pw[1] - y
        return np.array([c * dx + sn * dy, -sn * dx + c * dy, pw[2] - d.xpos[self.s.chassis][2]])

    def kit_and_report(self):
        """Look at the survivor, report, place the kit near their head or hand."""
        import mujoco
        from rubblecell.robot import arm_ik
        s = self.s
        d = s.d
        tor = s.surv_torso
        x, y, yaw = self.pose()
        self.turn_to(math.atan2(tor[1] - y, tor[0] - x), tol=0.04)
        s.set_head(0.0, math.radians(-12.0))
        self.wait(1.0)
        s.get_camera("mast", depth=True)
        s.get_thermal()
        xm, ym, _ = s.odom.to_map(float(tor[0]), float(tor[1]), 0.0)
        s.report_survivor(xm, ym, 0.0)
        self.note("reported")
        # where the kit goes: 0.38 m from the head or a hand, clear of the body and of the pocket's furniture
        x, y, yaw = self.pose()
        heads_hands = np.vstack([s.surv_heads, s.surv_hands])[:, :2]
        blockers = []
        for st in s.sc.site.statics:
            if st["name"].startswith(("wardrobe", "legslab", "legstack", "cupboard")):
                # the piece's own reach (half its diagonal) and a hand's breadth, never less than 0.55 m
                hx, hy = st.get("half", st["size"])[:2]
                blockers.append((np.array(st["pos"][:2]), max(0.55, math.hypot(hx, hy) + 0.15)))
        best = None
        for H in heads_hands:
            for a in np.radians(np.arange(0, 360, 15)):
                P = H + 0.38 * np.array([math.cos(a), math.sin(a)])
                if np.linalg.norm(P - tor[:2]) < 0.42:
                    continue
                if min(np.linalg.norm(P - Q) for Q in heads_hands) < 0.22:
                    continue
                if any(np.linalg.norm(P - B) < rb for B, rb in blockers):
                    continue
                dist = float(np.linalg.norm(P - np.array([x, y])))
                if best is None or dist < best[0]:
                    best = (dist, P)
        if best is None:
            self.log.append("no kit spot found")
            return True
        P = best[1]
        u = P - np.array([x, y])
        L = float(np.linalg.norm(u))
        u = u / max(L, 1e-6)
        # stand with the arm base 0.60 m from the spot, facing it (folded flippers: a short robot)
        s.set_flippers(math.radians(170.0), math.radians(170.0))
        self.settle_joints(3.0)
        self.go_line(np.array([x, y]), u, L - (0.60 + 0.165), v=0.10, tol=0.03)
        self.turn_to(math.atan2(u[1], u[0]), tol=0.03)
        # the kit from its cradle (joint move above it, straight down, grip, straight up)
        kp = np.array([-0.275, 0.055, 0.065 + 0.036])
        s.gripper(0.15)
        q = arm_ik(list(kp + np.array([0, 0, 0.16])), -math.pi / 2, q_ref=[0, 0, 0, 0, 0.0])
        if q is None:
            self.log.append("kit pre-grasp unreachable")
            return True
        s.set_arm(q)
        self.wait_arm()
        s.arm_to(list(kp + np.array([0, 0, 0.004])), pitch=-math.pi / 2, roll=0.0, speed=0.08)
        self.wait_arm()
        s.gripper(0.0)
        self.wait(1.5)
        self.note("kit grasp: held=%s" % (s.held == s.kit_body))
        s.arm_to(list(kp + np.array([0, 0, 0.14])), pitch=-math.pi / 2, roll=0.0, speed=0.08)
        self.wait_arm()
        # forward, low, to the spot; down; open
        Pc = self.to_chassis([P[0], P[1], 0.0])
        floor_c = -d.xpos[s.chassis][2]
        pitch = -math.radians(50.0)
        q = arm_ik([Pc[0], Pc[1], floor_c + 0.16], pitch, q_ref=[0, 0, 0, 0, 0.0])
        if q is None:
            self.log.append("kit spot unreachable from the stand")
            return True
        s.set_arm(q)
        self.wait_arm(tmax=15.0)
        s.arm_to([Pc[0], Pc[1], floor_c + 0.055], pitch=pitch, roll=0.0, speed=0.06)
        self.wait_arm()
        s.gripper(0.15)
        self.wait(1.5)
        s.arm_to([Pc[0] - 0.08, Pc[1], floor_c + 0.22], pitch=pitch, roll=0.0, speed=0.08)
        self.wait_arm()
        s.set_arm(STOW)
        self.wait_arm(tmax=15.0)
        kp_w = d.xpos[s.kit_body]
        self.note("kit placed at (%.2f, %.2f, %.2f)" % tuple(kp_w))
        # back away from the survivor before turning round
        x, y, yaw = self.pose()
        self.go_line(np.array([x, y]), u, -0.5, v=0.10, tol=0.04)
        return True


def run(srv):
    R = Ref(srv)
    s = srv
    wps = [dict(w) for w in s.sc.site.route]
    R.cfg_drive()
    s.set_gauge(0.40)
    s.set_flippers(DRIVE_FLIP, DRIVE_FLIP)
    R.settle_joints(1.0)
    path_done = []
    ok = True
    i = 1
    while i < len(wps) and ok:
        w = wps[i]
        crawl = w["cfg"] == "crawl"
        if w["note"] == "approach":
            R.note("approach opening %s %s" % (w.get("opening"), w.get("kinds")))
            s.set_mast(0.0)
            n = np.array(w["n"])
            pa = np.array([w["x"], w["y"]])
            ok = R.go_to(*(pa - n * 0.6), v=V_CRUISE if not crawl else V_SLOW + 0.06, tol=0.15)
            ok = ok and R.go_line(pa, n, 0.0, v=V_SLOW, tol=0.05)
            R.through(w)
            nxt = wps[i + 1]
            if "debris" in nxt.get("kinds", []):
                ok = ok and R.clear_beam(nxt)
            n = np.array(nxt["n"])
            p0 = np.array([w["x"], w["y"]])
            s_end = float((np.array([nxt["x"], nxt["y"]]) - p0) @ n)
            if "step" in nxt.get("kinds", []) and ok:
                ok = R.climb_step(p0, n, s_end, nxt)
            elif "gap" in nxt.get("kinds", []) and ok:
                ok = R.cross_gap(p0, n, nxt, s_end)
            else:
                lim = math.radians(38.0) if ("spill" in nxt.get("kinds", []) or "gap" in nxt.get("kinds", [])) else math.radians(24.0)
                ok = ok and R.go_line(p0, n, s_end, v=V_SLOW, tol=0.08, tmax=90.0, tilt_lim=lim)
            s.set_flippers(DRIVE_FLIP, DRIVE_FLIP)
            if not nxt.get("narrow"):
                s.set_gauge(0.30 if crawl else 0.40)
            path_done.append(i)
            path_done.append(i + 1)
            i += 2
            continue
        ok = R.go_to(w["x"], w["y"], v=V_CRUISE if not crawl else V_SLOW + 0.06)
        path_done.append(i)
        i += 1
    if ok:
        ok = R.kit_and_report()
    # return: the same waypoints backwards
    back = [wps[j] for j in path_done][::-1]
    if ok:
        j = 0
        # turn round first
        x, y, yaw = R.pose()
        if back:
            R.turn_to(math.atan2(back[0]["y"] - y, back[0]["x"] - x))
        while j < len(back) and ok:
            w = back[j]
            if w["note"] == "through" and j + 1 < len(back) and back[j + 1]["note"] == "approach":
                ok = R.go_to(w["x"], w["y"], v=V_SLOW + 0.06)
                R.through(w, back=True)
                n = -np.array(w["n"])
                p0 = np.array([w["x"], w["y"]])
                s_end = float((np.array([back[j + 1]["x"], back[j + 1]["y"]]) - p0) @ n)
                if "gap" in w.get("kinds", []) and ok:
                    # crossing back: from the far side, the gap starts (p - p0).n - d0 - w ... measured along -n
                    pp = np.array(w["p"])
                    s_edge = float((pp - p0) @ n) - float(w["gap_d0"]) - float(w["gap_w"])
                    s.set_flippers(0.0, 0.0)
                    ok = R.go_line(p0, n, s_edge - 0.75, v=0.12, tol=0.04)
                    R.turn_to(math.atan2(n[1], n[0]), tol=0.02)
                    s.set_flippers(GAP_FLIP, GAP_FLIP)
                    R.settle_joints(2.5)
                    ok = ok and R.go_line(p0, n, s_end, v=0.12, tol=0.08, tmax=90.0, tilt_lim=math.radians(80.0))
                elif "step" in w.get("kinds", []) and ok:
                    ok = R.climb_step(p0, n, s_end, w)
                else:
                    ok = ok and R.go_line(p0, n, s_end, v=V_SLOW, tol=0.08, tmax=90.0, tilt_lim=math.radians(38.0))
                s.set_flippers(DRIVE_FLIP, DRIVE_FLIP)
                s.set_gauge(0.40)
                j += 2
                continue
            ok = R.go_to(w["x"], w["y"], v=V_CRUISE)
            j += 1
        st = s.plan["start"]
        R.go_to(st["x"], st["y"], v=V_CRUISE)
    s.drive(0.0, 0.0)
    R.wait(1.0)
    s.finish()
    return {"log": R.log, "score": s.score()}
