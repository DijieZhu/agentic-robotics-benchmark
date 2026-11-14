"""Motion skills of the reference program (privileged: it reads the devices' true poses from the cell it runs in).

Everything here drives the cell through the server's own public calls (command, command_rail, set_control, step), so
the reference obeys the same rate limits, grip rule and safety watch as any program; only its eyes are privileged
(Scene.handles_world, the joints' true angles).
"""

import math

import numpy as np

from skidcell.control import ik_solve, mat_to_quat
from skidcell.layout import RAIL
from skidcell.physics import HOME_Q, OPERABLE

APPROACH_OFF = 0.08          # pre-grasp stand-off along the approach axis
STROKE_MAX = math.radians(110.0)
TURN_RATE = math.radians(35.0)  # rad/s of the turning motion
TURN_STIFFNESS = 0.4            # the arm turns valves compliant (the grip's weld is stiff)
PRE_OPEN = {"handwheel": 0.035, "lever": 0.045, "knob": 0.06, "selector": 0.035}
PRESS_DEPTH = 0.0065          # how far past the cap's face the fingertips are driven
FINGERTIP = 0.0095            # the fingertips' end beyond the tcp along the approach axis


class Stop(Exception):
    """The episode ended (done) while a skill was running."""


def rot_about(axis, ang):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


class Skills(object):
    def __init__(self, server, log=None):
        self.S = server
        self.sc = server.scene
        self.arm = server.scene.arm
        self.log = log or (lambda s: None)
        self.debug = False
        self.expected = {}           # tag -> intended joint angle of every operable device
        self._guard = True
        for t in OPERABLE:
            self.expected[t] = float(self.sc.data.qpos[self.sc.jq[t]])

    # ------------------------------------------------------------------ stepping
    def step(self, n=1):
        r = self.S.step(int(n))
        if r.get("done"):
            raise Stop()
        return r

    def t(self):
        return float(self.sc.data.time)

    def wait(self, seconds, poll=50):
        end = self.t() + seconds
        while self.t() < end - 1e-9:
            self.step(min(poll, max(1, int(round((end - self.t()) / 0.01)))))

    # ------------------------------------------------------------------ the carriage and the arm
    def rail_to(self, x, lift, tol=0.002, max_s=30.0):
        self.home()
        self.S.command_rail(x=x, lift=lift)
        end = self.t() + max_s
        while self.t() < end:
            self.step(10)
            if abs(self.sc.rail_x() - self.sc.rail_goal) < tol and abs(self.sc.lift_z() - self.sc.lift_goal) < tol:
                break
        self.step(10)

    def home(self, max_s=6.0):
        q = self.arm.q()
        if np.max(np.abs(q - np.asarray(HOME_Q))) < 0.02 and self.arm.mode == "joint":
            return
        self._retract()
        self.S.set_control("joint", 1.0)
        self.S.command(q=HOME_Q)
        end = self.t() + max_s
        while self.t() < end and np.max(np.abs(self.arm.q() - np.asarray(HOME_Q))) > 0.01:
            self.step(10)

    def _retract(self):
        """Back the tool straight out of the skid to y <= -0.30 before any joint move."""
        p, R = self.arm.tcp_actual()
        if p[1] < -0.29:
            return
        self.S.set_control("cartesian", 1.0)
        tgt = p.copy()
        tgt[1] = -0.31
        self.S.command(tcp_pos=tgt.tolist(), tcp_quat=mat_to_quat(R).tolist())
        self.settle_cart(0.004, 3.0)

    def go(self, pos, R, tol=0.0015, max_s=4.0, stiff=1.0):
        self.S.set_control("cartesian", stiff)
        self.S.command(tcp_pos=list(map(float, pos)), tcp_quat=mat_to_quat(R).tolist())
        return self.settle_cart(tol, max_s)

    def settle_cart(self, tol, max_s):
        end = self.t() + max_s
        while self.t() < end:
            self.step(5)
            res = self.arm.cartesian_residual()
            if res is not None and res[0] < tol and res[1] < 0.02:
                return True
        return False

    def lift_for(self, z):
        return float(np.clip(z - 0.80, 0.05, 0.60))

    # ------------------------------------------------------------------ handles and grips
    def _orientation(self, appr, pinch, roll):
        z = np.asarray(appr, float)
        z = z / np.linalg.norm(z)
        if pinch is None:
            y0 = np.array([0.0, 0.0, 1.0]) - z[2] * z
            y0 = y0 / np.linalg.norm(y0)
            y = rot_about(z, roll) @ y0
        else:
            y = np.asarray(pinch, float)
            y = y - (y @ z) * z
            y = y / np.linalg.norm(y)
            if roll:
                y = -y
        x = np.cross(y, z)
        return np.column_stack([x, y, z])

    def collides(self, q, base, ignore):
        """True when the arm at joint angles q with the carriage at base = (rail x, lift) touches (within 3 mm) anything
        but the device `ignore` (its moving and static parts) and the floor clearance is kept."""
        import mujoco
        m, sc = self.sc.model, self.sc
        s = self.arm.scratch
        s.qpos[:] = self.sc.data.qpos
        s.qpos[sc.rail_qadr] = base[0]
        s.qpos[sc.lift_qadr] = base[1]
        s.qpos[self.arm.qadr] = q
        mujoco.mj_forward(m, s)
        gc = sc.geom_class
        for i in range(s.ncon):
            c = s.contact[i]
            c1, c2 = gc[c.geom1], gc[c.geom2]
            if (c1 in (1, 2)) == (c2 in (1, 2)):
                continue
            og = c.geom2 if c1 in (1, 2) else c.geom1
            if gc[og] not in (5, 7):
                continue
            name = m.geom(og).name or ""
            bname = m.body(m.geom_bodyid[og]).name
            if ignore and (name.startswith(ignore + "/") or bname.startswith(ignore + "/")):
                continue
            if c.dist < 0.003:
                return True
        return False

    def _ik(self, p, R, seeds, iters=400):
        for q0 in seeds:
            q, ep, er, ok = ik_solve(self.arm, p, R, q0=np.clip(q0, self.arm.lower, self.arm.upper), iters=iters)
            if ok and min(np.min(q - self.arm.lower), np.min(self.arm.upper - q)) > 0.05:
                return q
        return None

    def _feasible_stroke(self, centre, axis, R0, p0, sign, need, shift, local=False):
        """How far (radians, up to `need` / STROKE_MAX) the grip pose (p0, R0) can be turned about `axis` through
        `centre` with converged IK at 15 degree steps, warm-started, the carriage moved by `shift` (the IK runs on the
        current carriage position, so the targets are shifted back by it); plus the IK solution at the grip pose."""
        base = np.asarray(HOME_Q, float)
        seeds = [base, base + np.array([0, 0, 0, 0, 0, 0, math.pi / 2]), base + np.array([0, 0, 0, 0, 0, 0, -math.pi / 2]),
                 base + np.array([0, 0, 0, 0, 0, 0, math.pi])]
        if local:
            seeds = [self.arm.q_cmd.copy()]
        q0 = self._ik(p0 - shift, R0, seeds)
        if q0 is None:
            return 0.0, None
        base = (self.sc.rail_x() + shift[0], self.sc.lift_z() + shift[2])
        tag = self._planning_tag
        if self.collides(q0, base, tag):
            return 0.0, None
        qa = self._ik(p0 - shift - APPROACH_OFF * R0[:, 2], R0, [q0], iters=200)
        if qa is None or self.collides(qa, base, tag):
            return 0.0, None
        q = q0
        lim = min(abs(need), STROKE_MAX)
        best = 0.0
        a = 0.0
        stepa = math.radians(15.0)
        while a < lim - 1e-9:
            a = min(lim, a + stepa)
            Rr = rot_about(axis, sign * a)
            q, ep, er, ok = ik_solve(self.arm, centre + Rr @ (p0 - centre) - shift, Rr @ R0, q0=q, iters=200)
            margin = min(np.min(q - self.arm.lower), np.min(self.arm.upper - q))
            if not ok or margin < 0.08 or self.collides(q, base, tag):
                break
            best = a
        return best, q0

    def base_for(self, tag):
        """(rail x, lift) the carriage takes to work on device `tag`."""
        c = np.asarray(self.sc.dev[tag]["centre"], float)
        return float(np.clip(c[0], RAIL["x_lo"], RAIL["x_hi"])), self.lift_for(c[2])

    def plan_grip(self, tag, turn_by=0.0, local=False, exclude=()):
        """The best handle and closing sign of device `tag` for turning it by `turn_by` radians (0: just reach it).
        local=True plans for the carriage where it stands (a re-grip); `exclude`: handles whose grip slipped at once."""
        dev = self.sc.dev[tag]
        self._planning_tag = tag
        hs = self.sc.handles_world(tag)
        centre = np.asarray(self.sc.data.xpos[self.sc.moving_body[tag]], float)
        axis = np.array([0.0, -1.0, 0.0])
        sign = 1.0 if turn_by >= 0 else -1.0
        if local:
            rx, lz = self.sc.rail_x(), self.sc.lift_z()
        else:
            rx, lz = self.base_for(tag)
        shift = np.array([rx - self.sc.rail_x(), 0.0, lz - self.sc.lift_z()])
        need = max(abs(turn_by), math.radians(20))
        want = min(need, STROKE_MAX)
        cands = []
        # spokes / levers first: they turn about a short radius; the rim only when nothing else works
        order = sorted(hs, key=lambda h: (1 if "/rim" in h[0] else 0))
        for name, hp, pinch, appr, width in order:
            if name in exclude:
                continue
            rolls = (0.0, math.pi / 2, -math.pi / 2, math.pi) if pinch is None else (False, True)
            for roll in rolls:
                R0 = self._orientation(appr, pinch, roll)
                stroke, q0 = self._feasible_stroke(centre, axis, R0, hp, sign, need, shift, local=local)
                if q0 is None:
                    continue
                cands.append((stroke, -float(np.abs(q0 - self.arm.q_cmd).sum()), name, hp, R0, width))
            if cands and max(c[0] for c in cands) >= want - 1e-6 and "/rim" not in name:
                break
        if not cands:
            return None
        cands.sort(key=lambda c: (round(c[0], 3), c[1]), reverse=True)
        st, _, name, hp, R0, width = cands[0]
        return {"tag": tag, "handle": name, "pos": hp, "R": R0, "width": width, "stroke": st, "centre": centre, "type": dev["type"],
                "base": (rx, lz)}

    def grip(self, plan, local=False):
        tag = plan["tag"]
        p, R = np.asarray(plan["pos"], float), plan["R"]
        z = R[:, 2]
        pre = PRE_OPEN.get(plan["type"], 0.04)
        if not local:
            self.rail_to(*plan["base"])
            self.S.command(width=pre)
            self.step(30)
            if not self.go(p - APPROACH_OFF * z, R, tol=0.003, max_s=5.0):
                self.log("[ref] %s: pre-grasp not reached" % tag)
        else:
            self.S.command(width=pre)
            self.go(p - 0.05 * z, R, tol=0.003, max_s=3.0)
        self.go(p, R, tol=0.0015, max_s=3.0)
        self.S.command(width=0.0)
        for _ in range(60):
            self.step(1)
            if self.sc.held == tag:
                break
        if self.sc.held != tag:
            self.log("[ref] %s: no grip on %s (width %.3f)" % (tag, plan["handle"], self.arm.width()))
            self.ungrip(back=0.05)
            return False
        self.step(5)
        return True

    def ungrip(self, back=APPROACH_OFF):
        p, Rt = self.arm.tcp_actual()
        self.S.command(width=0.05)
        self.step(20)
        self.go(p - back * Rt[:, 2], Rt, tol=0.005, max_s=2.0)

    def turn_held(self, angle):
        """Turn the held device by `angle` radians about its stem (counter-clockwise as the operator sees it, positive):
        the target pose rotates about the stem in 3 degree increments, each one tracked before the next; a growing
        wrist torque pauses the motion until the arm catches up."""
        tag = self.sc.held
        if tag is None:
            return 0.0
        centre = np.asarray(self.sc.data.xpos[self.sc.moving_body[tag]], float)
        axis = np.array([0.0, -1.0, 0.0])
        p0, R0 = self.arm.tcp_actual()
        q_start = float(self.sc.data.qpos[self.sc.jq[tag]])
        n = max(1, int(math.ceil(abs(angle) / math.radians(3.0))))
        ticks = max(2, int(round(abs(angle) / n / TURN_RATE / 0.01)))
        self.S.set_control("cartesian", TURN_STIFFNESS)
        for k in range(1, n + 1):
            Rr = rot_about(axis, angle * k / n)
            self.S.command(tcp_pos=(centre + Rr @ (p0 - centre)).tolist(), tcp_quat=mat_to_quat(Rr @ R0).tolist())
            self.step(ticks)
            for _ in range(40):
                if self.sc.held != tag:
                    break
                res = self.arm.cartesian_residual()
                ld = self.sc.grip_load()
                if res is not None and res[1] < 0.035 and res[0] < 0.004 and (ld is None or ld[1] < 3.5):
                    break
                self.step(2)
            if self.sc.held != tag:
                break
        self.step(15)
        return float(self.sc.data.qpos[self.sc.jq[tag]]) - q_start

    def set_device(self, tag, target_q, tol=math.radians(2.0), max_cycles=30):
        """Turn device `tag` until its joint angle is within tol of target_q (grip, turn a stroke, let go, re-grip in
        place, repeat)."""
        lo, hi = self.sc.jrange[tag]
        target_q = float(np.clip(target_q, lo + 1e-4, hi - 1e-4))
        local = False
        slipped = set()                  # handles whose grip slipped before the device moved (v0.5: try another one)
        for cyc in range(max_cycles):
            q = float(self.sc.data.qpos[self.sc.jq[tag]])
            need = target_q - q
            if abs(need) <= tol:
                break
            plan = self.plan_grip(tag, need, local=local, exclude=slipped)
            if (plan is None or plan["stroke"] < math.radians(3.0)) and slipped:
                slipped = set()
                plan = self.plan_grip(tag, need, local=local)
            if plan is None or plan["stroke"] < math.radians(3.0):
                if local:
                    self.home()
                    local = False
                    continue
                self.log("[ref] %s: no feasible grip (need %.1f deg)" % (tag, math.degrees(need)))
                self.home()
                return False
            if not self.grip(plan, local=local):
                local = False
                self.home()
                continue
            stroke = math.copysign(min(abs(need), plan["stroke"]), need)
            got = self.turn_held(stroke)
            if abs(got) < math.radians(0.5) and self.sc.held != tag:
                slipped.add(plan["handle"])
            if self.debug:
                self.log("[ref]   %s cycle %d: %s stroke %.1f turned %.1f (need %.1f) held %s" % (
                    tag, cyc, plan["handle"], math.degrees(stroke), math.degrees(got), math.degrees(need), self.sc.held))
            self.ungrip(back=0.05)
            local = True
        self.home()
        q = float(self.sc.data.qpos[self.sc.jq[tag]])
        self.expected[tag] = q if abs(target_q - q) > tol * 2 else target_q
        self.check_bumps(skip=tag)
        return abs(target_q - q) <= tol * 2

    def check_bumps(self, skip=None):
        """Restore any operable device the robot moved by accident (more than 3 % of its travel)."""
        if not self._guard:
            return []
        fixed = []
        for t in OPERABLE:
            if t == skip:
                continue
            lo, hi = self.sc.jrange[t]
            q = float(self.sc.data.qpos[self.sc.jq[t]])
            if abs(q - self.expected[t]) > 0.03 * (hi - lo):
                self.log("[ref] %s was moved by accident (%.3f -> %.3f): restoring" % (t, self.expected[t], q))
                self._guard = False
                try:
                    self.set_device(t, self.expected[t], tol=max(math.radians(2.0), 0.004 * (hi - lo)))
                finally:
                    self._guard = True
                fixed.append(t)
        return fixed

    def set_opening(self, tag, frac):
        lo, hi = self.sc.jrange[tag]
        return self.set_device(tag, lo + float(frac) * (hi - lo), tol=max(math.radians(2.0), 0.004 * (hi - lo)))

    # ------------------------------------------------------------------ the panel
    def press(self, button):
        dev = self.sc.dev[button]
        c = np.asarray(dev["centre"], float)        # the cap's face centre (at rest)
        R = self._orientation([0.0, 1.0, 0.0], [0.0, 0.0, 1.0], False)
        z = R[:, 2]
        self.rail_to(float(np.clip(c[0], RAIL["x_lo"], RAIL["x_hi"])), self.lift_for(c[2]))
        self.S.command(width=0.0)
        self.step(30)
        tip0 = c - (FINGERTIP + 0.03) * z
        self.go(tip0, R, tol=0.002, max_s=5.0)
        self.go(c + (PRESS_DEPTH - FINGERTIP) * z, R, tol=0.0015, max_s=2.0, stiff=0.35)
        self.step(30)
        depth = self.sc.button_depth(button)
        self.go(tip0, R, tol=0.003, max_s=2.0, stiff=0.35)
        self.step(10)
        self.check_bumps()
        return depth > 0.004

    def set_selector(self, position):
        dev = self.sc.dev["HS-101"]
        target = dev["positions"][position]
        return self.set_device("HS-101", target, tol=math.radians(4.0))
