"""Hotpot serving: the privileged reference controller.

It reads the simulator's truth (the poses of the pot, the ladle, the dishes, the bowls and the pieces, each piece's
doneness, each vessel's broth and the stream in flight) and acts only through the environment's action
(`SoupEnv.step`: both arms' 7-number operational-space steps, and the right arm's 8-number joint steps):
1. the right hand turns the knob to full power and takes the ladle from the pot by the top of its handle (from the
   side), lifts it and holds it beside the pot;
2. the left hand takes each dish the order needs by its tab, the slowest kind to cook first, carries it over the pot
   (each at its own spot, TIP_SPOTS) and rolls it until its pieces have slid into the broth, then puts it back;
3. the right hand slides the dish under a good piece (cooked, with at least SERVE_MARGIN s of its window left) of a
   kind the bowls still need, the one with the least window left first (the scoop whose path also collects what one
   bowl can take whole is better), and lifts it; when what the dish carries is all cooked and fits one bowl (of each
   kind no more than the bowl still needs), it tips all of it into that bowl; if not, over the pot it pitches the open
   front down a little at a time until the front-most piece has rolled off (with whatever rolls along), and again;
   while nothing is good yet but a wanted piece is still cooking, it waits; past control step SERVE_T_MAX it stops.
   Dish yaws 60-120 deg first (wider ones wind the right wrist toward its limits); when a right-arm joint comes within
   its margin of a limit, joint steps take the arm back to the posture it lifted the ladle in;
4. it lays the ladle on the counter, to the right of the bowls, joint-steps the right arm back to its start posture
   and turns the knob off;
5. both hands take the pot by its tabs and lift it; per bowl, the pot is carried so its front rim is over the bowl
   and both hands roll it about the line between the tabs, the tilt servoed on the flow until the broth delivered and
   in flight, plus the flow times a lead, reaches the order; then the pot is righted;
6. the pot goes back on its ring, both hands let go and move up.

In process (a check of the reference alone): `MUJOCO_GL=osmesa python3 -I reference.py --unit 1 --seed 7`. Over the
grading wire it runs in `wire_world.py`, on a shadow of the world set from the grader's token `truth`.
"""

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import episode as E  # noqa: E402
import soup_env  # noqa: E402
import soup_sim as S  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])
GRASP_S = 0.22                    # m along the ladle's handle from the neck: where the right hand holds it
SAFE_DISH_Z = S.TABLE_Z + 0.20    # the dish's height for moves over the counter
LEAD_S = 0.20                     # s: the pour is stopped when delivered + in flight + flow * LEAD_S reaches the order
FLOW_FAST = 70e-6                 # m^3/s, the flow aimed at while far from the order
FLOW_SLOW = 25e-6                 # m^3/s, the flow aimed at in its last SLOW_ML
SLOW_ML = 40.0
CARRY_PITCH = -0.15               # rad about the dish's y axis: the open front up while it carries pieces
MISSES_MAX = 2                    # scoops that came up empty at one piece before it is left alone
SCOOP_DZ = 0.0015                 # m: the dish's underside above the pot's floor while it slides (under a sunk piece)
REACH_SHORT = 0.035               # m: a piece this far ahead of the dish's centre at the scoop's end still rides in it
RELEASE_PITCH_POT = 1.0           # rad: the same over the pot
MARGIN_MIN = 0.35                 # rad: a right-arm joint this close to its limit: back to the carry posture
RELEASE_PITCH_MAX = 1.0           # rad: the steepest the front goes down to let a piece roll off
LADLE_DOWN = np.array([0.60, -0.34])   # where the ladle is laid down (the dish's centre), its front toward +y
SERVE_T_MAX = 5600                # control steps: past this the pieces stop and the broth is served
URGENT_S = 120.0                  # s: a piece with less of its window left than this is scooped first
SERVE_MARGIN = 20.0               # s: a piece is served only with this much of its window left
FLIP = np.diag([1.0, -1.0, -1.0])      # a hand pointing down, its x along world x (the pads close along x)
TIP_SPOTS = ((-0.05, -0.02), (0.05, -0.02), (0.0, -0.045), (0.0, 0.005))   # m: where the dishes are held over
                                  # the pot (their centres, from its axis), the first one tipped first               # m: where a dish is held over the pot (its centre, +y of the pot's axis)
TIP_MAX = 1.3                     # rad: how far a dish is rolled to let its pieces slide off
READY_R = np.array([0.40, -0.25, 1.10])  # the right hand's pose between the ladle and the pot


def rotvec(R):
    c = max(-1.0, min(1.0, (np.trace(R) - 1) / 2))
    th = math.acos(c)
    if th < 1e-9:
        return np.zeros(3)
    if th > math.pi - 1e-6:
        w, V = np.linalg.eigh((R + R.T) / 2)
        return V[:, np.argmax(w)] * th
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2 * math.sin(th)) * th


def rz(a):
    return S.axis_angle_mat([0.0, 0.0, a])


def dish_R(yaw, pitch=0.0):
    """The dish's orientation: its front toward `yaw`, pitched by `pitch` (positive: the front down)."""
    return rz(yaw) @ S.axis_angle_mat([0.0, pitch, 0.0])


class Done(Exception):
    """The episode ended (its step limit or an unstable simulation)."""


class Reference:
    def __init__(self, ep):
        self.ep = ep
        self.env = ep.env
        self.w = ep.env.w
        self.L, self.R = self.w.arms["l"], self.w.arms["r"]
        self.goal = {"l": None, "r": None}      # per arm: (pos, R, grip, speed), ("joint", q, grip) or None
        self.log = []
        self.phase = "start"
        self.phase_log = []
        self.T_hl = None
        self.failed = []
        m = self.w.m
        self.r_range = np.array([m.jnt_range[m.joint(f"r_joint{k}").id] for k in range(1, 8)])
        self.q_carry = None
        self.margin_min = MARGIN_MIN

    # --- one control step ---------------------------------------------------------------------
    def tick(self):
        act = {"cams": 0}
        for s, a in (("l", self.L), ("r", self.R)):
            g = self.goal[s]
            if g is None:
                act[s] = [0.0] * 6 + [-1.0]
                continue
            if isinstance(g[0], str):                  # a joint step toward a posture: ("joint", q, grip)
                act[s] = list(map(float, g[1])) + [float(g[2])]
                continue
            p, Rg, grip, spd = g
            dp = np.asarray(p, float) - a.pos()
            n = np.linalg.norm(dp)
            if n > spd:
                dp *= spd / n
            dr = rotvec(Rg @ a.R().T)
            n = np.linalg.norm(dr)
            if n > 0.15:
                dr *= 0.15 / n
            act[s] = list(map(float, dp)) + list(map(float, dr)) + [float(grip)]
        if not self.phase_log or self.phase_log[-1][0] != self.phase:
            self.phase_log.append((self.phase, self.w.t))
        _obs, done, _ = self.ep.step(act)
        if done:
            raise Done()

    def wait(self, n):
        for _ in range(n):
            self.tick()

    def servo(self, side, p, Rg, grip, speed=0.02, tol=0.003, max_steps=200, other=None):
        """Move `side`'s hand to (p, Rg); `other` (pos, R, grip, speed) for the other arm meanwhile, or keep."""
        a = self.w.arms[side]
        self.goal[side] = (np.asarray(p, float), Rg, grip, speed)
        for _ in range(max_steps):
            self.tick()
            if np.linalg.norm(np.asarray(p) - a.pos()) < tol and np.linalg.norm(rotvec(Rg @ a.R().T)) < 0.03:
                return True
        return False

    # --- the ladle ----------------------------------------------------------------------------
    def ladle_pose(self):
        d = self.w.d
        return d.xpos[self.w.ladle].copy(), d.xmat[self.w.ladle].reshape(3, 3).copy()

    def handle_point(self, s):
        p, Rl = self.ladle_pose()
        neck = np.array([-(S.LADLE_DISH_R + 0.004), 0.0, S.LADLE_RIM_H])
        u = np.array([-math.cos(S.LADLE_BETA), 0.0, math.sin(S.LADLE_BETA)])
        return p + Rl @ (neck + u * s)

    def hand_for_dish(self, dpos, dR):
        Rh = dR @ self.T_hl[1].T
        return dpos - Rh @ self.T_hl[0], Rh

    def regrip_frame(self):
        """The hand -> ladle transform as it is now (the ladle shifts a little in the fingers)."""
        p, Rl = self.ladle_pose()
        Rh, ph = self.R.R(), self.R.pos()
        self.T_hl = (Rh.T @ (p - ph), Rh.T @ Rl)

    def dish_go(self, dpos, dR, speed=0.02, tol=0.004, max_steps=200, grip=1.0):
        self.regrip_frame()
        p, Rh = self.hand_for_dish(dpos, dR)
        return self.servo("r", p, Rh, grip, speed, tol, max_steps)

    def local_of(self, i):
        p, Rl = self.ladle_pose()
        return [round(float(v) * 1000) for v in Rl.T @ (self.w.d.xpos[self.w.pieces[i]] - p)]

    def carried(self):
        p, Rl = self.ladle_pose()
        out = []
        for i, b in enumerate(self.w.pieces):
            x = Rl.T @ (self.w.d.xpos[b] - p)
            if math.hypot(x[0], x[1]) < S.LADLE_DISH_R + 0.012 and -0.012 < x[2] < 0.07:
                out.append(i)
        return out

    def take_ladle(self):
        self.phase = "take_ladle"
        p, Rl = self.ladle_pose()
        hp = self.handle_point(GRASP_S)
        u = Rl @ np.array([-math.cos(S.LADLE_BETA), 0.0, math.sin(S.LADLE_BETA)])
        a = -np.array([u[0], u[1], 0.0])
        a /= np.linalg.norm(a)                       # the approach: toward the pot, level
        x = Rl[:, 1] - (Rl[:, 1] @ a) * a
        x[2] = 0.0
        x /= np.linalg.norm(x)                       # the pads close across the handle's width
        Rg = np.column_stack([x, np.cross(a, x), a])
        self.servo("r", hp - 0.10 * a + 0.03 * UP, Rg, -1.0, max_steps=150)
        self.servo("r", hp - 0.10 * a, Rg, -1.0, tol=0.003)
        self.T_hl = None
        self.servo("r", hp, Rg, -1.0, speed=0.01, tol=0.002)
        self.goal["r"] = (hp, Rg, 1.0, 0.01)
        self.wait(8)
        p, Rl = self.ladle_pose()
        yaw = math.atan2(Rl[1, 0], Rl[0, 0])
        self.dish_go(p + np.array([0, 0, 0.03]), rz(yaw), speed=0.01, max_steps=40)
        self.dish_go(np.array([p[0], p[1], SAFE_DISH_Z]), rz(yaw), speed=0.015, max_steps=150)
        self.q_carry = self.w.d.qpos[self.R.jq].copy()     # the posture it carries the ladle in
        self.margin_min = min(MARGIN_MIN, self.r_margin() - 0.1)

    def r_margin(self):
        q = self.w.d.qpos[self.R.jq]
        return float(np.min(np.minimum(q - self.r_range[:, 0], self.r_range[:, 1] - q)))

    def recover_r(self, steps=60):
        """Joint steps back to the posture the ladle was lifted in (a shed or a scoop can fold the arm toward a
        joint limit, where it no longer follows the dish's targets)."""
        self.goal["r"] = ("joint", self.q_carry.copy(), 1.0)
        for _ in range(steps):
            self.tick()
            if np.max(np.abs(self.w.d.qpos[self.R.jq] - self.q_carry)) < 0.03:
                break
        self.goal["r"] = (self.R.pos(), self.R.R(), 1.0, 0.01)
        self.wait(3)
        self.log.append({"recover_r": round(self.r_margin(), 2), "t": self.w.t})

    # --- what the order still needs -----------------------------------------------------------
    def remaining(self):
        """Per bowl, per kind: how many more pieces it needs (negative: too many)."""
        w, order = self.w, self.env.episode["order"]
        places = [w.piece_place(i) for i in range(len(w.pieces))]
        return {nm: {kd: order[nm]["items"][kd] - sum(1 for i, p in enumerate(places) if p == nm and w.kinds[i] == kd)
                     for kd in S.KINDS} for nm in order}

    def wanted(self):
        """Per kind: how many more the bowls need in all."""
        rem = self.remaining()
        return {kd: sum(max(r[kd], 0) for r in rem.values()) for kd in S.KINDS}

    def good(self, i):
        """Cooked, with at least SERVE_MARGIN s of its window left (or out of the pot already, not overcooked)."""
        lq = self.w.liquid
        return bool(lq.cooked(i) and lq.over[i] < lq.window[i] - SERVE_MARGIN)

    # --- scooping -----------------------------------------------------------------------------
    def plan_scoop(self, want, pot_p, failed=()):
        for lo, hi, clear_t in ((70, 110, 0.085), (60, 120, 0.085), (70, 110, 0.075), (60, 120, 0.075),
                                (-20, 200, 0.075)):
            best = self._plan(want, pot_p, failed, lo, hi, clear_t)
            if best is not None:
                return best
        return self._plan(want, pot_p, (), 70, 110, 0.075)

    def _plan(self, want, pot_p, failed, lo, hi, clear_t):
        """The best scoop at a good piece of a wanted kind: the dish slides from `start` to `end` along its front; the
        pieces in its path ride along, and the more of them are wanted the better (the rest go back to the pot)."""
        w = self.w

        def wall_t(c, h):
            ch = c @ h
            return -ch + math.sqrt(max(ch * ch - c @ c + S.POT_R ** 2, 0.0))

        in_pot = [i for i in range(len(w.pieces)) if w.piece_place(i) == "pot"]
        xy = {i: (w.d.xpos[w.pieces[i]] - pot_p)[:2] for i in in_pot}
        rem = self.remaining()
        lim = S.POT_R - S.LADLE_DISH_R - 0.005
        best = None
        for i in in_pot:
            if (want.get(w.kinds[i], 0) <= 0 or not self.good(i)
                    or sum(1 for f in failed if f[0] == i) >= MISSES_MAX):
                continue
            c = xy[i]
            for ang in np.radians(np.arange(lo, hi + 1, 10)):
                if failed.count((i, int(round(math.degrees(ang))))) >= 1:
                    continue
                dvec = np.array([math.cos(ang), math.sin(ang)])
                cs = c - 0.065 * dvec
                if np.linalg.norm(cs) > S.POT_R - S.LADLE_DISH_R - 0.004:
                    continue
                t = wall_t(cs, -dvec)
                if t < clear_t:
                    continue
                ce = c + 0.01 * dvec
                for _ in range(30):
                    if np.linalg.norm(ce) <= lim:
                        break
                    ce = ce - 0.004 * dvec
                if (ce - c) @ dvec < -REACH_SHORT:           # the dish could not get far enough under it
                    continue
                seg = ce - cs
                L2 = seg @ seg
                group = [i]
                clear = 1.0
                for j in in_pot:
                    if j == i:
                        continue
                    clear = min(clear, np.linalg.norm(xy[j] - cs))
                    s_ = min(max((xy[j] - cs) @ seg / L2, 0.0), 1.0)
                    if np.linalg.norm(cs + s_ * seg - xy[j]) < S.LADLE_DISH_R:     # in the dish's path
                        group.append(j)
                okg = [j for j in group if self.good(j)]
                n = {kd: sum(1 for j in okg if w.kinds[j] == kd) for kd in S.KINDS}
                fit = max((sum(min(n[kd], max(r[kd], 0)) for kd in n) for r in rem.values()), default=0)
                whole = len(okg) == len(group) and any(all(n[kd] <= r[kd] for kd in n) for r in rem.values())
                # the whole load into one bowl is best; every piece that has to be tipped back costs a shed
                left_s = float(w.liquid.window[i] - w.liquid.over[i])    # the sooner it would be overcooked, the sooner
                score = (min(clear, 0.06) + 0.1 * t - 0.02 * abs(ang - math.pi / 2)
                         + (0.08 if whole else 0.0) + 0.02 * fit - 0.04 * (len(group) - fit)
                         + 0.08 * max(0.0, 1.0 - left_s / URGENT_S))
                if best is None or score > best[0]:
                    best = (score, i, w.d.xpos[w.pieces[i]].copy(), dvec, ang, ce)
        return best

    def scoop(self, want):
        """One scoop at a good piece of a wanted kind: returns the pieces the dish carries after lifting."""
        w = self.w
        pot_p = w.d.xpos[w.pot].copy()
        floor_z = pot_p[2] + S.POT_FLOOR
        if self.r_margin() < self.margin_min:
            self.recover_r()
        plan = self.plan_scoop(want, pot_p, self.failed)
        if plan is None:
            return None
        _sc, i, m, dvec, ang, endc = plan
        Rd = rz(ang)
        o3 = np.array([dvec[0], dvec[1], 0.0])
        start = np.array([m[0], m[1], floor_z + SCOOP_DZ]) - 0.065 * o3
        end = np.array([pot_p[0] + endc[0], pot_p[1] + endc[1], floor_z + SCOOP_DZ])
        self.dish_go(np.array([start[0], start[1], SAFE_DISH_Z]), Rd, max_steps=150)
        self.dish_go(start, Rd, speed=0.016, tol=0.004, max_steps=120)
        self.dish_go(end, Rd, speed=0.008, tol=0.004, max_steps=80)
        Rc = dish_R(ang, CARRY_PITCH)
        self.dish_go(end + np.array([0, 0, 0.01]), Rc, speed=0.004, max_steps=30)
        self.dish_go(end + np.array([0, 0, 0.04]), Rc, speed=0.006, max_steps=30)
        self.dish_go(np.array([end[0], end[1], SAFE_DISH_Z]), Rc, speed=0.016, max_steps=120)
        self.wait(5)
        c = self.carried()
        if not c:
            self.failed.append((int(i), int(round(math.degrees(ang)))))
        self.log.append({"scoop": int(i), "ang": round(math.degrees(ang)), "carried": len(c), "t": w.t})
        return c

    # --- handing the pieces out --------------------------------------------------------------
    def release_spot(self, dest):
        """Where the dish waits to let a piece roll off its front into `dest` (a bowl's name, or "pot"), and its
        yaw (the front toward +y over a bowl: the hand stays on the right arm's side)."""
        w = self.w
        if dest == "pot":
            pot_p = w.d.xpos[w.pot]
            return np.array([pot_p[0], pot_p[1], SAFE_DISH_Z - 0.06]), math.pi / 2
        bp = w.d.xpos[w.bowls[dest]].copy()
        yaw = math.pi / 2
        fx = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        rim_z = bp[2] + S.BOWL_FLOOR + S.BOWL_DEPTH
        return np.array([bp[0], bp[1], rim_z + 0.025]) - 0.05 * fx, yaw

    def release(self, dest, done):
        """Over `dest`, pitch the dish's open front down a little at a time until `done(carried)` holds, then pitch
        back at once. Returns whether it held."""
        centre, yaw = self.release_spot(dest)
        Rc = dish_R(yaw, CARRY_PITCH)
        self.dish_go(np.array([centre[0], centre[1], SAFE_DISH_Z]), Rc, speed=0.016, max_steps=200)
        self.dish_go(centre, Rc, speed=0.008, max_steps=60)
        th = CARRY_PITCH
        ok = False
        top = RELEASE_PITCH_POT if dest == "pot" else RELEASE_PITCH_MAX
        while th < top:
            th += 0.012
            self.dish_go(centre, dish_R(yaw, th), speed=0.01, max_steps=1)
            if done(self.carried()):
                ok = True
                break
        if not ok:                                      # it sticks: a little shake at the steepest pitch
            for k in range(6):
                self.dish_go(centre + np.array([0, 0, 0.01 if k % 2 else -0.01]), dish_R(yaw, th), speed=0.02,
                             max_steps=2)
                if done(self.carried()):
                    ok = True
                    break
        self.dish_go(centre, dish_R(yaw, CARRY_PITCH - 0.15), speed=0.01, max_steps=4)
        self.dish_go(centre, Rc, speed=0.01, max_steps=10)
        self.wait(10)
        if self.r_margin() < self.margin_min:
            self.recover_r()
        return ok

    def fits(self, c):
        """The bowl all of `c` can go into (of each kind no more than it still needs; the one that needs the most),
        or None."""
        w = self.w
        rem = self.remaining()
        if not all(self.w.liquid.cooked(i) and not self.w.liquid.overcooked(i) for i in c):
            return None
        n = {kd: sum(1 for i in c if w.kinds[i] == kd) for kd in S.KINDS}
        ok = [nm for nm in rem if all(n[kd] <= rem[nm][kd] for kd in n)]
        return max(ok, key=lambda nm: sum(max(v, 0) for v in rem[nm].values())) if ok else None

    def hand_out(self):
        """What the dish carries goes into a bowl whole, when it fits one; if not, the front-most piece is tipped
        back into the pot (those that roll with it go back too), and again."""
        w = self.w
        for _k in range(10):
            c = self.carried()
            if not c:
                return
            if w.t > SERVE_T_MAX:                    # out of time: all of it back into the pot
                self.release("pot", lambda c: not c)
                return
            nm = self.fits(c)
            if nm is not None:
                ok = self.release(nm, lambda c: not c)
                self.log.append({"dumped": nm, "n": len(c), "ok": ok, "landed": [w.piece_place(i) for i in c],
                                 "kinds": [w.kinds[i] for i in c], "t": w.t})
                if not ok:
                    self.release("pot", lambda c: not c)
                continue
            p, Rl = self.ladle_pose()
            f = max(c, key=lambda i: (Rl.T @ (w.d.xpos[w.pieces[i]] - p))[0])
            ok = self.release("pot", lambda c, f=f: f not in c)
            self.log.append({"shed": int(f), "kind": w.kinds[f], "ok": ok, "carried": len(self.carried()), "t": w.t})

    def serve_pieces(self):
        """Scoop and hand out good pieces of the kinds the bowls still need; while none is ready but a wanted one is
        still cooking in the pot, wait with the dish above the pot."""
        w = self.w
        self.phase = "pieces"
        for _attempt in range(80):
            if w.t > SERVE_T_MAX:
                self.log.append({"serve_time_up": True, "t": w.t})
                return
            want = self.wanted()
            if not any(v > 0 for v in want.values()):
                return
            c = self.scoop(want)
            if c is None:
                cooking = [i for i in range(len(w.pieces)) if w.piece_place(i) == "pot"
                           and want.get(w.kinds[i], 0) > 0 and not w.liquid.cooked(i)]
                if cooking:
                    self.wait(20)
                    continue
                self.log.append({"no_scoop": dict(want), "t": w.t})
                return
            if not c:
                continue
            self.hand_out()

    def park_ladle(self):
        """Hold the ladle up beside the pot, on the right, while the left hand tips the dishes."""
        self.phase = "park_ladle"
        pot = self.w.d.xpos[self.w.pot]
        self.dish_go(np.array([pot[0] - 0.05, pot[1] - 0.22, SAFE_DISH_Z + 0.05]), dish_R(math.pi / 2, CARRY_PITCH),
                     speed=0.016, max_steps=150)

    def dish_order(self):
        """The dishes whose kind the order lists, the slowest to cook first."""
        ep = self.env.episode
        need = {kd: sum(o["items"][kd] for o in ep["order"].values()) for kd in S.KINDS}
        sc = self.w.unit["piece_scale"]

        def cook_s(kd):
            p = S.PIECE[kd]
            return p["tau"] * sc * sc * math.log((S.BOIL - S.ROOM) / (S.BOIL - p["t_cook"]))
        return sorted((k for k, kd in enumerate(ep["dish_of"]) if need[kd] > 0), key=lambda k: -cook_s(ep["dish_of"][k]))

    def put_ladle_down(self):
        self.phase = "ladle_down"
        Rd = dish_R(math.pi / 2)
        tgt = np.array([LADLE_DOWN[0], LADLE_DOWN[1], SAFE_DISH_Z])
        self.dish_go(tgt, Rd, max_steps=150)
        self.dish_go(np.array([tgt[0], tgt[1], S.TABLE_Z + 0.012]), Rd, speed=0.01, max_steps=120)
        hp, Rh = self.R.pos(), self.R.R()
        self.goal["r"] = (hp, Rh, -1.0, 0.01)
        self.wait(8)
        a = Rh[:, 2]
        self.servo("r", hp - 0.08 * a, Rh, -1.0, max_steps=60)
        self.servo("r", hp - 0.08 * a + 0.15 * UP, Rh, -1.0, max_steps=60)
        self.servo("r", READY_R, np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]), -1.0, max_steps=150)

    def unwind_r(self, steps=60):
        """The right arm back to its start posture by joint steps (many scoops can leave its wrist near a limit)."""
        self.phase = "unwind_r"
        self.goal["r"] = ("joint", S.HOME_Q.copy(), -1.0)
        for _ in range(steps):
            self.tick()
            if np.max(np.abs(self.w.d.qpos[self.R.jq] - S.HOME_Q)) < 0.02:
                break
        self.goal["r"] = (self.R.pos(), self.R.R(), -1.0, 0.01)
        self.wait(5)

    # --- the knob and the dishes ---------------------------------------------------------------
    def turn_knob(self, angle, side="r"):
        """Pinch the knob's fin from above and turn it to `angle` (rad), then let go and move up."""
        w = self.w
        kp = np.array([S.KNOB_XY[0], S.KNOB_XY[1], S.TABLE_Z + 0.006 + S.KNOB_H + S.KNOB_FIN[2]])
        th = w.knob()
        self.phase = "knob"
        self.servo(side, kp + 0.08 * UP, rz(th) @ FLIP, -1.0, max_steps=200)
        self.servo(side, kp, rz(th) @ FLIP, -1.0, speed=0.01, tol=0.003, max_steps=100)
        self.goal[side] = (kp, rz(th) @ FLIP, 1.0, 0.01)
        self.wait(8)
        while abs(angle - th) > 1e-6:
            th += max(-0.08, min(0.08, angle - th))
            self.goal[side] = (kp, rz(th) @ FLIP, 1.0, 0.01)
            self.tick()
        self.wait(10)
        self.goal[side] = (kp, rz(th) @ FLIP, -1.0, 0.01)
        self.wait(8)
        self.servo(side, kp + 0.08 * UP, rz(th) @ FLIP, -1.0, max_steps=80)
        self.log.append({"knob": round(w.knob(), 3), "want": round(angle, 3), "t": w.t})

    def tip_dish(self, k, spot=(0.0, -0.02), side="l"):
        """Take dish k by its tab, carry it over the pot's +y half and roll it so its pieces slide off its +y edge
        into the broth; then put it back and let go."""
        w, d = self.w, self.w.d
        self.phase = f"dish{k}"
        db = w.dishes[k]
        p0 = d.xpos[db].copy()
        tab = p0 + np.array([-(S.DISH_R + S.DISH_WALL + S.DISH_TAB[0] + 0.004), 0.0,
                             S.DISH_TAB_Z])
        Rg = rz(math.pi / 2) @ FLIP                    # the pads close across the tab's width (world y)
        self.servo(side, tab + 0.08 * UP, Rg, -1.0, max_steps=200)
        self.servo(side, tab, Rg, -1.0, speed=0.01, tol=0.003, max_steps=100)
        self.goal[side] = (tab, Rg, 1.0, 0.01)
        self.wait(8)
        g = tab - p0                                   # the grasp relative to the dish's origin
        pot = d.xpos[w.pot]
        over = np.array([pot[0] + spot[0], pot[1] + spot[1], pot[2] + S.POT_FLOOR + S.POT_DEPTH + 0.05])
        self.servo(side, tab + 0.12 * UP, Rg, 1.0, speed=0.01, max_steps=120)
        self.servo(side, over + g + np.array([0, 0, 0.12 - 0.05]), Rg, 1.0, speed=0.015, max_steps=200)
        self.servo(side, over + g, Rg, 1.0, speed=0.01, max_steps=80)
        mine = [i for i in range(len(w.pieces)) if w.piece_place(i) == "dish"
                and np.linalg.norm((d.xpos[w.pieces[i]] - d.xpos[db])[:2]) < S.DISH_R + 0.01]
        phi = 0.0
        while phi < TIP_MAX:
            phi += 0.04
            self.goal[side] = (over + g, S.axis_angle_mat([-phi, 0.0, 0.0]) @ Rg, 1.0, 0.01)
            self.tick()
        for _ in range(40):
            self.tick()
            if all(w.piece_place(i) != "dish" for i in mine):
                break
        self.wait(10)
        while phi > 0.0:
            phi = max(0.0, phi - 0.08)
            self.goal[side] = (over + g, S.axis_angle_mat([-phi, 0.0, 0.0]) @ Rg, 1.0, 0.01)
            self.tick()
        self.servo(side, tab + 0.12 * UP, Rg, 1.0, speed=0.015, max_steps=200)
        self.servo(side, tab + 0.004 * UP, Rg, 1.0, speed=0.01, max_steps=100)
        self.goal[side] = (tab + 0.004 * UP, Rg, -1.0, 0.01)
        self.wait(8)
        self.servo(side, tab + 0.10 * UP, Rg, -1.0, max_steps=60)
        self.log.append({"dish": k, "kind": self.env.episode["dish_of"][k], "pieces": len(mine),
                         "in_pot": sum(1 for i in mine if w.piece_place(i) == "pot"),
                         "touched": sum(1 for i in mine if w.touched[i]), "t": w.t})

    # --- the pot ------------------------------------------------------------------------------
    def tab_points(self):
        d = self.w.d
        p, Rp = d.xpos[self.w.pot].copy(), d.xmat[self.w.pot].reshape(3, 3).copy()
        off = S.POT_R + S.POT_WALL + S.TAB_HALF[1] - 0.002
        z = S.POT_FLOOR + S.TAB_Z + 0.004
        return {"l": p + Rp @ np.array([0, off + 0.008, z]), "r": p + Rp @ np.array([0, -off - 0.008, z])}, Rp

    def both(self, goals, grip, speed=0.02, tol=0.004, max_steps=200):
        for s in ("l", "r"):
            self.goal[s] = (goals[s][0], goals[s][1], grip, speed)
        for _ in range(max_steps):
            self.tick()
            if all(np.linalg.norm(goals[s][0] - self.w.arms[s].pos()) < tol for s in ("l", "r")):
                return True
        return False

    def take_pot(self):
        self.phase = "take_pot"
        for attempt in range(3):
            T, Rp = self.tab_points()
            yaw = math.atan2(Rp[1, 0], Rp[0, 0])
            Rg = rz(yaw) @ np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])
            self.Rg = Rg
            self.both({s: (T[s] + 0.10 * UP, Rg) for s in T}, -1.0, max_steps=250)
            self.both({s: (T[s], Rg) for s in T}, -1.0, speed=0.01, tol=0.002, max_steps=120)
            for s in T:
                self.goal[s] = (T[s], Rg, 1.0, 0.01)
            self.wait(8)
            ok = {s: bool(np.linalg.norm(T[s] - self.w.arms[s].pos()) < 0.01 and self.w.arms[s].gripper() > 0.025)
                  for s in T}
            self.log.append({"take_pot": ok, "t": self.w.t})
            if all(ok.values()):
                break
            for s in T:
                self.goal[s] = (self.w.arms[s].pos() + 0.10 * UP, Rg, -1.0, 0.02)
            self.wait(10)
        self.T0 = T
        lift = {s: (T[s] + 0.10 * UP, Rg) for s in T}
        self.both(lift, 1.0, speed=0.01, max_steps=60)
        self.wait(10)
        self.sag = {s: float(lift[s][0][2] - self.w.arms[s].pos()[2]) for s in T}

    def pot_goals(self, centre, th, tab_half):
        """Hand goals for the tab line's midpoint at `centre`, the pot rolled by th about world y (its front down)."""
        Rth = S.axis_angle_mat([0.0, th, 0.0])
        out = {}
        for s, sg in (("l", 1.0), ("r", -1.0)):
            p = centre + np.array([0.0, sg * tab_half, 0.0]) + np.array([0, 0, self.sag[s]])
            out[s] = (p, Rth @ self.Rg)
        return out

    def pour(self, nm):
        w, lq = self.w, self.w.liquid
        self.phase = "pour_" + nm
        target = self.env.episode["order"][nm]["soup_ml"] / 1e6
        bi = 1 + S.BOWL_NAMES.index(nm)
        bp = w.d.xpos[w.bowls[nm]].copy()
        tab_half = float(np.linalg.norm(self.T0["l"] - self.T0["r"])) / 2
        line_z = S.TABLE_Z + 0.27
        centre = np.array([bp[0] - 0.165, bp[1], line_z])
        cur = (self.L.pos() + self.R.pos()) / 2
        if cur[2] < line_z - 0.01:
            self.both(self.pot_goals(np.array([cur[0], cur[1], line_z]), 0.0, tab_half), 1.0, max_steps=150)
        self.both(self.pot_goals(centre, 0.0, tab_half), 1.0, speed=0.015, max_steps=200)
        self.wait(8)
        th = 0.0
        start_v = lq.vessels[bi].vol
        for _ in range(600):
            got = lq.vessels[bi].vol + sum(pc[2] for pc in lq.parcels if pc[3] == 0)
            Q = lq.flow[0]
            if got + Q * LEAD_S >= target:
                break
            qd = FLOW_SLOW if target - got < SLOW_ML / 1e6 else FLOW_FAST
            if Q <= 0.0:
                th += 0.025
            else:
                th += float(np.clip(0.0004 * (qd - Q) * 1e6, -0.02, 0.02))
            th = min(th, 1.2)
            for s, g in self.pot_goals(centre, th, tab_half).items():
                self.goal[s] = (g[0], g[1], 1.0, 0.02)
            self.tick()
        while th > 0.0:
            th = max(0.0, th - 0.15)
            for s, g in self.pot_goals(centre, th, tab_half).items():
                self.goal[s] = (g[0], g[1], 1.0, 0.03)
            self.tick()
        for _ in range(30):
            self.tick()
            if not lq.parcels:
                break
        self.log.append({"pour": nm, "target_ml": round(target * 1e6, 1), "got_ml": round(1e6 * lq.vessels[bi].vol, 1),
                         "added_ml": round(1e6 * (lq.vessels[bi].vol - start_v), 1), "t": w.t})

    def put_pot_back(self):
        self.phase = "pot_back"
        tab_half = float(np.linalg.norm(self.T0["l"] - self.T0["r"])) / 2
        mid0 = (self.T0["l"] + self.T0["r"]) / 2
        cur = (self.L.pos() + self.R.pos()) / 2
        line_z = max(cur[2] - self.sag["l"], mid0[2] + 0.10)
        self.both(self.pot_goals(np.array([cur[0], cur[1], line_z]), 0.0, tab_half), 1.0, max_steps=100)
        self.both(self.pot_goals(np.array([mid0[0], mid0[1], line_z]), 0.0, tab_half), 1.0, speed=0.015, max_steps=200)
        self.both(self.pot_goals(mid0 + np.array([0, 0, 0.006]), 0.0, tab_half), 1.0, speed=0.008, max_steps=150)
        for s in ("l", "r"):
            self.goal[s] = (self.goal[s][0] - np.array([0, 0, self.sag[s] + 0.004]), self.goal[s][1], 1.0, 0.005)
        self.wait(10)
        for s in ("l", "r"):
            self.goal[s] = (self.w.arms[s].pos(), self.goal[s][1], -1.0, 0.005)
        self.wait(10)
        self.both({s: (self.w.arms[s].pos() + 0.15 * UP, self.goal[s][1]) for s in ("l", "r")}, -1.0, max_steps=60)

    def finish(self):
        self.phase = "finish"
        while True:
            self.tick()

    def run(self):
        w, order = self.w, self.env.episode["order"]
        try:
            self.turn_knob(S.KNOB_MAX)
            self.take_ladle()
            self.park_ladle()
            for n, k in enumerate(self.dish_order()):
                self.tip_dish(k, TIP_SPOTS[n])
            self.serve_pieces()
            self.put_ladle_down()
            self.unwind_r()
            self.turn_knob(0.0)
            self.goal["r"] = None
            self.take_pot()
            names = sorted(order, key=lambda nm: -w.d.xpos[w.bowls[nm]][1])
            for nm in names:
                self.pour(nm)
            self.put_pot_back()
            self.finish()
        except Done:
            pass
        return self.ep.score()


def run_one(unit_seed, seed, render=False):
    unit = S.draw_unit(unit_seed)
    env = soup_env.SoupEnv(unit, render=render)
    ep = E.Episode(env, seed)
    ep.reset()
    t0 = time.time()
    ref = Reference(ep)
    sc = ref.run()
    sc.update(unit_seed=unit_seed, seed=seed, wall=round(time.time() - t0, 1),
              steps_per_s=round(env.t / max(1e-9, time.time() - t0), 1), ref_log=ref.log, phases=ref.phase_log)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(json.dumps(run_one(a.unit, a.seed)))


if __name__ == "__main__":
    main()
