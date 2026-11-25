"""The privileged reference of cable_routing_through_clips (the token oracle's program).

It routes the way a harness fitter does: the span of cable between the last fixed point (the connector, a clip that
holds it, a post it bears on) and the hand is kept taut at a few newtons, swept over the next clip's lead-in, pressed
down and checked by a pull; round a post the hand moves on an arc with the span taut; after a clip it draws the cable
back onto an earlier post that later work has left it standing off. It reads the grader's truth (the cable's points,
the fixtures as mounted, the mean connector tension), which only a token run is given: it is not a solution an
agent could write. `Fitter.run` is a generator that yields public actions; oracle_main.py drives it over the wire,
`python3 -I fitter.py --seeds ...` runs it in process on the simulator (maintainers; needs tests/harness_env.py's
copy beside it). `variant(name)` gives the control programs of tests/review/mutants.md.
"""
import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import harness_env as H  # noqa: E402
from fitter_geom import unit, arc_of, at_arc, nearest_arc, lateral_at, rot_down, quat_mat, rotvec  # noqa: E402

REACH = 0.05             # the hand works this far past a fixture
Z_SWEEP = 0.075          # grip site above the board while a taut span is swept over a clip (the cable clears the lead-in by 1 cm)
Z_LAY = H.GROOVE_BELOW_SITE + 0.006    # grip site above the board with the held cable's axis 6 mm up: the groove's lower edge clears the board for every radius
Z_HOVER = 0.10
T_SPAN = 3.5             # N, the span's working tension
T_PRESS = 4.5            # N, while pressing into a clip
T_MAX = 9.0              # N, never pull past this on purpose
YAW_MID = 0.85           # rad: the middle of the hand's free yaw range, from home
PICK_EXTRA = 0.012       # m of cable beyond the straight distance: over a clip's top and down to the board costs up to 8 mm
PICK_EXTRA_POST = 0.028  # m: a span raised to pass over a post needs about 2 cm more than the straight distance
Z_OVER_MAX = 0.190       # the highest the hand goes for that
Z_OVER = 0.105           # grip site above the board while a span is carried over a post (the cable at the hand 88 mm up)
T_GUARD = 15.0           # N: stop and back off at this 0.1 s mean
T_ABORT = 18.0           # N, back off at once


class Fitter(object):
    name = "reference"
    watch_tension = True
    stop_after = None
    flip_sides = False
    flip_axes = False
    nominal = False

    def __init__(self, sheet):
        self.sheet = sheet
        self.grip = -1.0
        self.events = []

    # -- what it believes about the cell ---------------------------------------------------
    def fixtures(self, truth):
        if self.nominal:
            fx = [dict(f) for f in self.sheet["fixtures"]]
            for f, g in zip(fx, truth["fixtures"]):
                if f["type"] == "clip":
                    f["gap"], f["axis"] = g["gap"], g["axis"]      # the job sheet gives a clip's place only
            conn = self.sheet["connector"]["pos"]
        else:
            fx = [dict(f) for f in truth["fixtures"]]
            conn = truth["connector"]["pos"]
        for f in fx:
            f["pos"] = np.array(f["pos"], float)
            if f["type"] == "post" and self.flip_sides:
                f["side"] = "left" if f["side"] == "right" else "right"
            if f["type"] == "clip" and self.flip_axes:
                f["axis"] = f["axis"] + math.pi / 2
        return np.array(conn, float), fx

    def passages(self, conn, fx, R):
        pts = [f["pos"] if f["type"] == "clip" else None for f in fx]
        info = [None] * len(fx)
        for _ in range(3):
            for i, f in enumerate(fx):
                if f["type"] != "post":
                    continue
                prev = conn if i == 0 else (pts[i - 1] if pts[i - 1] is not None else fx[i - 1]["pos"])
                nxt = None
                if i + 1 < len(fx):
                    nxt = pts[i + 1] if pts[i + 1] is not None else fx[i + 1]["pos"]
                d_in = unit(f["pos"] - prev)
                if nxt is not None:
                    d_out = unit(nxt - f["pos"])
                else:
                    a = 1.0 if f["side"] == "left" else -1.0
                    d_out = np.array([math.cos(a) * d_in[0] - math.sin(a) * d_in[1], math.sin(a) * d_in[0] + math.cos(a) * d_in[1]])
                left = np.array([-d_in[1], d_in[0]])
                out = -left if f["side"] == "left" else left      # the cable passes on this side of the post
                bis = unit(d_in - d_out)
                if np.dot(bis, out) < 0.2:
                    bis = out
                pts[i] = f["pos"] + (H.POST_R + R) * bis
                info[i] = (d_in, d_out, bis, out)
        return pts, info

    # -- primitives (generators: yield an action, receive (obs, truth) through `st`) ----------
    def z(self, h):
        return H.BOARD_T + h

    def meas_psi(self, obs):
        R = quat_mat(obs["tcp"]["quat"])
        return math.atan2(R[1, 0], R[0, 0])

    def cur_psi(self, obs):
        """The hand's yaw, unwrapped (continuous with the yaw followed so far)."""
        m = self.meas_psi(obs)
        if getattr(self, "_psi_meas", None) is None:
            return m
        return self._psi_unw + ((m - self._psi_meas + math.pi) % (2 * math.pi) - math.pi)

    def psi_for(self, yaw):
        psi = yaw + math.pi / 2
        while psi - self.psi0 > math.pi / 2:
            psi -= math.pi
        while psi - self.psi0 <= -math.pi / 2:
            psi += math.pi
        return psi

    def tension(self, st):
        return st["truth"]["tension_mean"] if self.watch_tension else 0.0

    def act_toward(self, obs, pos, psi, vmax=0.025):
        p = np.array(obs["tcp"]["pos"])
        dp = np.asarray(pos, float) - p
        n = float(np.linalg.norm(dp))
        if n > vmax:
            dp *= vmax / n
        # the yaw is followed UNWRAPPED: the planned yaw may be more than half a turn from where the hand is, and the
        # shortest rotation toward it can be the one that runs the wrist into its limit (it stalled there for
        # hundreds of steps). The hand's yaw is tracked from step to step and the target approached on the planned side.
        meas = self.meas_psi(obs)
        last = getattr(self, "_psi_meas", None)
        if last is None:
            self._psi_unw = meas
        else:
            self._psi_unw += (meas - last + math.pi) % (2 * math.pi) - math.pi
        self._psi_meas = meas
        gap = psi - self._psi_unw
        psi_step = self._psi_unw + max(-0.15, min(0.15, gap))
        dr = rotvec(rot_down(psi_step) @ quat_mat(obs["tcp"]["quat"]).T)
        n2 = abs(gap)
        return [float(dp[0]), float(dp[1]), float(dp[2]), float(dr[0]), float(dr[1]), float(dr[2]), self.grip, 0], n, n2

    def step(self, st, a):
        st["obs"], st["truth"] = yield a
        t = st["truth"]["tension_mean"]
        ph = getattr(self, "phase", "start")
        pk = self.__dict__.setdefault("peaks", {})
        if t > pk.get(ph, 0.0):
            pk[ph] = round(t, 1)
        cnt = self.__dict__.setdefault("phase_steps", {})
        cnt[ph] = cnt.get(ph, 0) + 1

    def move(self, st, pos, psi, tol=0.003, max_steps=120, vmax=0.025, guard=True):
        last_n2, stalled = None, 0
        for _ in range(max_steps):
            if guard and self.grip > 0 and self.tension(st) > T_ABORT:
                return "tension"
            a, n, n2 = self.act_toward(st["obs"], pos, psi, vmax)
            if n < tol and n2 < 0.02:
                return "ok"
            if n < tol and last_n2 is not None and abs(last_n2 - n2) < 0.002:
                stalled += 1
                if stalled >= 4:
                    return "yaw"               # in place, the wrist at its limit
            else:
                stalled = 0
            last_n2 = n2
            yield from self.step(st, a)
        return "timeout"

    def hold(self, st, steps):
        for _ in range(steps):
            yield from self.step(st, [0, 0, 0, 0, 0, 0, self.grip, 0])

    def descend(self, st, x, y, z_des, psi, max_steps=45):
        for _ in range(max_steps):
            obs = st["obs"]
            p = np.array(obs["tcp"]["pos"])
            if p[2] <= z_des + 0.0007:
                return "ok"
            if obs["wrench"]["force"][2] < self.fz0 - 12.0:
                a, _n, _n2 = self.act_toward(obs, [p[0], p[1], p[2] + 0.02], psi, 0.02)
                yield from self.step(st, a)
                return "contact"
            a, _n, _n2 = self.act_toward(obs, [x, y, z_des - 0.0025], psi, 0.012)
            yield from self.step(st, a)
        return "timeout"

    def holding(self, st):
        P = np.array(st["truth"]["cable"])
        p = np.array(st["obs"]["tcp"]["pos"]) - np.array([0.0, 0.0, H.GROOVE_BELOW_SITE])
        return float(np.min(np.linalg.norm(P - p, axis=1))) < 0.014

    def clear_spot(self, P, s):
        for ds in (0.0, 0.01, 0.02, 0.03, 0.04, 0.05, -0.01, -0.02, 0.06, 0.07):
            g, tang = at_arc(P, s + ds)
            nrm = np.array([-tang[1], tang[0]])
            ok = True
            for f in self.fx:
                rf = 0.018 if f["type"] == "clip" else H.POST_CAP_R
                d = f["pos"] - g[:2]
                if abs(float(np.dot(d, tang))) < 0.022 + rf and abs(float(np.dot(d, nrm))) < 0.058 + rf:
                    ok = False
            if ok:
                return s + ds
        return s

    def plan_yaw(self, tang, span_dir):
        """The cable has a sense (connector end, free end) and the groove holds it along its axis: when the span is
        drawn taut along span_dir the cable in the hand must point along span_dir, not against it (a hand turned the
        other way round wraps the cable about itself). Choose the hand's yaw at the pick (psi_p) and with the span
        along span_dir (psi_t) so that the turn between them carries the cable's sense and both stay near home."""
        ang_t, ang_u = math.atan2(tang[1], tang[0]), math.atan2(span_dir[1], span_dir[0])
        delta = (ang_t - ang_u + math.pi) % (2 * math.pi) - math.pi
        best = None
        for k in (-2, -1, 0, 1, 2):
            psi_t = ang_u + math.pi / 2 + k * math.pi
            psi_p = psi_t + delta
            # the wrist (joint 7, 0.86 rad at home, limits +-2.9) turns against the yaw: about 1.5 rad of yaw is
            # free below home and 3.2 above; cost = the worse of the two yaws' distance from that window's middle
            cost = max(abs(psi_t - self.psi0 - YAW_MID), abs(psi_p - self.psi0 - YAW_MID))
            if best is None or cost < best[0]:
                best = (cost, psi_t, psi_p)
        self.psi_t, self.ang_u, self.yaw_cost = best[1], ang_u, best[0]        # yaw_cost above 2.35: a yaw is out of the wrist's range
        return best[2]

    def psi_dir(self, direction):
        """Hand yaw with the held cable along `direction` (continuous from the pick's plan)."""
        ang = math.atan2(direction[1], direction[0])
        return self.psi_t + ((ang - self.ang_u + math.pi) % (2 * math.pi) - math.pi)

    def pick(self, st, s_g, span_dir=None):
        """Close the groove on the cable at arc length s_g (moved to a spot the open hand can land on); lift 2 cm.
        True when the cable is in the hand. span_dir: the direction the span will be drawn in (see plan_yaw)."""
        for _ in range(3):
            P = np.array(st["truth"]["cable"])
            s = self.clear_spot(P, s_g)
            self.picked_s = s
            g, tang = at_arc(P, s)
            psi = self.psi_for(math.atan2(tang[1], tang[0])) if span_dir is None else self.plan_yaw(tang, span_dir)
            self.grip = -1.0
            yield from self.move(st, [g[0], g[1], self.z(Z_HOVER)], psi, tol=0.006)
            P = np.array(st["truth"]["cable"])
            g, tang = at_arc(P, s)
            psi = self.psi_for(math.atan2(tang[1], tang[0])) if span_dir is None else self.plan_yaw(tang, span_dir)
            yield from self.move(st, [g[0], g[1], g[2] + 0.05], psi, tol=0.003)
            r = yield from self.descend(st, g[0], g[1], g[2] - st["truth"]["radius"] + H.grasp_site_z(st["truth"]["radius"]), psi)
            if r != "contact":
                self.grip = 1.0
                yield from self.hold(st, 7)
                p = np.array(st["obs"]["tcp"]["pos"])
                yield from self.move(st, [p[0], p[1], p[2] + 0.02], psi, tol=0.004, vmax=0.01)
                if self.holding(st):
                    return True
            self.grip = -1.0
            yield from self.hold(st, 4)
            p = np.array(st["obs"]["tcp"]["pos"])
            yield from self.move(st, [p[0], p[1], self.z(Z_HOVER)], psi, tol=0.006)
        return False

    def release(self, st):
        self.grip = -1.0
        yield from self.hold(st, 5)
        p = np.array(st["obs"]["tcp"]["pos"])
        yield from self.move(st, [p[0], p[1], self.z(Z_HOVER)], self.cur_psi(st["obs"]), tol=0.006)

    # -- the span: length bookkeeping and tension by force ----------------------------------
    # The cable between the last fixed point and the hand is `self.free` long (arc length picked minus arc length at
    # the fixed point). A straight span cannot be longer than that: one millimetre of hand travel past it is 3.5 to
    # 6.5 N (measured). So every move with the cable in hand is planned on the span's length: the hand goes round the
    # fixed point on arcs, the radius kept inside the length the path needs, and tension is set by small steps on the
    # tension read back (never by a position step into a straight span).
    def cable_h(self, z_site):
        """Height of the held cable's axis above the board for a grip site height z_site (world)."""
        return z_site - H.BOARD_T - H.GROOVE_BELOW_SITE

    def path_len(self, origin, via, xy, z_site):
        """Length of the shortest cable path from the fixed point (on the board) to the hand, over the clip `via`
        = (xy, height) when there is one between them."""
        h = self.cable_h(z_site)
        a0 = np.array([origin[0], origin[1], self.R])
        hand = np.array([xy[0], xy[1], h])
        if via is None:
            return float(np.linalg.norm(hand - a0))
        v = np.array([via[0][0], via[0][1], via[1]])
        # the cable touches the clip only when the straight line would pass below it there
        d_av, d_ah = np.linalg.norm(v[:2] - a0[:2]), np.linalg.norm(hand[:2] - a0[:2])
        line_h = a0[2] + (h - a0[2]) * min(1.0, d_av / max(d_ah, 1e-9))
        if line_h >= via[1]:
            return float(np.linalg.norm(hand - a0))
        return float(np.linalg.norm(v - a0) + np.linalg.norm(hand - v))

    def r_cap(self, origin, via, direction, z_site, slack):
        """The largest distance from the fixed point, along `direction`, at which the path needs no more than the free
        length less `slack`."""
        lo, hi = 0.0, self.free + 0.02
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if self.path_len(origin, via, origin + mid * direction, z_site) <= self.free - slack:
                lo = mid
            else:
                hi = mid
        return lo

    def guard(self, st, origin, psi):
        """Back off toward the fixed point while the mean tension is over the guard. True when it had to."""
        hit = False
        for _ in range(12):
            if not (self.grip > 0 and self.tension(st) >= T_GUARD):
                break
            hit = True
            p = np.array(st["obs"]["tcp"]["pos"])
            d = unit(origin - p[:2])
            a, _n, _n2 = self.act_toward(st["obs"], [p[0] + 0.004 * d[0], p[1] + 0.004 * d[1], p[2]], psi, 0.004)
            yield from self.step(st, a)
        if hit:
            self.guards = getattr(self, "guards", 0) + 1
        return hit

    def span_move(self, st, origin, via, direction, z_site, psi, slack=0.006, r=None, vmax=0.012, ds=0.008, r_floor=0.0):
        """With the cable in hand: to distance r (default: as far as the free length allows less `slack`) from the
        fixed point along `direction` at height z_site, going round the fixed point (angle first, radius and height
        together, never a straight translation), the radius capped at every waypoint. Returns the radius reached."""
        p = np.array(st["obs"]["tcp"]["pos"])
        v0 = p[:2] - origin
        r0, th0 = float(np.linalg.norm(v0)), math.atan2(v0[1], v0[0])
        th1 = math.atan2(direction[1], direction[0])
        dth = (th1 - th0 + math.pi) % (2 * math.pi) - math.pi
        cap1 = self.r_cap(origin, via, direction, z_site, slack)
        r1 = cap1 if r is None else min(r, cap1)
        r1 = max(r1, r_floor)
        n = int(max(2, abs(dth) * max(r0, r1) / ds, abs(z_site - p[2]) / ds, abs(r1 - r0) / ds))
        rr = r0
        for k in range(1, n + 1):
            w = k / float(n)
            th = th0 + dth * w
            dirn = np.array([math.cos(th), math.sin(th)])
            zz = p[2] + (z_site - p[2]) * w
            rr = min(r0 + (r1 - r0) * w, self.r_cap(origin, via, dirn, zz, slack))
            rr = max(rr, r_floor)      # (the length over a clip's top is not monotonic in the radius: the cap alone can
            #                             put the hand back on the near side of the clip; past the clip it must stay)
            pos = origin + rr * dirn
            for _i in range(6):
                pc = np.array(st["obs"]["tcp"]["pos"])
                d = np.array([pos[0], pos[1], zz]) - pc
                if float(np.linalg.norm(d)) < 0.003:
                    break
                # outward travel (away from the fixed point) in one control step is bounded by the slack the path
                # still has plus what the boot and the arm give for a few newtons; sideways and inward travel is free
                rad_dir = unit(pc[:2] - origin)
                out_now = float(np.dot(d[:2], rad_dir))
                slack_now = self.free - self.path_len(origin, via, pc[:2], pc[2])
                tn = self.tension(st)
                allow = max(0.0015, slack_now - 0.001) if tn < 6.0 else 0.0005
                if self.grip > 0 and out_now > allow:
                    d[:2] -= (out_now - allow) * rad_dir
                v_now = vmax
                if self.grip > 0 and tn > 6.0:
                    # the span is tightening on the way (the cable bends at the connector, or bears on a clip's
                    # top): give length now, in proportion, and go slower
                    give = min(0.003, 0.0005 * (tn - 5.0))
                    d[:2] -= give * rad_dir
                    r0 -= give
                    r1 -= give
                    v_now = min(vmax, 0.004)
                a, _n, _n2 = self.act_toward(st["obs"], pc + d, psi, v_now)
                yield from self.step(st, a)
            if (yield from self.guard(st, origin, psi)):
                pnow = np.array(st["obs"]["tcp"]["pos"])
                rr = float(np.linalg.norm(pnow[:2] - origin))
                r1 = min(r1, rr)
                r0 = min(r0, rr)
            elif self.grip > 0 and self.tension(st) > 8.0:
                r1 -= 0.002                         # the span is tighter than the plan thought: give length
                r0 -= 0.002
            if not self.holding(st):
                break
        return rr

    def tension_to(self, st, origin, via, direction, r, z_site, psi, t_des, max_steps=70, band=0.6, r_min=0.0):
        """Set the span's tension by force: read the tension, move the hand out or in along the span by a step that
        the measured stiffness (up to 6.5 N a millimetre) turns into about a newton at most; while the path still has
        geometric slack the steps are larger, down to 2 mm of slack. Returns the radius."""
        calm = 0
        if not self.watch_tension:
            r = self.r_cap(origin, via, direction, z_site, -0.004)       # no reading: 4 mm past the straight length
            pos = origin + r * direction
            yield from self.move(st, [pos[0], pos[1], z_site], psi, tol=0.002, max_steps=30, guard=False)
            return r
        for _ in range(max_steps):
            if (yield from self.guard(st, origin, psi)):
                p = np.array(st["obs"]["tcp"]["pos"])
                r = float(np.linalg.norm(p[:2] - origin))
                calm = 0
                continue
            t = self.tension(st)
            err = t_des - t
            if abs(err) < band:
                calm += 1
                if calm >= 2:
                    return r
            else:
                calm = 0
            slack_geo = self.free - self.path_len(origin, via, origin + r * direction, z_site)
            if t < 0.8 and slack_geo > 0.003:
                step = min(0.004, slack_geo - 0.002)
            elif t < 0.8 and err > 0:
                step = 0.0015                       # still slack: with the boot about 3 N if this step straightens the span
            else:
                step = float(np.clip(0.0003 * err, -0.0020, 0.0005))     # about 2 N a millimetre with the boot: 1 N a step
            r = max(r_min, r + step)
            pos = origin + r * direction
            a, _n, _n2 = self.act_toward(st["obs"], [pos[0], pos[1], z_site], psi, 0.006)
            yield from self.step(st, a)
            if not self.holding(st):
                return r
        return r

    # -- fixtures ---------------------------------------------------------------------------
    def clip_seated(self, truth, f):
        P = np.array(truth["cable"])
        n = np.array([-math.sin(f["axis"]), math.cos(f["axis"])]) * (f["gap"] / 2 + H.PIN_R)
        return H._crossing(P, f["pos"] - n, f["pos"] + n, H.BOARD_T + H.SEATED_Z) is not None

    def do_clip(self, st, i, f, anchor, lateral_bias=0.0, t_press=T_PRESS, reach=REACH, align=True, attempts=3, z_stop=None):
        """Sweep the taut span over the clip, press it in, pull-test it. Returns True when it is seated at the end."""
        D = float(np.linalg.norm(f["pos"] - anchor))
        u = unit(f["pos"] - anchor)
        nrm = np.array([-u[1], u[0]])
        self.R = st["truth"]["radius"]
        z_stop = Z_LAY if z_stop is None else z_stop
        top = H.PIN_H + 0.004                       # the cable rides here (above the board) until the barbs let it in
        seated = False
        for attempt in range(attempts):
            P = np.array(st["truth"]["cable"])
            sa = nearest_arc(P, anchor) if i > 0 else 0.0
            self.phase = "pick"
            got = yield from self.pick(st, sa + D + reach + PICK_EXTRA, span_dir=u)
            Pp = np.array(st["truth"]["cable"])
            gp = np.array(st["obs"]["tcp"]["pos"])
            ev = {"fixture": i, "attempt": attempt, "got": bool(got), "seated": False, "D": round(D, 4)}
            self.events.append(ev)
            if not got:
                continue
            self.free = nearest_arc(Pp, gp[:2]) - sa
            ev["free"] = round(self.free, 4)
            ev["yaw_cost"] = round(self.yaw_cost, 2)
            psi = self.psi_dir(u)
            r_min = D + 0.03
            via_up = (f["pos"], top + self.R)
            corr = lateral_bias
            # come in from the side the hand is on: slack while travelling, the span drawn taut 7 cm to the side of
            # the slot, then stepped across it above the lead-in on arcs about the fixed point
            self.phase = "sweep"
            side = 1.0 if float(np.dot(gp[:2] - anchor, nrm)) >= 0 else -1.0
            aim = unit(f["pos"] + (corr + side * 0.07) * nrm - anchor)
            r = yield from self.span_move(st, anchor, None, aim, self.z(Z_SWEEP), psi, slack=0.008, vmax=0.025, ds=0.022)
            r = yield from self.tension_to(st, anchor, None, aim, r, self.z(Z_SWEEP), psi, T_SPAN, r_min=r_min)
            for off in (0.03, 0.0):
                aim = unit(f["pos"] + (corr + side * off) * nrm - anchor)
                r = yield from self.span_move(st, anchor, None, aim, self.z(Z_SWEEP), psi, slack=0.0, r=r)
                if not 1.5 <= self.tension(st) <= 7.0:      # an arc about the fixed point keeps the length: usually nothing to do
                    r = yield from self.tension_to(st, anchor, None, aim, r, self.z(Z_SWEEP), psi, T_SPAN, r_min=r_min, max_steps=20, band=1.2)
            for k in range(5):
                if not align:
                    break
                lat = lateral_at(np.array(st["truth"]["cable"]), f["pos"], u, sa, sa + self.free)
                ev["lat_mm"] = None if lat is None else round(lat[0] * 1000, 1)
                if lat is None or abs(lat[0] - lateral_bias) < 0.001:
                    break
                corr = float(np.clip(corr - 0.9 * (lat[0] - lateral_bias), -0.04, 0.04))
                aim = unit(f["pos"] + corr * nrm - anchor)
                r = yield from self.span_move(st, anchor, None, aim, self.z(Z_SWEEP), psi, slack=0.0, r=r)
                yield from self.hold(st, 2)
                if not 1.5 <= self.tension(st) <= 7.0:
                    r = yield from self.tension_to(st, anchor, None, aim, r, self.z(Z_SWEEP), psi, T_SPAN, r_min=r_min, max_steps=20, band=1.2)
            aim = unit(f["pos"] + corr * nrm - anchor)
            # press: the hand comes down in steps; the path over the clip's top gets longer as it does, so the radius
            # follows the length first and the tension is set by force after every step
            self.phase = "press"
            zs = np.linspace(Z_SWEEP, z_stop, 5)[1:]
            for kz, zz in enumerate(zs):
                r = yield from self.span_move(st, anchor, via_up, aim, self.z(zz), psi, slack=0.002, r=r + 0.02, vmax=0.012, r_floor=r_min)
                r = max(r, r_min)
                if kz == len(zs) - 1 or self.tension(st) < 1.0:
                    r = yield from self.tension_to(st, anchor, via_up, aim, r, self.z(zz), psi, t_press, r_min=r_min, max_steps=18, band=1.0)
                if self.clip_seated(st["truth"], f) or not self.holding(st):
                    break
            if self.holding(st) and not self.clip_seated(st["truth"], f):
                p = np.array(st["obs"]["tcp"]["pos"])
                r = yield from self.span_move(st, anchor, via_up, aim, self.z(z_stop), psi, slack=0.0015, r=r + 0.02, vmax=0.008, r_floor=r_min)
                r = yield from self.tension_to(st, anchor, via_up, aim, max(r, r_min), self.z(z_stop), psi, min(T_MAX, t_press + 3.0), r_min=r_min, max_steps=40)
            yield from self.hold(st, 3)
            ev["tension"] = round(st["truth"]["tension_mean"], 1)
            ev["holding"] = bool(self.holding(st))
            seated = self.clip_seated(st["truth"], f)
            self.phase = "pull_test"
            if seated and self.holding(st):
                # pull test: the cable is now fixed at the clip; the hand rises 3 cm on the length the path from the
                # clip needs (paying out first), the tension set by force at the top
                via_in = (f["pos"], self.R)
                p = np.array(st["obs"]["tcp"]["pos"])
                r = float(np.linalg.norm(p[:2] - anchor))
                for dz in (0.015, 0.03):                  # pay out 6 mm more than the path needs before each rise
                    r = yield from self.span_move(st, anchor, via_in, aim, self.z(Z_LAY + dz), psi, slack=0.006, r=r, vmax=0.010, r_floor=r_min)
                r = yield from self.tension_to(st, anchor, via_in, aim, max(r, r_min), self.z(Z_LAY + 0.03), psi, t_press, r_min=r_min, max_steps=20, band=1.0)
                yield from self.hold(st, 2)
                seated = self.clip_seated(st["truth"], f)
                ev["pull_test"] = bool(seated)
                ev["pull_tension"] = round(st["truth"]["tension_mean"], 1)
                r = yield from self.span_move(st, anchor, via_in, aim, self.z(Z_LAY), psi, slack=0.004, r=r - 0.004, vmax=0.012)
                # dress: work at this clip (the pick's lift, the slack paid out for the press) can leave the cable
                # standing off a post made earlier (seeds 1111, 1113: 16 and 18 mm off at the end, never over the cap).
                # The cable slides along a clip's passage, so a pull from here draws it back onto the post.
                loose = [pf for pf in self.fx[:i] if pf["type"] == "post" and not self.post_made(st["truth"], pf)]
                if loose and seated and self.holding(st):
                    ev["dress"] = len(loose)
                    for t_d in (6.0, T_MAX):
                        r = yield from self.tension_to(st, anchor, via_in, aim, max(r, r_min), self.z(Z_LAY), psi, t_d, r_min=r_min, max_steps=25, band=1.0)
                        yield from self.hold(st, 2)
                        if all(self.post_made(st["truth"], pf) for pf in loose):
                            break
                    ev["dress_ok"] = bool(all(self.post_made(st["truth"], pf) for pf in loose))
                    ev["dress_tension"] = round(st["truth"]["tension_mean"], 1)
                    r = yield from self.span_move(st, anchor, via_in, aim, self.z(Z_LAY), psi, slack=0.002, r=r - 0.002, vmax=0.012)
                    seated = self.clip_seated(st["truth"], f)
            ev["seated"] = bool(seated)
            ev["t"] = st["obs"]["t"]
            self.phase = "release"
            yield from self.release(st)
            if seated:
                break
        return seated

    def post_made(self, truth, f):
        P = np.array(truth["cable"]); c = f["pos"]
        dist = np.linalg.norm(P[:, :2] - c, axis=1); j = int(np.argmin(dist))
        jj = min(max(j, 1), len(P) - 2); tang = P[jj + 1, :2] - P[jj - 1, :2]
        cross = tang[0] * (c[1] - P[j, 1]) - tang[1] * (c[0] - P[j, 0])
        side_ok = (cross > 0) if f["side"] == "left" else (cross < 0)
        return bool(side_ok and dist[j] <= H.POST_R + truth["radius"] + H.POST_TOL and P[j, 2] < H.BOARD_T + H.POST_H)

    def do_post(self, st, i, f, anchor, target, geom):
        d_in, d_out, bis, out = geom
        D = float(np.linalg.norm(target - anchor))
        self.R = st["truth"]["radius"]
        P = np.array(st["truth"]["cable"])
        sa = nearest_arc(P, anchor) if i > 0 else 0.0
        aim = unit(target + 0.03 * out - anchor)
        self.phase = "post_pick"
        got = yield from self.pick(st, sa + D + REACH + PICK_EXTRA_POST, span_dir=aim)
        ev = {"fixture": i, "attempt": 0, "got": bool(got), "post": True}
        self.events.append(ev)
        if not got:
            return False
        self.phase = "post"
        Pp = np.array(st["truth"]["cable"]); gp = np.array(st["obs"]["tcp"]["pos"])
        self.free = nearest_arc(Pp, gp[:2]) - sa
        psi = self.psi_dir(aim)
        # A span that lies on the post's inner side has to go OVER the post to reach the outer side: drawn across at the
        # sweep height it passes under the cap (18 mm radius over a 12 mm post, top 33 mm above the board) and stays
        # hooked there, on the wrong side (traced on seeds 901 and 914; on 907 it happened to clear the cap). So: if
        # the hand is on the inner side, raise it first, draw the span straight at that height (the cable then passes
        # the post's place well above the cap), go round to the outer side on an arc, and only then come down.
        p0 = np.array(st["obs"]["tcp"]["pos"])
        inner = float(np.dot(p0[:2] - f["pos"], out)) < 0.02 or float(np.dot(unit(p0[:2] - anchor), out)) < float(np.dot(aim, out))
        ev["crossed_over"] = bool(inner)
        if inner:
            self.sub = "raise"
            d0 = unit(p0[:2] - anchor)
            # how high: the span's height at the post is the hand's times D / r, and it has to clear the cap's top by
            # 2 cm (seed 608, a post 9 cm from the fixed point: at the fixed 105 mm it cleared by 4.7 mm and hooked)
            r_now = float(np.linalg.norm(p0[:2] - anchor))
            need = (H.POST_H + 0.006 + self.R + 0.020) * max(r_now, D + 0.03) / D + H.GROOVE_BELOW_SITE
            z_over = float(min(Z_OVER_MAX, max(Z_OVER, need)))
            r = yield from self.span_move(st, anchor, None, d0, self.z(z_over), psi, slack=0.004, vmax=0.02, ds=0.02)
            r = yield from self.tension_to(st, anchor, None, d0, r, self.z(z_over), psi, 2.5, r_min=D + 0.02, max_steps=40, band=1.0)
            self.sub = "over"
            r = yield from self.span_move(st, anchor, None, aim, self.z(z_over), psi, slack=0.0, r=r, vmax=0.012)
            ev["over_clear_mm"] = round(float(self.cable_h(self.z(z_over)) * D / max(r, 1e-6) - (H.POST_H + 0.006 + self.R)) * 1000, 1)
            ev["z_over_mm"] = round(z_over * 1000, 1)
        # the span passes the post on its outer side, 3 cm clear: slack on the way, taut there, then down
        self.sub = "travel"
        r = yield from self.span_move(st, anchor, None, aim, self.z(Z_SWEEP), psi, slack=0.008, vmax=0.025, ds=0.022)
        self.sub = "taut_high"
        r = yield from self.tension_to(st, anchor, None, aim, r, self.z(Z_SWEEP), psi, T_SPAN, r_min=D + 0.02)
        self.sub = "lower"
        r = yield from self.span_move(st, anchor, None, aim, self.z(Z_LAY), psi, slack=0.002, r=r + 0.02, vmax=0.008)
        r = yield from self.tension_to(st, anchor, None, aim, r, self.z(Z_LAY), psi, T_SPAN, r_min=D + 0.02, max_steps=40)
        self.sub = "arc"
        # round the post: from here the cable bears on the post at the passage point, so the hand goes on an arc
        # about that point (the length from it stays the same) and the tension is kept by force
        p = np.array(st["obs"]["tcp"]["pos"])
        a0 = math.atan2(p[1] - target[1], p[0] - target[0])
        a1 = math.atan2(d_out[1], d_out[0])
        da = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
        rad = float(np.linalg.norm(p[:2] - target))
        n = max(4, int(abs(da) / 0.10))
        for k in range(1, n + 1):
            ang = a0 + da * k / n
            dirn = np.array([math.cos(ang), math.sin(ang)])
            psi_k = self.psi_dir(dirn)
            w = target + rad * dirn
            yield from self.move(st, [w[0], w[1], self.z(Z_LAY)], psi_k, tol=0.003, max_steps=10, vmax=0.006, guard=False)
            if (yield from self.guard(st, target, psi_k)):
                pn = np.array(st["obs"]["tcp"]["pos"]); rad = float(np.linalg.norm(pn[:2] - target))
            if self.tension(st) > T_SPAN + 2.0 or self.tension(st) < 1.0:
                save = self.free                          # about the post the length is kept by the arc itself:
                self.free = self.path_len(target, None, target + rad * dirn, self.z(Z_LAY)) + 0.001   # no geometric slack to run through
                rad = yield from self.tension_to(st, target, None, dirn, rad, self.z(Z_LAY), psi_k, T_SPAN, max_steps=12)
                self.free = save
        self.sub = "settle"
        yield from self.hold(st, 3)
        ev["tension"] = round(st["truth"]["tension_mean"], 1)
        ev["t"] = st["obs"]["t"]
        self.sub = "release"
        yield from self.release(st)
        self.sub = ""
        return True

    def run(self, st):
        truth = st["truth"]
        R = truth["radius"]
        self.psi0 = self.cur_psi(st["obs"])
        self.fz0 = st["obs"]["wrench"]["force"][2]
        conn, fx = self.fixtures(truth)
        self.fx = fx
        pts, info = self.passages(conn, fx, R)
        anchor = conn
        for i, f in enumerate(fx):
            if self.stop_after is not None and i >= self.stop_after:
                break
            if f["type"] == "clip":
                yield from self.do_clip(st, i, f, anchor)
            else:
                for _try in range(2):                 # a post is checked after the hand lets go and made again once
                    yield from self.do_post(st, i, f, anchor, pts[i], info[i])
                    if self.post_made(st["truth"], f):
                        break
            anchor = pts[i]
        yield from self.hold(st, 3)


def variant(name):
    class V(Fitter):
        pass
    V.name = name
    if name == "stop_after_1":
        V.stop_after = 1
    elif name == "stop_after_2":
        V.stop_after = 2
    elif name == "no_tension_watch":
        V.watch_tension = False
    elif name == "wrong_post_side":
        V.flip_sides = True
    elif name == "wrong_clip_orientation":
        V.flip_axes = True
    elif name == "nominal_positions":
        V.nominal = True
    elif name != "reference":
        raise SystemExit("unknown program " + name)
    return V


def drive(env, gen, st, limit=None):
    """Run a generator of actions on the environment; returns the number of control steps."""
    steps = 0
    try:
        a = next(gen)
        while True:
            obs, done = env.step(a)
            steps += 1
            if done or (limit is not None and steps >= limit):
                break
            a = gen.send((obs, env.truth()))
    except StopIteration:
        pass
    return steps


def run_episode(seed, cls, n_fixtures=None):
    env = H.HarnessEnv()
    t0 = time.time()
    obs = env.reset(seed, n_fixtures)
    prog = cls(env.job_sheet())
    st = {"obs": obs, "truth": env.truth()}
    steps = drive(env, prog.run(st), st)
    res = env.finish()
    res.update(seed=int(seed), steps=steps, wall_s=round(time.time() - t0, 1), kinds="".join(f["type"][0] for f in env.ep.fixtures),
               marks="".join("+" if c["ok"] else "-" for c in res["checks"]), program=cls.name, events=prog.events, peaks=getattr(prog, "peaks", {}), guards=getattr(prog, "guards", 0), phase_steps=getattr(prog, "phase_steps", {}))
    return res


def seeds_of(text):
    out = []
    for s in text.split(","):
        if "-" in s:
            lo, hi = s.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--mutant", default="reference")
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    rows = []
    for name in a.mutant.split(","):
        cls = variant(name)
        for seed in seeds_of(a.seeds):
            r = run_episode(seed, cls, a.n)
            rows.append(r)
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "checks"}), flush=True)
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
