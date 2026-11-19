"""Pantry jar refilling: the privileged reference controller.

It reads the simulator's truth (jar, lid and piece poses, the thread's state, the closing torque) and acts only
through the environment's action (`JarEnv.step`, both arms' 7-number operational-space steps). Per listed jar, in
the order of the row:
1. the left arm holds the jar's body from the front (+x approach), 45 mm above the counter (P1b);
2. the right arm unscrews the lid in regrasp cycles, lifts it and sets it down at the jar's lid spot (front right);
3. the right arm drops the listed pieces of the jar's kind in from 4 cm above the neck, one at a time;
4. the right arm fetches the lid, sets it on the neck, turns it clockwise until it catches, then in small steps until
   the closing torque reaches TORQUE_TARGET;
5. both hands let go and back off.

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
import jar_env  # noqa: E402
import jar_sim as S  # noqa: E402

Q7_LIMIT = 2.8973
HOLD_Z = 0.045
TORQUE_TARGET = 0.60              # N m: stop once the seat has reached it (the window is 0.5-2.5)
LID_GRASP_DZ = 0.0                # the lid's centre (a grip on its upper part let a freed lid squeeze out)
LID_SPOT_X = 0.34                 # the lid is set down at the front of the counter,
LID_SPOT_DY = -0.14               # to the right of its jar: clear of the left arm, which holds the jar from the front
SAFE_Z = S.TABLE_Z + 0.40         # the right hand's height for sideways moves
PIECE_OPEN = 0.1                  # grip while reaching for a piece: a 36 mm opening (a full one lands on the tray's rim)


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


class Done(Exception):
    """The episode ended (its step limit or an unstable simulation)."""


class Reference:
    def __init__(self, ep):
        self.ep = ep
        self.env = ep.env
        self.w = ep.env.w
        self.L, self.R = self.w.arms["l"], self.w.arms["r"]
        self.f_local = {}
        for s, a in self.w.arms.items():
            Rs = a.R()
            f = Rs.T @ (self.w.d.geom_xpos[a.pads[0]] - self.w.d.geom_xpos[a.pads[1]])
            f[2] = 0.0
            self.f_local[s] = f / np.linalg.norm(f)
        self.hold = None
        self.hold_int = np.zeros(3)
        self.log = []
        self.phase = "start"
        self.phase_log = []

    # --- one control step ---------------------------------------------------------------------
    def tick(self, r_target, r_R, r_grip, speed=0.02, rot_speed=0.15):
        a = self.R
        dp = np.asarray(r_target, float) - a.pos()
        n = np.linalg.norm(dp)
        if n > speed:
            dp *= speed / n
        dr = rotvec(r_R @ a.R().T)
        n = np.linalg.norm(dr)
        if n > rot_speed:
            dr *= rot_speed / n
        act = {"r": list(dp) + list(dr) + [r_grip], "cams": 0}
        if self.hold is not None:
            p, Rh, g, spd = self.hold
            e = p - self.L.pos()
            self.hold_int = np.clip(self.hold_int + 0.15 * e, -0.02, 0.02) if g > 0 else np.zeros(3)
            dpl = e + self.hold_int
            n = np.linalg.norm(dpl)
            if n > spd:
                dpl *= spd / n
            act["l"] = list(dpl) + list(np.clip(rotvec(Rh @ self.L.R().T), -0.15, 0.15)) + [g]
        else:
            act["l"] = [0.0] * 6 + [-1.0]
        if not self.phase_log or self.phase_log[-1][0] != self.phase:
            self.phase_log.append((self.phase, self.w.t))
        n_ev = len(self.w.events)
        _obs, done, _ = self.ep.step(act)
        for ev in self.w.events[n_ev:]:
            if ev[0] == "tipped":
                self.log.append({"tipped": ev[1], "t": ev[2], "phase": self.phase})
        if done:
            raise Done()

    def servo_r(self, target, R, grip, speed=0.02, tol=0.003, max_steps=150):
        R = self.R.R() if R is None else R
        for _ in range(max_steps):
            self.tick(target, R, grip, speed)
            if np.linalg.norm(np.asarray(target) - self.R.pos()) < tol and np.linalg.norm(rotvec(R @ self.R.R().T)) < 0.03:
                return True
        return False

    def servo_l(self, target, R, grip, speed=0.02, tol=0.003, max_steps=150):
        self.hold = (np.asarray(target, float), R, grip, speed)
        self.hold_int = np.zeros(3)
        rp, rR = self.R.pos(), self.R.R()
        for _ in range(max_steps):
            self.tick(rp, rR, -1)
            if np.linalg.norm(np.asarray(target) - self.L.pos()) < tol and np.linalg.norm(rotvec(R @ self.L.R().T)) < 0.03:
                return True
        return False

    def grasp_R(self, side, approach, finger):
        fl = self.f_local[side]
        A = np.column_stack([fl, np.cross([0.0, 0.0, 1.0], fl), [0.0, 0.0, 1.0]])
        Bm = np.column_stack([finger, np.cross(approach, finger), approach])
        return Bm @ A.T

    def top_R(self, yaw):
        return self.grasp_R("r", np.array([0.0, 0.0, -1.0]), np.array([math.cos(yaw), math.sin(yaw), 0.0]))

    def yaw_of(self):
        f = self.R.R() @ self.f_local["r"]
        return math.atan2(f[1], f[0])

    def nearest_yaw(self, yaw):
        """`yaw` or yaw + pi (the same grasp): the smaller turn from the hand's current finger yaw that keeps the
        wrist (q7) clear of its limit. With the hand pointing down q7 moves by minus the yaw's change (measured); the
        closer turn alone, after a lid's regrasp cycles had left q7 near a limit, pinned q7 there and the arm could
        not reach its targets for the rest of the jar."""
        cur = self.yaw_of()
        q7 = float(self.w.d.qpos[self.R.jq[6]])
        turns = [(y, (y - cur + math.pi) % (2 * math.pi) - math.pi) for y in (yaw, yaw + math.pi)]
        ok = [(y, d) for y, d in turns if abs(q7 - d) < Q7_LIMIT - 0.25]
        if ok:
            return min(ok, key=lambda t: abs(t[1]))[0]
        return min(turns, key=lambda t: abs(q7 - t[1]))[0]

    # --- the jar --------------------------------------------------------------------------------
    def hold_jar(self, j):
        jar = self.w.jar_pos(j)
        a, f = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        Rg = self.grasp_R("l", a, f)
        grasp = np.array([jar[0], jar[1], S.TABLE_Z + HOLD_Z])
        pre = grasp - a * 0.14
        self.servo_l(pre + [0, 0, 0.12], Rg, -1, max_steps=120)
        self.servo_l(pre, Rg, -1, max_steps=80)
        self.servo_l(grasp, Rg, -1, speed=0.01, max_steps=80)
        self.hold = (grasp, Rg, 1, 0.02)
        for _ in range(10):
            self.tick(self.R.pos(), self.R.R(), -1)

    def let_go_jar(self):
        p, Rg, _g, _s = self.hold
        self.hold = (p, Rg, -1, 0.02)
        for _ in range(6):
            self.tick(self.R.pos(), self.R.R(), -1)
        self.servo_l(p - np.array([0.14, 0, 0]), Rg, -1, max_steps=40)
        self.servo_l(p - np.array([0.14, 0, -0.15]), Rg, -1, max_steps=40)

    def axis_point(self, j):
        """The hand's target while it turns the lid: the lid's centre on the jar's neck axis, at the lid's height.
        A free lid in the hand is shifted by its offset in the grasp (it follows the hand, so its own position is no
        target); an engaged lid sits on the axis."""
        # a lid in the thread: on its own axis (the hand, kept upright by turn_R, straightens a leaning jar; a target
        # over the jar's base put the hand's turning axis off the lid's and the fingers squeezed the lid out sideways);
        # a free lid: over the neck, where it must sit to catch
        lid = self.w.lid_pos(j)
        top = lid if self.w.engaged[j] else self.w.neck_top(j)
        off = np.zeros(2)
        if not self.w.engaged[j] and self.R.gripper() < 0.075 and np.linalg.norm(self.R.pos() - lid) < 0.03:
            off = np.clip(self.R.pos()[:2] - lid[:2], -0.015, 0.015)
        return np.array([top[0] + off[0], top[1] + off[1], lid[2] + LID_GRASP_DZ])

    def lid_on_neck(self, j):
        """The free lid resting on its jar's neck (centred within 6 mm, its bottom within 6 mm of the neck's top)."""
        w = self.w
        if w.engaged[j]:
            return True
        jar = self.env.unit["jars"][j]
        p = w.jar_R(j).T @ (w.lid_pos(j) - w.jar_pos(j))
        top = jar["jar_h"] + jar["turns"] * S.PITCH
        return bool(np.hypot(p[0], p[1]) < 0.006 and abs(p[2] - S.LID_H / 2 - top) < 0.006)

    def rot_step(self, j, step_rot, fine):
        """The turn per step: full, or 0.15 of it near the seat (`fine`): 0.018 rad per step seats a lid at about
        1-2 N m (the seat is stiff: about 67 N m per rad of the hand's lead over the lid)."""
        if fine is None or not fine(self.w):
            return step_rot
        return step_rot * 0.15

    def turn_R(self, direction, rot):
        """The hand's target while it turns a lid: upright (fingers horizontal, approach straight down) at its current
        yaw plus `direction * rot`. A turn sent as a bare rotation about z let a tilt build up: the hand followed a
        leaning jar's lid and the jar leaned further in the left grasp."""
        return self.top_R(self.yaw_of() + direction * rot)

    def wind(self, j, direction, grasp_z, stop, press=0.002, step_rot=0.12, max_cycles=16, fine=None, in_hand=False,
             abort=None, keep=False):
        """Regrasp cycles turning the lid about world z (+1 = counter-clockwise, loosening); `fine(w)` True makes the
        turn steps small (seating). `in_hand`: the lid is already gripped, the first cycle turns at once."""
        w = self.w
        abort = abort or (lambda w: False)
        for cyc in range(max_cycles):
            if stop(w):
                return cyc
            if abort(w):
                return -1
            if in_hand and cyc == 0:
                q_start = float(w.d.qpos[self.R.jq[6]])
                for _ in range(120):
                    lid = self.axis_point(j)
                    rot = self.rot_step(j, step_rot, fine)
                    self.tick(np.array([lid[0], lid[1], lid[2] - press]), self.turn_R(direction, rot), 1)
                    if stop(w) or abort(w):
                        break
                    q = float(w.d.qpos[self.R.jq[6]])
                    if abs(q) > Q7_LIMIT - 0.3 and abs(q - q_start) > 0.3:
                        break
                for _ in range(5):
                    self.tick(self.R.pos(), self.R.R(), -1)
                continue
            lid = self.axis_point(j)
            above = np.array([lid[0], lid[1], lid[2] + 0.04])     # the lid's height now: it rises as it unscrews
            self.servo_r(above, self.top_R(self.yaw_of()), -1, max_steps=40)
            q_start = float(w.d.qpos[self.R.jq[6]])
            for _ in range(60):
                q = float(w.d.qpos[self.R.jq[6]])
                if abs(q) > Q7_LIMIT - 0.35 and abs(q - q_start) > 0.5:
                    break
                self.tick(above, self.turn_R(-direction, step_rot * 1.5), -1)
            Rg = self.top_R(self.yaw_of())
            lid = self.axis_point(j)
            if abort(w):
                return -1
            self.servo_r(np.array([lid[0], lid[1], lid[2]]), Rg, -1, speed=0.015, max_steps=60)
            for _ in range(6):
                self.tick(np.array([lid[0], lid[1], lid[2] - press]), Rg, 1)
            q_start = float(w.d.qpos[self.R.jq[6]])
            for _ in range(120):
                lid = self.axis_point(j)
                rot = self.rot_step(j, step_rot, fine)
                self.tick(np.array([lid[0], lid[1], lid[2] - press]), self.turn_R(direction, rot), 1)
                if stop(w) or abort(w):
                    break
                q = float(w.d.qpos[self.R.jq[6]])
                if abs(q) > Q7_LIMIT - 0.3 and abs(q - q_start) > 0.5:
                    break
            if stop(w):
                if not keep:                            # keep: the hand stays closed on the lid (it has come free)
                    for _ in range(4):
                        self.tick(self.R.pos(), self.R.R(), -1)
                return cyc + 1
            for _ in range(5):
                self.tick(self.R.pos(), self.R.R(), -1)
        return max_cycles

    def transit_r(self, xy, z, R, grip):
        """The right hand to (xy, z): up to SAFE_Z first and down last when the move is more than a few cm sideways
        (a straight line crosses the left arm, which holds a jar low from the front)."""
        p = self.R.pos()
        side = np.linalg.norm(p[:2] - np.asarray(xy))
        if side > 0.04:
            self.servo_r(np.array([p[0], p[1], max(p[2], SAFE_Z)]), self.R.R(), grip, max_steps=60, tol=0.01)
            self.servo_r(np.array([xy[0], xy[1], SAFE_Z]), R, grip, speed=0.03, max_steps=150, tol=0.01)
        elif side > 0.01:                       # a short move: 5 cm up first (a slanted path from a jar's neck down to
            up = max(z, p[2] + 0.05)            # a lid beside it dragged the fingers across the neck)
            self.servo_r(np.array([p[0], p[1], up]), self.R.R(), grip, max_steps=40, tol=0.005)
            self.servo_r(np.array([xy[0], xy[1], up]), R, grip, max_steps=60, tol=0.005)
        self.servo_r(np.array([xy[0], xy[1], z]), R, grip, max_steps=120)

    def pick_place(self, body, grasp_yaw, grasp_z, place, place_z, place_yaw=None, lift=0.10, release_steps=5,
                   open_grip=-1.0, align=False, check=None):
        """Top-down pick of `body` at `grasp_z`, carried over `place` (xy) and let go at `place_z`. `check()`, if
        given, runs once the body is lifted: False puts it back down where it was picked (returns False)."""
        p = self.w.d.xpos[body].copy()
        R = self.top_R(self.nearest_yaw(grasp_yaw))
        self.transit_r(p[:2], grasp_z + 0.08, R, open_grip)
        self.servo_r(np.array([p[0], p[1], grasp_z]), R, open_grip, speed=0.01, max_steps=60)
        for _ in range(8):
            self.tick(np.array([p[0], p[1], grasp_z]), R, 1)
        self.servo_r(np.array([p[0], p[1], grasp_z + lift]), R, 1, max_steps=40)
        if check is not None and not check():
            self.servo_r(np.array([p[0], p[1], grasp_z + 0.01]), R, 1, speed=0.01, max_steps=60)
            for _ in range(5):
                self.tick(self.R.pos(), R, open_grip)
            self.servo_r(np.array([p[0], p[1], grasp_z + 0.08]), R, open_grip, max_steps=40)
            return False
        R2 = R if place_yaw is None else self.top_R(self.nearest_yaw(place_yaw))
        self.transit_r(place, place_z + 0.06, R2, 1)
        if align:                                   # put the body's centre (not the hand's) over `place`
            off = self.R.pos()[:2] - self.w.d.xpos[body][:2]
            place = np.asarray(place, float)[:2] + off
            self.servo_r(np.array([place[0], place[1], place_z + 0.03]), R2, 1, speed=0.01, max_steps=40)
        self.servo_r(np.array([place[0], place[1], place_z]), R2, 1, speed=0.01, max_steps=80)
        for _ in range(release_steps):              # open only as wide as for the pick: wide fingers hit the neck
            self.tick(np.array([place[0], place[1], place_z]), R2, open_grip)
        if release_steps:
            self.servo_r(np.array([place[0], place[1], place_z + 0.06]), R2, open_grip, max_steps=40)
        return True

    def refill_jar(self, j):
        w, ep = self.w, self.env.episode
        kind = self.env.unit["jars"][j]["kind"]
        need = ep["refill"][kind]
        # 1-2: hold, open, set the lid down
        self.phase = f"hold {j}"
        self.hold_jar(j)
        self.phase = f"open {j}"
        self.wind(j, +1, w.lid_pos(j)[2], lambda w: not w.engaged[j], keep=True)
        self.phase = f"lid down {j}"
        spot = np.array([LID_SPOT_X, w.jar_start[j][1] + LID_SPOT_DY])
        lid_body = w.lidf[j]
        lid = w.d.xpos[lid_body].copy()
        self.servo_r(lid + [0, 0, 0.08], self.R.R(), 1, max_steps=40)
        R = self.top_R(self.nearest_yaw(0.0))            # fingers along x: the hand clear of the left wrist
        self.transit_r(spot, S.TABLE_Z + S.LID_H / 2 + 0.06, R, 1)
        self.servo_r(np.array([spot[0], spot[1], S.TABLE_Z + S.LID_H / 2 + 0.004]), R, 1, speed=0.01, max_steps=60)
        for _ in range(5):
            self.tick(self.R.pos(), R, -1)
        self.servo_r(self.R.pos() + [0, 0, 0.08], R, -1, max_steps=30)
        # 3: the pieces
        self.phase = f"pieces {j}"
        for _ in range(need + 4):
            have = sum(1 for i in range(len(w.pieces)) if w.piece_in_jar(i) == j and ep["pieces"][i]["kind"] == kind)
            if have >= need:
                break
            in_tray = [i for i in range(len(w.pieces)) if w.piece_in_tray(i)]
            cand = [i for i in in_tray if ep["pieces"][i]["kind"] == kind]
            if not cand:
                break
            # the piece with the most room around it: a neighbour inside the fingers' span came up with it
            i = max(cand, key=lambda i: min([np.linalg.norm(w.d.xpos[w.pieces[i]][:2] - w.d.xpos[w.pieces[k]][:2])
                                             for k in in_tray if k != i] or [1.0]))
            b = w.pieces[i]
            Rp = w.d.xmat[b].reshape(3, 3)
            yaw = math.atan2(Rp[1, 0], Rp[0, 0]) + math.pi / 2          # fingers across the piece's short side
            if ep["pieces"][i]["kind"] == "sugar":                     # a cube: either side; fingers nearest to x
                yaw = min((yaw, yaw + math.pi / 2), key=lambda a: abs(math.sin(a)))
            top = w.neck_top(j)
            alone = lambda i=i: not any(                                               # noqa: E731
                w.d.xpos[w.pieces[k]][2] > S.TABLE_Z + 0.03 and w.piece_in_jar(k) is None
                and np.linalg.norm(w.d.xpos[w.pieces[k]] - self.R.pos()) < 0.06 for k in range(len(w.pieces)) if k != i)
            self.pick_place(b, yaw, w.d.xpos[b][2], top[:2], top[2] + 0.04, open_grip=PIECE_OPEN, check=alone)
        # 4: the lid back on, caught and seated to the torque target; a lid off the neck is picked up and put back
        stop = lambda w: w.engaged[j] and w.seated(j) and w.seat_peak[j] >= TORQUE_TARGET   # noqa: E731
        fine = lambda w: w.engaged[j] and w.turn(j) < 0.6                                  # noqa: E731
        off_neck = lambda w: not self.lid_on_neck(j)                                       # noqa: E731
        self.phase = f"close {j}"
        for _attempt in range(3):
            lid = w.d.xpos[w.lid_body(j)].copy()
            if not w.engaged[j] and lid[2] < S.TABLE_Z - 0.01:
                break                                           # the lid is on the floor
            top = w.neck_top(j)
            place_z = top[2] + S.LID_H / 2 + 0.002 + LID_GRASP_DZ          # the hand's height: the lid 2 mm over the neck
            if not w.engaged[j]:
                # fingers across the line from the jar to the lid: the hand's body (10.4 cm either side of the fingers)
                # along that line hit the jar when the lid lay beside it
                dxy = lid[:2] - w.jar_pos(j)[:2]
                yaw = math.atan2(dxy[1], dxy[0]) + math.pi / 2 if np.linalg.norm(dxy) > 0.02 else 0.0
                self.pick_place(lid_body, yaw, lid[2] + LID_GRASP_DZ, top[:2], place_z, lift=0.06, release_steps=0,
                                align=True)
            if self.wind(j, -1, place_z, stop, fine=fine, max_cycles=20, in_hand=not w.engaged[j], abort=off_neck) >= 0:
                break
            self.servo_r(self.R.pos() + [0, 0, 0.06], self.R.R(), -1, max_steps=30)
        self.servo_r(self.R.pos() + [0, 0, 0.10], self.R.R(), -1, max_steps=30)
        # 5: let go
        self.phase = f"let go {j}"
        self.let_go_jar()

    def finish(self):
        """Hands up and open until the episode ends."""
        self.servo_r(self.R.pos() + [0, 0, 0.15], self.R.R(), -1, max_steps=30)
        self.hold = None
        while True:
            self.tick(self.R.pos(), self.R.R(), -1)

    def run(self):
        w, ep = self.w, self.env.episode
        order = sorted(ep["listed"], key=lambda j: -w.jar_start[j][1])     # the row from the left
        try:
            for j in order:
                self.refill_jar(j)
            self.finish()
        except Done:
            pass
        return self.ep.score()


def run_one(unit_seed, seed, render=False):
    unit = S.draw_unit(unit_seed)
    env = jar_env.JarEnv(unit, render=render)
    ep = E.Episode(env, seed)
    ep.reset()
    t0 = time.time()
    ref = Reference(ep)
    sc = ref.run()
    sc.update(unit_seed=unit_seed, seed=seed, wall=round(time.time() - t0, 1),
              steps_per_s=round(env.t / max(1e-9, time.time() - t0), 1), events=env.w.events, ref_log=ref.log,
              phases=ref.phase_log)
    return sc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(json.dumps(run_one(a.unit, a.seed)))


if __name__ == "__main__":
    main()
