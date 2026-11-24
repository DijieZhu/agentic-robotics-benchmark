"""Shelf restocking: the privileged reference for a whole episode (true poses).

The moves are `staging.online_plan`'s: a strategy that decides each move only from what it has seen by then (it
reads a unit's code once the unit is out of its lane). The lone lane first if the tote takes all its units; on the
shelf with two served lanes, tote units are first set down in front of one lane (other lanes' units deepest, then
the first lane's own delivered units, earliest code first) so the tote has room to empty the other lane, which is
then rebuilt back to front in code order (the latest deepest) from the tote and from the staging lane's front; then
the staging lane, then the lone lane if it waited. Expired units and extras stay in the tote. Perception is
privileged: true poses, true identities. Every grasp is the same:
the hand pointing down, the fingers across the lane (across a box's width), 22 mm under the unit's top; inside
the bay the hand moves level (in and out at the grasp height, +28 mm while carrying), in the tote it comes from
above. The lift brings the worked board 0.20 m above the base, and the tote's work happens with the base at
0.60 m; between the two the base travels with the hand in front of the bay. A box in the tote facing the wrong
way is grasped with the hand turned by -pi and turned back while it is carried, so it goes in face out.

    ref = Reference(world); ref.run() -> {"status", "moves", "failed"}
"""

import math

import numpy as np

import shelf_sim as S
import shelf_world as W
import staging as G
from shelf_ctl import GRIP_DEPTH, LIFT, down_R

OUT = 0.05                     # m, the entry point's distance in front of the lip (bay frame -x)
TRANSIT_CLEAR = 0.03           # m, a carried unit's bottom over the tote's units' tops


def hand_R(yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    return Rz @ down_R("y")


class Reference:
    def __init__(self, w, log=None):
        self.w = w
        self.kin = S.Kin(w.model)
        self.bay_yaw = w.bay["yaw"]
        self.R = hand_R(self.bay_yaw)
        self.yaw_now = self.bay_yaw
        self.moves = 0
        self.failed = []
        self.log = log or (lambda *a: None)

    # --- motion primitives ---------------------------------------------------------------------------
    def go(self, target, grip, speed=0.02, tol=0.004, max_steps=200, final=0.03):
        w = self.w
        target = np.asarray(target, float)
        self.kin.lift_z = w.lift()
        q, ok_ik = self.kin.ik(target, self.R, w.data.qpos[w.jq].copy())
        if ok_ik:
            w.q_null = q
        dist = np.linalg.norm(target - w.ee())
        if dist > final + 0.005:
            w.servo(w.ee() + (target - w.ee()) * (1 - final / dist), self.R, grip, speed=speed, tol=0.01,
                    max_steps=max_steps)
        return w.servo(target, self.R, grip, speed=min(speed, max(0.004, speed / 3)), tol=tol, max_steps=max_steps)

    def turn(self, yaw, grip):
        """Turn the hand to `yaw` in steps of at most pi/3 in the direction given (no shortest-path ambiguity)."""
        while abs(yaw - self.yaw_now) > 1e-6:
            step = max(-math.pi / 3, min(math.pi / 3, yaw - self.yaw_now))
            self.yaw_now += step
            self.R = hand_R(self.yaw_now)
            self.w.servo(self.w.ee(), self.R, grip, speed=0.01, tol=0.01, max_steps=40)

    def set_yaw(self, yaw):
        self.yaw_now = yaw
        self.R = hand_R(yaw)

    def bayw(self, x, y, z):
        return W.bay_to_world(self.w.bay, x, y, z)

    def shelf_base(self, si):
        return self.w.unit["shelves"][si]["board"] - W.LIFT_BELOW_BOARD

    def lift_to(self, z, grip):
        return self.w.lift_to(z, grip, max_steps=300)

    # --- moves ---------------------------------------------------------------------------------------
    def top_of(self, n):
        sku = self.w.unit["skus"][self.w.sku_of[n]]
        return float(self.w.pos(n)[2] + W.half_h(sku))

    def pick_shelf(self, n, si):
        """Grasp lane unit n (the lane's frontmost) and lift it LIFT; returns the carry height (site z)."""
        w = self.w
        p = w.to_bay(w.pos(n))
        g = self.top_of(n) - GRIP_DEPTH
        # to the lane's entry first, at the height that becomes the grasp height once the lift has moved
        self.turn(self.bay_yaw, -1)
        ok = self.go([*self.bayw(-OUT - 0.03, p[1], 0)[:2], g - (self.shelf_base(si) - w.lift())], -1,
                     speed=0.03, tol=0.008)
        ok &= self.lift_to(self.shelf_base(si), -1)
        ok &= self.go([*self.bayw(-OUT, p[1], 0)[:2], g], -1, speed=0.03, tol=0.006)
        ok &= self.go([*self.bayw(p[0], p[1], 0)[:2], g], -1, speed=0.015, tol=0.003)
        w.hold(1, 8)
        ok &= w.grasped(w.bodies[n])
        z = g + LIFT
        ok &= self.go([*self.bayw(p[0], p[1], 0)[:2], z], 1, speed=0.01)
        ok &= self.go([*self.bayw(-OUT - 0.03, p[1], 0)[:2], z], 1, speed=0.02, tol=0.006)
        return ok, z

    def place_shelf(self, n, si, li, x):
        """Carry the held unit n to bay x in lane (si, li) and let go."""
        w = self.w
        if not w.grasped(w.bodies[n]):
            return False
        sku = w.unit["skus"][w.sku_of[n]]
        board = w.unit["shelves"][si]["board"]
        y = W.lane_y(w.unit, li)
        z_carry = board + 2 * W.half_h(sku) - GRIP_DEPTH + LIFT
        ok = self.go([*self.bayw(-OUT - 0.03, y, 0)[:2], z_carry - (self.shelf_base(si) - w.lift())], 1,
                     speed=0.025, tol=0.006)
        ok &= self.lift_to(self.shelf_base(si), 1)
        self.turn(self.bay_yaw, 1)
        ok &= self.go([*self.bayw(-OUT, y, 0)[:2], z_carry], 1, speed=0.02, tol=0.004)
        ok &= self.go([*self.bayw(x, y, 0)[:2], z_carry], 1, speed=0.015)
        z_place = board + 2 * W.half_h(sku) - GRIP_DEPTH + 0.002
        ok &= self.go([*self.bayw(x, y, 0)[:2], z_place], 1, speed=0.008, tol=0.003, max_steps=80)
        w.hold(-1, 10)
        self.go([*self.bayw(-OUT, y, 0)[:2], z_place], -1, speed=0.015, tol=0.008, max_steps=120)
        return ok

    def transit_z(self, n):
        """Site height over the tote for carrying unit n above the tote's units."""
        sku = self.w.unit["skus"][self.w.sku_of[n]]
        tallest = max(2 * W.half_h(s) for s in self.w.unit["skus"].values())
        return W.CART["top"] + tallest + TRANSIT_CLEAR + 2 * W.half_h(sku) - GRIP_DEPTH

    def free_tote_slot(self):
        occupied = [self.w.pos(n)[:2] for n in self.w.bodies]
        cx = W.TOTE["centre"][0]
        slots = sorted((np.array(W.tote_slot_xy(i, j)) for i in range(W.TOTE_GRID[0]) for j in range(W.TOTE_GRID[1])),
                       key=lambda xy: abs(xy[0] - cx))          # the middle columns first
        for xy in slots:
            if all(np.linalg.norm(xy - o) > 0.07 for o in occupied):
                return xy
        return None

    def place_tote(self, n):
        w = self.w
        if not w.grasped(w.bodies[n]):
            return False
        xy = self.free_tote_slot()
        if xy is None:
            return False
        sku = w.unit["skus"][w.sku_of[n]]
        zt = self.transit_z(n)
        ok = self.lift_to(W.TOTE_BASE, 1)
        ok &= self.go([xy[0], xy[1], zt], 1, speed=0.025, tol=0.006)
        z_place = W.CART["top"] + 2 * W.half_h(sku) - GRIP_DEPTH + 0.002
        ok &= self.go([xy[0], xy[1], z_place], 1, speed=0.01, tol=0.003, max_steps=100)
        w.hold(-1, 10)
        self.go([xy[0], xy[1], zt], -1, speed=0.02, tol=0.008, max_steps=80)
        return ok

    def pick_tote(self, n, face_out):
        """Grasp tote unit n from above; a box is grasped across its width, the hand turned so that turning back
        to the bay's yaw leaves it face out."""
        w = self.w
        sku = w.unit["skus"][w.sku_of[n]]
        self.lift_to(W.TOTE_BASE, -1)
        p = w.pos(n)
        if sku["kind"] == "box" and face_out:
            # the hand's yaw equal to the box's (mod 2 pi), a half turn taken as -pi; +pi only where -pi leaves
            # joint 7 within 0.15 rad of its limit (on the tote's far side -pi took it past the limit)
            off = (w.yaw(n) - self.bay_yaw + math.pi) % (2 * math.pi) - math.pi
            if off > math.pi / 2:
                off -= 2 * math.pi
            if off < -math.pi / 2:
                # joint 7's angle for this yaw, predicted from IK at the bay's yaw and 0.3 rad off it (IK straight to
                # a half turn often fails to converge and cannot tell a reachable yaw from one that is not)
                g = self.top_of(n) - GRIP_DEPTH
                self.kin.lift_z = w.lift()
                tgt = np.array([p[0], p[1], g])
                qa, oka = self.kin.ik(tgt, hand_R(self.bay_yaw), w.data.qpos[w.jq].copy())
                qb, okb = self.kin.ik(tgt, hand_R(self.bay_yaw + 0.3), qa)
                if oka and okb:
                    slope = (qb[6] - qa[6]) / 0.3
                    q7 = qa[6] + slope * off
                    lo, hi = self.kin.lo[6] + 0.15, self.kin.hi[6] - 0.15
                    if not lo <= q7 <= hi and lo <= qa[6] + slope * (off + 2 * math.pi) <= hi:
                        off += 2 * math.pi
            yaw = self.bay_yaw + off
        else:
            yaw = self.bay_yaw
        zt = self.transit_z(n)
        # turn over the tote's centre (the palm sweeps 0.104 m around the hand; near the bay it met the shelves)
        cx, cy = W.TOTE["centre"]
        ok = self.go([cx, cy, zt], -1, speed=0.03, tol=0.01)
        self.turn(yaw, -1)
        ok &= self.go([p[0], p[1], zt], -1, speed=0.03, tol=0.008)
        g = self.top_of(n) - GRIP_DEPTH
        ok &= self.go([p[0], p[1], self.top_of(n) + 0.02], -1, speed=0.02, tol=0.004)
        ok &= self.go([p[0], p[1], g], -1, speed=0.012, tol=0.003)
        w.hold(1, 8)
        ok &= w.grasped(w.bodies[n])
        ok &= self.go([p[0], p[1], zt], 1, speed=0.015, tol=0.006)
        # turn back to the bay's yaw over the tote's centre, not in front of the bay
        ok &= self.go([cx, cy, zt], 1, speed=0.02, tol=0.01)
        self.turn(self.bay_yaw, 1)
        return ok

    # --- the episode ---------------------------------------------------------------------------------
    @staticmethod
    def body(uid):
        """staging's unit id -> the world's body name (shelf_world.layout)."""
        if uid.startswith("t"):
            return "u_" + uid
        key, o = uid.split("/")
        si, li = key.split(":")
        return f"u_{si}_{li}_{o}"

    def run(self):
        w = self.w
        moves, why = G.online_plan(w.unit, w.episode)
        if moves is None:
            self.log(f"no plan: {why}")
            moves = []
        self.log(f"{len(moves)} moves")
        for kind, uid, src, dst, x in moves:
            n = self.body(uid)
            if kind == "tote_lane":
                si, li = map(int, dst.split(":"))
                ok = self.pick_tote(n, face_out=True)
                ok &= self.place_shelf(n, si, li, x)
            else:
                si, _li = map(int, src.split(":"))
                ok, _z = self.pick_shelf(n, si)
                if kind == "lane_tote":
                    ok &= self.place_tote(n)
                else:
                    di, dl = map(int, dst.split(":"))
                    ok &= self.place_shelf(n, di, dl, x)
            self.note(ok, kind, n)
        self.lift_to(W.TOTE_BASE, -1)
        self.set_yaw(self.bay_yaw)
        self.go(w.home_pos, -1, speed=0.03, tol=0.01)
        return {"status": "done", "moves": self.moves, "failed": self.failed}

    def note(self, ok, what, n):
        self.moves += 1
        if not ok:
            self.failed.append(f"{what} {n}")
        self.log(f"  {what} {n}: {'ok' if ok else 'NOT ok'} t={self.w.t}")
