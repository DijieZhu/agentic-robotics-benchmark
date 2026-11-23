"""Privileged reference pilot (token-gated): flies the rescue from the grader's ground truth.

The graded service adds a `_gt` block to every answer only when the submission carries the oracle
token whose SHA-256 the grader holds (tests/oracle_token.sha256). `_gt` gives the aircraft's true
pose, the litter's pose and the site's layout; this pilot plans on that and closes its position loop
on the true pose. It reads no image. It proves that every graded site can be completed inside the
stated budgets; it is not a solution an agent could write, and it is not used to judge one.

`MUTANT` (environment variable `PILOT_MUTANT`, used only by the review's score table) degrades one
part of the plan: see MUTANTS below.
"""
from collections import deque
import math
import os
import sys

import numpy as np

sys.path.insert(0, "/opt/visual_robotics/client")

INC = np.array([.32, .32, .19])
TURN = math.radians(30)
DECK, CRUISE, SLOT_Z = 2.60, 1.30, .725
MUTANTS = ("stop_after_entry", "stop_upstairs", "stop_over_litter", "stop_after_grasp", "stop_after_lift",
           "stop_after_egress", "wrong_tool", "wrong_station", "no_board_alignment", "release_high",
           "collet_fixed", "collet_from_down_view", "collet_from_forward_view",
           "pose_noise_10cm", "pose_noise_25cm")


def wrap(a):
    return (a+math.pi) % (2*math.pi)-math.pi


def wrap_half(a):
    """Shortest turn that brings a line (not a vector) onto a direction."""
    return (a+math.pi/2) % math.pi-math.pi/2


class Stop(Exception):
    pass


class Pilot:
    def __init__(self, client, mutant=None):
        self.client, self.mutant = client, mutant
        self.rng = np.random.default_rng(7)
        self.obs = client.observe()
        if "_gt" not in self.obs:
            raise SystemExit("no ground truth in the answer: the oracle token did not match")
        self.layout = self.obs["_gt"]["layout"]
        self.loaded = False

    # --- state ------------------------------------------------------------------------------- #
    @property
    def gt(self):
        return self.obs["_gt"]

    def pose(self):
        p = np.array(self.gt["pose"])
        if self.mutant in ("pose_noise_10cm", "pose_noise_25cm"):
            p[:2] += self.rng.normal(0., .10 if self.mutant == "pose_noise_10cm" else .25, 2)
        return p

    def act(self, right=0., forward=0., up=0., turn=0., grip=0.):
        if self.obs["done"]:
            raise Stop("flight over")
        self.obs = self.client.act([float(np.clip(v, -1, 1)) for v in (right, forward, up, turn, grip)])
        return self.obs

    def hold(self, steps, grip=0.):
        for _ in range(steps):
            self.act(grip=grip)

    # --- motion ------------------------------------------------------------------------------ #
    def goto(self, x=None, y=None, z=None, yaw=None, speed=1., tol=.04, settle=.08):
        """Close the loop on the true pose until within `tol` and slower than `settle` m/s."""
        for _ in range(600):
            p, v = self.pose(), np.array(self.gt["velocity"])
            ahead = p+.15*v                       # where the servo will have brought it
            target = np.array([p[0] if x is None else x, p[1] if y is None else y,
                               p[2] if z is None else z, p[3] if yaw is None else yaw])
            err = target-ahead
            err[3] = wrap(target[3]-ahead[3])
            now = target-p
            now[3] = wrap(target[3]-p[3])
            if (np.linalg.norm(now[:3]) < tol and abs(now[3]) < .03
                    and np.linalg.norm(v[:3]) < settle and abs(v[3]) < .1):
                return
            c, s = math.cos(p[3]), math.sin(p[3])
            body = np.array([c*err[0]+s*err[1], -s*err[0]+c*err[1]])
            u = .7*np.array([*body, err[2]])/INC
            u[:2] *= min(1., speed/max(1e-9, np.abs(u[:2]).max()))   # keep the horizontal direction
            self.act(u[0], u[1], np.clip(u[2], -speed, speed), np.clip(.7*err[3]/TURN, -1, 1))
        raise Stop("could not reach a waypoint")

    def board_yaw(self):
        return self.gt["litter"][3]+math.pi/2     # direction of the board's long axis

    def align_board(self, direction):
        """Turn so that the carried board's long axis lies along `direction` (radians, world)."""
        if self.mutant == "no_board_alignment":
            return
        for _ in range(4):
            turn = wrap_half(direction-self.board_yaw())
            if abs(turn) < .03:
                return
            self.goto(yaw=self.pose()[3]+turn, speed=.5)

    # --- the building ------------------------------------------------------------------------ #
    def xy(self, cell):
        return np.array([self.layout["columns"][cell[0]], self.layout["rows"][cell[1]]])

    def route(self, start, goal, wells):
        """Cells from start to goal over the open doorways and the given hoist wells."""
        links = {}
        for a, b in self.layout["halls"]:
            a, b = tuple(a), tuple(b)
            if a[2] != b[2] and [a[0], a[1]] not in wells:
                continue
            links.setdefault(a, []).append(b)
            links.setdefault(b, []).append(a)
        seen, queue = {start: None}, deque([start])
        while queue:
            a = queue.popleft()
            if a == goal:
                break
            for b in links.get(a, ()):
                if b not in seen:
                    seen[b] = a
                    queue.append(b)
        path = [goal]
        while path[-1] != start:
            path.append(seen[path[-1]])
        return path[::-1]

    def fly_cells(self, path, speed):
        for a, b in zip(path, path[1:]):
            if a[2] != b[2]:
                self.cross_well(a, b, speed)
                continue
            base = a[2]*DECK
            pa, pb = self.xy(a), self.xy(b)
            if self.loaded:
                self.align_board(math.atan2(pb[1]-pa[1], pb[0]-pa[0]))
            mid = (pa+pb)/2
            self.goto(mid[0], mid[1], base+CRUISE, speed=speed, tol=.10, settle=1.)
            self.goto(pb[0], pb[1], base+CRUISE, speed=speed, tol=.06)

    def cross_well(self, a, b, speed):
        """(col, 3, level) -> the shaft behind the amber threshold -> (col, 3, other level)."""
        x, y = self.xy(a)
        shaft_y = self.layout["rows"][4]
        if self.loaded:
            self.align_board(math.pi/2)
        self.goto(x, shaft_y, a[2]*DECK+CRUISE, speed=speed, tol=.05)
        self.goto(x, shaft_y, b[2]*DECK+CRUISE, speed=speed, tol=.05)
        self.goto(x, y, b[2]*DECK+CRUISE, speed=speed, tol=.06)

    def checkpoint(self, name):
        if self.mutant == name:
            raise Stop(name)

    # --- the mission ------------------------------------------------------------------------- #
    def ring_width(self, camera):
        """the cyan ring's width in a picture, in pixels (0: not seen)"""
        import base64
        import io
        from PIL import Image
        im = np.asarray(Image.open(io.BytesIO(base64.b64decode(self.obs["images"][camera]))).convert("RGB")).astype(int)
        mask = (im[..., 2]-im[..., 0] > 50) & (im[..., 1]-im[..., 0] > 40) & (im[..., 2] > 90)
        cols = np.flatnonzero(mask.sum(0) >= 2)
        return 0 if len(cols) == 0 else int(cols[-1]-cols[0]+1)

    def scout_ring(self):
        L = self.layout
        inlet = L["inlet"]
        slot_x, door_y = L["columns"][inlet], (L["rows"][0]+L["rows"][1])/2
        self.goto(slot_x, 0., CRUISE, yaw=0.)
        self.goto(slot_x, 0., SLOT_Z, yaw=0., tol=.02, settle=.05)
        self.goto(slot_x, door_y+.80, SLOT_Z, yaw=0., speed=.5, tol=.03)
        first = (inlet, 1, 0)
        self.goto(*self.xy(first), CRUISE)
        target = tuple(L["target"])
        path = self.route(first, target, L["wells"])
        self.fly_cells(path, 1.)
        litter = self.gt["litter"]
        floor = target[2]*DECK
        centre = np.array(self.xy(target))
        away = centre-np.array(litter[:2])
        dist = float(np.linalg.norm(away))
        heading = math.atan2(-(litter[0]-centre[0]), litter[1]-centre[1]) if dist > .05 else 0.
        f = np.array([-math.sin(heading), math.cos(heading)])       # the aircraft's forward axis
        if self.mutant == "collet_from_down_view":                    # the down camera (0.46 m ahead of the centre) over the ring
            at = np.array(litter[:2])-.46*f
            self.goto(at[0], at[1], floor+CRUISE, yaw=heading, tol=.03, settle=.05)
            width, limit = self.ring_width(0), 119.4*.134/(CRUISE-.145-.26)
        else:                                                         # the forward camera, up to 1.5 m before the litter
            back = min(1.5, dist+.85)                                 # inside the room's turning circle, at cruise height
            at = np.array(litter[:2])-back*f
            self.goto(at[0], at[1], floor+CRUISE, yaw=heading, tol=.03, settle=.05)
            width, limit = self.ring_width(1), 119.4*.134/math.hypot(back-.17, CRUISE+.045-.24)
        self.ring_seen = (width, round(limit, 1))
        self.goto(*centre, floor+CRUISE)
        self.fly_cells(path[::-1], 1.)
        self.goto(*self.xy(first), SLOT_Z, yaw=0.)
        self.goto(slot_x, door_y+.80, SLOT_Z, yaw=0., tol=.03)
        self.goto(slot_x, 0., SLOT_Z, yaw=0., speed=.5, tol=.03)
        self.goto(slot_x, 0., CRUISE, yaw=0.)
        return width > limit

    def fly(self):
        L = self.layout
        inlet, outlet = L["inlet"], L["outlet"]
        slot_x, door_y = L["columns"][inlet], (L["rows"][0]+L["rows"][1])/2
        # 1. the collet that fits the flange (truth), from its stand in the staging strip
        fits = [i for i, (throat, seat) in enumerate(L["depot"]) if throat < L["flange"] < seat]
        tool = fits[0] if self.mutant != "wrong_tool" else 1-fits[0]
        if self.mutant == "collet_fixed":              # always the collet on the first stand
            tool = 0
        if self.mutant in ("collet_from_down_view", "collet_from_forward_view"):
            # the collet is chosen from a picture of the receiver's ring instead of the truth: a first flight to the
            # litter without a collet, one picture, back out, then the plan as usual
            wide = self.scout_ring()
            big = max(range(2), key=lambda i: L["depot"][i][1])
            tool = big if wide else 1-big
        dock = L["docks"][tool]
        self.goto(dock[0], dock[1], CRUISE)
        self.goto(dock[0], dock[1], dock[2]+.206, speed=.5, tol=.02, settle=.05)
        self.act(grip=1.)
        self.goto(dock[0], dock[1], CRUISE)
        # 2. in through the inspection slot
        self.goto(slot_x, 0., CRUISE, yaw=0.)
        self.goto(slot_x, 0., SLOT_Z, yaw=0., tol=.02, settle=.05)
        self.goto(slot_x, door_y+.80, SLOT_Z, yaw=0., speed=.5, tol=.03)
        first = (inlet, 1, 0)
        self.goto(*self.xy(first), CRUISE)
        self.checkpoint("stop_after_entry")
        # 3. up a hoist well and to the litter
        target = tuple(L["target"])
        path = self.route(first, target, L["wells"])
        up = next(i for i, (a, b) in enumerate(zip(path, path[1:])) if a[2] != b[2])
        self.fly_cells(path[:up+2], 1.)
        self.checkpoint("stop_upstairs")
        self.fly_cells(path[up+1:], 1.)
        litter = self.gt["litter"]
        floor = target[2]*DECK
        self.goto(litter[0], litter[1], floor+CRUISE)
        self.checkpoint("stop_over_litter")
        self.goto(litter[0], litter[1], litter[2]+.306, speed=.5, tol=.02, settle=.05)
        self.act(grip=1.)
        if not self.gt["held"]:
            raise Stop("the collet did not latch")
        self.loaded = True
        self.checkpoint("stop_after_grasp")
        self.goto(z=floor+CRUISE, speed=.6)
        self.checkpoint("stop_after_lift")
        self.goto(*self.xy(target), floor+CRUISE, speed=.6)
        # 4. down the freight well (the only bore the board passes) and out under the cargo shutter
        freight = [w for w in L["wells"] if L["well_half"][str(w[0])] > 1.][0]
        self.fly_cells(self.route(target, (outlet, 1, 0), [freight]), .6)
        self.align_board(math.pi/2)
        cx = L["columns"][outlet]
        self.goto(cx, 0., CRUISE, speed=.6)
        self.checkpoint("stop_after_egress")
        # 5. through the transfer throat, across the turning pocket, to the open receiving station
        entry, side, stagger = L["loading"]
        if self.mutant == "wrong_station":
            side = -side
        px, py = L["pocket"]
        lane = py+side*stagger
        self.goto(cx+entry, 0., CRUISE, speed=.6)
        self.goto(cx+entry, lane, CRUISE, speed=.35, tol=.05)     # slow through the 1.10 m throats
        self.align_board(0.)
        pad = L["pads"][str(int(side))]
        self.goto(pad[0], lane, CRUISE, speed=.35, tol=.05)
        self.goto(pad[0], pad[1], CRUISE, speed=.6, tol=.03)
        self.align_board(0.)
        if self.mutant != "release_high":
            self.goto(pad[0], pad[1], .47, speed=.5, tol=.02, settle=.05)
        self.act(grip=-1.)
        self.hold(12)


def main():
    from client import Client
    pilot = Pilot(Client(os.environ.get("ROBOT_SOCKET", "/run/visual_robotics/world.sock")),
                  os.environ.get("PILOT_MUTANT") or None)
    try:
        pilot.fly()
    except Stop as reason:
        print("stopped:", reason, flush=True)


if __name__ == "__main__":
    main()
