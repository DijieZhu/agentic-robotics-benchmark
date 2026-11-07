"""Privileged reference expert for aerial_search_geolocation.

Graded with the oracle token (see tests/verify.py), every observation carries
`privileged`: the objects placed in the scene, by description, in the mission start frame,
and `clearance`, how far each move could go from where the drone is before it touches
anything. Decoys (asked descriptions no placed object has) are declared absent from the
start; a present target is reported, at its true position, only once the drone has been to
it and looked, since a target only counts once the drone has been there:

* the nearest target not yet visited is next; the drone climbs to a transit height above it,
  turns toward it and flies until it is over a hover point 2 m from the target, a metre
  or more off any surface, climbing over whatever is in the way and flying out from under
  whatever shuts the way up;
* it descends there into the visit window (within 3 m horizontally, at most 6 m above the
  target, the target in the down camera's view), holds there for a step with the down
  camera on it, reports it, climbs back to the transit height and goes on; a hover point
  it cannot get down to is swapped for the next one round the target, and a target with
  none it can reach is reported anyway (an unvisited report scores nothing, but costs nothing);
* a move is made only when its clearance leaves a margin; once every present target is
  visited, it stops.

Without the privileged block (no or a wrong token) it reports nothing and stops.
"""

import json
import math
import sys

STOP, FORWARD, LEFT, RIGHT, ASCEND, DESCEND, ROTL, ROTR = range(8)
MOVE_NAME = {FORWARD: "forward", LEFT: "left", RIGHT: "right", ASCEND: "ascend",
             DESCEND: "descend"}
FOUND, ABSENT = 1, 2
VISIT_RADIUS, VISIT_HEIGHT = 2.6, 5.5      # inside the grader's 3 m / 6 m, with margin
CAMERA_DROP = 1.3                           # the down camera hangs 1 m low (with margin)
HOLD_STEPS = 2                              # steps in the window before the report
ARRIVE = 0.7                                # over a hover point within this
TRANSIT = 12.0                              # transit at least this high above the next target
SAFE = 1.5                                  # a 1 m move needs this much clearance
AHEAD = 3.0                                 # travelling: climb once something is this close
FACE = math.radians(20.0)                   # turn toward the goal beyond this
NEAR = 4.0                                  # within this of a hover point: no more turning
OFF = 1.0                                   # travelling: keep this far above surfaces
ESCAPE_CLIMB = 10.0                         # out from under something: climb this much
# escape legs: (degrees from the goal's direction, steps), longer each round
ESCAPE_LEGS = [(t, n) for n in (4, 8, 16) for t in (90, -90, 180, 45, -45, 135, -135, 0)]
# hover points: (distance from the target, degrees from the drone's side of it)
SPOTS = [(1.8, a) for a in (0, 45, -45, 90, -90, 135, -135, 180)] + [(0.6, 0)]


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class Expert(object):
    """Flight state is kept per set of asked descriptions: the probe lock's second ask of a
    step carries other descriptions, and must not move the real mission's state on."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.states = {}

    def act(self, t, obs):
        truth = obs.get("privileged")
        if truth is None:
            return [STOP]
        where = {item["description"]: item["position"] for item in truth["objects"]}
        asked = [target["description"] for target in obs["targets"]]
        self.s = self.states.setdefault(tuple(asked), {
            "visited": set(), "seen": set(), "hold": {}, "spot": None, "base": 0.0,
            "climb_to": None, "escape": None})
        self.clear = truth.get("clearance") or {}
        move = self._move(obs, {d: where[d] for d in asked if d in where})
        reports = []
        for description in asked:
            if description not in where:
                reports.extend([ABSENT, 0.0, 0.0, 0.0])
            elif description in self.s["seen"]:
                reports.extend([FOUND] + list(where[description]))
            else:
                reports.extend([0, 0.0, 0.0, 0.0])
        return [move] + reports

    def _ok(self, move, need=SAFE):
        return self.clear.get(MOVE_NAME[move], 0.0) >= need

    # -- flying -------------------------------------------------------------------------
    def _move(self, obs, targets):
        if obs["steps_used"] >= obs["budget"] - 1:
            return STOP
        pos, yaw = obs["odom"]["position"], obs["odom"]["yaw"]
        for name, p in targets.items():
            near, above = math.hypot(p[0] - pos[0], p[1] - pos[1]), p[2] - pos[2]
            if near <= VISIT_RADIUS and above <= VISIT_HEIGHT and near <= above - CAMERA_DROP:
                if name not in self.s["visited"]:
                    held = self.s["hold"].get(name, 0) + 1
                    self.s["hold"][name] = held
                    if held < HOLD_STEPS:
                        return ROTR              # stay put with the target under the camera
                    self.s["visited"].add(name)
                    self.s["seen"].add(name)
                    self.s["spot"] = None
                    self.s["climb_to"] = p[2] - TRANSIT
        left = {n: p for n, p in targets.items() if n not in self.s["visited"]}
        if not left:
            return STOP
        if self.s["climb_to"] is not None:
            if pos[2] > self.s["climb_to"] + 0.5 and self._ok(ASCEND):
                return ASCEND
            self.s["climb_to"] = None
        name, target = min(left.items(), key=lambda item: math.hypot(
            item[1][0] - pos[0], item[1][1] - pos[1]))
        if self.s["spot"] is None or self.s["spot"][0] != name:
            self.s["spot"] = (name, 0)
            self.s["base"] = math.atan2(pos[1] - target[1], pos[0] - target[0])
        spot = self._spot(target, self.s["spot"][1])
        dx, dy = spot[0] - pos[0], spot[1] - pos[1]
        far = math.hypot(dx, dy)
        if far > ARRIVE:
            return self._level(yaw, dx, dy, far, target, pos)
        # over the hover point: down into the visit window, or up into it if too low
        above = target[2] - pos[2]
        near = math.hypot(target[0] - pos[0], target[1] - pos[1])
        if above < near + CAMERA_DROP:
            if self._ok(ASCEND):
                return ASCEND
        elif self._ok(DESCEND):
            return DESCEND
        # no way into the window from here: the next hover point round the target
        self.s["spot"] = (name, self.s["spot"][1] + 1)
        if self.s["spot"][1] >= len(SPOTS):
            self.s["visited"].add(name)           # nowhere round it to look from: give up,
            self.s["seen"].add(name)              # reporting it all the same (no cost if unvisited)
            self.s["spot"] = None
            self.s["climb_to"] = pos[2] - 4.0
        return self._move(obs, targets)

    def _spot(self, target, k):
        """The k-th hover point: first on the drone's side of the target, then round it."""
        r, a = SPOTS[k]
        angle = self.s["base"] + math.radians(a)
        return target[0] + r * math.cos(angle), target[1] + r * math.sin(angle)

    def _level(self, yaw, dx, dy, far, target, pos):
        """One step toward a hover point (dx, dy) away: at transit height, facing it, forward;
        over what blocks the way, and out from under what blocks the way up."""
        s = self.s
        if s["escape"] is not None:
            move = self._escape(yaw, pos)
            if move is not None:
                return move
        travelling = far > NEAR
        if travelling:
            # keep off surfaces: a move grazing one may or may not touch it
            if self.clear.get("descend", 4.0) < OFF and self._ok(ASCEND):
                return ASCEND
            if target[2] - pos[2] < TRANSIT - 1.0 and self._ok(ASCEND):
                return ASCEND                    # too low to travel: climb first
        rel = _wrap(math.atan2(dy, dx) - yaw)
        if travelling and abs(rel) > FACE:
            return ROTR if rel > 0 else ROTL
        if travelling or abs(rel) <= FACE:
            move = FORWARD
        else:                                    # close: step sideways rather than turn
            ahead, side = far * math.cos(rel), far * math.sin(rel)
            if abs(ahead) >= abs(side):
                if ahead < 0:
                    return ROTR if rel > 0 else ROTL
                move = FORWARD
            else:
                move = RIGHT if side > 0 else LEFT
        if self._ok(move, AHEAD if (move == FORWARD and travelling) else SAFE):
            return move
        if self._ok(ASCEND):
            return ASCEND                        # over it
        if move == FORWARD and self._ok(FORWARD):
            return FORWARD                       # can't climb: creep on while it is clear
        # shut in ahead and above: out from under it
        s["escape"] = {"base": yaw + rel, "leg": 0, "left": None}
        move = self._escape(yaw, pos)
        return move if move is not None else ROTR

    def _escape(self, yaw, pos):
        """Out from under an overhang: straight legs in turn round the goal's direction,
        longer each round, until the way up opens; then a good climb before going on."""
        s, e = self.s, self.s["escape"]
        if self._ok(ASCEND) and e["left"] is not None:
            s["escape"] = None
            s["climb_to"] = pos[2] - ESCAPE_CLIMB
            return ASCEND
        while e["leg"] < len(ESCAPE_LEGS):
            turn, length = ESCAPE_LEGS[e["leg"]]
            rel = _wrap(e["base"] + math.radians(turn) - yaw)
            if e["left"] is None:
                if abs(rel) > math.radians(8.0):
                    return ROTR if rel > 0 else ROTL
                e["left"] = length
            if e["left"] > 0 and self._ok(FORWARD):
                e["left"] -= 1
                return FORWARD
            e["leg"], e["left"] = e["leg"] + 1, None
        s["escape"] = None                       # nothing found: carry on as before
        return None


def main():
    expert = Expert()
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        kind = message.get("type")
        if kind == "act":
            out = {"action": expert.act(message["t"], message["obs"])}
        elif kind == "reset":
            expert.reset()
            out = {"ok": True}
        elif kind == "close":
            print(json.dumps({"ok": True}), flush=True)
            return 0
        else:
            out = {"ok": True}
        print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
