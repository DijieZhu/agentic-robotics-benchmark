"""Reference supervisor (privileged expert) for driving_safety_supervisor.

Submitted with the oracle token, so every observation carries the grader's ground-truth
block (`obs["privileged"]`): every road user and prop near the car with its true position,
velocity and size (vehicles to 250 m, with their reversing lights), and the route's lane
geometry and traffic lights ahead, all in the car's frame (x forward, y right, metres, from
the car's centre). It does not look at the camera.

Each 0.1 s step it:

* projects every actor onto the route (station along it, offset across it), sized by the
  larger of its bounding box and its measured collision footprint (footprints.json, next to
  this file), and predicts moving ones at constant velocity (vehicles along their lane);
* rolls its own car forward 3 s from where it really is, under its own steering law, with
  the car's measured steering lag (a lagged kinematic bicycle), and caps the speed so that
  the swept path stops short of any actor it would touch;
* keeps the planner's standing and following gaps behind what is in the lane, and stops
  at the line of a red light on the route ahead (of a yellow one too while it still can
  gently);
* when its own lane is blocked by something that stays put (props, a car that has stood
  still away from a queue, a pedestrian walking slowly in the lane, a car backing toward
  it) and a lane to the left (or the shoulder on the right) is clear for the whole
  manoeuvre, drives around it on a path offset into that lane and back, re-checking the
  way round every step until it is committed and turning back while it still can;
* otherwise drives the planner's steering at the posted limit, and follows the planner
  whenever the planner brakes.

Wire: stdio JSON lines, READY first; each `act` gets `{"action": [0]}` or
`{"action": [1, throttle, steer, brake]}`. A repeated `t` (the probe lock) is answered from
the state the step started from; only the request whose block holds the true actors (not
the pasted one, id -1) moves the expert's memory on, so the order of the two requests
changes nothing.
"""

import copy
import json
import math
import os
import sys

DT = 0.1
WHEELBASE = 2.9
MAX_STEER = 1.22                 # rad at steer = 1
# The car's steering response, fitted by tests/review/fit_vehicle.py on recorded episodes:
# the wheels follow the command with this lag, and turn by this fraction of MAX_STEER * steer.
LAG_TAU, STEER_GAIN, SLIP = 0.09, 0.815, 0.5
DECEL = 4.0                      # m/s^2 for stopping envelopes
ACCEL = 2.5                      # m/s^2 the rollout assumes when the car speeds up
LEAD_DECEL = 30.0                # a lead car may brake this hard, without warning
BRAKE_PER_MS2 = 1.0 / 8.5        # brake command per m/s^2 of deceleration
# Below about 5.5 m/s any brake command locks this car's wheels (well over 7 m/s^2, a harsh
# brake); coasting slows it by roughly 3-6 m/s^2 there, so gentle stops coast.
LOCK_SPEED, COAST_DECEL = 5.5, 4.5
GAP = {"vehicle": 3.5, "prop": 3.0, "walker": 4.5}     # bumper gap kept when stopped
HEADWAY_S = 1.5                  # extra following gap per m/s of the car's speed
HEADWAY_TRUCK_S = 2.5            # ... behind a loaded truck: its load may come off
TRUCKS = ("carlacola", "european_hgv", "sprinter")
GAP_STATIC = 9.0                 # ... behind something that stays put: room to pull out
CREEP_AFTER_S = 8.0              # waited this long behind a queue and still no way: close up
SIDE = {"vehicle": 0.35, "prop": 0.45, "walker": 0.9}  # lateral clearance in the lane logic
PROP_MIN = 0.6                   # props with no measured footprint: at least this big
HORIZON = 4.0                    # s of constant-velocity prediction for crossers
ROLL_S = 3.0                     # s of the swept-path rollout
# what the swept path keeps from each actor (m): props and still cars, cars going the car's
# way (sideways), oncoming cars (sideways: they do not yield), people (on the road, beyond the
# kerb); people's pad grows with the prediction time
MARGIN_STATIC, MARGIN_ALONG, MARGIN_ONCOMING = 0.3, 0.4, 0.2
MARGIN_WALKER, MARGIN_WALKER_KERB, WALKER_PAD_PER_S = 0.8, 0.5, 0.3
V_PASS, V_PASS_FAST = 7.0, 7.8   # m/s while driving around a blocker (fast: 0.6 m to spare)
V_LIMIT_MAX = 50.0 / 3.6         # the planner's own cap; the expert cruises at the posted limit
# This car's automatic gearbox (measured in CARLA 0.9.15 with speed sweeps) shifts up from 2nd near
# 8.2 m/s and down from 3rd near 9.2 m/s and from 2nd near 6.75 m/s; a downshift with the
# throttle shut (or the brake on) decelerates it by more than 7 m/s^2 for a tick, a harsh
# brake, while a light throttle (0.15) keeps the jerk under 6.5 m/s^2. So at the town limit
# (30 km/h) the expert cruises in 2nd gear, and it sheds a little speed with a light throttle
# rather than none.
V_GEAR2 = 7.8
LIFT_THROTTLE = 0.15
SHIFT_BAND = (6.3, 7.2)          # the 2nd -> 1st downshift: crossed with the light throttle on
V_NEAR_WALKER = 7.0
V_STEEP = 3.5                    # moving out from a standstill on a short ramp
V_KERB_WALKER = 2.0              # passing a person who stands just beyond the lane
V_CREEP = 2.0                    # the swept path touches at speed but not at this speed
V_COMMITTED = 3.0                # beside the blocker in the passing lane: keep moving
RAMP_PER_M, RAMP_PER_MS = 5.0, 2.5    # move out over max(5 m per m across, 2.5 s of travel)
RAMP_BACK_PER_M, RAMP_BACK_PER_MS = 4.0, 2.0
RAMP_BACK_MIN = 4.5              # shortest way back in
OUT_BEFORE = 3.0                 # fully out this far before the first thing passed
PLAN_WITHIN = 25.0               # passes are planned this close to the blocker
BEYOND_CENTRE = 0.9              # m past the passing lane's centre the car may go
CLEAR = {"vehicle": 0.3, "prop": 0.3, "walker": 0.7}   # extra room when passing
TIGHT = 0.25                     # side room for a slow, tight pass of props or a parked car
V_TIGHT = 4.0
SHOULDER_EDGE = 0.3              # m kept from the outer edge of a shoulder or parking lane
LAT_ACCEL = 3.0                  # m/s^2 in curves when the expert sets the speed
NUDGE_SIDE = 0.4                 # room left to an oncoming car over the centre line
V_NUDGE = 5.0
STEER_TRUST = 0.35               # planner steer this far off the route's is not followed
STILL_CAR_S = 0.6                # a stopped car that is not queued is passed once it stood this long
PLANNER_STOP = 0.45              # the planner's emergency stop brakes this hard (its speed control
                                 # brakes at most 0.3, and only because the expert drives faster)
CAUTION_M = 30.0                 # a road user standing in the path this close: ease off the throttle
BACKING_S = 0.5                  # a car rolling toward the car this long (or with reversing lights) backs
ONCOMING_YAW = 120.0             # degrees from the route: a vehicle facing the car
ALONG_YAW = 30.0                 # degrees from the route: a vehicle going the car's way
WALKER_CLEAR_S = 0.5             # a person's path clear this long before the car goes again
STUCK_S = 1.0                    # held at zero this long by something new: plan again
ONCOMING_MARGIN_S = 2.0          # an oncoming car reaches the manoeuvre's end this long after it
EXTRAPOLATE_M = 5.0              # projections past the route block's ends are trusted this far
# a light on the route ahead: the stop it gets (m/s^2), the most a yellow one is stopped with
# when it turns yellow, the most a red one is, and how far short of its stop point the front
# bumper stays (m; the stop then lies where the planner sees the light too)
LIGHT_DECEL, LIGHT_YELLOW_DECEL, LIGHT_RED_DECEL, LIGHT_MARGIN = 3.0, 5.0, 8.0, 1.0
# an actor this far above or below the car (plus a 12 % grade over the distance to it) is on
# another road: a bridge over the route or a road under it
OTHER_LEVEL_M, OTHER_LEVEL_GRADE = 3.0, 0.12
DEBUG = bool(os.environ.get("SUP_DEBUG"))
LOG_PATH = os.environ.get("SUP_LOG")

HERE = os.path.dirname(os.path.abspath(__file__))
_current = None                  # the expert whose request is being answered (for log tags)


def log(*parts):
    line = "[expert] " + " ".join(str(p) for p in parts) + "\n"
    if _current is not None and _current.probing:
        line = "[expert] (probe) " + line[len("[expert] "):]
    if LOG_PATH:
        try:
            with open(LOG_PATH, "a") as handle:
                handle.write(line)
            return
        except OSError:
            pass
    sys.stderr.write(line)
    sys.stderr.flush()


def load_footprints(path=None):
    """{blueprint: (back, front, left, right)}: half sizes from the bounding box centre, the
    larger of the box and the measured collision footprint on each side."""
    path = path or os.path.join(HERE, "footprints.json")
    try:
        with open(path) as handle:
            table = json.load(handle).get("footprints", {})
    except (OSError, ValueError):
        return {}
    out = {}
    for name, fp in table.items():
        try:
            (x0, x1), (y0, y1) = fp["x"], fp["y"]
            ex, ey = fp["box"]["extent"][:2]
            lx, ly = fp["box"]["location"][:2]
        except (KeyError, TypeError, ValueError):
            continue
        out[name] = (max(ex, lx - x0), max(ex, x1 - lx), max(ey, ly - y0), max(ey, y1 - ly))
    return out


FOOTPRINTS = load_footprints()
# the ego (vehicle.lincoln.mkz_2020) from its origin, the point the block's frame is centred on
_EGO = FOOTPRINTS.get("vehicle.lincoln.mkz_2020")
HALF_L = max(2.45, max(_EGO[0], _EGO[1])) if _EGO else 2.45
HALF_W = max(0.92, max(_EGO[2], _EGO[3])) if _EGO else 1.05


def _wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def _clip(value, lo, hi):
    return lo if value < lo else hi if value > hi else value


def _boxes_overlap(ax, ay, ayaw, ahx, ahy, bx, by, byaw, bhx, bhy):
    """Whether two oriented rectangles overlap (separating axes)."""
    dx, dy = bx - ax, by - ay
    ca, sa, cb, sb = math.cos(ayaw), math.sin(ayaw), math.cos(byaw), math.sin(byaw)
    for ux, uy in ((ca, sa), (-sa, ca), (cb, sb), (-sb, cb)):
        ra = ahx * abs(ca * ux + sa * uy) + ahy * abs(-sa * ux + ca * uy)
        rb = bhx * abs(cb * ux + sb * uy) + bhy * abs(-sb * ux + cb * uy)
        if abs(dx * ux + dy * uy) > ra + rb:
            return False
    return True


class Path(object):
    """The route polyline in the car's frame, with stations and per-point lane data."""

    def __init__(self, route):
        start = 0
        for i, p in enumerate(route):
            if p["x"] < 0.0:
                start = i
        pts = route[start:]
        if len(pts) < 2:
            pts = [{"x": -1.0, "y": 0.0, "width": 3.5, "junction": False, "left": None},
                   {"x": 50.0, "y": 0.0, "width": 3.5, "junction": False, "left": None}]
        self.pts = pts
        self.s = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.s.append(self.s[-1] + math.hypot(b["x"] - a["x"], b["y"] - a["y"]))
        self.s0, self.d0, _ = self.project3(0.0, 0.0)
        self._kerbs = {}

    def project3(self, x, y):
        """(station, offset to the right, metres extrapolated past either end) of a point."""
        best = (float("inf"), 0.0, 0.0, 0.0)
        last = len(self.pts) - 2
        for i in range(last + 1):
            a, b = self.pts[i], self.pts[i + 1]
            dx, dy = b["x"] - a["x"], b["y"] - a["y"]
            seg = math.hypot(dx, dy)
            if seg < 1e-6:
                continue
            ux, uy = dx / seg, dy / seg
            u = (x - a["x"]) * ux + (y - a["y"]) * uy
            uc = min(seg, max(0.0, u))
            d2 = (x - a["x"] - ux * uc) ** 2 + (y - a["y"] - uy * uc) ** 2
            if d2 < best[0]:
                off = -(x - a["x"]) * uy + (y - a["y"]) * ux
                extra = u - uc if (i == 0 and u < 0) or (i == last and u > seg) else 0.0
                best = (d2, self.s[i] + uc + extra, off, abs(extra))
        return best[1], best[2], best[3]

    def project(self, x, y):
        s, d, _ = self.project3(x, y)
        return s, d

    def _seg(self, station):
        for i in range(len(self.s) - 1):
            if self.s[i + 1] >= station:
                return i
        return len(self.s) - 2

    def heading(self, station):
        i = self._seg(station)
        a, b = self.pts[i], self.pts[i + 1]
        return math.atan2(b["y"] - a["y"], b["x"] - a["x"])

    def point(self, station, offset=0.0):
        i = self._seg(station)
        a, b = self.pts[i], self.pts[i + 1]
        u = (station - self.s[i]) / max(1e-6, self.s[i + 1] - self.s[i])
        x, y = a["x"] + (b["x"] - a["x"]) * u, a["y"] + (b["y"] - a["y"]) * u
        h = math.atan2(b["y"] - a["y"], b["x"] - a["x"])
        return x - math.sin(h) * offset, y + math.cos(h) * offset

    def lane(self, station):
        return self.pts[self._seg(station)]

    def left_offset(self, station, k=0):
        """Offset (negative: to the left) of the centre of the k-th driving lane to the left,
        or None."""
        lefts = self.lane(station).get("lefts")
        if lefts is None:
            lefts = [self.lane(station)["left"]] if self.lane(station).get("left") else []
        if k >= len(lefts):
            return None
        return self.project(lefts[k]["x"], lefts[k]["y"])[1]

    def left_lane(self, station, k=0):
        lefts = self.lane(station).get("lefts") or []
        return lefts[k] if k < len(lefts) else None

    def right_offset(self, station, k=0):
        """Offset (positive: to the right) of the centre of the k-th lane to the right
        (driving, shoulder, parking or bike lane), or None."""
        rights = self.lane(station).get("rights") or []
        if k >= len(rights):
            return None
        return self.project(rights[k]["x"], rights[k]["y"])[1]

    def side_offset(self, station, k, side):
        return self.left_offset(station, k) if side < 0 else self.right_offset(station, k)

    def right_type(self, station, k=0):
        """Type of the k-th lane to the right ("Driving", "Shoulder", "Parking", "Biking"), or None."""
        rights = self.lane(station).get("rights") or []
        return rights[k].get("type") if k < len(rights) else None

    def kerbs(self, station):
        """(left edge, right edge) of the drivable road at a station, as offsets."""
        key = int(station)
        if key not in self._kerbs:
            p = self.lane(station)
            half = p.get("width", 3.5) / 2.0
            left, right = -half, half
            lefts, rights = p.get("lefts") or [], p.get("rights") or []
            if lefts:
                left = self.project(lefts[-1]["x"], lefts[-1]["y"])[1] - lefts[-1]["width"] / 2.0
            if rights:
                right = self.project(rights[-1]["x"], rights[-1]["y"])[1] + rights[-1]["width"] / 2.0
            self._kerbs[key] = (left, right)
        return self._kerbs[key]

    def lane_edges(self, station):
        """(left, right) edges of the car's own lane at a station."""
        half = self.lane(station).get("width", 3.5) / 2.0
        return -half, half

    def junction_from(self, a, horizon=60.0):
        """Station of the first junction point at or after `a`, or None."""
        s = a
        while s <= a + horizon and s <= self.s[-1]:
            if self.lane(s).get("junction"):
                return s
            s += 2.0
        return None

    def junction_between(self, a, b):
        s = a
        while s <= b:
            if self.lane(s).get("junction"):
                return True
            s += 2.0
        return False


class Expert(object):
    def __init__(self):
        self.reset()

    def reset(self):
        # pass plans are kept in odometer metres, since the path is rebuilt every step
        self.mem = {"still": {}, "back": {}, "vprev": {}, "moved": {}, "pass": None, "odo": 0.0,
                    "behind": 0.0, "thr_i": 0.45,
                    "recover": False, "gap_static": GAP_STATIC, "nudge": 0.0, "wheel": 0.0,
                    "walk_clear": 99.0, "stuck": 0.0, "aborted_at": None, "lights": {}}
        self.s0, self.red, self.ahead, self.path = 0.0, False, 0, None
        self.last_t, self.start, self.after_real = None, None, None
        self.roll = []
        self.rolling = False
        self.probing = False
        self.caution = 0.0
        self.speed_now = 0.0
        if DEBUG:
            log("reset")
        self.nudge_want = 0.0
        self.nudge_gap = None

    def act(self, t, obs):
        global _current
        _current = self
        return self._act(t, obs)

    def _act(self, t, obs):
        """One reply. Each step starts from the memory the step began with; the request whose
        block lists the pasted road user (id -1: the probe's copy) is answered but never moves
        the memory on, so the replies do not depend on which of the two came first."""
        priv = obs.get("privileged")
        probe = bool(priv) and any(a.get("id") == -1 for a in priv.get("actors", ()))
        self.probing = probe
        if t != self.last_t:
            self.start, self.last_t, self.after_real = copy.deepcopy(self.mem), t, None
        self.mem = copy.deepcopy(self.start)
        action = self._decide(obs, priv) if priv else [0]
        if probe:
            self.mem = self.after_real if self.after_real is not None else copy.deepcopy(self.start)
        else:
            self.after_real = copy.deepcopy(self.mem)
        return action

    # -- perception from ground truth ----------------------------------------------
    def _actors(self, path, priv):
        out = []
        for a in priv["actors"]:
            if abs(a.get("z", 0.0)) > OTHER_LEVEL_M + OTHER_LEVEL_GRADE * math.hypot(a["x"], a["y"]):
                continue                 # on a bridge over the route or a road under it
            fp = FOOTPRINTS.get(a.get("blueprint", ""))
            if a["type"] == "prop":
                if a.get("height", 1.0) < 0.15:
                    continue
                if fp is None:
                    half_l, half_w = max(a["length"], PROP_MIN) / 2.0, max(a["width"], PROP_MIN) / 2.0
                    fp = (half_l, half_l, half_w, half_w)
            if fp is None:
                fp = (a["length"] / 2.0, a["length"] / 2.0, a["width"] / 2.0, a["width"] / 2.0)
            back, front, left, right = fp
            # the footprint's centre and half sizes in the actor's own frame
            yaw = math.radians(a["yaw"])
            cy, sy = math.cos(yaw), math.sin(yaw)
            ox, oy = (front - back) / 2.0, (right - left) / 2.0
            cx, cyy = a["x"] + cy * ox - sy * oy, a["y"] + sy * ox + cy * oy
            hx, hy = (front + back) / 2.0, (left + right) / 2.0
            # the box's four corners on the route: its true span along and across it
            spans = [path.project3(cx + cy * dl - sy * dw, cyy + sy * dl + cy * dw)
                     for dl in (-hx, hx) for dw in (-hy, hy)]
            s_lo, s_hi = min(p[0] for p in spans), max(p[0] for p in spans)
            d_lo, d_hi = min(p[1] for p in spans), max(p[1] for p in spans)
            extra = max(p[2] for p in spans)
            s, d = (s_lo + s_hi) / 2.0, (d_lo + d_hi) / 2.0
            h = path.heading(_clip(s, 0.0, path.s[-1]))
            vs = a["vx"] * math.cos(h) + a["vy"] * math.sin(h)
            vd = -a["vx"] * math.sin(h) + a["vy"] * math.cos(h)
            speed = math.hypot(a["vx"], a["vy"])
            rel = abs(_wrap_deg(a["yaw"] - math.degrees(h)))
            beyond = extra > EXTRAPOLATE_M and (s > path.s0 or extra > 60.0)
            if beyond:
                # past the block's route: its lane is not known from the route; only the
                # heading says whether it faces the car
                d = 99.0 if rel < ONCOMING_YAW else -99.0
            ident = a["id"]
            if ident != -1:
                still = self.mem["still"].get(ident, 0.0)
                still = still + DT if speed < 0.3 else 0.0
                self.mem["still"][ident] = still
                back_t = self.mem["back"].get(ident, 0.0)
                back_t = back_t + DT if (a["type"] == "vehicle" and vs < -0.3 and rel < ALONG_YAW) else 0.0
                self.mem["back"][ident] = back_t
                accel = (speed - self.mem["vprev"].get(ident, speed)) / DT
                self.mem["vprev"][ident] = speed
                if speed > 0.5:
                    self.mem["moved"][ident] = True
            else:
                still, back_t, accel = (99.0 if speed < 0.3 else 0.0), 0.0, 0.0
            ha, hl = (d_hi - d_lo) / 2.0, (s_hi - s_lo) / 2.0
            out.append(dict(a, s=s, d=d, vs=vs, vd=vd, speed=speed, still=still, ha=ha, hl=hl,
                            rel=rel, beyond=beyond, fx=cx, fy=cyy, fyaw=yaw, fhx=hx, fhy=hy, accel=accel,
                            backing=a["type"] == "vehicle" and (
                                bool(a.get("reverse")) and speed > 0.05 or back_t >= BACKING_S),
                            oncoming=a["type"] == "vehicle" and rel > ONCOMING_YAW))
        return out

    # -- the planned lateral offset (passing) ------------------------------------------
    def _update_nudge(self, path, actors):
        """Move toward the kerb, within the lane, for an oncoming car that comes over the
        centre line into the car's lane; drift back once it has gone by."""
        lane_w = path.lane(path.s0).get("width", 3.5)
        room = lane_w / 2.0 - HALF_W - 0.05
        want = 0.0
        self.nudge_gap = None
        for a in actors:
            if not a["oncoming"] or a["vs"] > -1.0 or a["d"] > 0.0 or a["beyond"]:
                continue
            if a["s"] + a["hl"] < path.s0 - HALF_L or a["s"] - path.s0 > 60.0:
                continue
            # where it will be across the road when it meets the car, if it keeps coming over
            closing = max(1.0, self.speed_now - a["vs"])
            meet = max(0.0, a["s"] - path.s0) / closing
            d_meet = a["d"] + max(0.0, a["vd"]) * min(meet, 1.5)
            need = d_meet + a["ha"] + HALF_W + NUDGE_SIDE + 0.05
            if need > 0.0:
                want = max(want, need)
        want = min(max(0.0, want), room)
        self.nudge_want = want
        now = self.mem["nudge"]
        self.mem["nudge"] = now + max(-0.1, min(0.2, want - now))
        if self.mem["nudge"] < 0.02:
            self.mem["nudge"] = 0.0
        # how much room the nudge leaves to the oncoming car that asks for the most
        for a in actors:
            if a["oncoming"] and a["vs"] < -1.0 and -6.0 < a["d"] <= 0.0 and not a["beyond"] and \
                    0.0 < a["s"] - path.s0 < 60.0:
                gap = (self.mem["nudge"] - HALF_W) - (a["d"] + a["ha"])
                self.nudge_gap = gap if self.nudge_gap is None else min(self.nudge_gap, gap)

    def _odo(self, station):
        return station - self.s0 + self.mem["odo"]

    def _offset_at(self, station):
        p = self.mem["pass"]
        if p is None:
            return max(self.mem["nudge"], self.nudge_want) if self.rolling else self.mem["nudge"]
        station = self._odo(station)
        start = p.get("from", 0.0)
        if station >= p["e"]:
            return 0.0
        if station <= p["a"]:
            return start
        if station < p["b"]:
            return start + (p["off"] - start) * (station - p["a"]) / max(1e-6, p["b"] - p["a"])
        if station <= p["c"]:
            return p["off"]
        return p["off"] * (p["e"] - station) / max(1e-6, p["e"] - p["c"])

    def _pred_offset(self, station):
        """Where the car will be across the route at a station: the swept path's offset
        while it lasts, the plan's beyond it."""
        roll = self.roll
        if roll and station <= roll[-1][0]:
            for s, d in roll:
                if s >= station:
                    return d
        return self._offset_at(station)

    def _in_corridor(self, a, s, d, pad=0.0):
        p = self.mem["pass"]
        side = TIGHT if p is not None and p.get("tight") and a["id"] in p["ids"] else SIDE[a["type"]]
        if a["type"] == "vehicle" and a["oncoming"] and a["vs"] < -0.5:
            side = 0.0      # an oncoming car does not yield: what counts is whether it touches
        if a["type"] == "walker" and self.path is not None:
            left, right = self.path.kerbs(max(0.0, s))
            if d - a["ha"] > right or d + a["ha"] < left:
                side = SIDE["vehicle"]       # on the pavement, beyond the kerb
        return abs(d - self._pred_offset(s)) < HALF_W + a["ha"] + side + pad

    def _queued(self, path, actors, car):
        """A stopped car that is waiting its turn: another vehicle close ahead of it in its
        lane, a red light near it, or a junction just ahead of its front."""
        for other in actors:
            if other is car or other["type"] != "vehicle" or other["beyond"]:
                continue
            if 0.0 < other["s"] - car["s"] < 15.0 + car["hl"] + other["hl"] and \
                    abs(other["d"] - car["d"]) < 1.5:
                return True
        if self.red and car["s"] - path.s0 < 30.0:
            return True
        # a car waiting to enter a junction stands at its edge, not well short of it
        return path.junction_between(car["s"], car["s"] + car["hl"] + 8.0)

    def _limit(self, path, actors, v):
        """Speed cap from everything on, or coming onto, the path ahead, with the actor
        that sets it and the distance left to stop in (its stopping room counted)."""
        v_cap, worst, room = V_LIMIT_MAX, None, 100.0
        s0 = path.s0
        for a in actors:
            if a["beyond"]:
                continue
            gap0 = GAP[a["type"]]
            queue_wait = a["type"] == "vehicle" and self.mem["behind"] >= CREEP_AFTER_S and \
                self._queued(path, actors, a)
            stand = GAP[a["type"]] if (queue_wait or a["id"] == -1) else self.mem["gap_static"]
            if a["speed"] < 0.3 and a["type"] != "walker" and a["id"] != -1 and not queue_wait:
                gap0 = stand
            elif a["type"] == "walker" and a["id"] != -1 and a["vs"] > 0.3 and abs(a["vd"]) < 0.6:
                gap0 = max(gap0, stand)          # someone walking along the lane: room to pull out
            if a["s"] > s0 and self._in_corridor(a, a["s"], a["d"]):
                if a["type"] == "walker" and self._walking_away(a, v):
                    continue
                # only a vehicle is trusted to keep moving; a sliding prop or a person may stop
                lead = max(0.0, a["vs"]) if a["type"] == "vehicle" else 0.0
                walk = min(2.0, max(0.0, a["vs"])) if (a["type"] == "walker" and a["id"] != -1 and
                                                       abs(a["vd"]) < 0.6) else 0.0
                if a["type"] == "vehicle" and lead > 0.3 and not a["backing"]:
                    # from the standing gap behind a car that stands, to a time gap behind one
                    # that drives: continuous in the lead's speed, so a lead pulling away (or
                    # slowing to a stop) never makes the gap jump
                    w = min(1.0, lead / 3.0)
                    headway = HEADWAY_TRUCK_S if any(k in a.get("blueprint", "") for k in TRUCKS) else HEADWAY_S
                    # the time gap at the lead's speed (at the car's own it would grow as the car
                    # speeds up and make it brake again: stop-and-go behind a lead pulling away);
                    # closing in faster than the lead is the stopping envelope's business
                    gap0 = (1.0 - w) * stand + w * (GAP["vehicle"] + headway * min(v, lead))
                if a["vs"] < -0.3 and a["type"] == "vehicle":
                    gap0 += 4.0 + 2.0 * -a["vs"]        # backing (or driving) toward the car
                gap = a["s"] - a["hl"] - s0 - HALF_L - gap0 + lead * lead / (2 * LEAD_DECEL)
                cap = math.sqrt(2 * DECEL * gap) if gap > 0 else 0.0
                if walk > 0.0:
                    # behind someone walking along the lane: settle at their pace at the standing
                    # gap (a gentle approach; stop-and-go there locks the wheels every time)
                    follow = walk + 0.5 * gap
                    cap = max(0.0, min(cap, follow) if gap > 0.0 else follow)
                if cap < v_cap:
                    v_cap, worst, room = cap, a, gap
                continue
            if a["speed"] < 0.4:
                continue
            if a["s"] < s0 - HALF_L and a["vs"] > 0.0:
                continue                 # behind, going the same way: not ahead of the car
            if a["type"] == "vehicle" and abs(a["vd"]) < 0.5:
                continue                 # driving along its lane: a straight-line guess would cut bends
            if a["type"] == "walker" and abs(a["vd"]) < 0.4 and abs(a["vs"]) > 0.5:
                continue                 # walking along the road, swaying: not crossing
            tau = 0.25
            while tau <= HORIZON:
                s, d = path.project(a["x"] + a["vx"] * tau, a["y"] + a["vy"] * tau)
                closing = abs(d - self._pred_offset(s)) < abs(a["d"] - self._pred_offset(a["s"])) - 0.05
                if s > s0 + HALF_L and closing and self._in_corridor(a, s, d, pad=0.3):
                    reach = (s - a["hl"] - s0 - HALF_L) / max(v, 0.5)
                    clear = (s + a["hl"] - s0 + HALF_L) / max(v, 0.5)
                    if tau < clear + 1.0 and reach < tau + 2.0:
                        gap = s - a["hl"] - s0 - HALF_L - gap0
                        cap = math.sqrt(2 * DECEL * gap) if gap > 0 else 0.0
                        if cap < v_cap:
                            v_cap, worst, room = cap, a, gap
                    break
                tau += 0.25
        return v_cap, worst, room

    def _walking_away(self, a, v):
        """A person in the path who walks out of it before the car gets there."""
        if abs(a["vd"]) < 0.4:
            return False
        off = self._pred_offset(a["s"])
        if (a["d"] - off) * a["vd"] <= 0.0:
            return False                 # walking toward the car's path
        half = HALF_W + a["ha"] + SIDE["walker"]
        t_exit = max(0.0, half - abs(a["d"] - off)) / abs(a["vd"])
        t_arrive = max(0.0, a["s"] - a["hl"] - self.s0 - HALF_L) / max(v, 0.5)
        return t_exit + 0.5 < t_arrive

    # -- the swept path ------------------------------------------------------------------
    def _look(self, v, passing):
        look = max(5.0, 0.9 * v + 3.0)
        if passing:
            p = self.mem["pass"]
            if p is not None and p.get("tight_back") and self.mem["odo"] >= p["c"] - 6.0:
                return 3.5
            return min(look, max(4.0, 0.5 * v + 2.5))
        return look

    def _steer_from(self, path, x, y, yaw, v, passing):
        """Pure pursuit to the planned offset, seen from a pose in the car's frame now; the
        command makes up the steering gain the wheels actually give."""
        s_r, _d = path.project(x, y)
        look = self._look(v, passing)
        tx, ty = path.point(s_r + look, self._offset_at(s_r + look))
        dx, dy = tx - x, ty - y
        c, s = math.cos(yaw), math.sin(yaw)
        lx, ly = c * dx + s * dy, -s * dx + c * dy
        delta = math.atan2(2.0 * WHEELBASE * ly, max(lx * lx + ly * ly, 1.0))
        return _clip(delta / (STEER_GAIN * MAX_STEER), -1.0, 1.0)

    def _rollout(self, path, v, target, own, nominal_steer):
        """The car's own path for ROLL_S from its real pose: (time, x, y, yaw, travelled) every
        0.1 s, under the steering law it will drive with (its own, or the planner's for the
        first step and then pure pursuit on the lane) and the lagged steering response."""
        x = y = yaw = 0.0
        wheel = self.mem["wheel"]
        alpha = 1.0 - math.exp(-0.05 / LAG_TAU)
        passing = self.mem["pass"] is not None
        self.rolling = True
        try:
            out = self._roll(path, v, target, own, nominal_steer, x, y, yaw, wheel, alpha, passing)
        finally:
            self.rolling = False
        return out

    def _roll(self, path, v, target, own, nominal_steer, x, y, yaw, wheel, alpha, passing):
        out, travelled = [], 0.0
        for k in range(int(round(ROLL_S / DT))):
            cmd = self._steer_from(path, x, y, yaw, v, passing) if (own or k > 0) else nominal_steer
            for _ in range(2):
                wheel += (cmd - wheel) * alpha
                w = STEER_GAIN * MAX_STEER * wheel
                tw = math.tan(w)
                beta = math.atan(SLIP * tw)
                x += v * math.cos(yaw + beta) * 0.05
                y += v * math.sin(yaw + beta) * 0.05
                yaw += v * math.cos(beta) * tw / WHEELBASE * 0.05
                travelled += v * 0.05
            v = min(target, v + ACCEL * DT) if v < target else max(target, v - DECEL * DT)
            out.append(((k + 1) * DT, x, y, yaw, travelled))
        return out

    def _footprint_at(self, path, a, t):
        """An actor's footprint t seconds ahead: (x, y, yaw, half length, half width)."""
        if a["type"] == "vehicle" and abs(a["vd"]) < 0.5 and a["speed"] > 0.3 and not a["beyond"]:
            # along its lane: its station moves on, its offset stays
            s = a["s"] + a["vs"] * t
            px, py = path.point(_clip(s, 0.0, path.s[-1] + 50.0), a["d"])
            if s > path.s[-1]:
                h = path.heading(path.s[-1])
                px, py = px + math.cos(h) * (s - path.s[-1]), py + math.sin(h) * (s - path.s[-1])
            turn = path.heading(_clip(s, 0.0, path.s[-1])) - path.heading(_clip(a["s"], 0.0, path.s[-1]))
            return px, py, a["fyaw"] + turn, a["fhx"], a["fhy"]
        if a["speed"] < 0.05 or a["type"] == "prop" and a["speed"] < 0.3:
            return a["fx"], a["fy"], a["fyaw"], a["fhx"], a["fhy"]
        return a["fx"] + a["vx"] * t, a["fy"] + a["vy"] * t, a["fyaw"], a["fhx"], a["fhy"]

    def _margins(self, a, t):
        """(along, across) room the swept path keeps from an actor, t seconds ahead."""
        if a["type"] == "walker":
            if self.path is not None:
                left, right = self.path.kerbs(max(0.0, a["s"]))
                kerb = a["d"] - a["ha"] > right or a["d"] + a["ha"] < left
            else:
                kerb = False
            m = (MARGIN_WALKER_KERB if kerb else MARGIN_WALKER) + WALKER_PAD_PER_S * t
            return m, m
        if a["type"] == "vehicle" and a["speed"] > 0.3 and not a["backing"]:
            if a["oncoming"]:
                return MARGIN_STATIC, MARGIN_ONCOMING
            return MARGIN_STATIC, MARGIN_ALONG
        return MARGIN_STATIC, MARGIN_STATIC

    def _first_contact(self, path, actors, roll, extra=0.0):
        """The first rollout sample whose car box overlaps an actor's predicted footprint
        (each inflated by its margins plus `extra`): (travelled metres, actor), or (None, None)."""
        if not roll:
            return None, None
        reach = roll[-1][4] + HALF_L + 2.0
        near = []
        for a in actors:
            if a["beyond"]:
                continue
            dist = math.hypot(a["fx"], a["fy"]) - a["fhx"] - a["fhy"]
            if dist > reach + a["speed"] * ROLL_S + 3.0:
                continue
            if a["fx"] + a["fhx"] + a["fhy"] < -HALF_L + 0.3:
                continue                 # behind the car: braking cannot keep it off
            if a["type"] == "vehicle" and (a["backing"] or (a["oncoming"] and a["vs"] < -0.5)):
                continue                 # coming at the car: braking cannot keep it off either;
                                         # the way out is sideways (the nudge, the pass)
            near.append(a)
        if not near:
            return None, None
        for t, x, y, yaw, travelled in roll:
            for a in near:
                ax, ay, ayaw, ahx, ahy = self._footprint_at(path, a, t)
                m_along, m_across = self._margins(a, t)
                if _boxes_overlap(x, y, yaw, HALF_L, HALF_W, ax, ay, ayaw,
                                  ahx + m_along + extra, ahy + m_across + extra):
                    return travelled, a
        return None, None

    # -- passing -------------------------------------------------------------------
    def _blocker(self, path, actors, s_from):
        """The nearest group of things that stay put in the car's own lane ahead:
        (start station, end station, members)."""
        members = []
        for a in actors:
            if a["s"] <= s_from or a["s"] > path.s0 + 45.0 or a["id"] == -1 or a["beyond"]:
                continue
            if abs(a["d"]) > HALF_W + a["ha"] + SIDE[a["type"]] + 0.1:   # all the corridor stops for
                continue
            if a["type"] == "walker":
                if a["vs"] > 2.5 or abs(a["vd"]) > 0.8:
                    continue
            elif a["speed"] > 0.3 and not a["backing"]:
                continue
            members.append(a)
        if not members:
            return None
        members.sort(key=lambda a: a["s"])
        start, end = members[0]["s"] - members[0]["hl"], members[0]["s"] + members[0]["hl"]
        group = [members[0]]
        for a in members[1:]:
            if a["s"] - a["hl"] > end + 10.0:
                break
            end = max(end, a["s"] + a["hl"])
            group.append(a)
        return start, end, group

    def _left_clear(self, path, actors, a_s, e_s, off, v, skip, oncoming_lane=False):
        """Nothing in the passing lane over [a_s, e_s] for as long as the manoeuvre takes (an
        oncoming car, until ONCOMING_MARGIN_S after the car is back in its lane)."""
        duration = self._time_to(e_s - path.s0, v) + 1.0
        s0 = path.s0
        for act in actors:
            if act["id"] in skip or act["id"] == -1:
                continue
            if act["beyond"]:
                # far past the route block: only a car ahead coming the other way can reach the
                # manoeuvre in time, and only in an oncoming passing lane
                if oncoming_lane and act["oncoming"] and act["vs"] < -0.5 and act["s"] > e_s:
                    arrive = (act["s"] - act["hl"] - e_s) / max(0.5, -act["vs"])
                    if arrive < duration + ONCOMING_MARGIN_S:
                        return False, act
                continue
            if act["vs"] > max(3.0, v) and act["s"] - act["hl"] > s0 + HALF_L + 3.0:
                continue                 # ahead and pulling away
            if act["type"] == "vehicle" and act["vs"] > 0.5 and act["s"] < s0 - HALF_L:
                # coming up behind: clear if it cannot close the gap while the car is out there
                gap = s0 - HALF_L - (act["s"] + act["hl"])
                closing = act["vs"] - max(v, 3.0)
                if closing <= 0.0 or gap > max(3.0 * closing + 10.0, closing * closing / 6.0 + 10.0):
                    continue             # it keeps 3 s, and room to brake gently, behind the car
                if act["d"] - act["ha"] < off + HALF_W + SIDE["vehicle"] and \
                        act["d"] + act["ha"] > off - HALF_W - SIDE["vehicle"]:
                    return False, act
                continue
            span = duration + (ONCOMING_MARGIN_S if act["oncoming"] else 0.0)
            s_then = act["s"] + act["vs"] * span
            lo = min(act["s"], s_then) - act["hl"]
            hi = max(act["s"], s_then) + act["hl"]
            if hi < a_s - 6.0 or lo > e_s + 6.0:
                continue
            d_then = act["d"] + act["vd"] * min(duration, 2.0)
            side = SIDE[act["type"]]
            for d in (act["d"], d_then):
                if off - HALF_W - act["ha"] - side < d < off + HALF_W + act["ha"] + side:
                    return False, act
        return True, None

    @staticmethod
    def _time_to(distance, v):
        """Seconds to cover `distance` from speed v, speeding up at 2 m/s^2 to V_PASS_FAST."""
        distance = max(0.0, distance)
        v = max(0.0, v)
        if v >= V_PASS_FAST:
            return distance / v
        t_acc = (V_PASS_FAST - v) / 2.0
        d_acc = (v + V_PASS_FAST) / 2.0 * t_acc
        if distance <= d_acc:
            return (-v + math.sqrt(v * v + 4.0 * distance)) / 2.0 if distance > 0 else 0.0
        return t_acc + (distance - d_acc) / V_PASS_FAST

    def _ramp(self, delta, v):
        """Length of a move out across `delta` metres at speed v (shorter from a standstill,
        where the car can turn tighter)."""
        f = 0.5 + 0.5 * min(1.0, v / 5.0)
        return max(RAMP_PER_M * abs(delta) * f, RAMP_PER_MS * v)

    def _plan_pass(self, path, actors, blocker, v, red_light):
        start, end, group = blocker
        s0 = path.s0
        kinds = set(a["type"] for a in group)
        backing = any(a["backing"] for a in group)
        # stop far enough back to pull out round it: as far behind as the way round is wide
        need = max(abs(self._need(group, side=-1)), 3.0)
        self.mem["gap_static"] = max(self.mem["gap_static"], GAP_STATIC, 2.5 * need + 3.0)
        # within 25 m, or as far out as the move out at this speed needs (2.5 s of travel)
        within = max(PLAN_WITHIN, RAMP_PER_MS * v + OUT_BEFORE + 3.0)
        if start - s0 - HALF_L > within + (5.0 if backing else 0.0):
            return None
        if kinds == {"vehicle"} and not backing:
            # a car queued at a light or a junction is not passed; one parked askew or off
            # the lane centre, or one standing on its own, is (once it has stood a second)
            parked = any(abs(a["d"]) > 0.5 or abs(_wrap_deg(a["yaw"] - math.degrees(
                path.heading(max(0.0, a["s"]))))) > 15.0 for a in group)
            queued = any(self._queued(path, actors, a) for a in group)
            if queued and not parked:
                if DEBUG:
                    log("no pass: stopped car in a queue, at a light or a junction")
                return None
            if min(a["still"] for a in group) < STILL_CAR_S:
                return None
            # a car that has not moved at all yet, square in the lane, may just be about to go:
            # it is passed once the car has stopped behind it
            if not parked and not any(self.mem["moved"].get(a["id"]) for a in group) and \
                    self.mem["behind"] < STILL_CAR_S:
                return None
        # the left first (lane by lane, against the traffic if need be), then the right:
        # a shoulder or a parking lane when the left is closed
        for side in (-1, 1):
            k = 0
            while True:
                offs = [path.side_offset(s, k, side) for s in (start - 10.0, start, end, end + 6.0)]
                if any(o is None for o in offs):
                    if DEBUG:
                        log("no pass: no lane %d on the %s" % (k, "left" if side < 0 else "right"))
                    break
                plan, who = self._try_lane(path, actors, group, kinds, start, end, v, k, offs, side)
                if plan is not None:
                    return plan
                if who is not None and (who["speed"] > 0.3 or who["type"] == "walker"):
                    return None              # traffic or a person there: wait for it
                if who is None:
                    break                    # no room on this side as things stand
                # something that stays put in that lane too: look one lane further
                k += 1
                further = [path.side_offset(s, k, side) for s in (start, end)]
                if all(o is not None for o in further):
                    self.mem["gap_static"] = max(self.mem["gap_static"],
                                                 2.5 * max(abs(o) for o in further) + 3.0)
        return None

    def _try_lane(self, path, actors, group, kinds, start, end, v, k, offs, side):
        """A pass into the k-th lane on one side (-1 left, +1 right), or (None, what is in
        the way). Worked out as if to the left (offsets negative), then mirrored."""
        s0 = path.s0
        offs = [-side * o for o in offs]
        need = -side * self._need(group, side=side)
        # on the right that lane is often a shoulder against a kerb or a rail: keep clear of
        # its outer edge
        stations = (start - 10.0, start, end, end + 6.0)
        limit = max(offs) - (BEYOND_CENTRE if side < 0 else 0.0)
        if side > 0:
            outer = min(path.kerbs(s)[1] for s in stations)
            limit = max(limit, -(outer - HALF_W - SHOULDER_EDGE))
        tight = False
        settled = all(a["still"] >= 3.0 for a in group)    # nothing still sliding or rolling
        # a slow, tight pass: of props; and on a shoulder or parking lane (the way round a
        # closure) also of cars stopped there and not backing, such as a crash with its sign
        shoulder = side > 0 and kinds <= {"prop", "vehicle"} and \
            not any(a["backing"] for a in group) and \
            all(path.right_type(s, k) in ("Shoulder", "Parking") for s in stations)
        if k == 0 and need < limit and (kinds == {"prop"} or shoulder) and settled and \
                -side * self._need(group, True, side) >= limit:
            tight, need = True, -side * self._need(group, True, side)   # slowly, with less room
        off = max(limit, min(-0.5, need))
        if k > 0:
            off = max(limit, min(need, max(offs)))          # that lane's centre, at least
        if off > need + 1e-6:
            if DEBUG:
                log("no pass: needs %.2f, %s lane %d allows %.2f" % (
                    need, "left" if side < 0 else "right", k, limit))
            return None, None
        off, limit = -side * off, -side * limit
        from_off = path.d0 if abs(path.d0) > 0.3 else 0.0
        margin = 6.0 if "walker" in kinds else 4.0
        backing = any(a["backing"] for a in group)
        # fully out OUT_BEFORE before the front bumper reaches the first thing passed; the
        # move out needs more room the faster the car goes: too close, keep slowing and look
        # again
        b_s = start - HALF_L - OUT_BEFORE
        a_s = b_s - self._ramp(off - from_off, v)
        if backing:
            # it is coming back at the car: out at once, on the shortest move
            a_s = s0 + 0.3
            b_s = max(a_s + self._ramp(off - from_off, 0.0), min(b_s, a_s + self._ramp(off - from_off, v)))
        elif a_s < s0 + 0.3:
            # slow, close behind it: out slowly from here, as steep as the swept path allows (the
            # car may have had to stop close behind something that slid into its lane, or be
            # following someone who walks in it)
            close = start - HALF_L - 0.5
            if v < V_STEEP and max(b_s, close) - (s0 + 0.3) >= max(1.1 * abs(off - from_off), 2.5):
                a_s = s0 + 0.3
                if b_s - a_s < self._ramp(off - from_off, 0.0) * 0.65:
                    b_s = close
            else:
                if DEBUG:
                    log("no pass: too close (%.1f m to go at %.1f m/s)" % (b_s - s0, v))
                return None, None
        # the way back begins only once the car's rear is past the last thing passed, with the
        # pure pursuit's look-ahead counted (it steers for a point that far ahead); before a
        # junction, earlier and slowly (a short look-ahead: tight_back), never closer than
        # `margin` past it
        walkers = "walker" in kinds
        wide = margin if walkers else max(margin, self._look(V_PASS_FAST, True) + HALF_L + 0.5)
        c_s = end + wide
        e_s = c_s + max(RAMP_BACK_PER_M * abs(off), RAMP_BACK_PER_MS * V_PASS, RAMP_BACK_MIN)
        tight_back = False
        junction = path.junction_from(start)
        # past someone walking into a junction the car may finish its way back just inside it
        junction_end = None if junction is None else junction + (4.0 if walkers else -1.0)
        if junction_end is not None and e_s > junction_end:
            e_s = junction_end
            if e_s - c_s < RAMP_BACK_MIN:
                c_s, tight_back = max(end + margin, e_s - RAMP_BACK_MIN), True
            if e_s - c_s < RAMP_BACK_MIN - 1e-6 and not ("walker" in kinds and abs(off) < 2.0):
                if DEBUG:
                    log("no pass: junction %.1f m after the blocker" % (junction - end))
                return None, None
        margin = c_s - end
        ids = [a["id"] for a in group]
        lane = path.left_lane(start, k) if side < 0 else None
        oncoming_lane = bool(lane is not None and not lane.get("same_direction", True))
        clear, who = self._left_clear(path, actors, a_s, e_s, off, v, set(ids), oncoming_lane)
        if not clear:
            if DEBUG:
                log("no pass: %s#%s in %s lane %d (v %.1f, d %.1f)" % (
                    who["type"], who["id"], "left" if side < 0 else "right", k, who["speed"], who["d"]))
            return None, who
        log("pass", sorted(kinds), "in %.1f m, %s lane %d, offset %.2f%s" % (
            start - s0 - HALF_L, "left" if side < 0 else "right", k, off, ", tight" if tight else ""))
        return {"a": self._odo(a_s), "b": self._odo(b_s), "c": self._odo(c_s),
                "e": self._odo(e_s), "off": off, "from": from_off, "limit": limit, "ids": ids,
                "margin": margin, "tight": tight, "side": side, "k": k,
                "commit": self._odo(start - HALF_L - 1.0), "oncoming_lane": oncoming_lane,
                "tight_back": tight_back,
                "steep": abs(off - from_off) / max(1e-6, b_s - a_s) > 0.25}, None

    @staticmethod
    def _need(group, tight=False, side=-1):
        """Offset that clears every member on one side (negative: left) with room to spare
        (or, tight, with just enough)."""
        # (0.1 m more than the clearance wanted: the car tracks a little short of its plan)
        room = lambda a: HALF_W + 0.1 + (TIGHT + 0.05 if tight else SIDE[a["type"]] + CLEAR[a["type"]])  # noqa: E731
        if side < 0:
            return min(a["d"] - a["ha"] - room(a) for a in group)
        return max(a["d"] + a["ha"] + room(a) for a in group)

    def _turn_back(self, path, v, why):
        """Give the pass up: back to the car's own lane from where it is now."""
        odo = self.mem["odo"]
        d0 = path.d0
        back = max(RAMP_BACK_PER_M * abs(d0), RAMP_BACK_PER_MS * max(v, 2.0), RAMP_BACK_MIN)
        log("turn back (%s) from offset %.2f over %.1f m" % (why, d0, back))
        self.mem["pass"] = {"a": odo - 2.0, "b": odo - 1.0, "c": odo, "e": odo + back, "off": d0,
                            "from": d0, "limit": d0, "ids": [], "margin": 0.0, "tight": False,
                            "side": -1 if d0 < 0 else 1, "k": 0, "commit": odo - 1.0,
                            "oncoming_lane": False, "steep": False, "back": True}
        self.mem["aborted_at"] = odo

    def _track_pass(self, path, actors, v):
        """Follow the blocker while it stays put; until the car is committed, look again at
        the way round every step and turn back if it closes; if the blocker drives off, go
        back to the lane behind it."""
        p = self.mem["pass"]
        odo = self.mem["odo"]
        p.pop("hold_now", None)
        if p.get("back"):
            if odo > p["e"] or abs(path.d0) < 0.3 or (v < 0.3 and self.mem["behind"] >= 1.0):
                self.mem["pass"] = None
            return
        here = [a for a in actors if a["id"] in p["ids"]]
        staying = bool(here) and all(a["speed"] < 1.0 or a["type"] == "walker" or a["backing"]
                                     for a in here)
        moving_out = odo >= p["a"]
        if not staying:
            if not moving_out:
                self.mem["pass"] = None
            else:
                self._turn_back(path, v, "it drove off")
            return
        start = min(a["s"] - a["hl"] for a in here)
        end = max(a["s"] + a["hl"] for a in here)
        more = self._blocker(path, actors, end - 0.1)
        if more is not None and more[0] < end + 10.0:
            p["ids"] += [a["id"] for a in more[2] if a["id"] not in p["ids"]]
            here += [a for a in more[2] if a not in here]
            end = max(end, more[1])
        if p.get("side", -1) < 0:
            p["off"] = min(p["off"], max(p["limit"], self._need(here, p.get("tight"))))
        else:
            p["off"] = max(p["off"], min(p["limit"], self._need(here, p.get("tight"), 1)))
        p["commit"] = min(p["commit"], self._odo(start - HALF_L - 1.0))
        committed = odo >= p["commit"]
        if not committed:
            # still in time to turn back: the way round must stay clear
            clear, who = self._left_clear(path, actors, self._odo_to_station(p["a"]),
                                          self._odo_to_station(p["e"]), p["off"], v,
                                          set(p["ids"]), p.get("oncoming_lane", False))
            p["unclear"] = 0 if clear else p.get("unclear", 0) + 1
            if not clear and not moving_out and p["unclear"] < 3:
                p["hold_now"] = v < 0.5        # look again for a moment before giving it up
                clear = True
            if not clear:
                front_gap = start - path.s0 - HALF_L
                way_back = max(RAMP_BACK_PER_M * abs(path.d0), 3.0)
                if not moving_out:
                    self.mem["pass"] = None
                    return
                if abs(path.d0) <= abs(p["off"]) / 2.0 and front_gap > way_back + 1.0:
                    self._turn_back(path, v, "%s#%s in the way round" % (who["type"], who["id"]))
                    return
                if abs(path.d0) <= abs(p["off"]) / 2.0:
                    p["hold_now"] = True     # too close to turn back: wait where it is
                    if DEBUG:
                        log("hold: %s#%s in the way round" % (who["type"], who["id"]))
                    return
                # more than half way out: on round it, rather than stop across the lane
            # the move out is never pushed later once it has begun; the way back follows
            # the end of the blocker
            b = self._odo(start - HALF_L - OUT_BEFORE)
            if any(a["backing"] for a in here):
                b = p["b"]
            if odo < p["a"]:
                p["a"] = max(odo, min(p["a"], b - self._ramp(p["off"] - p.get("from", 0.0), v)))
                p["b"] = min(p["b"], b)
            else:
                p["b"] = min(p["b"], max(b, odo + 3.0))
            p["b"] = max(p["b"], p["a"] + 2.0)
        p["c"] = max(p["b"], self._odo(end + p["margin"]), p.get("c_min", -1e9))
        back = max(RAMP_BACK_PER_M * abs(p["off"]), RAMP_BACK_PER_MS * V_PASS, RAMP_BACK_MIN)
        if committed and p.get("oncoming_lane"):
            clear, _who = self._left_clear(path, actors, self._odo_to_station(odo),
                                           self._odo_to_station(p["e"]), p["off"], v,
                                           set(p["ids"]), True)
            if not clear:
                # traffic coming: back in as soon as the blocker is behind
                p["c"] = max(p["b"], self._odo(end + 2.0))
                back = max(RAMP_BACK_MIN, 1.5 * v)
                p["hurry"] = True
        p["e"] = p["c"] + back
        if odo > p["e"]:
            self.mem["pass"] = None

    def _odo_to_station(self, odo):
        return odo - self.mem["odo"] + self.s0

    # -- decision --------------------------------------------------------------------
    def _decide(self, obs, priv):
        v = self.speed_now = float(obs["speed"])
        nominal = obs["nominal"]
        path = self.path = Path(priv["route"])
        s0 = self.s0 = path.s0
        self.red = bool(priv.get("red_light"))
        self.mem["odo"] += v * DT
        actors = self._actors(path, priv)
        limit_ms = min(V_LIMIT_MAX, float(priv.get("speed_limit") or 30.0 / 3.6))
        if limit_ms < 9.0:
            limit_ms = min(limit_ms, V_GEAR2)      # the town limit: stay in 2nd gear

        self._update_nudge(path, actors)
        blocker = self._blocker(path, actors, s0)
        near = blocker is not None and blocker[0] - s0 - HALF_L < 15.0
        self.mem["behind"] = self.mem["behind"] + DT if near and v < 0.5 else 0.0
        if blocker is None:
            self.mem["gap_static"] = GAP_STATIC
        if self.mem["pass"] is not None:
            self._track_pass(path, actors, v)
        if self.mem["pass"] is None and blocker is not None:
            self.mem["pass"] = self._plan_pass(path, actors, blocker, v, priv.get("red_light"))
        p = self.mem["pass"]
        passing = p is not None

        # the speed the car means to go, before anything in the way
        cruise = limit_ms
        if passing:
            # on a fast road (a 60 or 90 sign) the pass goes faster too; the swept path still decides
            fast = V_PASS_FAST if limit_ms < 12.0 else 10.0
            cruise = V_TIGHT if p.get("tight") else fast
            if p.get("steep") and self.mem["odo"] < p["b"]:
                cruise = min(cruise, V_STEEP)
        if self.mem["nudge"] > 0.0 and (self.nudge_gap is None or self.nudge_gap < 0.3):
            cruise = min(cruise, V_NUDGE)
        for a in actors:
            if a["type"] == "walker" and a["id"] != -1 and 0 < a["s"] - s0 < 30 and abs(a["d"]) < 4.5 \
                    and not a["beyond"]:
                cruise = min(cruise, V_NEAR_WALKER)
        own = passing or self.mem["recover"] or self.mem["nudge"] > 0.0
        cruise = min(cruise, self._curve_speed(path))

        # the swept path at that speed: where the car will really be
        self.roll = []
        roll = self._rollout(path, v, cruise, own, nominal[1])
        self.roll = [path.project(x, y) for _t, x, y, _yaw, _tr in roll]
        v_cap, worst, room = self._limit(path, actors, v)
        target = min(cruise, v_cap)
        # a red (or yellow) light on the route ahead: the planner sees one only a few metres
        # before its line, too late at this pace, so the car slows for it from further out
        light_room = self._light_room(path, priv.get("lights") or (), v)
        if light_room is not None and math.sqrt(2.0 * LIGHT_DECEL * light_room) < target:
            target, room = math.sqrt(2.0 * LIGHT_DECEL * light_room), min(room, light_room)
        executed = self._rollout(path, v, target, own, nominal[1])
        contact, who = self._first_contact(path, actors, executed)
        if DEBUG and len(executed) >= 20:
            # the swept path, for the fidelity check: (metres travelled, offset) every 0.5 s
            plan_id = "-" if p is None else "%.1f/%.2f" % (p["a"], p["off"])
            log("pred t=%s odo=%.3f d0=%.3f plan=%s path=%s" % (
                obs.get("t"), self.mem["odo"], path.d0, plan_id, ";".join(
                    "%.2f,%.3f" % (executed[k][4], path.project(executed[k][1], executed[k][2])[1])
                    for k in range(4, len(executed), 5))))
        if contact is not None:
            cap = math.sqrt(2.0 * DECEL * max(0.0, contact - 1.0))
            slow, _ = self._first_contact(path, actors, self._rollout(path, v, V_CREEP, own, nominal[1]))
            if slow is None:
                cap = max(cap, min(V_CREEP, target))
            if cap < target:
                target, worst, room = cap, who, min(room, max(0.0, contact - 1.0))
        elif passing:
            wide, _ = self._first_contact(path, actors, roll, extra=0.3)
            if wide is not None:
                target = min(target, V_PASS)          # less than 0.6 m to spare: not the fast pass
        # people: once the car has stopped for one, it goes again only when the path has been
        # clear for a moment
        if worst is not None and worst["type"] == "walker" and target < max(v, 1.0):
            self.mem["walk_clear"] = 0.0
        else:
            self.mem["walk_clear"] = self.mem["walk_clear"] + DT
        if self.mem["walk_clear"] < WALKER_CLEAR_S and v < 1.0:
            target = 0.0
        # passing a person who stands just beyond the lane: slowly
        for a in actors:
            if a["type"] == "walker" and a["id"] != -1 and not a["beyond"] and a["speed"] < 0.3 and \
                    0.0 < a["s"] - s0 < 12.0 and abs(a["d"] - self._pred_offset(a["s"])) < HALF_W + a["ha"] + 1.2:
                target = min(target, V_KERB_WALKER)
        if passing and p.get("hold_now"):
            target = 0.0
        if passing and p.get("tight_back") and self.mem["odo"] >= p["c"] - 6.0:
            target = min(target, V_TIGHT)
        if passing and contact is not None and who is not None and who["id"] in p["ids"] and \
                not p.get("back") and target < 1.0 and abs(path.d0) > 0.5 * abs(p["off"]):
            # out beside what it passes, yet the swept path still touches it (it lies wider than
            # planned, or the car tracks short): move further over, as far as the lane allows
            if p.get("side", -1) < 0:
                p["off"] = max(p["limit"], p["off"] - 0.15)
            else:
                p["off"] = min(p["limit"], p["off"] + 0.15)
        if passing and contact is not None and who is not None and who["id"] in p["ids"] and \
                self.mem["odo"] >= p["commit"] and not p.get("back"):
            # turning back in too soon would sweep the car's side into what it passes: on
            # straight a little longer, at the speed the plan had
            back = p["e"] - p["c"]
            p["c_min"] = max(p["c"], self.mem["odo"] + HALF_L + 2.0)
            p["c"], p["e"] = p["c_min"], p["c_min"] + back
            again, _who = self._first_contact(path, actors, self._rollout(path, v, V_TIGHT, True, nominal[1]))
            if again is None:
                target = max(target, min(V_TIGHT, cruise))
        # committed beside the blocker in an oncoming lane: keep moving unless the swept path
        # touches something
        if passing and p.get("oncoming_lane") and self.mem["odo"] >= p["commit"] and contact is None:
            target = max(target, min(V_COMMITTED, cruise))
        # held at zero by something that was not part of the plan: plan again from here
        if passing and target < 0.1 and worst is not None and worst["id"] not in p["ids"] and \
                (worst["type"] == "prop" or worst["speed"] < 0.3) and not p.get("back"):
            self.mem["stuck"] += DT
            if self.mem["stuck"] >= STUCK_S:
                log("stuck behind %s#%s: plan again" % (worst["type"], worst["id"]))
                self.mem["pass"], self.mem["stuck"] = None, 0.0
                new = self._blocker(path, actors, s0)
                if new is not None:
                    self.mem["pass"] = self._plan_pass(path, actors, new, v, priv.get("red_light"))
                p = self.mem["pass"]
                passing = p is not None
        else:
            self.mem["stuck"] = 0.0

        self.ahead = sum(1 for a in actors if 0.0 < a["s"] - s0 < 25.0 and not a["beyond"] and
                         self._in_corridor(a, a["s"], a["d"]))
        standing = [a["s"] - a["hl"] - s0 - HALF_L for a in actors
                    if a["type"] in ("walker", "vehicle") and a["speed"] < 0.3 and not a["beyond"] and
                    a["s"] > s0 and self._in_corridor(a, a["s"], a["d"])]
        self.caution = max(0.0, 1.0 - min(standing) / CAUTION_M) if standing else 0.0
        if DEBUG:
            log("t=%s v=%.2f cap=%.2f target=%.2f room=%.1f worst=%s pass=%s nom=%s" % (
                obs.get("t"), v, v_cap, target, room,
                None if worst is None else "%s#%s s=%.1f d=%.1f v=%.1f" % (
                    worst["type"], worst["id"], worst["s"] - s0, worst["d"], worst["speed"]),
                None if not passing else "%.1f/%.1f/%.1f/%.1f off %.2f" % tuple(
                    [p[k] - self.mem["odo"] for k in "abce"] + [p["off"]]),
                nominal) + " d0=%.2f red=%d contact=%s light=%s" % (
                    path.d0, self.red, None if contact is None else round(contact, 1),
                    None if light_room is None else round(light_room, 1)))
        # back in the lane before the planner gets the wheel again
        if passing or self.mem["nudge"] > 0.0:
            self.mem["recover"] = False
        elif abs(path.d0) > 0.8 and not path.junction_between(s0 - 4.0, s0 + 8.0):
            self.mem["recover"] = True
        elif abs(path.d0) < 0.3:
            self.mem["recover"] = False
        own = passing or self.mem["recover"] or self.mem["nudge"] > 0.0   # steering and speed
        if self.mem["recover"]:
            target = min(target, V_PASS)
        steer = self._steer_from(path, 0.0, 0.0, 0.0, v, passing) if own else nominal[1]
        steer_only = not own and abs(nominal[1] - self._steer_from(path, 0.0, 0.0, 0.0, v, False)) > STEER_TRUST
        if steer_only:
            steer = self._steer_from(path, 0.0, 0.0, 0.0, v, False)
        # at a red light the planner's own braking stops at the line (while the expert steers,
        # the light is still the planner's call); whenever the planner brakes for anything
        # else the expert has not taken over for, follow it too
        # near a junction the planner's own pace and braking stand: it sees a red light only a
        # few metres before the line, so the car must not come in faster than it would
        near_junction = path.junction_from(s0, max(25.0, 3.0 * v)) is not None
        if self.red and (own or nominal[2] > 0.0):
            action = [1, 0.0, steer, max(nominal[2], 0.3)]
        elif not own and (nominal[2] >= PLANNER_STOP or (near_junction and nominal[2] > 0.0)):
            if worst is not None and room <= 40.0 and target < v:
                # following something the expert sees: its own (gentler) envelope, no throttle
                action = self._drive(v, target, room, steer)
                action[1] = 0.0
            else:
                action = [1, 0.0, steer, nominal[2]]   # the planner knows why (a light)
        else:
            action = self._drive(v, target, room, steer)
            if self.red or (near_junction and not own):
                action[1] = min(action[1], nominal[0])
        self.mem["wheel"] = self.mem["wheel"] + (action[2] - self.mem["wheel"]) * (
            1.0 - math.exp(-DT / LAG_TAU))
        return action

    def _light_room(self, path, lights, v):
        """Metres from the front bumper to LIGHT_MARGIN short of the stop point of the nearest
        red or yellow light on the route ahead that the car stops for, or None. Whether it
        stops is decided when the light is first seen in that colour, and kept: for a red one
        if it can stop short of it at LIGHT_RED_DECEL, for a yellow one if LIGHT_YELLOW_DECEL
        is enough (otherwise it goes on through)."""
        best, decided = None, {}
        for light in lights:
            state = light.get("state")
            if state not in ("red", "yellow"):
                continue
            room = path.project(light["x"], light["y"])[0] - path.s0 - HALF_L - LIGHT_MARGIN
            if room < -1.5:
                continue                 # at the line already, or past it
            room = max(0.0, room)
            key = str(light.get("id"))
            stop = self.mem["lights"].get(key)
            if stop is None or stop[0] != state:
                most = LIGHT_YELLOW_DECEL if state == "yellow" else LIGHT_RED_DECEL
                stop = [state, v <= 1.0 or v * v / (2.0 * max(room, 0.1)) <= most]
            decided[key] = stop
            if stop[1]:
                best = room if best is None else min(best, room)
        self.mem["lights"] = decided
        return best

    def _curve_speed(self, path):
        """Speed that keeps the lateral acceleration in the coming bends below LAT_ACCEL."""
        s0, best = path.s0, V_LIMIT_MAX
        for ahead in (4.0, 8.0, 12.0, 16.0, 20.0):
            h1, h2 = path.heading(s0 + ahead), path.heading(s0 + ahead + 4.0)
            turn = abs(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1)))
            if turn > 1e-3:
                radius = 4.0 / turn
                best = min(best, math.sqrt(LAT_ACCEL * radius + 2 * DECEL * max(0.0, ahead - 4.0)))
        return best

    def _drive(self, v, target, room, steer):
        """Brake with the deceleration the stop needs; shed small amounts of speed with a light
        throttle (and cross the 2nd -> 1st downshift that way), coast below LOCK_SPEED where
        that is gentle enough; otherwise a PI throttle toward the target."""
        if target < 0.3 and v < 0.3:
            # hold: the more there is in the way just ahead, the firmer
            return [1, 0.0, steer, min(1.0, 0.5 + 0.1 * self.ahead)]
        if v > target + 0.3:
            need = (v * v - target * target) / (2.0 * max(room, 0.5))
            need = max(need, 0.8 * (v - target))
            if need < 1.5 and v > 4.0:
                return [1, LIFT_THROTTLE, steer, 0.0]           # lift: drag sheds the speed
            if SHIFT_BAND[0] < v < SHIFT_BAND[1] and need < 4.0:
                return [1, LIFT_THROTTLE, steer, 0.0]           # through the downshift on throttle
            if v < LOCK_SPEED and need < COAST_DECEL and room > 2.5:
                return [1, 0.0, steer, 0.0]                     # coast
            brake = min(1.0, max(0.05, need * BRAKE_PER_MS2))
            return [1, 0.0, steer, brake]
        if target < 0.3:
            return [1, 0.0, steer, 0.0]
        err = target - v
        i_term = min(0.7, max(0.25, self.mem["thr_i"] + 0.02 * err))
        self.mem["thr_i"] = i_term
        throttle = min(0.85, max(LIFT_THROTTLE if v > 4.0 else 0.0, i_term + 0.25 * err))
        # someone standing in the path ahead: ease off a little, the nearer the more
        throttle *= 1.0 - 0.25 * self.caution
        return [1, throttle, steer, 0.0]


def main():
    sup = Expert()
    out = sys.stdout
    out.write("READY\n")
    out.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        typ = msg.get("type")
        if typ == "reset":
            sup.reset()
            reply = {"ok": True}
        elif typ == "close":
            out.write(json.dumps({"ok": True}) + "\n")
            out.flush()
            break
        elif typ == "act":
            try:
                action = [float(a) for a in sup.act(int(msg.get("t", 0)), msg["obs"])]
                if not all(math.isfinite(a) for a in action) or action[0] == 0:
                    action = [0]
                else:
                    action = [1, min(1.0, max(0.0, action[1])),
                              min(1.0, max(-1.0, action[2])), min(1.0, max(0.0, action[3]))]
            except Exception as exc:  # noqa: BLE001
                log("act error", repr(exc))
                action = [0]
            reply = {"action": action}
        else:
            reply = {"ok": True}
        out.write(json.dumps(reply) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
