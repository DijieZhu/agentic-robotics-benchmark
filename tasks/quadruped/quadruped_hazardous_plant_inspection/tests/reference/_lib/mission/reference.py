"""The privileged reference program.

Privileges, through Truth only: the true pose, the true plan (walls, element classes, the traversability map and the
reference route), each instrument's pose, mount and acceptance region, and the unlisted source's position for planning
the survey. Never a value: every reported value is read by the bricks reference reader from the robot's own inspect
or thermal frame with the round sheet's public hint (catalog.reader_hint); the survey mark's rate comes from its own
dosimeter counts; the source position from an inverse-square fit of its own dosimeter counts.

Behaviour per item of the route order: plan on the truth map (low-friction puddles avoided where the geometry allows),
walk, and cross every element that needs a mode square to it (approach point, turn, switch mode, cross straight,
switch back); cross every slow stretch (a ramp, and a puddle where no dry way exists) straight along its axis at a
reduced speed, then re-check the heading; at the observation pose stop, stand, aim the head from the geometry, take
frames, read, report citing the frames (a frame that leaves part of the feature outside the image: the head aimed once
more from the geometric prediction); other poses inside the acceptance region when the reader is not confident; an
instrument seen only from a low space: its crouching poses, entered and left crouching, the frames taken crouching;
the mark by counting; the source by a survey around it and a fit (stances behind a shielding box left out); back
through link coverage to the pad; finish.

A 'blocked' report is handled as the robot manual says (capability.json): up to two retries in the same mode
(nav.Driver), then a mode change (a slow stretch: walk -> high_step) or a re-plan with the element avoided. Version 2:
after a `blocked` the robot takes no walking command for 5 s; the driver keeps sending its command and goes on
when it is taken again. Version 2: a walk blocked with no known element at hand backs off without turning in the
cramped place (to the nearest cell with room to turn), and the place becomes an obstacle for the planner, so the same
route is never planned again (no other route: the goal is given up); survey stances never lie on cells the map calls
tight (no turning), and walking from a survey stance in a strong field the first `blocked` ends the attempt.
Version 2: a walk that ends blocked is put down to what the reference's own map shows in contact with the body or
within reach of it there: an element that needs a mode keeps the manual's mode change; else a door leaf, a wall or
equipment at hand is named, the robot backs off and plans again with clearance from it; a free element (walk is rated
across it) is never crossed in another mode.

Version 2: the tag plates carry text only. Before each instrument report the reference reads
the plate with its own text reader (bricks.readers.tagtext, reference only) from one of its own inspection frames: a
cited frame that holds the whole plate, else one more frame aimed at the plate (a temperature point always needs
that frame); it reports only when the text read equals the work order's tag, and makes no report at all for an item
whose plate it never read as its tag. Navigation and aim stay privileged (the plate's place in a frame comes from
the true pose; the text and the plate's pose come from the frame's pixels). The plate's mirror pose (the two planar
poses a border gives) is settled publicly: with the round sheet's recorded facing (to 5 degrees), and where the
round sheet records none (an undrawn annex) or the frame shows that the recorded facing is not this plate's (a moved
or relocated instrument: the recorded place is another wall or room), with the plate's surface normal from the
robot's own navigation depth frame (one frame aimed at the plate). Never the true plate normal: that is logged next
to each read for the record only (`true_normal_err_deg`). The quarter-turn valve's reader takes the plate's pose from
the plate's border. Every plate read is in the log (kind `plate_read`, with the facing's source).
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from . import paths
from .catalog import reader_hint
from .nav import MAX_BLOCKED, Driver, Halt, crossing_points, element_axes, near_polygon, refusable, wrap

paths.setup()

from bricks import consensus as r_consensus  # noqa: E402
from bricks.sensors.thermal import emissivity_correct, hottest_region  # noqa: E402
from bricks.targets import gauge as r_gauge  # noqa: E402
from bricks.targets import lamps as r_lamps  # noqa: E402
from bricks.readers import tagtext as r_text  # noqa: E402  (version 2: the plates' text reader, reference only)
from bricks.targets import level as r_level  # noqa: E402
from bricks.targets import valve as r_valve  # noqa: E402

INSTRUMENTS = ("gauge", "level", "valve_lever", "valve_handwheel", "lamps", "thermal")
# catalogue extents (half width, half height in the face plane, m) and the offset of their centre from the face
# origin: what the reader needs in its crop (gauge case; lever + its tag below; gate plate + pointer; level plate,
# tube and fittings; lamp panel)
ROI_HALF = {"gauge": (0.09, 0.09), "valve_lever": (0.20, 0.20), "valve_handwheel": (0.08, 0.10),
            "level": (0.07, 0.30), "lamps": (0.20, 0.11)}
ROI_OFFSET = {"gauge": (0.0, -0.02), "valve_lever": (0.0, -0.02), "valve_handwheel": (-0.02, 0.0),
              "level": (0.02, 0.0), "lamps": (0.0, 0.02)}
# crossing settings (from crossing trials): a 0.27 m body in high_step, 0.15 m/s
# and an exit point 0.75 m past the sill (chosen on a draft-layout site with the classical controller, which fell with a
# 0.25 m body; the learned controller passes the ladder site's sills with them as well)
CROSS = {"before": 0.50, "after": 0.75, "speed": {"high_step": 0.15, "crouch_walk": 0.13}, "final_slow": 0.12,
         "height": {"high_step": 0.27}}
# slow stretches (a slope of 10-18 deg at reduced speed, no turning on it; a low-friction puddle where no
# dry way exists): straight along the axis in the first (mode, speed), the next ones after an attempt that ended
# blocked or stalled (true progress < 0.10 m in 6 s: slipping feet, which the stumble stop does not see). Chosen in
# trials on the ladder site's bund ramp E038 with the puddle E039 (friction 0.134) at its foot: going
# in, walk at 0.10 m/s stalled 5/5 at the ramp foot (the hind feet slip on the puddle; status stayed 'ok'), high_step
# at 0.10 stalled 5/5 and at 0.15 2/5; walk at 0.15 / 0.20 / 0.30 m/s passed 5/5 each; coming out every setting
# passed 5/5
CAREFUL = {"tries": (("walk", 0.20), ("walk", 0.30), ("high_step", 0.20)), "before": 0.30, "after": 0.45,
           "max_s": 60.0, "stall": (6.0, 0.10)}
# a crossing that ended blocked and is the only way to the goal is tried this many more times (backed off first)
CROSS_RETRIES = 2
WALK_STALL = (8.0, 0.10)        # a walk that gains less than 0.10 m along its path in 8 s is stalled (true pose: it
                                # sees feet treading in place against an obstacle, which the controller may not report)
GOAL_MISS_M = 0.25              # go() re-plans (after backing off) when its walk ended farther than this from the goal
# the straight exit past a needs_high_step element must leave the hind feet clear of it with the whole body (0.55 m
# long; the traversability map's clearance is the inscribed radius 0.2025 m): free length >= half depth + EXIT_MIN
EXIT_MIN = 0.40
SURVEY = {"n_points": 5, "r_min": 1.2, "r_max": 2.6, "dwell_s": 8.0, "z_source": 0.45}
# the survey mark is counted for 60 s (about 1600 counts at 0.6 Gy/h, 2.5 % statistical error): the grader's
# truth is the field over the whole dwell inside 0.30 m, so the few seconds of the approach inside the disc weigh little
MARK_DWELL_S = 60.0
HEAD_Z_STAND = 0.295            # camera height above the floor standing at the default 0.24 m body height
HEAD_Z_CROUCH = 0.25            # camera height crouching at rest (crouch_walk; the site's crouch_height_m is 0.245)
MAST_X = 0.18                   # the camera is this far ahead of the body centre (head mast)
SLOW_KINDS = ("ramp", "slope", "puddle")
# a frame that failed only on its in-frame shares (a lever's grip end past the image edge: the first aim, 4 cm below
# the stem toward the tag, suits a lever near open; a lever near closed reaches up to 17 cm above the stem): aims on
# the face's up axis from 10 cm below to 12 cm above its centre, 2 cm apart (then 5 cm to either side), each checked
# with the geometric prediction from the camera's present place; the middle of the longest run of good ones
REAIM = {"z0": -0.10, "dz": 0.02, "n": 12, "y": (0.0, 0.05, -0.05)}
IN_FRAME_FAILS = ("in frame", "tag in frame")
# the camera frames the reference allows itself per instrument (three poses of three frames, as before the diverse
# retries): once the first pose gave no reading, poses from other parts of the region are tried while these last
REF_FRAMES_PER_ITEM = 9
# survey stances whose line to the source crosses a shielding box are left out of the fit (the inverse-square model
# has no shielding) when at least this many others remain
SURVEY_MIN_CLEAR_FIT = 4
# the survey's geometry (a development run of version 1, candidate 18: the fifth planned point could not be reached and the
# four read lay within 0.16 m of one line; the readings cannot tell the source's side of such a line, and the fit took
# the mirror side, 1.98 m off): while fewer than SURVEY["n_points"] were read, or the readings' RMS distance from their
# principal line is below SURVEY_MIN_SPREAD_M, further candidates are read (at most SURVEY_EXTRA_MAX), the one farthest
# from that line first, none within SURVEY_EXTRA_AWAY_M of a goal tried before. Every survey of that run that read its five
# points had a spread of 0.43 m or more, so this path changes no other mission of it
SURVEY_MIN_SPREAD_M = 0.35
SURVEY_EXTRA_MAX = 3
SURVEY_EXTRA_AWAY_M = 1.0
# a crossing starts within this of its approach point A (else the walk to A is tried again, then the way re-planned)
ACROSS_START_MISS_M = 0.25
# version 2: plate frames the reference takes per view pose when no cited frame showed the plate read as its
# tag (a temperature point: always, its record photos are thermal); a frame shows a plate when its four corners
# project inside the image by PLATE_EDGE_PX
PLATE_FRAMES = 2
PLATE_EDGE_PX = 6.0
# version 2: the plate's mirror pose is settled with the round sheet's recorded
# facing (public, to 5 degrees); a recorded facing farther than this from the chosen planar pose is not this plate's
# (C-moved: the recorded place is on another wall; C-relocated: another room), and then the plate's surface normal in
# the robot's own navigation depth frame settles it (at least this many depth pixels inside the plate's outline).
# 15 degrees, not 45: a relocated valve recorded at 180 deg hangs at 270; seen 33 deg off its
# face, its mirror pose lay 24 deg from the recorded facing and passed a 45 deg test (66 deg off the true normal)
FACING_AGREE_DEG = 15.0
PLATE_DEPTH_MIN_PX = 30
# version 2: a walk blocked with no known element at hand. On a development site
# the survey stance lay in a 0.60 m wide pocket between a wall and an open door leaf, on cells the
# map calls tight (no turning); the driver turned in place there toward its next point and the body met the wall and
# the leaf: 15 `blocked` in 95 s at 19 Gy/h, the same route re-planned five times. Now the robot backs off without
# turning in the cramped place (straight back, or forward, along its present heading or the heading it came in with,
# to the nearest cell with room to turn: a cell the map does not call tight), and the place becomes an obstacle for its
# planner (a disc of spot_r around the body centre where it was blocked), so that the next route is another one; when no
# other route exists the goal is given up: the same route is never planned again.
UNBLOCK = {"reach_m": 1.5, "cone_deg": 30.0, "speed": 0.12, "spot_r": 0.20, "fwd_pen_m": 0.30, "turn_pen_m": 0.20,
           "same_place_m": 0.30, "retrace_max_m": 2.5, "retrace_min_m": 0.25}
TRAIL_N = 90                    # seconds of the robot's own way kept for backing off along it
# a field in which one minute costs this share of the dose rating or more is strong (the survey's own counts at its
# stance): walking from a survey stance there, the first `blocked` ends the walk attempt (no retries in place inside the
# field) and goes to the back-off above at once (a survey stance in a strong field is cut short when it is held)
STRONG_FIELD_SHARE_PER_MIN = 0.01
# version 2: a walk that ends blocked is put down to what the reference's own map
# shows in contact with the body or within reach of it at that place, not to the nearest floor element. In the
# first admission a walk from a survey stance through the gap of an ajar door ended `blocked` with a front foot on the
# doorway's free 2 cm sill and a hind shank against the door's leaf; the reference named the sill (the nearest element
# ahead), tried a high_step crossing of it three times and stood 32 s. Now: the body's footprint (BODY_HALF) at the true
# pose grown by REACH_M; a traversal element that needs a mode (needs_high_step / needs_crouch) at hand keeps the mode
# change of the manual (or the step off it); else the layout's fixed geometry at hand (a door leaf, a wall, equipment:
# Truth.static_outlines, between OBSTACLE_Z above the floor) is what stopped the walk: the robot backs off (the back-off
# along its way in) and plans again with clearance from that thing (discs of CLEAR["r_m"], at least the body's turning
# radius, on the thing's outline within CLEAR["near_m"] of the place; kept for the rest of the mission wherever a way
# with them exists; where none exists the place itself becomes an obstacle); a free element (class free: walk is
# rated across it) is never crossed in another mode; with nothing else at hand it is a `blocked` at a rated size (the
# manual: retry): back off the way in and walk again, once per goal (then the place is an obstacle)
BODY_HALF = (0.275, 0.165)      # the robot's plan footprint 0.55 x 0.33 m (site.json robot_footprint)
REACH_M = 0.08                  # beyond the footprint: what a swinging leg or shank can touch (first 0.06 m)
OBSTACLE_Z = (0.035, 0.42)      # fixed geometry the body can meet: above a free step, below the body's top
CLEAR = {"r_m": 0.35, "near_m": 1.0, "step_m": 0.05}


class Replan(Exception):
    """A crossing ended blocked after its retries: plan again with these elements avoided. soft: the crossing was not
    tried (its straight exit is short); avoid it only if another way exists, else try it along the planner's path.
    spots: (x, y, r) discs to avoid as well (where a walk was blocked with no known element at hand).
    clear: (x, y, r) discs on the outline of the thing that stopped the walk (clearance from it, for the rest of
    the mission where a way with them exists); place: the (x, y, r) disc where it stopped, avoided when no way keeps
    that clearance."""

    def __init__(self, eids, soft=False, spots=(), clear=(), place=None):
        super().__init__(",".join(eids))
        self.eids = list(eids)
        self.soft = bool(soft)
        self.spots = [tuple(float(v) for v in s) for s in spots]
        self.clear = [tuple(float(v) for v in s) for s in clear]
        self.place = None if place is None else tuple(float(v) for v in place)


class Reference:
    clear_discs = ()                         # (an instance's own list from __init__; a bare instance has none)

    def __init__(self, robot, truth, out_dir=None):
        self.r, self.t = robot, truth
        self.out_dir = out_dir
        self.log = []
        # a lidar sweep every second while time passes (kept in the run folder for the localiser check and the
        # replay's minimap; the reference itself navigates on the true pose)
        self.d = Driver(robot, truth.pose, log=self.note, every=(1.0, self._sweep))
        self.mission = robot.mission()
        self.desc = robot.describe()
        self.rules = truth.evidence_rules()          # the grader's evidence gate (privileged; the documents state it)
        self.wo = self.mission["work_order"]
        self.items = {it["tag"]: it for it in self.wo["items"]}
        self.plate_reads = []                # every plate read (also in the log): tag, frame, text, ok
        self.plate_ok = {}                   # tag -> the frame whose plate read equalled the tag
        self._pose_frames = {}               # frame id -> inspection frame, of the present view pose
        self.site = truth.site()
        self.rooms = {r["id"]: r for r in self.site["rooms"]}
        self.elements = {e["id"]: e for e in self.site["elements"]}
        self.cam_K = {k: np.asarray(v["K"], float) for k, v in self.desc["cameras"].items()}
        self.background_gyh = None
        self.crossings = []
        # puddles a controller should not walk on (friction below the classical force distribution's 0.45; the
        # learned controller stops on them, capability.json 'friction') are planned around, first with a 0.20 m
        # margin around their discs, then 0.10 m; only where no dry way exists is a puddle crossed (slow stretch)
        self.avoid_kinds = ("puddle",) if any(e["kind"] == "puddle" and float(e["param"].get("friction", 1.0)) < 0.45
                                              for e in self.site["elements"]) else ()
        self.margins = (0.20, 0.10)
        self.avoid_ids = set()               # elements a crossing could not pass (re-plan around them)
        self.retries_left = {}               # crossings that are the only way: tries left (CROSS_RETRIES each)
        self.allow_short = set()             # short-exit crossings with no other way: tried along the planner's path
        self.mc_crossed = []                 # elements crossed by a mode change on a blocked walk, in order
        self.discs = truth.puddle_discs()
        self.view_posture = {}               # tag -> "crouch_walk" when its view poses are crouching ones
        self._crouch_exit = None             # the way back out of a low space (walking end last)
        self.blocked_spots = []              # (x, y, r) discs the planner avoids for the present goal
        self._block_places = []              # where walks toward the present goal ended blocked
        self._field_strong = False           # walking from a survey stance in a strong field
        self._grid = None                    # the truth map's grid (enterable / tight cells), for the back-off
        self._trail = []                     # (t, x, y, yaw) once a second, the way the robot came
        self.clear_discs = []                # (x, y, r) discs of clearance from things that stopped a walk
        self._obstacles = None               # the layout's fixed geometry at the body's height (Truth)

    def _sweep(self):
        self.r.get_lidar()
        # the way the robot came (true poses once a second, the last TRAIL_N), for backing off along it
        q = self.t.pose()
        self._trail.append((float(q["t"]), float(q["x"]), float(q["y"]), float(q["yaw"])))
        if len(self._trail) > TRAIL_N:
            del self._trail[0]

    def look_ahead(self, why):
        """A nav colour + depth frame straight ahead (head level, slightly down), for the record."""
        self.r.set_head(0.0, 0.25)
        self.d.wait(0.3)
        fr = refusable(self.r.get_camera, "nav", depth=True)
        if fr is not None:
            self.note("nav_frame", frame=fr["frame_id"], why=why)
        self.r.set_head(0.0, 0.0)

    # ------------------------------------------------------------------ log
    def note(self, kind, **kw):
        # (never call get_state here: it would restart the dosimeter count)
        rec = {"t": round(float(self.t.pose()["t"]), 2), "kind": kind}
        rec.update(kw)
        self.log.append(rec)
        if os.environ.get("MISSION_VERBOSE"):
            print("[reference] " + json.dumps(rec, default=_js)[:300], flush=True)

    # ------------------------------------------------------------------ navigation
    def go(self, x, y, yaw=None, tol=0.08, allow_unsafe=False, depth=0):
        """Walk to (x, y) on the truth map. Puddles are avoided where a dry way exists; a slow stretch on the way (a
        ramp, or a puddle with no dry way around it) is crossed straight and slowly (careful_cross) and the rest
        re-planned from its far side. A crossing that ends blocked after its retries makes the element avoided and
        the way re-planned. False when there is no way (the plain plan only with allow_unsafe)."""
        if depth == 0:
            self.retries_left = {}           # each goal has its own retries of only-way crossings
            self.blocked_spots = []          # ... and its own places blocked with no element at hand
            self._block_places = []
        for attempt in range(3 + CROSS_RETRIES):
            p = self.t.pose()
            plan, wet = self._plan((p["x"], p["y"]), (x, y))
            if plan is None and allow_unsafe:
                plan = self.t.plan((p["x"], p["y"]), (x, y))
                wet = True
                self.note("unsafe_plan", goal=[x, y], found=plan is not None)
            if plan is None:
                self.note("no_path", goal=[round(float(x), 3), round(float(y), 3)], avoided=sorted(self.avoid_ids))
                return False
            st = self._slow_stretch(plan, wet) if depth < 6 else None
            try:
                if st is not None:
                    A, B, ayaw, eids = st
                    self.note("slow_stretch", elements=eids, A=_r(A), B=_r(B), yaw=round(ayaw, 3), wet_plan=wet)
                    if not self.go(A[0], A[1], tol=0.06, depth=depth + 1):
                        return False
                    if not self.careful_cross(A, B, ayaw, eids):
                        raise Replan(eids)
                    return self.go(x, y, yaw=yaw, tol=tol, allow_unsafe=allow_unsafe, depth=depth + 1)
                reached = self.execute(plan, goal=(x, y), tol=tol)
            except Replan as e:
                if e.clear:
                    # stopped by a thing of the layout (the robot has backed off): clearance from it from now on;
                    # when no way to this goal keeps that clearance, the place itself is an obstacle for this goal
                    # (never the same route again)
                    self.clear_discs = list(self.clear_discs) + [s for s in e.clear if s not in self.clear_discs]
                    p = self.t.pose()
                    if self._plan_discs((p["x"], p["y"]), (x, y), tuple(self.blocked_spots) +
                                        tuple(self.clear_discs), quiet=True)[0] is None:
                        sp = [e.place] if e.place is not None else []
                        self.note("clearance_no_way", goal=[round(float(x), 3), round(float(y), 3)],
                                  place=[_r(s) for s in sp])
                        e.spots = list(e.spots) + [s for s in sp if s not in e.spots]
                    if not e.spots:
                        self.note("replan", avoided=sorted(self.avoid_ids), spots=[_r(s) for s in self.blocked_spots],
                                  clear=len(self.clear_discs), attempt=attempt)
                        continue
                if e.spots:
                    # blocked with no known element at hand (the robot has backed off): the place is an obstacle
                    # from now on for this goal, so the next route is another one; no other route: the goal is given up
                    self.blocked_spots += [s for s in e.spots if s not in self.blocked_spots]
                    p = self.t.pose()
                    if any(math.hypot(float(x) - sx, float(y) - sy) <= sr for sx, sy, sr in self.blocked_spots):
                        # the goal itself is where the robot was blocked: every route ends there (a view pose: the
                        # caller tries its next one; a survey point: the survey reads another)
                        self.note("goal_in_blocked_place", goal=[round(float(x), 3), round(float(y), 3)],
                                  spots=[_r(s) for s in self.blocked_spots], attempt=attempt)
                        return False
                    if self._plan((p["x"], p["y"]), (x, y))[0] is None:
                        self.note("no_route_around_blocked", goal=[round(float(x), 3), round(float(y), 3)],
                                  spots=[_r(s) for s in self.blocked_spots], attempt=attempt)
                        return False
                    kw = {"clear": len(self.clear_discs)} if self.clear_discs else {}
                    self.note("replan", avoided=sorted(self.avoid_ids), spots=[_r(s) for s in self.blocked_spots],
                              attempt=attempt, **kw)
                    continue
                if e.soft:
                    p = self.t.pose()
                    self.avoid_ids |= set(e.eids)
                    alt, _w = self._plan((p["x"], p["y"]), (x, y))
                    if alt is None:
                        # no other way: try the crossing along the planner's own path (it turns inside the gap)
                        self.avoid_ids -= set(e.eids)
                        self.allow_short |= set(e.eids)
                        self.note("short_exit_no_other_way", elements=e.eids)
                    else:
                        self.note("replan", avoided=sorted(self.avoid_ids), attempt=attempt)
                    continue
                self.avoid_ids |= set(e.eids)
                p = self.t.pose()
                if e.eids and self._plan((p["x"], p["y"]), (x, y))[0] is None:
                    # avoiding the element leaves no way: it is the only way, so it is tried again (backed off
                    # first) rather than every later item given up; at most CROSS_RETRIES more times per element
                    self.avoid_ids -= set(e.eids)
                    tries = [self.retries_left.setdefault(eid, CROSS_RETRIES) for eid in e.eids]
                    if min(tries) <= 0:
                        self.note("only_way_blocked", elements=e.eids)
                        return False
                    for eid in e.eids:
                        self.retries_left[eid] -= 1
                    self.note("only_way_retry", elements=e.eids, retries_left=min(tries) - 1)
                    self.r.walk(-0.10, 0.0, 0.0)
                    self.d.wait(2.0)
                    self.d.stop(0.4)
                    continue
                self.note("replan", avoided=sorted(self.avoid_ids), attempt=attempt)
                continue
            if not reached:
                p = self.t.pose()
                miss = math.hypot(p["x"] - x, p["y"] - y)
                if miss > max(GOAL_MISS_M, 3.0 * tol):
                    # the walk ended stalled or timed out away from the goal (wedged against equipment, treading in
                    # place): back off and plan again; never report a place as reached that was not
                    self.note("goal_not_reached", goal=[round(float(x), 3), round(float(y), 3)], miss_m=round(miss, 3),
                              status=self.d.last_status, attempt=attempt)
                    self.r.walk(-0.10, 0.0, 0.0)
                    self.d.wait(2.0)
                    self.d.stop(0.4)
                    continue
            if yaw is not None:
                self.d.stop(0.3)
                self.d.turn_to(yaw, tol=0.04)
            return True
        return False

    def _plan(self, a, b):
        """(plan, wet): the cheapest way with the puddles avoided (0.20 m, then 0.10 m around their discs), else the
        cheapest way that may cross a puddle (wet = True). With the clearance discs of things that stopped a walk
        where such a way exists, else without them."""
        discs = tuple(self.blocked_spots)
        if self.clear_discs:
            plan, wet = self._plan_discs(a, b, discs + tuple(self.clear_discs))
            if plan is not None:
                return plan, wet
            self.note("clearance_dropped", goal=[round(float(b[0]), 3), round(float(b[1]), 3)])
        return self._plan_discs(a, b, discs)

    def _plan_discs(self, a, b, discs, quiet=False):
        ids = tuple(sorted(self.avoid_ids))
        for k, mg in enumerate(self.margins):
            plan = self.t.plan_avoiding(a, b, avoid_kinds=self.avoid_kinds, margin=mg, avoid_ids=ids,
                                        avoid_discs=discs)
            if plan is not None:
                if k > 0 and not quiet:
                    self.note("tight_margin_plan", goal=[round(float(b[0]), 3), round(float(b[1]), 3)], margin=mg)
                return plan, False
            if not self.avoid_kinds:
                break
        if self.avoid_kinds:
            plan = self.t.plan_avoiding(a, b, avoid_kinds=(), margin=self.margins[-1], avoid_ids=ids,
                                        avoid_discs=discs)
            if plan is not None:
                if not quiet:
                    self.note("wet_plan", goal=[round(float(b[0]), 3), round(float(b[1]), 3)])
                return plan, True
        return None, False

    # ------------------------------------------------------------------ slow stretches (ramps, puddles)
    def _wet(self, x, y, margin=0.0):
        return any((x - cx) ** 2 + (y - cy) ** 2 <= (r + margin) ** 2 for _e, cx, cy, r in self.discs)

    def _careful_point(self, x, y, wet_plan):
        """The slow element a body cell at (x, y) is on, if it must be crossed carefully: a ramp / slope, or (in a
        plan that may cross a puddle) a puddle whose disc the feet would touch (body centre within 0.20 m of it)."""
        ci = self.t.cell_info(x, y)
        eid = ci.get("element")
        e = self.elements.get(eid) if eid else None
        if e is not None and e["kind"] in ("ramp", "slope"):
            return eid
        if wet_plan and self.discs:
            hit = next((d[0] for d in self.discs if (x - d[1]) ** 2 + (y - d[2]) ** 2 <= (d[3] + 0.20) ** 2), None)
            if hit is not None:
                return hit
        return None

    def _slow_stretch(self, plan, wet_plan=False):
        """The first stretch of the plan's walk waypoints over a slow element: (A, B, yaw, element ids) with A and B
        on dry, enterable floor on the stretch's axis (a ramp's slope axis through its centre, else the stretch's
        direction)."""
        pts, flags = [], []
        for sg in plan["segments"]:
            for q in sg["waypoints"]:
                pts.append(np.asarray(q, float))
                flags.append(self._careful_point(q[0], q[1], wet_plan) if sg["mode"] == "walk" else None)
        idx = [i for i, f in enumerate(flags) if f]
        if not idx:
            return None
        i0 = idx[0]
        i1 = i0
        for i in idx[1:]:
            if i - i1 > 3:
                break
            i1 = i
        eids = sorted({flags[i] for i in range(i0, i1 + 1) if flags[i]})
        ramps = [self.elements[e] for e in eids if self.elements[e]["kind"] in ("ramp", "slope")]
        if np.linalg.norm(pts[i1] - pts[i0]) < 0.10 and not ramps:
            return None
        d = pts[min(i1 + 1, len(pts) - 1)] - pts[max(i0 - 1, 0)]
        d = d / max(np.linalg.norm(d), 1e-9)
        if ramps:
            c, u, n, hu, hn = element_axes(ramps[0]["footprint"])
            ax = max((u, -u, n, -n), key=lambda v: float(v @ d))
            o = c                                   # the line through the ramp's centre along its slope axis
        else:
            ax = d
            o = pts[i0]
        s_in = min(float((pts[i] - o) @ ax) for i in range(i0, i1 + 1))
        s_out = max(float((pts[i] - o) @ ax) for i in range(i0, i1 + 1))
        A = self._dry_along(o, ax, s_in, -1.0, CAREFUL["before"])
        B = self._dry_along(o, ax, s_out, +1.0, CAREFUL["after"])
        if A is None or B is None:
            self.note("slow_stretch_no_ends", elements=eids)
            return None
        return A, B, math.atan2(ax[1], ax[0]), eids

    def _dry_ok(self, q):
        ci = self.t.cell_info(q[0], q[1])
        return "walk" in ci.get("modes", []) and not ci.get("slow") and not self._wet(q[0], q[1], margin=0.20)

    def _turn_room(self, q, r=0.12):
        """Room to turn in place at q: the enterable map (body inscribed radius) around q within r (the turning
        radius of the 0.55 x 0.33 m footprint is about 0.12 m more)."""
        for a in np.linspace(0, 2 * math.pi, 8, endpoint=False):
            if not self.t.enterable(q[0] + r * math.cos(a), q[1] + r * math.sin(a), "walk", avoid_kinds=()):
                return False
        return True

    def _dry_along(self, o, ax, s_edge, direction, extra):
        """From the stretch's edge outward along the axis: the first point that is walk-enterable, not slow and dry
        (the feet clear of every puddle disc); then up to `extra` farther while that holds, the farthest such point
        with room to turn in place (else the first one)."""
        first = None
        for k in range(0, 41):
            s = s_edge + direction * 0.05 * k
            q = o + s * ax
            if self._dry_ok(q):
                first = s
                break
        if first is None:
            return None
        best = o + first * ax
        good = best if self._turn_room(best) else None
        for k in range(1, int(round(extra / 0.05)) + 1):
            q = o + (first + direction * 0.05 * k) * ax
            if not self._dry_ok(q):
                break
            if self._turn_room(q):
                good = q
        return good if good is not None else best

    def careful_cross(self, A, B, yaw, eids):
        """Cross a slow stretch straight from A to B: turn square to its axis at A, the CAREFUL mode and speed with
        the heading held; after the third blocked report (two retries) once more in the fallback mode; at B stop and
        re-check the heading (the leg odometry drifts on slopes: logged against the true heading)."""
        self.d.stop(0.3)
        self.d.turn_to(yaw, tol=0.03)
        self.look_ahead("approach %s" % ",".join(eids))
        odo0 = self.r.get_state()["odometry"]
        ok = False
        nb = self.d.blocked
        used = None
        attempts = []
        for k, (mode, speed) in enumerate(CAREFUL["tries"]):
            used = (mode, speed)
            if k > 0:
                # back to A along the axis (straight, backwards) before the next attempt
                self.d.set_mode("walk")
                p = self.t.pose()
                self.r.walk(-0.10, 0.0, 0.0)
                back = float(np.dot(np.array([p["x"], p["y"]]) - np.asarray(A, float),
                                    np.array([math.cos(yaw), math.sin(yaw)])))
                self.d.wait(max(0.0, back) / 0.10)
                self.d.stop(0.4)
                self.d.turn_to(yaw, tol=0.03)
            self.d.set_mode(mode)
            p0 = self.t.pose()
            ok = self.d.follow([[p0["x"], p0["y"]], [float(B[0]), float(B[1])]], mode=mode, tol=0.10,
                               heading_hold=yaw, speed=speed, final_slow=0.3, max_s=CAREFUL["max_s"],
                               stall=CAREFUL["stall"])
            attempts.append({"mode": mode, "speed": speed, "status": self.d.last_status})
            self.d.stop(0.4)
            if ok:
                break
            self.note("careful_attempt_failed", elements=eids, mode=mode, speed=speed, status=self.d.last_status)
        self.d.set_mode("walk")
        p1 = self.t.pose()
        odo1 = self.r.get_state()["odometry"]
        # heading re-check after the stretch (no turning on a slope; the manual: odometry drifts on
        # slopes, re-localise after ramps): compare the odometry's heading change with the true one, then square up
        dyaw_odo = wrap(odo1["yaw"] - odo0["yaw"])
        dyaw_true = wrap(p1["yaw"] - yaw)
        self.d.turn_to(yaw, tol=0.03)
        self.crossings.append({"elements": eids, "mode": used[0], "kind": "slow", "speed": used[1], "A": _r(A),
                               "B": _r(B), "reached": bool(ok), "blocked": self.d.blocked - nb, "attempts": attempts,
                               "end": [round(p1["x"], 3), round(p1["y"], 3)]})
        self.note("careful_crossing", elements=eids, ok=bool(ok), mode=used[0], speed=used[1],
                  blocked=self.d.blocked - nb, heading_error_after_deg=round(math.degrees(dyaw_true), 2),
                  odometry_yaw_change_deg=round(math.degrees(dyaw_odo), 2),
                  odometry_xy_change=[round(odo1["x"] - odo0["x"], 3), round(odo1["y"] - odo0["y"], 3)],
                  true_xy_change=[round(p1["x"] - A[0], 3), round(p1["y"] - A[1], 3)])
        return bool(ok)

    # ------------------------------------------------------------------ crossings of needs_* elements
    def _crossing(self, seg, prev_pt, next_pt):
        """(A, B, yaw, mode, element ids) of a special segment."""
        mode = seg["mode"]
        wps = np.asarray(seg["waypoints"], float)
        a = np.asarray(prev_pt if prev_pt is not None else wps[0], float)
        b = np.asarray(next_pt if next_pt is not None else wps[-1], float)
        if mode == "high_step":
            hits = []
            for e in self.elements.values():
                if e["class"] != "needs_high_step":
                    continue
                if _poly_near(e["footprint"], wps, 0.08):
                    hits.append(e)
            if len(hits) == 1:
                A, B, n = crossing_points(hits[0]["footprint"], wps[0], wps[-1], before=CROSS["before"],
                                          after=CROSS["after"])
                A, B = self._fit_free(A, B, n)
                c_e, _u, _n, _hu, hn = element_axes(hits[0]["footprint"])
                q = A + float(np.dot(c_e - A, n)) * n        # where the crossing line meets the sill's centreline
                free = self._free_length(q, n, hn + CROSS["after"])
                if free < hn + EXIT_MIN and hits[0]["id"] not in self.allow_short:
                    # the straight way out is too short for the body (an ajar door leaf right behind the sill: the
                    # planner's path turns inside the gap, which the 0.55 m body cannot): another way if there is one
                    self.note("crossing_infeasible", element=hits[0]["id"], free_exit_m=round(free, 3),
                              need_m=round(hn + EXIT_MIN, 3))
                    raise Replan([hits[0]["id"]], soft=True)
                if free < hn + EXIT_MIN:
                    return A, None, math.atan2(n[1], n[0]), mode, [hits[0]["id"]]     # along the planner's path
                B = q + min(float(np.dot(B - q, n)), free - 0.05) * n
                return A, B, math.atan2(n[1], n[0]), mode, [hits[0]["id"]]
        dvec = wps[-1] - wps[0]
        nrm = float(np.linalg.norm(dvec))
        n = dvec / max(nrm, 1e-9)
        A, B = self._fit_free(wps[0] - 0.30 * n, wps[-1] + 0.30 * n, n)
        ids = [e["id"] for e in self.elements.values() if e["class"] in ("needs_crouch", "needs_high_step")
               and _poly_near(e["footprint"], wps, 0.05)]
        return A, B, math.atan2(n[1], n[0]), mode, ids

    def _swept_elements(self, p0, p1, exclude=()):
        """The needs_* elements (other than `exclude`) whose footprint the straight line from p0 to p1 (true poses)
        goes over: a square crossing that crossed them as well."""
        a = np.array([p0["x"], p0["y"]], float)
        b = np.array([p1["x"], p1["y"]], float)
        L = float(np.linalg.norm(b - a))
        out = set()
        if L < 1e-6:
            return out
        qs = [a + (b - a) * (s_ / L) for s_ in np.linspace(0.0, L, max(2, int(L / 0.03) + 1))]
        for e in self.elements.values():
            if e["id"] in exclude or e["class"] not in ("needs_high_step", "needs_crouch"):
                continue
            P = np.asarray(e["footprint"], float)
            if any(near_polygon(P, q, 0.0) for q in qs):
                out.add(e["id"])
        return out

    def _free_length(self, q, n, upto, lateral=(-0.06, 0.0, 0.06)):
        """How far the body centre can go straight from q along n (up to `upto`) on cells enterable in some mode,
        on the line and 6 cm to either side of it."""
        lat = np.array([-n[1], n[0]])
        t = 0.0
        while t <= upto + 1e-9:
            for dl in lateral:
                p = np.asarray(q, float) + t * n + dl * lat
                if not self.t.cell_info(p[0], p[1]).get("modes"):
                    return t
            t += 0.025
        return upto

    def _free_length_mode(self, q, n, upto, mode):
        """How far the body centre can go straight from q along n (up to `upto`) on cells enterable in `mode` or in
        walk (the element's own cells count: they need the mode), on the line and 6 cm to either side of it."""
        lat = np.array([-n[1], n[0]])
        t = 0.0
        while t <= upto + 1e-9:
            for dl in (-0.06, 0.0, 0.06):
                p = np.asarray(q, float) + t * n + dl * lat
                m = self.t.cell_info(p[0], p[1]).get("modes") or []
                if mode not in m and "walk" not in m:
                    return t
            t += 0.025
        return upto

    def _fit_free(self, A, B, n):
        """Pull A / B toward the element until they are enterable in walk (and not on a puddle)."""
        A, B = np.asarray(A, float), np.asarray(B, float)
        ak, mg = self.avoid_kinds, self.margins[-1]
        for _ in range(8):
            if self.t.enterable(A[0], A[1], "walk", avoid_kinds=ak, margin=mg):
                break
            A = A + 0.05 * n
        for _ in range(8):
            if self.t.enterable(B[0], B[1], "walk", avoid_kinds=ak, margin=mg):
                break
            B = B - 0.05 * n
        return A, B

    def _split_multi(self, segs):
        """A high_step segment over two or more needs_high_step elements (the planner keeps the mode between them)
        becomes one high_step segment per element with walk segments in between, so that each is crossed square."""
        out = []
        hs = [e for e in self.elements.values() if e["class"] == "needs_high_step"]
        for sg in segs:
            if sg["mode"] != "high_step" or len(sg["waypoints"]) < 3:
                out.append(sg)
                continue
            wps = sg["waypoints"]
            tags = [next((e["id"] for e in hs if _poly_near(e["footprint"], [q, q], 0.10)), None) for q in wps]
            ids = [t for t in dict.fromkeys(tags) if t]
            if len(ids) < 2:
                out.append(sg)
                continue
            # runs: consecutive waypoints of one element (with one neighbour each side), walk in between
            parts, cur, cur_id = [], [wps[0]], tags[0]
            for q, tg in zip(wps[1:], tags[1:]):
                if (tg is None) == (cur_id is None) and (tg == cur_id or tg is None):
                    cur.append(q)
                else:
                    parts.append(("high_step" if cur_id else "walk", cur))
                    cur, cur_id = [cur[-1], q], tg
            parts.append(("high_step" if cur_id else "walk", cur))
            for m, pts in parts:
                if len(pts) >= 2:
                    out.append({"mode": m, "waypoints": pts})
            self.note("split_high_step_segment", elements=ids, parts=[m for m, _ in parts])
        return out

    def execute(self, plan, goal=None, tol=0.08):
        segs = self._split_multi(plan["segments"])
        # actions: walk paths and crossings
        k = 0
        pending = []           # walk waypoints collected so far
        p = self.t.pose()
        pending.append([p["x"], p["y"]])
        swept = set()          # elements the last square crossing's straight line went over as well
        last_cross = None      # (B, unit direction) of the last square crossing
        while k < len(segs):
            s = segs[k]
            if s["mode"] == "walk":
                pending += s["waypoints"][1:] if pending else s["waypoints"]
                k += 1
                continue
            prev_pt = pending[-2] if len(pending) >= 2 else None
            nxt = segs[k + 1]["waypoints"][1] if k + 1 < len(segs) and len(segs[k + 1]["waypoints"]) > 1 else None
            A, B, yaw, mode, ids = self._crossing(s, prev_pt, nxt)
            if ids and set(ids) <= swept and last_cross is not None:
                # the previous square crossing went over this element too (a development site: a hose 0.2-0.7 m behind a
                # door sill; walking back to this crossing's approach point stepped onto the hose, then the robot
                # crossed the sill twice more by mode changes: 3 crossings, 19 holds, about 150 s): skip it, and the
                # next walk segment's leading points that are still behind the robot
                self.note("crossing_done_by_previous", elements=ids)
                k += 1
                if k < len(segs) and segs[k]["mode"] == "walk":
                    w = segs[k]["waypoints"]
                    j = after_crossing_index(w, last_cross[0], last_cross[1])
                    segs = segs[:k] + [{"mode": "walk", "waypoints": [w[0]] + w[j:]}] + segs[k + 1:]
                continue
            swept, last_cross = set(), None
            along = None
            if B is None or (mode == "crouch_walk" and _bends(s["waypoints"], 0.10)):
                # along the planner's own waypoints: a short-exit sill with no other way (the path turns in the gap)
                # or a crouch stretch that bends; square approach to its first direction
                wps = [list(q) for q in s["waypoints"]]
                tail = []
                if k + 1 < len(segs) and segs[k + 1]["mode"] == "walk":
                    tail = _take_length(segs[k + 1]["waypoints"][1:], 0.6)
                along = wps + tail
                B = np.asarray(along[-1], float)
                if mode == "crouch_walk":
                    d0 = _direction(along, 0.5)
                    A, _ = self._fit_free(np.asarray(along[0], float) - 0.30 * d0, B, d0)
                    yaw = math.atan2(d0[1], d0[0])
            path = _trim_to(pending, A)
            n_mc = len(self.mc_crossed)
            self._walk(path, tol=0.06)
            if not (ids and set(ids) <= set(self.mc_crossed[n_mc:])):
                # the crossing starts at A or not at all: a walk that ended stalled or blocked away from A (practice
                # t2 with text plates, v2: 0.6 m beside the door, the crossing then ran into the wall three times) is
                # backed off and walked again, at most twice; then the way is planned again from where the robot is
                for _try in range(2):
                    p = self.t.pose()
                    miss = math.hypot(p["x"] - A[0], p["y"] - A[1])
                    if miss <= ACROSS_START_MISS_M:
                        break
                    self.note("crossing_start_missed", elements=ids, miss_m=round(miss, 3), attempt=_try)
                    self.r.walk(-0.10, 0.0, 0.0)
                    self.d.wait(2.0)
                    self.d.stop(0.4)
                    p = self.t.pose()
                    self._walk([[p["x"], p["y"]], A.tolist()], tol=0.06)
                p = self.t.pose()
                if math.hypot(p["x"] - A[0], p["y"] - A[1]) > ACROSS_START_MISS_M:
                    raise Replan([])
            if ids and set(ids) <= set(self.mc_crossed[n_mc:]):
                # the walk to A was blocked by this very element and crossed it by a mode change: the robot is on
                # the far side already (crossing it again from there walks away from B: practice site t1)
                p1 = self.t.pose()
                self.note("crossing_done_on_approach", elements=ids, mode=mode)
            else:
                self.d.stop(0.3)
                self.d.turn_to(yaw, tol=0.04)
                self.look_ahead("approach %s" % ",".join(ids) if ids else "approach")
                self.d.set_mode(mode)
                if mode in CROSS["height"]:
                    self.r.set_body(height=CROSS["height"][mode])
                    self.d.wait(0.6)
                p0 = self.t.pose()
                nb = self.d.blocked
                if along is not None:
                    self.note("crossing_along_plan", elements=ids, mode=mode, n_points=len(along))
                    ok = self.d.follow([[p0["x"], p0["y"]]] + along, mode=mode, tol=0.10, lookahead=0.25,
                                       speed=CROSS["speed"].get(mode), final_slow=CROSS["final_slow"], max_s=120.0)
                else:
                    ok = self.d.follow([[p0["x"], p0["y"]], B.tolist()], mode=mode, tol=0.08, heading_hold=yaw,
                                       speed=CROSS["speed"].get(mode), final_slow=CROSS["final_slow"], max_s=90.0)
                status = self.d.last_status
                self.d.stop(0.4)
                self.d.set_mode("walk")
                p1 = self.t.pose()
                rec = {"elements": ids, "mode": mode, "A": _r(A), "B": _r(B), "reached": bool(ok),
                       "blocked": self.d.blocked - nb, "end": [round(p1["x"], 3), round(p1["y"], 3)]}
                if along is not None:
                    rec["along_plan"] = True
                self.crossings.append(rec)
                self.note("crossing", elements=ids, mode=mode, ok=bool(ok), blocked=self.d.blocked - nb,
                          status=status)
                if not ok and status == "blocked" and ids:
                    # three blocked reports in the element's own mode: re-plan around it
                    raise Replan(ids)
                if along is None:
                    swept = self._swept_elements(p0, p1, exclude=ids)
                    last_cross = (np.asarray([p1["x"], p1["y"]], float), np.array([math.cos(yaw), math.sin(yaw)]))
                    if swept:
                        self.note("crossing_swept", elements=ids, also=sorted(swept))
            pending = [[p1["x"], p1["y"]]]
            k += 1
            # drop the leading points of the next walk segment that are still on the crossing (not beyond B along the
            # crossing, within the crossing's lane); the path may turn right after the element, so never more
            if k < len(segs) and segs[k]["mode"] == "walk":
                w = segs[k]["waypoints"]
                if along is not None:
                    W = np.asarray(w, float)
                    j = int(np.argmin(np.linalg.norm(W - np.asarray([p1["x"], p1["y"]]), axis=1))) + 1
                else:
                    j = after_crossing_index(w, B, np.array([math.cos(yaw), math.sin(yaw)]))
                segs = segs[:k] + [{"mode": "walk", "waypoints": [w[0]] + w[j:]}] + segs[k + 1:]
        if goal is not None:
            pending.append(list(goal))
        reached = True
        if len(pending) >= 2:
            reached = bool(self._walk(pending, tol=tol))
        self.d.stop(0.3)
        return reached

    def _walk(self, path, tol=0.08):
        """Follow a walk path; when it ends blocked (three reports, e.g. a free element the stumble stop does not
        accept), cross the nearest element ahead square in high_step, then go on (a mode change, the manual)."""
        for attempt in range(3):
            p_start = self.t.pose()
            # from a survey stance in a strong field the first `blocked` ends the attempt (no retries in place)
            ok = self.d.follow(path, mode="walk", tol=tol, stall=WALK_STALL,
                               max_blocked=1 if self._field_strong else MAX_BLOCKED)
            if ok or self.d.last_status != "blocked":
                return ok
            p = self.t.pose()
            e = self._element_ahead(p)
            n_dir = np.array([math.cos(p["yaw"]), math.sin(p["yaw"])])
            if e is None:
                # version 2: the element the remaining walk path meets, whichever way the robot faces after its
                # back-offs (a path that turned back over the element just crossed: practice t1, v2 code, the hold
                # after each `blocked` made the robot retry from a stand and re-plan the same way 115 times)
                hit = self._element_on_path(path, p)
                if hit is not None:
                    e, n_dir = hit
            # what the reference's own map shows at the body: an element ahead or on the way counts only when
            # it is at hand and needs a mode (a free element: walk is rated across it, never another mode for it)
            hand = self._at_hand(p)
            not_by = None
            if e is not None:
                why = "free" if e["class"] == "free" else (None if e["id"] in hand["elements"] else "out_of_reach")
                if why is not None:
                    not_by = {"element": e["id"], "kind": e["kind"], "class": e["class"], "why": why,
                              "dist_m": hand["elements"].get(e["id"])}
                    e = None
            obst = hand["obstacles"][0] if hand["obstacles"] else None
            beside = [k for k in hand["elements"] if self.elements[k]["class"] in ("needs_high_step", "needs_crouch")]
            if e is not None:
                cause = {"element": e["id"], "how": "ahead"}
            elif beside:
                cause = {"element": min(beside, key=lambda k: hand["elements"][k]), "how": "beside"}
            elif obst is not None:
                cause = {k: obst[k] for k in ("what", "geom", "element", "dist_m")}
            else:
                cause = "none at hand"
            kw = {"not_by": not_by} if not_by is not None else {}
            self.note("walk_blocked", element=None if e is None else e["id"], attempt=attempt, cause=cause,
                      at_hand={"elements": dict(sorted(hand["elements"].items())),
                               "obstacles": [[o["what"], o["dist_m"]] for o in hand["obstacles"]]}, **kw)
            if e is None:
                # nothing that needs a mode ahead or on the way: a foot of the turning body on an element beside it
                # (practice t1, v2 code: the robot 0.4 m from a sill turned in place toward its next point and put a
                # foot on the sill, every 5 s for 2000 s); walk straight off it, without turning, before planning
                # again (only an element at hand)
                if self._step_clear(p, hand=hand):
                    # a second block for this goal at the same place makes the place an obstacle
                    raise Replan([], spots=self._repeat_spot(p))
                if obst is not None:
                    # a door leaf, a wall or equipment at hand stopped the walk: back off (the back-off, along
                    # the way in), then plan again with clearance from it; a second block for this goal at the same
                    # place makes the place an obstacle as well
                    again = self._repeat_spot(p, remember=False)
                    clear = self._clearance_discs(hand["obstacles"], p)
                    spot = self._unblock(p, p_start)
                    raise Replan([], spots=again, clear=clear, place=spot)
                if (not_by is not None and not_by["why"] == "free") or \
                        any(self.elements[k]["class"] == "free" for k in hand["elements"]):
                    # only a free element at hand (a foot on a low sill): a `blocked` at a rated size, which the
                    # manual answers with a retry in the same mode; back off the way in and walk again, once for this
                    # goal (a second block for it makes the place an obstacle, never the same route twice)
                    first = not self._block_places
                    spot = self._unblock(p, p_start)
                    raise Replan([], spots=[] if first else [spot])
                # no known element at hand: back off without turning
                # in the cramped place, and plan another route around the place where it was blocked
                raise Replan([], spots=[self._unblock(p, p_start)])
            A, B, n = crossing_points(e["footprint"], [p["x"] - 0.6 * n_dir[0], p["y"] - 0.6 * n_dir[1]],
                                      [p["x"] + 1.0 * n_dir[0], p["y"] + 1.0 * n_dir[1]], before=CROSS["before"],
                                      after=CROSS["after"])
            mode = "crouch_walk" if e["class"] == "needs_crouch" or e["kind"] == "low_overhead" else "high_step"
            yaw = math.atan2(n[1], n[0])
            # the exit B ends where the crossing's mode (or walk) can still enter the floor beyond the element
            # (on a development site 0.75 m past a floor pipe lay a crouch-only space under a rack; the crossing in
            # high_step met its beam, 3 `blocked`, and the element was given up although the robot was across)
            c_e, _u, _n, _hu, hn_e = element_axes(e["footprint"])
            q_c = A + float(np.dot(c_e - A, n)) * n
            free = self._free_length_mode(q_c, n, hn_e + CROSS["after"], mode)
            if free < hn_e + CROSS["after"]:
                B = q_c + max(hn_e + 0.10, free - 0.05) * n
            # the robot stands at the element that stopped it (a foot against it). The move back to the
            # approach point and the turn there are made in the element's own mode, and the move without turning
            # (straight back, the present heading held): in walk the follower turned in place toward A and walk's
            # step guard refused every step near the element (on development sites at a door sill, at a floor pipe,
            # at a sill beside a hose: 3-6 more `blocked` and 38-46 s standing before the
            # crossing)
            self.d.set_mode(mode)
            self.wait_hold()
            q0 = self.t.pose()
            moved_a = self._move_straight(A, float(q0["yaw"]), mode=mode)
            self.d.stop(0.3)
            self.d.turn_to(yaw, tol=0.04)
            self.note("mode_change_approach", element=e["id"], mode=mode, to_A_m=round(moved_a, 3),
                      miss_A_m=round(math.hypot(self.t.pose()["x"] - A[0], self.t.pose()["y"] - A[1]), 3))
            p0 = self.t.pose()
            nb = self.d.blocked
            okc = self.d.follow([[p0["x"], p0["y"]], B.tolist()], mode=mode, tol=0.08, heading_hold=yaw,
                                speed=CROSS["speed"].get(mode), final_slow=CROSS["final_slow"], max_s=60.0)
            self.d.stop(0.4)
            self.d.set_mode("walk")
            p1 = self.t.pose()
            self.crossings.append({"elements": [e["id"]], "mode": mode, "kind": "mode_change", "A": _r(A), "B": _r(B),
                                   "reached": bool(okc), "blocked": self.d.blocked - nb,
                                   "end": [round(p1["x"], 3), round(p1["y"], 3)]})
            self.note("mode_change_crossing", element=e["id"], mode=mode, ok=bool(okc), element_class=e["class"])
            if not okc:
                raise Replan([e["id"]])
            self.mc_crossed.append(e["id"])
            # continue along the rest of the path from the nearest point beyond B
            P = np.asarray(path, float)
            j = int(np.argmin(np.linalg.norm(P - np.asarray([p1["x"], p1["y"]]), axis=1)))
            path = [[p1["x"], p1["y"]]] + [list(q) for q in P[j + 1:]]
            if len(path) < 2:
                return True
        return False

    def _step_clear(self, p, within=0.75, dist=0.40, speed=0.15, hand=None):
        """Walk straight away (no turning), in the element's mode, from the nearest needs_* element whose footprint is
        within `within` of the body centre, `dist` metres, in the body frame's own directions (forward / sideways as
        the way away lies); then walk mode again. Only an element at hand (`hand`, _at_hand: the body's footprint
        within REACH_M of the element), not one that is merely near."""
        xy = np.array([p["x"], p["y"]], float)
        best = None
        if hand is None:
            hand = self._at_hand(p)
        for e in self.elements.values():
            if e["class"] not in ("needs_high_step", "needs_crouch") or e["id"] not in hand["elements"]:
                continue
            P = np.asarray(e["footprint"], float)
            # nearest point of the footprint polygon to the body centre
            for i in range(len(P)):
                a, b = P[i], P[(i + 1) % len(P)]
                ab = b - a
                u = float(np.clip(np.dot(xy - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
                q = a + u * ab
                d = float(np.linalg.norm(xy - q))
                if best is None or d < best[0]:
                    best = (d, q, e["id"], e["class"])
        if best is None or best[0] > within:
            return False
        away = (xy - best[1]) / max(best[0], 1e-6)
        c, s_ = math.cos(p["yaw"]), math.sin(p["yaw"])
        bx, by = c * away[0] + s_ * away[1], -s_ * away[0] + c * away[1]
        # the way away in the body frame at `speed`, sideways cut to 0.10 m/s (walk's sideways limit is 0.20)
        vx, vy = float(np.clip(bx * speed, -0.25, 0.25)), float(np.clip(by * speed, -0.10, 0.10))
        v = math.hypot(vx, vy)
        # in the element's own mode: a body that straddles a sill has feet on both sides, and walking off it lifts a
        # foot onto it, which walk's step guard refuses
        mode = "crouch_walk" if best[3] == "needs_crouch" else "high_step"
        self.note("step_clear", element=best[2], from_m=round(best[0], 3), vx=round(vx, 3), vy=round(vy, 3), mode=mode)
        if v < 1e-3:
            return False
        # _walk calls this at the third `blocked` of a walk, i.e. at the start of the 5 s hold: the mode is set
        # during the hold (taken at once), then the rest of the hold is waited out (a first version walked inside the hold:
        # every command refused, 0.06-0.09 m moved, the back-off). The walk ends when the body centre
        # has moved `dist` (the true pose), not after `dist / v` of commanded time: off a floor pipe or a hose the
        # feet slow the body (measured: 0.28 m at the hall's 5.2 cm floor pipe, 0.33 m at its hose)
        self.d.set_mode(mode)
        waited = self.wait_hold()
        t_cap = 2.0 * dist / v + 6.0
        t, t_all, refused = 0.0, 0.0, 0
        p_start = self.t.pose()
        moved = 0.0
        while moved < dist and t_all < t_cap:
            rep = self.r.walk(vx, vy, 0.0)
            if isinstance(rep, dict) and rep.get("held"):
                # a new `blocked` on the way off: wait that hold out as well, then go on
                refused += 1
                t0 = float(self.t.pose()["t"])
                self.wait_hold(rep)
                t_all += float(self.t.pose()["t"]) - t0
                continue
            self.d.step()
            t += self.d.tick * 0.01
            t_all += self.d.tick * 0.01
            q = self.t.pose()
            moved = math.hypot(q["x"] - p_start["x"], q["y"] - p_start["y"])
        self.d.stop(0.4)
        p_end = self.t.pose()
        self.note("step_clear_done", element=best[2], moved_m=round(math.hypot(p_end["x"] - p_start["x"],
                                                                                 p_end["y"] - p_start["y"]), 3),
                  walked_s=round(t, 2), hold_waited_s=round(waited, 2), refused=refused,
                  reached=bool(moved >= dist))
        self.d.set_mode("walk")
        return True

    def wait_hold(self, rep=None):
        """Wait out the hold after a `blocked`: while it lasts a walking command is refused with the time it ends
        (a zero command asks without moving anything). The simulated seconds waited."""
        t0 = float(self.t.pose()["t"])
        for _ in range(4):
            if rep is None:
                rep = self.r.walk(0.0, 0.0, 0.0)
            if not (isinstance(rep, dict) and rep.get("held")):
                break
            self.d.wait(max(0.02, float(rep["hold_until"]) - float(self.t.pose()["t"]) + 0.015))
            rep = None
        return float(self.t.pose()["t"]) - t0

    # ------------------------------------------------------------------ blocked with no element at hand
    def _repeat_spot(self, p, remember=True):
        """[(x, y, r)] when a walk toward the present goal already ended blocked within UNBLOCK["same_place_m"] of p
        (the place becomes an obstacle: the same route is not planned again), else []; p is remembered (remember)."""
        xy = (float(p["x"]), float(p["y"]))
        again = any(math.hypot(xy[0] - q[0], xy[1] - q[1]) <= UNBLOCK["same_place_m"] for q in self._block_places)
        if remember:
            self._block_places.append(xy)
        return [(xy[0], xy[1], UNBLOCK["spot_r"])] if again else []

    # ------------------------------------------------------------------ what the map shows at the body
    def _map_obstacles(self):
        """The layout's fixed geometry the body can meet (Truth.static_outlines between OBSTACLE_Z above the floor),
        without the traversal elements that are crossed (free, needs_*, slow: their footprints are checked as
        elements); a tight or barrier element's geometry (an ajar door's leaf, a high kerb) is an obstacle."""
        if self._obstacles is None:
            out = []
            for o in self.t.static_outlines():
                if o["class"] in ("free", "needs_high_step", "needs_crouch", "slow"):
                    continue
                if o["zmax"] <= OBSTACLE_Z[0] or o["zmin"] >= OBSTACLE_Z[1] or len(o["outline"]) < 2:
                    continue
                out.append(dict(o, outline=np.asarray(o["outline"], float), c=np.asarray(o["c"], float)))
            self._obstacles = out
        return self._obstacles

    def _at_hand(self, p, reach=REACH_M):
        """What the reference's own map shows in contact with the body or within `reach` of its footprint at
        pose p: {"elements": {traversal element id (not a narrow gap or puddle): plan distance m}, "obstacles": [the
        fixed geometry's records with "dist_m", nearest first]}."""
        R = _body_rect(p)
        c = np.array([float(p["x"]), float(p["y"])])
        r_body = math.hypot(*BODY_HALF) + reach
        els = {}
        for e in self.elements.values():
            if e["kind"] in ("narrow_gap", "puddle") or \
                    e["class"] not in ("free", "needs_high_step", "needs_crouch", "slow"):
                continue
            fp = np.asarray(e["footprint"], float)
            if float(np.min(np.linalg.norm(fp - c, axis=1))) > r_body + 1.5 * float(np.max(np.ptp(fp, axis=0))) + 0.05:
                continue
            d = _poly_dist(R, fp)
            if d <= reach:
                els[e["id"]] = round(d, 3)
        obs = []
        for o in self._map_obstacles():
            if float(np.linalg.norm(o["c"] - c)) > r_body + o["r"] + 0.01:
                continue
            d = _poly_dist(R, o["outline"])
            if d <= reach:
                obs.append({"what": o["what"], "geom": o["geom"], "element": o["element"], "class": o["class"],
                            "dist_m": round(d, 3), "outline": o["outline"]})
        obs.sort(key=lambda o: (o["dist_m"], o["geom"]))
        return {"elements": els, "obstacles": obs}

    def _clearance_discs(self, obstacles, p):
        """Discs of CLEAR["r_m"] (at least the body's turning radius) every CLEAR["step_m"] on the outlines of the
        things at hand, where they lie within CLEAR["near_m"] of the body centre: the planner keeps the body clear of
        them there."""
        c = np.array([float(p["x"]), float(p["y"])])
        out = []
        for o in obstacles:
            for q in _outline_points(o["outline"], CLEAR["step_m"]):
                if float(np.linalg.norm(q - c)) <= CLEAR["near_m"]:
                    s = (round(float(q[0]), 3), round(float(q[1]), 3), CLEAR["r_m"])
                    if s not in out:
                        out.append(s)
        return out

    def _turn_cells(self):
        """(origin, res, walk-enterable, walk-enterable and not tight) of the truth map; tight = less clearance than
        the body's turning circle (the map's 'walk, tight (no turning)')."""
        if self._grid is None:
            g = self.t.trav_grid()
            en = np.asarray(g["enter"]["walk"], bool)
            self._grid = (np.asarray(g["origin"], float), float(g["res"]), en,
                          en & ~np.asarray(g["tight"]["walk"], bool))
        return self._grid

    def _line_enterable(self, a, b, step=0.05):
        """The straight line a -> b on walk-enterable cells of the truth map (every `step` m)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        L = float(np.linalg.norm(b - a))
        for s_ in np.linspace(0.0, L, max(2, int(L / step) + 1)):
            q = a + (b - a) * (s_ / max(L, 1e-9))
            if not self.t.enterable(q[0], q[1], "walk", avoid_kinds=()):
                return False
        return True

    def _unblock(self, p, p_start=None):
        """A walk ended blocked with no known element at hand (the body met a wall, a door leaf or equipment
        while it turned or walked in a cramped place). Wait out the hold; turn back to the heading the walk started
        with when the robot turned in place since (that turn is what jammed it); then back off without turning there,
        along the way it came (its own true poses of the last seconds, retraced backwards, each stretch with the
        heading it had there) to the first point of that way with room to turn (a walk-enterable cell the truth map
        does not call tight) at least retrace_min_m away; without such a way, straight back or forward along that
        heading to the nearest cell with room to turn within reach_m on a straight line of enterable cells; failing
        that, 0.30 m straight back. The place where it was blocked, (x, y, spot_r), is returned for the planner to
        avoid."""
        spot = (float(p["x"]), float(p["y"]), UNBLOCK["spot_r"])
        self._block_places.append(spot[:2])
        waited = self.wait_hold()
        q = self.t.pose()
        xy = np.array([q["x"], q["y"]], float)
        turned = bool(p_start is not None and abs(wrap(float(p_start["yaw"]) - float(q["yaw"]))) > 0.30 and
                      math.hypot(float(p_start["x"]) - xy[0], float(p_start["y"]) - xy[1]) < 0.30)
        h0 = float(p_start["yaw"]) if turned else float(q["yaw"])
        o, res, en, room = self._turn_cells()

        def has_room(x, y):
            iy, ix = int(math.floor((y - o[1]) / res)), int(math.floor((x - o[0]) / res))
            return bool(0 <= iy < room.shape[0] and 0 <= ix < room.shape[1] and room[iy, ix])

        # 1. the way it came, backwards
        way, acc, prev, found = [], 0.0, xy, False
        for (_t, x, y, yaw) in reversed(self._trail):
            dd = math.hypot(x - prev[0], y - prev[1])
            if dd < 0.05:
                continue
            acc += dd
            if acc > UNBLOCK["retrace_max_m"]:
                break
            way.append((np.array([x, y]), yaw))
            prev = np.array([x, y])
            if has_room(x, y) and math.hypot(x - xy[0], y - xy[1]) >= UNBLOCK["retrace_min_m"]:
                found = True
                break
        if turned:
            self.d.turn_to(h0, tol=0.05, max_s=8.0)
        moved = 0.0
        if found:
            how = "retraced its way in (%d points, %.2f m)" % (len(way), acc)
            for pt, yaw in way:
                moved += self._move_straight(pt, yaw)
            tgt = way[-1][0]
        else:
            # 2. straight back or forward to the nearest cell with room to turn
            q = self.t.pose()
            xy = np.array([q["x"], q["y"]], float)
            r_cells = int(math.ceil(UNBLOCK["reach_m"] / res))
            iy0, ix0 = int(math.floor((xy[1] - o[1]) / res)), int(math.floor((xy[0] - o[0]) / res))
            cands = []
            for iy in range(max(0, iy0 - r_cells), min(room.shape[0], iy0 + r_cells + 1)):
                for ix in range(max(0, ix0 - r_cells), min(room.shape[1], ix0 + r_cells + 1)):
                    if room[iy, ix]:
                        c = np.array([o[0] + (ix + 0.5) * res, o[1] + (iy + 0.5) * res])
                        dd = float(np.linalg.norm(c - xy))
                        if 0.05 <= dd <= UNBLOCK["reach_m"]:
                            cands.append((dd, c))
            cands.sort(key=lambda z: z[0])
            cone = math.radians(UNBLOCK["cone_deg"])
            fwd = np.array([math.cos(float(q["yaw"])), math.sin(float(q["yaw"]))])
            best = None
            for sgn, pen in ((-1.0, 0.0), (1.0, UNBLOCK["fwd_pen_m"])):
                for dd, c in cands:
                    if best is not None and dd + pen >= best[0]:
                        break
                    u = (c - xy) / dd
                    if math.acos(float(np.clip(np.dot(u, sgn * fwd), -1.0, 1.0))) > cone:
                        continue
                    if not self._line_enterable(xy, c):
                        continue
                    best = (dd + pen, sgn, c)
                    break
            if best is None:
                tgt = xy - 0.30 * fwd
                how = "straight back 0.30 m (no cell with room to turn within reach)"
            else:
                tgt = best[2]
                how = "straight %s to the nearest cell with room to turn" % ("back" if best[1] < 0 else "forward")
            moved = self._move_straight(tgt, float(q["yaw"]))
        q2 = self.t.pose()
        self.note("unblock", at=_r(spot[:2]), how=how, turned_back=turned, target=_r(tgt), moved_m=round(moved, 3),
                  end=[round(q2["x"], 3), round(q2["y"], 3)], hold_waited_s=round(waited, 2),
                  room_to_turn=has_room(q2["x"], q2["y"]))
        return spot

    def _move_straight(self, target, heading, speed=None, max_s=None, mode="walk"):
        """Walk the body centre to `target` without turning (the heading held; backwards, forwards or sideways in the
        body frame as the target lies), in `mode` (walk, or an element's own mode at that element); holds after a
        `blocked` are waited out; ends at the target (5 cm), at max_s, or when 4 s gain less than 3 cm. The distance
        moved (true pose)."""
        speed = UNBLOCK["speed"] if speed is None else float(speed)
        self.d.set_mode(mode)
        p0 = self.t.pose()
        tgt = np.asarray(target, float)
        dist0 = math.hypot(tgt[0] - p0["x"], tgt[1] - p0["y"])
        max_s = max_s if max_s is not None else 2.0 * dist0 / max(speed, 0.05) + 6.0
        t, hist = 0.0, []
        while t < max_s:
            p = self.t.pose()
            d = tgt - np.array([p["x"], p["y"]])
            if float(np.linalg.norm(d)) < 0.05:
                break
            c, s_ = math.cos(p["yaw"]), math.sin(p["yaw"])
            bx, by = c * d[0] + s_ * d[1], -s_ * d[0] + c * d[1]
            n = max(math.hypot(bx, by), 1e-9)
            vx = float(np.clip(speed * bx / n, -speed, speed))
            vy = float(np.clip(speed * by / n, -0.10, 0.10))
            wz = float(np.clip(1.5 * wrap(heading - p["yaw"]), -0.3, 0.3))
            rep = self.r.walk(vx, vy, wz)
            if isinstance(rep, dict) and rep.get("held"):
                t0 = float(self.t.pose()["t"])
                self.wait_hold(rep)
                t += float(self.t.pose()["t"]) - t0
                hist = []
                continue
            self.d.step()
            t += self.d.tick * 0.01
            hist.append((t, p["x"], p["y"]))
            while len(hist) > 1 and t - hist[0][0] > 4.0:
                hist.pop(0)
            if t > 4.0 and t - hist[0][0] >= 3.9 and math.hypot(p["x"] - hist[0][1], p["y"] - hist[0][2]) < 0.03:
                break
        self.d.stop(0.4)
        p1 = self.t.pose()
        return math.hypot(p1["x"] - p0["x"], p1["y"] - p0["y"])

    def _element_on_path(self, path, p, reach=1.5):
        """(element, unit direction) of the first needs_* element the walk path meets within `reach` along it from the
        robot's place (the path's point nearest the robot onward), or None."""
        P = np.asarray(path, float).reshape(-1, 2)
        xy = np.array([p["x"], p["y"]], float)
        if len(P) == 0:
            return None
        j = int(np.argmin(np.linalg.norm(P - xy, axis=1)))
        pts = [xy] + [P[k] for k in range(j, len(P))]
        hard = [e for e in self.elements.values() if e["class"] in ("needs_high_step", "needs_crouch")]
        acc = 0.0
        for a, b in zip(pts[:-1], pts[1:]):
            seg = b - a
            L = float(np.linalg.norm(seg))
            if L < 1e-9:
                continue
            for s_ in np.linspace(0.0, L, max(2, int(L / 0.05) + 1)):
                if acc + s_ > reach:
                    return None
                q = a + seg * (s_ / L)
                for e in hard:
                    if near_polygon(np.asarray(e["footprint"], float), q, 0.05):
                        return e, seg / L
            acc += L
        return None

    def _element_ahead(self, p, reach=0.8):
        """The traversal element (not a narrow gap or puddle) nearest ahead of the robot within `reach`."""
        fwd = np.array([math.cos(p["yaw"]), math.sin(p["yaw"])])
        best = None
        for e in self.elements.values():
            if e["kind"] in ("narrow_gap", "puddle") or e["class"] == "barrier":
                continue
            fp = np.asarray(e["footprint"], float)
            for s in np.linspace(0.0, reach, 17):
                q = np.array([p["x"], p["y"]]) + s * fwd
                if near_polygon(fp, q, 0.05):
                    if best is None or s < best[0]:
                        best = (s, e)
                    break
        return None if best is None else best[1]

    # ------------------------------------------------------------------ head
    def aim(self, target, sensor="inspect", settle=0.7):
        """Point the given camera at a world point from the true head geometry (two iterations)."""
        off = self.t.camera_offset(sensor)
        for it in range(2):
            Tm = self._mast_frame()
            v = np.linalg.inv(Tm) @ np.r_[np.asarray(target, float), 1.0]
            v = v[:3] - np.array([0.0, 0.0, 0.012])          # the tilt axis is 12 mm above the pan joint
            rho = math.hypot(v[0], v[1])
            pan = math.atan2(v[1], v[0]) - math.asin(np.clip(off[1] / max(rho, 1e-6), -1, 1))
            tilt = -math.atan2(v[2] - off[2], max(rho, 1e-6))
            self.r.set_head(pan, tilt)
            self.d.wait(0.35 if it == 0 else settle)
        return pan, tilt

    def _mast_frame(self):
        # head_pan frame without the pan rotation = the mast frame at the pan joint height
        Tb = self.t.T_world_base()
        Tm = Tb.copy()
        Tm[:3, 3] = Tb[:3, 3] + Tb[:3, :3] @ np.array([0.165, 0.0, 0.031 + 0.012])
        return Tm

    def frame_center_error(self, fr, target):
        """Pixel offset of the target from the image centre in a frame (true base pose x frame's T_body_cam)."""
        T = self.t.T_world_base() @ np.asarray(fr["T_body_cam"], float)
        Tinv = np.linalg.inv(T)
        pc = Tinv[:3, :3] @ np.asarray(target, float) + Tinv[:3, 3]
        K = np.asarray(fr["K"], float)
        u = K[0, 0] * pc[0] / pc[2] + K[0, 2]
        v = K[1, 1] * pc[1] / pc[2] + K[1, 2]
        return u - K[0, 2], v - K[1, 2], pc[2]

    # ------------------------------------------------------------------ observation poses
    def view_poses(self, tag, n=6):
        """Candidate stand poses (body centre), each checked with the geometric evidence prediction for the camera
        MAST_X ahead of the body toward the instrument (with margin: 1.3 x the pixel size, 3 deg inside the
        off-axis limit, every sample point visible). The site's observation pose and readable cells are camera
        positions (site generator): each is tried as the place of the camera (body MAST_X behind it) and as the
        place of the body. The site's observation pose first, then the readable cells nearest to it."""
        inst = self.t.instrument(tag)
        c = np.asarray(inst["face_center"], float)
        typ = inst["type"]
        ob = inst["observation_pose"]
        rule = self.rules["types"][typ]
        cells = [np.asarray(ob["pos"], float)]
        rc = self.t.readable_cells(tag, "walk")
        if len(rc):
            dd = np.linalg.norm(rc - cells[0], axis=1)
            order = np.argsort(dd)
            cells += [rc[i] for i in order[::7][:300]]
        units = [u["label"] for u in inst.get("units", [])] if typ == "thermal" else [None]
        vis_need = 0.95 if float(rule["min_visible_share"]) >= 0.5 else 2.0 * float(rule["min_visible_share"])
        out = []
        # new sites say which position their observation pose and readable cells are: the body
        # centre ("body") or the camera (old sites); the stated reading is tried first, the other one second
        body_first = str(ob.get("semantics", "")) == "body"
        for cell in cells:
            u = (c[:2] - cell[:2]) / max(np.linalg.norm(c[:2] - cell[:2]), 1e-6)
            for q in ((cell, cell - MAST_X * u) if body_first else (cell - MAST_X * u, cell)):
                if not self.t.enterable(q[0], q[1], "walk", avoid_kinds=self.avoid_kinds, margin=self.margins[-1]):
                    continue
                uq = (c[:2] - q[:2]) / max(np.linalg.norm(c[:2] - q[:2]), 1e-6)
                cam = [q[0] + MAST_X * uq[0], q[1] + MAST_X * uq[1], HEAD_Z_STAND]
                good = True
                for un in units:
                    if typ == "thermal":
                        upos = next((x["pos"] for x in inst["units"] if x["label"] == un), c)
                        pr = self.t.predict_evidence(tag, cam, "thermal", aim=upos, unit=un)
                    else:
                        pr = self.t.predict_evidence(tag, cam, "inspect")
                    angle_margin = 3.0 if (typ != "thermal" or inst.get("has_face_normal")) else 0.0
                    good = good and (pr["ok"] and pr["app_px"] >= 1.3 * float(rule["min_px"])
                                     and pr["off_axis_deg"] <= float(rule["max_off_axis_deg"]) - angle_margin
                                     and pr["visible"] >= vis_need)
                    if not good:
                        break
                if good:
                    out.append(np.asarray(q, float))
                    break
            if len(out) >= n:
                break
        if not out:
            # no standing pose: an instrument seen only from a low space (its accepted region holds crouching cells
            # only, e.g. a B-low slot behind a low passage); its crouching cells, the camera at the crouching height,
            # and the frames are taken without standing up (read_instrument)
            out = self._crouch_view_poses(tag, inst, c, typ, rule, units, vis_need, n)
            if out:
                self.view_posture[tag] = "crouch_walk"
        return out, c, inst

    def _crouch_view_poses(self, tag, inst, c, typ, rule, units, vis_need, n):
        """view_poses for the crouching region: the observation pose when the site gives it crouching, then the
        crouching readable cells nearest to it, at least 0.25 m apart (a second pose differs from the first); each
        checked like a standing one with the camera at HEAD_Z_CROUCH."""
        ob = inst["observation_pose"]
        ob_pos = np.asarray(ob["pos"], float)
        cells = [ob_pos] if str(ob.get("mode", "")) == "crouch_walk" else []
        rc = self.t.readable_cells(tag, "crouch_walk")
        if len(rc):
            dd = np.linalg.norm(rc - ob_pos[:2], axis=1)
            cells += [rc[i] for i in np.argsort(dd)[::7][:300]]
        body_first = str(ob.get("semantics", "")) == "body"
        angle_margin = 3.0 if (typ != "thermal" or inst.get("has_face_normal")) else 0.0
        out = []
        for cell in cells:
            u = (c[:2] - cell[:2]) / max(np.linalg.norm(c[:2] - cell[:2]), 1e-6)
            for q in ((cell, cell - MAST_X * u) if body_first else (cell - MAST_X * u, cell)):
                if any(np.linalg.norm(np.asarray(q[:2], float) - o[:2]) < 0.25 for o in out):
                    continue
                if not self.t.enterable(q[0], q[1], "crouch_walk", avoid_kinds=self.avoid_kinds,
                                        margin=self.margins[-1]):
                    continue
                uq = (c[:2] - q[:2]) / max(np.linalg.norm(c[:2] - q[:2]), 1e-6)
                cam = [q[0] + MAST_X * uq[0], q[1] + MAST_X * uq[1], HEAD_Z_CROUCH]
                good = True
                for un in units:
                    if typ == "thermal":
                        upos = next((x["pos"] for x in inst["units"] if x["label"] == un), c)
                        pr = self.t.predict_evidence(tag, cam, "thermal", aim=upos, unit=un)
                    else:
                        pr = self.t.predict_evidence(tag, cam, "inspect")
                    good = good and (pr["ok"] and pr["app_px"] >= 1.3 * float(rule["min_px"])
                                     and pr["off_axis_deg"] <= float(rule["max_off_axis_deg"]) - angle_margin
                                     and pr["visible"] >= vis_need)
                    if not good:
                        break
                if good:
                    out.append(np.asarray(q, float))
                    break
            if len(out) >= n:
                break
        return out

    # ------------------------------------------------------------------ instruments
    def read_instrument(self, tag):
        """Read one instrument: the first view pose, a few frames at it; when it gives no reading, poses from other
        parts of the accepted region (other distance bands and bearings, _diverse_poses) while the item's frames
        last (REF_FRAMES_PER_ITEM; a temperature point: up to three of the first poses, as before); a visual
        reading is the readers' consensus over every usable reading taken so far (bricks.consensus.combine:
        the largest agreeing group, at least two readings), so a second view settles readings that disagree; no
        consensus after the poses gives a value-free report citing the frames whose geometry passed."""
        item = self.items[tag]
        poses, c, inst = self.view_poses(tag, n=8)
        typ = item["type"]
        crouch = self.view_posture.get(tag) == "crouch_walk"
        if crouch:
            self.note("instrument_start", tag=tag, type=typ, n_poses=len(poses), posture="crouch_walk")
        else:
            self.note("instrument_start", tag=tag, type=typ, n_poses=len(poses))
        if not poses:
            self.note("no_view_pose", tag=tag)
            return
        best_frames = []
        readings = []                    # (value, confidence, frame id) of the frames whose geometry passed
        tried = 0
        self._item_frames = 0            # camera frames taken for this item (_capture_aimed)
        queue = list(enumerate(poses))
        diverse = False
        tried_q = []
        while queue:
            k, q = queue.pop(0)
            # the first poses: as many as three; after the first pose that gave no reading, poses from other parts
            # of the region (_diverse_poses) while the item's frames last
            if (self._item_frames >= REF_FRAMES_PER_ITEM) if diverse else (tried >= 3):
                break
            p = self.t.pose()
            if self._plan((p["x"], p["y"]), (q[0], q[1]))[0] is None:
                self.note("view_pose_unreachable", tag=tag, pose=[round(float(q[0]), 3), round(float(q[1]), 3)])
                continue
            tried += 1
            tried_q.append(np.asarray(q, float))
            res = self._view_from(tag, item, inst, c, q, crouch, readings)
            if res is not None:
                value, fids, conf = res
                best_frames = fids or best_frames
                if value is not None and tag in self.plate_ok:
                    out = self.r.report(tag, value, fids[:3])
                    self.note("report", tag=tag, value=value, frames=fids[:3], conf=conf, pose_index=k, result=out,
                              plate_frame=self.plate_ok[tag])
                    return
                if value is not None:
                    # a value, but the plate was not read as the tag from this pose: no report from here
                    self.note("report_withheld", tag=tag, value=value, pose_index=k, why="plate not read as the tag")
            if not diverse and typ != "thermal":
                # no reading from the first pose: the next readable cells are the same spot (within a few
                # centimetres) and give the same frames; other distance bands and bearings first, the remaining
                # first poses after them
                diverse = True
                dq = self._diverse_poses(tag, inst, c, tried_q, crouch)
                self.note("diverse_poses", tag=tag, n=len(dq),
                          poses=[[round(float(x[0]), 3), round(float(x[1]), 3)] for x in dq])
                rest = [(kk, x) for kk, x in queue
                        if all(np.linalg.norm(np.asarray(x[:2], float) - t[:2]) >= 0.25 for t in tried_q + dq)]
                queue = [(len(poses) + i, x) for i, x in enumerate(dq)] + rest
        # no consensus: report without a value, citing qualifying frames (every frame in `readings` qualified); only
        # for an instrument whose plate was read as its tag (no match after its frames, no report)
        if tag not in self.plate_ok:
            self.note("no_report", tag=tag, why="the plate was never read as the tag",
                      reads=[(r["frame"], r["text"]) for r in self.plate_reads if r["tag"] == tag][-6:])
            return
        if readings:
            best_frames = [f for _v, _c, f in readings][-3:][::-1]
        out = self.r.report(tag, None, best_frames[:3])
        self.note("report_no_value", tag=tag, frames=best_frames[:3], result=out, plate_frame=self.plate_ok[tag])

    def _view_from(self, tag, item, inst, c, q, crouch, readings):
        """Go to the view pose q (body centre), stand (or stay crouching), take and read frames: (value or None, frame
        ids, confidence), or None when the pose was not reached or gave nothing to read."""
        typ = item["type"]
        yaw = math.atan2(c[1] - q[1], c[0] - q[0])
        if crouch:
            # a low space: crouching all the way to the pose, the frames taken crouching
            if not self._go_crouching(q, yaw, tag):
                self._leave_crouch()
                return None
        else:
            if not self.go(q[0], q[1], yaw=yaw):
                return None
            self.r.set_mode("stand")
            self.d.mode = "stand"
        self.look_ahead("view pose %s" % tag)
        dist = float(np.linalg.norm(c[:2] - np.asarray(q[:2])))
        if not crouch and math.degrees(math.atan2(c[2] - HEAD_Z_STAND, max(dist, 0.3))) > 8.0:
            self.r.set_body(height=0.28)          # raise the head 4 cm for instruments above the camera
        self.d.wait(1.0)
        room = self.rooms.get(inst["room"], {})
        lamp = room.get("light") in ("dark", "dim") and dist < 3.0 and typ != "thermal"
        self.r.set_lamp(lamp)
        self._pose_frames = {}
        if typ == "thermal":
            res = self._read_thermal(tag, item, inst)
        else:
            res = self._read_visual(tag, item, inst, c, readings)
        if res is not None and tag not in self.plate_ok:
            # version 2: the plate, from this pose, before any report of the item
            if typ == "thermal" and room.get("light") in ("dark", "dim") and dist < 3.0:
                self.r.set_lamp(True)
            self._confirm_plate(tag, item, inst, res[1])
        self._pose_frames = {}
        self.r.set_lamp(False)
        self.r.set_head(0.0, 0.0)
        if crouch:
            self._leave_crouch()
        return res

    # ------------------------------------------------------------------ the tag plate (version 2)
    def _plate_corners(self, inst):
        """The plate's four corners in the world (its box: the thinnest axis is the normal), or None."""
        pl = inst.get("plate")
        if not pl:
            return None
        c = np.asarray(pl["center"], float)
        R = np.asarray(pl["R"], float)
        h = np.asarray(pl["half"], float)
        order = np.argsort(h)                       # normal, height, width
        a, b = R[:, order[2]] * h[order[2]], R[:, order[1]] * h[order[1]]
        return np.array([c - a + b, c + a + b, c + a - b, c - a - b])

    def _plate_quad(self, fr, inst):
        """The plate's corners projected into a frame (true pose: aim), or None when it is not whole in the frame."""
        P = self._plate_corners(inst)
        if P is None:
            return None
        T = self.t.T_world_base() @ np.asarray(fr["T_body_cam"], float)
        Ti = np.linalg.inv(T)
        K = np.asarray(fr["K"], float)
        H, W = fr["image"].shape[:2]
        uv = []
        for p in P:
            pc = Ti[:3, :3] @ p + Ti[:3, 3]
            if pc[2] <= 0.1:
                return None
            uv.append((K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]))
        uv = np.asarray(uv)
        e = PLATE_EDGE_PX
        if uv[:, 0].min() < e or uv[:, 1].min() < e or uv[:, 0].max() > W - 1 - e or uv[:, 1].max() > H - 1 - e:
            return None
        return uv

    def _to_cam(self, fr, n_w):
        """A world direction in the frame's optical frame (the camera pose from the true pose: aim), turned to face
        the camera (a plate the camera sees faces it: optical z away from the plate)."""
        T = self.t.T_world_base() @ np.asarray(fr["T_body_cam"], float)
        n_c = T[:3, :3].T @ np.asarray(n_w, float)
        if n_c[2] > 0.0:
            n_c = -n_c
        return n_c

    def _facing_recorded_w(self, item):
        """The round sheet's recorded facing of the item (public, to 5 degrees) as a horizontal world direction,
        or None when the documents record none (an item in an undrawn annex)."""
        f = (item.get("location") or {}).get("facing_deg")
        if f is None:
            return None
        a = math.radians(float(f))
        return np.array([math.cos(a), math.sin(a), 0.0])

    def _plate_normal_depth_w(self, tag, inst):
        """The plate's surface normal (world) from the robot's own navigation depth frame: the navigation camera
        aimed at the plate (its place from the true plate box: aim), the depth pixels inside the plate's projected
        outline (shrunk by a fifth) back-projected and fitted with a plane. One frame per item, kept; None when the
        frame is refused or holds too few depth pixels there."""
        cache = self.__dict__.setdefault("_depth_normals", {})
        if tag in cache:
            return cache[tag]
        cache[tag] = None
        P = self._plate_corners(inst)
        if P is None:
            return None
        c = np.asarray(inst["plate"]["center"], float)
        self.aim(c, "nav")
        fr = refusable(self.r.get_camera, "nav", depth=True)
        if fr is None or fr.get("depth") is None:
            self.note("plate_depth", tag=tag, ok=False, why="no frame")
            return None
        T = self.t.T_world_base() @ np.asarray(fr["T_body_cam"], float)
        Ti = np.linalg.inv(T)
        K = np.asarray(fr["K"], float)
        D = np.asarray(fr["depth"], float)
        H, W = D.shape[:2]
        uv = []
        for p in c + 0.8 * (P - c):
            pc = Ti[:3, :3] @ p + Ti[:3, 3]
            if pc[2] <= 0.1:
                self.note("plate_depth", tag=tag, frame=fr["frame_id"], ok=False, why="behind")
                return None
            uv.append((K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]))
        uv = np.asarray(uv)
        x0, y0 = np.floor(uv.min(0)).astype(int)
        x1, y1 = np.ceil(uv.max(0)).astype(int)
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)
        pts = []
        for v in range(y0, y1 + 1):
            for u in range(x0, x1 + 1):
                if not _in_quad(uv, (u, v)):
                    continue
                z = D[v, u]
                if not np.isfinite(z) or z <= 0.05:
                    continue
                pts.append(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))
        if len(pts) < PLATE_DEPTH_MIN_PX:
            self.note("plate_depth", tag=tag, frame=fr["frame_id"], ok=False, why="%d depth pixels" % len(pts))
            return None
        X = np.asarray(pts, float)
        m = X.mean(0)
        _u, s, vt = np.linalg.svd(X - m, full_matrices=False)
        n = vt[2]
        if float(np.dot(n, -m)) < 0.0:
            n = -n
        n_w = T[:3, :3] @ n
        cache[tag] = n_w
        self.note("plate_depth", tag=tag, frame=fr["frame_id"], ok=True, n_px=len(pts),
                  rms_mm=round(1000.0 * float(s[2]) / math.sqrt(len(pts)), 2),
                  normal_w=[round(float(x), 3) for x in n_w])
        return n_w

    def _read_plate_text(self, tag, item, inst, fr, hint):
        """bricks.readers.tagtext.read with the mirror choice settled publicly: the round sheet's
        recorded facing; when the round sheet records none, or the frame shows it is not this plate's facing (both
        planar poses more than FACING_AGREE_DEG from it: a moved or relocated instrument, whose recorded place is
        another wall), the plate's own surface normal from the navigation depth frame; failing that, the reader's
        own choice (the smaller reprojection error). (value, confidence, debug, facing source)."""
        def read(h):
            try:
                return r_text.read(fr["image"], h)
            except Exception as e:  # noqa: BLE001  (a reader failure is no read)
                return None, 0.0, {"why": "%s: %s" % (type(e).__name__, e)}

        rec_w = self._facing_recorded_w(item)
        if rec_w is not None:
            fc = self._to_cam(fr, rec_w)
            v, conf, dbg = read(dict(hint, facing_cam=fc.tolist()))
            if v is None or v.get("R_cam_face") is None:
                return v, conf, dbg, "round_sheet"
            agree = float(np.asarray(v["R_cam_face"], float)[:, 0] @ fc)
            if agree >= math.cos(math.radians(FACING_AGREE_DEG)):
                return v, conf, dbg, "round_sheet (%.1f deg)" % math.degrees(math.acos(np.clip(agree, -1, 1)))
            src = "depth (round sheet's facing %.0f deg off the plate)" % math.degrees(math.acos(np.clip(agree, -1, 1)))
        else:
            src = "depth (no recorded facing)"
        n_w = self._plate_normal_depth_w(tag, inst)
        if n_w is not None:
            v, conf, dbg = read(dict(hint, facing_cam=self._to_cam(fr, n_w).tolist()))
            return v, conf, dbg, src
        v, conf, dbg = read(dict(hint))
        return v, conf, dbg, "reprojection"

    def _true_normal_err_deg(self, fr, inst, v):
        """Diagnostic for the log only (never used for a decision): the angle between the chosen planar pose's normal
        and the true plate box's normal (a mirror flip shows as about twice the plate's off-axis angle)."""
        pl = inst.get("plate")
        if not pl or not v or v.get("R_cam_face") is None:
            return None
        R = np.asarray(pl["R"], float)
        n_t = self._to_cam(fr, R[:, int(np.argmin(np.asarray(pl["half"], float)))])
        n_t = n_t / max(float(np.linalg.norm(n_t)), 1e-12)
        return round(math.degrees(math.acos(float(np.clip(np.asarray(v["R_cam_face"], float)[:, 0] @ n_t, -1, 1)))),
                     2)

    def _plate_read(self, tag, item, inst, fr, source):
        """Read the plate in one of the robot's own inspection frames (bricks.readers.tagtext, from the pixels); the
        crop around where the plate projects is the only privileged input (aim); the mirror pose is settled publicly
        (_read_plate_text). True when the text is the tag."""
        quad = self._plate_quad(fr, inst)
        if quad is None:
            return False
        H, W = fr["image"].shape[:2]
        lo, hi = quad.min(0), quad.max(0)
        m = np.maximum(0.6 * (hi - lo), 24.0)
        roi = [int(max(0, lo[0] - m[0])), int(max(0, lo[1] - m[1])), int(min(W, hi[0] + m[0])),
               int(min(H, hi[1] + m[1]))]
        hint = {"K": np.asarray(fr["K"], float), "roi": roi}
        v, conf, dbg, fsrc = self._read_plate_text(tag, item, inst, fr, hint)
        text = v.get("text") if v else None
        ok = bool(text == tag)
        rec = {"t": round(float(self.t.pose()["t"]), 2), "tag": tag, "frame": fr["frame_id"], "source": source,
               "text": text, "ok": ok,
               "conf": round(float(conf), 3), "why": None if v else (dbg or {}).get("why"), "facing": fsrc}
        if v:
            for k in ("cap_px", "pitch_px", "off_axis_deg", "dist_m"):
                if v.get(k) is not None:
                    rec[k] = round(float(v[k]), 2)
            rec["true_normal_err_deg"] = self._true_normal_err_deg(fr, inst, v)
        self.plate_reads.append(rec)
        self.note("plate_read", **{k: v for k, v in rec.items() if k != "t"})
        if ok:
            self.plate_ok[tag] = fr["frame_id"]
        return ok

    def _confirm_plate(self, tag, item, inst, cited):
        """The plate read as the tag from this pose: a cited frame, another frame of this pose, then up to
        PLATE_FRAMES frames aimed at the plate (they do not count against the item's reading frames)."""
        if inst.get("plate") is None:
            self.note("plate_unknown", tag=tag)
            return False
        tried = set()
        for fid in list(cited or []) + [f for f in self._pose_frames if f not in (cited or [])]:
            fr = self._pose_frames.get(fid)
            if fr is None or fid in tried:
                continue
            tried.add(fid)
            if self._plate_read(tag, item, inst, fr, "cited" if fid in (cited or []) else "pose"):
                return True
        n0 = self._item_frames
        c = np.asarray(inst["plate"]["center"], float)
        try:
            for _ in range(PLATE_FRAMES):
                fr = self._capture_aimed(c, "inspect")
                if fr is None:
                    break
                if self._plate_read(tag, item, inst, fr, "aimed"):
                    return True
        finally:
            self._item_frames = n0
        return False

    def _diverse_poses(self, tag, inst, c, tried, crouch):
        """View poses from other parts of the instrument's accepted region, after a pose gave no reading. The region's
        cells (the posture's) are split into three distance bands and three bearings around the face normal (thirds
        of the region's own cells); the parts the poses tried so far stand in are left out; the others are taken in
        the order: other band and other bearing, then one of the two; nearer bands and the middle bearing first.
        In each part the first cell, from the part's centre outward, at least 0.5 m from every pose so far, whose
        geometric prediction qualifies with view_poses' margins. One pose per part."""
        mode = "crouch_walk" if crouch else "walk"
        rc = self.t.readable_cells(tag, mode)
        if len(rc) < 9:
            return []
        typ = inst["type"]
        rule = self.rules["types"][typ]
        vis_need = 0.95 if float(rule["min_visible_share"]) >= 0.5 else 2.0 * float(rule["min_visible_share"])
        c = np.asarray(c, float)
        R = np.asarray(inst["face_R"], float)
        nrm = np.array([R[0, 0], R[1, 0]])
        if np.linalg.norm(nrm) < 1e-6:
            nrm = rc.mean(0) - c[:2]
        nrm = nrm / max(np.linalg.norm(nrm), 1e-9)

        def polar(P):
            v = np.atleast_2d(P)[:, :2] - c[None, :2]
            return np.linalg.norm(v, axis=1), np.arctan2(nrm[0] * v[:, 1] - nrm[1] * v[:, 0], v @ nrm)
        dist, bear = polar(rc)
        de = np.quantile(dist, [1.0 / 3.0, 2.0 / 3.0])
        be = np.quantile(bear, [1.0 / 3.0, 2.0 / 3.0])
        band, sect = np.digitize(dist, de), np.digitize(bear, be)
        tq = np.asarray([t[:2] for t in tried], float).reshape(-1, 2)
        td, tb = polar(tq) if len(tq) else (np.zeros(0), np.zeros(0))
        t_band, t_sect = set(np.digitize(td, de).tolist()), set(np.digitize(tb, be).tolist())
        parts = [(b, s) for b in range(3) for s in range(3) if not (b in t_band and s in t_sect)]
        parts.sort(key=lambda bs: ((bs[0] in t_band) + (bs[1] in t_sect), bs[0], abs(bs[1] - 1)))
        body_first = str(inst["observation_pose"].get("semantics", "")) == "body"
        head_z = HEAD_Z_CROUCH if crouch else HEAD_Z_STAND
        angle_margin = 3.0 if (typ != "thermal" or inst.get("has_face_normal")) else 0.0
        out = []
        for b, s in parts:
            idx = np.nonzero((band == b) & (sect == s))[0]
            if not len(idx):
                continue
            P = rc[idx]
            order = np.argsort(np.linalg.norm(P - P.mean(0)[None, :], axis=1))
            found = None
            for i in order[::3][:60]:
                cell = P[i]
                u = (c[:2] - cell) / max(np.linalg.norm(c[:2] - cell), 1e-6)
                for qq in ((cell, cell - MAST_X * u) if body_first else (cell - MAST_X * u, cell)):
                    if any(np.linalg.norm(qq - np.asarray(o[:2], float)) < 0.5 for o in list(tq) + out):
                        continue
                    if not self.t.enterable(qq[0], qq[1], mode, avoid_kinds=self.avoid_kinds, margin=self.margins[-1]):
                        continue
                    uq = (c[:2] - qq) / max(np.linalg.norm(c[:2] - qq), 1e-6)
                    cam = [qq[0] + MAST_X * uq[0], qq[1] + MAST_X * uq[1], head_z]
                    pr = self.t.predict_evidence(tag, cam, "inspect")
                    if (pr["ok"] and pr["app_px"] >= 1.3 * float(rule["min_px"])
                            and pr["off_axis_deg"] <= float(rule["max_off_axis_deg"]) - angle_margin
                            and pr["visible"] >= vis_need):
                        found = np.asarray(qq, float)
                        break
                if found is not None:
                    break
            if found is not None:
                out.append(found)
        return out

    def _capture_aimed(self, target, sensor="inspect", tries=2):
        fr = None
        for _ in range(tries):
            self.aim(target, sensor)
            fr = refusable(self.r.get_camera, sensor, depth=False) if sensor != "thermal" else \
                refusable(self.r.get_thermal)
            if fr is None:
                return None
            self._item_frames = getattr(self, "_item_frames", 0) + 1
            if sensor == "inspect":
                self._pose_frames[fr["frame_id"]] = fr
            du, dv, z = self.frame_center_error(fr, target)
            K = fr["K"]
            if abs(du) < 0.15 * K[0, 2] and abs(dv) < 0.15 * K[1, 2]:
                break
        return fr

    def _reaim(self, tag, inst, cam):
        """Another aim for a frame that failed only on its in-frame shares, from the camera's present place `cam`:
        the aims of REAIM on the face's up axis (then beside it), each checked with the geometric prediction; the
        middle of the longest run of aims that pass (the most room on both sides); None when none passes."""
        c = np.asarray(inst["face_center"], float)
        R = np.asarray(inst["face_R"], float)
        zs = [REAIM["z0"] + REAIM["dz"] * i for i in range(REAIM["n"])]
        for dy in REAIM["y"]:
            good = [bool(self.t.predict_evidence(tag, cam, "inspect", aim=c + dy * R[:, 1] + dz * R[:, 2])["ok"])
                    for dz in zs]
            best, start = None, None
            for i, g in enumerate(good + [False]):
                if g and start is None:
                    start = i
                elif not g and start is not None:
                    if best is None or i - start > best[1] - best[0]:
                        best = (start, i)
                    start = None
            if best is not None:
                dz = REAIM["z0"] + REAIM["dz"] * 0.5 * (best[0] + best[1] - 1)
                return c + dy * R[:, 1] + dz * R[:, 2]
        return None

    # ------------------------------------------------------------------ low spaces (crouching view poses)
    def _go_crouching(self, q, yaw, tag):
        """To a crouching view pose: the planner's way up to the walking end of its last crouching stretch (go), then
        that stretch in crouch_walk along the planner's own path to the pose, and the turn to the instrument
        crouching. The way back out is kept for _leave_crouch. False when the pose was not reached."""
        p = self.t.pose()
        plan, _wet = self._plan((p["x"], p["y"]), (q[0], q[1]))
        if plan is None:
            return False
        segs = plan["segments"]
        if segs[-1]["mode"] != "crouch_walk":
            return self.go(q[0], q[1], yaw=yaw)
        wps = [[float(v[0]), float(v[1])] for v in segs[-1]["waypoints"]]
        entry = wps[0]
        if len(segs) > 1 and not self.go(entry[0], entry[1], tol=0.06):
            return False
        self.d.stop(0.3)
        d0 = _direction(wps, 0.5)
        self.d.turn_to(math.atan2(d0[1], d0[0]), tol=0.04)
        self.look_ahead("approach crouching %s" % tag)
        self.d.set_mode("crouch_walk")
        p0 = self.t.pose()
        self._crouch_exit = wps[:-1][::-1]                 # back along the way in, its walking end last
        ok = self.d.follow([[p0["x"], p0["y"]]] + wps[1:], mode="crouch_walk", tol=0.08, lookahead=0.25,
                           speed=CROSS["speed"]["crouch_walk"], final_slow=CROSS["final_slow"], max_s=180.0)
        self.d.stop(0.4)
        p1 = self.t.pose()
        miss = math.hypot(p1["x"] - q[0], p1["y"] - q[1])
        self.note("crouch_in", tag=tag, ok=bool(ok), status=self.d.last_status, miss_m=round(miss, 3),
                  n_points=len(wps))
        if miss > GOAL_MISS_M:
            return False
        self.d.turn_to(yaw, tol=0.04)
        return True

    def _leave_crouch(self):
        """Out of a low space the way the robot came in, crouching, to the walking end of that way; then walk mode."""
        pts = self._crouch_exit
        self._crouch_exit = None
        if not pts:
            return
        p = self.t.pose()
        path = [[p["x"], p["y"]]] + [list(v) for v in pts]
        d0 = _direction(path, 0.5)
        self.d.turn_to(math.atan2(d0[1], d0[0]), tol=0.06)
        ok = self.d.follow(path, mode="crouch_walk", tol=0.08, lookahead=0.25, speed=CROSS["speed"]["crouch_walk"],
                           final_slow=CROSS["final_slow"], max_s=180.0)
        self.d.stop(0.4)
        p1 = self.t.pose()
        self.note("crouch_out", ok=bool(ok), status=self.d.last_status, end=[round(p1["x"], 3), round(p1["y"], 3)])
        self.d.set_mode("walk")

    def _read_visual(self, tag, item, inst, c, readings):
        """Up to three frames at this pose; after each, the consensus over all usable readings of the instrument
        (`readings`, kept across poses). (value or None, frame ids to cite, confidence)."""
        typ = item["type"]
        hint = reader_hint(item, pipe_axis=inst.get("pipe_axis_local"))
        span = None
        if typ == "gauge":
            lo, hi = item["scale"]["range"]
            span = float(hi) - float(lo)
        target = np.asarray(c, float)
        if typ == "valve_lever":
            # stem and tag plate in one frame: aim slightly below the stem toward the tag
            R = np.asarray(inst["face_R"], float)
            target = target - 0.04 * R[:, 2]
        fids = []
        reaimed = False
        for k in range(3):
            fr = self._capture_aimed(target)
            if fr is None:
                break
            # the acceptance geometry of this very frame (its camera pose: the stand pose is only within 8 cm and
            # the head swings with the body): only qualifying frames are read and cited; a first frame that does
            # not qualify leaves this view pose at once, unless it failed only on its in-frame shares: then the
            # head is aimed once more (_reaim) and the remaining frames are taken with that aim
            Tw = self.t.T_world_base() @ np.asarray(fr["T_body_cam"], float)
            fe = self.t.frame_evidence(tag, Tw, "inspect")
            if not fe["ok"]:
                self.note("view_rejected", tag=tag, frame=fr["frame_id"], failed=fe["failed"])
                if k == 0 and not reaimed and fe["failed"] and all(f.startswith(IN_FRAME_FAILS) for f in fe["failed"]):
                    t2 = self._reaim(tag, inst, Tw[:3, 3])
                    if t2 is not None:
                        reaimed = True
                        Rf = np.asarray(inst["face_R"], float)
                        dv = t2 - np.asarray(c, float)
                        self.note("reaim", tag=tag, frame=fr["frame_id"],
                                  offset_m=[round(float(dv @ Rf[:, 1]), 3), round(float(dv @ Rf[:, 2]), 3)])
                        target = t2
                        continue
                if k == 0:
                    break
                self.d.wait(0.5)
                continue
            fids.append(fr["frame_id"])
            rf = self._roi_frame(fr, inst)
            v, conf, dbg = self._reader(typ, rf, hint, tag, full=fr, inst=inst)
            self._overlay(rf, typ, v, conf, dbg)
            self.note("reading", tag=tag, frame=fr["frame_id"], value=v, conf=round(float(conf), 3))
            readings.append((v, conf, fr["frame_id"]))
            value, cconf, cdbg = r_consensus.combine(typ, [(rv, rc) for rv, rc, _f in readings], span=span)
            if value is not None:
                agreeing = [f for rv, rc, f in readings if rv is not None and float(rc) >= 0.5 and
                            r_consensus.agree(typ, rv, value, span)]
                if typ == "lamps":
                    value = [bool(x) for x in value]
                else:
                    value = round(float(value), 4)
                self.note("consensus", tag=tag, value=value, n=cdbg.get("n"), group=cdbg.get("group"))
                return (value, (agreeing[-3:] or fids[-3:])[::-1], float(cconf))
            self.d.wait(0.5)
        return (None, fids, 0.0)

    def _roi_frame(self, fr, inst):
        """The frame cropped to the instrument's known extent (projected from its pose, 60 % margin), with the
        intrinsics shifted to the crop; levers: pixels of another colour than the catalogue's red desaturated (the
        reader's lever mask is saturation-only and otherwise takes a coloured service pipe for the lever)."""
        import cv2
        typ = inst["type"]
        ext = ROI_HALF.get(typ, (0.15, 0.15))
        if typ == "gauge":
            D = float((inst.get("params") or {}).get("case_mm", 100)) / 1000.0
            ext = (0.6 * D, 0.6 * D)
        c = np.asarray(inst["face_center"], float)
        R = np.asarray(inst["face_R"], float)
        oy, oz = ROI_OFFSET.get(typ, (0.0, 0.0))
        pts = [c + R[:, 1] * (oy + sy * ext[0]) + R[:, 2] * (oz + sz * ext[1]) for sy in (-1, 1) for sz in (-1, 1)]
        T = self.t.T_world_base() @ np.asarray(fr["T_body_cam"], float)
        Tinv = np.linalg.inv(T)
        K = np.asarray(fr["K"], float)
        img = fr["image"]
        H, W = img.shape[:2]
        uv = []
        for p in pts:
            pc = Tinv[:3, :3] @ p + Tinv[:3, 3]
            if pc[2] <= 0.05:
                return fr
            uv.append((K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]))
        uv = np.asarray(uv)
        lo, hi = uv.min(0), uv.max(0)
        size = np.maximum(hi - lo, 60.0)
        lo, hi = lo - 0.6 * size, hi + 0.6 * size
        x0, y0 = int(max(0, np.floor(lo[0]))), int(max(0, np.floor(lo[1])))
        x1, y1 = int(min(W, np.ceil(hi[0]))), int(min(H, np.ceil(hi[1])))
        if x1 - x0 < 40 or y1 - y0 < 40:
            return fr
        out = dict(fr)
        crop = np.ascontiguousarray(img[y0:y1, x0:x1]).copy()
        if typ == "valve_lever":
            hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV).astype(np.float32)
            hue, sat = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0
            red = np.minimum(hue, 360.0 - hue) < 25.0
            g = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
            m = (sat > 0.25) & ~red
            crop[m] = g[m][:, None]
        Kc = K.copy()
        Kc[0, 2] -= x0
        Kc[1, 2] -= y0
        out["image"] = crop
        out["K"] = Kc
        out["roi"] = [x0, y0, x1, y1]
        return out

    def _reader(self, typ, fr, hint, tag, full=None, inst=None):
        img = fr["image"]
        K = np.asarray(fr["K"], float)
        try:
            if typ == "gauge":
                return r_gauge.read(img, hint)
            if typ == "level":
                v, c, d = r_level.read(img, hint)
                return (None if v is None else round(100.0 * v, 2)), c, d
            if typ == "valve_handwheel":
                v, c, d = r_valve.read_gate(img, hint)
                return (None if v is None else round(100.0 * v, 2)), c, d
            if typ == "lamps":
                return r_lamps.read(img, hint)
            if typ == "valve_lever":
                # version 2: the lever's plane from the text plate's border (planar pose with the
                # catalogue's plate size), the mirror solution settled publicly (_read_plate_text: the round
                # sheet's recorded facing, else the plate's surface in the navigation depth frame)
                # the plate is read on the whole frame (the lever crop has its non-red pixels desaturated, and the
                # plate is yellow); its pose is the camera's, so it holds for the crop's shifted K as well
                src = full if full is not None else fr
                th = {"K": np.asarray(src["K"], float), "expected": tag}
                if inst is not None:
                    tv, tc, _d, _fsrc = self._read_plate_text(tag, self.items[tag], inst, src, th)
                else:
                    tv, tc, _ = r_text.read(src["image"], th)
                h = dict(hint)
                if tv is not None:
                    h["K"] = K
                    h.update(r_text.lever_hint(tv))
                v, c, d = r_valve.read_lever(img, h)
                d = dict(d)
                d["plate_found"] = tv is not None
                d["plate_text"] = None if tv is None else tv.get("text")
                return (None if v is None else round(float(v), 2)), c, d
        except Exception as e:  # noqa: BLE001  (a reader failure is a failed reading)
            return None, 0.0, {"error": "%s: %s" % (type(e).__name__, e)}
        return None, 0.0, {}

    def _read_thermal(self, tag, item, inst):
        """One thermal frame per unit; the unit's temperature is the emissivity-corrected peak of the hot region
        around its projection; the hot unit is the hottest."""
        hint = reader_hint(item)
        eps = float(hint["emissivity"])
        res, fids = [], []
        for u in inst.get("units", []):
            fr = self._capture_aimed(np.asarray(u["pos"], float), "thermal")
            if fr is None:
                continue
            # the acceptance geometry from where the camera really is, for this unit (as for the visual types)
            Tw = self.t.T_world_base() @ np.asarray(fr["T_body_cam"], float)
            pr = self.t.frame_evidence(tag, Tw, "thermal", unit=u["label"])
            if not pr["ok"]:
                self.note("view_rejected", tag=tag, unit=u["label"], frame=fr["frame_id"], failed=pr["failed"])
                if len(inst.get("units", [])) == 1:
                    return None
            fids.append(fr["frame_id"])
            img = fr["image_c"]
            du, dv, z = self.frame_center_error(fr, u["pos"])
            K = fr["K"]
            cu, cv = int(round(K[0, 2] + du)), int(round(K[1, 2] + dv))
            h, w = img.shape
            # the crop around the unit's projection stays inside its own face (0.4 of the face's smaller side in
            # pixels, 2-9 px): an unlisted temperature point next to it (unlisted points render warm too), or any
            # hotter thing beside the face is never taken for it
            prm = inst.get("params") or {}
            side = min(float(prm.get("face_w", 0.0) or 0.0), float(prm.get("face_h", 0.0) or 0.0))
            r = 9 if side <= 0.0 or z <= 0.05 else int(max(2, min(9, round(0.4 * float(K[0, 0]) * side / z))))
            crop = img[max(0, cv - r):min(h, cv + r + 1), max(0, cu - r):min(w, cu + r + 1)]
            amb = float(np.median(img))
            reg = hottest_region(img) if crop.size == 0 else hottest_region(crop)
            if reg is None:
                t = emissivity_correct(float(np.max(crop)) if crop.size else amb, eps, amb)
                warm = False
            else:
                t = emissivity_correct(reg["t_peak"], eps, amb)
                warm = True
            res.append({"label": u["label"], "temp_C": round(float(t), 2), "frame": fr["frame_id"], "warm": warm})
            self.note("thermal_unit", tag=tag, unit=u["label"], temp=round(float(t), 2), frame=fr["frame_id"])
        if not res:
            return None
        hot = max(res, key=lambda x: x["temp_C"])
        order = [hot["frame"]] + [x["frame"] for x in res if x["frame"] != hot["frame"]]
        return ({"unit": hot["label"], "temp_C": hot["temp_C"]}, order[:3], 1.0)

    def _overlay(self, fr, typ, v, conf, dbg):
        """Reader overlay for the run folder (frames/<id>_overlay.png)."""
        if not self.out_dir:
            return
        try:
            import cv2
            img = cv2.cvtColor(np.ascontiguousarray(fr["image"]), cv2.COLOR_RGB2BGR).copy()
            if typ == "gauge" and dbg and "ellipse" in dbg:
                cx, cy, a, b, ang = dbg["ellipse"]
                cv2.ellipse(img, (int(cx), int(cy)), (int(a), int(b)), math.degrees(ang), 0, 360, (0, 255, 0), 2)
                # the reader's needle angle is in the rectified dial frame (x right, y up) of the ellipse axes
                t = math.radians(dbg.get("needle_deg", 0.0))
                ua = (math.cos(ang), math.sin(ang))
                va = (math.sin(ang), -math.cos(ang))
                dx = 0.85 * (a * math.cos(t) * ua[0] + b * math.sin(t) * va[0])
                dy = 0.85 * (a * math.cos(t) * ua[1] + b * math.sin(t) * va[1])
                cv2.line(img, (int(cx), int(cy)), (int(cx + dx), int(cy + dy)), (0, 0, 255), 3)
            if dbg and "quad" in dbg:
                q = np.asarray(dbg["quad"], np.int32).reshape(-1, 1, 2)
                cv2.polylines(img, [q], True, (0, 255, 0), 2)
            if typ == "lamps" and dbg and "lamps" in dbg:
                for e in dbg["lamps"]:
                    cv2.circle(img, (int(e[0]), int(e[1])), int(e[2]) + 3, (0, 255, 0), 2)
            txt = "%s: %s (conf %.2f)" % (typ, _short(v), conf)
            fs = max(0.5, img.shape[1] / 900.0)
            cv2.putText(img, txt, (10, int(30 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 5)
            cv2.putText(img, txt, (10, int(30 * fs)), cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 255, 255), 2)
            sc = 0.5 if img.shape[1] > 900 else 1.0
            small = cv2.resize(img, (int(img.shape[1] * sc), int(img.shape[0] * sc))) if sc != 1.0 else img
            cv2.imwrite(os.path.join(self.out_dir, "frames", fr["frame_id"] + "_overlay.jpg"), small,
                        [cv2.IMWRITE_JPEG_QUALITY, 88])
            if fr.get("roi"):
                with open(os.path.join(self.out_dir, "frames", fr["frame_id"] + "_overlay.json"), "w") as f:
                    json.dump({"roi_in_frame": fr["roi"], "type": typ, "value": v, "conf": conf}, f, default=_js)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ survey mark
    def do_mark(self, tag, dwell=MARK_DWELL_S):
        inst = self.t.instrument(tag)
        mx, my = inst["pose"]["pos"][:2]
        for _ in range(2):
            self.go(mx, my, tol=0.06)
            p = self.t.pose()
            if math.hypot(p["x"] - mx, p["y"] - my) <= 0.20:
                break
        p = self.t.pose()
        if math.hypot(p["x"] - mx, p["y"] - my) > 0.20:
            # not on the mark: a dwell elsewhere is no evidence (the report would score 0); no report, no 60 s
            self.note("mark_not_reached", tag=tag, off_mark_m=round(math.hypot(p["x"] - mx, p["y"] - my), 3))
            return
        self.r.set_mode("stand")
        self.d.mode = "stand"
        self.d.wait(1.0)
        self.r.get_state()                     # starts a fresh count
        self.d.wait(dwell)
        st = self.r.get_state()
        c, w = st["dosimeter"]["counts"], st["dosimeter"]["dwell_s"]
        rate = c / max(w, 1e-6) / float(self.desc["dosimeter"]["sensitivity_cps_per_gyh"])
        p2 = self.t.pose()
        out = self.r.report_mark(tag, rate)
        self.note("mark", tag=tag, counts=c, dwell=w, rate_gyh=round(rate, 5), off_mark_m=round(
            math.hypot(p2["x"] - mx, p2["y"] - my), 3), result=out)

    # ------------------------------------------------------------------ unlisted source
    def do_source(self, tag):
        try:
            self._do_source(tag)
        finally:
            self._field_strong = False          # only the survey's own walks take no retries in a strong field

    def _do_source(self, tag):
        src = np.asarray(self.t.source_position(tag), float)     # for planning the survey only
        cands = self._survey_candidates(src)
        pts = self._survey_points(src, cands)
        sens = float(self.desc["dosimeter"]["sensitivity_cps_per_gyh"])
        meas = []
        tried = []
        for q in pts:
            tried.append(np.asarray(q, float))
            if not self.go(q[0], q[1]):
                continue
            meas.append(self._survey_read(sens))
        for _ in range(SURVEY_EXTRA_MAX):
            spread = _line_spread(meas)
            if len(meas) >= SURVEY["n_points"] and spread >= SURVEY_MIN_SPREAD_M:
                break
            q = _survey_extra(cands, meas, tried)
            if q is None:
                self.note("survey_extra_none", read=len(meas), spread_m=round(spread, 3))
                break
            tried.append(np.asarray(q, float))
            self.note("survey_extra", x=round(float(q[0]), 3), y=round(float(q[1]), 3), read=len(meas),
                      spread_m=round(spread, 3))
            if self.go(q[0], q[1]):
                meas.append(self._survey_read(sens))
        # the stance, not the planned cell, decides whether a reading is shielded: a cell clear by a few centimetres
        # can be left, within the walk's tolerance, in the shadow of a wall's corner, and one such reading (a fraction
        # of the unshielded rate) pulls the unshielded inverse-square fit by metres; such readings are left out
        shields = self._shields(src)
        behind = [_crosses_boxes((m["x"], m["y"]), src[:2], shields) for m in meas]
        if any(behind) and len(meas) - sum(behind) >= SURVEY_MIN_CLEAR_FIT:
            self.note("survey_stance_shielded", left_out=[[round(m["x"], 3), round(m["y"], 3)]
                                                          for m, b in zip(meas, behind) if b])
            meas = [m for m, b in zip(meas, behind) if not b]
        fit = fit_source(meas, sens, self.background_gyh or 0.0, z_source=SURVEY["z_source"])
        self.note("source_fit", **fit)
        if fit.get("ok"):
            out = self.r.report_source(fit["x"], fit["y"])
            self.note("report_source", x=fit["x"], y=fit["y"], result=out)

    def _survey_read(self, sens):
        """Stand, count SURVEY["dwell_s"] at the present stance; the reading (pose, counts, dwell)."""
        self.r.set_mode("stand")
        self.d.mode = "stand"
        self.d.wait(0.6)
        self.r.get_state()
        self.d.wait(SURVEY["dwell_s"])
        st = self.r.get_state()
        p = self.t.pose()
        c, w = st["dosimeter"]["counts"], st["dosimeter"]["dwell_s"]
        rate = c / max(w, 1e-6) / sens
        # a strong field by its own counts (a minute here costs STRONG_FIELD_SHARE_PER_MIN of the rating or more):
        # the walk away from this stance takes no retries in place after a `blocked`
        rating = float((self.desc.get("dosimeter") or {}).get("rating_gy") or 0.0)
        self._field_strong = bool(rating > 0 and rate / 60.0 >= STRONG_FIELD_SHARE_PER_MIN * rating)
        self.note("survey_point", x=round(p["x"], 3), y=round(p["y"], 3), counts=int(c), dwell=float(w),
                  rate_gyh=round(rate, 4), dose_fraction=st["dosimeter"]["dose_fraction"],
                  strong_field=self._field_strong)
        return {"x": p["x"], "y": p["y"], "z": p["z"], "counts": int(c), "dwell": float(w)}

    def _survey_candidates(self, src):
        """Readable floor cells with a clear line of sight to the source, between r_min and r_max: [(xy, distance)]
        (the clear ones when there are enough of them, else all)."""
        g = self.t.trav_grid()
        x0, y0 = g["origin"]
        res = g["res"]
        en = g["enter"]["walk"]
        iy, ix = np.nonzero(en)
        P = np.c_[x0 + (ix + 0.5) * res, y0 + (iy + 0.5) * res]
        tight = np.asarray(g["tight"]["walk"], bool)[iy, ix]
        d = np.hypot(P[:, 0] - src[0], P[:, 1] - src[1])
        sel = (d >= SURVEY["r_min"]) & (d <= SURVEY["r_max"])
        P, d, tight = P[sel][::3], d[sel][::3], tight[sel][::3]
        ok = []
        shields = self._shields(src)
        for q, dd, ti in zip(P, d, tight):
            if ti:
                # never a stance where the body cannot turn (the map's tight cells: on a development site
                # a stance in a 0.60 m pocket between a wall and an open door leaf; the robot could not turn to
                # its next point and stood 95 s at 19 Gy/h)
                continue
            if not self.t.enterable(q[0], q[1], "walk", avoid_kinds=self.avoid_kinds, margin=self.margins[-1]):
                continue
            v = np.array([src[0] - q[0], src[1] - q[1], src[2] - 0.25])
            end = np.array([q[0], q[1], 0.25]) + v * max(0.0, 1.0 - 0.45 / max(np.linalg.norm(v), 1e-6))
            if self.t.line_of_sight([q[0], q[1], 0.25], end.tolist(), groups=(1, 0, 0, 0, 0, 0)):
                ok.append((q, dd, _crosses_boxes(q, src[:2], shields)))
        clear = [(q, dd) for q, dd, sh in ok if not sh]
        # the inverse-square fit models no shielding: survey points behind a shielding box (the layout's attenuating
        # walls and blocks) bias it by metres (practice site t2: 2.1 m with one such point, 0.1 m without); they
        # are used only when too few clear ones exist
        if len(clear) >= SURVEY["n_points"]:
            ok = clear
        else:
            self.note("survey_shielded_points_used", clear=len(clear), total=len(ok))
            ok = [(q, dd) for q, dd, _sh in ok]
        return ok

    def _survey_points(self, src, ok):
        """The survey's planned points among the candidates, spread in angle, in visiting order."""
        if not ok:
            return []
        p = self.t.pose()
        here = np.array([p["x"], p["y"]])
        # greedy angular spread, starting from the candidate nearest to the robot
        Q = np.array([q for q, _ in ok])
        ang = np.arctan2(Q[:, 1] - src[1], Q[:, 0] - src[0])
        chosen = [int(np.argmin(np.linalg.norm(Q - here, axis=1)))]
        while len(chosen) < SURVEY["n_points"]:
            gap = np.min(np.abs(np.angle(np.exp(1j * (ang[:, None] - ang[None, chosen])))), axis=1)
            dist_term = np.min(np.linalg.norm(Q[:, None, :] - Q[None, chosen, :], axis=2), axis=1)
            score = gap + 0.3 * dist_term
            score[chosen] = -1
            k = int(np.argmax(score))
            if score[k] <= 0.2:
                break
            chosen.append(k)
        # visiting order: nearest neighbour from the robot
        order, cur, left = [], here, list(chosen)
        while left:
            j = min(left, key=lambda i: np.linalg.norm(Q[i] - cur))
            order.append(Q[j])
            cur = Q[j]
            left.remove(j)
        return order

    def _shields(self, src):
        """The layout's attenuating boxes (cx, cy, half x, half y, rotation) that do not contain the source itself."""
        out = []
        for b in (self.site.get("radiation") or {}).get("boxes", []):
            cx, cy, hx, hy, rot = (float(v) for v in b[:5])
            if _in_box(src[0], src[1], (cx, cy, hx, hy, rot)):
                continue
            out.append((cx, cy, hx, hy, rot))
        return out

    # ------------------------------------------------------------------ the round
    def run(self):
        self.note("start", items=list(self.items), avoid_kinds=list(self.avoid_kinds))
        for mg in self.margins:                 # build the planning maps once (and the enterable caches)
            self.t.plan_avoiding((0.0, 0.0), (0.3, 0.0), avoid_kinds=self.avoid_kinds, margin=mg)
        try:
            self.r.set_mode("stand")
            self.d.mode = "stand"
            self.r.get_state()
            self.d.wait(10.0)
            st = self.r.get_state()
            sens = float(self.desc["dosimeter"]["sensitivity_cps_per_gyh"])
            self.background_gyh = st["dosimeter"]["counts"] / max(st["dosimeter"]["dwell_s"], 1e-6) / sens
            self.note("background", rate_gyh=self.background_gyh)
            order = list(self.t.route()["order"])
            for tag in self.items:
                if tag not in order:
                    order.append(tag)
            for tag in order:
                typ = self.items[tag]["type"]
                if typ in INSTRUMENTS:
                    self.read_instrument(tag)
                elif typ == "dose_point":
                    self.do_mark(tag)
                elif typ == "hotspot":
                    self.do_source(tag)
            pad = self.mission["datum"]["pos"]
            self.go(pad[0], pad[1], yaw=None, tol=0.10, allow_unsafe=True)
            self.r.set_mode("stand")
            self.d.mode = "stand"
            # wait for queued reports to go out
            for _ in range(20):
                if self.r.get_state()["reports_pending"] == 0:
                    break
                self.d.wait(0.5)
            self.d.wait(1.0)
            self.note("finish", pending=self.r.get_state()["reports_pending"])
            self.r.finish()
        except Halt:
            self.note("halt", reason=self.r.get_state()["done_reason"])
        if self.out_dir:
            with open(os.path.join(self.out_dir, "program_log.json"), "w") as f:
                json.dump({"log": self.log, "crossings": self.crossings}, f, indent=1, default=_js)


def solve(robot, truth, out_dir=None):
    Reference(robot, truth, out_dir=out_dir).run()


# ---------------------------------------------------------------------------------------------- helpers
AFTER_CROSS_AHEAD_M = 0.15       # a walk point is beyond the crossing's exit when this far past B along the crossing
AFTER_CROSS_LANE_M = 0.60        # ... or this far to the side of the crossing line (the path turns after the element)


def after_crossing_index(w, B, fwd):
    """Index of the first point of the walk segment `w` that follows a square crossing ending at B (direction fwd):
    the leading points still on the crossing (not more than AFTER_CROSS_AHEAD_M past B along fwd and within
    AFTER_CROSS_LANE_M of the crossing line) are skipped; w[0] (the planner's switch point) never counts.

    Version 2. Earlier the test was the distance to B (< 0.6 m): the planner's walk
    segment starts at the element's far edge, and B lies 0.75 m past the element's centreline, so on development site
    k02 (a 5.3 cm pipe E032 across the valve gallery's aisle) the first kept point was 0.626 m behind B, between B
    and the pipe: the robot turned round toward it, was stopped by the pipe, crossed it back by a mode change and
    then again (9 `blocked`, about 95 s with the holds; admission row 3's time share). Whether it happened depended
    on a few centimetres of where the plan began."""
    fwd = np.asarray(fwd, float)
    fwd = fwd / max(float(np.linalg.norm(fwd)), 1e-12)
    lat = np.array([-fwd[1], fwd[0]])
    B = np.asarray(B, float)
    j = 1
    while j < len(w):
        d = np.asarray(w[j], float) - B
        if float(np.dot(d, fwd)) > AFTER_CROSS_AHEAD_M or abs(float(np.dot(d, lat))) >= AFTER_CROSS_LANE_M:
            break
        j += 1
    return j


def _in_quad(Q, pt):
    """True when pt lies inside the convex quadrilateral Q (4 x 2, either winding)."""
    Q = np.asarray(Q, float)
    x, y = float(pt[0]), float(pt[1])
    sgn = 0.0
    for i in range(4):
        (x1, y1), (x2, y2) = Q[i], Q[(i + 1) % 4]
        c = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if abs(c) < 1e-12:
            continue
        if sgn == 0.0:
            sgn = math.copysign(1.0, c)
        elif math.copysign(1.0, c) != sgn:
            return False
    return True


def _in_box(x, y, box, margin=0.0):
    cx, cy, hx, hy, rot = box
    c, s_ = math.cos(rot), math.sin(rot)
    dx, dy = x - cx, y - cy
    u, v = c * dx + s_ * dy, -s_ * dx + c * dy
    return abs(u) <= hx + margin and abs(v) <= hy + margin


def _crosses_boxes(p, q, boxes, steps=48):
    """Whether the 2-D segment p -> q passes through any of the boxes (sampled every 1/steps of its length)."""
    p, q = np.asarray(p, float)[:2], np.asarray(q, float)[:2]
    for t in np.linspace(0.0, 1.0, steps + 1)[1:-1]:
        x, y = p + t * (q - p)
        for b in boxes:
            if _in_box(x, y, b):
                return True
    return False


def _line_spread(meas):
    """RMS distance (m) of the readings' stances from their principal line; 0 for fewer than three."""
    if len(meas) < 3:
        return 0.0
    P = np.array([[m["x"], m["y"]] for m in meas], float)
    s = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    return float(s[-1] / math.sqrt(len(P)))


def _survey_extra(cands, meas, tried):
    """A further survey candidate: farthest from the readings' principal line (with fewer than two readings, from the
    readings), at least SURVEY_EXTRA_AWAY_M from every goal tried and 0.5 m from every reading; None if none is left."""
    if not cands:
        return None
    Q = np.array([q for q, _ in cands], float)
    keep = np.ones(len(Q), bool)
    for t in tried:
        keep &= np.hypot(Q[:, 0] - t[0], Q[:, 1] - t[1]) >= SURVEY_EXTRA_AWAY_M
    P = np.array([[m["x"], m["y"]] for m in meas], float).reshape(-1, 2)
    for m in P:
        keep &= np.hypot(Q[:, 0] - m[0], Q[:, 1] - m[1]) >= 0.5
    if not keep.any():
        return None
    if len(P) >= 2:
        c0 = P.mean(axis=0)
        u = np.linalg.svd(P - c0)[2][0]
        score = np.abs((Q - c0) @ np.array([-u[1], u[0]]))
    elif len(P) == 1:
        score = np.hypot(Q[:, 0] - P[0, 0], Q[:, 1] - P[0, 1])
    else:
        score = np.zeros(len(Q))
    score[~keep] = -1.0
    return Q[int(np.argmax(score))]


def fit_source(meas, sens, background_gyh, z_source=0.45):
    """Inverse-square fit of (x, y, S) to dosimeter counts at known poses (Poisson deviance, background fixed)."""
    from scipy.optimize import least_squares
    if len(meas) < 3:
        return {"ok": False, "why": "fewer than 3 points"}
    X = np.array([[m["x"], m["y"], m["z"]] for m in meas], float)
    C = np.array([m["counts"] for m in meas], float)
    W = np.array([m["dwell"] for m in meas], float)
    rate = C / np.maximum(W, 1e-6) / sens
    k = int(np.argmax(rate))
    wts = np.maximum(rate - background_gyh, 1e-6) ** 2
    x0 = np.r_[(X[:, :2] * wts[:, None]).sum(0) / wts.sum(), max(rate[k], 1e-3) * 1.0]

    def model(p):
        r2 = (X[:, 0] - p[0]) ** 2 + (X[:, 1] - p[1]) ** 2 + (X[:, 2] - z_source) ** 2
        return W * sens * (background_gyh + abs(p[2]) / np.maximum(r2, 0.0625))

    def resid(p):
        lam = model(p)
        return (C - lam) / np.sqrt(np.maximum(lam, 1.0))

    best = None
    for dx, dy in ((0, 0), (0.7, 0), (-0.7, 0), (0, 0.7), (0, -0.7)):
        sol = least_squares(resid, x0 + np.array([dx, dy, 0.0]), method="lm", max_nfev=2000)
        cost = float(np.sum(sol.fun ** 2))
        if best is None or cost < best[0]:
            best = (cost, sol)
    cost, sol = best
    return {"ok": True, "x": round(float(sol.x[0]), 3), "y": round(float(sol.x[1]), 3),
            "S_gyh_at_1m": round(abs(float(sol.x[2])), 3), "chi2": round(cost, 3), "n": len(meas),
            "rates_gyh": [round(float(v), 4) for v in rate]}


def _poly_near(fp, pts, tol):
    """Whether the polyline pts passes within tol / 2 of the footprint fp (matplotlib's contains_point(radius=+-tol)
    semantics, the original test; nav.near_polygon reproduces it exactly)."""
    fp = np.asarray(fp, float)
    pts = np.asarray(pts, float)
    for i in range(len(pts) - 1):
        for tt in np.linspace(0, 1, 25):
            q = pts[i] + tt * (pts[i + 1] - pts[i])
            if near_polygon(fp, q, tol):
                return True
    return False


def _body_rect(p, half=None):
    """The body's footprint (BODY_HALF: half length, half width) at pose p as a counter-clockwise rectangle."""
    hl, hw = BODY_HALF if half is None else half
    c, s = math.cos(float(p["yaw"])), math.sin(float(p["yaw"]))
    x, y = float(p["x"]), float(p["y"])
    return np.array([[x + c * a - s * b, y + s * a + c * b] for a, b in ((hl, hw), (-hl, hw), (-hl, -hw), (hl, -hw))])


def _in_poly(P, q):
    """q inside the polygon P (even-odd rule) or on its boundary."""
    x, y = float(q[0]), float(q[1])
    inside = False
    n = len(P)
    for i in range(n):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % n]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def _seg_dist(q, a, b):
    ab = b - a
    u = float(np.clip(np.dot(q - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(q - (a + u * ab)))


def _poly_dist(P, Q):
    """Plan distance between two polygons (0 when they overlap)."""
    P, Q = np.asarray(P, float).reshape(-1, 2), np.asarray(Q, float).reshape(-1, 2)
    if len(P) == 0 or len(Q) == 0:
        return float("inf")
    if (len(Q) >= 3 and any(_in_poly(Q, p) for p in P)) or (len(P) >= 3 and any(_in_poly(P, q) for q in Q)):
        return 0.0
    best = float("inf")
    for A, B in ((P, Q), (Q, P)):
        m = len(B)
        for q in A:
            for i in range(m if m > 1 else 0):
                best = min(best, _seg_dist(q, B[i], B[(i + 1) % m]))
            if m == 1:
                best = min(best, float(np.linalg.norm(q - B[0])))
    # crossing edges (no vertex inside the other): the vertex distances above are then > 0 but the polygons touch
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        for j in range(len(Q)):
            c, d = Q[j], Q[(j + 1) % len(Q)]
            d1 = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            d2 = (b[0] - a[0]) * (d[1] - a[1]) - (b[1] - a[1]) * (d[0] - a[0])
            d3 = (d[0] - c[0]) * (a[1] - c[1]) - (d[1] - c[1]) * (a[0] - c[0])
            d4 = (d[0] - c[0]) * (b[1] - c[1]) - (d[1] - c[1]) * (b[0] - c[0])
            if d1 * d2 < 0 and d3 * d4 < 0:
                return 0.0
    return best


def _outline_points(P, step=0.05):
    """Points every `step` m along a closed polygon's boundary (its corners included)."""
    P = np.asarray(P, float).reshape(-1, 2)
    out = []
    for i in range(len(P)):
        a, b = P[i], P[(i + 1) % len(P)]
        L = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(L / step)))
        for k in range(n):
            out.append(a + (b - a) * (k / n))
    return out


def _bends(wps, tol):
    """Whether a polyline leaves the straight line between its ends by more than tol."""
    P = np.asarray(wps, float)
    if len(P) < 3:
        return False
    a, b = P[0], P[-1]
    d = b - a
    L = float(np.linalg.norm(d))
    if L < 1e-6:
        return False
    n = np.array([-d[1], d[0]]) / L
    return bool(np.max(np.abs((P - a) @ n)) > tol)


def _direction(pts, length):
    """Unit direction of the first `length` metres of a polyline."""
    P = np.asarray(pts, float)
    acc = 0.0
    for i in range(1, len(P)):
        acc += float(np.linalg.norm(P[i] - P[i - 1]))
        if acc >= length:
            d = P[i] - P[0]
            return d / max(np.linalg.norm(d), 1e-9)
    d = P[-1] - P[0]
    return d / max(np.linalg.norm(d), 1e-9)


def _take_length(pts, length):
    out, acc = [], 0.0
    for i, q in enumerate(pts):
        if i > 0:
            acc += math.hypot(q[0] - pts[i - 1][0], q[1] - pts[i - 1][1])
        out.append(list(q))
        if acc >= length:
            break
    return out


def _trim_to(pts, A):
    """The walk polyline up to the point nearest to A, then A."""
    pts = [list(p) for p in pts]
    A = np.asarray(A, float)
    if len(pts) <= 1:
        return pts + [A.tolist()]
    d = [np.linalg.norm(np.asarray(p) - A) for p in pts]
    k = int(np.argmin(d))
    out = pts[:max(1, k)]
    # drop a last point that lies beyond A as seen from the previous one
    out.append(A.tolist())
    return out


def _agree(typ, a, b, item):
    if typ == "gauge":
        lo, hi = item["scale"]["range"]
        return abs(a - b) <= 0.012 * (hi - lo)
    if typ in ("level", "valve_handwheel"):
        return abs(a - b) <= 2.5
    if typ == "valve_lever":
        return abs(a - b) <= 3.0
    if typ == "lamps":
        return list(a) == list(b)
    return True


def _combine(typ, vals):
    if typ == "lamps":
        from collections import Counter
        return list(Counter(tuple(v) for v in vals).most_common(1)[0][0])
    v = float(np.median(np.asarray(vals, float))) if len(vals) != 2 else float(np.mean(vals))
    return round(v, 4)


def _short(v):
    if isinstance(v, float):
        return "%.2f" % v
    return str(v)


def _r(a):
    return [round(float(x), 3) for x in a]


def _js(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)
