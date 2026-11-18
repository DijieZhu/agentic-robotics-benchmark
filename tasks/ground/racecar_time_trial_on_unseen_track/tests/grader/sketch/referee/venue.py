"""The cell of one venue: four sessions, the simulation, the referee and the staff.

    venue = Venue(world, config)               # config: the parsed grader configuration (default: this folder's)
    venue.handle("describe", {})               # the wire's dispatcher: a result dict or {"error", "detail"}
    venue.open_session("practice"); venue.step(0.1, 1.5); venue.close_session(); venue.request_pickup()
    venue.account(dt)                          # the program's time since the last reply (the transport measures it)
    venue.result()                             # the venue result record (records.schema.json#/$defs/result)
    venue.truth_state()                        # privileged state

Simulated time passes only in `step`. One tick: the command
through the delay line, then N physics steps, each `car.pre_step()`, slow cars' `pre_step()`, `mj_step`, the
kinematics refresh, the trackers' `update()` (ducts, boxes), the sensors' `after_step(i)` and the referee's tests; at
the end of the tick the observation and the events.

Later decisions applied here: after the fourth session's `session_end` the first
`close_session` is an acknowledgement, every later call `venue_over`, and the result is final at `session_end`;
one SensorSuite per venue, carried across the switch of models with `rebind` (the IMU bias is drawn once per venue);
`staff.putback_err_m` passed to both trackers; the privileged names live in `privileged.py`.
"""

import collections
import hashlib
import json
import math
import time

import mujoco
import numpy as np

from sketch.examples import check
from sketch.examples import constants as K

from . import config as C
from . import privileged as PRIV
from .geometry import FastFloor, FastTrack, Line, Tape, car_corners
from .providers import Providers, build_model, resolve
from .rules import LapClock, Overtakes, Touches, Watch
from .staff import Relocator

EPS = 1e-9
DETAILS = {
    "no_session_open": "no session is open",
    "no_session_ended": "the session has ended; close it or open the next one",
    "bad_order_open": "a session is open; close it first",
    "venue_over": "the venue is over",
    "forbidden": "not a method of the car interface",
}


def _err(code, detail=None):
    out = {"error": code}
    if detail:
        out["detail"] = detail
    return out


def _yaw_of(xmat):
    return math.atan2(float(xmat[3]), float(xmat[0]))


class _Sim(object):
    """One compiled model of a variant and the run-time objects bound to it."""

    def __init__(self, venue, variant, sensors=None):
        w, prov, cfg = venue.world, venue.prov, venue.cfg
        self.variant = variant
        self.model = build_model(w, variant, prov, timestep=cfg["sim"]["physics_dt_s"])
        self.model.opt.timestep = cfg["sim"]["physics_dt_s"]
        self.data = mujoco.MjData(self.model)
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        self.car = prov.CarRuntime(m, d, prefix=K.EGO_PREFIX, grip_at=venue.floor.grip_at)
        self.car.place(w["truth"]["start_pose"]["pose"])
        if sensors is not None and hasattr(sensors, "rebind"):
            # one suite per venue: its streams and IMU bias carry on in the new model
            sensors.rebind(m, d, self.car)
            self.sensors = sensors
        else:
            self.sensors = prov.SensorSuite(m, d, w, self.car, prefix=K.EGO_PREFIX, seed=venue.rng["sensors"])
        err = float(cfg["staff"]["putback_err_m"])
        self.ducts = prov.DuctTracker(m, d, w, seed=venue.rng["ducts"], putback_err_m=err)
        self.boxes = self.slow = None
        if variant == "avoid":
            self.boxes = prov.BoxTracker(m, d, w, seed=venue.rng["boxes"], putback_err_m=err)
            self.slow = prov.SlowCars(m, d, w, grip_at=venue.floor.grip_at, seed=venue.rng["others"])
        oid = mujoco.mjtObj
        self.cid = mujoco.mj_name2id(m, oid.mjOBJ_BODY, K.EGO_PREFIX + "chassis")
        self.ref_sid = mujoco.mj_name2id(m, oid.mjOBJ_SITE, K.EGO_PREFIX + "ref")
        self.wheel_bids = [mujoco.mj_name2id(m, oid.mjOBJ_BODY, K.EGO_PREFIX + w_ + "/wheel") for w_ in K.WHEELS]
        if min([self.cid, self.ref_sid] + self.wheel_bids) < 0:
            raise ValueError("the model lacks the ego car's binding names (CONTRACT 4.2)")
        root = m.body_rootid
        gbody = m.geom_bodyid
        collides = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        self.ego = (root[gbody] == self.cid) & collides
        obj = np.zeros(m.ngeom, dtype=bool)
        slow_roots = [mujoco.mj_name2id(m, oid.mjOBJ_BODY, K.slow_prefix(i) + "chassis")
                      for i in range(len(w["truth"]["avoidance"]["slow_cars"]))] if variant == "avoid" else []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, oid.mjOBJ_GEOM, g) or ""
            rule = K.geom_rule(name)
            if rule is not None and rule[1] in ("duct", "box"):
                obj[g] = True
            elif int(root[gbody[g]]) in slow_roots and rule is not None and rule[1] == "car":
                obj[g] = True
        self.obj = obj
        w = mujoco.mjtWarning                    # MuJoCo resets the state itself when one of these fires
        self.bad = [int(w.mjWARN_BADQPOS), int(w.mjWARN_BADQVEL), int(w.mjWARN_BADQACC)]

    def ref_xy(self):
        p = self.data.site_xpos[self.ref_sid]
        return float(p[0]), float(p[1])

    def car_pose(self):
        d = self.data
        return [float(d.xpos[self.cid][0]), float(d.xpos[self.cid][1]), _yaw_of(d.xmat[self.cid])]


class _Session(object):
    """The referee's state of one open session."""

    def __init__(self, venue, index):
        cfg = venue.cfg
        self.index = index
        self.cfg = cfg["sessions"][index]
        self.name = self.cfg["name"]
        self.ticks_total = C.session_ticks(cfg, self.cfg)
        self.tick = 0
        self.phases = self.cfg["phases"]
        self.phase_idx = 0
        self.lap = LapClock(len(venue.lines) - 1)
        self.touches = Touches(venue.params["touch_merge_s"])
        st = cfg["staff"]
        self.watch = Watch(cfg["sim"]["physics_dt_s"], st["speeding_hold_s"], cfg["scoring"]["avoid"]["standstill_s"],
                           cfg["scoring"]["avoid"]["standstill_speed_mps"])
        self.overtakes = Overtakes(venue.params["overtake_window_m"])
        self.delay = collections.deque([(0.0, 0.0)] * int(cfg["sim"]["cmd_delay_ticks"]))
        self.stopped = False
        self.t_stop = None
        self.relocated = False
        self.pickup_pending = False
        self.end = None                   # time / closed / warnings / void / aborted
        self.end_pending = None
        self.n_touch = self.crashes = self.pickups = self.warnings = self.out_of_bounds = 0
        self.max_disp = 0.0
        self.program_s = 0.0
        self.desk_s = 0.0
        self.ref_prev = None
        self.clean = None                 # (s, d) of the reference point at the last clean position
        self.known_displaced = set()
        self.last_obj_t = -math.inf
        self.log = []                     # internal: (t, what, detail); never on the wire

    def record(self):
        return {"name": self.name, "ticks": self.tick, "end": self.end or "aborted", "laps": list(self.lap.laps),
                "touches": self.n_touch, "crashes": self.crashes, "pickups": self.pickups, "warnings": self.warnings,
                "out_of_bounds": self.out_of_bounds, "max_displacement_m": round(self.max_disp, 6),
                "program_s": round(self.program_s, 6), "desk_s": round(self.desk_s, 6)}


class Venue(object):
    def __init__(self, world, config=None, providers="auto", params=None, priv_token_sha256=None):
        if world["layout_digest"] != check.world_digest(world):
            raise ValueError("layout_digest of %s does not match its content" % world.get("world_id"))
        self.world = world
        self.cfg = config if config is not None else C.load_config()
        self.params = params if params is not None else C.load_params()
        self.prov = providers if isinstance(providers, Providers) else resolve(providers)
        t = world["truth"]
        self.track = FastTrack(world)
        self.floor = FastFloor(world)
        self.lines = [Line(t["timing_line"], "timing_line", -1)] + [Line(c, c["id"], i) for i, c in enumerate(t["checkpoints"])]
        self.tapes = [Tape(o) for o in t["open_walls"]]
        seeds = t["seeds"]
        self.rng = {"sensors": np.random.default_rng(seeds["sensors"]),
                    "ducts": np.random.default_rng([seeds["putback"], 4]),
                    "boxes": np.random.default_rng([seeds["putback"], 5]),
                    "others": np.random.default_rng(seeds["others"])}
        self.relocator = Relocator(self.track, self.lines, self.cfg, self.params, seeds["relocation"])
        st = self.cfg["staff"]
        self.thr = float(st["crash_displacement_m"])
        self.wheels_over = int(st["open_wall_wheels_over"])
        self.cos_roll = math.cos(float(st["rollover_tilt_rad"]))
        self.hold_s, self.relocate_at_s = float(st["hold_s"]), float(st["relocate_at_s"])
        self.N = int(self.cfg["sim"]["physics_steps_per_tick"])
        self.dt = float(self.cfg["sim"]["physics_dt_s"])
        self.tick_s = float(self.cfg["sim"]["tick_s"])
        cmd = self.cfg["command"]
        self.steer_max, self.v_min, self.v_max = cmd["steer_max_rad"], cmd["speed_min_mps"], cmd["speed_max_mps"]
        self.caps = C.program_caps(self.cfg)
        self.state = "idle"               # idle | open | ended | between | over
        self.next_idx = 0
        self.ses = None
        self.sessions = []
        self.sims = {}
        self.sim = None
        self.counts = {"forbidden_calls": 0, "bad_calls": 0}
        self.out_of_bounds = 0
        self.end_reason = None
        self.step_called = False
        self.priv_sha = priv_token_sha256
        self.unlocked = False
        self.last_close_pending = False   # the fourth session ended by session_end, its close_session not yet seen
        self.program_total_s = 0.0
        self.after_over_s = 0.0
        self._desk = {}
        self.timing = {"ticks": 0, "cell_s": 0.0, "physics_s": 0.0, "sensors_s": 0.0, "trackers_s": 0.0, "referee_s": 0.0}
        self.log = []

    # ------------------------------------------------------------------ the dispatcher
    @property
    def over(self):
        return self.state == "over"

    def handle(self, method, params=None):
        """Answer one request: the `result` dict, or {"error": code, "detail": text}."""
        if method not in PRIV.METHODS or (method in PRIV.PRIVILEGED_METHODS and method != "priv_unlock"
                                           and not self.unlocked) or (method == "priv_unlock" and self.priv_sha is None):
            self.counts["forbidden_calls"] += 1
            return _err("forbidden")
        if self.over and not (method == "close_session" and self.last_close_pending):
            return _err("venue_over", DETAILS["venue_over"])
        detail = PRIV.check_params(method, params)
        if detail:
            self.counts["bad_calls"] += 1
            return _err("bad_params", detail)
        params = params or {}
        if method == "describe":
            return self.describe()
        if method == "open_session":
            return self.open_session(params["name"])
        if method == "step":
            return self.step(params["steer"], params["speed"])
        if method == "close_session":
            return self.close_session()
        if method == "request_pickup":
            return self.request_pickup()
        if method == "priv_unlock":
            if hashlib.sha256(params["token"].encode("ascii")).hexdigest() != self.priv_sha:
                self.counts["forbidden_calls"] += 1
                return _err("forbidden")
            self.unlocked = True
            return {"ok": True}
        if method == "priv_world":
            return json.loads(json.dumps(self.world))
        if method == "priv_state":
            return self.truth_state()
        return _err("forbidden")                                    # pragma: no cover

    # ------------------------------------------------------------------ budgets
    def account(self, dt):
        """Charge `dt` seconds of program time (wall clock between the last reply and this request) to the session
        that is open, or to the desk time before the next session. Returns the end reason when a cap is exceeded
        (the venue is then over) or None."""
        dt = max(0.0, float(dt))
        self.program_total_s += dt
        if self.over:
            self.after_over_s += dt
            return None
        if self.state == "open":
            self.ses.program_s += dt
            if self.ses.program_s > self.caps["track"][self.ses.name] + EPS:
                self.end("program_time_cap")
                return "program_time_cap"
        elif self.next_idx < len(K.SESSIONS):
            name = K.SESSIONS[self.next_idx]
            self._desk[name] = self._desk.get(name, 0.0) + dt
            if self._desk[name] > self.caps["desk"][name] + EPS:
                self.end("desk_time_cap")
                return "desk_time_cap"
        return None

    def budget_left(self):
        """Seconds of program time left before a cap ends the venue (inf when none applies)."""
        if self.over:
            return math.inf
        if self.state == "open":
            return self.caps["track"][self.ses.name] - self.ses.program_s
        if self.next_idx < len(K.SESSIONS):
            name = K.SESSIONS[self.next_idx]
            return self.caps["desk"][name] - self._desk.get(name, 0.0)
        return math.inf

    def end(self, reason):
        """End the venue (idempotent): an open session is aborted; the first reason given is kept."""
        if self.ses is not None and self.ses.end is None:
            self.ses.end = "aborted"
        if self.end_reason is None:
            self.end_reason = reason
        self.state = "over"

    # ------------------------------------------------------------------ public calls
    def describe(self):
        if self.over:
            return _err("venue_over", DETAILS["venue_over"])
        return C.describe_result(self.world, self.cfg)

    def open_session(self, name):
        if self.over:
            return _err("venue_over", DETAILS["venue_over"])
        if self.state == "open":
            return _err("bad_order", DETAILS["bad_order_open"])
        if self.next_idx >= len(K.SESSIONS) or name != K.SESSIONS[self.next_idx]:
            return _err("bad_order", "the next session is %s" % K.SESSIONS[self.next_idx])
        if self.state == "ended":
            self._close_current()
        ses = _Session(self, self.next_idx)
        ses.desk_s = self._desk.get(name, 0.0)
        self.next_idx += 1
        variant = "avoid" if name == "avoid" else "track"
        self._enter_variant(variant)
        sim = self.sim
        # the staff repair what the last session left displaced by the crash threshold or more; the rest stays
        self._put_back(sim)
        sim.car.place(self.world["truth"]["start_pose"]["pose"])
        sim.car.set_targets(0.0, 0.0)
        if hasattr(sim.sensors, "reset_motion"):
            sim.sensors.reset_motion()
        mujoco.mj_forward(sim.model, sim.data)
        ses.ref_prev = sim.ref_xy()
        s, d = self.track.project1(*ses.ref_prev)[:2]
        ses.clean = (s, d)
        self.ses = ses
        self.sessions.append(ses)
        self.state = "open"
        return {"session": name, "duration_s": ses.cfg["duration_s"], "ticks": ses.ticks_total,
                "phases": [dict(p) for p in ses.phases]}

    def _enter_variant(self, variant):
        if variant in self.sims:
            self.sim = self.sims[variant]
            return
        old = self.sim
        sim = _Sim(self, variant, sensors=old.sensors if old is not None else None)
        if old is not None:                      # carry the ducts' poses and the poses they were last set to
            ref = old.ducts.reference_poses() if hasattr(old.ducts, "reference_poses") else None
            sim.ducts.set_poses(old.ducts.poses(), reference=ref)
            mujoco.mj_forward(sim.model, sim.data)
        self.sims[variant] = sim
        self.sim = sim

    def close_session(self):
        if self.over and self.last_close_pending:
            # the first close_session after the fourth session's session_end is an acknowledgement like
            # for any other session; the venue is over already and its result is final, nothing changes
            self.last_close_pending = False
            ses = self.ses
            return {"session": ses.name, "ticks": ses.tick}
        if self.over:
            return _err("venue_over", DETAILS["venue_over"])
        if self.state not in ("open", "ended"):
            return _err("no_session", DETAILS["no_session_open"])
        ses = self.ses
        if ses.end is None:
            ses.end = "closed"
        self._close_current()
        return {"session": ses.name, "ticks": ses.tick}

    def _close_current(self):
        self.state = "between"
        if self.next_idx >= len(K.SESSIONS):
            self.state = "over"
            if self.end_reason is None:
                self.end_reason = "finish"

    def request_pickup(self):
        if self.over:
            return _err("venue_over", DETAILS["venue_over"])
        if self.state != "open":
            return _err("no_session", DETAILS["no_session_ended"] if self.state == "ended" else DETAILS["no_session_open"])
        ses = self.ses
        if ses.stopped or ses.pickup_pending:
            return {"accepted": False}
        ses.pickup_pending = True
        return {"accepted": True}

    def step(self, steer, speed):
        if self.over:
            return _err("venue_over", DETAILS["venue_over"])
        if self.state == "ended":
            return _err("no_session", DETAILS["no_session_ended"])
        if self.state != "open":
            return _err("no_session", DETAILS["no_session_open"])
        self.step_called = True
        t_start = time.perf_counter()
        out = self._tick(float(steer), float(speed))
        self.timing["cell_s"] += time.perf_counter() - t_start
        self.timing["ticks"] += 1
        if self.unlocked:
            out["truth"] = self.truth_state()
        return out

    # ------------------------------------------------------------------ one tick
    def _tick(self, steer, speed):
        ses, sim = self.ses, self.sim
        m, d, car = sim.model, sim.data, sim.car
        events = []
        t0 = ses.tick * self.tick_s
        while ses.phase_idx + 1 < len(ses.phases) and t0 >= ses.phases[ses.phase_idx + 1]["t0_s"] - EPS:
            ses.phase_idx += 1
            name = ses.phases[ses.phase_idx]["name"]
            events.append({"type": "phase", "name": name})
            if name == "dynamic" and sim.slow is not None:
                sim.slow.enter()
                ses.overtakes.forget()
        if ses.pickup_pending:
            ses.pickup_pending = False
            if not ses.stopped:
                self._stop(t0, "pickup", events)
        cmd = (0.0, 0.0) if ses.stopped else (max(-self.steer_max, min(self.steer_max, steer)),
                                              max(self.v_min, min(self.v_max, speed)))
        ses.delay.append(cmd)
        target = ses.delay.popleft()
        if ses.stopped:
            target = (0.0, 0.0)
        car.set_targets(*target)
        sim.sensors.begin_tick()
        tm = self.timing
        refresh = bool(self.params.get("refresh_kinematics", True))
        for i in range(1, self.N + 1):
            ta = time.perf_counter()
            car.pre_step()
            if sim.slow is not None:
                sim.slow.pre_step()
            mujoco.mj_step(m, d)
            if refresh:
                # mj_step leaves positions and com velocities at the state before its integration: bring them to the
                # state after step i, for the sensors, the trackers and the referee alike. The
                # contacts stay those that acted during the step.
                mujoco.mj_kinematics(m, d)
                mujoco.mj_comPos(m, d)
                mujoco.mj_comVel(m, d)
            tb = time.perf_counter()
            # the trackers before the sensors: a moving duct is where it is in this
            # step's sub-scan
            dmax = sim.ducts.update()
            if sim.boxes is not None:
                dmax = max(dmax, sim.boxes.update())
            tc = time.perf_counter()
            sim.sensors.after_step(i)
            td = time.perf_counter()
            self._referee_step(t0 + i * self.dt, dmax, events)
            te = time.perf_counter()
            tm["physics_s"] += tb - ta
            tm["trackers_s"] += tc - tb
            tm["sensors_s"] += td - tc
            tm["referee_s"] += te - td
        obs = sim.sensors.end_tick()
        ses.tick += 1
        if not np.isfinite(d.qpos).all() or any(d.warning[k].number > 0 for k in sim.bad):
            self.log.append((ses.name, ses.tick, "physics_fault", None))
            self.end("physics_fault")
        if ses.end_pending is None and ses.tick >= ses.ticks_total:
            ses.end_pending = "time"
        if ses.end_pending is not None:
            events.append({"type": "session_end", "reason": ses.end_pending})
            ses.end = ses.end_pending
            if ses.end == "void":
                self.end("void")
            elif self.state != "over":
                self.state = "ended"
                if self.next_idx >= len(K.SESSIONS):
                    self.state = "over"
                    self.last_close_pending = True
                    if self.end_reason is None:
                        self.end_reason = "finish"
        return {"tick": ses.tick, "scan": obs["scan"], "imu": obs["imu"], "wheel_speed": obs["wheel_speed"],
                "steer_echo": obs["steer_echo"], "events": events}

    # ------------------------------------------------------------------ the referee after one physics step
    def _contact(self, sim):
        d = sim.data
        n = d.ncon
        if not n:
            return False
        g = d.contact.geom
        a, b = g[:, 0], g[:, 1]
        hit = (sim.ego[a] & sim.obj[b]) | (sim.ego[b] & sim.obj[a])
        if not hit.any():
            return False
        return bool((d.contact.dist[hit] <= 0.0).any())        # a contact inside a margin is not a touch

    def _referee_step(self, t, dmax, events):
        ses, sim = self.ses, self.sim
        if dmax > ses.max_disp:
            ses.max_disp = dmax
        ref = sim.ref_xy()
        if ses.end_pending is not None:
            ses.ref_prev = ref
            return
        if ses.stopped:
            if not ses.relocated and t >= ses.t_stop + self.relocate_at_s - EPS:
                self._relocate(t)
            elif t >= ses.t_stop + self.hold_s - EPS:
                if not ses.relocated:
                    self._relocate(t)
                ses.stopped = False
                events.append({"type": "released"})
                ses.log.append((t, "released", None))
            ses.ref_prev = sim.ref_xy()
            return
        d = sim.data
        wheels = d.xpos[sim.wheel_bids, :2].tolist()
        touching_obj = self._contact(sim)
        beyond = 0
        for tape in self.tapes:
            if tape.near(ref):
                beyond = max(beyond, tape.count(wheels))
        touching = touching_obj or beyond > 0
        if touching_obj:
            ses.last_obj_t = t
        if ses.touches.step(t, touching):
            ses.n_touch += 1
            ses.lap.touch()
            ses.log.append((t, "touch", "tape" if not touching_obj else "object"))
        # lines: in the order the reference point meets them
        p0 = ses.ref_prev
        hits = []
        for ln in self.lines:
            c = ln.crossing(p0, ref)
            if c is not None:
                hits.append((c[1], ln, c[0]))
        for frac, ln, direction in sorted(hits, key=lambda h: h[0]):
            if ln.index < 0:
                ev = ses.lap.timing(t - self.dt + frac * self.dt, direction, ses.touches.active(t))
                if ev is not None:
                    events.append(ev)
            else:
                ses.lap.checkpoint(ln.index, direction)
        v = math.hypot(ref[0] - p0[0], ref[1] - p0[1]) / self.dt
        ses.ref_prev = ref
        warn, still = ses.watch.step(v, ses.phases[ses.phase_idx]["speed_limit_mps"])
        if still:
            ses.lap.standstill()
        if warn:
            ses.warnings += 1
            events.append({"type": "warning", "count": ses.warnings})
            ses.log.append((t, "warning", ses.warnings))
            if ses.warnings >= int(self.cfg["staff"]["warnings_end_session"]):
                ses.end_pending = "warnings"
                return
        s_ref, d_ref, _, _ = self.track.project1(*ref)
        if sim.slow is not None and ses.phases[ses.phase_idx]["name"] == "dynamic":
            gaps = {}
            for stt in sim.slow.states():
                if stt["on_track"]:
                    x, y, yaw = stt["pose"]
                    ss = self.track.project1(x + K.REF_POINT_C[0] * math.cos(yaw), y + K.REF_POINT_C[0] * math.sin(yaw))[0]
                    gaps[stt["id"]] = self.track.signed_gap(s_ref, ss)
            k = ses.overtakes.step(gaps)
            if k:
                ses.lap.overtake(k)
        inside = [self.track.inside1(x, y)[0] for x, y in wheels]
        crash = None
        if dmax >= self.thr:
            newly = set(("duct",) + tuple(p) for p in sim.ducts.displaced(self.thr))
            if sim.boxes is not None:
                newly |= set(("box", b) for b in sim.boxes.displaced(self.thr))
            newly -= ses.known_displaced
            if newly:
                if t - ses.last_obj_t <= float(self.params["displacement_attribution_s"]) + EPS:
                    crash = "displacement"
                else:
                    ses.known_displaced |= newly
                    ses.log.append((t, "displaced_unattributed", sorted(map(str, newly))))
        if crash is None and beyond >= self.wheels_over:
            crash = "open_wall"
        if crash is None and float(d.xmat[sim.cid][8]) < self.cos_roll:
            crash = "rollover"
        if crash is None and not any(inside):
            crash = "out_of_bounds"
        if crash is not None:
            self._stop(t, crash, events)
        elif all(inside) and not touching:
            ses.clean = (s_ref, d_ref)

    # ------------------------------------------------------------------ the staff
    def _stop(self, t, kind, events):
        ses, sim = self.ses, self.sim
        ses.stopped = True
        ses.t_stop = t
        ses.relocated = False
        if kind == "pickup":
            ses.pickups += 1
        else:
            ses.crashes += 1
        ses.lap.interrupt()
        sim.car.set_targets(0.0, 0.0)
        ses.delay = collections.deque([(0.0, 0.0)] * len(ses.delay))
        events.append({"type": "stopped"})
        ses.log.append((t, "stopped", kind))
        if kind == "out_of_bounds":
            ses.out_of_bounds += 1
            self.out_of_bounds += 1
            if self.out_of_bounds >= int(self.cfg["staff"]["out_of_bounds_void_at"]):
                ses.end_pending = "void"

    def _obstacles(self, sim):
        polys = []
        if sim.boxes is not None:
            disp = set(sim.boxes.displaced(self.thr))
            poses = sim.boxes.poses()
            for i, bx in enumerate(self.world["truth"]["avoidance"]["boxes"]):
                pose = bx["pose"] if bx["id"] in disp else list(poses[i])
                polys.append(check.box_corners(pose, bx["size_m"][0], bx["size_m"][1]))
        if sim.slow is not None:
            for stt in sim.slow.states():
                if stt["on_track"]:
                    polys.append(car_corners(stt["pose"]))
        return polys

    def _put_back(self, sim):
        pieces = sim.ducts.displaced(self.thr)
        if pieces:
            sim.ducts.put_back(pieces)
        if sim.boxes is not None:
            boxes = sim.boxes.displaced(self.thr)
            if boxes:
                sim.boxes.put_back(boxes)

    def _relocate(self, t):
        ses, sim = self.ses, self.sim
        pose, info = self.relocator.place(ses.clean[0], ses.clean[1], self._obstacles(sim))
        sim.car.place(pose)
        sim.car.set_targets(0.0, 0.0)
        self._put_back(sim)
        ses.known_displaced = set()
        mujoco.mj_forward(sim.model, sim.data)
        if hasattr(sim.sensors, "reset_motion"):
            sim.sensors.reset_motion()
        ses.touches.reset_gap()
        ses.watch.reset()
        ses.overtakes.forget()
        ses.relocated = True
        ses.log.append((t, "relocated", info))

    # ------------------------------------------------------------------ records
    def truth_state(self):
        """Privileged state at the end of the last tick."""
        sim, ses = self.sim, self.ses
        if sim is None:
            return {"session": None, "tick": 0, "stopped": False,
                    "car": {"pose": list(self.world["truth"]["start_pose"]["pose"]), "vel_mps": [0.0, 0.0], "yaw_rate_radps": 0.0},
                    "boxes": [], "slow_cars": [], "moved_pieces": []}
        d = sim.data
        x, y, yaw = sim.car_pose()
        jid = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_JOINT, K.EGO_PREFIX + "root")
        va = int(sim.model.jnt_dofadr[jid])
        vx, vy = float(d.qvel[va]), float(d.qvel[va + 1])
        c, s = math.cos(yaw), math.sin(yaw)
        boxes = []
        if sim.boxes is not None:
            for bid, p in zip([b["id"] for b in self.world["truth"]["avoidance"]["boxes"]], sim.boxes.poses()):
                boxes.append({"id": bid, "pose": [float(v) for v in p]})
        slow = sim.slow.states() if sim.slow is not None else []
        moved = []
        wp = [(dd["id"], k, p) for dd in self.world["truth"]["ducts"] for k, p in enumerate(dd["pieces"])]
        for (did, k, p0), p in zip(wp, sim.ducts.poses()):
            if math.hypot(p[0] - p0[0], p[1] - p0[1]) > 1e-3:
                moved.append([did, k, float(p[0]), float(p[1]), float(p[2])])
        return {"session": ses.name if ses is not None else None, "tick": ses.tick if ses is not None else 0,
                "stopped": bool(ses.stopped) if ses is not None else False,
                "car": {"pose": [x, y, yaw], "vel_mps": [c * vx + s * vy, -s * vx + c * vy],
                        "yaw_rate_radps": float(d.qvel[va + 5])},
                "boxes": boxes, "slow_cars": slow, "moved_pieces": moved}

    def result(self):
        """The venue result record."""
        sessions = [s.record() for s in self.sessions]
        ducts = (self.sims.get("avoid") or self.sims.get("track"))
        res = {"schema": K.RESULT_SCHEMA, "world_id": self.world["world_id"], "layout_digest": self.world["layout_digest"],
               "config_sha256": C.config_sha256(self.cfg),
               "end_reason": self.end_reason or ("finish" if self.over else "eof"), "step_called": self.step_called,
               "sessions": sessions, "time_trial": None, "avoid": None,
               "stats": {"touches": sum(s["touches"] for s in sessions), "crashes": sum(s["crashes"] for s in sessions),
                         "wall_displacement_m": round(float(ducts.ducts.total_displacement()), 6) if ducts else 0.0,
                         "laps_by_session": {s["name"]: len(s["laps"]) for s in sessions},
                         "program_s": round(self.program_total_s, 6), "warnings": sum(s["warnings"] for s in sessions),
                         "forbidden_calls": self.counts["forbidden_calls"], "bad_calls": self.counts["bad_calls"]}}
        res.update(check.summarise_laps(res, self.cfg))
        return res
