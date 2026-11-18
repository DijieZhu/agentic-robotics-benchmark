"""The reference program's sessions: privileged planning, public driving (numpy only).

    ref = Reference(car, params, token)      # unlocks the privileged calls, reads the world as built
    ref.run()                                # practice, tt1, tt2, avoid: every command goes through car.step

What it knows that a team does not: the world file (centreline, ducts as built, open-wall tapes, the grip
map, boxes, slow-car paths), and after every step the true pose, velocity, box and slow-car states and moved duct
pieces (`truth` in the step reply). What it does with it:

- desk time: lateral limits of the reference point from the duct capsules, tapes (and boxes), the minimum-curvature
  line inside them, quasi-steady-state speed profiles at factors of the grip-limited speed (the drive model's limits);
- practice: one timed lap at the mapping phase's limit (less 10 %), then it stops and closes the session;
- tt1 at `heat1_factor` (the steady heat: consecutive laps), tt2 at `heat2_factor` (the fast heat);
- avoid: a line that passes each box on one side; in the dynamic phase the traffic planner (traffic.py) passes the
  slow cars where the track is wide enough and follows them elsewhere.

Every session follows the referee's rules like any program: a `stopped` event holds the car until `released`.
At the start of a session and after a release the car joins its line on a join path (see `join`).
"""

import math
import sys
import time

import numpy as np

from .control import Tracker
from .geometry import Walls, car_envelope
from .planner import Line, Planner
from .traffic import Traffic


class Reference(object):
    def __init__(self, car, params, token, log=None):
        self.car = car
        self.P = params
        self.log = log or (lambda msg: print(msg, file=sys.stderr))
        car.call("priv_unlock", token=token)
        self.world = car.call("priv_world")
        self.info = car.describe()
        self.plan = Planner(self.world, params)
        self.tracker = Tracker(params)
        self.stats = {"sessions": {}, "plan": {}}
        self.moved_seen = []
        self.lines = {}
        self.profiles = {}
        self.cur_line = None
        self.trace = None

    # ------------------------------------------------------------------ planning
    def plan_track(self, moved=None):
        t0 = time.perf_counter()
        lo, hi, info = self.plan.bounds(moved=moved)
        self.track_bounds = (lo, hi)
        line = self.plan.raceline(lo, hi)
        sp = self.P["strategy"]
        profs = {}
        for key in ("practice", "heat1", "heat2"):
            f = float(sp[key + "_factor"])
            cap = sp.get(key + "_cap_mps")
            profs[key] = self.plan.profile(line, f, cap_mps=cap)
        self.lines["track"] = line
        self.profiles.update(profs)
        self.diag_walls = Walls.from_world(self.world, moved=moved, tapes=True)
        self.stats["plan"]["track"] = {"tight_stations": info["tight"], "plan_s": round(time.perf_counter() - t0, 3),
                                       "lap_model_s": {k: round(p.lap_time_s, 4) for k, p in profs.items()},
                                       "min_free_left_m": round(float(info["free_left"].min()), 4),
                                       "min_free_right_m": round(float(info["free_right"].min()), 4),
                                       **self._kappa_stats(line)}
        self.free = (info["free_left"], info["free_right"])

    def plan_avoid(self, moved=None):
        t0 = time.perf_counter()
        boxes = [(b["pose"], b["size_m"]) for b in self.world["truth"]["avoidance"]["boxes"]]
        lo, hi, info = self.plan.bounds(moved=moved, boxes=boxes)
        line = self.plan.raceline(lo, hi)
        f = float(self.P["strategy"]["avoid_factor"])
        prof = self.plan.profile(line, f)
        self.lines["avoid"] = line
        self.profiles["avoid"] = prof
        self.avoid_bounds = (lo, hi)
        self.free_avoid = (info["free_left"], info["free_right"])
        self.diag_walls = Walls.from_world(self.world, moved=moved, tapes=True, boxes=boxes, box_edges=True)
        self.stats["plan"]["avoid"] = {"tight_stations": info["tight"], "plan_s": round(time.perf_counter() - t0, 3),
                                       "lap_model_s": round(prof.lap_time_s, 4), "box_sides": info["box_side"],
                                       **self._kappa_stats(line)}

    @staticmethod
    def _kappa_stats(line):
        """The line's curvature against its cap."""
        return {"kappa_max_per_m": round(line.kappa_max, 4),
                "kappa_cap_per_m": None if line.kappa_cap is None else round(line.kappa_cap, 4),
                "kappa_cap_met": line.kappa_cap_met, "kappa_cap_resolves": line.kappa_cap_resolves}

    def moved_now(self):
        st = self.car.call("priv_state")
        return st.get("moved_pieces") or []

    # ------------------------------------------------------------------ the venue
    def run(self):
        self.plan_track()
        self.drive_session("practice", "practice")
        moved = self.moved_now()
        if moved:
            self.plan_track(moved=moved)
            self.moved_seen.append(("tt1", len(moved)))
        self.drive_session("tt1", "heat1")
        moved = self.moved_now()
        if moved:
            self.plan_track(moved=moved)
            self.moved_seen.append(("tt2", len(moved)))
        self.drive_session("tt2", "heat2")
        self.plan_avoid(moved=self.moved_now() or None)
        self.drive_session("avoid", "avoid")
        self.stats["moved_seen"] = self.moved_seen
        return self.stats

    def join(self, truth, line, factor):
        """A path from the car's present pose onto `line`: the line's offsets plus the car's lateral offset from
        it, faded out by a smoothstep over a distance that grows with that offset; its speed profile starts from
        the car's present speed (forward pass from there), so the car accelerates along the line's limits instead
        of catching up. Returns (path, profile, station on the path from which the main line takes over)."""
        jp = self.P["join"]
        plan = self.plan
        x, y, psi = (float(v) for v in truth["car"]["pose"])
        rx, ry = x + 0.12892 * math.cos(psi), y + 0.12892 * math.sin(psi)
        s0, d0, i0 = plan.loop.project(rx, ry)
        n_main = line.offsets
        delta = d0 - plan.loop.interp(n_main, s0)
        L = min(float(jp["length_max_m"]), max(float(jp["length_min_m"]), float(jp["length_per_offset"]) * abs(delta)))
        Lb = float(jp["behind_m"])
        ahead = (plan.loop.s - s0) % plan.loop.length               # 0 .. track length
        behind = (s0 - plan.loop.s) % plan.loop.length
        u = ahead / L
        ub = 1.0 - behind / Lb
        w = np.where(u < 1.0, 1.0 - u * u * (3.0 - 2.0 * u), 0.0)
        wb = np.where((behind < Lb) & (behind > 0.0), ub * ub * (3.0 - 2.0 * ub), 0.0)
        w = np.maximum(w, wb)
        raw = n_main + delta * w
        offs = np.clip(raw, self.cur_bounds[0], self.cur_bounds[1])
        near = w > 0.5                                               # near the car: where it is, even off the limits
        offs[near] = raw[near]
        path = Line(plan, offs)
        v_now = max(0.0, float(truth["car"]["vel_mps"][0]))
        sj, _dj, ij = path.loop.project(rx, ry)
        prof = plan.profile(path, factor, start_index=ij, start_speed=v_now)
        main = self.cur_profile
        # take-over: the first sample past the blend where the join profile has reached the main one
        n = path.loop.n
        k_end = int(math.ceil((L + 0.3) / path.loop.ds))
        k_max = min(n - 1, int(math.ceil(float(jp["takeover_max_m"]) / path.loop.ds)))
        s_take = None
        for k in range(k_end, max(k_end + 1, k_max)):
            j = (ij + k) % n
            sp = path.s_plan[j]
            vm = main.line.loop.interp(main.v, self._line_station(main.line, sp))
            if prof.v[j] >= vm - 0.02:
                s_take = float(path.loop.s[j])
                break
        if s_take is None:
            s_take = float(path.loop.s[(ij + k_max) % n])
        return path, prof, s_take, sj

    def _line_station(self, line, s_plan):
        """Station on `line` of the sample whose plan station is nearest `s_plan`."""
        j = int(np.searchsorted(line.s_plan_sorted, s_plan)) % line.loop.n
        return float(line.loop.s[line.s_plan_order[j]])

    def drive_session(self, name, key, limit_ticks=None):
        car = self.car
        sp = self.P["strategy"]
        opened = car.open_session(name)
        st = {"ticks": 0, "laps": [], "stops": 0, "max_abs_e_m": 0.0, "min_margin_m": 9.0, "join_ticks": 0,
              "phase_events": []}
        self.tracker.reset()
        traffic = None
        if key == "avoid":
            line, prof = self.lines["avoid"], self.profiles["avoid"]
            traffic = Traffic(self.world, self.plan, line, prof, self.avoid_bounds, self.P)
            self.cur_bounds = self.avoid_bounds
        else:
            line, prof = self.lines["track"], self.profiles[key]
            self.cur_bounds = self.track_bounds
        self.cur_profile = prof
        stopped = False
        joining = None
        finishing = False
        laps_done = 0
        steer, speed = 0.0, 0.0
        cap_finish = None
        n_ticks = int(opened["ticks"]) if limit_ticks is None else min(int(opened["ticks"]), int(limit_ticks))
        practice_laps = int(sp["practice_timed_laps"])
        cap_session = sp.get(key + "_cap_mps")
        truth = car.call("priv_state")
        joining = self.join(truth, line, prof.factor)
        for _k in range(n_ticks + (5 if limit_ticks is None else 0)):
            rep = car.step(steer, speed)
            st["ticks"] += 1
            truth = rep["truth"]
            end = False
            released = False
            for ev in rep["events"]:
                typ = ev["type"]
                if typ == "stopped":
                    stopped = True
                    st["stops"] += 1
                elif typ == "released":
                    stopped = False
                    released = True
                elif typ == "lap":
                    st["laps"].append(ev.get("lap_ms"))
                    if ev.get("lap_ms") is not None:
                        laps_done += 1
                elif typ == "phase":
                    st["phase_events"].append([ev["name"], rep["tick"]])
                elif typ == "session_end":
                    end = True
            if end:
                break
            if stopped:
                steer, speed = 0.0, 0.0
                continue
            if released:
                self.tracker.reset()
                if traffic is not None:
                    traffic.reset()
                joining = self.join(truth, line, prof.factor)
            if key == "practice" and laps_done >= practice_laps:
                finishing = True
            cap = cap_session
            if finishing:
                if cap_finish is None:
                    cap_finish = max(0.0, float(truth["car"]["vel_mps"][0]))
                cap_finish = max(0.0, cap_finish - float(sp["stop_decel_mps2"]) * 0.025)
                cap = cap_finish
            if joining is not None:
                jpath, jprof, s_take, s_start = joining
                path_loop, kappa, speed_fn, mu_arr = jpath.loop, jpath.loop.kappa, jprof.speed, jprof.mu
                path_id = "join%d" % st["stops"]
                st["join_ticks"] += 1
            else:
                path_loop, kappa, speed_fn, path_id, mu_arr = line.loop, line.loop.kappa, prof.speed, "line", prof.mu
            if traffic is not None and joining is None:
                tcap, path = traffic.update(truth, rep["tick"])
                if path is not None:
                    path_loop, kappa, speed_fn, mu_arr = path.loop, path.loop.kappa, path.speed, path.prof.mu
                    path_id = path.ident
                if tcap is not None:
                    cap = tcap if cap is None else min(cap, tcap)
            self.tracker.observe(rep["steer_echo"], truth)
            s_hint = self.tracker.last.get("s")
            same = s_hint is not None and self.tracker.path_id == path_id
            mu_here = path_loop.interp(mu_arr, s_hint) if same else float(np.min(mu_arr))
            a_up = self.plan.limits.c_trac * prof.factor ** 2 * mu_here * 9.81
            steer, speed, info = self.tracker.command(truth, path_loop, kappa, speed_fn, cap=cap, path_id=path_id,
                                                      mu_here=mu_here, a_up=a_up)
            if joining is not None:
                done = path_loop.gap(info["s"], s_take)
                if done >= 0.0 and path_loop.gap(info["s"], s_start) > 0.5:
                    joining = None
            ae = abs(info["e"])
            if joining is None and ae > st["max_abs_e_m"]:
                st["max_abs_e_m"] = ae
            m = self.margin(truth)
            if self.trace is not None:
                self.trace.append((name, rep["tick"], round(info["s"], 3), round(info["e"], 4), round(info["echi"], 4),
                                   round(info["v"], 3), round(info["vt"], 3), round(info["steer"], 4), round(m, 4),
                                   round(self.tracker.delta_act, 4), round(float(truth["car"]["yaw_rate_radps"]), 4),
                                   round(float(truth["car"]["vel_mps"][1]), 4), round(self.tracker.dk_us, 4),
                                   round(float(rep["wheel_speed"]), 3), 1 if joining is not None else 0, path_id))
            if m < st["min_margin_m"]:
                st["min_margin_m"] = m
                st["min_margin_at"] = [round(info["s"], 2), round(info["v"], 3), rep["tick"], path_id]
            if finishing and abs(float(truth["car"]["vel_mps"][0])) < 0.03:
                break
        try:
            car.close_session()
        except Exception as exc:  # noqa: BLE001 - after the last session the venue is over (venue_over)
            if getattr(exc, "code", None) != "venue_over":
                raise
        st["max_abs_e_m"] = round(st["max_abs_e_m"], 4)
        st["min_margin_m"] = round(st["min_margin_m"], 4)
        if traffic is not None:
            st["traffic"] = traffic.summary()
        st["brake_limited_ticks"] = self.tracker.n_brake_limited
        self.tracker.n_brake_limited = 0
        self.stats["sessions"][name] = st

    def margin(self, truth):
        """Smallest distance between the car's envelope (true pose) and the duct capsules, tapes and, in the
        avoidance test, the boxes' edges at their file poses (a diagnostic of the program; the referee judges)."""
        walls = self.diag_walls
        return walls.clearance(car_envelope(truth["car"]["pose"]), reach=0.5)
