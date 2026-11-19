"""The reference program's day of a head-to-head venue: privileged planning, public driving (numpy only).

    ref = RaceReference(car, params, race_params, token)    # unlocks the privileged calls, reads the world as built
    stats = ref.run()                                        # every session of the venue, in the order of describe()

What it knows that an entrant does not: the world file (centreline, ducts as built, open-wall tapes, the grip map, the
lines and paces of every heat's opponents), and after every step the true pose and velocity of its own car, the true
states of the opponents and the moved duct pieces (`truth` in the step reply). What it does with it:

- desk time: lateral limits of the reference point from the duct capsules and tapes, the minimum-curvature line inside
  them, quasi-steady-state speed profiles at factors of the grip-limited speed (the drive model's limits);
- practice: one timed lap at the mapping phase's limit (less 10 %), then it stops and closes the session;
- each heat: from its grid slot it joins its racing line at `race_factor` and races the three opponents of that heat
  with the racecraft module (following, passing, merging back). The heats' racing line and the racecraft's lateral
  limits keep `race_wall_clear_m` from the duct capsules and tapes (more than the planner's own clearance): after a
  pass or behind a slow car the tracking error is larger than on a lap of its own. It never reads the heat's lap target to decide how to
  drive; it drives until the referee ends the session.

Every session follows the referee's rules like any program: a `stopped` event holds the car until `released`; after a
release the car joins its line again on a join path.

The role of each session comes from its name (`practice` is practice, every scored session is a heat); a caller may
give its own table (`roles`: name -> "practice", "race" or "skip") and its own reader of a heat's opponents
(`field_of(world, name)` -> list of opponent records).
"""

import math
import sys
import time

import numpy as np

from .control import Tracker
from .geometry import Walls, car_envelope
from .planner import Line, Planner
from .racecraft import Racecraft

REF_X_M = 0.12892


def field_from_world(world, name):
    """The opponents of heat `name` in the world file (its `truth.head_to_head.heats`), in grid-slot order."""
    for h in world["truth"]["head_to_head"]["heats"]:
        if h["session"] == name:
            return sorted(h["field"], key=lambda c: int(c["slot"]))
    raise KeyError("no opponents for session %r" % name)


class RaceReference(object):
    def __init__(self, car, params, race_params, token, roles=None, field_of=None, log=None):
        self.car = car
        self.P = params
        self.R = race_params
        self.log = log or (lambda msg: print(msg, file=sys.stderr))
        car.call("priv_unlock", token=token)
        self.world = car.call("priv_world")
        self.info = car.describe()
        self.plan = Planner(self.world, params)
        self.tracker = Tracker(params)
        self.roles = dict(roles or {})
        self.field_of = field_of or field_from_world
        self.stats = {"sessions": {}, "plan": {}}
        self.moved_seen = []
        self.lines = {}
        self.profiles = {}
        self.trace = None
        self.cur_bounds = None
        self.cur_profile = None

    # ------------------------------------------------------------------ planning
    def plan_track(self, moved=None):
        t0 = time.perf_counter()
        lo, hi, info = self.plan.bounds(moved=moved)
        self.track_bounds = (lo, hi)
        line = self.plan.raceline(lo, hi)
        sp = self.R["strategy"]
        self.lines["track"] = line
        self.profiles["practice"] = self.plan.profile(line, float(sp["practice_factor"]), cap_mps=sp.get("practice_cap_mps"))
        rlo, rhi, _rinfo = self.plan.bounds(moved=moved, margin_m=float(sp["race_wall_clear_m"]))
        self.race_bounds = (rlo, rhi)
        rline = self.plan.raceline(rlo, rhi)
        self.lines["race"] = rline
        self.profiles["race"] = self.plan.profile(rline, float(sp["race_factor"]))
        self.diag_walls = Walls.from_world(self.world, moved=moved, tapes=True)
        self.stats["plan"]["track"] = {"tight_stations": info["tight"], "plan_s": round(time.perf_counter() - t0, 3),
                                       "lap_model_s": {k: round(p.lap_time_s, 4) for k, p in self.profiles.items()},
                                       "kappa_max_per_m": round(line.kappa_max, 4), "kappa_cap_met": line.kappa_cap_met}

    def moved_now(self):
        st = self.car.call("priv_state")
        return st.get("moved_pieces") or []

    # ------------------------------------------------------------------ the venue
    def role(self, ses):
        if ses["name"] in self.roles:
            return self.roles[ses["name"]]
        if ses["name"] == "practice" or not ses.get("scored", True):
            return "practice"
        return "race"

    def run(self):
        self.plan_track()
        first = True
        for ses in self.info["sessions"]:
            if not first:
                moved = self.moved_now()
                if moved:
                    self.plan_track(moved=moved)
                    self.moved_seen.append((ses["name"], len(moved)))
            first = False
            role = self.role(ses)
            if role == "skip":
                self.skip_session(ses["name"])
            elif role == "practice":
                self.drive_session(ses["name"], "practice")
            else:
                self.drive_session(ses["name"], "race")
        self.stats["moved_seen"] = self.moved_seen
        return self.stats

    def skip_session(self, name):
        self.car.open_session(name)
        self._close()
        self.stats["sessions"][name] = {"skipped": True}

    def _close(self):
        try:
            self.car.close_session()
        except Exception as exc:  # noqa: BLE001 - after the last session the venue is over (venue_over)
            if getattr(exc, "code", None) != "venue_over":
                raise

    def join(self, truth, line, factor):
        """A path from the car's present pose onto `line`: the line's offsets plus the car's lateral offset from it,
        faded out by a smoothstep over a distance that grows with that offset; its speed profile starts from the
        car's present speed. Returns (path, profile, station where the main line takes over, station of the car)."""
        jp = self.P["join"]
        plan = self.plan
        x, y, psi = (float(v) for v in truth["car"]["pose"])
        rx, ry = x + REF_X_M * math.cos(psi), y + REF_X_M * math.sin(psi)
        s0, d0, _i0 = plan.loop.project(rx, ry)
        n_main = line.offsets
        delta = d0 - plan.loop.interp(n_main, s0)
        L = min(float(jp["length_max_m"]), max(float(jp["length_min_m"]), float(jp["length_per_offset"]) * abs(delta)))
        Lb = float(jp["behind_m"])
        ahead = (plan.loop.s - s0) % plan.loop.length
        behind = (s0 - plan.loop.s) % plan.loop.length
        u = ahead / L
        ub = 1.0 - behind / Lb
        w = np.where(u < 1.0, 1.0 - u * u * (3.0 - 2.0 * u), 0.0)
        wb = np.where((behind < Lb) & (behind > 0.0), ub * ub * (3.0 - 2.0 * ub), 0.0)
        w = np.maximum(w, wb)
        raw = n_main + delta * w
        offs = np.clip(raw, self.cur_bounds[0], self.cur_bounds[1])
        near = w > 0.5
        offs[near] = raw[near]
        path = Line(plan, offs)
        v_now = max(0.0, float(truth["car"]["vel_mps"][0]))
        sj, _dj, ij = path.loop.project(rx, ry)
        prof = plan.profile(path, factor, start_index=ij, start_speed=v_now)
        main = self.cur_profile
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

    def rise_limit(self, truth, path_loop, kappa, s_hint, mu_here, factor):
        """How fast the speed target may rise: the profile's own traction capacity c_trac f^2 mu g inside the friction
        ellipse of the present lateral load (exponent of the drive model's limits), at least a tenth of it. Below its
        profile in a bend the car then accelerates as the profile would, not at full traction (which slides it)."""
        lim = self.plan.limits
        g_mu = 9.81 * float(mu_here)
        a_full = lim.c_trac * factor ** 2 * g_mu
        if s_hint is None:
            return a_full
        v = max(0.0, float(truth["car"]["vel_mps"][0]))
        lat = min(1.0, v * v * abs(float(path_loop.interp(kappa, s_hint))) / max(lim.c_lat * factor ** 2 * g_mu, 1e-6))
        p = float(lim.acc_exp)
        return a_full * max(0.1, max(0.0, 1.0 - lat ** p) ** (1.0 / p))

    @staticmethod
    def _line_station(line, s_plan):
        j = int(np.searchsorted(line.s_plan_sorted, s_plan)) % line.loop.n
        return float(line.loop.s[line.s_plan_order[j]])

    def drive_session(self, name, key):
        car = self.car
        sp = self.R["strategy"]
        opened = car.open_session(name)
        st = {"ticks": 0, "laps": [], "stops": 0, "max_abs_e_m": 0.0, "min_margin_m": 9.0, "join_ticks": 0,
              "phase_events": [], "cap_ticks": 0}
        self.tracker.reset()
        line = self.lines["race"] if key == "race" else self.lines["track"]
        prof = self.profiles[key]
        self.cur_bounds = self.race_bounds if key == "race" else self.track_bounds
        self.cur_profile = prof
        race = None
        if key == "race":
            race = Racecraft(self.plan, line, prof, self.race_bounds, self.field_of(self.world, name), self.P, self.R)
        stopped = False
        finishing = False
        laps_done = 0
        steer, speed = 0.0, 0.0
        cap_finish = None
        n_ticks = int(opened["ticks"])
        practice_laps = int(sp["practice_timed_laps"])
        cap_session = sp.get("practice_cap_mps") if key == "practice" else None
        truth = car.call("priv_state")
        joining = self.join(truth, line, prof.factor)
        had_path = False
        t_s = 0.0
        tick_s = 0.025
        for _k in range(n_ticks + 5):
            rep = car.step(steer, speed)
            st["ticks"] += 1
            truth = rep["truth"]
            t_s = int(rep["tick"]) * tick_s
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
                if race is not None:
                    race.reset()
                joining = self.join(truth, line, prof.factor)
            if key == "practice" and laps_done >= practice_laps:
                finishing = True
            cap = cap_session
            if finishing:
                if cap_finish is None:
                    cap_finish = max(0.0, float(truth["car"]["vel_mps"][0]))
                cap_finish = max(0.0, cap_finish - float(sp["stop_decel_mps2"]) * tick_s)
                cap = cap_finish
            if joining is not None:
                jpath, jprof, s_take, s_start = joining
                path_loop, kappa, speed_fn, mu_arr = jpath.loop, jpath.loop.kappa, jprof.speed, jprof.mu
                path_id = "join%d" % st["stops"]
                base_line, base_prof = jpath, jprof
                st["join_ticks"] += 1
            else:
                path_loop, kappa, speed_fn, path_id, mu_arr = line.loop, line.loop.kappa, prof.speed, "line", prof.mu
                base_line, base_prof = line, prof
            if race is not None:
                rcap, path = race.update(truth, int(rep["tick"]), t_s, base_line, base_prof)
                if path is not None:
                    if joining is not None:
                        joining = None
                    path_loop, kappa, speed_fn, mu_arr = path.loop, path.loop.kappa, path.speed, path.prof.mu
                    path_id = path.ident
                    had_path = True
                elif had_path:
                    had_path = False
                    # the racecraft has no path for the car any more: if the car is off its line, join it again
                    x, y, psi = (float(v) for v in truth["car"]["pose"])
                    rx, ry = x + REF_X_M * math.cos(psi), y + REF_X_M * math.sin(psi)
                    s0, d0, _i0 = self.plan.loop.project(rx, ry)
                    if joining is None and abs(d0 - self.plan.loop.interp(line.offsets, s0)) > float(sp["rejoin_dev_m"]):
                        joining = self.join(truth, line, prof.factor)
                        jpath, jprof, s_take, s_start = joining
                        path_loop, kappa, speed_fn, mu_arr = jpath.loop, jpath.loop.kappa, jprof.speed, jprof.mu
                        path_id = "rejoin%d" % rep["tick"]
                        st["rejoins"] = st.get("rejoins", 0) + 1
                if rcap is not None:
                    st["cap_ticks"] += 1
                    cap = rcap if cap is None else min(cap, rcap)
            self.tracker.observe(rep["steer_echo"], truth)
            s_hint = self.tracker.last.get("s")
            same = s_hint is not None and self.tracker.path_id == path_id
            mu_here = path_loop.interp(mu_arr, s_hint) if same else float(np.min(mu_arr))
            a_up = self.rise_limit(truth, path_loop, kappa, s_hint if same else None, mu_here, prof.factor)
            steer, speed, info = self.tracker.command(truth, path_loop, kappa, speed_fn, cap=cap, path_id=path_id,
                                                      mu_here=mu_here, a_up=a_up)
            if joining is not None:
                done = path_loop.gap(info["s"], s_take)
                if done >= 0.0 and path_loop.gap(info["s"], s_start) > 0.5:
                    joining = None
            ae = abs(info["e"])
            if joining is None and ae > st["max_abs_e_m"]:
                st["max_abs_e_m"] = ae
            m = self.diag_walls.clearance(car_envelope(truth["car"]["pose"]), reach=0.5)
            if self.trace is not None:
                self.trace.append((name, rep["tick"], round(info["s"], 3), round(info["e"], 4), round(info["v"], 3),
                                   round(info["vt"], 3), round(m, 4), path_id, None if cap is None else round(cap, 3)))
            if m < st["min_margin_m"]:
                st["min_margin_m"] = m
                st["min_margin_at"] = [round(info["s"], 2), round(info["v"], 3), rep["tick"], path_id]
            if finishing and abs(float(truth["car"]["vel_mps"][0])) < 0.03:
                break
        self._close()
        st["max_abs_e_m"] = round(st["max_abs_e_m"], 4)
        st["min_margin_m"] = round(st["min_margin_m"], 4)
        if race is not None:
            st["racecraft"] = race.summary()
        st["brake_limited_ticks"] = self.tracker.n_brake_limited
        self.tracker.n_brake_limited = 0
        self.stats["sessions"][name] = st
