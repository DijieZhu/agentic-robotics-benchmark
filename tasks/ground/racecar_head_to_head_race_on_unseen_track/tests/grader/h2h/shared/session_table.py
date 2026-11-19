"""A venue cell whose day is read from its configuration.

The shared cell (`sketch.referee.venue.Venue`) takes the order of its sessions from a module constant and builds the
model with obstacles for one session name. `ConfiguredVenue` takes the order, the model variant of each session and
the end of the day from the configuration it is given, and changes nothing else: with the shared cell's own
configuration it answers every request with the same bytes as the shared cell.

    venue = ConfiguredVenue(world, config, providers=prov, priv_token_sha256=sha)
    venue.session_names                 # ("practice", ...) in the configuration's order
    venue.variant_for(name)             # the model variant of a session (VARIANTS, default "track")

Hooks for a subclass: `SESSION_CLASS` (the referee's state of one open session), `VARIANTS`, `variant_for`,
`check_open_params`. At most `MAX_SESSIONS` sessions: the shared tick ends the day by itself after the fourth.
"""

from sketch.referee import venue as shared_venue

EPS = shared_venue.EPS
DETAILS = shared_venue.DETAILS
_err = shared_venue._err


class ConfiguredVenue(shared_venue.Venue):
    SESSION_CLASS = shared_venue._Session
    VARIANTS = {"avoid": "avoid"}        # session name -> model variant; every other session is "track"
    MAX_SESSIONS = 4

    def __init__(self, world, config=None, **kw):
        super().__init__(world, config, **kw)
        names = tuple(s["name"] for s in self.cfg["sessions"])
        if not 1 <= len(names) <= self.MAX_SESSIONS or len(set(names)) != len(names):
            raise ValueError("the configuration needs 1 to %d sessions with distinct names" % self.MAX_SESSIONS)
        self.session_names = names

    def variant_for(self, name):
        return self.VARIANTS.get(name, "track")

    # ------------------------------------------------------------------ the dispatcher
    def check_open_params(self, params):
        """None when `params` suit `open_session`; else the detail text of `bad_params` (the wording of the shared
        protocol rule, with this configuration's session names)."""
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return "params must be an object"
        if set(params) != {"name"}:
            return "open_session needs exactly the parameter `name`"
        if params["name"] not in self.session_names:
            return "the session names are %s" % ", ".join(self.session_names)
        return None

    def handle(self, method, params=None):
        if method != "open_session":
            return super().handle(method, params)
        if self.over:
            return _err("venue_over", DETAILS["venue_over"])
        detail = self.check_open_params(params)
        if detail:
            self.counts["bad_calls"] += 1
            return _err("bad_params", detail)
        return self.open_session(params["name"])

    # ------------------------------------------------------------------ budgets
    def account(self, dt):
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
        elif self.next_idx < len(self.session_names):
            name = self.session_names[self.next_idx]
            self._desk[name] = self._desk.get(name, 0.0) + dt
            if self._desk[name] > self.caps["desk"][name] + EPS:
                self.end("desk_time_cap")
                return "desk_time_cap"
        return None

    def budget_left(self):
        if self.over:
            return float("inf")
        if self.state == "open":
            return self.caps["track"][self.ses.name] - self.ses.program_s
        if self.next_idx < len(self.session_names):
            name = self.session_names[self.next_idx]
            return self.caps["desk"][name] - self._desk.get(name, 0.0)
        return float("inf")

    # ------------------------------------------------------------------ sessions
    def open_session(self, name):
        if self.over:
            return _err("venue_over", DETAILS["venue_over"])
        if self.state == "open":
            return _err("bad_order", DETAILS["bad_order_open"])
        if self.next_idx >= len(self.session_names) or name != self.session_names[self.next_idx]:
            return _err("bad_order", "the next session is %s" % self.session_names[self.next_idx])
        if self.state == "ended":
            self._close_current()
        ses = self.SESSION_CLASS(self, self.next_idx)
        ses.desk_s = self._desk.get(name, 0.0)
        self.next_idx += 1
        self._enter_variant(self.variant_for(name))
        sim = self.sim
        # the staff repair what the last session left displaced by the crash threshold or more; the rest stays
        self._put_back(sim)
        sim.car.place(self.world["truth"]["start_pose"]["pose"])
        sim.car.set_targets(0.0, 0.0)
        if hasattr(sim.sensors, "reset_motion"):
            sim.sensors.reset_motion()
        shared_venue.mujoco.mj_forward(sim.model, sim.data)
        ses.ref_prev = sim.ref_xy()
        s, d = self.track.project1(*ses.ref_prev)[:2]
        ses.clean = (s, d)
        self.ses = ses
        self.sessions.append(ses)
        self.state = "open"
        return {"session": name, "duration_s": ses.cfg["duration_s"], "ticks": ses.ticks_total,
                "phases": [dict(p) for p in ses.phases]}

    def _close_current(self):
        self.state = "between"
        if self.next_idx >= len(self.session_names):
            self.state = "over"
            if self.end_reason is None:
                self.end_reason = "finish"

    def _tick(self, steer, speed):
        out = super()._tick(steer, speed)
        # the shared tick ends the day after its fourth session; a shorter day ends after its own last one
        if self.state == "ended" and self.next_idx >= len(self.session_names):
            self.state = "over"
            self.last_close_pending = True
            if self.end_reason is None:
                self.end_reason = "finish"
        return out


__all__ = ["ConfiguredVenue"]
