"""The operating procedure's checks and the episode score (cell internals, run by the grader only).

The episode runs on the simulated clock: the startup until STARTUP_DEADLINE_S (600 s), the operating window until
SHIFT_END_S (1500 s), then the shutdown until the end of the episode (the sim cap, 1800 s, or `finish`). The score is
the sum of five parts (the operating procedure, sop.md, says what each step is):

  startup 0.20   line-up checked at the first start 0.05 (lost when T-101 overflows during the shift: v0.6), the pump
                 primed (vented) before it 0.04, a clean start (no
                 trip, no deadheading, still running 120 s later) 0.04, V-103 fully open within 240 s of the start and
                 before the window opens 0.04, P-301 dosing when the window opens 0.03
  operation 0.25 the fraction of the window with LG-201 in 55-75 % (0.125) and with AI-201 in 1.8-2.6 mg/L (0.125)
  events 0.25    split equally between the episode's events:
                 the consumer's demand step: 360 s after it, level in band and the transfer matched to the new demand
                   (within 0.35 m3/h, or within 1 m3/h on the side that brings the level back towards 65 %);
                 a debris slug on the strainer (it lands in the chamber in service when it comes): changed over from
                   that chamber to the other one within 240 s of the alarm 0.5, the chamber changed to free of air
                   (vented) when it came into service 0.3, the alarm reset after the changeover 0.2 (not judged if it
                   never raises the alarm);
                 T-101's float valve sticking shut: the make-up valve V-106 opened after the float valve stuck (closed
                   when it stuck) and before the pump trips on dry running, and no such trip until the shift ends (not
                   judged if T-101 never reaches its low alarm and V-106 was closed when it stuck; a V-106 already open
                   when the float valve stuck is not the procedure and earns nothing);
                 an air lock in the dosing pump: vented with the pump stopped and dosing again within 300 s, from the drum
                   lined up when it air-locked;
                 drum A running dry: V-303 turned to drum B once drum A was below 5 % (or dry), and dosing from drum B
                   within 300 s of drum A running dry;
                 a trip of P-101 (a supply dip): restarted within 180 s 0.5, against a closed discharge (V-103) 0.25,
                   dosing again within 300 s of the trip 0.25 (not judged if P-101 was not running)
  shutdown 0.15  after the shift ends: P-301 stopped before P-101 0.03, V-103 closed before P-101 stops 0.04, P-101
                 stopped 0.03, V-101 closed after it 0.03, HS-101 at OFF at the end 0.02
  log 0.15       two rounds (600-1050 s and 1050-1500 s, P-101 running): 0.0125 for each of PI-101, PI-103, PDI-111,
                 FI-105, LG-201 and AI-201 whose first reading logged in the round is within the tolerance of the
                 instrument's true value at that moment (1 % of the instrument's span; AI-201 0.05 mg/L)

A void (the pump deadheaded for 30 s, cavitation damage, T-201 overflowing, the emergency stop pressed, a safety
void of the robot: damage-level contact or more than five protective stops, server.py) makes the episode score 0.
v0.6: T-101 overflowing (to the drain) is no longer a void; it costs the start-up's line-up credit.
"""

import math

from .layout import CONC_BAND, LEVEL_BAND, SHIFT_END_S, STARTUP_DEADLINE_S
from .process import DRUM_L
from .textures import GAUGE_KINDS

__all__ = ["Monitor", "WEIGHTS", "LOG_TAGS", "UNITS", "tolerance", "log_round", "LOG_ROUNDS"]

WEIGHTS = {
    "startup": {"lineup": 0.05, "primed": 0.04, "clean_start": 0.04, "discharge_open": 0.04, "dosing_on": 0.03},
    "operation": {"level_in_band": 0.125, "conc_in_band": 0.125},
    "events": 0.25,
    "shutdown": {"dosing_stopped_first": 0.03, "discharge_closed_first": 0.04, "pump_stopped": 0.03, "suction_closed": 0.03,
                 "selector_off": 0.02},
    "log": 0.15,
}
LOG_TAGS = ("PI-101", "PI-103", "PDI-111", "FI-105", "LG-201", "AI-201")      # series tags; the log takes the site tags
# unit name -> (quantity, factor to the base unit: bar, m3/h, %, mg/L)
UNITS = {"bar": ("pressure", 1.0), "psi": ("pressure", 1.0 / 14.5037738), "kpa": ("pressure", 0.01),
         "m3/h": ("flow", 1.0), "m³/h": ("flow", 1.0), "l/min": ("flow", 0.06), "lpm": ("flow", 0.06),
         "%": ("level", 1.0), "mg/l": ("concentration", 1.0), "ppm": ("concentration", 1.0)}
QUANTITY = {"PI-101": "pressure", "PI-103": "pressure", "PDI-111": "pressure", "FI-105": "flow", "LG-201": "level",
            "AI-201": "concentration"}
DEMAND_EVAL_S = 360.0
LOG_ROUNDS = ((600.0, 1050.0), (1050.0, 1500.0))   # [lo, hi) and [lo, hi]: the log sheet's two rounds
LOG_TOL_FRAC = 0.01                  # of the instrument's span, a process gauge's accuracy class (v0.5; 3 % before)
SLUG_SWITCH_S = 240.0
AIRLOCK_FIX_S = 300.0
DEMAND_MATCH_M3H = 0.35
DEMAND_RECOVER_M3H = 1.0             # a transfer off the draw that is bringing the level back towards 65 % counts too
LEVEL_TARGET = 65.0
DRUM_FIX_S = 300.0
DRUM_LOW_FRAC = 0.05
TRIP_RESTART_S = 180.0
TRIP_DOSING_S = 300.0
LINEUP = {"V-101": (">=", 0.9), "V-103": ("<=", 0.05), "V-104": (">=", 0.9), "V-105": ("<=", 0.05), "V-106": ("<=", 0.05),
          "V-107": ("<=", 0.05), "V-111": ("<=", 0.1), "V-102": ("<=", 0.05), "V-112": ("<=", 0.05), "V-302": ("<=", 0.05),
          "V-303": ("<=", 0.1)}


def tolerance(tag, kinds):
    """The logging tolerance of `tag` in its base unit: 1 % of the instrument's span (AI-201: 0.05 mg/L)."""
    if tag == "AI-201":
        return 0.05
    if tag == "LG-201":
        return 100.0 * LOG_TOL_FRAC
    k = GAUGE_KINDS[kinds[tag]]
    return LOG_TOL_FRAC * (k["hi"] - k["lo"]) * k["scale"]


def log_round(t):
    """The log sheet's round of the clock time t: 1, 2 or None (outside the window)."""
    if LOG_ROUNDS[0][0] <= t < LOG_ROUNDS[0][1]:
        return 1
    if LOG_ROUNDS[1][0] <= t <= LOG_ROUNDS[1][1]:
        return 2
    return None


class Monitor(object):
    """Watches the process and the devices every tick and keeps what the score needs."""

    def __init__(self, layout):
        self.L = layout
        self.kinds = layout["gauge_kinds"]
        self.site_to_series = {v: k for k, v in layout["site"]["tags"].items()}
        self.events = [dict(e) for e in layout["events"]]
        self.first_start = None
        self.lineup = None
        self.primed = None
        self.trip_or_deadhead_early = False
        self.clean_start = None
        self.discharge_open = None
        self.dosing_on = None
        self.in_band = {"level": 0.0, "conc": 0.0}
        self.ev = []
        for e in self.events:
            self.ev.append({"type": e["type"], "t": float(e["t"]), "credit": 0.0, "detail": {}, "done": False})
        self.presses = []                     # (t, button)
        self.logs = []                        # accepted log calls (t, tag, value, unit, true, ok)
        self.first_log = {}                   # (round, series tag) -> the entry on the sheet
        self.stop101_after_shift = None
        self.stop301_after_shift = None
        self.prev_run101 = False
        self.prev_run301 = False
        self.t_last = 0.0
        self.end_state = None
        self.prev_drum = "A" if float(layout["initial"].get("V-303", 0.0)) < 0.5 * math.pi else "B"
        self.t101_overflow_t = None           # v0.6: T-101 ran over to the drain (the line-up credit is lost)
        self.drum_changes = []                # (t, from, to, drum A's level fraction at that moment)

    # ------------------------------------------------------------------ inputs
    def on_press(self, t, button):
        self.presses.append((float(t), button))

    def on_log(self, t, tag, value, unit, proc):
        """A log_reading call: returns the record (the program is told only that it was recorded)."""
        site_tag = str(tag).strip().upper()
        tag = self.site_to_series.get(site_tag)
        if tag not in LOG_TAGS:
            raise ValueError("%r is not on the log sheet (log one of %s)" % (site_tag, ", ".join(self.L["site"]["log_sheet"])))
        u = str(unit).strip().lower().replace(" ", "")
        if u not in UNITS:
            raise ValueError("unknown unit %r (one of bar, psi, kPa, m3/h, L/min, %%, mg/L)" % unit)
        v = float(value)
        if not math.isfinite(v):
            raise ValueError("non-finite value")
        q, f = UNITS[u]
        true = float(proc.values()[tag])
        ok = (q == QUANTITY[tag]) and abs(v * f - true) <= tolerance(tag, self.kinds)
        rnd = log_round(t)
        # the reply tells the program only what the clock says (53-isolation-2: whether P-101 runs stays internal)
        rec = {"t": float(t), "tag": site_tag, "series_tag": tag, "value": v, "unit": unit, "true_base": true, "ok": bool(ok),
               "round": rnd, "in_window": rnd is not None, "counts": rnd is not None and bool(proc.run101)}
        self.logs.append(rec)
        if rec["counts"] and (rnd, tag) not in self.first_log:
            self.first_log[(rnd, tag)] = rec
        return rec

    # ------------------------------------------------------------------ every tick
    def tick(self, t, dt, proc, openings, selector):
        u = openings
        # the first start of P-101
        if proc.run101 and not self.prev_run101 and self.first_start is None:
            self.first_start = t
            self.lineup = all((u[k] >= v) if op == ">=" else (u[k] <= v) for k, (op, v) in LINEUP.items())
            self.lineup_detail = {k: round(float(u[k]), 3) for k in LINEUP}
            self.primed = bool(proc.casing_air < 0.1)
        if self.first_start is not None and t <= self.first_start + 120.0:
            if proc.fault101 or proc.deadhead_s > 5.0:
                self.trip_or_deadhead_early = True
            if t >= self.first_start + 120.0 - 1e-9 and self.clean_start is None:
                self.clean_start = bool(proc.run101 and not self.trip_or_deadhead_early)
        if (self.first_start is not None and self.clean_start is None and t > self.first_start + 120.0):
            self.clean_start = bool(proc.run101 and not self.trip_or_deadhead_early)
        if self.first_start is not None and self.discharge_open is None and u["V-103"] >= 0.9 and proc.run101:
            self.discharge_open = bool(t <= self.first_start + 240.0 and t <= STARTUP_DEADLINE_S)
        if self.dosing_on is None and t >= STARTUP_DEADLINE_S:
            self.dosing_on = bool(proc.run301 and proc.q_dose > 0.0)
        # the window
        if STARTUP_DEADLINE_S <= t <= SHIFT_END_S:
            lev = proc.level_pct("T-201")
            if LEVEL_BAND[0] <= lev <= LEVEL_BAND[1]:
                self.in_band["level"] += dt
            if CONC_BAND[0] <= proc.c_disp <= CONC_BAND[1]:
                self.in_band["conc"] += dt
        drum = proc.selected_drum()
        if drum != self.prev_drum:
            self.drum_changes.append((t, self.prev_drum, drum, proc.drum["A"] / DRUM_L))
            self.prev_drum = drum
        self._events(t, proc, u)
        if self.t101_overflow_t is None and getattr(proc, "t101_overflow_t", None) is not None:
            self.t101_overflow_t = float(proc.t101_overflow_t)
        # the shutdown
        if t > SHIFT_END_S:
            if self.prev_run301 and not proc.run301 and proc.run101 and self.stop301_after_shift is None:
                self.stop301_after_shift = t
            if self.prev_run101 and not proc.run101 and self.stop101_after_shift is None and not proc.fault101:
                self.stop101_after_shift = t
                self.stop_state = {"V-103": float(u["V-103"]), "P-301_stopped_first": self.stop301_after_shift is not None}
        self.prev_run101 = proc.run101
        self.prev_run301 = proc.run301
        self.t_last = t
        self.last_u = dict(u)
        self.last_selector = selector

    def _events(self, t, proc, u):
        for e in self.ev:
            if e["done"] or t < e["t"]:
                continue
            d = e["detail"]
            if e["type"] == "demand_step":
                t_eval = e["t"] + DEMAND_EVAL_S       # the generator draws the step early enough (layout.py)
                if t >= t_eval:
                    lev = proc.level_pct("T-201")
                    gap = proc.Qm - proc.demand
                    matched = abs(gap) <= DEMAND_MATCH_M3H or (gap * (LEVEL_TARGET - lev) > 0.0 and abs(gap) <= DEMAND_RECOVER_M3H)
                    ok = LEVEL_BAND[0] <= lev <= LEVEL_BAND[1] and matched
                    e["credit"] = 1.0 if ok else 0.0
                    d.update({"level_pct": round(lev, 1), "transfer_m3h": round(proc.Qm, 3), "demand_m3h": round(proc.demand, 3)})
                    e["done"] = True
            elif e["type"] == "filter_slug":
                # 53-scoring-2 / -3: the slug lands in the chamber in service (process.py); the changeover that counts
                # takes the lever from that chamber's end to the other end after the slug came, and "vented" is the
                # air the incoming chamber held when it came into service, before it went into the pump
                if "chamber" not in d:
                    if proc.slug_chamber is None:
                        continue                      # the process has not taken the slug in yet (same tick)
                    d["chamber"] = proc.slug_chamber
                    d["n_changeovers_before"] = len(proc.changeovers)
                if "alarm_t" not in d and proc.alarms["PDI-111"]:
                    d["alarm_t"] = t
                other = "B" if d["chamber"] == "A" else "A"
                at_other_end = (u["V-111"] >= 0.9 * math.pi) if other == "B" else (u["V-111"] <= 0.1 * math.pi)
                if "switch_t" not in d and at_other_end:
                    came = [c for c in proc.changeovers[d["n_changeovers_before"]:] if c[2] == other]
                    if came:
                        d["switch_t"] = t
                        d["vented_before_switch"] = bool(came[-1][3] < 0.1)
                if "switch_t" in d and "reset_t" not in d:
                    if "alarm_t" not in d or any(b == "RESET" and tp >= d["switch_t"] for tp, b in self.presses):
                        d["reset_t"] = t
                if "alarm_t" not in d and "switch_t" not in d:
                    if t >= SHIFT_END_S:
                        e["excluded"] = True          # the slug never raised the alarm in the window: nothing to judge
                        e["done"] = True
                    continue
                deadline = min(d.get("alarm_t", t) + SLUG_SWITCH_S, SHIFT_END_S)
                if "switch_t" in d and "alarm_t" not in d:
                    deadline = SHIFT_END_S             # changed over before the alarm: on time
                if t >= deadline or ("switch_t" in d and "reset_t" in d):
                    on_time = "switch_t" in d and d["switch_t"] <= deadline
                    e["credit"] = (0.5 if on_time else 0.0) + (0.3 if on_time and d.get("vented_before_switch") else 0.0) + \
                                  (0.2 if on_time and "reset_t" in d else 0.0)
                    if t >= deadline or e["credit"] >= 1.0:
                        e["done"] = True
            elif e["type"] == "dosing_airlock":
                deadline = min(e["t"] + AIRLOCK_FIX_S, SHIFT_END_S)
                if "drum_at_event" not in d:
                    d["drum_at_event"] = proc.selected_drum()
                fixed = proc.airlock <= 0.0 and proc.run301 and proc.q_dose > 0.0 and proc.selected_drum() == d["drum_at_event"]
                if fixed and "fixed_t" not in d:
                    d["fixed_t"] = t
                if fixed or t >= deadline:
                    e["credit"] = 1.0 if fixed and t <= deadline else 0.0
                    e["done"] = True
            elif e["type"] == "drum_empty":
                deadline = min(e["t"] + DRUM_FIX_S, SHIFT_END_S)
                to_b = [c for c in self.drum_changes if c[2] == "B"]          # the changeover that lined drum B up
                in_time = bool(to_b) and to_b[-1][3] < DRUM_LOW_FRAC
                d["changeover_level"] = round(to_b[-1][3], 3) if to_b else None
                fixed = in_time and proc.selected_drum() == "B" and proc.airlock <= 0.0 and proc.run301 and proc.q_dose > 0.0
                if fixed and "fixed_t" not in d:
                    d["fixed_t"] = t
                if fixed or t >= deadline:
                    e["credit"] = 1.0 if fixed and t <= deadline else 0.0
                    e["done"] = True
            elif e["type"] == "supply_loss":
                # 53-scoring-4: the make-up counts only when opened after the float valve stuck (sop.md AOP-5: never
                # open V-106 while the float valve works)
                if "open_at_fault" not in d:
                    d["open_at_fault"] = bool(u["V-106"] > 0.05)
                if "alarm_t" not in d and proc.alarms["LG-101"]:
                    d["alarm_t"] = t
                if "opened_t" not in d and u["V-106"] >= 0.15 and not d["open_at_fault"]:
                    d["opened_t"] = t
                if proc.fault101 and proc.fault_reason == "dry running" and "dry_trip_t" not in d:
                    d["dry_trip_t"] = t
                if t >= SHIFT_END_S:
                    if "alarm_t" not in d and "opened_t" not in d and not d["open_at_fault"]:
                        e["excluded"] = True          # T-101 never reached its low alarm: nothing to judge
                    else:
                        ok = "opened_t" in d and "dry_trip_t" not in d
                        e["credit"] = 1.0 if ok else 0.0
                    e["done"] = True
            elif e["type"] == "pump_trip":
                if "tripped" not in d:
                    d["tripped"] = bool(proc.fault101 and (proc.fault_reason or "").startswith("motor protection: supply dip"))
                    if not d["tripped"]:
                        e["excluded"] = True          # P-101 was not running: nothing tripped
                        e["done"] = True
                        continue
                if "restart_t" not in d and proc.run101:
                    d["restart_t"] = t
                    d["discharge_closed_at_restart"] = bool(u["V-103"] <= 0.05)
                if "restart_t" in d and "dosing_t" not in d and proc.run301 and proc.q_dose > 0.0:
                    d["dosing_t"] = t
                end = min(e["t"] + TRIP_DOSING_S, SHIFT_END_S)
                if t >= end or "dosing_t" in d:
                    restarted = "restart_t" in d and d["restart_t"] <= e["t"] + TRIP_RESTART_S
                    e["credit"] = (0.5 if restarted else 0.0) + (0.25 if restarted and d.get("discharge_closed_at_restart") else 0.0) + \
                                  (0.25 if "dosing_t" in d and d["dosing_t"] <= e["t"] + TRIP_DOSING_S else 0.0)
                    e["done"] = True

    # ------------------------------------------------------------------ the score
    def summary(self, voids, end_t):
        W = WEIGHTS
        parts = {}
        # v0.6: an overflow of T-101 (the make-up left open or opened too far) costs the line-up credit
        st = {"lineup": bool(self.lineup) and self.t101_overflow_t is None, "primed": bool(self.primed), "clean_start": bool(self.clean_start),
              "discharge_open": bool(self.discharge_open), "dosing_on": bool(self.dosing_on)}
        parts["startup"] = sum(W["startup"][k] for k, v in st.items() if v)
        span = SHIFT_END_S - STARTUP_DEADLINE_S
        lev_f = min(1.0, self.in_band["level"] / span)
        conc_f = min(1.0, self.in_band["conc"] / span)
        parts["operation"] = W["operation"]["level_in_band"] * lev_f + W["operation"]["conc_in_band"] * conc_f
        judged = [e for e in self.ev if not e.get("excluded")]
        n_ev = max(1, len(judged))
        parts["events"] = W["events"] * sum(e["credit"] for e in judged) / n_ev if judged else W["events"] * 0.0
        sd = {k: False for k in W["shutdown"]}
        if self.stop101_after_shift is not None:
            sd["pump_stopped"] = True
            sd["dosing_stopped_first"] = bool(self.stop_state.get("P-301_stopped_first"))
            sd["discharge_closed_first"] = self.stop_state.get("V-103", 1.0) <= 0.05
            sd["suction_closed"] = self.last_u.get("V-101", 1.0) <= 0.05 if hasattr(self, "last_u") else False
        sd["selector_off"] = bool(sd["pump_stopped"] and getattr(self, "last_selector", "OFF") == "OFF")
        parts["shutdown"] = sum(W["shutdown"][k] for k, v in sd.items() if v)
        log_ok = {"%s@%d" % (t, r): bool(self.first_log.get((r, t), {}).get("ok", False)) for r in (1, 2) for t in LOG_TAGS}
        parts["log"] = W["log"] * sum(1 for v in log_ok.values() if v) / len(log_ok)
        raw = sum(parts.values())
        voided = len(voids) > 0
        return {"score": 0.0 if voided else float(min(1.0, max(0.0, raw))), "raw_score": float(raw), "voided": voided,
                "voids": list(voids), "parts": {k: round(v, 4) for k, v in parts.items()}, "startup": st,
                "lineup_at_start": getattr(self, "lineup_detail", None), "first_start_t": self.first_start,
                "lineup_at_first_start": bool(self.lineup), "t101_overflow_t": self.t101_overflow_t,
                "window_fraction": {"level": round(lev_f, 4), "conc": round(conc_f, 4)},
                "events": [{"type": e["type"], "t": round(e["t"], 1), "credit": e["credit"], "excluded": bool(e.get("excluded")),
                            "detail": e["detail"]} for e in self.ev],
                "shutdown": sd, "log": log_ok, "logs": list(self.logs), "end_t": float(end_t)}
