"""The shift's record and its score. Reads the world's own state only: the
simulator's event log (spawns, line crossings, falls, exits), its contact
flags and the safety watch's damage list. Nothing the program sent reaches
it except through the physics.

Per parcel (one per arrival):
* label position at arrival: the world direction of its label face's normal
  as it lands (the parcel lies flat, yawed by a multiple of 90 degrees within
  a few degrees): top, underneath, downstream (+y), upstream (-y), near_side
  (-x, towards the robot) or far_side (+x).
* class, fixed at arrival (nothing the robot does changes it), from the
  station's duty (``score.duty``: the kinds and the label positions the
  station turns):
    U  any parcel that arrives label up (any kind);
    W  a parcel in the duty (its kind and its label position both in the
       duty) whose label is not up: the station's work;
    X  a parcel outside the duty whose label is not up: it passes untouched
       to the line's exception lane; nothing is earned by turning it.
* judgement at the scan line: the tick in which its centre first crosses the
  scan line going downstream. READABLE when all hold at that tick:
    1. label normal within ``label_tilt_deg`` of vertical (up);
    2. it rests on the belt: it touched the belt surface in at least one
       tick of the last ``rest_window_s`` (that tick included) and its lowest
       point is at most ``rest_clearance_max_m`` above the belt surface (one
       tick's contact flag never decides: a soft mailer riding untouched
       loses and regains its belt contact for a few ticks at a time), angular
       speed below ``max_angular_speed_rad_s``, speed relative to the belt
       below ``max_rel_speed_m_s``;
    3. nothing of the robot touches it;
    4. no other parcel touches it;
    5. it has not been damaged.
* LOST when it fell off the belt, was damaged (whenever), or had not crossed
  the scan line when the shift ended (for any reason: its duration, a budget,
  ``finish``, a stall).

Shift: credit = #(W readable) - #(U not readable) - #(lost, any class);
shift_score = max(0, credit) / #W (0 when the shift has no W parcel).

A shift fed on settle (``Shift.setting``, the line setting) counts its list:
every parcel of the list has its class (its duty is the score block's), and
a parcel is DONE when it was judged at the scan line, fell, or was damaged.
A parcel that is not done when the shift ends (never set down, or in the
station and not across) is NOT DONE: it costs nothing and earns nothing.
Each parcel of the list has a DUE TIME at the scan line: its set-down time
plus its free ride ((scan line - its set-down y) / the shift's belt speed)
plus ``due_margin_s`` (the station's rated cycle); a parcel whose centre
crosses the scan line after it is not readable ("late"), whatever else holds.
LOST = fell or damaged; U not readable = a U parcel that is done and not
readable; #W counts every W parcel of the list.
"""

from __future__ import annotations

import math

__all__ = ["ShiftRecord", "label_position", "parcel_class", "CLASSES"]

CLASSES = ("W", "U", "X")
_FACE_N = {0: (0.0, 0.0, 1.0), 1: (0.0, 0.0, -1.0), 2: (1.0, 0.0, 0.0), 3: (-1.0, 0.0, 0.0),
           4: (0.0, 1.0, 0.0), 5: (0.0, -1.0, 0.0)}


def label_position(parcel) -> str:
    """The label's position as the parcel arrives: its label face's normal (parcel frame) turned by the arrival
    yaw, named by its dominant world direction."""
    n = _FACE_N[int(parcel.label_face)]
    if n[2] > 0.5:
        return "top"
    if n[2] < -0.5:
        return "underneath"
    c, s = math.cos(float(parcel.yaw)), math.sin(float(parcel.yaw))
    x, y = c * n[0] - s * n[1], s * n[0] + c * n[1]
    if abs(y) >= abs(x):
        return "downstream" if y > 0 else "upstream"
    return "far_side" if x > 0 else "near_side"


def parcel_class(parcel, duty: dict | None) -> str:
    """U (label up), W (in the duty, label not up) or X (outside the duty, label not up)."""
    pos = label_position(parcel)
    if pos == "top":
        return "U"
    d = duty or {}
    kinds = tuple(d.get("kinds", ("rigid", "soft")))
    places = tuple(d.get("label_positions", ("underneath", "downstream", "upstream", "near_side", "far_side")))
    return "W" if (parcel.kind in kinds and pos in places) else "X"


class ShiftRecord:
    def __init__(self, world, watch, score_cfg: dict):
        self.world = world
        self.watch = watch
        self.cfg = dict(score_cfg)
        self.tilt = float(self.cfg["label_tilt_deg"])
        self.duty = dict(self.cfg.get("duty") or {})
        self.line = bool(getattr(world.build_shift, "setting", None))
        self.due_margin = float(self.cfg["due_margin_s"]) if self.line else None
        self.reset()

    def reset(self) -> None:
        w = self.world
        self._ev_seen = 0
        self.parcels = []
        for i, a in enumerate(w.build_shift.arrivals):
            p = a.parcel
            self.parcels.append({
                "i": i, "t_arrival": a.t, "kind": p.kind, "size": list(p.size), "mass": p.mass,
                "friction": p.friction, "label_face": p.label_face,
                "t_spawn": None, "entry_delay_s": None,
                "label_position": label_position(p),
                "cls": parcel_class(p, self.duty),
                "cls_from": "arrival",
                "scan": None, "readable": None, "failed": [],
                "touch_first_t": None, "touch_last_t": None, "touch_ticks": 0,
                "peak_squeeze_N": 0.0, "damaged": None,
                "fall_t": None, "exit_t": None, "lost": None, "lost_reason": None})
        self.final = None

    # ---------------------------------------------------------------- tick
    def on_tick(self, world) -> None:
        ev = world.truth.events
        for e in ev[self._ev_seen:]:
            self._on_event(e)
        self._ev_seen = len(ev)
        t = float(world.time)
        touch = world.truth._touch
        for i in range(min(len(self.parcels), len(touch))):
            if touch[i] and world.slots[i].active:
                r = self.parcels[i]
                r["touch_ticks"] += 1
                if r["touch_first_t"] is None:
                    r["touch_first_t"] = round(t, 4)
                r["touch_last_t"] = round(t, 4)

    def _on_event(self, e: dict) -> None:
        i = e.get("arrival", -1)
        if i is None or i < 0 or i >= len(self.parcels):
            return                                   # injected tooling parcels
        r = self.parcels[i]
        typ = e["type"]
        if typ == "spawn":
            r["t_spawn"] = round(e["t"], 4)
            r["entry_delay_s"] = round(e["delay"], 4)
            if self.line:
                w = self.world
                a = w.build_shift.arrivals[i]
                y0 = float(w.station.entry_y if a.y is None else a.y)
                v = float(w.build_shift.belt_speed[0][1])
                r["y_set"] = round(y0, 4)
                r["free_ride_s"] = round((float(w.scan_y) - y0) / v, 4)
                r["due_t"] = round(float(e["t"]) + (float(w.scan_y) - y0) / v + self.due_margin, 4)
        elif typ == "line" and e["direction"] > 0:
            if e["line"] == "scan" and r["scan"] is None:
                self._judge(i, e)
        elif typ == "fall" and r["fall_t"] is None:
            r["fall_t"] = round(e["t"], 4)
        elif typ == "exit" and r["exit_t"] is None:
            r["exit_t"] = round(e["t"], 4)

    def _judge(self, i: int, e: dict) -> None:
        c = self.cfg
        r = self.parcels[i]
        failed = []
        if not e["label_angle_deg"] <= self.tilt:
            failed.append("label_not_up")
        if "belt_contact_age_s" in e and "clearance_m" in e:
            age, clear = e["belt_contact_age_s"], e["clearance_m"]
            rests = (age is not None and age <= float(c["rest_window_s"]) + 1e-9
                     and clear <= float(c["rest_clearance_max_m"]))
        else:                                        # a constructed record without the window's fields
            rests = bool(e["on_belt"])
        if not rests:
            failed.append("not_on_belt")
        if not e["angular_speed"] < float(c["max_angular_speed_rad_s"]):
            failed.append("turning")
        if not e["speed_rel_belt"] < float(c["max_rel_speed_m_s"]):
            failed.append("sliding")
        if e["touching_robot"]:
            failed.append("robot_touching")
        if e.get("touching_parcel"):
            failed.append("parcel_touching")
        dmg = self.watch.damaged.get(i)
        if dmg is not None and dmg["t"] <= e["t"] + 1e-9:
            failed.append("damaged")
        if self.line and r.get("due_t") is not None and e["t_cross"] > float(r["due_t"]) + 1e-9:
            failed.append("late")
        r["scan"] = {"t": round(e["t"], 4), "t_cross": round(e["t_cross"], 4),
                     "label_angle_deg": round(e["label_angle_deg"], 3), "on_belt": bool(e["on_belt"]),
                     "belt_contact_age_s": e.get("belt_contact_age_s"),
                     "clearance_m": None if e.get("clearance_m") is None else round(float(e["clearance_m"]), 5),
                     "rests_on_belt": bool(rests),
                     "angular_speed": round(e["angular_speed"], 4),
                     "speed_rel_belt": round(e["speed_rel_belt"], 4),
                     "touching_robot": bool(e["touching_robot"]),
                     "touching_parcel": bool(e.get("touching_parcel")),
                     "x_rel": round(e["x_rel"], 4)}
        r["readable"] = not failed
        r["failed"] = failed

    # ---------------------------------------------------------------- end
    def finalize(self) -> dict:
        """The judgement at the shift's end (idempotent)."""
        if self.final is not None:
            return self.final
        w = self.world
        for slot, d in self.watch.damaged.items():
            if 0 <= slot < len(self.parcels):
                self.parcels[slot]["damaged"] = dict(d)
        for slot, f in self.watch.peak_squeeze.items():
            if 0 <= slot < len(self.parcels):
                self.parcels[slot]["peak_squeeze_N"] = round(f, 2)
        if self.line:
            return self._finalize_line()
        nW = nU = nX = Wr = Unr = lost = 0
        for r in self.parcels:
            if r["fall_t"] is not None:
                r["lost"], r["lost_reason"] = True, "fall"
            elif r["damaged"] is not None:
                r["lost"], r["lost_reason"] = True, "damaged"
            elif r["scan"] is None:
                r["lost"], r["lost_reason"] = True, "not_across"
            else:
                r["lost"] = False
            if r["readable"] is None:
                r["readable"] = False
            if r["cls"] == "W":
                nW += 1
                Wr += bool(r["readable"])
            elif r["cls"] == "U":
                nU += 1
                Unr += not r["readable"]
            else:
                nX += 1                              # outside the duty: only losing it costs
            lost += bool(r["lost"])
        credit = Wr - Unr - lost
        score = (max(0, credit) / nW) if nW > 0 else 0.0
        delays = [r["entry_delay_s"] for r in self.parcels if r["entry_delay_s"] is not None]
        self.final = {
            "n_parcels": len(self.parcels), "n_W": nW, "n_U": nU, "n_X": nX,
            "W_readable": Wr, "U_not_readable": Unr, "lost": lost,
            "lost_by_class": {c: sum(1 for r in self.parcels if r["cls"] == c and r["lost"]) for c in CLASSES},
            "duty": {"kinds": list(self.duty.get("kinds", [])),
                     "label_positions": list(self.duty.get("label_positions", []))},
            "lost_by": {k: sum(1 for r in self.parcels if r["lost_reason"] == k)
                        for k in ("fall", "damaged", "not_across")},
            "credit": credit, "shift_score": float(score),
            "max_entry_delay_s": max(delays) if delays else 0.0,
            "spawned": len(delays), "t_end": round(float(w.time), 4),
            "parcels": self.parcels,
        }
        assert math.isfinite(self.final["shift_score"])
        return self.final

    def _finalize_line(self) -> dict:
        """The judgement of a shift fed on settle: the list's parcels that are not done cost nothing."""
        w = self.world
        nW = nU = nX = Wr = Unr = lost = 0
        not_done = {"not_set_down": 0, "in_station": 0}
        for r in self.parcels:
            if r["fall_t"] is not None:
                r["lost"], r["lost_reason"] = True, "fall"
            elif r["damaged"] is not None:
                r["lost"], r["lost_reason"] = True, "damaged"
            else:
                r["lost"] = False
            done = r["lost"] or r["scan"] is not None
            r["done"] = bool(done)
            r["not_done_reason"] = None if done else ("not_set_down" if r["t_spawn"] is None else "in_station")
            if not done:
                not_done[r["not_done_reason"]] += 1
            if r["readable"] is None:
                r["readable"] = False
            r["late"] = "late" in (r["failed"] or [])
            if r["cls"] == "W":
                nW += 1
                Wr += bool(r["readable"])
            elif r["cls"] == "U":
                nU += 1
                Unr += bool(done and not r["readable"])
            else:
                nX += 1
            lost += bool(r["lost"])
        credit = Wr - Unr - lost
        score = (max(0, credit) / nW) if nW > 0 else 0.0
        delays = [r["entry_delay_s"] for r in self.parcels if r["entry_delay_s"] is not None]
        self.final = {
            "n_parcels": len(self.parcels), "n_W": nW, "n_U": nU, "n_X": nX,
            "W_readable": Wr, "U_not_readable": Unr, "lost": lost,
            "lost_by_class": {c: sum(1 for r in self.parcels if r["cls"] == c and r["lost"]) for c in CLASSES},
            "duty": {"kinds": list(self.duty.get("kinds", [])),
                     "label_positions": list(self.duty.get("label_positions", []))},
            "lost_by": {k: sum(1 for r in self.parcels if r["lost_reason"] == k) for k in ("fall", "damaged")},
            "not_done": not_done, "done": sum(1 for r in self.parcels if r["done"]),
            "late": sum(1 for r in self.parcels if r["late"]), "due_margin_s": self.due_margin,
            "setting": dict(getattr(w.build_shift, "setting", None) or {}),
            "list_done_t": getattr(w, "list_done_t", None),
            "belt_halts": int(getattr(w, "belt_halts", 0)),
            "belt_halted_s": round(0.02 * int(getattr(w, "belt_halted_ticks", 0)), 4),
            "credit": credit, "shift_score": float(score),
            "max_entry_delay_s": max(delays) if delays else 0.0,
            "spawned": len(delays), "t_end": round(float(w.time), 4),
            "parcels": self.parcels,
        }
        assert math.isfinite(self.final["shift_score"])
        return self.final

    def so_far(self) -> dict:
        """Counts so far (privileged): crossings judged up to now."""
        out = {"judged": 0, "readable": 0, "W_readable": 0, "U_not_readable": 0}
        for r in self.parcels:
            if r["scan"] is not None:
                out["judged"] += 1
                out["readable"] += bool(r["readable"])
                if r["cls"] == "W" and r["readable"]:
                    out["W_readable"] += 1
                if r["cls"] == "U" and not r["readable"]:
                    out["U_not_readable"] += 1
        return out
