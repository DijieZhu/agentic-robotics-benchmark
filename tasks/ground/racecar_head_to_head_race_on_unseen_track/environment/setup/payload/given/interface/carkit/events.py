"""Decoding of the organiser's messages (the `events` list of a `step` reply).

    log = SessionLog()
    r = car.step(steer, speed)
    log.feed(r)                    # updates: tick, laps, lap_count, stopped, warnings, phase, ended
    if log.stopped: ...            # commands are ignored until `released`
    log.laps                       # organiser's lap times in seconds, in order

Events name no object, position, displacement or reason; they are: lap {count, lap_ms}, stopped, released,
warning {count}, phase {name}, session_end {reason}.
"""

from . import protocol


def of_type(events, kind):
    """The events of one type, in order."""
    return [e for e in events if e.get("type") == kind]


def lap_times_s(events):
    """Lap times (s) of the `lap` events that completed a lap (`lap_ms` not null)."""
    return [e["lap_ms"] / 1000.0 for e in events if e.get("type") == "lap" and e.get("lap_ms") is not None]


class SessionLog(object):
    """The state a program can know from the events of one session."""

    def __init__(self, session=None):
        self.session = session
        self.tick = 0
        self.laps = []                 # completed lap times, s
        self.lap_count = 0
        self.clock_started = False     # the timing line has been crossed forward at least once
        self.stopped = False
        self.stops = 0
        self.warnings = 0
        self.phase = None
        self.ended = None              # session_end reason
        self.history = []              # (tick, event)

    def feed(self, reply):
        """Take one `step` result (the dict returned by Car.step). Returns its events."""
        self.tick = int(reply.get("tick", self.tick))
        events = reply.get("events") or []
        for e in events:
            kind = e.get("type")
            if kind not in protocol.EVENT_TYPES:
                raise ValueError("unknown event %r" % (e,))
            self.history.append((self.tick, e))
            if kind == "lap":
                self.clock_started = True
                self.lap_count = int(e["count"])
                if e.get("lap_ms") is not None:
                    self.laps.append(e["lap_ms"] / 1000.0)
            elif kind == "stopped":
                self.stopped = True
                self.stops += 1
            elif kind == "released":
                self.stopped = False
            elif kind == "warning":
                self.warnings = int(e["count"])
            elif kind == "phase":
                self.phase = e["name"]
            elif kind == "session_end":
                self.ended = e["reason"]
        return events

    @property
    def best_lap_s(self):
        return min(self.laps) if self.laps else None
