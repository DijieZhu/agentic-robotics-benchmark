"""The referee's rules as small state machines, independent of the physics (each is unit-tested on its own).

  LapClock   the lap automaton of the contract: timing line and ordered checkpoints, lap validity, the per-lap flags
  Touches    touch episodes: contacts (and wheels over a tape) merged when less than `merge_s` apart
  Watch      ground speed of the reference point: speeding intervals (warnings) and standstills
  Overtakes  passes of slow cars by station
"""

import math


class LapClock(object):
    """A lap is the stretch between two forward crossings of the timing line with every checkpoint crossed forward,
    in order, in between. The first forward crossing starts the clock; a forward crossing without the whole
    checkpoint sequence restarts it and completes nothing; a backward crossing stops it until the next forward
    crossing. Crossing a checkpoint backward un-crosses it when it was the last one crossed. Checkpoints crossed
    before a stop stay crossed.

    Chain breaks: consecutive laps join end to start. When the clock throws away the
    lap in progress (a backward crossing of the timing line, or a forward crossing without the whole checkpoint
    sequence), the next completed lap is written with `chain_break` true: it does not start at the crossing that
    completed the lap before it, so a run of consecutive laps ends there. The first lap of a session has no lap
    before it and is written with `chain_break` false."""

    def __init__(self, n_checkpoints):
        self.n_cp = int(n_checkpoints)
        self.running = False
        self.t_start = None
        self.next_cp = 0
        self.count = 0
        self.laps = []
        self.cur = None
        self.dropped = False              # a lap in progress was thrown away since the last completed lap

    def _start(self, t, touch_active):
        self.running = True
        self.t_start = t
        self.next_cp = 0
        self.cur = {"interrupted": False, "touches": 1 if touch_active else 0, "standstill": False, "overtakes": 0}

    def timing(self, t, direction, touch_active=False):
        """A crossing of the timing line at session time t; returns the `lap` event of a forward crossing."""
        t = float(t)
        if direction < 0:
            if self.running:
                self.dropped = True
            self.running = False
            self.cur = None
            return None
        ev = {"type": "lap", "count": self.count, "lap_ms": None}
        if self.running and self.next_cp == self.n_cp:
            lap_s = t - self.t_start
            self.count += 1
            c = self.cur
            self.laps.append({"n": self.count, "lap_s": round(lap_s, 6), "t_end_s": round(t, 6),
                              "interrupted": c["interrupted"], "touches": c["touches"],
                              "clean": (not c["interrupted"]) and c["touches"] == 0, "standstill": c["standstill"],
                              "overtakes": max(0, c["overtakes"]),
                              "chain_break": bool(self.dropped and self.count > 1)})
            self.dropped = False
            ev = {"type": "lap", "count": self.count, "lap_ms": max(1, int(round(lap_s * 1000.0)))}
        elif self.running:
            self.dropped = True           # restarted without the whole checkpoint sequence: the lap is thrown away
        self._start(t, touch_active)
        return ev

    def checkpoint(self, i, direction):
        if not self.running:
            return
        if direction > 0 and i == self.next_cp:
            self.next_cp += 1
        elif direction < 0 and self.next_cp > 0 and i == self.next_cp - 1:
            self.next_cp -= 1

    def interrupt(self):
        if self.running:
            self.cur["interrupted"] = True

    def touch(self):
        if self.running:
            self.cur["touches"] += 1

    def standstill(self):
        if self.running:
            self.cur["standstill"] = True

    def overtake(self, k):
        if self.running:
            self.cur["overtakes"] += int(k)


class Touches(object):
    """Touch episodes. A new episode starts at a touching physics step when the last touching step is at least
    `merge_s` earlier; a scrape along a wall is one touch."""

    def __init__(self, merge_s):
        self.merge = float(merge_s)
        self.last_t = -math.inf
        self.episodes = 0

    def step(self, t, touching):
        """True when this step starts a new episode."""
        if not touching:
            return False
        new = t - self.last_t >= self.merge - 1e-12
        if new:
            self.episodes += 1
        self.last_t = t
        return new

    def active(self, t):
        return t - self.last_t < self.merge - 1e-12

    def reset_gap(self):
        """Forget the last contact (after the staff placed the car): the next contact is a new episode."""
        self.last_t = -math.inf


class Watch(object):
    """Speeding: the ground speed above the limit continuously for `hold_s` makes one warning per interval.
    Standstill: below `still_mps` continuously for more than `still_s`."""

    def __init__(self, dt, hold_s, still_s, still_mps):
        self.dt = float(dt)
        self.hold_steps = int(math.ceil(float(hold_s) / self.dt - 1e-9))
        self.still_steps = int(math.floor(float(still_s) / self.dt + 1e-9))
        self.still_mps = float(still_mps)
        self.fast_run = 0
        self.warned = False
        self.still_run = 0

    def step(self, v, limit):
        """(warning, standstill) for one physics step at ground speed v (limit None = no limit)."""
        warn = False
        if limit is not None and v > limit:
            self.fast_run += 1
            if self.fast_run >= self.hold_steps and not self.warned:
                self.warned = True
                warn = True
        else:
            self.fast_run = 0
            self.warned = False
        if v < self.still_mps:
            self.still_run += 1
        else:
            self.still_run = 0
        return warn, self.still_run > self.still_steps

    def reset(self):
        self.fast_run = 0
        self.warned = False
        self.still_run = 0


class Overtakes(object):
    """Net passes of slow cars by station: the ego car's reference point goes from behind to ahead of a slow car's
    reference point (+1) or back (-1) while they are less than `window_m` apart along the track."""

    def __init__(self, window_m):
        self.window = float(window_m)
        self.prev = {}

    def step(self, gaps):
        """`gaps`: {slow car id: ego station - slow car station, wrapped to (-L/2, L/2]}. Returns the net change."""
        net = 0
        for cid, g in gaps.items():
            p = self.prev.get(cid)
            if p is not None and abs(g) < self.window and abs(p) < self.window:
                if p < 0.0 <= g:
                    net += 1
                elif g < 0.0 <= p:
                    net -= 1
            self.prev[cid] = g
        return net

    def forget(self):
        self.prev = {}
