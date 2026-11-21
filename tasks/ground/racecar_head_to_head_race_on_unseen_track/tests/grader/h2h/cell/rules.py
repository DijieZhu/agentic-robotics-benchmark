"""The rules of a heat as small state machines, independent of the physics (each is unit-tested on its own).

  Standing      one car's distance along the track and its progress (laps completed plus the lap in progress)
  Ledger        the lead of the car on every field car, its overtake and loss events and their credit
  Contacts      contact bouts of the car with each field car: merging, push, staff reset, class, fault
  fault_of      whose fault a bout is, from the ledger at its first contact
  Handling      when each field car's own staff held it or put it back
  StandingStill how long the car's ground speed has stayed below a threshold

Distances are stations of the track's centreline (the nearest centreline point of a reference point). `L` is the
track length; `signed_gap(a, b)` is a - b wrapped into (-L / 2, L / 2].
"""

import math

EPS = 1e-9
ROUND = 6                       # the push's samples and the push itself are rounded to 1e-6 m


def signed_gap(a, b, length):
    g = (a - b) % length
    return g - length if g > 0.5 * length else g


def wrap_half(x, length):
    """x wrapped into (-length / 2, length / 2]."""
    g = float(x) % float(length)
    return g - length if g > 0.5 * length else g


# ------------------------------------------------------------------------------------------------ the push of a bout
def offset_at(alone, sigma):
    """The offset of the copy without the car at the place sigma of the field car's line: linear between the first
    two consecutive samples of the copy, in time order, that enclose sigma (the first of the two when both are at
    sigma); for a place behind all of them the offset of the sample with the smallest sigma, beyond all of them the
    one with the largest sigma. `alone` = [[sigma, e], ...]."""
    for (s0, e0), (s1, e1) in zip(alone, alone[1:]):
        if min(s0, s1) <= sigma <= max(s0, s1):
            return e0 if s1 == s0 else e0 + (e1 - e0) * (sigma - s0) / (s1 - s0)
    lo = min(alone, key=lambda p: p[0])
    hi = max(alone, key=lambda p: p[0])
    return lo[1] if sigma < lo[0] else hi[1]


def sample_push(pair, alone):
    """|e - e_copy(sigma)| of one sample [sigma, e] of the heat, rounded to 1e-6 m."""
    return round(abs(float(pair[1]) - offset_at(alone, float(pair[0]))), ROUND)


def push_of(actual, alone):
    """The push of a bout from its two lists of samples (the heat's and the copy's, both from the first contact on,
    starting with the same pair): the largest sample push of the heat's samples after the first, 0 without any."""
    a = [[round(float(s), ROUND), round(float(e), ROUND)] for s, e in actual]
    r = [[round(float(s), ROUND), round(float(e), ROUND)] for s, e in alone]
    if not a or not r or a[0] != r[0]:
        raise ValueError("the heat's and the copy's samples must start with the same pair")
    return max([sample_push(p, r) for p in a[1:]] or [0.0])


class Standing(object):
    """Distance along the track and progress of one car's reference point.

    `D` follows every motion of the reference point, a staff placement included: the lead of the car on a field car is
    the difference of their `D`. Both are counted from the timing line, so a car on the grid has a negative `D`
    (`D0` gives the start value when it is not the car's own wrapped distance from the line).
    `Dp` advances only by motion between two positions on the track and by staff placements; it is the distance of
    the progress. Progress = laps completed + (Dp - Dp at the last forward crossing of the timing line) / L: before the
    first crossing it is minus the distance to the line over L; behind the line after a backward crossing it is just
    below the laps completed."""

    def __init__(self, length, s_line, s0, on_track=True, D0=None):
        self.L = float(length)
        self.s_line = float(s_line)
        self.s = float(s0)
        self.D = signed_gap(self.s, self.s_line, self.L) if D0 is None else float(D0)
        self.Dp = self.D
        self.D_cross = 0.0
        self.laps = 0
        self.on = bool(on_track)
        self.last_dp = 0.0

    def move(self, s, on_track, placed=False):
        g = signed_gap(float(s), self.s, self.L)
        self.D += g
        self.last_dp = g if (placed or (on_track and self.on)) else 0.0
        self.Dp += self.last_dp
        self.s = float(s)
        self.on = bool(on_track)

    def crossed_forward(self, frac, laps):
        """A forward crossing of the timing line at fraction `frac` of the last move; `laps` = laps completed after it."""
        self.D_cross = self.Dp - (1.0 - float(frac)) * self.last_dp
        self.laps = int(laps)

    @property
    def progress(self):
        return self.laps + (self.Dp - self.D_cross) / self.L


class Handling(object):
    """Intervals in which a field car's own staff held it or were putting it back: from the step its hold is first
    seen to the step it stands on its line again (open while it is still held)."""

    def __init__(self, ids):
        self.iv = {j: [] for j in ids}
        self.held = {j: False for j in ids}

    def step(self, t, j, held):
        """Feed the state of field car j after a physics step; True when its hold started at this step (a reset was
        triggered)."""
        started = bool(held) and not self.held[j]
        if started:
            self.iv[j].append([float(t), None])
        elif self.held[j] and not held:
            self.iv[j][-1][1] = float(t)
        self.held[j] = bool(held)
        return started

    def overlaps(self, j, a, b):
        for h0, h1 in self.iv[j]:
            if h0 <= b + EPS and (h1 is None or h1 >= a - EPS):
                return True
        return False

    def now(self, j):
        return self.held[j]


class Bout(object):
    """One contact bout. `probe` and `k` name the copy of the heat without the car that its push is measured
    against, and the field car's index in it (None when no copy is run)."""

    __slots__ = ("car", "t0", "t1", "push", "reset", "cls", "fault", "warned", "t_judged", "probe", "k")

    def __init__(self, car, t, fault):
        self.car = car
        self.t0 = self.t1 = float(t)
        self.push = 0.0
        self.reset = False
        self.cls = None                 # None until judged: "crash" or "touch"
        self.fault = fault              # "car" or "opponent", judged at t0
        self.warned = False
        self.t_judged = None
        self.probe = None
        self.k = None

    def record(self):
        return {"car": self.car, "t0_s": round(self.t0, 6), "t1_s": round(self.t1, 6), "push_m": round(self.push, 6),
                "reset": bool(self.reset), "class": self.cls or "touch", "fault": self.fault, "warned": bool(self.warned),
                "replay": None if self.probe is None else self.probe.record(self.k)}


class Contacts(object):
    """Contact bouts of the car with each field car.

    Contacts with the same field car less than `merge_s` apart are one bout [t0, t1]. Its push is fed sample by sample
    (the largest so far counts; the samples belong to the window_s after t0, see the replay module); the bout is a
    car crash at the first moment at which the push reaches `push_m` or that car's staff reset is triggered within
    [t0, t1 + window_s], and a touch once t1 + window_s has passed without either. The fault is judged once, at t0."""

    def __init__(self, ids, merge_s, push_m, window_s):
        self.ids = list(ids)
        self.merge = float(merge_s)
        self.thr = float(push_m)
        self.window = float(window_s)
        self.bouts = {j: [] for j in ids}
        self.all = []
        self.open = []

    def contact(self, j, t, fault_fn, opened=None):
        """A contact with field car j at time t. `fault_fn()` and `opened(bout)` are called only for a new bout."""
        b = self.bouts[j][-1] if self.bouts[j] else None
        if b is not None and t - b.t1 < self.merge - EPS:
            b.t1 = float(t)
            return b
        b = Bout(j, t, fault_fn())
        self.bouts[j].append(b)
        self.all.append(b)
        self.open.append(b)
        if opened is not None:
            opened(b)
        return b

    def watching(self, j, t):
        """True when a bout of j is still open to a crash at time t."""
        return any(t <= b.t1 + self.window + EPS for b in self.bouts[j][-2:])

    def measure(self, j, t, reset_triggered, sample=None):
        """Field car j at time t: whether its staff reset was triggered now, and `sample(bout)` -> the push of the
        bout's sample taken now (None when it took none); returns the bouts judged car crashes now."""
        out = []
        for b in self.bouts[j][-2:]:
            if t > b.t1 + self.window + EPS:
                continue
            if sample is not None:
                p = sample(b)
                if p is not None and p > b.push:
                    b.push = float(p)
            if reset_triggered:
                b.reset = True
            if b.cls is None and (b.push >= self.thr - EPS or b.reset):
                b.cls = "crash"
                b.t_judged = float(t)
                out.append(b)
        return out

    def settle(self, t, final=False):
        """Bouts whose window has passed (all, when final) without a crash become touches."""
        keep = []
        for b in self.open:
            if b.cls is None and (final or t > b.t1 + self.window + EPS):
                b.cls = "touch"
            if b.cls is None:
                keep.append(b)
        self.open = keep

    def overlaps(self, j, a, b):
        """True when a contact with j lies in [a, b] (a bout's [t0, t1] overlaps it)."""
        for bt in self.bouts[j]:
            if bt.t0 <= b + EPS and bt.t1 >= a - EPS:
                return True
        return False


class Ledger(object):
    """The lead of the car on every field car and the overtake events.

    lambda_j = D(car) - D(j). With level m_j (0 at the start, the car behind every field car) and h half a car length:
    an overtake when lambda_j >= m_j L + h (then m_j + 1), a loss (the car re-overtaken) when lambda_j <= (m_j - 1) L - h
    (then m_j - 1). An overtake at t is clean when there is no contact with j in [t - before_s, t + after_s] (cut at
    the end of the heat), j is not held or put back by its staff in that interval, the car's reference point is on the
    track and the car is not stopped by the staff at t, and no staff placement of either car made the step; else its
    credit is 0. Its credit is final after_s after it, or at the end of the heat."""

    WHY = ("contact", "staff", "off_track", "placement")

    def __init__(self, ids, length, h, before_s, after_s):
        self.ids = list(ids)
        self.L = float(length)
        self.h = float(h)
        self.before = float(before_s)
        self.after = float(after_s)
        self.m = {j: 0 for j in ids}
        self.lam = {j: None for j in ids}
        self.n_over = {j: 0 for j in ids}
        self.t_last_over = {j: None for j in ids}
        self.events = []
        self.pending = []

    def step(self, t, lam, on_track, stopped, placed_car, placed):
        """Leads after a physics step at time t ({j: lambda_j}); returns the new events."""
        new = []
        for j in self.ids:
            x = float(lam[j])
            self.lam[j] = x
            while x >= self.m[j] * self.L + self.h - EPS:
                self.m[j] += 1
                e = {"t": float(t), "car": j, "kind": "overtake", "clean": None, "why_not": None,
                     "now": {"staff": bool(stopped), "off_track": not on_track, "placement": bool(placed_car or placed.get(j))}}
                self.n_over[j] += 1
                self.t_last_over[j] = float(t)
                new.append(e)
                self.pending.append(e)
            while x <= (self.m[j] - 1) * self.L - self.h + EPS:
                self.m[j] -= 1
                new.append({"t": float(t), "car": j, "kind": "loss", "clean": None, "why_not": None, "now": None})
        self.events += new
        return new

    def gap(self, j):
        """The on-track gap of the car on j: lambda_j wrapped into (-L / 2, L / 2] (positive: the car is ahead)."""
        g = self.lam[j] % self.L
        return g - self.L if g > 0.5 * self.L else g

    def settle(self, t, contacts, handling, t_end=None):
        """Decide the credit of every overtake whose window has passed by time t, or of all when t_end is given (the
        windows cut at t_end)."""
        keep = []
        for e in self.pending:
            if t_end is None and t < e["t"] + self.after - EPS:
                keep.append(e)
                continue
            a = e["t"] - self.before
            b = e["t"] + self.after if t_end is None else min(e["t"] + self.after, t_end)
            fails = {"contact": contacts.overlaps(e["car"], a, b),
                     "staff": handling.overlaps(e["car"], a, b) or e["now"]["staff"],
                     "off_track": e["now"]["off_track"], "placement": e["now"]["placement"]}
            why = next((w for w in self.WHY if fails[w]), None)
            e["clean"] = why is None
            e["why_not"] = why
        self.pending = keep

    def records(self):
        return [{"t_s": round(e["t"], 6), "car": e["car"], "kind": e["kind"],
                 "clean": e["clean"] if e["kind"] == "overtake" else None,
                 "why_not": e["why_not"] if e["kind"] == "overtake" else None} for e in self.events]

    def balance(self, j):
        clean = sum(1 for e in self.events if e["car"] == j and e["kind"] == "overtake" and e["clean"])
        contact = sum(1 for e in self.events if e["car"] == j and e["kind"] == "overtake" and not e["clean"])
        losses = sum(1 for e in self.events if e["car"] == j and e["kind"] == "loss")
        return clean, contact, losses


def fault_of(ledger, j, t0, clear_s):
    """Whose fault a contact bout with field car j that starts at t0 is, from the ledger as it stands then.

    An overtake of j is any overtake event of the ledger (the pass on the track, a lapping, or passing again a car
    that had lapped the car). The fault is the car's, with two exceptions, both with g_j (the on-track gap, positive
    when the car is ahead) above half a car length and no overtake of j completed within the `clear_s` before t0:
    (a) the car has overtaken j at least once (it had been clear ahead and was run into); (b) lambda_j < 0, j is ahead
    of the car in distance covered (j is coming up to lap the car). Then it is j's ("opponent")."""
    g = ledger.gap(j)
    tl = ledger.t_last_over[j]
    recent = tl is not None and t0 - tl < float(clear_s) - EPS
    ahead = g > ledger.h + EPS
    a = ledger.n_over[j] >= 1 and ahead and not recent
    b = ahead and ledger.lam[j] < 0.0 and not recent
    return "opponent" if (a or b) else "car"


class StandingStill(object):
    """Physics steps in a row with the car's ground speed below `speed_mps`; `step` is True once that lasted more
    than `still_s`."""

    def __init__(self, dt, still_s, speed_mps):
        self.dt = float(dt)
        self.steps = int(math.floor(float(still_s) / self.dt + 1e-9))
        self.v = float(speed_mps)
        self.run = 0

    def step(self, v):
        if v < self.v:
            self.run += 1
        else:
            self.run = 0
        return self.run > self.steps

    def reset(self):
        self.run = 0


__all__ = ["EPS", "ROUND", "signed_gap", "wrap_half", "offset_at", "sample_push", "push_of", "Standing", "Handling",
           "Bout", "Contacts", "Ledger", "fault_of", "StandingStill"]
