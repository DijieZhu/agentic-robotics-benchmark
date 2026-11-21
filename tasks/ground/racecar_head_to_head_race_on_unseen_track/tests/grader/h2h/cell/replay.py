"""The push of a contact bout, measured against a copy of the heat without the car.

When the car first touches a field car, the referee copies the simulation as it stands after that physics step, which
is the state the heat itself goes on from: the model (its run-time arrays included), every position, velocity,
actuator state and solver warm start, the car's run-time state, the driver state of each field car (speed command,
steering, hold and stuck timers, the phase of its planning tick), the track objects and their tracker. The car is
then taken out of the copy: its geoms collide with nothing but the floor, the driver of the field cars no longer sees
it (it is not a thing ahead in a band, and not a thing a put-back keeps clear of), its commands are zero. The copy
runs for the window (`heat.car_crash_window_s`) with the heat's own order of calls of a physics step (car, field
cars, physics step, kinematics refresh, track objects); nothing of it returns into the heat.

Every `heat.car_crash_sample_s` the referee takes, in the heat and in the copy, the pair (sigma, e) of each field
car's reference point against that car's own line: e its signed lateral offset (left positive), sigma the station of
its projection on the line minus that station at the first contact, wrapped into half the line's length either side.
Both numbers are rounded to 1e-6 m and used as rounded; the first sample of both sides is the same. A sample taken at
or after the step at which the field car was put back on its line by its own staff is not used, nor any later one of
that side. With `alone` the samples of the copy:

    e_copy(sigma) = linear between the first two consecutive samples of the copy, in time order, that enclose sigma
                    (the first of the two when both are at sigma); for a place behind all of them the offset of the
                    sample with the smallest sigma, beyond all of them the one with the largest sigma
    push          = max over the heat's samples k >= 1 of |e_k - e_copy(sigma_k)|      (0 when there is none)

so the two trajectories are compared at the same place on the field car's line, not at the same time.

    parts = cell_parts(sim, refresh)                      # what one physics step of a heat runs
    alone, cut, wall_s = run_copy(parts, lines, steps, every, keep)
    probe = Probe(i0, lines, refs, resets, alone, cut, every, n, sample_s, window_s)
    probe.observe(i, refs, resets)                        # the heat's own sample at step i0 + k * every
    push_of(actual, alone), sample_push(pair, alone)
    FieldProbe(sim)                                       # the same on a simulation of the field without the car

`run_copy` is deterministic: the copy is the heat's own code on a copied state, and a heat that the car does not
influence is repeated by its copy exactly.
"""

import copy
import math
import time

import mujoco
import numpy as np

from sketch.examples import constants as K

from .rules import ROUND, offset_at, push_of, sample_push, wrap_half

REF_X = K.REF_POINT_C[0]
CUTS_HEAT = (None, "heat_end", "put_back")
CUTS_COPY = (None, "put_back", "fault")


def refs_of(slow):
    """Reference points of the field cars of a driver (the cell's own formula of a reference point)."""
    out = []
    for st in slow.states():
        x, y, yaw = st["pose"]
        out.append((x + REF_X * math.cos(yaw), y + REF_X * math.sin(yaw)))
    return out


def resets_of(slow):
    return [int(c.resets) for c in slow.cars]


class Sampler(object):
    """(sigma, e) of one car's reference point against its own line, from the first sample on (see the module text).

    The projection is searched near the last one (1 m back, 2 m ahead), on the whole line for the first sample and
    whenever the point is more than 0.6 m from what that search found."""

    __slots__ = ("line", "s0", "hint")

    def __init__(self, line):
        self.line = line
        self.s0 = None
        self.hint = None

    def __call__(self, x, y):
        ln = self.line
        if self.hint is None:
            s, e, _ = ln.project(x, y)
        else:
            s, e, d2 = ln.project(x, y, s_hint=self.hint, back_m=1.0, ahead_m=2.0)
            if d2 > 0.36:
                s, e, _ = ln.project(x, y)
        self.hint = s
        if self.s0 is None:
            self.s0 = s
        return [round(wrap_half(s - self.s0, ln.length), ROUND) + 0.0, round(float(e), ROUND) + 0.0]   # no -0.0

    def __deepcopy__(self, memo):
        c = Sampler(self.line)
        c.s0, c.hint = self.s0, self.hint
        return c


# ================================================================================================ the copy
class Parts(object):
    """What one physics step runs: the model and its data, the car's run-time object (or None), the field cars'
    driver, the track objects' trackers (or None), whether the kinematics is refreshed after the step, and the
    collision geoms of the car (those the copy takes out)."""

    __slots__ = ("model", "data", "car", "slow", "ducts", "boxes", "refresh", "car_geoms")

    def __init__(self, model, data, car, slow, ducts=None, boxes=None, refresh=True, car_geoms=()):
        self.model, self.data, self.car, self.slow = model, data, car, slow
        self.ducts, self.boxes, self.refresh = ducts, boxes, bool(refresh)
        self.car_geoms = np.asarray(car_geoms, dtype=np.int64).reshape(-1)


def cell_parts(sim, refresh):
    """The parts of a heat's simulation object of the cell."""
    return Parts(sim.model, sim.data, sim.car, sim.slow, sim.ducts, sim.boxes, refresh, np.nonzero(sim.ego)[0])


def car_geoms_of(model, prefix=K.EGO_PREFIX):
    """The geoms of the car's root body that take part in contact (none when the model has no car)."""
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "chassis")
    if root < 0:
        return np.zeros(0, dtype=np.int64)
    rid = model.body_rootid[model.geom_bodyid]
    live = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    return np.nonzero((rid == root) & live)[0]


def take_car_out(model, car_geoms, car, slow):
    """In a copy only: the car's geoms collide with the floor and nothing else, the driver of the field cars does not
    see the car, the car's commands are zero."""
    g = np.asarray(car_geoms, dtype=np.int64).reshape(-1)
    if g.size:
        model.geom_contype[g] = 0
        model.geom_conaffinity[g] = K.BIT_FLOOR
    if car is not None:
        if getattr(car, "script", None) is not None:      # a stand-in car of the tests: driven by its commands here
            car.script = None
        car.set_targets(0.0, 0.0)
    without = getattr(slow, "without_car", None)
    if callable(without):                                  # a stand-in driver of the tests drops what the car did
        without()
    else:
        slow.ego = None                                    # the shared driver's binding of the car


class Copy(object):
    """A copy of the simulation as it stands now, the car taken out (see the module text). `keep` holds objects that
    are only read (the world, the floor, ...): they are shared with the copy instead of copied."""

    def __init__(self, parts, keep=()):
        m, d = parts.model, parts.data
        mc = copy.copy(m)
        dc = mujoco.MjData(mc)
        mujoco.mj_copyData(dc, mc, d)
        memo = {id(m): mc, id(d): dc}
        for o in keep:
            memo[id(o)] = o
        self.car, self.slow, self.ducts, self.boxes = copy.deepcopy((parts.car, parts.slow, parts.ducts, parts.boxes), memo)
        self.model, self.data, self.refresh = mc, dc, parts.refresh
        take_car_out(mc, parts.car_geoms, self.car, self.slow)
        self.warn0 = [int(dc.warning[k].number) for k in range(len(dc.warning))]

    def step(self):
        """One physics step in the heat's order of calls."""
        m, d = self.model, self.data
        if self.car is not None:
            self.car.pre_step()
        self.slow.pre_step()
        mujoco.mj_step(m, d)
        if self.refresh:
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            mujoco.mj_comVel(m, d)
        if self.ducts is not None:
            self.ducts.update()
        if self.boxes is not None:
            self.boxes.update()

    def faulted(self):
        d = self.data
        return (not np.isfinite(d.qpos).all()) or any(int(d.warning[k].number) > self.warn0[k] for k in range(len(self.warn0)))


def run_copy(parts, lines, steps, every, keep=()):
    """Copy the simulation now, take the car out and run `steps` physics steps. Returns (alone, cut, wall_s): per
    field car its samples [[sigma, e], ...] from the first one (now) and then every `every` steps, and why they ended
    early (None, "put_back" or "fault"); the wall-clock seconds of the copy."""
    t = time.perf_counter()
    c = Copy(parts, keep)
    n = len(lines)
    samplers = [Sampler(ln) for ln in lines]
    resets0 = resets_of(c.slow)
    alone = [[smp(*r)] for smp, r in zip(samplers, refs_of(c.slow))]
    cut = [None] * n
    for i in range(1, int(steps) + 1):
        c.step()
        if i % every:
            continue
        bad = c.faulted()
        refs, resets = refs_of(c.slow), resets_of(c.slow)
        for k in range(n):
            if cut[k] is not None:
                continue
            if bad:
                cut[k] = "fault"
            elif resets[k] != resets0[k]:
                cut[k] = "put_back"
            else:
                alone[k].append(samplers[k](*refs[k]))
        if all(x is not None for x in cut):
            break
    return alone, cut, time.perf_counter() - t


# ================================================================================================ the heat's side
class Probe(object):
    """One copy opened at step i0 and the heat's own samples over the same window, for every field car.

    The heat calls `observe(i, refs, resets)` at its steps i0 + k * every (k = 1 .. n) while it runs; `last` holds
    the step of each car's latest sample of the heat."""

    def __init__(self, i0, lines, refs, resets, alone, alone_cut, every, n, sample_s, window_s, ids=None):
        self.i0, self.every, self.n = int(i0), int(every), int(n)
        self.sample_s, self.window_s = float(sample_s), float(window_s)
        self.ids = list(ids) if ids is not None else list(range(len(lines)))
        self.samplers = [Sampler(ln) for ln in lines]
        self.actual = [[smp(*r)] for smp, r in zip(self.samplers, refs)]
        self.resets0 = [int(x) for x in resets]
        self.cut = [None] * len(lines)
        self.last = [self.i0] * len(lines)
        self.alone = alone
        self.alone_cut = list(alone_cut)
        for a, r in zip(self.actual, self.alone):
            if a[0] != r[0]:
                raise RuntimeError("the copy does not start where the heat is: %r against %r" % (r[0], a[0]))

    def due(self, i):
        d = int(i) - self.i0
        return d > 0 and d % self.every == 0 and d // self.every <= self.n

    def over(self, i):
        return int(i) - self.i0 >= self.n * self.every

    def observe(self, i, refs, resets):
        """The heat's samples at step i (a step of `due`)."""
        for k, smp in enumerate(self.samplers):
            if self.cut[k] is not None:
                continue
            if int(resets[k]) != self.resets0[k]:
                self.cut[k] = "put_back"
                continue
            self.actual[k].append(smp(*refs[k]))
            self.last[k] = int(i)

    def latest_push(self, k, i):
        """The push of car k's sample of the heat taken at step i, or None when none was taken then."""
        if self.last[k] != int(i) or len(self.actual[k]) < 2:
            return None
        return sample_push(self.actual[k][-1], self.alone[k])

    def record(self, k):
        """The `replay` block of a bout with car k: the two lists, and why either has fewer than n + 1 samples."""
        a = [list(p) for p in self.actual[k]]
        cut = self.cut[k]
        if cut is None and len(a) < self.n + 1:
            cut = "heat_end"
        return {"sample_s": self.sample_s, "window_s": self.window_s, "actual": a,
                "alone": [list(p) for p in self.alone[k]], "cut": cut, "alone_cut": self.alone_cut[k]}

    def result(self):
        """Per field car: its id, the two lists, the cuts and the push (maintainer read-out)."""
        out = []
        for k in range(len(self.samplers)):
            rec = self.record(k)
            rec["id"] = self.ids[k]
            rec["push_m"] = push_of(rec["actual"], rec["alone"])
            out.append(rec)
        return out


class FieldProbe(object):
    """A probe on a simulation of the field that the caller steps itself (an object with `model`, `data`, `slow`,
    `ego` (the car's run-time object or None), `ducts`, `boxes`, `refresh`, stepped in the heat's order): the copy is
    run when the probe is made; call `after_step()` after every physics step of the original.

        p = FieldProbe(sim, window_s=1.5, sample_s=0.025)
        for _ in range(p.steps):
            sim.physics_step()
            p.after_step()
        p.result()          # per field car: actual, alone, cuts, push_m
    """

    def __init__(self, sim, window_s=1.5, sample_s=0.025, lines=None, keep=()):
        dt = float(sim.model.opt.timestep)
        self.every = int(round(float(sample_s) / dt))
        self.steps = int(round(float(window_s) / dt))
        if self.every < 1 or abs(self.every * dt - float(sample_s)) > 1e-9 or self.steps % self.every:
            raise ValueError("the window and the sample period must be whole numbers of physics steps and samples")
        n = self.steps // self.every
        self.sim = sim
        lines = list(lines) if lines is not None else [c.path for c in sim.slow.cars]
        car = getattr(sim, "ego", None)
        parts = Parts(sim.model, sim.data, car, sim.slow, getattr(sim, "ducts", None), getattr(sim, "boxes", None),
                      getattr(sim, "refresh", True), car_geoms_of(sim.model) if car is not None else ())
        alone, cut, self.wall_s = run_copy(parts, lines, self.steps, self.every,
                                           keep=tuple(keep) + (getattr(sim, "world", None),))
        self.probe = Probe(0, lines, refs_of(sim.slow), resets_of(sim.slow), alone, cut, self.every, n, sample_s,
                           window_s, ids=[c.id for c in sim.slow.cars])
        self.i = 0

    def after_step(self):
        self.i += 1
        if self.probe.due(self.i):
            self.probe.observe(self.i, refs_of(self.sim.slow), resets_of(self.sim.slow))

    @property
    def done(self):
        return self.probe.over(self.i)

    def result(self):
        return self.probe.result()


__all__ = ["ROUND", "CUTS_HEAT", "CUTS_COPY", "wrap_half", "refs_of", "resets_of", "Sampler", "offset_at",
           "sample_push", "push_of", "Parts", "cell_parts", "car_geoms_of", "take_car_out", "Copy", "run_copy", "Probe",
           "FieldProbe"]
