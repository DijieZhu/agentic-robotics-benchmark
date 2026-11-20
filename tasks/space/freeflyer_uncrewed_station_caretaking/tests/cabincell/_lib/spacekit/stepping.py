"""State-consistent stepping with external wrenches.

The rule: external loads must be computed from the CURRENT state.  Writing
``xfrc_applied`` from ``data.xipos`` before a plain ``mj_step`` uses the previous step's kinematics
(one-step lag: 0.22 m error after one orbit at dt = 10 ms, 2.2 m at dt = 0.1 s).  Two correct ways:

* ``step()``: ``mj_step1`` (positions / velocities of the current state) -> sum the sources into
  ``xfrc_applied`` -> ``mj_step2``.  Works for Euler / implicit / implicitfast, NOT for RK4 (MuJoCo would
  silently fall back to Euler; we raise instead).  ``leapfrog_start()`` shifts the initial free-body
  velocities by -dt/2 * a so that semi-implicit Euler becomes leapfrog (removes the O(a dt) start error).
* ``PassiveCallback``: the sources are evaluated inside ``mj_passive`` at every RK4 stage, with the stage
  time.  This is MuJoCo's process-global ``mjcb_passive``; the context manager installs and removes it
  and refuses to overwrite a foreign callback.  It is the only global state spacekit ever touches.

A wrench source is any object with ``accumulate(model, data, W)`` adding a (nbody, 6) array of world-axes
[force, torque] at the body centres of mass.  ``SlowForceCache`` re-evaluates slowly varying sources
every ``every`` calls only (configurable rate) and replays the cached wrench in between.
``Simulator`` schedules physics steps, slow forces and periodic tasks (firmware ticks) deterministically
from integer step counts; no wall clock is read anywhere.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

__all__ = [
    "DT_CONTACT_MAX", "DT_FREE_MAX", "step", "leapfrog_start", "sum_sources", "SlowForceCache",
    "PassiveCallback", "check_timestep", "check_finite", "Simulator", "StateError",
]

DT_CONTACT_MAX = 0.002   # s; grasp / push / perch / rail phases (5 ms: 4 mm compliance slip, 10 ms: grasp lost)
DT_FREE_MAX = 0.010      # s; contact-free drift (RK4 + stiff servos diverge above this)

_RK4 = int(mujoco.mjtIntegrator.mjINT_RK4)


class StateError(RuntimeError):
    """Raised when the simulation state became non-finite or a guard is violated."""


def sum_sources(model, data, sources, W=None) -> np.ndarray:
    """Sum the wrenches of ``sources`` (in order) into W (zeroed first); returns W (nbody, 6)."""
    if W is None:
        W = np.zeros((model.nbody, 6))
    else:
        W[:] = 0.0
    for s in sources:
        s.accumulate(model, data, W)
    return W


def step(model, data, sources=(), nstep: int = 1, W=None) -> None:
    """``nstep`` physics steps with state-consistent external wrenches (mj_step1 -> sources -> mj_step2).

    The summed wrench replaces ``data.xfrc_applied`` at every step.  Raises ValueError for RK4 (use
    PassiveCallback).  W is an optional (nbody, 6) work buffer.
    """
    if int(model.opt.integrator) == _RK4:
        raise ValueError("step() cannot be used with RK4; use PassiveCallback + mujoco.mj_step")
    if W is None:
        W = np.zeros((model.nbody, 6))
    for _ in range(nstep):
        mujoco.mj_step1(model, data)
        sum_sources(model, data, sources, W)
        data.xfrc_applied[:] = W
        mujoco.mj_step2(model, data)


def leapfrog_start(model, data, sources) -> np.ndarray:
    """Shift free-joint linear velocities by -dt/2 * a(q0), a = subtree external force / subtree mass.

    Call once after setting the initial state (it runs mj_forward).  Makes MuJoCo's semi-implicit Euler a
    leapfrog scheme for position-dependent forces: one orbit at dt = 10 ms goes from 4.5e-4 m to 4.5e-6 m
    error.  Returns the applied velocity shifts (njnt_free, 3).
    """
    mujoco.mj_forward(model, data)
    W = sum_sources(model, data, sources)
    shifts = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco.mjtJoint.mjJNT_FREE):
            continue
        root = model.body_rootid[model.jnt_bodyid[j]]
        sub = np.nonzero(model.body_rootid == root)[0]
        msum = float(model.body_mass[sub].sum())
        if msum <= 0:
            continue
        acc = W[sub, 0:3].sum(0) / msum
        adr = model.jnt_dofadr[j]
        dv = -0.5 * model.opt.timestep * acc
        data.qvel[adr:adr + 3] += dv
        shifts.append(dv)
    return np.array(shifts).reshape(-1, 3)


class SlowForceCache:
    """Wrap slowly varying sources: evaluate them every ``every``-th call, replay the cache otherwise.

    every=1 means no caching.  The refresh is counted in calls (= physics steps), so it is deterministic.
    ``invalidate()`` forces a refresh at the next call (use after teleporting bodies).
    """

    def __init__(self, sources, every: int = 10):
        self.sources = list(sources)
        self.every = max(1, int(every))
        self._count = 0
        self._W = None

    def invalidate(self) -> None:
        """Force re-evaluation at the next call."""
        self._count = 0

    def accumulate(self, model, data, W) -> None:
        """Add the (possibly cached) wrench of the wrapped sources into W."""
        if self._W is None or self._W.shape != W.shape:
            self._W = np.zeros_like(W)
            self._count = 0
        if self._count % self.every == 0:
            self._W[:] = 0.0
            for s in self.sources:
                s.accumulate(model, data, self._W)
        self._count += 1
        W += self._W


class PassiveCallback:
    """Context manager evaluating wrench sources inside mj_passive (every RK4 stage, stage time).

    with PassiveCallback([frame, air]):
        mujoco.mj_step(model, data)

    Installs MuJoCo's process-global ``mjcb_passive`` and restores the previous value on exit.  Raises
    RuntimeError if another callback is already installed (unless replace=True).  One scene per process.
    """

    def __init__(self, sources, replace: bool = False):
        self.sources = list(sources)
        self.replace = replace
        self._prev = None
        self._W = None
        self.calls = 0

    def _cb(self, m, d):
        if self._W is None or self._W.shape[0] != m.nbody:
            self._W = np.zeros((m.nbody, 6))
        W = sum_sources(m, d, self.sources, self._W)
        self.calls += 1
        nz = np.nonzero(np.any(W != 0.0, axis=1))[0]
        for b in nz:
            mujoco.mj_applyFT(m, d, W[b, 0:3].copy(), W[b, 3:6].copy(), d.xipos[b].copy(), int(b), d.qfrc_passive)

    def __enter__(self):
        prev = mujoco.get_mjcb_passive()
        if prev is not None and not self.replace:
            raise RuntimeError("an mjcb_passive callback is already installed in this process")
        self._prev = prev
        mujoco.set_mjcb_passive(self._cb)
        return self

    def __exit__(self, *exc):
        mujoco.set_mjcb_passive(self._prev)
        self._prev = None
        return False


def check_timestep(model, phase: str = "contact") -> None:
    """Raise ValueError if model.opt.timestep exceeds the limit for the phase ('contact' or 'free')."""
    lim = {"contact": DT_CONTACT_MAX, "free": DT_FREE_MAX}[phase]
    if model.opt.timestep > lim + 1e-12:
        raise ValueError(f"timestep {model.opt.timestep} s exceeds the {phase} limit {lim} s")


def check_finite(data) -> None:
    """Raise StateError if qpos or qvel contains NaN or Inf."""
    if not (np.all(np.isfinite(data.qpos)) and np.all(np.isfinite(data.qvel))):
        raise StateError(f"non-finite state at t = {data.time}")


class Simulator:
    """Deterministic scheduler: physics steps + external wrench sources + periodic tasks.

    sources: evaluated every step from the current state.  slow_sources: wrapped in one SlowForceCache
    refreshed every ``slow_every`` steps.  Periodic tasks ``fn(sim)`` run before the physics step whose
    start time reaches k * period (k = 0, 1, ...), counted in integer steps when period is a multiple of
    dt.  For RK4 models the fast sources run inside a PassiveCallback (installed only during ``step``),
    the slow ones through ``xfrc_applied``.  phase ('contact' | 'free' | None) enables the dt guard.
    """

    def __init__(self, model, data, sources=(), slow_sources=(), slow_every: int = 1, phase: str | None = None,
                 check_every: int = 1000):
        self.model, self.data = model, data
        self.sources = list(sources)
        self.slow = SlowForceCache(slow_sources, slow_every)
        self._tasks = []
        self._k = 0
        self.check_every = max(1, int(check_every))
        self._W = np.zeros((model.nbody, 6))
        self._Wslow = np.zeros((model.nbody, 6))
        if phase is not None:
            check_timestep(model, phase)

    @property
    def steps(self) -> int:
        """Number of physics steps taken by this simulator."""
        return self._k

    def add_source(self, src, slow: bool = False) -> None:
        """Register a wrench source (fast, or inside the slow-force cache)."""
        (self.slow.sources if slow else self.sources).append(src)

    def add_periodic(self, fn, period: float, name: str | None = None) -> None:
        """Run fn(sim) every ``period`` seconds of simulated time (before the physics step)."""
        dt = self.model.opt.timestep
        ratio = period / dt
        every = int(round(ratio))
        if every < 1 or abs(ratio - every) > 1e-6 * max(1.0, ratio):
            raise ValueError(f"period {period} is not an integer multiple of dt {dt}")
        self._tasks.append([fn, every, name or getattr(fn, "__name__", "task")])

    def _run_tasks(self):
        for fn, every, _ in self._tasks:
            if self._k % every == 0:
                fn(self)

    def step(self, n: int = 1) -> None:
        """Advance n physics steps."""
        m, d = self.model, self.data
        rk4 = int(m.opt.integrator) == _RK4
        for _ in range(n):
            self._run_tasks()
            if rk4:
                mujoco.mj_forward(m, d)
                self._Wslow[:] = 0.0
                self.slow.accumulate(m, d, self._Wslow)
                d.xfrc_applied[:] = self._Wslow
                with PassiveCallback(self.sources):
                    mujoco.mj_step(m, d)
            else:
                mujoco.mj_step1(m, d)
                self._W[:] = 0.0
                for s in self.sources:
                    s.accumulate(m, d, self._W)
                self.slow.accumulate(m, d, self._W)
                d.xfrc_applied[:] = self._W
                mujoco.mj_step2(m, d)
            self._k += 1
            if self._k % self.check_every == 0:
                check_finite(d)

    def run_until(self, t_end: float) -> None:
        """Step until data.time >= t_end (within half a step)."""
        dt = self.model.opt.timestep
        n = int(math.floor((t_end - self.data.time) / dt + 0.5))
        if n > 0:
            self.step(n)
