"""Executor: the off-board command interface of the robot, tick-based (given/station_api.md, Commands).

    run_primitive(name, bottom_nodes)   one of the 11 published primitives, symmetry-mapped from the PASSED bottom
                                        triple (the cell does not check it against the truth: a wrong triple runs
                                        the wrong mapping, as on the real system). A triple that is not one of the
                                        six rollable faces is refused ('invalid_bottom') and nothing moves.
    set_targets(lengths_mm)             one target shape: six short-sensor lengths (motors 0-5), 100-240 mm,
                                        run by the same PID (tracked band, slack rule), at most 10 s
    stop()                              abort the current command, motors braked
    status()                            what the off-board PID knows: busy, kind, state index, done[6], timeouts
    step(dt)                            the only place where simulated time advances (0.01 <= dt <= 10.0 s, a
                                        multiple of 0.01 s); stepped in control periods of the base (100 Hz)

A primitive is executed exactly as the base's Simulator.run_primitive does in its default execution mode
(ysim 0.3.1): the primitive's own states (start_states with the mode's bands and slack rule), the authors' re-plan
rest phase, then the braked hold `settle_after` of the base's parameters (0.911 s, which the base's settle() steps as
92 control periods = 0.92 s; read from ysim.params, not a constant here); only the symmetry map
comes from the passed triple instead of the true one.
With the true triple passed, the two are bit-identical (test test_executor_matches_base). The base's
start_states(bottom=...) is wrapped, never patched: the triple is checked here before the base sees it, so the
base's own NotRollableFace is never reached from this interface.
"""
import math

from . import world as wmod

ERRORS = ("busy", "invalid_name", "invalid_bottom", "out_of_range", "budget_exhausted", "not_available",
          "bad_argument")
CABLE_MM = (100.0, 240.0)
SET_TARGETS_TIMEOUT_S = 10.0
SETTLE_AFTER_S = None          # None = the base's params value settle_after (0.911 s in ysim 0.3.1)
PRIMITIVE_CAP_S = 120.0        # the base's run_until_done(120.0)
DT_MIN, DT_MAX, DT_QUANTUM = 0.01, 10.0, 0.01


def as_triple(b):
    """Three distinct node numbers 0..5 as a sorted tuple, or None."""
    try:
        t = [int(x) for x in b]
        if len(t) != 3 or any(float(x) != int(x) for x in b):
            return None
    except (TypeError, ValueError):
        return None
    if len(set(t)) != 3 or not all(0 <= x <= 5 for x in t):
        return None
    return tuple(sorted(t))


class Executor:
    def __init__(self, S, t0=None, mode=None):
        ys = wmod.ysim()
        self.ys = ys
        self.S = S
        self.t0 = S.time if t0 is None else float(t0)
        self.mode = mode or ys.params.value("execution_mode")
        self.ctrl_dt = S.ctrl_every * S.dt
        self.cmd = None
        self.history = []          # finished commands
        self.observers = []        # callables obs(executor) after every control period
        self.ticks = 0
        sa = ys.params.value("settle_after") if SETTLE_AFTER_S is None else SETTLE_AFTER_S
        self.settle_after = float(sa)
        # the base's settle(t) steps range(0, round(t / dt), ctrl_every): that many control periods
        self.settle_ticks = len(range(0, int(round(self.settle_after / S.dt)), S.ctrl_every))
        if S.nstep % S.ctrl_every:
            raise RuntimeError("simulator not on a control-period boundary")

    # ---------------------------------------------------------------- time
    def t(self):
        return self.S.time - self.t0

    @property
    def busy(self):
        return self.cmd is not None

    # ---------------------------------------------------------------- commands
    def run_primitive(self, name, bottom_nodes):
        if self.cmd is not None:
            return {"accepted": False, "error": "busy"}
        if not isinstance(name, str) or name not in self.ys.gaits.PRIMITIVES:
            return {"accepted": False, "error": "invalid_name"}
        b = as_triple(bottom_nodes)
        if b is None:
            return {"accepted": False, "error": "bad_argument"}
        if b not in self.ys.gaits.SYMMETRY:
            return {"accepted": False, "error": "invalid_bottom"}
        p = self.ys.gaits.get(name)
        S = self.S
        t_hi, t_lo, rest, slack = S.mode_bands(self.mode)
        S.start_states(p.states(), p.left, p.right, t_hi, t_lo, bottom=b, rest_phase=rest, slack_rule=slack)
        self.cmd = dict(kind="primitive", name=name, bottom=list(b), phase="exec", t_start=self.t(), exec_ticks=0,
                        settle_left=self.settle_ticks, capped=False)
        return {"accepted": True, "error": None}

    def set_targets(self, lengths_mm):
        if self.cmd is not None:
            return {"accepted": False, "error": "busy"}
        try:
            L = [float(x) for x in lengths_mm]
        except (TypeError, ValueError):
            return {"accepted": False, "error": "bad_argument"}
        if len(L) != 6 or not all(math.isfinite(x) for x in L):
            return {"accepted": False, "error": "bad_argument"}
        if not all(CABLE_MM[0] <= x <= CABLE_MM[1] for x in L):
            return {"accepted": False, "error": "out_of_range"}
        prm = self.ys.params
        lo_mm, rng_mm = prm.value("min_length") * 1000.0, prm.value("default_range") * 1000.0
        targets = [(x - lo_mm) / rng_mm for x in L]
        S = self.S
        t_hi, t_lo, _rest, slack = S.mode_bands(self.mode)
        S.start_states([targets], rng_mm, rng_mm, t_hi, t_lo, raw=True, rest_phase=None, slack_rule=slack,
                       timeout=SET_TARGETS_TIMEOUT_S)
        self.cmd = dict(kind="set_targets", lengths_mm=L, phase="exec", t_start=self.t(), exec_ticks=0,
                        settle_left=0, capped=False)
        return {"accepted": True, "error": None}

    def stop(self):
        if self.cmd is not None:
            self._finish(aborted=True)
        self.S.exec = None
        self.S.duty = [0.0] * 6
        return {}

    def status(self):
        ex = self.S.exec
        st = dict(t=round(self.t(), 6), busy=self.busy, kind=None if self.cmd is None else self.cmd["kind"],
                  state_index=None, done=[False] * 6, timeouts=0)
        if self.cmd is not None and ex is not None:
            st["state_index"] = int(ex["k"])
            st["done"] = [bool(x) for x in self.S.pid.done]
            st["timeouts"] = sum(1 for s in ex["log"] if s["timeout"])
        elif self.cmd is not None:
            st["state_index"] = "settle"
            st["timeouts"] = self.cmd.get("timeouts", 0)
        return st

    # ---------------------------------------------------------------- time stepping
    def step(self, dt):
        try:
            dt = float(dt)
        except (TypeError, ValueError):
            return {"error": "bad_argument"}
        n = int(round(dt / DT_QUANTUM))
        if not (DT_MIN - 1e-9 <= dt <= DT_MAX + 1e-9) or abs(n * DT_QUANTUM - dt) > 1e-9:
            return {"error": "bad_argument"}
        k = int(round(dt / self.ctrl_dt))
        for _ in range(k):
            self._tick()
        return {"t": round(self.t(), 6)}

    def run_ticks(self, k):
        """k control periods without the dt validation (internal use by the cell and the reference)."""
        for _ in range(int(k)):
            self._tick()

    def _tick(self):
        S = self.S
        S.step(S.ctrl_every)
        self.ticks += 1
        c = self.cmd
        if c is not None:
            if c["phase"] == "exec":
                c["exec_ticks"] += 1
                finished = S.exec is not None and S.exec["finished"]
                if not finished and c["exec_ticks"] * self.ctrl_dt >= PRIMITIVE_CAP_S - 1e-9:
                    finished, c["capped"] = True, True
                if finished:
                    c["log"] = S.exec["log"] if S.exec is not None else []
                    c["timeouts"] = sum(1 for s in c["log"] if s["timeout"])
                    if c["settle_left"] > 0:
                        c["phase"] = "settle"
                        S.exec = None            # the base's settle(): braked, no command
                    else:
                        self._finish()
            elif c["phase"] == "settle":
                c["settle_left"] -= 1
                if c["settle_left"] <= 0:
                    self._finish()
        for obs in self.observers:
            obs(self)

    def _finish(self, aborted=False):
        c = self.cmd
        c["t_end"] = self.t()
        c["aborted"] = bool(aborted)
        if "log" not in c:
            c["log"] = self.S.exec["log"] if self.S.exec is not None else []
            c["timeouts"] = sum(1 for s in c["log"] if s["timeout"])
        rest = [s for s in c["log"] if s.get("phase") == "rest"]
        c["rest_reached"] = bool(rest and not rest[-1]["timeout"]) if c["kind"] == "primitive" else None
        c["n_states"] = len(c["log"])
        c.pop("log", None)
        self.history.append(c)
        self.cmd = None
        if self.S.exec is not None and self.S.exec.get("finished"):
            self.S.exec = None

    def wait_idle(self, max_s=200.0):
        """Step control periods until no command runs (reference / measurement helper)."""
        k = 0
        kmax = int(round(max_s / self.ctrl_dt))
        while self.cmd is not None and k < kmax:
            self._tick()
            k += 1
        return self.cmd is None
