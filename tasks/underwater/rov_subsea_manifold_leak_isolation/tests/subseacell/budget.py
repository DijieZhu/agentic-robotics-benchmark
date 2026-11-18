"""Per-run budgets of the subsea cell (cell side; DESIGN v1.2 section 1.5, decision M12; package P09).

The six limits of one scored run (values from spec/task_values.json `budget.*`, overridable by the grader's
configuration; P24 owns grader_config.json):

  resource            limit (DESIGN 1.5)              unit kept here               enforced by
  sim_time_s          2400 s of simulated time        physics steps (integer)      the server: step() / sonar_scan()
                                                                                   advance only what is left; at 0
                                                                                   the run ends ("sim_cap")
  camera_frame_units  600 units; 640 x 480 = 1.0,     centi-units (integer):       the server: camera_frame() is
                      320 x 240 = 0.40                640 x 480 = 100, 320 = 40    refused once the frame's units
                                                                                   do not fit
  sonar_pings         16 000 pings                    pings (integer)              the server: a sonar_scan() whose
                                                                                   pings do not all fit is refused
  protective_stops    6 (the 6th disables)            stops (integer, mirrored)    the firmware (protect.py) counts
                                                                                   and disables; mirrored here
  program_time_s      1200 s (wall clock minus the    seconds (float)              the wire (cellkit.wire), from the
                      cell's own work)                                             server's work_wall
  silence_s           180 s between a reply and       seconds (float)              the wire (cellkit.wire)
                      the next request

Every charge is atomic under one lock: `take` (all or nothing), `grant` (as much as is left, for the simulated time),
`refund` (a reservation not used, e.g. a frame the cell failed to render, the pings of a scan cut short by the end of
the run).  Integer units make the arithmetic exact, so N concurrent charges can never overspend and the books always
balance (tests/test_server.py: 20 threads).  The program's wall-clock items are not charged here; the cell's own work
time (`add_work`) is accumulated here under the same lock so the wire's program time is exact under concurrency too.

    B = Budget(Limits.from_values(task_values_flat), physics_dt=0.002)
    B.take("frame_units", B.frame_cost((640, 480)))   -> bool
    n = B.grant("sim_steps", 500)                       -> steps granted (<= 500)
    B.snapshot()                                        -> the counters for state()["budget"] (deterministic)
"""
from __future__ import annotations

import json
import math
import os
import threading
from dataclasses import dataclass, field, fields

CENTI = 100                         # frame units are kept in hundredths (0.40 -> 40)

# task_values.json key -> Limits field
VALUE_KEYS = {
    "budget.sim_time_s": "sim_time_s",
    "budget.camera_frame_units": "camera_frame_units",
    "budget.frame_weight_640x480": "frame_weight_640x480",
    "budget.frame_weight_320x240": "frame_weight_320x240",
    "budget.sonar_pings": "sonar_pings",
    "budget.protective_stops": "protective_stops",
    "budget.program_time_s": "program_time_s",
    "budget.silence_s": "silence_s",
}
# grader_config.json "budgets" block (P24) -> Limits field; frame weights as {"640x480": w, "320x240": w}
CONFIG_KEYS = {
    "sim_time_s": "sim_time_s", "sim_s_per_station": "sim_time_s",
    "camera_frame_units": "camera_frame_units",
    "sonar_pings": "sonar_pings",
    "protective_stops": "protective_stops",
    "program_time_s": "program_time_s", "program_s_per_station": "program_time_s",
    "silence_s": "silence_s",
}

# the counters of state()["budget"] (agent-visible key table; P21 prints it)
SNAPSHOT_SCHEMA = (
    ("sim_time_used_s", "float", "s", "simulated time used"),
    ("sim_time_limit_s", "float", "s", "simulated time of the run"),
    ("frame_units_used", "float", "units", "camera frame units used (640 x 480 = 1.0, 320 x 240 = 0.40)"),
    ("frame_units_limit", "float", "units", "camera frame units of the run"),
    ("sonar_pings_used", "int", "-", "sonar pings used"),
    ("sonar_pings_limit", "int", "-", "sonar pings of the run"),
    ("protective_stops_used", "int", "-", "collision protective stops so far"),
    ("protective_stops_limit", "int", "-", "the stop that disables the vehicle"),
)
SNAPSHOT_KEYS = tuple(r[0] for r in SNAPSHOT_SCHEMA)


class BudgetError(ValueError):
    """A budget value that is not exact in the integer units (or not positive)."""


def _centi(x, name):
    v = float(x) * CENTI
    n = int(round(v))
    if abs(v - n) > 1e-6 or n <= 0:
        raise BudgetError(f"{name} = {x!r} is not a positive multiple of 1/{CENTI}")
    return n


@dataclass(frozen=True)
class Limits:
    sim_time_s: float
    camera_frame_units: float
    frame_weight_640x480: float
    frame_weight_320x240: float
    sonar_pings: int
    protective_stops: int
    program_time_s: float
    silence_s: float

    def __post_init__(self):
        for f in fields(self):
            v = getattr(self, f.name)
            if not (isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0):
                raise BudgetError(f"budget {f.name} must be a positive number, got {v!r}")
        _centi(self.camera_frame_units, "camera_frame_units")
        _centi(self.frame_weight_640x480, "frame_weight_640x480")
        _centi(self.frame_weight_320x240, "frame_weight_320x240")
        if int(self.sonar_pings) != self.sonar_pings or int(self.protective_stops) != self.protective_stops:
            raise BudgetError("sonar_pings and protective_stops must be integers")

    @classmethod
    def from_values(cls, values: dict) -> "Limits":
        """From a flat {task_values key: value} mapping; every budget.* key of VALUE_KEYS must be there."""
        missing = [k for k in VALUE_KEYS if k not in values]
        if missing:
            raise KeyError(f"budget values missing: {missing}")
        kw = {f: values[k] for k, f in VALUE_KEYS.items()}
        kw["sonar_pings"] = int(round(float(kw["sonar_pings"])))
        kw["protective_stops"] = int(round(float(kw["protective_stops"])))
        return cls(**kw)

    def override(self, budgets: dict) -> "Limits":
        """The grader configuration's "budgets" block over these limits (unknown keys are an error)."""
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        for k, v in (budgets or {}).items():
            if k in ("frame_unit_weight", "frame_weight"):
                for sz, w in dict(v).items():
                    key = {"640x480": "frame_weight_640x480", "320x240": "frame_weight_320x240"}.get(str(sz))
                    if key is None:
                        raise KeyError(f"unknown frame size {sz!r} in the budget configuration")
                    kw[key] = float(w)
            elif k in CONFIG_KEYS:
                kw[CONFIG_KEYS[k]] = v
            elif k in ("note", "source"):
                continue
            else:
                raise KeyError(f"unknown budget key {k!r}")
        kw["sonar_pings"] = int(round(float(kw["sonar_pings"])))
        kw["protective_stops"] = int(round(float(kw["protective_stops"])))
        return Limits(**kw)

    def as_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def load_task_values(path: str) -> dict:
    """spec/task_values.json -> the flat {key: value} mapping."""
    with open(path, encoding="utf-8") as f:
        vals = json.load(f)["values"]
    return {k: e["value"] for k, e in vals.items()}


def default_values_path() -> str:
    """Where the task values are: $SUBSEACELL_TASK_VALUES, else the development layout (<DEV>/spec/task_values.json
    next to <DEV>/cell/subseacell)."""
    p = os.environ.get("SUBSEACELL_TASK_VALUES")
    if p:
        return p
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "..", "spec", "task_values.json"))


class Budget:
    """Atomic counters of one run (see the module doc).  physics_dt: the cell's physics step (sim time is counted in
    steps, so 2400 s is exact)."""

    def __init__(self, limits: Limits, physics_dt: float):
        self.limits = limits
        self.dt = float(physics_dt)
        n = limits.sim_time_s / self.dt
        if abs(n - round(n)) > 1e-6:
            raise BudgetError("the simulated-time budget is not a whole number of physics steps")
        self._lock = threading.Lock()
        self.cap = {"sim_steps": int(round(n)), "frame_units": _centi(limits.camera_frame_units, "camera_frame_units"),
                    "sonar_pings": int(limits.sonar_pings), "protective_stops": int(limits.protective_stops)}
        self.used = {k: 0 for k in self.cap}
        self.refused = {k: 0 for k in self.cap}
        self.frame_weight = {(640, 480): _centi(limits.frame_weight_640x480, "frame_weight_640x480"),
                             (320, 240): _centi(limits.frame_weight_320x240, "frame_weight_320x240")}
        self._work = 0.0

    # ------------------------------------------------------------------------------------------ charges
    def take(self, kind: str, n: int) -> bool:
        """Charge n units if all of them fit (atomic); False (nothing charged, counted as a refusal) otherwise."""
        n = int(n)
        if n < 0:
            raise ValueError("a charge must not be negative")
        with self._lock:
            if self.used[kind] + n > self.cap[kind]:
                self.refused[kind] += 1
                return False
            self.used[kind] += n
            return True

    def grant(self, kind: str, n: int) -> int:
        """Charge as much of n as is left (atomic); returns the units charged."""
        n = int(n)
        if n < 0:
            raise ValueError("a charge must not be negative")
        with self._lock:
            g = min(n, self.cap[kind] - self.used[kind])
            self.used[kind] += g
            if g < n:
                self.refused[kind] += 1
            return g

    def refund(self, kind: str, n: int) -> None:
        n = int(n)
        with self._lock:
            if n < 0 or n > self.used[kind]:
                raise ValueError("refund larger than the charge")
            self.used[kind] -= n

    def set_used(self, kind: str, n: int) -> None:
        """Mirror a counter kept elsewhere (protective stops: the firmware counts them)."""
        with self._lock:
            self.used[kind] = min(int(n), self.cap[kind])

    def left(self, kind: str) -> int:
        with self._lock:
            return self.cap[kind] - self.used[kind]

    def exhausted(self, kind: str) -> bool:
        return self.left(kind) <= 0

    def frame_cost(self, size) -> int:
        """Centi-units of one frame of `size` (KeyError for a size the camera does not have)."""
        return self.frame_weight[(int(size[0]), int(size[1]))]

    # ------------------------------------------------------------------------------------------ the cell's work time
    def add_work(self, seconds: float) -> None:
        with self._lock:
            self._work += float(seconds)

    @property
    def work_wall(self) -> float:
        with self._lock:
            return self._work

    def reset_work(self, value: float = 0.0) -> None:
        with self._lock:
            self._work = float(value)

    # ------------------------------------------------------------------------------------------ readout
    def sim_time_used_s(self) -> float:
        return self.used["sim_steps"] * self.dt

    def snapshot(self) -> dict:
        """state()["budget"]: counters only (no wall-clock value, so the reply is a function of the run)."""
        with self._lock:
            u, c = dict(self.used), dict(self.cap)
        return {"sim_time_used_s": round(u["sim_steps"] * self.dt, 6), "sim_time_limit_s": float(self.limits.sim_time_s),
                "frame_units_used": u["frame_units"] / CENTI, "frame_units_limit": c["frame_units"] / CENTI,
                "sonar_pings_used": int(u["sonar_pings"]), "sonar_pings_limit": int(c["sonar_pings"]),
                "protective_stops_used": int(u["protective_stops"]), "protective_stops_limit": int(c["protective_stops"])}

    def public_limits(self) -> dict:
        """The run's budgets as the work order states them (mission()["budget"])."""
        L = self.limits
        return {"sim_time_s": float(L.sim_time_s), "program_time_s": float(L.program_time_s),
                "silence_s": float(L.silence_s), "camera_frame_units": float(L.camera_frame_units),
                "frame_weight": {"640x480": float(L.frame_weight_640x480), "320x240": float(L.frame_weight_320x240)},
                "sonar_pings": int(L.sonar_pings), "protective_stops": int(L.protective_stops)}

    def books(self) -> dict:
        """Everything (grader side): used, caps, refusals, work time."""
        with self._lock:
            return {"used": dict(self.used), "cap": dict(self.cap), "refused": dict(self.refused),
                    "work_wall_s": self._work}
