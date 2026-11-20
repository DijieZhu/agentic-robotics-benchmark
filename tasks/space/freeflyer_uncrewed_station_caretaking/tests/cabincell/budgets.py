"""budgets.py -- the per-site budgets of the caretaking cell (robot_api.md, budgets).

Every number is read by key from the runtime copy of the task's numbers (tests/cabincell/data/site_spec.json, checked
against its sha256 file when it is loaded; never a hard-coded hash), never written here:

  sim_time         cell.budget.sim_time_s          the running call and every later one answer TIME_UP; the run ends
  program_compute  cell.budget.program_compute_s   program time (wall clock minus the cell's work) over it: the run ends
  silence          cell.budget.silence_s           no call for this long (wall): the run ends
  colour_frames    cell.budget.colour_frames       camera() of FIC / FWC / AWC answers REFUSED_BUDGET
  depth_frames     cell.budget.depth_frames        camera() of FDC / ADC answers REFUSED_BUDGET
  capture          cell.budget.capture_calls       capture() answers REFUSED_BUDGET (counted when a capture starts)
  fan stops        station.fan_stop.max_per_unit   the station service (REJECTED_QUOTA + a flight-rule violation)
  light requests   station.lights.requests_per_mission   the station service (REJECTED_QUOTA, no violation)
  report           cell.report_counted ('first')   ALREADY_REPORTED after the first valid report
  step             cell.step_max_dt_s              step(dt) advances at most this much
  record           cell.record_period_s, cell.settle_after_exit_s

``Budgets(values, overrides)``: ``overrides`` (tests only: the verifier passes none) replaces single
limits, e.g. ``{"silence_s": 2.0}``.  Standard library only.
"""
from __future__ import annotations

import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RUNTIME = os.path.join(HERE, "data", "site_spec.json")
RUNTIME_SHA = os.path.join(HERE, "data", "site_spec.sha256")
RUNTIME_SCHEMA = "caretaking.site_spec_runtime/1"

KEYS = dict(sim_time_s="cell.budget.sim_time_s", program_compute_s="cell.budget.program_compute_s",
            silence_s="cell.budget.silence_s", colour_frames="cell.budget.colour_frames",
            depth_frames="cell.budget.depth_frames", capture_calls="cell.budget.capture_calls",
            step_max_dt_s="cell.step_max_dt_s", record_period_s="cell.record_period_s",
            settle_after_exit_s="cell.settle_after_exit_s", report_counted="cell.report_counted",
            forbidden_reply="cell.forbidden_reply", light_requests="station.lights.requests_per_mission",
            fan_stops_per_unit="station.fan_stop.max_per_unit")
COLOUR_CAMERAS = ("FIC", "FWC", "AWC")
DEPTH_CAMERAS = ("FDC", "ADC")


class SpecError(RuntimeError):
    """The runtime copy does not match its sha256 file or misses a key."""


def load_runtime(path: str = RUNTIME, sha_path: str = RUNTIME_SHA) -> dict:
    """The runtime document after its sha256 check: {'values', 'sha256', 'version', 'path'}."""
    raw = open(path, "rb").read()
    want = open(sha_path, "rb").read().decode("ascii").split()[0]
    got = hashlib.sha256(raw).hexdigest()
    if got != want:
        raise SpecError(f"{path}: sha256 {got} != {want} (its .sha256 file)")
    doc = json.loads(raw.decode("ascii"))
    if doc.get("schema") != RUNTIME_SCHEMA:
        raise SpecError(f"{path}: schema {doc.get('schema')!r} is not {RUNTIME_SCHEMA}")
    missing = [k for k in KEYS.values() if k not in doc["values"]]
    if missing:
        raise SpecError(f"{path}: missing keys {missing}")
    return dict(values=doc["values"], sha256=got, version=doc.get("contract_version"), path=path)


class Budgets:
    """Limits and counters of one site's run (module doc)."""

    def __init__(self, values: dict, overrides: dict | None = None):
        L = {k: values[key] for k, key in KEYS.items()}
        for k, v in (overrides or {}).items():
            if k not in L:
                raise KeyError(f"unknown budget {k!r} (known: {sorted(L)})")
            L[k] = v
        self.limits = L
        self.n_colour = 0               # counters (the limits are attributes of the same names as the KEYS)
        self.n_depth = 0
        self.n_capture = 0
        self.n_forbidden = 0

    def __getattr__(self, name):
        lim = self.__dict__.get("limits")
        if lim is not None and name in lim:
            return lim[name]
        raise AttributeError(name)

    # -- frame and capture budgets (consumed when the call is carried out) -------------------------------------
    def take_frame(self, camera: str) -> bool:
        """True and one frame used, or False (REFUSED_BUDGET) when the camera's class is used up."""
        if camera in COLOUR_CAMERAS:
            if self.n_colour >= int(self.limits["colour_frames"]):
                return False
            self.n_colour += 1
            return True
        if camera in DEPTH_CAMERAS:
            if self.n_depth >= int(self.limits["depth_frames"]):
                return False
            self.n_depth += 1
            return True
        raise ValueError(camera)

    def take_capture(self) -> bool:
        """The firmware's capture budget hook: True and one capture used, or False (REFUSED_BUDGET)."""
        if self.n_capture >= int(self.limits["capture_calls"]):
            return False
        self.n_capture += 1
        return True

    def forbidden(self) -> None:
        self.n_forbidden += 1

    def used(self, sim_s: float, program_compute_s: float, light_requests: int) -> dict:
        """the record's budgets_used."""
        return dict(sim_s=float(sim_s), program_compute_s=float(program_compute_s), colour_frames=int(self.n_colour),
                    depth_frames=int(self.n_depth), capture_calls=int(self.n_capture),
                    light_requests=int(light_requests), forbidden_calls=int(self.n_forbidden))
