"""Grader configuration of the head-to-head day and the record schemas of the heat cell.

`grader_config.json` (this folder) is the configuration the cell reads: the shared cell's keys for the simulation,
the budgets, the command limits, the staff and the route gate; the three sessions practice, heat1, heat2; the `heat`
block of the heat rules; the `scoring` block of the score. `records.schema.json` (this folder) holds the schemas of
the configuration (`config`) and of the venue result (`result`).

    cfg = load_config()                 # parsed and checked; ValueError with every error otherwise
    check_config(cfg) -> [error, ...]   # schema and the rules below
    heat_sessions(cfg)                  # ("heat1", "heat2"): the sessions whose model has the field
"""

import json
import os

from sketch.examples import check

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "grader_config.json")
SCHEMA_PATH = os.path.join(HERE, "records.schema.json")

CONFIG_SCHEMA = "racecar-grader-config-h2h/1"
RESULT_SCHEMA = "racecar-venue-result-h2h/1"
SESSIONS = ("practice", "heat1", "heat2")
HEATS = ("heat1", "heat2")
RACE_PHASE = "race"
CAR_ID = "car"                          # the id of the car in a heat record (field cars carry their world-file id)
END_REASONS = ("laps", "time", "warnings", "void", "closed", "aborted")
WIRE_END_REASONS = ("time", "laps", "warnings", "void")


def load_schema(path=SCHEMA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def record_errors(kind, obj, schema=None):
    """Schema errors of a record: kind 'config' or 'result' (or any definition of the schema file)."""
    s = schema or load_schema()
    if kind not in s["$defs"]:
        return ["unknown record kind %r" % kind]
    sub = {"$schema": s["$schema"], "$ref": "#/$defs/" + kind, "$defs": s["$defs"]}
    return check.schema_errors(obj, sub)


def check_config(cfg):
    """Errors of a grader configuration of the head-to-head day (empty when it is usable)."""
    err = record_errors("config", cfg)
    if err:
        return err
    sim = cfg["sim"]
    if abs(sim["physics_dt_s"] * sim["physics_steps_per_tick"] - sim["tick_s"]) > 1e-12:
        err.append("sim: physics_dt_s x physics_steps_per_tick is not tick_s")
    names = [s["name"] for s in cfg["sessions"]]
    if names != list(SESSIONS):
        err.append("sessions are not %s in this order" % (SESSIONS,))
    for ses in cfg["sessions"]:
        t0 = [p["t0_s"] for p in ses["phases"]]
        if t0[0] != 0.0 or t0 != sorted(t0) or t0[-1] >= ses["duration_s"]:
            err.append("session %s: phases must start at 0, ascend and begin before the session ends" % ses["name"])
        if abs(round(ses["duration_s"] / sim["tick_s"]) * sim["tick_s"] - ses["duration_s"]) > 1e-9:
            err.append("session %s: duration is not a whole number of ticks" % ses["name"])
        if ses["name"] in HEATS:
            if [(p["name"], p["speed_limit_mps"]) for p in ses["phases"]] != [(RACE_PHASE, None)] or not ses["scored"]:
                err.append("session %s: a heat is scored and has the one phase %r without a speed limit" % (ses["name"], RACE_PHASE))
        elif ses["scored"] or any(p["name"] == RACE_PHASE for p in ses["phases"]):
            err.append("session %s: practice is not scored and has no %r phase" % (ses["name"], RACE_PHASE))
    w = cfg["scoring"]["heat"]
    if abs(w["place_weight"] + w["pass_weight"] - 1.0) > 1e-9:
        err.append("scoring.heat: place_weight and pass_weight do not sum to 1")
    if w["warnings_zero_at"] != cfg["staff"]["warnings_end_session"]:
        err.append("scoring.heat.warnings_zero_at is not staff.warnings_end_session (the warning that ends a heat zeroes it)")
    h = cfg["heat"]
    if h["contact_merge_s"] > h["clean_before_s"] + h["clean_after_s"]:
        err.append("heat: contact_merge_s is longer than the clean window")
    every = h["car_crash_sample_s"] / sim["physics_dt_s"]
    if abs(every - round(every)) > 1e-9 or round(every) < 1:
        err.append("heat: car_crash_sample_s is not a whole number of physics steps")
    n = h["car_crash_window_s"] / h["car_crash_sample_s"]
    if abs(n - round(n)) > 1e-9 or round(n) < 1:
        err.append("heat: car_crash_window_s is not a whole number of samples (car_crash_sample_s)")
    return err


def load_config(path=CONFIG_PATH):
    cfg = check.load_json(path)
    err = check_config(cfg)
    if err:
        raise ValueError("grader configuration %s: %s" % (os.path.basename(path), "; ".join(err)))
    return cfg


def is_h2h(cfg):
    """True for a configuration of the head-to-head day (it has the heat block)."""
    return isinstance(cfg, dict) and cfg.get("schema") == CONFIG_SCHEMA and "heat" in cfg


def heat_sessions(cfg):
    """Names of the sessions that are heats, in the configuration's order (none for any other configuration)."""
    if not is_h2h(cfg):
        return ()
    return tuple(s["name"] for s in cfg["sessions"] if s["name"] in HEATS)


__all__ = ["CONFIG_PATH", "SCHEMA_PATH", "CONFIG_SCHEMA", "RESULT_SCHEMA", "SESSIONS", "HEATS", "CAR_ID", "END_REASONS",
           "WIRE_END_REASONS", "load_config", "check_config", "record_errors", "load_schema", "is_h2h", "heat_sessions"]
