"""Configuration of the hidden world. Every shared number comes from the task's table of numbers by key (the world's
copy, world_table.json, holds the values the world reads); the world's own parameters come from PARAMETERS.json with
SENSOR_PARAMETERS.json laid over it, and the acceptance sessions' inputs from SESSION_INPUTS.json. Every path is an
argument of configure(); no number of the contract is typed here.



















"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DEFAULT_PARAMS = os.path.join(HERE, "PARAMETERS.json")
DEFAULT_SENSOR_PARAMS = os.path.join(HERE, "SENSOR_PARAMETERS.json")
DEFAULT_SESSION_INPUTS = os.path.join(HERE, "SESSION_INPUTS.json")

_cfg = {"spec": None, "sessions_dir": None, "key": None, "params": DEFAULT_PARAMS, "sensor_params": DEFAULT_SENSOR_PARAMS,
        "scenarios_dir": None, "session_inputs": DEFAULT_SESSION_INPUTS}
_spec = None
_params = None

SIGNED = ["+x", "-x", "+y", "-y", "+z", "-z"]


def configure(spec=None, sessions_dir=None, key=None, params=None, sensor_params=None, scenarios_dir=None, session_inputs=None):
    """set the paths the world reads (each an argument of the runner). Calling it again replaces the given ones and
    drops the cached files. sensor_params="" switches the overlay off. scenarios_dir and session_inputs:
    the hidden set's scenario files and the acceptance sessions' generated inputs."""
    global _spec, _params
    for k, v in (("spec", spec), ("sessions_dir", sessions_dir), ("key", key), ("params", params), ("sensor_params", sensor_params),
                 ("scenarios_dir", scenarios_dir), ("session_inputs", session_inputs)):
        if v is not None:
            _cfg[k] = os.path.abspath(v) if v != "" else ""
    _spec = None
    _params = None


def optional_path(name):
    """a configured path that may be unset (None)."""
    return _cfg.get(name)


def path(name):
    p = _cfg[name]
    if p is None:
        raise RuntimeError("world.config.configure(%s=...) has not been called" % name)
    return p


def load_spec():
    global _spec
    if _spec is None:
        with open(path("spec")) as f:
            _spec = json.load(f)["values"]
    return _spec


READ_KEYS = set()          # every table key read through V (pending_read() reports the pending ones)


def V(key):
    """value of a table entry (raises if missing or null)."""
    READ_KEYS.add(key)
    v = load_spec()[key]["value"]
    if v is None:
        raise KeyError("table entry %s is pending without a value" % key)
    return v


def pending_read():
    """the table entries read so far whose status is pending, with the status (what the world runs with before its owner
    sets them)."""
    sp = load_spec()
    return {k: sp[k]["status"] for k in sorted(READ_KEYS) if str(sp[k].get("status", "")).startswith("pending")}


def params():
    global _params
    if _params is None:
        with open(path("params")) as f:
            base = json.load(f)
        sp = _cfg.get("sensor_params")
        if sp and os.path.exists(sp):
            # the sensor numbers fill the entries left pending; an entry set in PARAMETERS.json wins
            with open(sp) as f:
                over = json.load(f)["parameters"]
            filled = []
            for k in sorted(over):
                e = base["parameters"].get(k)
                if e is None or e.get("value") is None:
                    m = dict(e or {})
                    m.update(over[k])
                    base["parameters"][k] = m
                    filled.append(k)
            base["sensor_parameters_overlay"] = {"file": os.path.basename(sp), "filled": filled}
        _params = base
    return _params


inputs = params          # another name of params(), the world's parameters (PARAMETERS.json)


def param(name):
    """the value of one world parameter of PARAMETERS.json; a pending one raises with the part that sets it."""
    e = params()["parameters"][name]
    if e.get("status", "").startswith("pending") and e.get("value") is None:
        raise KeyError("world parameter %s is %s" % (name, e["status"]))
    return e["value"]


def key_bytes():
    with open(path("key"), "rb") as f:
        return f.read()


def no_numbers_from_file():
    """an empty dict for code that asks: the world reads no such numbers from a file."""
    return {}


def torque_gains(level):
    if level == 1:
        assert V("world.torque_gain.level1") == "equal to level 2"
        level = 2
    g = V("world.torque_gain.level%d" % level)
    return [float(g[s]) for s in SIGNED]


def force_gains(level, inp=None):
    """world.force_gain.rule applied to the level-3 calibration:
    per signed axis the calibrated value where its leave-one-day-out sd is at most the rule's line, else the era's
    pooled ratio; levels 2 and 1 use the level-2 pooled ratio on every axis. The values are in PARAMETERS.json
    (force_gain); a maintainers' tool derives them and the tests check the derivation."""
    inp = inp or params()
    fg = inp["parameters"]["force_gain"]["value"]
    if level in (1, 2):
        return [float(fg["level2_and_level1"][s]) for s in SIGNED]
    return [float(fg["level3"][s]) for s in SIGNED]
