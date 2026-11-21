"""Grader configuration, the cell's parameters and the public tables derived from them.

`grader_config.json` (this folder) is the file the cell reads; its key names are frozen by
`examples/records.schema.json#/$defs/config`. `referee_params.json` holds the cell's knobs that the frozen
configuration has no key for.
"""

import copy
import hashlib
import json
import os

from sketch.examples import check
from sketch.examples import constants as K

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "grader_config.json")
PARAMS_PATH = os.path.join(HERE, "referee_params.json")


def load_config(path=CONFIG_PATH):
    cfg = check.load_json(path)
    err = check.check_record("config", cfg)
    if err:
        raise ValueError("grader configuration %s: %s" % (path, "; ".join(err)))
    return cfg


def load_params(path=PARAMS_PATH):
    return check.load_json(path)


def config_sha256(cfg):
    """sha256 of the canonical text of the configuration."""
    return hashlib.sha256(check.canonical_json(cfg).encode("ascii")).hexdigest()


def session_ticks(cfg, ses):
    return int(round(ses["duration_s"] / cfg["sim"]["tick_s"]))


def session_table(cfg):
    """`describe().sessions`: name, duration_s, ticks, desk_before_s, scored, phases."""
    return [{"name": s["name"], "duration_s": s["duration_s"], "ticks": session_ticks(cfg, s),
             "desk_before_s": s["desk_before_s"], "scored": s["scored"], "phases": copy.deepcopy(s["phases"])}
            for s in cfg["sessions"]]


def car_table(cfg):
    """`describe().car`: the datasheet's numbers."""
    return {"wheelbase_m": K.WHEELBASE_M, "track_front_m": K.TRACK_FRONT_M, "track_rear_m": K.TRACK_REAR_M,
            "tyre_radius_m": K.TYRE_RADIUS_M, "mass_kg": K.MASS_KG, "com_m": list(K.COM_C),
            "length_m": K.ENVELOPE_M[0], "width_m": K.ENVELOPE_M[1], "height_m": K.ENVELOPE_M[2],
            "steer_max_rad": cfg["command"]["steer_max_rad"], "steer_rate_max_radps": K.STEER_RATE_MAX_RADPS,
            "min_turn_radius_m": K.MIN_TURN_RADIUS_M, "final_drive_ratio": K.FINAL_DRIVE_RATIO,
            "speed_cmd_min_mps": cfg["command"]["speed_min_mps"], "speed_cmd_max_mps": cfg["command"]["speed_max_mps"],
            "lidar": {"pos_m": list(K.LIDAR_POS_C), "beams": K.N_BEAMS, "angle_min_rad": round(K.ANGLE_MIN_RAD, 9),
                      "angle_inc_rad": round(K.ANGLE_INC_RAD, 9), "range_min_m": K.RANGE_MIN_M,
                      "range_max_m": K.RANGE_MAX_M, "no_return": K.NO_RETURN},
            "imu": {"pos_m": list(K.IMU_POS_C), "rate_hz": K.IMU_RATE_HZ}}


def describe_result(world, cfg):
    """The whole `describe` result: the venue sheet verbatim and the public tables."""
    return {"protocol": K.WIRE_PROTOCOL, "venue_sheet": check.public_view(world), "sessions": session_table(cfg),
            "budgets": {k: cfg["budgets"][k] for k in ("program_ms_per_tick", "cpu_factor", "silence_s")},
            "timing": {"tick_s": cfg["sim"]["tick_s"], "cmd_delay_ticks": cfg["sim"]["cmd_delay_ticks"],
                       "imu_samples_per_tick": cfg["sim"]["imu_samples_per_tick"],
                       "subscans_per_tick": cfg["sim"]["subscans_per_tick"]},
            "car": car_table(cfg)}


def program_caps(cfg):
    """Program-time caps in seconds: {"track": {session: cap}, "desk": {session: cap}, "cpu": total CPU cap}."""
    track = {s["name"]: cfg["budgets"]["program_ms_per_tick"] * 1e-3 * session_ticks(cfg, s) for s in cfg["sessions"]}
    desk = {s["name"]: float(s["desk_before_s"]) for s in cfg["sessions"]}
    cpu = cfg["budgets"]["cpu_factor"] * (sum(track.values()) + sum(desk.values()))
    return {"track": track, "desk": desk, "cpu": cpu}


def dumps(obj):
    return json.dumps(obj, indent=1) + "\n"
