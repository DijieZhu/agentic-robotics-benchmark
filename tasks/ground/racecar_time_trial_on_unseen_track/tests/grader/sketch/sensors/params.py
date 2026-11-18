"""The sensor parameter file `sensor_params.json` (the contract: its frozen keys) and overrides for tests."""

import copy
import json
import os

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sensor_params.json")
SCHEMA = "racecar-sensor-params/1"

# reflectivity classes of LiDAR targets, in the order of the class index used inside the LiDAR
TARGET_CLASSES = ("orange_ribbed", "yellow_ribbed", "black_plain", "normal", "dark", "cardboard")
FLOOR_RETURN_KINDS = ("painted_concrete", "sealed_concrete", "polished_concrete", "exhibition_carpet",
                      "dust", "metal_cover", "carpet_tile")

# keys other packages may read; a test keeps them present
FROZEN_KEYS = (
    ("lidar", "range_by_class_m"), ("lidar", "noise_sigma_m"), ("lidar", "floor_return_prob"),
    ("imu", "accel_noise_mps2"), ("imu", "gyro_noise_radps"), ("imu", "accel_bias_mps2"), ("imu", "gyro_bias_radps"),
    ("wheel_speed", "noise_mps"),
)
NUMERIC = {
    "lidar": ("range_fade_frac", "noise_sigma_m", "floor_shallow_max_rad", "floor_range_m"),
    "imu": ("accel_noise_mps2", "gyro_noise_radps", "accel_bias_mps2", "gyro_bias_radps",
            "accel_bias_walk_mps2_per_sqrt_s", "gyro_bias_walk_radps_per_sqrt_s",
            "accel_vibration_mps2_per_mps", "gyro_vibration_radps_per_mps", "accel_range_mps2", "gyro_range_radps"),
    "wheel_speed": ("noise_mps",),
}


def _merge(base, over):
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def validate(p):
    """List of errors of a parameter dict (empty = valid)."""
    err = []
    if p.get("schema") != SCHEMA:
        err.append("schema must be %r" % SCHEMA)
    for sec, key in FROZEN_KEYS:
        if key not in p.get(sec, {}):
            err.append("missing frozen key %s.%s" % (sec, key))
    for sec, keys in NUMERIC.items():
        for key in keys:
            v = p.get(sec, {}).get(key)
            if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
                err.append("%s.%s must be a number >= 0" % (sec, key))
    lid = p.get("lidar", {})
    rbc = lid.get("range_by_class_m", {})
    for c in TARGET_CLASSES:
        v = rbc.get(c)
        if not isinstance(v, (int, float)) or not 0.0 < v <= 10.0:
            err.append("lidar.range_by_class_m.%s must be in (0, 10]" % c)
    extra = sorted(set(rbc) - set(TARGET_CLASSES))
    if extra:
        err.append("lidar.range_by_class_m has unknown classes %s" % extra)
    frp = lid.get("floor_return_prob", {})
    for k in FLOOR_RETURN_KINDS:
        v = frp.get(k)
        if not isinstance(v, (int, float)) or not 0.0 <= v <= 1.0:
            err.append("lidar.floor_return_prob.%s must be in [0, 1]" % k)
    extra = sorted(set(frp) - set(FLOOR_RETURN_KINDS))
    if extra:
        err.append("lidar.floor_return_prob has unknown kinds %s" % extra)
    return err


def load_params(overrides=None, path=PARAMS_PATH):
    """The parameters of `path` with `overrides` (a nested dict) merged in; raises ValueError if invalid."""
    with open(path, "r", encoding="utf-8") as f:
        p = json.load(f)
    if overrides:
        p = _merge(copy.deepcopy(p), copy.deepcopy(overrides))
    err = validate(p)
    if err:
        raise ValueError("sensor parameters: " + "; ".join(err))
    return p


def noiseless(**more):
    """Overrides that switch every random effect off (for exact tests): no noise, no bias, no fade, floor returns 0 / 1."""
    o = {"lidar": {"noise_sigma_m": 0.0, "range_fade_frac": 0.0},
         "imu": {k: 0.0 for k in ("accel_noise_mps2", "gyro_noise_radps", "accel_bias_mps2", "gyro_bias_radps",
                                  "accel_bias_walk_mps2_per_sqrt_s", "gyro_bias_walk_radps_per_sqrt_s",
                                  "accel_vibration_mps2_per_mps", "gyro_vibration_radps_per_mps")},
         "wheel_speed": {"noise_mps": 0.0}}
    return _merge(o, more)
