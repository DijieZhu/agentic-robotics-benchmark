"""`others_params.json`: the parameters of boxes, slow cars and their placement.

    from sketch.others.params import load_params
    p = load_params()                 # the file next to this module; load_params(path) or load_params(dict) also work
"""

import copy
import hashlib
import json
import os

PARAMS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "others_params.json")
SCHEMA = "racecar-others-params/1"
FROZEN_KEYS = ("box_floor_friction", "slow_car_mode", "tail_box_centre_m", "entry_min_gap_m")
MODES = ("dynamic", "mocap")


def check_params(p):
    """Errors of a parameter dict (empty list when it is usable)."""
    err = []
    if p.get("schema") != SCHEMA:
        err.append("schema is %r, not %r" % (p.get("schema"), SCHEMA))
    for k in FROZEN_KEYS:
        if k not in p:
            err.append("frozen key %r is missing (CONTRACT 8.2)" % k)
    if p.get("slow_car_mode") not in MODES:
        err.append("slow_car_mode must be one of %s" % (MODES,))
    if len(p.get("tail_box_centre_m", [])) != 3:
        err.append("tail_box_centre_m is a point [x, y, z] in the car frame")
    b = p.get("box", {})
    for i, (lo, hi) in enumerate(b.get("size_range_m", [])):
        if not 0.1 <= lo <= hi <= (0.35, 0.32, 0.30)[i]:
            err.append("box.size_range_m[%d] %s is outside the world schema's 0.1 .. %.2f" % (i, [lo, hi], (0.35, 0.32, 0.30)[i]))
    lo, hi = b.get("mass_range_kg", [0.3, 0.6])
    if not 0.3 <= lo <= hi <= 0.6:
        err.append("box.mass_range_kg must lie inside 0.3 .. 0.6 (DESIGN 4.1)")
    lo, hi = p.get("slow_cars", {}).get("speed_factor_range", [0.35, 0.5])
    if not 0.35 <= lo <= hi <= 0.5:
        err.append("slow_cars.speed_factor_range must lie inside 0.35 .. 0.50 (DESIGN 4.1)")
    return err


def load_params(source=None):
    """The parameters as a dict: from `source` (a path or a dict) or from the file next to this module."""
    if isinstance(source, dict):
        p = copy.deepcopy(source)
    else:
        with open(source or PARAMS_PATH, "r", encoding="utf-8") as f:
            p = json.load(f)
    err = check_params(p)
    if err:
        raise ValueError("others_params: " + "; ".join(err))
    return p


def params_digest(p):
    """sha256 of the canonical text of a parameter dict."""
    return hashlib.sha256(json.dumps(p, sort_keys=True, separators=(",", ":")).encode("ascii")).hexdigest()
