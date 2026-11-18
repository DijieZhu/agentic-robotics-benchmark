"""The grader's configuration: grader_config.json is the only source of every knob (no hidden defaults in code).

    cfg = load(path)                  # the parsed file, checked (ValueError on a missing key)
    sec = section(cfg, "C2")          # one criterion's section
    band_down(err, full, zero)        # 1 at or below full, 0 at or above zero, linear between (errors)
    band_up(val, zero, full)          # 0 at or below zero, 1 at or above full, linear between (values)

A band key holds a two-element list in the order the function takes it: [full, zero] for band_down (an error band,
e.g. C2 pose_band_m [0.03, 0.10]) and [zero, full] for band_up (a value band, e.g. C2 shape_band [0.5, 0.9]).
check() also validates the keys of scoring v1.2: C2.pose_corner_clip_m a positive
number, C6.unit_routes_need "paired" or "none", and C6.truth.max_progress_spread (the truth's keep rule for blocked
routes) not above C6.progress_band[0] (a kept route's own spread must keep full credit, or the oracle could lose it).
"""
import copy
import json
import math

FORMAT = "site-twin-grader-config/1"

# keys the core reads (module agents check their own sections)
REQUIRED = {
    "criteria": ("require_all", "order"),
    "alignment": ("ransac", "refine", "identity_snap", "selftest_sfm"),
    "matching": ("tau_m", "tau_match_m", "f_min", "spacing_m", "max_samples_per_unit", "seed", "pair_bbox_margin_m",
                 "table_min_f"),
    "zones": ("footprint_cell_m", "max_footprint_cells"),
    "C2": ("pose_band_m", "pose_corner_clip_m", "shape_band", "vert_band_m", "registration", "vert", "shape_on"),
    "C6": ("progress_band", "unit_routes_need", "truth"),
    "C7": ("band",),
    "selftest": ("camera_tol_m", "camera_tol_deg", "intr_tol_px", "surface_tol_m", "c3_agree_m", "c3_agree_frac"),
    "gt": ("twin", "cameras_true", "region", "items_truth", "items_surface", "c3_samples"),
    "runtime": ("default_budget_s", "reserve_s", "min_worker_s", "max_worker_s", "worker_margin_s", "worker_account",
                "worker_address_space_mb", "stage_caps_s"),
}
CRITERION_KEYS = ("name", "module", "weight", "truth", "gate")
# C6's condition on unit-decided routes: "paired" = every deciding unit paired with a unit that walks; "none"
UNIT_ROUTES_NEED = ("paired", "none")


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _positive(v):
    return _number(v) and v > 0


def check(cfg):
    """Raise ValueError when a key the core reads is missing or a criterion entry is malformed."""
    if cfg.get("format") != FORMAT:
        raise ValueError("grader_config.json: format must be %r" % FORMAT)
    for sec, keys in REQUIRED.items():
        if not isinstance(cfg.get(sec), dict):
            raise ValueError("grader_config.json: section %r is missing" % sec)
        for k in keys:
            if k not in cfg[sec]:
                raise ValueError("grader_config.json: %s.%s is missing" % (sec, k))
    if cfg["C2"]["shape_on"] not in ("F", "F_registered"):
        raise ValueError("grader_config.json: C2.shape_on must be 'F' or 'F_registered'")
    if not _positive(cfg["C2"]["pose_corner_clip_m"]):
        raise ValueError("grader_config.json: C2.pose_corner_clip_m must be a positive number")
    c6 = cfg["C6"]
    if c6["unit_routes_need"] not in UNIT_ROUTES_NEED:
        raise ValueError("grader_config.json: C6.unit_routes_need must be one of %s" % ", ".join(
            repr(v) for v in UNIT_ROUTES_NEED))
    band = c6["progress_band"]
    if not (isinstance(band, list) and len(band) == 2 and all(_number(v) for v in band) and
            0.0 <= float(band[0]) < float(band[1])):
        raise ValueError("grader_config.json: C6.progress_band must be [full, zero] with 0 <= full < zero")
    spread = c6["truth"].get("max_progress_spread") if isinstance(c6["truth"], dict) else None
    if not _number(spread) or float(spread) < 0.0:
        raise ValueError("grader_config.json: C6.truth.max_progress_spread must be a number >= 0")
    if float(spread) > float(band[0]):
        raise ValueError("grader_config.json: C6.truth.max_progress_spread (%g) is above C6.progress_band[0] (%g): "
                         "a kept blocked route's own spread would cost the true twin credit" % (float(spread),
                                                                                                float(band[0])))
    crit = cfg["criteria"]
    order = crit["order"]
    if len(set(order)) != len(order):
        raise ValueError("grader_config.json: criteria.order repeats a criterion")
    for c in order:
        e = crit.get(c)
        if not isinstance(e, dict) or any(k not in e for k in CRITERION_KEYS):
            raise ValueError("grader_config.json: criteria.%s needs %s" % (c, ", ".join(CRITERION_KEYS)))
        if not (isinstance(e["weight"], (int, float)) and e["weight"] >= 0 and math.isfinite(e["weight"])):
            raise ValueError("grader_config.json: criteria.%s.weight must be a finite number >= 0" % c)
        if e["gate"] is not None and e["gate"] not in order[:order.index(c)]:
            raise ValueError("grader_config.json: criteria.%s.gate must name a criterion run before it" % c)
        for k in e["truth"]:
            if k not in cfg["gt"]:
                raise ValueError("grader_config.json: criteria.%s needs truth %r, which gt does not list" % (c, k))
    return cfg


def load(path):
    with open(path) as f:
        return check(json.load(f))


def section(cfg, name):
    return cfg[name]


def merged(cfg, over):
    """A deep copy of cfg with `over` merged in, one level below each top section (for the dev benches)."""
    out = copy.deepcopy(cfg)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(copy.deepcopy(v))
        else:
            out[k] = copy.deepcopy(v)
    return out


def band_down(err, full, zero):
    """1 at or below full, 0 at or above zero, linear between (smaller errors are better)."""
    err = float(err)
    if not math.isfinite(err) or err >= zero:
        return 0.0                                  # a non-finite error (no measurement) scores 0
    if err <= full:
        return 1.0
    return float((zero - err) / (zero - full))


def band_up(val, zero, full):
    """0 at or below zero, 1 at or above full, linear between (larger values are better)."""
    val = float(val)
    if not math.isfinite(val):
        return 0.0
    if val >= full:
        return 1.0
    if val <= zero:
        return 0.0
    return float((val - zero) / (full - zero))
