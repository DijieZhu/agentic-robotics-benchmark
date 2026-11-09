"""Knobs of the twin grader, part 1 (alignment, inventory C1, pose and shape C2, environment C3, scale C6).

verify.py may pass a dict that overrides any of these (merged one level deep); every number the contract states
lives here or in the public twin_gate / twin_sim rules.
"""
import copy

DEFAULTS = {
    "frame_alignment": "principal_ray_then_scale",   # or "scale_only"
    "principal_ray_clamp_deg": 2.0,
    "scale": {"bearing_deg": 3.0, "min_pairs": 3, "trim_mm": 30.0, "iterations": 4},
    "match": {"iou_min": 0.25, "grid_mm": 3.0, "fine_grid_mm": 1.0, "thin_mm": 40.0,
              "max_cells": 20_000_000, "volume_ratio_skip": 0.2, "cap": "extents_after_icp"},
    "pose": {"pos_mm": [4.0, 15.0], "rot_deg": [3.0, 15.0], "iou": [0.40, 0.85], "combine": "product",
             "icp_samples": 4000, "icp_max_deg": 30.0, "icp_iterations": 40, "icp_trim": 0.9,
             "icp_step_rad": 0.15, "icp_fit_scale": True, "icp_scale_range": [0.5, 2.0], "sym_iou": 0.90,
             "seed": 20261105},
    "environment": {"tol_mm": 15.0, "stride": 2, "cos_min": 0.25, "vertical_deg": [2.0, 8.0],
                    "w_depth": 0.75, "w_vertical": 0.25, "cutoff_m": 1000.0,
                    "selftest_tol_mm": 1.0, "selftest_frac": 0.999, "scale_choice": "best_of_sstar_and_own",
                    "class_balance": True},
    "scale_band": {"band": [0.10, 0.30], "min_pairs": 5},
    # work areas (twin/regions.py): of_item maps every true item to its area (the task's grader_config.json);
    # of_supporter maps the true furniture supporters of the support truth to theirs (checked in part 2)
    "regions": {"balanced": False, "of_item": {}, "of_supporter": {}},
    # C3 and C6 multiplied by C1
    "gating": {"scene_by_inventory": True},
}


def merged(over=None):
    cfg = copy.deepcopy(DEFAULTS)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def band_down(err, full, zero):
    """1 at or below full, 0 at or above zero, linear between (smaller errors are better)."""
    err = float(err)
    if err <= full:
        return 1.0
    if err >= zero:
        return 0.0
    return float((zero - err) / (zero - full))


def band_up(val, zero, full):
    """0 at or below zero, 1 at or above full, linear between (larger values are better)."""
    val = float(val)
    if val >= full:
        return 1.0
    if val <= zero:
        return 0.0
    return float((val - zero) / (full - zero))
