"""The metric whitelist: the only keys the verdict may carry, each a finite number. No truth value (no range, no stop
distance, no move list) and no clock value is among them.

    m = metrics_of(grade, stats, flights)      # {key: float} on the whitelist, finite
"""
import math

# keys of the verify-start lockdown and the worker (verify.py) that do not depend on the clock or the machine
LOCKDOWN = ("agent_processes_killed", "agent_processes_left", "agent_processes_running", "agent_processes_frozen",
            "stage_locked", "stage_parent_locked", "submission_late_writes", "late_write_gate_active",
            "submission_snapshot", "submission_root_rejected", "submission_root_violation", "submission_oversize",
            "submission_irregular", "worker_deprivileged", "G1")
TOTAL = ("twin_score", "twin_score_sum", "twin_score_product", "aggregation_is_product",
         "key_frames_named", "hazard_aggregation_is_poses", "hazard_aggregation_is_clean_poses",
         "depth_aggregation_is_product")
PER_FLIGHT = ("G0", "Z3", "Z7", "score", "sum", "product", "D", "Dc", "Ds", "H", "J", "aC", "aBC", "aBS",
              "valid_poses", "inliers", "frames_z4", "frames_z5", "frames_z6", "frames_scored",
              "D_weighted", "D_product", "H_moves", "H_poses", "H_clean_poses", "key_poses", "key_poses_kept",
              "key_poses_kept_clean", "entries_without_valid_pose")


def whitelist(flights):
    keys = set(LOCKDOWN) | set(TOTAL)
    for f in flights:
        keys |= {"%s_%s" % (f, k) for k in PER_FLIGHT}
    return keys


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def metrics_of(grade, stats, flights):
    """The verdict's metrics from the grade (grade.grade_submission's result or None) and the lockdown stats."""
    out = {}
    for k in LOCKDOWN:
        if k in stats:
            out[k] = stats[k]
    if grade is not None:
        out["twin_score"] = grade["twin_score"]
        out["twin_score_sum"] = grade["twin_score_sum"]
        out["twin_score_product"] = grade["twin_score_product"]
        out["aggregation_is_product"] = 1.0 if grade["aggregation"] == "product" else 0.0
        out["key_frames_named"] = 1.0 if grade["key_frames_named"] else 0.0
        out["hazard_aggregation_is_poses"] = 1.0 if grade["hazard_aggregation"] == "min_of_moves_and_poses" else 0.0
        out["hazard_aggregation_is_clean_poses"] = (1.0 if grade["hazard_aggregation"] == "min_of_moves_and_clean_poses"
                                                    else 0.0)
        out["depth_aggregation_is_product"] = 1.0 if grade["depth_aggregation"] == "product" else 0.0
        for name in flights:
            f = grade["flights"][name]
            row = dict(f["terms"])
            row.update({"G0": f["G0"], "Z3": 1.0 if f["z3"] else 0.0, "Z7": 1.0 if f["z7"] else 0.0,
                        "valid_poses": f["counts"]["valid_poses"], "inliers": f["counts"]["inliers"],
                        "frames_z4": f["counts"]["z4"], "frames_z5": f["counts"]["z5"],
                        "frames_z6": f["counts"]["z6"], "frames_scored": f["counts"]["scored_frames"],
                        "key_poses": f["counts"]["key_poses"], "key_poses_kept": f["counts"]["key_poses_kept"],
                        "key_poses_kept_clean": f["counts"]["key_poses_kept_clean"],
                        "entries_without_valid_pose": f["counts"]["entries_without_valid_pose"]})
            for k in PER_FLIGHT:
                out["%s_%s" % (name, k)] = row[k]
    allowed = whitelist(flights)
    clean = {}
    for k in sorted(out):
        v = _f(out[k])
        if k in allowed and v is not None:
            clean[k] = v
    return clean
