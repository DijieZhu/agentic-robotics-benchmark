"""Scoring of the motion-controller task (`motion-score/1`): per job, then per episode.

    job = completed x no_contact x clearance x conditioning x accuracy x (0.6 + 0.2 time + 0.2 smoothness)

(time and smoothness only count for a tool point that got where it had to: an arm that does not move scores 0, a precise
but slow one keeps 0.6 + 0.2 smoothness of its accuracy)

  completed     the controller said done within the job's time cap and no protective stop happened
  no_contact    any contact of the arm, the clamped part or the needle with the cell or itself -> 0
  clearance     the smallest distance to the TRUE geometry during the job (exemptions skipped): >= 10 mm -> 1,
                0-10 mm -> linear from 0.5 to 1
  conditioning  1 - 0.5 x the fraction of motion cycles in which the smallest singular value of the scaled task
                Jacobian (plant.conditioning) is below COND_MIN
  accuracy      reach: L(position) x L(approach) x L(jaw) (no jaw factor without a jaw axis), measured
                SETTLE_S after done; path: L(RMS tip deviation from p(t - t_start)) x (1 - fraction of path samples
                with the needle axis outside its cone) x progress, progress = the tip's motion projected on the
                reference's motion over the reference's own, clipped to [0, 1] (a needle that does not trace the path
                scores 0)
  time          1 up to the time target, linear to 0 at 2x
  smoothness    reach: 0.5 L(drift during the settle) + 0.5 L(overshoot); path: L(peak tip deviation)
  L(e; full, zero) = clip(log(zero / e) / log(zero / full), 0, 1): a log band

Episode = 0.5 mean(reach jobs) + 0.4 mean(path jobs) + 0.1 J (a category with no job is left out and the weights are
renormalised), where a feasible job the controller refused scores 0 and
J = (correct refusals / infeasible jobs) - (refusals of feasible jobs / feasible jobs), clipped at 0 (Youden's J).
"""

import math

import numpy as np

SCORE_FORMAT = "motion-score/1"
WEIGHTS = {"reach": 0.5, "path": 0.4, "infeasibility": 0.1}
# Bands (revision 5, 2026-11-16; measured on the hidden set of three units): the full ends
# sit just above what the privileged oracle reaches on a perfectly known unit (position max 1.10 mm, path RMS max
# 1.98 mm, peak max 3.69 mm: tick quantisation, hysteresis, tracking), the zero ends just above the middle of what the
# documents alone give (the calibration record's +-0.5 deg, the geometry tolerances and the compliance spread leave
# position p50 2.0 / p90 3.7 mm, path RMS p50 2.3 / p90 2.8 mm on a documents-ideal controller). The acceptance asks
# for the accuracy of a well-calibrated arm: a controller that only follows the documents scores about 0.5 (the
# documents-ideal reference 0.50; with revision 4's bands, 4 -> 40 mm, it scored 0.84).
BANDS = {"pos_mm": (1.2, 3.0), "app_deg": (1.5, 12.0), "jaw_deg": (2.0, 15.0),
         "path_rms_mm": (2.2, 4.0), "path_peak_mm": (4.0, 7.0),
         "settle_mm": (0.3, 6.0), "overshoot_mm": (1.0, 20.0)}
CLEAR_REQ_MM = 10.0
COND_MIN = 0.008               # m/rad: about an elbow within 15-20 deg of straight, or the tool over the base axis
SETTLE_S = 0.5


def L(e, band):
    full, zero = band
    e = float(e)
    if not math.isfinite(e):
        return 0.0
    if e <= full:
        return 1.0
    if e >= zero:
        return 0.0
    return math.log(zero / e) / math.log(zero / full)


def clearance_factor(min_clear_m):
    c = float(min_clear_m) * 1e3
    if c >= CLEAR_REQ_MM:
        return 1.0
    return 0.5 + 0.5 * max(0.0, c) / CLEAR_REQ_MM


def conditioning_factor(conds):
    if not conds:
        return 1.0
    below = sum(1 for c in conds if c < COND_MIN)
    return 1.0 - 0.5 * below / len(conds)


def time_factor(t_used, t_target):
    if t_used <= t_target:
        return 1.0
    return max(0.0, 1.0 - (t_used - t_target) / t_target)


def reach_accuracy(err):
    """Position x approach x jaw axis: a tool in the right place pointing the wrong way has not reached the pose."""
    a = L(err["pos_mm"], BANDS["pos_mm"]) * L(err["app_deg"], BANDS["app_deg"])
    if err.get("jaw_deg") is not None:
        a *= L(err["jaw_deg"], BANDS["jaw_deg"])
    return a


def job_score(rec):
    """rec: the grader's record of one feasible job (grade.run_job). Returns (score, parts)."""
    parts = {"completed": 1.0 if rec.get("completed") else 0.0, "no_contact": 0.0 if rec.get("contact") else 1.0}
    if rec.get("refused"):
        parts["refused"] = 1.0
        return 0.0, parts
    if parts["completed"] == 0.0 or parts["no_contact"] == 0.0:
        return 0.0, parts
    parts["clearance"] = clearance_factor(rec["min_clear_m"])
    parts["conditioning"] = conditioning_factor(rec["conds"])
    parts["time"] = time_factor(rec["t_used"], rec["time_target_s"])
    if rec["type"] == "reach":
        parts["accuracy"] = reach_accuracy(rec["err"])
        parts["smoothness"] = 0.5 * L(rec["settle_mm"], BANDS["settle_mm"]) + 0.5 * L(rec["overshoot_mm"], BANDS["overshoot_mm"])
    else:
        parts["progress"] = float(rec.get("progress", 1.0))
        parts["accuracy"] = L(rec["rms_mm"], BANDS["path_rms_mm"]) * (1.0 - rec["axis_out_frac"]) * parts["progress"]
        parts["smoothness"] = L(rec["peak_mm"], BANDS["path_peak_mm"])
    q = parts["accuracy"] * (0.6 + 0.2 * parts["time"] + 0.2 * parts["smoothness"])
    s = parts["clearance"] * parts["conditioning"] * q
    if not math.isfinite(s):                   # a non-finite measurement (0 x nan is nan) scores 0, never more
        s = 0.0
    return float(s), parts


def youden(records):
    inf = [r for r in records if r.get("infeasible")]
    feas = [r for r in records if not r.get("infeasible")]
    if not inf:
        return None
    tpr = sum(1 for r in inf if r.get("refused")) / len(inf)
    fpr = (sum(1 for r in feas if r.get("refused")) / len(feas)) if feas else 0.0
    return max(0.0, tpr - fpr)


def episode(records):
    """records = every job record of the episode (all cells). Returns {"overall", "reach", "path", "infeasibility", ...};
    a category with no job (e.g. a layout without paths) is left out of the weighted mean, the weights renormalised."""
    reach = [job_score(r)[0] for r in records if not r.get("infeasible") and r["type"] == "reach"]
    path = [job_score(r)[0] for r in records if not r.get("infeasible") and r["type"] == "path"]
    J = youden(records)
    out = {"reach": float(np.mean(reach)) if reach else 0.0, "path": float(np.mean(path)) if path else 0.0,
           "infeasibility": float(J) if J is not None else 0.0}
    present = {"reach": bool(reach), "path": bool(path), "infeasibility": J is not None}
    wsum = sum(WEIGHTS[k] for k in WEIGHTS if present[k])
    out["overall"] = (sum(WEIGHTS[k] * out[k] for k in WEIGHTS if present[k]) / wsum) if wsum > 0 else 0.0
    out["n_reach"], out["n_path"] = len(reach), len(path)
    out["n_infeasible"] = sum(1 for r in records if r.get("infeasible"))
    out["categories_present"] = [k for k in WEIGHTS if present[k]]
    return out
