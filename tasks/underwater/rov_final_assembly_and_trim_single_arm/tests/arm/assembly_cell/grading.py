"""Per-episode metrics -> the public criteria of the assembly task (CHAIN_PLAN 4.3).

The same functions serve the hidden grader (verify.py, with live anchoring) and the practice grader
(raw values only). Everything here is arithmetic on the episode summaries the server writes.

  gates      contract_ok (program imports / exposes solve / route gate / >= 1 step), per-episode safety multiplier
  criteria   assembly_complete 0.40   sticky weighted fraction of achieved mates x safety, mean over episodes,
                                      anchored: clamp(agent_mean / oracle_mean, 0, 1)
             final_assembled   0.30   success = all in-scope mates achieved and held null at the end (x safety);
                                      clamp(rate / oracle_rate, 0, 1)
             functional_articulation 0.15  (completed joints / in-scope joints) x (passed / 8), x safety; anchored
             as_built_pose_error 0.15 weighted mean over in-scope mates of band(pre-snap error); an unachieved mate
                                      counts 0; band = min(f_pos, f_ang), f_pos = clip((1.5 - e_pos) / 0.75, 0, 1),
                                      f_ang = clip((3 - e_ang) / 1.5, 0, 1) (full <= 0.75 mm / 1.5 deg); anchored
"""

__all__ = ["WEIGHTS", "FULL_AT", "band", "episode_values", "criteria_from_episodes", "anchored"]

WEIGHTS = {"assembly_complete": 0.40, "final_assembled": 0.30, "functional_articulation": 0.15, "as_built_pose_error": 0.15}
BANDS = {"pos_full_mm": 0.75, "pos_zero_mm": 1.5, "ang_full_deg": 1.5, "ang_zero_deg": 3.0}


def _clip(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def band(pos_err_mm, ang_err_deg, bands=None):
    b = dict(BANDS)
    b.update(bands or {})
    f_pos = _clip((b["pos_zero_mm"] - pos_err_mm) / (b["pos_zero_mm"] - b["pos_full_mm"]))
    f_ang = _clip((b["ang_zero_deg"] - ang_err_deg) / (b["ang_zero_deg"] - b["ang_full_deg"]))
    return min(f_pos, f_ang)


def episode_values(summary, bands=None):
    """Raw per-episode values (before anchoring) of the four criteria + the safety multiplier."""
    safety = 1.0 if summary.get("safety_ok", True) else 0.0
    mates = summary["mates"]
    weights = summary.get("weights") or {r["id"]: r.get("weight", 1.0) for r in mates}
    W = sum(weights.values()) or 1.0
    sticky = sum(weights[r["id"]] for r in mates if r["achieved"]) / W
    pose = sum(weights[r["id"]] * band(r["pos_err_mm"], r["ang_err_deg"], bands) for r in mates if r["achieved"]) / W
    func = float(summary.get("functional", {}).get("score", 0.0))
    return {
        "safety": safety,
        "assembly_complete": sticky * safety,
        "final_assembled": (1.0 if summary.get("success") else 0.0) * safety,
        "functional_articulation": func * safety,
        "as_built_pose_error": pose * safety,
        "sim_time_s": float(summary.get("sim_time_s", 0.0)),
    }


FULL_AT = {"as_built_pose_error": 0.9}      # 90 % of the reference's banded pose score earns full credit: the
                                             # reference's own value scatters +-0.03 between runs (servos resting in
                                             # the pocket play), so an equivalent program is not penalised for noise


def anchored(agent_values, oracle_values, key, zero_oracle="error", full_at=None):
    """clamp((mean_agent / mean_oracle) / full_at, 0, 1) on the same seeds; full_at defaults to FULL_AT[key] or 1.0.
    A zero oracle mean is a task error by default (M13')."""
    a = sum(v[key] for v in agent_values) / len(agent_values)
    o = sum(v[key] for v in oracle_values) / len(oracle_values)
    fa = FULL_AT.get(key, 1.0) if full_at is None else float(full_at)
    if o <= 1e-12:
        if zero_oracle == "error":
            raise RuntimeError("live oracle scored 0 on %s over the drawn seeds (M13' violated)" % key)
        return 0.0 if zero_oracle == "zero" else _clip(a)
    return _clip((a / o) / fa)


def criteria_from_episodes(agent_values, oracle_values=None, weights=None):
    """Criteria dict (+ overall). Without oracle values the raw means are returned (practice grader)."""
    w = dict(WEIGHTS)
    w.update(weights or {})
    out = {}
    for key in WEIGHTS:
        if oracle_values:
            out[key] = anchored(agent_values, oracle_values, key)
        else:
            out[key] = sum(v[key] for v in agent_values) / len(agent_values)
    out["overall"] = sum(out[k] * w[k] for k in WEIGHTS) / sum(w[k] for k in WEIGHTS)
    return out
