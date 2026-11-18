"""The score of `racecar_time_trial_on_unseen_track`.

    row = score_venue(result, reference, config)                  # one venue: fast, cons, avoid, venue, rank
    rec = score_run(results, references, config, gates, anchor)   # the score record (records.schema.json #/$defs/score)
    rep = report(results, references, config, gates, anchor)      # {"score": rec, "statistics": [...], "checks": [...]}

Inputs are only the cell's records: the venue result (`#/$defs/result`, written by the cell's `Venue.result()`), the frozen
venue reference (`#/$defs/reference`) and the `scoring` block (plus the avoidance session's phase table) of the grader
configuration. The scored quantities are recomputed here from the lap lists (`lap_summary`), not read from the
result's own summary; the two must agree (a result whose summary differs from its laps is rejected as corrupt).

Per venue:

    fast  = clip((1.33 T_ref - T_best) / (0.30 T_ref), 0, 1)                      no uninterrupted heat lap: 0
    cons  = clip((N_best - 0.43 N_ref) / ((N_ref - 1) - 0.43 N_ref), 0, 1)        N_ref >= 3 (asserted)
    a(p)  = clip((1.50 T_phase_ref[p] - T_clean[p]) / (0.40 T_phase_ref[p]), 0, 1)  no clean lap in phase p: 0
    avoid = 0.5 a(static) + 0.5 a(dynamic)
    venue = 0.35 fast + 0.35 cons + 0.30 avoid

Run: time_trial_score = gate x mean(venue over the reference set), gate = 1 iff the five gates are true;
reward = clip(time_trial_score / anchor.value / full_at, 0, 1). The numbers (1.33, 0.30, ...) are read from
`config["scoring"]`; nothing is hard-coded.

`consecutive` (N_best): consecutive = joined end to start (the default, `contiguous_runs=True`,
the same reading as the contract's `check.summarise_laps`): each lap of a run starts at the timing-line crossing that ended the
lap before it. A lap the lap clock throws away (a backward crossing of the timing line, or a forward crossing without
every checkpoint, drops the lap in progress together with a crash or pick-up inside it) breaks the run: the referee
writes `chain_break` on the next completed lap, and a start that does not meet the previous end breaks it as well
(records written before 1.2 carry no flag). `contiguous_runs=False` is the scoring module's letter reading (consecutive entries of
the list without an interrupted one), kept for the regression that the reference program's N_ref is the same number
under both readings.

Gates: the verifier supplies `submission_ok`, `no_network`, `no_late_write` and the process part of `clean_run`
(no process of the program survived its venue); this module ANDs in what the records show: `step_called` = some
venue answered a step; `clean_run` additionally needs a result for every reference and no venue that ended by the
cell's own failure (`internal_error`, `physics_fault`; the verifier re-runs such a venue once before scoring).

Pure Python arithmetic (no numpy): the scoring is on the result path, which is BLAS-free.
"""

import hashlib
import json
import math

SCORE_SCHEMA = "racecar-score/1"
RESULT_SCHEMA = "racecar-venue-result/1"
REFERENCE_SCHEMA = "racecar-venue-reference/1"
GATE_KEYS = ("submission_ok", "step_called", "clean_run", "no_network", "no_late_write")
CELL_FAILURES = ("internal_error", "physics_fault")
HEATS = ("tt1", "tt2")
PHASES = ("static", "dynamic")
DIGITS = 6
EDGE_EPS = 1e-9          # phase edges (the lap records carry 6 decimals)
JOIN_EPS = 1e-5          # contiguity: start of a lap vs end of the lap before (both rounded to 1e-6 s)

__all__ = ["SCORE_SCHEMA", "GATE_KEYS", "CELL_FAILURES", "ScoringError", "clip01", "lap_summary", "check_summary",
           "score_venue", "venue_statistics", "virtual_rank", "score_run", "report", "config_sha256"]


class ScoringError(ValueError):
    """A record the scorer refuses: wrong schema, venues mixed up, a summary that contradicts its laps, N_ref < 3."""


def clip01(v):
    return 0.0 if v <= 0.0 else (1.0 if v >= 1.0 else float(v))


def config_sha256(config):
    """sha256 of the canonical text of a grader configuration (the same text as `sketch.referee.config_sha256`)."""
    text = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _sessions(result):
    out = {}
    for s in result.get("sessions") or []:
        if s["name"] in out:
            raise ScoringError("session %s appears twice in the result of %s" % (s["name"], result.get("world_id")))
        out[s["name"]] = s
    return out


def _avoid_phases(config):
    ses = [x for x in config["sessions"] if x["name"] == "avoid"]
    if len(ses) != 1:
        raise ScoringError("the configuration has no single avoid session")
    ses = ses[0]
    names = [p["name"] for p in ses["phases"]]
    if names != list(PHASES):
        raise ScoringError("the avoid session's phases are %s, the scoring needs %s" % (names, list(PHASES)))
    edges = [float(p["t0_s"]) for p in ses["phases"]] + [float(ses["duration_s"])]
    return [(PHASES[i], edges[i], edges[i + 1]) for i in range(len(PHASES))]


def _runs(laps, contiguous):
    """Length of the longest run of consecutive laps that are not interrupted (and, with `contiguous`, each starting
    where the lap before it ended and not marked `chain_break`)."""
    best = run = 0
    prev_end = None
    for lap in laps:
        if lap["interrupted"]:
            run = 0
            prev_end = None
            continue
        start = float(lap["t_end_s"]) - float(lap["lap_s"])
        if contiguous and run > 0 and (lap.get("chain_break") or prev_end is None or abs(start - prev_end) > JOIN_EPS):
            run = 0
        run += 1
        prev_end = float(lap["t_end_s"])
        if run > best:
            best = run
    return best


def lap_summary(result, config, contiguous_runs=True):
    """The scored quantities of one venue from its lap lists:
    {"time_trial": {"T_best_s", "N_best"}, "avoid": {"static": {"T_clean_s", "clean_laps"}, "dynamic": {...}}}.

    T_best_s  shortest lap of tt1 and tt2 that is not interrupted (touches do not matter); None without one
    N_best    longest run of consecutive uninterrupted laps inside one heat, the larger of the two heats (consecutive =
              joined end to start by default)
    T_clean_s per avoidance phase: shortest lap that starts and ends inside the phase, is not interrupted, has no
              touch and no standstill and, in the dynamic phase, at least `min_overtakes_dynamic` net overtakes"""
    by = _sessions(result)
    t_best, n_best = None, 0
    for name in HEATS:
        laps = by[name]["laps"] if name in by else []
        n_best = max(n_best, _runs(laps, contiguous_runs))
        for lap in laps:
            if not lap["interrupted"]:
                t = float(lap["lap_s"])
                t_best = t if t_best is None or t < t_best else t_best
    sc = config["scoring"]["avoid"]
    laps = by["avoid"]["laps"] if "avoid" in by else []
    avoid = {}
    for phase, lo, hi in _avoid_phases(config):
        need = int(sc["min_overtakes_dynamic"]) if phase == "dynamic" else 0
        times = []
        for lap in laps:
            start = float(lap["t_end_s"]) - float(lap["lap_s"])
            if lap["interrupted"] or lap["touches"] != 0 or lap["standstill"] or lap["overtakes"] < need:
                continue
            if start >= lo - EDGE_EPS and float(lap["t_end_s"]) <= hi + EDGE_EPS:
                times.append(float(lap["lap_s"]))
        avoid[phase] = {"T_clean_s": min(times) if times else None, "clean_laps": len(times)}
    return {"time_trial": {"T_best_s": t_best, "N_best": n_best}, "avoid": avoid}


def check_summary(result, config):
    """Errors of a venue result as the scorer sees it: schema basics, lap flags, and the result's own summary
    against its lap lists (the contract's reading, consecutive = joined end to start)."""
    err = []
    if result.get("schema") != RESULT_SCHEMA:
        err.append("not a venue result (%r)" % result.get("schema"))
        return err
    for s in result.get("sessions") or []:
        for i, lap in enumerate(s["laps"]):
            if lap["n"] != i + 1:
                err.append("session %s: lap numbers do not count up from 1" % s["name"])
            if lap["clean"] != ((not lap["interrupted"]) and lap["touches"] == 0):
                err.append("session %s lap %d: clean is not (not interrupted and no touch)" % (s["name"], lap["n"]))
            if not (math.isfinite(float(lap["lap_s"])) and float(lap["lap_s"]) > 0.0):
                err.append("session %s lap %d: lap_s is not a positive number" % (s["name"], lap["n"]))
            if "chain_break" in lap:      # contract 1.2; records written before carry no flag
                start = float(lap["t_end_s"]) - float(lap["lap_s"])
                if i == 0 and lap["chain_break"]:
                    err.append("session %s lap 1 has chain_break" % s["name"])
                elif i > 0 and not lap["chain_break"] and abs(start - float(s["laps"][i - 1]["t_end_s"])) > JOIN_EPS:
                    err.append("session %s lap %d does not start where lap %d ended but has no chain_break"
                               % (s["name"], lap["n"], lap["n"] - 1))
    if err:
        return err
    mine = lap_summary(result, config, contiguous_runs=True)
    for key in ("time_trial", "avoid"):
        if result.get(key) != mine[key]:
            err.append("%s %s differs from the lap lists, which give %s" % (key, json.dumps(result.get(key)), json.dumps(mine[key])))
    return err


def virtual_rank(t_best, t_ref, ratios):
    """1 + the number of ratios in [1.0] + `ratios` below T_best / T_ref (the real field's places); None without a lap."""
    if t_best is None:
        return None
    ratio = float(t_best) / float(t_ref)
    return 1 + sum(1 for r in [1.0] + [float(x) for x in ratios] if r < ratio)


def _check_reference(ref):
    if ref.get("schema") != REFERENCE_SCHEMA:
        raise ScoringError("not a venue reference (%r)" % ref.get("schema"))
    n_ref = ref["N_ref"]
    if not (isinstance(n_ref, int) and not isinstance(n_ref, bool) and n_ref >= 3):
        raise ScoringError("reference %s: N_ref = %r, the scoring needs an integer >= 3" % (ref.get("world_id"), n_ref))
    for v in [ref["T_ref_s"], ref["T_phase_ref_s"]["static"], ref["T_phase_ref_s"]["dynamic"]]:
        if not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0.0):
            raise ScoringError("reference %s: a reference time is not a positive number" % ref.get("world_id"))


def score_venue(result, reference, config, contiguous_runs=True, strict=True):
    """One venue's row of the score record: world_id, fast, cons, avoid_static, avoid_dynamic, avoid, venue,
    virtual_rank (components rounded to 6 decimals; `venue` from the unrounded components)."""
    _check_reference(reference)
    if result is not None:
        if result.get("layout_digest") != reference["layout_digest"]:
            raise ScoringError("result %s and reference %s are of different venues"
                               % (result.get("world_id"), reference.get("world_id")))
        if strict:
            err = check_summary(result, config)
            if err:
                raise ScoringError("result %s: %s" % (result.get("world_id"), "; ".join(err)))
    sc = config["scoring"]
    t_ref, n_ref = float(reference["T_ref_s"]), int(reference["N_ref"])
    summ = lap_summary(result, config, contiguous_runs) if result is not None else \
        {"time_trial": {"T_best_s": None, "N_best": 0}, "avoid": {p: {"T_clean_s": None, "clean_laps": 0} for p in PHASES}}
    t_best, n_best = summ["time_trial"]["T_best_s"], summ["time_trial"]["N_best"]
    fs = sc["fast"]
    fast = 0.0 if t_best is None else clip01((fs["zero_ratio"] * t_ref - t_best) / (fs["span_ratio"] * t_ref))
    cs = sc["cons"]
    zero = cs["zero_frac"] * n_ref
    cons = clip01((n_best - zero) / ((n_ref - cs["full_margin_laps"]) - zero))
    av = sc["avoid"]
    a = {}
    for p in PHASES:
        t_clean, t_pref = summ["avoid"][p]["T_clean_s"], float(reference["T_phase_ref_s"][p])
        a[p] = 0.0 if t_clean is None else clip01((av["zero_ratio"] * t_pref - t_clean) / (av["span_ratio"] * t_pref))
    avoid = av["phase_weights"]["static"] * a["static"] + av["phase_weights"]["dynamic"] * a["dynamic"]
    w = sc["weights"]
    venue = w["fast"] * fast + w["cons"] * cons + w["avoid"] * avoid
    return {"world_id": reference["world_id"], "fast": round(fast, DIGITS), "cons": round(cons, DIGITS),
            "avoid_static": round(a["static"], DIGITS), "avoid_dynamic": round(a["dynamic"], DIGITS),
            "avoid": round(avoid, DIGITS), "venue": round(venue, DIGITS),
            "virtual_rank": virtual_rank(t_best, t_ref, sc["virtual_rank_ratios"])}


def venue_statistics(result, reference=None, config=None, contiguous_runs=True):
    """The unscored statistics of one venue (reported, never scored): per session ticks, end, laps, touches,
    crashes, pick-ups, warnings, out-of-bounds, largest displacement, program and desk time; wall displacement,
    forbidden and bad calls, end reason; with a reference the ratios to it and the virtual rank."""
    if result is None:
        return {"world_id": reference["world_id"] if reference else None, "result": None}
    ses = []
    for s in result.get("sessions") or []:
        ses.append({"name": s["name"], "ticks": s["ticks"], "end": s["end"], "laps": len(s["laps"]),
                    "laps_interrupted": sum(1 for x in s["laps"] if x["interrupted"]),
                    "laps_clean": sum(1 for x in s["laps"] if x["clean"]), "touches": s["touches"],
                    "crashes": s["crashes"], "pickups": s["pickups"], "warnings": s["warnings"],
                    "out_of_bounds": s["out_of_bounds"], "max_displacement_m": s["max_displacement_m"],
                    "program_s": s["program_s"], "desk_s": s["desk_s"]})
    st = result["stats"]
    out = {"world_id": result["world_id"], "end_reason": result["end_reason"], "step_called": result["step_called"],
           "sessions": ses, "touches": st["touches"], "crashes": st["crashes"],
           "pickups": sum(s["pickups"] for s in result.get("sessions") or []),
           "out_of_bounds": sum(s["out_of_bounds"] for s in result.get("sessions") or []),
           "warnings": st["warnings"], "wall_displacement_m": st["wall_displacement_m"],
           "laps_by_session": dict(st["laps_by_session"]), "program_s": st["program_s"],
           "desk_s": round(sum(s["desk_s"] for s in result.get("sessions") or []), DIGITS),
           "forbidden_calls": st["forbidden_calls"], "bad_calls": st["bad_calls"]}
    if reference is not None and config is not None:
        summ = lap_summary(result, config, contiguous_runs)
        t_best, n_best = summ["time_trial"]["T_best_s"], summ["time_trial"]["N_best"]
        out["T_best_s"], out["N_best"] = t_best, n_best
        out["T_best_over_T_ref"] = None if t_best is None else round(t_best / float(reference["T_ref_s"]), DIGITS)
        out["N_best_over_N_ref"] = round(n_best / float(reference["N_ref"]), DIGITS)
        for p in PHASES:
            t = summ["avoid"][p]["T_clean_s"]
            out["T_clean_%s_s" % p] = t
            out["T_clean_%s_over_ref" % p] = None if t is None else round(t / float(reference["T_phase_ref_s"][p]), DIGITS)
        out["virtual_rank"] = virtual_rank(t_best, reference["T_ref_s"], config["scoring"]["virtual_rank_ratios"])
    return out


def _match(results, references):
    """Pair every reference with its result (by layout digest); a reference without a result gets None."""
    by_digest = {}
    for r in results:
        if r is None:
            continue
        dg = r.get("layout_digest")
        if dg in by_digest:
            raise ScoringError("two results for the venue %s" % r.get("world_id"))
        by_digest[dg] = r
    known = set(ref["layout_digest"] for ref in references)
    if len(known) != len(references):
        raise ScoringError("two references for one venue")
    extra = [r.get("world_id") for r in by_digest.values() if r.get("layout_digest") not in known]
    if extra:
        raise ScoringError("results of venues without a reference: %s" % extra)
    return [(by_digest.get(ref["layout_digest"]), ref) for ref in references]


def _gates(pairs, gates):
    given = dict(gates) if gates is not None else {k: True for k in GATE_KEYS}
    missing = [k for k in GATE_KEYS if k not in given]
    if missing or set(given) - set(GATE_KEYS):
        raise ScoringError("gates must have exactly the keys %s" % (GATE_KEYS,))
    stepped = any(res is not None and bool(res.get("step_called")) for res, _ in pairs)
    complete = all(res is not None for res, _ in pairs)
    no_fault = all(res is None or res.get("end_reason") not in CELL_FAILURES for res, _ in pairs)
    out = {k: bool(given[k]) for k in GATE_KEYS}
    out["step_called"] = out["step_called"] and stepped
    out["clean_run"] = out["clean_run"] and complete and no_fault
    return out


def score_run(results, references, config, gates=None, anchor_value=1.0, contiguous_runs=True, strict=True):
    """The score record of one verify (records.schema.json #/$defs/score): one row per reference (the frozen hidden
    set defines the venues), the five gates, gate, time_trial_score, anchor, reward."""
    if not references:
        raise ScoringError("no references")
    if not (isinstance(anchor_value, (int, float)) and math.isfinite(anchor_value) and anchor_value > 0.0):
        raise ScoringError("anchor value must be a positive number")
    pairs = _match(list(results), list(references))
    rows = [score_venue(res, ref, config, contiguous_runs=contiguous_runs, strict=strict) for res, ref in pairs]
    g = _gates(pairs, gates)
    gate = 1 if all(g.values()) else 0
    mean = sum(r["venue"] for r in rows) / len(rows)
    score = round(gate * mean, DIGITS)
    full_at = float(config["scoring"]["full_at"])
    return {"schema": SCORE_SCHEMA, "venues": rows, "gates": g, "gate": gate, "time_trial_score": score,
            "anchor": {"value": anchor_value, "full_at": full_at},
            "reward": round(clip01(score / anchor_value / full_at), DIGITS)}


def report(results, references, config, gates=None, anchor_value=1.0, contiguous_runs=True, strict=True):
    """The score record plus the unscored statistics per venue and the record checks (what the verifier writes)."""
    pairs = _match(list(results), list(references))
    rec = score_run([r for r, _ in pairs if r is not None], [ref for _, ref in pairs], config, gates=gates,
                    anchor_value=anchor_value, contiguous_runs=contiguous_runs, strict=strict)
    stats = [venue_statistics(res, ref, config, contiguous_runs) for res, ref in pairs]
    sha = config_sha256(config)
    checks = []
    for res, ref in pairs:
        if res is None:
            checks.append({"world_id": ref["world_id"], "result": False})
            continue
        checks.append({"world_id": ref["world_id"], "result": True, "config_sha256_matches": res.get("config_sha256") == sha,
                       "summary_errors": check_summary(res, config),
                       "letter_N_best": lap_summary(res, config, contiguous_runs=False)["time_trial"]["N_best"]})
    return {"score": rec, "statistics": stats, "checks": checks}
