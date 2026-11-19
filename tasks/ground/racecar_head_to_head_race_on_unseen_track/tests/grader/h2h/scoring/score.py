"""The score of racecar_head_to_head_race_on_unseen_track.

    terms = heat_terms(block, C_ref, config)                       # one heat: opened, classified, place, pass, conduct, heat
    row = score_venue(result, reference, config)                   # one venue: its heats and their mean
    rec = score_run(results, references, config, gates, anchor)   # the score record (score.schema.json, "score")
    rep = report(results, references, config, gates, anchor)      # {"score", "statistics", "checks", "crosscheck"}

Inputs are the cell's records only: the venue results written by the heat cell, the frozen venue references and the
`scoring`, `heat` and `staff` blocks of the grader configuration. A heat is scored on its heat block as the referee
left it, whatever ended the heat (the N-th lap of any car, the time cap, the car's third warning, the void rule, a
closed session, a program fault):

    heat    = classified x conduct x (place_weight x place + pass_weight x pass)
    place   = beaten / (number of field cars)
    pass    = clip(C / (C_ref - pass_margin), 0, 1)
    conduct = max(0, 1 - K / warnings_zero_at)
    venue   = mean of the configuration's heats; a heat that was never opened scores 0 (every term 0)

`C` is the car's clean overtake balance over the three field cars, `beaten` the number of field cars it beat, `K` its
warnings in the heat and `classified` whether it completed at least N minus `heat.classified_laps_short` laps with
fewer than `staff.warnings_end_session` warnings; all four are read from the heat block. `C_ref` is the reference
program's balance in the same heat of the same venue (the venue reference), an integer of at least C_REF_MIN.

Run: head_to_head_score = gate x mean(venue over the references), gate = 1 iff every gate of GATE_KEYS is true;
reward = clip(head_to_head_score / anchor / full_at, 0, 1). The verifier supplies the gates; this module ANDs in what
the records show: `step_called` needs some venue that answered a step, `clean_run` a result for every reference and no
venue that ended by the cell's own failure (CELL_FAILURES).

Refused (ScoringError, never a score): a number that is not finite anywhere in a result, a reference, the anchor or
the scoring block; a reference whose balance is not an integer of at least C_REF_MIN or whose lap count is not the
heat block's; results and references of different venues; and, with `strict` (the default), a result that fails the
heat cell's record checker, which recomputes `C`, `beaten`, `K` and `classified` from the block's lists, so a result
whose numbers contradict its lists is never scored. The numbers are not rounded. Pure Python arithmetic: nothing here
calls a BLAS-backed routine.
"""

import json
import math
import os

from sketch.examples import check
from sketch.referee.config import config_sha256

from h2h.cell import config as cell_config
from h2h.cell.record import check_result

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "score.schema.json")

SCORE_SCHEMA = "racecar-score-h2h/1"
RESULT_SCHEMA = "racecar-venue-result-h2h/1"
REFERENCE_SCHEMA = "racecar-venue-reference-h2h/1"
GATE_KEYS = ("submission_ok", "step_called", "clean_run", "no_network", "no_late_write")
CELL_FAILURES = ("internal_error", "physics_fault")
C_REF_MIN = 5
TERM_KEYS = ("opened", "classified", "place", "pass", "conduct", "heat")

__all__ = ["SCORE_SCHEMA", "RESULT_SCHEMA", "REFERENCE_SCHEMA", "GATE_KEYS", "CELL_FAILURES", "C_REF_MIN", "TERM_KEYS",
           "ScoringError", "clip01", "heats_of", "field_size", "check_reference", "check_scoring_config", "heat_terms",
           "score_venue", "score_run", "venue_statistics", "report", "record_errors", "make_reference", "config_sha256"]


class ScoringError(ValueError):
    """A record the scorer refuses: another schema, venues mixed up, a non-finite number, a reference balance below
    C_REF_MIN, a result whose numbers contradict its lists."""


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _finite(v, what):
    if not _is_num(v) or not math.isfinite(float(v)):
        raise ScoringError("%s is not a finite number" % what)
    return v


def _non_finite_path(obj, path=""):
    """The path of the first number in obj that is not finite, or None."""
    if isinstance(obj, float):
        return None if math.isfinite(obj) else (path or "/")
    if isinstance(obj, dict):
        for k in sorted(obj):
            p = _non_finite_path(obj[k], "%s/%s" % (path, k))
            if p is not None:
                return p
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            p = _non_finite_path(v, "%s/%d" % (path, i))
            if p is not None:
                return p
    return None


def _require_finite(obj, what):
    p = _non_finite_path(obj)
    if p is not None:
        raise ScoringError("%s holds a number that is not finite at %s" % (what, p))


def clip01(v):
    """v clipped to [0, 1]; a value that is not finite is refused (checked before the clamp)."""
    _finite(v, "a score term")
    return 0.0 if v <= 0.0 else (1.0 if v >= 1.0 else float(v))


def _schema():
    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def record_errors(kind, obj):
    """Schema errors of a record of this module: kind 'reference' or 'score'."""
    s = _schema()
    if kind not in s["$defs"]:
        return ["unknown record kind %r" % kind]
    return check.schema_errors(obj, {"$schema": s["$schema"], "$ref": "#/$defs/" + kind, "$defs": s["$defs"]})


def heats_of(config):
    """The heat sessions of the configuration, in its order."""
    heats = cell_config.heat_sessions(config)
    if not heats:
        raise ScoringError("the configuration has no heats")
    return heats


def field_size(config):
    """Number of field cars in a heat (every car of the heat but the car itself)."""
    n = config["heat"]["cars"]
    if not _is_int(n) or n < 2:
        raise ScoringError("heat.cars is not an integer of at least 2")
    return n - 1


def check_scoring_config(config):
    """Errors of the blocks of the grader configuration that the score reads."""
    err = []
    try:
        sc = config["scoring"]["heat"]
        full_at = config["scoring"]["full_at"]
        short = config["heat"]["classified_laps_short"]
        end_at = config["staff"]["warnings_end_session"]
    except (KeyError, TypeError):
        return ["the configuration lacks scoring.heat, scoring.full_at, heat.classified_laps_short or "
                "staff.warnings_end_session"]
    for k in ("place_weight", "pass_weight"):
        v = sc.get(k)
        if not (_is_num(v) and math.isfinite(float(v)) and 0.0 <= v <= 1.0):
            err.append("scoring.heat.%s is not a finite number in [0, 1]" % k)
    if not err and abs(sc["place_weight"] + sc["pass_weight"] - 1.0) > 1e-12:
        err.append("scoring.heat.place_weight and pass_weight do not sum to 1")
    if not (_is_int(sc.get("pass_margin")) and 0 <= sc["pass_margin"] < C_REF_MIN):
        err.append("scoring.heat.pass_margin is not an integer in [0, %d)" % C_REF_MIN)
    if not (_is_int(sc.get("warnings_zero_at")) and sc["warnings_zero_at"] >= 1):
        err.append("scoring.heat.warnings_zero_at is not a positive integer")
    if not (_is_num(full_at) and math.isfinite(float(full_at)) and 0.0 < full_at <= 1.0):
        err.append("scoring.full_at is not a finite number in (0, 1]")
    if not (_is_int(short) and short >= 0) or not (_is_int(end_at) and end_at >= 1):
        err.append("heat.classified_laps_short or staff.warnings_end_session is not a usable integer")
    try:
        heats_of(config)
        field_size(config)
    except (ScoringError, KeyError, TypeError) as e:
        err.append(str(e))
    return err


def check_reference(ref, config):
    """Errors of a venue reference: its schema, a reference balance per heat that is an integer of at least
    C_REF_MIN, a positive lap count, finite numbers."""
    if not isinstance(ref, dict):
        return ["a reference is not an object"]
    p = _non_finite_path(ref)
    if p is not None:
        return ["reference %s: a number is not finite at %s" % (ref.get("world_id"), p)]
    err = record_errors("reference", ref)
    if err:
        return ["reference %s: %s" % (ref.get("world_id"), e) for e in err]
    n_heats = len(heats_of(config))
    if len(ref["C_ref"]) != n_heats or len(ref["T_N_s"]) != n_heats:
        err.append("reference %s: C_ref and T_N_s need one value per heat (%d)" % (ref["world_id"], n_heats))
    for c in ref["C_ref"]:
        if not (_is_int(c) and c >= C_REF_MIN):
            err.append("reference %s: C_ref %r is not an integer of at least %d" % (ref["world_id"], c, C_REF_MIN))
    return err


def make_reference(world, C_ref, T_N_s, environments=None):
    """A venue reference of a final world: its id, layout digest and lap count, the reference program's balance and
    the session time of its N-th lap in each heat (checked like any reference)."""
    ref = {"schema": REFERENCE_SCHEMA, "world_id": world["world_id"], "layout_digest": world["layout_digest"],
           "heat_laps": world["venue_sheet"]["heat_laps"], "C_ref": list(C_ref), "T_N_s": list(T_N_s)}
    if environments is not None:
        ref["environments"] = environments
    return ref


def _empty_terms():
    return {"opened": False, "classified": 0, "place": 0.0, "pass": 0.0, "conduct": 0.0, "heat": 0.0}


def heat_terms(block, C_ref, config):
    """The terms of one heat from its heat block; None (a heat that was never opened) scores 0 in every term."""
    if block is None:
        return _empty_terms()
    sc = config["scoring"]["heat"]
    C, beaten, K = block["C"], block["beaten"], block["K"]
    cls = block["car"]["classified"]
    if not (_is_int(C) and _is_int(beaten) and _is_int(K)):
        raise ScoringError("C, beaten and K of a heat block must be integers")
    if not isinstance(cls, bool):
        raise ScoringError("the car's classified flag is not a boolean (a world without its lap count is not scored)")
    if not (_is_int(C_ref) and C_ref >= C_REF_MIN):
        raise ScoringError("C_ref %r is not an integer of at least %d" % (C_ref, C_REF_MIN))
    nf = field_size(config)
    if len(block["field"]) != nf or not 0 <= beaten <= nf or K < 0:
        raise ScoringError("a heat block with %d field cars, beaten %d, K %d" % (len(block["field"]), beaten, K))
    w_place = float(_finite(sc["place_weight"], "scoring.heat.place_weight"))
    w_pass = float(_finite(sc["pass_weight"], "scoring.heat.pass_weight"))
    margin = sc["pass_margin"]
    zero_at = sc["warnings_zero_at"]
    classified = 1 if cls else 0
    place = clip01(beaten / nf)
    pass_ = clip01(_finite(C / (C_ref - margin), "C / (C_ref - pass_margin)"))
    conduct = clip01(_finite(1.0 - K / zero_at, "1 - K / warnings_zero_at"))
    heat = clip01(_finite(classified * conduct * (w_place * place + w_pass * pass_), "the heat score"))
    return {"opened": True, "classified": classified, "place": place, "pass": pass_, "conduct": conduct, "heat": heat}


def _sessions(result):
    out = {}
    for s in result.get("sessions") or []:
        if s["name"] in out:
            raise ScoringError("session %s appears twice in the result of %s" % (s["name"], result.get("world_id")))
        out[s["name"]] = s
    return out


def score_venue(result, reference, config, strict=True):
    """One venue's row of the score record: world_id, the terms of every heat (keyed by session name), venue."""
    err = check_reference(reference, config)
    if err:
        raise ScoringError("; ".join(err))
    heats = heats_of(config)
    by = {}
    if result is not None:
        _require_finite(result, "result %s" % result.get("world_id"))
        if result.get("schema") != RESULT_SCHEMA:
            raise ScoringError("not a venue result of this task (%r)" % result.get("schema"))
        if result.get("world_id") != reference["world_id"] or result.get("layout_digest") != reference["layout_digest"]:
            raise ScoringError("result %s and reference %s are of different venues"
                               % (result.get("world_id"), reference["world_id"]))
        if strict:
            err = check_result(result, config)
            if err:
                raise ScoringError("result %s: %s" % (result.get("world_id"), "; ".join(err[:5])))
        by = _sessions(result)
    row = {"world_id": reference["world_id"]}
    for i, name in enumerate(heats):
        ses = by.get(name)
        block = None
        if ses is not None:
            block = ses.get("heat")
            if block is None:
                raise ScoringError("result %s: session %s has no heat block" % (reference["world_id"], name))
            if block["laps_target"] != reference["heat_laps"]:
                raise ScoringError("result %s: session %s was run with N = %r, the reference has %r"
                                   % (reference["world_id"], name, block["laps_target"], reference["heat_laps"]))
        row[name] = heat_terms(block, reference["C_ref"][i], config)
    row["venue"] = clip01(sum(row[n]["heat"] for n in heats) / len(heats))
    return row


def _match(results, references):
    """Every reference with its result (by world id; the layout digests must agree); None where there is none."""
    by_id = {}
    for r in results:
        if r is None:
            continue
        wid = r.get("world_id")
        if wid in by_id:
            raise ScoringError("two results for the venue %s" % wid)
        by_id[wid] = r
    ids = [ref.get("world_id") for ref in references]
    if len(set(ids)) != len(ids):
        raise ScoringError("two references for one venue")
    extra = sorted(str(w) for w in by_id if w not in set(ids))
    if extra:
        raise ScoringError("results of venues without a reference: %s" % ", ".join(extra))
    return [(by_id.get(ref.get("world_id")), ref) for ref in references]


def _gates(pairs, gates):
    given = dict(gates) if gates is not None else {k: True for k in GATE_KEYS}
    if set(given) != set(GATE_KEYS):
        raise ScoringError("gates must have exactly the keys %s" % (GATE_KEYS,))
    stepped = any(res is not None and res.get("step_called") is True for res, _ in pairs)
    complete = all(res is not None for res, _ in pairs)
    no_fault = all(res is None or res.get("end_reason") not in CELL_FAILURES for res, _ in pairs)
    out = {k: bool(given[k]) for k in GATE_KEYS}
    out["step_called"] = out["step_called"] and stepped
    out["clean_run"] = out["clean_run"] and complete and no_fault
    return out


def score_run(results, references, config, gates=None, anchor_value=1.0, strict=True):
    """The score record of one verify: one row per reference in its order (the frozen hidden set defines the
    venues), the gates, gate, head_to_head_score, anchor and reward."""
    references = list(references or [])
    if not references:
        raise ScoringError("no references")
    err = check_scoring_config(config)
    if err:
        raise ScoringError("the configuration: %s" % "; ".join(err))
    _finite(anchor_value, "the anchor value")
    if not anchor_value > 0.0:
        raise ScoringError("the anchor value must be positive")
    pairs = _match(list(results or []), references)
    rows = [score_venue(res, ref, config, strict=strict) for res, ref in pairs]
    g = _gates(pairs, gates)
    gate = 1 if all(g.values()) else 0
    mean = sum(r["venue"] for r in rows) / len(rows)
    score = clip01(float(gate) * mean)
    full_at = float(config["scoring"]["full_at"])
    reward = clip01(_finite(score / float(anchor_value) / full_at, "the reward before its clamp"))
    return {"schema": SCORE_SCHEMA, "venues": rows, "gates": g, "gate": gate, "head_to_head_score": score,
            "anchor": {"value": float(anchor_value), "full_at": full_at}, "reward": reward}


def venue_statistics(result, reference=None):
    """The unscored numbers of one venue (reported, never scored): per session its length, end and counts; per heat
    its end, the car's laps and progress, its overtakes, contacts, stops and placements, the field cars' laps,
    resets and touches; the run's calls."""
    if result is None:
        return {"world_id": reference["world_id"] if reference else None, "result": None}
    ses = []
    for s in result.get("sessions") or []:
        row = {"name": s["name"], "ticks": s["ticks"], "end": s["end"], "laps": len(s["laps"]),
               "laps_clean": sum(1 for x in s["laps"] if x["clean"]), "touches": s["touches"], "crashes": s["crashes"],
               "pickups": s["pickups"], "warnings": s["warnings"], "out_of_bounds": s["out_of_bounds"],
               "max_displacement_m": s["max_displacement_m"]}
        h = s.get("heat")
        if h is not None:
            ov = h["overtakes"]
            row["heat"] = {
                "end_reason": h["end_reason"], "end_t_s": h["end_t_s"], "ended_by_car": h["ended_by"] == cell_config.CAR_ID,
                "ended_by_field": h["ended_by"] is not None and h["ended_by"] != cell_config.CAR_ID,
                "laps_target": h["laps_target"], "car_laps": h["car"]["laps"], "car_progress": h["car"]["progress"],
                "C": h["C"], "beaten": h["beaten"], "K": h["K"],
                "overtakes_clean": sum(1 for e in ov if e["kind"] == "overtake" and e["clean"] is True),
                "overtakes_with_contact": sum(1 for e in ov if e["kind"] == "overtake" and e["clean"] is not True),
                "losses": sum(1 for e in ov if e["kind"] == "loss"),
                "contacts": len(h["contacts"]), "contact_crashes": sum(1 for b in h["contacts"] if b["class"] == "crash"),
                "contacts_car_fault": sum(1 for b in h["contacts"] if b["fault"] == "car"),
                "stops": len(h["stops"]), "stops_warned": sum(1 for x in h["stops"] if x["warned"]),
                "placement_fallbacks": h["car"]["placement_fallbacks"],
                "field_resets": sum(f["resets"] for f in h["field"]),
                "field_wall_touches": sum(f["wall_touches"] for f in h["field"]),
                "field_touches": sum(f["field_touches"] for f in h["field"])}
        ses.append(row)
    st = result["stats"]
    return {"world_id": result["world_id"], "end_reason": result["end_reason"], "step_called": result["step_called"],
            "sessions": ses, "touches": st["touches"], "crashes": st["crashes"], "warnings": st["warnings"],
            "wall_displacement_m": st["wall_displacement_m"], "program_s": st["program_s"],
            "forbidden_calls": st["forbidden_calls"], "bad_calls": st["bad_calls"]}


def report(results, references, config, gates=None, anchor_value=1.0, strict=True, crosscheck=True, tolerance=1e-9):
    """The score record, the unscored statistics per venue and the record checks; with `crosscheck` the score is
    computed a second time by the separate implementation from the lists of the records, and a difference above
    `tolerance` in any number is refused."""
    references = list(references or [])
    rec = score_run(results, references, config, gates=gates, anchor_value=anchor_value, strict=strict)
    pairs = _match(list(results or []), references)
    stats = [venue_statistics(res, ref) for res, ref in pairs]
    sha = config_sha256(config)
    checks = []
    for res, ref in pairs:
        if res is None:
            checks.append({"world_id": ref["world_id"], "result": False})
            continue
        checks.append({"world_id": ref["world_id"], "result": True,
                       "config_sha256_matches": res.get("config_sha256") == sha,
                       "record_errors": check_result(res, config)})
    out = {"score": rec, "statistics": stats, "checks": checks}
    if crosscheck:
        from . import crosscheck as X
        other = X.score_run(results, references, config, gates=gates, anchor_value=anchor_value)
        d = X.max_difference(rec, other)
        out["crosscheck"] = {"max_abs_difference": d, "tolerance": tolerance}
        if not d <= tolerance:
            raise ScoringError("the two computations of the score differ by %.3g" % d)
    return out
