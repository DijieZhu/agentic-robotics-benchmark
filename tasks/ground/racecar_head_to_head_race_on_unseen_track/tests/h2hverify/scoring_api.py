"""The one call of the verifier into this task's scoring module, and the checks on what comes back.

    rec = score(results, references, config, gates, anchor_value)    # the score record, racecar-score-h2h/1

The scoring module `h2h.scoring` (vendored under grader/) provides

    score_run(results, references, config, gates=..., anchor_value=...) -> the score record

with `references` the reference records of the graded venues in venue order, `results` the cell's records of the
venues that were run (matched to the references by `world_id`; none when the submission gate failed, and a venue
without a record scores 0), `config` the cell's configuration, `gates` {gate name: bool} for the five gates of
GATE_KEYS and `anchor_value` the anchor's value. The record holds `venues[]` (one row per reference, in its order:
its `world_id`, per heat `classified`, `place`, `pass`, `conduct` and `heat`, and `venue`), `gates`, `gate`,
`head_to_head_score`, `anchor` and `reward`.

`check_score` refuses a record that cannot be the score of these venues and gates (a task error, never a score):
another schema, another number or order of venues, gates other than the ones given, `gate` not 1 exactly when every
gate holds, a score that is not 0 under a closed gate, a number that is not finite or lies outside [0, 1].
"""

import math

__all__ = ["score", "check_score", "SCORE_SCHEMA", "GATE_KEYS", "ScoreRecordError"]

SCORE_SCHEMA = "racecar-score-h2h/1"
GATE_KEYS = ("submission_ok", "step_called", "clean_run", "no_network", "no_late_write")


class ScoreRecordError(RuntimeError):
    pass


def _unit(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and 0.0 <= float(v) <= 1.0


def check_score(rec, world_ids, gates):
    """[problem] of a score record against the venues (their ids, in order) and the gates that were given."""
    if not isinstance(rec, dict):
        return ["the score record is not an object"]
    err = []
    if rec.get("schema") != SCORE_SCHEMA:
        err.append("schema %r" % rec.get("schema"))
    rows = rec.get("venues")
    if not isinstance(rows, list) or [r.get("world_id") if isinstance(r, dict) else None for r in rows] != list(world_ids):
        err.append("the venues are not the graded ones in their order")
    else:
        for i, r in enumerate(rows):
            if not _unit(r.get("venue")):
                err.append("venue %d: the venue score is not a number in [0, 1]" % (i + 1))
    if {k: bool(v) for k, v in (rec.get("gates") or {}).items()} != {k: bool(gates[k]) for k in GATE_KEYS}:
        err.append("the gates of the record are not the ones given")
    want_gate = 1 if all(bool(gates[k]) for k in GATE_KEYS) else 0
    if rec.get("gate") != want_gate:
        err.append("gate is not %d" % want_gate)
    for k in ("head_to_head_score", "reward"):
        if not _unit(rec.get(k)):
            err.append("%s is not a number in [0, 1]" % k)
    if want_gate == 0 and _unit(rec.get("head_to_head_score")) and float(rec["head_to_head_score"]) != 0.0:
        err.append("a closed gate with a score above 0")
    return err


def score(results, references, config, gates, anchor_value):
    """The score record of a run (checked; raises ScoreRecordError when the record cannot be this run's)."""
    from h2h import scoring
    fn = getattr(scoring, "score_run", None)
    if not callable(fn):
        raise ScoreRecordError("the scoring module has no score_run")
    missing = [k for k in GATE_KEYS if k not in gates]
    if missing:
        raise ScoreRecordError("gates missing: %s" % ", ".join(missing))
    rec = fn(list(results), list(references), config, gates={k: bool(gates[k]) for k in GATE_KEYS},
             anchor_value=float(anchor_value))
    problems = check_score(rec, [r["world_id"] for r in references], gates)
    if problems:
        raise ScoreRecordError("the score record: %s" % "; ".join(problems[:5]))
    return rec
