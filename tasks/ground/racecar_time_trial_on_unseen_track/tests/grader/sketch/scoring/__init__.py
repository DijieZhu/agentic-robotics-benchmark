"""sketch.scoring: the score of the task, and the grader's quality control.

    from sketch.scoring import score_venue, score_run, report
    row = score_venue(result, reference, config)                 # the contract, one venue
    rec = score_run(results, references, config, gates, anchor)  # the score record (records.schema.json #/$defs/score)

`score.py` is the scoring module of the task (it is frozen by sha256 at selection and vendored into the task body's
tests/). `audit.py` is the reference implementation of the submission gate's static code-as-policy audit and of the
submission digest (no_late_write).
"""

from .score import (CELL_FAILURES, GATE_KEYS, SCORE_SCHEMA, ScoringError, check_summary, clip01, config_sha256,
                    lap_summary, report, score_run, score_venue, venue_statistics, virtual_rank)

__all__ = ["score_venue", "score_run", "report", "lap_summary", "check_summary", "venue_statistics", "virtual_rank",
           "clip01", "config_sha256", "ScoringError", "SCORE_SCHEMA", "GATE_KEYS", "CELL_FAILURES"]
