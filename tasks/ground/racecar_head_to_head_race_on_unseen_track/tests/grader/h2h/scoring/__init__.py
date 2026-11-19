"""Scoring of racecar_head_to_head_race_on_unseen_track: heat, venue, score and reward.

    from h2h.scoring import score_run, report
    rec = score_run(results, references, config, gates, anchor_value)     # the score record
    rep = report(results, references, config, gates, anchor_value)        # score, statistics, record checks

`score.py` is the score of the task (the verifier calls `score_run`); `crosscheck.py` computes the same numbers a
second time from the records' lists, apart from it; `score.schema.json` holds the schemas of the venue reference and
of the score record.
"""

from .score import (C_REF_MIN, CELL_FAILURES, GATE_KEYS, REFERENCE_SCHEMA, RESULT_SCHEMA, SCORE_SCHEMA, TERM_KEYS,
                    ScoringError, check_reference, check_scoring_config, clip01, config_sha256, field_size, heat_terms,
                    heats_of, make_reference, record_errors, report, score_run, score_venue, venue_statistics)

__all__ = ["score_run", "score_venue", "heat_terms", "report", "venue_statistics", "check_reference",
           "check_scoring_config", "make_reference", "record_errors", "heats_of", "field_size", "clip01",
           "config_sha256", "ScoringError", "SCORE_SCHEMA", "RESULT_SCHEMA", "REFERENCE_SCHEMA", "GATE_KEYS",
           "CELL_FAILURES", "C_REF_MIN", "TERM_KEYS"]
