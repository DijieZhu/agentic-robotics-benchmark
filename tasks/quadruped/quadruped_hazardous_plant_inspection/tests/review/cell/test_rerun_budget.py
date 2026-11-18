"""verify.py's re-run budget: at most one mission per verify is run again; a second mission in
which the cell failed is a task error. No cell: verify's `_episode_with_retry` with a stand-in `run_episode` whose
missions fail on the cell's side where the case says so.

    python -I -B tests/review/cell/test_rerun_budget.py

Cases (six sites each):
  A. no cell failure: six missions, 0 re-runs;
  B. the cell fails in site 3's first run: site 3 run again (its second run counts), 1 re-run, every site scored;
  C. the cell fails in site 2 and in site 5: site 2 run again, site 5 ends the verify as a task error (SystemExit);
  D. an internal error after the mission's end (internal_errors 1, internal_errors_live 0) is no cell failure: no re-run.
"""

import os
import sys

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TESTS)
sys.dont_write_bytecode = True        # (no __pycache__ in the task tree from a review run)

import verify  # noqa: E402


def run_case(fail_runs):
    """fail_runs: the run tags whose summary says the cell failed. Returns (summaries or 'task error', runs, metrics)."""
    runs = []

    def fake_run_episode(cfg, site, rn, privileged, tag):
        runs.append(tag)
        s = {"episode_score": 0.5, "internal_errors": 0, "internal_errors_live": 0, "frames_lost": 0,
             "done_reason": "finish", "end_reason": "finish"}
        if tag in fail_runs:
            s.update(internal_errors=1, internal_errors_live=1, done_reason="internal_error")
        if tag == "after_end":
            s.update(internal_errors=1, internal_errors_live=0)
        return s
    orig = verify.run_episode
    verify.run_episode = fake_run_episode
    metrics = {"episode_retries": 0.0}
    out = []
    try:
        for i in range(6):
            out.append(verify._episode_with_retry({}, "site%d" % (i + 1), None, False, fail_runs.get("tag%d" % i,
                                                  "ep%02d" % (i + 1)), metrics))
    except SystemExit:
        return "task error", runs, metrics
    finally:
        verify.run_episode = orig
    return out, runs, metrics


def main():
    problems = []
    res, runs, m = run_case({})
    if res == "task error" or len(runs) != 6 or m["episode_retries"] != 0:
        problems.append("A: %s %s" % (runs, m))
    res, runs, m = run_case({"ep03": 1})
    if res == "task error" or runs != ["ep01", "ep02", "ep03", "ep03r", "ep04", "ep05", "ep06"] or m["episode_retries"] != 1:
        problems.append("B: %s %s %s" % (res if res == "task error" else "ok", runs, m))
    res, runs, m = run_case({"ep02": 1, "ep05": 1})
    if res != "task error" or runs != ["ep01", "ep02", "ep02r", "ep03", "ep04", "ep05"]:
        problems.append("C: %s %s %s" % (res if res == "task error" else "no task error", runs, m))
    res, runs, m = run_case({"tag0": "after_end"})
    if res == "task error" or len(runs) != 6 or m["episode_retries"] != 0:
        problems.append("D: %s %s" % (runs, m))
    for p in problems:
        print("PROBLEM", p)
    print("re-run budget: %s" % ("OK" if not problems else "FAIL (%d problems)" % len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
