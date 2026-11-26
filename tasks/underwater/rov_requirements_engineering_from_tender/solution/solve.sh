#!/usr/bin/env bash
# Reference solution: a static submission, no token and no privileged channel. solution/submission/ holds the three
# files written from the hidden world by the development pipeline: evaluation.json (every bid's verdict with the
# cheapest fit option subset, or at most two of the grounds that hold for it, the award and the site predictions of
# every accepted bid) is the frozen world's own truth, spec_review.json holds the world's QA labels of the Part B
# clauses, and acceptance_schedules.json holds one schedule per accepted bid, at most 12 clauses each (the contract's
# cap), each sufficient at every tank-water density that bidder declares (the whole of a declared range) and
# economical on its delivery band. Runs as the agent account in the agent's sandbox; asks the enquiry and witness desk
# nothing (no budget spent) and reads nothing outside solution/. It copies exactly the three files the submission
# schemas name (evaluation.json, spec_review.json, acceptance_schedules.json); it checks all three first, so a missing
# one stops it with an error before anything is copied.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUB="${ALE_HOME:-/home/user}/submission"
FILES="evaluation.json spec_review.json acceptance_schedules.json"
for f in $FILES; do
  test -f "$HERE/submission/$f" || { echo "[reference] missing solution/submission/$f" >&2; exit 1; }
done
mkdir -p "$SUB"
for f in $FILES; do
  cp "$HERE/submission/$f" "$SUB/$f"
done
echo "[reference] submission ready at $SUB" >&2
