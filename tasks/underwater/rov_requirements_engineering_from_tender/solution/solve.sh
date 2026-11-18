#!/usr/bin/env bash
# Reference solution: a static submission, no token and no privileged channel. solution/submission/ holds the three
# files written from the hidden world by the development pipeline: the evaluation (every bid's verdict with the
# cheapest fit option subset or the grounds that hold, the award and the site predictions of every accepted bid) is
# the frozen world's own truth table, the QA labels of the 24 template clauses are the world's labels, and the
# acceptance schedule is the reference clause table for the award's delivery band. Runs as the agent account in the
# agent's sandbox; asks the enquiry and witness desk nothing (no budget spent) and reads nothing outside solution/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUB="${ALE_HOME:-/home/user}/submission"
mkdir -p "$SUB"
cp "$HERE"/submission/evaluation.json "$HERE"/submission/spec_review.json "$HERE"/submission/acceptance_schedule.json "$SUB/"
echo "[reference] submission ready at $SUB" >&2
