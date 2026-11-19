#!/usr/bin/env bash
# Reference solution: a static submission, no token and no privileged channel. solution/submission/ holds the three
# files generated from the hidden truth by the task's generators: the sketch is the real car's key quantities,
# the requirements sheet the reference sheet with the procedure's own values for the BOM, the BOM the cheapest
# acceptable car (tests/README.md). Runs as the agent account in the agent's sandbox and reads nothing outside
# solution/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SUB="${ALE_HOME:-/home/user}/submission"
mkdir -p "$SUB"
cp "$HERE"/submission/requirements.yaml "$HERE"/submission/bom.json "$HERE"/submission/kinematic_sketch.json "$SUB/"
echo "[oracle] submission ready at $SUB" >&2
