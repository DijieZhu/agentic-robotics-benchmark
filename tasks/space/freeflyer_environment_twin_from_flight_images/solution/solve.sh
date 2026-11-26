#!/usr/bin/env bash
# The privileged reference answer: per flight, the twin built by the maintainers from the withheld depth record of
# that flight (scene.xml and its convex mesh pieces) and the wide camera's pose at every given frame of the flight (cameras.json),
# copied into the submission as it is. It is the twin the grader's scored sets are restricted to, so it scores 1.0
# by construction.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
HERE="$(cd "$(dirname "$0")" && pwd -P)"
mkdir -p "$SUB"
cp -r "$HERE/answer/." "$SUB/"
echo "[solve] submission ready at $SUB" >&2
