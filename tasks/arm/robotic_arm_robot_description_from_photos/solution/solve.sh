#!/usr/bin/env bash
# Oracle: the photographed arm's own description (the true robot.xml, scene.xml, assets and its exact calibration.json),
# run in place of the agent by `ale validate` as the same unprivileged account. Scores 1.0 on every gate and criterion
# by construction: every band is normalised by the ground truth, and the downstream criterion's full end sits above the
# servos' tick quantisation.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
SRC="$(dirname "$0")/answer"
mkdir -p "$SUB"
cp -r "$SRC"/. "$SUB/"
echo "[oracle] reference description installed at $SUB ($(ls "$SUB/assets" | wc -l) assets)" >&2
