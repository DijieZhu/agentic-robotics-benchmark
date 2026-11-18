#!/usr/bin/env bash
# Oracle: the ground-truth answer of the hidden instance (the true kinematic sketch, the requirements sheet valued in
# the real world with the true derived items of the cheapest accepted BOM, and that BOM), run in place of the agent by
# `ale validate` as the same unprivileged account. Scores 1.0 on every criterion by construction.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
mkdir -p "$SUB"
cp "$(dirname "$0")/answer/requirements.yaml" "$(dirname "$0")/answer/bom.json" "$(dirname "$0")/answer/kinematic_sketch.json" "$SUB/"
echo "[oracle] submission ready at $SUB" >&2
