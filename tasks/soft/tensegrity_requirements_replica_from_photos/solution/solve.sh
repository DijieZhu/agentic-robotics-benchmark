#!/usr/bin/env bash
# Oracle (privileged reference): the three deliverables transcribed from the hidden truth ledger (tests/gt/ledger.json)
# by the maintainers' build; nothing is solved here. `ale validate` runs this in place of the agent,
# as the same unprivileged account; the grader scores it exactly 1.0 on every criterion (tests/anchor.json).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
HERE="$(cd "$(dirname "$0")" && pwd -P)"
mkdir -p "$SUB"
cp "$HERE/answer/requirements.yaml" "$HERE/answer/bom.json" "$HERE/answer/cable_net.json" "$SUB/"
echo "[oracle] submission ready at $SUB" >&2
