#!/usr/bin/env bash
# Oracle: the privileged reference program, run in place of the agent by `ale validate` as the same unprivileged
# account. It installs assemble.py (one frame of each camera, then one privileged call) and the 256-bit oracle token as
# .oracle_token; the grader compares the token's sha256 with tests/hidden/oracle_token.sha256 and, on a match, gives
# this submission's cells the bench plan's nominal layout and the privileged call `reference_assemble` (the reference
# program inside the cell).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
A="$(dirname "$0")/reference"
mkdir -p "$SUB"
cp "$A/assemble.py" "$SUB/"
cp "$A/oracle_token" "$SUB/.oracle_token"
echo "[oracle] submission ready at $SUB" >&2
