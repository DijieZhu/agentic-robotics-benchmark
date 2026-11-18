#!/usr/bin/env bash
# Oracle: the privileged reference program, run in place of the agent by `ale validate` as the same unprivileged
# account. It installs patrol.py (one privileged call) and the 256-bit oracle token as .oracle_token; the grader
# compares the token's sha256 with tests/hidden/oracle_token.sha256 and, on a match, gives this submission's cells the
# privileged calls `get_state_privileged` and `reference_run` (the reference program inside the cell, with the true
# pose and the site's true layout; it reads every value from its own frames).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
A="$(dirname "$0")/reference"
mkdir -p "$SUB"
cp "$A/patrol.py" "$SUB/"
cp "$A/oracle_token" "$SUB/.oracle_token"
echo "[oracle] submission ready at $SUB" >&2
