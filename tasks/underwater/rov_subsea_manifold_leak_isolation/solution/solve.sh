#!/usr/bin/env bash
# Oracle: the privileged reference program, run in place of the agent by `ale validate` as the same unprivileged
# account. It installs intervene.py (one privileged call) and the 256-bit oracle token as .oracle_token; the grader
# compares the token's sha256 with tests/hidden/oracle_token.sha256 and, on a match, gives this submission's cells the
# privileged calls `get_state_privileged` and `reference_run` (the reference program inside the cell, with the true
# poses of the vehicle and the structure; it finds the leak, reads the tags and the handles from its own frames and
# sonar returns).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
A="$(dirname "$0")/reference"
mkdir -p "$SUB"
cp "$A/intervene.py" "$SUB/"
cp "$A/oracle_token" "$SUB/.oracle_token"
echo "[oracle] submission ready at $SUB" >&2
