#!/usr/bin/env bash
# Oracle: the privileged reference program, run in place of the agent by `ale validate` as the same unprivileged
# account. It installs operate.py (one privileged call) and the 256-bit oracle token as .oracle_token; the grader
# compares the token's sha256 with tests/hidden/oracle_token.sha256 and, on a match, gives this submission's cells the
# privileged call `reference_run` (tests/reference/reference.py inside the cell: the true pose and the site's true
# layout for planning; every command goes through the station's own calls; it never takes a camera frame).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
A="$(dirname "$0")/reference"
mkdir -p "$SUB"
cp "$A/operate.py" "$SUB/"
cp "$A/oracle_token" "$SUB/.oracle_token"
echo "[oracle] submission ready at $SUB" >&2
