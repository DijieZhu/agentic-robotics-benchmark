#!/usr/bin/env bash
# Oracle: the privileged reference program, run in place of the agent by `ale validate` as the same unprivileged
# account.  It installs caretake.py (one privileged call, two lines) and the 256-bit oracle token as .oracle_token;
# the verifier compares the token's sha256 with tests/hidden/oracle_token.sha256 and, on a match, gives this
# submission's cells the privileged call `reference_run` (tests/reference/reference.py inside the cell: the five
# privileged reads the reference may make, every move through the public calls; it takes one frame of every camera).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
A="$(dirname "$0")/reference"
mkdir -p "$SUB"
cp "$A/caretake.py" "$SUB/"
cp "$A/oracle_token" "$SUB/.oracle_token"
echo "[oracle] submission ready at $SUB" >&2
