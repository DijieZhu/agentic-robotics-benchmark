#!/usr/bin/env bash
# Oracle: the privileged reference program, run in place of the agent by `ale validate` as the same unprivileged
# account. It installs drive.py with its module folder s7ref/ (the reference program, byte for byte)
# and the 256-bit token as `oracle_token` next to drive.py, where the program reads it. The grader runs this submission
# exactly as any other; the cell holds the sha256 of the token (tests/hidden/oracle_token.sha256) and answers the
# privileged calls (the venue's true geometry and floor, the car's true state, the other cars' state and lines) only to a program
# that unlocks them with a token of that sha256, so the program is of no use without it. The program still drives only
# through the public step call, under the same staff rules and budgets.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
A="$(dirname "$0")/reference"
mkdir -p "$SUB"
cp "$A/drive.py" "$SUB/"
cp -r "$A/s7ref" "$SUB/"
cp "$A/oracle_token" "$SUB/oracle_token"
echo "[oracle] submission ready at $SUB" >&2
