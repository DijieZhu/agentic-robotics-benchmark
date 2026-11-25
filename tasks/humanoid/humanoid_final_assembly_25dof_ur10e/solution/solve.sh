#!/usr/bin/env bash
# Reference solution: the privileged reference program, run in place of the agent by `ale validate` as the same
# unprivileged account. It installs assemble.py with its package href/ (the program's modules and its grasp table) and
# the 256-bit token as .oracle_token; the grader compares the token's sha256 with tests/hidden/oracle_token.sha256 and,
# on a match, gives this submission's cells the privileged reads (get_state_privileged, describe_privileged: the true
# poses) and the nominal layout. The program moves the arm only through the cell's public calls, like any program.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
HERE="$(cd "$(dirname "$0")" && pwd -P)"
mkdir -p "$SUB"
cp "$HERE/reference/assemble.py" "$SUB/assemble.py"
rm -rf "$SUB/href"
cp -r "$HERE/reference/href" "$SUB/href"
cp "$HERE/reference/oracle_token" "$SUB/.oracle_token"
echo "[reference] submission ready at $SUB" >&2
