#!/usr/bin/env bash
# Reference solution: the privileged reference program, run in place of the agent by `ale validate` as the same
# unprivileged account. It installs assemble.py (one frame of each camera, then one privileged call) and the 256-bit
# token as .oracle_token; the grader compares the token's sha256 with tests/hidden/oracle_token.sha256 and, on a match,
# gives this submission's cells the privileged call `reference_assemble` (the reference program inside the cell: it
# weighs, trims, turns and fits every unit) on the same four fixed hidden episodes as every other submission.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
HERE="$(cd "$(dirname "$0")" && pwd -P)"
mkdir -p "$SUB"
cp "$HERE/assemble.py" "$SUB/assemble.py"
printf '%s\n' "6bedaf3de93b1ebe76e7da91ed113d75833b5dd57c73417a959b83aec4f79fe1" > "$SUB/.oracle_token"
echo "[reference] submission ready at $SUB" >&2
