#!/usr/bin/env bash
# Oracle: the privileged reference controller, run in place of the agent by `ale validate` as the same unprivileged
# account. It installs controller/controller.py (the reference of record as a submitted controller) and the 256-bit
# oracle token as controller/oracle_token; the grader compares the sha256 of the token's stripped bytes with
# tests/hidden/oracle_token.sha256 and, on a match, hands the truth object (the drawn unit, its site and its shift) to
# setup, with which the controller computes the reference's frames cycle by cycle. Without the token it holds the
# hand-over pose.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission/controller"
A="$(dirname "$0")/reference"
mkdir -p "$SUB"
cp "$A/controller.py" "$SUB/controller.py"
cp "$A/oracle_token" "$SUB/oracle_token"
echo "[oracle] submission ready at $SUB" >&2
