#!/usr/bin/env bash
# The task's solution: the sensor-only reference controller with its access token, installed as the submission's
# controller folder (run as the agent's account at the end of the agent phase). The grader compares the token's
# sha256 with its own copy and, on a match, hands the session's registered reads to the controller at set-up.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission/controller"
SRC="$(cd "$(dirname "$0")" && pwd)/controller"
mkdir -p "$SUB"
cp -R "$SRC/." "$SUB/"
echo "[solution] the reference controller is installed as the submission" >&2
