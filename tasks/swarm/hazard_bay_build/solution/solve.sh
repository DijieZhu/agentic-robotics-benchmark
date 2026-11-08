#!/usr/bin/env bash
# The reference build: the bay and its fleet authored as a USD stage by reference/build_scene.py, plus the reference
# fleet controller (reference/controller.py + vehicle.py behind run.sh: the flying robots' hoist and the vehicle's
# program) (runs Isaac Sim headless on the sandbox's GPU, as the agent's account, within the agent budget). It also
# installs the 256-bit oracle token as .oracle_token: the grader compares its sha256 with tests/hidden/oracle_token.sha256
# and, on a match, adds the true state to every observation of the bay mission (`privileged`), which the reference
# program reads to do the whole job on the fixed hidden set's days, the same days every submission meets.
set -euo pipefail
SUB="${ALE_HOME:-/home/user}/submission"
mkdir -p "$SUB"
/opt/venv/bin/python3 "$(dirname "$0")/reference/build_scene.py" --out "$SUB/scene.usda"
cp "$(dirname "$0")/reference/controller.py" "$(dirname "$0")/reference/vehicle.py" "$(dirname "$0")/reference/run.sh" "$SUB/"
cp "$(dirname "$0")/reference/oracle_token" "$SUB/.oracle_token"
echo "[oracle] scene, controller and token ready in $SUB" >&2
