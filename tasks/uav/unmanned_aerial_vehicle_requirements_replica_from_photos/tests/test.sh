#!/usr/bin/env bash
# Verify stage: root, cwd = this staged folder, after the agent phase, in a SEPARATE verifier sandbox (task.toml): a
# fresh sandbox from the task image into which the engine restored only /home/user/submission. setup.sh has not run
# there and nothing the agent left in its own sandbox is present, so every program this script runs (chmod, the verify
# Python) is the image's own. The engine uploads the folder world-readable, so the first thing done here, before the
# interpreter starts, is to close the folder and its parent to every other account; verify.py then freezes and kills
# the agent account's processes (in the separate verifier only the image's own desktop session), snapshots the
# submission, gates late writes and locks the stage. The grader executes no agent code (static YAML / JSON).
set -euo pipefail
chmod 700 . ..
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec "${ALE_VERIFY_PYTHON:-python3}" -I -B verify.py
