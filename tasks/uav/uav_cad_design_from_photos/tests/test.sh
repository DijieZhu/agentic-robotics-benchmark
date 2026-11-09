#!/usr/bin/env bash
# Verify stage: root, cwd = this staged folder, after the agent phase, in a separate verifier sandbox (task.toml): a
# fresh sandbox from the task image holding only the restored /home/user/submission, where setup.sh never ran. The
# engine uploads the folder world-readable, so the FIRST thing done here, before the interpreter starts, is to close
# the folder and its parent to every other account (verify.py records the original modes from the two variables,
# freezes and kills the agent account's processes, here the image's own desktop, snapshots the submission, gates late
# writes, locks every file and restores the modes at the end: CHAIN_PLAN 4.0 stage lock, T9). The grader executes no
# agent code (the FreeCAD document is data read by headless FreeCAD).
set -euo pipefail
STAGE="$(pwd -P)"
export ARB_STAGE_MODE="$(stat -c %a "$STAGE")"
chmod 700 "$STAGE"
PARENT="$(dirname "$STAGE")"
if [ "$PARENT" != "/" ] && [ "$PARENT" != "${ALE_HOME:-/home/user}" ]; then
  export ARB_STAGE_PARENT_MODE="$(stat -c %a "$PARENT")"
  chmod 700 "$PARENT"
fi
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${ALE_VERIFY_PYTHON:-python3}" -B verify.py
