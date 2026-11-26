#!/usr/bin/env bash
# Verify stage: root, cwd = this staged folder, after the agent phase, in a SEPARATE verifier sandbox (task.toml): a
# fresh sandbox from the task image into which the engine restored only /home/user/submission. setup.sh has not run
# there and nothing the agent left in its own sandbox is present, so every program this script runs (stat, chmod,
# dirname, the verify Python) is the image's own. The engine uploads the folder world-readable, so the FIRST thing done
# here, before the interpreter starts, is to close the folder and its parent to every other account (verify.py records
# the original modes from the two variables, kills and freezes the agent account's processes, snapshots the
# submission within its caps, locks every file and restores the modes at the end). The submitted controller runs only
# behind the runner (tests/runner), in its own process under the runner's system-call filter; the hidden world flies
# each hidden session and the grader scores what the world's true state records.
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
export NPY_DISABLE_CPU_FEATURES="AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL" OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256
exec "${ALE_VERIFY_PYTHON:-python3}" -B verify.py
