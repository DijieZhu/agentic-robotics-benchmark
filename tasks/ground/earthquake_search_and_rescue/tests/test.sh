#!/usr/bin/env bash
# Verify stage: runs as root, cwd = this staged folder, after the agent phase, in a SEPARATE verifier sandbox (task.toml
# [verifier] environment_mode = "separate"): a fresh sandbox from the task image that the agent never touched, holding
# only the declared artifact (/home/user/submission, restored root-owned); environment/setup/setup.sh never runs here.
# The engine uploads this folder with modes 0644 / 0755, so the FIRST thing done here -- before the interpreter starts --
# is to close the folder and its parent to every other account (one chmod each; verify.py records the original modes
# from the two variables and restores them at the end, then kills the agent account's leftover processes, snapshots the
# submission, gates late writes and locks every file).
#
# The grader runs as root with nothing of the agent home on its import or configuration paths: HOME and the XDG /
# matplotlib folders point at root's own, PYTHONNOUSERSITE=1 and `-s` keep any user site-packages out, and the cell
# renders through OSMesa (the training ground's cell uses the same path).
set -euo pipefail
STAGE="$(pwd -P)"
export ARB_STAGE_MODE="$(stat -c %a "$STAGE")"
chmod 700 "$STAGE"
PARENT="$(dirname "$STAGE")"
if [ "$PARENT" != "/" ] && [ "$PARENT" != "${ALE_HOME:-/home/user}" ]; then
  export ARB_STAGE_PARENT_MODE="$(stat -c %a "$PARENT")"
  chmod 700 "$PARENT"
fi
# The grader's root-owned, read-only copy of the givens (/opt/arb): verify.py checks it file by file against
# given_hashes.json and hands its client library to the programs, so nothing the agent wrote is an input of the grader.
# It holds what environment/setup/setup.sh stages into the agent home: setup_payload/, this folder's byte copy of
# environment/setup/payload/ (which never reaches the verifier; change both together, then given_hashes.json).
ROOT_PAYLOAD="${ARB_ROOT_PAYLOAD:-/opt/arb}"
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD"
cp -r "$STAGE/setup_payload/." "$ROOT_PAYLOAD/"
chown -R root:root "$ROOT_PAYLOAD"
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
export HOME=/root PYTHONNOUSERSITE=1 XDG_CONFIG_HOME=/root/.config XDG_CACHE_HOME=/root/.cache \
  MPLCONFIGDIR=/root/.config/matplotlib
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
# the cell's and the reference program's numerics the same on every x86-64 CPU: OpenBLAS otherwise picks its kernels
# per CPU and numpy its SIMD paths; the program under test keeps its own minimal environment (cellkit.runner.child_env)
export OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
exec "${ALE_VERIFY_PYTHON:-python3}" -s verify.py
