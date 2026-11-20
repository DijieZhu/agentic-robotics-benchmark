#!/usr/bin/env bash
# Verify stage: runs as root, cwd = this staged folder, after the agent phase, in a SEPARATE verifier sandbox (task.toml
# [verifier] environment_mode = "separate"): a fresh sandbox from the task image that the agent never touched, holding
# only the declared artifact (/home/user/submission, restored root-owned); environment/setup/setup.sh never runs here.
# The engine uploads this folder world-readable, so the FIRST thing done here -- before the interpreter starts -- is to
# close the folder and its parent to every other account (one chmod each; verify.py records the original modes from
# the two variables and restores them at the end, then kills the agent account's leftover processes, snapshots the
# submission, gates late writes and locks every file: the stage lock; in a separate
# verifier the agent left no process and wrote nothing, and these stop the image's own desktop session of the agent
# account and set aside what it writes). verify.py then fetches the cell's kit into this closed folder (gt/).
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
# given_hashes.json, compiles its physics (the Panda) from it, renders the cameras' scenery from it and hands it to the
# programs as ARB_GIVEN_DIR, so nothing the agent wrote is an input of the grader. It holds what
# environment/setup/setup.sh stages into the agent home: the image's task assets and setup_payload/, this folder's byte
# copy of environment/setup/payload/ (which never reaches the verifier; change both together, then given_hashes.json).
ROOT_PAYLOAD="${ARB_ROOT_PAYLOAD:-/opt/arb}"
ASSETS=/opt/task-assets/payload
[ -d "$ASSETS" ] || { echo "[verify] $ASSETS missing: the image was built without its task assets" >&2; exit 1; }
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD"
cp -r "$STAGE/setup_payload/." "$ROOT_PAYLOAD/"
cp -r "$ASSETS/." "$ROOT_PAYLOAD/"
chown -R root:root "$ROOT_PAYLOAD"
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
# The cell's kit (gt.tar.xz) before it is on the asset dataset: the local copy in this folder replaces the
# download (verify.py fetch_hidden_gt checks it against the sha256 and the file count grader_config.json pins, exactly as
# a download). Once published, assets/ is not in the task folder and the pinned URL is used.
if [ -f "$STAGE/assets/gt.tar.xz" ]; then
  export ARB_HIDDEN_GT_TARBALL="$STAGE/assets/gt.tar.xz"
fi
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
# The cell's, the privileged reference's and the grading's numerics the same on every x86-64 CPU: numpy's and
# scipy's OpenBLAS otherwise pick their kernel per CPU (SkylakeX on an AVX-512 host) and numpy its SIMD paths. The L2
# size keeps the Haswell kernel's dgemm blocking at its built-in 256 (numpy 2.5's OpenBLAS otherwise raises it to 512 on
# Zen 4 / Zen 5 CPUs, where a product with an inner dimension above about 300 then overruns the kernel's buffer: a crash
# or wrong numbers); with every inner dimension at most 256 it changes no bit. Set here, before the interpreter starts,
# so before numpy is first imported; the programs' processes get the same three variables from
# arm/assembly_cell/runner.py child_env (their environment is built from scratch).
export OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
exec "${ALE_VERIFY_PYTHON:-python3}" verify.py
