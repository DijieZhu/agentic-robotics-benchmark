#!/usr/bin/env bash
# Verify stage: runs as root, cwd = this staged folder, after the agent phase, in a SEPARATE verifier sandbox (task.toml
# [verifier] environment_mode = "separate"): a fresh sandbox from the task image that the agent never touched, holding
# only the declared artifact (/home/user/submission, restored root-owned); environment/setup/setup.sh never runs here.
# The engine uploads this folder with modes 0644 / 0755, so the FIRST thing done here -- before the interpreter starts --
# is to close the folder and its parent to every other account (one chmod each; verify.py records the original modes
# from the two variables and restores them at the end, then kills the agent account's leftover processes, snapshots the
# submission, gates late writes and locks every file; in a separate verifier the agent left no process and wrote
# nothing, so these only stop what the image's own session of the agent account does).
#
# The grader runs as root with nothing of the agent home on its import or configuration paths: HOME and the XDG /
# matplotlib folders point at root's own, PYTHONNOUSERSITE=1 and `-s` keep any user site-packages out, and the cell
# renders every camera's colour through MuJoCo's Filament renderer on Vulkan (Mesa's lavapipe, the image's only
# driver) and depth through MuJoCo's classic renderer on OSMesa (the commissioning service's cell uses the same path
# and settings). Every thread count is exported here, so that grading does not depend on the image's variables
# (environment/Dockerfile sets values for the agent's sandbox): 8 rasteriser threads (the grader renders for one shift
# at a time), BLAS 2, OpenMP 2.
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
# given_hashes.json and hands its client library to the programs (and the folder to them as ARB_GIVEN_DIR), so nothing
# the agent wrote is an input of the grader. It holds what environment/setup/setup.sh stages into the agent home:
# setup_payload/, this folder's byte copy of environment/setup/payload/ (which never reaches the verifier; change both
# together, then given_hashes.json).
# The robot's mesh files and the gripper's licence are not in this folder: they come from the image's task assets
# (/opt/task-assets/payload, fetched at the image's build from the benchmark's asset repository at the pinned commit),
# as environment/setup/setup.sh stages them into the agent home; given_hashes.json pins them at the same paths. The
# grader's cell (parcelcell/assets/ here) gets the same mesh files.
ROOT_PAYLOAD="${ARB_ROOT_PAYLOAD:-/opt/arb}"
ASSETS=/opt/task-assets/payload
[ -d "$ASSETS/given" ] || { echo "[verify] $ASSETS missing: the image was built without its task assets" >&2; exit 1; }
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD"
cp -r "$STAGE/setup_payload/." "$ROOT_PAYLOAD/"
cp -r "$ASSETS/." "$ROOT_PAYLOAD/"
mkdir -p "$STAGE/parcelcell/assets/robot/meshes" "$STAGE/parcelcell/assets/gripper_2f85/assets"
cp "$ASSETS"/given/robot/meshes/*.STL "$STAGE/parcelcell/assets/robot/meshes/"
cp "$ASSETS"/given/robot/gripper_2f85/assets/*.stl "$STAGE/parcelcell/assets/gripper_2f85/assets/"
chown -R root:root "$ROOT_PAYLOAD"
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
export HOME=/root PYTHONNOUSERSITE=1 XDG_CONFIG_HOME=/root/.config XDG_CACHE_HOME=/root/.cache \
  MPLCONFIGDIR=/root/.config/matplotlib
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2
# the cell's and the reference program's numerics the same on every x86-64 CPU: OpenBLAS otherwise picks its kernels
# per CPU and numpy its SIMD paths; the program under test keeps its own minimal environment (cellkit.runner.child_env:
# no pin). OPENBLAS_L2_SIZE=256 guards the kernel pin: the OpenBLAS that numpy 2.5.2 to 2.5.5 bundle (2.5.3 here)
# raises its block size on AMD Zen 4 / Zen 5 hosts even for a forced kernel, and the Haswell kernel can then crash or
# return wrong numbers for float64 products with an inner dimension above about 300; with the size kept at 256 it
# computes as on every other CPU.
export OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
exec "${ALE_VERIFY_PYTHON:-python3}" -s verify.py
