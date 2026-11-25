#!/usr/bin/env bash
# Verify stage: runs as root, cwd = this staged folder, after the agent phase, in a SEPARATE verifier sandbox (task.toml
# [verifier] environment_mode = "separate"): a fresh sandbox from the task image that the agent never touched, holding
# only the declared artifact (/home/user/submission, restored root-owned); environment/setup/setup.sh never runs here.
# The engine uploads this folder with modes 0644 / 0755, so the FIRST thing done here -- before the interpreter starts --
# is to close the folder and its parent to every other account (one chmod each; verify.py records the original modes
# from the two variables and restores them at the end). Precedent: quadruped_hazardous_plant_inspection/tests/test.sh.
#
# The grader runs as root with nothing of the agent home on its import or configuration paths: HOME and the XDG /
# matplotlib folders point at root's own, PYTHONNOUSERSITE=1 and `-s` keep any user site-packages out. The cell renders
# the search camera on the CPU (the practice ground's cell uses the same path and settings): the robot's depth and mask
# by MuJoCo's classic renderer on OSMesa, the robot's colour by MuJoCo's physically based renderer on Mesa's software
# Vulkan driver, into the site's plate bundle; it never writes Mesa's shader cache. Every thread count is pinned here,
# so that grading does not depend on the image's variables: the physics is single-threaded (the simulation base), 8
# rasteriser threads for the frames, BLAS 2 (numpy, scipy: the privileged reference's planner).
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
# given_hashes.json and hands its client library (given/lib/voidcell) to the programs, so nothing the agent wrote is an
# input of the grader. It holds what environment/setup/setup.sh stages into the agent home: setup_payload/, this
# folder's byte copy of environment/setup/payload/ (change both together, then given_hashes.json).
ROOT_PAYLOAD="${ARB_ROOT_PAYLOAD:-/opt/arb}"
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD"
cp -r "$STAGE/setup_payload/." "$ROOT_PAYLOAD/"
chown -R root:root "$ROOT_PAYLOAD"
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
# The hidden sites' plate bundles (plates.tar.xz) before they are on the asset dataset: the local copy in this folder
# replaces the download (verify.py checks it against the sha256 and the file count grader_config.json pins, exactly as
# a download; with no hidden_plates in grader_config.json nothing is fetched). Once published, assets/ is not in the
# task folder and the pinned URL is used.
if [ -f "$STAGE/assets/plates.tar.xz" ]; then
  export ARB_HIDDEN_PLATES_TARBALL="$STAGE/assets/plates.tar.xz"
fi
export HOME=/root PYTHONNOUSERSITE=1 XDG_CONFIG_HOME=/root/.config XDG_CACHE_HOME=/root/.cache \
  MPLCONFIGDIR=/root/.config/matplotlib
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa MESA_SHADER_CACHE_DISABLE=true LP_NUM_THREADS=8 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=1
# the cell's and the reference program's numerics the same on every x86-64 CPU: OpenBLAS otherwise picks its kernels
# per CPU and numpy its SIMD paths. The L2 size 256 keeps the Haswell kernel's built-in dgemm blocking: numpy 2.5's
# OpenBLAS otherwise raises it on Zen 4 / Zen 5 CPUs, where a product with an inner dimension above about 300 then
# crashes or returns wrong numbers. The program under test gets the same three values in its own minimal environment
# (verify.py BLAS_PIN, through Runner.env)
export OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
exec "${ALE_VERIFY_PYTHON:-python3}" -s verify.py
