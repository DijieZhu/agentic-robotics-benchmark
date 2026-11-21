#!/usr/bin/env bash
# Verify stage of freeflyer_uncrewed_station_caretaking: runs as root, cwd = this staged folder, after the agent phase,
# in a SEPARATE verifier sandbox (task.toml [verifier] environment_mode = "separate"): a fresh sandbox from the task
# image that the agent never touched, holding only the declared artifact (/home/user/submission, restored root-owned);
# environment/setup/setup.sh never runs here. The engine uploads this folder with modes 0644 / 0755, so the FIRST thing
# done here -- before the interpreter starts -- is to close the folder and its parent to every other account (one
# chmod each; verify.py records the original modes from the two variables and restores them at the end, then kills the
# agent account's leftover processes, snapshots the submission, gates late writes and locks every file).
#
# The grader runs as root with nothing of the agent home on its import or configuration paths (HOME and the XDG
# folders point at root's own, PYTHONNOUSERSITE=1 and `-s` keep any user site-packages out). The cell renders every
# camera through MuJoCo's classic renderer on OSMesa (cabincell/sensors_render.py); 8
# rasteriser threads (the setting the frame times were measured with). The cell's numerics: one
# BLAS / OpenMP thread and the CPU pins of the runtime key cell.recipe.env, set here because numpy reads them
# when it is imported; verify.py checks them against the runtime spec (a difference is a task error). The forced kernel
# always comes with OPENBLAS_L2_SIZE=256: without it, on AMD Zen 4 / Zen 5 hosts numpy's bundled OpenBLAS (0.3.33-0.3.34)
# gives the forced table 512-row blocks and its Haswell kernel overruns a stack buffer; verify.py refuses the pin without
# it and runs a BLAS self-test in child processes before the cell starts (a failure is a task error).
#
# The cell's three asset components (robot, station, robot_author) are read from the asset root CABINCELL_ASSET_ROOT,
# default /opt/task-assets (the image's root-owned copy, environment/Dockerfile): verify.py checks every archive there
# against the sha256 pinned in cabincell/data/<component>/ASSETS.json and every unpacked component file by file
# against that manifest before any run; a missing or altered component is a task error, never a run without it.
#
# Host-side runs (the maintainers' runs, as another account) use the same file: ARB_ROOT_PAYLOAD, ARB_RECORD_ROOT, ALE_HOME and
# CABINCELL_ASSET_ROOT name scratch locations; HOME is kept, and nothing is chowned.
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
# given_hashes.json and hands its client library (given/lib/cabinkit) to the programs, so nothing the agent wrote is an
# input of the grader. It holds what environment/setup/setup.sh stages into the agent home: setup_payload/, this
# folder's byte copy of environment/setup/payload/ (which never reaches the verifier; change both together, then
# given_hashes.json). The cell's as-surveyed map is read from setup_payload/given/station_map (cabincell/server.py).
ROOT_PAYLOAD="${ARB_ROOT_PAYLOAD:-/opt/arb}"
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD"
cp -r "$STAGE/setup_payload/." "$ROOT_PAYLOAD/"
# The givens' pictures (figures, catalogue photographs, map views) are task assets of the image (environment/Dockerfile:
# <asset root>/payload/given/): copied in beside setup_payload/ as setup.sh stages them into the agent home;
# given_hashes.json pins them too, so an image without them is a task error, never a run without them.
ASSET_PAYLOAD="${CABINCELL_ASSET_ROOT:-/opt/task-assets}/payload"
[ ! -d "$ASSET_PAYLOAD" ] || cp -r "$ASSET_PAYLOAD/." "$ROOT_PAYLOAD/"
if [ "$(id -u)" = 0 ]; then
  chown -R root:root "$ROOT_PAYLOAD"
  export HOME=/root PYTHONNOUSERSITE=1 XDG_CONFIG_HOME=/root/.config XDG_CACHE_HOME=/root/.cache \
    MPLCONFIGDIR=/root/.config/matplotlib
fi
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
export PYTHONNOUSERSITE=1
export MUJOCO_GL="${MUJOCO_GL:-osmesa}" PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-osmesa}" LP_NUM_THREADS=8 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CABINCELL_ASSET_ROOT="${CABINCELL_ASSET_ROOT:-/opt/task-assets}"
# the cell's and the reference program's numerics the same on every x86-64 CPU (runtime key cell.recipe.env)
export OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4,AVX512_ICL"
exec "${ALE_VERIFY_PYTHON:-python3}" -s verify.py
