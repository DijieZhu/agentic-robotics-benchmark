#!/usr/bin/env bash
# Verify stage: runs as root, cwd = this staged folder, after the agent phase, in a SEPARATE verifier sandbox (task.toml
# [verifier] environment_mode = "separate"): a fresh sandbox from the task image that the agent never touched, holding
# only the declared artifact (/home/user/submission, restored root-owned). environment/setup/setup.sh never runs here:
# there is no ~/given and no process the agent started. The engine uploads this folder world-readable,
# so the FIRST thing done here -- before the interpreter starts -- is to close the folder and its parent to every other
# account (one chmod each; verify.py records the original modes from the two variables and restores them at the end,
# then kills the agent account's processes, snapshots the submission, gates late writes and locks every file; in the
# separate verifier the agent account's processes are the image's own desktop session, started with the sandbox).
#
# The grader's root-owned, read-only copy of the givens (/opt/arb) is built next: verify.py checks it file by file
# against given_hashes.json, runs the programs with its client library (/opt/arb/given/lib) and hands it to them as
# ARB_GIVEN_DIR, so nothing the agent wrote is an input of the grader. It holds what setup.sh stages into the agent
# home: setup_payload/, this folder's byte copy of environment/setup/payload/ (which never reaches the verifier;
# scripts/lint_domain.py checks that the two stay identical: change both, then given_hashes.json), the image's robot
# model (/opt/arb-assets/unitree_g1 -> given/robot/unitree_g1) and the image's survey photos (/opt/arb-assets/survey ->
# given/cell/survey), owned by root, group/world read-only.
#
# The grader runs as root with nothing of the agent home on its import or configuration paths: HOME and the XDG /
# matplotlib folders point at root's own, PYTHONNOUSERSITE=1 and `-s` keep any user site-packages out, and the cell's
# cameras render their depth through OSMesa and their colour with MuJoCo's Filament renderer on lavapipe (the paths the
# survey photos were rendered on).
set -euo pipefail
STAGE="$(pwd -P)"
export ARB_STAGE_MODE="$(stat -c %a "$STAGE")"
chmod 700 "$STAGE"
PARENT="$(dirname "$STAGE")"
if [ "$PARENT" != "/" ] && [ "$PARENT" != "${ALE_HOME:-/home/user}" ]; then
  export ARB_STAGE_PARENT_MODE="$(stat -c %a "$PARENT")"
  chmod 700 "$PARENT"
fi
ROOT_PAYLOAD="${ARB_ROOT_PAYLOAD:-/opt/arb}"
G1_DIR="${ARB_G1_DIR:-/opt/arb-assets/unitree_g1}"
SURVEY_DIR="${ARB_SURVEY_DIR:-/opt/arb-assets/survey}"
[ -d "$STAGE/setup_payload/given" ] || { echo "[verify] setup_payload/ missing from the verify stage" >&2; exit 1; }
[ -d "$G1_DIR" ] || { echo "[verify] $G1_DIR missing: the image was built without the robot model" >&2; exit 1; }
[ -d "$SURVEY_DIR" ] || { echo "[verify] $SURVEY_DIR missing: the image was built without the survey photos" >&2; exit 1; }
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD/given/robot"
cp -r "$STAGE/setup_payload/." "$ROOT_PAYLOAD/"
cp -r "$G1_DIR" "$ROOT_PAYLOAD/given/robot/unitree_g1"
cp -r "$SURVEY_DIR" "$ROOT_PAYLOAD/given/cell/survey"
chown -R root:root "$ROOT_PAYLOAD"
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
export HOME=/root PYTHONNOUSERSITE=1 XDG_CONFIG_HOME=/root/.config XDG_CACHE_HOME=/root/.cache \
  MPLCONFIGDIR=/root/.config/matplotlib
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
# The cell (physics, cleaning processes, the oracle's reference planner) runs in this process: one BLAS thread and one
# pinned OpenBLAS kernel set, so the same scene replays the same numbers on any x86-64 grading machine (OpenBLAS
# otherwise picks its kernels by CPU). The program under test gets its own environment (cellkit.runner.child_env).
export OPENBLAS_CORETYPE=Haswell OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec "${ALE_VERIFY_PYTHON:-python3}" -s verify.py
