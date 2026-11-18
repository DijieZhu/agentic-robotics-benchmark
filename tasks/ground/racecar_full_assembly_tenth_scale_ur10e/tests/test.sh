#!/usr/bin/env bash
# Verify stage: runs as root, cwd = this staged folder, after the agent phase, in a SEPARATE verifier sandbox (task.toml
# [verifier] environment_mode = "separate"): a fresh sandbox from the task image that the agent never touched, holding
# only the declared artifact (/home/user/submission, restored root-owned); environment/setup/setup.sh never runs here.
# The engine uploads this folder world-readable, so the FIRST thing done here -- before the interpreter starts -- is to
# close the folder and its parent to every other account (one chmod each; verify.py records the original modes from
# the two variables and restores them at the end, then kills the runner account's leftover processes, snapshots the
# submission, gates late writes and locks every file; in a separate verifier the agent left no process and wrote
# nothing, and these stop the image's own desktop session of the agent account and set aside what it writes).
# verify.py then unpacks the cell's hidden kit into this closed folder (gt/): from the local development copy
# assets/gt.tar.xz when it is present (an engine asset root, never in git), else downloaded from the benchmark's asset
# dataset (grader_config.json "hidden_gt"); both are checked against the same sha256 and file count. The agent's
# program runs as an unprivileged account under netguard.py (no IP sockets) and talks to the cell over stdio pipes.
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
# given_hashes.json, compiles its physics (the arm and the gripper) from it, renders the cameras' scenery from it and
# hands it to the programs as ARB_GIVEN_DIR, so nothing the agent wrote is an input of the grader. It holds what
# environment/setup/setup.sh stages into the agent home: the image's task assets and setup_payload/, this folder's byte
# copy of environment/setup/payload/ (which never reaches the verifier; change both together, then given_hashes.json).
ROOT_PAYLOAD="${ARB_ROOT_PAYLOAD:-/opt/arb}"
ASSETS=/opt/task-assets/payload
[ -d "$ASSETS" ] || { echo "[verify] $ASSETS missing: the image was built without its task assets" >&2; exit 1; }
[ -d "$STAGE/setup_payload" ] || { echo "[verify] setup_payload/ missing in the stage folder" >&2; exit 1; }
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD"
cp -r "$STAGE/setup_payload/." "$ROOT_PAYLOAD/"
cp -r "$ASSETS/." "$ROOT_PAYLOAD/"
chown -R root:root "$ROOT_PAYLOAD"
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
export MUJOCO_GL="${MUJOCO_GL:-osmesa}"
exec "${ALE_VERIFY_PYTHON:-python3}" verify.py
