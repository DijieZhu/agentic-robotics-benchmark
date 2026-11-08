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
# renders through OSMesa (the agent's cameras use the same path in the practice cell).
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
# given_hashes.json, compiles its physics (the cell library, the Panda) from it and hands it to the programs as
# ARB_GIVEN_DIR, so nothing the agent wrote is an input of the grader. It holds what environment/setup/setup.sh stages
# into the agent home: setup_payload/, this folder's byte copy of environment/setup/payload/ (which never reaches the
# verifier; change both together, then given_hashes.json), and the image's Panda (/opt/task-assets/panda).
ROOT_PAYLOAD="${ARB_ROOT_PAYLOAD:-/opt/arb}"
[ -d /opt/task-assets/panda ] || { echo "[verify] /opt/task-assets/panda missing: the image was built without its robot model" >&2; exit 1; }
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD/given"
cp -r "$STAGE/setup_payload/." "$ROOT_PAYLOAD/"
cp -r /opt/task-assets/panda "$ROOT_PAYLOAD/given/panda"
chown -R root:root "$ROOT_PAYLOAD"
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
export HOME=/root PYTHONNOUSERSITE=1 XDG_CONFIG_HOME=/root/.config XDG_CACHE_HOME=/root/.cache \
  MPLCONFIGDIR=/root/.config/matplotlib
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
exec "${ALE_VERIFY_PYTHON:-python3}" -s verify.py
