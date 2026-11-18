#!/usr/bin/env bash
# The verify stage's entry point: root, cwd = this staged folder (/opt/ale/tests), after the agent phase, in a SEPARATE
# verifier sandbox (task.toml [verifier] environment_mode = "separate"): a fresh sandbox from the task image that the
# agent never touched, holding only the declared artifact (/home/user/submission, restored root-owned);
# environment/setup/setup.sh never runs here.
#
# `$ALE_VERIFY_PYTHON`: the engine stages its `ale_verify` package, through which the kit's write_verdict writes
# /logs/verifier/reward.json, into the first 3.12+ interpreter it finds (`python3`, then `python3.12`) and names it
# here. In this image that `python3` is /opt/venv/bin/python3 (the image's PATH, and /usr/local/bin/python3 wraps it),
# the interpreter this stage named explicitly before.
set -euo pipefail
chmod 700 .
# The grader's root-owned, read-only copy of the givens (/opt/arb): verify.py checks it file by file against
# given_hashes.json, builds the cell's physics (the kit, the xArm 7) and the cameras' scenery (render/) from it and
# starts the submitted program through its cell_runner.py, so nothing the agent wrote is an input of the grader. It
# holds what environment/setup/setup.sh stages into the agent home, in the same order: the image's task assets, then
# setup_payload/, this folder's byte copy of environment/setup/payload/ (which never reaches the verifier; change both
# together, then given_hashes.json).
ROOT_PAYLOAD=/opt/arb
ASSETS=/opt/task-assets/payload
[ -d "$ASSETS" ] || { echo "[verify] $ASSETS missing: the image was built without its task assets" >&2; exit 1; }
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD"
cp -r "$ASSETS/." "$ROOT_PAYLOAD/"
cp -r setup_payload/. "$ROOT_PAYLOAD/"
chown -R root:root "$ROOT_PAYLOAD"
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
export ARB_ASSETS="$ROOT_PAYLOAD"
unset DISPLAY WAYLAND_DISPLAY
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=8
exec "${ALE_VERIFY_PYTHON:-python3}" -u verify.py
