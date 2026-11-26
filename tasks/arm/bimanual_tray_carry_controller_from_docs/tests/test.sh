#!/usr/bin/env bash
# Verify stage: root, cwd = this staged folder, after the agent phase, in a SEPARATE verifier sandbox (task.toml
# [verifier] environment_mode = "separate"): a fresh sandbox from the task image that the agent never touched, holding
# only the declared artifact (/home/user/submission, restored root-owned); environment/setup/setup.sh never runs here.
# The engine uploads this folder world-readable, so the FIRST thing done here, before the interpreter starts, is to
# close the folder and its parent to every other account (the grader records the original modes from the two variables
# and restores them at the end, stops the agent account's leftover processes, snapshots the submission, gates late
# writes and locks every file of this folder).
#
# The grader (code/tasks/tray_carry/grader/verify.py) is started from this folder's own code/, for every submission,
# and runs every submitted controller as an unprivileged account through the runtime. This script reads no token and
# copies no code: the grader, the runtime, the duty generator, the draws, the rules and hidden/ stay in this folder,
# which no other account can read. The privileged reference (solution/) runs the unit's own simulator in its own
# process; for a submission whose token is valid the grader itself makes, as root, a root-owned, world-readable copy of
# the instrument alone (code/tasks/tray_carry/ref/*.py, the contract, the shared unit's package and its base files;
# ARB_ORACLE_INSTRUMENT_ROOT, default /opt/arb-oracle-instrument), checks it file by file, and names it in the truth
# the reference receives. Run as another account than root, the grader makes no copy and names its own ref/.
set -euo pipefail
STAGE="$(pwd -P)"
export ARB_STAGE_MODE="$(stat -c %a "$STAGE")"
chmod 700 "$STAGE"
PARENT="$(dirname "$STAGE")"
if [ "$PARENT" != "/" ] && [ "$PARENT" != "${ALE_HOME:-/home/user}" ]; then
  export ARB_STAGE_PARENT_MODE="$(stat -c %a "$PARENT")"
  chmod 700 "$PARENT"
fi
PY="${ALE_VERIFY_PYTHON:-python3}"
# The grader's root-owned, read-only copy of the givens: the grader checks it file by file against given_hashes.json.
ROOT_PAYLOAD="${ARB_ROOT_PAYLOAD:-/opt/arb}"
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD"
cp -r "$STAGE/setup_payload/." "$ROOT_PAYLOAD/"
if [ "$(id -u)" = "0" ]; then chown -R root:root "$ROOT_PAYLOAD"; fi
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
# The givens' copy itself is closed to every other account: the controller's runner account cannot read the documents
# at verify (a controller carries what it needs in its own folder; the documents are not on its computer), while the
# grader checks and reads the copy as root; the folder above stays open for the run folders the grader makes in it.
chmod 0700 "$ROOT_PAYLOAD/given"
# The numerics of the grader's process: one thread per library, the linear-algebra library's kernel and cache size,
# Python's hash seed; nothing is rendered. The runtime starts each controller's process with an environment of its own:
# of the variables below it carries the thread counts, the kernel and cache size, MPLBACKEND and MUJOCO_GL with the same
# values, and none of the others.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg SOURCE_DATE_EPOCH=0 MUJOCO_GL=disable
export ALE_STAGE_DIR="$STAGE"
exec "$PY" -B "$STAGE/code/tasks/tray_carry/grader/verify.py"
