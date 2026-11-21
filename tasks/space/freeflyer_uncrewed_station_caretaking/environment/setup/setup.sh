#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the robot's manual and API,
# the station's handbook, telemetry dictionary, lost-item register and rehearsal guide, the station map, the telemetry
# sample, the client library, the example program and the rehearsal tool) into the agent home, owned by the agent, and
# starts the rehearsal facility (below). The pictures of the givens (the figures, the catalogue's photographs and the
# map views) are task assets: they come from the image (/opt/task-assets/payload, environment/Dockerfile). The grader
# never reads these copies: verification is separate (task.toml), and tests/test.sh builds the grader's own ROOT-OWNED
# read-only copy of the same givens in the fresh verifier sandbox, from the image's task assets and
# tests/setup_payload/ (a byte copy of payload/ here: change both together, then tests/given_hashes.json).
# Dynamic-only: packages, the rehearsal facility's program and the task assets live in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
ASSETS=/opt/task-assets/payload
[ -d "$ASSETS/given" ] || { echo "[setup] $ASSETS/given missing: the image was built without its task assets" >&2; exit 1; }
cp -r payload/. "$HOME_DIR/"
cp -r "$ASSETS/." "$HOME_DIR/"
for entry in payload/* payload/.[!.]* "$ASSETS"/*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
chmod -R u+rwX,go+rX,go-w "$HOME_DIR/given"     # the image's copy is root-only (750 / 640): the agent's copy is not
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2

# The rehearsal facility: the image installs it under /opt/rehearsal for the account
# `rehearsal` only (mode 700: the agent account can neither read nor list it); setup starts it now, to run through
# the agent's session, under a loop that restarts it should it ever exit (its allowance survives a restart). The agent
# reaches it through /run/rehearsal/rehearsal.sock (~/rehearsal/rehearse.py): the robot's public calls, reset(card)
# and each run's observation sheet; /run/rehearsal/status.json and /run/rehearsal/debriefs/ are world-readable. The
# cell's asset root is the image's one copy (/opt/task-assets, the root tests/test.sh names for the grader), checked
# here first against the pins of the facility's copy of the cell (a broken image fails the setup, never a rehearsal
# run without its model files; the cell checks the same again at every build); its numerics are pinned as the grader
# pins them (runtime key cell.recipe.env, tests/test.sh): one BLAS / OpenMP thread, the CPU paths of numpy and OpenBLAS
# (the forced kernel always with OPENBLAS_L2_SIZE=256, as tests/test.sh sets it), 4 rasteriser threads (two runs may be
# open at a time on the sandbox's 8 CPUs). These variables are set here only, not in the image's ENV: the agent's shell
# does not show them.
SERVICE=/opt/rehearsal
ASSET_ROOT=/opt/task-assets
[ -f "$SERVICE/practiced.py" ] || { echo "[setup] the image lacks the rehearsal facility ($SERVICE)" >&2; exit 1; }
runuser -u rehearsal -- env CABINCELL_ASSET_ROOT="$ASSET_ROOT" /opt/venv/bin/python3 -I -B "$SERVICE/cabincell/assetroot.py" \
    robot station robot_author > /dev/null \
  || { echo "[setup] the task assets at $ASSET_ROOT are missing, altered or not the pinned ones" >&2; exit 1; }
install -d -o rehearsal -g rehearsal -m 755 /run/rehearsal
rm -f /run/rehearsal/rehearsal.sock
setsid runuser -u rehearsal -- env HOME="$SERVICE" CABINCELL_ASSET_ROOT="$ASSET_ROOT" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4 \
    OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512_ICL \
    /bin/bash -c 'exec >> /opt/rehearsal/state/practiced.out 2>&1; while :; do /opt/venv/bin/python3 -B -s /opt/rehearsal/practiced.py; sleep 2; done' \
    < /dev/null > /dev/null 2>&1 &
for _ in $(seq 1 120); do [ -S /run/rehearsal/rehearsal.sock ] && break; sleep 0.5; done
[ -S /run/rehearsal/rehearsal.sock ] || { echo "[setup] the rehearsal facility did not come up" >&2; exit 1; }
echo "[setup] rehearsal facility listening on /run/rehearsal/rehearsal.sock" >&2
