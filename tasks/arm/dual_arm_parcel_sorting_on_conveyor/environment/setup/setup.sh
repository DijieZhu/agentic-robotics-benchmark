#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the cell documents, the
# robot's and the station's model files and the robot's meshes from the image's task assets, the client library, the commissioning launcher and the example program) into
# the agent home, owned by the agent, and starts the commissioning service (below). The grader never reads these
# copies: verification is separate (task.toml), and tests/test.sh builds the grader's own ROOT-OWNED read-only copy of
# the same givens in the fresh verifier sandbox, from tests/setup_payload/ (a byte copy of payload/ here: change both
# together, then tests/given_hashes.json).
# Dynamic-only: packages and the commissioning service's program live in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
ASSETS=/opt/task-assets/payload
# the robot's mesh files and the gripper's licence come from the image (environment/Dockerfile, task assets: the
# benchmark's asset repository at the pinned commit); the render bundle is never staged into the home
[ -d "$ASSETS/given" ] || { echo "[setup] $ASSETS missing: the image was built without its task assets" >&2; exit 1; }
cp -r payload/. "$HOME_DIR/"
cp -r "$ASSETS/." "$HOME_DIR/"
for entry in payload/* payload/.[!.]* "$ASSETS"/*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2

# The commissioning service: the station's cell on its development shifts (never a graded one). The image installs it
# under /opt/parcelcell-commission for the account `parcelcell` only (the agent account can neither read nor list it);
# setup starts it now, to run through the agent's session, under a loop that restarts it should it ever exit (its
# shift count survives a restart). The agent reaches it through the socket /run/parcelcell/commission.sock
# (parcelcell.client.connect_commissioning, ~/commission/run_shift.py): the cell's public calls and reset(site), with
# the graded shift's budgets and a fixed number of shifts; no ground truth, no score, no debrief. Its numerics and
# renderer settings are pinned as tests/test.sh pins the grader's (8 rasteriser threads, BLAS and OpenMP 2, the
# OpenBLAS kernel with its block size, the numpy CPU-feature mask; tests/test.sh says why); its development shifts'
# parcel looks were made at the image's build (PARCELCELL_KIT_CACHE), so a shift starts in seconds.
SERVICE=/opt/parcelcell-commission
[ -f "$SERVICE/commissiond.py" ] || { echo "[setup] the image lacks the commissioning service ($SERVICE)" >&2; exit 1; }
install -d -o parcelcell -g parcelcell -m 755 /run/parcelcell
rm -f /run/parcelcell/commission.sock
setsid runuser -u parcelcell -- env HOME="$SERVICE" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 LP_NUM_THREADS=8 \
    OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \
    PARCELCELL_KIT_CACHE=/opt/parcelcell-commission/kitcache \
    /bin/bash -c 'cd /opt/parcelcell-commission/state; exec >> /opt/parcelcell-commission/state/commissiond.out 2>&1; while :; do /opt/venv/bin/python3 -B -s /opt/parcelcell-commission/commissiond.py; sleep 2; done' \
    < /dev/null > /dev/null 2>&1 &
for _ in $(seq 1 120); do [ -S /run/parcelcell/commission.sock ] && break; sleep 0.5; done
[ -S /run/parcelcell/commission.sock ] || { echo "[setup] the commissioning service did not come up" >&2; exit 1; }
echo "[setup] commissioning service listening on /run/parcelcell/commission.sock" >&2
