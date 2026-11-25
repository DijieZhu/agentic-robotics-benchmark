#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the robot's handbook, the
# station's API, the site briefing, the camera datasheet, the calibration site's pages, the client library, the example
# program and the practice launcher) into the agent home, owned by the agent, and starts the practice ground (below).
# The grader never reads these copies: verification is separate (task.toml), and tests/test.sh builds the grader's own
# ROOT-OWNED read-only copy of the same givens in the fresh verifier sandbox, from tests/setup_payload/ (a byte copy of
# payload/ here: change both together, then tests/given_hashes.json). Precedent:
# quadruped_hazardous_plant_inspection/environment/setup/setup.sh (this task has no render bundle).
# Dynamic-only: packages and the practice ground's program and site packages live in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/"
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2

# The practice ground: the image installs it under /opt/voidcell-practice for the account `voidcell` only (the agent
# account can neither read nor list it); setup starts it now, to run through the agent's session, under a loop that
# restarts it should it ever exit (its episode allowance survives a restart). The agent reaches it through
# /run/voidcell/practice.sock (voidcell.client, ~/training/run_mission.py): the station's public calls, reset(site) and
# each episode's debrief. Its numerics and renderer settings are pinned as the grader pins them (tests/test.sh): the
# physics single-threaded, 8 rasteriser threads, BLAS 2, the same OpenBLAS kernels and numpy SIMD paths on every CPU;
# Mesa never writes its shader cache; temporary files stay in the service's own folder.
SERVICE=/opt/voidcell-practice
[ -f "$SERVICE/practiced.py" ] || { echo "[setup] the image lacks the practice ground ($SERVICE)" >&2; exit 1; }
install -d -o voidcell -g voidcell -m 755 /run/voidcell
rm -f /run/voidcell/practice.sock
setsid runuser -u voidcell -- env HOME="$SERVICE" TMPDIR="$SERVICE/state" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    MESA_SHADER_CACHE_DISABLE=true LP_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \
    /bin/bash -c 'exec >> /opt/voidcell-practice/state/practiced.log 2>&1; while :; do /opt/venv/bin/python3 -B -s /opt/voidcell-practice/practiced.py; sleep 2; done' \
    < /dev/null > /dev/null 2>&1 &
for _ in $(seq 1 120); do [ -S /run/voidcell/practice.sock ] && break; sleep 0.5; done
[ -S /run/voidcell/practice.sock ] || { echo "[setup] the practice ground did not come up" >&2; exit 1; }
echo "[setup] practice ground listening on /run/voidcell/practice.sock" >&2
