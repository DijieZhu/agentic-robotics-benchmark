#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the vehicle's manual and API,
# the field documents of the manifold, the training facilities' sheets, the client library, the example program and the
# practice launcher) into the agent home, owned by the agent, and starts the practice facilities (below). The grader
# never reads these copies: verification is separate (task.toml), and the grading stage builds its own ROOT-OWNED
# read-only copy of the same givens in the fresh verifier sandbox from its byte copy of payload/ (both written by the
# development tree's assembler).
# Dynamic-only: packages and the practice facilities' program and packages live in the image (environment/Dockerfile).
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

# The practice facilities: the image installs them under /opt/subseacell-practice for the account `subseacell` only (the
# agent account can neither read nor list them); setup starts the service now, to run through the agent's session,
# under a loop that restarts it should it ever exit (the run allowance survives a restart). The agent reaches it
# through /run/subseacell/practice.sock (subseacell.client.connect_practice, ~/given/training/run_session.py): the
# vehicle's public calls, reset(site, setup) and, in the training tank, instruments(). Its numerics and renderer
# settings are pinned as the grader pins them; since amendment 04 the kernel pin comes with OPENBLAS_L2_SIZE=256,
# without which the bundled linear-algebra library enlarges the pinned kernel's blocks on some hosts, where that
# kernel then overruns its stack buffer.
# The service's environment and its restart loop are written into a launcher inside the practice folder (owner
# subseacell, mode 500, in the mode-700 folder: root and the service account read it, the agent account can neither
# read nor list it). The service starts from the launcher's path alone: no command line in the process list carries an
# environment setting, and the service's environment is readable by its own account and root only.
SERVICE=/opt/subseacell-practice
LAUNCHER="$SERVICE/run_service.sh"
[ -f "$SERVICE/practiced.py" ] || { echo "[setup] the image lacks the practice facilities ($SERVICE)" >&2; exit 1; }
install -d -o subseacell -g subseacell -m 755 /run/subseacell
rm -f /run/subseacell/practice.sock "$LAUNCHER"
(umask 077 && cat > "$LAUNCHER") <<'LAUNCH'
#!/bin/bash
# The practice service (written by setup.sh; run by the account subseacell): its pinned environment, then a loop that
# restarts it should it ever exit.
unset DISPLAY
export HOME=/opt/subseacell-practice MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=1 \
    OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 \
    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" PYTHONDONTWRITEBYTECODE=1 \
    SUBSEACELL_PRACTICE_KIT=/opt/subseacell-practice/kit SUBSEACELL_PRACTICE_CELL=/opt/subseacell-practice/cell \
    SUBSEACELL_TASK_VALUES=/opt/subseacell-practice/task_values.json UWKIT_MESHLIB=/opt/subseacell-practice/meshes \
    SUBSEACELL_VEHICLE_MESHES=/opt/subseacell-practice/meshes SUBSEACELL_VISUAL_MESHES=/opt/subseacell-practice/meshes \
    SUBSEACELL_PRACTICE_SOCKET=/run/subseacell/practice.sock \
    SUBSEACELL_PRACTICE_STATUS=/run/subseacell/practice_status.json \
    SUBSEACELL_PRACTICE_RECORDS=/run/subseacell/tank_records
exec >> /opt/subseacell-practice/state/practiced.log 2>&1
while :; do /opt/cellenv/bin/python3 -B -s /opt/subseacell-practice/practiced.py; sleep 2; done
LAUNCH
chown subseacell:subseacell "$LAUNCHER" && chmod 500 "$LAUNCHER"
setsid runuser -u subseacell -- "$LAUNCHER" < /dev/null > /dev/null 2>&1 &
for _ in $(seq 1 240); do [ -S /run/subseacell/practice.sock ] && break; sleep 0.5; done
[ -S /run/subseacell/practice.sock ] || { echo "[setup] the practice facilities did not come up" >&2; exit 1; }
echo "[setup] practice facilities listening on /run/subseacell/practice.sock" >&2
