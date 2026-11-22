#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the robot's documents, the
# client library, the example program, the training-ground launcher with the grader's submission checks, the
# session-time script) into the agent home, owned by the agent, writes the session's time stamp and starts the
# training ground (below). The grader never reads these copies: verification is separate (task.toml), and
# tests/test.sh builds the grader's own ROOT-OWNED read-only copy of the same givens in the fresh verifier sandbox, from
# tests/setup_payload/ (a byte copy of payload/ here: change both together, then tests/given_hashes.json).
# Dynamic-only: packages and the training ground's program live in environment/Dockerfile.
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

# The session's time stamp (session_time.sh in the agent's home reads it): written now, moments before the agent's
# session starts, in a folder that is root's, so that the agent's account can read it and cannot change or replace it.
/usr/bin/install -d -o root -g root -m 755 /run/session
/bin/date -u +%s > /run/session/start
/bin/chown root:root /run/session/start && /bin/chmod 0444 /run/session/start

# The training ground: the image installs it under /opt/rubblecell-training for the account `rubblecell` only (the
# agent account can neither read nor list it); setup starts it now, to run through the agent's session, under a loop
# that restarts it should it ever exit. The agent reaches it through /run/rubblecell/training.sock
# (rubblecell.client.connect_training, ~/training/run_mission.py): the robot's public calls, reset(site) and each
# mission's log (the robot's own: no truth of the site, no score; also kept, world-readable, in /run/rubblecell/logs/).
# Every program in the restart loop is named by its absolute path: /usr/local/bin is the agent account's in the base
# image and comes first on PATH.
SERVICE=/opt/rubblecell-training
[ -f "$SERVICE/trainingd.py" ] || { echo "[setup] the image lacks the training ground ($SERVICE)" >&2; exit 1; }
/usr/bin/install -d -o rubblecell -g rubblecell -m 755 /run/rubblecell
/bin/rm -f /run/rubblecell/training.sock
/usr/bin/setsid /usr/sbin/runuser -u rubblecell -- /usr/bin/env -i PATH=/usr/bin:/bin HOME="$SERVICE" MUJOCO_GL=osmesa \
    PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LP_NUM_THREADS=2 OPENBLAS_CORETYPE=Haswell \
    OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" RUBBLECELL_LOOK_DIR=/opt/rubblecell-look \
    /bin/bash -c 'while :; do /opt/venv/bin/python3 -s /opt/rubblecell-training/trainingd.py; /bin/sleep 2; done' \
    < /dev/null > /dev/null 2>&1 &
for _ in $(/usr/bin/seq 1 60); do [ -S /run/rubblecell/training.sock ] && break; /bin/sleep 0.5; done
[ -S /run/rubblecell/training.sock ] || { echo "[setup] the training ground did not come up" >&2; exit 1; }
echo "[setup] training ground listening on /run/rubblecell/training.sock" >&2
