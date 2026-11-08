#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the cell documents, the cell
# library, the practice scripts and the example program) into the agent home, owned by the agent. The grader never reads
# these copies: verification is separate (task.toml), and tests/test.sh builds the grader's own ROOT-OWNED read-only
# copy of the same givens in the fresh verifier sandbox, from tests/setup_payload/ (a byte copy of payload/ here: change
# both together, then tests/given_hashes.json) and the image's robot model (/opt/task-assets/panda).
# Dynamic-only: packages and the robot model live in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/"
if [ -d /opt/task-assets/panda ]; then
  mkdir -p "$HOME_DIR/given"
  cp -r /opt/task-assets/panda "$HOME_DIR/given/panda"
fi
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
