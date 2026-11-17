#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens into the agent home, owned by
# the agent: first the image's task assets (the kit, the xArm 7 model, the picture manual and the cameras' scenery,
# render/; environment/Dockerfile), then payload/ (the connection client and runner, the cell and provenance
# documents). The grader never reads these copies: verification is separate (task.toml), and tests/test.sh builds the
# grader's own root-owned, read-only copy of the same givens in the fresh verifier sandbox, from the image's task assets
# and tests/setup_payload/, a byte copy of payload/ here (change both together, then tests/given_hashes.json).
# Dynamic-only: packages live in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r /opt/task-assets/payload/. "$HOME_DIR/"
cp -r payload/. "$HOME_DIR/"
for entry in /opt/task-assets/payload/* payload/*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission"
chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
