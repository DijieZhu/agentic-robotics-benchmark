#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens into the agent home, owned by
# the agent: payload/ (the league's documents, the car's datasheet, the interface: its manual, the client library and
# the protocol stub) and the image's task assets (the lab logs, the venue photographs and the car's meshes),
# which merge into ~/given/. The scoring stage never reads these copies: it runs in a separate sandbox (task.toml) with
# only the submission restored into it, and its programs get nothing of the givens. Dynamic-only: the Python environment
# and the task assets live in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
ASSETS=/opt/task-assets/payload
[ -d "$ASSETS" ] || { echo "[setup] $ASSETS missing: the image was built without its task assets" >&2; exit 1; }
[ -d payload ] || { echo "[setup] payload/ missing next to setup.sh" >&2; exit 1; }
cp -r payload/. "$HOME_DIR/"
cp -r "$ASSETS/." "$HOME_DIR/"
for entry in payload/* payload/.[!.]* "$ASSETS"/*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
