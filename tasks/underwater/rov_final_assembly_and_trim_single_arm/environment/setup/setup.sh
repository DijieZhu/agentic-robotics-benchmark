#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the packing list, the units'
# STL files, the drawings, the trim sheet, the water density chart and the propeller data sheet, the picture manual,
# the station scope and the work-order example, the cell documents, the robot models and the cell library) and the
# cameras' scenery into the agent home, owned by the agent. The scoring stage never reads these copies: it runs in a
# separate sandbox (task.toml) and builds its own root-owned read-only copy of the same givens from the image's task
# assets and a byte copy of payload/ here (change both together). Dynamic-only: packages live in
# environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
ASSETS=/opt/task-assets/payload
# the large givens and the cameras' scenery (render/) come from the image (environment/Dockerfile, task assets)
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
