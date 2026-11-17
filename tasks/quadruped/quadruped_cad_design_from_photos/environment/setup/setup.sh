#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the photo set of the robot and
# the bought components' documents and STEP files), fetched into the image from the benchmark's asset repository
# (environment/assets.sha256, docs/task-assets.md), into the agent home, owned by the agent. Nothing hidden is among
# them: the photographed robot's ground truth lives in tests/ (absent during the agent phase).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
ASSETS=/opt/task-assets/payload
[ -d "$ASSETS" ] || { echo "[setup] $ASSETS missing: the image was built without its assets" >&2; exit 1; }
# cp -r, NOT -a: `$ASSETS/.` with -a would copy the folder's root ownership onto the agent home
cp -r "$ASSETS/." "$HOME_DIR/"
for entry in "$ASSETS"/*; do
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" "$HOME_DIR/work" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission" "$HOME_DIR/work"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
