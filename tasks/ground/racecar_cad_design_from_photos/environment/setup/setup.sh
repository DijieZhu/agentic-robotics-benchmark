#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the photo set of the unit and
# the purchased components' documents and STEP files) into the agent home, owned by the agent: from the image's task
# assets (/opt/task-assets/payload, fetched from the benchmark's asset repository: environment/assets.sha256,
# docs/task-assets.md) and from a payload/ folder next to this script, whichever is present. Nothing hidden is among
# them: the unit's reference geometry lives in tests/ (absent during the agent phase).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
staged=0
for SRC in /opt/task-assets/payload "$(dirname "$0")/payload"; do
  [ -d "$SRC" ] || continue
  # cp -r, NOT -a: `$SRC/.` with -a would copy the folder's root ownership onto the agent home
  cp -r "$SRC/." "$HOME_DIR/"
  for entry in "$SRC"/*; do
    chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
  done
  staged=1
done
[ "$staged" = 1 ] || { echo "[setup] no givens found (neither /opt/task-assets/payload nor a payload/ next to setup.sh)" >&2; exit 1; }
mkdir -p "$HOME_DIR/submission" "$HOME_DIR/work" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission" "$HOME_DIR/work"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
