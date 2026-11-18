#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens into the agent home, owned by
# the agent: from payload/ the photo index, the controller firmware and the host software (with its tracking config and
# two calibration files), the board's fabrication data, the two sensor cut-outline drawings, the vendor pages, the
# conventions, the three submission schemas and the public structure check; from the image's task assets
# (/opt/task-assets/payload, environment/Dockerfile) the photo set photos/photo_001.jpg .. photo_320.jpg; and an empty
# submission/ folder. Nothing hidden is in this folder: the robot's truth ledger, the price table and the grader live
# in tests/ (absent during the agent phase). Dynamic-only: packages are in environment/Dockerfile. Same steps as the
# bimanual requirements task's setup.sh.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/"
# the photo set comes from the image (environment/Dockerfile, task assets)
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/"
fi
for entry in payload/* payload/.[!.]* /opt/task-assets/payload/*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
chmod +x "$HOME_DIR/schema_check.py" 2>/dev/null || true
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
