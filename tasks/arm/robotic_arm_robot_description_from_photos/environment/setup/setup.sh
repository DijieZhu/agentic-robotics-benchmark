#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the photo session of the arm
# with its bus log, the published parts, the servo documents, the EEPROM dump, the masses, the build notes, the contract
# and the contract gate) into the agent home, owned by the agent. Nothing hidden is in this folder: the photographed
# arm's ground truth lives in tests/ (absent during the agent phase). Dynamic-only: packages are in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
# cp -r, NOT -a: `payload/.` with -a would copy the payload directory's root ownership onto the agent home
cp -r payload/. "$HOME_DIR/"
# the large givens (photos, parts, datasheet images) come from the image (environment/Dockerfile, task assets)
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/"
fi
for entry in payload/* payload/.[!.]* /opt/task-assets/payload/*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
chmod +x "$HOME_DIR/check_submission.py" 2>/dev/null || true
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
