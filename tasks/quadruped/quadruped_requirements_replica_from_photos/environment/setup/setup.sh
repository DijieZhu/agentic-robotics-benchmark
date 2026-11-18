#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (brief, build log, quote,
# photos, datasheet and erratum pages, the three lot reports, schemas, the structure check and the public, data-free
# grading modules) into the agent home, owned by the agent. Nothing hidden is in this folder: the reference unit's
# ground truth and the vendors' own numbers live in the grader's stage (absent during the agent phase). Dynamic-only:
# packages are in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/"
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
# the large givens (photos, datasheet and erratum pages, lot reports) come from the image (task assets)
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/"
  for entry in /opt/task-assets/payload/*; do chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"; done
fi
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
chmod +x "$HOME_DIR/schema_check.py" 2>/dev/null || true
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
