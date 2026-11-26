#!/usr/bin/env bash
# Setup (root): stage the agent's kit and the recordings rendered into the image (agent-visible data only).
set -euo pipefail
umask 077
export HOME=/root PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONSTARTUP
cd "$(dirname "$0")"
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
# fixed order (sorted), fixed mtime, agent-owned, world-readable
(cd payload && find . -type d | LC_ALL=C sort | while read -r d; do mkdir -p "$HOME_DIR/$d"; done)
(cd payload && find . -type f | LC_ALL=C sort | while read -r f; do cp "$f" "$HOME_DIR/$f"; done)
# the recordings rendered into the image at build time (environment/gen/) join data/ before it is normalised
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/"
  for entry in /opt/task-assets/payload/*; do chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"; done
fi
for entry in payload/*; do
  [ -e "$entry" ] || continue
  tgt="$HOME_DIR/$(basename "$entry")"
  chown -R "$AGENT_USER:$AGENT_USER" "$tgt"
  find "$tgt" -type d -exec chmod 0755 {} + ; find "$tgt" -type f -exec chmod 0644 {} +
  find "$tgt" -exec touch -h -d '2026-01-01 00:00:00 UTC' {} +
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
# The session's time stamp (session_time.sh in the agent's home reads it): written now, a few minutes before the
# agent's session starts, in a folder that is root's, so that the agent's account can read it and cannot change it.
install -d -o root -g root -m 755 /run/session
date -u +%s > /run/session/start
chown root:root /run/session/start && chmod 0444 /run/session/start
chmod 0755 "$HOME_DIR/session_time.sh"
