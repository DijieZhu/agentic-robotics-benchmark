#!/usr/bin/env bash
# Setup — runs as trusted root, cwd = this staged folder (/opt/ale/setup, root-only), before the agent phase:
# stage the agent-visible payload (the simulator, the practice units, the practice grader, the wire skeleton, the
# session-time script) into the agent home and write the session's time stamp. Nothing hidden is in this folder.
set -euo pipefail
umask 022
export HOME=/root PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONSTARTUP

HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"

# cp -r, NOT -a: with -a the `payload/.` form also copies the payload directory's own root ownership
# onto the agent home, locking the agent out of its own workspace.
cp -r payload/. "$HOME_DIR/"
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"

# The session's time stamp (session_time.sh in the agent's home reads it): written now, moments before the agent's
# session starts, in a folder that is root's, so that the agent's account can read it and cannot change or replace it.
install -d -o root -g root -m 755 /run/session
date -u +%s > /run/session/start
chown root:root /run/session/start && chmod 0444 /run/session/start
