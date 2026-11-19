#!/usr/bin/env bash
# Setup — runs as trusted root, cwd = this staged folder (/opt/ale/setup, root-only), before the agent phase:
# stage the agent-visible payload (the simulator, the practice units, the practice grader, the wire skeleton) into the
# agent home. Nothing hidden is in this folder.
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
