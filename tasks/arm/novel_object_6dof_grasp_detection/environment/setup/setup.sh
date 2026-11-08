#!/usr/bin/env bash
# Setup — runs as trusted root, cwd = this staged folder, before the agent phase.
#
# One job here: put the agent-visible material (the practice grader, the agent's own
# editable copy of the environment) into the agent home, owned by the agent. Keep this
# stage dynamic-only: package installation and fixed downloads belong in image/Dockerfile,
# and lint rejects them here.
#
# Everything under payload/ is staged, whatever it is called — add, rename or remove
# payload files without touching this script.
set -euo pipefail

HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"

# cp -r, NOT -a: with -a the `payload/.` form also copies the payload directory's own
# root ownership onto the agent home, locking the agent out of its own workspace.
cp -r payload/. "$HOME_DIR/"

# chown every top-level entry that came from payload/ (recursively) and nothing else: the
# home itself keeps the ownership the image gave it.
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
