#!/usr/bin/env bash
# Setup — runs as trusted root, cwd = this staged folder, before the agent phase.
#
# One job here: put the agent-visible material into the agent home as the agent's own
# writable copy: the documents in payload/given/ (the plant's datasheet model.md, the sensors'
# camera.md, the target's target.md, the sample frames' README) and the sample frames the
# image holds in /opt/task-assets/payload/given/frames/ (environment/Dockerfile). The agent
# gets no simulator, no renderer of the grader's scene and nothing that grades a controller.
# Keep this stage dynamic-only: package installation and fixed downloads belong in
# environment/Dockerfile, and lint rejects them here.
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

# The image's task assets (the sample frames), next to payload/ in the agent home.
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/"
  for entry in /opt/task-assets/payload/*; do
    [ -e "$entry" ] || continue
    chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
  done
fi
