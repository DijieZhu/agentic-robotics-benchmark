#!/usr/bin/env bash
# Setup — runs as trusted root, cwd = this staged folder (the root-only setup stage), before the agent
# phase. Stage the agent-visible payload into the agent home, and the image's task-asset payload when the
# image carries one, then check that every photograph and every page image is there, byte for byte
# (given_images.sha256). Nothing else runs: there is no service and no hidden file in this folder.
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
# The photographs and page images come with the image (fetched at build time) once they are published; until then
# they sit in payload/ itself.
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/"
  for entry in /opt/task-assets/payload/* /opt/task-assets/payload/.[!.]*; do
    [ -e "$entry" ] || continue
    chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
  done
fi
# Every photograph and page image must now be in the agent home, byte for byte: given_images.sha256 names each one
# with its sha256 and its path in the agent home. A missing or different image fails setup, so a build that has
# neither the images in payload/ nor the fetched task assets never starts an episode (a missing or empty list fails
# too).
IMAGES="$PWD/given_images.sha256"
if [ ! -s "$IMAGES" ] || ! (cd "$HOME_DIR" && sha256sum --check --strict --quiet "$IMAGES"); then
  echo "setup: an image is missing from the agent home or differs from given_images.sha256" >&2
  exit 1
fi
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
