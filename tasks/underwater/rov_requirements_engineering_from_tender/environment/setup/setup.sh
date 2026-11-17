#!/usr/bin/env bash
# Setup: runs as trusted root, cwd = this staged folder (the root-only setup stage), before the agent
# phase. (1) Stage the agent-visible payload into the agent home, and the image's task-asset payload
# when the image carries one, then check that every page image is there, byte for byte
# (page_images.sha256). (2) Close the hidden desk tables to everybody but root. (3) Start the
# enquiry and witness desk: the lab server reads ref/desk_table.json as root, drops to the dedicated
# `alelab` account and serves /run/ale-desk/lab.sock (fixed argv, clean environment, no hidden value
# outside this folder).
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
# Large page images come with the image (fetched at build time) once they are published; until then
# they sit in payload/ itself.
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/"
  for entry in /opt/task-assets/payload/* /opt/task-assets/payload/.[!.]*; do
    [ -e "$entry" ] || continue
    chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
  done
fi
# Every page image must now be in the agent home, byte for byte: page_images.sha256 names each one with its
# sha256 and its path in the agent home. A missing or different image fails setup, so a build that has neither the
# images in payload/ nor the fetched task assets never starts an episode (a missing or empty list fails too).
IMAGES="$PWD/page_images.sha256"
if [ ! -s "$IMAGES" ] || ! (cd "$HOME_DIR" && sha256sum --check --strict --quiet "$IMAGES"); then
  echo "setup: a page image is missing from the agent home or differs from page_images.sha256" >&2
  exit 1
fi
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"

# A second gate on the hidden desk table besides the root-only stage folder: root only, all the way
# down. The lab worker reads the table as root before it drops its privileges, also after a restart.
chmod -R go-rwx "$PWD/ref"

umask 077
# Returns once the lab serves (the daemon is detached, stdio on /dev/null) or fails setup.
env -i PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 HOME=/nonexistent \
  python3 -I -B "$PWD/lab/alelab/server.py" --config "$PWD/lab/desk.json" </dev/null >/dev/null 2>&1
