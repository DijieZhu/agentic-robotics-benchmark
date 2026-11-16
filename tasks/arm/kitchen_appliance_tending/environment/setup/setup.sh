#!/usr/bin/env bash
# Setup — runs as trusted root, cwd = this staged folder (/opt/ale/setup, root-only), before the agent
# phase. (1) Stage the agent-visible payload into the agent home. (2) Start the lab service that holds
# the hidden units: it reads ref/units.json and ref/units/ as root, drops to the dedicated `alelab` account and
# serves /run/ale-lab/lab.sock (fixed argv, clean environment, no hidden value outside this folder).
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
# The practice units' models are task assets (environment/Dockerfile fetches them to /opt/task-assets/payload/units):
# copied to ~/units beside units.json. The room bundle beside them (render/) is read in place by the cameras' renderer.
mkdir -p "$HOME_DIR/units"
cp -r /opt/task-assets/payload/units/. "$HOME_DIR/units/"
chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/units"
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"

umask 077
# Returns once the lab serves (the daemon is detached, stdio on /dev/null) or fails setup.
env -i PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 HOME=/nonexistent \
  python3 -I -B "$PWD/lab/alelab/server.py" --config "$PWD/lab/lab.json" </dev/null >/dev/null 2>&1
