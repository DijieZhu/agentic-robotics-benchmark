#!/usr/bin/env bash
# Setup (root): stage the agent's kit (practice environment, practice failure states, practice grader) into its home.
set -euo pipefail
umask 077
export HOME=/root PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH PYTHONSTARTUP
cd "$(dirname "$0")"
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
(cd payload && find . -type d | LC_ALL=C sort | while read -r d; do mkdir -p "$HOME_DIR/$d"; done)
(cd payload && find . -type f | LC_ALL=C sort | while read -r f; do cp "$f" "$HOME_DIR/$f"; done)
for entry in payload/*; do
  [ -e "$entry" ] || continue
  tgt="$HOME_DIR/$(basename "$entry")"
  chown -R "$AGENT_USER:$AGENT_USER" "$tgt"
  find "$tgt" -type d -exec chmod 0755 {} + ; find "$tgt" -type f -exec chmod 0644 {} +
  find "$tgt" -exec touch -h -d '2026-01-01 00:00:00 UTC' {} +
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
