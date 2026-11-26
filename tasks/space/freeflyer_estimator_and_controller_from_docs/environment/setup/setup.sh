#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase.
#
# Stage the givens into the agent's home, owned by the agent: everything under this folder's payload/ (payload/ and
# controller/) and, when the image has them, the asset repository's files (/opt/task-assets/payload, the published
# form: the given records under payload/unit/records/). The home then holds payload/ and controller/ side by side,
# the two roots of every path the task's pages print, and the empty deliverable folder submission/controller/. No
# given file needs an executable bit (the self-test is run as python3 selftest.py). Nothing hidden is in this
# folder: the grader arrives only at verify, in a separate sandbox. No warm-up, no download, no build.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for root in "$SETUP_DIR/payload" /opt/task-assets/payload; do
  [ -d "$root" ] || continue
  cp -r "$root/." "$HOME_DIR/" || { echo "[setup] staging $root failed" >&2; exit 1; }
  for entry in "$root"/*; do
    [ -e "$entry" ] || continue
    chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
  done
done
missing=0
for d in payload controller payload/unit/records; do
  [ -d "$HOME_DIR/$d" ] || { echo "[setup] $d/ is missing from the givens" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || exit 1
mkdir -p "$HOME_DIR/submission/controller"
chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
exit 0
