#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the maker's published files
# of the pair, the joint modules' data pages, the integrator's sheet of observations, the delivery record's format with
# the record of another unit, the site's sheets, the acceptance specification, the controller's interface with its
# example and self-test) into the agent home, owned by the agent. The grader never reads these copies: verification is
# separate (task.toml), and tests/test.sh builds the grader's own root-owned copy from tests/setup_payload/, a byte copy
# of payload/ here (both are written by the task folder's generator, with tests/given_hashes.json).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/"
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
chmod -R a-w "$HOME_DIR/given"
mkdir -p "$HOME_DIR/submission/controller" && chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
