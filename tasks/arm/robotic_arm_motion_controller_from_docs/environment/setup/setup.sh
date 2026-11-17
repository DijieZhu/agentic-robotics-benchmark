#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the vendor's documents of
# the arm and the servo, the unit-record format, the cells' documentation, the acceptance specification, the
# controller interface and its self-test) into the agent home, owned by the agent. The grader never reads these
# copies: verification is separate (task.toml), and tests/test.sh builds the grader's own root-owned copy from
# tests/setup_payload/, a byte copy of payload/ here (change both together, then tests/given_hashes.json).
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/"
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission/controller" && chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
