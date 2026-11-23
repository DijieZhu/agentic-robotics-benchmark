#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the kit, the manual, the
# cell documents, the Panda and the cell library) into the agent home, owned by the agent. The grader never reads these
# copies: verification is separate (task.toml), and tests/test.sh builds the grader's own ROOT-OWNED read-only copy of
# the same givens in the fresh verifier sandbox, from the image's task assets and tests/setup_payload/, a byte copy of
# payload/ here (change both together, then tests/given_hashes.json). Dynamic-only: packages live in
# environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/"
# the large givens (the manual, the parts' STL files, the Panda model) and the cameras' scenery (render/) come from the
# image (environment/Dockerfile, task assets)
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/"
fi
for entry in payload/* payload/.[!.]* /opt/task-assets/payload/*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2
