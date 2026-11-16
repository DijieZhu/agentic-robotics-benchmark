#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. One job, the givens: payload/ into the agent
# home, owned by the agent, plus what the image holds: the robot model (/opt/arb-assets/unitree_g1 ->
# ~/given/robot/unitree_g1) and the survey photos (/opt/arb-assets/survey -> ~/given/cell/survey). There is no practice
# cell: nothing here runs the house cell.
# The grader never reads anything set up here: verification is separate (task.toml), and tests/test.sh builds the
# grader's own ROOT-OWNED read-only copy of the givens (/opt/arb) in the fresh verifier sandbox, from
# tests/setup_payload/ (a byte copy of payload/ here: change both together, then tests/given_hashes.json) and the
# image's robot model and survey photos.
# Dynamic-only: packages, the robot model and the survey photos live in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
G1_DIR="${ARB_G1_DIR:-/opt/arb-assets/unitree_g1}"
SURVEY_DIR="${ARB_SURVEY_DIR:-/opt/arb-assets/survey}"

# cp -r, NOT -a: with -a the `payload/.` form also copies the payload directory's own root ownership onto the agent
# home, locking the agent out of its own workspace. Every top-level entry that came from payload/ is chowned
# (recursively), nothing else: the home itself keeps the ownership the image gave it.
cp -r payload/. "$HOME_DIR/"
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
[ -d "$G1_DIR" ] || { echo "[setup] $G1_DIR missing: the image was built without the robot model" >&2; exit 1; }
[ -d "$SURVEY_DIR" ] || { echo "[setup] $SURVEY_DIR missing: the image was built without the survey photos" >&2; exit 1; }
mkdir -p "$HOME_DIR/given/robot"
rm -rf "$HOME_DIR/given/robot/unitree_g1" "$HOME_DIR/given/cell/survey"
cp -r "$G1_DIR" "$HOME_DIR/given/robot/unitree_g1"
cp -r "$SURVEY_DIR" "$HOME_DIR/given/cell/survey"
chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/given"
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER): $(find "$HOME_DIR/given/cell/survey" -type f | wc -l) survey files" >&2
