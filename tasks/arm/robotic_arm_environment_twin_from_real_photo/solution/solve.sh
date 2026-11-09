#!/usr/bin/env bash
# Oracle: the true twin of the photographed scene as a submission (scene.xml with photo_cam at the true intrinsics and
# pose, the collision geometry of every item and of the console and its drawer, objects.json), run in place of the
# agent by `ale validate` as the same unprivileged account. Collision geometry only: the grader ignores appearance.
# It is the twin the grader's truth is built from, so it scores 1.0 on every criterion by construction.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
mkdir -p "$SUB"
cp -r "$(dirname "$0")/answer/." "$SUB/"
echo "[oracle] submission ready at $SUB" >&2
