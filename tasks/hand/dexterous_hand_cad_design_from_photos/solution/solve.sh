#!/usr/bin/env bash
# Oracle: build the reference document natively in FreeCAD at oracle time (it is large: every given part's STEP is
# imported unchanged), from solution/reference (oracle.json, format hand-oracle/1: the tags with their kinds' solids and
# poses, the given placements, the links with their anchors, the hinges with their frames and limits; parts/<kind>.brep:
# the made kinds' solids, byte-identical to the grader's tests/reference/parts) and the payload that setup.sh staged in
# the agent home (purchased/<PN>/<PN>.step; oracle.json's step paths are relative to the home). build_oracle.py checks
# that nothing moves on recompute, solve, save and reopen, and writes its report next to the document
# (hand.FCStd.report.json); the document is moved into the submission only when that report says it built cleanly.
# forked from tasks/ground/racecar_cad_design_from_photos/solution/solve.sh (benchmark main 850dee3, sha256
# 6aec5f54e45ca56b126e504fa53e10d84fa2852d9dfe146b143b3034d8d8c333): the file name hand.FCStd.
set -euo pipefail
TASK_HOME="${ALE_HOME:-/home/user}"
HERE="$(cd "$(dirname "$0")" && pwd -P)"
FREECADCMD="${ARB_FREECADCMD:-/opt/freecad/usr/bin/freecadcmd}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
install -d -m 0755 "$TASK_HOME/submission"
# a private HOME: FreeCAD writes its settings there, away from the desktop session's own
HOME="$WORK/home" QT_QPA_PLATFORM=offscreen "$FREECADCMD" -c "import sys; sys.argv = ['build_oracle.py', '$HERE/reference', '$TASK_HOME', '$WORK/hand.FCStd']; exec(open('$HERE/build_oracle.py').read())" > "$WORK/build.log" 2>&1 || true
REPORT="$WORK/hand.FCStd.report.json"
if [ -s "$REPORT" ]; then
  echo "ORACLE-BUILD $(tr -d '\n' < "$REPORT")" >&2
fi
if [ ! -s "$REPORT" ] || ! grep -q '"ok": true' "$REPORT" || [ ! -s "$WORK/hand.FCStd" ]; then
  tail -c 3000 "$WORK/build.log" >&2
  echo "[solve] the oracle document did not build" >&2
  exit 1
fi
mv "$WORK/hand.FCStd" "$TASK_HOME/submission/hand.FCStd.part"
mv "$TASK_HOME/submission/hand.FCStd.part" "$TASK_HOME/submission/hand.FCStd"
if [ "$(id -u)" = 0 ]; then
  chown -R "$(basename "$TASK_HOME"):$(basename "$TASK_HOME")" "$TASK_HOME/submission"
fi
echo "[solve] $TASK_HOME/submission/hand.FCStd written" >&2
