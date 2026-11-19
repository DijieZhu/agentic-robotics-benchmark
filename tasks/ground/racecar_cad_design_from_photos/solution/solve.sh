#!/usr/bin/env bash
# Oracle: build the reference document natively in FreeCAD at oracle time (it is large: every given part's STEP is
# imported unchanged), from solution/reference (oracle.json: tags, families, the ground component, the given placements;
# parts/Pnn.brep: the made parts' solids, byte-identical to the grader's tests/reference/parts) and the payload that
# setup.sh staged in the agent home (base/base_assembly.step, purchased/<PN>/<PN>.step; oracle.json's step paths are
# relative to the home). build_oracle.py checks that nothing moves on recompute, solve, save and reopen, and writes its
# report next to the document (racecar.FCStd.report.json); the document is moved into the submission only when that
# report says it built cleanly.
# forked from tasks/uav/uav_cad_design_from_photos/solution/solve.sh (benchmark main b7b55ac, sha256
# f4c876339e791e0c4ee18b502bd904c1b2f001b8460e581d42890ffddebdea88): the file name racecar.FCStd, the agent home as the
# second argument, success read from the report file.
set -euo pipefail
TASK_HOME="${ALE_HOME:-/home/user}"
HERE="$(cd "$(dirname "$0")" && pwd -P)"
FREECADCMD="${ARB_FREECADCMD:-/opt/freecad/usr/bin/freecadcmd}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
install -d -m 0755 "$TASK_HOME/submission"
# a private HOME: FreeCAD writes its settings there, away from the desktop session's own
HOME="$WORK/home" QT_QPA_PLATFORM=offscreen "$FREECADCMD" -c "import sys; sys.argv = ['build_oracle.py', '$HERE/reference', '$TASK_HOME', '$WORK/racecar.FCStd']; exec(open('$HERE/build_oracle.py').read())" > "$WORK/build.log" 2>&1 || true
REPORT="$WORK/racecar.FCStd.report.json"
if [ -s "$REPORT" ]; then
  echo "ORACLE-BUILD $(tr -d '\n' < "$REPORT")" >&2
fi
if [ ! -s "$REPORT" ] || ! grep -q '"ok": true' "$REPORT" || [ ! -s "$WORK/racecar.FCStd" ]; then
  tail -c 3000 "$WORK/build.log" >&2
  echo "[solve] the oracle document did not build" >&2
  exit 1
fi
mv "$WORK/racecar.FCStd" "$TASK_HOME/submission/racecar.FCStd.part"
mv "$TASK_HOME/submission/racecar.FCStd.part" "$TASK_HOME/submission/racecar.FCStd"
if [ "$(id -u)" = 0 ]; then
  chown -R "$(basename "$TASK_HOME"):$(basename "$TASK_HOME")" "$TASK_HOME/submission"
fi
echo "[solve] $TASK_HOME/submission/racecar.FCStd written" >&2
