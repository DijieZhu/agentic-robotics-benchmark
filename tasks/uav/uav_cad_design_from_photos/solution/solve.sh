#!/usr/bin/env bash
# Oracle: build the reference document natively in FreeCAD at oracle time (it is large: every purchased module's vendor
# STEP is imported unchanged), from solution/reference (oracle.json: tags, families, purchased placements; parts/Pnn.brep:
# the original manufactured solids, byte-identical to the grader's tests/reference/parts) and the vendor files that
# setup.sh staged in the agent home (~/purchased/<PN>/<PN>.step). build_oracle.py checks that nothing moves on recompute,
# solve, save and reopen; the document is moved into the submission only when it built cleanly.
set -euo pipefail
TASK_HOME="${ALE_HOME:-/home/user}"
HERE="$(cd "$(dirname "$0")" && pwd -P)"
FREECADCMD="${ARB_FREECADCMD:-/opt/freecad/usr/bin/freecadcmd}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
install -d -m 0755 "$TASK_HOME/submission"
# a private HOME: FreeCAD writes its settings there, away from the desktop session's own
HOME="$WORK/home" QT_QPA_PLATFORM=offscreen "$FREECADCMD" -c "import sys; sys.argv = ['build_oracle.py', '$HERE/reference', '$TASK_HOME/purchased', '$WORK/drone.FCStd']; exec(open('$HERE/build_oracle.py').read())" > "$WORK/build.log" 2>&1 || true
grep -a 'ORACLE-BUILD' "$WORK/build.log" | sed 's/.*ORACLE-BUILD/ORACLE-BUILD/' >&2 || true
if ! grep -aq 'ORACLE-BUILD {.*"ok": true' "$WORK/build.log" || [ ! -s "$WORK/drone.FCStd" ]; then
  tail -c 3000 "$WORK/build.log" >&2
  echo "[solve] the oracle document did not build" >&2
  exit 1
fi
mv "$WORK/drone.FCStd" "$TASK_HOME/submission/drone.FCStd.part"
mv "$TASK_HOME/submission/drone.FCStd.part" "$TASK_HOME/submission/drone.FCStd"
if [ "$(id -u)" = 0 ]; then
  chown -R "$(basename "$TASK_HOME"):$(basename "$TASK_HOME")" "$TASK_HOME/submission"
fi
echo "[solve] $TASK_HOME/submission/drone.FCStd written" >&2
