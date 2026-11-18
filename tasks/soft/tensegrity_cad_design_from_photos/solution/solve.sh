#!/usr/bin/env bash
# Oracle: build the reference document natively in FreeCAD at oracle time (it is large: every manufactured part is its
# original triangle mesh as a faceted solid, and the vendor STEP of every purchased part is imported unchanged), from
# solution/reference (oracle.json: tags, families, instance poses, purchased placements, joints; parts/<family>.ply: the
# original manufactured parts, byte-identical to the grader's tests/reference/parts) and the vendor files that setup.sh
# staged in the agent home (~/purchased/<PN>/<PN>.step). build_oracle.py checks every input's sha256 and that nothing
# moves on recompute, solve, save, reopen and a second solve; the document is moved into the submission only when it
# built cleanly.
set -euo pipefail
TASK_HOME="${ALE_HOME:-/home/user}"
HERE="$(cd "$(dirname "$0")" && pwd -P)"
FREECADCMD="${ARB_FREECADCMD:-/opt/freecad/usr/bin/freecadcmd}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
install -d -m 0755 "$TASK_HOME/submission"
# a private HOME: FreeCAD writes its settings there, away from the desktop session's own
HOME="$WORK/home" QT_QPA_PLATFORM=offscreen "$FREECADCMD" -c "import sys; sys.argv = ['build_oracle.py', '$HERE/reference', '$TASK_HOME/purchased', '$WORK/strut.FCStd']; exec(open('$HERE/build_oracle.py').read())" > "$WORK/build.log" 2>&1 || true
grep -a 'ORACLE-BUILD' "$WORK/build.log" | sed 's/.*ORACLE-BUILD/ORACLE-BUILD/' >&2 || true
if ! grep -aq 'ORACLE-BUILD {.*"ok": true' "$WORK/build.log" || [ ! -s "$WORK/strut.FCStd" ]; then
  tail -c 3000 "$WORK/build.log" >&2
  echo "[solve] the oracle document did not build" >&2
  exit 1
fi
mv "$WORK/strut.FCStd" "$TASK_HOME/submission/strut.FCStd.part"
mv "$TASK_HOME/submission/strut.FCStd.part" "$TASK_HOME/submission/strut.FCStd"
if [ "$(id -u)" = 0 ]; then
  chown -R "$(basename "$TASK_HOME"):$(basename "$TASK_HOME")" "$TASK_HOME/submission"
fi
echo "[solve] $TASK_HOME/submission/strut.FCStd written" >&2
