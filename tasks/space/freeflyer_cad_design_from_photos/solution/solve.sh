#!/usr/bin/env bash
# Oracle: build the reference document natively in FreeCAD at oracle time, from solution/reference (oracle.json: the
# tags, the pins of the maker files, the skin parameters and the joint table; placement.json: the maker description's
# placement of every tag in configuration Z and its joints; maker/: byte copies of the maker's meshes and description),
# read by solution/makerstd (the task's mesh reader) and made into closed solids by solution/refsolids.py.
# build_oracle.py checks that nothing moves on recompute, solve, save, reopen and a second solve, and counts the
# reopened document; the document is moved into the submission only when it built cleanly.
set -euo pipefail
TASK_HOME="${ALE_HOME:-/home/user}"
HERE="$(cd "$(dirname "$0")" && pwd -P)"
FREECADCMD="${ARB_FREECADCMD:-/opt/freecad/usr/bin/freecadcmd}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
install -d -m 0755 "$TASK_HOME/submission"
# freecadcmd -c ends at once when its code is 256 characters or longer: the paths go through the environment
export FFCAD_ORACLE_PY="$HERE/build_oracle.py" FFCAD_ORACLE_REF="$HERE/reference" FFCAD_ORACLE_OUT="$WORK/unit.FCStd"
CODE='import os,sys;sys.dont_write_bytecode=True;e=os.environ;p=e["FFCAD_ORACLE_PY"];sys.argv=[p,e["FFCAD_ORACLE_REF"],e["FFCAD_ORACLE_OUT"]];exec(compile(open(p).read(),p,"exec"),{"__name__":"__main__","__file__":p})'
# a private HOME: FreeCAD writes its settings there, away from the desktop session's own
HOME="$WORK/home" XDG_CONFIG_HOME="$WORK/home/.config" XDG_CACHE_HOME="$WORK/home/.cache" \
  XDG_DATA_HOME="$WORK/home/.local/share" QT_QPA_PLATFORM=offscreen \
  "$FREECADCMD" -c "$CODE" > "$WORK/build.log" 2>&1 || true
grep -a 'ORACLE-BUILD' "$WORK/build.log" | sed 's/.*ORACLE-BUILD/ORACLE-BUILD/' >&2 || true
if ! grep -aq 'ORACLE-BUILD {.*"ok": true' "$WORK/build.log" || [ ! -s "$WORK/unit.FCStd" ]; then
  tail -c 3000 "$WORK/build.log" >&2
  echo "[solve] the oracle document did not build" >&2
  exit 1
fi
mv "$WORK/unit.FCStd" "$TASK_HOME/submission/unit.FCStd.part"
mv "$TASK_HOME/submission/unit.FCStd.part" "$TASK_HOME/submission/unit.FCStd"
if [ "$(id -u)" = 0 ]; then
  chown -R "$(basename "$TASK_HOME"):$(basename "$TASK_HOME")" "$TASK_HOME/submission"
fi
echo "[solve] $TASK_HOME/submission/unit.FCStd written" >&2
