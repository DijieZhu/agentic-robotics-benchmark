#!/usr/bin/env bash
# Oracle: the photographed robot as the FreeCAD document the task asks for (42 PartDesign Bodies labelled P01..P42, the 12
# actuator units and 4 foot pads imported from the given STEP files, one Assembly with its Grounded, Fixed and Revolute
# joints and limits, saved in configuration Z), written to $ALE_HOME/submission/quadruped.FCStd. `ale validate` stages
# this folder in the agent's home and runs this script as the agent account (cwd = the staged folder).
#
# The document is BUILT here rather than shipped: prebuilt it is about 41 MB (PartDesign stores every part's B-rep three
# times), above the task's 20 MB limit for a stored oracle. reference/ holds the inputs: assembly.json and parts/*.step
# (byte-identical to the grader's tests/reference/ files: the robot in its own form, one STEP file per part geometry),
# core.step and foot.step (byte-identical to the payload's actuator and foot pad STEP files) and oracle_build.py, run by
# the image's headless FreeCAD (12-20 s, most of it the save). Scores 1.0 on every gate and criterion.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd -P)"
REF="$HERE/reference"
HOME_DIR="${ALE_HOME:-/home/user}"
SUB="$HOME_DIR/submission"
FREECADCMD="${ARB_FREECADCMD:-/opt/freecad/usr/bin/freecadcmd}"     # ARB_FREECADCMD: host runs of this script only
for f in assembly.json core.step foot.step oracle_build.py; do
  [ -f "$REF/$f" ] || { echo "[oracle] $REF/$f missing" >&2; exit 1; }
done
[ -x "$FREECADCMD" ] || { echo "[oracle] FreeCAD's headless interpreter $FREECADCMD not found" >&2; exit 1; }
mkdir -p "$SUB"
# FreeCAD writes its settings and caches under HOME: a fresh private one, never the account's own (the GUI session's)
WORK="$(mktemp -d "${TMPDIR:-/tmp}/ale_oracle.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/home"
OUT="$WORK/quadruped.FCStd"
t0=$(date +%s)
# argv of the builder as Python literals (paths without quotes or backslashes: checked)
case "$REF$OUT" in *\'*|*\\*) echo "[oracle] unexpected characters in $REF or $OUT" >&2; exit 1 ;; esac
PY="import sys; R = '$REF'; sys.argv = ['oracle_build.py', R, R + '/core.step', R + '/foot.step', '$OUT']; exec(open(R + '/oracle_build.py').read())"
set +e
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 HOME="$WORK/home" XDG_CONFIG_HOME="$WORK/home/.config" \
    XDG_CACHE_HOME="$WORK/home/.cache" XDG_DATA_HOME="$WORK/home/.local/share" TMPDIR="$WORK" QT_QPA_PLATFORM=offscreen \
    timeout -k 30 1800 "$FREECADCMD" -c "$PY" > "$WORK/build.log" 2>&1
code=$?
set -e
rec="$(grep -a 'ORACLE-BUILD ' "$WORK/build.log" | tail -n 1 || true)"
if [ "$code" -ne 0 ] || [ ! -s "$OUT" ] || [ -z "$rec" ]; then
  echo "[oracle] building the document failed (exit $code); the builder's last output:" >&2
  tail -c 3000 "$WORK/build.log" >&2
  exit 1
fi
# the builder's own record must show the whole robot: 42 Bodies, 12 units, 4 pads, 12 Revolute joints
for want in '"bodies": 42' '"cores": 12' '"feet": 4' '"revolute": 12'; do
  case "$rec" in *"$want"*) ;; *) echo "[oracle] incomplete document ($want expected): $rec" >&2; exit 1 ;; esac
done
# into the submission folder by a rename on its own file system (no partial quadruped.FCStd at any time)
cp "$OUT" "$SUB/.quadruped.FCStd.part"
mv -f "$SUB/.quadruped.FCStd.part" "$SUB/quadruped.FCStd"
echo "[oracle] document built in $(( $(date +%s) - t0 )) s and installed at $SUB/quadruped.FCStd ($(stat -c %s "$SUB/quadruped.FCStd") bytes)" >&2
echo "[oracle] ${rec#*ORACLE-BUILD }" >&2
