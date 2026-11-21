#!/bin/bash
# FreeCAD is the only application on the desktop; it is started again if it is closed.
export LANG=C.UTF-8 LC_ALL=C.UTF-8
cd "$HOME" || exit 1
# FreeCAD creates its per-version directories only on a first start; the preset preferences skip that.
mkdir -p "$HOME/.cache/FreeCAD/v1-1/Cache" "$HOME/.local/share/FreeCAD/v1-1"
while true; do
    /opt/freecad/AppRun >>/tmp/freecad-session.log 2>&1
    sleep 2
done
