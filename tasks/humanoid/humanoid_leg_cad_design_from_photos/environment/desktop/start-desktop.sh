#!/bin/bash
# The task's desktop session: openbox (no compositor), a panel (windows and a clock) and FreeCAD, nothing else. It replaces the base image's GNOME
# session, whose compositor stops repainting under Xvnc with llvmpipe on some hosts (DESIGN.md section 3). The readiness
# files keep the base's contract: cua-driver waits for /tmp/gnome-ready and reads /tmp/dbus-session-bus-address.
set -u
while ! xdpyinfo -display "$DISPLAY" >/dev/null 2>&1; do sleep 0.3; done
mkdir -p "$XDG_RUNTIME_DIR" && chmod 700 "$XDG_RUNTIME_DIR"
exec dbus-run-session -- bash -c '
  echo "$DBUS_SESSION_BUS_ADDRESS" > /tmp/dbus-session-bus-address
  chmod 644 /tmp/dbus-session-bus-address
  xsetroot -solid "#3b4252" 2>/dev/null || true
  openbox --config-file /etc/arb-desktop/openbox-rc.xml &
  for _ in $(seq 1 50); do pgrep -u "$(id -u)" -x openbox >/dev/null && break; sleep 0.2; done
  tint2 -c /etc/arb-desktop/tint2rc >/tmp/tint2.log 2>&1 &
  touch /tmp/gnome-ready
  exec /usr/local/bin/freecad-session.sh
'
