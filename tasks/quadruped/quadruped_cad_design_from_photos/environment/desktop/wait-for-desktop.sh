#!/bin/bash
# One-shot readiness log for the task's desktop (replaces the base's GNOME watchdog). It records, it does not gate:
# ALE waits for a desktop by taking a screenshot.
for _ in $(seq 1 120); do
    if [ -f /tmp/gnome-ready ] && pgrep -x openbox >/dev/null; then
        echo "desktop ready (openbox up)" >&2
        exit 0
    fi
    sleep 1
done
echo "desktop not ready after 120 s" >&2
exit 1
