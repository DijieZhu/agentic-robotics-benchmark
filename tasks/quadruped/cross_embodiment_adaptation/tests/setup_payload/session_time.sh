#!/usr/bin/env bash
# Minutes used and left of the 120-minute development session (the start is stamped by the task's setup).
start=$(cat /run/session/start 2>/dev/null) || { echo "session: no start stamp"; exit 1; }
used=$(( ( $(date +%s) - start ) / 60 )); left=$(( 120 - used )); [ "$left" -lt 0 ] && left=0
echo "session: $used min used, $left min left of 120"
