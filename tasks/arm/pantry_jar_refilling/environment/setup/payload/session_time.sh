#!/usr/bin/env bash
# How much of the 2-hour session is used and left. The stamp was written when this sandbox was set up, a few minutes
# before the session started, so the time left is slightly more than printed, never less.
#   bash /home/user/session_time.sh
set -u
LIMIT_MIN=120
STAMP="$(cd "$(dirname "$0")" && pwd)/.session_start"
if ! start=$(cat "$STAMP" 2>/dev/null) || ! [ "$start" -gt 0 ] 2>/dev/null; then
  echo "session_time: no stamp at $STAMP; note the time with 'date' yourself" >&2
  exit 1
fi
now=$(date -u +%s)
used=$(( (now - start) / 60 ))
left=$(( LIMIT_MIN - used ))
[ "$left" -lt 0 ] && left=0
echo "session: ${used} min used, ${left} min left of ${LIMIT_MIN} (set up $(date -u -d "@$start" +%H:%M:%S) UTC, now $(date -u -d "@$now" +%H:%M:%S) UTC)"
