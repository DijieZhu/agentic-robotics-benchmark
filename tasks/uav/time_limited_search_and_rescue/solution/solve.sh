#!/usr/bin/env bash
# The reference submission: the privileged pilot (solution/oracle/pilot.py), a run.sh that starts it, and the 256-bit
# oracle token as .oracle_token. The grader compares the token's SHA-256 with tests/oracle_token.sha256 and, on a
# match, starts each graded site's service with the privileged `_gt` block in its answers; the pilot plans on that.
set -euo pipefail
A="$(cd "$(dirname "$0")" && pwd)/oracle"
SUB=/home/user/submission
/usr/bin/mkdir -p "$SUB"
/usr/bin/cp "$A/pilot.py" "$SUB/pilot.py"
/usr/bin/cp "$A/run.sh" "$SUB/run.sh"
/usr/bin/cp "$A/oracle_token" "$SUB/.oracle_token"
