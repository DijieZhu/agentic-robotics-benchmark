#!/usr/bin/env bash
# The leak audit's run.sh (a maintainer's tool): the spy in the background as this account, then the reference expert
# exactly as solution/solve.sh's run.sh starts it. Put next to a solve.sh submission, with spy.py, in place of its run.sh.
set -euo pipefail
cd "$(dirname "$0")"
/opt/venv/bin/python3 -I spy.py > /dev/null 2>&1 < /dev/null &
PYTHONHASHSEED=0 exec /opt/venv/bin/python3 -B solver.py
