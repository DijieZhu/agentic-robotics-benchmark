#!/usr/bin/env bash
set -euo pipefail
TASK_HOME="${ALE_HOME:-/home/user}"
mkdir -p "$TASK_HOME/submission"
cp assemble.py "$TASK_HOME/submission/assemble.py"
cp token.txt "$TASK_HOME/submission/.oracle_token"
