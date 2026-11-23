#!/usr/bin/env bash
# The privileged oracle's mission program: one mission per launch, planned on the site's truth (token.txt).
set -euo pipefail
cd "$(dirname "$0")"
exec /opt/venv/bin/python3 -B oracle.py
