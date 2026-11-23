#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec /usr/bin/env -i PATH=/opt/venv/bin:/usr/bin:/bin HOME=/root LANG=C.UTF-8 /opt/venv/bin/python3 -E -s setup_runtime.py
