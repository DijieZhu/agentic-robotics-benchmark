#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=/opt/visual_robotics/client
exec /opt/venv/bin/python3 -B oracle.py
