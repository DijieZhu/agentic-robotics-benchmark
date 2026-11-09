#!/usr/bin/env bash
# Minimal controller that speaks the grader's wire. Copy this folder to /home/user/submission
# to start from a working skeleton; replace controller.py with your own policy.
set -euo pipefail
cd "$(dirname "$0")"
exec /opt/venv/bin/python3 controller.py
