#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec "${ALE_VERIFY_PYTHON:-python3}" grader.py
