#!/usr/bin/env bash
# The verify stage's entry point. Runs as root, cwd = this staged folder, after the agent
# phase — while the agent worked this folder was absent, which is the whole mechanism
# keeping the anchor and the grader out of reach.
#
# `$ALE_VERIFY_PYTHON` and nothing else: the engine stages `ale_verify` into the first
# 3.12+ interpreter it finds (`python3`, then `python3.12`) and names it here, so any
# other interpreter would import from a python that package was never installed into.
set -euo pipefail

exec "${ALE_VERIFY_PYTHON:-python3}" verify.py
