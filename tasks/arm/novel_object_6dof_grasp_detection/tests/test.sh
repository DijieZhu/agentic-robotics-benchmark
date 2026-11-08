#!/usr/bin/env bash
# The verify stage's entry point. Runs as root, cwd = this staged folder, after the agent
# phase — while the agent worked this folder was absent, which is the whole mechanism
# keeping the anchor and the grader out of reach.
#
# `python3` and nothing else: the engine stages `ale_verify` into the interpreter that
# answers to that name, so selecting a different one would import from a python that
# package was never installed into.
set -euo pipefail

exec "${ALE_VERIFY_PYTHON:-python3}" verify.py
