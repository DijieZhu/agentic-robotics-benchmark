#!/usr/bin/env bash
# The verify stage's entry point. After the agent phase the engine uploads tests/ to
# /opt/ale/tests (root-only) and runs `bash test.sh` there as root, cwd = that folder,
# network open — while the agent worked this folder was absent, which is the whole
# mechanism keeping the anchor and the grader out of reach.
#
# `$ALE_VERIFY_PYTHON` and nothing else: before this script starts, the engine stages its
# `ale_verify` package into the first interpreter that is Python 3.12 or newer — `python3`,
# else `python3.12` — and names that interpreter in ALE_VERIFY_PYTHON (in this image, the
# /opt/venv `python3`). The kit's write_verdict writes /logs/verifier/reward.json through
# that package, so any other interpreter would import from a python it was never installed
# into. Run by hand outside the engine (in the image, with ROBOTICS_GRADER_VERDICT_PATH set),
# the variable is unset and `python3` stands in.
set -euo pipefail

exec "${ALE_VERIFY_PYTHON:-python3}" verify.py
