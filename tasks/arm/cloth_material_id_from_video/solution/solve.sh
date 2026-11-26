#!/usr/bin/env bash
# Privileged reference: writes the reference materials of the ten hidden fabrics (solution/ref/params.json: each
# calibrated against the hidden model of the real fabric by tests/review/make_set.py) as the submission.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p /home/user/submission
cp ref/params.json /home/user/submission/params.json
chmod 0644 /home/user/submission/params.json
