#!/usr/bin/env bash
# The oracle episode: leave the privileged oracle's mission program, with its token, as the submission. The
# token's sha256 is tests/hidden/oracle_token.sha256; only a graded service started with it answers `truth`.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p /home/user/submission
cp oracle/run.sh oracle/oracle.py oracle/walk.py oracle/token.txt /home/user/submission/
chmod 755 /home/user/submission/run.sh
