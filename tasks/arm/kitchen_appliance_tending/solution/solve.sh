#!/usr/bin/env bash
# Privileged reference solution: a scripted controller that knows the hidden units (solution/oracle/units.json and
# their models) and is told their placement and control states by the grader's token truth (tests/verify.py adds
# `truth` to every observation only when the submission's .oracle_token hashes to tests/hidden/oracle_token.sha256).
# It builds its own copy of the scene each episode (reference.Shadow) and runs one scripted primitive per request.
# Runs as the agent account with the agent's timeouts and network policy; spends no lab trials; reads nothing
# outside solution/.
set -euo pipefail
SUB="${ALE_HOME:-/home/user}/submission"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SUB"
cp -r "$HERE"/oracle/. "$SUB/"
cp "$HERE/oracle_token" "$SUB/.oracle_token"
cat > "$SUB/run.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /opt/venv/bin/python3 -I oracle_main.py --units units.json
SH
chmod +x "$SUB/run.sh"
echo "[oracle] submission ready at $SUB" >&2
