#!/usr/bin/env bash
# Privileged reference solution. It is NOT a calibrator: tests/verify.py hands each rig's true parameters and fault
# list (`--truth <dir>`) to a submission whose .oracle_token hashes to tests/hidden/oracle_token.sha256, and to no
# other; the program writes them out in the output format. It shows that the hidden rigs, their truth and the
# grading path agree (the reward of the truth is the anchor). Runs as the agent account with the agent's limits.
set -euo pipefail
SUB="${ALE_HOME:-/home/user}/submission"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SUB"
cp "$HERE/oracle/oracle_main.py" "$SUB/oracle_main.py"
cp "$HERE/oracle_token" "$SUB/.oracle_token"
cat > "$SUB/run.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec /opt/venv/bin/python3 -I oracle_main.py "$@"
SH
chmod +x "$SUB/run.sh"
echo "[oracle] submission ready at $SUB" >&2
