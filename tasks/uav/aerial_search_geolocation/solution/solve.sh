#!/usr/bin/env bash
# The reference expert, run in place of the agent by `ale validate` as the same unprivileged
# account. It installs expert.py and the 256-bit oracle token as .oracle_token; the grader
# compares the token's sha256 with tests/hidden/oracle_token.sha256 and, on a match, adds
# the placed objects' true positions to every observation (privileged mode). It leaves a
# self-contained /home/user/submission/ that the verify stage drives.
set -euo pipefail

SUB="${ALE_HOME:-/home/user}/submission"
A="$(dirname "$0")/reference"
mkdir -p "$SUB"
cp "$A/expert.py" "$SUB/expert.py"
cp "$A/oracle_token" "$SUB/.oracle_token"

cat > "$SUB/run.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec /opt/venv/bin/python3 expert.py
SH
chmod +x "$SUB/run.sh"

echo "[oracle] submission ready at $SUB" >&2
