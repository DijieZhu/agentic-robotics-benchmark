#!/usr/bin/env bash
# Privileged reference solution: the fitter's program (solution/oracle/fitter.py) driven over the grading wire on the
# grader's token truth (tests/verify.py adds `truth` to every observation only when the submission's .oracle_token
# hashes to tests/hidden/oracle_token.sha256): the fixtures as mounted, the cable's points and the connector's
# tension. It is not a solution an agent could write: it shows that every board of the hidden set can be routed in
# the cell, within the step limit and without overloading the connector. Runs as the agent account with the agent's
# timeouts and network policy; reads nothing outside solution/.
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
exec /opt/venv/bin/python3 -I oracle_main.py
SH
chmod +x "$SUB/run.sh"
echo "[oracle] submission ready at $SUB" >&2
