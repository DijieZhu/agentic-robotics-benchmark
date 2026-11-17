#!/usr/bin/env bash
# The reference supervisor, run in place of the agent by `ale validate`, as the same
# unprivileged account under the same limits. It is a privileged expert: it installs
# solver.py and the 256-bit oracle token as .oracle_token; the grader compares the token's
# sha256 with tests/oracle_token.sha256 and, on a match, adds the simulator's ground truth
# (road users, props, lane geometry) to every observation. The expert drives from that, with
# the measured collision footprints of the props (footprints.json), and never looks at the
# camera. It leaves a self-contained /home/user/submission/.
set -euo pipefail

SUB="${ALE_HOME:-/home/user}/submission"
A="$(dirname "$0")"
mkdir -p "$SUB"
cp "$A/solver.py" "$SUB/solver.py"
cp "$A/footprints.json" "$SUB/footprints.json"
cp "$A/reference/oracle_token" "$SUB/.oracle_token"

cat > "$SUB/run.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHONHASHSEED=0 exec /opt/venv/bin/python3 -B solver.py
RUN
chmod +x "$SUB/run.sh"

echo "[oracle] submission ready at $SUB" >&2
