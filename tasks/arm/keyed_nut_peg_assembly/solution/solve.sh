#!/usr/bin/env bash
# Privileged reference solution: a scripted controller that reads the ground-truth scene state
# (nut pose and keyway angle, peg poses). The grader attaches that state to the observations only
# for a submission carrying solution/oracle_token, whose SHA-256 alone is staged with the grader;
# the agent never sees the token. The solution only shows that every hidden episode can be done
# with the public robot, controller and step budget. Runs as the agent account; reads nothing
# outside solution/.
set -euo pipefail
SUB="${ALE_HOME:-/home/user}/submission"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SUB"
cp "$HERE"/solver/oracle_ctrl.py "$HERE"/solver/oracle_main.py "$SUB/"
cp "$HERE/oracle_token" "$SUB/oracle_token"
cat > "$SUB/run.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /opt/venv/bin/python3 -I oracle_main.py
SH
chmod +x "$SUB/run.sh"
echo "[oracle] submission ready at $SUB" >&2
