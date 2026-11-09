#!/usr/bin/env bash
# The task's reference solution, run in place of the agent by `ale validate`, as the same
# unprivileged account with the same timeout. It installs the PRIVILEGED reference controller
# (solution/privileged/oracle.py) and the reference token: the grader hands a submission whose
# .oracle_token matches the digest in tests/ the simulator's ground truth in every observation,
# and the controller plans from it (true object identities and poses, container contents, the
# open request's object). It shows that the hidden episodes can be done; the anchor is its own
# measured score on them. An agent gets no such block and works from the cameras.
#
# The non-privileged reference that works from the observation alone (oracle v17, the evidence
# that the agent's perception suffices) is solution/reference_v17/; install.sh there puts it in
# a submission folder for a re-measurement. This script does not use it.
set -euo pipefail

SUB="${ALE_HOME:-/home/user}/submission"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SUB"

cp "$HERE/privileged/oracle.py" "$HERE/privileged/geometry.json" "$SUB/"
cp "$HERE/privileged/oracle_token" "$SUB/.oracle_token"

cat > "$SUB/run.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MESSY_DESK_ORACLE_LOG=0
exec /opt/venv/bin/python3 -B oracle.py
SH
chmod +x "$SUB/run.sh"

echo "[oracle] privileged submission ready at $SUB" >&2
