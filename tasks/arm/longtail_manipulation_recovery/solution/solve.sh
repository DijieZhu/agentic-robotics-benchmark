#!/usr/bin/env bash
# The reference controller, run in place of the agent by `ale validate`, as the same unprivileged account. It installs
# oracle_main.py, the recorded rescue action sequences and the 256-bit oracle token as .oracle_token; the grader compares
# the token's sha256 with tests/oracle_token.sha256 and, on a match, tells this controller which hidden state each
# episode is. Each sequence is the executed action list of a successful rescue rollout recorded on the authoring machine.
set -euo pipefail
SUB="${ALE_HOME:-/home/user}/submission"
A="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SUB/rescues"
cp "$A/oracle_main.py" "$SUB/oracle_main.py"
cp "$A"/rescues/*.npz "$SUB/rescues/"
cp "$A/oracle_token" "$SUB/.oracle_token"
cat > "$SUB/run.sh" <<'RUN'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell NUMBA_CPU_NAME=haswell
exec /opt/venv/bin/python3 -I oracle_main.py
RUN
chmod +x "$SUB/run.sh"
echo "[oracle] submission ready at $SUB" >&2
