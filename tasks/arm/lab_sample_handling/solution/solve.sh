#!/usr/bin/env bash
# Privileged reference solution: the worklist reference (camera estimators for the bench and the rotor, the scale
# for the masses, a transfer planned in joint space, the rotor turned to the loading port by a turn policy that
# identifies the unit's slip and play from its first grips) driven over the grading wire on a shadow of the world that the grader's
# token truth sets every step (tests/verify.py adds `truth` to every observation only when the submission's
# .oracle_token hashes to tests/hidden/oracle_token.sha256). Runs as the agent account with the agent's timeouts and
# network policy; reads nothing outside solution/.
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
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa   # the grader starts run.sh with PATH, HOME and LANG only
exec /opt/venv/bin/python3 -I oracle_main.py
SH
chmod +x "$SUB/run.sh"
echo "[oracle] submission ready at $SUB" >&2
