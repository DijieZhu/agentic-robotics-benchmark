#!/usr/bin/env bash
# Privileged reference solution: a scripted two-arm controller (reference.py) that reads the simulator's truth.
# It is driven over the grading wire (oracle_main.py, wire_world.py) on a shadow of the world that the grader's
# token truth sets every step: tests/verify.py adds `truth` to every observation only when the submission's
# .oracle_token hashes to tests/hidden/oracle_token.sha256. The right hand takes one elbow and holds it at the work
# station; the left hand pushes pipe B into its outlet, then pipe A's banded end into its other socket; the right
# hand turns the spool about A's axis as the order's twist needs, and the left hand takes the other elbow, turns it to
# the twist and pushes it onto A's plain end; the right hand sets the spool down on the bench and both hands back off.
# Runs as the agent account with the agent's timeouts and network policy; reads nothing outside solution/.
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
