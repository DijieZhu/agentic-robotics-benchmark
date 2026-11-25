#!/usr/bin/env bash
# Privileged reference solution: a scripted two-arm controller (reference.py) that reads the simulator's truth.
# It is driven over the grading wire (oracle_main.py, wire_world.py) on a shadow of the world that the grader's
# token truth sets every step: tests/verify.py adds `truth` to every observation only when the submission's
# .oracle_token hashes to tests/hidden/oracle_token.sha256. The right hand turns the burner on and takes the ladle out
# of the pot; the left hand tips the dishes the order needs into the broth, the slowest to cook first; the right hand
# serves the cooked pieces (scoop, tip unwanted ones back over the pot, tip a load that fits a bowl into it), lays the
# ladle down and turns the burner off; both hands then take the pot by its tabs, pour each bowl's broth by rolling the
# pot about the line between the tabs with the tilt servoed on the flow, and set the pot back on its ring. Runs as the agent account with the agent's
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
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa   # the grader starts run.sh with PATH, HOME and LANG only
exec /opt/venv/bin/python3 -I oracle_main.py
SH
chmod +x "$SUB/run.sh"
echo "[oracle] submission ready at $SUB" >&2
