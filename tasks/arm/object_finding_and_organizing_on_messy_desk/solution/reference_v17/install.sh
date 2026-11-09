#!/usr/bin/env bash
# install.sh [submission dir]: put the non-privileged reference controller (oracle v17: the
# observation only, three static RGB-D cameras, the hand camera, proprioception, the request
# text, with reference views of every model alone on the desk in refs.json) into a submission
# folder, for a re-measurement on the hidden episodes. solve.sh does not use it.
set -euo pipefail

SUB="${1:-${ALE_HOME:-/home/user}/submission}"
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SUB"
cp "$HERE/solver.py" "$HERE/refs.json" "$SUB/"
cat > "$SUB/run.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
exec /opt/venv/bin/python3 solver.py
SH
chmod +x "$SUB/run.sh"
echo "[reference_v17] submission ready at $SUB" >&2
