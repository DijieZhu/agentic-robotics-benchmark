#!/usr/bin/env bash
# launch_ref.sh STAGE_DIR OUT_DIR CPUSET SEED [run_ref.py flags]: one privileged reference episode (run_ref.py) in the
# task image, as a detached container pinned to CPUSET, with the grader's numeric environment (tests/test.sh's BLAS
# pins). STAGE_DIR is a copy of this task's tests/ folder; IMG (environment variable) the task image built from
# environment/.
set -euo pipefail
STAGE="$1"; OUT="$2"; CPUS="$3"; SEED="$4"; shift 4
IMG="${IMG:?set IMG to the task image}"
TOOLS="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$OUT"
docker run -d --rm --name "hc-ref-$(basename "$OUT")-$SEED" --cpuset-cpus "$CPUS" --memory 12g --network none \
  -e HOME=/root -e PYTHONNOUSERSITE=1 -e MUJOCO_GL=osmesa -e PYOPENGL_PLATFORM=osmesa \
  -e OPENBLAS_CORETYPE=Haswell -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$STAGE":/stage:ro -v "$OUT":/out -v "$TOOLS":/tools:ro \
  --entrypoint /opt/venv/bin/python3 "$IMG" -s -B /tools/run_ref.py /stage "$SEED" "/out/seed_$SEED.json" "$@"
