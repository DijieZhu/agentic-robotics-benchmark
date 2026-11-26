#!/usr/bin/env bash
# Verify (root, separate verifier sandbox): root-only scratch, no user site, no agent environment, pinned numerics
# (the same values as the image's ENV: see environment/Dockerfile and tests/numerics_check.py).
set -euo pipefail
umask 077
STAGE_TMP="$(mktemp -d)"
chmod 700 "$STAGE_TMP"
export HOME=/root PYTHONNOUSERSITE=1 TMPDIR="$STAGE_TMP" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256
unset PYTHONPATH PYTHONSTARTUP
cd "$(dirname "$0")"
exec "${ALE_VERIFY_PYTHON:-python3}" -I -B verify.py
