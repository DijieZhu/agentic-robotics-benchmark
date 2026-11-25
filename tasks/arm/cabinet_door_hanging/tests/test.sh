#!/usr/bin/env bash
# Verify stage entry: runs as root after the agent phase. Root-stage hygiene first (the image
# sets HOME=/home/user): no user site, root-only caches and temp files, nothing world-readable.
set -euo pipefail
umask 077
export HOME=/root PYTHONNOUSERSITE=1 MESA_SHADER_CACHE_DISABLE=true
ROOT_TMP="$(mktemp -d)"
export TMPDIR="$ROOT_TMP" XDG_CACHE_HOME="$ROOT_TMP/cache" XDG_CONFIG_HOME="$ROOT_TMP/config" MPLCONFIGDIR="$ROOT_TMP/mpl"
unset PYTHONPATH PYTHONSTARTUP
exec "${ALE_VERIFY_PYTHON:-python3}" -I verify.py
