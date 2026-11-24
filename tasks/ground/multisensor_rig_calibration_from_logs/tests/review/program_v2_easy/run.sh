#!/bin/bash
cd "$(dirname "$0")"
IN=""; OUT=""
while [ $# -gt 0 ]; do case "$1" in --input) IN="$2"; shift 2;; --output) OUT="$2"; shift 2;; *) shift;; esac; done
python3 baseline.py --input "$IN" --output "$OUT"
if [ -f calib.py ]; then python3 calib.py --input "$IN" --output "$OUT"; fi
