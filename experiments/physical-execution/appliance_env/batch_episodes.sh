#!/usr/bin/env bash
# P2: whole reference episodes (run_episodes.py) on every candidate model x seeds, sharded.
# Usage: bash batch_episodes.sh <out_dir> <shards> <seeds...>
set -euo pipefail
OUT=$1; SHARDS=$2; shift 2; SEEDS="$*"
A=~/arb-data/robocasa-assets/extracted/fixtures
P=~/arb-data/venv-p1/bin/python
mkdir -p "$OUT"
MODELS=$(for k in stovetops microwaves ovens stoves; do ls "$A/$k" | sed "s#^#$k/#"; done | grep -v -e Stove001 -e Stove074)
i=0
for m in $MODELS; do echo "$m" >> "$OUT/shard_$((i % SHARDS)).txt"; i=$((i+1)); done
for f in "$OUT"/shard_*.txt; do
  n=$(basename "$f" .txt)
  MUJOCO_GL=osmesa OMP_NUM_THREADS=1 nohup "$P" -I run_episodes.py --assets "$A" --out "$OUT/$n.jsonl" \
    --models $(cat "$f") --seeds $SEEDS > "$OUT/$n.log" 2>&1 &
done
wait
