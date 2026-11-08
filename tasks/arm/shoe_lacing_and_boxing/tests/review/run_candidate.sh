#!/usr/bin/env bash
# run_candidate.sh <image> <container name> <submission dir> <seed> <out.json> [--dev]
#
# One episode through the grader's own call (verify.run_episode, by `select_set.py run-one`) in a fresh CPU container
# of the task image with no network (a maintainer's tool; the grader never runs it). The submission (the one
# solution/solve.sh installs, made beforehand) is copied to /home/user/submission and left root-owned as the engine
# restores an artifact; tests/ is staged root-only at /opt/ale/tests (no __pycache__); the grader's root-owned givens
# at /opt/arb are built as tests/test.sh builds them; the grader's environment is test.sh's (HOME=/root,
# PYTHONNOUSERSITE, OSMesa, python3 -s) with the image's own BLAS settings. The task folder is mounted read-only at
# /in/task for the frozen-file check. <out.json> gets the run's record, <out>.summary.json the cell's whole summary,
# <out>.stderr the grader's and the reference's log. The container is removed when it ends.
# Environment: CPUSET (the host cores the container may use, e.g. 32-47), CPUS (its CFS quota, default 1), MEMORY
# (default 8g); DOCKER_HOST as for any docker call. Without --dev it refuses an image other than the one
# tests/hidden/selection.json records, and run-one refuses frozen files that differ from the record.
set -euo pipefail
if [ $# -lt 5 ]; then
  sed -n '2,16p' "$0" >&2
  exit 2
fi
IMAGE="$1"
NAME="$2"
SUB="$(realpath "$3")"
SEED="$4"
OUT="$(realpath -m "$5")"
MODE="${6:-}"
HERE="$(cd "$(dirname "$0")" && pwd -P)"
TASK="$(dirname "$(dirname "$HERE")")"
SEL="$TASK/tests/hidden/selection.json"
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
ARGS="--seed $SEED --out /out/$(basename "$OUT") --label $NAME --task /in/task"
if [ -f "$SEL" ]; then
  ARGS="$ARGS --selection /opt/ale/tests/hidden/selection.json"
fi
if [ "$MODE" = "--dev" ]; then
  ARGS="$ARGS --dev"
else
  WANT="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["image"]["id"])' "$SEL")"
  [ "$IMAGE_ID" = "$WANT" ] || { echo "[run_candidate] image $IMAGE is $IMAGE_ID; the selection records $WANT" >&2; exit 2; }
fi
STEM="$(basename "$OUT" .json)"
mkdir -p "$(dirname "$OUT")"
CPUARGS=(--cpus "${CPUS:-1}" --memory "${MEMORY:-8g}")
if [ -n "${CPUSET:-}" ]; then CPUARGS+=(--cpuset-cpus "$CPUSET"); fi
echo "[run_candidate] $(date -u +%FT%TZ) start $NAME seed=$SEED image=$IMAGE_ID"
set +e
docker run --rm --name "$NAME" "${CPUARGS[@]}" --network none \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e PR48_IMAGE_ID="$IMAGE_ID" -e RUN_ARGS="$ARGS" -e STEM="$STEM" \
  -v "$SUB":/in/submission:ro -v "$TASK":/in/task:ro -v "$(dirname "$OUT")":/out \
  "$IMAGE" bash -c '
    set -e
    rm -rf /home/user/submission && cp -a /in/submission /home/user/submission && chown -R root:root /home/user/submission
    mkdir -p /opt/ale && rm -rf /opt/ale/tests && cp -a /in/task/tests /opt/ale/tests
    find /opt/ale/tests -name __pycache__ -prune -exec rm -rf {} +
    chown -R root:root /opt/ale/tests && chmod -R go-rwx /opt/ale/tests && chmod 700 /opt/ale
    rm -rf /opt/arb && mkdir -p /opt/arb/given
    cp -r /opt/ale/tests/setup_payload/. /opt/arb/
    cp -r /opt/task-assets/panda /opt/arb/given/panda
    chown -R root:root /opt/arb && chmod -R u+rwX,go+rX,go-w /opt/arb
    export HOME=/root PYTHONNOUSERSITE=1 XDG_CONFIG_HOME=/root/.config XDG_CACHE_HOME=/root/.cache \
      MPLCONFIGDIR=/root/.config/matplotlib MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
      ALE_STAGE_DIR=/opt/ale/tests ALE_HOME=/home/user
    cd /opt/ale/tests
    set +e
    python3 -s -B review/select_set.py run-one $RUN_ARGS 2> "/out/$STEM.stderr"
    rc=$?
    chown "$HOST_UID:$HOST_GID" /out/"$STEM".* 2>/dev/null
    exit $rc'
RC=$?
set -e
echo "[run_candidate] $(date -u +%FT%TZ) done $NAME rc=$RC"
exit $RC
