#!/usr/bin/env bash
# regrade.sh <gpu index> <container name> <submission dir> <out dir> [tests dir]
#
# One verify run the way the separate verifier does it (a maintainer's tool; the grader never runs it): a fresh
# container of the task image (no network, 32 GB, the cores in CPUSET, one GPU with NVIDIA_DRIVER_CAPABILITIES
# including graphics), the submission copied to /home/user/submission and left root-owned as the engine restores an
# artifact (verify.py hands it to the agent account), no setup and no ~/drive, tests/ staged root-only at
# /opt/ale/tests (no __pycache__), `bash test.sh` as root. The verdict goes to <out>/verdict.json through
# ROBOTICS_GRADER_VERDICT_PATH, the grader's stderr to <out>/verify.stderr, the submission folder as the supervisor
# left it to <out>/submission_after.tar. [tests dir] defaults to the tests/ folder this script sits in. The
# container starts only once that GPU has 8 GB free, and is removed when it ends.
# Environment: CPUSET (default 16-23), IMAGE (default the task image of review/dev_batch.py), DOCKER_HOST as for any
# docker call.
set -euo pipefail
if [ $# -lt 4 ]; then
  sed -n '2,14p' "$0" >&2
  exit 2
fi
GPU="$1"
NAME="$2"
SUB="$(realpath "$3")"
OUT="$(realpath -m "$4")"
TESTS="$(realpath "${5:-$(dirname "$0")/..}")"
IMAGE="${IMAGE:-ale-framework:c803df660b128dc54c63d9d48cca95b5}"
CPUSET="${CPUSET:-16-23}"
UUIDS=(GPU-d19373da-d7df-4311-0891-9f572961c478 GPU-18963cff-56a1-bce7-9532-6af2a14ebe90)
mkdir -p "$OUT"
while true; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
  if [ "$free" -ge 8192 ]; then break; fi
  echo "[regrade] $(date -u +%T) GPU $GPU has ${free} MiB free; waiting" >&2
  sleep 60
done
echo "[regrade] $(date -u +%FT%TZ) start $NAME image=$IMAGE gpu=$GPU (${free} MiB free) cpus=$CPUSET sub=$SUB" \
  "tests=$TESTS" | tee "$OUT/regrade.log"
START=$(date +%s)
set +e
docker run --rm --name "$NAME" --cpuset-cpus "$CPUSET" --memory 32g --shm-size 8g --network none \
  --gpus "device=${UUIDS[$GPU]}" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" --user root \
  -v "$SUB":/in/submission:ro -v "$TESTS":/in/tests:ro -v "$OUT":/out \
  "$IMAGE" bash -c '
    set -e
    rm -rf /home/user/submission /home/user/drive /home/user/practice_grader
    cp -a /in/submission /home/user/submission && chown -R root:root /home/user/submission
    mkdir -p /opt/ale && rm -rf /opt/ale/tests && cp -a /in/tests /opt/ale/tests
    find /opt/ale/tests -name __pycache__ -prune -exec rm -rf {} +
    chown -R root:root /opt/ale/tests && chmod -R go-rwx /opt/ale/tests && chmod 700 /opt/ale
    cd /opt/ale/tests
    set +e
    ALE_STAGE_DIR=/opt/ale/tests ALE_HOME=/home/user ROBOTICS_GRADER_VERDICT_PATH=/out/verdict.json \
      bash test.sh 2> /out/verify.stderr
    rc=$?
    tar -C /home/user -cf /out/submission_after.tar submission 2>/dev/null
    chown "$HOST_UID:$HOST_GID" /out/verdict.json /out/verify.stderr /out/submission_after.tar 2>/dev/null
    exit $rc'
RC=$?
set -e
END=$(date +%s)
echo "[regrade] $(date -u +%FT%TZ) done $NAME rc=$RC wall_s=$((END-START))" | tee -a "$OUT/regrade.log"
tail -3 "$OUT/verify.stderr" | tee -a "$OUT/regrade.log"
exit $RC
