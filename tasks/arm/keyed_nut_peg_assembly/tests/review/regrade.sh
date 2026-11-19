#!/usr/bin/env bash
# regrade.sh <image> <container name> <submission dir> <out dir> [tests dir]
#
# One verify run the way the separate verifier does it (a maintainer's tool; the grader never runs it): a fresh CPU
# container of the task image with [verifier.environment]'s resources (4 CPUs, 12 GB) and no network, the submission
# copied to /home/user/submission and left root-owned as the engine restores an artifact, tests/ staged root-only at
# /opt/ale/tests (no __pycache__), `bash test.sh` as root from there (ALE_HOME; the kit's envelope goes to
# ROBOTICS_GRADER_VERDICT_PATH since the engine's ale_verify is not staged here). Out: verdict.json, verify.stderr (the
# grader's log: one line per episode with kit_complete, nuts_assembled, steps and controller_s) and regrade.log (start,
# end, exit code, wall time). [tests dir] defaults to the tests/ folder this script sits in.
# Environment: CPUSET (optional, the host cores the container may use), CPUS (default 4), MEMORY (default 12g);
# DOCKER_HOST as for any docker call. The container is removed when it ends.
set -euo pipefail
if [ $# -lt 4 ]; then
  sed -n '2,13p' "$0" >&2
  exit 2
fi
IMAGE="$1"
NAME="$2"
SUB="$(realpath "$3")"
OUT="$(realpath -m "$4")"
TESTS="$(realpath "${5:-$(dirname "$0")/..}")"
mkdir -p "$OUT"
CPUARGS=(--cpus "${CPUS:-4}" --memory "${MEMORY:-12g}")
if [ -n "${CPUSET:-}" ]; then CPUARGS+=(--cpuset-cpus "$CPUSET"); fi
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
echo "[regrade] $(date -u +%FT%TZ) start $NAME image=$IMAGE ($IMAGE_ID) sub=$SUB tests=$TESTS" | tee "$OUT/regrade.log"
START=$(date +%s)
set +e
docker run --rm --name "$NAME" "${CPUARGS[@]}" --network none \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$SUB":/in/submission:ro -v "$TESTS":/in/tests:ro -v "$OUT":/out \
  --user root --entrypoint bash "$IMAGE" -c '
    set -e
    rm -rf /home/user/submission && cp -a /in/submission /home/user/submission && chown -R root:root /home/user/submission
    mkdir -p /opt/ale && rm -rf /opt/ale/tests && cp -a /in/tests /opt/ale/tests
    find /opt/ale/tests -name __pycache__ -prune -exec rm -rf {} +
    chown -R root:root /opt/ale/tests && chmod -R go-rwx /opt/ale/tests && chmod 700 /opt/ale
    cd /opt/ale/tests
    set +e
    ALE_HOME=/home/user ROBOTICS_GRADER_VERDICT_PATH=/out/verdict.json bash test.sh 2> /out/verify.stderr
    rc=$?
    chown -R "$HOST_UID:$HOST_GID" /out 2>/dev/null
    exit $rc'
RC=$?
set -e
END=$(date +%s)
echo "[regrade] $(date -u +%FT%TZ) done $NAME rc=$RC wall_s=$((END-START))" | tee -a "$OUT/regrade.log"
tail -3 "$OUT/verify.stderr" | tee -a "$OUT/regrade.log"
exit $RC
