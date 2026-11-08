#!/usr/bin/env bash
# regrade.sh <image> <container name> <submission dir> <out dir> [tests dir]
#
# One verify run the way the separate verifier does it (a maintainer's tool; the grader never runs it): a fresh CPU
# container of the task image with [verifier.environment]'s resources (8 CPUs, 16 GB) and no network, the submission
# copied to /home/user/submission and left root-owned as the engine restores an artifact, tests/ staged root-only at
# /opt/ale/tests (no __pycache__), `bash test.sh` as root from there (ALE_STAGE_DIR, ALE_HOME). Out: verdict.json (the
# stage's verdict.local.json: the engine's ALE_VERIFY_PYTHON is not set here), verify.stderr (the grader's log, the
# reference's included), record/ (the grader's root-only record /opt/arb-verify: out/episodes.json, logs/) and
# regrade.log (start, end, exit code, wall time). [tests dir] defaults to the tests/ folder this script sits in.
# Environment: CPUSET (optional, e.g. 32-47: the host cores the container may use), CPUS (default 8), MEMORY (default
# 16g); DOCKER_HOST as for any docker call. The container is removed when it ends.
set -euo pipefail
if [ $# -lt 4 ]; then
  sed -n '2,14p' "$0" >&2
  exit 2
fi
IMAGE="$1"
NAME="$2"
SUB="$(realpath "$3")"
OUT="$(realpath -m "$4")"
TESTS="$(realpath "${5:-$(dirname "$0")/..}")"
mkdir -p "$OUT"
CPUARGS=(--cpus "${CPUS:-8}" --memory "${MEMORY:-16g}")
if [ -n "${CPUSET:-}" ]; then CPUARGS+=(--cpuset-cpus "$CPUSET"); fi
IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
echo "[regrade] $(date -u +%FT%TZ) start $NAME image=$IMAGE ($IMAGE_ID) sub=$SUB tests=$TESTS" | tee "$OUT/regrade.log"
START=$(date +%s)
set +e
docker run --rm --name "$NAME" "${CPUARGS[@]}" --network none \
  -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
  -v "$SUB":/in/submission:ro -v "$TESTS":/in/tests:ro -v "$OUT":/out \
  "$IMAGE" bash -c '
    set -e
    rm -rf /home/user/submission && cp -a /in/submission /home/user/submission && chown -R root:root /home/user/submission
    mkdir -p /opt/ale && rm -rf /opt/ale/tests && cp -a /in/tests /opt/ale/tests
    find /opt/ale/tests -name __pycache__ -prune -exec rm -rf {} +
    chown -R root:root /opt/ale/tests && chmod -R go-rwx /opt/ale/tests && chmod 700 /opt/ale
    cd /opt/ale/tests
    set +e
    ALE_STAGE_DIR=/opt/ale/tests ALE_HOME=/home/user bash test.sh 2> /out/verify.stderr
    rc=$?
    cp /opt/ale/tests/verdict.local.json /out/verdict.json 2>/dev/null
    mkdir -p /out/record && cp -r /opt/arb-verify/. /out/record/ 2>/dev/null
    chown -R "$HOST_UID:$HOST_GID" /out 2>/dev/null
    exit $rc'
RC=$?
set -e
END=$(date +%s)
echo "[regrade] $(date -u +%FT%TZ) done $NAME rc=$RC wall_s=$((END-START))" | tee -a "$OUT/regrade.log"
tail -3 "$OUT/verify.stderr" | tee -a "$OUT/regrade.log"
exit $RC
