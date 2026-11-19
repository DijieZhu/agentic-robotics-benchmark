#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase. Stages the givens (the robot's documents, the
# plant's documents and instrument catalogue, the training hall's sheets, the client library, the example program and
# the practice launcher) into the agent home, owned by the agent, and starts the practice ground (below). The grader
# never reads these copies: verification is separate (task.toml), and tests/test.sh builds the grader's own ROOT-OWNED
# read-only copy of the same givens in the fresh verifier sandbox, from tests/setup_payload/ (a byte copy of payload/
# here: change both together, then tests/given_hashes.json).
# Dynamic-only: packages and the practice ground's program and site packages live in environment/Dockerfile.
set -euo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/"
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2

# The practice ground: the image installs it under /opt/plantcell-practice for the account `plantcell` only (the agent
# account can neither read nor list it); setup starts it now, to run through the agent's session, under a loop that
# restarts it should it ever exit (its mission allowance survives a restart). The agent reaches it through
# /run/plantcell/practice.sock (plantcell.client.connect_practice, ~/training/run_mission.py): the robot's public calls,
# reset(site) and each mission's debrief. Its numerics, renderer settings and render bundle are pinned as the grader
# pins them (tests/test.sh): 8 rasteriser threads, BLAS and OpenMP 2, the bundle required at its one path (checked here
# first, its manifest against the pin of the practice ground's cell and every file against the manifest: a broken image
# fails the setup, never a practice mission without the bundle; the cell checks the same again at every mission).
# Amendment 1: the dressing's shadow switch is pinned off as tests/test.sh pins it, and the dressing's
# variables are set here only, not in the image's ENV (the agent's shell does not show them).
# The BLAS kernel pin carries OPENBLAS_L2_SIZE=256 here as in tests/test.sh, which says why.
SERVICE=/opt/plantcell-practice
BUNDLE=/opt/task-assets/look
[ -f "$SERVICE/practiced.py" ] || { echo "[setup] the image lacks the practice ground ($SERVICE)" >&2; exit 1; }
/opt/venv/bin/python3 -I -s -c 'import hashlib, json, os, sys
r, pin = sys.argv[1], sys.argv[2]
raw = open(os.path.join(r, "manifest.json"), "rb").read()
assert hashlib.sha256(raw).hexdigest() == json.load(open(pin))["manifest_sha256"], "manifest not the pinned one"
m = json.loads(raw)
f = m["files"]
bad = [p for p, h in sorted(f.items()) if not os.path.isfile(os.path.join(r, p)) or hashlib.sha256(open(os.path.join(r, p), "rb").read()).hexdigest() != (h["sha256"] if isinstance(h, dict) else h)]
assert f and not bad, bad[:5]' "$BUNDLE" "$SERVICE/plantcell/_lib/mission/config/render_bundle.json" \
  || { echo "[setup] the render bundle at $BUNDLE is missing, altered or not the pinned one" >&2; exit 1; }
install -d -o plantcell -g plantcell -m 755 /run/plantcell
rm -f /run/plantcell/practice.sock
setsid runuser -u plantcell -- env HOME="$SERVICE" MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 LP_NUM_THREADS=8 SITE_GEN_DRESS=1 SITE_GEN_DRESS_BUNDLE="$BUNDLE" SITE_GEN_DRESS_SHADOWS=0 \
    OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \
    /bin/bash -c 'exec >> /opt/plantcell-practice/state/practiced.log 2>&1; while :; do /opt/venv/bin/python3 -s /opt/plantcell-practice/practiced.py; sleep 2; done' \
    < /dev/null > /dev/null 2>&1 &
for _ in $(seq 1 120); do [ -S /run/plantcell/practice.sock ] && break; sleep 0.5; done
[ -S /run/plantcell/practice.sock ] || { echo "[setup] the practice ground did not come up" >&2; exit 1; }
echo "[setup] practice ground listening on /run/plantcell/practice.sock" >&2
