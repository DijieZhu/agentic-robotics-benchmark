#!/usr/bin/env bash
# Verify stage: runs as root, cwd = this staged folder, after the agent phase, in a SEPARATE verifier sandbox (task.toml
# [verifier] environment_mode = "separate"): a fresh sandbox from the task image that the agent never touched, holding
# only the declared artifact (/home/user/submission, restored root-owned); environment/setup/setup.sh never runs here.
# The engine uploads this folder with modes 0644 / 0755, so the FIRST thing done here -- before the interpreter starts --
# is to close the folder and its parent to every other account (one chmod each; verify.py records the original modes
# from the two variables and restores them at the end, then kills the agent account's leftover processes, snapshots the
# submission, gates late writes and locks every file; in a separate verifier the agent left no process and wrote
# nothing, so these only stop what the image's own session of the agent account does).
#
# The grader runs as root with nothing of the agent home on its import or configuration paths: HOME and the XDG /
# matplotlib folders point at root's own, PYTHONNOUSERSITE=1 and `-s` keep any user site-packages out, and the cell
# renders every sensor through MuJoCo's classic renderer on OSMesa (the practice ground's cell uses the same path and
# settings). Version 2: every thread count and the render bundle are exported here, so that grading does not depend on
# the image's variables (environment/Dockerfile sets the same values for the agent's sandbox): 8 rasteriser threads
# (frames byte-equal to one thread's, measured in the task image; the grader renders one mission at a time, and 8
# threads made its frames cheaper there), BLAS 2 as in version 1 (numpy, scipy;
# the anchor's numerics), OpenMP 2 (nothing of the cell uses it; measured without effect on the pace).
set -euo pipefail
STAGE="$(pwd -P)"
export ARB_STAGE_MODE="$(stat -c %a "$STAGE")"
chmod 700 "$STAGE"
PARENT="$(dirname "$STAGE")"
if [ "$PARENT" != "/" ] && [ "$PARENT" != "${ALE_HOME:-/home/user}" ]; then
  export ARB_STAGE_PARENT_MODE="$(stat -c %a "$PARENT")"
  chmod 700 "$PARENT"
fi
# The grader's root-owned, read-only copy of the givens (/opt/arb): verify.py checks it file by file against
# given_hashes.json and hands its client library to the programs, so nothing the agent wrote is an input of the grader.
# It holds what environment/setup/setup.sh stages into the agent home: setup_payload/, this folder's byte copy of
# environment/setup/payload/ (which never reaches the verifier; change both together, then given_hashes.json).
ROOT_PAYLOAD="${ARB_ROOT_PAYLOAD:-/opt/arb}"
rm -rf "$ROOT_PAYLOAD"
mkdir -p "$ROOT_PAYLOAD"
cp -r "$STAGE/setup_payload/." "$ROOT_PAYLOAD/"
chown -R root:root "$ROOT_PAYLOAD"
chmod -R u+rwX,go+rX,go-w "$ROOT_PAYLOAD"
export HOME=/root PYTHONNOUSERSITE=1 XDG_CONFIG_HOME=/root/.config XDG_CACHE_HOME=/root/.cache \
  MPLCONFIGDIR=/root/.config/matplotlib
export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  MKL_NUM_THREADS=2
# the cell's render bundle (the buildings' look, in the image readable by root and the practice ground's account
# only): required, and checked here before anything else runs: its manifest's sha256 against the pin this folder holds
# (plantcell/_lib/mission/config/render_bundle.json, the same file the cell checks at every mission), then
# every file against the manifest; a missing, altered or other bundle is the image's fault (a task error), never a
# mission without it. The dressing's shadow switch is pinned too (amendment 1): shadows stay off, as they
# are when the variable is unset (the frames the admission and the reference value were measured on), whatever a
# caller's environment says. The image's ENV no longer names these variables (amendment 1).
export SITE_GEN_DRESS=1 SITE_GEN_DRESS_BUNDLE=/opt/task-assets/look SITE_GEN_DRESS_SHADOWS=0
"${ALE_VERIFY_PYTHON:-python3}" -I -s -c 'import hashlib, json, os, sys
r = os.environ["SITE_GEN_DRESS_BUNDLE"]
pin_file = os.path.join(os.getcwd(), "plantcell", "_lib", "mission", "config", "render_bundle.json")
try:
    want = json.load(open(pin_file))["manifest_sha256"]
    raw = open(os.path.join(r, "manifest.json"), "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    m = json.loads(raw)
    f = m["files"]
    bad = [p for p, h in sorted(f.items()) if not os.path.isfile(os.path.join(r, p)) or hashlib.sha256(open(os.path.join(r, p), "rb").read()).hexdigest() != (h["sha256"] if isinstance(h, dict) else h)]
except Exception as e:
    sys.exit("[verify] the render bundle at %s or its pin is unusable (%s): the image is broken" % (r, e))
if got != want:
    sys.exit("[verify] the render bundle at %s: manifest sha256 %s is not the pinned %s: the image is broken" % (r, got, want))
if not f or bad:
    sys.exit("[verify] the render bundle at %s: %d file(s) missing or altered %s: the image is broken" % (r, len(bad), bad[:5]))
print("[verify] render bundle %s: manifest as pinned, %d files verified" % (m.get("name") or m.get("bundle"), len(f)), file=sys.stderr)'
# the cell's and the reference program's numerics the same on every x86-64 CPU: OpenBLAS otherwise picks its kernels
# per CPU and numpy its SIMD paths; the program under test keeps its own minimal environment (cellkit.runner.child_env)
export OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
exec "${ALE_VERIFY_PYTHON:-python3}" -s verify.py
