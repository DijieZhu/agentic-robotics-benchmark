# freeflyer_uncrewed_station_caretaking's environment image. `environment/` (less its `setup/`) is the sole build
# context and the image derives DIRECTLY from the official ALE base, at an exact version. Nothing hidden is here: the
# givens are staged by environment/setup/setup.sh; the grader, its thresholds, the hidden sites and the reference
# program live in tests/ and solution/ (absent from the image). The image does hold the rehearsal facility (the robot
# cell on the training module, for the agent's rehearsal runs), readable by its own account only, and the cell's
# model files (the task assets), readable by root and that account only.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (no GPU in the task's resources): every camera of the cell renders with MuJoCo's
# classic renderer on OSMesa (llvmpipe); the GL client libraries MuJoCo links even when headless; libgomp for numpy /
# scipy threads. util-linux's runuser, prlimit and choom (the grader's runner) are in the base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`), the rehearsal facility
# (`rehearsal`), the scored programs (`nobody`) and verify (root) all use the same interpreter. UV_PYTHON_INSTALL_DIR
# must be world-readable too. Python 3.12: the engine stages `ale_verify` into the `python3` it finds and refuses
# anything below 3.12. The base image sets HOME=/home/user for root too, so the build points uv's cache away from the
# agent home, which must not hold root-owned entries.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN export HOME=/root UV_CACHE_DIR=/tmp/uv-cache \
    && uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements.txt \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; } \
    && rm -rf /tmp/uv-cache /tmp/requirements.txt
# `uv venv` leaves its lock file /opt/venv/.lock world-writable (mode 666): a program could carry what it saw on one
# scored site into the next through it. Writable by root only.
RUN [ ! -e /opt/venv/.lock ] || chmod 0644 /opt/venv/.lock

# The rehearsal facility's own account (the service and the task assets below are readable by it).
RUN useradd --system --no-create-home --home-dir /opt/rehearsal --shell /usr/sbin/nologin rehearsal

# ---- task assets (docs/task-assets.md): every file listed with its sha256 in assets.sha256, under the task's folder of
# the benchmark's asset dataset. cell/: the cell's three model archives (robot: the robot's collision and light visual
# model with the dock; station: the station modules' look; robot_author: the robot's visible meshes as its makers
# published them), each unpacked to /opt/task-assets/<component>/ -- the asset root CABINCELL_ASSET_ROOT names
# (tests/test.sh exports it for the grader, environment/setup/setup.sh for the rehearsal facility; not in this image's
# ENV) -- and not kept as an archive, so the image holds one copy, which the grader and the rehearsal facility both
# read and check file by file against the pins cabincell/data/<component>/ASSETS.json (verify.py; the cell's
# assetroot.py at every build). given/: the agent-visible pictures (the robot's and the station's figures, the item
# catalogue's photographs, the map views), kept at /opt/task-assets/payload/given/, staged into the agent home by
# setup.sh and into the grader's copy of the givens by tests/test.sh (tests/given_hashes.json pins them).
# Owned by root, group rehearsal, modes 750 / 640: the agent's account can neither list nor read any of it (a proof
# below) and nobody but root can write it.
# Where the files come from (build argument CARETAKE_ASSETS): `hub` = fetch_assets.py (the benchmark's canonical copy,
# shared/task_assets/) from the asset dataset at the commit pinned in assets.sha256; `local` = the same files placed in
# the build context at local_assets/<path in assets.sha256> (a build that has the files at hand;
# never in git); `auto` (the default) = local when every listed file is there, else hub. Either way every
# file's sha256 must equal the pinned one, else the build fails.
ARG CARETAKE_ASSETS=auto
COPY fetch_assets.py assets.sha256 local_assets* /opt/task-assets/build/
RUN set -eu; B=/opt/task-assets/build; R=/opt/task-assets; mode="$CARETAKE_ASSETS"; \
    grep -E '^[0-9a-f]{64}  [^/]' "$B/assets.sha256" > "$B/list.sha256"; \
    n=$(wc -l < "$B/list.sha256"); \
    if [ "$mode" = auto ]; then mode=local; \
      while read -r _h p; do [ -f "$B/$p" ] || { mode=hub; break; }; done < "$B/list.sha256"; fi; \
    case "$mode" in \
      local) (cd "$B" && sha256sum --strict --quiet -c list.sha256); SRC="$B" ;; \
      hub) python3 -I "$B/fetch_assets.py" "$B/assets.sha256" "$B/hub"; SRC="$B/hub" ;; \
      *) echo "[assets] CARETAKE_ASSETS must be auto, local or hub, not $mode" >&2; exit 1 ;; \
    esac; \
    for a in robot_model_v1_2 station_look robot_author; do ls "$SRC"/cell/"$a"*.tar.xz > /dev/null; done; \
    for a in "$SRC"/cell/*.tar.xz; do tar -xJf "$a" -C "$R" --no-same-owner --no-same-permissions; done; \
    mkdir -p "$R/payload"; cp -r "$SRC/given" "$R/payload/given"; \
    rm -rf "$B"; \
    echo "[assets] $n file(s) checked (source: $mode): $(ls "$R" | tr '\n' ' ')"; \
    chown -R root:rehearsal "$R"; chmod -R u=rwX,g=rX,o= "$R"

# The declared artifact path must exist even when nobody works (the engine captures declared artifacts after every
# episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# The rehearsal facility: the robot cell on the training module L3 (practice_ground/: the service practiced.py, its
# cards and debriefs, the station map's copy and cabincell/, a byte copy of the grader's cell without the grading
# module), installed for its own account `rehearsal` only: mode 700, so the agent
# account can neither read nor list it. environment/setup/setup.sh starts it before the agent's session (and restarts
# it should it ever exit); the agent reaches it through the unix socket /run/rehearsal/rehearsal.sock
# (~/rehearsal/rehearse.py): the robot's public calls, reset(card) and each run's observation sheet.
COPY practice_ground /opt/rehearsal
RUN find /opt/rehearsal -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && rm -rf /opt/rehearsal/state \
    && mkdir -p /opt/rehearsal/state \
    && chown -R rehearsal:rehearsal /opt/rehearsal \
    && chmod -R u+rwX,go-rwx /opt/rehearsal

# Headless defaults of the agent's shell: every render goes through OSMesa; OMP_NUM_THREADS=8 makes GNU `nproc` print
# the sandbox's 8 CPUs (it prints OMP_NUM_THREADS when that is set, the host's count otherwise); numpy and scipy keep 2
# BLAS threads. The verify stage and the rehearsal facility do not depend on these: tests/test.sh and setup.sh export
# their own values.
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 LP_NUM_THREADS=8

# Fail the build early if the toolchain is broken; prove the AGENT account can use the venv (the given client library
# is staged by setup.sh, outside this build context; the container check imports it as the agent); prove the rehearsal
# facility's account reads the three asset components and that the facility's cell finds them in the asset root and
# checks every file against its pins (assetroot.py of the practice copy), builds a training-module site of card EX-A
# with the frame renderer and renders a colour frame; that it refuses with the assets' own error (AssetError) an asset
# root without the components; `nproc` prints 8 for the agent account; the agent account can neither list nor read the
# task assets nor the rehearsal facility, and can write nothing of them; nothing in the agent home is root-owned; the
# agent account's environment names no CABINCELL_ or REHEARSAL_ variable; the image holds no world-writable regular
# file. Before the rehearsal cell is built, a BLAS self-test under the cell's numeric pins (one thread, the forced kernel
# with OPENBLAS_L2_SIZE=256, as tests/test.sh and setup.sh set them): the pin must be complete, and three fresh child
# processes run level-3 products of every dtype at the block size the forced kernel cannot take without it, a solve and
# an SVD, with core dumps off. One import line per -c: a continuation line starting with that word reads as a Dockerfile
# instruction to the final-stage lint.
RUN export HOME=/root \
    && test "$(runuser -u user -- nproc)" = 8 && echo "[caretake] nproc for the agent account: 8" \
    && python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; import mujoco, numpy, scipy, PIL, cv2; print('[caretake] import OK python', sys.version.split()[0], 'mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'cv2', cv2.__version__, 'PIL', PIL.__version__)" \
    && runuser -u user -- python3 -c "import mujoco, numpy, scipy, PIL, cv2; print('[caretake] agent-UID import OK')" \
    && runuser -u rehearsal -- env CABINCELL_ASSET_ROOT=/opt/task-assets /opt/venv/bin/python3 -I -B /opt/rehearsal/cabincell/assetroot.py robot station robot_author > /tmp/assets-proof.json \
    && python3 -I -c "import json, sys; d = json.load(open('/tmp/assets-proof.json')); assert len(d) == 3 and all(e.get('source') == 'env' and e.get('dir', '').startswith('/opt/task-assets/') for e in d), d; print('[caretake] the rehearsal account finds and checks the three asset components:', ', '.join('%s %s %d files' % (e['component'], e['archive'], e['files']) for e in d))" \
    && rm -f /tmp/assets-proof.json \
    && NEG="$(runuser -u rehearsal -- env CABINCELL_ASSET_ROOT=/opt/rehearsal/state /opt/venv/bin/python3 -I -B /opt/rehearsal/cabincell/assetroot.py robot 2>&1 || true)" \
    && printf '%s\n' "$NEG" | grep -q 'has no robot/ directory' \
    && echo "[caretake] the cell refuses an asset root without its components (AssetError)" \
    && env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512_ICL /opt/venv/bin/python3 -I -B -c "import os,subprocess,sys; c=os.environ.get('OPENBLAS_CORETYPE'); c and os.environ.get('OPENBLAS_L2_SIZE','1024')=='1024' and sys.exit('BLAS self-test FAILED: OPENBLAS_CORETYPE without OPENBLAS_L2_SIZE'); p='import ctypes;ctypes.CDLL(None).prctl(4,0,0,0,0);import numpy as n;[(n.ones((64,512),t)@n.ones((512,64),t)==512).all() or exit(3) for t in (n.float64,n.float32,n.complex128,n.complex64)];a=n.random.default_rng(0).random((700,700))+30*n.eye(700);n.linalg.solve(a,n.ones(700));n.linalg.svd(a[:400])'; r=[subprocess.run([sys.executable,'-I','-B','-c',p]).returncode for _ in range(3)]; sys.exit('BLAS self-test FAILED, child exit codes %s (negative = signal)' % r) if any(r) else print('[caretake] BLAS self-test OK under the cell pins')" \
    && install -d -o rehearsal -g rehearsal -m 700 /tmp/rh-proof \
    && runuser -u rehearsal -- env HOME=/opt/rehearsal CABINCELL_ASSET_ROOT=/opt/task-assets REHEARSAL_STATE=/tmp/rh-proof OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 LP_NUM_THREADS=4 OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512_ICL /opt/venv/bin/python3 -B -s -c "import sys; sys.path.insert(0, '/opt/rehearsal'); import practiced as P; srv, info = P.open_cell('EX-A', 60.0, 0); r = srv.handle('camera', {'name': 'FIC'}); srv.close(); assert isinstance(r, dict) and r.get('code') == 'OK', str(r)[:300]; print('[caretake] rehearsal cell OK: card EX-A built, colour frame', sorted(r)[:6])" \
    && rm -rf /tmp/rh-proof \
    && ! runuser -u user -- ls /opt/task-assets > /dev/null 2>&1 \
    && ! runuser -u user -- ls /opt/task-assets/robot > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/task-assets/robot/mjcf/freeflyer_v1.xml > /dev/null 2>&1 \
    && ! runuser -u user -- ls /opt/task-assets/payload/given > /dev/null 2>&1 \
    && runuser -u rehearsal -- test -r /opt/task-assets/robot/mjcf/freeflyer_v1.xml \
    && test -z "$(runuser -u user -- find /opt/task-assets /opt/rehearsal -writable -print -quit 2>/dev/null)" \
    && echo "[caretake] the agent account can neither list nor read the task assets; the rehearsal account reads them" \
    && ! runuser -u user -- ls /opt/rehearsal > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/rehearsal/practiced.py > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/rehearsal/cabincell/data/robot/spec_values.json > /dev/null 2>&1 \
    && echo "[caretake] the agent account can neither list nor read the rehearsal facility" \
    && test -z "$(find /home/user -user root -print -quit)" \
    && ! runuser -u user -- env | grep -q -E '^(CABINCELL_|REHEARSAL_)' \
    && echo "[caretake] the agent account's environment names no CABINCELL_ or REHEARSAL_ variable" \
    && ! runuser -u user -- test -w /opt/venv/.lock \
    && test -z "$(find / -xdev -type f -perm -0002 -print -quit)" \
    && echo "[caretake] no world-writable regular file in the image (the venv's lock file included)"
