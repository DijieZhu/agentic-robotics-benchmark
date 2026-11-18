# quadruped_hazardous_plant_inspection's environment image. `environment/` (less its `setup/`) is the sole build
# context and the final stage derives DIRECTLY from the official ALE base, at an exact version. Nothing hidden is here:
# the givens are staged by environment/setup/setup.sh; the grader, the anchor, the hidden sites and the reference
# program live in tests/ and solution/ (absent from the image). The image does hold the practice ground (the robot
# cell on the team's training hall and training sites, for the agent's practice missions), readable by its own account
# only.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (no GPU in the task's resources; EGL enumerates no device in the sandbox): every sensor
# of the cell renders with MuJoCo's classic renderer on OSMesa (llvmpipe); the GL client libraries MuJoCo links even
# when headless; libgomp for numpy / scipy threads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`), the practice ground
# (`plantcell`) and verify (root) all use the same interpreter. UV_PYTHON_INSTALL_DIR must be world-readable too (the
# default /root/.local/share/uv is untraversable for the other accounts). Python 3.12: the engine stages `ale_verify`
# into the `python3` it finds and refuses anything below 3.12. The base image sets HOME=/home/user for root too, so the
# build points uv's cache away from the agent home, which must not hold root-owned entries.
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

# Amendment 1: `uv venv` leaves its lock file /opt/venv/.lock world-writable (mode 666, which `chmod -R
# a+rX` keeps), the only such file in the image: a program could carry what it saw in one scored mission into the
# next through it. Writable by root only. A step of its own, so that the venv's layer above stays the one the
# admission and the reference value were measured with; the last proof below finds no world-writable regular file.
RUN [ ! -e /opt/venv/.lock ] || chmod 0644 /opt/venv/.lock

# The practice ground's own account (the service and the render bundle below are readable by it).
RUN useradd --system --no-create-home --home-dir /opt/plantcell-practice --shell /usr/sbin/nologin plantcell

# ---- task assets: the cell's render bundle (version 2: the buildings' look; albedo maps, meshes and a
# manifest with each file's sha256, made by the maintainers' bundle tool from the licensed art packs).
# One tar listed with its sha256 in assets.sha256, unpacked to /opt/task-assets/<its name>, the folder
# SITE_GEN_DRESS_BUNDLE names (set by environment/setup/setup.sh for the practice ground and by tests/test.sh for the
# grader, not in this image's ENV: amendment 1), where the cell's loader reads it; the cell checks the
# manifest's sha256 against its pin (tests/plantcell/_lib/mission/config/render_bundle.json, also in the practice
# ground's copy) and every file against the manifest at every mission (SITE_GEN_DRESS=1: a missing, altered or other
# bundle stops the cell, never an undressed frame). Owned by root, group plantcell, modes 750 / 640: root (setup, the
# grader's and the oracle's cell) and the practice ground's account read it, the agent's account can neither list nor
# read it (a proof below), and nobody but root can write it. Nothing of it is a given.
# Where the tar comes from (build argument QINSPECT_ASSETS): `hub` = fetch_assets.py (the benchmark's canonical copy,
# shared/task_assets/) from the asset dataset at the commit pinned in assets.sha256; `local` = the same file placed in
# the build context at local_assets/<its path in assets.sha256> (builds on the maintainers' machines without a
# download; never in git); `auto` (the default) = local when that file is there, else hub. Either way its sha256
# must equal the pinned one, else the build fails.
ARG QINSPECT_ASSETS=auto
COPY fetch_assets.py stage_assets.py assets.sha256 local_assets* /opt/task-assets/build/
RUN python3 -I /opt/task-assets/build/stage_assets.py /opt/task-assets/build/assets.sha256 /opt/task-assets/build \
        /opt/task-assets "$QINSPECT_ASSETS" \
    && rm -rf /opt/task-assets/build \
    && chown -R root:plantcell /opt/task-assets \
    && chmod -R u=rwX,g=rX,o= /opt/task-assets

# The declared artifact path must exist even when nobody works (the engine captures declared artifacts after every
# episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# The practice ground: the robot cell on the team's training hall and training sites (practice_ground/: the service
# program, its stored site packages and allowance, and the cell library plantcell/, a byte copy of the grader's cell
# without the reference program), installed for its own account `plantcell` only: mode 700, so the agent account can
# neither read nor list it. environment/setup/setup.sh starts it before the agent's session (and restarts it should it
# ever exit); the agent reaches it through a unix socket and gets the robot's public calls, reset(site) and each
# mission's debrief.
COPY practice_ground /opt/plantcell-practice
RUN find /opt/plantcell-practice -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && rm -rf /opt/plantcell-practice/state \
    && mkdir -p /opt/plantcell-practice/state \
    && chown -R plantcell:plantcell /opt/plantcell-practice \
    && chmod -R u+rwX,go-rwx /opt/plantcell-practice

# Headless defaults: every render of the cell goes through OSMesa. Version 2: the
# rasteriser uses 8 threads (the frames are byte-equal to one thread's, and 15-40 % cheaper); OMP_NUM_THREADS=8 makes
# GNU `nproc` print the sandbox's 8 CPUs (it prints OMP_NUM_THREADS when that is set, the host's count otherwise: the
# sandbox is limited by a CPU quota, which this coreutils does not read); numpy and scipy keep OPENBLAS_NUM_THREADS=2.
# The verify stage does not depend on these: tests/test.sh exports its own values.
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 LP_NUM_THREADS=8

# Fail the build early if the toolchain is broken; prove the AGENT account can use the venv and import the client
# library (a temporary copy of the practice ground's client.py, the same bytes as the given one); prove the practice
# ground's cell builds a training site from its package and renders a navigation frame (colour and depth, OSMesa) for
# its own account, with the render bundle (SITE_GEN_DRESS=1), and refuses with the bundle's own error (BundleError,
# not just any failure) both without it (a bundle path that does not exist) and with a manifest that is not the pinned
# one (a copy of the bundle whose manifest gained one byte); `nproc` prints 8 for the agent account; the agent account
# can neither list nor read the render bundle and can write nothing of it; the agent account can neither list nor read
# the practice ground, and nothing in the agent home is root-owned; finally (amendment 1), the agent account's
# environment names no SITE_GEN_ variable (the proofs pass the dressing's variables explicitly, as
# setup.sh and tests/test.sh do) and the image holds no world-writable regular file. One import line per
# -c: a continuation line starting with `from` reads as a Dockerfile FROM to the final-stage lint.
RUN export HOME=/root \
    && test "$(runuser -u user -- nproc)" = 8 && echo "[plant] nproc for the agent account: 8" \
    && test -z "$(runuser -u user -- find /opt/task-assets -writable -print -quit)" \
    && python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; import mujoco, numpy, scipy, PIL, cv2, yaml; print('[plant] import OK mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__)" \
    && runuser -u user -- python3 -c "import mujoco, numpy, scipy, PIL, cv2, yaml; print('[plant] agent-UID import OK')" \
    && mkdir -p /tmp/client-proof/plantcell \
    && cp /opt/plantcell-practice/plantcell/client.py /tmp/client-proof/plantcell/client.py \
    && touch /tmp/client-proof/plantcell/__init__.py \
    && chmod -R a+rX /tmp/client-proof \
    && runuser -u user -- python3 -B -E -s -c "import sys; sys.path.insert(0, '/tmp/client-proof'); import plantcell.client as c; assert callable(c.launcher_argv) and callable(c.connect_stdio); print('[plant] agent-UID client import OK')" \
    && rm -rf /tmp/client-proof \
    && runuser -u plantcell -- env HOME=/opt/plantcell-practice SITE_GEN_DRESS=1 SITE_GEN_DRESS_BUNDLE=/opt/task-assets/look SITE_GEN_DRESS_SHADOWS=0 /opt/venv/bin/python3 -B -s -c "import sys, json, os; sys.path.insert(0, '/opt/plantcell-practice'); import plantcell.server as S; doc = json.load(open('/opt/plantcell-practice/sites.json')); site = doc['sites'][0]; s = S.PlantCellServer(config=doc.get('config'), mode='practice'); s.reset(os.path.join('/opt/plantcell-practice', site['package']), seed=1, kind='practice'); f = s.handle('get_camera', {'name': 'nav', 'depth': True}); assert isinstance(f, dict) and 'error' not in f, f; s.close(); print('[plant] practice-ground cell OK: site', site['name'], 'built, nav frame', sorted(f)[:6])" \
    && NEG="$(runuser -u plantcell -- env HOME=/opt/plantcell-practice SITE_GEN_DRESS=1 SITE_GEN_DRESS_BUNDLE=/opt/task-assets/absent /opt/venv/bin/python3 -B -s -c "import sys, json, os; sys.path.insert(0, '/opt/plantcell-practice'); import plantcell.server as S; doc = json.load(open('/opt/plantcell-practice/sites.json')); site = doc['sites'][0]; s = S.PlantCellServer(config=doc.get('config'), mode='practice'); s.reset(os.path.join('/opt/plantcell-practice', site['package']), seed=1, kind='practice'); print('BUILT')" 2>&1 || true)" \
    && printf '%s\n' "$NEG" | grep -q 'BundleError: the render bundle is required' && ! printf '%s\n' "$NEG" | grep -q '^BUILT' \
    && echo "[plant] practice-ground cell refuses to build without its render bundle (BundleError)" \
    && mkdir -p /tmp/pin-proof && cp -r /opt/task-assets/look/. /tmp/pin-proof/ && printf ' ' >> /tmp/pin-proof/manifest.json \
    && chown -R plantcell:plantcell /tmp/pin-proof \
    && NEG="$(runuser -u plantcell -- env HOME=/opt/plantcell-practice SITE_GEN_DRESS=1 SITE_GEN_DRESS_BUNDLE=/tmp/pin-proof /opt/venv/bin/python3 -B -s -c "import sys, json, os; sys.path.insert(0, '/opt/plantcell-practice'); import plantcell.server as S; doc = json.load(open('/opt/plantcell-practice/sites.json')); site = doc['sites'][0]; s = S.PlantCellServer(config=doc.get('config'), mode='practice'); s.reset(os.path.join('/opt/plantcell-practice', site['package']), seed=1, kind='practice'); print('BUILT')" 2>&1 || true)" \
    && rm -rf /tmp/pin-proof \
    && printf '%s\n' "$NEG" | grep -q 'BundleError: render bundle /tmp/pin-proof: its manifest' && ! printf '%s\n' "$NEG" | grep -q '^BUILT' \
    && echo "[plant] practice-ground cell refuses a bundle whose manifest is not the pinned one (BundleError)" \
    && ! runuser -u user -- ls /opt/task-assets > /dev/null 2>&1 \
    && ! runuser -u user -- ls /opt/task-assets/look/ > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/task-assets/look/manifest.json > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/task-assets/qinspect_look_v1/manifest.json > /dev/null 2>&1 \
    && runuser -u plantcell -- test -r /opt/task-assets/look/manifest.json \
    && echo "[plant] the agent account can neither list nor read the render bundle; the practice ground's account reads it" \
    && ! runuser -u user -- ls /opt/plantcell-practice > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/plantcell-practice/sites.json > /dev/null 2>&1 \
    && test -z "$(find /home/user -user root -print -quit)" \
    && ! runuser -u user -- env | grep -q '^SITE_GEN_' \
    && echo "[plant] the agent account's environment names no SITE_GEN_ variable" \
    && ! runuser -u user -- test -w /opt/venv/.lock \
    && test -z "$(find / -xdev -type f -perm -0002 -print -quit)" \
    && echo "[plant] no world-writable regular file in the image (the venv's lock file included)"
