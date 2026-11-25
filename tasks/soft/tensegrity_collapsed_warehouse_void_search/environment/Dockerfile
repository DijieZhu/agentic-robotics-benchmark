# tensegrity_collapsed_warehouse_void_search's environment image. `environment/` (less its `setup/`) is the sole build
# context and the final stage derives DIRECTLY from the official ALE base, at an exact version. Nothing hidden is here:
# the givens (the documents, the client library, the example and the practice launcher) are small text files and
# pictures staged by environment/setup/setup.sh from environment/setup/payload/; the grader, the anchor, the hidden
# sites, their plate bundles and the reference program live in tests/ and solution/ (absent from the image). The image
# does hold the practice ground (the station on the team's calibration site and three practice sites, for the agent's
# practice runs) with the plate bundles of those four sites, readable by its own account only; the bundles come from
# the benchmark's asset dataset, fetched at the commit pinned in assets.sha256 and checked file by file
# (docs/task-assets.md). Precedent: quadruped_hazardous_plant_inspection's image (the same base, interpreter, package
# pins, practice-ground account pattern and proofs). This task's cell renders the search camera's colour image with the
# plate backend (the site's pre-rendered environment plate, with the robot drawn by MuJoCo's Filament renderer on Mesa's
# lavapipe Vulkan driver) and its depth image from the site's depth plate and the robot's depth (MuJoCo's classic
# renderer on OSMesa), and carries its own decimated robot meshes inside the practice ground's (and the grader's) copy
# of the cell.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (no GPU in the task's resources; EGL enumerates no device in the sandbox): the robot's
# depth and mask in the search camera's frames render with MuJoCo's classic renderer on OSMesa (llvmpipe); the GL
# client libraries MuJoCo links even when headless; libgomp for numpy / scipy threads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# The search camera's colour image (the plate backend, tests/voidcell/_lib/plate_filament.py): the robot rendered by
# MuJoCo's Filament renderer (in the mujoco wheel) on Vulkan; Mesa's lavapipe is the CPU Vulkan driver (no GPU in the
# task's resources). The depth stays on OSMesa.
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`), the practice ground
# (`voidcell`), the scored episodes' runner (`nobody`) and verify (root) all use the same interpreter.
# UV_PYTHON_INSTALL_DIR must be world-readable too (the default /root/.local/share/uv is untraversable for the other
# accounts). Python 3.12: the engine stages `ale_verify` into the `python3` it finds and refuses anything below 3.12.
# The base image sets HOME=/home/user for root too, so the build points uv's cache away from the agent home, which must
# not hold root-owned entries.
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

# `uv venv` leaves its lock file /opt/venv/.lock world-writable (mode 666, which `chmod -R a+rX` keeps): a program could
# carry what it saw in one scored episode into the next through it (as found for the precedent). Writable by root only;
# the last proof below finds no world-writable regular file.
RUN [ ! -e /opt/venv/.lock ] || chmod 0644 /opt/venv/.lock

# Filament on Vulkan (lavapipe) starts; DISPLAY (the base image's :1) is dropped as the backend drops it.
RUN env -u DISPLAY python3 -c "import mujoco._render_filament as F, mujoco.experimental.studio.window; c = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN)); print('[void] Filament on Vulkan OK')"

# The practice ground's own account (the service below is readable by it only).
RUN useradd --system --no-create-home --home-dir /opt/voidcell-practice --shell /usr/sbin/nologin voidcell

# The declared artifact path must exist even when nobody works (the engine captures declared artifacts after every
# episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# The practice ground: the station on the team's calibration site and three practice sites (practice_ground/: the
# service program practiced.py, its stored site packages and allowance sites.json, and the cell library voidcell/, a
# byte copy of the grader's cell plus its scoring and debrief, without the reference program and without any token),
# installed for its own account `voidcell` only: mode 700, so the agent account can neither read nor list it.
# environment/setup/setup.sh starts it before the agent's session (and restarts it should it ever exit); the agent
# reaches it through the unix socket /run/voidcell/practice.sock and gets the station's public calls, reset(site) and
# each episode's debrief.
COPY practice_ground /opt/voidcell-practice

# The practice ground's plate bundles (the plate backend's offline environment of the calibration site and the three
# practice sites; not in git): fetched at the dataset commit pinned in assets.sha256 and checked file by file
# (docs/task-assets.md), installed into the practice ground's own plates/ under the names the cell looks for
# (voidcell/_lib/plate_install.py, the same code as the grader's: each neutral folder renamed to the site package
# sha256 its meta.json records, checked against sites.json); the download and the list's copy are removed in the same
# step, so nothing stays in /opt/task-assets (whose payload/ setup.sh would copy into the agent home). The next step
# gives plates/ to the practice ground's account only, like the rest of its folder.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /tmp/plates-fetch \
    && /opt/venv/bin/python3 -B -s /opt/voidcell-practice/voidcell/_lib/plate_install.py practice /tmp/plates-fetch/plates \
       /opt/voidcell-practice/plates /opt/voidcell-practice/sites.json \
    && rm -rf /tmp/plates-fetch /opt/task-assets
RUN find /opt/voidcell-practice -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && rm -rf /opt/voidcell-practice/state \
    && mkdir -p /opt/voidcell-practice/state \
    && chown -R voidcell:voidcell /opt/voidcell-practice \
    && chmod -R u+rwX,go-rwx /opt/voidcell-practice

# Headless defaults for the agent's shell: every render goes through OSMesa and never writes Mesa's shader cache;
# OMP_NUM_THREADS=8 makes GNU `nproc` print the sandbox's 8 CPUs (it prints OMP_NUM_THREADS when that is set, the
# host's count otherwise: the sandbox is limited by a CPU quota, which this coreutils does not read); numpy and scipy
# keep OPENBLAS_NUM_THREADS=2. Neither the practice ground nor the verify stage depends on these:
# environment/setup/setup.sh and tests/test.sh export their own values (the same ones for both).
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV MESA_SHADER_CACHE_DISABLE=true
ENV OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 LP_NUM_THREADS=8

# Fail the build early if the toolchain is broken; prove the AGENT account can use the venv and import the client
# library (a temporary copy of the practice ground's client.py, the same bytes as the given one); prove the practice
# ground's cell builds the calibration site from its package and renders a camera frame for its own account (colour
# through the plate backend from the site's installed plate bundle, depth on OSMesa); `nproc` prints 8 for the agent
# account; the agent account can neither list nor read the practice ground or its plate bundles, no plate file is in
# the agent home or elsewhere under /opt, and nothing in the agent home is root-owned; the image holds no world-writable
# regular file. One import line
# per -c: a continuation line starting with the keyword of a Dockerfile base-image line reads as one to the final-stage
# lint.
RUN export HOME=/root \
    && test "$(runuser -u user -- nproc)" = 8 && echo "[void] nproc for the agent account: 8" \
    && python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; import mujoco, numpy, scipy, PIL, cv2, yaml; print('[void] import OK mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'cv2', cv2.__version__)" \
    && runuser -u user -- python3 -c "import mujoco, numpy, scipy, PIL, cv2, yaml; print('[void] agent-UID import OK')" \
    && mkdir -p /tmp/client-proof/voidcell \
    && cp /opt/voidcell-practice/voidcell/client.py /tmp/client-proof/voidcell/client.py \
    && touch /tmp/client-proof/voidcell/__init__.py \
    && chmod -R a+rX /tmp/client-proof \
    && runuser -u user -- python3 -B -E -s -c "import sys; sys.path.insert(0, '/tmp/client-proof'); import voidcell.client as c; assert callable(c.launcher_argv) and callable(c.decode_telemetry); print('[void] agent-UID client import OK')" \
    && rm -rf /tmp/client-proof \
    && runuser -u voidcell -- env HOME=/opt/voidcell-practice TMPDIR=/opt/voidcell-practice/state MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa MESA_SHADER_CACHE_DISABLE=true OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=2 LP_NUM_THREADS=8 OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 /opt/venv/bin/python3 -B -s -c "import sys, json, os; sys.path.insert(0, '/opt/voidcell-practice'); import voidcell.server as S; doc = json.load(open('/opt/voidcell-practice/sites.json')); site = doc['sites'][0]; cfg = dict(doc.get('config') or {}); cfg.update({'plate_root': os.path.join('/opt/voidcell-practice', cfg['plate_root'])} if cfg.get('plate_root') else {}); s = S.VoidCellServer(config=cfg, mode='practice'); s.reset(os.path.join('/opt/voidcell-practice', site['package']), seed=1, kind=site.get('kind', 'practice'), sim_time_limit=site.get('time_limit_s'), name=site['name']); f = s.handle('get_frame', {}); assert isinstance(f, dict) and 'error' not in f, f; s.close(); print('[void] practice-ground cell OK: site', site['name'], 'built, frame', sorted(f)[:6])" \
    && rm -rf /opt/voidcell-practice/state && mkdir -p /opt/voidcell-practice/state \
    && chown voidcell:voidcell /opt/voidcell-practice/state && chmod 700 /opt/voidcell-practice/state \
    && test -z "$(find /opt/voidcell-practice -name '__pycache__' -print -quit)" \
    && ! runuser -u user -- ls /opt/voidcell-practice > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/voidcell-practice/sites.json > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/voidcell-practice/voidcell/server.py > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/voidcell-practice/cal/cal.tar.xz > /dev/null 2>&1 \
    && ! runuser -u user -- ls /opt/voidcell-practice/plates > /dev/null 2>&1 \
    && test -z "$(find /home/user /opt -path /opt/voidcell-practice -prune -o -name plate_ldr.png -print -quit)" \
    && echo "[void] the agent account can neither list nor read the practice ground" \
    && test -z "$(find /home/user -user root -print -quit)" \
    && ! runuser -u user -- test -w /opt/venv/.lock \
    && test -z "$(find / -xdev -type f -perm -0002 -print -quit)" \
    && echo "[void] no world-writable regular file in the image (the venv's lock file included)"
