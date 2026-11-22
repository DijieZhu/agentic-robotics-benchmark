# earthquake_search_and_rescue's environment image. `environment/` (less its `setup/`) is the sole build context and the
# final stage derives DIRECTLY from the official ALE base, at an exact version. Nothing hidden is here: the givens are
# staged by environment/setup/setup.sh; the grader, the anchor, the hidden sites and the reference program live in
# tests/ and solution/ (absent from the image). The image does hold the training ground (the robot cell on the team's
# training sites, for the agent's development missions), readable by its own account only.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (no GPU in the task's resources; EGL enumerates no device in the sandbox), the GL
# client libraries MuJoCo links even when headless, libgomp for numpy/scipy threads. The cameras' colour images
# (rubblecell/look.py): MuJoCo's Filament renderer (in the mujoco wheel) on Vulkan, with Mesa's lavapipe as the CPU
# Vulkan driver; depth, lidar and thermal stay on OSMesa.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
        mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`), the training ground
# (`rubblecell`) and verify (root) all use the same interpreter. UV_PYTHON_INSTALL_DIR must be world-readable too (the
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

# The declared artifact path must exist even when nobody works (the engine captures declared artifacts after every
# episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# The training ground: the robot cell on the team's training sites (training_ground/: the service program, its sites
# and allowance, and the cell library rubblecell/, a byte copy of the grader's cell without the reference program),
# installed for its own account `rubblecell` only: mode 700, so the agent account can neither read nor list it.
# environment/setup/setup.sh starts it before the agent's session (and restarts it should it ever exit); the agent
# reaches it through a unix socket and gets the robot's public calls, reset(site) and each mission's log (the robot's
# own: no truth of the site, no score).
RUN useradd --system --no-create-home --home-dir /opt/rubblecell-training --shell /usr/sbin/nologin rubblecell
COPY training_ground /opt/rubblecell-training
RUN find /opt/rubblecell-training -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && rm -rf /opt/rubblecell-training/state \
    && mkdir -p /opt/rubblecell-training/state \
    && chown -R rubblecell:rubblecell /opt/rubblecell-training \
    && chmod -R u+rwX,go-rwx /opt/rubblecell-training

# The numerics pin of a mission program (setup/payload/training/submission_checks.py `numerics_pin`, which the grader
# and the training launcher set for the program they start), here for the agent's own shell and scripts too, so that
# what it computes while developing is what the graded program computes: fixed OpenBLAS kernels on one thread, NumPy
# without its AVX-512 loops (the names of numpy 2.5's dispatch groups; the build fails below if this numpy rejects
# one), OpenCV's IPP at AVX2. The training ground's cell and the grader's set their own environment.
ENV OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" OPENCV_IPP=avx2
# Set before the look's textures are made, so that they are computed under the pin on whatever machine builds the image.

# The cell's look pack (docs/task-assets.md: fetched at the dataset commit pinned in assets.sha256 and checked by its
# sha256; one tar): the photographed texture sets, the sky, the posed bodies, the robot's drawn parts, the vehicles and
# the models of the site (furniture, window modules, ruin pieces, the rescue post, apartment blocks). Then the look's
# own texture sets (rubblecell/pbr.py: procedural, from fixed seeds, the same pixels on every machine) are made beside
# them. All of it is for the cell's two users, the grader (root, in its separate verifier sandbox from this image) and
# the training ground (`rubblecell`); the agent account cannot read it (it is no given), and no layer keeps the tar.
ENV RUBBLECELL_LOOK_DIR=/opt/rubblecell-look
COPY fetch_assets.py assets.sha256 /tmp/task-assets/
RUN export HOME=/root \
    && python3 /tmp/task-assets/fetch_assets.py /tmp/task-assets/assets.sha256 /tmp/task-assets/files \
    && mkdir -p /opt/rubblecell-look \
    && tar -x -C /opt/rubblecell-look -f /tmp/task-assets/files/look/rubblecell_look_pack.tar \
    && rm -rf /tmp/task-assets \
    && cd /opt/rubblecell-training && python3 -B -s -m rubblecell.pbr /opt/rubblecell-look \
    && test -s /opt/rubblecell-look/props.npz && test -s /opt/rubblecell-look/vehicles.npz && test -s /opt/rubblecell-look/robot.npz \
    && chown -R root:rubblecell /opt/rubblecell-look \
    && chmod -R u+rwX,g+rX,o-rwx /opt/rubblecell-look

# Headless defaults: every render of the cell goes through OSMesa.
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV LP_NUM_THREADS=2
# Fail the build early if the toolchain is broken, prove the AGENT account can use the venv, prove a Filament context
# on lavapipe, prove the training ground's cell builds a site and renders a frame (Filament colour, OSMesa depth) for
# its own account; finally, the agent account can read neither the training ground nor the look's textures, and
# nothing in the agent home is root-owned. One import line per -c.
RUN export HOME=/root \
    && python3 -W error::ImportWarning -c "import numpy; print('[rubble] numerics pin: numpy', numpy.__version__, 'accepts NPY_DISABLE_CPU_FEATURES')" \
    && python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; import mujoco, numpy, scipy, PIL, cv2; print('[rubble] import OK mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__)" \
    && runuser -u user -- python3 -c "import mujoco, numpy, scipy, PIL, cv2; print('[rubble] agent-UID import OK')" \
    && env -u DISPLAY python3 -c "import mujoco._render_filament as F, mujoco.experimental.studio.window; c = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN)); print('[rubble] Filament on Vulkan (lavapipe) OK')" \
    && runuser -u rubblecell -- env HOME=/opt/rubblecell-training /opt/venv/bin/python3 -B -s -c "import sys; sys.path.insert(0, '/opt/rubblecell-training'); from rubblecell.server import RubbleCellServer; s = RubbleCellServer(mode='hidden'); s.reset(7, 'lanes'); f = s.get_camera('mast'); print('[rubble] training-ground cell OK: lanes built, mast frame', f.get('width'), 'x', f.get('height'))" \
    && ! runuser -u user -- ls /opt/rubblecell-training > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/rubblecell-training/sites.json > /dev/null 2>&1 \
    && ! runuser -u user -- ls /opt/rubblecell-look > /dev/null 2>&1 \
    && test -z "$(find /home/user -user root -print -quit)"

# The engine's commands start in the image's working directory: keep it out of the agent's home (a root-owned place the
# agent account cannot write), so that nothing the agent leaves there is picked up by a command run as root.
WORKDIR /
