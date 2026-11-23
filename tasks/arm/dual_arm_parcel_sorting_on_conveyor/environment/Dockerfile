# dual_arm_parcel_sorting_on_conveyor's environment image. `environment/` (less its `setup/`) is the sole build context
# and the final stage derives DIRECTLY from the official ALE base, at an exact version. Nothing hidden is here: the
# givens are staged by environment/setup/setup.sh; the grader, the anchor, the hidden shifts and the reference program
# live in tests/ and solution/ (absent from the image). The image does hold the commissioning service (the station's
# cell on its development shifts, for the agent's commissioning runs), readable by its own account only.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (no GPU in the task's resources; EGL enumerates no device in the sandbox): the cell's
# cameras render with MuJoCo's classic renderer on OSMesa (llvmpipe); the GL client libraries MuJoCo links even when
# headless; libgomp for numpy / scipy threads; Mesa's lavapipe (the CPU Vulkan driver) for MuJoCo's Filament renderer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
        mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`), the commissioning service
# (`parcelcell`), the program under test (`nobody`) and verify (root) all use the same interpreter.
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
# carry what it saw in one graded shift into the next through it. Writable by root only (the last proof below finds no
# world-writable regular file).
RUN [ ! -e /opt/venv/.lock ] || chmod 0644 /opt/venv/.lock

# The declared artifact path must exist even when nobody works (the engine captures declared artifacts after every
# episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# The commissioning service's account (the service itself is installed below, after the task assets).
RUN useradd --system --no-create-home --home-dir /opt/parcelcell-commission --shell /usr/sbin/nologin parcelcell

# ---- task assets ---------------------------------------------------------------------------------------------------
# Two groups live in the benchmark's Hugging Face asset repository (tasks/arm/dual_arm_parcel_sorting_on_conveyor/payload/
# there), fetched at the commit pinned in assets.sha256 and checked file by file against the sha256 recorded there
# (fetch_assets.py is the benchmark's canonical copy; scripts/task_assets.py writes both files, never by hand):
#   render/parcelcell_render_v3.tar  the agent cameras' render bundle (the station's surroundings, the robot's drawn
#       parts, the image light and the depth twin's near meshes; one tar), unpacked root-owned to
#       /opt/task-assets/render, readable by the grader (root) and the commissioning service's account only (the agent's
#       account and the program under test cannot read it), then checked file by file against the bundle's manifest;
#       the tar itself is removed;
#   given/robot/                     the robot's mesh files (the dual-arm platform's links under meshes/, the gripper's
#       under gripper_2f85/assets/ and flattened beside the links for dual_arm_with_grippers.xml) and the gripper's
#       licence file, world-readable at /opt/task-assets/payload/given: environment/setup/setup.sh stages them into the
#       agent home next to environment/setup/payload/ (the model XML files there refer to them), tests/test.sh copies
#       them into the grader's root-owned copy of the givens and into its cell's assets, and the steps below copy them
#       into the build proofs' robot model and into the commissioning service's cell. The same bytes everywhere
#       (tests/given_hashes.json pins them at their paths in the agent home).
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN set -eu; \
    python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload; \
    mkdir -p /opt/task-assets/render; \
    tar -xf /opt/task-assets/payload/render/parcelcell_render_v3.tar -C /opt/task-assets/render --no-same-owner; \
    rm -rf /opt/task-assets/payload/render; \
    chown -R root:parcelcell /opt/task-assets/render; \
    chmod -R u+rwX,g+rX,g-w,o-rwx /opt/task-assets/render; \
    chown -R root:root /opt/task-assets/payload; \
    chmod -R u+rwX,go+rX,go-w /opt/task-assets/payload; \
    cd /opt/task-assets/render; \
    python3 -c "import hashlib, json; m = json.load(open('manifest.json')); bad = [p for p, h in m['files'].items() if hashlib.sha256(open(p, 'rb').read()).hexdigest() != h]; assert not bad, bad; print('[image] render bundle:', len(m['files']), 'files verified')"; \
    echo "[image] robot payload: $(find /opt/task-assets/payload/given -type f | wc -l) files"

# The robot model: the dual-arm platform (arm/: its MJCF, generic names, and the link meshes) and the two-finger parallel
# gripper (gripper_2f85/: MuJoCo Menagerie's robotiq_2f85, BSD-2-Clause, LICENSE and SOURCE.txt inside), the same bytes
# the cell builds the robot from (the model files from commission_service/parcelcell/assets/, a byte copy of
# tests/parcelcell/assets/; the meshes from the fetched payload), at /opt/task-assets/robot, readable by root only (it is
# the cell's own model with its true dynamics; the agent's model files in its home carry the maker's nominal data): the
# build proofs below compile and render it.
COPY commission_service/parcelcell/assets/robot /opt/task-assets/robot/arm
COPY commission_service/parcelcell/assets/gripper_2f85 /opt/task-assets/robot/gripper_2f85
RUN set -eu; \
    mkdir -p /opt/task-assets/robot/arm/meshes /opt/task-assets/robot/gripper_2f85/assets; \
    cp /opt/task-assets/payload/given/robot/meshes/*.STL /opt/task-assets/robot/arm/meshes/; \
    cp /opt/task-assets/payload/given/robot/gripper_2f85/assets/*.stl /opt/task-assets/robot/gripper_2f85/assets/; \
    chown -R root:root /opt/task-assets/robot; \
    chmod -R u+rwX,go-rwx /opt/task-assets/robot; \
    echo "[image] robot model: $(find /opt/task-assets/robot -type f | wc -l) files"

# The commissioning service: the station's cell on its development shifts, the line an automation engineer commissions
# a robot program on before it runs unattended (commission_service/: the service program, its development shifts and
# window, the grader's wire loop cellkit/wire.py and the cell library parcelcell/, byte copies of the grader's; the
# cell's mesh files from the fetched payload, as the grader's cell gets them), installed for its own account
# `parcelcell` only: mode 700, so the agent account can neither read nor list it. environment/setup/setup.sh starts it
# before the agent's session (and restarts it should it ever exit); the agent reaches it through a unix socket and gets
# the cell's public calls and reset(site): no ground truth, no score.
COPY commission_service /opt/parcelcell-commission
RUN find /opt/parcelcell-commission -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && rm -rf /opt/parcelcell-commission/state \
    && mkdir -p /opt/parcelcell-commission/state /opt/parcelcell-commission/parcelcell/assets/robot/meshes \
       /opt/parcelcell-commission/parcelcell/assets/gripper_2f85/assets \
    && cp /opt/task-assets/payload/given/robot/meshes/*.STL /opt/parcelcell-commission/parcelcell/assets/robot/meshes/ \
    && cp /opt/task-assets/payload/given/robot/gripper_2f85/assets/*.stl /opt/parcelcell-commission/parcelcell/assets/gripper_2f85/assets/ \
    && chown -R parcelcell:parcelcell /opt/parcelcell-commission \
    && chmod -R u+rwX,go-rwx /opt/parcelcell-commission

# Headless defaults: every classic render goes through OSMesa. OMP_NUM_THREADS=8 makes GNU `nproc` print the sandbox's
# 8 CPUs (it prints OMP_NUM_THREADS when that is set, the host's count otherwise: the sandbox is limited by a CPU quota,
# which this coreutils does not read); numpy and scipy keep OPENBLAS_NUM_THREADS=2; the rasteriser 8 threads. The
# verify stage does not depend on these: tests/test.sh exports its own values; the program under test gets the
# grader's minimal environment (BLAS, OpenMP and rasteriser threads 2).
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 LP_NUM_THREADS=8

# Fail the build early if the toolchain is broken; prove an offscreen OSMesa render of the robot model; prove the AGENT
# account can use the venv and compile the gripper model; prove a Filament context on Vulkan (lavapipe); make the
# development shifts' parcel looks once, as the commissioning service's account, into its look cache (so that a
# development shift starts in seconds), and prove that account builds a development shift from that cache and renders a
# head frame with depth (OSMesa, the service's numerics); the render bundle is readable by that account and not by the
# agent's;
# `nproc` prints 8 for the agent account; the agent account can neither list nor read the service; nothing in the
# agent home is root-owned; the image holds no world-writable regular file. One import line per -c: a continuation
# line starting with `from` reads as a Dockerfile FROM to the final-stage lint.
RUN export HOME=/root \
    && cd /tmp \
    && test "$(runuser -u user -- nproc)" = 8 && echo "[parcel] nproc for the agent account: 8" \
    && python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; import mujoco, numpy, scipy, PIL, cv2; print('[parcel] import OK mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__)" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_path('/opt/task-assets/robot/arm/dual_arm.xml'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); print('[parcel] OSMesa render OK', px.shape, int(px.max()), 'robot nbody', m.nbody)" \
    && runuser -u user -- python3 -c "import mujoco, numpy, scipy, PIL, cv2; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=\'0.1\'/></body></worldbody></mujoco>'); print('[parcel] agent-UID import OK, a model compiles, nbody', m.nbody)" \
    && ! runuser -u user -- cat /opt/task-assets/robot/arm/dual_arm.xml > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/task-assets/robot/gripper_2f85/2f85.xml > /dev/null 2>&1 \
    && echo "[parcel] the cell's own robot model is not readable by the agent account" \
    && env -u DISPLAY python3 -c "import mujoco._render_filament as F, mujoco.experimental.studio.window; c = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN)); print('[parcel] Filament on Vulkan (lavapipe) OK')" \
    && runuser -u parcelcell -- env HOME=/opt/parcelcell-commission LP_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" /opt/venv/bin/python3 -B -s -c "import sys, json, os, glob; sys.path.insert(0, '/opt/parcelcell-commission'); from parcelcell import kitcache; d = json.load(open('/opt/parcelcell-commission/sites.json')); r = kitcache.prebuild([os.path.join('/opt/parcelcell-commission', s['package']) for s in d['sites']], '/opt/parcelcell-commission/kitcache', workers=4); assert r['missing'] == 0, r; print('[parcel] development shifts looks prebuilt:', r)" \
    && runuser -u parcelcell -- env HOME=/opt/parcelcell-commission LP_NUM_THREADS=8 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" PARCELCELL_KIT_CACHE=/opt/parcelcell-commission/kitcache /opt/venv/bin/python3 -B -s -c "import sys, json, os, time; sys.path.insert(0, '/opt/parcelcell-commission'); import parcelcell.server as S; d = json.load(open('/opt/parcelcell-commission/sites.json')); s = S.ParcelCellServer(config=d['config'], mode='hidden', privileged=False); t0 = time.time(); s.reset(os.path.join('/opt/parcelcell-commission', d['sites'][0]['package'])); tb = time.time() - t0; f = s.handle('get_camera', {'name': 'head', 'depth': True}); assert isinstance(f, dict) and 'error' not in f, f; st = s.handle('step', {'n': 5}); assert st['ticks'] == 5, st; vis = s.world.visuals.describe(); assert vis['cache']['misses'] == 0, vis; s.close(); print('[parcel] commissioning cell OK for its account: development shift 1 built in %.1f s from the look cache, head frame' % tb, f['width'], 'x', f['height'], 'with depth, 5 ticks')" \
    && runuser -u parcelcell -- test -r /opt/task-assets/render/manifest.json && ! runuser -u user -- test -r /opt/task-assets/render/manifest.json \
    && echo "[parcel] the render bundle is readable by root and the service account, not by the agent account" \
    && ! runuser -u user -- ls /opt/parcelcell-commission > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/parcelcell-commission/sites.json > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/parcelcell-commission/shifts/dev_01.json > /dev/null 2>&1 \
    && echo "[parcel] the agent account can neither list nor read the commissioning service" \
    && test -z "$(find /home/user -user root -print -quit)" \
    && ! runuser -u user -- test -w /opt/venv/.lock \
    && test -z "$(find / -xdev -type f -perm -0002 -print -quit)" \
    && echo "[parcel] no root-owned entry in the agent home, no world-writable regular file in the image"
