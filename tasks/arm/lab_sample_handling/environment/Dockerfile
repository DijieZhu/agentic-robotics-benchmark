# Task image. Nothing here depends on tests/, solution/, the anchor or the hidden units. The hidden units (the rotor's
# hidden turn properties) and the hidden episodes are NOT in the image (tests/ and solution/oracle only). The simulator
# is in the image only inside the commissioning service's folder, which the agent's account cannot read (below); the
# agent gets the equipment's documents and the service's client through environment/setup/payload. The cameras' render
# bundle (the room the bench stands in; a public asset) is fetched into /opt/task-assets, which root and the service's
# account read and the agent's account does not.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONNOUSERSITE=1
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir --no-deps robosuite==1.5.2 \
    && rm -f /tmp/requirements.txt \
    && chmod -R a+rX /opt/venv

# Numerics. The simulation's operational-space controller calls LAPACK every physics step, and NumPy's OpenBLAS picks
# its kernels by CPU: on another kernel the simulation differs from the first step. OPENBLAS_CORETYPE=Haswell pins the
# kernels in this image and in the grading image; the .pth sets it for a /opt/venv Python that is started without the
# variable too (the grader starts run.sh with PATH, HOME and LANG only), and takes NumPy's AVX-512 code paths out on a
# CPU that has them, so that a development run here and the graded run compute the same on every machine. The grader
# checks it (tests/numerics_check.py) and refuses to grade on a machine that computes differently.
# OPENBLAS_L2_SIZE=256 guards the pin: the OpenBLAS bundled with numpy 2.5.2 to 2.5.5 (2.5.4 here) raises its block
# size on AMD Zen 4 / Zen 5 hosts even for a forced kernel, and the Haswell kernel can then crash or differ in the
# last bits for float64 products with an inner dimension above about 300 (none in the simulator; a controller's
# own products may be that large). It keeps the block at 256 on every host.
ENV OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256
RUN printf '%s\n' \
      'import os' \
      'os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")' \
      'os.environ.setdefault("OPENBLAS_L2_SIZE", "256")' \
      'try:' \
      '    with open("/proc/cpuinfo") as _f:' \
      '        _flags = set(_f.read().split())' \
      'except OSError:' \
      '    _flags = set()' \
      '_have = [n for n, f in (("AVX512F", "avx512f"), ("AVX512CD", "avx512cd"), ("AVX512_SKX", "avx512bw"), ("AVX512_CLX", "avx512_vnni"), ("AVX512_CNL", "avx512ifma"), ("AVX512_ICL", "avx512_vbmi2"), ("AVX512_SPR", "avx512_fp16")) if f in _flags]' \
      'if _have:' \
      '    os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", " ".join(_have))' \
      > /opt/venv/lib/python3.12/site-packages/lab_numerics_pin.py \
    && printf '%s\n' 'import lab_numerics_pin' > /opt/venv/lib/python3.12/site-packages/lab_numerics_pin.pth \
    && chmod 0644 /opt/venv/lib/python3.12/site-packages/lab_numerics_pin.py /opt/venv/lib/python3.12/site-packages/lab_numerics_pin.pth \
    && env -u OPENBLAS_CORETYPE -u OPENBLAS_L2_SIZE /opt/venv/bin/python3 -I -c "import os; assert os.environ.get('OPENBLAS_CORETYPE') == 'Haswell' and os.environ.get('OPENBLAS_L2_SIZE') == '256'; import numpy; print('[image] numerics pin OK')"

# Software rendering for MuJoCo: Mesa OSMesa (llvmpipe, no display or device needed), checked below as the agent user.
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa MPLBACKEND=Agg OMP_NUM_THREADS=1 LP_NUM_THREADS=2

# The bench cameras' renderer (lab_render.py beside lab_sim.py in the commissioning service's folder): MuJoCo's
# Filament module (in the mujoco wheel) on Vulkan; Mesa's lavapipe is the CPU Vulkan driver (the task has no GPU).
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md):
# render/ (the cameras' render bundle: room.npz, room.json, ibl.ktx; lab_render.py reads it in place). It is part of
# the bench's simulator, not of the agent's files: nothing copies it into the agent's home, and further down it is
# made readable by root and by the commissioning service's account only.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload
ENV LAB_RENDER_DIR=/opt/task-assets/payload/render

RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# The commissioning service: development benches the agent may work on, behind a socket. It runs as its own account
# from a folder that account alone can read: its code, the simulator (byte copies of tests/lab_sim.py, lab_env.py,
# box_transport_env.py, lab_render.py and lab_look.json), the development units and the ledger. The credit rule (tests/episode.py) is not in it.
# setup.sh starts it. The grading sandbox has its own image (tests/Dockerfile) without this folder.
RUN useradd --system --no-create-home --home-dir /opt/labbench-commission --shell /usr/sbin/nologin labbench
COPY commission_service /opt/labbench-commission
RUN rm -rf /opt/labbench-commission/state /opt/labbench-commission/__pycache__ \
    && mkdir -p /opt/labbench-commission/state \
    && chown -R labbench:labbench /opt/labbench-commission \
    && chmod -R u+rwX,go-rwx /opt/labbench-commission \
    && chown -R root:labbench /opt/task-assets && chmod -R u=rwX,g=rX,o= /opt/task-assets

# No job schedulers: nothing the agent leaves may run later as anyone.
RUN (apt-get purge -y cron anacron at >/dev/null 2>&1 || true) \
    && rm -rf /etc/cron.d /etc/cron.daily /etc/cron.hourly /var/spool/cron /var/spool/at \
    && ! command -v crond && ! command -v cron && ! command -v atd

RUN python3 -c "import numpy, scipy, cv2, mujoco; print('[image] mujoco', mujoco.__version__)" \
    && python3 -c "import importlib.util, os; s = importlib.util.find_spec('robosuite'); p = os.path.join(list(s.submodule_search_locations)[0], 'models', 'assets', 'robots', 'panda', 'robot.xml'); assert os.path.isfile(p), p" \
    && python3 -c "import site, sys; assert sys.prefix != sys.base_prefix" \
    && runuser -u user -- /opt/venv/bin/python3 -c "import numpy, scipy, cv2, mujoco; print('[image] agent-UID import OK')" \
    && runuser -u user -- env -i HOME=/tmp PATH=/opt/venv/bin:/usr/bin:/bin /opt/venv/bin/python3 -I -c "import os; assert os.environ.get('OPENBLAS_CORETYPE') == 'Haswell'" \
    && runuser -u user -- env HOME=/tmp /opt/venv/bin/python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>'); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m, 48, 64); r.update_scene(d); im = r.render(); assert im.max() > 0; print('[image] agent-UID MuJoCo OSMesa render OK', im.shape)"

# The service's account can run a bench (a reset, a step, the four cameras drawn by lab_render.py from the render
# bundle); the agent's account can neither list nor read the service's folder nor the render bundle, and nothing in
# the service's folder is open to a group or to others.
RUN install -d -o labbench -g labbench -m 700 /tmp/labbench-check \
    && runuser -u labbench -- env -i HOME=/tmp/labbench-check TMPDIR=/tmp/labbench-check \
      PATH=/opt/venv/bin:/usr/bin:/bin MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 LP_NUM_THREADS=2 \
      MESA_SHADER_CACHE_DISABLE=true PYTHONDONTWRITEBYTECODE=1 \
      /opt/venv/bin/python3 -B -s -c "import sys; sys.path.insert(0, '/opt/labbench-commission'); import json, lab_env as E; u = json.load(open('/opt/labbench-commission/bench.json'))['units']; env = E.LabEnv(u['d0'], 'd0', render=True); o = env.reset(1); o = env.step([0.0] * 6 + [-1.0, 15.0]); assert o['rack'].shape == (480, 640, 3); assert all(o[c].ndim == 3 and o[c].std() > 8 and 20 < o[c].mean() < 235 for c in ('left', 'right', 'wrist', 'rack')); env.close(); print('[image] the service account runs a bench and draws its four cameras')" \
    && rm -rf /tmp/labbench-check \
    && ! runuser -u user -- ls /opt/labbench-commission \
    && ! runuser -u user -- cat /opt/labbench-commission/bench.json \
    && ! runuser -u user -- cat /opt/labbench-commission/lab_sim.py \
    && ! runuser -u user -- ls /opt/task-assets \
    && ! runuser -u user -- cat /opt/task-assets/payload/render/room.json \
    && [ -z "$(find /opt/task-assets -perm /007)" ] \
    && [ -z "$(find /opt/labbench-commission ! -user labbench)" ] \
    && [ -z "$(find /opt/labbench-commission -perm /077)" ]

# The image's working directory is not the agent's home: the engine runs its own root commands from the image's
# working directory, which must not be a folder the agent can write to. The agent still starts in its home.
WORKDIR /
