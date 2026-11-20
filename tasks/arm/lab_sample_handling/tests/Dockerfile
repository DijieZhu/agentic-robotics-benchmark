# Verification image of lab_sample_handling (build context: tests/). The same stack as the agent's image
# (environment/Dockerfile: the same base, requirements, renderer and numerics pin), without a desktop session.
# Nothing hidden is copied into it: the engine stages tests/ at verify time, root-only. The cameras' render bundle (a
# public asset) is fetched into /opt/task-assets, root-only.
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
# variable too (the grader starts run.sh with PATH, HOME and LANG only), so that a development run here and the graded
# run compute the same on every machine. The grader checks it (tests/numerics_check.py) and refuses to grade on a
# machine that computes differently.
# OPENBLAS_L2_SIZE=256 guards the pin: the OpenBLAS bundled with numpy 2.5.2 to 2.5.5 (2.5.4 here) raises its block
# size on AMD Zen 4 / Zen 5 hosts even for a forced kernel, and the Haswell kernel can then crash or differ in the
# last bits for float64 products with an inner dimension above about 300 (none in the simulator; a controller's
# own products may be that large). It keeps the block at 256 on every host.
# The .pth pins a controller's own arithmetic as well, which the kernels alone do not (the same submission scored
# 8 of 12 episodes on an AVX2 host and 7 of 12 on an AVX-512 host: np.arctan2 differed in the last bit):
#   * NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" takes NumPy's AVX-512 loops out (arctan2, exp, log,
#     power, arcsin, cbrt and others run SVML there and round differently). These are the names of numpy 2.5's
#     dispatch groups: the names of numpy 1.x / 2.0-2.2 (AVX512F, AVX512_SKX, ...) are rejected by this numpy, with an
#     ImportWarning that nobody sees, and disable nothing. The build fails below if this numpy does not know a name;
#   * OPENBLAS_NUM_THREADS=1 and OMP_NUM_THREADS=1: a threaded product is split by the host's core count and sums
#     in another order;
#   * OPENCV_IPP=avx2: OpenCV's IPP picks its code by CPU (a float cubic resize and matchTemplate differ between an
#     AVX2 and an AVX-512 host), and OPENCV_CPU_DISABLE=AVX512-SKX on a CPU that has it (OpenCV prints a warning for
#     a feature the CPU lacks, so it is set only there).
# Each is a default: a variable that is already set is kept.
ENV OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256
RUN printf '%s\n' \
      'import os' \
      'os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")' \
      'os.environ.setdefault("OPENBLAS_L2_SIZE", "256")' \
      'os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "X86_V4 AVX512_ICL AVX512_SPR")' \
      'os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")' \
      'os.environ.setdefault("OMP_NUM_THREADS", "1")' \
      'os.environ.setdefault("OPENCV_IPP", "avx2")' \
      'try:' \
      '    with open("/proc/cpuinfo") as _f:' \
      '        _flags = set(_f.read().split())' \
      'except OSError:' \
      '    _flags = set()' \
      'if "avx512f" in _flags:' \
      '    os.environ.setdefault("OPENCV_CPU_DISABLE", "AVX512-SKX")' \
      > /opt/venv/lib/python3.12/site-packages/lab_numerics_pin.py \
    && printf '%s\n' 'import lab_numerics_pin' > /opt/venv/lib/python3.12/site-packages/lab_numerics_pin.pth \
    && chmod 0644 /opt/venv/lib/python3.12/site-packages/lab_numerics_pin.py /opt/venv/lib/python3.12/site-packages/lab_numerics_pin.pth \
    && env -i PATH=/opt/venv/bin:/usr/bin:/bin HOME=/tmp /opt/venv/bin/python3 -I -W error::ImportWarning -c "import os; e = os.environ; assert (e.get('OPENBLAS_CORETYPE'), e.get('OPENBLAS_L2_SIZE'), e.get('OPENBLAS_NUM_THREADS'), e.get('OMP_NUM_THREADS'), e.get('OPENCV_IPP')) == ('Haswell', '256', '1', '1', 'avx2'), dict(e); import numpy; from numpy._core._multiarray_umath import __cpu_dispatch__ as D, __cpu_features__ as F; assert set(e['NPY_DISABLE_CPU_FEATURES'].split()) == set(D) - {'X86_V3'}, D; assert [k for k in D if F[k]] in ([], ['X86_V3']), [k for k in D if F[k]]; print('[image] numerics pin OK', D)"

# Software rendering for MuJoCo: Mesa OSMesa (llvmpipe, no display or device needed), checked below as the agent user.
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa MPLBACKEND=Agg OMP_NUM_THREADS=1 LP_NUM_THREADS=2

# The bench cameras' renderer (tests/lab_render.py): MuJoCo's Filament module (in the mujoco wheel) on Vulkan; Mesa's
# lavapipe is the CPU Vulkan driver (the task has no GPU).
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md):
# render/ (the cameras' render bundle: room.npz, room.json, ibl.ktx; the same list as environment/assets.sha256).
# The grader (root) reads it in place; the controller's account cannot.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chown -R root:root /opt/task-assets && chmod -R u=rwX,go= /opt/task-assets
ENV LAB_RENDER_DIR=/opt/task-assets/payload/render

RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# No job schedulers: nothing may run later as anyone.
RUN (apt-get purge -y cron anacron at >/dev/null 2>&1 || true) \
    && rm -rf /etc/cron.d /etc/cron.daily /etc/cron.hourly /var/spool/cron /var/spool/at \
    && ! command -v crond && ! command -v cron && ! command -v atd

# No desktop in the grading sandbox. The base image's supervisord runs a desktop session (Xvnc, websockify, PulseAudio,
# GNOME) as the agent's account and restarts it when it dies. The controller runs as that account: a session that is
# started again after the grader's kill runs the account's autostart entries outside the launcher's no-network
# filter, and a live session can be reached from the controller over its D-Bus and X sockets. Grading draws nothing
# on a desktop (the cameras are rendered offscreen), so this image starts none: supervisord keeps running, as the
# container's command, with no program; DISPLAY is empty, which is how the engine tells that a sandbox has no desktop
# to wait for; /usr/local, which the base image leaves to the agent's account ahead of /usr/bin on root's PATH, is
# root's. verify.py checks at its start that no process of the account exists and none appears by itself, and during
# the episodes that every process of the account descends from its launcher.
RUN rm -f /etc/supervisor/conf.d/*.conf \
    && chown -R root:root /usr/local
ENV DISPLAY=
WORKDIR /

RUN python3 -c "import numpy, scipy, cv2, mujoco; print('[image] mujoco', mujoco.__version__)" \
    && python3 -c "import importlib.util, os; s = importlib.util.find_spec('robosuite'); p = os.path.join(list(s.submodule_search_locations)[0], 'models', 'assets', 'robots', 'panda', 'robot.xml'); assert os.path.isfile(p), p" \
    && runuser -u user -- env -i HOME=/tmp PATH=/opt/venv/bin:/usr/bin:/bin /opt/venv/bin/python3 -I -c "import os, numpy, scipy, cv2, mujoco; assert os.environ.get('OPENBLAS_CORETYPE') == 'Haswell'; print('[image] agent-UID import OK')" \
    && runuser -u user -- env HOME=/tmp /opt/venv/bin/python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>'); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m, 48, 64); r.update_scene(d); im = r.render(); assert im.max() > 0; print('[image] agent-UID MuJoCo OSMesa render OK', im.shape)" \
    && [ -z "$(ls /etc/supervisor/conf.d 2>/dev/null)" ] && [ "$(stat -c %U /usr/local/bin)" = root ]

# The grader's account (root) draws the four cameras of a bench through lab_render.py from the render bundle (the
# simulator's files are mounted from the build context for this step only: nothing of them stays in the image; the
# unit is a maintainer's, inside the documented ranges); the controller's account cannot list or read the bundle.
RUN --mount=type=bind,source=.,target=/tmp/lab-draw-check,readonly \
    env -i HOME=/root PATH=/opt/venv/bin:/usr/bin:/bin MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 LP_NUM_THREADS=2 \
      MESA_SHADER_CACHE_DISABLE=true PYTHONDONTWRITEBYTECODE=1 \
      /opt/venv/bin/python3 -B -s -c "import sys; sys.path.insert(0, '/tmp/lab-draw-check'); import lab_env as E; env = E.LabEnv({'rotor_friction': 0.21, 'handle_play': 7.0, 'rotor_inertia': 0.019}, 'image', render=True); o = env.reset(1); o = env.step([0.0] * 6 + [-1.0, 15.0]); assert o['rack'].shape == (480, 640, 3); assert all(o[c].ndim == 3 and o[c].std() > 8 and 20 < o[c].mean() < 235 for c in ('left', 'right', 'wrist', 'rack')); env.close(); print('[image] the grader draws the four cameras')" \
    && ! runuser -u user -- ls /opt/task-assets \
    && ! runuser -u user -- cat /opt/task-assets/payload/render/room.json \
    && [ -z "$(find /opt/task-assets -perm /077)" ]
