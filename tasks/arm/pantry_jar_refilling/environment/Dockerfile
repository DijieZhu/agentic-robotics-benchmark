# Task image. Nothing here depends on tests/, solution/, the anchor or the hidden units: an image layer is readable by
# anyone who can run the image. The hidden units (the jars' hidden properties) are NOT in the image (tests/ only);
# the simulator and the practice units reach the agent through environment/setup/payload.
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
# its kernels by CPU: on another kernel the simulation differs from the first step, and a controller then runs another
# trajectory (measured: one reference episode took 5024-5190 control steps under four kernels, one practice episode
# scored differently). OPENBLAS_CORETYPE=Haswell pins the kernels in this image, which is also the grading image;
# the .pth sets it for a /opt/venv Python that is started without the variable too (the grader starts run.sh with
# PATH, HOME and LANG only), and takes NumPy's AVX-512 code paths out on a CPU that has them, so that a development
# run here and the graded run compute the same on every machine. The grader checks it (tests/numerics_check.py) and
# refuses to grade on a machine that computes differently.
# OPENBLAS_L2_SIZE=256 guards the pin: the OpenBLAS bundled with numpy 2.5.2 to 2.5.5 (2.5.4 here) raises its block
# size on AMD Zen 4 / Zen 5 hosts even for a forced kernel, and the Haswell kernel can then crash or differ in the
# last bits for float64 products with an inner dimension above about 300 (none in the simulator; a controller's own
# products may be that large). It keeps the block at 256 on every host.
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
      > /opt/venv/lib/python3.12/site-packages/jar_numerics_pin.py \
    && printf '%s\n' 'import jar_numerics_pin' > /opt/venv/lib/python3.12/site-packages/jar_numerics_pin.pth \
    && chmod 0644 /opt/venv/lib/python3.12/site-packages/jar_numerics_pin.py /opt/venv/lib/python3.12/site-packages/jar_numerics_pin.pth \
    && env -u OPENBLAS_CORETYPE -u OPENBLAS_L2_SIZE /opt/venv/bin/python3 -I -c "import os; assert os.environ.get('OPENBLAS_CORETYPE') == 'Haswell' and os.environ.get('OPENBLAS_L2_SIZE') == '256'; import numpy; print('[image] numerics pin OK')"

# Software rendering for MuJoCo: Mesa OSMesa (llvmpipe, no display or device needed), checked below as the agent user.
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa MPLBACKEND=Agg OMP_NUM_THREADS=1 LP_NUM_THREADS=2

RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# No job schedulers: nothing the agent leaves may run later as anyone.
RUN (apt-get purge -y cron anacron at >/dev/null 2>&1 || true) \
    && rm -rf /etc/cron.d /etc/cron.daily /etc/cron.hourly /var/spool/cron /var/spool/at \
    && ! command -v crond && ! command -v cron && ! command -v atd

RUN python3 -c "import numpy, scipy, cv2, mujoco; print('[image] mujoco', mujoco.__version__)" \
    && python3 -c "import importlib.util, os; s = importlib.util.find_spec('robosuite'); p = os.path.join(list(s.submodule_search_locations)[0], 'models', 'assets', 'robots', 'panda', 'robot.xml'); assert os.path.isfile(p), p" \
    && python3 -c "import site, sys; assert sys.prefix != sys.base_prefix" \
    && runuser -u user -- /opt/venv/bin/python3 -c "import numpy, scipy, cv2, mujoco; print('[image] agent-UID import OK')" \
    && runuser -u user -- env HOME=/tmp /opt/venv/bin/python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>'); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m, 48, 64); r.update_scene(d); im = r.render(); assert im.max() > 0; print('[image] agent-UID MuJoCo OSMesa render OK', im.shape)"

# The image's working directory is not the agent's home: the engine starts the agent in its home whatever this says,
# and a root process the engine starts in the working directory must not find agent-written files there.
WORKDIR /
