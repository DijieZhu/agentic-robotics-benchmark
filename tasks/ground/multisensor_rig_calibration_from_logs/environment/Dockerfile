# The practice rigs (logs only: no truth, no held-out data) are generated once, offline, and fetched here as
# one pinned archive; the generator (tests/gen/) never enters the agent's image, and the hidden rigs never enter
# any image (the verify stage fetches their logs; tests/verify.py).
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv -r /tmp/requirements.txt \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# Numerics pin (docs/contributing-a-task.md section 5): the graded program's arithmetic repeats on
# every grading host. OpenBLAS on its Haswell kernels with numpy 2.5's block size, one thread,
# NumPy's AVX-512 loops off, OpenCV's IPP on AVX2. A .pth sets each as a default, so a Python
# started with a scrubbed environment (the graded run) gets them too.
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
      > /opt/venv/lib/python3.12/site-packages/rigcal_numerics_pin.py \
    && printf '%s\n' 'import rigcal_numerics_pin' > /opt/venv/lib/python3.12/site-packages/rigcal_numerics_pin.pth \
    && chmod 0644 /opt/venv/lib/python3.12/site-packages/rigcal_numerics_pin.py /opt/venv/lib/python3.12/site-packages/rigcal_numerics_pin.pth \
    && env -i PATH=/opt/venv/bin:/usr/bin:/bin HOME=/tmp /opt/venv/bin/python3 -I -W error::ImportWarning -c "import os; e = os.environ; assert (e.get('OPENBLAS_CORETYPE'), e.get('OPENBLAS_L2_SIZE'), e.get('OPENBLAS_NUM_THREADS'), e.get('OMP_NUM_THREADS'), e.get('OPENCV_IPP')) == ('Haswell', '256', '1', '1', 'avx2'), dict(e); import numpy; from numpy._core._multiarray_umath import __cpu_dispatch__ as D, __cpu_features__ as F; assert set(e['NPY_DISABLE_CPU_FEATURES'].split()) == set(D) - {'X86_V3'}, D; assert [k for k in D if F[k]] in ([], ['X86_V3']), [k for k in D if F[k]]; print('[image] numerics pin OK', D)"

# The practice rigs. fetch_assets.py is the benchmark's canonical copy (shared/task_assets/) and the repository named
# in assets.sha256 is written from the benchmark's task-assets.toml (scripts/task_assets.py sync; docs/task-assets.md).
# One deterministic tar (about 4,300 files inside: one request instead of thousands).
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && mkdir -p /opt/rigcal \
    && tar -xf /opt/task-assets/payload/practice_rigs.tar -C /opt/rigcal \
    && rm -rf /opt/task-assets/payload \
    && test -f /opt/rigcal/practice/rig_a/manifest.json && test ! -e /opt/rigcal/practice_truth \
    && chown -R root:root /opt/rigcal && chmod -R u=rwX,go=rX /opt/rigcal

ENV OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
ENV MPLBACKEND=Agg

# The declared artifact path, owned by the agent.
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# No scheduler may start agent code after the agent phase.
RUN (apt-get purge -y cron anacron at >/dev/null 2>&1 || true) \
    && rm -rf /etc/cron.d /etc/cron.daily /etc/cron.hourly /var/spool/cron /var/spool/at

# The practice data is read-only for the agent.
RUN ! runuser -u user -- sh -c ': >> /opt/rigcal/practice/rig_a/manifest.json' 2>/dev/null \
    && python3 -c "import numpy, scipy, cv2, yaml, matplotlib; print('[image] numpy', numpy.__version__, 'scipy', scipy.__version__, 'cv2', cv2.__version__)" \
    && runuser -u user -- /opt/venv/bin/python3 -c "import numpy, scipy, cv2, yaml; print('[image] agent-UID import OK')"
