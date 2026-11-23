FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && chmod -R a+rX /opt/venv
# Arithmetic pin. A mission is a long closed loop, so a last-bit difference in a program's own arithmetic between
# two CPUs grows into another trajectory (measured on a sibling task: the same submission scored differently on an
# AVX2 and an AVX-512 host). The .pth sets, for every /opt/venv Python however it is started (the grader starts
# run.sh with a minimal environment), NumPy without its AVX-512 loops (the names are this NumPy's own dispatch
# groups; the build fails if it does not know one), one BLAS / OpenMP thread and one OpenBLAS kernel and block size. Each is a default: a variable that
# is already set is kept. The grader checks the result (tests/numerics_check.py) and refuses to grade on a machine
# that computes differently.
RUN SITE=$(/opt/venv/bin/python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
    && printf '%s\n' \
      'import os' \
      'os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "AVX512F AVX512CD AVX512_KNL AVX512_KNM AVX512_SKX AVX512_CLX AVX512_CNL AVX512_ICL")' \
      'os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")' \
      'os.environ.setdefault("OMP_NUM_THREADS", "1")' \
      'os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")' \
      'os.environ.setdefault("OPENBLAS_L2_SIZE", "256")' \
      > "$SITE/numerics_pin.py" \
    && printf '%s\n' 'import numerics_pin' > "$SITE/numerics_pin.pth" \
    && chmod 0644 "$SITE/numerics_pin.py" "$SITE/numerics_pin.pth" \
    && env -i PATH=/opt/venv/bin:/usr/bin:/bin HOME=/tmp /opt/venv/bin/python3 -I -W error::ImportWarning -c "import os; e = os.environ; assert (e.get('OPENBLAS_NUM_THREADS'), e.get('OMP_NUM_THREADS'), e.get('OPENBLAS_CORETYPE')) == ('1', '1', 'Haswell'), dict(e); import numpy; from numpy._core._multiarray_umath import __cpu_dispatch__ as D, __cpu_features__ as F; off = set(e['NPY_DISABLE_CPU_FEATURES'].split()); assert off == {k for k in D if k.startswith('AVX512')}, D; assert not [k for k in D if F[k] and k in off]; print('[image] arithmetic pin OK')"
RUN mkdir -p /home/user/submission && chown user:user /home/user/submission
RUN runuser -u user -- python3 -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco/>'); r=mujoco.Renderer(m,64,64); r.close()"
# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md).
# They are the robot models and textures the service builds its models with (MuJoCo Menagerie's Go1 and Crazyflie,
# Poly Haven textures and three baked models; sources and licences: visual_assets/provenance.json and the LICENSE.*
# files) and the render bundle of its two cameras (visual_assets/hall_look/: the hall, its dressing, the robots'
# meshes, one texture atlas and the image light; hall_look/README.md).
# The agent's account never reads them: the step moves them into a root-only folder and keeps no other copy.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && mkdir -p /opt/visual_robotics \
    && mv /opt/task-assets/payload/visual_assets /opt/visual_robotics/assets \
    && rm -rf /opt/task-assets/payload \
    && chown -R root:root /opt/visual_robotics/assets && chmod 0700 /opt/visual_robotics/assets \
    && test -f /opt/visual_robotics/assets/provenance.json && test -f /opt/visual_robotics/assets/hall_look/scene.npz \
    && ! runuser -u user -- ls /opt/visual_robotics/assets >/dev/null 2>&1
