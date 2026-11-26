# Task image. Nothing here depends on tests/, solution/ or the anchor: an image layer is readable by anyone who can
# run the image. The recordings were rendered once by environment/gen/render_clips.py from the real fabrics' model
# (hidden: tests/creal.py and environment/gen/, neither of which is in this image) and are fetched from the
# benchmark's asset repository at the commit and checksums pinned in assets.sha256 (docs/task-assets.md).
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv -r /tmp/requirements.txt \
    && uv pip install --python /opt/venv --no-deps robosuite==1.5.2 \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# Software rendering for MuJoCo (Mesa OSMesa, no display or GPU) and ffmpeg (video decoding and encoding for the agent).
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ---- numerics: the same arithmetic on every host, here and in the grading run (tests/numerics_check.py).
# The real fabric's model calls np.exp once per physics step, and NumPy's AVX-512 loops round it differently from its
# other loops: without the pin every real outcome differed between an AVX2 and an AVX-512 host (measured: 36 of 36
# outcomes, up to 0.25 mm), with NPY_DISABLE_CPU_FEATURES they agreed to the bit. The names are numpy 2.5's; a name
# this numpy does not know is rejected with an ImportWarning and disables nothing, so the build fails on that warning.
# The OpenBLAS kernel pin (with OPENBLAS_L2_SIZE, which numpy 2.5.2-2.5.5 need beside it), one thread and OpenCV's IPP
# pin are set as well, so that a simulation or a measurement the agent makes repeats too. The .pth sets each value as a
# default for an /opt/venv Python that is started without the image's environment.
ENV MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=4 \
    OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256 OPENCV_IPP=avx2
ENV NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
RUN printf '%s\n' 'import os' \
      'os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")' \
      'os.environ.setdefault("OPENBLAS_L2_SIZE", "256")' \
      'os.environ.setdefault("NPY_DISABLE_CPU_FEATURES", "X86_V4 AVX512_ICL AVX512_SPR")' \
      'os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")' \
      'os.environ.setdefault("OMP_NUM_THREADS", "1")' \
      'os.environ.setdefault("OPENCV_IPP", "avx2")' \
      > /opt/venv/lib/python3.12/site-packages/cloth_numerics_pin.py \
    && printf '%s\n' 'import cloth_numerics_pin' > /opt/venv/lib/python3.12/site-packages/cloth_numerics_pin.pth \
    && chmod 0644 /opt/venv/lib/python3.12/site-packages/cloth_numerics_pin.py /opt/venv/lib/python3.12/site-packages/cloth_numerics_pin.pth \
    && env -i PATH=/opt/venv/bin:/usr/bin:/bin HOME=/tmp /opt/venv/bin/python3 -W error::ImportWarning -c "import os, numpy; e = os.environ; assert e.get('NPY_DISABLE_CPU_FEATURES') == 'X86_V4 AVX512_ICL AVX512_SPR' and e.get('OPENBLAS_CORETYPE') == 'Haswell' and e.get('OPENBLAS_L2_SIZE') == '256', dict(e); print('[image] numerics pin OK, numpy', numpy.__version__)"

# ---- the recordings: 11 fabrics x 7 clips x 3 cameras, with the clamp's and the arm's motion (agent-visible data only)
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets

RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

RUN python3 -c "import numpy, scipy, mujoco, cv2, skimage; print('[image] mujoco', mujoco.__version__, 'cv2', cv2.__version__)" \
    && python3 -c "import site, sys; assert not site.ENABLE_USER_SITE or sys.prefix != sys.base_prefix" \
    && runuser -u user -- python3 -c "import numpy, scipy, mujoco, cv2; print('[image] agent-UID import OK')" \
    && runuser -u user -- env HOME=/tmp python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>'); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m, 48, 64); r.update_scene(d); im = r.render(); assert im.max() > 0; print('[image] agent-UID MuJoCo OSMesa render OK', im.shape)" \
    && test -s /opt/task-assets/payload/data/fabrics/J/clip06/cam2.mp4
