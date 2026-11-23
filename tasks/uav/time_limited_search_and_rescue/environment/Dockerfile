FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV MUJOCO_GL=osmesa LIBGL_ALWAYS_SOFTWARE=1 LP_NUM_THREADS=2
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONNOUSERSITE=1
# One numerics path on every host, here and for the graded program (tests/grader.py sets the same two): the OpenBLAS
# kernel and NumPy's dispatch otherwise follow the CPU, and an image-driven program then flies slightly differently on
# an AVX2 and an AVX-512 machine.
ENV OPENBLAS_CORETYPE=HASWELL NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL,AVX512_SPR
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt \
    && chmod -R a+rX /opt/venv
RUN mkdir -p /home/user/submission && chown user:user /home/user/submission
COPY visual_assets.json fetch_visual_assets.py /tmp/visual_assets/
RUN /opt/venv/bin/python3 /tmp/visual_assets/fetch_visual_assets.py /tmp/visual_assets/visual_assets.json /opt/visual_robotics/assets \
    && chmod -R a+rX /opt/visual_robotics/assets
# ---- the cameras' render bundle (surface maps and the image light; rescue_render.py reads it in place), fetched at
# the dataset commit pinned in assets.sha256 and checked file by file (docs/task-assets.md). It is part of the
# aircraft service, not of the agent's files: root reads it, the agent's account does not, and nothing stages it
# into the agent's home.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && tar -xf /opt/task-assets/payload/render/look.tar -C /opt/task-assets/payload/render \
    && rm /opt/task-assets/payload/render/look.tar \
    && chown -R root:root /opt/task-assets && chmod -R u=rwX,go= /opt/task-assets \
    && ! runuser -u user -- ls /opt/task-assets \
    && [ -z "$(find /opt/task-assets -perm /077)" ]
# The cameras' renderer starts here as it does in a flight: MuJoCo's Filament on Mesa's lavapipe (Vulkan on the CPU),
# as root (the aircraft service's account), with the bundle's image light and one of its maps.
COPY render_check.py /tmp/render_check.py
RUN /opt/venv/bin/python3 -I -B /tmp/render_check.py /opt/task-assets/payload/render && rm /tmp/render_check.py
