FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 libgomp1 libxrender1 libxext6 libxi6 libxcursor1 libxinerama1 libxrandr2 mesa-vulkan-drivers libvulkan1 && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 && uv pip install --no-cache --python /opt/venv -r /tmp/requirements.txt && chmod -R a+rX /opt/venv && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV MPLBACKEND=Agg
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission
RUN runuser -u user -- python3 -c "import mujoco, numpy, scipy, trimesh, PIL; print('agent imports OK')"
RUN rm -f /usr/local/bin/python3 /usr/local/bin/python
COPY python3 /usr/local/bin/python3
RUN chmod 755 /usr/local/bin/python3 && ln -s /usr/local/bin/python3 /usr/local/bin/python && runuser -u user -- env PATH=/usr/local/bin:/usr/bin:/bin python3 -c "import mujoco, numpy, scipy, trimesh, PIL; print('agent sanitized PATH imports OK')"
RUN timeout 45s env -u DISPLAY -u WAYLAND_DISPLAY python3 -c 'import os,glob,gc; p=sorted(glob.glob("/usr/share/vulkan/icd.d/lvp_icd*.json")); assert p,"lavapipe ICD missing"; os.environ["VK_ICD_FILENAMES"]=p[0]; os.environ["VK_DRIVER_FILES"]=p[0]; import mujoco._render_filament as F, mujoco.experimental.studio.window; c=F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN)); print("Filament CPU Vulkan OK",p[0],flush=True); del c; gc.collect()'
# Task assets: the kit, the xArm 7 model, the picture manual and the cameras' scenery (render/) live in the benchmark's
# Hugging Face dataset (tasks/uav/uav_assembly_agipix_xarm7/payload/ there) as two archives, fetched
# at the commit pinned in assets.sha256, checked against the sha256 recorded there and unpacked into
# /opt/task-assets/payload. environment/setup/setup.sh stages them into the agent home, and the verify stage
# (tests/test.sh, in its separate verifier sandbox started from this image) copies them into the grader's root-owned
# copy of the givens. fetch_assets.py is the benchmark's canonical copy (shared/task_assets/) and the dataset named in
# assets.sha256 is written from the benchmark's task-assets.toml (scripts/task_assets.py sync; docs/task-assets.md).
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN printf 'precedence ::ffff:0:0/96 100\n' >> /etc/gai.conf
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload && tar --no-same-owner -xzf /opt/task-assets/payload/payload.tar.gz -C /opt/task-assets/payload && rm -rf /opt/task-assets/payload/given/manual && tar --no-same-owner -xzf /opt/task-assets/payload/manual.tar.gz -C /opt/task-assets/payload && rm /opt/task-assets/payload/payload.tar.gz /opt/task-assets/payload/manual.tar.gz && chmod -R a+rX /opt/task-assets
