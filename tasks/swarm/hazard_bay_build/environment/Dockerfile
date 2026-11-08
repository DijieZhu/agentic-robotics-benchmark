# The task's environment: NVIDIA Isaac Sim + Isaac Lab on the official ALE container base, for one NVIDIA GPU, with an
# offline asset library and the task's own assets (the airframes, the plant's hall).
# `environment/` (less its `setup/`) is the sole build context. Nothing from tests/ or solution/ is in here.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# Isaac Sim's runtime libraries. The GPU driver's own libraries are mounted by the container runtime at start
# (the sandbox needs the `graphics` driver capability for the RTX renderer), but the Vulkan and EGL loaders and
# their ICD manifests have to be in the image, as in NVIDIA's own Isaac Sim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libvulkan1 libglu1-mesa libegl1 libgl1 libxrandr2 libxt6 \
       libxkbcommon0 libxi6 libsm6 libice6 libxrender1 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /etc/vulkan/icd.d /etc/vulkan/implicit_layer.d /usr/share/glvnd/egl_vendor.d \
    && printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.3.194"}\n}\n' \
       > /etc/vulkan/icd.d/nvidia_icd.json \
    && printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {"library_path": "libEGL_nvidia.so.0"}\n}\n' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# Isaac Sim asks to accept its licence (NVIDIA Software License Agreement) on first start.
ENV OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y PRIVACY_CONSENT=Y

# A world-readable venv behind a world-traversable interpreter: setup and verify run as root, the agent and the
# oracle unprivileged, and all of them use this interpreter. Python 3.12: the engine stages `ale_verify` into
# whatever answers `python3` and refuses anything older; Isaac Sim 6 / Isaac Lab 3 are built for 3.12 as well.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
# The published pair does not resolve as is: isaacsim-kernel 6.2.0.0 pins urllib3==2.7.0, isaaclab 3.0.0 asks for
# urllib3>=2.8.0. The override settles it on the newer release. isaacsim-core pins a pre-release (tinyobjloader rc).
# (The base image's HOME is the agent's home: uv's cache goes to a folder of its own and is removed.)
RUN uv venv /opt/venv --python 3.12 \
    && echo "urllib3==2.8.0" > /tmp/overrides.txt \
    && UV_CACHE_DIR=/tmp/uv-cache UV_HTTP_TIMEOUT=600 uv pip install --python /opt/venv --index-strategy unsafe-best-match \
       --prerelease=if-necessary-or-explicit --override /tmp/overrides.txt -r /tmp/requirements.txt \
    && rm -rf /tmp/uv-cache /root/.cache/uv /home/user/.cache/uv \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# An offline part of NVIDIA's public Isaac Sim asset library (arms, grippers, wheeled bases, multirotors, YCB and other
# objects, a warehouse; each folder carries its own licence: Apache-2.0, BSD-3-Clause, MIT, CC-BY-4.0), pinned file by
# file, and Isaac Sim / Isaac Lab pointed at it instead of the cloud (the sandbox has no network).
COPY fetch_isaac_assets.py isaac_assets.sha256 /opt/assets/
RUN python3 /opt/assets/fetch_isaac_assets.py /opt/assets/isaac_assets.sha256 /opt/assets/isaac \
    && chmod -R a+rX /opt/assets \
    && grep -rlZ "Assets/Isaac/6.2" /opt/venv/lib/python3.12/site-packages/isaacsim/apps \
       /opt/venv/lib/python3.12/site-packages/isaacsim/exts/isaacsim.storage.native/config \
       /opt/venv/lib/python3.12/site-packages/isaaclab/apps \
       | xargs -0 sed -i -E 's#https://omniverse-content-production\.s3(-us-west-2)?\.amazonaws\.com/Assets/Isaac/6\.2#/opt/assets/isaac#g'

# The task's own assets in /opt/assets/arb: four multirotor airframes (generated USD), the Flow volumes of the fire and
# the steam, and the art, all of it visual only (no physics): the plant's hall round the bay (`hall/`) and the
# airframes' look layer (`airframes/look/`). The art and the photographs come from the benchmark's asset dataset,
# fetched at the commit pinned in assets.sha256 and checked file by file (docs/task-assets.md); the art is moved into
# place here, the photographs (`given/photos/`) stay in /opt/task-assets/payload for setup.sh to stage.
COPY arb_assets/ /opt/assets/arb/
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && cp -r /opt/task-assets/payload/arb/. /opt/assets/arb/ \
    && rm -rf /opt/task-assets/payload/arb \
    && chmod -R a+rX /opt/assets/arb /opt/task-assets

# Kit writes its caches (cooked physics meshes, shaders, derived data), logs and settings next to its own files; the
# pip install leaves those folders absent and the agent account could not create them (RTX then fails to start).
# Shared and sticky here; the grader empties them before it opens a submission.
RUN for d in cache logs data; do install -d -m 1777 /opt/venv/lib/python3.12/site-packages/isaacsim/kit/$d; done

# The declared artifact path must exist even when nobody works.
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

ENV MPLBACKEND=Agg

# Fail the build early if the venv is broken, and prove the agent account can use it (then leave the agent's home as
# the agent's: nothing root-owned in it).
RUN HOME=/tmp/selfcheck python3 -c "import torch, isaaclab, isaacsim, cv2; print('[image] torch', torch.__version__, 'isaaclab ok')" \
    && runuser -u user -- python3 -c "import torch, isaaclab, isaacsim, cv2; print('[image] agent-UID import OK')" \
    && rm -rf /tmp/selfcheck /home/user/.nvidia-omniverse /home/user/.cache \
    && chown -R user:user /home/user
