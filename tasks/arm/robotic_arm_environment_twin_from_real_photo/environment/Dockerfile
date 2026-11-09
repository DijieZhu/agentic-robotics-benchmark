# robotic_arm_environment_twin_from_real_photo's environment image. MuJoCo, usd-core and the geometry and image tools
# in /opt/venv (the task's Python: setup, the agent's tools, the first process of render_twin.py and the grader);
# NVIDIA OVRTX + ovstage in their own venv /opt/ovrtx-venv, which only the second process of render_twin.py uses.
# `environment/` without its `setup/` folder is the build context. No task material and no hidden data are in here: the
# givens are staged by environment/setup/setup.sh, the grader arrives only at verify.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# NVIDIA OVRTX and ovstage are proprietary NVIDIA software, licensed under the NVIDIA Software License Agreement and
# the product-specific terms for NVIDIA AI products (see the LICENSE file inside each package). The wheels are
# downloaded from NVIDIA's package index (pypi.nvidia.com) when this image is built and are never redistributed by
# this repository. Building the image means accepting those terms.
ARG ACCEPT_NVIDIA_EULA=1
RUN test "$ACCEPT_NVIDIA_EULA" = "1"

# MuJoCo's OSMesa path and the GL / X client libraries MuJoCo and OpenCV link even when headless; the Vulkan loader,
# GLVND and the X / hwloc libraries OVRTX loads; libgomp for manifold3d / CoACD. The GPU driver's own libraries come
# from the container runtime at start, which does not always add the loader manifests (the NVIDIA hook path adds none
# of them), so the Vulkan ICD and the EGL vendor manifests are written here, at the paths the runtime's CDI
# specification uses (a CDI mount then replaces them rather than adding a second entry).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libegl1 libglvnd0 libopengl0 libglx0 libvulkan1 \
        libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 libsm6 libice6 libxkbcommon0 \
        libfontconfig1 libfreetype6 libgomp1 libhwloc15 \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /etc/vulkan/icd.d /usr/share/glvnd/egl_vendor.d \
    && printf '{\n  "file_format_version": "1.0.1",\n  "ICD": {"library_path": "libGLX_nvidia.so.0", "api_version": "1.4.312"}\n}\n' \
       > /etc/vulkan/icd.d/nvidia_icd.json \
    && printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {"library_path": "libEGL_nvidia.so.0"}\n}\n' \
       > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# World-readable venvs: setup and verify run as root, the agent and the oracle as `user`. Python 3.12 (the engine
# stages ale_verify into the python3 it finds and refuses anything older); the base image's CPython 3.12 is used.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt requirements-ovrtx.txt /tmp/
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements.txt \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# OVRTX and ovstage from NVIDIA's index only, at exact versions and wheel hashes, without dependencies (they declare
# none); numpy from PyPI for the arrays the renderer hands back.
RUN uv venv /opt/ovrtx-venv --python 3.12 \
    && uv pip install --no-cache --python /opt/ovrtx-venv --no-deps --require-hashes \
       --index-url https://pypi.nvidia.com -r /tmp/requirements-ovrtx.txt \
    && uv pip install --no-cache --python /opt/ovrtx-venv numpy==2.5.4 \
    && find /opt/ovrtx-venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/ovrtx-venv \
    && rm -f /tmp/requirements.txt /tmp/requirements-ovrtx.txt
ENV ARB_OVRTX_PYTHON=/opt/ovrtx-venv/bin/python

# The declared artifact path must exist even when nobody works; the agent's cache folder (OVRTX keeps its shader and
# texture caches under ~/.cache/ovrtx) belongs to the agent.
RUN install -d -o user -g user /home/user/submission /home/user/.cache /home/user/.cache/ovrtx

ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Fail the build early if a toolchain is broken, and prove the agent account can use both venvs. There is no GPU
# during the build, so OVRTX can only be imported here. One import statement per -c.
RUN python3 -c "import mujoco, numpy, scipy, cv2, skimage, PIL, trimesh, manifold3d, rtree, coacd, shapely; print('[image] mujoco', mujoco.__version__, 'numpy', numpy.__version__)" \
    && python3 -c "import pxr.Usd; print('[image] usd-core ok')" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); print('[image] OSMesa render OK', px.shape, int(px.max()))" \
    && /opt/ovrtx-venv/bin/python -c "import ovrtx, ovstage, numpy; print('[image] ovrtx', ovrtx.__version__)" \
    && runuser -u user -- python3 -c "import mujoco, cv2, skimage, trimesh, coacd, pxr.Usd; print('[image] agent-UID import OK')" \
    && runuser -u user -- /opt/ovrtx-venv/bin/python -c "import ovrtx, ovstage; print('[image] agent-UID ovrtx import OK')" \
    && chown -R user:user /home/user/.cache

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md)
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets
