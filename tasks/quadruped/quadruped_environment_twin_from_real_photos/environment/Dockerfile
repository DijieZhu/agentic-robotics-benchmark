# quadruped_environment_twin_from_real_photos' environment image. CPU only: MuJoCo (CPU rendering through OSMesa), the
# geometry and image tools and pycolmap in /opt/venv (the task's Python: setup, the agent's tools and the grader);
# OpenMVS's command-line tools in /opt/openmvs. `environment/` without its `setup/` folder is the build context. No task
# material and no hidden data are in here: the givens are staged by environment/setup/setup.sh, the grader arrives only
# at verify.

# ---- builder: OpenMVS 2.4.0 from conda-forge (the CPU build), installed from the exact package list
# openmvs-conda-explicit.txt (URLs and checksums; no solve), then reduced to the tools and the libraries they load.
# Everything conda installs stays in this stage; the one RUN deletes the prefix and the package cache, so neither the
# image nor the build cache keeps them. micromamba is the official static release binary, checked by sha256.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1 AS openmvs-build
ARG MICROMAMBA_VERSION=2.10.0-0
ARG MICROMAMBA_SHA256=701f39c06f8e6a670a0e186c8479b9e5c0502a2f0cda6cef65d83bdd2875e8bd
COPY openmvs_bundle.py openmvs_README.md openmvs-conda-explicit.txt /tmp/
RUN curl -fsSL -o /tmp/micromamba \
        "https://github.com/mamba-org/micromamba-releases/releases/download/${MICROMAMBA_VERSION}/micromamba-linux-64" \
    && echo "${MICROMAMBA_SHA256}  /tmp/micromamba" | sha256sum -c - \
    && chmod 0755 /tmp/micromamba \
    && export MAMBA_ROOT_PREFIX=/tmp/mamba CONDA_OVERRIDE_CUDA="" MAMBA_EXTRACT_THREADS=4 \
    && /tmp/micromamba create -y -q -p /tmp/openmvs-env --file /tmp/openmvs-conda-explicit.txt \
    && python3 /tmp/openmvs_bundle.py /tmp/openmvs-env /tmp/mamba/pkgs /opt/openmvs \
        InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh TransformScene \
    && cp /tmp/openmvs-conda-explicit.txt /opt/openmvs/conda-explicit.txt \
    && cp /tmp/openmvs_README.md /opt/openmvs/README.md \
    && chown -R root:root /opt/openmvs && chmod -R a+rX,go-w /opt/openmvs \
    && rm -rf /tmp/openmvs-env /tmp/mamba /tmp/micromamba /tmp/openmvs_bundle.py /tmp/openmvs_README.md \
        /tmp/openmvs-conda-explicit.txt /root/.cache /root/.mamba

# ---- the task image
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# MuJoCo's OSMesa path and the GL / X client libraries MuJoCo, OpenCV and Open3D link even when headless; libgomp for
# manifold3d / CoACD / Open3D. No Vulkan, no NVIDIA manifests: the task has no GPU.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libegl1 libglvnd0 libopengl0 libglx0 \
        libx11-6 libxext6 libxrender1 libxi6 libxrandr2 libxcursor1 libxinerama1 libsm6 libice6 libxkbcommon0 \
        libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv: setup and verify run as root, the agent and the oracle as `user`. Python 3.12 (the engine
# stages ale_verify into the python3 it finds and refuses anything older); the base image's CPython 3.12 is used.
# requirements.txt is the ONE list of Python dependencies, every distribution pinned: installed without resolving
# (--no-deps), then checked for consistency. Open3D's TensorFlow / PyTorch op libraries are removed (they load only
# with those frameworks, which the image does not have).
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --no-deps --python /opt/venv -r /tmp/requirements.txt \
    && uv pip check --python /opt/venv \
    && find /opt/venv -name 'open3d_tf_ops.so' -delete \
    && find /opt/venv -name 'open3d_torch_ops.so' -delete \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; } \
    && rm -f /tmp/requirements.txt

# OpenMVS: real binaries with their libraries beside them (RPATH $ORIGIN/../lib), so nothing is activated and no
# LD_LIBRARY_PATH is set; the directory goes at the END of PATH, after the system's own tools. Owner and modes were set
# in the builder stage (COPY --from keeps them; a chmod here would store the 260 MB a second time).
COPY --from=openmvs-build /opt/openmvs /opt/openmvs
ENV PATH=$PATH:/opt/openmvs/bin

# The declared artifact path must exist even when nobody works.
RUN install -d -o user -g user /home/user/submission

ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
# The sandbox's CPU count (task.toml [environment] cpus): the container reports the host's processors, and OpenMP
# programs would otherwise start one thread per host processor. The grader sets its own (one thread per process).
ENV OMP_NUM_THREADS=8

# The claude-code harness aborts a model stream after 300 s without a byte (a long tool-call input is buffered and
# sends nothing meanwhile); both timers are raised to the gateway's 1800 s upstream ceiling, as in the benchmark's
# other tasks whose agents write long files. ENV covers processes the guest service starts, profile.d the CLI's
# login shell.
ENV BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
RUN printf '%s\n' 'export BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800' 'export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000' \
        > /etc/profile.d/99-ale-idle-timeouts.sh \
    && chmod 0644 /etc/profile.d/99-ale-idle-timeouts.sh

# Fail the build early if a toolchain is broken, and prove the agent account can use it (it must traverse everything
# the venv and /opt/openmvs point at). One import statement per -c.
RUN python3 -c "import mujoco, numpy, scipy, cv2, skimage, PIL, trimesh, manifold3d, rtree, shapely, coacd, open3d, pycolmap; print('[image] mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__, 'open3d', open3d.__version__, 'pycolmap', pycolmap.__version__)" \
    && python3 -c "import cv2; s = cv2.SIFT_create(); print('[image] SIFT ok', type(s).__name__)" \
    && python3 -c "import pycolmap; assert not pycolmap.has_cuda; print('[image] pycolmap CPU build')" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); print('[image] OSMesa render OK', px.shape, int(px.max()))" \
    && cd /tmp && for t in InterfaceCOLMAP DensifyPointCloud ReconstructMesh RefineMesh TextureMesh TransformScene; do \
         env -i /opt/openmvs/bin/$t --help 2>&1 | grep -q 'OpenMVS x64 v2.4.0' || { echo "[image] $t does not start"; exit 1; }; done \
    && echo '[image] OpenMVS tools start (clean environment)' \
    && runuser -u user -- python3 -c "import mujoco, cv2, skimage, trimesh, coacd, open3d, pycolmap; print('[image] agent-UID import OK')" \
    && runuser -u user -- python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 32, 32); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); print('[image] agent-UID OSMesa render OK', int(r.render().max()))" \
    && runuser -u user -- sh -c 'cd /tmp && DensifyPointCloud --help 2>&1 | grep -q "OpenMVS x64 v2.4.0" && echo "[image] agent-UID OpenMVS OK"' \
    && rm -f /tmp/*.log

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md)
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets
