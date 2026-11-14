# quadruped_full_assembly_12dof_ur10e's environment image (assembly stage: the whole quadruped from its body and loose units;
# the same layers as robotic_arm_full_assembly_so101_panda's image, so they come from its layer cache).
#
# `environment/` (less its `setup/`) is the sole build context and the final stage derives DIRECTLY from the official
# ALE base, at an exact version. The block below is the chain's shared base `shared/images/arb-base-mujoco/Dockerfile`
# reproduced verbatim (same RUN / COPY / ENV lines, same requirements.txt bytes), so its layers come from the Docker
# layer cache of that base. Task layers: opencv-python-headless + scikit-image (the assembly engineer's camera-calibration
# and image tools), Mesa's Vulkan drivers (the agent cameras' Filament renderer), the task assets, then the Q15
# idle-timeout ENV block. No hidden data: the givens are staged by environment/setup/setup.sh, the grader lives in tests/
# (absent from the image) and downloads the cell's kit itself at verify time.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (owner decision Q12 / 05 B5: EGL enumerates 0 devices on the
# host and in Docker, so every sim render of the chain goes through MUJOCO_GL=osmesa); the X /
# GL client libraries OCP and MuJoCo link even when headless; libgomp for manifold3d / CoACD.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`) and
# verify (root) all use the same interpreter. UV_PYTHON_INSTALL_DIR must be world-readable too
# (the default /root/.local/share/uv is untraversable for the agent account).
#
# Python 3.12: the engine stages `ale_verify` into the `python3` it finds and refuses anything
# below 3.12 (engine-facts.md "Verification API"; drone_hover/image/Dockerfile). The ALE base
# ships a standalone CPython 3.12 at /opt/python, so uv links the venv against it.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
# --no-cache: the wheel cache would otherwise stay inside this layer (~1 GB); the image must
# stay lean (host / is at 99 %, owner decision Q12: Docker data-root move deferred).
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements.txt \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# The declared artifact path must exist even when nobody works (the engine captures declared
# artifacts after every episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# Headless defaults. MUJOCO_GL=osmesa is the chain's rendering path (CHAIN_PLAN.md section 4.0:
# "Sim renders through OSMesa inside the image"); a task may override to egl where a GPU exists.
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Fail the build early if the toolchain is broken, prove the AGENT account can use the venv, and
# prove an offscreen OSMesa render works. One import line per -c: a continuation line starting
# with `from` reads as a Dockerfile FROM to the final-stage lint.
RUN python3 -c "import mujoco, mink, pinocchio, coacd, trimesh, manifold3d, gymnasium, numpy, scipy, jsonschema, yaml, PIL, matplotlib, cadquery, OCP; print('[arb-base-mujoco] import OK mujoco', mujoco.__version__, 'cadquery', cadquery.__version__)" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); print('[arb-base-mujoco] OSMesa render OK', px.shape, int(px.max()))" \
    && runuser -u user -- python3 -c "import mujoco, mink, cadquery, coacd; print('[arb-base-mujoco] agent-UID import OK')"

# ---- task-specific layers (shared with arm_requirements_replica_from_photos: one cached layer) ----------------------
# Photogrammetry / image tools for the agent (T17, requirements.md Q-R6): OpenCV (headless) and scikit-image, pinned
# against the base's numpy 2.5.3.
# Only the NEW files are touched after the install (a chmod -R over the whole venv would copy the base's 2 GB venv
# into this layer): the marker file bounds the find.
COPY requirements-photo.txt /tmp/requirements-photo.txt
RUN touch /tmp/.before-photo-layer && sleep 1 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements-photo.txt \
    && find /opt/venv -newer /tmp/.before-photo-layer -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && find /opt/venv -newer /tmp/.before-photo-layer \( -type d -o -type f \) -exec chmod a+rX {} + \
    && rm -f /tmp/.before-photo-layer
RUN runuser -u user -- python3 -c "import cv2, skimage, numpy, scipy, mujoco, yaml, jsonschema; print('[arm-assemble-info] agent-UID import OK cv2', cv2.__version__, 'skimage', skimage.__version__)"

# The agent cameras' renderer (tests/arm/assembly_quad/agent_render.py): MuJoCo's Filament module (in the mujoco wheel)
# on Vulkan; Mesa's lavapipe is the CPU Vulkan driver (no GPU in the task's resources). Proved at build: a Filament
# context on lavapipe.
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -c "import mujoco._render_filament as F, mujoco.experimental.studio.window; c = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN)); print('[arm-full-assembly] Filament on Vulkan OK')"

# ---- task assets ---------------------------------------------------------------------------------------------------
# The picture manual, the units' STL files, the actuator module's documents, the arm and gripper models and the scenery
# of the agent cameras' renderer (render/: the factory hall, the bench top's wood maps) live in the benchmark's Hugging
# Face asset repository (tasks/quadruped/quadruped_full_assembly_12dof_ur10e/payload/ there).
# They are fetched at the commit pinned in assets.sha256 and checked file by file against the sha256 recorded there;
# environment/setup/setup.sh stages them into the agent home next to environment/setup/payload/, and the verify stage
# (tests/test.sh, in its separate verifier sandbox started from this image) copies them into the grader's root-owned
# copy of the givens. fetch_assets.py is the benchmark's canonical copy (shared/task_assets/) and the repository named in
# assets.sha256 is written from the benchmark's task-assets.toml (scripts/task_assets.py sync; docs/task-assets.md).
# The grader's kit is not among them: the verify stage downloads it itself, and it never enters this image.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets

# CHAIN_PLAN Q15 workaround: the claude-code harness aborts a model stream after 300 s without a byte; both timers are
# raised to the gateway's 1800 s upstream ceiling. ENV covers processes the guest service starts, profile.d the CLI's
# login shell. Nothing else in this image reads them. Appended after every toolchain layer so the layers above stay
# cached.
ENV BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
RUN printf '%s\n' 'export BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800' 'export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000' \
        > /etc/profile.d/99-ale-idle-timeouts.sh \
    && chmod 0644 /etc/profile.d/99-ale-idle-timeouts.sh
