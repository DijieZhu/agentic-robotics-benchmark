# racecar_full_assembly_tenth_scale_ur10e's environment image (assembly stage: a UR10e-class arm with a 2F-85-class gripper
# builds a tenth-scale car from its rolling chassis and eight loose units in a MuJoCo cell).
#
# `environment/` (less its `setup/`) is the sole build context and the final stage derives DIRECTLY from the official
# ALE base, at an exact version. The layers are those of the benchmark's other robot-assembly cells: the shared MuJoCo
# base (OSMesa, a world-readable venv at /opt/venv from requirements.txt), the camera-calibration and image tools
# (requirements-photo.txt), Mesa's Vulkan drivers for the agent cameras' renderer, then the task assets and the
# idle-timeout ENV block. No hidden data: the givens are staged by environment/setup/setup.sh, the cell's grading code
# is not in this image, and the cell's hidden kit never enters it.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (EGL enumerates no device on the host and in Docker, so every classic MuJoCo render
# goes through MUJOCO_GL=osmesa); the X / GL client libraries the CAD kernel and MuJoCo link even when headless;
# libgomp for manifold3d / CoACD.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`) and the scoring stage (root) all use the same
# interpreter. UV_PYTHON_INSTALL_DIR must be world-readable too (the default /root/.local/share/uv is untraversable for
# the agent account). Python 3.12: the engine stages its verification package into the `python3` it finds and refuses
# anything below 3.12; the ALE base ships a standalone CPython 3.12 at /opt/python, so uv links the venv against it.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
# --no-cache: the wheel cache would otherwise stay inside this layer (~1 GB).
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements.txt \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# The declared artifact path must exist even when nobody works (the engine captures declared artifacts after every
# episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# Headless defaults: classic MuJoCo renders through OSMesa inside the image.
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Fail the build early if the toolchain is broken, prove the AGENT account can use the venv, and prove an offscreen
# OSMesa render works. One import line per -c: a continuation line starting with that keyword would read as a stage
# line to the final-stage lint.
RUN python3 -c "import mujoco, mink, pinocchio, coacd, trimesh, manifold3d, gymnasium, numpy, scipy, jsonschema, yaml, PIL, matplotlib, cadquery, OCP; print('[arb-base-mujoco] import OK mujoco', mujoco.__version__, 'cadquery', cadquery.__version__)" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); print('[arb-base-mujoco] OSMesa render OK', px.shape, int(px.max()))" \
    && runuser -u user -- python3 -c "import mujoco, mink, cadquery, coacd; print('[arb-base-mujoco] agent-UID import OK')"

# ---- task-specific layers -------------------------------------------------------------------------------------------
# Camera calibration and image tools for the agent: OpenCV (headless) and scikit-image, pinned against the base's
# numpy 2.5.3. Only the NEW files are touched after the install (a chmod -R over the whole venv would copy the base's
# 2 GB venv into this layer): the marker file bounds the find.
COPY requirements-photo.txt /tmp/requirements-photo.txt
RUN touch /tmp/.before-photo-layer && sleep 1 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements-photo.txt \
    && find /opt/venv -newer /tmp/.before-photo-layer -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && find /opt/venv -newer /tmp/.before-photo-layer \( -type d -o -type f \) -exec chmod a+rX {} + \
    && rm -f /tmp/.before-photo-layer
RUN runuser -u user -- python3 -c "import cv2, skimage, numpy, scipy, mujoco, yaml, jsonschema; print('[arm-assemble-info] agent-UID import OK cv2', cv2.__version__, 'skimage', skimage.__version__)"

# The agent cameras' renderer: MuJoCo's Filament module (in the mujoco wheel) on Vulkan; Mesa's lavapipe is the CPU
# Vulkan driver (no GPU in the task's resources). Proved at build: a Filament context on lavapipe.
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -c "import mujoco._render_filament as F, mujoco.experimental.studio.window; c = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN)); print('[arm-full-assembly] Filament on Vulkan OK')"

# ---- task assets ---------------------------------------------------------------------------------------------------
# The large givens (the units' STL files, the drawings, the picture manual, the arm and gripper models) and the scenery
# of the agent cameras' renderer (render/: the factory hall, the bench top's wood maps), fetched at the dataset commit
# pinned in assets.sha256 and checked file by file (docs/task-assets.md). assets.sha256 is written from the benchmark's
# task-assets.toml (scripts/task_assets.py); it pins every file by sha256. One check, two sources:
#   * by default (every checkout: the build context has no assets/payload/) the benchmark's Hugging Face asset
#     dataset at the pinned commit, downloaded by fetch_assets.py (the benchmark's canonical copy in
#     shared/task_assets/; it checks every file against its sha256 as it downloads);
#   * only in a maintainer's development build, the same files placed in the build context's assets/payload/ (an
#     engine asset root, never in git): exactly the listed files are copied, an unlisted or missing file fails the
#     build, and every copied file is checked against its sha256.
# environment/setup/setup.sh stages them into the agent home next to environment/setup/payload/; the scoring stage
# copies the same files from this image into its own root-owned copy of the givens. The grader's kit is not among
# them: the verify stage downloads it itself, and it never enters this image.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN --mount=type=bind,target=/tmp/build-context \
    set -eu; \
    list=/opt/task-assets/assets.sha256; dest=/opt/task-assets/payload; src=/tmp/build-context/assets/payload; \
    if [ -d "$src" ]; then \
        grep -v -e '^#' -e '^[[:space:]]*$' "$list" > /tmp/assets.list; \
        sed -e 's/^[0-9a-fA-F]\{64\}  \*\{0,1\}//' /tmp/assets.list | LC_ALL=C sort > /tmp/assets.want; \
        (cd "$src" && find . -type f | sed -e 's#^\./##' | LC_ALL=C sort) > /tmp/assets.have; \
        extra="$(LC_ALL=C comm -13 /tmp/assets.want /tmp/assets.have)"; \
        missing="$(LC_ALL=C comm -23 /tmp/assets.want /tmp/assets.have)"; \
        if [ -n "$extra" ]; then echo "[assets] files in assets/payload/ that assets.sha256 does not list:" $extra >&2; exit 1; fi; \
        if [ -n "$missing" ]; then echo "[assets] files assets.sha256 lists that assets/payload/ lacks:" $missing >&2; exit 1; fi; \
        mkdir -p "$dest"; \
        (cd "$src" && tar -cf - -T /tmp/assets.want) | (cd "$dest" && tar -xf - --no-same-owner); \
        (cd "$dest" && sha256sum --check --strict --quiet /tmp/assets.list); \
        echo "[assets] $(wc -l < /tmp/assets.list) file(s) from the build context's assets/payload/ verified into $dest"; \
        rm -f /tmp/assets.list /tmp/assets.want /tmp/assets.have; \
    else \
        python3 /opt/task-assets/fetch_assets.py "$list" "$dest"; \
    fi; \
    chmod -R a+rX /opt/task-assets

# The claude-code harness aborts a model stream after 300 s without a byte; both timers are raised to the gateway's
# 1800 s upstream ceiling. ENV covers processes the guest service starts, profile.d the CLI's login shell. Nothing else
# in this image reads them. Appended after every toolchain layer so the layers above stay cached.
ENV BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
RUN printf '%s\n' 'export BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800' 'export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000' \
        > /etc/profile.d/99-ale-idle-timeouts.sh \
    && chmod 0644 /etc/profile.d/99-ale-idle-timeouts.sh
