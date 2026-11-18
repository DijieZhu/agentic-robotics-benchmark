# robotic_arm_requirements_replica_from_photos's environment image (information tier, requirements stage).
#
# `environment/` without its `setup/` folder is the sole build context and the final stage derives DIRECTLY from the
# official ALE base. The block below is the chain's shared base `shared/images/arb-base-mujoco/Dockerfile`
# reproduced verbatim (same RUN / COPY / ENV lines, same requirements.txt bytes), so its layers come from the Docker
# layer cache of that base. Task layers (PLAN.md T17): opencv-python-headless + scikit-image (a human engineer's
# photogrammetry tools), then the Q15 idle-timeout ENV block. No task material, no hidden data: the givens are staged
# by environment/setup/setup.sh, the grader lives in tests/ (absent from the image).
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

# ---- task-specific layers (robotic_arm_requirements_replica_from_photos) ------------------------------------------------
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
RUN runuser -u user -- python3 -c "import cv2, skimage, numpy, scipy, mujoco, yaml, jsonschema; print('[arm-req-photos] agent-UID import OK cv2', cv2.__version__, 'skimage', skimage.__version__)"

# CHAIN_PLAN Q15 workaround (2026-09-27; docs/plan/arm/eval/engine-patches/Q15-request-reissue.md section 6). The
# claude-code harness aborts a model stream after 300 s without a byte (Bun's HTTP idle timeout and the CLI's stream
# watchdog); both timers are raised to the gateway's own 1800 s upstream ceiling. ENV is inherited by every process
# the guest service starts, and the profile.d file covers the login shell the CLI is launched with (`bash -lc`).
# Nothing else in this image reads them. Appended after every toolchain layer so the layers above stay cached.
ENV BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
RUN printf '%s\n' 'export BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800' 'export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000' \
        > /etc/profile.d/99-ale-idle-timeouts.sh \
    && chmod 0644 /etc/profile.d/99-ale-idle-timeouts.sh

# ---- the task's own last layers (2026-10-29; after every toolchain layer, so the layers above stay cached) ---------
# The agent harness starts its shell with PATH=$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin, so the agent's `python3`
# is /usr/local/bin/python3, the base interpreter of /opt/venv. A .pth file in that interpreter's site-packages appends
# the venv's site-packages, so the plain `python3` imports the venv's packages too (its own packages win a name clash).
# The RUN line is the one the evaluation copies of the measured rounds appended, so this layer is the one those rounds
# ran on.
RUN SP=$(/usr/local/bin/python3 -c "import site; print(site.getsitepackages()[0])") \
    && echo /opt/venv/lib/python3.12/site-packages > "$SP/zz-arb-eval-venv.pth" && chmod 0644 "$SP/zz-arb-eval-venv.pth" \
    && runuser -u user -- env PATH=/usr/local/bin:/usr/bin:/bin python3 -c "import numpy, trimesh; print(\"[eval-copy] plain python3 sees the venv\", numpy.__version__)"
# Build-time check: the agent account's python3, with the harness's PATH, is 3.12 and imports every tool the
# instruction names (matplotlib's cache in a scratch folder). Then that folder and the pip requirement lists copied
# to /tmp for the installs above are removed, so nothing of the build stays readable in the agent's /tmp.
RUN runuser -u user -- env PATH=/home/user/.local/bin:/usr/local/bin:/usr/bin:/bin MPLCONFIGDIR=/tmp/mpl-build-check python3 -c "import sys, numpy, scipy, cv2, skimage, mujoco, matplotlib, PIL, yaml, jsonschema, trimesh; assert sys.version_info[:2] == (3, 12) and mujoco.__version__.startswith('3.14'), sys.version; print('[task] harness python3', sys.version.split()[0], 'mujoco', mujoco.__version__, 'cv2', cv2.__version__)" \
    && rm -rf /tmp/mpl-build-check /tmp/requirements.txt /tmp/requirements-photo.txt

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md)
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets
