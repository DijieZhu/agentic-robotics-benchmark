# robotic_arm_motion_controller_from_docs's environment image (the control stage of the robotic-arm workflow: a motion
# controller written from the vendor's documents).
#
# `environment/` without its `setup/` folder is the sole build context and the final stage derives DIRECTLY from the
# official ALE base. The block below is the arm tasks' common base (the same RUN / COPY / ENV lines and the same pins in
# requirements.txt as the other arm tasks). Task layers: none beyond the base (the controls engineer works with numpy,
# scipy, pinocchio, coal, trimesh, matplotlib), then the idle-timeout ENV block. No hidden data: the givens are staged
# by environment/setup/setup.sh, the grader and its model of the arm live in tests/ (absent from the image).
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (the image has no GPU, so every MuJoCo render goes through
# MUJOCO_GL=osmesa); the X / GL client libraries OCP and MuJoCo link even when headless; libgomp
# for manifold3d / CoACD.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`) and
# verify (root) all use the same interpreter. UV_PYTHON_INSTALL_DIR must be world-readable too
# (the default /root/.local/share/uv is untraversable for the agent account).
#
# Python 3.12: the engine stages `ale_verify` into the `python3` it finds and refuses anything
# below 3.12. The ALE base ships a standalone CPython 3.12 at /opt/python, so uv links the venv
# against it.
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

# The declared artifact path must exist even when nobody works (the engine captures declared
# artifacts after every episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# Headless defaults: MuJoCo renders through OSMesa inside the image; a task may override to egl
# where a GPU exists.
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Fail the build early if the toolchain is broken, prove the AGENT account can use the venv, and
# prove an offscreen OSMesa render works. One import line per -c: a continuation line starting
# with `from` reads as a Dockerfile FROM to the final-stage lint.
RUN python3 -c "import mujoco, mink, pinocchio, coacd, trimesh, manifold3d, gymnasium, numpy, scipy, jsonschema, yaml, PIL, matplotlib, cadquery, OCP; print('[arb-base-mujoco] import OK mujoco', mujoco.__version__, 'cadquery', cadquery.__version__)" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); print('[arb-base-mujoco] OSMesa render OK', px.shape, int(px.max()))" \
    && runuser -u user -- python3 -c "import mujoco, mink, cadquery, coacd; print('[arb-base-mujoco] agent-UID import OK')"

# ---- task layers -----------------------------------------------------------------------------------------------------
# Stream idle timers: the claude-code harness of earlier engines aborted a model stream after 300 s without a byte
# (Bun's HTTP idle timeout and the CLI's stream watchdog); both are raised to the gateway's own 1800 s upstream ceiling.
# ENV is inherited by every process the guest service starts, and the profile.d file covers the login shell the CLI is
# launched with (`bash -lc`). Nothing else in this image reads them; the block stays as in the other arm tasks.
ENV BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
RUN printf '%s\n' 'export BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800' 'export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000' \
        > /etc/profile.d/99-ale-idle-timeouts.sh \
    && chmod 0644 /etc/profile.d/99-ale-idle-timeouts.sh
