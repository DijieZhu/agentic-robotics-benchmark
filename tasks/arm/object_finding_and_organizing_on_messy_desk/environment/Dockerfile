# The task's environment image. `environment/` (less its `setup/`) is the sole build context,
# and the final stage must derive directly from an official ALE base, at an exact version.
# Nothing from tests/ or solution/ is here.
#
# Two interpreters, on purpose:
#   /opt/venv   py3.12  answers `python3`; the verify stage and the agent's tooling use it
#                       (the engine stages ale_verify into it).
#   /opt/isaac  py3.10  runs the Isaac Sim 4.5 simulator process, the grader's and the agent's
#                       desk lab (Isaac supports 3.10 only).
#                       Never on PATH; the environment launches it by absolute path.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_NO_CACHE=1
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Shared libraries Isaac Sim's Kit runtime loads even when headless.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglu1-mesa libxt6 libxrandr2 libxinerama1 libxcursor1 libxi6 libgl1 libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# The simulator interpreter (py3.10) with Isaac Sim 4.5 from NVIDIA's index.
COPY requirements-isaac.txt /tmp/requirements-isaac.txt
# Permissions are fixed inside the same layer: a later `chmod -R` over 15 GB of files
# would copy every one of them up into a new layer and double the image.
RUN uv venv /opt/isaac --python 3.10 \
    && UV_HTTP_TIMEOUT=900 UV_CONCURRENT_DOWNLOADS=4 uv pip install --python /opt/isaac \
         -r /tmp/requirements-isaac.txt --extra-index-url https://pypi.nvidia.com \
    && mkdir -p /opt/isaac/lib/python3.10/site-packages/omni/cache \
    && chmod -R a+rX /opt/isaac \
    && chmod -R a+rwX /opt/isaac/lib/python3.10/site-packages/omni/cache \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# Stock Isaac assets (Franka, KLT bin, YCB, Simple Room, materials; NVIDIA Omniverse asset
# licence), mirrored from NVIDIA's public bucket at build time so that every stage runs
# with the network blocked. `NVIDIA/` exists so that Isaac's asset-root probe accepts the
# folder; the task only uses `Isaac/`.
COPY mirror_assets.py /tmp/mirror_assets.py
RUN python3 -I /tmp/mirror_assets.py /opt/isaac_assets && mkdir -p /opt/isaac_assets/NVIDIA \
    && chmod -R a+rX /opt/isaac_assets

# The verify-stage interpreter (py3.12), world-readable, agent-usable. It comes after the
# heavy Isaac and asset layers so that a change to its small requirement list rebuilds
# only this layer.
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv -r /tmp/requirements.txt \
    && chmod -R a+rX /opt/venv

# Isaac's Kit runtime writes shader and material caches next to its own packages (the
# world-writable omni/cache above). Every identity (root in verify, the agent account in
# the agent and oracle phases) must be able to, or Kit stalls silently at start-up.
ENV ACCEPT_EULA=Y
ENV OMNI_KIT_ACCEPT_EULA=YES

# The declared artifact path must exist even when nobody works.
RUN mkdir -p /home/user/submission && chown -R user:user /home/user

ENV MPLBACKEND=Agg

# Fail the build early: the verify interpreter is 3.12, and the AGENT account can import
# the simulator from the second venv.
RUN python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; import numpy, PIL; print('[image] python3', sys.version.split()[0])" \
    && runuser -u user -- /opt/isaac/bin/python3 -c "import isaacsim, numpy, PIL; print('[image] agent-UID isaac import OK')"
