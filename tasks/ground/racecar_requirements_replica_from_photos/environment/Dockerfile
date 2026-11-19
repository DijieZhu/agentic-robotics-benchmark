# The task's environment image. The build context is `environment/` without `setup/`
# (setup/ reaches the sandbox only when its stage runs, root-only). The final stage derives from the
# official ALE base at an exact version; the engine builds the image on demand and adds its own layer
# on top (its guest service); nothing here is pushed by hand.
#
# What belongs here: the Python stack every account uses (the public procedure and the agent's own
# tools). What must NOT be here: the grader, the solution, the hidden truth, or anything derived from
# them — an image layer is readable by anyone who can run the image.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# A world-readable venv, and a world-TRAVERSABLE interpreter behind it. Setup and verify run as root,
# the agent runs unprivileged, and all of them use this one interpreter — so `uv`'s default
# managed-Python location (/root/.local/share/uv) would break the venv for exactly the account that is
# not root. Python 3.12: at verify time the engine stages its own package into the first `python3` /
# `python3.12` that is 3.12 or newer; with /opt/venv/bin first on PATH, that interpreter is this venv's
# `python3`. The ALE base ships a standalone CPython 3.12, so uv links the venv against it;
# /opt/uv-python exists only when uv had to download one.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONNOUSERSITE=1
# requirements.txt is the ONE list of Python dependencies; this image installs it.
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# The declared artifact path (task.toml `artifacts`), owned by the agent. The engine never creates an
# artifact path or fixes its ownership, so the image does both.
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# Headless by default.
ENV MPLBACKEND=Agg

# No job scheduler may run agent-scheduled work outside the graded solver tree.
RUN (apt-get purge -y cron anacron at >/dev/null 2>&1 || true) \
    && rm -rf /etc/cron.d /etc/cron.daily /etc/cron.hourly /var/spool/cron /var/spool/at \
    && ! command -v crond && ! command -v cron && ! command -v atd

# Fail the build early if the venv is broken — and prove the AGENT account can use it.
RUN python3 -c "import numpy, scipy, PIL, cv2, yaml, jsonschema, matplotlib; print('[image] numpy', numpy.__version__, 'scipy', scipy.__version__, 'cv2', cv2.__version__, 'matplotlib', matplotlib.__version__)" \
    && python3 -c "import sys; assert sys.prefix != sys.base_prefix and sys.version_info[:2] == (3, 12)" \
    && runuser -u user -- /opt/venv/bin/python3 -c "import numpy, scipy, PIL.Image, cv2, yaml, jsonschema, matplotlib; print('[image] agent-UID import OK')"

# ---- task assets: the 33 photographs and 27 page images, fetched at the dataset commit pinned in assets.sha256 and
# checked file by file (docs/task-assets.md); setup.sh stages them into the agent home and checks them once more
# against setup/given_images.sha256.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets
