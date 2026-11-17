# The task's environment image. The build context is `environment/` without `setup/` (setup/ reaches
# the sandbox only when its stage runs, root-only). The final stage derives from the official ALE base
# at an exact version; the engine builds the image on demand and adds its own layer on top (its guest
# service); nothing here is pushed by hand.
#
# What belongs here: the Python stack every account uses (the agent's own tools, the public procedure,
# the desk service's plugin) and the desk service's dedicated account. What must NOT be here: the
# grader, the reference submission, the hidden desk tables, or anything derived from them; an image
# layer is readable by anyone who can run the image.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# A world-readable venv on a world-traversable interpreter: setup and the graders run as root, the agent
# and the desk's lab account run unprivileged, and all of them use this one interpreter. Python 3.12:
# at verify time the engine stages its own package into the first `python3` that is 3.12 or newer, and
# with /opt/venv/bin first on PATH that is this venv's interpreter. The ALE base ships a standalone
# CPython 3.12, so uv links the venv against it; /opt/uv-python exists only when uv had to download one.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONNOUSERSITE=1
# requirements.txt is the one list of Python dependencies; this image installs it.
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# The declared artifact path (task.toml `artifacts`), owned by the agent. The engine never creates an
# artifact path or fixes its ownership, so the image does both.
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# The desk's socket: the shipped client (labclient.py) reads ALE_LAB_SOCKET, so it reaches the desk
# without arguments.
ENV ALE_LAB_SOCKET=/run/ale-desk/lab.sock

# The lab service's dedicated account: the lab never runs as root or as the agent while it serves the
# hidden desk table.
RUN useradd --system --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin alelab

# No job scheduler may run agent-scheduled work outside the graded solver tree.
RUN (apt-get purge -y cron anacron at >/dev/null 2>&1 || true) \
    && rm -rf /etc/cron.d /etc/cron.daily /etc/cron.hourly /var/spool/cron /var/spool/at \
    && ! command -v crond && ! command -v cron && ! command -v atd

# Fail the build early if the venv is broken, and prove that the agent account can use every tool and
# that the lab account can load the plugin's one dependency.
RUN python3 -c "import importlib.metadata as m, numpy, scipy, pandas, PIL, cv2, yaml, jsonschema; print('[image]', ', '.join('%s %s' % (d, m.version(d)) for d in ('numpy', 'scipy', 'pandas', 'pillow', 'opencv-python-headless', 'PyYAML', 'jsonschema')))" \
    && python3 -c "import sys; assert sys.prefix != sys.base_prefix and sys.version_info[:2] == (3, 12)" \
    && runuser -u user -- /opt/venv/bin/python3 -c "import numpy, scipy, pandas, PIL.Image, cv2, yaml, jsonschema; print('[image] agent-UID import OK')" \
    && runuser -u alelab -- /opt/venv/bin/python3 -c "import numpy; print('[image] lab-UID import OK')"

# The payload's 71 page images and 24 CSV logs are served from the benchmark's asset dataset (docs/task-assets.md):
# environment/setup/setup.sh stages them into the agent home next to environment/setup/payload/, at the same paths.
# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md)
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets
