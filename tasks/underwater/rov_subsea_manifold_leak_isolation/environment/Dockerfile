# rov_subsea_manifold_leak_isolation's environment image. `environment/` (less its `setup/`) is the sole build context
# and the final stage derives DIRECTLY from the official ALE base, at an exact version. Nothing hidden is here: the
# givens are staged by environment/setup/setup.sh; the grader, its reference value, the hidden sites and the reference
# program live in the grading stage and the reference solution (absent from the image). The image does hold the
# practice facilities (the vehicle cell on the training tank and the sister site, for the agent's practice runs),
# readable by their own account only, and the cell's own environment, readable by that account and root only.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# Software rendering libraries (no GPU in the task's resources): the cell renders the camera through the classic
# OpenGL path on llvmpipe; GL client libraries the renderer links even when headless; libgomp for numpy / scipy.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Accounts: `subseacell` runs the practice facilities' service; `cellrun0` .. `cellrun2` run the program of the
# grader's three station slots (one account per slot, so the clean-up after one station never touches another).
RUN useradd --system --no-create-home --home-dir /opt/subseacell-practice --shell /usr/sbin/nologin subseacell \
    && for k in 0 1 2; do useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin cellrun$k; done

# Two venvs on one world-readable uv Python 3.12 (the engine stages its verify helper into the `python3` on PATH and
# refuses anything below 3.12): /opt/venv, world-readable, the program's packages (setup, the agent, the oracle and the
# grader's parent process); /opt/cellenv, the same packages plus the cell's physics engine, readable by root and the
# practice account only (the cell runs there; the agent account can neither list nor import it). The base image sets
# HOME=/home/user for root too, so the build points uv's cache away from the agent home.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt requirements-cell.txt /tmp/req/
RUN export HOME=/root UV_CACHE_DIR=/tmp/uv-cache \
    && uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/req/requirements.txt \
    && uv venv /opt/cellenv --python 3.12 \
    && cd /tmp/req && uv pip install --no-cache --python /opt/cellenv -r /tmp/req/requirements-cell.txt \
    && find /opt/venv /opt/cellenv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; } \
    && chown -R root:subseacell /opt/cellenv \
    && chmod -R u+rwX,g+rX,g-w,o-rwx /opt/cellenv \
    && rm -rf /tmp/uv-cache /tmp/req

# The declared artifact path must exist even when nobody works (the engine captures declared artifacts after every
# episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# The practice facilities: the vehicle cell on the team's training tank and sister site (practice/: the service
# program, its stored facility packages and allowance, the cell library cell/subseacell/ - a byte copy of the grader's
# cell without its scoring module or the reference program - the simulation kit, the pinned vehicle meshes and the task
# values), installed for the account `subseacell` only: mode 700, so the agent account can neither read nor list it.
# environment/setup/setup.sh starts it before the agent's session (and restarts it should it ever exit); the agent
# reaches it through a unix socket and gets the vehicle's public calls, reset(site, setup) and, in the tank, the tank's
# instruments.
COPY practice /opt/subseacell-practice
RUN find /opt/subseacell-practice -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && rm -rf /opt/subseacell-practice/state \
    && mkdir -p /opt/subseacell-practice/state \
    && chown -R subseacell:subseacell /opt/subseacell-practice \
    && chmod -R u+rwX,go-rwx /opt/subseacell-practice

# Headless defaults for everyone: no matplotlib window.
ENV MPLBACKEND=Agg

# Fail the build early if the toolchain is broken; prove the AGENT account can use its venv and import the client
# library (a temporary copy of the practice cell's client.py, the same bytes as the given one) and can NOT reach the
# cell's environment; prove the practice service's cell builds the training tank from its package and renders a camera
# frame (and a sonar sweep) as its own account; finally, the agent account can neither list nor read the practice
# facilities, and nothing in the agent home is root-owned. One import line per -c: a continuation line starting with
# `from` reads as a Dockerfile FROM to the final-stage lint.
COPY buildcheck.py /tmp/buildcheck.py
RUN export HOME=/root \
    && python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; import numpy, scipy, PIL, cv2, yaml; print('[subsea] import OK numpy', numpy.__version__, 'scipy', scipy.__version__, 'cv2', cv2.__version__)" \
    && runuser -u user -- python3 -c "import numpy, scipy, PIL, cv2, yaml; print('[subsea] agent-UID import OK')" \
    && ! runuser -u user -- python3 -c "import mujoco" > /dev/null 2>&1 \
    && ! runuser -u user -- ls /opt/cellenv > /dev/null 2>&1 \
    && mkdir -p /tmp/client-proof/subseacell \
    && cp /opt/subseacell-practice/cell/subseacell/client.py /opt/subseacell-practice/cell/subseacell/__init__.py /tmp/client-proof/subseacell/ \
    && chmod -R a+rX /tmp/client-proof /tmp/buildcheck.py \
    && runuser -u user -- python3 -B -E -s /tmp/buildcheck.py agent-client /tmp/client-proof \
    && rm -rf /tmp/client-proof \
    && runuser -u subseacell -- env -u DISPLAY HOME=/opt/subseacell-practice MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=1 \
        UWKIT_MESHLIB=/opt/subseacell-practice/meshes SUBSEACELL_VEHICLE_MESHES=/opt/subseacell-practice/meshes \
        SUBSEACELL_VISUAL_MESHES=/opt/subseacell-practice/meshes \
        /opt/cellenv/bin/python3 -B -s /tmp/buildcheck.py practice-cell \
    && ! runuser -u user -- ls /opt/subseacell-practice > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/subseacell-practice/sites/sites.json > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/subseacell-practice/cell/subseacell/server.py > /dev/null 2>&1 \
    && test -z "$(find /home/user -user root -print -quit)" \
    && rm -f /tmp/buildcheck.py
