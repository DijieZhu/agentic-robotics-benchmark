# racecar_head_to_head_race_on_unseen_track's environment image (behavior stage: a tenth-scale car's program races a
# head-to-head day unattended at a venue it has never seen; the grader runs it in its own MuJoCo cell).
#
# `environment/` (less its `setup/`) is the sole build context and the final stage derives DIRECTLY from the official
# ALE base, at an exact version. Nothing hidden is here: the givens (documents, lab logs, the client library and the
# protocol stub) are staged by environment/setup/setup.sh; the cell, the venues, the other cars, the reference numbers and
# the reference program live in the task's grading and solution folders, which never enter this image. The image holds
# no simulator of the venues and no racing stack.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (no GPU in the task's resources; EGL enumerates no device in the sandbox) and the GL
# client libraries MuJoCo links even when headless: the agent may render its own twin; the grader never renders.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`) and the scoring stage (root,
# and the unprivileged account its programs run as) all use the same interpreter. UV_PYTHON_INSTALL_DIR must be
# world-readable too (the default /root/.local/share/uv is untraversable for the agent account). Python 3.12: the
# engine stages its verification code into the `python3` it finds and refuses anything below 3.12; the ALE base
# ships a standalone CPython 3.12 at /opt/python, so uv links the venv against it. The base image sets HOME=/home/user
# for root too, so the build points uv's cache away from the agent home.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN export HOME=/root UV_CACHE_DIR=/tmp/uv-cache \
    && uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements.txt \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; } \
    && rm -rf /tmp/uv-cache /tmp/requirements.txt

# The declared artifact path must exist even when nobody works (the engine captures declared artifacts after every
# episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# Headless defaults; numeric libraries single-threaded for the agent's tools as for the graded run.
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Fail the build early if the toolchain is broken, prove the AGENT account can use the venv, prove an offscreen OSMesa
# render works, and prove the account switch and the limits the scoring stage starts every program with exist here
# (util-linux setpriv and prlimit). One import line per -c.
RUN python3 -c "import mujoco, numpy, scipy, jsonschema, PIL, cv2; print('[racecar-h2h] import OK mujoco', mujoco.__version__, 'numpy', numpy.__version__)" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); print('[racecar-h2h] OSMesa render OK', px.shape, int(px.max()))" \
    && runuser -u user -- python3 -c "import mujoco, numpy, scipy, cv2, PIL; print('[racecar-h2h] agent-UID import OK')" \
    && setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs -- prlimit --cpu=60 --core=0 -- /opt/venv/bin/python3 -I -c "import os; assert os.getuid() == 65534 and os.getgid() == 65534; print('[racecar-h2h] setpriv + prlimit OK')"

# ---- task assets ---------------------------------------------------------------------------------------------------
# The large givens (the car's lab logs, the venue photographs and the car's meshes), fetched at
# the dataset commit pinned in assets.sha256 and checked file by file (docs/task-assets.md). One check, two sources:
#   * by default (every checkout: the build context has no assets/payload/) the benchmark's Hugging Face asset
#     dataset at the pinned commit, downloaded by fetch_assets.py (the benchmark's canonical copy in
#     shared/task_assets/; it checks every file against its sha256 as it downloads);
#   * only in a maintainer's local build, the same files placed in the build context's assets/payload/ (an
#     engine asset root, never in git): exactly the listed files are copied, an unlisted or missing file fails the
#     build, and every copied file is checked against its sha256.
# environment/setup/setup.sh stages them into the agent home next to environment/setup/payload/. The image's copy is
# root-only: setup runs as root and is its only reader, and in the separate verifier sandbox (where setup never
# runs) the graded program's account must not find the givens there either (the instruction: the given files are
# not there at a venue; the submission cap of 60 MB is not to be side-stepped by reading the lab logs in place).
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
    chown -R root:root /opt/task-assets; chmod -R go-rwx /opt/task-assets; \
    if setpriv --reuid=65534 --regid=65534 --clear-groups -- test -r /opt/task-assets/assets.sha256; then \
        echo "[assets] /opt/task-assets is readable by another account" >&2; exit 1; fi; \
    echo "[assets] /opt/task-assets is root-only"

# The claude-code harness aborts a model stream after 300 s without a byte; both timers are raised to the gateway's
# 1800 s upstream ceiling. ENV covers processes the guest service starts, profile.d the CLI's login shell. Nothing else
# in this image reads them. Appended after every toolchain layer so the layers above stay cached.
ENV BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
RUN printf '%s\n' 'export BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800' 'export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000' \
        > /etc/profile.d/99-ale-idle-timeouts.sh \
    && chmod 0644 /etc/profile.d/99-ale-idle-timeouts.sh
