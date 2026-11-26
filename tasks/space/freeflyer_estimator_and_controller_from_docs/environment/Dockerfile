# freeflyer_estimator_and_controller_from_docs' environment image (the control stage of the cabin free-flyer: an
# estimator and a controller written from documents, records and photographs). CPU only: Python 3.12 in /opt/venv
# with numpy and scipy, the controller's libraries (interface.libraries); no simulator. `environment/` without its
# `setup/` folder is the build context, so no file of setup/payload/ is in the image: the givens (setup/payload/:
# payload/ and controller/, and in the published form the given records, which the image build fetches into
# /opt/task-assets/payload) are staged into the agent's home by setup/setup.sh. Nothing of the grader is in here
# either: tests/ and solution/ arrive only at their own stages.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# World-readable venv: setup and verify run as root, the agent and the reference solution as `user`. Python 3.12 (the
# engine stages ale_verify into the python3 it finds and refuses anything older); requirements.txt is the one list of
# Python dependencies, every distribution pinned, installed without resolving and then checked.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --no-deps --python /opt/venv -r /tmp/requirements.txt \
    && uv pip check --python /opt/venv \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; } \
    && rm -f /tmp/requirements.txt

# The numerics' code paths, the same for the agent's own runs, the controller's process and the grader: numpy's
# dispatch without the AVX-512 kernels and OpenBLAS's Haswell kernels (its cache size set with them).
ENV NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512CD,AVX512_KNL,AVX512_KNM,AVX512_SKX,AVX512_CLX,AVX512_CNL,AVX512_ICL OPENBLAS_CORETYPE=Haswell OPENBLAS_L2_SIZE=256

# The declared artifact path must exist even when nobody works; the deliverable is its controller/ folder.
RUN install -d -o user -g user /home/user/submission /home/user/submission/controller

# The claude-code harness aborts a model stream after 300 s without a byte (a long tool-call input is buffered and
# sends nothing meanwhile); both timers are raised to the gateway's 1800 s upstream ceiling, as in the benchmark's
# other tasks whose agents write long files. ENV covers processes the guest service starts, profile.d the CLI's
# login shell.
ENV BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
RUN printf '%s\n' 'export BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800' 'export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000' \
        > /etc/profile.d/99-ale-idle-timeouts.sh \
    && chmod 0644 /etc/profile.d/99-ale-idle-timeouts.sh

# Fail the build early if the toolchain is broken, and prove the agent account can use it. One import line per -c.
RUN PYTHONDONTWRITEBYTECODE=1 python3 -c "import numpy, scipy, scipy.linalg; print('[image] numpy', numpy.__version__, 'scipy', scipy.__version__)" \
    && runuser -u user -- env PYTHONDONTWRITEBYTECODE=1 python3 -c "import numpy, scipy; print('[image] agent-UID import OK')"

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md)
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets
