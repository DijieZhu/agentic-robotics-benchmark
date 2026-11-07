# The separate verification image. Same simulator stack and controller Python as the
# agent's image (the submitted controller runs here), plus what the agent must never see:
#
#   * HELD-OUT scenes: three TravelUAV scenes the agent's image does not carry, owned by
#     the simulator account `aerialsim` and unreadable by anyone else;
#   * the hidden placement pool of those scenes, from which the grader assembles missions:
#     root-only.
#
# Build-time network is needed (scenes, spawn areas, models). Every source is public and
# ungated, pinned to a revision.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

ARG SCENES_DATASET=wangxiangyu0814/TravelUAV_env
ARG SCENES_REVISION=44de5739a95a2f6a88767446421cddada9606642
ARG SPAWN_AREAS_URL=https://raw.githubusercontent.com/XuPeng23/AeroVLA/2d7ebd60e36ce12c3c0898f7b7a4ad52c1315e31/data/meta/map_spawnarea_info.json
ARG SPAWN_AREAS_SHA256=120438c62c6fa286c8cf44213dd03f5b7cd863e33889ae21c913ab5f68b530e6
ARG HELDOUT_MAPS="NewYorkCity NYCEnvironmentMegapa TropicalIsland"

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_NO_CACHE=1

# Unreal Engine (offscreen Vulkan) runtime libraries and 7-Zip for the split scene archive.
RUN apt-get update && apt-get install -y --no-install-recommends \
        p7zip-full libvulkan1 vulkan-tools libgl1 libglu1-mesa libxrandr2 libxinerama1 \
        libxcursor1 libxi6 libxt6 libnss3 libasound2 libpulse0 libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /var/lib/aerialsim --shell /usr/sbin/nologin aerialsim

# ---- simulator side: Python 3.10 with OpenCV and msgpack (identical to the agent image) ----
COPY requirements-sim.txt /tmp/requirements-sim.txt
RUN uv venv /opt/aerialsim/venv --python 3.10 \
    && uv pip install --python /opt/aerialsim/venv -r /tmp/requirements-sim.txt \
    && chmod -R a+rX /opt/aerialsim \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; } \
    && /opt/aerialsim/venv/bin/python -c "import cv2, msgpack, numpy; print('[image] sim stack OK')"

# ---- held-out scenes and the hidden pool ----
COPY fetch_assets.py /tmp/
RUN uv run --no-project --python 3.12 --with huggingface_hub==0.36.0 python /tmp/fetch_assets.py \
        scenes --repo "$SCENES_DATASET" --revision "$SCENES_REVISION" --maps $HELDOUT_MAPS \
        --out /opt/aerialsim/scenes
# the catalogue and pool builder come after the scenes: editing them rebuilds only the pool
COPY build_episode_pool.py targets.json /tmp/
RUN python3 /tmp/fetch_assets.py spawn-areas --url "$SPAWN_AREAS_URL" \
        --sha256 "$SPAWN_AREAS_SHA256" --out /tmp/map_spawnarea_info.json \
    && mkdir -p /opt/aerialsim/hidden \
    && python3 /tmp/build_episode_pool.py --spawn-areas /tmp/map_spawnarea_info.json \
         --targets /tmp/targets.json --maps $HELDOUT_MAPS --out /opt/aerialsim/hidden/pool.json \
    && rm -f /tmp/map_spawnarea_info.json /tmp/build_episode_pool.py /tmp/targets.json \
    && chown -R aerialsim:aerialsim /opt/aerialsim/scenes \
    && chmod 0700 /opt/aerialsim/scenes \
    && chown -R root:root /opt/aerialsim/hidden \
    && chmod 0700 /opt/aerialsim/hidden && chmod 0600 /opt/aerialsim/hidden/pool.json

# ---- the controller's Python and models (identical to the agent image) ----
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv -r /tmp/requirements.txt \
    && /opt/venv/bin/python3 /tmp/fetch_assets.py models --out /opt/models \
    && rm /tmp/fetch_assets.py \
    && chmod -R a+rX /opt/venv /opt/models \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 AERIAL_MODELS=/opt/models
ENV AERIALSIM_PYTHON=/opt/aerialsim/venv/bin/python

RUN mkdir -p /home/user/submission && chown -R user:user /home/user
ENV MPLBACKEND=Agg

RUN python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; print('[image] python3', sys.version.split()[0])" \
    && runuser -u user -- /opt/venv/bin/python3 -c "import numpy, PIL, torch, transformers; print('[image] agent-UID controller import OK')" \
    && runuser -u aerialsim -- /opt/aerialsim/venv/bin/python -c "import cv2, msgpack; print('[image] simulator-UID import OK')" \
    && ! runuser -u user -- ls /opt/aerialsim/scenes >/dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/aerialsim/hidden/pool.json >/dev/null 2>&1 \
    && echo "[image] held-out scenes and hidden pool are unreadable by the agent account"
