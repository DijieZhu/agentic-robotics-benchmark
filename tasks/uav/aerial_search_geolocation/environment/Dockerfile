# The agent's environment: the simulator side (Python 3.10 with OpenCV and msgpack, which
# speaks to AirSim's RPC), two PRACTICE TravelUAV scenes (ModularPark, ModernCityMap) with
# their placement pool (from which practice missions are assembled), and the open-vocabulary
# models a controller may use offline. The held-out scenes the grading run uses are NOT
# here: they live only in the separate verification image.
#
# Build-time network is needed (scenes, spawn areas, models). Every source is public and
# ungated, pinned to a revision.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

ARG SCENES_DATASET=wangxiangyu0814/TravelUAV_env
ARG SCENES_REVISION=44de5739a95a2f6a88767446421cddada9606642
ARG SPAWN_AREAS_URL=https://raw.githubusercontent.com/XuPeng23/AeroVLA/2d7ebd60e36ce12c3c0898f7b7a4ad52c1315e31/data/meta/map_spawnarea_info.json
ARG SPAWN_AREAS_SHA256=120438c62c6fa286c8cf44213dd03f5b7cd863e33889ae21c913ab5f68b530e6
ARG PRACTICE_MAPS="ModularPark ModernCityMap"

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_NO_CACHE=1

# Unreal Engine (offscreen Vulkan) runtime libraries and 7-Zip for the split scene archive.
RUN apt-get update && apt-get install -y --no-install-recommends \
        p7zip-full libvulkan1 vulkan-tools libgl1 libglu1-mesa libxrandr2 libxinerama1 \
        libxcursor1 libxi6 libxt6 libnss3 libasound2 libpulse0 libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ---- simulator side: Python 3.10 with OpenCV and msgpack ----
COPY requirements-sim.txt /tmp/requirements-sim.txt
RUN uv venv /opt/aerialsim/venv --python 3.10 \
    && uv pip install --python /opt/aerialsim/venv -r /tmp/requirements-sim.txt \
    && chmod -R a+rX /opt/aerialsim \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; } \
    && /opt/aerialsim/venv/bin/python -c "import cv2, msgpack, numpy; print('[image] sim stack OK')"

# ---- practice scenes and their placement pool ----
COPY fetch_assets.py /tmp/
RUN uv run --no-project --python 3.12 --with huggingface_hub==0.36.0 python /tmp/fetch_assets.py \
        scenes --repo "$SCENES_DATASET" --revision "$SCENES_REVISION" --maps $PRACTICE_MAPS \
        --out /opt/aerialsim/scenes
# the catalogue and pool builder come after the scenes: editing them rebuilds only the pool
COPY build_episode_pool.py targets.json /tmp/
RUN python3 /tmp/fetch_assets.py spawn-areas --url "$SPAWN_AREAS_URL" \
        --sha256 "$SPAWN_AREAS_SHA256" --out /tmp/map_spawnarea_info.json \
    && mkdir -p /opt/aerialsim/practice \
    && python3 /tmp/build_episode_pool.py --spawn-areas /tmp/map_spawnarea_info.json \
         --targets /tmp/targets.json --maps $PRACTICE_MAPS --out /opt/aerialsim/practice/pool.json \
    && rm -f /tmp/map_spawnarea_info.json /tmp/build_episode_pool.py /tmp/targets.json \
    && chown -R user:user /opt/aerialsim/scenes \
    && chmod -R a+rX /opt/aerialsim

# ---- the controller's Python: 3.12 with torch + transformers, and offline models ----
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
    && runuser -u user -- /opt/aerialsim/venv/bin/python -c "import cv2, msgpack; print('[image] agent-UID simulator import OK')" \
    && test -s /opt/aerialsim/practice/pool.json
