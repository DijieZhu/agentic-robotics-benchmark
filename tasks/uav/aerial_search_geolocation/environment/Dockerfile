# The agent's environment: the simulator side (Python 3.10 with OpenCV and msgpack, which
# speaks to AirSim's RPC), two PRACTICE TravelUAV scenes (ModularPark, ModernCityMap) with
# their placement pool (from which practice missions are assembled), and the open-vocabulary
# models a controller may use offline. The held-out scenes the grading run uses are NOT
# here: they live only in the separate verification image.
#
# Build-time network is needed (scenes, spawn areas, models).
#
# The scenes, the placement areas and the models come from the benchmark's asset dataset,
# which mirrors them byte for byte from their public sources; assets.sha256 pins the dataset
# commit and every file's sha256 (docs/task-assets.md):
#   * scenes/: the five parts of the `closeloop_envs` archive of the Hugging Face dataset
#     wangxiangyu0814/TravelUAV_env at 44de5739;
#   * spawn_areas/map_spawnarea_info.json: AeroVLA's per-map placement areas
#     (github.com/XuPeng23/AeroVLA at 2d7ebd6, data/meta/);
#   * models/: openai/clip-vit-base-patch16 at 57c21647 and
#     google/owlv2-base-patch16-ensemble at cfd3195b (Hugging Face).
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

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

# ---- task assets: the fetcher and the asset list. Each step below fetches the files under
# one folder of the list (checked against their sha256) and leaves none of them in /tmp.
COPY fetch_assets.py assets.sha256 /tmp/task-assets/

# ---- practice scenes and their placement pool ----
COPY unpack_scenes.py /tmp/
RUN grep -E '^(#|[0-9a-f]{64}  scenes/)' /tmp/task-assets/assets.sha256 > /tmp/task-assets/scenes.sha256 \
    && python3 /tmp/task-assets/fetch_assets.py /tmp/task-assets/scenes.sha256 /tmp/task-assets/files \
    && python3 /tmp/unpack_scenes.py --archive /tmp/task-assets/files/scenes --maps $PRACTICE_MAPS \
         --out /opt/aerialsim/scenes \
    && rm -rf /tmp/task-assets/files /tmp/task-assets/scenes.sha256 /tmp/unpack_scenes.py
# the catalogue and pool builder come after the scenes: editing them rebuilds only the pool
COPY build_episode_pool.py targets.json /tmp/
RUN grep -E '^(#|[0-9a-f]{64}  spawn_areas/)' /tmp/task-assets/assets.sha256 > /tmp/task-assets/spawn_areas.sha256 \
    && python3 /tmp/task-assets/fetch_assets.py /tmp/task-assets/spawn_areas.sha256 /tmp/task-assets/files \
    && mv /tmp/task-assets/files/spawn_areas/map_spawnarea_info.json /tmp/map_spawnarea_info.json \
    && rm -rf /tmp/task-assets/files /tmp/task-assets/spawn_areas.sha256 \
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
    && grep -E '^(#|[0-9a-f]{64}  models/)' /tmp/task-assets/assets.sha256 > /tmp/task-assets/models.sha256 \
    && python3 /tmp/task-assets/fetch_assets.py /tmp/task-assets/models.sha256 /tmp/task-assets/files \
    && mv /tmp/task-assets/files/models /opt/models \
    && rm -rf /tmp/task-assets \
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
