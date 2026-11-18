# The environment for driving_safety_supervisor, used by the agent phase AND by the
# separate verification sandbox (task.toml reuses this image there). Nothing in it is
# secret: the grading run's secret is its episode seeds, drawn in the grader's memory.
#
#   * CARLA 0.9.15 (packaged release; simulator code MIT, assets CC-BY; Unreal Engine 4 under
#     Epic's terms), fetched at build time. Base towns Town01-05 and Town10HD.
#   * /opt/carla/venv: Python 3.10 with the CARLA client and what CARLA's PythonAPI `agents`
#     package needs (the nominal planner is its stock BehaviorAgent).
#   * /opt/venv: Python 3.12 for the supervisor (torch, transformers, OpenCV) and two
#     open-vocabulary models in /opt/models, usable offline.
#
# The CARLA release and the model weights come from the benchmark's asset dataset, which
# mirrors them byte for byte from their sources; assets.sha256 pins the dataset commit and
# every file's sha256 (docs/task-assets.md):
#   * carla/CARLA_0.9.15.tar.gz: downloads.carlasim.com/Linux/CARLA_0.9.15.tar.gz;
#   * models/: openai/clip-vit-base-patch16 at 57c21647 and
#     google/owlv2-base-patch16-ensemble at cfd3195b (Hugging Face);
#   * torchvision/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth: torchvision's COCO weights
#     (download.pytorch.org/models/).
#
# CARLA refuses to run as root: the agent runs it as `user`; the grader runs it as
# `carlasim`, an account that is neither root nor the agent.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_NO_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates libvulkan1 vulkan-tools libomp5 xdg-user-dirs fontconfig \
        libpng16-16 libjpeg8 libtiff5 libsdl2-2.0-0 libxrandr2 libxinerama1 libxcursor1 \
        libxi6 libsm6 libice6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /var/lib/carlasim --shell /usr/sbin/nologin carlasim

# ---- task assets: the fetcher and the asset list. Each step below fetches the files under
# one folder of the list (checked against their sha256) and leaves none of them in /tmp.
COPY fetch_assets.py assets.sha256 /tmp/task-assets/

# ---- CARLA server ----
RUN mkdir -p /opt/carla/server \
    && grep -E '^(#|[0-9a-f]{64}  carla/)' /tmp/task-assets/assets.sha256 > /tmp/task-assets/carla.sha256 \
    && python3 /tmp/task-assets/fetch_assets.py /tmp/task-assets/carla.sha256 /tmp/task-assets/files \
    && tar -xz -C /opt/carla/server -f /tmp/task-assets/files/carla/CARLA_0.9.15.tar.gz \
    && rm -rf /tmp/task-assets/files /tmp/task-assets/carla.sha256 \
    && rm -rf /opt/carla/server/Import /opt/carla/server/PythonAPI/examples \
    && mkdir -p /opt/carla/server/CarlaUE4/Saved \
    && chmod 1777 /opt/carla/server/CarlaUE4/Saved \
    && chmod -R a+rX /opt/carla/server \
    && test -x /opt/carla/server/CarlaUE4.sh

# ---- simulator-side Python (3.10) ----
COPY requirements-sim.txt /tmp/requirements-sim.txt
RUN uv venv /opt/carla/venv --python 3.10 \
    && uv pip install --python /opt/carla/venv -r /tmp/requirements-sim.txt \
    && chmod -R a+rX /opt/carla/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; } \
    && /opt/carla/venv/bin/python -c "import sys; sys.path.insert(0, '/opt/carla/server/PythonAPI/carla'); import carla; from agents.navigation.behavior_agent import BehaviorAgent; print('[image] CARLA client + agents OK')"

# ---- supervisor-side Python (3.12) and models ----
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
# The models: CLIP and OWLv2 as their Hub snapshots (`from_pretrained(<path>)`), and
# torchvision's COCO Faster R-CNN (ResNet-50 FPN v2) state dict re-saved under a plain name,
# loadable with `fasterrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
# .load_state_dict(torch.load(...))`.
COPY requirements.txt /tmp/
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv -r /tmp/requirements.txt \
    && grep -E '^(#|[0-9a-f]{64}  (models|torchvision)/)' /tmp/task-assets/assets.sha256 > /tmp/task-assets/models.sha256 \
    && python3 /tmp/task-assets/fetch_assets.py /tmp/task-assets/models.sha256 /tmp/task-assets/files \
    && mv /tmp/task-assets/files/models /opt/models \
    && /opt/venv/bin/python3 -c "import torch; torch.save(torch.load('/tmp/task-assets/files/torchvision/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth', weights_only=False), '/opt/models/fasterrcnn_resnet50_fpn_v2_coco.pth')" \
    && rm -rf /tmp/task-assets \
    && chmod -R a+rX /opt/venv /opt/models \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
ENV DRIVE_SIM_PYTHON=/opt/carla/venv/bin/python

RUN mkdir -p /home/user/submission && chown -R user:user /home/user
ENV MPLBACKEND=Agg

RUN python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; print('[image] python3', sys.version.split()[0])" \
    && runuser -u user -- /opt/venv/bin/python3 -c "import numpy, PIL, torch, torchvision, transformers; print('[image] agent-UID supervisor import OK')" \
    && runuser -u user -- /opt/carla/venv/bin/python -c "import carla; print('[image] agent-UID CARLA client OK')" \
    && runuser -u carlasim -- /opt/carla/venv/bin/python -c "import carla; print('[image] simulator-UID CARLA client OK')"
