# The task's environment image. `image/` is the sole build context, and the final stage
# derives directly from an official ALE base. Nothing from verify/ or oracle/, and none of
# the hidden frames, is in here: the verify stage fetches those itself.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# open3d needs OpenGL and glib at import time even headless; grasp_nms is a C extension
# compiled at install time against Ubuntu 22.04's own Python 3.10 (hence python3.10-dev).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libgomp1 libglib2.0-0 build-essential ca-certificates curl python3.10-dev python3.10-venv \
 && rm -rf /var/lib/apt/lists/*

# Two interpreters, both world-traversable (setup and verify run as root, the agent and the
# oracle unprivileged):
#   /opt/venv   Python 3.12, the template's venv. The engine stages its `ale_verify` package
#               into whatever answers `python3` at verify time and refuses anything below
#               3.12, so this one stays first on PATH. It only needs requirements.txt.
#   /opt/grasp  Ubuntu's Python 3.10 with graspnetAPI 1.2.11 and the vision stack. graspnetAPI pins
#               numpy==1.20.3, which predates the numpy.typing that trimesh imports, and its
#               evaluation code needs numpy < 1.24, which has no Python 3.12 build. Its
#               dependencies are therefore pinned here and graspnetAPI itself is installed
#               with --no-deps; grasp_nms is compiled against that numpy. The agent's solver,
#               the practice grader and the scoring all run on this interpreter.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --python /opt/venv -r /tmp/requirements.txt \
    && uv venv /opt/grasp --python /usr/bin/python3.10 --seed \
    && /opt/grasp/bin/pip install --no-cache-dir "numpy==1.23.5" "Cython==3.0.11" "setuptools==69.5.1" "wheel==0.43.0" \
    && /opt/grasp/bin/pip install --no-cache-dir --no-build-isolation "grasp_nms==1.0.2" \
    && /opt/grasp/bin/pip install --no-cache-dir "torch==2.4.1" --index-url https://download.pytorch.org/whl/cpu \
         --extra-index-url https://pypi.org/simple \
    && /opt/grasp/bin/pip install --no-cache-dir \
         "open3d==0.18.0" "trimesh==4.4.9" "scipy==1.10.1" "scikit-learn==1.6.1" "scikit-image==0.21.0" \
         "opencv-python==4.10.0.84" "pillow==10.4.0" "matplotlib==3.7.5" "h5py==3.11.0" "cvxopt==1.3.2" \
         "dill==0.3.8" "tqdm==4.66.5" "transforms3d==0.4.2" "PyWavefront==1.3.3" \
         "autolab-core==1.1.1" "autolab-perception==1.0.0" "numpy==1.23.5" \
    && /opt/grasp/bin/pip install --no-cache-dir --no-deps --no-build-isolation "graspnetAPI==1.2.11" \
    && chmod -R a+rX /opt/venv /opt/grasp \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md):
# graspnet/, the agent-visible data, derived from GraspNet-1Billion (Fang et al., CVPR 2020, https://graspnet.net)
# and redistributed under its own non-commercial licence (CC BY-NC-SA 4.0; see NOTICE): 20 development scenes and 12
# practice scenes at 8 viewpoints each, meshes with signed-distance fields for the practice objects, and grasp
# annotations for ten development objects, as one archive and its manifest. The archive is unpacked into the
# root-owned /data/graspnet and is not staged into the agent home; the list holds nothing else, so the fetched
# folder is removed. None of the 26 objects in the hidden frames appears in it.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
 && mkdir -p /data/graspnet \
 && tar -xzf /opt/task-assets/payload/graspnet/agent_bundle.tar.gz -C /data/graspnet --strip-components=1 \
 && cp /opt/task-assets/payload/graspnet/agent_manifest.json /data/graspnet/manifest.json \
 && rm -rf /opt/task-assets/payload \
 && chown -R root:root /data/graspnet && chmod -R a+rX,go-w /data/graspnet

# Cut the bundle down to the annotated objects: no meshes and no practice-scene poses, so the
# benchmark evaluator has nothing to label against inside the sandbox (see prune_data.py).
COPY prune_data.py /tmp/prune_data.py
RUN /opt/grasp/bin/python /tmp/prune_data.py /data/graspnet && rm /tmp/prune_data.py \
 && chown -R root:root /data/graspnet && chmod -R a+rX,go-w /data/graspnet

# The declared artifact path must exist even when nobody works.
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

ENV MPLBACKEND=Agg
ENV GRASPNET_DATA=/data/graspnet

# Fail the build early if either interpreter is broken, and prove the AGENT account can use
# both: python3 must be the 3.12 venv, and the grasp stack must import with the pinned numpy.
RUN python3 -c "import sys, numpy; assert sys.version_info >= (3, 12); print('[image] python3', sys.version.split()[0], 'numpy', numpy.__version__)" \
    && runuser -u user -- python3 -c "import numpy; print('[image] agent-UID /opt/venv OK')" \
    && runuser -u user -- /opt/grasp/bin/python -c "import numpy, sklearn, open3d, cv2, torch, graspnetAPI; from graspnetAPI.utils.eval_utils import eval_grasp; assert numpy.__version__ == '1.23.5'; print('[image] agent-UID /opt/grasp OK')"
