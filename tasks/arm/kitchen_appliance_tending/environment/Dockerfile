# Task image. Nothing here depends on tests/, solution/, the anchor or the hidden units: an image layer is readable by
# anyone who can run the image. The hidden units' models are NOT in the image (tests/units, environment/setup/ref and
# solution/oracle only). The public task assets (the cameras' room bundle and the practice units' models) are fetched
# below; environment/setup/setup.sh stages the practice units into the agent's home.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONNOUSERSITE=1
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir --no-deps robosuite==1.5.2 \
    && rm -f /tmp/requirements.txt \
    && chmod -R a+rX /opt/venv

# MuJoCo's classic renderer (mujoco.Renderer) through Mesa OSMesa (llvmpipe, no display or device needed), checked below
# as the agent user. The task's cameras do not use it; it stays for anything else an agent draws in its dev sim.
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 libgl1 \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa MPLBACKEND=Agg OMP_NUM_THREADS=1 LP_NUM_THREADS=2

# The cameras' renderer (kitchen_render.py beside appliance_env.py): MuJoCo's Filament module (in the mujoco wheel) on
# Vulkan; Mesa's lavapipe is the CPU Vulkan driver (the task has no GPU). The renderer sets its own lavapipe thread
# count (kitchen_look.json); LP_NUM_THREADS above is for OSMesa.
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md):
# render/ (the room bundle: room.npz, room.json, ibl.ktx, sky.ktx; the cameras' renderer reads it in place) and units/
# (the practice units' models, which setup.sh copies into the agent's home)
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets
ENV KITCHEN_RENDER_DIR=/opt/task-assets/payload/render

RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# The lab service's own account (environment/setup/lab/alelab/server.py drops to it).
RUN useradd --system --user-group --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin alelab

# No job schedulers: nothing the agent leaves may run later as anyone.
RUN (apt-get purge -y cron anacron at >/dev/null 2>&1 || true) \
    && rm -rf /etc/cron.d /etc/cron.daily /etc/cron.hourly /var/spool/cron /var/spool/at \
    && ! command -v crond && ! command -v cron && ! command -v atd

RUN python3 -c "import numpy, scipy, cv2, mujoco; print('[image] mujoco', mujoco.__version__)" \
    && python3 -c "import importlib.util, os; s = importlib.util.find_spec('robosuite'); p = os.path.join(list(s.submodule_search_locations)[0], 'models', 'assets', 'robots', 'panda', 'robot.xml'); assert os.path.isfile(p), p" \
    && python3 -c "import site, sys; assert sys.prefix != sys.base_prefix" \
    && runuser -u user -- /opt/venv/bin/python3 -c "import numpy, scipy, cv2, mujoco; print('[image] agent-UID import OK')" \
    && runuser -u user -- env HOME=/tmp /opt/venv/bin/python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"sphere\" size=\".1\"/></worldbody></mujoco>'); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r = mujoco.Renderer(m, 48, 64); r.update_scene(d); im = r.render(); assert im.max() > 0; print('[image] agent-UID MuJoCo OSMesa render OK', im.shape)"

# The cameras' renderer works for every account that draws images: root (the grader), the agent account (the dev sim,
# the practice grader) and the lab's account (the lab worker starts rendering as root and serves as alelab, with the
# lab's clean environment). Each draws the three images of practice unit p0 from the room bundle through the task's own
# modules (render_check/: copies of appliance_env.py, kitchen_render.py, kitchen_fx.py, kitchen_scene.py and
# kitchen_look.json, mounted for this step only, so nothing of them stays in the image).
RUN --mount=type=bind,source=render_check,target=/tmp/render_check \
    MESA_SHADER_CACHE_DISABLE=true python3 -I -B /tmp/render_check/check_render.py \
    && runuser -u user -- env HOME=/tmp MESA_SHADER_CACHE_DISABLE=true /opt/venv/bin/python3 -I -B /tmp/render_check/check_render.py \
    && runuser -u alelab -- env -i PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 HOME=/nonexistent \
       OMP_NUM_THREADS=1 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa LP_NUM_THREADS=2 MESA_SHADER_CACHE_DISABLE=true \
       /opt/venv/bin/python3 -I -B /tmp/render_check/check_render.py
