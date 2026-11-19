# Environment image for keyed_nut_peg_assembly. Final stage derives from the official ALE base.
# Contains the simulator stack, the public robot assets and the commissioning service (the cell's
# simulator on its development kits, readable by its own account only). The grader, the reference
# solution, the hidden episodes and the reference value are never copied into the image (the build
# context is environment/ less setup/).
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

# Python 3.12 venv owned by root, readable and traversable by every account, first on PATH for every
# stage (root stages then run a venv interpreter with user site disabled).
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

# Software rendering for MuJoCo's own OpenGL renderer (Mesa llvmpipe through EGL, CPU only). The cell's cameras no
# longer use it (see the next step); it stays for depth and segmentation checks of the review tools.
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libgles2 libegl-mesa0 libgl1-mesa-dri \
    && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MPLBACKEND=Agg OMP_NUM_THREADS=1 LP_NUM_THREADS=2

# The cameras' renderer (keyed_render.py beside keyed_env.py): MuJoCo's Filament module (in the mujoco wheel) on
# Vulkan; Mesa's lavapipe is the CPU Vulkan driver (the task has no GPU).
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md):
# render/ (the cameras' render bundle: room.npz, room.json, ibl.ktx; keyed_render.py reads it in place). It is part
# of the cell's simulator, not of the agent's files: nothing copies it into the agent's home, and further down it is
# made readable by root (the grader) and by the commissioning service's account only.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload
ENV KEYED_RENDER_DIR=/opt/task-assets/payload/render

# The declared artifact path exists even when nobody works.
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# No job scheduler may run agent-scheduled work outside the graded solver tree.
RUN (apt-get purge -y cron anacron at >/dev/null 2>&1 || true) \
    && rm -rf /etc/cron.d /etc/cron.daily /etc/cron.hourly /var/spool/cron /var/spool/at \
    && ! command -v crond && ! command -v cron && ! command -v atd

# Build-time checks: the venv works for root and for the agent account, robosuite's Panda assets are
# present (robosuite itself is never imported), and MuJoCo renders offscreen.
RUN python3 -c "import numpy, scipy, cv2, mujoco; print('[image] mujoco', mujoco.__version__)" \
    && python3 -c "import importlib.util, os; s = importlib.util.find_spec('robosuite'); p = os.path.join(list(s.submodule_search_locations)[0], 'models', 'assets', 'robots', 'panda', 'robot.xml'); assert os.path.isfile(p), p" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom size=\"1 1 .1\" type=\"plane\"/><camera name=\"c\" pos=\"0 0 1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 32, 32); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d, 'c'); assert r.render().mean() > 10; print('[image] offscreen render OK')" \
    && python3 -c "import site, sys; assert sys.prefix != sys.base_prefix" \
    && runuser -u user -- /opt/venv/bin/python3 -c "import numpy, scipy, cv2, mujoco; print('[image] agent-UID import OK')"

# The commissioning service: the cell on its development kits, for the agent's commissioning window
# (commission_service/: the service program, the cell's simulator keyed_env.py, a byte copy of the
# grader's, and the development kits with the window's size), installed for its own account `keyedcell`
# only: mode 700, so the agent account can neither read nor list it. environment/setup/setup.sh starts
# it before the agent's session (and restarts it should it ever exit); the agent reaches it through a
# unix socket and gets the public wire's observations and how an episode ended: no ground truth, no
# score, no completion verdict.
RUN useradd --system --no-create-home --home-dir /opt/keyedcell-commission --shell /usr/sbin/nologin keyedcell
COPY commission_service /opt/keyedcell-commission
RUN find /opt/keyedcell-commission -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && rm -rf /opt/keyedcell-commission/state \
    && mkdir -p /opt/keyedcell-commission/state \
    && chown -R keyedcell:keyedcell /opt/keyedcell-commission \
    && chmod -R u+rwX,go-rwx /opt/keyedcell-commission

# The render bundle: root and the commissioning service's account read it, the agent account does not.
RUN chown -R root:keyedcell /opt/task-assets && chmod -R u=rwX,g=rX,o= /opt/task-assets \
    && ! runuser -u user -- ls /opt/task-assets > /dev/null 2>&1

# The commissioning service works for its own account (the cell loads, lays a development kit out and
# draws its three cameras through keyed_render.py: this proves the image's Vulkan driver for that account),
# and the agent account can neither list nor read it. One import line
# per -c: a continuation line starting with `from` reads as a Dockerfile FROM to the final-stage lint.
RUN runuser -u keyedcell -- env -i HOME=/opt/keyedcell-commission/state TMPDIR=/opt/keyedcell-commission/state \
        PATH=/opt/venv/bin:/usr/bin:/bin LANG=C.UTF-8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 \
        LP_NUM_THREADS=2 MESA_SHADER_CACHE_DISABLE=true PYTHONNOUSERSITE=1 \
        /opt/venv/bin/python3 -B -s -c "import json, sys; sys.path.insert(0, '/opt/keyedcell-commission'); import commissiond, keyed_env as E; k = commissiond.KITS['kits'][0]; env = E.NutEnv(k['kit']['nuts']); obs = env.reset(int(k['layout_seed']), k['kit']['order']); assert obs['wrist'].shape == (192, 256, 3) and obs['left'].shape == (240, 320, 3) and min(obs[c].mean() for c in E.CAMERA_NAMES) > 20 and min(obs[c].std() for c in E.CAMERA_NAMES) > 10; assert env.max_steps == 850 and env.unit is not None; env.close(); print('[image] commissioning service: the cell loads and renders for its account,', len(commissiond.KITS['kits']), 'development kits')" \
    && rm -rf /opt/keyedcell-commission/state && mkdir -p /opt/keyedcell-commission/state \
    && chown keyedcell:keyedcell /opt/keyedcell-commission/state && chmod 700 /opt/keyedcell-commission/state \
    && ! runuser -u user -- ls /opt/keyedcell-commission > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/keyedcell-commission/keyed_env.py > /dev/null 2>&1 \
    && ! runuser -u user -- cat /opt/keyedcell-commission/kits.json > /dev/null 2>&1 \
    && test -z "$(find /opt/keyedcell-commission ! -user keyedcell -print -quit)" \
    && test -z "$(find /opt/keyedcell-commission -perm /077 -print -quit)"

# The same cameras work for root, the grader's account (the grader's copy of the cell arrives with tests/ at verify;
# the service's byte copy stands in for it here), and a second drawing of the same state is the same bytes.
RUN env HOME=/tmp MESA_SHADER_CACHE_DISABLE=true /opt/venv/bin/python3 -B -s -c "import json, sys; sys.path.insert(0, '/opt/keyedcell-commission'); import keyed_env as E; k = json.load(open('/opt/keyedcell-commission/kits.json'))['kits'][0]; env = E.NutEnv(k['kit']['nuts'], render=False); env.reset(int(k['layout_seed']), k['kit']['order']); v = E._KR.open_view(env); a = [v.render(c) for c in E.CAMERA_NAMES]; b = [v.render(c) for c in E.CAMERA_NAMES]; assert all((x == y).all() for x, y in zip(a, b)) and min(x.std() for x in a) > 10; env.close(); print('[image] the cameras render for root; a state drawn twice is the same bytes')" \
    && find /opt/keyedcell-commission -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && test -z "$(find /opt/keyedcell-commission ! -user keyedcell -print -quit)"

# The engine's root helpers (the guest service that copies the declared artifact out of this sandbox when
# the agent stops) run in the image's working directory, and `python3 -c` puts that directory first on
# sys.path. The base image's is /home/user, which the agent can write: a module the agent left there
# would run as root and could read the commissioning service's files. A root-owned working directory
# closes that.
WORKDIR /
