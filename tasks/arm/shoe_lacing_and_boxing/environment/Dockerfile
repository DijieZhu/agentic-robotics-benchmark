# shoe_lacing_and_boxing's environment image. `environment/` (less its `setup/`) is the sole build context and the final
# stage derives DIRECTLY from the official ALE base, at an exact version. Nothing hidden is here: the givens are staged
# by environment/setup/setup.sh, the grader and the reference program live in tests/ and solution/ (absent from the
# image).
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (no GPU in the task's resources; EGL enumerates no device in the sandbox), the GL
# client libraries MuJoCo links even when headless, libgomp for numpy/scipy threads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`) and verify (root) all use
# the same interpreter. UV_PYTHON_INSTALL_DIR must be world-readable too (the default /root/.local/share/uv is
# untraversable for the agent account). Python 3.12: the engine stages `ale_verify` into the `python3` it finds and
# refuses anything below 3.12. The base image sets HOME=/home/user for root too, so the build points uv's cache away
# from the agent home, which must not hold root-owned entries the agent can neither write nor remove.
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

# The robot model: MuJoCo Menagerie's Franka Emika Panda (folder franka_emika_panda, Apache-2.0, LICENSE inside),
# fetched at the commit pinned in panda_assets.sha256 (a sparse, blob-filtered fetch: only that folder's files are
# downloaded) and checked file by file against the sha256 recorded there; a missing, extra or changed file fails the
# build. obj2stl.py then turns the visual OBJ meshes into binary STL (same geometry, about half the size) and rewrites
# the mesh references, and the result is checked file by file against panda_converted.sha256, so the cell, the givens
# and tests/given_hashes.json see exactly the model this task was built with. environment/setup/setup.sh stages it
# into the agent home (~/given/panda) and into the grader's root-owned copy.
COPY panda_assets.sha256 panda_converted.sha256 obj2stl.py /opt/task-assets/build/
RUN set -eu; \
    build=/opt/task-assets/build; \
    list="$build/panda_assets.sha256"; \
    repo="$(sed -n 's/^# repo: *//p' "$list")"; \
    commit="$(sed -n 's/^# commit: *//p' "$list")"; \
    folder="$(sed -n 's/^# folder: *//p' "$list")"; \
    test -n "$repo"; test -n "$commit"; test -n "$folder"; \
    work="$(mktemp -d)"; \
    git -C "$work" init -q; \
    git -C "$work" remote add origin "$repo"; \
    git -C "$work" config core.sparseCheckout true; \
    echo "/$folder/" > "$work/.git/info/sparse-checkout"; \
    for try in 1 2 3; do \
        if git -C "$work" fetch -q --depth 1 --filter=blob:none origin "$commit" \
           && git -C "$work" -c advice.detachedHead=false checkout -q FETCH_HEAD; then break; fi; \
        if [ "$try" = 3 ]; then echo "[image] fetching $repo@$commit failed" >&2; exit 1; fi; \
        sleep 15; \
    done; \
    test "$(git -C "$work" rev-parse HEAD)" = "$commit"; \
    cd "$work/$folder"; \
    grep -v '^#' "$list" | sha256sum --check --strict --quiet; \
    test "$(find . -type f | wc -l)" -eq "$(grep -vc '^#' "$list")"; \
    test -z "$(find . ! -type f ! -type d)"; \
    cd /; \
    rm -rf /opt/task-assets/panda; \
    HOME=/root /opt/venv/bin/python3 -I -B "$build/obj2stl.py" "$work/$folder" /opt/task-assets/panda; \
    rm -rf "$work"; \
    cd /opt/task-assets/panda; \
    grep -v '^#' "$build/panda_converted.sha256" | sha256sum --check --strict --quiet; \
    test "$(find . -type f | wc -l)" -eq "$(grep -vc '^#' "$build/panda_converted.sha256")"; \
    cd /; \
    rm -rf "$build"; \
    chown -R root:root /opt/task-assets; \
    chmod -R u+rwX,go+rX,go-w /opt/task-assets; \
    echo "[image] $folder @ $commit: converted, $(find /opt/task-assets/panda -type f | wc -l) files verified"

# The agent cameras' colour images (tests/shoecell/look.py): MuJoCo's Filament renderer (in the mujoco wheel) on Vulkan;
# Mesa's lavapipe is the CPU Vulkan driver (no GPU in the task's resources). Their depth stays on OSMesa.
RUN apt-get update && apt-get install -y --no-install-recommends mesa-vulkan-drivers libvulkan1 \
    && rm -rf /var/lib/apt/lists/*

# ---- task assets: the agent cameras' render bundle (render/shoecell_render_v2.tar in the benchmark's Hugging Face asset
# repository: the factory hall and its props from the approved OVRTX scene, the sneaker skin template, the cell's
# textures, the image light; one tar, as one request per file runs into the Hub's rate limit), fetched at the dataset
# commit pinned in assets.sha256 and checked by its sha256 (docs/task-assets.md; fetch_assets.py is the benchmark's
# canonical copy, shared/task_assets/), unpacked root-owned and world-readable to /opt/task-assets/render, where the
# cell's renderer reads it and checks every file against the bundle's manifest. Nothing of it is a given: the agent's
# cell and the grader's (a separate verifier sandbox from this image) read the same folder.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN set -eu; \
    python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload; \
    mkdir -p /opt/task-assets/render; \
    tar -xf /opt/task-assets/payload/render/shoecell_render_v2.tar -C /opt/task-assets/render --no-same-owner; \
    rm -rf /opt/task-assets/payload; \
    chown -R root:root /opt/task-assets/render; \
    chmod -R u+rwX,go+rX,go-w /opt/task-assets/render; \
    cd /opt/task-assets/render; \
    python3 -c "import hashlib, json; m = json.load(open('manifest.json')); bad = [p for p, h in m['files'].items() if hashlib.sha256(open(p, 'rb').read()).hexdigest() != h]; assert not bad, bad; print('[image] render bundle:', len(m['files']), 'files verified')"

# The declared artifact path must exist even when nobody works (the engine captures declared artifacts after every
# episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# Headless defaults: the classic renders of the cell (the cameras' depth) go through OSMesa.
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
ENV OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 LP_NUM_THREADS=2

# Fail the build early if the toolchain is broken, prove the AGENT account can use the venv, prove an offscreen OSMesa
# render works and that the robot model compiles, prove a Filament context on lavapipe; finally, nothing in the agent
# home is root-owned. One import line per -c: a continuation line starting with `from` reads as a Dockerfile FROM to the
# final-stage lint.
RUN export HOME=/root \
    && python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; import mujoco, numpy, scipy, PIL, cv2; print('[shoe-cell] import OK mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__)" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_path('/opt/task-assets/panda/panda.xml'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); print('[shoe-cell] OSMesa render OK', px.shape, int(px.max()), 'panda nbody', m.nbody)" \
    && runuser -u user -- python3 -c "import mujoco, numpy, scipy, PIL, cv2; m = mujoco.MjModel.from_xml_path('/opt/task-assets/panda/panda_nohand.xml'); print('[shoe-cell] agent-UID import OK, panda_nohand nbody', m.nbody)" \
    && env -u DISPLAY python3 -c "import mujoco._render_filament as F, mujoco.experimental.studio.window; c = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN)); print('[shoe-cell] Filament on Vulkan (lavapipe) OK')" \
    && test -z "$(find /home/user -user root -print -quit)"
