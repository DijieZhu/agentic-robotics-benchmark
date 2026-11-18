# tensegrity_requirements_replica_from_photos's environment image (requirements stage, a three-strut tensegrity robot).
#
# `environment/` without its `setup/` folder is the sole build context and the final stage derives DIRECTLY from the
# official ALE base. Every instruction below is the merged bimanual requirements task's
# (tasks/arm/bimanual_requirements_replica_from_photos/environment/Dockerfile), byte for byte and in the same order:
# the chain's shared base block, its requirements.txt (byte-identical), the photo tools layer (requirements-photo.txt:
# the same two pins plus ezdxf) and the idle-timeout block. Two instructions are added after the photo layer: the
# agent home's config and cache folders are handed to the agent account, and the instruction's Tools line is checked
# as that account. No task material, no hidden data: the small givens are staged by environment/setup/setup.sh, the
# photo set comes from the task assets (assets.sha256, last layer), and the grader lives in tests/ (absent from the
# image).
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (owner decision Q12 / 05 B5: EGL enumerates 0 devices on the
# host and in Docker, so every sim render of the chain goes through MUJOCO_GL=osmesa); the X /
# GL client libraries OCP and MuJoCo link even when headless; libgomp for manifold3d / CoACD.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`) and
# verify (root) all use the same interpreter. UV_PYTHON_INSTALL_DIR must be world-readable too
# (the default /root/.local/share/uv is untraversable for the agent account).
#
# Python 3.12: the engine stages `ale_verify` into the `python3` it finds and refuses anything
# below 3.12 (engine-facts.md "Verification API"; drone_hover/image/Dockerfile). The ALE base
# ships a standalone CPython 3.12 at /opt/python, so uv links the venv against it.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
# --no-cache: the wheel cache would otherwise stay inside this layer (~1 GB); the image must
# stay lean (host / is at 99 %, owner decision Q12: Docker data-root move deferred).
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements.txt \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

# The declared artifact path must exist even when nobody works (the engine captures declared
# artifacts after every episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# Headless defaults. MUJOCO_GL=osmesa is the chain's rendering path (CHAIN_PLAN.md section 4.0:
# "Sim renders through OSMesa inside the image"); a task may override to egl where a GPU exists.
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# Fail the build early if the toolchain is broken, prove the AGENT account can use the venv, and
# prove an offscreen OSMesa render works. One import line per -c: a continuation line starting
# with `from` reads as a Dockerfile FROM to the final-stage lint.
RUN python3 -c "import mujoco, mink, pinocchio, coacd, trimesh, manifold3d, gymnasium, numpy, scipy, jsonschema, yaml, PIL, matplotlib, cadquery, OCP; print('[arb-base-mujoco] import OK mujoco', mujoco.__version__, 'cadquery', cadquery.__version__)" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); print('[arb-base-mujoco] OSMesa render OK', px.shape, int(px.max()))" \
    && runuser -u user -- python3 -c "import mujoco, mink, cadquery, coacd; print('[arb-base-mujoco] agent-UID import OK')"

# ---- task-specific layers (tensegrity_requirements_replica_from_photos) ---------------------------------------------
# Photogrammetry / image tools for the agent, as in the bimanual requirements task: OpenCV (headless) and scikit-image,
# pinned against the base's numpy 2.5.3; plus ezdxf for the two sensor cut-outline drawings (DXF).
# Only the NEW files are touched after the install (a chmod -R over the whole venv would copy the base's 2 GB venv
# into this layer): the marker file bounds the find.
COPY requirements-photo.txt /tmp/requirements-photo.txt
RUN touch /tmp/.before-photo-layer && sleep 1 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements-photo.txt \
    && find /opt/venv -newer /tmp/.before-photo-layer -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && find /opt/venv -newer /tmp/.before-photo-layer \( -type d -o -type f \) -exec chmod a+rX {} + \
    && rm -f /tmp/.before-photo-layer
RUN runuser -u user -- python3 -c "import cv2, skimage, numpy, scipy, mujoco, yaml, jsonschema; print('[arm-req-photos] agent-UID import OK cv2', cv2.__version__, 'skimage', skimage.__version__)"
# The base block's root-run checks (python3 importing matplotlib and cadquery with HOME=/home/user) and uv leave
# /home/user/.config/matplotlib, /home/user/.cache/ezdxf and /home/user/.cache/uv owned by root: hand them to the
# agent account, so that its matplotlib can keep a font cache instead of warning and rebuilding it on every import.
RUN chown -R user:user /home/user/.config /home/user/.cache
# The instruction's Tools line, checked as the agent account in a login shell (the harness launches the agent's
# commands with `bash -lc` and keeps the image's PATH): python3 is /opt/venv's 3.12, every listed module imports, ezdxf
# creates a document, and no OCR engine (tesseract) or OCR module is installed.
RUN runuser -u user -- bash -lc 'test "$(command -v python3)" = /opt/venv/bin/python3 && ! command -v tesseract >/dev/null' \
    && runuser -u user -- bash -lc "python3 -c \"import sys, numpy, scipy, cv2, skimage, matplotlib, PIL, yaml, jsonschema, trimesh, ezdxf; assert sys.version_info[:2] == (3, 12); d = ezdxf.new(); d.modelspace().add_line((0, 0), (1, 0)); print('[tensegrity-req] agent tools OK python', sys.version.split()[0], 'ezdxf', ezdxf.__version__, 'cv2', cv2.__version__)\"" \
    && runuser -u user -- python3 -c "import importlib.util as u; bad = [m for m in ('pytesseract', 'tesserocr', 'easyocr', 'paddleocr', 'rapidocr_onnxruntime', 'doctr') if u.find_spec(m)]; assert not bad, bad; print('[tensegrity-req] no OCR module')"

# CHAIN_PLAN Q15 workaround (2026-09-27; docs/plan/arm/eval/engine-patches/Q15-request-reissue.md section 6). The
# claude-code harness aborts a model stream after 300 s without a byte (Bun's HTTP idle timeout and the CLI's stream
# watchdog); both timers are raised to the gateway's own 1800 s upstream ceiling. ENV is inherited by every process
# the guest service starts, and the profile.d file covers the login shell the CLI is launched with (`bash -lc`).
# Nothing else in this image reads them. Appended after every toolchain layer so the layers above stay cached.
ENV BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
RUN printf '%s\n' 'export BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800' 'export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000' \
        > /etc/profile.d/99-ale-idle-timeouts.sh \
    && chmod 0644 /etc/profile.d/99-ale-idle-timeouts.sh

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md)
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets
