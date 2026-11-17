# quadruped_cad_design_from_photos's environment image (hardware-design stage of a 12-DoF quadruped, FreeCAD MCP).
#
# `environment/` without its `setup/` folder is the sole build context and the final stage derives DIRECTLY from the
# official ALE base. The block down to the venv self-check is the chain's shared base
# `shared/images/arb-base-mujoco/Dockerfile` reproduced verbatim (same layers, same requirements.txt bytes): the grader
# needs it. Task layers: FreeCAD 1.1.4, the openbox desktop that runs FreeCAD only, the FreeCAD MCP server, the Q15
# idle-timeout ENV. No task material, no hidden data: the givens are staged by environment/setup/setup.sh, the grader
# lives in tests/ (absent from the image).
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        libxcursor1 libxinerama1 libxrandr2 libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
RUN uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements.txt \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; }

RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

RUN python3 -c "import mujoco, mink, pinocchio, coacd, trimesh, manifold3d, gymnasium, numpy, scipy, jsonschema, yaml, PIL, matplotlib, cadquery, OCP; print('[arb-base-mujoco] import OK mujoco', mujoco.__version__, 'cadquery', cadquery.__version__)" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); print('[arb-base-mujoco] OSMesa render OK', px.shape, int(px.max()))" \
    && runuser -u user -- python3 -c "import mujoco, mink, cadquery, coacd; print('[arb-base-mujoco] agent-UID import OK')"

# ---- task-specific layers (the same instructions as the arm CAD task's image, so the layers are shared) ---------------
# FreeCAD 1.1.4, the release AppImage pinned by sha256 and extracted (containers have no FUSE). The GUI is the agent's
# tool; the same tree's headless `freecadcmd` is the grader's document reader. Workbenches that need the network or
# external programs (Addon Manager, CAM post-processors, OpenSCAD, Robot, Test, Web, BIM, Plot) are removed; FEM stays
# (the MCP addon imports it).
ARG FREECAD_URL=https://github.com/FreeCAD/FreeCAD/releases/download/1.1.4/FreeCAD_1.1.4-Linux-x86_64-py311.AppImage
ARG FREECAD_SHA256=f6dc6ba676e5ac96a565ebc8d657232f94c6158e85b4352141bd1a46f6b43434
RUN curl -fsSL -o /tmp/FreeCAD.AppImage "$FREECAD_URL" \
    && echo "$FREECAD_SHA256  /tmp/FreeCAD.AppImage" | sha256sum -c - \
    && cd /tmp && chmod +x FreeCAD.AppImage && ./FreeCAD.AppImage --appimage-extract >/dev/null \
    && mv /tmp/squashfs-root /opt/freecad && rm -f /tmp/FreeCAD.AppImage \
    && cd /opt/freecad/usr/Mod && rm -rf AddonManager CAM OpenSCAD Robot Test Web BIM Plot Idf \
    && chown -R root:root /opt/freecad && chmod -R a+rX,go-w /opt/freecad

# The FreeCAD MCP server (neka-nat/freecad-mcp, MIT, vendored at a pinned commit: vendor/freecad-mcp/VENDORED.md): its
# addon serves the RPC calls inside the GUI session (started with FreeCAD, local connections only), its MCP server
# (/opt/freecad-mcp) is the task's declared stdio MCP server (task.toml, [[environment.mcp_servers]] `freecad`).
COPY vendor/freecad-mcp /opt/freecad-mcp-src
RUN cp -r /opt/freecad-mcp-src/addon/FreeCADMCP /opt/freecad/usr/Mod/FreeCADMCP \
    && cd /opt/freecad-mcp-src && VIRTUAL_ENV= UV_PROJECT_ENVIRONMENT=/opt/freecad-mcp uv sync --frozen --no-dev --no-cache --python 3.12 \
    && find /opt/freecad-mcp /opt/freecad-mcp-src -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX,go-w /opt/freecad-mcp /opt/freecad-mcp-src /opt/freecad/usr/Mod/FreeCADMCP \
    && /opt/freecad-mcp/bin/python -c "import freecad_mcp.server; print('[arm-links-freecad] freecad-mcp import OK')"

# The session's initial preferences and the MCP addon's settings (auto-start the RPC server, local only).
COPY freecad/user.cfg /home/user/.config/FreeCAD/v1-1/user.cfg
COPY freecad/freecad_mcp_settings.json /home/user/.local/share/FreeCAD/v1-1/freecad_mcp_settings.json
RUN chown -R user:user /home/user/.config/FreeCAD /home/user/.local

# The desktop: openbox (no compositor) with a panel (windows, clock) running FreeCAD only, in place of the base's GNOME
# session (DESIGN.md section 3); the MCP server's views are screenshots of this session.
# Terminals, the file manager, text editors and the archive manager are removed, and the base's settings program (it
# starts a terminal server) is dropped.
RUN apt-get update && apt-get install -y --no-install-recommends openbox tint2 x11-xserver-utils \
    && rm -rf /var/lib/apt/lists/* \
    && rm -f /usr/local/bin/gnome-terminal /usr/bin/gnome-terminal /usr/bin/gnome-terminal.real \
             /usr/bin/x-terminal-emulator /usr/bin/nautilus /usr/bin/gedit /usr/bin/gnome-text-editor \
             /usr/bin/file-roller /usr/bin/gnome-screenshot /usr/bin/nano \
    && rm -f /etc/supervisor/conf.d/35-apply-settings.conf
COPY desktop/start-desktop.sh desktop/freecad-session.sh desktop/wait-for-desktop.sh /usr/local/bin/
COPY desktop/30-gnome.conf desktop/31-gnome-ready.conf /etc/supervisor/conf.d/
COPY desktop/openbox-rc.xml desktop/openbox-menu.xml desktop/tint2rc /etc/arb-desktop/
RUN chmod 0755 /usr/local/bin/start-desktop.sh /usr/local/bin/freecad-session.sh /usr/local/bin/wait-for-desktop.sh \
    && chmod 0644 /etc/supervisor/conf.d/30-gnome.conf /etc/supervisor/conf.d/31-gnome-ready.conf \
                  /etc/arb-desktop/openbox-rc.xml /etc/arb-desktop/openbox-menu.xml /etc/arb-desktop/tint2rc
ENV VNC_RESOLUTION=1440x900

# Self-check under a throwaway HOME: FreeCAD writes its per-user directories on start, and the build runs as root.
# The build steps above ran as root with HOME=/home/user and left root-owned caches there (uv, matplotlib, FreeCAD):
# a root-owned FreeCAD cache directory makes the GUI exit at start (it cannot create its single-instance lock). The
# uv cache is dropped and the agent home is handed back to the agent account.
RUN HOME=/tmp/fc-selfcheck /opt/freecad/usr/bin/freecadcmd -c "import FreeCAD, Part, PartDesign, Sketcher, JointObject; print('[arm-links-freecad] FreeCAD', '.'.join(FreeCAD.Version()[:3]))" \
    && rm -rf /tmp/fc-selfcheck /home/user/.cache/uv /home/user/.cache/FreeCAD \
              /home/user/.config/FreeCAD/v1-1/system.cfg \
    && chown -R user:user /home/user

# CHAIN_PLAN Q15 workaround: the claude-code harness aborts a model stream after 300 s without a byte; both timers are
# raised to the gateway's 1800 s upstream ceiling. ENV covers processes the guest service starts, profile.d the CLI's
# login shell.
ENV BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
RUN printf '%s\n' 'export BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800' 'export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000' \
        > /etc/profile.d/99-ale-idle-timeouts.sh \
    && chmod 0644 /etc/profile.d/99-ale-idle-timeouts.sh

# ---- task assets ---------------------------------------------------------------------------------------------------
# The photo set and the bought components' documents and STEP files live in the benchmark's Hugging Face asset
# repository (tasks/quadruped/quadruped_cad_design_from_photos/payload/ there). They are fetched at the commit pinned in
# assets.sha256 and checked file by file; environment/setup/setup.sh stages them into the agent home. fetch_assets.py is
# the benchmark's canonical copy (shared/task_assets/) and the repository named in assets.sha256 is written from
# task-assets.toml (scripts/task_assets.py sync; docs/task-assets.md).
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && chmod -R a+rX /opt/task-assets
