#!/usr/bin/env bash
# Setup — runs as trusted root, cwd = this staged folder (/opt/ale/setup, root-only), before the agent
# phase: stage the agent-visible payload (the cell's data sheet, the commissioning client and runner, the
# wire helper) into the agent home, and start the commissioning service.
set -euo pipefail
umask 022
export HOME=/root PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONSTARTUP

HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"

# cp -r, NOT -a: with -a the `payload/.` form also copies the payload directory's own root ownership
# onto the agent home, locking the agent out of its own workspace.
cp -r payload/. "$HOME_DIR/"
for entry in payload/* payload/.[!.]*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" && chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"

# The commissioning service: the cell on its development kits (never a graded kit). The image installs
# it under /opt/keyedcell-commission for the account `keyedcell` only (the agent account can neither read
# nor list it); setup starts it now, to run through the agent's session, under a loop that restarts it
# should it ever exit. The agent reaches it through the socket /run/keyedcell/commission.sock
# (~/commission/client.py, ~/commission/run_kit.py): reset(kit), step(action), close(), a fixed number of
# episodes; no ground truth, no score, no completion verdict. It runs the cell with the settings the
# grader's cell runs with (the image's rendering and thread settings). A separate verifier never runs
# this script, so no service runs where the controller is graded.
SERVICE=/opt/keyedcell-commission
[ -f "$SERVICE/commissiond.py" ] || { echo "[setup] the image lacks the commissioning service ($SERVICE)" >&2; exit 1; }
install -d -o keyedcell -g keyedcell -m 755 /run/keyedcell
rm -f /run/keyedcell/commission.sock
setsid runuser -u keyedcell -- env -i HOME="$SERVICE/state" TMPDIR="$SERVICE/state" \
    PATH=/opt/venv/bin:/usr/bin:/bin LANG=C.UTF-8 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=1 \
    LP_NUM_THREADS=2 MESA_SHADER_CACHE_DISABLE=true PYTHONNOUSERSITE=1 \
    /bin/bash -c 'cd /opt/keyedcell-commission/state && while :; do /opt/venv/bin/python3 -B -s /opt/keyedcell-commission/commissiond.py; sleep 2; done' \
    < /dev/null > /dev/null 2>&1 &
for _ in $(seq 1 60); do [ -S /run/keyedcell/commission.sock ] && break; sleep 0.5; done
[ -S /run/keyedcell/commission.sock ] || { echo "[setup] the commissioning service did not come up" >&2; exit 1; }
echo "[setup] commissioning service listening on /run/keyedcell/commission.sock" >&2
