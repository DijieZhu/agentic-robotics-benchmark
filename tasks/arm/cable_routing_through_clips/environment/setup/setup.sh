#!/usr/bin/env bash
# Setup — runs as trusted root, cwd = this staged folder (/opt/ale/setup, root-only), before the agent phase:
# stage the agent-visible payload (the commissioning client and runner, the wire skeleton, the
# session-time script) into the agent home, write the session's time stamp, and start the
# commissioning service. Nothing hidden is in this folder.
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

# The session's time stamp (session_time.sh in the agent's home reads it): written now, moments before the agent's
# session starts, in a folder that is root's, so that the agent's account can read it and cannot change or replace it.
install -d -o root -g root -m 755 /run/session
date -u +%s > /run/session/start
chown root:root /run/session/start && chmod 0444 /run/session/start


# The commissioning service (environment/commission_service, installed by the image at /opt/harness-commission,
# readable by its own account only): started here, as that account, with a fixed environment and PATH (no
# agent-writable directory on it), detached, restarted if it ever exits. It is not started in the grading sandbox:
# this script does not run there.
SERVICE=/opt/harness-commission
install -d -o harnessbench -g harnessbench -m 755 /run/harnessbench
rm -f /run/harnessbench/commission.sock
setsid runuser -u harnessbench -- /usr/bin/env -i HOME="$SERVICE/state" TMPDIR="$SERVICE/state" \
  PATH=/opt/venv/bin:/usr/bin:/bin LANG=C.UTF-8 MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa OMP_NUM_THREADS=1 \
  LP_NUM_THREADS=4 MESA_SHADER_CACHE_DISABLE=true PYTHONDONTWRITEBYTECODE=1 HARNESS_RENDER_DIR=/opt/task-assets/payload/render \
  /bin/bash -c 'cd /opt/harness-commission/state && while :; do /opt/venv/bin/python3 -B -s /opt/harness-commission/commissiond.py; /usr/bin/sleep 2; done' \
  < /dev/null > /dev/null 2>&1 &
for _ in $(seq 1 60); do [ -S /run/harnessbench/commission.sock ] && break; sleep 0.5; done
[ -S /run/harnessbench/commission.sock ] || { echo "[setup] the commissioning service did not come up" >&2; exit 1; }
