#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase.
#
# 1. Stage the givens (the photo and its facts, the site brief, the twin contract, the product names, check_twin.py with
#    the public gate and standard simulation, render_twin.py with ovrtx_render/) into the agent home, owned by the
#    agent. Nothing hidden is in this folder: the truth lives in tests/ (absent during the agent phase). The photo comes
#    from the image's task assets when the image has them (the published task), else from payload/ (development).
# 2. Fill OVRTX's caches with one warm-up render as the agent account. With an empty cache the first OVRTX render
#    compiles while the stage loads: first on up to 8 CPUs (the renderer's materials), then on one CPU (the driver's
#    Vulkan pipelines), up to about 13 GB resident. Measured 621-998 s in sandboxes and 1371 s in an 8-CPU container
#    on a heavily loaded host. The cache is specific to the GPU and its driver, and the image build has no GPU, so it
#    can only be filled here, per episode. It lands where the agent's own renders look for it (HOME=/home/user:
#    ~/.cache/ovrtx). The warm-up runs in a session of its own under a time limit: render_twin's own --timeout of
#    WARMUP_CAP_S - 30 s (it then stops its OVRTX process and exits 5), and TERM at WARMUP_CAP_S, KILL 30 s later, as a
#    backstop. WARMUP_CAP_S = 1440 keeps the worst case (1470 s, plus staging and clean-up) more than 5 min inside the
#    engine's 1800 s setup limit. Afterwards every process left in that session, and any OVRTX process of the agent
#    account, is stopped, so nothing of it runs into the agent phase. A failed or slow warm-up never fails setup: the
#    agent's first render then compiles instead. The outcome goes to /tmp/render_twin_warmup.json (readable by the
#    agent), the log to /tmp/render_twin_warmup.log and the processes found left in the warm-up's session to
#    /tmp/render_twin_warmup_leftovers.txt.
set -uo pipefail
WARMUP_CAP_S=1440
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/" || { echo "[setup] staging the payload failed" >&2; exit 1; }
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/" || { echo "[setup] staging the task assets failed" >&2; exit 1; }
fi
for entry in payload/* payload/.[!.]* /opt/task-assets/payload/*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission" "$HOME_DIR/.cache/ovrtx"
chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission" "$HOME_DIR/.cache" "$HOME_DIR/.cache/ovrtx"
chmod +x "$HOME_DIR/check_twin.py" "$HOME_DIR/render_twin.py" 2>/dev/null
if [ ! -f "$HOME_DIR/photo.png" ]; then
  echo "[setup] photo.png is missing from the givens" >&2
  exit 1
fi
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER)" >&2

REC=/tmp/render_twin_warmup.json
LOG=/tmp/render_twin_warmup.log
LEFT=/tmp/render_twin_warmup_leftovers.txt
PY=/opt/venv/bin/python3
t0=$(date +%s.%N)
status="skipped_no_gpu"
rc=-1
if [ -e /dev/nvidiactl ]; then
  setsid runuser -u "$AGENT_USER" -- env -u XDG_CACHE_HOME HOME="$HOME_DIR" \
    timeout -s TERM -k 30 "$WARMUP_CAP_S" "$PY" "$HOME_DIR/render_twin.py" --warmup --timeout "$((WARMUP_CAP_S - 30))" \
    > "$LOG" 2>&1 < /dev/null &
  sid=$!
  wait "$sid"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    status="ok"
  elif [ "$rc" -eq 5 ] || [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    status="timed_out"
  else
    status="failed"
  fi
  # whatever the warm-up left behind: its session (listed in $LEFT first), then any OVRTX process of the agent account
  ps -o pid=,ppid=,user=,stat=,etime=,args= -s "$sid" > "$LEFT" 2>/dev/null
  left=$(pgrep -s "$sid" | wc -l)
  pkill -KILL -s "$sid" 2>/dev/null
  left2=$(pgrep -u "$AGENT_USER" -f 'ovrtx_render/ovrtx_job.py|render_twin.py' | wc -l)
  pkill -KILL -u "$AGENT_USER" -f 'ovrtx_render/ovrtx_job.py|render_twin.py' 2>/dev/null
  sleep 1
  remaining=$(pgrep -u "$AGENT_USER" -f 'ovrtx_render/ovrtx_job.py|render_twin.py' | wc -l)
else
  left=0; left2=0; remaining=0
fi
t1=$(date +%s.%N)
printf '{"status": "%s", "exit_code": %s, "seconds": %s, "cap_s": %s, "left_in_session": %s, "left_ovrtx": %s, "remaining": %s, "start_unix": %s, "end_unix": %s}\n' \
  "$status" "$rc" "$(awk -v a="$t0" -v b="$t1" 'BEGIN { printf "%.1f", b - a }')" "$WARMUP_CAP_S" "$left" "$left2" \
  "$remaining" "$t0" "$t1" > "$REC"
chmod 0644 "$REC" "$LOG" "$LEFT" 2>/dev/null
echo "[setup] OVRTX warm-up: $(cat "$REC")" >&2
exit 0
