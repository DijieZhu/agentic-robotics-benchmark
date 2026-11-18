#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase.
#
# Stage the givens into the agent home, owned by the agent: the survey photos (photos/site_001.jpg .. site_044.jpg),
# photo_info.json, the site brief, the twin contract, the class names, check_twin.py with the public gate, standard
# simulation, camera and slice modules (and the grader's walk layer as twin_walk.py), and the probe robot (robot/:
# model, controller, walk_probe.py, robot card). Nothing hidden is in this folder: the truth lives in tests/ (absent
# during the agent phase). The photos come from the image's task assets when the image has them (the published task,
# /opt/task-assets/payload), else from payload/ (local runs). An empty submission/ is created (the image has one too).
# Nothing else runs here: no warm-up, no download, no build.
set -uo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
N_PHOTOS=44
cp -r payload/. "$HOME_DIR/" || { echo "[setup] staging the payload failed" >&2; exit 1; }
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/" || { echo "[setup] staging the task assets failed" >&2; exit 1; }
fi
for entry in payload/* payload/.[!.]* /opt/task-assets/payload/*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
mkdir -p "$HOME_DIR/submission"
chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
chmod +x "$HOME_DIR/check_twin.py" 2>/dev/null
missing=0
for i in $(seq -f "%03g" 1 "$N_PHOTOS"); do
  [ -s "$HOME_DIR/photos/site_$i.jpg" ] || { echo "[setup] photos/site_$i.jpg is missing from the givens" >&2; missing=1; }
done
for f in photo_info.json site_brief.md twin_contract.md class_names.json check_twin.py twin_gate.py twin_sim.py \
         twin_sim_config.json twin_camera.py twin_slice.py robot/walk_probe.py robot/robot_card.md; do
  [ -s "$HOME_DIR/$f" ] || { echo "[setup] $f is missing from the givens" >&2; missing=1; }
done
[ "$missing" -eq 0 ] || exit 1
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER): $N_PHOTOS photos" >&2
exit 0
