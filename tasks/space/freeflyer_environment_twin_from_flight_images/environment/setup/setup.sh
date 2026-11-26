#!/usr/bin/env bash
# Setup: trusted root, cwd = this staged folder, before the agent phase.
#
# Stage the givens into the agent home, owned by the agent: the flight brief, the twin contract, the sensor files
# (sensors/), the module's shell and zone record (module_shell/), the check tool (tools/check_twin.py with its
# configuration and tools/twinprobe/) and, per flight, flights/<flight>/ with its frames (frames/<name>.png),
# frames.csv, imu.csv and onboard_estimate.csv. Nothing hidden is in this folder: the truth lives in the
# grader's stage (absent during the agent phase). The frames come from the image's task assets when the image has them
# (the published task, /opt/task-assets/payload), else from payload/ (local runs); either way each flight's frames
# arrive as ONE archive, flights/<flight>/frames.tar, which is unpacked here into flights/<flight>/frames/ and then
# removed. One empty submission/<flight>/ folder is created per given flight. Nothing else runs here: no warm-up, no
# download, no build.
#
# No count is written here: the flights are the folders under flights/, and the frames each flight must have are the
# names listed in its own frames.csv (a comment line, the header row, then one row per frame, its name first).
set -uo pipefail
HOME_DIR="${ALE_HOME:-/home/user}"
AGENT_USER="$(basename "$HOME_DIR")"
cp -r payload/. "$HOME_DIR/" || { echo "[setup] staging the payload failed" >&2; exit 1; }
if [ -d /opt/task-assets/payload ]; then
  cp -r /opt/task-assets/payload/. "$HOME_DIR/" || { echo "[setup] staging the task assets failed" >&2; exit 1; }
fi
for archive in "$HOME_DIR"/flights/*/frames.tar; do
  [ -f "$archive" ] || continue
  flight_dir="$(dirname "$archive")"
  tar -x -f "$archive" -C "$flight_dir" --no-same-owner --no-same-permissions --touch \
    || { echo "[setup] unpacking flights/$(basename "$flight_dir")/frames.tar failed" >&2; exit 1; }
  rm -f "$archive"
done
for entry in payload/* payload/.[!.]* /opt/task-assets/payload/*; do
  [ -e "$entry" ] || continue
  chown -R "$AGENT_USER:$AGENT_USER" "$HOME_DIR/$(basename "$entry")"
done
chmod +x "$HOME_DIR/tools/check_twin.py" 2>/dev/null
missing=0
for f in flight_brief.md twin_contract.md sensors/camera_sheet.json sensors/depth_rays.npy sensors/wide_to_depth.json \
         sensors/probe_card.json module_shell/shell.obj module_shell/zone_record.json tools/check_twin.py \
         tools/check_twin_config.json; do
  [ -s "$HOME_DIR/$f" ] || { echo "[setup] $f is missing from the givens" >&2; missing=1; }
done
[ -d "$HOME_DIR/tools/twinprobe" ] || { echo "[setup] tools/twinprobe/ is missing from the givens" >&2; missing=1; }
mkdir -p "$HOME_DIR/submission"
chown "$AGENT_USER:$AGENT_USER" "$HOME_DIR/submission"
n_flights=0
n_frames=0
for dir in "$HOME_DIR"/flights/*/; do
  [ -d "$dir" ] || continue
  flight="$(basename "$dir")"
  n_flights=$((n_flights + 1))
  for f in frames.csv imu.csv onboard_estimate.csv; do
    [ -s "$dir$f" ] || { echo "[setup] flights/$flight/$f is missing from the givens" >&2; missing=1; }
  done
  if [ -s "${dir}frames.csv" ]; then
    listed=0
    while IFS=, read -r name _; do
      listed=$((listed + 1))
      [ -s "${dir}frames/$name.png" ] || { echo "[setup] flights/$flight/frames/$name.png is missing from the givens" >&2; missing=1; }
    done < <(grep -v '^#' "${dir}frames.csv" | tail -n +2)
    [ "$listed" -gt 0 ] || { echo "[setup] flights/$flight/frames.csv lists no frame" >&2; missing=1; }
    n_frames=$((n_frames + listed))
  fi
  install -d -o "$AGENT_USER" -g "$AGENT_USER" "$HOME_DIR/submission/$flight"
done
[ "$n_flights" -gt 0 ] || { echo "[setup] no flight folder under flights/ in the givens" >&2; missing=1; }
[ "$missing" -eq 0 ] || exit 1
echo "[setup] givens staged in $HOME_DIR (owner $AGENT_USER): $n_flights flights, $n_frames frames" >&2
exit 0
