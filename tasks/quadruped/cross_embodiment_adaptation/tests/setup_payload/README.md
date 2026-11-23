# Robot service

The public client (`client.py`) and the MCP tools reach the service on `/run/visual_robotics/world.sock`
(`ROBOT_SOCKET` overrides it). Run programs with `/opt/venv/bin/python3`. The task instruction has the camera
calibration, the joint semantics, the observation fields and the budgets; `go1_datasheet.md` has the robot.

- `observe()` moves nothing.
- UAV (a survey site): `act([dx, dy])` translates it and returns a down-camera image.
  `camera_image(obs)` returns a NumPy RGB array.
- Go1 (the pad here, a site when graded): `act({"target_rad": [...], "ticks": n})` holds one pose;
  `look(detent)` is one mast-camera exposure.
- Development only: `new_site()` starts another survey site (UAV only); `pad()` starts a pad episode.
  Replies here carry `pad_episodes_remaining` and `pad_ticks_remaining`.
- Grading only: `advance()` retires the UAV and deploys the Go1 once.
- `bash /home/user/session_time.sh` prints the minutes used and left of the development session.
