# Cell API (`cell`), version house-cell/1.0

Your program `/home/user/submission/clean.py` exposes `solve(cell) -> None`. `cell` is a proxy
(`house_client.client.Cell`, in `~/given/lib/house_client/`): every method is one JSON-lines request
`{"id": n, "method": ..., "params": {...}}` answered by `{"id": n, "result": {...}}` or `{"id": n, "error": "..."}`
(the proxy raises `CellError` on an error). Nothing moves between requests: only `step` advances time.

| call | returns | notes |
|---|---|---|
| `describe()` | the static facts: the robot, joint names and ranges, control limits, grasp-site definition, grip limits, cameras (intrinsics; the ceiling cameras' exact poses; the head camera's mount), camera budget, the apartment's survey (`room`), budgets, safety thresholds | static for the episode; nothing about the mess |
| `get_state()` | `sim_time`, `done`, `robot`: `base_pose` [x, y, yaw] (localised, noisy), `stand_height`, `base_moving`, the motor state of the waist, arms and hands: `q` {joint name: rad}, `dq` {joint name: rad/s}, `tau_est` {joint name: N m}, `hands: {left/right: {closed, tactile_N, grasp_pose_in_robot [x, y, z, qw, qx, qy, qz], ik_residual}}` | see `cell_manual.md` section 6 |
| `command(**targets)` | `{"ok": true, "notes": [...]}` | any subset of the targets below; each target is held until replaced |
| `step(n_ticks)` | `sim_time`, `done`, `events` | advances `n_ticks` x 10 ms (1 <= n_ticks <= 500); `events`: only `safety:<what>` and `episode_over` |
| `get_camera(name, depth=True)` | `camera`, `frame_index`, `width`, `height`, `format` (`png`), `rgb_png_b64`, `intrinsics`, `pose_in_room` {`position`, `rotation` (3 x 3, columns = the image's x right, y down, z forward axes in the room)}, `sim_time`, and with depth `sensor_depth_u16_mm_b64`, `depth_unit_m` (0.001), `depth_invalid_value` (0) | `name`: `ceiling_kitchen`, `ceiling_dining`, `ceiling_living`, `head`; at most 1500 frames per episode, then `{"error": "camera_budget"}`; `{"error": "camera_unavailable"}` if the renderer fails (the grader then does not score the run: an infrastructure failure, never your score) |
| `finish()` | `{"ok": true}` | ends the episode; the grader then reads the apartment's state |

`command` targets (cell_manual.md section 5):

| key | value |
|---|---|
| `base_velocity` | [vx, vy, wz] in the robot's heading frame (cancels a base goal) |
| `base_goal` | [x, y, yaw] in the room, or `null` to cancel |
| `stand_height` | metres, 0.48 - 0.80 |
| `waist` | [yaw, roll, pitch] joint targets |
| `left_arm`, `right_arm` | `{"q": [7 joint targets]}`, `{"pose": [x, y, z, qw, qx, qy, qz], "frame": "base" or "room", "path": "line" or "joint"}` (the grasp site's target pose) or `{"pos": [x, y, z], "frame": ..., "path": ...}` (its position, the wrist free); `frame` defaults to `"base"`, `path` to `"line"` |
| `left_hand`, `right_hand` | `"open"` or `"close"` |

`house_client.client.decode_frame(reply)` returns `(rgb uint8 H x W x 3, depth float32 H x W in metres with NaN where the
sensor reported nothing)`.

Malformed parameters (a wrong type or shape, an unknown key, a number that is not finite or exceeds 1000 in magnitude,
`base_velocity` and `base_goal` in the same `command`, an arm target with other than exactly one of `q` / `pose` /
`pos`) answer `{"error": "bad_params", "detail": "..."}` and change nothing. Any other method answers
`{"error": "forbidden"}`, does nothing and is counted as an illegal call.

## Running a program

* Graded episodes: the grader launches your program itself (see the instruction): the client's `main()` imports `clean`
  from the current directory and calls `solve(cell)` on a cell connected to the program's stdin / stdout, then calls
  `finish()`. `print()` goes to stderr.
* There is no practice cell: nothing in your session runs the cell. `describe.json` (this folder) is the reply of
  `describe()`; the survey photos (`survey/`, cell_manual.md section 11) are camera frames as `get_camera` returns
  them (`house_client.client.load_frame(path)` rebuilds the reply, `decode_frame` reads it). `Cell(send, recv)`
  accepts any pair of line functions, so a stand-in cell of your own can drive your program for testing.
