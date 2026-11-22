# SR-2 robot interface (`rubblecell.client`)

Your program is `/home/user/submission/rescue.py` with a function `solve(robot)`. The grader (and
`~/training/run_mission.py`) imports it and calls `solve(robot)` once per mission; `robot` is a
`rubblecell.client.Robot`. Every method is one request to the robot cell and returns a dict (a reply carrying an
error raises `rubblecell.client.CellError`). Units: metres, radians, seconds; frames in `robot_manual.md` section 1.

**Time passes only in `step`.** Commands set targets that the robot's controllers follow while the simulation runs;
sensor requests return what the sensor sees at the current simulated instant (they take no simulated time).

## Information

| call | returns |
|---|---|
| `describe()` | static facts: limits, rates, the cameras' sizes and intrinsics, the lidar, the budgets, the map frame |
| `get_state()` | `sim_time`, `done`, `done_reason`; `pose` (odometry, map frame: `x`, `y`, `z`, `yaw`, plus IMU `roll`, `pitch`); `velocity` (`forward`, `lateral`, `yaw_rate`, chassis frame, an estimate with noise); `imu` (`gyro`, `roll`, `pitch`, with noise: manual section 6); `joints` (`gauge`, `flipper_front`, `flipper_rear`, `mast`, `head_pan`, `head_tilt`, `arm` [5], `gripper`); `commands` (`v`, `w`); `tracks` (belt speeds `left`, `right`); `gripper` (`holding`, `load_N`); `arm_moving`; `kit_in_cradle`; `link` (`up`, `rssi_dbm`); `reports_pending` |

## Sensors (each counts against the mission's budget)

| call | returns | decode with |
|---|---|---|
| `get_camera(name, depth=True)` | `name` in `mast`, `front`, `rear`, `gripper`: `rgb_png_b64` (PNG), for `mast` and `gripper` also `depth_u16_mm_b64` (16-bit PNG, millimetres, 0 = no reading); `intrinsics` (`fx`, `fy`, `cx`, `cy`, size); `T_chassis_cam` (4 x 4, optical frame) | `decode_frame(reply)` -> (rgb uint8 H x W x 3, depth float32 metres with NaN) |
| `get_lidar()` | one sweep: `points_f32_b64` (float32 x, y, z, channel per point, lidar frame), `n`, `T_chassis_lidar` (4 x 4) | `decode_lidar(reply)` -> float32 N x 4 |
| `get_thermal()` | `centikelvin_u16_png_b64` (16-bit PNG, kelvin x 100), `intrinsics`, `T_chassis_cam` (the thermal camera's optical frame) | `decode_thermal(reply)` -> float32 H x W in degrees Celsius |

Budgets per mission: 700 camera frames (all cameras together), 1800 lidar sweeps, 1800 thermal frames; past a budget
the call answers `camera_budget` / `lidar_budget` / `thermal_budget`.

## Motion (targets, held until changed)

| call | does |
|---|---|
| `drive(v, w)` | forward speed v (-0.6 .. 0.6 m/s) and yaw rate w (-1.2 .. 1.2 rad/s) |
| `set_flippers(front=None, rear=None)` | flipper angles (-80 deg .. 190 deg; 0 = straight out, + = tip raised) |
| `set_gauge(width)` | track gauge, 0.30 .. 0.50 |
| `set_mast(height)` | mast extension, 0 .. 0.45 |
| `set_head(pan=None, tilt=None)` | pan -175 deg .. 175 deg (+ = left), tilt -60 deg .. 45 deg (+ = up) |
| `set_arm(q)` | the five arm joint targets (clipped to their ranges); cancels a running `arm_to` |
| `arm_to(p, pitch=-pi/2, roll=None, elbow_up=True, speed=0.15)` | straight-line move of the tool centre point to chassis-frame point p with tool pitch `pitch` and wrist roll `roll` (None keeps the current roll); answers `{"ok": false, "error": "unreachable"}` when the target has no inverse-kinematics solution, else `{"ok": true, "q": [...], "duration_s": ...}` |
| `gripper(width)` | jaw opening target, 0 .. 0.16 (grip rule: `robot_manual.md` section 4) |
| `step(n_ticks=1)` | advance n x 10 ms (at most 500 ticks per call); returns `{"sim_time", "done", "events"}` |

Each axis follows its target at its rated speed (manual sections 2 to 4). Events in `step`'s reply: `grasped`,
`released`, `slipped`, `grip_slip` (too heavy), `kit_unlatched`, `link_up`, `link_down`, `report_sent`, `collapse`
(a secondary collapse: the mission is aborted and scores 0), `protective_stop` (contact with the survivor above the
limit: the drives stop for good and the mission ends; manual section 8), `robot_overturned`, `robot_fell`,
`episode_over`.

Every number in a request must be finite: a request carrying `NaN`, an infinity or a malformed parameter is answered
`bad_params` and changes nothing.

## The mission

| call | does |
|---|---|
| `report_survivor(x, y, z=0)` | the survivor's position in the map frame; sent to the team at once when the link is up, else queued and sent when it comes up. You may report again: the team holds the last report that reached it. Returns `{"ok", "sent"}` |
| `finish()` | ends the mission (also done for you when `solve` returns) |

A mission also ends at 1800 s of simulated time, when the robot is lost, at a protective stop and at a secondary
collapse; calls that move the robot then answer `episode_over` / `robot_lost`. After the end your program may still read its state; it should return.

## Running

* Before any mission the grader applies the submission checks of `~/training/submission_checks.py` (listed in the
  task's instruction: no weight files, no training calls, no networking or process-starting import, no computed
  import or code); a submission that fails one scores 0 on every site.
* The grader launches your program as `python -B -E -s` with the client library on `PYTHONPATH`, the working
  directory = a read-only copy of your submission folder (writing a file there raises `PermissionError`), an empty
  writable `HOME` and `TMPDIR`, no network, the numerics pin (one BLAS thread on fixed kernels, NumPy without its
  AVX-512 loops, OpenCV's IPP at AVX2: `numerics_pin` in `submission_checks.py`), and limits: 120 s (wall) and 120 CPU-s to import `rescue.py`; per mission
  1200 s of program time (the wall clock minus the cell's simulation and sensor rendering), 4800 CPU-s, 180 s of
  silence between requests.
* Ordinary prints go to stderr (the launcher binds the wire to the original stdout first): print as much as you like.
* A submission that ships `run.sh` is started with `bash run.sh` instead; it must then speak the wire itself (one JSON
  request per line on stdout, replies on stdin: see `client.py`).
* `~/training/run_mission.py` applies the same checks, then runs your submission on the training ground in the same
  regime (as your account) and prints the robot's mission log; `--check` applies the checks only.
  `connect_training(site)` gives you an interactive `Robot` on a training site (one mission per connection).
