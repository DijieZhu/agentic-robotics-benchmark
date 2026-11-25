# Search station interface (`voidcell.client`)

The search station is the team's PC outside the building: it receives the search camera's frames and the robot's
telemetry and runs the robot's cable controller. Your program talks to it through the `station` proxy of the given
client library. This file is the interface: what every call takes, returns and costs.

## 1. Your program

```python
# /home/user/submission/operate.py
def solve(station):
    m = station.mission()                      # site rectangle, pad, search points, camera, budgets (section 3)
    f = station.get_frame()                    # f["rgb"], f["depth"]: numpy arrays (section 4.1)
    r = station.run_primitive("roll_120_120", [0, 3, 5])   # {"accepted": True, "error": None} (section 5.1)
    while station.status()["busy"]:
        station.step(0.5)                      # simulated time passes only here (section 2)
    tl = station.telemetry()                   # the robot's records since the last call (section 4.2)
    station.report_point(1)                    # section 6
    station.finish()
```

`solve(station)` is called once per episode. Helper modules and data files may lie next to `operate.py`. Every method
of `station` is one request to the station and returns its reply as a Python object; the client decodes camera frames
and telemetry into numpy arrays. A call that the station refuses outright raises `voidcell.client.StationError`
(`.code`, `.detail`; section 7); a command that it does not take comes back as a result with `accepted` false.

<!-- BEGIN GENERATED calls -->
| call | arguments | reply (client-side keys) | what |
|---|---|---|---|
| `mission` | — | `site`, `pad`, `search_points`, `dwell_s`, `credit_radius_m`, `still_tolerance_m`, `camera`, `limits`, `pose_source` | the episode's mission (section 3) |
| `get_frame` | — | `index`, `t`, `width`, `height`, `encoding`, `depth_unit_m`, `frames_left`, `rgb`, `depth` | the newest camera frame; uses one frame of the budget (section 4.1) |
| `telemetry` | — | `n`, `dropped`, `columns`, `t`, `sensor_length_mm`, `sensor_capacitance_pF`, `motor_encoder_counts`, `motor_encoder_length_mm`, `motor_target_mm`, `motor_done`, `motor_duty`, `imu_accel_g`, `imu_gyro_dps` | the robot's records since the previous call (section 4.2) |
| `status` | — | `t`, `busy`, `kind`, `state_index`, `done`, `timeouts` | what the cable controller knows (section 5.4) |
| `run_primitive` | `name`, `bottom_nodes` | `accepted`, `error` | start one rolling primitive (section 5.1) |
| `set_targets` | `lengths_mm` | `accepted`, `error` | start one target shape (section 5.2) |
| `stop` | — | `{}` | abort the running command, motors braked (section 5.3) |
| `report_point` | `i` | `accepted`, `reports_left`, `error` | report search point `i` (section 6) |
| `step` | `dt` | `t`, `done` | advance the simulated time (section 2) |
| `finish` | — | `done` | end the episode (section 6) |
| `pose` | — | `t`, `index`, `nodes_xyz`, `visible_caps` | the robot's pose from the station, if the task provides it (section 4.3) |
<!-- END GENERATED calls -->

## 2. Time

- Simulated time advances only inside `station.step(dt)`: `dt` in seconds, from 0.01 to 10.0, a multiple of 0.01. The
  reply is `{t, done}`: the simulated time after the step and whether the episode has ended. Every other call takes
  no simulated time. The episode clock starts at 0 with the robot standing on the pad.
- The robot and its cable controller run in control periods of 10 ms. A command (a primitive, a `set_targets`) is
  started by its call and runs while your program steps.
- When the simulated time limit is reached, `step` stops at the limit and returns `done` true: the episode has ended
  (section 6).
- Your program's own time is limited too: the program time, the silence between two requests and the CPU time
  (section 8). The time the station itself needs inside `step`, `get_frame`, `telemetry` and `pose` does not count as
  program time.

## 3. `mission()`

Returns what the team knows of this episode's site. It can be called at any time and always returns the same.

<!-- BEGIN GENERATED mission_keys -->
| key | content |
|---|---|
| `site` | `frame`: the site frame in words; `size_m`: `[x, y]` extent of the site rectangle (m) |
| `pad` | `xy`: the set-down pad's centre; `retrieval_radius_m`: the retrieval area's radius; `setdown_error_m`: the largest distance of the robot's centre from the pad's centre at the start |
| `search_points` | list of `{id, xy, weight}`: the three search points, `id` 1 to 3, site-frame `xy` (m), the point's weight in the score |
| `dwell_s` | the dwell window before a report (s) |
| `credit_radius_m` | `[full, zero]`: full credit within the first distance, none beyond the second (m) |
| `still_tolerance_m` | how far the robot's position may lie from its position at the report during the dwell (m) |
| `camera` | `width`, `height` (px); `K` = `[fx, fy, cx, cy]` (px); `depth_unit_m`; `depth_range_m`; `rate_hz`; `T_world_camera`: 4 × 4, camera frame to site frame, the calibrated extrinsic (camera_datasheet.md) |
| `limits` | `time_limit_s`, `frames`, `reports_per_point`, `cable_length_mm` (`[min, max]`), `program_time_s`, `silence_s`, `cpu_s`: the budgets of this episode |
| `pose_source` | `"estimated"` or `"cell"` (section 4.3) |
<!-- END GENERATED mission_keys -->

Nothing else of the site is in it: no layout, no opening, no ground patch. As an example, `mission()` of the
calibration site:

<!-- BEGIN GENERATED mission_example -->
```json
{
 "site": {"frame": "x east, y north, z up, origin = south-west floor corner", "size_m": [4.4, 5.8]},
 "pad": {"xy": [2.2, 0.45], "retrieval_radius_m": 0.6, "setdown_error_m": 0.1},
 "search_points": [{"id": 1, "xy": [0.95, 1.05], "weight": 0.05}, {"id": 2, "xy": [1.75, 2.7], "weight": 0.15}, {"id": 3, "xy": [3.75, 5.15], "weight": 0.8}],
 "dwell_s": 20.0,
 "credit_radius_m": [0.2, 0.4],
 "still_tolerance_m": 0.02,
 "camera": {"width": 640, "height": 480, "K": [607.686, 607.987, 326.55, 250.662], "depth_unit_m": 0.00025, "depth_range_m": [0.25, 9.0], "rate_hz": 7.0, "T_world_camera": [[0.999813, -0.018677, 0.004915, 2.236767], [-0.019313, -0.966275, 0.256788, 0.313129], [-4.6e-05, -0.256835, -0.966455, 5.085189], [0.0, 0.0, 0.0, 1.0]]},
 "limits": {"time_limit_s": 1400.0, "frames": 600, "reports_per_point": 3, "cable_length_mm": [100.0, 240.0], "program_time_s": 600.0, "silence_s": 180.0, "cpu_s": 2400.0},
 "pose_source": "estimated"
}
```
<!-- END GENERATED mission_example -->

## 4. Sensing

### 4.1 `get_frame()`: the search camera

Returns the newest frame of the search camera (`camera_datasheet.md` describes the camera and how a pixel maps to a
site-frame point).

<!-- BEGIN GENERATED frame_keys -->
| key | content |
|---|---|
| `index` | the exposure's number, counted from 0 at the episode's start |
| `t` | the simulated time of the exposure (s) |
| `width`, `height` | 640, 480 |
| `rgb` | numpy `uint8`, shape (480, 640, 3), red-green-blue |
| `depth` | numpy `uint16`, shape (480, 640): depth along the optical axis in counts of `depth_unit_m`; 0 = no reading |
| `depth_unit_m` | 0.00025 (so 4000 counts per metre) |
| `encoding` | the wire's encoding of the two images (section 9); the client has already decoded them |
| `frames_left` | frames left of the episode's 600 after this one |
<!-- END GENERATED frame_keys -->

- The camera exposes at its frame rate: exposure k is taken at the first control period at or after k / rate seconds
  of simulated time; exposure 0 shows the robot at the start. `get_frame()` returns the newest exposure, so a frame's
  `t` is never later than the current simulated time and at most one frame period before it.
- Every call uses one frame of the episode's budget, also when it returns the same exposure as the call before (the
  same `index`, the same pixels). When the budget is used up the call raises `budget_exhausted` and nothing changes.
- The camera does not move. The robot, when it is in view, is in the picture as it is; the pad and the search points
  are not marked.

### 4.2 `telemetry()`: the robot's records

Returns the records the robot has sent since your previous call, one per control period, oldest first, as numpy
arrays with n rows (`robot_handbook.md`, section 9, explains every reading and its noise):

<!-- BEGIN GENERATED telemetry_keys -->
| key | shape | content |
|---|---|---|
| `t` | (n,) | simulated time at the end of the control period (s) |
| `sensor_length_mm` | (n, 9) | the nine stretch sensors' lengths (mm): 0-5 the short sensors, 6-8 the long ones |
| `sensor_capacitance_pF` | (n, 9) | the same nine sensors' raw capacitance readings (pF) |
| `motor_encoder_counts` | (n, 6) | encoder counts of the six motors at the output shaft; + = line wound in |
| `motor_encoder_length_mm` | (n, 6) | the same as line length on the spool (mm) |
| `motor_target_mm` | (n, 6) | the controller's current target length of short sensors 0-5 (mm); NaN while no command runs |
| `motor_done` | (n, 6) | the controller's done flag of each motor for the current target state |
| `motor_duty` | (n, 6) | the duty sent to each motor, -1 to 1; + = wind in |
| `imu_accel_g` | (n, 3, 3) | per strut k (first index) the accelerometer's three axes (g), in that strut's IMU frame |
| `imu_gyro_dps` | (n, 3, 3) | per strut k the gyroscope's three axes (deg/s), in the same frame |
| `n`, `dropped` | integers | the number of records in this reply; the number of records that were lost because more than 30 s had passed since the previous call |
<!-- END GENERATED telemetry_keys -->

The station keeps the records of the last 30 s of simulated time: call `telemetry()` at least that often if you want
all of them. A call before the first `step` returns n = 0 and arrays with no rows. The call uses no budget.

### 4.3 `pose()`

<!-- BEGIN GENERATED pose_source -->
The station gives your program **no pose of the robot**: `mission()["pose_source"]` is `"estimated"` and `pose()` answers `not_available`. Your program finds the robot in the camera frames while it is in view, and keeps track of it from the primitives it has commanded and from the telemetry while it is not.

The other value of the switch, `"cell"`, is not used in this task's scored episodes or practice ground. With it `pose()` would answer `{t, index, nodes_xyz, visible_caps}` for the newest camera exposure: the six node positions (site frame, m) with independent Gaussian noise of 10 mm per axis when at least 4 of the 6 end caps are in the camera's line of sight, otherwise `nodes_xyz` null (always null under a roof); `visible_caps` lists the node numbers in sight.
<!-- END GENERATED pose_source -->

## 5. Commands

One command runs at a time. A command call returns at once with `{accepted, error}`; an accepted command then runs
during the following `step` calls until it ends by itself or `stop()` aborts it. A refused command changes nothing.

### 5.1 `run_primitive(name, bottom_nodes)`

Starts one of the eleven primitives of the handbook (`roll_100_100`, `roll_120_120`, `roll_140_140`, `roll_100_120`,
`roll_120_100`, `roll_100_140`, `roll_140_100`, `roll_120_140`, `roll_140_120`, `ccw`, `cw`). `bottom_nodes` is a
list of the three node numbers that are on the ground, in any order; it must be one of the six rollable faces. The
station uses it to map the primitive onto the motors (handbook, section 6) and does **not** check it against the
robot: with a wrong triple the robot executes the primitive of another face. Refusals: `busy`, `invalid_name`,
`invalid_bottom`, `bad_argument`.

### 5.2 `set_targets(lengths_mm)`

Starts one target shape: a list of six lengths in millimetres, the targets of short sensors 0 to 5, each within the
cable-length limits of `mission()["limits"]["cable_length_mm"]`. The cable controller drives the six motors to them as
it drives a target state of a primitive (handbook, section 5); the command ends when the state ends. Refusals:
`busy`, `out_of_range`.

### 5.3 `stop()`

Aborts the running command and brakes the motors; returns `{}`. Without a running command it does nothing.

### 5.4 `status()`

`{t, busy, kind, state_index, done, timeouts}`: the simulated time; whether a command is running; `"primitive"`,
`"set_targets"` or null; the running command's current target state; the six done flags of that state; the number of
states of the running command that ended by their timeout (handbook, section 5). It uses no budget and takes no
simulated time, so the usual loop is `while station.status()["busy"]: station.step(...)`.

## 6. Reports and the end of an episode

`report_point(i)`, with `i` = 1, 2 or 3, reports search point `i` at the current simulated time. The reply is
`{accepted, reports_left, error}`. The station accepts a point's reports up to the number in
`mission()["limits"]["reports_per_point"]`; a further one is refused (`accepted` false, `error` `budget_exhausted`,
`reports_left` 0) and changes nothing. Only the last accepted report of a point counts. The reply never says where
the robot is or what the report is worth.

`finish()` ends the episode and returns `{done: true}`. An episode ends:

- when your program calls `finish()` or `solve` returns (the launcher then calls `finish()` for you), also when
  `solve` raises an exception;
- when the simulated time limit is reached (`step` returns `done` true);
- when your program exceeds its program time, its CPU time or the silence limit, sends a line that is not a request,
  or exits;
- when the station itself fails (`internal_error`).

However it ends, the reports accepted until then are scored and the retrieval is judged at that instant. After
`finish()` the station closes the connection at once. After an end that leaves your program running (the time
limit), `run_primitive`, `set_targets`, `stop`, `report_point` and `step` raise `episode_over`, while `mission`,
`get_frame`, `telemetry` and `status` still answer; the station then closes the connection after at most 50 further
`step` calls, 200 further requests or 60 s.

## 7. Errors

<!-- BEGIN GENERATED errors -->
| code | how it arrives | when |
|---|---|---|
| `busy` | result / raised | `run_primitive` or `set_targets` while a command is executing: wait, or `stop()` first (result); in the practice ground a connection opened while the allowed number of episodes is running (raised); an episode counts as running until about a second after its last line, so open the connection again |
| `invalid_name` | result | `run_primitive` with a name that is not one of the eleven primitives |
| `invalid_bottom` | result | `run_primitive` with three distinct node numbers that are not a rollable face |
| `out_of_range` | result | `set_targets` with a length outside the cable-length limits |
| `budget_exhausted` | result / raised | a report of a point that has used its reports (result, with `reports_left` 0); `get_frame()` when the frames are used (raised); in the practice ground also `reset` when the allowance is used |
| `not_available` | raised | `pose()` when the task does not provide it; in the practice ground a call before `reset`, or a site that cannot be opened |
| `bad_argument` | result / raised | `bottom_nodes` that are not three distinct whole numbers 0 to 5 (result); a missing or unknown argument name, a wrong JSON type, a list of the wrong length, a number that is not finite, a `dt` that `step` does not take, an `i` that is not 1, 2 or 3; in the practice ground a `reset` with an unknown site (raised) |
| `forbidden` | raised | a call that is not one of the station's: nothing happens |
| `episode_over` | raised | `run_primitive`, `set_targets`, `stop`, `report_point` or `step` after the episode has ended |
| `internal_error` | raised | the station itself failed; the episode ends, and a scored episode in which this happened is run again from scratch |
<!-- END GENERATED errors -->

"Result" means a normal reply with `accepted` false and the code in `error`; "raised" means that the client raises
`StationError` with the code in `.code` and an explanation in `.detail`. The client raises three codes of its own:
`closed` (the station closed the connection), `protocol` (a reply that does not match the request) and `unreachable`
(the practice ground's socket cannot be opened). A refused call changes nothing and takes no simulated time.

## 8. How your program is run

<!-- BEGIN GENERATED regime -->
**In a scored episode** your program is one fresh process under an unprivileged account that is not yours, started as `python -B -E -s -c <launcher> <library folder> operate`. Its working directory is a read-only copy of your submission folder (regular files only: symbolic links are dropped). It has no network, and its environment holds nothing of yours:

| variable | value |
|---|---|
| `PATH` | the image's |
| `HOME`, `TMPDIR` (also as `TMP` and `TEMP`) | the run's own two empty folders, writable; emptied after the episode |
| `LANG` | `C.UTF-8` |
| `PYTHONPATH` | a read-only copy of the given library (`~/given/lib`) |
| `ARB_GIVEN_DIR` | a read-only copy of `~/given` |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `LP_NUM_THREADS` | 2: the numerical libraries are held to 2 threads |
| `OPENBLAS_CORETYPE`, `OPENBLAS_L2_SIZE`, `NPY_DISABLE_CPU_FEATURES` | `Haswell`, `256`, `X86_V4 AVX512_ICL AVX512_SPR`: the numerical libraries use the same kernels on every x86-64 host, in a practice run as in a scored episode |
| a few more | settings of the image's plotting and rendering libraries; your program does not need them |

**In a practice run** (`~/training/run_mission.py`) the program is started the same way: the same flags, environment and CPU limit, from read-only copies of your submission folder, of the library and of `~/given`. One thing differs: it runs under your own account, so it can read what your account can read. A program that reads a file of your home directory by its path therefore works in a practice run and fails in a scored episode.

`print` writes to the standard error stream (the launcher keeps the standard output for the requests). In a practice run it comes out with the output of `run_mission.py`; in a scored episode it goes to a log that is not shown to you.

A submission that ships a regular file `run.sh` is started as `bash run.sh` instead of the launcher, in the same folder and environment, and must then speak the wire itself (section 9).
<!-- END GENERATED regime -->

Reach the given files from your program through `os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`.

<!-- BEGIN GENERATED budgets -->
| budget | per episode |
|---|---|
| simulated time | 1820 s (calibration site: 1400 s) |
| program time: the wall clock minus the station's own work inside `step`, `get_frame`, `telemetry`, `pose` | 600 s |
| silence: from the start, or from a reply, to your next request | 180 s |
| CPU time of your program, all threads together | 2400 s |
| camera frames (`get_frame()`) | 600 |
| reports (`report_point(i)`) | 3 per search point |
| cable-length targets (`set_targets`) | 100 to 240 mm each |
<!-- END GENERATED budgets -->

Exceeding the program time, the CPU time or the silence limit ends the episode as it stands, as does a line on the
wire that is not a request, or the end of your process.

The contract of a submission (one that breaks it scores 0 on every episode):

<!-- BEGIN GENERATED audit -->
- `operate.py` is a regular file, parses, and defines `solve`; importing it under the run regime above finishes within 120 s of wall clock and 120 s of CPU time (all threads).
- The submission folder is at most 60 MB and holds no file whose name ends in `.pt`, `.pth`, `.onnx`, `.safetensors`, `.ckpt`, `.h5`, `.pb`, `.pkl`, `.npz`; no `.py` or `.sh` file in it contains the text `torch.optim`, `.backward(`, `jax.grad`.
- Your own `.py` files (a byte-identical copy of the given `client.py` is exempt) import none of these modules, nor a sub-module of one: `socket`, `_socket`, `ssl`, `_ssl`, `urllib`, `urllib2`, `urllib3`, `http`, `httplib`, `requests`, `httpx`, `aiohttp`, `ftplib`, `smtplib`, `imaplib`, `poplib`, `nntplib`, `telnetlib`, `websocket`, `websockets`, `paramiko`, `pycurl`, `xmlrpc`, `asyncio`, `socketserver`, `grpc`, `zmq`, `huggingface_hub`, `boto3`, `botocore`, `wget`, `ctypes`, `_ctypes`, `cffi`, `selectors`, `multiprocessing.connection`, `multiprocessing.managers`, `logging.handlers`, `subprocess`, `pty`, `_posixsubprocess`.
- They load no module under a computed name (`__import__`, `import_module`, `find_spec` with an argument that is not a string literal, or naming a module of the list above), run no computed code (the built-ins `exec`, `eval`, `compile` called by their bare names with an argument that is not a string literal; a method of the same name, such as `re.compile(...)`, is fine), and start no program: no call `os.<name>(...)` or `posix.<name>(...)`, and no call by the bare name `<name>(...)`, with `<name>` one of `system`, `popen`, `execv`, `execve`, `execvp`, `execvpe`, `execl`, `execle`, `execlp`, `execlpe`, `spawnv`, `spawnve`, `spawnvp`, `spawnvpe`, `spawnl`, `spawnle`, `spawnlp`, `spawnlpe`, `posix_spawn`, `posix_spawnp`, `fork_exec`.
- Over the episodes the program makes at least one `step` call, leaves no process of the run account behind when an episode ends, and never holds a network socket (it runs under a filter that refuses every socket except a Unix-domain one).
<!-- END GENERATED audit -->

## 9. The wire

The client library does this for you; a submission that ships `run.sh` speaks it itself. The program writes requests
on its standard output and reads replies on its standard input, one JSON object per line, at most 4 MB per line:

```
{"id": 7, "method": "step", "params": {"dt": 0.5}}      ->   {"id": 7, "result": {"t": 12.5, "done": false}}
{"id": 8, "method": "pose", "params": {}}               ->   {"id": 8, "error": "not_available", "detail": "..."}
```

`params` holds the call's arguments by name (section 1's table). On the wire a frame is
`{index, t, width, height, encoding, rgb_b64, depth_b64, depth_unit_m, frames_left}` with the encoding
`zlib-raw-v1`: `rgb_b64` is base64 of the zlib-compressed `uint8` image, height × width × 3, row by row;
`depth_b64` is base64 of the zlib-compressed little-endian `uint16` image, height × width, row by row. A telemetry
reply carries each column as nested lists, with null in `motor_target_mm` where no command runs, plus `n`, `dropped`
and `columns` (the column names). Anything the program prints on its standard output that is not a request ends the
episode; the given launcher sends `print` to the standard error stream.

## 10. The practice ground

The practice ground is a service in your sandbox that runs the scored episodes' station on four sites:

<!-- BEGIN GENERATED practice -->
| number | name | what | simulated time limit |
|---|---|---|---|
| 1 | `cal` | the calibration site | 1400 s |
| 2 | `p1` | a practice site | 1820 s |
| 3 | `p2` | a practice site | 1820 s |
| 4 | `p3` | a practice site | 1820 s |

Allowance: 30 practice episodes in your whole session, at most 2 running at the same time.
<!-- END GENERATED practice -->

**Running your submission.** `/opt/venv/bin/python3 ~/training/run_mission.py --site <name or number>` runs
`~/submission` (or `--program <folder>`) in one practice episode as section 8 describes, prints one line when the
episode starts, one line `end: ...` with why it ended and what was used, and then the debrief. Options:
`--list` (the sites and the allowance, no episode is used), `--debrief <file>` (also write the debrief as JSON),
`--transcript <file>` (every request and reply as JSON lines, frames elided), `--keep <folder>` (keep the run's `HOME`
and `TMPDIR`). Keep `run_mission.py` out of your submission folder: it starts a process, which the contract forbids.

**Driving the station from your own script.** `voidcell.client.connect_practice(site)` returns a `station` on a new
practice episode; use it from a script of your own, not from `operate.py` (a scored episode has no practice ground to
connect to). One connection is one episode: it begins with the `reset`
that `connect_practice(site)` sends, whose reply is `{ok, site, episode, episodes_left, time_limit_s}`, and it runs
under the budgets of section 8, the silence limit included. When the episode ends, the practice ground sends the
debrief as the connection's last line and closes; the client then raises `StationError("episode_over")` on the next
call and holds the debrief in `station.debrief` and the reason in `station.episode_over`. Call `station.close()` when
you are done.

**Allowance.** An episode counts from its `reset`, whatever happens afterwards; a `reset` that fails uses none. The
file `/run/voidcell/practice_status.json` (what `--list` prints) holds `practice_episodes`, `episodes_used`,
`episodes_left`, `open_at_a_time`, the limits `program_time_s`, `silence_s`, `cpu_s`, the list `sites` (`number`,
`name`, `about`, `time_limit_s`) and the folder `debriefs`. Every debrief is also saved in that folder,
`/run/voidcell/debriefs/`, as `episode_<number>.json` with the episode's number in three digits.

**The debrief.** The last line of a practice connection is `{"event": "episode_over", "reason", "debrief"}`. The
debrief is a JSON object with exactly these keys:

<!-- BEGIN GENERATED debrief_keys -->
| key | content |
|---|---|
| `debrief_version` | the debrief's format version |
| `kind` | `cal` or `practice` |
| `site` | the site's name |
| `episode` | the episode's number in your allowance |
| `ended_by` | why the practice ground closed the episode (`finish`, `eof`, `silence`, `program_time_cap`, `time_limit`, ...) |
| `end_reason` | why the station ended the episode |
| `t_end` | the simulated time at the end (s) |
| `snapshot` | false; true only when the practice ground itself failed and sends its last saved state |

Calibration site (`kind` = `cal`), in addition:

| key | content |
|---|---|
| `true_nodes` | `hz`, `t`, `nodes_xyz`: the six nodes' true positions (site frame, m; one 6 × 3 list per time of `t`), `hz` times a second, under the roof too |
| `primitives` | one entry per primitive that ran: `name`, `bottom_nodes` (as you passed them), `t_start`, `t_end`, `aborted`, and the true displacement `dx_m`, `dy_m`, `dtheta_deg` of the robot's planar pose, in the frame of its pose at the primitive's start (the handbook's convention) |

Practice site (`kind` = `practice`), in addition:

| key | content |
|---|---|
| `reports` | one entry per `report_point` call: `t`, `point`, `accepted`, `counted` (it is the point's last accepted report), `distance_m` (true distance to the point), `dwell_valid`, `dwell_failed` (the conditions that failed: `window_before_episode_start`, `command_in_window`, `moved_in_window`), `credit` |
| `retrieval` | `retrieved`, `within_radius`, `command_running_at_end` |
| `stuck_periods` | `t_start`, `t_end`, `primitives`: runs of at least 3 consecutive primitives that each moved the robot less than 0.03 m and turned it less than 5° |
| `budgets` | `frames_used`, `frames_limit`, `frames_refused`, `reports_used`, `reports_limit`, `reports_refused`, `sim_time_s`, `time_limit_s`, `program_time_s`, `program_time_limit_s` |
| `events` | `t`, `event`, `detail`: `command_refused`, `command_stopped`, `end_triangle` (the robot rests on an end triangle after a primitive), `end` |

The true node positions of the calibration site come 2 times a second. A practice site's debrief holds no position, no path, no layout and no total.
<!-- END GENERATED debrief_keys -->

**Differences from a scored episode.** None in the station, the robot, the rules or the budgets. A practice episode
draws a new set-down and new sensor noise every time; it has a debrief; and your program runs under your own account
(section 8).

<!-- BEGIN GENERATED costs -->
With one episode running, the station takes about 0.23 s of wall clock per simulated second (rolling or standing alike) and about 0.44 s per camera frame. An episode that uses all 1820 s of simulated time therefore takes about 6.9 minutes for the simulation alone, a full calibration-site episode about 5.3, and all 600 frames add about 4.4 minutes, before your program's own computing. The whole allowance spent on full-length episodes with every frame taken, 2 at a time, would come to about 2.8 hours: end a practice episode (`finish()`) as soon as it has shown what you wanted. The sandbox has 8 CPUs and 16 GB of memory, which your program shares with the station, in a practice run as in a scored episode; inside either, the numerical libraries are held to 2 threads.
<!-- END GENERATED costs -->

## 11. What no reply contains

No reply of a scored episode or of a practice episode contains the layout of the site, the position of the robot,
whether a command moved the robot, or the worth of a report. The only place where true positions and credits appear is
the practice ground's debrief, after the episode. A scored episode has no debrief and writes nothing your program
could read back in a later episode.
