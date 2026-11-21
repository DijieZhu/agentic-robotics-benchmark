# The car's program interface

Protocol `racecar-wire/1`. Your program drives the car at a venue through five calls. This file describes them,
the observation every `step` returns, the organiser's messages, the budgets, and the three things in this folder
that help you write and check a program: the client library `carkit/`, the protocol stub `stub/`, and the format of
the lab logs in `../lab_logs/`.

## 1. How your program is run

- You hand in `drive.py` with a function `solve(car)`. At every venue it is started once, as a new process, and
  `solve(car)` is called once. `car` is a `carkit.Car` (section 8). When `solve` returns, or the day at the venue
  is over, the process ends. If `solve` returns or raises an error before the day is over, the day at that venue
  ends there with what was recorded (rules, clause S-18). Nothing is kept from one venue to the next.
- One process runs the three sessions of a venue (rules, clause S-07), so whatever the program keeps in memory in
  practice is still there in the heats.
- Time at the venue passes only inside `step`: one call is one tick of 25 ms. Between two calls the car and the
  world stand still, while the program's own clock runs against its budgets (section 6).
- The messages are JSON lines on the standard output and input your process was started with: one request, one
  reply, strictly in turn. `carkit` binds them before your code runs and sends `print` to the error stream, so
  ordinary prints do not disturb them. You do not need to handle the lines yourself; they are described in section 7.
- The process is started in a read-only copy of the folder you handed in, which is its working directory and the
  first place modules are imported from; `carkit` can be imported as well. The interpreter is started with the
  options `-B -E -s`. The environment holds `PATH`, `HOME` and `TMPDIR` (with `TMP` and `TEMP`; empty folders the
  program may write to), `LANG`, `PYTHONPATH`, `MPLBACKEND`, the two variables that choose the off-screen graphics
  of the given tools (with the values of your session), and nothing else but the four variables of the next item.
  The files of `/home/user/given` are not there: what the program needs from them it carries in its own folder.
- The program runs with single-threaded numeric libraries: the process starts with `OMP_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS` and `MKL_NUM_THREADS` set to 1, and with `NPY_DISABLE_CPU_FEATURES` set to
  `X86_V4 AVX512_ICL AVX512_SPR`. Nothing else about the numeric libraries is fixed.
- The process may run threads of its own, but it cannot start another process: every way of starting one fails
  (`os.fork`, `os.posix_spawn` and the like raise a permission error) and no process is made.

## 2. Calls

| call | parameters | result |
|---|---|---|
| `car.describe()` | none | the venue sheet, the session table, the budgets, the timing and the car's constants (section 3). May be called at any time until the day is over; costs no venue time |
| `car.open_session(name)` | `name`: `"practice"`, `"heat1"`, `"heat2"`, in this order | `session`, `duration_s`, `ticks`, `phases`. The car stands at rest, on the start position in practice and in its slot of the grid in a heat (rules, clauses S-05 and S-21); the session clock is 0 |
| `car.step(steer, speed)` | `steer`: target of the steering angle, rad, left positive; `speed`: target of the forward speed, m/s | the observation of this tick (section 4). Advances the session by one tick |
| `car.close_session()` | none | `session`, `ticks`. Ends the open session now, or acknowledges one that has ended |
| `car.request_pickup()` | none | `accepted`. Asks the staff to carry the stuck car to the side of the track (rules, clause S-16): the next `step` reports `stopped`. `accepted` is false if the car is stopped already or a pick-up is on its way |

- The command of a `step` is clipped to the car's limits (`steer_max_rad`, `speed_cmd_min_mps`, `speed_cmd_max_mps`
  of `describe()`). It acts `cmd_delay_ticks` ticks later: with the value 1, the command sent with one `step` acts
  during the next tick, and the first tick of a session runs with both targets at zero. While the car is stopped
  by the staff, commands are discarded: the first command that can act is the one sent after the program has read
  `released`.
- Sessions cannot be skipped or repeated. After one of the first two sessions has ended (message `session_end`),
  `step` answers with the error `no_session`; `close_session()` acknowledges the end, and `open_session` of the next
  session does so too.
- When the third session has ended or been closed, the day at the venue is over and its results are final. If
  heat 2 ended with its `session_end`, the first `close_session()` after it is still answered like for any
  other session (`session`, `ticks`): it acknowledges the end and changes nothing. Every other call from then on is
  answered with the error `venue_over`.
- The day is also over, at once, when a session ends with the reason `void` (rules, clause S-17): from then on
  every call, `close_session()` included, is answered `venue_over`.
- Once the day is over the organiser closes the connection after a moment; when a budget is exceeded (section 6)
  it is closed at once, without a further reply. A call on a closed connection raises `carkit.CarError` with the
  code `closed`, and the process is ended.
- A call that is not in this table is answered with the error `forbidden` and counted.

## 3. `describe()`

| field | content |
|---|---|
| `protocol` | `"racecar-wire/1"` |
| `venue_sheet` | the venue sheet of this venue (`../league/venues.md`, section 1): `venue_label`, `floor.kind`, `floor.side_pull`, `walls.kinds`, `walls.height_m`, `track.min_width_m`, `track.max_footprint_m`, `features`, `heat_laps` |
| `sessions` | for each session: `name`, `duration_s`, `ticks`, `desk_before_s`, `scored`, and `phases`, each with `name`, `t0_s` (start of the phase on the session clock) and `speed_limit_mps` (or null) |
| `budgets` | `program_ms_per_tick`, `cpu_factor`, `silence_s` (section 6) |
| `timing` | `tick_s` 0.025, `cmd_delay_ticks`, `imu_samples_per_tick` 5, `subscans_per_tick` 5 |
| `car` | the constants of `../car/datasheet.md`: `wheelbase_m`, `track_front_m`, `track_rear_m`, `tyre_radius_m`, `mass_kg`, `com_m`, `length_m`, `width_m`, `height_m`, `steer_max_rad`, `steer_rate_max_radps`, `min_turn_radius_m`, `final_drive_ratio`, `speed_cmd_min_mps`, `speed_cmd_max_mps`, `lidar` (`pos_m`, `beams`, `angle_min_rad`, `angle_inc_rad`, `range_min_m`, `range_max_m`, `no_return`), `imu` (`pos_m`, `rate_hz`) |

The session table of a head-to-head day:

| `name` | `duration_s` | `ticks` | `desk_before_s` | `phases` |
|---|---|---|---|---|
| `practice` | 480 | 19200 | 60 | `mapping` from 0 s with `speed_limit_mps` 2.0; `free` from 120 s |
| `heat1` | 300 | 12000 | 300 | `race` |
| `heat2` | 300 | 12000 | 120 | `race` |

For a heat, `duration_s` and `ticks` are the most it can last: a heat ends earlier when a car completes the laps
of the heat, `venue_sheet.heat_laps` (rules, clause S-23).

`describe()` holds no layout and no position: no map, no centreline, no place of a wall, a gap, an opening, the
timing line or another car, no line or pace of another car, and no grip.

## 4. The observation of a `step`

| field | type | content |
|---|---|---|
| `tick` | integer, 1 and up | ticks of this session completed with this step; session time = `tick` x 0.025 s |
| `scan` | 1081 integers | ranges of the scanning sensor in millimetres; 0 = no return, otherwise 60 to 10000. Index 0 is the beam at -135 degrees (right rear), index 540 straight ahead, index 1080 at +135 degrees, 0.25 degrees apart. The revolution is fired as five sub-scans while the car moves (`../car/datasheet.md`, section 5.1) |
| `imu` | 5 rows of 6 numbers | `[ax, ay, az, gx, gy, gz]`, oldest first, one every 5 ms: specific force in m/s^2 and angular rate in rad/s in the inertial frame |
| `wheel_speed` | number | road speed computed from the motor speed, m/s, signed. Not the speed over the floor when the wheels spin, slide or lock |
| `steer_echo` | number | the steering servo's present target, rad |
| `events` | list | the organiser's messages that fell into this tick, in order (section 5); usually empty |

With `carkit` and numpy, `scan` and `imu` arrive as arrays.

There is nothing else. No call returns the car's position, heading or speed over the floor, a map, what the car
touched, how far something was pushed, whether a contact was a touch or a crash, or a score. Nothing is returned
about another car either: not where it is or how fast, not its line, its laps or whether its staff are handling
it, not an overtake, a contact, a fault or the standing. Another car is in the `scan` like everything else the
sensor sees, and nowhere else. While the car is stopped by the staff the observations go on: the sensors show the
car being set down at the side of the track.

## 5. Messages of the organiser (`events`)

Each event is an object with a `type` and the fields below.

| `type` | fields | meaning |
|---|---|---|
| `lap` | `count`, `lap_ms` | the car's reference point (the midpoint of the wheelbase) crossed the timing line in the driving direction. `lap_ms` is the organiser's time of the lap this crossing completed, in ms; it is null when the crossing only started the clock: the first crossing of a session, or a crossing without a whole lap behind it. `count` = laps completed in this session so far |
| `stopped` | none | the staff have stopped the car; commands are discarded until `released`. No reason is given |
| `released` | none | the car stands at the side of the track, the track is repaired, the car may go |
| `warning` | `count` | in practice, speeding in the mapping period; in a heat, a warning of that heat (rules, clause S-31), sent in the tick of the `stopped` it belongs to. `count` = 1, 2, 3; the third ends the session |
| `phase` | `name` | the session entered a later phase: `free` in practice. It comes with the first tick of the new phase. A heat has one phase, `race`, and sends no `phase` |
| `session_end` | `reason` | the session is over: `time` (its length is reached), `laps` (a car completed the laps of the heat), `warnings` (third warning), `void` (the remaining sessions of the venue are void) |

`carkit.SessionLog` keeps count of them for you (section 8).

The `lap` events are the car's own laps: `count` is the number the rules compare with the laps of the heat (rules,
clauses S-23 and S-25). No message tells of a lap of another car: when a heat ends with the reason `laps` while
the car's own `count` is below `heat_laps`, another car completed them first. A crossing of the timing line
against the driving direction gives no message.

## 6. Budgets

The league wants all computing on the car (rules, clause S-11): the time your program may spend is limited.
Program time is the wall-clock time between a reply and your next request; the time the venue itself needs to
answer a `step` is not charged.

| budget | rule |
|---|---|
| on the track | summed over a session, program time may not exceed `program_ms_per_tick` (15 ms) x the ticks of the session: 288 s in practice, 180 s in each heat. The ticks of a heat are those of the session table, however early the heat ends. The time may be spent unevenly, and the time while the car is stopped by the staff counts like any other |
| desk time | program time between the end of a session (or the first reply of the day) and the next `open_session` may not exceed that session's `desk_before_s`: 60 s, 300 s, 120 s |
| start | the first request must come within 120 s of the start of the process; this time is charged to no other budget |
| CPU | over the whole day at a venue, the CPU time of the process, all its threads together, may not exceed `cpu_factor` (4) x the sum of the budgets on the track and at the desk, that is 4 x 1128 s |
| silence | no more than `silence_s` (180 s) between a reply and the next request |

The silence budget holds at the desk as well: a desk computation that takes longer than `silence_s` must make a call
in between, for example `describe()`, at least every `silence_s`.

If a budget is exceeded, the program ends with an error or closes its output, a line is not a request, or a line
is longer than 4 MiB, the day at that venue ends there; what was recorded before stays (rules, clause S-18). A
program whose decisions depend on the wall clock will not drive the same way twice. Nothing forbids it; your
results will vary with the load of the machine.

## 7. The lines on the wire

For completeness; `carkit` writes and reads them.

- Request: `{"id": n, "method": "<name>", "params": {...}}`, one JSON object per line, UTF-8, ended by a newline;
  `id` counts up from 1; `params` may be left out for a call without parameters. Methods: `describe`,
  `open_session` (`name`), `step` (`steer`, `speed`), `close_session`, `request_pickup`.
- Reply: `{"id": n, "result": {...}}` or `{"id": n, "error": "<code>", "detail": "<text>"}`.

| error | when | effect |
|---|---|---|
| `forbidden` | the method is not one of the five | counted; nothing else |
| `bad_params` | a parameter is missing, extra, of the wrong type or not a finite number; or the request has no proper `id` or has a key other than `id`, `method`, `params` | counted; no time passes |
| `no_session` | `step`, `close_session` or `request_pickup` while no session is open | nothing |
| `bad_order` | `open_session` with a name that is not the next session, or while a session is open | nothing |
| `venue_over` | any call after the day at the venue is over | nothing |

An error reply never advances the venue's time. `carkit` raises `carkit.CarError` (with `.code`) for it.

## 8. `carkit`: the client library

Standard library only; numpy is used when it can be imported.

```python
# drive.py
from carkit import SessionLog, scan

def solve(car):
    info = car.describe()                      # section 3
    for name in ("practice", "heat1", "heat2"):
        car.open_session(name)                 # also acknowledges the session before it, if that has ended
        log = SessionLog(name)
        while log.ended is None:               # until the `session_end` message
            r = car.step(0.0, 1.0)             # steer (rad), speed (m/s): one tick of 25 ms
            log.feed(r)                        # log.laps, log.stopped, log.warnings, log.phase, log.ended
            xy, idx = scan.points(r["scan"], frame="car")   # the returns as points in the car frame, in metres
        if log.ended == "void":                # the day is over at once (section 2)
            return
        car.close_session()                    # acknowledges the end (after heat 2 too)
    # the day is over: any further call is answered `venue_over`
```

| name | what |
|---|---|
| `carkit.Car` | `describe()`, `open_session(name)`, `step(steer, speed)`, `close_session()`, `request_pickup()`; they are built on `call(method, **params)`, which sends one request of any name |
| `carkit.CarError` | raised on an error reply; `.code`, `.detail`. The code is `closed` when the connection is gone |
| `carkit.SessionLog` | `feed(reply)` after every `step`; then `tick`, `laps` (lap times in s), `lap_count`, `clock_started`, `stopped`, `stops`, `warnings`, `phase` (None until the first `phase` message of the session), `ended`, `best_lap_s`, `history` |
| `carkit.scan` | beam geometry: `N_BEAMS`, `ANGLE_MIN_RAD`, `ANGLE_INC_RAD`, `RANGE_MIN_M`, `RANGE_MAX_M`, `NO_RETURN`, `SUBSCAN_SLICES`, `LIDAR_POS_M`, `beam_angle(i)`, `beam_index(angle)`, `beam_angles()`, `ranges_m(scan)`, `points(scan, frame)`, `subscan_of(i)` |
| `carkit.read_log(path)` | reads a lab log (section 10) |
| `carkit.connect_handler(fn)` | a `Car` whose requests are answered by your own function `fn(line) -> line` in the same process, for a test harness of your own; `connect_socket(path)` does the same over a Unix socket |

## 9. `stub/`: the protocol stub

`stub/stub.py LOG.jsonl PROGRAM_DIR` starts the program in `PROGRAM_DIR` the way a venue does and answers its calls
by the rules of this file: the five calls, the errors, the order of the sessions, the budgets. For the
observations it replays one lab log: the reply to the k-th `step` of a session carries the k-th row of the log,
and the session ends after the last row, so the sessions of the stub are as long as the log and their budgets on
the track are counted on that length. With `--report FILE` the report is written to a file, with `--strict` the
run ends at the first exceeded budget, as at a venue; without it the stub notes the excess and goes on. Unlike a
venue, the stub runs the program in `PROGRAM_DIR` itself, not in a read-only copy, and with your own `HOME` and
`TMPDIR`.

The stub simulates nothing. It has no car, no track, no other cars, no laps, no warnings, no touch and no crash;
a session of the stub ends only with the reason `time`; your commands
are recorded and ignored, and its venue sheet is a placeholder. What it tells you is whether your program speaks
the protocol and how its time compares with the budgets: it reports the requests per call, the errors, the program
time per step (median, 95th percentile, largest), each session's time on the track and each desk time against its
budget, and the longest silence. Nothing in this folder, and nothing else you are given, simulates a venue.

## 10. Lab logs

A lab log (`../lab_logs/*.jsonl`) is a JSON-lines file: one header line, then one line per tick.

| line | fields |
|---|---|
| header | `schema` (`"racecar-lab-log/1"`), `name`, `description`, `floor` (`kind`, `grip`), `tick_s` (0.025), `tracker_rate_hz` (100), `ticks` |
| tick | `tick` (1, 2, ...), `cmd` = `[steer, speed]` sent with that step, `obs` = `scan`, `imu`, `wheel_speed`, `steer_echo` exactly as in section 4, `tracker` |

`tracker` holds the samples of the lab's overhead tracker that fall into the tick: `[t_ms, x, y, yaw]`, the time
since the start of the log in ms and the pose of the car frame in the lab frame (m, rad), 100 per second. A venue
has no such tracker.

```python
from carkit import read_log
log = read_log("../lab_logs/circle_r10_ramp.jsonl")
log.header, log.cmd, log.scan, log.imu, log.wheel_speed, log.steer_echo, log.tracker   # arrays with numpy
```
