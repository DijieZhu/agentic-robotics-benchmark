# CFR-1 program interface (`cabinkit`)

Your program drives the robot through `cabinkit`: a `Robot` object whose methods are the calls of this document.
Each call is a request to the robot's cell and returns its answer. This document lists every call with its arguments,
its replies and return codes, the budgets, and how a run ends. What the robot is and what its firmware does on its
own is in `robot_manual.md`.

## 1. Your program

```python
# ~/submission/caretake.py
def solve(robot):
    m = robot.mission()                 # the documents of this site
    r = robot.undock()                  # an action call returns a dict with "code" and "t"
    f = robot.camera("FIC")             # a frame: f["image"], f["frame_id"]
    robot.step(5.0)                     # lets simulated time pass
    robot.dock(m["berth"])
```

- The cell imports `caretake.py` and calls `solve(robot)`, in a fresh process for every site. `robot` is the only
  way to the robot and the station.
- The program runs without network access on a read-only copy of `~/submission/`, started as
  `python3 -B -E -s`. Besides the standard library it can import `numpy`, `opencv-python-headless`, `scipy`.
- `print` goes to a log, not to the cell.
- The run of a site ends when `solve` returns or raises, or when a budget ends it (section 6).
- The form of the submission is checked as well (the task instruction has the rules). Of this interface the check
  asks that on every site at least 1 call lets simulated time pass: `step`, or any of the calls that wait
  for the robot (section 3). A program that lets no simulated time pass on a site does not pass it.

## 2. Conventions

- **Arguments** are passed by position or by name, as in the signatures. Numbers have to be finite and not above
  1e6 in magnitude; vectors are lists (numpy arrays are converted). A call with a missing, unknown or malformed
  argument returns `INVALID_ARGS`, does nothing and uses no budget; some of these replies carry a `detail` text that
  says what was wrong.
- **Replies.** The calls that act return a dict with `code` (a return code, section 5), `t` (mission seconds) and
  the keys named with the call. The queries `mission`, `status`, `pose` and `station_telemetry` return a dict of
  data without a code.
- **Units and frames**: metres, seconds, radians; the arm calls take degrees. Positions and attitudes are in the
  station frame, except where a call says body frame or plate frame (`robot_manual.md`, frames). Quaternions are
  (w, x, y, z), body to station.
- **Mission seconds** count the simulated time of the run from the program's start.
- **Ids**: return inlets `RI-<module>-<n>`; cabin air units `CAU-<module>`; stowage units `SB-<module>` with the
  holding screen `SB-<module>-S`, the pouch pads `-P1` and `-P2` and the fiducial plates `-F1` to `-F4`; the dock's
  plate `DOCK-F1`; register references `LI-<nn>`; modules `L2`, `H2`, `L1`.
- **Calls outside this interface** raise `cabinkit.client.CellError` with the code `forbidden`; the cell counts
  them. After the cell has closed the run, every call raises `CellError` with the code `run_over` (section 6).

## 3. Time

Simulated time passes only inside calls, never between them.

- It passes inside `step(dt)` and inside the calls that wait for the robot: `set_mode`, `fly_to`, `fly_rel`, `gripper`, `capture`, `dock`, `undock`, and the
  arm calls (`arm_deploy`, `arm_stow`, `arm`) with `wait=True`.
- Every other call answers at the same simulated instant: `mission`, `status`, `pose`, `stop`, `hold`, `camera`, `lamp`, `station_telemetry`, `station_request`, `report`, and the arm calls
  without `wait`.
- What the program computes between calls costs no simulated time. It costs program time, which has its own
  budget (section 6).

## 4. The calls

### 4.1 `mission()`

The documents of this site. Takes no time.

| key | content |
|---|---|
| `work_order` | the work order, text |
| `log` | the station's telemetry log before the sortie: `{"t_min": [...], "<channel>": [...], ...}`, 72 h at a row every 1 min (4320 rows). `t_min` counts the minutes of the log from 0; the sortie starts a row period after the last row: mission second 0 |
| `config_record` | `{"units": {"CAU-<module>": {"build", "clean_reference_pa", "advisory_limit_pa", "fan_stop", "inhibit_reason"}}, "lights": {"<module>": "general" or "night"}}`; `build` is `"pad-type"` or `"element-type"`, `fan_stop` is `"allowed"` or `"inhibited"` |
| `advisory_log` | the advisories the station raised: a list of `{"unit", "t_min", "mean_pa", "limit_pa"}` |
| `register` | this site's lost-item register: a list of `{"ref", "class", "colour", "size_m", "mass_kg", "features", "location"}` |
| `map` | `{"zones", "fixtures", "landmarks"}`: the content of `~/given/station_map/` |
| `battery_pct` | the charge at the start, percent |
| `berth` | the berth the robot starts on |

What the entries mean is in `station_handbook.md`, `telemetry_dictionary.md` and `lost_item_register.md`.

```python
m = robot.mission()
unit = m["config_record"]["units"]["CAU-L2"]      # unit["build"], unit["fan_stop"]
minutes = m["log"]["t_min"]
```

### 4.2 `status()`

The robot's state. Takes no time.

| key | content |
|---|---|
| `motion` | `DOCK`: on the berth. `FLY`: a move is running or the robot is braking. `STOP`: holding its place. `DRIFT`: the controller is idle or the confidence is not `GOOD` |
| `last_result` | the code of the last `fly_to`, `fly_rel`, `capture`, `dock` or `undock`; `None` before any |
| `battery_pct` | charge in percent, refreshed at 1 Hz in steps of 0.1 |
| `low_battery` | `True` below 25 % |
| `flight_mode` | `STANDARD`, `AGILE`, `HUSH`, `FINE`, or `OFF` on the berth with the fans stopped |
| `fan_level` | the fan level of the flight mode, 0 to 3 |
| `arm_tilt_deg`, `arm_pan_deg` | the arm joint encoders, degrees |
| `tendon_pct` | the tendon motor's position, percent (100: wound, 0: let go); not the fingers' position |
| `arm_current_a` | `[tilt, pan]`: the currents of the arm joint servos, amperes |
| `imu` | `{"accel_m_s2": [x, y, z], "gyro_rad_s": [x, y, z]}`: the inertial unit's latest sample in the body frame (62.5 Hz) |
| `interlock` | `True` while the holding interlock is on |
| `t` | mission seconds |

```python
s = robot.status()
charge, mode = s["battery_pct"], s["flight_mode"]
```

### 4.3 `pose()`

The firmware's estimate of where the robot is. Takes no time.

| key | content |
|---|---|
| `pos` | `[x, y, z]` of the body origin in the station frame, metres |
| `quat` | `[w, x, y, z]`, body to station |
| `cov_diag` | 6 variances: position x, y, z in m², then attitude about x, y, z in rad² |
| `confidence` | `GOOD`, `POOR` or `LOST` |
| `loc_mode` | `MAP`, `FIDUCIAL` or `RAIL` |
| `t` | mission seconds |

```python
p = robot.pose()
var_x, var_y, var_z, *attitude_vars = p["cov_diag"]
good = p["confidence"] == "GOOD"
```

### 4.4 `step(dt)`

| argument | type | default | meaning |
|---|---|---|---|
| `dt` | number | required | simulated seconds to advance |

Lets simulated time pass while the robot keeps doing what it does. A `dt` above 5 s is cut to 5 s (a
call with `dt = 12` advanced the clock by 5 s); a `dt` that is not a positive, finite number returns
`INVALID_ARGS` and lets no time pass. For the check of the submission's form a `step` counts like every other call
in which simulated time passes (section 1).

Returns `OK`, `INVALID_ARGS`, `TIME_UP`.

```python
robot.step(5.0)
```

### 4.5 `set_mode(mode)`

| argument | type | default | meaning |
|---|---|---|---|
| `mode` | string | required | `STANDARD`, `AGILE`, `HUSH` or `FINE` |

Selects the flight mode and waits until the fans run at its level (`robot_manual.md`, flight modes). `OFF` cannot
be selected. The call is also taken on the berth: the fans then spin up there (15 s from rest), `flight_mode` reports
the mode and the robot stays berthed.

Returns `DONE`, `INVALID_ARGS`, `TIME_UP`.

```python
robot.set_mode("FINE")
```

### 4.6 `fly_to(pos, quat, face_forward=True, max_speed=None)`

| argument | type | default | meaning |
|---|---|---|---|
| `pos` | list of numbers [x, y, z] | required | goal position [x, y, z] in the station frame, metres |
| `quat` | list of numbers (w, x, y, z) | required | goal attitude (w, x, y, z) in the station frame |
| `face_forward` | `True` or `False` | `True` | turn the forward face along the segment before flying it |
| `max_speed` | number or `None` | `None` | a speed limit in m/s; it has an effect only below the flight mode's limit |

Flies a straight segment from the present estimate to the goal and comes to rest there
(`robot_manual.md`, flight modes and moves). Blocks until the move has ended.

- The segment is checked against the zone map before the robot moves (`REFUSED_ZONE`); with the arm deployed the
  larger inflation applies, without exception.
- `LOCALIZATION` at once when the confidence is not `GOOD`; `REFUSED_STATE` on the berth.
- A `max_speed` below 0.005 m/s, or a `quat` of zero length, is `INVALID_ARGS`.
- `SUCCESS` says that the robot is at rest within the end tolerance of the goal, on the estimate.

Returns `SUCCESS`, `TOLERANCE_PATH`, `TOLERANCE_ATT`, `TOLERANCE_END`, `LOCALIZATION`, `OVERSPEED_STOP`, `REFUSED_ZONE`, `TIMEOUT`, `REFUSED_STATE`, `INVALID_ARGS`, `TIME_UP`.

```python
p = robot.pose()
r = robot.fly_to(pos=goal, quat=p["quat"], face_forward=False)      # goal: [x, y, z], station frame
if r["code"] != "SUCCESS":
    ...
```

### 4.7 `fly_rel(fiducial_id, pos, quat, max_speed=None)`

| argument | type | default | meaning |
|---|---|---|---|
| `fiducial_id` | string | required | the id of a fiducial plate of the map |
| `pos` | list of numbers [x, y, z] | required | goal position [x, y, z] in the plate frame, metres |
| `quat` | list of numbers (w, x, y, z) | required | goal attitude (w, x, y, z) in the plate frame |
| `max_speed` | number or `None` | `None` | as in `fly_to` |

Flies to a goal given in the frame of a fiducial plate, on the estimate that the aft wide camera takes from that
plate (`loc_mode` `FIDUCIAL` while the call runs). The plate frame sits on the plate where it really is.

- Plates: `SB-<module>-F1` and `-F2` beside the holding screen of the module's stowage unit (`-F1` on its deck side,
  `-F2` on its overhead side), `-F3` beside pouch pad `-P1`, `-F4` beside pouch pad `-P2`, and `DOCK-F1` on the dock.
  `station_map/fixtures.json` lists them with their frames. An id that is not a plate of the map is `INVALID_ARGS`.
- The plate has to be seen by the aft wide camera at the call, else `NO_FIDUCIAL`; what counts as seen is in
  `robot_manual.md`, localisation. Lost for more than 20 s on the way: the robot stops, `NO_FIDUCIAL`.
- The firmware takes 1 s on the plate before it plans the move; the move is flown as `fly_to` flies without
  `face_forward`.
- Zone check as `fly_to`. With the arm deployed the larger inflation applies except inside the approach zone of
  the plate's own stowage unit, or of the dock for `DOCK-F1` (`robot_manual.md`, the zone check).
- A `max_speed` below 0.005 m/s, or a `quat` of zero length, is `INVALID_ARGS`, as in `fly_to`.
- When the call ends the estimate is the map's again (`loc_mode` `MAP`).

Returns `SUCCESS`, `TOLERANCE_PATH`, `TOLERANCE_ATT`, `TOLERANCE_END`, `LOCALIZATION`, `OVERSPEED_STOP`, `REFUSED_ZONE`, `TIMEOUT`, `NO_FIDUCIAL`, `REFUSED_STATE`, `INVALID_ARGS`, `TIME_UP`.

```python
r = robot.fly_rel("SB-L2-F1", pos=goal_in_plate_frame, quat=attitude_in_plate_frame)
if r["code"] == "NO_FIDUCIAL":
    ...
```

### 4.8 `stop()`

Brakes and holds the place where the robot comes to rest, on the present estimate. Returns at once.

Returns `OK`, `TIME_UP`.

```python
robot.stop()
```

### 4.9 `hold()`

Keeps station at the present estimate. Returns at once. `stop` and `hold` each end the hold that follows a
`capture` (`robot_manual.md`, what a call leaves behind).

Returns `OK`, `TIME_UP`.

```python
robot.hold()
```

### 4.10 `arm_deploy(wait=False)`

| argument | type | default | meaning |
|---|---|---|---|
| `wait` | `True` or `False` | `False` | block until the joints have arrived |

Moves the arm to the grasp attitude (tilt 0, pan 0). Without `wait` the call returns `STARTED` at once and the
arm moves while the program goes on; with `wait=True` it blocks and returns `DONE`.

- `REFUSED_HOLDING` while the holding interlock is on; the joints keep their angles.
- `REFUSED_ZONE` when the robot's estimated position is not free with the arm's inflation (`robot_manual.md`, the
  zone check). `REFUSED_STATE` on the berth.

Returns `STARTED`, `DONE`, `REFUSED_HOLDING`, `REFUSED_ZONE`, `REFUSED_STATE`, `INVALID_ARGS`, `TIME_UP`.

```python
r = robot.arm_deploy(wait=True)
```

### 4.11 `arm_stow(wait=False)`

| argument | type | default | meaning |
|---|---|---|---|
| `wait` | `True` or `False` | `False` | block until the joints have arrived |

Moves the arm to the stowed attitude (tilt 180, pan 0); `wait` as in `arm_deploy`. `REFUSED_HOLDING` while the
holding interlock is on. No zone check; the call is taken on the berth too.

Returns `STARTED`, `DONE`, `REFUSED_HOLDING`, `INVALID_ARGS`, `TIME_UP`.

```python
robot.arm_stow()            # returns STARTED; the arm folds while the robot flies on
```

### 4.12 `arm(tilt_deg, pan_deg, wait=False)`

| argument | type | default | meaning |
|---|---|---|---|
| `tilt_deg` | number | required | tilt goal in degrees |
| `pan_deg` | number | required | pan goal in degrees |
| `wait` | `True` or `False` | `False` | block until the joints have arrived |

Moves the arm to a tilt and a pan; `wait` as in `arm_deploy`. Tilt from -30 to 180, pan from -90 to 90
(`robot_manual.md`, joints); a goal outside is `INVALID_ARGS`. Interlock, zone check (for a tilt that counts as
deployed) and berth as in `arm_deploy`.

Returns `STARTED`, `DONE`, `REFUSED_HOLDING`, `REFUSED_ZONE`, `REFUSED_STATE`, `INVALID_ARGS`, `TIME_UP`.

```python
r = robot.arm(tilt_deg=tilt, pan_deg=pan, wait=True)
```

### 4.13 `gripper(open)`

| argument | type | default | meaning |
|---|---|---|---|
| `open` | `True` or `False` | required | `True`: wind the tendon, the fingers open; `False`: let it go, the springs close the fingers |

Winds or lets go the tendon and waits for the stroke (1.3 s). The reply carries `tendon_pct`, the tendon motor's
position after the call. `gripper(open=True)` takes the holding interlock off at once. The call does not say what
the fingers closed on.

Returns `DONE`, `INVALID_ARGS`, `TIME_UP`.

```python
r = robot.gripper(open=True)        # r["tendon_pct"]
```

### 4.14 `capture(bar_pos, bar_axis, palm_dir=None)`

| argument | type | default | meaning |
|---|---|---|---|
| `bar_pos` | list of numbers [x, y, z] | required | centre of the bar [x, y, z] in the body frame at the moment of the call, metres |
| `bar_axis` | list of numbers [x, y, z] | required | direction of the bar's axis in the body frame (any length, either sense) |
| `palm_dir` | list of numbers [x, y, z] or `None` | `None` | a direction in the body frame that picks the attitude of the closing; `None`: the smaller turn |

Closes the gripper around the bar that the program points at. The firmware locks a bar that its detector sees
within 4 cm and 8° of the hint, flies the hand onto it and closes the fingers (`robot_manual.md`, the capture hint
and the closing). Blocks until the action has ended.

- The arm has to be at the grasp attitude, else `REFUSED_STATE`; `REFUSED_HOLDING` while the holding interlock is
  on; `REFUSED_STATE` on the berth.
- Each call that starts uses one of the site's 20 captures, whatever its ending; a refused call and an
  `INVALID_ARGS` use none (1 counted of 3 calls when 2 were refused). With the budget used up: `REFUSED_BUDGET`.
- `DONE` says that the closing was carried out. The holding interlock is on from then. The robot is left in
  `FINE`, holding its place.
- The approach after the lock is not zone-checked.

Returns `DONE`, `NO_BAR`, `LOST_BAR`, `STALLED`, `TIMEOUT`, `REFUSED_BUDGET`, `REFUSED_HOLDING`, `REFUSED_STATE`, `INVALID_ARGS`, `TIME_UP`.

```python
r = robot.capture(bar_pos=centre_in_body_frame, bar_axis=axis_in_body_frame)
if r["code"] == "NO_BAR":
    ...
```

### 4.15 `dock(berth)`

| argument | type | default | meaning |
|---|---|---|---|
| `berth` | integer | required | the berth's number |

Berths the robot on berth 1 or 2 (`robot_manual.md`, dock and berths). Blocks until the robot is berthed or the call
has given up. The firmware flies the approach and the berthing itself; the arm has to be stowed.

Returns `DOCKED`, `REFUSED_CONE`, `REFUSED_ARM`, `FAILED`, `REFUSED_STATE`, `INVALID_ARGS`, `TIME_UP`.

```python
r = robot.dock(robot.mission()["berth"])
```

### 4.16 `undock()`

Leaves the berth and stops at its approach point, in `STANDARD`. `REFUSED_STATE` when the robot is not berthed.

Returns `UNDOCKED`, `FAILED`, `REFUSED_STATE`, `TIME_UP`.

```python
r = robot.undock()
```

### 4.17 `camera(name)`

| argument | type | default | meaning |
|---|---|---|---|
| `name` | string | required | `FIC`, `FWC`, `AWC` (colour) or `FDC`, `ADC` (depth) |

Takes a frame at the present instant. Takes no simulated time.

| key | content |
|---|---|
| `frame_id` | an integer that is unique on the site, counted from 1; `report` cites it |
| `camera`, `width`, `height` | the camera's name and the frame's size in pixels |
| `intrinsics` | `{"fx", "fy", "cx", "cy"}` of the frame, pixels |
| `image` | colour cameras: a numpy array of uint8, height by width by the channels R, G, B |
| `depth` | depth cameras: a numpy array of float32, height by width, metres along the optical axis; 0 where nothing returns |
| `amplitude` | depth cameras: a numpy array of uint16, height by width |
| `depth_unit_m` | depth cameras: the step of the delivered depth values, metres |
| `t` | mission seconds of the frame |

A colour frame uses one of the site's 400 colour frames (`FIC`, `FWC` and `AWC` together), a depth frame one of its
300 depth frames (`FDC` and `ADC` together); with the budget used up the call returns `REFUSED_BUDGET` and no frame.
The cameras themselves are described in `robot_manual.md`.

Returns `OK`, `REFUSED_BUDGET`, `INVALID_ARGS`, `TIME_UP`.

```python
f = robot.camera("FIC")             # f["image"], f["frame_id"], f["intrinsics"]["fx"]
d = robot.camera("ADC")             # d["depth"], d["amplitude"]
```

### 4.18 `lamp(which, on)`

| argument | type | default | meaning |
|---|---|---|---|
| `which` | string | required | `forward` or `aft` |
| `on` | `True` or `False` | required | `True`: on; `False`: off |

Switches a lamp; each lamp that is on draws 3.3 W. Returns at once.

Returns `OK`, `INVALID_ARGS`, `TIME_UP`.

```python
robot.lamp("forward", True)
```

### 4.19 `station_telemetry()`

The station's live telemetry: `{"t": mission seconds, "channels": {name: latest value}}`, refreshed at
1 Hz. Takes no time.

| channel | unit | content |
|---|---|---|
| `cau.<module>.return_dp_a` | Pa | return-loop differential pressure, sensor a |
| `cau.<module>.return_dp_b` | Pa | return-loop differential pressure, sensor b (redundant) |
| `cau.<module>.fan_speed` | % of design | cabin air unit fan speed |
| `cau.<module>.fan_current` | A | supply channel current (other loads included) |
| `cau.<module>.state` | - | running or stopped |
| `cau.<module>.stops_used` | - | stops used in this mission |
| `co2.<module>` | mmHg | partial pressure of CO2 (one analyser, sampled in turn) |
| `cabin.<module>.temp` | degC | cabin temperature |
| `cabin.<module>.dew_point` | degC | dew point |
| `lights.<module>` | - | commanded lighting state general or night |
| `imd.<module>-<module>.last_survey` | m/s + date | last manual anemometer survey at the hatch |
| `advisory_log` | - | advisories raised by the station health monitor |

`<module>` is `L2`, `H2` or `L1`; the hatch channels are `imd.L2-H2.last_survey` and `imd.H2-L1.last_survey`. What
each channel measures, and how well, is in `telemetry_dictionary.md`.

```python
tel = robot.station_telemetry()
dp = tel["channels"]["cau.L2.return_dp_a"]
```

### 4.20 `station_request(kind, target, value)`

| argument | type | default | meaning |
|---|---|---|---|
| `kind` | string | required | `cau_fan` or `lights` |
| `target` | string | required | the module: `L2`, `H2` or `L1` |
| `value` | string | required | `cau_fan`: `off` or `on`; `lights`: `general` or `night` |

Asks the station to stop or restart a module's cabin air unit fan, or to switch a module's lights. Returns at once
with the station's answer; what the request does shows in the telemetry and in the frames only.

- `cau_fan`, `off`: the unit's fan runs down and every return inlet of the unit loses its suction; the blowers of the
  stowage units are not affected. A unit takes at most 2 stops in a sortie. The station restarts a stopped fan by itself 60 s after the stop; the reply to an
  accepted stop carries `auto_restart_at`, the mission second of that restart. A new stop is accepted
  120 s after the unit's last restart at the earliest. The checks come in this order: `REJECTED_INHIBITED`, `REJECTED_STATE`, `REJECTED_QUOTA`, `REJECTED_INTERVAL`.
- `cau_fan`, `on`: restarts a stopped fan; for a running fan the answer is `ACCEPTED` and nothing changes.
- The station counts as violations of its flight rules: a stop request beyond the unit's quota, a stop request
  inside the interval, and a stop that lasts until the station's own restart. A request to an inhibited unit, `off`
  for a stopped fan and `on` for a running fan are not violations.
- `lights`: at most 6 requests in a sortie, then `REJECTED_QUOTA`. A module whose light circuit has failed takes the
  request and stays dark; the channel `lights.<module>` reports the commanded state.

Returns `ACCEPTED`, `REJECTED_INHIBITED`, `REJECTED_QUOTA`, `REJECTED_INTERVAL`, `REJECTED_STATE`, `INVALID_ARGS`, `TIME_UP`.

```python
r = robot.station_request("cau_fan", "L2", "off")
if r["code"] == "ACCEPTED":
    restart = r["auto_restart_at"]
```

### 4.21 `report(inlet_id, item_ref, onset_time, frame_id)`

| argument | type | default | meaning |
|---|---|---|---|
| `inlet_id` | string | required | the id of the blocked return inlet, as in the map |
| `item_ref` | string | required | the reference of the blocking item in this site's lost-item register |
| `onset_time` | number | required | when the blockage began, in minutes on the time axis of the log |
| `frame_id` | integer | required | the `frame_id` of a frame of this run that shows the inlet |

Records the program's diagnosis: which return inlet is blocked, by which item of the register, and since when.
Only the first valid report is recorded; a later one returns `ALREADY_REPORTED`. A report is not valid, returns
`INVALID_ARGS` and does not use up the report, when `inlet_id` is not an inlet of the map, `item_ref` is not a
reference of this site's register, `onset_time` is not a finite number, or `frame_id` is not the id of a frame taken
in this run before the call.

**The frame that is cited has to show the inlet.** A frame of the forward imaging camera (`FIC`) and a frame of
the forward wide camera (`FWC`) count alike; frames of the other cameras do not count. With the robot where it truly
was when the frame was taken:

- the centre of the inlet's opening lies inside the frame;
- the lens is at most 2.0 m from it;
- the line of sight makes at most 60° with the inlet's normal;
- the long side of the opening spans at least 80 px in the frame;
- of 20 lines of sight to points on the inlet's frame, 5 on each side, at most 20 % are blocked by the station, by the
  robot itself or by an item that is not on this inlet;
- the blocking item still covers more than 10 % of the opening at that moment.

Returns `RECORDED`, `ALREADY_REPORTED`, `INVALID_ARGS`, `TIME_UP`.

```python
f = robot.camera("FIC")
r = robot.report(inlet_id="RI-L2-1", item_ref="LI-03", onset_time=onset_minute, frame_id=f["frame_id"])
```

## 5. Return codes

| code | meaning | returned by |
|---|---|---|
| `ACCEPTED` | the station took the request | `station_request` |
| `ALREADY_REPORTED` | a report was recorded before; this call changes nothing | `report` |
| `DOCKED` | berthed and held by the dock | `dock` |
| `DONE` | the action was carried out; for `capture`: the closing was carried out, which does not say that anything is held | `set_mode`, `arm_deploy`, `arm_stow`, `arm`, `gripper`, `capture` |
| `FAILED` | `dock`: the berthing ended without the dock holding the robot; `undock`: the robot did not get out to the approach point | `dock`, `undock` |
| `INVALID_ARGS` | a missing, unknown, malformed or out-of-range argument; nothing happened and no budget was used; some replies carry a `detail` text | any call given such an argument |
| `LOCALIZATION` | the confidence of the estimate was not `GOOD` at the start or left `GOOD` during the move | `fly_to`, `fly_rel` |
| `LOST_BAR` | the locked bar was unseen for 3 s before the fingers were lined up | `capture` |
| `NO_BAR` | no bar locked inside the gate within 5 s | `capture` |
| `NO_FIDUCIAL` | the plate was not seen at the call, or was lost for more than 20 s during the move | `fly_rel` |
| `OK` | carried out | `step`, `stop`, `hold`, `camera`, `lamp` |
| `OVERSPEED_STOP` | the move was stopped because the estimated speed passed an overspeed level (`robot_manual.md`, fault responses) | `fly_to`, `fly_rel` |
| `RECORDED` | the report is recorded | `report` |
| `REFUSED_ARM` | the arm joints do not read stowed | `dock` |
| `REFUSED_BUDGET` | the site's budget of this call is used up | `capture`, `camera` |
| `REFUSED_CONE` | the start is more than 1.0 m from the berth's approach point, or the start of the berthing stayed outside its limits after the retries | `dock` |
| `REFUSED_HOLDING` | the holding interlock is on: no arm motion and no new `capture` until `gripper(open=True)` | `arm_deploy`, `arm_stow`, `arm`, `capture` |
| `REFUSED_STATE` | not possible in the present state: on the berth (`fly_to`, `fly_rel`, `arm_deploy`, `arm`, `capture`, `dock`), off the berth (`undock`), or the arm not at the grasp attitude (`capture`) | `fly_to`, `fly_rel`, `arm_deploy`, `arm`, `capture`, `dock`, `undock` |
| `REFUSED_ZONE` | the zone check refuses the move or the arm motion (`robot_manual.md`, the zone check) | `fly_to`, `fly_rel`, `arm_deploy`, `arm` |
| `REJECTED_INHIBITED` | stop requests for this cabin air unit are inhibited (configuration record) | `station_request` |
| `REJECTED_INTERVAL` | a stop request less than 120 s after the unit's last restart | `station_request` |
| `REJECTED_QUOTA` | the quota is used up: a stop request after 2 stops of the unit, or a lights request after 6 | `station_request` |
| `REJECTED_STATE` | `off` for a fan that is already stopped | `station_request` |
| `STALLED` | after the lock the fingertips came no nearer to the closing point for 20 s | `capture` |
| `STARTED` | the arm motion has begun; the call did not wait for it | `arm_deploy`, `arm_stow`, `arm` |
| `SUCCESS` | the move ended at rest inside its tolerances | `fly_to`, `fly_rel` |
| `TIMEOUT` | a move not at rest 30 s after its planned end, or a `capture` not closed 150 s after the call | `fly_to`, `fly_rel`, `capture` |
| `TIME_UP` | the site's simulated time is used up; the run is ending | every call that acts |
| `TOLERANCE_ATT` | the move left its attitude tolerance and was stopped | `fly_to`, `fly_rel` |
| `TOLERANCE_END` | the move ended farther from its goal than the end tolerance | `fly_to`, `fly_rel` |
| `TOLERANCE_PATH` | the move left its path tolerance and was stopped | `fly_to`, `fly_rel` |
| `UNDOCKED` | free of the berth, at the approach point | `undock` |

## 6. Budgets and the end of a run

| budget | for each site | when it is used up |
|---|---|---|
| simulated time | 2400 s | the call that is running and every later call that acts return `TIME_UP`; the queries still answer; the run ends |
| program time: the wall clock between the calls (the cell's own work inside a call is not counted) | 600 s | the program is stopped; the run ends |
| silence: the wall clock from a reply to the next call | 120 s | the program is stopped; the run ends |
| colour frames (`FIC`, `FWC`, `AWC` together) | 400 | `camera` returns `REFUSED_BUDGET` |
| depth frames (`FDC`, `ADC` together) | 300 | `camera` returns `REFUSED_BUDGET` |
| `capture` calls that start | 20 | `capture` returns `REFUSED_BUDGET` |
| stops of a cabin air unit's fan | 2 for each unit | `REJECTED_QUOTA`, and the station counts a violation of its flight rules |
| `lights` requests | 6 | `REJECTED_QUOTA` |
| `report` | the first valid one | `ALREADY_REPORTED` |
| a `step` | 5 s | a longer `dt` is cut |

A call outside the interface is answered `forbidden` and counted.

After `TIME_UP` the cell still answers a limited number of calls in the same way (the queries answer, the calls that
act return `TIME_UP`) and then closes the run. When the program time or the silence ends a run, the run is closed
at once.

**The end of a run.** When the simulated time is used up, when `solve` returns, or when the program crashes or is
stopped, the firmware stops the robot and holds it, and the station runs on for 60 s. The site is judged on the
state at the end of those 60 s. A run that ends early is judged like any other: nothing is deducted for it and
no run is void.

## 7. What no reply contains

- Whether the gripper holds anything. `DONE` of `capture` and `gripper` say that the motion was made.
- Which item is where: no reply names, counts or locates an item, on an inlet, in the hand or adrift.
- Distances to walls, fixtures or items, and contact with them.
- The robot's true pose: `pose()` is an estimate with its own uncertainty.
- Any score.
