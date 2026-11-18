# Intervention vehicle: programming interface

Your program is a Python module with a function `solve(vehicle)`. The `vehicle` object it receives is the only way to
the vehicle: every method is one request to the vehicle cell and one reply. This document lists the 17 calls, their
arguments, their replies, the units and frames they use and the budgets of a run. What the vehicle is and how its
firmware behaves is in `vehicle_manual.md`.

## 1. How a run works

At the start of a run the vehicle is powered up and sits on the deck of its deployment cage with the leak marker in
its jaws. The firmware is in mode `idle`, the lamps are off and the camera tilt is zero.

Simulated time passes only inside `step` and `sonar_scan`. Every other call takes no simulated time: the vehicle and
the water stand still between two steps, however long the program computes. While simulated time passes the firmware
keeps working in the mode of the last motion call.

| Call | Simulated time it takes | Purpose |
|---|---|---|
| `mission()` | none | this site's work order and the budgets of the run |
| `clock()` | none | simulated time and program time |
| `step(dt)` | `dt` | let simulated time pass |
| `finish()` | none | end the run |
| `state()` | none | the vehicle's own state (navigation, sensors, firmware, budgets) |
| `set_nav_fix(x, y)` | none | set the horizontal position of the navigation estimate |
| `idle()` | none | firmware mode `idle` |
| `set_attitude(roll=0.0, pitch=0.0, heading=None, depth=None, thrust=(0.0, 0.0, 0.0), quat=None)` | none | firmware mode `attitude` |
| `set_velocity(u, v, depth, heading)` | none | firmware mode `velocity` |
| `set_hold(x, y, depth, heading)` | none | firmware mode `station` |
| `set_wrench(force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0), free=(True, True, True, True, True, True), frame='body', quat=None, depth=None)` | none | firmware mode `wrench` |
| `camera_frame(size=(640, 480))` | none | one camera frame |
| `set_camera_tilt(deg)` | none | camera tilt setpoint |
| `set_lights(level)` | none | lamp level |
| `sonar_scan(start_deg, stop_deg, range_m, gain=1)` | one ping period per ping | a sector or a full turn of the scanning sonar |
| `gripper_open()` | none | open the jaws |
| `gripper_close()` | none | close the jaws |

Any other method name is refused and counted. The run ends when `solve` returns or the program calls `finish()`, when
a budget of section 7 that ends the run is used up, or when the vehicle is disabled or surfaces (`vehicle_manual.md`,
section 8.2). What the vehicle has done to the world by then stays done.

A call returns a `dict`, or raises `CellError` (section 8). Ordinary printing from the program does not disturb the
connection.

## 2. Units and frames

- Units: metres, seconds, radians, newtons and newton metres. Only the camera tilt and the sonar angles are in
  degrees, and their argument names say so.
- Body axes: x forward, y to port, z up, with the origin of the body frame of the manual.
- World axes: x east, y north, z up. The sea surface is at zero height; depth is positive down.
- Navigation frame: the frame of the navigation estimate. Its axes are the world axes; horizontally it starts at the
  place where the vehicle was powered up, until the program sets it with `set_nav_fix`.
- Attitude: a quaternion (w, x, y, z) that turns body axes into world axes.
- Angles: heading is the yaw about the upward axis, measured from east and counter-clockwise seen from above; it is
  not a compass bearing. Roll is about the body x axis and positive when the port side rises. Pitch is about the body
  y axis and positive when the nose goes down. An attitude given by angles is built from the heading first, then the
  pitch, then the roll.
- Depth: the depth of the body origin, positive down, as the depth sensor measures it.
- Velocity in body axes: u forward, v to port.
- Forces and torques: body axes; torques are about the centre of gravity.
- Arrays in replies are plain lists, except the camera image and the sonar intensities, which are `numpy` arrays.

## 3. Run control

### `vehicle.mission()`

The work order of this site and the budgets of the run. It can be read at any time and does not change during a run.

| Key | Content |
|---|---|
| `chart` | charted position and heading of the structure and the as-laid routes of the lines tied in to it (nominal values of the chart, not the positions as built) |
| `cage` | position and heading of the deployment cage and the height of its deck (exact) |
| `panel_layout` | the as-designed layout of the intervention panel (the tag plates on the panel prevail) |
| `state_table` | the handover status table: the recorded position of every valve (entries can be out of date) |
| `tide_forecast` | the tide forecast for the dive |
| `telemetry` | the summary of the platform's telemetry |
| `debris_report` | the report of the dropped object |
| `notes` | remarks of the work order |
| `budget` | the budgets of this run (section 7) |

The blocks are plain JSON objects; they are the site documents of this dive in machine-readable form, and the field
documents explain how to read them. Positions in the work order are in world coordinates; headings follow section 2.

The fields of the blocks, as the work order of a site that is assessed carries them:

| Block | Field | Type and unit | Content |
|---|---|---|---|
| `chart` | `origin_xy` | [x, y] in m | charted position of the origin of the structure frame (the centre of the manifold, general arrangement), world coordinates; nominal |
| `chart` | `psi` | rad | charted heading of the structure: the direction of its x axis, measured as a heading (section 2); nominal |
| `chart` | `note` | text | a reminder that the chart is nominal |
| `chart` | `lines` | list of objects | the as-laid survey of the lines at the tie-ins: one object per line, with the fields `lines[]` below |
| `chart` | `lines[].name` | text | the name of the line |
| `chart` | `lines[].tie_in` | text | the header and the end of it at which the line is tied in (structure frame of the general arrangement); the first node of the route is there |
| `chart` | `lines[].outer_diameter_m` | m | outer diameter of the pipe |
| `chart` | `lines[].pipe` | text | the kind of pipe |
| `chart` | `lines[].route_xy_m` | list of [x, y] in m | the as-laid route of the line as a polyline from the tie-in outwards, in world coordinates as charted: surveyed together with `origin_xy` and `psi`, so offset and turned against the line as it lies exactly as the charted manifold is against the manifold as built; nominal |
| `chart` | `lines[].node_spacing_m` | m | distance along the line, in plan, from one node of `route_xy_m` to the next; the last node can be closer |
| `chart` | `lines[].mattresses_xy_m` | list of [x, y] in m | approximate positions of the centres of the concrete mattresses on the line, charted like the route |
| `chart` | `lines[].mattress_size_m` | [across, along] in m | size of one mattress across the line and along it |
| `chart` | `lines[].continues` | text | a reminder that the line goes on beyond the last node, which is at the edge of the surveyed area |
| `chart` | `lines_note` | text | a reminder of how the lines were surveyed |
| `cage` | `position_m` | [x, y] in m | the point of the sea bed under the centre of the cage, world coordinates; exact |
| `cage` | `heading_rad` | rad | the direction out of the cage's entrance, as a heading (section 2); exact. The vehicle starts heading this way |
| `cage` | `deck_z_m` | m | world z of the top of the cage deck (negative: below the surface); exact |
| `cage` | `deck_above_seabed_m` | m | height of the deck top above the sea bed |
| `panel_layout` | `rows_bottom_up` | list of lists of tags | the valve tags as designed: the rows from the bottom row up, each row from the viewer's left to right |
| `panel_layout` | `row_heights_m` | list, m | height of each row above the skid's mudmat, bottom row first |
| `panel_layout` | `column_pitch_m` | m | spacing of the columns, which is also the spacing of the ports |
| `panel_layout` | `ports_left_to_right` | list of names | the ports as designed, from the viewer's left |
| `panel_layout` | `note` | text | a reminder that this is the as-designed layout and that the tag plates govern |
| `state_table` | one entry per valve tag | `open` or `closed` | the position recorded at hand-over |
| `tide_forecast` | `slack_time_s` | s | forecast time of slack water, in simulated time of the dive; it can lie before the start or after the end |
| `tide_forecast` | `slack_err_s` | s | the true time of slack is within this much of the forecast |
| `tide_forecast` | `flood_direction_rad` | rad | forecast direction of the main axis of the tidal current (the flood direction), as a heading (section 2) |
| `tide_forecast` | `direction_err_rad` | rad | the true direction is within this much of the forecast |
| `tide_forecast` | `rate_class` | `neap`, `mean` or `spring` | class of the peak speed of the tide (metocean summary) |
| `telemetry` | - | text | one sentence: what the platform's telemetry shows |
| `debris_report` | - | text | one sentence: where an object was seen and which classes of the register it can be |
| `notes` | `facility` | text | what kind of site this is |
| `notes` | `water_depth_m` | m | water depth at the site |
| `budget` | the keys of section 7 | numbers | the budgets of the run |

`chart.lines` holds the lines that are tied in to the structure as they were charted when they were laid: for each
line the header end at which it is tied in, its outer diameter, its route as a polyline with one node every
`node_spacing_m` along the line (the last node, at the edge of the surveyed area, can be closer to the one before),
and the approximate positions of the concrete mattresses that lie on it. The routes are positions of the chart. They
were surveyed together with the charted position and heading of the structure, so a route is offset and turned against
the line on the sea bed exactly as `origin_xy` and `psi` are against the structure as built; the work order does not
give the positions as built. Every line goes on beyond its last node. The general arrangement of the structure shows
the tie-ins (field documents).

`telemetry` and `debris_report` are short texts for a reader. They do not say which branch leaks, which class the
object is or where exactly it stands. The work order of the sister site has the same blocks and fields. The work order
of the training tank differs in three places: its `chart` carries `frame` in place of `note` and has no `lines`, it
has no `tide_forecast`, and its `notes` also hold `instruments` and `run_settings`, the settings of that run (training
documents).

### `vehicle.clock()`

Reply keys: `t`, `sim_time_limit_s`, `program_time_used_s`, `program_time_limit_s`.

`t` is the simulated time in seconds since the start of the run and `sim_time_limit_s` its limit.
`program_time_used_s` is the program time used so far and `program_time_limit_s` its limit (section 7).

### `vehicle.step(dt)`

Lets `dt` seconds of simulated time pass. `dt` must be greater than zero and at most 10 s; it is rounded to whole
control periods of 10 ms, at least one.

Reply keys: `t`, `dt`, `done`, `end_reason`.

`t` is the simulated time after the step and `dt` the time that actually passed, which is less than asked when the run
ends inside the step and zero when it was over already. `done` says whether the run is over and `end_reason` why
(section 7).

### `vehicle.finish()`

Ends the run at once. The launcher also calls it when `solve` returns.

Reply keys: `t`, `done`, `end_reason`.

`t`, `done` and `end_reason` as for `step`.

## 4. State and navigation

### `vehicle.state()`

Everything the vehicle knows about itself. `state()` takes no simulated time and its round trip is far shorter than a
millisecond: it can be called every control period. It contains no information about anything outside the vehicle.

| Key | Type | Unit | Decimals sent | Meaning |
|---|---|---|---|---|
| `t` | float | s | 6 | simulation clock |
| `nav_position_m` | float[3] | m | 4 | navigation estimate of the vehicle position: x (east) and y (north) by dead reckoning on the bottom-track velocity (reset with set_nav_fix), z = minus the depth-sensor depth |
| `nav_velocity_m_s` | float[3] | m/s | 4 | velocity over ground in world axes used by the estimate; held at the last value while the bottom lock is lost |
| `nav_cov_m2` | float[2][2] | m^2 | four significant digits | nominal covariance of the horizontal position estimate (from the estimator's own assumptions, not the true error) |
| `nav_status` | str | - | - | LOCK (bottom lock); HOLD (lock lost, last velocity held); STALE (no valid velocity for longer than the hold time, or none since power-up) |
| `nav_since_lock_s` | float or null | s | 3 | time since the last valid bottom-track velocity (0 while locked, null if there was none) |
| `depth_m` | float or null | m | 4 | depth of the body origin, positive down (depth sensor, lever arm corrected with the attitude); the quantity of every depth setpoint |
| `altitude_m` | float or null | m | 3 | DVL altitude above the bottom (null without three valid beams) |
| `dvl_lock` | bool | - | - | the last DVL velocity was valid (bottom lock) |
| `dvl_beam_range_m` | (float or null)[4] | m | 3 | slant range along each DVL beam (null: no return on that beam) |
| `dvl_beam_valid` | bool[4] | - | - | per-beam validity of the last DVL report |
| `quat_wxyz` | float[4] or null | - | 7 | attitude estimate (AHRS): body (x forward, y left, z up) to world (x east, y north, z up) |
| `heading_rad` | float or null | rad | 7 | yaw of the attitude estimate about +z from +x (east), counter-clockwise; the convention of every heading argument |
| `rates_b_rad_s` | float[3] or null | rad/s | 5 | body angular rates (gyro, bias corrected) |
| `velocity_b_m_s` | float[3] or null | m/s | 4 | velocity over ground in body axes from the navigation estimate (null before the first bottom lock) |
| `battery_V` | float | V | 3 | battery terminal voltage |
| `battery_A` | float | A | 3 | battery current (thrusters + hotel load) |
| `battery_soc` | float | - | 5 | state of charge, 0..1 |
| `mode` | str | - | - | idle; attitude; velocity; station; wrench |
| `degraded` | bool | - | - | station / velocity lost the bottom lock since the last motion call (depth and attitude still held, horizontal thrust held, then ramped to zero) |
| `wrench_groups` | {rotation, depth, translation: float[6]} | N, N m | 3 | body wrench [Fx, Fy, Fz, Mx, My, Mz] delivered by each allocation group after scaling: rotation = attitude loop and released-axis torques, depth = depth loop, translation = horizontal loops, normalised thrust and released-axis forces |
| `group_scale` | {rotation, depth, translation: float} | - | 4 | scale factor applied to each group (1 = delivered in full; a lower group only gets the thrust left by the higher ones) |
| `saturated` | bool | - | - | at least one group was scaled below 1 |
| `current_limit_scale` | float | - | 4 | factor applied to all eight thruster commands by the supply-current limit (1 = not limiting) |
| `thrusters` | float[8] | - | 4 | normalised thruster commands of the autopilot, -1..1 (before the supply-current factor) |
| `camera_tilt_deg` | float or null | deg | 2 | camera tilt servo reading, optical axis up positive |
| `gripper` | str | - | - | open; closed_empty; stalled_on_object (from the jaw travel and stall; no object identity) |
| `protect_stops` | int | - | - | collision protective stops so far |
| `protect_phase` | str or null | - | - | null; stop (thrusters off); hold (depth and attitude hold); surfacing; disabled; motion calls are refused while it is not null |
| `protect_remaining_s` | float | s | 3 | time left in the current stop and hold (0 when none) |
| `budget` | object | - | - | budget usage of this run (filled by the cell) |

The entry `budget` holds the counters of the run:

| Key | Type | Unit | Meaning |
|---|---|---|---|
| `sim_time_used_s` | float | s | simulated time used |
| `sim_time_limit_s` | float | s | simulated time of the run |
| `frame_units_used` | float | units | camera frame units used (640 x 480 = 1.0, 320 x 240 = 0.40) |
| `frame_units_limit` | float | units | camera frame units of the run |
| `sonar_pings_used` | int | - | sonar pings used |
| `sonar_pings_limit` | int | - | sonar pings of the run |
| `protective_stops_used` | int | - | collision protective stops so far |
| `protective_stops_limit` | int | - | the stop that disables the vehicle |

Right after power-up, before the first samples of the sensors have arrived, several entries are null: let a short step
pass before relying on them. Numbers are sent rounded to the decimals of the table, which is finer than the resolution
and the noise of the sensors.

### `vehicle.set_nav_fix(x, y)`

Sets the horizontal position of the navigation estimate to (`x`, `y`) and so defines the navigation frame from then
on. It takes effect at once; a station that is being kept is from then on held at its coordinates in the new frame.
The estimate starts at zero at power-up; to work in the coordinates of the work order, set it to the charted position
of the cage at the start.

Reply keys: `accepted`, `reason`.

`accepted` is true unless the run is over; then it is false and `reason` says so.

## 5. Motion calls

The five motion calls select the firmware mode and its setpoints (`vehicle_manual.md`, section 4). A setpoint stays in
force until the next motion call. A motion call replies with `accepted` and, when accepted, the `mode` now active. A
call that the firmware refuses is not an error and raises nothing: `accepted` is false, `reason` says why, and during
a collision stop `protect_phase` and `remaining_s` say which part of the stop is in progress and how many seconds of
it are left.

| `reason` | Meaning |
|---|---|
| `protective_stop` | a collision stop is in progress (`protect_phase`, `remaining_s` say how long) |
| `episode_over` | the run is over |
| `disabled` | the vehicle is disabled |
| `out_of_bounds` | the vehicle is surfacing after leaving the work area |
| `undervoltage` | the vehicle is surfacing after an undervoltage |

Numbers in the arguments must be finite and at most 1000000 in magnitude. A larger number is refused (`CellError` with
code `bad_params`); a number that is not finite raises `ValueError` in the client library before anything is sent, and
this holds for every call.

### `vehicle.idle()`

Mode `idle`: the thrusters stop.

Reply keys: `accepted`, `mode`, `reason`, `protect_phase`, `remaining_s`.

### `vehicle.set_attitude(roll=0.0, pitch=0.0, heading=None, depth=None, thrust=(0.0, 0.0, 0.0), quat=None)`

Mode `attitude`: the firmware holds an attitude, with or without depth keeping. It does not need the bottom lock.

- `roll`, `pitch`, `heading`: the attitude in radians (section 2). `heading=None` keeps the present heading.
- `quat`: the attitude as a quaternion; when given it replaces the three angles.
- `depth`: the depth to keep, in metres. `depth=None` means no depth keeping.
- `thrust`: three numbers, each -1..1, for the body x, y and z axes: the fraction of the nominal capacity of that axis
  the thrusters are to give, open loop (`vehicle_manual.md`, section 4.3). The z component is ignored while a depth is
  kept.

Reply keys: `accepted`, `mode`, `reason`, `protect_phase`, `remaining_s`.

### `vehicle.set_velocity(u, v, depth, heading)`

Mode `velocity`: the firmware keeps a velocity over ground, a depth and a heading, with the vehicle level. It needs
the DVL bottom lock.

- `u`, `v`: velocity over ground along the body x axis (forward) and the body y axis (to port), in m/s.
- `depth`: in metres. `heading`: in radians.

Reply keys: `accepted`, `mode`, `reason`, `protect_phase`, `remaining_s`.

### `vehicle.set_hold(x, y, depth, heading)`

Mode `station`: the firmware keeps the point (`x`, `y`) of the navigation frame, a `depth` and a `heading`, with the
vehicle level. It needs the DVL bottom lock.

Reply keys: `accepted`, `mode`, `reason`, `protect_phase`, `remaining_s`.

### `vehicle.set_wrench(force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0), free=(True, True, True, True, True, True), frame='body', quat=None, depth=None)`

Mode `wrench`: the firmware applies the force and torque of the call on the axes named in `free` and holds the other
axes. The axes are body axes (`vehicle_manual.md`, section 4.6).

- `free`: six booleans in the order `Fx`, `Fy`, `Fz`, `Mx`, `My`, `Mz`, or a list of these names. The default frees
  all six: then nothing is held, not even the attitude.
- `force`: three numbers in N; `torque`: three numbers in N m. A component on an axis that is not freed is ignored.
- `frame`: `"body"` or `"world"`. With `"world"` the force is given in world axes and the firmware expresses it in
  body axes every control period before it applies `free`. The torque is always in body axes.
- `quat`: the attitude that the rotation axes which are not freed hold. Without it they hold the attitude at the
  moment of the call.
- `depth`: the depth kept when `Fz` is not freed. Without it the depth at the moment of the call is kept.
- `Fx` or `Fy` not freed: the firmware keeps station along that body axis at the position of the call. This needs the
  bottom lock.

Calling `set_wrench` again with the same `free` only updates what the call gives; a `quat` or `depth` that is left out
keeps its previous value. A call with a different `free` starts the mode anew: the integrators are cleared and the
held position, depth and attitude are taken at that moment.

Reply keys: `accepted`, `mode`, `reason`, `protect_phase`, `remaining_s`.

## 6. Sensors and devices

The device calls reply with `accepted`. After the end of the run it is false and `reason` says so.

### `vehicle.camera_frame(size=(640, 480))`

One colour frame of the forward camera at the present simulated time.

| `size` (width, height) [px] | Cost [frame units] | Focal length f [px] | Principal point (cx, cy) [px] |
|---|---|---|---|
| (640, 480) | 1.00 | 381.4 | (320.0, 240.0) |
| (320, 240) | 0.40 | 190.7 | (160.0, 120.0) |

Reply keys: `t`, `frame_index`, `width`, `height`, `image`.

`image` is a `numpy` array of type uint8 with the shape (`height`, `width`, three colour channels), in the order red,
green, blue, with the first row at the top. `t` is the simulated time of the frame and `frame_index` counts the frames
of the run.

The image is a pinhole projection. With the camera's own axes x along the optical axis, y to the left of the image and
z up in the image, a point at (x, y, z) appears at column cx - f y / x and row cy - f z / x, with the focal length f
and the principal point (cx, cy) of the table. Columns and rows are measured in pixels from the top left corner of the
image, so the centre of the pixel in column i and row j, counted from zero, is at (i + one half, j + one half); the
principal point is the exact centre of the image. At a tilt angle a, positive upward, the optical axis is (cos a,
zero, sin a) in body axes, image up is (-sin a, zero, cos a) and image left is the body y axis; the lens turns about
the tilt axis given in the manual.

### `vehicle.set_camera_tilt(deg)`

Sets the tilt setpoint in degrees, positive upward, limited to the range from -90 to 90 deg. The servo moves to it at
about 60 deg per second of simulated time; `state()` reports where it is.

Reply keys: `accepted`, `reason`.

### `vehicle.set_lights(level)`

Sets the common level of the lamps, 0..1; a value outside this range is limited to it.

Reply keys: `accepted`, `level`, `reason`.

`level` is the level now set.

### `vehicle.sonar_scan(start_deg, stop_deg, range_m, gain=1)`

Scans a sector, or a full turn, with the scanning sonar and returns the echoes. The scan takes simulated time: one
ping period for every ping. The vehicle keeps moving under the active firmware mode while the sonar scans, and every
ping is taken from where the vehicle is at that moment.

- `start_deg`, `stop_deg`: the sector in degrees, measured clockwise seen from above, from the body x axis. Both are
  rounded to the head's steps of 0.9 deg. The head steps clockwise from `start_deg` to `stop_deg`, one ping per step,
  both ends included; a sector that passes through zero is allowed. Equal angles mean a full turn of 400 pings.
- `range_m`: the range setting, from 0.75 to 50 m.
- `gain`: zero for low, one for normal, two for high.

A ping takes the travel time of sound to the set range and back, at the sonar's speed-of-sound setting of 1500 m/s,
plus 9.0 ms, plus 0.14 ms for every metre of range. The pings fall on the time grid of the cell, so single periods
differ from this rule by a fraction of a millisecond while a whole scan follows it. The head is at `start_deg` at
once: moving it to the start of a sector takes no time. The table gives examples; the reply carries the time of every
ping.

| `range_m` [m] | Ping period by this rule [ms] | Full turn [s] |
|---|---|---|
| 2 | 11.9 | 4.8 |
| 5 | 16.3 | 6.5 |
| 10 | 23.7 | 9.5 |
| 20 | 38.4 | 15.4 |

Reply keys: `t_start`, `t_end`, `n_pings`, `n_samples`, `range_m`, `gain`, `sample_period_s`, `speed_of_sound_m_s`, `angle_deg`, `t`, `nav_position_m`, `depth_m`, `heading_rad`, `quat_wxyz`, `valid`, `done`, `intensity`.

- `intensity`: a `numpy` array of type uint8 with one row per ping and `n_samples` columns. Column k, counted from
  zero, is the echo from the range (k + one half) x `range_m` / `n_samples`. The values are on a logarithmic scale:
  the full range of the type, from zero to its largest value, spans 60 dB of echo intensity in equal steps. Each gain
  step shifts the whole scale by 6 dB: at a higher gain the same echo reads higher, weaker echoes come onto the scale
  and strong ones reach its top. The echo level that reads zero is not specified.
- `n_pings`, `n_samples`: the shape of `intensity`. `n_pings` is smaller than asked when the run ended during the
  scan; `done` says so.
- `angle_deg`: the angle of each ping, on the step grid, from zero up to a full turn.
- `t`: the simulated time of each ping. `t_start`, `t_end`: the simulated time before the first ping and after the
  last.
- `nav_position_m`, `depth_m`, `heading_rad`, `quat_wxyz`: the navigation estimate, the depth and the attitude
  estimate at each ping, as in `state()`.
- `valid`: whether each ping was taken under water.
- `range_m`, `gain`: the settings used. `sample_period_s`: the time between two samples of a ping.
  `speed_of_sound_m_s`: the setting with which times were turned into ranges; where the true speed of sound differs,
  ranges are scaled by the ratio.

A scan is charged one ping of the budget per ping. A scan whose pings do not all fit into what is left of the budget
is refused as a whole (`CellError` with code `budget`) and charges nothing; pings that were not taken because the run
ended are not charged.

### `vehicle.gripper_open()`

Opens the jaws and releases whatever they hold.

Reply keys: `accepted`, `reason`.

### `vehicle.gripper_close()`

Closes the jaws. Opening and closing take simulated time. While the jaws travel, the entry `gripper` of `state()`
reads `open`; the result of closing appears there only once the jaws have stopped, so allow about 1.75 s after the
call before reading it (`vehicle_manual.md`, section 6). `accepted` is false, without a `reason`, when the jaws
already hold something: nothing moves then and `gripper` keeps its value, which is `stalled_on_object` for as long as
the leak marker is in the jaws.

Reply keys: `accepted`, `reason`.

## 7. Budgets and the end of a run

Every run has the same budgets. The work order repeats them, and `state()` reports what has been used.

| Key in `mission()["budget"]` | Value | Unit | What it limits | When it is used up |
|---|---|---|---|---|
| `sim_time_s` | 2400 | s | simulated time of the run | the run ends (`sim_cap`) |
| `program_time_s` | 1200 | s | program time: wall-clock time of the run less the cell's own time inside `step`, `camera_frame` and `sonar_scan` | the run ends |
| `silence_s` | 180 | s | longest pause between a reply and the program's next request | the run ends |
| `camera_frame_units` | 600 | frame units | camera frames | `camera_frame` is refused (`budget`); the run goes on |
| `frame_weight` | 1.00 for 640 x 480, 0.40 for 320 x 240 | frame units | cost of one frame of each size | - |
| `sonar_pings` | 16000 | pings | sonar pings | a `sonar_scan` whose pings do not all fit is refused (`budget`); the run goes on |
| `protective_stops` | 6 | stops | collision stops; the last one disables the vehicle | the run ends (`disabled`) |

- Camera frames are counted in frame units; the two frame sizes cost differently.
- Sonar pings are counted one by one.
- Program time is the wall-clock time since the start of the run less the time the cell itself spends inside `step`,
  `camera_frame` and `sonar_scan`. It is the time your own code takes, and `clock()` reports it.
- The program runs on one CPU core. It cannot change which core it runs on, and more threads do not make it faster.

| `end_reason` | The run ended because | A program can still read it in a reply |
|---|---|---|
| `finish` | the program called `finish()` (the launcher calls it when `solve` returns) | yes |
| `sim_cap` | the simulated time of the run is used up | yes |
| `disabled` | the last allowed collision stop disabled the vehicle | yes |
| `out_of_bounds` | the vehicle left the work area and surfaced | yes |
| `undervoltage` | the battery voltage stayed below the undervoltage line and the vehicle surfaced | yes |
| `eof` | the program exited or closed its connection | no |
| `silence` | the program sent no request within the silence limit | no |
| `stalled` | the program did not read a reply within the silence limit | no |
| `protocol` | the program sent a line that is not a request | no |
| `program_time_cap` | the program time of the run is used up | no |
| `internal_error` | a fault of the cell | no |
| `physics_fault` | a fault of the cell | no |
| `solve_returned` | `solve` returned without `finish()` | no |
| `closed` | the connection was closed | no |
| `done` | the run had ended | no |

Once the run is over `step` lets no more time pass, motion and device calls are refused with the reason
`episode_over`, `camera_frame` and `sonar_scan` raise `CellError`, and `state()`, `clock()` and `mission()` still
answer.

## 8. Errors

A call that cannot be carried out raises `CellError`; its attribute `code` is one of the codes below and `detail`
explains. A motion call that the firmware refuses is not among them (section 5).

| `CellError.code` | Raised by | Meaning |
|---|---|---|
| `forbidden` | cell | the call is not one of the calls of this document (it is counted) |
| `bad_params` | cell | a parameter is missing, of the wrong kind or out of range; `detail` names it |
| `no_mission` | cell | the run has not started |
| `episode_over` | cell | the run is over (`camera_frame` and `sonar_scan` after the end) |
| `budget` | cell | the frame or ping budget does not cover the request; nothing is charged |
| `sensor_unavailable` | cell | the frame could not be taken; nothing is charged |
| `internal_error` | cell | a fault of the cell; inside a call that changes the run it also ends the run |
| `closed` | client library | the cell closed the connection |
| `protocol` | client library | a reply did not match its request |
| `unreachable` | client library | the practice facilities could not be reached (practice connections only) |

## 9. Practice connections

The same vehicle and the same calls are available in the practice facilities described in the training documents. A
practice connection is opened with `subseacell.client.connect_practice(site)`, which returns a `Vehicle`; it has two
more calls, `reset(site)` to start the run of that connection and, in the training tank only, `instruments()` for the
readings of the tank's instruments. An evaluation run refuses both.

## 10. Example

```python
def solve(vehicle):
    order = vehicle.mission()                      # the work order and the budgets
    vehicle.step(0.5)                              # let the sensors start
    st = vehicle.state()
    vehicle.set_attitude(depth=st["depth_m"])      # level, present heading, keep this depth
    vehicle.set_lights(1.0)
    vehicle.set_camera_tilt(-45.0)
    vehicle.step(2.0)
    frame = vehicle.camera_frame((640, 480))["image"]
    scan = vehicle.sonar_scan(-45.0, 45.0, 10.0)   # a sector ahead, at normal gain
    echoes = scan["intensity"]                     # one row per ping
    vehicle.finish()
```
