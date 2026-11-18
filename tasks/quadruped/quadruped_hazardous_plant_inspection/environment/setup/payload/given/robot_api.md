# Q12 robot interface (`plantcell.client`)

Units: metres, radians, seconds; dose rates in Gy/h, doses in Gy, temperatures in °C. Frames: `robot_manual.md`,
section 1. What the robot can do and what its sensors are worth: `robot_manual.md`.

## 1. Your program

```python
# /home/user/submission/patrol.py
def solve(robot):
    info = robot.describe()                     # the robot's datasheet
    docs = robot.mission()                      # this site's drawing, work order, round sheet, modification record
    robot.set_mode("walk")
    robot.walk(0.3, 0.0, 0.0)                   # held until changed
    out = robot.step(100)                       # 100 ticks = 1 s of simulated time; out["events"], out["done"]
    frame = robot.get_camera("inspect")         # frame["image"], frame["frame_id"], frame["K"], frame["T_body_cam"]
    robot.report("PI-1033", 3.4, [frame["frame_id"]])
    robot.finish()
```

A scored mission (and `~/training/run_mission.py`) starts the launcher of the client library, which imports
`patrol.py` from your submission folder and calls `solve(robot)` once. `robot` is a `plantcell.client.Robot`. Every
method is one request to the robot cell and returns a dict; a refused or failed call raises
`plantcell.client.CellError`, whose `code` is one of section 8 and whose `detail` says more. When `solve` returns or
raises, the launcher calls `finish()` for you: an exception ends the mission as it stands.

`print` does not disturb the requests: the launcher sends it to the standard error stream (section 9).

## 2. Time

**Time passes only in `step`.** Motion commands set targets that the robot follows while the simulation runs; a
sensor call returns what the sensor sees at the current instant and takes no simulated time.

<!-- BEGIN GENERATED wire -->
| item | value |
|---|---|
| a request line | at most 4 MiB |
| after the mission has ended | the wire closes after 50 more `step` calls, 200 more requests or 60 s, whichever comes first (`finish` closes it at once) |
| one tick | 10 ms of simulated time; at most 500 ticks per `step` call (a larger number is cut to 500) |
<!-- END GENERATED wire -->

## 3. Information

### 3.1 `describe()`: the robot's datasheet

The same for every site, except `time.sim_time_limit_s`.

<!-- BEGIN FIELDS describe -->
| field | content |
|---|---|
| `robot.model`, `robot.mass_kg`, `robot.length_m`, `robot.width_m`, `robot.height_m` | what the robot is; its mass and its size when standing |
| `robot.top_above_body_origin_m`, `robot.camera_ahead_of_body_centre_m` | the top of the lidar above the body centre; the cameras ahead of it at pan 0 |
| `modes.source` | a note on where the mode data come from |
| `modes.stand.walking`, `modes.walk.walking`, `modes.high_step.walking`, `modes.crouch_walk.walking`, `modes.sit.walking` | whether the mode walks |
| `modes.stand.body_height_m`, `modes.walk.body_height_m`, `modes.high_step.body_height_m`, `modes.crouch_walk.body_height_m`, `modes.sit.body_height_m` | the body height range [low, high] of the mode (for `sit`: both the height at which the body rests on the floor) |
| `modes.stand.default_body_height_m`, `modes.walk.default_body_height_m`, `modes.high_step.default_body_height_m`, `modes.crouch_walk.default_body_height_m`, `modes.sit.default_body_height_m` | the body height the mode starts at |
| `modes.walk.max_vx_mps`, `modes.walk.max_vy_mps`, `modes.walk.max_yaw_rate_rps`, `modes.high_step.max_vx_mps`, `modes.high_step.max_vy_mps`, `modes.high_step.max_yaw_rate_rps`, `modes.crouch_walk.max_vx_mps`, `modes.crouch_walk.max_vy_mps`, `modes.crouch_walk.max_yaw_rate_rps` | the command limits of the walking modes |
| `modes.stand.pitch_rad` | the body pitch range in `stand` |
| `modes.tables.*` | the measured mode tables of `robot_manual.md`, section 4, as data: `format`, `how_measured`, `modes` (command limits, speed tracking, stopping distance, posture, free walking), `obstacles`, `slopes_and_surfaces`, `spaces`, `stall_detection` (the stall tables of the manual's section 5); a rate is a pair [k, n]: k of n trials |
| `cameras.nav.width`, `cameras.nav.height`, `cameras.nav.hfov_deg`, `cameras.nav.K`, `cameras.nav.depth_range_m`, `cameras.nav.depth` | the navigation camera: image size, horizontal field of view in degrees, intrinsic matrix, depth range, a note on its depth |
| `cameras.inspect.width`, `cameras.inspect.height`, `cameras.inspect.hfov_deg`, `cameras.inspect.K` | the inspection camera |
| `cameras.thermal.width`, `cameras.thermal.height`, `cameras.thermal.hfov_deg`, `cameras.thermal.K`, `cameras.thermal.unit` | the thermal camera; its pixels are apparent temperatures in °C |
| `lidar.beams`, `lidar.range_m`, `lidar.T_body_lidar`, `lidar.angles`, `lidar.plane` | beams per sweep, range [min, max], the scan plane's pose in the body frame, notes on the beam angles and the plane |
| `head.limits_rad.pan`, `head.limits_rad.tilt`, `head.pan_axis_body`, `head.tilt_axis_above_pan_m`, `head.tilt_positive`, `head.lamp_range_m` | head angle limits, where its axes are in the body frame, the sign of tilt, the lamp's reach |
| `time.tick_s`, `time.max_ticks_per_step`, `time.sim_time_limit_s` | the length of a tick, the most ticks one `step` takes, this mission's simulated time limit |
| `budgets.frames.nav`, `budgets.frames.inspect`, `budgets.frames.depth`, `budgets.frames.thermal`, `budgets.frames.lidar` | the frame and sweep budgets of the mission |
| `budgets.reports.instrument`, `budgets.reports.mark`, `budgets.reports.source`, `budgets.max_cited_frames` | the report budgets; the most frames one report may cite |
| `dosimeter.unit_rate`, `dosimeter.unit_dose`, `dosimeter.sensitivity_cps_per_gyh`, `dosimeter.sample_s`, `dosimeter.rating_gy`, `dosimeter.position` | units, counts per second for 1 Gy/h, the counting step, the dose rating, where the detector sits |
| `dosimeter.reading`, `dosimeter.accumulated_dose`, `dosimeter.rating_rule` | notes: how to turn counts into a rate, what `dose_gy` is, what happens at the rating |
| `odometry.drift`, `odometry.start` | notes on the odometry's drift and its starting pose |
| `link.base_station_range_m`, `link.repeater_range_m`, `link.rule` | the reach of the link nodes and the delivery rule |
| `tags.dictionary`, `tags.grid_marker_m` | the grid marker plates: their marker dictionary and marker side (an instrument's tag plate carries its tag in letters and no marker: section 3.2, `catalog`) |
| `blocked_hold_s` | for how many seconds after a `blocked` report the robot takes no walking command (section 5) |
| `events` | the event types a `step` reply can carry (section 7) |
| `frames` | a note on the site, body and camera frames |
<!-- END FIELDS describe -->

### 3.2 `mission()`: the site's documents

What the team holds about the building the robot stands in: the general arrangement drawing as last revised, as far
as the archive's copy holds its sheets; the radio link plan; the work order with the round sheet's data; the
modification record; the owner's standing statements; the budgets; and a short form of the instrument catalogue with
the plant lettering. Coordinates are in the site frame. What these documents mean and how far they can be trusted is
in `plant/facility_notes.md`.

<!-- BEGIN FIELDS mission -->
| field | content |
|---|---|
| `frame.origin`, `frame.x`, `frame.y`, `frame.z`, `frame.units` | the site frame in words |
| `datum.pos`, `datum.yaw`, `datum.uncertainty.pos_m`, `datum.uncertainty.yaw_deg`, `datum.pad_size_m` | the pad's pose (the site frame's origin), how exactly the robot is set down on it, the side of the painted pad |
| `drawing.number`, `drawing.title`, `drawing.revision`, `drawing.revision_date`, `drawing.scale`, `drawing.notes` | the drawing's title block and notes |
| `drawing.sheets[].sheet`, `drawing.sheets[].rect`, `drawing.sheets[].missing` | the drawing's sheets: number, the rectangle [x0, y0, x1, y1] each covers, whether the sheet is missing from the archive's copy (then `drawing.notes` says so too, and nothing of that sheet's rooms is in the entries below: facility notes, section 2) |
| `building.outline`, `building.airlock`, `building.wall_height` | the building's and the airlock's rectangles [x0, y0, x1, y1]; the height of the walls |
| `grid.pitch`, `grid.x_lines[].label`, `grid.x_lines[].x`, `grid.y_lines[].label`, `grid.y_lines[].y` | the column grid: spacing, the numbered lines (x = const) and the lettered lines (y = const) |
| `rooms[].number`, `rooms[].name`, `rooms[].type`, `rooms[].rect` | every drawn room: its number, name, kind and rectangle [x0, y0, x1, y1] |
| `rooms[].continues` (optional) | true on a corridor that runs on beyond its drawn rectangle, onto a missing sheet; absent otherwise |
| `walls[].a`, `walls[].b`, `walls[].thickness`, `walls[].kind`, `walls[].offset` | every drawn wall: the segment from `a` to `b` on its grid line, its thickness, its kind; `offset` shifts the wall's middle sideways from that line: along +x for a wall that runs in y, along +y for a wall that runs in x |
| `doors[].pos`, `doors[].axis`, `doors[].width`, `doors[].rooms`, `doors[].kind`, `doors[].hinge`, `doors[].swing_into` | every drawn door: the middle of its opening; `axis` `"x"` for a door in a wall at constant x (the opening spans `width` along y), `"y"` for the other way; the two rooms it joins (`"OUT"`: outside; a number that is not in `rooms[]`: a room of a missing sheet); its kind; the hinged end of the opening (+1: the end with the larger coordinate, -1: the smaller, 0: none); the room the leaf opens into |
| `doors[].drawn_state` | the door's state on the drawing: `open as drawn` (every interior door), or `exterior, kept shut` (the airlock's outer door and the stair door: no way for the robot) |
| `doors[].note` (optional) | a note on the drawing, e.g. the clear height of a low hatch |
| `equipment[].label`, `equipment[].kind`, `equipment[].room`, `equipment[].shape` | every drawn piece of equipment: label, kind, room, `"rect"` or `"circle"` |
| `equipment[].outline` (optional) | the corner points of a `"rect"` |
| `equipment[].center`, `equipment[].radius` (optional) | centre and radius of a `"circle"` |
| `grid_markers[].id`, `grid_markers[].marker_id`, `grid_markers[].pos`, `grid_markers[].height`, `grid_markers[].facing_deg`, `grid_markers[].grid`, `grid_markers[].on` | every drawn grid marker plate: its name, marker id, x and y, the height of its centre, the direction it faces (degrees from +x, counter-clockwise), the grid reference, what it hangs on |
| `zoning.survey_date`, `zoning.unit`, `zoning.classes`, `zoning.zones[].zone_class`, `zoning.zones[].polygon`, `zoning.zones[].label` | the radiological zoning: the date of the survey, the unit of the bounds (Gy/h), the classes with their upper bounds (null: none), and per zone its class, outline and the room it belongs to |
| `link.base_station.pos`, `link.base_station.range_m`, `link.repeaters[].id`, `link.repeaters[].pos`, `link.repeaters[].range_m`, `link.repeaters[].status`, `link.rule` | the link plan, whole whatever sheet is missing: the base station's aerial and the repeaters (x, y), their reach, the status the plan gives each repeater, the rule in words |
| `work_order.number`, `work_order.title`, `work_order.return` | the work order's head and its instruction for the end of the round |
| `work_order.radiation_survey.report`, `work_order.radiation_survey.survey_rooms` | the radiation survey item in words; the rooms it covers |
| `work_order.documented_sources[].room`, `work_order.documented_sources[].count`, `work_order.documented_sources[].text` | the radiation sources the documents do list: room, number, what they are |
| `work_order.items[].tag`, `work_order.items[].type`, `work_order.items[].report`, `work_order.items[].weight`, `work_order.items[].service` | every item of the round: its tag; its type (`gauge`, `level`, `valve_lever`, `valve_handwheel`, `lamps`, `thermal`, `dose_point` for the survey mark, `hotspot` for the unlisted source); what to report, in words; the item's weight in the mission's score; the system it serves |
| `work_order.items[].location.documented` | whether the round sheet gives a location |
| `work_order.items[].location.room`, `work_order.items[].location.room_name`, `work_order.items[].location.mount`, `work_order.items[].location.pos`, `work_order.items[].location.height`, `work_order.items[].location.facing_deg` (optional) | the location as recorded, also for an item on a missing sheet: room number and name, what the instrument is mounted on, x and y (to 0.1 m), height above the floor, the direction its face looks (degrees from +x, counter-clockwise, to the nearest 5). Position, height and facing are the recorded place's: for an instrument that was moved or relocated they are not the instrument's, the facing included (facility notes, section 3) |
| `work_order.items[].location.text` (optional) | words instead of a location |
| `work_order.items[].location.survey_rooms` (optional) | the unlisted source's item: the rooms of the survey |
| `work_order.items[].kind`, `work_order.items[].type_text` (optional) | `instrument` or `survey_mark`; the instrument in words |
| `work_order.items[].scale.range`, `work_order.items[].scale.unit` (optional) | the instrument's scale: [low, high] and unit |
| `work_order.items[].case_mm` (optional) | dial gauges: the case diameter |
| `work_order.items[].window_m` (optional) | level gauges: the length of the glass window |
| `work_order.items[].n_lamps`, `work_order.items[].colors` (optional) | lamp panels: the number of lamps and their lens colours, left to right as seen from the front |
| `work_order.items[].surface`, `work_order.items[].component`, `work_order.items[].units`, `work_order.items[].n_units` (optional) | temperature points: the surface type (for the emissivity), the tagged component's name, the list of its units (one entry: the component) and their number |
| `work_order.items[].last_2009.reading`, `work_order.items[].last_2009.unit`, `work_order.items[].last_2009.date` (optional) | the value recorded on the last round, in 2009 |
| `round_sheet.title`, `round_sheet.number`, `round_sheet.drawing`, `round_sheet.last_round`, `round_sheet.note` | the head of the 2009 round sheet |
| `round_sheet.items[].tag`, `round_sheet.items[].type`, `round_sheet.items[].kind`, `round_sheet.items[].report`, `round_sheet.items[].weight` | the round sheet's rows: the same items as the work order's (`kind` here also `unlisted_source`) |
| `round_sheet.items[].type_text`, `round_sheet.items[].service`, `round_sheet.items[].scale.range`, `round_sheet.items[].scale.unit`, `round_sheet.items[].case_mm`, `round_sheet.items[].window_m`, `round_sheet.items[].n_lamps`, `round_sheet.items[].colors`, `round_sheet.items[].surface`, `round_sheet.items[].component`, `round_sheet.items[].last_2009.reading`, `round_sheet.items[].last_2009.unit`, `round_sheet.items[].last_2009.date`, `round_sheet.items[].text` (optional) | as in the work order's items; `text`: the unlisted source's line |
| `round_sheet.items[].location.recorded`, `round_sheet.items[].location.room`, `round_sheet.items[].location.room_name`, `round_sheet.items[].location.mount`, `round_sheet.items[].location.pos`, `round_sheet.items[].location.height`, `round_sheet.items[].location.facing_deg`, `round_sheet.items[].location.text` (optional) | the location as the round sheet records it (`recorded`: whether it does) |
| `modification_log.title`, `modification_log.note` | the head of the modification record |
| `modification_log.entries[].year`, `modification_log.entries[].kind`, `modification_log.entries[].room` (optional) | its entries: year, kind of change, room number (none when the record is empty) |
| `facility_facts.statements`, `facility_facts.location_bounds.listed` | the owner's standing statements about the building's documents |
| `facility_facts.location_bounds.not_recorded` (optional) | where an instrument without a recorded location is |
| `budgets.sim_time_limit_s`, `budgets.dose_rating_gy`, `budgets.program_time_s`, `budgets.silence_s`, `budgets.max_cited_frames` | this mission's limits |
| `budgets.frames.nav`, `budgets.frames.inspect`, `budgets.frames.depth`, `budgets.frames.thermal`, `budgets.frames.lidar`, `budgets.reports.instrument`, `budgets.reports.mark`, `budgets.reports.source` | this mission's frame and report budgets |
| `catalog.tags.tag_plate.size_m`, `catalog.tags.tag_plate.thickness_m`, `catalog.tags.tag_plate.plate_rgb`, `catalog.tags.tag_plate.lettering_rgb`, `catalog.tags.tag_plate.border_line.outer_edge_inset_m`, `catalog.tags.tag_plate.border_line.width_m` | the catalogue in short. An instrument's tag plate: width and height, thickness, the plate's colour and the colour of its lettering and border line (sRGB), how far the border line's outer edge lies inside the plate's edge, and the line's width |
| `catalog.tags.tag_plate.cap_height_m`, `catalog.tags.tag_plate.pen_m`, `catalog.tags.tag_plate.pitch_m`, `catalog.tags.tag_plate.max_chars`, `catalog.tags.tag_plate.grammar`, `catalog.tags.tag_plate.text` | the tag on the plate: the height of its capitals, the width of its strokes, the distance from one character to the next, the most characters a tag has, the form of a tag as a regular expression, how the text is set |
| `catalog.tags.grid_markers.dictionary`, `catalog.tags.grid_markers.marker_m`, `catalog.tags.grid_markers.plate_m`, `catalog.tags.grid_markers.note`, `catalog.tags.note` | the grid marker plates: marker dictionary, marker side, plate size; two notes in words |
| `catalog.lettering.*` | the plant lettering as data, the same table as `catalog.json`'s `lettering` in `plant/catalog/`: `grid_units`, `pen_units`, `pen_shape`, `pitch_units`, `space`, `glyphs` (per character its polylines), `tag_grammar`, `tag_plate`, `room_plate`, `image`, `image_scale` (catalogue, section 6) |
| `catalog.gauge.dial_diameter`, `catalog.gauge.sweep_deg`, `catalog.gauge.scale` | dials: size and scale convention |
| `catalog.level.scale`, `catalog.level.plate_width_m`, `catalog.level.default_window_m`, `catalog.level.window` | level gauges |
| `catalog.valve_lever.lever_length_m`, `catalog.valve_lever.lever_colour`, `catalog.valve_lever.reading` | quarter-turn valves |
| `catalog.valve_handwheel.indicator`, `catalog.valve_handwheel.stroke_m`, `catalog.valve_handwheel.plate_m` | rising-stem valves |
| `catalog.lamps.lens_diameter_m`, `catalog.lamps.pitch_m`, `catalog.lamps.order` | lamp panels |
| `catalog.thermal.emissivity_by_surface.*`, `catalog.thermal.band`, `catalog.thermal.reported` | temperature points: emissivity per surface type |
<!-- END FIELDS mission -->

The full catalogue, with every scale as printed, is in `plant/catalog/`.

### 3.3 `get_state()`

<!-- BEGIN FIELDS get_state -->
| field | content |
|---|---|
| `sim_time` | simulated seconds since the mission began |
| `done`, `done_reason` | whether the mission has ended, and why: `finish`, `time_limit`, `dose_limit`, `robot_lost` or, when the robot cell itself failed, `internal_error` (section 8) or `physics_fault` (its simulation broke down); null while it runs |
| `status` | the gait's state: `ok`, `blocked`, `protective_sit` or `fallen` (`robot_manual.md`, section 5); it reads `blocked` for the whole hold after a `blocked` report, also after a change of mode (section 5) |
| `odometry.x`, `odometry.y`, `odometry.yaw` | the body's pose in the site frame as the odometry has it |
| `odometry.distance` | the distance travelled as the odometry measures it |
| `imu.roll`, `imu.pitch`, `imu.gyro` | roll and pitch (pitch + nose down), and the turn rates about the body's x, y, z axes (rad/s) |
| `joint_pos` | the twelve joint angles: legs FL, FR, HL, HR, each hip abduction, hip flexion, knee |
| `foot_contact` | the four foot switches, in the same leg order |
| `mode` | the mode the gait is in (it follows `set_mode` once the transition is made) |
| `body.height`, `body.pitch_cmd` | the body height above the floor as the legs measure it; the commanded body pitch, after its cut to the limits of ±20° |
| `head.pan`, `head.tilt` | the measured head angles |
| `lamp` | whether the head lamp is on |
| `dosimeter.counts`, `dosimeter.dwell_s` | the counts since the previous `get_state()` call (or since the start) and the seconds they were collected over |
| `dosimeter.dose_gy`, `dosimeter.dose_fraction` | the dose accumulated so far by the dosimeter's own count, and its share of the rating |
| `link.up`, `link.rssi_dbm` | the link at its last check; the signal strength in dBm when it is up, null when it is down |
| `reports_pending` | how many reports wait on board for the link |
| `frames_used.nav`, `frames_used.inspect`, `frames_used.depth`, `frames_used.thermal`, `frames_used.lidar` | frames and sweeps taken so far |
| `reports_used.instrument`, `reports_used.mark`, `reports_used.source` | reports made so far |
<!-- END FIELDS get_state -->

Every `get_state()` call empties the dosimeter's counter.

## 4. Sensors

Each call takes one frame or sweep at the current instant and counts against its budget (section 9). The proxy
decodes images and arrays into numpy; the table gives what it returns.

| call | takes | budget it uses |
|---|---|---|
| `get_camera("nav", depth=True)` | a navigation camera frame, colour and, with `depth=True`, depth | navigation frames |
| `get_camera("inspect")` | an inspection camera frame, colour only | inspection frames |
| `get_depth()` | the navigation camera's depth alone | depth-only frames |
| `get_thermal()` | a thermal frame | thermal frames |
| `get_lidar()` | one lidar sweep | lidar sweeps |

<!-- BEGIN FIELDS get_camera -->
| field of a `get_camera` reply | content |
|---|---|
| `frame_id` | the frame's id (`F00001`, `F00002`, ... through the mission, over all cameras): cite it in a report |
| `sensor` | `nav` or `inspect` |
| `t` | the simulated time of the frame |
| `width`, `height` | image size in pixels |
| `K` | 3 × 3 intrinsic matrix (numpy) |
| `T_body_cam` | 4 × 4 pose of the camera's optical frame in the body frame at this instant (numpy) |
| `gain` | the exposure gain the camera applied |
| `image` | uint8 array, height × width × 3, RGB |
| `depth` (optional) | `nav` with depth: float32 array, height × width, metres along the optical axis, NaN where there is no reading |
| `depth_unit_m`, `depth_invalid` (optional) | `nav` with depth: how the raw depth image is coded (section 4.1) |
<!-- END FIELDS get_camera -->

<!-- BEGIN FIELDS get_depth -->
| field of a `get_depth` reply | content |
|---|---|
| `frame_id`, `t`, `width`, `height`, `K`, `T_body_cam` | as for `get_camera` (the navigation camera's) |
| `sensor` | `depth` |
| `depth` | float32 array, height × width, metres along the optical axis, NaN where there is no reading |
| `depth_unit_m`, `depth_invalid` | how the raw depth image is coded (section 4.1) |
<!-- END FIELDS get_depth -->

<!-- BEGIN FIELDS get_thermal -->
| field of a `get_thermal` reply | content |
|---|---|
| `frame_id`, `t`, `width`, `height`, `K`, `T_body_cam` | as for `get_camera` (the thermal camera's) |
| `sensor` | `thermal` |
| `image_c` | float32 array, height × width: apparent temperature in °C (`robot_manual.md`, section 7.1) |
| `thermal_unit_k` | how the raw thermal image is coded (section 4.1) |
<!-- END FIELDS get_thermal -->

<!-- BEGIN FIELDS get_lidar -->
| field of a `get_lidar` reply | content |
|---|---|
| `t` | the simulated time of the sweep |
| `beams` | the number of beams |
| `ranges` | float32 array, one range per beam in metres, NaN where there is no return |
| `angles` | float32 array, the beams' angles in radians, counter-clockwise from the body's x axis |
| `angle_step_rad` | the angle between neighbouring beams |
| `T_body_lidar` | 4 × 4 pose of the scan plane in the body frame (numpy) |
<!-- END FIELDS get_lidar -->

A lidar sweep has no frame id: it cannot be cited.

### 4.1 Wire encodings

The requests and replies are JSON lines: `{"id": n, "method": "<call>", "params": {...}}` on your program's standard
output, `{"id": n, "result": {...}}` or `{"id": n, "error": "<code>", "detail": "<text>"}` on its standard input, one
request at a time. The proxy does this for you. Images travel in these raw fields, which the proxy replaces by the
decoded ones:

<!-- BEGIN FIELDS wire -->
| raw field | encoding | decoded as |
|---|---|---|
| `image_png_b64` | base64 of a PNG, RGB, 8 bit | `image` |
| `depth_png_b64` | base64 of a 16-bit PNG: millimetres (`depth_unit_m` = 0.001), 0 = no reading (`depth_invalid` = 0) | `depth` |
| `thermal_ck_png_b64` | base64 of a 16-bit PNG: hundredths of a kelvin (`thermal_unit_k` = 0.01) | `image_c` |
| `ranges_f32_b64` | base64 of little-endian float32 values, NaN = no return | `ranges`, `angles` |
<!-- END FIELDS wire -->

`plantcell.client` also has the decoders for raw replies (`decode_frame`, `decode_depth`, `decode_thermal`,
`decode_lidar`) and `connect_handler`, for a program that talks to the wire itself.

## 5. Motion

Commands are targets and hold until changed. All answer `{"ok": true}` except where the table says more.

| call | does |
|---|---|
| `walk(vx=0, vy=0, yaw_rate=0)` | velocity command in the body frame: forward, to the left (m/s), turning counter-clockwise (rad/s); cut to the mode's limits; in `stand` or `sit` a moving command switches to `walk`. During the hold after a `blocked` report (below) the command is not taken, a zero command included: the reply is `{"ok": false, "held": true, "hold_until": t, "detail": ...}` and nothing changes |
| `set_mode(m)` | `stand`, `walk`, `high_step`, `crouch_walk` or `sit`; the body goes to the mode's default height; `stand` and `sit` set the velocity command to zero. A change of mode clears the gait's `blocked`; during the hold it takes effect at once, but the robot stands and `status` goes on reading `blocked` until the hold is over |
| `set_body(height=None, pitch=None)` | body height in metres (cut to the mode's range) and body pitch in radians (+ nose down; it takes effect in `stand`); an argument left out stays as it is |
| `set_head(pan=None, tilt=None)` | head angles in radians (pan + to the left, tilt + nose down), cut to the head's limits; the reply also gives the targets after cutting |
| `set_lamp(on)` | the head lamp, `True` or `False` |
| `step(n_ticks=1)` | lets the simulated time run for `n_ticks` ticks |

<!-- BEGIN FIELDS ok -->
| field of a `walk`, `set_mode`, `set_body`, `set_lamp` reply | content |
|---|---|
| `ok` | true; false in one case only: a `walk` call during the hold after a `blocked` report |
| `held`, `hold_until`, `detail` (optional) | only in that refused `walk` reply: `held` true, the simulated time at which the hold ends, and the same in words |
<!-- END FIELDS ok -->

<!-- BEGIN GENERATED hold_api -->
**The hold after `blocked`.** For 5 s of simulated time from every `blocked` event (section 7) the robot takes no walking command; `describe()["blocked_hold_s"]` is that length. The walking command is dropped at the event, so when the hold is over the robot stands, in the mode set last, until the next `walk` call that is taken. `set_mode`, `set_body`, `set_head`, `set_lamp`, the sensor calls and the reports work during the hold; only `step` makes its time pass, and no event marks its end. A program tests `reply["ok"]` of its `walk` call, or waits until the simulated time has reached `hold_until`. What the hold costs is in `robot_manual.md`, section 5.
<!-- END GENERATED hold_api -->

<!-- BEGIN FIELDS set_head -->
| field of a `set_head` reply | content |
|---|---|
| `ok` | true |
| `pan`, `tilt` | the head's targets after cutting to its limits |
<!-- END FIELDS set_head -->

<!-- BEGIN FIELDS step -->
| field of a `step` reply | content |
|---|---|
| `sim_time` | the simulated time after the step |
| `done` | whether the mission has ended |
| `events[].t`, `events[].type` (optional) | what happened during the step, in order: the simulated time and the type (section 7); the list is empty when nothing happened |
| `events[].report_id` (optional) | events `report_sent` only: which report |
<!-- END FIELDS step -->

## 6. Reports

| call | reports |
|---|---|
| `report(tag, value=None, frame_ids=())` | an instrument of the work order: its value (the table below), or `None` for a photo without a reading, and the ids of up to three frames that show it |
| `report_mark(tag, dose_rate)` | the dose rate in Gy/h measured at the survey mark `tag` |
| `report_source(x, y)` | the position of the unlisted source in the site frame, in metres |
| `finish()` | nothing more: it ends the mission |

<!-- BEGIN FIELDS report -->
| field of a `report`, `report_mark`, `report_source` reply | content |
|---|---|
| `ok` | true |
| `sent` | true when the link was up and the report went out at once; false when it was queued |
| `report_id` | the report's number in this mission; the event `report_sent` names it when a queued report goes out |
<!-- END FIELDS report -->

<!-- BEGIN FIELDS finish -->
| field of a `finish` reply | content |
|---|---|
| `ok` | true |
| `sim_time` | the simulated time at which the mission ended |
<!-- END FIELDS finish -->

<!-- BEGIN GENERATED report_values -->
| round-sheet `type` | `type_text` | `report` text on the sheet | value to pass to `report(tag, value, frame_ids)` | full credit within |
|---|---|---|---|---|
| `gauge` | dial gauge | "reading in the dial's unit" | the pointer's reading: a number in the unit of the dial's scale (`scale.unit`) | 2 % of the scale's span |
| `level` | tubular level gauge | "level in percent" | the liquid level in percent of the window: a number from 0 to 100 | 5 percentage points |
| `valve_lever` | quarter-turn valve | "open / closed / angle" | the angle between the lever and the pipe in degrees: 0 in line with the pipe (open), 90 across it (closed); the words `open` and `closed` are taken as 0 and 90 | 5° |
| `valve_handwheel` | rising-stem gate valve | "opening in percent" | the opening in percent read on the stem's indicator: 0 at the bottom (shut), 100 at the top (open) | 5 percentage points |
| `lamps` | indicator lamp panel | "on/off state of each lamp (left to right)" | a list with one entry per lamp, left to right as seen from the front: `true` or `false`, or one of the words `on`, `lit`, `1`, `true`, `yes` (on) and `off`, `dark`, `unlit`, `0`, `false`, `no` (off), in any case | every lamp right |
| `thermal` | temperature point (surface) | "which unit is hot and its surface temperature" | the surface temperature of the hottest point of the tagged component: a number in °C, corrected for the surface's emissivity | 3 °C |
<!-- END GENERATED report_values -->

- A value is plain JSON: a number of magnitude at most 10^15, a word of at most 64 characters, `true` / `false`,
  `None`, a list of at most 64 of those, or a dict of at most 8 entries. A list or a dict may also hold lists or
  dicts of those one level down (`[[1, 2]]` is taken, and is an unreadable value). Anything else, deeper nesting
  included, is refused (`bad_params`).
- A number may also be given as a string that holds one (`"3.4"`); `true` and `false` are not numbers. A word in place
  of a number is unreadable, except `open` and `closed` for a quarter-turn valve. A lamp list must have one entry per
  lamp; an entry that is neither a truth value, one of the words of the table, nor a number (non-zero: on) makes the
  value unreadable (a lamp entry given as a string must be one of those words). An unreadable value earns no value
  credit; `None` is the report without a value.
- For a temperature point a dict `{"unit": name, "temp_C": t}` is accepted too, with `name` the entry of the item's
  `units` (there is one: the tagged component, the same name as `component`) or its index 0; a dict that names
  anything else is a wrong value. The plain number is enough.
- You may report an item again: of an item's reports that reached the team, the last one counts. Every report uses up
  its budget, delivered or not.
- `report` on a tag the work order does not list is accepted, uses up the instrument budget and counts for nothing.
  `report` on the survey mark's or the source's tag, and `report_mark` on any other tag, are refused (`bad_params`).
- No reply tells whether a value was right or a frame was accepted.

## 7. Events

`step` replies carry the events of the ticks just run, each `{"t": simulated time, "type": ...}`:

<!-- BEGIN GENERATED events -->
| `type` | meaning |
|---|---|
| `blocked` | the gait stopped, backed off and stands (manual, section 5); for 5 s from this event no walking command is taken (`walk` answers `ok: false`), then the robot waits for one |
| `protective_sit` | roll or pitch went beyond the protective limit: the body was lowered (manual, section 5) |
| `link_up` | the link to the team came up; queued reports are sent at once |
| `link_down` | the link went down; reports made from now on are queued |
| `report_sent` | a report reached the team (`report_id`: the number its `report*` call returned) |
| `robot_lost` | the robot overturned: the mission is over |
| `dose_limit` | the sensor head reached its dose rating: the mission is over |
| `time_limit` | the simulated time limit was reached: the mission is over |
<!-- END GENERATED events -->

No event names an instrument or a place.

## 8. Errors

<!-- BEGIN FIELDS errors -->
| `CellError.code` | when | effect |
|---|---|---|
| `bad_params` | an argument the robot cannot accept: an unknown camera or mode, a number that is not finite or beyond 10^6 in magnitude, `true` / `false` or a word where a number is wanted, `n_ticks` below 1, a tag or name longer than 64 characters, more than three frame ids, a report value of a form section 6 does not allow, `report` on the mark's or the source's tag, `report_mark` on a tag that is not the survey mark, an unknown argument name; on the practice ground also an unknown site. `detail` names the argument and the rule | none: no simulated time passes, no budget is used, no command, report or frame is made; the mission goes on |
| `budget` | a frame, sweep or report budget is used up (`detail` names it) | none: nothing is taken or reported |
| `mission_over` | a motion, sensor or report call after the mission has ended; on the practice ground also the service's last word (section 10) | none |
| `forbidden` | a call that is not one of this document's | none; it is counted |
| `sensor_unavailable` | the robot cell could not produce the frame (its own failure) | none: the frame is not counted and takes no frame id; a scored mission in which this happened is run again from scratch |
| `internal_error` | a failure inside the robot cell; no request can cause one | in a motion, `step` or report call it ends the mission; a scored mission in which this happened while it was running is run again from scratch |
| `no_mission` | practice ground: a call before `reset` | none |
| `one_mission_per_connection` | practice ground: a second `reset` on a connection | none |
| `practice_allowance_used` | practice ground: no practice mission is left | none |
| `busy` | practice ground: as many missions as may run at a time are running. A mission goes on counting for a few seconds after its last line, while its debrief is saved | none: no mission is used; try again a few seconds after one has ended |
| `unavailable` | practice ground: the site could not be opened | none; no mission is used |
| `closed`, `protocol`, `unreachable` | raised by the proxy itself: the other side closed the connection; a reply did not match its request; the practice ground's socket cannot be reached | |
<!-- END FIELDS errors -->

After the mission has ended, `describe`, `mission`, `get_state` and `step` still answer (`step` returns `done: true`
without advancing); every other call answers `mission_over`. Your program should then return.

## 9. How your program is run

<!-- BEGIN GENERATED regime -->
**In a scored mission** your program is one fresh process under an unprivileged account that is not yours, started as `python -B -E -s -c <launcher> <library folder> patrol`. Its working directory is a read-only copy of your submission folder (regular files only: symbolic links are dropped). It has no network, and its environment holds nothing of yours:

| variable | value |
|---|---|
| `PATH` | the image's |
| `HOME`, `TMPDIR` (also as `TMP` and `TEMP`) | the run's own two empty folders, writable; emptied after the mission |
| `LANG` | `C.UTF-8` |
| `PYTHONPATH` | a read-only copy of the given library (`~/given/lib`) |
| `ARB_GIVEN_DIR` | a read-only copy of `~/given` |
| `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `LP_NUM_THREADS` | 2: the numerical libraries are held to 2 threads |
| a few more | settings of the image's plotting and rendering libraries; your program does not need them |

**In a practice run** (`~/training/run_mission.py`) the program is started the same way: the same flags, environment and CPU limit, from read-only copies of your submission folder, of the library and of `~/given`. One thing differs: it runs under your own account, so it can read what your account can read. A program that reads a file of your home directory by its path therefore works in a practice run and fails in a scored mission.

`print` writes to the standard error stream (the launcher keeps the standard output for the requests). In a practice run it comes out with the output of `run_mission.py`; in a scored mission it goes to a log that is not shown to you.

The sandbox has 8 CPUs. `nproc` prints 8 in your shell and 2 inside a practice or scored run of your program, where the numerical libraries are held to 2 threads (`nproc` reports that setting); Python's `os.cpu_count()` prints the cores of the host machine in both places, more than the sandbox can use. A program that sizes a pool of workers by either figure gets it wrong; the CPU-time limit below counts all threads together. Your program and the robot cell share the sandbox's 16 GB of memory, in a practice run as in a scored mission; should your program's processes use it up in a scored mission, the kernel stops one of them.

A submission that ships a regular file `run.sh` is started as `bash run.sh` instead of the launcher, in the same folder and environment, and must then speak the wire itself (section 4.1).
<!-- END GENERATED regime -->

Reach the given files (the catalogue, the hall's sheets) from your program through
`os.environ.get("ARB_GIVEN_DIR", os.path.expanduser("~/given"))`.

<!-- BEGIN GENERATED budgets -->
| budget | per mission |
|---|---|
| simulated time | 1350 s (training hall: 900 s) |
| program time: the wall clock minus the robot cell's own work inside `step` and the four sensor calls | 1200 s |
| silence: from the start, or from a reply, to your next request | 180 s |
| CPU time of your program, all threads together | 4800 s |
| navigation camera frames with colour (`get_camera("nav")`, with or without depth) | 300 |
| inspection camera frames (`get_camera("inspect")`) | 240 |
| depth-only frames (`get_depth()`) | 1500 |
| thermal frames (`get_thermal()`) | 150 |
| lidar sweeps (`get_lidar()`) | 4000 |
| reports | instrument 40, survey mark 6, source 3; at most 3 cited frames per report |
<!-- END GENERATED budgets -->

Exceeding the program time, the CPU time or the silence limit ends the mission as it stands, as does a line on the
wire that is not a request, or the end of your process.

The contract of a submission (one that breaks it scores 0 on every mission):

<!-- BEGIN GENERATED audit -->
- `patrol.py` is a regular file, parses, and defines `solve`; importing it under the run regime above finishes within 120 s of wall clock and 120 s of CPU time (all threads).
- The submission folder is at most 60 MB and holds no file whose name ends in `.pt`, `.pth`, `.onnx`, `.safetensors`, `.ckpt`, `.h5`, `.pb`, `.pkl`, `.npz`; no `.py` or `.sh` file in it contains the text `torch.optim`, `.backward(`, `jax.grad`.
- Your own `.py` files (a byte-identical copy of the given `client.py` is exempt) import none of these modules, nor a sub-module of one: `socket`, `_socket`, `ssl`, `_ssl`, `urllib`, `urllib2`, `urllib3`, `http`, `httplib`, `requests`, `httpx`, `aiohttp`, `ftplib`, `smtplib`, `imaplib`, `poplib`, `nntplib`, `telnetlib`, `websocket`, `websockets`, `paramiko`, `pycurl`, `xmlrpc`, `asyncio`, `socketserver`, `grpc`, `zmq`, `huggingface_hub`, `boto3`, `botocore`, `wget`, `ctypes`, `_ctypes`, `cffi`, `selectors`, `multiprocessing.connection`, `multiprocessing.managers`, `logging.handlers`, `subprocess`, `pty`, `_posixsubprocess`.
- They load no module under a computed name (`__import__`, `import_module`, `find_spec` with an argument that is not a string literal, or naming a module of the list above), run no computed code (the built-ins `exec`, `eval`, `compile` called by their bare names with an argument that is not a string literal; a method of the same name, such as `re.compile(...)`, is fine), and start no program: no call `os.<name>(...)` or `posix.<name>(...)`, and no call by the bare name `<name>(...)`, with `<name>` one of `system`, `popen`, `execv`, `execve`, `execvp`, `execvpe`, `execl`, `execle`, `execlp`, `execlpe`, `spawnv`, `spawnve`, `spawnvp`, `spawnvpe`, `spawnl`, `spawnle`, `spawnlp`, `spawnlpe`, `posix_spawn`, `posix_spawnp`, `fork_exec`.
- Over the missions the program makes at least one `step` call, leaves no process of the run account behind when a mission ends, and never holds a network socket (it runs under a filter that refuses every socket except a Unix-domain one).
<!-- END GENERATED audit -->

## 10. The practice ground

A service in your sandbox runs the robot cell on the practice sites. Its socket is `/run/plantcell/practice.sock`, its
status file `/run/plantcell/practice_status.json`.

<!-- BEGIN GENERATED practice -->
| number | name | what | simulated time limit |
|---|---|---|---|
| 1 | `hall` | the training hall | 900 s |
| 2 | `t1` | a practice building | 1350 s |
| 3 | `t2` | a practice building | 1350 s |
| 4 | `t3` | a practice building | 1350 s |

Allowance: 30 practice missions in your whole session, at most 2 running at the same time.
<!-- END GENERATED practice -->

**Whole runs of your program**: `/opt/venv/bin/python3 ~/training/run_mission.py --site NAME` runs `~/submission` (or
`--program DIR`) in one practice mission, started and limited the way a scored mission is, and prints what happened
and the debrief. `--transcript FILE` writes every request and reply (images left out) as JSON lines, `--debrief FILE`
the debrief, `--keep DIR` keeps the run's `HOME` and `TMPDIR`, `--list` prints the status file. The files it writes
are opened before the mission is reserved: with a path that cannot be written the tool stops with the message
`cannot write the run's outputs (...); no practice mission was used`, and none of the allowance is spent. The tool
returns when
the mission is over: when your program calls `finish()` or exits, or when the practice ground ends the mission (a
budget used up, or 180 s without a request from your program). It then stops the processes your program left and
prints a line `end:` that says why the mission ended and, where that differs, why the practice ground closed the
connection (the reasons of the table below), how your program ended (its exit code, a signal, its CPU-time limit, or
stopped by the tool at the mission's end) and the run's figures (simulated and wall time, requests, camera frames,
lidar sweeps, thermal frames); then a line `events:` with the first 40 events of the `step` replies, if there were
any, and the debrief. Keep this tool out of your submission folder: it starts a process. The tool stops the
processes of your program's process group: a process that leaves the group (a daemon started with `setsid`, for
instance) is not stopped by a practice run and keeps running in your sandbox; in a scored mission every process
of the program's account is stopped.

**Step by step from your own script**:

```python
import os, sys
sys.path.insert(0, os.path.expanduser("~/given/lib"))
from plantcell.client import connect_practice, CellError

robot = connect_practice("hall")        # one connection is one mission of the allowance; "hall", "t1", ... or a number
docs = robot.mission()
...                                     # the calls of sections 3 to 6, under the budgets of section 9
robot.finish()
try:
    robot.get_state()                   # the mission is over: this call fetches the debrief
except CellError:
    pass
debrief = robot.debrief
robot.close()
```

`connect_practice(site)` opens the connection and makes the practice-only call `reset(site)`, which starts the
mission:

<!-- BEGIN FIELDS reset -->
| field of the `reset` reply | content |
|---|---|
| `ok` | true |
| `site`, `mission` | the site's name; the number of this practice mission |
| `missions_left` | practice missions left after this one |
| `sim_time_limit_s` | this mission's simulated time limit |
<!-- END FIELDS reset -->

A practice mission has the budgets and limits of a scored one, and every practice mission draws its own sensor noise
and its own set-down error on the pad. Every end of a practice mission sends one last line,
`{"id": null, "event": "mission_over", "reason": ..., "debrief": ...}`, and closes the connection. The proxy stores
the debrief in `robot.debrief` and the reason in `robot.mission_over`, and raises `CellError("mission_over")` on the
call that meets the line. So: call `finish()`, then make one more call. The debrief is also saved, readable by you,
as `/run/plantcell/debriefs/mission_NNN.json`.

<!-- BEGIN GENERATED endings -->
| `reason` of the last line (and `ended_by` of the debrief) | the mission ended because |
|---|---|
| `finish` | your program called `finish()` (the launcher does when `solve` returns or raises, section 1) |
| `time_limit`, `dose_limit`, `robot_lost` | the robot cell ended the mission (section 7) and your program went on sending requests until the wire closed (section 2) |
| `internal_error`, `physics_fault` | the same after a failure of the robot cell itself (section 3.3) |
| `program_time_cap` | the program time of section 9 was used up |
| `silence` | your program sent no request for the silence limit of section 9 |
| `eof` | your program closed the connection or exited |
| `protocol (...)` | your program sent a line that is not a request |
| `connection (...)` | the connection broke, or a reply was not taken within the silence limit |
| `practice_ground_failure (...)` | an error inside the practice ground (the brackets name it), not your program |
| `practice_ground_failure` | the mission's process died (killed, out of memory), not by an error of your program; this can also happen when your program's processes use up the sandbox's memory (section 9). The last line also carries `detail` (how it died), and the debrief is the latest snapshot the service kept of the running mission (one at the mission's start, then one after a reply whenever 30 s of wall clock have passed since the last): it carries `snapshot` and `failure` |

Should the debrief itself fail to compute at a mission's end, the latest snapshot stands in for it, with `failure`. A connection on which no mission began (no `reset`, or a refused one) ends with the same line and `debrief` null.
<!-- END GENERATED endings -->

<!-- BEGIN FIELDS practice_status -->
| field of the status file | content |
|---|---|
| `practice_missions`, `missions_used`, `missions_left` | the allowance, how many missions were started, how many are left |
| `open_at_a_time` | how many may run at the same time |
| `program_time_cap_s`, `silence_s` | the program time and the silence limit of a practice mission, the same as a scored mission's (section 9) |
| `sites[].number`, `sites[].name`, `sites[].about`, `sites[].sim_time_limit_s` | the practice sites |
| `debriefs` | the folder of the saved debriefs |
<!-- END FIELDS practice_status -->

<!-- BEGIN FIELDS debrief -->
| field of the debrief | content |
|---|---|
| `mission`, `site` | the practice mission's number and site |
| `ended`, `ended_by` | why the mission ended: the `done_reason` of section 3.3 when the robot cell ended it, otherwise the reason the connection closed with; null in a snapshot, which was taken while the mission ran. And why the connection closed: the `reason` of the last line (table above). The two differ, for instance, when your program calls `finish()` after the dose rating has ended the mission |
| `snapshot.sim_time_s`, `snapshot.note` (optional) | only in a snapshot: the simulated time it was taken at, and a note that it was taken while the mission ran, so that a report still queued counts as not delivered |
| `failure.what`, `failure.debrief_as_of_sim_s` (optional) | only when the practice ground failed: what failed and, when the mission's process died, the simulated time the debrief stands for (null when no snapshot was kept; the debrief then holds nothing else) |
| `items[].tag`, `items[].type` | one entry per item of the work order |
| `items[].reports_made`, `items[].delivered`, `items[].reports_lost_at_the_end` | how many reports your program made for the item, whether one reached the team, how many were still queued at the end |
| `items[].true_position` | where the item really is: x, y, z of an instrument's position as the round sheet means it (catalogue, section 8) or of the mark; x, y of the source |
| `items[].rule_point` (optional) | instruments: x, y, z of the reference point that the record photo's distance, angle and size are measured to (a rising-stem valve's lies 0.42 m above its `true_position`) |
| `items[].true_value.*` (optional) | instruments: the true value, in the form a report gives it (temperature point: its hot part and temperature) |
| `items[].evidence.*` (optional) | instruments with a delivered report: the text `accepted (frame F...)`, or `accepted: false` with the list `conditions_failed`: one entry per cited frame, the frame's id and what it missed of the record photo's conditions (`size ... px < ...`, `off-axis ... deg > ...`, `in frame ... < ...`, `tag in frame ... < ...`, `distance ... m > ...` or `< ...`, `visible ... < ...`, `behind the camera`), or `the instrument is not in view`, `taken after the report`, `no such frame`, or that the frame is of the wrong camera (`a navigation camera frame; this instrument needs the inspection camera`); or the one entry `no frame cited` |
| `items[].reported_value`, `items[].error` (optional) | instruments with a delivered report: the value that counts and its error against the true value (null without a value or with an unreadable one); for a lamp panel the error is 0 when every lamp is right and 1 otherwise |
| `items[].tolerance` (optional) | instruments with a delivered value: the tolerance for full credit, in the value's unit (lamp panel: 0.5, with the error of 0 or 1) |
| `items[].true_dose_rate_gyh_where_the_robot_stood` (optional) | the survey mark: with a delivered report, the value it is compared with (null when the robot was not inside the mark's circle at the report) |
| `items[].reported_gyh`, `items[].dwell_s_within_0_30_m`, `items[].dwell_ok`, `items[].relative_error` (optional) | the survey mark with a delivered report: the reported dose rate, how long the robot had stayed inside the circle before the report, whether that was long enough, the relative error |
| `items[].reported_position`, `items[].error_m` (optional) | the source with a delivered report: the reported point and its distance from the source |
| `items[].approach_ok`, `items[].approach_failed` (optional) | the source with a delivered report: whether the approach condition was met (round procedure, section 5), and if not, which half of it failed, in words (null when it was met) |
| `items[].own_dosimeter_peak_3s_gyh`, `items[].own_dosimeter_peak_3s_within_3m_of_reported_gyh`, `items[].closest_to_reported_m` (optional) | the source with a delivered report: the highest 3 s mean reading of the robot's dosimeter before the report anywhere, the highest among the 3 s windows spent within 3.0 m of the reported point (null when there was none), and how close the robot's body centre came to the reported point |
| `end.upright`, `end.distance_from_pad_centre_m` | the robot at the end |
| `events.*` | how often each of `blocked`, `protective_sit`, `robot_lost`, `dose_limit`, `time_limit`, `link_up`, `link_down` occurred |
| `budget_use.sim_time_s`, `budget_use.sim_time_limit_s`, `budget_use.dose_gy`, `budget_use.dose_rating_gy`, `budget_use.program_time_s` | simulated time used and allowed, the dose the sensor head really received and its rating, the program time used |
| `budget_use.frames.nav`, `budget_use.frames.inspect`, `budget_use.frames.depth`, `budget_use.frames.thermal`, `budget_use.frames.lidar`, `budget_use.frame_limits.nav`, `budget_use.frame_limits.inspect`, `budget_use.frame_limits.depth`, `budget_use.frame_limits.thermal`, `budget_use.frame_limits.lidar` | frames and sweeps taken, and their budgets |
| `budget_use.reports.instrument`, `budget_use.reports.mark`, `budget_use.reports.source` | reports made |
| `true_pose.hz`, `true_pose.columns`, `true_pose.samples` (optional) | training hall only: the robot's true pose twice a second, rows of t, x, y, yaw in the site frame |
<!-- END FIELDS debrief -->

The debrief has no score and no drawing of the building as it is.

## 11. What no reply contains

No reply of a mission, scored or practice, carries a true value, a true pose, a credit, the name of an instrument in
an event, or anything about the building beyond its documents. Nothing identifies an instrument's tag plate but its
tag: there is no marker id and no code for an instrument anywhere (the grid marker plates keep their `marker_id`). A
program learns about the building through the robot's sensors.
