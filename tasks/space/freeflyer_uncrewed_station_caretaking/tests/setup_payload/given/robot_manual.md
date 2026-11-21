# CFR-1 cabin free-flyer: operation manual

The CFR-1 is a cube-shaped, fan-propelled free-flying robot for the pressurised cabin of a station. It flies on the
cabin's air with a propulsion module on each side, finds its way with cameras and an inertial unit against a map of the
station, carries an arm with a tendon gripper in its top-aft bay and lives on a dock. This manual is its datasheet for
an unattended sortie: frames, dimensions, propulsion, flight modes, localisation, cameras and lamps, arm and gripper,
what the firmware does on its own, the zone check, the dock, power and the fault responses. The calls are in
`robot_api.md`. The station, its air system and its rules are in `station_handbook.md`; its map is in `station_map/`.

Lengths are in metres, times in seconds of the robot's own (simulated) clock, forces in newtons, power in watts.
Angles are in degrees in this manual and in the arm calls; attitudes are unit quaternions (w, x, y, z); the covariance
that `pose()` reports is in metres squared and radians squared.

## 1. Frames and conventions

- **Station frame**: the frame of the map (`station_map/`): x forward, y to starboard, z toward the deck. `pose()`,
  the goals of `fly_to` and every position of the map are in it.
- **Body frame**: origin at the centre of the cube; +x out of the forward face, +y out of the starboard propulsion
  module, +z out of the bottom face (right-handed). The arm's bay is on the top side (-z) at the aft end (-x).
- **Attitude**: the quaternion q = (w, x, y, z), Hamilton convention, turns a vector of the body frame into the station
  frame. With q = (1, 0, 0, 0) the body axes lie along the station axes: the forward face looks station-forward and the
  top side faces the overhead.
- **Plate frame** of a fiducial plate: origin at the plate's centre, x along its long side, z its normal into the
  cabin, y = z cross x. `station_map/fixtures.json` lists every plate with its centre and axes as surveyed. The goals
  of `fly_rel` are given in this frame, on the plate where it really is.
- **Image frame**: pixel (u, v) with u to the right and v down from the top-left corner. Every frame is a pinhole
  image without distortion: for each metre along the optical axis, the line of sight of pixel (u, v) goes
  (u - cx) / fx metres to the image's right and (v - cy) / fy metres down. Section 5.1 gives each
  camera's optical axis, image-right and image-up directions in the body frame.

## 2. The robot

| item | value |
|---|---|
| mass, arm included | 10.45 kg |
| outer size | a cube of 0.32 m over its soft corner bumpers (0.29 m of rigid structure) |
| centre of mass, arm stowed | (-1.7, -0.4, -10.2) mm in the body frame |
| propulsion | 2 propulsion modules, starboard (+y) and port (-y), each a fan with 6 nozzles |
| battery | 4 packs of 49 Wh: 196 Wh |
| arm | 2 joints (tilt, pan) and a gripper of 2 fingers with 2 joints each, stowed in the top-aft bay |
| cameras that deliver frames | 5: forward imaging, forward wide, forward depth, aft wide, aft depth |
| lamps | 2: forward and aft |
| dock | 2 berths |

Seen from outside the robot is a cube: the core in the middle, a propulsion module on the starboard and on the port
side, soft bumpers on the corners. The forward face (+x) carries the forward imaging, wide and depth cameras and the
forward lamp; the aft face (-x) carries the dock interface, the aft wide and depth cameras and the aft lamp; the arm
lies folded in the bay on the top side at the aft end. `figures/robot_views.png` shows the robot from its sides with the cameras.

### 2.1 Propulsion

Each propulsion module holds a fan that keeps the module under pressure; the air leaves through the module's nozzles
at about 11 m/s. The firmware sets the fan speed by the flight mode and meters the
12 nozzles to make the force and the torque the controller asks for, so the robot moves along any
direction and turns about any axis without turning beforehand.

| nozzle | module | position in the body frame (m) | blows toward | pushes the robot toward |
|---|---|---|---|---|
| N1 | starboard | (0.152, 0.102, -0.040) | +x | -x |
| N2 | starboard | (-0.152, 0.102, 0.040) | -x | +x |
| N3 | starboard | (0.072, 0.152, 0.072) | +y | -y |
| N4 | starboard | (-0.072, 0.152, -0.072) | +y | -y |
| N5 | starboard | (-0.068, 0.102, 0.152) | +z | -z |
| N6 | starboard | (0.068, 0.102, -0.152) | -z | +z |
| N7 | port | (-0.152, -0.102, -0.040) | -x | +x |
| N8 | port | (0.152, -0.102, 0.040) | +x | -x |
| N9 | port | (-0.072, -0.152, 0.072) | -y | +y |
| N10 | port | (0.072, -0.152, -0.072) | -y | +y |
| N11 | port | (0.068, -0.102, 0.152) | +z | -z |
| N12 | port | (-0.068, -0.102, -0.152) | -z | +z |

The jets leave along the body axes named in the table, on every side of the robot (`figures/nozzle_map.png`, with the same
nozzle names).

Largest force along a body axis with no torque, and largest torque about a body axis with no force, by fan level (the
weaker sign of each):

| fan level | fan speed (rpm) | flight modes | force along x (N) | along y (N) | along z (N) | torque about x (N m) | about y (N m) | about z (N m) |
|---|---|---|---|---|---|---|---|---|
| 1 | 2000 | `HUSH` | 0.28 | 0.16 | 0.24 | 0.029 | 0.020 | 0.032 |
| 2 | 2500 | `STANDARD` | 0.44 | 0.25 | 0.37 | 0.046 | 0.031 | 0.049 |
| 3 | 2800 | `AGILE`, `FINE` | 0.55 | 0.31 | 0.47 | 0.058 | 0.039 | 0.062 |

These are capabilities of the propulsion in free flight, with the force through the centre of mass. A force applied
elsewhere on the robot also asks the propulsion for torque, and force and torque share the same nozzles.

## 3. Flight modes and moves

A program selects among 4 flight modes. A further state, `OFF`, is the berth with the fans stopped; it cannot be
selected.

| mode | fan level | speed up to (m/s) | acceleration up to (m/s²) | turn rate up to (deg/s) | angular acceleration up to (deg/s²) | path tolerance (m) | end tolerance (m) | attitude tolerance (deg) |
|---|---|---|---|---|---|---|---|---|
| `STANDARD` | 2 | 0.20 | 0.0175 | 10 | 10.0 | 0.25 | 0.10 | 20 |
| `AGILE` | 3 | 0.40 | 0.0200 | 30 | 14.3 | 0.20 | 0.10 | 20 |
| `HUSH` | 1 | 0.02 | 0.0020 | 1 | 1.0 | 0.10 | 0.10 | 10 |
| `FINE` | 3 | 0.02 | 0.0100 | 10 | 10.0 | 0.10 | 0.10 | 10 |

- **Changing the mode** (`set_mode`). When the new mode has another fan level the call waits until the fans have
  reached it: 4.4 to 6.9 s in 8 measured changes. Between modes of the same fan level the
  change takes no waiting.
- **A move** (`fly_to`, `fly_rel`) is a straight segment, flown from rest to rest with a trapezoidal speed profile
  inside the mode's limits (`max_speed` lowers the speed limit, down to 0.005 m/s). With `face_forward` the robot
  turns its forward face along the segment, flies it, then turns to the goal attitude. Without it, and always in
  `fly_rel`, it translates and turns together, and the linear and the angular acceleration limit are divided by 2.
- **Tolerances**. While a move runs the firmware compares the estimate with the planned path. More than the path
  tolerance off for longer than 1 s: the move is stopped and returns `TOLERANCE_PATH`; the same for the
  attitude tolerance, `TOLERANCE_ATT`. More than the end tolerance from the goal at the planned end:
  `TOLERANCE_END`. Otherwise the move returns `SUCCESS` when the robot has come to rest. A move that has not come to
  rest 30 s after its planned end is stopped and returns `TIMEOUT`.
- **No look-ahead**. The planner checks the segment against the zone map (section 8) and nothing else: it does
  not look for obstacles, items or the station's hardware on the way.
- **Station keeping**. After a move, `hold()` or `stop()` the controller holds the pose where the robot came to rest,
  on the estimate. Over 600 s the distance to the held point was 0.045 m RMS (median of 8 runs, largest
  0.059 m) with the cabin lights in general mode, and 0.17 m (largest 0.30 m) in a module without usable light
  (section 7.1).
- **Reliability**. In the acceptance campaign 400 of 400 moves in general lighting and 249 of 250 in night
  lighting returned `SUCCESS` at the first call.

## 4. Localisation

`pose()` returns the firmware's estimate of the body frame in the station frame, the diagonal of its covariance and
a confidence. The estimate comes from 3 sources (`loc_mode`):

| `loc_mode` | when | what it uses |
|---|---|---|
| `MAP` | always, except in the cases below | the mapped visual features that the forward wide camera sees, with the inertial unit in between |
| `FIDUCIAL` | inside `fly_rel`, and in the berthing of `dock` | the fiducial plate named in the call, seen by the aft wide camera |
| `RAIL` | inside `capture`, from the lock to the end of the call | the locked bar, seen by the aft depth camera |

- **The reported covariance is honest.** In the acceptance campaign the true position error divided by the reported
  one-sigma value had a median between 0.64 and 1.07 on routes through night-lit modules.
- **`MAP` accuracy.** With the cabin lights of the module in general mode the reported uncertainty settles at
  0.028 m per axis and 0.9° about each axis (one sigma).
- **What `MAP` lives on.** The estimator matches mapped features within 60° of the forward (+x) axis, between
  0.3 and 6 m, that face the robot and are not hidden. A feature counts by the light on it: 1 in a module whose
  cabin lights are in general mode, 0.05 in night mode, 0 where the light circuit has failed, plus what the robot's own
  lamps add (section 5.5). About 40 fully lit features give the full accuracy; with too few the estimator runs on the
  inertial unit alone and its uncertainty grows with time.
- **`FIDUCIAL` accuracy.** About 1 cm (RMS of the error norm) at 0.6 m from the plate, in proportion to the
  distance, not below 2 mm.
- **When a plate counts as seen.** The plate lies within 40° of the aft wide camera's axis and inside its
  delivered frame, 0.05 to 1.5 m from the lens, seen within 75° of its normal, and the lines of sight to at least
  3 of its markers are free: not blocked by an item in the hand, by the robot itself as the aft wide camera's frame
  shows it (with the arm stowed or at the grasp attitude no part of the robot is in that frame) or by the station.
  A plate that drops out of view leaves the estimate on the inertial unit; lost for more than 20 s, `fly_rel` stops
  and returns `NO_FIDUCIAL`.
- **Inertial unit.** It sits near the centre of the cube. `status()` gives its latest sample, the linear
  acceleration and the turn rate in the body frame, taken at 62.5 Hz.
- **Examples** from a place in the middle of a night-lit module (reported one-sigma position uncertainty per axis):

| view and light | reported uncertainty |
|---|---|
| facing the rack wall across the module, cabin lights in night mode | no fix: from 0.044 to 0.074 m in 41 s, still growing |
| the same, aft lamp on | no change |
| the same, forward lamp on | 0.037 m, reached 6 s after switching on |
| turned toward the open hatch of a module in general lighting | 0.032 m |
| the module's lights requested to general mode | 0.028 m, reached 3 s after the request |

The forward lamp lights a feature as the cabin's general lighting does at about 1.7 m on its axis and
falls off with the square of the distance; the aft lamp lights what is behind the robot, where the forward wide
camera does not look.

## 5. Cameras and lamps

### 5.1 The cameras

| name | camera | frame (px) | fx, fy (px) | cx, cy (px) | field of view (deg) | lens position in the body frame (m) | looks along | image right is | image up is |
|---|---|---|---|---|---|---|---|---|---|
| `FIC` | forward imaging camera, colour | 1366 × 1024 | 859.44, 849.35 | 754.24, 487.73 | 76.7 × 62.1 | (0.140, 0.000, -0.096) | +x | +y | -z |
| `FWC` | forward wide camera, colour | 1280 × 960 | 611.05, 610.21 | 637.96, 558.05 | 92.7 × 75.8 | (0.137, -0.042, -0.083) | +x | +y | -z |
| `FDC` | forward depth camera, depth | 224 × 171 | 217.13, 216.12 | 112.04, 82.60 | 54.6 × 43.2 | (0.133, 0.036, -0.083) | +x | -y | +z |
| `AWC` | aft wide camera, colour | 1280 × 960 | 830.01, 829.33 | 566.95, 529.29 | 75.0 × 60.0 | (-0.125, -0.054, -0.006) | -x | -y | -z |
| `ADC` | aft depth camera, depth | 224 × 171 | 209.21, 207.62 | 94.69, 84.04 | 56.1 × 44.8 | (-0.133, 0.051, -0.017) | -x | +z | -y |

`camera(name)` returns a frame of the named camera with these intrinsics in the reply; every frame counts against
the site's frame budget (`robot_api.md`). The lens positions and directions are fixed in the body frame: a camera's
pose in the station frame follows from `pose()`. Frames of the forward imaging camera and of the forward wide camera
can be cited as evidence in `report`, on the same terms (`robot_api.md`).

- The forward imaging camera and the forward wide camera look the same way with the same side up: image up is the
  robot's top side (-z), image right is starboard.
- The forward depth camera's image is turned by 180° against theirs: its image up is the robot's bottom side.
- The aft wide camera looks aft with the robot's top side up; port is on its right.
- The aft depth camera's image is turned by 90° against the aft wide camera's: its image up is port and its image
  right is the robot's bottom side.
- The principal point (cx, cy) is not the centre of the frame: the angles to the frame's edges differ from side to
  side.

### 5.2 Colour frames

`FIC`, `FWC` and `AWC` deliver RGB frames of uint8 values. The wide cameras' lenses see more than the delivered
frame: the frame is the pinhole image of the table, cut from the lens's field.

- **Exposure** is automatic and each camera keeps its own state: a centre-weighted meter sets the gain, between
  1/16 and 8 times the base exposure, starting from the gain of the camera's previous frame, and holds back when
  highlights would clip. Frames carry read noise and shot noise.
- **Light**. At most 8 light sources contribute to a frame: the fill light of the module the camera is in, the
  robot's lamps that are on, and the cabin lights nearest to the camera. Night mode dims the module's lights; a
  failed light circuit leaves its module dark whatever the station reports as commanded.
- **Near limit**. Nothing nearer than 1.5 cm along the optical axis is in a frame.

### 5.3 Depth frames

`FDC` and `ADC` deliver a depth image (float32, metres along the optical axis, 0 where nothing returns) and an
amplitude image (uint16). They range from 0.1 to 4 m. The noise of a pixel is 1 % of its range (one sigma), and
the image comes in steps of 1 mm. A depth frame returns the robot's own arm and fingers from the robot's contact
model, which lies within about 5 mm of the surface the colour frames show.

### 5.4 What the cameras see of the robot itself

With the arm stowed (tilt 180), at tilt 90 and at the grasp attitude (tilt 0) no frame of any camera shows any
part of the robot. Tilted beyond the grasp attitude the arm swings into the aft cameras' view:

| arm tilt | -10° | -20° | -30° |
|---|---|---|---|
| share of the aft wide camera's frame that shows the arm and fingers | 1.7 % | 6.1 % | 10.9 % |

The arm and fingers enter the aft wide camera's frame from about tilt -10°, at its top edge, and the aft depth
camera's frame from about tilt -20°, at its left edge.

### 5.5 Lamps

| lamp | position in the body frame (m) | shines along |
|---|---|---|
| `forward` | (0.135, 0.000, 0.037) | +x |
| `aft` | (-0.135, -0.054, 0.030) | -x |

Each lamp gives 216 lm in a beam of 45° (452 cd on its axis) and draws 3.3 W while it is on.
`lamp(which, on)` switches it. A lamp lights the scene for the cameras on its side; the forward lamp also lights the
features the localisation uses (section 4).

## 6. Arm and gripper

### 6.1 Joints

| joint | range (deg) | meaning |
|---|---|---|
| tilt | -30 to 180 | 180: stowed, folded into its bay. 90: the hand points straight out of the top side. 0: the grasp attitude, the hand points straight aft. Below that the hand swings on toward the aft face. |
| pan | -90 to 90 | turns the hand sideways, about an axis that is parallel to the body z axis at the grasp attitude. 0: straight. A positive pan turns the hand toward port (-y), a negative one toward starboard (+y). |

- The joints are position servos that move at 0.12 rad/s (6.9°/s): `arm_deploy` (stowed to the grasp attitude)
  took 26.7 s and `arm_stow` 26.7 s in 3 measured runs each.
- Arm calls return at once and the arm moves while the robot flies, or they block until the joints have arrived
  (`wait=True`).
- The firmware tells these states from the joint encoders. **Stowed**: tilt and pan within 5° of tilt 180, pan 0
  (`dock` needs it). **Deployed**: the tilt encoder or the tilt command more than 5° from stowed (the zone check
  uses it, section 8). **At the grasp attitude**: tilt and pan within 3° of tilt 0, pan 0 (`capture` needs
  it).

### 6.2 The gripper

The gripper's fingers are closed by springs and opened by a tendon that a small motor winds. `gripper(open=True)`
winds the tendon and the fingers open; `gripper(open=False)` lets it go and the springs close the fingers on whatever
lies between them, or on each other. A full stroke takes 1 s. The closed grip needs no power.

- **It closes, it does not clamp.** The closed fingers form a cage around a bar of handrail section. A caged bar
  keeps play: it can shift inside the cage, toward the fingertips and back, by some centimetres (the measurement of
  section 7.2), and turn about its own axis, and what hangs on the bar turns with it.
- **Sized for a crew handrail.** The station's crew handrails have a section of 19 mm by 30 to 35 mm. The bars on loose
  items (carry strap, handle loop, pouch handle) have a section of 19 × 35 mm. The carry strap and the handle loop of a
  sheet-like item are up to 44 mm long; the handle of a pouch is longer (`lost_item_register.md`, pouches).
- **Detachment.** Pulled straight off a crew handrail, the closed gripper let go at 6.4 N (the makers' measurement,
  mean of 5 trials).
- **Holding over time.** In 12 holds of 30 min each, hovering with a sheet-like item held by its carry strap or
  its handle loop while the cabin fans ran, the item stayed in the hand in 12 of them. In some of the holds the bar was
  outside the cage for a time and the item stayed hooked on the fingers.

### 6.3 Where the hand is

At the grasp attitude the hand points straight aft from the top-aft corner (`figures/grip_envelope.png`). In the body frame:

| point | position (m) |
|---|---|
| palm (the root of the fingers) | (-0.229, 0.000, -0.130) |
| fingertips | (-0.344, 0.000, -0.180) |
| centre of a caged bar of a loose item | (-0.327, 0.000, -0.180) |
| centre of a caged crew handrail | (-0.311, 0.000, -0.180) |

The fingers lie in the plane z = -0.180 m of the body frame and close around a bar whose axis is parallel to the
body z axis. The positions of a caged bar are where the firmware closes (section 7.3), on its own estimate of
the bar; the error of that estimate and the play of the cage come on top. In this manual's measurement the centre of
the caged crew handrail ended at (-0.335, -0.002, -0.168) m.

### 6.4 What the arm reports

`status()` gives the joint angles from the encoders (resolution 0.088°, refreshed at 10 Hz), the currents of the
joint servos (the servo's torque divided by 1.47 N m/A, in steps of 2.69 mA) and the tendon position in
percent (100: wound, fingers commanded open; 0: let go). The tendon position is the motor's, not the fingers'. There
is no finger angle, no gripper motor current and no contact sensor.

## 7. What the firmware does on its own

The calls of `robot_api.md` run on firmware that has rules of its own. This section states them; none is advice.

### 7.1 Localisation confidence

The estimator reports its own uncertainty and a confidence (section 4). The confidence is `GOOD` while the
position uncertainty is below 0.20 m and the attitude uncertainty below 20° (one sigma, the geometric mean over the axes);
`POOR` when either is above, `LOST` when the position and the attitude are above. While the confidence is not
`GOOD` the controller idles: no force, no torque, and the robot drifts on with the velocity it has. A move that is
running ends with `LOCALIZATION`; `fly_to` and `fly_rel` called in that state return `LOCALIZATION` without moving
the robot. The firmware does nothing else about it: it switches no light, turns toward nothing and repeats no move.

Where the forward wide camera matches too few lit features the estimate runs on the inertial unit and its uncertainty
grows: a robot that kept station in a module without usable light, starting from a good estimate, went `POOR` after
457 s (in 4 of 8 runs; the others kept a fix). On routes through night-lit modules the true position error was
0.08 to 0.15 m in the median and 0.21 to 0.38 m at the 95th percentile, with the confidence `GOOD`
throughout, and 10 of 10 routes ended docked. Light on the mapped features brings the uncertainty down within seconds
(the examples of section 4).

### 7.2 Pulling against something fixed

A move is a position command. When something holds the robot back, the controller turns the growing position error
into a growing force command; past what the propulsion can give (section 2.1) the nozzles saturate in turn,
and the force that is delivered stops growing and can fall. A pull at the hand acts 0.33 m aft of the centre of
the cube and 0.18 m toward its top side, so it asks for torque as well as force. Held back by something fixed, the
steady pull is largest in `FINE`, smaller in `AGILE` and smallest in `STANDARD`. The firmware reports no force.

A move ends with `SUCCESS` when the robot has come to rest within the end tolerance of the goal, 0.1 m in every
mode. A goal nearer than that to the place where the robot is held is reached without the robot moving. Measured with
the gripper closed around a crew handrail: a move of 0.05 m away from the handrail returned `SUCCESS` while the
robot moved 0.03 m; the next move, of 0.30 m, returned `TOLERANCE_PATH` after 10 s, the robot having moved a further 0.02 m; after
`gripper(open=True)` the same move returned `SUCCESS`.

### 7.3 The capture hint and the closing

`capture` closes the gripper around a bar, and the program has to say which bar: the centre of the bar (`bar_pos`,
the middle of its length) and the direction of its axis (`bar_axis`), in the body frame at the moment of the call.

- **Detector.** The firmware's bar detector works on the aft depth camera at 5 Hz. It measures every bar of
  handrail section it sees: the bars of loose items and the crew handrails. A bar is seen when at least 3 of 5
  points along it lie in the camera's field, 0.125 to 1.2 m from the lens, are seen within 75° of the normal of the
  bar's face and more than 30° off the bar's axis, and are not hidden. The detector's field is the camera's full
  field, 59° along the image's width and 47° along its height, about the optical axis; the delivered frame
  shows a little less of it (section 5.1). A measurement has a noise of 5 mm and 3° (one sigma).
- **Gate.** A bar locks when the mean of its last 3 measurements, all taken after the call, lies within 4 cm of
  the hinted centre and 8° of the hinted axis; of several bars inside the gate the best fit wins, the distance and the angle each counted as a fraction of its gate. Nothing locked
  5 s after the call: `NO_BAR`. The firmware does not search, does not widen the gate and does not take the bar
  from the map.
- **Approach.** After the lock the firmware flies on the bar. It opens the gripper, lines the fingers up with the bar 0.2 m outside
  the fingertips, to within 12 mm sideways and 10° in attitude, at up to 0.03 m/s in the current mode; then it
  switches to `FINE` and goes straight in at 0.02 m/s without turning. The robot ends with its aft face toward the bar
  and the bar along the body z axis. Of the attitudes that do this, `palm_dir` picks the one whose +z axis points to
  the side of `palm_dir`; without it the firmware takes the smaller turn.
- **Blind end.** Near the hand the bar leaves the camera's view, and the firmware flies the rest on the inertial
  unit: the last 14 s in this manual's measurement on a crew handrail.
- **Closing.** The fingers close when the tracked bar is 98 mm (a bar of a loose item) or 82 mm (a crew handrail) in
  front of the palm; the program cannot choose the depth. `DONE` comes 1.5 s after the closing.
- **Other endings.** `LOST_BAR`: the locked bar unseen for 3 s before the fingers are lined up. `STALLED`: the
  fingertips no nearer to the closing point by 5 mm for 20 s. `TIMEOUT`: not closed 150 s after the call. A
  `capture` that ends after the lock without closing leaves the gripper open. The approach of `capture` is not
  zone-checked.
- **Durations.** The lock came 0.4 s after the call and `DONE` after 40 s in this manual's measurement. On the
  firmware's bench 60 of 60 captures with the hint inside the gate ended `DONE` with the bar in the cage, after
  42 to 56 s (median 48 s).
- **Bars close together.** The carry strap and the handle loop of a sheet-like item lie on a common line, about
  0.12 m apart; the detector measures each bar it sees, and the gate decides.

### 7.4 No grasp feedback

Nothing that the firmware returns says whether the gripper holds something. `capture` returns `DONE` when the
closing has been carried out, around a bar or around nothing. `gripper` returns the tendon position, which is the
motor's. `status()` has no finger angle, no gripper motor current and no contact sensor (section 6.4). `dock`
reads the arm joints only. At the grasp attitude no camera sees the fingers (section 5.4).

What the sensors do show is the mechanics. Fingers that close on a bar that is fixed, to the station or to an item
that the station's suction holds, tie the robot to it, and a move away that is longer than the end tolerance does not
arrive (section 7.2). The arm's joint servos carry the pull, and their currents show little of it: with the
gripper closed around a crew handrail they read at most 1 step of the reading (section 6.4) in the 3 s after
`DONE`, up to 2 steps after the move of 0.05 m and up to 3 steps after the move of 0.30 m of
section 7.2, against 0 in a free hover with the arm deployed. The station's side of a return inlet, and
what its telemetry measures there, is in `station_handbook.md` and `telemetry_dictionary.md`.

### 7.5 The holding interlock

From the `DONE` of a `capture` until the program calls `gripper(open=True)` the firmware refuses every arm call:
`arm_deploy`, `arm_stow` and `arm` return `REFUSED_HOLDING` and the joints keep their angles; another `capture`
returns `REFUSED_HOLDING` too. The interlock follows the program's own calls, not a sensor: it is on after a `DONE`
on an empty closing as well, and it is not on after `NO_BAR`, `LOST_BAR`, `STALLED` or `TIMEOUT`.
`gripper(open=True)` takes it off at once. `status()` shows it (`interlock`).

### 7.6 HUSH and its thrust margin

`HUSH` runs the fans at level 1. It has the least margin against a disturbance: the smallest thrust to answer it with,
0.16 N along the weakest body axis against 0.25 N in `STANDARD` (section 2.1), and the least room before a tolerance stops
the move, 0.10 m off the path against 0.25 m and 10° in attitude against 20° (section 3). In the acceptance campaign
36 of 36 test moves in `HUSH`, each a straight move along a module wall with the cabin fans running, returned
`SUCCESS`. A change between `HUSH` and another mode took 5.4 to 6.9 s in 4 measured changes.

### 7.7 What a call leaves behind

- **Flight mode.** `capture` switches to `FINE` for its straight approach and does not switch back: after `DONE` the
  robot is in `FINE`, and a later move flies with the limits of `FINE` until `set_mode` is called. `undock` leaves the
  robot in `STANDARD`, `dock` in `OFF`. No other call changes the mode.
- **Estimate.** `fly_rel` flies on the plate and `capture` on the bar only while the call runs. When the call ends
  `pose()` reports `MAP` again, and the robot holds its place on the map estimate without a jump. `fly_rel` returns as
  `fly_to` does, at rest within the end tolerance of its goal: in 3 measured calls the robot ended 8 to 13 mm from
  the goal. Afterwards it keeps station as on any map estimate (section 3).
- **Hold after a closing.** From the closing of a `capture` until the next call that moves the robot (`fly_to`,
  `fly_rel`, `hold`, `stop`, `dock`, `undock`, `capture`) the robot holds the place where it stopped by its inertial
  unit, and the wander of the map estimate does not move it.

## 8. The zone check

`station_map/zones.json` holds the keep-in and keep-out boxes of the map. The flyable volume is the union of the
keep-in boxes without the keep-out boxes. The firmware checks moves against it on a grid of 0.08 m cells, with the
robot as the point at its body origin and the walls of the volume made thicker by an inflation:

| arm | inflation (m) | on the grid (m) |
|---|---|---|
| stowed | 0.30: a robot radius of 0.28 and a margin of 0.02 | 0.32 |
| deployed | 0.50 | 0.56 |

The inflation is rounded up to whole cells: in the station's x-y plane it is a disc of that radius, along the
station's z axis the same distance up and down. The arm counts as deployed as section 6.1 says.

How the boxes fall onto the grid: the cells have their corners on whole multiples of 0.08 m in the station frame. A
keep-out box occupies every cell that holds a part of one of its faces; a keep-in box is walled by the cells that hold
the points 0.08 m outside its faces, wherever no other keep-in box continues the volume. The inflation is added to
every occupied cell in whole cells (4 cells with the arm stowed, 7 deployed). So the free volume begins 0.32 to
0.40 m from a face of a keep-out box (0.57 to 0.64 m with the arm deployed) and 0.24 to 0.32 m inside a face of a
keep-in box (0.48 to 0.56 m deployed), depending on where the face falls in its cell. Near the edges and corners of a
box, and where keep-in boxes meet, the band follows the disc of the inflation and can be narrower.

- **Moves.** `fly_to` and `fly_rel` plan their segment from the estimate and are refused with `REFUSED_ZONE`, before
  the robot moves, when a point of the segment or its end lies outside the keep-in volume or inside an inflated
  wall.
- **Starting inside an inflated band.** A segment that starts inside the inflation is accepted when it leaves the
  band moving away from the wall, or when it never comes nearer to the wall than it started and does not come back
  into a band after leaving it.
- **Arm calls.** `arm_deploy`, and `arm` toward a tilt that counts as deployed, are refused with `REFUSED_ZONE`
  when the robot's estimated position is not free with an inflation of 0.50 m. `arm_stow` is not checked.
- **Not checked**: the approach of `capture` after the lock; the moves of `dock` and `undock` between the berth and
  its approach point; and, for `fly_rel` with the arm deployed, the part of the segment inside the approach zone of
  the plate's own stowage unit or of the dock, where the stowed inflation is used instead. An approach zone is the
  outline of the unit (or of the dock) grown by 0.2 m, from the wall (for the dock: from its face) to 1.0 m into the
  cabin; `station_map/fixtures.json` gives each zone as a box. `fly_to` has no such exception.

## 9. Dock and berths

The dock has 2 berths. The robot sits on a berth with its aft face on the dock; on berth 2 it sits rolled by
180° about its x axis against berth 1. `station_map/fixtures.json` gives, for each berth, the approach point in front of
it, the attitude there and the axis along which the robot leaves.

- **On the berth** the fans are stopped (`flight_mode` reads `OFF`), the dock holds the robot and feeds it
  (section 10). `fly_to`, `fly_rel`, `arm_deploy`, `arm` and `capture` return `REFUSED_STATE`.
- **`undock()`** spins the fans up to `STANDARD`, has the dock release the robot and flies it 0.5 m out to the approach
  point: 29 to 32 s in 3 measured runs.
- **`dock(berth)`** needs the arm stowed (`REFUSED_ARM` otherwise; the firmware does not know what the gripper
  holds) and the estimated position within 1.0 m of the berth's approach point (`REFUSED_CONE` otherwise). The firmware
  flies to the approach point, takes the dock's fiducial plate with the aft wide camera and checks the start of the
  berthing: within 6° of the berth's axis, seen from the berth, and within 10° of the berth's attitude. Outside,
  it goes back to the approach point and tries again, at most 2 times, then returns `REFUSED_CONE`. Inside, it
  berths in `FINE` at 0.02 m/s, checks that the dock holds it and stops the fans: `DOCKED`. `FAILED`: the berthing ended
  without the dock holding the robot.
- **Durations.** `dock` took 67 s from the approach point and 92 s from 0.9 m away from it. In the acceptance
  campaign 70 of 70 berthings started inside the limits ended docked.

## 10. Power

The battery holds 196 Wh when full: each percent of charge is 1.96 Wh. `status()` gives the charge in percent,
refreshed at 1 Hz in steps of 0.1 %, and the flag `low_battery` below 25 %. The flag is a notice: the
firmware does nothing else at any charge.

| state | draw (W) | of it the fans (W) |
|---|---|---|
| keeping station in `HUSH` (fan level 1) | 89 | 10 |
| keeping station in `STANDARD` (fan level 2) | 99 | 20 |
| keeping station in `AGILE` (fan level 3) | 108 | 29 |
| keeping station in `FINE` (fan level 3) | 108 | 29 |
| on the berth, fans stopped | 79 | - |

- The computers, the sensors and the arm's electronics draw 79 W at all times; the fans add to it by their level and
  by the thrust they give; each lamp that is on adds 3.3 W.
- Whole sorties of the acceptance campaign, flown in `STANDARD`, drew 98 W on average.
- On the berth the dock feeds the robot 72 W: with the fans stopped the charge still falls, at 7 W.

## 11. Fault responses

| condition | what the firmware does | what the program sees |
|---|---|---|
| the confidence of `pose()` is not `GOOD` | the controller idles (no force, no torque) until it is `GOOD` again; the robot keeps its velocity | a running move: `LOCALIZATION`; a new move: `LOCALIZATION`, and the robot is not moved |
| estimated speed above 0.50 m/s during a move | the move is stopped | `OVERSPEED_STOP` |
| estimated speed above 0.75 m/s | the power of the propulsion modules is cut for the rest of the sortie; the fans run down, and the thrust they were giving is down to half after 4.4 s at fan level 2 (3.9 s at level 3) | a running move ends without reaching its goal; later moves give no thrust |
| path or attitude tolerance exceeded for more than 1 s | the move is stopped | `TOLERANCE_PATH`, `TOLERANCE_ATT` |
| a move not at rest 30 s after its planned end | the move is stopped | `TIMEOUT` |
| charge below 25 % | none | `low_battery` in `status()` |
| the site's time is used up | the firmware stops and holds | `TIME_UP` (`robot_api.md`) |

The firmware has no sense of touch: it does not notice contact with the station, an item or the dock, and it does
not avoid it.
