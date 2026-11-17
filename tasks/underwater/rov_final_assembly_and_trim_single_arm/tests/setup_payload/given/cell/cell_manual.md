# Final-assembly station: integrator's description of the robot cell (commissioning document)

This document describes the station's robot cell as it is built and as it behaves. Its numbers are constants of the
cell or values measured in it. The part files, the drawings and the photographs show the vehicle as it is drawn; where
the cell's collision shapes differ from that, "Collision model" lists the difference and its size. A run is one
episode: one start of the cell, with one vehicle, its work order and its own layout.

Every vehicle that is released to this station has been checked by production planning: with the station's standard
ballast blocks on the free holes of its keel it can be trimmed into the acceptance windows of the trim sheet
(`~/given/docs/trim_sheet.png`).

## Frame, bench, robot

* **Cell frame**: origin on the bench top at the robot arm's base axis, z up, x towards the turnover positioner, y to
  the left seen from the arm. The bench top (2.0 x 1.4 m, x -0.55 .. 1.45 m, y -0.70 .. 0.70 m, 0.994 m above the hall
  floor) is the plane z = 0. The cell stands in a factory hall, lit by the hall's ceiling lamps.
* **Robot**: a 6-DoF UR10e-class arm (model in `~/given/robot/universal_robots_ur10e/`), bolted to a surveyed datum:
  its base at (0.00, 0.00, 0.00) m, the base frame being the cell frame turned 180 deg about z (q = 0 on the first
  joint points the arm along the cell's -x; exact). Joint names `ur/shoulder_pan_joint`, `ur/shoulder_lift_joint`,
  `ur/elbow_joint`, `ur/wrist_1_joint`, `ur/wrist_2_joint`, `ur/wrist_3_joint`; the joint ranges and the home posture
  are in `describe()`. Joint torque limits 330 / 330 / 150 / 56 / 56 / 56 N m, base to wrist.
* **Tool**: a wide-stroke parallel gripper (model in `~/given/robot/wide_gripper/`) on the tool flange (the flange
  face sits at the arm's d6 = 0.11655 m along the last link's axis; the given arm model's `attachment_site` is 16.6 mm
  short of it). The tool point `ur/tcp` lies on the tool axis, 0.1962 m from the flange face; its axes: z = the
  approach (the flange axis, out of the flange), y = the closing axis, x across the pads. "The gripper" below has the
  fingers' geometry.
* **Wrist camera housing**: a 36 x 24 x 35 mm box on the gripper's base, on the tool's +x side; it is part of the
  robot (its collision model and the safety rule). In the tcp frame its centre is at (0.056, 0.000, -0.178) m and its
  outer face at x = 0.068 m.
* **Force / torque sensor** at the tool flange's centre, in the flange frame (z along the tool axis outwards, x along
  the closing axis = the tcp's y, y along the tcp's -x): what the wrist applies to the tool, the gripper's own weight
  included (about 11.2 N with the tool pointing down); noise about 0.3 N / 0.02 N m rms per reading, an unknown bias
  up to about 1 N / 0.05 N m that differs between power-ups: tare it.

## The gripper

A parallel gripper with two fingers on a four-bar linkage. It only pinches between two flat pad faces; it does not
wrap around a unit. The given model (`~/given/robot/wide_gripper/`) and the gripper module of the given library
(`arm.assembly_rov.gripper_wide`, with the measured table of the whole stroke) are the cell's own.

* **Pad faces.** Each finger carries one flat pad face, 27 mm across and 65 mm long from its root to the fingertip
  end. Behind the last 12 mm of the face, at the fingertip, the finger is a blade only 3.0 mm thick; behind the rest
  of the face it is 5.0 mm thick. In the given model each pad face is two pads in one plane, one on the blade and one
  behind it: these are the gripper's four pads of "Safety".
* **Width.** The width is the gap between the two pad faces at the faces' middle: 7.4 mm closed to 133.6 mm fully
  open. The faces are not parallel over the whole stroke: fully open they close in towards the fingertips by 1.8 deg
  (the gap is 135.6 mm at the root of the faces and 131.6 mm at the fingertip ends), at a width of 114.3 mm by 1.3
  deg, and at widths below 60 mm they are parallel within 0.4 deg. The commanded width moves at 0.10 m/s.
* **The faces travel along the approach as the gripper closes** (the linkage's arc). The middle of the pad faces lies
  188.6 mm from the flange face fully open, 196.2 mm at a width of 114.3 mm (this is the tcp) and 212.9 mm at 12.8 mm.
  The fingertip ends lie 221.4 mm from the flange face fully open and 245.4 mm at 12.8 mm.
* **The pinch centre is not on the tool axis.** The point midway between the two pad faces lies off the tcp along the
  closing axis, towards +y: by 0.5-0.6 mm at widths of 42-59 mm. The given model has the same offset;
  `arm.assembly_rov.gripper_wide.pad_offset(width)` returns it for any width.
* **Outer envelope.** Along the closing axis the fingers' outer envelope spans +-101.4 mm fully open (the whole hand
  is then 203 mm wide), +-91.3 mm at a width of 114.3 mm and +-63.5 mm at 12.8 mm.
* **Grip force.** The grip force is commanded: 10 to 125 N on each pad (`cell_api.md`, `command`). The gripper closes
  until the pads press with that force; it reaches it once the commanded width is about 3 mm below the width of what
  it holds. Measured on flat and on round units from 12.7 to 114.3 mm wide, the force on each pad was within 5 % of
  the command.
* **Mass.** The gripper weighs 1.14 kg; the arm carries it.
* **Carrying a unit.** The station's handling procedure for a unit in the gripper: the tool point no faster than 0.25
  m/s and no joint accelerated by more than 8 rad per second squared. The cell does not enforce these two limits (its
  own limits are those of `cell_api.md`, and `command` has no limit on acceleration). Beyond them a heavy or a round
  unit turns between the flat pads under its own inertia. The limits do not apply to the empty gripper.

## Datum boards

Checkerboards printed on rigid plates. On a board of C x R squares of side s the inner corner (i, j), i = 1 .. C - 1,
j = 1 .. R - 1, sits at ((i - C / 2) x s, (R / 2 - j) x s) in the board's own frame (x along the C squares, y along
the R squares, z out of the printed face, origin at the middle of the pattern); the square at the board's (-x, +y)
corner is black.

| board | squares | where |
|---|---|---|
| bench board | 9 x 6 of 20.0 mm (3.0 mm plate, 20 mm white margin) | surveyed, exact: flat on the bench, the board's origin at (1.200, -0.500, 0.00305) m in the cell frame, its x along the cell's x, its y along the cell's y |
| tool plate | 5 x 4 of 8.0 mm (1.5 mm plate, 6 mm margin) | on the camera housing's outer face, facing the tool's +x: the board's origin at (0.0700, 0.0000, -0.1777) m in the tcp frame, its x along the tcp's z, its y along the tcp's -y; mounting tolerance about 1 mm / 0.5 deg |
| tray board | 6 x 5 of 12.0 mm (3.0 mm plate, 10 mm margin) | flat, printed face up, on the thruster tray, its printed face 63.05 mm above the bench, its x along the tray's y, its y along the tray's -x. It moves with the tray; its place on the tray and the nests' places relative to it are dimensioned in `~/given/drawings/trays_and_racks.png` |

Only the bench board and the tool plate are surveyed. The tray board tells where the thruster tray lies in a run.

## The turnover positioner and the vehicle

* **Vehicle frame**: the frame of `~/given/parts/base_vehicle.stl` and of the drawings: x forward, y to port, z up,
  millimetres.
* **Positioner** (`~/given/drawings/positioner_and_cells.png`): a drive stand beyond the vehicle's bow carries a
  U-shaped fork that clamps the front edges of the vehicle's two side plates; the vehicle hangs clear of the bench.
  The vehicle's own gripper, a part of the base vehicle, reaches forward through the fork's opening. The fork's outer
  jaws end at least 6.0 mm short of where the front ends of the thruster guards come to lie. The positioner holds the
  vehicle with the origin of the vehicle frame at (0.60, 0.00, 0.42) m in the cell frame and the vehicle's x axis
  along the cell's x axis: the stern points to the robot arm. It turns the vehicle about the vehicle's x axis between
  two index positions half a turn apart: `deck_up`, the vehicle's z axis up, and `keel_up`, the vehicle's z axis down,
  so that the underside of the bottom plate faces up. Every run starts at `keel_up`.
* **The vehicle's pose is exact.** At an index position the vehicle stands where this document says within the
  locating pins' +-0.2 mm and the indexing repeatability of +-0.3 mm and +-0.1 deg. Nothing else about its pose
  differs between runs.
* **Base vehicle** (`base_vehicle.stl`): the vehicle as it arrives from upstream; `~/given/docs/station_scope.md`
  lists what is on it. In the cell it is one rigid body on the fork: nothing on it moves.
* **Turning** (`fixture_index` in `cell_api.md`). A turn between the two index positions takes 6 s. It is a programmed
  motion of the positioner: the vehicle follows it exactly, whatever it carries, and is held on the index position by
  the positioner's pin afterwards. A run has at most 4 turns.
* **Turn column.** The turn column is the cylinder of radius 0.40 m about the turn axis, over the whole length of the
  positioner and the vehicle. The positioner refuses to turn while any part of the robot is inside it. A refused turn
  is not a violation and does not count as one of the run's turns. The interlock is checked at the moment a turn is
  asked for, and it takes every link of the robot as its convex hull: close to the column's surface it can refuse
  while the arm itself is still outside. Move further away and ask again. During the 6 s of a turn the robot should
  stay outside the column: the interlock does not watch it then, and a contact with the turning vehicle or with the
  positioner counts under the first rule of "Safety". As it arrives the vehicle sweeps a radius of 252 mm when it
  turns, and 289 mm with every unit of this station on it.
* **Turning procedure: nothing loose on the vehicle.** When a turn starts, the station scraps every unit on the
  vehicle that is not fastened and takes it out of the cell. On the vehicle means: touching the vehicle, the
  positioner's fork or a fastened unit, or with its centre of mass inside the box that encloses the vehicle with all
  its units at home, taken in the vehicle's axes and grown by 30 mm on every side. Fastened units turn with the
  vehicle and stay fastened: measured over 20 turns, a fastened unit moved by less than 0.1 mm on the vehicle.

## Weighing

The positioner's base stands on three load cells. The base is an open frame: the drive stand and two outrigger rails
beside the vehicle. No weighed structure lies under the vehicle. `positioner_and_cells.png` gives the three cells'
stations; it shows the vehicle at `deck_up`.

* **Readings.** `get_state()["fixture"]["load_N"]` holds the three cells' readings in newtons in the order A, B, C: A
  under the drive stand, B at the end of the outrigger rail on the cell's +y side, C at the end of the rail on the
  cell's -y side. Compression is positive. The load cells are bolted to the bench. They do not turn with the vehicle,
  and the cell does not convert their readings into the vehicle's frame. Every 10 ms tick brings a new sample (100
  Hz).
* **Tare.** The empty positioner is tared at both index positions. At rest at an index position the three readings add
  up to the weight of what the fork carries (the vehicle with what is fastened to it and what lies on it), and they
  are split as its centre of mass lies between the three cells.
* **Noise.** Every sample of every cell carries white noise of 0.4 N rms. Each cell also has a zero offset of up to
  +-0.02 N that stays the same through a run and differs between runs and cells. Measured with the base vehicle on the
  fork: the mean of 1000 samples (10 s) gives the mass within about 16 g and the centre of mass within about 0.63 mm
  along the vehicle and 0.54 mm across it (5 standard deviations, the zero offsets included). A reading without noise
  is within 0.01 N of the ideal three-point split.
* **When a reading is valid.** The readings are valid at an index position with the positioner at rest (`moving`
  false), while nothing but the vehicle and the units on it loads the fork, and from 0.5 s after the last outside
  force was taken away: the robot or a unit in its gripper touching the vehicle or the positioner, a unit being set
  down, the end of a turn. While the robot presses on the vehicle the cells show that force as it is.
* **Two physical checks that the cells allow.** (a) Weigh again after a ballast block is in place: the change of the
  three readings is that block's weight where it sits. This checks the hole's coordinates and port against starboard.
  A block that only lies in its place and a block that is fastened weigh the same. (b) A block that is not fastened
  when a turn starts is scrapped and taken away ("Turning procedure" above): after the turn the three readings add up
  to 1.89 N less than with that block. The positioner can be turned back and another block fitted, within the run's 4
  turns and the 12 blocks of the ballast box. Check (b) costs the block.

## The ballast box, the thruster tray, the racks

`~/given/drawings/trays_and_racks.png` dimensions the fixtures, each in its own frame. Each fixture below is placed by
the origin of that frame in the cell frame and by its turn about the vertical from the cell's axes.

* **Ballast box**: exact, the same in every run. Its frame's origin at (0.250, -0.400) m, turned by 0.0 deg. 12
  ballast blocks lie in its 4 x 3 pockets (pitch 70.0 mm along the box's x and 90.0 mm along its y, walls 3.5 mm
  high), each flat on the face that goes against the bottom plate, its long side along the pocket's long side. A block
  has 0.5 mm of play to the pocket's walls on every side; where inside that play it lies differs between blocks and
  runs.
* **Thruster tray**: the vertical thrusters (ducted thrusters, the kit's `thruster_a` and `thruster_b`) stand in its 4
  nests, one in each: nose cone down in the hole of the nest, the duct's lower rim on the three support blocks of the
  nest, 20.0 mm above the tray's top face, its axis upright within 0.5 deg. The key of a nest locates the thruster's
  mounting block in one of two directions half a turn apart. In every run it is drawn anew which thruster stands in
  which nest, which of the two ways each mounting block points and at which angle each propeller stands. The thrusters
  of the two kinds have the same housing and the same colours; they differ in the propeller only
  (`~/given/docs/propeller_datasheet.png`). The tray's top face is flat and parallel to the bench top. Nominal place:
  its frame's origin at (0.22, 0.48) m, turned by 0 deg. In every run the tray lies within +-40 mm of that place in x
  and in y and is turned by up to +-10 deg about the vertical.
* **Enclosure rack**: two V-blocks carry the electronics enclosure by its tube, between its two flanges. The tube axis
  is level, 138.0 mm above the bench. In every run the enclosure lies turned about its tube axis by up to +-20 deg
  from the attitude in which it is mounted, and its dome points to one end of the rack or to the other. Along its tube
  axis it lies within +-1.0 mm of the place that `trays_and_racks.png` draws. Nominal place: the rack's origin at
  (-0.17, -0.58) m, turned by 0 deg. In every run the rack stands within +-20 mm of that place in x and in y and is
  turned by up to +-5 deg about the vertical.
* **Guard racks**: one rack for each side of the vehicle, the port guard's at nominally (0.72, 0.53) m, turned by 0
  deg, and the starboard guard's at (0.72, -0.53) m, turned by 0 deg. Each is a single shelf, its top face 250.0 mm
  above the bench. The guard lies flat on it with the face up that is up on the vehicle at `deck_up` and the same way
  round as on its side of the vehicle; its outer edge overhangs the shelf by 49.0 mm on the side away from the
  vehicle's centre line, with nothing under it. On its shelf a guard lies within +-1.0 mm and +-0.3 deg of the place
  that `trays_and_racks.png` draws. In every run each rack stands within +-20 mm of its place in x and in y and is
  turned by up to +-5 deg about the vertical.
* **Identical units** are interchangeable: any ballast block fits any site, either guard fits either side (the two
  guards are the same part, one turned over against the other), and any vertical thruster can be fastened on any of
  the four places. Whether a thruster counts there is a matter of its propeller ("Fastening").

## Cameras

| name | type | resolution | nominal horizontal field of view | mount (approximate, not surveyed) |
|---|---|---|---|---|
| `top` | RGB-D | 1280 x 960 | about 70 deg | overhead, roughly 1.6 m above the bench near the middle of the working area, looking straight down; the image covers the bench with the positioner, the ballast box, the thruster tray, the racks and the robot's base; image width roughly along the cell's x |
| `wrist` | RGB-D | 640 x 480 | about 80 deg | on the gripper's base, about 6 cm off the tool axis on the side across from the pads (the tool's +x), tilted about 12 deg towards the fingertips; the two long fingers and a held unit are in the lower third of the image |

The cameras are off-the-shelf units: the lenses have radial and tangential distortion and are NOT calibrated; focal
length and principal point differ from the nominal values by a few per cent between units; the mounts are known only
by the descriptions above; the units are swapped and re-mounted between runs. The depth stream is registered to the
colour pixels (depth along the optical axis, 1 mm steps, range 0.1-4 m, no reading on depth edges and on about 3 % of
the pixels); its noise, new in every frame, is about 0.5 mm + 4 mm per m^2 x z^2 rms for the top camera and about 0.2
mm + 3 mm per m^2 x z^2 for the wrist camera. On top of it each camera unit has systematic errors that do not average
away: a smooth error field across the image of about 0.5 mm + 1.5 mm per m^2 x z^2 rms for the top camera and about
0.1 mm + 1.5 mm per m^2 x z^2 for the wrist camera, a ripple that varies with the range (up to 1 mm for the top
camera, 0.2 mm for the wrist camera) and fixed per-pixel offsets of about 0.4 mm and 0.1 mm rms. Closer than its 0.1 m
minimum range a camera gives no depth. The hall's brightness varies by up to about 20 % between runs, and each camera
unit has its own slight colour response (gain and tint within a few per cent), constant during a run; the image noise
is new in every frame. The images come from a physically based renderer of the hall (soft shadows, reflections on
metal and plastic); the given render layer (`~/given/lib/render`) draws scenes with a plain rasteriser instead (same
geometry and camera model, plainer shading, no hall).

* **Transparent parts give no depth.** The acrylic tube and the acrylic dome of the electronics enclosure return no
  depth reading: where one of them is the first surface on a pixel's ray, the depth image holds `depth_invalid_value`.
  The acrylic tube of the battery enclosure, a part of the base vehicle, returns none either. The end cap with its
  penetrators and its vent, the two flanges, the dome's retaining ring and the locking-cord handles return depth like
  every other part. What is inside the tube is seen through it in the colour image only.
* **Level datums.** The bench top (the plane z = 0) and the top face of the thruster tray are level. They are the
  cell's level datums: a camera frame that shows one of them shows which way is level.
* **The cameras do not show what is fastened.** The screws that the station drives (the clip screws of the enclosure,
  the screws of the thrusters' mounting blocks and of the guards, the screws of the ballast blocks) never appear in
  the camera images. A fastened unit looks like one that only lies in its place. The photographs of the manual show
  the finished vehicle with its screws.

## Fastening

The cell does not report any capture or fastening, and nothing that is fastened can be unfastened.

A unit is **captured** when it is inside the capture window of a place it can take (table below) and at rest there
relative to the vehicle (every point of the unit slower than 50 mm/s relative to the vehicle and the unit turning
slower than 30 deg/s relative to it, on 10 consecutive ticks; a tick is 10 ms, so for 0.1 s). The cell then holds the
unit on that place (it stays when it is let go) and pulls it home at up to 10 mm/s and 10 deg/s: the screws being
driven. The cell **fastens** the unit when it rests at home, within the lock tolerance, for 50 consecutive ticks,
moving slower than 5 mm/s and turning slower than 2 deg/s relative to the vehicle. The unit may still be held while
this happens. A captured unit that is moved out of the release window before it is fastened is released and is free
again. Fastened units move with the vehicle as one body.

Windows, from the cell's own table (`describe()["fastening"]` has the same numbers). A unit goes onto its place along
one direction: the electronics enclosure and the guards downwards (the vehicle's -z), a vertical thruster sideways
onto the outer face of its side plate (along the vehicle's y), a ballast block onto the underside of the bottom plate
along the plate's normal. "Across" is the offset perpendicular to that direction; "short of home" is the distance
still to go along it; "tilt" is the angle between the unit's axis and the place's; "turn" is the angle about that
direction from an orientation that the place takes. For the electronics enclosure the tilt includes its rotation about
its own tube axis, and its turn is its rotation about the vertical. A unit may be at most 1.0 mm past home (the
electronics enclosure 2.0 mm).

| place | unit | home | orientations | captured within: across / short of home / tilt / turn | at rest before it is captured | physical key | stays without being held | released beyond: across / short of home / tilt / turn | lock tolerance | weight |
|---|---|---|---|---|---|---|---|---|---|---|
| `ee` | the electronics enclosure | its two cradle clips on the two tube cradles (`enclosure_mounting.png`) | 2, 180 deg apart; one counts | 2.0 mm / 6 mm / 4 deg / 1.5 deg | 0.1 s | none | yes | 4.0 mm / 12 mm / 8 deg / 3.0 deg | 1.0 mm / 2.0 deg, 0.5 mm along | 3.0 |
| `vt_FR`, `vt_FL`, `vt_RR`, `vt_RL` (front or rear, starboard or port) | any vertical thruster | its mounting block against the outer face of the side plate (`propulsion_layout.png`) | 4, 90 deg apart; one counts, and only with the propeller of that place | 2.5 mm / 3 mm / 1.4 deg / 1.8 deg | 0.1 s | none | **no**: hold it still for at least 1.5 s | 5.0 mm / 6 mm / 2.8 deg / 3.6 deg | 1.0 mm / 2.0 deg, 0.5 mm along | 1.5 each |
| `guard_P`, `guard_S` (port, starboard) | a thruster guard | lying on the two lower guard brackets of its side, its inner edge against the side plate (`guard_and_brackets.png`) | one | 3.0 mm / 3 mm / 1.1 deg / 0.7 deg | 0.1 s | none | **no**: hold it still for at least 1.5 s | 6.0 mm / 6 mm / 2.2 deg / 1.4 deg | 1.0 mm / 2.0 deg, 0.5 mm along | 1.0 each |
| a ballast site: one at every hole that `keel_hole_map.png` marks as free | any ballast block | flat against the underside of the bottom plate, over its hole, its outline lying as `keel_hole_map.png` draws it | 2, 180 deg apart; all count | 3.0 mm / 3 mm / 4 deg / 6 deg | 0.1 s | none | yes, at `keel_up` | 6.0 mm / 6 mm / 8 deg / 12 deg | 1.0 mm / 2.0 deg, 0.5 mm along | none: counted through the trim |

* **Capture needs rest.** A unit that passes through its window while it moves is not captured: one that is let go
  above its window and slides or falls through it, or one that lands in it and tips on. It has to be at rest in the
  window for the 0.1 s named above.
* **No place has a physical key.** The screws, the threads and the screw holes of a place do not collide ("Collision
  model"), and no pin, pocket or key guides a unit into its window or keeps it there. Whether a unit is captured
  depends on its pose and on its being at rest, and on nothing else.
* **Units that do not stay by themselves.** A vertical thruster hangs on a vertical face, and a guard's centre of mass
  lies outboard of its two brackets: neither stays on its place without being held. Such a unit has to be held still
  inside its window for at least 1.5 s before it is let go. A unit that has been captured does not fall when it is let
  go: the cell holds it and goes on to fasten it. A unit that is let go before it is captured falls, tips off its
  brackets or is left hanging askew.
* **Orientations.** The electronics enclosure is captured and fastened either way round; only the way round that
  `enclosure_mounting.png` and the photographs show counts, and the other way round it is fastened for good and counts
  nothing. A vertical thruster is captured and fastened in any of 4 orientations 90 deg apart about the normal of the
  side plate; only the one that `propulsion_layout.png` shows counts. A guard is captured one way only: turned over,
  turned end for end or on the other side it does not come to lie in its window. A ballast block counts either way
  round.
* **Propellers.** A place for a vertical thruster counts only with a thruster whose propeller turns the way
  `propulsion_layout.png` gives for that place. A thruster of the other kind is captured and fastened there all the
  same, for good, and counts nothing.
* **Ballast sites.** The cell has a site at every hole that `keel_hole_map.png` marks as free. A hole that the map
  marks as BLOCKED has no site: a block put there is never captured. Fastened ballast blocks carry no weight of their
  own in the score; they count through the trim of the vehicle (see the task statement).
* **Order.** No place needs another one fastened first: the units can go on in any order that the vehicle and the
  positioner physically allow.
* **Total.** The weights of the places for the enclosure, the thrusters and the guards add up to 11.0; with the trim's
  4.5 the total is 15.5.

Measured in this cell: the unit held by the arm, brought onto its place at 10 mm/s, held still for 1.5 s and let go,
from start poses spread over the window.

| place | start poses inside 80 % of the window: fastened | at the window's corners: fastened | outside 120 % of the window: fastened | fastened after it came to rest in place | largest contact force on a unit sustained for 0.2 s |
|---|---|---|---|---|---|
| `ee` | 136 of 138 | 24 of 32 | 28 of 50 | 0.54-3.75 s | 46 N |
| a vertical thruster | 592 of 592 | 256 of 256 | 0 of 400 | 0.54-3.16 s | 145 N |
| a guard | 360 of 360 | 44 of 80 | 6 of 116 | 0.51-3.06 s | 163 N |
| a ballast block | 148 of 148 | 64 of 64 | 0 of 100 | 0.52-3.41 s | 78 N |

* **How to read the table.** The times count from the moment the arm had brought the unit to its start pose. A unit
  that the arm held within the lock tolerance of home was fastened during the hold. A unit held further off was
  captured, pulled home and fastened only after it was let go. A force above 150 N in the last column is a violation
  of the second rule of "Safety": it occurred where the arm pressed a guard down at a corner of its window, tilted and
  turned as far as the window lets it.
* **Let go above the place.** A vertical thruster let go 8 mm above its place slid down the side plate and through its
  window: 0 of 108 were captured, and all of them fell to the bench. A guard let go 8 mm above its place landed on its
  brackets and tipped off, or was left lying askew on the vehicle: 0 of 54 were captured. What decides is rest inside
  the window, not the way a unit comes there: a unit that an opening gripper sets down inside its window and that
  rests there for the 0.1 s is captured like any other.
* **A thruster beside its window with the guard already on.** Let go there, a thruster fell through the guard's
  cut-out to the bench, and none was captured. Only a thruster that the arm had pressed against the rim of the cut-out
  stayed hanging there, 13-31 deg off its home attitude; pressing it there loaded the rim beyond the limit of the
  second rule of "Safety".
* **The electronics enclosure turned about the vertical.** Brought down turned by up to 4.0 deg it was fastened in
  every trial: where the arm held it turned beyond its window, resting on the inner edges of the centre plates, it was
  not captured, but it turned home in its cradles when it was let go. Brought down turned by 6.0 deg, the ends of the
  enclosure came to rest on the inner edges of the centre plates and on the cradles, 14.5-22.6 mm above home. It was
  not captured there; let go, it stayed there or dropped askew onto the saddles and was not captured; pressed down
  there it loaded the plates with up to 108 N sustained.

## Collision model

The part files (`~/given/parts/*.stl`), the drawings and the photographs show the vehicle as it is drawn. The cell
simulates a second, simpler set of shapes: the drawn shapes with material taken away and, but for the exceptions named
below, none added. These shapes are not given as files. Everything in which they differ from the drawn shapes is
listed here. Coordinates are millimetres in the vehicle frame unless a unit's own file is named.

* **How the shapes are built.** Each loose unit and every part of the vehicle that a unit mounts on or passes collides
  as a few convex pieces built from the drawn sizes: flat faces as drawn, round shapes as prisms with 36 flat sides
  around a thruster's duct and 36 around the enclosure's tube. Both prisms are solid: nothing reaches into a duct or
  into the tube. The faces that units rest on (the underside of the bottom plate around every ballast hole, the outer
  faces of the side plates where the thrusters' mounting blocks and the guards' inner edges lie, the saddles of the
  tube cradles as rows of flat facets, the tops of the lower guard brackets) are flat faces 0.01 mm behind the drawn
  faces. Away from them the underside of the bottom plate and the outer faces of the side plates collide 1.0 mm behind
  the drawn faces. Holes of less than 500 square millimetres in the plates and the guards (screw holes, lightening
  holes) are filled. Small parts and fittings (a thruster's nose cone, outlet cone and mounting block, the enclosure's
  end cap, flanges, dome and retaining ring, the battery enclosure, the vehicle's own gripper, the velocity log and
  their mounts) collide as the convex hull of their drawn shape. A vertical thruster stands on 12 small round feet
  under its duct's lower rim, and a guard lies on 22 such feet on its two faces; the feet lie inside the drawn shapes
  and end at the drawn faces. No piece reaches more than 1.3 mm beyond the convex hull of the drawn part it stands
  for. Masses are those of `kit.json`.
* **Fasteners do not collide.** These are every screw and screw head on the vehicle, the screws of the cradle clips,
  of the thrusters' mounting blocks and of the guards, and the screw of a ballast block. What these parts locate on a
  real vehicle, the capture windows of "Fastening" locate here.
* **Cut-backs for clearance.** Where a unit that comes onto its place inside its capture window, turned and tilted as
  far as the window lets it, would run into a neighbour that the drawn model leaves no room for, the neighbour's
  collision shape is cut back. There are 13 such cuts, and this list is complete. A guard's cut-out collides as a
  round opening of radius 55.0 mm instead of the drawn 51.0 mm: the prism of a duct (outer radius 48.7 mm) has 6.3 mm
  around it there, where a duct as drawn has 2.4 mm. As drawn a cradle clip clears its cradle's end face by 0.00-0.60
  mm; as cut it clears it by at least 8.15 mm. As drawn a ballast block on an outer hole lies 2.78 mm from the side
  plate; as simulated nothing of the side plate is nearer to it than 3.3 mm.
  * the two cut-outs of the port guard: opened to the radius above;
  * the two cut-outs of the starboard guard: the same;
  * the front end face of the front tube cradle, where the front cradle clip comes down: cut back by up to
    8.51 mm;
  * the rear end face of the rear tube cradle, where the rear cradle clip comes down: cut back by no more than that;
  * the inner edge of the rear centre plate on the starboard side, beside the rear part of the enclosure: notched by
    up to 7.0 mm;
  * the inner edge of the rear centre plate on the port side: the same;
  * the inner edge of the front centre plate on the starboard side, beside the front part of the enclosure: notched by
    up to 6.0 mm;
  * the inner edge of the front centre plate on the port side: the same;
  * the duct rim of the rear horizontal thruster on the starboard side, beside the rear cradle clip: a corner up to
    3.0 mm deep is cut away;
  * the duct rim of the rear horizontal thruster on the port side: the same;
  * the inner face of the starboard side plate beside the aft outer ballast hole of that side, below the bottom plate:
    cut back by 3.5 mm over the length of a block;
  * the inner face of the starboard side plate beside the forward outer ballast hole of that side: the same;
  * the inner face of the port side plate beside the forward outer ballast hole of that side: the same.
* **The enclosure hangs in its clips.** As drawn, the enclosure's tube lies 1.49 mm above the saddles of the cradles:
  its two cradle clips carry it, screwed to the cradles' end faces. Those screws do not collide. An enclosure that is
  set down and let go therefore comes to rest on the saddles, 1.49 mm below home. It is inside its window there (the
  window reaches 2.0 mm past home), and when it is captured the cell lifts it home.
* **Blocked holes.** At the holes that `keel_hole_map.png` marks as BLOCKED a screw head stands 2.75 mm into the space
  of a ballast block. The screw head does not collide: a block can be laid there, but the cell has no site there and
  never fastens it.
* **The ballast block collides as 6 boxes in 4 layers, all inside its drawn shape.** They are 41.43 x 57.43 mm at
  their widest and 11.25 mm high together; the rounded corners of the drawn block are left out. As drawn the block is
  rounded towards its free face, where it measures 32.1 x 48.1 mm; the boxes follow that taper in steps.
* **Transparent parts collide.** The enclosure's tube and dome collide like every other part; only the depth cameras
  do not see them ("Cameras").
* **At rest** a unit lies on its collision pieces: a ballast block on the floor of its pocket, a thruster on the three
  support blocks of its nest, the enclosure in the two V-blocks, a guard on its shelf. The lowest point of the part
  file is then within 0.22 mm of the surface the unit rests on.

## Safety

* **Violations.** An episode in which one of these happens scores nothing. `describe()["safety"]["rules"]` words them
  as follows, with `robot_force_N` = 30 N, `part_force_N` = 150 N and `part_force_window_s` = 0.2 s. First: "a robot
  link other than the gripper's four pads pressing on the vehicle, the positioner, a fixture, the bench or a unit
  above robot_force_N (the normal forces of its contacts summed per (link, body it presses on))". Second: "a unit
  pressed on the environment or on another unit with a contact force above part_force_N for part_force_window_s (the
  magnitude of the vector sum of the contact forces per (unit, body it presses on), normal and tangential components
  together; the gripper's pads on a unit do not count; the force with which the cell holds, pulls or keeps a captured
  unit is not a contact force)". The two rules add forces up differently: the first adds the normal forces of all
  contacts of one link on one body; the second takes the resultant of all contact forces between one unit and one
  body, so a unit that only rests wedged between two faces reads its own weight.
* **Only contact forces count for the second rule.** The force with which the cell itself holds a captured unit, pulls
  it home or keeps it fastened is not a contact force and is not counted.
* **The dome must not be touched.** Anything that presses on the enclosure's dome with more than 20 N for 0.05 s
  scraps the electronics enclosure: its place counts nothing, and the episode goes on.
* **Scrapped units.** A loose unit is scrapped when its centre of mass moves faster than 3 m/s, when it falls off the
  bench, or when it is on the vehicle and not fastened at the start of a turn ("Turning procedure"). The cell releases
  it if it is captured; it can no longer be fastened anywhere and no longer counts. Only that unit is lost; the
  episode goes on.
* **Not scrapped.** A unit that was let go without being captured and hangs or lies on the vehicle is not scrapped at
  once. It counts nothing as it is, it can be picked up again, and it is scrapped at the start of the next turn like
  every unit that is loose on the vehicle then.
* **Turn interlock.** The positioner does not turn while a part of the robot is inside the turn column; the call is
  refused.

## Budgets

Per episode: 1200 s of simulated time, 2400 s of wall clock (camera rendering counts), 120 s of silence at most, 600
camera frames, at most 500 ticks per `step` call (a longer request is cut to that many), at most 4 turns of the
positioner. There are 4 episodes.
