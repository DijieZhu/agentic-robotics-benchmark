# DA-6 desktop cells: integration manual

Integrator's manual of the three desktop cells in which the lab's DA-6 desktop arm (six BUS-SERVO-12V-A servos) works: what stands in each cell, how an installation is described to the motion controller at run time, how accurately it is surveyed and what is clamped in the gripper. The work itself (the jobs) and its acceptance are specified in `../acceptance/acceptance_spec.md`; the arm in `../robot/`, the servo in `../servo/`, the unit's calibration record in `../unit/`, the controller interface in `../controller/`.

## 1. The cells

| cell | `kind` | work | objects (name, kind) | clamped in the gripper |
|---|---|---|---|---|
| shelf picking | `shelf` | positioning the gripper inside the compartments of a shelf organiser and above a fixture block | `shelf` (shelf), `fixture` (fixture) and one or two posts: `post` (post) and, when there are two, `post2` (post) | nothing (`{"kind": "gripper"}`) |
| bin placement | `bins` | setting a clamped part down inside open bins and on a fixture block | `bin1` (bin), `bin2` (bin), `fixture` (fixture) and one or two posts: `post` (post) and, when there are two, `post2` (post) | a part (`"part"`) |
| adhesive dispensing | `dispense` | dispensing paths over a tilted board; purge positions over a cup | `board` (board), `cup` (cup), `clamp` (fixture) | a dispensing needle in its holder (`"needle"`) |

The cells share one table and are worked one after the other (shelf, bins, dispense). They are rebuilt and re-surveyed for every batch: which objects stand where, their sizes and their orientation, and in the shelf and bin cells the number of posts (one or two), change from one installation to the next. The controller learns the installation it works in at run time, from the scene description passed at the start of each cell (section 6). The example scenes of this folder (section 11) show the format; they are not the installations the arm will meet.

## 2. Frames, units and conventions

The scene description gives lengths in metres and angles in degrees; the setup `tool` argument gives lengths in millimetres (its field names end in `_mm`) and masses in kilograms.

| frame | origin and axes |
|---|---|
| cell frame | the cell's reference corner on the table top; x and y in the table plane, z up; the table top is z = 0. Scene descriptions and jobs are given in this frame. The corner lies outside the installation: the robot base and every point of every object have x > 0 and y > 0 (all coordinates of a scene description are positive). The table top extends beyond the installation on every side. |
| robot base frame | frame 0 of the kinematics sheet (`../robot/kinematics_sheet.md`): origin on the mounting plane (the underside of the base) on the base-yaw axis; z up; +y forward (the direction the arm leans when the shoulder pitches negative at base yaw 0); x = y × z. The base stands on the table: its origin is at cell z = 0 and its z axis is the cell's z axis. |
| object frame | one per object: origin on the table top under the object's centre (z = 0 at the object's underside), z up. |
| primitive frame | one per primitive: origin at the primitive's centre; its axes give the primitive's orientation. |
| TCP frame | on the gripper (section 9): +z = the approach, +x = the jaw axis; the clamped tools are given in it. |

**Poses.** A pose `{"pos": [x, y, z], "yaw_deg": ψ}` places a frame in its parent frame: a point with coordinates p in the frame has the parent coordinates Rz(ψ) p + (x, y, z), with Rz(ψ) = [[cos ψ, -sin ψ, 0], [sin ψ, cos ψ, 0], [0, 0, 1]]. The robot base pose and the object poses are poses in the cell frame; the base's +y axis points along (-sin ψ, cos ψ, 0) in the cell frame.

**Quaternions.** A primitive's orientation in its object frame is a unit quaternion [w, x, y, z] (scalar first, w ≥ 0). With v = (x, y, z) and [v]× the cross-product matrix of v, its rotation matrix is R = I + 2w [v]× + 2 [v]×²; a point p of the primitive frame has the object coordinates pos + R p.

**Chain.** p_cell = T_object (T_primitive p) and p_base = T_base⁻¹ p_cell, with T_base the robot base pose. Directions (approach vectors, surface normals) transform with the rotations only.

## 3. Table, robot and objects

- The table top (z = 0) carries the robot and the objects. It is an obstacle for every part of the arm and of the clamped tools, like the objects; only the robot's base stands on it.
- The robot base stands on the table at the surveyed pose `robot_base` of the scene description.
- Every object stands on the table top and is rigid. It is described by solid primitives (boxes and cylinders); the primitives are the object's envelope, against which contact and clearance are judged (acceptance specification). Apart from the table, the objects of the scene description are the only things within reach of the arm.
- The objects stand around the robot, within the reach of the arm (section 10), apart from each other and clear of the robot's base. The shelf organiser and the dispensing board stand in front of the robot and face it (the organiser's open front, the board's tilted surface, turned towards the base axis); the other objects stand anywhere around the robot, in any orientation about z.

## 4. Object families

`<name>` is the object's name; every primitive name starts with it, and primitive names are unique in a scene. Boxes are given by their half sizes, cylinders by radius and half height (section 6).

| kind | primitives | description |
|---|---|---|
| `shelf` | `<name>_plinth`, `<name>_back`, `<name>_board0` to `<name>_board2`, `<name>_wall0` to `<name>_wall<N>` (boxes) | A shelf organiser on a plinth with two levels of compartments, open on the organiser's front (its -y side). `board0` is the floor of the lower level, `board1` the floor of the upper level, `board2` the top; `wall0` to `wall<N>` are the side walls and the dividers, from -x to +x (N + 1 walls for N compartments per level); `back` closes the +y side; `plinth` fills the space under `board0`. Boards, walls and back are 4 mm thick. A compartment is the space between two neighbouring walls and two neighbouring boards; the floor of a compartment of level k is `board<k>`. |
| `fixture` | `<name>` (box) | A fixture block standing on the table; its top face is a work surface. |
| `post` | `<name>` (cylinder) | A vertical post standing on the table (a light or camera stand). The shelf and bin cells have one or two posts: `post` and, when there are two, `post2`. |
| `bin` | `<name>_floor`, `<name>_wxn`, `<name>_wxp`, `<name>_wyn`, `<name>_wyp` (boxes) | An open bin: a 4 mm floor and four 4 mm walls (on the bin's -x, +x, -y and +y sides) of the bin's full height; open at the top. |
| `board` | `<name>`, `<name>_rail0`, `<name>_rail1` (boxes) | A dispensing board: a 6 mm plate (`<name>`) tilted about the object's x axis so that its upper surface faces the object's -y side and upwards, on two rails along the object's x axis that stand on the table under it (`<name>_rail0` under its lower front part, `<name>_rail1` under its upper back part). The working surface is the plate primitive's +z face; its outward normal is the plate primitive's +z axis. |
| `cup` | `<name>` (cylinder) | A purge cup, described by its outer envelope (a solid cylinder); purge positions are above its top face. |

The dispensing cell's clamp block (`clamp`) is of kind `fixture`.

## 5. The cells in detail

**Shelf picking** (`shelf`, `fixture`, `post` and, when there are two posts, `post2`). The organiser stands in front of the robot with its open front towards the base axis. Nothing is clamped in the gripper (`{"kind": "gripper"}`); the jobs position the bare gripper inside the compartments and above the fixture block, without gripping. Servo 6 (the jaw) follows the controller's goals.

**Bin placement** (`bin1`, `bin2`, `fixture`, `post` and, when there are two posts, `post2`). A part is clamped in the gripper before the cell starts and stays clamped for the whole cell (section 9.1); the jobs set it down inside the bins and on the fixture block's top face (set-down poses; the part is not released).

**Adhesive dispensing** (`board`, `cup`, `clamp`). The board stands in front of the robot with its tilted surface towards the base axis. The dispensing needle, in its holder, is clamped before the cell starts and stays clamped for the whole cell (section 9.2); the jobs are dispensing paths over the board's surface and purge positions over the cup.

## 6. The scene description (setup argument `cell`)

The cell computer passes the surveyed installation as one JSON object. Shape (`…` = more of the same):

```json
{
 "format": "scene-description/1",
 "frame": "cell (the table's reference corner): x, y >= 0 on the table, z up; metres, degrees",
 "robot_base": {"pos": [x, y, z], "yaw_deg": ψ},
 "objects": [
  {
   "name": "shelf",
   "kind": "shelf",
   "pose": {"pos": [x, y, z], "yaw_deg": ψ},
   "primitives": [
    {"name": "shelf_plinth", "type": "box", "size": [hx, hy, hz], "pos": [x, y, z], "quat": [w, x, y, z]},
    …
   ]
  },
  …
 ],
 "table": {"z": 0.0, "note": "the robot base stands on the table; the table top is z = 0"},
 "survey_tolerance": {"objects_mm": 2.0, "objects_deg": 1.0, "robot_base_mm": 1.0, "robot_base_deg": 0.3}
}
```

| field | type | meaning |
|---|---|---|
| `format` | string | `"scene-description/1"` |
| `frame` | string | the frame statement (section 2) |
| `robot_base` | pose | the surveyed pose of the robot base frame in the cell frame (`pos` in m, z = 0; `yaw_deg`) |
| `objects` | list | the objects of the cell |
| `objects[].name` | string | the object's name, unique in the scene; jobs refer to it (`exempt`) |
| `objects[].kind` | string | the object family: `bin`, `board`, `cup`, `fixture`, `post`, `shelf` |
| `objects[].pose` | pose | the surveyed pose of the object frame in the cell frame (`pos` in m, z = 0; `yaw_deg`) |
| `objects[].primitives` | list | the solid primitives that make up the object |
| `primitives[].name` | string | the primitive's name, unique in the scene; jobs may refer to it (`exempt`) |
| `primitives[].type` | string | `"box"` or `"cylinder"` |
| `primitives[].size` | [3] or [2], m | box: half sizes along the primitive's x, y, z; cylinder: radius and half height, the axis along the primitive's z |
| `primitives[].pos` | [3], m | the primitive's centre in the object frame |
| `primitives[].quat` | [4] | the primitive's orientation in the object frame, [w, x, y, z] (section 2) |
| `table` | object | `z`: the table top height in the cell frame (m); `note`: free text |
| `survey_tolerance` | object | the survey's tolerance (section 7): `objects_mm`, `objects_deg`, `robot_base_mm`, `robot_base_deg` |

Poses are given to 1 µm and 0.0001°, primitive sizes and positions to 0.01 mm, quaternions to six decimals. The primitive sizes and their placement in the object frame are the objects' drawing dimensions; the object poses and the robot base pose are survey results (section 7).

## 7. Survey tolerance

Every installation is surveyed before work starts. The scene description states the survey's tolerance (`survey_tolerance`):

| surveyed item | position, along each horizontal axis of its frame | rotation about z (about its own origin) |
|---|---|---|
| each object (its frame) | ±2 mm | ±1° |
| the robot base (its frame) | ±1 mm | ±0.3° |

Heights are not part of the survey error: the robot and every object stand on the table top (z = 0). Each object is rigid, so its error moves all its primitives together. The errors of different objects and of the robot base are independent. Job targets (acceptance specification) are given in the cell frame; the controller's own transformation of them into the robot base frame carries the base's survey error.

Derived worst cases (horizontal; both axes at their bound, plus the rotation's chord 2 d sin(θ/2)):

| distance d from the object's origin (m) | 0.05 | 0.10 | 0.15 | 0.20 |
|---|---|---|---|---|
| a point of the object may be off by up to (mm) | 3.70 | 4.57 | 5.45 | 6.32 |

| distance r from the base-yaw axis (m) | 0.10 | 0.20 | 0.30 | 0.40 | 0.46 |
|---|---|---|---|---|---|
| a cell point, expressed in the base frame via `robot_base`, may be off by up to (mm) | 1.94 | 2.46 | 2.99 | 3.51 | 3.82 |

## 8. What the controller receives at setup

`setup(unit, cell, tool)` is called once at the start of each cell, before its first job, with the arm at the home pose and the cell's tool clamped, and again whenever the controller had to be restarted during the cell (the arm then stands where the last job left it; acceptance specification):

- `unit`: the unit's serial number and calibration record (`../unit/calibration_record.md`);
- `cell`: the scene description (section 6);
- `tool`: what is clamped in the gripper for the whole cell:

| cell | `tool` |
|---|---|
| `shelf` | `{"kind": "gripper"}` |
| `bins` | `{"kind": "part", "half_mm": [12.1, 11.75, 12.9], "seat_mm": {"x": 4.26, "z": 5.16}, "mass_range_kg": [0.03, 0.15], "com_note": "the centre of mass is inside the part; not known"}` (values of an example part) |
| `dispense` | `{"kind": "needle", "length_mm": 45.0, "radius_mm": 1.2, "mass_kg": 0.02, "axis_offset_x_mm": -0.001, "holder_half_mm": [7.84, 6.0, 9.0], "holder_centre_z_mm": -3.0}` |

| field | of | meaning |
|---|---|---|
| `kind` | all | `"gripper"` (nothing clamped), `"part"` or `"needle"` |
| `half_mm` | part | half sizes [hx, hy, hz] of the part's box along the TCP's x, y and z, mm (section 9.1) |
| `seat_mm` | part | `{"x": sx, "z": sz}`: the part's centre in the TCP frame is (sx, 0, sz), mm; sx = the fixed pad face's x + hx (section 9.1) |
| `mass_range_kg` | part | the range of the part's mass, kg (the mass itself is not given) |
| `com_note` | part | a note on the part's centre of mass, which is not given |
| `length_mm` | needle | the needle tip's distance from the TCP along the TCP's +z, mm (section 9.2) |
| `radius_mm` | needle | the needle's radius, mm |
| `mass_kg` | needle | the mass of the needle and its holder together, kg |
| `axis_offset_x_mm` | needle | the needle axis's offset from the TCP along the TCP's x, mm: the needle tip is the point (`axis_offset_x_mm`, 0, `length_mm`) of the TCP frame |
| `holder_half_mm` | needle | half sizes of the holder block along the TCP's x, y and z, mm |
| `holder_centre_z_mm` | needle | the holder block's centre along the TCP's z, mm (it lies on the needle axis) |

## 9. The clamped tools

The TCP and its frame are defined in the kinematics sheet and the gripper drawing (`../robot/kinematics_sheet.md`, `../robot/drawings/gripper.png`): the TCP lies between the two finger pads, at the pads' height, 0.2 mm off the wrist-roll axis, in the middle of the gap when the pads are 15.68 mm apart (the needle holder's width, section 9.2). At the jaw's reference angle q6 = 0 the pads are parallel and 15.80 mm apart, and the middle of the gap lies 0.06 mm from the TCP towards the moving jaw. +z = the approach (out of the fingers, along the roll axis), +x = the jaw axis (from the fixed finger towards the moving jaw). In the TCP frame the fixed finger's pad face is the plane x = -7.84 mm; the moving jaw's pad face lies one pad gap (section 9.4) further along +x.

While a part or the needle is clamped, servo 6 holds the clamp: goal positions sent to servo 6 are ignored, and its present position is still reported on the bus. The clamped tool counts as part of the arm: its contact with the cell, the table or a link other than the wrist, the gripper and the jaw stops the arm (a protective stop, which ends the cell's jobs: acceptance specification, section 1.2), and the clearance is measured from it as from the links (acceptance specification, section 4.1).

### 9.1 Part (bin cell)

- A rectangular block clamped across the jaws: its -x face rests on the fixed finger's pad face (x = -7.84 mm in the TCP frame) and the moving jaw closes on its +x face. Part frame: origin at the block's centre, at (`seat_mm.x`, 0, `seat_mm.z`) in the TCP frame, with `seat_mm.x` = -7.84 mm + hx; axes parallel to the TCP frame.
- Envelope: the box with the half sizes `half_mm` = [hx, hy, hz] along the part frame's x, y and z: across the jaws from x = -7.84 mm to -7.84 mm + 2 hx, along y from -hy to +hy, along the TCP's +z from `seat_mm.z` - hz to `seat_mm.z` + hz (a negative value lies behind the TCP). Its far face, `seat_mm.z` + hz beyond the TCP, is the face that is set down.
- Jaw: servo 6 holds the jaw where the pad gap equals the part's width 2 hx (section 9.4); the jaw's envelope stands there.
- Mass: somewhere in `mass_range_kg`; the centre of mass is somewhere inside the part and is not given. The servos' load registers (`../servo/servo_datasheet.md`) show the load the arm carries.
- Tool point of the bin cell's jobs: the TCP.

Example (the part of `../acceptance/example_jobs_bins.json`, `half_mm` = [12.1, 11.75, 12.9], `seat_mm` = {"x": 4.26, "z": 5.16}): the part spans x = -7.84 to 16.36 mm, y = -11.75 to 11.75 mm and z = -7.74 to 18.06 mm in the TCP frame; it is set down with its face z = 18.06 mm; the jaw holds at q6 = 7.00° (pad gap 24.20 mm).

### 9.2 Dispensing needle (dispensing cell)

- A straight, blunt dispensing needle in a holder block, clamped between the pads.
- Holder: a block with the half sizes `holder_half_mm` (7.84 × 6 × 9 mm) along the TCP's x, y and z, its centre at (`axis_offset_x_mm`, 0, `holder_centre_z_mm`) = (-0.001, 0, -3) mm in the TCP frame. Across the jaws it fills the pad gap, from the fixed pad face x = -7.84 mm to x = 7.84 mm: servo 6 holds the jaw at q6 = -0.10°, where the pad gap equals the holder's width (section 9.4); along the TCP's z it extends from z = -12 mm to z = 6 mm.
- Needle: radius `radius_mm` (1.2 mm), on the line x = `axis_offset_x_mm`, y = 0, parallel to the TCP's +z, from the holder's top face (z = 6 mm) to its tip at z = `length_mm` (45 mm). The tip, the point (`axis_offset_x_mm`, 0, `length_mm`) of the TCP frame, is the tool point of every job of the dispensing cell; the dispensing gap is measured from it to the board's surface.
- Envelope: the holder block, and every point within `radius_mm` of the needle axis between z = 6 mm and z = `length_mm` - `radius_mm` (43.8 mm): a rounded end whose surface reaches the tip.
- Mass: `mass_kg` (0.02 kg) for the needle and its holder together, the centre of mass on the needle axis 8 mm beyond the TCP.

### 9.3 Bare gripper (shelf cell)

Nothing is clamped; the TCP is the tool point and servo 6 (the jaw) is the controller's. The shelf cell's jobs do not involve gripping; the jaw's angle is up to the controller (an open jaw widens the gripper's envelope).

### 9.4 The jaw's pad gap

The gap between the two pad faces across the jaws (at the TCP's height) against the jaw's design angle q6, from the gripper's design (the table of the kinematics sheet, `../robot/kinematics_sheet.md`, section 5.2); between rows it is linear in q6. At q6 = 0 (the zero pose) the pads are parallel, 15.80 mm apart. A clamped part holds the jaw at the q6 at which the gap equals the part's width 2 hx, the needle holder at the q6 of its width 15.68 mm (q6 = -0.10°).

| q6 (°) | q6 (rad) | pad gap (mm) |
|---|---|---|
| -5.12 | -0.0894 | 9.366 |
| -3.87 | -0.0675 | 10.992 |
| -2.61 | -0.0456 | 12.548 |
| -1.35 | -0.0236 | 14.130 |
| -0.10 | -0.0017 | 15.682 |
| 0.00 | 0.0000 | 15.800 (zero pose) |
| 1.16 | 0.0203 | 17.185 |
| 2.42 | 0.0422 | 18.690 |
| 3.68 | 0.0642 | 20.199 |
| 4.93 | 0.0861 | 21.712 |
| 6.19 | 0.1080 | 23.227 |
| 7.45 | 0.1300 | 24.746 |
| 8.70 | 0.1519 | 26.265 |
| 9.96 | 0.1739 | 27.786 |
| 11.22 | 0.1958 | 29.307 |
| 12.48 | 0.2177 | 30.827 |
| 13.73 | 0.2397 | 32.346 |
| 14.99 | 0.2616 | 33.863 |
| 16.25 | 0.2836 | 35.377 |
| 17.50 | 0.3055 | 36.888 |
| 18.76 | 0.3275 | 38.394 |
| 20.02 | 0.3494 | 39.896 |
| 21.28 | 0.3713 | 41.392 |
| 22.53 | 0.3933 | 42.881 |
| 23.79 | 0.4152 | 44.363 |
| 25.05 | 0.4372 | 45.838 |
| 26.30 | 0.4591 | 47.303 |
| 27.56 | 0.4811 | 48.760 |
| 28.82 | 0.5030 | 50.207 |
| 30.08 | 0.5249 | 51.643 |
| 31.33 | 0.5469 | 53.068 |
| 32.59 | 0.5688 | 54.480 |
| 33.85 | 0.5908 | 55.880 |
| 35.11 | 0.6127 | 57.687 |
| 36.36 | 0.6346 | 59.431 |
| 37.62 | 0.6566 | 61.516 |
| 38.88 | 0.6785 | 63.485 |
| 40.13 | 0.7005 | 65.427 |
| 41.39 | 0.7224 | 67.607 |
| 42.65 | 0.7444 | 69.769 |
| 43.91 | 0.7663 | 71.977 |
| 45.16 | 0.7882 | 74.267 |

## 10. Reach of the arm

For the layout of a cell: the TCP's reach at the design dimensions (`../robot/kinematics_sheet.md`), from the kinematics and the design-angle limits, with the arm clear of itself and of the table; the clearance to the cell's objects and the size of the gripper and its tool come on top. Radii are horizontal distances of the TCP from the base-yaw axis; the approach is parallel to the arm's vertical plane (the plane in which the shoulder, elbow and wrist pitch move) and points away from the base, at the stated angle below the horizontal; the wrist roll is free (the TCP lies up to 0.2 mm beside that plane, depending on the roll). The base yaw turns this plane about the base-yaw axis, from -105.3° to +107.1° (0° = the base's +y, positive towards -x).

| TCP height above the table (m) | radius (m), approach horizontal | radius (m), approach 45° below the horizontal | radius (m), approach straight down |
|---|---|---|---|
| 0.02 | not reachable | 0.317 to 0.417 | 0.198 to 0.283 |
| 0.05 | 0.343 to 0.447 | 0.314 to 0.415 | 0.186 to 0.256 |
| 0.10 | 0.361 to 0.462 | 0.297 to 0.405 | not reachable |
| 0.15 | 0.363 to 0.463 | 0.258 to 0.384 | not reachable |
| 0.20 | 0.350 to 0.455 | 0.176 to 0.348 | not reachable |
| 0.25 | 0.317 to 0.437 | not reachable | not reachable |

The TCP reaches at most 0.464 m from the base-yaw axis (the arm stretched horizontally, the TCP 0.132 m above the table). For a dispensing job add the needle (45 mm along the approach).

## 11. Example scenes

`example_scene_shelf.json`, `example_scene_bins.json` and `example_scene_dispense.json` are scene descriptions of three example installations, exactly as the controller receives them (only the line breaks differ). They show the format and typical layouts; the installations of a working day are surveyed on the day. Overview (distances from the base-yaw axis computed with the surveyed poses):

**`example_scene_shelf.json`** (shelf picking cell): robot base at x = 0.2582 m, y = 0.3596 m, yaw -95.54°.

| object | kind | x (m) | y (m) | yaw (°) | primitives | types | distance from the base axis (m) |
|---|---|---|---|---|---|---|---|
| `shelf` | `shelf` | 0.5320 | 0.1517 | -126.44 | 8 | box | 0.344 |
| `fixture` | `fixture` | 0.4673 | 0.4745 | 170.02 | 1 | box | 0.239 |
| `post` | `post` | 0.2774 | 0.1781 | -124.48 | 1 | cylinder | 0.182 |
| `post2` | `post` | 0.2590 | 0.5852 | 173.80 | 1 | cylinder | 0.226 |

**`example_scene_bins.json`** (bin placement cell): robot base at x = 0.3725 m, y = 0.3057 m, yaw 91.98°.

| object | kind | x (m) | y (m) | yaw (°) | primitives | types | distance from the base axis (m) |
|---|---|---|---|---|---|---|---|
| `bin1` | `bin` | 0.1364 | 0.4333 | -130.19 | 5 | box | 0.268 |
| `bin2` | `bin` | 0.2967 | 0.5598 | -52.21 | 5 | box | 0.265 |
| `post` | `post` | 0.3678 | 0.0861 | -27.23 | 1 | cylinder | 0.220 |
| `fixture` | `fixture` | 0.1708 | 0.2352 | 141.07 | 1 | box | 0.214 |

**`example_scene_dispense.json`** (adhesive dispensing cell): robot base at x = 0.2507 m, y = 0.2849 m, yaw -92.60°.

| object | kind | x (m) | y (m) | yaw (°) | primitives | types | distance from the base axis (m) |
|---|---|---|---|---|---|---|---|
| `board` | `board` | 0.5059 | 0.2850 | -90.52 | 3 | box | 0.255 |
| `cup` | `cup` | 0.2154 | 0.5486 | 176.05 | 1 | cylinder | 0.266 |
| `clamp` | `fixture` | 0.3301 | 0.4613 | 101.39 | 1 | box | 0.193 |
