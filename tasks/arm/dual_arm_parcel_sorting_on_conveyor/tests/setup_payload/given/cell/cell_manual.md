# The label-facing station: the integrator's data sheet

This is the integrator's description of the station your program drives: the conveyor, the robot and its controller, its
grippers, the cameras, the parcels the line brings, how a shift is scored, the safety watch, the budgets and
commissioning. Everything stated here as exact is exact. The wire is described in `cell_api.md`, the public numbers are
collected in `cell_config.json`, and the model files of the robot and of the station are in `../robot/` and
`../station/`.

## 1. Frame and units

World frame: z up, the floor at z = 0. The robot's column stands at the origin and the robot faces +x. The belt runs
along y in front of it, and parcels travel towards +y (downstream). Units: metres, radians, seconds, newtons;
quaternions are written w, x, y, z. One control tick is 20 ms of simulated time (10 physics steps of 2 ms).

## 2. The station

A straight belt conveyor passes in front of the robot. Its parts, all rigid and fixed except the belt itself:

* **Belt surface.** Top at z = 0.9 m, centre line at x = 0.38 m, 0.44 m wide (x from 0.16 to 0.6 m). The belt runs on
  beyond both ends of the frame.
* **Frame (shell).** From y = -1.3 to +1.3 m (2.6 m long), its top 2 mm below the belt surface, 10 mm wider than the
  belt on each side, 0.1 m deep.
* **Legs.** 3 legs under the frame at y = -1.15, 0, 1.15 m, each two square posts (0.04 m wide, their centres 0.11 m
  apart across the belt) with a cap plate under the frame and a brace between them, up to 0.102 m below the belt
  surface.
* **Far rail.** Along the whole frame on the far edge (centre x = 0.622 m), its top 0.045 m above the belt surface.
* **Near rails.** On the near (robot) side outside the work window (centre x = 0.138 m), their tops 0.09 m above the
  belt surface.
* **Work window.** Along |y| < 0.42 m (0.84 m) the near side has no rail, so the arms can reach in. A low lip runs along
  the near edge through the whole work window (y from -0.42 to 0.42 m, centre x = 0.142 m), its top 30 mm above the belt
  surface.
* **Hardware near the arms.** The infeed camera's housing, its mounting plate and its drop rod above the upstream belt,
  the belt's drive unit under the belt near its downstream end with its motor on the far side, the head camera's housing
  on the robot's head and the further station hardware in the table below are solid too. The collision shapes are the
  drawn parts (boxes around them where a hand or a parcel can meet them).
* **The robot's base.** The robot's pedestal (a box 0.31 m in half size around the column, 0.295 m tall) stands on the
  floor at the origin; the belt's near edge is at x = 0.16 m.

The collision shapes the cell uses (boxes; centre and half sizes in the world frame):

| part (model name) | centre (m) | half sizes (m) | contact class |
|---|---|---|---|
| belt surface (moving) | [0.38, y, 0.88] | [0.22, runs on, 0.02] | the moving belt (contact with it: the belt rule) |
| far rail `belt_rail_far` | [0.622, 0, 0.895] | [0.012, 1.3, 0.05] | station structure |
| near-side guide, upstream of the window `belt_rail_near_up` | [0.138, -0.86, 0.92] | [0.012, 0.44, 0.07] | station structure |
| near rail, downstream of the window `belt_rail_near_down` | [0.138, 0.86, 0.92] | [0.012, 0.44, 0.07] | station structure |
| near lip, upstream half of the window `belt_lip_near_up` | [0.142, -0.21, 0.904] | [0.008, 0.21, 0.025] | station structure |
| near lip, downstream half of the window `belt_lip_near_down` | [0.142, 0.21, 0.904] | [0.008, 0.21, 0.025] | station structure |
| frame (shell) `belt_shell_col` | [0.38, 0, 0.848] | [0.23, 1.3, 0.05] | station structure |
| infeed camera housing `infeed_camera_body` | [0.38, -0.75, 1.782] | [0.045, 0.014, 0.032] | station structure |
| infeed camera mounting plate `infeed_camera_bracket` | [0.38, -0.738, 1.82] | [0.022, 0.022, 0.003] | station structure |
| belt drive motor guard (under the belt) `belt_drive_motor_cover` | [0.684, 0.975, 0.75] | [0.062, 0.085, 0.074] | station structure |
| belt drive motor terminal box `belt_drive_motor_terminal` | [0.674, 1.13, 0.825] | [0.04, 0.045, 0.022] | station structure |
| infeed camera drop rod `infeed_camera_rod` | [0.38, -0.715, 2.125] | [0.015, 0.015, 0.295] | station structure |
| belt drive unit (under the belt) `belt_drive_unit` | [0.38, 0.98, 0.762] | [0.217, 0.126, 0.079] | station structure |
| belt drive motor (under the belt) `belt_drive_motor` | [0.674, 1.18, 0.75] | [0.064, 0.12, 0.063] | station structure |
| head camera housing (on the robot's head) `head_camera_housing` | [0.078, 0, 1.304] | [0.087, 0.044, 0.043] | the robot's own head (arm to arm) |
| leg 1, near post `belt_leg_col_0` | [0.325, -1.15, 0.405] | [0.02, 0.02, 0.393] | station structure |
| leg 1, far post `belt_leg_col_0_far` | [0.435, -1.15, 0.405] | [0.02, 0.02, 0.393] | station structure |
| leg 1, cap plate `belt_leg_cap_0` | [0.38, -1.15, 0.795] | [0.085, 0.03, 0.003] | station structure |
| leg 1, brace `belt_leg_brace_0` | [0.38, -1.15, 0.2] | [0.035, 0.015, 0.01] | station structure |
| leg 2, near post `belt_leg_col_1` | [0.325, 0, 0.405] | [0.02, 0.02, 0.393] | station structure |
| leg 2, far post `belt_leg_col_1_far` | [0.435, 0, 0.405] | [0.02, 0.02, 0.393] | station structure |
| leg 2, cap plate `belt_leg_cap_1` | [0.38, 0, 0.795] | [0.085, 0.03, 0.003] | station structure |
| leg 2, brace `belt_leg_brace_1` | [0.38, 0, 0.2] | [0.035, 0.015, 0.01] | station structure |
| leg 3, near post `belt_leg_col_2` | [0.325, 1.15, 0.405] | [0.02, 0.02, 0.393] | station structure |
| leg 3, far post `belt_leg_col_2_far` | [0.435, 1.15, 0.405] | [0.02, 0.02, 0.393] | station structure |
| leg 3, cap plate `belt_leg_cap_2` | [0.38, 1.15, 0.795] | [0.085, 0.03, 0.003] | station structure |
| leg 3, brace `belt_leg_brace_2` | [0.38, 1.15, 0.2] | [0.035, 0.015, 0.01] | station structure |
| floor `floor` | [0, 0, -0.05] | [4, 4, 0.05] | station structure |

The lines across the belt, from upstream to downstream:

| line | y (m) | what happens there |
|---|---|---|
| set-down | -1.15 to -1 | each parcel of the list is set down on the belt's centre line here (section 7) |
| work window | -0.42 to +0.42 | the near side is open (above) |
| indexing zone | -0.15 to 0.1 | on an indexing line the belt stands while the parcel in the station has its centre here (section 3) |
| scan line | 0.95 | each parcel is judged as its centre crosses this line (section 8) |
| belt end | 1.32 | a parcel whose centre passes this line leaves the station |

A parcel that drops more than 0.35 m below the belt surface, or whose centre gets more than 2.2 m from the column along
x or along y, has left the line: it is removed and counts as fallen (section 8).

`../station/station.xml` holds this geometry as a MuJoCo model (the boxes above, the belt surface as a fixed box, the
`infeed` camera at the installation drawing's pose with the data sheet's field of view, and marker sites for the lines;
`station_with_robot.xml` adds the robot with its three cameras, section 6). It is geometry only: its friction, contact
and drive settings are not the cell's, and it holds no parcel.

## 3. The belt

* **Belt speed.** Constant through a shift and set per shift between 0.05 and 0.35 m/s. Your program does not command
  the belt, as on the real line. The belt's drive is stiff: parcels and the robot pressing on the belt do not change its
  speed measurably.
* **Running and indexing lines.** On a running line the belt runs through the whole shift. On an indexing line the belt
  stands while the centre of the parcel in the station (the one set down last and not yet settled, section 7) is between
  y = -0.15 and 0.1 m, and runs again, at the shift's speed, once that parcel's centre has left that zone (pushed on
  downstream, or taken back upstream); the drive checks it at the start of every tick. A parcel left in the zone keeps
  the belt standing, and the next parcel of the list does not come before it is settled. A shift's line is one or the
  other from its start to its end; which, is not on the wire. The drive brings the belt to a stand within about 0.2 s
  (at most 1.6 mm/s 0.2 s after the halt, at every belt speed); a parcel riding into the zone stops with it at most 10
  mm further on, its centre between y = -0.15 and -0.12 m; once its centre has left the zone the belt is back at the
  shift's speed within half a second (1555 halts on 40 indexing shifts replayed).
* **Grip of the belt.** A rubber top cover (friction coefficient between 0.75 and 1.15 against a parcel, by the data
  sheet): a parcel lying on the belt rides at belt speed; a parcel held back or blocked slips on the belt, and the belt
  keeps pulling it downstream.
* **No line signals.** The robot cell is not wired to the line's own sensors: nothing on the wire gives the belt's speed
  or its mode, or tells that a parcel has arrived.
* **Arrivals in the infeed camera's view.** A parcel is set down with its centre between y = -1.15 and -1 m (section 7).
  The top face of the tallest parcel (0.16 m above the belt) is inside the `infeed` camera's view from y = -1.16 to
  -0.34 m. The largest footprint at its longest along the belt (its diagonal, 0.397 m: a parcel may lie at any yaw) lies
  wholly inside the view while its centre is between y = -0.97 and -0.53 m: at the fastest belt (0.35 m/s) that lasts
  1.23 s. Some part of a parcel is in that view for about 2.5 s after its set-down at the fastest belt, longer at a
  slower one.

## 4. The robot

A fixed dual-arm platform: a column on a pedestal with a head on top (z up to 1.33 m), and two arms of 7 joints each,
their shoulders 0.096 m to either side of the column's axis at z = 1.217 m. The left arm is on the +y side (downstream),
the right arm on the -y side (upstream).

* **Joints.** 14 joints, always in this order (left arm, then right arm, each from the shoulder out). The ranges and the
  torque limits:

| index | joint | lower (rad) | upper (rad) | torque limit (N m) |
|---|---|---|---|---|
| 0 | `Left_Shoulder_Pitch_Joint` | -2.97 | 1.4 | 80 |
| 1 | `Left_Shoulder_Roll_Joint` | -0.35 | 3.49 | 80 |
| 2 | `Left_Shoulder_Yaw_Joint` | -2.79 | 2.79 | 36 |
| 3 | `Left_Elbow_Pitch_Joint` | -0.3 | 2.5 | 36 |
| 4 | `Left_Wrist_Yaw_Joint` | -2.79 | 2.79 | 36 |
| 5 | `Left_Wrist_Pitch_Joint` | -1.57 | 1.57 | 36 |
| 6 | `Left_Wrist_Roll_Joint` | -1.57 | 1.57 | 36 |
| 7 | `Right_Shoulder_Pitch_Joint` | -2.97 | 1.05 | 80 |
| 8 | `Right_Shoulder_Roll_Joint` | -3.49 | 1.2 | 80 |
| 9 | `Right_Shoulder_Yaw_Joint` | -2.79 | 2.79 | 36 |
| 10 | `Right_Elbow_Pitch_Joint` | -0.3 | 2.5 | 36 |
| 11 | `Right_Wrist_Yaw_Joint` | -2.79 | 2.79 | 36 |
| 12 | `Right_Wrist_Pitch_Joint` | -1.57 | 1.57 | 36 |
| 13 | `Right_Wrist_Roll_Joint` | -1.57 | 1.57 | 36 |

* **Mirror image.** The right arm is a mirror image of the left about the x-z plane, but not every joint axis is
  mirrored the same way: a mirror-symmetric posture has the same angle at `Elbow_Pitch` and the opposite angle at
  `Shoulder_Pitch`, `Shoulder_Roll`, `Shoulder_Yaw`, `Wrist_Yaw`, `Wrist_Pitch`, `Wrist_Roll` (the start posture is such
  a pair). The ranges are not symmetric (the table).
* **Commands.** Each tick the robot follows 14 joint position targets and two gripper commands. A target is held until
  you replace it. The cell limits how fast the applied target moves: at most 6 rad/s, that is 0.12 rad per tick for each
  joint, toward the commanded target. A target outside a joint's range is clipped to the range, never refused.
  `q_target` in the state is the applied target after that limit. The drives are position servos with velocity damping,
  limited to the torque limits: a small commanded step is followed smoothly and without overshoot, a moving target with
  a lag (the tool-pose level below gives it as measured), and under a steady load a joint settles off its target in
  proportion to the load (the servo's stiffness: the arm's compliance under load, as measured below). Their gains are
  not published; they are found in commissioning. `q` and `dq` are measured exactly; `tau` is the drives' torque
  reading, the motor torque with the sensor's noise: a zero-mean noise of 1.5 % of the joint's torque limit (one sigma,
  independent per joint and per tick) and a slowly wandering offset of at most 1 % of the limit. The drive's own data
  sheet allows 10 rad/s; the cell's command limit is the lower one above.
* **Tool-pose level.** Instead of an arm's joint targets, a `step` may give that arm a target pose of its tool centre
  point in the world frame (`tool_pose`: a position and a unit quaternion, `cell_api.md`); the two arms may use
  different levels in one step, one arm only one. The station's controller then computes that arm's joint targets
  itself, every tick: it solves the arm's inverse kinematics by damped least squares from the arm's present solution (at
  most 6 iterations per tick, each kept only when it brings the tool closer), taking the smallest joint change that
  reduces the pose error, so the posture stays near the present one and the redundant joint does not drift; it keeps
  every joint 0.01 rad inside its range; it scales the joint step to at most 95 % of the command speed limit per tick
  (the path's direction is kept and the limiter never has to cut it); and it adds a static gravity correction to the
  joint targets (the servos' sag under the arm's own weight), so that the measured tool pose, not only the computed one,
  settles at the target. A pose counts as reached when the computed solution is within 1 mm and 0.5 degrees of it. The
  solve weighs one radian of orientation error like 2 m of position error, so the orientation comes first: a target the
  arm cannot reach from its posture (beyond its reach, or beyond a joint's range) is approached in position as far as
  the arm goes with the tool's orientation kept close to the target's (the tool is not turned away to gain position);
  the arm then stays there, smoothly, without a stop, and the state says so: `tool_target.out_of_reach` turns true when
  a tick no longer brings the solution 0.2 % closer and then stays true (it latches) until the target changes, the
  target is reached, or the solution's position error falls more than 5 mm (or 5 %, whichever is larger) below its value
  when the flag was set; `error_m` and `error_deg` give the remaining error. Where it stops short, the hand may also be
  displaced from the target in any direction, sideways included, by several centimetres, and so may a hand holding a
  parcel between the two hands, into that parcel: measured without a parcel, both tool axes horizontal along the belt
  and pointing towards each other, the fingers' closing axes across the belt, the tool centre points 0.19-0.25 m apart
  over the belt at 0.04-0.07 m above it, then lifted 0.1 m, turned half a turn about the belt's direction and lowered 5
  cm, where neither arm reaches the turned orientation: both hands stopped 6.9-14.2 cm from their targets, 4.6-10.3 cm
  of it sideways and 1.8-7.5 cm of that towards the other hand (the two hands 6.6-10.7 cm closer together than
  commanded), with error_m 0.069-0.142 m and out_of_reach true, and no protective stop; `out_of_reach` and `error_m`
  report it at every step. The solve is local: it follows the present posture and does not search other postures. The
  controller avoids nothing: the tool moves towards the target whatever lies on the way (the safety watch still
  applies). A tool-pose target is held until replaced; joint targets for that arm, a protective stop or a lock-out end
  it (the arm then holds its posture). `q_target` in the state shows the joint targets the controller applied. Each
  tool's measured pose (forward kinematics of the measured joints) is `tool` in every state reply. As measured on this
  station: at rest the measured pose is within 0.04 mm and 0.14 degrees of a reached target; a 50 mm move settles within
  the tolerance about 0.4 to 1.0 s after the command (the computed solution gets there within the first few ticks; the
  rest is the servos following). Straight paths of 20 cm at 0.2 m/s, given as one target per tick, measured per region:
  with a hand over the belt's near half in the work window (the tool axis horizontal and along the belt), a path that
  stays in reach (along the belt towards the column, straight up) is followed within 2.2 mm of the line and 0.6 degrees
  of the orientation; near the start posture (the hands parked beside the column), a path that stays in reach (towards
  the belt, straight down) within 5.8 mm and 1.5 degrees; elsewhere a path soon meets a joint's bound or the arm's
  reach, and there the tool leaves the line (30 to 70 mm off it at the end of the measured paths that did) while its
  orientation is kept within about 2 degrees, up to 14 degrees where a wrist joint stands at its bound; the tool trails
  the commanded point by about 14 to 17 mm along the path (the servos' lag, about 75 to 80 ms at that speed); `error_m`
  and `error_deg` in every state give how far the solution is from the target. A target beyond the arm's reach is
  approached at no more than 0.114 rad per tick of any joint, with no protective stop and no cut by the speed limiter;
  the arm ends short in position with the tool's orientation within about 2 degrees of the target's (up to 14 degrees
  when a wrist joint is at its bound) and comes to rest; `out_of_reach` turns true about 0.12 s after the command (after
  5 ticks without progress) and then stays true, clearing only when the solution gains clearly on the target again (it
  did so once or twice in the measured cases, while the arm still slid along a joint's bound) or the target changes. The
  solve is local: it does not swing the arm into another configuration, so a pose that needs one (for example a large
  turn of the tool from the start posture) is reported out of reach from where the arm is. A load held in the gripper
  adds its own sag, which the correction does not know: about 2 mm and 0.3 degrees per kilogram with a hand over the
  belt in the work window (the tool axis horizontal and along the belt), under 1 mm per kilogram in the start posture.
* **Start posture.** Every shift starts with the arms at `describe()["robot"]["start_posture"]` = [-0.7842, 2.1067,
  0.7271, 1.946, -1.9143, -1.54, -1.1825, 0.7842, -2.1067, -0.7271, 1.946, 1.9143, 1.54, 1.1825]: both hands parked
  beside the column, palms down and tilted, the grippers open, nothing touching. Before t = 0 the cell lets the arms
  settle there for 0.3 s; the measured posture then differs from the commanded one by at most 5 mrad (the servos' sag
  under gravity). Every part of the robot is then behind the belt's near edge (its largest x is 0.156 m, 4.4 mm behind
  the edge at x = 0.16 m), so parcels ride past untouched. The tool centre points are then at (0.08, 0.22, 1.16) (left)
  and (0.08, -0.22, 1.16) (right).
* **Tool centre point.** The grasp centre between the finger pads, 0.1628 m from the wrist flange along the tool axis
  (the flange's z axis): `describe()["robot"]["T_flange_tcp"]`, with the flange sites `Left_flange` and `Right_flange`
  of the model.
* **Contacts.** The robot's links and grippers collide physically with each other, with the station and with the
  parcels; nothing passes through anything. The safety watch (section 9) watches the contacts with the station, between
  the two arms and with the belt.
* **Model files.** `../robot/` holds the robot with both grippers (`dual_arm_with_grippers.xml`, a MuJoCo model with the
  maker's data), the arm alone and the gripper alone, and their meshes. Exact in them: the kinematics, the collision
  shapes and the torque limits. Nominal: every link's mass and inertia and every joint's damping, friction loss and
  armature; the station's robot differs from these values by up to 10 % (masses and inertias) and up to 30 % (joint
  damping, friction loss and armature). Not in them: the servos' gains (each actuator there is a torque input at the
  drive's torque limit). They contain neither the station nor any parcel. Its `README.md` gives the frames and names.

## 5. The grippers

* **Gripper.** A 2F-85 class adaptive parallel gripper on each wrist flange (the MuJoCo Menagerie model of the Robotiq
  2F-85, BSD-2-Clause, licence in `../robot/gripper_2f85/LICENSE`): two linked fingers, a stroke of 0 to 85 mm, two
  high-friction pads on each finger (friction coefficient between 1.5 and 2.3 against a parcel, by the data sheet). A
  parcel is held by friction only.
* **Command.** `grip` = 0 (open) .. 1 (closed), continuous: the finger drive's target. From fully open to closed on
  nothing takes about 0.36 s, from closed to fully open about 0.52 s. The opening a command settles at, on nothing: 0:
  85.1 mm; 0.2: 69.8 mm; 0.4: 52.9 mm; 0.6: 34.9 mm; 0.8: 16.2 mm; 1: 0.4 mm.
* **State.** As the real gripper reports it: `gripper.command` (the applied command), `gripper.opening_m` (the measured
  distance between the inner faces of the pads: 0.0851 m fully open at rest (0.0852 m at the shift's start, before the
  fingers have come to rest), 0 m closed on nothing) and `gripper.object_detected` (true when the fingers have stopped
  short of the opening the command gives on nothing, by more than 3 mm, and stood still within 0.5 mm for 200 ms: they
  hold something, or something blocks them). There is no force reading.
* **Grip strength.** Closed with command 1 on a block 50 mm wide, the pads press on it with about 95 N in total (the sum
  of the normal forces on the block), and `object_detected` turns true about 140 ms after the fingers stop on it. A
  pinch of that kind holds the heaviest parcel of the population (2 kg) against gravity.

## 6. The cameras

Four cameras, rendered by the cell: `infeed`, `head`, `wrist_left`, `wrist_right`.

| name | mount | size (px) | vertical field (deg) | horizontal field (deg) | depth range (m) | fx = fy (px) | sees |
|---|---|---|---|---|---|---|---|
| `infeed` | fixed | 640 x 480 | 62 | 77.4 | 0.2-4 | 399.43 | arriving parcels' top faces, the entry |
| `head` | fixed | 640 x 400 | 66 | 92.2 | 0.2-4 | 307.97 | the work window |
| `wrist_left` | wrist | 424 x 240 | 58 | 88.8 | 0.07-0.6 | 216.49 | close views from the left hand |
| `wrist_right` | wrist | 424 x 240 | 58 | 88.8 | 0.07-0.6 | 216.49 | close views from the right hand |

* **`infeed`.** Fixed (by the drawing) at (0.38, -0.75, 1.75), 0.85 m above the belt surface over its centre line,
  looking straight down. At the belt surface it sees y from -1.261 to -0.239 m and x within about 0.68 m of the camera's
  axis: the set-down range and the parcels' top faces as they arrive (section 3). Its image's top edge faces upstream,
  its right edge faces the robot. `T_world_cam` = [[-1, 0, 0, 0.38], [0, 1, 0, -0.75], [0, 0, -1, 1.75], [0, 0, 0, 1]].
* **`head`.** Fixed on the robot's head (by the drawing) at (0.15, 0, 1.32), looking forward and 53.1 degrees down; its
  optical axis meets the belt surface at (0.46, 0, 0.9), 0.52 m away: the work window. Its image's right edge faces
  upstream. `T_world_cam` = [[0, -0.8, 0.6, 0.15], [-1, 0, 0, 0], [0, -0.6, -0.8, 1.32], [0, 0, 0, 1]].
* **`wrist_left`, `wrist_right`.** Palm cameras: each sits flush in its gripper base's palm between the finger roots,
  behind a small dark window, 98.3 mm ahead of the flange along the tool axis and 4.2 mm beside it, looking along the
  tool axis between the fingers (its optical axis is 0 degrees off the tool axis). Nothing of the camera stands above
  the palm and it has no collision shape of its own. The image's horizontal axis is the fingers' closing direction; the
  tool centre point lies 64.5 mm ahead of the camera, 14 px from the image centre. What the views show: with the fingers
  open, the two fingertips stand at the image's left and right edges at mid-height (about 12 % of the frame together)
  and the view between them looks straight along the tool axis; a parcel face 10 cm ahead fills about 83 % of the frame;
  a face at the fingertips (the palm against it) fills everything between the fingertips (about 89 %) and a waybill on
  it is legible. `T_flange_cam` = [[-1, 0, 0, 0], [0, -1, 0, 0.0042], [0, 0, 1, 0.0983], [0, 0, 0, 1]] (left) and [[1,
  0, 0, 0], [0, 1, 0, -0.0042], [0, 0, 1, 0.0983], [0, 0, 0, 1]] (right); a wrist frame carries its own `T_world_cam`:
  the drawing's mount composed with the flange's pose from the measured joints at the frame's exposure.
* **Calibration.** `describe()["cameras"]` publishes the data sheet's optics and the installation drawing's poses: fx =
  fy = (H/2) / tan(fovy/2), cx = W/2 - 0.5, cy = H/2 - 0.5, pixel (u, v) = (column, row) with row 0 at the top; OpenCV
  axes (x right, y down, z forward); no distortion. A pixel centre's ray in the camera frame is ((u - cx) / fx, (v - cy)
  / fy, 1). The installed cameras differ from the drawing: the `infeed` camera by up to 15 mm and 1.5 degrees in the
  world frame, the `head` camera by up to 10 mm and 1.5 degrees against the robot's base, each palm camera by up to 3 mm
  and 1 degree in its gripper's frame; each focal length is within 1 % of the data sheet's. The installed poses are to
  be calibrated on site; the images, depth included, are taken by the installed cameras.
* **Timing.** All four cameras expose together every 40 ms of simulated time (25 frames per second), at t = 0 and every
  0.04 s after it; they run during the settle before t = 0 too, so a frame is available from your first request (its
  stamp may lie before t = 0). A frame is delivered 60 ms after its exposure. `get_camera` returns the newest delivered
  frame, so a frame is 60-80 ms old when you receive it. Its `stamp` is its exposure time, and it shows the world
  exactly as it was then (a global shutter: no motion blur, no rolling shutter).

* **Frame budget.** Each new frame costs one frame of the shift's pooled budget (all four cameras together: section 10).
  Asking again for a frame already delivered (the same camera and the same exposure) is free and returns the same bytes,
  and adding depth to a frame already delivered is free too. A frame the cell fails to render answers
  `camera_unavailable` and is not charged. Rendering takes wall time but no program time and no simulated time.
* **Images.** Colour: an 8-bit RGB PNG with sensor noise of sigma 2 levels per channel, independent per pixel and per
  frame. Depth (on request): the planar distance along the optical axis as a 16-bit PNG in millimetres, 0 = no reading;
  noise of 0.5 % of the distance (one sigma), 1 % of the pixels without a reading, and no reading outside the camera's
  depth range (0.2-4 m for the fixed cameras, 0.07-0.6 m for the wrist cameras). Inside a camera's minimum range there
  is no reading either: a palm camera's depth gives no reading wherever the surface is nearer than 70 mm, which includes
  the fingers and a face pressed by the pads or against the palm (the tool centre point is 64.5 mm ahead of the camera):
  while a hand holds or pushes a parcel's face, its palm camera's depth image of that face is empty, and the colour
  image still shows it. Depth sees the same surfaces as the colour image: the station's surroundings near the cameras
  (the fence, posts, the gantry, the conveyor's frame members, the photo-eye) as well as the belt, the robot and the
  parcels.
* **The picture.** The colour images are drawn with the station's own light, distilled from its path-traced look: an
  image light of the hall around the work area and the task light's two bars over the belt as seven lamps: one at the
  bars' centre casts soft-edged shadows, six along the bars light without shadows. They show the station as it stands in
  a parcel hall: the belt's dark rubber cover between brushed steel rails, the cell's wire-mesh fence in yellow frames,
  yellow and black floor markings, the robot's pearl-white shell and dark grippers, and the hall beyond the fence.
  Cartons are kraft board (light, mid, dark or reddish brown) or white board (about a third of them), closed with tape
  strips over the flap seams (clear, brown or white film tape, or matt kraft paper tape, 38 to 72 mm wide); many carry
  stickers: red, orange or small pale ones with a few printed words or arrows, and plain paper stickers about the
  waybill's size with no print at all; mailers are white, grey or black film, matt or glossy; the waybill is a white
  printed sheet (below). Light falls mostly from above, so top faces are brighter than side faces and the palm cameras
  see darker faces when a hand shades them; in the overhead camera the parked arm casts a soft shadow on the floor
  beside the belt; the exposure keeps a white waybill distinguishable on the palest carton. There is no motion blur, no
  depth of field and no distortion; reflections are approximate.

**What each camera resolves.** The waybill (section 7) as it appears in each camera. Sizes from the published
calibration for a waybill facing the camera (short side x long side, in pixels), for a parcel on the belt's centre line
(where the line sets it down); its yaw turns the waybill in the image but does not change these sizes; a face seen at a
slant covers fewer. A label on a parcel's bottom face is in view of no camera while the parcel lies on the belt.

| camera | where the waybill is | distance (m) | paper 100 x 180 mm (px) | paper 76 x 130 mm (px) |
|---|---|---|---|---|
| `infeed` | top face of a parcel 0.16 m tall, below the camera | 0.69 | 58 x 104 | 44 x 75 |
| `infeed` | top face of a parcel 0.05 m tall, below the camera | 0.8 | 50 x 90 | 38 x 65 |
| `head` | top face of a parcel 0.16 m tall where the optical axis meets it (seen obliquely: the along-belt side shrinks with the view's slant) | 0.32 | 95 x 171 | 72 x 123 |
| `head` | top face of a parcel 0.05 m tall where the optical axis meets it (seen obliquely: the along-belt side shrinks with the view's slant) | 0.46 | 67 x 120 | 51 x 87 |
| `wrist_left`, `wrist_right` | a face straight ahead | 0.15 | 144 x 260 | 110 x 188 |
| `wrist_left`, `wrist_right` | a face straight ahead | 0.25 | 87 x 156 | 66 x 113 |
| `wrist_left`, `wrist_right` | a face straight ahead | 0.4 | 54 x 97 | 41 x 70 |

## 7. The parcels

* **Kinds.** Rigid parcels (cartons and boxes, flat ones included) and soft parcels (poly mailers: a filled bag,
  compressible, a compliant cover that yields under a squeeze; its firm core 75-95 % of its full height, the cover
  reaching the full height; its film grips rails, pads and other parcels somewhat more than a carton's board, by a
  friction coefficient about 0.05-0.25 higher). Each parcel of a list is rigid or soft with an equal chance,
  independently of the others.
* **Sizes.** Full sizes along the parcel's own axes x, y, z, drawn independently on a millimetre grid (x may be the
  shorter side): rigid x 0.16-0.3 m, y 0.14-0.26 m, z 0.07-0.16 m; soft x 0.16-0.3 m, y 0.14-0.26 m, z 0.05-0.1 m.
* **Mass.** 0.2-2 kg, independent of the size; the centre of mass off the centre of its box (its contents shift):
  horizontally by up to 0.04 m along each side (at most 0.8 of the half side), vertically by up to 0.3 of the half
  height.
* **Surface friction** (against the rails, the lip, the pads' rubber and other parcels): coefficient 0.3-1.5 for rigid
  parcels, a little higher for soft ones (above). Against the belt the belt's cover sets the grip (section 3).
* **Set-down.** The line sets each parcel down at rest, lying flat (its own z axis up), 2 mm above the belt, its centre
  on the belt's centre line (x = 0.38 m) between y = -1.15 and -1 m, at any yaw (every heading on the full circle
  equally likely). The belt carries it away within a fraction of a second. Riding free, a parcel keeps its set-down pose
  closely: of 6418 parcels of 160 running-line shifts replayed with the arms at the start posture, none was more than 7
  mm off the centre line or turned more than 11 degrees about the vertical when it reached the scan line (on an indexing
  line a soft parcel may turn by up to about 14 degrees while the belt stops and starts again under it).
* **The shift's list and its feed.** A shift brings a fixed list of parcels, one at a time. The first is set down at the
  shift's first tick; each next one at the first tick after the one before it is settled: its centre crossed the scan
  line going downstream (section 8), it fell off the belt (section 2), or it was damaged (section 9). So one parcel of
  the list is in the station at a time (one already settled may still ride on to the belt's end). If a parcel is still
  near the set-down line, the next one waits until that place is clear.
* **The station's rated cycle.** Each parcel is due at the scan line by its set-down time plus its free ride (from its
  set-down y to the scan line at the shift's belt speed) plus 5 s: the station may delay a parcel by at most 5 s against
  the belt's own ride, an indexing halt included; a parcel that crosses later is late and not readable (section 8). A
  shift's list is sized at 4 s per parcel: it holds as many parcels as fit into the shift's duration at the belt's ride
  over 2.025 m (from the middle of the set-down range to the scan line) plus 4 s, that is floor(duration / (2.025 m / v
  + 4 s)) parcels at belt speed v: 13 to 34 parcels at 0.05-0.15 m/s; 34 to 49 parcels at 0.15-0.25 m/s; 49 to 61
  parcels at 0.25-0.35 m/s. The list's length and the due times are not on the wire (they follow from the belt's speed,
  which your program measures).
* **The waybill.** Every parcel carries one shipping label (the waybill) on one face: any face whose two sides are both
  at least 80 mm, each such face equally likely (the top or the bottom when no face qualifies). The waybill is a paper
  of 100 x 180 mm or 76 x 130 mm (either with equal chance where both fit the face with a margin, else the one that
  fits: on a face 80-90 mm across, the smaller paper with an edge margin of 2 mm), in any of the four reading
  directions, a little skewed, its centre offset from the face's centre by up to 55 % of the free margin along each
  side. A soft parcel has no side wall: a waybill on one of its side faces wraps over the rounded flank onto its top or
  bottom film. The weight printed on it is not the parcel's. The label is **up** when its outward normal is within 25
  degrees of vertical up. What it looks like: a white thermal-paper sheet of 100 x 180 mm (or the same design scaled to
  76 x 130 mm), printed in black only. From its top edge: a header band (3-24 mm) with the carrier's emblem (a black
  disc with a white diamond and a black centre dot), its word mark KESTOVAR with its tagline and, at the right, a
  service box (solid black box, white two-character service code); the sender's block (3 lines); the recipient's block
  (4 lines); the routing band (82-118 mm) with the routing code in very large bold black characters, e.g. NV27-4C, cap
  height 13 mm, and a square QR code (QR version 1, error correction M, quiet zone 2 modules) of the tracking number
  (about 28.8 mm wide); the barcode band (118-160 mm) with a Code 128 linear barcode of the tracking number, 30 mm tall,
  the number printed below it; and a footer. Black rules separate the bands. No other mark on a parcel carries its
  print: stickers and stamps have no routing code, no barcode and no 2-D code, and plain paper stickers, some about the
  waybill's size and tone, carry no print at all.

## 8. A shift and its score

* **The shift.** From t = 0 to at most `describe()["shift"]["duration_s"]`: 600 s for every graded shift and every full
  development shift (300 s for a bench shift), whatever its belt speed and its parcels. It ends earlier, at the end of
  the tick in which the last parcel of its list is settled (section 7).
* **The station's duty.** The station turns every parcel: rigid parcels and soft parcels, whatever face carries the
  label. The duty is published in `describe()["score"]["duty"]`: the kinds (`rigid`, `soft`) and the label positions in
  duty (`underneath`, `downstream`, `upstream`, `near_side`, `far_side`; the positions are `top`, `underneath`,
  `downstream`, `upstream`, `near_side`, `far_side`, the world direction of the label face's normal as the parcel
  lands).
* **Its class (W or U).** Fixed as the parcel is set down, from its label's position then; nothing the robot does
  changes it:
  * **W** (work): a parcel whose label is not up (more than 25 degrees from vertical);
  * **U** (up): a parcel, rigid or soft, that arrives with its label up.
* **Its judgement.** In the tick in which its centre first crosses the scan line (y = 0.95 m) going downstream, a parcel
  is **readable** when all of these hold at the end of that tick:
  1. its label is up (outward normal within 25 degrees of vertical);
  2. it rests on the belt: it touched the belt surface at least once in the last 0.5 s (that tick included) and its
     lowest point is at most 10 mm above the belt surface, it turns slower than 2 rad/s, and it moves at less than 0.1
     m/s relative to the belt (horizontal velocity). Contact is judged over that short window, never from one tick: a
     soft parcel riding untouched rocks on its soft underside and loses and regains contact with the belt for a few
     ticks at a time;
  3. no part of the robot touches it;
  4. no other parcel touches it;
  5. it has not been damaged (section 9);
  6. it crossed in time: at the latest at its due time (section 7).

  Only the first downstream crossing counts; a later one changes nothing.
* **Done, lost, not done.** A parcel of the list is done when it was judged at the scan line, fell (section 2) or was
  damaged (section 9). It is lost when it fell or was damaged, at any time. A parcel that is not done when the shift
  ends (never set down, or in the station and not across, whatever ended the shift) costs nothing and earns nothing. A
  parcel that crossed readable and is lost afterwards (damaged, or knocked off before the belt's end) keeps its reading
  and still counts as lost: net 0 for a W parcel, -1 for a U parcel.
* **The shift's score.** With W and U counted over the whole list:

  ```
  credit      = (W parcels readable) - (U parcels done and not readable) - (parcels lost)
  shift score = max(0, credit) / (number of W parcels in the shift's list)
  ```

  A W parcel turned label up and crossing calmly in time earns 1; a late one earns nothing. A U parcel left alone costs
  nothing; a U parcel that crosses unreadable (knocked over, still touched, sliding, late) costs 1. A lost parcel costs
  1 more: knocked off the belt or damaged, a W parcel costs 1 and a U parcel 2. Every W parcel of the list counts in the
  divisor, reached or not. A program that leaves the belt alone scores 0.
* **How a shift ends.** Normally at the end of the tick in which the last parcel of its list is settled, or at its
  duration. It ends early when your program calls `finish()`, when `solve` returns or raises, when the program time is
  used up, a silence runs over its limit or the wire breaks (section 10), and when the cell itself fails (then the shift
  is run again, `cell_api.md`). Whatever ends it, the parcels already done keep their judgement and every other parcel
  is not done; there is no grace period at the end. The end's reason, as the cell, the wire and the commissioning
  service name it: `shift_end` (the list is settled, or the duration elapsed), `finished` (`finish()`, or `solve`
  returned or raised), `program_time` (the program-time budget is used up), `stalled` (a silence over its limit, the
  start window used up, or a reply not taken in time), `protocol` (a request line that is not one JSON request, or too
  long), `crashed` (the program's process ended while the shift ran), and the cell's own failures `physics_fault` and
  `internal_error`.
* **The graded shifts.** 8 shifts drawn from the line's distribution above, nothing selected or left out by their
  contents: six on a running line and two on an indexing line; two slow (0.05-0.15 m/s), three medium (0.15-0.25 m/s),
  three fast (0.25-0.35 m/s).

## 9. Safety and damage

The cell watches every contact of the robot at every tick (the normal force of each contact, per pair of bodies, at the
tick's last physics step), as a collaborative robot's collision detection does. Three classes:

| class | what the robot touches | protective stop above |
|---|---|---|
| structure | any part of the station that is not a parcel and not the moving belt surface: the frame, the legs, the rails, the lip, the floor | 60 N |
| arm to arm | one arm touching the other arm, or an arm touching the robot's own column, head or pedestal | 60 N |
| belt | the moving belt surface (a light touch below the limit is allowed) | 100 N |

* **Protective stop.** A contact above its class's limit for 3 consecutive ticks (60 ms) stops the robot: the joint
  targets freeze at the measured posture (clipped to the ranges), the grippers keep their command, and for 5 s of
  simulated time every `step` that carries a joint target or a gripper command is refused (`refused: "protective_stop"`;
  the world still advances, with the frozen targets). The belt runs on and the stop is counted. `stop` in the state
  gives `active`, `until` (the time the stop ends), `count`, `locked_out` and `link` (the robot link of the latest stop;
  when the two arms touch, the link of each arm joined by "/"; never what it touched). After the stop the robot holds
  the frozen posture until you command it again. A contact that was pressing when the stop came, or that comes during
  its 5 s, stops the robot again only once its force has dropped below the limit.
* **Lock-out.** Any of those contacts above 400 N for 3 consecutive ticks, or the sixth protective stop of a shift,
  locks the robot out for the rest of the shift: it holds its posture and every command is refused (`refused:
  "locked_out"`, `stop.locked_out` true). The belt runs on and the parcels are still judged as they cross.
* **Parcel damage.** When the robot presses on one parcel with more than 300 N in total (the sum of the normal forces of
  all its contacts with that parcel) for 3 consecutive ticks (60 ms), the parcel is damaged: it is lost (section 8).
  Nothing on the wire reports it, but the cameras show it: from that tick on, every frame draws the parcel with a
  crushed look. Its faces turn darker and greyer and dark irregular creases cross them (a rigid parcel's board), or its
  film is scuffed and creased (a soft parcel); the waybill on it is creased too. On a crushed parcel's faces the mean
  grey level drops and the share of edge pixels (Canny edges of the grey image) rises; measured on 20 parcels of both
  kinds, crushed and intact, a window on each one's top face: in the `infeed` camera the mean grey level of the crushed
  face is 0.62-0.78 of the intact one and its share of edge pixels 1.7-24 times as high; in the `head` camera 0.45-0.66
  and 1.5-20 times (an edge share below 1 % counted as 1 %). Its shape, its pose and its depth image do not change, and
  no text or mark is printed on it. An undamaged parcel never shows this look.

## 10. Time and budgets

The world advances only inside `step`: nothing moves and no time passes between your requests. Per shift (all in
`describe()["budgets"]`):

* **Program time.** The shift's duration + 60 s (660 s in a graded shift): the shift's wall clock minus the cell's own
  time inside `step` (simulating) and `get_camera` (rendering). It measures your program's own computing and its other
  requests. Its CPU time is capped at 4 x the program-time budget (all threads together). `program_time_left` in the
  state counts it down; when it runs out, the shift ends.
* **Silence.** At most 20 s of program time from a reply to your next request; a longer silence ends the shift.
* **Frames.** 2 x the shift's duration in seconds, pooled over the four cameras (1200 frames in a graded shift). When
  they are used up, `get_camera` answers `no_frames_left` and the shift goes on.
* **Ticks per call.** One `step` runs at most 50 ticks (1 s); a larger `n` is cut to that (the reply's `ticks` says how
  many ran).
* **After the end.** Once the shift is over, `step` answers with `ticks: 0` and nothing moves; the program may read its
  state and must return. The cell answers at most 50 more steps, 200 more requests and 60 s more, then the connection
  ends.
* **Start.** From the program's start to its first request at most 120 s (the interpreter's start and your imports; the
  start window), inside the program-time budget; after the first request the silence limit applies.
* **Replies and requests.** Your program must take in each reply within the silence limit (a camera reply with depth is
  up to about 1.1 MB); one request line holds at most 4 MiB and must be one JSON request (the client writes them so). A
  reply not taken in, a malformed or longer request line, or the program's process ending while the shift runs ends the
  shift.

## 11. The client library

`~/given/lib/parcelcell/client.py` is the one part of the cell your program imports: the proxy (`Cell`), the frame
decoders, the program launcher the grader uses and the commissioning connection (section 12). The cell itself (the
physics, the robot's servos, the cameras, the scoring) runs in the grader and in the commissioning service, never in
your account. The launcher moves the wire off the standard streams before your program starts: `print()` and anything a
library writes to the standard output go to the standard error, never into the wire.

## 12. Commissioning on development shifts

A robot program is not handed over before it has run on the real line: yours can be commissioned before it runs the
shifts it is graded on. The commissioning service in your sandbox runs the station's cell on 8 development shifts drawn
from the line's distribution above: 3 bench shifts with one to three parcels each (bench 1 (slow belt, running): one
rigid parcel with its label underneath; bench 2 (medium belt, running): one soft parcel with its label underneath and
one rigid parcel with its label on a side face; bench 3 (slow belt, indexing): a flat rigid parcel with its label not
up, one parcel arriving label up and one soft parcel with its label not up) and 5 full shifts (slow belt, running;
medium belt, running; fast belt, running; slow belt, indexing; fast belt, indexing). None of them is a graded shift.

* **What it is.** The cell of the graded shifts (the robot with its servos and limits, the grippers, the safety watch,
  the cameras' frames, the belt, the budgets) on the wire of `cell_api.md`, plus `reset(site)`, which starts development
  shift `site` (1 to 8). A development shift is the same each time you start it. The `reset` answers once the shift's
  cell is built: about a second: the development shifts' parcel looks are made when the image is built (a shift's cell
  built from them in 0.8-1.0 s in the task image); a shift whose looks are not made yet takes them first, several
  seconds a parcel.
* **What it is not.** A simulator you can read or query: it answers no ground truth (no parcel pose, size, kind or label
  face), no score and no debrief. How a shift went, you learn from what your program saw, logged and saved.
* **The window.** 12 commissioning shifts in your session (a shift counts from its `reset`), at most 2 at a time (while
  that many are open, a new connection's first request is answered `busy` and the connection is closed; no shift is
  used), each under the graded shift's budgets. When the service closes a shift for any reason but `finished` (section
  8), it sends `{"id": null, "event": "shift_over", "reason": ...}` and then closes the connection. The count is in
  `/run/parcelcell/commission_status.json`. It does not grow.
* **Time.** A full development shift lasts as long as a graded one (600 s of simulated time, more in wall time with your
  program's own time and the frames).
* **How.** `parcelcell.client.connect_commissioning(site)` returns a `Cell` whose shift has begun (one connection, one
  shift): a script of your own that moves the arms, grabs frames and records what it sees is a commissioning engineer's
  teach-in. `/opt/venv/bin/python3 ~/commission/run_shift.py --site N` runs your whole program in a shift the way the
  grader runs it (a read-only copy of the folder, the grader's environment, launcher, CPU cap and no-network filter) and
  prints how the shift ended, the frames used, the protective stops and the refused commands (`--keep DIR` keeps what
  your program wrote in its HOME and TMPDIR, `--transcript FILE` the wire).
