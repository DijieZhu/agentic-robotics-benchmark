# The cleaning cell: the customer's studio apartment, one Unitree G1 humanoid, the rules

This is the integrator's description of the robot cell your program drives: the site survey, the robot and its
controllers, and the fixtures, tools and cleaning products as their data sheets describe them. Everything stated here
as exact is exact; everything else (where the litter, the dishes and the books lie, the grease spill, the crumbs, which
bin stands where, the robot's start pose, sensor noise) must be measured in each episode. The cell never reports the
pose or the state of anything but the robot itself. There is no practice cell: section 11 lists what you have to
prepare the program with.

## 1. The apartment (exact survey)

Room frame: origin at the south-west floor corner, **x east, y north, z up**, metres, radians, quaternions `w x y z`.
Boxes are given as `[x0, y0, z0, x1, y1, z1]`.

| item | geometry |
|---|---|
| floor, walls | floor z = 0 (oak parquet); inner wall faces at x = 0, x = 6.0, y = 0, y = 5.0; 2.6 m high, the ceiling at z = 2.6; door opening in the south wall for 4.80 <= x <= 5.70 (2.05 m high, onto a hallway); window in the east wall for 1.60 <= y <= 3.00, 0.90 <= z <= 2.10 |
| drawn only (no obstacles) | skirting, the window's frame and board and the plant on it, the curtains beside the window, pictures on the walls, an intercom, a light switch and sockets, two ceiling light panels, the rug in front of the couch (4.215 <= x <= 5.485, 3.39 <= y <= 4.91: it supports nothing, what is put on it rests on the floor) |
| couch (the customer's, fixed) | against the east wall, seat towards the west, within 5.09 <= x <= 5.95, 3.36 <= y <= 4.94; colliders: seat [5.09, 3.49, 0.16, 5.75, 4.81, 0.48], back [5.75, 3.49, 0.16, 5.95, 4.81, 0.94], arms [5.09, 3.36, 0.16, 5.95, 3.49, 0.72] and [5.09, 4.81, 0.16, 5.95, 4.94, 0.72], four legs 0.035 m square, 0.16 m high, at (5.12, 3.39), (5.12, 4.91), (5.93, 3.39), (5.93, 4.91) |
| plant (fixed) | north-west corner: pot radius 0.13 at (0.51, 4.51), 0 <= z <= 0.25; foliage radius 0.40 at (0.45, 4.55), 0.25 <= z <= 1.62 |
| kitchenette (fixed) | base cabinets along the west wall [0, 1.15, 0, 0.50, 3.35, 0.86] (toe kick: x <= 0.44 below z = 0.10), quartz top 0.04 thick to z = 0.90, overhanging to x = 0.52; wall cabinets [0, 1.15, 1.52, 0.32, 3.35, 2.22]; an induction hob flush in the top for 0.04 <= x <= 0.48, 1.32 <= y <= 1.90 |
| sink | an apron-front basin, open above: inside 0.12 <= x <= 0.50, 2.28 <= y <= 2.78, bottom z = 0.70; walls 12 mm; its front wall (the apron) [0.50, 2.268, 0.62, 0.515, 2.792, 0.90] |
| faucet | column radius 0.022 at (0.065, 2.66) from the top up to z = 1.06; a low spout arching east to its outlet at (0.30, 2.66, 1.00), from which the water falls straight down; a single lever on top, pivot (0.065, 2.66, 1.085), axis along -y (lifting its end turns it positive), lever 0.17 m long pointing east, radius 7.5 mm |
| soap pump | a squat bottle radius 0.040 on the top at (0.30, 2.99), 0.12 high; its pump head is a flat disc radius 0.040, top at z = 1.059 at rest, travelling 20 mm down against a spring |
| sponge tray | a steel tray on the top, 0.13 (x) by 0.16 (y), centre (0.37, 3.14), 15 mm rim |
| dish tray | a grey rubber tray on the top, 0.40 (x) by 0.32 (y), centre (0.27, 2.09), base 4 mm, rim 18 mm high |
| kitchen props (fixed obstacles) | a utensil crock [0.045, 3.215, 0.90, 0.155, 3.325, 1.05], a herb pot [0.05, 1.19, 0.90, 0.17, 1.31, 1.01], a cutting board leaning on the splashback near y = 1.60 (drawn only) |
| bins (fixed bodies) | two flip-lid bins, their backs to the west wall: bodies [0.05, yc - 0.16, 0, 0.33, yc + 0.16, 0.60] for yc = 0.36 (south slot) and yc = 0.80 (north slot), walls 10 mm, inside bottom at z = 0.04; one is the **blue recycling bin** (a white recycling badge on its front), the other the **grey general-waste bin**; which stands in which slot varies per episode |
| bin lids | a lid 12 mm thick covers each bin, hinged along its back edge (axis along -y through (0.06, yc, 0.602); opening turns it positive, up towards the wall); a tab (radius 8 mm, 0.12 m long, along y) along its front edge at x = 0.354 (closed) |
| tool rail | on the south wall, [0.82, 0, 0.97, 1.62, 0.025, 1.03]; two spring clips whose jaws' centres are at (1.00, 0.048, 1.00) (the broom's) and (1.42, 0.048, 1.00) (the dustpan's) |
| dining table (fixed) | top [2.25, 2.13, 0.72, 3.45, 2.77, 0.75] (centre (2.85, 2.45), 1.20 by 0.64), four legs 0.045 square inset 0.06, a steel apron 0.06 deep under the top inset 0.06 |
| dining chairs (fixed) | at the table's ends, seats 0.44 by 0.44 at z = 0.46 centred (2.10, 2.45) (facing east) and (3.60, 2.45) (facing west), backs 0.88 high on their far sides, legs 0.032 square |
| coffee table (fixed) | top [4.33, 3.75, 0.42, 4.83, 4.55, 0.45] on four legs 0.035 square inset 0.04, nothing else under the top |
| bookshelf (fixed) | against the north wall [1.05, 4.66, 0, 2.35, 5.0, 0.92]: sides and boards 18 mm; boards at z = 0.06, 0.36 and 0.64 (their tops 0.078, 0.378, 0.658), top at 0.92; a divider at x = 1.70 splits it into a west and an east bay; rows of books already stand in it; things stand on its top |
| the books' compartment | the top compartment of the west bay: on the board at z = 0.658, between the books already there (they end at x = 1.234, a bookend there) and the divider: 1.25 <= x <= 1.69 for a book's middle, 4.67 <= y <= 4.99, 0.244 m of clear height (up to the underside of the shelf's top at 0.902) |

`describe()` repeats these numbers (`room`).

## 2. The mess (what varies)

After the dinner: on the **dining table** two plates, two glasses, a drinks can, two crumpled paper napkins and two
books (lying on their back covers), a **grease spill** (a dark, glossy film, about 0.2-0.4 m across, between the place
settings; it may run under a plate or a book) and crumbs round the plates; on the **floor round the dining set**
(never under the table or a chair) crumbs in two or three patches and a few strays; on the **coffee table** a drinks
can, a plastic water bottle, a napkin and an apple core (in its west half). The sponge lies in its tray (damp, no
soap, clean); the broom and the dustpan hang in their clips; both bin lids are closed; the water is off. The robot
starts just inside the door (about (5.25, 0.75), facing north). Positions, facings, the spill's shape, the crumbs and
the bins' order are drawn per episode; the graded episodes are three fixed scenes drawn by these rules (the survey
photos of section 11 show two other scenes drawn the same way).

Sizes: cans radius 0.033, height 0.122 (30 g); water bottle radius 0.034, 0.205 high plus a cap (32 g); napkins lumps
of about 0.05 m (12 g); apple core about 0.08 by 0.045 (30 g); plates radius 0.130, 0.022 high (0.45 kg); glasses
radius 0.035 (0.033 at the foot), 0.115 high (0.22 kg); books 0.160 x 0.030 x 0.235 m (red, 0.45 kg) and 0.150 x 0.024
x 0.210 m (blue, 0.34 kg); sponge 0.110 x 0.072 x 0.034 (yellow foam on top, a green scouring pad 10 mm thick on the
bottom; 22 g); crumbs 7-14 mm.

## 3. The robot

One Unitree G1 humanoid (29 DoF with Dex3-1 hands). The model is MuJoCo Menagerie's `unitree_g1/g1_with_hands.xml`
(copy in `~/given/robot/unitree_g1/`, BSD-3-Clause) with the cell's modifications (those of the furniture-assembly cell
of this benchmark):

* **Locomotion controller.** The pelvis rides four base joints: `base_x`, `base_y` (room x / y), `base_yaw` and
  `base_z`, driven by force-limited servos (x / y 300 N, yaw 150 N m) that the robot's own locomotion controller
  commands; the legs are posed from the stand height and do not touch the floor (no balance falls are simulated). A
  robot obstructed by furniture stops; its controller does not wind up beyond 5 cm / 0.1 rad, but while it stays
  obstructed the base keeps pushing against the obstacle with up to its servo limit (300 N): held for 0.2 s, that is
  a safety violation (section 9). Cancel the goal (or stop the velocity) when the base stops short.
* **Legs in the camera images.** The cameras draw the legs stepping while the base travels or turns (drawn only: the
  physics keeps the posed legs; nothing above the hips changes in the images).
* **Stand height** (pelvis height above the floor) 0.48 - 0.80 m, nominal 0.793 m; lower means a squat (the knees go
  forward: about 0.21 m ahead of the pelvis at 0.70, 0.34 m at 0.48, at knee height).
* Every robot body is gravity-compensated; payloads are not.
* Joints: waist yaw / roll / pitch, 7 per arm (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw), 7 per hand;
  ranges as in the model (`describe()` lists them).
* **Collision:** body links are capsules (Menagerie's `g1_mjlab.xml` colliders); the robot collides with the furniture
  and the loose objects, never with itself or the floor. **The hands (the `wrist_yaw_link` and the finger links) carry
  no contact**: grasping is the cell's grip rule (section 4). **What a hand holds does not collide with the robot
  either** (it is part of the hand while held); it collides with everything else.
* **Grasp site** of each hand, `<side>_grasp`: in the `<side>_wrist_yaw_link` frame at (0.150, -0.046, 0) for the left
  hand and (0.150, +0.046, 0) for the right hand, with the link's axes: **x = approach** (along the fingers), **y = pinch
  axis**, z = x cross y.
* **Head camera** `head`: an RGB-D camera on `torso_link` at (0.0576, 0.0175, 0.4299), optical axis along the link's +x
  pitched down by 0.8308 rad; it moves with the waist.

## 4. Grip rule (hands)

Closing a hand (`"close"`) engages a grip on an object when, at that instant, the hand's grasp site lies at one of the
object's grasp features:

* **bars and cylinders** (cans, the bottle, glasses, the broom's and the dustpan's handles, a lid's tab, the faucet's
  lever): the grasp site within 15 mm of the surface (at most the radius + 15 mm from the axis), between the feature's
  ends (cans and glasses from 12 mm above their foot to 12 mm below their top, the bottle up to 0.16 m, the broom's
  handle from 0.18 m above its swivel to its top, the dustpan's handle from 4 cm above its foot to its top), the pinch
  axis within 35 deg of perpendicular to the axis;
* **slabs** (books; the sponge, which counts as a slab both across its thickness and across its width): the pinch axis
  within 35 deg of the slab's thickness (for the sponge: of its thickness or of its width), the grasp site within 15 mm
  of the slab and at most 45 mm inside one of its edges or faces (and at least 5 mm from the corners along it);
* **discs** (plates): the pinch axis within 35 deg of the plate's axis, the grasp site within 15 mm of its mid-plane and
  at most 45 mm inside its rim;
* **lumps** (napkins, the apple core, the soap pump's head): the grasp site within 12 mm of its bulk (napkins 30 mm,
  the apple core 30 mm, the pump's head 40 mm from its middle).

The fingers close on what the hand takes hold of and stop where they meet it or what it rests on (into the sponge or a
napkin they sink a few millimetres); a hand that closes on nothing closes fully. The fingers' joint angles
(`get_state()`) show the hand's shape. A loose object is then held rigidly at the relative pose of that instant until
the hand opens or the grip slips; a
fixture's moving part (a lid's tab, the faucet's lever, the pump's head) is held at that point only (it turns about its
hinge in the hand). A closed hand that found nothing holds nothing (open and close again). Grip limits: an object held
by one hand slips when the hand's load exceeds **30 N** or its torque about the grasp site exceeds **5 N m** (**12 N m**
when the hand holds it by a bar or cylinder feature: a power grasp); held by
both hands, when its net load per hand exceeds 30 N; a fixture's part held at a point slips beyond **60 N**; an excess
must last 50 ms. The hand's tactile reading (`get_state()`: `tactile_N`) is its share of the held object's load (0 when
it holds nothing).

## 5. Control (each 10 ms tick)

* Base: `base_velocity` [vx, vy, wz] in the robot's heading frame (x forward, y left; limits 0.5, 0.3 m/s, 0.8 rad/s;
  accelerations 0.8, 0.8 m/s^2, 1.5 rad/s^2), held until replaced; or `base_goal` [x, y, yaw] in the room: a straight-line
  approach (proportional, gain 1.2 /s, within the same limits, no obstacle avoidance) that stops within 5 mm / 0.01 rad.
  `stand_height` moves at 0.15 m/s.
* Waist: joint targets [yaw, roll, pitch], at most 0.8 rad/s.
* Arms: joint targets (`q`, 7 values), or a grasp-site pose target (`pose` [x, y, z, qw, qx, qy, qz]) or position target
  (`pos` [x, y, z]: the wrist's orientation left free) in the robot's **base frame** (origin on the floor under the
  pelvis, x along the heading, z up; the default) or in the **room frame**. A target is solved by the cell's
  damped-least-squares IK over the arm's 7 joints (base, waist and legs as they are): a target within 0.12 m / 0.6 rad of
  the arm's previous target keeps the arm's configuration, a farther one gets a seeded global solve. `path: "line"`
  (default) moves the grasp site on a straight line with a minimum-jerk timing (about 0.25 m/s and 1 rad/s on average,
  peak accelerations at most 1 m/s^2 and 4 rad/s^2); `path: "joint"` moves in joint space to the target's IK solution.
  Joint targets move at most 1.5 rad/s with 6 rad/s^2. An unreachable target leaves the arm where the IK gets closest
  (`get_state()`: `ik_residual` [position m, rotation rad]). A target in the base frame travels with the base.
* Hands: `"open"` or `"close"` (closing applies the grip rule once).

## 6. Sensors

* `get_state()`: the localised base pose [x, y, yaw] (independent noise of about 3 mm and 0.2 deg per reading), the
  stand height, whether the base is moving, the motor state of the waist, the arms and the hands (`q`, `dq`,
  `tau_est`, below), per hand `closed`, `tactile_N`, the grasp site's pose in the robot's base frame (exact kinematics)
  and the IK residual of a pose target.
* **Motor state** (`q`, `dq`, `tau_est`: one {joint name: value} map each), as the G1's low-level state reports it
  (the body's `LowState` and the Dex3-1 hands' `HandState`, one entry per motor), for every motor a program can read:
  the waist's 3, the arms' 2 x 7 and the hands' 2 x 7 (`describe()`: `robot.motor_state` lists them with their torque
  limits). The legs belong to the locomotion controller (section 5) and are not reported.

  | field | unit | what it is | accuracy |
  |---|---|---|---|
  | `q` | rad | the joint angle, from the joint's encoder | exact |
  | `dq` | rad/s | the joint velocity | reading-to-reading noise of about 0.01 rad/s (1 sigma), quantised to 0.001 rad/s |
  | `tau_est` | N m (the joint's sign convention) | the joint torque estimated from the motor's current | a calibration offset of up to a few percent of the joint's torque limit (1.5 %, 1 sigma) and a scale error of up to 5 %, both constant through an episode (read them at rest), plus reading-to-reading noise of about 0.5 % of the limit, quantised to 0.01 N m |

  `tau_est` is what the drive delivers: the torque holding the robot's own links up (the gravity feed-forward), the
  payload, the arm's acceleration, and the load of anything the arm or the torso presses on or is pressed by (a contact
  far from the joint's axis shows; a push along a link straight through a joint does not). A hand's fingers show their
  drives' own effort only: holding is the grip rule (section 4), so what a hand holds shows in its `tactile_N` and in
  the arm's and the waist's torques, not in its fingers'.
* Cameras (`get_camera`): `ceiling_kitchen`, `ceiling_dining` and `ceiling_living`, fixed RGB-D cameras 960 x 720
  (vertical field of view 72 deg) whose exact poses and intrinsics `describe()` gives; `head`, 640 x 480 (58 deg), whose
  pose in the room at capture time comes with each frame (composed from the localised base pose and the exact
  kinematics). Pinhole, no distortion: fx = fy = (H / 2) / tan(fovy / 2), principal point at the image centre. Depth is
  along the optical axis, registered to the colour pixels, 1 mm quantisation, noise sigma 1.5 mm + 2.5 mm/m^2 z^2, invalid
  (0) below 0.28 m (head) / 0.40 m (ceiling), beyond 6 m, on depth discontinuities and on 2 % of the pixels at random.
  Colour is rendered physically based (MuJoCo's Filament renderer): the morning sun through the east window, the two
  ceiling panels, a light under the wall cabinets, the pendant over the dining table, the light the room reflects (warm
  indoor light: white surfaces look a little warm); a per-episode white balance / brightness
  draw and pixel noise. The grease film, the crumbs, the water stream, the soap on the sponge and the sponge's colour
  (it darkens as it takes up grease) are drawn as they are. **At most 1500 frames per episode** over all cameras.

## 7. Fixtures and tools

* **Bin lids.** A lid turns 0 (closed) to 1.431 rad (82 deg, against its stop). Above 0.96 rad (55 deg) a hold-open
  spring (0.35 N m) holds it open; below, its weight closes it (a lid let go below 55 deg falls shut). Something falling
  onto a closed lid stays on it; through an open lid (above 35 deg) it falls in.
* **Faucet.** The lever turns 0 to 0.62 rad; the water runs while the lever is above 0.15 rad (full flow from 0.40);
  the lever stays where it is left (friction). The water falls straight down from the outlet to the basin's bottom.
* **Soap pump.** A stroke is the head pushed down at least 12 mm (of its 20 mm) and let back up above 4 mm; a stroke
  made while the **sponge presses on the head** puts one dose of soap into the sponge (at most 2 doses held); any
  other stroke wastes its dose.
* **Clips.** A tool whose handle is let go of with its axis within 12 mm of its clip's jaw centre, and the handle within
  12 deg of vertical (the jaw centre between 5 cm above the handle feature's lower end and 2 cm below its upper end),
  clicks in and hangs there; a hand that grips a clipped tool takes it out of the clip.
* **Broom.** A handle (aluminium, radius 11.5 mm, with a black grip at its top), its frame's origin on the head's swivel
  axis, the handle from 0.10 m to 1.20 m along its axis; the head (red block 0.30 x 0.055 x 0.035 with black bristles
  0.29 x 0.045 x 0.045 below it, their bottom 0.08 m below the swivel axis) swivels about its long axis within +-50 deg,
  so it stays flat on the floor while the handle leans along the stroke; 0.52 kg.
* **Lobby dustpan.** A grey pan 0.30 wide (x) and 0.24 deep (y, from the lip to the back wall), back wall 0.09 high,
  side walls lower towards the lip; a black rubber lip along its front edge; its frame's origin at the middle of the
  lip, on the floor when the pan rests flat; the handle (radius 12 mm, 0.80 long) rises from the back wall's top,
  leaning back by 8 deg, so the dustpan stands on its own on the floor; 0.45 kg.

## 8. The cleaning (what the products and the tools do; never reported)

The cell simulates the job's cleaning the way the products' and the tools' data sheets describe them; their states are
never reported, only drawn in the camera images.

* **Grease.** The spill is a film of cooking grease on the table's top, drawn darker and glossier where more of it is
  left. The sponge takes it off only while it is pressed flat on the top and sliding over it: firmer pressure (it
  starts to work at a couple of newtons; beyond about 25 N pressing harder adds nothing) and more strokes take more
  off. The green scouring pad face works several times better than the yellow foam; soap in the sponge makes it work
  several times better than water alone, and a dry sponge hardly works. Soap is used up as grease comes off. The
  grease that comes off loads the sponge: a loaded sponge takes less, and one near its capacity smears grease back
  onto the table where it passes. A whole spill holds more grease than the sponge does: rinse it (and soap it again)
  on the way.
* **Rinsing.** Held in the running water (in the stream below the faucet's outlet), the sponge sheds its grease within
  a second or two and its soap within a few seconds, and is wet.
* **Crumbs** rest on the highest surface below them and move only when pushed. The broom's bristles, flat on the floor
  and moving, push the floor crumbs under them ahead of them (swept fast, at about a metre per second or more, they
  scatter instead); the sponge pressed on a surface does the same to the crumbs on that surface, and a crumb pushed
  over an edge flies on a little the way it was pushed and falls. A crumb stops at whatever stands on the floor (the
  walls, the toe kick, the bins, the legs of the tables, chairs and couch, the plant's pot, the bookshelf). The dustpan
  resting flat on the floor (its lip down on it) takes in the floor crumbs pushed over its lip into the pan (a pan set
  down on crumbs only covers them); they stay in the pan as it is carried, unless it is tipped steeply (past about
  55 deg), when they pour out over its lowest edge (normally the lip) and fall. A crumb that falls into an open bin
  stays there; onto a closed lid it stays on the lid.
* **Dishes.** The plates (porcelain) and the glasses (soda-lime tumblers) are ordinary tableware: one survives a fall
  onto a hard surface of up to about 30 cm and a steady load of a few hundred newtons (about 600 N on a plate, 400 N
  on a glass); a harder fall or a heavier load breaks it. A broken dish earns nothing, and the customer charges for it
  (the instruction's scoring).

## 9. Safety (a violation makes the episode score 0; the episode goes on)

* a robot link other than the hands pressing on anything with more than 150 N for 0.2 s;
* a loose object let go of (or slipping) while moving faster than 1.5 m/s (thrown);
* a loose object not held, and not let go of in the last 2 s, moving faster than 4 m/s (struck).

## 10. Budgets per episode

4800 s of simulated time; 3600 s of program time (the episode's wall clock minus the cell's own time inside `step`,
simulating, and inside `get_camera`, rendering: your program's computation and all its other requests count); 120 s
from a reply to the next request (silence) ends the episode; 1500 camera frames; at most 500 ticks per `step` call.
The program's process (all its threads) may use at most 14,400 s of CPU time per episode (4 x the program-time
budget: the operating system ends it with SIGXCPU beyond that, and the episode is scored as it stands); its numerical
libraries run with OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS set to 2 (a thread pool of your own, e.g.
OpenCV's, counts against the CPU budget too: `cv2.setNumThreads` keeps it small).

## 11. Preparing the program (no practice cell)

The program is deployed on the customer's site as it is: nothing in your session runs the cell. You have:

* `describe.json` (this folder): exactly what the cell's `describe()` returns (the robot, joint names and ranges,
  control limits, grip limits, the cameras' intrinsics and the ceiling cameras' exact poses, the head camera's
  mount, the room's survey, the budgets, the safety limits);
* `survey/` (this folder): survey photos of two scenes of the same apartment, drawn by the rules of section 2 but not
  among the graded ones: per scene the three ceiling cameras and six head-camera frames from the robot standing at the
  start pose and at the dining table, the coffee table, the bins and the sink (`<camera>.json` holds the frame's
  metadata as `get_camera` returns it, `<camera>_rgb.png` and `<camera>_depth_mm.png` its colour and 16-bit depth;
  `house_client.client.load_frame(<camera>.json)` gives the reply that `decode_frame` reads);
* the robot's MuJoCo model (`~/given/robot/unitree_g1/`), from which the cell's robot is built (section 3), for your
  own kinematics, reach and collision checks.
