# Assembly cell: integrator's description (commissioning document)

## Frame, bench, robot

* **Cell frame**: origin on the bench top at the robot arm's base axis, z up, x towards the stand, y to the left seen
  from the arm. The bench top (2.0 x 1.4 m, x -0.55 .. 1.45 m, y -0.70 .. 0.70 m, oak veneer, 0.994 m above the hall
  floor, on a steel frame with a lower shelf) is z = 0. The cell stands in a factory hall, lit by the hall's ceiling
  lamps.
* **Robot**: a 6-DoF UR10e-class arm (model in `~/given/robot/universal_robots_ur10e/`), bolted to a surveyed datum:
  its base at (0.00, 0.00, 0.00) m, the base frame being the cell frame turned 180 deg about z (q = 0 on the first joint
  points the arm along the cell's -x; exact). Joint names `ur/shoulder_pan_joint`, `ur/shoulder_lift_joint`,
  `ur/elbow_joint`, `ur/wrist_1_joint`, `ur/wrist_2_joint`, `ur/wrist_3_joint`; the joint ranges (+-2 pi, the elbow
  +-pi) and the home posture are in `describe()`. Joint torque limits 330 / 330 / 150 / 56 / 56 / 56 N m.
* **Tool**: a 2F-85-class adaptive gripper (model in `~/given/robot/robotiq_2f85/`) on the tool flange (the flange
  face sits at the arm's d6 = 0.11655 m along the last link's axis; the given arm model's `attachment_site` is 16.6 mm
  short of it). The tool point `ur/tcp` is the pinch point between the pads, 0.1558 m along the tool axis from the
  flange face (NOT the gripper vendor's documented 174 mm tcp); its axes: z = the approach (the flange axis, out of the
  flange), y = the closing axis, x across the pads. The finger linkage is rigid (the gripper only pinches between its
  pads; it does not wrap around a unit); the pads are 22 mm across and 37.5 mm long, and along the approach their face
  centre lies about 14 mm short of the tcp fully open and 1-4 mm short for 10-45 mm grasps, the fingertips 5-18 mm
  beyond it, depending on the width; the finger linkage's outer envelope spans +-62-74 mm along the closing axis. Width
  0-85 mm (the gap between the pad faces at the pads' middle); the commanded width moves at 0.10 m/s. Grip force: see
  `describe()["robot"]["gripper"]["force_model"]` (about 6 N per mm of squeeze per pad, capped at about 80 N per pad).
* **Wrist camera housing**: a 36 x 24 x 35 mm box on the gripper's coupling, on the tool's +x side; it is part of the
  robot (its collision model and the safety rule). In the tcp frame its centre is at (0.056, 0.000, -0.137) m and its
  outer face at x = 0.068 m.
* **Force / torque sensor** at the tool flange's centre, in the flange frame (z along the tool axis outwards, x along
  the closing axis = the tcp's y, y along the tcp's -x): what the wrist applies to the tool, the gripper's own weight
  included (about 10.3 N with the tool pointing down); noise about 0.3 N / 0.02 N m rms per reading, an unknown bias up
  to about 1 N / 0.05 N m that differs between power-ups: tare it.

## Datum boards (the integrator's survey; exact)

Checkerboards printed on rigid plates. On a board of C x R squares of side s the inner corner (i, j), i = 1 .. C - 1,
j = 1 .. R - 1, sits at ((i - C / 2) s, (R / 2 - j) s) in the board's own frame (x along the C squares, y along the R
squares, z out of the printed face); the square at the board's (-x, +y) corner is black.

| board | squares | where (exact) |
|---|---|---|
| bench board | 9 x 6 of 20.0 mm (3 mm plate, 20 mm white margin) | flat on the bench near its +x / -y corner, printed face 3.05 mm above it; inner corners at x = 1.200 + (i - 4.5) x 0.020, y = -0.500 + (3 - j) x 0.020, z = 0.00305 m (cell frame; i = 1..8, j = 1..5); its black corner square at the board's -x, +y corner |
| stand board | 5 x 3 of 10.0 mm (2 mm plate, 4 mm margin) | vertical on the stand rail's face towards the robot arm (the cell's -x), centred on the rail, its centre 60 mm above the bench; with the stand at (sx, sy) (its nominal place +- 10 mm, below): inner corners at x = sx - 0.0226, y = sy + (i - 2.5) x 0.010, z = 0.060 - (1.5 - j) x 0.010 m (i = 1..4, j = 1..2); the square at its -y, bottom corner is black. It moves with the stand |
| tool plate | 5 x 4 of 6.0 mm (1.5 mm plate, 3 mm margin) | on the camera housing's outer face, facing the tool's +x; in the TCP frame inner corners at x = 0.0700, y = (j - 2) x 0.006, z = -0.1373 + (i - 2.5) x 0.006 m (i = 1..4, j = 1..3); its black corner square at the plate's -y, flange-side (-z) corner; mounting tolerance about 1 mm / 0.5 deg |

## The stand, the body and the units

* **Stand**: two forks on 40 x 40 mm aluminium posts joined by a rail 60 mm above the bench, at a nominal centre
  (0.62, 0.00) m, placed within +-10 mm, never turned; the forks stand at +-0.10 m along the cell's y. Each post stands
  on a 200 x 40 x 12 mm foot and carries under the body a 121 x 40 x 10 mm fork plate (both with their long side along
  the cell's x) with a 10 mm high tab either side of the body. The body is bolted on the stand, its centre 0.50 m above
  the bench, its long axis along the cell's y and its front towards +y (the left legs, FL and HL, towards the robot
  arm).
* **Body**: the body with its four hip ab/ad actuator modules (`~/given/parts/body.stl`); every module's output pulley
  carries the socket for a shoulder's key. Nothing else of the robot is assembled.
* **Units**: the sixteen other units (four shoulder modules, two plain and two with the hip adapter; four hip ab/ad
  supports; four upper legs; four lower legs) lie loose on the bench, each lying in a stable orientation on one of its
  long sides (none stands on an end), all within the robot's reach (roughly 0.35-0.95 m from the arm's base axis),
  outside the space the robot needs at the stand, at least about 20 mm apart; the lower legs lie across the bench's two
  long edges (y = +-0.70 m), key up and foot inwards, their knee end within about 2 cm of the edge (it may stick out a
  few mm past it). Where each unit lies and which way it is turned about the vertical differ between runs (within about
  +-30 mm and +-30 deg of its place in the shop's layout of the kit; a lower leg +-30 mm along the edge, +-10 mm across
  it and +-10 deg). The face a unit rests on is set by that layout and not drawn per run; identical units need not rest
  on the same face. Units with the same name in the packing list are interchangeable: any upper leg fits any shoulder,
  any lower leg any upper leg, any support any hip.
* **The actuators are unpowered** while the robot is built. Every joint (hip ab/ad, hip flexion, knee) turns against
  about 0.10 N m of friction (damping 1.4e-3 N m s/rad); each output pulley starts at its index within about 2 deg and
  stays wherever it is turned. A fastened shoulder limits its hip ab/ad joint to the module's range. Whatever pushes the
  half-built robot moves it, and a leg hanging from its hip swings when it is bumped.

## Cameras

| name | type | resolution | nominal horizontal field of view | mount (approximate, not surveyed) |
|---|---|---|---|---|
| `top` | RGB-D | 1280 x 960 | about 70 deg | overhead, roughly 1.6 m above the bench near the middle of the working area, looking straight down; the image covers the bench, the stand with the body and the robot's base; image width roughly along the cell's x |
| `wrist` | RGB-D | 640 x 480 | about 80 deg | on the gripper's coupling, about 6 cm off the tool axis on the side across from the pads (the tool's +x), a few cm out from the flange, tilted about 16 deg towards the fingertips; the fingers and a held unit are in the lower part of the image |

The cameras are off-the-shelf units: the lenses have radial and tangential distortion and are NOT calibrated; focal
length and principal point differ from the nominal values by a few per cent between units; the mounts are known only by
the descriptions above; the units are swapped and re-mounted between runs. The depth stream is registered to the colour
pixels (depth along the optical axis, 1 mm steps, range 0.1-4 m, no reading on depth edges and on about 3 % of the
pixels); its noise, new in every frame, is about 0.5 mm + 4 mm per m^2 x z^2 rms for the top camera and about 0.2 mm +
3 mm per m^2 x z^2 for the wrist camera. On top of it each camera unit has systematic errors that do not average away: a
smooth error field across the image of about 0.5 mm + 1.5 mm per m^2 x z^2 rms for the top camera (about 4 mm at the
bench) and about 0.1 mm + 1.5 mm per m^2 x z^2 for the wrist camera, a ripple that varies with the range (up to 1 mm for
the top camera, 0.2 mm for the wrist camera) and fixed per-pixel offsets of about 0.4 mm and 0.1 mm rms. The wrist
camera's lens sits about 0.13 m from the pinch point: closer than its 0.1 m minimum range it gives no depth. The hall's
brightness varies by up to about 20 % between runs, and each camera unit has its own slight colour response (gain and
tint within a few per cent), constant during a run; the image noise is new in every frame. The images come from a
physically based renderer of the hall (soft shadows, reflections on metal and plastic); the given render layer
(`~/given/lib/render`) draws scenes with MuJoCo's classic renderer instead (same geometry and camera model, plainer
shading, no hall).

## Fastening, safety, budgets

* **Fastening**: the cell does not report any fastening.
  * Keys: every hip ab/ad, hip flexion and knee output pulley has an 18 x 10 mm socket; a unit's key (plain shoulder
    4 mm, shoulder with the hip adapter 5 mm, upper leg 3 mm, lower leg 6 mm high:
    `describe()["fastening"]["key_heights_mm"]`) fits it two ways, 180 deg apart. A key is captured once at least 25 %
    of its height is inside the socket (counted from the pulley face: the lead-in counts), its axis within 2 mm and
    6 deg of the socket's and its turn about the axis within 6 deg of a key orientation, but only a key that comes in
    close to its socket gets that far (see "Collision model"). Measured in this cell with the arm pushing the held
    unit in at stiffness 0.5 and 10 mm/s and backing off to retry after a stall, the pulley up to 4 deg off its index
    and the hand aiming at the index, a key went in when it reached the pulley face at most this far off across the
    socket's short side: a plain shoulder's 0.4 mm within 1 deg of turn (0.1 mm at 2-3 deg, none at 4 deg); a shoulder
    with the hip adapter only about the 0.3 mm play, within 1 deg (a quarter of its 5 mm key lies deeper than the
    lead-in); an upper leg's 1.5 mm within 3 deg (1 mm at 4 deg); and a lower leg lying loose on the bench took a knee
    socket lowered onto it from 1.5 mm and 4 deg, the most tried (it slides into line). Farther off, the key stops on
    the chamfer or the pulley face. Each output pulley starts up to about 2 deg off its index, so measure or search
    the pulley's turn. The cell then holds the unit on the feature (it stays when it is let go) and pulls it home at
    up to 10 mm/s and 10 deg/s (the screws being driven), and fastens it when it rests at home (within 1.0 mm / 2 deg,
    0.5 mm axially) for 50 consecutive 10 ms ticks, moving slower than 5 mm/s and turning slower than 2 deg/s relative
    to the pulley (it may still be held). A captured key is released when its tip is pulled more than 2 mm clear of
    the socket, its axis is more than 4 mm / 12 deg off the socket's or its turn more than 12 deg off the key
    orientation.
  * Supports: a hip ab/ad support is captured on the body over its shoulder's journal up to 8 mm short of home along
    the hip ab/ad axis (within 3 mm across and 6 deg; its two screws reach their threads), pulled home and fastened like
    a key (within 1.5 mm / 3 deg); it needs its shoulder fastened first. A captured support is released when it is more
    than 16 mm short of home, 6 mm across or 12 deg off.
  * What counts: a shoulder only in its design key orientation (fastened the other way round it is bolted on and counts
    0: its hip flexion output then points the wrong way); a hip flexion or knee key in either orientation; a slot counts
    once it is fastened. Fastened units move as one body. Units can be fastened in any order the robot allows: a leg (an
    upper leg with its lower leg) may be built on the bench and mounted later.
* **Collision model**: the keys are exact 18 x 10 mm blocks and the sockets follow the actuator drawing: 0.3 mm of
  play per side (an 18.6 x 10.6 mm opening with square corners through the pulley) and the drawing's R1.2 lead-in,
  modelled as a 45 deg x 1.2 mm chamfer on both faces of every output pulley (the opening is 21.0 x 13.0 mm at the
  pulley face and narrows to 18.6 x 10.6 mm 1.2 mm deep); with the friction coefficient of 0.8 between the parts, a
  key pressed onto the chamfer is pushed across by only about a ninth of the force along the socket. Every unit
  collides with its true shape decomposed into convex pieces; the body, the stand, the bench and the robot arm with
  its gripper and camera housing collide with everything; inside a fastened group nothing collides. On the bench a
  unit rests on its collision pieces: its STL's lowest point lies about 0.7 mm above the bench top for a shoulder or
  an upper leg, about 1.0 mm above it for a lower leg and about 0.3 mm below it for a support (the collision pieces
  and the STL's visual tessellation differ by that much at the faces the units rest on).
* **Safety** (an episode with a violation scores 0 on every execution criterion): a robot link other than the gripper's
  four pad boxes pressing more than 30 N on the stand, the body, the bench or a unit; a unit's centre of mass faster
  than 3 m/s; a unit pressed on the environment or on another unit with more than 150 N for 0.2 s (the normal forces
  summed per unit and body it presses on; the gripper's pads holding a unit do not count).
* **Budgets per episode**: 1200 s of simulated time, 2400 s of wall clock (camera rendering counts), 120 s of silence at
  most, 600 camera frames.
