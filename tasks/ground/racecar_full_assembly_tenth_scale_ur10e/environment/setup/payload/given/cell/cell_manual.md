# Assembly cell: integrator's description (commissioning document)

This document describes the cell as it is built and as it behaves. Its numbers are constants of the cell or values
measured in it. The part files and the photographs show the car as it is drawn; where the cell's collision shapes
differ from that, "Collision model" lists the difference and its size. A run is one episode: one start of the cell,
with its own layout.

## Frame, bench, robot

* **Cell frame**: origin on the bench top at the robot arm's base axis, z up, x towards the work stand, y to the left
  seen from the arm. The bench top (2.0 x 1.4 m, x -0.55 .. 1.45 m, y -0.70 .. 0.70 m, oak veneer,
  0.994 m above the hall floor, on a steel frame with a lower shelf) is z = 0. The cell stands in a factory hall,
  lit by the hall's ceiling lamps.
* **Robot**: a 6-DoF UR10e-class arm (model in `~/given/robot/universal_robots_ur10e/`), bolted to a surveyed datum:
  its base at (0.00, 0.00, 0.00) m, the base frame being the cell frame turned 180 deg about z (q = 0 on the
  first joint points the arm along the cell's -x; exact). Joint names `ur/shoulder_pan_joint`,
  `ur/shoulder_lift_joint`, `ur/elbow_joint`, `ur/wrist_1_joint`, `ur/wrist_2_joint`, `ur/wrist_3_joint`; the joint
  ranges (a full turn either way, the elbow half a turn either way) and the home posture are in `describe()`. Joint
  torque limits 330 / 330 / 150 / 56 / 56 / 56 N m, base to wrist.
* **Tool**: a 2F-85-class adaptive gripper (model in `~/given/robot/robotiq_2f85/`) on the tool flange (the flange
  face sits at the arm's d6 = 0.11655 m along the last link's axis; the given arm model's `attachment_site` is
  16.55 mm short of it). The tool point `ur/tcp` is the pinch point between the pads, 0.1558 m along the tool axis
  from the flange face (not the tool point of the gripper maker's documents); its axes: z = the approach (the flange
  axis, out of the flange), y = the closing axis, x across the pads. The finger linkage is rigid (the gripper only
  pinches between its pads; it does not wrap around a unit); the pads are 22 mm across, 37.5 mm long and 8 mm
  thick, and along the approach their face centre lies about 14 mm short of the tcp fully open and 1-4 mm short
  for 10-45 mm grasps, the fingertips 5-18 mm beyond it, depending on the width; the finger linkage's outer
  envelope spans +-62-74 mm along the closing axis. Width 0-85 mm (the gap between the pad faces at the pads'
  middle; the pads cannot close below a gap of 0.9 mm); the commanded width moves at 0.10 m/s. Grip force: see
  `describe()["robot"]["gripper"]["force_model"]` (about 6 N per mm of squeeze per pad, capped at about 80 N per
  pad).
* **Wrist camera housing**: a 36 x 24 x 35 mm box on the gripper's coupling, on the tool's +x side; it is part
  of the robot (its collision model and the safety rule). In the tcp frame its centre is at (0.056, 0.000,
  -0.137) m and its outer face at x = 0.068 m.
* **Force / torque sensor** at the tool flange's centre, in the flange frame (z along the tool axis outwards, x along
  the closing axis = the tcp's y, y along the tcp's -x): what the wrist applies to the tool, the gripper's own weight
  included (about 10.3 N with the tool pointing down); noise about 0.3 N / 0.02 N m rms per reading, an unknown
  bias up to about 1 N / 0.05 N m that differs between power-ups: tare it.

## Datum boards

Checkerboards printed on rigid plates. On a board of C x R squares of side s the inner corner (i, j), i = 1 .. C - 1,
j = 1 .. R - 1, sits at ((i - C / 2) x s, (R / 2 - j) x s) in the board's own frame (x along the C squares, y along
the R squares, z out of the printed face, origin at the middle of the pattern); the square at the board's (-x, +y)
corner is black.

| board | squares | where |
|---|---|---|
| bench board | 9 x 6 of 20.0 mm (3.0 mm plate, 20 mm white margin) | surveyed, exact: flat on the bench near its +x / -y corner, the board's origin at (1.200, -0.500, 0.00305) m in the cell frame, its x along the cell's x, its y along the cell's y |
| tool plate | 5 x 4 of 6.0 mm (1.5 mm plate, 3 mm margin) | on the camera housing's outer face, facing the tool's +x: the board's origin at (0.07005, 0.0000, -0.1373) m in the tcp frame, its x along the tcp's z, its y along the tcp's -y; mounting tolerance about 1 mm / 0.5 deg |
| stand board | 6 x 5 of 15.0 mm (3.0 mm plate, 15 mm margin) | flat, printed face up, on the work stand's base plate on the side towards the robot arm, its printed face 18.05 mm above the bench. It moves with the stand; its place on the stand and the chassis's place relative to it are dimensioned in `~/given/drawings/stand_and_shelf.png` |
| shelf board | 5 x 4 of 12.0 mm (3.0 mm plate, 12 mm margin) | flat, printed face up, on the deck shelf's foot, its printed face 13.05 mm above the bench. It moves with the shelf; its place on the shelf and the deck's place relative to it are dimensioned in `~/given/drawings/stand_and_shelf.png` |
| tray board | 6 x 5 of 15.0 mm (3.0 mm plate, 15 mm margin) | flat, printed face up, on the parts tray, its printed face 15.05 mm above the bench. It moves with the tray; its place on the tray and the nests' places relative to it are dimensioned in `~/given/drawings/tray.png` |

Only the bench board and the tool plate are surveyed. The other three tell where their fixture stands in a run.

## The work stand and the rolling chassis

* **Work stand** (`~/given/drawings/stand_and_shelf.png`): a base plate on the bench, a post and a cradle plate on
  which the chassis tub is clamped by its flat underside. The car stands level, its front towards the robot arm (at
  the nominal placement the car's forward axis points along the cell's -x), its wheel axles 0.20 m above the bench.
  Nominal place: the point midway between the front and the rear axle at (0.62, 0.00) m. In every run the stand
  stands within +-15 mm of that place in x and in y and is turned by up to +-2 deg about the vertical. On the
  stand the chassis sits where the drawing says within about +-1 mm and +-0.3 deg: the clamping repeats that well,
  and where inside that it sits differs between runs.
* **Rolling chassis** (`~/given/parts/roller.stl`, in the car's own frame: origin on the ground plane under the middle
  of the rear axle, x forward, y to the left, z up, millimetres): the chassis tub with the drivetrain, the suspension,
  the steering, the front foam bumper, the body posts with the rear wing frame, the chassis brace, the motor, the kit
  speed controller, the radio receiver, the steering servo, the two battery hold-down posts, the four deck standoffs
  and the four wheel axles with their wheel hexes. In the cell it is one rigid body bolted to the stand: nothing of
  the suspension or the steering moves. Only the four wheel hubs turn.
* **Wheel hubs**: a hub is a wheel axle with its wheel hex and cross pin. It turns freely about its axle, unpowered,
  on its own: the drivetrain does not link the hubs. It turns against about 0.04 N m of dry friction (damping
  0.0002 N m s/rad): it stays where it is turned, and whatever applies more torque turns it. At the start of every
  run each hub stands turned by its own random angle between zero and 60 deg from the angle it has in `roller.stl`.
  The four axles are not parallel (`~/given/drawings/hub_and_wheel.png`).
* **Not fitted at this station**: the wheel nuts, and the battery hold-down bar with its two pins. The photographs of
  the finished car show them. In the cell the fastening rules stand for them; the threaded ends of the axles and the
  hold-down posts stay bare.

## The deck shelf, the parts tray, the loose units

* **Deck shelf** (`~/given/drawings/stand_and_shelf.png`): two rails carry the deck assembly by the underside of its
  plate, top face up, the same way round as on the car: at the nominal placements the deck on the shelf is parallel to
  its place on the car. The plate's top face lies 0.14 m above the bench; its right long edge overhangs the nearer
  rail by 47 mm with nothing under it. A back fence and two end stops locate the plate with +-1.5 mm of play.
  Nominal place: the middle of the deck plate at (0.30, -0.45) m. In every run the shelf stands within +-20 mm
  of that place in x and in y and is turned by up to +-5 deg about the vertical.
* **Parts tray** (`~/given/drawings/tray.png`): a plate with two nests, one for the compute module and one for the
  LiDAR. Ribs 5.5 mm high locate each unit with +-1.5 mm and +-2 deg of play; the compute module's nest locates
  its eight standoffs, and its floor has clearance holes for the screw ends under the standoffs. Both units stand in
  their nests on the face they stand on when mounted, but turned about the vertical: the storage orientation is not
  the mounting orientation. Nominal place: the middle of the tray at (0.30, 0.45) m. In every run the tray lies
  within +-40 mm of that place in x and in y and is turned by up to +-10 deg about the vertical.
* **Loose on the bench**: the four wheels and the stick pack. Every wheel lies on its inner face, the spoked outer
  face up. The stick pack lies flat on its underside. Each of the five lies within +-30 mm, in x and in y, of its
  place in the shop's layout of the kit and is turned about the vertical by up to +-30 deg from the direction that
  layout gives it; in addition the stick pack's lead end points either way along that direction, one or the other in a
  run. The face a unit rests on is the same in every run. The shop's layout is not given. All five lie 0.25-1.30 m
  from the arm's base axis, at least 60 mm apart and clear of the fixtures. Beside the car, where wheels travel onto
  the hubs (x 0.40 .. 0.85 m, |y| 0.09 .. 0.35 m at the nominal placement), nothing on the bench stands higher
  than 0.10 m.
* **Identical units** are interchangeable: any wheel fits any hub.

## Cameras

| name | type | resolution | nominal horizontal field of view | mount (approximate, not surveyed) |
|---|---|---|---|---|
| `top` | RGB-D | 1280 x 960 | about 70 deg | overhead, roughly 1.6 m above the bench near the middle of the working area, looking straight down; the image covers the bench with the work stand, the shelf, the tray and the robot's base; image width roughly along the cell's x |
| `wrist` | RGB-D | 640 x 480 | about 80 deg | on the gripper's coupling, about 6 cm off the tool axis on the side across from the pads (the tool's +x), a few cm out from the flange, tilted about 16 deg towards the fingertips; the fingers and a held unit are in the lower part of the image |

The cameras are off-the-shelf units: the lenses have radial and tangential distortion and are NOT calibrated; focal
length and principal point differ from the nominal values by a few per cent between units; the mounts are known only
by the descriptions above; the units are swapped and re-mounted between runs. The depth stream is registered to the
colour pixels (depth along the optical axis, 1 mm steps, range 0.1-4 m, no reading on depth edges and on about
3 % of the pixels); its noise, new in every frame, is about 0.5 mm + 4 mm per m^2 x z^2 rms for the top camera
and about 0.2 mm + 3 mm per m^2 x z^2 for the wrist camera. On top of it each camera unit has systematic errors
that do not average away: a smooth error field across the image of about 0.5 mm + 1.5 mm per m^2 x z^2 rms for the
top camera (about 4 mm at the bench) and about 0.1 mm + 1.5 mm per m^2 x z^2 for the wrist camera, a ripple that
varies with the range (up to 1 mm for the top camera, 0.2 mm for the wrist camera) and fixed per-pixel offsets of
about 0.4 mm and 0.1 mm rms. The wrist camera's lens sits about 0.13 m from the pinch point: closer than its
0.1 m minimum range it gives no depth. The hall's brightness varies by up to about 20 % between runs, and each
camera unit has its own slight colour response (gain and tint within a few per cent), constant during a run; the image
noise is new in every frame. The images come from MuJoCo's Filament renderer (physically based: soft shadows,
reflections on metal and plastic) in the hall, whose light is baked into its textures (`~/render`). The given render
layer (`~/given/lib/render`) renders the same way: the same renderer, hall and look for your own scenes, depth from
MuJoCo on the same geometry (`cell_api.md`, "Rendering your own frames"); it is the way to see what these cameras
would see. MuJoCo's own renderer (`mujoco.Renderer`, or the render layer's `"mujoco"` backend) stays available but
draws a different picture (no hall, MuJoCo's own lights, flat colours, no reflections), so image processing tuned on
its frames may behave differently on the cameras' frames.

## The gripper on a thin plate

The deck plate is 5.19 mm thick. Pinched across its two faces, the gripper is at its limit: the pads cannot close
below a gap of 0.9 mm, so whatever smaller width is commanded each pad presses with about 46-47 N and no more.
Measured in this cell with the deck assembly (232 g) held by its right long edge, the pads reaching 6-9 mm onto
the plate: the plate sags in the hand by 0.51-0.60 deg as it is lifted and, depending on where along the edge it
is held, by up to 1.11-3.28 deg over 20 s of carrying at up to 0.5 m/s, and it slips by 1.2-3.6 mm. With
the compute module and the LiDAR fastened on it (535 g) it sags by 0.86-0.94 deg at the lift and by
2.8-7.2 deg over the same carry, slipping by 2.8-7.6 mm; set down on the car from the pose it had in the hand
after the lift, it was fastened in 4 of 4 trials. Carried at up to 0.25 m/s instead, the plate sags by up to
0.80-0.95 deg over the same 20 s and slips by 0.87-1.02 mm; with the compute module and the LiDAR on it by
1.24-2.31 deg, slipping by 1.32-2.49 mm.

## Fastening

The cell does not report any fastening.

A unit is **captured** when it is inside the capture window of a place it can take (table below). The cell then holds
it on that place (it stays when it is let go) and pulls it home at up to 10 mm/s and 10 deg/s: the screws, or the
wheel nut, being driven. The stick pack and the deck assembly are held and pulled only once they move slower than
50 mm/s and turn slower than 30 deg/s relative to the chassis: until then a captured pack or deck moves freely,
and one that is still falling lands first. The cell **fastens** the unit when it rests at home, within the lock
tolerance, for 50 consecutive 10 ms ticks, moving slower than 5 mm/s and turning slower than 2 deg/s relative
to what it mounts on. The unit may still be held while this happens. A captured unit that is moved out of the release
window before it is fastened is released and is free again. Fastened units move as one body, and nothing can be
unfastened.

Windows, from the cell's own table (`describe()["fastening"]` has the same numbers). "Across" is the offset
perpendicular to the direction in which the unit goes on; "short of home" is the distance still to go along it, and a
unit may be at most 1.0 mm past home; "tilt" is the angle between the unit's axis and the place's; "turn" is the
angle about that axis from an orientation that the place takes. For the stick pack and the deck, tilt and turn are one
angle: the total rotation from the home pose about any axis.

| place | unit | mounts on | home | orientations | captured within: across / short of home / tilt / turn | released beyond: across / short of home / tilt / turn | lock tolerance | weight |
|---|---|---|---|---|---|---|---|---|
| `battery` | stick pack | the chassis | lying on the ribs of the battery bay (`battery_bay.png`) | one | 3.2 mm / 4.5 mm / 4 deg in total | 6.4 mm / 9.0 mm / 8 deg | 1.0 mm / 2 deg | 1.0 |
| `deck` | deck assembly | the chassis | resting on the tops of the four deck standoffs (`assembly_layout.png`) | one | 3.0 mm / 3.0 mm / 2.2 deg in total | 6.0 mm / 6.0 mm / 4.4 deg | 1.5 mm / 1 deg | 2.0 |
| `compute` | compute module | the deck plate | standing with its eight standoffs on the plate's top face (`assembly_layout.png`) | one | 2.5 mm / 4.485 mm / 4 deg / 3.1 deg | 5.0 mm / 7.98 mm / 8 deg / 6.2 deg | 1.0 mm / 2 deg, 0.5 mm along | 1.0 |
| `lidar` | LiDAR | the deck plate | standing with its base on the plate's top face (`assembly_layout.png`) | four, 90 deg apart; one counts | 2.5 mm / 3.0 mm / 4 deg / 5.0 deg | 5.0 mm / 6.0 mm / 8 deg / 10.0 deg | 1.0 mm / 2 deg, 0.5 mm along | 1.5 |
| `whl_FL`, `whl_FR`, `whl_RL`, `whl_RR` | any wheel | that corner's hub | the floor of the rim's hex pocket against the outer face of the wheel hex (`hub_and_wheel.png`) | six, 60 deg apart; all count | 2.0 mm / 3.75 mm / 6 deg / 6 deg | 4.0 mm / 7.00 mm / 12 deg / 12 deg | 1.0 mm / 2 deg, 0.5 mm along | 1.5 each |

* **Short of home.** The compute module, the LiDAR and the wheels are captured once 25 % of what goes in has gone
  in: of the 5.98 mm screw ends under the compute module's standoffs, of a nominal 4.0 mm for the LiDAR (it has no
  fasteners), of the 5.00 mm depth of the rim's hex pocket. They are released when that part is 2 mm clear of what
  it goes into.
* **Orientations.** The stick pack and the deck are captured one way round only; the other way round they are not
  captured and can be taken off and placed again. The compute module is captured in one orientation only. The LiDAR is
  captured and fastened in any of four orientations a quarter turn apart, but only the one the picture manual shows
  counts: in the other three it is fastened for good and counts nothing. A wheel counts in each of the six
  orientations of the hex.
* **Order.** No place needs another one fastened first: the units can go on in any order the car physically allows and
  the fixtures beside the car leave room for.
* **Sub-assemblies.** The compute module and the LiDAR mount on the deck plate wherever the deck assembly is: on the
  shelf or on the car. Fastened on a deck that is still on the shelf they count as they do on the car, as long as the
  deck assembly is not scrapped (see "Safety").
* **Total.** The weights add up to 11.5.

Measured in this cell: the unit held by the arm, moved onto its place at 10 mm/s with stiffness 0.5 and let go
there (the stick pack: let go above its bay, see below; a wheel: pushed on with its pocket turned to the hex), from
start poses spread over the window.

| place | start poses inside 80 % of the window: fastened | at the window's corners: fastened | outside 120 % of the window: fastened | largest force on a unit sustained for 0.2 s |
|---|---|---|---|---|
| `battery` | 62 of 66 | 10 of 19 | 6 of 12 | 25 N |
| `deck` | 71 of 71 | 32 of 32 | 4 of 22 | 28 N |
| `compute` | 71 of 71 | 32 of 32 | 4 of 30 | 61 N |
| `lidar` | 71 of 71 | 32 of 32 | 4 of 30 | 63 N |
| a wheel on a hub | 244 of 244 | 118 of 120 | 33 of 119 | 20 N |

## Collision model

The part files (`~/given/parts/*.stl`) and the photographs show the car as it is drawn: a display model in which
neighbouring parts touch or pass through each other and nothing has play. The cell simulates a second, simpler set of
shapes, which are not given as files. The chassis collides as its drawn shapes cut into convex pieces, with material
taken away and none added. The loose units, the hubs and the battery bay collide as a few convex pieces each, built
from the drawn sizes: these are solid where the drawn part is hollow, grooved or slotted, and in places they stand
beyond the drawn outline. Everything in which the collision shapes differ from the drawn shapes is listed here, with
its size. Coordinates are millimetres in the frame of `roller.stl` unless a unit's own file is named.

* **How the shapes are built.** The chassis, but for its battery bay with the two battery hold-down posts, the tops of
  the four deck standoffs and the four hubs, is cut into convex pieces that stay inside the drawn surface within
  0.16 mm; it loses material only where "Cut-backs for clearance" says. Each loose unit, each hub, the battery bay,
  the posts and the standoffs' tops collide as a few convex pieces built from the drawn sizes: boxes, and round shapes
  as prisms with 32 flat sides around the wheel's tyre and rim, 16 around each stick of the stick pack and 24
  around the LiDAR's head. The part files are simplified meshes of the drawn shapes: a round part can read up to about
  0.7 mm small across, and measured between two opposite faces a unit's collision shape is up to 0.64 mm wider
  than its part file. Masses are those of `kit.json`.
* **Where the built pieces differ from the drawn units.** Measured on the cell's pieces against the drawn solids: the
  largest distance of a piece's surface from the drawn material, and how far it stands beyond the drawn outline (the
  convex hull of the drawn unit or part).
  * A wheel stays within 0.22 mm of its drawn outline. It is solid in the openings between its spokes, in its centre
    hole and in its web around the pocket, up to 7.2 mm from the drawn material, and in the tyre's tread and the
    rim's profile, up to 4.0 mm.
  * A hub stays within 0.4 mm of its drawn outline. It is solid in the slots of the coupling cup at the axle's inner
    end, up to 5.9 mm from the drawn material, and elsewhere within 1.1 mm of it.
  * On the deck assembly the antenna mount collides as a box that stands up to 9.1 mm beyond the drawn mount on its
    top, on its two sides and on its face towards the middle of the plate (12.7 mm from the drawn material). Each
    antenna collides as a box around its base, up to 7.1 mm beyond the drawn base on every face (9.6 mm from the
    drawn material), and a cylinder around its whip, up to 4.0 mm from the drawn whip, the most near its tip. The
    motor controller board, the power distribution board and the spacer block are boxes whose corners stand up to
    2.5, 2.0 and 1.4 mm beyond the drawn ones. The plate is closed at its small holes, up to 1.8 mm from the
    drawn material.
  * The LiDAR's cap collides as a box whose corners stand up to 6.7 mm beyond the round cap (6.9 mm from the drawn
    material), its square base as a box whose corners stand up to 1.7 mm beyond the drawn base. Its head stays
    within 0.7 mm of the drawn head, and its connector block is solid where the drawn block is recessed, up to
    2.6 mm, and nowhere beyond it.
  * The compute module, the stick pack and the battery bay: see their own items below.
* **Closed openings.** The 36 small holes of the deck plate are closed (its central opening and its two long slots
  are open). The openings between the wheel's spokes and the wheel's centre hole are closed: the wheel's outer web is
  a closed disc.
* **Fasteners, leads and plugs do not collide.** These are: every screw on the chassis (11 groups of them) and the
  screws of the two battery hold-down posts; the threaded end of each wheel axle, which stands 8.75 mm out of the
  wheel hex's outer face; the leads and plugs of the stick pack, the kit speed controller, the motor, the steering
  servo and the radio receiver; and the screw ends under the standoffs of everything that stands on the deck plate:
  5.98 mm under the compute module's, 5.46 mm under the motor controller's, 5.34 mm under the antenna mount's
  and 8.94 mm under the power distribution board's. The last come out 3.75 mm below the plate's underside and are
  the lowest points of `deck_assembly.stl`; the deck does not rest on them. Above the plate's top face a standoff is a
  foot of its unit and collides. What these parts locate on a real car, the capture windows of "Fastening" locate
  here.
* **Cut-backs for clearance.** Where a unit that comes down inside its capture window, turned and tilted as far as the
  window lets it, would run into a neighbour that the drawn model leaves no room for, one of the two collision shapes
  is cut back. There are 46 box-shaped cuts on 9 parts, about 4.6 cubic centimetres in all, and this list is
  complete. Each item gives the outer bounds of its cut; inside them the cut is a set of boxes and can be stepped:
  * the chassis brace, beside the battery bay: between x = 47.6 and x = 191.1 everything of it at y below -7.32
    is removed (in steps, in places only below -9.82), about 3.5 of its 16.9 cubic centimetres. As drawn, the
    brace reaches 1.87 mm over the stick pack there. The cut ends at x = 47.6: there the brace is as drawn, and a
    stick pack that comes down offset towards the car's centre to the edge of its capture window, turned as far as the
    window lets it, can meet it with its rear end;
  * the chassis tub, at the front end of the battery bay: the boss around the front hold-down post begins at x =
    190.59 instead of x = 189.00;
  * the front damper stay: above z = 63.72 its rear face lies at x = 263.97 instead of x = 260.00 (at x =
    262.97 between y = 20 and y = 25);
  * the front differential cover: between x = 253.8 and x = 262.1, for |y| up to 11.2, its top lies at z =
    63.72 instead of z = 67.99;
  * the chassis tub, three bumps under the ends of the deck plate: two at the front, between x = 247.4 and x =
    257.1 for |y| from 7.8 to 11.2, lowered to z = 63.72; one at the rear, between x = 11.5 and x = 18.3
    for |y| up to 2.6, lowered to z = 64.32 (as drawn it stands 2.32 mm into the plate);
  * the rear differential cover: between x = 1.91 and x = 3.21 its top is lowered to z = 64.32;
  * the power distribution board, on the deck assembly: its front edge is cut back by 3.9-5.5 mm over its whole
    width;
  * the two front standoffs of the power distribution board: what stands more than 9.55 mm above the deck plate's
    top face is removed;
  * the LiDAR: the two lugs on the end face of its connector block are cut back by 4.2-5.0 mm, from 6.0 mm above
    the LiDAR's base plane upwards.
* **One pair does not collide.** The stick pack and the two right-hand deck standoffs, front right and rear right,
  pass through each other: as drawn, the pack's side lies within 0.5 mm of them. These two standoffs collide with
  everything else: the gripper, the deck plate, the other units.
* **The deck standoffs end under the deck.** Each of the four standoffs collides up to z = 64.97, the plane of the
  deck plate's underside at home. As drawn it goes on to z = 71.14, through a hole of the plate that is narrower
  than the standoff (2.70 mm against 2.78 mm). In the cell the plate does not go over the standoffs: it rests with
  its underside on their four tops, and its holes locate nothing. As drawn, the standoffs' lower ends hang 5.54 mm
  above the tub; in the cell they are rigid with the chassis.
* **The wheel's seat and stop.** As drawn, the rim's hex pocket (12.20 mm across flats, 5.00 mm deep, no lead-in)
  fits the wheel hex (12.00 mm across flats) with 0.10 mm of play per flat. That fit is not simulated. In the
  collision model the pocket is a round seat of radius 8.90 mm, 2.5 mm deep from the pocket's mouth (the hex's
  corners lie 6.90 mm from its axis, so the seat leaves 2.0 mm around them). Deeper, down to the pocket's floor,
  the wheel carries a stop: a cylinder of radius 8.40 mm. A wheel pushed onto a hub therefore comes to rest with its
  stop on the hex's outer face, 2.50 mm short of home, whatever its turn. When the wheel is captured (its pocket's
  turn inside the window of one of the hex's orientations), the cell switches that wheel's stop off and the fastening
  pulls the wheel home. The hex collides as drawn, with its one flattened corner and the 45 deg chamfer on the
  corners of its outer end. The seat is wider than the hub ring around the pocket's mouth (radius 8.5 mm), so that
  ring is not there: a wheel that comes on too far off across meets the hex with the face of its web, 3.84 mm short
  of home by the drawn sizes ("The wheels and the hubs" gives the measured values). The stop's contact is stiffer than
  the cell's other contacts. Apart from the closed openings and the differences of the built pieces listed above and
  below, the stop is the only material in the collision model that the drawn parts do not have.
* **Sunk bases are trimmed.** As drawn, the LiDAR's base lies 0.26 mm inside the deck plate's top face and the
  spacer block under the motor controller 0.09 mm; in the collision model both undersides end at the plate's top
  face.
* **Under the compute module's carrier board only its standoffs collide.** The lowest layer of `compute_module.stl` is
  a thin tray-shaped part of 91.28 x 80.20 mm with walls 5.75 mm high, under the carrier board of 88.62 x
  70.01 mm. It has no collision: pads, fingers and neighbouring units pass through it. On one side it is 8.9 mm
  wider than the carrier board. The tray's nest locates the standoffs, not this part. The carrier board collides as a
  box, solid at its holes, up to 1.2 mm from the drawn material. Above it the module collides as a stack of 23
  boxes, each the bounding box of the drawn module over its height, in bands within which that bounding box changes by
  at most 0.5 mm: they are solid in the module's gaps and recesses, up to 9.7 mm from the drawn material, and at
  some corners they stand up to 5.0 mm beyond the drawn outline.
* **The stick pack collides as its heat-shrink sleeve**: two round sticks side by side, 46.38 mm across the pair and
  23.88 mm high, over the sleeve's 131.40 mm length and 1.4 mm further over the end cap at its lead end. Its
  lead and plug do not collide. Against the drawn sleeve the two prisms stand at most 0.23 mm beyond it along its
  sides and up to 2.5 mm at its ends, the most at the corners of the lead end.
* **The battery bay's pieces.** The ribs, walls, gussets, ledges, floor ends and boss of the battery bay and the two
  battery hold-down posts are built pieces too. Where the stick pack meets them, within 6 mm of the pack at home,
  they lie within 1.8 mm of the drawn bay: the rib tops it lies on within 0.7 mm of the drawn ribs and the
  gussets' faces within 1.8 mm of the drawn gussets; "The stick pack and its bay" gives their places as the cell has
  them. Away from the pack they stand out of the drawn tub by up to 10.1 mm at the front end wall, which closes the
  bay beyond the boss, 4.1 mm on the outer wall's outside and 3.5 mm at the rear end wall and the bay's rear outer
  corner; the hold-down posts by up to 0.9 mm.
* **At rest** a unit lies on its collision pieces. For a wheel and the stick pack on the bench and for the LiDAR in
  its nest, the lowest point of the part file is then within 0.9 mm of the surface the unit rests on. The deck
  assembly lies on the shelf's rails with the underside of its plate, and the compute module stands on the floor of
  its nest with the feet of its standoffs: the screw ends below them, which are the lowest points of these two part
  files, do not collide.

## The stick pack and its bay

`~/given/drawings/battery_bay.png` shows the bay. What the stick pack does there, measured in this cell:

* **The bay guides sideways only.** The pack's seat is the tops of the bay's ribs. Between the walls it has 3.06 mm
  of play on the side of the car's centre and, on the outer side, 1.45 mm at the gussets and 4.46 mm between them;
  lying between the walls it can be turned by at most 2.01 deg. Lengthwise nothing guides it: at home it has
  8.0 mm to the bay's rear end wall and 5.6 mm to the boss at the bay's front end. So a pack that has dropped
  between the walls is inside its window sideways, but a pack that lies flat on the ribs more than 3.2 mm off
  lengthwise is not captured: it has to be taken out and placed again. Measured: lying 3.0 mm off lengthwise it was
  captured, 3.5 mm off it was not.
* **The gripper cannot follow the pack into the bay.** Holding the pack across its width, the finger on the side of
  the car's centre comes down beside the chassis brace. Even as cut back in the collision model, the brace leaves
  3.75 mm beside the pack, less than a pad's thickness. Measured with the given gripper, the lowest the pack can be
  brought without a finger link touching the chassis is with its underside 22 mm above its seat, and that only with
  the closing axis tilted by 10 deg about the pack's long axis, the finger on the side of the car's centre higher.
  Untilted it is 26 mm, tilted the other way 32 mm, and gripped near its ends 44 mm, where the fingers meet the
  right-hand deck standoffs.
* **Let go there, the pack falls.** At 22 mm its underside is still 3.0 mm above the tops of the bay's walls: it
  falls freely before it is between them. Measured from that height with the pack up to 1.5 mm off across and up to
  2 deg turned at the release: 19 of 20 packs were fastened, each captured 0.08-0.10 s after the release,
  still falling; it landed, was pulled home and was fastened 0.75-1.13 s after the release. With the capture
  switched off the same drops came to rest 0.2-3.4 mm off across and turned by 0.1-3.4 deg; a pack that lands
  on an outer gusset lies 0.7 mm high and tilted by about 3 deg. Opening the gripper by itself shifts the pack
  0.2-0.6 mm across. Released 1.5 mm towards the car's centre and turned by 2 deg, the pack's upper part
  brushes the cut-back brace with about 5 N.

## The wheels and the hubs

`~/given/drawings/hub_and_wheel.png` shows the hub and the rim. What a wheel does on a hub, measured in this cell with
the wheel held by the arm and pushed along the hub's axle at 10 mm/s:

* **Across the axle.** Up to 2.0 mm off, the round seat goes over the hex and the wheel comes to rest on its stop.
  From there up to 2.5 mm off, the chamfer on the hex's outer end guides the wheel into the seat under the push.
  Farther off, the hex's end face meets the wheel's web and the wheel stands 3.537-3.835 mm short of home,
  3.66 mm on average (3.84 mm by the drawn sizes).
* **On the stop.** The wheel rests 2.50 mm short of home: measured 2.44 mm at the first stall, 2.46 mm under a
  push of 5 N and 2.34 mm under 20 N. No push takes it further.
* **Capture.** With the wheel on the stop and its pocket's turn within 6 deg of one of the hex's orientations, the
  stop opens and the fastening pulls the wheel the rest of the way at up to 10 mm/s; it was fastened 1.1 s after
  the capture. Pushed on with the pocket held within 1 mm and 3 deg of the hex, 80 of 80 wheels were fastened.
  Pushed on without regard to the turn, 60 of 240 were.
* **The hub turns with the wheel.** The hub's friction is 0.04 N m. From a push of about 6 N on, a wheel on its
  stop takes the hub along when the hand turns it: the turn between pocket and hex then stays what it was. With a push
  of 5 N or less, turning the hand at 10 deg/s brought the pocket to an orientation of the hex within
  10-15 deg of turn in 24 of 24 trials, the hub creeping by less than 0.01 deg. While a captured wheel is
  pulled home with the gripper already open, the hub turns by less than 0.01 deg; a hand that still holds the wheel
  can turn the hub with it.
* **After a release.** A captured wheel that is pulled out of its release window gets its stop back only once the stop
  is more than 0.1 mm clear of every hub. Until then that wheel has no stop and can be slid home on a hub without
  being captured. It is then not fastened and does not count, and nothing in the cell says so.

## Safety

* **Violations.** Two things are safety violations: a robot link other than the gripper's four pad boxes pressing with
  more than 150 N on the chassis, a fixture, the bench or a unit (the normal forces of that link on that body
  summed); a unit pressed on the environment or on another unit with more than 150 N for 0.2 s (the normal forces
  summed per unit and body it presses on; the gripper's pads holding a unit do not count). The deck standoffs and the
  chassis brace are parts of the rigid chassis: a finger link, as opposed to a pad, that presses on them falls under
  the first rule.
* **What a violation costs.** A safety violation does not end the episode, but an episode with one keeps only 30 %
  of its score. One violation or several: the episode keeps the same 30 %.
* **Scrapped units.** A loose unit whose centre of mass moves faster than 3 m/s, or that falls off the bench, is
  scrapped. The cell releases it if it is captured; it can no longer be fastened anywhere, and if it was fastened it
  no longer counts. The episode goes on. If the scrapped unit is the deck assembly, the compute module and the LiDAR
  fastened on it stop counting with it.

## Budgets

Per episode: 650 s of simulated time, 3100 s of wall clock (camera rendering counts), 120 s of silence at most,
450 camera frames, at most 500 ticks per `step` call. There are 4 episodes.
