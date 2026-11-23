# Hand assembly cell: integrator's description (commissioning document)

Lengths in this document are millimetres where a table or a sentence says so and metres in the cell frame; the given
model files are in millimetres; the API speaks metres and radians (`cell_api.md`). "The palm base" is the bottom shell
of the palm with the digits' wall screws in it; "home" is the place and attitude that the picture manual shows for a
unit on the hand. Home is documented nowhere else: read it from the manual and from the units' own shapes.

## Frame, bench, robot

* **Cell frame**: origin at the centre of the bench top, z up, x away from the robot, y to the robot's left. The bench
  top (1.6 x 1.0 m, light wood, 0.994 m above the floor, on a steel frame) is z = 0. The cell stands in a hardware lab
  bay. Of the room's fourteen lamps the six nearest the bench light the cell and what stands on it (an LED panel over
  the cell, two LED bars on the camera portal and three ceiling panels), and the two nearest of those, the panel over
  the cell and one bar, cast the shadows.
* **Robot**: a 7-DoF Panda-class arm (model in `~/given/panda/`), bolted to a surveyed datum: its base at (-0.45, 0, 0)
  m with its axes aligned with the cell frame (exact). Joint names `panda/joint1..7`; the joint ranges are in
  `describe()`. Tool point `tcp`: 0.1034 m along the hand's z (the approach direction, along the fingers) from the
  flange; the fingers close along the hand's y. Its controller is `arm.assembly_cell.controller` of the given library;
  gravity is compensated.
* **Control tick**: one tick is 10 ms = 10 physics steps of 1 ms (MuJoCo, implicitfast integrator, elliptic friction
  cones). The arm's joint position targets are ramped in a straight line over the physics steps of each tick, from the
  last tick's targets to the new ones; they do not jump at the start of a tick.
* **Home posture** (`home_q` in `describe()`: [0, -1, 0, -2.6, 0, 1.6, 0.785] rad): at the start of an episode the arm
  is folded back over its base, the hand pointing down with the tool point at about (-0.2, 0, 0.43) m and the gripper
  open, so that the top camera sees the fixture with the palm base and both trays.
* **Gripper**: parallel jaw, opening up to 80 mm; the opening ramps at 50 mm/s. Each finger presses with 625 N/m x
  (object width - target width), at most 30 N. (The drive behind both fingers gives 2500 N/m x half that difference, at
  most 60 N, and each finger gets 0.5 of it; the `force_model` text of `describe()` states the force per finger.) The
  pads are the hand model's own (`~/given/panda/panda.xml`): on each finger a face of 17 x 17 mm with four corner blocks
  (6 x 6 mm and 6 x 7 mm) that stand 1.5 mm proud of it and meet a flat surface first; `pad_size_m` in `describe()` (17
  x 17 mm) is that face. The friction coefficient is 1.0 on the fingers' pads and 0.8 between units, trays, fixture and
  bench. The finger bodies collide with their own shapes as the given model has them; every face of a finger body counts
  as a pad (rule R1).
* **Force / torque sensor** at the flange (hand frame): noise about 0.3 N / 0.02 N m rms per reading, an unknown bias up
  to about 1 N / 0.05 N m that differs between episodes; it reads the hand's own weight.
* **Contact**: contacts are compliant. Between units, trays, fixture and bench the contact time constant is 4 ms; the
  fingers (pads and finger bodies), the nests' supports and the fixture's supports under the digits are stiffer (2 ms).
  A part that lands on another sinks into it for a moment by a fraction of a millimetre before it rests.

## Datum boards (the integrator's survey; exact)

Checkerboards printed on rigid plates. On a board of C x R squares of side s the inner corner (i, j), i = 1 .. C - 1, j
= 1 .. R - 1, sits at ((i - C / 2) s, (R / 2 - j) s) in the board's own frame (x along the C squares, y along the R
squares, z out of the printed face); the square at the board's (-x, +y) corner is black.

| board | squares | where (exact) |
|---|---|---|
| bench board | 9 x 6 of 20 mm (3 mm plate, 20 mm white margin) | flat on the bench; inner corners at x = 0.335 + (i - 4.5) x 0.020, y = 0.005 + (3 - j) x 0.020, z = 0.00305 (i = 1..8, j = 1..5) (cell frame, m); its black corner square at the board's -x, +y corner (cell axes) |
| fixture board | 5 x 4 of 10 mm (2 mm plate, 4 mm margin) | flat on the fixture's base plate beside the palm base, printed face up; inner corners at x = 21 + 10 x j, y = -22 + 10 x i, z = 14.05 (i = 1..4, j = 1..3) (fixture frame F below, mm); its black corner square at the board's -x, -y corner (F's axes). It moves with the fixture |
| hand plate | 5 x 4 of 7 mm (1.5 mm plate, 3 mm margin) | flat on the Panda hand's -x face, facing -x; in the HAND frame inner corners at x = -0.03505, y = 0.050 + (i - 2.5) x 0.007, z = 0.020 + (j - 2) x 0.007 (i = 1..4, j = 1..3) (m); its black corner square at the plate's -y, -z corner; mounting tolerance about 1 mm / 0.5 deg, drawn per episode |

The fixture board and the hand plate have the same inner pattern (4 x 3 corners), and the bench board's larger pattern
contains that pattern many times over. A chessboard call that is asked for the small pattern can therefore return
corners of any of the three boards, and it reports success either way: check which board a result lies on (the squares'
size in pixels tells the bench board from the other two; where a board can be tells the rest). Measured in this cell
with OpenCV 5.0, one call on a whole frame with the call's default flags; a result counts for a board when all of its
corners lie within 3 px of that board's own corners.

| frame | pattern asked for | `cv2.findChessboardCorners` returned | `cv2.findChessboardCornersSB` returned |
|---|---|---|---|
| `top`, at the start of an episode (10 episodes) | the bench board's | the bench board, every corner, in every episode | the bench board, every corner, in every episode |
| `top`, at the start of an episode (the same 10) | the small pattern | the fixture board, every corner, in 10 of 10 | never the fixture board: a piece of the bench board in 9 of 10, and in 1 a grid whose corners lie up to 29 px off every board |
| `top`, at the start, 10 other episodes | the small pattern | the fixture board in 9 of 10 | the fixture board in none |
| `top`, the hand turned so that the hand plate faces up (10 episodes) | the small pattern | the hand plate in 10 of 10 | the hand plate in 0 of 10 |
| `wrist`, the hand pointing down over the fixture board (10 episodes) | the small pattern | the fixture board in 10 of 10 | the fixture board in 10 of 10 |

In the `top` frame the bench board's squares are about 13.9-15 px, the fixture board's about 7.1-7.7 px and the upturned
hand plate's about 7.5-8.2 px; in those `wrist` frames the fixture board's squares were 18.6-19.4 px. At the home
posture the hand plate is edge-on to the top camera and neither call returns it.

## Fixture and palm base

The fixture is one rigid piece of light machined aluminium on the bench. **Fixture frame F** (`describe()` calls it the
palm frame): origin on the bench top, z up. Nominally F's origin is at (-0.01, 0) m of the cell frame with F's +y along
the cell's +x (away from the robot) and F's +x along the cell's -y; in an episode it is moved and turned within the
bounds given below ("What changes"). The fingers of the finished hand point along F's +y, the thumb is on F's -x side,
the back of the hand faces up.

| piece | in F (mm) |
|---|---|
| base plate | x -213 .. 67, y -81 .. 219, 12 mm thick, on the bench |
| pedestal | a block x -41 .. 8, y -27 .. 50 from the plate up to the palm base's palmar face, 150 mm above the bench |
| two jaws | on either side of the palm base's wrist end (y -41 .. -27), 6 mm thick, their tops 172.51 mm above the bench, joined by a bar under the palm base |
| rest deck | a plate 6 mm thick on two posts, its top 140.8 mm above the bench, under the three fingers: it reaches from about 84 to 135 mm in front of each mounted finger's sideways (ab/ad) joint and covers every angle that joint can take |
| thumb pad | 10 x 10 mm on a slender post at (-82, -6.8), its top 139.31 mm above the bench, with a pocket 0.3 mm deep in it that clears a screw head of the mounted thumb |

`cell_config.json` (`fixture.pieces`) lists every piece as the cell simulates it: boxes (centre, half sizes, turn about
the vertical) and vertical prisms (plan outline, bottom and top), in F, mm. The deck is 25 such prisms.

**The palm base** (`parts/palm_base.stl`) is fixed on the pedestal and does not move on the fixture. Its model file's
frame has its origin at (-28.73, 9.97, 163.75) mm of F, its +x along F's +y, its +y along F's -x and its +z up (exact).

## Parts trays and nests

Two aluminium plates carry the nine loose units in nests. Tray frame: origin on the plate's top face at its centre, x
along the plate's long side, z up.

| tray | plate (m) | nominal place (cell frame) | nests |
|---|---|---|---|
| `digit_tray` | 0.36 x 0.24 x 0.015 | centre (-0.181, -0.269) m, its x axis turned 45 deg from the cell's x axis about the vertical | `nest_f1` (finger module), `nest_f2` (finger module), `nest_f3` (finger module), `nest_thumb` (thumb module) |
| `parts_tray` | 0.24 x 0.20 x 0.015 | centre (-0.189, 0.322) m, its x axis turned 141 deg from the cell's x axis about the vertical | `nest_cover` (dorsal cover), `nest_top` (upper shell), `nest_stack` (electronics stack), `nest_hub` (power hub), `nest_cap` (press cap) |

`describe()["trays"]` lists, per tray, every nest with its name, the kind of unit it holds and its point `x_in_tray_m`,
`y_in_tray_m` in the tray's frame (the nests never move on their trays; `describe_example.json` shows the reply). Every
nest lies 0.30-0.62 m from the robot's base axis. Which finger module lies in which finger nest does not matter: the
three are identical.

How each unit lies at rest in its nest, by its model file's frame (`kit.json` gives each file's frame; +z is up in every
nest, within about 1 deg). "Turned" is the same nest with the part lying the other way round (see "What changes").
Heights are above the plate's top face; lowest and highest are those of the unit's collision shape.

| nest | holds | file frame's origin in the tray (x, y, z mm) | file frame's +x from the tray's +x (deg, counter-clockwise seen from above) | lowest / highest (mm) |
|---|---|---|---|---|
| `nest_f1` | finger module | -135.6, -0.3, 22.5 | -90.2 | 3 / 43.5 |
| `nest_f2` | finger module | -65.6, -0.3, 22.5 | -90.2 | 3 / 43.5 |
| `nest_f3` | finger module | 4.4, -0.3, 22.5 | -90.2 | 3 / 43.5 |
| `nest_thumb` | thumb module | 102, 55.2, 129.6 | -38.9 | 110 / 148 |
| `nest_cover` | dorsal cover | -55, -45, 25.5 | 0 | 11.9 / 41.6 |
| `nest_cover`, turned | dorsal cover | -55, -45, 25.5 | -180 | 11.9 / 41.6 |
| `nest_top` | upper shell | 66.5, -46.4, 15.2 | -170.7 | 11.9 / 22 |
| `nest_top`, turned | upper shell | 63.5, -53.6, 15.2 | 9.3 | 11.9 / 22 |
| `nest_stack` | electronics stack | -79.9, 59.4, 19.2 | 119.8 | 11.9 / 29 |
| `nest_hub` | power hub | -8, 60, 22 | 90 | 11.9 / 32.2 |
| `nest_hub`, turned | power hub | -8, 60, 22 | -90 | 11.9 / 32.2 |
| `nest_cap` | press cap | 78, 55, 17.5 | -90 | 12 / 23 |
| `nest_cap`, turned | press cap | 78, 55, 17.5 | 90 | 12 / 23 |

These are the cell's own values after the units have settled at the start of an episode; a finger's and the thumb's
values are those of its base link. They hold for every drawn chain angle within what was measured over 300 drawn
episodes: a finger's base lay within 0.15 mm on each axis and 0.1 deg about the vertical of the table's values, the
thumb's within 0.25 mm and 0.2 deg.

* **Finger nest**: a block under the base servo with a low wall on either side (1 mm off its sides) and an end stop
  behind it, a small block under the first yoke next to the sideways joint, and a deck under the fingertip from 84 to
  135 mm in front of that joint, wide enough for 13 deg either way: the chain lies on it at the angle it was laid down.
* **Thumb nest**: a riser 110 mm high under the base servo and its yoke, and a support under the thumb's two outer links
  over the angles its side joint can be laid at.
* **Stack nest**: the stack lies on four posts 12 mm high with a lip on either side of it.
* **Upper shell, dorsal cover, power hub**: each lies level on four or five posts (6 x 6 mm) that stand clear of its
  screws; **press-fit cap**: on a block 12 mm high.

`cell_config.json` (`trays`) lists every nest's pieces as the cell simulates them (boxes and vertical prisms, tray
frame, mm), for a flat part also as they stand when the part lies turned.

## The digit modules' joints

A finger module and the thumb module are chains of links; `kit.json` gives their links, joints, stops and the joint data
as the cell simulates them, and the module folders hold a MuJoCo model of each that behaves the same.

* **Unpowered**: no joint of a module is driven or locked while the hand is built. A servo joint has 0.02 N m of dry
  friction, 0.01 N m s/rad of damping and 0.0001 kg m^2 of reflected inertia; it turns when the torque on it passes its
  friction and stays where it is left. Measured in this cell: a joint under a steady load of up to 0.95 of its friction
  torque creeps by less than 0.0008 deg in 60 s; and with nothing touching the hand, a mounted joint drifts by at most
  0.005 deg in any 100 s and 0.0001 deg in 3 s (the half-built hand at two stages and the finished hand, all 19 joints).
* **Stops**: the ranges in `kit.json` are mechanical stops (rule R3 watches the torque a joint carries against one). A
  loose module's joints have their own ranges; two joints of each module end on the palm base once the module is mounted
  and then have a narrower range, given per mounting place.
* **A finger's last joint** is not driven: a small rocker link closes a four-bar loop between the two outer links, so
  that the last joint follows the one before it (`kit.json`: the closing point and the law). In the cell the rocker and
  the last link are bodies of their own and a pin constraint closes the loop.
* **Start pose**: every joint at zero (digits straight), except the one joint per module that the nest lets swing: a
  finger's sideways (ab/ad) joint and the thumb's side joint lie at the angle drawn for the episode.

## What changes from episode to episode

Every episode is drawn afresh; the values are in no reply. The bounds below are never left.

1. The fixture with the palm base: its place within +-10 mm in x and in y of the nominal place and its turn about the
   vertical within +-2 deg.
2. Each tray, on its own: within +-10 mm in x and in y of its nominal place and within +-5 deg of its nominal turn about
   the vertical.
3. Each finger module's sideways (ab/ad) angle in its nest: within +-10 deg of straight, each nest on its own.
4. The thumb module's side-joint angle in its nest: within +-8 deg of its zero.
5. The upper shell, the dorsal cover, the power hub and the press-fit cap, each on its own: as the table above gives it,
   or turned half a turn about the vertical, with equal chance. The jitter turns units about the vertical only: no unit
   ever lies on another face.
6. The electronics stack always lies the same way round in its nest.
7. The cameras (mounts, focal length, principal point, lens distortion, colour response, the depth units' systematic
   errors), the room's brightness, the force sensor's bias and the hand plate's mounting (see below).

When the cell sets up an episode it checks the drawn layout: the fixture and the two trays stand on the bench at least
15 mm apart, every nest lies in the ring named above, and every loose unit can be gripped where it lies. A layout that
fails is drawn again as a whole, within the same bounds. Not drawn: the robot's base, the bench board, the palm base's
place on the fixture, the nests' places on their trays, the joints other than the two named above.

## Cameras

| name | type | resolution | nominal horizontal field of view | nominal mount | how far a unit's mount is off |
|---|---|---|---|---|---|
| `top` | RGB-D | 1280 x 960 | 70 deg | the lens at (-0.097, -0.012, 1.25) m of the cell frame, looking straight down; image x along the cell's +x, the image's top towards the cell's +y; about 0.73 pixels per mm on the bench top | within 30 mm on each axis and 3 deg about an axis of any direction |
| `wrist` | RGB-D | 640 x 480 | 80 deg | on the Panda hand's +x flank: the lens at (57.2, 0, 41.2) mm of the hand frame, its axis tilted 16.92 deg from the hand's +z towards the hand's -x, image x along the hand's +y | within 3.3 mm on each axis and 1 deg |

The cameras are off-the-shelf units and are NOT calibrated; the units are swapped and re-mounted between episodes. A
unit's focal length is within 3 % of the nominal value (914 px for `top`, 381 px for `wrist`), in x and in y on their
own; its principal point is within 2 % of the image size of the image centre; its lens has radial distortion (`k1` in
-0.08 .. 0.02, `k2` in -0.02 .. 0.02) and tangential distortion (within +-0.001), in OpenCV's model. No reply gives a
unit's values. The depth stream is registered to the colour pixels (depth along the optical axis, 1 mm steps, range
0.1-4 m, no reading on depth edges and on about 3 % of the pixels); its noise is about 0.5 mm + 4 mm per m^2 x z^2 for
the top camera and about 0.2 mm + 3 mm per m^2 x z^2 for the wrist camera, plus unit-specific systematic errors of up to
about a millimetre that do not average away. The room's brightness varies by up to about 20 % between episodes, and each
camera unit has its own slight colour response (gain within 5 %, tint within 3 %), constant during an episode; the image
noise is new in every frame.

* The units are black (printed parts, servo cases, pads) on light trays, a light fixture and a light bench.
* The images come from a physically based renderer of the room (soft shadows, reflections on metal and plastic). Small
  hardware that cannot be seen from any direction because its own link encloses it is left out of the images (the model
  files contain it). The given library holds the camera pipeline itself (`~/given/lib/render`: the camera model with
  lens distortion, the photo and depth-sensor models, the checkerboards) and the cameras' renderer:
  `arm.assembly_hand.agent_render` pictures a twin of yours with the same renderer, room (`~/render`) and look as the
  cell's cameras, `render.filament.view` any MuJoCo scene of yours in that room (`cell_api.md`, "Picturing a model of
  your own"). MuJoCo's own renderer (`mujoco.Renderer`) draws a different picture: no room, its own lights and plain
  materials. The library holds no model of this cell: `arm.assembly_six`, `arm.assembly_info.cell_render` and the rest
  of `arm.assembly_cell` and `arm.assembly_info` came with earlier cells.

## Fastening (the cell's rules; the cell reports no fastening)

The cell has no screwdriver that you command. Every screw is drawn from the start on the unit its head bears on and
never appears, disappears or moves. The cell counts a unit's screws as driven, and so fastens the unit, when the
conditions below hold; from then on the unit is rigidly fixed at home and cannot be moved or taken off. In every case
"at rest" means slower than 5 mm/s and 2 deg/s relative to the palm base for 50 consecutive ticks (0.5 s); the unit may
still be held by the gripper.

* **A finger module** (three places side by side, any module in any place). Its base servo slides along its channel in
  the palm base (open above, a wall on either side, the floor below) up to the wall at the channel's end; the channel is
  35 mm long. In the collision model the channel is 0.5 mm wider on either side than the base, so contact alone does not
  centre the base (the real play is 0.195 mm a side).
  - *Guided.* The cell starts to guide the base when it is at least 8 mm inside the channel and still more than 3 mm
    from the end wall, within 0.5 mm of the channel's centre line, within 3 deg of the channel in turn about the
    vertical and in pitch, and on the floor (at most 1 mm above its seated height and at most 0.28 mm below it: the
    floor gives a little under a base that is pressed down). It then holds the base sideways within 0.195 mm of the
    centre line and within 0.72 deg about the vertical, moving it there at no more than 10 mm/s and 10 deg/s. That is
    all it does: the base stays free along the channel, upwards, in pitch and in roll. Nothing carries the module, holds
    it level or pulls it in; the floor, gravity, what the chain rests on and the robot do.
  - *Let go.* The cell stops guiding when less than 7 mm of the base is inside, when the base is more than 2 mm off the
    floor or when it is tilted more than 6 deg.
  - *The last 3 mm.* A base that is first found less than 3 mm from the end wall (lowered there from above) is not
    guided and cannot be fastened: on the real hand it would land on the ends of the wall screws, which stand into the
    channel there.
  - *Fastened* at rest while it is guided, its end face within 0.5 mm of the end wall and within 0.5 mm of the floor,
    every joint of the module inside the range it has once mounted at that place (`kit.json`), and the two order rules
    below hold. The joints of a fastened module stay free and get their mounted ranges.
* **The thumb module**. Its base stands against a wall of the palm base, open above. The cell does not guide it at all.
  It can be fastened once the base has been within 10 mm of that wall while still more than 3 mm from it, no further
  from home along the wall than 0.95 mm towards the fingers' side or 1.05 mm towards the wrist side, at most 1 mm above
  home height and within 3 deg of its home attitude; it loses that standing more than 12 mm from the wall, 2 mm off home
  height or 6 deg off. It is fastened at rest with its base within 0.5 mm of the wall and 0.5 mm of home height, its
  joints inside their mounted ranges, the order rules holding. Nothing of the palm base lies under the thumb module's
  base: none of 113 vertical rays cast down from the base's underside (a grid of 2 mm) meets a palm-base piece, and
  lowered from home by up to 20 mm the base slides along its wall without meeting a floor; the module rests on the
  fixture's thumb pad.
* **Electronics stack, upper shell, dorsal cover, power hub** (each comes down onto its seat).
  - *Centred.* When the part is within 2 mm of home across the seating direction, within 5 deg of its home attitude and
    at most 3 mm above home, and moves slower than 50 mm/s and 30 deg/s relative to the palm base, the cell centres it:
    it moves it across the seating direction and in angle towards home at no more than 10 mm/s and 10 deg/s (the screws
    finding their holes). It never moves the part along the seating direction: the part comes down by its own weight or
    by what your program does. While it centres a part that has come to rest it can raise it by up to 0.19 mm (the
    screws' tips riding up in their holes).
  - *Let go* beyond twice those values (4 mm, 10 deg, 6 mm).
  - *Fastened* at rest once centred, within 1.5 mm and 3 deg of home and at most 0.5 mm above it, while its seat and the
    parts fastened under it carry at least 0.5 of its weight along the seating direction over those 50 ticks, the order
    rules holding. The gripper and friction on side faces do not count as carrying: a part that hangs between side faces
    or is held still above its seat is not fastened; one pressed onto its seat while it is still held is; and one wedged
    on sloped faces by a sideways push counts as carried.
  - Measured over twelve complete builds in this cell: at the tick they were fastened the parts lay at most 0.20 mm (the
    electronics stack), 0.24 mm (the upper shell, which can come to rest on one point of its seat), 0.01 mm (the dorsal
    cover) and 0.00 mm (the power hub) above home, and the cap 0.22 mm.
  - The upper shell, the dorsal cover and the power hub go on one way round only (half a turn off they are outside the
    attitude window). The electronics stack also seats half a turn about the vertical away from home; fastened that way
    it counts nothing in the score and cannot be taken off, and for the order rules it is fastened.
* **Press-fit cap**. No screws: the cap's rim is a press fit in its opening. In the collision model the opening is 0.3
  mm wider all round than the rim; the fit is this rule instead.
  - *In the fit.* When the rim is within 0.3 mm of home across the pressing direction, within 3 deg and at most 7 mm
    above home, the cell holds the cap across that direction and in tilt, and the fit resists motion along it with 3 N
    of dry friction (the cap weighs 0.1 N). The cap stays where it is put; let go after a push it comes back by about 5
    thousandths of a millimetre per newton it carried.
  - *Let go* beyond twice those values. *Fastened* at rest within 0.5 mm of home, the first order rule holding. Like the
    three parts above, the cap goes in one way round only: half a turn off it is outside the attitude window.

**The two order rules.**

1. *From the inside out.* A unit is not guided, centred or fastened, even at home, until certain other units are
   fastened: those that it bears on or is screwed onto, and those whose screws, or whose way onto the hand, it would
   close off once in place. Until then the cell ignores it.
2. *The screwdriver must reach the screws.* A unit is fastened only while no other unit, fastened or loose, lies in the
   way of the screwdriver: for each screw of the unit, a straight path 2.5 mm in diameter from the screw's head outwards
   along the screw, 25 mm long or as long as the part the screw sits in leaves free, and where the screw is not
   vertical, from the end of that path straight up to 242 mm above the bench. The unit itself, the units it is screwed
   onto and the robot do not count. A unit in the way stops the fastening without any message for as long as it lies
   there; the cap has no screws and no such path.

Both follow from where the screws are; the picture manual shows every screw head.

## Collision model

The cameras show the units as built (the triangles of the model files). The physics collides convex pieces computed from
the same shapes (135 pieces for a finger module, 140 for the thumb module, 1408 for the palm base). The pieces are not
given. `kit.json` gives, per kind, the box of the collision pieces beside the box of the drawn shape, both in the model
file's frame. Where the two differ:

* **Drawn but not colliding**: every screw (those in the palm base's walls too, whose ends stand into the finger
  channels); the small wire clips; the pins and small bearings of the fingers' last joints; the servos' own case screws
  and connectors. A unit can therefore pass where only a screw head is drawn.
* **A servo collides as one closed solid** (case, output boss and rear ring in five convex slabs), which follows the
  drawn servo within 0.4 mm.
* **Two fits are opened to slots**: a finger channel is 0.5 mm wider on either side than the base that goes into it, and
  the cap's opening 0.3 mm wider all round than its rim (see "Fastening" for what the cell does instead).
* **Three side faces are set back**: the upper shell's against the finger base servos and against the bottom shell's
  side faces, and the dorsal cover's against the bottom shell's side faces, by 0.05 mm on the collision pieces. The real
  parts are line to line there; set back, these parts come down onto their seats instead of hanging between side faces.
* **Mounted neighbours**: two fingers mounted straight side by side are 2.00 mm apart on the drawn shapes and 1.88 mm on
  the collision pieces (the least gap, measured in bands of 2 mm along the fingers).
* **The thumb against the finger beside it**: how far the mounted thumb's side joint turns towards the fingers depends
  on the index finger. Measured on the collision pieces with the other joints at rest: with the index finger straight
  the thumb's last link meets it at -35 deg; with the index finger's sideways joint at -44.9 deg (swung towards the
  thumb) already at -9.2 deg; with no finger in the way the joint reaches the end of its mounted range in `kit.json`,
  -36.1 deg (the thumb's pieces would meet the palm base at -36.2 deg). The other way the thumb meets the fixture at 38
  deg.
* **Seats**: at home the collision pieces of every unit meet the pieces it seats against, along its way in, with a gap
  of at most 0.01 mm. The thumb module seats against its wall only (see "Fastening").
* **Model files and MuJoCo**: the given module models (`finger_module.xml`, `thumb_module.xml`) carry no collision
  geometry. MuJoCo reads at most 200,000 faces from one STL file: `parts/finger_module.stl` (251,673 faces) and
  `parts/thumb_module.stl` (353,858 faces) have more and do not load as one mesh; the per-link files in the two module
  folders do (the largest has 89,862 faces).

## Safety

An episode in which a rule is broken goes on and keeps 30 % of its score (`cell_config.json`, `scoring`).

* **R1, the robot pressing**: any part of the robot other than its two fingers (the links, the hand) pressing on the
  fixture with the palm base, a tray, the bench or a unit with more than 20 N in any tick; the normal forces of all such
  parts are summed per body pressed on. Every face of the two finger bodies counts as a pad and is exempt.
* **R2, a unit loaded**: for 0.2 s (20 consecutive ticks) above 25 N, read in two ways. (a) The contact force between a
  unit and any one other body (bench, tray, fixture, another unit), normal forces summed per pair; not counted: the
  pads' grip, and every contact inside what is fastened together, that is between units fastened to one another and
  between a fastened unit and the palm base, the fixture, a tray or the bench. (b) For a unit that the cell guides,
  centres, holds in its fit or has fastened: the net force that this hold or fastening carries, together with the unit's
  contacts against the units it is fastened onto, added as vectors (a fastening's own clamping cancels). So a push on a
  fastened unit counts whatever pushes, the pads included.
* **R3, a joint against its stop**: the torque that a joint of a digit module carries against one of its mechanical
  stops, above 0.2 N m for 0.2 s (20 consecutive ticks).
* **R4, a unit too fast**: a unit's centre of mass faster than 3 m/s, the speed of a free fall of 0.46 m. A unit that
  falls from the bench to the floor passes it, so that is a violation.

`step()` returns `safety:robot_contact:...`, `safety:part_force:...`, `safety:joint_stop:...` or `safety:part_speed:...`
when the cell records a violation. The cell also keeps counts for its maintainers (units gripped, let go, let fall);
none of them is a rule or reaches the score: a unit let go above where it comes to rest breaks no rule unless one of the
four above is broken.

## Budgets per episode

1200 s of simulated time, 2400 s of wall clock (camera rendering counts), 120 s of silence at most, 600 camera frames,
at most 500 ticks per `step`.
