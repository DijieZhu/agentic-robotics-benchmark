# Strut assembly cell: integrator's description (commissioning document)

## Frame, bench, robot

* **Cell frame**: origin at the centre of the bench top, z up, x away from the robot, y to the robot's left. The bench
  top (1.6 x 1.0 m, light-grey laminate, 0.994 m above the floor, on a steel frame) is z = 0. The cell stands in a
  university laboratory, with other benches and stools around it, lit by the room's ceiling lamps.
* **Robot**: a 7-DoF Panda-class arm (model in `~/given/panda/`), bolted to a surveyed datum: its base at
  (-0.45, 0.00, 0.00) m with its axes aligned with the cell frame (exact). Joint names `panda/joint1..7`; the joint
  ranges are in `describe()`. Tool point `tcp`: 0.1034 m along the hand's z (the approach direction, along the fingers)
  from the flange; the fingers close along the hand's y. Its controller is `arm.assembly_cell.controller` of the given
  library; gravity is compensated.
* **Home posture** (`home_q` in `describe()`): at the start of an episode the arm is folded back over its base, the hand
  pointing down with the tool point about 0.43 m above the bench at x = -0.20 m and the gripper open, so that the top
  camera sees the cradle and both trays.
* **Gripper**: parallel jaw, 0-80 mm, the hand model's stock flat rubber pads (17 x 16.5 mm), grip force 2500 N/m x
  squeeze / 2 per finger (<= 60 N). The friction coefficient is 1.0 on the pads and 0.8 between modules, trays, cradle
  and bench.
* **Force / torque sensor** at the flange (hand frame): noise about 0.3 N / 0.02 N m rms per reading, an unknown bias up
  to about 1 N / 0.05 N m that differs between power-ups; it reads the hand's own weight.

## Datum boards (the integrator's survey; exact)

Checkerboards printed on rigid plates. On a board of C x R squares of side s the inner corner (i, j), i = 1 .. C - 1,
j = 1 .. R - 1, sits at ((i - C / 2) s, (R / 2 - j) s) in the board's own frame (x along the C squares, y along the R
squares, z out of the printed face); the square at the board's (-x, +y) corner is black.

| board | squares | where (exact) |
|---|---|---|
| bench board | 9 x 6 of 20.0 mm (3 mm plate, 20 mm white margin) | flat on the bench, printed face 3.05 mm above it; inner corners at x = 0.335 + (i - 4.5) x 0.020, y = 0.005 + (3 - j) x 0.020, z = 0.00305 m (cell frame; i = 1..8, j = 1..5); its black corner square at the board's -x, +y corner |
| cradle board | 5 x 3 of 10.0 mm (2 mm plate, 4 mm margin) | flat on the cradle plate's top near its +x edge, printed face up, its five squares along the cell's y; relative to the cradle's reference point (cx, cy): inner corners at x = cx + 0.152 + (j - 1.5) x 0.010, y = cy + (i - 2.5) x 0.010, z = 0.02205 m (i = 1..4, j = 1..2); its black corner square at the board's -x, -y corner (cell axes). It moves with the cradle |
| hand plate | 5 x 4 of 7.0 mm (1.5 mm plate, 3 mm margin) | flat on the Panda hand's -x face, facing -x; in the HAND frame inner corners at x = -0.03505, y = 0.050 + (i - 2.5) x 0.007, z = 0.020 + (j - 2) x 0.007 m (i = 1..4, j = 1..3); its black corner square at the plate's -y, -z corner; mounting tolerance about 1 mm / 0.5 deg |

## Cradle and strut cores

* **Cradle**: a dark steel plate 0.36 x 0.18 x 0.02 m on the bench with three black clamp seats side by side, 0.100 m
  apart along x. Its reference point (cx, cy) is the plate's centre, which is also the middle seat's place; nominally
  (0.01, 0.00) m, placed within +-10 mm in x and in y, never turned. The seats are called `s1` (x = cx - 0.100, nearest
  the robot), `s2` (x = cx) and `s3` (x = cx + 0.100).
* **Strut cores**: one strut core (the electronics bay's base half with its boards, on the aluminium rod) is clamped in
  each seat, all three the same way round: the rod along the cell's y with its axis 0.075 m above the bench at the
  seat's x, the bay's open side (the lid's seat) facing straight up. A rod's two ends are called `p` (towards +y) and
  `n` (towards -y). A seat clamps only the bay's base: a saddle under it and two side jaws, 131 mm long, whose tops
  are 8 mm below the rod axis and whose outer faces are 20.0 mm (towards -x) and 26.4 mm (towards +x) from it. The
  cradle touches no rod, and no motor module, lid or end cap of a finished strut. The survey does not say where along
  y a core sits in its seat (where its rod ends are): measure it.

## Parts trays and nests

Two aluminium plates, 10 mm thick, carry the fifteen loose modules in nests. The nests' seats, stops and pocket walls
are blue, close in hue to the blue covers (the plates are bare metal). Tray frame: origin on the plate's top face at
its centre, z up, x across the nests, y along the modules' axes; a tray's -y side faces the cradle.

| tray | plate | nominal place (cell frame) | nests |
|---|---|---|---|
| `tray_l` (left of the robot) | 0.66 x 0.18 m | centre (-0.33, +0.36) m, turned 0 deg about the vertical | 3 lid, 3 cap, 3 motor |
| `tray_r` (right of the robot) | 0.46 x 0.18 m | centre (-0.30, -0.36) m, turned 180 deg about the vertical | 3 cap, 3 motor |

`describe()["trays"]["trays"]` lists, per tray, every nest with its name, its kind and its place `x_in_tray_m`,
`y_in_tray_m` in its tray's frame (metres; the nests never move on their trays, so these are the same in every episode;
`describe_example.json` shows the reply): the motor nests on one line (y = -0.025 m) 0.15 m apart, the cap nests on a second line (y = +0.055 m) between them in x, the three lid nests side
by side 0.06 m apart at the left tray's -x end. Every nest lies 0.30-0.62 m from the robot's base axis.

* **Motor nest**: two pairs of 45 deg V plates under the two ends of the module's enclosure, each 10 mm long along y,
  with 30 mm between them left free; an end stop behind the outer end face. The module lies with its bore axis
  horizontal along the tray's y, 30.0 mm above the plate's top face, its outer end towards +y and its inner end (where
  the battery lead sticks out) towards -y; in its model file's frame the point x = -18.0 mm of the bore axis is over the
  nest's (x, y). Its collision shape reaches 51 mm above the plate.
* **Cap nest**: a V seat tilted back 25 deg under the dome, a stop against the cap's rear step and a guard under its
  rear cylinder. The end cap lies with its bore axis horizontal along the tray's y, 25.8 mm above the plate's top face,
  the bore's mouth towards -y; its model file's origin is within 1 mm of the nest's (x, y). It reaches 43.3 mm above
  the plate.
* **Lid nest**: a shallow pocket with walls 4 mm high, 1.5 mm clear of the lid all round. The lid lies on its rim,
  outer face up, its long axis along the tray's y, centred on the nest's (x, y); it reaches 17.8 mm above the plate.

A motor module from the left tray reaches a `p` rod end, and one from the right tray an `n` rod end, without a half
turn about the vertical (its inner end already faces the bay; only the tray's own turn has to be taken out); the same
holds for the end caps' mouths.

## What changes from episode to episode

1. The cradle's place: within +-10 mm in x and in y of its nominal place; never turned.
2. Each tray's place and turn, on its own: within +-20 mm in x and in y of its nominal place and within +-10 deg of its
   nominal turn about the vertical.
3. Which motor module lies in which of the six motor nests (the three with a cable guide ring and the three without are
   shuffled over both trays), and which end cap in which of the six cap nests: the covers' colours are in different
   nests every time and can only be read from the images. The three lids are always in the three lid nests.
4. Each motor module's turn about its bore axis. The nests do NOT hold a motor module at its mounting turn: its turn in
   the nest differs from the mounting turn by an amount that changes from nest to nest and from episode to episode and
   is not documented, and the mounting turn itself is not the same at the two ends of a rod. Measure the turn.
5. Each end cap's turn about its bore axis: any (it does not matter for the mount).
6. Each lid, on its own: one way round, or turned end for end (180 deg about the vertical), with equal chance.
7. The cameras (mounts, focal length, principal point, lens distortion, colour response, the depth units' systematic
   errors), the room's brightness, the force sensor's bias and the hand plate's mounting (see below).

When the cell sets up an episode it checks the drawn layout: every loose module can be gripped from above where it
lies, with every other module in its nest; a layout that fails is drawn again. The robot's base, the bench board, the
cores' seats in the cradle and the nests' places on their trays never change.

## Cameras

| name | type | resolution | nominal horizontal field of view | mount (approximate, not surveyed) |
|---|---|---|---|---|
| `top` | RGB-D | 1280 x 960 | about 70 deg | overhead, roughly 1.25 m above the bench near its middle, looking straight down; the image covers the whole working area (both trays, the cradle with the three struts, the robot base, the bench board), about 0.73 pixels per mm on the bench top; image width roughly along the cell's x |
| `wrist` | RGB-D | 640 x 480 | about 80 deg | on the Panda hand's +x flank, a few centimetres off the flange axis, tilted towards the fingertips |

The cameras are off-the-shelf units: the lenses have radial and tangential distortion and are NOT calibrated; focal
length and principal point differ from the nominal values by a few per cent between units; the mounts are known only by
the descriptions above; the units are swapped and re-mounted between runs. The depth stream is registered to the
colour pixels (depth along the optical axis, 1 mm steps, range 0.1-4 m, no reading on depth edges and on about 3 % of
the pixels); its noise is about 0.5 mm + 4 mm per m^2 x z^2 for the top camera and about 0.2 mm + 3 mm per m^2 x z^2
for the wrist camera, plus unit-specific systematic errors of up to about a millimetre that do not average away. The
room's brightness varies by up to about 20 % between runs, and each camera unit has its own slight colour response
(gain and tint within a few per cent), constant during a run; the image noise is new in every frame.

* **The rods are almost invisible to the depth stream.** A rod (3.175 mm) is only 2-3 pixels wide in the top view; on
  its two free lengths the depth stream mostly has no reading (in a check frame 5-12 % of the pixels along them read
  the rod, none over the last 10 mm at a rod end). The colour image shows the rods and their ends.
* The three cover colours (red, green, blue) are clearly apart in hue in both cameras.
* The images come from a physically based renderer of the laboratory (soft shadows, reflections on metal and plastic).
  Hardware that cannot be seen from any direction because it is enclosed in a module is left out of the images (the
  model files contain it). The given library holds the camera pipeline itself (`~/given/lib/render`: the camera model
  with lens distortion, the photo and depth-sensor models, the checkerboards) and the cameras' renderer:
  `arm.assembly_strut.agent_render` pictures a twin of yours with the same renderer, laboratory (`~/render`) and look
  as the cell's cameras, `render.filament.view` any MuJoCo scene of yours in that laboratory (`cell_api.md`,
  "Picturing a model of your own"); it holds no model of this cell (`arm.assembly_six` and
  `arm.assembly_info.cell_render` build an earlier cell). Of the given library, this cell's own parts are the proxy
  (`arm.assembly_info.client`, on `arm.assembly_cell.client`), the arm's controller (`arm.assembly_cell.controller`) and
  this camera pipeline; the rest of `arm.assembly_cell` (its `server`, `cell_builder`, `layout` and the others) and of
  `arm.assembly_info` came with an earlier cell and describes none of this cell's modules, trays or nests.

## Fastening (the cell's rules; the cell reports no fastening)

"Design place" and "design turn" below are the place along the rod and the turn about it that the picture manual shows
for a motor module; they are not documented anywhere else.

* **Onto the rod (motor module, end cap)**. The cell takes a module onto a rod end once the leading mouth of its bore
  is within 2.0 mm of the rod's axis, its bore axis within 5 deg of the rod's and the rod end at least 3 mm inside the
  bore. A motor module's bore goes through, so it is taken on either way round; an end cap only mouth first. An end cap
  is taken on only when the motor module of that rod end is fastened and counts.
* **On the rod**. The cell then pulls the module towards the rod's axis (at no more than 10 mm/s and 10 deg/s). The
  pull keeps the leading mouth of the bore near the axis (a motor module taken on end for end: the other mouth) but
  does not keep the module level: let go before it is fastened, the module tilts about that mouth until its bore rests
  on the rod (with the rod through the whole bore by about 2-4 deg, at some turns about 8 deg) and the rod's friction
  (0.8) holds it; its turn moves by up to about 30 deg as it settles and then stays (at some turns it creeps on by up
  to about 0.15 deg/s). Held clear of the rod, it slides along the rod and turns about it against light damping only
  (0.5 N s/m, 0.0001 N m s per rad). The cell lets a module go again when the rod end leaves its bore or when it is
  more than twice the capture values (4 mm, 10 deg) off the axis: let go with less than about 40 mm of the rod in its
  bore, a module held by its leading mouth may tilt that far (with less than about 13 mm it always does), and it falls
  off the rod with less than about 14 mm inside (an end cap) or 10 mm (a motor module).
* **Fastened at rest**. A module on the rod is fastened where it rests, at the place along the rod and the turn about it
  that it has then, once it has moved slower than 5 mm/s and turned slower than 2 deg/s relative to its strut for 50
  consecutive 10 ms ticks (it may still be held) while it is inside its fastening window: a motor module within 3.0 mm
  of its design place along the rod, an end cap within 1.5 mm of home (the rod end against the bottom of its bore).
  Outside the window nothing is fastened and the module can be moved on. A fastened module cannot be moved again.
* **What counts**. A fastened motor module counts only when it is within 1.5 mm of its design place AND within 5 deg
  of its design turn (the wrong way round is 180 deg off). One fastened between 1.5 and 3.0 mm off, or more than 5 deg
  off, counts nothing and cannot be re-fitted; the end cap of its rod end and the lid of its strut are then never taken
  on, because both need a motor module that counts. A fastened end cap counts when it is the first cap fastened on its
  strut or has the same cover colour as that first cap; a second cap of another colour is fastened and counts nothing.
  An end cap's turn about the rod is free. Any motor module fits either end of any rod, and any strut may take any of
  the three colours.
* **Lid**. The cell takes a lid onto a bay only when both motor modules of that strut are fastened and count, and the
  lid is within 4 mm of its seat across the seating direction (straight down), at most 3 mm short of home and within
  8 deg of its seated orientation; a lid turned end for end is not taken. The cell pulls it home (no more than
  10 mm/s and 10 deg/s) and fastens it when it rests within 1.5 mm / 3 deg of home for 50 ticks (slower than 5 mm/s and
  2 deg/s); it lets it go beyond twice the capture values. A fastened lid always counts.

## Collision model

The cameras show the modules as built (the triangles of the model files). The physics (MuJoCo, 1 ms steps, 10 per
control tick) collides convex pieces computed from the same shapes, and differs from the look where it matters:

* **Bores are clear channels.** No collision piece of a motor module or of an end cap comes closer than 3.6 mm to its
  bore axis: the bore is a clear channel 7.2 mm in diameter (the real bore is 3.4 mm, for the 3.175 mm rod), along the
  whole motor module and, in an end cap, down to the bore's bottom, which is kept: a rod end stops against it. The rod
  collides as its own shape (a 36-sided prism, 3.175 mm across corners, 304.8 mm long). So contact alone does not
  centre a module on the rod: until the cell takes it on (the rule above), a module can hang up to 2.0 mm off the rod's
  axis.
* **Three things are drawn but do not collide**: (1) the battery lead that sticks 36 mm out of a motor module's inner
  end (really a flexible wire; with the module at its place the drawn lead passes through the bay's base and board);
  (2) the thin tubes around the bores inside a motor module and inside an end cap (4.0 mm in outer radius);
  (3) hardware enclosed in a module (not visible from outside), and fragments smaller than 1 mm^3 on the boards (pins,
  solder joints).
* **How closely the pieces follow the surface**: within 1 mm on 98 % of a motor module's exposed surface and on 92 % of
  an end cap's. The strut core's boards, connectors, screws and inserts are one convex hull each. The lid's pieces are
  cut back by 0.4 mm wherever the strut core's pieces would meet them on the way down, so that the lid goes onto the
  bay without touching the rod or the boards; they follow its outer top face (88 % within 1 mm, 99.6 % within 3 mm) and
  its side faces (75 % within 1 mm).
* **Stops**: a motor module pushed inwards past its design place stops against the end of the bay; an end cap at home
  touches the outer end face of its motor module (the two faces are flush in the finished strut).
* **Fingertip pads**: the stock flat pads of the given hand model (no grooved pads are fitted), with a stiff contact
  and torsional friction (`cell_config.json`, `pads`). Only the pads are exempt from rule R1 below; the fingers' other
  surfaces are not.

## Safety (an episode with a violation scores 0)

* **R1, the robot pressing**: any part of the robot other than the two fingertip pads (links, hand, the fingers' own
  surfaces) pressing on the cradle, a tray, the bench, a module or a strut core with more than 15 N in any tick; the
  normal forces of the whole robot are summed per body it presses on.
* **R2, a module pressed**: a module pressed on the bench, a tray, the cradle, a strut core or another module with more
  than 30 N for 0.2 s (20 consecutive ticks; normal forces summed per pair). Not counted: the pads' grip (no limit),
  and the contacts between a strut core and the modules fastened on it or between two modules fastened on one core.
* **R3, a rod loaded across**: on either side of a bay, the force across the rod above 4.0 N for 0.2 s. It is
  everything that part of the rod carries from outside: the contacts on the rod beyond the bay, and what the modules
  taken onto or fastened on that end pass to it (their weight, a push on them, a neighbour leaning on them). At rest a
  rod end with a motor module on it reads 0.57 N, with the end cap as well 0.72 N.
* **R4, a module too fast**: a module's centre of mass faster than 3 m/s.

`step()` returns `safety:robot_contact:...`, `safety:part_force:...`, `safety:rod_load:<seat>:<end>:...` or
`safety:part_speed:...` when the cell records a violation; the episode goes on and scores 0.

## Budgets per episode

1200 s of simulated time, 2400 s of wall clock (camera rendering counts), 120 s of silence at most, 600 camera frames,
at most 500 ticks per `step`.
