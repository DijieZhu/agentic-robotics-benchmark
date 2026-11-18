# Assembly cell: integrator's description (commissioning document)

## Frame, bench, robots

* **Cell frame**: origin at the centre of the bench top, z up. The bench top (1.6 x 1.4 m, x -0.8..0.8, y -0.7..0.7,
  oak veneer, 0.994 m above the hall floor, on a steel frame with a lower shelf) is z = 0. The cell stands in a factory
  hall, lit by the hall's ceiling lamps and its windows.
* **Robots**: two 7-DoF Panda-class arms (model in `~/given/panda/`), bolted to surveyed datums (exact):
  * `left`: base at (-0.275, -0.476, 0.00) m, turned +60 deg about z (its base x axis points to (cos 60, sin 60, 0));
    joint names `pl/joint1..7`;
  * `right`: base at (-0.275, +0.476, 0.00) m, turned -60 deg about z; joint names `pr/joint1..7`.
  The home postures and the joint ranges are in `describe()`; both start folded at home. Tool point `tcp`: 0.1034 m
  along the hand's z (the approach direction, along the fingers) from the flange; the fingers close along the hand's y.
  The two robots share the bench and can reach each other: they collide with each other like with anything else.
* **Grippers**: parallel jaw, 0-80 mm, rubber pads 17 x 16.5 mm, grip force 2500 N/m x squeeze / 4 per finger
  (<= 70 N per finger).
* **Force / torque sensors** at each flange (hand frame): noise about 0.3 N / 0.02 N m rms per reading, an unknown bias
  up to about 1 N / 0.05 N m that differs between power-ups; they read the hand's own weight.

## The gallows and the pair

* **Gallows**: a steel column on a 250 x 250 x 10 mm base plate standing on the bench (the plate nominally spans x
  0.025..0.275 m, y -0.125..0.125 m), its top arm turned 180 deg about z: the pair frame of `~/given/kit.json`
  (`design.T_world_pair_nominal`) sits 0.698 m above the bench at (0, 0) nominally; each run the gallows stands up to
  10 mm off its nominal place (not rotated). Its top carries the two J1 motors, one
  per arm (screwed on: they are not part of the job); each J1 motor's output flange turns freely until the J1 link is
  fastened onto it.
* **The arms**: every other unit of both arms (`kit.json`: 9 per arm) lies loose on the bench. Which arm a unit belongs
  to is in its id (`l_` left, `r_` right); the two arms are mirror images of each other.

## Datum boards (the integrator's survey; exact)

Checkerboards printed on rigid plates. On a board of C x R squares of side s the inner corner (i, j), i = 1 .. C - 1,
j = 1 .. R - 1, sits at ((i - C / 2) s, (R / 2 - j) s) in the board's own frame (x along the C squares, y along the R
squares, z out of the printed face); the square at the board's (-x, +y) corner is black.

| board | squares | where (exact) |
|---|---|---|
| bench board | 9 x 6 of 20.0 mm (3 mm plate, 20 mm white margin) | flat on the bench beyond the gallows' base plate, printed face 3.05 mm above the bench; its x axis along the cell's +y, its y axis along the cell's -x: inner corners at x = 0.46 - (3 - j) x 0.020, y = (i - 4.5) x 0.020, z = 0.00305 m (cell frame; i = 1..8, j = 1..5) |
| fixture board | 5 x 4 of 10.0 mm (2 mm plate, 4 mm margin) | flat on the gallows' base plate, beside the column's +x side, printed face 12.05 mm above the bench; its x axis along the cell's +y, its y axis along the cell's -x; relative to the gallows' offset (fx, fy) from its nominal place: inner corners at x = fx + 0.235 - (2 - j) x 0.010, y = fy + (i - 2.5) x 0.010, z = 0.01205 m (i = 1..4, j = 1..3). It moves with the gallows. The gallows' top hides it from the overhead camera; the wrist cameras see it |
| hand plates | 5 x 4 of 7.0 mm (1.5 mm plate, 3 mm margin), one on each Panda hand | flat on the hand's -x face, facing -x; in the HAND frame inner corners at x = -0.03505, y = 0.050 + (i - 2.5) x 0.007, z = 0.020 + (j - 2) x 0.007 m (i = 1..4, j = 1..3); its black corner square at the plate's -y, -z corner; mounting tolerance about 1 mm / 0.5 deg |

## The units on the bench

* The 18 units lie loose on the bench, each resting in a stable orientation, outside the space the arms need while they
  are built, at least about 6 mm apart. Each arm's units lie on its half of the bench, within reach of the Panda on that
  side, except the left arm's cable holder B (`l_HB`), which lies on the right (+y) half, within reach of the right
  Panda only. Where each unit lies, which
  way it is turned and which face it rests on differ between runs (the faces they rest on are the same in every run;
  their places and turns vary by a few centimetres and up to about 30 deg).
* **Risers**: the flat units (the J2 links and the four cable holders) lie across two loose aluminium blocks (10 x 40 x
  35 mm each, 40 g, `kit.json` "risers") under their ends, so that a finger reaches under their edges; each J2 motor
  stands on its rim in the gap between two such blocks (a cradle), its axis level. The blocks are loose: they move when
  pushed and stay on the bench.
* **Inner hinges**: a unit's output flange (the rotor of its motor) turns freely on the unit until the next unit is
  fastened onto it; J4_D (the forearm's second support, on the upper arm) turns freely on its bearing; the wrist's J6
  joint turns against its motor's friction. Their angles on the bench are whatever they were left at.

## Cameras

| name | type | resolution | nominal horizontal field of view | mount (approximate, not surveyed) |
|---|---|---|---|---|
| `top` | RGB-D | 1280 x 960 | about 70 deg | overhead, roughly 1.55 m above the bench near (-0.05, 0) m, looking straight down; the image covers the whole bench (every unit, the gallows, both robot bases, the bench board); image width roughly along the cell's x; the gallows' top and the robots hide what is under them |
| `wrist_left`, `wrist_right` | RGB-D | 640 x 480 | about 80 deg | on the left / right Panda hand's +x flank, a few centimetres off the flange axis, tilted towards the fingertips |

The cameras are off-the-shelf units: the lenses have radial and tangential distortion and are NOT calibrated; focal
length and principal point differ from the nominal values by a few per cent between units; the mounts are known only by
the descriptions above; the units are swapped and re-mounted between runs. The depth stream is registered to the
colour pixels (depth along the optical axis, 1 mm steps, range 0.1-4 m, no reading on depth edges and on about 3 % of
the pixels); its noise is about 0.5 mm + 4 mm per m^2 x z^2 for the top camera and about 0.2 mm + 3 mm per m^2 x z^2
for the wrist cameras, plus unit-specific systematic errors of up to about a millimetre that do not average away. The
hall's brightness varies by up to about 20 % between runs, and each camera unit has its own slight colour response
(gain and tint within a few per cent), constant during a run; the image noise is new in every frame. The images come
from MuJoCo's Filament renderer (physically based: soft shadows, reflections on metal and plastic) in the hall, whose
light is baked into its textures (`~/render`). The given render layer (`~/given/lib/render`, with
`arm.assembly_openarm.agent_render`) renders the same way: the same renderer, hall and look for your own scenes, depth
from MuJoCo on the same geometry (`cell_api.md`, "Rendering your own frames"); it is the way to see what these cameras
would see. MuJoCo's own renderer (`mujoco.Renderer`, or the render layer's `"mujoco"` backend) stays available but
draws a different picture (no hall, MuJoCo's own lights, flat colours, no reflections), so image processing tuned on
its frames may behave differently on the cameras' frames.

## Fastening, joints, collisions

* **Fastening**: the cell does not report any fastening.
  * On its feature: a unit within 4 mm across its feature and 8 deg of its design orientation (a flange: on the motor's
    axis within 4 mm, its face within 4 mm along the axis; the J2 motor in J1 link's yoke at most 3 mm short of home)
    is held there by the cell (the dowels; the press fit), its rotation about a flange's axis left free, so it stays on
    when it is let go and can still be pushed home or turned. Further out nothing holds a unit.
  * Fastened ("screwed"): when the unit rests within its tolerance (`kit.json`: 1.5 mm / 3 deg; a flange's faces within
    1.5 mm along the axis) for 50 consecutive 10 ms ticks, moving slower than 5 mm/s and turning slower than 2 deg/s
    relative to its carrier (it may still be held). A flange is bolted at the bolt position nearest its rotation about the
    axis (every 60 deg); it counts only on its design bolt position, where the rotor's yellow index mark meets the part's
    (the manual's close-up); a flange bolted elsewhere is screwed on but counts as not mounted. A fastened unit cannot be
    mounted again.
  * The forearm fastens onto J4's flange and onto J4_D in one go (`kit.json` "also_fastens"): J4_D, free on its bearing,
    must be within the tolerance of its place on the forearm as well.
  * Prerequisites (`kit.json` "requires"): the J2 link only after the J2 motor is seated. Parts can be fastened in any
    other order the arm allows; fastened units move as one body.
* **Joints**: J1, J2 and J3 of each arm hold the angle they were fastened at while the pair is built (position-held
  motors, at most 10 / 10 / 5 N m). J4, J5, J6 and J7 are unpowered and back-drive against their friction (J4: about
  0.1 N m; J5, J6, J7: about 0.04 N m; `kit.json` "joints"), within the arm's joint limits; J4_D's bearing turns with
  about 0.02 N m. A push moves an unpowered joint and it stays where it was pushed (or sags where gravity beats the
  friction).
* **Collision model**: every unit collides as its convex pieces (`~/given/collision/<unit>/`, in the unit's frame);
  the given looks (`~/given/parts/`) are drawn only. Contacts are off only between the two bodies of a mate (the
  bolted interface). The risers are boxes, the bench top a box, the robots their model's collision shapes.
* **Safety** (an episode with a violation scores 0 on every execution criterion): a Panda link other than its fingertip
  pads pressing more than 30 N on the gallows, the bench, a unit or the other Panda; a unit faster than 3 m/s; a unit
  pressed on the environment or on another unit with more than 150 N for 0.2 s.
* **Budgets per episode**: 1800 s of simulated time, 3000 s of wall clock (camera rendering counts), 120 s of silence at
  most, 900 camera frames.
