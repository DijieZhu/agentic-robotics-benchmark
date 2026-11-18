# Assembly cell: integrator's description (commissioning document)

## Frame, bench, robot

* **Cell frame**: origin at the centre of the bench top, z up, x away from the robot, y to the robot's left. The bench
  top (1.6 x 1.0 m, oak veneer, 0.994 m above the hall floor, on a steel frame with a lower shelf) is z = 0. The cell
  stands in a factory hall, lit by the hall's ceiling lamps and its windows.
* **Robot**: a 7-DoF Panda-class arm (model in `~/given/panda/`), bolted to a surveyed datum: its base at
  (-0.45, 0.00, 0.00) m with its axes aligned with the cell frame (exact). Joint names `panda/joint1..7`; the home
  posture and the joint ranges are in `describe()`. Tool point `tcp`: 0.1034 m along the hand's z (the approach
  direction, along the fingers) from the flange; the fingers close along the hand's y.
* **Gripper**: parallel jaw, 0-80 mm, rubber pads 17 x 16.5 mm, grip force 2500 N/m x squeeze / 2 per finger (<= 60 N).
* **Force / torque sensor** at the flange (hand frame): noise about 0.3 N / 0.02 N m rms per reading, an unknown bias up
  to about 1 N / 0.05 N m that differs between power-ups; it reads the hand's own weight.

## Datum boards (the integrator's survey; exact)

Checkerboards printed on rigid plates. On a board of C x R squares of side s the inner corner (i, j), i = 1 .. C - 1,
j = 1 .. R - 1, sits at ((i - C / 2) s, (R / 2 - j) s) in the board's own frame (x along the C squares, y along the R
squares, z out of the printed face); the square at the board's (-x, +y) corner is black.

| board | squares | where (exact) |
|---|---|---|
| bench board | 9 x 6 of 20.0 mm (3 mm plate, 20 mm white margin) | flat on the bench, printed face 3.05 mm above it; inner corners at x = 0.335 + (i - 4.5) x 0.020, y = 0.005 + (3 - j) x 0.020, z = 0.00305 m (cell frame; i = 1..8, j = 1..5); its black corner square at the board's -x, +y corner |
| fixture board | 5 x 4 of 10.0 mm (2 mm plate, 4 mm margin) | vertical on the +x face of the fixture plate, facing +x, its lower edge on the bench, centred on the plate; relative to the fixture plate's centre on the bench (fx, fy, 0): inner corners at x = fx + 0.07255, y = fy + (i - 2.5) x 0.010, z = 0.024 + (2 - j) x 0.010 m (i = 1..4, j = 1..3); the top square at its -y end is black. It moves with the fixture |
| hand plate | 5 x 4 of 7.0 mm (1.5 mm plate, 3 mm margin) | flat on the Panda hand's -x face, facing -x; in the HAND frame inner corners at x = -0.03505, y = 0.050 + (i - 2.5) x 0.007, z = 0.020 + (j - 2) x 0.007 m (i = 1..4, j = 1..3); its black corner square at the plate's -y, -z corner; mounting tolerance about 1 mm / 0.5 deg |

## Fixture and parts

* **Fixture**: a 0.14 x 0.14 x 0.02 m steel plate on the bench, nominal centre (0.10, 0.00) m, placed within +-10 mm
  (not rotated). The arm's base is screwed on it (always the same way round on its pins); nothing else of the arm is
  assembled.
* **Parts**: the fifteen other parts of the arm (the six bus servos of the packing list, the three servo brackets
  and the six printed links) lie loose on the bench around the robot, each resting in a stable orientation on one of its
  faces, all within the robot's reach (0.30-0.62 m from the robot's base axis), outside the space the arm needs while
  it is built, at least about 6 mm apart. Where each part lies, which way it is turned and which face it rests on
  differ between runs. The servos are interchangeable: any of them fits any servo pocket.
* **The servos are unpowered** while the arm is built. A servo's horn turns freely while its servo is loose; once a link
  is fastened on a horn, that joint of the arm turns against the gearbox's back-drive friction (about 0.4 N m) within the
  arm's own joint limits: a push moves the half-built arm and it stays where it was pushed (or sags where gravity beats
  the friction).

## Cameras

| name | type | resolution | nominal horizontal field of view | mount (approximate, not surveyed) |
|---|---|---|---|---|
| `top` | RGB-D | 1280 x 960 | about 70 deg | overhead, roughly 1.25 m above the bench near its middle, looking straight down; the image covers the whole working area (every part on the bench, the fixture, the robot base, the bench board); image width roughly along the cell's x |
| `wrist` | RGB-D | 640 x 480 | about 80 deg | on the Panda hand's +x flank, a few centimetres off the flange axis, tilted towards the fingertips |

The cameras are off-the-shelf units: the lenses have radial and tangential distortion and are NOT calibrated; focal
length and principal point differ from the nominal values by a few per cent between units; the mounts are known only by
the descriptions above; the units are swapped and re-mounted between runs. The depth stream is registered to the
colour pixels (depth along the optical axis, 1 mm steps, range 0.1-4 m, no reading on depth edges and on about 3 % of
the pixels); its noise is about 0.5 mm + 4 mm per m^2 x z^2 for the top camera and about 0.2 mm + 3 mm per m^2 x z^2
for the wrist camera, plus unit-specific systematic errors of up to about a millimetre that do not average away. The
hall's brightness varies by up to about 20 % between runs, and each camera unit has its own slight colour response
(gain and tint within a few per cent), constant during a run; the image noise is new in every frame. The images come from MuJoCo's Filament
renderer (physically based: soft shadows, reflections on metal and plastic) in the hall, whose light is baked into its
textures (`~/render`). The given render layer (`~/given/lib/render`, with `arm.assembly_full.agent_render` and
`arm.assembly_info.cell_render`) renders the same way: the same renderer, hall and look for your own scenes, depth from
MuJoCo on the same geometry (`cell_api.md`, "Rendering your own frames"); it is the way to see what these cameras would
see. MuJoCo's own renderer (`mujoco.Renderer`, or the render layer's `"mujoco"` backend) stays available but draws a
different picture (no hall, MuJoCo's own lights, flat colours, no reflections), so image processing tuned on its
frames may behave differently on the cameras' frames.

## Fastening, safety, budgets

* **Fastening**: the cell does not report any fastening.
  * Servo seat: the servo is fastened ("screwed") when it sits at its design pose in its carrier's pocket within 1.5 mm /
    3 deg for 50 consecutive 10 ms ticks while it moves slower than 5 mm/s relative to the carrier (it may still be
    held).
  * Bracket: the same rule for a bracket on its carrier; a bracket clamps a servo, so it can only be fastened once that
    servo is seated.
  * Horn mate: the fork is fastened wherever it RESTS on the horn of a seated servo: horn axis within 3 deg, axis line
    within 1.5 mm, axial offset <= 1.0 mm, and the fork moving slower than 5 mm/s and turning slower than 2 deg/s
    relative to the servo for 50 consecutive ticks (it may still be held). The cell's screwdriver bolts it at whatever
    rotation about the horn axis it rests at; a bolted fork cannot be mounted again. The mate counts only if that
    rotation is within 3 deg of the design (the calibration pose of the manual); a fork bolted further off counts as not
    mounted.
  * On its feature: a servo or bracket in its pocket within 4 mm across the insertion direction and at most 3 mm short
    of home, or a fork on its horn within 4 mm of the horn axis, 8 deg of its direction and 4 mm axially, is held there
    by the cell (the press fit; the horn boss) while it is not yet fastened, so it stays on when it is let go; it can then
    still be pushed home or turned on the horn. Further out nothing holds a part.
  * Parts can be fastened in any order the arm allows: a sub-assembly (for example the gripper with its servo and jaw)
    may be built on the bench and mounted later; fastened parts move as one body.
* **Collision model**: every servo is ONE box over the whole case footprint (45.4 x 24.8 mm in the servo frame, x from
  -35.2 to +10.2 mm, y +-12.4 mm) from the plane of the rear disc's face (z = -4.0 mm) to the horn-side flange face
  (z = +30.6 mm), shrunk by 0.4 mm on the long sides and the ends and by 0.8 mm on the two faces (so z -3.2 .. +29.8 mm);
  the horn disc and the rear disc themselves are visual only. So a servo resting horn up on the bench has its frame
  origin (z = 0 of the datasheet's CAD frame, the case bottom) about 3.2 mm above the bench, the box's top at about
  33.0 mm and the horn-side flange face at about 33.8 mm above it. The pockets leave about 0.7 mm of play per side, the forks' plates about 2 mm in total
  around the case, a bracket about 0.8 mm around its carrier's faces.
* **The horn-less servo** (the packing list's sixth servo) goes where its joint's link is screwed straight onto the
  output spline; the cell treats a bare spline like a horn (the same fastening rule), and any of the six servos fits
  any servo pocket.
* **Safety** (an episode with a violation scores 0 on every execution criterion): a Panda link other than the fingertip
  pads pressing more than 30 N on the fixture, the bench or a part; a part faster than 3 m/s; a part pressed on the
  environment or on another part with more than 150 N for 0.2 s.
* **Budgets per episode**: 1200 s of simulated time, 2400 s of wall clock (camera rendering counts), 120 s of silence at
  most, 600 camera frames.
