# The tolerances, the fit, and where each number comes from (maintainer notes, 2026-11-17)

Round 3 of the maintainer review. After round 2 the agent rebuilt the cell from its data sheet as a twin of its own
and developed there with the twin's ground truth: the drawing was the cell. A real cell is not its drawing. From this
round on every episode, graded or commissioning, is a real kit on the cell as it is that day: `keyed_env.sample_unit`
draws each value below from the episode's seed, uniformly within its range and independently, and the observations
never carry it. The data sheet (`environment/setup/payload/datasheet.md`, section 11, and `cell_data.TOL`) states
every range, so nothing is hidden; what a twin built from it can do is randomise. `tests/test_keyed_nut_peg_assembly_commissioning.py`
checks that the simulator's table is the data sheet's.

Sizing rule: every range is what real parts and sensors of this class have. Where a standard, a data sheet or a
paper gives the number, it is cited, and every citation below was checked on 2026-11-17 against a text that could be
read ("Sources as read", at the end; for the two ISO standards that is a public table of their values, since the
standards themselves are not free to read). Where no such text was read, the number is called an engineering
estimate and the note says what it rests on. Nothing here is cited from memory.

## Every nut

| value | range | source |
|---|---|---|
| each of the 24 sector boundaries of the hole wall (the channels' walls) | +-1 degree | ISO 2768-1:1989, limit deviations for angular dimensions, classes f and m: +-1 degree where the shorter side is up to 10 mm (source 1; the shorter side of a channel wall is its 6.2 mm depth). It equals +-0.22 mm at the hole radius, about the same table's linear +-0.2 mm for 6 to 30 mm. |
| each of the 8 inner layer boundaries (the channels' floors and ceilings) | +-0.2 mm | ISO 2768-1:1989, linear dimensions, class m (source 1): +-0.1 mm up to 6 mm, +-0.2 mm over 6 up to 30 mm, +-0.3 mm over 30 up to 120 mm. The boundaries lie 5 to 40 mm above the bottom face; all eight take the middle value. |
| hole flats from the axis | +-0.1 mm | the same table, class m, on the 25.6 mm hole: +-0.2 mm on the diameter |
| groove floor from the axis | +-0.2 mm | the same table, class m, over 6 up to 30 mm (a 19 mm radius) |
| the handle's direction against the wall's sector grid | +-0.5 degrees | ISO 2768-1:1989 angular (source 1), classes f and m, shorter side over 10 up to 50 mm (the handle is 16 mm wide): +-30 minutes |
| colour, each RGB component | +-0.04 | engineering estimate: batches of one nominal colour differ visibly in shade; +-0.04 (about 10 of 255) is enough to stop a controller from keying on one exact colour value and small against the distance between the eight colours |
| sliding friction against a peg | 0.3 to 0.7 | The Engineering ToolBox's friction table (source 4): aluminium on mild steel, clean and dry, 0.61 static and 0.47 sliding; the lower end for handling residue and oil films (the same table: steel on steel 0.16 with grease, 0.23 with light mineral oil). Before round 3 the nut's coefficient (0.6 to 1.0) never acted on a peg: MuJoCo uses the element-wise maximum of the two geoms' coefficients (source 5) and the pegs had 1.0. The pegs now have 0.1, so a nut on a peg slides with the nut's own value. |
| mass | 0.05 to 0.15 kg | unchanged (the contribution's range) |

## Every peg

| value | range | source |
|---|---|---|
| shaft radius | +-0.05 mm | ISO 2768-1:1989 linear, class f (source 1): +-0.1 mm for over 6 up to 30 mm, on the shaft's diameter |
| pin reach from the shaft axis | +-0.2 mm | the same table, class m, over 6 up to 30 mm (17.2 mm) |
| pin width and height | +-0.1 mm each | the same table, class m, up to 6 mm (3 mm and 4 mm) |
| pin centre above the collar top | +-0.3 mm | the same table, class m, over 30 up to 120 mm (37.5 mm) |

## The cell on the day

| value | range | source |
|---|---|---|
| finger pads: sliding friction against a nut | 0.6 to 1.2 | rubber pairs in The Engineering ToolBox's table (source 4): 0.6 to 0.85 sliding on dry concrete, 0.9 static and 0.5 to 0.8 sliding on dry asphalt, 1 for a dry tyre on a dry road, 1.16 rubber on rubber. The robosuite gripper model's 2.0 is above every entry of that table (its largest is 1.6); worn or dusty pads sit at the lower end. The pads' torsional coefficient is scaled in proportion (the model's 0.05 m at 2.0). |
| each arm joint's friction | the Panda's identified curve x 0.6 to 1.4 | the curve: C. Gaz, M. Cognetti, A. Oliva, P. Robuffo Giordano, A. De Luca, "Dynamic identification of the Franka Emika Panda robot with retrieval of feasible parameters using penalty-based optimization", IEEE Robotics and Automation Letters, 2019, supplementary material, Table I (source 6; phi1 = 0.546, 0.872, 0.641, 1.279, 0.839, 0.303, 0.565 N m; phi2, phi3 as in `keyed_env.JOINT_FRICTION`). The cell's tracking law is operational-space control with gravity and Coriolis compensation only, so this friction is not compensated. The x 0.6 to 1.4 spread is an engineering estimate for unit-to-unit differences and warm-up; no source is cited for it. What it does was measured: at this controller's stiffness (data sheet, section 5) it leaves the arm some tenths of a millimetre short of a small step. |
| wrench zero: offset | +-0.5 N, +-0.05 N m per axis | unchanged (the contribution's values) |
| wrench zero: drift | up to +-0.3 N and +-0.012 N m per minute per axis | engineering estimate, sized against a wrist sensor's product sheet (source 7: Robotiq FT 300-S: signal noise 0.1 N and 0.005 / 0.003 N m; recommended thresholds for contact detection 1 N and 0.02 / 0.01 N m). The sheet gives no drift figure. Over one takt the zero moves by up to 0.43 N and 0.017 N m: about half of that sheet's force threshold and about its torque thresholds, the order of what a sensor of this class resolves. |
| fixed cameras: position, orientation | +-2 mm per axis, +-0.2 degrees about each axis | engineering estimate; no standard or paper is cited. It rests on the cameras' own resolution: a fixed camera has a focal length of 257 pixels and looks at the work area from 0.92 m, so one pixel is 3.6 mm there; +-0.2 degrees is 0.9 pixel and +-2 mm is 0.6 pixel. A calibration made from these cameras' images of a board cannot place a camera much better than a fraction of a pixel, and a cell is calibrated when it is installed, not every day (mounts creep, brackets warm up): about one pixel is taken as what it holds in between. |
| wrist camera on the hand: position, orientation | +-1.5 mm per axis, +-0.3 degrees about each axis | the same estimate: focal length 125 pixels, so at 0.25 m one pixel is 2 mm; +-0.3 degrees is 0.65 pixel and +-1.5 mm is 0.75 pixel there |
| every camera: focal length, principal point | +-0.5 %, +-1 pixel per axis | the same estimate: +-0.5 % of the focal length moves a point at the image's corner by 1 pixel (fixed cameras) or 0.8 pixel (wrist camera) |
| lamps: brightness | +-15 % | engineering estimate: lamp ageing and daylight through a shift |

What they add up to (also stated in the data sheet): a nut on the pin rests at 39.5 +- 0.35 mm, stalls at 24.5 and
9.5 +- 0.55 mm; riser middles are 30, 45 or 60 +- 2 degrees apart; a nut left free on the pin drops into a riser
within +-4.5 to +-7 degrees of the riser's real middle (measured, 18 nuts; +-4 to +-5.5 on the drawing's nut). With the
published calibration alone a point on the table is typically 4 mm off in a fixed camera (p90 8 mm, at worst 16 mm) and
2 mm off in the wrist camera from 0.25 m (p90 3.5 mm, at worst 6 mm).

## The fit (amendment 3)

Tolerances and a 100 s takt were not enough (below, and `commissioning.md`): the contribution's hole was 25.6 mm on a
22 mm shaft, 1.8 mm of radial clearance, and that play absorbed every tolerance and most of the cameras' error. A
keyed nut is a locating part. From amendment 3 on:

| what | value | source |
|---|---|---|
| the fit | shaft 24.6 mm in the 25.6 mm hole: 1 mm on the diameter, 0.5 mm of radial clearance on the drawing, 0.35 to 0.65 mm with the tolerances of the hole (+-0.1 mm) and the shaft (+-0.05 mm) | the series of clearance holes for metric screws (fine, medium, coarse; tabulated under EN 20273 in source 2, and the subject of ISO 273): fine 23 mm for a 22 mm screw and 25 mm for 24 (the screw + 1 mm), medium 24 and 26 (+ 2 mm), coarse 26 and 28 (+ 4 mm). The contribution's pair (22 in 25.6) was between medium and coarse. A fit from the limits-and-fits system would be tighter still: its loosest running fit, H11/c11, has 0.11 to 0.37 mm on the diameter at 18 to 30 mm (source 3). The fine clearance series is what parts made to general tolerances can hold. |
| lead-in | none: the shaft end and the hole's edges are square | the geometry as contributed (a cylinder and a prism); the data sheet says so |
| the shaft, not the hole, was changed | | the nut and its groove are untouched: the inner wall blocks, the pin's reach and the window in which a nut drops into a riser depend on the hole's radius, not on the shaft's (measured: +-4 to +-5.5 degrees on the drawing's nut with either shaft) |
| the hand's fingers | one mechanism: the two finger slides are coupled (an equality constraint, `keyed_env.FINGER_COUPLING`) | the real hand has one degree of freedom: its interface commands and reports a single opening width, and its maker's robot description makes the second finger joint mimic the first (source 8). The robosuite model drives each finger on its own, both capped at 20 N: a gripped part then has nothing holding it along the closing direction but the slides' damping, and a sideways force of 2 or 3 N walks nut and fingers through the hand (measured: 10 to 15 mm during a search on the shaft end). With 1.8 mm of clearance that did not show; with a locating fit it would make a search by touch impossible while the reference, which is given the nut's real pose, would not notice. |

The data sheet's section 12 states the fit, that image positions are not good enough for it, what holds the nut,
the cell's compliance (section 5: the tracking law's static stiffness, measured at seven places of the work area)
and what the wrist reads when the nut is on the shaft end, when the hole comes over the shaft and when a tilted nut
binds. Every number there is measured on this cell or follows from the drawing.

Round 4 took a method out of that section. As first written (amendment 3) it also described the nut's rim pushed
against the shaft's side with the distance between the two axes at that contact, said that a nut slid over the
shaft end under a light push dips when it gets there, said that the shaft "has to be found by touch" and that
"both ways work with this cell's sensors". That is how to do the insertion, not what a drawing or a cell's data
sheet states, and the third frontier measurement used it as written (`commissioning.md`). What is left are the fit
and the readings. The sentences that advised how to spend the commissioning window went as well.

Clearances tried (the 16 development kits of amendment 2, in-process runs with the grader's cell and credit rule,
before amendment 3's candidates were drawn; fingers coupled, except where noted):

| radial clearance: drawing (real) | reference (given the real grooves and poses) | round-2 frontier submission, unchanged | the same with insertion by touch (below) |
|---|---|---|---|
| 1.8 mm (1.65 to 1.95), fingers not coupled: amendment 2 | 16 of 16 | 8 of 16 kits, 31 of 48 nuts (9 of 16 through the grader) | |
| 1.0 mm (0.85 to 1.15) | 16 of 16 | 5 of 16, 24 of 48 | 8 of 16, 38 of 48 (an earlier version of the search) |
| 0.75 mm (0.6 to 0.9) | 16 of 16 | 4 of 16, 17 of 48 | 6 of 16, 37 of 48 (an earlier version of the search) |
| 0.5 mm (0.35 to 0.65): amendment 3 | 16 of 16 (624 to 803 steps) | 1 of 16, 8 of 48 (fingers not coupled) | 6 of 16, 37 of 48; 48 of 48 insertions on the shaft |

On amendment 3's own development kits, through the grader, with the insertion as it ended up and the controller's
peg template set to the data sheet's shaft: the unchanged submission 3 of 16 kits (15 of 48 nuts), the one with
insertion by touch 9 of 16 (40 of 48), the reference 22 of the 24 candidates. What amendment 4's 850 steps do to
both, and to the third measurement's submission: `commissioning.md`, "Amendment 4".

**Insertion by touch with the agent's observations alone.** A maintainer's controller (kept with the review's
records, not shipped) changes the round-2 frontier submission's insertion only; its perception, grasp and groove
search are as submitted. The first nut of a
kit is brought beside its shaft and its rim is pushed against the shaft's side from two directions, which gives the
shaft's axis in the hand's own terms; the difference to the cameras' plan is kept in the hand's frame and corrects
the plans for the kit's other nuts; a nut that still comes to rest on the shaft end is slid along a small spiral
under a light push until it dips over the shaft. The insertion uses `ee_pos`, `ee_quat` and `wrench` and nothing
else. See `observability/README.md` for its result through the grader on amendment 3's development kits.

What did not work, for the record: reading the direction of the shaft from the wrist's torques while the nut is
pressed on the shaft end. The contact sits somewhere on a thin crescent of the shaft end beside the hole, and with
a nut that sags a few tenths of a degree in the grip it sits at one end of that crescent as often as in its
middle: 13 presses at 11 to 15 N gave the direction within 45 degrees in 10 cases and off by 75 to 100 degrees
in 3. The data sheet says so.

## What was not varied, and why

- The layout rules, the groove family's rules, the controller's law, gains and limits, the nut's outside and handle,
  the collar, the shaft's height and the bands: as drawn. They either vary already (the layout is drawn per
  episode) or a tolerance on them changes nothing a controller could feel or see at these resolutions.
- The pads' torsional friction keeps the robot model's proportion to the sliding friction (0.05 m at 2.0). The
  reference holds a nut by its handle, where the nut's weight turns it about the grip; a lower torsional value was
  not tried.
- No actuation or image latency, no joint encoder offsets: the first would change the cell's design, not a
  tolerance; the second is an engineering judgement (an arm of this class repeats its poses far more closely than
  the cameras' millimetres; no figure is cited).
- The tolerance code was checked against the earlier simulation at 45e088b (`tolerances=False` repeated the round-2
  runs to within one step). Since amendment 3 the drawing itself differs (the shaft, the coupled fingers), so that
  switch now gives the new drawing's kit on an ideal cell.

## What it does to the reference and to a round-2 controller

Before any candidate ran, on maintainer kits (`sample_kit` seeds 1000 on, layout seed + 4000):

- The reference (given the real grooves through the token): 28 of 32 kits complete in 576 to 821 steps, as many as
  without tolerances (the four it fails are wrist-range cases and one ruined nut, as before). With the tolerances
  switched off (`tolerances=False`) and the earlier kits it repeats its round-2 runs to within one step.
- The round-2 frontier submission (observations only), 16 kits, 1000 steps: 10 of 16 complete without tolerances,
  6 of 16 with them. One group at a time switched off: perfect cameras 10 of 16, no joint friction 8 of 16, the
  model's pad friction 5 of 16. The cameras' calibration error is what it does not cope with: it puts nuts on shaft
  tops several millimetres off and its search (up to 3 mm) does not find the shaft.

After the selection of amendment 2, on the commissioning service's 16 development kits: the same submission
completes 9 of 16 through the grader (12 of 16 with the cameras' error switched off). See `commissioning.md`.

## Sources as read (2026-11-17)

1. ISO 2768-1:1989, general tolerances. Its tables of limit deviations for linear and angular dimensions as
   reproduced in the German Wikipedia article "Toleranztabellen nach ISO 2768"
   (https://de.wikipedia.org/wiki/Toleranztabellen_nach_ISO_2768). The article notes that this part was withdrawn in
   November 2026 and that the new edition of ISO 2768 keeps the linear table and the angular table unchanged.
2. Clearance holes for screws: the table for metric screws in the German Wikipedia article "Durchgangsbohrung"
   (https://de.wikipedia.org/wiki/Durchgangsbohrung; M22: 23 / 24 / 26 mm) and the columns "Durchgangsbohrung
   gemäß EN 20273" (fein H12, mittel H13, grob H14) of "Metrisches ISO-Gewinde"
   (https://de.wikipedia.org/wiki/Metrisches_ISO-Gewinde; M20: 21 / 22 / 24, M24: 25 / 26 / 28). The ISO standard
   of this subject is ISO 273:1979, "Fasteners - Clearance holes for bolts and screws" (English Wikipedia, "List
   of ISO standards 1-1999"; German Wikipedia, "Normenliste DIN 1 bis DIN 499", entry DIN 69: replaced by DIN ISO
   273, 9.1979).
3. The limits-and-fits values: English Wikipedia, "Engineering fit" (https://en.wikipedia.org/wiki/Engineering_fit):
   "Loose running: H11/c11"; table "Useful tolerances" (from the Engineers Black Book): over 18 to 30 mm, H11
   +0.130 / 0, c11 -0.110 / -0.240.
4. The Engineering ToolBox, "Friction - Coefficients for Common Materials and Surfaces"
   (https://www.engineeringtoolbox.com/friction-coefficients-d_778.html), columns static and kinetic (sliding).
5. MuJoCo documentation, Modeling, "Contact parameters" (`doc/modeling.rst` of
   https://github.com/google-deepmind/mujoco): "if one of the two geoms has higher priority, its friction
   coefficients are used. Otherwise the element-wise maximum of each friction coefficient over the two geoms is
   used."
6. C. Gaz et al., supplementary material to the paper named above (revised version of October 15th, 2019), Table I,
   "Estimated joint friction parameters", and the friction equation: the PDF in the authors' repository
   (https://github.com/marcocognetti/FrankaEmikaPandaDynModel, `pdf/RA-L_2019_PandaDynIdent_SUPPLEMENTARY_MATERIAL.pdf`).
7. Robotiq, FT 300-S product sheet (updated February 2021):
   https://robotiq.com/hubfs/Product-sheets/FT%20300/ur/Product_sheet-FT300S_FC_EN.pdf.
8. The hand: `include/franka/gripper.h` and `gripper_state.h` of https://github.com/frankaemika/libfranka (`move(width,
   speed)`, "Current gripper opening width"), and `franka_description/robots/common/franka_hand.xacro` of
   https://github.com/frankaemika/franka_ros (finger joint 2 has `<mimic joint="..._finger_joint1"/>`).

Looked for and not found in a readable form, hence not cited: accuracy figures of camera-to-robot calibration (an
open-access evaluation was read, PLOS ONE 2022, doi 10.1371/journal.pone.0273261, but its figures are in plots and
supplementary tables), the temperature dependence of joint friction, the arm's pose repeatability.
