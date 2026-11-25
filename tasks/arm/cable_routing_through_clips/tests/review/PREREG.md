# cable_routing_through_clips: pre-registration of the graded set

Written before any graded candidate was drawn, and committed to this branch before the first candidate was run
(the commit that adds this file holds no `tests/hidden/episodes.json`). Everything below was fixed on development
seeds only: small integers typed by hand (11-16, 101-124, 201-222, 301-332, 401-406, 501-524, 601-608, 701-724,
801-824, 901-924, 1001-1024, 1101-1124, 1201-1212, 1301-1324, 1401-1424; 20001-22000 for the route generator's
statistics alone) and the
three development boards. Graded candidates come only from the key below.

## The cell and its ranges (`tests/harness_env.py`)

- Robot: Franka Panda (robosuite 1.5.2 model files), base at the origin on the bench top, the operational-space
  controller of tasks/arm/lab_sample_handling, 10 Hz, action = Cartesian step of the grip site (at most 3 cm and 0.2 rad a step),
  grip, camera mask.
- Gripper (confirmed 2026-11-24): on both fingers a cable jaw with a 120-degree groove along the cable (faces at 60
  degrees from the closing direction, 3.6 mm long, 2.4 mm thick, hard contact, friction 1.20 with priority), its
  axis 17 mm below the grip site and level with a cable lying on the board at the grasp height (17 mm + the cable's
  radius above the board); finger servo 40000 N/m limited to 60 N a finger. Measured on ten cables (radius
  3.84-4.71 mm): the four faces carry 138-139 N in sum; travel after 3 s under a 5 N pull within 0.61 mm, under 15 N
  within 0.51 mm (one cable 1.27 mm); slip force 62.5-119.5 N.
- Connector (confirmed 2026-11-24): the cable's fixed end is held by a strain-relief boot: 3.2 N/mm along the
  connector's axis (a TPE boot of Shore 60 A, 35 mm long, 1 mm wall over a 9 mm cable), 7 mm of travel, then a stop;
  damping 15 N s/m.
- Board: 0.52 x 0.92 x 0.012 m centred at (0.49, 0). Route area 0.76 m along the board, 0.34 m across.
- Route: the generator draws 3, 4 or 5 fixtures (graded: 3 or 4, see the graded set); the first is a clip; no two posts in a row; legs 0.12-0.19 m; a clip turns the route by
  at most 0.6 rad, a post by 0.7-1.3 rad; fixtures at least 9 cm apart; no leg within 4 cm of another fixture.
- Mounting: each fixture and the connector within 20 mm (uniform over the disc) of its printed place; a clip's axis
  and the connector's heading within 8 degrees. The mounting errors of a drawing are drawn again (up to 20 times)
  until the board as built is still a fair board for its drawing (`harness_env._built_ok`: legs 9-23 cm, fixtures
  at least 7 cm apart, no leg within 3 cm of another fixture, a post still turns the route to its printed side by
  at least 0.45 rad and a clip by at most 0.9 rad, the route fits the cable); on 1000 development seeds that took
  78 extra draws and dropped no drawing. Why 20 mm: at 12 mm a program that trusts the printed places completed 9
  of 12 development routes (the lead-in fins take a cable up to about 6.7 mm off and a press can be repeated), at
  20 mm 2 of 12 (seeds 1001-1012, before the fairness check) and 1 of 12 (seeds 1301-1312, with it), at 30 mm 1
  of 12; the reference completed 12 of 12 at 20 mm and 11 of 12 at 30 mm.
- Clip (confirmed 2026-11-24; one family, the size following the cable): pins of radius 2 mm, 20 mm tall; slot =
  cable diameter + 4.3-4.7 mm; on each pin's top a lead-in fin 7 mm long rising outward at 50 degrees; on each pin's
  inner top a retaining tab 0.6 mm thick, at rest 35 degrees below horizontal, reaching 3.0 mm over the cable's side
  (0.65-0.85 mm more than the cable's side play in the slot), folding down only (0-1.1 rad), spring 0.0015 N m/rad.
  Two modelling corrections: the tab's stop is rigid (joint limit impedance 0.9999) and the tab's hard contact governs
  its contact with the cable. Measured on twelve cables (radius 3.53-4.97 mm), 20 presses each (one unaligned press,
  lateral offset within 2 mm, press tension 3-5 N): inserted 240 of 240; all 240 still in the clip after a load
  that brought the connector's tension to 9.3-11.9 N; the release force was not measured. A clip's axis is not
  scored and does not have to be known: the reference presses the cable in along the line from the last fixed
  point and never reads the axis; with the axes mounted up to 40 degrees off the print it completed 10 of 12
  development routes, up to 75 degrees off 3 of 12 (at 8 degrees: 24 of 24). Capture window: a press
  5 mm off inserted 7 of 8 times, 8 mm off 4 of 8, 11 mm and 14 mm off 0 of 16; a press stopped 14 mm short 0 of 8.
- Post: radius 12 mm, 30 mm tall, cap radius 18 mm.
- Cable: 40 segments (40, 80 and 120 agree within 4 mm in shape and 1 N in tension response; the solver's no-slip
  pass is off), length 0.70-0.90 m, radius 3.5-5.0 mm, bending stiffness 0.0008-0.003 N m^2, friction 0.5-1.0, density
  1200-2000 kg/m^3. The job sheet gives radius and length; stiffness, friction and density are not given.
- Damage: the connector is damaged the moment the mean of its force over the last 0.1 s passes 40 N. A shorter peak
  counts only through that mean.
- Judged after the hand is opened and lifted 15 cm and 2 s pass: a clip when the cable's centre line crosses the
  segment between the pins below 13 mm above the board; a post when the cable comes within 10 mm of its surface
  below its top with the post on the stated side; in route order by arc length.
- Credit of an episode: 1 when every fixture is satisfied and the connector is undamaged, else 0. The prefix
  fraction is written to the metrics (`prefix_credit`) and scores nothing.
- An episode: at most 1500 control steps.

## The graded set

- Candidate k (k = 0, 1, 2, ...) has the seed: first 8 bytes (big endian) of
  sha256("cable_routing_through_clips/graded/v1/" + str(k)).
- Walk k upward. A candidate is KEPT when (1) the generator returns an episode, (2) its route has three or four
  fixtures and that class is not yet full (classes: three fixtures: 12 episodes, four fixtures: 8), (3) the
  privileged reference, run once through the in-process episode loop, ends with credit 1. Stop at 20 kept.
  Routes of five fixtures are not graded and a candidate that has one is skipped: the generator makes them 3 times in
  2000 (a five-fixture route does not fit most cables: forced, it cannot be drawn for 127 of 200), and the reference
  was never measured on one. Four-fixture routes are 19 % of the generator's and 40 % of the graded set by
  design. At the 20 mm tolerance the reference completes 24 of 24 three-fixture routes (development seeds
  1301-1324) and 23 of 24 four-fixture routes (1401-1424, the count forced), no connector damaged, the highest
  0.1 s mean tension 17.3 N, complete routes within 1208 of the 1500 steps.
- Every candidate walked is recorded with its outcome (kept, class full, reference failed and how). The keep rate
  among candidates whose class was open is reported. If it is below 0.7 the reference is improved and the walk
  starts again from k = 0 under the key ".../graded/v2/" (and so on), every earlier walk kept in the record.
- The reference is improved only on development seeds and on dev-board episodes, never by looking at a failed
  graded candidate's layout.
- The set lives only under `tests/hidden/`.

## Development boards (the commissioning service)

- Three boards, always 3 fixtures (the graded set has 3 or 4: stated in the instruction). Board j (j = 0, 1, 2):
  first 8 bytes of sha256("cable_routing_through_clips/dev/v1/" + str(j)); a board whose generator fails or whose
  seed equals a graded seed takes the next j.
- The service gives the public observation only (images on request, joint and hand state, wrist wrench, the job
  sheet): no cable state, no fixture pose, no tension, no credit, no word on whether a fixture is satisfied.
- Budget: 20000 control steps (about a third of the 2 h session at the measured 0.1 s a step with its pictures);
  a reset is charged 100 steps, a picture 2.

## Control programs (variants of the reference; measured before the set was drawn)

On development seeds 1301-1312 (ten of the twelve boards have a post), in process, 20 mm tolerance:

| program | complete routes | how it loses |
|---|---|---|
| stops after the first fixture | 0 of 12 | the credit is the whole route's |
| stops after two fixtures | 0 of 12 | the same (at 12 mm one of 12 had its last post made by chance) |
| does not watch the tension | 1 of 12 | the connector damaged in 10 of 12 (at 12 mm: 5 of 12 complete, 7 damaged) |
| takes every post on the wrong side | 0 of 10 boards with a post | the two boards without a post are the reference's (2 of 2) |
| trusts the printed places | 1 of 12 | presses beside the clips, passes posts wrongly |

The same programs run once over the wire on the graded set after it is drawn (tests/review/mutants.md).
Removed: "clip turned by 90 degrees". It changed nothing the reference does (10 of 12 at 12 mm): a two-pin clip
takes the cable along the route whatever its printed axis says, and the reference never reads the axis. The axis
is therefore neither scored nor asked for; a control without a mechanism proves nothing, so there are five.

## Fixed with it

- The reference: `solution/oracle/fitter.py` as of the commit that adds this file. On the wire the grader runs it
  through `oracle_main.py`; the walk runs the same program in process (`fitter.run_episode`).
- The controller's time budget: 3600 s for the set, 0.2 s a picture; a reply within 30 s; 1500 steps an episode.
- The cameras: `overhead` 960 x 720 at (0.49, 0, 1.05) looking straight down, 45 degrees; `wrist` 480 x 360, the
  hand's own (robosuite's eye-in-hand), 75 degrees.
- A simulation process that ends by itself is run again with the same seed (tests/verify.py); the walk does the
  same (a candidate whose process ends by itself is run once more and the event is recorded).

## Amendment after the draw (rendering and job sheet only)

After the set was drawn (commit `532d0c02`) and validated, two things changed in files this document froze, none
of them in the physics, the route generator or the reference's program:

- the pictures: the board's print carries marks of use (`harness_scene._worn`: soiling, scratches, rubbed
  stretches, filled screw holes, drawn from the board's seed through a stream of its own) and the worktop's map is
  the worn one; the cable is drawn as one mesh;
- the job sheet no longer carries a clip's `axis` or `clip_axis_tolerance` (`harness_env.job_sheet`): the axis is
  neither scored nor needed, and nothing should suggest measuring it. The board's print keeps its clip boxes,
  drawn from the drawing itself. The control "trusts the printed places" takes a clip's axis from the truth for
  its own seated check (it had it from the sheet).

Shown after the change: the numerics fingerprint's simulation part is unchanged, and the reference over the wire
completes 20 of 20 boards with the step count it had on every board when the set was drawn. Had one step count
moved, the draw would have been void.

## The walk's tool and record

`tests/review/select_set.py` walks the candidates and writes `tests/hidden/episodes.json` (the set, in the order
kept) and `tests/hidden/episode_selection.json` (every candidate walked and its outcome). Neither is in the image
or the agent's sandbox.
