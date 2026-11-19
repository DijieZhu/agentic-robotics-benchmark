# The commissioning service and the takt (maintainer notes, 2026-11-16; rounds 3 and 4 on 2026-11-17)

Why the task changed, what the agent has now, and how the window was sized. The agent-facing contract is
`instruction.md` and `environment/setup/payload/datasheet.md`; this page is for maintainers and validators.

## Why

The first frontier measurement (Claude Opus 5.5, Claude Code, effort high, 2026-11-12, on the version with a
development simulator in the agent's home) scored 0.92: 9 of 10 kits. The controller located the parts by vision and
found each groove by touch, as intended. What made it easy was the development loop. The simulator's source was in
the sandbox, so the agent's own harness read its internal state (`env.data`, `nut_state`, `pin_in_wall`,
`peg_poses`), printed the pin's true position in the groove at every step, saved and resumed simulator states, and
ran about 300 traced kit runs on about 95 kits. Removing the ground-truth accessor did not keep the answer from it.
Its kits took 1047 to 1834 steps of the 2400 allowed.

Round 2 (2026-11-16), two changes, both approved by the benchmark's owner:

1. **A commissioning service in place of the readable simulator.** A commissioning engineer has the cell's data
   sheet and a window on the cell, not the cell's source code and not a debugger on the parts' true state.
2. **A takt near the reference's.** Then 40 s per nut, 1200 steps per kit. A production cell's takt is set by what
   the job needs, not by twice what the slowest attempt took.

The second frontier measurement (the same agent, 2026-11-16, on that version) scored 0.71: 7 of 10 kits. Hiding the
simulator was not enough: within minutes the agent rebuilt the cell from the data sheet as a simulation of its own,
developed its controller there with that simulation's ground truth, and used the 32 commissioning episodes to
confirm it. The data sheet was the cell: every kit was exactly its drawing.

Round 3 (2026-11-17), approved by the benchmark's owner:

3. **A real kit on the cell of the day.** No real kit or cell is exactly its drawing. Every episode, graded or
   commissioning, draws its own actual values within tolerances that the data sheet states: the nuts' grooves, holes
   and friction, the pegs' shafts and pins, the pads' and the arm joints' friction, the wrench zero's drift, the
   cameras' mounts and intrinsics, the lamps (`tolerances.md`: every range and where it comes from). A simulation
   built from the data sheet can only draw from the ranges; the commissioning window is where the agent meets
   real ones.
4. **A takt of 100 s for a kit**, 1000 steps.
5. **A locating fit** of the nut on its shaft (0.5 mm of radial clearance on the drawing in place of 1.8 mm; square
   edges, no lead-in chamfer) and the hand's two fingers as one mechanism, after changes 3 and 4 alone proved not
   hard enough (below). A nut can no longer be put on a shaft from image positions: the shaft has to be found by
   touch. The data sheet states the fit, the cell's compliance and what the wrist reads when a nut meets a shaft
   (`tolerances.md`, "The fit"). The reference, which is given the real grooves and poses, needs 644 to 802 steps
   on the graded kits.

The third frontier measurement (the same agent, 2026-11-17, on that version) scored 0.61: 6 of 10 kits, 25 of 30
nuts, the complete kits in 730 to 892 steps. Two things kept it there. The data sheet's new section 12 did not
only state the fit and the wrist's readings: it described the nut's rim pushed against the shaft's side with the
distance between the two axes at that contact, and said that both ways of finding the shaft work. The agent
wrote exactly that insertion in half an hour (two touches, a circle intersection; by its own count 46 of 50
insertions on the real cell went on at the first place, at about 52 steps a nut). And its groove search by touch
is about as fast as the reference's known path. This time it developed mostly on the real cell (29 whole runs
over all 16 development kits) and stopped after 3.9 of its 8 hours, when the window was used up.

Round 4 (2026-11-17), approved by the benchmark's owner:

6. **The data sheet states readings, not a method.** The rim paragraph, the slide over the shaft end, "has to be
   found by touch" and "both ways work" are gone from section 12; so is the advice on how to spend the
   commissioning window. What stays is what a drawing and a cell's data sheet state: the fit and its series, the
   tolerances, no lead-in, the compliance, and what the wrist reads on the shaft end, over the shaft and when a nut
   binds. `tests/test_keyed_nut_peg_assembly_commissioning.py` keeps the removed wording out of the payload.
7. **A takt of 85 s for a kit**, 850 steps. The reference needs at most 802 steps on a graded kit and 798 on a
   development kit, so the kits stay the same (amendment 4, below).
8. **Sources as read.** Every source named for a tolerance was checked against a text that could be read; the
   camera calibration, the friction spread and two smaller items had been attributed to papers that were not read
   and are engineering estimates now (`tolerances.md`). No value changed.

## What the agent has

| what | where | note |
|---|---|---|
| the cell's data sheet | `~/datasheet.md`, `~/cell_data.py` | the kit's drawing data and tolerances, how the family and the layout are drawn, the tolerances (section 11), the robot and its controller (action format, tracking law and gains, gripper), sensors, camera calibration, safety stops, takt, the service's wire; `cell_data.py` holds the same numbers and geometry helpers and no simulator (`tests/test_keyed_nut_peg_assembly_commissioning.py` compares them with the simulator's, and checks that the payload holds no simulator code) |
| the robot's model | the robosuite 1.5.2 Panda and gripper files in the image's Python environment | public; MuJoCo is installed, so the agent can build a twin of its own from the data sheet |
| a commissioning window | the service on `/run/keyedcell/commission.sock`, `~/commission/client.py`, `~/commission/run_kit.py` | 32 episodes on 16 development kits |

What it does not have: the simulator's code (`keyed_env.py` is no longer in the payload), its model files (the scene
is built in memory by that code), its state, the contact and solver settings, any kit's grooves or poses, a score, a
completion verdict or a debrief.

## The service

`environment/commission_service/`: `commissiond.py`, `keyed_env.py` (a byte copy of `tests/keyed_env.py`), since
amendment 5 the cameras' renderer `keyed_render.py` with `keyed_look.json` (byte copies of the grader's too:
`rendering.md`) and `kits.json` (the development kits and the window). The image installs it under `/opt/keyedcell-commission` for the
account `keyedcell`, mode 700; `environment/setup/setup.sh` starts it under a restart loop with the image's rendering
and thread settings. The image's working directory is `/`, so that the engine's root helpers never import a module
from the agent's home (engine issue AgentsLastExam/ale#142).

- One connection is one episode: `reset(kit)`, `step(action)`, `close()`. It answers the observation of the public
  wire and how the episode ended: all 850 steps run, a safety stop (the ruined nut's colour, or the instability),
  the controller budget used up, 120 s of silence, or the client's close. `close(images=true)` adds the three
  cameras' images of the final state. About 1 % of the step replies carry the perturbed copy the grader uses for its
  repeated-step queries, so that `run_kit.py` can drive a controller the way the grader does.
- It runs `keyed_env.NutEnv`, never the grader's `GradedNutEnv`: the credit rule (`tests/keyed_credit.py`) is not in
  the image, and an episode never ends because a kit is complete.
- The window is counted under a lock in the service's own folder and survives a restart of the service; an episode
  that could not be started (a fault of the cell) is not counted. A kit's episode is the same every time (its own
  layout, its own actual values within the tolerances and its own noise, all from its seed), so two versions of a
  controller can be compared on it. The service never tells a kit's actual values.
- Checked in a container of the image, as the agent account (2026-11-16): the folder can be neither listed nor read;
  `/proc/<pid>/{environ,mem,maps,cwd,exe,fd}` of the service are refused; the service cannot be signalled; methods
  other than the three answer `forbidden`; a third open episode answers `busy` and uses no episode; a `reset`
  beyond the window answers `commissioning_window_used` (tried with a window of 4); the count survives a restart;
  silence and the budget end an episode with the documented words (tried with 6 s and 12 s). A step without images takes about 0.02 s there, with all three images 0.4 s; a whole
  whole episode through `run_kit.py` about half a minute. Repeated on the round-3 image (2026-11-17): the closed
  folder and process, the wire, two episodes at a time, the count, a cell fault; a whole episode ends with "all
  steps run" at step 1000 (850 since round 4).

## The development kits

Chosen like the graded set, from their own pre-registered candidates (in force: kit seed 9001 + i, layout and noise
seed 13001 + i, drawn for amendment 3 and run again under amendment 4's takt; amendment 2 used 8001 + i and
12001 + i, round 2 7001 + i and 11001 + i): the first 16 that the frozen reference completes within the takt, each run once through the grader (`select_set.py --pool dev`; the record is the selection
record's latest amendment, `development_kits`). The graded kits are
"draws that the reference completed"; unfiltered development kits would be a slightly different population (about
one draw in seven has a wrist-range case the reference does not handle), and with 32 episodes every development kit
has to be worth its episode. No development seed is a candidate's and no development kit has a graded kit's
grooves (the test above). The run under amendment 3 (2026-11-17): the reference completed 22 of the 24 candidates in
637 to 798 steps; U11 and U15 dropped (the takt ran out with none and two nuts assembled), so the kits are U01 to
U10, U12 to U14 and U16 to U18, which the agent addresses as kits 1 to 16. Under amendment 4's 850 steps the run
repeated itself: the same 22, each with the same steps, the same 16 kits. (Amendment 2's run, the same day: 23 of
24 in 636 to 804 steps. Round 2's run, 2026-11-16: 22 of 24 in 598 to 846 steps.)

## The window's size

32 episodes, 2 at a time, on 16 kits. The model is a real commissioning window: a plant gives the engineer who
brings a new kit family onto a cell one working shift and a tray of pre-series kits. A trial takes about a quarter
of an hour: a technician lays the kit out, the cell runs at most one takt (85 s), the engineer looks at the
result and the logs, the technician takes the nuts off the pegs, the engineer changes the program. That is 32
trials in 8 hours, two per kit on average: a few for teach-in and measurements (grasp, carry, how a nut on the pin
feels, what a riser and a blocked turn read), the rest for whole runs. Simulated time is not the limit (32 episodes
are 45 minutes of cell time and about a quarter of an hour of computing); the count is, as on a real cell. In the
third measurement the count was what ended the agent's work.

For comparison: the readable simulator of the first measurement served about 300 kit runs with the true state
printed at every step.

## The takt

Round 2: `keyed_env.CYCLE_STEPS` 800 -> 400, an episode of at most 1200 steps. Round 3: `keyed_env.MAX_STEPS = 1000`,
100 s for a kit. Round 4: `keyed_env.MAX_STEPS = 850`, 85 s for a kit. The controller budget (600 s of controller
time per kit) and the image charges are unchanged. (Amendment 5, the cameras' renderer, moved the wrist image's charge
from 0.25 s to 0.35 s: `rendering.md`.)

The tolerances change what a kit is (the reference is given the real groove through its token:
`keyed_credit.ground_truth`, `actual`), so amendment 2 of the selection record drew 16 fresh hidden candidates and 24
fresh development candidates and was committed before any of them ran (45e088b). Checks before that, on maintainer
kits only (`sample_kit` 1000 on, layout seed + 4000; neither candidates nor development kits):

- With the tolerances switched off (`NutEnv(tolerances=False)`) and the earlier kits (1000-1009, 1020, 1037) the
  reference repeats its round-2 runs to within one step (601 to 743 steps): the tolerance code leaves the drawing's
  cell as it was.
- With the tolerances, the reference completes 28 of 32 kits in 576 to 821 steps, as many as without them.

The hidden set under amendment 2: the reference completed 14 of the 16 fresh candidates (605 to 812 steps; E12 and
E14 dropped, a nut ruined each); the set was the first ten, E00 to E09, and its full verify measured 1.0 with every
episode repeating its selection run. Amendment 3 replaced it (below).

## Amendment 2: within reach, and not hard enough

A controller that has only the observations: the second measurement's submission, which was written for the
drawing's cell and a 1200-step takt, run by the grader on the 16 development kits of amendment 2 (the per-kit
record is in this folder's history, commit 6347893): **9 of 16 kits** (32 of 48 nuts), in 664 to 992 steps.
So the version is within reach, and it is not hard enough:

- Its groove search by touch is about as fast as the reference's known path (about 210 steps a nut when no false
  entrance is met), so the takt leaves it room for three or four exploration errors per kit. No tolerance slows a
  search by touch.
- Its failures are insertions that miss the shaft: the cameras' calibration error is larger than its search. With
  the cameras' error switched off it completes 12 of the 16 kits; with the other groups switched off one at a time
  (maintainer kits) nothing changes much. A controller written for these rules will widen its search.
- A shorter takt would not change the picture: of the same runs, 6 of 16 kits end within 900 steps and 4 of 16
  within 800 (9 and 7 of 16 with the cameras' error switched off).

## Amendment 3: the fit

The contribution's 1.8 mm of radial clearance absorbed every tolerance and most of the cameras' error, so the fit
became a locating fit and the hand's fingers one mechanism (`tolerances.md`, "The fit": sources, the clearances
tried, why the fingers). It changes what a kit is, so amendment 3 drew 16 fresh hidden candidates and 24 fresh
development candidates and was committed before any of them ran (32d05cc).

- On maintainer kits before that: the reference completes 27 of 32 (582 to 973 steps; 28 of 32 with the earlier
  fit), with no fault and no ruined nut. It needed no change: it steers the real nut onto the real shaft.
- Development kits: above. Hidden set: the reference completed 12 of the 16 fresh candidates (600 to 802 steps;
  F06, F08, F09 and F15 dropped, the takt ran out with one or two nuts assembled); the set is the first ten, F00 to
  F05, F07, F10, F11 and F12, and its full verify measured 1.0 with every episode repeating its selection run.
- Within reach: a maintainer's controller on the observations alone, with insertion by touch, completes **9 of the
  16 development kits** through the grader and puts 48 of 48 nuts on their shafts (`observability/README.md`).
- Hard enough to measure: the second measurement's submission, unchanged, completes **3 of the 16** (15 of 48
  nuts): it cannot find the shaft from image positions.

## Amendment 4: an 85 s takt, and a data sheet without the method

The takt changes nothing of a kit or of the physics, and the reference's runs are deterministic, so amendment 4 ran
the rule again on the same candidates; it was committed before any of them ran on the amended files (09c09d6).

- Development candidates: the reference completed the same 22 of 24, each with the steps of its amendment-3 run
  (637 to 798); the kits are the same 16.
- Hidden candidates: the same 12 of 16, each with the same steps (600 to 802); F06, F08, F09 and F15 ran out of
  the takt as before. The set is the same ten kits, and its full verify measured 1.0 with every episode repeating
  its selection run (644 to 802 steps).
- On 32 maintainer kits the reference completes 26 within 850 steps (582 to 793); it completed 27 within 1000.

What the takt does to controllers that have only the observations, each run by the grader (2026-11-17):

| controller | the service's 16 development kits | kept development candidates outside the service (U19 to U24) | 32 maintainer kits (the reference completes 26) |
|---|---|---|---|
| the third measurement's submission, unchanged | 9 of 16 (39 of 48 nuts), 679 to 836 steps | 1 of 6 (12 of 18 nuts) | 12 of 32 (75 of 96 nuts), all 12 among the reference's 26 |
| the maintainer's controller with the round-3 insertion by touch | 1 of 16 (31 of 48 nuts) | | 2 of 32 (55 of 96 nuts) |

- The third measurement's submission was commissioned on the service's 16 kits (29 whole runs), so its 9 of 16 is
  an in-sample figure. On kits it never met it completes 13 of 38, and 4 of the 10 graded kits by the steps of its
  own graded run (730, 752, 808, 841 within 850; 887 and 892 not). It carries the insertion that round 3's data
  sheet described; what removing that description does can only be measured by a new run.
- So the observations suffice within 85 s: a controller that sees and feels only what the agent gets completes
  kits end to end. The maintainer's controller no longer does: its insertion costs about 50 steps a nut at the
  median, like the measurement's own (about 52 by its count), and the rest of it is slower than the measurement's;
  the takt does not leave that.
- The camera-only check repeats on amendment 4's image with every kit's numbers identical.
