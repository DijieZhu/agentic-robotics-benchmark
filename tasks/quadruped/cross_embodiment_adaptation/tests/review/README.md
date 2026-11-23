# Review notes: version 10 (maintainers, 2026-11-22)

The task: a UAV surveys a warehouse; a loaded Unitree Go1, commanded joint by joint, has to cross it to a dock on
its first deployment there, crouching under a duct and standing tall over a deck's guides. The survey is out of
date and not to scale, the Go1's odometry drifts, and each site's robot and fittings differ.
The graded set and every rule with its range: [selection.md](selection.md).

Version 10 against version 9 (next section): both cameras draw the hall the course stands in, with its dressing,
the course parts as built objects, the Go1's equipment and the UAV (`tests/runtime/hall_render.py`, MuJoCo's
Filament renderer on the CPU; the bundle is `hall_look/` of the task's assets); the rail bollards are six again;
contact follows what is drawn (the pallet 0.144 m thick with two foot brackets, the dock mat a walked surface);
the commissioning pad is the bay inside its four boundary barriers. The rules were written first
(selection.md, "The cameras show the hall").

## Controls of version 10 (`controls.py`, `replay_submission.py`, `observability.py`)

| controller | graded sites (16) | development sites |
|---|---|---|
| oracle (true pose, the site as found) | 16 | 93 of 96 (46 and 47 of 48) |
| oracle on dead reckoning (true launch pose, then its own odometry) | 0 | |
| oracle on the surveyed route and dock | 4 | |
| oracle without stance-height feedback and pitch trim | 9 | |
| the saved version-9 frontier program (2 of 16 there), replayed | 0 | 0 of 32 |
| the saved version-8 frontier program (7 of 16 on version 9's set), replayed | 1 | 11 of 32 |

The oracle does not look at a picture, so its rows say that the new contact geometry costs it nothing (the same
three development misses as before: two profiler-on-floor contacts and one fall; slowest site 38,392 ticks of
48,000). The two saved programs were written for the old pictures and are here for reference only: the version-9
program walks into the first fitting on nearly every site; the version-8 program docks 11 of 32 development
sites (7 on version 9) and 1 of the 16 graded (7 on version 9), which is the spread of a single program between
two site sets, not a measure of the cameras.

The audit of the mast camera's pictures (a cap counts only if the picture shows a connected patch of its colour at
the place where it projects): at least 3 caps at every 0.4 m of the route on all sixteen graded sites (medians 12
to 14), the pallet seen before the middle rack and the beacon before the dock on all sixteen. On sixteen
development sites the same, except the one site where the oracle falls before the dock.

## The colour cues in the new and the old cameras

CIELAB distance from a cue's median colour to the nearest background colour in the same image (2nd percentile over
the pixels at least 3 px from any cue), median over views and, in brackets, the least. Development sites 1 to 4:
the mast camera at every 0.8 m of the oracle's route, eight detents; the survey camera on a 60-move sweep. Cue
pixels are the ones MuJoCo's classic renderer segments as that cue in the same view. "old" is the classic renderer
on the same models (version 9's cameras).

| camera | cue | views | new: L* a* b* | new: distance | old: distance |
|---|---|---|---|---|---|
| mast | green bollard cap | 315 | 25 -31 27 | 40 (0) | 59 (42) |
| mast | blue beacon cap | 43 | 48 27 -70 | 68 (50) | 83 (53) |
| mast | magenta pallet | 98 | 25 50 -13 | 47 (43) | 41 (17) |
| survey | blue beacon cap | 19 | 57 18 -61 | 62 (61) | 88 (10) |
| survey | magenta pallet | 80 | 24 48 -12 | 49 (46) | 8 (8) |
| survey | blue dock | 64 | 22 27 -53 | 57 (50) | 88 (45) |
| survey | orange barrier cap | 240 | 57 48 53 | 72 (40) | 88 (63) |
| survey | teal cover | 77 | 50 -26 -11 | 27 (26) | 36 (33) |

Every cue keeps a distance of 27 or more at the median. The green caps lose the most (40 against 59): behind them
are now blue rack uprights, orange beams and cardboard, not a grey backdrop. The least value 0 is a cap seen
against another cap. The sweep did not reach the rails, so the survey rows have no green cap.

## The renderer: time, memory, sameness

Measured in the task's image on 2 CPUs and 4 GB: a site's renderer process is ready 5.7 s after the first
request and draws its first image 3.6 s later; then a survey image takes 0.10 s and a mast image 0.13 s (0.16 s
at most); the process peaks at 1.2 GB. A mission draws at most 80 survey and 24 mast images. 141 images of three
development sites and their pads (states set, not simulated) have identical bytes in two processes on one host
and on the second validation host; `tests/numerics_check.py` hashes three camera frames, so the grader refuses a
machine that draws differently.

## Version 9 (2026-11-21), kept for the record

Version 9 against version 8 (next section): sixteen graded sites instead of eight (a measurement fix: one site is
a sixteenth of the score), and four bollards along the long walls instead of six (nine in the survey; version 10
went back to six).

## Controls of version 9 (`controls.py`, `replay_submission.py`, `observability.py`)

| controller | graded sites (16) | development sites |
|---|---|---|
| oracle (true pose, the site as found) | 16 | 93 of 96 (46 and 47 of 48) |
| oracle on dead reckoning (true launch pose, then its own odometry) | 0 | |
| oracle on the surveyed route and dock | 4 (three of them the unchanged sites) | |
| oracle without stance-height feedback and pitch trim | 9 | |
| the saved version-8 frontier program (4 of 8 there), replayed: six rail bollards | 7 | 7 of 32 |
| the same, four rail bollards | 7 | 6 of 32 |

The oracle rows other than the first were measured with six rail bollards; the oracle does not look at them.
The audit of the mast camera's frames with four rail bollards: at least two caps in view at every 0.4 m of
the route, the pallet in view before the middle rack and the beacon before the dock, on all sixteen graded sites.

## Version 8 (2026-11-21), kept for the record

Version 8 against version 7 (next section): the lidar stands inside a tubular guard frame and the profiler is a
line profiler mounted crosswise, which gives the equipment the reach the two boxes of version 6 had; the
odometry's heading error rate goes up to 0.8 degrees per second; a deployment has 24 mast exposures instead of
40; the guides' rise ends at 0.112 m and friction starts at 0.45 (for the oracle, see selection.md).

## Controls of version 8 (`controls.py`, `replay_submission.py`, `observability.py`)

| controller | graded sites (8) | development sites |
|---|---|---|
| oracle (true pose, the site as found) | 8 | 93 of 96 (46 and 47 of 48) |
| oracle on dead reckoning (true launch pose, then its own odometry) | 0 | |
| oracle on the surveyed route and dock | 2 (the two unchanged sites) | |
| oracle without stance-height feedback and pitch trim | 4 | |
| oracle with the fittings' nominal clearances | 8 | |
| the saved version-7 frontier program (5 of 8 there), replayed as written | 0 | 0 of 32 |
| the same with its own exposure cap set to 24 | | 7 of 32 |
| the saved version-6, version-5 and v8 frontier programs, replayed | 0, 0, 0 | |

Read with care: the saved programs spend the old exposure budget and then stand still, so their zeros say that
they were written for 40 exposures, not that the task is hard. The row that carries information is the
version-7 program with its one constant changed: 13 of 32 on version 7, 7 of 32 here. With the guides' range
ending at 0.112 m, the nominal clearances no longer cost the oracle anything; what the fittings ask is carried by
the load (the row without feedback) and by the duct.

The oracle's slowest development site takes 38,295 ticks of 48,000; its three development misses are two
profiler-on-floor contacts under a low duct and one fall. The audit of the mast camera's frames: at least three
caps in view all along the route, the pallet before the middle rack and the beacon before the dock, on every
graded site.

## Version 7 (2026-11-21), kept for the record

Version 7 is version 6 (next section) with the carried equipment and the barrier in the shapes of real objects: a
3D lidar head on a mast and a floor profiler under the nose instead of two boxes, a pallet on its short edge
instead of a block. No rule or range changed. The oracle's step onto a deck's apron is higher than before.

## Controls of version 7 (`controls.py`, `replay_submission.py`, `observability.py`)

| controller | graded sites (8) | development sites |
|---|---|---|
| oracle (true pose, the site as found) | 8 | 93 of 96 (47 and 46 of 48) |
| oracle on dead reckoning (true launch pose, then its own odometry) | 2 | |
| oracle on the surveyed route and dock | 2 (the two unchanged sites) | |
| oracle without stance-height feedback and pitch trim | 5 | |
| oracle with the fittings' nominal clearances | 4 | |
| the saved version-6 frontier program (1 of 8 there), replayed | 1 | 5 of 32 |
| the saved version-5 frontier program (7 of 8 there), replayed | 1 | |
| the saved v8 frontier program, replayed | 0 | |

The oracle uses up to 31,511 ticks of 48,000 on the graded sites and 38,087 on its slowest development site; its
three development misses are one profiler-on-floor contact under a low duct, one fall and one budget stop on a
deck's apron. The audit of the mast camera's frames reads as for version 6: at least three caps in view all along
the route, the pallet before the middle rack and the beacon before the dock, on every graded site.

## Version 6 (2026-11-21), kept for the record

## Why version 6

Version 5 (below) took away a rule no grader could check, gave the agent a commissioning pad and a datasheet and
made each site's load and clearances differ. One frontier episode on it scored 7 of 8 in 41 of 120 minutes: told
the ranges, the model built a faithful simulation, measured the two clearances and ran a prepared plan with two
corrections. Version 6 makes the mission need perception and reaction on site that a prepared plan cannot hold:

| | version 5 | version 6 |
|---|---|---|
| the survey | exact, and still true at deployment | its tracking mis-scales (0.97-1.03); by deployment a pallet stack may stand in the middle rack's other opening, the dock may have moved (its beacon with it), a bollard may be gone or added |
| the Go1's pose | ideal odometry from the launch pose | leg odometry with a heading error rate within 0.40 degrees per second and a distance factor 0.94-1.06; encoders and IMU with noise and bias |
| landmarks | five bollards round the staging bay, 16 exposures | eleven bollards round the bay and along the hall, a dock beacon, the pallet stack; 40 exposures |
| the pad | the loaded Go1 among bollards | the same with the sensing's noise and drift, a pallet stack and a beacon to look at |
| assets | fetched from their sources at build | from the benchmark's asset dataset at a pinned commit |

## Controls of version 6 (`controls.py`, `replay_submission.py`, `observability.py`)

| controller | graded sites (8) | development sites |
|---|---|---|
| oracle (true pose, the site as found) | 8 | 91 of 96 |
| oracle on dead reckoning (true launch pose, then its own odometry) | 1 | |
| oracle on the surveyed route and dock | 2 (the two unchanged sites) | |
| oracle without stance-height feedback and pitch trim | 4 | |
| oracle with the fittings' nominal clearances | 4 | |
| the saved version-5 frontier program (7 of 8 there), replayed | 1 | 3 of 32 |
| the saved v8 frontier program, replayed | 0 | 0 of 32 |

The oracle uses 24,634 to 32,228 ticks of 48,000 on the graded sites; its slowest development site takes
35,904 (two of the 96 run out of ticks, one falls, two touch the floor with the scanner under a low duct). What a program restricted to the stated sensors has to work with is audited in `observability.py`
(selection.md has the numbers). A saved program's replay is a lower bound on what a change does to a fresh agent,
not an estimate: version 5's replay of the v8 program said 8 of 32 and the fresh episode scored 7 of 8.

## Version 5 (2026-11-20), kept for the record

The task: a UAV surveys an unknown warehouse; a loaded Unitree Go1, commanded joint by joint, has to cross it to
a dock on its first deployment there, crouching under a suspended duct and standing tall over a deck's guides.
The graded set, the per-site properties and how their ranges were set: [selection.md](selection.md).

## What changed against the author's v8 (`d07a9c15`)

| | v8 | version 5 |
|---|---|---|
| the Go1 in development | never available; the instruction forbade simulating it | a commissioning pad (bare floor, bollards, mast camera; 40 episodes, 600,000 ticks; no truth, score or debrief), a datasheet, and any simulation of the agent's own |
| per-site physics | servo stiffness +/-10 % | carried mass 2.5-4.5 kg and where it sits, servo stiffness +/-15 %, torque limit, friction, the duct's clearance (0.410-0.445 m) and the guides' rise (0.070-0.125 m), all stated with their ranges, none reported |
| between graded sites | files in /tmp, /dev/shm, the account's home and lock directories survived | everything the account leaves is removed and shared files are restored (`tests/isolation.py`) |
| oracle | a reference from the public observations | privileged (a token; `solution/oracle`); the public reference is kept here as a control |
| tick budget | 36,000 | 48,000: 1.44 times the oracle's slowest of 48 development sites |
| arithmetic | as the host's CPU has it | pinned in the image, checked before grading (`tests/numerics_check.py`) |
| anchor | 1.0, `full_at` 1 | 1.0, `full_at` 0.98 |

Why: one frontier episode on v8 scored 4 of 8 in 29 of 120 minutes by building its own Go1 model and rehearsing
whole missions in it, which the instruction forbade and nothing could check; the same program docked 21 of 32
development sites. A prohibition a grader cannot check measures obedience, so it is gone, and the difficulty is
moved to what no rehearsal fixes in advance: each site's robot and fittings have to be measured there.

## Controls (`controls.py`, `replay_submission.py`)

Measured in the task's image; graded sites as text records only. None is a participant program or a gate.

| controller | graded sites (8) | development sites |
|---|---|---|
| oracle (`solution/oracle`, with truth) | 8 | 46 of 48 (two scanner-on-floor contacts under low ducts) |
| oracle without stance-height feedback | 7 | |
| oracle without pitch trim | 5 | |
| oracle without either | 4 | |
| oracle with the fittings' nominal clearances | 4 | |
| oracle without either feedback and with nominal clearances | 1 | |
| public reference (nominal load and postures) | 1 | |
| the saved v8 frontier program, replayed without a model call | 0 | 8 of 32 (was 21 of 32 on v8) |
| do nothing, survey only, stand still | 0 | |

On the graded sites the oracle uses 27,115 to 30,269 ticks, keeps 14 to 26 mm between the guard and the duct's
underside and puts four feet on every deck.

What the table says: a controller that knows the site but does not correct its stance from its joint angles and
gravity reading loses half the sites; one that assumes the nominal fittings loses half; with both it is left
with one. Friction, in the stated range, changed no controller's result here and is kept as part of the robot's
description, not as a lever.

## Isolation probe

A submission that writes a marker into every directory its account can write (and plants desktop start-up
files) and reports what it finds on the next site: under v8's grader every later site read the earlier ones'
markers in /tmp, /var/tmp, /dev/shm, the account's home and the lock directory; under this grader none does
(46 to 142 entries removed after each mission). It gets no inet socket and cannot read the tests, the records or
the robot models.

## Machines

`tests/numerics_check.py` gives one fingerprint on an AMD Zen 3 and an AMD Zen 5 host (simulation, camera frames
and a program's NumPy / SciPy arithmetic). Before the pin, the same controller's dock errors differed between
the two in the ninth decimal.

## Reproduction

In the task's image, as root, from the task's folder:

```bash
/opt/venv/bin/python3 -B tests/review/controls.py oracle --hidden
/opt/venv/bin/python3 -B tests/review/controls.py oracle --n 48
/opt/venv/bin/python3 -B tests/review/controls.py oracle:noheight+notrim+nominalfit --hidden
/opt/venv/bin/python3 -B tests/review/controls.py public --hidden
/opt/venv/bin/python3 -B tests/review/replay_submission.py <a submission folder> <out dir> --n 32
/opt/venv/bin/python3 -I -B tests/numerics_check.py
```
