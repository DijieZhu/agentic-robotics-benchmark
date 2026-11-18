# Observability of the hidden episodes: the I3 camera probe (summary)

Two rounds of the same camera-only program on the four hidden episodes, each episode once per round. Round 1
missed the design gate (mean >= 0.6 and every class fastened at least once) by 0.002; both of its losses were
defects of its overhead-camera stage. Round 2 (written at the task's finalization, by the lead designer's decision)
repairs exactly those two defects and changes nothing else: (a) a fixture board whose residual in the overhead
camera's joint fit stands out of the other boards', or whose fixture lands outside the placement bounds the cell
documents, is rejected (a board not seen alike); its fixture is then kept out of every planned move over those
bounds until the wrist camera has looked at its board; (b) when the overhead height map does not give four
wheels and the stick pack, the wrist camera surveys the bench. Round 2 was written after round 1's failures
were known: it is not a blind run. It was developed and tested on layouts outside the hidden set only, its code
frozen before its first hidden episode, and every hidden episode's result is reported as it came.

**What the probe knows and what it does not measure.** Every per-episode quantity (the two cameras'
calibration, the fixtures' and the car's places, every unit's pose, the hubs' angles) comes from the two cameras
over the public RPC and from the robot's own state. Its static design data are the hidden kit's, not the given
files': the places' design frames, the bodies' design poses, the collision pieces its planning twin checks the arm
against and the reference program's grasp table; its moves are the reference program's motion layer (grasps,
inverse kinematics, planner) driven by the probe's estimates. The probe assumes that an agent derives equivalent
static data from the drawings and the part files; it does not measure that derivation. So the gate's "public
information only" half is shown for the perception, not for the step from the documents to the homes, the grasps
and a collision model. Round 1 was developed on hidden episodes 0 and 1, so its two 1.000 there are not blind
either. Independent partial evidence for that step (an independent review's program with the public API and the
given files only, on the LiDAR's sub-assembly route) is in `tests/README.md`.

| | round 1 | round 2 |
|---|---|---|
| mean score (stage score x hard_ok) | 0.598 | 0.717 |
| episode 0 | 1.000 (8 of 8 places, safety ok) | 1.000 (8 of 8 places, safety ok) |
| episode 1 | 1.000 (8 of 8 places, safety ok) | 1.000 (8 of 8 places, safety ok) |
| episode 2 | 0.391 (3 of 8 places, safety ok) | 0.870 (7 of 8 places, safety ok) |
| episode 3 | 0.000 (3 of 8 places, a safety violation) | 0.000 (0 of 8 places, a safety violation) |
| battery places fastened | 2 of 4 | 3 of 4 |
| deck places fastened | 3 of 4 | 3 of 4 |
| compute places fastened | 3 of 4 | 3 of 4 |
| lidar places fastened | 3 of 4 | 3 of 4 |
| wheel places fastened | 8 of 16 | 11 of 16 |
| design gate | missed | passed |

What the two repairs did in round 2 (the probe's own record):

| episode | overhead boards rejected | pre-looks that placed a fixture | wrist survey of the bench |
|---|---|---|---|
| 0 | shelf: place, stand: place | 2 of 2 | not needed |
| 1 | none | 0 of 0 | not needed |
| 2 | none | 0 of 0 | 19 views: 3 wheels, 1 stick pack |
| 3 | shelf: residual and place, tray: place | 2 of 2 | 24 views: 3 wheels, 1 stick pack |

## Round 1: where the points were lost (ranked by the mean score lost)

| rank | mean score lost | category | what happened | document gap / window? |
|---|---|---|---|---|
| 1 | 0.250 | observability (overhead camera) + probe defect | a spurious grid accepted as the shelf board in the single overhead frame put the shelf far off in the probe's planning model; moving to a calibration view, the arm swept through the real deck: safety violation, episode voided | neither: the manual bounds the shelf's place (within +-20 mm of (0.30, -0.45) m, turned by up to +-5 deg) and calls the overhead camera uncalibrated; the estimate lay about 0.2 m and 70 deg outside that bound |
| 2 | 0.152 | observability (overhead depth) + probe defect | the overhead height map found no wheel (the wheels merged with false height bumps of the depth error) and took a merged wheel blob for the stick pack; the probe had no wrist search for unseen bench units: four wheels not attempted, the battery grasp closed on a wheel | neither: the manual states the overhead depth's noise and error field; with those numbers a 16 mm height threshold at 1.6 m is within about two sigma |
| 3 | 0.000 | window (checked, not binding) | no window was binding: every place the probe attempted with a correct identification and an undisturbed unit was fastened (e0 8 / 8, e1 8 / 8, e2 3 / 3, e3 3 / 3 before the thrown deck mattered); the weakest estimate, the deck on its shelf (about 1.9 mm, 0.8 deg against 3.0 mm, 2.2 deg), was fastened every time it was picked | no change suggested by I3 |
| 4 | 0.000 | document gap (checked, none found) | every documented number the probe relied on held in the cell: the board geometry and orientation, the wrist mount description, the depth model, the wrench's frame and bias, the stop 2.5 mm short of home, the rest heights within 0.9 mm, the search at about 4 N and 10 deg/s | none |

# Round 2 in detail

## End to end

| quantity | value |
|---|---|
| episodes run | 4 |
| mean score (stage score x hard_ok) | 0.717 |
| episodes with a safety violation | 1 |
| battery places fastened | 3 of 4 |
| deck places fastened | 3 of 4 |
| compute places fastened | 3 of 4 |
| lidar places fastened | 3 of 4 |
| wheel places fastened | 11 of 16 |
| design gate (mean >= 0.6 and every class fastened at least once) | passed |

| episode | score | stage score | safety | places fastened |
|---|---|---|---|---|
| 0 | 1.000 | 1.000 | ok | 8 of 8 |
| 1 | 1.000 | 1.000 | ok | 8 of 8 |
| 2 | 0.870 | 0.870 | ok | 7 of 8 |
| 3 | 0.000 | 0.000 | violated | 0 of 8 |

## Where the points were lost (ranked by the mean score lost)

| rank | mean score lost | category | what happened | document gap / window? |
|---|---|---|---|---|
| 1 | 0.250 | probe defect (wrist calibration seeded by a repair's fallback) + safety | the tray board was rejected (its overhead place read just outside the documented bound), the uncalibrated wrist pre-look placed the tray about 2 cm off, and the wrist calibration seeded from that place diverged and was accepted without a residual check: every later wrist measurement was centimetres off, the deck, compute module and LiDAR picks closed on nothing, a LiDAR pick pressed the unit into its nest above 150 N: safety violation, episode voided | neither: the tray stood inside its documented bounds; the wrist camera and the boards are as the manual says; no window was involved (nothing reached its place) |
| 2 | 0.033 | probe defect (wrist survey coverage) | the overhead height map found no wheel (as in round 1) and repair (b) surveyed the bench: 19 views found the stick pack and three wheels; the fourth wheel was seen only in part (67 x 46 mm, below the wheel's footprint) and was never labelled, so one wheel place was not attempted | neither |

## What the probe measured, and how well

| quantity | how it is observed | mean / max abs error | the window it feeds |
|---|---|---|---|
| overhead camera: bench-plane pixels | one frame: the surveyed bench board and the three fixture boards at their documented heights, with the depth | 2.10 / 3.83 (n 4) px | coarse places only |
| wrist camera: hand-eye position | views of the tray board from about a dozen tcp poses (the tcp is exact) | 6.976 / 27.605 (n 4) mm | every wrist measurement |
| wrist camera: hand-eye angle | same | 1.261 / 4.926 (n 4) deg | |
| wrist camera: pixel error 0.3 m ahead | same | 0.676 / 2.608 (n 4) px | |
| work stand place (overhead only) | its board in the overhead frame | 8.32 / 20.39 (n 4) mm, 0.379 / 1.010 (n 4) deg | |
| work stand place (wrist) | its board from the wrist camera | 0.43 / 0.78 (n 4) mm, 0.075 / 0.134 (n 4) deg | |
| deck shelf place (overhead only) | its board in the overhead frame | 8.83 / 17.99 (n 4) mm, 2.757 / 4.979 (n 4) deg | |
| deck shelf place (wrist) | its board from the wrist camera | 1.20 / 4.37 (n 4) mm, 0.117 / 0.366 (n 4) deg | |
| parts tray place (overhead only) | its board in the overhead frame | 12.13 / 36.67 (n 4) mm, 0.817 / 2.577 (n 4) deg | |
| parts tray place (wrist) | its board from the wrist camera | 0.11 / 0.39 (n 4) mm, 0.018 / 0.056 (n 4) deg | |
| the car on its stand (stand board + the drawing) | the clamping repeatability is not observed | 0.63 / 1.41 (n 4) mm, 0.111 / 0.189 (n 4) deg | deck 3.0, stick pack 3.2, wheels 2.0 mm |
| the car on its stand (part file fitted to depth) | `roller.stl` against a wrist depth view from above | 1.16 / 3.97 (n 4) mm, 0.082 / 0.134 (n 4) deg | |
| the car (the four hub hexes) | wrist close-ups along the axles | 0.05 / 0.11 (n 3) mm, 0.018 / 0.024 (n 3) deg | |
| compute module in its nest | its part file fitted to a wrist depth view | 3.82 / 15.13 (n 4) mm, 0.119 / 0.227 (n 4) deg | 2.5 mm, 3.1 deg |
| LiDAR in its nest | its part file fitted to a wrist depth view | 3.24 / 12.77 (n 4) mm, 1.378 / 4.932 (n 4) deg | 2.5 mm, 5.0 deg |
| deck assembly on its shelf | its part file fitted to a wrist depth view | 4.87 / 14.77 (n 4) mm, 1.217 / 2.970 (n 4) deg | 3.0 mm, 2.2 deg |
| stick pack on the bench | its part file fitted to a wrist depth view | 7.63 / 30.25 (n 4) mm, 0.429 / 1.580 (n 4) deg | 3.2 mm (lengthwise not guided) |
| wheels on the bench | its part file fitted to a wrist depth view | 1.95 / 12.49 (n 13; left out: 1 gross) mm | 2.0 mm |
| bench units (wheels, stick pack) the overhead view did not find | the overhead depth's blobs: never measured, never attempted | 6 of 20 | |
| hub hex: centre across the axle | the hex's outline in a wrist close-up's depth (its colour is too dark to use) | 0.10 / 0.15 (n 12) mm | wheel 2.0 mm across |
| hub hex: outer face along the axle | same | 0.25 / 0.44 (n 12) mm | the stop 2.5 mm short of home |
| hub hex: turn (mod 60 deg) | same | 1.34 / 4.44 (n 12) deg | +-6 deg (with the pocket's turn) |

The wheel's pocket turn is not observable from outside (the rim's outer face repeats every 36 deg, the pocket
every 60 deg). The probe found it by the documented search: on the stop, the push held at 4 N on the F/T sensor,
the hand turning the wheel at 10 deg/s: 11 searches, 11 caught.

## Budgets

| episode | frames (of 450) | sim s (of 650) | wall s (of 3100) | longest silence s (of 120) |
|---|---|---|---|---|
| 0 | 32 | 250.1 | 568 | 15.5 |
| 1 | 30 | 260.5 | 562 | 8.8 |
| 2 | 48 | 271.4 | 668 | 18.2 |
| 3 | 55 | 425.6 | 1119 | 13.8 |

All within the budgets: yes.
