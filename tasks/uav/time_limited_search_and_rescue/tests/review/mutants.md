# What partial and degraded flights score

The privileged pilot (`solution/oracle/pilot.py`) and its mutants (`PILOT_MUTANT=<name>`), each flown once on the eight
graded sites with the grader's own simulator. A mutant degrades one part of the plan; the "stop" mutants are the
partial programs. The table gives the mean over the eight sites; per-site values are not published.

| Flight | Mean progress | How the flights ended |
|---|---|---|
| reference | 1.00 | handoff on all eight |
| `stop_after_entry`: in through the inspection slot, then stop | 0.05 | stopped |
| `stop_upstairs`: up a hoist well, then stop | 0.09 | stopped |
| `stop_over_litter`: fly over the litter, then stop | 0.17 | stopped |
| `stop_after_grasp`: latch the litter, then stop | 0.32 | stopped |
| `stop_after_lift`: lift it to cruise height, then stop | 0.42 | stopped |
| `stop_after_egress`: carry it out under the cargo shutter, then stop | 0.67 | stopped |
| `wrong_tool`: mount the other collet | 0.17 | the collet does not latch |
| `wrong_station`: fly to the closed station | 0.67 | contact with the lowered shutter |
| `no_board_alignment`: never turn the board into the direction of travel | 0.42 | contact with a wall at the first doorway after the lift |
| `release_high`: release from cruise height | 0.67 | unsafe release |
| `collet_fixed`: always the collet on the first stand | 0.69 | handoff where it happens to fit (5 of 8), no latch elsewhere |
| `collet_from_down_view`: the collet chosen from one picture of the down camera over the litter | 1.00 | handoff on all eight |
| `collet_from_forward_view`: the collet chosen from one picture of the forward camera, up to 1.5 m before the litter | 1.00 | handoff on all eight |
| `pose_noise_10cm`, `pose_noise_25cm`: the pilot's pose feedback with 0.10 m / 0.25 m noise | 0.00 | the pilot never settles on a tool stand and stops before entering |

The means above 0.05 include the rooms-searched share (up to 0.03), which depends on the route. The weights are the
ones in force since they moved from reaching the litter to the work on it (entry 0.05, upper storey 0.04, over the litter
0.06, latch 0.15, lift 0.10, carry-out 0.25).

The reference needs 443 to 541 of the 4,400 actions per site and under 2 s of its own time. The anchor is the
reference's mean progress, 1.0; `full_at` is 0.98.

## Since 2026-11-22: the furnished scene and the receiver's ring

The table was flown again in the furnished scene (stock, furniture, the basket litter with its receiver). Every
earlier row came out as it was: the privileged pilot and its mutants meet no furniture.

The three `collet_*` rows are new. The receiver's ring is the flange, and it can now be seen from above through the
guide ring as well as from the side (the former fitting had a cover over it). The two picture-reading mutants fly
to the litter without a collet, take one picture, measure the width of the cyan ring in it, fly back out, mount the
collet that the width calls for, and then fly the reference's plan; everything else is privileged. Down camera
from cruise height: the ring is 10 to 11 pixels wide for the small flange and 24 to 25 for the large one (the
decision is at 17.9). Forward camera: 7 to 10 against 17 to 23 (decision at about 11.8). Both chose right on all
eight sites, so the size is readable in either camera; the forward view's margin is the smaller one. A program that
does not read it and always takes one collet scores 0.69 here (the first stand's collet fits on five of the eight
sites); with the other collet it would score 0.48.
