# Observability: what a controller can see and feel (maintainer evidence, 2026-11-12; rounds 3 and 4 on 2026-11-17; the cameras' renderer on 2026-11-18 and 2026-11-19)

The task's observations carry no object state. This folder shows, on development kits only, that they are enough
for the job: the camera images to find everything, and the hand's position with the wrist sensor to put a nut on its
shaft and to find its groove. The code uses only what a controller receives (the observation's RGB images,
`T_world_wrist`, the published camera calibration of the data sheet, `ee_pos`, `ee_quat`, `wrench`); the simulator's
state is read only to score the estimates.

## The cameras

`perceive.py` (the perception), `check_perception.py` (per kit: the first observation's free images, then wrist
close-ups over each peg and each nut's rough position, with plain actions as a controller would move; every estimate
against the simulator), `perception_dev_kits.json` (30 kits, 90 pegs, 90 nuts, on amendment 5's image: the cameras
drawn by `keyed_render.py`, see `../rendering.md`). These kits
(`sample_kit` seeds 1000-1029, layout seeds + 4000) are maintainer kits: neither graded kits, nor selection
candidates, nor the commissioning service's development kits.

Every episode is a real kit on the cell of the day: the real cameras sit within their tolerances of the published
calibration (data sheet, section 11), so an image position is off by millimetres.

| what | from | result (30 kits) |
|---|---|---|
| peg positions | the wrist camera's view from the start pose | 90 of 90 found; error 4.6 mm median, 9.5 mm p90, 13.3 mm max |
| which peg is which (1, 2, 3 bands) | the fixed left and right cameras | 90 of 90 right |
| a peg's shaft top | one wrist close-up from above | 1.7 mm median, 3.0 mm p90, 4.9 mm max |
| the same | two close-ups, the hand turned half a turn between them (about 60 steps) | 0.7 mm median, 1.2 mm p90, 1.6 mm max: the wrist camera's error is fixed in the hand's frame, so most of it cancels |
| the same | one close-up, corrected by the offset measured on the kit's first peg | 1.0 mm median, 1.8 mm p90, 2.5 mm max |
| rough nut positions | the fixed cameras | 90 of 90 found; 17 mm median (the blob's centre lies towards the handle), 24 mm max |
| ring centre and handle direction | one wrist close-up over the rough position | centre 1.8 mm median, 3.2 mm p90 (11 of 90 over 3 mm; the worst, 12 mm, a yellow nut whose handle was also read 9 degrees off); handle 0.3 degrees median (one orange nut 29 degrees off), one yellow nut read the wrong way round |

On amendment 5's image (2026-11-18, before the renderer's draw order was fixed: `../rendering.md`) the pegs' numbers
were the same to the last digit; the ring centres differed by 0.03 mm in the median and 1.2 mm at most, and the
yellow nut now read the wrong way round was read 6 degrees off. The fix changes about one pixel in a hundred of an
image, on the nuts, by one level.

With MuJoCo's own renderer (amendments 3 and 4, 2026-11-17) the same kits gave: peg positions 3.4 mm median, 90 of 90
band counts, shaft tops 1.6 mm (one close-up) and 1.1 mm (two), ring centres 1.8 mm median with 17 of 90 over 3 mm and
two handles read the wrong way round. The perception code is not the same: the new images needed their own rules
(the hues the cameras see under the warm light instead of the nominal ones, the oak bench top and its shaded
patches, a band count from the intensity profile down the shaft instead of from single samples).

Before the tolerances (the drawing's cell, 2026-11-12 and 2026-11-16) the old code gave: peg positions 0.5 mm median
(1.3 mm max), ring centres 0.15 mm median. The cameras' calibration error is the difference.

So the cameras find every part and tell the pegs apart, to a few millimetres. The fit has 0.35 to 0.65 mm of radial
clearance: no image position, however the views are combined, puts a hole over a shaft. That takes touch.

## Touch: controllers on the observations alone (2026-11-17)

Two controllers that act on the observations alone (no access to any state when they run), each run by the grader
on the commissioning service's 16 development kits (a one-episode plan per kit through `tests/review/regrade.sh`).

**Under amendment 4 (850 steps).**

- `earlier_submission_dev_kits.json`: the third frontier measurement's submission, unchanged (Claude Opus 5.5,
  2026-11-17, made under amendment 3's 1000 steps). It reads the layout from the images, finds each shaft by touch
  and each groove by touch. **9 of 16 kits complete** (679 to 836 steps), 39 of 48 nuts. It was commissioned on
  these very kits, so that is an in-sample figure; on kits it never met it completes 1 of 6 kept development
  candidates outside the service and 12 of 32 maintainer kits (of the 26 the reference completes there), and 4 of
  the 10 graded kits by the steps of its own graded run. The reference needs 637 to 798 steps on the 16 kits.
- `observation_only_controller_dev_kits.json`: the maintainer's controller described below. **1 of 16 kits**
  (831 steps), 31 of 48 nuts; 2 of 32 maintainer kits. It was written for 1000 steps and is too slow for 850.

So within 85 s the observations still suffice: a controller that has only them completes kits end to end. What
that controller carries is an insertion by touch that round 3's data sheet described; round 4 took the description
out of the data sheet (`../tolerances.md`, "The fit"), and the task now asks the agent to find a way to put a nut
on its shaft, with the fit, the cell's compliance and the wrist's readings stated.

**Under amendment 3 (1000 steps): the maintainer's controller.** It is the second frontier measurement's submission
(Claude Opus 5.5, 2026-11-16), which reads the layout from the first images and wrist close-ups, grips a nut across
its ring and finds the groove by touch, with its insertion replaced; it is not shipped (it is kept with the
review's records):

1. **The first nut of a kit: two touches.** The held nut is brought beside its shaft (the cameras' plan is good to
   a few millimetres, which is enough for that), lowered until its lower rim is below the shaft end, and pushed
   sideways against the shaft from two directions a quarter turn apart. The hand stops and the wrist reads the
   push; where the hand stopped gives the shaft's axis in the hand's own terms (the ring's outside is round about
   the hole's axis, so what the cameras said about the nut in the hand does not matter). The nut then goes on
   directly, or after a small spiral.
2. **The other nuts: a corrected plan.** How far the cameras' plan was off on the first shaft is the same in the
   hand's frame for the kit's other nuts (the wrist camera's error is fixed there). With that correction most of
   them go on directly or within a small spiral on the shaft end; the rest take two touches as well.
3. **The spiral.** A nut that comes to rest on the shaft end (its bottom stops 65 mm above the collar) is slid
   along a spiral of 0.5 mm pitch under a light push, following where the hand is rather than the command (the
   friction on the shaft end holds the hand back), until the nut dips; it then gives way sideways to what the
   shaft pushes and goes down.

With 1000 steps it completed 9 of the 16 development kits (841 to 959 steps), 40 of 48 nuts, and in an in-process
run put 48 of 48 nuts on their shafts; from above the shaft to the pin took 50 steps at the median (two touches 73
to 94; a corrected plan 23 to 51). The second measurement's submission, unchanged, which puts nuts on shafts from
image positions with a search of up to 3 mm in twelve places, completed 3 of 16 (15 of 48 nuts) under the same
rules. (These notes are for maintainers: `tests/` is never shown to an agent.)

What did not work is in `../tolerances.md` (reading the shaft's direction from the wrist's torques on the shaft end).

## Earlier versions

- Amendment 2 (2026-11-17: tolerances, 1000 steps, 1.8 mm of clearance): the second measurement's submission,
  unchanged, completed 9 of its 16 development kits (664 to 992 steps), 12 of 16 with the cameras' error switched
  off. Not hard enough; hence the fit.
- Round 2 (2026-11-16: no tolerances, 1200 steps): the first measurement's submission, made for 2400 steps,
  completed 3 of that round's 16 development kits and assembled 30 of 48 nuts.

Run (inside the task image; `tests/` mounted read-only):

    docker run --rm --network none -v <task>/tests:/stage:ro --entrypoint /opt/venv/bin/python3 <task image> \
        -I -B /stage/review/observability/check_perception.py --first 1000 --n 10
