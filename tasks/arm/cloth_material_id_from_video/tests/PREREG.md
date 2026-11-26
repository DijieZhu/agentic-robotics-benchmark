# Pre-registration of the hidden set (verify / maintainers only; never shown to the agent)

This is the second registration. A first one (commit `9304ab1e`) was followed by a draw that was stopped about ten
minutes after its start, before any of its results was read: a check of the arm found that drags towards the far
side of the table took the arm through the rod that stood there and that pulls over the front edge were outside
the arm's reach for half of the registered range. The rod is gone from the scene, the robot stands at the table's
front corner, the drags run towards the front edge and every fabric pose lies in the arm's working area; these
change files the first registration froze, so that draw is void and the key below is a new one.

Written and committed before any candidate of the hidden draw was simulated. The commit that adds this file also
adds the key (`tests/ref/key.json`) and the generator (`tests/review/make_set.py`); the hidden set
(`tests/ref/fabrics.json`, `tests/ref/selection.json`) comes in a later commit. A change of `tests/cphys.py`,
`tests/creal.py`, `tests/ctasks.py` or `tests/review/make_set.py` after this commit voids a draw made before it.

## What was decided on development draws only

On eight development fabrics with their own seeds (none can come out of the key): the two scored skills (slide,
overhang), their control ranges, target bands and tolerances (3 mm, 6 mm), the product of the two skill means as a
fabric's score, the fabric population, the reference's calibration and the keep rule below. The three test families
of the first version are gone: the corner hang carried about 20 mm of material signal against up to 20 mm of
difference between the planning simulator and the real fabric, and laying a fabric back over the table edge
bifurcates in both models.

## The rule

Every random number comes from `sha256(key + "/" + label)`; the labels are in `make_set.py`.

1. Fabric candidates 0, 1, 2, ... are drawn from the population: bending length log-uniform 3-16 cm, areal density
   uniform 0.06-0.40 kg/m2, Young's modulus log-uniform 1e4-5e5 Pa (the thickness follows from the three), Poisson's
   ratio uniform 0.2-0.4, damping log-uniform 1e-3 to 8e-3 s, friction uniform 0.2-0.9.
2. Each candidate gets three calibration tests and eight test candidates per skill, each with the real outcome on
   nine control values. Test poses are drawn in the arm's working area: slide, the fabric's centre at x -0.24 to
   -0.14 m, y -0.20 to 0 m, turned by up to 0.25 rad, dragged by its leading short edge towards the front edge,
   released after 0.09 to 0.14 m; overhang, the centre at x 0.10 to 0.13 m, y -0.20 to 0 m, turned by up to 0.12 rad,
   lifted 12 to 20 mm, pulled at 0.08 to 0.14 m/s. A candidate fabric is rejected if a real simulation of a calibration test fails.
3. A test candidate gets a target if its real curve has a monotone stretch between two neighbouring control values
   with both outcomes inside the skill's band (slide 35-110 mm; overhang 12-80 mm below the table top) and more than
   2 mm apart; one such stretch and a control value on it are drawn, and the target is the real outcome there.
4. The reference of a fabric is the planning-simulator material with the real material's Young's modulus, Poisson's
   ratio and damping and with friction and thickness calibrated against the real curves of the calibration tests
   (Nelder-Mead from the real values, 45 evaluations).
5. Test candidates are examined in order. One is kept if it has a target and the reference scores 1.0 on it. The
   first four kept per skill are the fabric's tests.
6. A candidate fabric that keeps fewer than four of its eight candidates in either skill is rejected whole and the
   next candidate is examined; no fabric is scored on fewer than four tests per skill.
7. The first ten accepted candidates are the scored fabrics A-J in that order; the eleventh is the fabric P.

Reported with the draw: the candidates examined, each rejection with its reason, and per skill the share of test
candidates on which the reference scored 1.0.

## Checks registered before the draw (run on the hidden set by exact execution, reported whatever they show)

| submission | expected |
|---|---|
| the reference | 1.0 |
| no file | 0.0 |
| the real materials as they are | below the reference, above 0.5 (the planning simulator's coarse grid bends more easily; the calibration answers with more thickness) |
| the reference with friction x 0.95, x 0.9, x 1.1 | falling; about 0.8 at 5 %, under 0.5 at 10 % |
| the reference with thickness x 0.9, x 1.1, x 1.2 | falling; under 0.5 at 20 % |
| the reference with Young's modulus x 2, x 0.5 | under 0.5 |
| the reference with damping x 3; with Poisson's ratio x 0.7 | above 0.8 (these two are barely scored; reported) |
| friction from the reference, stiffness blind (thickness and Young's modulus of the population's middle) | near 0: the slide mean near 1 and the overhang mean near 0 give a product near 0 |
| stiffness from the reference, friction blind (0.55) | near 0, the reverse |
| each fabric's reference written for all ten fabrics | about 0.1 each (its own fabric), not more |
| the middle of every admissible range; the first version's example material; the first version's practice material | at most 0.1 |
| the best constant on a grid of friction (0.2-0.9, 8 values) x thickness (7 values, Young's modulus 1e5 Pa) | at most about 0.10 |

If the reference is not 1.0, or a blind constant is above 0.15, the design is wrong: stop, report, change nothing
silently.

## The recordings and the look

The hidden draw depends on the physics only (`cphys`, `creal`, `ctasks`, the key). The agent's recordings are
rendered once from the drawn fabrics and published by checksum. A later re-render with other art (the same fabric
states, cameras, arm motion and camera degradations) replaces the recordings and their checksums and is followed by a
new validate; it does not touch the hidden set, the reference or the anchor.

## Amendment 1 (2026-11-26): the recordings' look; rendering only

After the draw, the checks and a first set of recordings (MuJoCo's native renderer, procedural prints), the
recordings are rendered again with the scene approved for this task: a workshop room, a white bench, the robot's
pedestal, the clamp as a machined part, drawn by MuJoCo's Filament renderer on the CPU from a render bundle
(`environment/gen/crender.py`, `cscene.py`, `cgeom.py`, `cprim.py`, `clook.json`; the bundle's checksums are in
`environment/gen/render_bundle.sha256`). This is the re-render the section above announced.

- **Nothing of the physics, the hidden set, the reference or the anchor changes.** `tests/cphys.py`, `tests/creal.py`,
  `tests/ctasks.py`, `tests/review/make_set.py`, `tests/ref/key.json`, `tests/ref/fabrics.json`,
  `tests/ref/selection.json` and `solution/ref/params.json` are the bytes of the draw's commit; the fabric states in
  the recordings are the same simulations (the clamp programs, the seeds and the camera effects' random numbers are
  unchanged), and so are the cameras and the arm's joint angles.
- **The looks.** The fabrics' looks are drawn from a library of twelve (four classes: canvas, denim, felt, coated
  fabric; three colours each; `looks.json` in the bundle). A fabric's look is a function of the fabric ids alone: the
  ids are put in the order of sha256("cloth_material_id_from_video/look/v1/" + id) and the k-th wears look k mod 12
  (`cscene.fabric_looks`). The ids were dealt before the materials were drawn, so a look carries nothing of a
  fabric's stiffness, weight or friction. Every fabric is drawn 1.5 mm thick.
- **What must hold and is checked again:** the reference scores 1.0 with the same outcome on every hidden test
  (`ale validate`; the grader does not read the recordings); the arm checks of every clip; the readability of the two
  scored quantities in the new pictures; the recordings' determinism.
