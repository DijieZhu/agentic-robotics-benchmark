# The cameras' renderer (maintainer record, 2026-11-18)

Asked for by the benchmark's owner: the agent's three cameras show the cell as a camera would see it (the robot in
its own covers, real materials, the laboratory around the bench, the light of that room) instead of MuJoCo's own
renderer with the robot's grey collision hulls on a checkered floor. Only the images change. This note says what
draws them now, what was held fixed and how that was checked. Every number below was measured on maintainer kits or
on the commissioning service's development kits; no graded kit was rendered.

## What draws an image

`tests/keyed_render.py` (a byte copy runs in the commissioning service): MuJoCo 3.16's Filament renderer on Vulkan,
which in the task image is Mesa's lavapipe on the CPU. `keyed_env.NutEnv.render` calls it and adds the sensor noise
as before.

- **The episode's own parts**, read from the episode's compiled model in the simulator's state at the moment of
  the image (the physics model is read and never written): the robot's visual meshes and not its collision shapes,
  the peg parts, every nut as one mesh of its blocks, with the look's PBR terms and the unit's own colours. The
  process makes their renderables once and an episode only writes data into them ("An image must not depend on
  the episodes before it", below).
- **The room** (`render/room.npz`, `room.json` of the render bundle, 303,646 triangles, 34 colour maps): a
  laboratory in a loft, its shelving and desks, the cell's oak bench top and cabinet, the robot's pedestal, the camera
  gantry with its lamp. Drawn only; the model's floor plane, wall, table block and pedestal block are not drawn, the
  bundle's bench top and pedestal stand exactly where they are. Sources: `../NOTICE.md`.
- **The light**: an image light (`render/ibl.ktx`, a path-traced probe of the room over the bench, the gantry's lamp
  off, the robot and the parts taken out) and the gantry's lamp, a bar, as two spot lights with a shadow map each;
  ambient occlusion on. The numbers are in `tests/keyed_look.json`.
- **The bundle** is fetched at the dataset commit pinned in `environment/assets.sha256` into
  `/opt/task-assets/payload/render`, readable by root (the grader) and by the service's account, not by the agent
  account, and never staged into the agent's home.

## What did not change, and how it is kept

| | how |
|---|---|
| physics, the wire, the takt, the kits | the renderer reads `model` and `data` only; amendment 5 (below) repeats every selection run and the anchor to the step |
| camera poses and intrinsics with their errors of the day | Filament is given the camera MuJoCo itself computes for the episode's model (`mjv_updateCamera`: position, axes, off-centre frustum, the model's near plane); edges fall on the same pixels as with MuJoCo's renderer |
| image sizes | the model's own camera resolutions (320 x 240, 256 x 192); an image is drawn at twice the size and averaged in linear light |
| sensor noise | the same line in `keyed_env.render`, from the episode's generator |
| published calibration, `T_world_wrist` | untouched (`cell_data.py` is unchanged) |
| the lamps' brightness of the day (+-15 %) | `NutEnv.lamp_gain` multiplies the image light and both lamps; at -15 % / +15 % the images' mean level (decoded to linear) is 0.90 / 1.09 of nominal after the display curve |
| the service and the grader draw the same image | byte copies of `keyed_env.py`, `keyed_render.py`, `keyed_look.json` (a test); `test_draw_determinism.py` |

`test_draw_determinism.py` (in the task image, as root): three processes. Two draw the same two kits with the arm
moving, one importing the grader's copy and one the service's; one draws each camera once per step, the other draws
an extra image first, each camera ten times, then another order with repeats. The third is the grader's way: a new
`NutEnv` per kit in one process, the two kits in turn twelve times (24 loadings), the same state drawn each time.
36 images compared between the first two, 0 differ; 0 repeated draws differ; 0 of the 72 images of the third
differ from their kit's first, and its images are those of the first process, where each kit was drawn after a
different number of episodes. Nine runs (three of them on the final file), nine passes. Filament's temporal
dithering (a new noise per frame) is switched off for this (`keyed_render._undither`). The third process was
added on 2026-11-19; the renderer of amendment 5 fails it (below).

## The look and how close it is

The reference is the path-traced look of the same laboratory that the task's demonstration video uses, rendered
through the agent's own three cameras (the cameras of the day of one development kit) at twelve states of that
kit's episode: once with all lights, once with the room alone, once with the lamp alone, plus the light probe.
Fitted in this order, in linear light: the image light against the room-only references, the lamps against the
lamp-only references, then one gain per class of surface against the references with all lights; nine states for
the fit, three held out. The display curve is Filament's filmic one, which matched the references' display images
without a further correction.

Median colour per class of surface, Filament against the reference's display image, both decoded to linear
(luminance ratio; held-out states in brackets), with the final materials at the references' exposure:

| bench top | floor | robot, white | robot, black | trolley | totes | shaft | collar | hand | steel parts | cabinet | pedestal | gantry |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.02 (1.01) | 0.99 (0.98) | 1.03 (0.94) | 1.05 (1.03) | 0.99 (1.02) | 0.98 (1.02) | 0.96 (1.84) | 0.98 (1.02) | 0.92 (0.88) | 1.11 (1.03) | 1.15 (1.15) | 1.14 (1.33) | 1.29 (1.36) |

Two things are set apart from the fit on purpose, for what the task says the cameras show:

- **Exposure 0.75 of the references'.** At the references' exposure a warm nut's brightest channel clips under
  the lamps, and a clipped channel moves the hue (red towards orange).
- **The nuts a quarter darker and matt** (gain 0.75 against the fitted 1.07, roughness 0.8 against 0.5). A glossy nut
  mirrors the lamps and the room in white, which washes its colour out.

What the image light cannot do, and so differs from the reference: the sun's patches on the floor are not there;
the arm's shadow on the bench is two half shadows with firm edges where a long lamp gives one wide, faint shadow
(in the three held-out states the hand stands over a shaft, which the reference shades more: 1.84 there);
the cabinet under the bench, the pedestal and the gantry are 15 to 35 % too bright.

## The facts the task states about its images, measured before and after

Same code on both renderers (the old one is the previous commit's `keyed_env.py`), real kits with every tolerance
drawn, images with the sensor noise unless stated.

**The eight colours** (16 kits running through the eight colours, each nut in each camera at the start pose: 144
samples; the median colour of the nut's pixels).

| | old renderer | new |
|---|---|---|
| named right from the published RGB table, by nearest hue | 143 of 144 | 141 of 144 |
| named right by the nearest of eight colour centres learnt on the other kits | 109 of 144 | 133 of 144 |
| smallest gap between two colours' ranges of hue | -9.6 degrees (orange and yellow overlap) | 8.4 degrees (red and orange) |
| saturation, mean per colour | 0.50 to 0.80 | 0.44 to 0.80 |
| hue seen against nominal | within 6 degrees | red +8, teal -8, green -18, purple +18, the rest within 6 |

The warm light moves the hues, which the data sheet now says; the eight stay apart.

**The bands** (12 kits, each peg in each fixed camera: the band pixels' luminance against the shaft's).
Bands darker than 0.6 of the shaft in at least one fixed camera: 36 of 36 pegs before and after. Ratio, median:
0.28 before, 0.35 after (a band is about one pixel tall there and the new images are anti-aliased; the shaft is near
white now, 198 against 101 of 255).

**The groove** (6 kits, 18 nuts, without the sensor noise; contrast = (wall - groove) / wall in luminance; d' counts
the sensor noise of 2 levels in).

| view | | old renderer | new |
|---|---|---|---|
| down the hole, 62 degrees above the horizon, 0.14 m away, eight azimuths (29 views with enough pixels) | contrast p10 / p50 / p90 | 0.03 / 0.17 / 0.28 | 0.20 / 0.40 / 0.59 |
| | d' p50 | 0.67 | 1.38 |
| the bottom face, square on from 0.14 m (31 views) | contrast p10 / p50 / p90 | 0.32 / 0.37 / 0.44 | 0.27 / 0.46 / 0.52 |
| | d' p50 | 2.36 | 4.54 |

Both readings are there; down the hole a cut cell now reads more clearly than before, because a recess under an
image light with ambient occlusion is in shade as a real one is, where the old renderer lit it like the wall around
it. This is the one place where an image carries more than it did.

**Camera-only perception** (`observability/`): 30 maintainer kits, 90 pegs and 90 nuts found, 90 of 90 band counts
right; the table is in `observability/README.md`. `perceive.py` had to be refitted to the new images (the colours'
hues as seen, the oak bench, the band count from the profile down a shaft); the old rules do not work on them.

## What an image costs

In the task image under the task's limits (4 CPUs, 12 GB, lavapipe on 2 threads), two episodes in one process, the
host busy with other work (load 27 to 33 of 64 threads); an image's cost is a step with it minus a step without.

| | old renderer | new |
|---|---|---|
| `left` / `right` / `wrist` image | 0.14 / 0.14 / 0.15 s | 0.43 / 0.40 / 0.33 s |
| a step without an image | 0.021 s | 0.020 s |
| first reset of a process (its three images) | 2.3 s | 7.2 s |
| later resets | 2.0 s | 1.4 s |
| resident memory, peak | 656 MB | 809 MB |

Most of the new cost is the two shadow passes (every mesh of the robot is drawn into each lamp's shadow map), then
the room. The stated charges were 0.45 s (`left`, `right`) and 0.25 s (`wrist`), measured with the old renderer on
a much busier host. A fixed-camera image still costs about its charge; a wrist image no longer does, so its charge
is 0.35 s now (`tests/grader_config.json`, `kits.json`, `run_kit.py`, instruction, data sheet). Nothing else in the
budget changed. The reference asks for no image.

## Amendment 5

`tests/hidden/selection.json`, amendment 5, pre-registered in the commit before its runs: the frozen list gained
`tests/keyed_render.py`, `tests/keyed_look.json` and `environment/assets.sha256`. The kits are unchanged, so the
rule ran on the same candidates:

- hidden candidates: the same 12 of 16 completed, each with the same steps; the same set (F00 to F05, F07, F10 to
  F12);
- development candidates: the same 22 of 24, each with the same steps; the same 16 kits (`kits.json` is byte for
  byte what it was but for the wrist charge);
- the reference's full verify: 10 of 10, steps 775, 738, 645, 654, 675, 644, 714, 707, 720, 802, each episode
  repeating its selection run and amendment 4's anchor to the step; measured 1.0; 243 s (223 s before).

## An image must not depend on the episodes before it (2026-11-19, amendment 7)

**What was wrong.** The renderer of amendment 5 loaded every episode's model into MuJoCo's `ModelResources` and
`ModelRenderables` anew. From a process's second episode on, an image was not byte for byte the image of the same
state in a process's first episode. Measured on that renderer in the task image, one process, maintainer kits in
turn, a new `NutEnv` each time and the same state drawn after every loading:

- 320 loadings in four processes: 201 of 960 images differ from the first image of their view, by 1 to 6 pixels
  (median 1) of one or two levels. In each process it is the kit of the first episode whose later images differ;
- the pixels are on the nuts. With the nuts not drawn, 0 of 180 images differ; with the robot or the pegs not
  drawn, the differences stay;
- 15 fresh processes drawing the same two kits give one picture per view: a process repeats itself, what counts
  is how many episodes it drew before;
- with the test's third process on that renderer, six runs: each fails, 12 of 72 reload images differ, and 3 of 6
  views differ between a kit drawn first in its process and the same kit drawn after another one.

The commissioning service draws every episode in a process of its own; the grader draws its ten kits in one. So
for the same state the two could differ in those pixels, which the task says they do not. The differences are
below the sensor noise (2 levels on every pixel); what was wrong is the statement, not what a controller could do
with the images.

**Why.** A nut is some four hundred blocks that overlap by design, each a renderable of its own. Where two
surfaces compute the same depth at a sample, the one Filament draws first shows, and the order in which it draws
two renderables is not a function of the scene. Three things changed it in the experiments: renderables made
again for a later episode (the defect); a renderable given a material again; and, for a nut drawn as four hundred
renderables made once, the process (one pixel in some views in 9 of 39 processes).

**The fix** (`keyed_render.py`, "PARTS"). Nothing that is drawn is made per episode. The process makes every
renderable, material, mesh and texture once, at its first episode and in a fixed order; an episode writes data
into them. The robot: MuJoCo's own renderables of a robot-only copy of the process's first model, moved to each
state's poses (every episode must have the same robot at the same geom indices; the renderer checks). The peg
parts: MuJoCo's own box and cylinder shapes, resized per episode. A nut: one mesh of all its blocks, written anew
per episode into a mesh of fixed size, so that two of its blocks at the same depth are drawn in the mesh's own
order; its colour of the day is a texture of one texel, written anew per episode, because a renderable must keep
the material it was made with.

**Checked** on the fixed renderer, in the task image:

- the same reproduction: 368 loadings in five processes (four of 80; one with eight kits in the other order), 0
  of 1,104 images differ from the first of their view, and the 24 views drawn in two processes are the same in
  both;
- 24 fresh processes: one picture per view;
- `test_draw_determinism.py` with its third process: nine runs, nine passes.

**What changed in the images.** Against the renderer of amendment 5, 54 images of six kits at three states: 1.16 %
of the pixels differ: 40,420 by one level, 1,246 by two, 408 by three or four, 343 (0.009 %) by five or more, at
most 27; the mean difference is under a quarter of a level per channel. 93 % of them are on a nut and 98.6 % on a
nut or within two pixels of one; the rest, by one level (at most 4), are on the bench around the nuts and along
the seams between the fingers and their pads. Two causes: a nut's colour is held in 8 bits per channel now (the
texel), where it was a float; and where two surfaces tie, the order is another one (for a nut's blocks the
model's order). The robot, the pegs and the room are drawn by the same shapes, meshes and materials as before.
The camera-only perception check repeated on the 30 kits: every
peg number identical; the nuts' ring centres move by 0.03 mm in the median (0.07 mm at the 90th percentile, 1.2 mm
at most), and one nut of 90 whose handle was read 6 degrees off is now read the wrong way round
(`observability/README.md`).

**Cost.** Three episodes in one process under the task's limits, the same hour, the same host load (about 10 of
64 threads): `left` / `right` / `wrist` 0.37 / 0.33 / 0.27 s before and 0.30 / 0.26 / 0.21 s now (a nut is one
draw instead of four hundred), later resets 1.26 s and 0.97 s, peak resident memory 808 MB and 735 MB. The stated
charges (0.45 s, 0.35 s) are unchanged.

**The kit's clearing.** The verify-stage kit clears the agent's account once per verifier sandbox (a few seconds).
It ran inside the first kit's timed solver start, so that kit's controller time read about 4 s instead of 0.5 s of
its 600 s. `verify.py` now calls `clear_agent_account()` once before the first kit; `Solver.__init__` gets that
report back at once.

**The view search.** `keyed_render._undither` finds Filament's view from the scene object by following pointers.
It walked depth first with one set of visited pointers for all depths, so a block that it met first at its last
level was never read at the level where its own pointers still counted; when that block was the one that holds
the view, the process's first image ended as a `RenderError` (an error of the environment: the grader stops, the
service answers `cell_fault` and counts no episode). Seen once, in about 250 process starts on 2026-11-19: in a
development candidate's run of this amendment's first registration. The search is breadth first now: 200 of
200 fresh processes found the view and drew the same first image.

**Amendment 7** (`tests/hidden/selection.json`), pre-registered in the commit before its runs: `tests/keyed_render.py`
and `tests/verify.py` changed. (It was registered twice. The first registration's runs were complete and repeated
amendment 6 but for one development candidate, U23, whose grader stopped with the `RenderError` above. The view
search was corrected, the amendment registered again before anything ran on the corrected file, and every
candidate ran again; what follows is the second.) The kits are unchanged, so the rule ran on the same candidates: the same 12 of 16
hidden candidates and the same 22 of 24 development candidates completed, each with the steps of its earlier run;
the same hidden set and the same 16 development kits (`kits.json` and `tests/ref/episodes.json` byte for byte).
The reference's full verify: 10 of 10, steps 775, 738, 645, 654, 675, 644, 714, 707, 720, 802, every episode
repeating its selection run and the anchors of amendments 4 to 6; measured 1.0; 224 s.

## Running the checks again

```bash
# in the task image, as root, the task's tests/ mounted at /task/tests
python3 -B /task/tests/review/test_draw_determinism.py
python3 -I -B /task/tests/review/observability/check_perception.py --first 1000 --n 30
# host: the repository's tests of this task
uv run pytest -q tests/test_keyed_nut_peg_assembly_commissioning.py tests/test_keyed_nut_peg_assembly_hidden_set.py
```
