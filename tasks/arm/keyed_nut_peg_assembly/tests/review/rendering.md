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

- **The episode's own parts**: a copy of the episode's compiled model with render-only fields edited (the robot's
  visual meshes and not its collision shapes, one material per nut and per peg part in the unit's own colours, PBR
  terms), in the simulator's state at the moment of the image. The physics model is read and never written.
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

`test_draw_determinism.py` (in the task image, as root): two processes, one importing the grader's copy and one the
service's, the same two kits, the arm moving; one draws each camera once per step, the other draws an extra image
first, each camera ten times, then another order with repeats. 36 images compared, 0 differ; 0 repeated draws
differ. Filament's temporal dithering (a new noise per frame) is switched off for this (`keyed_render._undither`).

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

## Running the checks again

```bash
# in the task image, as root, the task's tests/ mounted at /task/tests
python3 -B /task/tests/review/test_draw_determinism.py
python3 -I -B /task/tests/review/observability/check_perception.py --first 1000 --n 30
# host: the repository's tests of this task
uv run pytest -q tests/test_keyed_nut_peg_assembly_commissioning.py tests/test_keyed_nut_peg_assembly_hidden_set.py
```
