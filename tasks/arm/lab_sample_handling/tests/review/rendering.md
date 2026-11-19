# The cameras' renderer (maintainer record, 2026-11-19)

Asked for by the benchmark's owner: the bench's four cameras show the bench as a camera would see it (the robot in
its own covers, real materials, the laboratory around the bench, the light of that room) instead of MuJoCo's own
renderer with the robot's blotched collision hulls on a grey block in the dark. Only the images change. This note
says what draws them now, what was held fixed and how that was checked. Every number below was measured on a
maintainer's unit or on the commissioning service's development units; no graded unit was rendered.

## What draws an image

`tests/lab_render.py` (byte copies run in the commissioning service and in the reference's folder): MuJoCo 3.16's
Filament renderer on Vulkan, which in both task images is Mesa's lavapipe on the CPU. `lab_sim.render` calls it and
adds the sensor noise as before.

- **The episode's own bench**: a copy of the episode's compiled model with render-only fields edited (one material
  per kind of part in the part's own colour, with PBR terms), in the simulator's state at the moment of the image.
  Shapes, sizes, places, colours and marks are the model's: the orange sleeves, the blue caps, the yellow shield,
  the green mark and the knob bar are what `lab_sim` gives them, nothing is added to the bench and nothing taken
  from it. The physics model is read and never written.
- **The robot**: the model's visual meshes of the arm, the hand and the fingers and the fingers' rubber pads (not
  its collision hulls), in white and black shells and steel fingers, where the simulator's state has them. They
  are MuJoCo's own renderables of a robot-only copy of the model, which a process makes once (below).
- **The room** (`render/room.npz`, `room.json` of the render bundle, 226,397 triangles, 53 colour maps): a
  laboratory bay with wall benches, shelf structures, glassware and boxes on them, the floor, the ceiling's ducts and
  luminaires, and the bench's own worktop and cabinet. Drawn only; the model's floor plane and table block are not
  drawn, the bundle's floor and worktop stand exactly where they are. Left out of the room on purpose: every prop
  that an image could take for a part of the task (tube racks, tubes) and everything that would stand on the bench.
  Sources: `../NOTICE.md`.
- **The light**: an image light (`render/ibl.ktx`: a path-traced probe of the bay over the bench with the robot and
  the bench's parts taken out; the bay's ceiling luminaires, which a probe does not see, painted in) and two spot
  lights at the bay's two luminaires for half of their direct light; ambient occlusion on. No shadow maps: the
  luminaires are 1.4 m long, their shadows in the reference are wide and faint, two spot lights' shadows would be
  two firm ones (and would cost 0.2 s more per image, below). The numbers are in `tests/lab_look.json`.
- **The bundle** is fetched at the dataset commit pinned in `environment/assets.sha256` and `tests/assets.sha256`
  into `/opt/task-assets/payload/render`: in the task image readable by root and the service's account, in the
  verifier image by root; never by the agent's account, never staged into its home.

## What did not change, and how it is kept

| | how |
|---|---|
| physics, the wire, the units, the episodes | the renderer reads `model` and `data` only; amendments 4 and 5 of the episode selection repeat the reference on the fixed set: 12 of 12 episodes at credit 1.0 with the same number of steps as before the change, both times (`validate-4d9f1355`, `validate-567b3c6c`) |
| camera poses and intrinsics | Filament is given the camera MuJoCo itself computes for the episode's model (`mjv_updateCamera`: position, axes) and the frustum of the camera's vertical field of view at the image's aspect ratio; `K_<cam>` and `T_world_<cam>` are computed as before |
| image sizes | 320 x 240, 256 x 192, 640 x 480; the three small ones are drawn at twice the size and averaged in linear light |
| sensor noise | the same line in `lab_sim.render`, from the episode's generator: the same number of draws per image, so the wrench noise of an episode is what it was |
| what the wrist camera sees of the hand | the hand's shell and the two fingers along the image's lower edge, where MuJoCo's renderer showed the hand's collision box (the camera sits inside the hand: without the shell it would see more of the bench than before) |
| the price of an image | 0.2 s of the controller's budget, as before (`grader_config.json`); measured below |
| the service and the grader draw the same image | byte copies of `lab_sim.py`, `lab_env.py`, `lab_render.py`, `lab_look.json` (a repository test); `test_draw_determinism.py` |

`test_draw_determinism.py`, on a maintainer's unit:

- A imports the grader's copy and B the service's; both run the same two episodes with the arm moving; A draws each
  camera once per step, B draws an extra image first, each camera ten times, then another order with repeats: 48
  images compared, 0 differ; 0 repeated draws differ;
- R keeps two benches open and draws the left and right camera of each in turn twelve times, so that every turn
  loads the other bench's model into the renderer again: 0 of the 48 images differ from their first;
- twelve more fresh processes, by turns from the two copies, draw the first two steps of A's episodes: every image
  is A's.

Passed in the task image and in the verifier image. The images are also the same bytes with 2, 3 and 4 rasteriser threads (48 of 48). Three things were
needed for it:

- Filament's temporal dithering (a new noise per frame) is switched off (`lab_render._undither`). The view it
  belongs to is found by a breadth-first search from the scene object: a depth-first one missed it once in about a
  hundred process starts (a renderer error at a process's first image); breadth first, 480 of 480 starts found it;
- every drawn box's faces are set in by 0.02 to 0.26 mm (by its place in the model): the bench is built from boxes
  whose faces lie in one plane (a sleeve's wall segments, the shield's plates), and which of two coplanar faces wins
  a pixel differed between processes;
- the robot is not drawn from the episode's model but from MuJoCo's renderables of a robot-only copy of the
  process's first episode's model, made once, before anything of an episode, and moved every image
  (`lab_render.robot_model`). The arm's shells pass through each other at the joints; at a sample where two of them
  compute the same depth the one drawn first wins, and for renderables made anew per episode Filament's order
  changed from one loading of a model to the next: about one loading in seven gave another picture (one sample
  before the anti-aliasing and the ambient occlusion, a few dozen pixels by one level along the robot's seams after
  them), in the second and later episodes of a process only. A test with two episodes per process passed five times
  before it failed; phase R shows it every time (6 to 7 of 48 images on the renderer before the change). The first
  form of the fix (commit `d7a0f55`: the robot's meshes built from the model's arrays) passed the same tests; this
  form draws the robot exactly as MuJoCo's renderables do (its fingers were 2 levels lighter in the first form),
  as `keyed_nut_peg_assembly` does.

Fresh processes beyond the test (maintainer runs: 288 processes that draw two benches at three states through the
four cameras, 480 that draw one state through the `rack` and `left` cameras): one picture per view in all of them
but one. That one process (of
the 768 with the final renderer; none of the 96 with the first form) drew the `rack` camera's six pictures of both
benches differently from every other process and its other eighteen pictures the same; the pictures were not kept,
480 further starts with the pictures kept did not show it again, and no part of the robot is in that camera's
view. Its cause is open: what a process draws is a function of the state in every case measured since, and an
image of the service against the grader's may differ in that camera once in several hundred process pairs.

## The look and how close it is

The reference is the path-traced look of the same laboratory that the task's demonstration video uses, with the
task's own bench in it (every geom of the model in its own colour, not the video's instrument models), rendered
through the four cameras at twelve states of one development bench (unit d0, bench 0; the robot, the lid, the rotor
and the knob at their joint values of twelve moments of a reference episode): once with all lights, once without
the bay's two luminaires, plus the light probe. Fitted in linear light: the image light against the references
without the bay's luminaires, the two lamps and the painted luminaires against the difference, then gains for the
room's surfaces that an image light over-lights (the far wall, the ceiling, the ducts) and for the robot's white
shells; ten states for the fit, two held out. The bench's own parts have no gain: they are drawn in the model's
colours. The display curve is Filament's filmic one at the references' exposure.

Median colour per class of surface, Filament against the reference's display image, both decoded to linear
(luminance ratio; held-out states in brackets):

| bench top | bench cabinet, centrifuge housing | room furniture | floor | wall | robot, white | hand | caps | boxes |
|---|---|---|---|---|---|---|---|---|
| 0.97 (1.01) | 1.01 (0.99) | 0.94 (0.93) | 1.01 (1.03) | 0.98 (0.95) | 1.06 (1.04) | 1.04 (0.92) | 1.01 (0.97) | 0.98 (1.00) |

| tubes | sleeves | rotor | shield | lid handle | scale | scale's holder | robot, black | fingers |
|---|---|---|---|---|---|---|---|---|
| 1.21 (1.16) | 0.83 (0.88) | 0.85 (0.87) | 0.88 (0.90) | 0.89 | 0.75 (0.61) | 0.67 (0.63) | 0.48 (0.33) | 0.51 |

What the image light cannot do, and so differs from the reference: nothing casts a shadow (contact shading only);
dark and metallic parts (the scale, its holder, the robot's black joints, the steel fingers) are a quarter to a half
darker, because the reference's glossy reflections of the room are stronger; the saturated parts keep the model's
colour where the reference washes them out a little (a sleeve's blue channel is near zero here); the lid is the
model's translucent blue (alpha blending, as MuJoCo's renderer drew it) where the reference has tinted glass; the
windows of the bay show a plain dark grey instead of the hall behind them.

## The facts the task states about its images, measured before and after

Same code on both renderers (the old one is the previous commit's `tests/`), images without the sensor noise unless
stated.

**What is told from what** (the twelve states; a point counts in a camera when a ray from the camera reaches it
first and it lies 2 px inside the picture; the colour of a point is the mean of the 3 x 3 pixels round it; CIELAB
distance, median over the views, smallest in brackets):

| pair | views | before | after | by camera, before -> after |
|---|---|---|---|---|
| the green mark / the rotor beside it | 10 | 29 (9) | 19 (7) | left 23 -> 15, wrist 72 -> 64 |
| a hole's entrance / the rotor between holes | 33 | 60 (8) | 46 (6) | left 66 -> 60, right 22 -> 19, wrist 43 -> 40 |
| the port's gap / the shield beside it | 42 | 56 (24) | 58 (23) | left 39 -> 41, right 59 -> 62, wrist 86 -> 79 |
| the knob bar / the hub | 10 | 56 (54) | 53 (52) | right 55 -> 52, wrist 67 -> 62 |
| a tube's cap / its body | 355 | 45 (11) | 79 (21) | left 41 -> 81, right 40 -> 66, wrist 43 -> 90, rack 47 -> 80 |
| an occupied sleeve / an empty one (2 cm over the rim) | 317 | 41 (3) | 20 (12) | left 3 -> 20, right 41 -> 64, wrist 21 -> 24, rack 53 -> 15 |

Sizes are unchanged (the mark 2.9 px across in `left`, 3.6 in `right`, 5.1 in `wrist` at the reading height; a hole
entrance 4.6 / 5.7 / 8.1 px). Nothing became unreadable: no pair has a view under a distance of 5, and the views
under 10 went from 89 to 6 (the 83 views of the `left` camera in which a tube stood against the dark behind the
bench at a distance of 3 are now at 20). Harder than before: the mark in the fixed cameras (a 3 px dot of dark
green on a lit grey rotor, averaged in linear light, where MuJoCo's renderer had a brighter green), and a tube
against what is behind it in the `rack` camera (a lit room where there was black; the caps stand out more than
before in every camera). Easier: caps against tubes, and tubes in the `left` camera.

**The reference's readers** (it does not use them on the graded path, where it asks for no image; they are the
proof that the stated facts can be read):

- the rotor's angle from one wrist image 0.20 m above the rotor (`rotor_angle.read_angle`: the hole entrances, a
  loaded hole's cap, the green mark), 24 drawn rotors each with 0, 2 and 4 tubes in them: read 0 / 14 / 24 of 24
  before and after (an empty rotor is never read: the reader wants a cap); the error's median with two tubes
  4.45 -> 2.76 degrees, with four 1.15 -> 1.06; readings more than 5 degrees off 9 -> 6;
- the whole worklist from the cameras alone (`worklist_probe.py --read camera`: the tubes and the cold rack's holes
  from wrist images, the rotor's angle from the mark and the entrances), four drawn benches: seed 0 stopped while
  loading before and now finishes (4 tubes placed, balanced, the lid shut); seed 1 finishes before and after (8
  placed; 3 -> 1 failed readings); seeds 2 and 3 stop at the rotor's turn before and after (61 -> 59 and 81 -> 81
  failed readings of an almost empty rotor).

**Cost** (the task's 4 CPUs and 12 GB; one process, as the grader runs, 45 images per camera over three episodes;
both renderers on the same day on a host with a load of about 30, twice the load of the fit's measurements, at
which the new renderer took 0.149 / 0.142 / 0.103 / 0.132 s):

| | before | after |
|---|---|---|
| seconds per image, median (largest): left / right / wrist / rack | 0.059 / 0.062 / 0.042 / 0.047 (0.08) | 0.165 / 0.149 / 0.112 / 0.140 (0.19) |
| a process's first reset (four images; lavapipe compiles the shaders) | 3.2 s | 6.4 to 7.0 s |
| a later reset | 2.6 s | 2.0 to 2.1 s |
| a control step without an image | 22 ms | 22 ms |
| peak resident memory after 1 / 4 / 16 episodes in one process | 0.76 / 0.88 / 0.99 GB | 1.03 / 1.13 / 1.28 GB |

Through the commissioning service's socket, as the agent's account (a step with one camera against a step with
none), on the same loaded host: 0.182 / 0.171 / 0.116 / 0.172 s; a connection's first reset 6 to 8 s, a later one 2 s.

The price of an image stays 0.2 s: every camera's median is under it in every measurement, so the verifier's timeout
arithmetic in `task.toml` (45000 images within the 9000 s budget) holds. How far under depends on what else the
host runs (lavapipe is bound by memory bandwidth): a third measurement that evening, with other users' jobs on the
host, gave medians of 0.195 / 0.191 / 0.143 / 0.167 s and a 90th percentile of 0.22 s (the old renderer 0.064 /
0.072 / 0.044 / 0.051 s). The price is a fixed one whatever the machine takes; at 0.25 s an image the worst case
of `task.toml`'s arithmetic grows by 2250 s and stays inside the timeout's margin. That is what decided against shadow maps: with one for each lamp an
image took 0.42 / 0.36 / 0.29 / 0.85 s, and the price would have had to rise to 0.45 s or more, which changes what a
controller can afford to look at. For the same reason the `rack` camera (four times the pixels) is drawn at its own
size, the three small cameras at twice theirs.

**Who can read what**: the agent's account could not read the simulator before and cannot read it or the render
bundle now (both Dockerfiles assert it at build time); the bundle holds the room only.
