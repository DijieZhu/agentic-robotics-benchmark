# The cameras' pictures (maintainers only; never shown to the agent)

Since 2026-11-20 the five cameras are drawn by `jar_render.py` (MuJoCo 3.16's Filament renderer on Mesa's lavapipe,
on the CPU) instead of MuJoCo's classic renderer on OSMesa. Physics, the wire, the cameras' poses, intrinsics and
sizes, the pixel noise and its stream, the 0.2 s price of an image and the 9000 s budget are unchanged.

## What is drawn

- The task's own parts from the model's own numbers: each jar as the outline of its floor, wall, shoulder and neck
  turned round its axis (the model's rings are 16 boxes; the drawn jar is round, inside the boxes' outer corners),
  opaque, in the model's colour; the label as the model's solid cylinder in its kind's colour, with its two end
  faces, so a camera that looks into an open jar sees the label's colour and not the pieces under it, as with the
  model's own shapes; the live one of each jar's two lid bodies as a plain cylinder with a 1.2 mm bevel (nothing on
  it shows its angle); the tray as one pressed piece (the union of the model's floor and rims); the pieces as the
  model's boxes and capsules.
- Both robots: the model's visual meshes (no collision shapes, as before).
- A home kitchen around the counter and the counter as its island (the render bundle, `environment/assets.sha256`).
  The model's floor plane and counter block are not drawn; the bundle's floor and island top are where they are.
- Not drawn, on purpose: glass jars that show their contents, a thread on the neck, ribs or print on a lid, soft
  finger pads, mounting plates under the robots. The people's video of this task has them; the cameras do not.

## Light and look

An image light from a path-traced probe of the kitchen 0.2 m over the counter (the jars, lids, tray, pieces and
robots taken out), with the room's rectangle lights, which no probe sees, painted in so that an up-facing surface
gets the irradiance the counter's top has in path-traced pictures of the same scene. No lamp and no shadow map:
shadow passes are most of an image's time on a CPU and an image has a fixed price. Ambient occlusion on. Filmic tone
mapping, exposure 1.0. The task's own parts carry no gain: their colours are the model's. Gains only on room classes
(the kitchen units 0.6, the floor 1.22, steel 0.73) and on the robots' white shells (0.89).

Fitted against path-traced stills through the same five cameras at twelve states of a development episode (practice
unit p0, a practice seed). Class ratios Filament / reference in linear light after the fit, relative to the
counter's top: robots' white shells 1.0, floor 0.96, kitchen units 1.07, sugar pieces 1.08, sugar label 0.98; the
tray 0.62 and the saturated labels and sweets 0.7-0.96 (the tone curve and the missing clear coat; left as they are:
the model's colours).

## What stays readable (old renderer | new), development episode, 12 states x 5 cameras

Colour distance in CIELAB between the medians of two parts in a view; median (smallest) over the views showing both.

| pair | old | new |
|---|---|---|
| sugar label / jar | 21.5 (10.5) | 33.5 (28.3) |
| mint label / jar | 77.2 (34.8) | 73.7 (56.9) |
| berry label / jar | 80.8 (31.6) | 66.2 (26.7) |
| sugar / mint label | 75.1 (35.6) | 68.8 (49.8) |
| sugar / berry label | 86.6 (37.0) | 75.8 (45.2) |
| mint / berry label | 145.4 (70.8) | 124.3 (107.1) |
| lid / jar | 63.8 (37.9) | 72.1 (36.3) |
| lid / counter top | 52.8 (26.7) | 51.2 (32.5) |
| jar / counter top | 47.4 (20.8) | 39.7 (33.2) |
| lid / berry label | 68.5 (62.3) | 46.1 (14.4) |
| sugar piece / tray | 43.0 (32.0) | 62.3 (57.2) |
| mint piece / tray | 80.2 (68.6) | 78.7 (74.4) |
| berry piece / tray | 83.3 (72.3) | 57.8 (45.6) |
| sugar / mint piece | 66.0 (56.2) | 67.1 (63.3) |
| sugar / berry piece | 92.7 (84.4) | 73.3 (67.7) |
| mint / berry piece | 152.6 (142.6) | 120.5 (105.1) |
| tray / counter top | 45.1 (36.4) | 47.4 (43.4) |
| sugar piece / counter top | 22.6 (15.9) | 33.4 (28.7) |
| robots' white / counter top | 20.1 (11.0) | 33.6 (28.1) |
| sugar piece / robots' white | 0.0 (0.0) | 4.9 (4.3) |

Nothing a controller has to tell apart fell under a distance of 10; white pieces and labels against the white
robots were the same colour before and are nearly so now. Harder than before: the red lid against the purple label
in a dark wrist view (14.4), purple sweets against the tray. Easier: white pieces and the sugar label against the
counter and the jar.

## Cost (4 CPUs, 12 GB, host load 15)

| | old | new |
|---|---|---|
| s per image, overhead / front / tray / wrist | 0.07 / 0.07 / 0.03 / 0.03 | 0.12 / 0.10 / 0.06 / 0.06 (90th percentile 0.14 / 0.11 / 0.07 / 0.07; largest 0.195) |
| the same with three processes drawing at once in the sandbox | | 0.18 / 0.15 / 0.10 / 0.10 (90th percentile 0.20 / 0.18 / 0.11 / 0.11) |
| first image of a process | 0.5 | 4.3-5.1 |
| resident memory of a process after 1 / 8 / 40 episodes with images | | 1.08 / 1.34 / 1.49 GB |

The three 640 x 480 cameras are drawn at their own size, the two wrist cameras at twice theirs and averaged. In
the grader one process draws, so an image is drawn in less than its 0.2 s price and `task.toml`'s worst case holds.

## An image is a function of the state

Filament's temporal dithering is off; every renderable, mesh, texture and material is made once per process, at its
first episode, and an episode writes only its shapes and colours into them, so the order in which overlapping parts
are drawn never changes. `test_draw_determinism.py` (two copies of the simulator, other draw orders and repeats,
twelve reloads of two episodes of two different units, twelve fresh processes): 60 of 60 images equal, on a Zen 3
and on a Zen 5 host, and the two hosts' images are byte-identical to each other.
