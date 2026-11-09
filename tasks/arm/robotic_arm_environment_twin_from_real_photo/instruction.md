# Build the MuJoCo twin of a photographed console table

A home-robotics team will send a one-arm mobile manipulator to a client's home to clear the toy groceries off a
console table and out of its open drawer. Its plans are rehearsed first in simulation, on a digital twin of the site
as it was found. The site cannot be visited again: one colour photo of the console, taken before anything was
touched, is all there is.

Build that twin in MuJoCo from the photo. Every loose grocery toy is its own free rigid body, labelled with its
product; the drawer is one body on one slide; the console, the wall and the floor are static. When the simulation
starts, the twin must hold the photographed state: stacks stand, nothing sinks or falls. Lifting an item, or pulling
the drawer, must move the same items as it would on the real table. You are assessed against a hidden true twin of
the scene: where your items are and what shape they have, the environment's geometry seen through the photo's
camera, what stays at rest and what rests on what, and what moves when items are lifted or the drawer is pulled.

## What you have (`/home/user/`)

| path | content |
|---|---|
| `photo.png` | the photo, 1920 x 1080 colour |
| `photo_info.json` | what is known about the camera: an Intel RealSense D415 colour sensor, undistorted; intrinsics and pose not recorded |
| `site_brief.md` | the job: what the twin is for, what counts as an item and what as environment |
| `twin_contract.md` | the technical contract: files and caps, frames, the camera rules and formulas, items, the drawer, `objects.json`, the allowed MJCF, the standard simulation, the probes, and every gate, step, criterion and band of the assessment |
| `product_names.json` | the product names of the toy grocery set (the label vocabulary). It does not say which products are in the photo, or how many |
| `check_twin.py` | runs the validity gate on your submission (the same code as the grader's G0) and prints PASS / FAIL with every reason, the decoded `photo_cam` and per-item mass, density, friction and pieces; `--settle` runs the standard settle, `--probes` the probe protocol on your twin, `--overlay out.png` draws your twin's outlines through `photo_cam` on the photo; `--json`. It reads no truth and scores nothing |
| `twin_gate.py`, `twin_sim.py`, `twin_sim_config.json`, `twin_camera.py` | the gate, the standard simulation and probes, and the camera conventions, as importable modules |
| `render_twin.py`, `ovrtx_render/` | path-traced renders of your twin through `photo_cam` with NVIDIA OVRTX on the GPU, optionally blended with the photo; see `ovrtx_render/README.md` |
| `submission/` | where your twin goes (empty) |

There is no checker of the true scene, no practice scene and no way to query the truth.

## What you deliver (`/home/user/submission/`)

- `scene.xml`: the twin in MJCF, with world z up and metres; a fixed camera `photo_cam` with your estimate of the
  photo's intrinsics (resolution 1920 x 1080) and pose; every item a top-level
  body with a free joint and convex collision geometry (a mesh collides as its convex hull); the furniture static;
  the drawer one body with exactly one slide joint, limited to a plausible travel for its size (the photographed
  opening inside the range, a travel of 0.5-1
  x its depth, at least 50 mm more pull-out; otherwise C5's drawer term is 0).
- `assets/`: the meshes and textures `scene.xml` references (relative paths).
- `objects.json`: `{"format": "twin-objects/1", "items": [{"body": ..., "label": ...}], "drawer": ...}`.

Files that `scene.xml` does not reference are ignored but count toward the caps (5000 entries,
300 MB): keep renders, caches and scratch work outside `submission/`. `twin_contract.md` has every rule.

## How it is assessed

`twin_score = G1 x G0 x (weighted sum of six criteria)`, each criterion in [0, 1]. Full definitions, every band and
the probe protocol are in `twin_contract.md`, sections 8 and 9.

- **G1 submission integrity**: nothing in `submission/` written after your session ends; a real folder within the
  caps.
- **G0 validity**: `twin_gate.py` passes (what `check_twin.py` runs) and the standard settle does not blow up. An
  item that fails a plausibility rule (density 10-3000 kg/m^3, friction
  0.1-1.5, inertia, centre of mass) is excluded and scores 0, without failing the twin.
- **Alignment**: your twin is mapped into the true camera frame through your `photo_cam` (with a small correction of
  the principal ray, at most 2 deg), then scaled by one global factor s* fitted on
  matched items. Items are matched on their capped 3-D IoU (inflation never pays), labels playing no part.
- **Work areas**: the console top, the open drawer and the shelves count the same. C1, C2, C4 and C5's lift term are
  computed per area and averaged over the three; a twin of one place alone earns about a third.

| criterion | weight | what is measured |
|---|---|---|
| C1 inventory | 0.15 | matched items with the right label: 2 TP / (N_true + n_submitted), per area |
| C2 pose and shape | 0.25 | per matched item q = the product of position (full at 4 mm, zero at 15 mm), rotation (3 / 15 deg, symmetric spins free) and capped IoU (0.4 to 0.85); summed over max(N_true, n_submitted), per area |
| C3 environment | 0.1 | C1 times: the share of the photo's environment pixels whose depth through the true camera your furniture, wall and floor reproduce within 15 mm, each environment part counting the same (0.75), and your world's up direction (0.25; full at 2 deg, zero at 8 deg) |
| C4 static physics | 0.2 | after the standard settle: which items stay at rest (within 3 mm / 2 deg), and which items rest on which (support F1), each weighted by that item's q, per area |
| C5 interaction | 0.25 | lift each item 40 mm and pull the drawer 50 mm: which other items are carried, disturbed or ride along, against the true twin (Jaccard, weighted by q; the lift term per area), and the drawer's slide axis (4 / 15 deg); the drawer term is 0 when the travel rule fails |
| C6 scale | 0.05 | C1 times a band on \|s* - 1\|: full at 0.1, zero at 0.3 |

Only outcomes that are the same under every robustness variant of the true twin (masses, friction, timestep, probe
speed, small pose and support-height changes, simpler shapes) are scored. Masses beyond plausibility, friction
values, contact softness, the drawer's exact travel (beyond the travel rule), textures and visual geoms are not scored.

## Notes

- Machine: 6 h, 8 CPUs, 32 GB of memory, and one NVIDIA RTX GPU, used
  only by OVRTX renders. No network. No pretrained vision models are installed.
- Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV (`cv2`, headless), scikit-image, Pillow,
  trimesh, manifold3d, CoACD, MuJoCo 3.16 (CPU rendering through OSMesa) and usd-core; OVRTX 0.6 in its own Python,
  used through `render_twin.py`. OVRTX is NVIDIA software under NVIDIA's licence terms.
- Renders: the setup already rendered one warm-up scene so that the GPU shader cache is filled
  (`/tmp/render_twin_warmup.json` says how it went). If it did not finish, your first render compiles shaders for
  several minutes: start it in the background and keep working. Only one OVRTX render runs at a time (a second
  waits). `check_twin.py --overlay` draws outlines with MuJoCo on the CPU and needs no GPU.
- Do not modify the given files in your home directory; the grader uses its own copies.
- Harness limits: a single tool call whose input is above roughly 25-30k tokens is dropped (it never runs and no
  error comes back); a response that reaches the 64k output-token limit is lost and, after a few retries, ends the
  session; a session whose output stream stays silent for 600 s is ended. Keep each call and response well below
  these limits, write large files in parts, run long jobs in the background with their output in a log file, and
  write a first complete submission early, then refine it, running `check_twin.py` after each change.
