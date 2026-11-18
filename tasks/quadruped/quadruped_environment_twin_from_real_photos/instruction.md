# Build the MuJoCo twin of a surveyed goods delivery bay

A facility-services team will send a small four-legged robot on night rounds through the goods delivery bay of a
large building. Its rounds are rehearsed first in simulation, on a digital twin of the bay as it was found. The bay
cannot be surveyed again: 44 colour photos, taken on one walk through the hall, are all there is.

Build that twin in MuJoCo from the photos. The building is static; every loose unit (what a person or a forklift
moves as one piece) is its own free rigid body, labelled with its class; each named cage gate is one body on one
slide; every photo you used is a named camera in the twin. When the simulation starts, the twin must hold the
surveyed state. You are assessed against a hidden true twin of the bay: which units you have, where and in what
shape; the bay's geometry as the robot meets it; what stays at rest and what rests on what; the gates; the scale;
and whether the robot, walked along hidden routes with the controller you are given, gets through in your twin
exactly where it gets through in the true one.

## What you have (`/home/user/`)

| path | content |
|---|---|
| `photos/site_001.jpg` .. `site_044.jpg` | the survey photos, 6208 x 4135 colour, in the order they were taken |
| `photo_info.json` | what is known about the camera: one full-frame DSLR, a wide-angle lens at one fixed setting, on a tripod, lens distortion removed; intrinsics and poses not recorded |
| `site_brief.md` | the job: what the twin is for, what counts as a unit, what as environment, which gates slide, the surveyed area and its parts |
| `twin_contract.md` | the technical contract: every rule for the files, the photo cameras, the units and their solids, the gates and the MJCF; the standard simulation; the robot's walks; every step, criterion and band of the assessment; the tools; practical notes |
| `class_names.json` | the class names for labels, with a line each. The list names more classes than the bay contains; it does not say which are present, or how many |
| `robot/` | the robot's model, its walking controller, the route runner `walk_probe.py` and `robot_card.md` (its size, what it gets through). The assessment uses the same files |
| `check_twin.py` | runs the validity gate (the grader's G0 code) on your submission: PASS / FAIL with every reason and note, a line per photo camera, the gates, the units. `--settle`, `--probes`, `--slice`, `--walk`, `--overlay` and `--render` run the standard settle and the lift probes on your twin, map where the robot can stand, crouch or is blocked, walk the robot along your own routes, and show your twin through a photo's camera. It reads no truth and scores nothing |
| `twin_gate.py`, `twin_sim.py`, `twin_sim_config.json`, `twin_camera.py`, `twin_slice.py`, `twin_walk.py` | the gate, the standard simulation and probes, the camera conventions, the stand / crouch / blocked map and the walk conditions, as importable modules |
| `submission/` | where your twin goes (empty) |

There is no checker of the true bay, no practice bay and no way to query the truth.

## What you deliver (`/home/user/submission/`)

- `scene.xml`: the twin in MJCF, world z up, metres.
  - A fixed camera `photo_NNN` for every photo `site_NNN` you registered, with your estimate of its pose and
    intrinsics; at least 12. Your twin is placed on the site through these cameras, so they
    must be the cameras of the twin you submit, in its frame and at its scale.
  - Every unit a top-level body with one free joint and convex collision geometry (a mesh collides as its convex
    hull, so a unit that is not convex is several geoms) that follows the contract's solid convention.
  - The environment static, built from primitives and convex meshes; no height fields.
  - Each cage gate named in `site_brief.md` one body with exactly one slide joint along its rail, limited to a
    plausible travel (the surveyed position inside the range, a total travel of
    0.5-1 x the gate's width, at least
    0.4 m of further opening, at most 0.1 m of travel the other
    way beyond what the real gate has there; otherwise the gate scores 0).
- `assets/`: the meshes and textures `scene.xml` references (relative paths).
- `objects.json`: `{"format": "site-twin-objects/1", "items": [{"body": ..., "label": ...}], "gates": [{"body": ..., "joint": ...}]}`.

Files that `scene.xml` does not reference are ignored but count toward the caps (5000 entries,
300 MB): keep photos, reconstructions, renders and scratch work outside `submission/`.
`twin_contract.md` has every rule.

## How it is assessed

`twin_score = G1 x G0 x (weighted sum of seven criteria)`, each criterion in [0, 1]. Full definitions, every band
and the probe protocols are in `twin_contract.md`, sections 8-10.

- **G1 submission integrity**: nothing in `submission/` written after your session ends; a real folder within the
  caps.
- **G0 validity**: `twin_gate.py` passes (what `check_twin.py` runs), at least
  8 of your photo cameras agree with the true ones under one similarity, and
  the standard settle does not blow up. A unit that fails a plausibility rule (density
  10-3000 kg/m^3, friction 0.2-1.5,
  inertia, centre of mass) is excluded and scores 0, without failing the twin.
- **Alignment**: your twin is mapped onto the site by a similarity fitted to your photo cameras, then refined on
  your building (within 0.5 deg, 0.15 m,
  1 %). The robot keeps its true size in your twin; the fitted scale s*
  is charged in C7. Units are matched to the true ones on their surfaces, not on their labels.
- **Areas**: the parts of the surveyed area named in `site_brief.md` count the same: C1, C2 and C4 are computed per
  part and averaged. The units carry the rest: C3, C5, C6 and C7 are multiplied by C1.

| criterion | weight | what is measured |
|---|---|---|
| C1 inventory | 0.15 | matched units with the right class and a recognisable shape: 2 TP / (N_true + n_submitted), per part |
| C2 pose and shape | 0.3 | per matched unit q = pose (the largest horizontal displacement of its footprint's corners, a corner further than 2 m from the footprint's centre taken at 2 m: full at 3 cm, zero at 10 cm) x shape (surface F-score at 3 cm after registration: 0.5 to 0.9) x support (the gap to what it stands on: 2 / 6 cm); summed over max(N_true, n_submitted), per part |
| C3 site geometry | 0.15 | C1 times: where the robot can stand, can only crouch or is blocked, on 5 cm floor cells, scored per class that the true map has enough cells of (here stand and blocked) (0.5); the building's surfaces along the true cameras' rays, within 2 / 6 cm, each kind of surface counting the same (0.3); your up direction (0.2; 0.5 / 3 deg) |
| C4 static physics | 0.14 | after the standard settle: which units stay at rest (within 10 mm / 2 deg) and which rest on what (support F1), each weighted by that unit's q, per part |
| C5 interaction | 0.06 | C1 times: each named gate's place, its slide axis (3 / 12 deg) and its travel rule |
| C6 walk predictivity | 0.15 | C1 times: the robot walks hidden routes in your twin (units held, gates fixed); a route counts when it ends as in the true twin: reached where the site lets the robot through, stopped at the same place where it does not. A route that a unit decides counts only when your twin has that unit (it is paired with one of your units). Open and blocked routes are balanced: an empty hall and a solid block score 0 |
| C7 scale | 0.05 | C1 times a band on \|s* - 1\|: full at 0.03, zero at 0.12 |

Only what the photos determine is assessed, and only outcomes that hold under every robustness variant of the true
twin (masses, friction, timestep, small pose and height changes, simpler shapes, the robot's start). Not assessed:
masses beyond plausibility, friction values, a gate's exact travel (beyond the travel rule), wheels, textures, the
environment above 2.2 m.

## Notes

- Machine: 6 h, 8 CPUs, 32 GB of memory, no GPU, no network. No
  pretrained vision models are installed.
- Tools: Python 3.12 (`/opt/venv`, on PATH) with numpy, scipy, OpenCV (`cv2`, headless, with SIFT),
  scikit-image, Pillow, trimesh, manifold3d, CoACD, Open3D, pycolmap (CPU: sparse reconstruction only) and MuJoCo
  3.16.0 (CPU rendering through OSMesa: `MUJOCO_GL=osmesa`); OpenMVS 2.4.0 command-line tools on
  PATH (`/opt/openmvs/README.md`; AGPL-licensed software).
- The machine reports more processors than you have: give every tool an explicit thread count
  (8 at most), or it starts dozens of threads and runs slower.
- The photos are large: work on reduced copies where that is enough.
- Do not modify the given files in your home directory; the grader uses its own copies.
- Harness limits: a single tool call whose input is above roughly 25-30k tokens is dropped (it never runs and no
  error comes back); a response that reaches the 64k output-token limit is lost and, after a few retries, ends the
  session; a session whose output stream stays silent for 600 s is ended. Keep each call and response well below
  these limits, write large files in parts, run long jobs in the background with their output in a log file, and
  write a first complete submission early, then refine it, running `check_twin.py` after each change.
