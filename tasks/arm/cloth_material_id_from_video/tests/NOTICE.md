# NOTICE - sources of this task's assets (verify / maintainers only; never shown to the agent)

- **The robot** in the videos and in `carm.py` is the Franka Panda model of the robosuite 1.5.2 package (MIT licence,
  https://github.com/ARISE-Initiative/robosuite), installed unmodified in the task image from PyPI; only its MJCF and
  mesh files are used. The fabric clamp on its flange is this task's own geometry.
- **The scene in the recordings** is drawn from a render bundle (`environment/gen/render_bundle.sha256` lists its
  files; it is published beside the recordings in the asset repository under `render/` and is not fetched by the task
  image): a loft work space with its furniture (decimated from a purchased environment pack the maintainers hold the
  rights to; the same room other tasks of this benchmark draw), the bench's laminate, a library of twelve fabric
  looks (canvas, denim, felt, coated fabric; generated maps), the camera bodies and an image light. The fabrics,
  their materials, the clip programs and the hidden tests are this task's own and synthetic.
- **The real fabrics** are `tests/creal.py` (copy: `environment/gen/creal.py`): a finer sheet than the agent's
  `cphys` (41 x 21 vertices, 1 ms, softer contact), in air (quadratic normal drag), with speed-dependent friction
  (static above kinetic). The agent never gets this model; it plans in `cphys`, as the grader does.
- **The hidden set** (`tests/ref/fabrics.json`: ten scored fabrics A-J and the fabric P, their real materials, masses,
  four tests per skill and the reference; `tests/ref/selection.json`: every candidate examined) is drawn by
  `tests/review/make_set.py` from the key `tests/ref/key.json` by the rule of `tests/PREREG.md`, which was committed
  with the key before any candidate was simulated. `solution/ref/params.json` (the references of A-J) and
  `environment/gen/fabrics.json` (real materials and masses, for the renderer) are written by the same run.
- **The recordings** (11 fabrics x 7 clips x 3 cameras, the clamp's and the arm's motion) were rendered once by
  `environment/gen/render_clips.py` from the real fabrics (creal) with MuJoCo 3.16's Filament renderer on the CPU, in
  the task image's Python under the image's numerics pin, and are published in the benchmark's asset repository;
  `environment/assets.sha256` pins the commit and every file. `environment/gen/` is in no image layer. A fabric's
  look is a function of its id alone (tests/PREREG.md, amendment 1) and does not depend on the material.
- **Copies that must stay byte-identical:** `environment/setup/payload/{cphys,ctasks}.py` = `tests/{cphys,ctasks}.py`
  = `environment/gen/{cphys,ctasks}.py`, `environment/setup/payload/carm.py` = `environment/gen/carm.py`, and
  `tests/creal.py` = `environment/gen/creal.py` (`tests/review/check_copies.py`; the repository's test).
- **Numerics:** `tests/numerics_check.py` and `tests/numerics.json` (the fingerprint of the image's pinned
  arithmetic); the grader refuses to grade on a machine that does not reproduce it.
