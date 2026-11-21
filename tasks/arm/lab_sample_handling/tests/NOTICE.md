# NOTICE - sources of this task's code and units (verify / maintainers only; never shown to the agent)

- **The robot** is the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image.
- **The bench** is built from MuJoCo primitives and one generated mesh (`lab_sim.py`); no third-party assets.
- **The cameras' pictures** are drawn by `lab_render.py` (MuJoCo 3.16's Filament renderer on Mesa's lavapipe) with
  the numbers of `lab_look.json`, from a render bundle that both images fetch at the dataset commit pinned in
  `environment/assets.sha256` and `tests/assets.sha256` (the benchmark's Hugging Face asset dataset,
  `tasks/arm/lab_sample_handling/payload/render/`): `room.npz` and `room.json`, a bench bay of the Unity Asset Store
  pack "Archviz Laboratory Pack" (wall benches, shelf structures, bench-top glassware and boxes, ceiling
  luminaires; the pack's tube racks left out) as decimated meshes with base-colour maps, with the bench's worktop
  and cabinet; `ibl.ktx`, the image light, from a path-traced light probe of that bay over the bench (OVRTX; sky:
  Poly Haven's `kloofendal_48d_partly_cloudy_puresky`, CC0), prefiltered with Filament's cmgen. The bundle is public
  (it holds nothing of the bench's parts, the units or the episodes) and is kept from the agent's account all the
  same: root and the commissioning service's account read it. The record: `review/rendering.md`.
- **The scored units** (tests/units.json, solution/oracle/units.json: hidden u0-u5) are the rotor's hidden turn
  properties, a Latin hypercube over `lab_sim.UNIT_RANGES`. `hidden/selection.json` holds the rule, the os.urandom
  key and the 20 recheck seeds (committed before the recheck); `hidden/recheck.json` the reference's results on them.
  Since 2026-11-20 a unit also has three fit properties (the rotor bores' clearance and friction, the lid hinge's
  friction) and a calibration error per camera (`lab_sim.FIT_PROPS`, `lab_sim.CALIB`): `review/unit_fit.py` is the
  rule, `hidden/selection_fit.json` its record (the key, the values, the reference's recheck on 20 seeds per unit).
  The development units have draws of their own by the same rule (`bench.json`'s `fit_draw`). `review/mutants.py`
  runs the reference with one thing wrong at a time on the fixed set.
- **The development units** (environment/commission_service/bench.json: d0-d2) are a separate Latin hypercube over
  the same ranges, with its own os.urandom key and rule in that file's `draw`. They are in the task image under the
  service's account only; the agent's account cannot read them. A development episode's seed is a hash of the
  unit's id and the episode's number (`commissiond.episode_seed`); the repository test checks that none of the first
  5000 per unit is a hidden episode's seed, a selection candidate's or a recheck seed. The three practice units of
  the first version (p0-p2, `lab_sim.PRACTICE_RANGES`, in the agent's home with the simulator) are gone;
  `PRACTICE_RANGES` is unused and stays only so that `lab_sim.py` keeps the bytes the set was selected on.
- **The hidden episodes** (`hidden/episodes.json`) were chosen by `review/select_set.py` under the rule in its
  docstring; `hidden/episode_selection.json` is its record (the key, the frozen files' sha256, every candidate run,
  the amendments).
- **The code**: `lab_sim.py`, `lab_env.py`, `box_transport_env.py`, `lab_render.py` and `lab_look.json` are
  byte-identical in tests/, solution/oracle/ and environment/commission_service/ (the grader, the reference and the
  commissioning service run the same simulator and draw the same pictures); `episode.py` (the episode and its credit) is byte-identical in tests/ and solution/oracle/
  and is not in the service's folder nor anywhere the agent can read. A change to one copy is a change to all, and
  to the frozen hashes of the episode selection.
