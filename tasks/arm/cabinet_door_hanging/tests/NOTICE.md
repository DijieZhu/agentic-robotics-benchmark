# NOTICE - sources of this task's code and units (verify / maintainers only; never shown to the agent)

- **The robots** are the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image.
- **The cell** the simulation touches is built from MuJoCo primitives (`door_sim.py`).
- **The cameras' pictures** are drawn by `door_render.py` (MuJoCo 3.16's Filament on lavapipe) from `door_scene.py`,
  `door_parts.py`, `door_geom.py`, `door_looks.py`, `door_look.json` and the render bundle pinned in
  `environment/assets.sha256` (dataset `tasks/arm/cabinet_door_hanging/payload/render/`: `room.npz` / `room.json`,
  the living room and kitchen of the Unity Asset Store pack "ArchVizPRO Interior Vol.6", decimated; `textures.npz`,
  oak maps from Poly Haven's `oak_veneer_05`, CC0; `ibl.ktx`, a path-traced light probe of that room). The same
  scene is the one of the task's demonstration video.
- **The units**: `units.json` holds the twelve hidden units (u0-u11), each `door_sim.draw_unit(seed)` for a 64-bit seed
  keyed by the key in `hidden/units_selection.json` (os.urandom, drawn before any hidden episode was chosen);
  `environment/setup/payload/units.json` holds the two practice units (p0 = `draw_unit(1)`, p1 = `draw_unit(9)`).
  Practice and hidden units are drawn from the same ranges (`door_sim.N_KN_SET`, `HB_SET`, `HT_SET`,
  `PIN_CLASSES`, `UNIT_RANGES`).
- **The hidden episodes** (`hidden/episodes.json`) were chosen by `review/select_set.py` under the rule in its
  docstring; `hidden/episode_selection.json` is its record (the key, the frozen files' sha256, every candidate run).
- **The code**: `door_sim.py`, `door_env.py` and `episode.py` are byte-identical in tests/, environment/setup/payload/
  and solution/oracle/ (the grader, the agent and the reference run the same simulator); a change to one is a change
  to all three, and to the frozen hashes of the episode selection. `door_grade.py` is the grader's episode loop;
  `robotics_grader/` is the repository's kit, byte-identical to `shared/robotics_grader/`.
- **The numerics**: `numerics_check.py` and `numerics.json` are the fingerprint the grader checks before grading (the
  image's numerics pin, environment/Dockerfile); `numerics.json` was written by `numerics_check.py --write` in the
  task image.
