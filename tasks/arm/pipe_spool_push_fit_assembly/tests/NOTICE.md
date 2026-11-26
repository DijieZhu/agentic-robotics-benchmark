# NOTICE - sources of this task's code and units (verify / maintainers only; never shown to the agent)

- **The robots** are the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image.
- **The cell** is built from MuJoCo primitives and a generated convex mesh (`spool_sim.py`); no third-party assets.
- **The units**: `units.json` holds the twelve hidden units (u0-u11), each `spool_sim.draw_unit(seed)` for a 64-bit
  seed keyed by the key in `hidden/units_selection.json` (os.urandom, drawn before any hidden episode was chosen);
  `environment/setup/payload/units.json` holds the two practice units (p0 = `draw_unit(1)`, p1 = `draw_unit(9)`).
  Practice and hidden units are drawn from the same ranges (`spool_sim.UNIT_RANGES`).
- **The hidden episodes** (`hidden/episodes.json`) were chosen by `review/select_set.py` under the rule in its
  docstring; `hidden/episode_selection.json` is its record (the key, the frozen files' sha256, every candidate run).
- **The code**: `spool_sim.py`, `spool_env.py` and `episode.py` are byte-identical in tests/,
  environment/setup/payload/ and solution/oracle/ (the grader, the agent and the reference run the same simulator); a
  change to one is a change to all three, and to the frozen hashes of the episode selection. `spool_grade.py` is the
  grader's episode loop; `robotics_grader/` is the repository's kit, byte-identical to `shared/robotics_grader/`.
- **The numerics**: `numerics_check.py` and `numerics.json` are the fingerprint the grader checks before grading (the
  image's numerics pin, environment/Dockerfile); `numerics.json` was written by `numerics_check.py --write` in the
  task image.
