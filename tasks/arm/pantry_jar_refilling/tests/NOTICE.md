# NOTICE - sources of this task's code and units (verify / maintainers only; never shown to the agent)

- **The robots** are the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image.
- **The counter** is built from MuJoCo primitives (`jar_sim.py`); no third-party assets.
- **The units**: `units.json` holds the six hidden units (u0-u5), each `jar_sim.draw_unit(seed)` for a 64-bit seed
  keyed by the key in `hidden/units_selection.json` (os.urandom, drawn before any hidden episode was chosen);
  `environment/setup/payload/units.json` holds the two practice units (p0 = `draw_unit(1006)`, p1 =
  `draw_unit(1004)`). Practice and hidden units are drawn from the same ranges (`jar_sim.UNIT_RANGES`).
- **The hidden episodes** (`hidden/episodes.json`) were chosen by `review/select_set.py` under the rule in its
  docstring; `hidden/episode_selection.json` is its record (the key, the frozen files' sha256, every candidate run).
- **The code**: `jar_sim.py`, `jar_env.py` and `episode.py` are byte-identical in tests/, environment/setup/payload/
  and solution/oracle/ (the grader, the agent and the reference run the same simulator); a change to one is a change
  to all three, and to the frozen hashes of the episode selection. `jar_grade.py` is the grader's episode loop;
  `robotics_grader/` is the repository's kit, byte-identical to `shared/robotics_grader/`.
- **Numerics**: `numerics_check.py` and `numerics.json` are the fingerprint of the simulation under the image's pinned
  OpenBLAS kernel (`OPENBLAS_CORETYPE=Haswell`, `OPENBLAS_L2_SIZE=256`, environment/Dockerfile); `verify.py` refuses
  to grade on a machine that does not reproduce it. After a change of `jar_sim.py` or `jar_env.py`:
  `/opt/venv/bin/python3 -I numerics_check.py --write numerics.json` in the task image.
