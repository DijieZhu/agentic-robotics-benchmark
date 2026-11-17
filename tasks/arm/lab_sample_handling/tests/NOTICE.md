# NOTICE - sources of this task's code and units (verify / maintainers only; never shown to the agent)

- **The robot** is the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image.
- **The bench** is built from MuJoCo primitives and one generated mesh (`lab_sim.py`); no third-party assets.
- **The units** (tests/units.json, solution/oracle/units.json: hidden u0-u5; environment/setup/payload/units.json:
  practice p0-p2) are the rotor's hidden turn properties, a Latin hypercube over `lab_sim.UNIT_RANGES` (hidden) and
  `lab_sim.PRACTICE_RANGES` (practice). `hidden/selection.json` holds the rule, the os.urandom key and the 20 recheck
  seeds (committed before the recheck); `hidden/recheck.json` the reference's results on them.
- **The hidden episodes** (`hidden/episodes.json`) were chosen by `review/select_set.py` under the rule in its
  docstring; `hidden/episode_selection.json` is its record (the key, the frozen files' sha256, every candidate run).
- **The code**: `lab_sim.py`, `lab_env.py`, `episode.py` and `box_transport_env.py` are byte-identical in tests/,
  environment/setup/payload/ and solution/oracle/ (the grader, the agent and the reference run the same simulator);
  a change to one is a change to all three, and to the frozen hashes of the episode selection.
