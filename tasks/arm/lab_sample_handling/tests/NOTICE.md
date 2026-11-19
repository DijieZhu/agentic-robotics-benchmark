# NOTICE - sources of this task's code and units (verify / maintainers only; never shown to the agent)

- **The robot** is the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image.
- **The bench** is built from MuJoCo primitives and one generated mesh (`lab_sim.py`); no third-party assets.
- **The scored units** (tests/units.json, solution/oracle/units.json: hidden u0-u5) are the rotor's hidden turn
  properties, a Latin hypercube over `lab_sim.UNIT_RANGES`. `hidden/selection.json` holds the rule, the os.urandom
  key and the 20 recheck seeds (committed before the recheck); `hidden/recheck.json` the reference's results on them.
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
- **The code**: `lab_sim.py`, `lab_env.py` and `box_transport_env.py` are byte-identical in tests/,
  solution/oracle/ and environment/commission_service/ (the grader, the reference and the commissioning service run
  the same simulator); `episode.py` (the episode and its credit) is byte-identical in tests/ and solution/oracle/
  and is not in the service's folder nor anywhere the agent can read. A change to one copy is a change to all, and
  to the frozen hashes of the episode selection.
