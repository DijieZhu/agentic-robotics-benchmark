# NOTICE - sources of this task's code and units (verify / maintainers only; never shown to the agent)

- **The robot** is the Franka Panda model of robosuite 1.5.2 (MIT), installed unmodified in the task image.
- **The bay, the cart, the tote and the products** are MuJoCo primitives (`shelf_sim.py`, `shelf_world.py`); no
  third-party assets.
- **The units** (tests/units.json: hidden u0-u5; environment/setup/payload/units.json: practice p0-p2) are
  `shelf_world.draw_unit(seed)`. Practice: the public seeds in that file, and per unit eight episode seeds kept by
  `solution/oracle/staging.keep`. Hidden: `hidden/selection.json` holds the os.urandom key and every candidate's
  unit seed and 20 recheck seeds, each kept by `staging.keep` (committed before any recheck ran); a
  candidate is kept when the privileged reference completes (a correct, faced lane) at least 95 % of its served lanes
  over its 20 recheck seeds; the first six kept, in candidate order, are u0-u5. `hidden/recheck.json` holds every
  candidate's results.
- **The hidden episodes** (`hidden/episodes.json`) were chosen by `review/select_set.py` under the rule in its
  docstring, from seeds whose episodes `staging.keep` keeps (the tote alone is not enough; the reference's online
  strategy restocks every lane); `hidden/episode_selection.json` is its record (the key, the frozen files' sha256, every candidate run).
- **The code**: `box_transport_env.py`, `shelf_sim.py`, `shelf_ctl.py`, `shelf_world.py`, `shelf_env.py` and
  `episode.py` are byte-identical in tests/, environment/setup/payload/ and solution/oracle/ (the grader, the agent
  and the reference run the same simulator); a change to one is a change to all three, and to the frozen hashes of
  the episode selection.
- **v4 (2026-11-20) kept the units and the hidden episodes.** v4 changed only the date code in `shelf_world.py`:
  smaller digits, and a place drawn per unit from a random stream of its own (`rng_of(seed, 4)`). Every other draw
  (unit, episode, stock, codes, tote) is bit-identical. The code's geoms are visual (no contact, 1e-6 kg each, as
  many as before). So the frozen hashes in `hidden/episode_selection.json` name the v3 files the set was chosen
  under. On the v4 files the reference was measured again by the five `ale validate` runs that `anchor.json` cites.
