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
- **v5 (2026-11-21, maintainer fixes) drew the hidden episodes again; the units are kept.** v5 changed frozen
  files: `tests/verify.py` and `tests/grader_config.json` (the budget is the controller's own time, with a fixed
  price per image; the numerics check before grading) and `solution/oracle/wire_world.py` (the reference waits for
  its next command without a wall-clock limit of its own). The image now pins the numerics of the simulation and of
  a controller's process (`environment/Dockerfile`, `tests/nonet_exec.py`), with a fingerprint the grader checks
  (`tests/numerics_check.py`, `tests/numerics.json`); these three files joined the frozen list. By rule step 5 of
  `review/select_set.py` draw 1 is void (the record keeps it, its candidates and the hashes it was frozen under) and
  draw 2 has a new key, written and committed before any candidate ran. The hidden units u0-u5 and the practice
  units are unchanged: their rule (`hidden/selection.json`, `hidden/recheck.json`) reads the unit's design and the
  reference's lane completion, and the world files did not change.
- **The three candidates draw 1 rejected were run again under v5** (three wire runs each, the reference without its
  wall-clock limit): none reached 1.0 in all three runs, so the rejections were the reference's own margin on
  those episodes, not the machine's load. The record of these runs is with the maintainers' review notes, not here.
