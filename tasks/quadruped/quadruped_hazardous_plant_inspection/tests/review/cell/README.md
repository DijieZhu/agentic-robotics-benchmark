# The cell's own tests (maintainers; nothing runs them in the verify stage)

Each runs from a clean process with only this task's `tests/` folder on the import path (`python -I`), against a
site package (`<name>.tar.xz`) or a site folder, with `MUJOCO_GL=osmesa` (one worker each; `LP_NUM_THREADS=1`).

| file | what it proves |
|---|---|
| `smoke.py SITE` | the cell builds from the vendored library; all 17 public calls answer; the refusals (`bad_params`, `budget`, `forbidden`, `mission_over`) change nothing; finalize / trajectory / debrief are complete and JSON-clean; timings |
| `test_no_leak.py SITE [--reference]` | every reply of a full mission (a scripted program over every call; with `--reference` also the whole privileged reference mission) against strict per-call key allow-lists, forbidden keys, the site's true values, event contents and forbidden words; the practice debrief has no total and no drawing |
| `test_link.py SITE` | the link model on hand-placed cases (site_110), its agreement with the package's coverage grid, the queue rule (queued, sent on return, lost at the end) |
| `determinism.py --site S --program P [--seed N] [--runs 2]` | the same site and program give the same score, final state hash and trajectory hash; `--compare A.json B.json` across machines |
| `run_wire.py --site S --program P [--privileged]` | one mission over the real wire the way verify runs it (no lockdown, no account switch); the reference check and the baselines use it |
| `test_practice.py --hall H --site S --work DIR` | the practice ground: run_mission.py, the debrief, one mission per connection, `busy`, the allowance and its persistence over a restart, the hall's 900 s and its true pose at 2 Hz, a mismatching package refused |
| `test_boundary_fuzz.py --site S [--calls 3000]` | the boundary: oversized and non-finite arguments answer `bad_params` with nothing changed; thousands of calls with random JSON values of every type and size in every argument of every public call: no reply but a normal one, `bad_params`, `budget`, `mission_over` or `forbidden`, no refused call changes the mission, no internal error, physics fault or lost frame; the re-run rule (an error while live re-runs, one after the end does not) |
| `test_rerun_budget.py` | at most one mission per verify is run again; a second mission in which the cell failed is a task error; an error after a mission's end is no cell failure |
| `test_wire_guard.py` | the request-line guard (`plantcell/wire_guard.py`): a line nested too deep for the JSON parser ends the mission as `protocol` (the shared wire alone raised out of its loop), odd method names and ids are answered |
| `test_run_mission_outputs.py --hall H --work DIR` | version 2: `training/run_mission.py` opens its output files (and makes their folders and `--keep`) before it reserves a mission: an unwritable path uses no mission and leaves no run folder; folders that do not exist are made and the mission runs |
| `test_practice_boundary.py --hall H --work DIR` | the practice ground against the same inputs: no request crashes the service, refused requests use no mission, the cell's boundary check inside a mission, `rule_point` in the debrief |
| `test_source_approach.py --site S` | the source's approach rule (3 s at >= 2.0 Gy/h within 3.0 m of the reported point): a waste-store dwell gives E_X 0 and the debrief names the failed half, a ring survey gives E_X 1 |
| `test_thermal.py SITE [SITE ...]` | every temperature point's unit, and every unlisted one, renders with its surface's emissivity at its temperature; a program's inversion gives the temperature back |
| `programs/glfree/patrol.py` | a test program that renders nothing (physics, gait, lidar, dosimeter, link, reports): comparable bit for bit across machines whatever their OpenGL |

`../baselines.py` builds and runs the baselines as real submissions; `../_lib/` holds the modules the baselines import
beyond the cell's and the reference's (written by the maintainers' assembly tool).
