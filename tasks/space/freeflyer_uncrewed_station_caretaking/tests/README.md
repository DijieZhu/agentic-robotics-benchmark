# freeflyer_uncrewed_station_caretaking: maintainers' notes

Platform `space`, stage `behavior`. A cabin free-flyer (a cube-shaped, fan-propelled free-flying robot with a perching
arm and a tendon gripper, five cameras, two lamps; MuJoCo 3.14.0 on the CPU, the classic renderer on OSMesa) works
alone in an uncrewed three-module station at orbital night (laboratory modules L1, L2 and hub H2). One return inlet
of the station's cabin air system is blocked by a soft item. The agent writes the robot's autonomous program
(`caretake.py`, `solve(robot)`): from the station's telemetry and its configuration record it works out which inlet is
blocked, flies there, takes the sheet off (peeling a pad-type inlet's sheet, or asking for the unit's fan to be
stopped on an element-type inlet, within the flight rules), carries it to a stowage unit's holding screen and presses
it on, puts loose pouches on the unit's pouch pads, reports the diagnosis with an evidence frame, and docks with
charge left. The grader runs the program on six hidden sites (`hidden/selection.json`).

## Files

| path | what |
|---|---|
| `test.sh` | closes the stage and its parent before Python starts; builds the root-owned givens at `/opt/arb` from `setup_payload/` and the image's task-asset pictures (`/opt/task-assets/payload`); OSMesa, 8 rasteriser threads, one BLAS / OpenMP thread and the CPU pins of the runtime key `cell.recipe.env`; names the asset root (`CABINCELL_ASSET_ROOT`, default `/opt/task-assets`); `exec python3 -s verify.py` |
| `verify.py` | the grader's driver (below) |
| `grader_config.json` | every threshold and weight of the grader (generated, never edited by hand; never read by the cell's run settings or the rehearsal facility) |
| `anchor.json` | metric `caretaking_score` (higher), `value` = the reference's score on the hidden set (1.0 by the selection rule of `review/select_set.py`), `full_at` 0.95 |
| `cabincell/` | the cell: `server.py` (CabinCellServer, the wire side of the robot's API), `api.py`, the task firmware (`firmware/`), the station services (`station_service.py`, `telemetry_synth.py`), the site builder (`site/`), the frame renderer (`sensors_render.py`: a render-only twin that shows the maker's robot meshes), the grade (`grade.py`, `predicates.py`), the vendored simulation library (`_lib/spacekit`, `_lib/station`) and the runtime copy of the task's numbers (`data/site_spec.json` with its sha256 file); `data/<component>/ASSETS.json` pin the three model archives |
| `reference/` | the privileged reference program (`reference.py`, driven by the privileged call `reference_run`) and its modules |
| `hidden/` | `selection.json` (the fixed hidden set in episode order: package, sha256, the reference's energy on the site; the frozen file list, the per-cell decisions and the recorded amendments), `candidates.json` (all 36 candidates and their verdicts), `sites/*.site.tar.xz` (the six stored site packages), `oracle_token.sha256` |
| `review/` | maintainers' tools; nothing here runs at verify: the site generator (`sitegen/`), the selection (`select_set.py`), the observability check and its stored results, the grader's mutants |
| `cellkit/`, `robotics_grader/` | the verify-stage process kit (lockdown, runner as `nobody`, no-network seccomp filter, network audit, the JSON-lines wire, the freezer) and the benchmark's verdict kit, byte copies of the benchmark's shared code |
| `setup_payload/`, `given_hashes.json` | the grader's byte copy of `environment/setup/payload/`, and the sha256 of every file the grader's copy of the givens holds (the mirror and the task-asset pictures) |
| `NOTICE.md` | sources, the fictional station, what is not modelled, the neutral names |

## Scoring

Per site, from the cell's record of the world (`cabincell/grade.py`; every threshold from `grader_config.json`):
site score = R x (D 0.14 + U1 0.10 + H 0.10 + U2 0.14 + V 0.08 + K 0.18 + L 0.18 + E 0.08). D: the first report names
the blocked inlet, the register item and the onset, with an evidence frame taken before the report; U1: the inlet
clear for a continuous stretch; H: docked at the end with charge above the line at every docking, nothing held;
U2: inlet clear and the sheet under control through the final window; V: all fans running, no flight-rule violation;
K: the sheet fixed on a holding screen; L: the pouches on pouch pads (own footprint fraction); E: the energy against the
reference's on the same site, the fan-stop seconds against fixed limits. R = 0.85 after hard collisions or keep-out time, else 1. caretaking_score =
contract_ok x the mean over the six sites; reward = clamp(score / anchor / 0.95, 0, 1).

## The verify stage

Separate verification (`task.toml`): a fresh verifier sandbox from the task image, with only `/home/user/submission`
restored. `verify.py`: the early lockdown; the task's inputs, each a task error when broken (the runtime numbers and
the grader configuration, the numeric pins, `/opt/arb` against `given_hashes.json`, the hidden set, the three asset
components in the asset root checked file by file against their pins, the no-network self-test); the form gates
(`caretake.py` parses and defines `solve`, the route gate, the static and runtime network audit, the import check as
`nobody`); the oracle token; then per site, in the set's order, a fresh cell with the frame renderer, the program as
`nobody` from a read-only copy under an opaque run tag, on the cellkit wire with the runtime spec's budgets; a site
whose cell failed runs once more (at most one rerun per verify); a program that advanced no simulated time on a site
fails the contract. Records: `/opt/arb-verify` (root only).

## The image (environment/)

`Dockerfile`: the ALE Ubuntu 22.04 base 0.1.1, OSMesa, a Python 3.12 venv at `/opt/venv` with the pins of
`requirements.txt` (mujoco 3.14.0, numpy 2.5.4, scipy 1.18.1, pillow 13.0.0, opencv-python-headless 5.0.0.93); the
task assets of `assets.sha256` (the cell's three model archives, unpacked once to `/opt/task-assets/{robot, station,
robot_author}`, and the 40 pictures of the givens at `/opt/task-assets/payload/given/`), root-owned with group
`rehearsal`, modes 750 / 640 (the agent account can neither list nor read them: proofs in the build); the rehearsal
facility (`practice_ground/` at `/opt/rehearsal`, account `rehearsal`, mode 700). Build argument `CARETAKE_ASSETS`:
`hub` (the asset dataset at the pinned revision), `local` (the build context's `local_assets/`), `auto` (local when
every listed file is in the build context, else hub). `assets.sha256` carries the revision of the asset dataset the
files were uploaded at. `setup/setup.sh`: stages the givens and the
task-asset pictures into the agent home, checks the asset root and starts the rehearsal facility (socket
`/run/rehearsal/rehearsal.sock`, status `/run/rehearsal/status.json`, debriefs `/run/rehearsal/debriefs/`) under a
restart loop.

## Time limits

The agent has 14 400 s, no network. The verifier's limit (task.toml) follows the time-limit formula: per site the build, the
simulated-time budget at the measured wall seconds per simulated second, the colour and depth frame budgets at the
measured p95 frame times, the program-time budget, the settle and the clean-up; seven runs (six sites and the one
rerun) plus the fixed steps, and at least 15 % margin, from the worst values measured in a container of the task
image on the measuring host (the comment of `task.toml` gives the numbers).

## Maintenance

`environment/assets.sha256`, `setup_payload/` and `given_hashes.json` are generated, never edited by hand.
