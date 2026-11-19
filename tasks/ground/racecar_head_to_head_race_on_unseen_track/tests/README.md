# racecar_head_to_head_race_on_unseen_track: notes for the maintainers

**Design rule.** The agent gets what a team at a real competition has (the league's rules, notes and photographs of
past venues, its own car's datasheet and laboratory logs, the interface and its client library) and nothing else: no
simulator, no venue, no other car, no racing stack, no practice service. The car is the owner's model as it is. The
hidden world, and with it every other car's line and pace, belongs to the grader. Solvability is shown by the
privileged reference program completing every hidden venue with full marks; difficulty is measured by one frontier
episode, never predicted.

**Where the files come from.** The maintainers assemble this folder with their own build tooling, which is not part
of the repository; do not edit the assembled files by hand. What this task shares with its sibling
`racecar_time_trial_on_unseen_track` (the car, the sensors, the ducts, the venue generator, the other cars' driver,
the referee's touch, crash, staff and lap rules, the wire, the contract's checker, the client library, the house
kits, the reference program's line planner and tracker, and of the givens the 14 lab logs with their README, all 18
of the car's mesh files and 11 of the 16 photographs) is byte-identical to the sibling's copy, one of those
photographs (`league/photos/photo_12_league_car.jpg`) under a name of this task's own. Two of the car's mesh files had
lettering removed (`car/meshes/compute_unit.obj`, `car/meshes/textures/controller_board.png`); the sibling's copies
are the same corrected bytes. This task's own copies are the three files that had to change for this day (the client
library's protocol constants and one docstring, the protocol stub's session table), the car's datasheet (one
sentence names the other cars as this task's rules do) and the five scrutineering photographs
(`league/photos/photo_13_scrutineering_rear.jpg` to `photo_17_scrutineering_front_quarter.jpg`).

**Venues of this build.** `hidden/` holds the hidden set: six venues, one for each class of track and tier of floor
grip, drawn from pre-registered seeds that nobody had seen and kept by the pre-registered selection rule, with their
reference numbers taken in the task image the way `verify.py` runs a program (`hidden/selection.json`, `"placeholder":
false`, is the whole record of the selection; `hidden/selection.prereg.json` is the part of it written before the
first candidate venue existed). The anchor was measured on them (`review/anchor_verify.json`, `anchor.json`). After
the selection the heat cell changed in three respects only, the stop of a pick-up becoming the stop of a car crash at
the car's fault judged during it, the count of standing still beginning again from zero once nothing holds it, and a
pick-up taking no warning away (until the staff set the car down, the car's crashes and its car crashes are judged and
warned as at any other time), so the selection records name the cell's files as they were at the selection, and the
reference program's records on the six venues are unchanged item by item.

**Givens of this build (published 2026-11-19).** The large givens (14 lab logs, 16 photographs and the car's 18 mesh
files: 48 files, 87.9 MB) are in the benchmark's asset dataset `grounded-intelligence-initiative/ale-robotics-assets`
under this task's prefix `tasks/ground/racecar_head_to_head_race_on_unseen_track/payload/`, put there by one dataset
commit, `d34fb25c493217239576d4af61da10ce72b0a03a`: 43 of them by a server-side copy from the sibling's prefix at
the commit the sibling's asset list pins, `d37fd7ed3b11365ed9e73b5ff7faa009be311659` (the sibling's files byte for
byte, 42 at the same path and the one photograph under its new name), and this task's own 5, the scrutineering
photographs, by an upload. `environment/assets.sha256` pins that commit and every file's sha256; the image fetches
the files while it builds and checks each one against its sha256. The dataset holds nothing of the hidden side: the
hidden set, the reference records, the anchor, the grader and the reference program are in git, under `tests/` and
`solution/`.

## Files

| path | what |
|---|---|
| `test.sh` | closes the stage folder, sets the cell's numerics (single-threaded, `NPY_DISABLE_CPU_FEATURES`; the OpenBLAS kernel never pinned), runs `verify.py` |
| `verify.py` | the grader (its docstring lists every step) |
| `grader_config.json` | the heat cell's configuration (`racecar-grader-config-h2h/1`), byte-identical to `grader/h2h/cell/grader_config.json` |
| `verify_config.json` | the run regime's knobs: the runner account, the import check, the address-space and file-size limits, the survivor wait, the two variables added to the program's environment (the sibling's, byte for byte) |
| `anchor.json` | `head_to_head_score`, value 1.0, `full_at` 0.98 |
| `grader/` | `sketch/` (the shared trees: car, sensors, ducts, other cars and their driver, venue generator, referee, wire, contract checker), `h2h/` (this task's own: `cell/` the heat cell, `field/` the world-file additions of the grid and the other cars, `scoring/` the score, `shared/` the session order), `carkit/` (the client library; also in the agent's givens) |
| `grader_manifest.json` | sha256 of every file under `grader/`, `cellkit/`, `h2hverify/`; `verify.py` checks it before it imports any of them |
| `cellkit/` | the house's process kit (lockdown, freezer, run directories, the no-network filter, the network audit): its six files are byte-identical to the files of the same name in `tasks/arm/pump_skid_startup_and_operation/tests/cellkit/` |
| `h2hverify/` | this grader's pieces: the run regime (`regime.py`), the killing seccomp filter (`netkill.py`), the static audit of the submission (`audit.py`), the three byte-identical to the sibling's; the hidden set (`hiddenset.py`), the manifest check (`manifest.py`), the call into the scoring module (`scoring_api.py`), the reported counts (`stats.py`) |
| `robotics_grader/` | the benchmark kit, byte-identical to `shared/robotics_grader/` |
| `hidden/` | `selection.json` (the record of the selection), `selection.prereg.json` (its part written before the first candidate venue), `<world_id>.json` (world files, `racecar-venue-world-h2h/1`), `reference/<world_id>.json` (reference numbers, `racecar-venue-reference-h2h/1`: `heat_laps`, `C_ref` and `T_N_s` per heat), `oracle_token.sha256` |
| `review/` | the selection's measurements kept for review, which `verify.py` does not read: `anchor_verify.json` (the anchor, the reference program's verify in the task image) and `observability.json` (what the car's sensors get to see of each hidden venue) |
| `NOTICE.md` | where the parts of this task come from and the licences of the image's distributions |

## The run regime (what an agent's program gets)

`setpriv` to `nobody` (no groups, no capabilities, no new privileges) -> `prlimit` (CPU backstop = the cell's CPU cap +
60 s, address space 12 GiB, file size 1 GiB, no core) -> the house filter `cellkit.nonet` (no socket but AF_UNIX,
EPERM) -> `h2hverify.netkill` (the same calls kill the process with SIGSYS, so an attempt is recorded and zeroes
`no_network`; and no new process: fork, vfork and a clone that is not a thread fail with EPERM) -> the carkit
launcher. The environment is the cell's `child_env` plus `program_env_extra` of `verify_config.json` (`MUJOCO_GL`,
`PYOPENGL_PLATFORM` = `osmesa`, as in the agent's session: MuJoCo's default GLFW loader would start a process). Every
stage execs: the program keeps the pid that the cell's wire server (`sketch.wire.server.serve`) started in its own
session and whose CPU time it reads. The run folder (an opaque tag) holds a root-owned copy of the submission and of
`carkit`, an empty HOME and TMPDIR owned by `nobody`. After a venue: a process of `nobody` still alive once the
program's group is gone is a survivor (`clean_run` false); then every process of the account is killed and its writes
under `/tmp`, `/dev/shm`, `/var/tmp`, `/run/lock` are removed. The stage folder, the hidden set, the grader's records
(`/opt/arb-verify`), its process memory and the image's copy of the large givens (`/opt/task-assets`, which only the
setup stage reads) are root-only.

## Gates

`submission_ok` (the static audit + one import under the regime), `step_called` (the program drove at least once),
`clean_run` (every venue has a result, no cell failure after the one re-run, no survivor, no process that could not be
killed), `no_network` (no process ended by the killing filter, no remote inet socket sampled), `no_late_write` (no
late entry in the submission, the root-owned snapshot unchanged). A program that crashes, exceeds a budget or stops
answering ends only its venue. Every venue record must pass the heat cell's own `check_result` (the balance `C`, the
cars beaten, the warnings `K` and `classified` recomputed from the record's lists); one that does not is a task error,
never a score.

## How to redo

The maintainers' build tooling for this folder is not part of the repository. The gate, as in `CONTRIBUTING.md`:

```bash
cd vendor/ale
VIRTUAL_ENV= uv run ale lint ../../tasks/ground/racecar_head_to_head_race_on_unseen_track
VIRTUAL_ENV= DOCKER_HOST=unix:///var/run/docker.sock \
    uv run ale validate ../../tasks/ground/racecar_head_to_head_race_on_unseen_track --runs-dir <runs>   # untouched 0.0, oracle 1.0
cd ../.. && uv run python scripts/lint_domain.py tasks/ground/racecar_head_to_head_race_on_unseen_track
uv run python scripts/task_assets.py check
```

Measured in the task image (one container, one venue at a time), the oracle's verify of the six venues takes 1 621 to
1 627 s with 8 CPUs on a loaded host, 794 s on an idle one and 2 118 s with 2 CPUs on an older one; the verifier's
limit in `task.toml` is derived from measured worst-case days.
