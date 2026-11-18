# racecar_time_trial_on_unseen_track: notes for the maintainers

**Design rule.** The agent gets what a team at a real competition has (the league's rules, notes and photographs of
past venues, its own car's datasheet and laboratory logs, the interface and its client library) and nothing else: no
simulator, no venue, no racing stack, no practice service. The car is the owner's model as it is. The hidden world
belongs to the grader. Solvability is shown by the privileged reference program completing every hidden venue with
full marks; difficulty is measured by one frontier episode, never predicted.

**Where the files come from.** The maintainers assemble this folder from frozen sources with their own build tooling,
which is not part of the repository; do not edit the frozen files (below) by hand.

1. the hidden set is the real selection (`hidden/selection.json`, `"placeholder": false`): six venues, one per grip
   tier and layout class, drawn from seeds registered before any world was generated (`hidden/selection.prereg.json`);
   their reference numbers were taken in the task image the way `verify.py` runs a program;
2. the agent-visible documents are in `environment/setup/payload/given/`;
3. the large givens (14 lab logs, 12 photographs, the car's meshes: 44 files, 77.3 MB) are in the benchmark's asset
   dataset `grounded-intelligence-initiative/ale-robotics-assets` under
   `tasks/ground/racecar_time_trial_on_unseen_track/payload/` at commit `d37fd7ed3b11365ed9e73b5ff7faa009be311659` (the
   files first published on 2026-11-17 under `tasks/driving/racecar_time_trial_on_unseen_track/payload/`, copied
   unchanged to this folder in dataset commit `6e0561c9981213310dd05c014a38509d8a1fcc06`; then two of them were
   replaced: `car/meshes/compute_unit.obj` without a vendor mark and lettering that were modelled as geometry, and
   `car/meshes/textures/controller_board.png` with two more small inscriptions painted over; the other 42 files, the
   physics and the sensors are unchanged): `environment/assets.sha256` pins that commit and every file's sha256, and the
   image fetches them while it builds (`environment/fetch_assets.py`); a maintainer's local build may place the same
   files in the engine asset root `environment/assets/payload/` (never in git) instead. Nothing hidden is in the
   dataset: the hidden set, the reference records and the anchor are here, the reference program in `../solution/`, all
   in git.

## Files

| path | what |
|---|---|
| `test.sh` | closes the stage folder, sets the cell's numerics (single-threaded, `NPY_DISABLE_CPU_FEATURES`; the OpenBLAS kernel never pinned), runs `verify.py` |
| `verify.py` | the grader (its docstring lists every step) |
| `grader_config.json` | the cell's configuration, byte-identical to `grader/sketch/referee/grader_config.json` (key names frozen by the contract's `records.schema.json#/$defs/config`) |
| `verify_config.json` | the run regime's knobs: the runner account, the import check, the address-space and file-size limits, the survivor wait, the two variables added to the program's environment |
| `anchor.json` | `race_day_score` (the scoring module calls it `time_trial_score`; the benchmark's domain lint does not allow the token `time` in a metric name), value 1.0, `full_at` 0.98 |
| `grader/` | the vendored frozen trees (`sketch/` car, sensors, ducts, other objects, venue builder, referee, wire, contract, scoring; `carkit/`), the groups generator, contract, cell and scoring of the freeze recorded in `hidden/selection.json` (`frozen_sha256`, `frozen_trees`, `freeze_g1`, as amended by `freeze_amendments`; the maintainers' QC programs, which import the reference program, are left out), plus `sketch/examples/greybox.py`, which the cell imports but the freeze table does not list |
| `grader_manifest.json` | sha256 of every file under `grader/`, `cellkit/`, `ttverify/`; `verify.py` checks it before it imports any of them |
| `cellkit/` | the house's process kit (lockdown, freezer, run directories, the no-network filter, the network audit), byte-identical to `tasks/arm/pump_skid_startup_and_operation/tests/cellkit/` |
| `ttverify/` | this task's pieces: the run regime (`regime.py`), the killing seccomp filter (`netkill.py`), the hidden set (`hiddenset.py`), the manifest check (`manifest.py`) |
| `robotics_grader/` | the benchmark kit, byte-identical to `shared/robotics_grader/` |
| `hidden/` | `selection.json` (the rule, the candidates in drawn order, every run and rejection, the freeze, its amendments and the recorded changes), `selection.prereg.json` (the pre-registration as written, see below), `<world_id>.json` (world files), `reference/<world_id>.json` (frozen reference numbers), `oracle_token.sha256` |
| `review/observability.json` | the per-venue observability evidence made with the reference program |
| `review/anchor_verify.json` | the anchor's measurement: one full verify of the reference program on this hidden set (not read by `verify.py`) |

## The run regime (what an agent's program gets)

`setpriv` to `nobody` (no groups, no capabilities, no new privileges) -> `prlimit` (CPU backstop = the cell's CPU cap +
60 s, address space 12 GiB, file size 1 GiB, no core) -> the house filter `cellkit.nonet` (no socket but AF_UNIX,
EPERM) -> `ttverify.netkill` (the same calls kill the process with SIGSYS, so an attempt is recorded and zeroes
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

`submission_ok` (the scoring package's static audit + one import under the regime), `step_called`, `clean_run`
(every venue has a result, no cell failure after the one re-run, no survivor, no process that could not be killed),
`no_network` (no process ended by the killing filter, no remote inet socket sampled), `no_late_write` (no late
entry in the submission, the root-owned snapshot unchanged). A program that crashes, exceeds a budget or stops
answering ends only its venue.

## Recorded changes since the selection

`hidden/selection.json` records every change made after the draw, dated, with the old and new sha256 of each changed
field or file:

- `freeze_amendments` T1 (2026-11-18): the comments, docstrings and note strings of the frozen trees under `grader/`
  and `../solution/reference/` no longer cite documents outside the repository or name machines. Every changed Python
  file has the same syntax tree once its docstrings are removed; every changed JSON file is the same once its note
  strings are removed; no string that code uses changed (those that still cite a design section or hold a path of the
  machine the gates were developed on are listed in `code_strings_kept`); the reference program's records at the six
  venues and the anchor are reproduced by a fresh `ale validate`.
- `recorded_changes` 1 (2026-11-18): the folder moved from `tasks/driving/` to `tasks/ground/` (the benchmark merged
  its platforms driving and ground into ground) and the 44 asset files were copied unchanged, in one dataset commit,
  to the new folder of the dataset; `environment/assets.sha256` changed in its `# revision:` and `# prefix:` lines only.
- `recorded_changes` 2 (2026-11-18): machine paths, machine names and a LAN address in the records of the selection
  were replaced (paths relative to the folder the selection ran in, machines by their role); citations of documents
  outside the repository in `hidden/selection.json`, `anchor.json` and `review/` were reworded. `selection.prereg.json` changed
  in those machine-specific strings only, so it is otherwise byte for byte as registered; its sha256 at registration
  (in `prereg`) and now are both in the entry. The pins in `frozen_after_selection_sha256` follow the files.
- `freeze_amendments` T1b (2026-11-18) completes T1 with the same proofs: the ruling and fix ids, the bibliography and
  digest section references, the name of a file outside the repository, the contract's section numbers, a proposal's
  name and the design's gate names that T1 had left in comments, docstrings and the note strings of three parameter
  files are gone. Every changed file keeps its number of lines; the changed note strings sit in `provenance` and
  `notes` dicts that no code names (who reads each file, and which keys, is in the entry). The agent's protocol stub
  and its grader copy keep two comments that name a decision id, so the agent's view stays byte-identical to the
  version the frontier episode ran on.
- `freeze_amendments` T2 (2026-11-18), one string constant of code: `SCRATCH` in `grader/sketch/others/gates.py`, the
  default of that gate script's own `--out`, was the machine path that T1 kept; it is now the relative folder name
  `gates_out`. The syntax tree differs in that one constant, and nothing in this folder imports the script.
- `recorded_changes` 3 (2026-11-18): the note of `anchor.json` no longer refers to a name in the design; its pin in
  `frozen_after_selection_sha256` follows the file.

## How to redo

The maintainers' build tooling for this folder (the assembly from the frozen sources, the selection programs, the
harness that runs this grader in the task image with any submission) is not part of the repository. In any checkout
there is no `environment/assets/`: the image fetches the pinned files from the dataset. The gate, as in
`CONTRIBUTING.md`:

```bash
cd vendor/ale
VIRTUAL_ENV= uv run ale lint ../../tasks/ground/racecar_time_trial_on_unseen_track
VIRTUAL_ENV= DOCKER_HOST=unix:///var/run/docker.sock \
    uv run ale validate ../../tasks/ground/racecar_time_trial_on_unseen_track --runs-dir <runs>   # untouched 0.0, oracle 1.0
cd ../.. && uv run python scripts/lint_domain.py tasks/ground/racecar_time_trial_on_unseen_track
uv run python scripts/task_assets.py check
python3 tasks/ground/racecar_time_trial_on_unseen_track/environment/fetch_assets.py \
    tasks/ground/racecar_time_trial_on_unseen_track/environment/assets.sha256 <empty folder>   # the 44 pinned files, checked
```

The oracle's verify of the six venues takes about 20 minutes.
