# rov_subsea_manifold_leak_isolation · maintainers' notes (grading stage)

Behavior-stage task: an inspection-class ROV, alone on the seabed, finds and isolates the leaking branch of a subsea
production manifold with its own sensors (camera and lamps, scanning sonar, DVL, depth, attitude, a drifting
navigation estimate). The agent writes the vehicle's autonomous program (`intervene.py`, `solve(vehicle)`); the grader
runs it on six fixed hidden sites and scores the hidden world's terminal state and the cell's event log. The design of
record is the development tree's `DESIGN.md`; this folder is generated (see "Assembly" below): do not edit it here.

## Files

| Path | What |
|---|---|
| `test.sh` | stage lock, root-owned copy of the givens at `/opt/arb`, pinned numerics, `verify.py` |
| `verify.py` | the grader's parent: lockdown, gates, the station slots, aggregation, the verdict |
| `station.py` | one station's cell process (build, the program on the wire, clean-up, terminal state, scoring) |
| `slots.py` | the slot planner / scheduler and the CPU pinning with the affinity lock (standard library) |
| `layout.py` | the paths of this folder and the cell's environment |
| `grader_config.json` | run settings: stations, slots, runner accounts, caps, the budgets block, the route gate |
| `anchor.json` | the reference value (`intervention_score`, higher, `full_at` 0.95) and its provenance |
| `given_hashes.json` | sha256 of every file of `setup_payload/` (the grader checks `/opt/arb` against it) |
| `setup_payload/` | byte copy of `environment/setup/payload/` |
| `cellkit/`, `robotics_grader/` | byte copies of the shared process kit and grading kit |
| `subseacell/` | the vehicle cell (server, firmware, world, sensors, scoring); `environment/practice/cell/subseacell/` holds the same bytes less `scoring.py` |
| `kit/` | the simulation kit (byte copy) and the reference vehicle's two data files |
| `meshes/` | the vehicle's pinned meshes (`coll/` physics, `lo/` the render twin) with their sha256 lists |
| `data/task_values.json` | every number of the task the cell and the scoring read (verifier side only) |
| `reference/` | the privileged reference program (loaded only on a token-privileged run) and its perception package |
| `hidden/` | `stations.json` (six pinned site packages), `sites/`, `selection.json`, `oracle_token.sha256` |
| `NOTICE.md` | sources and attributions |

## Scoring

Per station, eleven items in integer percentage points (M1 3, M2 12, H 12, I1 8, I2 22, I3 8, D1 7, D2 5, S1 4, S2 14,
R 5); a critical error zeroes I1-I3, S1-S2 and H of its station; station score in [0, 1], no carry-over between
stations. `intervention_score` = contract gate x the mean of the six station scores (exact rational arithmetic);
`reward` = clamp(intervention_score / anchor / full_at, 0, 1). The verdict's metrics are deterministic (no wall-clock
statistic); wall clock, slot layout and timelines are in the record (`/opt/arb-verify/out/plan.json`).

## Verify by hand (inside the task image, as root)

```bash
cp -r <task>/tests /opt/ale/tests && cd /opt/ale/tests          # a writable root-owned copy of this folder
ROBOTICS_GRADER_VERDICT_PATH=/tmp/verdict.json ARB_RECORD_ROOT=/tmp/rec bash test.sh
ARB_VERIFY_MAX_SLOTS=1 ...                                        # the same stations one at a time (same verdict)
ARB_TEST_INJECT_CELL_FAULT=2@5 ...                                # self-test: station 2's first attempt faults at 5 s
```

The verdict's `late_writes_quarantined` counts the files written after the agent phase that the late-write gate moves
aside before the stations run (`submission_late_writes` counts those inside the submission, which fail the contract).
A container run by hand as above gives 0. In the engine's separate verifier sandbox it is not 0 (a few dozen: the
engine writes its own files under `/tmp` and the home folder while it prepares that sandbox, which runs no `setup.sh`);
the submission is never among them and the score does not depend on the count.

## Parallel stations (decisions M12 / N7)

Six stations, up to three at a time: each slot is one cell process (`station.py`, single-threaded, pinned to a core)
and one program process (the slot's own account `cellrun<k>`, pinned to another core, affinity locked by a seccomp
filter on top of the no-network filter). Stations start in their fixed order as slots free up; results are merged in
station order; one station whose cell failed is run again after all have ended. A station's result depends only on the
station (time passes only in `step()`), so the slot count changes nothing: `ARB_VERIFY_MAX_SLOTS=1` gives the same
verdict. Stations running at the same time cannot pass anything to each other: during verify `/tmp`, `/var/tmp`,
`/dev/shm` and `/run/lock` are mode 1770 root (a runner account can neither create, list nor read a file there), each
program's `HOME` and `TMPDIR` are its station's own mode-700 folders, removed when the station ends, and each runner
account's processes and files are swept after its station.

## Timeout

`timeout_sec` (task.toml) follows `min(10800, max(formula, formula with stand-ins for the finished cell's objects,
design value 9600))`; the header of `task.toml` gives the itemised derivation from the measured unit costs. A full
oracle verify of the hidden set takes well under half of it on three slots; one slot (`ARB_VERIFY_MAX_SLOTS=1`) takes
roughly three times as long and still fits.

## Release checks (maintainers)

Run on the assembled folder, in this order (the development tree's `records/P2_P29.md` has the commands and the last
results): the assembler's own checks (`tools/assemble_task.py --check`); `ale lint` on the folder and the benchmark's
`scripts/lint_domain.py`; `diff -r` of `robotics_grader/` against the benchmark's `shared/robotics_grader`; nothing
under the folder ignored by git (`git status --ignored`); `ale validate` (untouched 0.0, oracle 1.0); the development
tree's QC (`qc/run_qc.py`: replays and mutants, baselines at 0 on the hidden set, two-layer leak scan, payload audit
inside the image, determinism, documents against the world). `anchor.json` is re-measured whenever one of the ids it
was measured on changes (the cell, the reference program, the kit, the meshes, the task values, the stations, the
harness): the assembler refuses a stale record, and for the hidden set it accepts only an oracle run with 100 points on
every station.

## Assembly

Generated by the development tree's `tools/assemble_task.py` (packages P24, P29) with `--hidden v1`.

| Item | Value |
|---|---|
| hidden set | `v1`: `site_A_0.tar.xz` (4bf938bae754), `site_B_0.tar.xz` (bae1b1db37da), `site_C_0.tar.xz` (2a10e2808480), `site_D_2.tar.xz` (baf950c96d04), `site_E_2.tar.xz` (cabfb42d5ad3), `site_F_1.tar.xz` (14c593974b5b) |
| simulation kit | uwkit 0.3.0, tree `d7cf5cb1da0f9e37` |
| reference digest | v3 (`spec_values.json` `358676915495`, manifest `99b937ff958b`) |
| cell | tree `db47c8d1badd1e80` |
| reference program | tree `191fa779594b25e2` |
| meshes | tree `5353671ba1e4808b` (`coll/` physics, `lo/` the render twin's 161 appearance meshes) |
| task values | `adfa77b94ecdb51f` (the practice copy: less `score.*` / `selection.*` / `generator.*`, `score.imaged.*` kept) |
| practice facilities' hydro draw | `048e2b477c911cd3` = the cell's draw of each of the six stations (pinned numerics) |
| anchor | `intervention_score` 1.0, `full_at` 0.95, measured (one full oracle verify of this set; 100 points on each of the six stations) |

Amendment 05 (no local path, account or host name in the folder): the two copies of the digest's manifest (`kit/` and the practice kit) carry `"mesh_root": ""` in place of the shared file's local mesh folder, every other byte the shared file's (the cell, the reference program and the practice service always name their mesh folder explicitly, so that value is never read); two maintainer comment lines of the reference program (`reference.py` line 12, `steps_contact.py` line 1095) name no local path; `hidden/selection.json` names the maintainer machine by its role.
