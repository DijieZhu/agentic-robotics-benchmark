# racecar_full_assembly_tenth_scale_ur10e: grading notes and evidence for maintainers

The grader (`verify.py`, started by `test.sh`) runs as root in a separate verifier sandbox from the task image; the
agent never sees this folder. This note says what the grader does, how the fixed hidden set was chosen and how its
freeze works, what the zero-cost evidence of development shows and what is left for the owner. The maintainers' build
tooling for this task (the generators of the givens and the drawings, the integration and quality-control harnesses,
the probes and their evidence files) is not part of the repository: this note and `hidden/selection.json` carry what a
maintainer needs, and "How to redo" lists the checks that run from the repository. No value of the hidden set (seeds,
layout digests, the bench plan, the oracle token) appears outside `hidden/`.

## What is here

| path | what |
|---|---|
| `test.sh` | closes the stage folder, builds the grader's root-owned copy of the givens (`/opt/arb`: `setup_payload/` + the image's task assets), starts `verify.py` |
| `verify.py` | the grader: contract and route gates, the hidden kit (`gt/`), four episodes of `hidden/episodes.json` through the cell server and the runner, the score formula, the metrics |
| `grader_config.json` | every knob: budgets, cameras and their draws, layout jitter, safety thresholds, scoring switches, the hidden kit's pin (`hidden_gt`) |
| `anchor.json` | metric `assembly_score`, value 1.0, `full_at` 1.0 (the oracle's own score by construction) |
| `arm/assembly_car/` | this task's cell: `coupling.py` (per-slot capture, pull home, lock, scrap, carrier voiding, the wheel stops), `cell.py`, `layout.py`, `bench_plan.py`, `fixtures.py`, `server.py` (public RPC, the token-only `reference_assemble`), `reference.py` + `reference_shop.py` + `car_grasps.py` (the privileged reference program), `render_twin.py`, `agent_render.py`, `agent_look.json` |
| `arm/{assembly_cell, assembly_info, assembly_six, assembly_quad}/`, `robotics_grader/`, `common/`, `render/`, `render_filament/`, `room/`, `episode_trace.py`, `netguard.py` | vendored byte for byte from the benchmark's published assembly cells and kit (`netguard.py`: seccomp, no IP sockets for the agent's program) |
| `hidden/` | `episodes.json` (the four graded episodes), `selection.json` (every candidate, run and decision of the selection, the dated amendments A1 to A4 of its freeze rule and the recorded changes), `bench_plan.json`, `oracle_token.sha256` |
| `review/select_set.py` | the pre-registered selection of the hidden set and its freeze check (a maintainer's tool; the grader never runs it) |
| `review/observability.md` | the camera-only probe's summary, both rounds side by side (statistics only) |
| `setup_payload/` | byte copy of `environment/setup/payload/` (the small givens: `kit.json`, the cell documents, the client library) |
| `given_hashes.json` | sha256 of every file the grader's `/opt/arb` must hold |
| `hidden_gt.url` of `grader_config.json` | where the hidden kit is: `verifier/gt.tar.xz` of the task's folder on the benchmark's asset dataset at commit `6cfa80d098df6a6def71f8304edbcd4245fa829f` (the commit `environment/assets.sha256` pins for the givens); `verify.py` downloads it at verify start and checks the pinned sha256 and file count; it is never in the repository or an image. A maintainer's local copy (`ARB_HIDDEN_GT_TARBALL`, or `assets/gt.tar.xz` here, an engine asset root that no checkout has) is used instead under the same checks |

## The score

```
counted(s)     = s LOCKED and design_ok at finalize, and neither its unit nor a unit in its carrier chain scrapped
stage_e        = sum of counted weights / 11.5          (battery 1.0, deck 2.0, compute 1.0, LiDAR 1.5, four wheels 1.5 each)
hard_ok_e      = 0 on a robot link above L (pads excluded) or a unit pressed above 150 N for 0.2 s, else 1
L              = grader_config.json safety.robot_force_N = 150 N                     (v1.8; 30 N in v1.0-v1.7)
f              = grader_config.json scoring.safety_violation_factor = 0.3          (v1.7; 0 in v1.0-v1.6)
value_e        = stage_e x (1 if hard_ok_e else f)       (one factor per episode, however many violations)
contract_ok    = assemble.py valid x step calls > 0 x clean run x no network x no late write / surviving process
assembly_score = contract_ok x mean over the 4 episodes of value_e                 = reward (anchor 1.0)
```

**The safety rule (v1.7, the owner's decision of 2026-11-16).** A safety violation (either hard rule above) does not
end the episode, but an episode with one keeps only 30 % of its stage score; before v1.7 it scored 0. The factor is
one number in `grader_config.json` (`scoring.safety_violation_factor`): `verify.py` scores with it (`safety_factor`,
`episode_values`; a value outside [0, 1] is a task error) and records it as the metric `safety_violation_factor`, the
cell's `describe()` prints it, and the instruction, the cell manual, the cell API and `cell_config.json` take their
figure from it through their generator. With the factor set to 0 the grader reproduces the verdicts of v1.6 byte for
byte (the reward envelope, every metric, the diagnostic; checked on the recorded verdicts of the reference program's
validates and of the calibration episode). The metrics keep the
ungated mean (`stage_score_mean_ungated`), the hard-rule counts (`hard_violation_rate`, `robot_contact_episodes`,
`part_force_episodes`) and the largest sustained force; `out/episodes.json` records each episode's `episode_value`.

**The robot-contact limit (v1.8, the owner's decision of 2026-11-17).** A robot link other than the gripper's four pad
boxes pressing on one body with more than 150 N (the normal forces of that link on that body summed over one control
tick) is a safety violation; in v1.0-v1.7 the limit was 30 N, a number inherited from the first assembly task and never
derived for this car. 150 N is the force at which an arm of this class stops itself by default, so the rule now marks
a contact at which the real robot would have stopped. The limit is one number in `grader_config.json`
(`safety.robot_force_N`): `verify.py` checks it (`contact_limit`; anything but a finite number > 0 is a task error)
and records it as the metric `robot_contact_limit_N`, the cell's watch takes it through the server
(`CarInfoServer.cell_config`), `describe()` prints it, and the instruction, the cell manual and `cell_config.json`
take their figure from it through their generator; the cell's own default (`arm/assembly_car/cell.py`, used only by a
cell built without the grader's configuration) is the same number, and the server's default takes the cell's. The
unit-pressed rule (150 N for 0.2 s), the factor, the gates and everything else are unchanged. With the limit set to
30 N the cell and the grader give the verdicts of v1.7: on every recorded episode summary (the watch keeps each
episode's largest robot-link force: 33 records and the oracle of 14 validates) and in one run of a violating QC case (M9 at 30 N, the same per-episode results as its v1.7 run). While a
link's force on a body stays above the limit, the watch (`CarCell.watch_tick`) logs a `safety:` event for that link
and body at most once every 0.5 s, each with the force of its own tick (its de-duplication looks only at the episode's
last five violations); the event list is therefore not the largest force of the episode, which is in
`out/episodes.json` (`watch.max_robot_link_force_N`).

A unit faster than 3 m/s or below the bench is scrapped (the episode goes on); a scrapped deck voids the compute
module and the LiDAR fastened on it. Compute module and LiDAR fastened on the deck while it is still on its shelf count
(`scoring.count_subassembly = true`; the cap of that route alone is 2.5 / 11.5 = 0.217); the lead designer kept this
as designed. Everything else (ungated mean, per-slot mate rates, LOCKED_OFF, scrapped and voided counts, frames, sim
and wall time, forces, illegal calls) goes to `metrics` only, all finite numbers.

## The fixed hidden set, its freeze and determinism

- **Rule** (`review/select_set.py`, module docstring; copied into `hidden/selection.json` before any candidate ran):
  freeze, draw 12 candidates from `os.urandom` in four strata (the stand's yaw sign x the stick pack's lead end),
  run each once in three environments (the development venv, the task image through `test.sh` / `verify.py` with the
  oracle submission, a second machine with another CPU), keep a candidate only when every run has 8 / 8 mates LOCKED,
  0 unplanned contacts, hard_ok, sustained force below 150 N, every capture within 50 % of its window and every loose
  unit at least 30 % visible in the top camera; per stratum the first passing candidate in drawn order.
- **Result** (drawn 2026-11-15T04:57:01Z): 11 of 12 candidates passed in all three environments; the one dropped
  failed only the capture clause (the stick pack captured at 0.77 of its window) and only on the second CPU. The set
  is c00-c03, the first candidate of each stratum. Hollowing check: 1 / 12 dropped, one cause, no mate class whose
  kept capture ratios are better than the dropped ones by more than 0.10 of the window.
- **Determinism.** Same CPU (development venv and task image): outcome fingerprints and episode traces identical for
  12 / 12 candidates. Another CPU: not bit-identical (the first difference 9-16 s into an episode), outcomes identical
  for 11 / 12. Acceptance verifies of the set in the image: 1.0 / 1.0, the same fingerprints and traces. The layout
  digest of every episode is recomputed at every reset and compared with `episodes.json`; a mismatch is a task error.
- **What invalidates the set: the frozen table as amended (A1, 2026-11-15; A2, 2026-11-16; A3, 2026-11-17; A4, 2026-11-17).** The rule
  as drawn froze two files whose bytes carry more than an episode's outcome (`grader_config.json` also holds the
  hidden kit's download location, then still to be filled in at publication; the kit was frozen by its archive's
  bytes). Amendment A1, written into `selection.json` (`amendments`) at 2026-11-15T14:50:36Z, before any file changed
  and after checking every frozen file and tree against the draw, keeps the set and redefines the table: frozen is
  what determines an episode's outcome. `verify.py`, `test.sh`, `netguard.py`, `episode_trace.py`,
  `hidden/bench_plan.json`, `hidden/oracle_token.sha256`, the three files of `solution/` and the trees `arm/`,
  `common/`, `render/`, `render_filament/`, `room/` and the given robot models stay frozen by their bytes;
  `grader_config.json` is frozen by its values (`#values`: its canonical JSON without `hidden_gt.url`,
  `hidden_gt.sha256` and `hidden_gt.note`); the hidden kit by its content (`gt.tar.xz#members`: the names and sha256
  of its 14 620 file members). Recorded by hash, never invalidating: that publication metadata (the archive's sha256,
  the url, its hash, the note) and the files the agent reads (`instruction.md`, the setup payload, the asset list, the
  given parts, drawings, photo manual and render assets, `setup_payload/`, `given_hashes.json`). `select_set.py check`
  applies the table in force (a changed recorded entry is a note); `select_set.py record` enters recorded changes with
  their date and reason (`recorded_changes`: the corrected cell manual, 2026-11-15T16:13:21Z; an independent review's
  text fixes of the instruction, the packing list and `assembly_layout.png`, 2026-11-15T20:24:13Z; the order
  statement's half-sentence on the fixtures beside the car, 2026-11-15T22:26:03Z; the publication,
  2026-11-15T22:33:35Z: the asset list's revision and `hidden_gt.url` at the dataset commit, `hidden_gt.note`; the
  v1.7 texts of the safety rule, 2026-11-17T00:33Z; the v1.8 texts of the robot-contact limit, 2026-11-17T18:22:55Z;
  the move of the task's folder to `tasks/ground/`, 2026-11-17T22:19:43Z: the asset list's revision and prefix and
  `hidden_gt.url` at the dataset commit of the copy; the record's own text and the selection tool, 2026-11-17T23:18:19Z:
  the 71 run paths of the record made relative to the folder the development runs were kept under (`path_rule`), its
  citations of documents outside the repository reworded in place (the rule text as drawn in three places, the
  expectation, the amendments' authorities, the reasons of recorded changes 1-6; each field with the sha256 of its old
  and new value under `edited`), and `select_set.py` recording run paths relative to a run root (`--root`); no frozen
  entry, no recorded entry and no value the selection or the check uses changed).
  No reselection and no redraw were made. Amendment A2 (written at 2026-11-17T00:32:59Z, after checking the A1 table
  on the tree just before the change) lets a dated decision of the owner that changes only how an episode's outcome is
  scored re-enter the frozen table instead of a redraw: the v1.7 safety rule changed `verify.py`,
  `grader_config.json#values` and, inside `arm/`, only `assembly_car/server.py` (the safety strings of `describe()`);
  every other entry, the kit's members and the cell's physics are as A1 recorded them, and the layout digests and the
  reference program's per-episode outcomes were checked again in the validates of v1.7. Amendment A3 (written at
  2026-11-17T18:22:36Z, after checking the A2 table on the tree just before the change) does the same for a dated
  decision of the owner that changes only a threshold of the cell's watch: the v1.8 robot-contact limit changed
  `grader_config.json#values` (`safety.robot_force_N` and its note), `verify.py` (`contact_limit`) and, inside `arm/`,
  only the lines of `assembly_car/cell.py` and `assembly_car/server.py` that carry the number and comments; a
  threshold enters no physics or layout, the selection's keep rule required no robot link above 30 N, which is
  stricter than 150 N, so every selected episode still qualifies and nothing is reselected, and the reference
  program's robot links press on nothing in any recorded run (A3's rationale says so since its dated correction of
  2026-11-17T21:21:16Z, in `corrections`). Amendment A4 (the lead designer's decision of 2026-11-17, written at
  2026-11-18T00:23:26Z by the run that made the change, after checking the A3 table on the unchanged tree) lets the
  comments, docstrings and note strings of `verify.py`, `grader_config.json` and the files of `arm/assembly_car/` that
  cited documents or files outside the repository be reworded without a redraw, under conditions checked on a copy of
  the task before the change was accepted: (a) every changed Python file has the same abstract syntax tree once the
  docstrings of its module, classes and functions are removed (the docstrings that changed are listed in the
  amendment); (b) `grader_config.json` is equal once every key named `note` or ending in `_note` is deleted, and only
  such notes changed (five; `hidden_gt.note` is not among them); (c) the hidden set is unchanged: the four layout
  digests recomputed from their seeds equal `episodes.json`, which has the set's sha256, and the kit has its pinned
  sha256 and its 14 620 members; (d) the validate of the tree that carries A4 reproduces the per-episode grader lines
  and the reference program's per-mate lines of the validate before it. One string of code changed with them, the
  provenance label `FROZEN_SOURCE["file"]` of `arm/assembly_car/car_grasps.py`, which no code reads (its uses are
  listed in the amendment); the grasp labels by which the reference program tells its grasp tiers apart are unchanged.
  A4 re-anchors the table on the new bytes of those three entries (12 files); every other entry is as A3 recorded it,
  and nothing the agent sees changed. Any other change of a frozen entry still means `select_set.py draw
  --supersedes`, then the full QC cases, the camera probe and the noise ladder again. Filling in the real revision of
  `hidden_gt.url` at publication did not.

## The cell manual's collision model

The cell simulates collision shapes that differ from the part files and the photographs; the manual
(`setup_payload/given/cell/cell_manual.md`, "Collision model") lists every difference with its size. The chassis is
its drawn shapes cut into convex pieces (within 0.16 mm of the drawn surface, material only taken away, the cuts
listed). The loose units, the hubs, the battery bay and its hold-down posts are a few convex pieces built from the
drawn sizes; the manual states per unit how far they stand beyond the drawn outline and where they are solid in
hollows of the drawn part, with the side it concerns: a wheel within 0.22 mm of its outline; the antenna mount a box up
to 9.1 mm beyond the drawn mount; the LiDAR's cap a box whose corners stand 6.7 mm beyond the round cap; the compute
module's boxes up to 9.7 mm from the drawn material in its recesses and 5.0 mm beyond its outline at some corners;
the stick pack's prisms 0.23 mm beyond the sleeve along its sides and up to 2.5 mm at its ends; the battery bay's
faces that the stick pack meets within 1.8 mm of the drawn bay (its larger differences, up to 10.1 mm, face away from
the pack). An earlier text called the shapes "the drawn shapes with material taken away and none added", true for the
chassis only; it was corrected (the kit unchanged). Every number of the manual comes from a named source and has an
independent reading (499 numbers, 0 mismatches; the built pieces measured face by face).

## Integration gates (finalization, 2026-11-15; rerun after the independent review's text fixes, for v1.7, 2026-11-16, for v1.8, 2026-11-17, after recorded change 9, 2026-11-17, and after amendment A4, 2026-11-18)

The repository's gates (`ale lint`, `scripts/lint_domain.py`, `scripts/task_assets.py check`, the task card check, the
repository tests) and `ale validate` of this tree (at the finalization two runs, compared with each other; after every
later change one more, compared with the previous tree's: rewards, every episode's grader line, the reference
program's per-mate lines, every metric but wall-clock values and three counters of the verifier sandbox's own desktop
session), plus the checks below; the rows marked (build tooling) ran with the maintainers' build tooling, the others
run from the repository ("How to redo").

| check | result |
|---|---|
| hidden set's frozen table (amendment A4 since 2026-11-18; A3 since v1.8; A2 since v1.7; A1 before) | unchanged; `episodes.json` is the selection's set |
| `review/select_set.py selftest` | 10 / 10 (among them: the publication metadata and the documents may change, a grader value or a kit member may not, a table written by the rule as drawn is checked by that rule) |
| givens rebuilt by their generator (build tooling) | 127 files and 3 evidence files, 0 differences; manual audit 499 numbers, all with an independent reading, 0 mismatches; generator selftest 32 / 32 |
| the givens' text sources' own checks (build tooling, 38 tests) | ok |
| the five drawings rebuilt by their generator (build tooling, 26 tests) | identical; audit: 87 printed dimensions, 0 failures (`assembly_layout.png` gained note 5, its values unchanged) |
| leak list and hidden values over the agent-visible files of the task folder (build tooling) | 251 files, 0 unwaived hits (the waived ones: the word "anchor" in the reused, hash-pinned gripper model and hall bundle) |
| `review/observability.md` against the hidden values | 0 hits |

## Quality control (the full-run cases; build tooling)

Every case runs the grader's own `verify.main()` on the four hidden episodes; expectations are the score formula's
values, written before each run and never edited after it. 30 rows and one diagnostic, 0 mismatches. For v1.7 the two
rows that expected a zeroed episode (M9, M10) were re-derived by the new rule (their v1.6 expectation stays in their
case file) and run again, with the new rule's own rows S1 and S4 and the degenerate row M1 = C4, on the development
host; the contract cases and the M12 cross-check ran again in the task image (their givens changed). For v1.8 (the
150 N robot-contact limit) the rows whose forearm press depends on the limit were rebuilt and run again: M9, S1 and S4
press until the watch records a violation (they stop at the limit + 10 N or at the first violation; under 30 N that is
v1.7's press, and M9 run so with the limit set to 30 N gave its v1.7 results episode for episode), with their v1.7
expectations; the new row S5 presses between 30 and 150 N; the contract cases and M12 ran again in the task image.
M10's press involves no robot link above 15 N, and the degenerate rows fasten no mate, so the limit leaves them as
they were. The other rows had no safety violation in any episode, so neither rule changes their scores:

| case | what | expected | measured per episode |
|---|---|---|---|
| oracle | the token submission | 1.0 each, twice the same | 1.0 x 4 in each validate run |
| M1 = C4 | the nominal layout's joint trajectory replayed open loop | <= 0.02 | 0 / 0 / 0 / 0 (v1.7 re-run: exactly 0.0, violations in three episodes, no mate) |
| M4 | deck first (the stick pack cannot go in) | 0.913 | 0.913 x 4 |
| M5 | deck thrown off the bench after the compute module and LiDAR are fastened on it | 0.609, episode not zeroed | 0.609 x 4 |
| M6 | the tray's storage turns copied, the lead end ignored | 0.783 / 0.696 by stratum, mean 0.739 | 0.783 / 0.696 / 0.783 / 0.696 |
| M7 + M13 | wheels pushed to their stops without handling the hub turn; held at 20 N | per wheel: locked iff the hub is within 6 deg | 16 / 16 wheels as predicted (3 locked), mean 0.576; M13 10 / 10 not fastened |
| M9, M10 | forearm on the car above the robot-contact limit after the full build; a unit pressed into the bench above 150 N, then the full build | 0.3 each (1.0 kept at 30 %; v1.6: 0) | 0.3 x 4 each (M9, v1.8: robot_contact at 161 / 165 / 174 / 162 N; M10: part_force; 8 / 8 mates) |
| S1 (v1.7) | the full build; the forearm pressed on the car above the limit once, in the first episode only | 0.3 / 1.0 / 1.0 / 1.0, reward 0.825 | 0.3 / 1.0 / 1.0 / 1.0 (v1.8: 161 N), reward 0.825 |
| S4 (v1.7) | in every episode a wheel pressed into the bench, the full build, then the forearm on the car above the limit | 0.3 each: one factor however many violations | 0.3 x 4 (3 violations of both kinds per episode), reward 0.3 |
| S5 (v1.8) | the full build, then the forearm pressed slowly on the car until a link presses it with 40 N (above the earlier 30 N, below 150 N), in every episode | 1.0 each: no violation | 1.0 x 4 (40.1-40.4 N, no event), reward 1.0 |
| M12 = C8 | compute module and LiDAR onto the deck on its shelf only | 0.217 | 0.217 x 4 (also in the task image) |
| B1-B5 | wheels first; LiDAR before compute; wheels swapped; the sub-assembly route; the stick pack held level, let go at 30 mm | 1.0 | 1.0 x 4 each |
| C1, C2, C3 | no `assemble.py`; empty `solve`; steps only | 0 | contract_ok 0 / contract_ok 0 / 0.0 with contract_ok 1 |
| C5, C6, C7 | the reference on the nominal layout (no cameras); with the nominal intrinsics; every unit dropped 30 mm above its nominal home | <= 0.02 | 0 x 4 each |
| C9a-f | undocumented RPCs; reading `/opt`; a weight file; a subprocess; a socket from a `.py`; sockets from a `run.sh` program | forbidden / contract_ok 0 | 48 / 48 calls forbidden; every read refused; contract_ok 0 x 3; EACCES on every IP socket |

A scrapped unit was never a safety violation (8 episodes with scrapped units). The physics cases ran in-process with two
documented harness patches (the runner's account, the shared machine's network counters); the contract cases and an
M12 cross-check ran in the task image exactly as the verifier does (root, `nobody`, netguard, Filament). None of the
physics cases depends on the texts. The image cases depend on the givens (the grader checks `/opt/arb` against
`given_hashes.json`): after every change of the givens (the corrected manual; the review's text fixes; the v1.7 texts
of the safety rule; the v1.8 texts of the robot-contact limit) they run again in the task image that the validate of
that tree built (for v1.7 and for v1.8 too: C1-C3, C9a-f and M12 in the image read as before).

## Zero-cost difficulty evidence

- **I0 clearance audit.** On the raw meshes 78 / 78 insertion orders pass (clearance including the capture window's
  rotation envelope >= window + 0.5 mm). On the simulated kit 76 / 78: the stick pack's two orders are 1.40 mm short
  at the rear end of the chassis brace. The design's pre-written fallback applies (that direction uses the measured
  entry window, printed in the cell manual, which also says where the brace's cut ends); the lead designer accepted
  it, the dynamic gates and the 100-layout campaign of the reference program having passed. The brace was not
  re-sized.
- **I1 entry windows** (1731 trials in the real cell, nominal layout): deck, compute module and LiDAR lock 71 / 71
  inside 80 % of their windows and 32 / 32 at the window corners; wheels 244 / 244 and 118 / 120; stick pack 62 / 66
  and 10 / 19. Wheel stop 2.44 mm short of home; the hub turns with the wheel from 6 N of push; the documented search
  (<= 5 N, 10 deg/s) 24 / 24; blind insertion 0.25 on uniform hub angles (3 / 16 on this set). No window changed.
- **I2-lite noise ladder** (180 runs: pose error sigma 0-3 mm x hub angle known / searched / blind x 4 episodes x 3
  noise seeds): sigma 0 with known angles equals the oracle bit for bit. Lock rates hold to sigma 0.5 mm and fall
  between 1 and 2 mm (logistic s50: stick pack 1.20, deck 1.29, compute 1.98, LiDAR 1.75, wheels known 1.58 /
  searched 0.91 mm; blind 0.19). The reference program's hard-violation rate rises 0 / 0.14 / 0.44 / 0.69 / 0.72.
- **I3 camera-only probe** (`review/observability.md`). What it knows: every per-episode quantity (the two cameras'
  calibration, the fixtures, the car, every unit's pose, the hub angles) comes from the cameras over the public RPC
  and the robot's own state, with no privileged call. Its static design data are not the given files but the hidden
  kit's (the places' design frames, the bodies' design poses, the planner's collision pieces, the grasp table), and
  its moves are the reference program's motion layer (grasps, inverse kinematics, planner) driven by its estimates.
  It assumes that an agent derives equivalent static data from the drawings and the part files and does not measure
  that derivation: of the gate's purpose (doable from public information only) it shows the perception half.
  Round 1 was developed on hidden episodes 0 and 1, so it is not blind there either. Round 1: 1.000 / 1.000 /
  0.391 / 0.000, mean 0.598, the design gate (mean >= 0.6 and every mate class fastened at least once) missed by
  0.002; both losses were defects of its single overhead frame. Round 2 repairs exactly those two (reject a fixture
  board whose residual stands out or whose place lies outside the documented bounds and look at it with the wrist
  camera before moving near it; survey the bench with the wrist camera when the overhead height map does not give the
  bench units), developed on layouts outside the hidden set and frozen before its first hidden episode; it is not a
  blind run. Round 2: 1.000 / 1.000 / 0.870 / 0.000, mean 0.717, every class fastened, every budget kept, public RPC
  only: **the gate as registered is passed**, for the perception half as said above. Its two remaining losses are
  again the probe's: a wheel the survey saw only in part (episode 2), and a wrist calibration that diverged after a
  board inside its bound was rejected and was accepted unchecked, after which a pick pressed the LiDAR into its nest
  above 150 N (episode 3). No document gap and no binding window in either round.
- **Independent partial evidence for the step from the documents to a home** (an independent review, 2026-11-15). A
  camera-only program written by the reviewer with the public API and the given files only (the LiDAR's home and
  its scored orientation from the drawings, the part files and the photographs; its own calibration, fixture and
  unit poses and moves; nothing of the kit, the reference program or the grader) built the LiDAR onto the deck on
  its shelf (the sub-assembly route, at most 1.5 / 11.5 = 0.130). On the four hidden episodes in the task image
  through `test.sh`: v1, frozen before its first hidden episode, 0.000 / 0.130 / 0.130 / 0.130, mean 0.098; v2,
  only its calibration changed after v1's failure (not blind), the same. Together 7 placements and 6 fastened, every
  one in the scored orientation; no safety event, no illegal call. Its two losses were its own: a diverged wrist
  calibration (v1, episode 0), and the two documented +-1.5 mm plays (the LiDAR in its nest, the deck on its shelf)
  adding up to about the 2.5 mm window when neither is measured (v2, episode 0). Measured by no program so far: the
  homes of the stick pack, the deck and the wheels found from `roller.stl`'s datums, a collision model built from the
  part files and the manual, and moves planned on it.
- **Prior for a frontier model** (score model with the measured lock curves): centre about 0.15 (0.13-0.18), range
  0.04-0.40; the width comes from attempt rates counted on other tasks, so it is an order of magnitude. Levers frozen
  as they were; no configuration edit followed. The independent review reads the centre as more likely high than low:
  the probe behind the measured error levels had the kit's static data, and the steps no program has measured (above)
  are where an honest program loses most.

## Budgets and timing

Per episode sim 650 s, wall 3100 s, 450 camera frames, `step(n <= 500)`, 120 s of silence; 4 episodes. Measured in
the task image's 8-CPU quota: control tick 20-23 ms at the start and about 8 ms once everything is locked, top frame
1.04-1.17 s (the first one 14-15 s more), cell build 4.3 s. Worst case for an agent's program: 4 x (build + wall cap)
plus the kit's unpacking = 12 459 s, plus at most 120 s for the import check, about 12 600 s of the verifier's
14 400 s. The wall cap is a fixed clock: on a slower grading machine only the few hundred seconds of build, unpacking
and finalizing grow, so the margin of about 1 800 s holds for them. The reference program uses 170-177 s of sim per
episode; the oracle's verify takes about 1 250 s. The camera-only probe used 26-55 frames, 188-426 s of sim and
486-1119 s of wall per episode over both rounds.

## Where the files are (published 2026-11-15; folder moved to `tasks/ground/` 2026-11-17)

- The task's large givens (`environment/assets/payload/`: 195 files, 102 319 431 bytes: the six part files, five
  drawings, the picture manual, the arm and gripper models, the cameras' scenery) are on the benchmark's asset dataset
  `grounded-intelligence-initiative/ale-robotics-assets` under `tasks/ground/racecar_full_assembly_tenth_scale_ur10e/payload/`,
  and the hidden kit (`gt.tar.xz`, 36 816 332 bytes, sha256 `328939c3…`, 14 620 files) under
  `…/verifier/gt.tar.xz`. They were uploaded on 2026-11-15 in one dataset commit,
  `a6bfe73e80bc6ea336b91140f229ececc08fba6e`, under the task's folder of that day, `tasks/driving/…` (with the owner's
  approval; the dataset card's row followed in `fd06624f7814c36c4b79688e63d693be83d5cde5`); the bytes downloaded at
  that commit equal the uploaded ones. When the benchmark merged its platforms `driving` and `ground` into `ground`
  (2026-11-17), all 196 files were copied unchanged, server side and in one commit,
  `6cfa80d098df6a6def71f8304edbcd4245fa829f`, to the task's new folder: every file has the size, git blob id and
  sha256 of its source, and the folder downloaded at that commit equals the uploaded bytes (the card's row in
  `ce8a3d2dd5b49cfe66c4b2ba17b660765a9a5899`). 169 of the 195 payload files (`given/robot/`, `render/`) are byte copies
  of `quadruped_full_assembly_12dof_ur10e`'s, uploaded again under this task's prefix, because each task's asset list
  names one prefix.
- `environment/assets.sha256` pins `6cfa80d0…` and every file (the image build downloads and checks them with
  `fetch_assets.py`); `grader_config.json → hidden_gt.url` pins the kit at the same commit (`verify.py` downloads it at
  verify start and checks its sha256 and file count). Both downloads were run against the dataset after the upload
  and again after the move, anonymously, as the image build and the grader do: 195 / 195 files and 14 620 / 14 620
  (`ale validate` repeats both: the image build fetches the givens, verify downloads the kit).
- Filling in the commit changed publication metadata and a document only, recorded under amendment A1
  (`recorded_changes`, 2026-11-15T22:33:35Z); the move changed the same two entries (the asset list's `# revision:` and
  `# prefix:` lines, `hidden_gt.url`), recorded under amendment A3 (`recorded_changes`, 2026-11-17T22:19:43Z). The
  frozen table is unchanged, and the four set episodes' layout digests, recomputed from their seeds, equal
  `hidden/episodes.json`.

## The calibration episode and open points (as of 2026-11-16)

1. **The calibration episode** ran on 2026-11-15/16 with the owner's approval (one Claude Opus 5.5 episode, effort
   high, 6 h): reward 0.104.
2. **The order statement** (a finding of the independent review, settled by the lead designer on 2026-11-15): with the
   deck still on its shelf, the front-left wheel cannot be fitted with the reference program's grasps (every arm
   solution collides with the deck). The instruction and the cell manual say the order is the agent's within what
   the car physically allows and what the fixtures beside the car leave room for; they name no fixture and no unit
   and give no plan (the given models and drawings show the geometry). The calibration episode did not trip on it.

## How to redo

The maintainers' build tooling for this task (the generators, the integration and quality-control harnesses, the
probes) is not part of the repository. From the repository root, with the engine set up as `CLAUDE.md` describes
(`T` = `tasks/ground/racecar_full_assembly_tenth_scale_ur10e`):

```bash
# the freeze check reads the two engine asset roots, which no checkout has: fetch them at the pinned commit first
python3 $T/environment/fetch_assets.py $T/environment/assets.sha256 $T/environment/assets/payload
python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["hidden_gt"]["url"])' $T/tests/grader_config.json \
    | xargs curl -fsSL --create-dirs -o $T/tests/assets/gt.tar.xz     # its sha256 must be grader_config.json hidden_gt.sha256
python3 -I -B $T/tests/review/select_set.py check --selection $T/tests/hidden/selection.json   # the freeze in force (amendment A4)
python3 -I -B $T/tests/review/select_set.py selftest                     # the selection tool's own tests (10)
python3 -I -B $T/tests/review/select_set.py accept --selection <copy of hidden/selection.json> --image <task image> \
    --episodes $T/tests/hidden/episodes.json --root <dir> --out <dir>/accept   # two verify runs of the set in the task image
rm -rf $T/environment/assets $T/tests/assets                             # back to the published form
(cd vendor/ale && VIRTUAL_ENV= uv run ale lint ../../$T)
uv run python scripts/lint_domain.py $T
python3 scripts/task_assets.py check                                     # the asset list against task-assets.toml
uv run pytest -q tests/test_racecar_full_assembly_tenth_scale_ur10e_set.py tests/test_verify_survivor_racecar_full_assembly_tenth_scale_ur10e.py
(cd vendor/ale && VIRTUAL_ENV= uv run ale validate ../../$T --runs-dir <runs dir>)
    # untouched 0.0, oracle 1.0: the image build fetches the givens at the pinned commit, verify downloads the kit
    # and checks its sha256 and file count, every reset checks the episode's layout digest
```
