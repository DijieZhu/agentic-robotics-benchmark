# quadruped_hazardous_plant_inspection: maintainers' notes

**Version 2.1** (2026-11-17). Version 2.0 (the text below) was measured once at reward 0.65 on its sealed bytes (engine
content digest sha256:b2140bdef406c0be...), then amended (amendment 1, section 10) and sealed again
(sha256:5aa95ba4b24fcac1...); it stays on record as version 2.0. Version 2.1 differs from it in exactly two things:
**twelve scored buildings instead of six** (the selection's quota rule with every quota doubled; version 2.0's six are
among them) and **a simulated-time limit of 1350 s instead of 2400 s for every mission**, scored and practice alike (the
training hall keeps its 900 s). Why: one program's score varies by building and by sensor noise from 0.0 to 0.9, and
twelve buildings bring the standard error of a mean from about 0.10 to about 0.07; with 2400 s the slow safe gait and
long searches cost nothing (the reference needs 39 to 48 % of it). Everything else is version 2.0's: the dose rating,
the frame budgets, the program-time cap, the practice buildings and the allowance, the look, the scoring formula, the
lettering, the generator, the reference. What followed from the two changes: the documents, the grader's configuration,
the selection rule and a new admission (A4), the hidden set (section 8), the verifier's time limit (section 7) and the
reference value (section 6); section 11 gives the record. The notes below are version 2.0's where they do not say v2.1.

**Version 2.0** (2026-11-16). Platform `quadruped`, stage `behavior`. Version 1.0 (2026-11-15, engine content digest
sha256:f0752d0757813e7b...) is replaced. What changed against version 1:
the tag plates carry their tag as text only, in the plant's own lettering (no fiducial code; the client's replies name
no marker code), so a program must read a plate to know which instrument it faces; one sheet of each building's
drawing is missing (the documents say what a missing sheet takes away and what stays, not how many instruments it
holds); sills, hoses and low racks also stand away from the drawn doors and the drawn pipe chase; after every
`blocked` the robot takes no walking command for 5 s; the practice buildings show both bund-kerb classes; `nproc`
prints the sandbox's 8 CPUs in the shell (2 inside a run); the buildings are dressed from a render bundle fetched from
the benchmark's asset dataset (`environment/assets.sha256`) and every sensor still renders with MuJoCo's classic
renderer, now with 8 rasteriser threads (decided on 2026-11-16: a Filament render path in a
baked building was prototyped and not adopted). Sections or paragraphs marked "v2" describe version 2; where a
paragraph gives version 1's record it says so. The reference value, the verifier's time limit and the documents' cost
figures are version 2's, measured on the grading machine (sections 6 and 7). Amendment 1 (2026-11-17, the corrections
of the three reviews of the sealed version 2; scores unchanged): section 10. Amendment 2 (2026-11-18, the text for
publication): section 12. Amendment 3 (2026-11-18, a guard beside the BLAS kernel pin and text corrections; scores
unchanged): section 13.

**How to validate.** From the engine checkout `vendor/ale`, with the repository's gate (`CONTRIBUTING.md`):

```bash
uv run ale validate ../../tasks/quadruped/quadruped_hazardous_plant_inspection --runs-dir <runs>
```

It needs no GPU (`task.toml`: the sandbox and the separate verifier each take 8 CPUs and 16 GB; every camera renders on
the CPU). The first run builds the image and fetches the render bundle from the asset dataset. The untouched case ends
within seconds; the oracle's verify runs the reference on all twelve scored buildings and took 1.3 to 1.7 hours on the
loaded grading machine (6,121 s in the sealing validate of version 2.1, 4,639 s in that of the folder as first
published). The expected line:

```
ok    grounded-intelligence-initiative/quadruped_hazardous_plant_inspection: untouched={'reward': 0.0}, oracle={'reward': 1.0}
```

## 1. The task

A 3 kg twelve-actuator quadruped with an on-board gait (modes `walk`, `high_step`, `crouch_walk`, `stand`, `sit`; the
gait is the robot's firmware, the program only commands it), a pan-tilt sensor head (a 90 degree navigation RGB-D
camera, an 18 degree inspection camera, a radiometric thermal camera, a head lamp), a planar lidar, a dosimeter and
drifting leg odometry is set down on the deployment pad in the airlock of a legacy nuclear plant building. The team
holds an outdated general arrangement drawing (a sealed door, an instrument moved to another wall, one moved next door,
an undrawn annex) and a round sheet of ten items: read eight tagged instruments, measure the dose rate at a survey mark,
find one unlisted source. The radio link ends a few metres past the airlock, so the agent writes the robot's
autonomous program (`patrol.py`, `solve(robot)`); the grader runs it on twelve hidden sites (version 2.0: six), each
mission within 1350 s of simulated time (version 2.0: 2400 s).

## 2. Files

| path | what |
|---|---|
| `test.sh` | closes the stage and its parent (chmod 700) before Python starts; builds the root-owned givens at `/opt/arb` from `setup_payload/`; pins HOME, the user site, OSMesa, the BLAS and NumPy CPU paths (the BLAS kernel with its block-size guard, section 13); v2: exports every thread count itself (8 rasteriser threads, OpenMP / BLAS / MKL 2) and the render bundle's switch and path (`SITE_GEN_DRESS=1`, `SITE_GEN_DRESS_BUNDLE=/opt/task-assets/look`), checks the bundle's manifest against its pin (`plantcell/_lib/mission/config/render_bundle.json`: the manifest's sha256; the cell checks the same at every mission) and every bundle file against the manifest (a broken or other bundle exits 1, a task error); `exec "${ALE_VERIFY_PYTHON:-python3}" -s verify.py` |
| `verify.py` | the grader (section 4) |
| `grader_config.json` | run settings only: episodes, the program-time, silence and CPU caps, the sim cap and the frame and report budgets (deep-merged over the cell's defaults), the route gate |
| `anchor.json` | metric `inspection_score` (higher), `value` = the reference's measured score on the hidden set, `full_at` 0.95 |
| `cellkit/` | the verify-stage process kit (lockdown, runner as `nobody`, no-network seccomp filter, network audit, the JSON-lines wire with program-time accounting, the verify-stage freezer, pinned seeds), byte-identical to the copies of the pump skid task (`tasks/arm/pump_skid_startup_and_operation/tests/cellkit/`) and of the ROV subsea manifold task (`tasks/underwater/rov_subsea_manifold_leak_isolation/tests/cellkit/`) |
| `robotics_grader/` | the benchmark's verdict kit, byte-identical to `shared/robotics_grader/` |
| `plantcell/` | the cell: `server.py` (PlantCellServer, the wire side of the robot's API), `client.py` (byte-identical to the agent's `given/lib/plantcell/client.py`), `wire_guard.py` (the grader's guard on the program's request lines around `cellkit.wire`: a line the JSON parser cannot read is a protocol end, method names and ids reach the wire loop plain), `_lib/` (the vendored mission world, facade, evidence and scoring, the instruments and sensors, the site loader, the gait firmware), `data/` (the acceptance file `acceptance_v1.yaml` -- the one source of every tolerance and evidence condition --, the robot model, the shared textures) |
| `reference/` | the privileged reference program (`reference.py`, driven by `reference_run`) and the modules only it needs (its navigation, the reference readers, the Truth object) |
| `hidden/` | `sites/h01..h12.tar.xz` (the stored site packages of the hidden set), `seeds.json` (the twelve sites in episode order: seed, archive, sha256), `selection.json` (the pre-registered admission rule, the frozen files' sha256, every candidate's result), `oracle_token.sha256` |
| `setup_payload/`, `given_hashes.json` | the grader's byte copy of `environment/setup/payload/` and the sha256 of every file in it |
| `review/` | maintainers' tools nothing runs at verify: the generator's authoring half (`sitegen/`), the selection, the baselines as real submissions, the observability and agreement checks, the cell's smoke tests |
| `NOTICE.md` | attribution (the robot description the model follows, the gait's training tools) and the render bundle's provenance |

## 3. Scoring

Per site, from the cell's own state (`plantcell/_lib/mission/scoring.py`, conditions from
`plantcell/data/acceptance_v1.yaml`): S = R x sum(w x c). Each of the eight instruments weighs 0.10 with c = D x E x V:
D, at least one report for the tag reached the team over the link (the last one delivered counts); E, one of the
report's cited frames (at most three), taken in this mission before the report, satisfies the instrument type's evidence
condition, decided by geometry on the grader's side (feature size in pixels, off-axis angle, in-frame share, visible
share by ray casts, distance at most 5 m, the right camera); V, 1 within the type's tolerance, linear to 0 at 2.5 times
it, 0.25 for a value-free report. The survey mark weighs 0.06 (a 10 s dwell within 0.30 m of its centre before the
report; 10 % relative tolerance on the true rate where the robot stood), the unlisted source 0.14 (before the report the
robot's own dosimeter showed a 3 s mean of at least 2.0 Gy/h while its body centre stayed within 3.0 m of the reported
point; full within 1.0 m, 0 at 3.0 m). R = 1.0 when the robot ends upright (roll and pitch below 30 degrees, at any body
height: a robot sitting on the pad has returned) within 1.0 m of the pad centre, else 0.85 (always 0.85 after a fall or
the dose limit). No negative marks. inspection_score = contract_ok x the mean of S over the twelve sites (version 2.0:
six); reward = clamp(score / anchor / 0.95, 0, 1).

## 4. The verify stage (verify.py)

The grader's stages, in order (the architecture of the benchmark's other graders built on `cellkit/`):

0. `_early_lockdown` (stdlib only, first): the stage upload time from the earliest ctime under the stage; kill and
   freeze the agent account until the verdict (the image's supervisord respawns its desktop); lock the stage; a
   root-owned snapshot of the submission (regular files only); late writes into the submission zero the contract gate,
   late entries elsewhere in `~` and the sticky folders are quarantined.
1. The givens at `/opt/arb` against `given_hashes.json` (root-owned, not group/world-writable, sha256): a mismatch is a
   task error.
2. Static gates: `patrol.py` a regular file that parses and defines `solve`; the route gate (60 MB, no weight files, no
   training calls); the import scan (no networking module, no computed import or code, no program start; files
   byte-identical to the shipped client exempt); the import check as `nobody` under the no-network filter (120 s wall,
   120 CPU-s).
3. The oracle token: `.oracle_token` whose sha256 equals `hidden/oracle_token.sha256` makes the cells privileged.
4. The hidden set: `hidden/seeds.json` must pin `episodes` sites (`grader_config.json`: twelve); every archive must be a
   regular file under `hidden/` whose sha256 matches the record (else a task error, also for an untouched submission).
5. Before the first mission: the frozen desktop settles, the no-network filter proves itself.
6. Per site, under an opaque tag `epNN`: a fresh `PlantCellServer` on the package (`reset(package, seed=...)`); the
   program as `nobody` from a read-only copy of the snapshot, `python -B -E -s`, the client library from `/opt/arb`,
   `prlimit --cpu` = 4 x the program-time cap, the seccomp no-network filter, its runner account's remote inet sockets
   sampled; the wire (`cellkit.wire`, with `plantcell/wire_guard.py`) until the program finishes, falls silent (180 s),
   stalls, breaks the protocol or reaches its program-time cap (1200 s of its own time: the wall clock minus the cell's
   work); the privileged oracle's cap is 3600 s. Every process of the runner account is killed and its stray files swept
   after each mission.
7. A mission in which the cell failed (an internal error of the cell's own work while the mission was running -- every
   argument is checked at the server's boundary first and refused with `bad_params`, an error after the end never
   counts --, a physics fault, a frame it could not render) is run once more from scratch and the second run counts; a
   render failure that repeats is a task error. At most one mission of a verify is run again (`MAX_RERUNS_PER_VERIFY`,
   the timeout below is derived for it): a second mission in which the cell failed is a task error.
   A mission score that is not a finite number in [0, 1] is a task error (`min(1.0, nan)` is 1.0).
8. The verdict through `robotics_grader` (finite metrics only); records under the root-only `/opt/arb-verify`
   (`out/episodes.json`, `out/traj_epNN.npz`, `logs/epNN.log`).

Untouched sandbox: no `patrol.py` -> contract_ok 0 -> reward 0. Oracle: the token and `reference_run` -> the anchor.

## 5. Realism audit

* Given: the robot's manual and API (datasheet level: rated envelope per gait mode and obstacle class as measured,
  sensor specifications, odometry and dosimeter specifications, the evidence conditions), the facility notes, the round
  procedure and site rules, the instrument catalogue as printed (v2: with the plant lettering as numbers: a tag plate
  carries its tag in letters and no code), the training hall's plan and calibration sheet, the client library, an
  example program that scores 0, and the practice ground (a black-box service under its own
  account `plantcell`, mode 700: the training hall and three training sites from a label disjoint from the hidden set,
  30 missions, two at a time, a per-item debrief after each mission, no total, no as-built drawing).
* v2: one sheet of a building's drawing can be missing (the documents say what a missing sheet takes away and what
  stays; they do not say how many instruments it holds or how the sheet is chosen).
* Not given: the simulator's code, the site generator and its parameters, any hidden site's layout, values or truth,
  the true pose (only drifting odometry), the noise processes, the physics settings, the hidden packages' names or
  seeds, any score of a graded site.

## 6. The oracle and the anchor

The oracle installs `patrol.py` (one frame of every sensor, then `reference_run`) and the token. The reference program
runs inside the cell with the privileged Truth (the true pose, the true layout, the traversability map, each
instrument's pose and acceptance region, the unlisted source's position for planning its survey); it never gets a
value: every reading comes from its own frames through the reference readers, the mark's rate from its own counts, the
source's position from a fit of its own counts. The anchor is the inspection_score of one full verify of the oracle
submission in a fresh container of the task image, in the separate-verifier layout (`anchor.json` `source`).
v2: **1.0**, `full_at` 0.95, measured on the grading machine on 2026-11-16 22:08-22:55 EST (an image built from this
`environment/` with the render bundle fetched from the asset dataset; `tests/test.sh` as root with only the submission
restored, the verifier's 8 CPUs and 16 GB, host load 110-200 of 48 cores): every site S 1.0, contract_ok 1, no mission
run twice, 2763 s of verify wall clock (392-562 s a mission), a cgroup memory peak of 3.9 GB. The reference's time
share 0.387-0.478 of the 2400 s, dose share 0.132-0.309 of the rating, at most 0.280 of any frame budget (the
lidar's). The six missions ended in the admission's final states bit for bit (the admission ran on another machine
with another CPU generation), and in the same states as two earlier verify rehearsals there; the untouched
submission scored 0.0 (contract_ok 0). Version 1's anchor (1.0, 2026-11-14, 2169 s) was measured on version 1's set.
Amendment 1 (section 10): the anchor's `source` now names an oracle verify of the amended bytes (2026-11-17, value
1.0, the same six final states).

v2.1: **0.999586**, `full_at` 0.95, measured on the grading machine on 2026-11-17 15:24-16:52 EST (an image built from
this `environment/` with the render bundle fetched from the asset dataset; `tests/test.sh` as root with only the
submission restored, the verifier's 8 CPUs and 16 GB, host load 29-305 of 48 cores): S 1.0 on eleven of the twelve
sites and 0.995 on one, contract_ok 1, no mission run twice, 5290 s of verify wall clock (362-572 s a mission), a
cgroup memory peak of 4.2 GB. The reference's simulated time 879-1146 s (0.65-0.85 of the 1350 s; it does not read the
limit), dose share 0.124-0.401 of the rating, at most 0.280 of any frame budget (the lidar's). The twelve missions
ended in admission A4's final states bit for bit (A4 ran on another machine with another CPU generation; its final
states are A3's); the untouched submission scored 0.0 (contract_ok 0). By this anchor the oracle's reward is 1.0
(0.999586 / 0.999586 / 0.95, capped at 1).

v2 (the reference as it is in `reference/`). The tag plates carry text only: before each instrument report the reference
reads the plate with its own text reader (`bricks.readers.tagtext`, reference only) from its own inspection frames and
reports only when the text equals the work order's tag (no report otherwise); every read is in its log (`plate_read`),
and the selection's row 2 asks for all eight read before their reports. What is privileged and what is not, for the
plates: the plate's place in a frame (the crop, the aim) comes from the true pose and the true plate box; the text and
the plate's pose come from the frame's pixels; **the pose's mirror solution (the two planar poses a border gives) is
settled with the round sheet's recorded facing** (public, to 5 degrees), and where the round sheet records none (an
undrawn annex) or the chosen pose lies more than 15 degrees from it (a moved or relocated instrument: its recorded place
is another wall or room), **with the plate's surface normal from the robot's own navigation depth frame** (one frame
aimed at the plate); the true plate normal is never an input (the log carries the chosen pose's angle to it,
`true_normal_err_deg`, for the record only). The quarter-turn valve's reader takes the lever's plane from that plate
pose. The hold after `blocked`: the reference's driver keeps its command going and goes on when the hold ends; its step
off an element it stands on waits the hold out first; after a square crossing its walk never turns back toward the
element it has just crossed, and an element the crossing went over as well (a hose right behind a door sill) counts as
crossed (on two development sites these had cost 9 and 19 `blocked` and about 95 and 150 s). Further: the step off an
element ends when the body has moved 0.40 m (true pose), not after a commanded time; a walk blocked with no known
element at hand backs off without turning in the cramped place (back along the way it came, to the first point with room
to turn), and the place where it was blocked becomes an obstacle for its planner, so the same route is never planned
again (no other route: that goal is given up); survey stances never lie on cells the map calls tight (no turning);
walking from a survey stance in a strong field (a minute there costs 1 % of the dose rating or more, by its own counts)
the first `blocked` ends the attempt (no retries in place); when a walk ends `blocked` at an element, the mode-change
crossing moves back to its approach point in the element's own mode and without turning (in walk the step guard refused
every step there: 38-46 s standing on two development sites), and its exit ends where that mode can still enter the
floor beyond the element. On a development site that case had cost 15 `blocked` and 95 s at 19 Gy/h with the same route
re-planned five times; the reference stays time-optimal otherwise (no dose-aware planning). And: a walk that ends
`blocked` is put down to what the reference's own map shows in contact with the body or within 8 cm of its footprint: an
element that needs a mode keeps the mode change; a door leaf, a wall or equipment there is named, and the robot backs
off along the way it came and plans again with clearance from it (0.35 m from its outline near the place, kept for the
rest of the mission wherever a way with it exists; where none does, the place itself is avoided for that goal); a free
element (one walk is rated across) is never crossed in another mode: when it is all that is at hand the block is a
`blocked` at a rated size, backed off and walked again once for that goal. (In the first admission: a hind shank against
an ajar door's leaf, a free 2 cm sill under a front foot; the reference had tried a high_step crossing of the sill three
times and stood 32 s.)

## 7. Timeouts

`task.toml`'s header derives the verifier timeout from numbers measured in a container of the task image with the
verifier's 8 CPUs and 16 GB: per site at the bounds, the build and the renderers' start, the 1200 s program-time cap,
the mission's simulated-time limit (1350 s since version 2.1; 2400 s before) at the slowest measured pace, every frame
budget at its sensor's p95 cost, the wire's JSON, the 60 s grace after a mission is over and the clean-up; one mission
more than there are sites (verify.py runs at most one mission again: a second cell failure is a task error) plus the
fixed steps (import check, desktop settle, no-network self-test, lockdown); at least 15 % to spare. v2: measured on the
grading machine on 2026-11-16 21:51-22:05 EST in a container of the final task image (the render bundle fetched from the
asset dataset at build time) with the verifier's 8 CPUs and 16 GB, host load 81-124 of 48 cores: a bench program (a
fixed program through `~/training/run_mission.py`, two runs of one 300 s mission alone and of two at a time on practice
buildings t1 and t2, the program's own timing of every call) and a probe of the cell's first frames on the four practice
packages, and the cell's build of the six scored packages in the reference verify of section 6; every input is the
largest measured, over the runs of one mission alone (the verifier runs its missions one after the other): build 3.0 s,
renderers' start 4.2 s, the whole-loop pace 0.4435 s of wall clock per simulated second (its own frames and the wire
included; `step()` alone 0.348), p95 per call 0.466 s (navigation with depth), 0.549 s (inspection), 0.124 s (depth),
0.108 s (thermal), 0.028 s (lidar), encoding and wire included; about 2947 s a site at the bounds, 20882 s in all for
seven missions, so version 2.0's timeout was 24100 s (15.4 % to spare; version 1's was 23400 s). **v2.1** (twelve sites,
1350 s): the same measured bench and costs (no pace or frame cost depends on the limit; the bench was not run again),
the 1350 s at 0.4435 s per simulated second (599 s) and the build 3.4 s (the twelve's largest, section 6): about 2482 s
a site at the bounds, thirteen missions and the fixed steps 32516 s, so the timeout is **37400 s** (15.0 % to spare; the
engine and the lints set no upper bound on it). The documents' figures derived from the limit (a full mission about 10
min, the whole practice allowance 2.5 to 2.6 h) come from the same costs file. A memory peak of 3.9 GB for the whole
oracle verify (cgroup). Eight rasteriser threads (`LP_NUM_THREADS=8`, byte-equal frames to one thread's) in the image,
the practice ground and `test.sh`. The same bench is the source of the documents' cost figures (`instruction.md` and
`given/robot_manual.md`, blocks `costs`, written by the maintainers' document tool from the bench's cost record: the
lowest median to the highest 95th percentile over the runs).

## 8. The hidden set and the measured programs

### v2.1 (2026-11-17; the scored set of version 2.1)

* **Selection** (`hidden/selection.json`, `review/select_set.py` of version 2.1): the same 48 candidates of
  `.../hidden/v2/<k>`, the same label, tables and exclusion list; version 2.1's rule: the quota rule with every quota
  doubled (at most six of each bund kerb class, of each B-step variant and of each B-low variant, at most four of each
  sheet place, in k order, complete at twelve), row 3's time part the absolute bound it meant (the reference finishes
  within 1200 s of simulated time; 50 % of the dose rating and 60 % of each frame budget as before), rows 1, 2 and 4
  unchanged. The amended rule was fixed on 2026-11-17 at 12:14 EST (in the maintainers' records, with its set) before
  version 2.0's submission was run on the candidates, and pre-registered before admission A4 ran (the registered part,
  as the version 2.1 tool's `rule` wrote it, was fixed by its digest at 14:38:54, before any mission of A4: d1071ea2...,
  the digest of the same part in `hidden/selection.json`; only `rule` and `rule_at` differ from version 2.0's registered
  part, digest 4167b0e0...). Frozen: 114 files, A3's with three changed (`review/select_set.py`, `grader_config.json`,
  the cell's `mission.yaml`; a freeze amendment of the maintainers' sources), and the 48 packages, A3's bytes.
  **Admission A4** (`amendments` A2, A3, A4; `a3_admission`, `notes_a4`): all 46 hidden and 11 practice oracle runs (on
  the admission's machine, 2026-11-17 14:39-15:20, in an image of the version 2.1 task built with the render bundle from
  the Hub) end in A3's final states bit for bit, with A3's score, time, dose, frame counts, reports, holds, model hash,
  attempts and row-4 scores; the same 40 admitted; the same practice triple (practice k 1 2 6; the practice ground's
  index and packages byte-identical); only the status words of candidates whose place the doubled quotas change differ;
  the route-health check on all 57 runs: no defect. The set: k 1 2 3 5 6 7 8 9 10 11 28 36 = h01..h12; version 2.0's six
  are among them with the same packages and seeds (k 28 is h11 now, h06 before).
* **The twelve** (aggregate): bund kerb 6 low / 6 high, B-step 6 at door sills / 6 in the room, B-low 6 at the chase / 6
  at a rack, the missing sheet 4 along / 4 across / 4 diagonal; listed instruments on the missing sheet: 2 on seven, 3
  on five (none with 4); C-sealed reached through the chase on four, through the room on eight. The reference: S 1.0 on
  eleven and 0.995 on one; 879-1146 s simulated (0.65-0.85 of the 1350 s), dose share 0.124-0.401, at most 0.280 of any
  frame budget (the lidar's), 0 or 1 `blocked` hold; route (the generator's plan) 139-195 m. Row 4 on each: the four
  degenerate programs 0.00, corridor-only at most 0.10.
* **Measured programs**: as for version 2.0, no pilot and no baseline ladder; version 2.0's episode (reward 0.65) was
  measured on version 2.0's bytes before amendment 1 and is not a measurement of version 2.1.

### v2 (2026-11-16; the scored set of version 2.0; replaced by v2.1's above)

* **Selection** (`hidden/selection.json`, `review/select_set.py`): candidates k = 1..48 of the label `.../hidden/v2/<k>`
  (the table by candidate number with its type repair and the exclusion list, sha256 3413ac3b), each put through the
  admission rows on its own; the set by the quota rule over the admitted candidates in k order (at most three of each
  bund kerb class, B-step variant and B-low variant, two of each sheet place; complete at six). The rule, the label, the
  k range, the tables and the exclusion list were written before any seed of the label existed (2026-11-16 13:00 EST,
  `rule_at`) and are kept byte for byte (the pre-registered part's digest 4167b0e0...). Frozen: 114 files (the cell, the
  reference, the verify kit, verify.py, test.sh, the oracle's program, the baselines, the selection and practice tools,
  Dockerfile, requirements, `environment/assets.sha256`, the exclusion list) and the 48 candidates' packages. Four
  admissions, every record kept: the first (the reference before its fix for blocks at the layout's fixed geometry)
  stopped on the route-health condition (32 s standing still on k3); A1 (the reference with that fix), A2 (the same,
  with `environment/assets.sha256` carrying the bundle's dataset commit, the trial image's bundle fetched from the Hub)
  and A3 (amendment 1's three frozen files, section 10; this record, `amendments` A2 and A3, with `first_admission`,
  `a1_admission` and `a2_admission`) end bit for bit alike on all 46 + 11 runs; against the first, only k3 (hidden) and
  k10 (practice) changed. 46 of 48 candidates built (k15, k43: no layout within 64 sub-seeds); all 46 pass row 2 (all
  ten items with accepted evidence, the eight plates read as their tags before the reports, upright on the pad; S 1.0 on
  43, 0.92-0.995 on three); six drop at row 3 on the dose share (0.55-0.75 of the rating: the accepted lean of row 3);
  row 4 holds on all 46 (the four degenerate programs 0.00, corridor-only at most 0.10); 40 admitted. The set: k 1 2 3 5
  6 28 = h01..h06 (18 admitted candidates skipped by a quota, 16 not taken once the set was complete). The record's
  `set_detail`, `first_admission`, `a1_admission`, `admissions_compared` and `route_health_a1` hold the details.
* **The six**: bund kerb low / high / low / low / high / high; B-step door sills / in the room / in the room / door
  sills / in the room / door sills; B-low chase / rack / chase / rack / chase / rack; the missing sheet along / across /
  diagonal / across / diagonal / along; two listed instruments on the missing sheet on each (the built candidates 2-4,
  mean 2.52); C-sealed reached through the chase on two, through the room on four. The reference's time share
  0.387-0.478 of the 2400 s, dose share 0.132-0.309, at most 0.280 of any frame budget (the lidar's), 0 or 1 `blocked`
  hold; route 139-173 m.
* **Route health** on every admission run (46 + 11): no stand-still over 30 s without a report (the longest
  27.5 s: instrument readings with their reports), no route planned twice, no mode-change crossing at a free element.
* **The practice ground** (the practice rule over practice candidates 1..12 of `.../practice/v2/<k>`, 11 built; the
  practice rule has no dose row): t1, t2, t3 = practice k 1, 2, 6 (passed over k 4, 5), with the training hall built by
  the frozen tree (package fcf76a39...); between them both bund classes, both variants of each B obstacle, the three
  sheet places and every deviation class.
* **Measured programs**: no pilot and no baseline ladder were run for version 2 (the maintainers' rule: a privileged
  reference that completes the task, plus one evaluation episode of a frontier model). The degenerate checks are the
  admission's row 4 on every candidate (above) and the untouched submission in the engine's `ale validate` (0.0).
* **The grading machine** (2026-11-16): the documents' cost figures and the verifier's time limit (section 7), the
  reference value through the real verify path (section 6), `ale lint`, the benchmark's domain lint and `ale validate`
  on these bytes.

### Version 1 (2026-11-14; its scored set is replaced)

* **Selection** (`hidden/selection.json`, `review/select_set.py`): candidates k = 1..24 of the site generator (the label
  rule in the record), frozen at 23:07 EST before any run: 99 files (the cell with its acceptance file, firmware and
  loader; the reference with its readers and navigation; the verify kit, verify.py, test.sh, the oracle's program, the
  baselines, the selection tools, Dockerfile, requirements), the 24 packages' sha256 and version 1's selection rule as
  amended (row 3: 50 % of the 2400 s, 50 % of the dose rating, 60 % of each frame budget). One amendment, with its
  reason: it replaces a freeze of 21:56 that was never decided for the record (the generator's documents-only
  regeneration of every plant package after the documents review, and three cell fixes of that review). The pool: the
  frozen reference and the five row-4 baselines on all 24 (144 missions, each counted on its first attempt). 21 of 24
  pass every row; three drop on row 3 (two on the dose share, one on the time and dose shares). The set: the first six
  admitted in k order, k 1 2 3 5 6 7 = h01..h06.
* **The six** (aggregate): route length 123-156 m; the reference's time share 0.35-0.39 of the 2400 s, dose share
  0.16-0.25 of the rating, at most 0.23 of any frame budget (the lidar's); every site carries the ladder's required
  deviations (a sealed door, a moved and a relocated instrument, an undrawn annex, an as-built opening, a dead
  repeater, an outdated zoning), the optional added opening two of them and the added partition three; the eight
  slots hold six different instrument types each over the six sites. Row 3 leans the set to the lower-dose sites of
  the 24. No place on a site's required access where two floor elements, or an element and an obstacle, lie within
  0.6 m of each other needed more than one try of the reference.
* **Baselines and the pilot** of version 1 were measured on version 1's set and documents; they say nothing about
  version 2 and are not repeated here.
* **Timings** (8-CPU containers of the task image on the loaded build host): the oracle's verify 2169 s (the engine's
  `ale validate` 2166 s); the selection's 144 missions on a 32-CPU pod with 12 workers 1284 s. `ale validate`:
  untouched 0.0, oracle 1.0.
* **The reviews and what was done**: the scoring review (every argument checked at the cell's boundary; the request line
  guarded; at most one mission re-run per verify, only for the cell's own failure while the mission runs; the source's
  approach within 3.0 m of the reported point; the evidence rule's reference point in the practice debrief; the valve
  words for quarter-turn valves only; unlisted temperature points warm; the lamp and number forms), the
  grading-environment review (the one-re-run budget; the snapshot's skipped entries counted), the documents review (the
  moved temperature point's own component and units: that documents-only regeneration; the practice tool ends with the
  mission; the steps after the end counted once; `pitch_cmd` after the cut; no privileged calls in the agents' client;
  every end of a practice mission gives its reason and the debrief).

## 9. Known limitations

* Every sensor uses MuJoCo's classic renderer through OSMesa (per-vertex lighting); the readers' thresholds were
  measured on it, and are re-measured if the look changes. v2: the look comes from a render bundle in the image
  (`environment/assets.sha256`, `stage_assets.py`, `fetch_assets.py`; `/opt/task-assets/look`), required at grading
  and in the practice ground (`mission/bundle.py` refuses a missing or altered bundle); 8 rasteriser threads give
  byte-equal frames to one thread's (measured in the task image) and are cheaper. v2: the bundle is fetched from the
  benchmark's asset dataset at commit `6f72bfe3329ecb60d8ab11b5c266da0d83cd8abe` (`environment/assets.sha256`, written
  with the benchmark's `scripts/task_assets.py list`; a build with `QINSPECT_ASSETS=hub` and no local copy verified the
  tar and its 62 files, 2026-11-16; again on the grading machine for the final image). The robot's cameras stay on the
  classic renderer (decided on 2026-11-16): the benchmark's house method for agent cameras
  (MuJoCo's Filament renderer in a Cycles-baked scene) was prototyped on generated buildings and not adopted, because
  it would have needed a bake per building, the reading studies again, a new freeze and a new admission.
* The gait is a learned controller; its protective behaviours (blocked, protective sit, fallen) are the firmware's.
* One of the maintainers' cell tests fails on one practice building: `review/cell/test_source_approach.py --site`
  with t1 ends its ring mission `robot_lost` before the report (a `CellError`: the mission is over); with t2 and t3
  it passes. The test places the robot kinematically; at its second ring stance on t1 (x 18.38 m, y 1.38 m, a
  walk-enterable floor cell beside a wall) the robot is thrown over within the first half second (roll -130 deg,
  pitch 55 deg), touches wall W020 and comes to rest on its back, and the cell's tilt protection (more than 70 deg
  for 2 s) ends the mission. The test's code dates from version 1 (2026-11-14; only its docstring changed since),
  when it passed on version 1's practice buildings; it fails the same way on version 2.1's sealed bytes and on
  these. No program can place the robot, and the rule the test checks is shown on t2 and t3.

## 10. Amendment 1 (2026-11-17): the corrections of the three reviews

Version 2 was sealed on 2026-11-16 and measured once (one episode of a frontier model). Three independent
reviews of the sealed bytes (documents against the cell, scoring and evidence, the grading environment) found nothing
that made that measurement unfair, and asked for these corrections before publication. No rule of scoring, no
tolerance, no budget, no site package, no file of the cell (`tests/plantcell/`) or of the reference (`tests/reference/`)
and no file of the shared kit (`tests/cellkit/`) changes; scores are unchanged (below).

**The grader** (`tests/verify.py`):
* Memory: the import check's and every mission's program command start with `choom -n 1000 --` (util-linux, in the
  image), innermost, inside `cellkit.runner.wrap_limits`' account switch, CPU limit and no-network filter. The program
  and every process it starts run with oom_score_adj 1000, the grader with the image's 0: when a submission's processes
  together use up the verifier's 16 GB, the kernel kills one of them and the mission is scored as it stands; before, it
  could kill the grader (a verify without a result). A program could still write its own score back down to 0 (the floor
  it inherits); that brings back the old outcome, a task error, never a better score.
* Unreadable sources: where verify.py parses `patrol.py` it catches RecursionError, MemoryError and ValueError beside
  SyntaxError, and before the network audit it parses every `.py` file of the submission itself (`unparsable_files`: the
  audit's walk, bytes and decoding); a file the parser cannot read (about 10,000 chained operators or more) fails the
  contract, closed, and the call of the audit is guarded with the same errors. A SyntaxError in a file nothing imports
  stays the audit's rule (skipped). When the static gates fail the contract, the verdict's diagnostic now names the
  reason (a verdict with contract_ok 1 reads as before; the local `verdict.local.json` carries the diagnostic too).

**The image and the entry points**: `/opt/venv/.lock` (mode 666 from `uv venv`, the only world-writable regular file of
the image, a channel from one scored mission to the next) gets mode 0644 in a step of its own after the venv's layer;
`SITE_GEN_DRESS_SHADOWS=0` is pinned in `tests/test.sh` and `environment/setup/setup.sh` (the value an unset
variable already gives); the two dressing variables leave the image's `ENV` (the agent's shell no longer shows
them), setup.sh and test.sh set them, and the Dockerfile's proofs pass them explicitly. Two proofs are new: the agent
account's environment names no `SITE_GEN_` variable, and the image holds no world-writable regular file.

**The documents** (through the maintainers' document tool and its sources; its own test, now 49 cases): the practice
sentence of the instruction says what the practice buildings show (a relocated instrument only in a neighbouring
room); a value nested one level down is taken and is unreadable; `stall_detection` in the API's row of
`modes.tables`; a tag plate shows darker than its paint colour in a frame, with the same hue; the two-pose
ambiguity of a single grid marker seen near head-on; every lamp word the scoring accepts, generated from
`scoring._lamp`; a process that leaves the program's process group is not stopped by a practice run, every
process of the account is in a scored mission (robot_api.md section 10 and `training/run_mission.py`'s
docstring); one sentence on memory, generated from task.toml (the program and the robot cell share the sandbox's 16 GB;
in a scored mission the kernel then stops one of the program's processes); the practice debrief's
`practice_ground_failure` row also names a sandbox whose memory the program used up; a printed range whose two ends
round alike prints one value ("about 0.44", never "0.44 to 0.44"). Not changed: the sentence on the reference program,
the tools list; build inputs that float between rebuilds were not pinned.

**Tests** (maintainers', `tests/review/grader/`, as root in a fresh container of the task image, on practice building t1
as a one-site set, never a scored building): `test_oom_first.py` (the program, a child and the import check at 1000, the
grader below; `--fill`, in a container capped at 16 GB without swap: 44 children of 400 MiB, the kernel kills children,
test.sh exits 0 with a verdict) and `test_parse_limits.py` (four submissions beyond the parser's limits: contract_ok 0
with the file named; a plain SyntaxError helper: contract_ok 1 as before). On the bytes before amendment 1 both fail
(the import check at 0; test.sh killed, exit -9; the grader's exception, exit 1, no verdict).

**Scores unchanged** (measured on the amended bytes in a scratch copy, image built with the bundle from the Hub): the
oracle's verify gives reward 1.0 (inspection_score 1.0, every site S 1.0, no mission run again, 1785 s), its six final
states (`final_state_sha256`) and every scored field equal the sealing run's (2026-11-16) 6 of 6; the episode's kept
submission, verified again, gives 0.54, 0.17, 0.90, 0.80, 0.64, 0.64 (inspection_score 0.615, reward 0.65), as the
episode did, with its six final states equal 6 of 6 to a reproduction on the sealed bytes. The test submissions of the
engineer who re-checked the grading-environment review (deep expressions at and beyond the parser's limits; one process
and then 44 small ones filling the 16 GB; a pool of 48 workers) each end with test.sh exit 0 and a verdict.

**Records.** `selection.json` freezes `tests/verify.py`, `tests/test.sh` and `environment/Dockerfile` (and the grader's
environment, which gains `SITE_GEN_DRESS_SHADOWS=0`), so the admission was run again under the pre-registered rule (the
selection's rule: a change to a frozen file needs a new admission), after a freeze amendment of the sources (two of the
maintainers' tools and the documents' own test; source digest c8617733...; the earlier freeze records unchanged).
**Admission A3** (2026-11-17 06:33-07:24 EST, on the admission's machine, in an image of these bytes built with the
bundle from the Hub; this record, `amendments` A2 and A3, with `a2_admission`): the frozen files differ from A2's in
exactly those three, the recorded environment in exactly that variable; the candidates' frozen part and the
pre-registered part (4167b0e0...) are A2's. All 46 hidden and 11 practice oracle runs end in A2's final states bit for
bit, and all 276 + 66 missions (the oracle's and the five row-4 programs') with the same scores; every time, dose, frame
count, report count, hold count, model hash and status is the same; the route-health check finds no defect. 40 admitted,
the same scored set (candidates 1 2 3 5 6 28 = h01..h06, the same six packages and `seeds.json`) and the same practice
triple (practice candidates 1 2 6, the practice ground's packages and index byte for byte). **The anchor**
(`anchor.json` `source`) comes from an oracle verify of these bytes on the grading machine (2026-11-17 07:25-08:04 EST,
an image built from this `environment/` with the bundle from the Hub, `tests/test.sh` as root with only the submission
restored, the verifier's 8 CPUs and 16 GB, host load 37-47 of 48 cores): value 1.0 (inspection_score 1.0, every site S
1.0, contract_ok 1, no mission run again, 2335 s of verify wall clock, 352-440 s a mission, a cgroup memory peak of 4.0
GB); its six final states equal the sealing run's and A3's records 6 of 6, and every scored field the sealing run's; the
untouched submission scored 0.0 (contract_ok 0). **`ale validate`** (engine bfb7505, 0.1.1) on these bytes in the
benchmark's layout, 2026-11-17 08:05-08:47 EST: ok, untouched 0.0, oracle 1.0 (inspection_score 1.0, every site 1.0, no
mission run twice, verify phase 2505 s), no failure, no warning. `ale lint` and the benchmark's domain lint: ok.

## 11. Version 2.1 (2026-11-17): twelve scored buildings and a 1350 s limit

Decided after version 2.0's episode: twelve scored buildings and a tighter time limit, 1350 s. Version 2.0's
episode (reward 0.65, `inspection_score` 0.615) ran on version 2.0's bytes before amendment 1 and is version 2.0's
measurement; it says nothing about version 2.1 except as a point of comparison.

* **What changed and why.** (1) Twelve scored buildings instead of six, by the quota rule with every quota doubled
  (section 8): one program's score varies by building and by sensor noise from 0.0 to 0.9, and twelve buildings bring
  the standard error of a mean from about 0.10 to about 0.07. (2) The simulated-time limit of every mission, scored and
  practice, is 1350 s instead of 2400 s (the training hall keeps 900 s): with 2400 s the slow safe gait and long
  searches cost nothing. The admission's row 3 keeps its time bound as the absolute 1200 s it meant, so the admitted
  candidates are the same; the reference does not read the limit and finishes every one of the twelve within 1146 s.
  Unchanged: the dose rating 2.0 Gy, the frame and report budgets, the 1200 s program-time cap, the practice buildings
  and the allowance of 30 missions, the look, the scoring formula and weights, the lettering, the generator, the cell's
  code, the reference.
* **Files.** `grader_config.json` (`episodes` 12, `time.sim_time_limit_s` 1350); the cell's `mission.yaml` (its three
  copies: the cell, the practice ground's cell, the generator's copy; from the maintainers' source);
  `environment/practice_ground/sites.json` (the buildings' 1350 s; the hall 900 s); `review/select_set.py` and its test
  (twelve, quotas 6 / 6 / 6 / 4, row 3's absolute 1200 s; the rule's text says so);
  `review/cell/test_time_limit.py` and `test_practice.py` (they expect 1350 s); the documents (below);
  `hidden/` (twelve packages, `seeds.json`, `selection.json` of admission A4); `task.toml`'s verifier timeout (section
  7); `anchor.json` (section 6). In the maintainers' source tree: their assembly, check, document and anchor tools, the
  documents' test, and a freeze amendment of the sources.
* **What the agent is told** (through the maintainers' document tool; its checked sentences): the instruction says
  twelve hidden sites, 1350 s of simulated time, the mean over the 12 missions, and that the reference used "at most
  1200 s of simulated time (about 89 % of the limit), half of the dose rating and 60 % of each frame budget"; the
  budgets tables (instruction, API, manual), the practice tables (1350 s per practice building), the site rules' end
  of a mission; the cost figures derived from the limit, from the same measured paces (the bench's cost
  record): a full mission about 10 minutes for the simulation alone (was 18), the whole practice allowance at
  full length 2.5 to 2.6 hours (was 4.5 to 4.6).
* **Records.** a freeze amendment of the maintainers' sources (the five changed sources, source digest 6570a859...; the
  cell's own freeze check now reports exactly the mission.yaml and the selection tool and its test, as recorded); the
  pre-registration of the rule's registered part (written 2026-11-17 14:38:54 before any mission of A4; the part's
  digest d1071ea2... is that of the registered part in `hidden/selection.json`); admission A4 (section 8;
  `hidden/selection.json` with `amendments` A2, A3, A4 and A4's comparison with A3; `hidden/seeds.json`: twelve seeds,
  version 2.0's six unchanged); the reference value through the real verify path on the grading machine (section 6); the
  verifier's time limit (section 7).

## 12. Amendment 2 (2026-11-18): the text for publication

Before the folder was published, its text lost every reference to material outside this repository: paths into the
maintainers' source tree and scratch folders, machine and host names, the names of the build's work lanes and waves,
tool, test and record names that are not in this folder, commits of the maintainers' own repository, and the ids of
design decisions, draft rules, work items and review findings, which cite documents that are not in this repository.
Where an id stood for a reason, the reason stays in plain words; where it stood alone, it went. Comments and notes that
were no longer true were corrected in the same passes (pending notes whose condition had been met, a key the packages do
not hold, the shared kit's copies in other tasks, a replay tool said to be in the folder, the practice label of version
2). The history above keeps its facts and numbers. The amendment was made in four passes, each after the first following
an independent check of the one before; the record below covers all four. What could not change without changing code,
data that code reads, the agent's bytes or the registered rule is listed under "What stays".

* **What changed.** Comments and docstrings of the Python files (the cell, the reference, the generator's copy, the
  baselines, the review tools, `verify.py`, the practice service `environment/practice_ground/practiced.py`), the
  comments of the YAML files, `test.sh`, `setup.sh`, the Dockerfile and `task.toml` (its header said six hidden sites;
  it says twelve), the note strings of the render bundle's pin (`plantcell/_lib/mission/config/render_bundle.json` and
  its copies), the generator's `capability.json`, `anchor.json`'s `source`, `hidden/seeds.json`'s note, the practice
  ground's `sites.json` (its note, and its `rule` text, which named version 1's practice label and which no code reads),
  the prose of the acceptance file's `plate_readable.measured` block (string values that no code reads, two of their
  keys renamed to plain names; below), `NOTICE.md` (the render bundle's provenance; it said that no third-party dataset
  is used, which stopped being true with version 2's look), the cell's test list `review/cell/README.md`, these notes,
  and the notes of `hidden/selection.json` (eight of its keys that were decision ids or wave names now have plain names,
  among them `notes_a1_a2` and `route_health_a1.blocked_cause_flags`; each old name is recorded by its sha256 only, and
  no code reads them). The vendored copies (`plantcell/_lib/`, the practice ground's copy of the cell,
  `reference/_lib/`, `review/sitegen/`, `review/_lib/`) are byte copies of the amended sources, as before.
* **What did not change.** No code, no data value that code reads and no configuration value (`anchor.json`'s `source`,
  a note the grader does not read, is copied by the repository's registry export as the task's `anchor_source`, so the
  registry shows its new words); no byte the agent can read (`instruction.md`, `environment/setup/payload/`, its mirror
  `setup_payload/`, `given_hashes.json`, the client library): the measured episode read these bytes. The site packages,
  the seeds, the anchor's value and the selection rule's registered wording are the same.
* **The record.** `hidden/selection.json`, `amendments` entry `A5-text` (kind `text-only`): each of the 62 frozen files
  that changed with its sha256 before and after (the table `frozen` stays the one admission A4 froze), the notes of the
  record cleaned field by field (each old text and each renamed key by its sha256 only), and the proofs. No admission
  was run: no file's behaviour changed, and the stored packages are what the generator makes from their seeds apart
  from the three provenance labels and the mission limit recorded at generation (proof (c)).
* **Proofs.** (a) and (b) were made on the planned copies of the files before any file was replaced, (c) and (d) on the
  amended bytes afterwards. (a) Every changed Python file has the same syntax tree once its docstrings are left out, the
  same string constants and the same number of comments. (b) Every changed JSON file other than `hidden/selection.json`
  has the same keys and structure and differs in note strings only (the practice ground's `sites.json` also in its
  `rule` text, which no code reads; `hidden/selection.json`: the `A5-text` entry, and its notes and renamed keys as
  recorded there); every changed YAML file parses to the same value, except the acceptance file (all three copies),
  which parses to the same value once its `plate_readable.measured` block is set aside: that block holds eight prose
  strings on both sides (two of its keys, named after the build's work waves, now have plain names, `board_study_first`
  and `in_site_first`; `A5-text` records each old name by its sha256), no code reads, iterates over, serialises or
  hashes it, and no number appears in a new string that the old one does not hold except dates written out from run
  names and the numbers of the per-cell rule it now quotes; the shell scripts, the Dockerfile (whose instructions are
  identical) and `task.toml` (which parses to the same value) are equal once comment and blank lines are left out. (c)
  The twelve scored packages and the four practice packages were generated again from their recorded seeds, twice: with
  the three labelled files below at their earlier bytes and the earlier mission limit of 2400 s, every package is
  byte-identical to the stored one; with the amended files and the 1350 s limit, every package differs from the stored
  one only in the three labels, in `budgets.time_limit_s` of `site.json` and `prior.json` (2400 s at generation, 1350 s
  now; the training hall's package keeps its 900 s) and in its manifest's sha256 of those two members. The cell and the
  reference do not read the packages' `budgets.time_limit_s`: a mission's limit comes from the cell's configuration, and
  the program's mission document carries the enforced value. The stored packages are kept as they are. (d) An
  `ale validate` of the published folder, compared with the sealing validate of version 2.1, is in the pull request's
  description.
* **Provenance labels.** Every stored package names in its `site.json` three source files by the first 16 hex digits
  of their sha256 (`capability_source`, `observability_source`, `evidence_source`). Their comments, notes or prose
  changed here, so the stored packages keep the labels of the three files' earlier bytes; the packages themselves are
  not changed. Before and after this amendment:
  `review/sitegen/capability.json` 59338f8017d71357c0018947c2c1150066248e89c11b5ac6f4d0a3509879aafb ->
  660684d59438576df61c9dd724d56ae899fa9a56802c200d0639f99d5708053b; the acceptance file `acceptance_v1.yaml` (all
  three copies) 2ebe20b99a10af26697a7a1c9fdc6932784843e8827c2b0b5946d885022138fd ->
  f17f52c8994f7499d664ef4c8363d7dca4fbbb118a1110633f545de36e9f372d; `mission/evidence.py` (all three copies)
  c41a6d5b991930602f9b6d9f09fb7b84bb2565e85d137f804d15eee06566bba8 ->
  17809131b8fd28a281a8459b7551670e6bbd73c6d258fb3aeeb4c6b376384525.
* **Commit ids.** These notes and the review tools' docstrings no longer cite commits of the maintainers' own
  repository, which a reader here cannot resolve; where an identity helps, they give the engine's content digest of
  that version. The commit id that remains in the selection rule's registered wording (and in the rule constant of
  `review/select_set.py` that writes it) is of the maintainers' development history, not a commit of this repository.
* **What stays.** The agent-visible bytes. The selection rule's registered wording in `hidden/selection.json`
  (`rule`, `tables`, `exclusion`), with the ids, the paths of the maintainers' files and the development label it
  cites, and the rule constant of `review/select_set.py` that writes it. The stored exclusion list
  `review/sitegen/site/config/exclusion_v2.json`, pinned by its sha256 in the selection tool and the record. String
  constants of code: the generator's rejection messages that cite a decision id, argument help texts, the selection
  tool's and its test's messages, the bundle error that names the maintainers' bundle tool, the cell's version string
  `2.0.0-dev` (written into the cell's records). Data values that code reads: the generator's `defaults.yaml`
  (including a `source` string naming a study note; the generator hashes its whole configuration into each site),
  the vendored layout file's relative paths, and the robot model's generator comment (`plantcell/data/robot/robot.xml`,
  the model file the cell and the generator load). File names that code reads or writes (the source tree's optional
  `dev_paths.json`, run outputs). One test of the selection tool (`review/test_select_set.py`, its rule tests) falls
  back to a sibling development folder's copy of the exclusion list when the folder's own copy is missing or altered.
  The stored packages' labels and time-limit field (above). `hidden/seeds.json`'s note was cleaned here, but the
  selection tool's string that writes it was not: a re-run of the tool would bring the ids back. The shared kits
  (`cellkit/`, `robotics_grader/`, `environment/fetch_assets.py`), byte-identical to copies already in this
  repository. No machine path, account name or host name stays anywhere in the folder.

## 13. Amendment 3 (2026-11-18): a guard beside the BLAS kernel pin; text corrections

The grader's cell and the reference program inside it run with numpy's OpenBLAS held to its Haswell kernels
(`OPENBLAS_CORETYPE=Haswell` in `test.sh`; the practice service gets the same variables in
`environment/setup/setup.sh`), so that their linear algebra does not depend on the CPU. The OpenBLAS that numpy 2.5.2 to
2.5.5 bundle (this image has numpy 2.5.4) raises the inner-dimension block of its float64 matrix products from 256 to
512 on AMD hosts with AVX-512, 1 MB of L2 cache per core and an L3 cache of a multiple of 32 MB (most Zen 4 and Zen 5
desktop, workstation and server parts), for every kernel table, the forced Haswell one included. The Haswell kernel's
stack buffer can then overflow for float64 products whose result has 24 or more rows and 2 or more columns (in numpy's
row-major terms) and whose inner dimension is above about 300: the process crashes or, in a narrow band, returns wrong
numbers. On AMD hosts up to Zen 3 the block stays 256 (measured on a Zen 3 host, below), and on Intel hosts by the
library's start-up conditions (no Intel host was measured). The fault is known upstream (OpenMathLib/OpenBLAS issues
6021 and 6026) and fixed in OpenBLAS's sources, but no numpy release up to 2.5.5 (2026-11-15, the latest on 2026-11-18)
bundles a fixed library. `OPENBLAS_L2_SIZE=256` beside the pin switches the raise off, so the block is 256 on every
host; with a fixed library the variable changes nothing. Neither the cell nor the reference makes a product in the crash
range, and a submitted program cannot bring the cell to one (the evidence is below): the variable is a guard.

* **Where.** `tests/test.sh`: the variable on the line that exports the pin, and three comment lines that say why. Their
  "changes no result of this cell" speaks of outcomes: for a submitted program no product of the cell reaches the block
  (below); with the reference inside the cell, the last bits of its long covariances can depend on the block on hosts
  that raise it, and no score, final state or admission verdict changed in any run (below).
  `environment/setup/setup.sh`: the same variable in the practice service's environment, and one comment line. The
  program under test never carried the pin and does not now: it runs in its own minimal environment
  (`cellkit.runner.child_env`), and the agent's shell has no kernel pin (of OpenBLAS's variables the image's ENV sets
  only the thread count). `review/cell/determinism.py` records the variable beside `OPENBLAS_CORETYPE` in its output
  (one entry added to a dictionary of a maintainers' tool that no module imports).
* **The library** (an image with the task image's Python and packages, numpy 2.5.4): on an AMD Threadripper PRO 9965WX
  (Zen 5) the active Haswell table holds the double-precision blocks P / Q (row block / inner-dimension block) 512 / 512
  with the pin alone, and a float64 700 x 700 product crashes in 5 of 5 fresh processes, at one thread and at the
  grader's two; with the variable the table holds 512 / 256 and the product runs in 5 of 5. On an AMD Threadripper PRO
  5975WX (Zen 3) the table holds 512 / 256 either way and nothing crashes.
* **What the cell asks of it.** A tracer of every double-precision level-3 BLAS call (products, syrk, trsm, trmm, syr2k,
  symm) that numpy makes into its OpenBLAS, on practice building t1: in a mission of the measured episode's submission,
  the cell's process made 1,690,073 calls, none with an inner dimension above 4; in a mission of the reference,
  1,225,636 calls: covariances of two-column point sets (a 2 x 2 result; inner dimension up to 13,309, above 256 in 11
  calls) and otherwise no call with an inner dimension above 4. A 2 x 2 result is outside the overflow's condition at
  any inner dimension, so no product of these missions comes near the crash. By the code, a submitted program cannot
  change this: in its missions every matrix-matrix product (level-3 BLAS call) and decomposition the cell computes (the
  mission, its scoring, the debrief, the records) has an inner dimension of at most 4 and an order of at most 4, set by
  the code, the cell's configuration and the stored building; a program sets only how often they run. The gait policy's
  layers, up to 512 wide, are vector-matrix products (level 2), which do not use the block (the same bytes with and
  without the variable). The reference has run on every candidate building on a host that raises the block, with the pin
  alone, without a crash (admission A4). On the hosts that raise the block, its size can still change the last bits of
  the reference's long covariances (it does for random point sets of these lengths); the selection's rule asks for the
  admission again after any change to a frozen file that is not text-only, and it checks the outcome directly:
* **The admission again** (`hidden/selection.json`, `amendments` entry `A6`): on these bytes, in an image built from
  this `environment/` with the render bundle from the Hub, on the AMD Threadripper PRO 9965WX where admission A4 ran:
  the same pre-registered rule records, the same 48 hidden and 12 practice candidates with their seeds, the reference
  and the five row-4 programs on every candidate that builds. All 276 hidden and 66 practice runs (46 and 11 of them the
  reference's) end as A4's, with the same score, final state, number of attempts and simulated time; the same 40
  candidates admitted, the same twelve scored buildings and the same three practice buildings (the practice packages the
  tool lays out are byte-identical to this folder's); the hidden set the tool writes (packages and seeds) equals
  `hidden/`; the route-health check finds no defect in the 57 reference runs. So with the variable no score, final state
  or admission verdict changed in any of these runs, and the reference value stays.
* **The same on another CPU.** The reference with the variable on the four practice packages (training hall, t1, t2, t3)
  and on two of the scored packages, on the Zen 3 host and on the Zen 5 host: the same final state, score, simulated
  time, end and dose in each of the 6 missions. The grading machine of sections 6 and 10 is an AMD Zen 4 host and the
  admission's machine a Zen 5 host, both of the class that raises the block: the reference values, the validates and the
  bit-for-bit agreements between those two machines that these sections report were measured with the raised block on
  both and without the variable; this check is the first in these notes between a host of that class and a host outside
  it.
* **A validate and a re-grade.** An `ale validate` of this amended folder, compared with the two earlier validates of
  version 2.1, and a re-grade of the measured episode's kept submission on these bytes are in the pull request's
  description.
* **Text corrections.** The header of `task.toml` (the oracle's program is not two lines: it takes one frame of every
  sensor, then makes one privileged call); a comment of the generator's `defaults.yaml` (walk was measured on cable
  protectors and is not refused by them reliably); line breaks only, no word changed: the module docstring of
  `plantcell/wire_guard.py` (and its practice-ground copy) and five lines over 120 columns in
  `environment/stage_assets.py`, `verify.py` and the generator's `furnish.py`, `hall.py`, `ladder.py`; in these notes:
  the header's pointers to sections 12 and 13 and how to validate (top), the files table's row of `test.sh`, three
  sentences that pointed to records not in this folder (two of section 8: the pre-registration of version 2.1's rule and
  version 1's baselines; one of section 11: its records), and in section 12 what its passes also corrected, that
  `anchor.json`'s `source` is shown by the repository's registry, and its proofs item re-wrapped; in
  `hidden/selection.json`, the same clause in A5-text's `what`.
* **What did not change.** No byte the agent can read (`instruction.md`, `environment/setup/payload/`, its mirror
  `setup_payload/`, `given_hashes.json`, the client library); no other code, data value or configuration value; the site
  packages, the seeds, the selection and its registered rule, the reference value in `anchor.json`, the version (2.1).
  Proofs as in section 12, made before any file was replaced: every changed Python file but `determinism.py` has the
  same syntax tree once its docstrings are left out and the same string constants (for the generator's three files,
  whose comments were re-wrapped, the comments' words are the same in the same order); `defaults.yaml` parses to the
  same value; `task.toml` is equal once comment and blank lines are left out and parses to the same value; `test.sh` and
  `setup.sh` are equal once comment and blank lines are left out except for the one line that gains the variable.
* **The record.** `hidden/selection.json`, entry `A6` (kind `environment-variable`): the three frozen files that changed
  (`test.sh`, `verify.py`, `plantcell/wire_guard.py`) with their sha256 before and after and the kind of each change,
  the recorded environment's one added variable, the admission with its comparison with A4, the check on the other CPU,
  and A5-text's clause by sha256. The table `frozen` and the record `environment` stay the ones admission A4 wrote (A6's
  `environment_changed` gives the added variable and both digests); a file that A5-text and A6 both list is followed
  through the two lists in order.
