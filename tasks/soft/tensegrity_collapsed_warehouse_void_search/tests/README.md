# tensegrity_collapsed_warehouse_void_search: maintainers' notes

**Version 2.0.** Versions 1.0 (2026-11-17) and 1.1 (2026-11-19) came before it; their files and records are kept by
the maintainers. Platform `soft`, stage `behavior`. Hidden side: this folder is never staged into
the agent's sandbox, so names that the agent must not see may appear here; the attribution is in `NOTICE.md`.
The measurements behind these notes, and the task's design, are kept by the maintainers and are not
part of this repository.

**How these notes state numbers.** A number printed here is either a constant of the task (the notes name the file
that holds it) or a property of the frozen bytes that anyone can compute again (a file's sha256, a count, the
reference's simulated time on a site). What changes from run to run (wall clock, host load, an image's id, a
validation's id, the task's content digest, the dates of runs) and what only a measurement gives (the anchor's value,
the oracle's wall clock, the QC's results, the cost figures, the verifier's timeout) is not printed here: the notes say
where it is recorded (`anchor.json`, a field of `task.toml`, the pull request that adds this version) or that the
maintainers keep it (the results of the QC runs, the oracle's wall clock, the measured costs).

**What changed from version 1.1.** One model episode on version 1.1 read 0.23. Version 2.0 keeps the robot, the
primitives, the cable controller, the station's calls and replies, the telemetry, the camera's intrinsics and depth
model, every budget but the time limit, the scoring (weights, credit radii, dwell rule, retrieval rule), the
time-limit factor 1.6, the contract, the session, the practice allowance, the debrief and everything under the slab
(the three mouths, the void, the rubble boxes in its passages). It changes three things:

1. **Heaps on the open floor** (site family W3 -> W4). The grey free blocks of the open floor are gone; 3 to 6 heaps
   stand there instead, each a rubble heap or a mound of fill, each against the site wall, the ridge or the slab's
   front edge. A heap is an obstacle: the robot cannot get over one. Its collision body is a height field made from
   the same mesh that the camera's picture shows (section 8).
2. **The camera's picture.** The frames no longer come from MuJoCo's classic renderer on plain boxes. Each site has a
   plate: the dressed site without the robot, computed offline, with its depth; the robot and its shadow are rendered
   into it at every frame (render backend `plate_filament`; below, and sections 2 and 4). The dressing changes
   nothing the robot touches: the collision world is the site package's boxes and height fields.
3. **The time limit's rounding.** Steps of 10 s instead of 60 s: 1800 s -> 1820 s (calibration site: 1440 s -> 1400 s).

The hidden set, the practice sites and the calibration site are new (the calibration site gained one rubble heap and
one mound of fill), the agent's documents were rewritten where they speak of heaps, light and litter, and the anchor
was measured again (section 6).

**What the robot is made of.** The robot in the simulation and in the camera frames is the designers' published
whole-strut mesh, three times (a decimated copy for the agent camera, section 2); its mass and inertia are the
designers' simulation file's, its collision bodies are spheres, cylinders and a convex hull fitted to that mesh, and
its cables and stretch sensors are the 15 tendons of the designers' simulation model (lengths between two attachment
points, with no shape). Nothing of the robot was modelled, redrawn or completed, and there is no hidden variant of it:
every site has the same robot. The heaps and the dressing of a site are scene objects, not parts of the robot.

**What the camera's picture is made of.** A site package holds the collision world only: boxes, and one height field
for each heap. A scene builder dresses that world with pieces of the maintainers' art packs (walls, racking, the slab,
rubble, crates, the heaps' own meshes, the loose-ground patches, thin flat litter), a path tracer computes the search
camera's view of it once, and the result is stored as the site's plate bundle: the picture, its scene-linear
radiance, a depth plate at the camera's resolution and one at 4 times it, the sun's share and visibility, the fog,
light probes and the sun (`voidcell/_lib/plate_bundle.py` names the files and checks them whenever a camera opens).
At run time the backend renders only the robot (MuJoCo's physically based renderer, on the CPU), lights it from the
bundle, composes it and its shadow into the plate, and returns the nearer of the plate's depth and the robot's depth;
the camera's depth sensor model is the same as before. A pixel that the robot and its shadow leave alone is the
plate's byte. One look (`voidcell/_lib/agent_look.json`) serves every site. The bundles are not in this repository
(section 2). The art packs, the tools and the renderers are named in `NOTICE.md` and nowhere else in the task.

## 1. The task

A rescue team has lowered a mast with a fixed RGB-D search camera through the roof of a collapsed warehouse; the
camera looks down on the south 4.4 x 4.0 m of a 4.4 x 5.8 m site; the north part lies under a fallen slab. A 0.33 m,
0.73 kg three-strut tensegrity is set down on a pad. It carries no camera and no computer: a program on a PC outside
reads the camera's raw frames and the robot's telemetry and sends rolling primitives and cable-length targets, as in
the designers' own experiments. The agent writes that program (`operate.py`, `solve(station)`). A mission gives three
search points; at each the robot must stand still for 20 s and report. Points 1 and 2 lie in the camera's view (point
2 behind a ridge of rubble). Point 3 lies in a chamber under the slab, in one of the site's back corners, out of the
camera's view: three mouths that look the same open under the slab's front edge; one leads in, the other two are
plugged a short way inside. Behind the open mouth a straight passage with a dead end, a connecting passage and, on
some sites, a back passage with a second dead end lead to the chamber; the way from the last pose the camera sees to
point 3 is 4.0 to 6.0 m long by rule, with at least two right-angle bends, and one or two blocks of fallen material
stand against one wall of each straight passage, leaving beside them a gap as wide as the mouths. On the open floor
stand 3 to 6 heaps, which the camera sees and the robot has to go around. Then the robot is to be brought back to
the pad. The grader runs the program on ten hidden sites, each within 1820 s of simulated time.

## 2. Files

| path | what |
|---|---|
| `test.sh` | closes the stage and its parent (chmod 700) before Python starts; builds the root-owned givens at `/opt/arb` from `setup_payload/`; points `ARB_HIDDEN_PLATES_TARBALL` at `assets/plates.tar.xz` when that local copy exists; pins HOME, the user site, OSMesa (no shader cache), every thread count (8 rasteriser threads, OpenMP 1, BLAS 2) and the numerical kernels (`OPENBLAS_CORETYPE=Haswell`, `OPENBLAS_L2_SIZE=256`, `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`); `exec "${ALE_VERIFY_PYTHON:-python3}" -s verify.py` |
| `verify.py` | the grader (section 4) |
| `grader_config.json` | run settings only: `episodes` 10, the cell's budgets (600 frames, cable targets 100-240 mm, program time 600 s, silence 180 s, CPU 2400 s), `pose_source` (`estimated`), `render_backend` (`plate_filament`), `plate_root` (`hidden/plates`, a folder of the staged copy of this folder), `hidden_plates` (the URL of the hidden sites' plates tarball at a commit of the asset dataset, its sha256 and its file count), the privileged cap (3600 s), the route gate. No scoring threshold and no time limit |
| `anchor.json` | metric `void_search_score` (higher); `value` = the reference's measured mean on the hidden set (two runs) and `full_at` are recorded there and nowhere else |
| `voidscore/` | `acceptance.yaml` (version 2), the single source of every scoring threshold (weights, credit radii, dwell window and still tolerance, reports per point, retrieval radius and factors, the debrief's "stuck" rule); `scoring.py`; `debrief.py` (the practice ground's debrief; never imported by `verify.py`) |
| `voidcell/` | the cell: `server.py` (VoidCellServer, the station's wire side), `client.py` (byte-identical to the agent's `given/lib/voidcell/client.py`), `wire_guard.py`, `_lib/` (the vendored simulation base `ysim` 0.3.1 with its lock `ysim.lock.json` and data tables, the site loader, the executor of primitives and cable targets, the truth record, the camera with its two backends, the telemetry, the budgets, `frames.md`; the plate backend: `plate_filament.py`, `plate_bundle.py`, `plate_look.py`, `plate_robot.py`, `plate_shadow.py` and the look `agent_look.json`; `plate_install.py`, which puts plate bundles where the backend reads them), `data/meshes/` (the robot's visual meshes for the agent camera: the designers' strut mesh regrouped by material; four groups at full resolution, the group of small electronics decimated by 94 %; `MESHES.json` records tool, parameters and sha256) |
| `reference/` | the privileged reference program, voidref 1.1.1 (`reference.py`, run by the cell's `reference_run`) and its modules (`_lib/voidref/`: planner, controller, monitor, its primitive table); the same program and table values as in version 1.1 (one docstring line of `reference.py` and the table's `source` note were reworded) |
| `hidden/` | `sites/h01..h10.tar.xz` (the stored site packages of family W4: the heaps and the rubble under the slab included), `seeds.json` (the ten sites in episode order: archive, sha256, episode seed, the reference's final state, time and primitive count), `selection.json` (the admission and selection rule's hash, pin time and full text, every candidate's value of every admission row, the sets, the set-level checks, the cue checks), `limits.json` (`time_limit_s` 1820 and its rule, the thirteen reference times, `calibration_time_limit_s` 1400), `oracle_token.sha256`. `plates/` exists only in the stage: `verify.py` installs the hidden sites' plate bundles there (section 4) |
| `assets/` | not in git: a maintainer's local copy `plates.tar.xz` of the hidden sites' plates tarball; when it is there, `test.sh` hands it to `verify.py` in place of the download (below) |
| `setup_payload/`, `given_hashes.json` | the grader's byte copy of `environment/setup/payload/` (11 files) and the sha256 of each |
| `cellkit/`, `robotics_grader/` | the verify-stage process kit (lockdown, runner as `nobody`, no-network seccomp filter, network audit, the JSON-lines wire with program-time accounting, the freezer, pinned seeds), byte-identical to `quadruped_hazardous_plant_inspection`'s copy; the benchmark's verdict kit, byte-identical to `shared/robotics_grader/` |
| `review/` | maintainers' tools that nothing runs at verify (its own `README.md` and `SHA256SUMS`): the site generators `w2gen` (the open ground), `w3gen` (the slab, the three mouths, the void and the rubble) and `w4gen` (the heaps), the calibration site's generator `calgen` 0.4.0, the QC scripts, the degenerate submissions, the maintainer variants of the reference with their expected scores. The heap library and the selection, packaging, baking and oracle-verify tools are not here: the maintainers keep them and they are not published (section 8 describes what they did); the admission rule's hash, pin time and full text are in `hidden/selection.json` |
| `README.md`, `NOTICE.md` | these notes; attribution |

Outside this folder: `solution/` (the oracle: a short `operate.py` and the 256-bit token), `environment/` (the
image; the payload; `practice_ground/` = the practice service `practiced.py`, a byte copy of `voidcell/` and
`voidscore/` without the reference and the token, the calibration site `cal`, the practice sites `p1..p3`;
`assets.sha256` and `fetch_assets.py`, the list and the fetcher of the four plate bundles the image needs).

**Where the plates come from.** The plate bundles are large binary files and are not in git. Each is bound to one
site package: its `meta.json` records the package's sha256, and the backend refuses a bundle whose record, camera or
files do not match the episode's package. They travel under neutral folder names, because the name the backend looks
for is the package's sha256, which for a hidden site is a hidden value:

* the practice ground's four bundles (`cal`, `p1`, `p2`, `p3`) are files of the benchmark's asset dataset under this
  task's `payload/` prefix, listed with their sha256 in `environment/assets.sha256`; the image build fetches them,
  installs them into the practice service's own folder, readable by its account only, and removes the list and the
  fetched copy; nothing of them reaches the agent's home;
* the hidden sites' ten bundles (`h01` ... `h10`) are one tarball, `plates.tar.xz`, under the task's `verifier/`
  prefix of the same dataset, pinned in `grader_config.json` (`hidden_plates`: URL, sha256, file count); `verify.py`
  fetches it in the verifier's sandbox (section 4).

Both sides install with the same code (`voidcell/_lib/plate_install.py`): each neutral folder is renamed to the
sha256 its own `meta.json` records, after a check that this is the sha256 the site of that name is pinned to
(`hidden/seeds.json`; `environment/practice_ground/sites.json`). The files are on the dataset at the commit that
`environment/assets.sha256` and `grader_config.json` pin (`NOTICE.md`, Distribution). A local copy of the tarball
at `assets/plates.tar.xz` (ignored by git) replaces the download when it exists: `test.sh` hands it to `verify.py`,
which checks it exactly as a download.

## 3. Scoring

Per site, from the cell's own truth record (`voidscore/scoring.py`, every threshold from `voidscore/acceptance.yaml`
version 2): S = F x (0.05 c1 + 0.15 c2 + 0.80 c3).

* Position = the planar centroid of the six nodes (the end-cap centres).
* A report is a `report_point(i)` the station accepts (at most 3 per point); only the last accepted report of a point
  counts, also when an earlier one was better.
* c = dwell x distance. Dwell is 1 when the report comes at least 20 s after the episode's start, no command (a
  primitive or a `set_targets`, from acceptance to end) was executing at any moment of the 20 s before it, and every
  10 Hz truth sample of that window lies within 0.02 m of the position at the report. Distance is 1 within 0.20 m of
  the search point, linear to 0 at 0.40 m. The 0.20 m is a setting this task adds, not the designers' figure: an air
  reading is taken where the sampler stands, so the robot has to be brought to the spot in the void, not near it (the
  designers counted a stop within one body length, 0.33 m, as reaching a goal; version 1.0 used that radius).
* F = 1.0 when, at the instant the episode ends, the position is within 0.60 m of the pad centre and no command is
  executing; otherwise 0.85.
* No negative marks; the robot cannot be damaged; touching rubble or a heap costs nothing.

`void_search_score` = `contract_ok` x the mean of S over the ten sites; reward = clamp(score / `value` / `full_at`,
0, 1) with the two numbers of `anchor.json`. The two visible points and the retrieval alone give 0.20; all three
points without the retrieval 0.85. On every selected site no pose that touches a wall or a block and no pose the
camera sees earns anything of point 3 (predicate G8a, one of the admission's G rows): point 3 stands at least 0.62 m
from every solid and every site edge by rule.

The cell announces six of the constants in `mission()` and holds its own copies (`voidcell/_lib/mission.py`,
`budgets.py`); every episode checks them against `acceptance.yaml` and a difference is a task error.

## 4. The verify stage (verify.py)

The stages and the kit of `quadruped_hazardous_plant_inspection`'s grader, with this task's cell, program and score:

0. `_early_lockdown` (stdlib only, first): the stage upload time; kill and freeze every process of the agent account
   until the verdict; lock the stage; a root-owned snapshot of the submission (regular files only); late writes into
   the submission zero the contract gate, late entries elsewhere in the home and the sticky folders are quarantined.
1. The givens at `/opt/arb` against `given_hashes.json`, and the cell's announced constants against
   `acceptance.yaml`: a mismatch is a task error. Then, with `hidden_plates` in `grader_config.json`, the hidden
   sites' plates (`fetch_hidden_plates`): the tarball is downloaded (at most 4 attempts and 900 s in all) or, when
   `ARB_HIDDEN_PLATES_TARBALL` is set, copied from that file; its sha256 and its file count must be the pinned
   ones; it may hold nothing but regular files under the neutral folders `h01` ... `h10`, each name once (a link, a
   folder entry, a device, an absolute or dotted path is refused before anything is unpacked); it is unpacked into a
   temporary folder of the root-only stage and installed into `plate_root`, each folder renamed to the site package
   sha256 its `meta.json` records, which must be the sha256 `hidden/seeds.json` pins for the site of that name. A
   `plate_root` that exists and holds anything but installed bundles is left untouched. Every failure here is a task
   error; the temporary folder and the tarball's copy are removed either way. The download happens before the
   desktop settles and outside every episode's network count.
2. Gates (`contract_ok`): `operate.py` is a regular file that parses and defines `solve`; the route gate (60 MB, no
   weight files, no training calls); every `.py` file readable by the parser; the static network audit (no networking,
   process or foreign-function import, no computed import or code, no program start; files byte-identical to the
   shipped client exempt); the import check as `nobody` under the no-network filter (120 s wall, 120 CPU-s); over the
   episodes at least one `step` call, no late write, no surviving process, no remote inet socket.
3. The oracle token: `.oracle_token` whose sha256 equals `hidden/oracle_token.sha256` makes the cells privileged (the
   call `reference_run`); a wrong token is ignored and recorded (`oracle_token_mismatch`).
4. The hidden set: `hidden/seeds.json` must pin `episodes` sites; every archive must be a regular file under
   `hidden/` whose sha256 matches (else a task error, also for an untouched submission). The simulated-time limit
   comes from `hidden/limits.json`, every other budget from `grader_config.json`. With a `plate_root`, every pinned
   site must have its bundle there before the first episode (`plates_for_sites`).
5. Before the first episode: the frozen desktop settles, the no-network filter proves itself.
6. Per site, under an opaque tag `epNN`: a fresh `VoidCellServer` on the package (mode hidden; the robot set down
   with the pinned seed; the search camera opens the site's plate bundle and checks it file by file: a missing,
   foreign or damaged bundle is a task error, and there is no fall-back to the classic image); the program as `nobody`
   from a read-only copy of the snapshot, `python -B -E -s`, the client
   library from `/opt/arb`, `prlimit --cpu`, `oom_score_adj` 1000, the seccomp no-network filter, its account's
   remote sockets sampled, and the same numerical kernels as the cell (`BLAS_PIN`: `OPENBLAS_CORETYPE=Haswell`,
   `OPENBLAS_L2_SIZE=256`, the NumPy SIMD cap; the practice ground's `~/training/run_mission.py` sets the same three;
   the verdict records that the pin held); the wire until the program finishes, calls `finish()`, reaches the
   simulated-time limit, falls silent (180 s), exceeds its program time (600 s of its own time: the wall clock minus
   the cell's simulation and rendering) or breaks the protocol. Whatever ends the episode, the reports accepted until
   then are scored and the retrieval is judged at that instant. Every process of the runner account is killed after
   each episode, which is why the episodes run one after another.
7. An episode in which the cell failed (an internal error while the episode was live, a frame it could not render)
   is run once more from scratch and the second run counts; at most one per verify (`MAX_RERUNS_PER_VERIFY`), a second
   one is a task error. A failure on the grader's side (the cell's build, the launch, the wire, the clean-up, a score
   that is not a finite number in [0, 1]) is a task error; a program that crashes or hangs ends its own episode.
8. The verdict through `robotics_grader`; records under the root-only `/opt/arb-verify` (`out/episodes.json`,
   `out/traj_<tag>.npz` with the 10 Hz truth samples, `logs/<tag>.log` with the program's stderr).

The kernel pin exists because OpenBLAS picks its kernels per CPU and NumPy its SIMD paths: without it the program's
numbers could differ between hosts, and the Haswell kernel alone, without the L2 size, crashes or returns wrong
products on Zen 4 / Zen 5 CPUs with numpy 2.5. A scored episode has no debrief: `verify.py` imports nothing of
`voidscore/debrief.py`, the cell writes none, and the reply to a report says only that it was accepted. Untouched
sandbox: no `operate.py`, `contract_ok` 0, reward 0.

The frames are rendered on the CPU in both sandboxes: the robot's depth and mask by MuJoCo's classic renderer on
OSMesa, the robot's colour by MuJoCo's physically based renderer on Mesa's software Vulkan driver; the backend drops
the display variables before that renderer starts. A tool that starts `verify.py` directly (the QC's `qc_verify.py`,
`review/degenerate/run_verify.py`) does not pass through `test.sh` and has to set `ARB_HIDDEN_PLATES_TARBALL`
itself while the local copy is in use.

## 5. Realism audit

What in the task's world is real (the designers' robot and experiments) and what this task adds. "Base assumption"
marks a value of the simulation base that no public measurement of the real robot fixes.

| in the task's world | real or added | source, or why it was added |
|---|---|---|
| The robot's body, 15 tendons (6 motor-wound lines, 6 short and 3 long stretch sensors), 6 gearmotors, one 6-axis IMU per strut | real | the designers' mesh, simulation file, bill of materials and firmware |
| No on-board camera or computer; planning, the cable-length PID and pose estimation on an off-board PC; a radio link | real | the designers' paper and control code |
| One fixed overhead RGB-D camera, 640 x 480, 7 Hz, calibrated; red, green and blue end caps, both ends of a strut the same colour | real | the paper; the intrinsics and the depth unit are those of the designers' tracking configuration |
| Eleven primitives, target shapes (six cable lengths), re-planning after each primitive, the symmetry table by the bottom triple | real | the paper and the control code |
| Navigation among obstacles; success = stopping within one body length of the goal | real | the paper. This task's credit is tighter and added: full within 0.20 m, none beyond 0.40 m (the sampler has to stand at the spot) |
| After a bad primitive the robot can end in an orientation from which it cannot roll on | real | the paper; in the simulation these are the two end triangles (16 of 7600 primitives among obstacles, 0 of 4640 on a free floor; the handbook prints the counts) |
| How far each primitive moves the robot, the stop band of 5 % (the designers' code: 15 %), which end-cap hook carries which cable, the line's stiffness, the 0.911 s rest between primitives, the IMU's axes | base assumption | no calibration data of the real robot is public. The handbook's primitive table is this simulated robot's own measurement on a flat floor, by start face, not the designers' published table (against that table: forward rolls 0.04-0.10 m off in translation, the clockwise turn qualitatively different) |
| A narrow passage and a low beam | the designers' simulation scenes, not the real robot | only their sizes are borrowed (the tight opening class T and the low slot XL, 0.15 m clear) |
| The scene: a collapsed warehouse after an earthquake | added | the maintainers' choice of setting |
| The camera on a mast through the roof, 4.95-5.25 m up, tilted 14-17 degrees (the lab: 2.34 m, straight down) | added | it has to cover the open part of the site and see the mouths |
| The program gets raw RGB-D only and estimates the pose itself (switch `pose_source`, default `estimated`) | added | in the real system a tracker supplies the pose; with a given pose the task would be the designers' published planner |
| The crawl space out of view: three mouths that look the same (one open), straight passages with dead ends meeting at right angles, a chamber in a back corner whose two walls are the site's own, a blind way of 4-6 m | added | the real experiments ran in view throughout; the paper names navigation on on-board sensing alone as open |
| Fallen material against one wall of a passage under the slab (1-2 blocks per straight passage, 0.20-0.30 m long, floor to slab; the gap beside each 0.559 m, the mouths' width) | added | a passage in a collapsed building is not clear; a program that rolls down the middle of a passage without knowing the site meets a block (by the family's rule at least one block on the way in reaches 0.04 m or more into the centre band of the widest roll); the reference passes every block without contact |
| Heaps on the open floor: rubble heaps (bricks and lumps of broken concrete) and mounds of fill (dumped earth), 0.14 to 0.30 m high at the top, each against a wall, the ridge or the slab's front edge; the robot cannot get over one | added | the floor of a collapsed building is not clear either. In the heap library's trial runs the simulated robot never got onto or across a heap (540 runs; the handbook prints the counts). A heap is seen by the camera and passed by going around it, which is the designers' own way among obstacles |
| The dressed picture: a ruin of real scanned and modelled pieces, daylight through holes in the roof with sun patches and hard shadows, dust in the air, thin flat litter | added | with plain boxes and no texture, finding the robot in the picture was the easy part of the task. The dressing is picture only: nothing of it is a collision body |
| A search point = a 20 s dwell for one air reading (a gas sensor of a few grams on the board's expansion port; not modelled) | added | it gives a search point a meaning; only the dwell and the place are judged |
| 8 to 14 flat plastic chips in the end caps' colours on the floor; another light at every site | added | a single colour threshold would otherwise find the end caps; the paper reports a coloured surface disturbing the tracker |
| The radio holds everywhere (under the slab too); the telemetry's and the depth's noise | added (a simplification; rated values) | no public measurements; the documents call them rated values of this unit |
| The simulated-time limit (1820 s) as the team's working window; no energy term | added | endurance is not published; ten minutes of mixed primitives use about 6 % of the motor battery in the simulation |
| Loose ground = flush low-friction patches (dust 0.35, gravel 0.60); no relief, no granular medium; a level floor without steps or slopes | added (a simplification) | granular simulation is slow and unstable; this simulated robot climbs neither a 2 cm step nor a 4 degree slope, so a site has none |
| Lowered on a rope and hooked for retrieval; set down on a rollable face | added | no primitive starts from an end triangle |
| The floor friction of a site, drawn from 0.76-0.92 | range real, the draw added | the range of the designers' identified ground friction; its effect on a primitive is at most 3.3 mm and 1.9 degrees |
| Rolling backwards | not offered | the paper does not mention it and the designers' code says it was never run |

Given to the agent: the handbook (neutral model name RT-3; dimensions, numbering, the cable controller, the eleven
primitives with their target states and the symmetry table, the flat-floor table of every primitive from every
rollable face, what is not guaranteed, what a roll against a heap does, the telemetry with its noise, the IMU frames;
three purchased parts under their real vendors and part numbers; no picture of the robot), the station's API, the
site briefing (the site rectangle and its walls, the slab from wall to wall, three like-looking mouths of which one
leads in, the passages' shape in words, the chamber in a back corner with nothing within 0.6 m of point 3, the heaps'
two kinds, their size range, where they can stand and what is kept free of them, the light and the litter), the
camera's data sheet, the calibration site's pages (every dimension, among them a roofed passage narrowed by a block,
a roofed chamber in a corner, one rubble heap and one mound of fill; nowhere whether an opening can be passed), the
client library, an example that runs and scores 0, and the practice ground (a black-box service under its own
account: the calibration site and three practice sites, 30 episodes, two at a time, a debrief after each).

Not given: the simulator's code, the site generators and their ranges, the heap library, the grading code, any scored
site's layout, where its heaps stand, which openings the robot can pass, which mouth is the open one, the passages'
widths, lengths and turns, where the blocks stand, the robot's motion on loose ground, the true pose, any score of a
scored site, and every name of the robot's designers, their schools, labs, papers, design and repositories, of an art
pack and of a renderer. Left for the agent to measure on the practice ground (the maintainers' disclosure
check): the roll's swept width and peak height; that floor friction hardly matters; the dust patch's numbers; that
the robot gets out of a plugged mouth again; the clearance from walls and the footprint of a turn; how far the table
drifts when nothing corrects it; what "stuck" looks like in the telemetry. No document says that the surface the
depth image shows on a heap is the surface the robot touches (section 11, item 4).

## 6. The oracle and the anchor

`solution/solve.sh` installs `operate.py` (the mission, one frame, the telemetry and the status, then
`reference_run`) and the token. The reference program runs inside the cell with the privileged truth: the true node
positions (the pose, and the bottom triple it passes to each primitive) and the site's true occupancy (every box,
which mouth is open, the plugs, the rubble). It plans with a hybrid A* over the robot's own flat-floor primitive
table, re-plans after every primitive from the true pose, and sends every command through the station's own calls
with the same argument checks, budgets and time limit as an agent's program. It takes no camera frame for its
decisions and reads no telemetry, so its result does not depend on the renderer. Version 1.1.1 of the reference is
1.1.0 with comments and docstrings changed; its runs end bit for bit in the same final states.

The reference's program and table values are unchanged in version 2.0. A heap is a solid in the cell's truth occupancy (its placed rectangle from
the floor to its top), so the planner goes around heaps as it goes around boxes; on every selected site the reference
touches neither a heap nor a rubble box under the slab (admission row R8, section 8).

The anchor (`anchor.json`): `value` is the mean `void_search_score` of two full runs of `verify.py` with the oracle
submission on the ten frozen sites, with the plate backend on, outside a container and not as root (no lockdown and
no account switch; the givens copy built as `test.sh` builds it; `test.sh`'s numerics); the two runs must give the
same ten final states, and these must be the final states of the admission runs (`hidden/seeds.json`
`reference_final_state_sha256`). The value, `full_at` and the pins of the files the value was measured on are in
`anchor.json`; each site's simulated time is in `hidden/limits.json`; the runs' wall clock and the host's load are
kept by the maintainers and are not published.

The validations in the task image (untouched submission, oracle) are named, with the run's id, the image and the
task's content digest, in the pull request that adds this version: all of them depend on these very files, so they
cannot be printed in them.

## 7. Timeouts

Agent: `[agent] timeout_sec` of `task.toml`, 14 400 s (4 h), no network. Verifier: `[verifier] timeout_sec` of
`task.toml`. Its value is not repeated here; it follows from the formula in `task.toml`'s header,

    timeout = safety factor x (fixed steps + (episodes + re-runs) x bound of one site)

whose measured inputs are taken in containers of the task image with the verifier's CPUs and memory and are kept,
with the image they were measured in and the host's load, by the maintainers:

* fixed steps: the import check (at most `IMPORT_WALL_S` of `verify.py`), the no-network self-test, the desktop
  settle, the lockdown, the givens and the acceptance check, and fetching, checking and unpacking the hidden sites'
  plates;
* the bound of one site, at the limits of the configuration: the cell's build and first frame (the plate bundle is
  opened and checked there); the program's `program_time_s`; `time_limit_s` simulated seconds at the slowest
  whole-episode pace measured (wall clock per simulated second, a primitive running, the wire included); `frames`
  frames at the measured cost of `get_frame`; the wire's grace after the episode and the end, scoring and clean-up;
* episodes + re-runs: `episodes` of `grader_config.json` plus `MAX_RERUNS_PER_VERIFY` of `verify.py` (10 + 1).

The documents' cost paragraphs (written by the maintainers' document generator, not published) print figures of the
same measured costs. A program that spends every budget on every site holds the verifier for hours, and the episodes
run one after another.

## 8. The hidden set and the measured programs

**Generator.** The site family W4 (`review/sitegen/w4gen`, version 0.3.0) is W3, the family of version 1.1
(`review/sitegen/w3gen`, imported unchanged; it takes the open ground from W2, `review/sitegen/w2gen`), with heaps on
the open floor. It builds a site from a seed: `site.json` (every coordinate and truth), `site.xml` (the world only)
and the generator's own QC (`qc.json`, not part of a frozen package). Everything the camera or `mission()` can show
is drawn first and does not depend on which mouth is open; then a void is built for each of the three mouths taken as
the open one, and the open mouth is drawn last (`default_rng([seed, 12])`), so given everything visible the open mouth
is equally likely to be any of the three. Of W2's free blocks only the red and the blue crate stay. The heaps come
from a visible stream of their own (`default_rng([seed, 14])`) and are placed after everything of W3's visible stage
is fixed, against the union of the solids of the three counterfactual sites, so they too are the same whichever mouth
is open. The rubble under the slab is W3's, unchanged: the void's straight passages get 1-2 blocks each from their
own stream (`default_rng([seed, 13])`; the rule is in `review/sitegen/w3gen/rubble.py`), each against one wall,
0.20-0.30 m long, floor to slab, the gap beside it the mouths' width, not in the chamber, a dead end or a junction,
at least 0.5 m from the mouth's inner end and every corner, at least 0.8 m apart along a passage, at least 0.62 m
from point 3. Sizes come from the simulation base's measured capabilities (not published): class P 0.559 m wide (0.387 m
clear in the first ridge), T 0.459 m, the narrow decoy XN 0.257 m, the low decoy XL 0.559 m wide and 0.15 m clear;
under the slab the mouths 0.559 m wide, every passage and the chamber 0.407 m clear, the passages 0.779-0.919 m wide.

**Heaps.** Two classes, both obstacles: R, a rubble heap (placed rectangle 0.28-0.46 m wide and 0.22-0.34 m deep,
top 0.16-0.30 m), and M, a mound of fill (0.40-0.56 m wide and 0.28-0.36 m deep, top 0.14-0.20 m); a heap's class is
drawn R with probability 0.2. Band A (south of the ridge) and band B (between the ridge and the slab) each want 2 or
3 heaps; every heap leans on the south or the north line of its band (the site's south wall or the ridge's south
face; the ridge's north face or the slab's front edge) and none stands free; the first heap of band B is tried
against the slab's front edge first. A heap's occupancy for every placement rule is its placed rectangle. A draw is
rejected when the rectangle comes within 0.45 m of the pad's centre, or within 0.40 m of search point 1 or 2 or of
the floor point 0.25 m in front of or behind the middle of a ridge opening; when it meets a mouth's apron (the
mouth's width and 0.05 m on each side, 0.70 m southwards) or the dust patch; when the open floor between it and
another solid is neither closed nor as wide as the family's spacing rule asks; when the gap that the documents
promise (less than 0.15 m, or at least 0.60 m, measured on the compiled model) fails; when the camera sees less than
0.70 of its surface or not its top; or when its class geometry or its piece bound fails on the compiled world. Heaps
are dropped from the last one while a route fails on any of the three counterfactual sites.

Each heap is one piece of the maintainers' heap library (version 3: 9 usable pieces, 4 mounds and 5 rubble heaps,
each admitted by its own 60 trial runs of the simulated robot against it; the library and its registry stay with the
maintainers), turned, mirrored and scaled inside the piece's registered interval. The picture, the depth and
the collision come from one mesh: the collision body is the mesh's upper envelope on a 5 mm grid, written into
`site.xml` as an inline MuJoCo height field, sunk 10 mm, with a ring of zero cells around it and its largest value
equal to the field's height size. The ring and that maximum are needed: MuJoCo rescales an inline height field
between its smallest and its largest value, and a grid without zero cells compiles to another shape. For the same
reason every statement about a heap's shape (the predicates G20-G24, the QC's Q7 and Q8, the sentence checks of the
documents) is measured on the compiled model, by rays at the geoms, and is never read from a stored array or from a
label of the package. A mound keeps the scan's natural toe and has no rim; a rubble heap has to be 0.03 m high
0.02 m inside its outline along at least 95 % of every exposed side. The earth and grit strewn around a heap's foot
in the picture are picture only: they are in neither the package nor the collision. No name of a library piece, of
an art pack or of a texture is in a site package.

**Predicates of a candidate** (all from the package). G1-G19 are version 1.1's, with every heap's rectangle a solid:
G1-G11 (camera coverage of the camera rectangle, projection against
the given intrinsics within 1 px, visibility of the open floor and of points 1 and 2, nothing of point 3 or the void
visible, the three mouths alike with no camera ray on a plug or any hidden box, point 3 at least 0.62 m from every
solid, the blind way 4.0-6.0 m, routes, minimum distances), G12-G16 for the void (at least two bends, at least one dead
end, passage widths, no straight line of sight from a mouth to point 3, the two known walls reachable from point 3),
G17-G19 for the rubble (the gap beside every block is the mouths' width and a route through the gaps exists; at least
one block on the way in, reaching at least 0.04 m into the centre band of the widest roll; clearances). G20-G24 are
the heaps', each measured on the compiled model: G20 the class geometry (the placed rectangle and the compiled top
inside the class's ranges, the compiled footprint inside the rectangle to within the field's one-cell roll-off; for
class R the edge rule and a core at least 0.10 m high on at least 90 %; for class M a compiled depth of at least
0.28 m, a compiled gap to its line of at most 0.05 m, the scale inside the registered interval and the piece's
registered trial verdict); G21 the keep-outs and the gap promise on the compiled occupancy; G22 visibility (the
camera sees at least 0.70 of every heap's surface and its top; no camera ray ends under the slab; every heap south of
the slab's front edge); G23 the count and the lines (at least 3 heaps, at least 1 in band A, at least 2 in band B of
which at least 1 against the slab's front edge, each heap leaning on its line); G24 the piece bound (the zero ring
and the maximum; 200 vertices of the compiled field within 0.5 mm of the library piece's grid; the fit against the
piece's mesh on a 1 mm lattice: nearest-point distance at most 3 mm on at least 0.98 of it for class R and 0.99 for
class M, at most 8 mm anywhere; the scale inside the interval the generator draws from). G7c compares the three
counterfactual sites (each with its own rubble) ray for ray, frame for frame and `mission()` for `mission()`, the
heaps' public records and collision bodies included; G9 and G17-G24 are judged on all three, so admission does not
depend on which mouth is open.

**Rule first.** The admission and selection rule (rule v5; sha256
`7b5c8b46d3c84026acc818b559445bdf0a26f559c97337c9e69de9c705c9b172`, 44 472 bytes, a file kept by the maintainers; its
hash, pin time and full text are in `hidden/selection.json`) was pinned at 2026-11-21T21:44:43Z, before any seed of
its hidden pool was generated, run or opened with this family's generator. Pools: hidden 1101-1200 (extension
1201-1300). The practice sites are not a pool of this rule: they are the triple that the rule before it, pinned
before its own pools, chose from its practice block 901-940, carried over byte for byte with that block's rows. That
earlier rule's hidden selection, a greedy in seed order without backtracking, came out short and was dropped whole;
rule v5 was written without any per-site result of it in view and uses new pools. Of the hidden main pool's 100
seeds 43 pass G1-G24 and 30 are admissible; of the practice block's 40 seeds 20 pass and 16 are admissible; the
extension was not needed and was not generated.

Admission (every row's value of every candidate is in `hidden/selection.json`): the package comes out byte-identical
when it is generated again in a fresh process; G1-G24 hold; and the privileged reference, run through the station's
calls on the package with the episode seed, meets R1-R8: full score without a stuck period or abort (R1); at most
1300 s, and after the selection at most time limit / 1.6 (R2); the primitive count inside [111, 145], a window that
the earlier rule computed once from its own pool and that this rule takes as a constant (R3); no end triangle, set
down on a rollable face (R4); at least 16 primitives on the blind leg going in (R5); strut penetration at most 5 mm
(R6); the same final state in a second process (R7); no contact with a rubble box under the slab and none with a heap
of the open floor, counted at every control period and in the 10 Hz samples (R8).

**Selection.** The hidden set is the lexicographically smallest feasible 10-tuple: among all sets of 10 admissible
candidates, each written as its seeds in ascending order, the smallest one that meets every quota and every minimum
at once. Quotas: third opening of the first ridge T / XN / XL at most 3 / 3 / 4, corner E / W at most 5 / 5, rank of
the open mouth by its distance to point 3 near / middle / far at most 4 each. Minimums: the open mouth west / centre
/ east on at least 2 sites each, tunnel roof flat / sloped on at least 4 each, sites with 3-4 heaps and sites with
5-6 heaps at least 3 each, heaps of class R and of class M at least 12 each over the set. Two fallback tiers were
written into the rule in advance (the minimum of sites with 5-6 heaps lowered to 2, then to 1; nothing else is ever
relaxed); neither was used: the set was found at the rule's own minimums in the main pool (`hidden/selection.json`
`pools.hidden`). Episode order h01..h10 is the set in seed order. The practice triple is the earlier rule's first
feasible triple in seed order (one site of each rank, at most 2 of a corner, at least one site with 5 heaps).
Nothing was hand-picked. The realised strata, the set-level checks (`set_level_checks`, all holding) and every
candidate's rows are in `hidden/selection.json` and are not repeated here: these notes print no seed, no package
hash, no open mouth and no layout of a scored site.

Two pool-level checks, both fixed in the rule before its pools were opened. Dead reckoning: chaining the handbook's
table over the reference's blind leg ends within 0.20 m of point 3 on 11 of the 62 candidates of the check's pool
(0.177; the rule's line 0.35: select as usual). The cue check (per-mouth features that a program can get without the
truth: version 1.1's, and for the heaps the distance to the nearest heap, the heaps near the mouth's apron, the
nearest heap's class and top, the site's heap counts and the heap pixels in the mouth's zone; single-feature rules;
20 000 label permutations): permutation p 0.380 on the 46 admissible candidates of the gate pool and 0.9994 on the 13
selected sites; the family-wise gate (p above 0.05) holds on both.

Seeds, archives, `site.json` and `site.xml` of the hidden and practice sites are pairwise disjoint and differ from the
development packages, from the candidates of every earlier selection and from the calibration package
(`set_level_checks.disjoint`). Every hidden episode's seed is its site seed (the set-down and every noise stream
derive from it), so a scored episode repeats exactly; a practice episode draws a fresh set-down and noise. Each
frozen package holds `site.json` and `site.xml`, byte for byte the members of the candidate archive that the
admission runs used (PAX tar, members `<tag>/site.json` and `<tag>/site.xml`, mtime 0, mode 0644, uid / gid 0, xz
preset 6 with CRC64; written twice, byte for byte the same). Rebuilding a package from its seed is the
maintainers' work and cannot be done from this folder alone: the generator reads the capability tables and the
heap library, which are not published (the copies in `review/sitegen/` are for reading).

The thirteen selected sites and the reference's admission run on each (`hidden/limits.json`, `hidden/seeds.json`,
`hidden/selection.json`); everything else about a site is in `hidden/selection.json`:

| site | reference time (s) | primitives |
|---|---|---|
| h01 | 1078.57 | 139 |
| h02 | 944.30 | 121 |
| h03 | 1080.19 | 139 |
| h04 | 1069.38 | 138 |
| h05 | 911.89 | 116 |
| h06 | 1103.67 | 143 |
| h07 | 1134.18 | 145 |
| h08 | 1031.37 | 133 |
| h09 | 1125.75 | 145 |
| h10 | 983.17 | 125 |
| p1 | 875.85 | 111 |
| p2 | 1129.74 | 145 |
| p3 | 990.73 | 128 |

**Time limit.** 10 s x ceil(1.6 x the reference's longest time over the 13 selected sites / 10 s) = 1820 s (longest:
h07, 1134.18 s, 145 primitives); every selected site's reference time is at most 1820 / 1.6 = 1137.5 s. The practice
sites have the same limit; the calibration site max(900, 10 x ceil(1.6 x 872.5 / 10)) = 1400 s.

**Measured programs.** By the maintainers' rule the only programs measured are the privileged reference, the
degenerate submissions and the maintainer variants below, each through the real `verify.py` on the frozen set. No
pilot, probe, baseline ladder or non-privileged solver is built. The table gives what each must score; the measured
values are the results of the maintainers' QC runs (section 9).

| program | what it does | must score (`void_search_score`) |
|---|---|---|
| oracle | the reference with the token | the anchor's `value` on every run, the final states of `hidden/seeds.json` |
| D0 | `solve` returns at once | 0.000: `contract_ok` 0 (the contract asks for at least one `step` call) |
| D1 | waits 25 s, reports all three points from the pad, ends | 0.000 |
| D2 | never moves, steps to the time limit | 0.000: F = 1, c = 0 |
| D3 | random primitives with a fixed seed to the time limit, no report | 0.000 |
| D4 | imports `socket` | 0.000: `contract_ok` 0 (network audit), no episode runs |
| D5 / D6 | the oracle's `operate.py` without a token / with a wrong token | 0.000 / 0.000: the privileged call is refused in every episode; D6 records `oracle_token_mismatch` |
| D8 / D9 | the true pose: point 3 reported against the back wall and the corner's side wall / the robot pressed against the first rubble box under the slab, all three points reported there | 0.000 / 0.000 |
| D10 | the true pose: the robot rolls at the heap nearest to the pad until the time limit, no report | 0.000, and no internal error of the cell while the robot leans on the heap |
| example | `example/operate.py` of the payload | 0.000 on the calibration site (`contract_ok` 1) |
| V1, V2, V3 and a fourth | maintainer variants of the reference (`review/qc/variants/`), expected values written before the runs | 0.200, 0.850, 0.600, 0.680 on every site: the two visible points with retrieval; all three without retrieval; point 3 at 0.30 m; the fourth: the remaining edge cases (15 s dwell, 0.41 m, pad at 0.61 m) |

D7 is D1. The model episodes run on a version are described in the pull request that adds it.

## 9. The grader's own QC

The checks, and the line of each. Their results on this version (the QC runs) are kept by the maintainers and are in
the pull request, not here.

| # | check | holds when |
|---|---|---|
| D0-D10 | the degenerate submissions of section 8 through the real `verify.py` on the frozen set | every one scores 0.000; no internal error of the cell |
| Q1 | determinism | the oracle twice through `verify.py`: the ten final states equal between the runs and to `hidden/seeds.json`; a fixed sequence of frames on a hidden site in two processes: the same depth, the same bytes outside the robot's window, and inside the window differences within a bound of the maintainers' bake tools (not published) |
| Q2 / Q3 | the scoring's edges, through the maintainer variants | a report 0.19 m from its point earns full credit, 0.30 m half and 0.41 m none with a valid dwell, a 15 s dwell none, a fourth report is refused, the pad 0.59 m away gives F 1 and 0.61 m away F 0.85, no retrieval F 0.85, and motion beyond the still tolerance in the dwell window voids the point; the four variants equal their expected scores on every site |
| Q4 | the interface | bad arguments, the reply whitelist, the mission on all ten sites without a layout value (the heaps' rectangles, tops and lines among the hidden values), refusals, privileged calls, the pose switch, no debrief |
| Q5 | leak scan (the maintainers' scanner; it and its word lists are not published) | no fail word in the instruction, the payload and the client; no hidden value or name of the 13 frozen sites (the heaps' included) among the numbers the static agent-visible files print, apart from the family's public constants; no site name and no package hash; the scan for machine paths, account and host names, internal ids and citations of documents that are not in the repository, over every file of the task folder and its repository tests, without a failing hit |
| Q6 | the selection, recomputed from the stored packages | on the 13 sites: every archive, `site.json` and `site.xml` hash as pinned; G1-G24 (G20-G24 on the compiled model, G9 and G17-G24 on the three counterfactual sites) and R1-R8 as in `hidden/selection.json`; the set-level checks hold |
| Q7 | the heaps' depth in the plate against the collision, per heap of the open floor, on the 4 x depth plate against rays at the compiled height field | (1) pixels where the art mesh is nearer to the camera than the height field by more than 10 mm: at most 0.1 %; (2) in the heap's foot (height field at most 0.03 m high): pixels within 10 mm at least 99 %; (3) the two-sided share and the share where the art mesh is farther are recorded per heap and are not lines. Depth edges and the heap's outline are left out 2 px to each side; nothing under the slab has a pixel |
| Q8 | no heap can be got over, per heap of every frozen site, in that site's own compiled world: 4 privileged trials from each exposed side | in every trial the robot's centre comes no deeper inside the heap's outline than 0.10 m and 40 % of the heap's depth for a rubble heap, 0.16 m and 50 % for a mound; for a mound also no primitive ends with the three lowest end caps above 0.03 m and the centre never comes over the crest (the part at 0.85 of the top or higher). How often the robot turns away and rolls on, and how often it ends on an end triangle, are counted per class against the library's rates, not judged trial by trial |
| Q9 | the plates' binding and the bake's own checks | every pinned site has a bundle whose `meta.json` names the pinned package sha256; every file's sha256 is the one of the asset list or of the pinned tarball; per site the differences between the plates baked for the three counterfactual sites stay inside the bake's line in front of the mouths, and the bake's marker, colour, patch and calibration-line checks hold on the bake's own readings, which the maintainers keep |
| documents | every number of the agent's documents against the configuration (the maintainers' document checks, not published) | no stale generated block, no mismatch of a hand-written number, every row of the sentence table holding; the statements about the station's behaviour and about the heaps hold in the real cell on the 13 frozen sites and the calibration site |
| image | isolation and frames in the task image (the maintainers' image checks, not published) | the agent account can neither list nor read the practice service's files, the plate bundles among them; the leak scans over the image's agent-readable file list without a hit |

Q7 is one-sided on purpose. Under an overhanging brick or stone the height field, an upper envelope, fills a gap that
the camera looks into from the side: there the picture's surface is farther than the collision surface, inside the
heap's outline and above the floor, where the robot cannot come. What must not happen is the opposite (a surface the
camera shows in front of what the robot can touch) and any difference at the foot, where the robot does touch.

## 10. Frozen files, and what to re-run after a change

The site set was frozen at 2026-11-22T01:18:33Z (of that freeze the maintainers keep the sha256 of
every frozen file, the rule's hash and pin time and the set-level checks, and every dated revision after it). The
files that decide a score, with their sha256:

| file | sha256 |
|---|---|
| `tests/grader_config.json` | `1a608cd1065560f17fcab8e05b92b8e01237b22574a4b82daedede40ee456b71` |
| `tests/voidscore/acceptance.yaml` | `729f1bf381e1c0d1904c7ae88e496e9e8636016628f3eba8831b76c6d4f05c06` |
| `tests/voidscore/scoring.py` | `b458710b300f63ec4d38f645b6d54625822290ce15759c088806f198aa92a4c8` |
| `tests/verify.py` | `d04a829865c6d140a29ce2fdae4f9dc5124d0070c019697d49999b4227ab7649` |
| `tests/test.sh` | `1c02f20ff76668e06d7d5b5448358af73935098ddf420d9841ea23de03db9996` |
| `tests/anchor.json` | `2ea1735ca7dc00fde436a1b0ff04d875bbd9e8e57d6e3136c126f6383c8f884b` |
| `tests/hidden/seeds.json` | `af74174c8f0fe32f6143bc82c90b843f19d066667be8e0154d8fac8407f422f2` |
| `tests/hidden/selection.json` | `e3a40b3fb84538d7772f62a9fba6c0ad1f477de7fbfeb0761a5977620c27a33c` |
| `tests/hidden/limits.json` | `22f5bb0bc3fc4da098dbfc3da7201cf161a23f8b3f78642aa256f105f2344724` |
| `tests/hidden/oracle_token.sha256` | `9a16ed975ed3162d863de09226617582ee47af2c0fbfe95fd8a55f6250b6e754` |
| `tests/voidcell/_lib/ysim.lock.json` | `c606b4376e7ae718fc2d3314f016bfc179988393ca021ce8b031affd1487f58c` |
| `tests/voidcell/server.py` | `19748bcba01fcb35ff008dcfbe76459b6d2337ba5e0dff059809fa9fe17c54b0` |
| `tests/voidcell/client.py` | `a903e92d8b4026912d330fbd642718dc90ca9629ee4822c6e4fc2e349fea2646` |
| `tests/reference/reference.py` | `1255f4bf14e4bd066ef67e1b5767aab166634f91312fa63c7d244bb7935c2cd6` |
| `tests/given_hashes.json` | `fd4c09d1e86ba5ad25e0d2427d34e971299ab5bd20098ae1501297719dbd55a2` |
| `environment/practice_ground/sites.json` | `65a4dd272421d688e690cb8e86c865a3f9b538936621eac830291029720fa7ca` |
| `environment/practice_ground/cal/cal.tar.xz` | `4d430654a65a6fd922fe93849ed34fc31b1deee28b44d6beb464ef2e62aac9f3` |
| `instruction.md` | `38118f8fc72d71b94469c492371b44beb3e9c5944cb5047ff2d90d0ea2c2e4fd` |
| `task.toml` | `2bc399dee936d67669fb6c521f7fa91afff9e14054dd4270babc7d8207f03fab` |
| `tests/reference/` (tree) | `2c50629f70d7218e11e8aee693bd7d7d4476ba393da8de6e5b8f5a0e6630630c` |
| `tests/voidcell/` (tree) | `5ddb6656e80e59978db66570766e94086eea63dc41c5a24fdb61917d90f4227a` |
| `tests/voidscore/` (tree) | `30990f979f2f20aa13eec98ad55a6403c29270e1d1e7db2ebe9b5182e260f0fd` |

A tree's sha256: of the text lines `<sha256>  tests/<tree>/<path>` (one per file, sorted by path, `__pycache__`
skipped). The hidden sites' archives are pinned by `hidden/seeds.json`, the practice sites' by
`environment/practice_ground/sites.json`, the hidden sites' plates by `grader_config.json` and the practice ground's
by `environment/assets.sha256`; these pins are not copied here.

The maintainers run these steps with their own tools; a file under `review/` named below is this folder's copy of
such a tool, kept for reading and review and not run from here. Python is that of the maintainers' environment
(MuJoCo 3.14.0, numpy 2.5.4, scipy
1.18.1, PyYAML 6.0.3, the image's pins), with the three kernel variables of `test.sh` (`OPENBLAS_CORETYPE=Haswell`,
`OPENBLAS_L2_SIZE=256`, `NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"`) and `MESA_SHADER_CACHE_DISABLE=true`;
the generator runs with `OPENBLAS_CORETYPE=Haswell` and one BLAS thread. A frame of the plate backend outside the
image needs the image's Mesa (OSMesa and the software Vulkan driver); with another Vulkan driver the robot's colour
is not the image's.

| after a change of | run |
|---|---|
| `voidcell/` or `voidscore/` | `review/degenerate/practice_sync.py sync` then `check` (the practice ground's copy); the cell's tests (the maintainers' tests of the cell, of its core, of the reference and of the scoring; not published); the QC of section 9 (`review/qc/qc_verify.py`, `check_runs.py`) |
| the client | the same, and the grader's copy of the payload written again (`setup_payload/` and `given_hashes.json`; the maintainers' tool, not published) |
| any payload file or generated block | the maintainers' document tools (not published): the generator of the blocks, the strict check of the documents and the check of the stated behaviour in the real cell; the payload copy as in the row above; then lint and a real validation |
| the cell's speed or the image | the checks in the image and the measurement of the costs (the maintainers' tools, not published), the documents' chain above, and `task.toml`'s timeout by its header's formula |
| the look `agent_look.json` or a module of the plate backend | `practice_sync.py sync`, then the frame checks (Q1, Q9, the image's). A change of the look's display transform also changes every bundle: a bundle's picture is that transform of its stored radiance and the backend refuses a bundle packed with another one, so the pictures are derived and the bundles packed and pinned again (next row) |
| a site package | its plate bundle is bound to the package's sha256: bake the bundle again (the maintainers' bake tools, not published), write the tarball (the maintainers' packer, not published: PAX tar, regular files `hNN/<path>` only, sorted, mtime 0, mode 0644, uid / gid 0, xz preset 6 with CRC64) and pin it in `grader_config.json`, or list the practice ground's bundle in `environment/assets.sha256` and rebuild the image; then Q7, Q8 and Q9 |
| the generator, a frozen size, the heap library or the simulation base | back to the selection: a new admission rule pinned before any result, the candidates regenerated, the reference run on them twice, the set selected by that rule, the packages written (format: PAX tar, members `<tag>/site.json`, `<tag>/site.xml`, mtime 0, mode 0644, uid / gid 0, xz preset 6 with CRC64), every plate baked, the oracle through `verify.py` twice, then the anchor and a new freeze |
| a text-only edit of `reference/`, `voidcell/`, `voidscore/` or the generator | prove the final states unchanged (the reference on a development site, the regenerated packages byte for byte) and, for the plate backend, the frames unchanged; record the trees' new sha256 next to the pinned ones; any byte of `tests/` after the freeze is a dated revision with the oracle through `verify.py` again |
| anything | the leak scan of Q5 (with its scan for machine paths, names and internal ids); the task's three repository tests; the check of these notes' numbers (the maintainers' tool, not published) |

The oracle through the real `verify.py` outside a container (not root: no lockdown, every other gate as in the image):
the stage a byte copy of `tests/`, the submission made by `solution/solve.sh`, the root-owned givens copy built from
`setup_payload/` as `test.sh` builds it, `ARB_HIDDEN_PLATES_TARBALL` pointing at the local copy of the plates, and
the image's Mesa on the library path; expected: ten episodes of 1.000 and the ten final states of
`hidden/seeds.json`.

## 11. Known limitations

1. **One model episode is the only measurement of difficulty.** By the maintainers' rule the only evidence built is the
   privileged reference, the degenerate submissions and the grader's QC. Version 1.0's one model episode read 1.0,
   version 1.1's 0.23 and version 2.0's 0.25; the episode on this version is described in the pull request that
   adds it. One episode is one reading, not a mean.
2. **The time limit is tight by arithmetic.** The factor is 1.6 on the reference's longest time, and the reference
   knows the site. The maintainers' arithmetic for a careful program that does not know the site (an estimate, not a
   measurement of any program) is kept by the maintainers and is not published; what such a program loses
   first is the retrieval (F 0.85) and the third mouth tried. The rounding to 10 s gives 20 s more than version 1.1.
3. **The simulated robot is not a calibrated twin.** Its primitives are this model's (section 5, base assumptions).
   It needs a stop band three times tighter than the designers' code to tip on every roll; among the 64 ways the
   assembly guide allows the cables to be hooked, the mean turn per forward roll spans -25.7 to +1.2 degrees, and the
   one in use was chosen by geometry alone. `roll_100_100` and `roll_100_120` are close to the tipping threshold.
4. **The picture and the collision are two things that are kept close, not one.** The plate shows the dressed scene;
   the robot touches boxes and height fields. For a heap both come from one mesh and Q7 bounds the difference in the
   direction that matters, but under overhanging bricks the depth image shows a surface behind the collision surface.
   The dressing of a box (a rack, a wall, a crate) lies on the box's faces by the scene builder's own checks; where
   two site boxes meet, the dressing can show one piece running into another (a board into a crate): that is the
   site's shape, the two boxes do meet there.
5. **A plate is one still.** The light, the dust and every shadow but the robot's are computed once per site and do
   not change in an episode; the robot is lit from probes of the bundle, and its shadow is drawn on the plate. The
   frame does not come from one consistent render of robot and scene.
6. **Regularities the generator leaves in.** Given everything visible the open mouth is equally likely to be any of
   the three, and the heaps are placed before the open mouth is drawn. Two geometric regularities of the void remain,
   neither naming the open mouth: the passage behind a mouth runs away
   from the corner when the mouth is the one nearest to the corner and toward it when it is the farthest (the other
   side has no room); on admissible sites the rank of the open mouth sets the way's shape (ranks 0 and 1: U; rank 2:
   L, about one in ten Z).
7. **Dead reckoning alone ends inside the full-credit circle on some sites.** Chaining the handbook's table over the
   reference's blind leg, from the true pose at the mouth, ends within 0.20 m of point 3 on 11 of 62 candidates
   (section 8); the chamber's two walls are the site's own and their coordinates are known, so a program can touch
   them and come back to the point.
8. **The low decoy can be forced.** With the planner told that the 0.15 m slot is open, the robot scraped through in
   1 of 20 runs of 40 pushes; in the simulation base's capability measurements (not published), 0 of 32. Scoring only looks at where the robot dwells, and no
   document promises that an opening can or cannot be passed.
9. **Determinism.** The physics is single-threaded and repeats bit for bit; the anchor comes from a reference that
   uses no frame. The frames are rendered on the CPU by two software renderers; Q1 checks them between two processes
   on one machine, and equality of frames on another CPU generation is not guaranteed and was not measured.
10. **`pose_source = "cell"`** is implemented and unit-tested in process, but no verify run and no image check used it.
11. **The practice ground's protection is the account isolation alone.** Its copy of the cell (59 files) carries the
    prototype's names and the designers' repository names, exactly as `tests/voidcell/` does, and its plate bundles
    lie beside it. Linux leaves the service's `/proc/<pid>/cmdline` readable (the interpreter and the service's path,
    nothing else).
12. **The practice sites do not show every kind of opening.** The three practice sites were selected without a
    quota on the first ridge's third opening and share one kind; the calibration site has one opening of each kind,
    with its dimensions.
13. **Host DNS search domain.** On the maintainers' build host a sandbox inherits a `search` line that names the
    institution the host sits in. Model episodes on this host are run through a per-process `docker` wrapper that
    adds `--dns-search .`; the task itself does not depend on it.
14. **Serial verify, long worst case.** A program that spends every budget holds the verifier for hours (section 7).
    Host load slows frames and physics; nothing in the score depends on wall clock except the program's own 600 s of
    program time and the 180 s silence limit.
15. **Costs in the documents** are the task image's on one host at one load; a busier host is slower.
16. **The plates are not in the repository.** They are files of the benchmark's asset dataset at the pinned commit:
    the image build and the grader download them, and neither runs from the repository alone (section 2).
17. **Accepted blemishes of the picture.** On one practice site one ray at the foot of one heap meets a back face of
    the heap's own mesh, below the heap's envelope and a few millimetres above the floor. On two scored sites the
    floor decoration in front of the three mouths differs by one triangle in one mouth's window; the decoration was
    generated for each of the three counterfactual sites and is byte for byte the same, so the difference does not
    follow the open mouth. Both were accepted as they are.
18. **A neighbouring task.** A search-and-rescue task with a tracked vehicle in an earthquake-struck building was in
    review for this benchmark when version 1.1 was written; this one differs in the platform, the off-board camera,
    the unseen void and the rolling primitives.
19. **One accepted exception of Q8.** On one practice site, in one trial against one mound of fill, the robot ended
    three primitives standing on the mound's low toe with none of its end caps on the floor, the lowest of them a
    few millimetres above Q8's line for end caps. Its centre stayed well inside Q8's depth limits and never came
    over the crest, and the robot came back to the floor. No trial fails on a scored site. The site package is
    frozen and the line was not changed; the trial is listed among the results of the QC runs.
20. **The handbook's heap counts are the heap library's.** They were measured against one heap standing alone on a
    free floor, with rolls at its front and along it. In the sites' own worlds (Q8) the turn-away after a trial
    succeeds less often, mostly because the rolls away from the heap end at a wall, at the ridge or at a post, and
    pushing a mound of fill from its side ends on an end triangle more often than the handbook's count; for rolls
    at a heap's front the two agree. The readings are those of the QC runs. No document promises a rate, and the
    reference touches no heap.

## 12. Record

| date (UTC) | what |
|---|---|
| 2026-11-17 | version 1.0: generator W2; cell, station, reference on 107 candidates (ysim 0.3.1); scoring, verify, practice ground, calibration site; selection and freeze (rule pinned 17:24:06, frozen 18:24:46, anchor 1.0); the grader's QC; the agent's documents; image, layout, two real validations; one model episode (read 1.0) |
| 2026-11-18 | version 1.1 designed: site family W3 with the rubble, credit 0.20 / 0.40 m, weights 0.05 / 0.15 / 0.80, time-limit factor 1.6, the kernel pin of the program's process; generator, calibration site (calgen 0.2.0), reference 1.1.0 |
| 2026-11-19 | version 1.1: admission rule pinned 00:51:35; frozen 13:21, anchor 1.0; the grader's QC, the image, a real validation; one model episode (read 0.23) |
| 2026-11-21 | version 2.0: the admission and selection rule pinned 21:44:43, after the heap library (version 3), the generator `w4gen` 0.3.0 and the calibration site with its two heaps (calgen 0.4.0) were fixed |
| 2026-11-22 | version 2.0: the site set frozen 01:18:33 |
| after | the plates of the 14 sites, the switch to the plate backend, the anchor, the grader's QC, the image and the validation: dated in the maintainers' list of revisions after the freeze (not published) and in the pull request |
