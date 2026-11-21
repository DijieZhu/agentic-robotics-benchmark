# tensegrity_strut_assembly_3bar_panda: notes for maintainers and validators

The agent-facing contract is `instruction.md`; the cell is described in
`environment/setup/payload/given/cell/cell_manual.md`, its wire in `cell_api.md` and its numbers in `cell_config.json`
(the givens are byte-copied here as `setup_payload/`). This page records what the grader does, the three kinds of mate,
what the hidden kit is and how it is rebuilt, how the oracle runs, how the grader's own QC is re-run and what is known
to be rough. Where the cell's robot, room and tools come from is in `NOTICE.md`.

## 1. What the robot in this task is made of

Every module the cameras see is the authors' published whole-strut mesh, triangle for triangle: the mesh split into its
494 edge-connected pieces (no hole filled, no vertex merged, no triangle reduced) and the pieces grouped into modules.
Nothing of the robot is modelled, redrawn or completed. The collision pieces are derived from the same mesh (convex
pieces, carved slots where a part slides on the rod). The cables, the strain sensors and the wires have no geometry in
the sources and are not in the task: the end state is three mechanically complete struts in the cradle, waiting for
the cable net.

## 2. The episode and the grade

* `verify.py` (from the quadruped assembly task's grader): stage lock, contract gate, `fetch_hidden_gt` (the kit tarball,
  section 4), then `episodes` (3) hidden 63-bit episode seeds from OS entropy (`hidden/seeds.json` pins them for a
  maintainer's re-run); per episode a fresh `arm.assembly_strut.server.StrutInfoServer`, `reset(seed)`, then the
  agent's `assemble.py` `solve(cell)` run as the unprivileged runner account (`nobody`) over the wire.
* Each episode's layout is drawn from the seed (`arm/assembly_strut/layout.py`): the cradle +-10 mm, each of the two
  trays +-20 mm and +-10 deg, the six motor modules permuted over the six motor nests and the six end caps over the six
  cap nests, each motor module turned about its bore 6-16 deg off a straight transfer (a hidden 11 deg offset per nest
  plus +-5 deg per episode), each cap's turn, each lid end for end half the time; cameras (mount, intrinsics,
  distortion), light and F/T bias jitter. The built cell must pass the reach check (`server._reach_check`, as in the
  quadruped cell) or the whole layout is drawn again (at most 8 draws, then the nominal layout with this episode's
  cradle). 300 seeds: 300 pass at the first draw.
* Stage score = the weight of the scoring mates / 18.0 (6 motor modules x 1.5, 3 lids x 1.0, 6 end caps x 1.0). An
  episode with a safety violation scores 0. `assembly_score` = the mean over the three episodes; `overall` =
  `assembly_score` x `contract_ok`. Statistics (not scored): per-mate rates, LOCKED_OFF reasons, frames per camera,
  simulated and wall time, illegal calls, safety events, peak readings of the four rules, the renderer used.
* The contract gate is the precedent's: `assemble.py` a regular file defining `solve`, the route gate (no weight files,
  no training calls, <= 60 MB), the network audit (no networking imports), importable as the runner within 120 s; a
  privileged call without the oracle token is answered `forbidden` and counted (`illegal_call_count`).

## 3. The three kinds of mate (`arm/assembly_strut/coupling.py`, the numbers in the kit's `mating_graph.json`)

The precedent's six states: WAITING (a required mate does not score yet) -> FREE -> ENGAGED (captured) -> HOLDING ->
LOCKED (scores) or LOCKED_OFF (fastened, does not score). A fastened module cannot be moved again.

| kind | slots | captured when | fastened when | scores when |
|---|---|---|---|---|
| rod sleeve, motor module | `s<k>_motor_p`, `s<k>_motor_n` | its leading bore mouth within 2.0 mm of the rod axis, the bore within 5 deg of it, the rod end >= 3 mm into the bore, at the rod end of that slot | at rest (< 5 mm/s, < 2 deg/s relative to the strut for 50 ticks, it may still be held) within 3.0 mm of the design depth; a captured module slides on a rail along the rod and turns about it with light damping | axial <= 1.5 mm and roll <= 5 deg from the design; either build fits either end |
| rod sleeve, end cap | `s<k>_cap_p`, `s<k>_cap_n` | as above, only once that end's motor module scores | at rest within 1.5 mm of the design depth (bottomed out on the rod) | always (the turn about the rod is free), except that the strut's first fastened cap sets its colour and a cap of another colour on the same strut is LOCKED_OFF |
| seat, bay lid | `s<k>_lid` | within 4 mm across, at most 3 mm short of its seat and 8 deg, only once both motor modules of its strut score | pulled home at <= 10 mm/s, fastened at rest | within 1.5 mm and 3 deg of the design |

## 4. The hidden kit

`gt/` (fetched at verify start, never in the image): `kit.json` (modules, pieces, frames, masses, the delivery-frame
transforms), `mating_graph.json` (the 15 mates, their frames, depths, windows and tolerances), `meshes/visual/` (the
render meshes: the authors' triangles grouped by module and material) and `meshes/collision/` (the convex pieces and
`collision.json`). It travels as one deterministic tarball, `gt.tar.xz` (43 files, 6 372 372 bytes, sha256
`63c5b207b3e992e00126aa39c791791c934b875a4bafcc2a0639641bfb390169`), pinned in `grader_config.json` `hidden_gt`
(sha256 and file count; the URL is a placeholder until the asset dataset holds it). Until then `tests/test.sh` points
`ARB_HIDDEN_GT_TARBALL` at the local copy `tests/assets/gt.tar.xz` (not in git); `verify.py` checks it exactly as
a download. The three struts of the cell use the same files.

Rebuild (the maintainers' build tooling, not part of this repository): the split of the mesh and the grouping into
modules (with a check of the grouping), the carved convex pieces (CoACD with seed 0, recomputed byte for byte without
a cache), then the deterministic tarball of `gt/` (its check rebuilds it and compares the sha256), then the new sha256
and file count into `hidden_gt`. `kit_info.json` (the cover colours), `arm/assembly_strut/render_omit.json`
(the fully hidden pieces the agent cameras leave out, pinned by `agent_render.omit_list_sha256`) and
`hidden/bench_plan.json` (the reference program's nominal layout, nest offsets and stored grasps) live in `tests/`.

## 5. The oracle

`solution/solve.sh` submits `solution/reference/assemble.py` with the 256-bit token (`hidden/oracle_token.sha256` pins
its hash). With the token the server is privileged: `oracle_nominal_layout` puts the episode on the bench plan's
nominal layout (the reference program is planned for it and is not claimed to work on jittered layouts) and
`reference_assemble()` runs `arm/assembly_strut/reference.py` inside the cell through the same routed ticks and safety
watch. The oracle first takes one frame of each camera, so every validation renders through the agent cameras' path
(Filament on CPU Vulkan in the baked room) and fails loudly if a frame does not come back. One reference run (the agent
home of `cell.robot_home_q`): 15 / 15, 319.5 s simulated, no unplanned robot contact, no violation; peaks R1 0 N,
R2 2.48 N, R3 0.79 N (0.2 s sustained), R4 0.48 m/s.

## 6. The four hard rules and their numbers

R1 the robot (any collision geom but the two pads) pressing on the cradle, a tray, the bench, a module or a strut core
above 15 N in any tick; R2 a module pressed on another body above 30 N for 20 consecutive ticks (not the pads' grip, not
two bodies of one fastened cluster); R3 a rod overhang loaded across the rod above 4.0 N for 20 ticks (contacts on the
rod plus the rail / lock constraint forces of the modules on it, gravity included); R4 a module's centre of mass faster
than 3 m/s. The values were frozen by a rule written before the reference ran: max(proposed, min(credible upper bound,
ceil(1.5 x the reference's maximum))), applied on one strut and on three (both gave the proposed values 15 / 30 / 4.0 N,
3 m/s).

## 7. The grader's own QC (re-run)

The maintainers' QC tooling is not part of this repository; what it runs and what it found is below. These are the
grader's tests: the jigs and the external forces are privileged hooks (`qc_jigs`), never part of an episode, not a way
of solving.

* Jig cases: the real server with this `grader_config.json` plus `qc_jigs`, privileged; a
  mocap jig carries each part (10 mm outside the mouth, 1 mm off the axis, 3 deg tilt, in at 10 mm/s). Benign: two full
  orders (by strut; by type with the builds swapped and the caps at arbitrary turns) -> 15 / 15, episode 1.0. Mutants:
  2.5 mm short, 8 deg off, end for end -> LOCKED_OFF; 10 mm short -> captured, not fastened, then pushed home ->
  LOCKED; lid first, cap before its motor module -> WAITING; lid turned 180 deg -> not captured (the jig forcing it
  into the bay trips R2 / R3: episode 0); a second cap of another colour -> LOCKED_OFF, the first still scores.
* Rule cases: each rule at 0.9 x and 1.1 x its frozen value with one mate fastened first: 0.9 x keeps the
  episode at 1.5 / 18, 1.1 x triggers and gives 0; 52 N between a locked pair does not trigger.
* Contract cases through tests/test.sh in the task image: no file, no `solve`, a networking import -> `contract_ok` 0;
  the privileged calls without the token -> `forbidden`, counted (9 over three episodes), reward 0.
* Determinism: two runs on one maintainer seed (task image, Filament): the same layout, the same lock tick of every mate,
  the same depth frames; RGB differs at most by Filament's own noise.
* The leak scan: names, hidden numbers, sha256 intersections, delivery frames (section 8).
* The documents against the cell: a fact check of every number of the four Markdown texts and every leaf of
  `cell_config.json` against `grader_config.json`, the built cell, `describe()` and the grader's constants.

## 8. Leaks

The givens, the instruction, the image and the payload name neither the robot's designers nor their schools, labs,
papers, design names or repositories (the vocabulary is the CAD sibling task's curated list plus this task's hidden
words); the purchased parts carry their real vendors and part numbers. The given STLs are in frames made from each
module's own geometry (long axis = x, the footprint's minimum-area rectangle; M0 and M2 with a recorded fixed turn), so
their turn about the rod is 4.1-6.9 deg off every multiple of 15 deg and their translation is never zero. No file of
`tests/`, `solution/` or the hidden kit is in the image or the payload, except by design: `setup_payload/` (the byte
copy of the givens) and the precedent's library modules that ship byte-identical to the agent and to the grader.

## 9. Known limits

* The reference program and its bench plan hold for the nominal layout only (as in the precedents); the jittered
  layouts are checked for reachability (300 / 300) but not by running the reference on them.
* Two end-cap releases of the reference (the +z caps of the first and third struts) give a rod-load transient of 10-15
  ticks peaking near 5 N and 7.4 N (a pad's end face slides off the locked cap): below the 20-tick window of R3,
  deterministic, accepted.
* The agent cameras leave out the pieces no ray reaches when their module stands alone (591 561 of the cell's 3 354 993
  triangles); the given STLs and every review render keep every triangle.
* Frame time on a loaded host: top RGB-D 1.7-1.9 s, wrist 1.2-1.4 s (median), single frames up to about 2.1 s; the
  first frame of an episode 15-20 s (the Filament session, the display transform table).
* The authors' mesh has its own small interferences (the main board's header pins reach 0.71 mm into the lid; the lid
  screws' threads overlap their inserts by 0.7 mm^3); nothing is changed, the lid path tolerates them.
* The safety events name the cell's units (`M3_2`, ...); the names map to fixed colours but appear only after a
  violation.
