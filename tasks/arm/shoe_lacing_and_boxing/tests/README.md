# shoe_lacing_and_boxing: design notes for maintainers and validators

The agent-facing contract is `instruction.md`; the cell is described in `environment/setup/payload/given/cell/cell_manual.md` and
its wire in `cell_api.md`. This page records the design decisions, what the grader checks and why, and what is
known to be rough in version 0.2 (2026-10-30, Yuanfang Peng; reference lacing and the lace physics reworked 2026-11-05;
the lid tray's delivery spot 2026-11-07).

## 1. The job and why it is worth measuring

Lacing is the last manual operation of a sneaker line (Adidas's Speedfactory left it to people; Pou Chen and Solomon
hold patents on camera-guided lacing cells). The cell here is the end of a children's-sneaker line: pair after pair,
two Panda arms seat each lasted-open shoe in a cradle, lace it criss-cross with a lacing hook (from below the eyelet
flaps) and a gripper (from above), erect the flat box blank, pack the laced pair heel-to-toe, put the lid on and push
the box to the outfeed. It is a Physical Execution task: a work order of one pair per episode (900 s simulated, 6 hidden episodes), a long
chain of dependent stages, explicit exceptions (a shoe on its side, an off-centre lace), and nothing reported about
the objects: the program works from calibrated RGB-D cameras only. The closest prior art is UniLace (Unity, Obi rope,
bimanual lacing simulator), RLBench's `put_shoes_in_box`, pi0 / RL-100 box folding and EmbodiedSWE-Bench's packing
tasks; none chains lacing with box erecting and packing, runs in CPU MuJoCo, or scores a coding agent that has to build
its own perception.

## 2. The cell (`environment/setup/payload/given/lib/shoecell/`, byte-identical to `tests/shoecell/`)

* `physics.py`: the scene composed with MjSpec from the blocks `cell.py` (bench, pedestals, Menagerie Pandas, hook
  tool, welds, lights), `cameras.py`, `shoe.py` (a rigid hollow shell with real eyelet holes: the flaps are box
  segments between 8-capsule rings), `lace.py` (a 63-link chain of 10 mm segments on two-hinge joints, bending only,
  with two aglets; see the constants at the top of `physics.py` for why 10 mm hinge pairs), `carton.py` (hinged
  panels with joint-equality snap-locks) and a rigid lid tray. MuJoCo 3.14, timestep 2 ms, implicitfast, Newton,
  elliptic cones, dense Jacobian, 20 iterations; a 10 ms control tick = 5 steps. The laces have `condim 1`
  (frictionless against everything but the eyelet rings, which have priority 2, condim 3 and friction 0.3): this is
  what makes the full scene run at 0.6-1.6x real time instead of 0.2x; the rings' friction is what holds a lace in a
  ring. A per-body linear drag on the lace (`LACE_DRAG`, applied in `Scene.step_physics`) brings loose cord to rest
  (MuJoCo's fluid viscosity and a condim-3 lace were both measured at 2-3x the step cost). The batch has two
  eyelet rows (row 1 delivered threaded and tacked; section 5). Caught aglets glide into their seat on the gripper or
  the hook over 0.3 s (a snap to the canonical pose drove the lace through the ring wire), and a held aglet pulled
  harder than 6 N for 30 ms slips out (a welded pull on a pinned lace otherwise diverged).
* Contact bits (`Scene._index`, documented in the manual): ordinary geoms (1, 31), the gripper hand + fingers
  (1, 11), the hook tool (8, 0), lace segments (2, 5); a body held by the gripper becomes (4, 0) (shoe, panel, lid) or
  (16, 0) (an aglet) while held. Hence a lace never touches itself, another lace or the hook; a held shoe keeps
  colliding with its lace (it used to fall out of row 1 on every carry); a held aglet touches neither its lace nor
  the hand. Closing the gripper on an aglet that sits on the hook takes it off the hook (the handover).
* Delivery (`Scene._thread_lace`): the lace path is computed in the world frame from the shoe's actual pose (upright
  or on its side), the tails are laid on the tray plate as a spiral with 90-degree corners that stays 3 cm inside the
  rim and clear of the lid tray (its footprint where it is set down, plus 1 cm) and the other shoe. Row 1 survived the
  settle on all 28 seeds tried. The lid tray's centre is drawn within +-3 cm (x) and +-3 mm (y) of (0.40, 0.135), turned
  at most 0.03 rad either way round, so that even the biggest batch's lid (0.28 x 0.20 m) lies at least 3.8 mm inside
  the tray's near rim and rests flat on the plate (since 2026-11-07; section 5).
* Blow-up guard: `Scene.step_physics` flags `physics_fault` when MuJoCo's BADQACC auto-reset fires (time goes
  backwards or the warning counter is set); the server ends the episode (`done_reason physics_fault`) instead of
  crediting a reset world. The safety watch also counts arm-arm contacts above the force limit.
* Grasping is a documented attach rule (a weld equality retargeted at run time), not friction: UniLace made the same
  choice because "friction alone cannot hold an aglet". The gripper attaches any documented handle inside its pinch
  box at the current pose (an aglet at a canonical pose along the approach axis); the hook's clip catches an aglet
  within 12 mm of its tip. The cradle clamp and the carton locks are welds / joint equalities too.
* One batch per episode (shoe model and box size from the seed), two shoe bodies, two laces, one blank and one lid
  re-used from pair to pair (colours change through `geom_rgba`; a shipped pair is parked under the bench and the
  next pair is set down at once). Geometry cannot change at run time, so the batch is what varies between episodes.
* `cameras.py` + `look.py`: the agent cameras (since 2026-11-07; before, the classic renderer on the physics model
  with three shadowless lights and a checker floor). Colour: MuJoCo's Filament renderer (on Vulkan: Mesa's lavapipe on
  the CPU) on a render twin compiled from a copy of the cell's MjSpec at the first frame of an episode (again when a
  pair's colours change): the same bodies, joints and cameras, so a frame copies the physics qpos and runs forward
  kinematics only. The twin dresses the cell in the factory hall of the owner-approved OVRTX demo scene (the hall,
  the conveyor, fence, racks, cabinets, table, carts and boxes as drawn-only meshes; nothing on the bench), PBR
  materials per part (oak bench top, powder-coated legs, brushed tray, kraft and white-lined board, the Pandas'
  paint), and a sneaker skin on each shoe body (the demo's skin rebuilt with the task's two eyelet rows only: the
  demo's three visual-only grommet rows would show eyelets the shoe does not have; warped piecewise linearly in x and
  |y| onto each batch with unit slope round every ring and over the toe cap and heel block, and its grommets placed
  at the batch's rings). The shoes' shells, rings and seams, the checker floor and every collision geom are hidden in
  the twin. Light: the cell's LED panel 2.05 m above the bench as a shadow-casting spot (93 klm-equivalent candela,
  4800 K) and an image light from an OVRTX probe of the hall at the work area with the cell's live objects hidden
  (cmgen KTX); the strengths and a luminance gain per material class were fitted in linear units to OVRTX stills of
  the five agent cameras at 12 states of the seed-1046 reference episode (the final look, OVRTX / Filament in
  pooled linear luminance: on the fitted states the bench 0.97, the white paint 0.98, the floor 1.00, the shelf 0.94,
  the blank 0.98, the lid 1.00; on 4 held-out states the main classes within 7 %, while metal reflections (the tray
  0.54), the cavity lining and the floor behind the fence (0.47), which the probe light cannot occlude, are off by up
  to 2x and more; section 5). Depth: the classic renderer on a
  depth twin, the physics model's own geoms (groups 0-2) plus the hall's and the props' meshes within 4.5 m of the
  bench: on the cell the depth is the physics model's to the bit, the shoe's depth is its shell's (the skin follows
  it within a few millimetres), and the dressing reads like any surface. The sensor model and the encoding are the
  server's, unchanged. The render bundle (42 MB, one tar in the asset dataset, pinned by `environment/assets.sha256`)
  holds the hall, the skin template, the textures and the image light; `look.py` checks it against its manifest. Cost
  on 8 CPUs: 0.4-0.6 s a frame (front and side most), ~6-16 s to build the twins per episode, RSS 2.2 GB after the
  first episode and 3.2 GB after seven in one process (peak 3.9 GB) under the task's 16 GB.
* `episode.py`: the work order (seeded), the faults, the lacing topology check and the milestones.
* `server.py`: the API, the attach / clamp / lock logic in `physics_tick`, the camera budget and noise, the safety
  watch, the summary. `mode="practice"` additionally answers `reset` and `truth`; `privileged=True` (the oracle token)
  answers `get_state_privileged` and `reference_run`.

## 3. What the grader checks (`episode.py`)

Per pair, judged on the physical state when the box ships (frozen) or at the end of the episode:

| milestone | credit | check |
|---|---|---|
| lacing of each shoe | 0.30 + 0.30 | the lace polyline is intersected with every eyelet ring's disc; walking each tail from row 1 outward, the passes must alternate sides and every ring must be entered from the outside (row 2 IN on the other side, the lace back out through the throat slot, row 3 IN on its own side, ...: the "throat-return" criss-cross); rows are credited consecutively until the first miss; snug = lace between consecutive rows <= 1.5 x the straight distance (+ 3 cm for the dip under the flap and the throat return; a pulled lace springs back 0.5-2.5 cm through the ring when let go, a loose one lies at 2.5-5 x); 0 if an aglet was left inside the cavity: an end that passed in and was never seen back out (the work order latches, every 0.5 s, each end it sees clearly out of the shoe after its last pass; a finished end that later slides or falls back in, e.g. while the shoe is packed, does not count) |
| erected | 0.12 | the four wall locks engaged |
| packed | 0.13 | both shoe centres inside the box interior, sole down within 15 deg, headings within 30 deg of opposite, sole bottom on the floor |
| closed | 0.10 | the lid tray centred within 12 mm, aligned within 6 deg (mod 180), at wall-top height, resting, with the pair inside |
| shipped | 0.05 | closed, pair inside, floor centre across the outfeed line |

The work order is a laced pair in a shipped box: packed, closed and shipped are credited times the fraction of the
pair's shoes laced through every row (0, 1/2 or 1; snugness aside), so a pair boxed unlaced earns the erected credit
(0.12) only. (Before 2026-11-05 the box credits did not depend on the lacing: an Opus 5.5 run erected, packed, closed
and shipped in three of four episodes without lacing a shoe and read 0.33 = 0.64 of full credit.)

`work_score` = mean over the hidden episodes of (sum of credits) / pairs. An untouched cell scores exactly 0 (the
delivered lace passes only row 1; nothing is erected). A safety violation (a Panda link other than the pads / hook
pressing more than 30 N on the bench, the cradle or an object, or any arm-arm contact above 30 N, for three control
ticks in a row (a one-step spike of MuJoCo's soft contact is not a press), or an object faster than 3 m/s) voids the credit of the pair at work when it happens (`safety_void` in the pair record; the whole episode
was zeroed in an earlier draft, which made one 30 N touch of cardboard worth the whole window); a physics
fault ends the episode with the shipped pairs' credit only. The contract gate (`verify.py`) is the
one of `robotic_arm_full_assembly_so101_panda`: a regular `packer.py` with `solve`, the route gate, the static and
runtime network audit, the deprivileged run as `nobody` from a read-only snapshot, the stage lock, the late-write
gate; the run directories carry opaque tags (`ep01`), never the seed. Verification is separate (`task.toml`): the
grader runs in a fresh verifier sandbox from the task image with only the submission restored, so nothing else the
agent left reaches it; `environment/setup/setup.sh` does not run there, and `tests/test.sh` builds the grader's
root-owned copy of the givens at `/opt/arb` from `tests/setup_payload/` (a byte copy of
`environment/setup/payload/`: change both together, then `tests/given_hashes.json`) and the image's Panda.

## 4. The reference (oracle)

`solution/solve.sh` installs `packer.py` (one privileged call, `reference_run`) and the token. The controller
(`tests/reference/`) runs inside the grader with the exact state: `primitives.py` (the Rig: Cartesian / joint moves
through the cell's own controllers, seeded IK for wrist-flip poses, gripper / hook / clamp calls), `lace_skills.py`
(seat; per tail: the gripper picks the aglet from the bench and drops it IN through the row-2 ring on the other side,
the hook enters the cavity through the throat, catches the aglet under the flap, rises out through the throat slot and
draws the lace until the row 1 -> row 2 segment is snug, then lays the aglet on the bench; unseat back to the shoe's
own delivery spot), `box_skills.py` (erect, pack, lid, ship). It obeys the same rate limits, attach rules and physics
as a submitted program; its measured `work_score` is the anchor. Until 2026-11-06 the hidden seeds were drawn from OS
entropy per verify run, so `full_at` had to cover the oracle's seed-to-seed spread; since then the hidden episodes are a
fixed set (`tests/hidden/seeds.json`, the scoring paragraph below), the same six for every verify.

**State on 2026-11-05.** `grader_config.json` runs the full reference (`reference_lacing: true`). Measured on the host
with the grader's configuration on seeds 1-64 (one episode each, the call verify makes): mean work_score 0.695; both
shoes laced in 30 episodes, one in 26 (its box credits count half), none in 4 (the erected credit only), 4 voided by a
sustained contact (a pitched hand's forearm on the lid while folding the wall next to it, a finger on the shoe while
re-picking an aglet left on the flap, the hand on the shoe while picking an aglet beside it), 0 physics faults. Per
shoe the lacing succeeds about 80 % of the time when the shoe arrives upright and about half the time when it arrives
on its side (section 5). The verify statistic is the mean over 6 episodes. The four validate runs voided 5 of 24
reference episodes against 4 of 64 here (all by the contacts of section 5; the reference reads no wall clock, so this is
seed-to-seed spread or a host/engine difference not found), so full_at is sized for a 12-21 % void rate:
`tests/anchor.json` had value 0.695 and full_at 0.50, below which the 6-episode mean falls in 1.8-6.1 % of verify
runs (bootstrap of these episodes; the last validate read ratio 0.94). Six
episodes rather than four because with four the same risk needs a much lower full_at. A run that erects every box but
laces nothing reads 0.12 per episode, i.e. reward 0.35; lacing about one shoe per pair and boxing it reads full.

**State on 2026-11-06 (the reference reworked by the maintainers after the review of PR #48).** The seat carries the
shoe high to above the cradle, lets the tails hang straight and lowers it straight down, looking under the sole before
it lets go (the earlier sideways approach left a tail under the sole after every upright seat); a slip is noticed at
once and each tail gets up to four tries, each from where the tail lies; a bench aglet is grasped high in the pinch box
while the hand rises as the cell seats it; every move of either arm stops and backs off at 8 N on a link (12 N on a
finger pad), checked every tick; the shift of a righted shoe on the infeed tray, the shoe grasps and the wall folds are
planned clear of the tray, the neighbour shoe and the lid (a guard cannot stop a link that meets something at speed); a
lid tray lying in the way of a wall fold is lifted back onto the infeed tray; the box is pushed only with its lid
closed. Measured as above (the reference of `2e9191d`; one queue of `2eed6f9` and seed 1004, the one episode whose path
`2e9191d` changes, again), seeds 1-64: mean work_score 0.993, 63 of 64 episodes at 1.0, both shoes laced in 63, one in
1, none in 0, 0 voided, 0 physics faults; seeds 1001-1064: mean 0.968, 60 of 64 at 1.0, 0 voided, 0 physics faults. Per
shoe the lacing succeeds in 184 of 187 upright and 67 of 69 side-lying arrivals; an episode takes 423-608 s simulated
and 678-1568 s of wall clock on one core of the host (mean 942 s). The anchor was measured again with this reference, on
the fixed hidden set (next paragraph).

**The cell change of 2026-11-07** (the lid tray's delivery spot and the tails' keep-out around it; section 5). The
reference, unchanged, ran again with the fixed cell on 32 episodes (host venv, one core each): the first selection's 24
candidates (all 1.0 with the old cell) and 8 development seeds, the four whose lid the old cell threw off the tray
(1014, 1034, 1039, 1046) and the four with the new lid nearest a side-lying shoe (2436, 2644, 4900, 5385). 31 scored
1.0 (425-614 s simulated, 663-1198 s of wall clock, mean 832 s; one slip, recovered; the closest call a hand on a
shoe at 38.6 N for 2 ticks, seed 5385); seed 2644 was voided: an aglet that had slid under a lid wall dragged the lid
off the tray when its shoe was lifted (section 5).

**Scoring since 2026-11-06: the fixed hidden set and the anchor** (the second set, 2026-11-07). The owner's rule for
contributed tasks (2026-11-04): the oracle may use ground truth the agent cannot see, unlocked by the token whose sha256
lives only in `tests/hidden/`; a random task grades a fixed hidden set of configurations that the oracle completes; the
anchor is the oracle's full or near-full score with `full_at` near 1. Here:

* The set, `tests/hidden/seeds.json`: six 63-bit seeds, run in their order by `verify.py` (`hidden_set`; a missing or
  malformed set is a task error, never a run on other seeds). Before `e1f9b27` verify drew `stream.derive("episodes")`,
  which hashes a pinned seed list into other seeds: a pinned set would never have run.
* Its selection, `tests/hidden/selection.json` (format "selection 2", the rule in `tests/review/select_set.py`). The
  first selection (2026-11-06, `e1f9b27`-`85fa776`: 24 candidates, all 1.0, draws 0-5 kept, anchor 1.0 / 0.98) was
  superseded by the lid fix: its scene 4 started with the lid tray jammed on the infeed tray's rim (section 5). The
  second, on the fixed cell: the reference, the cell, the grader and the image's inputs frozen first (38 sha256s; against
  the first freeze only `tests/shoecell/episode.py`, `physics.py` and `given_hashes.json` differ, the reference is
  byte-identical; the image `pr48fix-task:fdec188`, `sha256:42897dfdf249...`, built once from `environment/` without its
  `setup/`, whose files did not change; `52bee23`); 24 candidates drawn from `os.urandom` at 2026-11-07T04:17:57Z and
  committed before any ran (`779d2d6`); each ran once in its own container of that image with no network, through
  `verify.run_episode` with the `solve.sh` submission (04:18-04:44Z); a candidate passes with work_score 1.0 and a
  physical start (the start-state check, added to the rule before this draw: no contact deeper than 1 mm, the lid tray
  flat and inside the rim, no lace inside the lid tray, nothing moving more than 2 mm in 1 s idle); the first six in
  drawn order that pass are the set (`8f876cc`). 23 of 24 passed (0 voids, 1 slip, 0 physics faults; 440-542 s
  simulated, 643-1124 s of wall clock; every start state physical: the lid 8.3-25.3 mm inside the rim, flat, the largest
  move in 1 s idle 0.12 mm); draw 1 was dropped (work_score 0.56: one tail came to rest in the reference's dead zone,
  below), so the set is draws 0, 2, 3, 4, 5, 6. The hollowing check (the kept scenes against all candidates):
  side-lying arrivals per pair 0 / 1 / 2 in 2 / 3 / 1 kept scenes, 12 / 9 / 3 candidates: not hollow (the rule calls a
  set hollow when a count a candidate has is missing from it). Narrower by the cutoff and the one drop: the kept shoes
  are 174.9-187.9 mm long (the candidates 171.2-208.7 mm; the dropped draw 1 is 202.3 mm), the side-lying shoe is A 3 /
  B 2 (candidates A 7 / B 8), upper colours black, navy, orange and teal (eight among the candidates).
* Determinism: the image and the engine are those of the first selection, where on four development seeds (17, 30,
  1001, 1040) the image reproduced the host venv's runs of the oracle report exactly (work score, simulated time, every
  event and the final qpos sha256; CPython 3.12.14 against 3.12.11, OpenBLAS with 2 threads against 1, the wire and two
  camera frames against an in-process call). In the verify each of the six episodes repeated its selection run (the
  same score, simulated time, ticks and per-pair detail).
* The anchor, `tests/anchor.json`: one full verify of the `solve.sh` submission in a fresh container of the image laid
  out as the separate verifier (`tests/review/regrade.sh`: tests/ root-only at /opt/ale/tests, only the submission
  restored, `bash test.sh` as root, no network, 8 CPUs; from the committed tests/ of `8f876cc`): all six episodes 1.0,
  work_score 1.0, reward 1.0, no violation, no internal error, no network event. `value` 1.0 and `full_at` 0.98 (every
  episode equal to its selection run; the margin for a last-bit difference on another host). A
  submission needs work_score 0.9751 for reward 1.0 (two decimals); boxing every pair without lacing (0.12 per episode)
  reads reward 0.12, lacing one shoe per pair and boxing it (0.56) 0.57; one shoe lost by the reference in one episode
  (mean 0.927) would read 0.95. The repository test `tests/test_shoe_lacing_and_boxing_hidden_set.py` pins the set to
  the selection, the start state of every candidate, the frozen files to their sha256, the cell's three copies to each
  other and to `given_hashes.json`, the lid spot's worst case, and the anchor to the lowest recorded verify.
* The renderer amendment (2026-11-07, `amendments` in `tests/hidden/selection.json`): after the set was selected the
  agent cameras moved to Filament (section 2). That changed frozen files (`cameras.py`, `physics.py`'s one argument,
  the new `look.py`, `given_hashes.json` for them and the manual, the Dockerfile's lavapipe and render-bundle layers)
  but nothing the reference or the scoring computes with: the physics model is never touched (the twins are copies),
  the reference's privileged run reads its two frames from the twins, and neither the scene's nor the sensor's random
  generator is drawn differently. The record holds the new frozen hashes, what changed (the repository test allows
  only these renderer files) and the check: the reference's full verify on the amended image (`pr48fil-task:26749c5`,
  run `pr48fil-verify-2`, `tests/review/regrade.sh` as before) repeated every kept selection run, the same score,
  simulated time, ticks and per-pair detail in all six (each 1.00; verify wall 6669 s on a host shared with other
  runs: the episodes took 1023-1298 s against 610-760 s, the cell's own simulation and frames 5.7-25 s of them). The anchor keeps both verifies; value and
  full_at are unchanged.
* What the agent sees: `instruction.md` says the hidden episodes use other seeds and that the cell builds their scenes
  with the same code (it said "fresh seeds of the same distribution" before; a selected set is neither). Nothing
  agent-visible names the set, its selection or the token (the repository test); the cell library's own mentions of
  its privileged calls (`server.py`, `client.py`, `cell_api.md`; the PR's, byte-identical to the grader's copy) are
  pinned there as they are.
* Time: the verify took 4009 s of wall clock (610-760 s per episode; its first 80 s beside the end of a
  camera-only localisation run of the maintainers). The first set's verify took 4644 s.

## 5. Known limitations of version 0.2

* The agent cameras' light is distilled, not path traced: one OVRTX probe at the work area lights everything as
  distant light, so nothing occludes it (the cavity lining, the floor behind the fence and under the racks, the bench
  legs read brighter than in the OVRTX demo; per-class gains bring the classes' means to the demo's on the fitted
  states), metal reflections vary more than the gains can follow (the infeed tray on held-out states 0.54 of OVRTX),
  and the panel's shadows are a spot's, softened. The skin is a render layer: its rounded edges differ from the depth
  (the shell's) by up to a few millimetres.

* The shoe is rigid; "snug" is a lace-slack criterion, not throat closure.
* The lace has no self-collision and no lace-lace collision (speed); a knot is not modelled and not scored.
* The hook can only reach eyelets within ~11 mm of the throat edge (its foot is 14 mm): the batch's eyelet inset is
  9 mm. Its J shape means it can only carry an aglet upward through a ring; the gripper does the downward passes.
* Two eyelet rows per shoe in this version (one throat-return pass per tail): each pass consumes about 13 cm of
  cord twice over (down into the cavity and back out through the slot), so two passes per tail exhaust the 0.64 m
  lace and draw the other tail back through row 1. A three-row batch needs a ~0.9 m lace (60 chain bodies, ~30 %
  slower physics) and a re-tuned reference: the intended next step.
* Why the throat-return pattern instead of the textbook criss-cross (the lace coming OUT through the next ring from
  under the flap): the hook can lift an aglet through a ring by only ~6 mm (its foot stays under the flap), so every
  such pass would end with the gripper taking the aglet off the hook tip right at the ring, where the hook arm's wrist
  stands; two Pandas cannot share that space (two rounds of oracle work ended in arm-arm contacts or an infeasible
  column). With every ring entered from above by the gripper and the hook only fetching the aglet out through the
  slot, the two arms work in turn at a safe distance and the hand-over happens in free space or via the bench.
* The cradle holds the shoe with its throat slot along y (toe toward the hook arm): the hook arm works over the toe,
  the gripper over the heel.
* The lace is tacked at row 1 (a weld between the shoe and the segment crossing the cavity, activated at delivery):
  without it the first lift of a tail drew the other tail back through row 1 on most seeds (the reference lost the
  delivered row on 4 of 5 seeds), and the "uneven tails" fault that relied on sliding through row 1 is retired.
* The lid is a separate rigid tray (a one-piece hinged-lid blank was tried: its flat footprint, 0.94 x 0.43 m, is
  out of a bench-mounted Panda's reach).
* The oracle's sequence is scripted and conservative; its timing sets the pairs-per-window budget. Hidden episodes are
  expensive (11-26 min of wall clock each for the reference, 16 on average; the oracle's verify of the fixed set took
  4644 s), so the fixed set has six episodes (`grader_config.json`): with the seeds drawn per run, six was the least
  that kept `full_at` well above the boxing-only level (section 4).
* The reference's weak spot (2026-11-06) is a tail that comes to rest where neither arm can pick it up. With the toe
  toward the hook arm, the tail that leaves a seated shoe on its R side can end within 0.22 m of the hook arm's base
  (the hook arm cannot reach the bench that close to itself) and more than 0.68 m from the gripper's base (beyond its
  top-down reach), and a hook-only aglet re-laid toward the gripper can end inside the gripper's 0.26 m. The seat lowers
  the shoe again up to three times while an aglet lies out of reach, a tail gets four tries and an aglet only the hook
  reaches is re-laid in the gripper's zone, but an aglet in that dead zone stays there: 4 of the 128 episodes of
  2026-11-06 lost a shoe (one lost both) this way; none of the first selection's 24 candidates did, one of the
  second's did (draw 1, 0.67 m from the gripper's base and 0.12 m from the hook's: dropped; section 4). Until
  2026-11-05 the weak spot was the side-lying shoe (its tails were dragged under the sole by the sideways seat); the
  reworked seat laces side-lying arrivals as often as upright ones (67 of 69 and 184 of 187).
* The reference's voids: none in the 128 episodes of 2026-11-06, none in the 48 candidates of the two selections; one
  in the 32 development episodes with the fixed lid delivery (seed 2644: a lid an aglet had dragged off the tray, in
  the lid bullet below). Every move stops and backs off at 8 N on a link, checked every tick, but a link that meets
  something at speed builds its press within one 10 ms tick, so the contacts
  found (the forearm on the lid at a pitched wall fold, a finger on the shoe while re-picking an aglet, the hand on the
  shoe beside an aglet, the forearm on the infeed tray and the hand on the neighbour after a righted shoe's shift) are
  avoided by planning, and a box is pushed only with its lid closed (pushed from where a failed packing had left it,
  near the gripper's base, the hand met the bench at 295 N). The closest call: a finger on the lid tray at 32 N for 2
  ticks (the rule voids at 3) in the joint-space line home from above the infeed tray after a righted shoe was set down
  beside the lid (seed 48, `box_skills._go_home`; that line is checked kinematically, but it runs even when the check
  finds it blocked at both lift heights).
* The lid tray's delivery spot, fixed 2026-11-07 (a cell change, so a new hidden set and anchor: section 4). Until
  then the lid was drawn at tray centre y - 0.14 +- 0.012, turned +-0.2 rad, and set down 2 mm above the plate: the
  bigger lids' near wall came down inside the tray's near rim (y = 0.016-0.024). Over 952 seeds (1-64, 1001-1064, the
  first selection's 24 candidates, 2001-2800; `scan.py` of the maintainers' notes): 758 lids were set down with a wall
  in the rim, 379 then rested tilted 3-6 degrees on it, 283 were moved more than 3 mm by the reset, and 36 (3.8 %) were
  thrown off the tray and stayed jammed on its rim: 6 contacts, the deepest 5.0-6.2 mm, carrying 216-222 N against the
  lid's 0.66 N weight, the lid still sliding at 0.45-0.81 m/s at t = 0. Hidden scene 4 of the first selection was one of
  them (5.5 mm, 219 N, 0.55 m/s at t = 0; in 20 s idle it slid 9 cm toward the blank and met it), so there the
  instruction's "fetch the lid tray from the infeed tray", the manual's "the lid tray (near (0.40, 0.12), either way
  round)" and describe()'s "the lid tray on the infeed tray" were false (the reference lifted it back and scored 1.0).
  Moving the spot alone does not fit: between the rim and the shoes' side-lying arrivals (a shoe on its side reaches
  8.8 cm toward the lid) there is about 0.25 m, and the biggest lid turned 0.2 rad needs 0.25 m (y - 0.12: 135 of the
  952 still set down in the rim, one jammed; y - 0.10: none in the rim, but lids set down 27 mm into side-lying shoes
  and 9 thrown off). Now the centre is drawn within +-3 mm of y - 0.125 and the turn within +-0.03 rad (the x jitter,
  +-3 cm, and every other draw of the work order are unchanged: the same uniforms of the same stream), which keeps even
  the biggest lid at least 3.8 mm inside the rim (pinned by the repository test), and the tails' keep-out box is the
  lid's footprint at its spot plus 1 cm (it was the nominal spot's plus 3 cm, which the moved lid would have made the
  start of a side-lying shoe's tail four times as often, and such a tail is laid under its shoe). Over 3952 seeds (the
  952 and 3001-6000): no lid set down into the rim, none tilted or moved more than 3 mm by the reset, the nearest shoe
  at least 6.9 mm away; a side-lying shoe's tail laid under it: 21 tails of 3814 over the 952 seeds (50 before). What
  remains is the frictionless cord sliding in the 0.3 s settle after the delivery: lace touches a lid wall from outside
  in 109 scenes (at most 0.09 N), and in 6 an aglet ends inside the lid's footprint, under a wall (2 of the 3952 with
  the old delivery, 4 with a 3 cm keep-out for the tails that start beyond it: not adopted, it changes the tails of most
  scenes); `cell_manual.md` says so. Such an aglet can drag the lid: in seed 2644 (32 development episodes of the
  reference with the fixed cell, section 4) lifting its side-lying shoe pulled the lid off the tray onto the bench
  beside the box, and the reference, carrying the lid from there toward a placement without a joint solution, pressed
  the forearm on the infeed tray (381 N): the pair was voided (with the old delivery seed 2644 scored 1.0). The
  hidden set's rule drops a scene that does not start physical, this one included (section 4). A lace pressed more
  than 1 mm into the plate (2 of 3952) is as before. The reference is unchanged: its lid recovery
  (`box_skills._shift_lid_away`) stays for a lid a wall fold meets, and its `LID_TRAY_IN` comment still describes the
  old delivery.
* Three-row batches need a longer lace (section 2).
