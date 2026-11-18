# driving_safety_supervisor: grading notes for maintainers

The grader (`verify.py`) runs as root in a separate verifier sandbox. The agent never sees this folder. This note records how the hidden set, the reference expert and the anchor came about, and how to redo each step.

## Owner direction (2026-11-06)

- The token-privileged reference expert must complete every graded episode: no collision, no hazard hit, no red light, no abort, the route completed. The graded episodes are therefore a fixed hidden set, selected once by a pre-registered rule, and the anchor is measured on it.
- The staging fixes S1 to S4 (below) are in. They make every graded episode completable and remove no difficulty. A review found that the first S4, which always moved shoulder_pass's warning sign to the lane's left, made every shoulder pass shallower. S4 now keeps the sign at the lane centre, and the set was selected again (see "The first selection").
- The excused-wait claim for walkers is deleted. It would only have removed a pace penalty that the expert pays too, so it could not raise the difficulty. The only excused wait is time stopped at a red or yellow light, up to 90 s an episode.
- The pace formula stays as it is.
- `full_at` follows the design's rule: 0.98 if the verify repeats the selection to 4 decimals, F1 otherwise.
- Agent-visible docstrings do not name the oracle token. The practice `"privileged"` feature stays.
- A test runs once: one selection run per candidate, and one anchor verify.

## The fixed hidden set

`hidden/episodes.json` has the format string `driving_safety_supervisor fixed set 1` and holds 24 episodes. Each episode has an id, a town, a variant, a kind, a seed, a distractor count, a lock key and a layout digest. The episodes are in grading order, which is `drive_variants.graded_cells` over `grader_config.towns`: Town04 (7 episodes), Town05 (10, three of them shoulder_pass), then Town10HD (7).

What `verify.load_set` checks:
- the format string;
- one episode per cell, in grading order;
- each kind matches the one its variant stages;
- seeds are distinct integers (not bools) in [2^32, 2^62);
- distractor counts are 0 to 2;
- lock keys and layout digests are 64 lowercase hex digits;
- lock keys and ids are distinct.

Any defect is a task error, and its message names no value of the set.

At every reset, `LayoutChecked` compares the bridge's `truth["layout"]` with the recorded digest. The digest covers the town, the route's lane keys, the spawned blueprints in spawn order, and the generator's state after every draw of the reset. A mismatch is a task error, never a silently different episode.

Each episode draws its probes from its own stream (`seed_rng(entropy=lock_key)`). Specs travel only over root's pipe to the bridge, and the grader's log prints no seed.

### Selection (`review/select_set.py`; the rule is its module docstring)

1. **Freeze.** The following files were committed first: the bridge (both copies), `drive_protocol`, `drive_variants`, `episode_score`, `verify`, `sim_client`, the expert, `footprints.json` and `solve.sh`. The set was selected with them as they are at the freeze commit `f5b1258`. `draw` refuses to start on a tree where any of them differs from HEAD; `run` and `make` refuse once any of them differs from the draw's record. The repository test also fails when any of them changes, because the set would then have to be selected again.
2. **Draw** (2026-11-06T22:43:35Z). 96 candidates, c096 to c191: four rounds (4 to 7) of the 24 cells, all from `os.urandom`:
   - the seed is 8 bytes, little-endian, mod 2^62, drawn again below 2^32 or on a repeat;
   - the distractor count is one byte mod 3;
   - the lock key is 32 bytes.

   The ids go on from the first selection's last (c095), and no candidate shares a seed or lock key with it. `hidden/selection.json` was committed (`2ef9086`) before any candidate ran.
3. **Run.** Each candidate ran once, exactly as graded. `review/run_episode.py` calls `verify.grade_episode` in a container of the task image. The run is a token run of the `solve.sh` submission, through netguard, with the bridge running as `carlasim` from `/run` and the candidate's own lock key. There is one simulator per container, and the world is reloaded before every episode. Round r runs only the cells that do not have a kept candidate yet.
4. **Keep.** A candidate is kept if all of these hold:
   - termination is `completed`;
   - no abort;
   - walker, vehicle and static collisions are 0, and hazard collisions are 0;
   - red lights are 0;
   - the box clearance to every actor is at least 0.15 m;
   - the timeout clock (time minus excused time) is at most 0.85 of the time limit;
   - no walker within 25 m of the car moves faster than 4 m/s over the ground on two consecutive steps. People walk at 0.9 to 1.7 m/s in every staging, so one that fast has been launched by the physics, and a car a few metres further on could be hit through no fault of its driver. A single fast step is allowed: it is how the staging putting a turner back on its spot by hand looks in the block.

   Pace, harsh brakes and the score are not part of the rule.
5. **Result.** Round 4 ran the 24 first candidates, and all 24 were kept; 72 did not run.
   - The kept scores range from 0.7251 to 1.0000, with a mean of 0.8998.
   - The box clearance is at least 0.342 m, the timeout clock at most 0.665 of the limit, and the slowest reply 0.017 s. Three runs (c102, c110, c111) waited 28 to 30 s at a red light, which is excused.
   - Two crowd runs (c102, c112) have one fast step each, 18.3 and 10.6 m/s: the turner put back on its spot by hand (`returned_by_hand` is 1 in both).
6. **Hollowing check** (`select_set.py report`): no flag, since nothing was dropped. The thresholds are:
   - no variant drops more than 1/3 of the candidates it ran;
   - no cause is shared by two drops;
   - no value of a categorical staging parameter holds two drops.
7. **Replacement rule.** A kept episode that fails the keep rule in an anchor verify is replaced by its cell's next kept candidate. A second such failure is an expert defect: fix it, freeze again and draw a new selection. In the one anchor verify, every episode met the recorded clauses. The grader measures neither clearance nor walker speeds, so those clauses could not be checked there.

`hidden/selection.json` holds every candidate: its draw, its run (truth, staging record, score, layout, probes, clearance, the fastest walker near the car and its longest fast run, speed limits, slowest reply, wall time), the clauses it failed, the cause found, and its outcome (kept, dropped or not run). Its `supersedes` entry names the first selection's record and its sha256.

### The first selection (superseded)

- It was frozen at `c74c883` and drawn at 2026-11-06T20:55:23Z (c000 to c095). It kept c000 to c023, with c027 in place of c003, which was dropped on the clearance clause (0.118 m to an overtaker 1.09 m into the lane). Its anchor was 0.9143 with `full_at` 0.955. The whole record is `hidden/superseded/selection-1.json`.
- A review found two defects. The set was then selected again from a new draw, never patched:
  - S4 then always moved the warning sign 0.7 to 0.9 m to the lane's left. A centred sign leaves at least 1.47 m to spare on every Town05 spot, so the move made every shoulder pass shallower.
  - The run of c006 (Town04 crowd_against_light) had a crowd member launched by the physics. Another member of the group walked into it as it stood in the car's lane, and it flew toward the car at about 15 m/s and back, missing it by 1.97 m. A re-run of that seed launched it again at the same moment (31 m/s for 6 steps at t = 5.5 s).

## Determinism

- **Bridge.** Synchronous settings are applied before every load. The world is reloaded for every episode. Physics substeps are explicit: 0.01 s, at most 10 per step. Collisions and camera frames are taken tick by tick on the main thread, with exactly one camera frame per tick. The layout digest is computed at every reset. The old code carried state between episodes: run first and third in one process, the same episode differed from tick 0. This one does not: the two runs differ no earlier than two fresh processes do (`review/determinism.py`).
- **Residual.** CARLA's vehicle physics splits by one float ulp about 10 ticks into an episode, even in a fresh process, and diverges from there. Verifies are therefore not bit-identical.
  - The selection runs and the anchor verify agree on 9 of 24 episode scores to 4 decimals. The rest differ by at most 0.0470 (mean 0.0087).
  - In development, a repeat of 30 episodes moved scores by a mean of 0.011 and at most 0.080.
  - Layouts are identical across processes and GPUs.
- **Pinning.** Neither the CARLA tarball nor the requirements are pinned. A rebuilt image that lays an episode out differently becomes a task error through the digest, but drift in physics or the planner would be silent.

## Staging changes (`drive_variants.py`, grader only)

- **S1 dropped_cargo.** The truck stays on its handbrake until it has stood still for 1 s and the car is within 30 m. Every piece is drawn at staging. Pieces fall on a scripted, physics-off path and rest inside the car's lane, so the next lane stays open.
- **S2 crowd_against_light.** A turner who has not reached the pavement within its out-distance divided by its speed, plus 2 s, is put back on its spot. Crossers are marked done once across.
- **S3 reversing_car.** `left_free` walks the left lane from 60 m behind to 30 m ahead, across road ids. For an oncoming lane it also refuses while a car there would arrive within 8 s.
- **S4 shoulder_pass.** Two parts:
  - The warning sign stands 8 m before the crash at the lane centre, as the author staged it, wherever the car has at least 1.0 m to spare passing it on the shoulder. The spare width runs from the sign's right side to the shoulder's outer edge, less the car's 1.836 m. The sign's half-width is the larger of its box and its measured footprint: 1.436 m for the trailer, 0.663 m for the small sign. Only where the spare width is less does the sign stand 0.7 to 0.9 m left of centre, like the wrecks. Every Town05 spot leaves at least 1.47 m, so every graded sign is centred. The left lateral is drawn either way, so no other draw depends on the choice. The staging records the spare width as `sign_room`.
  - The spot's shoulder must be clear of the map's poles, signs and traffic lights at a car's height, from 30 m before the spot to 40 m after it, short of the shoulder's outer 0.3 m. Town05 has speed-limit signs standing on its shoulders, 0.6 to 1.9 m in from the lane edge. CARLA keeps them as map objects, which the ground-truth block does not list. Beside a centred trailer the pass runs up to 0.8 m deeper onto the shoulder than beside a sign on the left. In development, passes beside centred signs hit these speed-limit signs in 4 of 40 runs, and one of them, 0.58 m in, is hit by any shoulder pass. 549 of Town05's 681 spots pass the test.
- **S5.** Every staging records what it drew in `truth["staging"]`, which never reaches the wire.
- **S6.** Kerb props and pavement people stand beyond the road's real edge, never on a shoulder or parking lane.

## Grader fixes made just before the freeze (both bridge copies)

- **Red lights count once per crossing.** The Tracker used to count a red light on every tick the car moved through the junction while still in the light's trigger volume: 12 for one run. A light now counts once for the crossing. The score's ×0.7 per red light (up to 3) uses this count.
- **`lights` in the privileged block.** The block lists the traffic lights whose stop line lies on the route block's stretch, each with its stop point and state. The planner sees a light only a few metres before its line, too late at the expert's pace: in d7-crowd_-0 (a dev seed) the next junction's light, inside the route's tail, was run. The block is computed only for a privileged spec. Nothing a graded supervisor sees changes, and the layout digest does not involve it.

## The reference expert (`solution/solver.py`)

The expert submits the oracle token (`solution/reference/oracle_token`; its sha256 is in `oracle_token.sha256`), so every observation carries the bridge's ground-truth block. The block contains:
- vehicles within 250 m, with their reversing lights;
- walkers and props within 80 m;
- the route's lane geometry;
- the lights on the route;
- `red_light` and `speed_limit`.

The expert never looks at the camera. It reacts to the probe's pasted road user, which the probe copy of the block lists as id −1, so the probe lock holds.

The expert is pure Python with no randomness and no clock. Its module docstring and constants describe how it drives:
- probe bookkeeping that does not depend on the order of a probe pair;
- a swept-path rollout of the car's lagged bicycle model against measured footprints (`solution/footprints.json`);
- feasible ramps;
- re-checking and turning back;
- handling of backing and stopped cars, and of walkers;
- gearbox-aware pace at the posted limit.

Two rules came later. Just before the first freeze, it learned to stop at the line of a red light on the route ahead, and of a yellow one when 5 m/s² is enough at the moment the light turns yellow. With the second freeze, it passes a closure on a shoulder or parking lane slowly (4 m/s) and 0.4 m from the wrecks and the sign, when its usual 0.85 m does not fit inside the shoulder's centre. That happens beside a centred trailer, mostly on a bend.

## Anchor

`anchor.json`: 0.8971378275677083 with `full_at` 0.955. It comes from one verify, the separate verifier's way (`review/regrade.sh`, run `pr24t2-g5-spy`, 435 s): 24 of 24 completed with no infraction, and the lock held (46 of 111).
- **Why `full_at` is 0.955.** The run is not deterministic (see above), so `full_at` is F1 = ⌊200 · (1 − max_k s_k / Σ s_k) / 0.995⌋ / 200. Losing any one episode still rounds to reward 1.00; losing two gives about 0.95.
- **Why the reference stays below 1.** Completion ends at 97 %, pace counts from a standstill, and every full stop costs pace and a harsh brake. The ratio absorbs this, because every submission meets the same episodes.

The same verify ran the leak audit's spy beside the expert, as the agent's account under netguard:
- **Refused:**
  - every AF_INET and AF_INET6 socket, to the internet, to CARLA's 127.0.0.1:2000, to the Traffic Manager on 2050, to the other listening ports and to ::1;
  - raw ICMP, packet and vsock sockets;
  - name lookup;
  - every read of `/opt/ale/tests`, `/run/drive-bridge-*` (the stagings) and `/tmp/drive-simlog-*`;
  - the environ, maps, mem and fd of the root and `carlasim` processes.
- **Readable:**
  - an AF_NETLINK route socket opened; netguard denies the IP families, and nothing listens on netlink;
  - the guard's own folder;
  - process command lines, none of which holds a seed, lock key, digest or the token's digest.

`ptrace_scope` was 1. The first selection's verify ran the same audit with the same outcome.

## What stays open

- **CARLA server crashes.** The server can die (signal 11) in a reset late in a long server life; it was seen at the 31st reset. `sim_client` then starts a new bridge, which builds the same episode again. A server that dies during an episode is still a task error. A verify is 24 resets.
- **Walker launches.** CARLA can launch a walker at about 31 m/s for half a second. In c006 another member of the group walked into a turner standing in the car's lane; in a development debris episode it was one of the people walking along the pavement. In every repeat of the three seeds that showed one, the launch came again at the same moment. The keep rule drops a candidate whose run shows one within 25 m of the car, and no kept run shows one. A launch that happens only in another driver's run, or more than 25 m from the expert's car, is not excluded.
- **Map objects.** The block lists vehicles, walkers and props, but not the map's poles, signs and kerbs. S4 keeps them off shoulder_pass's shoulders; elsewhere the expert does not see them. In one development run it also stood on a shoulder beside the wreck behind a car queued at a red light ahead, and it did not set off again when the queue left (a timeout). The keep rule would drop such a run.
- **Overtaking near the top of the range.** The expert's nudge leaves little room for an oncoming overtaker near the top of the intrusion range. The first selection dropped one such candidate; the second dropped none.
- **Speeding headroom.** The score has no speeding penalty, so an agent that drives above the posted limit can buy back some pace.
- **Kit docstrings.** The kit's docstrings still speak of fresh seeds (`robotics_grader/seeds.py`). The kit stays byte-identical to the shared one.

## How to redo it

All commands run from this task folder. On apollo, CPU work goes under `taskset -c 16-31`. CARLA runs only in `pr24t2-*` containers of the task image with `--network none` and one GPU that has at least 8 GB free.

```
python3 -B tests/review/dev_batch.py make --out SPECS.json --per-variant 8 --rng N        # dev seeds < 2**32
python3 -B tests/review/dev_batch.py run --specs SPECS.json --out DIR --gpus 0,1 --cpus 16-23,24-31
python3 -B tests/review/dev_batch.py report DIR
git mv tests/hidden/selection.json tests/hidden/superseded/selection-N.json                # a new selection only
python3 -B tests/review/select_set.py draw --selection tests/hidden/selection.json \
    [--first-round R --supersedes tests/hidden/superseded/selection-N.json --why TEXT]   # commit it, then:
python3 -B tests/review/select_set.py run --selection tests/hidden/selection.json --round R --out DIR
python3 -B tests/review/select_set.py make --selection tests/hidden/selection.json --out tests/hidden/episodes.json --causes CAUSES.json
python3 -B tests/review/select_set.py report --selection tests/hidden/selection.json
ALE_HOME=/tmp/h bash solution/solve.sh   # then cp tests/review/spy/{spy.py,run.sh} /tmp/h/submission/ for the audit
CPUSET=16-23 bash tests/review/regrade.sh 0 pr24t2-verify /tmp/h/submission OUT
```

Any change to a frozen file invalidates the set: freeze again, draw a new selection, and measure the anchor again.
