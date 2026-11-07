# fault_adaptive_spacecraft_docking: grading notes

Notes for maintainers and reviewers. The agent-facing rules are in `instruction.md` and in the documents it points to
(`environment/setup/payload/given/`, staged at `/home/user/given/`: the plant's datasheet `model.md`, the sensors'
`camera.md`, the target's `target.md`, and `frames/`, sample camera frames whose PNGs come from the image's task
assets); the simulator is `env.py` in this folder, byte-identical to the reference's copy
(`solution/privileged/spacecraft_docking_env.py`), and the docking camera's renderer is `camera.py`. The agent gets no
copy of either. This folder is the verify stage: it is absent while the agent works and root-only at verify time.

## The owner's direction (2026-11-05)

The anchor is the measured score of a reference that plans from ground truth the agent never gets, on a fixed set of
hidden episodes it completes, so that the anchor sits near full and a controller that identifies the plant from its
observation has real headroom. The task author's identifying controller stayed as the non-privileged reference (until
round 2, below). What the agent had to do was unchanged, apart from the instruction's account of the fixed set, the
network and the time budget.

The same day the owner removed the practice tools, by the rule applied to PR #31's round 3: a spacecraft with unknown
faults is docked zero-shot, so the agent gets neither the grader's simulator nor a practice grader on visible seeds of
the graded distribution. Until then the payload held a byte-identical copy of `env.py` and `practice_grader/grade.py`
(five visible seeds), and the instruction called the simulator's source the complete statement of every range. From
then the payload held one document, `given/model.md`, which stated env.py's model exactly (round 2 replaced it).

## Round 2 (2026-11-07): the chaser's sensors and a datasheet

Opus 5.5 scored 0.60 on round 1 (28 of 48 docks): it rebuilt the simulator from round 1's `model.md` and tuned its
controller on hundreds of seeds. The owner raised the difficulty two ways: a datasheet-level plant document instead of
an executable one, and camera images instead of the numeric delayed pose, with the sensors where a real chaser carries
them for rendezvous and docking. The oracle keeps its ground truth through the token, so its results and the anchor do
not move; the agent keeps a limited perception, now real sensors; deployment stays zero-shot. The contract is the
spec in the maintainers' notes (`docs/research/20261107-pr10-difficulty-r2/spec.md` of the dev workspace).

**What stayed byte-identical:** `env.py` (sha256 c2900397), the oracle (`solution/privileged/oracle.py`, 84a75a0b,
whose docstring still says "the 16-number vector": it reads only the block either way), the fixed set and its lock
streams, the kit, `nonet_exec.py`. So the world digests, the selection and the reference's flights stand.

**The sensors** (`camera.py`, documented to the agent in `given/camera.md` and `given/target.md`, whose JSON blocks a
repository test compares with `camera.CAMERA_SPEC` and `camera.TARGET_SPEC`):

- the docking camera DOCK on the docking face, its lens at (-0.082, 0, -0.115) m in the body frame (0.115 m below the
  probe axis, as Soyuz's TORU camera sits beside its probe), looking along the docking axis: a 640 x 480 pinhole,
  fx = fy = 320 px (90 x 73.7 degrees), monochrome, linear, 8-bit, at a fixed exposure (EV100 15), read noise 0.8 DN
  and shot noise at 39 e-/DN; one frame per step, delivered with the episode's fixed 1-3 step latency and its capture
  stamp;
- the star tracker: the craft's yaw now, Gaussian noise 2e-4 rad;
- the IMU, the fuel gauge, the clock and the boom's encoder (now the target's telemetry), env.py's own values;
- on the target, LED fiducials: the docking target DT (four flush LEDs in a 50 mm square and a standoff LED 30 mm in
  front, enlarged from the spec's 40 / 20 mm by the evidence's pre-registered rule) on the drum below the port at the
  camera's height, the way the camera sits below the probe, and three LEDs on the boom tip (BT), 0.144 m apart at the
  drogue's rim; the sun's disc, composited (the skybox carries none).

**The observation** (`verify.Sensors`): env.py still makes its 16 numbers with all its draws, and the wire carries
`{"imu", "fuel", "episode_fraction", "boom_encoder", "star_tracker_yaw", "frame"}` instead. The frame is a render of
the pose record (env.pose_history) env.py's delayed pose came from, at the same stamp; it is `null` exactly when
env.py's pose was not fresh (B1, B2, the re-key's L - 1 observations, the 3.5 % dropouts), so every blackout keeps its
bite. The star tracker is `null` exactly during the re-key's restart (the same L - 1 observations), so the re-key's
29-41 observations remain IMU-only, as in round 1. Nothing here draws from env.py's generator (a repository test steps
two simulators side by side). Noise: a frame's per-pixel noise is keyed by (the set's key, the seed, the stamp) alone
(numpy PCG64), the star tracker's by (key, seed, step) (BLAKE2b): no history dependence, and a probe copy has the same
noise field.

**The probe copy** draws dx, dy and dyaw exactly as round 1 (the lock streams are unchanged) and renders the same
record with the craft at (x + dx, y + dy, yaw + dyaw), the same noise; the star tracker's copy is turned by dyaw; a
request without a frame or a star-tracker reading has a copy without it. In a token run the block moves by the same
draws as before.

**The grader's rendering** (`camera.Camera`): MuJoCo 3.16's Filament on Mesa lavapipe (CPU), post-processing off (linear,
no temporal dither: the same state renders the same bytes, in one process and across processes), one context and one
model in verify.py's main thread, glibc's arenas capped at import. The scene is the render-realism step's bundle
(`review/build_camera_bundle.py` from `pr10_render/bundle_v3`: the NASA Wind drum with its docking well, the boom and
drogue, textured PBR materials at 512 px with the fitted per-material gains; the e1022 v3 light rig: the sun at
azimuth 40, elevation 36, 118,422 lux, the OVRTX probe's image light at 0.3772, the Earth skybox), plus the fiducials
and, for its shadow, the craft's bus as a box with six bells at the nominal geometries' places (never an episode's
installed ones). The look's +0.5 EV (the demo video's exposure) is not applied: the camera's exposure is its own. No
plumes (cold gas is close to invisible; a drawn plume would show which nozzle fires). The bundle is 15.3 MB, root-only
in the image at `/opt/fault-docking-render` (mode 0700): with it and `camera.py` an agent could render the grader's
frames. Until the owner approves the upload to the benchmark's Hugging Face dataset it is staged locally in
`environment/assets/payload/render/` (gitignored), with the sample frames in `environment/assets/payload/given/frames/`
(`review/make_sample_frames.py`: four dev episodes, rendered in the task image).

**Transport and budget.** A frame travels inline, base64 in the request line (a line is 409,671 bytes). The kit's
`Solver` writes a request before it starts the request's deadline, so a controller that stopped reading could have
stalled the grader: `verify.BoundedStdin` gives the controller's stdin a 1 MB pipe, a non-blocking descriptor and the
request's deadline (30 s) for the write, past which the write fails as a closed pipe and the kit aborts the episode.
`GuardedSolver` sums the controller's reply time from the moment each request's write begins to its reply
(`reply_s_total`); past `reply_budget_s` (1800 s) that request's episode and every later one abort
(`reply_budget_spent`). The instruction's "30 minutes" is this rule; the grader's own time (stepping, rendering) is not
charged.

**The render canary.** Before episode 0 the grader builds the renderer and renders a fixed dev state
(`camera_canary.json`, recorded in the task image); `camera_canary_match` (whether the RGB render's sha256 equals the
recorded one) and `camera_canary_s` go into the metrics and never score. A renderer that cannot start is a task error
("the camera cannot be rendered here"), like the network guard.

**What the agent no longer gets:** the equations, the update order, the noise and dropout laws and the distribution
families (`given/model.md` is a datasheet: The agent's documents, below), and the numeric pose.

## The fixed episode set and how it was selected

`hidden/episodes.json` holds the hidden episodes: 48 seeds of 64 bits (each at least 2^32; the former practice seeds
were below 2^13), each with the sha256 of the world `env.reset(seed)` builds, and the hidden key of the
match-lock streams. The same 48 seeds run in the same order for every submission. `verify.load_episodes` checks the
format, the key, the seeds and every world digest before anything else runs; any defect is a task error whose message
names no value of the set. The digest formats every float with 13 significant digits (`format(v, ".12e")`), so a
last-bit libm difference on another host cannot fail it, while any change to env.py's draws does.

The selection was pre-registered (`review/select_set.py`, the rule in its docstring):

1. **Freeze.** The reference (`solution/privileged/oracle.py` and its simulator copy, `solution/solve.sh`) and the
   grader (`verify.py`, `env.py`) were committed first (08ccda6); their sha256 are in the set's `source`. Since then
   only `verify.py` changed, after the set was made: its network guard fails closed (The network guard, below), and
   round 2 replaced its observation by the sensors' (the lock's draws, env.py's draws and the oracle's input are
   unchanged). The episode loop the selection ran is the same in effect, and the reference's verify reproduced the
   anchor to the last bit after each change (The anchor). `select_set.py run` and `make` compare against the hashes recorded at the draw, so they
   refuse to extend this selection: a new selection starts with a new draw.
2. **Draw.** 64 candidates from `os.urandom` (8 bytes each) and the probe key from `os.urandom(32)`, written to
   `selection.json` before any candidate ran (2026-11-05T16:31:23Z).
3. **Run.** Every candidate through `verify.run_episode`: a token run of the `solve.sh` submission, a fresh program per
   candidate, its own lock stream, the full 400 steps, in a CPU container of the task image.
4. **Keep.** A candidate is kept when it docked without an abort; the first 48 kept, in drawn order, are the set.

62 of 64 docked. Kept: draws 0-28 and 30-48. Dropped: draw 29, undocked by all 266 policies from its planning state,
re-key on time; cause not identified (its episode ended in a collision at step 395: The two references). Draws 49-63
ran after the cutoff and are not used (draw 58 among them timed out). `hidden/selection.json` records every candidate:
its outcome, the policy the reference chose, how many of the reference's 266 policies dock from its planning state,
and its world (the hidden draws below).

**Hollowing check.** The kept 48 span every hidden range of `instruction.md`: the first event at steps 59-103
(range 58-103), the re-key's earliest step 108-128 (108-128) and deadline 148-165 (148-165), its proximity 1.181-1.299 m
(1.18-1.30), the four degraded scales 0.352-0.497 (0.35-0.50), 0.561-0.697 (0.55-0.70), 0.457-0.596 (0.45-0.60) and
0.651-0.796 (0.65-0.80), the boom's amplitude 0.100-0.157 m (0.10-0.16) and frequency 0.182-0.278 rad/s (0.18-0.28),
the start radius 2.870-3.648 m (2.85-3.65) and bearing from the port -1.444..1.414 rad (+-1.45), the target's slew rate
before the re-key -0.1229..0.1228 rad/s (+-0.125: 0.0021 and 0.0022 rad/s short of the two limits) and after it
-0.0437..0.0446 rad/s (+-0.045: 0.0013 and 0.0004 rad/s short), with the smallest speeds 0.0612 and 0.0203 rad/s
(0.060 and 0.020). The outward authority the re-keyed plant has at the docking pose in the initial swirl pattern (the
largest outward acceleration with no tangential and no yaw acceleration, the bias included) runs from -0.045 to
0.228 m/s^2 in the kept set (negative in 9 kept draws and undefined in one, all of which dock), and 38 to 64 of the 64
swirl patterns have outward authority. The one drop lies inside all of these ranges but one: its moderate scale of
the first event (`secondary_failure_scale`, 0.551) is below the kept 0.561-0.697, though inside its range 0.55-0.70;
that scale acts from the first event (step 89 in draw 29) until the re-key replaces it (step 115), before the
reference's planning state. None of the other quantities tells the drop apart (`review/select_set.py report`).

## One lock stream per episode; determinism

Each episode's match lock (`ProbeLock`: a probed step with probability 0.06, three draws for the copy and one for the
order) draws from `seed_rng(entropy=(key << 64) | seed)`: the same probes for the same submission every time, whatever
ran before. Nothing in a verify draws from OS entropy or the clock (only the per-request deadlines and the reply
budget are wall clock). The sensors' noise is keyed too (round 2): a frame's by (the set's key, the seed, its stamp), the
star tracker's by (key, seed, step).

The privileged reference is deterministic on the set: the selection runs, three fresh-container verifies
(`review/regrade.sh`; the third on 2026-11-06, with the network guard failing closed) and the in-process replays
(`review/oracle_batch.py`) gave the same 48 episode qualities to the last bit, mean 0.9835537203921798. It is pure
Python (no numpy, so no BLAS kernels and no tie-breaking in `argpartition`); on 20 dev seeds its replies under host
CPython 3.12.11 and the image's 3.12.14 are identical, and a program serving 48 episodes in a row answers exactly as a
fresh one per episode. Before 3.12 the builtin `sum` is not compensated, so fuel's last bit differs; the grader
always runs 3.12 or newer.

## What the set fixes, and what it does not

A seed fixes every hidden parameter of its episode: env.py draws them all in `reset`, and `step` never draws. After
the reset, randomness is only the sensor noise and the dropouts in `_observe`. Before the re-key the blackout
schedule is fixed by the seed, so every controller meets the same IMU noise and dropouts; the re-key fires when the
controller comes close enough, and its own blackout shifts env.py's noise stream, so after the re-key two controllers
see different IMU noise and dropouts on the same seed. The camera's and the star tracker's noise do not depend on
history (keyed by stamp and step), but a frame shows the controller's own trajectory. The selection therefore vouches
for the reference's own trajectory: another controller meets the same world, not the same episode. The reference
reads no observation, so its success does not depend on the noise.

## The privileged channel

`solution/solve.sh` installs the privileged reference and the reference token (`.oracle_token`; this folder keeps
only its sha256, `hidden/oracle_token.sha256`). `verify.token_state` reads the token before any submission code runs: a
regular file, opened without following a link and without blocking, at most 4 KB, compared by `hmac.compare_digest`.
A token that is there but wrong (a wrong value, a link, a directory, a FIFO) is a mismatch and is otherwise ignored.
The metrics record `privileged_mode` and `oracle_token_mismatch`; nothing agent-visible names the channel (a repository
test checks the instruction and every file under `environment/`).

With the token, every observation is the sensors' object with one more key, last:
`"privileged": {"format": ..., "state": {...}}` (round 1: `{"vector": <the 16 numbers>, "privileged": ...}`; the
oracle reads only the block, so either shape serves it).
The block (`verify.privileged_block`) is the simulator's current truth, named as env.py names it: every quantity
`env.step` reads that is in effect now (`PRIV_STATE`, `PRIV_ROUTING` once the first event has fired, `PRIV_CONSTANTS`).
It never carries the pre-drawn future (the event times, the re-key's plant, its slew, the encoder polarities) nor
anything of the observation model; each future value reaches the block when it takes effect, so a perfect identifier
would know it one step late, as the reference does. Building the block reads attributes only: it draws nothing and
changes nothing. A probed step's copy shifts the block's true x, y and yaw by the same draws as the copy's frame and
star tracker (the rule of hazard_bay_build), so a program that reads the block answers the copy on its own. Apart from
the block, a token run's requests are those of a run without the token (a repository test compares the two on a set
episode with a scripted program).

## The network guard

The verifier sandbox is provisioned with open egress, and once the set is public a controller could fetch it and
identify its episode from the first observation. So the controller runs behind `nonet_exec.py` (byte-identical to the
mechanism box's): no_new_privs, a seccomp filter under which only AF_UNIX sockets open, the agent account, and a fixed
environment (PATH, HOME, LANG). A self-test before episode 0 checks that the filter holds. Where it cannot be
installed, the verify fails closed: a task error ("the grading environment cannot block the controller's network")
before any controller starts, so nothing is graded with the network open. A verdict therefore always records
`egress_guard_seccomp` 1. Measured with a spy submission in a fresh container: AF_INET and AF_INET6 sockets (even to
127.0.0.1) fail with EPERM, AF_UNIX sockets and socketpair work, `/opt/ale` is unreadable, and the environment holds
only PATH, HOME and LANG (plus what bash adds).

Until 2026-11-06 a detect fallback (inherited from hazard_bay_build and the mechanism box) let a controller run without
the filter while the grader looked at the account's internet sockets every 50 steps. It never looked before READY, at
a reset or between episodes, and it skipped closed and unconnected sockets: in a review spy, a controller that fetched
bytes before READY and closed the socket went unseen. The fallback is gone; a repository test checks that a verify
without the filter is a task error and that no controller starts.

## The reference

**The privileged reference** (`solution/privileged/oracle.py`) plans from the block alone. It rebuilds the plant from
each block (`Replica`: env.py's own class with no observation and no event to come), tells the real request from a
probe copy by exact prediction (only a verified request commits anything, so the copy gets the same law on its shifted
pose and never touches the plan), stages at r = 1.60 m on the port's radial line before the re-key, pins the relay
trigger once per event, and allocates the wrench exactly on the next step's true plant. Two steps after the re-key it
starts simulating, from the state eight steps ahead, a fixed family of 266 policies (descent delay, hold gap, speed, a
load penalty on one nozzle, a relative-speed cap, a yaw offset) to the episode's end, and commits the first that docks.
Its invariant is that a planned dock is an executed dock: the planned rollout and the episode are the same
computation, so a committed policy that docks in its rollout docks in the episode at the same step with the same
quality. When no policy docks, the reference keeps the one with the best quality among rollouts that stop as soon as
a dock can no longer be completed in time (about step 371): that fallback's last steps are never simulated, and the
episode can still end in a collision or an escape. On 2000 dev seeds (`review/oracle_batch.py`,
random.Random(20261105)) it docks 1943 (97.15%): every planned dock was executed at the planned step with the planned
quality, none of the 57 episodes without a planned dock docked, and 15 of those 57 ended in a collision or an escape
(draw 29 of the selection ended so too: fallback policy 112, a collision at step 395). The probes pass at 0.9998, and
its slowest request (a plan) took 5.1 s on a loaded host and 2.9 s in a 2-CPU container. The prototypes' 80 policies
alone dock 95.45% of those seeds; a 0.6 rad waypoint in the far field (the prototypes') let 96 of 2000 re-keys fire
late.

**Round 1's non-privileged reference**, the task author's controller (`solution/reference/solver.py` until round 2),
worked from the 16-number observation. On the fixed set in a fresh container it scored 0.53997 (26 of 48 docked, 4
collisions; the lock 794 of 1064 probes), ratio 0.549, reward 0.56. It read the numeric delayed pose, which round 2
removed, so it was removed with it (its code is in the repository's history).

## The anchor

`anchor.json`: 0.9835537203921798, the lowest of the two fresh-container verifies (identical), `full_at` 0.98. Reward
by docks on the set, with the anchor's per-episode qualities and the lost episodes the best ones, scored 0: 48 docked
1.00; 47 0.999 (rounds to 1.00); 46 0.978; 44 0.935; 40 0.850; 36 0.765; 24 0.509; 12 0.254; 0 0. `full_at` 0.98 is the
margin for one episode flipped by a last-bit libm difference on another host, and no more. Before: 0.4580 with
`full_at` 0.75 (16 docks of 48 earned full reward). The verify of 2026-11-06, with the guard failing closed, measured
the anchor again to the last bit (reward 1.0, ratio 1.0, 48 of 48 docked, the lock 777 of 778).

Round 2 (2026-11-07): one verify of the `solve.sh` submission in a fresh container of the round-2 task image
(`review/regrade.sh`, Mesa 23.2.1 lavapipe, CPython 3.12.14, 2 CPUs, 4 GB, no network) measured
0.9835537203921798 again, 48 of 48 docked, the lock 777 of 778, every episode's quality equal to the selection's to the
last bit (`anchor.json` `recheck`). The anchor did not move, as it should not: the oracle reads only the block, and
env.py, the set and the lock streams are unchanged.

## Verify time and limits (round 2)

- **Measured** (the verify above): 969 s wall; the camera ready in 6.4 s (its canary 4.0 s, the first frame's shader
  compile); 10,651 frames for 12,816 steps (fresh frames and probe copies) at 85.5 ms each (render, luma, noise,
  base64); the reference's replies 23.5 s in all. Round 1 took about 30 s.
- **The renderer's cost** in the task image (2 CPUs, `docs/research/.../scripts/cam_bench_image.py` and
  `cam_variants.py` of the dev workspace): 62 ms mean and 90 ms p95 a render with the 2048 shadow map and the craft's
  box, 8.6 ms for noise, encoding and JSON; 513 MB resident. A 4096 map and the full craft cost 78 / 105 ms and change
  no pixel by more than 8 DN but the craft's shadow edge.
- **The worst case** an agent can cause: every episode to 400 steps, about 16,800 frames; `task.toml`'s
  `[verifier] timeout_sec` 7200 is 1800 s of replies + 2 x 16,800 x 0.12 s + 480 s of READY handshakes + 300 s, rounded
  up to 10 minutes (the rule is in the comment there).
- **The reply budget** (`grader_config.json` `reply_budget_s` 1800) is the controller's own time, summed from the start
  of each request's write to its reply; the per-request deadline stays the kit's 30 s, and the write of a request is
  bounded by it too (`BoundedStdin`).
- **Memory**: the grader with the renderer about 0.5-0.7 GB; the task's 4 GB leave the controller the rest.

## What the reference does not dock

Draw 29 of the selection: undocked by all 266 policies from its planning state, re-key on time; cause not identified.
About 3% of random draws are undocked the same way: on 57 of the 2000 dev seeds none of the policies docks from the
planning state, although the re-key fires at its arming step in all 57; they have not been examined one by one.
Whether another controller could dock them is not known.

## The agent's documents

The agent gets `instruction.md` and `given/`, nothing else: no simulator, no renderer of the grader's scene, no grader
of its own. The image has numpy, OpenCV, Pillow and MuJoCo (with Filament on lavapipe), so the agent can build its own
synthetic views from `target.md`: perception tooling, not a plant simulator.

- `given/model.md` is the plant's datasheet (round 2): the frames and units; the six nominal nozzle geometries (exact)
  with their installation tolerances; the residual damping as a range (1.5-2.0 %/s and 3.0-3.6 %/s, env.py's 1.65 and
  3.29); the bus (rotor, stride, degraded pairs, relay, valves as first-order lags with time-constant ranges, thrust
  curves with exponent ranges); the events' windows; the target, the boom and the goal; the plume in words with its
  ranges (29-82 % loss at the port, swirl 0.9-1.2, flips after 0.24-0.38 s of fully aimed firing); the start's ranges;
  the docking test, the end and the quality exactly; one example `act` line (a dev seed, its frame cut). It withholds
  the equations, the update order, the noise and dropout laws and the distribution families, which round 1's page
  stated so exactly that a plant written from it equalled env.py after every step.
- `given/camera.md`: the sensors (mount, axes, intrinsics, response, exposure, noise, timing, outages in words and
  ranges, the frame's format and decoding, the probe copy's meaning) and a JSON block equal to `camera.CAMERA_SPEC`.
- `given/target.md`: the drum, its surfaces in words, the well, the boom and the drogue, the fiducials' exact places,
  the light; a JSON block equal to `camera.TARGET_SPEC`.
- `given/frames/`: `README.md` (in git) and 40 frames with `truth.json` (task assets): four dev episodes (labels A-D,
  their seeds in `review/make_sample_frames.py`, none in the set or its selection), the privileged reference's flights
  through the grader's own loop; per episode the start, 3 far-field frames aimed at the drum, 2 mid-range, the first
  after the re-key's restart, 2 on the final approach and 1 at the hold. Each is the exact frame the grader's camera
  renders for that pose in the task image, its noise keyed by a dev key (never the set's); `truth.json` names no seed.

The repository test (`DatasheetTests`, which replaced round 1's rebuild-from-text check, `docplant.py` and
`doc_vs_env.py`, now removed) is inverted: (1) every range or tolerance the page states holds env.py's draws on 20,000
dev worlds (`random.Random(20261107)`, none in the set or its selection) and env.py's constants (the damping per
second, the plume's loss at the port and its swirl, the flip thresholds, the fuel burn, the nominal table); (2) the
page, the instruction and the sensors' documents state none of the withheld formulas, noise laws or exact values
(round 1's `0.998`, `0.996`, the dropout `0.035`, the pose noise `0.012` and `0.008`, `mod 13`, `U(`, `U{`, `** 4`,
`z_i <-`, `hypot`, the valve-memory and threshold formulas, the exact swirl magnitudes, among others; numbers matched
whole); (3) the documents' JSON blocks equal `camera.py`'s constants; (4) the page's example request is the request
the grader builds for that dev episode (but the frame's bytes, which need the renderer).

`env.py`'s own text still describes the agent's copy (the module docstring's "An identical development copy is staged
for the agent" and "this source can be shared with the agent", `_RNG`'s "The agent holds this code"): the file is
frozen (its sha256 pins the set and the reference's copy), so those lines stay, and this README is the current
account.

## The completability evidence (round 2)

`review/check_perception.py` with `review/estimate_pose.py`, run on 2026-11-07 in containers of the task image (four
workers, 2 CPUs each). The privileged reference's 48 fixed-set flights were recorded in process (12,816 real requests;
the same flights as its verify), and at each of their 10,014 fresh stamps the grader's camera rendered the frame twice:
as flown (the reference's yaw; it ignores the camera) and aimed (the same position, the boresight through the drum's
centre), 20,028 frames with the grader's own noise. The estimator uses only the frames, the star tracker's readings and
the numbers of `given/camera.md` and `given/target.md`: LED blobs at 250 DN; the DT found by its symmetry about the
image's centre row (the camera is at the DT's height); the camera's planar pose from the DT in closed form, then
Gauss-Newton on the five LEDs; r_port from the BT given that pose; in the far field the BT triangle (0.144 m wide)
with the encoder's radius, its polarity taken as calibrated (O3). A fix is kept when its reprojection RMS is under
0.2 px (DT) or 0.3 px (BT) and the camera is in front of the target: the gates were added after the first run, in
which sun glints on the insulation, saturated like the LEDs, now and then mimicked the pattern (the medians were
already 4 mm and 4 mrad). The drum's silhouette is no help: at the fixed exposure its unlit side is as dark as the sky
(about 0.5 DN against 0.3), and the boom stands out past the limb when the port is near it. Frames of fixed-set
episodes never left the scratch folder; these are the aggregates.

The first layout of the docking target (a 40 mm square, a 20 mm standoff) met E1 (7.3 mm RMS) and O1 but missed E2
(8.1 mrad RMS, 27.5 p99); as pre-registered, the target was enlarged (a 50 mm square, a 30 mm standoff) and the check
ran once more. That second run:

| row | condition | pre-registered pass | result | |
|---|---|---|---|---|
| E1 the craft's position about the drum's centre | all 5 DT LEDs in frame, the camera within 1.2 m of the DT (7,189 frames) | RMS <= 0.010 m, p99 <= 0.030 m | RMS 0.0054 m, p99 0.018 m, max 0.037 m; a fix in 6,846 (95.2 %) | pass |
| E2 the target's phase | as E1 | RMS <= 0.004 rad, p99 <= 0.012 rad | RMS 0.0056 rad, p99 0.019 rad, max 0.045 rad | **miss** |
| E3 the port's radius from the BT | >= 2 BT LEDs in frame (5,852 frames) | RMS <= 0.010 m | RMS 0.0052 m, p99 0.022 m; an estimate in 2,192 (it needs the DT's pose too) | pass |
| E4 the craft's position, far field | aimed, both drum limbs in frame, 1.2-6.2 m (8,708 frames) | RMS <= 0.05 m | RMS 0.029 m, p99 0.11 m; a fix in 7,365 (85 %; 90 % of those with a fiducial in frame): DT 7,136, BT 229 | pass |
| O1 the DT in view at the hold | flown; the 30 docked steps and the 17 before | every fresh frame, all 48 episodes | 47 of 48 episodes; 2 of 2,175 frames (episode 44) | **miss** |
| O2 the drum in view at range | aimed; the lens 1.2 m or more from the drum's axis | the whole width in frame | 8,283 of 8,283 frames (by the craft's centre at 1.2 m: 8,708 of 9,153; the misses have the lens 1.12-1.2 m out) | pass |
| O3 the encoder's polarity after the restart | flown; >= 2 BT LEDs in frame and \|r_port - 1.14\| >= 3 x E3's RMS | reported | median 48 frames an episode; 45 episodes with 5 or more, 3 (1, 32, 44) with none: the reference flies the rest of those approaches with the BT out of view | reported |
| O4 the start | flown, t = 0 | reported | the drum's centre in frame in 26 of 48 starts, part of the drum in 32, the largest off-axis angle 118 degrees; a fix at t = 0 in 16 | reported |

- **E2 by range** (the per-frame phase error grows with the range, as a 30 mm standoff's parallax shrinks): RMS
  0.0019 rad below 0.4 m, 0.0046 at 0.4-0.6 m, 0.0056 at 0.6-0.8 m, 0.011-0.013 beyond 0.8 m. In the hold window
  (E1's frames among O1's): the phase RMS 0.0023 rad, p99 0.0067, max 0.0115 against the docking tolerance of
  0.020 rad; the position RMS 0.0022 m, p99 0.0058 m against 0.085 m. Round 1's numeric port yaw had sigma 0.0035 rad at
  every range. The rule allowed one rerun, so E2 stays a miss beyond 0.4 m; a filter that follows the target's steady
  slew averages it down over the frames of an approach.
- **O1's two frames** are episode 44's: 17 steps before its hold the reference slides in 0.185 m to the side of the
  port's line at 0.28 m from the DT (the standoff LED at the image's right edge, u 642), and at its last docked step it
  sits 0.048 m off the port at 0.076 m from the DT (u 646). The first layout kept all of them in frame; the enlarged
  standoff LED sits 1 cm nearer the lens. O1's own remedy (a 100-degree field, fx = 268.5) was not applied: it would
  cost 16 % of every fix's resolution (E2 already short) and the LEDs' pixel cover beyond about 3.8 m, and a controller
  that watches its camera holds well inside the 0.085 m tolerance.
- **Far field, as flown:** the reference ignores the camera, yet a fix (DT or BT) exists in 4,300 of its 6,220 frames
  beyond 1.47 m.

**Observability statement.** At the hold the docking target is in the camera's field in every fresh frame of 47 of
the 48 reference holds (2 frames of 2,175 miss, both with the craft more than 4 cm off the port's line), and the
relative pose comes out of a single frame at 2.2 mm and 2.3 mrad RMS, better than round 1's numeric pose. On the
approach (0.4-1.2 m) a single frame gives the craft's position at 4-14 mm and the target's phase at 5-13 mrad RMS. In
the far field the docking target or the boom-tip triangle gives the position at 29 mm RMS in 85 % of the frames of a
craft that points its docking axis at the drum (O2: the drum's width is always in the field from 1.2 m out); pointing
is the controller's choice, since every nozzle couples translation and yaw and the six span the wrench space at any
yaw. The encoder's polarity is decidable from the BT before the re-key and, in 45 of 48 reference approaches, after
the restart (O3); otherwise from the DT's range at the hold.

## Review tools (`review/`, never run by the grader)

- `oracle_batch.py`: the dev batch in process (dock rate, probes, planned vs executed, verification failures, timing);
  black frames (`verify.BlankCamera`) unless a camera is passed: the reference reads none.
- `run_episode.py`: grader-exact episodes of a submission folder (its run.sh started as the current user), with an
  in-process comparison and a one-process mode (reset hygiene).
- `select_set.py`: the pre-registered selection (`draw`, `run`, `make`, `report`).
- `regrade.sh`: one verify in a fresh CPU container of the task image, the separate verifier's layout, no network.
- `build_camera_bundle.py`: the camera bundle from the render-realism step's `bundle_v3`, with its manifest and sha256s.
- `make_sample_frames.py`: the agent's sample frames (`frames`), the render canary (`canary`) and the datasheet's
  example request (`act`), in a container of the task image.
- `estimate_pose.py` and `check_perception.py`: the completability evidence (below): a camera-only estimator that uses
  the agent's documents alone, run on frames of the reference's fixed-set flights; only aggregates leave its scratch
  folder.
