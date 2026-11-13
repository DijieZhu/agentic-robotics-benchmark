# Design: kitchen appliance tending (draft 2, 2026-11-12)

Status: **draft for the owner's review; P1 done (2026-11-12, §10).** No task folder exists yet. Decisions marked
**[owner]** are open; the rest is a recommendation the phases in §9 can still overturn.

Draft 2 replaces draft 1 (same day), which used a mobile base inside RoboCasa's full runtime.
Two reasons for the change: base navigation from cameras would likely push every frontier model
to a reward of 0, which measures nothing; and `tasks/arm/arm_hidden_mechanism_box` (#18) already
solves the same kind of problem on `main` (a fixed Panda operating articulated parts it has no
model of), with a shape that has passed review and `ale validate`. This draft follows that task
wherever it can, and cites it as **[box]**.

Feasibility evidence: `experiments/physical-execution/APPLIANCE_PILOT_TEST.md` (RoboCasa, a
privileged scripted controller: stove knob, microwave start, oven door 10/10 each on the tuned
kitchen; 17/28 on unseen kitchens with the rack included, the rack 0/7).

## 1. The task

A Franka Panda stands fixed in front of a kitchen appliance it has not seen: a stove, a microwave
or an oven. Over an episode it receives a few household requests in plain text ("turn on the
rear-left burner", "open the microwave door", "press start"), one at a time, each with a deadline.
It must find the right control from its cameras, operate it through contact, let go, and leave
every other control of the appliance as it should be. The agent delivers a controller
(`/home/user/submission/run.sh`) that the hidden grader drives over the closed_loop stdio wire on
hidden appliances after the agent is gone.

- Proposed slug: `kitchen_appliance_tending` **[owner: confirm; register it in the sheet]**.
  Platform `arm`. Verify pattern closed_loop; `environment_mode = "separate"`.
- What it measures: finding small controls on an unseen appliance from images, contact-rich
  actuation with weak fingers (20 N per finger against a knob with 1.0 N m friction; a 6.6 kg
  oven door), and not disturbing the rest of the appliance. No locomotion.

## 2. What we take from `arm_hidden_mechanism_box`

| [box] does | Here |
|---|---|
| One environment file (`box_env.py`, plain MuJoCo 3.16, about 800 lines: scene, Panda, OSC controller, cameras, sensor noise, credit), **byte-identical** in the agent's dev sim, the lab and the grader; only the loaded box differs | `appliance_env.py`, the same way; only the loaded **unit** differs |
| No robosuite runtime: `robosuite==1.5.2 --no-deps` for the Panda model files; the LIBERO tabletop scene from a pinned dataset revision | The same Panda files; the appliance models extracted from RoboCasa's fixture assets (§5); no RoboCasa runtime and no 10 GB asset download in the image |
| A box spec = an MJCF body fragment + a stage list; 4 hidden boxes in `tests/ref/boxes.json` | A **unit spec** = an MJCF fragment + its controls (joint, kind, label, target predicates); hidden units in `tests/ref/units.json` |
| A practice box with its ground truth (`env.gt()`) in the agent's home | Practice units (different models from the hidden ones) with ground truth |
| A lab service (`/run/ale-lab/lab.sock`, started by `setup.sh`, dedicated `alelab` account) with **16 trials** on the hidden boxes; observations only, no scores | The same service and rules on the hidden units |
| Fixed Panda; the box at a fresh pose each episode (±4 cm, ±5 cm, ±25° yaw) | Fixed Panda; the unit at a fresh pose (ranges from the reach study, P1) |
| Observation: joints, end-effector pose, gripper, noisy wrist force/torque, wrist-camera pose; RGB from two fixed cameras and the wrist camera **on request** (bit mask in the action; images cost time); no depth, no box state | The same, plus `request` (`null` or `{"text", "deadline_t"}`) **[owner: add depth?]** |
| Action `[dx, dy, dz, drx, dry, drz, grip, cams]` at 10 Hz, clipped 0.03 m / 0.2 rad | The same |
| Staged partial credit with a hold rule (3 steps) | §4 |
| 8 episodes (2 per box), placements and noise from OS entropy (`seed_rng`); probe rate 1 %, at least 10 % of probe pairs must differ; 6000 s budget; seccomp network guard (`nonet_exec.py`); agent processes stopped first; 200 MiB / 5000-file submission cap; one controller process for the whole run (online adaptation allowed) | The same, with the counts in §6 |
| Privileged reference: knows every hidden box (`solution/ref/boxes.json`), estimates the box pose from the two fixed cameras at t = 0, then runs scripted primitives | The same: knows every hidden unit's controls, estimates the unit pose from the fixed cameras, runs the pilot's primitives (pinch-turn, fingertip press, handle drag) |
| CPU only (4 CPUs, 12 GB, no GPU); MuJoCo's Filament renderer on Vulkan lavapipe | The same unless P1 measures it too slow |
| Agent budget 8 h | 8 h **[owner]** (the lab trials and per-model debugging need it) |

## 3. Units and requests

A unit is one appliance model mounted in a short counter section, with the robot's mount height
chosen per kind so every control is reachable (P1 measures reach; a wall oven may need a raised
mount or a lower base). RoboCasa's fixture registry offers about 17 stove, 50 microwave and 21 oven
models, enough to keep the practice and hidden models disjoint.

Proposal: **4 hidden units** (2 stoves or cooktops, 1 microwave, 1 oven), **3 practice units**
(one of each kind) **[owner: drop the oven from v1 if P1 shows its reach needs a different
mount]**. Each episode loads one hidden unit and opens **K = 3** requests in sequence; a request
closes on success or at its deadline (proposal: 600 steps = 60 s), then the next opens.

| Kind | Requests (v1 candidates) | Predicate (as RoboCasa defines it, re-implemented in `appliance_env.py`) | Pilot |
|---|---|---|---|
| stove | turn on / turn off burner `<location>` | knob angle ≥ 0.35 rad (`Stove.is_burner_on`) / below it | on: 10/10 tuned, 5/7 unseen; off: untested |
| microwave | press start / press stop | `turned_on` true / false, gripper ≥ 0.15 m from the button (`gripper_button_far`) | start: 10/10, 6/7; stop: untested |
| microwave | open / close the door | door ≥ 0.90 of its range (`Fixture.is_open`) / closed | untested |
| oven | open the door | ≥ 0.90 of its range | 10/10, 6/7 |
| oven | preheat | temperature knob ≥ 0.25 (`PreheatOven`) | untested |

Left out: the oven rack (0/7 unseen), closing the oven door (lifting 6.6 kg; untested). The
initial control states are drawn per episode so that "off", "stop" and "close" requests start from
"on", "running" and "open". P2 fixes the final list from what the reference does reliably.

## 4. Scoring

Per request, two stages, each judged by the grader from its own simulator state:

- **reached** (0.5): the requested control reaches its target state and holds it for 3
  consecutive steps;
- **completed** (0.5): the target state holds for 20 consecutive steps with the hand at least
  0.10 m from the control (the pilot's and RoboCasa's release rule), **and** every other control of
  the unit is in its expected state (its initial state, changed only by earlier successful
  requests) during those steps.

A counted stage stays counted. Episode score = mean of its K request scores; metric
`mean_request_credit` = mean over the 8 episodes, direction higher. Untouched scores 0: every
request needs a state change the robot causes. An episode in which the controller crashes, answers
late or malformed, or makes the simulation unstable scores 0 ([box]'s rule, including the joint
speed limit). **[owner: partial credit for progress toward the target (knob angle, door
fraction) instead of the 0.5 "reached" stage?]**

## 5. Assets and licence

- Appliance models: RoboCasa's fixture assets, CC-BY-4.0 (RoboCasa README; the Hugging Face dataset
  `robocasa/robocasa-assets` carries `fixtures.zip` and `textures.zip` under that licence). The
  v1.0 "lightwheel" fixtures come through `download_kitchen_assets.py` from a Box link instead; P1
  checks which source holds the chosen models and that its licence is CC-BY-4.0 **[owner: sign
  off]**.
- The extracted MJCF and meshes of the chosen models go into the benchmark's asset dataset
  (`ale-robotics-assets`, fetched at build through `environment/fetch_assets.py` +
  `assets.sha256`); `tests/NOTICE.md` gives the attribution, as
  `robocasa_object_dynamics_from_video` does. Hidden models' meshes must not be in the image: the
  grader carries them in `tests/` (as [box] carries `tests/ref/boxes.json`), so the asset dataset
  holds only the practice models and the robot's scene.
- Behaviour RoboCasa implements in Python, not in MJCF (the microwave's button state, the burner
  threshold, door "open" fractions), is re-implemented in `appliance_env.py` and checked against
  RoboCasa on the same models (P1).

## 6. Grader, reference, anchor

- Grader: [box]'s `verify.py` structure. 8 episodes (2 per hidden unit, hidden order; the
  observation names the unit, so the controller may adapt across episodes), fresh 64-bit seeds from
  `seed_rng` (unit pose jitter, initial control states, request queue, sensor noise).
- Reference (`solution/`): privileged, and given the truth by token (owner decision 2026-11-12; the
  spacecraft and messy-desk pattern): only in a run that presents the oracle token from `solution/`
  (its sha256 in `tests/hidden/`), the grader adds the unit's true pose and every control's state to
  each observation; untokened runs never see them. It then runs the primitives and spends no lab
  trials. The anchor is therefore "perfect perception + these primitives"; the reference does no
  camera perception (this replaces draft 2's box-style pose estimation).
- Anchor: the reference's `mean_request_credit` over at least 20 fresh oracle runs through the
  engine; `full_at` = its worst measured ratio ([box] measured 1.0 in 24 of 24 runs and set
  `full_at` 1.0). If its run-to-run spread is wide, switch to a fixed hidden episode set
  (spacecraft, messy desk) instead of widening `full_at`.

## 7. Why the score should not floor at 0

- The "reached" stage credits half a request for getting the control to its target at all.
- Every episode has a fixed base and an appliance in view at t = 0; no navigation.
- P3 measures it before the task is built (§9).

If frontier models still score near 0 in P3, simplify further (more practice units, depth images,
a larger lab budget) before adding difficulty anywhere else.

## 8. Risks

1. **Extracting fixtures.** RoboCasa builds fixtures through Python (`models/fixtures/*.py`), not
   only MJCF; some joints, sites and button logic may need rework. P1 extracts two models per kind
   first.
2. **Reach.** Oven doors and wall ovens sit low or high relative to a counter-mounted Panda. P1
   measures reach per model; the mount can differ per kind.
3. **Render speed.** [box] renders with Filament on the CPU (about 0.07 s per image); appliances are
   larger scenes. P1 measures.
4. **Reference reliability on hidden models.** The pilot's privileged controller transferred to
   17 of 28 unseen scenes without knowing their models; this reference knows them, but P2 must show
   it before the hidden set is chosen.
5. **The agent holds the dev simulator.** Hidden units never leave `tests/`, the lab and
   `solution/`; seeds are 64-bit OS entropy; the lab returns observations only. Same as [box].

## 9. Plan of work (each phase writes its result in §10)

- **P1 — environment.** Pick candidate models (3-4 per kind); extract them from RoboCasa; build
  `appliance_env.py` with the [box] Panda, controller, cameras and F/T; re-implement the predicates
  and check them against RoboCasa; measure reach per model, step rate with images, memory.
- **P2 — reference.** Port the pilot's primitives to the new environment with privileged unit
  specs; add burner off, microwave stop and door open/close; add camera pose estimation; run every
  candidate model over many fresh seeds. Choose the practice and hidden models and the final
  request list from the results.
- **P3 — solvability and floor check.** A camera-only baseline controller (no unit spec) on the
  practice units, and one frontier-model run on a minimal version (one practice unit, the practice
  grader). Go on only if neither is stuck at 0.
- **P4 — the task folder.** From [box]: lab service, grader, practice grader, wire skeleton,
  payload, instruction, `tests/NOTICE.md`, assets in the dataset. `ale lint`,
  `scripts/lint_domain.py`, `scripts/task_assets.py check`.
- **P5 — gates and difficulty.** `ale validate` (untouched 0.0, oracle 1.0), anchor from fresh
  oracle runs, frontier runs (`ale run`); target reward roughly below 0.5.

## 10. Results log

### P1 — environment (2026-11-12, dev host TASL)

Code: `experiments/physical-execution/appliance_env/` (its README has the set-up and every command);
outputs in its git-ignored `../outputs/p1/`. Python 3.12, MuJoCo 3.16.0, NumPy 2.5.4, robosuite 1.5.2
model files only; no RoboCasa code runs.

- **Assets and licence.** RoboCasa v1.0 fixtures come as `fixtures_lightwheel.zip` (530 MB, sha256
  `62c1c877…bd6c3e`), whose README names the source: Hugging Face
  `nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF` (revision `420a04af`), licence
  **CC-BY-4.0**. The older `robocasa/robocasa-assets` `fixtures.zip` (227 MB) is the v0.2 set. This
  settles §5's licence question for the models used here, subject to the owner's sign-off.
- **Inventory.** 26 stoves (ranges), 24 stovetops, 50 microwaves, 25 ovens; **all 125 compile
  standalone** in plain MuJoCo. Controls as RoboCasa names them: burner knobs
  `knob_<location>_joint` (4-6 per model; ranges: horizontal axis on the front panel, range about
  0-3.1/3.7 rad, friction loss 0.75; stovetops: vertical axis on the top surface, friction loss 1.0);
  oven and range doors `door_joint` (range 0-1.15 / 0-1.57 rad, drop-down); microwave door `microjoint`
  (-1.57-0, side hinge); `knob_temp_joint`, `knob_time_joint`, `rack<k>_joint`; microwave
  `start_button` / `stop_button` are fixed geoms, not joints (RoboCasa switches the state on contact).
  Every model has a `door_handle_main` geom where it has a door.
- **Masses.** The models set no masses, so MuJoCo's default density gives every visual mesh mass too
  (raw door subtrees: microwave 3-36 kg, oven 2.5-35 kg, range 17-88 kg). With visual meshes made
  massless (as robosuite does for its objects) the doors weigh: Microwave011 4.75 kg, Oven048 6.25 kg,
  Stove026 8.8 kg, in line with the pilot's 6.6 kg oven door. Whether to set door masses explicitly is
  a P2 decision.
- **Environment.** `appliance_env.py` builds floor, support (counter, cabinet column or nothing), back
  wall, the [box] Panda and controller, cameras and F/T, and the unit turned to face the robot; build
  0.7-0.9 s. Re-implemented predicates: burner on (`0.35 <= angle mod 2pi <= 2pi - 0.35`), door
  fraction (flipped for negative ranges), open >= 0.90, closed <= 0.005, preheated >= 0.25, microwave
  on/off by gripper contact with start/stop (door open switches it off), button far > 0.15 m; burner
  flame sites shown while on (RoboCasa's visual cue).
- **Predicate check** (`check_predicates.py`): **all pass**. 15 threshold cases on Stove026 and
  Microwave011, and physical presses through the controller on Microwave011: start turns it on, stop
  turns it off, start again on, opening the door turns it off, start after closing on; `*_far` false
  at the button and true after backing off. Not compared against a running RoboCasa: the formulas
  are transcribed from RoboCasa `3443cdf` (`stove.py`, `microwave.py`, `fixture.py`, `oven.py`).
- **Reach** (`reach.py`, kinematic IK, no collision check; pre-contact point 8 cm out and the contact
  point; approach nominal, 30 degrees down or 30 degrees to either side; door handles at fractions 0
  and 0.5 for microwaves, 0 and 0.3 for drop-down doors). Fully reachable models:

  | Mount (front of unit, robot base height) | stovetop | microwave | oven | range |
  |---|---|---|---|---|
  | 0.50 m, 0.65 m | 24/24 | 43/50 | 25/25 | 24/26 |
  | 0.50 m, 0.80 m | 24/24 | 35/50 | 25/25 | 24/26 |
  | **0.55 m, 0.65 m** | **24/24** | **50/50** | **25/25** | **24/26** |
  | 0.55 m, 0.80 m | 24/24 | 49/50 | 25/25 | 24/26 |

  The two ranges that fail (Stove001, Stove074) have their knobs on a rear panel 1.34 m out and
  1.22 m up; they drop out. The microwave's open door swings towards the robot, which is what needs
  the extra distance. Default mount now: front 0.55 m, base 0.65 m, for every kind.
- **Collisions matter more than kinematics.** With a level approach the hand's lower edge hit the
  counter before the fingertips reached Microwave011's start button (3 cm above the counter); a
  35-degree downward approach pressed it. P2's primitives must plan approaches against the unit and
  its support, not only reach.
- **Cost** (`perf.py`, one thread, classic renderer through OSMesa on the CPU): 183-231 control steps
  per second without images; 0.014-0.06 s per image (320 x 240 fixed, 256 x 192 wrist); 7-16 steps
  per second with all three images every step; peak RSS 1.27 GB over five units. An episode of
  3 x 600 steps costs under 5 minutes of simulator time even with every image, so 8 episodes fit
  [box]'s 6000 s. Filament (what [box] uses) is not measured yet.
- **Images.** Fixed cameras beside and above the robot's shoulders show the whole front of each unit;
  the arm crosses the centre of both views at the home pose. Rendering is plain (one light, no image-
  based lighting); P4 decides between this and [box]'s Filament set-up.

**For P2:** candidate pool = every model except Stove001 and Stove074. Open questions carried over:
explicit door masses; an approach planner that respects the counter and the unit body; a home pose
whose wrist camera sees the unit.

### P2 — privileged reference, first pass (2026-11-12, dev host TASL; in progress)

Code: `experiments/physical-execution/appliance_env/` `reference.py` (primitives), `run_reference.py`
(trials), `batch.sh` (every candidate model, sharded), `summarize.py`; outputs in `../outputs/p2/`.
P1's own checks still pass (`check_predicates.py`: all pass).

- **Environment changes.** Unit state (microwave on/off, burner flames) is updated once per control
  step, as RoboCasa does, and "the gripper touches" means the gripper subtree only, as RoboCasa's
  `check_contact(robot.gripper)`. The wrench is reported in the world frame. Knob angles display in
  (-pi, pi]. **Home pose:** the gripper site at the robot base + (0.45, 0, 0.55) m, pointing at the unit's
  front 0.15 m below its top (at least 20 degrees below horizontal), solved by IK from four fixed starts;
  joint-limit margins 0.6-1.05 rad on the test units, and the wrist camera sees each unit's front (the
  P1 home pose did not).
- **Knob physics (deviation from RoboCasa; approved by the owner, 2026-11-12).** RoboCasa's knobs have friction loss 0.75 N m
  (ranges) to 1.0 N m (cooktops) and armature 1.0 kg m^2. This environment's controller ([box]'s: the
  target re-based every step, at most 0.03 m / 0.2 rad ahead) delivers roughly 0.5-0.8 N m at the
  wrist, and a pinch with the 20 N fingers holds about 0.6 N m on a 24 mm knob: the reference could
  not turn a range knob. P2 sets knob friction loss to 0.20 N m and armature to 0.002 kg m^2
  (`appliance_env.KNOB_FRICTIONLOSS`, `KNOB_ARMATURE`), close to a real cooker knob (about 0.1-0.3 N m).
- **Reference.** Reads control geometry and joint values from the simulator; acts only through the
  8-number action. A planner checks each primitive's key poses by full-pose IK chained from the
  current joints (what the controller's small steps follow), with a collision check against the unit
  and its support (the target part allowed); when that fails it solves the first pose from fixed starts
  and hands the executor the Cartesian poses of the joint-space interpolation (`via`). Primitives:
  - knob: pinch across a bar's thin side (or a round knob's diameter), fingertips 3 mm proud of the
    knob's base, approach along the axis or tilted (20/35 degrees down, 20 degrees sideways), either
    finger direction; a turn the wrist cannot make at once is made in up to three grasps;
  - button: closed fingertips, approach tilted 35/20/0/50 degrees down (a level hand meets the counter
    first, P1), the hand's wide side across or upright;
  - drop-down door (range, oven): the pilot's drag from above until the door's weight takes over;
  - microwave door: pull by the handle to 0.45 rad, then push from inside the gap with the side of
    the closed fingers; close: push the outer face round. **Neither works yet.**
- **Batch** (`batch3`: 123 candidate models x applicable primitives x seeds 0-2, 1032 trials, about
  2 minutes on 32 processes):

  | Kind | Primitive | ok / trials | Main failure |
  |---|---|---|---|
  | stovetop | knob on / off | 62/72, 56/72 | plan (no collision-free grasp) |
  | range | knob on / off | 63/72, 53/72 | plan, approach |
  | range | temperature knob (preheat) | 41/72 | plan, twist |
  | range | door open | 54/72 | drag (the door does not follow) |
  | microwave | start / stop | 139/150, 141/150 | press (no switch) |
  | microwave | door open | 0/150 | the door falls back closed during the push |
  | oven | temperature knob (preheat) | 51/75 | plan |
  | oven | door open | 16/75 | drag: oven doors carry a 2.0 N m/rad closing spring (ranges 0.05) |

  Models whose every trial of the candidate v1 set succeeds: stovetops 17/24 (knob on/off),
  microwaves 40/50 (start/stop), ovens 13/25 (preheat), ranges 1/24 (knob on/off, preheat, door
  open; 4 more miss one trial).
- **Not done in P2 yet:** camera pose estimation (the reference still reads the unit pose from the
  simulator); microwave door open/close and the oven door; a request sequence within one episode
  (initial states, the hold-and-clear rule across requests); choosing practice and hidden models.
- **Owner decisions (2026-11-12):** the knob physics above is approved (friction loss 0.20 N m, armature
  0.002 kg m^2). The v1 request list is: cooktop and range burner on / off, range and oven preheat and
  turn the oven off (added the same day, below), range door open, microwave start / stop. Microwave
  doors and the oven door are out of v1.

### P2 — second pass: episodes, leaks, the reference's truth, model reliability (2026-11-12)

Code: `episode.py` (the grader-side episode: initial state, request queue, credit), `run_episodes.py`
(reference or untouched episodes), `batch_episodes.sh` (every candidate model, sharded).

- **Owner decisions.** Knob friction 0.20 N m and the v1 request list (above) are approved. The
  reference gets the truth by token (§6), so P2 does no camera pose estimation.
- **Episode** (`episode.py`): the initial state is drawn per episode (each burner on with p 0.3 at
  0.8-1.6 rad; the microwave running with p 0.5); then K = 3 requests, each drawn from those valid in
  the state it opens on (burner on/off, preheat, range door open, microwave start/stop), with a
  600-step deadline. Credit follows §4: reached (target held 3 steps) 0.5, completed (held 20 steps,
  hand at least 0.10 m from the control, every other control in the discrete state it had when this
  request opened) 0.5. "Other controls as they were when the request opened" means a failed earlier
  request does not cost later ones. Request text is fixed templates ("Turn on the rear left burner.",
  "Preheat the oven.", "Open the oven door.", "Start the microwave.").
- **Two leaks fixed in `appliance_env.py`.** The fixed cameras were placed relative to the jittered
  unit, so the unit looked the same in every episode and the camera poses changed instead; and the home
  pose was solved for the jittered unit, so the initial joint angles encoded its pose. Both now follow
  the nominal mount: cameras and initial joints are identical across seeds (checked: max joint
  difference 0.0); only the unit moves.
- **Checks.** Untouched episodes score 0.00 (4 kinds). Reference episodes on one model per kind,
  seeds 0-1: all 1.00.
- **Reliability over every candidate model** (`ep_batch1`: 123 models x 5 seeds = 615 episodes; median
  401 control steps and 3.6 s wall per episode without images):

  | Kind | Mean episode score | Models at 1.00 in all 5 seeds |
  |---|---|---|
  | cooktop | 0.83 | 17/24 |
  | range | 0.79 | 2/24 (Stove063, Stove064) |
  | microwave | 0.91 | 34/50 |
  | oven | 0.70 | 11/25 |

  Requests completed: burner on 300/344, burner off 203/247, start 315/358, stop 367/392, range door
  open 22/30, preheat 142/292 (a failed preheat leaves the oven valid for another preheat, so ovens
  repeat it).
- **Selection rule (proposed for P4).** Candidates are the models at 1.00 in all 5 seeds. P4 re-runs
  each candidate on 20 fresh seeds, keeps those still at 1.00, then draws with `os.urandom` (recorded
  before any candidate runs, as spacecraft's `select_set.py` does) the hidden units (one cooktop, one
  range, one microwave, one oven) and, from the rest, the practice units (one per kind).
- **"Turn off the oven" added (owner decision, 2026-11-12).** Ovens offered one v1 request (preheat), so
  a successful oven episode had one request. The temperature knob back below RoboCasa's 0.25 is now a
  request too ("Turn off the oven."), and an oven starts preheated with p 0.4. Measured with it
  (`ep_batch2`, all models x 5 seeds), the temperature-knob turns were the weak point (oven mean 0.60,
  2/25 oven models at 1.00 in every seed). Neither returning the hand home between requests (oven 0.57)
  nor a higher robot base (0.85 m: 0.63; 1.0 m: 0.60) helped. Smaller turns did (`ep_oven_t32`): the
  preheat target 0.32 of the knob's range instead of 0.40 (`reference.TEMP_TARGET`; RoboCasa's threshold
  is 0.25) and a preheated start at 30-50 % of the range instead of 40-80 %
  (`episode.PREHEATED_RANGE`):

  | Kind | Mean episode score | Models at 1.00 in all 5 seeds |
  |---|---|---|
  | range | 0.85 | 2/24 (Stove028, Stove075) |
  | oven | 0.67 | 8/25 (Oven034, 036, 037, 038, 039, 049, 050, 053) |

  Requests completed in that run: preheat 162/262, turn the oven off 120/181, burner on 162/169, burner
  off 91/107, range door open 13/16. Cooktops and microwaves have no temperature knob; their numbers
  above stand. The range pool (2 models) is just enough for one hidden and one practice range; if P4's
  20-seed recheck drops one, the temperature-knob primitive needs work before the draw.

### P3 — draft task, baseline, gates, first frontier trial (2026-11-12 to 2026-11-13)

Plan approved by the owner: build a draft task folder (most of P4) so a frontier model can be run through `ale
run`, add the practice-model baseline, validate, run one Opus 5.5 trial (subscription auth).

- **The reference over the wire** (`reference.Shadow`, `oracle_main.py`, `wire_grade.py`): the reference keeps
  its own copy of the scene in step with each observation and answers every query (a probe's repeat too) from
  the queried observation. Over the wire it scores what it scored in process (1.0 on four P2 units).
- **Unit selection** (`tests/hidden/selection.json`): the 20 recheck seeds and the draw key were drawn with
  `os.urandom` and committed (`3e7ad6f`) before any recheck ran. Survivors at 1.00 in all 20: cooktops 16/17,
  microwaves 22/34, ovens 5/8, **ranges 0/2** (Stove028 14/20, Stove075 15/20). **Owner decision:** ranges are
  dropped for now; "Open the oven door" leaves v1. Drawn: hidden u0 Stovetop011, u1 Microwave081, u2 Oven036;
  practice p0 Stovetop018, p1 Microwave075, p2 Oven039.
- **Draft task** `tasks/arm/kitchen_appliance_tending/` (`47fa7ed`), after arm_hidden_mechanism_box: grader
  (seccomp launcher, kit Solver and ProbeLock, 9 episodes = 3 per hidden unit, token truth for the oracle), lab
  service with an appliance plugin (16 trials; hidden models read into memory before the privilege drop),
  payload (dev sim, practice units, task card, practice grader, helpers), image (OSMesa), instruction. The
  observation now carries the cameras' intrinsics and poses (the fixed cameras' poses depend on the unit's
  model) and always the three image keys (empty when not asked); the request is three flat fields
  (`request_id`, `request_text`, `request_deadline_t`), as the lab's reply schema needs. Agent budget 2 h for
  the trial. `sync_task.py` keeps the code and models byte-identical across dev sim, lab, grader and reference.
- **Gates.** `ale lint` ok; `lint_domain` ok; task assets ok; kit byte-identical; `pytest` passes.
  **Anchor:** `mean_request_credit` 1.0, `full_at` 1.0: the reference scored 1.0 in all 7 engine runs on fresh
  seeds (63 of 63 episodes). **`ale validate` (`validate-9c43caa4`): untouched 0.0, oracle 1.0.** A verify takes
  about a minute.
- **Baseline** (`baseline/`): knows only the practice models; estimates the unit's placement by
  render-and-compare against the fixed cameras (2 mm / 0.25 degree on the practice units, about 5 s) and a
  knob's angle before an "off" request (about 0.05 rad), tracks the knob it turns and the microwave flag itself,
  and runs the reference's primitives. Practice grader: **0.926** over 9 episodes (8 at 1.0; one microwave start
  missed), 37 of 37 probe pairs differed. The task is solvable as instructed from observations alone, given a
  unit's model; on the hidden units an agent must also find the controls of models it has not seen.
- **Frontier trial 1** (2026-11-13, `ale run` e83a9535, Opus 5.5, subscription auth, claude-code 2.1.309,
  2 h budget): **reward 0.48** (`mean_request_credit` 0.4815). Per unit: u0 cooktop 0.67, u1 microwave 0.11,
  u2 oven 0.67; per request type: burner off 1.0, burner on 0.57, preheat 1.0, oven off 0.57, microwave start
  1.0, microwave stop 0.0. Episodes are bimodal: four at 1.0, four at 0.0 (the controller stalls on the first
  request until every deadline passes), one at 0.33. 116 of 116 probe pairs differed; no aborts, no simulation
  faults, seccomp guard on, verify 134 s.
  - **The agent stopped after 934 s of its 7200 s** (13 %) with 4 of its 16 lab trials used. It ran `date`
    once, at the start, and after about 15 minutes wrote "~13 min left — finalizing": it misjudged the clock
    (nothing in its context reported a deadline). The score is therefore a lower bound for this budget.
  - **What it built:** per-unit calibration measured in the lab: u0's knob height (z 0.937) and its
    left-to-right knob-to-location map (two locations confirmed, three inferred), u1's start / stop buttons as
    wrist-image templates taken at one placement, u2's temperature knob (the right-hand one) as templates.
    The controller finds a control with the wrist camera (blob detection on u0, two-view template matching
    and triangulation on u1 / u2), grips and turns knobs until the wrist torque shows a stop, and presses
    buttons until contact. It does not handle the practice units at all. Microwave failures are consistent
    with templates taken at a single placement not transferring to the placement jitter.
  - **Reading:** one run, below the 0.5 target, but with most of the budget and trials unused; it does not
    establish the difficulty. In 15 minutes the agent learned each hidden unit through the lab and solved
    two of the three kinds in most episodes, which suggests a full-budget run would score higher. Open owner
    decision: a second trial before changing anything, or raise the difficulty now (fewer lab trials, more
    placement jitter, more request types, or hidden units drawn per grading run).
