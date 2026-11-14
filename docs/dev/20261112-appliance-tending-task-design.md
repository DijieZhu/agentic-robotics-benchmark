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
  oracle runs, frontier runs (`ale run`); target: a frontier model (Opus 5.5) scores reward at most 0.3
  (owner rule 2026-11-13, after trial 6; it replaces the project-wide "roughly below 0.5" for this task).

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
- **Frontier trial 2** (2026-11-13, `ale run` a9132c62, the same task and settings as trial 1): **reward 0.93**
  (`mean_request_credit` 0.926). Per unit: u0 1.0, u1 0.78, u2 1.0; every request type 1.0 except microwave
  stop 0.5; eight episodes at 1.0, one at 0.33 (a microwave episode). 52 of 53 probe pairs differed (lock ok);
  no aborts or faults; verify 87 s. Agent phase 4198 s of 7200 s; **all 16 lab trials used.**
  - The agent again declared itself done after about 15 minutes (its first report expected u1 to fail), but it
    had scheduled its own wake-up first; on waking it ran `date`, found about an hour had passed rather than
    two, and spent the rest fixing the microwave: a press-by-press scan in the lab located the start and stop
    buttons (the stop target is about 3 mm wide).
  - **Same strategy as trial 1:** per-unit calibration through the lab. Reference image patches of u0, u1
    and u2 taken in lab trials (`refs.pkl`) are matched in the fixed cameras and triangulated each episode;
    the knob-to-location map, the oven's temperature knob and the button positions were measured on the
    hidden units themselves. The controller does nothing on the practice units.
  - **Reading:** the task is too easy as built. Because the lab serves exactly the scored units, the
    intended difficulty (controls on models the agent has not seen) reduces to calibrating three known units
    and coping with the placement jitter. Both trials took this route.
- **Owner decision (2026-11-13): keep the calibration framing, tighten the budget** (as in
  arm_hidden_mechanism_box, the capability is system identification with few real trials; ALE asks for no
  out-of-distribution test, the project asks for a frontier reward roughly below 0.5). Changes (`12e02b3`):
  six scored units, two per kind (u3 Stovetop002, u4 Microwave087, u5 Oven053: the next models in each kind's
  committed draw order, `selection.json`), **8 lab trials** (was 16), 12 graded episodes (2 per unit); agent
  time stays 2 h.
- **Anchor re-measured** (`aab6476`): 14 oracle runs of the same content (168 episodes), mean
  `mean_request_credit` **0.9911**, worst run 0.9722; `full_at` 0.95. Every lost credit is u5 (Oven053, 7 of 28
  episodes lost one request): near the jitter limit (e.g. yaw -8 degrees, y +3.7 cm) the reference finds no
  feasible grasp pose for the temperature knob, in process as over the wire. The next oven in the draw
  order (Oven034) fails at the same placement, and smaller turns per grasp do not help, so the measured
  anchor stands rather than a planner change. `ale validate` (`validate-c8a6192e`): untouched 0.0, oracle 1.0.
- **Frontier trial 3** (`ale run` ed315672, Opus 5.5, 2 h, six units, 8 trials): **reward 0.43**
  (`mean_request_credit` 0.403, ratio 0.406). Cooktops u0 and u3 1.0 in all four episodes; microwaves u1 and
  u4 0.0; ovens u2 0.17 and u5 0.25. 143 of 143 probe pairs differed; no aborts or faults; verify 136 s. Agent
  phase 1746 s: it stopped when its 8 trials were spent.
  - It spent six trials on one first image of each unit (controls marked by hand, triangulated from the two
    fixed cameras) and the last two on the cooktops (each knob turned on and off, which gave u0's
    knob-to-burner map). Microwave buttons and oven knobs were never tested on a scored unit: it assumed the
    practice microwave's keypad layout and turned both front knobs of each oven. It also believed the lab
    gives no evidence of microwave or oven state, though a request that closes early does (trial 2 used
    that).
  - Practice grader on the practice units: cooktop and microwave 1.0 in 8 of 8, oven 1.0 in 6 of 8.
  - **Reading:** inside the target band, with the expected failure mode (too few trials to calibrate every
    unit, so the untested kinds fail). One run; trials 1-2 differed by 0.45 on the same task, so the spread
    of a single run is large.
- **Frontier trial 4** (`ale run` 322e0b17, same settings as trial 3): **reward 0.41** (`mean_request_credit`
  0.389, ratio 0.392). Per unit: u0 0.67, u3 1.0 (cooktops); u1 0.0, u4 0.5 (microwaves); u2 0.17, u5 0.0
  (ovens). 142 of 142 probe pairs differed; no aborts or faults; verify 118 s. Agent phase 2881 s; all 8 trials
  used. It mapped both cooktops' knobs in the lab (flames, knob symbols), found each oven's temperature knob
  (one "Turn off the oven" closed on u2; on u5 every lab attempt failed to reach the knob), and guessed the
  microwave buttons from wrist images (u4 taking the green button as start, which held in one of two
  episodes); practice grader 1.0 on all six practice episodes.
  - **Reading:** trials 3 and 4 agree (0.43, 0.41), both below the 0.5 target with the same pattern:
    cooktops calibrated and solved, microwaves and ovens under-tested for lack of trials. Time was not the
    limit (29 and 48 of 120 minutes); the 8 trials were. Proposed next: P4.
- **Owner decisions (2026-11-13, after trial 4):** state indicators (a microwave's lit display bar while it
  runs, an oven's lit lamp while preheated; RoboCasa shows neither, so the lab gave no feedback on them);
  lab trials = units + 5; the frontier score should fall below 0.3 through other design. Changes (`f37a63a`):
  indicators (visual only, 3 cm below the visible front's top edge, placed clear of every control);
  compositional cooktop requests (two burners on, two off, all off; probability 0.5 when valid; credit over
  every named control; 1000-step deadline); nine scored units, three per kind (u6 Stovetop025, u7
  Microwave071, u8 Oven034, next in the draw order); 14 lab trials; 18 episodes; agent time 3 h. Reference:
  48 in-process cooktop episodes with compositional requests and 12 wire episodes on new units, all 1.0.
  Anchor (`6e5807d`): 7 oracle runs (126 episodes), mean 0.9987 (one request lost on u5), `full_at` 0.95;
  `validate-0d5f702e` passes.
- **Frontier trial 5** (`ale run` 2be5b9ac, Opus 5.5, 3 h): reward 0.52, **not a valid measurement**. The
  lab's reply schema lists request texts as an enum and lacked the compositional ones, so a cooktop trial
  whose open request was compositional failed with "internal lab error" (the trial still charged): u3 and u6
  were never seen (three trials lost), two u0 trials crashed after a deadline. The grader does not use that
  schema and scored normally. Per unit: u0 1.0, u1 1.0, u4 1.0 (microwaves now work with the indicator),
  u7 0.0 (its buttons' contact areas are not where its keys appear), u2 0.17, u5 0.58, u8 0.67, u3 and u6
  0.0 (no data); compositional requests: two off 0.0, two on 0.29. Agent phase 3917 s, all 14 trials used.
  Fix (with this record): `episode.request_texts()` lists every text an episode can open, the lab's schema
  uses it, `_open_next` asserts membership, and `lab_check.py` runs the plugin through the server's own
  schema check (the pre-fix plugin reproduces the error; the fixed one passes 14 whole episodes past every
  deadline). The grader's behaviour and the anchor are unchanged.
- **Frontier trial 6** (`ale run` 944638e3, the fixed lab, otherwise as trial 5): **reward 1.0**
  (`mean_request_credit` 0.944, ratio 0.946 >= full_at). Eight of nine units 1.0 in both episodes; u1 lost one
  episode (stop 0.73). Every request type 1.0 except microwave stop, compositional ones included. 111 of 113
  probe pairs differed; verify 157 s; agent phase 3174 s, all 14 trials used. The agent recorded per-unit
  knowledge from lab exploration (control positions, knob directions, knob-to-burner maps, where the flame,
  lamp and display bar show), estimated each episode's placement against its reference images (wrist camera
  for microwaves and ovens, fixed cameras for cooktops), and checked every action by looking for the colour
  at the indicator, retrying when the state was wrong. **Reading:** with visible state the task is easy; the
  indicators turned every lab trial into a fully observed experiment. Owner rule (2026-11-13): above 0.3,
  add per-episode lighting and camera variation (direction 2) and setting requests (direction 3).
- **Directions 2 + 3** (`9a72828`, anchor `846a23d`): per-episode light (position, aim, strength 0.5-1.2x,
  tint +-15 %), headlight and fixed-camera offsets (+-4 cm, about +-3 degrees; poses still in every
  observation); burner settings (thirds of the on-range; the flame grows with the setting) and oven heat
  settings (0.25-0.5 / 0.5-0.75 / 0.75-1.0); request categories simple 0.3 / compositional 0.35 / setting
  0.35; deadlines 1000 steps (1500 compositional), lab step cap 4800. Reference: goal-angle turns, regrasp on a
  stalled twist, at most 1.2 rad per grasp for a setting; 126 recheck episodes about 0.98. Anchor: 7 oracle
  runs (`74fc7475`), mean 0.9802, worst 0.9352, `full_at` 0.90; `validate-5c621f96` passes. Practice
  baseline (camera poses from the observation, edge-based comparison, 120 s step limit because its first
  placement estimate takes 15-40 s): 0.81 on the practice units.
- **Frontier trial 7** (`ale run` dee3156a): **void**. The subscription's rate limit rejected the agent's
  calls (429) 49 minutes into the agent phase; the engine ended the episode with `env_error`
  (`SubscriptionProviderLimitError`) and no verification ran.
- **Direction 1: hidden unit mechanisms** (owner 2026-11-14; `ba19db7`, `7c85eab`, anchor `915a96f`): a unit
  may carry one mechanism, stated nowhere for the hidden units: push_to_turn (a knob unlocks only while the
  gripper pushes along its axis; it relocks over 1 s with hysteresis), reversed (every knob turns the other
  way), hold_start (the microwave starts after its button is held 15 steps). Six of nine scored units carry
  one (u0, u5 push; u2, u3 reversed; u1, u4 hold); practice units one each. Anchor: 7 oracle runs
  (`e92fb05d`, on `7c85eab`), mean 0.9974 (one lost request, u2), `full_at` 0.90; `validate-c70cab67`
  passes.
- **Frontier trial 8** (Opus 5.5, 3 h budget, the mechanisms build `915a96f`). The first attempt (`616e01ed`)
  was void like trial 7: 429 after 4523 s of agent phase, `env_error`, no verification. Rerun on another
  subscription token (`ale run` 1d5abaed): **reward 0.27** (`mean_request_credit` 0.241, ratio 0.241).
  Per unit: cooktops u0 0.67, u3 0.83, u6 0.0; oven u2 0.67, u5 and u8 0.0; microwaves u1, u4, u7 0.0. Per
  request: all off 1.0, burner off 1.0, burner level 0.5, burner on 0.33, burners on 0.33, oven level 0.3,
  oven off 0.2, preheat 0.0, microwave start 0.0. 423 of 436 probe pairs differed (lock ok); no aborts, no
  faults, seccomp guard on; verify 281 s.
  - **The agent stopped after 1247 s of its 10800 s** (12 %) with **4 of its 14 lab trials used**, and
    reported that "the 3 hours ran out". It ran `date` once, at the start, and never again; it briefed its
    subagent to finish "in ~40 minutes". This is trial 1's clock misjudgment again.
  - **What it built:** a controller for four units only (cooktops u0, u3; ovens u2, u5), parked on the other
    five. Placement from the fixed cameras against reference images saved in the lab; knobs gripped and
    pushed in at about 14 N along the axis (so push-to-turn unlocks), turned to the torque stop for "off" and
    "high", by a measured amount from the off stop for the other settings. The microwave hold-to-start
    failed on the practice unit and was never fixed.
  - **Reading:** at or below the 0.3 target, but a lower bound: 88 % of the time and 10 of 14 trials were
    unused, and five units were never attempted. It does not establish that the task meets the target.
- **Budget tightened** (owner 2026-11-14, `f8eecac`): **10 lab trials** (units + 1; was 14) and **2 h** of agent
  time (was 3 h). The reference spends no lab trials and the grader is unchanged, so the anchor stands;
  `validate-d3b7721a` passes (untouched 0.0, oracle 1.0).
- **Frontier trial 9** (`ale run` 4abdc596, Opus 5.5, 2 h, 10 trials): **reward 0.27** (`mean_request_credit`
  0.241, ratio 0.241). Per unit: cooktops u0 0.0, u3 0.67, u6 0.75; ovens u2 0.25, u5 0.17, u8 0.33; microwaves
  u1, u4, u7 0.0. Per request: burner level 0.79, burner on 0.5, burners on 0.5, preheat 0.6, oven level 0.25;
  all off, burner off, burners off, oven off, microwave start and stop 0.0. 502 of 511 probe pairs differed
  (lock ok); no aborts or faults; verify 333 s.
  - **All 10 trials were spent in the first 11 minutes** (one reference capture per unit, then one full
    episode on u3); the agent stopped after 1156 of 7200 s because it had no trials left, not because it
    misread the clock (it saved its start time). Practice grader on its controller: 0.81.
  - **What it built:** placement from 11 wrist-camera views matched in 3D against each unit's lab capture;
    knobs pinched and always pushed along the axis (push-to-turn), zeroed at the stop, then turned to the
    target; "reversed" learned per unit from whether the flame or lamp lights after zeroing; buttons held
    about 22 steps (hold-to-start). Controls of six units marked by hand from the captures; the cooktops'
    knob-to-burner maps and the ovens' temperature knob assumed from the practice units; the microwaves left
    out (it could not tell start from stop in the images).
  - **Reading:** at or below the 0.3 target with the budget binding (trials, not time), the intended
    limit. Trials 8 and 9 agree (0.27, 0.27) but trial 8 was a lower bound; one more run on this build would
    show the spread.
- **Agent budget 1 h** (owner 2026-11-14, after trial 9; `bb32926`): the verifier's timeout and the grader's
  own budget (6000 s, never close to binding: verify took 87-436 s so far) are unchanged; `validate-51561368`
  passes (untouched 0.0, oracle 1.0).
- **Frontier trial 10** (`ale run` d64acd2d, Opus 5.5, 1 h, 10 trials): **reward 0.0** (`mean_request_credit`
  0.0 on all nine units and every request type). 580 of 580 probe pairs differed; no aborts or faults; verify
  436 s.
  - **All 10 trials spent in the first two minutes** (a reference capture of each unit, then one test on u7);
    **the agent stopped after 628 of 3600 s** reporting that "the time ran out" (it ran `date` once, at the
    start): the clock misjudgment of trials 1 and 8 again.
  - **What it built:** a controller for the microwaves only (u1, u4, u7; it parks on cooktops and ovens):
    placement from feature matching against the capture, a 3 x 7 grid of presses across a hand-marked
    keypad, each held about 18 steps, checked by counting amber pixels at the display bar. Its only lab test
    (u7, an earlier version) missed every button: the keypad was assumed 3 cm nearer than the arm found it.
    The fix was never tested, and the display-bar check counted no amber pixels even in its dev sim.
  - **Reading:** a floor score from a short run on one appliance kind (11 of 60 minutes), not a measurement
    of the task's difficulty at 1 h. Trials 9 and 10 (0.27, 0.0) are both at or below the 0.3 target; the
    spread is the open question.
- **Agent budget back to 2 h** (owner 2026-11-14, after trial 10; `86c3f40`); 10 lab trials stay;
  `validate-fec59fbd` passes (untouched 0.0, oracle 1.0).
- **Frontier trial 11** (`ale run` 8a96f820, Opus 5.5, 2 h, 10 trials): **reward 0.0** on every unit and
  request type. 585 of 585 probe pairs differed; no aborts or faults; verify 334 s.
  - **The agent stopped after 806 of 7200 s with 2 of 10 trials used**, reporting that it "couldn't get the arm
    to work any hidden unit's controls in the time available" (it ran `date` once, at the start). Its
    controller deliberately holds the start pose for every request.
  - In the dev sim it turned the practice cooktop's knobs (push-to-turn included) and pressed the practice
    microwave's buttons (hold-to-start included); its fingertips slipped on the practice oven's knob. In the
    lab, u0's knob would not turn by press, grasp or grasp-plus-push (u0 is a push-to-turn unit), and a press
    at u1's estimated button position met no surface. Seven units were never examined.
- **Outcome (owner rule 2026-11-14: done if trial 11 scores below 0.3): the difficulty step is complete.**
  Final build `86c3f40`: nine scored units with mechanisms, 10 lab trials, 2 h agent time; anchor 0.9974,
  `full_at` 0.90. Frontier results on the 10-trial build: 0.27 (trial 9, 2 h), 0.0 (trial 10, 1 h), 0.0
  (trial 11, 2 h); all three agents stopped early (11 to 19 of 60 or 120 minutes), two of them believing the
  time was up. The two zeros come from agents that quit before building a working controller, not from a
  ceiling the task imposes (the reference scores 1.0 and the practice-unit skills worked in their dev sims).
- **Frontier trial 12** (`ale run` 7f305f80, the final build `86c3f40`, 2 h, 10 trials): **reward 0.32**
  (`mean_request_credit` 0.287, ratio 0.288; reward = ratio / `full_at` 0.90). Per unit: microwaves u1 0.17,
  u4 0.67, u7 0.67; cooktops u0 0.17, u3 0.0, u6 0.67; ovens u2 0.08, u5 0.0, u8 0.17. Per request: microwave
  start 0.71, stop 0.36, burners off 1.0, all off 0.5, burner off 0.5, burner level 0.25, burners on 0.25,
  preheat 0.25, burner on 0.0, oven level 0.0. 504 of 504 probe pairs differed; no aborts or faults; verify
  553 s.
  - **The first run on this build to use the time:** agent phase 4059 of 7200 s; it ran `date` ten times and
    stopped with about 50 minutes left, reporting itself done. Seven trials went on first-frame captures in
    the first three minutes; two trials it held open for later testing were dropped by the lab's 900 s idle
    timeout; the last, a full u0 episode, turned no knob (the fingertips rested on the cooktop surface). Its
    controller scores 0.98 on the practice units.
  - **What it built:** placement by fitting hand-marked image patches in both fixed cameras; knobs grasped
    across the bar seen by the wrist camera, pushed about 13 N along the axis while turned, stops found by
    stalls, on/off settled by flame and lamp; per-unit learning carried from a unit's first graded episode to
    its second (knob direction, oven knob range and which knob is the temperature one); microwave presses
    held 2 s around a guessed keypad spot, checked by the amber bar, the working spot remembered. Knob-to-burner
    maps guessed from the practice cooktop; push-to-turn ovens not handled.
  - **Reading:** just above the 0.3 target. The four Opus 5.5 runs on the 10-trial budget: 0.27, 0.0 (1 h),
    0.0, 0.32 (2 h runs mean 0.20). Owner decision pending on whether 0.32 reopens the difficulty step.
