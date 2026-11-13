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
- Reference (`solution/`): privileged, knows `units.json`; estimates the unit pose from the fixed
  cameras at t = 0 ([box]'s `perceive.py`), then runs the primitives. It spends no lab trials.
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
