# Design: kitchen appliance tending (draft, 2026-11-12)

Status: **draft for the owner's review.** No task folder exists yet. Decisions marked
**[owner]** are open; everything else is a recommendation that the pilots in §8 can still
overturn. Feasibility evidence: `experiments/physical-execution/APPLIANCE_PILOT_TEST.md`.

## 1. The task in one paragraph

A Franka Panda on an Omron mobile base (RoboCasa's `PandaOmron`) stands in a kitchen it has
not seen. Over one episode it receives a queue of household requests in plain text ("turn on
the front-left burner", "start the microwave", "open the oven"), one at a time, each with a
deadline. It must drive to the right appliance, find the right control from its cameras,
operate it through ordinary contact, and leave every other appliance as it should be. The agent
delivers a controller (`/home/user/submission/run.sh`) that the hidden grader drives over the
closed_loop stdio wire, in hidden kitchens, after the agent is gone. This is the second task of
the physical-execution family after `object_finding_and_organizing_on_messy_desk`, and it
reuses that task's structure wherever it can.

- Proposed slug: `kitchen_appliance_tending` **[owner: confirm the name and register it in the
  sheet]**. Platform: `arm` (no mobile-manipulator key exists; the scored skill is arm
  manipulation).
- Verify pattern: closed_loop. Verification: `environment_mode = "separate"`, like every task
  on `main`.

## 2. Why this task, and where the difficulty comes from

The pilot shows the controls are physically operable through ordinary actions (no joint
edits, no teleports), but only with privileged geometry. The agent gets none of it. The
difficulty is real engineering, not penalties:

- **Perception of small controls.** A stove knob is a few centimetres wide and its "on" angle
  is 0.35 rad; the agent must find it, and the right one of four to six, from RGB-D alone.
- **Unseen appliance models and layouts.** Knobs sit on the stovetop's front edge in one layout
  and on a rear panel in another; microwave buttons and oven handles differ by model. The
  pilot's own privileged controller generalized to 17/28 unseen scenes.
- **Contact-rich actuation with weak fingers.** 20 N per finger turns a knob with 1.0 N m
  friction slowly (about 375 steps for 0.7 rad); the oven door is 6.6 kg and not gravity
  compensated.
- **Mobile base and blocking.** An open oven door blocks the robot; the base must stand beside
  the door's sweep, and kitchen islands limit where it can stand.
- **Long horizon with state.** Several requests per episode; earlier actions change what later
  ones need (a door left open, a burner that must stay on).

Expected frontier result: well below 0.5 against a privileged reference (§6). That must be
measured (§8, P5), not assumed.

## 3. Episode

- **Scene.** One RoboCasa kitchen per episode: a layout with an oven, a style, RoboCasa's
  default base spawn deviation and arm-joint noise. No task objects in v1.
- **Requests.** A queue of K requests (proposal: K = 5) drawn from the v1 grammar below. One
  request is open at a time; the next opens when the current one closes. Each has a deadline
  in control steps (proposal: 1500 steps = 75 s at 20 Hz; the pilot's passing trials took
  355-1226 steps from a spawn next to the appliance, so the deadline includes driving).
- **Initial state.** Some controls may start in a non-default state (a burner on, the oven door
  open) so that "off" and "close" requests make sense; the initial state is part of the hidden
  episode.

v1 request grammar (from what the pilot established; §8 P2 extends the reverse directions):

| Request | Judged by (RoboCasa's own predicate) | Pilot evidence |
|---|---|---|
| turn on burner `<location>` | `Stove` knob joint ≥ 0.35 rad (burner shown on) | 10/10 tuned, 5/7 unseen |
| turn off burner `<location>` | knob joint < 0.35 rad | untested (P2) |
| start the microwave | `Microwave.get_state()["turned_on"]` | 10/10 tuned, 6/7 unseen |
| open the oven door | `Fixture.is_open` ≥ 0.90 of range | 10/10 tuned, 6/7 unseen |
| open the microwave door | `Fixture.is_open` on the microwave | untested (P2) |

Left out of v1: sliding the oven rack (0/7 on unseen scenes; the hook does not transfer),
closing the oven door (needs lifting 6.6 kg; untested). Candidates once a pilot passes them:
sink faucet on/off and the coffee machine button (RoboCasa has atomic tasks with predicates
for both).

## 4. Interface (the wire)

The template's closed_loop wire, as the messy-desk task uses it:

```
READY
{"type": "reset"}                        -> {"ok": true}
{"type": "act", "t": 0, "obs": {...}}    -> {"action": [12 numbers]}
{"type": "close"}                        -> {"ok": true}, then exit
```

- **Observation:** `request` (`null` or `{"text", "deadline_t"}`), RGB and depth (16-bit mm
  PNG, base64) from `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`
  (resolution decided by the render pilot, P1); arm joint positions and velocities, gripper
  width, end-effector pose in the base frame; base odometry (integrated base motion since
  reset, with drift noise) **[owner: odometry, or no base pose at all]**. No fixture poses,
  no joint values of appliances, no segmentation.
- **Action:** RoboCasa's `PandaOmron` composite action at 20 Hz (arm delta pose, gripper, base
  velocity, torso), clipped; the instruction documents each element.
- Probe lock: a probed step is asked twice with the same `t`, perturbed proprioception in the
  copy, as in messy desk. An aborted episode scores 0 and is never dropped; at most 3 restarts.
- Controller time budget per episode (wait time for replies only; simulator time not counted),
  as in messy desk, so a slow host does not cost the agent.

## 5. Scoring

Per request: **success = 1** if, before its deadline, the requested control reaches its target
state, holds it for 20 consecutive control steps with the hand at least 0.10 m from the control
(the pilot's acceptance rule), **and** every other tracked control is in its expected state at
that moment (the state after all earlier requests). Otherwise 0. The request closes on success
or at its deadline.

- Metric: `request_success_rate` = successes / requests over the hidden set, direction higher.
  An outcome on a positive scale; untouched = 0, because every request needs a state change the
  robot causes (an "off" request starts from "on").
- "Every other control as expected" is what makes it tending rather than five isolated
  primitives: turning on the wrong burner, or knocking a knob while passing, fails the request.
- Not scored: time to completion (wall-clock is banned as a metric; the deadline is a step
  count), smoothness, collisions with the kitchen beyond their effect on control states
  **[owner: add a collision-free condition?]**.

## 6. Reference, anchor and hidden set

Follow the messy-desk / spacecraft pattern:

- **Privileged reference** (`solution/privileged/`): the pilot's scripted controller, extended
  to sequence requests and drive between appliances. In a token run only (`.oracle_token`,
  sha256 in `tests/hidden/`), the grader puts the simulator's truth (fixture geometry, joint
  values) into the observation. Untokened runs never see it.
- **Fixed hidden set** (`tests/hidden/episodes.json`): N episodes (proposal: 8 kitchens × 5
  requests = 40 requests), 64-bit seeds from `os.urandom`, each world's digest recorded,
  selected by a written rule before any candidate runs (spacecraft's `select_set.py`). The
  hidden set uses layouts and styles that the reference was not tuned on.
- **Anchor** = the reference's own `request_success_rate` on that set; `full_at` sized from
  its run-to-run spread (MuJoCo is deterministic on one binary, so expect a tight `full_at`
  such as 0.95-0.98, as in spacecraft).
- A reference that cannot reach 1.0 on the set is fine: the anchor is its measured value.
  Candidate episodes it fails can stay in the set (unlike spacecraft's dropped draw)
  **[owner]**.

## 7. Sandbox, image and budget

- **Image:** `container-ubuntu22-base:0.1.1`, py3.12 `/opt/venv` with robosuite 1.5.2 and
  RoboCasa 1.0.1 (the pilot's commits `5ce6643` / `3443cdf`), pinned in `requirements.txt`;
  the agent-UID import and render self-checks of `robocasa_object_dynamics_from_video`.
- **Assets:** RoboCasa's kitchen fixtures and textures go into the asset dataset and are
  fetched at build through `environment/fetch_assets.py` + `assets.sha256` (no build-time
  download from elsewhere, per #59). Size and the assets' licences must be checked
  **[owner: licence sign-off]**.
- **Agent's practice tools** (`environment/setup/payload/`): a kitchen lab (the same simulator
  with dev seeds, a practice grader speaking the real wire and printing the raw metric only, a
  task card with camera intrinsics and the robot's limits). The agent has RoboCasa itself in the
  image, so it can build any stock kitchen locally (§9, risk 1).
- **Resources:** RoboCasa is CPU physics; rendering decides the rest (P1). Proposal: 8 CPUs,
  16 GB, `gpus = 1` only if EGL on the GPU works in the sandbox and is needed for speed.
- **Agent budget:** proposal 2 h **[owner]** (messy desk uses 1 h; its 4 h runs ran out while
  still debugging). **Verifier budget:** from P1's measured steps per second with cameras.

## 8. Plan of work (each phase ends with a written result in this file's §10)

- **P1 — image and render pilot.** Build the image on the dev host (TASL: 96 cores, 4 × RTX 6000
  Ada); run the pilot `--no-render`, then with 3 cameras through OSMesa and through EGL; record
  control steps per second, memory, asset size. Decides resolution, resources and the verifier
  budget.
- **P2 — reference coverage.** Add knob off and microwave door open to the privileged
  controller; add base navigation between appliances and request sequencing; run dev seeds,
  then a frozen run on unseen layouts and styles. Decide the final v1 grammar from the result.
- **P3 — camera-only feasibility.** A simple non-privileged controller (perception from the
  task's own cameras, no simulator truth) that passes at least some requests of each type. This
  shows the task is solvable as instructed, and it is the first practice-grader user.
- **P4 — the task folder.** From `templates/task/`: environment server and wire, grader,
  practice grader, instruction, task card, oracle token, hidden-set selection, anchor.
  `ale lint`, `scripts/lint_domain.py`, `scripts/task_assets.py check`.
- **P5 — gates and difficulty.** `ale validate` (untouched 0.0, oracle 1.0), then frontier
  runs (`ale run`); target reward roughly below 0.5.

## 9. Risks

1. **The agent has the simulator.** RoboCasa's stock kitchens are in the image, so a controller
   could recognise the layout and style from an image and read exact fixture geometry from its
   own copy. Options **[owner]**: (a) accept it as a map prior, which is legitimate robotics but
   removes most of the perception challenge; (b) perturb hidden kitchens at reset (fixture model
   swaps within RoboCasa's registry, small placement jitter, lighting and texture changes) and
   say so in the instruction. Recommendation: (b).
2. **Determinism.** The anchor and `full_at` assume the reference repeats its score in a fresh
   container. Check with two fresh verifies (as spacecraft did) before freezing.
3. **Render speed.** Three cameras through OSMesa may be slow enough to dominate verify time;
   P1 measures it.
4. **The oven door blocks the robot.** A later request can become infeasible after "open the
   oven door". The episode generator must check each queue with the reference, or order oven
   requests last.
5. **Seeds.** The agent holds a simulator copy, so episode seeds must be 64-bit and hidden values
   drawn through a keyed hash (PROGRESS gotcha).

## 10. Results log

(empty: P1 not started)
