# Rescue a kitchen task from where a learned policy failed

A mobile manipulator (RoboCasa `PandaOmron`: a Franka Panda arm on a holonomic base with a torso lift) works in home
kitchens under a frozen learned policy. Some of its runs fail: a mug knocked over against a coffee machine, an object
dropped short of its receptacle, a utensil pushed around a drawer. When that happens the robot stops where it failed and
your controller takes over. Finish the job from there, seeing only what the robot sees.

Deliver a controller: an executable `/home/user/submission/run.sh` plus any files beside it. It speaks the wire protocol
below on stdin/stdout. The grader runs it on **27 hidden failure states**, recorded from real failed rollouts
of the policy in kitchens that are not in your sandbox. You have **8 hours** of wall time, **4 CPU cores and 12 GB of
memory, no GPU and no network**. One kitchen simulator process needs about 6 GB, so run at most one at a time.

## What is in your home directory

| Path | What it is |
|---|---|
| `kitchen_sim.py` | The simulator wrapper the grader itself uses: builds a kitchen from `(task, seed, split)`, restores a failure state by replaying its recorded action prefix, steps it, renders cameras on request, builds the wire observation. `Kitchen.raw` is the RoboCasa environment, so on practice kitchens you have the full simulator state for development and debugging. |
| `sim_worker.py` | The child process that serves one kitchen to a grader; the practice grader uses it. |
| `practice_states/` | 14 practice failure states (`index.json`, action prefixes, recorded physical states, cards), real failures of the same policy on five of its jobs (every job of the hidden set among them), in other kitchens (RoboCasa's `pretrain` split). They were not filtered by the hidden set's rule below, so some are easier or harder than graded states. |
| `practice_grader/grade.py` | Runs your `run.sh` over the wire on the practice states and prints what happened (`/opt/venv/bin/python3 practice_grader/grade.py --submission ~/submission [--states p00,p03] [--max-steps N]`). It imports `wire_rules.py`, the same file the grader imports, so deadlines, the action check, line and queue caps, restarts, the image cap, pausing between requests and the score are the grader's; only the account-wide process count, the check that every process descends from the grader's launcher and the asset checks are the grader's alone. |
| `wire_rules.py` | The grader's wire limits, action check and score, byte for byte. |
| `helpers/wire_skeleton.py` | Minimal controller on the wire, to copy and extend. |

Run Python as `/opt/venv/bin/python3` (`numpy`, `scipy`, `opencv` (`cv2`), `mujoco`, `robosuite`, `robocasa`, `pillow`).
The sandbox holds the assets of the practice kitchens only. The graded kitchens use RoboCasa's held-out kitchen layouts,
styles and object instances, which are not installed here (asking RoboCasa for `split="target"` will fail).

## The hidden states and the jobs

Each hidden state is the final state of a real failed rollout of the policy, recorded for this task with fresh random
seeds in RoboCasa's held-out kitchens (none of it is in the policy's public recordings): the policy ran to the job's
time limit without success, and the state is the arm wherever it ended and the object where it ended up. A state was
kept only if all of these hold:

- the failure physically changed the problem: the target object moved at least 5 cm horizontally from where it started,
  or its orientation turned by at least 30 degrees (about any axis), and its origin is at least 0.35 m high (not on the
  floor);
- it is recoverable: a recorded rescue from exactly that state finished the job. A rescue is the same learned policy run
  again from the state with a new noise seed, in some cases after a short scripted prefix (open the gripper and pull the
  hand 12 cm up and 15 cm back, and possibly return the hand to its starting pose). Replaying that rescue finished the
  job in 3 of 3 runs, and also in 2 of 2 runs with a 1e-9 perturbation of the restored joint positions;
- it is not solved, over the full step budget, by standing still with the gripper open, by opening the gripper and
  backing away to the hand pose the rollout started from, or by blindly retrying the grasp at the failed hand pose.

A job is done when its success check holds for **10 consecutive control steps** within the step budget. The checks are
RoboCasa's own for each job, used unchanged; the mug job also requires RoboCasa's upright check, because a mug lying
or leaning under a dispenser is not a finished coffee setup (`gripper site` is the point between the fingertips;
`origin` is the object body's reference point):

- **CoffeeSetupMug** (6 of the 27 states): the mug's origin is within 0.04 m horizontally and within 0.10 m vertically of the coffee machine's receptacle site under the dispenser, the mug is upright (RoboCasa's `check_obj_upright`: the roll and the pitch of its body are each below 15 degrees), and the gripper site is more than 0.25 m from the mug's origin.
- **PickPlaceDrawerToCounter** (11 of the 27 states): the object touches any counter, and the gripper site is more than 0.25 m from the object's origin.
- **PickPlaceCounterToStove** (10 of the 27 states): the object touches the pan, its origin is within 0.07 m horizontally of the pan's origin, and the gripper site is more than 0.25 m from the object's origin.

Note what the checks do not ask for, so you neither rely on it nor over-deliver: the mug may face any way round (its
handle may point anywhere; only its tilt counts); for the drawer job, contact with any geometry of any counter counts (its
top, its sides or its base panels); for the stove job, the pan is not required to stay on the stove.

## Wire protocol

`run.sh` is started once as `bash run.sh`, working directory `/home/user/submission`, as your user, with no network
(only AF_UNIX sockets work); see "The grading sandbox" below for what it can rely on. Every line on stdin/stdout is one JSON object except the handshake; stdout carries only
the protocol, use stderr for logs. Print `READY` within **40 s** of start, as one of the first 200 lines on stdout. Then:

- `{"type": "reset"}` starts an episode: reply `{"ok": true}`.
- `{"type": "act", "t": T, "obs": {...}}`: reply `{"action": [...]}`. `t` counts control steps from 0 within the episode
  and is asked once per step. The reply to each request must arrive within **10 s**.
- `{"type": "close"}`: reply `{"ok": true}` and exit. It is sent once, after the last state, and only when the last
  exchange succeeded: after an aborted episode the controller is stopped without it.

Your process runs for the whole grading run and may keep state freely between episodes, and may also keep files in
its submission folder. Lines you write while no request is outstanding are discarded; at most 64 reply lines are
buffered (beyond that your writes block). After an aborted episode (a crash, a late or malformed reply) the next `reset` restarts `run.sh`,
which must print `READY` again; after **3 restarts**, or after a restart that fails, every remaining episode scores 0.

**Action.** A list of 12 numbers, optionally followed by a 13th camera request (below). The grader clips the 12 numbers
to `[-1, 1]`:

| Index | Meaning |
|---|---|
| 0-2 | end-effector displacement in the base frame (x forward, y left, z up): input 1.0 = 0.05 m |
| 3-5 | end-effector rotation vector in the base frame: input 1.0 = 0.5 rad |
| 6 | gripper: at least 0.5 closes, anything below opens |
| 7-9 | base motion in the base's own frame: forward, left, yaw. At full input the base moves at about 0.03-0.036 m per step and yaws about 0.06 rad per step (measured) |
| 10 | torso lift command |
| 11 | arm goal mode: below 0.5 the arm's goal is updated from the pose it has reached, at least 0.5 from its previous goal (which keeps the arm steady relative to the base while the base moves). The base and torso commands act in both modes (measured) |

Control runs at 20 Hz; displacements are per control step, and the arm controller reaches roughly a fifth of a commanded
displacement per step.

**Camera request.** The 13th number is a bit mask for the images of the *next* observation: 1 left agent camera, 2 right
agent camera, 4 wrist camera, 8 add depth to the requested cameras. Leave it out or send 0 for no images (the image
pipeline costs time). The first observation of an episode always has all three RGB-D views. At most **3000 images** are
rendered per episode (each camera counts one); beyond that, images come back empty.

**Observation.** A JSON object:

- `t`: the step.
- `proprio`: `ee_pos_base` (gripper site position, base frame, m), `ee_rotvec_base` (axis-angle, base frame), `gripper_qpos` (the two finger joint positions), `base_pos_world`, `base_rotvec_world` (world pose of the base, the frame of the two `*_base` fields), `arm_q` (7 joints).
- `wrist_ft`: the wrist force/torque sensor (N, N m), 6 numbers.
- `camera_pose_world`: for `robot0_agentview_left`, `robot0_agentview_right`, `robot0_eye_in_hand`, the 4 x 4 world pose of the camera in the OpenCV convention (x right, y down, z forward).
- `images`: per requested camera `{"rgb": {...}, "depth_mm": {...}}`, each `{"dtype", "shape", "b64"}` with raw row-major bytes (`uint8` H x W x 3, row 0 at the top, 256 x 256; `uint16` millimetres along the optical axis, clipped to 65000; a pixel that sees no surface reads 65000, since the far
plane lies beyond 65 m).
- `card` (only at `t = 0`): `task_family`; `instruction` (the sentence the robot was given); `success_rule` (the exact check for this job, as above); `object` and, where the job has a movable receptacle, `receptacle`, each `{"name", "image_png_b64" (a 128 x 128 rendered catalogue picture of that object on grey), "size_m" (bounding box, m)}`; `camera_intrinsics` (3 x 3 pixel matrix per camera, fixed); `kitchen_map` (list of `{"type", "center", "box_center", "size_xyz", "yaw_rad"}` per fixture in world coordinates: `center` is the fixture body's origin (RoboCasa's `pos`), which need not be the middle of its box; `box_center` is the centre of its bounding box; `size_xyz` is the box's full extent along the fixture's own x, y and z (width, depth, height; not half sizes); `yaw_rad` turns the fixture's x and y about the world z axis. `"role": "target"` marks the fixture the job names, where there is one, and `"support"` the counter it starts from: outlines only, no handles, knobs or object poses); `step_budget`.

There is no object pose, no fixture joint state, no success flag and no seed on the wire. Calibration and the map are
exact. Use `practice_states` and `kitchen_sim.py` to develop perception; the practice kitchens use different layouts,
styles and object instances from the graded ones.

## Grading

Each of the 27 hidden states is one episode of at most **1500 control steps**. An episode ends when the job's
success check has held for 10 consecutive steps (it counts as recovered) or at the step budget. The score is the mean
over the jobs of each job's recovered fraction (CoffeeSetupMug: 6 states, PickPlaceDrawerToCounter: 11 states, PickPlaceCounterToStove: 10 states), so every job weighs the same whatever its number of
states. There is no partial credit within an episode.

Limits, all enforced by the grader:

- Controller time is capped at **400 s per episode**: for each `act` request, the time from the grader starting to
  write the request to your reply line arriving and your processes being stopped again, summed over the episode. Reading the request is your time: an
  observation with images is larger than the pipe's buffer, so the grader writes it as you read it, and the request's
  10 s deadline covers the write as well (a request you have not read by then ends the episode like a late reply).
  Simulation and rendering do not count, and your processes cannot run outside that time (see below). An episode that
  exceeds the cap ends and scores 0. The cap was sized on the author's machine (Intel Core i9-14900KF, the sandbox's 4
  CPUs); on a slower host the same controller needs more of it.
- A reply later than 10 s, a malformed action (not 12 or 13 numbers, not finite, or a camera mask outside 0-15), a reply
  line longer than 64 KiB (65,536 bytes of UTF-8, the newline not counted), more than 1000 blank lines before the reply to one request, a crash or a closed stdout ends that episode, which scores 0; the next episode restarts `run.sh` (at most 3 restarts in the run; after that the
  remaining episodes score 0).
- The submission folder may hold at most 200 MiB and 5000 files; a larger one scores 0. A missing or unstartable
  `run.sh` (no `READY` within 40 s among its first 200 lines of output) scores 0.
- More than 512 processes of your account when the grader pauses them ends the episode (before the first episode: the
  controller counts as not started and every state scores 0). No file you write may exceed 2 GiB.
- Every process of your account must descend from the grader's start of `run.sh`. The grading sandbox starts nothing
  else as your user, so a process of your account that does not is one your controller got started some other way; it
  is killed, the episode ends, and that state and every later one score 0.
- The simulator's files are checked before every state; if they changed, that state and every later one score 0.

## The grading sandbox

Grading runs in a fresh sandbox built from the same software stack plus the graded kitchens. Only
`/home/user/submission` is copied there from your sandbox: `kitchen_sim.py`, `sim_worker.py`, `wire_rules.py`,
`practice_states/`, `helpers/` and anything else in your home are absent, so copy whatever your controller imports into
the submission folder. Your controller runs as your user with the environment `PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin`,
`HOME=/home/user`, `LANG=C.UTF-8` and nothing else (no `MUJOCO_GL`, no thread settings). The Python stack is the same
(`/opt/venv/bin/python3`, and `import robocasa` works), but RoboCasa's asset folder belongs to the grader and is not
readable, so your controller cannot build kitchens there. The sandbox has the same 4 CPUs and 12 GB as yours, shared with
the grader's simulator (about 6 GB). The grading sandbox runs no desktop session and nothing else as your user: the
only processes of your account there are the ones `run.sh` starts. Every one of them is paused (SIGSTOP)
from the moment your reply is read
until the grader starts writing the next request, which includes kitchen builds, simulation and rendering; background
threads and processes therefore only run while a request is outstanding, and that time is charged to you. A pause
holds only once every thread of every process of your account shows as stopped; one that has not stopped within 5 s
ends the episode. Before the next request is written the grader checks that the pause held: every thread it stopped
is still there and stopped and has not run since (the scheduler's record of its run time), and no new thread has
appeared. A process that continued itself during the pause (a timer that sends it SIGCONT, for example) ends the
episode. Your controller's processes run at the lowest scheduling priority (nice 19). The practice grader starts
`run.sh` with the same three variables and the same deadlines (it pauses
only `run.sh`'s process group, where the grader pauses every process of your account, and it does not check that the
pause holds, look again before the next request or apply the 5 s rule).

The whole grading run has an outer limit of 8 hours (28,800 s), which these caps keep out of reach: on the author's
machine a controller that requested every camera with depth at every step took about 156 s per state for simulation,
rendering and the wire, its own time included (4,993 s for the 32 states of an earlier form of the set, measured while
other grading runs shared the machine); adding your controller's at most 400 s and a 40 s restart per state gives about
16,100 s for the 27 states. Reaching the outer limit would be a grading failure, not your score.

The graded states are the same every time, in the same order. Nothing but your controller's replies influences the
simulation.
