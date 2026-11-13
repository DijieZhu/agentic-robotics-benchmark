# Tend the kitchen appliances

A Franka Panda arm stands fixed in a kitchen, in front of one appliance at a time: a cooktop, a microwave or a
wall oven. Someone asks it, one request at a time, to turn a burner on or off, to start or stop the microwave, or
to preheat or turn off the oven. The appliances are models you have not seen; you see them only through
cameras and feel them through the arm.

Deliver a controller that carries out these requests from camera images, joint readings and the wrist
force/torque sensor. You have **2 hours** of wall time for the whole job (exploring, lab trials, building and
testing the controller); have your controller in `/home/user/submission/` well before the time is up. A session
still running at the limit is cut off.

## What is in your home directory

| Path | What it is |
|---|---|
| `appliance_env.py` | The complete robot environment: the cell, the Panda arm and gripper (robosuite models), the operational-space controller, cameras, sensor noise. The lab and the grader run this exact file; only the unit differs. Read its docstrings. |
| `episode.py` | The exact episode logic the lab and the grader run: initial states, requests, deadlines and the credit rule. |
| `task_card.md` | Frames, mounts, the units, the cameras, the sensors and the numbers. |
| `units.json`, `units/` | Three practice units (p0 cooktop, p1 microwave, p2 oven) with their models; `env.unit_state()` and `env.truth()` give their ground truth in your dev sim. They are examples of the kinds, not the scored units. |
| `practice_grader/grade.py` | Runs your controller over the grading wire against the practice units on visible seeds and prints the raw metric. |
| `labclient.py` | Client for the lab (library and command line). |
| `helpers/lab_bridge.py` | Optional boilerplate: runs your controller (`run.sh`) for one lab trial and logs every observation and action. |
| `helpers/wire_skeleton.py` | Optional boilerplate for the grading wire (READY / reset / act / close, repeated step indices). |
| `NOTICE.md` | Where the appliance models come from (CC-BY-4.0). |

Run Python as `/opt/venv/bin/python3` (Python 3.12 with NumPy, SciPy, OpenCV, Pillow, MuJoCo 3.16). Rendering is
MuJoCo through OSMesa on the CPU.

## The scored units

Six hidden units, two of each kind: `u0` and `u3` (cooktops), `u1` and `u4` (microwaves), `u2` and `u5` (ovens),
each a different model from the practice units and from each other. Same as in your dev sim: the robot, the
controller, the action and observation formats, the cameras, the sensor noise, the physics settings, the request texts and the credit rule. Different:
the appliance model (its size, look, where its controls are and how they move) and the placement draw.

## Requests

One request is open at a time; `request_text` names it and `request_deadline_t` is the step by which it must be
done. The texts are exactly:

- `Turn on the <location> burner.` / `Turn off the <location> burner.` (cooktop), where `<location>` is one of
  `rear left`, `rear centre`, `rear right`, `front left`, `front centre`, `front right`, `centre`, `left`, `right`
  as the cooktop has them, seen from the front;
- `Start the microwave.` / `Stop the microwave.`;
- `Preheat the oven.` / `Turn off the oven.` (the temperature knob: preheated at 0.25 of its range or more,
  off below).

Each request is valid in the state it opens on (an "off" request names a burner that is on). An episode has three
requests; a request closes when it is completed or at its deadline (600 steps), and the next one opens at once.

## Credit

For each request:
- **reached** (0.5): the requested state holds for 3 consecutive steps;
- **completed** (0.5): the requested state holds for 20 consecutive steps with the gripper at least 0.10 m from
  the control, while every other control of the unit keeps the state it had when the request opened (turning on
  the wrong burner as well, or knocking one off on the way, costs the completion).

A counted stage stays counted. The episode score is the mean over its three requests; your result is the mean
over the graded episodes.

## Observation and action

Observation (a JSON object; the lab replies with the same keys): `t` (step index), `unit` (`"u0"`...), `q`, `dq`
(7 arm joints), `ee_pos` (gripper site, world frame, m), `ee_quat` (w, x, y, z), `gripper` (finger opening, m),
`wrench` (force N and torque N m at the wrist sensor, world frame, noisy), `T_world_wrist`, `T_world_left`,
`T_world_right` (4 x 4 camera poses, OpenCV convention), `K_left`, `K_right`, `K_wrist` (intrinsics), the RGB
images `left`, `right` (320 x 240) and `wrist` (256 x 192), each `{"dtype": "uint8", "shape": [H, W, 3], "b64":
<raw row-major bytes, base64>}`, and `request_id` (-1 when none is open), `request_text`, `request_deadline_t`. An
image is empty (shape `[0, 0, 3]`) unless the previous action asked for it; the first observation of an episode
carries all three. There is no depth image and no appliance state of any kind.

Action: a list of 8 numbers `[dx, dy, dz, drx, dry, drz, grip, cams]` at 10 Hz: a gripper-site displacement
(world frame, clipped to 0.03 m), a world-frame rotation vector (clipped to 0.2 rad), the gripper command (-1
open ... +1 closed) and a camera bit mask for the next observation (1 = left, 2 = right, 4 = wrist).

## The lab (your only access to the six scored units)

`labclient.py` talks to a service on `/run/ale-lab/lab.sock` (JSON lines; `python3 labclient.py --help`). A
**trial** is one episode on one unit: `reset` with the unit id starts it (and charges 1 trial; you have **8
trials in total**, shared by all six units, never refunded), then `step` with an action returns the next observation
(steps are free) until the episode ends. A trial ends when its episode ends, when you start a new trial or close
the connection, after 3600 s of wall time or after 900 s without a request; a step cap of 2000 requests applies.
Each trial draws a fresh placement, initial state, request queue and sensor noise. The lab never reports scores,
credit or the unit's state: only observations. The lab units are exactly the units you are scored on. The lab is
gone when grading starts.

## Deliverable and grading

Leave a controller in `/home/user/submission/` with an executable `run.sh` (started as `bash run.sh` with that
folder as working directory, as your user, with no network: only AF_UNIX sockets work). Everything it needs must
be inside that folder (at most 5000 files and 200 MiB; a larger folder scores 0), and the controller may write
there; nothing else of your home is available when it is graded. Protocol (one JSON object per line on
stdin/stdout; stdout carries only protocol messages, use stderr for logs): print `READY` within 120 s; then answer
`{"type": "reset"}` with `{"ok": true}`, `{"type": "act", "t": T, "obs": {...}}` with `{"action": [8 numbers]}`
within 30 s, and `{"type": "close"}` with `{"ok": true}` before exiting. At about 1 % of the steps the grader asks
the same `t` twice, once with the real observation and once with a slightly perturbed copy (joint readings,
gripper and wrist-camera pose, wrench; the images are unchanged), in a hidden order, and applies only the reply to
the real one. If fewer than 10 % of these pairs get two replies that differ (by more than 1e-4 in some number),
the whole run scores 0: a controller that ignores its input, or answers a repeated `t` from a cache, fails.

The grader runs 12 episodes: 2 on each unit with fresh hidden placements and requests, in a hidden order (the
observation names the unit). An episode in which the controller crashes, answers late or malformed, or makes the
simulation unstable (an arm joint faster than 8 rad/s or a unit joint faster than 30 rad/s) scores 0; after 3
restarts of your controller the remaining episodes score 0. The whole grading run must finish within 6000 s; the
episode under way when it runs out, and the episodes after it, score 0. Your controller is one process for the
whole grading run (it is started again only after a crash), so it may carry what it learned about a unit in one
episode into later episodes on the same unit. Only the final controller is scored; trial usage is not.
