# Tend the kitchen appliances

A Franka Panda arm stands fixed in a kitchen, in front of one appliance at a time: a cooktop, a microwave or a
wall oven. Someone asks it, one request at a time, to turn a burner on or off, to start or stop the microwave, or
to preheat, set or turn off the oven. The appliances are models you have not seen; you see them only through
cameras and feel them through the arm.

Deliver a controller that carries out these requests from camera images, joint readings and the wrist
force/torque sensor. You have **2 hours** of wall time for the whole job (exploring, lab trials, building and
testing the controller); have your controller in `/home/user/submission/` well before the time is up. A session
still running at the limit is cut off.

## What is in your home directory

| Path | What it is |
|---|---|
| `appliance_env.py` | The complete robot environment: the cell, the Panda arm and gripper (robosuite models), the operational-space controller, cameras, sensor noise. The lab and the grader run this exact file; only the unit differs. Read its docstrings. |
| `kitchen_render.py`, `kitchen_scene.py`, `kitchen_fx.py`, `kitchen_look.json` | The cameras' renderer, the kitchen around the cell and the burners' flames: `appliance_env.py` draws every image with them, and the lab and the grader run these exact files. They read the kitchen's room bundle in `/opt/task-assets/payload/render` (read-only). |
| `episode.py` | The exact episode logic the lab and the grader run: initial states, requests, deadlines and the credit rule. |
| `task_card.md` | Frames, mounts, the units, the cameras, the sensors and the numbers. |
| `units.json`, `units/` | Three practice units (p0 cooktop, p1 microwave, p2 oven) with their models and mechanisms (`quirks`); `env.unit_state()` and `env.truth()` give their ground truth in your dev sim. They are examples of the kinds, not the scored units. |
| `practice_grader/grade.py` | Runs your controller over the grading wire against the practice units on visible seeds and prints the raw metric, each request's outcome and whether the repeated-step check would pass. |
| `labclient.py` | Client for the lab (library and command line). |
| `helpers/lab_bridge.py` | Optional boilerplate: runs your controller (`run.sh`) for one lab trial and logs every observation and action. |
| `helpers/wire_skeleton.py` | Optional boilerplate for the grading wire (READY / reset / act / close, repeated step indices; one controller object for the whole run, with a reset hook per episode). |
| `NOTICE.md` | Where the appliance models come from (CC-BY-4.0). |

Run Python as `/opt/venv/bin/python3` (Python 3.12 with NumPy, SciPy, OpenCV, Pillow, MuJoCo 3.16). The cameras are
drawn by MuJoCo's Filament renderer (physically based) on the CPU (Mesa's lavapipe Vulkan driver; there is no GPU),
the same way in your dev sim, in the lab and in the grader. The cell stands in a furnished kitchen that is drawn
only: the cabinets, walls, shelf and furniture in the images have no collision geometry, and the robot can touch
exactly what the simulation holds (the unit, the box it stands on, the wall behind it and the floor). The first
image of a process takes about 5-7 s (the renderer starts and compiles its shaders; a later episode's first images
take well under a second), and each process that renders needs about 1 GB of memory. `mujoco.Renderer` (through
OSMesa) still works for anything else you want to draw. The sandbox has 4 CPUs and 12 GB of memory, while grading
too (the image sets `OMP_NUM_THREADS=1`, so `nproc` prints 1).

## The scored units

Nine hidden units, three of each kind: `u0`, `u3`, `u6` (cooktops), `u1`, `u4`, `u7` (microwaves) and `u2`, `u5`,
`u8` (ovens), each a different model from the practice units and from each other. Same as in your dev sim: the
robot, the controller, the action and observation formats, the cameras, the sensor noise, the physics settings, the
request texts and the credit rule. Different: the appliance model (its size, look, where its controls are and how
they move), its mechanism and the placement draw.

A unit may have one of these mechanisms (nothing reports which, if any; `units.json` lists the practice units'):
- **push to turn**: every knob of the unit turns only while the gripper presses on it along its axis with 10 N or
  more (as on gas cookers: the size of the gripper's contact force on the knob along the knob's axis, read once per
  control step); once that drops below 5 N it locks again within about a second;
- **reversed**: every knob of the unit turns the other way round (on / higher is the opposite rotation);
- **hold to start**: the microwave starts only after its start button has been held for 15 steps (1.5 s).

## Requests

One request is open at a time; `request_text` names it and `request_deadline_t` is the step by which it must be
done. The texts are exactly:

- `Turn on the <location> burner.` / `Turn off the <location> burner.` (cooktop), where `<location>` is one of
  `rear left`, `rear centre`, `rear right`, `front left`, `front centre`, `front right`, `centre`, `left`, `right`
  as the cooktop has them, seen from the front;
- compositional cooktop requests: `Turn on the <location> and <location> burners.`, `Turn off the <location> and
  <location> burners.` and `Turn off all the burners.`;
- cooktop settings: `Turn the <location> burner to <setting>.`, where `<setting>` is `low`, `medium` or `high`:
  the burner's knob in the lower, middle or upper third of its on-range (from 0.35 rad, where it lights, to the
  knob's stop); an off burner is turned on to that setting;
- `Start the microwave.` / `Stop the microwave.`;
- `Preheat the oven.` / `Turn off the oven.` (the temperature knob: preheated at 0.25 of its range or more,
  off below);
- oven settings: `Set the oven to <setting> heat.`: the temperature knob in 0.25-0.5 (`low`), 0.5-0.75
  (`medium`) or 0.75-1.0 (`high`) of its range.

Each request is valid in the state it opens on (an "off" request names burners that are on; "all the burners" is
asked while two or more are on; a setting request names a setting the control is not at). An episode has three
requests; a request closes when it is completed (see Credit) or at its deadline (1000 steps, 1500 for a
compositional request), and the next one opens at once.

A burner shows its flame while it is on: a gas burner a ring of small blue flame jets around the burner head, longer
and leaning further out at a higher setting; a glass-ceramic zone a glowing coil pattern under the glass (a double
spiral with a soft halo), dull dark red at the lowest setting and larger, brighter and more orange at a higher one.
A microwave shows a lit pale amber display bar while it runs, and an oven a lit salmon-red lamp while it is
preheated (both on the front, near the top; dark grey otherwise). Nothing in the images casts a real shadow: the
shadows you see are drawn, soft dark patches around the foot of the unit and of the furniture and a soft round
shadow straight below the hand and forearm on the unit's top or on the top the unit stands on (fading out as the
hand rises). The kitchen's light (where the work light over the unit sits and points, and the strength of all the
light) changes from episode to episode, the whole image (the flames and the lit displays included) carries a colour
tint of up to 15 % per channel that changes too, and the two fixed cameras sit a little differently in each
episode; their poses are in every observation.

## Credit

For each request, decided when it closes:
- **reached** (0.5): when the request closes (completed, or at its deadline), the requested state (of every control
  a compositional request names) has held for the last 3 consecutive steps. A state that was passed through and
  left earns nothing.
- **completed** (0.5): the requested state holds for 20 consecutive steps with the gripper (its site between the
  fingertips) at least 0.10 m from the centre of every control the request names, while every other control of the
  unit keeps the state it had when the request opened; the request closes at that step.
- **Every other control** means exactly: on a cooktop, each burner the request does not name stays on or off as it
  was; on an oven, its door stays closed (at most 0.005 of its opening range) or open (0.90 or more) as it was; on a
  microwave nothing else is compared (its door is not; opening it stops the microwave). Not compared: a burner's
  setting (only on or off), the oven's timer knob. They are compared during those 20 steps only: a control disturbed
  on the way (the wrong burner turned on, one knocked off) must be back in its state before the 20 steps count.

The episode score is the mean over its three requests; your result is the mean over the graded episodes.

## Observation and action

Observation (a JSON object; the lab replies with the same keys): `t` (step index), `unit` (`"u0"`...), `q`, `dq`
(7 arm joints), `ee_pos` (gripper site, world frame, m), `ee_quat` (w, x, y, z), `gripper` (finger opening, m),
`wrench` (force N and torque N m at the wrist sensor, world frame, noisy; `q`, `ee_pos`, `ee_quat` and
`T_world_wrist` carry 6 decimals, `wrench` 4), `T_world_wrist`, `T_world_left`,
`T_world_right` (4 x 4 camera poses, OpenCV convention), `K_left`, `K_right`, `K_wrist` (intrinsics), the RGB
images `left`, `right` (320 x 240) and `wrist` (256 x 192), each `{"dtype": "uint8", "shape": [H, W, 3], "b64":
<raw row-major bytes, base64>}`, and `request_id` (-1 when none is open), `request_text`, `request_deadline_t`. An
image is empty (shape `[0, 0, 3]`) unless the previous action asked for it; the first observation of an episode
carries all three. There is no depth image and no appliance state of any kind.

Action: a list of 8 numbers `[dx, dy, dz, drx, dry, drz, grip, cams]` at 10 Hz: a gripper-site displacement
(world frame, clipped to 0.03 m), a world-frame rotation vector (clipped to 0.2 rad), the gripper command (-1
open ... +1 closed) and a camera bit mask for the next observation (1 = left, 2 = right, 4 = wrist), which must be
an integer from 0 to 7 (anything else is a malformed action).

## The lab (your only access to the nine scored units)

`labclient.py` talks to a service on `/run/ale-lab/lab.sock` (JSON lines; `python3 labclient.py --help`). This lab
serves `hello`, `budget`, `reset`, `step` and `close` (the client's `execute`, an open-loop plan, is not served here:
the lab answers `unsupported`). A **trial** is one episode on one unit: `reset` with the unit id starts it (and
charges 1 trial; you have **10 trials in total**, shared by all nine units, never refunded), then `step` with an
action returns the next observation (steps are free) until the episode ends. A trial ends when its episode ends (when
its third request closes, or when the simulation becomes unstable), when you start a new trial or close the
connection, after 3600 s of wall time or after 900 s without a request; a step cap of 4800 requests applies. A
`reset` on a connection that has a trial ends that trial and charges another. The lab serves at most two connections
at a time, and a call it cannot answer within 120 s restarts its worker (a trial already charged is not refunded).
Each trial draws a fresh placement, initial state, request queue, lighting and sensor noise. A reset takes about 1 s
and a step with all three images about 0.2 s (much less without images). The lab never reports scores, credit or the
unit's state: only observations; but a request that closes before its deadline was completed. The lab units are
exactly the units you are scored on. The lab is gone when grading starts.

## Deliverable and grading

Leave a controller in `/home/user/submission/` with an executable `run.sh` (started as `bash run.sh` with that
folder as working directory, as your user, with no network: only AF_UNIX sockets work). Everything it needs must
be inside that folder (at most 5000 files and 200 MiB; a larger folder scores 0), and the controller may write
there; nothing else of your home is available when it is graded. Protocol (one JSON object per line on
stdin/stdout; stdout carries only protocol messages, use stderr for logs): print `READY` within 120 s of its start
(a controller that does not, at its first start, scores 0 for the whole run); then answer `{"type": "reset"}` with
`{"ok": true}`, `{"type": "act", "t": T, "obs": {...}}` with `{"action": [8 numbers]}` within 30 s, and
`{"type": "close"}` with `{"ok": true}` before exiting. At about 1 % of the steps the grader asks the same `t`
twice, once with the real observation and once with a slightly perturbed copy, in a hidden order, and applies only
the reply to the real one. The copy differs in `q` (each joint by up to 0.001 rad) and in what follows from it
(`ee_pos`, `ee_quat`, `T_world_wrist`, recomputed), and in the wrench's force (up to 0.2 N per axis); `dq`,
`gripper`, the images and everything else are the same. A pair in which both replies command no motion (the first
six numbers all zero, within 1e-9) is not counted either way; if fewer than 10 % of the other pairs get two replies
that differ (by more than 1e-4 in some number), the whole run scores 0: a controller that ignores its input, or
answers a repeated `t` from a cache, fails.

The grader runs 18 episodes: 2 on each unit, a fixed hidden set (each episode's seed is the same for every
submission, so are its placement, initial state, lighting and camera offsets), in a hidden order (the observation
names the unit). The lab, in contrast, draws a fresh episode at every trial. An episode in which the controller
crashes, answers late or malformed, or makes the simulation unstable (an arm joint faster than 8 rad/s, a burner knob
or a door faster than 30 rad/s, or a MuJoCo warning of a bad state) scores 0. After a crash, a late answer or a reply
that breaks the protocol, the controller is started again before the next episode, at most 3 times: the episode that
would need a fourth restart and every later one score 0. The whole grading run must finish within 6000 s; the
episode under way when it runs out, and the episodes after it, score 0. The 6000 s cover the simulator and the
rendering as well as your controller: measured over the grading wire on 4 CPUs of a busy machine (medians), a step
costs the grader about 10 ms without an image, and each image asked for adds about 30 ms (wrist) or 55 ms (a fixed
camera), so a step with all three images costs about 0.15 s; a run in which every request runs to its deadline has
between 54,000 and 63,000 steps, so all three images at every step of such a run do not fit. Before the first
episode the grader draws one test image of each camera, which takes a few seconds of the 6000 s. Your controller is
one process for the whole grading run (it is started again only after a crash, a late answer or a reply that breaks
the protocol, at most 3 times), so it may carry what it learned about a unit in one episode into later episodes on
the same unit. Only the final controller is scored; trial usage is not.
