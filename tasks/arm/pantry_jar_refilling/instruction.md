# Refill the pantry jars

You are a robotics engineer. Two Franka Pandas stand side by side at a kitchen counter with three screw-top jars and
a tray of refill pieces. They see through cameras and feel through wrist force sensors. Write the controller that
refills the jars on a refill list. It runs closed loop against a simulator and is graded on jars and counters you have
not seen. You have 2 hours of wall time from the start of your session; check the clock (`date`) as you go. Whatever
is in `/home/user/submission/` when the time ends is graded, so keep a working controller there.

**The counter.** World frame: z up, metres; the counter top is at z = 0.80. The left arm's base is at (0, 0.28), the
right arm's at (0, -0.28), both facing +x.
- **Three jars** stand in a row at x = 0.55 (each within 2 cm), one at y = 0.17, one at 0 and one at -0.17 (each
  within 1 cm). Which jar stands where changes every episode.
- **Labels.** Each jar has a coloured label band, and the colour is its kind: white for `sugar`, green for `mint`,
  purple for `berry`. Each unit has one jar of each kind.
- **Lids.** All lids are red and look alike, but their diameters differ by at least 4 mm. A lid catches only on its own
  jar's thread.
- **The tray** is at (0.32, -0.42), with a 6 mm rim. It holds the pieces:
  - `sugar`: 16 mm cubes;
  - `mint` and `berry`: capsules 24 mm long and 14 mm across, in their kinds' colours.
- **What the tray holds.** For each listed kind, the listed count plus one or two spares; one or two pieces of the
  kind not on the list.

`task_card.md` gives every number in one place.

**The job.** Each episode's refill list, `refill` in the observation (for example `{"mint": 3, "sugar": 2}`), names
two of the three jars by kind and how many pieces of that kind each needs (2-4). For each listed jar:
1. open it;
2. put exactly that many pieces of its kind inside;
3. put its own lid back and screw it down until it seats, closing it with a torque between 0.5 and 2.5 N m.

Leave the third jar closed. Keep every jar standing and every piece on the counter.

**The jars and their threads.** A jar stands free on the counter: turn its lid without holding the jar and the jar
turns with it. So one hand holds the jar while the other turns the lid.
- **Opening.** Turning a lid counter-clockwise (seen from above) raises it 5 mm per turn. After the jar's number of
  turns it comes free.
- **Catching again.** A free lid catches again when it rests centred (within 3 mm) and level on its own jar's neck and
  is turned clockwise near the thread's start. A lid that has just come free must first leave the neck (more than
  3 mm off its axis or 4 mm above it) before it can catch.
- **Closing.** Turned clockwise, a caught lid screws down until it seats. The closing torque is the torque the seat
  takes; the largest value since the lid last caught is the one that counts.

**What is hidden per unit.** Each unit's three jars differ in:

| Property | Range |
|---|---|
| Turns to free the lid | 1, 1.5, 2 or 2.5 |
| Running torque of the thread | 0.2-0.8 N m |
| Jar mass | 0.25-0.90 kg |
| Counter friction (jar on counter) | 0.4-0.8 |
| Jar height | 10-14 cm |
| Lid radius | 27, 29, 31, 33 or 35 mm (the jar's body 1-3 mm narrower) |

The two practice units in `units.json` are drawn from the same ranges, with their values given. The scored units'
values are not given: your controller has to find out, for example from the wrist forces, how its jars behave.

## What is in your home directory

| Path | What it is |
|---|---|
| `jar_sim.py` | The simulator of the counter: the robots and their controllers, the jars and threads, the tray and pieces, the cameras, each unit's and each episode's draw. The grader runs this exact file. |
| `jar_env.py` | The environment your controller talks to: `JarEnv(unit).reset(seed)`, `step(action)`, the observation. The grader runs this exact file. |
| `episode.py` | An episode, its end and its credit. The grader runs this exact file. |
| `units.json` | The two practice units (p0, p1), each three jars' properties. |
| `task_card.md` | The counter's numbers in one place. |
| `practice_grader/grade.py` | Runs your controller over the wire, as the grader does, on the practice units; prints the raw credit. |
| `helpers/wire_skeleton.py` | Optional boilerplate for the wire (no control logic): copy it into your submission. |

Python is `/opt/venv/bin/python3` (MuJoCo 3.16, numpy, scipy, OpenCV, Pillow; robosuite 1.5.2's robot model files).
There is no network.

## Credit

Each episode's credit (0 to 1) is computed from the simulation when the episode ends. It is the mean over the two
listed jars of two halves:

- **Refill, 0.5:** the pieces of the jar's kind inside it over the count listed. A piece is inside when its centre is
  within the jar's wall and between its floor and its shoulder. This half is 0 if the jar holds any piece of another
  kind, or more pieces than listed.
- **Sealed, 0.5:** the jar was opened during the episode, and its own lid is back in its thread and seated (within
  0.02 rad of the seat). The largest closing torque since the lid last caught must be between 0.5 and 2.5 N m.

The episode's credit is halved if any of these happened:
- a jar not on the list was opened;
- a jar is tilted more than 45 deg at the end;
- a piece is off the counter (below its top, or outside x -0.15 to 0.85 or y -0.80 to 0.80).

A controller that does nothing scores 0.

An episode ends at the first of:
- 12000 control steps (20 simulated minutes);
- 20 consecutive steps in which:
  - every listed jar has been opened and is seated again with its own lid;
  - both grippers are open at least 70 mm;
  - both hands (the gripper sites) are at least 12 cm from every jar's top;
- an unstable simulation (an arm joint faster than 8 rad/s, or a jar, lid or piece faster than 5 m/s), which scores 0.

The score is the mean credit over the graded episodes.

## Observation and action

The controller runs at 10 Hz. The observation is a JSON object:

- **Step and list:** `t` (the step), `unit` (the unit's id), `refill` (`{kind: pieces}`).
- **Each arm**, under `l` and `r`:
  - `q` and `dq`: 7 joint angles and velocities;
  - `ee_pos` and `ee_quat`: the gripper site's pose, quaternion wxyz;
  - `gripper`: the fingers' opening, m;
  - `wrench`: the wrist's force and torque in the world frame, with noise of 0.3 N and 0.03 N m.
- **Cameras.** For each camera:
  - `T_world_<cam>`: its 4 x 4 pose, OpenCV convention (x right, y down, z forward);
  - `K_<cam>`: its intrinsics;
  - `<cam>`: the image, `{"dtype": "uint8", "shape": [H, W, 3], "b64": ...}`.

  The cameras are `overhead`, `front` and `tray` (640 x 480, fixed) and `wrist_l` and `wrist_r` (320 x 240, on the
  hands). An image is empty (`[0, 0, 3]`) unless the previous action asked for it; the first observation has all five.
  There is no depth image.

The action is a JSON object `{"l": [...], "r": [...], "cams": mask}`. Each arm's list has one of three shapes:

- **7 numbers** `[dx, dy, dz, drx, dry, drz, grip]`. The operational-space controller moves the hand's target from the
  hand's current pose by (dx, dy, dz) m (clipped to 0.03 m) and turns it by the rotation vector (drx, dry, drz) rad
  (clipped to 0.2 rad), both in the world frame. `grip` runs from -1 (fully open, 80 mm) to +1 (closed).
- **14 numbers:** the same 7, then seven joint angles. These are the posture the controller's null space pulls the arm
  toward from now on (the start posture until you set one).
- **8 numbers** `[q1..q7, grip]`: a joint step. A gravity-compensated joint PD moves toward q, each joint's target at
  most 0.1 rad from its current angle. q also becomes the null-space posture.

`cams` is a bitmask of the images the next observation carries: 1 overhead, 2 front, 4 tray, 8 left wrist, 16 right
wrist. Images are large (a 640 x 480 image is about 1.2 MB on the wire) and cost the grader time to draw, so ask for
one only when you need it.

## Deliverable and grading

Put your controller in `/home/user/submission/` with a `run.sh`.

**Only the submission exists at grading.** The grader runs in a separate sandbox where only `/home/user/submission/`
is restored. Nothing else from your home directory is there: not the simulator files, `units.json`, the practice
grader or `helpers/`. Everything your controller needs must be inside the submission folder: copy in any of the
files above that it uses, and import them from there. The folder may hold at most 5000 files and 200 MB, and it must
be a real folder, not a link. The practice grader runs in your home, where every file exists, so it cannot show this
failure; run it on a copy of your submission in another folder to check.

**The wire.** The grader starts your controller with `bash run.sh` in that folder, as your user, with only local
(AF_UNIX) sockets. It speaks one JSON object per line on the controller's stdin and stdout:

- the controller prints `READY` within 120 s of starting; a controller that never prints it scores 0 on every
  episode;
- `{"type": "reset"}` → `{"ok": true}`: an episode starts;
- `{"type": "act", "t": t, "obs": {...}}` → `{"action": {...}}`, within 30 s, on one line of at most 4 MiB;
- `{"type": "close"}` → `{"ok": true}`, then exit.

Stdout carries only these messages; write logs to stderr. `run.sh` starts with only `PATH`, `HOME` and `LANG` set:
export anything else you need there (to render with MuJoCo, `MUJOCO_GL=osmesa`).

**Probes.** About 1 % of steps are asked twice with the same `t`, in a hidden order:
- once with the real observation;
- once with a slightly perturbed copy: each arm's joints moved by up to 1 mrad, with the hand poses and the wrist
  cameras' poses recomputed, and each force by up to 0.2 N; images unchanged.

Only the reply to the real observation is applied. A pair is not counted either way when both replies command no
motion: both arms send 7 numbers whose first six are all within 1e-9 of zero. If fewer than 10 % of the counted pairs
get replies that differ (by more than 1e-4 in some number), the whole run scores 0.

A controller fails this check when its replies do not depend on the arms' joints, hand poses or wrist forces. That
includes:
- a replayed action stream;
- a controller that answers a repeated `t` from a cache;
- one that holds a fixed absolute joint target over most of its moving steps;
- one that moves the hands by steps computed only from images.

Cartesian steps toward a target computed from the observed hand pose pass. Which steps are probed depends only on the
episode, not on your replies.

**Grading.**
- **The episodes.** There are 12: 2 for each of 6 hidden units. They are a fixed hidden set: each episode's seed is
  the same for every submission, and so are its jar places, refill list, tray and noise. They run in a hidden order;
  the observation names the unit.
- **The render test.** Before the first episode the grader draws one test image of each camera.
- **Failures.** A crash, a late or oversize reply, a malformed action, a simulator fatal error or an unstable
  simulation scores that episode 0, and grading goes on. After a failure the next episode restarts the controller;
  after 3 restarts the remaining episodes score 0.
- **The time budget.** The grader has 9000 s for all 12 episodes. That covers the simulation, drawing the images you
  ask for, and your controller's own time. On the reference host (a shared 96-core x86-64 Linux server) the
  reference controller, which asks for no images, used 3532-3835 s for all 12 episodes (4071-6552 steps each). If the
  budget runs out, the episode running is aborted and the rest are not run; all of them score 0.
- **One process.** Your controller runs as one process for the whole run, so it may carry what it learns from one
  episode to the next.
