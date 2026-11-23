# Replace the cabinet door

You are a robotics engineer. Two Franka Pandas stand side by side at a workbench in front of a wall cabinet whose
old door hangs on the wrong pins. The new door stands in a rack on the bench; loose hinge pins stand in six stands.
The robots see through cameras and feel through wrist force sensors. Write the controller that replaces the door:
take the old door off and set it aside, then hang the new one: one arm holds it in the opening with its hinge
knuckles lined up, the other drives a pin down through each hinge. It runs closed loop against
a simulator and is graded on cabinets you have not seen. You have 2 hours of wall time (see "Session time").
Whatever is in `/home/user/submission/` when the time ends is graded, so keep a working controller there.

**The cell.** World frame: z up, metres; the bench top is at z = 0.80 and ends at x = 0.55. The left arm's base is at
(0, 0.28), the right arm's at (0, -0.28), both facing +x. `task_card.md` gives every number in one place.
- **The left arm** has the Panda parallel gripper.
- **The right arm** has a vacuum cup on an L-shaped bracket: with the hand pointing down, the cup (radius 30 mm)
  faces forward. With vacuum on, the cup seals on the door when its face touches the door's front face square (within
  3 mm and 10 deg, the whole cup on the leaf), and then holds the door. The seal breaks, and the door is let go, when
  vacuum is switched off or when the held load stays above 150 N or 15 N m (the moment about the cup's face centre)
  for 20 ms.
- **The cabinet** hangs on the wall beyond the bench; nothing is below its door. Its one opening has two hinges on
  its left (+y) side, the bottom one 0.05-0.07 m and the top one 0.38-0.40 m above the hung door's bottom edge. Each
  episode moves the cabinet by up to 8 mm in x, y and z and turns it by up to 2 deg about z.
- **A hinge** is a 75 mm stack of 3 or 5 knuckles on one vertical axis, alternately carcass and door, with carcass
  knuckles at both ends. Each door knuckle sits in its slot with 1 mm of play at each end. A pin goes down through
  every knuckle; its head seats on the top knuckle. A door hangs only on its pins: without them it falls out.
- **The old door** (dark brown, its knuckles dark) hangs in the opening on two old pins, one in each hinge. They are
  4 mm across, thinner than every pin of the series: they slide freely in any bore (no detent) and rest on their
  heads. They are the wrong pins and do not count.
- **The new door** (pale beige, brass knuckles; 0.40 x 0.45 m, 18 mm thick, 1.5-3.5 kg) stands in a rack on the
  bench with its front face toward the robots and its knuckles on its left (+y) edge, nominally at (0.38, 0.10); each
  episode moves it by up to 8 mm. Both doors are the same model, with the same knuckles, mass and handle.
- **The parking rack** is empty: a channel on the bench, nominally at (0.50, -0.36) (each episode moves it by up to
  8 mm), with 8 mm free on each side of a door. It is there for the old door; any other place where the old door
  stays up is as good.
- **The pins** stand in six stands at x = 0.45, y = 0.30, 0.38, 0.46, 0.54, 0.62 and 0.70. Each pin has a 71 mm
  shaft below its head and an arm to a square grip post 50 mm in front of its axis. Pins come in three diameters:
  S 5 mm, M 6.5 mm, L 8 mm. A cabinet's hinges take one of them, and its bore is the pin's radius + the radial
  clearance. Two stands hold pins of each diameter; which stand holds which changes every episode, the same way for
  every cabinet, so the stands do not tell you the cabinet's diameter.
- **The knuckles all look alike.** Every knuckle of the series is 18 mm across whatever its bore.
- **The detent.** Inside a hinge's bore a pin of the hinge's own diameter slides only under a push of about 6 N: it
  never drops through under its own weight. When its head comes within 0.7 mm of its seat it clicks in; it then
  stays (it still turns about its axis) unless it is pulled up harder than 40 N. A pin of a smaller diameter slides
  through the bore freely (no detent, no click) and rests on its head; it does not count. A pin of a larger diameter
  does not go into the bore.

**The job.** Replace the door, and pin both hinges of the new one with pins of the right diameter:
1. hold the old door, pull its two old pins out and put them aside (anywhere on the bench);
2. slide the old door's knuckles out from between the carcass knuckles and set the old door down off the cabinet;
   **if the old door falls below the bench top, the episode ends and scores 0**;
3. take the new door from its rack and bring it to the opening;
4. slide its knuckles in between the carcass knuckles and hold it there, the door's bores lined up with the carcass's;
5. drive a pin of the right diameter down through each hinge until it clicks into its seat;
6. let go of the door and the pin, and move both tools away from the door.

**What is hidden per unit.** Each unit is one cabinet of the series:

| Property | Range |
|---|---|
| Knuckles per hinge | 3 or 5 |
| Pin diameter | S 5 mm, M 6.5 mm or L 8 mm |
| Radial clearance (bore - pin radius) | 0.2-0.5 mm |
| Bottom hinge centre above the door's bottom edge | 0.05, 0.06 or 0.07 m |
| Top hinge centre | 0.38, 0.39 or 0.40 m |
| Door mass | 1.5-3.5 kg |
| Handle mass | 0.1-0.4 kg |

The two practice units in `units.json` are drawn from the same ranges, with their values given. The scored units'
values are not given: your controller has to find out, from the cameras and the wrist forces, where the hinges are,
which pins fit and how the door hangs.

## What is in your home directory

| Path | What it is |
|---|---|
| `door_sim.py` | The simulator of the cell: the robots and their controllers, the vacuum cup, the cabinet and its hinges, the doors, the pins and the detent, the cameras, each unit's and each episode's draw. The grader runs this exact file. |
| `door_env.py` | The environment your controller talks to: `DoorEnv(unit).reset(seed)`, `step(action)`, the observation. The grader runs this exact file. |
| `episode.py` | An episode, its end and its credit. The grader runs this exact file. |
| `units.json` | The two practice units (p0, p1). |
| `task_card.md` | The cell's numbers in one place. |
| `practice_grader/grade.py` | Runs your controller over the wire, as the grader does, on the practice units; prints the raw credit and what it is made of (see below). |
| `session_time.sh` | Prints the minutes of your session used and left (see "Session time"). |
| `helpers/wire_skeleton.py` | Optional boilerplate for the wire (no control logic): copy it into your submission. |

Python is `/opt/venv/bin/python3` (MuJoCo 3.16, numpy, scipy, OpenCV, Pillow; robosuite 1.5.2's robot model files).
There is no network. Your sandbox has 4 CPUs and 12 GB of memory; the grading sandbox has the same.

**Practice.** You may practise as much as the session allows: the simulator is yours to run, and the practice
units' truth is yours to read (`units.json` gives their hinges, pin class and door mass; the simulator's state gives
everything else). For each practice episode the practice grader prints the credit, whether the new door hangs, which
hinges are pinned, whether the door swings and how far it opened, which pins are home, the pin class the unit needs,
whether the cup still holds a door, whether the old door fell, and the controller's time as the grader counts it
(`controller_seconds`). None of this exists for the scored units: they are other cabinets, every hidden property of
which (knuckle count, pin class, clearance, hinge heights, door and handle mass) is drawn anew within the ranges
above, and nothing but the observation is given for them.

**Numerics.** Every `/opt/venv` Python computes the same here and in the graded run, on whatever machine: the image
sets, when the variables are missing, NumPy's linear algebra on one fixed set of kernels (`OPENBLAS_CORETYPE=Haswell`
with `OPENBLAS_L2_SIZE=256`) and on one thread (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`), NumPy without its
AVX-512 loops (`NPY_DISABLE_CPU_FEATURES`) and OpenCV's IPP at AVX2 (`OPENCV_IPP=avx2`). Do not unset or change them:
the simulator would compute differently, and your controller's results would depend on the grading machine's CPU.

## Credit

Each episode's credit (0 to 1) is computed from the simulation when the episode ends:

Only the new door counts.

- **Pinned, 0.25 per hinge:** the new door hangs on the opening and a pin of the cabinet's diameter is home in that
  hinge.
  - The door hangs: its knuckles' axis within 2 mm of the hinge axis, its bottom edge within 2 mm of the hung height,
    and its leaf within 2 deg of upright and 3 deg of closed.
  - A pin is home: on the axis within 1.5 mm, tilted at most 3 deg, its head at most 3 mm above the top knuckle.
  - A pin of a smaller diameter (an old pin included) goes in but does not count.
- **Swings, 0.5:** both hinges pinned, and then the swing test passes. The test parks both arms out of the way and,
  with a test hand that applies at most 0.3 N m about the hinge axis, turns the door toward 70 deg for 2 s, back to
  closed for 4 s, and lets it settle. It passes when the door opened at least 60 deg and afterwards still hangs closed
  with both hinges pinned.

A controller that does nothing scores 0.

An episode ends at the first of:
- 6000 control steps (10 simulated minutes);
- the old door's centre of mass below the bench top less 5 cm (it fell off), which scores 0;
- 20 consecutive steps in which:
  - the new door hangs and both hinges are pinned;
  - the cup has let go (vacuum off or no seal);
  - the gripper is open at least 70 mm;
  - both tools are at least 8 cm in front of the door's front face;
- an unstable simulation (an arm joint faster than 8 rad/s, or a door or a pin faster than 5 m/s), which scores 0.

The score is the mean credit over the graded episodes.

## Session time

- This session is stopped 2 hours of wall-clock time after it started. What is in `/home/user/submission/` at
  that moment is graded as it is: keep a complete, runnable submission there from early on and replace it as you
  improve it.
- The only clock for this limit is the wall clock. `bash /home/user/session_time.sh` prints the minutes used and
  the minutes left, from a time stamp written when your sandbox was set up (moments before your session started,
  so you have slightly more than it says, never less); `date` shows the same clock. Run the script when you begin,
  again before every decision that depends on the time left, and whenever you pick the work up from notes or from
  a summary of what was done. Do not estimate the time used from how much you have done.
- Two other counts in this task are not your session's time: an episode's length (6000 control steps are 10
  simulated minutes and a few minutes of wall time in your simulator) and the controller's 9000 s budget at
  grading. Neither says how much of the 2 hours is left.

## Observation and action

The controller runs at 10 Hz. The observation is a JSON object:

- **Step and unit:** `t` (the step) and `unit` (the unit's id).
- **Each arm**, under `l` and `r`:
  - `q` and `dq`: 7 joint angles and velocities;
  - `ee_pos` and `ee_quat`: the tool frame's pose, quaternion wxyz (left: the gripper's grip site; right: the cup
    face's centre, its z axis along the way the cup faces);
  - `wrench`: the wrist's force and torque in the world frame, with noise of 0.3 N and 0.03 N m;
  - left only, `gripper`: the fingers' opening, m; right only, `vacuum`: true while the cup holds a door.
- **Cameras.** For each camera:
  - `T_world_<cam>`: its 4 x 4 pose, OpenCV convention (x right, y down, z forward);
  - `K_<cam>`: its intrinsics;
  - `<cam>`: the image, `{"dtype": "uint8", "shape": [H, W, 3], "b64": ...}`.

  The cameras are `overhead`, `front` and `side` (640 x 480, fixed), `wrist_l` (320 x 240, on the left hand) and
  `wrist_r` (320 x 240, on the cup bracket, looking the way the cup faces). An image is empty (`[0, 0, 3]`) unless the
  previous action asked for it; the first observation has all five. There is no depth image.

The action is a JSON object `{"l": [...], "r": [...], "cams": mask}`. Each arm's list has one of three shapes:

- **7 numbers** `[dx, dy, dz, drx, dry, drz, grip]`. The operational-space controller moves the tool frame's target
  from its current pose by (dx, dy, dz) m (clipped to 0.03 m) and turns it by the rotation vector (drx, dry, drz) rad
  (clipped to 0.2 rad), both in the world frame. Left: `grip` runs from -1 (fully open, 80 mm) to +1 (closed). Right:
  `grip` > 0 asks for vacuum, `grip` <= 0 releases it.
- **14 numbers:** the same 7, then seven joint angles. These are the posture the controller's null space pulls the arm
  toward from now on (the start posture until you set one).
- **8 numbers** `[q1..q7, grip]`: a joint step. A gravity-compensated joint PD moves toward q, each joint's target at
  most 0.1 rad from its current angle. q also becomes the null-space posture.

The operational-space controller compensates the arm's own weight, not what the tool holds: a held door pulls the
hand down until you command against it.

`cams` is a bitmask of the images the next observation carries: 1 overhead, 2 front, 4 side, 8 left wrist, 16 right
wrist. Images are large (a 640 x 480 image is about 1.2 MB on the wire) and each one costs 0.1 s of your
controller's time budget at grading, so ask for one only when you need it.

## Deliverable and grading

Put your controller in `/home/user/submission/` with a `run.sh`.

**Only the submission exists at grading.** The grader runs in a separate sandbox where only `/home/user/submission/`
is restored. Nothing else from your home directory is there: not the simulator files, `units.json`, the practice
grader or `helpers/`. Everything your controller needs must be inside the submission folder: copy in any of the
files above that it uses, and import them from there. The folder may hold at most 5000 files and 200 MiB, and it must
be a real folder, not a link. The practice grader runs in your home, where every file exists, so it cannot show this
failure; run it on a copy of your submission in another folder to check.

**The wire.** The grader starts your controller with `bash run.sh` in that folder, as your user, with only local
(AF_UNIX) sockets. It speaks one JSON object per line on the controller's stdin and stdout:

- the controller prints `READY` within 120 s of starting; a controller that never prints it scores 0 on every
  episode;
- `{"type": "reset"}` → `{"ok": true}`: an episode starts;
- `{"type": "act", "t": t, "obs": {...}}` → `{"action": {...}}`, within 30 s, on one line of at most 4 MiB;
- `{"type": "close"}` → `{"ok": true}`, then exit.

Stdout carries only these messages; write logs to stderr. `run.sh` starts with only `PATH`, `HOME`, `LANG` and the
numerics variables above set: export anything else you need there (to render with MuJoCo, `MUJOCO_GL=osmesa`).

**Probes.** About 1 % of steps are asked twice with the same `t`, in a hidden order:
- once with the real observation;
- once with a slightly perturbed copy: each arm's joints moved by up to 1 mrad, with the tool poses and the wrist
  cameras' poses recomputed, and each force by up to 0.2 N; images unchanged.

Only the reply to the real observation is applied. A pair is not counted either way when both replies command no
motion: both arms send 7 numbers whose first six are all within 1e-9 of zero. If fewer than 10 % of the counted pairs
get replies that differ (by more than 1e-4 in some number), the whole run scores 0.

A controller fails this check when its replies do not depend on the arms' joints, tool poses or wrist forces. That
includes:
- a replayed action stream;
- a controller that answers a repeated `t` from a cache;
- one that holds a fixed absolute joint target over most of its moving steps;
- one that moves the tools by steps computed only from images.

Cartesian steps toward a target computed from the observed tool pose pass. Which steps are probed depends only on the
episode, not on your replies.

**Grading.**
- **The episodes.** There are 12: one for each of 12 hidden units. They are a fixed hidden set: each episode's seed
  is the same for every submission, and so are its cabinet pose, rack places, pin stands and noise. They run in a
  hidden order; the observation names the unit.
- **The render test.** Before the first episode the grader draws one test image of each camera.
- **Failures.** A crash, a late or oversize reply, a malformed action, a simulator fatal error or an unstable
  simulation scores that episode 0, and grading goes on. After a failure the next episode restarts the controller;
  after 3 restarts the remaining episodes score 0.
- **The time budget.** Your controller has a budget of 9000 s of its own time over the whole run: the time the
  grader waits for `READY` (at the first start and after a restart), for each reset and for each of your replies,
  plus 0.1 s for every image an applied action asks for (a fixed price, whatever the grading machine takes to draw
  it). The simulation's own time is not counted. The reference controller, which asks for no images, used well under
  a minute of it for all 12 episodes (1614-1941 steps each). If the budget runs out, the episode running is aborted
  and the rest are not run; all of them score 0. The practice grader counts the same way and prints
  `controller_seconds`.
- **Leftover processes.** The whole run scores 0 if the grader
  cannot stop every process your submission left running (it checks when it starts and when the last episode is
  over): do not start anything that respawns or detaches itself to survive a kill.
- **No internet sockets.** Your processes cannot open an internet socket (the call fails). On a grading machine
  where that guard cannot be installed, an internet socket of yours scores the whole run 0.
- **One process.** Your controller runs as one process for the whole run, so it may carry what it learns from one
  episode to the next.
