# Build the pipe spool

You are a robotics engineer. Two Franka Pandas stand side by side at a plumbing workbench with two copper pipes and
two push-fit elbows on it. They see through cameras and feel through wrist force sensors. Write the controller that
builds a small pipe spool from these parts **in free space**: one hand holds a part while the other pushes the next
part onto it. Nothing is clamped, so whatever the pushing hand pushes, the holding hand has to take. The controller
runs closed loop against a simulator and is graded on fittings and work orders you have not seen. You have 3 hours of
wall time (see "Session time"). Whatever is in `/home/user/submission/` when the time ends is graded, so keep a
working controller there.

**The cell.** World frame: z up, metres; the bench top is at z = 0.80. The left arm's base is at (0, 0.28), the right
arm's at (0, -0.28), both facing +x. Both arms have the Panda gripper, which closes on what it holds with 70 N per
finger. `task_card.md` gives every number in one place.
- **Pipes.** Two 22 mm copper pipes with chamfered ends: pipe A (0.22-0.30 m long) and pipe B (0.19-0.23 m). Each
  lies loose on the bench roughly along y (turned by up to 30 deg), A nominally at (0.40, 0.24), B at (0.58, 0.18),
  each moved by up to 15 mm every episode. Nothing holds them: a pipe that is pushed rolls. A carries a blue band near
  one end: its **banded end**. Which end is banded changes every episode.
- **Elbows.** Two identical 90 deg push-fit elbows are on the bench, nominally at (0.42, -0.24) and (0.62, 0.44),
  each moved by up to 15 mm and turned by any angle every episode. An elbow is a corner block with two sockets at
  90 deg; a socket's mouth is 50 mm from the corner's centre. An elbow either lies flat (both sockets level) or stands
  on one socket's mouth (that socket pointing down, the other one level, its axis 50 mm above the bench). At least
  one elbow stands in every episode; often both do. Which elbow stands, on which socket, and how it is turned change
  every episode.
- **A push-fit socket.** A pipe end pushed into a socket meets an O-ring 4 mm inside the mouth. The O-ring lets the pipe
  past only when the pipe comes square (its axis within 1 deg and 0.5 mm of the socket's) and pushes on it harder than
  the unit's insertion force; a skewed pipe rests on it however hard it is pushed. Past the O-ring the pipe slides in
  against the same friction until it meets the socket's stop. Within 1 mm of the stop the joint is **home** and locks:
  the pipe cannot be drawn back any more, and it still turns in the socket against the unit's turning torque. A pipe
  that is not yet home can be drawn back out.

**The job.** The observation's `order` (for example `{"twist_deg": 180}`) is the work order. Build the spool:
1. an elbow on A's banded end (call it the B-elbow);
2. B's one end in the B-elbow's other socket;
3. the other elbow on A's other end, its free socket turned the order's twist (0 or 180 deg) from the B-elbow's
   B-socket about A's axis: project both sockets' outward axes on the plane normal to A; twist 0 means they point the
   same way, 180 opposite ways (the measured angle runs counter-clockwise from the B-socket's axis to the free
   socket's when A's axis, pointing from the B-elbow to the other elbow, points at you);
4. set the spool down on the bench, open both hands and move them away from it. This step ends the episode early
   (see "Credit"); it is not part of the credit itself: a spool still held in the air, every part above the bench
   top, when the episode's steps run out scores the same.

**In free space.** A joint counts only if it was made in the air: if, at the moment its pipe goes past the O-ring or
at any moment while its pipe is pushed further in, any part connected to it touches anything but the two grippers and
the other spool parts (the bench, a robot's arm), the joint is made on a support and does not count. A home
joint made on a support cannot be undone (it is locked). So one hand holds while the other pushes.

**What is hidden per unit.** The two elbows and the two pipes of a unit come from one batch:

| Property | Range |
|---|---|
| Insertion force (the O-ring's push to pass, and the friction after it) | 10-25 N |
| Stop depth from the mouth | 20-28 mm |
| Turning torque of a joint | 0.5-1.2 N m |
| Bore clearance (radial) | 0.2-0.4 mm |
| Elbow mass | 0.08-0.16 kg |
| Pipe mass per metre | 0.45-0.75 kg/m |

The elbows look the same whatever their stop depth. The two practice units in `units.json` are drawn from the same
ranges, with their values given. The scored units' values are not given: your controller has to find out, from the
wrist forces and the cameras, when a pipe has passed the O-ring and when it is home.

## What is in your home directory

| Path | What it is |
|---|---|
| `spool_sim.py` | The simulator of the cell: the robots and their controllers, the pipes, the elbows and their push-fit joints, the cameras, each unit's and each episode's draw. The grader runs this exact file. |
| `spool_env.py` | The environment your controller talks to: `SpoolEnv(unit).reset(seed)`, `step(action)`, the observation. The grader runs this exact file. |
| `episode.py` | An episode, its end and its credit. The grader runs this exact file. |
| `spool_render.py`, `spool_scene.py`, `spool_parts.py`, `spool_geom.py`, `spool_looks.py`, `spool_look.json` | What the cameras draw and the renderer that draws it (see "The pictures"). The grader runs these exact files. |
| `units.json` | The two practice units (p0, p1). |
| `task_card.md` | The cell's numbers in one place. |
| `practice_grader/grade.py` | Runs your controller over the wire, as the grader does, on the practice units; prints the raw credit and what it is made of (see below). |
| `session_time.sh` | Prints the minutes of your session used and left (see "Session time"). |
| `helpers/wire_skeleton.py` | Optional boilerplate for the wire (no control logic): copy it into your submission. |

Python is `/opt/venv/bin/python3` (MuJoCo 3.16, numpy, scipy, OpenCV, Pillow; robosuite 1.5.2's robot model files).
There is no network. Your sandbox has 4 CPUs and 12 GB of memory; the grading sandbox has the same. A control step
takes 15-40 ms of wall time, so a whole 6000-step episode takes 2-4 minutes without images; an image takes 0.06-0.12 s
to draw, and an episode's first image about 3 s more (its render process starts then). Run a few simulations at a
time, not dozens.

**The pictures.** The cameras show the cell as a workshop would have it, drawn by MuJoCo's Filament renderer on the
CPU (`spool_render.py`; a process of its own per episode) from `spool_scene.py`'s parts and a render bundle in
`/opt/task-assets/payload/render` (readable, not writable):
- the parts at the simulation's own geometry: the pipes as 22 mm copper tube, A's band as a ring of blue tape, the
  elbows as brass push-fit elbows with a dark collar at each socket's mouth, the Pandas with their grippers' soft
  pads. An elbow is drawn inside its collision shape, and nothing drawn shows a unit's stop depth;
- the bench as a wooden worktop on a steel frame, with a steel plate under each robot (flush, in the top's plane);
- a factory hall round the bench (racks, tool chests, pallets): drawn only, nothing of it is in the simulation;
- the camera rig, drawn and with contact boxes of the same size in the simulation (geoms `x_*`; a hand cannot pass
  through them): a mast behind the robots at (-0.47, 0) with a boom over the bench at z = 2.25 carrying the
  `overhead` camera and, on an arm, the `front` camera; a floor stand at (0.45, -1.0) carrying the `side` camera; a
  camera body at each of these three cameras; and a small camera block on each hand at its wrist camera (geom
  `wcam`: part of the hand, not of the gripper, so a spool part that touches it touches "something else").
The light is fixed and no lamp casts a shadow. The sensor noise is added to the drawn picture as before.

**Practice.** You may practise as much as the session allows: the simulator is yours to run, and the practice units'
truth is yours to read (`units.json` gives their values; the simulator's state gives everything else). For each
practice episode the practice grader prints the credit, every joint (socket, pipe, end, home, made on a support, depth
in mm), how many joints count, whether the spool lies on the bench, its shape and twist, the order's twist, the
largest push each O-ring took before it let a pipe in, and the controller's time as the grader counts it
(`controller_seconds`). None of this exists for the scored units: they are other batches, every hidden property of
which is drawn anew within the ranges above, and nothing but the observation is given for them.

**Numerics.** Every `/opt/venv` Python computes the same here and in the graded run, on whatever machine: the image
sets, when the variables are missing, NumPy's linear algebra on one fixed set of kernels (`OPENBLAS_CORETYPE=Haswell`
with `OPENBLAS_L2_SIZE=256`) and on one thread (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`), NumPy without its
AVX-512 loops (`NPY_DISABLE_CPU_FEATURES`) and OpenCV's IPP at AVX2 (`OPENCV_IPP=avx2`). Do not unset or change them:
the simulator would compute differently, and your controller's results would depend on the grading machine's CPU.

## Credit

Each episode's credit (0 to 1) is computed from the simulation when the episode ends:
- **0.2 per joint** that is home, was made in the air, while every spool part's centre of mass is above the bench top
  (at most three joints count);
- **0.4 for the shape**: the three joints count, the B-elbow is on A's banded end, and the twist is within 5 deg of
  the order's.

Examples, for the order `{"twist_deg": 180}`:

| At the end | Credit |
|---|---|
| all three joints home and made in the air, the B-elbow on the banded end, twist 183 deg, the spool on the bench | 1 |
| as above, twist 172 deg | 0.6 |
| as above, but the B-elbow on A's plain end | 0.6 |
| two joints home in the air; B pushed in while A's other end rested on the bench | 0.4 |
| three joints home in the air, the spool dropped to the floor | 0 |

A controller that does nothing scores 0.

An episode ends at the first of:
- 6000 control steps (10 simulated minutes);
- 20 consecutive steps in which three joints are home, both grippers are open at least 70 mm, both hands (the grip
  sites) are at least 8 cm from every spool part, and every spool part moves slower than 2 cm/s;
- an unstable simulation (an arm joint faster than 8 rad/s, or a pipe or an elbow faster than 5 m/s), which scores 0.

The score is the mean credit over the graded episodes.

## Session time

- This session is stopped 3 hours of wall-clock time after it started. What is in `/home/user/submission/` at
  that moment is graded as it is: keep a complete, runnable submission there from early on and replace it as you
  improve it.
- The only clock for this limit is the wall clock. `bash /home/user/session_time.sh` prints the minutes used and
  the minutes left, from a time stamp written when your sandbox was set up (moments before your session started,
  so you have slightly more than it says, never less); `date` shows the same clock. Run the script when you begin,
  again before every decision that depends on the time left, and whenever you pick the work up from notes or from
  a summary of what was done. Do not estimate the time used from how much you have done.
- Two other counts in this task are not your session's time: an episode's length (6000 control steps are 10
  simulated minutes and a few minutes of wall time in your simulator) and the controller's 9000 s budget at
  grading. Neither says how much of the 3 hours is left.

## Observation and action

The controller runs at 10 Hz. The observation is a JSON object:

- **Step, unit and order:** `t` (the step), `unit` (the unit's id), `order` (`{"twist_deg": ...}`).
- **Each arm**, under `l` and `r`:
  - `q` and `dq`: 7 joint angles and velocities;
  - `ee_pos` and `ee_quat`: the grip site's pose, quaternion wxyz;
  - `gripper`: the fingers' opening, m;
  - `wrench`: the wrist's force and torque in the world frame, with noise of 0.3 N and 0.03 N m.
- **Cameras.** For each camera:
  - `T_world_<cam>`: its 4 x 4 pose, OpenCV convention (x right, y down, z forward);
  - `K_<cam>`: its intrinsics;
  - `<cam>`: the image, `{"dtype": "uint8", "shape": [H, W, 3], "b64": ...}`.

  The cameras are `overhead`, `front` and `side` (640 x 480, fixed) and `wrist_l` and `wrist_r` (320 x 240, on the
  hands, looking along the fingers). An image is empty (`[0, 0, 3]`) unless the previous action asked for it; the
  first observation has all five. There is no depth image. Part poses are not in the observation.

The action is a JSON object `{"l": [...], "r": [...], "cams": mask}`. Each arm's list has one of three shapes:

- **7 numbers** `[dx, dy, dz, drx, dry, drz, grip]`. The operational-space controller moves the grip site's target
  from its current pose by (dx, dy, dz) m (clipped to 0.03 m) and turns it by the rotation vector (drx, dry, drz) rad
  (clipped to 0.2 rad), both in the world frame. `grip` runs from -1 (fully open, 80 mm) to +1 (closed, squeezing
  with 70 N per finger).
- **14 numbers:** the same 7, then seven joint angles. These are the posture the controller's null space pulls the arm
  toward from now on (the start posture until you set one).
- **8 numbers** `[q1..q7, grip]`: a joint step. A gravity-compensated joint PD moves toward q, each joint's target at
  most 0.1 rad from its current angle. q also becomes the null-space posture.

Joint angles in an action are clipped to the robot's joint limits. The operational-space controller compensates the
arm's own weight, not what the hand holds or pushes: a push or a held load moves the hand off its target until you
command against it.

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

**The wire.** The grader starts your controller with `bash run.sh` in that folder, as your user, at a low CPU priority
(nice 19), with only local (AF_UNIX) sockets and at most 4096 processes for your user. It speaks one JSON object per
line on the controller's stdin and stdout:

- the controller prints `READY` within 120 s of starting; a controller that never prints it scores 0 on every
  episode;
- `{"type": "reset"}` → `{"ok": true}`: an episode starts;
- `{"type": "act", "t": t, "obs": {...}}` → `{"action": {...}}`, within 30 s, on one line of at most 4 MiB;
- `{"type": "close"}` → `{"ok": true}`, then exit.

Stdout carries only these messages; write logs to stderr. `run.sh` starts with only `PATH`, `HOME`, `LANG` and the
numerics variables above set: export anything else you need there (to render with MuJoCo, `MUJOCO_GL=osmesa`).

**Probes.** About 1 % of steps are asked twice with the same `t`, in a hidden order:
- once with the real observation;
- once with a slightly perturbed copy: each arm's joints moved by up to 1 mrad, with the grip-site poses and the
  wrist cameras' poses recomputed from them the way the real observation's are, and each wrist's force and torque
  read once more (the same true wrench with noise drawn anew); images unchanged.

Only the reply to the real observation is applied. A pair is not counted either way when both replies command no
motion: both arms send 7 numbers whose first six are all within 1e-9 of zero. A pair whose two replies hold a
different count of numbers is counted and does not show a difference. If fewer than 10 % of the counted pairs get
replies that differ (the same count of numbers, one of them by more than 1e-4), the whole run scores 0.

A controller fails this check when its replies do not depend on the arms' joints, grip-site poses or wrist forces.
That includes:
- a replayed action stream;
- a controller that answers a repeated `t` from a cache;
- one that holds a fixed absolute joint target over most of its moving steps;
- one that moves the hands by steps computed only from images.

Cartesian steps toward a target computed from the observed grip-site pose pass. Which steps are probed depends only
on the episode, not on your replies.

**Grading.**
- **The episodes.** There are 12: one for each of 12 hidden units. They are a fixed hidden set: each episode's seed
  is the same for every submission, and so are its parts' places, its order and its noise. They run in a hidden
  order; the observation names the unit.
- **The render test.** Before the first episode the grader draws one test image of each camera.
- **Failures.** A crash, a late or oversize reply, a malformed action, a simulator fatal error or an unstable
  simulation scores that episode 0, and grading goes on. After a failure the next episode restarts the controller;
  after 3 restarts the remaining episodes score 0.
- **The time budget.** Your controller has a budget of 9000 s of its own time over the whole run: the time the
  grader waits for `READY` (at the first start and after a restart), for each reset and for each of your replies,
  plus 0.1 s for every image an applied action asks for (a fixed price, whatever the grading machine takes to draw
  it). The simulation's own time is not counted. The reference controller, which asks for no images, used about
  20 s of it for all 12 episodes (863-1056 steps each). If the budget runs out, the episode running is
  aborted and the rest are not run; all of them score 0. The practice grader counts the same way and prints
  `controller_seconds`.
- **Leftover processes.** The whole run scores 0 if the grader cannot stop every process your submission left running
  (it checks when it starts and when the last episode is over): do not start anything that respawns or detaches
  itself to survive a kill.
- **No internet sockets.** Your processes cannot open an internet socket (the call fails). On a grading machine
  where that guard cannot be installed, an internet socket of yours scores the whole run 0.
- **One process.** Your controller runs as one process for the whole run, so it may carry what it learns from one
  episode to the next.
