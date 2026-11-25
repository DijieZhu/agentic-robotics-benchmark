# Serve the hotpot

You are a robotics engineer. Two Franka Pandas stand side by side at a kitchen counter with a hotpot: a pot of clear
broth on a burner with a knob, four small dishes of raw ingredients and three empty bowls. They see through cameras
and feel through wrist force sensors. Write the controller that serves an order: cook the listed ingredients in the
broth and serve each bowl its pieces, cooked and not overcooked, with the ladle (a piece a robot touches does not
count), then pour each bowl its broth from the pot and turn the burner off. It runs closed loop against a simulator
and is graded on pots, burners and ingredients you have not seen. You have 2 hours of wall time (see "Session time"
below for how to read the time left). Whatever is in `/home/user/submission/` when the time ends is graded, so keep a
working controller there.

**The counter.** World frame: z up, metres; the counter top is at z = 0.80. The left arm's base is at (0, 0.28), the
right arm's at (0, -0.28), both facing +x.
- **The burner** is a ring centred at (0.45, 0), radius 14 cm, 4 mm high. It heats the pot only while the pot stands
  on it (as "on its ring" under "Credit").
- **The knob** stands on the counter at (0.25, -0.22): a cap 36 mm across and 20 mm high on a 6 mm base, turning about
  z, with a fin on top (8 mm thick, 32 mm long, 24 mm high; thin along x when the knob is at 0). Its angle, 0 to pi/2
  rad, sets the burner's power in proportion (0: off; pi/2: the unit's full power); below 0.05 rad the burner is off.
  It stays where it is turned.
- **The pot** is a braiser standing on the ring (its centre within 1 cm of the ring's centre, turned at most 3 deg
  about z): inside 25 cm across and 9.5 cm deep (an 8 mm floor, 6 mm walls). It has a **grip tab** on each side, at
  y = +-(about 0.16) from its centre: a block 32 mm along x, 60 mm out from the wall along y and 32 mm high, its
  centre 6.5 cm above the pot's floor. Held by one tab, the pot turns about it: it needs both hands.
- **The broth** is clear: its surface is drawn tinted and half transparent, and what is under it shows through. It is
  4.5-6 cm deep in the pot and at 80 C at the start.
- **The dishes** are four small plates on the left of the pot, centred within 1 cm of (0.36, 0.34), (0.52, 0.34),
  (0.36, 0.49) and (0.52, 0.49): 10 cm across inside, a 4 mm floor and a 2 mm lip, and a flat grip tab 44 mm long, 24
  mm wide and 8 mm thick sticking out toward -x, raised so its centre is 22 mm above the plate's underside. Each holds
  the raw pieces of one ingredient; which dish holds which changes every episode.
- **The ladle** is a slotted skimmer standing in the pot: a dish 9 cm across whose floor is five bars with 4 mm slots
  (they let broth through, not pieces), a 15 mm rim round its back and sides and none at its front (an opening of
  127 deg), and a handle 25 cm long, 18 mm wide and 10 mm thick leaving the back of the rim at 70 deg from the dish's
  plane. It starts with its dish on the pot's floor 3.5 cm to the -y side of the pot's centre, its front toward +y
  and its handle leaning toward -y (the right arm's side).
- **The bowls**, white, blue and yellow, stand in a row at x = 0.72, one at y = 0.15, one at 0 and one at -0.15 (each
  within 1 cm); which bowl stands where changes every episode. Each is 13 cm across at the rim (inside), 8 cm at the
  floor, 6 cm deep inside, with a 6 mm floor, and holds about 530 ml to the rim.

`task_card.md` gives every number in one place.

**The ingredients** (sizes at the unit's size scale 1; each unit scales them all by 0.85-1.15):

| Kind | Shape | Raw / cooked / overcooked colour | In the broth | tau (s) | Cooked at (C) | Window (s) |
|---|---|---|---|---|---|---|
| `fishball` | sphere, 24 mm | grey-white / white / yellowish | sinks raw, floats once cooked | 90 | 72 | 270 |
| `shrimp` | capsule, 15 mm thick, 31 mm long | grey / pink-orange / dark red | sinks | 50 | 63 | 240 |
| `tofu` (puff) | cube, 24 mm | pale yellow / golden / brown | floats | 150 | 70 | 360 |
| `mushroom` (cap) | flattened ellipsoid, 28 x 28 x 16 mm | beige / brown / dark brown | floats | 60 | 65 | 240 |

**How a piece cooks.** Each piece has a core temperature, 20 C raw. While at least 30 % of it is under the pot's broth
surface it follows dT/dt = (T_broth - T) / (tau x scale^2); anywhere else (on a dish, in the ladle, in a bowl, in the
air) it cools toward 20 C with a 600 s time constant. Only the pot's broth cooks: a bowl's broth does not. A piece is
**cooked** once its core has reached the kind's "cooked at" temperature, and **overcooked** once it has spent more
than the kind's window under the pot's broth since. Its colour moves continuously from raw to cooked as its hottest
core temperature so far rises from 20 C to the cooked temperature, then from cooked to overcooked over the window. A
fish ball's lift in the broth grows the same way, from sinking (1080 kg/m^3) to floating (960). Shrimp are 1070
kg/m^3, tofu puffs 550, mushroom caps 700.

**How the broth heats.** Its temperature T_broth (C, at most 100: it boils) follows
m c dT_broth/dt = P - 9 (T_broth - 20) - (the heat going into the pieces under it), with m the broth's mass (1 kg per
litre), c = 4000 J/(kg K) and P the burner's power (the knob's share of the unit's full power, while the pot is on the
ring). A piece of mass m_i takes m_i x 3600 x (T_broth - T_i) / tau_i watts. Neither the broth's nor a piece's
temperature is observed.

**The job.** Each episode's order, `order` in the observation (for example
`{"white": {"items": {"fishball": 1, "shrimp": 1, "tofu": 0, "mushroom": 0}, "soup_ml": 150}, "blue": {"items":
{"fishball": 0, "shrimp": 0, "tofu": 2, "mushroom": 1}, "soup_ml": 220}, "yellow": {"items": {"fishball": 0,
"shrimp": 2, "tofu": 0, "mushroom": 0}, "soup_ml": 100}}`), lists for every bowl how many pieces of each kind (2 or 3
in a bowl, so 6-9 in all; each of a kind at random, no kind more than 4 in all) and how much broth (100-250 ml, in
steps of 10 ml) it must get. Each dish holds the order's count of its kind and one more. Get the pieces into the broth
(for example by tipping a dish over the pot by its tab), serve them, pour the broth, put the pot back on its ring,
turn the burner off and let go of the pot.

**What is hidden per unit.** Each unit differs in:

| Property | Range |
|---|---|
| The pot's mass, empty | 1.4-2.4 kg |
| Broth in the pot at the start | 2.2-3.0 litres |
| How fast the broth pours (its discharge coefficient Cd, below) | 0.35-0.75 |
| The burner's full power | 2.0-3.5 kW |
| The pieces' size scale | 0.85-1.15 |
| The pieces' friction | 0.4-0.8 |

The two practice units in `units.json` are drawn from the same ranges, with their values given. The scored units'
values are not given: your controller has to find out how its burner, broth and pieces behave, for example from the
pieces' colours in the cameras, and from the wrist forces when it lifts and tilts the pot.

**How the broth pours.** It is simulated as a liquid volume per vessel (the pot and each bowl), not as particles:
- its surface is level, at the height that holds the vessel's volume at the vessel's pose;
- where the surface stands above the rim, broth leaves over the rim at q = (2/3) Cd sqrt(2 g) d^1.5 m^3/s per metre of
  rim, d being how far the surface stands above that point of the rim;
- what leaves flies as a stream (launched from the rim with the rim's velocity plus the outflow's) and lands in the
  vessel whose mouth it falls through, or on the counter: that broth is spilled;
- broth weighs 1030 kg/m^3 and pulls on its vessel at its centroid, so the hands holding the pot feel it get lighter
  as it pours;
- the broth does not move the pieces sideways: they move only when something pushes them.

## What is in your home directory

| Path | What it is |
|---|---|
| `soup_sim.py` | The simulator of the counter: the robots and their controllers, the burner and its knob, the pot, the broth and its heat, the dishes and the pieces, the ladle and the bowls, the cameras, each unit's and each episode's draw. The grader runs this exact file. |
| `soup_env.py` | The environment your controller talks to: `SoupEnv(unit).reset(seed)`, `step(action)`, the observation. The grader runs this exact file. |
| `episode.py` | An episode, its end and its credit. The grader runs this exact file. |
| `units.json` | The two practice units (p0, p1). |
| `task_card.md` | The counter's numbers in one place. |
| `practice_grader/grade.py` | Runs your controller over the wire, as the grader does, on the practice units; prints the raw credit. |
| `helpers/wire_skeleton.py` | Optional boilerplate for the wire (no control logic): copy it into your submission. |
| `session_time.sh` | Prints the minutes of your session used and left (see "Session time"). |

Python is `/opt/venv/bin/python3` (MuJoCo 3.16, numpy, scipy, OpenCV, Pillow; robosuite 1.5.2's robot model files).
There is no network.

**The sandbox** has 4 CPUs and 12 GB of memory, and so has the sandbox your controller is graded in. One simulation
takes about 0.5 GB and one CPU. A control step takes 20-50 ms of wall time, so a whole 8000-step episode takes 3-7
minutes without images; an image takes about 0.05 s to draw. Run a few simulations at a time, not dozens.

**Numerics.** Every `/opt/venv` Python computes the same here and in the graded run, on whatever machine: the image
sets, when the variables are missing, NumPy's linear algebra on one fixed set of kernels (`OPENBLAS_CORETYPE=Haswell`
with `OPENBLAS_L2_SIZE=256`) and on one thread (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`), NumPy without its
AVX-512 loops (`NPY_DISABLE_CPU_FEATURES`) and OpenCV's IPP at AVX2 (`OPENCV_IPP=avx2`). Do not unset or change them:
the simulator would compute differently, and your controller's results would depend on the grading machine's CPU.

## Credit

Each episode's credit (0 to 1) is computed from the simulation when the episode ends.

**A bowl is done** when both of these hold at the end:
- **Pieces:** it holds exactly the pieces the order lists for it, of each kind exactly that kind's count, and every
  piece in it is cooked, not overcooked and untouched. A piece is in a bowl when its centre is inside the bowl's wall,
  above its floor and below its rim. A piece that any part of either robot (an arm link, a hand, a finger) has touched
  at any time in the episode (any contact between them in the simulation, `World.touched`) spoils the bowl it ends
  in: carry the pieces in the ladle or on their dish (one you do not want can be tipped from the ladle back into the
  pot).
- **Broth:** it holds the listed broth within 20 ml (for 150 ml: 130-170 ml).

The episode's credit is the number of bowls done, over three: 0, 1/3, 2/3 or 1.

The episode's credit is halved if, at the end, any of these holds:
- more than 30 ml of broth has been spilled in all (onto the counter, from the pot or from a bowl);
- a piece is neither in the pot, nor in a bowl, nor on a dish (on the counter, in the ladle, on the floor); a piece is
  in the pot when its centre is inside the pot's wall, above its floor and below its rim, and on a dish when its
  centre is within the dish's lip and less than 4 cm above its floor;
- a bowl is tilted more than 30 deg;
- the pot is not on its ring: its centre more than 3 cm from the ring's centre, tilted more than 10 deg, or its
  bottom more than 1 cm above where it stood at the start;
- the burner is on (the knob at 0.05 rad or more).

Where the ladle and the dishes end does not matter. A controller that does nothing scores 0.

An episode ends at the first of:
- 8000 control steps (13 1/3 simulated minutes);
- 20 consecutive steps in which:
  - the pot has been lifted (its bottom more than 2 cm above where it stood at the start) at some time in the
    episode, and is on its ring as above;
  - the burner is off;
  - both grippers are open at least 70 mm;
  - both hands (the gripper sites) are at least 12 cm from the pot's rim (the circle of the top of its wall);
- an unstable simulation (an arm joint faster than 8 rad/s, or the pot, the ladle, a bowl, a dish or a piece faster
  than 5 m/s), which scores 0.

The score is the mean credit over the graded episodes.

## Observation and action

The controller runs at 10 Hz. The observation is a JSON object:

- **Step, order and knob:** `t` (the step), `unit` (the unit's id), `order`
  (`{bowl: {"items": {kind: n}, "soup_ml": v}}`), `knob` (the knob's angle, rad).
- **Each arm**, under `l` and `r`:
  - `q` and `dq`: 7 joint angles and velocities;
  - `ee_pos` and `ee_quat`: the gripper site's pose, quaternion wxyz;
  - `gripper`: the fingers' opening, m;
  - `wrench`: the wrist's force and torque in the world frame, with noise of 0.3 N and 0.03 N m.
- **Cameras.** For each camera:
  - `T_world_<cam>`: its 4 x 4 pose, OpenCV convention (x right, y down, z forward);
  - `K_<cam>`: its intrinsics;
  - `<cam>`: the image, `{"dtype": "uint8", "shape": [H, W, 3], "b64": ...}`.

  The cameras are `overhead` and `front` (640 x 480, fixed) and `wrist_l` and `wrist_r` (320 x 240, on the hands). An
  image is empty (`[0, 0, 3]`) unless the previous action asked for it; the first observation has all four. There is
  no depth image.

  What an image shows: the counter, the two robots, the burner, the knob, the pot, the broth's surface (tinted and
  half transparent: the pieces under it show through), the stream while broth pours, the dishes, the pieces in their
  current colours, the ladle and the bowls with their broth, in the sizes, places and colours the simulator has them. An image is a function of the simulator's state
  and the camera (plus the pixel noise): the same state gives the same picture here and in the graded run.

The action is a JSON object `{"l": [...], "r": [...], "cams": mask}`. Each arm's list has one of three shapes:

- **7 numbers** `[dx, dy, dz, drx, dry, drz, grip]`. The operational-space controller moves the hand's target from the
  hand's current pose by (dx, dy, dz) m (clipped to 0.03 m) and turns it by the rotation vector (drx, dry, drz) rad
  (clipped to 0.2 rad), both in the world frame. `grip` runs from -1 (fully open, 80 mm) to +1 (closed); the fingers
  squeeze with up to 70 N.
- **14 numbers:** the same 7, then seven joint angles. These are the posture the controller's null space pulls the arm
  toward from now on (the start posture until you set one).
- **8 numbers** `[q1..q7, grip]`: a joint step. A gravity-compensated joint PD moves toward q, each joint's target at
  most 0.1 rad from its current angle. q also becomes the null-space posture.

Joint angles in an action (a null-space posture, a joint step's q) are clipped to the robot's joint limits.

`cams` is a bitmask of the images the next observation carries: 1 overhead, 2 front, 4 left wrist, 8 right wrist.
Images are large (a 640 x 480 image is about 1.2 MB on the wire) and each one you ask for costs 0.2 s of your
controller's time budget (see "Grading"), so ask for one only when you need it.

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
export anything else you need there (for the simulator's camera images, `MUJOCO_GL=osmesa`). The numerics variables
need no export: a `/opt/venv` Python sets them itself.

**Probes.** About 1 % of steps are asked twice with the same `t`, in a hidden order:
- once with the real observation;
- once with a slightly perturbed copy: each arm's joints moved by up to 1 mrad, with the hand poses and the wrist
  cameras' poses recomputed, and each force by up to 0.2 N; images unchanged.

Only the reply to the real observation is applied. A pair is not counted either way when both replies command no
motion: both arms send 7 numbers whose first six are all within 1e-9 of zero. If fewer than 10 % of the counted pairs
get replies that differ (the same count of numbers, at least one of them by more than 1e-4), the whole run scores 0.

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
  the same for every submission, and so are its bowls' places, order, dishes and pieces, and noise. They run in a
  hidden order; the observation names the unit.
- **The render test.** Before the first episode the grader draws one test image of each camera.
- **Failures.** A crash, a late or oversize reply, a malformed action, a simulator fatal error or an unstable
  simulation scores that episode 0, and grading goes on. After a failure of the wire itself (the controller crashed,
  was late, or wrote an oversize or non-JSON line) the next episode restarts the controller, and after 3 restarts
  the remaining episodes score 0. After a malformed action or an unstable simulation the controller is not
  restarted: the same process gets the next episode's `reset`.
- **The time budget** is your controller's own time, 8000 s for all 12 episodes: the time it takes to print `READY`
  and to answer each request (the two requests of a probe both count), summed over the run, plus 0.2 s for every
  image you ask for (a fixed price, whatever the grader's machine takes to draw it; the four images of an episode's
  first observation are free). The simulation's own time is not charged, so the budget does not depend on how fast
  or busy the grading machine is. For scale: 96,000 steps (12 episodes run to their last step) at 20 ms of
  controller time each use 1920 s, and an image every tenth step adds 1920 s. If the budget runs out, the episode
  running is aborted and the rest are not run; all of them score 0. The practice grader reports the same sum for
  your controller.
- **The grading sandbox** has 4 CPUs and 12 GB of memory, like yours, and the same image.
- **One process.** Your controller runs as one process for the whole run, so it may carry what it learns from one
  episode to the next.

## Session time

Your session is stopped 2 hours after it started, and what is in `/home/user/submission/` at that moment is graded.
- **How to read the time.** `bash /home/user/session_time.sh` prints the minutes used and the minutes left, from a
  time stamp written when your sandbox was set up (a few minutes before your session started, so you have slightly
  more than it says, never less). Run it whenever you plan your next step; do not estimate the time from memory or
  from how much work you have done.
- **Use the time.** Stopping early earns nothing: the score is the controller's, not the session's. If the script
  says there is time left, there is.
- **Keep a working controller in the folder** from early on, and put a new version there only after you have run it.
  Make your last change at least 10 minutes before the end: an edit that is cut off, or never run, is what gets
  graded.
