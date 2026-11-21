# Restock a grocery shelf

You are a retail-automation engineer. A Franka Panda on a vertical lift stands in a store aisle in front of one
gondola bay of three shelves, with a tote of delivered products on a cart beside it. Write the controller that
restocks the bay from the tote, following the store's planogram and rotating the stock by date. It runs closed loop
against a simulator and is graded on bays you have not seen. You have 2 hours of wall time; read the clock (`date`) before you
decide that time is short.

**The bay.** Three shelves, five lanes each. Shelves are numbered 0-2 from the bottom; lanes 0-4 from the right as
seen from the aisle. Each shelf holds products of one height, and its headroom (from the board to the shelf above) is
that height plus 0.32-0.35 m. Each episode places the bay within a few centimetres of a nominal place, turned by up
to 5 deg, and each bay has its own board heights, depth and lane spacing. `task_card.md` gives the nominal place and
the ranges; your cameras have to find the real ones.

**The products.** Each shelf carries two products (SKUs) of the same size class and height; two shelves of a bay may
carry products of the same class and height too, so a unit's shape does not tell which shelf it belongs to. A
product is known only by its printed look: a body colour, a label band, a logo and, on bottles, a cap colour. SKUs
may share a body colour; their bands and logos tell them apart. A box has a printed front, the face with the logo.

**The dates.** Every unit carries its best-before date: a two-digit day number in dark seven-segment digits 7 mm
tall on a small light patch, on one side only. The place differs from unit to unit: any one of a box's four sides,
any angle round a can or bottle, at any height in the upper part of the unit (the task card gives the ranges).
You read it facing that side with the unit upright, and from close up: the digits' strokes are 0.8 mm wide. Units on
the shelf and in the tote stand at any turn about their axis (boxes in a lane face out), so a code may face away from
every camera.

**The order.** Every observation carries `order`:
- `planogram`: each lane's SKU, e.g. `{"0:3": "bottle0a", "1:2": "box1b", ...}` (`"<shelf>:<lane>"`);
- `delivery`: how many units the tote holds for each lane to be restocked, e.g. `{"0:3": 2, "1:2": 1}`;
- `today`: today's day number, e.g. `47`.

The first observation also carries `catalog`: one front-view image of each SKU, named as in the planogram.

**The job.** For every lane in `delivery`:
- **remove the expired units:** a unit whose day number is before `today` leaves the bay and ends standing upright
  in the tote;
- put the delivered units of the lane's SKU from the tote into the lane, together with its unexpired old units;
- **rotate the stock by date:** from the front of the lane to the back, each unit's day number is no earlier than
  the one in front of it (the earliest sells first; equal numbers in either order). A delivered unit is not always
  later than the old ones;
- leave the lane faced: its front unit within 15 mm of the shelf's front lip and, for a box, its printed front
  facing the aisle.

A lane to be restocked holds one or more old units, anywhere in its depth, not faced; units behind the front one cannot be seen from the aisle. The delivery fills every
restocked lane: once its expired units are gone, the lane holds exactly as many units as fit (`task_card.md`). The
tote holds the delivery and extras: units of SKUs that no lane in `delivery` carries, which are not part of the
delivery and stay standing in the tote. At most one of its eight slots is free, often none. Only the unit at the
front of a lane can be taken out, and a unit can only be set down in front of the others, so plan where units wait:
in the tote, or in a lane being restocked (any SKU while it waits). In every graded episode the tote's free slots
alone are not enough: some unit has to be set down in a lane and taken out again before it reaches its place. Do
not disturb the lanes that are not being restocked.

## What is in your home directory

| Path | What it is |
|---|---|
| `shelf_env.py` | The environment your controller talks to: `ShelfEnv(unit).reset(seed)`, `step(action)`, the observation. The grader runs this exact file. |
| `shelf_world.py` | The world: the bay, the cart and tote, the products and their looks, the robot on its lift, each unit's and episode's draw. The grader runs this exact file. |
| `shelf_ctl.py`, `shelf_sim.py`, `box_transport_env.py` | The controller, the scene's building blocks (and a kinematic IK), the robot's definitions. The grader runs these exact files. |
| `episode.py` | An episode and its credit. The grader runs this exact file. |
| `units.json` | Three practice units (p0-p2): bays and SKUs to develop against, each with eight seeds of episodes like the graded ones (`seeds`). The scored bays and SKUs are different. |
| `task_card.md` | The scene's numbers in one place: frames, nominal places, ranges, sizes, cameras. |
| `practice_grader/grade.py` | Runs your controller over the wire, as the grader does, on the practice units; prints the raw credit. |
| `helpers/wire_skeleton.py` | Optional boilerplate for the wire (no control logic); copy it into your submission to use it. |

Python is `/opt/venv/bin/python3` (MuJoCo 3.16, numpy, scipy, OpenCV, Pillow). There is no network.

## Credit

Each episode's credit (0 to 1) is computed from the simulation when the episode ends. A unit **stands in a lane**
when its centre is within 30 mm of the lane's centre line, between the lip and the back panel, resting on that
lane's board (its centre within 12 mm of its resting height) and tilted less than 10 deg.

- **Correct lanes, 0.7**, split evenly over the lanes in `delivery`. A lane is correct when the units standing in it
  are exactly its unexpired old units and the delivered units of its SKU, in date order (above). Anything else is a
  wrong lane: a unit missing, an expired unit or a unit of another SKU in it, two units out of order.
- **Facing, 0.2**, split evenly over the lanes in `delivery`: a correct lane whose front unit's front face is within
  15 mm of the lip's inner face; for a box, its printed front also faces the aisle (its yaw within 20 deg of the
  bay's).
- **Kept, 0.1**: every unit of the lanes not in `delivery` within 20 mm of where it started, no unit anywhere lying
  down (tilted more than 45 deg) or on the floor, and every expired unit and every extra standing upright in the
  tote. It counts
  only in an episode with at least one correct lane.

An episode ends:
- when your controller sends the action `[0]` (finished);
- after 15000 control steps (25 simulated minutes);
- at an unstable simulation (an arm joint faster than 8 rad/s, or a product faster than 5 m/s), which scores 0.

The score is the mean credit over the graded episodes.

## Observation and action

The controller runs at 10 Hz. The observation is a JSON object:

- **Robot:** `t` (step), `unit` (the unit's id), `q` and `dq` (7 joint angles and velocities), `lift` (the robot
  base's height, m), `ee_pos` and `ee_quat` (the gripper site, wxyz), `gripper` (the fingers' opening, m), `wrench`
  (the wrist's force and torque in the world frame, with noise: 0.3 N, 0.03 N m).
- **Cameras:**
  - `T_world_<cam>`: 4 x 4 pose in the OpenCV convention (x right, y down, z forward);
  - `K_<cam>`: intrinsics, a pixel's centre at its index + 0.5;
  - `<cam>`: the image, `{"dtype": "uint8", "shape": [H, W, 3], "b64": ...}`.

  The four cameras are `left` and `right` (400 x 300, on the lift's carriage beside the robot, looking into the
  shelf in front of it: they ride with the lift), `wrist` (256 x 192, on the hand) and `overview` (640 x 480, fixed,
  behind the robot). An image is empty (`[0, 0, 3]`) unless the previous action asked for it; the first observation
  has all four. There is no depth image.
- **The order:** `order` (above); on the first observation only, `catalog` (128 x 128 images, the same encoding).

The action is a JSON list in one of four shapes:

- 9 numbers `[dx, dy, dz, drx, dry, drz, grip, dlift, cams]`: the operational-space controller moves the gripper
  site's target by (dx, dy, dz) m (clipped to 0.03 m) and turns it by the rotation vector (drx, dry, drz) rad
  (clipped to 0.2 rad), both in the world frame. `grip` runs from -1 (fully open, 80 mm) to +1 (closed). `dlift`
  moves the lift's set point by up to 0.01 m (within 0.05-1.30 m); the hand's target rides with the base.
- 16 numbers: the same, then seven joint angles. These are the posture the controller's null space pulls the arm
  toward from now on (the start posture until you set one).
- 10 numbers `[q1..q7, grip, dlift, cams]`: a joint step. A gravity-compensated joint PD moves toward q, each
  joint's target at most 0.1 rad from its current angle. q also becomes the null-space posture.
- `[0]`: finished; the episode ends and is scored.

`cams` is a bitmask of the images the next observation carries: 1 left, 2 right, 4 wrist, 8 overview. Images are
large (the overview alone is about 1.2 MB on the wire), so ask for one only when you need it.

## Deliverable and grading

Put your controller in `/home/user/submission/` with a `run.sh`. Grading runs in a fresh sandbox that holds only
that folder (at the same path, at most 5000 files and 200 MiB, else the run scores 0) and the image's Python: nothing
else of your home directory is there, so copy into the submission everything the controller imports or reads
(the world files above, your own modules, any data). The practice grader runs in your home directory and does not
show this. The grader starts the controller with `bash run.sh` in that folder, as your user, with only local
(AF_UNIX) sockets, and speaks one JSON object per line on its stdin and stdout:

- your controller prints `READY` within 120 s (a controller that never does scores 0 on every episode);
- `{"type": "reset"}` → `{"ok": true}` (an episode starts);
- `{"type": "act", "t": t, "obs": {...}}` → `{"action": [...]}`, within 30 s;
- `{"type": "close"}` → `{"ok": true}`, then exit.

Stdout carries only these messages (a reply line longer than 4 MiB ends the episode); write logs to stderr. `run.sh` starts with only `PATH`, `HOME` and `LANG` set:
export anything else you need there (to render with MuJoCo, `MUJOCO_GL=osmesa`).

**Probes.** About 1 % of steps are asked twice with the same `t`: the real observation and a slightly perturbed copy
(joints by up to 1 mrad with the poses recomputed, force by up to 0.2 N; images unchanged), in a hidden order. Only
the reply to the real one is applied. A pair in which both replies command no motion (`[0]`, or 9 numbers whose
first six and eighth are all zero, within 1e-9) is not counted either way. If fewer than 10 % of the other pairs
get replies that differ (by more than 1e-4 in some number), the whole run scores 0: a controller that ignores its
input, answers a repeated `t` from a cache, or commands a fixed absolute joint target (the same 10 numbers whatever
it is told) fails. Which steps are probed depends only on the episode, not on
your replies.

**Grading.**
- There are 12 episodes: 2 for each of 6 hidden units (bays with their own SKUs), a fixed hidden set (each
  episode's seed is the same for every submission, so are its bay pose, planogram, delivery, old stock and noise),
  in a hidden order (the observation names the unit).
- Before the first episode the grader draws one test image of each camera and the catalog, which takes a few
  seconds of its budget.
- A crash, a late reply, a malformed action, an unstable simulation or a simulator error scores the episode 0. After 3 restarts the
  remaining episodes score 0.
- The grader's time budget is 9000 s in all. An episode still running when it runs out is aborted (scores 0), and
  episodes it cannot start in time score 0.
- Your controller is one process for the whole run, so it may carry what it learns about a unit from one episode
  to the next.
