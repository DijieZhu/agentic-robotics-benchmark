# Cable routing through clips

A Franka Panda stands at a bench in a factory hall. On the bench lies a harness form board: a laminate sheet printed
with a harness drawing, with cable clips and guide posts screwed onto it. One cable is fixed to the board at its
connector; the rest of it lies loose on the board. Your job is a controller that makes the robot route the cable
through the clips and round the posts in the order of the drawing, without overloading the connector, on boards it
has not seen, from two cameras, the robot's own sensors and a wrist force/torque sensor.

You have no simulator of the cell. You have this document, a commissioning service with three development boards
(a limited amount of bench time, no ground truth, no score), and 2 hours.

## 1. What you hand in

A folder `/home/user/submission/` with a `run.sh`. The grader starts `bash run.sh` in that folder (as your account,
with `PATH`, `HOME` and `LANG` only, no network, at the lowest priority) and talks to it over standard input and
output, one JSON object per line:

| the grader writes | your program answers |
|---|---|
| (nothing; within 120 s of the start) | the line `READY` |
| `{"type": "reset"}` at the start of every episode | `{"ok": true}` |
| `{"type": "act", "t": <step>, "obs": {...}}` | `{"action": [...]}` within 30 s |
| `{"type": "close"}` | `{"ok": true}`, then exit |

Anything else you print goes to standard error. `/home/user/helpers/wire_skeleton.py` is boilerplate for this wire
(no control logic). Python is `/opt/venv/bin/python3` (NumPy, SciPy, OpenCV, Pillow and MuJoCo are installed, and
robosuite 1.5.2's files with the Panda's model; there is no model of this cell).

Some steps are asked twice with the same `t` and two slightly different observations: in the other copy the joints
read up to 1 mrad differently, the grip site's pose goes with them, and the wrist sensor's noise is a fresh draw
around the same reading; the pictures are the same. Only one of the two replies is applied and you are not told
which. A controller whose replies to such pairs are identical in more than 90 % of the pairs that command any
motion scores 0 for the whole run: it does not react to what it is shown. Two replies of different lengths (8 and
9 numbers) count as identical.

The processes of your account are counted at every reset and before every request: more than 64 at such a moment
end the episode, and your program is started again for the next one.

## 2. The cell

World frame: the robot's base is at the origin, `x` points from the robot across the bench, `z` up; the bench top
is `z = 0`. Lengths in metres, angles in radians.

- **Bench**: `x` from -0.30 to 1.00, `y` from -0.75 to 0.75. At `x = 0.88` stand two cable reels, two parts bins and
  a roll of tape; they are solid and the arm or the cable can touch them.
- **Board**: 0.52 x 0.92 m and 12 mm thick, centred at (0.49, 0): its top is `z = 0.012`. Its print shows a faint
  5 cm grid, the route as a blue line with arrow heads from the connector on, a box at each clip turned to the
  clip's passage, a ring at each post with a tick on the side on which the cable has to pass, and each fixture's
  number in route order. The board has been in use: it carries soiling, fine scratches, rubbed stretches and the
  filled screw holes of earlier fixtures (pale dots in pairs); the bench top is scratched and stained.
- **Cable**: orange, round, 0.70 to 0.90 m long, radius 3.5 to 5.0 mm (this board's values are on the job sheet).
  Its bending stiffness is between 0.0008 and 0.003 N m^2, its friction coefficient between 0.5 and 1.0 and its
  density between 1200 and 2000 kg/m^3; these three are not told. It starts lying on the board in a loose curve,
  clear of the fixtures.
- **Connector**: a black housing on a steel bracket, with a rubber strain-relief boot through which the cable
  leaves along the connector's heading. The boot gives 7 mm at 3.2 kN/m before the conductors take the load. **The
  connector is damaged, and the episode's credit is 0, if the pull on it, averaged over any 0.1 s, exceeds 40 N.**
- **Clip** (a cable saddle): two steel pins 4 mm thick and 20 mm high, a slot between them 4.3 to 4.7 mm wider than
  the cable, a black lead-in fin on each pin's top that slopes outward (7 mm long, 50 degrees from horizontal), and
  on each pin an off-white sprung retaining tab that reaches over the slot. A cable pressed down between the fins
  pushes the tabs aside and they close over it; in tests they kept it in under a pull of 9 N on the cable, and the
  cable can still slide along the passage. The fins guide a cable that comes down within about 5 mm of the slot's
  centre line; 8 mm off, half of the presses fail; 11 mm off, all do. The clip's passage axis is the direction in
  which the cable runs through it; it follows the route, and a cable that arrives along the route from the fixture
  before enters the clip: the axis does not have to be measured. **A clip is made when the settled cable's centre line passes between the two
  pins lower than 13 mm above the board.**
- **Post**: an anodised post of 12 mm radius and 30 mm height with a black cap of 18 mm radius, 6 mm thick. The cable
  has to pass it on the stated side. **A post is made when the settled cable comes within 10 mm of the post's
  surface, below its top, with the post on the stated side of the cable** (`"left"`: the post is on the left of the
  cable as it runs from the connector).
- **The route is judged on the settled cable**: when an episode ends the cell opens the hand and holds it
  for 0.4 s, raises it 15 cm over 1.2 s and holds it there for 0.8 s; then the cable is judged. A stiff cable that is left slack can spring away from a post after the hand lets go, and work at a
  later fixture can pull or push it off a post that was made before. Fixtures count in route order, by where
  along the cable they hold it.
- **The hand**: the Panda's two-finger hand with cable jaws: a V groove on each finger, 16 mm long, whose axis is
  17 mm below the grip site toward the fingertips. The fingers close along the grip site's `x` axis; the groove,
  and a held cable, lie along its `y` axis; its `z` axis points out of the hand. To take a cable that lies on the
  board, the grip site has to be 17 mm plus the cable's radius above the board's top, with the cable between the
  open jaws. Each finger presses with up to 60 N; a gripped cable does not slip in the groove below a pull of 60 N,
  so the connector's limit is reached before the grip slips.
- **Mounting tolerances**: the connector and every fixture are mounted within 20 mm of their printed places. The
  print and the job sheet give the nominal places; where the fixtures really are you find from the pictures.

## 3. Observation and action

Every observation `obs` holds:

| key | what |
|---|---|
| `t`, `time` | the control step (10 a second) and the time in seconds |
| `q`, `dq` | the arm's seven joint angles and speeds |
| `gripper_width` | the opening between the fingers (0.08 open) |
| `tcp` | `pos` and `quat` (w, x, y, z) of the grip site in the world, as the cell's last physics step computed it: the pose 2 ms before the joint angles reported beside it |
| `wrench` | `force` (N) and `torque` (N m) of the wrist sensor in world axes, with noise (0.3 N and 0.03 N m standard deviation per axis); it includes the hand's own weight |
| `overhead`, `wrist` | a picture, only when asked for: `{"w", "h", "png"}`, a base64 PNG (RGB) |
| `job` | the job sheet, the same in every observation of an episode |

`job`: `connector` (`pos` [x, y], `heading`: the direction in which the cable leaves it), `fixtures` in route order
(`type` "clip" with `pos`, or "post" with `pos` and `side`), all nominal; `cable` (`radius`, `length`);
`mount_tolerance`; `max_steps` (1500), `control_dt` (0.1); `cameras`.

**Cameras** (`job.cameras`: `pos`, `quat`, `fovy_deg`, `w`, `h`; a camera looks along its own -z axis, +x to the
picture's right, +y up):

- `overhead`, 960 x 720, fixed 1.05 m above the board's centre, looking straight down; the picture's top is the far
  side of the board (+x), its right is -y. It shows the whole board and a strip of bench round it. **The arm reaches
  over the board from the picture's bottom edge and hides what is under it**, at the start pose the middle of the
  board's near half. A clip is about 10 pixels across in it: enough to find it, not to place a cable in
  its slot.
- `wrist`, 480 x 360, on the hand (its pose is given in the grip site's frame), looking out of the hand along the
  grip site's `z`. The lowest sixth of its picture is the hand's own flange, and the two jaws show at the lower
  edge. From 15 to 30 cm above the board it shows a fixture large enough to measure its place.

There are no shadows of a lamp in the pictures; the light is even.

An **action** is 8 numbers `[dx, dy, dz, drx, dry, drz, grip, cams]`, or 9 with `finish` as the last:

- `dx, dy, dz`: a step of the grip site from where it is, in world axes, at most 0.03 m long (longer is cut);
- `drx, dry, drz`: a rotation of the grip site as a rotation vector in world axes, at most 0.2 rad (cut);
- `grip`: -1 open to 1 closed;
- `cams`: the pictures wanted with the next observation: 0 none, 1 overhead, 2 wrist, 3 both. The first observation
  of an episode has both;
- `finish`: 1 hands the job over: the step is not applied and the episode ends. Without it the episode ends after
  1500 steps (150 s).

The arm is driven by an operational-space controller toward the commanded pose; it holds its pose when you send
zeros.

## 4. Practice: the commissioning service

Three development boards (numbers 0, 1, 2; three fixtures each; none is a graded board) run behind a socket under
another account. They give exactly the observations above and nothing else: no fixture's true place, no cable
shape, no tension, no word on whether a fixture is made or the connector was overloaded. After an episode has ended
the cell has opened the hand, lifted it and let the cable settle, as the graded cell does; a `look` then shows you
the settled board.

```python
import sys; sys.path.insert(0, "/home/user")
from commission.client import Bench, decode_image
b = Bench(); r = b.reset(0); obs = r["obs"]
r = b.step([0, 0, -0.01, 0, 0, 0, -1, 2])      # r["obs"], r["done"], r["ended"], r["steps_left"]
r = b.look(3); r = b.probe(); b.close()
```

`/home/user/commission/run_bench.py --board 0` runs your `run.sh` on a board exactly as the grader does (the same
wire, deadlines and repeated steps) and reports what the grader would know about the controller.

The window is **20000 control steps** in all; staging a board costs 100, every picture costs 2 (in an action or a
`look`). At most two boards at a time. A step takes about 0.1 s of real time and a picture about 0.1 s more, a
reset about 8 s. `Bench().status()` tells what is left.

## 5. Grading

The grader runs your controller on a fixed hidden set of 20 boards with 3 or 4 fixtures each (clips and posts in
any order), drawn like the development boards. An episode's credit is **1 if the whole route is made in order and
the connector was not damaged, otherwise 0**; the score is the mean over the 20 boards. An episode ends at your
`finish`, after 1500 steps, or at a fault of your program; the cable is then judged as it lies.

- A reply that is late or missing, a program that has ended, or too many processes: your program is started again
  for the next episode, at most three times in a run. After the third restart, the next such fault ends every
  remaining episode at its start (those boards score 0).
- A malformed action ends the episode without a restart: your program keeps running and gets the next `reset`.
- A simulation that turns unstable (a numerical blow-up, or an arm joint faster than 8 rad/s) ends the episode,
  which scores 0.
- If the grader's own simulation process fails during an episode, the episode is run again from its start on the
  same board and its time is not charged: your program gets another `reset` in the middle of an episode and is not
  told why. Clear your controller's state at every `reset`.

Your controller has **3600 s of its own time** for the whole set: the time the grader waits for `READY`, for each
reset and for each reply, plus **0.2 s for every picture you ask for** (whatever the grader's machine takes to draw
it; the first observation's two pictures are free). Not counted: the simulation, the drawing, and what your program computes between two requests (the cell does
not pause your processes; they run at the lowest priority beside the simulation). When the budget is used
up the running episode ends as it lies and the remaining boards score 0. Asking for both pictures at every step of
every episode would cost more than three times the budget.

The grading machine gives your controller 4 CPUs shared with the simulation and no GPU; this machine has 4 CPUs and
12 GB.

## 6. Session time

You have 2 hours from the start of the session, for everything. `bash /home/user/session_time.sh` prints how much is
used and left. Whatever is in `/home/user/submission/` when the session ends is graded; keep a working `run.sh`
there from early on.
