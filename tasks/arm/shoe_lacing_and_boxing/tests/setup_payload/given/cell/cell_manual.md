# Finishing cell: the integrator's description

Everything below is fixed for the task unless marked *per batch* (it then comes from `describe()`) or *per episode /
per pair* (it then varies and must be perceived). Units: metres, radians, seconds, quaternions `wxyz`.

## 1. Cell frame and bench

The cell frame is the bench frame: the bench top is the plane z = 0, x runs along the bench (1.6 m, x in
[-0.8, 0.8]), y across it (1.0 m, y in [-0.5, 0.5]), z up. The bench top is a wooden plate 4 cm thick on legs; the
floor is 0.75 m below it. Nothing else is on the bench but what this manual lists.

## 2. Robots

Two Franka Emika Panda arms (MuJoCo Menagerie model, `~/given/panda/`) stand ON the bench top on square pedestals
(0.20 x 0.20 m, flush with the bench top). Base poses are exact:

| arm | base position (x, y, z) | base yaw (about z) | tool |
|---|---|---|---|
| `left` | (-0.05, 0.30, 0.0) | -pi/2 (its +x faces -y, the bench centre) | Franka parallel-jaw hand |
| `right` | (-0.55, -0.30, 0.0) | +pi/2 (faces +y) | lacing hook |

Home posture (joints 1-7): left (0, -0.40, 0, -2.43, 0, 2.03, -0.785), right (0, -0.40, 0, -2.43, 0, 2.03, 2.356):
elbow up, tool pointing down, tcp about 0.35 m above the bench, 0.42 m in front of the base. Joint ranges are the
Panda's (see `describe()`). Every arm link is gravity-compensated; the seven joints are position-controlled
(`control`: joint mode, rate-limited at 1.5 rad/s; Cartesian mode, one damped-least-squares differential IK step per
10 ms tick on the commanded configuration toward the target pose, <= 0.5 m/s, 1.5 rad/s; `stiffness_scale` in
[0.2, 1] scales the joint gains). The Cartesian controller is local: a target that needs a wrist flip can stall at a
joint limit; solve such targets yourself (the model is given) and travel in joint mode.

**The gripper (`left`).** The tool site `left/tcp` is the midpoint between the finger pads, 0.1034 m from the hand
body's origin along the hand's z axis: +z = the approach direction (out of the fingers), +y = the finger closing axis,
+x across the pads. The fingertips reach about 10 mm beyond the tcp along +z; the hand body is about 60 mm behind the
tcp. `command(... width=w)` ramps the opening to w in [0, 0.08] at 0.05 m/s; the actuator is a position controller on
the fingers (<= 60 N).

**The hook (`right`).** A lacing hook on the flange: a straight shaft (radius 2.5 mm, length 0.12 m along the tool
+z = the approach direction), an L-foot (radius 2 mm, 14 mm long along the tool +x at the shaft end) and an upturned
tip (radius 2 mm, 12 mm long back along the tool -z at the foot end) with a clip. `right/tcp` is the shaft end
(the foot's root); `right/hook_tip` is the tip END, its +z along the tip (= tool -z). With the shaft vertical the tip
points up and the foot lies horizontal 14 mm from the shaft.

## 3. Cameras (exact calibration)

Five RGB-D cameras; pixel (u, v) = (column, row), row 0 at the top; OpenCV camera convention (x right, y down,
z forward); depth = planar z along the optical axis in metres (the 16-bit PNG holds millimetres, 0 = no reading; the
sensor adds 1.5 mm of noise, drops 3 % of the pixels, and reads nothing below 0.15 m or beyond 3 m; the colour image
carries mild noise). Intrinsics for integer pixel indices: cx = W/2 - 0.5, cy = H/2 - 0.5, fx = fy = (H/2)/tan(fovy/2).
The colour is a physically based render (MuJoCo's Filament renderer) of the cell in the factory hall it stands in, lit
by an LED panel 2.05 m above the bench and the hall's lamps. Around the cell the cameras see the hall's concrete floor,
its walls and lamps and, off the bench, the line's other equipment: an outfeed conveyor from x = 1.25 (a steel plate
bridges the gap from the bench end at the bench top's height, below it), a wire fence with shelving behind the cell,
cabinets, a table with boxes, carts and pallets. None of it is part of the cell, nothing of it stands on the bench, it
never moves, and the depth image reads it like any other surface.

| camera | size | fovy | fx = fy | mount |
|---|---|---|---|---|
| `overhead` | 640 x 480 | 58 deg | 432.97 | fixed at (0, 0, 1.30) looking straight down; image right = +x, image up = +y |
| `front` | 640 x 480 | 52 deg | 492.07 | fixed at (0.10, -1.10, 0.70) looking at (-0.05, 0, 0.05), image up = world +z |
| `side` | 640 x 480 | 52 deg | 492.07 | fixed at (-1.25, 0.05, 0.65) looking at (-0.30, 0, 0.05), image up = world +z |
| `wrist_left` | 480 x 360 | 70 deg | 257.07 | on the body `left/hand`: T_body_cam below; looks along the approach axis 0.15 m ahead of the tcp |
| `wrist_right` | 480 x 360 | 70 deg | 257.07 | on the body `right/hook` (the tool body at the flange) |

`describe()["cameras"]` gives every T_world_cam (fixed cameras) and T_body_cam (wrist cameras) as 4 x 4 matrices with
full precision; a wrist camera's world pose is T_world_body @ T_body_cam, where the body pose follows from the arm's
joint angles (exact in `get_state()`) through the given robot model. Fixed camera poses (OpenCV frames):

```
overhead: R = diag(1, -1, -1), t = (0, 0, 1.30)
front:    T_world_cam = [[ 0.99083  0.06827 -0.11660  0.10]
                        [ 0.13511 -0.50063  0.85505 -1.10]
                        [ 0.00000 -0.86297 -0.50526  0.70]]
side:     T_world_cam = [[-0.05256 -0.53273  0.84465 -1.25]
                        [-0.99862  0.02804 -0.04446  0.05]
                        [ 0.00000 -0.84582 -0.53347  0.65]]
wrist_left  (in left/hand):  R = [[-0.98058 0 -0.19612] [0 -1 0] [-0.19612 0 0.98058]], t = (0.04, 0, 0.0534)
wrist_right (in right/hook): R = [[0 -1 0] [0.98058 0 -0.19612] [0.19612 0 0.98058]], t = (0, 0.04, 0.07)
```

Frames take the cell 0.1-0.4 s each, which is the cell's time, not your program's (section 7); they count against
the episode's frame budget.

## 4. Zones (nominal; the objects on them vary per pair)

| zone | where | what |
|---|---|---|
| infeed tray | a grey plate 0.48 x 0.48 m, 6 mm thick, centred at (0.40, 0.26), with a low rim | the two shoes of the pair (centres near x = 0.40 -/+ 0.115, y = 0.36, lying along x either way round, +-0.25 rad; a shoe may lie on its side; each lace's tails lie folded on the plate beside and behind its shoe) and the lid tray (near (0.40, 0.135), either way round) |
| blank spot | (0.30, -0.22) +- 1.5 cm, yaw 0 or pi (+-0.15 rad) | the flat box blank, inner (light) face up |
| cradle | nominal centre (-0.36, 0.05), yaw -pi/2 (the seated shoe's toe points along -y, toward the hook arm, so its throat slot runs along y); per episode jittered by +-2 cm and +-0.12 rad | the lacing seat: a plate 8 mm thick sized to the batch's sole with 4 mm of clearance, 12 mm corner bumpers and a heel stop at its -x end |
| outfeed line | the yellow line at x = 0.55 | a closed box whose floor centre crosses it is shipped |

## 5. The objects (per batch: one shoe model, one box size; per pair: colours and poses)

**Shoe.** A lasted-open children's sneaker modelled as one rigid hollow shell (`describe()["objects"]["shoe"]` gives
the batch's length L (0.17-0.21), outer width W (0.07-0.08), rows, row spacing s (16-19 mm) and throat gap g
(24-28 mm)). Shoe frame: heel at -x, toe at +x, y across, z up, origin at the sole-bottom centre. Sole 15 mm thick;
side walls 6 mm; the cavity floor (insole) at z = 0.017; two horizontal eyelet flaps (3 mm plates) whose TOP is at
z_f = 0.063, spanning from each side wall inward to the throat slot (|y| < g/2) that runs between the toe cap (the
front 45 mm, top at 0.078) and the heel counter (the back 25 mm, top at 0.088); the cavity under the flaps is open to
the throat slot. Two eyelet rows: ring eyelets (hole radius 4.5 mm, wire radius 1.2 mm, outer radius 7.4 mm,
metallic) sitting on the flap tops at |y| = g/2 + 9 mm, at x_k = (L/2 - 0.045) - 0.012 - k s, k = 0..1; row 1 (k = 0)
is nearest the toe; side L is the shoe's +y side. Mass 0.22 kg. Colours per pair (upper, sole). Grasp handles: the
heel counter top (pinch 25 mm across the counter, along the shoe's x) and the toe cap top (pinch 45 mm). In the colour
images the shell wears a sneaker's skin: a rubber cupsole in the sole colour, knit and leather panels in the upper
colour with a white side wing, a dark lining, metal grommets at the two eyelet rows (one per ring, round its hole) and
stitching; the skin follows the shell within a few millimetres (rounded where the shell has corners), and the depth
image is the shell's.

**Lace.** A cord 0.64 m long, radius 1.5 mm, with two metallic aglets (20 mm, radius 1.8 mm), per pair's colour. It
slides frictionlessly over the shell, the bench, the tray and the fingers, and a loose lace comes to rest within about
a second (it is lightly damped); the eyelet rings have friction (a lace through a ring does not slide out by itself).
Delivered threaded through row 1 and tacked there: the lace crosses the cavity under the flaps, is stitched to the
insole between the two row-1 rings (pulling an end never slides the lace through row 1; each tail keeps its length,
about 0.27 m from its ring), and both ends leave the shoe outward through the row-1 rings; the tails hang down beside
the shoe and lie folded on the tray plate alongside and behind it, laid clear of the other shoe and of the lid tray
(the frictionless cord may still slide a little before it rests: rarely an aglet ends up under a wall of the lid
tray).

**Box blank.** One die-cut piece: a floor L_b x W_b with four walls H high on creases (3 mm cardboard; `describe()`
gives L_b, W_b, H per batch: L_b fits the shoe length + 3.5-5 cm, W_b two shoe widths + 1.2-2 cm, H 9.4-10.4 cm,
taller than a shoe, so the lid rests on the wall tops),
delivered flat, inner (light) face up, brown outside. Walls fold up toward the inner face; a wall held at >= 87 deg
for 0.3 s snap-locks at 90 deg (the lock never releases); a wall let go before locking falls flat again. Each wall
has three grasp handles 12 mm in from its free edge (centre and +-30 % along the edge; pinch 3 mm).

**Lid.** A rigid tray (3 mm cardboard; top L_b + 2 t + 2 x 4 mm + 2 t by W_b likewise, walls 25 mm deep hanging down,
inner clearance 4 mm per side over the erected box), delivered printed side (a coloured logo) up on the infeed tray.
Grasp handles: the middle of each wall's top edge (pinch 3 mm).

## 6. Attach rules (how the tools hold things)

* **Gripper.** When the fingers close and a documented grasp handle lies inside the pinch box of the tcp frame
  (|x| <= 12 mm across the pads, |y| <= half the current opening along the closing axis, z in [-10, +12] mm along the
  approach) the object is attached to the hand at its current pose, and the fingers stop at the handle's pinch
  width (that is the gripper's only grasp signal, plus `holding` in `get_state()`). An aglet is the exception: it is
  drawn into a seat along the approach axis over 0.3 s, its centre 14 mm beyond the tcp and its tip 24 mm beyond. Opening the fingers
  releases. Nothing else can be grasped. Closing the fingers on an aglet that sits on the hook takes it off the hook
  (the clip opens by itself).
* **Hook.** `hook(True)` closes the clip: the nearest free aglet whose centre is within 12 mm of the tip segment is
  drawn into a seat along the tip over 0.3 s, its centre 8 mm beyond the tip end, pointing away from the tip. `hook(False)` releases.
  `get_state()["arms"]["right"]["hook_loaded"]` is the clip's sensor. The hook also pushes like any rigid tool.
* **Aglet pull.** The pinch and the clip hold an aglet against at most 6 N of pull: a held aglet pulled harder than
  that for 30 ms slips out (the `step` event `aglet_slipped:left` or `aglet_slipped:right`; `holding` / `hook_loaded`
  turn false). Pulling a lace snug takes far less.
* **Contacts.** An object held by the gripper does not touch the hand that holds it but everything else (a held shoe
  keeps its lace through row 1); a held aglet touches neither its own lace nor the hand. The hook's shaft, foot and
  tip pass through lace cord (only the clip interacts with a lace); fingers, shell, rings, bench and tray push lace
  cord like any solid. Lace never collides with lace.
* **Cradle clamp.** A shoe resting sole-down, heel toward the heel stop, within 6 mm and 4 deg of the seat for 0.3 s
  (and not held by the gripper) is clamped; `release_clamp()` frees it. `clamp_engaged` in `get_state()` is the
  clamp's sensor.
* **Carton locks** as in section 5. **Lid closed** = the lid tray sits over the erected walls: centred within 12 mm,
  aligned within 6 deg (either way round), at wall-top height, resting; the `closed` credit also needs both shoes in
  the box as packing does (inside, sole down within 15 deg, not resting on a wall; heel-to-toe is packing's alone).
  **Shipped** = closed so, with both shoes inside, and the floor centre across the outfeed line; only then does the
  cell take the box and the pair away (a box that crosses the line otherwise stays where it is), which completes the
  episode's work order (one pair per episode).

## 7. Safety and budgets

A Panda link other than the finger pads and the hook tool pressing more than 30 N on the bench, the cradle or any
object, or any contact between the two arms (links, pads, hook) above 30 N, for 30 ms (three control ticks) in a row,
or any shoe, the lid or the box faster than 3 m/s, is a safety violation: the pair being worked on when it happens earns no credit (even if it ships). A physics
fault (the simulation diverging under a violent
action: the step event `physics_fault`) ends the episode: pairs already shipped keep their credit, nothing after the
fault counts. Budgets (`describe()["budgets"]`): 900 s of simulated time; 1200 s of program time (the episode's
wall clock minus the cell's own time inside `step`, simulating, and inside `get_camera`, rendering: your program's
computation and all its other requests count); 120 s from a reply to the next request; 500 ticks per `step`; 450 camera
frames per episode.

## 8. What is not given

No object pose, no eyelet or aglet position, no lace state, no lock state, no lid state, no milestone or credit is
ever on the wire. The practice cell's `truth()` exists for development only.
