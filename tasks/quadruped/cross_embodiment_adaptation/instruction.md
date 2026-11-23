# Cross-Embodiment Adaptation: survey the loading line for a loaded Go1

Write an unattended program that surveys an unknown warehouse floor from a UAV, then walks a loaded Unitree Go1
to the blue dock on its first deployment there. The program transfers the route from its own survey to the Go1.
The Go1 has no target map, dock sensor, height command, walking primitive or balance controller: your program
commands its twelve joints.

Three things make this a job for a program that senses and reacts on site, not for a plan prepared in advance:

- **The survey is out of date and not to scale.** Between the survey and the deployment the site can change: a
  pallet moved from one opening of the middle rack to the other, the dock moved, a bollard taken away or
  added. The UAV's own tracking over- or under-reads its displacement by a factor that differs per site.
- **The Go1 does not know where it is.** Its leg odometry drifts in heading and mis-scales distance, its joint
  encoders and its IMU have noise and bias. It has to find itself again, on its way, from what its mast camera
  shows.
- **Each site's Go1 and fittings differ** within stated ranges: the load it carries and where the load sits, the
  stiffness and strength of its joints, the floor's friction, how low the duct hangs, how high the deck's guides
  rise.

None of these values is reported. Every range is stated below and in the datasheet. `/home/user/robot/go1_datasheet.md` gives the
robot's parameters and every range.

## Session time

You have **2 hours** from the start of this session to leave your program; the session is ended there, and
whatever is in `/home/user/submission/` at that moment is graded. `bash /home/user/session_time.sh` prints the
minutes used and left (it reads a start stamp written when the session began). Check it instead of estimating,
especially after a long tool call or when your context has been summarised. Make your last change at least
10 minutes before the end.

## Mission

The warehouse workcell is about 8 m by 4 m. The Go1 starts somewhere in the western staging bay. A blue dock
with a yellow east edge is near the east end: a mat 0.58 m by 0.56 m and 4 mm thick, its yellow strip 6 mm
high, both walked on like the floor. Three transverse rack lines separate four bays. The first and
last rack lines each have a near-centre shortcut and a side passage. Both are real openings, but a clear
centreline does not mean the Go1's trunk, hips and stepping feet fit. Determine which openings are usable by
measuring the scene at each rack; the side passage can be north or south independently at the two racks. Leave
room to align before entering a passage and to clear the rack before changing direction. Passage positions,
widths and the dock's centre vary. The first frame does not show the openings. The orange caps mark actual
collision footprints; they do not encode passability.

**What the cameras show.** The workcell stands on the worn concrete floor of a warehouse hall, and both cameras
show the hall as it is. The barriers are steel with orange top caps and striped ends at the openings; the
bollards are yellow and black posts; the dock is a blue mat. Outside the barriers stand tall racks loaded with
cardboard boxes and wooden pallets (blue uprights, orange beams), a forklift, a reach truck, a table, carts and
a charging station. Overhead are a dark roof, grey ducts, pipes, cable trays and rows of white strip lamps. The mast
camera looks up, so the racks, the roof and the lamps fill most of its images, behind and around the caps it is
meant to find. Nothing outside the barriers can be reached, and nothing there is a cue. The cues are the colours
this instruction names: orange barrier caps, green bollard caps, the blue beacon cap, the magenta pallet, teal
covers, the blue dock.

The middle rack has two openings, one north and one south of the centre, each 0.84 to 0.92 m wide. A **pallet**
(magenta plastic, 1.20 m by 0.80 m by 0.144 m, stood on its short edge so that it is 0.80 m wide and 1.20 m tall)
stands inside one of them, in line with the rack, and closes it. A steel foot bracket holds it near each end: a
floor plate 0.22 m by 0.10 m and 8 mm thick that reaches 0.04 m beyond each face of the pallet, with an upright
0.08 m tall against each face. Touching the pallet or a bracket ends the mission.

**What can have changed when the Go1 arrives** (each on about half of the sites, independently; a site can
have none or several):

- the pallet stands in the middle rack's other opening, so the opening that was free in the survey is closed
  and the other is free;
- the dock was moved 0.10 to 0.25 m west and 0.10 to 0.25 m north or south. The **dock beacon**, a post like
  the bollards with a blue cap (radius 0.105 m, centre 0.72 m above the floor), always stands 0.42 m east and
  0.50 m north of the dock's centre and moved with it. The mission is judged on the dock where it is;
- one bollard was taken away, or one was added on a rail.

Nothing else changes: racks, fittings, the other bollards and the bay stay as surveyed.

One wide opening contains a suspended loading duct, the other a raised loading deck; their order varies.

- **Duct.** 0.44 m long along X. Its underside is between 0.410 and 0.445 m above the floor (per site) and its
  sheet assembly is 0.068 m thick.
- **Deck.** A 0.62 m level deck 0.065 m above the floor, with a 0.33 m sloping apron at each end ending in a
  0.012 m toe. It has two longitudinal wheel tracks, each 0.150 m wide, centred 0.125 m to either side of the
  passage's centre. Between and outside the tracks, raised guides climb above the deck to a crest between 0.070
  and 0.112 m (per site). A guide starts at the deck's west edge, reaches its crest 0.061 m east of the deck's
  centre, holds it for 0.188 m and falls to the deck within the next 0.056 m.
- Both fittings span their passage. The duct's roof and the deck's centre guide carry the same 0.160 x 0.078 m
  teal service cover, 1.4 mm proud of the surface (on the deck, 0.155 m east of the deck's centre). The covers are
  not colour-coded: separated UAV views of a fixed cover give its height by parallax, and that height tells the
  fitting and its clearance. A nadir image does not see through a roof.

Only the UAV uses the survey frame: origin at the UAV's first camera centre projected onto the floor, +X east
and +Y north. The Go1 is absent during the survey, and the UAV is not in its own images. During the deployment
the UAV hovers where the survey ended, 1.77 m above the floor, and can be in a mast image. It deploys with its own launch position and yaw; there is no
common tracker, compass, launch mark or target pose on the map. To transfer a route, use the same fixed objects
as seen from above and from the Go1. Steel bollards with identical green spherical safety caps (radius 0.105 m,
centres 0.72 m above the floor) stand around the staging bay and along both long walls of the hall, eleven in
the survey. Their arrangement varies; the caps carry no identifiers.

## Phase 0: UAV

- `observe()` returns one 256 x 256 nadir RGB PNG in `images` and `tracking_xy_m: [x, y]`. It does not step
  time. The first image of a site takes about ten seconds (the site's renderer starts); later ones about 0.1 s. `tracking_xy_m` is the UAV's own tracking: its true displacement times a factor between 0.97 and 1.03
  (per site, the same in x and y). The camera is exact, so two frames of the same object show the factor. The pinhole camera has a 78-degree vertical field of view and its centre is 1.65 m above the ground.
  Image right is +X; image up is +Y. Barrier tops are 0.32 m above the floor. There is no full-site image,
  semantic mask or depth output.
- `act([dx, dy])` moves the idealized survey UAV at fixed altitude. The displacement's norm cannot exceed
  0.60 m (true metres); the allowed rectangle is `[-0.75, 6.80] x [-1.55, 1.55]` (true metres), and a move
  clips at its edge. The
  action returns a new camera observation. There are 80 moves, counted by `scout_actions_remaining`.
- In grading, call `advance()` once after at least four scout actions. It retires the UAV for good, deploys
  the Go1 and returns its first observation.

## Phase 1: Go1

Send `act({"target_rad": [12 angles], "ticks": n})`, where `n` is an integer from 1 to 40, not exceeding
`ticks_remaining`. The position servos hold those targets for `n` physics steps of 2 ms (or until the mission
ends). Out-of-range or non-finite commands are rejected. The order is FR, FL, RR, RL, each hip (abduction about
+X), thigh (about +Y), calf (about +Y). In the trunk frame +X is forward, +Y left and +Z up.

| Joint | Target range (radians) |
|---|---|
| Hip | [-0.863, 0.863] |
| Thigh | [-0.686, 4.501] |
| Calf | [-2.818, -0.888] |

The hip pivots sit at x = +/-0.1881 m (front/rear) and y = +/-0.04675 m (left/right). The thigh axis is a
further 0.08 m outward, so opposite thigh axes are about 0.254 m apart at zero hip abduction. Thigh and shank
are each 0.213 m along local -Z at zero angle. The home targets are `[0, 0.9, -1.8]` per leg; the Go1 starts
settled at them. The trunk has six free degrees of freedom. There are no hidden force, yaw, touchdown or
foot-placement commands. Masses, servo stiffness and damping, torque limits and friction are in the datasheet.

The Go1 carries inspection equipment, rigidly mounted (trunk coordinates):

- a **3D lidar** on a mast above the trunk origin (an Ouster OS1-class head, a cylinder 0.087 m in diameter and
  0.0745 m tall, on the trunk's Z axis) inside a **tubular guard frame**: a rectangle of 12 mm tube that occupies
  X [-0.102, 0.102], Y [-0.085, 0.085] with its top at Z +0.199 m, on four corner posts. The frame's top is the
  highest point of the equipment. Under the lidar are a 0.024 m mast and a compute and IMU unit
  (0.100 x 0.080 x 0.030 m). The mast camera sits just above the frame;
- a **floor profiler** under the nose, mounted crosswise: a line-profiler housing (an LMI Gocator 2340-class unit,
  0.049 x 0.197 x 0.075 m) that occupies X [0.341, 0.390], Y [-0.0985, 0.0985], Z [-0.163, -0.088] m, on a bracket
  from the trunk. Its underside is the lowest point of the equipment;
- an **instrument pack** on a rail along the trunk. Equipment and pack together weigh between 2.50 and 4.50 kg,
  and the pack sits between 0.05 and 0.28 m ahead of the trunk origin (both per site).

A heavier or more nose-ward pack makes the legs sag and the trunk pitch under the same joint targets, so the
guard frame and the profiler are not where a nominal posture would put them. Keep the frame below the duct and
the profiler off the floor, the aprons, the deck and the guides.

Go1 `observe()`, `advance()` and `act()` return `odom_xy_m` and `odom_yaw_rad`: the leg odometry's planar
displacement (rounded to 1 mm) and counter-clockwise yaw change (wrapped to [-pi, pi]) relative to the
**settled launch trunk frame**, not to the survey. Both start at zero; initial forward is +x, initial left is
+y. **The odometry drifts:** its heading gains a constant error rate between -0.80 and +0.80 degrees per second
(per site, also while standing), and it reports each step's length times a factor between 0.94 and 1.06 (per
site), laid along its own drifting heading. Over a mission that is tens of degrees and tens of centimetres:
dead reckoning from the launch pose does not reach the dock. The other readings are `gravity_body_unit`
(noise 0.01 per axis), `gyro_body_rad_s` (noise 0.01 rad/s, a constant bias within 0.015 rad/s per axis),
`joint_rad` (noise 0.002 rad, a constant bias within 0.008 rad per joint), `joint_rad_s`, `foot_contact` in leg
order and `ticks_remaining`. There is no range, absolute heading, absolute position, depth or joint-torque
reading. `foot_contact` includes contact with the floor, an apron, the deck or a guide.

`look(detent)` returns one RGB PNG in `images`, plus `requests_remaining`, with **no physics step**. It does
not turn the Go1. An ideal pan mount shares one optical centre 0.20 m above the trunk origin along trunk +Z.
Integer detents 0..7 pan counter-clockwise by `detent * 45` degrees from trunk +X toward trunk +Y. At each
detent the camera points **36 degrees up** from the trunk's horizontal plane; it tilts with the trunk. Image
right is clockwise from the pan direction, image up is toward higher elevation. The 256 x 256 undistorted
pinhole has a 70-degree vertical field of view, principal point (127.5, 127.5) in zero-based pixel-centre
coordinates, and square pixels. The camera sees what stands above its own height: the bollards' green caps,
the dock beacon's blue cap, the pallet, the duct, and behind them the hall's racks, roof and lamps. Each
deployment allows 24 exposures. Regular Go1
observations carry no image.
Observations in both phases include `phase` (0/1), `done` and `requests_remaining`.

**Budgets per site:** 48,000 physics ticks (96 s of robot time) and 4,000 requests.

## What ends a mission, and what scores

A mission scores only if it ends **safely docked**: the trunk origin within 0.17 m of the dock's centre, the
heading within 16 degrees of survey +X, upright (trunk origin above 0.16 m and body-up dot world-up above
0.80), below 0.09 m/s linear and 0.35 rad/s angular speed, for 0.45 s continuously. Every other ending scores
nothing for that site:

- any part of the robot or its equipment touching a barrier, a bollard, the dock beacon, the pallet, one of
  its foot brackets or the duct;
- the equipment (lidar, guard frame, mast, compute unit, profiler or its bracket) touching the floor, the dock
  mat, an apron, the deck or a guide (feet and legs may touch them);
- a fall: 0.20 s continuously not upright;
- the tick or request budget spent, the program exiting before the dock, the program's own computing time
  passing 600 s (the service's time is not counted) or the mission's wall time passing 1,800 s.

No gait, manoeuvre, route or waypoint is prescribed. The score is the number of sites safely docked, out of
sixteen.

## Development: what you can commission

The service in this session (`/run/visual_robotics/world.sock`, the client `/home/user/robot/client.py`, the
same calls as in grading, and the MCP tools) offers two things. Neither gives ground truth, a score or a
debrief, and the service's code and state are not readable.

- **Survey sites.** `new_site()` starts another random site with its UAV. `advance()` is refused here: a
  development site has no Go1.
- **The commissioning pad.** `pad()` starts a pad episode: the loaded Go1 in a bare bay of the same hall, inside
  the bay's four boundary barriers and among bollards, with
  the readings (noise, biases and odometry drift included), the mast camera and `act` exactly as on a site, and
  its own draw of every per-site property from the same ranges (the values are not reported). A pallet on
  edge and a dock beacon stand somewhere on the pad, 1.6 to 2.6 m from the launch point, so that seeing them
  can be tried. There is no rack, fitting or dock on the pad. An episode
  ends on a fall, on touching a barrier, a bollard, the pallet or the beacon, or on its tick budget; start the
  next with `pad()`. The pad's budget for the
  whole session is **40 episodes and 600,000 physics ticks** (20 minutes of robot time); replies carry
  `pad_episodes_remaining` and `pad_ticks_remaining`.

You may also build and use any model or simulation of your own; `/opt/venv/bin/python3` has NumPy, SciPy,
Pillow and MuJoCo.

## Submission and grading

- Leave `/home/user/submission/run.sh` and its dependencies. Only regular files transfer (at most 2,000 files
  and 64 MB, no symlinks or `__pycache__`). A missing `run.sh` scores zero.
- A fresh, separate verifier launches `run.sh` once per hidden site from a read-only copy, as your account,
  with a fresh HOME and TMPDIR and no network, on 2 CPUs and 4 GB without a GPU. Use the Unix socket in
  `$ROBOT_SOCKET` (default `/run/visual_robotics/world.sock`) through `/opt/visual_robotics/client/client.py`;
  its `camera_image(obs)` decodes either camera's RGB.
- Every site is a first deployment. Nothing carries over from one site to the next: after each mission the
  verifier kills your account's processes, removes every file your account left anywhere (and its shared
  memory, semaphores and message queues) and restores any shared file it changed. If a process cannot be
  stopped or a changed file cannot be restored, the whole submission scores zero.
- `/opt/venv` Pythons run NumPy without its AVX-512 code paths and with one BLAS thread, here and in the
  verifier, so that a program computes the same on every machine.
- Do not read the service's files, guess site seeds or hard-code site maps.
