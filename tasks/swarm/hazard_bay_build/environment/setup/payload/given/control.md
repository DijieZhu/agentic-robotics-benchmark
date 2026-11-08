# The bay mission: your fleet's program

The second half of the submission is the program that runs the fleet through the plant's whole job (`bay_brief.md`):
close the leaking line's valve, put the fire out with the bucket, clear the assembly station into the crate. You write
the control for every robot and the planning that sequences them; how you do it is yours. After the scene checks, the
grader runs your program in **the real bay**, twice, on two days of its own (What varies per run): its own stage of the
bay as it really is, drawn for your robots' sensors as the photographs show it, with your four robots in it exactly as
your scene builds them (The real bay, below). What the robots sense is rendered images and scans from the sensors
your scene declares, and each robot's own state (The sensors; What you observe). It measures only what happens:
whether the steam is released, the fire out, the parts in the crate, how long it took, how steadily the robots moved,
what was spilt, lost or hit, and how alike the two runs were.

Your program runs as a separate process, launched as `bash run.sh` in a copy of your submission folder, and talks over
its stdin / stdout: one JSON object per line (The wire). Nothing in `~/given` runs the mission: you rehearse it in your
own twin, with a harness of your own (Rehearsing, at the end). This page says what the grader simulates and measures.

## The mission

At the start the flying robots stand on their pads, the ropes rigged (slack, lying on the floor), the bucket full beside
the tank, the fire burning, the steam leaking, the vehicle in its bay, the six parts somewhere on the two tables. The
jobs:

- **The steam**: turn the valve's lever, from wherever it was left, to its closed stop. The line's pressure then falls
  to nothing over 3 s and the leak stops (the real valve's behaviour). The vehicle's arm works the lever by pushing or
  pulling it; nothing turns it for you. The valve's packing holds the lever wherever it is: its own weight does not
  turn it, a push does.
- **The fire**: lift the bucket on its three ropes, carry it to the pan, tip it over the pan. Once 1 litre has been
  poured on the pan the fire is out (The water).
- **The station**: every part off the tables and into the crate.

**The valve, as the grader keeps it.** The lever's angle is counted from its open stop, where it points straight up;
it closes by turning toward +x (to the right, seen from the bay), a quarter turn to its closed stop. It counts as closed
from 85° on, and once closed it counts as open again only below 75°. It turns only while a body of a robot touches it
(a part the tool holds does not count) and for 0.5 s after the last touch, its own weight cancelled; the rest of the
time it stays exactly where it is. The line's pressure is 8 bar while the lever counts as open; from the moment it
counts as closed the pressure falls linearly to 0 over 3 s, and the steam is on as long as the pressure is above 0;
opened again, the line is back at 8 bar at once.

Time limit 600 s of simulation. The mission ends "done" when all three jobs are done (no pressure, the fire out, all six
parts in the crate) and everything has stayed still for 3 s: every flying robot slower than 0.3 m/s, the bucket hanging
or standing within 8° of upright, the vehicle slower than 0.1 m/s. The grader checks this every 0.5 s on the true
state; the done time is when the still period began. The mission also ends at the time limit, or when every robot is
lost.

**The bay's rules, as the grader keeps them**: a flying robot whose true position (its root body's origin, What you
observe) stays inside the steam plume's box for 1 s in a row while the steam is on is lost; the vehicle whose true
position stays within 2 m (horizontally) of the pan's centre, where the pan stands that day, for 1 s in a row while the
fire burns is lost. A lost robot is out for the rest of the run: a flying robot's thrust is cut (the wind still pushes
it); the vehicle's velocity targets go to 0, its position targets stay as they were and its grip lets go. Your scene's
hazard logic does not come into the real bay (the scene checks test it).

**The start.** Before your program starts, the grader sets the day up (What varies per run): the pan moved, the vehicle
moved and turned, the lever turned, each part set down 1 mm above its table's top. Then it runs 1 s of physics with
nothing commanded (no thrust, no wind, the vehicle's drives on your scene's own targets) while the parts and the ropes
settle, and only then starts the clock: t = 0, every contact so far forgotten, `mission.uav.*.start` and the vehicle's
odometry taken there. Only then is your program started.

## The wire

```
READY                                       <- within 40 s of launch
{"type": "reset"}                           -> {"ok": true}
{"type": "act", "t": 12, "obs": {...}}      -> {"action": {"thrust": {"UAV_1": [fx, fy, fz], ...}, "ugv": {...}}}
{"type": "close"}                           -> {"ok": true}, then exit
```

- **Starting.** The grader starts your program once the run is set up: `bash run.sh`, as your account, in a session of
  its own, with exactly this environment and nothing else: `PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin`,
  `HOME=/home/user`, `LANG=C.UTF-8`. Call `/opt/venv/bin/python3` explicitly (NumPy, OpenCV, PyTorch and `pxr` are
  there); `run.sh` may start anything that speaks the wire. Its stdin and stdout are the wire; its stderr goes nowhere.
- **The working directory** is a fresh copy of your submission folder (your scene included) that belongs to root: your
  program can read everything in it and write nothing, and no file in it keeps an executable bit, so start everything
  through an interpreter (`exec /opt/venv/bin/python3 controller.py`, `bash helper.sh`; not `./helper`; with `-I` Python
  leaves the script's own folder off its import path). Write logs
  and scratch files elsewhere, under `/tmp`. The submission may hold only regular files and directories: a symbolic
  link or any other kind of file anywhere in it, or more than 2 GiB in all, and the whole submission scores 0.
- **No network.** When it is graded your program has no network: it cannot open an internet socket, not even one to
  localhost; only Unix-domain sockets and pipes work.
- **`READY`, then `reset`.** The first line that is exactly `READY` must come within 40 s of launch (up to 199 other
  lines before it are skipped). The grader then sends `{"type": "reset"}` at once; any JSON object answers it.
- **stdout carries replies only.** Each request has one deadline, 10 s of wall time from when it is written, and after
  `READY` every line your program prints is read as the reply to the request in hand (blank lines are skipped). A line
  that is not a JSON object, or an `act` reply without an `action` (`"action": null` included), aborts the run; a
  second line for one request is taken as the next request's reply. Print your logs to stderr or to a file.
- **Aborts.** A missed deadline, a crash, a malformed reply or a reply without an action aborts the run: every check of
  that run scores 0. Each run starts `run.sh` afresh, after every process of your account has been stopped, and a run
  that fails on the grader's side before its first step is run once more from its set-up, `run.sh` included. So your
  program must be able to start more than once (what it left under `/tmp` is still there).
- **Your program's time.** Besides the 10 s per request, a run gives your program 30 minutes of its own time: the time
  from each `act` request to its reply, summed over the run. A run whose replies take longer than that in all is
  stopped and counts as aborted. The simulator's own time does not count against your program, and neither does the
  time between a reply and the next request: that time is the simulator's (the physics, the rendering, the frames).
  Your program may work then too, but it shares the machine (8 cores and the GPU) with the simulator. A run that lasts
  its full 600 s (36,000 steps, about one step in twenty asked twice, 6,000 frame sets) leaves your program about 47 ms
  per reply on average, or about 0.27 s per frame set if it answers the other steps in 5 ms. (A run is also stopped, as
  aborted, 210 minutes of wall clock after it started, whatever the time went into; the simulator takes about
  90 minutes for a full run of a fleet that streams all the sensors the cap allows, its frames at their full size.)
- **Steps.** Control steps come at 60 Hz of simulation time (two physics steps of 1/120 s each). `t` counts them from
  0; `obs.time` is the simulation clock. A step's observation is taken before its action is applied (Rehearsing).
- **A step may be asked more than once with the same `t`.** About one step in twenty is asked twice, in an order
  nobody tells you: once with its observation and once with a copy in which every flying robot's `pos` is shifted by up
  to 0.4 m along each axis and the vehicle's `pose` by up to 0.4 m along x and along y (each robot and axis by its own
  amount; the heading, the frames and everything else as in the observation: a repeated request names the same
  frames). Only the answer to the true observation is applied,
  and the shift does not carry into later steps. Answer each request from the observation it carries and from your
  state as it was before that `t`; count steps by `t`, not by requests, and move your state on once per `t` (the next
  step's observation continues the true one).
- **The end.** No message announces the mission's end: when the run ends (done, at the time limit, every robot lost, or
  aborted), the grader sends `{"type": "close"}`, closes your stdin, waits up to 10 s, then stops your process group and
  every process of your account. Exit on `close` or when stdin ends.

## What you command

```
{"thrust": {"UAV_1": [fx, fy, fz], "UAV_2": [...], "UAV_3": [...]},
 "ugv": {"vel": {"<joint>": rad_per_s, ...}, "pos": {"<joint>": rad, ...}, "grip": true | false}}
```

- `thrust`: one force vector per flying robot in newtons, world axes, applied to its root body (What you observe) in
  both physics steps of the control step, together with the wind. The grader clamps it in this order: F_z to ≥ 0; the
  horizontal part to F_z · tan 40° (a 40° tilt of the thrust vector from vertical, the attitude limit); then the whole
  vector to the airframe's `thrust max` (`catalog.json`, `mission.uav.*.thrust_max`; 0 when the robot's
  `arb:airframe_model` names no catalog airframe). The flying robots' attitude is held by a perfect inner loop: after
  every physics step the robot's orientation is set back to the one it had at t = 0 (as it stood on its pad: its
  heading never changes) and its angular velocity to zero; its position is pure dynamics: your thrust, gravity, the
  rope, contacts, the wind. A robot without a thrust entry gets zero. The flying robots' arms are not on the wire:
  their drives keep the targets your scene authors.
- `ugv.vel`, `ugv.pos`: velocity and position targets for the vehicle's joints, by the joint names of your scene (the
  `mission` lists them; names must be unique within the vehicle: a target goes to the first joint of that name),
  applied to the joints' drives as your scene authored them (a drive with stiffness follows a position target, one with
  damping only a velocity target). Velocity targets are clamped to ±40 rad/s, position targets to the joint's limits;
  an unknown name, or a value that is not a finite number, is ignored. A target keeps its last value until you send
  another. At t = 0 every position target is the joint's position when the mission was set up (before the 1 s settle)
  and every velocity target is 0; from the first step on the grader writes every joint's targets at each step (your
  scene's own targets apply only during the settle).
- `ugv.grip`: the vehicle's suction tool at the origin of the arm's last link (the last rigid body, in the vehicle's
  link order, of those under the prim that declares `arb:arm_model`; the `mission` names it). The grader applies it
  once per control step, before the physics. While it is on and holds nothing, it takes the nearest part whose bounding
  box (axis-aligned, in world axes) comes within 4 cm of that point and holds it rigidly: the part then follows the
  link exactly. Switched off, it lets go, and the part leaves with the link's velocity (without spin). An action without
  `grip` leaves it as it was. It holds parts only (the lever is worked by contact).
- **Shapes.** Send exactly these shapes. An `action` that is not an object (null aside: The wire) counts as an empty
  one, and so does every key an action leaves out: no thrust, the vehicle's targets and grip as they were. `thrust`,
  `ugv`, `ugv.vel` and `ugv.pos` must be objects where they are present, and each thrust entry a list of three finite
  numbers: anything else there counts as zero or aborts the run.

## What you observe (per control step)

```
{"t": 12, "time": 0.2, "mission": {...},
 "uav": {"UAV_1": {"pos": [x, y, z], "vel": [vx, vy, vz], "load": [fx, fy, fz],
                   "joints": {"<joint>": [position, velocity], ...}, "disabled": false}, "UAV_2": {...}, "UAV_3": {...}},
 "ugv": {"pose": [x, y, heading_deg], "vel_body": [forward, left], "yaw_rate": deg_per_s,
         "joints": {"<joint>": [position, velocity], ...}, "grip_point": [x, y, z], "vacuum": true | false,
         "disabled": false},
 "frames": {"n": 1, "t": 6, "time": 0.1, "slot": 1} | null}
```

The observation holds two kinds of thing. **Numbers**: each robot's own state, as its own onboard systems estimate it
(its flight computer, odometry, encoders, vacuum switch), with the noise below (Gaussian, standard deviations, along
each axis); what the table calls true is exact. **Frames**: the raw output of the sensors your scene declares, rendered
in the real bay at the moment of their capture (The sensors); `frames` names the latest set delivered. No field says
where an object is, what state it is in, or which part is which: everything about the bay, the parts, the bucket, the
pan, the fire, the steam and the other robots reaches your program only through the frames, and through its own
robots' force and motion sensing (the rope loads in `load`).

**Positions.** A robot's position is the origin of its articulation's root body: the prim under the robot that carries
`ArticulationRootAPI`, or, when that prim is not a rigid body, the first rigid body below it. The numbers are in world
axes, except `hook` and `grip_point` (in the robot's root body's frame), `ugv.vel_body` (in the vehicle's own frame) and
`ugv.pose` (the vehicle's odometry, in the parking bay's nominal frame); lengths are in metres.

**Sensors that count.** A camera counts only as a `UsdGeom.Camera` with `arb:sensor_model` naming a catalog camera and
that camera's focal length and apertures (within 2 %); its field of view is the rectangle hfov × vfov (from the focal
length and the apertures) about its −Z axis, +Y up. A lidar is any prim with `arb:sensor_model` naming a catalog
lidar. Declarations and sensors inside an instanceable prim are not read (`interface.md`).

| field | what it is | from |
|---|---|---|
| `uav.*.pos` | the robot's position by visual-inertial odometry: the true position plus a slow random walk (0.02 m/√s along each axis, height included) and white noise (1 cm). Indoors there is no GPS: the odometry drifts and nothing resets it. | the flight computer |
| `uav.*.vel` | its velocity, white noise 0.03 m/s | the same odometry |
| `uav.*.load` | the force on the airframe other than its thrust and its weight (the rope, the wind, the ground while it stands on the pad): m·a − T − m·g, with m = `mission.uav.*.mass`, a its true acceleration over the last control step, T the thrust as the grader clamped it for that step (counted even after the robot is lost and its thrust cut), g = (0, 0, −9.81) m/s²; noise 0.4 N. Standing on its pad it reads about (0, 0, +m · 9.81). | IMU + known thrust |
| `uav.*.joints` | every joint of the robot's articulation (its arm's), in PhysX's order (`mission.uav.*.joints` names them): position (rad, or m for a sliding joint; noise 0.001) and velocity (true). A camera on an arm link follows the arm (The sensors): these place it. | the encoders |
| `ugv.pose` | the vehicle's position (x, y) and heading by its own dead reckoning, in the parking bay's nominal frame. At t = 0 it reads the parking bay's nominal pose, `mission.ugv.start` with heading 0, not where the vehicle really stands (What varies per run): find out where it is from what its sensors see. From then on, each step, the heading moves on by the gyro's rate (`yaw_rate`) × 1/60 s, and the position by the step's travel as the wheels measure it in the vehicle's own frame (its true displacement there × (1 + a per-run scale error of up to ±2 %), plus noise 0.02 m × √(the step's distance in m) along each of its axes), turned by the odometry's own heading at mid-step: an error in the heading turns the track. Reported with white noise 5 mm on x and y and none on the heading. The heading is 0 facing +y (as parked), counter-clockwise positive, in degrees, in (−180, 180]. | wheel odometry + gyro |
| `ugv.vel_body` | the vehicle's velocity [forward, left] in its own frame (its parked +y is forward): its root body's true velocity × (1 + the same scale error) + noise 0.02 m/s per component | wheel odometry |
| `ugv.yaw_rate` | the gyro, °/s, counter-clockwise positive: the vehicle's true turn over the last control step ÷ 1/60 s, × (1 + a per-run scale error of up to ±1 %), plus noise 0.387 °/s per step (in the heading: a random walk of 0.05°/√s, which runs on while the vehicle stands still) | the gyro |
| `ugv.joints` | every joint of the vehicle (its wheels included), in PhysX's order: position (rad, or m for a sliding joint; noise 0.001) and velocity (true) | the encoders |
| `ugv.grip_point` | the suction tool's point in the vehicle's own frame (its root body's frame), from the joints, true | forward kinematics |
| `ugv.vacuum` | true while the suction tool holds a part; it does not say which | the vacuum switch |
| `disabled` | lost by the bay's rules (true) | the grader |
| `frames` | the latest frame set delivered (The sensors): its index `n`, its capture step `t` and simulation time, and the `slot` it sits in (`n` mod 2). It repeats in every observation until the next set arrives, every 6 steps from step 6 on (a new `n`); `null` before step 6. | the sensors |

`mission` repeats the fixed facts every step:

```
{"dt": 0.016666666666666666, "time_limit": 600.0, "cruise_z": 2.6, "pour_r": 0.6, "pour_zmax": 2.5, "rope_length": 1.5,
 "tilt_max_deg": 40.0, "bucket": {"mass_full": 11.2, "radius": 0.12, "height": 0.25},
 "uav": {"UAV_1": {"mass": 5.4, "rope_mass": 0.45, "thrust_max": 220.0, "hook": [x, y, z], "start": [x, y, z],
                   "joints": ["<joint>", ...]}, ...},
 "ugv": {"start": [x, y, z], "joints": ["<joint>", ...], "grip_link": "<link>", "limits": {"<joint>": [lo, hi], ...}},
 "sensors": {"dir": "<the frames folder>", "manifest": "manifest.json", "every": 6, "delay": 6, "slots": 2}}
```

`mass` is the robot's mass as simulated (the bodies of its articulation, without the rope), `rope_mass` that of the
plant's rope it carries, `thrust_max` its airframe's, `hook` where its rope attaches (the hook joint's point on the
robot), in its root body's frame, `start` its position at t = 0 (true: the origin of its own odometry) and `joints` the
names of its articulation's joints in PhysX's order; the hook in the world is `pos` + R0 · `hook`, R0 the robot's
orientation (held as at t = 0). `cruise_z` is the height at which the plant's rig carries its hooks (`facts.md`); no
rule uses it. `ugv.start` is the parking bay's nominal pose (the vehicle's position there), where the odometry starts;
the vehicle itself stands up to 0.30 m (along x and along y) and 25° off it. `ugv.joints` are the vehicle's joints in
PhysX's order, `ugv.grip_link` the suction tool's link, `ugv.limits` the joints' limits as PhysX has them (rad, or m;
an unlimited joint shows ±3.4e38). `sensors` says where the frames are (The sensors). The bucket's actual water is not
told: between 8.5 and 10 litres. Where the pan and the parts stand that day is not told either: the photographs show
the bay as it stands photographed (`bay_brief.md`), and What varies per run gives the ranges.

One `act` request as it comes over the wire, on one line. The format, the order of the keys and the rounding are the
grader's; the numbers are made up (another bay, another fleet), so read nothing about this bay or a fleet into them:

```
{"type": "act", "t": 3720, "obs": {"t": 3720, "time": 62.0, "mission": {"dt": 0.016666666666666666, "time_limit": 600.0, "cruise_z": 2.6, "pour_r": 0.6, "pour_zmax": 2.5, "rope_length": 1.5, "tilt_max_deg": 40.0, "bucket": {"mass_full": 11.2, "radius": 0.12, "height": 0.25}, "uav": {"UAV_1": {"mass": 8.4127, "rope_mass": 0.45, "thrust_max": 360.0, "hook": [0.0, 0.0, -0.162], "start": [2.45, -0.45, 0.2843], "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]}, "UAV_2": {"mass": 8.4127, "rope_mass": 0.45, "thrust_max": 360.0, "hook": [0.0, 0.0, -0.162], "start": [4.1, -1.7, 0.2843], "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]}, "UAV_3": {"mass": 8.4127, "rope_mass": 0.45, "thrust_max": 360.0, "hook": [0.0, 0.0, -0.162], "start": [4.15, 0.6, 0.2843], "joints": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]}}, "ugv": {"start": [-1.15, -3.5, 0.0731], "joints": ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr", "arm_joint_1", "arm_joint_2", "arm_joint_3", "arm_joint_4", "arm_joint_5", "arm_joint_6"], "grip_link": "tool_link", "limits": {"wheel_fl": [-3.4028234663852886e+38, 3.4028234663852886e+38], "wheel_fr": [-3.4028234663852886e+38, 3.4028234663852886e+38], "wheel_rl": [-3.4028234663852886e+38, 3.4028234663852886e+38], "wheel_rr": [-3.4028234663852886e+38, 3.4028234663852886e+38], "arm_joint_1": [-6.2831854820251465, 6.2831854820251465], "arm_joint_2": [-6.2831854820251465, 6.2831854820251465], "arm_joint_3": [-3.1415927410125732, 3.1415927410125732], "arm_joint_4": [-6.2831854820251465, 6.2831854820251465], "arm_joint_5": [-6.2831854820251465, 6.2831854820251465], "arm_joint_6": [-6.2831854820251465, 6.2831854820251465]}}, "sensors": {"dir": "/var/tmp/arb_verify_k2x9q1/frames", "manifest": "manifest.json", "every": 6, "delay": 6, "slots": 2}}, "uav": {"UAV_1": {"pos": [-0.5069, 1.7184, 2.7731], "vel": [-0.1478, 0.2409, 0.0217], "load": [16.2371, 0.4127, -38.6904], "joints": {"shoulder_pan": [0.0012, 0.0], "shoulder_lift": [-1.5703, 0.0], "elbow_flex": [1.5694, 0.0], "wrist_flex": [0.0021, 0.0], "wrist_roll": [-0.0008, 0.0], "gripper": [0.3491, 0.0]}, "disabled": false}, "UAV_2": {"pos": [0.6147, 1.0012, 2.7346], "vel": [-0.1951, 0.1823, -0.0315], "load": [-8.3017, 13.8562, -38.9915], "joints": {"shoulder_pan": [-0.0007, 0.0], "shoulder_lift": [-1.5711, 0.0], "elbow_flex": [1.5702, 0.0], "wrist_flex": [-0.0013, 0.0], "wrist_roll": [0.0004, 0.0], "gripper": [0.3484, 0.0]}, "disabled": false}, "UAV_3": {"pos": [0.6572, 2.3897, 2.7615], "vel": [-0.1702, 0.2167, 0.0044], "load": [-7.9128, -14.2209, -38.4472], "joints": {"shoulder_pan": [0.0009, 0.0], "shoulder_lift": [-1.5698, 0.0], "elbow_flex": [1.5711, 0.0], "wrist_flex": [0.0006, 0.0], "wrist_roll": [-0.0015, 0.0], "gripper": [0.3502, 0.0]}, "disabled": false}}, "ugv": {"pose": [-0.1052, -1.0207, 1.31], "vel_body": [0.0007, -0.0012], "yaw_rate": 0.09, "joints": {"wheel_fl": [17.20416, 0.0021], "wheel_fr": [-3.88257, -0.0017], "wheel_rl": [16.95533, 0.0019], "wheel_rr": [-4.10872, -0.0024], "arm_joint_1": [0.03126, 0.0012], "arm_joint_2": [-1.10593, 0.2417], "arm_joint_3": [1.92874, -0.3108], "arm_joint_4": [-2.39042, 0.0695], "arm_joint_5": [-1.57164, 0.0003], "arm_joint_6": [0.01893, -0.0001]}, "grip_point": [0.6125, -0.0418, 0.5873], "vacuum": true, "disabled": false}, "frames": {"n": 619, "t": 3714, "time": 61.9, "slot": 1}}}
```

## The sensors

Every sensor of your fleet that counts (What you observe) and streams sends frames: images and scans the grader
renders in the real bay, from the sensor's own pose at the moment of capture, as the catalog's model delivers them.

- **Which sensors stream.** Per robot, in prim order (the order `Usd.PrimRange` visits the robot's prims), the first
  **four cameras** (RGB, depth and thermal together) and the first **lidar**. Any further sensor streams nothing in the
  bay mission; it still counts for the robot's mass and in the design checks (`requirements.md`). `check_structure.py`
  notes such sensors, and the manifest lists them (`not_streamed`).
- **Mount.** A sensor rides on the rigid body it sits under: its nearest ancestor with `RigidBodyAPI` inside its
  robot, or the articulation's root body if there is none. Its pose in that body's frame is the one your scene composes
  at rest; it follows that body as it moves, so a camera on an arm link follows the arm.
- **Cameras.** Pinhole, no distortion, the catalog's values exactly (not your prim's within 2 %): fx = fy = focal ÷
  horizontal aperture × W pixels; the principal point at the image's centre, ((W − 1)/2, (H − 1)/2), (0, 0) being the
  centre of the top-left pixel; u runs right, v runs down, row 0 is the top. The camera frame is the USD camera's: it
  looks along −Z, +Y is up, +X is right. A depth value is the distance along the optical axis (z), not along the ray.
  Every camera draws from 0.01 m to 100 m. The grader renders through cameras of its own placed at your camera's pose:
  your prim's projection, clipping range, lens and render settings are not used.
- **Lidars.** A lidar's frame is a range image of rows × columns beams. Beam (row c, column k) points along
  (cos e cos a, cos e sin a, sin e) in the lidar prim's frame, with a_k = k · 360° / columns (counter-clockwise about
  the prim's +Z, from its +X) and, for `LIDAR-3D`, e_c = +45° − c · 90° / 31 (row 0 the highest beam), for `LIDAR-2D`
  e = 0. A range is measured from the prim's origin to the first drawn surface along the beam beyond the model's
  minimum range (nothing nearer is drawn for a lidar: its own housing round the prim's origin does not block it), your
  own robot's body included; 0 means no return (nothing within the range below).

| model | frame (W × H, or rows × columns) | format (row-major, little-endian) | noise added | range |
|---|---|---|---|---|
| `CAM-RGB-W`, `-M`, `-T` | 1920 × 1200 | uint8 RGB, H × W × 3: the renderer's sRGB output | none: the renderer's own sampling noise | — |
| `CAM-RGB-S` | 640 × 480 | the same | the same | — |
| `THERM-A` | 160 × 120 | float32 kelvin, H × W | N(0, 0.10 K) per pixel and frame | — |
| `THERM-B`, `THERM-C` | 640 × 512 | float32 kelvin | N(0, 0.05 K) | — |
| `TOF-D` | 320 × 240 | float32 m, z-depth; 0 = no return | N(0, 4 mm + 0.5 % of z) | 0.15 to 4.0 m (outside: 0) |
| `STEREO-X` | 1280 × 720 | float32 m, z-depth; 0 = no return | N(0, 1.55 mm · z²/m²) | 0.3 to 10 m |
| `LIDAR-2D` | 1 × 1440 (0.25° apart) | float32 range, m; 0 = no return | N(0, 0.02 m) | 0.15 to 25 m |
| `LIDAR-3D` | 32 × 1024 (elevations +45° to −45°, 2.903° apart; azimuth 0.3516° apart) | float32 range, m; 0 = no return | N(0, 0.02 m) | 0.3 to 30 m |

- **When.** Every 6 control steps (10 Hz), at steps t = 0, 6, 12, ..., every streaming sensor is captured together, of
  the state that step's observation reports. A frame set is delivered with the observation 6 steps later (100 ms):
  the set captured at step 6n is set `n`, and it arrives at step 6n + 6. Every frame's noise is its own: drawn for its
  sensor and its set, whatever came before it.
- **What the frames show.** Isaac Sim 6.2's RTX renderer in its "Real-Time Path Tracing" mode with DLSS
  "performance", motion blur and depth of field off, a fixed exposure (every setting in Rehearsing). The photographs
  were path-traced at many samples a pixel; the frames are not, and RGB frames of one state are not exactly alike
  from one render to the next. Drawn: the real bay as photographed, in the plant's hall round it
  (`/opt/assets/arb/hall/bay_hall.usda`, at the world's origin, with its own 24 ceiling panel lights; it has no
  colliders), lit also by the photographs' two lights (a dome light of 300 and a distant light of 1800, 4° wide, turned
  35° about z and −50° about x); the parts, the lever, the bucket and the ropes (black) where they are; the gauge's
  needle at the line's pressure (`facts.md`); your four robots as composed, each robot's own body included. Not drawn:
  flames, smoke and steam (the photographs' volumes), the steam plume's hazard box, the water, anything of your scene
  but your four robots; the lights your robots carry are switched off. The steam's state shows on the gauge and the
  lever (The mission).
- **Thermal.** The thermal cameras see a stated temperature model: every drawn surface is at 293.15 K, except the fire
  pan, the whole of it, which is at 723.15 K in every capture whose state has the fire burning, and after that at 313.15 + 410 ·
  exp(−Δ / 8 s) K, Δ the time since the first observation in which the fire is out. Emissivity 1, no reflections, no
  atmosphere, no blur: a pixel's value is the temperature of the surface drawn at its centre (293.15 K where nothing is
  drawn), plus the noise.
- **The frames folder.** `mission.sensors.dir` names it. It holds `manifest.json`, written before your program
  starts, and two slot files per streaming sensor, `<file>.0` and `<file>.1`; your program can read and map them, and
  cannot write, create or delete anything there. The manifest gives "format", "every", "delay", "slots",
  "header_bytes", and per streaming sensor its `id`, `file`, `robot`, `prim` (your path), `model`, `kind`, `width`,
  `height`, `channels`, `dtype`, `unit`, its `intrinsics` {fx, fy, cx, cy} or `beams` {rows, columns, the elevations
  of the first and the last row, the azimuth step}, `range_m`, `noise`, the `body` it rides on (your path) and its
  `pose_in_body` {pos, quat [w, x, y, z]}; and `not_streamed`. Frame set n sits in slot n mod 2 of each sensor's
  files, behind a 64-byte header (little-endian: the magic `BAYFRM1\0`, u64 seq, u64 n, u64 capture step, f64 capture
  time, u32 width, u32 height, u32 channels, u32 dtype (1 uint8, 2 float32), u64 payload bytes), until set n + 2
  replaces it 12 steps later. Nothing writes the slots while a request waits for your reply; a program that reads
  between requests checks that seq is the same and even before and after it reads (the writer makes it odd while it
  writes). `~/given/frame_reader.py` reads the manifest and a slot that way; copy it into your submission to use it.
- **The GPU.** Your program may use it (CUDA, PyTorch). It shares it with the grader's renderer: on a 24 GB GPU the
  renderer leaves at least 4 GB of its memory free (it holds about 12 GB for a fleet that streams all the sensors the
  cap allows at their full size, and more, up to about 20 GB, while it compiles shaders for looks it has not drawn
  before). Your program cannot start Isaac Sim when it is graded.

## The real bay

The graded runs do not use your scene's bay. The grader builds the mission in its own stage of the real bay (its
floor, tables, crate, steam line and valve, gauge, pan, tank, the bucket and the plant's rope rig, and the hall round
it, lit as in the photographs) and puts your four robots in it.

- **Your robots, as your scene builds them**: everything under `/World/Fleet/UAV_1`, `UAV_2`, `UAV_3` and `UGV` comes
  over as composed (bodies, joints, drives, colliders, masses, materials, sensors; the lights they carry are switched
  off), except the ropes and any joint from a robot to something outside it. The other robots' cameras see your robots
  as composed: their meshes and materials are what the frames show. Nothing else of your scene comes over: not your bay, not your OmniGraph
  logic, not your lights or helpers, not your physics scene (the real bay's: one scene, 120 Hz, gravity 9.81 m/s²,
  the TGS solver, on the CPU; the floor's friction is 0.5, `facts.md`); collision filtering in your stage reaches only
  your own robots (a collision group or collection of yours that takes in the whole stage takes in only your robots
  there), and every contact of a robot's body is reported, whatever contact-report settings your scene authors.
- **Where they stand**: each flying robot on the real pad nearest to where your scene puts it (the three assigned so
  that their distances add up to the least), the vehicle in the real parking bay (then moved as What varies per run
  says), each by one horizontal shift of the whole robot: heights and headings as your scene has them, its footprint
  (the box round everything under the robot but its `Rope`) centred on the marking as the layout check measures it.
- **The rig**: each flying robot carries one of the plant's three ropes (1.50 m from hook to ring, 0.30 kg/m, a chain
  of links), hooked where your scene's hook joint holds your rope (the same body, the same point), its other end on the
  bucket's ring that faces the robot's pad; it starts slack, lying on the floor. A rope never collides with the bucket;
  its contacts with the robot that carries it never count (and its links that start next to that robot's colliders do
  not collide with it at all). The bucket is the real one (1.2 kg plus the water; its rings at the rim, 0.14 m from its
  axis).
- **A fleet that does not fit** aborts the run before your program starts (every check of that run scores 0): a robot
  with no articulation root outside its rope, a flying robot with no hook joint between one of its bodies and its rope,
  or a hook that, with its robot on the real pad, stands more than 1.45 m from its ring.
- **What your program knows**: its working directory is a copy of your submission, so what it reads there (your
  scene) describes your twin, not the real bay. Nothing else of your home is there when it is graded: `~/given` is
  absent, so copy into the submission whatever your program imports or reads.

## The water

The bucket holds V litres (its mass follows: 1.2 kg + 1 kg per litre, updated as water leaves). The free surface
stands normal to the apparent gravity in the bucket, gravity minus the bucket's acceleration. When the surface reaches
the rim, water leaves at 15 L/s per metre of excess (R·tan θ above the freeboard; the bucket is 0.12 m in radius,
0.25 m tall, and 10 L leave 3 cm of freeboard). Water that leaves while the bucket's bottom centre is within 0.6 m
(horizontally) of the pan's centre, where the pan stands that day, and no higher than 2.5 m is **poured**; any other
water is **spilt**. So: tilt only over the pan, and accelerate gently.

The grader's model, once per control step, from the bucket body's true motion: the acceleration is the velocity's
change over the step, smoothed (a ← 0.7 a + 0.3 a_step); θ is the angle between the bucket's axis and the apparent
gravity's upward direction; the freeboard is 0.25 m − V / (π · 0.12²) (V in m³); water leaves at 15 L/s per metre of
max(0, 0.12 m · tan θ − freeboard), and at 20 L/s once θ ≥ 89°, for that step's 1/60 s.

## What varies per run

The two runs are the grader's own days, drawn once from these ranges and kept hidden: every submission meets the same
two. On each day:

- the water: between 8.5 and 10 L;
- the wind: 2 or 3 gusts on the flying robots and the bucket (not on the ropes or the vehicle): one that starts 2 to
  12 s after the bucket first leaves the floor (lift-off, as How it is scored defines it), and 1 or 2 more that start
  at times between 8 and 240 s; each 4 to 10 m/s for 1.5 to 5 s, steady while it blows, from its own horizontal
  direction. A gust pushes along its direction with 0.5 · 1.2 kg/m³ · A · v², A = 0.12 m² on each flying robot,
  whatever its airframe, and 0.06 m² on the bucket (7.2 N and 3.6 N at 10 m/s); gusts that overlap add up;
- the parts: each on Table A or Table B (1 to 5 on each), anywhere on its top with its footprint at least 4 cm inside
  the edges and 5 cm from any other part's, turned to any heading; the cracker box, the sugar box, the mustard bottle
  and the power drill each standing as in the photographs or lying on a broad side (a quarter turn, either way, about
  the long side of its footprint), the soup can and the mug always standing;
- the vehicle: up to 0.30 m (along x and along y) and 25° off the parking bay's nominal pose, turned about its position,
  so it may stand partly over the bay's marking; its odometry starts from the nominal pose;
- the valve's lever: left between its open stop and 60° toward closed (the line is under full pressure until the lever
  counts as closed);
- the pan: up to 0.45 m (along x and along y) from where the photographs show it;
- the sensors' noise and the odometry's per-run errors.

## How it is scored (the `control` section)

The grader measures the outcomes in the real bay. Each check below is the mean over the two runs; values between full
and zero score linearly. The mission's score (the checks' weighted mean) multiplies the scene's (`instruction.md`).
The three checks marked † (how the bucket was handled) count only in a run that ends with the fire out: lifting the
bucket and holding it steady earns nothing by itself. The five checks marked * count in proportion to what each run got
done: a run's value times its progress, (steam released + fire out + station cleared) / 3, and for consistency the two
runs' mean progress; a run that does none of the three jobs earns nothing for being careful.

| check | weight | full marks | zero |
|---|---|---|---|
| program ran | 0.5 | the run went to its end without an abort | aborted (and then every check of that run is 0) |
| steam released | 1.0 | the pressure is gone at the end (the lever at its closed stop) | the line still under pressure |
| fire out | 1.5 | the fire is out at the end | still burning |
| station cleared | 1.5 | all six parts inside the crate at the end | none |
| in time | 1.0 | the mission ended "done" (all jobs, everything still) with its still period begun by 300 s | not done by 600 s |
| lift-off level † | 0.6 | the bucket's tilt ≤ 3° through the 1.5 s after lift-off | ≥ 10° |
| swing † | 0.6 | the bucket's bottom centre never more than 0.08 m, horizontally, from the hooks' centroid during the carry (from 2 s after lift-off until the pour begins, whenever the bucket is outside the pan zone) | ≥ 0.30 m |
| water kept † | 1.0 | ≤ 0.05 L spilt (The water) | ≥ 1.0 L |
| parts not dropped * | 0.4 | no part dropped | 2 or more |
| no losses * | 1.0 | no robot lost (by the bay's rules), none crashed, the bucket never dropped | any |
| no collisions * | 1.0 | no contact (below); and every robot keeps the colliders of the library models it declares (each model referenced from the library, not a copy; none of its colliders removed, switched off, made a trigger or shrunk: `check_structure.py` says so; otherwise this check scores 0) | 5 or more contacts |
| separation * | 0.6 | the flying robots never closer than 1.2 m to each other (position to position, from t = 0 on) | ≤ 0.9 m |
| steady end | 0.4 | the run ended "done" | otherwise |
| consistency * | 1.0 | the two runs' scores (over the checks above) within 0.05 of each other | 0.30 apart or more |

What the grader measures, on the true state:

- **Lift-off**: the first moment the bucket's bottom centre is more than 0.08 m above the floor while all three hooks
  are within 1.7 m of the centre of its rim.
- **The pan zone**: within 0.6 m (horizontally) of the pan's centre, where the pan stands that day. **The pour** begins
  when water first leaves over the pan (The water).
- **The bucket dropped**: its bottom centre below 0.03 m more than 1 s after lift-off and before the pour has begun;
  setting the bucket down before the pour counts as dropping it.
- **A crash**: a flying robot whose position comes below 0.30 m more than 0.6 m (horizontally) from where it stood at
  t = 0; landing anywhere but on its own pad counts.
- **Where a part is**: in the crate when the centre of its bounding box lies inside the crate's box seen from above and
  no more than 0.10 m above the crate's top; on a table when that centre lies inside the table's box seen from above
  and no lower than 2 cm below the table's top; anywhere else the part is dropped. A part still held at the end is
  neither in the crate nor dropped.
- **Contacts**: a body of a robot (its rope aside) touching another robot or another robot's rope, a table, the crate,
  the tank, the pan, the steam line (its valve body and gauge included, its lever not) or the bucket. The floor, the
  parts, the lever, the floor markings and the calibration mat are free. Each robot and each thing it touches count
  once per onset: a contact that has been gone for 0.5 s starts a new one.

## Rehearsing

Nothing in `~/given` runs the mission: rehearse it in your own twin, with a harness of your own that keeps the
grader's model as this page gives it (to rehearse a day, move your twin's pan, vehicle, lever and parts as What varies
per run says). The grader runs Isaac Sim 6.2 (`SimulationContext`, physics at 1/120 s on the CPU) on the real bay's
physics scene, and drives and reads the bodies through PhysX's tensor views (articulation and rigid-body views; the
forces applied at every physics step). It renders the robots' sensors with Isaac Sim 6.2's RTX renderer through
Replicator render products, one per streaming camera and one per lidar (a 360° × 180° equirectangular range
camera at the lidar prim's pose, read at the beams' directions), each product set to: `omni:rtx:rendermode`
"RealTimePathTracing", `omni:rtx:post:aa:op` "dlss" ("none" for the lidars), `omni:rtx:post:dlss:execMode`
"performance", `omni:rtx:post:motionblur:enabled` false, `omni:rtx:post:dof:enabled` false,
`omni:rtx:post:tonemap:operator` "acesApproximation" and `omni:rtx:post:tonemap:exposureKey` 0.25 (a fixed exposure:
Kit's automatic exposure, `/rtx/post/histogram/enabled`, is off), `omni:rtx:rt:ecoMode:enabled` false; the RGB frames
are the `rgb` annotator's output, depth `distance_to_image_plane`, the lidars' ranges `distance_to_camera`, thermal
the `instance_id_segmentation` mapped to the temperatures above. A capture is one Replicator orchestrator step
(`rep.orchestrator.step(rt_subframes=1, delta_time=0, pause_timeline=False)` with `/app/player/playSimulations` off
around it), after which every product's read shows that capture; it never steps the physics. One control step:

1. the observation of the step: the noise drawn, the odometry's random walks moved on;
2. at a capture step (t = 0, 6, 12, ...), after the observation's numbers: every streaming sensor posed from its body
   and rendered once, its frames processed and held; the frame set captured 6 steps earlier is written to its slots
   and named in this step's observation;
3. your action: the thrusts clamped (What you command), the vehicle's targets written, the grip applied;
4. two physics steps of 1/120 s. Before each: the thrust and the wind on each flying robot's root body (no thrust on a
   lost one), the wind on the bucket, the lever kept where it is unless a robot has touched it within 0.5 s, the held
   part put where the grip link carries it. After each: each flying robot's orientation set back to its orientation at
   t = 0 and its angular velocity to zero, the contacts read;
5. the water (The water); the bay's rules (the valve, the pressure, the fire, the hazards); the measurements (lift-off,
   the bucket's tilt, swing and height, the separation, crashes); every 30 steps, where the parts are and whether the
   mission is done.
