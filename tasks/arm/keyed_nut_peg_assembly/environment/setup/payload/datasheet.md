# Bayonet nut kit cell: data sheet

What the engineer who commissions a controller on this cell is given: the kit's drawing data and tolerances, how a
kit is laid out, the robot and its controller, the sensors and cameras, the safety stops, the takt, the wire of
the commissioning service, and what the fit of a nut on a shaft asks of the cell (section 12). The numbers, the camera calibration and a few geometry helpers are also in
`cell_data.py` (plain Python and NumPy). The cell itself (its simulator, its model files, its state) is not in your
account: you meet it through the commissioning service (section 10) and when your controller is graded.

**Sections 2 to 7 give the drawing: nominal values. No real nut, peg or cell has exactly these.** Every episode,
graded or commissioning, is a real kit on the cell as it is that day, each within the tolerances of section 11, and
none of its actual values is told to your controller. A model built from this sheet is the drawing's cell; the
commissioning service is where you meet real ones.

## 1. Frames and units

World frame: metres, radians, seconds. The table top is the plane z = 0.80 and spans x, y in [-0.40, 0.40]. The
Panda base stands at (-0.56, 0, 0.80) facing +x. Quaternions are (w, x, y, z). Angles in a nut are counted
counter-clockwise seen from above, from the handle (the nut's local +x axis).

## 2. The pegs (drawing)

| part | geometry |
|---|---|
| collar (base) | a 50 x 50 x 10 mm block on the table, dark grey |
| shaft | a cylinder of radius 12.3 mm, 65 mm tall, on the collar, light grey; its top is 75 mm above the table; its end is square (no lead-in chamfer) |
| pin | a block on the shaft surface on the peg's local +x side: radial extent 11.8 to 17.2 mm from the shaft axis, 3 mm wide (tangential), 4 mm tall, centred 37.5 mm above the collar top (its top face 39.5 mm above it) |
| bands | 1, 2 or 3 near-black rings around the shaft, each 7 mm tall and 0.4 mm proud of the shaft; centres 10, 23 and 36 mm below the shaft top (a peg with one band has the top one, with two the top two) |

The three pegs of a kit are identical except for their bands. A peg's yaw turns its pin with it.

## 3. The nuts (drawing)

- **Ring.** 45 mm tall, a 24-sided prism: 24 sectors of 15 degrees (sector s spans [15 s, 15 (s + 1)] degrees from
  the handle) by 9 layers of 5 mm (layer 0 at the bottom). Outside flats 28 mm from the hole axis. The nut's origin
  is the point of the hole axis on the bottom face.
- **Hole.** A 24-sided hole whose flats are 12.8 mm from the axis, with square edges (no lead-in chamfer). It
  fits over the 12.3 mm shaft with a **locating fit**: 1 mm on the diameter, the fine series of clearance holes
  for this size, so a nut on a shaft has 0.5 mm of radial clearance on the drawing and 0.35 to 0.65 mm on real
  parts (section 11). What that asks of the cell: section 12.
- **Wall cells.** In every (sector, layer) cell the wall is two blocks: an outer block from 19 to 28 mm (always
  there) and an inner block from 12.8 to 19 mm, which is either solid or cut away. The cut cells form the
  **groove**; its floor is at 19 mm from the axis, so a cut cell is a recess 6.2 mm deep. An inner block is 5.4 mm
  wide (tangentially), wider than its sector is at the hole (3.4 mm), so the solid cells beside a channel overhang
  it near the hole: a channel two sectors wide is nominally 30 degrees wide but opens only about 21 degrees at the
  hole surface. With the pin's width, a nut resting on the pin drops into a riser only while the riser's middle
  is within about +-5 degrees of the pin (measured on nuts left free on the pin: +-4 to +-5.5 degrees).
- **Handle.** A bar along the nut's +x axis from 26 to 98 mm from the hole axis, 16 mm wide, from 25 to 41 mm above
  the bottom face.
- **Mass and surfaces.** Mass 0.05 to 0.15 kg, centre of mass 15 mm from the hole axis towards the handle and
  18 mm above the bottom face. Sliding friction between a nut and a peg (shaft, pin, collar) 0.3 to 0.7, its own
  value for every nut. One colour per nut, nominal RGB in `cell_data.NUT_COLOURS`: blue, green, orange, pink, purple,
  red, teal, yellow.
- **Groove family** (identical rules for every nut; `cell_data.groove_cells` lists the cut cells of a groove):
  - channels are 2 sectors (30 degrees) wide; a groove is given by its entrance sector e (0 to 23), two turns
    k1 and k2, two dead-end lengths and its false entrances;
  - the **entrance riser**: layers 0-1 at sectors (e, e + 1);
  - the **first turn**: layers 1-2, running k1 sectors from sector e (k1 = 2, 3 or 4 sectors either way:
    counter-clockwise for k1 > 0), then a **riser** in layers 2-4 at the new sector e + k1;
  - the **second turn**: layers 4-5, k2 sectors from there (2, 3 or 4, either way), then the last riser in
    layers 5-7;
  - **dead ends**: each turn channel may continue 1, 2 or 3 sectors the other way from where it starts, and stop;
  - **false entrances** (0, 1 or 2): risers in layers 0-2 at other sectors that lead nowhere; each is at least
    2 sectors clear of every cut cell of the true path in layers 0-2 and of the other false entrance;
  - layer 8 (the top one) is always solid, and the outer blocks are all there: the top face and the outside of the
    ring show nothing. The groove is plain geometry, though: looking down into the hole at a steep angle you see
    the far side of the hole wall below the top layer, where every cut cell is a recess; the entrances are openings
    in the bottom face.
- **How a nut comes down.** The pin enters the groove from below. With the nut resting on the pin, the nut's
  bottom is 39.5 mm above the collar. It comes down to 24.5 mm once the pin is in the entrance riser (or a false
  entrance), to 9.5 mm once it has passed the first turn and riser, and to the collar once it has passed the
  second. Passing a turn means rotating the nut about the shaft while the pin is in the turn channel: the nut
  turns by -15 degrees per sector the pin advances counter-clockwise in the nut's frame.
- **How the family is drawn** (the graded nuts and the development nuts alike): the entrance uniformly over the 24
  sectors; each turn 2, 3 or 4 sectors, either way, each with equal chance; each dead end 0, 1, 2 or 3 sectors
  with equal chance; 0, 1 or 2 false entrances with equal chance, each placed uniformly among the sectors that
  keep the clearance above (fewer if none is free); mass and friction uniformly in their ranges. The real nut then
  has this groove within the tolerances of section 11.

## 4. A kit and how it is laid out

A kit is three nuts of three different colours (chosen uniformly among the eight) and a **work order**: a random
assignment of the nuts to the pegs with 1, 2 and 3 bands. Every observation carries it as `order`: a list of
`{"colour": name, "peg_bands": 1 | 2 | 3}`, one entry per nut.

Layout tolerances (every episode draws them anew, uniformly, and redraws until all hold):

- peg centres: x in [-0.02, 0.15], y in [-0.21, 0.21], at least 0.13 m apart; peg yaws uniform over the full
  circle; which peg carries how many bands is random;
- nut centres (hole axes): x in [-0.24, -0.06], y in [-0.24, 0.24], at least 0.12 m apart and at least 0.13 m from
  every peg centre; the nuts lie flat on the table, bottom face down;
- nut yaw (the handle's direction) = the nut's azimuth seen from the robot base, atan2(y, x + 0.56), plus 135
  degrees plus a uniform offset within +-60 degrees;
- the handle's centre (63 mm from the hole axis) has |y| < 0.30 and x > -0.28, lies at least 0.09 m from every peg
  centre, at least 0.07 m from the other nuts' centres and at least 0.05 m from the other handles' centres.

## 5. The robot and its controller

- **Arm and gripper.** A Franka Panda with its parallel-jaw gripper: the robosuite 1.5.2 models
  `robots/panda/robot.xml` and `grippers/panda_gripper.xml` under
  `/opt/venv/lib/python3.12/site-packages/robosuite/models/assets/` (kinematics, inertias, joint limits, torque
  limits of 80 N m on joints 1-5 and 12 N m on joints 6-7, the collision shapes of hand and fingers). The base is at
  (-0.56, 0, 0.80). An episode starts at rest at joints (0, -0.35, 0, -2.35, 0, 2.0, pi/4) with the fingers open,
  the gripper pointing down. `ee_pos` / `ee_quat` are the pose of the gripper's `grip_site` (between the finger
  pads).
- **Action** (10 Hz): `[dx, dy, dz, drx, dry, drz, grip, cams]`.
  - `dx, dy, dz`: a displacement of the gripper site in the world frame (m; the vector is clipped to a length of
    0.02). The position target of the step is the site's **current** position plus this displacement, clamped to
    the box x, y in [-0.45, 0.45], z in [0.802, 1.35]. Targets do not accumulate: every step starts again from
    where the gripper is.
  - `drx, dry, drz`: a world-frame rotation vector (rad; clipped to a length of 0.15) applied to the gripper's
    current orientation; the result is the step's orientation target.
  - `grip`: -1 = fingers fully open (80 mm) ... +1 = fully closed, linear in between. Each finger is
    position-controlled to its half of the commanded opening with a stiffness of 3000 N/m, and its force is
    capped at 20 N. The two fingers are one mechanism, as on the real hand: they move symmetrically about the
    hand's axis. A part is therefore centred between them when it is gripped (it is pushed there, or the hand
    gives), and a held part cannot push the fingers aside together; across the closing direction and in rotation
    it is held by the pads' friction alone.
  - `cams`: an integer bit mask of the cameras whose image the next observation carries (1 = left, 2 = right,
    4 = wrist; 0 = none).
- **Tracking.** The arm is torque-controlled at 500 Hz by an operational-space controller that holds the step's
  target for 0.1 s: a task-space acceleration of 400 x (position error) - 40 x (velocity) and 200 x (orientation
  error) - 28.3 x (angular velocity), i.e. critically damped with natural frequencies of 20 and 14.1 rad/s, turned
  into joint torques through the arm's task-space inertia, with gravity and Coriolis compensation. In the null
  space the arm is pulled towards the posture (0, -0.3, 0, -2.2, 0, 1.9, pi/4) (5 N m/rad, damping 6 N m s/rad);
  every joint has an extra damping of 0.5 N m s/rad and is braked above 1.5 rad/s. Because the target is re-based
  on the current pose at every step, a displacement command that is blocked keeps pressing with a force that
  grows with the commanded displacement (the task-space inertia times 400 times the error); it is a compliant
  position controller, not a force controller. The law knows nothing of the joints' friction (section 11), so the
  real arm lags a little behind what this law alone would do: a displacement command of a few tenths of a
  millimetre may not move it at all.
- **Compliance.** What a blocked displacement command presses with, at the grip site with the gripper pointing
  down (the task-space inertia times 400, at the height where a nut meets a shaft): over the pegs' area 3.3 to
  5.8 N per mm in x, about 2.5 N per mm in y and 2.9 to 3.6 N per mm in z; over the nuts' area about 3.3 N per mm
  in x and y and about 6 N per mm in z. The directions are coupled through the arm's inertia: a push down also
  pushes away from the robot base (+x) with 0.3 to 0.45 of the downward force, and sideways (y) with up to 0.15
  of it at the sides of the area. In rotation the hand resists with 170 to 220 N m/rad about world x, 125 to
  170 N m/rad about y and 40 N m/rad about the vertical axis.

## 6. Sensors

- `q`, `dq`: the seven arm joints' positions and velocities (rounded to 1e-6).
- `gripper`: the finger opening in m (0.08 fully open).
- `wrench`: force (N) and torque (N m) at the wrist force/torque sensor, expressed in world axes. The sensor sits
  at the gripper's mounting face (the `ft_frame` of the gripper model, 97 mm behind the grip site) and reads what
  the arm applies to the gripper there: with nothing touched it reads the gripper's weight, about +5.1 N in z; a
  contact that pushes the gripper up lowers z; pushing the gripper against something in +x raises x. Every sample
  carries noise (sigma 0.3 N per force axis, 0.03 N m per torque axis) and the sensor's zero error: an offset that
  is drawn anew for each episode (within +-0.5 N per force axis, +-0.05 N m per torque axis) plus a steady drift
  during the episode (section 11).
- `T_world_wrist`: the 4 x 4 pose of the wrist camera in the world, OpenCV convention.
- There is no depth image and no object state of any kind.

## 7. Cameras

| camera | pose | field of view (vertical) | image |
|---|---|---|---|
| `left` | fixed at (-0.30, 0.55, 1.45), looking at (0.05, 0, 0.80), image up along world +z | 50 degrees | 320 x 240 |
| `right` | fixed at (-0.30, -0.55, 1.45), looking at (0.05, 0, 0.80), image up along world +z | 50 degrees | 320 x 240 |
| `wrist` | on the hand (the robot model's `eye_in_hand`), looking along the gripper's approach axis; its pose is `T_world_wrist` | 75 degrees | 256 x 192 |

This table and the functions below are the cell's **published calibration**: the cameras as drawn. The real
cameras sit within the tolerances of section 11 of it (mounting position and orientation, focal length, principal
point), and `T_world_wrist` is the hand's pose with the drawn mount, not the real camera's pose.

Pinhole cameras without distortion; the focal length in pixels is (H / 2) / tan(fovy / 2) for both axes and the
principal point is at (W / 2, H / 2), so the centre of pixel (u, v) is at (u + 0.5, v + 0.5).
`cell_data.fixed_camera_matrices(name)` returns (K, T_world_cam) of a fixed camera in the OpenCV convention (x
right, y down, z forward), `cell_data.wrist_camera_intrinsics()` the wrist camera's K. Images are RGB, 8 bit, with
pixel noise of sigma 2; on the wire each is `{"dtype": "uint8", "shape": [H, W, 3], "b64": <raw row-major bytes,
base64>}` (`cell_data.decode_image`). An image is empty (shape `[0, 0, 3]`) unless the previous action asked for
it; the first observation of an episode carries all three.

**What the images show.** The cell in its laboratory: the robot in its covers, the bench top of light oak (the
table), the pegs (dark grey collars, near-white shafts, black bands), the nuts, and around the bench the room (a
wooden floor, shelving, other benches, the camera gantry with its lamp over the robot's side of the bench). The
light is the gantry's lamp, a bar, and the room's warm daylight, so a colour in an image is not its nominal RGB:
hues lie up to about 20 degrees from the nominal ones (towards the warm light) and lit faces are lighter; the eight
colours stay apart from each other. The arm and the hand shade the bench under the lamp (where the lamp is cut off
altogether the oak is lit by the room alone and looks orange-brown). The wrist camera does not see the hand it is
mounted in, only the fingers. The lamps' brightness of the day (section 11) scales the light of the whole image; the
cameras' exposure does not change.

## 8. Safety stops (they end an episode, on the commissioning service as when graded)

- **A ruined nut.** While a nut is on its own peg (its hole axis within 5 mm of that peg's shaft axis, the nut
  below the shaft top), the centre of the pin lies inside a solid cell of the nut's wall, more than 5 degrees
  sideways or 1.5 mm up or down from every cut cell, for 3 consecutive steps. The groove walls are solid, so this
  only happens when the pin is forced through one. It is judged on the real nut and the real pin (their own
  walls, floors and heights within the tolerances), not on the drawing.
- **An unstable simulation.** Any arm joint faster than 8 rad/s, a nut faster than 5 m/s, or a state that is not
  finite.

## 9. Takt and the episode

One action every 0.1 s. An episode is one kit and lasts at most 850 steps: the takt time, 85 s per kit.
Observation format, the grading wire (READY / reset / act / close), the deadlines, the controller budget and the
rule by which a nut counts as assembled are in the task instruction.

## 10. The commissioning service

The cell, on 16 development kits, for your commissioning window. It is the cell your controller is graded on (the
same robot and controller, sensors, cameras, kit family, layout tolerances, safety stops, takt and budgets, and the
same tolerances: every development kit is a real kit, with its own actual values and its own state of the cell);
the development kits are draws from the family of section 3 that the reference controller completed within the
takt, like the graded kits, and none of them is a graded kit.

- **The window.** 32 episodes in your session, at most 2 at a time. An episode counts from its `reset`. The count
  is in `/run/keyedcell/commission_status.json` and does not grow. (The size is that of a
  real commissioning window: one working shift on the cell with a tray of 16 pre-series kits, at about a quarter
  of an hour per trial: lay the kit out, run one takt, look at the result, clear the pegs, change the program.)
- **A kit's episode.** `reset(kit)` lays development kit `kit` (1 to 16) out. Each kit has its own layout, its
  own actual values within the tolerances and its own state of the cell, and its episode starts the same way
  every time you start it (the same layout, parts, cameras, friction, sensor zero and noise). The graded kits
  have other values within the same tolerances.
- **What it answers.** What the cell gives a controller: the observation after every action, and how the episode
  ended. It answers no object pose, no groove, no score, no completion verdict and no debrief: it does not judge
  the assembly, so an episode does not end when the kit is complete (when graded, it does). How an episode went,
  you judge from what your program saw and recorded and from the cameras.
- **How an episode ends.** After 850 steps ("all steps run"); on a safety stop (it names the ruined nut's colour
  or the instability); when the episode's controller budget is used up (600 s: the service's wait for each of your
  requests plus 0.45 s per left or right image and 0.35 s per wrist image asked for; the first observation's
  images are free); after 120 s without a request; or when you close it.

Wire: a unix stream socket, `/run/keyedcell/commission.sock`; one connection is one episode; one JSON object per
line (`~/commission/client.py` speaks it).

```
{"id": 1, "method": "reset", "params": {"kit": 3}}
        -> {"id": 1, "result": {"ok": true, "kit": 3, "episode": 5, "episodes_left": 27, "max_steps": 850,
                                "controller_budget_s": 600.0, "obs": {...}}}
{"id": 2, "method": "step", "params": {"action": [8 numbers]}}
        -> {"id": 2, "result": {"obs": {...}, "done": false}}
        -> {"id": 2, "result": {"obs": {...}, "done": false, "probe_obs": {...}}}      (about 1 % of the steps)
        -> {"id": 2, "result": {"obs": {...}, "done": true, "ended": "all steps run"}}
{"id": 3, "method": "close", "params": {"images": true}}
        -> {"id": 3, "result": {"ok": true, "ended": "...", "steps": 850, "controller_s": 7.9,
                                "controller_budget_s": 600.0, "images_requested": 12,
                                "final_images": {"left": {...}, "right": {...}, "wrist": {...}}}}
```

- `obs` is the observation of the task instruction (`t`, `order`, `q`, `dq`, `ee_pos`, `ee_quat`, `gripper`,
  `wrench`, `T_world_wrist`, `left`, `right`, `wrist`). It is `null` only on the reply that ends an episode on its
  budget or on an unstable step.
- `probe_obs` holds the readings (no images) of a slightly perturbed copy of `obs`, the copy the grader sends when
  it asks a step twice; `~/commission/run_kit.py` uses it to query your controller the way the grader does.
- `close` ends the episode; with `"images": true` its reply carries the three cameras' images of the final state
  (free of charge). After an episode is over you have 60 s to close it.
- A refusal is `{"id": n, "error": word, "detail": sentence}`: `bad_params` (the kit number), `busy` (two episodes
  are open; no episode is used), `commissioning_window_used`, `no_episode` (a `step` before `reset`),
  `one_episode_per_connection`, `bad_action` (not 8 finite numbers with an integer camera mask 0 to 7; the step is
  not applied and the episode goes on), `episode_over`, `forbidden` (any other method), `cell_fault`, `bad_request`
  (a line that is not one JSON request object of at most 64 KiB; the connection closes).
- When a limit ends the episode while the service waits, it sends `{"id": null, "event": "episode_over",
  "reason": ...}` and closes the connection.

## 11. Tolerances: the real kit and the cell on the day

Every value below is drawn anew for every episode's kit and cell, uniformly within its range and independently of
the others (`cell_data.TOL`; `cell_data.sample_unit` draws one set). Nothing else differs from the drawing: the
layout rules of section 4, the family rules of section 3, the controller's law and limits, the nut's outside and
handle, the collar, the shaft's height and the bands are as drawn.

| what | drawing | a real one |
|---|---|---|
| **Every nut** | | |
| each of the 24 sector boundaries of the hole wall (where a channel ends there: its wall) | 15 degrees x s from the handle | +-1 degree |
| each of the 8 layer boundaries inside the wall (the channels' floors and ceilings) | 5 mm x l above the bottom face | +-0.2 mm |
| hole flats from the axis | 12.8 mm | +-0.1 mm |
| groove floor from the axis | 19 mm | +-0.2 mm |
| the handle's direction against the wall's sector grid | 0 | +-0.5 degrees |
| colour, each RGB component | `cell_data.NUT_COLOURS` | +-0.04 |
| sliding friction against a peg | | 0.3 to 0.7 |
| mass | | 0.05 to 0.15 kg |
| **Every peg** | | |
| shaft radius | 12.3 mm | +-0.05 mm |
| pin: reach from the shaft axis | 17.2 mm | +-0.2 mm |
| pin: width / height | 3 mm / 4 mm | +-0.1 mm each |
| pin: centre above the collar top | 37.5 mm | +-0.3 mm |
| **The cell on the day** | | |
| finger pads: sliding friction against a nut | 2 in the robot model | 0.6 to 1.2 (the pads' torsional friction in proportion) |
| each arm joint: friction torque | the curve below | x 0.6 to 1.4, its own factor for every joint |
| wrench zero: offset per axis | 0 | +-0.5 N, +-0.05 N m |
| wrench zero: drift per axis, steady through the episode | 0 | up to +-0.3 N and +-0.012 N m per minute |
| fixed cameras: position per axis / orientation about each axis | section 7 | +-2 mm / +-0.2 degrees |
| wrist camera on the hand: position per axis / orientation about each axis | the robot model's mount | +-1.5 mm / +-0.3 degrees |
| every camera: focal length / principal point per axis | section 7 | +-0.5 % / +-1 pixel |
| lamps: brightness | | +-15 % |

- **Joint friction.** The friction torque at arm joint j turning at w rad/s opposes the motion and is
  phi1 / (1 + exp(-phi2 (w + phi3))) - phi1 / (1 + exp(-phi2 phi3)), with phi1 = (0.546, 0.872, 0.641, 1.279,
  0.839, 0.303, 0.565) N m, phi2 = (5.12, 9.07, 10.14, 5.59, 8.35, 17.13, 10.34) s/rad and phi3 = (0.0395, 0.0259,
  -0.0461, 0.0362, 0.0262, -0.0210, 0.0036) rad/s for joints 1 to 7: the curve identified on this robot type
  (`cell_data.JOINT_FRICTION`, `cell_data.joint_friction_torque`). It levels off at about half of phi1.
- **What the tolerances add up to.** A nut resting on the pin has its bottom 39.5 +- 0.35 mm above the collar; it
  stalls at 24.5 +- 0.55 mm after the entrance riser and at 9.5 +- 0.55 mm after the first turn. The middles of two
  risers are 30, 45 or 60 degrees apart, +-2 degrees. The window in which a nut left free on the pin drops into a
  riser is between about +-4.5 and +-7 degrees about the riser's real middle. The radial clearance of a nut on a
  shaft is 0.35 to 0.65 mm.
- **What the cameras' tolerances add up to.** With the published calibration alone, a point on the table seen by a
  fixed camera is typically about 4 mm off (one time in ten more than 8 mm, at worst about 16 mm); a point seen by
  the wrist camera from 0.25 m is typically about 2 mm off (one time in ten more than 3.5 mm, at worst about 6 mm).
  A camera's error is the same through an episode, and the wrist camera's is fixed in the hand's frame.

## 12. The fit: what putting a nut on a shaft asks of the cell

- **The fit.** A 25.6 mm hole (across flats) on a 24.6 mm shaft: 1 mm on the diameter, the fine series of
  clearance holes for this size; 0.5 mm of radial clearance on the drawing, 0.35 to 0.65 mm on real parts. The
  shaft end and the hole's edges are square: there is no lead-in chamfer to guide the nut on. The hole is 45 mm
  deep, so a nut tilted against the shaft by more than twice the clearance over that depth (about 0.9 to 1.6
  degrees) binds on the shaft at two points instead of sliding down.
- **The fit and the cameras.** A position taken from an image with the published calibration is off by
  millimetres (section 11), several times the clearance, for the shaft and for the held nut alike: the nut's hole
  cannot be put over a shaft from image positions alone.
- **What holds the nut.** Along the fingers' closing direction the nut is centred on the hand's axis (the fingers
  are one mechanism, section 5). Across it and in rotation the pads' friction holds it (0.6 to 1.2, 20 N per
  finger): a nut gripped by its ring sags in the grip by a few tenths of a degree (up to about 0.8 degrees
  measured).
- **What the wrist reads, the nut on the shaft end.** A nut lowered with its hole's axis more than the clearance
  off the shaft's axis stops with its bottom face on the shaft end, 65 mm above the collar top. A displacement
  command that presses it there reads as the downward push (about 3 N per mm of commanded displacement) together
  with the sideways force the tracking law adds to it (section 5, Compliance), which the friction on the shaft
  end holds (0.3 to 0.7; a larger share slides the nut over the end). The push's line of action lies on the part
  of the shaft end beside the hole, 13 mm or more from the hole's axis, on the side where the shaft's axis is;
  because two faces that do not meet flat touch at an edge, it can lie up to about a quarter turn to either side
  of that direction. The wrist reads the push and its moment about the sensor (torque = lever x force), which is
  97 mm above the grip site.
- **What the wrist reads, the hole over the shaft.** Once the hole's axis is within the clearance of the shaft's
  axis the nut goes down under the same push, until the pin stops it 39.5 mm above the collar (section 3).
- **What the wrist reads, a nut that binds.** A nut tilted against the shaft by more than the fit allows stops
  part of the way down under the same push that moved it before; the two contacts' couple reaches the wrist as a
  torque about a horizontal axis.
