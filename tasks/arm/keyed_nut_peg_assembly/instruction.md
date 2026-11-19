# Bayonet nut kit

A Franka Panda arm stands at a table with three upright pegs and three nuts of different colours: a
kit. The work order says which nut goes on which peg. Assemble the whole kit: pick up each nut by its
handle, fit it onto its peg, work it down until it sits on the peg's base and let go. The difficulty
is that each peg carries a pin and each nut has a **groove** in the wall of its hole, a small
bayonet maze: a nut only comes down while the pin travels along the groove, which means lowering it,
turning it about the shaft, lowering it again, turning it again. Grooves also have dead ends and
false entrances. Every episode brings new nuts whose grooves you have never seen. The outside of a
nut does not show its groove; the groove is in the wall of the hole, where you can see it only by
looking into the hole at a steep angle or at the nut's bottom face. Your controller has to work out
each groove within the episode, by looking into the hole, by feel with the arm's force/torque
sensor and the motion of the nut, or both, and then turn the nut through it. A kit counts only when
all three nuts are assembled.

You are the engineer who commissions the cell for this kit family. Deliver a controller that does
the job from camera images, joint readings and the wrist force/torque sensor, within the cell's
**takt time: 85 s (850 control steps) for a kit**. You have **8 hours** of wall time
for the whole job (studying the data sheet, commissioning on the cell, building and testing the
controller). Have your controller in `/home/user/submission/` well before the time is up. Your
sandbox has 4 CPU cores and 12 GB of memory (processes that exceed it are killed), no GPU and no
network.

You work the way a commissioning engineer does. You have the cell's data sheet (the kit's drawing
data and tolerances, the robot and its controller, the sensors, the cameras, the wire), the robot's
model, and a **commissioning window** on the cell: a fixed number of episodes on development kits,
through a service that answers what the cell gives a controller and nothing else (see
"Commissioning"). The cell's simulator, its model files and its state are not readable from your
account, and there is no rehearsal of the graded kits.

**The data sheet is a drawing, and no real kit or cell is exactly its drawing.** Every episode,
graded or commissioning, is a real kit on the cell as it is that day: each nut's groove walls,
floors, hole and friction, each peg's shaft and pin, the gripper pads' friction, the arm joints'
friction, the force/torque sensor's zero and its drift, each camera's mounting and intrinsics and
the lamps' brightness have their own actual values within the tolerances the data sheet states
(`datasheet.md`, section 11). The actual values are never told. The published camera calibration is
the drawing's, so a position taken from an image alone is off by millimetres, and **the nut fits
its shaft with a locating fit: 0.5 mm of radial clearance on the drawing (0.35 to 0.65 mm on real
parts), square edges, no lead-in chamfer**. A nut cannot be put on a shaft from image positions
alone (`datasheet.md`, section 12: the fit, the cell's compliance and what the wrist sensor reads
when a nut meets a shaft). A model you build from the data sheet is the drawing's cell; the
commissioning service is where you meet real ones.

## What is in your home directory

| Path | What it is |
|---|---|
| `datasheet.md` | The cell's data sheet: frames and units, the pegs' and the nuts' drawing data (geometry, the groove family and how it is drawn, mass, friction, colours), how a kit is laid out, the robot and its controller (action format, tracking law, compliance, gripper), the sensors, the cameras with their published calibration, the safety stops, the takt, the commissioning service's wire, the tolerances within which every real kit and the cell on the day differ from the drawing (section 11), and the fit of a nut on a shaft with what the wrist sensor reads when they meet (section 12). |
| `cell_data.py` | The data sheet's numbers, the tolerance table and a function that draws one set of real values within it, the camera matrices and a few geometry helpers (plain Python and NumPy; it contains no simulator). Copy it into your submission if your controller uses it. |
| `commission/client.py` | The client of the commissioning service: start an episode on a development kit, send actions, get observations (`Episode`, `status`). |
| `commission/run_kit.py` | Runs your controller over the grading wire in one commissioning episode, the way the grader runs it, and reports how the episode ended (see "Commissioning"). It does not judge the assembly. |
| `helpers/wire_skeleton.py` | Optional boilerplate for the grading wire (READY / reset / act / close, repeated step indices). |

Python 3.12 is `/opt/venv/bin/python3` (NumPy, SciPy, OpenCV (headless), Pillow, MuJoCo 3.16.0 and
the robosuite 1.5.2 asset files of the robot model, `robots/panda/robot.xml` and
`grippers/panda_gripper.xml` under
`/opt/venv/lib/python3.12/site-packages/robosuite/models/assets/`); call it by that path. The robot's
model is yours to simulate the arm with (MuJoCo renders offscreen on the CPU, `MUJOCO_GL=egl`, already
set; a simulation with rendering takes about 0.5 GB and one core, and the commissioning service's
episodes run on the same four cores).

## The scene (what is known about every episode)

- World frame: metres, radians; the table top is the plane z = 0.80, the Panda base stands at
  (-0.56, 0, 0.80) facing +x. Quaternions are (w, x, y, z).
- **Pegs.** Three pegs stand on the table. Each has a 50 x 50 x 10 mm base collar and on it a round
  shaft of radius 12.3 mm, 65 mm tall, with a square end and one **pin**: a small block on the shaft surface, 3 mm
  wide and 4 mm tall, reaching 17.2 mm from the shaft axis, centred 37.5 mm above the collar top.
  The pegs are identical except for 1, 2 or 3 dark bands near the top of the shaft. Each episode
  draws the peg positions (centres with x in [-0.02, 0.15], y in [-0.21, 0.21], at least 0.13 m
  apart) and their yaws (uniform over the full circle, which turns the pin with the peg) anew.
- **Nuts.** Each nut is a ring 45 mm tall (outer radius 28 mm) whose hole (inscribed radius 12.8 mm) fits over
  the shaft with 0.5 mm of radial clearance, with a handle bar (16 mm wide, from 25 to 41 mm above the nut's bottom face) reaching
  98 mm from the hole axis. The ring is 24 sectors (15 degrees each, counted counter-clockwise from
  the handle) by 9 layers (5 mm each). The groove is cut into the inner part of the hole wall (out
  to 19 mm from the axis). Its top layer is always solid, so the top face and the outside of the
  ring show nothing, and its entrances (the real one and any false ones) open only on the bottom
  face, which lies on the table. The groove is plain geometry in the hole wall, though: looking down
  into the hole at a steep angle, you see the far side of the wall below the top layer, where every
  cut cell is a recess about 6 mm deep, and the bottom face shows the entrances if you get to see
  it. Every nut's groove belongs to the same family, documented exactly in
  `datasheet.md` (section 3; a real nut has it within the tolerances of section 11): an entrance riser at the bottom, a first turn of 2 to 4 sectors either way, a
  riser, a second turn of 2 to 4 sectors either way and a last riser, all 2 sectors wide; turn
  channels may continue 1 to 3 sectors the other way into a dead end; and there may be up to two
  false entrances (risers as deep as the real one that lead nowhere). A channel is 30 degrees wide,
  but the solid cells beside it narrow the opening near the shaft: a nut resting on the pin drops
  into a riser only while the riser is within about +-5 degrees of the pin (+-4.5 to +-7 on real
  nuts). Resting on the pin, the nut's bottom is 39.5 mm above the collar; it can come down to
  24.5 mm in an entrance riser, to 9.5 mm after the first turn and riser, and to the collar after the
  second (each within about half a millimetre on real parts). Each episode places the three
  nuts flat on the table with their centres at x in [-0.24, -0.06], y in [-0.24, 0.24], at least
  0.12 m apart, handles clear of the other nuts, each with its yaw (direction of the handle) equal to
  its azimuth seen from the robot base plus 135 degrees plus a uniform offset within +-60 degrees.
- The two fingers move together, symmetrically about the hand's axis, so a gripped part is centred
  between them; across the closing direction and in rotation it is held by the pads' friction only,
  so a held nut sags about the grip under its own weight. The hand is wide enough to touch a
  neighbouring peg if it passes close to it.
- Every graded episode has its own kit, drawn from the family (`datasheet.md`, sections 3 and 4)
  with a seed you do not know: three nuts of three different colours (blue, green, orange, pink,
  purple, red, teal or yellow), each with its own groove, mass (0.05 to 0.15 kg) and friction against
  a peg (0.3 to 0.7), and a random assignment of the nuts to the pegs; and its own actual values
  within the tolerances, for the kit and for the cell. (The graded kits are draws that a reference
  controller given the real grooves and poses completed with this robot within the takt; it needs
  about 640 to 800 steps for a kit.) Everything else (robot, controller, action and observation formats,
  cameras, sensor noise, physics, layout rules, tolerances, the safety stops, the takt) is the same
  as on the commissioning service's cell.
- The **work order** is in every observation as `order`: a list of `{"colour": name, "peg_bands":
  1 | 2 | 3}`, one entry per nut, each peg used once. You may assemble the nuts in any sequence.

## Observation and action

Observation (a JSON object): `t` (step index), `order`, `q`, `dq` (7 arm joints), `ee_pos` (gripper site, world frame, m), `ee_quat`
(w, x, y, z), `gripper` (finger opening, m), `wrench` (force N and torque N m at the wrist sensor,
world frame, with noise, a per-episode zero offset and a slow drift of that zero), `T_world_wrist` (4 x 4 pose of the wrist
camera, OpenCV convention), and the RGB images `left`, `right` (fixed cameras, 320 x 240) and
`wrist` (256 x 192), each `{"dtype": "uint8", "shape": [H, W, 3], "b64": <raw row-major bytes,
base64>}`. An image is empty (shape `[0, 0, 3]`) unless the previous action asked for it; the
first observation of an episode carries all three. The published camera intrinsics and fixed-camera
poses: `datasheet.md` (section 7, which also says what the images show: the cell in its laboratory
under a warm light) and `cell_data.py` (`fixed_camera_matrices`, `WRIST`); the
principal point is drawn at (W/2, H/2), so the centre of pixel (u, v) is at (u + 0.5, v + 0.5). The
real cameras are within their tolerances of that, and `T_world_wrist` is the hand's pose with the
drawn mount. There is no depth image and no object state of any kind.

Action: a list of 8 numbers `[dx, dy, dz, drx, dry, drz, grip, cams]` at 10 Hz: a gripper-site
displacement (world frame, clipped to 0.02 m), a world-frame rotation vector (clipped to 0.15
rad), the gripper command (-1 open ... +1 closed; the finger force is capped at 20 N) and a camera
bit mask for the next observation (1 = left, 2 = right, 4 = wrist). How the arm tracks an action:
`datasheet.md` (section 5). An episode lasts at most **850 steps** (85 s simulated): the takt time
for a kit.

## Deliverable and grading

Leave a controller in `/home/user/submission/` with an executable `run.sh`. The grader starts it
as `bash run.sh` with that folder as working directory, as your user, with no network (only
AF_UNIX sockets work). Everything it needs must be inside that folder (at most 5000 files and
200 MiB; a larger folder scores 0), and the controller may write there; nothing else of your home
is available when it is graded. Protocol: one JSON object per line on stdin/stdout. Keep stdout
for protocol messages and send diagnostics to stderr.

```
                                  <- READY                  (when you are loaded)
{"type": "reset"}                 -> {"ok": true}
{"type": "act", "t": 0, "obs": {...}}
                                  -> {"action": [8 numbers]}
{"type": "close"}                 -> {"ok": true}, then exit
```

Deadlines: print `READY` within 120 s of launch, and answer every request within 30 s. That 30 s
is a single budget per request, timed from when the request is sent: blank or partial lines count
against the same 30 s and do not buy more time, so emit nothing but your one JSON reply. A missed
deadline, a reply that is not one JSON object with an `action` of 8 numbers (finite, the last one an
integer camera mask 0 to 7), a reply line longer than 64 KiB, or a crash aborts the episode, which
then scores 0.

**Every episode is graded from scratch.** Before each episode the grader stops every process of
your account, deletes every file your account owns outside the submission folder (in `/tmp`,
`/var/tmp`, `/dev/shm`, `/run`, `/usr/local`, your home), replaces the submission folder with a
fresh copy of the one you left, and launches `bash run.sh` again (it must print `READY` again).
Nothing your controller writes or learns during one graded episode is available in the next one:
whatever it should know has to be in the submission you leave. So `run.sh` has to
work when started again: no fixed ports, PID files or locks that an earlier run leaves behind.
Processes your controller leaves behind are stopped; the whole run scores 0 only if one of them
cannot be stopped, or if a process of your account is found holding an internet connection (only
possible where the network filter could not be installed).

`t` is the step index within the episode, starting at 0 after each `reset`. A step may be queried
more than once with the same `t`; answer each request from the observation it carries. Repeated
queries do not advance the episode, so do not count requests as steps; use `t`. At about 1 % of
the steps the grader asks the same `t` twice, once with the real observation and once with a
slightly perturbed copy (joint readings, gripper and wrist-camera pose, wrench; the images are
unchanged), in a hidden order, and applies only the reply to the real one. The grader records
whether the two replies differ; this record does not change your score. Your controller may keep
state between the steps of an episode (`helpers/wire_skeleton.py` shows one way to do that safely
with repeated `t`).

The grader runs **10 episodes** in a fixed hidden order, each with its own hidden kit and layout. A
nut counts as **assembled** when it sits on **its** peg's collar (its bottom within 3 mm of the collar
top, tilt below 5 degrees) for 10 consecutive steps with the fingers open at least 60 mm and not
touching it, and it still sits there at the end of the episode. A nut on the wrong peg, left partway
down its groove or still held is not assembled. The groove walls are solid: forcing a pin through
them (its centre more than 5 degrees sideways or 1.5 mm up or down inside a solid part of the real
nut's wall, for 3 consecutive steps, while the nut is on its peg) ruins the nut and ends the episode.
A kit is **complete** when all three nuts are assembled; the episode ends at 850 steps (the takt) or
as soon as the kit is complete: a kit that is not complete when the takt runs out scores 0. Your result is
the **fraction of the 10 kits completed**. (The grader also records how many nuts each kit got
assembled and how far each nut got, for diagnosis only; they do not count.) An episode in which the
controller crashes, answers late or malformed, or makes the simulation unstable (any arm joint
faster than 8 rad/s or a nut faster than 5 m/s) scores 0.

**Time.** Every kit has a **controller budget of 600 s**: the time the grader waits for your
controller (from its launch to `READY` and the `reset` reply, then every reply, each timed from when
the request is sent) plus a fixed charge for every image you ask for: **0.45 s** for a `left` or
`right` image and **0.35 s** for a `wrist` image (about what rendering one takes on the grading
machine; the three images of an episode's first observation are free). The simulation's own time is
not charged, so the grading machine's speed does not change your result. When a kit's budget runs
out, its episode stops at that step and is judged as it stands (an incomplete kit scores 0). For
scale: on the grading machine a simulation step without images takes about 0.03 s of the grader's
time, and the reference controller uses less than 1 s of its budget per kit. As a safety net, the whole
grading run stops after 12000 s of wall time (within these budgets a run takes at most about 2
hours); the episode under way and the ones after it would then score 0. Only the final controller
is scored.

## Commissioning (no rehearsal of the graded kits)

Your controller is graded on ten kits it has never met, as a controller goes into production on kits
it has never met. Before that, like any commissioning engineer's, it can be run on the cell: the
commissioning service in your sandbox runs the cell on 16 development kits (`datasheet.md`, section
10).

* **What it is.** The cell of the graded episodes (the same robot and controller, sensors, cameras,
  kit family, layout rules, tolerances, safety stops, takt and controller budget) behind a socket:
  `reset(kit)` lays development kit 1 to 16 out and answers the first observation, `step(action)`
  applies an action and answers the next one, `close()` ends the episode. The development kits are
  draws from the kit family that the reference controller completed within the takt, like the
  graded kits. Each is a real kit: it has its own layout, its own actual values within the
  tolerances and its own state of the cell, and its episode starts the same way every time.
* **What it is not.** A simulator you can read or query, a rehearsal of the graded kits or a grader:
  no development kit is a graded kit, and the service answers no ground truth (no nut or peg pose,
  no groove), no score, no completion verdict and no debrief. It reports only how an episode ended:
  all 850 steps run, a safety stop (with the colour of the ruined nut, or the instability), the
  controller budget used up, 120 s without a request, or your `close`. Because it does not judge
  completion, an episode does not end when the kit is complete. How an episode went, you judge from
  what your program saw and recorded and from the cameras (`close(images=True)` returns the three
  cameras' images of the final state).
* **Its window.** **32 episodes** in your session, at most 2 at a time; an episode counts from its
  `reset`, whatever its length. That is the size of a real commissioning window: one working shift
  on the cell with a tray of 16 pre-series kits, at about a quarter of an hour per trial. The count
  is in `/run/keyedcell/commission_status.json` and does not grow. `commission/client.py` drives an
  episode from a script of your own; `commission/run_kit.py` runs your controller in one (below).
* **Whole runs.** `/opt/venv/bin/python3 /home/user/commission/run_kit.py --kit K` runs your
  controller in one episode the way the grader runs it: the same request shape, the same deadlines,
  a few repeated-`t` queries, a new controller process from a fresh copy of the submission, the same
  network filter and the same controller budget. It prints the steps run, how the episode ended,
  your controller's budget use and its image count (`--final-images DIR` saves the final state's
  images, `--record FILE` the observations' readings and the actions).
