# The skid cell: the integrator's manual

This is the integrator's description of the robot cell your program drives. Everything stated here as exact is exact;
the skid itself is described in `../equipment_manual.md`, the procedure in `../sop.md`, the wire in `cell_api.md`.

## 1. Frame

World frame = the skid's frame: z up, the floor at z = 0, the skid's front plane y = 0 (the skid occupies y >= 0, its
operable parts protrude towards -y), x along the skid from 0 to 5 m. The robot's carriage runs along x with the
Panda's base on the line y = -0.70; its floor track and lift mast stand behind the base (y = -1.16 to -1.02).

## 2. The robot

* **Panda.** MuJoCo Menagerie's `franka_emika_panda` with its hand (copy in `~/given/panda/`, Apache-2.0), on the lift
  of a carriage. The base (link0's origin) stands at (x, -0.70, lift) with its x axis along world +y: the arm faces the
  skid. Joint names `panda/joint1..7`, ranges and the home posture in `describe()`. The links are gravity-compensated.
* **Rail and lift.** The carriage moves along x in [0.30, 4.70] at up to 0.30 m/s; the lift sets the base height in
  [0.05, 0.60] m at up to 0.15 m/s (`command_rail`). Their drives are stiff position servos; `get_state()["rail"]` gives
  where they are. The carriage and the lift do not collide with anything.
* **Reach.** With the tool pointing horizontally into the skid (+y) the Panda reaches handles about 0.5 to 1.1 m above
  its base and 0.4 to 0.6 m in front of it; set the lift so the handle you work on is about 0.8 m above the base.
* **Control** (`set_control`, `command`): `joint` mode tracks joint targets (rate-limited, 1.5 rad/s); `cartesian` mode
  tracks a world-frame pose of the `panda/tcp` site with one damped-least-squares IK step per 10 ms tick on the
  commanded configuration (0.5 m/s, 1.5 rad/s). The joints are position servos (Menagerie gains: 4500 / 3500 / 2000
  N m/rad and 10 % of that in damping): a moving target is tracked about 0.1 s late, and `stiffness_scale` (0.2..1)
  scales their stiffness.
* **Tool.** `panda/tcp` lies between the finger pads, 0.1034 m from the hand along its +z (the approach axis); +y of
  the tcp is the fingers' closing axis. The fingertips end about 9 mm beyond the tcp along +z. The hand is about
  0.2 m wide along its closing axis. Width 0..0.08 m (`command(width=...)`), ramped at 0.05 m/s.
* **Wrist load** (`get_state()["arm"]["wrist_load"]`): the external wrench on the hand, estimated from the joint
  torques as a Franka's own estimate is (its `O_F_ext_hat_K`): `force_N` (3) at the tcp and `torque_Nm` (3) about the
  tcp, in world axes, as the surroundings push on the robot (a fingertip pressed into a button along +y reads a force
  along -y). It feels every contact of the robot with the skid or the floor (fingers, hand and links: a contact on an
  upper link shows as the wrench at the tcp that loads the joints the same way) and the load of a held handle; it
  reads with or without a grip. Like the real estimate it is not exact: about 0.3 N and 0.03 N m of noise per reading,
  and an offset of up to about 2 N and 0.2 N m per axis that changes with the arm's posture (the dynamic model's
  error: it stays put while the arm stands still and moves by up to about 1 N and 0.1 N m while the hand turns a valve
  through 30 degrees). Read it with the hand free and still just before a contact and subtract that reading (re-zero),
  as one does on the real robot.
* **Wrist camera** `wrist` rides on the hand (section 5).

## 3. The grip rule (how a handle is held)

A valve handle is held by the cell's grip rule, not by friction. When the fingers close (a `width` below the current
opening) and, while closing, the opening reaches a handle's width + 6 mm with that handle inside the pinch box, the
handle's moving part (a handwheel, a lever, a knob, the selector's wing) is attached rigidly to the hand at the pose of
that instant (the event `grasped`), and the fingers stay at the handle's width. The pinch box, in the tcp frame: |x| <=
12 mm (across the pads), |y| <= half the opening (along the closing axis), -12 mm <= z <= +14 mm (along the approach).
The tcp's approach axis (+z) must lie within 30 degrees of the handle's approach direction (+y, into the skid), and its
closing axis within 30 degrees of the handle's pinch axis:

| handle | where | pinch axis | width |
|---|---|---|---|
| handwheel spoke | middle of each spoke | across the spoke, in the wheel's plane | 13 mm |
| handwheel rim | on the rim midway between two spokes | along the radius | 18 mm |
| lever | at 35, 55, 72 and 90 % of the lever's length | across the lever, in its plane of turning | 20-26 mm |
| needle-valve knob | the knob's centre | any axis across the stem | 38 mm |
| needle-valve knob (small) | the knob's centre | any axis across the stem | 32 mm |
| selector wing | the wing's middle | across the wing | 15 mm |

The held part moves with the hand only as its own joint allows (a handwheel turns about its stem, nothing else): move
the hand on that circle. The grip carries at most 60 N of force, and 3 N m of moment about the held part's pivot
(where its stem's axis meets the handle's plane: a handwheel's hub, a lever's nut, the knob's or the wing's centre):
a load above either for 30 ms slips the handle out of the hand (`slipped`). Opening the fingers wider than the handle
releases it (`released`). The events do not name the device. An 80 mm opening straddling a spoke of a small wheel
touches the rim: open just enough. Buttons are pushed with the fingertips (fingers closed), not held.

**Turning tolerance.** The grip is rigid, so the hand must follow the handle's own circle about the stem: centre the
circle within about 1 mm of the stem's axis for a long stroke (2 mm for a short one). A hand whose circle is off the
axis pulls the handle across its stem, by about 1 N more for every degree turned at 2-3 mm off, and the grip slips
when the moment about its pivot passes 3 N m. Measured on development sites of the series, turning 30 degrees (the
selector 45) at 35 degrees a second in 3-degree steps with the circle's centre off the axis:

| off the axis | handwheel | lever 160-200 mm (V-101, V-103) | other levers | needle valve | selector |
|---|---|---|---|---|---|
| 1 mm | 30 degrees and on | slips after 19-22 degrees | 30 degrees and on | 30 degrees and on | to HAND |
| 2 mm | slips after 17-30 degrees | slips after 11-14 degrees | 16-30 degrees (V-104: 25) | 30 degrees and on | to HAND |
| 3 mm | slips after 12-21 degrees | slips after 8-10 degrees | slips after 10-22 degrees | about 30 degrees | about 43 degrees |

The wrist load shows the error growing before the slip (the force across the stem, and the moment about axes across
it), so a program can steer the circle by it or let go and grip again before it slips.

**What a turn reads.** Take the wrist load's moment about the stem's axis (the moment about the tcp plus the moment of
the force about the axis: for a grip at radius r on the handle, mostly r times the force along the turning
direction). A valve turned at up to 35 degrees a second takes its turning effort (`equipment_manual.md` section 2)
plus its viscous part, at most the effort plus 0.25 N m: on development sites, turned at 35 degrees a second in
3-degree steps (about 17 degrees a second on average), the moment about the stem read the effort plus 0.01-0.09 N m,
never more than 1.8 N m on any valve (the selector up to 0.5 N m against its detent). At an end stop the valve
stops and the hand does not: within the next 3 degrees of commanded turning at stiffness 0.4 the moment about the
stem jumps past the grip's 3 N m (to 7-14 N m on handwheels and levers, about 3.8 N m on needle valves) and the grip
slips. So a moment about the stem above about 2.2 N m while the handle stops following the hand means an end stop (or
a hand off the circle: the force across the stem tells which), not a stiff valve.

## 4. Safety

The cell watches the contacts between the robot and the skid (and the floor) every 10 ms tick, as a collaborative
robot's collision detection does.

* **Protective stop.** A robot link or finger pressing on anything of the skid but a handle or a button with more
  than 40 N, or on a handle or a button with more than 150 N, for 3 consecutive ticks stops the robot: the arm and the
  carriage halt where they are (their targets become the measured pose), `step` reports
  `protective_stop:<robot link>:<force>N` (the robot's link; never what it touched), and for 5 s of simulated time
  every motion command (`command` with a joint or Cartesian target, `command_rail`) answers
  `{"ok": false, "error": "protective_stop", "remaining_s": ...}`; the gripper (`command(width=...)` alone) and
  `set_control` still work. Then `step` reports `protective_stop_cleared` and the robot moves again when commanded
  (it does not resume a motion by itself). `get_state()["arm"]["protective_stop"]` gives `active`, `remaining_s`, the
  shift's `count` and `locked_out`. A press that was on when the stop came, or that comes during its 5 s, stops the
  robot again only once its force has dropped below the limit: back off before you push again.
* **Out of service.** Contact of more than 400 N for 3 consecutive ticks (an impact that can bend a stub or crack a
  sight glass), or a sixth protective stop in one shift (a robot that keeps stopping is taken out of service and
  checked), locks the robot out for the rest of the shift (motion answers `locked_out`; `safety:damage:...` or
  `safety:locked_out` in `step`'s events) and the episode scores 0 (`sop.md` sections 5 and 6).

The robot never collides with itself, with the carriage or with a part it holds.

## 5. Cameras

| name | mount | size | fovy |
|---|---|---|---|
| `overview` | fixed, behind the rail, high: the whole skid. In its frame the analyser's digits are about 4 px tall (legible in an enlarged crop), FI-105's scale is about 67 px long and LG-201's about 245 px, so the rotameter's float and LG-201's flap column show where they stand; a dial is 22-25 px across and a tag plate about 6 px tall, too small for a pointer or a tag | 1280 x 720 | 50 |
| `wrist` | on `panda/hand`: 0.05 m behind the tcp and 0.045 m to the hand's +x side, looking at the point 0.15 m ahead of the tcp; image up = the closing axis | 640 x 480 | 70 |

`describe()["cameras"]` gives the exact intrinsics (fx = fy = (H/2) / tan(fovy/2), cx = W/2 - 0.5, cy = H/2 - 0.5,
pixel (u, v) = (column, row)), the fixed camera's exact pose `T_world_cam` and the wrist camera's mount `T_body_cam`
(OpenCV convention: x right, y down, z forward). A wrist frame carries its `T_world_cam` at the frame's instant. The
colour images carry a little sensor noise (sigma 2 levels); depth is the planar distance along the optical axis in
millimetres (0 = no reading; 2 % dropout, 1.5 mm noise, 0.1-6 m). The colour images are physically based renders of
the skid in its water-treatment room: the skid's back wall runs on as the room's lining wall, drums, bagged chemicals
and other stores stand on the floor beside the skid and the robot's rail, and two ceiling panels over the rail and the
room's own lamps light the scene; lit lamps and the analyser's lit segments glow. Lights vary a little from skid to
skid. Depth reads the same surfaces as the colour image, the room included.

## 6. Time and budgets

The cell's clock advances only inside `step` (10 ms per tick); the process (flows, levels, alarms, the window's events)
advances with it. Your program's own computing time is not simulated time but is capped per episode (`cell_api.md`).
The schedule: the window opens at 600 s, the shift ends at 1500 s, the episode ends at 1800 s.

## 7. The client library

`~/given/lib/skidcell/client.py` is the one part of the cell your program imports: the proxy (`Cell`), `decode_frame`,
the program launcher the grader uses and the commissioning connection (section 8). The cell itself (the skid, its
process, the robot's physics, the scoring) runs in the grader and in the commissioning service, never in your
account. What you know of a site before its shift is what this manual, `cell_api.md`, `cell_config.json`, the
operating procedure, the P&ID, the equipment manual and the instrument catalog say, and what commissioning on the
series' sister sites showed you; on the shift, your program learns the rest through the cameras and the robot's
state, as an operator would.

## 8. Commissioning on the series' sister sites

A robot program is not handed over before it has run on a real skid: like any automation engineer's, yours can be
commissioned on sister sites of the series before it runs the sites it is graded on. The commissioning service in
your sandbox runs the series' cell on 9 development sites: skids of the same series, each numbered and built within
the equipment manual (between them they show every build option of the manual, every gauge scale and every abnormal
situation of the SOP), none of them a graded site.

* **What it is.** The cell of the graded episodes (the robot with its controller, the grip rule, the wrist load, the
  safety watch and its protective stops, the cameras' frames, the process with its faults, the budgets) on the wire
  of `cell_api.md`, plus `reset(site)`, which starts a shift on development site `site` (1-9) from the state its
  previous shift left. A site's shift is the same each time you start it.
* **What it is not.** A simulator you can read or query: it answers no ground truth (no device pose, no valve
  position, no process value), no score and no debrief. How a shift went, you learn from what your program saw,
  logged and saved. The graded sites are other sites of the series, with their own numbering, builds, leftovers,
  faults and timings.
* **The window.** 12 commissioning shifts in your session (a shift counts from its `reset`), at most 2 at a time.
  Each runs under the graded episode's budgets (1800 s of simulated time, 1200 s of program time, 600 camera frames,
  180 s of silence) and the service ends it when one runs out (`{"event": "shift_over", "reason": ...}`, then the
  connection closes). The count: `/run/skidcell/commission_status.json`. It does not grow: plan your shifts.
* **How.** `skidcell.client.connect_commissioning(site)` returns a `Cell` whose shift has begun (one connection, one
  shift): a script that jogs the robot to a device, grabs frames and records what it sees is a commissioning
  engineer's teach-in. `/opt/venv/bin/python3 ~/commission/run_shift.py --site N` runs your whole program in a shift,
  the way the grader runs it, and prints how the shift ended, the frames, the grips, slips and protective stops
  (`--keep DIR` keeps what your program wrote in its HOME and TMPDIR, `--transcript FILE` the wire).
