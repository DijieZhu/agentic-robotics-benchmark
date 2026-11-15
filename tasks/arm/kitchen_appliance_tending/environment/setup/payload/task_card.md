# Task card: the cell, the units, the numbers

`appliance_env.py` (your dev sim) is the exact environment the lab and the grader run; `episode.py` is the exact
request and credit logic. Read their docstrings. This card collects the numbers.

## Frames and the robot

- World frame: z up, metres, radians, seconds; quaternions are (w, x, y, z).
- A Franka Panda (robosuite 1.5.2 model) stands on a fixed pedestal; its base is at (0, 0, 0.65) and faces +x.
- Control: 10 Hz (`CONTROL_DT` 0.1 s, 50 physics steps of 2 ms). An operational-space controller tracks a target
  the action sets relative to the gripper site's current pose: translation clipped to 0.03 m, rotation vector to
  0.2 rad per step.
- Gripper: `grip` -1 = fully open (80 mm between the fingers) ... +1 = closed; 20 N per finger at most.
- Start pose: every episode starts from the same joint pose for a given unit (the gripper about 0.45 m in front of
  the base and 0.55 m above it, pointing at the unit's front).

## The units

- One appliance per episode, its front face towards the robot (-x). Nominal placement: the centre of the front
  face 0.55 m in front of the robot base (x), centred (y). Each episode adds a uniform offset of up to +-0.03 m in
  x, +-0.04 m in y and +-10 degrees of yaw about the vertical; nothing reports it.
- Kinds and mounts: a **cooktop** set into a counter (counter top at z = 0.92), a **microwave** standing on the
  counter, a wall **oven** in a cabinet column (oven bottom at z = 0.55).
- Controls: cooktop burner knobs (vertical axis, on the top surface); microwave start and stop buttons (on the
  front panel; RoboCasa's rule: touching start switches it on, touching stop switches it off, opening the door
  switches it off); oven temperature knob (horizontal axis, on the front panel).
- Knob physics: friction loss 0.20 N m, armature 0.002 kg m^2. A burner is on while its knob angle `a` satisfies
  0.35 <= a mod 2 pi <= 2 pi - 0.35 (a flame marker shows while it is on). The oven is preheated while its
  temperature knob is at 0.25 or more of its range (normalized as RoboCasa does).
- Mechanisms (`appliance_env.QUIRKS`; at most one per unit; a unit spec's `quirks`): `push_to_turn` (a knob unlocks
  while the gripper presses on it along its axis with `PUSH_FORCE` 10 N or more: the size of the sum of the gripper's
  contact forces on the knob along the knob's axis, read once per control step after its physics steps; once that
  drops below `RELOCK_FORCE` 5 N it locks again over about 1 s: friction 4 N m plus heavy damping), `reversed` (every
  knob's joint axis is flipped: the same joint values, the opposite rotation), `hold_start` (the start button must be
  held `HOLD_START_STEPS` 15 control steps). The practice units carry one each (p0 push_to_turn, p1 hold_start, p2
  reversed); the hidden units' are not reported.
- Practice units in your home: `units.json` (ids p0, p1, p2) and `units/` (their models). Hidden units: u0, u3,
  u6 (cooktops), u1, u4, u7 (microwaves), u2, u5, u8 (ovens); you reach them only through the lab.
- Settings: a burner's setting is its knob angle over its on-range, (a - 0.35) / (stop - 0.35), in thirds (low,
  medium, high); the oven's heat is its normalized temperature knob in 0.25-0.5 (low), 0.5-0.75 (medium) or
  0.75-1.0 (high).
- State shown on the unit: a burner's flame while it is on (RoboCasa's; its radius grows from 0.6x to 1.4x from
  the lowest to the highest setting); a lit amber display bar (10 x 2 cm) while a
  microwave runs and a lit red-orange lamp (3 x 3 cm) while an oven is preheated, both on the front face 3 cm
  below the top edge of its visible front (dark grey when off). Visual only: no contact, no mass.

```python
import sys; sys.path.insert(0, "/home/user")
import json, appliance_env as E, episode as EP
p0 = json.load(open("/home/user/units.json"))["p0"]        # model, dir, quirks
spec = E.unit_spec(p0["model"], files=E.read_unit_files("/home/user/" + p0["dir"]), quirks=p0["quirks"])
ep = EP.Episode(E.ApplianceEnv(spec, None), seed=1)
obs = ep.reset()                       # obs["request_text"] is the open request
obs, done, info = ep.step([0, 0, 0, 0, 0, 0, -1, 7])
ep.env.unit_state(); ep.env.truth()    # ground truth, dev sim only
```

## Cameras

- `left` and `right`: fixed, 320 x 240, beside and above the robot's shoulders, looking at the unit's front.
- `wrist`: on the hand, 256 x 192.
- Every observation carries the intrinsics (`K_left`, `K_right`, `K_wrist`, pinhole, no distortion) and the poses
  (`T_world_left`, `T_world_right`, `T_world_wrist`; 4 x 4, OpenCV convention: x right, y down, z forward). The
  fixed cameras' nominal poses depend on the unit's model; each episode offsets them by up to +-4 cm and about
  +-3 degrees (the observation carries the actual pose).
- Lighting changes per episode: the scene light's position, aim, strength (0.5-1.2x) and tint (+-15 % per
  channel), and the headlight; nothing reports it.
- Images are 8-bit RGB with sensor noise (sigma 2 levels). An image is empty (shape [0, 0, 3]) unless the previous
  action asked for it; the first observation of an episode carries all three. Rendering costs wall time, and in a
  grading run it counts against the grader's 6000 s: with the task's Python on 4 idle CPUs about 35 ms per wrist
  image and 50 ms per fixed-camera image, against about 10 ms for a step without an image (up to twice that on a
  busy machine).

## Sensors

- `wrench`: force (N) and torque (N m) at the wrist sensor, world frame, with noise (force sigma 0.3 N, torque
  sigma 0.03 N m).
- Decimals: `q`, `ee_pos`, `ee_quat` and `T_world_wrist` carry 6, `wrench` 4 (`dq`, `gripper` and the fixed
  cameras' poses are not rounded).

## Episodes

- Initial state (drawn per episode): each burner on with probability 0.3 (knob at 0.8-1.6 rad), the microwave
  running with probability 0.5, the oven preheated with probability 0.4 (knob at 0.30-0.50 of its range).
- Three requests, one at a time, each valid in the state it opens on, each with a deadline of 1000 control steps
  (1500 for a compositional cooktop request: two burners on, two off, or all off). A request's category is drawn
  first (simple 0.3, compositional 0.35, setting 0.35, over the categories valid at the time), then a request.
- Credit (`episode.py`, the instruction's Credit section): "reached" is decided when a request closes (the state
  has held its last 3 steps), "completed" needs 20 steps held with the gripper 0.10 m clear of the named controls
  and every other control's on / off (a door's closed / open) state as it was; a setting request whose control
  stands in a wrong setting for 20 steps with the gripper clear can no longer be completed.
- Episodes end when the third request closes; the simulation is unstable (and the episode scores 0) if an arm
  joint moves faster than 8 rad/s, a burner knob or a door joint faster than 30 rad/s (`MAX_UNIT_SPEED`; the oven's
  knobs and the microwave's buttons are not checked), or MuJoCo raises a bad acceleration, position or velocity
  warning (MuJoCo's automatic reset is off).
- Graded episodes: a fixed hidden set of 18 (2 per scored unit), the same for every submission, in a hidden order;
  the lab draws a fresh episode at every trial. A grading run in which every request runs to its deadline has
  56,000 control steps (at most 63,000).
- Sandbox: 4 CPUs and 12 GB of memory in the lab phase and at grading (`OMP_NUM_THREADS=1` is set, so `nproc`
  prints 1).
