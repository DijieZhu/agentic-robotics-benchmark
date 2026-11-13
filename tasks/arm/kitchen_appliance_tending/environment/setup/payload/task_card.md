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
- Practice units in your home: `units.json` (ids p0, p1, p2) and `units/` (their models). Hidden units: u0 and
  u3 (cooktops), u1 and u4 (microwaves), u2 and u5 (ovens); you reach them only through the lab.

```python
import sys; sys.path.insert(0, "/home/user")
import appliance_env as E, episode as EP
spec = E.unit_spec("stovetops/Stovetop018", files=E.read_unit_files("/home/user/units/Stovetop018"))
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
  fixed cameras' poses depend on the unit's model (not on the episode).
- Images are 8-bit RGB with sensor noise (sigma 2 levels). An image is empty (shape [0, 0, 3]) unless the previous
  action asked for it; the first observation of an episode carries all three. Rendering costs wall time.

## Sensors

- `wrench`: force (N) and torque (N m) at the wrist sensor, world frame, with noise (force sigma 0.3 N, torque
  sigma 0.03 N m).

## Episodes

- Initial state (drawn per episode): each burner on with probability 0.3 (knob at 0.8-1.6 rad), the microwave
  running with probability 0.5, the oven preheated with probability 0.4 (knob at 0.30-0.50 of its range).
- Three requests, one at a time, each valid in the state it opens on, each with a deadline of 600 control steps.
- Episodes end when the third request closes; the simulation is unstable (and the episode scores 0) if an arm
  joint moves faster than 8 rad/s or a unit joint faster than 30 rad/s.
