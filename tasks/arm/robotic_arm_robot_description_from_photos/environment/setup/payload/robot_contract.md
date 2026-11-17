# Robot description contract (the chain's robot-description hand-off)

The files you deliver in `/home/user/submission/` are consumed as they are by the next stages (control, environment):
they must follow this contract exactly. The grader checks every rule of the "gate" list below before it scores anything.

## Files

| file | content |
|---|---|
| `robot.xml` | the MJCF of the arm (MuJoCo 3.14): the seven bodies, six hinge joints, two sites, visual and collision geometry, inertials, six position actuators, the keyframe `home` |
| `scene.xml` | `<include file="robot.xml"/>`, a table plane at z = 0 and a light (nothing of the robot outside `robot.xml`) |
| `assets/` | every mesh (STL / OBJ) `robot.xml` references; paths relative to the submission; no symbolic links |
| `calibration.json` | `{"joints": {"<joint>": {"sign": 1 or -1, "zero_offset_rad": <float>}, ...}}` for all six joints |
| `robot.urdf`, `estimate.json`, `build.py` | optional; recorded (never executed, never scored) |

## Names

- Bodies: `base`, `shoulder`, `upper_arm`, `forearm`, `wrist`, `gripper`, `jaw` (parent chain in this order: `base` is welded
  to the world, `jaw` is the child of `gripper`). Extra bodies WITHOUT joints are allowed; each belongs to its nearest
  contract ancestor (its geoms and mass count for that link).
- Joints (hinges, in this chain order, no other joint): `base_yaw` (moves `shoulder`), `shoulder_pitch` (`upper_arm`),
  `elbow_pitch` (`forearm`), `wrist_pitch` (`wrist`), `wrist_roll` (`gripper`), `jaw` (`jaw`).
- Actuators: `act_base_yaw`, `act_shoulder_pitch`, `act_elbow_pitch`, `act_wrist_pitch`, `act_wrist_roll`, `act_jaw`
  (position servos on their joints, gear 1).
- Sites: `tcp` (rigid on `gripper`), `jaw_tip` (rigid on `jaw`).
- Servo ids on the bus: 1 base_yaw, 2 shoulder_pitch, 3 elbow_pitch, 4 wrist_pitch, 5 wrist_roll, 6 jaw.

## Frames, angles, calibration

- World frame: the frame of `parts/base_in_world.stl`: origin on the mounting plane (the platform the base is screwed to), nominally on
  the base-yaw axis (the build's errors of about a millimetre apply to every joint, the yaw axis included), z up, +y = the
  base's forward direction (the direction the arm reaches at design yaw 0; reaching forward is a negative shoulder pitch), +x = the
  pitch-axis direction at yaw 0. SI units everywhere in `robot.xml` (metres, radians, kilograms).
- `robot.xml` states `<compiler angle="radian">` explicitly (every angle and joint range in radians).
- Design angle q = the `qpos` of a joint. q = 0 is the straight-up configuration (the shoulder-to-elbow and
  elbow-to-wrist axis offsets vertical, the roll axis vertical, the tool approach +z; figure `contract/zero_pose.png`);
  positive q is right-handed about the joint axis. No `ref` attribute on any joint (every joint's qpos0 is 0).
- Bus angle b = the servo's reading in radians: b = (ticks - 2048) x 2 pi / 4096 (ticks 0 .. 4095).
- `calibration.json`, per joint: b = sign x (q - zero_offset_rad), i.e. q = sign x b + zero_offset_rad.
- Only the composite map bus reading -> pose of the real arm is graded (see instruction.md, "How the submission is graded"): the split between your joint-axis
  directions, your zero configuration and your offsets is free as long as the composite map is right. The contract
  conventions above are what the next stages assume; keep them.
- Joint ranges (`range`, limited): the servo's firmware limits (`servo_config.json`: EEPROM min / max angle-limit
  registers in ticks) expressed in design angles through your calibration.

## Tool points

- `tcp` (rigid on `gripper`): on the fixed finger's gripping face (the face that looks at the moving jaw), on the finger's
  mid-line (the middle of its width), 10.0 mm back along the finger from the fingertip; site frame +z = approach (along
  the finger, out of the gripper), +x = the face normal pointing towards the moving jaw. The gripping face is stepped: it
  is flat over the finger's top 10 mm and steps back below that; the site lies on the PLANE of that flat tip face, i.e. on
  the step's edge. Figure `contract/tcp_definition.png`.
- `jaw_tip` (rigid on `jaw`): the same construction on the moving jaw's gripping face, +x pointing towards the fixed finger.

## Actuators and joint parameters (per joint; one servo drives each joint)

- `act_<joint>`: a position servo (`<position kp=... kv=...>`, or an equivalent `<general>` with gaintype fixed, biastype
  affine, biasprm[1] = -kp) on the joint, gear 1, `ctrlrange` = the joint range (+-0.001 rad), `forcerange` = [-f, +f]
  with 0.5 x 2.94 N m <= f <= 2.94 N m (the servo's stall torque at 12 V).
- kp in [250, 2500] N m / rad, kv in [0.5, 8] N m s / rad.
- Joint `armature` in [0.005, 0.1] kg m^2, `damping` in [0.1, 3.0] N m s / rad, `frictionloss` in [0.01, 0.25] N m.

## MJCF rules ("MJCF is data")

- Allowed: the usual body / geom / joint / site / inertial / asset (mesh, texture, material) / default (nested classes and
  `childclass` are fine) / contact (`exclude` between robot bodies) / actuator / keyframe / sensor elements.
- Not allowed: `<plugin>`, `<extension>`, `<replicate>`, `<frame>`, `<attach>`, tendons, flex / composite / skin elements,
  `<equality>`, mocap or gravcomp bodies, `<include>` or any `file=` outside the submission, DOCTYPE / ENTITY declarations.
  Every `file=` and every `<compiler>` `meshdir` / `texturedir` / `assetdir` is a relative path without a `..` component
  (so it stays inside the submission whatever directory MuJoCo resolves it against).
  `<option>` only `timestep` in [0.0005, 0.005] and `gravity="0 0 -9.81"`; friction / solref / solimp / margin inside the
  usual bands (MuJoCo's defaults pass).
- Both `scene.xml` and `robot.xml` compile without any warning.
- Caps: <= 200 MB in total (the grader counts the larger of the apparent size and the allocated disk space), at most
  20 000 files and directories, regular files only; <= 400 000 mesh triangles in total; collision geoms (contype or
  conaffinity non-zero) <= 48 per link and <= 320 in total, <= 50 000 faces per collision mesh.

## Gate `model_loads` (the structural rules listed here; the actuator and joint-range rules above are scored in
`limits_actuators`, not gated)

Files present and regular; the whitelist on `scene.xml` (with its include) and on `robot.xml` before anything is compiled;
the relative-path rule for `file=` / `meshdir` / `texturedir` / `assetdir`;
the explicit `<compiler angle="radian">`; the include; zero-warning compiles of both files; exactly the six contract hinges
in chain order, nq = nv = 6, no `ref`; the seven bodies with the contract parents (base welded to the world, only the
contract bodies carry joints, each joint on its body); `tcp` on the gripper link and `jaw_tip` on the jaw link; the keyframe
`home`; exactly the six actuators, each a position servo on its joint with gear 1; `calibration.json` valid (six joints,
sign +-1, finite offsets); the caps; nothing in the submission written after your session ends. `check_submission.py`
runs every rule of this gate on any folder (no ground truth is involved) except the last one, which only the grader can
judge, and the entry-count and allocated-size caps, which the grader measures when it takes the submission.
