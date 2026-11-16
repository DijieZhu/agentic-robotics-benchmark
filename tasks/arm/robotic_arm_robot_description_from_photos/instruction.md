# Write the MuJoCo description of the arm in the photos

You are the robotics engineer of a lab that inherited a customised build of a published six-servo desktop arm. The
builder printed the published parts with changes (the two long beams lengthened or shortened, the base and shoulder
possibly taller, the shoulder's offset from the yaw axis possibly changed, possibly a longer or shorter wrist, both
fingers longer or shorter, possibly printed spacer blocks in some servo pockets), fitted the servos at arbitrary
positions and left no dimensions. Like every printed
arm, this one also carries the small errors of its own build. The arm has been shipped to a partner; before it left, a
colleague photographed it once in the lab's photo cove while its servos' readings were logged.
The lab needs the arm's simulation model: the MJCF whose kinematics match the REAL arm (joint axes and positions, the pose
at each servo reading, the tool points, the gripper, the firmware limits) with sane inertials and collision geometry, in
the chain's contract, so the control and environment stages can use it.

Your description is graded against the photographed arm's ground truth, which only the grader has: at hidden servo
readings the grader poses your model through your own calibration and the real arm through its true one and compares
them in the world frame; it also drives the real arm with an IK solved on your model. The published nominal model is
tens of millimetres off; only what you measure from the photos makes it right. Nobody has another build of this arm, and
there is no checker that knows the real arm.

Everything you need is in your home directory, `/home/user/`. There is no internet access. Use the interpreter
`/opt/venv/bin/python3` (call it by its full path): MuJoCo 3.14 (offscreen rendering through OSMesa,
`MUJOCO_GL=osmesa`), OpenCV (`cv2`, headless), scikit-image, numpy, scipy, trimesh, manifold3d, CoACD, Pinocchio,
Pillow, CadQuery, jsonschema and pyyaml are installed there. How you get the numbers out of the photos is up to you; the
grader scores the files you leave behind.

## Given inputs

| File | What it is |
|---|---|
| `photos/photo_01.jpg` ... `photo_21.jpg`, `photos/photos.json`, `photos/README.md` | the photo session (1600 x 1200 JPEG, one hand-held phone, one lens, not calibrated, camera poses not recorded): 01-18 = six arm poses A-F, three photos each from different sides; 19 / 20 = close-ups of the gripper lowered near the platform, jaw nearly closed (the fingers not touching) / open; 21 = a close-up of the base (the arm in one of the poses A-F). `photos.json`: per photo the group, the arm configuration label, the scale references on the platform, the lens's 35 mm-equivalent focal-length class |
| `photos/bus_log.json` | per photo the six servo readings logged at the moment of the photo: encoder ticks and the bus angle b = (ticks - 2048) x 2 pi / 4096 rad. Exact (the servos' own encoders) |
| `scale_refs.md` | the references that were on the platform in some photos: a 300 mm steel rule (flat), a 66 x 122 mm drink can (standing), an ID-1 card (flat); their poses were not recorded. There is no calibration target |
| `parts/P01.stl` ... `P10.stl`, `parts/parts.json`, `parts/base_in_world.stl`, `parts/servo.stl` | the PUBLISHED printed parts as printed (mm, each in its own print frame: resting on a face on z = 0, bounding box centred, arbitrary rotation about z; part roles and links in `parts.json`), the published base placed in the world frame, the servo's vendor mesh (mm, vendor frame) |
| `servo_datasheet.md`, `servo_datasheet.png`, `servo_drawing.png` | the servo BUS-SERVO-12V-A (all six joints): 60 g, 2.94 N m stall at 12 V, 4096-tick encoder, dimensions |
| `servo_config.json` | the EEPROM dump of the six servos of THIS arm, read off the bus: drive-mode bit (1 = the reading's direction inverted), firmware min / max angle limits in ticks, offset register, torque limit; servo ids 1-6 = base_yaw, shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll, jaw |
| `masses.json` | the slicer's mass estimates of the ten parts AS PUBLISHED (outer-shell volume x 520 kg/m3), the servo mass, and the complete arm on a kitchen scale (printed parts + six servos, 2 g resolution). No part was weighed individually; the real infill is not known |
| `build_notes.md` | the builder's qualitative notes (what may have been changed; no dimension) |
| `robot_contract.md`, `contract/zero_pose.png`, `contract/tcp_definition.png` | the contract of the files you deliver: names, frames, angles, calibration, tool points, actuator rules, MJCF rules, caps, the gate |
| `check_submission.py`, `rd_info_check/` | the grader's own contract gate (no ground truth: it says whether a folder is a valid description, not how accurate it is) |

## Frames and conventions (full text: `robot_contract.md`)

- World frame: the frame of `parts/base_in_world.stl` (origin on the mounting plane, the platform the base is screwed to, nominally
  on the base-yaw axis, within the build's errors of about a millimetre), z up, +y = the base's forward direction
  (reaching forward is a negative shoulder pitch), +x = the pitch-axis direction at yaw 0. SI units in `robot.xml`;
  `<compiler angle="radian">` stated explicitly.
- Bodies `base`, `shoulder`, `upper_arm`, `forearm`, `wrist`, `gripper`, `jaw`; hinges `base_yaw`, `shoulder_pitch`,
  `elbow_pitch`, `wrist_pitch`, `wrist_roll`, `jaw` (in this chain order, no other joint); position actuators
  `act_<joint>`; sites `tcp` (on `gripper`) and `jaw_tip` (on `jaw`); keyframe `home`.
- Design angle q = qpos; q = 0 = the straight-up configuration; positive q right-handed about the joint axis; no `ref`.
- Calibration: b = sign x (q - zero_offset_rad) per joint (`calibration.json`), b = the bus angle of `bus_log.json`.
- `tcp`: on the fixed finger's gripping face (the face looking at the moving jaw), on the finger's mid-line, 10.0 mm back
  along the finger from the fingertip; +z = approach (along the finger, out of the gripper), +x = the face normal towards
  the moving jaw. The gripping face is stepped (flat over the finger's top 10 mm, stepping back below): the site lies on
  the plane of that flat tip face. `jaw_tip`: the same construction on the moving jaw (+x towards the fixed finger).

## Deliverable: `/home/user/submission/`

```
/home/user/submission/
├── robot.xml          the MJCF of the arm (contract above; visual + collision geometry, inertials, actuators, keyframe home)
├── scene.xml          <include file="robot.xml"/> + a table plane at z = 0 + a light
├── assets/            every mesh robot.xml references (<= 400 000 triangles in total)
├── calibration.json   {"joints": {"<joint>": {"sign": 1 | -1, "zero_offset_rad": float}}} for all six joints
└── robot.urdf, estimate.json, build.py     optional (recorded, never executed or scored)
```

## How the submission is graded

The grader runs after you finish and executes none of your code. The gate `model_loads` multiplies every criterion;
the score is the weighted mean of seven criteria in [0, 1].

**Composite map.** At a hidden bus reading b (six servo readings inside the real arm's firmware limits, drawn fresh at
every grading run, only readings at which the real arm touches neither itself nor the table), your model is set to
q = sign x b + zero_offset_rad from YOUR `calibration.json` and the real arm to its true pose; everything is compared in
the world frame. Your body frames, mesh origins and zero convention are therefore free; only the resulting poses count.

**Band.** L(e; full, zero) = 1 for e <= full, 0 for e >= zero, ln(zero / e) / ln(zero / full) between (every halving of an
error earns about the same increment). lin(x; a -> 0, b -> 1) is linear between a and b, clamped.

| Gate | Rule |
|---|---|
| `model_loads` | every rule of `robot_contract.md` "Gate" (files, whitelist before compiling, relative `file=` / mesh / texture paths without `..`, the explicit radian header, zero-warning compiles of `scene.xml` and `robot.xml`, the six hinges in chain order with no `ref`, the seven bodies with the contract parents, the two sites on their links, `home`, six position actuators with gear 1, a valid `calibration.json`, the caps) and nothing in the submission written after your session ends |

| Criterion | Weight | What is measured |
|---|---|---|
| `fk_tcp` | 0.30 | 200 hidden readings: tcp position error L(e; 0.5, 30 mm) x 0.60 + approach-axis (+z) angle L(e; 0.25, 10 deg) x 0.25 + closing-axis (+x) angle L(e; 0.25, 10 deg) x 0.15; mean |
| `fk_axes` | 0.15 | the same readings, joints shoulder_pitch, elbow_pitch, wrist_pitch, wrist_roll, jaw: 0.5 L(angle between your axis line and the real one; 0.25, 10 deg) + 0.5 L(distance of the real axis's hub point from your axis line; 0.5, 20 mm); mean over joints and readings. Per link the mean of its driving joint's item is its kinematic factor kf (base 1, shoulder = the base-yaw item), which multiplies that link's share of `inertial` and `collision` |
| `downstream_reach` | 0.20 | 48 hidden targets (tcp position + approach of the real arm at readings with the tcp 0.12-0.30 m from the yaw axis, 0.02-0.25 m above the table, within 100 deg of +y): the grader's IK (damped least squares, 24 restarts) solves each target on YOUR compiled model for the five arm joints within your joint ranges (to 0.1 mm / 0.1 deg; else its best solution within 3 mm / 2 deg), converts q to bus angles through your calibration, rounds to encoder ticks (the jaw is commanded to bus angle 0, tick 2048), the real servos clamp to their firmware limits, and the real arm (jaw included in its contact test) executes: 0.7 L(position error; 1.0, 30 mm) + 0.3 L(approach error; 0.3, 10 deg); no IK solution, or a real-arm self / table contact deeper than 1 mm, scores 0; mean |
| `gripper` | 0.10 | 5 hidden jaw readings over the real jaw's range, each at 8 of the readings: 0.5 L(error of the tcp-to-jaw_tip distance; 0.3, 10 mm) + 0.5 L(jaw_tip position error; 0.5, 30 mm); mean |
| `limits_actuators` | 0.05 | per joint: 0.6 lin(IoU of your joint range, expressed in bus angles through your calibration, with the real firmware range; 0.80 -> 0, 0.99 -> 1) + 0.4 x the pass fraction of the actuator rules of `robot_contract.md` (ctrlrange = range, forcerange, kp, kv, armature, damping, frictionloss); mean over joints |
| `inertial` | 0.10 | per link (mass of the link's bodies incl. jointless sub-bodies): the geometric mean of L(relative mass error; 3 %, 30 %), L(world centre-of-mass error at 16 readings; 1, 20 mm) and L(relative Frobenius error of the inertia tensor about the centre of mass, world frame; 15 %, 100 %), x kf; mean over the seven links, x L(relative error of the total mass; 1 %, 10 %) |
| `collision` | 0.10 | per link (collidable geoms: contype or conaffinity non-zero; meshes count as their convex hulls), at 32 readings: 0.4 lin(coverage = share of 600 points of the real link's surface, inset 0.5 mm, within 2 mm of your link's collision geometry; 0.60 -> 0, 0.92 -> 1) + 0.3 lin(bloat = 95th percentile distance of 600 points of your collision surface outside the real link's solid, pooled over the first 8 of the 32 readings; 15 mm -> 0, 4 mm -> 1) + 0.3 clean (share of 64 hidden readings at which your collision model has no self-penetration deeper than 2 mm between non-adjacent links, measured geometrically; credited when the link's coverage > 0.5); x kf; mean over the seven links |

Recorded but not scored: tcp error mean / p95 / max (mm), approach and closing errors, per-joint axis angles and hub
distances, the kinematic factors, downstream errors and IK failures, gripper errors, per-joint range IoU and actuator
rules, per-link mass / CoM / inertia errors, per-link coverage and bloat, the clean fraction, your zero convention's
distance from the real one, whether `robot.urdf` / `estimate.json` exist. The photographed arm's own description
scores 1.0 on everything.

```bash
/opt/venv/bin/python3 ~/check_submission.py ~/submission     # the contract gate (no ground truth)
```

## Practical notes

- The bus log is exact; the cameras are not known. There is no planar calibration target; the rule, the card and the can
  have their everyday nominal sizes and are on the platform in a few photos only (`photos.json`). The lens is one lens for
  the whole session; its focal length changes slightly from photo to photo (autofocus) and its distortion is not
  corrected.
- MuJoCo renders offscreen through OSMesa in this environment (a 800 x 600 frame of the arm takes a fraction of a second).
- The photos are meant for programmatic processing; viewing all of them costs a large share of your context.
  `photos.json` tells you which photo shows what.
- The grader reads its own copies of the given files; your home's copies are yours to read.
- Environment facts (harness limits): a single tool call whose input runs above roughly 25-30k tokens is dropped by the
  harness (it is never executed and no error comes back), and a response that reaches the 64k output-token limit is lost
  and, after at most a few retries, ends the session with whatever is in `/home/user/submission/` at that moment. Write
  large files in several parts (one part or one function group per write) and keep every single response well below that
  limit.
- Session time: this session is stopped after 6 hours of wall-clock time from its start. A session stopped by that limit is not graded at all, so a late finish scores nothing. Note the start time with `date` when you begin, keep a complete submission in place from early on, and end the session yourself (stop working and say that you are done) at least 15 minutes before the limit.
