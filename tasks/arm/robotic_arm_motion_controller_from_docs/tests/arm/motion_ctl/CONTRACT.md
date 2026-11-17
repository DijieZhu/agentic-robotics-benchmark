# motion_ctl: the runtime contract between the grader and the agent's controller (binding)

This file is the single source of truth for every format and number the controller sees. The grader
(`grade.py`, `wire.py`, `client.py`, `jobs.public`, `cells.Cell.description`, `unit.calibration_record`, `plant.py`)
implements it; the agent's documents state it in a vendor's / integrator's words. Change both or neither.
Units: SI unless a field name says `_mm`, `_deg`, `_ms`; angles in documents are degrees, in data radians only where
stated. Revision 2 (2026-11-05, after the document review): the TCP between the pads, the clamped tools' geometry, the
plan's bus state, the error semantics, the home pose, positive cell coordinates. Revision 3 (2026-11-05, after the
adversarial grader review and the document review): path progress, renormalised episode weights, the 1 MiB reply line,
CPU of exited threads and of child processes, the reply bounds, the pad gap at q6 = 0 (15.8 mm), infeasible jobs' form.
Revision 5 (2026-11-16, difficulty): the accuracy bands of a well-calibrated arm
(`metrics.BANDS`: position 1.2 -> 3 mm, path RMS 2.2 -> 4 mm, peak 4 -> 7 mm) and the reach accuracy as the product
L(position) x L(approach) x L(jaw).
Revision 4 (2026-11-09, difficulty): a contact fails the cell (its later jobs score 0), blocked infeasible targets, one
or two posts in the shelf and bins cells, the fixed hidden set of three units, the no-network seccomp filter.

## 1. Frames and angles

- **Base frame** (the kinematics sheet's frame 0): origin on the mounting plane (the underside of the base, which
  stands on the table) on the base-yaw axis; z up; +y forward = the direction the arm leans when the shoulder pitches
  negative at base yaw 0; x = y cross z.
- **Design angle** q_j of joint j: radians, q = 0 is the zero pose (the arm straight up: upper arm, forearm and roll axis
  vertical; the jaw at its reference, pads parallel and 15.8 mm apart); positive directions are those of the kinematics
  sheet (right-hand rule about the joint axis as drawn). Joint order and servo ids: 1 base_yaw, 2 shoulder_pitch,
  3 elbow_pitch, 4 wrist_pitch, 5 wrist_roll, 6 jaw.
- **Link CAD frames**: each link's STL / mass properties are given in the link's frame, which at the zero pose is
  parallel to the base frame with its origin on the joint axis that carries the link (the anchor points are tabulated
  at the zero pose). The links' collision envelope (convex pieces; contact and clearance at acceptance are judged on it)
  is published next to the CAD.
- **TCP**: on the gripper, in the middle of the pad gap when the pads are 15.68 mm apart (the needle holder's width; at
  q6 = 0 the middle lies 0.06 mm towards the jaw), at the pads' height, on the wrist-roll axis (0.2 mm off it; the
  gripper drawing); TCP frame: +z = approach (out of the fingers, along the roll axis), +x = the jaw axis (from the
  fixed finger towards the moving jaw).
- **Clamped tools** (in the TCP frame): the bins cell's part is a box whose -x face rests on the fixed finger's pad face
  (x = -7.84 mm) with its centre at (seat x, 0, seat z) = (-7.84 mm + hx, 0, seat z), axes parallel to the TCP frame;
  the jaw closes on its +x face. The dispense cell's needle sits in a holder block (half sizes 7.84 x 6 x 9 mm, centre at
  z = -3 mm) that fills the pad gap; the needle (radius 1.2 mm) runs along the TCP's +z from z = 6 mm to its tip at
  z = 45.0 mm (the tool point of dispense jobs).
- **Cell frame**: the table's reference corner; z up, table top z = 0, every object and the robot base at x, y >= 0.05 m
  (cells.draw_cell; a surveyed description stays above 0.04 m). Scene descriptions and jobs are in the cell frame; the
  scene description gives the robot base's pose in the cell frame (x, y, z and yaw about z, degrees).

## 2. Unit information (given at runtime to `setup`)

```json
{"serial": "MU-1A2BC",
 "calibration": {"format": "calibration-record/1", "serial": "...", "convention": "...",
                 "uncertainty_deg": {"arm joints": 0.5, "jaw": 1.0},
                 "servos": {"base_yaw": {"id": 1, "direction": 1, "zero_ticks": 1795, "min_ticks": 578, "max_ticks": 2995}, ...}}}
```
q = direction x (ticks - zero_ticks) x 2 pi / 4096. EEPROM limits: the servo refuses goals outside
[min_ticks, max_ticks] (clamps them); every tick value of a unit lies in 0..4095. The record's zero has the stated
uncertainty (the real unit's zero differs by up to that); the documents state the other unit-to-unit tolerances
(geometry: joint frames +-0.3 mm per axis, pitch-axis direction +-0.15 deg about each of the two perpendicular axes,
upper-arm / forearm axis distances +-0.3 %, wrist-to-tip +-0.5 mm; masses +-5 %; servo dynamics) as ranges.

## 3. The controller

`submission/controller/controller.py` defines `class Controller` with:

| call | when | budget (grader-measured CPU of the controller's processes since the previous reply) |
|---|---|---|
| `__init__(self, workdir)` | once; `workdir` = an empty writable directory | import + init <= 60 s wall |
| `setup(self, unit, cell, tool)` | at the start of each cell (3 per acceptance) | <= 60 s CPU, 120 s wall |
| `plan(self, job, state)` -> `"ok"` or `"infeasible"` | before each job's motion (the arm holds) | <= 20 s CPU, 60 s wall |
| `step(self, state)` -> command | every 10 ms bus cycle of a job's motion | 10 ms CPU per cycle (see 5) |

`plan` receives the present bus state (the `state` of section 5 without `cycle` and `t`): where the arm is now.
CPU counts every thread and every process the controller starts (also the ones that have exited by the time of the
reply).
Compatibility: a controller whose `plan` takes only `job` is called with the job alone.

Imports allowed: the Python standard library and any package installed in the image (numpy, scipy, mujoco,
pinocchio, mink, trimesh, ...), with the network audit's exceptions: no networking module (socket, ssl, http, urllib,
ftplib, smtplib, asyncio, selectors, xmlrpc, multiprocessing.connection / managers, logging.handlers, requests, ...), no
ctypes / cffi, no subprocess / os.system / os.popen / os.exec* / os.spawn* / pty, no import under a computed name
(`__import__` / `importlib.import_module` / `find_spec` with a non-literal argument) and no exec / eval / compile of
computed code. The process runs as an unprivileged account with no network (a seccomp filter on it and on everything
it starts refuses every socket but a local Unix one), a fresh HOME / TMPDIR, BLAS threads pinned
to 1, and only the submission's own files: the controller must be self-contained (the documents are not on the
controller PC; copy what it needs, e.g. meshes or tables, into `submission/controller/`; size cap 80 MB).

`tool` (per cell): `{"kind": "gripper"}` (shelf cell; the jaw servo is the controller's), `{"kind": "part",
"half_mm": [hx, hy, hz], "seat_mm": {"x": sx, "z": sz}, "mass_range_kg": [0.03, 0.15], "com_note": "the centre of mass
is inside the part; not known"}` (bins cell: the part of section 1, its centre at TCP + sx along x + sz along z; mass
and centre unknown), `{"kind": "needle", "length_mm": 45.0, "radius_mm": 1.2, "mass_kg": 0.02, "axis_offset_x_mm": ...,
"holder_half_mm": [7.84, 6.0, 9.0], "holder_centre_z_mm": -3.0}` (dispense cell; mass = needle + holder). While a part
or the needle is clamped, servo 6 holds the clamp: its goal is ignored. Contacts of the clamped tool with the gripper,
the jaw and the wrist are not contacts; with anything else they are.

## 4. Jobs (`plan` argument; cell frame)

Common fields: `id`, `type` ("reach" | "path"), `tool` ("gripper" | "gripper+part" | "needle"), `clearance_mm` (10),
`time_target_s`, `exempt` (object or primitive names whose distance to the tool or the arm does not count as clearance:
the job's own support; contact still counts). The previous job's exempt names stay exempt during the next job (the arm
leaves that support at its start).

- **reach**: `position` [3] m (the TCP, or the needle tip when tool = needle), `approach` [3] unit vector (the TCP's +z),
  `approach_tol_deg` 5, `jaw_axis` [3] or null (the TCP's +x, either sign), `jaw_tol_deg` 5. Done = the tool point at
  the position with the approach (and jaw axis) inside the tolerances, at rest.
- **path**: `t` [n] s (0.01 s steps from 0), `points` [n][3] m (the needle tip's reference p(t), already 3 mm off the
  board along `normal`), `normal` [3] (the board's surface normal), `axis_tol_deg` 8 (the needle axis = TCP +z must be
  within 8 deg of -normal), `gap_mm` 3, `gap_tol_mm` 1.5, `speed_mm_s`, `shape`, `approach_time_target_s`. The
  controller first brings the tip to points[0], then sets `path_start` in a step reply: the reference starts at the end
  of that cycle and runs for t[-1] s; done after the path ends.
- An infeasible job (the target or the path cannot be reached within the tolerances and the clearance) must be answered
  `"infeasible"` by `plan`; the arm then holds. Infeasible jobs carry the same fields as feasible ones, with the same
  numerical form (the jaw axis exactly horizontal or in the arm's plane for both; time targets, exempt names and
  positions drawn like the cell's feasible reaches). Maintainer note (not in the agent's documents): the generator
  makes reach targets infeasible only; the shelf and bins cells have one or two blocked targets (the arm reaches the
  pose, but every solution, also with the approach tilted 4.5 deg four ways, puts the arm, the gripper or the part at
  least 4 mm into the cell or itself: only a collision check tells; on a support a feasible job of the cell uses) and
  the rest out of the arm's plane or inside the fixture; the dispensing cell's are at the cup (out of plane, inside the
  cup); no target floats in free space beyond the reach; paths are always feasible.

## 5. The bus cycle (`step`)

`state` (every 10 ms): `{"cycle": k (from 0 at the job's first cycle), "t": s since the job's start,
"position": [6] ticks, "speed": [6] ticks/s, "load": [6] per mille of stall torque (signed, along +ticks),
"moving": bool (any servo's profile or motion active)}`.

Reply: `{"goal": [6] ticks (numbers; rounded to whole ticks) or null = keep the last goals, "speed": [6] ticks/s or
null (whole numbers; 0 = the servo's maximum), "acc": [6] or null (whole numbers, x 100 ticks/s^2; 0 = default),
"path_start": bool, "done": bool}`. Missing keys read as null / false; a null reply is no command. A reply that is not
an object, or a list that is not 6 finite numbers, is a protocol error: the job is aborted (not completed). Reply
bounds (grade._parse_cmd): booleans are not numbers, goal entries within +-8192, speed and acc entries within +-1e6. A
reply line over 1 MiB is a protocol error that also ends the controller (it is restarted as after a timeout).

Timing: a step that uses more than 10 ms CPU delays its command by floor(cpu / 10 ms) cycles (the bus keeps the last
goals meanwhile and the controller gets no state for those cycles: the real-time loop missed its deadline); a step
over 1 s CPU or 5 s wall ends the controller (it is restarted, `setup` is called again, and the next job follows).
Gate: in each run, at least 90 % of the cycles answered within 10 ms CPU. Errors: a `step` that raises aborts the job; a `plan`
that raises skips the job (not run, not a refusal); a failed `setup` is retried once with a fresh controller, then the
cell's jobs are not run.

Servo behaviour the documents state: position quantised to whole ticks, goals clamped to the EEPROM limits, command
latency 15-40 ms (unit-dependent, bus + firmware), trapezoidal profile (max speed = rated x 0.85-1.0, acceleration
40 rad/s^2 by default, the speed and acceleration registers can only lower them), position loop (gain spread +-15 %),
an elastic output (60-600 N m/rad by joint) with 0.05-0.25 deg of hysteresis behind the encoder (the encoder reads the
servo's output shaft, not the link), load register = output torque / stall x 0.9-1.1 + 1 % noise, the whole register
refreshed every 20 ms.

## 6. Acceptance (what happens to the controller's commands)

The acceptance is three runs, each on its own unit (the task's fixed hidden set: `scenarios.py`, three scenarios kept
by the maintainers' pre-registered selection; the reward is the mean of the three). Per run: one unit,
three cells in the order shelf, bins, dispense; about 14 jobs per cell in the given order;
each job starts where the previous one ended. A cell starts at the home pose, which the operator's homing tool
commands through the unit's calibration record (design angles base yaw 0, shoulder 20.05 deg, elbow -83.08 deg, wrist
pitch -71.62 deg, roll 0; jaw 8.59 deg or the clamp). A contact (any penetration of the arm, the part, the holder or
the needle into the cell, the table or the arm itself) stops the arm and fails the cell's acceptance: that job and
every later job of the cell score 0 (they are not run; an infeasible one among them counts as not refused), and the
next cell starts from home with its own setup. A job not done within its cap (reach: max(2 x time_target,
time_target + 2 s); path: 2 x approach_time_target + duration + 2 s) is not completed. After `done` the grader waits
0.5 s (the bus keeps the last goals) and measures.

Scoring: `metrics.py` (job = completed x no_contact x clearance x conditioning x accuracy x (0.6 + 0.2 time + 0.2
smoothness); reach accuracy = L(position) x L(approach) x L(jaw axis, when the job has one); path accuracy = L(RMS deviation) x (1 - the fraction of path samples with the needle axis outside its
cone) x progress, progress = the tip's motion projected on the reference's motion over the reference's own, clipped
to [0, 1]; episode = 0.5 reach + 0.4 path + 0.1 J, a category without jobs left out and the weights renormalised).
The conditioning measure: sigma_min of the 4 x 4 matrix
[J_p (3 x 4); 0.1 m x (a_p^T J_w) (1 x 4)] at the tool point, columns base yaw, shoulder, elbow, wrist pitch (the output
angles), J_p = the tool point's linear Jacobian, J_w = the angular Jacobian, a_p = the unit pitch-axis direction;
threshold `metrics.COND_MIN` (m/rad). Overshoot: the largest distance of the tool point from its final position after it
first came within 1 mm of it, minus that 1 mm.

## 7. Privileged runs (the oracle only)

When `submission/controller/oracle_token` hashes to `tests/hidden/oracle_token.sha256`, the grader adds `"truth"` to
the `setup` and `plan` payloads (the unit's true parameters, the job's precomputed tick trajectory). Never documented
to the agent; counted in the stat `truth_payloads`.
