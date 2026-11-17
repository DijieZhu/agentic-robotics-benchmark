# DA-6 motion controller: site acceptance test specification

The lab accepts the motion controller of its DA-6 desktop arm in three acceptance runs, each on its own unit (three units, one run on each), in the three desktop cells (`../cells/cell_manual.md`). There is no trial run and no time on the units beforehand: the controller is installed as delivered and runs once on each unit. This specification states how a run proceeds, what the controller is asked to do, how each job is measured and how the measurements make the acceptance score (0 to 1), the mean of the three runs' scores (section 5.3). The controller interface (the Python class, its calls, the software environment) is specified in `../controller/controller_interface.md`, the bus cycle in `../servo/bus_interface.md`; frames, cells, the scene description and the clamped tools in `../cells/cell_manual.md`.

## 1. The acceptance runs

Each of the three runs, on its own unit, proceeds as follows:

1. The controller is installed from the delivered files on the cell computer (the documentation is not on that computer: the controller carries what it needs) and constructed: `Controller(workdir)`, a new controller for each run, with an empty `workdir`. The cell computer has no network: creating a network socket fails, for the controller and for every process it starts (only local Unix-domain sockets can be created).
2. The three cells are worked in the order shelf, bins, dispense. For each cell the operator clamps the cell's tool (bins: the part; dispense: the needle in its holder), brings the arm to the home pose (section 1.1) and the cell computer calls `setup(unit, cell, tool)` (`../cells/cell_manual.md`, section 8).
3. Then the cell's jobs follow, about 14 per cell, in the given order. Each job starts with the arm where the previous job left it (the first job of a cell: at the home pose). A protective stop ends the cell (section 1.2).
4. Per job: `plan(job, state)` with the arm holding still (`state`: the bus readings where the arm stands, section 2; the time spent in `plan` does not count towards the job's time); the answer is `"ok"` or `"infeasible"`, and any other return value counts as `"ok"`. After `"infeasible"` nothing moves and the next job follows. After `"ok"` the motion runs: `step(state)` every 10 ms bus cycle; the reply's goals go to the servos; until a reply says `done` or the job's time cap is reached (section 4.5).
5. After `done` the servos hold the last goals for 0.5 s, during which `step` is not called; then the job is measured (section 4).

### 1.1 Home pose

At the start of each cell the operator's homing tool brings the arm to its home pose and the arm comes to rest there: the upper arm leaning back, the gripper in front of the base pointing forward and down (at the design dimensions the TCP at x = 0.000 m, y = 0.234 m, z = 0.205 m of the base frame, the approach 44.6° below the horizontal):

| joint | base yaw | shoulder pitch | elbow pitch | wrist pitch | wrist roll | jaw |
|---|---|---|---|---|---|---|
| design angle (°) | 0 | 20.05 | -83.08 | -71.62 | 0 | 8.59 (shelf cell); with a part or the needle clamped: the clamp angle (`../cells/cell_manual.md`, section 9.4) |

The homing tool commands these design angles through the unit's calibration record (`../unit/calibration_record.md`): for each servo the goal = round(zero_ticks + direction × q × 4096 / (2π)) ticks with q in radians, limited to [min_ticks, max_ticks]. The arm's actual angles therefore differ from the table by the record's error (within the uncertainty the record states) and by the servos' elastic deflection under load (`../servo/servo_datasheet.md`). `plan` receives the bus readings there (section 2), and so does the first bus cycle of the job.

### 1.2 Protective stop

Any contact stops the arm: a penetration deeper than 0.01 mm between the arm (its moving links: shoulder, upper arm, forearm, wrist, gripper and jaw; the clamped part, or the needle and its holder) and a cell object or the table top, or between two parts of the arm that are not neighbours in the chain (the base included; the clamped tool counts with the gripper: it is not checked against the gripper, the jaw and the wrist), at any time during the job's motion or the 0.5 s hold after `done`. The base stands on the table: it is checked against the moving links only, not against the table and the cell objects. A protective stop fails the cell: the job and every later job of the cell score 0 (the later jobs are not run; a job among them that cannot be executed counts as not refused, section 5.3). The next cell starts as usual (section 1, step 2): its tool is clamped, the arm is brought to the home pose and `setup` is called; after a stop in the dispensing cell the run ends.

Contact and clearance (section 4.1) are judged on the arm's collision envelope, not on the link CAD: 166 convex pieces, published as STL files in the link CAD frames in `../robot/cad/collision/` (described in `../robot/cad/README.md`): 160 convex hulls that follow the printed parts (they bridge their recesses and gaps) and 6 boxes, one around each servo body. The clamped part, the needle holder and the needle are judged on their envelopes (`../cells/cell_manual.md`, section 9).

The controller is not told why a job ended. A job that ends without the controller's `done` (the time cap, a failed call) is followed by the next job's `plan`; its `state` reports where the arm is. After a protective stop the cell's remaining jobs are not passed to the controller: its next call is the next cell's `setup`, with the arm at the home pose.

### 1.3 Limits of the calls

The CPU time charged to a call is the CPU time that all the controller's processes and threads used since the controller's previous reply, threads and processes that have ended by the reply included: work done between calls counts against the next call.

| call | limit |
|---|---|
| import and `Controller(workdir)` | 60 s wall clock |
| `setup(unit, cell, tool)` | 60 s CPU, 120 s wall clock |
| `plan(job, state)` | 20 s CPU, 60 s wall clock |
| `step(state)` | 10 ms CPU per bus cycle (real time); at most 1 s CPU, 5 s wall clock |

- A `step` that uses c > 10 ms of CPU delays its command by floor(c / 10 ms) bus cycles; meanwhile the servos keep the last goals. Those 1 + floor(c / 10 ms) cycles count as late (section 6).

Failures and what they cost (the controller is not told; the next call follows as listed):

| event | consequence |
|---|---|
| `plan` raises an exception | the job is skipped: it is not run and it is not a refusal (an executable job scores 0; a job that cannot be executed does not count as refused, section 5.3); the next job's `plan` follows |
| `step` raises an exception, or its reply cannot be sent as JSON | the job is aborted: not completed (score 0); the servos keep the last goals; the next job's `plan` follows |
| a `step` reply that is neither a JSON object nor null, or whose `goal`, `speed` or `acc` is neither null nor a list of six finite numbers (booleans are not numbers; `goal` entries within ±8192, `speed` and `acc` entries within ±10⁶) | a protocol error: the job is aborted (not completed); the next job's `plan` follows. A null reply is no command; a missing key reads as null (`goal`, `speed`, `acc`) or false (`path_start`, `done`); goals are rounded to whole ticks |
| a call over its limit (a `step`: over 1 s CPU or 5 s wall clock), or a controller process that died | the call's job ends (a `plan`: the job is not run; a `step`: the job is aborted); the controller is started again (`Controller(workdir)`, with an empty `workdir`) and `setup` is called again for the current cell before the next job |
| the controller's answer to a call is not one JSON line of at most 1 MiB (1048576 bytes), e.g. because something was written to the original standard output (file descriptor 1) or a reply is larger | a protocol error that also ends the controller, as a call over its limit does: the call's job is not run (`plan`) or aborted (`step`), the controller is started again and `setup` is called again for the current cell before the next job (for `setup` itself: the next row) |
| `setup` fails (an exception, over its limit, a process that died or a broken answer line), also when the controller cannot be imported and constructed within 60 s | the controller is restarted and set up once more; if that fails too, the cell's remaining jobs are not run (each scores 0, none counts as refused) |
| a protective stop (a contact, section 1.2) | the job and every later job of the cell score 0 (the later jobs are not run, none counts as refused); the next call is the next cell's `setup` |

### 1.4 The bus cycle, in short

The full format is that of the bus interface (`../servo/bus_interface.md`). For this specification: `state.cycle` counts the job's bus cycles from 0 and `state.t` is the time since the job's start (s) at the cycle's start; in the reply, `goal` (six goal positions in ticks, or null to keep the last goals) is sent to the servos when the reply arrives (they act on it after their command latency, `../servo/servo_datasheet.md`), `path_start` starts a path's reference (section 3.3) and `done` ends the job's motion.

## 2. Jobs

`plan(job, state)` receives one job as a JSON object in the cell frame (lengths in m unless a field name says `_mm`, angles in degrees) and `state`, the bus readings where the arm stands: the fields of the bus cycle's state without `cycle` and `t` (`../servo/bus_interface.md`). A `plan` that takes only `job` is called with the job alone.

| `state` field | type | meaning |
|---|---|---|
| `position` | [6] integers | present position of servos 1 to 6, ticks |
| `speed` | [6] integers | present speed, ticks/s |
| `load` | [6] integers | present load, per mille of the stall torque (signed, along +ticks) |
| `moving` | bool | true while any servo's profile or motion is active |

Example: the `state` of the first job of the example shelf cell (`../cells/example_scene_shelf.json`), at the home pose, for the unit whose record is `../unit/example_calibration_record.json`:

```json
{"position": [1714, 2072, 1590, 2707, 2168, 1785], "speed": [0, 0, 0, 0, 0, 0], "load": [3, 86, 139, -19, 23, 2], "moving": false}
```

Fields of every job:

| field | type | meaning |
|---|---|---|
| `id` | string | the job's identifier (the cell's initial and a running number) |
| `type` | string | `"reach"` or `"path"` |
| `tool` | string | `"gripper"` (shelf cell), `"gripper+part"` (bin cell) or `"needle"` (dispensing cell) |
| `clearance_mm` | number | the required clearance, mm (section 4.1) |
| `time_target_s` | number | the job's time target, s (section 4.5) |
| `exempt` | list of strings | names of objects or primitives that the job's working element may come closer to than the clearance: the job's own support (section 3.4) |

Reach jobs (`"type": "reach"`) add:

| field | type | meaning |
|---|---|---|
| `position` | [3], m | where the tool point must be: the TCP, or the needle tip when `tool` is `"needle"` |
| `approach` | [3] | unit vector: the required direction of the TCP's +z axis |
| `approach_tol_deg` | number | the tolerance of the approach, degrees |
| `jaw_axis` | [3] or null | unit vector: the required direction of the TCP's +x axis (either sign), or null when free |
| `jaw_tol_deg` | number | the tolerance of the jaw axis, degrees |

Path jobs (`"type": "path"`, dispensing cell, tool `"needle"`) add:

| field | type | meaning |
|---|---|---|
| `t` | [n], s | the reference times, from 0 in 0.01 s steps |
| `points` | [n][3], m | the needle tip's reference positions at the times `t`; already `gap_mm` off the board's surface along `normal` |
| `normal` | [3] | unit vector: the board surface's outward normal |
| `axis_tol_deg` | number | the needle axis (the TCP's +z) must be within this angle of -`normal`, degrees |
| `gap_mm` | number | the dispensing gap between the needle tip and the board's surface, mm |
| `gap_tol_mm` | number | the tolerance of the gap, mm |
| `speed_mm_s` | number | the speed of the reference along the path, mm/s (constant) |
| `shape` | object | `{"shape": "line", "length_mm": …}` or `{"shape": "arc", "radius_mm": …, "sweep_deg": …}` |
| `approach_time_target_s` | number | the time target of the move to `points[0]`, s (section 4.5) |

Values in this lab's jobs: `clearance_mm` 10, `approach_tol_deg` 5, `jaw_tol_deg` 5, `axis_tol_deg` 8, `gap_mm` 3, `gap_tol_mm` 1.5.

## 3. What each job asks for

### 3.1 The tool point

The TCP for the tools `"gripper"` and `"gripper+part"`; for `"needle"` the needle tip, the point (`axis_offset_x_mm`, 0, `length_mm`) = (-0.001, 0, 45) mm of the TCP frame (`../cells/cell_manual.md`, section 9.2).

### 3.2 Reach

Bring the tool point to `position` with the TCP's +z within `approach_tol_deg` of `approach` and, when `jaw_axis` is given, the TCP's +x within `jaw_tol_deg` of `jaw_axis` or of its opposite; come to rest there, keeping the clearance on the way, and reply `done`. In the shelf cell the reaches bring the gripper into the compartments and above the fixture block, in the bin cell they set the part down inside the bins and on the fixture block, and in the dispensing cell they bring the needle to purge positions over the cup between the paths.

### 3.3 Path

First bring the needle tip to `points[0]` with the needle axis inside its cone (this approach has no reference trajectory; its time target is `approach_time_target_s`). Then set `path_start` in a step reply: the reference starts at the end of the bus cycle in which that reply is written to the bus (normally the reply's own cycle; a later one when the reply is late; the servos' command latency does not enter; only the first `path_start` counts) and runs for t[-1] seconds. At the end of the k-th bus cycle after the path start the needle tip should be at `points[k]`, with the needle axis within `axis_tol_deg` of -`normal`; `gap_mm` ± `gap_tol_mm` is the gap that the dispensing allows. Reply `done` in a cycle that ends at or after the end of the reference; an earlier `done` fails the job. The paths are lines and arcs over the board's surface.

### 3.4 Exemptions

`exempt` names the job's own support. Each name is looked up first among the object names of the scene description, which exempts all the primitives of that object (the name of a dispensing board exempts its plate `<name>` and its rails `<name>_rail0` and `<name>_rail1`), and otherwise among the primitive names, which exempts that primitive only. The working element of the job may come closer to the named support than the clearance; contact still stops the arm. The working element is:

| `tool` | exempt from the clearance against the named support |
|---|---|
| `"gripper"` | the whole arm (every link, the gripper and the jaw) |
| `"gripper+part"` | the clamped part only (the links keep the clearance) |
| `"needle"` | the needle and its holder only (the links keep the clearance) |

The arm leaves the previous job's support at the start of a job: during a job's motion the support named by the previous job of the cell whose motion ran stays exempt as well (jobs answered `"infeasible"` or not run do not count). The first job of a cell, which starts at the home pose, has only its own support exempt.

### 3.5 Jobs that cannot be executed

A job is executable when the arm can do it within the tolerances while keeping the clearance: its target (or every point of its path) can be reached with the required approach (and jaw axis), or needle axis and gap, without coming closer than the clearance to anything other than the named support. Some jobs of each acceptance run cannot be executed; the job lists do not mark them, and they have the same fields as the other jobs of their cell. `plan` must answer `"infeasible"` for them, without moving the arm; it must answer `"ok"` for the others and do them.

## 4. Measurements

The lab measures the arm itself, independently of the controller: the pose of the tool point, the distances between the arm and the cell as built (not as surveyed), and the arm's configuration.

### 4.1 Clearance

c = the smallest distance between the arm (the collision envelope of its moving links, section 1.2, and the clamped part, or the needle and its holder) and the cell (its objects and the table top), at the end of every bus cycle from the job's first cycle to the end of the 0.5 s hold, skipping the exempt pairs (section 3.4); the base is not included. Required: c ≥ `clearance_mm` = 10 mm.

### 4.2 Conditioning

At the end of every bus cycle of the job's motion (from its first cycle to the cycle that replies `done`, late cycles included; not during the hold), the lab evaluates the arm's task Jacobian at the tool point:

```
        [ J_p                ]   J_p : 3 x 4, the tool point's linear velocity per joint rate (m/rad)
  J_t = [                    ]   J_w : 3 x 4, the gripper's angular velocity per joint rate (rad/rad)
        [ 0.1 m x a_p^T J_w  ]   a_p : the unit vector of the wrist-pitch axis
```

with the four columns base yaw, shoulder pitch, elbow pitch and wrist pitch, at the arm's actual pose; a_p^T J_w is the rate of the approach's pitch. The conditioning is σ_min, the smallest singular value of the 4 × 4 matrix J_t (m/rad); a cycle counts as ill-conditioned when σ_min < 0.008 m/rad.

σ_min never exceeds the tool point's horizontal distance from the base-yaw axis (the base-yaw column of J_t has that length), so a tool point within 8 mm of that axis is always ill-conditioned; σ_min also falls towards 0 as the elbow straightens. Examples at the design dimensions (base yaw 0, shoulder pitch -30°, wrist pitch -40°, roll 0):

| elbow pitch (°) | σ_min at the TCP (m/rad) | σ_min at the needle tip (m/rad) |
|---|---|---|
| 0 | 0.0000 | 0.0000 |
| -5 | 0.0033 | 0.0028 |
| -10 | 0.0063 | 0.0054 |
| -15 | 0.0092 | 0.0078 |
| -20 | 0.0118 | 0.0101 |
| -30 | 0.0165 | 0.0140 |
| -45 | 0.0225 | 0.0190 |
| -60 | 0.0273 | 0.0232 |
| -90 | 0.0352 | 0.0299 |

### 4.3 Reach jobs

- t_used: from the start of the job's first bus cycle to the end of the cycle whose reply says `done` (late cycles included).
- The measurement is taken at the end of the 0.5 s hold after `done`: p_end = the tool point, z_end and x_end = the TCP's +z and +x axes.
- Position error e_pos = |p_end - `position`| (mm).
- Approach error e_app = the angle between z_end and `approach` (degrees).
- Jaw-axis error e_jaw = the smaller of the angles between x_end and `jaw_axis` and between x_end and -`jaw_axis` (degrees); only when `jaw_axis` is given.
- Settle drift e_settle = |p_end - p_done| (mm), p_done = the tool point at the end of the `done` cycle.
- Overshoot e_over (mm): with d(t) = the distance of the tool point from p_end at the end of each bus cycle of the motion and of the hold, and t₁ the first cycle with d < 1 mm, e_over = max(0, max over t ≥ t₁ of d(t) - 1 mm): how far the tool point leaves the final position again once it has come within 1 mm of it.

### 4.4 Path jobs

- t_start = the end of the bus cycle in which the first reply with `path_start` is written to the bus (section 3.3; the command latency does not enter); T = t[-1].
- Samples: the needle tip at the end of every bus cycle with 0 < t - t_start ≤ T; the sample at t is compared with the reference point `points[i]`, i = round((t - t_start) / 0.01 s).
- e_rms = the RMS over the samples of |tip - `points[i]`| (mm); e_peak = the largest of them (mm). The gap is part of the reference points, so a gap error is a deviation like any other.
- f_out = the fraction of the samples with the needle axis (the TCP's +z) more than `axis_tol_deg` from -`normal`.
- P (progress) = Σ (tip_k - tip_(k-1)) · (ref_k - ref_(k-1)) / Σ |ref_k - ref_(k-1)|², clipped to [0, 1], where tip_k is the k-th sample, ref_k its reference point `points[i]` and the sums run over consecutive samples (P = 0 when the reference points do not move between the samples, e.g. with fewer than two samples): the tip's motion projected on the reference's motion, over the reference's own. P = 1 for a tip that moves with the reference; it is lower for a tip that falls behind or cuts the path short, and about 0 for a tip that stays where it is, however close to the path.
- t_used: as for reach jobs (the approach and the path included).

### 4.5 Time

Each job states its time target `time_target_s` (for a path it covers the approach, the path and a short margin); for an executable job the target allows a collision-free motion at a moderate speed, settling included. The time cap, after which the job is stopped and not completed: reach: max(2 × `time_target_s`, `time_target_s` + 2 s); path: 2 × `approach_time_target_s` + t[-1] + 2 s.

## 5. Scoring

### 5.1 Job score

```
job score = completed x no_contact x F_clear x F_cond x A x (0.6 + 0.2 T + 0.2 S)
```

Time and smoothness count only through the accuracy: a job whose tool point does not get where it has to (A = 0; for a path also a needle that does not move along the reference, P = 0) scores 0 however fast and smooth it was; a precise but slow job (T = 0) still scores F_clear × F_cond × A × (0.6 + 0.2 S).

| factor | value |
|---|---|
| completed | 1 if the controller replied `done` within the time cap (paths: after a path start, at or after the end of the reference), else 0 |
| no_contact | 0 if a protective stop happened during the job (motion or hold), else 1 (the cell's later jobs are then not run and score 0, section 1.2) |
| F_clear | 1 if c ≥ 10 mm; 0.5 + 0.5 × c / 10 mm for 0 ≤ c < 10 mm |
| F_cond | 1 - 0.5 × (the fraction of the motion's bus cycles with σ_min < 0.008 m/rad) |
| T (time) | 1 if t_used ≤ `time_target_s`; 1 - (t_used - `time_target_s`) / `time_target_s` up to 2 × `time_target_s`; 0 beyond |
| A (accuracy), reach | L(e_pos) × L(e_app) × L(e_jaw) with a jaw axis; L(e_pos) × L(e_app) without |
| A (accuracy), path | L(e_rms) × (1 - f_out) × P |
| S (smoothness), reach | 0.5 L(e_settle) + 0.5 L(e_over) |
| S (smoothness), path | L(e_peak) |

The reach accuracy is a product: a tool point at the right position with the tool pointing the wrong way (L(e_app) = 0, or L(e_jaw) = 0 when the job gives a jaw axis) has A = 0, and the job scores 0.

A job that cannot be executed earns nothing itself (it counts through J, section 5.3); an executable job that the controller refused, or that was not run, scores 0.

### 5.2 The band function L

```
L(e; full, zero) = 1                                  for e <= full
                 = ln(zero / e) / ln(zero / full)     for full < e < zero
                 = 0                                  for e >= zero
```

L is 0.5 at the geometric mean √(full × zero) of a band.

| quantity | unit | full (L = 1 at or below) | zero (L = 0 at or above) | L = 0.5 at |
|---|---|---|---|---|
| reach: position error of the tool point | mm | 1.2 | 3 | 1.90 |
| reach: approach error | deg | 1.5 | 12 | 4.24 |
| reach: jaw-axis error | deg | 2 | 15 | 5.48 |
| path: RMS deviation of the needle tip | mm | 2.2 | 4 | 2.97 |
| path: peak deviation of the needle tip | mm | 4 | 7 | 5.29 |
| reach: settle drift | mm | 0.3 | 6 | 1.34 |
| reach: overshoot | mm | 1 | 20 | 4.47 |

The position and path bands ask for the accuracy of a well-calibrated arm: a reach has full position credit only within 1.2 mm of the target and none from 3 mm, a path full accuracy credit only within 2.2 mm RMS of the reference and none from 4 mm (peak deviation: 4 mm and 7 mm). A unit's calibration record is only good to its stated uncertainty (`../unit/calibration_record.md`, section 2), and the units differ from the drawings and the datasheet within the stated tolerances (`../robot/kinematics_sheet.md`, section 7; `../servo/servo_datasheet.md`). A controller that only follows the design values of the documents and the record as written will therefore lose a large part of the accuracy credit.

### 5.3 Run score and acceptance score

```
run score = 0.5 x mean(reach job scores) + 0.4 x mean(path job scores) + 0.1 x J
acceptance score = (run score on unit 1 + run score on unit 2 + run score on unit 3) / 3
```

- The means are over the executable jobs of the run's three cells: the reach jobs (bare gripper, part and needle) and the path jobs; an executable job that was refused or not run enters its mean with 0. Jobs that cannot be executed are not in the means.
- J = max(0, n_refused,cannot / n_cannot - n_refused,executable / n_executable) over all jobs of the run: the share of correct refusals minus the share of wrong ones (a job that was not run is not a refusal). Refusing everything gives J = 0.
- A term without jobs is left out: if the run has no executable reach job, no executable path job or no job that cannot be executed, the run score is the weighted mean of the other terms, their weights divided by their sum (a run without path jobs: (0.5 × mean(reach job scores) + 0.1 × J) / 0.6).
- A protective stop costs the rest of its cell (section 1.2): the job and the cell's later jobs score 0; the executable ones among them enter the means with 0, and those that cannot be executed count as not refused.
- The acceptance score is the mean of the three run scores, one run on each unit; the conditions of section 6 make it 0.

### 5.4 Worked examples

A reach job with a jaw axis: done after 2.4 s against a target of 2 s, c = 7 mm, 10 % of the cycles ill-conditioned, e_pos = 1.6 mm, e_app = 3°, e_jaw = 4°, e_settle = 0.5 mm, e_over = 2 mm:

| F_clear | F_cond | T | A | S | job score |
|---|---|---|---|---|---|
| 0.850 | 0.950 | 0.800 | 0.300 | 0.799 | 0.223 |

A path job: done after 4 s against a target of 4.2 s, c ≥ 10 mm, no ill-conditioned cycle, e_rms = 2.8 mm, e_peak = 5 mm, f_out = 0.05, P = 0.98:

| F_clear | F_cond | T | A | S | job score |
|---|---|---|---|---|---|
| 1.000 | 1.000 | 1.000 | 0.555 | 0.601 | 0.511 |

A run that refuses 80 % of the jobs that cannot be executed and 5 % of the executable jobs has J = 0.80 - 0.05 = 0.75; with means of 0.6 (reach) and 0.5 (path) its run score is 0.5 × 0.6 + 0.4 × 0.5 + 0.1 × 0.75 = 0.575. The same run without path jobs would score (0.5 × 0.6 + 0.1 × 0.75) / 0.6 = 0.625.

With this run on one unit and run scores of 0.480 and 0.700 on the other two units, the acceptance score is (0.575 + 0.480 + 0.700) / 3 = 0.585.

## 6. Conditions that fail the acceptance

The acceptance score is 0 if any of the following happens in any of the three runs:

- fewer than 90 % of the run's bus cycles are answered within 10 ms of CPU time (late cycles: section 1.3; each run counts on its own);
- the controller's code breaks the restrictions of the controller interface (`../controller/`): networking modules (the cell computer has no network, section 1), starting other programs, and the other items listed there;
- a file of the delivered controller is written after the hand-over (the installed copy must be the delivered one);
- a process started by the controller is still running after the run.

It is 0 as well when in all three runs no job is refused and no job's motion runs (no `step` call), e.g. because the controller cannot be imported and constructed within 60 s (section 1.3).

## 7. Example job lists

`example_jobs_shelf.json`, `example_jobs_bins.json` and `example_jobs_dispense.json` are the job lists of the three example installations of `../cells/` (`example_scene_<kind>.json`), in the form `plan` receives them, with the `tool` argument of the cell's `setup`. As in the acceptance runs, some of these jobs cannot be executed and the lists do not mark them. Each file: `{"cell_kind", "scene_file", "setup_tool", "jobs": [job, ...]}`.

- `example_jobs_shelf.json`: 13 jobs, all reaches, tool `"gripper"`.
- `example_jobs_bins.json`: 14 jobs, all reaches, tool `"gripper+part"`.
- `example_jobs_dispense.json`: 12 jobs, 7 paths and 5 reaches, tool `"needle"`.

The example paths:

| id | shape | speed (mm/s) | duration t[-1] (s) | reference points |
|---|---|---|---|---|
| `D01` | arc, radius 18.2 mm, 234.7° | 21.93 | 3.4 | 341 |
| `D03` | arc, radius 16 mm, 234.4° | 34.46 | 1.89 | 190 |
| `D07` | arc, radius 15.7 mm, 104.3° | 25.94 | 1.09 | 110 |
| `D09` | arc, radius 26.7 mm, 111.5° | 18.72 | 2.78 | 279 |
| `D10` | arc, radius 17.2 mm, 181° | 28.95 | 1.87 | 188 |
| `D11` | arc, radius 19 mm, 120.7° | 18.21 | 2.19 | 220 |
| `D12` | arc, radius 21.4 mm, 146.2° | 22.4 | 2.43 | 244 |
