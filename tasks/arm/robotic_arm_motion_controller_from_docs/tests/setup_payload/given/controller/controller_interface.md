# DA-6 cells: motion controller interface specification

Document LAB-DA6-CTL, revision A. Issued by the lab for the motion controller of its DA-6 desktop arm in the three
desktop cells (shelf, bins, dispense).

This specification states how the cell computer loads and calls the motion controller: the deliverable (section 1),
the run environment (2), the calls and their budgets (3), the payloads of `setup` (4) and `plan` (5), the 10 ms cycle
(6), how a job ends (7), failures and restarts (8) and the acceptance runs (9); section 10 describes the example
controller and the interface self-test.

Companion documents: the bus interface manual ([../servo/bus_interface.md](../servo/bus_interface.md): what the bus
values mean and what the servos do with a command), the servo datasheet
([../servo/servo_datasheet.md](../servo/servo_datasheet.md)), the unit calibration record
([../unit/calibration_record.md](../unit/calibration_record.md)), the kinematics sheet
([../robot/kinematics_sheet.md](../robot/kinematics_sheet.md)), the drawings
([../robot/drawings/](../robot/drawings/); frames, design angles, the TCP, the needle), the link CAD and the collision
envelope ([../robot/cad/README.md](../robot/cad/README.md)), the cell documentation
([../cells/cell_manual.md](../cells/cell_manual.md): the scene description and the clamped tools) and the acceptance
specification ([../acceptance/acceptance_spec.md](../acceptance/acceptance_spec.md): job types, tolerances, time
targets and the acceptance criteria).

## 1. Deliverable

`submission/controller/controller.py`, defining

```python
class Controller:
    def __init__(self, workdir): ...        # once per controller process
    def setup(self, unit, cell, tool): ...  # at the start of each cell
    def plan(self, job, state): ...         # before each job's motion: "ok" or "infeasible"
    def step(self, state): ...              # every 10 ms bus cycle of the job's motion: the command
```

together with every file it needs (your robot model, meshes, tables, modules) in the folder `submission/controller/`.
The controller must be self-contained: the controller computer has this folder and nothing else; these documents and
their files (the link CAD and the collision envelope of [../robot/cad/](../robot/cad/), the tables) are not on it.
Copy what the controller reads at run time into the folder and open it relative to the controller's own module
(`os.path.dirname(os.path.abspath(__file__))`). The folder holds at most 80 MB. It is copied to the controller
computer as it is, regular files and directories only (symbolic links and `__pycache__` are not copied), and is
read-only there.

A `plan` written with the job as its only argument (`def plan(self, job)`) is called with the job alone.

## 2. Run environment

| item | value |
|---|---|
| interpreter | Python 3.12 of the delivered image, started as `python3 -B -E -s` (no bytecode files, the PYTHON* environment variables ignored, no user site-packages) |
| working directory | the read-only copy of `submission/controller/`; it comes first on `sys.path`, so `controller.py` and your own modules are imported from it |
| packages | the standard library and every package installed in the image (numpy, scipy, pinocchio, trimesh, ...), except the modules and calls of the next row |
| not allowed | networking modules (socket, ssl, http, urllib, ftplib, smtplib, asyncio, selectors, xmlrpc, multiprocessing.connection and multiprocessing.managers, logging.handlers, requests and the like); ctypes and cffi; subprocess, os.system, os.popen, os.exec*, os.spawn*, pty; imports under a computed name (`__import__`, `importlib.import_module` or `find_spec` with an argument that is not a literal); exec, eval or compile of computed code |
| account | an unprivileged account |
| network | none: the controller computer has no network. Creating a network socket fails (`PermissionError`), for the controller and for every thread and process it starts; only local Unix-domain sockets can be created |
| files | only the copy of `submission/controller/`, the empty `workdir` (also the HOME) and a fresh TMPDIR; these documents and their files are not on the controller computer |
| threads | the BLAS threads (OMP, OpenBLAS, MKL) pinned to 1; at most 256 processes and threads |
| standard streams | stdin and stdout carry the cell computer's protocol: never read stdin or write to the original standard output (file descriptor 1); `print()` output goes to the run log (stdout is redirected to it before `controller.py` is imported) |
| end of the run | there is no closing call: use daemon threads; processes the controller starts must end with it (processes left running are stray processes and fail the acceptance) |

Arguments and return values are JSON: the arguments arrive as plain dicts, lists, strings, numbers, booleans and None,
and the return values must be plain Python values of the same kinds (`dict`, `list`, `str`, `int`, `float`, `bool`,
`None`). NumPy arrays and most NumPy scalars (`int64`, `float32`, `bool_`, ...) cannot be encoded; convert them
(`int(x)`, `float(x)`, `bool(x)`, `a.tolist()`). A reply that cannot be encoded counts as a call that raised (section
8).

## 3. Calls and budgets

The cell computer measures the CPU time of the controller's processes: its interpreter, its threads and every other
process of its account, that is every process the controller starts. The CPU time charged to a call is all the CPU
time they used since the controller's previous reply (for the first call of a controller process: since that call),
threads and processes that have ended by the reply included, so work done between the calls, by threads or by other
processes, counts against the next call.

| call | when | arguments | returns | budget |
|---|---|---|---|---|
| `Controller(workdir)` | once per controller process (also after a restart) | `workdir`: an empty writable directory | - | import of `controller.py` + `__init__` <= 60 s wall-clock |
| `setup(unit, cell, tool)` | at the start of each cell (three per acceptance run), the arm at rest at the home pose with the cell's tool clamped; again after a restart (the arm then stands where the last job left it) | section 4 | ignored | <= 60 s CPU, <= 120 s wall-clock |
| `plan(job, state)` | before each job's motion; the arm holds | section 5 | `"ok"` or `"infeasible"` | <= 20 s CPU, <= 60 s wall-clock |
| `step(state)` | every 10 ms bus cycle of the job's motion | section 6 | the command (section 6) | 10 ms CPU per cycle; more than 1 s CPU or 5 s wall-clock ends the process |

## 4. setup(unit, cell, tool)

### unit

The unit's serial number and its calibration record (format and conversions:
[../unit/calibration_record.md](../unit/calibration_record.md)); abbreviated example:

```text
{
 "serial": "MU-DEMO1",
 "calibration": {
  "format": "calibration-record/1",
  "serial": "MU-DEMO1",
  "convention": "q = direction x (ticks - zero_ticks) x 2 pi / 4096 (q = design angle of the kinematics sheet)",
  "uncertainty_deg": {
   "arm joints": 0.5,
   "jaw": 1.0
  },
  "servos": {
   "base_yaw": {
    "id": 1,
    "direction": 1,
    "zero_ticks": 1714,
    "min_ticks": 517,
    "max_ticks": 2933
   },
   "shoulder_pitch": {
    "id": 2,
    "direction": 1,
    "zero_ticks": 1844,
    "min_ticks": 744,
    "max_ticks": 2801
   },
   ... (six servos)
  }
 }
}
```

### cell

The scene description of the cell, format `scene-description/1`, in the cell frame (origin at the cell's reference
corner on the table top, x and y in the table plane, z up, the table top at z = 0; the robot base and every object at
positive x and y; metres, degrees): the robot base's pose in the cell (`robot_base`: position and yaw about z), every
object as primitives in the object's own frame (`pose`: position and yaw about z), the table and the survey tolerance.
The cell documentation defines it ([../cells/cell_manual.md](../cells/cell_manual.md), sections 2 and 6); abbreviated
example, the example shelf cell ([../cells/example_scene_shelf.json](../cells/example_scene_shelf.json)):

```text
{
 "format": "scene-description/1",
 "frame": "cell (the table's reference corner): x, y >= 0 on the table, z up; metres, degrees",
 "robot_base": {
  "pos": [0.258185, 0.359614, 0.0],
  "yaw_deg": -95.5439
 },
 "objects": [
  {
   "name": "shelf",
   "kind": "shelf",
   "pose": {
    "pos": [0.531973, 0.151721, 0.0],
    "yaw_deg": -126.439
   },
   "primitives": [
    {
     "name": "shelf_plinth",
     "type": "box",
     "size": [0.10079, 0.03461, 0.01763],
     "pos": [0.0, 0.0, 0.01763],
     "quat": [1.0, 0.0, 0.0, 0.0]
    },
    {
     "name": "shelf_back",
     "type": "box",
     "size": [0.10079, 0.002, 0.1251],
     "pos": [0.0, 0.03261, 0.16035],
     "quat": [1.0, 0.0, 0.0, 0.0]
    },
    ... (8 primitives in all)
   ]
  },
  ... (4 objects in all)
 ],
 "table": {
  "z": 0.0,
  "note": "the robot base stands on the table; the table top is z = 0"
 },
 "survey_tolerance": {
  "objects_mm": 2.0,
  "objects_deg": 1.0,
  "robot_base_mm": 1.0,
  "robot_base_deg": 0.3
 }
}
```

### tool

What is clamped in the gripper for the whole cell. The fields, the part's seat and the needle's holder are defined in
the cell documentation ([../cells/cell_manual.md](../cells/cell_manual.md), sections 8 and 9), lengths in mm in the
TCP frame (section 5):

| cell | tool | meaning |
|---|---|---|
| shelf | `{"kind": "gripper"}` | the bare gripper; servo 6 (the jaw) is the controller's |
| bins | `{"kind": "part", "half_mm": [12.1, 11.75, 12.9], "seat_mm": {"x": 4.26, "z": 5.16}, "mass_range_kg": [0.03, 0.15], "com_note": "the centre of mass is inside the part; not known"}` (the part of the example bin cell) | a box part clamped across the jaws for the whole cell: its -x face on the fixed finger's pad face, its centre at (`seat_mm.x`, 0, `seat_mm.z`) in the TCP frame, its axes parallel to the TCP frame, half sizes `half_mm`; servo 6 holds the jaw closed on it; its mass (within `mass_range_kg`) and its centre of mass (inside the part) are not given |
| dispense | `{"kind": "needle", "length_mm": 45.0, "radius_mm": 1.2, "mass_kg": 0.02, "axis_offset_x_mm": -0.001, "holder_half_mm": [7.84, 6.0, 9.0], "holder_centre_z_mm": -3.0}` | the dispensing needle in its holder block, clamped in the pad gap for the whole cell: the holder (half sizes `holder_half_mm`, centre at (`axis_offset_x_mm`, 0, `holder_centre_z_mm`) in the TCP frame) and the needle (radius `radius_mm`) along the TCP's +z up to its tip at (`axis_offset_x_mm`, 0, `length_mm`), the tool point; `mass_kg` is that of the needle and the holder together |

While a part or the needle is clamped, servo 6 holds the clamp: its goal is ignored and its readings stay valid. The
clamped tool belongs to the arm: its contact with the cell, the table or a link other than the wrist, the gripper and
the jaw stops the arm, and the clearance is measured from it as from the links (acceptance specification, sections 1.2
and 4.1).

## 5. plan(job, state)

Jobs are given in the cell frame: positions in metres, directions as unit vectors, `_mm` and `_deg` fields in
millimetres and degrees. The TCP is defined in the kinematics sheet
([../robot/kinematics_sheet.md](../robot/kinematics_sheet.md), section 5) and on the gripper drawing
([../robot/drawings/gripper.png](../robot/drawings/gripper.png)): the point between the finger pads, at the pads'
height, 0.2 mm off the wrist-roll axis, in the middle of the gap when the pads are 15.68 mm apart (the needle holder's
width; at the jaw's reference angle q6 = 0 the pads are 15.80 mm apart and the middle of the gap lies 0.06 mm from the
TCP towards the moving jaw); TCP frame: +z = the approach direction (out of the fingers, along the roll axis), +x =
the jaw axis (from the fixed finger towards the moving jaw). The tool point is the TCP, or with the needle its tip,
the point (`axis_offset_x_mm`, 0, `length_mm`) = (-0.001, 0, 45) mm of the TCP frame (the needle drawing,
[../robot/drawings/needle.png](../robot/drawings/needle.png)).

### state

`state` holds the bus readings where the arm stands while `plan` runs: the fields of the cycle's state (section 6)
without `cycle` and `t`. Nothing moves between `plan` and the job's first cycle, which reads the same values.

```text
state = {"position": [6 int], "speed": [6 int], "load": [6 int], "moving": bool}
```

Example: the first job of the example shelf cell, unit MU-DEMO1 at rest at the home pose (section 9):

```json
{"position": [1714, 2072, 1590, 2707, 2168, 1785], "speed": [0, 0, 0, 0, 0, 0], "load": [3, 86, 139, -19, 23, 2], "moving": false}
```

### Fields of every job

| field | type | content |
|---|---|---|
| id | string | the job's name |
| type | string | `"reach"` or `"path"` |
| tool | string | `"gripper"`, `"gripper+part"` or `"needle"`: the tool point is the TCP (gripper, gripper+part) or the needle tip (needle) |
| clearance_mm | number | the clearance the motion keeps to the cell (10 mm) |
| time_target_s | number | the job's cycle-time target (s) |
| exempt | [string] | names of objects or primitives (the job's own support) that the job's working element may come closer to than the clearance; during the job the support of the previous job of the cell whose motion ran stays exempt too (acceptance specification, section 3.4); contact with them still counts |

### Reach jobs (`type` = "reach")

| field | type | content |
|---|---|---|
| position | [3] m | where the tool point must be |
| approach | [3] unit vector | the direction of the TCP's +z (the approach direction) |
| approach_tol_deg | number | 5: the largest angle between the TCP's +z and `approach` |
| jaw_axis | [3] unit vector or null | the direction of the TCP's +x (the jaw axis), either sign; null = free |
| jaw_tol_deg | number | 5: the largest angle between the TCP's +x (or -x) and `jaw_axis` |

A reach job is done when the tool point is at `position`, with the approach (and the jaw axis) inside their
tolerances, and the arm is at rest. Example: job S01 of the example shelf job list
([../acceptance/example_jobs_shelf.json](../acceptance/example_jobs_shelf.json)):

```json
{
 "id": "S01",
 "type": "reach",
 "tool": "gripper",
 "clearance_mm": 10.0,
 "time_target_s": 2.0,
 "exempt": ["shelf_board1"],
 "position": [0.48606, 0.10565, 0.21995],
 "approach": [0.613718, -0.68103, -0.399435],
 "approach_tol_deg": 5.0,
 "jaw_axis": [0.267398, -0.296726, 0.916762],
 "jaw_tol_deg": 5.0
}
```

### Path jobs (`type` = "path", tool "needle")

| field | type | content |
|---|---|---|
| t | [n] s | the reference's sample times: 0, 0.01, 0.02, ... s |
| points | [n][3] m | the needle tip's reference p(t) at those times, already 3 mm off the board along `normal` |
| normal | [3] unit vector | the board's surface normal |
| axis_tol_deg | number | 8: the needle axis (the TCP's +z) must stay within this angle of -normal |
| gap_mm | number | 3: the needle tip's distance from the board |
| gap_tol_mm | number | 1.5: the tolerance of that distance |
| speed_mm_s | number | the tip's speed along the path |
| shape | object | `{"shape": "line", "length_mm"}` or `{"shape": "arc", "radius_mm", "sweep_deg"}` |
| approach_time_target_s | number | the time target for bringing the tip to `points[0]` |

Example: job D01 of the example dispensing job list
([../acceptance/example_jobs_dispense.json](../acceptance/example_jobs_dispense.json)), an arc over the board:

```text
{
 "id": "D01",
 "type": "path",
 "tool": "needle",
 "clearance_mm": 10.0,
 "time_target_s": 5.85,
 "exempt": ["board"],
 "t": [0.0, 0.01, 0.02, ..., 3.4],
 "points": [[0.52783, 0.27507, 0.05288], [0.52784, 0.27529, 0.05289], ..., [0.50048, 0.26191, 0.04146]],
 "normal": [-0.385951, 0.001351, 0.922518],
 "axis_tol_deg": 8.0,
 "gap_mm": 3.0,
 "gap_tol_mm": 1.5,
 "speed_mm_s": 21.93,
 "shape": {
  "shape": "arc",
  "radius_mm": 18.2,
  "sweep_deg": 234.7
 },
 "approach_time_target_s": 1.65
}
(t and points shortened: 341 samples)
```

### The answer

`plan` returns `"ok"` (the job's motion follows: `step`) or `"infeasible"`: the target or the path cannot be reached
within the tolerances and the clearance. After `"infeasible"` the arm holds and the next job follows. Jobs that cannot
be done have the same fields as the others. Any return value other than the string `"infeasible"` is read as `"ok"`;
after `"ok"` the job's motion runs, also for a job that cannot be done. A `plan` that raises skips the job (section
8).

## 6. step(state) and the command

The fields, their units and what the servos do with them are specified in the bus interface manual
([../servo/bus_interface.md](../servo/bus_interface.md), sections 3 to 5).

```text
state = {"cycle": int, "t": float, "position": [6 int], "speed": [6 int], "load": [6 int], "moving": bool}
reply = {"goal": [6 number] | None, "speed": [6 number] | None, "acc": [6 number] | None,
         "path_start": bool, "done": bool}
```

Example of a state and a reply (unit MU-DEMO1: the first cycle of a job that starts at the home pose, with the
readings of the `plan` example of section 5; the reply sends the goals of the bus interface manual, sections 4 and 6,
which explain the numbers):

```json
{"cycle": 0, "t": 0.0, "position": [1714, 2072, 1590, 2707, 2168, 1785], "speed": [0, 0, 0, 0, 0, 0], "load": [3, 86, 139, -19, 23, 2], "moving": false}
{"goal": [2055, 1616, 1852, 2404, 3192, 1801], "speed": [0, 977, 0, 0, 0, 0], "acc": [0, 65, 0, 0, 0, 0], "path_start": false, "done": false}
```

- Arrays are in servo order (ids 1 to 6: `base_yaw`, `shoulder_pitch`, `elbow_pitch`, `wrist_pitch`, `wrist_roll`,
  `jaw`).
- `goal`: six numbers in ticks, rounded to the nearest whole tick, or null to keep the last goals. `speed`, `acc`: six
  numbers or null, written without sign and rounded down to whole numbers (null or 0 = the servo's maximum speed / the
  default acceleration), applied to the goals of the same reply.
- A missing key is read as null (`goal`, `speed`, `acc`) or false (`path_start`, `done`); a step that returns None
  sends nothing and is not done. `path_start` and `done` are read as Python's `bool()` of the value (0, an empty value
  and null are false): send true or false.
- A reply that is not an object (or None), or whose `goal`, `speed` or `acc` is neither null nor a list of exactly six
  finite numbers (true and false are not numbers; `goal` entries within ±8192, `speed` and `acc` entries within ±10⁶),
  is a protocol error: nothing is written and the job is aborted (section 8).
- Timing: the step of a cycle may use 10 ms of CPU, counted since the controller's previous reply (section 3). A step
  that used more is executed floor(cpu / 10 ms) cycles late and the cycles in between deliver no state (bus interface
  manual, section 2.1); the `cycle` and `t` of the next state show it. A step over 1 s CPU or 5 s wall-clock ends the
  controller (section 8).
- Acceptance gate: at least 90 % of all bus cycles of each acceptance run answered within 10 ms of CPU (the cycle of a
  late step and the cycles it delays count as not answered in time).

## 7. How a job ends

- Reach: the controller sets `done` once the job is done (section 5). The command of that reply is still executed; the
  bus then keeps the last goals and `step` is not called; 0.5 s later the result is measured, and `plan` is called for
  the next job.
- Path: the controller first brings the needle tip to `points[0]` (untimed within the job's cap) and then sets
  `path_start` in one reply. The reference starts at the end of the cycle in which that reply is written to the bus
  (normally the reply's own cycle: the state's `t` + 0.01 s; later if the reply is late; the servos' command latency
  does not enter) and runs for `t[-1]` seconds: the tip should be at `points[i]` at the reference's start + `t[i]`.
  Only the first `path_start` counts; it is ignored in reach jobs. The controller sets `done` after the path has
  ended; a path job without `path_start`, or with `done` before the end of the path, is not completed.
- Cap: a job without `done` is ended at its cap, counted from its first cycle: reach jobs max(2 x `time_target_s`,
  `time_target_s` + 2 s), path jobs 2 x `approach_time_target_s` + `t[-1]` + 2 s. It is not completed.

## 8. Failures and restarts

| event | what follows |
|---|---|
| `setup` raises, exceeds a budget or the process dies (also when `controller.py` cannot be imported and `Controller(workdir)` built within 60 s) | the controller is restarted (a new process with a fresh, empty `workdir`: import, `Controller(workdir)`) and `setup` is called once more; if that fails too, the cell's remaining jobs are not run (none of them is completed and none counts as a refusal) |
| `plan` raises | the job is skipped: it is not run and it does not count as a refusal; the next job's `plan` follows (the controller is not restarted) |
| `plan` exceeds a budget or the process dies | the job is not run; the controller is restarted and `setup` is called again for the current cell before the next job |
| `step` raises, or its reply cannot be encoded | the job is aborted: it is not completed; the servos keep the last goals; the next job's `plan` follows |
| `step` returns a malformed reply (section 6) | a protocol error: nothing is written and the job is aborted as above |
| `step` uses more than 10 ms CPU | its command is executed floor(cpu / 10 ms) cycles late; no state is delivered for the cycles in between; these cycles count as not answered in time |
| `step` uses more than 1 s CPU or 5 s wall-clock, or the process dies | the job is aborted; the controller is restarted and `setup` is called again before the next job |
| the controller's answer to a call is not one JSON line of at most 1 MiB (1048576 bytes), e.g. because something was written to the original standard output or a reply is larger | the connection is broken: the controller is ended and restarted and `setup` is called again (for `setup` itself: as in the first row); the call's job is not run (`plan`) or aborted (`step`) |
| contact (protective stop) | the arm is stopped and the cell fails: the job and every later job of the cell fail (the later jobs are not run: none of them is completed and none counts as a refusal); the next cell starts as usual, at the home pose with its `setup` (after the dispensing cell the run ends); the controller is not told: its next call is that `setup` |
| no `done` within the job's cap | the job is not completed; the next job starts where the arm is |

A restarted controller is a new process: nothing of the previous one is kept, its `workdir` neither.

## 9. The acceptance runs

- Three runs, each on its own unit (three units, one run on each; the unit's record arrives in `setup`). Each run
  starts a new controller process (`Controller(workdir)` with an empty `workdir`) and works the three cells in the
  order shelf, bins, dispense; about 14 jobs per cell, run in the given order, each once.
- Each cell starts with the arm at rest at the home pose and the cell's tool clamped; then `setup` is called. The
  operator's homing tool commands the home pose through the unit's calibration record: the design angles base yaw 0°,
  shoulder pitch 20.05°, elbow pitch -83.08°, wrist pitch -71.62°, wrist roll 0° and jaw 8.59°, or the clamp angle
  with a part or the needle clamped (acceptance specification, section 1.1). The `state` of each `plan` tells where
  the arm is.
- Each job starts where the previous one ended. A contact stops the arm and ends the cell: its later jobs are not run
  and the next cell starts at the home pose with its `setup` (section 8).
- Each acceptance run is the controller's first run on its unit: there is no run on a unit before it.
- Job types, tolerances, time targets and the acceptance criteria (the three runs count equally): the acceptance
  specification ([../acceptance/acceptance_spec.md](../acceptance/acceptance_spec.md)).

## 10. Example controller and self-test

- [example_controller.py](example_controller.py): reads the record, converts ticks and design angles, holds the arm
  where `plan` found it and reports each job done after 1 s (a path job: `path_start` in its first cycle, `done` when
  the path's time has run out). It shows the interface; it is not a motion controller.
- [check_controller.py](check_controller.py): `python3 check_controller.py PATH/TO/controller.py`. It runs the
  controller as described in sections 2 to 8, with payloads in the formats above (`plan` with the state, or with the
  job alone for the older form), charges each call the CPU time used since the previous reply and checks every reply
  as the cell computer does, against a toy plant of six toy servos (it is not the DA-6). It reports protocol errors,
  exceptions, budget overruns and the fraction of cycles answered in time; `--help` lists its options (another unit
  record, your own cells and jobs). It checks the interface and the timing only.
