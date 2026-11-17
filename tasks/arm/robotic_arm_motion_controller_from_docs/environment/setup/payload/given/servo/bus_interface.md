# DA-6 desktop arm: bus interface manual

Document DA6-BUS, revision A. Applies to the DA-6 desktop arm (six BUS-SERVO-12V-A servos on one bus) driven by the
cell computer's bus master.

This manual describes the 10 ms bus cycle: what the bus master reads from the servos and hands to the motion
controller every cycle (the state), what it writes back (the command) and what the servos do with it. The Python calls
through which the controller receives the state and returns the command are specified in the motion controller
interface specification ([../controller/controller_interface.md](../controller/controller_interface.md)); the servo's
data are in its datasheet ([servo_datasheet.md](servo_datasheet.md)); the conversion between ticks and design angles
of a particular unit is its calibration record ([../unit/calibration_record.md](../unit/calibration_record.md)).

## 1. Servos on the bus

| array index | servo id | joint |
|---|---|---|
| 0 | 1 | base_yaw |
| 1 | 2 | shoulder_pitch |
| 2 | 3 | elbow_pitch |
| 3 | 4 | wrist_pitch |
| 4 | 5 | wrist_roll |
| 5 | 6 | jaw |

Every array of the state and of the command has six entries in this order. Physical layer: half-duplex asynchronous
serial (TTL level), 1 Mbit/s; the bus master addresses the six servos with one synchronous read and one synchronous
write per cycle.

## 2. The cycle

The bus master runs a fixed cycle of 10 ms (100 Hz) while a job's motion is running. In cycle k it

1. reads the six servos at the start of the cycle (the state of cycle k: section 3);
2. hands the state to the motion controller and receives its command (section 4);
3. writes the command to the six servos in the same cycle.

The servos execute the command after the command latency of 15-40 ms, counted from the start of cycle k. The latency
is constant for a given unit, the same for all six servos and not reported.

Example with a latency of 25 ms:

| time | event |
|---|---|
| 0.500 s | cycle k: the state is read, the controller's command is written |
| 0.510 s, 0.520 s | cycles k+1 to k+2: the servos still run on the previous goals |
| 0.525 s | the command of cycle k takes effect (latency 25 ms) |
| 0.530 s | cycle k+3: the first state that can show it |

### 2.1 Late replies

The controller has 10 ms of CPU time per cycle. The CPU time charged to a reply is all the CPU time the controller's
processes and threads used since its previous reply, the time between the calls and threads or processes that have
ended included (controller interface, section 3). A reply that used more is written floor(cpu / 10 ms) cycles late:
the bus master runs those cycles without writing (the servos keep their last goals and settings), then writes the
reply and only then reads the next state. No state is delivered for the cycles in between; the `cycle` and `t` fields
of the next state show the jump. A step that uses more than 1 s of CPU or 5 s of wall-clock time ends the controller
(controller interface, section 8).

Example: the reply to the state of cycle 40 used 23 ms of CPU; it is written in cycle 42 and the next state is that of
cycle 43.

### 2.2 Cycles without a new command

When nothing new is written (the reply has `goal` null or is null itself, or the bus master is waiting for a late
reply), every servo keeps its last goal and its speed and acceleration settings and continues its profile towards that
goal: the arm does not stop by itself. The same holds after a job has ended, until the next job's first command.

## 3. State (read every cycle)

| field | type | unit | content |
|---|---|---|---|
| cycle | int | - | bus cycle number, 0 in the job's first cycle |
| t | float | s | time since the job's start (0.01 s per cycle) |
| position | [6] int | ticks | present position of each output shaft, 0 to 4095 (datasheet section 3) |
| speed | [6] int | ticks/s | present speed of each output shaft, signed along +ticks |
| load | [6] int | per mille of 2.94 N m | present load, signed along +ticks, -1000 to +1000; reading scale 0.9-1.1 and a noise of about 1 % of the stall torque (datasheet section 3). The servo samples it every 20 ms and holds the whole sample in between: at the 10 ms cycle a new value arrives in every second state and the state between repeats it |
| moving | bool | - | true while any servo has a goal it has not reached (also one still within the command latency) or any output shaft turns faster than 0.01 rad/s |

Example: unit MU-DEMO1 at rest at the home pose (acceptance specification, section 1.1) in the example shelf cell
([../cells/example_scene_shelf.json](../cells/example_scene_shelf.json)): the readings of the `plan` call of the
cell's first job (acceptance specification, section 2) with the `cycle` and `t` of a job's first cycle; a job that
starts there reads these values in its first cycle:

```json
{"cycle": 0, "t": 0.0, "position": [1714, 2072, 1590, 2707, 2168, 1785], "speed": [0, 0, 0, 0, 0, 0], "load": [3, 86, 139, -19, 23, 2], "moving": false}
```

## 4. Command (written every cycle)

| field | type | unit | content |
|---|---|---|---|
| goal | [6] number or null | ticks | goal positions, rounded to the nearest whole tick; null = no new goals (the servos keep the last ones) |
| speed | [6] number or null | ticks/s | goal speed register: the profile's speed limit for these goals, written without its sign and rounded down to a whole number; 0 = the servo's maximum speed; null = 0 for all six |
| acc | [6] number or null | 100 ticks/s² | acceleration register: the profile's acceleration for these goals, written without its sign and rounded down to a whole number; 0 = the default 40 rad/s²; null = 0 for all six |
| path_start | bool | - | starts the reference of a path job (controller interface, section 7) |
| done | bool | - | the job is done (controller interface, section 7) |

- `goal`: rounded to the nearest whole tick (a value exactly halfway between two ticks to the even one). Each servo
  clamps its goal to its EEPROM angle limits, the `min_ticks` and `max_ticks` of the unit's calibration record,
  without an error (datasheet section 2.1): a goal outside them, also one below 0 or above 4095, runs to the nearer
  limit.
- `speed` and `acc` are written with the goals of the same reply and apply to them; with `goal` null they are not
  written. A goal written with `speed` (`acc`) null or 0 runs at the servo's maximum speed (the default acceleration),
  whatever was written before; so does one written with a value below 1, which rounds down to 0.
- Both registers can only lower the servo's own values: a speed above the servo's maximum speed has no effect, an
  acceleration register of 261 or more (26100 ticks/s² or more) selects the default 40 rad/s².
- Servo 6 (jaw) while a part or the needle is clamped in the gripper: it holds the clamp and its goal is ignored; its
  readings stay valid.

A command is written only when it is well formed: an object (or null: no command) whose `goal`, `speed` and `acc` are
each null or a list of exactly six finite numbers (true and false are not numbers), every `goal` entry within ±8192
and every `speed` and `acc` entry within ±10⁶; a missing key reads as null (`goal`, `speed`, `acc`) or false
(`path_start`, `done`). Anything else is a protocol error: nothing is written and the job is aborted (controller
interface, section 8).

Example (the goals of section 6.1, servo 2 with the speed and acceleration registers of section 6.3):

```json
{"goal": [2055, 1616, 1852, 2404, 3192, 1801], "speed": [0, 977, 0, 0, 0, 0], "acc": [0, 65, 0, 0, 0, 0], "path_start": false, "done": false}
```

## 5. Register units and quantisation

| quantity | register unit | SI |
|---|---|---|
| position | 1 tick | 2π/4096 rad = 1.5340 mrad = 0.08789° |
| speed | 1 tick/s | 1.5340 mrad/s |
| acceleration register | 1 = 100 ticks/s² | 0.15340 rad/s² |
| load | 1 per mille | 2.94 mN m |
| maximum speed (typical) | 4230.8 ticks/s | 6.49 rad/s |
| default acceleration | 260.76 register units (26076 ticks/s²) | 40 rad/s² |

- Goals are whole ticks (0.0879°) once rounded. Readings are whole numbers too: position in ticks, speed in ticks/s,
  load in per mille.
- Every position of a DA-6 lies in 0 to 4095 (datasheet sections 2.1 and 6): the readings, the record's `zero_ticks`,
  `min_ticks` and `max_ticks`, and so every goal the servos execute. The count does not pass from 4095 to 0 anywhere
  within a joint's travel.

## 6. Worked examples (record MU-DEMO1)

The record is [../unit/example_calibration_record.json](../unit/example_calibration_record.json); its conversions are
explained in [../unit/calibration_record.md](../unit/calibration_record.md).

### 6.1 Design angles to goal ticks

ticks = zero_ticks + direction x q x 4096 / 360 (q in degrees), rounded to the nearest whole tick; the servo then
clamps the goal to [min_ticks, max_ticks].

| servo | joint | direction | zero_ticks | q (°) | zero_ticks + direction x q x 4096/360 | goal (ticks) | [min_ticks, max_ticks] | inside | goal as design angle (°) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | base_yaw | +1 | 1714 | 30.0 | 2055.33 | 2055 | [517, 2933] | yes | 29.971 |
| 2 | shoulder_pitch | +1 | 1844 | -20.0 | 1616.44 | 1616 | [744, 2801] | yes | -20.039 |
| 3 | elbow_pitch | +1 | 2535 | -60.0 | 1852.33 | 1852 | [1373, 2744] | yes | -60.029 |
| 4 | wrist_pitch | -1 | 1892 | -45.0 | 2404.00 | 2404 | [945, 2827] | yes | -45.000 |
| 5 | wrist_roll | +1 | 2168 | 90.0 | 3192.00 | 3192 | [379, 3660] | yes | 90.000 |
| 6 | jaw | +1 | 1687 | 10.0 | 1800.78 | 1801 | [1629, 2758] | yes | 10.020 |

### 6.2 A goal beyond a limit

`wrist_roll` at q = 140.0°: 2168 + (+1) x 140.0 x 4096/360 = 3760.89, rounded 3761. That is above max_ticks = 3660:
the servo executes 3660, i.e. q = 131.133°.

### 6.3 Speed and acceleration registers

- Speed limit 1.5 rad/s for `shoulder_pitch`: 1.5 x 4096 / (2π) = 977.8 ticks/s, rounded down: register 977 (= 1.4987
  rad/s, the next lower step).
- Acceleration 10 rad/s² for `shoulder_pitch`: 10 x 4096 / (2π) = 6519.0 ticks/s² = 65.19 register units, rounded
  down: register 65 (= 6500 ticks/s² = 9.971 rad/s²).

### 6.4 Reading a state

The state example of section 3 converted with the record: q = direction x (ticks - zero_ticks) x 360/4096 degrees;
torque along +q = direction x load / 1000 x 2.94 N m (typical reading scale; the scale of a servo's reading lies
within 0.9-1.1). The homing tool commands the home pose through the unit's record (acceptance specification, section
1.1), so the readings there convert back to the home pose's design angles, here to within 0.03°; the arm's actual
angles differ from them by the record's error (calibration record, section 2).

| servo | joint | position (ticks) | design angle (°) | home pose (°) | load (per mille) | torque along +q (N m, typical scale) |
|---|---|---|---|---|---|---|
| 1 | base_yaw | 1714 | 0.00 | 0.00 | 3 | 0.009 |
| 2 | shoulder_pitch | 2072 | 20.04 | 20.05 | 86 | 0.253 |
| 3 | elbow_pitch | 1590 | -83.06 | -83.08 | 139 | 0.409 |
| 4 | wrist_pitch | 2707 | -71.63 | -71.62 | -19 | 0.056 |
| 5 | wrist_roll | 2168 | 0.00 | 0.00 | 23 | 0.068 |
| 6 | jaw | 1785 | 8.61 | 8.59 | 2 | 0.006 |
