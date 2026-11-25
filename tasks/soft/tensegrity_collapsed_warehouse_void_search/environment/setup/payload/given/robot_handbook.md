# RT-3 search tensegrity: robot handbook

This handbook describes the team's robot as the station runs it. Every table of measurements in it was taken on this
robot; where something has not been measured, the handbook says so.

## 1. What the robot is

The RT-3 is a three-strut tensegrity: three rigid struts that never touch each other, held in shape by a net of six
cables that the robot can shorten and lengthen and nine elastic stretch sensors that pull back. Each strut is an
aluminium rod that carries a motor module near either end, the strut's electronics at its middle, and an end cap at
either tip. The centre of an end cap is a **node**; the robot has six. It moves by changing the lengths of
its six cables: the shape shifts, the weight goes over an edge, and the robot tips onto its next face. That is one
roll.

The robot has no camera, no range sensor and no computer that plans. Each strut's controller board reads its sensors
and drives its two motors; everything else runs on the team's PC, which closes the cable-length loop over the radio
at the control rate. The program you write runs on that PC, next to the cable controller (section 5).

<!-- BEGIN GENERATED dimensions -->
| quantity | value |
|---|---|
| struts | 3 |
| nodes (end-cap centres) of one strut, apart | 290.8 mm |
| end cap | a sphere of 35 mm diameter around the node, in the strut's colour |
| mass | 242.9 g per strut, 0.73 kg in all |
| body length (the robot standing in its rest shape) | about 0.33 m |
| height of the highest point, standing in the rest shape | about 0.23 m |
| floor of this handbook's measurements | flat concrete, friction coefficient 0.80 |
| floors of the sites | flat concrete, friction coefficient between 0.76 and 0.92, one value per site |
<!-- END GENERATED dimensions -->

Purchased components that the program meets through the telemetry:

| component | per robot | vendor and part number |
|---|---|---|
| gear motor with encoder, one per cable | 6 | Pololu 5192 (150:1 Micro Metal Gearmotor HPCB 6V with 12 CPR Encoder, Back Connector) |
| strut rod | 3 | McMaster-Carr 8974K19 (aluminium rod, 1/8 in diameter, 12 in long) |
| controller board with the strut's IMU | 3 | Arduino Nano 33 IoT, ABX00027; its on-board IMU is an STMicroelectronics LSM6DS3 (three-axis accelerometer and three-axis gyroscope; no magnetometer) |

## 2. Numbering and colours

<!-- BEGIN GENERATED numbering -->
| strut | even node | odd node | colour of both end caps |
|---|---|---|---|
| 0 | 0 | 1 | red |
| 1 | 2 | 3 | green |
| 2 | 4 | 5 | blue |

| motor, line and short sensor | motor at node | line and short sensor run to node |
|---|---|---|
| 0 | 5 | 3 |
| 1 | 3 | 1 |
| 2 | 1 | 5 |
| 3 | 2 | 0 |
| 4 | 0 | 4 |
| 5 | 4 | 2 |

| long sensor | between nodes |
|---|---|
| 6 | 2 and 5 |
| 7 | 0 and 3 |
| 8 | 1 and 4 |

Rollable faces (the node triples a primitive can start from): (0, 2, 5) (0, 3, 4) (0, 3, 5) (1, 2, 4) (1, 2, 5) (1, 3, 4). End triangles: (0, 2, 4) (1, 3, 5).
<!-- END GENERATED numbering -->

Both end caps of a strut have the same colour: the colour tells the strut, not which of its two ends is which.

Motor i winds line i. Short sensor i is stretched between the same two nodes as line i, so its length is the length
that motor i controls. The long sensors are not driven; they only pull.

## 3. Rest shape and faces

In the **rest shape** all six short sensors are at their rest target (section 5). The robot then stands on three
nodes, one of each strut. Of the eight triples with one node of each strut, six are **rollable faces**: a primitive
can start from them. The other two are the **end triangles**: the robot stands on one end of every strut, like a
tripod on its tips, and no primitive is defined from there. Standing on an end triangle, the robot can only be moved
with `set_targets` shapes of your own (section 5); the handbook has no procedure for it.

After any primitive on a free floor the robot stands on one of three of the six rollable faces; the other three occur
after the set-down and after a roll into a wall or a block (section 7 gives both groups).

## 4. Planar pose

The station, the scoring and this handbook use one convention for where the robot is on the floor:

- **position** = the centroid of the six nodes, projected on the floor;
- **heading** = the direction from the mean of the odd nodes (1, 3, 5) to the mean of the even nodes (0, 2, 4),
  projected on the floor, as an angle counter-clockwise from the site's +x axis;
- a **displacement** (dx, dy, dθ) is the pose after a primitive expressed in the frame of the pose before it: dx along
  the old heading, dy to the left of it (the old heading turned by +90°), dθ the change of heading, positive
  counter-clockwise seen from above.

The robot rolls sideways: a forward roll moves it mainly along +dy.

## 5. The cable controller

<!-- BEGIN GENERATED control -->
| item | value |
|---|---|
| control rate | 100 Hz: the controller reads the six short sensors and sets the six motor duties once per control period of 10 ms |
| target of motor k | a length of short sensor k; as a normalised target x it means 100 mm + x × the range (the range is 100 mm unless a primitive sets another) |
| done | motor k is done with a target state, and is stopped, once its sensor is within 0.05 of the range of its target (5 mm at a range of 100 mm); the flag holds until the state ends |
| end of a target state | when every motor is done or, at that control period, is being asked to pay out a line that is already slack; or after 10 s (a timeout) |
| `set_targets(lengths_mm)` | one target state, six lengths from 100 to 240 mm, at the range 100 mm; it ends as a state ends, at the latest after 10 s; no return to the rest shape, no hold |
| rest shape | all six short sensors at 200 mm (the normalised target 1 at the range 100 mm) |
| end of a primitive | after its own states the robot is driven back to the rest shape, then the motors are held braked for 0.92 s; a primitive that has not ended after 120 s is ended |
| encoders | 1806.96 counts per turn of the output shaft (12 counts per motor turn × the gear ratio 150.58); the spool radius is 3.175 mm, so one count is 0.01104 mm of line |
<!-- END GENERATED control -->

A **command** is one primitive or one `set_targets`. One command runs at a time; it runs while your program steps the
simulated time, and `stop()` aborts it and brakes the motors where they are. The motors hold the lines braked
whenever no command runs.

What the controller tells your program about a running command is `status()`: `busy`; `kind` (`"primitive"` or
`"set_targets"`); `state_index`, the number of the target state it is working on, counted from 0 (for a primitive the
last number is the return to the rest shape, and during the braked hold it reads `"settle"`); `done`, the six done
flags of that state; `timeouts`, the number of states of this command that ended by their timeout. When no command
runs, `busy` is false, `kind` and `state_index` are null, the flags are false and `timeouts` reads 0: read `timeouts`
before the command has ended, during the hold at the latest. Nothing in `status()` or in any other reply says where
the robot is or whether it moved.

## 6. The primitives

A primitive is a fixed sequence of target states. The eleven primitives are nine forward rolls, named
`roll_<L>_<R>`, and two turns on the spot, `ccw` (counter-clockwise seen from above) and `cw`. There is no primitive
that rolls backwards: the robot reverses by turning.

<!-- BEGIN GENERATED primitive_states -->
| kind | target states, in order (x0 … x5 for motors 0 … 5 on the reference faces) |
|---|---|
| roll | [1, 1, 1, 1, 1, 1] → [1, 1, 0.1, 1, 1, 0.1] → [0, 1, 1, 0, 1, 0.1] → [1, 1, 1, 1, 1, 1] |
| ccw | [1, 1, 1, 1, 1, 1] → [1, 1, 1, 0, 1, 1] → [1, 0, 1, 0, 1, 1] → [0, 0, 0, 0, 0, 0] → [1, 1, 1, 1, 1, 1] |
| cw | [1, 1, 1, 1, 1, 1] → [0, 0, 0, 1, 0, 1] → [0, 0, 0, 0, 0, 0.7] → [0, 0, 0.7, 0, 1, 1] → [1, 1, 1, 1, 1, 1] |

| primitive | kind | L: range of motors 0, 1, 2 (mm) | R: range of motors 3, 4, 5 (mm) |
|---|---|---|---|
| `roll_100_100` | roll | 100 | 100 |
| `roll_120_120` | roll | 120 | 120 |
| `roll_140_140` | roll | 140 | 140 |
| `roll_100_120` | roll | 100 | 120 |
| `roll_120_100` | roll | 120 | 100 |
| `roll_100_140` | roll | 100 | 140 |
| `roll_140_100` | roll | 140 | 100 |
| `roll_120_140` | roll | 120 | 140 |
| `roll_140_120` | roll | 140 | 120 |
| `ccw` | ccw | 100 | 100 |
| `cw` | cw | 100 | 100 |
<!-- END GENERATED primitive_states -->

A normalised target x of motor k means the length `100 mm + x × range` of short sensor k, with the range L for motors
0, 1, 2 and R for motors 3, 4, 5. The states above are written for the reference faces (the first row of the table
below). For another face the six targets are permuted, and that is what `bottom_nodes` is for:

<!-- BEGIN GENERATED symmetry -->
| `bottom_nodes` | m[0] … m[5]: motor k takes the base target x of index m[k] |
|---|---|
| (0, 2, 5) or (0, 3, 5) | 0 1 2 3 4 5 |
| (1, 2, 4) or (1, 2, 5) | 1 2 0 4 5 3 |
| (0, 3, 4) or (1, 3, 4) | 2 0 1 5 3 4 |

The two end triangles (0, 2, 4) and (1, 3, 5) are not in the table: no primitive starts from them.
<!-- END GENERATED symmetry -->

`run_primitive(name, bottom_nodes)` takes the three nodes that are on the ground **as your program says**, looks up
m, and sends motor k the target x[m[k]] of every state, at its own range (L or R by k, not permuted). The station does
not compare your triple with the truth: a wrong triple makes the robot execute the primitive of another face, and
what it does then is in no table. Telling which three nodes are down is your program's
job; section 9 says what the telemetry offers for it.

## 7. The primitive table: this robot on a flat floor

Measured on this robot, on the flat concrete floor of the first table of section 1, in the convention of section 4.
A forward roll ends on another face than it started from; a turn from one of the first table's faces ends on the same
face.

<!-- BEGIN GENERATED primitive_table -->
**Start faces (0, 3, 5), (1, 2, 5), (1, 3, 4): the faces the robot stands on after any primitive that ended normally on a free floor.**

| primitive | start face | dx (m) | dy (m) | dθ (°) | duration (s) | landing face |
|---|---|---|---|---|---|---|
| `roll_100_100` | (0, 3, 5) | +0.062 ± 0.004 | +0.230 ± 0.002 | -9.1 ± 1.3 | 5.80 ± 0.10 | (1, 2, 5) |
| `roll_100_100` | (1, 2, 5) | +0.065 ± 0.005 | +0.231 ± 0.002 | -8.8 ± 1.1 | 5.83 ± 0.11 | (1, 3, 4) |
| `roll_100_100` | (1, 3, 4) | +0.064 ± 0.007 | +0.231 ± 0.002 | -9.7 ± 1.1 | 5.91 ± 0.15 | (0, 3, 5) |
| `roll_120_120` | (0, 3, 5) | +0.054 ± 0.007 | +0.262 ± 0.002 | -9.3 ± 1.4 | 7.32 ± 0.05 | (1, 2, 5) |
| `roll_120_120` | (1, 2, 5) | +0.056 ± 0.003 | +0.263 ± 0.003 | -9.3 ± 0.7 | 7.32 ± 0.05 | (1, 3, 4) |
| `roll_120_120` | (1, 3, 4) | +0.055 ± 0.008 | +0.262 ± 0.002 | -9.5 ± 1.4 | 7.29 ± 0.05 | (0, 3, 5) |
| `roll_140_140` | (0, 3, 5) | +0.021 ± 0.007 | +0.271 ± 0.002 | -17.9 ± 1.6 | 8.13 ± 0.08 | (1, 2, 5) |
| `roll_140_140` | (1, 2, 5) | +0.020 ± 0.005 | +0.272 ± 0.003 | -17.6 ± 1.2 | 8.15 ± 0.07 | (1, 3, 4) |
| `roll_140_140` | (1, 3, 4) | +0.019 ± 0.011 | +0.273 ± 0.004 | -18.0 ± 2.3 | 8.12 ± 0.07 | (0, 3, 5) |
| `roll_100_120` | (0, 3, 5) | +0.052 ± 0.004 | +0.254 ± 0.002 | -3.5 ± 0.9 | 7.08 ± 0.06 | (1, 2, 5) |
| `roll_100_120` | (1, 2, 5) | +0.052 ± 0.002 | +0.255 ± 0.001 | -3.5 ± 0.6 | 7.14 ± 0.04 | (1, 3, 4) |
| `roll_100_120` | (1, 3, 4) | +0.051 ± 0.007 | +0.254 ± 0.002 | -3.1 ± 1.4 | 7.08 ± 0.07 | (0, 3, 5) |
| `roll_120_100` | (0, 3, 5) | +0.074 ± 0.003 | +0.241 ± 0.005 | -14.8 ± 0.5 | 6.69 ± 0.04 | (1, 2, 5) |
| `roll_120_100` | (1, 2, 5) | +0.075 ± 0.003 | +0.242 ± 0.003 | -15.3 ± 0.9 | 6.71 ± 0.05 | (1, 3, 4) |
| `roll_120_100` | (1, 3, 4) | +0.074 ± 0.004 | +0.242 ± 0.003 | -14.3 ± 1.1 | 6.71 ± 0.05 | (0, 3, 5) |
| `roll_100_140` | (0, 3, 5) | +0.031 ± 0.006 | +0.264 ± 0.002 | +1.0 ± 1.3 | 8.45 ± 0.07 | (1, 2, 5) |
| `roll_100_140` | (1, 2, 5) | +0.034 ± 0.004 | +0.266 ± 0.001 | +0.2 ± 1.0 | 8.45 ± 0.06 | (1, 3, 4) |
| `roll_100_140` | (1, 3, 4) | +0.031 ± 0.007 | +0.264 ± 0.001 | +0.8 ± 1.6 | 8.43 ± 0.05 | (0, 3, 5) |
| `roll_140_100` | (0, 3, 5) | +0.086 ± 0.003 | +0.256 ± 0.005 | -25.3 ± 0.9 | 7.13 ± 0.08 | (1, 2, 5) |
| `roll_140_100` | (1, 2, 5) | +0.090 ± 0.004 | +0.256 ± 0.005 | -26.2 ± 1.8 | 7.07 ± 0.03 | (1, 3, 4) |
| `roll_140_100` | (1, 3, 4) | +0.085 ± 0.005 | +0.252 ± 0.004 | -24.8 ± 1.0 | 7.05 ± 0.06 | (0, 3, 5) |
| `roll_120_140` | (0, 3, 5) | +0.028 ± 0.006 | +0.265 ± 0.001 | -6.3 ± 1.1 | 8.31 ± 0.06 | (1, 2, 5) |
| `roll_120_140` | (1, 2, 5) | +0.032 ± 0.003 | +0.267 ± 0.002 | -6.6 ± 0.6 | 8.38 ± 0.08 | (1, 3, 4) |
| `roll_120_140` | (1, 3, 4) | +0.026 ± 0.008 | +0.266 ± 0.002 | -5.8 ± 1.5 | 8.33 ± 0.03 | (0, 3, 5) |
| `roll_140_120` | (0, 3, 5) | +0.051 ± 0.008 | +0.272 ± 0.004 | -17.7 ± 1.6 | 7.36 ± 0.08 | (1, 2, 5) |
| `roll_140_120` | (1, 2, 5) | +0.055 ± 0.003 | +0.272 ± 0.003 | -18.7 ± 1.2 | 7.40 ± 0.08 | (1, 3, 4) |
| `roll_140_120` | (1, 3, 4) | +0.051 ± 0.008 | +0.268 ± 0.002 | -17.4 ± 2.2 | 7.34 ± 0.07 | (0, 3, 5) |
| `ccw` | (0, 3, 5) | -0.015 ± 0.001 | -0.042 ± 0.001 | +23.9 ± 0.6 | 7.35 ± 0.10 | (0, 3, 5) |
| `ccw` | (1, 2, 5) | -0.014 ± 0.002 | -0.040 ± 0.001 | +23.7 ± 0.4 | 7.37 ± 0.10 | (1, 2, 5) |
| `ccw` | (1, 3, 4) | -0.014 ± 0.002 | -0.042 ± 0.002 | +24.3 ± 0.6 | 7.40 ± 0.14 | (1, 3, 4) |
| `cw` | (0, 3, 5) | +0.034 ± 0.003 | +0.010 ± 0.002 | -17.0 ± 2.0 | 7.41 ± 0.13 | (0, 3, 5) |
| `cw` | (1, 2, 5) | +0.035 ± 0.002 | +0.008 ± 0.002 | -18.4 ± 0.9 | 7.42 ± 0.07 | (1, 2, 5) |
| `cw` | (1, 3, 4) | +0.033 ± 0.004 | +0.008 ± 0.002 | -17.0 ± 1.8 | 7.43 ± 0.14 | (1, 3, 4) |

**Start faces (0, 2, 5), (1, 2, 4), (0, 3, 4): they occur after the set-down and after a roll into a wall or a block.**

| primitive | start face | dx (m) | dy (m) | dθ (°) | duration (s) | landing face |
|---|---|---|---|---|---|---|
| `roll_100_100` | (0, 2, 5) | -0.013 ± 0.002 | +0.204 ± 0.002 | -7.9 ± 2.4 | 5.79 ± 0.07 | (1, 2, 5) |
| `roll_100_100` | (1, 2, 4) | -0.012 ± 0.003 | +0.206 ± 0.001 | -8.8 ± 0.8 | 5.79 ± 0.08 | (1, 3, 4) |
| `roll_100_100` | (0, 3, 4) | -0.012 ± 0.003 | +0.205 ± 0.002 | -8.0 ± 2.3 | 5.84 ± 0.05 | (0, 3, 5) |
| `roll_120_120` | (0, 2, 5) | -0.036 ± 0.005 | +0.236 ± 0.001 | -5.6 ± 2.4 | 7.12 ± 0.14 | (1, 2, 5) |
| `roll_120_120` | (1, 2, 4) | -0.034 ± 0.005 | +0.236 ± 0.002 | -6.1 ± 1.6 | 7.12 ± 0.13 | (1, 3, 4) |
| `roll_120_120` | (0, 3, 4) | -0.033 ± 0.005 | +0.235 ± 0.003 | -5.8 ± 2.3 | 7.13 ± 0.15 | (0, 3, 5) |
| `roll_140_140` | (0, 2, 5) | -0.066 ± 0.006 | +0.229 ± 0.002 | -11.6 ± 1.6 | 7.49 ± 0.14 | (1, 2, 5) |
| `roll_140_140` | (1, 2, 4) | -0.066 ± 0.007 | +0.227 ± 0.002 | -10.6 ± 3.0 | 7.50 ± 0.13 | (1, 3, 4) |
| `roll_140_140` | (0, 3, 4) | -0.065 ± 0.006 | +0.230 ± 0.003 | -11.7 ± 1.9 | 7.48 ± 0.15 | (0, 3, 5) |
| `roll_100_120` | (0, 2, 5) | -0.035 ± 0.001 | +0.226 ± 0.002 | -0.3 ± 1.4 | 7.05 ± 0.05 | (1, 2, 5) |
| `roll_100_120` | (1, 2, 4) | -0.035 ± 0.003 | +0.225 ± 0.003 | -0.3 ± 0.8 | 7.05 ± 0.05 | (1, 3, 4) |
| `roll_100_120` | (0, 3, 4) | -0.034 ± 0.002 | +0.225 ± 0.003 | -0.1 ± 1.7 | 7.04 ± 0.08 | (0, 3, 5) |
| `roll_120_100` | (0, 2, 5) | -0.012 ± 0.002 | +0.215 ± 0.001 | -12.6 ± 0.9 | 6.66 ± 0.03 | (1, 2, 5) |
| `roll_120_100` | (1, 2, 4) | -0.012 ± 0.003 | +0.215 ± 0.001 | -11.5 ± 1.3 | 6.74 ± 0.05 | (1, 3, 4) |
| `roll_120_100` | (0, 3, 4) | -0.010 ± 0.003 | +0.217 ± 0.002 | -12.0 ± 0.5 | 6.74 ± 0.02 | (0, 3, 5) |
| `roll_100_140` | (0, 2, 5) | -0.061 ± 0.006 | +0.232 ± 0.001 | +6.5 ± 2.3 | 8.21 ± 0.19 | (1, 2, 5) |
| `roll_100_140` | (1, 2, 4) | -0.057 ± 0.003 | +0.233 ± 0.002 | +5.1 ± 1.6 | 8.10 ± 0.24 | (1, 3, 4) |
| `roll_100_140` | (0, 3, 4) | -0.059 ± 0.005 | +0.232 ± 0.002 | +5.8 ± 2.5 | 8.10 ± 0.25 | (0, 3, 5) |
| `roll_140_100` | (0, 2, 5) | -0.003 ± 0.003 | +0.219 ± 0.004 | -19.9 ± 1.5 | 7.03 ± 0.08 | (1, 2, 5) |
| `roll_140_100` | (1, 2, 4) | -0.003 ± 0.003 | +0.218 ± 0.003 | -20.3 ± 1.3 | 7.05 ± 0.09 | (1, 3, 4) |
| `roll_140_100` | (0, 3, 4) | -0.002 ± 0.004 | +0.222 ± 0.002 | -20.4 ± 1.3 | 7.02 ± 0.06 | (0, 3, 5) |
| `roll_120_140` | (0, 2, 5) | -0.060 ± 0.007 | +0.228 ± 0.003 | -0.4 ± 2.0 | 7.55 ± 0.09 | (1, 2, 5) |
| `roll_120_140` | (1, 2, 4) | -0.058 ± 0.004 | +0.228 ± 0.003 | -0.7 ± 1.2 | 7.55 ± 0.10 | (1, 3, 4) |
| `roll_120_140` | (0, 3, 4) | -0.056 ± 0.004 | +0.228 ± 0.004 | -1.0 ± 1.6 | 7.52 ± 0.11 | (0, 3, 5) |
| `roll_140_120` | (0, 2, 5) | -0.036 ± 0.006 | +0.234 ± 0.003 | -14.0 ± 2.2 | 7.38 ± 0.15 | (1, 2, 5) |
| `roll_140_120` | (1, 2, 4) | -0.035 ± 0.006 | +0.233 ± 0.002 | -13.6 ± 2.5 | 7.36 ± 0.17 | (1, 3, 4) |
| `roll_140_120` | (0, 3, 4) | -0.039 ± 0.008 | +0.234 ± 0.002 | -13.3 ± 2.9 | 7.33 ± 0.14 | (0, 3, 5) |
| `ccw` | (0, 2, 5) | -0.070 ± 0.005 | -0.090 ± 0.004 | +23.0 ± 0.9 | 7.37 ± 0.16 | (0, 3, 5) |
| `ccw` | (1, 2, 4) | -0.071 ± 0.005 | -0.089 ± 0.004 | +22.1 ± 0.9 | 7.41 ± 0.15 | (1, 2, 5) |
| `ccw` | (0, 3, 4) | -0.069 ± 0.007 | -0.091 ± 0.005 | +23.7 ± 1.5 | 7.48 ± 0.15 | (1, 3, 4) |
| `cw` | (0, 2, 5) | -0.035 ± 0.004 | -0.013 ± 0.001 | -16.0 ± 1.5 | 7.33 ± 0.10 | (0, 3, 5) |
| `cw` | (1, 2, 4) | -0.036 ± 0.004 | -0.013 ± 0.002 | -16.7 ± 0.5 | 7.38 ± 0.07 | (1, 2, 5) |
| `cw` | (0, 3, 4) | -0.036 ± 0.007 | -0.015 ± 0.002 | -16.3 ± 1.4 | 7.43 ± 0.12 | (1, 3, 4) |

Each row: mean ± standard deviation over 12 executions (half of them started right after a set-down on that face, half after one to three other primitives).
<!-- END GENERATED primitive_table -->

The duration is the whole command, from its acceptance to the end of the braked hold. The standard deviations are
those of repeated executions on one floor; they say nothing about a floor of another friction, about loose ground or
about a robot that touches something.

## 8. What the table does not cover

<!-- BEGIN GENERATED reliability -->
- **Tipping.** In the table's 648 executions of the nine forward rolls on the flat floor, 0 did not tip the robot onto the landing face. `roll_100_100` and `roll_100_120` are close to the tipping threshold: they are not guaranteed to tip.
- **End triangles.** Counted after every primitive: on a free flat floor 0 of 4640 ended on an end triangle; after rolls into a wall 0 of 175; while moving among obstacles 16 of 7600; after sequences of `set_targets` shapes 0 of 127.
- **Creep.** A robot left standing after a primitive moves on by 1.0 mm on average and by at most 2.0 mm over the next 30 s (22 measurements).
<!-- END GENERATED reliability -->

- **Loose ground.** On dust and on gravel the table does not hold. Rolls come out with another length and drift
  sideways, the turns turn less, and at the edge of a patch, with some end caps on it and some off, a turn can come to
  almost nothing. No table is given for loose ground. The calibration site has a strip of each kind.
- **Walls, blocks and heaps.** A robot that has rolled against a wall, a block or a heap does not get further by
  rolling on the same way: it has to be turned first. A roll into an obstacle can leave the robot on any of the six
  rollable faces and, rarely, on an end triangle. A roll against a heap can also leave the robot leaning on it, with
  one end cap up on its flank; the lower end of each strut is then still what its accelerometer shows, and the robot
  gets off the heap as it gets away from a wall: by turning away and rolling on. A roll that touched a heap can turn
  the robot by tens of degrees: the table does not hold for it.

<!-- BEGIN GENERATED heaps -->
- **Heaps, measured.** In 540 runs against heaps of the two kinds (240 against mounds of fill, 300 against rubble heaps) the robot never got onto or across one: its centre came at most 0.11 m inside a mound's outline and never inside a rubble heap's. 444 of 2361 rolls that touched a heap left an end cap resting on it; 3 of 2361 ended on an end triangle. After a roll that touched a heap the robot's heading differed from the table's by 10° in the median, by more than 20° in one roll of ten, and by 139° at most. After each run the robot was turned away from the heap and rolled three times: in 535 of the 540 runs it then stood on a rollable face on the floor, clear of the heap.
<!-- END GENERATED heaps -->

- **No reply says that a primitive failed.** A primitive that did not tip the robot, or that pushed it against
  something, ends like any other: the states run, the robot returns to the rest shape, the hold passes. What your
  program can look at is the camera, while the robot is in view, and the telemetry anywhere. A roll that tipped the
  robot has changed the face it stands on, so the lower end of at least one strut has changed, and each strut's
  accelerometer shows which of its ends is lower (section 9). Whether a robot that is held by an obstacle produces
  timeouts or other marks in the cable controller's flags has not been measured: try it before you rely on it.

## 9. Telemetry

The robot sends one record per control period; `telemetry()` returns them (the fields, their shapes and the buffer are
in `station_api.md`, section 4.2).

<!-- BEGIN GENERATED telemetry -->
| reading | rated noise of this robot |
|---|---|
| stretch-sensor length | white noise of 0.5 mm rms, then rounded to 0.1 mm; no drift |
| accelerometer | white noise of 2 mg rms per axis |
| gyroscope | white noise of 0.1 °/s rms per axis, plus one constant bias per axis that is drawn anew for every episode, between -0.5 and +0.5 °/s |
| encoders, targets, done flags, duties | none |
<!-- END GENERATED telemetry -->

- **Stretch sensors.** `sensor_length_mm` is each sensor's length, already converted from the raw capacitance by the
  robot's own calibration; `sensor_capacitance_pF` is the raw reading and carries nothing more.
- **Motors.** The encoder counts are cumulative and do not start at zero with the episode. `motor_target_mm`,
  `motor_done` and `motor_duty` are the cable controller's own values for the running target state.
- **IMU.** Each strut has one IMU, on its controller board near the strut's middle. Its frame: **x along the strut,
  pointing from the odd node to the even node**; z along the normal of the controller board; y = z × x. Only the x axis
  is tied to the node numbering. Which way the board's normal points around the strut has not been verified on this
  robot, so do not build on the signs of y and z without checking them. The accelerometer reads specific force: a
  strut at rest reads 1 g along the upward vertical. So at rest, `imu_accel_g[k][0] > 0` means that the even node 2k
  is the higher end of strut k and the odd node 2k+1 the lower one, and a negative value the reverse. The gyroscope
  reads the strut's angular rate in the same frame. A record holds the rates at one instant of its control period.
  While a command runs, the struts shake, and the rates change quickly from one record to the next.
- There is no magnetometer and no power reading.

## 10. Set-down, link, limits

- **Set-down.** The team lowers the robot on a rope, in its rest shape, and releases it just above the pad. It comes
  to rest on one of the six rollable faces, never on an end triangle, close to the pad's centre, at any heading
  (`site_briefing.md`).
- **Link.** The radio link to the robot holds everywhere on a site, also under a slab: commands arrive, and the
  telemetry keeps coming.
- **Cable limits.** A `set_targets` length outside the cable-length limits is refused and nothing moves. The
  primitives stay inside the limits by themselves.
- **Damage.** Nothing on a site damages the robot, and pushing against rubble does not harm it.
