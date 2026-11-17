# BUS-SERVO-12V-A smart serial-bus servo

Datasheet DS-BUS-SERVO-12V-A, revision B.

Half-duplex serial-bus servo of the 12 V class with a 12-bit contactless magnetic encoder on the output shaft,
position, speed and load feedback, a trapezoidal motion profile, programmable angle limits (EEPROM) and a metal gear
train. Revision B adds the position-loop, gear-train, latency and feedback-register data, the output elasticity and
hysteresis, and the values of the servo as installed in the DA-6 desktop arm (section 6).

Conventions: one output turn = 4096 ticks; 1 tick = 2π/4096 rad = 1.534 mrad = 0.0879°. "+ticks" is the direction in
which the position reading increases. A single value in the tables applies to every servo. Where a range is given, the
value of an individual servo lies somewhere inside it; individual values are neither measured nor marked on the servo.

## 1. Ratings

| item | value |
|---|---|
| supply voltage | 9.0-12.6 V; ratings at 12.0 V |
| stall torque | 2.94 N m (30.0 kgf cm), momentary peak at the output shaft |
| no-load speed | 6.49 rad/s typical (4231 ticks/s, 0.161 s per 60°); production range in section 7 |
| encoder | magnetic, 12 bit, on the output shaft: 4096 ticks per turn |
| gear train | metal |
| mass | 60 g |
| case | 45.4 x 24.8 x 28.8 mm between the flanges, 34.6 mm high overall; output horn dia. 20.0 mm with 4 x dia. 2.5 on PCD 14.0 (square pattern), its axis 10.2 mm from the front end of the case |
| bus | half-duplex asynchronous serial (TTL level), two 3-pin connectors (daisy chain) |

## 2. Position mode

Signal path of a command:

```
goal position (ticks) -> angle-limit clamp (EEPROM) -> command latency -> profile generator
  -> position loop on the output encoder -> motor and gear train -> output shaft (encoder)
  -> elastic coupling to the driven link (section 5)
```

### 2.1 Goal position and angle limits

Positions are single-turn values: 0 to 4095 ticks over one output turn, for the goal, the angle limits and the
position reading alike. The goal position register takes whole ticks. The servo clamps every goal to its EEPROM angle
limits [min angle limit, max angle limit] before it executes it: a goal outside the limits runs to the nearer limit,
without an error flag. The limits are set when the servo is installed (the DA-6: the unit's calibration record); an
installation keeps the travel between them clear of the point where the count passes from 4095 to 0.

### 2.2 Command latency

A command (goal, speed and acceleration registers) takes effect 15-40 ms after it is sent: the delay of the bus
adapter plus that of the servo firmware. The latency is the same for all servos of one bus and constant for a given
installation; the servo does not report it.

### 2.3 Motion profile

Trapezoidal velocity profile. On every new goal the profile generator continues from its present setpoint and setpoint
speed: it accelerates (or brakes) at the acceleration setting up to the speed setting and decelerates so as to stop at
the goal (a triangular profile when the move is short). A new goal may arrive at any time, also during a move.

| setting | default | register |
|---|---|---|
| speed | the servo's maximum speed, i.e. its no-load speed (6.49 rad/s typical, 4231 ticks/s; range in section 7) | goal speed, ticks/s, an unsigned whole number; 0 = maximum. It can only lower the speed: a value above the maximum has no effect |
| acceleration | 40 rad/s² (26076 ticks/s²) | acceleration, an unsigned whole number of units of 100 ticks/s² (1 unit = 0.1534 rad/s²); 0 = default. It can only lower the acceleration: 261 and above select the default |

The speed and acceleration registers are written together with the goal and apply to that goal: a goal written with 0
in a register runs at the maximum speed or the default acceleration.

### 2.4 Position loop

Proportional-derivative loop on the output encoder with the profile setpoint as its reference, its torque limited to
the stall torque:

```
torque = kp x (setpoint - output angle) - kv x (output speed)
```

| parameter | value |
|---|---|
| kp, position gain | 998.22 N m/rad typical; ±15 % (848.5-1148.0 N m/rad) |
| kv, damping | 2.731 N m s/rad |
| torque limit | ±2.94 N m (the stall torque); position errors above 2.95 mrad (0.169°, 1.9 ticks) saturate the loop at full torque |

The loop has no integral action: a steady torque τ at the output leaves the output shaft τ / kp away from the setpoint
(1.00 mrad = 0.057° per N m at the typical kp).

### 2.5 Gear train (referred to the output shaft)

| parameter | value |
|---|---|
| reflected inertia of motor and gears | 0.028 kg m² |
| viscous friction | 0.6 N m s/rad |
| Coulomb friction | 0.052 N m typical; 0.7-1.4 x typical (0.036-0.073 N m) |

## 3. Feedback registers

| register | unit | content |
|---|---|---|
| present position | ticks (whole numbers) | angle of the output shaft (encoder); resolution 1 tick |
| present speed | ticks/s (whole numbers), signed along +ticks | speed of the output shaft |
| present load | per mille of the stall torque (1 = 2.94 mN m), signed along +ticks, -1000 to +1000 | the output torque x the servo's reading scale (0.9-1.1) plus a noise of about 1 % of the stall torque (RMS); sampled every 20 ms: the register holds the whole sample (value and noise) until the next one, so reads within one 20 ms period return the same value |
| moving | flag | set while the servo has a goal it has not reached yet (also while that goal is still within the command latency) or its output shaft turns faster than 0.01 rad/s (6.5 ticks/s) |

The encoder sits on the output shaft. The deflection of the elastic coupling behind it (section 5) does not show in
the position register; the torque that causes it shows in the load register.

## 4. EEPROM registers set at installation

| register | unit | content |
|---|---|---|
| min angle limit | ticks | the lowest goal the servo executes |
| max angle limit | ticks | the highest goal the servo executes |

## 5. Output elasticity and hysteresis

The position loop holds the output shaft. Between the shaft and the driven link lie the output stage of the gear train
beyond the encoder magnet (output shaft and spline), the horn with its screw joint and the driven printed link.
Together they form an elastic coupling with a small play:

```
link angle = output shaft angle + d
d = tau / k at rest, inside a play band of half-width h:  d <- min(max(d, tau/k - h), tau/k + h)
```

tau is the torque the coupling carries (signed along the joint), k the coupling stiffness and h the half-width of the
play band. When the load changes, the deflection d moves only once tau/k has moved more than h away from it, so the
coupling shows a hysteresis loop 2h wide (0.10-0.50° peak to peak). The coupling is well damped. k and h depend on the
joint (section 6); the position loop does not see d.

Example: a static torque of 0.5 N m deflects a coupling of k = 60 N m/rad by 8.33 mrad (0.477°) and one of k = 180 N
m/rad by 2.78 mrad (0.159°); at a lever of 0.30 m that is 2.50 mm and 0.83 mm. The play band adds up to ±h.

## 6. As installed in the DA-6 desktop arm

| servo id | joint | coupling stiffness k (N m/rad) | play half-width h (°) |
|---|---|---|---|
| 1 | base_yaw | 200-600 | 0.05-0.25 |
| 2 | shoulder_pitch | 60-180 | 0.05-0.25 |
| 3 | elbow_pitch | 60-180 | 0.05-0.25 |
| 4 | wrist_pitch | 120-400 | 0.05-0.25 |
| 5 | wrist_roll | 200-600 | 0.05-0.25 |
| 6 | jaw | negligible (rigid) | negligible |

Over the arm the coupling stiffness spans 60-600 N m/rad. The six servos share one bus and therefore one command
latency (15-40 ms). Each servo is installed so that its joint's whole travel lies well inside one turn of the encoder:
every position of a DA-6 servo (reading, goal, angle limit) lies in 0 to 4095 ticks.

## 7. Production ranges

| quantity | typical | range |
|---|---|---|
| maximum speed (no-load speed) | 6.49 rad/s | 0.85-1.0 x typical: 5.52-6.49 rad/s (3596-4231 ticks/s), per servo |
| position gain kp | 998.22 N m/rad | ±15 %, per servo |
| Coulomb friction | 0.052 N m | 0.7-1.4 x typical, per servo |
| load reading scale | 1.0 | 0.9-1.1, per servo |
| load reading noise | 1 % of the stall torque (RMS) | - |
| command latency | - | 15-40 ms, per installation |
| coupling stiffness k, play half-width h | - | per joint as installed (section 6) |

Single values (no production range): the stall torque, kv, the gear train's inertia and viscous friction, the encoder
resolution and the default acceleration.
