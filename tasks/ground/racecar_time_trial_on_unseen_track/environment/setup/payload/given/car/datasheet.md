# The car: datasheet

A tenth-scale, four-wheel-drive touring-class chassis with an autonomy deck: a 2D scanning sensor at the front, a
motor controller board with an inertial unit, a compute unit, a steering servo, a brushed motor and a battery pack.
It has no camera. This sheet gives what the build records and the component sheets say about the car. How the car
behaves on a floor is not on any component sheet: for that there are the lab runs in `../lab_logs/`. The outer
geometry of the car is in `meshes/`.

The same constants are returned by `describe()` at every venue (`../interface/car_api.md`). The car is the same
car at every venue: nothing on it is changed, added or retuned between venues.

## 1. Frames

| frame | origin | axes |
|---|---|---|
| car frame | on the floor under the midpoint of the rear axle, the car at rest | x forward, y to the left, z up |
| scan frame | the scan origin of the scanning sensor, at (0.21315, 0.0, 0.1099) m in the car frame | parallel to the car frame |
| inertial frame | the inertial unit, at (0.0603, -0.0051, 0.0780) m in the car frame | parallel to the car frame |

Angles are in radians, counter-clockwise positive seen from above; a steering angle to the left is positive.

## 2. Geometry and mass

| quantity | value |
|---|---|
| wheelbase | 0.25784 m |
| track, front / rear | 0.15591 m / 0.15595 m |
| tyres | outer diameter 0.0658 m (rolling radius 0.03290 m), width 0.0272 m |
| length x width x height | 0.38776 m x 0.18492 m x 0.15252 m |
| foremost point / rearmost point | x = +0.33446 m / x = -0.05330 m in the car frame |
| ground clearance (under the chassis plate; the suspension arms reach lower) | 0.01469 m |
| mass, ready to run | 1.8359 kg |
| centre of mass | (0.12495, -0.00263, 0.05014) m in the car frame |
| share of the weight on the front axle, at rest | 48.5 % |
| suspension | independent at the four corners, sprung and damped; a few millimetres of travel |
| reference point (rules: timing, ground speed, overtaking) | the midpoint of the wheelbase on the floor, (0.12892, 0, 0) m in the car frame |
| wheels on the floor (rules: tapes, leaving the track) | front (0.25784, +-0.07796) m, rear (0, +-0.07798) m in the car frame, the car at rest |

## 3. Steering

| quantity | value |
|---|---|
| steering angle | the angle of the equivalent single front wheel (bicycle model), -0.4189 rad to +0.4189 rad (24.0 degrees each way) |
| smallest turning radius | 0.5791 m at the midpoint of the rear axle: wheelbase / tan(0.4189) |
| servo | follows the commanded angle at no more than 3.2 rad/s |

The steering command of a `step` is the target of this angle; a command beyond the range is clipped to it.
`steer_echo` is the servo's present target, that is, the command after the latency of section 6 and the rate limit
above. It is not a measurement of the wheels: the wheels follow the servo's target with a short lag of their own,
which the steering steps of the lab runs show.

## 4. Drive

| quantity | value |
|---|---|
| layout | shaft-driven four-wheel drive: one motor, a propeller shaft joining the front and the rear axle, an open differential in each axle |
| final drive | 8.3474 motor revolutions per wheel revolution |
| motor | 540-size brushed DC motor; rated 7.2 V; speed constant 2250 rpm/V; torque constant 0.00424 N m/A; winding resistance 0.105 ohm; no-load current 1.6 A; largest speed 26 000 rpm; can 35.7 mm x 54.5 mm; 160 g |
| battery | 6-cell nickel-metal-hydride pack, 7.2 V nominal, 5.3 Ah; internal resistance about 0.025 ohm |
| motor controller | closed speed loop on the motor speed; current limit 40 A, driving and braking |
| speed command | the target of the forward road speed, -2.0 m/s to +8.0 m/s; negative is reverse; a command beyond the range is clipped to it |
| top speed | about 6.3 m/s on the lab floor (`launch_top_speed_brake` of the lab runs) |

The speed command of a `step` is the target of the controller's speed loop, as road speed: motor speed x 0.03290 m
/ 8.3474. A target far above the present speed makes the controller drive with all the current it may; a target
of 0 at speed makes it brake with all the current it may, which locks the wheels on most floors. How fast the loop
follows smaller changes is shown by `speed_steps_reverse` of the lab runs.

## 5. Sensors

### 5.1 Scanning sensor (2D LiDAR)

| quantity | value |
|---|---|
| field of view | 270 degrees in one plane: from -135 degrees (right rear) over straight ahead to +135 degrees (left rear) |
| beams | 1081, one every 0.25 degrees; beam 0 at -135 degrees, beam 540 straight ahead, beam 1080 at +135 degrees |
| blind sector | the 90 degrees behind the car are never measured |
| revolutions | 40 per second; one revolution is one tick of 25 ms |
| range | 0.06 m to 10 m on bright targets |
| output | one range per beam in whole millimetres; 0 means no return |
| scan plane | parallel to the car's floor plane, 0.1099 m above the floor with the car at rest |

- **The scan is not a snapshot.** A revolution is fired as five sub-scans, beams 0 to 215, 216 to 431, 432 to 647,
  648 to 863 and 864 to 1080, at 2.5, 7.5, 12.5, 17.5 and 22.5 ms into the tick, each from where the sensor is at
  that moment. At 5 m/s the car moves 12.5 cm during a revolution and 10 cm between the first and the last
  sub-scan.
- **The scan plane moves with the car.** When the car pitches under braking or acceleration, rolls in a bend or
  goes over an edge, the plane tilts with it. Beams then meet the floor, or pass a low object at another height.
- **Floor.** A beam that meets a hard, shiny floor at a shallow angle gives no return as a rule. On carpet (a
  carpeted hall floor or a carpet tile) such a beam sometimes returns the range to the floor.
- **Dark targets.** Plain black ducts and other dark, matt things return only up to about 4 m; beyond that the
  beam gives no return.
- **Out of range.** A target nearer than the smallest range or farther than the largest gives no return.
- **Other cars.** The body of a car of this class lies almost entirely below the scan plane; only its sensor
  head, a cylinder of 35 mm diameter, reaches into it, and that is dark. This is why the slow cars carry the
  league's marker box (rules, clause S-22).
- **Round walls.** The scan plane cuts a duct of 0.33 m diameter below its widest point: the range to a wall is
  the range to the duct at 0.11 m above the floor.
- **Noise.** Every range carries measurement noise, anew in every scan. Its size is not on the component sheet;
  a car standing still in the lab runs shows it.

### 5.2 Inertial unit

| quantity | value |
|---|---|
| output | specific force (m/s^2) along x, y, z and angular rate (rad/s) about x, y, z of the inertial frame |
| rate | 200 samples per second, five per tick, oldest first; each sample is the mean over the 5 ms before it |
| at rest on a level floor | about (0, 0, +9.81) m/s^2 and (0, 0, 0) rad/s, plus the bias |
| measuring range | +-16 g, +-2000 degrees/s; a reading beyond it is clipped |
| resolution on the wire | 0.001 m/s^2, 0.0001 rad/s |

The readings carry a bias and noise. The bias is different after every power-up and drifts slowly; a car that
stands still shows it. At a venue the car stays powered from the first session to the last: its biases do not
change between the sessions of a venue (beyond the slow drift), and they differ from one venue to the next. The
noise grows with the speed of the drive train. The sizes of bias, drift and noise are not on the component sheet;
the lab runs show them.

### 5.3 Speed from the motor

`wheel_speed` is the motor speed turned into road speed: motor speed x 0.03290 m / 8.3474, signed, in m/s with a
resolution of 0.001 m/s. There is one number for the whole car: with the shaft drive it follows the mean speed of
the four wheels, and no wheel has a sensor of its own. It is the speed of the wheels, not of the car over the
floor. It reads high while the wheels spin (hard acceleration, low grip) and low while they lock or slide (hard
braking: it can read 0 while the car is still moving). It carries a little noise. The car has no sensor of its
speed over the floor.

## 6. Timing

| quantity | value |
|---|---|
| tick | 25 ms: one `step`, one revolution of the scanning sensor, five inertial samples |
| command latency | 1 tick: the command sent with one `step` acts during the next tick |

## 7. What this sheet does not give

There is no dynamics model of this car among the files you are given: no model file, no tyre curve, no controller
gains. The grip of the tyres on a floor, the behaviour of the tyres near and beyond their limit, the response of
the speed loop and of the steering, the pitch and roll of the car, and the size of the sensor noise are properties
of the car on a floor, not of its parts. The lab runs were recorded for that purpose.
