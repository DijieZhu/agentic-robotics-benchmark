# Servo documents

| file | content |
|---|---|
| [servo_datasheet.md](servo_datasheet.md) | BUS-SERVO-12V-A datasheet: ratings, position mode (angle limits, command latency, motion profile, position loop), gear train, feedback registers, output elasticity and hysteresis, the values as installed in the DA-6 desktop arm, production ranges |
| [bus_interface.md](bus_interface.md) | DA-6 bus interface manual: the 10 ms bus cycle, the state and command fields, which replies are written, register units, clamping, quantisation, latency, late replies, worked examples |

The servo ids and the conversion between ticks and design angles of a particular unit:
[../unit/calibration_record.md](../unit/calibration_record.md). The Python interface of the motion controller:
[../controller/controller_interface.md](../controller/controller_interface.md).
