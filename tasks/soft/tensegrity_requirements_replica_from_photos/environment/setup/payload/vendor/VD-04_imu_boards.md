# VD-04 Inertial measurement boards

This page is an excerpt of data that the vendors themselves publish (product pages, datasheets, drawings): breakout
boards that carry an inertial measurement unit. The values are the vendors' own, in the vendors' units; nothing is
converted. "not published" marks a value that the vendor pages used for this excerpt do not state. The page sets
comparable parts side by side; it does not say which of them, if any, is on the robot.

## Adafruit 4438

| field | value |
|---|---|
| name | Adafruit LSM6DSOX 6 DoF Accelerometer and Gyroscope |
| main chip | LSM6DSOX |
| package(s) of that chip, as its maker lists them | not published |
| outline | 25.6 mm x 17.8 mm |
| mounting holes | 2 |
| header pins | 14 |
| header pitch | 0.1" |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| top | left to right | 5 | SCX, SDX, CS, DO, GND |
| bottom | left to right | 9 | VIN, 3Vo, GND, SCL, SDA, DO, CS, I1, I2 |
| STEMMA QT connectors (other connection) | - | 2 | STEMMA QT, STEMMA QT |

## Adafruit 4554

| field | value |
|---|---|
| name | Adafruit TDK InvenSense ICM-20948 9-DoF IMU (MPU-9250 Upgrade) |
| main chip | ICM-20948 |
| package(s) of that chip, as its maker lists them | 24-pin QFN, 3 mm x 3 mm x 1 mm |
| outline | 25.7 mm x 17.7 mm |
| mounting holes | 4 |
| header pins | 12 |
| header pitch | 0.1" |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| top | left to right | 6 | FS, AD, AC, G, SDO, CS |
| bottom | left to right | 6 | VIN, 1V8, GND, SCL, SDA, INT |
| STEMMA QT connectors (other connection) | - | 2 | STEMMA QT, STEMMA QT |

## Pimoroni PIM448

| field | value |
|---|---|
| name | ICM20948 9DoF Motion Sensor Breakout |
| main chip | ICM-20948 |
| package(s) of that chip, as its maker lists them | 24-pin QFN, 3 mm x 3 mm x 1 mm |
| outline | 19.0 mm x 19.0 mm |
| mounting holes | 2 |
| header pins | 5 |
| header pitch | 1x5 pin header |
| revision note | Qwiic/STEMMA QT connector only on breakouts manufactured after October 2022 |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| edge | as labelled | 5 | 2-5V, SDA, SCL, INT, GND |
| Qw/ST connector (other connection) | - | 1 | Qw/ST |

## SparkFun SEN-15335

| field | value |
|---|---|
| name | SparkFun 9DoF IMU Breakout - ICM-20948 (Qwiic) |
| main chip | ICM-20948 |
| package(s) of that chip, as its maker lists them | 24-pin QFN, 3 mm x 3 mm x 1 mm |
| outline | not published |
| mounting holes | 4 |
| header pins | 12 |
| header pitch | 0.1-inch-spaced |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| left (I2C side) | top to bottom | 6 | GND, VIN, SDA, SCL, AD0, NC |
| right | top to bottom | 6 | GND, INT, FSYNC, ADA, ACL, VIN |
| Qwiic connectors (other connection) | - | 2 | Qwiic, Qwiic |
