# VD-05 Current and power monitor boards

This page is an excerpt of data that the vendors themselves publish (product pages, datasheets, drawings): breakout
boards that measure a current or a power. The values are the vendors' own, in the vendors' units; nothing is
converted. "not published" marks a value that the vendor pages used for this excerpt do not state. The page sets
comparable parts side by side; it does not say which of them, if any, is on the robot.

## Adafruit 4226

| field | value |
|---|---|
| name | Adafruit INA260 High or Low Side Voltage, Current, Power Sensor |
| main chip | INA260 |
| package(s) of that chip, as its maker lists them | TSSOP (16) |
| outline | 22.9 mm x 22.8 mm |
| mounting holes | 2 |
| header pins | 8 |
| header pitch | 0.1" |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| bottom | left to right | 8 | Vcc, GND, SCL, SDA, Alert, VBus, Vin+, Vin- |
| terminal block (other connection) | left to right | 2 | Vin+, Vin- |

## Adafruit 904

| field | value |
|---|---|
| name | INA219 High Side DC Current Sensor Breakout - 26V +/-3.2A Max |
| main chip | INA219 |
| package(s) of that chip, as its maker lists them | SOT-23 (8), SOIC (8) |
| outline | 25.6 mm x 20.4 mm |
| mounting holes | 4 |
| header pins | 6 |
| header pitch | 0.1" |
| revision note | we've updated this sensor to our STEMMA QT standard for sensors. There are now four mounting holes |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| bottom | left to right | 6 | VIN, GND, SCL, SDA, VIN-, VIN+ |
| terminal block (other connection) | left to right | 2 | Vin-, Vin+ |
| STEMMA QT connectors (other connection) | - | 2 | STEMMA QT, STEMMA QT |

## Pololu 4041

| field | value |
|---|---|
| name | ACS724LLCTR-05AB Current Sensor Carrier -5A to +5A |
| main chip | ACS724LLCTR-05AB |
| package(s) of that chip, as its maker lists them | 8-Pin SOIC |
| outline | 0.8 in x 0.7 in |
| mounting holes | 2 |
| header pins | 3 |
| header pitch | 2.54 mm |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| header | pin 1 to 3 | 3 | GND, VCC, VOUT |
| current path (other connection) | pin 1 to 2 | 2 | IP+, IP- |

## SparkFun SEN-13679

| field | value |
|---|---|
| name | SparkFun Current Sensor Breakout - ACS723 |
| main chip | ACS723 |
| package(s) of that chip, as its maker lists them | 8-pin SOIC |
| outline | not published |
| mounting holes | 1 |
| header pins | 3 |
| header pitch | 0.1" |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| right | top to bottom | 3 | VCC, VOUT, GND |
| current path (other connection) | top to bottom | 2 | IP+, IP- |
