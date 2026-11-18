# VD-03 Capacitive touch sensing boards

This page is an excerpt of data that the vendors themselves publish (product pages, datasheets, drawings): breakout
boards that measure the capacitance of several electrodes. The values are the vendors' own, in the vendors' units;
nothing is converted. "not published" marks a value that the vendor pages used for this excerpt do not state. The page
sets comparable parts side by side; it does not say which of them, if any, is on the robot.

## Adafruit 1602

| field | value |
|---|---|
| name | CAP1188 - 8-Key Capacitive Touch Sensor Breakout - I2C or SPI |
| main chip | CAP1188 |
| package(s) of that chip, as its maker lists them | 24-Pin QFN |
| outline | 42 mm x 18 mm |
| mounting holes | 4 |
| header pins | 26 |
| header pitch | 0.1" |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| left | top to bottom | 13 | SDA/MISO, SCK, VIN, GND, L1, L2, L3, L4, L5, L6, L7, L8, IRQ |
| right | top to bottom | 13 | MOSI, CS, RST, 3Vo, C1, C2, C3, C4, C5, C6, C7, C8, AD |

## Adafruit 1982

| field | value |
|---|---|
| name | Adafruit 12-Key Capacitive Touch Sensor Breakout - MPR121 - STEMMA QT |
| main chip | MPR121 |
| package(s) of that chip, as its maker lists them | 20-pin QFN, 3 mm x 3 mm x 0.65 mm |
| outline | 1.3 in x 0.75 in |
| mounting holes | 2 |
| header pins | 20 |
| header pitch | 0.1" |
| revision note | The physical size of the board and two mounting holes location have not changed. However, we did change the power/I2C breakout section pinout to match our standard pinout order! |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| top | left to right | 13 | 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, GND |
| bottom | left to right | 7 | IRQ, SCL, SDA, ADDR, GND, 3Vo, VIN |
| bottom (STEMMA QT revision) (other connection) | left to right | 7 | VIN, 3Vo, GND, SCL, SDA, INT, ADDR |
| STEMMA QT connectors (other connection) | - | 2 | STEMMA QT, STEMMA QT |

## Seeed Studio 101020872

| field | value |
|---|---|
| name | Grove - 12 Key Capacitive I2C Touch Sensor V3 (MPR121) |
| main chip | MPR121 |
| package(s) of that chip, as its maker lists them | 20-pin QFN, 3 mm x 3 mm x 0.65 mm |
| outline | 60 mm x 20 mm |
| mounting holes | 4 |
| header pins | 10 |
| header pitch | not published |
| revision note | it adds two more GND ports to the head in order to be more stable |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| Grove connector | as labelled | 4 | SCL, SDA, VCC, GND |
| small header | as labelled | 6 | GND, CH4, CH5, CH6, CH7, INT |
| alligator pads (other connection) | as labelled | 10 | CH0, CH1, CH2, CH3, CH8, CH9, CH10, CH11, GND, GND |

## SparkFun SEN-09695

| field | value |
|---|---|
| name | SparkFun Capacitive Touch Sensor Breakout - MPR121 |
| main chip | MPR121QR2 |
| package(s) of that chip, as its maker lists them | 20-pin QFN, 3 mm x 3 mm x 0.65 mm |
| outline | not published |
| mounting holes | 0 |
| header pins | 18 |
| header pitch | 0.1" |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| left | top to bottom | 6 | 3.3V, IRQ, SCL, SDA, ADD, GND |
| right | top to bottom | 12 | 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 |
