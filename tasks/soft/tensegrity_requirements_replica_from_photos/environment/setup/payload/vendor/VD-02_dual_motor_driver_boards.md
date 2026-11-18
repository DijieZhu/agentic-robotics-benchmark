# VD-02 Dual motor driver boards

This page is an excerpt of data that the vendors themselves publish (product pages, datasheets, drawings): breakout
boards that drive two brushed DC motors. The values are the vendors' own, in the vendors' units; nothing is converted.
"not published" marks a value that the vendor pages used for this excerpt do not state. The page sets comparable parts
side by side; it does not say which of them, if any, is on the robot.

## Adafruit 2448

| field | value |
|---|---|
| name | Adafruit TB6612 1.2A DC/Stepper Motor Driver Breakout Board |
| main chip | TB6612 |
| package(s) of that chip, as its maker lists them | SSOP24 |
| outline | 27 mm x 19 mm |
| mounting holes | 2 |
| header pins | 16 |
| header pitch | 0.1" |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| top | left to right | 10 | PWMA, AIN2, AIN1, STBY, BIN1, BIN2, PWMB, GND, Vcc, VM |
| bottom | left to right | 6 | MOTORA, MOTORA, GND, GND, MOTORB, MOTORB |
| terminal block (other connection) | top to bottom | 2 | Vmotor +, Vmotor - |

## Pololu 2130

| field | value |
|---|---|
| name | DRV8833 Dual Motor Driver Carrier |
| main chip | DRV8833 |
| package(s) of that chip, as its maker lists them | TSSOP (16), HTSSOP (16) |
| outline | 0.8 in x 0.5 in |
| mounting holes | 0 |
| header pins | 16 |
| header pitch | 2.54 mm |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| left | top to bottom | 8 | GND, VMM, BIN1, BIN2, AIN2, AIN1, nSLEEP, nFAULT |
| right | top to bottom | 8 | GND, VIN, BOUT1, BOUT2, AOUT2, AOUT1, AISEN, BISEN |

## Pololu 713

| field | value |
|---|---|
| name | TB6612FNG Dual Motor Driver Carrier |
| main chip | TB6612FNG |
| package(s) of that chip, as its maker lists them | SSOP24 |
| outline | 0.8 in x 0.6 in |
| mounting holes | 0 |
| header pins | 16 |
| header pitch | 2.54 mm |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| left | top to bottom | 8 | GND, VCC, AO1, AO2, BO2, BO1, VMOT, GND |
| right | top to bottom | 8 | PWMA, AIN2, AIN1, STBY, BIN1, BIN2, PWMB, GND |

## SparkFun ROB-14450

| field | value |
|---|---|
| name | SparkFun Motor Driver - Dual TB6612FNG (with Headers) |
| main chip | TB6612FNG |
| package(s) of that chip, as its maker lists them | SSOP24 |
| outline | not published |
| mounting holes | 0 |
| header pins | 16 |
| header pitch | 0.1" |
| revision note | not published |

Pins, named and ordered as on the vendor's own labelled picture or drawing of the board (not reproduced here):

| group | order | number of pins | pin names |
|---|---|---|---|
| left | top to bottom | 8 | PWMA, AIN2, AIN1, STBY, BIN1, BIN2, PWMB, GND |
| right | top to bottom | 8 | VM, VCC, GND, A01, A02, B02, B01, GND |
