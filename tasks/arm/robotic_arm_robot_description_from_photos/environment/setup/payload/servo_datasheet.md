# BUS-SERVO-12V-A (the six servos of the arm)

Datasheet page `servo_datasheet.png`, dimensioned drawing `servo_drawing.png`, vendor mesh `parts/servo.stl` (vendor frame,
millimetres). Key facts:

| item | value |
|---|---|
| mass | 60 g |
| stall torque | 2.94 N m at 12 V |
| no-load speed | 6.49 rad/s at 12 V |
| supply | 9.0 - 12.6 V |
| encoder | 12-bit magnetic, 4096 ticks / rev; tick 2048 = mid position |
| bus angle | b = (ticks - 2048) x 2 pi / 4096 rad |
| drive-mode bit (EEPROM) | 1 inverts the direction of the reading |
| firmware angle limits (EEPROM) | min / max angle-limit registers in ticks; commands outside are refused |
| case | 45.4 x 24.8 x 28.8 mm between two 1.5 mm flanges, 34.6 mm overall with the horn |
| horn | disc 20 mm, 14.0 mm bolt circle; horn axis 12.5 mm off the case centre |
| output backlash | <= 0.5 deg |
