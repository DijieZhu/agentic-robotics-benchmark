# Bought parts inside the modules

The modules of `kit.json` were put together by hand from parts made in the lab (the printed enclosures, winches and
end caps, the cast silicone covers, the custom main board) and from the bought parts below, listed as the build's bill
of materials orders them: maker or vendor and part number. No drawings and no CAD files of these parts are given; their
shapes are in the modules' model files (`~/given/parts/`). Quantities are per module.

## Strut core (`M0`)

| item | maker / vendor | part number | per strut core |
|---|---|---|---|
| aluminium rod, 1/8 in diameter, 12 in long | McMaster-Carr | 8974K19 | 1 |
| controller board, Nano 33 IoT | Arduino | (product name only on the bill of materials) | 1 |
| dual motor driver board, TB6612FNG | SparkFun | ROB-14450 | 1 |
| capacitive sensing board, MPR121 | Adafruit | 1982 | 1 |
| 9-axis IMU board, ICM20948 | Pimoroni | PIM448 | 1 |
| power meter board, INA260 | Adafruit | 4226 | 1 |
| pushbutton switch, SPST | CW Industries | GPTS203212B | 1 |
| MOSFET pair, N / P channel | Alpha and Omega Semiconductor | AO4629 | 3 |
| connector, XH, 6 positions (charging port) | JST | S6B-XH-A | 1 |
| connector, PH, 2 positions (battery sockets) | JST | S2B-PH-K-S | 4 |
| connector, SH, 6 positions (motor sockets) | JST | BM06B-SRSS-TB | 2 |
| connector, XH, 2 positions | JST | S2B-XH-A-1 | 2 |
| right-angle pin header, 4 positions (sensor sockets) | Sullins | PRPC004SBAN-M71RC | 2 |
| female header, 15 positions | Sullins | PPPC151LFBN-RC | 2 |
| female header, 13 positions | Sullins | PPPC131LFBN-RC | 1 |
| female header, 8 positions | Sullins | PPPC081LFBN-RC | 2 |
| female header, 7 positions | Sullins | PPPC071LFBN-RC | 1 |
| female header, 5 positions | Sullins | PPPC051LFBN-RC | 1 |
| heat-set threaded insert, 4-40 | McMaster-Carr | 94459A250 | 2 |
| heat-set threaded insert, 2-56 | McMaster-Carr | 94459A240 | 4 |
| socket head cap screw, 4-40 (clamp the rod in the bay) | McMaster-Carr | 92196A108 | 2 |
| socket head cap screw, 2-56 (hold the IMU board) | McMaster-Carr | 92196A088 | 2 |
| spacer (under the IMU board) | McMaster-Carr | 93657A504 | 2 |

## Motor module (`M1`)

| item | maker / vendor | part number | per motor module |
|---|---|---|---|
| 150:1 Micro Metal Gearmotor HPCB 6V with 12 CPR Encoder, Back Connector | Pololu | 5192 | 1 |
| LiPo pouch cell, 220 mAh | bought as an Amazon listing | B08NYCY9SX | 2 |
| heat-set threaded insert, 4-40 | McMaster-Carr | 94459A250 | 4 |
| socket head cap screw, 4-40 | McMaster-Carr | 92196A108 | 4 |
| set screw, 4-40 (winch on the motor shaft) | McMaster-Carr | 92311A102 | 1 |
| cable guide ring (on three of the six modules: `M1a_motor_module.stl`) | Mud Hole (CRB nano micro guide) | BNMG-1 | 1 |

## Bay lid (`M2`)

| item | maker / vendor | part number | per lid |
|---|---|---|---|
| socket head cap screw, 2-56 (the lid's two long screws) | McMaster-Carr | 92196A088 | 2 |

## End cap with cover (`M3`)

No bought parts: a printed resin cap with a cast silicone cover.

Not listed: cables, wires and small passive components (the model files do not contain them), and one small
three-pin block on the main board that is not on the bill of materials.
