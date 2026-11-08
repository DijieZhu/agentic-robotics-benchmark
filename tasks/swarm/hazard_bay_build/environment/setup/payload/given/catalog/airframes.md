# Airframes

Multirotor airframes the flying robots may be built on. `empty` is the airframe with its battery and
flight electronics; `MTOW` the maximum take-off mass, everything included; `thrust max` what its motors can give
together (the mission's controller is clamped to it, and to a 40° tilt of the thrust from vertical). The four
`arb_*` frames are in `/opt/assets/arb/airframes/` (each a rigid body `Body` with a payload mount frame `Body/Mount`
under the hull and a cargo hook frame `Body/Hook` for a rope); the others are in the Isaac library (robots folder).

| model | empty [kg] | MTOW [kg] | thrust max [N] | asset |
|---|---|---|---|---|
| `crazyflie` | 0.027 | 0.042 | 0.6 | `Bitcraze/Crazyflie` |
| `arl_robot_1` | 1.250 | 2.20 | 45 | `NTNU/ARL-Robot-1` |
| `quadcopter` | 0.900 | 1.60 | 32 | `IsaacSim/Quadcopter` |
| `arb_quad_450` | 1.150 | 2.40 | 48 | `airframes/arb_quad_450` |
| `arb_quad_650` | 2.400 | 6.00 | 120 | `airframes/arb_quad_650` |
| `arb_hex_800` | 4.100 | 12.00 | 220 | `airframes/arb_hex_800` |
| `arb_x8_1000` | 6.900 | 18.00 | 360 | `airframes/arb_x8_1000` |

Declare the one used with `arb:airframe_model` under the robot; the robot must be built from that asset.
