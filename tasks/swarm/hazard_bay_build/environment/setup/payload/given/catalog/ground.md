# Ground vehicles and arms

## Wheeled bases

Bases for the ground vehicle (Isaac library, robots folder). `payload` is what the
deck may carry (arm and sensors, by their catalog masses).

| model | mass [kg] | payload [kg] | asset |
|---|---|---|---|
| `jackal` | 17.00 | 20.00 | `Clearpath/Jackal` |
| `dingo` | 9.00 | 20.00 | `Clearpath/Dingo` |
| `limo` | 4.20 | 2.00 | `AgilexRobotics/limo` |
| `jetbot` | 0.90 | 0.50 | `NVIDIA/Jetbot` |
| `kaya` | 1.30 | 1.00 | `NVIDIA/Kaya` |
| `leatherback` | 3.50 | 2.00 | `NVIDIA/Leatherback` |
| `create3` | 3.30 | 9.00 | `iRobot/Create3` |
| `turtlebot` | 1.00 | 5.00 | `Turtlebot/` |

## Arms

Arms for the flying robots and the vehicle (Isaac library, robots folder). `payload` is the rated payload at the
wrist; `reach` the nominal reach from the base. Declare with `arb:arm_model`.

| model | mass [kg] | payload [kg] | reach [m] | asset |
|---|---|---|---|---|
| `so101` | 0.70 | 0.25 | 0.35 | `RobotStudio/so101` |
| `so100` | 0.70 | 0.25 | 0.35 | `RobotStudio/so100` |
| `ur3` | 11.00 | 3.00 | 0.50 | `UniversalRobots/ur3/` |
| `ur3e` | 11.20 | 3.00 | 0.50 | `UniversalRobots/ur3e/` |
| `ur5` | 18.40 | 5.00 | 0.85 | `UniversalRobots/ur5/` |
| `ur5e` | 20.60 | 5.00 | 0.85 | `UniversalRobots/ur5e/` |
| `ur10` | 28.90 | 10.00 | 1.30 | `UniversalRobots/ur10/` |
| `ur10e` | 33.50 | 12.50 | 1.30 | `UniversalRobots/ur10e/` |
| `ur16e` | 33.10 | 16.00 | 0.90 | `UniversalRobots/ur16e/` |
| `ur20` | 64.00 | 20.00 | 1.75 | `UniversalRobots/ur20/` |
| `ur30` | 63.50 | 30.00 | 1.30 | `UniversalRobots/ur30/` |
| `kinova_gen3` | 8.20 | 4.00 | 0.90 | `Kinova/Gen3` |
| `kinova_jaco2` | 5.50 | 1.30 | 0.90 | `Kinova/Jaco2` |
| `xarm6` | 12.20 | 5.00 | 0.70 | `Ufactory/xarm6` |
| `xarm7` | 13.70 | 3.50 | 0.70 | `Ufactory/xarm7` |
| `lite6` | 7.30 | 0.60 | 0.44 | `Ufactory/lite6` |
| `uf850` | 18.50 | 5.00 | 0.85 | `Ufactory/uf850` |
| `franka_panda` | 18.00 | 3.00 | 0.85 | `FrankaRobotics/Franka` |
| `franka_factory` | 18.00 | 3.00 | 0.85 | `FrankaRobotics/FactoryFranka` |
| `sawyer` | 19.00 | 4.00 | 1.26 | `RethinkRobotics/Sawyer` |
| `tm12` | 33.30 | 12.00 | 1.30 | `Techman/TM12` |
| `cobotta_pro_900` | 27.00 | 6.00 | 0.90 | `Denso/CobottaPro900` |
| `cobotta_pro_1300` | 45.00 | 12.00 | 1.30 | `Denso/CobottaPro1300` |
| `rizon4` | 18.50 | 4.00 | 0.90 | `Flexiv/Rizon4` |
| `kr210` | 1100.00 | 150.00 | 2.70 | `Kuka/KR210` |
| `openarm` | 8.00 | 2.00 | 0.60 | `OpenArm/openarm_unimanual` |
| `mecademic_meca500` | 4.60 | 0.50 | 0.33 | `Mecademic/` |
