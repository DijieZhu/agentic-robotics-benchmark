# Unitree Go1 with its inspection equipment: datasheet

Values of the robot as simulated (MuJoCo 3.16, 2 ms time step, elliptic friction cones). "Per site" means: drawn
once for each graded site and for each pad episode, uniformly from the stated range, and not reported; a program
finds a site's values from what its robot measures there.

## Body

| quantity | value |
|---|---|
| mass without the equipment | 12.74 kg (trunk 5.20 kg; each leg 1.88 kg: hip 0.68, thigh 1.01, calf and foot 0.19) |
| trunk inertia about its centre (x, y, z) | 0.017, 0.063, 0.072 kg m^2 |
| hip pivots | x = +/-0.1881 m, y = +/-0.04675 m in the trunk frame |
| thigh axis | 0.08 m outboard of the hip pivot |
| thigh, shank | 0.213 m each |
| foot | sphere, radius 0.023 m |
| home pose | hip 0, thigh 0.9, calf -1.8 rad; the trunk origin then stands about 0.27 m above the floor |

## Joints (FR, FL, RR, RL; hip, thigh, calf)

| quantity | value |
|---|---|
| command | position target; torque = stiffness x (target - angle) - damping x speed, clipped at the limit |
| stiffness | 100 N m/rad x a per-joint factor, **per site in [0.85, 1.15]** |
| damping | hip 1, thigh 2, calf 2 N m s/rad |
| torque limit | hip and thigh 23.7 N m, calf 35.55 N m, x one factor for all joints, **per site in [0.85, 1.00]** |
| rotor inertia, dry friction | 0.01 kg m^2, 0.2 N m |
| target range | hip [-0.863, 0.863], thigh [-0.686, 4.501], calf [-2.818, -0.888] rad |

## Equipment and load

| part | what it is, and its envelope in the trunk frame |
|---|---|
| 3D lidar | Ouster OS1 class: a cylinder 0.087 m in diameter and 0.0745 m tall, axis on the trunk's Z axis, top at Z +0.187 m; 0.447 kg |
| lidar guard frame | 12 mm tube: a rectangle X [-0.102, 0.102], Y [-0.085, 0.085] with its top at Z +0.199 m, on four corner posts down to Z +0.057 m; 0.20 kg. The highest point of the equipment |
| lidar mast | a cylinder 0.024 m in diameter between the compute unit and the lidar; 0.033 kg |
| compute and IMU unit | 0.100 x 0.080 x 0.030 m on the trunk under the mast, top at Z +0.087 m; 0.25 kg |
| floor profiler | LMI Gocator 2340 class line profiler, mounted crosswise, 0.049 x 0.197 x 0.075 m: X [0.341, 0.390], Y [-0.0985, 0.0985], Z [-0.163, -0.088] m; 0.74 kg. The lowest point of the equipment |
| profiler bracket | 0.150 x 0.040 x 0.020 m under the nose: X [0.225, 0.375], Z [-0.070, -0.050] m; 0.10 kg |
| instrument pack | on a rail 0.06 m above the trunk origin, on the centre line |
| total carried mass (the 1.77 kg of equipment above + pack) | **per site in [2.50, 4.50] kg** |
| pack position along the trunk's x axis | **per site in [+0.05, +0.28] m** ahead of the trunk origin |

A heavier or more nose-ward pack makes the legs sag and the trunk pitch under the same joint targets.

## Ground

| quantity | value |
|---|---|
| sliding friction of every walked surface (floor, aprons, deck, guides) | **per site in [0.45, 0.90]** |
| torsional, rolling friction | 0.02, 0.001 |

## Fittings (see the instruction for the rest of their dimensions)

| quantity | value |
|---|---|
| underside of the suspended duct above the floor | **per site in [0.410, 0.445] m** |
| crest of the deck's guides above the deck | **per site in [0.070, 0.112] m** |

Both show in the survey: the teal cover lies 1.4 mm above the duct's roof (the roof is 0.068 m above the
underside) and 1.4 mm above the centre guide's crest (the deck is 0.065 m above the floor).

## Sensors

| reading | what it is |
|---|---|
| `joint_rad` | joint angles: noise 0.002 rad (one standard deviation per reading), a constant bias per joint within 0.008 rad, rounded to 1e-4 |
| `joint_rad_s` | joint speeds, rounded to 1e-4 |
| `gravity_body_unit` | unit vector of gravity in the trunk frame: noise 0.01 per axis before normalising, rounded to 1e-5 |
| `gyro_body_rad_s` | trunk angular velocity in the trunk frame: noise 0.01 rad/s, a constant bias per axis within 0.015 rad/s, rounded to 1e-4 |
| `foot_contact` | per foot: touching the floor, an apron, the deck or a guide |
| `odom_xy_m`, `odom_yaw_rad` | leg odometry since the settled launch pose, in the launch frame: each step's body-frame displacement times a distance factor, laid along the odometry's own heading; that heading is the true yaw change plus a constant error rate |
| odometry heading error rate | **per site in [-0.80, +0.80] degrees per second**, also while standing |
| odometry distance factor | **per site in [0.94, 1.06]** |
| mast camera | 256 x 256 RGB pinhole, 70 degree vertical field of view, 0.20 m above the trunk origin, 36 degrees up, eight pan detents of 45 degrees; exact, no noise; 24 exposures per deployment |

There is no joint torque, range, depth or absolute pose reading.

## Survey UAV

| quantity | value |
|---|---|
| nadir camera | 256 x 256 RGB pinhole, 78 degree vertical field of view, 1.65 m above the floor; exact |
| `tracking_xy_m` | the UAV's true displacement times a factor, **per site in [0.97, 1.03]** |
