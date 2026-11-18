"""The ego car's sensors on the grader side. Entry point: `SensorSuite(model, data, world, car, prefix="car/", seed=None)` with `begin_tick()`,
`after_step(i)` for i = 1 .. N and `end_tick() -> {"scan", "imu", "wheel_speed", "steer_echo"}` in wire form.

Modules: `lidar` (2D LiDAR, sub-scans through mj_multiRay, beam rules), `imu` (200 Hz accelerometer and gyro),
`motor` (motor speed as road speed, steering echo), `suite` (the per-tick schedule), `params` (`sensor_params.json`),
`frames` (chassis pose from the free joint), `reference` (an independent analytic ray caster for tests and gates),
`scenes` (test and benchmark worlds, kinematic motion), `gates` (the package's gate measurements, command line).
"""

from .imu import Imu
from .lidar import Lidar, beam_directions, geom_target_classes
from .motor import MotorSpeed, SteerEcho
from .params import PARAMS_PATH, load_params, noiseless
from .suite import SensorSuite, default_step, steps_per_tick

__all__ = ["PARAMS_PATH", "Imu", "Lidar", "MotorSpeed", "SensorSuite", "SteerEcho", "beam_directions", "default_step",
           "geom_target_classes", "load_params", "noiseless", "steps_per_tick"]
