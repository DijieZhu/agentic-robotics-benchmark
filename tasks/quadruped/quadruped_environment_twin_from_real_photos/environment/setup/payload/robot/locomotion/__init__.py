"""Locomotion controller of the 12-DoF quadruped (plain numpy / scipy / mujoco, CPU, deterministic).

Entry point for the mission code: `QuadrupedController` (see controller.py for the interface).
"""

from .controller import QuadrupedController, MODES, Mode, Gains
from .plant import RobotIO, SensorNoise

__all__ = ["QuadrupedController", "MODES", "Mode", "Gains", "RobotIO", "SensorNoise"]
