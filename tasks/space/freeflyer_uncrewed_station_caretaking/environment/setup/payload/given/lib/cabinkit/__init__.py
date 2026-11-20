"""cabinkit: the client library of the robot API (robot_api.md).  ``from cabinkit.client import Robot, CellError``."""

from .client import CellError, Robot, connect_handler, connect_stdio, decode_frame  # noqa: F401

__version__ = "1.0.0"
