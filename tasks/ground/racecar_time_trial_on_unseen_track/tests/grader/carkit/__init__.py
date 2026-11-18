"""carkit: the client library of the car's program interface (protocol racecar-wire/1).

    from carkit import Car, CarError, SessionLog, read_log, scan

`Car` is the proxy a program's `solve(car)` receives; `scan` holds the beam geometry; `SessionLog` decodes the
organiser's events; `read_log` reads lab logs. Standard library only (numpy is used when present).
"""

from . import events, logs, protocol, scan
from .client import Car, CarError, connect_handler, connect_socket, connect_stdio, launcher_argv
from .events import SessionLog
from .logs import LabLog, read_log

__all__ = ["Car", "CarError", "SessionLog", "LabLog", "read_log", "connect_stdio", "connect_socket", "connect_handler",
           "launcher_argv", "events", "logs", "protocol", "scan"]
__version__ = "0.1.0"
