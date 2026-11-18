"""The program's side of the wire: the `Car` proxy, the connections and the launcher.

    def solve(car):                          # in drive.py; the launcher calls it once per venue
        info = car.describe()                # venue sheet, session table, budgets, timing, the car's constants
        car.open_session("practice")
        r = car.step(0.0, 1.0)               # steer target (rad, left positive), speed target (m/s): one 25 ms tick
        r["scan"], r["imu"], r["wheel_speed"], r["steer_echo"], r["events"]
        car.close_session()

Every call is one JSON-lines request on this process's original stdout, answered on its stdin. The launcher rebinds
`sys.stdout` to stderr first, so ordinary prints never corrupt the wire. An error reply raises `CarError` (its
`code` is one of forbidden, bad_params, no_session, bad_order, venue_over). Standard library only; numpy is used
for the scan and the IMU rows when it can be imported (`Car(numpy=False)` keeps plain lists).
"""

import io
import json
import os
import socket
import sys
import traceback

from . import protocol

__all__ = ["Car", "CarError", "connect_stdio", "connect_socket", "connect_handler", "main", "BOOTSTRAP",
           "PYTHON_FLAGS", "launcher_argv"]

BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from carkit.client import main; "
             "sys.exit(main(sys.argv[2:]))")
PYTHON_FLAGS = ["-B", "-E", "-s"]

try:                                        # numpy is optional for the client
    import numpy as _np
except ImportError:                         # pragma: no cover - the task image has numpy
    _np = None


class CarError(RuntimeError):
    """An error reply of the venue (or a broken connection: code `closed`)."""

    def __init__(self, code, detail=None, method=None):
        self.code = code
        self.detail = detail
        self.method = method
        super().__init__("%s%s%s" % (code, (": " + detail) if detail else "", (" (%s)" % method) if method else ""))


class Car(object):
    """The proxy of the car and the venue. `rpc(line: bytes) -> bytes` carries one request line and returns one reply
    line; use `connect_stdio()` (the graded run), `connect_socket(path)` or `connect_handler(fn)` to make one."""

    def __init__(self, rpc, numpy=None, close=None):
        self._rpc = rpc
        self._id = 0
        self._close = close
        self.numpy = (_np is not None) if numpy is None else (bool(numpy) and _np is not None)

    # ---------------------------------------------------------------- the wire
    def call(self, method, **params):
        """One request of any method; returns the `result` object, raises CarError on an error reply."""
        self._id += 1
        req = {"id": self._id, "method": method}
        if params:
            req["params"] = params
        line = self._rpc((json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8"))
        if not line:
            raise CarError("closed", "the venue closed the connection", method)
        reply = json.loads(line)
        if "error" in reply:
            raise CarError(reply["error"], reply.get("detail"), method)
        if reply.get("id") != self._id:
            raise CarError("closed", "reply id %r does not answer request %d" % (reply.get("id"), self._id), method)
        return reply["result"]

    # ---------------------------------------------------------------- the public calls
    def describe(self):
        """Venue sheet, sessions, budgets, timing and the car's constants. Costs no simulated time."""
        return self.call("describe")

    def open_session(self, name):
        """Open the next session (practice, tt1, tt2, avoid, in this order). The car stands at the start, at rest."""
        return self.call("open_session", name=str(name))

    def step(self, steer, speed):
        """Advance one tick (25 ms) with these targets. Returns tick, scan (1081 ranges in mm, 0 = no return), imu
        (5 x 6), wheel_speed, steer_echo and events."""
        r = self.call("step", steer=float(steer), speed=float(speed))
        if self.numpy:
            r["scan"] = _np.asarray(r["scan"], dtype=_np.int32)
            r["imu"] = _np.asarray(r["imu"], dtype=_np.float64)
        return r

    def close_session(self):
        """End the open session now, or acknowledge one that has ended."""
        return self.call("close_session")

    def request_pickup(self):
        """Ask the staff to take the car to the track edge (the lap is interrupted; the staff's hold time passes)."""
        return self.call("request_pickup")

    def close(self):
        if self._close is not None:
            self._close()


# -------------------------------------------------------------------- connections
def connect_stdio(numpy=None):
    """Bind the wire to this process's original stdin / stdout and send prints to stderr."""
    out_fd = os.dup(1)
    inp = io.open(os.dup(0), "rb", buffering=1 << 16)
    sys.stdout = sys.stderr

    def rpc(data):
        view = memoryview(data)
        while len(view):
            n = os.write(out_fd, view)
            view = view[n:]
        return inp.readline(protocol.MAX_LINE_BYTES + 1)

    def close():
        try:
            os.close(out_fd)
        except OSError:
            pass
        inp.close()

    return Car(rpc, numpy=numpy, close=close)


def connect_socket(path, numpy=None):
    """A connection over a Unix socket (development only; the same messages as the graded run)."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.connect(path)
    rf = sock.makefile("rb")

    def rpc(data):
        sock.sendall(data)
        return rf.readline(protocol.MAX_LINE_BYTES + 1)

    def close():
        rf.close()
        sock.close()

    return Car(rpc, numpy=numpy, close=close)


def connect_handler(handler, numpy=None):
    """An in-process connection: `handler(line: bytes) -> bytes` answers one request line (development and tests)."""
    return Car(handler, numpy=numpy)


# -------------------------------------------------------------------- the launcher
def launcher_argv(python, lib_dir, module="drive"):
    """argv that runs `solve(car)` of `<module>.py` in the current directory through this launcher."""
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Import `<module>` (default drive) from the current directory and call its solve(car) once."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else "drive"
    car = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        mod.solve(car)
    except CarError as e:
        print("[carkit] %s" % e, file=sys.stderr)
        code = 2
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        code = 1
    finally:
        car.close()
    return code


if __name__ == "__main__":
    sys.exit(main())
