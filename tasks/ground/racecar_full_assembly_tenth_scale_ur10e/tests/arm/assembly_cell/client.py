"""Agent-side proxy of the assembly cell + the launcher (`python3 -m assembly_cell.client`).

    def solve(cell):            # in /home/user/submission/assemble.py
        info = cell.describe()
        st = cell.get_state()
        cell.set_control("cartesian", 0.35)
        cell.command(tcp_pos=[...], tcp_quat=[w, x, y, z], width=0.08)
        r = cell.step(20)       # 20 x 10 ms; r["events"], r["done"]
        cell.finish()

Every method is one JSON-lines request on this process's original stdout, answered on stdin by the
grader. The launcher rebinds `sys.stdout` to stderr first, so ordinary prints never corrupt the wire.
A reply carrying "error" raises CellError (a `forbidden` error is counted by the grader).
"""

import json
import os
import sys
import traceback

__all__ = ["Cell", "CellError", "connect_stdio", "main"]


class CellError(RuntimeError):
    pass


class Cell(object):
    def __init__(self, send, recv):
        self._send = send
        self._recv = recv
        self._id = 0

    def _call(self, method, **params):
        self._id += 1
        req = {"id": self._id, "method": method, "params": params}
        self._send(json.dumps(req) + "\n")
        line = self._recv()
        if not line:
            raise CellError("the cell closed the connection")
        reply = json.loads(line)
        if reply.get("id") != self._id:
            raise CellError("reply id mismatch")
        if "error" in reply:
            raise CellError(reply["error"])
        return reply.get("result")

    def describe(self):
        return self._call("describe")

    def get_state(self):
        return self._call("get_state")

    def set_control(self, mode="joint", stiffness_scale=1.0):
        return self._call("set_control", mode=mode, stiffness_scale=float(stiffness_scale))

    def command(self, q=None, tcp_pos=None, tcp_quat=None, width=None):
        params = {}
        if q is not None:
            params["q"] = [float(v) for v in q]
        if tcp_pos is not None:
            params["tcp_pos"] = [float(v) for v in tcp_pos]
        if tcp_quat is not None:
            params["tcp_quat"] = [float(v) for v in tcp_quat]
        if width is not None:
            params["width"] = float(width)
        return self._call("command", **params)

    def step(self, n_ticks=1):
        return self._call("step", n_ticks=int(n_ticks))

    def get_camera(self, name="overview"):
        return self._call("get_camera", name=name)

    def finish(self):
        return self._call("finish")

    # practice servers only
    def reset(self, seed=0):
        return self._call("reset", seed=int(seed))


def connect_stdio():
    """Bind the wire to the original stdin/stdout and send prints to stderr."""
    out = os.fdopen(os.dup(1), "w", buffering=1)
    inp = os.fdopen(os.dup(0), "r", buffering=1)
    sys.stdout = sys.stderr

    def send(text):
        out.write(text)
        out.flush()

    def recv():
        return inp.readline()

    return Cell(send, recv)


def main(argv=None):
    """Launcher: import assemble.py from the current directory and run solve(cell)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else "assemble"
    cell = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        solve = getattr(mod, "solve")
        solve(cell)
    except CellError as e:
        print("[assemble] cell error: %s" % e, file=sys.stderr)
        code = 2
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        code = 1
    finally:
        try:
            cell.finish()
        except Exception:  # noqa: BLE001
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
