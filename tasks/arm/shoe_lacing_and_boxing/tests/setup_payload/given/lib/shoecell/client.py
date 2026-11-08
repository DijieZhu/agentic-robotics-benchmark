"""Agent-side proxy of the shoe lacing and boxing cell + the program launcher (PUBLIC CODE).

    def solve(cell):                                   # in /home/user/submission/packer.py
        info = cell.describe()
        f = cell.get_camera("overhead")                # one frame: PNG rgb + 16-bit PNG depth (mm, 0 = no reading)
        rgb, depth_m = decode_frame(f)                 # uint8 H x W x 3, float32 H x W metres (NaN where invalid)
        st = cell.get_state()                          # both arms' own state; no object poses
        cell.set_control("left", "cartesian", 0.6)
        cell.command("left", tcp_pos=[...], tcp_quat=[w, x, y, z], width=0.08)
        r = cell.step(20)                              # 20 x 10 ms of physics; r["events"], r["done"]
        cell.hook(True)                                # the right tool's clip
        cell.finish()

Every method is one JSON-lines request on this process's original stdout, answered on stdin by the grader's
physics server. The launcher rebinds `sys.stdout` to stderr first, so ordinary prints never corrupt the wire.
A reply carrying "error" raises CellError (a `forbidden` error is counted by the grader as an illegal call).
"""

import base64
import io
import json
import os
import sys
import traceback

__all__ = ["Cell", "CellError", "decode_frame", "connect_stdio", "main", "BOOTSTRAP", "launcher_argv"]

BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from shoecell.client import main; "
             "sys.exit(main(sys.argv[2:]))")
PYTHON_FLAGS = ["-B", "-E", "-s"]


class CellError(RuntimeError):
    pass


class Cell(object):
    """The proxy. Units: metres, radians, quaternions wxyz, world frame = the bench frame (cell_manual.md)."""

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

    # ---- information
    def describe(self):
        return self._call("describe")

    def get_state(self):
        return self._call("get_state")

    def get_camera(self, name="overhead", depth=True):
        return self._call("get_camera", name=str(name), depth=bool(depth))

    # ---- motion
    def set_control(self, arm, mode="joint", stiffness_scale=1.0):
        return self._call("set_control", arm=str(arm), mode=str(mode), stiffness_scale=float(stiffness_scale))

    def command(self, arm, q=None, tcp_pos=None, tcp_quat=None, width=None):
        params = {"arm": str(arm)}
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

    # ---- the cell's fixtures and tools
    def hook(self, on=True):
        return self._call("hook", on=bool(on))

    def release_clamp(self):
        return self._call("release_clamp")

    def finish(self):
        return self._call("finish")

    # ---- practice servers only (the graded cell answers {"error": "forbidden"})
    def reset(self, seed=0):
        return self._call("reset", seed=int(seed))

    def truth(self):
        return self._call("truth")

    # ---- privileged (the reference program's token only)
    def get_state_privileged(self):
        return self._call("get_state_privileged")

    def reference_run(self):
        return self._call("reference_run")


def decode_frame(reply):
    """(rgb uint8 H x W x 3, depth float32 H x W in metres with NaN where the sensor reported no reading / None when absent)."""
    import numpy as np
    from PIL import Image
    rgb = np.asarray(Image.open(io.BytesIO(base64.b64decode(reply["rgb_png_b64"]))).convert("RGB"))
    depth = None
    if reply.get("depth_u16_mm_b64"):
        d = np.asarray(Image.open(io.BytesIO(base64.b64decode(reply["depth_u16_mm_b64"]))), dtype=np.float32)
        depth = d * float(reply.get("depth_unit_m", 0.001))
        depth[d == float(reply.get("depth_invalid_value", 0))] = np.nan
    return rgb, depth


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


def launcher_argv(python, lib_dir, program_dir, module="packer"):
    """argv of the program: `bash run.sh` when the submission ships one (regular file), else the client launcher
    `python -B -E -s -c <bootstrap> <lib_dir> <module>`."""
    run_sh = os.path.join(program_dir, "run.sh")
    if os.path.isfile(run_sh) and not os.path.islink(run_sh):
        return ["bash", "run.sh"]
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: import packer.py from the current directory and run solve(cell)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else "packer"
    cell = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        getattr(mod, "solve")(cell)
    except CellError as e:
        print("[packer] cell error: %s" % e, file=sys.stderr)
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
