"""The program's side of the station's wire: the ``Cell`` proxy that
``solve(cell)`` receives, the launcher, the commissioning connection and
frame decoding (the one module of the cell a program gets; the cell itself
runs in the grader and in the commissioning service).

    def solve(cell):                                 # in sort.py
        info = cell.describe()                       # static facts (geometry, cameras, budgets, ...)
        st = cell.get_state()                        # the robot's own state (joints, tools, grippers), budgets
        f = cell.get_camera("head", depth=True)      # newest delivered frame of a camera
        rgb, depth_m = decode_frame(f)               # uint8 H x W x 3, float32 H x W m (NaN = no reading)
        st = cell.step(q_target=q14, grip=[0, 1], n=5)   # 5 ticks of 20 ms holding the targets
        st = cell.step(tool_pose={"left": {"pos": [0.3, 0.2, 1.1], "quat": [0, 1, 0, 0]}}, n=25)
        cell.finish()

Every call is one JSON-lines request on this process's original stdout,
answered on its stdin. The launcher moves the wire away from the standard
streams first: ordinary prints, and native libraries writing to file
descriptor 1, go to stderr, never into the wire. A reply carrying "error"
raises CellError (``.code`` and ``.detail``).

Commissioning (development only): ``connect_commissioning(site)`` opens one
shift on a development shift of the commissioning service: the same calls,
plus ``reset(site)`` which starts it. ``~/commission/run_shift.py`` runs a
whole program the way the grader runs it.

Runtime dependencies: the Python standard library; numpy and Pillow for
``decode_frame`` only.
"""

import base64
import io
import json
import os
import sys
import traceback

__all__ = ["Cell", "CellError", "decode_frame", "decode_rgb", "decode_depth", "connect_stdio",
           "connect_commissioning", "main", "BOOTSTRAP", "launcher_argv", "COMMISSION_SOCKET",
           "COMMISSION_STATUS", "CAMERAS"]

BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from parcelcell.client import main; "
             "sys.exit(main(sys.argv[2:]))")
PYTHON_FLAGS = ["-B", "-E", "-s"]
PROGRAM = "sort"
COMMISSION_SOCKET = "/run/parcelcell/commission.sock"
COMMISSION_STATUS = "/run/parcelcell/commission_status.json"
CAMERAS = ("infeed", "head", "wrist_left", "wrist_right")


class CellError(RuntimeError):
    def __init__(self, code, detail=None):
        RuntimeError.__init__(self, code if not detail else "%s: %s" % (code, detail))
        self.code = code
        self.detail = detail


def _floats(v, n, name):
    if v is None:
        return None
    out = [float(x) for x in v]
    if len(out) != n:
        raise ValueError("%s needs %d values" % (name, n))
    return out


class Cell(object):
    """The proxy. Units: metres, radians, seconds; world frame: the robot's column at
    the origin, z up, the robot facing +x, the belt running along +y."""

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
            raise CellError("closed", "the cell closed the connection")
        reply = json.loads(line)
        if reply.get("event") == "shift_over":
            raise CellError("shift_over", "the commissioning shift is over (%s)" % reply.get("reason"))
        if reply.get("id") != self._id:
            raise CellError("protocol", "reply id mismatch")
        if "error" in reply:
            raise CellError(reply["error"], reply.get("detail"))
        return reply.get("result")

    # ---- information
    def describe(self):
        """Static facts: tick, joints and limits, start posture, gripper, station geometry
        and lines, camera table, sensor models, the shift's duration, budgets, safety and
        score numbers."""
        return self._call("describe")

    def get_state(self):
        """t, q, dq, tau, q_target, tool {left, right: {pos, quat}} (each tool centre
        point's measured pose), tool_target {left, right: null or {pos, quat, reached,
        out_of_reach, error_m, error_deg}}, gripper {command, opening_m, object_detected},
        stop {active, until, count, locked_out, link}, frames_left, program_time_left,
        shift_over, over_reason."""
        return self._call("get_state")

    def get_camera(self, name, depth=False):
        """The newest delivered frame of camera `name` (infeed, head, wrist_left,
        wrist_right): PNG RGB (+ 16-bit depth in mm when depth=True), its exposure
        `stamp`, for the wrist cameras `T_world_cam` at exposure. A repeat of the same
        frame is free."""
        return self._call("get_camera", name=str(name), depth=bool(depth))

    # ---- motion
    def step(self, q_target=None, grip=None, n=1, tool_pose=None):
        """Advance n ticks (1..50, 20 ms each) holding the targets.

        q_target: 14 joint position targets (rad, both arms), or {"left": 7 values,
        "right": 7 values} (either arm may be left out), or None (keep).
        tool_pose: {"left": {"pos": [x, y, z], "quat": [w, x, y, z]}, "right": ...}: a
        target pose of that arm's tool centre point in the world frame (the cell's
        controller solves the arm's joints every tick), or None (keep). One level per arm
        per step: an arm cannot get joint targets and a tool pose in the same call.
        grip: [left, right] in 0 (open) .. 1 (closed) or None (keep).
        Returns the state after the ticks (plus `ticks` run and `refused` when a
        protective stop refused the targets)."""
        params = {"n": int(n)}
        if isinstance(q_target, dict):
            q = {}
            for k, v in q_target.items():
                q[str(k)] = _floats(v, 7, "q_target[%r]" % (k,))
        else:
            q = _floats(q_target, 14, "q_target")
        g = _floats(grip, 2, "grip")
        if q is not None:
            params["q_target"] = q
        if g is not None:
            params["grip"] = g
        if tool_pose is not None:
            tp = {}
            for k, p in tool_pose.items():
                tp[str(k)] = None if p is None else {"pos": _floats(p["pos"], 3, "pos"),
                                                     "quat": _floats(p["quat"], 4, "quat")}
            params["tool_pose"] = tp
        return self._call("step", **params)

    def finish(self):
        """End the shift now (parcels not yet done earn nothing: cell_manual.md section 8)."""
        return self._call("finish")

    # ---- privileged (the reference program's token only)
    def get_state_privileged(self):
        return self._call("get_state_privileged")

    def reference_run(self):
        return self._call("reference_run")

    # ---- the commissioning service only (the graded cell answers {"error": "forbidden"})
    def reset(self, site):
        """Start this connection's commissioning shift on development shift `site`
        (1 .. the number of development shifts). Once per connection."""
        return self._call("reset", site=int(site))

    def close(self):
        closer = getattr(self, "_close", None)
        if closer is not None:
            closer()


def decode_rgb(reply):
    """uint8 H x W x 3 of a frame reply."""
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(reply["rgb_png_b64"]))).convert("RGB"))


def decode_depth(reply):
    """float32 H x W depth in metres with NaN where the sensor had no reading, or None."""
    import numpy as np
    from PIL import Image
    if not reply.get("depth_png_b64"):
        return None
    d = np.asarray(Image.open(io.BytesIO(base64.b64decode(reply["depth_png_b64"]))), dtype=np.float32)
    out = d * float(reply.get("depth_unit_m", 0.001))
    out[d == float(reply.get("depth_invalid", 0))] = np.nan
    return out


def decode_frame(reply):
    """(rgb uint8 H x W x 3, depth float32 H x W metres with NaN where invalid / None)."""
    return decode_rgb(reply), decode_depth(reply)


def connect_stdio():
    """Bind the wire to the original stdin / stdout; afterwards file descriptor 1 and
    sys.stdout both point at stderr (prints and native writes never reach the wire)."""
    wire_out = os.dup(1)
    os.dup2(2, 1)                      # fd 1 -> stderr: native code's writes leave the wire alone
    wire_in = os.dup(0)
    out = os.fdopen(wire_out, "w", buffering=1)
    inp = os.fdopen(wire_in, "r", buffering=1)
    sys.stdout = sys.stderr

    def send(text):
        out.write(text)
        out.flush()

    def recv():
        return inp.readline()

    return Cell(send, recv)


def connect_commissioning(site=None, path=None):
    """A connection to the commissioning service (a Cell): one shift, started by
    `reset(site)` (done here when `site` is given)."""
    import socket
    path = path or COMMISSION_SOCKET
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except OSError as e:
        sock.close()
        raise CellError("unreachable", "the commissioning service is not reachable at %s (%s)" % (path, e))
    rf = sock.makefile("r", encoding="utf-8", newline="\n")
    wf = sock.makefile("w", encoding="utf-8", newline="\n")

    def pending_event():
        """The service's last line before it closed the connection, if it sent one (shift_over)."""
        try:
            sock.settimeout(2.0)
            line = rf.readline()
        except (OSError, ValueError):
            return None
        try:
            obj = json.loads(line) if line else None
        except ValueError:
            return None
        return obj if isinstance(obj, dict) and obj.get("event") else None

    def send(text):
        try:
            wf.write(text)
            wf.flush()
        except OSError as e:
            ev = pending_event()
            if ev is not None and ev.get("event") == "shift_over":
                raise CellError("shift_over", "the commissioning shift is over (%s)" % ev.get("reason"))
            raise CellError("closed", "the commissioning service closed the connection (%s)" % e)

    def recv():
        try:
            return rf.readline()
        except OSError:
            return ""

    def close():
        for f in (wf, rf):
            try:
                f.close()
            except OSError:
                pass
        sock.close()

    cell = Cell(send, recv)
    cell._close = close
    if site is not None:
        cell.reset(site)
    return cell


def launcher_argv(python, lib_dir, program_dir, module=PROGRAM):
    """argv of the program: `bash run.sh` when the submission ships one (regular file),
    else the client launcher `python -B -E -s -c <bootstrap> <lib_dir> <module>`."""
    run_sh = os.path.join(program_dir, "run.sh")
    if os.path.isfile(run_sh) and not os.path.islink(run_sh):
        return ["bash", "run.sh"]
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: import sort.py (or the named module) from the current directory and
    run solve(cell)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else PROGRAM
    cell = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        getattr(mod, "solve")(cell)
    except CellError as e:
        print("[sort] cell error: %s" % e, file=sys.stderr)
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
