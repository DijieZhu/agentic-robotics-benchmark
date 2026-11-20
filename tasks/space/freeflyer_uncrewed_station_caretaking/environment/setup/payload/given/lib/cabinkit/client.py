"""cabinkit.client -- the program-side proxy of the robot, and the program launcher.

    def solve(robot):                              # in ~/submission/caretake.py
        m = robot.mission()                        # work order, 72 h log, configuration record, register, map, ...
        r = robot.undock()                         # action calls return {"code": ..., "t": ...}
        f = robot.camera("FIC")                    # f["frame_id"], f["intrinsics"], f["image"] (uint8 H x W x 3, RGB)
        robot.step(1.0)                            # simulated time passes only in step() and in blocking calls
        robot.dock(m["berth"])

Every method is one JSON-lines request ({"id", "method", "params"}) answered by the cell ({"id", "result"} or
{"id", "error", "detail"}).  Arguments are sent by name (robot_api.md lists them, with every return code and budget).
Frames are decoded into numpy arrays: ``image`` (colour, uint8 H x W x 3, RGB), ``depth`` (float32 H x W metres,
0 = no return) and ``amplitude`` (uint16 H x W) of a depth camera; the raw reply keys end in ``_b64``.  A reply that
carries "error" raises ``CellError`` (``.code``: ``forbidden`` for a call outside the API, counted by the cell;
``run_over`` once the run has ended).  The launcher rebinds ``sys.stdout`` to stderr first, so ordinary prints never
corrupt the wire.
"""

import base64
import io
import json
import os
import sys
import traceback

__all__ = ["Robot", "CellError", "decode_frame", "connect_stdio", "connect_handler", "main", "BOOTSTRAP", "launcher_argv",
           "PROGRAM", "PYTHON_FLAGS"]

PROGRAM = "caretake"
BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from cabinkit.client import main; "
             "sys.exit(main(sys.argv[2:]))")
PYTHON_FLAGS = ["-B", "-E", "-s"]
ERROR_KEYS = frozenset(("error", "detail"))


class CellError(RuntimeError):
    """A failed call: ``code`` is the cell's error code, ``detail`` its explanation."""

    def __init__(self, code, detail=""):
        RuntimeError.__init__(self, code + ((": " + detail) if detail else ""))
        self.code = code
        self.detail = detail


def _plain(v):
    """Arguments as plain JSON (numpy scalars and arrays become numbers and lists)."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return dict((str(k), _plain(x)) for k, x in v.items())
    return v


class Robot(object):
    """The proxy.  Units: metres, seconds, radians (tilt_deg / pan_deg in degrees); quaternions (w, x, y, z)."""

    def __init__(self, send, recv, decode=True):
        self._send = send
        self._recv = recv
        self._id = 0
        self._decode = bool(decode)

    def _call(self, method, **params):
        self._id += 1
        req = {"id": self._id, "method": method, "params": _plain(params)}
        try:
            self._send(json.dumps(req) + "\n")
        except OSError:
            raise CellError("closed", "the cell closed the connection")
        line = self._recv()
        if not line:
            raise CellError("closed", "the cell closed the connection")
        reply = json.loads(line)
        if reply.get("id") != self._id:
            raise CellError("protocol", "reply id mismatch")
        if "error" in reply:
            raise CellError(str(reply["error"]), str(reply.get("detail") or ""))
        res = reply.get("result")
        if self._decode and method == "camera" and isinstance(res, dict):
            return decode_frame(res)
        return res

    # ---- time
    def step(self, dt):
        """Advance the simulation by dt seconds (at most the step limit of robot_api.md)."""
        return self._call("step", dt=dt)

    # ---- queries
    def status(self):
        return self._call("status")

    def pose(self):
        return self._call("pose")

    def mission(self):
        return self._call("mission")

    def station_telemetry(self):
        return self._call("station_telemetry")

    # ---- motion
    def set_mode(self, mode):
        return self._call("set_mode", mode=mode)

    def fly_to(self, pos, quat, face_forward=True, max_speed=None):
        return self._call("fly_to", pos=pos, quat=quat, face_forward=face_forward, max_speed=max_speed)

    def fly_rel(self, fiducial_id, pos, quat, max_speed=None):
        return self._call("fly_rel", fiducial_id=fiducial_id, pos=pos, quat=quat, max_speed=max_speed)

    def stop(self):
        return self._call("stop")

    def hold(self):
        return self._call("hold")

    def dock(self, berth):
        return self._call("dock", berth=berth)

    def undock(self):
        return self._call("undock")

    # ---- arm and gripper
    def arm_deploy(self, wait=False):
        return self._call("arm_deploy", wait=wait)

    def arm_stow(self, wait=False):
        return self._call("arm_stow", wait=wait)

    def arm(self, tilt_deg, pan_deg, wait=False):
        return self._call("arm", tilt_deg=tilt_deg, pan_deg=pan_deg, wait=wait)

    def gripper(self, open):  # noqa: A002 (the API's argument name)
        return self._call("gripper", open=open)

    def capture(self, bar_pos, bar_axis, palm_dir=None):
        return self._call("capture", bar_pos=bar_pos, bar_axis=bar_axis, palm_dir=palm_dir)

    # ---- sensors, lamps, station
    def camera(self, name):
        """name: FIC | FWC | AWC (colour) or FDC | ADC (depth)."""
        return self._call("camera", name=name)

    def lamp(self, which, on):
        """which: forward | aft."""
        return self._call("lamp", which=which, on=on)

    def station_request(self, kind, target, value):
        return self._call("station_request", kind=kind, target=target, value=value)

    # ---- the diagnosis
    def report(self, inlet_id, item_ref, onset_time, frame_id):
        return self._call("report", inlet_id=inlet_id, item_ref=item_ref, onset_time=onset_time, frame_id=frame_id)

    def close(self):
        closer = getattr(self, "_close", None)
        if closer is not None:
            closer()


# ---------------------------------------------------------------------------------------------- decoding
def _png(b64):
    import numpy as np
    raw = base64.b64decode(b64)
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        if img is not None:
            return img[..., ::-1].copy() if img.ndim == 3 else img
    except ImportError:
        pass
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(raw)))


def decode_frame(reply):
    """A camera reply with its pixels as numpy arrays (image / depth / amplitude; absent when the reply has none)."""
    import numpy as np
    out = {k: v for k, v in reply.items() if not k.endswith("_b64")}
    if reply.get("image_png_b64"):
        out["image"] = _png(reply["image_png_b64"])[..., :3]
    if reply.get("depth_png_b64"):
        mm = _png(reply["depth_png_b64"]).astype(np.float32)
        out["depth"] = mm * np.float32(reply.get("depth_unit_m", 0.001))
    if reply.get("amplitude_png_b64"):
        out["amplitude"] = _png(reply["amplitude_png_b64"]).astype(np.uint16)
    return out


# ---------------------------------------------------------------------------------------------- transports
def connect_stdio():
    """Bind the wire to the original stdin / stdout and send prints to stderr."""
    out = os.fdopen(os.dup(1), "w", buffering=1)
    inp = os.fdopen(os.dup(0), "r", buffering=1)
    sys.stdout = sys.stderr

    def send(text):
        out.write(text)
        out.flush()

    def recv():
        return inp.readline()

    return Robot(send, recv)


def connect_handler(handle, decode=True):
    """A Robot whose requests go to ``handle(method, params) -> result or {"error", "detail"}`` in this process,
    through the same JSON encoding as the wire (tests, a team's own simulator)."""
    box = []

    def send(text):
        req = json.loads(text)
        res = handle(req["method"], req.get("params") or {})
        if isinstance(res, dict) and "error" in res and set(res) <= ERROR_KEYS:
            msg = dict({"id": req["id"]}, **res)
        else:
            msg = {"id": req["id"], "result": res}
        box.append(json.dumps(msg))

    def recv():
        return (box.pop(0) + "\n") if box else ""

    return Robot(send, recv, decode=decode)


def launcher_argv(python, lib_dir, program_dir, module=PROGRAM):
    """argv of the program: the launcher ``python -B -E -s -c <bootstrap> <lib_dir> <module>``, run in program_dir."""
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: import caretake.py from the current directory and run solve(robot)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else PROGRAM
    robot = connect_stdio()
    sys.path.insert(0, os.getcwd())
    try:
        mod = __import__(module)
        getattr(mod, "solve")(robot)
    except CellError as e:
        print("[%s] cell error: %s" % (module, e), file=sys.stderr)
        return 2
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
