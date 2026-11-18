"""Program-side proxy of the inspection robot, the program launcher and the practice-ground connection (the one module
of the robot cell a team gets; the robot's simulation runs out of sight, in the scored missions and in the practice
ground).

    def solve(robot):                                   # in /home/user/submission/patrol.py
        info = robot.describe()                         # the robot's datasheet
        docs = robot.mission()                          # drawing, work order, round sheet, modification record, ...
        st = robot.get_state()                          # odometry (site frame), IMU, joints, dosimeter, link, ...
        robot.set_mode("walk"); robot.walk(0.3, 0.0, 0.0)
        r = robot.step(50)                              # 50 x 10 ms of simulated time; r["events"], r["done"]
        f = robot.get_camera("inspect")                 # f["image"]: uint8 H x W x 3 (RGB), f["frame_id"], K, T_body_cam
        n = robot.get_camera("nav", depth=True)         # n["depth"]: float32 H x W metres, NaN = no reading
        th = robot.get_thermal()                        # th["image_c"]: float32 H x W degrees C
        sw = robot.get_lidar()                          # sw["ranges"], sw["angles"]: float32 (beams,), NaN = no return
        robot.report("PI-015", 6.4, [f["frame_id"]])    # value, cited frames (at most 3)
        robot.finish()

Every method is one JSON-lines request ({"id", "method", "params"}) answered by the cell ({"id", "result"} or
{"id", "error", "detail"}). The proxy decodes images and arrays into numpy (key names above; the raw reply keys end in
_b64). A reply carrying "error" raises CellError (`.code`: bad_params, budget, mission_over, forbidden, ...; every
`forbidden` call is counted as an illegal call). The launcher rebinds `sys.stdout` to stderr first, so
ordinary prints never corrupt the wire.

Practice ground (development only): `connect_practice(site)` opens a mission on one of the practice ground's sites:
the same robot and calls, plus `reset(site)`, which starts the mission (done here when `site` is given). One
connection is one mission, under the scored missions' budgets; the practice ground has a fixed number of missions.
The mission's debrief arrives as the connection's last line (`Robot.debrief` after the mission ends).
`~/training/run_mission.py` runs a whole program in a practice mission the way a scored mission runs it and prints
the debrief.
"""

import base64
import io
import json
import os
import sys
import traceback

__all__ = ["Robot", "CellError", "decode_frame", "decode_depth", "decode_thermal", "decode_lidar", "connect_stdio",
           "connect_practice", "connect_handler", "main", "BOOTSTRAP", "launcher_argv", "PRACTICE_SOCKET",
           "PRACTICE_STATUS", "PROGRAM"]

PROGRAM = "patrol"
BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from plantcell.client import main; "
             "sys.exit(main(sys.argv[2:]))")
PYTHON_FLAGS = ["-B", "-E", "-s"]
PRACTICE_SOCKET = "/run/plantcell/practice.sock"
PRACTICE_STATUS = "/run/plantcell/practice_status.json"
ERROR_KEYS = frozenset(("error", "detail"))


class CellError(RuntimeError):
    """A refused or failed call: `code` is the cell's error code, `detail` its explanation."""

    def __init__(self, code, detail=""):
        RuntimeError.__init__(self, code + ((": " + detail) if detail else ""))
        self.code = code
        self.detail = detail


class Robot(object):
    """The proxy. Units: metres, radians, seconds; the site frame and the body frame are described by describe()."""

    def __init__(self, send, recv, decode=True):
        self._send = send
        self._recv = recv
        self._id = 0
        self._decode = bool(decode)
        self.debrief = None             # practice ground: the mission's debrief, once the mission is over
        self.mission_over = None        # practice ground: why the mission ended

    def _over(self, reply):
        self.mission_over = reply.get("reason")
        self.debrief = reply.get("debrief")
        raise CellError("mission_over", "the practice mission is over (%s)" % reply.get("reason"))

    def _call(self, method, **params):
        self._id += 1
        req = {"id": self._id, "method": method, "params": params}
        try:
            self._send(json.dumps(req) + "\n")
        except (CellError, OSError):
            # the other side may have closed right after its last line (the practice ground's debrief): read it
            try:
                line = self._recv()
            except OSError:
                line = ""
            try:
                last = json.loads(line) if line else None
            except ValueError:
                last = None
            if isinstance(last, dict) and last.get("event") == "mission_over":
                self._over(last)
            raise CellError("closed", "the cell closed the connection")
        line = self._recv()
        if not line:
            raise CellError("closed", "the cell closed the connection")
        reply = json.loads(line)
        if reply.get("event") == "mission_over":
            self._over(reply)
        if reply.get("id") != self._id:
            raise CellError("protocol", "reply id mismatch")
        if "error" in reply:
            raise CellError(str(reply["error"]), str(reply.get("detail") or ""))
        res = reply.get("result")
        return _decoded(method, res) if self._decode else res

    # ---- information
    def describe(self):
        """The robot's datasheet: dimensions, modes, cameras (intrinsics K), lidar, head, time, budgets,
        dosimeter, odometry, link, the grid marker plates' markers (`tags`), the hold after a `blocked` report
        (`blocked_hold_s`), frames."""
        return self._call("describe")

    def mission(self):
        """This site's documents: the drawing, the work order, the round sheet, the modification record, the
        facility facts, the plant catalogue (with the tag plates and the plant lettering) and the enforced
        budgets."""
        return self._call("mission")

    def get_state(self):
        """Sim time, done, status (it reads `blocked` for the whole hold after a `blocked` report, also after a mode
        change), odometry (site frame), IMU, joints, foot contacts, mode, body, head, lamp, dosimeter (counts since
        the previous get_state, their dwell, accumulated dose), link, reports pending, frames and reports used."""
        return self._call("get_state")

    # ---- sensors (each frame / sweep counts against the mission's budget)
    def get_camera(self, name="nav", depth=True):
        """name: nav (640 x 480, 90 deg, with depth) | inspect (1280 x 960, 18 deg, colour only)."""
        return self._call("get_camera", name=str(name), depth=bool(depth))

    def get_depth(self):
        """A depth-only frame of the nav camera."""
        return self._call("get_depth")

    def get_thermal(self):
        return self._call("get_thermal")

    def get_lidar(self):
        return self._call("get_lidar")

    # ---- motion (commands held until changed; time passes only in step)
    def walk(self, vx=0.0, vy=0.0, yaw_rate=0.0):
        """Velocity command in the body frame (m/s, m/s, rad/s); a moving command in stand or sit switches to walk.
        After a `blocked` report no walking command is taken for describe()["blocked_hold_s"] seconds of simulated
        time (5 s): the reply is then {"ok": False, "held": True, "hold_until": <sim time the hold ends>, "detail":
        ...} and nothing changes (a zero command neither); otherwise {"ok": True}. The walking command in force is
        dropped at the report, so after the hold the robot stands until the next walking command."""
        return self._call("walk", vx=float(vx), vy=float(vy), yaw_rate=float(yaw_rate))

    def set_mode(self, m):
        """stand | walk | high_step | crouch_walk | sit. During the hold after a `blocked` report the mode is taken at
        once (the robot stands in it) and the hold goes on; walking starts with the next walking command after it."""
        return self._call("set_mode", m=str(m))

    def set_body(self, height=None, pitch=None):
        p = {}
        if height is not None:
            p["height"] = float(height)
        if pitch is not None:
            p["pitch"] = float(pitch)
        return self._call("set_body", **p)

    def set_head(self, pan=None, tilt=None):
        p = {}
        if pan is not None:
            p["pan"] = float(pan)
        if tilt is not None:
            p["tilt"] = float(tilt)
        return self._call("set_head", **p)

    def set_lamp(self, on):
        return self._call("set_lamp", on=bool(on))

    def step(self, n_ticks=1):
        """Advance n_ticks x 10 ms (at most 500 per call): {"sim_time", "done", "events"}."""
        return self._call("step", n_ticks=int(n_ticks))

    # ---- reports
    def report(self, tag, value=None, frame_ids=()):
        """An instrument report: the value (None = a photo without a reading) and up to 3 cited frame ids."""
        if isinstance(frame_ids, str):
            frame_ids = [frame_ids]
        return self._call("report", tag=str(tag), value=_plain(value), frame_ids=[str(f) for f in (frame_ids or [])])

    def report_mark(self, tag, dose_rate):
        """The dose rate (Gy/h) measured at a survey mark."""
        return self._call("report_mark", tag=str(tag), dose_rate=None if dose_rate is None else float(dose_rate))

    def report_source(self, x, y):
        """The unlisted source's position (site frame, m)."""
        return self._call("report_source", x=float(x), y=float(y))

    def finish(self):
        return self._call("finish")

    # ---- the practice ground only (a scored mission answers forbidden)
    def reset(self, site):
        """Start this connection's practice mission on practice site `site` (a name or number of the status
        file's list). Once per connection; it uses one mission of the practice ground's allowance."""
        return self._call("reset", site=str(site))

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
    """uint8 H x W x 3 RGB image of a get_camera reply (raw)."""
    return _png(reply["image_png_b64"])[..., :3]


def decode_depth(reply):
    """float32 H x W depth in metres (NaN where the camera had no reading) of a get_camera / get_depth reply (raw),
    or None when the reply has no depth."""
    import numpy as np
    if not reply.get("depth_png_b64"):
        return None
    d = _png(reply["depth_png_b64"]).astype(np.float32)
    z = d * float(reply.get("depth_unit_m", 0.001))
    z[d == float(reply.get("depth_invalid", 0))] = np.nan
    return z


def decode_thermal(reply):
    """float32 H x W apparent temperatures in degrees C of a get_thermal reply (raw; the PNG is centi-kelvin)."""
    import numpy as np
    ck = _png(reply["thermal_ck_png_b64"]).astype(np.float32)
    return ck / 100.0 - 273.15


def decode_lidar(reply):
    """(ranges, angles): float32 (beams,) metres (NaN = no return) and the beam angles (rad, counter-clockwise from
    the body's x axis) of a get_lidar reply (raw)."""
    import numpy as np
    r = np.frombuffer(base64.b64decode(reply["ranges_f32_b64"]), dtype="<f4").astype(np.float32)
    a = (np.arange(len(r)) * float(reply["angle_step_rad"])).astype(np.float32)
    return r, a


def _decoded(method, res):
    """The reply with images and arrays as numpy (the raw _b64 keys removed)."""
    if not isinstance(res, dict):
        return res
    import numpy as np
    if method in ("get_camera", "get_depth", "get_thermal"):
        out = {k: v for k, v in res.items() if not k.endswith("_b64")}
        if "image_png_b64" in res:
            out["image"] = decode_frame(res)
        if "depth_png_b64" in res:
            out["depth"] = decode_depth(res)
        if "thermal_ck_png_b64" in res:
            out["image_c"] = decode_thermal(res)
        for k in ("K", "T_body_cam"):
            if k in out:
                out[k] = np.asarray(out[k], float)
        return out
    if method == "get_lidar":
        out = {k: v for k, v in res.items() if not k.endswith("_b64")}
        out["ranges"], out["angles"] = decode_lidar(res)
        if "T_body_lidar" in out:
            out["T_body_lidar"] = np.asarray(out["T_body_lidar"], float)
        return out
    return res


def _plain(v):
    """A report value as plain JSON (numpy scalars and arrays become numbers and lists)."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return dict((str(k), _plain(x)) for k, x in v.items())
    return v


# ---------------------------------------------------------------------------------------------- transports
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

    return Robot(send, recv)


def connect_practice(site=None, path=None):
    """A connection to the practice ground (a Robot): one mission, started by `reset(site)` (done here when `site`
    is given). The service runs a few missions at a time; the allowance left is in PRACTICE_STATUS."""
    import socket
    path = path or PRACTICE_SOCKET
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except OSError as e:
        sock.close()
        raise CellError("unreachable", "the practice ground is not reachable at %s (%s)" % (path, e))
    rf = sock.makefile("r", encoding="utf-8", newline="\n")
    wf = sock.makefile("w", encoding="utf-8", newline="\n")

    def send(text):
        try:
            wf.write(text)
            wf.flush()
        except OSError as e:
            raise CellError("closed", "the practice ground closed the connection (%s)" % e)

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

    robot = Robot(send, recv)
    robot._close = close
    if site is not None:
        robot.reset(site)
    return robot


def connect_handler(handle, decode=True):
    """A Robot whose requests go to `handle(method, params) -> result or {"error", "detail"}` in this process, through
    the same JSON encoding as the wire (tests, a team's own simulator)."""
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
    """argv of the program: `bash run.sh` when the submission ships one (regular file), else the client launcher
    `python -B -E -s -c <bootstrap> <lib_dir> <module>`."""
    run_sh = os.path.join(program_dir, "run.sh")
    if os.path.isfile(run_sh) and not os.path.islink(run_sh):
        return ["bash", "run.sh"]
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: import patrol.py from the current directory and run solve(robot)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else PROGRAM
    robot = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        getattr(mod, "solve")(robot)
    except CellError as e:
        print("[patrol] cell error: %s" % e, file=sys.stderr)
        code = 2
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        code = 1
    finally:
        try:
            robot.finish()
        except Exception:  # noqa: BLE001
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
