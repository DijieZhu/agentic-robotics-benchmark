"""Program-side proxy of the intervention vehicle, the program launcher and the practice connections (the only module
of the vehicle cell a program gets; the vehicle, its firmware and its sensors run on the other side of the wire).

    def solve(vehicle):                                   # in /home/user/submission/intervene.py
        wo = vehicle.mission()                            # this site's work order and the run's budgets
        st = vehicle.state()                              # navigation, depth, DVL, attitude, battery, mode, gripper ...
        vehicle.set_velocity(0.3, 0.0, depth=st["depth_m"], heading=st["heading_rad"])
        r = vehicle.step(1.0)                             # 1 s of simulated time; r["done"], r["end_reason"]
        f = vehicle.camera_frame((640, 480))              # f["image"]: uint8 H x W x 3 (RGB)
        s = vehicle.sonar_scan(-45.0, 45.0, 10.0, 1)      # s["intensity"]: uint8 (n_pings, n_samples)
        vehicle.finish()

Every method is one JSON-lines request ({"id", "method", "params"}) answered by the cell ({"id", "result"} or
{"id", "error", "detail"}).  Simulated time passes only in step() and in sonar_scan() (each ping advances the clock by
the ping period); every other call takes no simulated time.  A reply with "error" raises CellError (`.code`:
bad_params, budget, episode_over, sensor_unavailable, no_mission, forbidden, internal_error).  A motion call that
the firmware refuses (a protective stop, the end of the run) is NOT an error: it returns {"accepted": False, "reason":
...}.  Calls outside the 17 below are refused with forbidden and counted.  The launcher rebinds sys.stdout to stderr
first, so ordinary prints never corrupt the wire.

Units: metres, seconds, radians (camera tilt and sonar angles in degrees, as named); quaternions (w, x, y, z), body
axes x forward, y left, z up; world x east, y north, z up; depth positive down; heading = yaw about +z from +x,
counter-clockwise.  vehicle_api.md is the full contract.

Practice facilities (development only, not in an evaluation run): `connect_practice(site)` opens one run on a practice
site (the training tank or the sister site) with the same vehicle and calls plus reset(site) (done here when `site` is
given) and, in the training tank only, instruments().  An evaluation run refuses both.
"""

import base64
import json
import os
import select
import sys
import time
import traceback
import zlib

__all__ = ["Vehicle", "CellError", "PUBLIC_CALLS", "decode_png", "decode_frame", "decode_sonar", "connect_stdio",
           "connect_practice", "connect_handler", "launcher_argv", "main", "BOOTSTRAP", "PROGRAM", "PRACTICE_SOCKET"]

PROGRAM = "intervene"
BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from subseacell.client import main; "
             "sys.exit(main(sys.argv[2:]))")
PYTHON_FLAGS = ["-B", "-E", "-s"]
PRACTICE_SOCKET = "/run/subseacell/practice.sock"
ERROR_KEYS = frozenset(("error", "detail"))
PUBLIC_CALLS = ("mission", "clock", "step", "finish", "state", "set_nav_fix", "idle", "set_attitude", "set_velocity",
                "set_hold", "set_wrench", "camera_frame", "set_camera_tilt", "set_lights", "sonar_scan", "gripper_open",
                "gripper_close")
ALL_AXES = (True, True, True, True, True, True)


class CellError(RuntimeError):
    """A refused or failed call: `code` is the cell's error code, `detail` its explanation."""

    def __init__(self, code, detail=""):
        RuntimeError.__init__(self, code + ((": " + detail) if detail else ""))
        self.code = code
        self.detail = detail


def _plain(v):
    """Parameters as plain JSON (numpy scalars and arrays become numbers and lists)."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, dict):
        return dict((str(k), _plain(x)) for k, x in v.items())
    return v


class Vehicle(object):
    """The proxy (see the module doc)."""

    def __init__(self, send, recv, decode=True):
        self._send = send
        self._recv = recv
        self._id = 0
        self._decode = bool(decode)
        self.debrief = None             # practice: the facility's last line, once the run is over
        self.run_over = None            # practice: why the run ended

    def _call(self, method, **params):
        self._id += 1
        req = {"id": self._id, "method": method, "params": _plain(params)}
        self._send(json.dumps(req, allow_nan=False) + "\n")
        line = self._recv()
        if not line:
            raise CellError("closed", "the cell closed the connection")
        reply = json.loads(line)
        if reply.get("event") == "run_over":
            self.run_over = reply.get("reason")
            self.debrief = reply.get("debrief")
            raise CellError("episode_over", "the practice run is over (%s)" % reply.get("reason"))
        if reply.get("id") != self._id:
            raise CellError("protocol", "reply id mismatch")
        if "error" in reply:
            raise CellError(str(reply["error"]), str(reply.get("detail") or ""))
        res = reply.get("result")
        return _decoded(method, res) if self._decode else res

    def raw_call(self, method, **params):
        """Send one request with this method name and parameters as they are and return the reply's result (the
        calls below have their own methods; any other name is refused with forbidden and counted)."""
        return self._call(str(method), **params)

    # ---- the run
    def mission(self):
        """This site's work order: the chart (structure position and heading, nominal), the deployment cage (exact),
        the as-designed panel layout, the handover status table, the tide forecast, the telemetry summary, the
        dropped-object report and the run's budgets."""
        return self._call("mission")

    def clock(self):
        """{"t": simulated time [s], "sim_time_limit_s", "program_time_used_s", "program_time_limit_s"} (program time
        = the run's wall clock minus the cell's own simulation and rendering)."""
        return self._call("clock")

    def step(self, dt):
        """Advance the simulation by dt seconds (0 < dt <= 10; rounded to whole 10 ms control periods, at least one).
        Returns {"t", "dt" (advanced), "done", "end_reason"}; once the run is over it advances nothing."""
        return self._call("step", dt=float(dt))

    def finish(self):
        """End the run now (the world's state at this moment is what counts)."""
        return self._call("finish")

    def state(self):
        """The vehicle's own state: navigation estimate, depth, DVL, attitude, rates, battery, firmware mode and
        groups, thrusters, camera tilt, gripper (open | closed_empty | stalled_on_object), protective stops, budget
        use.  Takes no simulated time; cheap enough to call every control period."""
        return self._call("state")

    def set_nav_fix(self, x, y):
        """Reset the navigation estimate's horizontal position to (x, y) (your own position fix)."""
        return self._call("set_nav_fix", x=float(x), y=float(y))

    # ---- firmware modes (setpoints are held until the next motion call)
    def idle(self):
        """Thrusters stopped."""
        return self._call("idle")

    def set_attitude(self, roll=0.0, pitch=0.0, heading=None, depth=None, thrust=(0.0, 0.0, 0.0), quat=None):
        """Attitude mode: any attitude (roll, pitch, heading in rad, or quat), optional depth keeping, normalised body
        thrust (-1..1 per axis; thrust z is ignored while depth is held).  heading None = keep the present one."""
        return self._call("set_attitude", roll=roll, pitch=pitch, heading=heading, depth=depth, thrust=thrust,
                          quat=quat)

    def set_velocity(self, u, v, depth, heading):
        """Velocity mode: body velocity over ground u (forward), v (left) [m/s], depth [m], heading [rad]; level
        attitude; needs the DVL bottom lock."""
        return self._call("set_velocity", u=u, v=v, depth=depth, heading=heading)

    def set_hold(self, x, y, depth, heading):
        """Station mode: hold (x, y) of the navigation frame, depth and heading; level attitude; needs the DVL
        bottom lock."""
        return self._call("set_hold", x=x, y=y, depth=depth, heading=heading)

    def set_wrench(self, force=(0.0, 0.0, 0.0), torque=(0.0, 0.0, 0.0), free=ALL_AXES, frame="body", quat=None,
                   depth=None):
        """Wrench mode: the body axes selected by `free` (six booleans Fx Fy Fz Mx My Mz, or a list of axis names)
        take the open-loop force [N] / torque [N m]; the other axes are held (rotations hold `quat`, Fz holds
        `depth`, Fx / Fy keep station on that body axis and need the bottom lock).  frame="world": the force is
        given in world axes (the torque is always body axes)."""
        return self._call("set_wrench", force=force, torque=torque, free=free, frame=frame, quat=quat, depth=depth)

    # ---- sensors and devices
    def camera_frame(self, size=(640, 480)):
        """One colour frame: size (640, 480) costs 1.0 frame unit, (320, 240) 0.40.  Returns {"t", "frame_index",
        "width", "height", "image": uint8 H x W x 3 RGB}."""
        return self._call("camera_frame", size=[int(size[0]), int(size[1])])

    def set_camera_tilt(self, deg):
        """Camera tilt setpoint [deg], optical axis up positive; the servo slews to it (state()["camera_tilt_deg"])."""
        return self._call("set_camera_tilt", deg=float(deg))

    def set_lights(self, level):
        """Common dimming level of the four lamps, 0..1."""
        return self._call("set_lights", level=float(level))

    def sonar_scan(self, start_deg, stop_deg, range_m, gain=1):
        """Mechanical scan from start_deg to stop_deg (degrees, clockwise from the head's forward axis seen from above;
        equal angles = a full turn) in 0.9 deg steps, range_m in metres, gain 0 low / 1 normal / 2 high.  Each ping
        advances the simulated time by its ping period.  Returns {"n_pings", "n_samples", "range_m",
        "sample_period_s", "angle_deg": per ping, "t": per ping, "nav_position_m", "depth_m", "heading_rad",
        "quat_wxyz": the navigation pose at each ping, "valid", "intensity": uint8 (n_pings, n_samples)}."""
        return self._call("sonar_scan", start_deg=float(start_deg), stop_deg=float(stop_deg), range_m=float(range_m),
                          gain=int(gain))

    def gripper_open(self):
        return self._call("gripper_open")

    def gripper_close(self):
        return self._call("gripper_close")

    # ---- practice facilities only (an evaluation run refuses these and counts them)
    def reset(self, site):
        """Start this practice connection's run on practice site `site` (a name of the facility's list)."""
        return self._call("reset", site=str(site))

    def instruments(self):
        """Training tank only: the tank's instrument readings."""
        return self._call("instruments")

    def close(self):
        closer = getattr(self, "_close", None)
        if closer is not None:
            closer()


# ---------------------------------------------------------------------------------------------- decoding
def decode_png(raw):
    """The cell's PNG (8-bit RGB, filter 0 rows, no interlace) -> uint8 H x W x 3; other PNGs through OpenCV or PIL."""
    import numpy as np
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    pos, idat, ihdr = 8, [], None
    while pos + 8 <= len(raw):
        n = int.from_bytes(raw[pos:pos + 4], "big")
        tag = raw[pos + 4:pos + 8]
        data = raw[pos + 8:pos + 8 + n]
        pos += 12 + n
        if tag == b"IHDR":
            ihdr = data
        elif tag == b"IDAT":
            idat.append(data)
        elif tag == b"IEND":
            break
    w, h = int.from_bytes(ihdr[0:4], "big"), int.from_bytes(ihdr[4:8], "big")
    depth, ctype, interlace = ihdr[8], ihdr[9], ihdr[12]
    flat = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8)
    if depth == 8 and ctype == 2 and interlace == 0 and flat.size == h * (w * 3 + 1):
        rows = flat.reshape(h, w * 3 + 1)
        if not rows[:, 0].any():
            return rows[:, 1:].reshape(h, w, 3).copy()
    try:
        import cv2
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        return img[..., ::-1].copy()
    except ImportError:
        import io
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))


def decode_frame(reply):
    """uint8 H x W x 3 RGB image of a raw camera_frame reply."""
    return decode_png(base64.b64decode(reply["image_png_b64"]))


def decode_sonar(reply):
    """uint8 (n_pings, n_samples) echo intensities of a raw sonar_scan reply."""
    import numpy as np
    a = np.frombuffer(base64.b64decode(reply["intensity_u8_b64"]), np.uint8)
    return a.reshape(int(reply["n_pings"]), int(reply["n_samples"])).copy()


def _decoded(method, res):
    if not isinstance(res, dict):
        return res
    if method == "camera_frame" and "image_png_b64" in res:
        out = {k: v for k, v in res.items() if not k.endswith("_b64")}
        out["image"] = decode_frame(res)
        return out
    if method == "sonar_scan" and "intensity_u8_b64" in res:
        out = {k: v for k, v in res.items() if not k.endswith("_b64")}
        out["intensity"] = decode_sonar(res)
        return out
    return res


# ---------------------------------------------------------------------------------------------- transports
class _LineReader(object):
    """Reply lines from a pipe: polls the descriptor for up to `spin_s` before it blocks, so the reply to a short call
    (state(), and the step of a control period) is taken the moment it arrives instead of after a wake-up; a longer
    call blocks as usual.  The polling only happens while the cell is working on the call (its time is not the
    program's)."""

    def __init__(self, fd, spin_s=0.005):
        self.fd = fd
        self.spin_s = float(spin_s)
        self.buf = b""
        os.set_blocking(fd, False)

    def readline(self):
        while True:
            i = self.buf.find(b"\n")
            if i >= 0:
                line, self.buf = self.buf[:i + 1], self.buf[i + 1:]
                return line.decode("utf-8", "replace")
            chunk = self._read()
            if not chunk:
                line, self.buf = self.buf, b""
                return line.decode("utf-8", "replace")
            self.buf += chunk

    def _read(self):
        t_end = None
        while True:
            try:
                return os.read(self.fd, 1 << 20)
            except BlockingIOError:
                pass
            now = time.perf_counter()
            if t_end is None:
                t_end = now + self.spin_s
            if now >= t_end:
                select.select([self.fd], [], [])
                t_end = None


def connect_stdio(spin_s=0.005):
    """Bind the wire to the original stdin / stdout and send prints to stderr."""
    out_fd = os.dup(1)
    inp = _LineReader(os.dup(0), spin_s=spin_s)
    sys.stdout = sys.stderr

    def send(text):
        view = memoryview(text.encode("utf-8"))
        while len(view):
            view = view[os.write(out_fd, view):]

    return Vehicle(send, inp.readline)


def connect_practice(site=None, path=None):
    """A connection to the practice facilities (a Vehicle): one run, started by reset(site) (done here when `site`
    is given)."""
    import socket
    path = path or os.environ.get("SUBSEACELL_PRACTICE_SOCKET") or PRACTICE_SOCKET
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except OSError as e:
        sock.close()
        raise CellError("unreachable", "the practice facilities are not reachable at %s (%s)" % (path, e))
    rf = sock.makefile("r", encoding="utf-8", newline="\n")
    wf = sock.makefile("w", encoding="utf-8", newline="\n")

    def send(text):
        try:
            wf.write(text)
            wf.flush()
        except OSError as e:
            raise CellError("closed", "the practice connection closed (%s)" % e)

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

    vehicle = Vehicle(send, recv)
    vehicle._close = close
    if site is not None:
        vehicle.reset(site)
    return vehicle


def connect_handler(handle, decode=True):
    """A Vehicle whose requests go to `handle(method, params) -> result or {"error", "detail"}` in this process,
    through the same JSON encoding as the wire (tests, your own simulator)."""
    box = []

    def send(text):
        req = json.loads(text)
        res = handle(req["method"], req.get("params") or {})
        if isinstance(res, dict) and "error" in res and set(res) <= ERROR_KEYS:
            msg = dict({"id": req["id"]}, **res)
        else:
            msg = {"id": req["id"], "result": res}
        box.append(json.dumps(msg, allow_nan=False))

    def recv():
        return (box.pop(0) + "\n") if box else ""

    return Vehicle(send, recv, decode=decode)


def launcher_argv(python, lib_dir, program_dir, module=PROGRAM):
    """argv of the program: `bash run.sh` when the submission ships one (regular file), else the launcher
    `python -B -E -s -c <bootstrap> <lib_dir> <module>`."""
    run_sh = os.path.join(program_dir, "run.sh")
    if os.path.isfile(run_sh) and not os.path.islink(run_sh):
        return ["bash", "run.sh"]
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: import intervene.py from the current directory and run solve(vehicle); finish() at the end."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else PROGRAM
    vehicle = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        getattr(mod, "solve")(vehicle)
    except CellError as e:
        print("[intervene] cell error: %s" % e, file=sys.stderr)
        code = 2
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        code = 1
    finally:
        try:
            vehicle.finish()
        except Exception:  # noqa: BLE001
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
