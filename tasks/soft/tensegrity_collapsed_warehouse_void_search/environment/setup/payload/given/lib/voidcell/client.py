"""Program-side proxy of the search station, the program launcher and the practice-ground connection (the one module
of the station a team gets; the robot and the site are simulated out of sight, in the scored episodes and in the
practice ground).

    def solve(station):                                 # in /home/user/submission/operate.py
        m = station.mission()                           # site rectangle, pad, search points, camera K and extrinsic,
                                                        # budgets, pose source
        f = station.get_frame()                         # f["rgb"]: uint8 480 x 640 x 3; f["depth"]: uint16 480 x 640
                                                        # (x depth_unit_m = metres along the optical axis, 0 = none)
        tl = station.telemetry()                        # 100 Hz records since the last call (numpy arrays)
        r = station.run_primitive("roll_120_120", [1, 2, 5])   # {"accepted": True/False, "error": None or a code}
        while station.status()["busy"]:
            station.step(0.5)                           # simulated time passes only here
        station.report_point(1)                         # {"accepted", "reports_left", "error"}
        station.finish()

Every method is one JSON-lines request ({"id", "method", "params"}) answered by the station ({"id", "result"} or
{"id", "error", "detail"}). A command that the station does not take (busy, invalid_name, invalid_bottom,
out_of_range, bad_argument, budget_exhausted) comes back as a result with "accepted": False and its "error"; any
other refusal raises StationError (`.code`: bad_argument, budget_exhausted, not_available, episode_over, forbidden,
internal_error). The launcher rebinds `sys.stdout` to stderr first, so ordinary prints never corrupt the wire.
Only the standard library and numpy are needed.

Practice ground (development only): `connect_practice(site)` opens an episode on one of the practice ground's sites:
the same station and calls, plus `reset(site)`, which starts the episode (done here when `site` is given). One
connection is one episode, under the scored episodes' budgets; the practice ground has a fixed number of episodes.
The episode's debrief arrives as the connection's last line (`Station.debrief` after the episode ends).
"""

import base64
import json
import os
import sys
import traceback
import zlib

__all__ = ["Station", "StationError", "decode_frame", "decode_telemetry", "connect_stdio", "connect_practice",
           "connect_handler", "main", "BOOTSTRAP", "launcher_argv", "PRACTICE_SOCKET", "PRACTICE_STATUS", "PROGRAM"]

PROGRAM = "operate"
BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from voidcell.client import main; "
             "sys.exit(main(sys.argv[2:]))")
PYTHON_FLAGS = ["-B", "-E", "-s"]
PRACTICE_SOCKET = "/run/voidcell/practice.sock"
PRACTICE_STATUS = "/run/voidcell/practice_status.json"
ERROR_KEYS = frozenset(("error", "detail"))
TELEMETRY_ARRAYS = ("t", "sensor_length_mm", "sensor_capacitance_pF", "motor_encoder_counts",
                    "motor_encoder_length_mm", "motor_target_mm", "motor_done", "motor_duty", "imu_accel_g",
                    "imu_gyro_dps")
# the shape of one record of each column (an empty reply decodes to arrays of shape (0,) + this)
TELEMETRY_SHAPES = {"t": (), "sensor_length_mm": (9,), "sensor_capacitance_pF": (9,), "motor_encoder_counts": (6,),
                    "motor_encoder_length_mm": (6,), "motor_target_mm": (6,), "motor_done": (6,), "motor_duty": (6,),
                    "imu_accel_g": (3, 3), "imu_gyro_dps": (3, 3)}


class StationError(RuntimeError):
    """A refused or failed call: `code` is the station's error code, `detail` its explanation."""

    def __init__(self, code, detail=""):
        RuntimeError.__init__(self, code + ((": " + detail) if detail else ""))
        self.code = code
        self.detail = detail


class Station(object):
    """The proxy. Units: metres, seconds, millimetres for cable and sensor lengths; frames in mission()["site"]."""

    def __init__(self, send, recv, decode=True):
        self._send = send
        self._recv = recv
        self._id = 0
        self._decode = bool(decode)
        self.debrief = None             # practice ground: the episode's debrief, once the episode is over
        self.episode_over = None        # practice ground: why the episode ended

    def _over(self, reply):
        self.episode_over = reply.get("reason")
        self.debrief = reply.get("debrief")
        raise StationError("episode_over", "the practice episode is over (%s)" % reply.get("reason"))

    def _call(self, method, **params):
        self._id += 1
        req = {"id": self._id, "method": method, "params": params}
        try:
            self._send(json.dumps(req) + "\n")
        except (StationError, OSError):
            try:
                line = self._recv()
            except OSError:
                line = ""
            try:
                last = json.loads(line) if line else None
            except ValueError:
                last = None
            if isinstance(last, dict) and last.get("event") == "episode_over":
                self._over(last)
            raise StationError("closed", "the station closed the connection")
        line = self._recv()
        if not line:
            raise StationError("closed", "the station closed the connection")
        reply = json.loads(line)
        if reply.get("event") == "episode_over":
            self._over(reply)
        if reply.get("id") != self._id:
            raise StationError("protocol", "reply id mismatch")
        if "error" in reply:
            raise StationError(str(reply["error"]), str(reply.get("detail") or ""))
        res = reply.get("result")
        if not self._decode:
            return res
        if method == "get_frame":
            return decode_frame(res)
        if method == "telemetry":
            return decode_telemetry(res)
        return res

    # ---- information
    def mission(self):
        """The episode's mission: site rectangle and frame, pad, search points (id, xy, weight), dwell and credit
        constants, the camera (K = [fx, fy, cx, cy], depth unit, rate, T_world_camera), the budgets, pose_source."""
        return self._call("mission")

    def get_frame(self):
        """The newest camera frame: {index, t, rgb (uint8 H x W x 3), depth (uint16 H x W), depth_unit_m,
        frames_left}. Each call uses one frame of the episode's budget."""
        return self._call("get_frame")

    def telemetry(self):
        """The 100 Hz records since the previous call (at most the last 30 s) as numpy arrays (keys in
        TELEMETRY_ARRAYS; motor_target_mm is NaN where no command runs), plus n and dropped."""
        return self._call("telemetry")

    def status(self):
        """{t, busy, kind, state_index, done[6], timeouts}: what the off-board cable controller knows."""
        return self._call("status")

    def pose(self):
        """Only when mission()["pose_source"] == "cell": {t, index, nodes_xyz (6 x 3 or None), visible_caps}."""
        return self._call("pose")

    # ---- commands (time passes only in step)
    def run_primitive(self, name, bottom_nodes):
        """Start one primitive; bottom_nodes = the three node numbers on the ground (one of the six rollable
        faces). Not blocking: it runs in the following steps."""
        return self._call("run_primitive", name=str(name), bottom_nodes=[int(x) for x in bottom_nodes])

    def set_targets(self, lengths_mm):
        """One target shape: six short-sensor lengths (mm) for motors 0-5."""
        return self._call("set_targets", lengths_mm=[float(x) for x in lengths_mm])

    def stop(self):
        return self._call("stop")

    def report_point(self, i):
        return self._call("report_point", i=int(i))

    def step(self, dt):
        """Advance the simulation by dt seconds (0.01-10.0, a multiple of 0.01): {t, done}."""
        return self._call("step", dt=round(float(dt), 2))

    def finish(self):
        return self._call("finish")

    # ---- the practice ground only (a scored episode answers forbidden)
    def reset(self, site):
        return self._call("reset", site=str(site))

    def close(self):
        closer = getattr(self, "_close", None)
        if closer is not None:
            closer()


# ---------------------------------------------------------------------------------------------- decoding
def decode_frame(res):
    """A get_frame reply (raw) with rgb and depth as numpy arrays (the _b64 keys removed)."""
    import numpy as np
    H, W = int(res["height"]), int(res["width"])
    out = {k: v for k, v in res.items() if not k.endswith("_b64")}
    out["rgb"] = np.frombuffer(zlib.decompress(base64.b64decode(res["rgb_b64"])), np.uint8).reshape(H, W, 3).copy()
    out["depth"] = np.frombuffer(zlib.decompress(base64.b64decode(res["depth_b64"])), "<u2").reshape(H, W).astype(
        np.uint16)
    return out


def decode_telemetry(res):
    """A telemetry reply (raw) with its columns as numpy arrays of shape (n,) + TELEMETRY_SHAPES[key]; a reply with
    no records (n = 0, e.g. before the first step) gives arrays of the same column shape with n = 0."""
    import numpy as np
    out = {k: v for k, v in res.items() if k not in TELEMETRY_ARRAYS}
    for k in TELEMETRY_ARRAYS:
        if k not in res:
            continue
        v = res[k]
        shape = (-1,) + TELEMETRY_SHAPES[k]
        if k == "motor_target_mm":
            out[k] = np.array([[np.nan if x is None else x for x in row] for row in v], float).reshape(shape)
        elif k == "motor_done":
            out[k] = np.array(v, bool).reshape(shape)
        elif k == "motor_encoder_counts":
            out[k] = np.array(v, np.int64).reshape(shape)
        else:
            out[k] = np.array(v, float).reshape(shape)
    return out


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

    return Station(send, recv)


def connect_practice(site=None, path=None):
    """A connection to the practice ground (a Station): one episode, started by `reset(site)` (done here when
    `site` is given). The allowance left is in PRACTICE_STATUS."""
    import socket
    path = path or PRACTICE_SOCKET
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except OSError as e:
        sock.close()
        raise StationError("unreachable", "the practice ground is not reachable at %s (%s)" % (path, e))
    rf = sock.makefile("r", encoding="utf-8", newline="\n")
    wf = sock.makefile("w", encoding="utf-8", newline="\n")

    def send(text):
        try:
            wf.write(text)
            wf.flush()
        except OSError as e:
            raise StationError("closed", "the practice ground closed the connection (%s)" % e)

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

    st = Station(send, recv)
    st._close = close
    if site is not None:
        st.reset(site)
    return st


def connect_handler(handle, decode=True):
    """A Station whose requests go to `handle(method, params) -> result or {"error", "detail"}` in this process,
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

    return Station(send, recv, decode=decode)


def launcher_argv(python, lib_dir, program_dir, module=PROGRAM):
    """argv of the program: `bash run.sh` when the submission ships one (regular file), else the client launcher
    `python -B -E -s -c <bootstrap> <lib_dir> <module>`."""
    run_sh = os.path.join(program_dir, "run.sh")
    if os.path.isfile(run_sh) and not os.path.islink(run_sh):
        return ["bash", "run.sh"]
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: import operate.py from the current directory and run solve(station)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else PROGRAM
    station = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        getattr(mod, "solve")(station)
    except StationError as e:
        print("[operate] station error: %s" % e, file=sys.stderr)
        code = 2
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=sys.stderr)
        code = 1
    finally:
        try:
            station.finish()
        except Exception:  # noqa: BLE001
            pass
    return code


if __name__ == "__main__":
    sys.exit(main())
