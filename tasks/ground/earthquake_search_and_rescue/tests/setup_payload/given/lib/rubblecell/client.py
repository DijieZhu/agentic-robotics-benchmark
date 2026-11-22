"""Agent-side proxy of the SR-2 rescue crawler + the program launcher + the training-ground connection (the one module
of the robot cell the agent gets; the simulation runs in the grader and in the training-ground service, out of sight).

    def solve(robot):                                  # in /home/user/submission/rescue.py
        info = robot.describe()
        st = robot.get_state()                         # odometry pose (map frame), IMU, joints, link, kit, ...
        robot.drive(0.3, 0.0)                          # forward speed m/s, yaw rate rad/s (held until changed)
        r = robot.step(50)                             # 50 x 10 ms of simulated time; r["events"], r["done"]
        f = robot.get_camera("mast")                   # PNG colour + 16-bit depth (mm, 0 = no reading)
        rgb, depth_m = decode_frame(f)                 # uint8 H x W x 3, float32 H x W metres (NaN = no reading)
        pts = decode_lidar(robot.get_lidar())          # float32 N x 4: x, y, z (lidar frame), channel
        temp_c = decode_thermal(robot.get_thermal())   # float32 H x W degrees Celsius
        robot.report_survivor(x, y, z)                 # map frame; reaches the team when the radio link is up
        robot.finish()

Every method is one JSON-lines request on this process's original stdout, answered on its stdin by the cell. The
launcher rebinds `sys.stdout` to stderr first, so ordinary prints never corrupt the wire. A reply carrying "error"
raises CellError (a `forbidden` error is counted by the grader as an illegal call).

Training ground (development only): `connect_training(site)` opens a mission on one of the training ground's sites
(`robot_manual.md` section 9): the same robot and calls, plus `reset(site)`, which starts the mission. One connection is
one mission, under the graded mission's budgets; the training ground has a fixed number of missions.
`~/training/run_mission.py` runs a whole program in a mission under the grader's checks and run regime and prints the
robot's mission log.
"""

import base64
import io
import json
import os
import sys
import traceback

__all__ = ["Robot", "CellError", "decode_frame", "decode_lidar", "decode_thermal", "connect_stdio", "connect_training",
           "main", "BOOTSTRAP", "launcher_argv", "TRAINING_SOCKET", "TRAINING_STATUS"]

BOOTSTRAP = ("import sys; sys.path.insert(0, sys.argv[1]); from rubblecell.client import main; "
             "sys.exit(main(sys.argv[2:]))")
PYTHON_FLAGS = ["-B", "-E", "-s"]
TRAINING_SOCKET = "/run/rubblecell/training.sock"
TRAINING_STATUS = "/run/rubblecell/training_status.json"


class CellError(RuntimeError):
    pass


class Robot(object):
    """The proxy. Units: metres, radians, seconds; frames in robot_api.md (map frame = the robot's pose at deployment)."""

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
        if reply.get("event") == "mission_over":
            raise CellError("the training mission is over (%s)" % reply.get("reason"))
        if reply.get("id") != self._id:
            raise CellError("reply id mismatch")
        if "error" in reply:
            raise CellError(reply["error"] + ((": " + reply["detail"]) if reply.get("detail") else ""))
        return reply.get("result")

    # ---- information
    def describe(self):
        return self._call("describe")

    def get_state(self):
        return self._call("get_state")

    # ---- sensors (each frame / sweep counts against the mission's budget)
    def get_camera(self, name="mast", depth=True):
        """name: mast | front | rear | gripper."""
        return self._call("get_camera", name=str(name), depth=bool(depth))

    def get_lidar(self):
        return self._call("get_lidar")

    def get_thermal(self):
        return self._call("get_thermal")

    # ---- motion (targets held until changed; time passes only in step)
    def drive(self, v=0.0, w=0.0):
        return self._call("drive", v=float(v), w=float(w))

    def set_flippers(self, front=None, rear=None):
        p = {}
        if front is not None:
            p["front"] = float(front)
        if rear is not None:
            p["rear"] = float(rear)
        return self._call("set_flippers", **p)

    def set_gauge(self, width):
        return self._call("set_gauge", width=float(width))

    def set_mast(self, height):
        return self._call("set_mast", height=float(height))

    def set_head(self, pan=None, tilt=None):
        p = {}
        if pan is not None:
            p["pan"] = float(pan)
        if tilt is not None:
            p["tilt"] = float(tilt)
        return self._call("set_head", **p)

    def set_arm(self, q):
        return self._call("set_arm", q=[float(v) for v in q])

    def arm_to(self, p, pitch=-1.5707963267948966, roll=None, elbow_up=True, speed=0.15):
        params = {"p": [float(v) for v in p], "pitch": float(pitch), "elbow_up": bool(elbow_up), "speed": float(speed)}
        if roll is not None:
            params["roll"] = float(roll)
        return self._call("arm_to", **params)

    def gripper(self, width):
        return self._call("gripper", width=float(width))

    def step(self, n_ticks=1):
        return self._call("step", n_ticks=int(n_ticks))

    # ---- the mission
    def report_survivor(self, x, y, z=0.0):
        return self._call("report_survivor", x=float(x), y=float(y), z=float(z))

    def finish(self):
        return self._call("finish")

    # ---- privileged (the reference program's token only)
    def get_state_privileged(self):
        return self._call("get_state_privileged")

    def reference_run(self):
        return self._call("reference_run")

    # ---- the training ground only (the graded cell answers {"error": "forbidden"})
    def reset(self, site):
        """Start this connection's training mission on training site `site` (see the status file for the list). Once
        per connection; it uses one mission of the training ground's allowance."""
        return self._call("reset", site=str(site))

    def close(self):
        closer = getattr(self, "_close", None)
        if closer is not None:
            closer()


def _png(b64):
    import numpy as np
    from PIL import Image
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b64))))


def decode_frame(reply):
    """(rgb uint8 H x W x 3, depth float32 H x W in metres with NaN where the camera had no reading / None)."""
    import numpy as np
    rgb = _png(reply["rgb_png_b64"])[..., :3]
    depth = None
    if reply.get("depth_u16_mm_b64"):
        d = _png(reply["depth_u16_mm_b64"]).astype(np.float32)
        depth = d * float(reply.get("depth_unit_m", 0.001))
        depth[d == float(reply.get("depth_invalid_value", 0))] = np.nan
    return rgb, depth


def decode_lidar(reply):
    """float32 N x 4: x, y, z in the lidar frame (metres; axes of the chassis, origin at the lidar) and the channel
    index (0 = lowest beam). The lidar's pose in the chassis frame is reply["T_chassis_lidar"]."""
    import numpy as np
    return np.frombuffer(base64.b64decode(reply["points_f32_b64"]), dtype="<f4").reshape(-1, 4).copy()


def decode_thermal(reply):
    """float32 H x W temperatures in degrees Celsius (the radiometric image is centi-kelvin)."""
    import numpy as np
    ck = _png(reply["centikelvin_u16_png_b64"]).astype(np.float32)
    return ck / 100.0 - 273.15


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


def connect_training(site=None, path=None):
    """A connection to the training ground (a Robot): one mission, started by `reset(site)` (done here when `site`
    is given). The service runs a few missions at a time; the allowance left is in TRAINING_STATUS."""
    import socket
    path = path or TRAINING_SOCKET
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(path)
    except OSError as e:
        sock.close()
        raise CellError("the training ground is not reachable at %s (%s)" % (path, e))
    rf = sock.makefile("r", encoding="utf-8", newline="\n")
    wf = sock.makefile("w", encoding="utf-8", newline="\n")

    def send(text):
        try:
            wf.write(text)
            wf.flush()
        except OSError as e:
            raise CellError("the training ground closed the connection (%s)" % e)

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


def launcher_argv(python, lib_dir, program_dir, module="rescue"):
    """argv of the program: `bash run.sh` when the submission ships one (regular file), else the client launcher
    `python -B -E -s -c <bootstrap> <lib_dir> <module>`."""
    run_sh = os.path.join(program_dir, "run.sh")
    if os.path.isfile(run_sh) and not os.path.islink(run_sh):
        return ["bash", "run.sh"]
    return [python] + PYTHON_FLAGS + ["-c", BOOTSTRAP, lib_dir, module]


def main(argv=None):
    """Launcher: import rescue.py from the current directory and run solve(robot)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    module = argv[0] if argv else "rescue"
    robot = connect_stdio()
    sys.path.insert(0, os.getcwd())
    code = 0
    try:
        mod = __import__(module)
        getattr(mod, "solve")(robot)
    except CellError as e:
        print("[rescue] cell error: %s" % e, file=sys.stderr)
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
