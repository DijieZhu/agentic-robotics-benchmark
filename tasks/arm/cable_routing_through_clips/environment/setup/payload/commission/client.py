"""Client of the commissioning service: development boards you may work on before you hand in.

    from commission.client import Bench
    b = Bench()                       # one connection = one board; at most two at a time
    info = b.status()                 # the window: steps_left, reset_steps, image_steps, boards
    r = b.reset(0)                    # stage development board 0 (of 0..2); r["obs"] is the first observation
    r = b.step(action)                # r["obs"], r["done"], r["ended"] (why the episode ended), r["steps_left"]
    r = b.look(3)                     # the cameras now (bitmask as in an action), without a step
    r = b.probe()                     # the other copy of the last observation (what a repeated request carries)
    b.close()

The observations are the ones your controller gets when it is graded. The service tells nothing else: no fixture
as mounted, no cable point, no tension, no credit and no word on whether a fixture holds the cable. After an
episode has ended (your hand-over or the step limit) the cell has opened the hand, lifted it and let the cable
settle; `look` then shows the settled board. `BenchError.code` is one of: busy, bad_params, no_episode,
malformed_action, commissioning_window_used, bench_fault, forbidden, closed.
"""

import base64
import json
import socket

SOCK = "/run/harnessbench/commission.sock"
STATUS = "/run/harnessbench/commission_status.json"


class BenchError(Exception):
    def __init__(self, code, detail=""):
        Exception.__init__(self, "{}: {}".format(code, detail))
        self.code, self.detail = code, detail


def decode_image(d):
    """A wire image {"w", "h", "png"} as PNG bytes; with OpenCV:
    cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR) is the picture as BGR, (h, w, 3)."""
    return base64.b64decode(d["png"])


class Bench(object):
    def __init__(self, path=SOCK, timeout=600.0):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect(path)
        self.f = self.sock.makefile("rwb")
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        try:
            self.f.write((json.dumps({"id": self.n, "method": method, "params": params}) + "\n").encode("utf-8"))
            self.f.flush()
            line = self.f.readline()
        except (BrokenPipeError, ConnectionError):
            line = b""
        if not line:
            raise BenchError("closed", "the service closed the connection (120 s of silence end a bench)")
        msg = json.loads(line.decode("utf-8"))
        if "error" in msg:
            raise BenchError(msg["error"], msg.get("detail") or "")
        return msg["result"]

    def status(self):
        return self.call("status")

    def reset(self, board=0):
        return self.call("reset", board=int(board))

    def step(self, action):
        return self.call("step", action=[float(x) for x in action[:7]] + [int(x) for x in action[7:]])

    def look(self, cams=3):
        return self.call("look", cams=int(cams))

    def probe(self):
        return self.call("probe")

    def close(self):
        try:
            self.call("close")
        except (BenchError, OSError, ValueError):
            pass
        try:
            self.sock.close()
        except OSError:
            pass
