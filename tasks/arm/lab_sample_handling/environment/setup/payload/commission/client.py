"""Client of the commissioning service: development benches you may work on before you hand in.

    from commission.client import Bench
    b = Bench()                       # one connection = one bench; at most two at a time
    info = b.status()                 # the window: steps_left, reset_steps, units, ...
    r = b.reset("d0", 0)              # stage worklist 0 (of 0..4999) on unit d0; r["obs"] is the first observation
    r = b.step(action)                # r["obs"], r["done"], r["ended"] (why the episode ended), r["steps_left"]
    r = b.look(15)                    # the cameras now (bitmask as in an action), without a step
    r = b.probe()                     # a perturbed copy of the last observation (what a repeated request carries)
    b.close()

The observations are the ones your controller gets when it is graded (`unit` names the development unit). The service
tells nothing else: no pose, no mass, no rotor angle, no unit property, no credit and no word on how a worklist went.
`BenchError.code` is one of: busy, bad_params, no_episode, malformed_action, commissioning_window_used, bench_fault,
forbidden, closed.
"""

import base64
import json
import socket

SOCK = "/run/labbench/commission.sock"
STATUS = "/run/labbench/commission_status.json"


class BenchError(Exception):
    def __init__(self, code, detail=""):
        Exception.__init__(self, "{}: {}".format(code, detail))
        self.code, self.detail = code, detail


def decode_image(d):
    """A wire image {"dtype", "shape", "b64"} as bytes and its shape (H, W, 3); with NumPy:
    np.frombuffer(raw, np.uint8).reshape(shape)."""
    return base64.b64decode(d["b64"]), tuple(d["shape"])


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

    def reset(self, unit, episode=0):
        return self.call("reset", unit=unit, episode=int(episode))

    def step(self, action):
        return self.call("step", action=[float(x) for x in action])

    def look(self, cams=15):
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
