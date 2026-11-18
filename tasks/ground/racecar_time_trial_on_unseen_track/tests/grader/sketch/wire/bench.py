"""Round trip of the wire with a full scan (the cell's gate: at most 0.3 ms), measured from the program's side.

    python -m sketch.wire.bench [--n 3000] [--real] [--out results.json]

`echo`: a cell that answers every `step` with the same full reply (1081 real ranges cast in the greybox model of
the example venue, five IMU rows, events) and does nothing else, so the program's time per call is the wire alone.
`real`: the cell's venue with the stand-ins of the car, sensor, venue, duct and obstacle models; the cell's own handling time of each request (measured by the
server) is subtracted from the program's time of the same call. Both report, per call, `rpc` = request encode +
pipe + server decode + reply encode + pipe + readline, and `call` = rpc + the client's JSON decode and numpy
conversion (what `car.step` costs a program beyond the simulation).
"""

import argparse
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np

from carkit.client import launcher_argv

from .server import serve

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(os.path.dirname(HERE))

PROGRAM = '''import json, sys, time
N = %d
def solve(car):
    car.open_session("practice")
    rpc = car._rpc
    tr = []
    def timed(line):
        t0 = time.perf_counter()
        out = rpc(line)
        tr.append(time.perf_counter() - t0)
        return out
    car._rpc = timed
    tc = []
    for k in range(N):
        t0 = time.perf_counter()
        r = car.step(0.0, 0.0)
        tc.append(time.perf_counter() - t0)
    print("BENCH " + json.dumps({"rpc": tr, "call": tc, "len": len(r["scan"])}), file=sys.stderr)
'''


class EchoVenue(object):
    """Just enough of the venue's interface for `serve`: every step returns the same full observation."""

    def __init__(self, reply, cfg, params):
        self.reply = reply
        self.cfg = cfg
        self.params = params
        self.caps = {"cpu": 1e9}
        self.counts = {"forbidden_calls": 0, "bad_calls": 0}
        self.state = "idle"
        self.end_reason = None
        self.tick = 0

    @property
    def over(self):
        return self.state == "over"

    def handle(self, method, params):
        if method == "open_session":
            self.state = "open"
            return {"session": "practice", "duration_s": 480.0, "ticks": 19200, "phases": []}
        if method == "step":
            self.tick += 1
            r = dict(self.reply)
            r["tick"] = self.tick
            return r
        return {"error": "forbidden"}

    def account(self, dt):
        return None

    def budget_left(self):
        return math.inf

    def end(self, reason):
        if self.end_reason is None:
            self.end_reason = reason
        self.state = "over"


def full_reply():
    """A step result with a real scan of the example venue at its start pose."""
    import mujoco
    from sketch.examples import check, greybox
    w = check.load_json(os.path.join(DEV, "sketch", "examples", "min_world.json"))
    model = greybox.build_model(w, variant="track", cars={"car/": w["truth"]["start_pose"]["pose"]})
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scan = greybox.to_wire_mm(greybox.ideal_scan(model, data)[0])
    imu = [[0.412, 0.003, 9.806, 0.0002, -0.0011, 0.0031]] * 5
    return {"tick": 1, "scan": scan, "imu": imu, "wheel_speed": 2.315, "steer_echo": 0.0523, "events": []}


def _summary(xs):
    a = np.asarray(xs) * 1e3
    return {"n": int(len(a)), "p25_ms": round(float(np.percentile(a, 25)), 4), "median_ms": round(float(np.median(a)), 4),
            "p90_ms": round(float(np.percentile(a, 90)), 4), "mean_ms": round(float(a.mean()), 4)}


def measure(n=3000, real=False, warm=100):
    from sketch.referee import Venue, load_config, load_params
    cfg, params = load_config(), load_params()
    if real:
        from sketch.examples import check
        venue = Venue(check.load_json(os.path.join(DEV, "sketch", "examples", "min_world.json")), cfg, providers="stub")
    else:
        venue = EchoVenue(full_reply(), cfg, params)
    work = tempfile.mkdtemp(prefix="s6bench_")
    with open(os.path.join(work, "drive.py"), "w", encoding="utf-8") as f:
        f.write(PROGRAM % n)
    errp = os.path.join(work, "stderr.log")
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": work, "TMPDIR": work, "LANG": "C.UTF-8",
           "PYTHONPATH": DEV, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    st = {}
    try:
        serve(venue, launcher_argv(sys.executable, DEV, "drive"), cwd=work, env=env, stderr_path=errp, stats=st)
        with open(errp, "r", encoding="utf-8") as f:
            line = next(ln for ln in f if ln.startswith("BENCH "))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    prog = json.loads(line[6:])
    handle = st["handle_times"][1:1 + n]                    # request 1 opened the session
    rpc = [a - b for a, b in zip(prog["rpc"], handle)][warm:]
    call = [a - b for a, b in zip(prog["call"], handle)][warm:]
    reply_bytes = len(json.dumps({"id": 1, "result": venue.reply if not real else full_reply()}, separators=(",", ":")))
    return {"mode": "real" if real else "echo", "n": n, "warm": warm, "reply_bytes": reply_bytes,
            "rpc_minus_cell": _summary(rpc), "call_minus_cell": _summary(call),
            "cell_handle": _summary(handle[warm:]), "load1": os.getloadavg()[0]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    res = measure(a.n, a.real)
    text = json.dumps(res, indent=1)
    print(text)
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
