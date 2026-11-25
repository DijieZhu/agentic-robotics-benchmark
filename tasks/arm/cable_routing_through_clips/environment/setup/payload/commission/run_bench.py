"""Run your controller on a commissioning board the way the grader runs it.

    /opt/venv/bin/python3 /home/user/commission/run_bench.py --board 0
        [--submission /home/user/submission] [--max-steps N] [--images DIR [--every N]]

It copies the submission folder (regular files) to an empty temporary home, starts `bash run.sh` there with only
PATH, HOME and LANG set and only AF_UNIX sockets (the grader's filter), and speaks the grading wire: READY within
120 s, `reset`, then `act` requests with the board's observations, each reply within 30 s; about 1 % of steps are
asked twice, once with the other copy of the observation, and only the reply to the real observation is applied.
The board is one of the commissioning service's development boards; every step and every image uses bench time of
your window.

It prints what the grader would know about your controller, and nothing about the route: how the episode ended
(the step limit, your hand-over, an unstable simulation, a controller fault), the steps, the controller's time (its
replies plus 0.2 s per image asked for, against the grading budget's 180 s per episode), the repeated-step check.
How the route went you judge from what the cameras show: `--images DIR` saves both cameras' pictures at the start,
every `--every` steps (default: never) and at the end, after the cell has opened the hand and the cable has
settled. One difference to grading remains: there the controller's home holds the submission folder and nothing
else, here your real home still exists under its own path.
"""

import argparse
import json
import math
import os
import queue
import random
import shutil
import subprocess
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from commission.client import Bench, BenchError, decode_image  # noqa: E402

READY_TIMEOUT, STEP_TIMEOUT, MAX_LINE = 120.0, 30.0, 1 << 22
PROBE_RATE, IMAGE_COST, BUDGET_PER_EPISODE = 0.01, 0.2, 3600.0 / 20
CAMS = (("overhead", 1), ("wrist", 2))

_FILTER_SRC = r'''
import ctypes, os, platform, struct, sys
ARCH = {"x86_64": (0xC000003E, 41, 53, 425, 0x40000000), "aarch64": (0xC00000B7, 198, 199, 425, 0)}
def ins(c, jt, jf, k): return struct.pack("HBBI", c, jt, jf, k)
audit, ns, nsp, nu, x32 = ARCH[platform.machine()]
deny = 0x00050000 | 1; allow = 0x7FFF0000
p = [ins(0x20,0,0,4), ins(0x15,1,0,audit), ins(0x06,0,0,deny), ins(0x20,0,0,0)]
if x32: p += [ins(0x35,0,1,x32), ins(0x06,0,0,deny)]
p += [ins(0x15,0,1,nu), ins(0x06,0,0,deny), ins(0x15,2,0,ns), ins(0x15,1,0,nsp), ins(0x06,0,0,allow),
      ins(0x20,0,0,16), ins(0x15,0,1,1), ins(0x06,0,0,allow), ins(0x06,0,0,deny)]
prog = b"".join(p); buf = ctypes.create_string_buffer(prog, len(prog))
class F(ctypes.Structure): _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
f = F(len(prog) // 8, ctypes.cast(buf, ctypes.c_void_p)); libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(22, 2, ctypes.byref(f), 0, 0) != 0:
    sys.stderr.write("[bench] could not install the network filter\n")
env = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/home/user", "LANG": "C.UTF-8"}
os.execvpe("bash", ["bash", "run.sh"], env)
'''


class WireError(Exception):
    pass


class Controller(object):
    def __init__(self, submission):
        self.home = tempfile.mkdtemp(prefix="bench-home-")
        self.dir = os.path.join(self.home, "submission")
        shutil.copytree(submission, self.dir, symlinks=True)
        self.seconds = 0.0
        t0 = time.monotonic()
        self.proc = subprocess.Popen([sys.executable, "-I", "-c", _FILTER_SRC.replace("/home/user", self.home)],
                                     cwd=self.dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     universal_newlines=True, bufsize=1, start_new_session=True)
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        try:
            for _ in range(200):
                if self._read(READY_TIMEOUT - (time.monotonic() - t0), "READY").strip() == "READY":
                    return
            raise WireError("the controller never printed READY")
        finally:
            self.seconds += time.monotonic() - t0

    def _pump(self):
        while True:
            try:
                line = self.proc.stdout.readline(MAX_LINE + 1)
            except (ValueError, OSError):
                break
            if not line or (len(line) > MAX_LINE and not line.endswith("\n")):
                break
            self.lines.put(line)
        self.lines.put(None)

    def _read(self, timeout, what):
        try:
            line = self.lines.get(timeout=max(0.0, timeout))
        except queue.Empty:
            raise WireError("the controller exceeded its {} deadline".format(what))
        if line is None:
            raise WireError("the controller closed its stdout (or wrote a line over the length cap)")
        return line

    def request(self, msg):
        t0 = time.monotonic()
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self._read(STEP_TIMEOUT - (time.monotonic() - t0), "30 s reply").strip()
                if line:
                    break
            reply = json.loads(line)
            if not isinstance(reply, dict):
                raise WireError("the reply is not a JSON object")
            return reply
        except (OSError, ValueError) as exc:
            raise WireError(str(exc)[:200])
        finally:
            self.seconds += time.monotonic() - t0

    def act(self, t, obs):
        reply = self.request({"type": "act", "t": t, "obs": obs})
        if "action" not in reply:
            raise WireError("the reply has no action")
        return reply["action"]

    def stop(self):
        try:
            self.request({"type": "close"})
        except WireError:
            pass
        try:
            os.killpg(self.proc.pid, 9)
        except OSError:
            pass
        shutil.rmtree(self.home, ignore_errors=True)


def valid_action(action):
    if not isinstance(action, list) or len(action) not in (8, 9):
        return False
    for x in action:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            return False
    if not (0 <= action[7] <= 3 and float(action[7]).is_integer()):
        return False
    return len(action) == 8 or action[8] in (0, 1)


def still(action):
    return isinstance(action, list) and len(action) in (8, 9) and all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and abs(x) <= 1e-9 for x in action[:6])


def differ(a, b):
    try:
        aa, bb = [float(v) for v in a], [float(v) for v in b]
    except (TypeError, ValueError):
        return False
    return len(aa) != len(bb) or max([abs(x - y) for x, y in zip(aa, bb)] or [0.0]) > 1e-6


def save_images(bench, folder, tag):
    obs = bench.look(3)["obs"]
    os.makedirs(folder, exist_ok=True)
    for cam, _bit in CAMS:
        with open(os.path.join(folder, "{}_{}.png".format(tag, cam)), "wb") as f:
            f.write(decode_image(obs[cam]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=int, default=0)
    ap.add_argument("--submission", default="/home/user/submission")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0: run to the episode's end)")
    ap.add_argument("--images", default=None)
    ap.add_argument("--every", type=int, default=0)
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.submission, "run.sh")):
        print(json.dumps({"error": "no run.sh in {}".format(a.submission)}))
        return 2
    out = {"board": a.board}
    rng = random.Random()
    pairs = differed = idle = images = steps = 0
    ended = None
    try:
        bench = Bench()
    except OSError as exc:
        print(json.dumps({"error": "the commissioning service is not reachable: {}".format(exc)}))
        return 1
    ctl = None
    try:
        r = bench.reset(a.board)
        obs = r["obs"]
        if a.images:
            save_images(bench, a.images, "start")
        ctl = Controller(a.submission)
        ctl.request({"type": "reset"})
        done, t = False, 0
        while not done:
            if rng.random() < PROBE_RATE:
                other = bench.probe()["obs"]
                if rng.random() < 0.5:
                    b = ctl.act(t, other)
                    act = ctl.act(t, obs)
                else:
                    act = ctl.act(t, obs)
                    b = ctl.act(t, other)
                if still(act) and still(b):
                    idle += 1
                else:
                    pairs += 1
                    differed += int(differ(act, b))
            else:
                act = ctl.act(t, obs)
            if not valid_action(act):
                raise WireError("malformed action: 8 or 9 finite numbers, the camera bits an integer 0-3, the ninth 0 or 1")
            images += bin(int(act[7])).count("1")
            r = bench.step(act)
            obs, done, ended = r["obs"], r["done"], r["ended"]
            t += 1
            steps = t
            if a.images and a.every and t % a.every == 0:
                save_images(bench, a.images, "step{:05d}".format(t))
            if a.max_steps and t >= a.max_steps and not done:
                ended, done = "stopped by --max-steps", True
        if a.images:
            save_images(bench, a.images, "end")
    except WireError as exc:
        ended = "controller fault (the graded cell would judge the cable as it lies at that moment): {}".format(exc)
    except BenchError as exc:
        ended = "bench: {}".format(exc)
    finally:
        if ctl is not None:
            ctl.stop()
        try:
            out["steps_left_in_window"] = bench.status()["steps_left"]
        except (BenchError, OSError, ValueError):
            pass
        bench.close()
    seconds = (ctl.seconds if ctl is not None else 0.0) + IMAGE_COST * images
    out.update({"steps": steps, "ended": ended, "controller_seconds": round(seconds, 1), "images_asked": images,
                "within_grading_time_budget": seconds <= BUDGET_PER_EPISODE,
                "repeated_t_pairs": pairs, "repeated_t_pairs_that_differed": differed,
                "repeated_t_pairs_not_counted_no_motion": idle,
                "note": "a development board; nothing here says how the route went"})
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
