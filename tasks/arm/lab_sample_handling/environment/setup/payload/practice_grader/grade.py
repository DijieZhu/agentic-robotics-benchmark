"""Practice evaluation in YOUR DEV SIM on the practice units (visible seeds; prints raw metrics only).

    /opt/venv/bin/python3 /home/user/practice_grader/grade.py [--units p0 p1 p2] [--episodes N] [--seed S ...]
        [--submission /home/user/submission]

Runs your controller (bash run.sh in --submission) against /home/user/lab_env.py with the practice units
(/home/user/units.json: each unit's rotor properties), the episodes and credit of /home/user/episode.py. It speaks the same wire as the hidden grader: READY within 120 s, each reply within
30 s, repeated step indices with perturbed observations (only the reply to the real observation is
applied), an aborted episode scores 0 and the controller is restarted (at most 3 restarts), and the
controller runs with only AF_UNIX sockets (the same seccomp filter as at grading time). It reports the
controller's own time (what it took to print READY and to answer each request, plus 0.2 s for every image
it asked for: the grading budget is 9000 s of it over 12 episodes; the simulation and the drawing of the
images are not counted) and whether the run stayed inside that budget. It never touches the hidden units, does not know the reference value and prints
no score. One thing it cannot show: it starts your controller in your own home directory, while the graded
run starts it in a fresh sandbox that holds only your submission folder (see the instruction).
"""

import argparse
import json
import math
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time

import numpy as np

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the agent home
sys.path.insert(0, HOME)
import episode as EP  # noqa: E402
import lab_env as E  # noqa: E402

PRACTICE_SEEDS = [0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0x2545F4914F6CDD1D,
                  0xD1B54A32D192ED03, 0x8CB92BA72F3D8DD7, 0xA0761D6478BD642F, 0xE7037ED1A0B428DB]
READY_TIMEOUT = 120.0
STEP_TIMEOUT = 30.0
MAX_RESTARTS = 3
MAX_LINE = 1 << 22
BUDGET_PER_EPISODE = 9000.0 / 12
IMAGE_COST = 0.2                 # s of the budget per image asked for, as in grading
PROBE_RATE = 0.01

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
    sys.stderr.write("[practice] could not install the network filter\n")
env = {"PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin", "HOME": "/home/user", "LANG": "C.UTF-8"}
os.execvpe("bash", ["bash", "run.sh"], env)
'''


class WireError(Exception):
    pass


class Wire(object):
    def __init__(self, submission):
        self.submission = submission
        self.restarts = 0
        self.dead = False
        self.broken = False
        self.controller_s = 0.0          # the controller's own time: READY and every reply
        self._start()

    def _start(self):
        t0 = time.monotonic()
        try:
            self._launch()
        finally:
            self.controller_s += time.monotonic() - t0

    def _launch(self):
        self.proc = subprocess.Popen([sys.executable, "-I", "-c", _FILTER_SRC], cwd=self.submission,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True,
                                     bufsize=1, start_new_session=True)
        self.lines = queue.Queue()
        threading.Thread(target=self._pump, args=(self.proc, self.lines), daemon=True).start()
        for _ in range(200):
            if self._read(READY_TIMEOUT, "READY").strip() == "READY":
                return
        raise WireError("solver never printed READY")

    @staticmethod
    def _pump(proc, lines):
        while True:
            try:
                line = proc.stdout.readline(MAX_LINE + 1)
            except (ValueError, OSError):
                break
            if not line or (len(line) > MAX_LINE and not line.endswith("\n")):
                break
            lines.put(line)
        lines.put(None)

    def _read(self, timeout, what):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise WireError("solver exceeded its {} s {} deadline".format(timeout, what))
        if line is None:
            raise WireError("solver closed its stdout (or wrote a line over the length cap)")
        return line

    def stop(self):
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            pass

    def request(self, msg):
        t0 = time.monotonic()
        try:
            return self._request(msg)
        finally:
            self.controller_s += time.monotonic() - t0

    def _request(self, msg):
        if self.dead:
            raise WireError("solver could not be restarted after an earlier abort")
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
            while True:
                line = self._read(STEP_TIMEOUT, "step").strip()
                if not line:
                    continue
                try:
                    reply = json.loads(line)
                except ValueError:
                    raise WireError("non-JSON output on the protocol stdout: {!r}".format(line[:80]))
                if not isinstance(reply, dict):
                    raise WireError("reply is not a JSON object")
                return reply
        except (OSError, ValueError) as exc:
            self.broken = True
            raise WireError(str(exc))
        except WireError:
            self.broken = True
            raise

    def reset(self):
        if self.broken:
            if self.restarts >= MAX_RESTARTS:
                self.dead = True
                raise WireError("restart limit reached; remaining episodes abort")
            self.restarts += 1
            self.stop()
            self.broken = False
            self._start()
        self.request({"type": "reset"})

    def act(self, t, obs):
        reply = self.request({"type": "act", "t": t, "obs": obs})
        if "action" not in reply:
            self.broken = True
            raise WireError("solver error: {}".format(reply.get("error", "no action")))
        return reply["action"]

    def close(self):
        try:
            self.request({"type": "close"})
        except Exception:  # noqa: BLE001
            pass
        self.stop()


def differ(a, b):
    try:
        aa, bb = [float(v) for v in a], [float(v) for v in b]
    except (TypeError, ValueError):
        return False
    return len(aa) != len(bb) or max([abs(x - y) for x, y in zip(aa, bb)] or [0.0]) > 1e-4


def still(action):
    """An 8-number reply whose six motion numbers are all within 1e-9 of zero: it commands no motion."""
    if not isinstance(action, list) or len(action) != 8:
        return False
    for x in action[:6]:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > 1e-9:
            return False
    return True


def valid_action(action):
    if not isinstance(action, list) or len(action) not in (8, 9, 15):
        return False
    for x in action:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            return False
    cams = action[8] if len(action) == 9 else action[7]
    return 0 <= cams <= 15 and float(cams).is_integer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="/home/user/submission")
    ap.add_argument("--units", nargs="*", default=None, help="practice unit ids (default: all)")
    ap.add_argument("--units-file", default=os.path.join(HOME, "units.json"))
    ap.add_argument("--episodes", type=int, default=2, help="episodes per unit")
    ap.add_argument("--seed", type=int, action="append", help="run these seeds instead of the practice list")
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.submission, "run.sh")):
        print(json.dumps({"error": "no run.sh in {}".format(a.submission)}))
        return 2
    with open(a.units_file) as fh:
        table = json.load(fh)["units"]
    ids = a.units or sorted(table)
    seeds = a.seed if a.seed else PRACTICE_SEEDS[:max(1, min(a.episodes, len(PRACTICE_SEEDS)))]
    rng = random.Random(2026)
    pairs = differed = idle = 0
    results = []
    t0 = time.monotonic()
    try:
        wire = Wire(a.submission)
    except WireError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    try:
        for uid in ids:
            for seed in seeds:
                ep = EP.Episode(E.LabEnv(table[uid], uid, render=True), seed)
                try:
                    obs = ep.reset()
                    wire.reset()
                    t, done = 0, False
                    while not done:
                        wobs = E.to_wire(obs)
                        wobs["unit"] = uid
                        if rng.random() < PROBE_RATE:
                            other = E.to_wire(ep.env.perturbed_observation(
                                obs, np.random.default_rng(rng.getrandbits(64))))
                            other["unit"] = uid
                            if rng.random() < 0.5:
                                b = wire.act(t, other)
                                act = wire.act(t, wobs)
                            else:
                                act = wire.act(t, wobs)
                                b = wire.act(t, other)
                            if still(act) and still(b):
                                idle += 1          # both command no motion: not counted, as in grading
                            else:
                                pairs += 1
                                differed += int(differ(act, b))
                        else:
                            act = wire.act(t, wobs)
                        if not valid_action(act):
                            raise WireError("malformed action: 8, 15 or 9 finite numbers, the camera bits an integer 0-15")
                        wire.controller_s += IMAGE_COST * bin(int(act[8] if len(act) == 9 else act[7])).count("1")
                        obs, done, _info = ep.step(act)
                        t += 1
                        if ep.env.unstable():
                            raise WireError("simulation unstable (an arm joint over 8 rad/s)")
                    sc = ep.score()
                    results.append({"unit": uid, "seed": seed, "episode_credit": round(sc["credit"], 5), "steps": t,
                                    **{k: sc[k] for k in ("cold", "pairs", "need", "lid", "moved", "in_hole")}})
                except (WireError, ValueError, TypeError) as exc:
                    results.append({"unit": uid, "seed": seed, "episode_credit": 0.0, "aborted": str(exc)[:200]})
                finally:
                    ep.env.close()
    finally:
        wire.close()
    wall = time.monotonic() - t0
    mean = sum(r["episode_credit"] for r in results) / len(results) if results else 0.0
    print(json.dumps({
        "metric": "mean_episode_credit (dev sim, practice units {})".format(" ".join(ids)),
        "raw_value": round(mean, 5),
        "episodes": results,
        "repeated_t_pairs": pairs,
        "repeated_t_pairs_that_differed": differed,
        "repeated_t_pairs_not_counted_no_motion": idle,
        "wall_seconds": round(wall, 1),
        "controller_seconds": round(wire.controller_s, 1),
        "within_grading_time_budget": wire.controller_s <= BUDGET_PER_EPISODE * max(1, len(results)),
        "note": "dev sim with the practice units; this is not your score",
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
