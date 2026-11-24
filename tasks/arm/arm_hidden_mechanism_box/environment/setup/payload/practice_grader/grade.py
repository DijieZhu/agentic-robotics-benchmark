"""Practice evaluation in YOUR DEV SIM on the practice box (visible seeds; prints raw metrics only).

    /opt/venv/bin/python3 /home/user/practice_grader/grade.py [--episodes N] [--seed S ...]
        [--box /home/user/practice_box.json] [--submission /home/user/submission]

Runs your controller (bash run.sh in --submission) against /home/user/box_env.py with the practice
box (or any box spec file you pass with --box, e.g. a replica you built). It speaks the same wire
as the hidden grader: READY within 120 s, each reply within 30 s, repeated step indices with
perturbed observations (only the reply to the real observation is applied), an aborted episode
scores 0 and the controller is restarted (at most 3 restarts), replies longer than 64 KiB abort the
episode, and the controller runs with only AF_UNIX sockets (the same seccomp filter as at grading
time). It reports whether the run stayed inside the grading time budget (6000 s for 8 episodes).
It never touches the hidden boxes, does not know the reference value and prints no score.
"""

import argparse
import fcntl
import json
import math
import os
import queue
import random
import select
import signal
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the agent home
import box_env as E  # noqa: E402

PRACTICE_SEEDS = [0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0x2545F4914F6CDD1D,
                  0xD1B54A32D192ED03, 0x8CB92BA72F3D8DD7, 0xA0761D6478BD642F, 0xE7037ED1A0B428DB]
READY_TIMEOUT = 120.0
STEP_TIMEOUT = 30.0
MAX_RESTARTS = 3
MAX_LINE = 65536
BUDGET_PER_EPISODE = 6000.0 / 8
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
        self._start()

    def _start(self):
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

    def _write(self, text, deadline):
        """As the grader writes a request: without blocking, a chunk at a time as the pipe drains. A request the
        solver has not read by the deadline ends the episode like a late reply."""
        data = memoryview(text.encode("utf-8"))
        self.proc.stdin.flush()
        fd = self.proc.stdin.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while data:
                left = deadline - time.monotonic()
                if left <= 0 or not select.select([], [fd], [], left)[1]:
                    raise WireError("solver exceeded its {} s step deadline (it did not read the request)".format(
                        STEP_TIMEOUT))
                try:
                    data = data[os.write(fd, data[:65536]):]
                except BlockingIOError:
                    continue
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    def _read(self, timeout, what, budget=None):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise WireError("solver exceeded its {} s {} deadline".format(timeout if budget is None else budget, what))
        if line is None:
            raise WireError("solver closed its stdout (or wrote a line over 64 KiB)")
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
        if self.dead:
            raise WireError("solver could not be restarted after an earlier abort")
        try:
            # One deadline for the request, fixed before it is written, as in grading: the time the solver takes
            # to read the request counts, and every line it writes back is read against what is left.
            deadline = time.monotonic() + STEP_TIMEOUT
            self._write(json.dumps(msg) + "\n", deadline)
            while True:
                line = self._read(max(0.0, deadline - time.monotonic()), "step", STEP_TIMEOUT).strip()
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
    # the grader's rule: two replies with a different count of numbers show nothing
    return len(aa) == len(bb) and max([abs(x - y) for x, y in zip(aa, bb)] or [0.0]) > 1e-4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", default="/home/user/submission")
    ap.add_argument("--box", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "practice_box.json"))
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, action="append", help="run these seeds instead of the practice list")
    a = ap.parse_args()
    if not os.path.isfile(os.path.join(a.submission, "run.sh")):
        print(json.dumps({"error": "no run.sh in {}".format(a.submission)}))
        return 2
    with open(a.box) as fh:
        spec = json.load(fh)
    seeds = a.seed if a.seed else PRACTICE_SEEDS[:max(1, min(a.episodes, len(PRACTICE_SEEDS)))]
    rng = random.Random(2026)
    pairs = differed = 0
    results = []
    t0 = time.monotonic()
    try:
        wire = Wire(a.submission)
    except WireError as exc:
        print(json.dumps({"error": str(exc)}))
        return 1
    try:
        for seed in seeds:
            env = E.BoxEnv(spec, box_id=spec.get("id", "practice"), early_end=True)
            try:
                obs = env.reset(seed)
                wire.reset()
                for t in range(env.max_steps):
                    wobs = E.to_wire(obs)
                    if rng.random() < PROBE_RATE:
                        other = E.to_wire(env.perturbed_observation(obs, rng))
                        if rng.random() < 0.5:
                            b = wire.act(t, other)
                            act = wire.act(t, wobs)
                        else:
                            act = wire.act(t, wobs)
                            b = wire.act(t, other)
                        pairs += 1
                        differed += int(differ(act, b))
                    else:
                        act = wire.act(t, wobs)
                    obs, done, info = env.step(act)
                    if done:
                        break
                results.append({"seed": seed, "episode_score": round(env.episode_score(), 5), "steps": env.t,
                                "stages": [{"id": s["id"], "kind": s["kind"], "unlocked": bool(ok)}
                                           for (s, _q), ok in zip(env.stages, env.achieved)],
                                "sim_fault": env.fault})
            except (WireError, E.ActionError, ValueError, TypeError) as exc:
                results.append({"seed": seed, "episode_score": 0.0, "aborted": str(exc)[:200]})
            finally:
                env.close()
    finally:
        wire.close()
    wall = time.monotonic() - t0
    mean = sum(r["episode_score"] for r in results) / len(results) if results else 0.0
    print(json.dumps({
        "metric": "mean_opening_progress (dev sim, {})".format(os.path.basename(a.box)),
        "raw_value": round(mean, 5),
        "episodes": results,
        "repeated_t_pairs": pairs,
        "repeated_t_pairs_that_differed": differed,
        "wall_seconds": round(wall, 1),
        "within_grading_time_budget": wall <= BUDGET_PER_EPISODE * max(1, len(results)),
        "note": "dev sim with the given box; this is not your score",
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
