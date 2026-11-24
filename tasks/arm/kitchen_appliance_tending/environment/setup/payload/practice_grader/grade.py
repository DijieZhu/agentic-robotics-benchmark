"""Practice evaluation in YOUR DEV SIM on the practice units (visible seeds; prints raw metrics only).

    /opt/venv/bin/python3 /home/user/practice_grader/grade.py [--units p0 p1 p2] [--episodes N] [--seed S ...]
        [--submission /home/user/submission]

Runs your controller (bash run.sh in --submission) against /home/user/appliance_env.py with the practice
units (/home/user/units.json; models in /home/user/units/), the episodes and credit of
/home/user/episode.py. It speaks the same wire as the hidden grader: READY, each reply within 30 s,
repeated step indices with perturbed observations (only the reply to the real observation is applied; a
pair in which both replies command no motion is not counted, and fewer than 10 % of the counted pairs
differing would fail the hidden grader's check), an aborted episode scores 0 and the controller is
restarted (at most 3 restarts), and the controller runs with only AF_UNIX sockets (the same seccomp filter
as at grading time). A request has ONE deadline of 30 s, fixed before it is written: the write (an observation
with images is larger than the pipe's buffer), blank lines and the reply are all inside it, as in the hidden
grader. One difference: here each line printed before READY gets its own 120 s (at most 200 lines), while the
hidden grader allows 120 s for the whole start.

Time: it counts your controller's own time as the hidden grader does (`controller_seconds`: what it waits for
READY, for every reset and for every reply, plus 0.03 s for every image an applied action asks for; the simulation
and the drawing are not counted) and reports whether that stayed inside the grading budget for the episodes run
(6000 s for 18 episodes, so 333.3 s an episode here).

Two things the hidden grader does that this one cannot: it stops your controller's processes between requests
(they run only while a request is outstanding) and aborts an episode with more than 512 processes of your account
at such a pause; and it has only your submission folder, nothing else of this home. `paths_outside_submission`
lists the `/home/user/...` paths outside the submission that the text files of your submission name: none of
them exists when your controller is graded.

It never touches the hidden units, does not know the reference value and prints no score. A failure of the cameras' renderer (appliance_env.RenderError) is the environment's error, as in the
hidden grader: the run stops with an error that says so (exit status 3), never as an aborted episode.
"""

import argparse
import fcntl
import json
import math
import os
import queue
import random
import re
import select
import signal
import subprocess
import sys
import threading
import time

import numpy as np

HOME = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the agent home
sys.path.insert(0, HOME)
import appliance_env as E  # noqa: E402
import episode as EP  # noqa: E402

PRACTICE_SEEDS = [0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB, 0x2545F4914F6CDD1D,
                  0xD1B54A32D192ED03, 0x8CB92BA72F3D8DD7, 0xA0761D6478BD642F, 0xE7037ED1A0B428DB]
READY_TIMEOUT = 120.0
STEP_TIMEOUT = 30.0
MAX_RESTARTS = 3
MAX_LINE = 1 << 22
BUDGET_PER_EPISODE = 6000.0 / 18   # the controller's own time, the images' price included
IMAGE_COST = 0.03                  # s charged for every image an applied action asks for
PROBE_RATE = 0.01
MAX_BLANK_LINES = 100
SCAN_SUFFIXES = (".py", ".sh", ".json", ".txt", ".yaml", ".yml", ".cfg", ".toml", ".ini")

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
        self.seconds = 0.0           # the controller's own time: every wait for READY, a reset or a reply
        t0 = time.monotonic()
        try:
            self._start()
        finally:
            self.seconds += time.monotonic() - t0

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

    def _read(self, timeout, what):
        try:
            line = self.lines.get(timeout=max(0.0, timeout))
        except queue.Empty:
            raise WireError("solver exceeded its {} deadline".format(what))
        if line is None:
            raise WireError("solver closed its stdout (or wrote a line over the length cap)")
        return line

    def _send(self, text, deadline):
        """Write one line without blocking, giving up at `deadline`: a controller that does not read its request
        misses the deadline (the hidden grader's rule)."""
        fd = self.proc.stdin.fileno()
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        view = memoryview(text.encode("utf-8"))
        try:
            while len(view):
                try:
                    view = view[os.write(fd, view):]
                    continue
                except BlockingIOError:
                    pass
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise WireError("solver exceeded its {} s step deadline (it did not read the request)".format(
                        STEP_TIMEOUT))
                select.select([], [fd], [], min(remaining, 1.0))
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, flags)

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
        t0 = time.monotonic()
        deadline = t0 + STEP_TIMEOUT       # one deadline per request, fixed before it is written
        try:
            self._send(json.dumps(msg) + "\n", deadline)
            for _ in range(MAX_BLANK_LINES):
                line = self._read(deadline - time.monotonic(), "{} s step".format(STEP_TIMEOUT)).strip()
                if not line:
                    continue
                try:
                    reply = json.loads(line)
                except ValueError:
                    raise WireError("non-JSON output on the protocol stdout: {!r}".format(line[:80]))
                if not isinstance(reply, dict):
                    raise WireError("reply is not a JSON object")
                return reply
            raise WireError("{} blank lines to one request without a reply".format(MAX_BLANK_LINES))
        except (OSError, ValueError) as exc:
            self.broken = True
            raise WireError(str(exc))
        except WireError:
            self.broken = True
            raise
        finally:
            self.seconds += time.monotonic() - t0

    def reset(self):
        if self.broken:
            if self.restarts >= MAX_RESTARTS:
                self.dead = True
                raise WireError("restart limit reached; remaining episodes abort")
            self.restarts += 1
            self.stop()
            self.broken = False
            t0 = time.monotonic()
            try:
                self._start()
            finally:
                self.seconds += time.monotonic() - t0
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
    """The hidden grader's rule for a repeated-step pair: the two replies differ when they hold the same count of
    numbers and some number differs by more than 1e-4. A pair of replies of different lengths shows nothing: it is
    counted and does not count as differing."""
    try:
        aa, bb = [float(v) for v in a], [float(v) for v in b]
    except (TypeError, ValueError):
        return False
    return len(aa) == len(bb) and max([abs(x - y) for x, y in zip(aa, bb)] or [0.0]) > 1e-4


def outside_paths(submission, limit=1 << 20):
    """The /home/user/... paths outside the submission that its text files name (sorted, at most 20)."""
    found = set()
    sub = os.path.realpath(submission)
    pat = re.compile(r"/home/user(?:/[A-Za-z0-9_.\-]+)+")
    for dp, _dns, fns in os.walk(sub):
        for fn in fns:
            fp = os.path.join(dp, fn)
            if not fn.endswith(SCAN_SUFFIXES) or os.path.islink(fp):
                continue
            try:
                if os.path.getsize(fp) > limit:
                    continue
                with open(fp, errors="replace") as fh:
                    text = fh.read()
            except OSError:
                continue
            for m in pat.findall(text):
                if not (m == "/home/user/submission" or m.startswith("/home/user/submission/")):
                    found.add(m)
    return sorted(found)[:20]


def still(action):
    """The hidden grader's rule: a reply whose first six numbers are all within 1e-9 of zero commands no motion;
    a pair of two such replies is not counted."""
    if not isinstance(action, list) or len(action) < 6:
        return False
    for x in action[:6]:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or abs(x) > 1e-9:
            return False
    return True


def valid_action(action):
    if not isinstance(action, list) or len(action) != 8:
        return False
    for x in action:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            return False
    return 0 <= action[7] <= 7 and float(action[7]).is_integer()


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
        table = json.load(fh)
    base = os.path.dirname(os.path.abspath(a.units_file))
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
            c = table[uid]
            spec = E.unit_spec(c["model"], files=E.read_unit_files(os.path.join(base, c["dir"])), quirks=c.get("quirks"))
            for seed in seeds:
                ep = EP.Episode(E.ApplianceEnv(spec, None, render=True), seed)
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
                                idle += 1
                            else:
                                pairs += 1
                                differed += int(differ(act, b))
                        else:
                            act = wire.act(t, wobs)
                        if not valid_action(act):
                            raise WireError("malformed action: 8 finite numbers, the last an integer 0-7")
                        wire.seconds += IMAGE_COST * bin(int(act[7])).count("1")   # the images it asks for
                        obs, done, _info = ep.step(act)
                        t += 1
                        if ep.env.unstable():
                            raise WireError("simulation unstable (an arm joint over 8 rad/s, a burner knob or a door "
                                            "over 30 rad/s, or a MuJoCo bad-state warning)")
                    sc = ep.score()
                    results.append({"unit": uid, "seed": seed, "episode_credit": round(sc["credit"], 5), "steps": t,
                                    "requests": [{k: r[k] for k in ("text", "t_open", "t_close", "reached", "completed",
                                                                    "outcome")}
                                                 for r in sc["requests"]]})
                except (WireError, ValueError, TypeError) as exc:
                    results.append({"unit": uid, "seed": seed, "episode_credit": 0.0, "aborted": str(exc)[:200]})
                finally:
                    ep.env.close()
    except E.RenderError as exc:
        print(json.dumps({"error": "the cameras' renderer failed (an error of the environment, not of your "
                                   "controller; no metric): {}".format(str(exc)[:300]),
                          "episodes_before_the_failure": results}, indent=1))
        return 3
    finally:
        wire.close()
    wall = time.monotonic() - t0
    mean = sum(r["episode_credit"] for r in results) / len(results) if results else 0.0
    print(json.dumps({
        "metric": "mean_request_credit (dev sim, practice units {})".format(" ".join(ids)),
        "raw_value": round(mean, 5),
        "episodes": results,
        "repeated_t_pairs": pairs,
        "repeated_t_pairs_that_differed": differed,
        "repeated_t_pairs_not_counted_no_motion": idle,
        "repeated_t_check": "would pass" if pairs == 0 or differed >= 0.1 * pairs else "would FAIL (the whole run scores 0)",
        "wall_seconds": round(wall, 1),
        "controller_seconds": round(wire.seconds, 1),
        "within_grading_time_budget": wire.seconds <= BUDGET_PER_EPISODE * max(1, len(results)),
        "paths_outside_submission": outside_paths(a.submission),
        "note": "dev sim with the practice units; this is not your score. controller_seconds is your controller's "
                "own time plus 0.03 s per image asked for (the grading budget: 6000 s for 18 episodes); nothing of "
                "/home/user outside the submission exists at grading",
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
