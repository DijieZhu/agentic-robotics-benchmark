"""Commissioning service of lab_sample_handling: development benches the agent may work on before it hands in.

It runs as its own account (`labbench`), from a folder the agent's account cannot read (this code, the simulator,
the development units, the ledger), and answers on a UNIX socket with the bench's public wire only: the observation
a controller gets when it is graded, nothing else. No ground truth (no pose of anything, no mass, no fill, no rotor
angle, no unit property), no credit, no word on how a worklist went. An episode ends by the public rules only (the
step limit, the lid shut for LID_HOLD steps, an unstable simulation), never because a job is complete: the credit
rule is not in this folder.

The benches are other centrifuges of the same series (bench.json: development units drawn over the documents'
ranges, none of them a scored unit), each with EPISODES development worklists (`episode` 0, 1, 2, ...: the seed is a hash
of the unit's id and that number, so asking for the same one again stages the same bench again; the repository test
checks every one of them against the scored episodes).

The window is bench time: WINDOW_STEPS control steps in all, RESET_STEPS charged for staging a bench, at most
MAX_BENCHES connections at a time. The count is kept in state/ledger.json under a lock, so it survives reconnects
and a restart of this service.

Wire: one JSON object per line. Request {"id": ..., "method": ..., "params": {...}}, reply {"id": ..., "result": {...}}
or {"id": ..., "error": code, "detail": text}. Methods: status, reset {unit, episode}, step {action}, look {cams}
(the cameras now, no step), probe (a perturbed copy of the last observation, as the grader's repeated requests
carry), close.
"""

import fcntl
import hashlib
import json
import math
import os
import resource
import select
import signal
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

import lab_env as E  # noqa: E402

SOCK_DIR = "/run/labbench"
SOCK = os.path.join(SOCK_DIR, "commission.sock")
STATUS = os.path.join(SOCK_DIR, "commission_status.json")
STATE = os.path.join(HERE, "state")
LEDGER = os.path.join(STATE, "ledger.json")
LOCK = os.path.join(STATE, "ledger.lock")
with open(os.path.join(HERE, "bench.json")) as _f:
    BENCH = json.load(_f)
UNITS = BENCH["units"]
WINDOW_STEPS = int(BENCH["window_steps"])
RESET_STEPS = int(BENCH["reset_steps"])
MAX_BENCHES = int(BENCH["max_benches"])
EPISODES = int(BENCH["episodes"])                 # development worklists per unit: numbers 0 .. EPISODES - 1
EPISODE_STEPS = int(BENCH["episode_steps"])       # the graded episode's limits (tests/episode.py)
LID_HOLD, LID_SHUT, LID_OPENED = int(BENCH["lid_hold"]), float(BENCH["lid_shut"]), float(BENCH["lid_opened"])
SILENCE_S = float(BENCH["silence_s"])
MAX_LINE = 1 << 16
FLUSH = 50
PUBLIC = ("status", "reset", "step", "look", "probe", "close")


def episode_seed(unit, number):
    """The 64-bit seed of development worklist `number` of `unit` (never a scored episode's: tests check)."""
    h = hashlib.blake2b(("lab_sample_handling commissioning|%s|%d" % (unit, int(number))).encode(), digest_size=8)
    return int.from_bytes(h.digest(), "little")


def ledger(delta_steps=0, delta_resets=0):
    """Add to the ledger under its lock; returns the totals."""
    with open(LOCK, "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(LEDGER) as f:
                led = json.load(f)
        except (OSError, ValueError):
            led = {"steps_used": 0, "resets": 0}
        led["steps_used"] = int(led["steps_used"]) + int(delta_steps)
        led["resets"] = int(led["resets"]) + int(delta_resets)
        tmp = LEDGER + ".tmp"
        with open(tmp, "w") as f:
            json.dump(led, f)
        os.replace(tmp, LEDGER)
        return led


def write_status(open_benches):
    led = ledger()
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"window_steps": WINDOW_STEPS, "steps_used": led["steps_used"],
                   "steps_left": max(0, WINDOW_STEPS - led["steps_used"]), "resets": led["resets"],
                   "reset_steps": RESET_STEPS, "max_benches": MAX_BENCHES, "benches_open": open_benches,
                   "units": sorted(UNITS), "episodes": EPISODES}, f)
    os.chmod(tmp, 0o644)
    os.replace(tmp, STATUS)


def valid_action(action):
    if not isinstance(action, list) or len(action) not in (8, 9, 15):
        return False
    for x in action:
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
            return False
    cams = action[8] if len(action) == 9 else action[7]
    return 0 <= cams <= E.ALL_CAMS and float(cams).is_integer()


class Bench(object):
    """One connection: one bench at a time, staged by `reset`, stepped by `step`."""

    def __init__(self, conn):
        self.conn = conn
        self.buf = b""
        self.env = None
        self.unit = None
        self.obs = None
        self.done = True
        self.pending = 0
        self.lid_steps = 0
        self.lid_opened = False
        self.rng = np.random.default_rng()

    # -- the wire
    def read(self):
        while b"\n" not in self.buf:
            ready, _, _ = select.select([self.conn], [], [], SILENCE_S)
            if not ready:
                return None
            chunk = self.conn.recv(65536)
            if not chunk or len(self.buf) + len(chunk) > MAX_LINE:
                return None
            self.buf += chunk
        line, self.buf = self.buf.split(b"\n", 1)
        try:
            msg = json.loads(line.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return None
        return msg if isinstance(msg, dict) else None

    def send(self, rid, result=None, error=None, detail=None):
        msg = {"id": rid, "result": result} if error is None else {"id": rid, "error": error, "detail": detail}
        self.conn.sendall((json.dumps(msg) + "\n").encode("utf-8"))

    # -- the bench
    def wire(self, obs):
        w = E.to_wire(obs)
        w["unit"] = self.unit
        return w

    def flush(self, force=False):
        if self.pending and (force or self.pending >= FLUSH):
            ledger(delta_steps=self.pending)
            self.pending = 0

    def left(self):
        return max(0, WINDOW_STEPS - ledger()["steps_used"] - self.pending)

    def reset(self, params):
        unit, number = params.get("unit"), params.get("episode", 0)
        if unit not in UNITS or isinstance(number, bool) or not isinstance(number, int) or not 0 <= number < EPISODES:
            return None, "bad_params", "unit is one of {} and episode an integer from 0 to {}".format(
                sorted(UNITS), EPISODES - 1)
        self.flush(force=True)
        if self.left() < RESET_STEPS + 1:
            return None, "commissioning_window_used", "no bench time is left"
        ledger(delta_steps=RESET_STEPS, delta_resets=1)
        if self.env is not None:
            self.env.close()
        self.unit = unit
        self.env = E.LabEnv(UNITS[unit], unit, render=True)
        self.obs = self.env.reset(episode_seed(unit, number))
        self.done, self.lid_steps, self.lid_opened = False, 0, False
        return {"obs": self.wire(self.obs), "unit": unit, "episode": number, "steps_left": self.left()}, None, None

    def step(self, params):
        if self.env is None or self.done:
            return None, "no_episode", "stage a bench with reset first (the last episode is over)"
        action = params.get("action")
        if not valid_action(action):
            return None, "malformed_action", "8, 15 or 9 finite numbers, the camera bits an integer 0-15"
        if self.left() < 1:
            self.done = True
            return None, "commissioning_window_used", "no bench time is left"
        self.obs = self.env.step(action)
        self.pending += 1
        ended = None
        ang = self.env.p.lid_angle()
        self.lid_opened = self.lid_opened or ang > LID_OPENED
        self.lid_steps = self.lid_steps + 1 if (self.lid_opened and ang < LID_SHUT) else 0
        if self.env.unstable():
            ended = "unstable simulation (it would score the episode 0)"
        elif self.lid_steps >= LID_HOLD:
            ended = "the lid has been shut for {} steps".format(LID_HOLD)
        elif self.env.t >= EPISODE_STEPS:
            ended = "all {} steps run".format(EPISODE_STEPS)
        self.done = ended is not None
        self.flush(force=self.done)
        return {"obs": self.wire(self.obs), "done": self.done, "ended": ended, "steps_left": self.left()}, None, None

    def look(self, params):
        cams = params.get("cams", E.ALL_CAMS)
        if self.env is None or isinstance(cams, bool) or not isinstance(cams, int) or not 0 <= cams <= E.ALL_CAMS:
            return None, "no_episode" if self.env is None else "bad_params", "stage a bench first; cams is 0-15"
        self.env.render_next = cams
        return {"obs": self.wire(self.env.observation())}, None, None

    def probe(self):
        if self.env is None or self.obs is None:
            return None, "no_episode", "stage a bench with reset first"
        other = self.env.perturbed_observation(self.obs, np.random.default_rng(int(self.rng.integers(2 ** 63))))
        return {"obs": self.wire(other)}, None, None

    def serve(self):
        try:
            while True:
                msg = self.read()
                if msg is None:
                    break
                rid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
                if method not in PUBLIC or not isinstance(params, dict):
                    self.send(rid, error="forbidden", detail="methods: " + ", ".join(PUBLIC))
                    continue
                if method == "close":
                    self.send(rid, result={"closed": True})
                    break
                if method == "status":
                    led = ledger()
                    self.send(rid, result={"window_steps": WINDOW_STEPS,
                                           "steps_left": max(0, WINDOW_STEPS - led["steps_used"] - self.pending),
                                           "resets": led["resets"], "reset_steps": RESET_STEPS,
                                           "max_benches": MAX_BENCHES, "units": sorted(UNITS),
                                           "episodes": EPISODES})
                    continue
                try:
                    result, error, detail = (self.reset(params) if method == "reset" else
                                             self.step(params) if method == "step" else
                                             self.look(params) if method == "look" else self.probe())
                except Exception:  # noqa: BLE001 - the bench's own fault: no cause is told
                    self.done = True
                    result, error, detail = None, "bench_fault", "the bench stopped; stage it again"
                if error is None:
                    self.send(rid, result=result)
                else:
                    self.send(rid, error=error, detail=detail)
        except (OSError, ValueError):
            pass
        finally:
            self.flush(force=True)
            if self.env is not None:
                self.env.close()


def main():
    os.umask(0o077)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    os.makedirs(STATE, exist_ok=True)
    try:
        os.unlink(SOCK)
    except OSError:
        pass
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(SOCK)
    os.chmod(SOCK, 0o666)
    srv.listen(16)
    children = set()
    write_status(0)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        ready, _, _ = select.select([srv], [], [], 2.0)
        for pid in list(children):
            try:
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
                    children.discard(pid)
            except ChildProcessError:
                children.discard(pid)
        write_status(len(children))
        if not ready:
            continue
        conn, _ = srv.accept()
        if len(children) >= MAX_BENCHES:
            time.sleep(0.5)                     # a bench that has just been closed frees its place
            for pid in list(children):
                try:
                    if os.waitpid(pid, os.WNOHANG)[0] == pid:
                        children.discard(pid)
                except ChildProcessError:
                    children.discard(pid)
        if len(children) >= MAX_BENCHES:
            try:
                conn.sendall((json.dumps({"id": None, "error": "busy", "detail": "{} benches are in use".format(
                    MAX_BENCHES)}) + "\n").encode("utf-8"))
            except OSError:
                pass
            conn.close()
            continue
        pid = os.fork()
        if pid == 0:
            srv.close()
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            try:
                Bench(conn).serve()
            finally:
                try:
                    conn.close()
                except OSError:
                    pass
                os._exit(0)
        conn.close()
        children.add(pid)


if __name__ == "__main__":
    main()
