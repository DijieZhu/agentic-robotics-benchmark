"""Commissioning service of cable_routing_through_clips: development boards the agent may work on before it hands in.

It runs as its own account (`harnessbench`), from a folder the agent's account cannot read (this code, the
simulator, the renderer, the ledger), and answers on a UNIX socket with the cell's public wire only: the
observation a controller gets when it is graded, nothing else. No ground truth (no fixture as mounted, no cable
point, no tension), no credit, no word on whether a fixture holds the cable or the connector was overloaded. An
episode ends by the public rules only (the step limit, the controller's hand-over, an unstable simulation). After
an episode has ended the cell has opened the hand, lifted it and let the cable settle, as the graded cell does
before it judges; `look` then shows the settled board.

The boards are BOARDS development boards of three fixtures each (bench.json; board j's seed is the first 8 bytes,
big endian, of sha256("cable_routing_through_clips/dev/v1/" + str(j)): asking for the same board again stages the
same board again; the repository test checks them against the scored set). Each staged board runs in a process of
its own (harness_sim.EpisodeProcess).

The window is bench time: WINDOW_STEPS control steps in all; RESET_STEPS are charged for staging a board and
IMAGE_STEPS for every image (an action's or a `look`'s); at most MAX_BENCHES connections at a time. The count is
kept in state/ledger.json under a lock, so it survives reconnects and a restart of this service.

Wire: one JSON object per line. Request {"id": ..., "method": ..., "params": {...}}, reply {"id": ..., "result": {...}}
or {"id": ..., "error": code, "detail": text}. Methods: status, reset {board}, step {action}, look {cams} (the
cameras now, no step), probe (the other copy of the last observation, as the grader's repeated requests carry),
close.
"""

import fcntl
import hashlib
import json
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

import harness_sim as S  # noqa: E402

SOCK_DIR = "/run/harnessbench"
SOCK = os.path.join(SOCK_DIR, "commission.sock")
STATUS = os.path.join(SOCK_DIR, "commission_status.json")
STATE = os.path.join(HERE, "state")
LEDGER = os.path.join(STATE, "ledger.json")
LOCK = os.path.join(STATE, "ledger.lock")
with open(os.path.join(HERE, "bench.json")) as _f:
    BENCH = json.load(_f)
BOARDS = int(BENCH["boards"])                     # development boards 0 .. BOARDS - 1
BOARD_FIXTURES = int(BENCH["board_fixtures"])
WINDOW_STEPS = int(BENCH["window_steps"])
RESET_STEPS = int(BENCH["reset_steps"])
IMAGE_STEPS = int(BENCH["image_steps"])
MAX_BENCHES = int(BENCH["max_benches"])
SILENCE_S = float(BENCH["silence_s"])
MAX_LINE = 1 << 16
FLUSH = 50
PUBLIC = ("status", "reset", "step", "look", "probe", "close")


def board_seed(j):
    """The 64-bit seed of development board `j` (never a scored board's: the repository test checks)."""
    return int.from_bytes(hashlib.sha256(("cable_routing_through_clips/dev/v1/%d" % int(j)).encode()).digest()[:8], "big")


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
                   "image_steps": IMAGE_STEPS, "boards": BOARDS}, f)
    os.chmod(tmp, 0o644)
    os.replace(tmp, STATUS)


class Bench(object):
    """One connection: one board at a time, staged by `reset`, stepped by `step`."""

    def __init__(self, conn):
        self.conn = conn
        self.buf = b""
        self.ep = None
        self.board = None
        self.done = True
        self.have_obs = False
        self.pending = 0
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

    # -- the board
    def wire(self, obs):
        w = dict(obs)
        w["job"] = self.ep.job
        return w

    def flush(self, force=False):
        if self.pending and (force or self.pending >= FLUSH):
            ledger(delta_steps=self.pending)
            self.pending = 0

    def left(self):
        return max(0, WINDOW_STEPS - ledger()["steps_used"] - self.pending)

    def drop(self):
        ep, self.ep = self.ep, None
        if ep is not None:
            ep.close()

    def reset(self, params):
        board = params.get("board", 0)
        if isinstance(board, bool) or not isinstance(board, int) or not 0 <= board < BOARDS:
            return None, "bad_params", "board is an integer from 0 to {}".format(BOARDS - 1)
        self.flush(force=True)
        if self.left() < RESET_STEPS + 1:
            return None, "commissioning_window_used", "no bench time is left"
        ledger(delta_steps=RESET_STEPS, delta_resets=1)
        self.drop()
        self.board = board
        self.ep = S.EpisodeProcess(render=True)
        obs = self.ep.reset(board_seed(board), BOARD_FIXTURES)
        self.done, self.have_obs = False, True
        return {"obs": self.wire(obs), "board": board, "steps_left": self.left()}, None, None

    def step(self, params):
        if self.ep is None or self.done:
            return None, "no_episode", "stage a board with reset first (the last episode is over)"
        action = params.get("action")
        if not S.valid_action(action):
            return None, "malformed_action", "8 or 9 finite numbers, the camera bits an integer 0-3, the ninth 0 or 1"
        cost = 1 + IMAGE_STEPS * S.images_asked(action)
        if self.left() < cost:
            self.done = True
            return None, "commissioning_window_used", "no bench time is left"
        obs, done = self.ep.step(action)
        self.pending += cost
        ended = None
        if done:
            ended = ("the controller handed the job over" if S.wants_finish(action) else
                     "an unstable simulation (it would score the episode 0)" if self.ep.unstable() else
                     "all {} steps run".format(self.ep.job["max_steps"]))
            self.ep.finish()                 # the hand opens and lifts, the cable settles; the verdict is not kept
        self.done = bool(done)
        self.flush(force=self.done)
        return {"obs": self.wire(obs), "done": self.done, "ended": ended, "steps_left": self.left()}, None, None

    def look(self, params):
        cams = params.get("cams", S.ALL_CAMS)
        if self.ep is None or isinstance(cams, bool) or not isinstance(cams, int) or not 0 <= cams <= S.ALL_CAMS:
            return None, "no_episode" if self.ep is None else "bad_params", "stage a board first; cams is 0-3"
        cost = IMAGE_STEPS * bin(cams).count("1")
        if self.left() < cost:
            return None, "commissioning_window_used", "no bench time is left"
        self.pending += cost
        return {"obs": self.wire(self.ep.look(cams)), "steps_left": self.left()}, None, None

    def probe(self):
        if self.ep is None or not self.have_obs:
            return None, "no_episode", "stage a board with reset first"
        return {"obs": self.wire(self.ep.probe(int(self.rng.integers(2 ** 63))))}, None, None

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
                                           "image_steps": IMAGE_STEPS, "max_benches": MAX_BENCHES, "boards": BOARDS})
                    continue
                try:
                    result, error, detail = (self.reset(params) if method == "reset" else
                                             self.step(params) if method == "step" else
                                             self.look(params) if method == "look" else self.probe())
                except Exception:  # noqa: BLE001 - the bench's own fault (its process ended, its renderer failed): no cause is told
                    self.done = True
                    self.drop()
                    result, error, detail = None, "bench_fault", "the bench stopped; stage it again"
                if error is None:
                    self.send(rid, result=result)
                else:
                    self.send(rid, error=error, detail=detail)
        except (OSError, ValueError):
            pass
        finally:
            self.flush(force=True)
            self.drop()


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
