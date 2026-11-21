"""The kit's `Solver` never waits on a solver that does not read, and `ProbeLock` compares values only
(`shared/robotics_grader/solver.py`).

Four paths, each with a stand-in solver that runs as a child process of the test over the real wire:

- a request the solver does not read (longer than the pipe holds) aborts at the request's one deadline, which now
  starts before the write;
- `close()` returns after such a request, and after the restarts are used up;
- a solver that is given up on (no restart left) is stopped before the kit raises;
- two replies with a different count of numbers do not pass the lock; a task's `canonical=` decides what is compared.

Anything that could block is called on a thread and given a bounded time, so a regression fails the test and does
not hang the suite. Eight of the eleven tests fail on the kit as it was; the other three (a solver that reads, a
late reply followed by close, a grader's own stdin object) hold on both.
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "shared"))

from robotics_grader import seeds, solver as solver_mod  # noqa: E402

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc, pipe sizes")

LONG = [0.0] * 400000          # about 2 MB on the wire: far more than a pipe holds

# Answers the reset, then never reads again.
DEAF = """
    import json, sys, time
    print("READY", flush=True)
    sys.stdin.readline()
    print(json.dumps({"ok": True}), flush=True)
    time.sleep(120)
"""
# Answers everything; leaves a file when it is asked to close.
HEALTHY = """
    import json, sys
    print("READY", flush=True)
    for line in sys.stdin:
        req = json.loads(line.strip() or "{}")
        if req.get("type") == "close":
            open("closed", "w").close()
            print(json.dumps({"ok": True}), flush=True)
            break
        if req.get("type") == "act":
            print(json.dumps({"action": [float(len(req["obs"]))]}), flush=True)
        else:
            print(json.dumps({"ok": True}), flush=True)
"""
# Answers resets; stalls on every act, with the pipe still being read by nobody.
STALL = """
    import json, sys, time
    print("READY", flush=True)
    for line in sys.stdin:
        req = json.loads(line.strip() or "{}")
        if req.get("type") == "reset":
            print(json.dumps({"ok": True}), flush=True)
            continue
        time.sleep(120)
"""
# Waits before it reads each act, then answers at once.
SLOW_READER = """
    import json, sys, time
    print("READY", flush=True)
    sys.stdin.readline()
    print(json.dumps({"ok": True}), flush=True)
    while True:
        time.sleep(%(wait)s)
        line = sys.stdin.readline()
        if not line:
            break
        print(json.dumps({"action": [1.0]}), flush=True)
"""


class LocalSolver(solver_mod.Solver):
    """The kit's Solver over the real wire, started in-process (no account switch, no clearing)."""

    def __init__(self, script, step_timeout, **kwargs):
        super().__init__(step_timeout, submission=os.path.dirname(script), entry=os.path.basename(script),
                         clear_account=False, **kwargs)

    def _spawn(self):
        return subprocess.Popen([sys.executable, self.entry], cwd=self.submission, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, universal_newlines=True,
                                bufsize=1, start_new_session=True)


def script(tmp_path, body, name="solver.py"):
    path = tmp_path / name
    path.write_text(textwrap.dedent(body))
    return str(path)


def bounded(fn, seconds):
    """Run `fn` on a thread. Returns (finished, seconds it took, what it raised or None)."""
    box = {}

    def run():
        try:
            fn()
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            box["exc"] = exc

    thread = threading.Thread(target=run, daemon=True)
    began = time.monotonic()
    thread.start()
    thread.join(seconds)
    return not thread.is_alive(), time.monotonic() - began, box.get("exc")


def alive(pid):
    try:
        with open("/proc/%d/stat" % pid) as handle:
            return handle.read().rsplit(")", 1)[1].split()[0] not in ("Z", "X")
    except (OSError, IndexError):
        return False


def kill(solver):
    solver_mod._kill_group(solver.proc.pid)


def test_unread_request_aborts_at_the_deadline(tmp_path):
    solver = LocalSolver(script(tmp_path, DEAF), step_timeout=1.0)
    try:
        solver.reset()
        done, took, exc = bounded(lambda: solver.act(0, LONG), 15)
        assert done, "the request was still being written after 15 s"
        assert isinstance(exc, solver_mod.SolverAborted)
        assert "deadline" in str(exc)
        assert 0.9 < took < 5.0
        assert solver._broken and solver._cut
    finally:
        kill(solver)


def test_close_returns_after_an_unread_request(tmp_path):
    solver = LocalSolver(script(tmp_path, DEAF), step_timeout=1.0)
    try:
        solver.reset()
        done, _, _ = bounded(lambda: solver.act(0, LONG), 15)
        assert done
        pid = solver.proc.pid
        done, took, exc = bounded(solver.close, 15)
        assert done, "close() had not returned after 15 s"
        assert exc is None
        assert took < 5.0
        assert not alive(pid)
    finally:
        kill(solver)


def test_close_returns_when_the_pipe_is_full_of_whole_requests(tmp_path):
    """No request was cut short, so close() does try its message: the try itself must be bounded."""
    solver = LocalSolver(script(tmp_path, DEAF), step_timeout=0.3)
    try:
        solver.reset()
        fd = solver.proc.stdin.fileno()
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
        try:
            while True:
                os.write(fd, b" " * 4096)        # fill the pipe to the brim
        except BlockingIOError:
            pass
        fcntl.fcntl(fd, fcntl.F_SETFL, fcntl.fcntl(fd, fcntl.F_GETFL) & ~os.O_NONBLOCK)
        assert not solver._cut
        pid = solver.proc.pid
        done, took, exc = bounded(solver.close, 15)
        assert done, "close() had not returned after 15 s"
        assert exc is None
        assert solver_mod.CLOSE_WRITE_TIMEOUT * 0.9 < took < solver_mod.CLOSE_WRITE_TIMEOUT + 4.0
        assert not alive(pid)
    finally:
        kill(solver)


def test_given_up_solver_is_stopped(tmp_path):
    solver = LocalSolver(script(tmp_path, STALL), step_timeout=0.3, max_restarts=1)
    try:
        for _ in range(2):                        # the first abort, then the one restart and its abort
            with pytest.raises(solver_mod.SolverAborted):
                solver.reset()
                solver.act(0, [0.0])
        last = solver.proc.pid
        assert alive(last)
        with pytest.raises(solver_mod.SolverAborted):
            solver.reset()                        # no restart left
        assert solver._dead
        assert not alive(last), "the solver was left running after the kit gave up on it"
        began = time.monotonic()
        with pytest.raises(solver_mod.SolverAborted):
            solver.reset()
        assert time.monotonic() - began < 0.2     # later episodes still abort at once
        done, took, exc = bounded(solver.close, 15)
        assert done and exc is None and took < 2.0
    finally:
        kill(solver)


def test_given_up_solver_with_a_full_pipe_is_stopped_and_closes(tmp_path):
    solver = LocalSolver(script(tmp_path, DEAF), step_timeout=0.5, max_restarts=0)
    try:
        solver.reset()
        done, _, exc = bounded(lambda: solver.act(0, LONG), 15)
        assert done and isinstance(exc, solver_mod.SolverAborted)
        pid = solver.proc.pid
        with pytest.raises(solver_mod.SolverAborted):
            solver.reset()
        assert not alive(pid)
        done, took, exc = bounded(solver.close, 15)
        assert done and exc is None and took < 2.0
    finally:
        kill(solver)


def test_the_time_to_read_a_request_counts(tmp_path):
    """The solver takes 0.8 s to start reading a request that does not fit the pipe. With a 0.5 s deadline that
    request aborts; the write used to be outside the deadline, and the reply (immediate after the read) passed."""
    solver = LocalSolver(script(tmp_path, SLOW_READER % {"wait": 0.8}), step_timeout=0.5)
    try:
        solver.reset()
        done, took, exc = bounded(lambda: solver.act(0, LONG), 15)
        assert done
        assert isinstance(exc, solver_mod.SolverAborted)
        assert took < 0.8 + 0.5
    finally:
        kill(solver)


def test_a_reading_solver_is_unchanged(tmp_path):
    """Long requests arrive whole, a slow read inside the deadline is fine, and close is still delivered."""
    with LocalSolver(script(tmp_path, HEALTHY), step_timeout=5.0) as solver:
        solver.reset()
        for t in range(3):
            assert solver.act(t, LONG) == [float(len(LONG))]
        assert solver.act(3, [1.0, 2.0]) == [2.0]
        fd = solver.proc.stdin.fileno()
        assert not fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_NONBLOCK      # the descriptor is left as it was
    assert (tmp_path / "closed").exists()

    slow = LocalSolver(script(tmp_path, SLOW_READER % {"wait": 0.3}, "slow.py"), step_timeout=2.0)
    try:
        slow.reset()
        assert slow.act(0, LONG) == [1.0]
    finally:
        slow.close()


def test_close_after_a_reply_timeout_still_asks(tmp_path):
    """A reply that came late leaves the pipe at a line boundary: close() sends its message as before."""
    body = HEALTHY.replace('if req.get("type") == "act":', 'if req.get("type") == "act":\n            import time; time.sleep(0.6)')
    solver = LocalSolver(script(tmp_path, body), step_timeout=0.3)
    solver.reset()
    with pytest.raises(solver_mod.SolverAborted):
        solver.act(0, [0.0])
    assert solver._broken
    solver.close()
    assert (tmp_path / "closed").exists()


def test_a_graders_own_stdin_object_is_still_used(tmp_path):
    """A grader may replace `proc.stdin` with an object that bounds its own writes; the kit writes through it."""

    class Counting:
        def __init__(self, inner):
            self.inner, self.writes = inner, 0

        def write(self, text):
            self.writes += 1
            return self.inner.write(text)

        def flush(self):
            self.inner.flush()

        def close(self):
            self.inner.close()

    class Wrapped(LocalSolver):
        def _spawn(self):
            proc = super()._spawn()
            proc.stdin = Counting(proc.stdin)
            return proc

    with Wrapped(script(tmp_path, HEALTHY), step_timeout=5.0) as solver:
        solver.reset()
        assert solver.act(0, [1.0]) == [1.0]
        counted = solver.proc.stdin
    assert counted.writes == 3                    # reset, act, close
    assert (tmp_path / "closed").exists()


def test_probe_lock_compares_values_not_shapes():
    lock = solver_mod.ProbeLock(seeds.seed_rng(entropy=7))
    assert not lock.record([0.5, 0.0], [0.5])                     # one more number
    assert not lock.record([0.1] * 7, [0.1] * 7 + [0.0] * 7)      # the same step in two action shapes
    assert not lock.record([[0.5]], [[0.5], []] + [[0.0]])        # nesting is flattened first
    assert (lock.total, lock.passed) == (3, 0)
    assert not lock.ok
    assert lock.record([0.5, 0.0], [0.5, 0.2])                    # same shape, another value: sensitivity
    assert not lock.record([0.5, 0.0], [0.5, 0.0])


def test_probe_lock_canonical_form():
    def canonical(action):
        """Seven numbers, or fourteen whose second half is the posture: compare the first seven."""
        if not isinstance(action, list) or len(action) not in (7, 14):
            raise solver_mod.SolverAborted("action has neither 7 nor 14 numbers")
        return action[:7]

    lock = solver_mod.ProbeLock(seeds.seed_rng(entropy=7), canonical=canonical)
    assert not lock.record([0.1] * 7, [0.1] * 7 + [0.3] * 7)      # the same motion: no sensitivity shown
    assert lock.record([0.1] * 7, [0.2] * 7 + [0.3] * 7)          # another motion in another shape: shown
    assert (lock.total, lock.passed) == (2, 1)
    with pytest.raises(solver_mod.SolverAborted):
        lock.record([0.1] * 3, [0.1] * 7)
    with pytest.raises(solver_mod.SolverAborted):                 # a canonical form that is not numbers
        solver_mod.ProbeLock(seeds.seed_rng(entropy=7), canonical=lambda a: ["x"]).record([0.1], [0.1])
