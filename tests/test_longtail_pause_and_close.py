"""longtail_manipulation_recovery: the grader's pause and its close path (validation review of #80).

1. A controller whose main thread has exited shows in /proc/<pid>/stat as a zombie while its other threads run on.
   The pause must stop it all the same, and confirm the stop thread by thread.
2. After a request has timed out in the write, the pipe to the controller is full. Closing the controller must not
   write to it with a blocking write, on the last state or once the restarts are used up.
3. The time of a request that aborts counts into the `controller_seconds` metric; a process that has exited does not
   count toward the process cap.

Runs the grader's own `GuardedSolver` (tests/verify.py) and the practice grader's `Controller` against real child
processes of the test's own account, with the account's process list and the launcher stood in for. Stdlib only.
"""
import importlib.util
import os
import platform
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

TASK = Path(__file__).resolve().parents[1] / "tasks/arm/longtail_manipulation_recovery"
SYS_EXIT = {"x86_64": 60, "aarch64": 93}.get(platform.machine())      # exits the calling thread only

WORKER = r"""
import ctypes, sys, threading, time
def worker():
    n = 0
    while True:
        n += 1
        if n % 20000 == 0:
            with open(sys.argv[1], "a") as f:
                f.write("x")
threading.Thread(target=worker).start()
print("READY", flush=True)
if sys.argv[2] != "0":
    ctypes.CDLL(None).syscall(int(sys.argv[2]), 0)
while True:
    time.sleep(1)
"""

NO_READER = r"""
import json, sys, time
print("READY", flush=True)
sys.stdin.readline()
print(json.dumps({"ok": True}), flush=True)
while True:
    time.sleep(1)
"""


READER = r"""
import json, sys
print("READY", flush=True)
for line in sys.stdin:
    if json.loads(line).get("type") == "close":
        open(sys.argv[1], "w").write("closed")
        break
    print(json.dumps({"ok": True}), flush=True)
"""


def _load(name, path, folder):
    """A module from a file, with its own folder importable while it loads (wire_rules, the kit)."""
    before = list(sys.path)
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = before
    return mod


V = _load("longtail_verify", TASK / "tests/verify.py", TASK / "tests")
G = _load("longtail_practice_grade", TASK / "environment/setup/payload/practice_grader/grade.py",
          TASK / "environment/setup/payload")


def _thread_states(pid):
    out = []
    try:
        for tid in os.listdir("/proc/%d/task" % pid):
            raw = Path("/proc/%d/task/%s/stat" % (pid, tid)).read_bytes()
            out.append(raw[raw.rfind(b")") + 1:].split()[0].decode())
    except OSError:
        pass
    return out


def _leader_state(pid):
    raw = Path("/proc/%d/stat" % pid).read_bytes()
    return raw[raw.rfind(b")") + 1:].split()[0].decode()


def _kill(proc):
    for sig in (signal.SIGCONT, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except Exception:  # noqa: BLE001
        pass


@unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc")
class PauseTest(unittest.TestCase):
    def _solver(self):
        solver = object.__new__(V.GuardedSolver)
        solver.uid, solver._stopped, solver.outside, solver.unstoppable = os.getuid(), [], [], []
        return solver

    def _worker(self, tmp, main_thread_exits):
        progress = os.path.join(tmp, "progress")
        proc = subprocess.Popen([sys.executable, "-c", WORKER, progress, str(SYS_EXIT if main_thread_exits else 0)],
                                stdout=subprocess.PIPE, start_new_session=True)
        self.addCleanup(_kill, proc)
        self.assertEqual(proc.stdout.readline().strip(), b"READY")
        return proc, progress

    def _paused(self, proc, progress):
        solver = self._solver()
        with mock.patch.object(V, "_agent_pids", lambda uid: [proc.pid]), \
                mock.patch.object(V, "outside_launcher", lambda pid: False):
            count = solver.freeze()
        self.assertEqual(solver.unstoppable, [])
        states = _thread_states(proc.pid)
        size = os.path.getsize(progress) if os.path.exists(progress) else 0
        time.sleep(0.5)
        grown = (os.path.getsize(progress) if os.path.exists(progress) else 0) - size
        return count, states, grown

    def test_a_process_with_its_main_thread_alive_is_paused(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, progress = self._worker(tmp, main_thread_exits=False)
            count, states, grown = self._paused(proc, progress)
            self.assertEqual(count, 1)
            self.assertTrue(states and all(s in ("T", "t") for s in states), states)
            self.assertEqual(grown, 0)

    @unittest.skipIf(SYS_EXIT is None, "no thread-exit system call number for this machine")
    def test_a_process_whose_main_thread_has_exited_is_paused(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc, progress = self._worker(tmp, main_thread_exits=True)
            for _ in range(200):                              # the leader becomes a zombie, the worker runs on
                if _leader_state(proc.pid) == "Z":
                    break
                time.sleep(0.01)
            self.assertEqual(_leader_state(proc.pid), "Z")
            before = os.path.getsize(progress) if os.path.exists(progress) else 0
            time.sleep(0.3)
            self.assertGreater(os.path.getsize(progress), before, "the worker thread is not computing")
            count, states, grown = self._paused(proc, progress)
            self.assertEqual(count, 1)
            self.assertTrue(all(s in ("T", "t", "Z", "X") for s in states) and "T" in states, states)
            self.assertEqual(grown, 0, "the worker thread computed while the account was paused")

    def test_a_process_that_has_exited_does_not_count_toward_the_cap(self):
        proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
        self.addCleanup(_kill, proc)
        for _ in range(500):                                  # exited, not yet reaped
            if _leader_state(proc.pid) == "Z":
                break
            time.sleep(0.01)
        self.assertEqual(_leader_state(proc.pid), "Z")
        solver = self._solver()
        with mock.patch.object(V, "_agent_pids", lambda uid: [proc.pid]), \
                mock.patch.object(V, "outside_launcher", lambda pid: False):
            self.assertEqual(solver.freeze(), 0)


def _returns(fn, seconds):
    """True when fn() returns within `seconds` (it runs in a daemon thread, so a hang does not hold the test)."""
    done = threading.Event()

    def run():
        try:
            fn()
        finally:
            done.set()
    threading.Thread(target=run, daemon=True).start()
    return done.wait(seconds)


@unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc")
class CloseTest(unittest.TestCase):
    BIG = {"images": "x" * 1300000}                           # larger than a pipe's buffer

    def _grader(self, max_restarts, code=NO_READER, arg=""):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        Path(tmp, "run.sh").write_text("exit 0\n")
        procs = []

        def spawn(_self):
            proc = subprocess.Popen([sys.executable, "-c", code, arg], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    universal_newlines=True, bufsize=1, start_new_session=True)
            procs.append(proc)
            self.addCleanup(_kill, proc)
            return proc

        patches = [mock.patch.object(V.GuardedSolver, "_spawn", spawn),
                   mock.patch.object(V, "_agent_pids", lambda uid: [p.pid for p in procs if p.poll() is None]),
                   mock.patch.object(V, "outside_launcher", lambda pid: False),
                   mock.patch.object(V, "kill_agent_processes", lambda uid, **kw: None)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return V.GuardedSolver(os.getuid(), step_timeout=1.0, submission=tmp, max_restarts=max_restarts), procs

    def test_close_returns_after_a_request_timed_out_in_the_write(self):
        solver, procs = self._grader(max_restarts=3)
        solver.reset()
        with self.assertRaises(V.SolverAborted):
            solver.act(0, self.BIG)
        self.assertTrue(_returns(solver.close, 15), "close() blocks on the full pipe")
        self.assertIsNotNone(procs[-1].poll())

    def test_no_controller_is_left_once_the_restarts_are_used_up(self):
        solver, procs = self._grader(max_restarts=0)
        solver.reset()
        with self.assertRaises(V.SolverAborted):
            solver.act(0, self.BIG)
        with self.assertRaises(V.SolverAborted):
            solver.reset()                                    # the next state: no restart is left
        self.assertIsNotNone(procs[-1].poll(), "the controller that stopped reading is still running")
        self.assertTrue(_returns(solver.close, 15), "close() blocks on the full pipe")

    def test_a_healthy_controller_is_still_told_to_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            marker = os.path.join(tmp, "closed")
            solver, procs = self._grader(max_restarts=3, code=READER, arg=marker)
            solver.reset()
            self.assertTrue(_returns(solver.close, 15))
            self.assertTrue(os.path.exists(marker), "the close message did not reach the controller")
            self.assertIsNotNone(procs[-1].poll())

    def test_the_practice_grader_closes_after_a_request_timed_out_in_the_write(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        Path(tmp, "ctl.py").write_text(NO_READER)
        Path(tmp, "run.sh").write_text("exec %s ctl.py\n" % sys.executable)
        with mock.patch.object(G.W, "STEP_TIMEOUT_S", 1.0):
            ctrl = G.Controller(tmp)
            self.addCleanup(_kill, ctrl.proc)
            ctrl.reset()
            with self.assertRaises(G.Aborted):
                ctrl.act(0, self.BIG)
            self.assertTrue(_returns(ctrl.close, 15), "the practice grader's close() blocks on the full pipe")


class MetricTest(unittest.TestCase):
    def test_an_aborted_request_counts_into_controller_seconds(self):
        class Sim:
            prefix_max_err = 0.0

            def __init__(self, spec, log):
                pass

            def first(self):
                return {"obs": {}}

            def close(self):
                pass

        class Solver:
            last_reply_s = 0.0

            def reset(self):
                pass

            def act(self, t, obs):
                self.last_reply_s = 1.5
                raise V.SolverAborted("late")

        stats = {}
        state = {"id": "x", "task": "t", "seed": 0, "split": "s", "prefix": "p", "qpos": "q"}
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(V, "SimWorker", Sim):
            with self.assertRaises(V.SolverAborted):
                V.run_state({"step_budget": 5}, state, Solver(), {"_logdir": tmp}, False, stats)
        self.assertAlmostEqual(stats["controller_s"], 1.5)


if __name__ == "__main__":
    unittest.main()
