"""kitchen_appliance_tending and longtail_manipulation_recovery: a pause that a controller ends by itself.

SIGSTOP holds only until the next SIGCONT, and a process can arrange its own: a POSIX timer whose signal is SIGCONT
continues it after the grader has confirmed the pause (validation reviews of #80 and #81). Both graders therefore
look again before they write the next request: every thread that was stopped must still be there and stopped, with the
scheduler's record of its run time unchanged, and no thread may have appeared.

1. A controller that stays stopped passes the check, pause after pause.
2. A controller with such a timer computes during the pause; the check ends the episode.
3. A controller that starts a process during the pause (and stops again itself) is caught as well.

Runs the graders' own pause code (tests/verify.py) against real child processes of the test's own account, with the
account's process list and the launcher stood in for. Stdlib only.
"""
import importlib.util
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

TASKS = Path(__file__).resolve().parents[1] / "tasks/arm"

WORKER = r"""
import ctypes, os, signal, subprocess, sys, threading, time
mode, progress = sys.argv[1], sys.argv[2]
def worker():
    n = 0
    while True:
        n += 1
        if n % 20000 == 0:
            with open(progress, "a") as f:
                f.write("x")
if mode == "timer":
    class sigval(ctypes.Union):
        _fields_ = [("sival_int", ctypes.c_int), ("sival_ptr", ctypes.c_void_p)]
    class sigevent(ctypes.Structure):
        _fields_ = [("sigev_value", sigval), ("sigev_signo", ctypes.c_int), ("sigev_notify", ctypes.c_int),
                    ("pad", ctypes.c_char * 48)]
    class timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]
    class itimerspec(ctypes.Structure):
        _fields_ = [("it_interval", timespec), ("it_value", timespec)]
    libc = ctypes.CDLL(None, use_errno=True)
    if not hasattr(libc, "timer_create"):
        libc = ctypes.CDLL("librt.so.1", use_errno=True)
    signal.signal(signal.SIGCONT, lambda signum, frame: None)   # newer kernels drop a timer's ignored signal
    ev = sigevent()
    ev.sigev_signo, ev.sigev_notify = signal.SIGCONT, 0          # SIGEV_SIGNAL
    timer = ctypes.c_void_p()
    assert libc.timer_create(0, ctypes.byref(ev), ctypes.byref(timer)) == 0
    every = itimerspec(timespec(0, 20000000), timespec(0, 20000000))
    assert libc.timer_settime(timer, 0, ctypes.byref(every), None) == 0
if mode == "child":
    def spawn(signum, frame):
        subprocess.Popen(["sleep", "60"])
    signal.signal(signal.SIGUSR1, spawn)
threading.Thread(target=worker, daemon=True).start()
print("READY", flush=True)
while True:
    time.sleep(1)
"""


def _load(name, path, folder):
    before = list(sys.path)
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = before
    return mod


def _group(pgid):
    """The processes of the group `pgid` (the controller and what it started)."""
    pids = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                if os.getpgid(int(name)) == pgid:
                    pids.append(int(name))
            except OSError:
                pass
    return pids


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


def _size(path):
    return os.path.getsize(path) if os.path.exists(path) else 0


class _Base(object):
    """The three cases, for a grader given by `load()`, `pauser()`, the name of its process-list function and
    `pause(p)`, which freezes and raises as the grader does."""

    def _start(self, mode):
        tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(tmp, ignore_errors=True))
        progress = os.path.join(tmp, "progress")
        proc = subprocess.Popen([sys.executable, "-c", WORKER, mode, progress], stdout=subprocess.PIPE,
                                start_new_session=True)
        self.addCleanup(_kill, proc)
        self.assertEqual(proc.stdout.readline().strip(), b"READY")
        return proc, progress

    def _patched(self, proc):
        return mock.patch.object(self.V, self.PIDS, lambda uid: _group(proc.pid))

    def test_a_controller_that_stays_stopped_passes(self):
        proc, progress = self._start("plain")
        p = self.pauser()
        with self._patched(proc):
            for _ in range(5):
                self.pause(p)
                size = _size(progress)
                time.sleep(0.2)
                self.assertEqual(_size(progress), size, "it computed while paused")
                p.check_held()                      # raises if the pause did not hold
                p.thaw()
                time.sleep(0.05)
        self.assertEqual(p.resumed, 0)

    def test_a_controller_that_continues_itself_by_a_timer_is_caught(self):
        proc, progress = self._start("timer")
        p = self.pauser()
        with self._patched(proc):
            try:
                self.pause(p)
            except self.V.SolverAborted as exc:     # continued again before two readings agreed: never held
                self.assertIn("did not stop", str(exc))
                return
            size = _size(progress)
            time.sleep(0.5)
            if _size(progress) == size:
                self.skipTest("the timer's SIGCONT did not continue the process on this kernel")
            with self.assertRaises(self.V.SolverAborted) as caught:
                p.check_held()
        self.assertIn("ran during a pause", str(caught.exception))
        self.assertEqual(p.resumed, 1)

    def test_a_process_started_during_the_pause_is_caught(self):
        proc, progress = self._start("child")
        p = self.pauser()
        with self._patched(proc):
            self.pause(p)
            os.kill(proc.pid, signal.SIGUSR1)       # pending while it is stopped
            os.kill(proc.pid, signal.SIGCONT)       # stands in for a signal the controller arranged
            time.sleep(0.3)
            os.kill(proc.pid, signal.SIGSTOP)
            for pid in _group(proc.pid):
                os.kill(pid, signal.SIGSTOP)
            time.sleep(0.1)
            with self.assertRaises(self.V.SolverAborted):
                p.check_held()


@unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc")
class KitchenPauseRecheck(_Base, unittest.TestCase):
    PIDS = "_account_pids"

    @classmethod
    def setUpClass(cls):
        folder = TASKS / "kitchen_appliance_tending/tests"
        try:
            cls.V = _load("kitchen_verify", folder / "verify.py", folder)
        except ImportError as exc:                   # the grader imports the simulation at module level
            raise unittest.SkipTest("the kitchen grader's imports are not installed here ({})".format(exc))

    def pauser(self):
        return self.V.Pauser(os.getuid())

    def pause(self, p):
        p.pause_or_abort()


@unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc")
class LongtailPauseRecheck(_Base, unittest.TestCase):
    PIDS = "_agent_pids"

    @classmethod
    def setUpClass(cls):
        folder = TASKS / "longtail_manipulation_recovery/tests"
        cls.V = _load("longtail_verify_recheck", folder / "verify.py", folder)

    def pauser(self):
        solver = object.__new__(self.V.GuardedSolver)
        solver.uid, solver._stopped, solver.outside, solver.unstoppable = os.getuid(), [], [], []
        solver._held, solver.resumed, solver._broken = None, 0, False
        return solver

    def pause(self, p):
        with mock.patch.object(self.V, "outside_launcher", lambda pid: False):
            p._pause_or_abort()


if __name__ == "__main__":
    unittest.main()
