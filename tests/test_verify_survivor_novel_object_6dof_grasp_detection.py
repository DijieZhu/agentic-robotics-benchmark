"""novel_object_6dof_grasp_detection: a process of the agent account that outlives the verify-start kill (the base
image's supervisord respawns the desktop it runs as that account) is frozen and recorded, not held against the
submission; one that is still running once frozen still voids the run (task error). While the solvers run as that
account, every process of it the grader did not start is kept stopped.

Runs the grader's own `stop_account`, `_freeze`, `_descends_from` and `AccountFreezer`, taken from tests/verify.py by
name, against real child processes playing the survivors. Stdlib only: tests/verify.py imports the vendored grading kit
at module level, which is only staged in the verifier.
"""
import ast
import os
import signal
import subprocess
import threading
import time
import types
import unittest
from pathlib import Path

VERIFY = Path(__file__).resolve().parents[1] / "tasks/arm/novel_object_6dof_grasp_detection/tests/verify.py"
NAMES = ["_proc_state", "_describe", "_freeze", "stop_account", "_descends_from", "AccountFreezer"]


class Failed(Exception):
    pass


def _state(pid):
    with open("/proc/%d/status" % pid) as handle:
        return next(l.split()[1] for l in handle if l.startswith("State:"))


def _wait_state(pid, states, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if _state(pid) in states:
            return True
        time.sleep(0.02)
    return False


class _Os(types.ModuleType):
    """os, with kill() routed through `kill` (None: the real one)."""

    def __init__(self, kill=None):
        super().__init__("os")
        self._kill = kill

    def __getattr__(self, name):
        return getattr(os, name)

    def kill(self, pid, sig):
        return (self._kill or os.kill)(pid, sig)


def _load(pids, kill=None):
    tree = ast.parse(VERIFY.read_text())
    body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in NAMES]
    assert {n.name for n in body} == set(NAMES), "verify.py lacks %s" % (set(NAMES) - {n.name for n in body})
    logs = []

    def fail(msg):
        raise Failed(msg)

    ns = {"os": _Os(kill), "signal": signal, "time": time, "threading": threading, "log": logs.append, "fail": fail,
          "_live_uid_pids": lambda uid: [p for p in pids() if os.path.exists("/proc/%d" % p) and _state(p) not in "ZX"]}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(VERIFY), "exec"), ns)
    return ns, logs


class SurvivorAtVerifyStart(unittest.TestCase):
    def _child(self):
        proc = subprocess.Popen(["sleep", "60"])

        def close():
            try:
                os.kill(proc.pid, signal.SIGCONT)
            except OSError:
                pass
            proc.kill()
            proc.wait()
        self.addCleanup(close)
        return proc

    def _orphan(self):
        """A process that does not descend from this one (its parent exits, so it is reparented)."""
        out = subprocess.run(["sh", "-c", "sleep 60 >/dev/null 2>&1 & echo $!"], capture_output=True, text=True,
                             check=True)
        pid = int(out.stdout.strip())

        def close():
            for sig in (signal.SIGCONT, signal.SIGKILL):
                try:
                    os.kill(pid, sig)
                except OSError:
                    pass
        self.addCleanup(close)
        return pid

    def test_survivor_is_frozen_not_fatal(self):
        proc = self._child()
        # SIGKILL has no effect (a respawn as fast as the kill); SIGSTOP works
        kill = lambda pid, sig: None if sig == signal.SIGKILL else os.kill(pid, sig)
        ns, logs = _load(lambda: [proc.pid], kill)
        self.assertEqual(ns["stop_account"](4242), [proc.pid])
        self.assertTrue(_wait_state(proc.pid, "Tt"))
        self.assertTrue(any("outlived the kill" in m for m in logs), logs)

    def test_survivor_that_cannot_be_stopped_voids(self):
        proc = self._child()
        ns, logs = _load(lambda: [proc.pid], lambda pid, sig: None)
        with self.assertRaises(Failed):
            ns["stop_account"](4242)
        self.assertTrue(any("still running" in m for m in logs), logs)

    def test_clean_kill_records_nothing(self):
        ns, logs = _load(lambda: [])
        self.assertEqual(ns["stop_account"](4242), [])
        self.assertEqual(logs, [])

    def test_freezer_spares_the_graders_children_and_refreezes_the_rest(self):
        solver = self._child()           # descends from this process, as the solvers do under runuser
        stray = self._orphan()           # reparented, as the image's respawned desktop is
        ns, _ = _load(lambda: [solver.pid, stray])
        freezer = ns["AccountFreezer"](4242, period_s=0.05)
        freezer.start()
        self.addCleanup(freezer.halt.set)
        self.assertTrue(_wait_state(stray, "Tt"))
        os.kill(stray, signal.SIGCONT)   # a solver resuming it is undone within a period
        time.sleep(0.02)
        self.assertTrue(_wait_state(stray, "Tt"))
        self.assertNotIn(_state(solver.pid), "Tt")
        self.assertEqual(freezer.frozen, {stray})

    def test_main_wiring(self):
        """main() starts the freezer right after the first kill and records both counts in the verdict."""
        main = next(n for n in ast.parse(VERIFY.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        src = ast.unparse(main)
        self.assertLess(src.index("stop_account(account.pw_uid)"), src.index("freezer.start()"))
        self.assertLess(src.index("freezer.start()"), src.index("fetch_hidden(cfg)"))
        self.assertIn("'agent_processes_outlived_kill': outlived", src)
        self.assertIn("'agent_processes_frozen': len(freezer.frozen)", src)


if __name__ == "__main__":
    unittest.main()
