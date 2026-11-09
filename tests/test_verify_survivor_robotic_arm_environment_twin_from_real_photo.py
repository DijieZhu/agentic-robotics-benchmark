"""robotic_arm_environment_twin_from_real_photo: a process of the agent account that outlives the verify-start kill (the
base image's supervisord respawns the desktop it runs as that account) is frozen, not held against the submission; one
that is still running once frozen still voids the run (task error).

Runs the grader's own `_early_lockdown` (and its helpers), taken from tests/verify.py by name, against stand-ins for
the root-only parts (the kill, the account freezer, the stage lock) and a real child process playing the survivor.
Stdlib only: tests/verify.py imports the grading stack (numpy, mujoco, ...) that the host venv does not have.
"""
import ast
import os
import signal
import stat
import subprocess
import tempfile
import time
import types
import unittest
from pathlib import Path

VERIFY = Path(__file__).resolve().parents[1] / "tasks/arm/robotic_arm_environment_twin_from_real_photo/tests/verify.py"


class _Os(types.ModuleType):
    """os, as root (getuid 0), with an optional stand-in for kill."""

    def __init__(self, kill=None):
        super().__init__("os")
        self._kill = kill

    def __getattr__(self, name):
        return getattr(os, name)

    @staticmethod
    def getuid():
        return 0

    def kill(self, pid, sig):
        return (self._kill or os.kill)(pid, sig)


class _Survivor:
    """A real process standing in for the respawned desktop process that outlived the kill."""

    def __init__(self):
        self.proc = subprocess.Popen(["sleep", "60"])

    def pids(self, uid=None):
        return [self.proc.pid] if self.proc.poll() is None else []

    def close(self):
        try:
            os.kill(self.proc.pid, signal.SIGCONT)
        except OSError:
            pass
        self.proc.kill()
        self.proc.wait()


def _functions(names):
    tree = ast.parse(VERIFY.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names), "verify.py lacks %s" % (set(names) - {n.name for n in body})
    return compile(ast.Module(body=body, type_ignores=[]), str(VERIFY), "exec")


class SurvivorAtVerifyStart(unittest.TestCase):
    def setUp(self):
        self.survivor = _Survivor()
        self.addCleanup(self.survivor.close)
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.tmp / "stage").mkdir()

    def _lockdown(self, left, freeze=True):
        """Run _early_lockdown with the kill reporting `left` survivors; `freeze` False: SIGSTOP has no effect."""
        logs = []
        survivor = self.survivor
        stop = (lambda pid, sig: None) if not freeze else None
        started = []

        class Freezer:
            frozen = set()

            def start(self):
                started.append(True)
                return self

        lockdown = types.SimpleNamespace(
            stop_and_kill_uid=lambda uid, grace_s=5.0: (54, left),
            live_uid_processes=survivor.pids,
            stage_upload_time=lambda d: None, late_entries=lambda *a, **k: [], snapshot_dir=lambda a, b: None)
        runner = types.SimpleNamespace(lock_stage=lambda d: {})
        home = self.tmp / "home"
        ns = {"os": _Os(kill=stop), "pwd": types.SimpleNamespace(getpwnam=lambda n: types.SimpleNamespace(pw_uid=4242)),
              "time": time, "signal": signal, "statmod": stat, "lockdown": lockdown, "runner": runner,
              "FREEZER": Freezer(), "log": logs.append, "_submission_root_ok": lambda: (True, None),
              "_submission_measure": lambda p: (0, 0, 0), "SUBMISSION_ENTRIES_CAP": 5000,
              "SUBMISSION_BYTES_CAP": 300 * 1024 * 1024, "LATE_GATE_MAX_GAP_S": 600.0,
              "STAGE_DIR": str(self.tmp / "stage"), "HOME": str(home), "AGENT_USER": "user",
              "SUBMISSION": str(home / "submission"), "SNAP_ROOT": str(self.tmp / "snap"),
              "SNAPSHOT": str(self.tmp / "snap" / "submission")}
        exec(_functions(["_proc_state", "_still_running", "_early_lockdown"]), ns)
        out = ns["_early_lockdown"]()
        self.assertTrue(started, "the account freezer is started after the kill")
        return out, logs

    def test_frozen_survivor_does_not_void(self):
        out, logs = self._lockdown(left=1)
        self.assertEqual(out["agent_processes_left"], 1.0)
        self.assertEqual(out["agent_processes_running"], 0.0, out)
        self.assertTrue(any("outlived the kill" in m for m in logs), logs)
        self.assertEqual(Path("/proc/%d/status" % self.survivor.proc.pid).read_text().split("State:")[1].split()[0], "T")

    def test_survivor_that_cannot_be_stopped_still_voids(self):
        out, _ = self._lockdown(left=1, freeze=False)
        self.assertGreater(out["agent_processes_running"], 0.0, out)

    def test_clean_kill_records_nothing(self):
        out, logs = self._lockdown(left=0)
        self.assertEqual(out["agent_processes_left"], 0.0)
        self.assertEqual(out["agent_processes_running"], 0.0)
        self.assertFalse(any("outlived" in m for m in logs), logs)

    def test_only_a_running_survivor_voids(self):
        """main() voids the run (fail(), a task error) on agent_processes_running, never on the kill's raw count."""
        main = next(n for n in ast.parse(VERIFY.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")

        def voids(node):
            return any(isinstance(b, ast.Raise) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Call)
                                                   and getattr(b.value.func, "id", None) == "fail")
                       for b in node.body)
        tests = [ast.unparse(n.test) for n in ast.walk(main) if isinstance(n, ast.If) and voids(n)]
        self.assertFalse([t for t in tests if "agent_processes_left" in t], tests)
        self.assertTrue([t for t in tests if "agent_processes_running" in t], tests)


if __name__ == "__main__":
    unittest.main()
