"""robotic_arm_full_assembly_so101_panda: a process of the agent account that outlives the verify-start kill (the base image's supervisord respawns
the desktop it runs as that account) is frozen, not held against the submission; one that is still running once
frozen still zeroes contract_ok.

Runs the grader's own `_early_lockdown` (and its helpers), taken from tests/verify.py by name, against stand-ins for
the root-only parts (the kill, the stage lock, the snapshot) and a real child process playing the survivor. Stdlib
only: tests/verify.py imports the grading stack (numpy, ...) that the host venv does not have.
"""
import ast
import os
import signal
import stat
import subprocess
import time
import types
import unittest
from pathlib import Path

VERIFY = Path(__file__).resolve().parents[1] / "tasks/arm/robotic_arm_full_assembly_so101_panda/tests/verify.py"
FAMILY = "contract"   # "contract": the survivor reason zeroes contract_ok; "void": a survivor raises a task error


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
        self.tmp = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        (self.tmp / "stage").mkdir()

    def _lockdown(self, left, freeze=True):
        """Run _early_lockdown with the kill reporting `left` survivors; `freeze` False: SIGSTOP has no effect."""
        logs = []
        survivor = self.survivor
        stop = (lambda pid, sig: None) if not freeze else None

        class Freezer:
            frozen = set()

            @classmethod
            def for_user(cls, user="user", **kw):
                return cls()

            def start(self):
                return self

            def freeze_now(self):
                for pid in survivor.pids():
                    (stop or os.kill)(pid, signal.SIGSTOP)
                return 1

        lockdown = types.SimpleNamespace(
            stop_and_kill_uid=lambda uid, grace_s=5.0: (52, left),
            live_uid_processes=survivor.pids,
            stage_upload_time=lambda d: None, late_entries=lambda *a, **k: [], quarantine=lambda *a, **k: 0,
            snapshot_dir=lambda a, b: None, process_age_ms=lambda: 1.0)
        runner = types.SimpleNamespace(lock_stage=lambda d: {})
        ns = {"os": _Os(kill=stop), "pwd": types.SimpleNamespace(getpwnam=lambda n: types.SimpleNamespace(pw_uid=4242)),
              "time": time, "signal": signal, "statmod": stat, "lockdown": lockdown, "runner": runner,
              "lock_stage": runner.lock_stage, "verify_freeze": types.SimpleNamespace(AgentFreezer=Freezer),
              "FREEZER": [None], "log": logs.append, "_uid_of": lambda n: 4242,
              "_submission_root_ok": lambda: (True, None),
              "STAGE_DIR": str(self.tmp / "stage"), "HOME": str(self.tmp / "home"), "AGENT_USER": "user",
              "SUBMISSION": str(self.tmp / "home" / "submission"), "LATE_GATE_MAX_GAP_S": 600.0, "WATCH_ROOTS": [],
              "RUNS_ROOT": str(self.tmp / "runs"), "SNAPSHOT": str(self.tmp / "snap"), "SNAP_ROOT": str(self.tmp),
              "QUARANTINE": str(self.tmp / "q")}
        exec(_functions(["_proc_state", "_still_running", "_early_lockdown"]), ns)
        fn = ns["_early_lockdown"]
        out = fn(Freezer()) if fn.__code__.co_argcount else fn()
        return out, logs

    def _voided(self, out):
        if FAMILY == "contract":
            return any("survived the kill" in r for r in out["reasons"])
        return out["agent_processes_running"] > 0

    def test_frozen_survivor_does_not_void(self):
        out, logs = self._lockdown(left=1)
        self.assertEqual(out["agent_processes_left"], 1.0)
        self.assertFalse(self._voided(out), out)
        self.assertTrue(any("outlived the kill" in m for m in logs), logs)
        self.assertEqual(Path("/proc/%d/status" % self.survivor.proc.pid).read_text().split("State:")[1].split()[0], "T")

    def test_survivor_that_cannot_be_stopped_still_voids(self):
        out, _ = self._lockdown(left=1, freeze=False)
        self.assertTrue(self._voided(out), out)

    def test_clean_kill_records_nothing(self):
        out, logs = self._lockdown(left=0)
        self.assertEqual(out["agent_processes_left"], 0.0)
        self.assertFalse(self._voided(out), out)
        self.assertFalse(any("outlived" in m for m in logs), logs)

    def test_only_a_running_survivor_raises(self):
        """main() voids the run on agent_processes_running, never on the kill's raw survivor count."""
        if FAMILY != "void":
            self.skipTest("contract family: the reason list is the gate")
        main = next(n for n in ast.parse(VERIFY.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        tests = [ast.unparse(n.test) for n in ast.walk(main)
                 if isinstance(n, ast.If) and any(isinstance(b, ast.Raise) for b in n.body)]
        self.assertFalse([t for t in tests if "agent_processes_left" in t], tests)
        self.assertTrue([t for t in tests if "agent_processes_running" in t], tests)


if __name__ == "__main__":
    unittest.main()
