"""tensegrity_collapsed_warehouse_void_search: a process of the agent account that outlives the verify-start kill (the
base image's supervisord respawns the desktop it runs as that account) is frozen until the verdict, not held against
the submission: it adds no contract reason, the kill's survivor count is recorded, and the freezer (the grader's own
cellkit.verify_freeze.AgentFreezer) keeps it stopped. The same design as the precedent's grader
(quadruped_hazardous_plant_inspection; the two tasks share cellkit byte for byte).

Runs the grader's own `_early_lockdown`, taken from tests/verify.py by name, against stand-ins for the root-only parts
(the kill, the stage lock, the snapshot), the real AgentFreezer (pointed at a stand-in process list) and a real child
process playing the survivor; and checks in main() that the run is gated on the early reasons only, never on the raw
survivor count, and that the freezer is stopped whatever happens. Stdlib only: tests/verify.py imports the cell
(mujoco, numpy) after the lockdown, which the host venv does not need.
"""
import ast
import importlib.util
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path

TESTS = Path(__file__).resolve().parents[1] / "tasks/soft/tensegrity_collapsed_warehouse_void_search/tests"
VERIFY = TESTS / "verify.py"
FREEZE = TESTS / "cellkit" / "verify_freeze.py"


def _module(path, name):
    """Load a module of the task's tests/ by path without writing bytecode into the task folder."""
    keep = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = keep
    return mod


class _Os(types.ModuleType):
    """os, as root (getuid 0)."""

    def __init__(self):
        super().__init__("os")

    def __getattr__(self, name):
        return getattr(os, name)

    @staticmethod
    def getuid():
        return 0


class _Survivor:
    """A real process standing in for the respawned desktop process that outlived the kill."""

    def __init__(self):
        self.proc = subprocess.Popen(["sleep", "60"])

    def pids(self, uid=None):
        return [self.proc.pid] if self.proc.poll() is None else []

    def state(self):
        return Path("/proc/%d/status" % self.proc.pid).read_text().split("State:")[1].split()[0]

    def close(self):
        try:
            os.kill(self.proc.pid, signal.SIGCONT)
        except OSError:
            pass
        self.proc.kill()
        self.proc.wait()


def _tree():
    return ast.parse(VERIFY.read_text())


def _functions(names):
    body = [n for n in _tree().body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names), "verify.py lacks %s" % (set(names) - {n.name for n in body})
    return compile(ast.Module(body=body, type_ignores=[]), str(VERIFY), "exec")


class SurvivorAtVerifyStart(unittest.TestCase):
    def setUp(self):
        self.survivor = _Survivor()
        self.addCleanup(self.survivor.close)
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.tmp / "stage").mkdir()
        self.freeze_mod = _module(FREEZE, "p09_verify_freeze")

    def _lockdown(self, left):
        """Run _early_lockdown with the kill reporting `left` survivors (the survivor process is alive either way)."""
        logs = []
        survivor = self.survivor
        real = self.freeze_mod.AgentFreezer
        made = []

        class Freezer(real):
            @classmethod
            def for_user(cls, user="user", **kw):
                f = cls(4242, period_s=0.02, lister=lambda uid: survivor.pids())
                made.append(f)
                return f

            def start(self, require_root=True):
                return real.start(self, require_root=False)    # the stand-in os says root; the module's os does not

        lockdown = types.SimpleNamespace(
            stage_upload_time=lambda d: None, stop_and_kill_uid=lambda uid, grace_s=8.0: (7, left),
            late_entries=lambda *a, **k: [], quarantine=lambda *a, **k: 0, snapshot_dir=lambda a, b: None,
            process_age_ms=lambda: 1.0)
        runner = types.SimpleNamespace(lock_stage=lambda d: {})
        ns = {"os": _Os(), "pwd": types.SimpleNamespace(getpwnam=lambda n: types.SimpleNamespace(pw_uid=4242)),
              "statmod": stat, "lockdown": lockdown, "runner": runner,
              "verify_freeze": types.SimpleNamespace(AgentFreezer=Freezer), "FREEZER": [None], "log": logs.append,
              "STAGE_DIR": str(self.tmp / "stage"), "HOME": str(self.tmp / "home"), "AGENT_USER": "user",
              "SUBMISSION": str(self.tmp / "home" / "submission"), "LATE_GATE_MAX_GAP_S": 600.0, "WATCH_ROOTS": [],
              "RUNS_ROOT": str(self.tmp / "runs"), "SNAPSHOT": str(self.tmp / "snap"), "QUARANTINE": str(self.tmp / "q")}
        exec(_functions(["_early_lockdown"]), ns)
        out = ns["_early_lockdown"]()
        self.assertEqual(len(made), 1)
        self.addCleanup(made[0].stop)
        self.assertIs(ns["FREEZER"][0], made[0])
        deadline = time.time() + 5.0
        while time.time() < deadline and self.survivor.state() != "T":
            time.sleep(0.02)
        return out, logs, made[0]

    def test_frozen_survivor_does_not_void(self):
        out, logs, freezer = self._lockdown(left=1)
        self.assertEqual(out["agent_processes_left"], 1.0)
        self.assertEqual(out["agent_processes_killed"], 7.0)
        self.assertEqual(out["reasons"], [])
        self.assertTrue(any("outlived the kill" in m for m in logs), logs)
        self.assertEqual(self.survivor.state(), "T")
        self.assertIn(self.survivor.proc.pid, freezer.frozen)

    def test_clean_kill_records_nothing(self):
        out, logs, _ = self._lockdown(left=0)
        self.assertEqual(out["agent_processes_left"], 0.0)
        self.assertEqual(out["reasons"], [])
        self.assertFalse(any("outlived" in m for m in logs), logs)

    def test_a_process_respawned_after_the_kill_is_frozen_too(self):
        """The freezer polls: a process of the account that appears after the kill is stopped as well."""
        out, _logs, freezer = self._lockdown(left=0)
        late = _Survivor()
        self.addCleanup(late.close)
        freezer._lister = lambda uid: self.survivor.pids() + late.pids()
        deadline = time.time() + 5.0
        while time.time() < deadline and late.state() != "T":
            time.sleep(0.02)
        self.assertEqual(late.state(), "T")
        self.assertEqual(out["reasons"], [])

    def test_main_gates_on_the_early_reasons_and_always_stops_the_freezer(self):
        funcs = {n.name: n for n in _tree().body if isinstance(n, ast.FunctionDef)}
        locked = ast.unparse(funcs["_main_locked"])
        self.assertIn('if ok and EARLY.get("reasons"):', locked.replace("'", '"'))
        tests = [ast.unparse(n.test) for n in ast.walk(funcs["_main_locked"]) if isinstance(n, ast.If)]
        tests += [ast.unparse(n.test) for n in ast.walk(funcs["main"]) if isinstance(n, ast.If)]
        self.assertFalse([t for t in tests if "agent_processes_left" in t], tests)
        main = funcs["main"]
        finals = [ast.unparse(s) for n in ast.walk(main) if isinstance(n, ast.Try) for s in n.finalbody]
        self.assertTrue(any("FREEZER[0].stop()" in s for s in finals), finals)
        # the runner falls back to the agent account only when `nobody` is missing: its programs must then run
        self.assertIn("FREEZER[0].stop()", locked)


if __name__ == "__main__":
    unittest.main()
