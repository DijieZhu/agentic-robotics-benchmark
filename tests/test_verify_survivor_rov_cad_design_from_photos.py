"""rov_cad_design_from_photos: a process of the agent account that outlives the verify-start kill (the base image's
supervisord respawns the desktop it runs as that account) is frozen, not held against the submission; one that is still
running once frozen still voids the run (task error).

Runs the grader's own `_early_lockdown` (and its helpers), taken from tests/lockdown_runtime.py by name, against
stand-ins for the root-only parts (the kill, the stage lock, the snapshot) and a real child process playing the
survivor; and checks that tests/verify.py voids the run on the running count only. Stdlib only: the grading stack
(numpy, ...) that tests/verify.py imports later is not needed.

Also checks that the grade child runs on OpenBLAS's Haswell kernels with their own block sizes whatever the verify
stage's own environment says: tests/verify.py's module-level assignments of OPENBLAS_CORETYPE=Haswell and, beside it,
OPENBLAS_L2_SIZE=256 (on some AMD CPUs with 1024 KB of L2 cache per core the OpenBLAS that numpy bundles otherwise
enlarges its matrix-product blocks, for the forced kernel too, beyond what the Haswell kernel's fixed stack buffer
holds), run against other values of both, then the grader's own `grade_child` with the child stubbed.
"""
import ast
import json
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
from unittest import mock

TESTS = Path(__file__).resolve().parents[1] / "tasks/underwater/rov_cad_design_from_photos/tests"
RUNTIME = TESTS / "lockdown_runtime.py"
VERIFY = TESTS / "verify.py"


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


def _functions(path, names):
    tree = ast.parse(path.read_text())
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names), "%s lacks %s" % (path.name, set(names) - {n.name for n in body})
    return compile(ast.Module(body=body, type_ignores=[]), str(path), "exec")


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

        class Freezer:
            frozen = set()

            def start(self):
                return self

        lockdown = types.SimpleNamespace(
            stop_and_kill_uid=lambda uid, grace_s=5.0: (52, left),
            live_uid_processes=survivor.pids,
            stage_upload_time=lambda d: None, late_entries=lambda *a, **k: [],
            snapshot_dir=lambda a, b: None)
        ns = {"os": _Os(kill=stop), "pwd": types.SimpleNamespace(getpwnam=lambda n: types.SimpleNamespace(pw_uid=4242)),
              "time": time, "signal": signal, "statmod": stat, "lockdown": lockdown,
              "lock_stage": lambda d: {}, "log": logs.append, "_submission_root_ok": lambda: (True, None),
              "STAGE_DIR": str(self.tmp / "stage"), "HOME": str(self.tmp / "home"), "AGENT_USER": "user",
              "SUBMISSION": str(self.tmp / "home" / "submission"), "LATE_GATE_MAX_GAP_S": 600.0,
              "SNAP_ROOT": str(self.tmp / "snap_root"), "SNAPSHOT": str(self.tmp / "snap_root" / "submission")}
        exec(_functions(RUNTIME, ["_proc_state", "_still_running", "_early_lockdown"]), ns)
        return ns["_early_lockdown"](Freezer()), logs

    def test_frozen_survivor_does_not_void(self):
        out, logs = self._lockdown(left=1)
        self.assertEqual(out["agent_processes_left"], 1.0)
        self.assertEqual(out["agent_processes_running"], 0.0, out)
        self.assertTrue(any("outlived the kill" in m for m in logs), logs)
        self.assertEqual(Path("/proc/%d/status" % self.survivor.proc.pid).read_text().split("State:")[1].split()[0], "T")

    def test_survivor_that_cannot_be_stopped_still_voids(self):
        out, _ = self._lockdown(left=1, freeze=False)
        self.assertGreater(out["agent_processes_running"], 0, out)

    def test_clean_kill_records_nothing(self):
        out, logs = self._lockdown(left=0)
        self.assertEqual(out["agent_processes_left"], 0.0)
        self.assertEqual(out["agent_processes_running"], 0.0)
        self.assertFalse(any("outlived" in m for m in logs), logs)

    def test_only_a_running_survivor_raises(self):
        """main() voids the run on agent_processes_running, never on the kill's raw survivor count."""
        main = next(n for n in ast.parse(VERIFY.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        tests = [ast.unparse(n.test) for n in ast.walk(main)
                 if isinstance(n, ast.If) and any(isinstance(b, ast.Raise) for b in n.body)]
        self.assertFalse([t for t in tests if "agent_processes_left" in t], tests)
        self.assertTrue([t for t in tests if "agent_processes_running" in t], tests)


class GradeChildKernels(unittest.TestCase):
    """The environment grade_child hands grade.py: OpenBLAS's Haswell kernels with the L2 size beside the kernel, one
    thread each."""
    PINS = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256"}

    def _pins(self):
        """tests/verify.py's module-level assignments of the two variables (each once; nothing else names them)."""
        tree = ast.parse(VERIFY.read_text())
        pins = []
        for key, value in self.PINS.items():
            stmt = "os.environ[%r] = %r" % (key, value)
            found = [n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n) == stmt]
            self.assertEqual(len(found), 1, "tests/verify.py sets %s once, at module level" % key)
            named = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == key]
            self.assertEqual(len(named), 1, "nothing else in tests/verify.py touches %s" % key)
            pins += found
        return pins

    def test_grade_child_runs_on_the_haswell_kernels_with_their_block_sizes(self):
        # the verify stage's own values: unset, the AVX-512 kernel / the L2 size that lets the blocks grow, an empty
        # value; whatever they are, the child gets Haswell and 256
        pins = self._pins()
        for core in (None, "SkylakeX", ""):
            for l2 in (None, "1024", ""):
                with self.subTest(parent_core=core, parent_l2=l2):
                    self._check_child_env(pins, core, l2)

    def _check_child_env(self, pins, core, l2):
        tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        seen = []

        def run(argv, cwd=None, env=None, **kwargs):
            seen.append((argv, dict(env)))
            Path(argv[6]).write_text(json.dumps({"ok": True}))      # python -B grade.py EXTRACT REF PURCHASED OUT ...
            return types.SimpleNamespace(returncode=0, stderr="")

        ns = {"os": os, "sys": sys, "json": json, "time": time, "STAGE_DIR": str(tmp),
              "tempfile": types.SimpleNamespace(mkdtemp=lambda prefix="": str(tmp)),
              "subprocess": types.SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired)}
        exec(_functions(RUNTIME, ["grade_child"]), ns)
        parent = {"OPENBLAS_CORETYPE": core, "OPENBLAS_L2_SIZE": l2, "STAGE_ONLY": "kept"}
        with mock.patch.dict(os.environ, {k: v for k, v in parent.items() if v is not None}):
            for key in [k for k, v in parent.items() if v is None]:
                os.environ.pop(key, None)
            exec(compile(ast.Module(body=pins, type_ignores=[]), str(VERIFY), "exec"), {"os": os})
            res, _tail = ns["grade_child"](str(tmp / "extract"), str(tmp / "purchased"), {"grader_cpus": 8}, {})
        self.assertEqual(res, {"ok": True})
        self.assertEqual(len(seen), 1)
        argv, env = seen[0]
        self.assertEqual(argv[2], str(tmp / "grade.py"))
        self.assertEqual({k: env.get(k) for k in self.PINS}, self.PINS)
        self.assertEqual(env["STAGE_ONLY"], "kept")  # the rest of the stage's environment is kept
        self.assertEqual({k: env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                         {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertEqual(env["ARB_GRADER_CPUS"], "8")

    def test_the_grade_goes_through_grade_child(self):
        """grade.py is started in one place, the runtime's grade_child (which copies os.environ), and verify.py's
        main() calls it once, after the module-level pins."""
        main = next(n for n in ast.parse(VERIFY.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
        calls = [ast.unparse(n.func) for n in ast.walk(main) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count("runtime.grade_child"), 1, calls)
        for path, expected in ((RUNTIME, {"grade_child"}), (VERIFY, set())):
            tree = ast.parse(path.read_text())
            holders = {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef)
                       for c in ast.walk(f) if isinstance(c, ast.Constant) and c.value == "grade.py"}
            self.assertEqual(holders, expected, path.name)


if __name__ == "__main__":
    unittest.main()
