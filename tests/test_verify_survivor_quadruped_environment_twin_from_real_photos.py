"""quadruped_environment_twin_from_real_photos: a process of the agent account that outlives the verify-start kill (the
base image's supervisord respawns the desktop it runs as that account) is frozen, not held against the submission; one
that is still running once frozen still voids the run (task error). A write into the submission after the stage upload
began zeroes gate G1 (submission integrity), and the lockdown runs in its order: the stage closed before the
interpreter starts, the upload instant read, the kill, the freeze, the account freezer, the stage lock, the late-write
gate, the snapshot, all before the grader reads its configuration or starts the worker.

Runs the grader's own `_early_lockdown` (and its helpers), taken from tests/verify.py by name, against stand-ins for
the root-only parts (the kill, the account freezer, the stage lock, the snapshot) and a real child process playing the
survivor; the late-write gate also runs with the vendored lockdown's own `stage_upload_time` and `late_entries`
(tests/arm/assembly_six/lockdown.py) on real files. Stdlib only: tests/verify.py imports the vendored lockdown at
module level and the grading stack (numpy, mujoco, ...) in its worker, neither of which the host venv has.
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

TESTS = Path(__file__).resolve().parents[1] / "tasks/quadruped/quadruped_environment_twin_from_real_photos/tests"
VERIFY = TESTS / "verify.py"
TEST_SH = TESTS / "test.sh"
LOCKDOWN = TESTS / "arm/assembly_six/lockdown.py"
GRADER = ["_proc_state", "_still_running", "_submission_root_ok", "_submission_measure", "_early_lockdown", "_g1"]


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

    def state(self):
        return Path("/proc/%d/status" % self.proc.pid).read_text().split("State:")[1].split()[0]

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


def _vendored_lockdown():
    """The vendored lockdown's `stage_upload_time` and `late_entries` (stdlib only), loaded without importing the
    module, so nothing is written into the task folder."""
    ns = {"os": os, "stat": stat}
    exec(_functions(LOCKDOWN, ["stage_upload_time", "late_entries"]), ns)
    return ns["stage_upload_time"], ns["late_entries"]


def _main():
    return next(n for n in ast.parse(VERIFY.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")


class SurvivorAtVerifyStart(unittest.TestCase):
    def setUp(self):
        self.survivor = _Survivor()
        self.addCleanup(self.survivor.close)
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.tmp / "stage").mkdir()
        self.home = self.tmp / "home"
        self.submission = self.home / "submission"

    def _lockdown(self, left, freeze=True, upload_time=None, late_entries=None, events=None):
        """Run _early_lockdown with the kill reporting `left` survivors; `freeze` False: SIGSTOP has no effect.
        `upload_time` / `late_entries`: the stage-upload instant and the late-write scan (default: no upload instant,
        so the late-write gate is off, and no late entry); `events`: a list the stand-ins append their names to."""
        logs = []
        survivor = self.survivor
        stop = (lambda pid, sig: None) if not freeze else None
        started = []
        note = (events.append if events is not None else (lambda e: None))

        class Freezer:
            frozen = set()

            def start(self):
                note("freezer")
                started.append(survivor.state() if survivor.proc.poll() is None else "X")
                return self

        def stop_and_kill_uid(uid, grace_s=5.0):
            note("kill")
            return 54, left

        def stage_upload_time(d):
            note("upload_time")
            return upload_time(d) if callable(upload_time) else upload_time

        def scan_late(root, t_ref, **kw):
            note("late_entries")
            return late_entries(root, t_ref, **kw) if late_entries else []

        def lock_stage(d):
            note("lock_stage")
            return {}

        def snapshot_dir(a, b):
            note("snapshot")

        lockdown = types.SimpleNamespace(
            stop_and_kill_uid=stop_and_kill_uid, live_uid_processes=survivor.pids,
            stage_upload_time=stage_upload_time, late_entries=scan_late, snapshot_dir=snapshot_dir)
        runner = types.SimpleNamespace(lock_stage=lock_stage)
        ns = {"os": _Os(kill=stop), "pwd": types.SimpleNamespace(getpwnam=lambda n: types.SimpleNamespace(pw_uid=4242)),
              "time": time, "signal": signal, "statmod": stat, "lockdown": lockdown, "runner": runner,
              "FREEZER": Freezer(), "log": logs.append, "SUBMISSION_ENTRIES_CAP": 5000,
              "SUBMISSION_BYTES_CAP": 300 * 1024 * 1024, "LATE_GATE_MAX_GAP_S": 600.0,
              "STAGE_DIR": str(self.tmp / "stage"), "HOME": str(self.home), "AGENT_USER": "user",
              "SUBMISSION": str(self.submission), "SNAP_ROOT": str(self.tmp / "snap"),
              "SNAPSHOT": str(self.tmp / "snap" / "submission")}
        exec(_functions(VERIFY, GRADER), ns)
        out = ns["_early_lockdown"]()
        self.assertTrue(started, "the account freezer is started after the kill")
        self.started = started
        return out, logs, ns

    # --- a process that survives into verify

    def test_frozen_survivor_does_not_void(self):
        out, logs, _ = self._lockdown(left=1)
        self.assertEqual(out["agent_processes_left"], 1.0)
        self.assertEqual(out["agent_processes_running"], 0.0, out)
        self.assertTrue(any("outlived the kill" in m for m in logs), logs)
        self.assertEqual(self.survivor.state(), "T")

    def test_survivor_that_cannot_be_stopped_still_voids(self):
        out, _, _ = self._lockdown(left=1, freeze=False)
        self.assertGreater(out["agent_processes_running"], 0.0, out)

    def test_clean_kill_records_nothing(self):
        out, logs, _ = self._lockdown(left=0)
        self.assertEqual(out["agent_processes_left"], 0.0)
        self.assertEqual(out["agent_processes_running"], 0.0)
        self.assertFalse(any("outlived" in m for m in logs), logs)

    def test_only_a_running_survivor_voids(self):
        """main() voids the run (fail(), a task error) on agent_processes_running, never on the kill's raw count."""
        def voids(node):
            return any(isinstance(b, ast.Raise) or (isinstance(b, ast.Expr) and isinstance(b.value, ast.Call)
                                                   and getattr(b.value.func, "id", None) == "fail")
                       for b in node.body)
        tests = [ast.unparse(n.test) for n in ast.walk(_main()) if isinstance(n, ast.If) and voids(n)]
        self.assertFalse([t for t in tests if "agent_processes_left" in t], tests)
        self.assertTrue([t for t in tests if "agent_processes_running" in t], tests)

    # --- a write that survives into verify

    def _late_write(self, rewrite):
        """The submission written during the session, then the stage uploaded (its files' ctime is the upload instant),
        then, if `rewrite`, scene.xml rewritten in place (the folder itself unchanged), as a process left running by
        the agent would."""
        if os.geteuid() == 0:
            self.skipTest("late_entries skips root-owned entries: run as an unprivileged account")
        upload_time, late_entries = _vendored_lockdown()
        self.submission.mkdir(parents=True)
        (self.submission / "scene.xml").write_text("<mujoco/>\n")
        (self.submission / "objects.json").write_text("{}\n")
        time.sleep(0.1)
        for name in ("grader_config.json", "anchor.json"):
            (self.tmp / "stage" / name).write_text("{}\n")
        time.sleep(0.1)
        if rewrite:
            with open(self.submission / "scene.xml", "w") as fh:
                fh.write("<mujoco model='late'/>\n")
        return self._lockdown(left=0, upload_time=upload_time, late_entries=late_entries)

    def test_late_write_zeroes_g1(self):
        out, _, ns = self._late_write(rewrite=True)
        self.assertEqual(out["late_write_gate_active"], 1.0, out)
        self.assertEqual(out["submission_late_writes"], 1.0, out)
        self.assertEqual(out["submission_root_violation"], 0.0, out)
        self.assertTrue(any("written after the agent phase ended (scene.xml)" in r for r in out["reasons"]), out)
        self.assertEqual(ns["_g1"](out), 0.0)

    def test_submission_final_at_session_end_keeps_g1(self):
        out, _, ns = self._late_write(rewrite=False)
        self.assertEqual(out["late_write_gate_active"], 1.0, out)
        self.assertEqual(out["submission_late_writes"], 0.0, out)
        self.assertEqual(out["submission_snapshot"], 1.0, out)
        self.assertEqual(out["reasons"], [])
        self.assertEqual(ns["_g1"](out), 1.0)

    # --- the lockdown order

    def test_lockdown_order(self):
        """In _early_lockdown: the upload instant is read before anything chmods the stage, the kill comes first, the
        survivor is already stopped when the account freezer starts, then the stage lock, the late-write gate and the
        snapshot every later read uses."""
        self.submission.mkdir(parents=True)
        (self.submission / "scene.xml").write_text("<mujoco/>\n")
        events = []
        out, _, _ = self._lockdown(left=1, upload_time=time.time() + 3600.0, events=events)
        self.assertEqual(events, ["upload_time", "kill", "freezer", "lock_stage", "late_entries", "snapshot"])
        self.assertEqual(self.started, ["T"], "the survivor is frozen before the account freezer starts")
        self.assertEqual(out["submission_snapshot"], 1.0, out)
        self.assertEqual(out["stage_locked"], 1.0, out)

    def test_main_and_test_sh_wiring(self):
        """test.sh closes the stage and its parent before the interpreter starts; main() runs the lockdown first,
        voids on a running survivor and writes G1 = 0 before it reads its configuration or starts the worker, records
        the kill's counts and the frozen count in the verdict, and stops the freezer only after the verdict."""
        sh = TEST_SH.read_text()
        launch = sh.index('exec "${ALE_VERIFY_PYTHON:-python3}" -B verify.py')
        self.assertLess(sh.index('chmod 700 "$STAGE"'), launch)
        self.assertLess(sh.index('chmod 700 "$PARENT"'), launch)
        main = _main()
        src = ast.unparse(main)
        first = src.index("early = _early_lockdown()")
        void = src.index("if early['agent_processes_running'] > 0:")
        g1 = src.index("if g1 < 1.0:")
        config = src.index("open(os.path.join(STAGE_DIR, 'grader_config.json'))")
        worker = src.index("run_worker(cfg, sub_dir, budget, stats)")
        self.assertLess(first, void)
        self.assertLess(void, config)
        self.assertLess(config, g1)
        self.assertLess(g1, worker)
        self.assertIn("stats = {k: v for k, v in early.items() if isinstance(v, float)}", src)
        self.assertGreaterEqual(src.count("stats['agent_processes_frozen'] = float(len(FREEZER.frozen))"), 2, src)
        finals = [ast.unparse(s) for n in ast.walk(main) if isinstance(n, ast.Try) for s in n.finalbody]
        self.assertIn("_restore(early)", finals)
        self.assertIn("FREEZER.stop()", finals)


if __name__ == "__main__":
    unittest.main()
