"""freeflyer_environment_twin_from_flight_images: the verify-start lockdown, the submission's integrity gate G1, the
grading worker's environment and the reward's source, each run from the grader's own code.

A process of the agent account that outlives the verify-start kill (the base image's supervisord respawns the desktop it
runs as that account) is frozen, not held against the submission; one that is still running once frozen voids the run
(task error). The lockdown runs in its order: the upload instant read before anything chmods the stage, the kill, the
account freezer, the stage lock, the configuration's caps, and only then is the submission walked, scanned for late
writes and copied into the snapshot that every later read uses. Gate G1 is 0 for a file written into the submission
after the stage upload began, for a submission root that is a symbolic link, and for a root replaced by another folder
after the upload began; a root replaced after the lockdown is never read (the worker gets the snapshot). The late-write
scan is active only while the upload instant can be trusted (grader_config.json's ctime within
runtime.late_gate_max_gap_s of it): outside that the scan is off and says so in the verdict's late_write_gate_active.
When G1 is 0 nothing of the submission reaches the worker, which still grades the reference, and the reward is 0.

The grading worker runs on OpenBLAS's Haswell kernels with the L2 size beside the kernel and numpy's AVX-512 code paths
off (FP_PINNING), one thread each, in an environment built from nothing, whatever the verify stage's own environment
says; the worker is started in one place, and its ray-scene child is a fork of it, so it inherits the same values. A
worker that ends in any other way than finishing its grade (an exit code, a signal, the hard timeout, a failed
self-test, an internal error) is a task error with this grader's wording, never a score. The reward is G1 x the
worker's twin_score, handed once to the kit's ratio_rewards against anchor.json, which must equal the configuration's
anchor; the verdict keeps whitelisted, finite metrics only.

The grader's functions are taken from tests/verify.py by name through its syntax tree (a name that is not there fails
the test) and run against stand-ins for the root-only parts (the kill, the account freezer, the stage lock, runuser,
the worker's start) with a real child process playing the survivor; the late-write gate and the snapshot run with the
vendored lockdown's own functions (tests/arm/) on real files. Nothing is imported from the task folder, so nothing is
written into it. Stdlib only: tests/verify.py imports the vendored lockdown at module level and the grading stack
(numpy, mujoco, ...) in its worker, neither of which the host venv has. No process other than the test's own child is
ever signalled: the stand-in for os.kill refuses every other id.
"""
import ast
import json
import math
import os
import shutil
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

TASK = Path(__file__).resolve().parents[1] / "tasks/space/freeflyer_environment_twin_from_flight_images"
TESTS = TASK / "tests"
VERIFY = TESTS / "verify.py"
TEST_SH = TESTS / "test.sh"
CONFIG = TESTS / "grader_config.json"
ANCHOR = TESTS / "anchor.json"
LOCKDOWN = TESTS / "arm/assembly_six/lockdown.py"
RUNNER = TESTS / "arm/assembly_cell/runner.py"
METRICS = TESTS / "twin/metrics.py"
TWIN = TESTS / "twin"
GRADE = TESTS / "twin/grade.py"
LOCKDOWN_FUNCTIONS = ["_read_config_doc", "_caps", "_submission_root_ok", "submission_measure", "_proc_state",
                      "_still_running", "_early_lockdown", "_g1"]
MAIN_FUNCTIONS = ["_g1", "_budget", "_diagnostic", "_timing", "main"]


def _tree(path):
    return ast.parse(path.read_text())


def _functions(path, names):
    """The module-level functions `names` of `path`, compiled from its syntax tree; a name that is missing fails."""
    body = [n for n in _tree(path).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in body} == set(names), "%s lacks %s" % (path.name, sorted(set(names) - {n.name for n in body}))
    return compile(ast.Module(body=body, type_ignores=[]), str(path), "exec")


def _function_node(path, name):
    nodes = [n for n in _tree(path).body if isinstance(n, ast.FunctionDef) and n.name == name]
    assert len(nodes) == 1, "%s lacks %s" % (path.name, name)
    return nodes[0]


def _literal(name):
    """A module-level literal of tests/verify.py (FP_PINNING, KIT_METRICS)."""
    nodes = [n for n in _tree(VERIFY).body
             if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == [name]]
    assert len(nodes) == 1, "verify.py lacks the literal %s" % name
    return ast.literal_eval(nodes[0].value)


def _module(name, path, names=None):
    """A stdlib-only file of the task as a module object, compiled from its source (no import, no bytecode written);
    with `names`, only those functions of it."""
    mod = types.ModuleType(name)
    mod.__dict__.update({"os": os, "stat": stat, "shutil": shutil, "math": math})
    code = _functions(path, names) if names else compile(path.read_text(), str(path), "exec")
    exec(code, mod.__dict__)
    return mod


def _config():
    return json.loads(CONFIG.read_text())


class _Os(types.ModuleType):
    """os with getuid() as given (0: the root path) and a kill that only ever reaches the test's own child."""

    def __init__(self, allowed_pid, uid=0, kill=None):
        super().__init__("os")
        self._allowed_pid = allowed_pid
        self._uid = uid
        self._kill = kill

    def __getattr__(self, name):
        return getattr(os, name)

    def getuid(self):
        return self._uid

    def kill(self, pid, sig):
        if pid != self._allowed_pid:
            raise AssertionError("the grader signalled pid %r, which is not the test's own child" % (pid,))
        return (self._kill or os.kill)(pid, sig)

    def killpg(self, pgid, sig):
        raise AssertionError("the grader signalled the process group %r" % (pgid,))


class _Survivor:
    """A real process standing in for the respawned desktop process that outlived the kill."""

    def __init__(self):
        self.proc = subprocess.Popen(["sleep", "60"])
        self.pid = self.proc.pid                    # recorded at the start; the only id this test ever signals

    def pids(self, uid=None):
        return [self.pid] if self.proc.poll() is None else []

    def state(self):
        return Path("/proc/%d/status" % self.pid).read_text().split("State:")[1].split()[0]

    def _signal(self, sig):
        if isinstance(self.pid, int) and self.pid > 1000 and self.pid != os.getpid() and self.proc.poll() is None:
            os.kill(self.pid, sig)
            return True
        return False

    def close(self):
        self._signal(signal.SIGCONT)
        if self._signal(signal.SIGKILL):
            self.proc.wait(timeout=30)


class _Fail(Exception):
    pass


class _Base(unittest.TestCase):
    def setUp(self):
        self.survivor = _Survivor()
        self.addCleanup(self.survivor.close)
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.stage = self.tmp / "stage"
        self.stage.mkdir()
        self.home = self.tmp / "home"
        self.home.mkdir()
        self.submission = self.home / "submission"
        self.snap_root = self.tmp / "snap"
        self.snapshot = self.snap_root / "submission"
        self.doc = _config()

    # --- the stage and the submission on real files

    def _upload_stage(self):
        """The engine's upload of the verify stage: its files are written anew, so their ctime is the upload instant."""
        shutil.copyfile(CONFIG, self.stage / "grader_config.json")
        shutil.copyfile(ANCHOR, self.stage / "anchor.json")

    def _write_submission(self, root=None, text="<mujoco/>\n"):
        root = Path(root or self.submission)
        (root / "flight_a").mkdir(parents=True)
        (root / "flight_a" / "scene.xml").write_text(text)
        (root / "flight_a" / "cameras.json").write_text("{}\n")
        return root

    @staticmethod
    def _contents(root):
        out = {}
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                p = os.path.join(dirpath, name)
                out[os.path.relpath(p, root)] = Path(p).read_text()
        return out

    # --- the grader's _early_lockdown

    def _lockdown(self, left=0, freeze=True, upload_time=None, late_entries=None, events=None, real_snapshot=False,
                  uid=0):
        """Run _early_lockdown with the kill reporting `left` survivors; `freeze` False: SIGSTOP has no effect.
        `upload_time` / `late_entries`: the stage-upload instant (a value or a function of the stage) and the late-write
        scan (default: no upload instant, so the late-write gate is off, and no late entry); `events`: a list the
        stand-ins append their names to; `real_snapshot`: the vendored snapshot_dir on real files; `uid`: getuid()."""
        if not (self.stage / "grader_config.json").exists():
            self._upload_stage()
        logs = []
        survivor = self.survivor
        stop = (lambda pid, sig: None) if not freeze else None
        started = []
        note = (events.append if events is not None else (lambda e: None))
        vendored_snapshot = self._vendored_snapshot() if real_snapshot else None

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
            return []

        def snapshot_dir(a, b):
            note("snapshot")
            return vendored_snapshot(a, b) if vendored_snapshot else 0

        lockdown = types.SimpleNamespace(
            stop_and_kill_uid=stop_and_kill_uid, live_uid_processes=survivor.pids,
            stage_upload_time=stage_upload_time, late_entries=scan_late, snapshot_dir=snapshot_dir)
        runner = types.SimpleNamespace(lock_stage=lock_stage)
        ns = {"os": _Os(survivor.pid, uid=uid, kill=stop), "json": json, "time": time, "signal": signal,
              "statmod": stat, "pwd": types.SimpleNamespace(getpwnam=lambda n: types.SimpleNamespace(pw_uid=4242)),
              "lockdown": lockdown, "runner": runner, "FREEZER": Freezer(), "log": logs.append,
              "STAGE_DIR": str(self.stage), "HOME": str(self.home), "AGENT_USER": "user",
              "SUBMISSION": str(self.submission), "SNAP_ROOT": str(self.snap_root), "SNAPSHOT": str(self.snapshot)}
        exec(_functions(VERIFY, LOCKDOWN_FUNCTIONS), ns)
        read_config, measure = ns["_read_config_doc"], ns["submission_measure"]

        def noted_config():
            note("config")
            return read_config()

        def noted_measure(*a, **kw):
            note("measure")
            return measure(*a, **kw)

        ns["_read_config_doc"], ns["submission_measure"] = noted_config, noted_measure
        out = ns["_early_lockdown"]()
        if uid == 0:
            self.assertTrue(started, "the account freezer is started after the kill")
        self.started = started
        return out, logs, ns

    @staticmethod
    def _vendored_lockdown():
        """The vendored lockdown's `stage_upload_time` and `late_entries`, compiled from its source."""
        mod = _module("lockdown", LOCKDOWN, ["stage_upload_time", "late_entries"])
        return mod.stage_upload_time, mod.late_entries

    @staticmethod
    def _vendored_snapshot():
        """The vendored lockdown's `snapshot_dir` with the vendored runner's copy and chmod, compiled from source."""
        runner = _module("arm.assembly_cell.runner", RUNNER, ["copy_regular_files", "chmod_tree"])
        arm = types.ModuleType("arm")
        cell = types.ModuleType("arm.assembly_cell")
        arm.assembly_cell, cell.runner = cell, runner
        mod = _module("lockdown", LOCKDOWN, ["snapshot_dir"])

        def snapshot_dir(src, dst):
            with mock.patch.dict(sys.modules, {"arm": arm, "arm.assembly_cell": cell,
                                               "arm.assembly_cell.runner": runner}):
                return mod.snapshot_dir(src, dst)
        return snapshot_dir

    def _staged(self, late=None, upload_shift=0.0):
        """The submission written during the session, then the stage uploaded (its files' ctime is the upload instant),
        then `late` (a function) run as a process left running by the agent would; `upload_shift` moves the upload
        instant the lockdown reads (negative: the instant lies further before the configuration's ctime)."""
        if os.geteuid() == 0:
            self.skipTest("late_entries skips root-owned entries: run as an unprivileged account")
        upload_time, late_entries = self._vendored_lockdown()
        self._write_submission()
        time.sleep(0.1)
        self._upload_stage()
        time.sleep(0.1)
        if late:
            late()
        return self._lockdown(upload_time=lambda d: upload_time(d) + upload_shift, late_entries=late_entries,
                              real_snapshot=True)

    # --- the grader's main()

    def _main(self, early, worker=None, metrics_calls=None):
        """Run main() with `early` as the lockdown's result and `worker(doc, sub_dir, budget, stats)` standing in for
        run_worker (default: a finished grade of 0.5 under the configured aggregation). Returns the record."""
        rec = {"worker": [], "verdict": [], "fail": [], "restore": 0, "freezer_stop": 0, "config_reads": 0}

        def fail(reason):
            rec["fail"].append(reason)
            raise _Fail(reason)

        kit = types.ModuleType("robotics_grader")
        kit.fail = fail
        metrics = _module("twin.metrics", METRICS)
        real_metrics_of = metrics.metrics_of

        def metrics_of(grade, stats, flights):
            # the lockdown's and the worker's statistics as the verdict carries them (the grade's own terms are not
            # this test's concern)
            if metrics_calls is not None:
                metrics_calls.append((grade, dict(stats)))
            return real_metrics_of(None, stats, flights)
        metrics.metrics_of = metrics_of
        config = types.ModuleType("twin.config")
        config.ConfigError = type("ConfigError", (Exception,), {})
        config.from_document = lambda doc: None
        twin = types.ModuleType("twin")
        twin.config, twin.metrics = config, metrics

        def default_worker(doc, sub_dir, budget, stats):
            return ({"selftest": {"ok": True}, "done": True, "submission": _grade(doc, 0.5, 0.75)}, "finished", "")

        def run_worker(doc, sub_dir, budget, stats):
            rec["worker"].append(sub_dir)
            if sub_dir is not None:
                rec.setdefault("worker_read", []).append(self._contents(sub_dir))
            return (worker or default_worker)(doc, sub_dir, budget, stats)

        def write_verdict(overall, m, diagnostic, doc):
            rec["verdict"].append((overall, dict(m), diagnostic))

        def read_config():
            rec["config_reads"] += 1
            return _config()

        def restore(e):
            rec["restore"] += 1

        class Freezer:
            frozen = {4242}

            def stop(self):
                rec["freezer_stop"] += 1

        ns = {"os": os, "sys": sys, "time": time, "json": json, "math": math, "_early_lockdown": lambda: early,
              "_read_config_doc": read_config, "FREEZER": Freezer(), "run_worker": run_worker,
              "write_verdict": write_verdict, "_restore": restore, "log": lambda msg: None,
              "STAGE_DIR": str(self.stage), "HOME": str(self.home), "SUBMISSION": str(self.submission),
              "SNAP_ROOT": str(self.snap_root), "SNAPSHOT": str(self.snapshot),
              "NONDETERMINISTIC": _literal("NONDETERMINISTIC")}
        exec(_functions(VERIFY, MAIN_FUNCTIONS), ns)
        env = {k: v for k, v in os.environ.items()
               if k not in ("ARB_TWIN_TIMING_PATH", "ARB_TWIN_GRADE_PATH", "ALE_VERIFY_DEADLINE_SECONDS")}
        with mock.patch.dict(sys.modules, {"robotics_grader": kit, "twin": twin, "twin.config": config,
                                           "twin.metrics": metrics}), \
                mock.patch.object(sys, "path", list(sys.path)), mock.patch.dict(os.environ, env, clear=True):
            try:
                ns["main"]()
            except _Fail:
                pass
        return rec


def _grade(doc, score, total_sum):
    """A worker's grade of the submission with the configured twin_score `score` and the sum aggregation `total_sum`."""
    terms = {"score": score, "sum": total_sum, "product": score, "D": score, "H": 1.0}
    flights = {f: {"G0": 1.0, "terms": terms, "reasons": []} for f in doc["values"]["flights.names_neutral"]}
    return {"twin_score": score, "twin_score_sum": total_sum, "twin_score_product": score, "flights": flights}


class SurvivorAtVerifyStart(_Base):
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
        main = _function_node(VERIFY, "main")
        tests = [ast.unparse(n.test) for n in ast.walk(main) if isinstance(n, ast.If) and voids(n)]
        self.assertFalse([t for t in tests if "agent_processes_left" in t], tests)
        self.assertTrue([t for t in tests if "agent_processes_running" in t], tests)

    def test_a_running_survivor_voids_before_anything_is_read(self):
        out, _, _ = self._lockdown(left=1, freeze=False)
        rec = self._main(out)
        self.assertEqual(len(rec["fail"]), 1, rec)
        self.assertIn("survived the verify-start kill and could not be stopped: run void", rec["fail"][0])
        self.assertEqual((rec["config_reads"], rec["worker"], rec["verdict"]), (0, [], []), rec)
        self.assertEqual((rec["restore"], rec["freezer_stop"]), (1, 1), rec)
        frozen = self._main(self._lockdown(left=1)[0])
        self.assertEqual(frozen["fail"], [], frozen)
        self.assertEqual(len(frozen["verdict"]), 1, frozen)

    def test_the_freezer_is_the_vendored_one_for_the_agent_account(self):
        assigns = {ast.unparse(n.value) for n in _tree(VERIFY).body
                   if isinstance(n, ast.Assign) and [ast.unparse(t) for t in n.targets] == ["FREEZER"]}
        self.assertEqual(assigns, {"verify_freeze.AgentFreezer.for_user(AGENT_USER)"})
        imports = {ast.unparse(n) for n in _tree(VERIFY).body if isinstance(n, ast.ImportFrom)}
        self.assertIn("from arm import verify_freeze", imports)

    # --- the lockdown order

    def test_lockdown_order(self):
        """In _early_lockdown: the upload instant is read before anything chmods the stage, the kill comes first, the
        survivor is already stopped when the account freezer starts, then the stage lock and the configuration's caps;
        only then is the submission walked, scanned for late writes and copied into the snapshot every later read
        uses."""
        self._write_submission()
        events = []
        out, _, _ = self._lockdown(left=1, upload_time=time.time() + 3600.0, events=events)
        self.assertEqual(events, ["upload_time", "kill", "freezer", "lock_stage", "config", "measure", "late_entries",
                                  "snapshot"])
        self.assertEqual(self.started, ["T"], "the survivor is frozen before the account freezer starts")
        self.assertEqual(out["late_write_gate_active"], 1.0, out)
        self.assertEqual(out["submission_snapshot"], 1.0, out)
        self.assertEqual(out["stage_locked"], 1.0, out)

    def test_main_and_test_sh_wiring(self):
        """test.sh records the modes and closes the stage and its parent before the interpreter starts; main() runs the
        lockdown first, voids on a running survivor before it reads its configuration, computes G1 and picks what the
        worker reads before it starts the worker, records the frozen count, and restores the stage and stops the
        freezer in its finally."""
        sh = TEST_SH.read_text()
        launch = sh.index('exec "${ALE_VERIFY_PYTHON:-python3}" -B verify.py')
        self.assertLess(sh.index('export ARB_STAGE_MODE="$(stat -c %a "$STAGE")"'), sh.index('chmod 700 "$STAGE"'))
        self.assertLess(sh.index('export ARB_STAGE_PARENT_MODE="$(stat -c %a "$PARENT")"'),
                        sh.index('chmod 700 "$PARENT"'))
        self.assertLess(sh.index('chmod 700 "$STAGE"'), launch)
        self.assertLess(sh.index('chmod 700 "$PARENT"'), launch)
        main = _function_node(VERIFY, "main")
        src = ast.unparse(main)
        first = src.index("early = _early_lockdown()")
        void = src.index("if early['agent_processes_running'] > 0:")
        config = src.index("doc = _read_config_doc()")
        g1 = src.index("g1 = _g1(early)")
        pick = src.index("sub_dir = SNAPSHOT if early['submission_snapshot'] else")
        worker = src.index("run_worker(doc, sub_dir, budget, stats)")
        self.assertLess(first, void)
        self.assertLess(void, config)
        self.assertLess(config, g1)
        self.assertLess(g1, pick)
        self.assertLess(pick, worker)
        nothing = [n for n in ast.walk(main) if isinstance(n, ast.If) and ast.unparse(n.test) == "g1 < 1.0"
                   and [ast.unparse(b) for b in n.body] == ["sub_dir = None"]]
        self.assertEqual(len(nothing), 1, "with G1 0 nothing of the submission reaches the worker")
        self.assertIn("stats = {k: v for k, v in early.items() if isinstance(v, float)}", src)
        self.assertIn("stats['agent_processes_frozen'] = float(len(FREEZER.frozen))", src)
        finals = [ast.unparse(s) for n in ast.walk(main) if isinstance(n, ast.Try) for s in n.finalbody]
        self.assertIn("_restore(early)", finals)
        self.assertIn("FREEZER.stop()", finals)


class SubmissionIntegrity(_Base):
    """Gate G1 on real files, with the vendored lockdown's own stage_upload_time, late_entries and snapshot_dir."""

    def _rewrite(self):
        with open(self.submission / "flight_a" / "scene.xml", "w") as fh:
            fh.write("<mujoco model='late'/>\n")

    # --- a write that survives into verify

    def test_late_write_zeroes_g1(self):
        out, _, ns = self._staged(late=self._rewrite)
        self.assertEqual(out["late_write_gate_active"], 1.0, out)
        self.assertEqual(out["submission_late_writes"], 1.0, out)
        self.assertEqual(out["submission_root_violation"], 0.0, out)
        self.assertTrue(any("written after the agent phase ended (flight_a/scene.xml)" in r for r in out["reasons"]),
                        out)
        self.assertEqual(ns["_g1"](out), 0.0)
        rec = self._main(out)
        self.assertEqual(rec["worker"], [None], "nothing of the submission reaches the worker; it grades the reference")
        self.assertEqual(len(rec["verdict"]), 1, rec)
        overall, m, diagnostic = rec["verdict"][0]
        self.assertEqual((overall, m["twin_score"], m["G1"], m["submission_late_writes"]), (0.0, 0.0, 0.0, 1.0), m)
        self.assertTrue(diagnostic.startswith("G1 0: "), diagnostic)

    def test_late_file_added_zeroes_g1(self):
        out, _, ns = self._staged(late=lambda: (self.submission / "flight_a" / "extra.obj").write_text("v 0 0 0\n"))
        self.assertEqual(out["late_write_gate_active"], 1.0, out)
        self.assertGreaterEqual(out["submission_late_writes"], 1.0, out)
        self.assertEqual(ns["_g1"](out), 0.0)

    def test_submission_final_at_session_end_keeps_g1(self):
        out, _, ns = self._staged()
        self.assertEqual(out["late_write_gate_active"], 1.0, out)
        self.assertEqual(out["submission_late_writes"], 0.0, out)
        self.assertEqual(out["submission_snapshot"], 1.0, out)
        self.assertEqual(out["reasons"], [])
        self.assertEqual(ns["_g1"](out), 1.0)
        rec = self._main(out)
        self.assertEqual(rec["worker"], [str(self.snapshot)], "the worker reads the snapshot, not the live folder")
        self.assertEqual(rec["worker_read"], [self._contents(self.submission)])

    def test_late_gate_switches_off_when_the_upload_instant_is_not_to_be_trusted(self):
        """The gate's own condition: grader_config.json's ctime more than runtime.late_gate_max_gap_s after the
        earliest ctime of the stage means the upload instant is not trusted, and then neither the late-write scan nor
        the root-replacement check runs: a late write is graded, and late_write_gate_active 0 in the verdict says
        so. This pins the behaviour as it is."""
        gap = float(self.doc["runtime"]["late_gate_max_gap_s"])
        out, _, ns = self._staged(late=self._rewrite, upload_shift=-(gap + 60.0))
        self.assertEqual(out["late_write_gate_active"], 0.0, out)
        self.assertEqual((out["submission_late_writes"], out["submission_root_violation"]), (0.0, 0.0), out)
        self.assertEqual(out["submission_snapshot"], 1.0, out)
        self.assertEqual(ns["_g1"](out), 1.0)
        rec = self._main(out)
        self.assertEqual(rec["worker"], [str(self.snapshot)])
        self.assertIn("late", rec["worker_read"][0]["flight_a/scene.xml"])
        m = rec["verdict"][0][1]
        self.assertEqual((m["G1"], m["late_write_gate_active"], m["submission_late_writes"]), (1.0, 0.0, 0.0), m)

    # --- a replaced submission root

    def test_a_root_that_is_a_symbolic_link_is_rejected(self):
        target = self._write_submission(self.home / "elsewhere")
        self.submission.symlink_to(target, target_is_directory=True)
        self._upload_stage()
        for gate in (3600.0, None):
            with self.subTest(upload_instant=gate):
                out, _, ns = self._lockdown(upload_time=(time.time() + gate) if gate else None)
                self.assertEqual(out["submission_root_rejected"], 1.0, out)
                self.assertTrue(any("is a symbolic link: not read" in r for r in out["reasons"]), out)
                self.assertEqual(out["submission_snapshot"], 0.0, out)
                self.assertEqual(ns["_g1"](out), 0.0)
                rec = self._main(out)
                self.assertEqual(rec["worker"], [None])
                self.assertEqual(rec["verdict"][0][0], 0.0)

    def test_a_root_replaced_by_another_folder_after_the_upload_began_is_a_violation(self):
        def swap():
            fresh = self._write_submission(self.home / "fresh", text="<mujoco model='swapped'/>\n")
            os.rename(self.submission, self.home / "old")
            os.rename(fresh, self.submission)
        out, _, ns = self._staged(late=swap)
        self.assertEqual(out["late_write_gate_active"], 1.0, out)
        self.assertEqual(out["submission_root_violation"], 1.0, out)
        self.assertTrue(any("the submission directory itself was replaced or changed after the agent phase ended" in r
                            for r in out["reasons"]), out)
        self.assertEqual(ns["_g1"](out), 0.0)
        rec = self._main(out)
        self.assertEqual(rec["worker"], [None])
        self.assertEqual(rec["verdict"][0][0], 0.0)

    def test_a_root_replaced_after_the_lockdown_is_never_read(self):
        """Once the lockdown returned, main() hands the worker the root-owned snapshot: a link or another folder put in
        place of the submission root afterwards is never read."""
        out, _, ns = self._staged()
        self.assertEqual(ns["_g1"](out), 1.0)
        taken = self._contents(self.snapshot)
        self.assertEqual(taken, self._contents(self.submission))
        other = self._write_submission(self.home / "other", text="<mujoco model='swapped'/>\n")
        os.rename(self.submission, self.home / "old")
        for kind in ("link", "folder"):
            with self.subTest(replaced_by=kind):
                if kind == "link":
                    self.submission.symlink_to(other, target_is_directory=True)
                else:
                    shutil.copytree(other, self.submission)
                rec = self._main(out)
                self.assertEqual(rec["worker"], [str(self.snapshot)])
                self.assertEqual(rec["worker_read"], [taken])
                self.assertNotIn("swapped", json.dumps(rec["worker_read"]))
                if kind == "link":
                    self.submission.unlink()
                else:
                    shutil.rmtree(self.submission)


def _run_worker(case, tmp, parent_env=None, pinning=None, returncode=0, timed_out=False):
    """Run the grader's run_worker with `parent_env` as the verify stage's environment and `pinning` as FP_PINNING
    (default: the one tests/verify.py defines); the worker's start is a stand-in that ends with `returncode` or, with
    `timed_out`, runs past the hard timeout. Returns (environment, argv, how, stopped)."""
    started, stopped = [], []
    work = Path(tmp) / "work"

    def prepare_work(doc, sub_dir):
        for d in ("out", "home"):
            (work / d).mkdir(parents=True)
        return str(work), {"progress": str(work / "out" / "progress.json"), "code": str(work / "code")}, []

    class Proc:
        pid = 0                           # never a real process; _stop_worker is a stand-in and signals nothing

        def __init__(self, argv, **kwargs):
            self.returncode = None
            started.append((argv, kwargs))

        def wait(self, timeout=None):
            if timed_out and timeout is not None:
                raise subprocess.TimeoutExpired("worker", timeout)
            self.returncode = -9 if timed_out else returncode
            return self.returncode

    ns = {"os": os, "sys": sys, "json": json, "time": time, "shutil": shutil,
          "subprocess": types.SimpleNamespace(Popen=Proc, STDOUT=subprocess.STDOUT,
                                              TimeoutExpired=subprocess.TimeoutExpired),
          "runner": types.SimpleNamespace(runner_account=lambda name, default: None),
          "_prepare_work": prepare_work, "_read_progress": lambda path: {"selftest": {"ok": True}},
          "_stop_worker": stopped.append, "log": lambda msg: None,
          "FP_PINNING": _literal("FP_PINNING") if pinning is None else pinning}
    exec(_functions(VERIFY, ["run_worker"]), ns)
    doc = {"runtime": _config()["runtime"]}
    parent_env = parent_env or {}
    with mock.patch.dict(os.environ, {k: v for k, v in parent_env.items() if v is not None}):
        for key in [k for k, v in parent_env.items() if v is None]:
            os.environ.pop(key, None)
        out, how, _ = ns["run_worker"](doc, None, 600.0, {})
    case.assertEqual(out, {"selftest": {"ok": True}})
    case.assertEqual(len(started), 1)
    argv, kwargs = started[0]
    case.assertIn("from twin import worker", argv[-1])
    case.assertIn("worker.worker_main(", argv[-1])
    case.assertTrue(kwargs.get("start_new_session"), kwargs)
    return kwargs["env"], argv, how, stopped


class WorkerKernels(unittest.TestCase):
    """The environment run_worker hands the grading worker: OpenBLAS pinned to its Haswell kernels with the L2 size
    beside the kernel, numpy's AVX-512 code paths off, one thread each, built from FP_PINNING and nothing else."""

    PINNED = {"NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR", "OPENBLAS_CORETYPE": "Haswell",
              "OPENBLAS_L2_SIZE": "256"}

    def setUp(self):
        self.tmp = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def _run_worker(self, parent_env=None, pinning=None):
        return _run_worker(self, self.tmp, parent_env, pinning)

    def test_fp_pinning_holds_the_three_variables(self):
        self.assertEqual(_literal("FP_PINNING"), self.PINNED)

    def test_the_worker_runs_on_the_haswell_kernels_with_their_block_sizes(self):
        # the verify stage's own values: unset, the AVX-512 kernel / the L2 size that lets the blocks grow, an empty
        # value; whatever they are, the worker gets Haswell and 256
        for core in (None, "SkylakeX", ""):
            for l2 in (None, "1024", ""):
                with self.subTest(parent_core=core, parent_l2=l2):
                    env, _, how, _ = self._run_worker({"OPENBLAS_CORETYPE": core, "OPENBLAS_L2_SIZE": l2,
                                                       "NPY_DISABLE_CPU_FEATURES": None, "STAGE_ONLY": "x"})
                    self.assertEqual(how, "finished")
                    self.assertEqual({k: env.get(k) for k in self.PINNED}, self.PINNED)
                    self.assertEqual({k: env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                          "MKL_NUM_THREADS")},
                                     {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                    self.assertNotIn("STAGE_ONLY", env)  # the worker's environment is built from nothing

    def test_the_worker_environment_comes_from_fp_pinning(self):
        """Whatever FP_PINNING holds reaches the worker, and nothing else sets these variables: an entry dropped from
        FP_PINNING is dropped from the worker (and the test above fails)."""
        env, _, _, _ = self._run_worker({}, pinning={"ARB_PINNING_PROBE": "1"})
        self.assertEqual(env.get("ARB_PINNING_PROBE"), "1")
        self.assertEqual({k: env.get(k) for k in self.PINNED}, {k: None for k in self.PINNED})

    def test_the_worker_is_started_in_one_place(self):
        """The worker's program is started in one place, run_worker, and main() calls it once."""
        main = _function_node(VERIFY, "main")
        calls = [ast.unparse(n.func) for n in ast.walk(main) if isinstance(n, ast.Call)]
        self.assertEqual(calls.count("run_worker"), 1, calls)
        tree = _tree(VERIFY)
        holders = {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) for c in ast.walk(f)
                   if isinstance(c, ast.Constant) and isinstance(c.value, str) and "worker_main(" in c.value}
        self.assertEqual(holders, {"run_worker"})
        popen = {f.name for f in ast.walk(tree) if isinstance(f, ast.FunctionDef) for c in ast.walk(f)
                 if isinstance(c, ast.Call) and ast.unparse(c.func) == "subprocess.Popen"}
        self.assertEqual(popen, {"run_worker"})

    def test_the_ray_scene_child_inherits_the_worker_environment(self):
        """The grading code under tests/twin/ starts no process but the ray scene's child, a fork of the worker (so it
        holds the worker's pinned values), and nothing there writes the environment."""
        starts, writes = [], []
        for path in sorted(TWIN.rglob("*.py")):
            for node in ast.walk(_tree(path)):
                if isinstance(node, ast.Call):
                    name = ast.unparse(node.func)
                    if (name.startswith(("subprocess.", "os.exec", "os.spawn", "os.posix_spawn", "multiprocessing."))
                            or name in ("os.fork", "os.forkpty", "os.system", "os.popen")):
                        starts.append((path.relative_to(TESTS).as_posix(), name))
                    if name in ("os.putenv", "os.unsetenv") or name.startswith(("os.environ.update", "os.environ.pop",
                                                                                "os.environ.setdefault")):
                        writes.append((path.relative_to(TESTS).as_posix(), name))
                if isinstance(node, (ast.Assign, ast.AugAssign, ast.Delete)):
                    targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
                    writes += [(path.relative_to(TESTS).as_posix(), ast.unparse(t)) for t in targets
                               if ast.unparse(t).startswith("os.environ")]
        self.assertEqual(starts, [("twin/grade.py", "os.fork")])
        self.assertEqual(writes, [])
        scene = _function_node(GRADE, "scene_in_child")
        self.assertIn("os.fork", [ast.unparse(n.func) for n in ast.walk(scene) if isinstance(n, ast.Call)])


class WorkerEndText(_Base):
    """How a worker's end is worded: run_worker reads 'finished', 'exit code N' (a negative N is a signal) or 'killed
    at the hard timeout'; main() turns every end but a finished grade into a task error with this grader's wording,
    never a verdict."""

    def test_run_worker_words_the_end(self):
        for returncode, timed_out, how in ((0, False, "finished"), (-11, False, "exit code -11"),
                                           (139, False, "exit code 139"), (3, False, "exit code 3"),
                                           (0, True, "killed at the hard timeout")):
            with self.subTest(returncode=returncode, timed_out=timed_out):
                _, _, got, stopped = _run_worker(self, self.tmp, {}, returncode=returncode, timed_out=timed_out)
                self.assertEqual(got, how)
                self.assertEqual(len(stopped), 1 if timed_out else 0)

    def test_main_turns_every_other_end_into_a_task_error(self):
        early = self._lockdown()[0]
        grade = _grade(self.doc, 0.5, 0.75)
        cases = (
            (None, "exit code -11", "the grader's worker produced no self-test result (exit code -11): tail"),
            ({"selftest": {"ok": True}, "done": False}, "exit code -11",
             "the worker was exit code -11 before the submission was graded: the verification is void"),
            ({"selftest": {"ok": True}, "done": False}, "exit code 139",
             "the worker was exit code 139 before the submission was graded: the verification is void"),
            ({"selftest": {"ok": True}, "done": True, "submission": grade}, "exit code 3",
             "the worker was exit code 3 before the submission was graded: the verification is void"),
            ({"selftest": {"ok": True}, "done": False}, "killed at the hard timeout",
             "the worker was killed at the hard timeout before the submission was graded: the verification is void"),
            ({"selftest": {"ok": False, "reason": "a term is not 1"}}, "exit code 3",
             "the reference or the truth failed the self-test (exit code 3): a term is not 1"),
            ({"selftest": {"ok": True}, "task_error": {"reason": "an internal error"}}, "exit code 3",
             "the verification is void (exit code 3): an internal error"))
        for out, how, text in cases:
            with self.subTest(how=how, text=text):
                rec = self._main(early, worker=lambda d, s, b, st, out=out, how=how: (out, how, "tail"))
                self.assertEqual(rec["fail"], [text])
                self.assertEqual(rec["verdict"], [])
                self.assertEqual((rec["restore"], rec["freezer_stop"]), (1, 1), rec)


class RewardSource(_Base):
    """The reward is G1 x the worker's twin_score (the configured aggregation), written once through write_verdict,
    which hands it to the kit's ratio_rewards against anchor.json; anchor.json must equal the configuration's anchor and
    the verdict keeps the whitelist's finite metrics only."""

    def test_the_reward_is_g1_times_the_graded_twin_score(self):
        main = _function_node(VERIFY, "main")
        overall = [ast.unparse(n.value) for n in ast.walk(main) if isinstance(n, ast.Assign)
                   and [ast.unparse(t) for t in n.targets] == ["overall"]]
        self.assertEqual(overall, ["g1 * float(grade['twin_score'])"])
        verdicts = [ast.unparse(n.args[0]) for n in ast.walk(main) if isinstance(n, ast.Call)
                    and ast.unparse(n.func) == "write_verdict"]
        self.assertEqual(sorted(verdicts), ["0.0", "overall"])
        out = self._lockdown()[0]
        rec = self._main(out)
        self.assertEqual(rec["fail"], [], rec)
        self.assertEqual(len(rec["verdict"]), 1, rec)
        self.assertEqual((rec["verdict"][0][0], rec["verdict"][0][1]["twin_score"]), (0.5, 0.5), rec)

    def test_with_g1_zero_the_reward_is_zero_and_the_reference_is_still_graded(self):
        out = dict(self._lockdown()[0], submission_irregular=1.0)
        rec = self._main(out)
        self.assertEqual(rec["worker"], [None])
        self.assertEqual(len(rec["verdict"]), 1, rec)
        self.assertEqual((rec["verdict"][0][0], rec["verdict"][0][1]["twin_score"]), (0.0, 0.0), rec)

    def _write_verdict(self, overall, metrics, anchor=None, ale_python=True):
        rec = {"ratio_rewards": [], "written": [], "fail": []}
        doc = self.doc
        value = doc["values"]["score.anchor"] if anchor is None else anchor
        meta = json.loads(ANCHOR.read_text())

        def fail(reason):
            rec["fail"].append(reason)
            raise _Fail(reason)

        def ratio_rewards(measured, anchor, higher_is_better=True, metric_name="measured", extra=None, full_at=1.0,
                          **kw):
            rec["ratio_rewards"].append((measured, anchor, higher_is_better, metric_name, full_at))
            m = dict(extra or {})
            m.update({"ratio": measured / anchor, "anchor_recorded": 1.0})
            return {"rewards": {"reward": min(1.0, measured / anchor / full_at)}, "metrics": m}

        kit = types.ModuleType("robotics_grader")
        kit.fail, kit.ratio_rewards = fail, ratio_rewards
        kit.read_anchor = lambda: (value, meta)
        kit.write_verdict = lambda verdict, path=None, diagnostic=None: rec["written"].append(verdict)
        metrics_mod = _module("twin.metrics", METRICS)
        twin = types.ModuleType("twin")
        twin.metrics = metrics_mod
        ns = {"os": os, "math": math, "STAGE_DIR": str(self.stage), "KIT_METRICS": _literal("KIT_METRICS")}
        exec(_functions(VERIFY, ["write_verdict"]), ns)
        env = {k: v for k, v in os.environ.items() if k not in ("ALE_VERIFY_PYTHON", "ROBOTICS_GRADER_VERDICT_PATH")}
        if ale_python:
            env["ALE_VERIFY_PYTHON"] = sys.executable
        else:
            env["ROBOTICS_GRADER_VERDICT_PATH"] = str(self.tmp / "verdict.json")
        with mock.patch.dict(sys.modules, {"robotics_grader": kit, "twin": twin, "twin.metrics": metrics_mod}), \
                mock.patch.dict(os.environ, env, clear=True):
            try:
                ns["write_verdict"](overall, metrics, "diagnostic", doc)
            except _Fail:
                pass
        return rec

    def test_write_verdict_hands_the_twin_score_to_the_kit_with_whitelisted_metrics(self):
        flight = self.doc["values"]["flights.names_neutral"][0]
        given = {"twin_score": 0.25, "G1": 1.0, "%s_score" % flight: 0.25, "lockdown_s": 1.5, "bogus_metric": 3.0,
                 "%s_D" % flight: float("nan")}
        for ale_python in (True, False):
            with self.subTest(ale_python=ale_python):
                rec = self._write_verdict(0.25, given, ale_python=ale_python)
                self.assertEqual(rec["fail"], [])
                self.assertEqual([r[:2] for r in rec["ratio_rewards"]], [(0.25, self.doc["values"]["score.anchor"])])
                self.assertEqual(len(rec["written"]), 1)
                verdict = rec["written"][0]
                self.assertEqual(verdict["rewards"], {"reward": 0.25})
                self.assertEqual(sorted(verdict["metrics"]),
                                 sorted(["twin_score", "G1", "%s_score" % flight, "ratio", "anchor_recorded"]))

    def test_an_anchor_that_is_not_the_configurations_is_a_task_error(self):
        rec = self._write_verdict(0.25, {"twin_score": 0.25}, anchor=0.5)
        self.assertEqual(len(rec["fail"]), 1, rec)
        self.assertIn("is not the configuration's score.anchor / score.full_at", rec["fail"][0])
        self.assertEqual(rec["written"], [])


if __name__ == "__main__":
    unittest.main()
