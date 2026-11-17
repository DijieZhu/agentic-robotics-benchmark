"""novel_object_6dof_grasp_detection: every graded process runs with OPENBLAS_CORETYPE=Haswell, and the verify stage
ends as a task error, before the hidden frames are fetched, on a machine where /opt/grasp's matrix products are wrong.

numpy 1.23.5 bundles OpenBLAS 0.3.20, which picks its Cooperlake kernels on Intel CPUs with AVX512_BF16, and with those
DGEMM returns wrong products (graspnetAPI's transform_points among them). Checked here: the pin in the solver processes'
environment (CHILD_ENV, and that the solvers are started with it), in the scoring subprocess's, in the image (the ENV,
and the .pth that makes it /opt/grasp/bin/python's default when the variable is missing) and in the practice check;
that check_numerics() refuses when tests/numerics_check.py reports a problem and passes otherwise (subprocess stubbed);
the report contract of numerics_check.py's main() (its numpy work stubbed); and its pure decision function. Stdlib
only: no Docker, no numpy 1.23.5, no network. tests/verify.py imports the vendored grading kit at module level, which
is only staged in the verifier, so its pieces are taken from it by name.
"""
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

TASK = Path(__file__).resolve().parents[1] / "tasks/arm/novel_object_6dof_grasp_detection"
VERIFY = TASK / "tests/verify.py"
CHECK = TASK / "tests/numerics_check.py"
DOCKERFILE = TASK / "environment/Dockerfile"
PRACTICE = TASK / "environment/setup/payload/practice_check/check.py"
INSTRUCTION = TASK / "instruction.md"
PIN = {"OPENBLAS_CORETYPE": "Haswell"}
PTH = "/opt/grasp/lib/python3.10/site-packages/openblas_coretype.pth"
PTH_LINE = 'import os; os.environ.setdefault("OPENBLAS_CORETYPE", "Haswell")'
CONSTANTS = {"GRASP_PY", "HOME", "SUBMISSION", "NUMERICS_ENV", "NUMERICS_TIMEOUT_S", "CHILD_ENV"}
FUNCTIONS = {"scoring_env", "_numerics_report", "check_numerics"}
STAGE = "/opt/ale/tests"

HASWELL = {"numpy.libs/libopenblas64_p-r0-742d56dc.3.20.so": "Haswell",
           "scipy.libs/libopenblasp-r0-41284840.3.18.so": "Haswell",
           "opencv_python.libs/libopenblas-r0-f650aae0.3.3.so": "Haswell",
           "cvxopt.libs/libopenblasp-r0-5c2b7639.3.23.so": "Haswell"}
TINY = {"transform_points": {1: 0.0, 255: 1.2e-16, 93750: 3.4e-16},
        "matmul(T, X.T)": {255: 1.3e-16, 1000: 9.5e-17},
        "matmul((G,N,3), (G,3,3))": {192: 2.2e-16}}


class Failed(Exception):
    pass


def _main():
    main = next(n for n in ast.parse(VERIFY.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "main")
    return main


def _env_of_calls(tree, func):
    """The source of the env= argument of every call to `func` (a dotted name) in `tree`."""
    return [next((ast.unparse(k.value) for k in n.keywords if k.arg == "env"), None) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == func]


def _load_verify(run=None):
    """verify.py's constants and numerics functions, with subprocess.run replaced by `run`."""
    body, found = [], set()
    for node in ast.parse(VERIFY.read_text()).body:
        if isinstance(node, ast.Assign):
            names = {t.id for t in node.targets if isinstance(t, ast.Name)} & CONSTANTS
            if names:
                body.append(node)
                found |= names
        elif isinstance(node, ast.FunctionDef) and node.name in FUNCTIONS:
            body.append(node)
            found.add(node.name)
    assert found == CONSTANTS | FUNCTIONS, "verify.py lacks %s" % ((CONSTANTS | FUNCTIONS) - found)
    logs = []

    def require(condition, reason):
        if not condition:
            raise Failed(reason)

    fake = types.SimpleNamespace(run=run, PIPE=subprocess.PIPE, TimeoutExpired=subprocess.TimeoutExpired)
    ns = {"os": os, "json": json, "shutil": shutil, "tempfile": tempfile, "subprocess": fake, "require": require,
          "log": logs.append, "stage_dir": lambda: STAGE}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(VERIFY), "exec"), ns)
    return ns, logs


def _load_check():
    """tests/numerics_check.py as a module, compiled here so that no __pycache__ is left in the task folder."""
    module = types.ModuleType("grasp_numerics_check")
    module.__file__ = str(CHECK)
    exec(compile(CHECK.read_text(), str(CHECK), "exec"), module.__dict__)
    return module


def _report(problems=(), ok=None):
    ok = not problems if ok is None else ok
    return {"machine": "x86_64", "cores": HASWELL, "max_rel_error": 3.4e-16, "ok": ok, "problems": list(problems)}


class _Runs:
    """subprocess.run stand-in: answers the solver-environment run, then the scoring-environment run."""

    def __init__(self, *answers):
        self.answers, self.calls, self.homes = list(answers), [], []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        home = kwargs["env"]["HOME"]
        self.homes.append((home, os.path.isdir(home), os.stat(home).st_uid if os.path.isdir(home) else None))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        code, out = answer
        if isinstance(out, dict):
            out = json.dumps(out) + "\n"
        return subprocess.CompletedProcess(argv, code, stdout=out.encode())


class PinInVerify(unittest.TestCase):
    def test_child_env_carries_the_pin(self):
        ns, _ = _load_verify()
        self.assertEqual(ns["NUMERICS_ENV"], PIN)
        for key, value in PIN.items():
            self.assertEqual(ns["CHILD_ENV"].get(key), value)
        self.assertEqual(ns["CHILD_ENV"]["OPENBLAS_NUM_THREADS"], "1")      # the solvers' thread setting is kept

    def test_solvers_are_started_with_child_env(self):
        tree = ast.parse(VERIFY.read_text())
        self.assertEqual(_env_of_calls(tree, "deprivileged"), ["CHILD_ENV"])

    def test_scoring_env_is_the_stage_env_plus_the_pin(self):
        ns, _ = _load_verify()
        with mock.patch.dict(os.environ, {"OPENBLAS_CORETYPE": "Cooperlake", "STAGE_ONLY": "kept"}):
            env = ns["scoring_env"]()
        self.assertEqual(env["OPENBLAS_CORETYPE"], "Haswell")
        self.assertEqual(env["STAGE_ONLY"], "kept")

    def test_score_frames_is_started_with_the_scoring_env(self):
        calls = [n for n in ast.walk(_main()) if isinstance(n, ast.Call) and "score_frames.py" in ast.unparse(n)
                 and ast.unparse(n.func) in ("subprocess.call", "subprocess.run", "subprocess.Popen")]
        self.assertEqual(len(calls), 1, [ast.unparse(c) for c in calls])
        env = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}.get("env")
        self.assertEqual(env, "scoring_env()")

    def test_check_runs_after_the_kill_and_before_the_download(self):
        src = ast.unparse(_main())
        self.assertLess(src.index("zero_verdict("), src.index("check_numerics()"))
        self.assertLess(src.index("freezer.start()"), src.index("check_numerics()"))
        self.assertLess(src.index("check_numerics()"), src.index("fetch_hidden(cfg)"))


class CheckNumerics(unittest.TestCase):
    def test_passes_when_both_runs_pass(self):
        runs = _Runs((0, _report()), (0, "a library's chatter\n" + json.dumps(_report()) + "\n"))
        ns, logs = _load_verify(runs)
        ns["check_numerics"]()
        self.assertEqual(len(runs.calls), 2)
        for argv, kwargs in runs.calls:
            self.assertEqual(argv, [ns["GRASP_PY"], "-I", os.path.join(STAGE, "numerics_check.py")])
            self.assertEqual(kwargs["cwd"], STAGE)
            self.assertGreater(kwargs["timeout"], 0)
            self.assertEqual(kwargs["env"]["OPENBLAS_CORETYPE"], "Haswell")
        solver_env, scoring_env = runs.calls[0][1]["env"], runs.calls[1][1]["env"]

        def without_home(env):
            return {k: v for k, v in env.items() if k != "HOME"}
        self.assertEqual(without_home(solver_env), without_home(ns["CHILD_ENV"]))
        self.assertEqual(without_home(scoring_env), without_home(ns["scoring_env"]()))
        # both runs get a throwaway HOME of this (root) process, never the agent's (which the image also gives the
        # stage), removed afterwards
        for home, existed, owner in runs.homes:
            self.assertNotIn(home, (ns["CHILD_ENV"]["HOME"], os.environ.get("HOME")))
            self.assertTrue(existed)
            self.assertEqual(owner, os.getuid())
            self.assertFalse(os.path.exists(home))
        self.assertEqual(sum("numerics check" in m and '"ok": true' in m for m in logs), 2, logs)

    def test_the_stage_home_is_not_used_for_the_scoring_run(self):
        runs = _Runs((0, _report()), (0, _report()))
        ns, _ = _load_verify(runs)
        with mock.patch.dict(os.environ, {"HOME": "/home/user"}):
            ns["check_numerics"]()
        self.assertNotEqual(runs.calls[1][1]["env"]["HOME"], "/home/user")

    def _refused(self, *answers):
        runs = _Runs(*answers)
        ns, logs = _load_verify(runs)
        with self.assertRaises(Failed) as caught:
            ns["check_numerics"]()
        self.assertEqual(len(runs.calls), 2)                  # both environments are checked and logged
        self.assertFalse(any(os.path.exists(home) for home, _, _ in runs.homes))
        return str(caught.exception), logs

    def test_refuses_a_reported_problem(self):
        problem = "numpy.libs/libopenblas64_p-r0-742d56dc.3.20.so uses its Cooperlake kernels, not its Haswell ones"
        reason, logs = self._refused((0, _report()), (1, _report([problem])))
        self.assertIn("scoring environment: " + problem, reason)
        self.assertNotIn("solver environment", reason)
        self.assertTrue(any(problem in m for m in logs), logs)

    def test_refuses_wrong_products_in_the_solver_environment(self):
        problem = "matmul(T, X.T): relative error up to 1.72 (limit 1e-09) for N = 255, 1000"
        reason, _ = self._refused((1, _report([problem])), (0, _report()))
        self.assertIn("solver environment: " + problem, reason)

    def test_refuses_a_run_without_a_report(self):
        reason, logs = self._refused((-4, ""), (0, _report()))     # e.g. SIGILL before anything is printed
        self.assertIn("exited -4 without a report", reason)
        self.assertTrue(any("no report" in m for m in logs), logs)

    def test_refuses_a_timeout(self):
        reason, _ = self._refused((0, _report()), subprocess.TimeoutExpired(["python"], 300))
        self.assertIn("did not finish", reason)

    def test_refuses_an_inconsistent_report(self):
        for code, report in ((1, _report()), (0, _report(ok=False)), (0, {"ok": "yes", "problems": []})):
            with self.subTest(code=code, report=report):
                reason, _ = self._refused((code, report), (0, _report()))
                self.assertIn("solver environment", reason)

    def test_refuses_garbage(self):
        reason, _ = self._refused((0, "not json\n"), (0, "[1, 2]\n"))
        self.assertIn("solver environment", reason)
        self.assertIn("scoring environment", reason)


class Decide(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.check = _load_check()

    def test_module_import_needs_only_the_standard_library(self):
        code = ("import importlib.util, sys; spec = importlib.util.spec_from_file_location('c', sys.argv[1]); "
                "m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
                "print(sorted({'numpy', 'scipy', 'cv2', 'cvxopt', 'graspnetAPI'} & set(sys.modules)))")
        out = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(CHECK)], capture_output=True, text=True,
                             check=True)
        self.assertEqual(out.stdout.strip(), "[]")

    def test_main_prints_one_json_line_and_exits_by_the_verdict(self):
        """main()'s contract with verify.py, its numpy work stubbed: one JSON line on stdout and nothing else (what the
        libraries print goes to stderr), exit 0 only when ok, and a non-finite error still reported and parsed."""
        code = textwrap.dedent("""
            import platform, sys, types
            m = types.ModuleType("grasp_numerics_check")
            exec(compile(open(sys.argv[1]).read(), sys.argv[1], "exec"), m.__dict__)
            def load_stack():
                print("a library's chatter")
                return None, None
            m.load_stack = load_stack
            m.self_test = lambda np, transform_points: {"transform_points": {255: float(sys.argv[2])}}
            m.openblas_cores = lambda: {"numpy.libs/libopenblas64_p-r0-742d56dc.3.20.so": "Haswell"}
            platform.machine = lambda: "x86_64"
            sys.exit(m.main())
        """)
        for err, rc, ok in (("1e-16", 0, True), ("nan", 1, False)):
            with self.subTest(err=err):
                out = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(CHECK), err], capture_output=True,
                                     text=True)
                self.assertEqual(out.returncode, rc, out.stderr)
                self.assertEqual(len(out.stdout.splitlines()), 1, out.stdout)
                report = json.loads(out.stdout)
                self.assertIs(report["ok"], ok)
                self.assertIn("a library's chatter", out.stderr)
                if not ok:
                    self.assertEqual(report["max_rel_error"], float("inf"))
                ns, _ = _load_verify(_Runs((out.returncode, out.stdout)))
                line, problems = ns["_numerics_report"](dict(PIN, HOME=tempfile.gettempdir()))
                self.assertEqual(line, out.stdout.strip())
                self.assertEqual(problems == [], ok, problems)

    def test_haswell_everywhere_and_tiny_errors_pass(self):
        self.assertEqual(self.check.decide("x86_64", HASWELL, TINY), [])

    def test_a_cooperlake_copy_fails(self):
        cores = dict(HASWELL, **{"numpy.libs/libopenblas64_p-r0-742d56dc.3.20.so": "Cooperlake"})
        problems = self.check.decide("x86_64", cores, TINY)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("numpy.libs/libopenblas64_p-r0-742d56dc.3.20.so", problems[0])
        self.assertIn("Cooperlake", problems[0])

    def test_a_copy_that_cannot_be_queried_fails(self):
        cores = dict(HASWELL, **{"scipy.libs/libopenblasp-r0-41284840.3.18.so": None})
        self.assertEqual(len(self.check.decide("x86_64", cores, TINY)), 1)

    def test_no_copy_found_fails_on_x86(self):
        self.assertEqual(len(self.check.decide("x86_64", {}, TINY)), 1)

    def test_a_large_product_error_fails(self):
        errors = dict(TINY, **{"matmul(T, X.T)": {192: 1e-16, 255: 0.97, 1000: 1.11}})
        problems = self.check.decide("x86_64", HASWELL, errors)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("matmul(T, X.T)", problems[0])
        self.assertIn("N = 255, 1000", problems[0])
        self.assertNotIn("192", problems[0])

    def test_an_error_just_above_the_limit_or_not_finite_fails(self):
        for err in (2e-9, float("nan"), float("inf")):
            with self.subTest(err=err):
                errors = dict(TINY, **{"transform_points": {255: err}})
                self.assertEqual(len(self.check.decide("x86_64", HASWELL, errors)), 1)

    def test_a_self_test_that_did_not_run_fails(self):
        self.assertEqual(len(self.check.decide("x86_64", HASWELL, {})), 1)
        self.assertEqual(len(self.check.decide("aarch64", {}, {"transform_points": {}})), 1)

    def test_other_architectures_need_correct_products_only(self):
        cores = {"numpy.libs/libopenblas64_p-r0-0.3.20.so": None, "scipy.libs/libopenblasp-r0-0.3.18.so": "NEOVERSEN1"}
        self.assertEqual(self.check.decide("aarch64", cores, TINY), [])
        self.assertEqual(self.check.decide("aarch64", {}, TINY), [])
        errors = dict(TINY, **{"transform_points": {1000: 1.5}})
        self.assertEqual(len(self.check.decide("aarch64", cores, errors)), 1)

    def test_maps_scan_names_every_copy_and_reports_one_it_cannot_ask(self):
        with tempfile.TemporaryDirectory() as tmp:
            maps = os.path.join(tmp, "maps")
            missing = os.path.join(tmp, "numpy.libs", "libopenblas64_p-r0-742d56dc.3.20.so")
            with open(maps, "w") as handle:
                handle.write("7f00-7f01 r-xp 00000000 08:01 42   %s\n" % missing)
                handle.write("7f01-7f02 r--p 00001000 08:01 42   %s\n" % missing)
                handle.write("7f02-7f03 r-xp 00000000 08:01 43   /usr/lib/x86_64-linux-gnu/libm.so.6\n")
                handle.write("7f03-7f04 rw-p 00000000 00:00 0    [heap]\n")
                handle.write("7f04-7f05 rw-p 00000000 00:00 0\n")
            cores = self.check.openblas_cores(maps)
        self.assertEqual(cores, {"numpy.libs/libopenblas64_p-r0-742d56dc.3.20.so": None})
        self.assertEqual(len(self.check.decide("x86_64", cores, TINY)), 1)


class PinInImageAndPractice(unittest.TestCase):
    def test_dockerfile_sets_the_variable_after_the_heavy_layers(self):
        lines = DOCKERFILE.read_text().splitlines()
        self.assertIn("ENV OPENBLAS_CORETYPE=Haswell", lines)
        pin = lines.index("ENV OPENBLAS_CORETYPE=Haswell")
        heavy = max(i for i, line in enumerate(lines) if "pip install" in line or "fetch_assets.py" in line)
        self.assertGreater(pin, heavy)
        last_run = max(i for i, line in enumerate(lines) if line.startswith("RUN "))
        self.assertGreater(last_run, pin)
        final = "\n".join(lines[last_run:])
        self.assertIn("assert os.environ.get('OPENBLAS_CORETYPE') == 'Haswell'", final)
        self.assertIn("einsum", final)
        self.assertIn("assert e < 1e-9", final)
        self.assertIn("runuser -u user", final)
        # the agent account's /opt/grasp/bin/python, with the variable removed, must still end up with it (the .pth)
        self.assertIn("runuser -u user -- env -u OPENBLAS_CORETYPE /opt/grasp/bin/python", final)

    def _pth_run(self):
        """The Dockerfile's RUN that writes the .pth, as (its first line's index, its logical line)."""
        lines = DOCKERFILE.read_text().splitlines()
        at = [i for i, line in enumerate(lines) if line.startswith("RUN ") and PTH_LINE in line]
        self.assertEqual(len(at), 1, at)
        end = at[0]
        while lines[end].endswith("\\"):
            end += 1
        return at[0], " ".join(line.rstrip("\\").strip() for line in lines[at[0]:end + 1])

    def test_dockerfile_writes_the_interpreter_default_next_to_the_env(self):
        lines = DOCKERFILE.read_text().splitlines()
        at, run = self._pth_run()
        self.assertEqual(at, lines.index("ENV OPENBLAS_CORETYPE=Haswell") + 1)     # its own RUN, after the heavy layers
        self.assertEqual(run, "RUN printf '%%s\\n' '%s' > %s && chmod 0644 %s" % (PTH_LINE, PTH, PTH))

    def test_the_pth_line_sets_a_default_that_an_explicit_value_overrides(self):
        """The line as Python's site module runs a .pth: the variable is Haswell when it was missing, and an explicit
        value is kept."""
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, os.path.basename(PTH)), "w") as handle:
                handle.write(PTH_LINE + "\n")
            code = "import os, site, sys; site.addsitedir(sys.argv[1]); print(os.environ.get('OPENBLAS_CORETYPE'))"
            for given, expected in ((None, "Haswell"), ("Cooperlake", "Cooperlake")):
                env = {k: v for k, v in os.environ.items() if k != "OPENBLAS_CORETYPE"}
                if given is not None:
                    env["OPENBLAS_CORETYPE"] = given
                out = subprocess.run([sys.executable, "-I", "-B", "-c", code, tmp], env=env, capture_output=True,
                                     text=True, check=True)
                self.assertEqual(out.stdout.strip(), expected)

    def test_practice_check_sets_the_default_before_numpy(self):
        body = ast.parse(PRACTICE.read_text()).body
        numpy_at = next(i for i, n in enumerate(body) if isinstance(n, (ast.Import, ast.ImportFrom))
                        and any(a.name == "numpy" for a in n.names))
        pin_at = [i for i, n in enumerate(body) if isinstance(n, ast.Expr)
                  and ast.unparse(n) == "os.environ.setdefault('OPENBLAS_CORETYPE', 'Haswell')"]
        self.assertEqual(len(pin_at), 1)
        self.assertLess(pin_at[0], numpy_at)

    def test_the_agent_is_told(self):
        text = INSTRUCTION.read_text()
        self.assertIn("`OPENBLAS_CORETYPE=Haswell` is set in this environment and in the graded run", text)
        self.assertIn("`/opt/grasp/bin/python` falls back to it", text)


if __name__ == "__main__":
    unittest.main()
