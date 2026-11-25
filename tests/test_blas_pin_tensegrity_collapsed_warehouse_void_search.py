"""tensegrity_collapsed_warehouse_void_search: the numerics of the program under test, of the cell and of the
practice ground do not depend on the CPU they run on, and a practice run uses the same ones as a scored episode.

numpy's bundled OpenBLAS picks its kernel per CPU when it is loaded (SkylakeX on an AVX-512 host, Haswell on an AVX2
one) and numpy picks its SIMD paths per CPU; a simulation and a program's own decisions amplify a last-bit
difference. The pin is OPENBLAS_CORETYPE=Haswell, OPENBLAS_L2_SIZE=256 (the Haswell kernel's built-in dgemm blocking:
numpy 2.5's OpenBLAS otherwise raises it on Zen 4 / Zen 5 CPUs, where a product with an inner dimension above about
300 then crashes or returns wrong numbers) and NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR", read when
numpy is first imported, so it must be in the environment a process starts with:

  * the verifier's own process (the cell, the scorer, the privileged reference): tests/test.sh exports the three
    values before it starts the interpreter;
  * the program's processes in a scored episode (the import check and every episode): their environment is built
    from scratch (verify.py Runner.env: cellkit.runner.child_env with verify.py BLAS_PIN as its `extra`);
  * the practice ground (environment/setup/setup.sh starts it with the same three values) and the program in a
    practice run (~/training/run_mission.py NUMERICS);
  * the verdict records it (verify.py blas_pin_stats, called in _finish_stats: statistics only).

The stdlib checks run everywhere. The checks that start a child process with numpy need an interpreter with numpy
(this one, or ARB_NUMPY_PYTHON; skipped otherwise): a child started with Runner.env's environment loads OpenBLAS with
the Haswell kernel and dispatches none of the three disabled SIMD targets, verify.py's own statistic says so, and it
says the opposite in a child without the pin on a host whose CPU picks another kernel. Nothing is downloaded.
"""
import ast
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

SLUG = "tensegrity_collapsed_warehouse_void_search"
PIN = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256", "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR"}


def _task_dir():
    env = os.environ.get("ARB_TASK_DIR")
    here = Path(__file__).resolve()
    for cand in ([Path(env)] if env else []) + [here.parents[1] / "tasks" / "soft" / SLUG, here.parents[1].parent / SLUG]:
        if (cand / "tests" / "verify.py").is_file():
            return cand
    raise unittest.SkipTest("task folder %s not found" % SLUG)


TASK = _task_dir()
TESTS = TASK / "tests"
RUN_MISSION = TASK / "environment" / "setup" / "payload" / "training" / "run_mission.py"


def _module_dict(path, name):
    """The value of the module-level `name = {...}` of a Python file (a literal)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError("%s defines no %s" % (path, name))


def _function_source(path, name, cls=None):
    src = path.read_text(encoding="utf-8")
    body = ast.parse(src).body
    if cls:
        body = [n for n in body if isinstance(n, ast.ClassDef) and n.name == cls][0].body
    for node in body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node)
    raise AssertionError("%s defines no %s" % (path, name))


def _cellkit_runner():
    spec = importlib.util.spec_from_file_location("blas_pin_cellkit_runner", TESTS / "cellkit" / "runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _program_env(home):
    """The environment verify.py gives the program: Runner.env's own source, bound to cellkit.runner and BLAS_PIN."""
    import textwrap
    ns = {"runner": _cellkit_runner(), "ROOT_GIVEN": "/opt/arb/given", "BLAS_PIN": _module_dict(TESTS / "verify.py", "BLAS_PIN")}
    exec(textwrap.dedent(_function_source(TESTS / "verify.py", "env", cls="Runner")), ns)

    class _R(object):
        threads = 2
    return ns["env"](_R(), {"home": home, "tmp": home, "lib": home})


def _numpy_python():
    for py in (sys.executable, os.environ.get("ARB_NUMPY_PYTHON")):
        if py and os.path.isfile(py):
            r = subprocess.run([py, "-I", "-B", "-c", "import numpy"], capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return py
    return None


# the child: numpy (and scipy when it is there) loaded as the program would load them, then verify.py's own statistic
# run in that process on its own environment, and the kernels read directly
CHILD = r"""
import ctypes, json, os, re, sys
import numpy
try:
    import scipy.linalg  # noqa: F401
except Exception:
    pass
ns = {"os": os, "BLAS_PIN": json.loads(sys.argv[2])}
exec(open(sys.argv[1]).read(), ns)
with open("/proc/self/maps") as fh:
    libs = sorted(set(re.findall(r"(/\S*openblas\S*\.so\S*)", fh.read())))
cores = {}
for lib in libs:
    h = ctypes.CDLL(lib)
    for sym in ("scipy_openblas_get_corename64_", "scipy_openblas_get_corename", "openblas_get_corename64_", "openblas_get_corename"):
        f = getattr(h, sym, None)
        if f is not None:
            f.restype = ctypes.c_char_p
            cores[os.path.basename(lib)] = f().decode()
            break
try:
    from numpy._core._multiarray_umath import __cpu_features__ as F, __cpu_dispatch__ as D
except ImportError:
    from numpy.core._multiarray_umath import __cpu_features__ as F, __cpu_dispatch__ as D
a = numpy.random.default_rng(0).random((700, 700))
p = a @ a.T
print(json.dumps({"stats": ns["blas_pin_stats"](dict(os.environ)), "cores": cores, "numpy": numpy.__version__,
                  "dispatch_enabled": [k for k in D if F.get(k)], "mm700_trace": float(numpy.trace(p))}))
"""


class BlasPin(unittest.TestCase):
    def test_one_constant_everywhere(self):
        self.assertEqual(_module_dict(TESTS / "verify.py", "BLAS_PIN"), PIN)
        self.assertEqual(_module_dict(RUN_MISSION, "NUMERICS"), PIN)
        given = TESTS / "setup_payload" / "training" / "run_mission.py"
        if given.is_file() and given.read_bytes() == RUN_MISSION.read_bytes():
            self.assertEqual(_module_dict(given, "NUMERICS"), PIN)

    def test_program_env_carries_the_pin(self):
        env = _program_env("/h")
        for k, v in PIN.items():
            self.assertEqual(env.get(k), v, k)
        self.assertEqual((env["OMP_NUM_THREADS"], env["OPENBLAS_NUM_THREADS"], env["MKL_NUM_THREADS"], env["LP_NUM_THREADS"]),
                         ("2", "2", "2", "2"))
        os.environ["BLAS_PIN_PROBE_VARIABLE"] = "x"
        try:
            self.assertNotIn("BLAS_PIN_PROBE_VARIABLE", _program_env("/h"))
        finally:
            del os.environ["BLAS_PIN_PROBE_VARIABLE"]
        # every program process verify.py starts takes Runner.env (the import check and the episodes)
        src = (TESTS / "verify.py").read_text(encoding="utf-8")
        self.assertEqual(len(re.findall(r"env=rn\.env\(dirs", src)), 2)

    def test_run_mission_gives_the_practice_program_the_pin(self):
        tree = ast.parse(RUN_MISSION.read_text(encoding="utf-8"))
        ups = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == "update" and isinstance(n.func.value, ast.Name) and n.func.value.id == "env"
               and n.args and isinstance(n.args[0], ast.Name) and n.args[0].id == "NUMERICS"]
        self.assertEqual(len(ups), 1, "env.update(NUMERICS) once in run_mission.py")
        pops = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "Popen"]
        self.assertTrue(pops and all(any(k.arg == "env" for k in p.keywords) for p in pops))
        self.assertLess(ups[0].lineno, min(p.lineno for p in pops))

    def _shell_values(self, line):
        r = subprocess.run(["bash", "-c", "set -- ; " + line + '; printf "%s|%s|%s" "$OPENBLAS_CORETYPE" '
                            '"$OPENBLAS_L2_SIZE" "$NPY_DISABLE_CPU_FEATURES"'],
                           capture_output=True, text=True, timeout=30, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
        return r.stdout

    def test_test_sh_exports_the_pin_before_the_interpreter(self):
        text = (TESTS / "test.sh").read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        exp = [i for i, ln in enumerate(lines) if ln.startswith("export ") and "OPENBLAS_CORETYPE=" in ln]
        ex = [i for i, ln in enumerate(lines) if re.match(r'\s*exec .*verify\.py', ln)]
        self.assertEqual(len(exp), 1, "one export line with the pin")
        self.assertEqual(len(ex), 1, "one exec of verify.py")
        self.assertLess(exp[0], ex[0])
        if shutil.which("bash"):
            self.assertEqual(self._shell_values(lines[exp[0]]), "|".join(PIN.values()))

    def test_practice_ground_starts_with_the_pin(self):
        text = (TASK / "environment" / "setup" / "setup.sh").read_text(encoding="utf-8")
        joined = text.replace("\\\n", " ")
        start = [ln for ln in joined.splitlines() if "practiced.py" in ln and "runuser" in ln]
        self.assertEqual(len(start), 1, "one start line of the practice ground")
        words = shlex.split(start[0].split("/bin/bash -c")[0])
        vals = dict(w.split("=", 1) for w in words if "=" in w and not w.startswith("-"))
        for k, v in PIN.items():
            self.assertEqual(vals.get(k), v, k)

    def test_verify_imports_numpy_only_after_stdlib_modules_and_records_the_pin(self):
        src = (TESTS / "verify.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        first_np = min(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Import) and any(a.name == "numpy" for a in n.names))
        before = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and n.lineno < first_np]
        mods = set()
        for n in before:
            mods |= {a.name for a in n.names} if isinstance(n, ast.Import) else {n.module}
        self.assertTrue(mods <= {"ast", "json", "math", "os", "pwd", "shutil", "stat", "subprocess", "sys", "time",
                                 "traceback", "cellkit", "hashlib"}, mods)
        fin = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_finish_stats"][0]
        calls = [c.func.id for c in ast.walk(fin) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
        self.assertIn("blas_pin_stats", calls)

    def _child(self, py, env):
        d = tempfile.mkdtemp(prefix="blas_pin_")
        try:
            src = os.path.join(d, "blas_pin_stats_src.py")
            with open(src, "w") as fh:
                fh.write(_function_source(TESTS / "verify.py", "blas_pin_stats") + "\n")
            argv = [py, "-B", "-E", "-s", "-c", CHILD, src, json.dumps(PIN)]
            r = subprocess.run(argv, cwd=d, env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, "child ended with %d: %s" % (r.returncode, r.stderr[-2000:]))
            return json.loads(r.stdout.strip().splitlines()[-1])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_child_with_the_program_env_runs_on_the_haswell_kernel(self):
        py = _numpy_python()
        if py is None:
            self.skipTest("no interpreter with numpy (set ARB_NUMPY_PYTHON)")
        home = tempfile.mkdtemp(prefix="blas_pin_home_")
        try:
            env = _program_env(home)
            pinned = self._child(py, env)
            print("\n[blas pin] pinned child:", json.dumps(pinned, sort_keys=True))
            self.assertTrue(pinned["cores"], "no OpenBLAS loaded in the child")
            self.assertEqual(set(pinned["cores"].values()), {"Haswell"})
            if int(pinned["numpy"].split(".")[0]) >= 2:      # numpy 2 (the image's) names the AVX2 group X86_V3
                self.assertIn("X86_V3", pinned["dispatch_enabled"])
            for k in PIN["NPY_DISABLE_CPU_FEATURES"].split():
                self.assertNotIn(k, pinned["dispatch_enabled"])
            self.assertEqual(pinned["stats"], {"blas_pin_env": 1.0, "blas_pin_child_env": 1.0, "blas_l2_size_256": 1.0,
                                               "blas_kernel_haswell": 1.0, "blas_numpy_simd_capped": 1.0})
            # the control: the same environment without the pin; the statistic must follow what the CPU picked
            bare = {k: v for k, v in env.items() if k not in PIN}
            own = self._child(py, bare)
            print("[blas pin] child without the pin:", json.dumps(own, sort_keys=True))
            self.assertEqual((own["stats"]["blas_pin_env"], own["stats"]["blas_pin_child_env"],
                              own["stats"]["blas_l2_size_256"]), (0.0, 0.0, 0.0))
            haswell = set(own["cores"].values()) == {"Haswell"}
            self.assertEqual(own["stats"]["blas_kernel_haswell"], 1.0 if haswell else 0.0)
            wide = any(k in own["dispatch_enabled"] for k in PIN["NPY_DISABLE_CPU_FEATURES"].split())
            self.assertEqual(own["stats"]["blas_numpy_simd_capped"], 0.0 if wide else 1.0)
            self.assertAlmostEqual(own["mm700_trace"], pinned["mm700_trace"], delta=1e-9 * abs(pinned["mm700_trace"]))
        finally:
            shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
