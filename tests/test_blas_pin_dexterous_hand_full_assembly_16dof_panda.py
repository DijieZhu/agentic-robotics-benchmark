"""dexterous_hand_full_assembly_16dof_panda: the grader's numerics do not depend on the CPU it runs on.

numpy's (and scipy's) bundled OpenBLAS picks its kernel per CPU when it is loaded (SkylakeX on an AVX-512 host,
Haswell on an AVX2 one) and numpy picks its SIMD paths per CPU; the cell's control (damped least squares), its inverse
kinematics, the layout and the privileged reference go through both, and a simulation amplifies a last-bit difference.
The pin is OPENBLAS_CORETYPE=Haswell, OPENBLAS_L2_SIZE=256 (the Haswell kernel's built-in dgemm blocking: numpy 2.5's
OpenBLAS otherwise raises it on Zen 4 / Zen 5 CPUs, where a product with an inner dimension above about 300 then
crashes or returns wrong numbers) and NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR", read when numpy is
first imported, so it must be in the environment a process starts with:

  * the verifier's own process (the cell, the reference, the grading): tests/test.sh exports both before it starts the
    interpreter (the same values as tests/arm/assembly_cell/runner.py BLAS_PIN);
  * the programs' processes (the import check and every episode): their environment is built from scratch by
    runner.child_env, which carries BLAS_PIN (an `extra` of the caller can still add to it);
  * the verdict records it (verify.py blas_pin_stats, called in _finish_stats: statistics only; blas_l2_size_256 for
    the L2 size in both environments).

The stdlib checks run everywhere. The checks that start a child process with numpy need an interpreter with numpy
(this one, or ARB_NUMPY_PYTHON; skipped otherwise): a child started with child_env's environment loads OpenBLAS with
the Haswell kernel and dispatches none of the three disabled SIMD targets, and verify.py's own statistic says so
(and says the opposite in a child without the pin on a host whose CPU picks another kernel). Nothing is downloaded.
"""
import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

SLUG = "dexterous_hand_full_assembly_16dof_panda"
PIN = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256", "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR"}


def _task_dir():
    env = os.environ.get("ARB_TASK_DIR")
    here = Path(__file__).resolve()
    for cand in ([Path(env)] if env else []) + [here.parents[1] / "tasks" / "hand" / SLUG, here.parents[1].parent / SLUG]:
        if (cand / "tests" / "verify.py").is_file():
            return cand
    raise unittest.SkipTest("task folder %s not found" % SLUG)


TASK = _task_dir()
TESTS = TASK / "tests"


def _runner():
    spec = importlib.util.spec_from_file_location("blas_pin_grader_runner", TESTS / "arm" / "assembly_cell" / "runner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _numpy_python():
    for py in (sys.executable, os.environ.get("ARB_NUMPY_PYTHON")):
        if py and os.path.isfile(py):
            r = subprocess.run([py, "-I", "-B", "-c", "import numpy"], capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return py
    return None


def _stats_source():
    src = (TESTS / "verify.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name == "blas_pin_stats":
            return ast.get_source_segment(src, node)
    raise AssertionError("verify.py defines no blas_pin_stats")


# the child: numpy (and scipy's OpenBLAS when scipy is there) loaded the way the cell loads them, then verify.py's own
# statistic run in that process, and the kernels read directly
CHILD = r"""
import ctypes, json, os, re, sys
import numpy
try:
    import scipy.linalg  # noqa: F401
except Exception:
    pass
spec = __import__("importlib.util").util.spec_from_file_location("runner", sys.argv[1])
runner = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(runner)
ns = {"os": os, "runner": runner}
exec(open(sys.argv[2]).read(), ns)
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
print(json.dumps({"stats": ns["blas_pin_stats"](), "cores": cores, "numpy": numpy.__version__,
                  "dispatch_enabled": [k for k in D if F.get(k)]}))
"""


class BlasPin(unittest.TestCase):
    def test_runner_constant_and_child_env(self):
        rn = _runner()
        self.assertEqual(rn.BLAS_PIN, PIN)
        env = rn.child_env("/h", "/t", "/l", "/g")
        for k, v in PIN.items():
            self.assertEqual(env.get(k), v, k)
        # the minimal environment otherwise as before: the thread counts, PATH from the caller, nothing else of it
        self.assertEqual((env["OMP_NUM_THREADS"], env["OPENBLAS_NUM_THREADS"], env["MKL_NUM_THREADS"]), ("2", "2", "2"))
        os.environ["BLAS_PIN_PROBE_VARIABLE"] = "x"
        try:
            self.assertNotIn("BLAS_PIN_PROBE_VARIABLE", rn.child_env("/h", "/t", "/l", "/g"))
        finally:
            del os.environ["BLAS_PIN_PROBE_VARIABLE"]
        # an extra of the caller is applied after the pin (verify.py passes none for the episodes)
        self.assertEqual(rn.child_env("/h", "/t", "/l", "/g", extra={"X": "1"})["OPENBLAS_CORETYPE"], "Haswell")

    def test_test_sh_exports_the_pin_before_the_interpreter(self):
        text = (TESTS / "test.sh").read_text(encoding="utf-8")
        lines = [ln for ln in text.splitlines() if not ln.lstrip().startswith("#")]
        exp = [i for i, ln in enumerate(lines) if ln.startswith("export ") and "OPENBLAS_CORETYPE=" in ln]
        ex = [i for i, ln in enumerate(lines) if re.match(r'\s*exec .*verify\.py', ln)]
        self.assertEqual(len(exp), 1, "one export line with the pin")
        self.assertEqual(len(ex), 1, "one exec of verify.py")
        self.assertLess(exp[0], ex[0])
        line = lines[exp[0]]
        self.assertIn("OPENBLAS_CORETYPE=%s" % PIN["OPENBLAS_CORETYPE"], line)
        self.assertIn("OPENBLAS_L2_SIZE=%s" % PIN["OPENBLAS_L2_SIZE"], line)
        self.assertIn('NPY_DISABLE_CPU_FEATURES="%s"' % PIN["NPY_DISABLE_CPU_FEATURES"], line)
        if shutil.which("bash"):
            r = subprocess.run(["bash", "-c", line + '; printf "%s|%s|%s" "$OPENBLAS_CORETYPE" "$OPENBLAS_L2_SIZE" '
                                '"$NPY_DISABLE_CPU_FEATURES"'],
                               capture_output=True, text=True, timeout=30, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
            self.assertEqual(r.stdout, "%s|%s|%s" % (PIN["OPENBLAS_CORETYPE"], PIN["OPENBLAS_L2_SIZE"],
                                                     PIN["NPY_DISABLE_CPU_FEATURES"]))

    def test_verify_imports_numpy_only_after_stdlib_modules_and_records_the_pin(self):
        src = (TESTS / "verify.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        first_np = min(n.lineno for n in ast.walk(tree) if isinstance(n, ast.Import) and any(a.name == "numpy" for a in n.names))
        before = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and n.lineno < first_np]
        mods = set()
        for n in before:
            mods |= {a.name for a in n.names} if isinstance(n, ast.Import) else {n.module}
        self.assertTrue(mods <= {"ast", "json", "math", "os", "pwd", "shutil", "stat", "subprocess", "sys", "time",
                                 "traceback", "arm.assembly_cell", "arm.assembly_six", "arm", "hashlib"}, mods)
        fin = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_finish_stats"][0]
        calls = [c.func.id for c in ast.walk(fin) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name)]
        self.assertIn("blas_pin_stats", calls)
        self.assertIn("blas_pin_stats", _stats_source())

    def _child(self, py, env):
        d = tempfile.mkdtemp(prefix="blas_pin_")
        try:
            src = os.path.join(d, "blas_pin_stats_src.py")
            with open(src, "w") as fh:
                fh.write(_stats_source() + "\n")
            argv = [py] + _runner().PYTHON_FLAGS + ["-c", CHILD, str(TESTS / "arm" / "assembly_cell" / "runner.py"), src]
            r = subprocess.run(argv, cwd=d, env=env, capture_output=True, text=True, timeout=300)
            self.assertEqual(r.returncode, 0, r.stderr[-2000:])
            return json.loads(r.stdout.strip().splitlines()[-1])
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_child_with_child_env_runs_on_the_haswell_kernel(self):
        py = _numpy_python()
        if py is None:
            self.skipTest("no interpreter with numpy (set ARB_NUMPY_PYTHON)")
        rn = _runner()
        home = tempfile.mkdtemp(prefix="blas_pin_home_")
        try:
            env = rn.child_env(home, home, home, home)
            pinned = self._child(py, env)
            print("\n[blas pin] pinned child:", json.dumps(pinned, sort_keys=True))
            self.assertTrue(pinned["cores"], "no OpenBLAS loaded in the child")
            self.assertEqual(set(pinned["cores"].values()), {"Haswell"})
            for k in PIN["NPY_DISABLE_CPU_FEATURES"].split():
                self.assertNotIn(k, pinned["dispatch_enabled"])
            self.assertEqual(pinned["stats"], {"blas_pin_env": 1.0, "blas_pin_child_env": 1.0, "blas_l2_size_256": 1.0,
                                               "blas_kernel_haswell": 1.0, "blas_numpy_simd_capped": 1.0})
            # the control: the same environment without the pin; the statistic must follow what the CPU picked
            bare = {k: v for k, v in env.items() if k not in PIN}
            own = self._child(py, bare)
            print("[blas pin] child without the pin:", json.dumps(own, sort_keys=True))
            self.assertEqual(own["stats"]["blas_pin_env"], 0.0)
            self.assertEqual(own["stats"]["blas_l2_size_256"], 0.0)
            haswell = set(own["cores"].values()) == {"Haswell"}
            self.assertEqual(own["stats"]["blas_kernel_haswell"], 1.0 if haswell else 0.0)
            wide = any(k in own["dispatch_enabled"] for k in PIN["NPY_DISABLE_CPU_FEATURES"].split())
            self.assertEqual(own["stats"]["blas_numpy_simd_capped"], 0.0 if wide else 1.0)
        finally:
            shutil.rmtree(home, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
