"""bimanual_tray_carry_controller_from_docs: the linear-algebra numerics do not depend on the CPU the task runs on.

numpy's bundled OpenBLAS picks its kernel per CPU when it is loaded (SkylakeX on an AVX-512 host, Haswell on an AVX2
one), and a simulated shift amplifies a last-bit difference of a product. The task fixes the kernel and the cache size
that kernel's blocking assumes, OPENBLAS_CORETYPE=Haswell and OPENBLAS_L2_SIZE=256, in every process that computes.
Both are read when the library is loaded, so they must be in the environment a process starts with:

  * the agent's session: one ENV line of environment/Dockerfile carries both;
  * the grader's own process: tests/test.sh exports both before the line that starts the grader
    (code/tasks/tray_carry/grader/verify.py);
  * the submitted controller's process at grading: the runtime (tests/code/tasks/tray_carry/runtime/wire.py) builds
    that process's environment from nothing (procutil.child_env, with wire.CHILD_THREADS threads per library) and
    Runner.start applies wire.CHILD_NUMERICS to it as its last change before the process starts;
  * what the solver is told: the paragraph on the process in environment/setup/payload/given/controller/interface.txt
    (and the grader's copy of that file) names both values.

The task sets no NPY_DISABLE_CPU_FEATURES, and nothing here asks for it.

The checks that read files need only the standard library. The run of the published Runner needs Linux and an
account other than root: as root the runtime starts the controller as an unprivileged account under a no-network
filter it requires, and this test runs the runtime as the account it is run by. The check of the kernel needs an
interpreter with numpy (this one, or ARB_NUMPY_PYTHON; skipped otherwise); it reads the kernel's name from the
OpenBLAS library the process has loaded. Nothing is downloaded.
"""
import ast
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

SLUG = "bimanual_tray_carry_controller_from_docs"
PIN = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256"}
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
RUNTIME_REL = Path("tests", "code", "tasks", "tray_carry", "runtime")
INTERFACES = (Path("environment", "setup", "payload", "given", "controller", "interface.txt"),
              Path("tests", "setup_payload", "given", "controller", "interface.txt"))


def _task_dir():
    env = os.environ.get("ARB_TASK_DIR")
    here = Path(__file__).resolve()
    for cand in ([Path(env)] if env else []) + [here.parents[1] / "tasks" / "arm" / SLUG, here.parents[1].parent / SLUG]:
        if (cand / RUNTIME_REL / "wire.py").is_file():
            return cand
    raise unittest.SkipTest("task folder %s not found" % SLUG)


TASK = _task_dir()
WIRE = TASK / RUNTIME_REL / "wire.py"


def _numpy_python():
    for py in (sys.executable, os.environ.get("ARB_NUMPY_PYTHON")):
        if py and os.path.isfile(py):
            r = subprocess.run([py, "-I", "-B", "-c", "import numpy"], capture_output=True, text=True, timeout=120)
            if r.returncode == 0:
                return py
    return None


def _dockerfile_instructions(text):
    """(INSTRUCTION, arguments) of each instruction, continuation lines joined, comment lines dropped."""
    out, cur = [], ""
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("#") or (not cur and not line):
            continue
        if line.endswith("\\"):
            cur += line[:-1] + " "
            continue
        cur += line
        word, _, rest = cur.partition(" ")
        out.append((word.upper(), rest.strip()))
        cur = ""
    if cur.strip():
        word, _, rest = cur.strip().partition(" ")
        out.append((word.upper(), rest.strip()))
    return out


def _env_pairs(args):
    """The variables one ENV instruction sets: KEY=VALUE words, or the older `ENV KEY VALUE` form."""
    words = shlex.split(args)
    if words and "=" not in words[0]:
        return {words[0]: " ".join(words[1:])}
    return dict(w.split("=", 1) for w in words)


def _module_constant(tree, name):
    """The literal value of a module-level NAME = ..., which must be assigned exactly once in the module."""
    hits = [n for n in ast.walk(tree) if isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign))
            and any(isinstance(t, ast.Name) and t.id == name
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]))]
    if len(hits) != 1 or hits[0] not in tree.body or not isinstance(hits[0], ast.Assign):
        raise AssertionError("wire.py must assign %s exactly once, at module level (found %d)" % (name, len(hits)))
    return ast.literal_eval(hits[0].value)


def _dotted(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        inner = _dotted(node.value)
        return inner + "." + node.attr if inner else None
    return None


# Reads the kernel of every OpenBLAS the process has loaded (numpy's bundled one, and scipy's when it is loaded).
KERNEL = r'''
import ctypes, os, re


def _kernels():
    import numpy
    numpy.dot(numpy.ones((64, 64)), numpy.ones((64, 64)))
    with open("/proc/self/maps") as fh:
        libs = sorted(set(re.findall(r"(/\S*openblas\S*\.so\S*)", fh.read())))
    cores = {}
    for lib in libs:
        h = ctypes.CDLL(lib)
        for sym in ("scipy_openblas_get_corename64_", "scipy_openblas_get_corename", "openblas_get_corename64_",
                    "openblas_get_corename"):
            f = getattr(h, sym, None)
            if f is not None:
                f.restype = ctypes.c_char_p
                cores[os.path.basename(lib)] = f().decode()
                break
    return {"numpy": numpy.__version__, "cores": cores}
'''

# The controller the published Runner starts: its setup answers with its own process's environment (and, when the
# site asks for it, the OpenBLAS kernel it has loaded); its steps send no frame.
CONTROLLER = KERNEL + r'''
import sys


class Controller(object):
    def setup(self, unit_record, site):
        out = {"env": dict(os.environ), "executable": sys.executable}
        if site.get("kernel"):
            out["blas"] = _kernels()
        return out

    def step(self, feedback):
        return {"frames": [None] * 16}
'''

# A process started directly with a given environment (the control of the kernel check).
PROBE = KERNEL + r'''
import json
print(json.dumps(_kernels()))
'''

# Starts the published Runner on the controller above, calls its setup once and closes it.
DRIVER = r'''
import importlib.util, json, sys
wire_path, program, run_root, stderr_path, kernel = sys.argv[1:6]
spec = importlib.util.spec_from_file_location("tray_carry_wire", wire_path)
wire = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wire)
runner = wire.Runner(program, run_root, cpu_cap_s=300, stderr_path=stderr_path)
out = {}
try:
    ok, rep = runner.start()
    out["started"] = bool(ok)
    out["start_reply"] = rep
    if ok:
        status, reply, cpu, wall = runner.call("setup", {"unit_record": {}, "site": {"kernel": kernel == "1"}},
                                               cpu_s=60.0, wall_s=120.0)
        out["status"] = status
        out["reply"] = reply
finally:
    out["stats"] = runner.close()
print(json.dumps(out))
'''


def _grader_like_env(home):
    """The grader's own process as tests/test.sh starts it (thread counts, the hash seed, an import path), except that
    the thread counts and the two pinned names hold other values: what the controller's process then holds can only
    come from the runtime, and nothing of this environment may reach it."""
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": home, "TMPDIR": home, "LANG": "C.UTF-8",
            "OMP_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3", "MKL_NUM_THREADS": "3",
            "OPENBLAS_CORETYPE": "Sandybridge", "OPENBLAS_L2_SIZE": "4096",
            "PYTHONHASHSEED": "0", "PYTHONPATH": home, "PYTHONDONTWRITEBYTECODE": "1",
            "BLAS_PIN_PROBE_VARIABLE": "x"}


class BlasPin(unittest.TestCase):
    def test_the_image_env_line_sets_both(self):
        ins = _dockerfile_instructions((TASK / "environment" / "Dockerfile").read_text(encoding="utf-8"))
        last_from = max(i for i, (word, _) in enumerate(ins) if word == "FROM")
        setting = [(i, _env_pairs(args)) for i, (word, args) in enumerate(ins)
                   if word == "ENV" and set(_env_pairs(args)) & set(PIN)]
        self.assertEqual(len(setting), 1, "exactly one ENV instruction sets the pin: %s" % setting)
        i, pairs = setting[0]
        self.assertGreater(i, last_from, "the ENV line is in the final stage")
        for k, v in PIN.items():
            self.assertEqual(pairs.get(k), v, k)
        self.assertFalse([args for word, args in ins if word == "ARG" and any(k in args for k in PIN)],
                         "no build argument of the pinned names")

    def test_the_grader_script_exports_both_before_the_grader_starts(self):
        lines = [ln for ln in (TASK / "tests" / "test.sh").read_text(encoding="utf-8").splitlines()
                 if ln.strip() and not ln.lstrip().startswith("#")]
        starts = [i for i, ln in enumerate(lines) if re.match(r"\s*exec\s.*code/tasks/tray_carry/grader/verify\.py", ln)]
        self.assertEqual(len(starts), 1, "one line starts the grader")
        named = [i for i, ln in enumerate(lines) if any(k in ln for k in PIN)]
        self.assertEqual(len(named), 1, "one line of the script names the pin: %s" % [lines[i] for i in named])
        line = lines[named[0]]
        self.assertTrue(line.lstrip().startswith("export "), line)
        self.assertLess(named[0], starts[0], "the pin is exported before the grader starts")
        words = shlex.split(line)
        for k, v in PIN.items():
            self.assertIn("%s=%s" % (k, v), words)
        if shutil.which("bash"):
            r = subprocess.run(["bash", "-c", line + '\nprintf "%s|%s" "$OPENBLAS_CORETYPE" "$OPENBLAS_L2_SIZE"'],
                               capture_output=True, text=True, timeout=30,
                               env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(r.stdout, "%s|%s" % (PIN["OPENBLAS_CORETYPE"], PIN["OPENBLAS_L2_SIZE"]))

    def test_the_runtime_constant_and_where_runner_start_applies_it(self):
        tree = ast.parse(WIRE.read_text(encoding="utf-8"))
        self.assertEqual(_module_constant(tree, "CHILD_NUMERICS"), PIN)
        self.assertEqual(_module_constant(tree, "CHILD_THREADS"), 1)
        changes = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                   and _dotted(c.func.value) == "CHILD_NUMERICS"]
        self.assertFalse(changes, "nothing calls a method of CHILD_NUMERICS")
        runner = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Runner"]
        self.assertEqual(len(runner), 1, "wire.py defines class Runner")
        start = [n for n in runner[0].body if isinstance(n, ast.FunctionDef) and n.name == "start"]
        self.assertEqual(len(start), 1, "Runner defines start")
        start = start[0]
        popen = [c for c in ast.walk(start) if isinstance(c, ast.Call) and _dotted(c.func) == "subprocess.Popen"]
        self.assertEqual(len(popen), 1, "Runner.start starts one process")
        env_kw = [k.value for k in popen[0].keywords if k.arg == "env"]
        self.assertTrue(env_kw and isinstance(env_kw[0], ast.Name), "the process gets an environment of its own")
        name = env_kw[0].id
        built = [n for n in ast.walk(start) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        self.assertEqual(len(built), 1, "the environment is bound once")
        self.assertTrue(isinstance(built[0].value, ast.Call) and (_dotted(built[0].value.func) or "").endswith(
            "child_env"), "the environment is built by procutil.child_env, from nothing")
        self.assertTrue(any(k.arg == "threads" and _dotted(k.value) == "CHILD_THREADS" for k in built[0].value.keywords),
                        "with CHILD_THREADS threads per library")
        applied = [c for c in ast.walk(start) if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                   and c.func.attr == "update" and _dotted(c.func.value) == name
                   and [_dotted(a) for a in c.args] == ["CHILD_NUMERICS"]]
        self.assertEqual(len(applied), 1, "Runner.start applies CHILD_NUMERICS to the process's environment")
        line = applied[0].lineno
        later = []
        for n in ast.walk(start):
            if n is applied[0]:
                continue
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and _dotted(n.func.value) == name:
                later += [n.lineno] if n.lineno >= line else []
            targets = (n.targets if isinstance(n, (ast.Assign, ast.Delete)) else
                       [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else [])
            for t in targets:
                if isinstance(t, ast.Subscript) and _dotted(t.value) == name and n.lineno >= line:
                    later.append(n.lineno)
        self.assertFalse(later, "the environment changes after CHILD_NUMERICS is applied (lines %s)" % later)
        self.assertLess(line, popen[0].lineno, "CHILD_NUMERICS is applied before the process starts")

    def _run_runner(self, py, kernel):
        """Starts the published Runner (as the account the test runs as) on the reporting controller; returns what
        its setup answered and, with kernel, the control: a process started with that environment without the pin."""
        if not sys.platform.startswith("linux") or not os.path.isfile("/proc/self/stat"):
            self.skipTest("the runtime reads /proc: Linux only")
        if os.getuid() == 0:
            self.skipTest("as root the runtime starts the controller as an unprivileged account; run as another account")
        d = tempfile.mkdtemp(prefix="tray_carry_blas_")
        try:
            program = os.path.join(d, "controller")
            os.makedirs(program)
            with open(os.path.join(program, "controller.py"), "w", encoding="utf-8") as fh:
                fh.write(CONTROLLER)
            home = os.path.join(d, "grader_home")
            os.makedirs(home)
            log = os.path.join(d, "controller_stderr.log")
            r = subprocess.run([py, "-I", "-B", "-c", DRIVER, str(WIRE), program, os.path.join(d, "run"), log,
                                "1" if kernel else "0"], cwd=d, env=_grader_like_env(home), capture_output=True,
                               text=True, timeout=300)
            err = Path(log).read_text(encoding="utf-8", errors="replace")[-2000:] if os.path.isfile(log) else ""
            self.assertEqual(r.returncode, 0, r.stderr[-2000:] + err)
            out = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertTrue(out.get("started"), "%s\n%s" % (out, err))
            self.assertEqual(out.get("status"), "ok", "%s\n%s" % (out, err))
            result = out["reply"]["result"]
            control = None
            if kernel:
                bare = {k: v for k, v in result["env"].items() if k not in PIN}
                p = subprocess.run([py, "-B", "-E", "-s", "-c", PROBE], cwd=d, env=bare, capture_output=True,
                                   text=True, timeout=300)
                self.assertEqual(p.returncode, 0, p.stderr[-2000:])
                control = json.loads(p.stdout.strip().splitlines()[-1])
            return result, control
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def _assert_controller_env(self, env):
        for k, v in PIN.items():
            self.assertEqual(env.get(k), v, k)
        for k in THREADS:
            self.assertEqual(env.get(k), "1", k)
        for k in ("PYTHONPATH", "PYTHONHASHSEED", "BLAS_PIN_PROBE_VARIABLE"):
            self.assertNotIn(k, env, "the controller's process inherits %s from the grader" % k)

    def test_the_controller_process_holds_the_pin(self):
        result, _ = self._run_runner(sys.executable, kernel=False)
        self._assert_controller_env(result["env"])

    def test_the_controller_process_runs_on_the_haswell_kernel(self):
        py = _numpy_python()
        if py is None:
            self.skipTest("no interpreter with numpy (set ARB_NUMPY_PYTHON)")
        result, control = self._run_runner(py, kernel=True)
        print("\n[blas pin] controller's process:", json.dumps(result["blas"], sort_keys=True))
        print("[blas pin] the same environment without the pin:", json.dumps(control, sort_keys=True))
        self.assertTrue(result["blas"]["cores"], "no OpenBLAS loaded in the controller's process")
        self.assertEqual(set(result["blas"]["cores"].values()), {PIN["OPENBLAS_CORETYPE"]},
                         "the kernel the controller's process loaded")
        self._assert_controller_env(result["env"])
        # the control: the kernel is read, not assumed (on a CPU that picks Haswell itself both read Haswell)
        self.assertTrue(control["cores"], "no OpenBLAS loaded in the control")
        self.assertEqual(set(control["cores"]), set(result["blas"]["cores"]), "the same library in both processes")

    def test_the_interface_names_both_values_as_the_runtime_gives_them(self):
        tree = ast.parse(WIRE.read_text(encoding="utf-8"))
        runtime = {k: str(v) for k, v in _module_constant(tree, "CHILD_NUMERICS").items()}
        threads = _module_constant(tree, "CHILD_THREADS")
        for rel in INTERFACES:
            text = (TASK / rel).read_text(encoding="utf-8")
            para = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("The process:")]
            self.assertEqual(len(para), 1, "%s: one paragraph on the process" % rel)
            para = para[0]
            told = {}
            for k in PIN:
                m = re.findall(r'\b%s is "?([A-Za-z0-9_]+)"?' % k, para)
                self.assertEqual(len(m), 1, "%s names %s once in the paragraph on the process" % (rel, k))
                told[k] = m[0]
            self.assertEqual(told, runtime, rel)
            self.assertEqual(told, PIN, rel)
            m = re.findall(r"OMP_NUM_THREADS, OPENBLAS_NUM_THREADS and MKL_NUM_THREADS are (\d+)", para)
            self.assertEqual(m, [str(threads)], "%s: the thread counts as the runtime sets them" % rel)


if __name__ == "__main__":
    unittest.main()
