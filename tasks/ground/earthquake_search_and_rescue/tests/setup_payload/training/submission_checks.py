"""The checks a submission must pass before the grader runs it on any site, and the regime it then runs under.

The grader loads THIS file (its own root-owned copy) and applies `check_submission` to your submission folder; a
submission that fails one check scores 0 on every site. `~/training/run_mission.py` applies the same checks before it
uses a training mission, so a submission the grader would refuse never gets as far as a mission here.

    /opt/venv/bin/python3 ~/training/submission_checks.py [~/submission]      # check only; uses no training mission

The checks (instruction.md, "What a submission must satisfy"):
  1. rescue.py is a regular file that parses as Python and defines `solve`;
  2. no model-weight files (FORBIDDEN_EXTENSIONS), no training calls in .py / .sh files (FORBIDDEN_STRINGS: the text
     itself is searched, comments and strings included), the folder at most MAX_SUBMISSION_MB;
  3. the .py files import no networking or process-starting module (FORBIDDEN_MODULES), import no module under a
     computed name, run no computed code (exec / eval / compile of a non-literal) and start no other program
     (PROCESS_CALLS); a file byte-identical to a file of the given client library is exempt;
  4. `import rescue` succeeds within IMPORT_WALL_S seconds and IMPORT_CPU_S CPU-seconds in the run regime below.
The run regime (`run_environment`, `nonet_argv`): python -B -E -s, the working directory a READ-ONLY copy of the
submission (regular files only; symbolic links and __pycache__ dropped), an empty writable HOME and TMPDIR, the given
client library on PYTHONPATH, no network (a seccomp filter: only AF_UNIX sockets), and the numerics pin
(`numerics_pin`: one BLAS thread on fixed kernels, NumPy without its AVX-512 loops, OpenCV's IPP at AVX2), so that a
program computes the same on the training ground and on the grading machine.
During the missions the grader also requires: at least one `step` call over the missions, no process of the program
left behind after a mission, no remote network socket.
"""

import ast
import hashlib
import os
import sys

FORBIDDEN_EXTENSIONS = (".pt", ".pth", ".onnx", ".safetensors", ".ckpt", ".h5", ".pb", ".pkl", ".npz")
FORBIDDEN_STRINGS = ("torch.optim", ".backward(", "jax.grad")
MAX_SUBMISSION_MB = 60
IMPORT_WALL_S = 120
IMPORT_CPU_S = 120
PROGRAM = "rescue"

FORBIDDEN_MODULES = ("socket", "_socket", "ssl", "_ssl", "urllib", "urllib2", "urllib3", "http", "httplib", "requests", "httpx",
                     "aiohttp", "ftplib", "smtplib", "imaplib", "poplib", "nntplib", "telnetlib", "websocket", "websockets", "paramiko",
                     "pycurl", "xmlrpc", "asyncio", "socketserver", "grpc", "zmq", "huggingface_hub", "boto3", "botocore", "wget",
                     "ctypes", "_ctypes", "cffi", "selectors", "multiprocessing.connection", "multiprocessing.managers",
                     "logging.handlers", "subprocess", "pty", "_posixsubprocess")
# calls that start another program (attribute or plain names; `os.` calls by their attribute)
PROCESS_CALLS = ("system", "popen", "execv", "execve", "execvp", "execvpe", "execl", "execle", "execlp", "execlpe", "spawnv", "spawnve",
                 "spawnvp", "spawnvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "posix_spawn", "posix_spawnp", "fork_exec")
DYNAMIC_IMPORT_CALLS = ("__import__", "import_module", "find_spec", "spec_from_file_location", "load_module")
CODE_CALLS = ("exec", "eval", "compile")
PYTHON_FLAGS = ["-B", "-E", "-s"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def library_digests(lib_package_dir):
    """sha256 of every .py file of the given client library (a submission may carry copies of it)."""
    out = set()
    for dp, _dn, fns in os.walk(lib_package_dir):
        for fn in fns:
            if fn.endswith(".py"):
                out.add(sha256_file(os.path.join(dp, fn)))
    return out


def defines_solve(src):
    """(ok, why): rescue.py is a regular file that parses and defines solve."""
    path = os.path.join(src, PROGRAM + ".py")
    if not os.path.isfile(path) or os.path.islink(path):
        return False, "no %s.py (regular file) in the submission" % PROGRAM
    try:
        tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
    except SyntaxError as e:
        return False, "%s.py does not parse: %s" % (PROGRAM, e)
    has_solve = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "solve" for n in ast.walk(tree)) or \
        any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "solve" for t in n.targets) for n in ast.walk(tree)) or \
        any(isinstance(n, (ast.Import, ast.ImportFrom)) and any((a.asname or a.name) == "solve" for a in n.names) for n in ast.walk(tree))
    if not has_solve:
        return False, "%s.py defines no solve" % PROGRAM
    return True, "ok"


def route_gate(src):
    """[problem]: weight files, training calls, the size cap."""
    total, problems = 0, []
    for dp, _, fns in os.walk(src):
        for fn in fns:
            p = os.path.join(dp, fn)
            try:
                total += os.lstat(p).st_size
            except OSError:
                continue
            if fn.lower().endswith(FORBIDDEN_EXTENSIONS):
                problems.append("weight file %s" % os.path.relpath(p, src))
            if fn.endswith((".py", ".sh")) and not os.path.islink(p):
                try:
                    text = open(p, encoding="utf-8", errors="replace").read()
                except OSError:
                    continue
                for c in FORBIDDEN_STRINGS:
                    if c in text:
                        problems.append("%s contains the text %s" % (os.path.relpath(p, src), c))
    if total > MAX_SUBMISSION_MB * 1024 * 1024:
        problems.append("submission larger than %d MB" % MAX_SUBMISSION_MB)
    return problems


def _hit(name):
    parts = (name or "").split(".")
    return any(".".join(parts[:k]) in FORBIDDEN_MODULES for k in range(1, len(parts) + 1))


def _literal(node):
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def scan_imports(root, allow_digests=()):
    """[(relpath, what)] of forbidden imports and calls in the .py files under `root` (symbolic links skipped)."""
    allow = set(allow_digests or ())
    out = []
    for dp, dns, fns in os.walk(root):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in sorted(fns):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dp, fn)
            if os.path.islink(p):
                continue
            try:
                data = open(p, "rb").read()
            except OSError:
                continue
            if hashlib.sha256(data).hexdigest() in allow:
                continue
            rel = os.path.relpath(p, root)
            try:
                tree = ast.parse(data.decode("utf-8", "replace"))
            except SyntaxError:
                continue
            for n in ast.walk(tree):
                if isinstance(n, ast.Import):
                    out += [(rel, a.name) for a in n.names if _hit(a.name)]
                elif isinstance(n, ast.ImportFrom):
                    if n.level == 0 and _hit(n.module):
                        out.append((rel, n.module))
                    elif n.level == 0 and n.module and any(_hit(n.module + "." + a.name) for a in n.names):
                        out.append((rel, n.module + "." + ",".join(a.name for a in n.names)))
                elif isinstance(n, ast.Call):
                    f = n.func
                    fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
                    owner = f.value.id if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) else ""
                    arg0 = n.args[0] if n.args else None
                    if fname in DYNAMIC_IMPORT_CALLS:
                        if arg0 is None or not _literal(arg0):
                            if fname not in ("spec_from_file_location", "load_module"):
                                out.append((rel, "%s(<computed name>)" % fname))
                        elif _hit(arg0.value):
                            out.append((rel, arg0.value))
                    elif fname in CODE_CALLS and isinstance(f, ast.Name) and (arg0 is None or not _literal(arg0)):
                        out.append((rel, "%s(<computed code>)" % fname))
                    elif fname in PROCESS_CALLS and (owner in ("os", "posix", "_posixsubprocess") or isinstance(f, ast.Name)):
                        out.append((rel, "%s.%s (starts a program)" % (owner or "", fname)))
    return out


def check_submission(src, lib_package_dir=None):
    """(ok, why, forbidden imports): checks 1 to 3 on the folder `src` (check 4, the import, needs a process: see
    `import_check_argv`)."""
    ok, why = defines_solve(src)
    if not ok:
        return False, why, []
    problems = route_gate(src)
    if problems:
        return False, "route gate: " + "; ".join(problems[:5]), []
    net = scan_imports(src, allow_digests=library_digests(lib_package_dir) if lib_package_dir else ())
    if net:
        return False, "import audit: forbidden import(s) or call(s) %s" % ", ".join("%s: %s" % x for x in net[:5]), net
    return True, "ok", []


# ---------------------------------------------------------------------------------------------- the run regime
# Run as `python3 -I -B -c NONET_SRC -- <argv...>`: sets PR_SET_NO_NEW_PRIVS, installs a seccomp filter under which
# socket() and socketpair() succeed only for AF_UNIX, io_uring_setup fails and a non-native syscall ABI fails, then
# execs argv. The filter is inherited by every descendant and cannot be removed.
NONET_SRC = r'''
import ctypes, os, platform, struct, sys
ARCH = {"x86_64": (0xC000003E, 41, 53, 425, 0x40000000), "aarch64": (0xC00000B7, 198, 199, 425, 0)}
def ins(code, jt, jf, k):
    return struct.pack("HBBI", code, jt, jf, k)
def main():
    audit, nr_socket, nr_socketpair, nr_uring, x32 = ARCH[platform.machine()]
    LD, JEQ, JGE, RET = 0x20, 0x15, 0x35, 0x06
    allow, deny = 0x7FFF0000, 0x00050000 | 1
    prog = [ins(LD, 0, 0, 4), ins(JEQ, 1, 0, audit), ins(RET, 0, 0, deny), ins(LD, 0, 0, 0)]
    if x32:
        prog += [ins(JGE, 0, 1, x32), ins(RET, 0, 0, deny)]
    prog += [ins(JEQ, 0, 1, nr_uring), ins(RET, 0, 0, deny), ins(JEQ, 2, 0, nr_socket), ins(JEQ, 1, 0, nr_socketpair),
             ins(RET, 0, 0, allow), ins(LD, 0, 0, 16), ins(JEQ, 0, 1, 1), ins(RET, 0, 0, allow), ins(RET, 0, 0, deny)]
    raw = b"".join(prog)
    buf = ctypes.create_string_buffer(raw, len(raw))
    class Fprog(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_void_p)]
    fp = Fprog(len(raw) // 8, ctypes.cast(buf, ctypes.c_void_p))
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(22, 2, ctypes.byref(fp), 0, 0) != 0:
        sys.stderr.write("[nonet] the seccomp filter could not be installed (errno %d)\n" % ctypes.get_errno())
        os._exit(97)
    argv = sys.argv[1:]
    if argv and argv[0] == "--":
        argv = argv[1:]
    os.execvp(argv[0], argv)
main()
'''


def nonet_argv(argv, python=None):
    """argv run under the no-network filter."""
    return [python or sys.executable, "-I", "-B", "-c", NONET_SRC, "--"] + list(argv)


def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def numerics_pin():
    """The variables that make a program's own arithmetic the same on every x86-64 machine: NumPy's linear algebra on
    one fixed set of kernels and one thread (a threaded product is split by the host's core count), NumPy without its
    AVX-512 loops (they round arctan2, exp, log and others differently), OpenCV's IPP at AVX2."""
    pin = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_L2_SIZE": "256", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "NPY_DISABLE_CPU_FEATURES": "X86_V4 AVX512_ICL AVX512_SPR", "OPENCV_IPP": "avx2"}
    if "avx512f" in _cpu_flags():
        pin["OPENCV_CPU_DISABLE"] = "AVX512-SKX"
    return pin


def run_environment(home, tmpdir, lib_dir, given_dir):
    """The program's whole environment: nothing of the caller's except PATH; the numerics pin."""
    env = {"PATH": os.environ.get("PATH", "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"),
           "HOME": home, "TMPDIR": tmpdir, "TMP": tmpdir, "TEMP": tmpdir, "LANG": "C.UTF-8", "MPLBACKEND": "Agg",
           "PYTHONPATH": lib_dir, "ARB_GIVEN_DIR": given_dir, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa",
           "LP_NUM_THREADS": "2"}
    env.update(numerics_pin())
    return env


def import_check_argv(python, lib_dir, program_dir):
    """argv of check 4 (run it with cwd = program_dir, the run environment, IMPORT_WALL_S / IMPORT_CPU_S)."""
    code = ("import sys, importlib; sys.path.insert(0, sys.argv[1]); sys.path.insert(0, sys.argv[2]); "
            "m = importlib.import_module('%s'); assert callable(m.solve)" % PROGRAM)
    return [python] + PYTHON_FLAGS + ["-c", code, lib_dir, program_dir]


def main():
    src = os.path.abspath(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/submission"))
    here = os.path.dirname(os.path.abspath(__file__))
    lib = os.path.join(os.path.dirname(here), "given", "lib", "rubblecell")
    ok, why, _net = check_submission(src, lib if os.path.isdir(lib) else None)
    print("%s: %s" % (src, "checks 1-3 pass (run_mission.py also runs the import check)" if ok else "REFUSED: " + why))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
