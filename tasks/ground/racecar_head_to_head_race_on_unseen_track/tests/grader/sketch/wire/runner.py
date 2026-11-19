"""How a program is started for one venue in the sketch (the graded wrapping vendors the house `cellkit` isolation:
`runner.py`, `nonet.py`, `lockdown.py`).

What the sketch already does: a new process per venue, its own session (process group, killed at the end), a fresh
run directory with a read-only copy of the program (regular files only; links and __pycache__ dropped) and of
`carkit`, an empty writable HOME / TMPDIR, a minimal environment (PATH, HOME, TMPDIR, LANG, PYTHONPATH = the copy of
carkit, numeric libraries single-threaded and NPY_DISABLE_CPU_FEATURES as in every compared process, the OpenBLAS kernel
never pinned), the interpreter with `-B -E -s` through the carkit launcher, and `prlimit --cpu` as a
hard backstop above the CPU budget when prlimit exists. Not yet: another account (`nobody`), seccomp, the stage lock.

    result, venue, stats = run_venue(world, program_dir, config=cfg)
"""

import os
import shutil
import stat
import sys
import tempfile

from carkit.client import launcher_argv as _carkit_argv

from sketch.referee import Venue

from .server import serve

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.dirname(os.path.dirname(HERE))
CARKIT_DIR = os.path.join(DEV, "carkit")


def copy_regular_files(src, dst):
    """Copy a tree keeping regular files and directories (symbolic links, special files, __pycache__ and tests/
    skipped). Returns the number of skipped entries."""
    skipped = 0
    os.makedirs(dst, exist_ok=True)
    for dirpath, dirnames, filenames in os.walk(src, followlinks=False):
        rel = os.path.relpath(dirpath, src)
        target = dst if rel == "." else os.path.join(dst, rel)
        keep = []
        for d in dirnames:
            p = os.path.join(dirpath, d)
            if d in ("__pycache__",) or os.path.islink(p):
                skipped += 1 if os.path.islink(p) else 0
                continue
            os.makedirs(os.path.join(target, d), exist_ok=True)
            keep.append(d)
        dirnames[:] = keep
        for f in filenames:
            p = os.path.join(dirpath, f)
            if os.path.islink(p) or not stat.S_ISREG(os.lstat(p).st_mode):
                skipped += 1
                continue
            shutil.copyfile(p, os.path.join(target, f))
    return skipped


def _read_only(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            os.chmod(os.path.join(dirpath, f), 0o444)
        os.chmod(dirpath, 0o555)


def prepare_run_dir(root, program_dir, carkit_dir=CARKIT_DIR):
    """root/program (read-only copy of the program), root/lib/carkit (read-only copy of the client), root/home and
    root/tmp (empty, writable). Returns the paths."""
    os.makedirs(root, exist_ok=True)
    program = os.path.join(root, "program")
    skipped = copy_regular_files(program_dir, program)
    lib = os.path.join(root, "lib")
    copy_regular_files(carkit_dir, os.path.join(lib, "carkit"))
    tests = os.path.join(lib, "carkit", "tests")
    if os.path.isdir(tests):
        shutil.rmtree(tests)
    _read_only(program)
    _read_only(lib)
    out = {"root": root, "program": program, "lib": lib, "skipped": skipped}
    for name in ("home", "tmp"):
        d = os.path.join(root, name)
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o700)
        out[name] = d
    return out


NPY_DISABLE_CPU_FEATURES = "X86_V4 AVX512_ICL AVX512_SPR"


def child_env(home, tmpdir, lib_dir, threads=1):
    """The program's whole environment. The three variables of every compared run: one thread for
    OpenMP / OpenBLAS (and MKL), NPY_DISABLE_CPU_FEATURES; OPENBLAS_CORETYPE is never passed."""
    return {"PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"), "HOME": home, "TMPDIR": tmpdir, "TMP": tmpdir,
            "TEMP": tmpdir, "LANG": "C.UTF-8", "PYTHONPATH": lib_dir, "MPLBACKEND": "Agg",
            "OMP_NUM_THREADS": str(int(threads)), "OPENBLAS_NUM_THREADS": str(int(threads)),
            "MKL_NUM_THREADS": str(int(threads)), "NPY_DISABLE_CPU_FEATURES": NPY_DISABLE_CPU_FEATURES}


def program_argv(lib_dir, module="drive", python=None, cpu_s=None):
    argv = _carkit_argv(python or sys.executable, lib_dir, module)
    if cpu_s is not None and shutil.which("prlimit"):
        argv = ["prlimit", "--cpu=%d" % int(cpu_s), "--core=0", "--"] + argv
    return argv


def cleanup(root):
    """Remove a run directory (its copies are read-only)."""
    for dirpath, _dirnames, _filenames in os.walk(root):
        os.chmod(dirpath, 0o755)
    shutil.rmtree(root, ignore_errors=True)


def run_venue(world, program_dir, config=None, providers="auto", module="drive", run_root=None, stderr_path=None,
              priv_token_sha256=None, keep=False, log=None, venue=None):
    """One venue: a fresh cell, a fresh process of the program, the wire until either ends.
    Returns (result record, venue, wire stats)."""
    venue = venue or Venue(world, config, providers=providers, priv_token_sha256=priv_token_sha256)
    root = run_root or tempfile.mkdtemp(prefix="s6run_")
    paths = prepare_run_dir(os.path.join(root, "run"), program_dir)
    env = child_env(paths["home"], paths["tmp"], paths["lib"])
    argv = program_argv(paths["lib"], module=module, cpu_s=venue.caps["cpu"] + 60.0)
    stats = {}
    try:
        serve(venue, argv, cwd=paths["program"], env=env, stderr_path=stderr_path or os.path.join(paths["root"], "stderr.log"),
              log=log, stats=stats)
        if stderr_path is None:
            try:
                with open(os.path.join(paths["root"], "stderr.log"), "r", encoding="utf-8", errors="replace") as f:
                    stats["stderr_tail"] = f.read()[-4000:]
            except OSError:
                stats["stderr_tail"] = ""
    finally:
        if not keep:
            cleanup(root if run_root is None else paths["root"])
    return venue.result(), venue, stats
