"""dexterous_hand_full_assembly_16dof_panda: what a submitted program's native code writes to file descriptor 1 never reaches
the wire.

The wire between a program and the grader is the program's original stdout (one JSON request per line) and stdin (the
replies); tests/arm/assembly_cell/wire.py ends the episode on the first line that is not a JSON request. The launcher
(arm.assembly_cell.client.connect_stdio, used by arm.assembly_info.client.main) takes the wire as a duplicate of fd 1,
then points fd 1 itself at stderr (os.dup2(2, 1)) and rebinds sys.stdout to stderr, so that Python's prints AND what
native code writes to fd 1 (the given renderer's engine prints its start-up lines there through C stdio; any C library
or child process may) go to the program's log, not into the wire.

The checks (stdlib only, nothing downloaded): the three copies of the launcher's client.py (the agent's given library,
the grader's copy, the grader's copy of the givens) are byte-identical and connect_stdio redirects fd 1 after it
duplicated it; a child process started the way the grader starts a program (`python -B -E -s -c BOOTSTRAP <lib>
assemble`, its own session, stdin / stdout pipes, stderr to a file) whose solve() writes to fd 1 with os.write and with
C stdio (printf + fflush through ctypes) and prints, then makes three cell calls: the wire side receives exactly the
three JSON requests and the program's finish, and the log holds the native lines; the same child with the redirect
removed from a temporary copy of the library puts the native lines on the wire (the check sees the fault).
"""
import ctypes.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

SLUG = "dexterous_hand_full_assembly_16dof_panda"
REDIRECT = "    os.dup2(2, 1)"


def _task_dir():
    env = os.environ.get("ARB_TASK_DIR")
    here = Path(__file__).resolve()
    for cand in ([Path(env)] if env else []) + [here.parents[1] / "tasks" / "hand" / SLUG, here.parents[1].parent / SLUG]:
        if (cand / "tests" / "verify.py").is_file():
            return cand
    raise unittest.SkipTest("task folder %s not found" % SLUG)


TASK = _task_dir()
LIBS = [TASK / "environment" / "setup" / "payload" / "given" / "lib", TASK / "tests" / "setup_payload" / "given" / "lib", TASK / "tests"]
PROGRAM = '''import ctypes
import os
import sys


def solve(cell):
    os.write(1, b"native line from os.write on fd 1\\n")
    libc = ctypes.CDLL(None)
    libc.printf(b"native line from C printf\\n")
    libc.fflush(None)
    print("a Python print")
    sys.__stdout__.write("a write on sys.__stdout__\\n")
    sys.__stdout__.flush()
    cell.describe()
    cell.get_state()
    cell.step(1)
'''
NATIVE = ("native line from os.write on fd 1", "native line from C printf", "a write on sys.__stdout__")


def _bootstrap(lib):
    text = (Path(lib) / "arm" / "assembly_info" / "client.py").read_text(encoding="utf-8")
    start = text.index("BOOTSTRAP = (")
    end = text.index("\n\n", start)
    ns = {}
    exec(text[start:end], ns)  # noqa: S102 - the literal assignment of the bootstrap string only
    return ns["BOOTSTRAP"]


def _run_program(lib, workdir):
    """(wire lines as read by the grader's side, the program's stderr, return code)"""
    prog = Path(workdir) / "program"
    prog.mkdir()
    (prog / "assemble.py").write_text(PROGRAM, encoding="utf-8")
    err_path = Path(workdir) / "program.log"
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(workdir), "TMPDIR": str(workdir), "LANG": "C.UTF-8",
           "PYTHONPATH": str(lib), "PYTHONDONTWRITEBYTECODE": "1"}
    with open(err_path, "wb") as err:
        p = subprocess.Popen([sys.executable, "-B", "-E", "-s", "-c", _bootstrap(lib), str(lib), "assemble"], cwd=str(prog),
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, env=env, text=True, bufsize=1,
                             start_new_session=True)
        lines = []
        try:
            for line in p.stdout:
                lines.append(line)
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict) and "method" in obj:
                    p.stdin.write(json.dumps({"id": obj.get("id"), "result": {"ok": True}}) + "\n")
                    p.stdin.flush()
        finally:
            for f in (p.stdin, p.stdout):
                try:
                    f.close()
                except OSError:
                    pass
            rc = p.wait(timeout=60)
    return lines, err_path.read_text(encoding="utf-8", errors="replace"), rc


@unittest.skipUnless(ctypes.util.find_library("c") or sys.platform.startswith("linux"), "needs the C library through ctypes")
class WireStdout(unittest.TestCase):
    def test_the_three_launcher_copies_are_identical_and_redirect_fd1(self):
        blobs = [(lib / "arm" / "assembly_cell" / "client.py").read_bytes() for lib in LIBS]
        self.assertEqual(len(set(blobs)), 1, "the three copies of arm/assembly_cell/client.py differ")
        text = blobs[0].decode("utf-8")
        body = text[text.index("def connect_stdio():"):text.index("def main(")]
        self.assertIn('out = os.fdopen(os.dup(1), "w", buffering=1)', body)
        self.assertIn(REDIRECT + "\n", body)
        self.assertLess(body.index("os.dup(1)"), body.index(REDIRECT), "fd 1 must be duplicated for the wire before it is redirected")
        self.assertLess(body.index(REDIRECT), body.index("sys.stdout = sys.stderr"))

    def test_native_writes_to_fd1_stay_off_the_wire(self):
        for lib in LIBS[:2]:
            with self.subTest(lib=str(lib.relative_to(TASK))), tempfile.TemporaryDirectory() as td:
                lines, log, rc = _run_program(lib, td)
                self.assertEqual(rc, 0, log)
                reqs = [json.loads(x) for x in lines]            # every line on the wire is JSON ...
                self.assertTrue(all(isinstance(r, dict) and "method" in r for r in reqs), lines)   # ... and a request
                self.assertEqual([r["method"] for r in reqs], ["describe", "get_state", "step", "finish"])
                for s in NATIVE + ("a Python print",):
                    self.assertIn(s, log)

    def test_without_the_redirect_the_native_lines_reach_the_wire(self):
        with tempfile.TemporaryDirectory() as td:
            lib = Path(td) / "lib"
            shutil.copytree(LIBS[0] / "arm", lib / "arm", ignore=shutil.ignore_patterns("__pycache__"))
            cp = lib / "arm" / "assembly_cell" / "client.py"
            text = cp.read_text(encoding="utf-8")
            self.assertEqual(text.count(REDIRECT + "\n"), 1)
            cp.write_text(text.replace(REDIRECT + "\n", ""), encoding="utf-8")
            lines, log, rc = _run_program(lib, td)
            bad = [x.strip() for x in lines if not x.strip().startswith("{")]
            self.assertIn(NATIVE[0], bad)
            self.assertIn(NATIVE[1], bad)


if __name__ == "__main__":
    unittest.main()
