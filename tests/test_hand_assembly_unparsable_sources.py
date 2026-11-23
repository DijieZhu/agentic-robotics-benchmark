"""dexterous_hand_full_assembly_16dof_panda: the network audit fails closed on a source it cannot read.

The grader's network audit (tests/arm/assembly_info/netaudit.py `scan_imports`) reads the submission's .py files and
lists every networking import. A file the parser cannot handle must be a hit, never a crash of the grader (no verdict)
and never a silent skip (an import the audit did not see). The checks (stdlib only, nothing downloaded, the audit loaded
from the task folder by path):
  * the three copies of netaudit.py (the grader's, the grader's copy of the givens, the agent's given library) are
    byte-identical;
  * a 100 000-deep expression (RecursionError in the parser), a null byte, bytes that are not UTF-8 and an unknown
    declared encoding are each a hit `cannot be parsed (...)`;
  * a file that declares a text encoding other than UTF-8 / ASCII / Latin-1 (punycode, utf-7, cp1252) is a hit
    `declares the text encoding <name>; ...`, decided from the declaration alone (so a utf-7 file can no longer hide an
    import in its body); a file in Latin-1 with its declaration and a non-ASCII byte, and a plain UTF-8 file, are
    accepted;
  * a FIFO, a symbolic link and a directory named like a source are not opened (the scan returns at once, a FIFO
    never blocks it) and give no hit;
  * past the size limit (lstat, before any read) the scan stops with one hit and never opens the file that passes it;
  * a plain source gives no hit;
  * the per-file scan the grader runs in its separate process (`source_hits`, `tree_hits`) gives the hits
    `scan_imports` gives;
  * the sampler keeps the memory killer's order: with `oom_floor` it KILLS a process that reads below the floor and
    counts it (`oom_killed`), while a process named like the launch chain (prlimit) is spared.
"""
import ast
import builtins
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

SLUG = "dexterous_hand_full_assembly_16dof_panda"
REL = Path("arm") / "assembly_info" / "netaudit.py"


def _task_dir():
    env = os.environ.get("ARB_TASK_DIR")
    here = Path(__file__).resolve()
    for cand in ([Path(env)] if env else []) + [here.parents[1] / "tasks" / "hand" / SLUG, here.parents[1].parent / SLUG]:
        if (cand / "tests" / "verify.py").is_file():
            return cand
    raise unittest.SkipTest("task folder %s not found" % SLUG)


TASK = _task_dir()
COPIES = [TASK / "tests" / REL, TASK / "tests" / "setup_payload" / "given" / "lib" / REL,
          TASK / "environment" / "setup" / "payload" / "given" / "lib" / REL]


def _netaudit():
    spec = importlib.util.spec_from_file_location("hand_assembly_netaudit_under_test", str(COPIES[0]))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NA = _netaudit()
DEEP = b"x = 1" + b"+1" * 100000 + b"\n"


class TestNetauditFailsClosed(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp(prefix="hand_assembly_audit_")

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, name, data):
        p = os.path.join(self.root, name)
        with open(p, "wb") as fh:
            fh.write(data)
        return p

    def _scan(self, **kw):
        """scan_imports in a thread with a deadline (a FIFO that were opened would block it for ever)."""
        box = {}
        t = threading.Thread(target=lambda: box.setdefault("out", NA.scan_imports(self.root, **kw)), daemon=True)
        t.start()
        t.join(30)
        self.assertFalse(t.is_alive(), "scan_imports did not return within 30 s")
        return box["out"]

    def test_the_three_copies_are_identical(self):
        digests = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in COPIES}
        self.assertEqual(len(set(digests.values())), 1, digests)

    def test_a_deep_expression_is_a_hit(self):
        self._write("assemble.py", b"def solve(cell):\n    cell.step(1)\n")
        self._write("deep.py", DEEP)
        self.assertEqual(self._scan(), [("deep.py", "cannot be parsed (RecursionError)")])

    def test_a_null_byte_bad_bytes_and_an_unknown_encoding_are_hits(self):
        self._write("a_nul.py", b"x = 1\n\x00\n")
        self._write("b_latin.py", b"s = '\xe9t\xe9'\n")
        self._write("c_enc.py", b"# coding: no-such-codec\nx = 1\n")
        out = self._scan()
        self.assertEqual([r for r, _ in out], ["a_nul.py", "b_latin.py", "c_enc.py"], out)
        for _rel, what in out:
            self.assertTrue(what.startswith("cannot be parsed ("), out)

    def test_a_disallowed_declared_encoding_is_a_hit_and_cannot_hide_an_import(self):
        # utf-7 hid an import in an earlier grader; now the declaration alone is refused, before the body is decoded
        self._write("helper.py", b"# coding: utf-7\n+AGk-mport socket\n")
        self.assertEqual(self._scan(), [("helper.py", "declares the text encoding utf-7; sources must be UTF-8, ASCII or Latin-1")])

    def test_punycode_and_cp1252_declarations_are_hits(self):
        self._write("a_puny.py", b"# coding: punycode\nx-\n")
        self._write("b_cp.py", b"# -*- coding: cp1252 -*-\nx = 1\n")
        out = dict(self._scan())
        self.assertEqual(out["a_puny.py"], "declares the text encoding punycode; sources must be UTF-8, ASCII or Latin-1")
        self.assertEqual(out["b_cp.py"], "declares the text encoding cp1252; sources must be UTF-8, ASCII or Latin-1")

    def test_a_latin1_declaration_with_a_non_ascii_byte_is_accepted(self):
        # a Latin-1 source with its declaration decodes and parses; a UTF-8 source with a BOM is fine too
        self._write("assemble.py", b"# coding: latin-1\n# a comment with \xe9\n\n\ndef solve(cell):\n    cell.step(1)\n")
        self._write("ok_bom.py", b"\xef\xbb\xbfimport math\n")
        self.assertEqual(self._scan(), [])

    def test_fifo_symlink_and_directory_named_like_a_source_are_not_read(self):
        self._write("target.txt", b"import socket\n")
        os.mkfifo(os.path.join(self.root, "fifo.py"))
        os.symlink(os.path.join(self.root, "target.txt"), os.path.join(self.root, "link.py"))
        os.mkdir(os.path.join(self.root, "pkg.py"))
        self._write(os.path.join("pkg.py", "inner.py"), b"x = 1\n")
        self.assertEqual(self._scan(), [])

    def test_the_size_limit_stops_the_scan_before_the_read(self):
        self._write("a_small.py", b"x = 1\n")
        big = self._write("b_big.py", b"#" * 5000 + b"\n")
        opened = []
        real_open = builtins.open

        def tracking_open(path, *a, **kw):
            opened.append(os.path.abspath(str(path)))
            return real_open(path, *a, **kw)
        NA.open = tracking_open          # the module's global shadows the builtin
        try:
            out = self._scan(max_bytes=1000)
        finally:
            del NA.open
        self.assertEqual(out, [("b_big.py", "beyond the audit's size limit")])
        self.assertNotIn(os.path.abspath(big), opened)

    def test_a_plain_source_gives_no_hit(self):
        self._write("assemble.py", b"import math\n\n\ndef solve(cell):\n    cell.step(int(math.pi))\n")
        self.assertEqual(self._scan(), [])



class TestPerFileScanAndOomFloor(unittest.TestCase):
    def test_the_per_file_scan_gives_the_hits_of_scan_imports(self):
        tmp = tempfile.mkdtemp(prefix="hand_assembly_na_")
        try:
            files = {"a.py": b"import socket\nimport os\nos.system('x')\n", "b.py": b"x = 1\n",
                     "c.py": DEEP, "d.py": b"from urllib import request\n__import__(name)\n"}
            for fn, data in files.items():
                with open(os.path.join(tmp, fn), "wb") as fh:
                    fh.write(data)
            per_file = []
            for fn in sorted(files):
                per_file += NA.source_hits(fn, files[fn])
            self.assertEqual(NA.scan_imports(tmp), per_file)
            self.assertEqual(NA.tree_hits("a.py", ast.parse(files["a.py"])), NA.source_hits("a.py", files["a.py"]))
            self.assertIn(("c.py", "cannot be parsed (RecursionError)"), per_file)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_the_sampler_kills_a_process_below_the_floor(self):
        with open("/proc/self/oom_score_adj") as fh:
            own = int(fh.read().strip())
        if own >= 1000:
            self.skipTest("this process already runs at oom_score_adj 1000: nothing below the floor to kill")
        child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(30)"])  # inherits own (< 1000)
        try:
            self.assertEqual(int(open("/proc/%d/oom_score_adj" % child.pid).read().strip()), own)
            smp = NA.SocketSampler(pids_fn=lambda: [child.pid], period_s=0.05, oom_floor=1000)
            with smp:
                deadline = time.time() + 10.0
                while child.poll() is None and time.time() < deadline:
                    time.sleep(0.05)
            self.assertEqual(child.poll(), -9, "a process below the floor was not SIGKILLed")
            # at least one: until this loop reaps the killed child it is a zombie whose value still reads below the
            # floor, so a pass of the sampler that comes before the reap signals it again and counts it again
            self.assertGreaterEqual(smp.oom_killed, 1)
            self.assertGreaterEqual(smp.samples, 1)
        finally:
            child.kill()
            child.wait()

    def test_the_sampler_spares_a_launch_chain_process(self):
        with open("/proc/self/oom_score_adj") as fh:
            own = int(fh.read().strip())
        if own >= 1000:
            self.skipTest("this process already runs at oom_score_adj 1000: nothing below the floor to kill")
        child = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(5)"])  # at own (< 1000)
        real = NA._proc_comm
        NA._proc_comm = lambda pid: "prlimit"           # the launch chain's helper, before choom writes the floor
        try:
            smp = NA.SocketSampler(pids_fn=lambda: [child.pid], period_s=0.05, oom_floor=1000)
            with smp:
                time.sleep(1.0)
            self.assertIsNone(child.poll(), "a launch-chain-named process was killed")
            self.assertEqual(smp.oom_killed, 0)
        finally:
            NA._proc_comm = real
            child.kill()
            child.wait()


if __name__ == "__main__":
    unittest.main()
