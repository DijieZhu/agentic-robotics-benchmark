"""Amendment 1: when the sandbox's memory runs out, the kernel stops one of the PROGRAM's processes, never the
grader. tests/verify.py starts the import check and every mission's program with `choom -n 1000 --`, so the program,
everything it starts and the import check run with oom_score_adj 1000 while the grader keeps the image's score.

    python3 -I -B tests/review/grader/test_oom_first.py [--tests DIR] [--site PACKAGE] [--fill]

As root in a fresh container of the task image (the verifier's 8 CPUs and 16 GB), one mission on a practice package
(tests/review/grader/stage.py). Cases:
  score   a program that refuses to import unless its oom_score_adj is 1000 (so the import check is tested too) and,
          in its mission, reports its own score, a forked child's and the grader's: contract_ok 1, both 1000, the
          grader's lower; test.sh exits 0 and writes a verdict
  fill    (--fill; needs a container whose memory has a cap and no swap: `docker run --memory 16384m --memory-swap
          16384m`) the review's case: children of 400 MiB each, pages touched, one every 0.3 s, until together they pass
          the cap while each stays smaller than the grader: the kernel kills children (memory.events oom_kill rises),
          the grader survives, test.sh exits 0 with a verdict
On version 2.0's bytes before amendment 1 (--tests pointing at that tests/ folder) `score` fails (the import check runs at 0,
contract_ok 0) and `fill` ends with test.sh killed (exit 137, no verdict).
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

from stage import Stage, report  # noqa: E402

SCORE = r'''
import os
import sys


def _adj(pid="self"):
    try:
        return open("/proc/%s/oom_score_adj" % pid).read().strip()
    except OSError:
        return "?"


# the import check imports this module: it must already run at 1000
if _adj() != "1000":
    raise SystemExit("AMEND1 oom_score_adj %s at import" % _adj())


def _grader_adj():
    pid = os.getpid()
    for _ in range(8):
        try:
            stat = open("/proc/%d/stat" % pid).read()
            pid = int(stat.rsplit(")", 1)[1].split()[1])
            cmd = open("/proc/%d/cmdline" % pid, "rb").read().replace(b"\0", b" ")
        except (OSError, ValueError, IndexError):
            return "?"
        if b"verify.py" in cmd:
            return _adj(pid)
    return "?"


def solve(robot):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.write(w, _adj().encode())
        os._exit(0)
    os.close(w)
    child = os.read(r, 16).decode()
    os.waitpid(pid, 0)
    print("AMEND1 program=%s child=%s grader=%s" % (_adj(), child, _grader_adj()), file=sys.stderr, flush=True)
    robot.step(10)
'''

FILL = r'''
import os
import sys
import time


def _cap():
    try:
        v = open("/sys/fs/cgroup/memory.max").read().strip()
        return None if v == "max" else int(v)
    except (OSError, ValueError):
        return None


def solve(robot):
    robot.step(10)
    cap = _cap() or (16 << 30)
    n = cap // (400 << 20) + 4
    for i in range(n):
        pid = os.fork()
        if pid == 0:
            try:
                b = bytearray(400 * (1 << 20))
                for j in range(0, len(b), 4096):
                    b[j] = 1
                time.sleep(600)
            finally:
                os._exit(0)
        robot.step(20)
        time.sleep(0.3)
    print("AMEND1 forked %d children of 400 MiB" % n, file=sys.stderr, flush=True)
    t0 = time.time()
    k = 0
    while time.time() - t0 < 20:
        robot.step(20)
        k += 1
    print("AMEND1 parent still here after %d more steps" % k, file=sys.stderr, flush=True)
'''


def cgroup(name):
    try:
        return open("/sys/fs/cgroup/" + name).read().strip()
    except OSError:
        return None


def oom_kills():
    ev = cgroup("memory.events") or ""
    for ln in ev.splitlines():
        k, _, v = ln.partition(" ")
        if k == "oom_kill":
            return int(v)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", default=None, help="the tests/ folder to verify with (default: this task's)")
    ap.add_argument("--site", default=None, help="a practice package (default: the image's training site t1)")
    ap.add_argument("--fill", action="store_true", help="also the memory-filling case (a capped container without swap)")
    ap.add_argument("--keep", action="store_true", help="keep the run folders (printed)")
    a = ap.parse_args(argv)
    st = Stage(a.tests, a.site, keep=a.keep)
    cases = []
    mine = open("/proc/self/oom_score_adj").read().strip()
    o = st.run({"patrol.py": SCORE}, name="score")
    line = next((ln for ln in o["program_log"].splitlines() if ln.startswith("AMEND1 program=")), "")
    got = dict(kv.split("=", 1) for kv in line[len("AMEND1 "):].split()) if line else {}
    ok = (o["rc"] == 0 and o["verdict"] is not None and o["contract_ok"] == 1.0 and got.get("program") == "1000"
          and got.get("child") == "1000" and got.get("grader") not in (None, "?", "1000"))
    cases.append(("score", ok, "test.sh exit %s, verdict %s, contract_ok %s (%s); program %s, child %s, grader %s "
                  "(this test's own %s)%s" % (o["rc"], "written" if o["verdict"] else "none", o["contract_ok"],
                                             o["contract_reason"][:160], got.get("program"), got.get("child"),
                                             got.get("grader"), mine, (" [kept %s]" % o["work"]) if a.keep else "")))
    if a.fill:
        cap, swap = cgroup("memory.max"), cgroup("memory.swap.max")
        if cap in (None, "max") or swap not in ("0",):
            cases.append(("fill", False, "needs a container with a memory cap and no swap (memory.max %s, memory.swap.max "
                          "%s): docker run --memory 16384m --memory-swap 16384m" % (cap, swap)))
        else:
            k0 = oom_kills()
            o = st.run({"patrol.py": FILL}, name="fill")
            k1 = oom_kills()
            said = [ln for ln in o["program_log"].splitlines() if ln.startswith("AMEND1")]
            ok = o["rc"] == 0 and o["verdict"] is not None and k0 is not None and k1 is not None and k1 > k0
            cases.append(("fill", ok, "memory.max %s B: test.sh exit %s, verdict %s, contract_ok %s, oom_kill %s -> %s, "
                          "wall %s s; program said %s; last grader line %r" % (
                              cap, o["rc"], "written" if o["verdict"] else "none", o["contract_ok"], k0, k1, o["wall_s"],
                              said[-2:], (o["grader"] or ["?"])[-1][:200])))
    return report(cases)


if __name__ == "__main__":
    sys.exit(main())
