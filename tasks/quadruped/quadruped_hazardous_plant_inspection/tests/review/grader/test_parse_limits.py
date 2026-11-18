"""Amendment 1: a .py file of the submission that the parser cannot read (beyond CPython's limits ast.parse
raises RecursionError or MemoryError, not SyntaxError) fails the contract (contract_ok 0, the reason in the
diagnostic); it never stops the grader. tests/verify.py catches those beside SyntaxError where it parses patrol.py,
and parses every .py file of the submission itself before the network audit (whose shared-kit copy catches only
SyntaxError), so that no such file reaches the audit unparsed.

    python3 -I -B tests/review/grader/test_parse_limits.py [--tests DIR] [--site PACKAGE]

As root in a fresh container of the task image, on a practice package (tests/review/grader/stage.py). Cases (the
submissions of the engineer who re-checked the grading-environment review, rebuilt here):
  deep_helper   a trivial patrol.py and a table.py that nothing imports, `X = 1 + 1 + ...` with 100,000 terms
  deep_patrol   the same chain inside patrol.py
  deep_unary    a helper with 50,000 unary minus signs (MemoryError: the parser's stack)
  at_limit      patrol.py imports a table.py whose chain is the shortest one this interpreter's ast.parse refuses (it
                may still compile: the one-term window the review found)
                -> each: test.sh exit 0, a verdict, contract_ok 0, the file and the error named in the diagnostic
  syntax_helper a trivial patrol.py and a helper with a plain SyntaxError that nothing imports: as before amendment 1,
                the audit's rule (a file no program can import) -> contract_ok 1, the mission runs
On version 2.0's bytes before amendment 1 the first four end with test.sh exit 1 (an exception in the grader, no verdict).
"""

import argparse
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True

from stage import Stage, report  # noqa: E402

TRIVIAL = "def solve(robot):\n    robot.step(10)\n"


def chain(n):
    return "X = " + " + ".join(["1"] * n) + "\n"


def parse_fails(src):
    try:
        ast.parse(src)
        return False
    except (RecursionError, MemoryError):
        return True


def shortest_refused():
    lo, hi = 100, 200000                      # ast.parse reads lo terms and refuses hi
    while hi - lo > 1:
        m = (lo + hi) // 2
        if parse_fails(chain(m)):
            hi = m
        else:
            lo = m
    try:
        compile(chain(hi), "table.py", "exec")
        compiles = True
    except (RecursionError, MemoryError, SyntaxError):
        compiles = False
    return hi, compiles


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", default=None, help="the tests/ folder to verify with (default: this task's)")
    ap.add_argument("--site", default=None, help="a practice package (default: the image's training site t1)")
    ap.add_argument("--keep", action="store_true", help="keep the run folders (printed)")
    a = ap.parse_args(argv)
    st = Stage(a.tests, a.site, keep=a.keep)
    n_lim, compiles = shortest_refused()
    subs = [
        ("deep_helper", {"patrol.py": TRIVIAL, "table.py": chain(100000)}, "table.py", "RecursionError"),
        ("deep_patrol", {"patrol.py": chain(100000) + TRIVIAL}, "patrol.py", "RecursionError"),
        ("deep_unary", {"patrol.py": TRIVIAL, "helper.py": "Y = " + "-" * 50000 + "1\n"}, "helper.py", "MemoryError"),
        ("at_limit", {"patrol.py": "import table\n\n\n" + TRIVIAL, "table.py": chain(n_lim)}, "table.py", "Error"),
    ]
    cases = []
    for name, files, culprit, err in subs:
        o = st.run(files, name=name)
        diag = (o["verdict"] or {}).get("diagnostic") or ""
        ok = (o["rc"] == 0 and o["verdict"] is not None and o["contract_ok"] == 0.0 and culprit in o["contract_reason"]
              and err in o["contract_reason"] and culprit in diag)
        extra = " (%d terms; compiles: %s)" % (n_lim, compiles) if name == "at_limit" else ""
        cases.append((name, ok, "test.sh exit %s, verdict %s, contract_ok %s, reason %r%s%s" % (
            o["rc"], "written" if o["verdict"] else "none", o["contract_ok"], o["contract_reason"][:200], extra,
            "" if o["verdict"] else "; last grader lines %r" % [ln[:160] for ln in o["grader"][-2:]])))
    o = st.run({"patrol.py": TRIVIAL, "broken.py": "def f(:\n    pass\n"}, name="syntax_helper")
    ok = o["rc"] == 0 and o["verdict"] is not None and o["contract_ok"] == 1.0
    cases.append(("syntax_helper", ok, "test.sh exit %s, verdict %s, contract_ok %s (%s)" % (
        o["rc"], "written" if o["verdict"] else "none", o["contract_ok"], o["contract_reason"][:160])))
    return report(cases)


if __name__ == "__main__":
    sys.exit(main())
