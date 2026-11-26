"""Maintainers' check: the world modules shipped to the agent, used by the grader and used by the video renderer
are byte-identical. usage: python3 tests/review/check_copies.py (from the task folder)."""
import filecmp
import os
import sys

TASK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PAY = os.path.join(TASK, "environment", "setup", "payload")
PAIRS = [("cphys.py", "tests"), ("ctasks.py", "tests"), ("cphys.py", "environment/gen"),
         ("ctasks.py", "environment/gen"), ("carm.py", "environment/gen")]
bad = [os.path.join(d, f) for f, d in PAIRS if not filecmp.cmp(os.path.join(PAY, f), os.path.join(TASK, d, f), shallow=False)]
# the real fabric's model: hidden, one copy for the grader and one for the renderer
if not filecmp.cmp(os.path.join(TASK, "tests", "creal.py"), os.path.join(TASK, "environment", "gen", "creal.py"), shallow=False):
    bad.append("environment/gen/creal.py")
print("differ: %s" % bad if bad else "all copies identical")
sys.exit(1 if bad else 0)
