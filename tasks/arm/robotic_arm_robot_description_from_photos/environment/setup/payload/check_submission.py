#!/usr/bin/env python3
"""The contract gate of the robot-description deliverable (the grader's own gate module; public rules, no ground truth).

    /opt/venv/bin/python3 check_submission.py [DIR]      # default /home/user/submission
    options: --json FILE (write the verdict and every violation)

Prints PASS or FAIL and every violation. The gate is the same code the grader runs before any criterion; it does not
look at the arm's ground truth and cannot tell how accurate a description is.
"""
import argparse
import json
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "rd_info_check"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--json", default=None)
    ap.add_argument("dir", nargs="?", default=os.path.join(os.environ.get("HOME", "/home/user"), "submission"))
    a = ap.parse_args(argv)
    import rd_info_gate as GT
    ok, det, _models = GT.gate_model_loads(os.path.abspath(a.dir))
    print("gate model_loads: {}".format("PASS" if ok else "FAIL"))
    for v in det.get("violations", []):
        print("  - " + v)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"model_loads": 1.0 if ok else 0.0, "details": det}, f, indent=1, default=str)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
