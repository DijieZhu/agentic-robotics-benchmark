#!/opt/venv/bin/python3
"""Practice grader: runs YOUR program over the real wire against the practice cell on visible seeds and prints the raw
work score with the per-pair milestones, nothing else (the hidden grader scores the same way on hidden seeds and
relates the score to a hidden reference value).

    /opt/venv/bin/python3 ~/practice/grade.py                    # ~/submission on seed 1
    /opt/venv/bin/python3 ~/practice/grade.py --seed 7 --program ~/submission --sim-cap 300
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GIVEN = os.environ.get("ARB_GIVEN_DIR", os.path.join(os.path.dirname(HERE), "given"))
LIB = os.path.join(GIVEN, "lib")
sys.path.insert(0, LIB)

from shoecell.practice import serve_program  # noqa: E402
from shoecell.server import ShoeCellServer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--program", default=os.path.expanduser("~/submission"))
    ap.add_argument("--sim-cap", type=float, default=None, help="simulated seconds per episode (default: the cell's)")
    ap.add_argument("--program-time-cap", type=float, default=1200.0,
                    help="program seconds per episode: the wall clock minus the cell's simulation and rendering")
    ap.add_argument("--silence", type=float, default=120.0)
    ap.add_argument("--json", default=None, help="write the episode summary here")
    args = ap.parse_args()
    cfg = {}
    if args.sim_cap:
        cfg["sim_cap_s"] = float(args.sim_cap)
    server = ShoeCellServer(GIVEN, config=cfg, mode="practice", privileged=False, allow_reset=False, render=True,
                            log=lambda s: print(s, file=sys.stderr, flush=True))
    server.reset(args.seed)
    reason = serve_program(server, os.path.abspath(args.program), LIB, args.program_time_cap, args.silence,
                           log=lambda s: print(s, file=sys.stderr, flush=True))
    summary = server.finalize()
    print("end reason: %s   sim %.1f s   wall %.1f s (program %.1f s, cell %.1f s)   frames %d   illegal calls %d" % (
        reason, summary["sim_time_s"], server.wall_used, summary["program_time_s"], summary["cell_work_s"],
        summary["camera_frames_total"], summary["illegal_call_count"]))
    for i, p in enumerate(summary["pairs"]):
        print("pair %d: credit %.3f  milestones %s" % (i + 1, p["credit"], json.dumps({k: round(v, 3) for k, v in p["milestones"].items()})))
    print("safety_ok: %s  violations: %s  pairs voided by a violation: %s" % (
        summary["safety_ok"], [v["what"] for v in summary["violations"]][:5], [p["index"] + 1 for p in summary["pairs"] if p.get("safety_void")]))
    print("work_score: %.4f" % summary["work_score"])
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1, default=float)


if __name__ == "__main__":
    main()
