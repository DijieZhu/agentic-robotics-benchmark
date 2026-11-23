"""Review controls: a controller on the graded or on development sites, in process, text records only.

    /opt/venv/bin/python3 -B tests/review/controls.py oracle --hidden
    /opt/venv/bin/python3 -B tests/review/controls.py oracle --n 48 --rows
    /opt/venv/bin/python3 -B tests/review/controls.py oracle:noheight+notrim --hidden
    /opt/venv/bin/python3 -B tests/review/controls.py public --hidden

Run in the task's image as root (the robot models are readable by root only). Controllers:

  oracle                 solution/oracle with the site's truth (what tests/anchor.json records)
  oracle:<mutants>       the oracle with parts of its on-site adaptation taken out, joined by '+':
                           noheight    no stance-height feedback (the sag of a nominal load assumed)
                           notrim      no pitch trim from the gravity reading
                           nominalfit  the fittings' nominal clearances (0.426 m, 0.082 m) instead of the site's
                           deadreckon  the launch pose from the truth, then the Go1's own (drifting) odometry
                           surveyroute the route and the dock as surveyed, not as the Go1 finds them
  public                 tests/review/public_reference: UAV survey, cap registration and a gait with nominal
                         postures, all from the public observations; it assumes the nominal load

Development sites are drawn from numpy default_rng(20261120); a seed of the graded set is skipped. None of these
is a participant program and none is a gate: they show what the per-site properties ask of a controller.
"""
import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(TASK/"tests/runtime"), str(TASK/"tests/setup_payload")]
from mission import World  # noqa: E402


class Local:
    """The service's calls without the socket."""
    def __init__(self, seed):
        self.w = World(seed)
        self.n = 0

    def _answer(self, obs):
        self.n += 1
        obs["requests_remaining"] = 4000-self.n
        return obs

    def observe(self):
        return self._answer(self.w.observe())

    def act(self, action):
        self.last = self._answer(self.w.act(action))
        return self.last

    def look(self, detent):
        return self._answer(self.w.look(detent))

    def advance(self):
        self.last = self._answer(self.w.advance())
        return self.last

    mutants = ""

    def request(self, method, **_):
        assert method == "truth"
        t = self.w.truth()
        if "surveyroute" in self.mutants:
            # The plan of a program that trusts its survey: the opening that was free then, the dock where it was.
            was = t["surveyed"]["pallet"][2:]
            middle = t["racks"][1]
            if middle["wide"][0] <= (was[0]+was[1])/2 <= middle["wide"][1]:   # the pallet stood where the free opening is now
                middle["wide"] = [v for v in self.w.racks[1].other]
            t["goal"] = t["surveyed"]["goal"]
        if "deadreckon" in self.mutants and t["pose"] is not None:
            import math
            x0, y0, yaw0 = t["launch_pose"]
            ox, oy = self.last["odom_xy_m"]
            c, s = math.cos(yaw0), math.sin(yaw0)
            t["pose"] = [x0+c*ox-s*oy, y0+s*ox+c*oy, yaw0+self.last["odom_yaw_rad"]]
        return t


def controller(name):
    if name.startswith("oracle"):
        sys.path.insert(0, str(TASK/"solution/oracle"))
        import oracle
        import walk
        mutants = name.split(":")[1] if ":" in name else ""
        Local.mutants = mutants
        base = walk.Walk

        class Mutant(base):
            def __init__(self, waypoints, fittings):
                if "nominalfit" in mutants:
                    fittings = [{**f, "clearance": .426, "rise": .082} for f in fittings]
                super().__init__(waypoints, fittings)

            def feedback(self, obs, target, level=True, nose_up=0.):
                super().feedback(obs, target, level, nose_up)
                if "noheight" in mutants:
                    self.lift = .017
                if "notrim" in mutants:
                    self.trim = 0.
        oracle.Walk = Mutant
        return lambda robot: oracle.mission(robot, "review")
    sys.path.insert(0, str(TASK/"tests/review/public_reference"))
    import oracle as reference
    from survey import scan
    return lambda robot: reference.deploy(robot, scan(robot))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("controller")
    parser.add_argument("--hidden", action="store_true", help="the eight graded sites")
    parser.add_argument("--n", type=int, default=16, help="development sites")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--rows", action="store_true", help="one JSON line per site")
    args = parser.parse_args()
    graded = json.loads((TASK/"tests/hidden/sites.json").read_text())["seeds"]
    draws = [int(x) for x in np.random.default_rng(20261120).integers(1, 2**62, size=args.start+args.n)]
    seeds = graded if args.hidden else [s for s in draws if s not in set(graded)][args.start:]
    run = controller(args.controller)
    rows = []
    for i, seed in enumerate(seeds):
        robot = Local(seed)
        try:
            run(robot)
            m = robot.w.metrics()
            row = {"site": i, "outcome": m["target_outcome"], "contact": m["contact_wall"], "part": m["contact_part"],
                   "moved": [int(robot.w.found[3] != robot.w.surveyed[3]), int(tuple(robot.w.found[1]) != tuple(robot.w.surveyed[1])),
                             int(robot.w.found[2] != robot.w.surveyed[2])],
                   "ticks": m["target_ticks"], "duct_gap_m": m["min_duct_gap_m"], "deck_feet": m["deck_support_feet"]}
        except Exception as exc:            # a controller that gives up (no route, no registration)
            row = {"site": i, "outcome": "error", "contact": repr(exc)[:80], "part": None, "ticks": 0}
        rows.append(row)
        robot.w.close()
        if args.rows:
            print(json.dumps(row), flush=True)
    docked = [r for r in rows if r["outcome"] == "docked"]
    fails = collections.Counter((r["outcome"], r["contact"], r["part"]) for r in rows if r["outcome"] != "docked")
    print(json.dumps({"controller": args.controller, "docked": "{}/{}".format(len(docked), len(rows)),
                      "max_ticks": max((r["ticks"] for r in docked), default=0),
                      "failures": {str(k): v for k, v in fails.items()}}), flush=True)


if __name__ == "__main__":
    main()
