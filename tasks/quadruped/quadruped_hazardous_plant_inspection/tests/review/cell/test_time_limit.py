"""The simulated time limit of a mission (version 2.1, 2026-11-17: 1350 s for a scored mission and a
practice building; version 2.0: 2400 s), 900 s for the training hall; the mission ends exactly there with the event `time_limit`.

    python -I -B tests/review/cell/test_time_limit.py --site SITE [--hall HALL] [--json OUT]

GL-free (no frame is taken). The cell is built with the grader's own run settings (tests/grader_config.json), the
practice ground's (environment/practice_ground/sites.json) for the hall. Walking the whole limit would cost minutes of
wall clock, so the mission's clock is moved forward instead: the world counts mission time as `data.time - t0`, and
the test lowers `t0` so that the mission stands at limit - 1 s. Then, through the client proxy (the wire's JSON):
  * describe()["time"] and mission()["budgets"] quote the limit;
  * 99 ticks later (limit - 0.01 s) the mission is not over and no `time_limit` event came;
  * the next tick ends it: `done` true, one `time_limit` event at the limit, get_state() done_reason `time_limit`;
  * a step after the end returns done without advancing; finalize() records done_reason `time_limit` and the
    simulated time equal to the limit.
"""

import argparse
import json
import math
import os
import sys

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TASK = os.path.dirname(TESTS)
sys.path.insert(0, TESTS)

from plantcell.client import connect_handler  # noqa: E402
from plantcell.server import PlantCellServer  # noqa: E402

SCORED_LIMIT_S = 1350.0                # version 2.1 (version 2.0: 2400.0)
HALL_LIMIT_S = 900.0


def one(site, config, mode, kind, sim_time_limit, want):
    problems = []
    srv = PlantCellServer(config=config, mode=mode)
    srv.reset(site, seed=7, kind=kind, sim_time_limit=sim_time_limit)
    r = connect_handler(srv.handle)
    d, m = r.describe(), r.mission()
    if float(d["time"]["sim_time_limit_s"]) != want:
        problems.append("describe() quotes %s s" % d["time"]["sim_time_limit_s"])
    if float(m["budgets"]["sim_time_limit_s"]) != want:
        problems.append("mission() quotes %s s" % m["budgets"]["sim_time_limit_s"])
    r.step(20)
    w = srv.world
    w.t0 = float(w.data.time) - (want - 1.0)          # the mission's clock now stands at limit - 1 s
    out = r.step(99)
    if out["done"] or any(e["type"] == "time_limit" for e in out["events"]):
        problems.append("the mission ended before its limit (t %.3f s, events %s)" % (out["sim_time"], out["events"]))
    if not (want - 0.02 < float(out["sim_time"]) < want):
        problems.append("sim_time %.4f s after 99 ticks (expected just below %.0f)" % (out["sim_time"], want))
    end = r.step(50)
    tl = [e for e in end["events"] if e["type"] == "time_limit"]
    if not end["done"]:
        problems.append("the mission did not end at its limit (t %.3f s)" % end["sim_time"])
    if len(tl) != 1 or abs(float(tl[0]["t"]) - want) > 0.0105:
        problems.append("time_limit events %s (expected one at %.0f s)" % (tl, want))
    if abs(float(end["sim_time"]) - want) > 0.0105:
        problems.append("the mission ended at %.4f s" % end["sim_time"])
    st = r.get_state()
    if st["done_reason"] != "time_limit" or not st["done"]:
        problems.append("get_state(): done %s, done_reason %s" % (st["done"], st["done_reason"]))
    after = r.step(10)
    if not after["done"] or after["sim_time"] != end["sim_time"]:
        problems.append("a step after the end advanced (%s -> %s)" % (end["sim_time"], after["sim_time"]))
    summ = srv.finalize()
    if summ["done_reason"] != "time_limit" or abs(summ["sim_time_s"] - want) > 0.0105 or \
            summ["sim_time_limit_s"] != want:
        problems.append("finalize(): %s" % {k: summ[k] for k in ("done_reason", "sim_time_s", "sim_time_limit_s")})
    srv.close()
    return problems, {"limit_s": want, "ended_at_s": end["sim_time"], "event_t": tl[0]["t"] if tl else None,
                      "done_reason": summ["done_reason"], "quoted": [d["time"]["sim_time_limit_s"],
                                                                     m["budgets"]["sim_time_limit_s"]]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="a practice building or hidden-set package")
    ap.add_argument("--hall", default=None, help="the training hall's package (900 s)")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    problems, res = [], {}
    with open(os.path.join(TESTS, "grader_config.json")) as f:
        gcfg = json.load(f)
    if float(gcfg["time"]["sim_time_limit_s"]) != SCORED_LIMIT_S:
        problems.append("grader_config.json: sim_time_limit_s %s" % gcfg["time"]["sim_time_limit_s"])
    p, res["scored"] = one(a.site, gcfg, "hidden", "hidden", None, SCORED_LIMIT_S)
    problems += ["scored: " + x for x in p]
    with open(os.path.join(TASK, "environment", "practice_ground", "sites.json")) as f:
        sites = json.load(f)
    if float(sites["config"]["time"]["sim_time_limit_s"]) != SCORED_LIMIT_S:
        problems.append("practice sites.json: buildings at %s s" % sites["config"]["time"]["sim_time_limit_s"])
    hall = next((s for s in sites["sites"] if s["name"] == "hall"), None)
    if hall is None or float(hall.get("sim_time_limit_s", math.nan)) != HALL_LIMIT_S:
        problems.append("practice sites.json: hall at %s s" % (hall or {}).get("sim_time_limit_s"))
    # a practice building as the practice daemon opens it (its sites.json config, no per-site limit)
    p, res["practice_building"] = one(a.site, sites["config"], "practice", "practice", None, SCORED_LIMIT_S)
    problems += ["practice building: " + x for x in p]
    if a.hall:
        p, res["hall"] = one(a.hall, sites["config"], "practice", "hall", hall["sim_time_limit_s"], HALL_LIMIT_S)
        problems += ["hall: " + x for x in p]
    print(json.dumps(res, indent=1))
    for x in problems:
        print("PROBLEM", x)
    print("time limit: %s (%d problems)" % ("OK" if not problems else "FAIL", len(problems)))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"results": res, "problems": problems}, f, indent=1)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
