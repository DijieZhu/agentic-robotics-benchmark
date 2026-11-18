"""The link model: line of sight from the lidar head to a working node in range (base station 12 m,
repeaters 9 m; the drawn but broken repeater does not relay), checked against hand-placed cases on the ladder site
110 and against the site package's own 2-D coverage grid (fields.npz `link`, the generator's estimate) on random
interior points; and the queue rule: a report made without the link is queued, sent when the link comes up, lost
when the mission ends while queued.

    MUJOCO_GL=osmesa python -I tests/review/cell/test_link.py SITE [--name site_110]
"""

import argparse
import json
import math
import os
import sys

import numpy as np

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TESTS)

from plantcell.client import connect_handler  # noqa: E402
from plantcell.server import PlantCellServer  # noqa: E402

CASES = {"site_110": [((0.0, 0.0), True, "pad"), ((2.125, 0.275), True, "slot A reading place (C01, airlock side)"),
                      ((4.0, 0.0), False, "C01 at 4 m: the airlock wall hides the antenna, RP1 9.6 m away"),
                      ((10.0, 0.0), True, "C01 near RP1"), ((28.0, -4.0), False, "inside the bund (tank room 161)"),
                      ((13.1, 7.9), False, "C03 at the dead repeater RP2")]}


def run(site, name, n_random=400):
    problems = []
    srv = PlantCellServer()
    srv.reset(site)
    w = srv.world
    z = float(w.data.site_xpos[w.lidar_site][2])
    cases = []
    for (x, y), want, what in CASES.get(name, []):
        up, node, d = w.link_visible(np.array([x, y, z]))
        cases.append({"pos": [x, y], "want": want, "got": bool(up), "node": node, "what": what})
        if bool(up) != want:
            problems.append("case %s: link %s, expected %s" % (what, up, want))
    f = w.pkg.fields
    L = f["link"]
    x0, y0 = [float(v) for v in f["link_origin"]]
    res = float(f["link_res"])
    inside = f["inside"] if "inside" in f else np.ones_like(L, bool)
    rng = np.random.default_rng(0)
    iy, ix = np.nonzero(inside)
    pick = rng.choice(len(iy), size=min(n_random, len(iy)), replace=False)
    agree = 0
    for k in pick:
        x, y = x0 + (ix[k] + 0.5) * res, y0 + (iy[k] + 0.5) * res
        up, _n, _d = w.link_visible(np.array([x, y, z]))
        agree += int(bool(up) == bool(L[iy[k], ix[k]]))
    share = agree / float(len(pick))
    if share < 0.90:
        problems.append("agreement with the package's coverage grid %.3f < 0.90" % share)
    # the queue rule, through the robot's calls: walk out of coverage, report (queued), walk back (sent); then a
    # report out of coverage and the end of the mission (lost)
    r = connect_handler(srv.handle)
    items = [it for it in r.mission()["work_order"]["items"] if it["type"] not in ("dose_point", "hotspot")]
    r.set_mode("walk")
    events = []

    def go(x_goal, back=False):
        """Walk along the corridor axis to x_goal; outbound, stop as soon as the link is down (before the sill)."""
        for _ in range(400):
            st_ = r.get_state()
            o = st_["odometry"]
            if (not back and (o["x"] >= x_goal or not st_["link"]["up"])) or (back and o["x"] <= x_goal):
                break
            want = math.pi if back else 0.0
            e = math.atan2(math.sin(want - o["yaw"]), math.cos(want - o["yaw"]))
            r.walk(0.3 if abs(e) < 0.35 else 0.0, 0.0, 1.2 * e)
            events.extend(r.step(20)["events"])
        r.walk(0.0, 0.0, 0.0)
        events.extend(r.step(80)["events"])

    go(4.6)
    st = r.get_state()
    rep1 = r.report(items[0]["tag"], 1.0, [])
    if st["link"]["up"] or rep1["sent"]:
        problems.append("at x %.2f m the link is %s and the report sent %s (expected down, queued)" % (
            st["odometry"]["x"], st["link"]["up"], rep1["sent"]))
    go(0.6, back=True)
    sent = [e for e in events if e["type"] == "report_sent"]
    if not any(e.get("report_id") == rep1["report_id"] for e in sent):
        problems.append("the queued report was not sent on the way back (events %s)" % [e["type"] for e in events])
    go(4.6)
    rep2 = r.report(items[1]["tag"], 2.0, [])
    r.finish()
    st_end = {rr["id"]: rr["status"] for rr in srv.world.reports}
    if st_end.get(rep1["report_id"]) != "delivered" or st_end.get(rep2["report_id"]) != "lost":
        problems.append("report statuses at the end %s" % st_end)
    srv.close()
    return problems, cases, share, [e["type"] for e in events]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    name = a.name or os.path.basename(a.site.rstrip("/")).split(".")[0]
    problems, cases, share, evs = run(a.site, name)
    print(json.dumps({"cases": cases, "agreement_with_grid": round(share, 3), "events": evs}, indent=1))
    for p in problems:
        print("PROBLEM", p)
    print("link: %s (%d problems)" % ("OK" if not problems else "FAIL", len(problems)))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
