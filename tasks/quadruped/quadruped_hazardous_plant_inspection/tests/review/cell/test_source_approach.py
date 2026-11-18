"""The source's approach condition: E_X = 1 only when, before the
`report_source` call, there were 3 consecutive seconds in which the robot's own dosimeter showed a mean of at least
2.0 Gy/h AND its body centre stayed within 3.0 m (horizontally) of the point that report gives.

    MUJOCO_GL=osmesa python -I -B tests/review/cell/test_source_approach.py --site SITE [--json OUT]

On a practice building (its documented waste store gives more than 2 Gy/h on reachable floor far from the unlisted
source), two missions through the client proxy; the robot is moved kinematically (nothing walks there) and stands
while the dosimeter counts:
  1. waste store: 8 s on floor where the documented sources give >= 2.5 Gy/h, at least 4.5 m from the unlisted source,
     then report_source at the source's true position: E_X 0, the debrief names the failed half (`near`), the 3 s
     peak >= 2.0 Gy/h far away, the peak within 3.0 m of the reported point below 2.0;
  2. ring survey: 8 s each at three points 1.2-2.6 m from the source where the field gives >= 2.5 Gy/h, then the same
     report: E_X 1, the source's credit 1 (position error 0);
  3. the old rule (any 3 s mean before the report) would have given E_X 1 in case 1: the debrief's 3 s peak anywhere
     shows it.
"""

import argparse
import json
import math
import os
import sys

TESTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, TESTS)

import numpy as np  # noqa: E402

from plantcell.client import connect_handler  # noqa: E402
from plantcell.server import PlantCellServer  # noqa: E402


def grader_config():
    with open(os.path.join(TESTS, "grader_config.json")) as f:
        return json.load(f)


def floor_points(w):
    """(xy, total rate Gy/h, distance to the unlisted source) of walk-enterable cells (every 3rd)."""
    from mission.world import DoseField
    so = w.site_obj
    t = so.trav
    iy, ix = np.nonzero(t.enter["walk"])
    iy, ix = iy[::3], ix[::3]
    xs = np.array([t.g.xy(int(a), int(b)) for a, b in zip(iy, ix)], float)
    F = DoseField(w.site)
    k = float(w.cfg["dose"]["gyh_per_site_unit"])
    rate = np.array([F.rate(np.array([x, y, 0.24])) * k for x, y in xs])
    src = next(s for s in w.site["sources"] if not s.get("documented", True))
    d = np.hypot(xs[:, 0] - src["pos"][0], xs[:, 1] - src["pos"][1])
    return xs, rate, d, src


def put(w, xy):
    """The robot's body centre at xy (kinematically), standing still."""
    import mujoco
    m, d = w.model, w.data
    jr = m.joint("robot/root")
    a = int(m.jnt_qposadr[jr.id])
    d.qpos[a:a + 2] = xy
    d.qvel[int(m.jnt_dofadr[jr.id]):int(m.jnt_dofadr[jr.id]) + 6] = 0.0
    mujoco.mj_forward(m, d)


def mission(site, stances, dwell_s):
    srv = PlantCellServer(config=grader_config(), mode="practice")
    srv.reset(site, seed=5, kind="practice")
    r = connect_handler(srv.handle)
    w = srv.world
    r.set_mode("stand")
    r.step(100)
    for xy in stances:
        put(w, xy)
        for _ in range(int(round(dwell_s / 0.5))):
            put(w, xy)                       # (held in place: the stand controller does not drift it)
            r.step(50)
    src = next(s for s in w.site["sources"] if not s.get("documented", True))
    r.report_source(float(src["pos"][0]), float(src["pos"][1]))
    pad = (w.prior.get("datum") or {}).get("pos", [0.0, 0.0])
    for _ in range(6):                       # back on the pad (link up there): the queued report goes out
        put(w, (float(pad[0]), float(pad[1])))
        r.step(50)
    r.finish()
    db = srv.debrief()
    row = next(x for x in db["items"] if x["type"] == "hotspot")
    rec = next(x for x in srv.finalize()["items"] if x["type"] == "hotspot")
    srv.close()
    return row, rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    problems = []
    srv = PlantCellServer(config=grader_config(), mode="practice")
    srv.reset(a.site, seed=5, kind="practice")
    xs, rate, d, src = floor_points(srv.world)
    srv.close()
    store = np.nonzero((rate >= 2.5) & (d >= 4.5))[0]
    ring = np.nonzero((rate >= 2.5) & (d >= 1.2) & (d <= 2.6))[0]
    out = {"site": os.path.basename(a.site), "store_cells": int(len(store)), "ring_cells": int(len(ring))}
    if not len(store):
        problems.append("no floor >= 2.5 Gy/h at least 4.5 m from the unlisted source on this site")
    if len(ring) < 3:
        problems.append("fewer than 3 ring cells")
    if not problems:
        i = store[int(np.argmax(rate[store]))]
        row1, rec1 = mission(a.site, [xs[i]], 8.0)
        out["waste_store"] = {"stance": np.round(xs[i], 2).tolist(), "field_gyh": round(float(rate[i]), 2),
                              "from_source_m": round(float(d[i]), 2), "debrief": row1, "E": rec1.get("E")}
        if rec1.get("E") != 0 or row1.get("approach_ok"):
            problems.append("waste-store dwell gave the approach: %s" % row1)
        if not (row1.get("own_dosimeter_peak_3s_gyh") or 0) >= 2.0:
            problems.append("the waste-store dwell's 3 s peak is below 2.0 (the case does not test the rule): %s" % row1)
        if row1.get("approach_failed") is None or "farther than 3.0 m" not in str(row1.get("approach_failed")):
            problems.append("the debrief does not name the failed half: %s" % row1.get("approach_failed"))
        ang = np.arctan2(xs[ring, 1] - src["pos"][1], xs[ring, 0] - src["pos"][0])
        pick = [ring[int(np.argmin(np.abs(np.angle(np.exp(1j * (ang - a0))))))] for a0 in (-2.1, 0.0, 2.1)]
        row2, rec2 = mission(a.site, [xs[j] for j in pick], 8.0)
        out["ring"] = {"stances": [np.round(xs[j], 2).tolist() for j in pick],
                       "field_gyh": [round(float(rate[j]), 2) for j in pick],
                       "from_source_m": [round(float(d[j]), 2) for j in pick], "debrief": row2, "E": rec2.get("E"),
                       "credit": rec2.get("credit")}
        if rec2.get("E") != 1 or not row2.get("approach_ok"):
            problems.append("ring survey without the approach: %s" % row2)
    out["problems"] = problems
    print(json.dumps(out, indent=1, default=str))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=str)
    print("PASS" if not problems else "FAIL (%d problems)" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
