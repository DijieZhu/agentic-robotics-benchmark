"""The shop's bench plan of the race-car cell (hidden; `tests/hidden/bench_plan.json`): where the five units that lie
loose on the bench (the four wheels and the stick pack) start, in which resting orientation, before the episode's
jitter (`layout.episode_layout`). The deck, the compute module and the LiDAR start on their fixtures
(`fixtures.nominal_unit_poses`), whose places are the grader config's.

The spots (cell frame, m) and why:
* the four wheels in a 2 x 2 block beyond the car's rear on the +y side (x 0.95 / 1.15, y 0.25 / 0.47), inner face
  on the bench (outer face up, the hex pocket down; the cell manual's "inner face down"), so that a pick there swings
  the arm beside the car, never over it;
* the stick pack beyond the car's rear on the -y side (0.97, -0.25), on its underside, laid as it is mounted (its
  lead end towards the car's rear: at the nominal placement the cell's +x); the episode turns it by the stratum's lead
  end (0 / 180 deg).
Every spot keeps, under the worst +-30 mm / +-30 deg draw, its footprint centre inside the reach ring RING_M (the
arm's pick map: 8 hand yaws feasible on 855 / 844 of 951 cells, none failing inside 1.30 m for a wheel), 60 mm from
every other unit, the fixtures and the bench board, and out of the wheel lanes LANES (beside the car at the nominal
placement, where nothing taller than LANES["height_m"] may stand; the loose units are kept out of them altogether).

`rest_plan(kit)` -> {"format", "rest": {unit: {pos, quat, yaw_deg, face, ...}}, "ring_m", "lanes", "oracle", ...};
the rest heights come from the kit's collision (the lowest collision point SETTLE_LIFT_M above the bench top). It is
geometric: the grasp hints of the earlier cell's plan are not part of it (the reference program chooses its grasps).

    python -I -B bench_plan.py --kit <kit dir> --out <tests/hidden/bench_plan.json>
"""
import json
import math
import os
import sys

import numpy as np

RING_M = (0.25, 1.30)
LANES = {"x_min_m": 0.40, "x_max_m": 0.85, "y_min_m": 0.09, "y_max_m": 0.35, "height_m": 0.10}
SETTLE_LIFT_M = 0.0003
# unit -> (x, y of the footprint's centre, m; yaw of the unit's x from the cell's x, deg); the units' frames up
SPOTS = {"WHL1": (0.95, 0.25, 0.0), "WHL2": (0.95, 0.47, 0.0), "WHL3": (1.15, 0.25, 0.0), "WHL4": (1.15, 0.47, 0.0),
         "BAT": (0.97, -0.25, 180.0)}
FACES = {"WHL": "inner face on the bench (the wheel's z, its outer side, up)",
         "BAT": "on its underside (the pack's z up), laid as mounted: the lead end towards the cell's +x"}
FORMAT = "racecar-bench-plan/1"


def _rotz(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rest_plan(kit):
    """the plan for `kit` (a loaded kit dict with "dir"; module docstring)"""
    from .layout import _quat_wxyz, unit_hulls
    hulls = unit_hulls(kit, units=list(SPOTS))
    rest = {}
    for u, (x, y, yaw) in SPOTS.items():
        R = _rotz(yaw)
        P = hulls[u] @ R.T
        c = 0.5 * (P[:, :2].min(0) + P[:, :2].max(0))
        pos = [x - float(c[0]), y - float(c[1]), -float(P[:, 2].min()) + SETTLE_LIFT_M]
        utype = kit["units"][u]["type"]
        rest[u] = {"pos": [round(v, 7) for v in pos], "quat": [round(float(v), 9) for v in _quat_wxyz(R)], "yaw_deg": yaw,
                   "spot_xy": [x, y], "face": FACES[utype], "type": utype}
        if u == "BAT":
            rest[u]["lead_end_deg"] = 0
    return {"format": FORMAT, "rest": rest, "ring_m": list(RING_M), "lanes": dict(LANES), "settle_lift_m": SETTLE_LIFT_M,
            "fixture_units": "DCK on the shelf, CMP and LDR in the tray: fixtures.nominal_unit_poses() at the grader config's "
                             "fixture places",
            "oracle": {"hub_deg": {h: 0.0 for h in ("hub_FL", "hub_FR", "hub_RL", "hub_RR")},
                       "note": "the privileged run's hub angles when it runs on the nominal layout (oracle_nominal_layout); the fixed "
                               "hidden set draws its own"},
            "source": "arm/assembly_car/bench_plan.py rest_plan() on the kit's collision (variant %s)" % kit.get("variant")}


def write(plan, path):
    with open(path, "w") as f:
        json.dump(plan, f, indent=1, sort_keys=True)
        f.write("\n")


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--kit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.dirname(os.path.dirname(here)))
    from arm.assembly_car.cell import load_kit
    from arm.assembly_car.bench_plan import rest_plan as rp, write as wr
    wr(rp(load_kit(a.kit)), a.out)


if __name__ == "__main__":
    main()
