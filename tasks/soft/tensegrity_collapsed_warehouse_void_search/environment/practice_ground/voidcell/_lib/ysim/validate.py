"""Validation against the authors' published per-primitive displacement table, in full SE(2) (ysim 0.3.0; re-run on
0.3.1; the results are not shipped with this copy).

Protocol: for each of the 11 primitives P and each of the six rollable start faces F, N executions; execution j
starts from a fresh simulator: stand() (rest state), the j-th preparation sequence of PREP[F] (published primitives
that land the robot on F; found by a face map of 400 two-action sequences), then P, measured from the rest
state before P to the rest state after it (P's own states, the re-plan rest phase, the hold params.settle_after), with the authors'
SE(2) formula (ysim.metrics). The preparation sequences make the executions differ in their start shape.

Compared, per (P, F) cell on the mean over its executions, against the table (authors_tables.json):
  distance |t|, heading change dtheta, and the direction of travel: the translation error vector t_sim - t_table in
  the start frame, split into along-track (y, the rolling direction) and lateral (x) components.
SE(2) criterion (stated before the runs): |t_sim - t_table| <= 25 % of |t_table| and |dtheta_sim - dtheta_table|
  <= 10 deg. The criterion of ysim 0.1.0 / 0.2.0 (distance within 25 % and heading within 10 deg, no direction)
  is reported alongside for comparison only.
The table is the authors' simulation output, not a measurement; they report their primitive model to differ from
the real robot by 0.20 +- 0.05 m and 4.7 +- 2.7 deg per primitive (RA-L Table II).

usage: from ysim import validate; res = validate.run(n_exec=4)   (package parent on sys.path; python -I -B)
"""
import json
import math
import sys
import time

import numpy as np

from . import gaits
from . import model as mdl
from . import paths
from .sim import Simulator

# face -> preparation sequences [(name, reverse), ...] from stand(): every one lands the robot on that face in the
# ysim 0.3.0 face map (400 two-action sequences at dt 0.25 ms, default mode); re-checked on
# ysim 0.3.1: all 24 land on their face. Forward
# published primitives where they reach the face; the faces 034 and 124 (and 025 after a first primitive) are
# reached only through the authors' reverse rolls. A run whose preparation misses the face is recorded as a miss.
_F, _R = False, True
PREP = {
    (0, 2, 5): [[], [("roll_100_100", _F), ("roll_120_120", _R)], [("roll_100_120", _F), ("roll_120_100", _R)],
                [("roll_140_140", _F), ("roll_140_100", _R)]],
    (0, 3, 5): [[("ccw", _F)], [("cw", _F)], [("ccw", _F), ("cw", _F)], [("cw", _F), ("ccw", _F)]],
    (1, 2, 4): [[("roll_100_140", _R), ("roll_100_140", _R)], [("roll_120_120", _R), ("roll_120_140", _R)],
                [("roll_140_100", _R), ("roll_140_120", _R)], [("roll_140_140", _R), ("roll_120_100", _R)]],
    (1, 2, 5): [[("roll_100_100", _F)], [("roll_120_120", _F)], [("roll_140_140", _F)], [("roll_120_140", _F)]],
    (0, 3, 4): [[("roll_120_120", _R)], [("roll_140_100", _R)], [("ccw", _F), ("roll_120_120", _R)],
                [("cw", _F), ("roll_140_140", _R)]],
    (1, 3, 4): [[("roll_120_120", _F), ("roll_140_100", _F)], [("roll_140_140", _F), ("roll_120_140", _F)],
                [("roll_120_140", _F), ("roll_100_120", _F)], [("roll_100_100", _F), ("roll_100_100", _F)]],
}


def table():
    with open(paths.data_path("authors_tables.json")) as fh:
        return json.load(fh)["primitives"]


def jobs(names=None, faces=None, n_exec=4, mode=None):
    """One job per execution: (primitive, face, j, preparation)."""
    out = []
    for name in (names or gaits.NAMES):
        for face in (faces or gaits.ROLLABLE_FACES):
            for j, prep in enumerate(PREP.get(tuple(face), [])[:n_exec]):
                out.append({"id": "%s|%s|%d" % (name, "".join(map(str, face)), j), "primitive": name,
                            "face": list(face), "j": j, "prep": [list(x) for x in prep], "mode": mode})
    return out


def run_job(job, cfg_kw=None, sim_kw=None):
    S = Simulator(mdl.ModelConfig(visual=False, **(cfg_kw or {})), **(sim_kw or {}))
    S.stand(mode=job["mode"])
    for name, rev in job["prep"]:
        S.run_primitive(name, mode=job["mode"], reverse=rev)
    face = list(S.bottom()[0])
    out = {"id": job["id"], "primitive": job["primitive"], "face": job["face"], "j": job["j"], "face_before": face}
    if face != list(job["face"]):
        out["face_miss"] = True
        return out
    r = S.run_primitive(job["primitive"], mode=job["mode"])
    out.update({k: (list(r[k]) if isinstance(r[k], tuple) else r[k]) for k in
                ("dx", "dy", "dist", "dtheta_deg", "t_total", "t_exec", "t_rest_phase", "timeouts", "rest_reached",
                 "bottom_start", "bottom_end", "top_z_end", "com_z_end", "peak_cable_tension_N", "energy_J")})
    out["se2_before_rest_phase"] = r["se2_before_rest_phase"] and {
        k: (list(v) if isinstance(v, tuple) else v) for k, v in r["se2_before_rest_phase"].items()}
    return out


def cell(rows, ref):
    """Statistics of one (primitive, face) cell and its errors against the table entry ref."""
    dx = np.array([r["dx"] for r in rows])
    dy = np.array([r["dy"] for r in rows])
    th = np.array([r["dtheta_deg"] for r in rows])
    d = np.hypot(dx, dy)
    mx, my, mth = float(dx.mean()), float(dy.mean()), float(th.mean())
    tx, ty, tth, td = ref["dx_m"], ref["dy_m"], ref["dtheta_deg"], ref["dist_m"]
    e_vec = math.hypot(mx - tx, my - ty)
    dir_sim, dir_tab = math.degrees(math.atan2(mx, my)), math.degrees(math.atan2(tx, ty))
    out = {"n": len(rows), "dist_mean": float(d.mean()), "dist_sd": float(d.std()), "dx_mean": mx, "dx_sd": float(dx.std()),
           "dy_mean": my, "dy_sd": float(dy.std()), "dtheta_mean": mth, "dtheta_sd": float(th.std()),
           "travel_dir_deg": dir_sim, "t_total_mean": float(np.mean([r["t_total"] for r in rows])),
           "end_faces": sorted({"".join(map(str, r["bottom_end"])) for r in rows}),
           "rest_reached": "%d/%d" % (sum(bool(r["rest_reached"]) for r in rows), len(rows)),
           "top_z_end_mean": float(np.mean([r["top_z_end"] for r in rows])),
           "err_dist_pct": 100 * (float(d.mean()) - td) / td, "err_dtheta_deg": mth - tth,
           "err_lateral_m": mx - tx, "err_along_m": my - ty, "err_vec_m": e_vec, "err_vec_pct": 100 * e_vec / td,
           "err_travel_dir_deg": (dir_sim - dir_tab + 180.0) % 360.0 - 180.0}
    out["pass_se2"] = bool(out["err_vec_pct"] <= 25.0 and abs(out["err_dtheta_deg"]) <= 10.0)
    kind = gaits.get(rows[0]["primitive"]).kind
    out["pass_dist_heading_only"] = bool(abs(out["err_dtheta_deg"]) <= 10.0 and
                                         (kind != "roll" or abs(out["err_dist_pct"]) <= 25.0))
    return out


def summarize(results, mode=None):
    T = table()
    by = {}
    for r in results:
        if r.get("face_miss"):
            continue
        by.setdefault(r["primitive"], {}).setdefault("".join(map(str, r["face"])), []).append(r)
    out = {"mode": mode, "criterion": "SE(2): |t_sim - t_table| <= 25 % |t_table| and |dtheta error| <= 10 deg, on "
                                      "the cell mean", "cells": {}, "per_primitive": {}, "per_face": {}}
    for name in gaits.NAMES:
        if name not in by:
            continue
        out["cells"][name] = {f: cell(rows, T[name]) for f, rows in sorted(by[name].items())}
        cs = out["cells"][name]
        best = min(cs, key=lambda f: cs[f]["err_vec_pct"] + 2.5 * abs(cs[f]["err_dtheta_deg"]))
        out["per_primitive"][name] = {"table": {k: T[name][k] for k in ("dx_m", "dy_m", "dist_m", "dtheta_deg")},
                                      "faces_pass_se2": sorted(f for f in cs if cs[f]["pass_se2"]),
                                      "faces_pass_dist_heading_only": sorted(f for f in cs if cs[f]["pass_dist_heading_only"]),
                                      "best_face": best, "best": {k: cs[best][k] for k in
                                                                  ("dist_mean", "dtheta_mean", "err_vec_pct", "err_lateral_m",
                                                                   "err_dtheta_deg", "err_travel_dir_deg")}}
    faces = sorted({f for name in out["cells"] for f in out["cells"][name]})
    for f in faces:
        cs = {n: out["cells"][n][f] for n in out["cells"] if f in out["cells"][n]}
        out["per_face"][f] = {"n_primitives": len(cs), "pass_se2": sum(c["pass_se2"] for c in cs.values()),
                              "pass_dist_heading_only": sum(c["pass_dist_heading_only"] for c in cs.values()),
                              "median_err_vec_pct": float(np.median([c["err_vec_pct"] for c in cs.values()])),
                              "median_abs_err_dtheta_deg": float(np.median([abs(c["err_dtheta_deg"]) for c in cs.values()]))}
    allc = [c for n in out["cells"] for c in out["cells"][n].values()]
    out["n_cells"] = len(allc)
    out["n_pass_se2"] = sum(c["pass_se2"] for c in allc)
    out["n_pass_dist_heading_only"] = sum(c["pass_dist_heading_only"] for c in allc)
    out["n_primitives_pass_se2_any_face"] = sum(bool(v["faces_pass_se2"]) for v in out["per_primitive"].values())
    return out


def run(names=None, faces=None, n_exec=4, mode=None, cfg_kw=None, sim_kw=None):
    t0 = time.time()
    res = [run_job(j, cfg_kw, sim_kw) for j in jobs(names, faces, n_exec, mode)]
    out = summarize(res, mode)
    out["executions"] = res
    out["wall_s"] = time.time() - t0
    return out


def summary_lines(s):
    lines = []
    for name, cs in s["cells"].items():
        for f, c in cs.items():
            lines.append("%-13s %s n%d | %.3f+-%.3f m %+6.1f+-%.1f deg dir %+6.1f | err %+5.0f %% %+6.1f deg lat %+.3f m vec %3.0f %% | %s" % (
                name, f, c["n"], c["dist_mean"], c["dist_sd"], c["dtheta_mean"], c["dtheta_sd"], c["travel_dir_deg"],
                c["err_dist_pct"], c["err_dtheta_deg"], c["err_lateral_m"], c["err_vec_pct"],
                "SE2-PASS" if c["pass_se2"] else "fail"))
    return lines


if __name__ == "__main__":
    res = run(n_exec=int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print("\n".join(summary_lines(res)))
    print("SE(2) pass %d/%d cells" % (res["n_pass_se2"], res["n_cells"]))
