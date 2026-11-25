"""QC of the degenerate submission D9 on the practice ground's own path: the robot pressed against a rubble box, read by
the practice debrief (a hidden episode has none).

    d9_practice_debrief.py OUT.json PLAN.json [--sites p1,p2,p3]

For each practice site (environment/practice_ground/sites/pN.tar.xz, pinned in sites.json): one in-process cell in
mode practice (the practice ground's configuration from sites.json: limits, time limit, reports per point from
acceptance.yaml), the D9 plan of the maintainer variant of the privileged reference (variants/qc_reference.py next
to this file, with the plan file of tests/review/degenerate/D9/) as the cell's privileged call,
the practice ground's debrief recorder attached from the reset, and the practice debrief built at the end exactly as
the practice service builds it (voidscore.debrief.build, kind practice: the schema check runs inside). Checks per
site: the debrief builds and passes its schema (each debrief is written next to OUT.json as debrief_<site>.json, for
the leak scan); its reports give credit 0; its stuck periods hold the time the robot
was pressed against the box (the stuck rule of acceptance.yaml on the recorder's exact poses); the cell has no
internal error; the score is 0. Not evidence for the hidden set (the practice sites are not scored); evidence that the
cell's stuck reading and the debrief stay sound with the robot against a hidden box.
"""
import argparse
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
TASK = os.path.join(os.environ["ARB_SRC_ROOT"], os.path.basename(DEV)[:-len("-dev")]) if os.environ.get("ARB_SRC_ROOT") else DEV[:-len("-dev")]
TESTS = os.path.join(TASK, "tests")
PG = os.path.join(TASK, "environment", "practice_ground")
VARIANTS = os.path.join(HERE, "variants")
sys.path.insert(0, TESTS)

from voidcell.server import VoidCellServer  # noqa: E402
from voidscore import debrief, scoring  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("plan")
    ap.add_argument("--sites", default="p1,p2,p3")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    sites = json.load(open(os.path.join(PG, "sites.json")))
    acc = scoring.load_acceptance()
    work = os.path.join(os.path.dirname(os.path.abspath(a.out)), "ref_d9_practice")
    if os.path.exists(work):
        shutil.rmtree(work)
    shutil.copytree(os.path.join(TESTS, "reference"), work, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(os.path.join(VARIANTS, "qc_reference.py"), os.path.join(work, "reference.py"))
    with open(os.path.join(work, "qc_variant.json"), "w") as fh:
        json.dump(plan, fh, indent=1)
    rows = []
    for name in a.sites.split(","):
        k = [i for i, s in enumerate(sites["sites"]) if s["name"] == name][0]
        site = sites["sites"][k]
        cfg = dict(sites.get("config") or {})
        limits = dict(cfg.get("limits") or {})
        limits["reports_per_point"] = int(acc["reports_per_point"])
        cfg["limits"] = limits
        tl = site.get("time_limit_s") or (cfg.get("limits") or {}).get("time_limit_s")
        srv = VoidCellServer(config=cfg, mode="practice", privileged=True, render=False, reference_dir=work)
        srv.reset(os.path.join(PG, site["package"]), seed=int(site.get("noise_seed") or 7),
                  kind=site.get("kind", "practice"), sim_time_limit=tl, name=site["name"])
        scoring.check_site(srv.site, acc)
        recorder = debrief.Recorder(srv)
        r = srv.handle("reference_run", {})
        err = None
        try:
            db = debrief.build(srv, recorder, "practice", acc, episode=1, ended_by="finish")
        except Exception as e:  # noqa: BLE001
            db, err = None, repr(e)[:400]
        rec = srv.finalize()
        sc = scoring.score_record(rec, srv.site, acc)
        mv, tn = float(acc["stuck"]["move_m"]), float(acc["stuck"]["turn_deg"])
        prim = []
        for h, n0, n1, _b in recorder.rows:
            if h.get("kind") != "primitive":
                continue
            dx, dy, dth = debrief._relative(n0, n1)
            prim.append(dict(t_start=round(float(h["t_start"]), 3), name=h.get("name"), moved_m=round((dx * dx + dy * dy) ** 0.5, 4),
                             turned_deg=round(dth, 2), below_both=bool((dx * dx + dy * dy) ** 0.5 < mv and abs(dth) < tn)))
        ref = rec.get("reference") or {}
        qlog = (ref.get("qc_variant") or {}).get("log") or []
        pressed = [e for e in qlog if e["kind"] == "pressed"]
        tp = pressed[0]["t"] if pressed else None
        stuck = (db or {}).get("stuck_periods") or []
        row = {"site": name, "reply_ok": r.get("ok"), "debrief_error": err, "debrief_keys": sorted(db) if db else None,
               "reports": (db or {}).get("reports"), "stuck_periods": stuck,
               "pressed_at_t": tp, "push": pressed[0]["push"] if pressed else None,
               "stuck_period_ends_at_the_press": any(abs(float(p["t_end"]) - tp) < 1.0 for p in stuck) if tp else False,
               "S": sc["S"], "c": sc["c"], "last_primitives_recorder": prim[-6:],
               "internal_errors": rec.get("internal_errors"),
               "internal_errors_live": rec.get("internal_errors_live"), "done_reason": rec.get("done_reason"),
               "monitor_box_samples": (ref.get("monitor") or {}).get("contact_box_samples")}
        row["ok"] = err is None and all(float(x["credit"] or 0.0) == 0.0 for x in row["reports"] if x["accepted"]) and \
            len([x for x in row["reports"] if x["accepted"]]) == 3 and row["S"] == 0.0 and \
            int(row["internal_errors"] or 0) == 0 and int(row["internal_errors_live"] or 0) == 0
        rows.append(row)
        if db is not None:
            with open(os.path.join(os.path.dirname(os.path.abspath(a.out)), "debrief_%s.json" % name), "w") as fh:
                json.dump(db, fh, indent=1)
        print(json.dumps({k: row[k] for k in ("site", "ok", "debrief_error", "S", "stuck_periods", "pressed_at_t",
                                              "stuck_period_ends_at_the_press", "internal_errors")}), flush=True)
        srv.close()
    out = {"package": "QC D9 on the practice path", "plan": plan.get("name"), "rows": rows,
           "pass": all(r["ok"] for r in rows)}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    return 0 if out["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
