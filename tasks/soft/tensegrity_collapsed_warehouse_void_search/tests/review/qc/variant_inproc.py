"""A development check of one maintainer variant in ONE in-process cell (no verify, no program process).

    variant_inproc.py WORKDIR {VARIANT | --plan-file PLAN.json} [--site h01]

Builds WORKDIR/ref_<name>/ = a copy of tests/reference/ with reference.py replaced by variants/qc_reference.py (next to
this file) and the plan (VARIANT of variants/plans.json, or the plan file: the degenerate submissions that need the
true pose keep theirs in their folder of tests/review/degenerate/) written as qc_variant.json (what qc_verify.py does in a
stage), then runs it as the privileged call of a hidden-mode cell on one frozen hidden site (the camera off: the
reference never takes a frame) and prints the score and the variant's log. Only to find mistakes in a variant before
the real runs through verify.py; its numbers are not evidence.
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
VARIANTS = os.path.join(HERE, "variants")
sys.path.insert(0, TESTS)

from voidcell.server import VoidCellServer  # noqa: E402
from voidscore import scoring  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("work")
    ap.add_argument("variant", nargs="?", default=None)
    ap.add_argument("--plan-file", default=None)
    ap.add_argument("--site", default="h01")
    a = ap.parse_args()
    if a.plan_file:
        plan = json.load(open(a.plan_file))
    else:
        plan = json.load(open(os.path.join(VARIANTS, "plans.json")))["variants"][a.variant]
    name = plan.get("name") or a.variant
    ref = os.path.join(a.work, "ref_%s_%s" % (name, a.site))
    if os.path.exists(ref):
        shutil.rmtree(ref)
    shutil.copytree(os.path.join(TESTS, "reference"), ref, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copyfile(os.path.join(VARIANTS, "qc_reference.py"), os.path.join(ref, "reference.py"))
    with open(os.path.join(ref, "qc_variant.json"), "w") as fh:
        json.dump(plan, fh, indent=1)
    cfg = json.load(open(os.path.join(TESTS, "grader_config.json")))
    lim = json.load(open(os.path.join(TESTS, "hidden", "limits.json")))
    c = {k: cfg[k] for k in ("limits", "pose_source", "render_backend", "msaa_rgb") if k in cfg}
    c["limits"] = dict(c.get("limits") or {}, time_limit_s=float(lim["time_limit_s"]))
    row = [r for r in json.load(open(os.path.join(TESTS, "hidden", "seeds.json")))["sites"] if r["tag"] == a.site][0]
    srv = VoidCellServer(config=c, mode="hidden", privileged=True, render=False, reference_dir=ref)
    srv.reset(os.path.join(TESTS, "hidden", row["archive"]), seed=int(row["seed"]), kind="hidden")
    srv.scorer = scoring.scorer_for(srv.site, scoring.load_acceptance())
    srv.handle("mission", {})
    srv.handle("telemetry", {})
    r = srv.handle("reference_run", {})
    rec = srv.finalize()
    sc = rec["score"]
    ref_s = rec.get("reference") or {}
    out = {"variant": name, "site": a.site, "reply": {k: r.get(k) for k in ("ok", "t", "wall_s", "error", "detail")},
           "S": sc["S"], "c": sc["c"], "F": sc["F"], "expected": plan.get("expected"),
           "points": [{k: p.get(k) for k in ("i", "accepted", "counted", "d_m", "dwell_failed", "c")}
                      for p in sc["points"]], "retrieval": sc["retrieval"],
           "internal_errors": rec["internal_errors"], "done_reason": rec.get("done_reason"),
           "sim_time_s": rec.get("sim_time_s"), "log": (ref_s.get("qc_variant") or {}).get("log"),
           "rows": (ref_s.get("qc_variant") or {}).get("rows"), "aborted": ref_s.get("aborted"),
           "monitor": ref_s.get("monitor"), "final_state_sha256": rec.get("final_state_sha256"),
           "pinned": row.get("reference_final_state_sha256")}
    with open(os.path.join(a.work, "inproc_%s_%s.json" % (name, a.site)), "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps({k: out[k] for k in ("variant", "site", "S", "c", "F", "expected", "internal_errors", "aborted",
                                          "done_reason", "sim_time_s")}, default=str))
    for p in out["points"]:
        print("  ", json.dumps(p))
    for e in out["log"] or []:
        print("  log", json.dumps(e, default=str)[:700])
    return 0


if __name__ == "__main__":
    sys.exit(main())
