"""The fixed hidden set's selection, re-run (a maintainer's tool; the grader never runs it).

The rule was pre-registered on 2026-11-07 before any candidate ran (tests/hidden/selection.json "rule"): the frozen
reference program runs each of the 24 candidates ONCE, in drawn order, as the grader drives the oracle submission
(describe, one overview and one wrist frame, then reference_run; OPENBLAS_CORETYPE=Haswell and NPY_DISABLE_CPU_FEATURES
as tests/test.sh sets them); a candidate is kept when the reference completes it (no void, start-up 0.20, shutdown
0.15, log sheet 0.15, all four of its events judged and each credited in full; the in-band fractions are what the
anchor measures and are not a criterion); the hidden set is the first 8 kept candidates in drawn order.

The first selection ran with the maintainers' development harness (the same calls). This file is that harness in the
task folder, used for the amendments that re-run the rule on an amended cell (v0.6, 2026-11-10: the cell's safety
watch, the wrist's wrench estimate, V-104's start state, T-101's overflow):

    # inside the task image (it imports the cell); one process per candidate, `--jobs` at a time:
    docker run --rm --network none -e OPENBLAS_CORETYPE=Haswell -e NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR" \\
        -v <task>/tests:/stage:ro -v <out>:/out --entrypoint /opt/venv/bin/python3 <image> \\
        -I -B /stage/review/select_set.py run --out /out --jobs 12 [--index 0 1 ...]
    # anywhere (standard library only): the rule over the results, the hollowing check, the amendment's record
    python3 -B tests/review/select_set.py make --results <out> [<out2> ...] [--compare <other machine's out>]

`run` writes one JSON per candidate (cand_<i>.json: its score, parts, events, voids, protective stops, ticks, the
layout features of the hollowing check, a sha256 of the final physics state); `make` prints the kept candidates, the
set, the hollowing check and, with --compare, whether another machine's runs are equal (score, parts and ticks).
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
SELECTION = os.path.join(TESTS, "hidden", "selection.json")
SET_SIZE = 8
FROZEN_DIRS = ("reference", "skidcell")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_hashes():
    out = {}
    for d in FROZEN_DIRS:
        for p in sorted(glob.glob(os.path.join(TESTS, d, "*.py"))):
            out["%s/%s" % (d, os.path.basename(p))] = sha256_file(p)
    return out


def features(layout):
    """The hollowing check's features of one candidate (as in the first selection)."""
    f = {}
    for tag, kind in sorted(layout["gauge_kinds"].items()):
        f["%s:%s" % (tag, kind)] = 1
    for tag in ("V-101", "V-102", "V-103", "V-104", "V-112", "V-302"):
        f["%s:%s" % (tag, layout["styles"][tag])] = 1
    step = [e for e in layout["events"] if e["type"] == "demand_step"][0]
    f["demand_step:%s" % ("up" if step["delta_m3h"] > 0 else "down")] = 1
    for x in layout["faults"]:
        f["fault:%s" % x] = 1
    for k in ("meter_first", "mirror", "panel_p101_left"):
        f["%s:%s" % (k, layout[k])] = 1
    f["wrong_n:%d" % len(layout["wrong_at_start"])] = 1
    return f


# ---------------------------------------------------------------------------------------------------------- run
def _given_dir():
    g = "/tmp/given_%d" % os.getpid()
    if not os.path.isdir(g):
        shutil.copytree(os.path.join(TESTS, "setup_payload", "given"), g)
        shutil.copytree("/opt/task-assets/panda", os.path.join(g, "panda"))
    return g


def _one(args):
    i, seed, out = args
    sys.path.insert(0, TESTS)
    from skidcell.server import SkidCellServer
    cfg = json.load(open(os.path.join(TESTS, "grader_config.json")))
    t0 = time.time()
    logs = []
    srv = SkidCellServer(_given_dir(), config=cfg, mode="hidden", privileged=True, log=logs.append,
                         reference_dir=os.path.join(TESTS, "reference"))
    srv.reset(int(seed))
    srv.wall_start = time.time()
    srv.handle("describe", {})
    srv.handle("get_camera", {"name": "overview"})
    srv.handle("get_camera", {"name": "wrist"})
    r = srv.handle("reference_run", {})
    s = srv.finalize()
    d = srv.scene.data
    state = hashlib.sha256(d.qpos.tobytes() + d.qvel.tobytes()).hexdigest()
    rec = {"i": i, "seed": int(seed), "score": round(s["episode_score"], 4), "raw_score": s["raw_score"], "parts": s["parts"],
           "events": [{"type": e["type"], "credit": e["credit"], "excluded": e["excluded"]} for e in s["events"]],
           "voids": s["voids"], "protective_stops": s["protective_stops"], "t101_overflow_t": s["t101_overflow_t"],
           "startup": s["startup"], "shutdown": s["shutdown"], "ticks": s["ticks"], "sim_time_s": s["sim_time_s"],
           "faults": s["layout"]["faults"], "features": features(dict(srv.layout)), "state_sha256": state,
           "internal_errors": s["internal_errors"], "ref_error": r.get("error") if isinstance(r, dict) else None,
           "wall_s": round(time.time() - t0, 1), "frozen": frozen_hashes(), "tool_sha256": sha256_file(os.path.abspath(__file__)),
           "cell_log_tail": [x for x in logs if "[ref]" not in x][-5:]}
    with open(os.path.join(out, "cand_%02d.json" % i), "w") as f:
        json.dump(rec, f, indent=1, default=float)
    return i, rec["score"], rec["wall_s"]


def cmd_run(a):
    import multiprocessing as mp
    sel = json.load(open(SELECTION))
    cands = [int(x) for x in sel["candidates"]]
    idx = a.index if a.index else list(range(len(cands)))
    os.makedirs(a.out, exist_ok=True)
    jobs = [(i, cands[i], a.out) for i in idx]
    with mp.get_context("fork").Pool(a.jobs, maxtasksperchild=1) as pool:
        for i, score, wall in pool.imap_unordered(_one, jobs):
            print("candidate %2d score %.4f wall %.0f s" % (i, score, wall), flush=True)


# ---------------------------------------------------------------------------------------------------------- make
def kept(rec):
    """The pre-registered rule: the reference completes the candidate."""
    why = []
    if rec["voids"]:
        why.append("void: %s" % ", ".join(v["what"] for v in rec["voids"][:2]))
    p = rec["parts"]
    for part, full in (("startup", 0.2), ("shutdown", 0.15), ("log", 0.15)):
        if abs(float(p[part]) - full) > 1e-9:
            why.append("%s %.4f" % (part, float(p[part])))
    judged = [e for e in rec["events"] if not e["excluded"]]
    if len(judged) < 4:
        why.append("%d of 4 events judged" % len(judged))
    for e in judged:
        if float(e["credit"]) < 1.0 - 1e-9:
            why.append("%s credited %.2f" % (e["type"], float(e["credit"])))
    if rec.get("internal_errors") or rec.get("ref_error"):
        why.append("cell error")
    return not why, why


def load(dirs):
    recs = {}
    for d in dirs:
        for p in glob.glob(os.path.join(d, "cand_*.json")):
            r = json.load(open(p))
            recs[r["i"]] = r
    return [recs[i] for i in sorted(recs)]


def cmd_make(a):
    sel = json.load(open(SELECTION))
    recs = load(a.results)
    n = len(sel["candidates"])
    if len(recs) != n:
        sys.exit("%d of %d candidates have results" % (len(recs), n))
    rows, kept_seeds = [], []
    for r in recs:
        assert int(r["seed"]) == int(sel["candidates"][r["i"]]), "candidate %d is not the pre-registered seed" % r["i"]
        ok, why = kept(r)
        rows.append({"i": r["i"], "seed": r["seed"], "score": r["score"], "parts": r["parts"], "faults": r["faults"], "kept": ok,
                     "drop_reasons": why, "protective_stops": len(r["protective_stops"]), "ticks": r["ticks"]})
        if ok:
            kept_seeds.append(int(r["seed"]))
    hidden = kept_seeds[:SET_SIZE]
    keys = sorted(set(k for r in recs for k in r["features"]))
    ks = [r for r in recs if int(r["seed"]) in hidden]
    hollow = {k: {"kept": round(sum(1 for r in ks if k in r["features"]) / max(1, len(ks)), 3),
                  "all": round(sum(1 for r in recs if k in r["features"]) / len(recs), 3)} for k in keys}
    cut = [x for x in rows if x["seed"] in hidden][-1]["i"] if hidden else None
    dropped_before = ["%d (%s)" % (x["i"], "; ".join(x["drop_reasons"])) for x in rows if not x["kept"] and cut is not None and x["i"] < cut]
    out = {"runs": rows, "kept": kept_seeds, "hidden_set": hidden, "kept_count": len(kept_seeds),
           "summary": "%d of %d candidates completed; the hidden set is candidates %s (the first %d completed); dropped before the %dth: %s" % (
               len(kept_seeds), n, ", ".join(str(x["i"]) for x in rows if x["seed"] in hidden), SET_SIZE, SET_SIZE,
               ", ".join(dropped_before) or "none"),
           "kept_mean_score": round(sum(x["score"] for x in rows if x["seed"] in hidden) / max(1, len(hidden)), 4),
           "hollowing_check": hollow, "frozen": recs[0]["frozen"], "tool_sha256": recs[0]["tool_sha256"],
           "same_frozen_files_in_every_run": all(r["frozen"] == recs[0]["frozen"] for r in recs)}
    if a.compare:
        other = {r["i"]: r for r in load(a.compare)}
        diff = [r["i"] for r in recs if r["i"] not in other or (r["score"], r["parts"], r["ticks"]) != (
            other[r["i"]]["score"], other[r["i"]]["parts"], other[r["i"]]["ticks"])]
        same_state = [r["i"] for r in recs if r["i"] in other and r["state_sha256"] == other[r["i"]]["state_sha256"]]
        out["compare"] = {"candidates": len(other), "differ_in_score_parts_or_ticks": diff, "final_state_identical": len(same_state)}
    print(out["summary"])
    print("kept mean score %.4f; set %s" % (out["kept_mean_score"], hidden))
    for x in rows:
        print("  %2d %20d %.4f %s %s%s" % (x["i"], x["seed"], x["score"], "kept" if x["kept"] else "drop", "; ".join(x["drop_reasons"]),
                                          " stops %d" % x["protective_stops"] if x["protective_stops"] else ""))
    if "compare" in out:
        print("compare:", out["compare"])
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--jobs", type=int, default=8)
    r.add_argument("--index", nargs="*", type=int, default=None)
    m = sub.add_parser("make")
    m.add_argument("--results", nargs="+", required=True)
    m.add_argument("--compare", nargs="*", default=None)
    m.add_argument("--out", default=None)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_make(a)


if __name__ == "__main__":
    main()
