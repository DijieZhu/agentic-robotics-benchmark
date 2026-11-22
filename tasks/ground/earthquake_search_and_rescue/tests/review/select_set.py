"""The hidden set's and the training sites' pre-registered selection rule (tests/hidden/selection.json, rule version 2),
re-runnable by maintainers. Nothing in the grader runs this script; it documents and applies the rule.

  python3 tests/review/select_set.py --run hidden --index 3        one candidate: the frozen reference in a privileged
                                                                   cell of the task image; prints one JSON record
  python3 tests/review/select_set.py --assemble <records folder>   applies the rule to the records and prints the sets

The candidates are the seeds listed in tests/hidden/selection.json ("candidates": "hidden" and "training"), drawn once
from the operating system's entropy and written there before the reference ran on any of them. A candidate is ADMITTED
when the frozen reference program completes the mission on it: score 1.0 (located, kit, returned), no secondary
collapse, no protective stop, the robot not lost. The rule itself is the "rule" text of selection.json; `assemble`
below is its implementation.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)

TECHNIQUES = ("debris", "step", "narrow", "gap")


def run_one(pool, index):
    from rubblecell.server import RubbleCellServer
    sel = json.load(open(os.path.join(TESTS, "hidden", "selection.json")))
    seed = int(sel["candidates"][pool][index])
    cfg = json.load(open(os.path.join(TESTS, "grader_config.json")))
    srv = RubbleCellServer(config=cfg, mode="hidden", privileged=True, reference_dir=os.path.join(TESTS, "reference"))
    srv.reset(seed, "hidden" if pool == "hidden" else "training")
    srv.handle("reference_run", {})
    s = srv.finalize()
    kinds = []
    for o in sorted((o for o in srv.plan["openings"] if o.get("on_route")), key=lambda o: o["order"]):
        kinds.append([o.get("challenge") or "none"] + list(o.get("extra", [])))
    rec = {"pool": pool, "index": index, "seed": seed, "cell_version": s["cell_version"], "score": s["episode_score"],
           "parts": s["parts"], "lost": s["lost"], "voids": [v["what"] for v in s["voids"]],
           "stops": [v["what"] for v in s.get("stops", [])], "done_reason": s["done_reason"], "sim_time_s": s["sim_time_s"],
           "report_error_m": s["report_error_m"], "kit_dist_m": s["kit_dist_m"], "home_dist_m": s["home_dist_m"],
           "max_survivor_force_N": s["max_survivor_force_N"], "route_obstacles": kinds,
           "props": sum(1 for o in srv.plan["openings"] if o.get("challenge") == "prop")}
    rec["admitted"] = bool(rec["score"] >= 1.0 - 1e-9 and not rec["voids"] and not rec["stops"] and not rec["lost"])
    return rec


def _techniques(rec):
    return set(k for ks in rec["route_obstacles"] for k in ks if k in TECHNIQUES)


def assemble(records, n_hidden=8, n_training=6):
    """The rule (selection.json "rule"): the hidden set = the first 8 admitted hidden candidates in draw order; while
    one of the four technique obstacles lies on the route of fewer than 2 of the 8, the last site of the set that can
    go without breaking another technique's count is replaced by the next admitted candidate that has the missing one.
    The training sites = the first 6 admitted training candidates in draw order; while fewer than 2 of them have a
    load-bearing prop, the last one without is replaced by the next admitted candidate with one."""
    out = {}
    hid = sorted((r for r in records if r["pool"] == "hidden" and r["admitted"]), key=lambda r: r["index"])
    chosen, rest = hid[:n_hidden], hid[n_hidden:]
    notes = []

    def count(t, rs):
        return sum(1 for r in rs if t in _techniques(r))
    for t in TECHNIQUES:
        while len(chosen) == n_hidden and count(t, chosen) < 2:
            nxt = next((r for r in rest if t in _techniques(r)), None)
            if nxt is None:
                notes.append("technique %s on fewer than 2 routes and no admitted candidate left with it" % t)
                break
            out_r = None
            for r in reversed(chosen):
                others = [x for x in chosen if x is not r] + [nxt]
                if t not in _techniques(r) and all(count(u, others) >= min(2, count(u, chosen)) for u in TECHNIQUES):
                    out_r = r
                    break
            if out_r is None:
                notes.append("technique %s: no site can leave the set without breaking another count" % t)
                break
            chosen.remove(out_r)
            chosen.append(nxt)
            rest.remove(nxt)
            notes.append("hollowing rule: candidate %d replaced by candidate %d (technique %s)" % (out_r["index"], nxt["index"], t))
    chosen.sort(key=lambda r: r["index"])
    out["hidden"] = {"indices": [r["index"] for r in chosen], "seeds": [r["seed"] for r in chosen],
                     "technique_counts": {t: count(t, chosen) for t in TECHNIQUES}, "admitted": [r["index"] for r in hid],
                     "complete": len(chosen) == n_hidden}
    tr = sorted((r for r in records if r["pool"] == "training" and r["admitted"]), key=lambda r: r["index"])
    tch, trest = tr[:n_training], tr[n_training:]
    while len(tch) == n_training and sum(1 for r in tch if r["props"] > 0) < 2:
        nxt = next((r for r in trest if r["props"] > 0), None)
        last = next((r for r in reversed(tch) if r["props"] == 0), None)
        if nxt is None or last is None:
            notes.append("fewer than 2 training sites with a prop and no admitted candidate left with one")
            break
        tch.remove(last)
        tch.append(nxt)
        trest.remove(nxt)
        notes.append("training rule: candidate %d replaced by candidate %d (a prop)" % (last["index"], nxt["index"]))
    tch.sort(key=lambda r: r["index"])
    out["training"] = {"indices": [r["index"] for r in tch], "seeds": [r["seed"] for r in tch],
                       "with_prop": sum(1 for r in tch if r["props"] > 0), "admitted": [r["index"] for r in tr],
                       "complete": len(tch) == n_training}
    out["notes"] = notes
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["hidden", "training"])
    ap.add_argument("--index", type=int)
    ap.add_argument("--assemble")
    a = ap.parse_args()
    if a.run:
        print("RECORD " + json.dumps(run_one(a.run, a.index)), flush=True)
    elif a.assemble:
        recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(a.assemble, "*.json")))]
        print(json.dumps(assemble(recs), indent=1))


if __name__ == "__main__":
    main()
