"""Walk the pre-registered candidates and write the graded set (tests/review/PREREG.md).

    /opt/venv/bin/python3 -I tests/review/select_set.py walk --out <dir> [--workers 6] [--version 1]
    /opt/venv/bin/python3 -I tests/review/select_set.py candidate <k> [--version 1]        (one candidate, a JSON line)

`walk` runs candidates k = 0, 1, 2, ... in order, each in a process of its own (`candidate`: the route generator,
then the privileged reference solution/oracle/fitter.py once, in process), `--workers` at a time but DECIDED in
order: a candidate is kept when the generator returns a route of three or four fixtures whose class is not yet full
(three: 12, four: 8) and the reference ends with credit 1. A candidate whose class was already full when its turn
came is recorded as such even if it was run. A candidate whose process ends by itself is run once more. It stops
at 20 kept and writes <dir>/episodes.json (the set, in the order kept) and <dir>/episode_selection.json (every
candidate walked: k, the outcome, and for a run candidate the reference's result). Copy both to tests/hidden/.
The keep rate among candidates whose class was open is printed; below 0.7 the pre-registration says: improve the
reference on development seeds and walk again under the next version of the key.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK = os.path.dirname(os.path.dirname(HERE))
CLASSES = {3: 12, 4: 8}
TOTAL = 20


def candidate_seed(k, version=1):
    key = "cable_routing_through_clips/graded/v%d/%d" % (int(version), int(k))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def run_candidate(k, version):
    sys.path.insert(0, os.path.join(TASK, "solution", "oracle"))
    import fitter as F
    import harness_env as H
    seed = candidate_seed(k, version)
    try:
        ep = H.make_episode(seed)
    except H.EnvError as exc:
        return {"k": k, "generator": "failed: %s" % exc}
    n = len(ep.fixtures)
    out = {"k": k, "generator": "ok", "fixtures": n, "kinds": "".join(f["type"][0] for f in ep.fixtures)}
    if n not in CLASSES:
        return out
    r = F.run_episode(seed, F.variant("reference"))
    out.update(credit=float(r["credit"]), marks=r["marks"], steps=int(r["steps"]), damaged=bool(r["damaged"]),
               peak_mean_tension=round(float(r["peak_mean_tension"]), 2), unstable=bool(r.get("unstable", False)))
    return out


def spawn(k, version):
    cmd = [sys.executable, "-I", os.path.abspath(__file__), "candidate", str(k), "--version", str(version)]
    for attempt in (1, 2):
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if p.returncode == 0 and p.stdout.strip():
            r = json.loads(p.stdout.decode().strip().splitlines()[-1])
            r["process_attempts"] = attempt
            return r
    return {"k": k, "generator": "process ended by itself twice", "process_attempts": 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("walk", "candidate"))
    ap.add_argument("k", nargs="?", type=int)
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.mode == "candidate":
        print(json.dumps(run_candidate(a.k, a.version)))
        return
    os.makedirs(a.out, exist_ok=True)
    kept, record, counts = [], [], {n: 0 for n in CLASSES}
    open_seen = open_kept = 0
    k = 0
    with concurrent.futures.ThreadPoolExecutor(a.workers) as pool:
        pending = {}
        while len(kept) < TOTAL:
            while len(pending) < a.workers:
                pending[k + len(pending)] = pool.submit(spawn, k + len(pending), a.version)
            r = pending.pop(k).result()
            n = r.get("fixtures")
            if r.get("generator") != "ok":
                r["outcome"] = "no episode"
            elif n not in CLASSES:
                r["outcome"] = "skipped: %d fixtures" % n
            elif counts[n] >= CLASSES[n]:
                r["outcome"] = "class full"
            else:
                open_seen += 1
                if r.get("credit") == 1.0:
                    r["outcome"] = "kept"
                    counts[n] += 1
                    open_kept += 1
                    kept.append(k)
                else:
                    r["outcome"] = "reference failed"
            record.append(r)
            print("candidate %d: %s" % (k, r["outcome"]), flush=True)
            k += 1
        for f in pending.values():
            f.cancel()
    rate = open_kept / max(1, open_seen)
    json.dump({"episodes": [{"seed": str(candidate_seed(i, a.version))} for i in kept]},
              open(os.path.join(a.out, "episodes.json"), "w"), indent=1)
    json.dump({"key": "cable_routing_through_clips/graded/v%d/<k>" % a.version, "walked": k, "kept": kept,
               "classes": {str(n): c for n, c in counts.items()}, "open_class_candidates": open_seen,
               "keep_rate_open_class": round(rate, 4), "candidates": record},
              open(os.path.join(a.out, "episode_selection.json"), "w"), indent=1)
    print("walked %d candidates; kept %d (three fixtures %d, four %d); keep rate among open-class candidates %.3f%s" % (
        k, len(kept), counts[3], counts[4], rate, "" if rate >= 0.7 else "  BELOW 0.7: see PREREG.md"))


if __name__ == "__main__":
    main()
