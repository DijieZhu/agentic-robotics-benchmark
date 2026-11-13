"""Summarize P2 reference trials (run_reference.py JSON lines): success per kind x task, failure stages,
and the models whose every applicable trial succeeded.

    python -I summarize.py ../outputs/p2/batch1/*.jsonl
"""
import collections
import json
import sys


def main(paths):
    rows = [json.loads(line) for p in paths for line in open(p) if line.strip()]
    rows = [r for r in rows if r["status"] != "n/a"]
    by = collections.defaultdict(lambda: [0, 0])
    stages = collections.defaultdict(collections.Counter)
    model_ok = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        kind = r["model"].split("/")[0]
        k = (kind, r["task"])
        by[k][1] += 1
        model_ok[r["model"]][1] += 1
        if r["status"] == "ok":
            by[k][0] += 1
            model_ok[r["model"]][0] += 1
        else:
            stages[k][r.get("stage", r["status"])] += 1
    print("%-12s %-10s %8s  %s" % ("kind", "task", "ok/n", "failures by stage"))
    for k in sorted(by):
        ok, n = by[k]
        print("%-12s %-10s %4d/%-4d %s" % (k[0], k[1], ok, n, dict(stages[k].most_common(4))))
    per_kind = collections.defaultdict(lambda: [0, 0])
    for m, (ok, n) in model_ok.items():
        per_kind[m.split("/")[0]][1] += 1
        if ok == n:
            per_kind[m.split("/")[0]][0] += 1
    print("\nmodels with every trial ok:", {k: "%d/%d" % tuple(v) for k, v in sorted(per_kind.items())})
    print("trials: %d, wall %.0f s total, median steps %d" % (
        len(rows), sum(r.get("wall_s", 0) for r in rows), sorted(r.get("steps", 0) for r in rows)[len(rows) // 2]))


if __name__ == "__main__":
    main(sys.argv[1:])
