"""Maintainers' tool (never run by the grader): the checks registered in tests/PREREG.md, on the hidden set, by exact
execution (ctasks.run_test: plan in cphys with the submitted material, one execution on the real fabric).

    python3 -I -B tests/review/mutants.py --workers 40 --cache FILE --out tests/review/mutants.json   (from the task folder)

Every row is a fixed params.json. The grid of blind constants is evaluated per skill (the slide tests for each
friction value, the overhang tests for each thickness value) and combined by the grader's product; the best
combination is then executed in full as one submission.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

TASK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(TASK, "tests"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cgrading as CG  # noqa: E402
import cphys as W  # noqa: E402
import make_set as MS  # noqa: E402

MUS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TS = [0.002, 0.003, 0.0045, 0.0065, 0.009, 0.0125, 0.017]
BASE = dict(young=1.0e5, poisson=0.3, thickness=0.0065, damping=0.003, mu=0.6)
MID_E = float(np.sqrt(1e4 * 5e5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache")
    ap.add_argument("--out", default=os.path.join(TASK, "tests", "review", "mutants.json"))
    a = ap.parse_args()
    MS.CACHE_PATH = a.cache
    if a.cache and os.path.exists(a.cache):
        for line in open(a.cache):
            try:
                r = json.loads(line)
                MS.CACHE[r["k"]] = r["y"]
            except ValueError:
                pass
    ref = CG.load_ref(os.path.join(TASK, "tests", "ref", "fabrics.json"))
    F = CG.FABRICS
    R = {k: ref[k]["reference"] for k in F}

    def scaled(**f):
        return {k: W.clamp_material({p: R[k][p] * f.get(p, 1.0) for p in W.KEYS}) for k in F}

    def const(m):
        return {k: W.clamp_material(m) for k in F}

    subs = {
        "reference": R,
        "real materials as they are": {k: ref[k]["material"] for k in F},
        "reference, friction x 0.95": scaled(mu=0.95), "reference, friction x 0.9": scaled(mu=0.9), "reference, friction x 1.1": scaled(mu=1.1),
        "reference, thickness x 0.9": scaled(thickness=0.9), "reference, thickness x 1.1": scaled(thickness=1.1), "reference, thickness x 1.2": scaled(thickness=1.2),
        "reference, Young's modulus x 2": scaled(young=2.0), "reference, Young's modulus x 0.5": scaled(young=0.5),
        "reference, damping x 3": scaled(damping=3.0), "reference, Poisson's ratio x 0.7": scaled(poisson=0.7),
        "friction from the reference, stiffness blind": {k: W.clamp_material(dict(R[k], young=MID_E, thickness=0.0065)) for k in F},
        "stiffness from the reference, friction blind": {k: W.clamp_material(dict(R[k], mu=0.55)) for k in F},
        "the middle of every range": const(dict(young=3.16e5, poisson=0.225, thickness=2.2e-3, damping=0.005, mu=0.775)),
        "the first version's example material": const(dict(young=2.0e5, poisson=0.3, thickness=0.002, damping=0.003, mu=0.5)),
        "the first version's practice material": const(dict(young=76990.34943013576, poisson=0.32664123825621316, thickness=0.003860282543357989,
                                                             damping=0.006747108105034395, mu=0.450937112093715)),
    }
    for k in F:
        subs["fabric %s's reference for all ten" % k] = const(R[k])
    jobs, meta = [], []
    for name, sub in subs.items():
        for k in F:
            for i, t in enumerate(ref[k]["tests"]):
                jobs.append(("run", sub[k], ref[k]["material"], ref[k]["mass_kg"], t))
                meta.append((name, k, i))
    for m in MUS:
        for k in F:
            for i, t in enumerate(ref[k]["tests"]):
                if t["fam"] == "slide":
                    jobs.append(("run", W.clamp_material(dict(BASE, mu=m)), ref[k]["material"], ref[k]["mass_kg"], t))
                    meta.append((("mu", m), k, i))
    for th in TS:
        for k in F:
            for i, t in enumerate(ref[k]["tests"]):
                if t["fam"] == "overhang":
                    jobs.append(("run", W.clamp_material(dict(BASE, thickness=th)), ref[k]["material"], ref[k]["mass_kg"], t))
                    meta.append((("t", th), k, i))
    print("exact runs", len(jobs), flush=True)
    with ProcessPoolExecutor(a.workers) as pool:
        res = MS.run(pool, jobs)
        q = {}
        for (name, k, i), r in zip(meta, res):
            q.setdefault(name, {}).setdefault(k, {})[i] = r["q"]

        def score(name):
            per, fam = {}, {}
            for k in F:
                s, fm = CG.aggregate([q[name][k][i] for i in range(len(ref[k]["tests"]))], ref[k]["tests"])
                per[k], fam[k] = s, fm
            return dict(metric=float(np.mean([per[k] for k in F])), slide=float(np.mean([fam[k]["slide"] for k in F])),
                        overhang=float(np.mean([fam[k]["overhang"] for k in F])), per_fabric=per)

        rows = {name: score(name) for name in subs}
        fm_mu = {m: {k: float(np.mean(list(q[("mu", m)][k].values()))) for k in F} for m in MUS}
        fm_t = {th: {k: float(np.mean(list(q[("t", th)][k].values()))) for k in F} for th in TS}
        grid = sorted(((float(np.mean([fm_mu[m][k] * fm_t[th][k] for k in F])), m, th) for m in MUS for th in TS), reverse=True)
        best = grid[0]
        bm = W.clamp_material(dict(BASE, mu=best[1], thickness=best[2]))
        bj = [("run", bm, ref[k]["material"], ref[k]["mass_kg"], t) for k in F for t in ref[k]["tests"]]
        br = MS.run(pool, bj)
    it = iter(br)
    q["best"] = {k: {i: next(it)["q"] for i in range(len(ref[k]["tests"]))} for k in F}
    rows["best blind constant of the grid, executed in full (friction %.1f, thickness %.4f)" % (best[1], best[2])] = score("best")
    out = dict(rows=rows, grid=dict(top5=grid[:5], slide_mean_by_friction={str(m): fm_mu[m] for m in MUS},
                                    overhang_mean_by_thickness={str(t): fm_t[t] for t in TS}, base=BASE))
    json.dump(out, open(a.out, "w"), indent=1)
    for name, r in rows.items():
        print("%-72s %.3f  slide %.2f overhang %.2f" % (name, r["metric"], r["slide"], r["overhang"]), flush=True)
    print("grid top 5 (product of skill means, friction, thickness):", [(round(a_, 3), m, t) for a_, m, t in grid[:5]])


if __name__ == "__main__":
    main()
