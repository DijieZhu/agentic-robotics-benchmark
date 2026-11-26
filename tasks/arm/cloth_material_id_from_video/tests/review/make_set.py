"""Maintainers' tool (never run by the grader): draws the hidden fabrics, their tests and the reference by the rule
of tests/PREREG.md from the committed key (tests/ref/key.json).

    python3 -I -B tests/review/make_set.py --workers 40 --cache FILE [--candidates 16]     (from the task folder)

The rule, in the order it is applied (every random number comes from sha256(key + "/" + label)):
  1. Fabric candidates j = 0, 1, 2, ...: a real material drawn from the population (bending length log-uniform
     3-16 cm, areal density uniform 0.06-0.40 kg/m2, Young's modulus log-uniform 1e4-5e5 Pa, thickness from the three,
     Poisson's ratio uniform 0.2-0.4, damping log-uniform 1e-3 to 8e-3 s, friction uniform 0.2-0.9).
  2. Per candidate: three calibration tests and eight test candidates per family (slide, overhang), each with the
     real outcome curve on nine control values. A candidate fabric is rejected if a real simulation of a
     calibration test fails.
  3. A test candidate gets a target if its real curve has a monotone stretch between two neighbouring control values
     with both outcomes inside the family's band (slide 35-110 mm; overhang -80 to -12 mm) and more than 2 mm apart:
     one such stretch is drawn, a control value is drawn uniformly on it, and the target is the real outcome there.
  4. The reference of the fabric: the cphys material with the real material's Young's modulus, Poisson's ratio and
     damping, and with friction and thickness calibrated against the real curves of the calibration tests
     (Nelder-Mead from the real values, 45 evaluations, squared error in tolerances capped at 16 per point).
  5. Test candidates are examined in order; one is kept if it has a target and the reference scores 1.0 on it
     (plan in cphys with the reference, one execution on the real fabric, error within the tolerance). The first
     four kept per family are the fabric's tests.
  6. A candidate fabric that keeps fewer than four of its eight candidates in either family is rejected whole (it is
     not scored on what is left).
  7. The first ten accepted candidates are the scored fabrics A-J, in order; the eleventh is the practice fabric P
     (its tests and reference are computed and recorded but not used).

Writes tests/ref/fabrics.json (real materials, masses, tests, references, the practice fabric), tests/ref/selection.json
(every candidate examined and why it was kept or not), solution/ref/params.json (the references of A-J) and
environment/gen/fabrics.json (real materials and masses for the renderer).
"""
import argparse
import hashlib
import json
import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np

TASK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(TASK, "tests"))
import cphys as W  # noqa: E402
import creal as R  # noqa: E402
import ctasks as T  # noqa: E402

AREA = W.SIZE_X * W.SIZE_Y
FAMS = ("slide", "overhang")
BAND = {"slide": (0.035, 0.110), "overhang": (-0.080, -0.012)}
GRID = 9
N_CAL, N_CAND, N_KEEP = 3, 8, 4
N_SCORED = 10
CAL_IDX = (1, 3, 4, 5, 7)
IDS = "ABCDEFGHIJ"
CACHE, LOCK, CACHE_PATH = {}, threading.Lock(), None


def rng_for(key, label):
    return np.random.default_rng(int(hashlib.sha256((key + "/" + label).encode()).hexdigest()[:16], 16))


def draw_fabric(rng):
    c = float(np.exp(rng.uniform(np.log(0.03), np.log(0.16))))
    rho = float(rng.uniform(0.06, 0.40))
    nu = float(rng.uniform(0.2, 0.4))
    E = float(np.exp(rng.uniform(np.log(1e4), np.log(5e5))))
    B = c ** 3 * rho * W.G
    t = float((12 * B * (1 - nu * nu) / E) ** (1 / 3))
    mat = dict(young=E, poisson=nu, thickness=t, damping=float(10 ** rng.uniform(-3.0, -2.1)), mu=float(rng.uniform(0.2, 0.9)))
    return mat, rho, c


def draw_test(fam, rng):
    if fam == "slide":
        return dict(fam=fam, pose=[float(rng.uniform(-0.24, -0.14)), float(rng.uniform(-0.20, 0.0)), float(rng.uniform(-0.25, 0.25))],
                    side=+1, d_release=float(rng.uniform(0.09, 0.14)))
    return dict(fam=fam, pose=[float(rng.uniform(0.10, 0.13)), float(rng.uniform(-0.20, 0.0)), float(rng.uniform(-0.12, 0.12))],
                lift=float(rng.uniform(0.012, 0.020)), speed=float(rng.uniform(0.08, 0.14)))


def grid(fam):
    return [float(u) for u in np.linspace(*T.U_RANGE[fam], GRID)]


def _key(*a):
    return json.dumps(a, sort_keys=True)


def _job(a):
    if a[0] == "sim":
        _, world, mat, mass, test, u = a
        return T.outcome(mat, mass, test, u, R if world == "real" else None)
    _, mat, real_mat, mass, test = a
    q, info = T.run_test(mat, real_mat, mass, test, R)
    return dict(info, q=q)


def run(pool, jobs):
    todo, seen = [], set()
    for j in jobs:
        k = _key(*j)
        if k not in CACHE and k not in seen:
            seen.add(k)
            todo.append(j)
    for j, y in zip(todo, pool.map(_job, todo, chunksize=1)):
        with LOCK:
            CACHE[_key(*j)] = y
            if CACHE_PATH:
                with open(CACHE_PATH, "a") as f:
                    f.write(json.dumps({"k": _key(*j), "y": y}) + "\n")
    return [CACHE[_key(*j)] for j in jobs]


def examine(pool, key, j, log):
    """Everything about fabric candidate j. Returns a record (accepted or not, with the reasons)."""
    mat, rho, c = draw_fabric(rng_for(key, "fabric/%d" % j))
    mass = rho * AREA
    rec = dict(candidate=j, material=mat, areal_density=rho, bending_length=c, mass_kg=mass)
    rc = rng_for(key, "fabric/%d/cal" % j)
    cal = [draw_test(fam, rc) for fam in FAMS for _ in range(N_CAL)]
    cands = {fam: [draw_test(fam, rng_for(key, "fabric/%d/test/%s/%d" % (j, fam, i))) for i in range(N_CAND)] for fam in FAMS}
    allt = cal + [t for fam in FAMS for t in cands[fam]]
    ys = run(pool, [("sim", "real", mat, mass, t, u) for t in allt for u in grid(t["fam"])])
    curve = {id(t): ys[i * GRID:(i + 1) * GRID] for i, t in enumerate(allt)}
    if any(y is None for t in cal for y in curve[id(t)]):
        return dict(rec, accepted=False, reason="a real simulation of a calibration test failed")
    # targets
    targets = {}
    tj = []
    for fam in FAMS:
        lo_b, hi_b = BAND[fam]
        us = grid(fam)
        for i, t in enumerate(cands[fam]):
            y = curve[id(t)]
            ok = [k for k in range(GRID - 1) if y[k] is not None and y[k + 1] is not None and lo_b <= y[k] <= hi_b and lo_b <= y[k + 1] <= hi_b
                  and y[0] is not None and y[-1] is not None and np.sign(y[k + 1] - y[k]) == np.sign(y[-1] - y[0]) and abs(y[k + 1] - y[k]) > 0.002]
            if not ok:
                continue
            r = rng_for(key, "fabric/%d/target/%s/%d" % (j, fam, i))
            k = ok[int(r.integers(len(ok)))]
            ustar = float(r.uniform(us[k], us[k + 1]))
            targets[(fam, i)] = ustar
            tj.append(("sim", "real", mat, mass, t, ustar))
    ty = run(pool, tj)
    for (fi, ustar), y in zip(list(targets.items()), ty):
        targets[fi] = None if y is None else (ustar, y)
    # reference
    real_cal = [curve[id(t)] for t in cal]

    def mk(th):
        return W.clamp_material(dict(mat, mu=float(th[0]), thickness=float(10 ** th[1])))

    def loss(th):
        m = mk(th)
        yc = run(pool, [("sim", "cphys", m, mass, t, grid(t["fam"])[k]) for t in cal for k in CAL_IDX])
        tot, i = 0.0, 0
        for t, yr in zip(cal, real_cal):
            for k in CAL_IDX:
                e = 4.0 if yc[i] is None else min(4.0, abs(yc[i] - yr[k]) / T.TOL[t["fam"]])
                tot += e * e
                i += 1
        return tot

    from scipy.optimize import minimize
    th0 = np.array([mat["mu"], np.log10(mat["thickness"])])
    r = minimize(loss, th0, method="Nelder-Mead",
                 options=dict(initial_simplex=np.array([th0, th0 + [0.04, 0], th0 + [0, 0.03]]), maxfev=45, xatol=1e-3, fatol=0.2))
    ref = mk(r.x)
    rec.update(reference=ref, calibration_loss=float(r.fun), calibration_tests=cal)
    # keep rule
    tests, exam = [], {}
    jobs, meta = [], []
    for fam in FAMS:
        for i, t in enumerate(cands[fam]):
            if targets.get((fam, i)):
                jobs.append(("run", ref, mat, mass, dict(t, target=targets[(fam, i)][1])))
                meta.append((fam, i))
    res = dict(zip(meta, run(pool, jobs)))
    ok_all = True
    for fam in FAMS:
        kept, rows = 0, []
        for i, t in enumerate(cands[fam]):
            tg = targets.get((fam, i))
            if not tg:
                rows.append(dict(i=i, kept=False, why="no target"))
                continue
            info = res[(fam, i)]
            full = info["q"] >= 1.0
            take = full and kept < N_KEEP
            rows.append(dict(i=i, kept=bool(take), why="kept" if take else ("reference full, beyond the first four" if full else "reference not full"),
                             q=info["q"], real=info["real"], target=tg[1], u_hidden=tg[0], u_planned=info["u"]))
            if take:
                kept += 1
                tests.append(dict(t, target=tg[1]))
        exam[fam] = rows
        if kept < N_KEEP:
            ok_all = False
    rec.update(examined=exam, tests=tests)
    if not ok_all:
        return dict(rec, accepted=False, reason="fewer than four of eight test candidates kept in a family")
    log("candidate %d accepted: c %.1f cm, mu %.2f; kept of examined: %s" % (
        j, 100 * c, mat["mu"], {fam: "%d/%d" % (sum(r_["kept"] for r_ in exam[fam]), len(exam[fam])) for fam in FAMS}))
    return dict(rec, accepted=True)


def main():
    global CACHE_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache")
    ap.add_argument("--candidates", type=int, default=16)
    ap.add_argument("--out", default=TASK)
    ap.add_argument("--dry", action="store_true", help="development only: one batch, whatever is accepted, never the task's files")
    a = ap.parse_args()
    key = json.load(open(os.path.join(TASK, "tests", "ref", "key.json")))["key"]
    CACHE_PATH = a.cache
    if a.cache and os.path.exists(a.cache):
        for line in open(a.cache):
            try:
                r = json.loads(line)
                CACHE[r["k"]] = r["y"]
            except ValueError:
                pass

    def log(s):
        print(s, flush=True)

    with ProcessPoolExecutor(a.workers) as pool:
        n = a.candidates
        while True:
            with ThreadPoolExecutor(n) as th:
                recs = list(th.map(lambda j: examine(pool, key, j, log), range(n)))
            # the rule looks at candidates in order: stop at the eleventh accepted one
            acc = [r for r in recs if r["accepted"]]
            if a.dry:
                assert os.path.abspath(a.out) != TASK
                break
            if len(acc) >= N_SCORED + 1:
                break
            n += 6
            log("only %d accepted of %d: examining %d" % (len(acc), len(recs), n))
    n_scored = N_SCORED if not a.dry else max(0, len(acc) - 1)
    if a.dry and not acc:
        log(json.dumps([dict(candidate=r["candidate"], reason=r.get("reason")) for r in recs]))
        return
    last = acc[n_scored]["candidate"]
    recs = [r for r in recs if r["candidate"] <= last]
    fabrics = {IDS[i]: acc[i] for i in range(n_scored)}
    P = acc[n_scored]
    keep = {fam: [sum(r_["kept"] or r_["why"].startswith("reference full") for r_ in r["examined"][fam]) / len(r["examined"][fam])
                  for r in recs if "examined" in r] for fam in FAMS}
    summary = dict(candidates_examined=len(recs), accepted=len(acc[:n_scored + 1]),
                   rejected=[dict(candidate=r["candidate"], reason=r["reason"]) for r in recs if not r["accepted"]],
                   reference_full_rate={fam: float(np.mean(v)) for fam, v in keep.items()})
    log(json.dumps(summary))
    out = a.out
    os.makedirs(os.path.join(out, "tests", "ref"), exist_ok=True)
    pub = lambda r: dict(material=r["material"], mass_kg=r["mass_kg"], tests=r["tests"], reference=r["reference"])  # noqa: E731
    json.dump({"fabrics": {k: pub(r) for k, r in fabrics.items()}, "practice": {"P": pub(P)}},
              open(os.path.join(out, "tests", "ref", "fabrics.json"), "w"), indent=1)
    json.dump(dict(summary=summary, candidates=recs, ids={k: r["candidate"] for k, r in fabrics.items()}, practice=P["candidate"]),
              open(os.path.join(out, "tests", "ref", "selection.json"), "w"), indent=1)
    os.makedirs(os.path.join(out, "solution", "ref"), exist_ok=True)
    json.dump({"fabrics": {k: r["reference"] for k, r in fabrics.items()}},
              open(os.path.join(out, "solution", "ref", "params.json"), "w"), indent=1)
    os.makedirs(os.path.join(out, "environment", "gen"), exist_ok=True)
    gen = {"P": dict(material=P["material"], mass_kg=P["mass_kg"])}
    gen.update({k: dict(material=r["material"], mass_kg=r["mass_kg"]) for k, r in fabrics.items()})
    json.dump({"fabrics": gen}, open(os.path.join(out, "environment", "gen", "fabrics.json"), "w"), indent=1)
    log("written")


if __name__ == "__main__":
    main()
