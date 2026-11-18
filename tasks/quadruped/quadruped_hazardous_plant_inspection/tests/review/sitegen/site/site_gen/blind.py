"""The blind-report rule for the generator's truth distributions: the best expected credit of a
report made without looking, per ladder instrument type.

The truths come from the generator's own sampler (ladder.truth_params, the 2009 values from ladder.last_2009 on
an independent stream), the credit from the cell's scoring (mission.scoring.value_error + credit_A with the
tolerances of acceptance_v1.yaml), so nothing is defined twice. What a program knows before looking is what the
round sheet prints: the type, the dial's scale, the lamp count and colours, the surface; so the best constant is
taken per scale (gauges) and per lamp count (lamp panels), and the bound is the largest of them.

    best_blind(cfg, "valve_lever") -> {"best_value", "credit", ...}
    last_2009_credit(cfg, "valve_lever") -> expected credit of citing an accepted frame and reporting the 2009 value
    mark_blind(rates_gyh) -> best constant for the survey mark over a set of realised mark rates
"""

from __future__ import annotations

import math

import numpy as np

TYPES = ("gauge_63", "gauge_100", "gauge_160", "level", "valve_lever", "valve_handwheel", "lamps", "thermal")
BOUND = 0.22


def _scoring():
    from mission.scoring import credit_A, value_error
    return credit_A, value_error


def _ev(cfg):
    from .accept import evidence_config
    return evidence_config(cfg)


def _ip(typ, prm, tr):
    from .ladder import BASE_TYPE
    t = BASE_TYPE[typ]
    tr = dict(tr)
    if t == "thermal":
        tr = {"units": [{"label": "unit", "temp_C": float(tr["temp_C"])}], "hot_index": 0}
    return {"type": t, "params": prm, "truth": tr}


def samples(cfg, typ, n=20000, seed=0):
    """n (params, truth) pairs from the generator's sampler."""
    from .ladder import truth_params
    g = np.random.default_rng(seed)
    return [truth_params(cfg, g, typ) for _ in range(n)]


def value_credit(ip, value, ev, credit_A, value_error):
    """V of a report with a value, as mission/scoring.py's score loop computes it: value_error, then 1 / 0 for an
    'all_correct' type (lamp panels), else credit_A with the zero factor."""
    err, tol, _ = value_error(ip, value, ev)
    if ev["types"][ip["type"]]["tolerance"]["kind"] == "all_correct":
        return 1.0 if err == 0.0 else 0.0
    return credit_A(err, tol, float(ev["common"]["fall_off"])) if err is not None else 0.0


def _credits(ips, value, ev, credit_A, value_error):
    return np.array([value_credit(ip, value, ev, credit_A, value_error) for ip in ips], float)


def _candidates(typ, prm):
    if typ.startswith("gauge"):
        lo, hi = float(prm["lo"]), float(prm["hi"])
        return list(np.linspace(lo, hi, 401))
    if typ in ("level", "valve_handwheel"):
        return list(np.arange(0.0, 100.01, 0.5))
    if typ == "valve_lever":
        return list(np.arange(-5.0, 95.01, 0.5))
    if typ == "thermal":
        return list(np.arange(20.0, 140.01, 0.5))
    raise KeyError(typ)


def best_blind(cfg, typ, n=20000, seed=0) -> dict:
    """The best constant report for `typ` given what the round sheet prints, and its expected credit."""
    credit_A, value_error = _scoring()
    ev = _ev(cfg)
    S = samples(cfg, typ, n, seed)
    if typ == "lamps":
        by_n = {}
        for prm, tr in S:
            by_n.setdefault(len(prm["colors"]), []).append(tuple(bool(x) for x in tr["states"]))
        best = {}
        for k, pats in sorted(by_n.items()):
            vals, cnt = np.unique(np.array(pats, int), axis=0, return_counts=True)
            j = int(np.argmax(cnt))
            best[k] = {"pattern": vals[j].tolist(), "credit": float(cnt[j] / len(pats)), "n": len(pats)}
        k = max(best, key=lambda q: best[q]["credit"])
        return {"type": typ, "credit": best[k]["credit"], "best_value": best[k]["pattern"], "given": "lamp count %d" % k,
                "per_condition": best, "samples": n, "bound": BOUND, "ok": best[k]["credit"] <= BOUND}
    groups = {}
    for prm, tr in S:
        key = (prm["lo"], prm["hi"], prm["unit"]) if typ.startswith("gauge") else "all"
        groups.setdefault(key, []).append(_ip(typ, prm, tr))
    fall = float(ev["common"]["fall_off"])
    per = {}
    for key, ips in sorted(groups.items(), key=lambda kv: str(kv[0])):
        cands = np.asarray(_candidates(typ, ips[0]["params"]), float)
        # the cell's value_error gives each sample's truth and tolerance (a numeric report's error is |v - t|);
        # the credit over all candidates at once, checked against credit_A on a few
        tt, tol = [], []
        for ip in ips:
            _, tl, t = value_error(ip, 0.0, ev)
            tt.append(float(t["temp_C"] if isinstance(t, dict) else t))
            tol.append(float(tl))
        tt, tol = np.asarray(tt), np.asarray(tol)
        E = np.abs(cands[:, None] - tt[None, :])
        C = np.clip((fall * tol[None, :] - E) / ((fall - 1.0) * tol[None, :]), 0.0, 1.0)
        for j in (0, len(cands) // 2, len(cands) - 1):
            ref = _credits(ips[:50], float(cands[j]), ev, credit_A, value_error)
            assert np.allclose(ref, C[j, :50], atol=1e-12), "vectorised credit differs from credit_A"
        cr = C.mean(axis=1)
        j = int(np.argmax(cr))
        per[str(key)] = {"best_value": float(cands[j]), "credit": float(cr[j]), "n": len(ips)}
    k = max(per, key=lambda q: per[q]["credit"])
    return {"type": typ, "credit": per[k]["credit"], "best_value": per[k]["best_value"], "given": k,
            "per_condition": per, "samples": n, "bound": BOUND, "ok": per[k]["credit"] <= BOUND}


def last_2009_credit(cfg, typ, n=20000, seed=1) -> dict:
    """Expected credit of citing an accepted frame and reporting the round sheet's 2009 value."""
    from .ladder import last_2009, truth_params
    credit_A, value_error = _scoring()
    ev = _ev(cfg)
    g = np.random.default_rng(seed)
    gv = np.random.default_rng(seed + 1000)          # the generator draws the 2009 values on their own stream
    from mission.scoring import _num
    cr = np.zeros(n)
    tv, lv = [], []
    for i in range(n):
        prm, tr = truth_params(cfg, g, typ)
        old = last_2009(cfg, gv, typ, prm)["value"]
        ip = _ip(typ, prm, tr)
        cr[i] = value_credit(ip, old, ev, credit_A, value_error)
        _, _, t = value_error(ip, old, ev)
        if typ != "lamps":
            t = float(t if not isinstance(t, dict) else t["temp_C"])
            o = float(_num(old))
            if typ.startswith("gauge"):                  # as a fraction of the dial's scale
                lo, hi = float(prm["lo"]), float(prm["hi"])
                t, o = (t - lo) / (hi - lo), (o - lo) / (hi - lo)
            tv.append(t)
            lv.append(o)
    r = float(np.corrcoef(tv, lv)[0, 1]) if tv else None
    return {"type": typ, "credit": float(cr.mean()), "corr_truth_2009": r, "samples": n, "bound": BOUND,
            "ok": float(cr.mean()) <= BOUND}


def mark_blind(rates_gyh, tol_rel=0.10, fall=2.5) -> dict:
    """The best constant report of the survey mark's dose rate over realised mark rates (relative tolerance)."""
    from mission.scoring import credit_A
    r = np.asarray(rates_gyh, float)
    if not len(r):
        return {"credit": None}
    cands = np.exp(np.linspace(math.log(r.min() * 0.7), math.log(r.max() * 1.3), 600))
    cr = [float(np.mean([credit_A(abs(v - t) / t, tol_rel, fall) for t in r])) for v in cands]
    j = int(np.argmax(cr))
    return {"best_value_gyh": float(cands[j]), "credit": cr[j], "n": int(len(r)), "bound": BOUND,
            "ok": cr[j] <= BOUND}


def table(cfg, n=20000) -> dict:
    out = {}
    for t in TYPES:
        b = best_blind(cfg, t, n)
        l2 = last_2009_credit(cfg, t, n)
        out[t] = {"best_constant": b["best_value"], "given": b["given"], "blind_credit": round(b["credit"], 4),
                  "last_2009_credit": round(l2["credit"], 4), "corr_truth_2009": l2["corr_truth_2009"],
                  "ok": b["ok"] and l2["ok"]}
    return out
