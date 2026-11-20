"""ref_log.py -- the reference program's reading of the 72 h log (operator-visible: mission()['log'], the
configuration record): which cabin air unit's return air stepped up, when, and which route its build allows.

The step detector: ``onsets``, ``blocker_onset`` (306 / 306 synthetic logs right, max error 1.1 min).  The log is
the cell's mission() document ({t_min: [...], 'cau.<M>.return_dp_a': [...], ...}); the unit is reported
by its module name ('L2' | 'H2' | 'L1'); the route of the unit comes from the configuration record (element-type
with stops allowed: grasp first, then the fan stop; pad-type with stops inhibited: peel).
"""
from __future__ import annotations

import math

import numpy as np

LOGP = dict(k_sigma=6.0, win_min=30, min_step_pa=0.5)        # the step detector's parameters


def onsets(t_min, dp_a, dp_b, P=None) -> list:
    """Step candidates of one unit's two redundant return-pressure sensors (unchanged): a two-window mean
    difference above k_sigma noise, refined by a least-squares step fit; per step the size on the mean, on each
    sensor, and the pre-step level."""
    P = dict(LOGP, **(P or {}))
    W = int(P["win_min"])
    a, b = np.asarray(dp_a, float), np.asarray(dp_b, float)
    n = len(a)
    if n < 2 * W + 8:
        return []
    s = 0.5 * (a + b)
    sig = max(1.4826 * np.median(np.abs(np.diff(s) - np.median(np.diff(s)))) / math.sqrt(2.0), 1e-3)
    cs = np.concatenate([[0.0], np.cumsum(s)])
    diff = np.full(n, np.nan)
    for i in range(W, n - W + 1):
        diff[i] = (cs[i + W] - cs[i]) / W - (cs[i] - cs[i - W]) / W
    z = diff / (sig * math.sqrt(2.0 / W))
    out, i = [], W
    t_min = np.asarray(t_min, float)
    while i <= n - W:
        if np.isfinite(z[i]) and abs(z[i]) >= P["k_sigma"] and abs(diff[i]) >= P["min_step_pa"]:
            j1 = min(n - W, i + 2 * W)
            k = i + int(np.nanargmax(np.abs(z[i:j1 + 1])))
            lo, hi = max(0, k - W), min(n, k + W)
            tt = np.arange(lo, hi, dtype=float)
            best, kb = None, k
            for c in range(lo + 3, hi - 3):
                X = np.stack([np.ones_like(tt), (tt >= c).astype(float), tt - tt.mean()], 1)
                coef, *_ = np.linalg.lstsq(X, s[lo:hi], rcond=None)
                r = float(np.sum((s[lo:hi] - X @ coef) ** 2))
                if best is None or r < best:
                    best, kb = r, c
            pa, qa = float(np.median(a[max(0, kb - W):kb])), float(np.median(a[kb:kb + W]))
            pb, qb = float(np.median(b[max(0, kb - W):kb])), float(np.median(b[kb:kb + W]))
            out.append(dict(t=float(t_min[kb - 1] + t_min[kb]) / 2.0,
                            size=float(np.median(s[kb:kb + W]) - np.median(s[max(0, kb - W):kb])),
                            size_a=qa - pa, size_b=qb - pb, pre=float(np.median(s[max(0, kb - W):kb]))))
            i = j1 + 1
        else:
            i += 1
    return out


def log_units(log: dict) -> dict:
    """{module: (dp_a, dp_b)} of the mission log's return-pressure channels ('cau.<M>.return_dp_a' / '_b')."""
    out = {}
    for k in log:
        if not (isinstance(k, str) and k.startswith("cau.") and k.endswith(".return_dp_a")):
            continue
        m = k.split(".")[1]
        kb = f"cau.{m}.return_dp_b"
        if kb in log:
            out[m] = (log[k], log[kb])
    return out


def blocker_onset(log: dict, P=None) -> dict | None:
    """The largest positive step that both sensors of one unit show (each sensor carries more than half
    the step and neither more than twice the other): dict(unit, t, size, size_a, size_b, pre) or None."""
    t = np.asarray(log["t_min"], float)
    best = None
    for m, (da, db) in sorted(log_units(log).items()):
        for s in onsets(t, da, db, P):
            both = s["size_a"] > 0.5 * s["size"] and s["size_b"] > 0.5 * s["size"] and \
                max(s["size_a"], s["size_b"]) <= 2.0 * max(1e-6, min(s["size_a"], s["size_b"]))
            if s["size"] > 0 and both and (best is None or s["size"] > best["size"]):
                best = dict(s, unit=m)
    return best


def route_of(config_record: dict, module: str) -> str:
    """'fan_stop' (element-type, stops allowed: grasp first, then the stop) or 'peel' (pad-type or stops inhibited)
    from the configuration record's entry of CAU-<module> (the route from the configuration record)."""
    u = (config_record.get("units") or {}).get(f"CAU-{module}", {})
    if str(u.get("build", "")).startswith("element") and u.get("fan_stop") == "allowed":
        return "fan_stop"
    return "peel"


def diagnose(mission: dict) -> dict:
    """The reference's reading of the mission documents: the stepped unit, the onset [min on the log's axis], the
    route, the start-of-mission lighting per module."""
    ons = blocker_onset(mission["log"])
    cfg = mission.get("config_record") or {}
    unit = ons["unit"] if ons else None
    return dict(unit=unit, onset_min=(ons["t"] if ons else None), step_pa=(ons["size"] if ons else None),
                pre_pa=(ons["pre"] if ons else None), route=(route_of(cfg, unit) if unit else None),
                lights=dict(cfg.get("lights") or {}))
