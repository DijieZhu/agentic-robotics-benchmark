import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C

import numpy as np

SH = C.SH
SPC = SH.SPC
FROM_RECORD_ONLY = ("truth at presentation", "keepout_at_removal")


class _Ev:
    __slots__ = ("kind", "step", "fields")

    def __init__(self, d):
        self.kind, self.step, self.fields = d["kind"], int(d["step"]), d.get("fields") or {}


def _record(a, k):
    rec = {"step": int(a["step"][k]), "T_w": a["T_w"][k], "stop_margin": a["stop_margin"][k],
           "status": a["status"][k], "tray.pose": a["tray.pose"][k], "S.normal_N": {"S|tray": float(a["S_tray_N"][k])}}
    for s in SH.SIDES:
        rec[f"g_{s}.weld_state"] = a[f"g_{s}.weld_state"][k]
        rec[f"g_{s}.wrench"] = a[f"g_{s}.wrench"][k]
        rec[f"g_{s}.clamp_force"] = a[f"g_{s}.clamp_force"][k]
    return rec


def replay_monitor(result, trace, site, slots):
    a = trace.arrays()
    for f in ("T_w", "status", "stop_margin"):
        if f not in a or (len(a[f]) and np.isnan(a[f]).all()):
            return None
    mon = SH.Monitor(site, shelf_exception=result.get("shelf_exception", "step"),
                     zero_end=bool(result.get("zero_end", False)))
    step0 = int(result["step0"])
    pres = sorted(result.get("presentations", []), key=lambda p: p["step"])
    starts = [(int(p["step"]), slots[int(p["index"])]["slot"], p.get("te_step")) for p in pres]
    uev = sorted(result.get("unit_events", []), key=lambda e: int(e["step"]))
    st = a["step"]
    k = int(np.searchsorted(st, step0))
    n = len(st)
    j = 0
    pre = []
    while j < len(uev) and int(uev[j]["step"]) < step0:
        pre.append(_Ev(uev[j]))
        j += 1
    mon.unit_events(pre)
    i_p = 0
    fed = 0
    stop_step = None
    if result.get("events"):
        stop_step = None
    while k < n:
        c_first = int(st[k])
        while i_p < len(starts) and starts[i_p][0] <= c_first:
            mon.set_slot(starts[i_p][1], starts[i_p][2])
            i_p += 1
        hi = min(n, k + SPC)
        last = None
        for kk in range(k, hi):
            r = _record(a, kk)
            mon.feed(r, r["step"])
            last = r["step"]
            fed += 1
        evs = []
        while j < len(uev) and int(uev[j]["step"]) <= last:
            evs.append(_Ev(uev[j]))
            j += 1
        mon.unit_events(evs)
        k = hi
        if stop_step is not None and last >= stop_step:
            break
    return {"events": mon.events, "breaches": mon.breaches, "steps_fed": fed,
            "missing_fields": sorted(getattr(trace, "missing", set()))}


def first_event_check(result, trace, site, slots):
    rep = replay_monitor(result, trace, site, slots)
    rec_ev = (result.get("events") or [None])[0]
    if rep is None:
        return {"agree": None, "record": rec_ev, "recomputed": None, "source": "the trace lacks the monitor's fields"}
    re_ev = rep["events"][0] if rep["events"] else None
    if rec_ev is not None and (rec_ev.get("source") == "truth at presentation" or rec_ev.get("keepout_at_removal")):
        ok = re_ev is None or int(re_ev["step"]) >= int(rec_ev["step"])
        return {"agree": bool(ok), "record": rec_ev, "recomputed": re_ev, "source": "the monitor's record"}

    def key(e):
        return None if e is None else (e["kind"], int(e["step"]))
    return {"agree": key(rec_ev) == key(re_ev), "record": rec_ev, "recomputed": re_ev, "source": "the trace",
            "breaches_recomputed": len(rep["breaches"]), "breaches_record": len(result.get("breaches") or [])}
