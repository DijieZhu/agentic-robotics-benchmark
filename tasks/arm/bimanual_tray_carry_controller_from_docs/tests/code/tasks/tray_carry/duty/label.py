import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
for _p in (os.path.join(TASK_DIR, "ref"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shift as SH
import numpy as np
import minimal as MR
import quantities as QT

import generator as GEN

K = SH.K
SWIVEL_STEP_DEG = 10.0
REDRAWS_MAX = 64
REASON_LABEL = "no tray of class {cls} passes its label on this unit"
ARM_MODULES = tuple(m["index"] for m in K.MODULES if m["kind"] == "arm")


def label_swivels():
    lo, hi = K.LABEL_SWIVEL_SEARCH_DEG
    n = int(round((hi - lo) / SWIVEL_STEP_DEG))
    return tuple(lo + SWIVEL_STEP_DEG * i for i in range(n + 1))


class TwRecorder:

    def __init__(self, store):
        self.store = store
        self.step, self.tw, self.mod = [], [], []
        self.first_out = {}

    def add(self, rec):
        self.store.add(rec)
        tw = np.asarray(rec["T_w"], dtype=np.float64)
        j = int(np.argmax(tw))
        self.step.append(int(rec["step"]))
        self.tw.append(float(tw[j]))
        self.mod.append(j)
        st = np.asarray(rec["status"])
        if np.any(st != 1):
            for k in np.flatnonzero(st != 1):
                self.first_out.setdefault(int(k), int(rec["step"]))

    def arrays(self):
        return np.asarray(self.step, np.int64), np.asarray(self.tw, np.float64), np.asarray(self.mod, np.int64)


def _cy(t):
    return int(round(t / K.CYCLE_S))


def carry_figures(res, run, rec, slots):
    st, tw, md = rec.arrays()
    pres = sorted(((int(p["index"]), int(p["step"])) for p in res["presentations"]), key=lambda x: x[1])
    end = int(st[-1]) + 1 if len(st) else 0
    bounds = {}
    for n, (i, p0) in enumerate(pres):
        p1 = pres[n + 1][1] if n + 1 < len(pres) else end
        bounds[i] = (p0, p1)
    events = res.get("events") or []

    def margin(a, b):
        m = (st >= a) & (st < b)
        if not m.any():
            return None, None
        k = int(np.argmax(np.where(m, tw, -np.inf)))
        return float(K.OT_VALUE_C - tw[k]), int(md[k])
    out = {"slots": [], "stopped": res.get("stopped"), "first_event": run.first_event(),
           "cycles": res.get("cycles"), "planned_cycles": res.get("planned_cycles"),
           "state_digest": res.get("state_digest"), "motion_digest": res.get("motion_digest"),
           "winding_margin_min_K_run": res.get("winding_margin_min_K"),
           "stop_margin_min_deg": res.get("stop_margin_min_deg"), "cpu_s": res.get("cpu_s")}
    first_p = pres[0][1] if pres else end
    wm, wmod = margin(int(st[0]) if len(st) else 0, first_p)
    out["before_first_presentation"] = {"winding_margin_K": wm, "module": wmod,
                                        "events": [e for e in events if int(e["step"]) < first_p]}
    for i, sl in enumerate(slots):
        row = {"slot": sl["slot"], "name": sl.get("name"), "label": sl["label"], "presented": i in bounds}
        if i in bounds:
            a, b = bounds[i]
            ev = [e for e in events if a <= int(e["step"]) < b]
            g = run.grasp_load_largest(from_step=a, until_step=b)
            wm, wmod = margin(a, b)
            row.update(from_step=a, until_step=b, events=ev[:5], first_event_kind=ev[0]["kind"] if ev else None,
                       grasp_held=g["held"]["value"], grasp_held_side=g["held"]["side"],
                       grasp_single=g["single"]["value"], grasp_single_side=g["single"]["side"],
                       winding_margin_K=wm, winding_module=wmod)
            if sl["label"] == "carry":
                c = run.completed_row(i)
                row["completed"] = bool(c["completed"])
                row["completed_reason"] = c.get("reason")
            else:
                d = run.decline_row(i)
                row["declined"] = bool(d["declined"])
                row["decline_cycle"] = d.get("decline_cycle")
        out["slots"].append(row)
    return out


NOT_PRESENTED = "not presented (the run ended before the slot)"


def carry_slot_verdict(row):
    r = []
    if not row.get("presented"):
        return False, [NOT_PRESENTED]
    if row.get("first_event_kind"):
        r.append(f"event {row['first_event_kind']} in the slot")
    if row.get("grasp_held") is None or row["grasp_held"] > K.LABEL_GRASP_SHARE_MAX:
        r.append(f"grasp load held {row.get('grasp_held')} above {K.LABEL_GRASP_SHARE_MAX}")
    if row.get("winding_margin_K") is None or row["winding_margin_K"] < K.LABEL_WINDING_MARGIN_K:
        r.append(f"winding margin {row.get('winding_margin_K')} K below {K.LABEL_WINDING_MARGIN_K} K")
    if row["label"] == "carry" and not row.get("completed"):
        r.append(f"not completed ({row.get('completed_reason')})")
    if row["label"] == "decline" and not row.get("declined"):
        r.append("not declined")
    return not r, r


def carry_run(shift, which, state, session, build, site_run, cell=None, trace_path=None, max_cycles=None):
    t0, c0 = time.perf_counter(), time.process_time()
    u = build()
    slots, jobs = GEN.run_slots(shift), GEN.run_jobs(shift)
    B = MR.beliefs_from_truth(u, state, session, site_run, slots, spare_coil=True)
    ref = MR.MinimalReference(B, slots, jobs)
    tr = QT.TraceStore()
    rec = TwRecorder(tr)
    res = SH.run_shift(u, site_run, slots, ref, jobs=jobs, recorder=rec, stop_at_event=True, max_cycles=max_cycles)
    run = QT.Run(res, tr, slots, jobs, site_run)
    fig = carry_figures(res, run, rec, slots)
    fig.update(which=which, wall_s=time.perf_counter() - t0, total_cpu_s=time.process_time() - c0,
               site_label=site_run.get("label"), J=run.J()["J"])
    if trace_path:
        tr.save(trace_path)
        fig["trace"] = trace_path
    return GEN.plain(fig)


def decline_feasibility(shift, i, best_state, best_session, best_unit, swivels=None, cell=None):
    sl = shift["slots"][i]
    tray = GEN.tray_of(sl)
    tc = GEN.true_check_of(best_state, best_session, best_unit)
    cm_nom = SH.nominal_cell_model(shift.get("cell"))
    ent = bytes.fromhex(sl["job_entropy"])
    out = []
    for s in (label_swivels() if swivels is None else swivels):
        j = SH.generate_job(sl["job"], ent, GEN.nominal_kin(), tray, cm_nom, label="decline",
                            contact_every=shift.get("contact_every", GEN.CONTACT_EVERY), true_check=tc, swivel=s)
        out.append({"swivel_deg": s, "passes": j["targets"] is not None,
                    "rejection_reasons": GEN.rejection_reasons(j["rejections"], "decline"),
                    "job": None if j["targets"] is None else GEN.plain(
                        {"targets": j["targets"], "swivels": j["swivels"], "swivel_fixed": j["swivel_fixed"],
                         "record": j["record"], "check": j["check"], "check_true": j["check_true"]})})
    return out


def decline_run(shift, i, job_s, best_state, best_session, build, site_b, trace_path=None, max_cycles=None):
    t0, c0 = time.perf_counter(), time.process_time()
    u = build()
    sl = dict(shift["slots"][i])
    slots, jobs = [sl], [job_s]
    n = len(job_s["targets"])
    carry = K.TARGET_S * n
    B = MR.beliefs_from_truth(u, best_state, best_session, site_b, slots, spare_coil=True)
    ref = MR.MinimalReference(B, slots, jobs, options={"carry_declines": True})
    tr = QT.TraceStore()
    rec = TwRecorder(tr)
    max_c = _cy(K.FREE_INTERVAL_S) + _cy(K.CARRY_START_S + K.LABEL_DECLINE_CARRY_SHARE * carry) + 1
    if max_cycles is not None:
        max_c = min(max_c, int(max_cycles))
    res = SH.run_shift(u, site_b, slots, ref, jobs=jobs, recorder=rec, stop_at_event=False,
                       ambient_at_presentation=float(best_session.ambient_C), max_cycles=max_c)
    p0 = int(res["presentations"][0]["step"]) if res["presentations"] else None
    ue = [e for e in (res.get("unit_events") or []) if e.get("kind") in ("fault_set",)]

    def share_of(step):
        return None if step is None or p0 is None else ((step - p0) * K.STEP_S - K.CARRY_START_S) / carry
    fo = {m: s for m, s in rec.first_out.items()}
    first = arm = None
    if fo:
        s0 = min(fo.values())
        first = {"step": s0, "modules": sorted(m for m, s in fo.items() if s == s0), "share": share_of(s0)}
        fa = {m: s for m, s in fo.items() if m in ARM_MODULES}
        if fa:
            s1 = min(fa.values())
            arm = {"step": s1, "modules": sorted(m for m, s in fa.items() if s == s1), "share": share_of(s1)}
    out = {"slot": sl["slot"], "name": sl.get("name"), "swivel_deg": job_s.get("swivel_fixed"), "n_targets": n,
           "carry_s": carry, "presentation_step": p0,
           "first_protection": first, "first_arm_protection": arm, "modules_out": fo, "fault_events": ue[:8],
           "events_n": len(res.get("events") or []),
           "event_kinds": sorted({e["kind"] for e in (res.get("events") or [])}), "stopped": res.get("stopped"),
           "cycles": res.get("cycles"), "max_cycles": max_c, "winding_margin_min_K": res.get("winding_margin_min_K"),
           "wall_s": time.perf_counter() - t0, "cpu_s": time.process_time() - c0}
    if trace_path:
        tr.save(trace_path)
        out["trace"] = trace_path
    return GEN.plain(out)


def decline_verdict(feas, runs):
    ok_sw = [f["swivel_deg"] for f in feas if f["passes"]]
    if not ok_sw:
        return False, ["the generator passes the job at no swivel of the search"], None
    by = {r["swivel_deg"]: r for r in runs}
    missing = [s for s in ok_sw if s not in by]
    if missing:
        return False, [f"no run at the passing swivels {missing}"], None

    def key(r):
        fp = r.get("first_protection")
        return math.inf if fp is None or fp.get("share") is None else fp["share"]
    kind = max((by[s] for s in ok_sw), key=lambda r: (key(r), -r["swivel_deg"]))
    sh = key(kind)
    if sh < K.LABEL_DECLINE_CARRY_SHARE:
        return True, [], kind["swivel_deg"]
    inv = plant_invalid_stop(kind.get("stopped"))
    if sh == math.inf and inv is not None:
        return False, [f"at the kindest swivel {kind['swivel_deg']} deg the run stopped at a step the plant cannot "
                       f"integrate (physics step {inv['step']}, {inv['kind']}) before any protection, so no "
                       f"protection before {K.LABEL_DECLINE_CARRY_SHARE} of the carry time"], kind["swivel_deg"]
    return False, [f"at the kindest swivel {kind['swivel_deg']} deg the first protection comes at "
                   f"{'none' if sh == math.inf else round(sh, 4)} of the carry time, not before "
                   f"{K.LABEL_DECLINE_CARRY_SHARE}"], kind["swivel_deg"]


def plant_invalid_stop(stopped):
    if not isinstance(stopped, dict):
        return None
    for k in ("plant_invalid", "integration"):
        if k in stopped:
            d = stopped[k] if isinstance(stopped[k], dict) else {}
            kind = d.get("unit_kind") or (", ".join(sorted(d["rose"])) if d.get("rose") else None) or k
            return {"step": d.get("step"), "kind": kind}
    return None


def tray_verdicts(shift, carry_true, carry_worst, decline):
    rows = []
    for i, sl in enumerate(shift["slots"]):
        side = "decline" if sl["class"] == "decline" else "carry"
        if side != sl["label"]:
            raise ValueError(f"slot {sl['slot']}: label {sl['label']} against class {sl['class']}")
        reasons = []
        for name, fig in (("true", carry_true), ("U_worst", carry_worst)):
            ok, r = carry_slot_verdict(fig["slots"][i])
            reasons += [f"{name}: {x}" for x in r]
        kind = None
        if side == "decline":
            d = decline.get(i) or decline.get(str(i))
            if d is None:
                reasons.append("decline: not run")
            else:
                ok, r, kind = decline_verdict(d["feasibility"], d["runs"])
                reasons += [f"U_best: {x}" for x in r]
        rows.append({"slot": sl["slot"], "name": sl["name"], "class": sl["class"], "side": side,
                     "passes": not reasons, "reasons": reasons, "kindest_swivel_deg": kind})
    shift_rows = []
    for name, fig in (("true", carry_true), ("U_worst", carry_worst)):
        b = fig["before_first_presentation"]
        if b["events"] or b["winding_margin_K"] is None or b["winding_margin_K"] < K.LABEL_WINDING_MARGIN_K:
            shift_rows.append(f"{name}: the hand-over or the free interval fails ({b})")
    return {"trays": rows, "shift": shift_rows, "ok": all(r["passes"] for r in rows) and not shift_rows}


def label_check(shift, ctx, trace_dir=None):
    t0, c0 = time.perf_counter(), time.process_time()

    def tp(name):
        return None if trace_dir is None else os.path.join(trace_dir, f"{shift['label']}_{name}.npz")
    st, ses, b, site = ctx["true"]
    ct = carry_run(shift, "true", st, ses, b, site, trace_path=tp("true"))
    st, ses, b, site = ctx["U_worst"]
    cw = carry_run(shift, "U_worst", st, ses, b, site, trace_path=tp("U_worst"))
    dec = {}
    st, ses, b, site = ctx["U_best"]
    for i, sl in enumerate(shift["slots"]):
        if sl["class"] != "decline":
            continue
        feas = decline_feasibility(shift, i, st, ses, b())
        runs = [decline_run(shift, i, f["job"], st, ses, b, site, trace_path=tp(f"best_{i}_{int(f['swivel_deg'])}"))
                for f in feas if f["passes"]]
        dec[i] = {"feasibility": [{k: v for k, v in f.items() if k != "job"} for f in feas], "runs": runs}
    v = tray_verdicts(shift, ct, cw, dec)
    v.update(carry_true=ct, carry_worst=cw, decline={str(k): x for k, x in dec.items()},
             shift_digest=shift["shift_digest"], wall_s=time.perf_counter() - t0, cpu_s=time.process_time() - c0)
    return GEN.plain(v)


def redraw_index(shift, role):
    sl = next(s for s in shift["slots"] if s["name"] == role)
    if sl["class"] == "decline":
        return int(sl["take_up"]["taken_draw"]) + 1
    return int(sl["tray_index"]) + 1


def only_not_presented(tray_row):
    rs = tray_row.get("reasons") or []
    return bool(rs) and all(str(x).endswith(NOT_PRESENTED) for x in rs)


def label_and_redraw(key, label, state, session, unit, n_decline, ctx, tray_index=None, redraws_max=REDRAWS_MAX,
                     trace_dir=None, log=None):
    ti = dict(tray_index or {})
    count = {}
    redraws = []
    n = 0
    while True:
        sh = GEN.generate_shift(key, label, state, session, unit, n_decline, tray_index=ti)
        if not sh["ok"]:
            return {"ok": False, "reason": sh["reason"], "shift": sh, "label": None, "redraws": redraws,
                    "checks": n}
        lc = label_check(sh, ctx, trace_dir=trace_dir)
        n += 1
        if log is not None:
            log({"check": n, "tray_index": ti, "ok": lc["ok"], "trays": lc["trays"], "shift": lc["shift"]})
        if lc["ok"]:
            return {"ok": True, "reason": None, "shift": sh, "label": lc, "redraws": redraws, "checks": n}
        failed = [t for t in lc["trays"] if not t["passes"] and not only_not_presented(t)]
        if not failed:
            reason = ("the hand-over or the free interval fails the label's CARRY side" if lc["shift"] else
                      "the label's CARRY runs end before a slot while no presented tray fails")
            return {"ok": False, "reason": reason, "shift": sh, "label": lc, "redraws": redraws, "checks": n}
        for t in failed:
            role = t["name"]
            count[role] = count.get(role, 0) + 1
            if count[role] > redraws_max:
                return {"ok": False, "reason": REASON_LABEL.format(cls=t["class"]), "shift": sh, "label": lc,
                        "redraws": redraws, "checks": n}
            new = redraw_index(sh, role)
            redraws.append({"role": role, "class": t["class"], "from": ti.get(role, 0), "to": new,
                            "reasons": t["reasons"]})
            ti[role] = new
