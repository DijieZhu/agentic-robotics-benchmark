import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C

import numpy as np

SH, QT, K = C.SH, C.QT, C.K
DT, SPC = SH.DT, SH.SPC
J6_INDEX = {m["side"]: m["index"] for m in K.MODULES if m["joint"] == 6}


def form_of(frozen):
    d = frozen["decision"]
    place = bool(d["place_passes"]) and bool(frozen.get("audit", {}).get("S_height_shown", True))
    return {"quantities": d["quantities"], "terms": d["terms"], "weights": d["weights"], "place_passes": place}


def shift_end(run):
    e = run.first_event()
    tiw = run.res.get("tray_in_work")
    if e is None:
        return {"kind": "end", "step": None, "t_shift_s": run.t_shift(run.end_step()), "slot": None,
                "tray_in_work": tiw}
    kind = e["kind"]
    if kind == "ReferenceAbort" and str(e.get("message", "")).startswith("late_time"):
        kind = "late_time"
    return {"kind": kind, "step": int(e["step"]), "t_shift_s": e["t_shift_s"],
            "slot": None if e.get("slot_index") is None else run.slots[e["slot_index"]]["slot"], "tray_in_work": tiw,
            "detail": {k: v for k, v in e.items() if k not in ("load_over_capacity_before",)}}


def tray_events(run, i):
    sl = run.slots[i]
    out = {"slot": sl["slot"], "label": sl["label"], "job": sl["job"], "reached": i in run.pres}
    if i not in run.pres:
        return out
    p = run.pres[i]
    out["missed_capture"] = bool(p.get("missed_capture"))
    out["decline_cycle"] = p.get("decline_cycle")
    out["carried"] = bool(p.get("carried"))
    caps = {}
    for e in run._events_of("grasp_closed", i):
        arm = e["fields"].get("arm")
        if arm in SH.SIDES and arm not in caps:
            caps[arm] = round((int(e["step"]) - int(p["step"])) * DT, 6)
    out["captured_after_s"] = caps
    end = run.end_step()
    nxt = [int(q["step"]) for q in run.pres.values() if int(q["step"]) > int(p["step"])]
    out["shift_ended_in_slot"] = run.first_event() is not None and (not nxt or end < min(nxt)) and end >= int(p["step"])
    return out


def tray_score(run, i, form):
    comp = run.completed_row(i)
    v = run.visited(i)
    rows = run.quantity_rows(i)
    Q = QT.q_value(rows, form)
    pl = run.place_row(i)
    place = QT.place_value(pl, form)
    if comp["completed"]:
        tray = QT.tray_value(comp["completed"], v["V"], Q, place)
    else:
        tray = 0.0
        Q = Q if math.isfinite(Q) else None
        place = place if math.isfinite(place) else None
    terms = term_values(rows, form)
    return {"slot": run.slots[i]["slot"], "completed": bool(comp["completed"]), "completed_row": comp,
            "V": v["V"], "visited": v["visited"], "holds": v["of"], "Q": Q, "terms": terms, "place": place,
            "place_row": pl, "tray": tray}


def term_values(rows, form):
    q = form["quantities"]
    out = {}

    def L(qn, x):
        return QT.L_credit(x, q[qn]["F"], q[qn]["zero_end"]["Z"])
    for t in form["weights"]:
        kept = form["terms"][t]
        if t == "H":
            hs = [math.prod(L(qn, h[qn]) for qn in kept) for h in rows["hold_rows"]]
        elif t == "P":
            hs = [L("d_m_mm", r["d_m_mm"]) * r["progress"] for r in rows["path_rows"]]
        else:
            hs = [L(kept[0], x) for x in rows[kept[0]]]
        out[t] = {"mean": (sum(hs) / len(hs)) if hs else 0.0, "rows": len(hs)}
    return out


def unit_score(run, form, weights=None):
    w = C.weights_of_record()["unit"] if weights is None else weights
    trays, evs = [], []
    for i, sl in enumerate(run.slots):
        evs.append(tray_events(run, i))
        if sl["label"] == "carry":
            trays.append(tray_score(run, i, form))
    j = run.J()
    carry = [t["tray"] for t in trays]
    mean = sum(carry) / len(carry) if carry else 0.0
    u = w["carry_trays"] * mean + w["J"] * j["J"]
    return {"unit_u": u, "carry_mean": mean, "J": {k: j[k] for k in ("TPR", "FPR", "J", "n_decline", "n_lap_carry",
                                                                      "note")},
            "decline_rows": j["decline_rows"], "lap_rows": j["lap_rows"], "trays": trays, "slot_events": evs,
            "shift_end": shift_end(run)}


GATE_NAMES = ("imports_and_setup", "size_cap", "static_import_audit", "answered", "on_time", "no_late_write",
              "no_process_left")


def gates(unit_records, static):
    g = {}
    recs = list(unit_records)
    ran = bool(recs)

    def stat(name, default=None):
        v = static.get(name, default)
        return None if v is None else bool(v)
    if not static.get("controller_file"):
        g["imports_and_setup"] = False
    elif not ran:
        g["imports_and_setup"] = None
    else:
        g["imports_and_setup"] = all(r["gates"].get("started") and r["gates"].get("setup_ok") for r in recs)
    g["size_cap"] = stat("size_cap")
    g["static_import_audit"] = stat("static_import_audit")
    g["answered"] = any(r["gates"].get("answered") for r in recs) if ran else None
    g["on_time"] = all(r["gates"].get("on_time_ok") for r in recs) if ran else None
    g["no_late_write"] = stat("no_late_write", True)
    g["no_process_left"] = all(r["gates"].get("no_process_left") for r in recs) if ran else None
    g = {k: (None if v is None else bool(v)) for k, v in g.items()}
    return {"gates": 1.0 if all(v is True for v in g.values()) else 0.0, "each": g,
            "not_run": [k for k in GATE_NAMES if g[k] is None]}


def raw_and_reward(unit_values, gate, anchor):
    vals = [float(v) for v in unit_values]
    if not vals:
        raise ValueError("no unit")
    raw = float(gate) * sum(vals) / len(vals)
    if not math.isfinite(raw) or not (anchor > 0 and math.isfinite(anchor)):
        raise ValueError(f"raw {raw} or anchor {anchor} is not a finite positive number")
    return raw, min(1.0, raw / float(anchor))


def run_of(record, trace, slots, jobs, site):
    res = record["shift"] if "shift" in record and "gates" in record else record
    if res is None:
        return None
    return QT.Run(res, trace, slots, jobs, site)


def grade_unit(record, trace, slots, jobs, site, form, check_events=True):
    run = run_of(record, trace, slots, jobs, site)
    if run is None:
        return {"unit_u": 0.0, "carry_mean": 0.0, "J": None, "trays": [], "shift_end": None,
                "no_shift": (record.get("setup_failed") or {}).get("why", "no shift was run")}
    out = unit_score(run, form)
    out["reported"] = reported(run, record, trace)
    if check_events:
        import events as EV
        out["events_check"] = EV.first_event_check(run.res, trace, site, slots)
    return out


def _slot_steps(run, i):
    p0 = int(run.pres[i]["step"])
    nxt = [int(q["step"]) for q in run.pres.values() if int(q["step"]) > p0]
    return p0, (min(nxt) if nxt else run.end_step())


def reported(run, record, trace):
    a = trace.arrays()
    st = a["step"]
    out = {"trays": []}
    kinds = None
    if isinstance(record, dict) and "cycle_kinds" in record:
        kinds = record["cycle_kinds"]
    slot_recs = {r["slot"]: r for r in run.res.get("slots", [])}
    for i, sl in enumerate(run.slots):
        if i not in run.pres:
            continue
        lo, hi = _slot_steps(run, i)
        k0, k1 = int(np.searchsorted(st, lo)), int(np.searchsorted(st, hi))
        row = {"slot": sl["slot"], "label": sl["label"]}
        row["internal_force"] = run.internal_force(i)
        gl = run.grasp_load_largest(from_step=lo, until_step=hi)
        row["grasp_load_held"] = gl["held"]["value"]
        row["grasp_load_single"] = gl["single"]["value"]
        if k1 > k0:
            mom = {s: float(np.max(np.linalg.norm(a[f"g_{s}.wrench"][k0:k1, 3:6], axis=1))) for s in SH.SIDES}
            row["weld_moment_max_Nm"] = mom
            if "T_w" in a and not np.isnan(a["T_w"][k0:k1]).all():
                row["winding_margin_min_K"] = float(K.OT_VALUE_C - np.nanmax(a["T_w"][k0:k1]))
            if "T_del" in a and not np.isnan(a["T_del"][k0:k1]).all():
                T = a["T_del"][k0:k1]
                row["J6_torque_max_Nm"] = {s: float(np.max(np.abs(T[:, J6_INDEX[s]]))) for s in SH.SIDES}
                p0 = int(run.pres[i]["step"])
                te = run.te_step(i)
                phases = {"closing": (p0, p0 + int(round(K.CARRY_START_S / DT))),
                          "carry": (p0 + int(round(K.CARRY_START_S / DT)), te if te is not None else hi),
                          "set_down": (te, te + int(round(K.SLOT["set_down_s"] / DT))) if te is not None else None}
                gp = {}
                for ph, span in phases.items():
                    if span is None:
                        continue
                    j0, j1 = int(np.searchsorted(st, span[0])), int(np.searchsorted(st, span[1]))
                    if j1 > j0:
                        gp[ph] = {s: float(np.max(np.abs(a["T_del"][j0:j1, SH.GRIPPER[s]]))) for s in SH.SIDES}
                row["grip_torque_max_Nm"] = gp
        still = run.stillness_rows(i)
        row["stillness_max_mm"] = max((r["stillness_mm"] for r in still), default=None)
        tilt = None
        if sl["label"] == "carry" or run.pres[i].get("carried"):
            p = int(run.pres[i]["step"])
            n = len(run.jobs[i]["targets"])
            tilts = []
            for kk in range(n):
                s0 = p + (QT.cy(K.CARRY_START_S) + kk * QT.cy(K.TARGET_S)) * SPC
                j0, j1 = int(np.searchsorted(st, s0)), int(np.searchsorted(st, s0 + QT.cy(K.MOVE_S) * SPC))
                if j1 > j0 and j1 <= len(st):
                    q = a["tray.pose"][j0:j1, 3:7]
                    R22 = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)
                    tilts.append(float(np.degrees(np.arccos(np.clip(R22, -1.0, 1.0))).max()))
            tilt = max(tilts) if tilts else None
            row["tilt_in_moves_max_deg"] = tilt
            ld = run.landing(i)
            row["v_c_mm_s"] = None if ld is None else 1e3 * ld["v_c_m_s"]
            sr = slot_recs.get(sl["slot"])
            row["d_s"] = None if sr is None else {"mm": sr.get("seat_dxy_mm"), "yaw_deg": sr.get("seat_yaw_deg")}
        if kinds is not None:
            c0 = (lo - run.step0) // SPC
            c1 = (hi - run.step0) // SPC
            late = 0
            for name, first, count in kinds:
                if name in ("late", "malformed", "held_late"):
                    late += max(0, min(first + count, c1) - max(first, c0))
            row["late_cycles"] = late
        out["trays"].append(row)
    out["shift_end"] = shift_end(run)
    out["load_over_capacity_max"] = run.res.get("load_over_capacity_max")
    out["winding_margin_min_K"] = run.res.get("winding_margin_min_K")
    return out
