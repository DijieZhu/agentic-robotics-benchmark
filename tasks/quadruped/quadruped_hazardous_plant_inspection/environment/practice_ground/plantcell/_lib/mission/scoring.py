"""The v1 score and the per-item debrief.

  S = R x sum_j w_j c_j
  instrument   c = D x E x V      V = 0.25 without a value, else A(|error|; tolerance)
  survey mark  c = D x E_M x A(relative error; 10 %)
  source       c = D x E_X x clip((3.0 m - error) / 2.0 m, 0, 1)
  A(e; tol) = 1 for e <= tol, falling linearly to 0 at 2.5 tol.  R = 1.0 upright within 1.0 m of the pad centre at
  the end, else 0.85. No negative term, nothing voids. D: the tag's last report to arrive over the link counts.
  E: one of its cited frames (at most 3) was taken in this mission before the report and passes the type's
  acceptance conditions (evidence.py). E_M: during the 10 s before the report the dosimeter stayed within 0.30 m of
  the mark centre (truth: the mean true rate where the robot stood during that dwell). E_X: before the report there
  were 3 consecutive seconds in which the robot's own dosimeter showed a mean >= 2.0 Gy/h and its body centre stayed
  within 3.0 m (horizontally) of the point the report gives.

Weights, tolerances and evidence conditions come from the shared acceptance_v1.yaml (config.load_cfg): eight
instruments at 0.10 each (a draft site with fewer instruments shares their 0.80 equally), the mark 0.06, the source
0.14 (shared equally if a site has several). On a ladder site every item carries its slot (A, B-step, B-kerb, B-low,
C-sealed, C-moved, C-relocated, C-annex, M, X), shown in the score table.

The functions take plain data (site.json, prior.json, the world's reports / frames / dose log / end state), so a run
folder can be re-scored without the simulator.
"""

from __future__ import annotations

import math

import numpy as np

INSTRUMENT_TYPES = ("gauge", "level", "valve_lever", "valve_handwheel", "lamps", "thermal")
CAMERA_NAME = {"nav": "navigation camera", "inspect": "inspection camera", "thermal": "thermal camera",
               "depth": "depth (navigation) camera"}
DRAFT_SLOT = {"needs_crouch": "B-low*", "needs_high_step": "B-step*", "room_missing": "C-annex*",
              "spread": "-", "survey": "X"}


def credit_A(err: float, tol: float, fall_off: float = 2.5) -> float:
    """1 up to the tolerance, linear to 0 at fall_off x tolerance, 0 beyond."""
    if err is None or not np.isfinite(err):
        return 0.0
    err = abs(float(err))
    if err <= tol * (1.0 + 1e-9):                # an error exactly at the tolerance (0.48 - 0.40 bar) is inside
        return 1.0
    hi = fall_off * tol
    if err >= hi:
        return 0.0
    return float((hi - err) / (hi - tol))


def source_credit(err_m: float, full_m: float = 1.0, zero_m: float = 3.0) -> float:
    if err_m is None or not np.isfinite(err_m):
        return 0.0
    if err_m <= full_m:
        return 1.0
    return float(np.clip((zero_m - err_m) / (zero_m - full_m), 0.0, 1.0))


def item_weights(prior: dict, cfg_weights: dict) -> dict:
    items = prior["work_order"]["items"]
    inst = [it["tag"] for it in items if it["type"] in INSTRUMENT_TYPES]
    marks = [it["tag"] for it in items if it["type"] == "dose_point"]
    srcs = [it["tag"] for it in items if it["type"] == "hotspot"]
    w = {}
    n = len(inst)
    each = float(cfg_weights.get("instrument_each", 0.10))
    total = float(cfg_weights.get("instruments_total", 8 * each))
    for t in inst:
        w[t] = (each if n == 8 else total / max(n, 1))
    for t in marks:
        w[t] = float(cfg_weights["mark"]) / len(marks)
    for t in srcs:
        w[t] = float(cfg_weights["source"]) / len(srcs)
    return w


def slot_of(ip: dict) -> str:
    """The ladder slot when the site carries one (ten-slot ladder sites), else a label from the draft grammar."""
    if ip.get("slot"):
        return str(ip["slot"])
    if ip["type"] == "dose_point":
        return "M"
    if ip["type"] == "hotspot":
        return "X"
    d = ip.get("discrepancy")
    if d == "ip_moved":
        return "C-moved*"
    if d == "ip_undocumented":
        return "C-undoc*"
    return DRAFT_SLOT.get(ip.get("reason"), "-")


# ---------------------------------------------------------------------------------------------- evidence decision
def evidence_failures(rec: dict, cfg_ev: dict) -> list:
    """The acceptance conditions of the evidence gate (cfg['evidence'], from acceptance_v1.yaml) that a stored evidence record (evidence.EvidenceModel.record: the
    measures sensor, distance_m, off_axis_deg, in_frame, tag_in_frame, app_px, visible, in_front) fails. Used when the
    record is made and again when a run folder is scored, so that a changed acceptance file re-decides old runs."""
    typ = rec["type"]
    rule = cfg_ev["types"][typ]
    com = cfg_ev["common"]
    failed = []
    if rec.get("sensor") != rule["sensor"]:
        failed.append("sensor (%s frame, needs %s)" % (rec.get("sensor"), rule["sensor"]))
    in_front = rec.get("in_front")
    if in_front is None:                                   # records written before the field existed
        in_front = "behind the camera" not in (rec.get("failed") or [])
    if not in_front:
        failed.append("behind the camera")
    app, off, dist = float(rec["app_px"]), float(rec["off_axis_deg"]), float(rec["distance_m"])
    if app < float(rule["min_px"]):
        failed.append("size %.1f px < %s" % (app, rule["min_px"]))
    # the off-axis limit; for a thermal item only where the site gives a face normal (acceptance_v1 off_axis_note;
    # the record says so in 'normal'; records of the draft sites' bearing housings have none)
    angle_applies = not (typ == "thermal" and rule.get("off_axis_needs_face_normal", True) and not rec.get("normal"))
    if angle_applies and off > float(rule["max_off_axis_deg"]):
        failed.append("off-axis %.1f deg > %s" % (off, rule["max_off_axis_deg"]))
    inf = rec.get("in_frame")
    if not (inf is not None and inf >= float(rule["min_in_frame"]) - 1e-9):
        failed.append("in frame %.2f < %s" % (float("nan") if inf is None else inf, rule["min_in_frame"]))
    if "min_tag_in_frame" in rule and not (rec.get("tag_in_frame", 0.0) >= float(rule["min_tag_in_frame"]) - 1e-9):
        failed.append("tag in frame %.2f < %s" % (rec.get("tag_in_frame", 0.0), rule["min_tag_in_frame"]))
    if dist > float(com["max_distance_m"]):
        failed.append("distance %.2f m > %s" % (dist, com["max_distance_m"]))
    if float(rule.get("min_distance_m", 0.0)) > 0.0 and dist < float(rule["min_distance_m"]):
        failed.append("distance %.2f m < %s" % (dist, rule["min_distance_m"]))
    vis_min = float(rule.get("min_visible_share", com.get("min_visible_share", 0.9)))
    vis = rec.get("visible")
    if not (vis is not None and vis >= vis_min):
        failed.append("visible %.2f < %s" % (float("nan") if vis is None else vis, vis_min))
    return failed


# ---------------------------------------------------------------------------------------------- value errors
def hottest_index(truth: dict) -> int:
    """The hot unit of a thermal item; when none runs hot (hot_index None, 15 % of the generator's thermal items),
    the warmest one (the round sheet asks for the hottest point of the tagged component)."""
    hi = truth.get("hot_index")
    units = truth.get("units", [])
    if hi is None:
        return int(np.argmax([float(u["temp_C"]) for u in units])) if units else 0
    return int(hi)



def _num(v, words=False):
    """A reported number (also given as a numeric string, e.g. "1.25"; true / false are not numbers:
    unreadable). words: the quarter-turn valve's words `open` / `opened` (0) and `closed` / `shut` (90), case
    and surrounding spaces ignored, accepted for that type only."""
    if isinstance(v, bool):
        return None
    try:
        if isinstance(v, str):
            s = v.strip().lower()
            if words and s in ("open", "opened"):
                return 0.0
            if words and s in ("closed", "shut"):
                return 90.0
        x = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def _lamp(x):
    """A lamp state as reported: true / false, a number (non-zero = on), or a word (case and surrounding spaces
    ignored: on / lit / 1 / true / yes = on; off / dark / unlit / 0 / false / no = off); null, a list or a dict is
    unreadable (null used to count as off)."""
    if x is None or isinstance(x, (list, tuple, dict)):
        return None
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("on", "lit", "1", "true", "yes"):
            return True
        if s in ("off", "dark", "unlit", "0", "false", "no"):
            return False
        return None
    try:
        return bool(x)
    except (TypeError, ValueError):
        return None


def _unit_index(u):
    """A temperature point's unit given by its index in the item's `units` (0, 1, ...): an integer (or a number with
    an integer value); true / false and other numbers are not an index."""
    if isinstance(u, bool) or not isinstance(u, (int, float)):
        return None
    if isinstance(u, float) and not (math.isfinite(u) and u.is_integer()):
        return None
    return int(u)


def value_error(ip: dict, value, cfg_ev: dict):
    """(error in the tolerance's unit, tolerance, truth for display) of a reported value."""
    typ = ip["type"]
    tr = ip.get("truth") or {}
    rule = cfg_ev["types"][typ]["tolerance"]
    if typ == "gauge":
        lo, hi = float(ip["params"]["lo"]), float(ip["params"]["hi"])
        t = float(tr["value"])
        v = _num(value)
        tol = float(rule["value"]) * (hi - lo) if rule["kind"] in ("fraction_of_span", "span_fraction") \
            else float(rule["value"])
        return (None if v is None else abs(v - t)), tol, t
    if typ == "level":
        t = float(tr["level_pct"])
        v = _num(value)
        return (None if v is None else abs(v - t)), float(rule["value"]), t
    if typ == "valve_handwheel":
        t = float(tr["opening_pct"])
        v = _num(value)
        return (None if v is None else abs(v - t)), float(rule["value"]), t
    if typ == "valve_lever":
        t = float(tr["angle_deg"])
        v = _num(value, words=True)
        return (None if v is None else abs(v - t)), float(rule["value"]), t
    if typ == "lamps":
        t = [bool(x) for x in tr["states"]]
        if not isinstance(value, (list, tuple)):          # a list, left to right (a word or a dict is no list)
            return None, 0.5, t
        v = [_lamp(x) for x in value]
        if any(x is None for x in v):
            return None, 0.5, t
        return (0.0 if v == t else 1.0), 0.5, t
    if typ == "thermal":
        units = tr.get("units", [])
        hi = hottest_index(tr)
        t_hot = float(units[hi]["temp_C"]) if units else float("nan")
        label = units[hi]["label"] if units else None
        if isinstance(value, dict):
            u = value.get("unit")
            temp = _num(value.get("temp_C"))
            same = (isinstance(u, str) and u == label) or (_unit_index(u) == hi)
            if not same and tr.get("hot_index") is None and units:
                # no unit is running hot (the generator's 15 % case): any unit within the tolerance of the hottest
                # one is a right answer to "which unit is hottest"
                k = next((i for i, x in enumerate(units) if (isinstance(u, str) and x["label"] == u)
                          or _unit_index(u) == i), None)
                same = k is not None and float(units[k]["temp_C"]) >= t_hot - float(rule["value"])
            if not same or temp is None:
                return float("inf"), float(rule["value"]), {"unit": label, "temp_C": t_hot}
            return abs(temp - t_hot), float(rule["value"]), {"unit": label, "temp_C": t_hot}
        temp = _num(value)
        return (None if temp is None else abs(temp - t_hot)), float(rule["value"]), {"unit": label, "temp_C": t_hot}
    return None, 1.0, None


# ---------------------------------------------------------------------------------------------- dose helpers
def _window_dwell(dose_log, t_report, center_xy, radius, need_s, sample_s=0.1, max_s=120.0):
    """(ok, truth rate, dwell) for E_M: the dosimeter stayed within `radius` of the mark centre for `need_s` before
    the report; truth = mean true rate over the contiguous in-radius dwell that ends at the report (up to max_s)."""
    t = np.asarray(dose_log["t"], float)
    if len(t) == 0:
        return False, None, 0.0
    P = np.asarray(dose_log["pos"], float)
    rate = np.asarray(dose_log["rate"], float)
    sel = t <= t_report + 1e-9
    t, P, rate = t[sel], P[sel], rate[sel]
    if len(t) == 0:
        return False, None, 0.0
    inside = np.hypot(P[:, 0] - center_xy[0], P[:, 1] - center_xy[1]) <= radius
    if not inside[-1]:
        return False, None, 0.0
    j = len(t) - 1
    while j - 1 >= 0 and inside[j - 1] and t[-1] - t[j - 1] < max_s:
        j -= 1
    dwell = float(t[-1] - t[j]) + sample_s
    return bool(dwell >= need_s - 1e-6), float(np.mean(rate[j:])), dwell


def _ex_reached(dose_log, t_report, thr_gyh, window_s, sens, sample_s, point=None, within_m=None):
    """E_X: before the report there was a window of `window_s` consecutive
    seconds (the 10 Hz dosimeter samples) in which the robot's own dosimeter showed a mean of at least thr_gyh AND the
    body centre (the dosimeter's place) stayed within `within_m` (horizontally) of `point`, the position the report
    gives. (ok, details): details carry the largest window mean anywhere, the largest among the windows spent within
    `within_m` of the point, the closest approach to the point, and which half failed."""
    t = np.asarray(dose_log["t"], float)
    c = np.asarray(dose_log["counts"], float)
    P = np.asarray(dose_log["pos"], float).reshape(-1, 3) if len(dose_log.get("pos", [])) else np.zeros((len(t), 3))
    sel = t <= t_report + 1e-9
    c, P = c[sel], P[sel]
    n = int(round(window_s / sample_s))
    out = {"max_3s_reading_gyh": 0.0, "max_3s_reading_near_gyh": 0.0, "closest_m": None, "failed": None}
    if point is not None and len(P):
        d = np.hypot(P[:, 0] - float(point[0]), P[:, 1] - float(point[1]))
        out["closest_m"] = round(float(d.min()), 3)
    if len(c) < n:
        out["failed"] = "rate"
        return False, out
    cs = np.concatenate([[0.0], np.cumsum(c)])
    reading = (cs[n:] - cs[:-n]) / (window_s * sens)
    out["max_3s_reading_gyh"] = float(reading.max())
    if point is None or within_m is None:
        near = np.ones(len(reading), bool)
    else:
        far = np.concatenate([[0], np.cumsum(d > float(within_m) + 1e-9)])
        near = (far[n:] - far[:-n]) == 0
    if near.any():
        out["max_3s_reading_near_gyh"] = float(reading[near].max())
    ok = bool(np.any(near & (reading >= thr_gyh)))
    if not ok:
        rate_any = bool(np.any(reading >= thr_gyh))
        out["failed"] = "near" if rate_any else ("rate" if near.any() else "rate_and_near")
    return ok, out


# ---------------------------------------------------------------------------------------------- the score
def score(site: dict, prior: dict, reports: list, frames: dict, dose_log: dict, end_state: dict, cfg: dict) -> dict:
    ev = cfg["evidence"]
    D = cfg["dose"]
    sens = float(D["sensitivity_cps_per_gyh"])
    k_gy = float(D["gyh_per_site_unit"])
    weights = item_weights(prior, cfg["weights"])
    ips = {ip["tag"]: ip for ip in site["ips"]}
    items = []
    for it in prior["work_order"]["items"]:
        tag = it["tag"]
        ip = ips.get(tag)
        typ = it["type"]
        w = weights.get(tag, 0.0)
        mine = [r for r in reports if r["tag"] == tag]
        dl = [r for r in mine if r["status"] == "delivered"]
        last = max(dl, key=lambda r: (r["t_sent"], r["id"])) if dl else None
        row = {"tag": tag, "type": typ, "slot": slot_of(ip) if ip else "-", "weight": round(w, 6),
               "reports": len(mine), "delivered": bool(last), "lost": sum(1 for r in mine if r["status"] == "lost")}
        credit = 0.0
        if typ in INSTRUMENT_TYPES and ip is not None:
            err, tol, truth = (None, None, None)
            row["truth"] = value_error(ip, None, ev)[2]
            if last is not None:
                row["value"] = last["value"]
                row["frame_ids"] = last["frame_ids"]
                E, why, used = 0, [], None
                for fid in last["frame_ids"][:3]:
                    fr = frames.get(fid)
                    if fr is None:
                        why.append("%s: no such frame" % fid)
                        continue
                    if fr["t"] > last["t"] + 1e-9:
                        why.append("%s: taken after the report" % fid)
                        continue
                    rec = (fr.get("evidence") or {}).get(tag)
                    if rec is None:
                        need = (ev["types"].get(typ) or {}).get("sensor")
                        if need and fr.get("sensor") and fr["sensor"] != need:
                            # a frame of the wrong camera is named as such; the
                            # world records evidence only in frames of the type's camera, so this is the only case
                            got = CAMERA_NAME.get(fr["sensor"], fr["sensor"] + " camera")
                            why.append("%s: %s %s frame; this instrument needs the %s" % (
                                fid, "an" if got[0] in "aeiou" else "a", got, CAMERA_NAME.get(need, need + " camera")))
                        else:
                            why.append("%s: the instrument is not in view" % fid)
                        continue
                    # re-decide from the stored measures with the current evidence gate (synthetic test records
                    # without measures keep their stored decision)
                    failed = evidence_failures(rec, ev) if "app_px" in rec else list(rec.get("failed", []))
                    if (not failed) if "app_px" in rec else rec["ok"]:
                        E, used = 1, fid
                        break
                    why.append("%s: %s" % (fid, "; ".join(failed)))
                if not last["frame_ids"]:
                    why.append("no frame cited")
                row["evidence"] = {"ok": bool(E), "frame": used, "why": why if not E else []}
                if used:
                    row["evidence"]["record"] = frames[used]["evidence"][tag]
                if last["value"] is None:
                    V = float(ev["common"]["value_free_credit"])
                    row["error"] = None
                else:
                    err, tol, truth = value_error(ip, last["value"], ev)
                    if ev["types"][typ]["tolerance"]["kind"] == "all_correct":
                        V = 1.0 if err == 0.0 else 0.0           # every lamp right, or nothing
                    else:
                        V = credit_A(err, tol, float(ev["common"]["fall_off"])) if err is not None else 0.0
                    row["error"] = None if err is None or not np.isfinite(err) else round(float(err), 4)
                    row["tolerance"] = tol
                row["V"] = round(V, 4)
                row["E"] = E
                credit = 1.0 * E * V
        elif typ == "dose_point" and ip is not None:
            mark = ip["pose"]["pos"][:2]
            row["truth_note"] = "mean true rate where the robot stood during the dwell (Gy/h)"
            if last is not None:
                ok, truth, dwell = _window_dwell(dose_log, last["t"], mark, float(ev["mark"]["radius_m"]),
                                                 float(ev["mark"]["dwell_s"]))
                row["value"] = last["value"]
                row["dwell_s"] = round(dwell, 2)
                row["E"] = int(ok)
                row["truth"] = None if truth is None else round(truth, 5)
                if ok and truth and last["value"] is not None:
                    rel = abs(float(last["value"]) - truth) / truth
                    row["error"] = round(rel, 5)
                    A = credit_A(rel, float(ev["mark"]["tolerance_rel"]), float(ev["common"]["fall_off"]))
                else:
                    row["error"] = None
                    A = 0.0
                row["A"] = round(A, 4)
                credit = float(ok) * A
            else:
                row["truth"] = round(float((ip.get("truth") or {}).get("dose_rate_uSv_h", float("nan"))) * k_gy, 5)
        elif typ == "hotspot" and ip is not None:
            src = (ip.get("truth") or {}).get("pos", ip["pose"]["pos"])
            row["truth"] = [round(float(src[0]), 3), round(float(src[1]), 3)]
            if last is not None:
                x, y = last["value"]
                okx, ex = _ex_reached(dose_log, last["t"], float(D["e_x_threshold_gyh"]),
                                      float(ev["source"].get("ex_window_s", D.get("e_x_window_s", 3.0))),
                                      sens, float(D["sample_s"]), point=(float(x), float(y)),
                                      within_m=float(D.get("e_x_within_m", 3.0)))
                err = math.hypot(float(x) - float(src[0]), float(y) - float(src[1]))
                row.update(value=[round(float(x), 3), round(float(y), 3)], error=round(err, 4), E=int(okx),
                           max_3s_reading_gyh=round(ex["max_3s_reading_gyh"], 3),
                           max_3s_reading_near_gyh=round(ex["max_3s_reading_near_gyh"], 3),
                           closest_to_reported_m=ex["closest_m"], approach_failed=ex["failed"])
                A = source_credit(err, float(ev["source"]["full_m"]), float(ev["source"]["zero_m"]))
                row["A"] = round(A, 4)
                credit = float(okx) * A
        row["credit"] = round(credit, 4)
        row["weighted"] = round(w * credit, 6)
        row["_w_c"] = w * credit
        items.append(row)
    RF = cfg.get("return") or {"within_m": 1.0, "value": 1.0, "otherwise": 0.85}
    up = bool(end_state.get("upright")) and float(end_state.get("pad_distance", 99.0)) <= float(RF["within_m"])
    # a fall or the dose limit always gives 0.85; a time-out or crash upright on the pad gives 1.0
    if end_state.get("reason") in ("robot_lost", "dose_limit"):
        up = False
    R = float(RF["value"]) if up else float(RF["otherwise"])
    total = sum(r.pop("_w_c") for r in items)          # unrounded (six weights of 0.8/6 sum to 0.8, not 0.799998)
    return {"items": items, "R": R, "end": {"pad_distance_m": round(float(end_state.get("pad_distance", 99.0)), 3),
                                            "upright": bool(end_state.get("upright")),
                                            "reason": end_state.get("reason")},
            "sum_weighted": round(total, 6), "S": round(R * total, 6)}


def debrief(scored: dict, site: dict, events: list) -> str:
    """The per-item debrief a training ground would print (no total)."""
    ips = {ip["tag"]: ip for ip in site["ips"]}
    out = ["MISSION DEBRIEF (per item; no total score)", ""]
    for r in scored["items"]:
        ip = ips.get(r["tag"], {})
        pos = ip.get("pose", {}).get("pos")
        where = ("actual position (%.2f, %.2f)" % (pos[0], pos[1])) if pos else ""
        line = "%-10s %-16s " % (r["tag"], r["type"])
        if not r["reports"]:
            line += "no report.  " + where
        elif not r["delivered"]:
            line += "%d report(s), none delivered (lost: %d).  %s" % (r["reports"], r["lost"], where)
        else:
            if r["type"] in INSTRUMENT_TYPES:
                e = r.get("evidence", {})
                line += "delivered; evidence %s" % ("ok (%s)" % e.get("frame") if e.get("ok") else
                                                   "not accepted: " + " | ".join(e.get("why", [])[:3]))
                if r.get("value") is None:
                    line += "; no value"
                else:
                    line += "; reported %s, true %s, error %s" % (_fmt(r.get("value")), _fmt(r.get("truth")),
                                                                  _fmt(r.get("error")))
            elif r["type"] == "dose_point":
                line += "delivered; dwell %.1f s %s; reported %s Gy/h, true %s Gy/h, relative error %s" % (
                    r.get("dwell_s", 0.0), "ok" if r.get("E") else "too short or off the mark", _fmt(r.get("value")),
                    _fmt(r.get("truth")), _fmt(r.get("error")))
            elif r["type"] == "hotspot":
                why = {"rate": "the 3 s mean never reached 2.0 Gy/h",
                       "near": "2.0 Gy/h only farther than 3.0 m from the reported point",
                       "rate_and_near": "never 3 s within 3.0 m of the reported point"}.get(r.get("approach_failed"))
                line += ("delivered; reported %s, actual %s, error %s m; dosimeter 3 s peak %s Gy/h, within 3.0 m of "
                         "the reported point %s Gy/h (%s)") % (
                    _fmt(r.get("value")), _fmt(r.get("truth")), _fmt(r.get("error")),
                    _fmt(r.get("max_3s_reading_gyh")), _fmt(r.get("max_3s_reading_near_gyh")),
                    "approached" if r.get("E") else "not approached: %s" % why)
            line += ".  " + where
        out.append(line)
    out.append("")
    out.append("end of mission: %s; robot %s, %.2f m from the pad centre" % (
        scored["end"]["reason"], "upright" if scored["end"]["upright"] else "not upright",
        scored["end"]["pad_distance_m"]))
    kinds = {}
    for e in events:
        if e["type"] in ("blocked", "protective_sit", "robot_lost", "dose_limit", "time_limit"):
            kinds[e["type"]] = kinds.get(e["type"], 0) + 1
    if kinds:
        out.append("events: " + ", ".join("%s x%d" % kv for kv in sorted(kinds.items())))
    return "\n".join(out) + "\n"


def _fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "%.3f" % v
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join("%s: %s" % (k, _fmt(x)) for k, x in v.items()) + "}"
    return str(v)
