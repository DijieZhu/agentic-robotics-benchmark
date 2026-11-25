"""Read the QC verify runs on task version 2.0 and judge them (Q1 oracle runs, Q2 / Q3 variants, degenerate
submissions D0-D6, D8, D9, D10).

    check_runs.py --runs DIR [--oracle-dir DIR2 --oracle TAG ...] [--frames A.json B.json | --frames-compare CMP.json]
                  --out OUT.json

Tags: D0-D4, D8, D9, D10 as named; D5 / D6 as D5_no_token / D6_wrong_token or D5 / D6. --frames-compare is the output
of q1_frames.py compare (the plate backend's reading: depth hash and the bytes outside the robot window equal, >= 99 %
of the window's pixels within 8 levels); --frames compares the two runs' rgb + depth hashes (the classic camera).

DIR holds the summary_<tag>.json files of qc_verify.py and their stages (stage_<tag>/out/episodes.json: the cell's
full record of every episode); the oracle runs (the freeze's two anchor runs through the same verify.py) may sit in
another folder. For every scored episode the score is recomputed HERE from the record by this file's own reading of
the scoring rules (tests/README.md, 'Scoring'; not by importing tests/voidscore): command intervals from the
executor's history, the 20 s window, the 0.02 m still rule on the 10 Hz truth log, the distance credit (full 0.20 m,
zero 0.40 m), the weights 0.05 / 0.15 / 0.80, the last accepted report per point, F (0.60 m; 1.0 / 0.85). The search
points and the pad are the site package's (from the frozen archive), except where a maintainer variant moved them
(its log in the reference summary gives the new coordinates). Expected values: oracle S = 1 everywhere; the
variants' from variants/plans.json (next to this file); the degenerate submissions 0 (D8 and D9: their plan files in
tests/review/degenerate/).
Record mutants (Q2's motion row in isolation, on real records with a counted, valid report of point 3): one 10 Hz
truth sample inside the window of that report moved by 0.03 m -> c3 = 0 with the only reason 'moved_in_window'; by
0.019 m -> no change; both through tests/voidscore.scoring.score_record (the grader's function) and this file's own
scorer.
D8's wall check: per wall, the push touched a wall box (the monitor's contacts of the push primitives) or stopped
against it; every report of point 3 there gives c3 = 0 (and would give 0 with a valid dwell).
D9's stuck reading: the debrief's rule (acceptance.yaml `stuck`: 3 primitives in a row each moving < 0.03 m and
turning < 5 deg) applied here to the record (the 10 Hz centroid and heading at each primitive's start and end); a
hidden episode has no debrief, so this is the reading a practice debrief would give from the same motion.
D10's check (heap_press): the plan found a heap of the open floor and pushed into it until the time limit, no report
and no finish were sent, the cell-side monitor counted contact samples with the pressed heap (the count is kept for
the read-out tool's heap-contact line), the episode ended at the time limit with 0 internal errors; the stuck reading
as for D9.
"""
import argparse
import copy
import io
import json
import lzma
import math
import os
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = os.path.normpath(os.path.join(HERE, "..", ".."))
TASK = os.path.join(os.environ["ARB_SRC_ROOT"], os.path.basename(DEV)[:-len("-dev")]) if os.environ.get("ARB_SRC_ROOT") else DEV[:-len("-dev")]
TESTS = os.path.join(TASK, "tests")
DEGEN = os.path.join(DEV, "tools", "degenerate")

W = {1: 0.05, 2: 0.15, 3: 0.80}         # the scoring weights (spelled here on purpose: an independent reading)
FULL, ZERO, WIN, STILL, RET, FNOT = 0.20, 0.40, 20.0, 0.02, 0.60, 0.85
EPS_T, EPS_L = 1e-9, 1e-12
STUCK_N, STUCK_MOVE, STUCK_TURN = 3, 0.03, 5.0


def site_of(archive):
    raw = open(archive, "rb").read()
    tf = tarfile.open(fileobj=io.BytesIO(lzma.decompress(raw)), mode="r:")
    for m in tf.getmembers():
        if m.name.endswith("site.json"):
            return json.loads(tf.extractfile(m).read().decode("utf-8"))
    raise ValueError(archive)


def my_score(rec, points, pad):
    hist = [h for h in rec.get("executor_history") or [] if h.get("kind") in ("primitive", "set_targets")]
    iv = [(float(h["t_start"]), float(h["t_end"])) for h in hist]
    acc = [c for c in rec.get("commands") or [] if c.get("kind") in ("primitive", "set_targets") and c.get("accepted")]
    if len(acc) > len(hist):
        iv.append((float(acc[-1]["t"]), math.inf))
    log = rec.get("log10") or []
    last = {}
    detail = []
    for k, r in enumerate(rec.get("reports") or []):
        if not r.get("accepted"):
            detail.append({"k": k, "i": r["i"], "accepted": False})
            continue
        t = float(r["t"])
        xy = r["xy"]
        why = []
        if t < WIN - EPS_T:
            why.append("before_start")
        if r.get("busy") or any(a < t - EPS_T and b > t - WIN + EPS_T for a, b in iv):
            why.append("command")
        dev = max([math.hypot(float(s[1]) - xy[0], float(s[2]) - xy[1]) for s in log
                   if t - WIN - EPS_T <= float(s[0]) <= t + EPS_T] or [math.inf])
        if dev > STILL + EPS_L:
            why.append("moved")
        p = points[int(r["i"])]
        d = math.hypot(xy[0] - p[0], xy[1] - p[1])
        c = (0.0 if why else 1.0) * min(1.0, max(0.0, 1.0 - (d - FULL) / (ZERO - FULL)))
        last[int(r["i"])] = c
        detail.append({"k": k, "i": r["i"], "t": t, "accepted": True, "d": d, "why": why, "max_dev": dev, "c": c,
                       "c_if_dwell_valid": min(1.0, max(0.0, 1.0 - (d - FULL) / (ZERO - FULL)))})
    c = {i: last.get(i, 0.0) for i in (1, 2, 3)}
    end = rec.get("end") or {}
    dp = math.hypot(end["xy"][0] - pad[0], end["xy"][1] - pad[1])
    running = bool(end.get("busy")) or any(math.isinf(b) for _a, b in iv)
    F = 1.0 if (dp <= RET + EPS_L and not running) else FNOT
    return {"S": F * sum(W[i] * c[i] for i in c), "c": c, "F": F, "d_pad": dp, "running_at_end": running,
            "reports": detail}


def moved_coords(rec, site):
    pts = {int(p["id"]): list(p["xy"]) for p in site["search_points"]}
    pad = list(site["pad"]["xy"])
    qv = (rec.get("reference") or {}).get("qc_variant") if isinstance(rec.get("reference"), dict) else None
    if qv:
        for e in qv.get("log") or []:
            if e["kind"] == "shift_point":
                pts[int(e["point"])] = list(e["new_xy"])
            elif e["kind"] == "shift_pad":
                pad = list(e["new_xy"])
    return pts, pad, qv


def load_run(runs, tag):
    p = os.path.join(runs, "summary_%s.json" % tag)
    if not os.path.isfile(p):
        return None, None
    s = json.load(open(p))
    ep = os.path.join(s["stage"], "out", "episodes.json")
    recs = json.load(open(ep)) if os.path.isfile(ep) else []
    return s, recs


def stuck_reading(rec):
    """The debrief's stuck rule on the record: per primitive, centroid move and heading change between the 10 Hz samples
    at its start and end; periods of >= STUCK_N primitives in a row below both thresholds."""
    log = rec.get("log10") or []
    ts = [float(s[0]) for s in log]

    def at(t):
        import bisect
        k = min(max(bisect.bisect_left(ts, t - 1e-9), 0), len(ts) - 1)
        return log[k]
    prims = []
    for h in rec.get("executor_history") or []:
        if h.get("kind") != "primitive":
            continue
        a, b = at(float(h["t_start"])), at(float(h["t_end"]))
        mv = math.hypot(float(b[1]) - float(a[1]), float(b[2]) - float(a[2]))
        dth = math.degrees((float(b[3]) - float(a[3]) + math.pi) % (2 * math.pi) - math.pi)
        prims.append((float(h["t_start"]), float(h["t_end"]), mv, dth, mv < STUCK_MOVE and abs(dth) < STUCK_TURN))
    periods = []
    k = 0
    while k < len(prims):
        if not prims[k][4]:
            k += 1
            continue
        j = k
        while j + 1 < len(prims) and prims[j + 1][4]:
            j += 1
        if j - k + 1 >= STUCK_N:
            periods.append({"t_start": prims[k][0], "t_end": prims[j][1], "primitives": j - k + 1})
        k = j + 1
    return {"n_primitives": len(prims), "periods": periods,
            "last_primitives": [{"t_start": p[0], "t_end": p[1], "moved_m": round(p[2], 4), "turned_deg": round(p[3], 2)}
                                for p in prims[-6:]]}


def judge_scored(s, recs, seeds, expected=None, variant=None):
    rows = []
    for rec, pin in zip(recs, seeds["sites"]):
        site = site_of(os.path.join(TESTS, "hidden", pin["archive"]))
        pts, pad, qv = moved_coords(rec, site)
        mine = my_score(rec, pts, pad)
        sc = rec.get("score") or {}
        row = {"tag": pin["tag"], "verify_S": rec.get("episode_score"), "verify_c": sc.get("c"), "verify_F": sc.get("F"),
               "my_S": mine["S"], "my_c": mine["c"], "my_F": mine["F"], "my_reports": mine["reports"],
               "same_as_my_scorer": abs(float(rec.get("episode_score")) - mine["S"]) < 1e-9 and
               all(abs(float(sc["c"][str(i)]) - mine["c"][i]) < 1e-9 for i in (1, 2, 3)) and
               abs(float(sc["F"]) - mine["F"]) < 1e-12,
               "internal_errors": rec.get("internal_errors"), "internal_errors_live": rec.get("internal_errors_live"),
               "render_failures": rec.get("render_failures"), "illegal_call_count": rec.get("illegal_call_count"),
               "retried_after": rec.get("retried_after"), "final_state_sha256": rec.get("final_state_sha256"),
               "pinned_final_state_sha256": pin.get("reference_final_state_sha256"),
               "done_reason": rec.get("done_reason"), "sim_time_s": rec.get("sim_time_s"),
               "points": [{k: p.get(k) for k in ("i", "accepted", "counted", "d_m", "dwell_valid", "dwell_failed", "c")}
                          for p in sc.get("points") or []],
               "d_pad_m": (sc.get("retrieval") or {}).get("d_pad_m")}
        if expected:
            e = expected
            row["expected"] = e
            row["as_expected"] = abs(float(rec.get("episode_score")) - e["S"]) < 1e-9 and \
                all(abs(float(sc["c"][k]) - v) < 1e-9 for k, v in e["c"].items()) and \
                ("F" not in e or abs(float(sc["F"]) - e["F"]) < 1e-12)
        if variant:
            row["variant_checks"] = variant_checks(variant, rec, sc, qv, mine)
        rows.append(row)
    return rows


def _per_point(rec):
    per = {}
    for r in rec.get("reports") or []:
        per.setdefault(int(r["i"]), []).append(bool(r["accepted"]))
    return per


def _shift_exact(log, pts, i, ret=None):
    sh = [e for e in log if e["kind"] == "shift_point" and int(e["point"]) == i]
    p = [q for q in pts if q.get("i") == i and q.get("counted")]
    if not sh or not p:
        return False, None, None
    return abs(p[0]["d_m"] - sh[0]["d_target"]) < 1e-9, p[0]["d_m"], sh[0]["d_target"]


def _pad_exact(log, sc, d):
    pad = [e for e in log if e["kind"] == "shift_pad"]
    got = (sc.get("retrieval") or {}).get("d_pad_m")
    return bool(pad) and abs(pad[0]["d_target"] - d) < 1e-12 and abs(got - d) < 1e-9, got


def variant_checks(name, rec, sc, qv, mine):
    out = {}
    log = (qv or {}).get("log") or []
    pts = sc.get("points") or []
    by = {p["i"]: p for p in pts if p.get("counted")}
    no_cmd_end = not (sc.get("retrieval") or {}).get("command_running_at_end")
    if name.startswith("V1"):
        out["p1_d_exact_0.19"], out["p1_d"], _ = _shift_exact(log, pts, 1)
        per = _per_point(rec)
        out["accepted_per_point"] = {str(k): v for k, v in per.items()}
        ks2 = [p["k"] for p in pts if p["i"] == 2]
        out["p2_counted_is_third"] = len(ks2) == 4 and by.get(2, {}).get("k") == ks2[2]
        out["p2_fourth_refused"] = per.get(2) == [True, True, True, False]
        out["p3_not_reported"] = 3 not in per
        early = [p for p in pts if p["i"] == 2 and p["accepted"] and not p.get("counted")]
        out["p2_early_reports_would_score"] = [p.get("c") for p in early]
        wb = [e for e in log if e["kind"] == "wrong_bottom"]
        it = [e for e in log if e["kind"] == "invalid_triple"]
        out["wrong_bottom"] = wb[0] if wb else None
        out["invalid_triple"] = it[0] if it else None
        out["pad_exact_0.59"], out["d_pad"] = _pad_exact(log, sc, 0.59)
        out["no_command_at_end"] = no_cmd_end
        out["ok"] = out["p1_d_exact_0.19"] and out["p2_counted_is_third"] and out["p2_fourth_refused"] and \
            out["p3_not_reported"] and len(early) == 2 and all(c == 0.0 for c in out["p2_early_reports_would_score"]) and \
            bool(wb) and bool(it) and wb[0]["reply"] == {"accepted": True, "error": None} and \
            wb[0]["history_bottom"] == wb[0]["sent"] and sorted(wb[0]["sent"]) != sorted(wb[0]["true_bottom"]) and \
            wb[0]["internal_errors"] == 0 and wb[0]["state_changed"] and \
            it[0]["reply"] == {"accepted": False, "error": "invalid_bottom"} and it[0]["state_unchanged"] and \
            out["pad_exact_0.59"] and no_cmd_end and int(rec.get("internal_errors") or 0) == 0
    elif name.startswith("V2"):
        out["p3_d_exact_0.19"], out["p3_d"], _ = _shift_exact(log, pts, 3)
        out["all_dwell_valid"] = all(by.get(i, {}).get("dwell_valid") is True for i in (1, 2, 3))
        out["end_d_pad_m"] = (sc.get("retrieval") or {}).get("d_pad_m")
        out["ok"] = out["p3_d_exact_0.19"] and out["all_dwell_valid"] and (out["end_d_pad_m"] or 0) > RET
    elif name.startswith("V3"):
        out["p3_d_exact_0.30"], out["p3_d"], _ = _shift_exact(log, pts, 3)
        out["p3_dwell_valid"] = by.get(3, {}).get("dwell_valid") is True
        out["same_final_state_as_admission_run"] = None   # filled by the caller (needs the pin)
        out["ok"] = out["p3_d_exact_0.30"] and out["p3_dwell_valid"]
    elif name.startswith("V4"):
        out["p1_dwell_failed"] = by.get(1, {}).get("dwell_failed")
        out["p2_d_exact_0.41"], out["p2_d"], _ = _shift_exact(log, pts, 2)
        out["p2_dwell_valid"] = by.get(2, {}).get("dwell_valid") is True
        out["p3_dwell_valid"] = by.get(3, {}).get("dwell_valid") is True
        out["pad_exact_0.61"], out["d_pad"] = _pad_exact(log, sc, 0.61)
        out["no_command_at_end"] = no_cmd_end
        out["ok"] = "command_in_window" in (out["p1_dwell_failed"] or []) and \
            set(out["p1_dwell_failed"]) <= {"command_in_window", "moved_in_window"} and out["p2_d_exact_0.41"] and \
            out["p2_dwell_valid"] and out["p3_dwell_valid"] and out["pad_exact_0.61"] and no_cmd_end
    return out


def record_mutants(recs, seeds):
    if TESTS not in sys.path:
        sys.path.insert(0, TESTS)
    from voidscore import scoring
    acc = scoring.load_acceptance()
    rows = []
    for rec, pin in zip(recs, seeds["sites"]):
        site = site_of(os.path.join(TESTS, "hidden", pin["archive"]))
        pts, pad, qv = moved_coords(rec, site)
        for e in (qv or {}).get("log") or []:            # the grader's function reads the site: give it the moved one
            if e["kind"] == "shift_point":
                [p for p in site["search_points"] if int(p["id"]) == int(e["point"])][0]["xy"] = list(e["new_xy"])
            elif e["kind"] == "shift_pad":
                site["pad"]["xy"] = list(e["new_xy"])
        r3s = [r for r in rec["reports"] if r["accepted"] and int(r["i"]) == 3]
        sc = rec.get("score") or {}
        p3 = [p for p in sc.get("points") or [] if p["i"] == 3 and p.get("counted")]
        if not r3s or not p3 or not p3[0].get("dwell_valid"):
            rows.append({"tag": pin["tag"], "skipped": "no counted valid report of point 3", "ok": None})
            continue
        r3 = r3s[-1]
        t = float(r3["t"])
        idx = [k for k, s in enumerate(rec["log10"]) if t - 15.0 <= float(s[0]) <= t - 5.0][0]
        out = {"tag": pin["tag"], "sample_t": rec["log10"][idx][0], "report_t": t, "c3_unmutated": p3[0]["c"]}
        for name, dm in (("shift_0.030", 0.030), ("shift_0.019", 0.019)):
            m = copy.deepcopy(rec)
            s = m["log10"][idx]
            s[1] = float(s[1]) + dm
            g = scoring.score_record(m, site, acc)
            mine = my_score(m, pts, pad)
            gp3 = [p for p in g["points"] if p["i"] == 3 and p["counted"]][0]
            out[name] = {"grader_c3": g["c"]["3"], "grader_reasons": gp3["dwell_failed"], "my_c3": mine["c"][3],
                         "grader_S": g["S"], "my_S": mine["S"]}
        c0 = float(p3[0]["c"])
        out["ok"] = out["shift_0.030"]["grader_c3"] == 0.0 and out["shift_0.030"]["grader_reasons"] == [
            "moved_in_window"] and out["shift_0.030"]["my_c3"] == 0.0 and \
            abs(out["shift_0.019"]["grader_c3"] - c0) < 1e-12 and abs(out["shift_0.019"]["my_c3"] - c0) < 1e-12
        rows.append(out)
    return rows


def judge_d8(rec, mine, qv):
    log = (qv or {}).get("log") or []
    wt = [e for e in log if e["kind"] == "wall_touch"]
    rep3 = [r for r in mine["reports"] if r["accepted"] and int(r["i"]) == 3]
    sent12 = [r for r in rec.get("reports") or [] if int(r["i"]) in (1, 2)]
    walls = []
    rows = (qv or {}).get("rows") or []
    for e in wt:
        ph = "wall_%s_push" % e["wall"]
        touched = sorted({b for r in rows if r.get("phase") == ph for b in (r.get("boxes") or []) if b.startswith("wall")})
        walls.append({"wall": e["wall"], "contact": e["push"]["contact"] or bool(touched), "push_stop": e["push"]["stop"],
                      "wall_boxes_touched_in_push": touched,
                      "push_n": e["push"]["n"], "approach_reached": e["approach"]["reached"],
                      "centroid_to_wall_m": e["at"].get("centroid_to_wall_m"),
                      "nearest_node_to_wall_m": e["at"].get("nearest_node_to_wall_m"),
                      "contacts": e["at"].get("contacts"), "d_sp3_m": e["d_sp3_m"], "reply": e["reply"],
                      "window": e["window"]})
    return {"walls": walls, "point3_reports": [{k: r[k] for k in ("t", "d", "why", "c", "c_if_dwell_valid")}
                                               for r in rep3],
            "reports_1_2_sent": len(sent12),
            "min_d_sp3_m": min([r["d"] for r in rep3] or [None]),
            "ok": len(wt) == 2 and all(w["contact"] for w in walls) and len(rep3) == 2 and
            all(r["c"] == 0.0 and r["c_if_dwell_valid"] == 0.0 and r["d"] > ZERO for r in rep3) and not sent12}


def judge_d9(rec, mine, qv):
    log = (qv or {}).get("log") or []
    tg = [e for e in log if e["kind"] == "debris_target"]
    pr = [e for e in log if e["kind"] == "pressed"]
    rp = [e for e in log if e["kind"] == "reports"]
    acc = [r for r in mine["reports"] if r["accepted"]]
    box = tg[0]["box"] if tg else None
    out = {"box": box, "target": tg[0] if tg else None,
           "pressed": {k: pr[0][k] for k in ("push", "at", "d_box_face_m", "box_centre_to_sp3_m")} if pr else None,
           "box_contact_samples_10hz": int(((rec.get("reference") or {}).get("monitor") or {}).get(
               "contact_box_samples", {}).get(box, 0)) if box else 0,
           "push_rows_touching_box": sum(1 for r in (qv or {}).get("rows") or []
                                         if r.get("phase") == "debris_push" and box in (r.get("boxes") or [])),
           "nearest_node_to_face_at_report_m": rp[0]["at"].get("nearest_node_to_wall_m") if rp else None,
           "contacts_at_report": rp[0]["at"].get("contacts") if rp else None,
           "window": rp[0]["window"] if rp else None,
           "reports": [{k: r[k] for k in ("i", "t", "d", "why", "c", "c_if_dwell_valid")} for r in acc],
           "stuck_reading": stuck_reading(rec)}
    out["ok"] = bool(tg) and bool(pr) and pr[0]["push"]["contact"] and out["push_rows_touching_box"] >= 1 and \
        out["box_contact_samples_10hz"] >= 1 and \
        len(acc) == 3 and all(r["c"] == 0.0 and r["c_if_dwell_valid"] == 0.0 for r in acc) and \
        [r for r in acc if int(r["i"]) == 3][0]["d"] >= 0.64 and int(rec.get("internal_errors") or 0) == 0 and \
        int(rec.get("internal_errors_live") or 0) == 0
    return out


def judge_d10(rec, mine, qv):
    log = (qv or {}).get("log") or []
    tg = [e for e in log if e["kind"] == "heap_target"]
    pr = [e for e in log if e["kind"] == "pressed_to_the_limit"]
    heap = tg[0]["heap"] if tg else None
    mon = ((rec.get("reference") or {}).get("monitor") or {}).get("contact_box_samples") or {}
    heap_samples = {k: int(v) for k, v in mon.items() if k.startswith("heap__")}
    out = {"heap_cls": tg[0].get("cls") if tg else None, "approach_side": pr[0].get("side") if pr else None,
           "push_n": (pr[0].get("push") or {}).get("n") if pr else None,
           "heap_contact_samples_10hz": heap_samples, "pressed_heap_samples_10hz": heap_samples.get(heap, 0),
           "reports_sent": len(rec.get("reports") or []), "finish_sent": pr[0].get("finish_sent") if pr else None,
           "done_reason": rec.get("done_reason"), "sim_time_s": rec.get("sim_time_s"),
           "accepted_reports_mine": sum(1 for r in mine["reports"] if r["accepted"]),
           "stuck_reading": stuck_reading(rec)}
    out["ok"] = bool(tg) and bool(pr) and out["pressed_heap_samples_10hz"] >= 1 and out["reports_sent"] == 0 and \
        out["accepted_reports_mine"] == 0 and pr[0].get("finish_sent") is False and \
        str(rec.get("done_reason")).startswith("time_limit") and int(rec.get("internal_errors") or 0) == 0 and \
        int(rec.get("internal_errors_live") or 0) == 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--oracle", nargs="*", default=["oracleA", "oracleB"])
    ap.add_argument("--oracle-dir", default=None, help="where the oracle runs' summaries are (default: --runs)")
    ap.add_argument("--frames", nargs=2, default=None)
    ap.add_argument("--frames-compare", default=None, help="q1_frames.py compare's output (the plate backend)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seeds = json.load(open(os.path.join(TESTS, "hidden", "seeds.json")))
    anchor = json.load(open(os.path.join(TESTS, "anchor.json")))
    plans = json.load(open(os.path.join(HERE, "variants", "plans.json")))["variants"]
    out = {"package": "the grader's QC on version 2.0: Q1 / Q2 / Q3 and the degenerate submissions", "runs_dir": a.runs,
           "anchor": {k: anchor.get(k) for k in ("metric", "value", "full_at")}}
    # ---- Q1: the oracle runs
    q1 = {"runs": {}}
    hashes = []
    for tag in a.oracle:
        s, recs = load_run(a.oracle_dir or a.runs, tag)
        if s is None:
            q1["runs"][tag] = None
            continue
        rows = judge_scored(s, recs, seeds, expected={"S": 1.0, "c": {"1": 1.0, "2": 1.0, "3": 1.0}, "F": 1.0})
        hashes.append([r["final_state_sha256"] for r in rows])
        q1["runs"][tag] = {"void_search_score": s["void_search_score"], "reward": s["reward"],
                           "contract_ok": s["contract_ok"], "privileged_mode": s["privileged_mode"],
                           "root_payload_ok": s.get("root_payload_ok"), "wall_s": s["wall_s"],
                           "stage_anchor": (s.get("anchor_in_stage") or {}).get("value"),
                           "debrief_files": s["debrief_files"], "episodes": rows,
                           "all_S_1": all(r["as_expected"] for r in rows),
                           "all_same_as_my_scorer": all(r["same_as_my_scorer"] for r in rows),
                           "all_equal_admission_hash": all(r["final_state_sha256"] == r["pinned_final_state_sha256"]
                                                           for r in rows),
                           "mutants_on_record": record_mutants(recs, seeds)}
    q1["runs_hash_equal"] = len(hashes) >= 2 and all(h == hashes[0] for h in hashes)
    q1["anchor_value"] = anchor.get("value")
    q1["anchor_reproduced"] = all(v and abs(float(v["void_search_score"]) - float(anchor["value"])) < 1e-12
                                  for v in q1["runs"].values())
    if a.frames:
        fa, fb = (json.load(open(p)) for p in a.frames)
        same = [x["sha256_rgb_depth"] == y["sha256_rgb_depth"] and x["cell_sha256"] == y["cell_sha256"]
                for x, y in zip(fa["frames"], fb["frames"])]
        q1["frames"] = {"site": fa["site"], "n": len(same), "identical": sum(same), "pids": [fa["pid"], fb["pid"]],
                        "state_hash_equal": fa["state_hash"] == fb["state_hash"],
                        "distinct_frames": len({x["sha256_rgb_depth"] for x in fa["frames"]}),
                        "indices": [x["index"] for x in fa["frames"]], "env": fa["env"],
                        "ok": all(same) and len(same) == 20 and fa["state_hash"] == fb["state_hash"]
                        and fa["pid"] != fb["pid"]}
    if a.frames_compare:
        fc = json.load(open(a.frames_compare))
        q1["frames"] = {k: fc.get(k) for k in ("n", "processes_differ", "state_equal", "depth_equal", "outside_equal",
                                               "windows", "window_min_share", "window_max_diff", "rgb_hash_equal",
                                               "distinct_frames", "lines")}
        q1["frames"]["ok"] = bool(fc.get("pass_")) and fc.get("n") == 20
    q1["ok"] = all(v and v["all_S_1"] and v["all_same_as_my_scorer"] and v["all_equal_admission_hash"] and
                   not v["debrief_files"] and all(m["ok"] for m in v["mutants_on_record"])
                   for v in q1["runs"].values()) and q1["runs_hash_equal"] and q1["anchor_reproduced"] and \
        (q1.get("frames") or {}).get("ok", False)
    out["Q1"] = q1
    # ---- variants
    var = {}
    for name, plan in plans.items():
        s, recs = load_run(a.runs, name)
        if s is None:
            var[name] = None
            continue
        rows = judge_scored(s, recs, seeds, expected=plan["expected"], variant=name)
        if name.startswith("V3"):
            for r in rows:
                r["variant_checks"]["same_final_state_as_admission_run"] = \
                    r["final_state_sha256"] == r["pinned_final_state_sha256"]
                r["variant_checks"]["ok"] = r["variant_checks"]["ok"] and \
                    r["variant_checks"]["same_final_state_as_admission_run"]
        var[name] = {"void_search_score": s["void_search_score"], "reward": s["reward"], "contract_ok": s["contract_ok"],
                     "privileged_mode": s["privileged_mode"], "root_payload_ok": s.get("root_payload_ok"),
                     "wall_s": s["wall_s"], "debrief_files": s["debrief_files"], "stage_added": s.get("stage_added"),
                     "expected_S": plan["expected"]["S"], "episodes": rows,
                     "max_abs_S_minus_expected": max(abs(float(r["verify_S"]) - plan["expected"]["S"]) for r in rows),
                     "all_as_expected": all(r["as_expected"] for r in rows),
                     "all_same_as_my_scorer": all(r["same_as_my_scorer"] for r in rows),
                     "all_variant_checks": all(r["variant_checks"]["ok"] for r in rows),
                     "no_cell_failure": all(int(r["internal_errors"] or 0) == 0 and not r["retried_after"]
                                            for r in rows)}
        if name.startswith(("V2", "V3")):
            var[name]["mutants_on_record"] = record_mutants(recs, seeds)
        var[name]["ok"] = len(rows) == 10 and var[name]["all_as_expected"] and var[name]["all_same_as_my_scorer"] and \
            var[name]["all_variant_checks"] and var[name]["no_cell_failure"] and not s["debrief_files"] and \
            abs(float(s["void_search_score"]) - plan["expected"]["S"]) < 1e-9 and \
            all(m["ok"] for m in var[name].get("mutants_on_record", []) if m["ok"] is not None)
    out["variants"] = var
    # ---- degenerate submissions
    deg = {}
    expect = {"D0": (0.0, 0.0), "D1": (1.0, 0.0), "D2": (1.0, 0.0), "D3": (1.0, 0.0), "D4": (0.0, 0.0),
              "D5_no_token": (1.0, 0.0), "D6_wrong_token": (1.0, 0.0), "D8": (1.0, 1.0), "D9": (1.0, 1.0),
              "D10": (1.0, 1.0)}
    alias = {"D5_no_token": "D5", "D6_wrong_token": "D6"}
    for tag, (ec, epriv) in expect.items():
        s, recs = load_run(a.runs, tag)
        if s is None and tag in alias:
            s, recs = load_run(a.runs, alias[tag])
        if s is None:
            deg[tag] = None
            continue
        rows = judge_scored(s, recs, seeds) if recs else []
        deg[tag] = {"void_search_score": s["void_search_score"], "reward": s["reward"], "contract_ok": s["contract_ok"],
                    "privileged_mode": s["privileged_mode"], "oracle_token_mismatch": s.get("oracle_token_mismatch"),
                    "root_payload_ok": s.get("root_payload_ok"), "illegal_call_count": s.get("illegal_call_count"),
                    "wall_s": s["wall_s"], "n_episodes": len(recs), "episode_S": [r["verify_S"] for r in rows],
                    "my_S": [r["my_S"] for r in rows], "F": [r["verify_F"] for r in rows],
                    "sim_time_s": [r["sim_time_s"] for r in rows],
                    "done_reasons": sorted({str(r["done_reason"]) for r in rows}),
                    "internal_errors": sum(int(r["internal_errors"] or 0) for r in rows),
                    "render_failures": sum(int(r["render_failures"] or 0) for r in rows),
                    "debrief_files": s["debrief_files"]}
        ok = float(s["void_search_score"] or 0.0) == 0.0 and float(s["reward"] or 0.0) == 0.0 and \
            float(s["contract_ok"] or 0.0) == ec and not s["debrief_files"] and \
            float(s["privileged_mode"] or 0) == epriv and float(s.get("root_payload_ok") or 0) == 1.0
        if recs:
            ok = ok and len(recs) == 10 and all(r["verify_S"] == 0.0 and abs(r["my_S"]) < 1e-12 and
                                                r["same_as_my_scorer"] for r in rows)
            deg[tag]["max_c_if_dwell_valid"] = max([x["c_if_dwell_valid"] for r in rows for x in r["my_reports"]
                                                    if x.get("accepted")] or [0.0])
        if tag == "D6_wrong_token":
            ok = ok and float(s.get("oracle_token_mismatch") or 0.0) == 1.0
        if tag in ("D5_no_token", "D6_wrong_token"):
            deg[tag]["illegal_calls_per_episode"] = [r["illegal_call_count"] for r in rows]
            ok = ok and all(int(r["illegal_call_count"] or 0) >= 1 for r in rows)
        if tag in ("D8", "D9", "D10"):
            per = []
            for rec, r in zip(recs, rows):
                site = site_of(os.path.join(TESTS, "hidden", [p for p in seeds["sites"] if p["tag"] == r["tag"]][0]["archive"]))
                pts, pad, qv = moved_coords(rec, site)
                mine = my_score(rec, pts, pad)
                j = {"D8": judge_d8, "D9": judge_d9, "D10": judge_d10}[tag](rec, mine, qv)
                j["tag"] = r["tag"]
                j["internal_errors"] = r["internal_errors"]
                per.append(j)
            deg[tag]["per_site"] = per
            deg[tag]["all_site_checks"] = all(p["ok"] for p in per)
            ok = ok and deg[tag]["all_site_checks"] and deg[tag]["internal_errors"] == 0
        deg[tag]["ok"] = ok
    out["degenerate"] = deg
    out["pass"] = {"Q1": q1["ok"], "variants": {k: (v or {}).get("ok") for k, v in var.items()},
                   "degenerate": {k: (v or {}).get("ok") for k, v in deg.items()}}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(json.dumps(out["pass"]))


if __name__ == "__main__":
    main()
