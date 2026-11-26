import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
FORMAT = "t05/hidden/select/1"

STEP_LABEL = "label check"
STEP_REFERENCE = "the privileged reference's run"
STEP_SPREAD = "the spread rule"
KILL_REASON_COUNT = 2
KILL_WINDOW = 8
PROPOSED_W_MIN = {"C1": 0.3, "P1": 0.25, "PL1": 0.00025}


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        return o.item()
    return o


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_plain(obj), f, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str)
    os.replace(tmp, path)


def _read(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _contract():
    p = os.path.join(TASK_DIR, "contract")
    if p not in sys.path:
        sys.path.insert(0, p)
    import constants as K
    return K


def reference_verdict(ref_run, K=None):
    K = K or _contract()
    r, cls = [], []
    fe = ref_run.get("first_event")
    if fe:
        r.append(f"an event: {fe.get('kind')} at step {fe.get('step')} ({fe.get('message', '')})".strip())
        cls.append(f"event {fe.get('kind')}")
    if not ref_run.get("ran_to_its_end"):
        r.append(f"the run did not reach its planned end ({ref_run.get('cycles')} of {ref_run.get('planned_cycles')} cycles; "
                 f"stopped: {ref_run.get('stopped_kind')})")
        if not fe:
            cls.append(f"stopped {ref_run.get('stopped_kind')}")
    whole = not fe and bool(ref_run.get("ran_to_its_end"))
    for row in ref_run.get("carry", []):
        if not row.get("completed"):
            r.append(f"carry tray of slot {row['slot']} not completed ({row.get('reason')})")
            if whole:
                cls.append("a carry tray not completed")
    for row in ref_run.get("decline", []):
        if not row.get("declined"):
            r.append(f"decline tray of slot {row['slot']} not DECLINED ({row.get('reason')})")
            if whole:
                cls.append("a decline tray not declined")
    g = ref_run.get("grasp_held")
    if g is None or g > K.LABEL_GRASP_SHARE_MAX:
        r.append(f"the grasp load held over the capacity rule's window {g} is above {K.LABEL_GRASP_SHARE_MAX} of "
                 f"capacity (single step {ref_run.get('grasp_single')})")
        cls.append("the grasp load above 0.80 of capacity")
    w = ref_run.get("winding_margin_min_K")
    if w is None or w < K.LABEL_WINDING_MARGIN_K:
        r.append(f"the smallest winding margin {w} K is below {K.LABEL_WINDING_MARGIN_K} K "
                 f"(module {ref_run.get('winding_module')})")
        cls.append("a winding less than 10 K below its protection")
    if not ref_run.get("every_cycle_on_time", False):
        r.append("not every cycle on time")
        cls.append("a late cycle")
    fig = {k: ref_run.get(k) for k in ("grasp_held", "grasp_held_side", "grasp_single", "winding_margin_min_K",
                                   "winding_module", "cycles", "planned_cycles", "state_digest", "motion_digest")}
    return {"passes": not r, "reasons": r, "classes": sorted(set(cls)), "figures": fig}


def kill_point(rows, window=KILL_WINDOW, count=KILL_REASON_COUNT):
    by = {}
    for r in rows:
        if int(r["unit_index"]) >= window:
            continue
        v = r.get("reference_check")
        if not v or v.get("passes"):
            continue
        for c in v.get("classes", []):
            by.setdefault(c, []).append(r["label"])
    for c in sorted(by):
        if len(by[c]) >= count:
            return {"class": c, "labels": by[c]}
    return None


def spread_ok(distances_to_kept, w_min, need=2):
    detail, ok = [], True
    for lab, d in sorted(distances_to_kept.items()):
        hits = sorted(q for q in w_min if d.get(q) is not None and d[q] >= float(w_min[q]))
        detail.append({"against": lab, "distance": d, "at_least_w_min": hits, "ok": len(hits) >= need})
        ok &= len(hits) >= need
    return ok, detail


def accept_decision(admission_outcome, ref_run_verdict, spread):
    if admission_outcome.get("outcome") != "admitted":
        return "rejected", admission_outcome.get("step"), admission_outcome.get("reason")
    if ref_run_verdict is None:
        raise ValueError("an admitted candidate needs its reference run")
    if not ref_run_verdict["passes"]:
        return "rejected", STEP_REFERENCE, "; ".join(ref_run_verdict["reasons"])
    if spread is not None and not spread[0]:
        bad = [d["against"] for d in spread[1] if not d["ok"]]
        return "rejected", STEP_SPREAD, f"closer than w_min in more than one scored quantity to {', '.join(bad)}"
    return "accepted", None, None


def _paths():
    for p in (os.path.join(TASK_DIR, "ref"), os.path.join(TASK_DIR, "duty"), HERE, os.path.join(TASK_DIR, "evidence", "shift_runner"),
              os.path.join(TASK_DIR, "grader")):
        if p not in sys.path:
            sys.path.insert(0, p)


def _mods():
    _paths()
    import shift as SH
    import draw as DR
    import generator as GEN
    import label as LB
    import dev as DV
    return SH, DR, GEN, LB, DV


def labelpart(args):
    SH, DR, GEN, LB, DV = _mods()
    reg = DR.open_registry(args["root"])
    sh = _read(args["shift_path"])
    site = args["site"]
    label = args["label"]
    part = args["part"]
    if part == "true":
        st, ses = reg.load_unit(label)
        return LB.carry_run(sh, "true", st, ses[0], DV.make_dev_builder(reg, label, site), site)
    if part == "U_worst":
        _rec, stb, sesb = reg.load_bracket("U_worst")
        site_b = DV.bracket_site(site, "U_worst", sesb)
        stb, sesb, build = DV.make_bracket_builder(reg, "U_worst", site_b)
        return LB.carry_run(sh, "U_worst", stb, sesb, build, site_b)
    if part == "decline":
        i = int(args["slot_index"])
        _rec, stb, sesb = reg.load_bracket("U_best")
        site_b = DV.bracket_site(site, "U_best", sesb)
        stb, sesb, build = DV.make_bracket_builder(reg, "U_best", site_b)
        feas = LB.decline_feasibility(sh, i, stb, sesb, build())
        runs = [LB.decline_run(sh, i, f["job"], stb, sesb, build, site_b) for f in feas if f["passes"]]
        return {"feasibility": [{k: v for k, v in f.items() if k != "job"} for f in feas], "runs": runs}
    raise ValueError(f"unknown part {part!r}")


def run_children(jobs, procs, log=None):
    pending = list(jobs)
    running = {}
    codes = {}
    env = dict(os.environ)
    while pending or running:
        while pending and len(running) < max(1, int(procs)):
            tag, argv, _rp, lp, cwd = pending.pop(0)
            os.makedirs(os.path.dirname(lp), exist_ok=True)
            lf = open(lp, "a", encoding="utf-8")
            lf.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S %Z')} {' '.join(argv)}\n")
            lf.flush()
            os.makedirs(cwd, exist_ok=True)
            p = subprocess.Popen(argv, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=cwd)
            running[tag] = (p, lf)
            if log:
                log(f"  child {tag}: pid {p.pid}")
        time.sleep(0.5)
        for tag in list(running):
            p, lf = running[tag]
            rc = p.poll()
            if rc is not None:
                lf.close()
                codes[tag] = rc
                del running[tag]
    return codes


def graded_site(DR, site, shift):
    return DR._cut_site(site, len(shift["slots"]))


def label_reference_agreement(label_record, ref_run_summary):
    ref = {"state_digest": (ref_run_summary or {}).get("state_digest"),
           "motion_digest": (ref_run_summary or {}).get("motion_digest")}
    ct = ((label_record or {}).get("label_check") or {}).get("carry_true") or {}
    lab = {"state_digest": ct.get("state_digest"), "motion_digest": ct.get("motion_digest")}
    if not label_record or not label_record.get("run", False) or not ct:
        return {"agree": False, "label": lab, "reference": ref,
                "why": "the candidate's label record holds no run of the label check on the true unit"}
    if None in lab.values() or None in ref.values():
        return {"agree": False, "label": lab, "reference": ref, "why": "a digest is missing"}
    ok = lab == ref
    return {"agree": ok, "label": lab, "reference": ref,
            "why": None if ok else "the label check's run on the true unit and the reference's run differ"}


def parallel_label_check(shift, ctx, trace_dir=None):
    _SH, DR, GEN, LB, _DV = _mods()
    t0 = time.perf_counter()
    n = ctx.setdefault("n_checks", 0) + 1
    ctx["n_checks"] = n
    wd = os.path.join(ctx["work"], f"check{n:02d}")
    shp = os.path.join(wd, "shift.json")
    _write(shp, shift)
    site_run = graded_site(DR, ctx["site"], shift)
    parts = [("true", None), ("U_worst", None)] + [("decline", i) for i, sl in enumerate(shift["slots"])
                                                   if sl["class"] == "decline"]
    jobs = []
    for part, i in parts:
        tag = part if i is None else f"decline{i}"
        ap, rp = os.path.join(wd, f"{tag}.args.json"), os.path.join(wd, f"{tag}.result.json")
        _write(ap, {"root": ctx["root"], "label": ctx["label"], "site": site_run, "shift_path": shp,
                    "part": part, "slot_index": i})
        argv = ["nice", "-n", "10", sys.executable, "-B", os.path.abspath(__file__), "labelpart", ap, rp]
        jobs.append((tag, argv, rp, os.path.join(wd, f"{tag}.log"), os.path.join(wd, "cwd")))
    codes = run_children(jobs, ctx.get("procs", 4), ctx.get("log"))
    res = {}
    for tag, _argv, rp, lp, _cwd in jobs:
        if codes.get(tag) != 0 or not os.path.exists(rp):
            raise RuntimeError(f"label part {tag} of {ctx['label']} ended with code {codes.get(tag)} (log {lp})")
        res[tag] = _read(rp)
    dec = {i: res[f"decline{i}"] for _p, i in parts if i is not None}
    v = LB.tray_verdicts(shift, res["true"], res["U_worst"], dec)
    cpu = sum(float(x.get("cpu_s") or 0) for x in (res["true"], res["U_worst"]))
    cpu += sum(float(r.get("cpu_s") or 0) for d in dec.values() for r in d["runs"])
    v.update(carry_true=res["true"], carry_worst=res["U_worst"], decline={str(k): x for k, x in dec.items()},
             shift_digest=shift["shift_digest"], wall_s=time.perf_counter() - t0, cpu_s=cpu,
             site_run={"seats_per_side": {k: len(x) for k, x in site_run["seats"].items()},
                       "how": "draw._cut_site(site, number of the shift's slots): the site the candidate record "
                              "stores (the graded world)"})
    out = GEN.plain(v)
    _write(os.path.join(wd, "label_check.json"), out)
    for tag, _argv, rp, _lp, _cwd in jobs:
        os.remove(rp)
    return out


def selection_shift_generator(root, work, procs=4, log=None):
    SH, DR, GEN, LB, _DV = _mods()

    def gen(ctx):
        label = ctx["label"]
        wl = os.path.join(work, label)
        os.makedirs(wl, exist_ok=True)
        cn = DR.capture_note(ctx["state"], ctx["site"])
        if not cn["ok"]:
            r = DR.generate_shift_duty(ctx)
            _write(os.path.join(wl, "label.json"), {"label": label, "run": False,
                                                    "why": "the capture note fails: no label check"})
            return r
        lctx = {"root": root, "label": label, "site": ctx["site"], "work": wl, "procs": procs, "log": log}
        orig_check, orig_gen = LB.label_check, GEN.generate_shift
        cache = {}

        def gen_cached(key, lab, state, session, unit, n_decline, tray_index=None, cell=None, **kw):
            k = json.dumps([lab, int(n_decline), sorted((tray_index or {}).items()), cell, sorted(kw.items())])
            if k not in cache:
                cache[k] = orig_gen(key, lab, state, session, unit, n_decline, tray_index=tray_index, cell=cell, **kw)
            return cache[k]
        try:
            LB.label_check = parallel_label_check
            GEN.generate_shift = gen_cached
            lr = LB.label_and_redraw(ctx["key"], label, ctx["state"], ctx["session"], ctx["unit"],
                                     ctx["n_decline"], lctx)
            final = lr["shift"]

            def gen_final(key, lab, state, session, unit, n_decline, tray_index=None, cell=None, **kw):
                return final
            GEN.generate_shift = gen_final
            r = DR.generate_shift_duty(ctx)
        finally:
            LB.label_check, GEN.generate_shift = orig_check, orig_gen
        rec = {"label": label, "run": True, "ok": lr["ok"], "reason": lr["reason"], "redraws": lr["redraws"],
               "checks": lr["checks"], "tray_index": {s["name"]: s.get("tray_index") for s in final.get("slots", [])},
               "shift_digest": final.get("shift_digest"), "sheet_digest": final.get("sheet_digest"),
               "label_check": lr["label"], "generations": len(cache)}
        _write(os.path.join(wl, "label.json"), rec)
        if not lr["ok"] and r.get("rejection") is None:
            r = dict(r)
            r["rejection"] = {"reason": lr["reason"], "slot": None,
                              "detail": {"step": STEP_LABEL, "redraws": lr["redraws"], "checks": lr["checks"]}}
        return r
    return gen


def candidate_step(root, place, work, procs=4, log=None):
    _SH, DR, _GEN, _LB, DV = _mods()
    reg = DR.open_registry(root)
    for end in ("U_worst", "U_best"):
        DV.bracket(reg, end)
    return DR.candidate(root, place, reg=reg, shift_generator=selection_shift_generator(root, work, procs, log))


def reference_summary(root, label, out_dir):
    _paths()
    import devset_run as DS
    import gtrace as TR
    import units as UN
    import common as C
    out = DS.run(root, label, "reference", out_dir, keep_trace=True)
    ctx = UN.UnitContext(root, label)
    tr = TR.GradeTrace.load(os.path.join(out_dir, out["trace"]))
    res = _read(os.path.join(out_dir, out["result"]))
    run = C.QT.Run(res, tr, ctx.slots, ctx.jobs, ctx.site)
    g = run.grasp_load_largest()
    chain = out["chain"]
    carry, dec = [], []
    for s in chain["slots"]:
        if s["label"] == "carry":
            c = s.get("completed") or {}
            carry.append({"slot": s["slot"], "completed": bool(c.get("completed")), "reason": c.get("reason"),
                          "V": s.get("V")})
        else:
            d = s.get("decline") or {}
            dec.append({"slot": s["slot"], "declined": bool(d.get("declined")), "reason": d.get("reason"),
                        "decline_cycle": d.get("decline_cycle")})
    tw = res.get("winding_max_C") or []
    wmod = int(max(range(len(tw)), key=lambda k: tw[k])) if tw else None
    st = res.get("stopped") or {}
    stopped_kind = None if not st else ("event" if st.get("event") else sorted(k for k in st if k != "cycle"))
    summ = {"label": label, "first_event": out["first_event"], "stopped_kind": stopped_kind,
            "cycles": res.get("cycles"), "planned_cycles": res.get("planned_cycles"),
            "ran_to_its_end": not st,
            "carry": carry, "decline": dec, "grasp_held": g["held"]["value"], "grasp_held_side": g["held"]["side"],
            "grasp_single": g["single"]["value"], "load_over_capacity_max": res.get("load_over_capacity_max"),
            "winding_margin_min_K": res.get("winding_margin_min_K"), "winding_module": wmod,
            "every_cycle_on_time": True,
            "every_cycle_on_time_basis": "the reference runs in the grader's process: every cycle is answered with "
                                         "its frames (no wire, no late reply); the run's cycles against its plan above",
            "state_digest": out["state_digest"], "motion_digest": out["motion_digest"], "J": chain.get("J"),
            "rows_n": out["rows_n"], "run_json": os.path.basename(os.path.join(out_dir, f"{label}__reference.json")),
            "trace": out["trace"], "result": out["result"], "run_cpu_s": out["run_cpu_s"],
            "run_wall_s": out["run_wall_s"]}
    return _plain(summ)


def spread_distances(root, label, kept, quantities):
    _SH, DR, _GEN, _LB, _DV = _mods()
    from openarm_unit import law as L
    reg = DR.open_registry(root)
    ua = reg.load_unit(label)[0]
    out = {}
    for k in kept:
        ub = reg.load_unit(k)[0]
        out[k] = {q: float(L.spread_distance(q, ua, ub)) for q in quantities}
    return out


def reject_in_registry(root, label, step, reason):
    _SH, DR, _GEN, _LB, _DV = _mods()
    reg = DR.open_registry(root)
    reg.reject(label, step, reason)
    return {"rejected": label, "step": step, "reason": reason}


def selection_records(out_dir, kept, label_files, ref_run_files):
    units = []
    for lab in kept:
        lf = label_files[lab]
        lr = _read(lf)
        lc = lr.get("label_check") or {}
        trays = []
        for i, t in enumerate(lc.get("trays", [])):
            row = {"slot": t["slot"], "name": t["name"], "class": t["class"], "side": t["side"],
                   "passes": t["passes"], "reasons": t["reasons"], "kindest_swivel_deg": t.get("kindest_swivel_deg")}
            for which, key in (("true", "carry_true"), ("U_worst", "carry_worst")):
                s = (lc.get(key) or {}).get("slots", [])[i]
                row[which] = {k: s.get(k) for k in ("presented", "completed", "declined", "grasp_held",
                                                     "grasp_held_side", "grasp_single", "winding_margin_K",
                                                     "winding_module", "first_event_kind")}
            if t["class"] == "decline":
                d = (lc.get("decline") or {}).get(str(i)) or {}
                row["U_best"] = {"feasibility": d.get("feasibility"),
                                 "runs": [{k: r.get(k) for k in ("swivel_deg", "first_protection",
                                                                  "first_arm_protection", "carry_s", "stopped")}
                                          for r in d.get("runs", [])]}
            trays.append(row)
        units.append({"label": lab, "label_record": os.path.relpath(lf, out_dir), "label_record_sha256": sha_file(lf),
                      "label_ok": lc.get("ok"), "shift_rows": lc.get("shift"), "shift_digest": lr.get("shift_digest"),
                      "redraws": lr.get("redraws"), "trays": trays,
                      "reference_run": os.path.relpath(ref_run_files[lab], out_dir),
                      "reference_run_sha256": sha_file(ref_run_files[lab])})
    rec = {"format": FORMAT + "/records", "units": units,
           "what": "the selection's own records of the set: the label check's figures per tray on the true unit, on "
                   "U_worst and on U_best, and the reference's run"}
    p = os.path.join(out_dir, "selection.json")
    _write(p, rec)
    return rec


def audit_selection_folder(folder, kept, label_files):
    man = {}
    for lab in kept:
        lr = _read(label_files[lab])
        rec = {"label": lr["label_check"], "unit": lab, "ok": lr["ok"], "redraws": lr["redraws"],
               "checks": lr["checks"], "shift_digest": lr["shift_digest"]}
        p = os.path.join(folder, f"{lab}.json")
        _write(p, rec)
        man[f"{lab}.json"] = sha_file(p)
    _write(os.path.join(folder, "MANIFEST.json"), man)
    return {"folder": folder, "manifest": man}


def main(argv):
    if argv and argv[0] == "labelpart":
        args = _read(argv[1])
        res = labelpart(args)
        _write(argv[2], res)
        print(argv[2], "done")
        return 0
    raise SystemExit(f"unknown command {argv[:1]}")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
