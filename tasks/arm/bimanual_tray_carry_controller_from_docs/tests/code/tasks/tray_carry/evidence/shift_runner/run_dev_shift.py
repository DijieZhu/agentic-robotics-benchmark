import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(TASK_DIR, "ref"))
sys.path.insert(0, os.path.join(TASK_DIR, "ref", "tests"))
if HERE not in sys.path:
    sys.path.append(HERE)
import shift as SH
import minimal as MR
import mutants as MU
import quantities as QT
import dev_run
from openarm_unit import rng as urng
from openarm_unit.registry import Registry
import uworst_carrying as UC

CARRYING = {}


def load(label, n_captures, registry, shelf_dz=None, cell=None):
    if label == "nominal":
        st, ses = SH.nominal_unit_state()
        site = SH.nominal_site(n_captures)
        if shelf_dz is not None:
            site = MU.shelf_site(site, shelf_dz)
        return (st, ses, site, SH.build_unit(st, ses, site, cell=cell),
                lambda sl: dev_run.TEST_ENTROPY + bytes([sl]), None)
    root = registry if os.path.isabs(registry) else os.path.join(TASK_DIR, "ref", registry)
    reg = Registry.open(root, SH.K.PROFILE_PATH)
    CARRYING.clear()
    if label in ("U_worst", "U_best") or UC.is_carrying(label):
        end = label
        if UC.is_carrying(label):
            end = "U_worst"
            st, ses, rec = UC.load_carrying(label, reg, root)
            CARRYING.update(rec)
        elif os.path.exists(os.path.join(root, "constructed", f"{label}.json")):
            _, st, ses = reg.load_bracket(label)
        else:
            _, st, ses = reg.store_bracket(label)
        site = SH.nominal_site(n_captures)
        site["label"] = label
        site["ambient_C"] = ses.ambient_C
        if shelf_dz is not None:
            site = MU.shelf_site(site, shelf_dz)
        u = SH.build_unit(st, ses, site, noise_key=urng.digest(reg.key, f"noise_key/{end}", 0), cell=cell)
        return st, ses, site, u, lambda sl: urng.digest(reg.key, f"jobs/{end}/slot{sl}", 0), reg.registry_id
    st, sess = reg.load_unit(label)
    site = SH.draw_site(reg.key, label, n_captures, ambient_C=sess[0].ambient_C)
    if shelf_dz is not None:
        site = MU.shelf_site(site, shelf_dz)
    u = reg.build(label, world=SH.make_world(site, cell=cell), policy="stage05", truth_profile=SH.trace_fields(cell))
    return st, sess[0], site, u, lambda sl: urng.digest(reg.key, f"jobs/{label}/slot{sl}", 0), reg.registry_id


def _gen(sl, tray, ent, kin, cm_nom, true_check, swivel):
    if swivel is None:
        return SH.generate_job(sl["job"], ent(sl["slot"]), kin, tray, cm_nom, label=sl["label"], contact_every=5,
                               true_check=true_check)
    return SH.generate_job(sl["job"], ent(sl["slot"]), kin, tray, cm_nom, label=sl["label"], contact_every=5,
                           true_check=true_check, swivel=swivel)


def _torque_only(rej):
    if not rej or rej.get("on") != "true unit":
        return False
    r = rej["reason"]
    return (r["torque_share"] > 1.0 and r["margin_min_deg"] >= SH.K.GENERATOR_STOP_MARGIN_DEG and r["fk_err_mm"] <= 0.2
            and r["fk_err_deg"] <= 0.05 and r["speed_share"] <= SH.K.GENERATOR_SPEED_SHARE_MAX
            and not rej.get("contacts") and r["tray_to_S_min_m"] >= SH.K.GENERATOR_TRAY_TO_S_MIN_M)


def make_jobs(slots, ent, kin, cm_nom, true_check, swivel=None, left_out=None, take_up=None):
    jobs = []
    for sl in slots:
        tray = {"mass_kg": sl["mass_kg"], "com": (sl["fore_aft_m"], sl["along_m"], -sl["below_m"])}
        j = _gen(sl, tray, ent, kin, cm_nom, true_check, swivel)
        if (j["targets"] is None and sl["label"] == "decline" and true_check is not None and len(true_check) > 5
                and true_check[3] == "command" and left_out is not None and _torque_only(j["rejections"][-1])):
            j0 = _gen(sl, tray, ent, kin, cm_nom, None, swivel)
            basis = SH.TorqueBasis(true_check[0], j0["targets"], swivel=swivel)
            first = {"mass_kg": sl["mass_kg"], "along_m": sl["along_m"], "fore_aft_m": sl["fore_aft_m"],
                     "below_m": sl["below_m"]}
            tu = SH.take_up_decline_tray(ent(sl["slot"]), basis, true_check[2], true_check[4], true_check[5],
                                         first=first)
            if take_up is not None:
                take_up[str(sl["slot"])] = {"tried": tu["tried"], "tray": tu["tray"], "reason": tu["reason"],
                                            "first_share": tu["shares"][0]["share"],
                                            "best_share": min(x["share"] for x in tu["shares"])}
            if tu["tray"] is None:
                left_out.append({"slot": sl["slot"], "name": sl.get("name"), "reason": tu["reason"]})
                jobs.append(None)
                continue
            d = tu["tray"]
            sl.update(mass_kg=d["mass_kg"], along_m=d["along_m"], fore_aft_m=d["fore_aft_m"], below_m=d["below_m"])
            sl["decline_tray"] = d
            tray = {"mass_kg": sl["mass_kg"], "com": (sl["fore_aft_m"], sl["along_m"], -sl["below_m"])}
            j = _gen(sl, tray, ent, kin, cm_nom, true_check, swivel)
        if j["targets"] is None:
            raise SystemExit(f"slot {sl['slot']}: no job passed: {j['rejections'][:2]}")
        jobs.append(j)
    return jobs


def candidate_checks(label, registry, slots="B", cell=None, swivel=None, st_ses_site_u=None):
    sl_list = parse_slots(slots) if isinstance(slots, str) else [dict(x) for x in slots]
    if st_ses_site_u is None:
        st, ses, site, u, ent, reg_id = load(label, len(sl_list), registry, None, cell)
    else:
        st, ses, site, u, ent, reg_id = st_ses_site_u
    out = {"unit": label, "registry": None if label == "nominal" else {"name": registry, "registry_id": reg_id},
           "slots": [x["slot"] for x in sl_list], "decline_draws_max": SH.DECLINE_DRAWS_MAX}
    cn = SH.capture_note_check(st, site["F"]["true"])
    out["capture_note"] = {"ok": bool(cn["ok"]), "reason": cn["reason"], "worst_axis_mm": cn["worst_axis_mm"],
                           "worst_angle_deg": cn["worst_angle_deg"], "limits": cn["limits"],
                           "per_swivel": cn["per_swivel"], "F_true": site["F"]["true"]}
    kin = SH.Kin()
    cm_nom = SH.nominal_cell_model(cell)
    true_check = (SH.true_kin(st), SH.CellModel(u), tuple(st.cal.k), "command", SH.true_offsets(st, ses),
                  SH.module_clamps(u))
    rows = []
    for sl in sl_list:
        if sl["label"] != "decline":
            continue
        tray = {"mass_kg": sl["mass_kg"], "com": (sl["fore_aft_m"], sl["along_m"], -sl["below_m"])}
        row = {"slot": sl["slot"], "name": sl.get("name"), "job": sl["job"], "admitted": False, "how": None,
               "tried": 0, "first_share": None, "best_share": None, "tray": None, "reason": None}
        j = _gen(sl, tray, ent, kin, cm_nom, true_check, swivel)
        if j["targets"] is not None:
            sh = (j.get("check_true") or {}).get("torque_share")
            row.update(admitted=True, how="slot sheet", tried=1, first_share=sh, best_share=sh,
                       tray={k: sl[k] for k in ("mass_kg", "along_m", "fore_aft_m", "below_m")})
        elif j["rejections"] and _torque_only(j["rejections"][-1]):
            j0 = _gen(sl, tray, ent, kin, cm_nom, None, swivel)
            basis = SH.TorqueBasis(true_check[0], j0["targets"], swivel=swivel)
            first = {k: sl[k] for k in ("mass_kg", "along_m", "fore_aft_m", "below_m")}
            tu = SH.take_up_decline_tray(ent(sl["slot"]), basis, true_check[2], true_check[4], true_check[5],
                                         first=first)
            row.update(tried=tu["tried"], first_share=tu["shares"][0]["share"],
                       best_share=min(x["share"] for x in tu["shares"]), reason=tu["reason"])
            if tu["tray"] is not None:
                d = tu["tray"]
                sl2 = dict(sl, mass_kg=d["mass_kg"], along_m=d["along_m"], fore_aft_m=d["fore_aft_m"],
                           below_m=d["below_m"])
                tray2 = {"mass_kg": d["mass_kg"], "com": (d["fore_aft_m"], d["along_m"], -d["below_m"])}
                j2 = _gen(sl2, tray2, ent, kin, cm_nom, true_check, swivel)
                if j2["targets"] is not None:
                    row.update(admitted=True, how="slot sheet" if d["draw"] == "slot sheet" else f"draw {d['draw']}",
                               tray=d)
                else:
                    row.update(tray=d, reason="the taken tray's job fails the true unit's check",
                               rejection=j2["rejections"][-1] if j2["rejections"] else None)
        else:
            row.update(reason="the decline slot's job fails a check other than the torques",
                       rejection=j["rejections"][-1] if j["rejections"] else None)
        rows.append(row)
    out["decline"] = rows
    reasons = []
    if not out["capture_note"]["ok"]:
        reasons.append(out["capture_note"]["reason"])
    for r in rows:
        if not r["admitted"]:
            reasons.append(f"slot {r['slot']}: {r['reason']}")
    out["ok"] = not reasons
    out["reasons"] = reasons
    return out


class ModuleTrace:
    FIELDS = ("q", "dq", "tau", "status", "t_rotor")

    def __init__(self, ref):
        import numpy as np
        from openarm_unit import hostcodec as hc
        self.np, self.hc, self.ref = np, hc, ref
        self.cycles, self.rows, self.tw, self.tw_step = [], {k: [] for k in self.FIELDS}, [], []
        step0, on0 = ref.step, ref.on_trace

        def step(c, sig, replies):
            self._replies(c, replies)
            return step0(c, sig, replies)

        def on_trace(recs):
            if recs:
                self.tw.append(np.asarray(recs[-1]["T_w"], dtype=np.float64).copy())
                self.tw_step.append(int(recs[-1]["step"]))
            return on0(recs)
        ref.step = step
        ref.on_trace = on_trace

    def _replies(self, c, replies):
        np = self.np
        row = {k: np.full(16, np.nan) for k in self.FIELDS}
        for f in replies or ():
            i = self.ref.reply_key.get((f.bus, f.can_id))
            if i is None:
                continue
            r = self.hc.unpack_state(f.data, self.ref.lim[i])
            if r is None:
                continue
            for k in self.FIELDS:
                row[k][i] = float(r[k])
        self.cycles.append(int(c))
        for k in self.FIELDS:
            self.rows[k].append(row[k])

    def save(self, path):
        np = self.np
        a = {"cycle": np.asarray(self.cycles, dtype=np.int64), "T_w": np.asarray(self.tw, dtype=np.float64),
             "T_w_step": np.asarray(self.tw_step, dtype=np.int64)}
        for k in self.FIELDS:
            a[k] = np.asarray(self.rows[k], dtype=np.float64).reshape(-1, 16)
        np.savez_compressed(path, **a)
        return {"path": path, "cycles": int(len(self.cycles)), "fields": list(a)}


def parse_slots(text):
    if text == "B":
        return SH.dev_slots(SH.SHIFT_B)
    toks = []
    for x in text.split(","):
        x = x.strip()
        toks.append(int(x) if x.lstrip("-").isdigit() else x)
    return SH.dev_slots(tuple(toks))


def _opt_value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


def main():
    argv = list(sys.argv[1:])
    cell = SH.CELL_DEFAULT
    if "--cell" in argv:
        k = argv.index("--cell")
        cell = argv[k + 1]
        del argv[k:k + 2]
    shelf_exception = "step"
    if "--shelf-exception" in argv:
        k = argv.index("--shelf-exception")
        shelf_exception = argv[k + 1]
        del argv[k:k + 2]
    decline_check = "command"
    if "--decline-check" in argv:
        k = argv.index("--decline-check")
        decline_check = argv[k + 1]
        del argv[k:k + 2]
    max_cycles = None
    if "--max-cycles" in argv:
        k = argv.index("--max-cycles")
        max_cycles = int(argv[k + 1])
        del argv[k:k + 2]
    zero_end = False
    if "--zero-end" in argv:
        argv.remove("--zero-end")
        zero_end = True

    def take(flag, n=1):
        if flag not in argv:
            return None
        k = argv.index(flag)
        v = argv[k + 1:k + 1 + n]
        del argv[k:k + 1 + n]
        return v[0] if n == 1 else v

    def flag(name):
        if name in argv:
            argv.remove(name)
            return True
        return False
    keepout = take("--keepout-at-removal") or "every"
    precheck_forms = flag("--precheck-forms")
    squeeze_from_belief = flag("--squeeze-from-belief")
    record_frames = take("--record-frames")
    replay = take("--replay")
    compose = take("--compose")
    skip_job = None
    if "--skip-job" in argv:
        k = argv.index("--skip-job")
        nxt = argv[k + 1] if k + 1 < len(argv) else ""
        if nxt and all(x.strip().isdigit() for x in nxt.split(",")):
            skip_job = [int(x) for x in nxt.split(",")]
            del argv[k:k + 2]
        else:
            skip_job = True
            del argv[k:k + 1]
    law_options = {}
    while "--law-option" in argv:
        k = argv.index("--law-option")
        kv = argv[k + 1]
        del argv[k:k + 2]
        if "=" not in kv:
            raise SystemExit(f"--law-option {kv!r}: <key>=<value>")
        a, b = kv.split("=", 1)
        law_options[a] = _opt_value(b)
    trace_path = registry = None
    ambient = False
    if "--trace" in argv:
        k = argv.index("--trace")
        trace_path = argv[k + 1]
        del argv[k:k + 2]
    registry = "registry"
    shelf_dz = None
    if "--shelf-dz" in argv:
        k = argv.index("--shelf-dz")
        shelf_dz = float(argv[k + 1])
        del argv[k:k + 2]
    if "--registry" in argv:
        k = argv.index("--registry")
        registry = argv[k + 1]
        del argv[k:k + 2]
    stop_at_event = True
    if "--continue-after-event" in argv:
        argv.remove("--continue-after-event")
        stop_at_event = False
    if "--ambient-at-presentation" in argv:
        argv.remove("--ambient-at-presentation")
        ambient = True
    carry_declines = False
    if "--carry-declines" in argv:
        argv.remove("--carry-declines")
        carry_declines = True
    swivel = None
    if "--swivel" in argv:
        k = argv.index("--swivel")
        swivel = float(argv[k + 1])
        del argv[k:k + 2]
    module_trace = None
    if "--module-trace" in argv:
        k = argv.index("--module-trace")
        module_trace = argv[k + 1]
        del argv[k:k + 2]
    label, slots_s, spare, out = argv[0], argv[1], argv[2] != "0", argv[3]
    mut = (argv[4], argv[5]) if len(argv) > 5 else None
    with open("/proc/loadavg") as f:
        la0 = f.read().split()[:3]
    t0 = time.perf_counter()
    c0 = time.process_time()
    slots = parse_slots(slots_s)
    st, ses, site, u, ent, reg_id = load(label, len(slots), registry, shelf_dz, cell)
    kin = SH.Kin()
    cm_nom = SH.nominal_cell_model(cell)
    if label == "nominal":
        true_check = None
    elif decline_check == "command":
        true_check = (SH.true_kin(st), SH.CellModel(u), tuple(st.cal.k), decline_check, SH.true_offsets(st, ses),
                      SH.module_clamps(u))
    else:
        true_check = (SH.true_kin(st), SH.CellModel(u), tuple(st.cal.k), decline_check)
    left_out, take_up = [], {}
    jobs = make_jobs(slots, ent, kin, cm_nom, true_check, swivel, left_out=left_out, take_up=take_up)
    if left_out:
        keep = [k for k, j in enumerate(jobs) if j is not None]
        slots = [slots[k] for k in keep]
        jobs = [jobs[k] for k in keep]
    t_gen = time.perf_counter() - t0
    B = MR.beliefs_from_truth(u, st, ses, site, slots, spare_coil=spare)
    opt = {}
    applied = None
    if compose is not None:
        specs = []
        for sp in json.loads(compose):
            kw = dict(sp[2]) if len(sp) > 2 and sp[2] else {}
            if "slot" in kw:
                kw["slots"] = slots
            specs.append((sp[0], sp[1], kw))
        B, opt, applied = MU.compose(B, specs)
    elif mut is not None:
        size, slot_one = mut[1], None
        if "@" in size:
            size, slot_one = size.split("@", 1)
            slot_one = int(slot_one)
        try:
            size = float(size)
        except ValueError:
            pass
        kw = {}
        if slot_one is not None:
            kw.update(slot=slot_one, slots=slots)
        if squeeze_from_belief:
            kw["squeeze_from_belief"] = True
        if precheck_forms and mut[0] == "M-geom-d":
            kw["precheck_form"] = True
        B, opt = MU.mutant(mut[0], B, size, **kw)
    if precheck_forms:
        opt = dict(opt)
        opt["decline_mode"] = "at_presentation"
        if mut is not None and mut[0] == "M-geom-d":
            opt["displacement_scope"] = "shift"
    if skip_job is not None:
        opt = dict(opt)
        opt["skip_job"] = skip_job
    if carry_declines:
        opt = dict(opt)
        opt["carry_declines"] = True
    if law_options:
        opt = dict(opt)
        opt.update(law_options)
    if replay is not None:
        ref = MU.Replay(replay, slots, jobs)
    else:
        ref = MR.MinimalReference(B, slots, jobs, options=opt)
    flog = None if record_frames is None else MU.FrameLog(ref)
    mtr = None if module_trace is None else ModuleTrace(ref)
    tr = QT.TraceStore()
    t1 = time.perf_counter()
    res = SH.run_shift(u, site, slots, ref, jobs=jobs, recorder=tr, stop_at_event=stop_at_event,
                       ambient_at_presentation=ses.ambient_C if ambient else None,
                       shelf_exception=shelf_exception, zero_end=zero_end, max_cycles=max_cycles,
                       keepout_at_removal=keepout)
    res["cell"] = cell
    if keepout != "every":
        res["keepout_at_removal"] = keepout
    if precheck_forms:
        res["precheck_forms"] = True
    if left_out:
        res["left_out"] = left_out
    if take_up:
        res["decline_take_up"] = take_up
    if replay is not None:
        res["replay"] = replay
    if applied is not None:
        res["composition"] = applied
    if flog is not None:
        res["frames"] = flog.save(record_frames)
    if max_cycles is not None:
        res["max_cycles"] = max_cycles
    if shelf_exception != "step":
        res["shelf_exception"] = shelf_exception
    if decline_check != "command":
        res["decline_check"] = decline_check
    if law_options:
        res["law_options"] = law_options
    res["plan_and_run_s"] = time.perf_counter() - t1
    res["generator_s"] = t_gen
    res["total_cpu_s"] = time.process_time() - c0
    res["total_wall_s"] = time.perf_counter() - t0
    with open("/proc/loadavg") as f:
        res["loadavg"] = {"before": la0, "after": f.read().split()[:3]}
    res["unit"] = label
    if UC.is_carrying(label):
        res["carrying"] = dict(CARRYING)
    res["registry"] = None if label == "nominal" else {"name": registry, "registry_id": reg_id}
    res["site"] = {k: site[k] for k in ("pad_friction", "F", "S", "ambient_C")}
    res["spare_coil"] = spare
    res["shelf_dz_m"] = shelf_dz
    res["stop_at_event"] = stop_at_event
    res["squeeze_allowance_N"] = MR.SQUEEZE_ALLOWANCE_N
    res["mutant"] = None if mut is None else {"name": mut[0], "size": mut[1], "options": {k: str(v) for k, v in
                                                                                         opt.items()}}
    res["jobs"] = [{"slot": sl["slot"], "n_targets": len(j["targets"]), "targets": j["targets"],
                    "record": j["record"], "rejections": j["rejections"], "check": j["check"],
                    "check_true": j["check_true"]} for sl, j in zip(slots, jobs)]
    if swivel is not None:
        res["swivel_deg"] = swivel
        for jj, j in zip(res["jobs"], jobs):
            jj["swivel_fixed"] = j["swivel_fixed"]
            jj["swivels"] = j["swivels"]
    if carry_declines:
        res["carry_declines"] = True
    if mtr is not None:
        res["module_trace"] = mtr.save(module_trace)
    res["slots_run"] = slots
    kin_true = SH.true_kin(st)
    res["plan_margins_true_deg"] = {} if replay is not None else MR.plan_margins_by_kind(ref, kin_true)
    run = QT.Run(res, tr, slots, jobs, site)
    res["quantities"] = run.summary()
    res["chain"] = run.chain_rows()
    if trace_path:
        tr.save(trace_path)
        res["trace"] = {"path": trace_path, "records": int(len(tr.arrays()["step"]))}
    with open(out, "w") as f:
        json.dump(res, f, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o), indent=1,
                  sort_keys=True)


if __name__ == "__main__":
    main()
