import copy
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
for _p in (os.path.join(TASK_DIR, "ref"), os.path.join(TASK_DIR, "evidence", "shift_runner")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import shift as SH
import openarm_unit
import numpy as np
from openarm_unit import law as L
from openarm_unit import rng as urng
from openarm_unit.registry import Registry, RegistryError, UnitStale
from openarm_unit.state import HostProfile, canonical_json
from openarm_unit.unit import UnitError, UnitInvalid

K = SH.K
PROFILE_PATH = K.PROFILE_PATH
CELL = SH.CELL_DEFAULT
FORMAT = "t05/hidden/draw/1"
MAX_CANDIDATES = 8
N_UNITS = K.N_UNITS
DECLINE_COUNTS = tuple(int(x) for x in K.PER_UNIT["decline"])
N_CARRY_SLOTS = int(K.PER_UNIT["short"]) + int(K.PER_UNIT["medium"]) + int(K.PER_UNIT["lap_band"])
N_SLOTS_MAX = N_CARRY_SLOTS + max(DECLINE_COUNTS)
GRIPPER_MODES = tuple(K.GRIPPER_MODES)
GRIPPER_MODE_PAIRS = tuple((a, b) for a in GRIPPER_MODES for b in GRIPPER_MODES)
REASON_CAPTURE_NOTE = "fails the capture note"
REASON_NO_DECLINE_TRAY = SH.NO_DECLINE_TRAY
STEP_DRAW, STEP_SANITY, STEP_HOLD, STEP_SHIFT, STEP_CAPTURE = ("draw", "sanity", "hand-over hold",
                                                               "shift", "capture note")

GENERATOR_FILES = ("ref/shift.py", "ref/minimal.py", "ref/quantities.py", "evidence/shift_runner/run_dev_shift.py",
                   "contract/constants.py", "hidden/draw.py")
GENERATOR_GLOBS = ("duty",)
DUTY_PINNED = ("generator.py", "sheet.py", "label.py", "tells.py", "dev.py")
STANDIN_GENERATOR = ("stand-in: the instrument's own generator (ref/shift.generate_job, TorqueBasis, "
                     "take_up_decline_tray through evidence/shift_runner/run_dev_shift._gen / _torque_only) on development "
                     "shift B; NOT the duty generator")
STANDIN_SLOTS = {1: SH.SHIFT_B, 2: ("B1", "c3", "B2", "B3", "B4", "B5", "B6")}


def duty_generator():
    d = os.path.join(TASK_DIR, "duty")
    if not os.path.isfile(os.path.join(d, "generator.py")):
        return None
    if d not in sys.path:
        sys.path.insert(0, d)
    import generator as GEN
    return GEN if hasattr(GEN, "generate_shift") else None


def default_generator():
    return "duty" if duty_generator() is not None else "standin"


class GeneratorStale(RegistryError):

    def __init__(self, msg, differs=()):
        super().__init__(msg)
        self.differs = tuple(differs)


def check_environment():
    env = openarm_unit.ENVIRONMENT
    if not env["openblas_in_effect"] or env["numpy_loaded_before"]:
        raise RuntimeError(f"not the scored environment: openblas_in_effect {env['openblas_in_effect']}, "
                           f"numpy_loaded_before {env['numpy_loaded_before']}")
    return {"openblas_in_effect": bool(env["openblas_in_effect"]),
            "numpy_loaded_before": bool(env["numpy_loaded_before"])}


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _sha_file(path):
    with open(path, "rb") as f:
        return _sha(f.read())


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x if np.isfinite(x) else repr(x)
    if o is None or isinstance(o, (int, str)):
        return o
    return str(o)


def digest_of(obj):
    return _sha(canonical_json(_plain(obj)).encode("ascii"))


def _write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="ascii") as f:
        f.write(canonical_json(_plain(obj)))
    os.replace(tmp, path)


def _read(path):
    with open(path, "r", encoding="ascii") as f:
        return json.loads(f.read())


def _loadavg():
    with open("/proc/loadavg") as f:
        return f.read().split()[:3]


def generator_digests(duty_pinned=False):
    files, info = {}, {}
    for rel in GENERATOR_FILES:
        files[rel] = _sha_file(os.path.join(TASK_DIR, rel))
    for d in GENERATOR_GLOBS:
        p = os.path.join(TASK_DIR, d)
        if os.path.isdir(p):
            for n in sorted(os.listdir(p)):
                if n.endswith(".py"):
                    pin = duty_pinned and n in DUTY_PINNED
                    (files if pin else info)[f"{d}/{n}"] = _sha_file(os.path.join(p, n))
    comb = _sha("\n".join(f"{k} {v}" for k, v in sorted(files.items())).encode("ascii"))
    return {"files": files, "information": info, "combined": comb, "duty_pinned": bool(duty_pinned)}


def decline_count_order(key):
    u = float(urng.uniform_in(key, "set/decline_count_order", 0, 0.0, float(N_UNITS)))
    two = min(int(u), N_UNITS - 1)
    lo, hi = min(DECLINE_COUNTS), max(DECLINE_COUNTS)
    return [hi if k == two else lo for k in range(N_UNITS)]


def _log(root, kind, **kw):
    p = os.path.join(root, "draw_log.json")
    lg = _read(p) if os.path.exists(p) else {"format": FORMAT + "/log", "steps": []}
    lg["steps"].append(_plain(dict(kind=kind, time=time.strftime("%Y-%m-%d %H:%M:%S %Z"), **kw)))
    _write(p, lg)


def create(root, entropy=None):
    check_environment()
    which = default_generator()
    reg = Registry.create(root, PROFILE_PATH, entropy=entropy)
    _log(root, "registry", registry_json_sha256=_sha_file(os.path.join(root, "registry.json")),
         registry_id=reg.registry_id, entropy_source=reg.header["entropy_source"],
         digests=reg.header["digests"], registry_log_steps=len(reg.log["steps"]))
    GEN = duty_generator() if which == "duty" else None
    if GEN is not None:
        order = {"decline_count_order": [int(x) for x in GEN.decline_counts(reg.key)],
                 "stream": "duty generator: decline_counts(key) (stream 'duty/decline_counts')"}
    else:
        order = {"decline_count_order": decline_count_order(reg.key),
                 "stream": "set/decline_count_order (rng.uniform_in of the registry's key, counter 0)"}
    gen = {"format": FORMAT + "/generator", "registry_id": reg.registry_id, "shift_generator": which,
           "code": generator_digests(duty_pinned=which == "duty"),
           "duty_code_digest": None if GEN is None else GEN.code_digest(),
           "duty_generator_version": None if GEN is None else getattr(GEN, "GENERATOR_VERSION", None),
           "profile_sha256": reg.header["digests"]["profile_sha256"], "cell": CELL,
           "candidate_indices": MAX_CANDIDATES, "units_wanted": N_UNITS, "set": order,
           "written_before_any_draw": len(reg.log["steps"]) == 0}
    if not gen["written_before_any_draw"]:
        raise RegistryError("generator.json must be written before any draw")
    _write(os.path.join(root, "generator.json"), gen)
    _log(root, "generator", generator_json_sha256=_sha_file(os.path.join(root, "generator.json")),
         combined=gen["code"]["combined"], registry_log_steps=len(reg.log["steps"]))
    return reg


def open_registry(root):
    return Registry.open(root, PROFILE_PATH)


def generator_stale(root):
    code = _read(os.path.join(root, "generator.json"))["code"]
    rec = code["files"]
    now = generator_digests(duty_pinned=code.get("duty_pinned", False))["files"]
    return sorted(k for k in set(rec) | set(now) if rec.get(k) != now.get(k))


def require_fresh(reg, root):
    d = reg.stale()
    if d:
        raise UnitStale("the registry's recorded digests differ from the tree's: " + ", ".join(d), d)
    g = generator_stale(root)
    if g:
        raise GeneratorStale("the task's generator code differs from generator.json: " + ", ".join(g), g)


def mode_pair_name(pair):
    return f"left {pair[0]}, right {pair[1]}"


def handover_hold(reg, label, site, modes=GRIPPER_MODE_PAIRS, hold_profile="A"):
    out = {"ok": True, "reasons": [], "modes": {}, "hold_s": K.HANDOVER_HOLD_S, "gain_set": hold_profile}
    th = np.zeros(16)
    th[3] = th[11] = K.HOME_POSE_RAD[3]
    th[7], th[15] = 0.45, -0.45
    for pair in modes:
        pair = (pair, pair) if isinstance(pair, str) else tuple(pair)
        mode = mode_pair_name(pair)
        r = {"left": pair[0], "right": pair[1], "faults": [], "contact_events": [], "contacts_at_end": [],
             "enabled_at_end": None, "error": None, "invalid": None}
        u = reg.build(label, world=SH.make_world(site, cell=CELL), policy="stage05",
                      truth_profile=SH.trace_fields(CELL))
        hp = HostProfile.default(hold_profile, gripper_mode="force_position" if all(m == "posforce" for m in pair)
                                 else "mit")
        ctrl = {SH.GRIPPER[s]: pair[i] for i, s in enumerate(SH.SIDES)}
        try:
            u.reset(bus_angles=th, hold=hp, hold_s=K.HANDOVER_HOLD_S, ctrl_mode=ctrl)
        except UnitInvalid as exc:
            r["invalid"] = {"step": exc.step, "kind": exc.kind}
            out["reasons"].append(f"gripper modes {mode}: the plant cannot integrate physics step {exc.step} "
                                  f"({exc.kind})")
            out["modes"][mode] = r
            continue
        except UnitError as exc:
            r["error"] = str(exc)
            out["reasons"].append(f"gripper modes {mode}: a fault in the hand-over hold: {exc}")
        ev = u.events()
        r["faults"] = [e.to_dict() for e in ev if e.kind == "fault_set"]
        r["contact_events"] = [e.to_dict() for e in ev if e.kind == "contact_begin"]
        u.truth_trace()
        if r["error"] is None:
            t = u.truth(("contacts", "enabled"))
            r["contacts_at_end"] = sorted(str(k) for k in dict(t["contacts"]))
            r["enabled_at_end"] = [bool(x) for x in t["enabled"]]
            if not all(r["enabled_at_end"]):
                out["reasons"].append(f"gripper modes {mode}: a module not enabled at the end of the hold")
        if r["faults"]:
            out["reasons"].append(f"gripper modes {mode}: {len(r['faults'])} fault(s) in the hand-over hold: "
                                  f"{r['faults'][:2]}")
        if r["contact_events"] or r["contacts_at_end"]:
            out["reasons"].append(f"gripper modes {mode}: a contact in the hand-over hold: "
                                  f"{(r['contact_events'] or r['contacts_at_end'])[:2]}")
        out["modes"][mode] = r
    out["ok"] = not out["reasons"]
    return out


def _slot_tray(sl):
    return {k: sl[k] for k in ("mass_kg", "along_m", "fore_aft_m", "below_m")}


def generate_shift_standin(ctx):
    import run_dev_shift as RD
    st, ses, unit, entropy = ctx["state"], ctx["session"], ctx["unit"], ctx["entropy"]
    n_decline, slots = ctx["n_decline"], ctx.get("slots")
    names = STANDIN_SLOTS[int(n_decline)] if slots is None else tuple(slots)
    sl_list = RD.parse_slots(",".join(str(x) for x in names))
    kin = SH.Kin()
    cm_nom = SH.nominal_cell_model(CELL)
    clamps = SH.module_clamps(unit)
    true_check = (SH.true_kin(st), SH.CellModel(unit), tuple(st.cal.k), "command", SH.true_offsets(st, ses), clamps)
    out = {"generator": STANDIN_GENERATOR, "standin": True, "n_decline": int(n_decline), "slots": [], "jobs": [],
           "decline": [], "rejection": None}
    for sl in sl_list:
        sl = dict(sl)
        tray = SH.tray_of_draw(_slot_tray(sl))
        j = RD._gen(sl, tray, entropy, kin, cm_nom, true_check, None)
        if j["targets"] is None and sl["label"] == "decline":
            last = j["rejections"][-1] if j["rejections"] else None
            row = {"slot": sl["slot"], "name": sl.get("name"), "tried": 0, "how": None, "tray": None,
                   "first_share": None, "best_share": None, "reason": None}
            if not RD._torque_only(last):
                row["reason"] = "the decline slot's job fails a check other than the torques"
                out["decline"].append(row)
                out["rejection"] = {"reason": row["reason"], "slot": sl["slot"], "detail": last}
                break
            j0 = RD._gen(sl, tray, entropy, kin, cm_nom, None, None)
            basis = SH.TorqueBasis(true_check[0], j0["targets"])
            tu = SH.take_up_decline_tray(entropy(sl["slot"]), basis, true_check[2], true_check[4], true_check[5],
                                         first=_slot_tray(sl))
            row.update(tried=tu["tried"], first_share=tu["shares"][0]["share"],
                       best_share=min(x["share"] for x in tu["shares"]), reason=tu["reason"])
            if tu["tray"] is None:
                out["decline"].append(row)
                out["rejection"] = {"reason": REASON_NO_DECLINE_TRAY, "slot": sl["slot"],
                                    "detail": {"tried": tu["tried"], "best_share": row["best_share"]}}
                break
            d = tu["tray"]
            row.update(tray=d, how="slot sheet" if d["draw"] == "slot sheet" else f"draw {d['draw']}")
            sl.update(_slot_tray(d))
            sl["decline_tray"] = d
            j = RD._gen(sl, SH.tray_of_draw(d), entropy, kin, cm_nom, true_check, None)
            if j["targets"] is None:
                row["reason"] = "the taken tray's job fails the true unit's check"
                out["decline"].append(row)
                out["rejection"] = {"reason": row["reason"], "slot": sl["slot"],
                                    "detail": j["rejections"][-1] if j["rejections"] else None}
                break
            out["decline"].append(row)
        elif sl["label"] == "decline":
            sh = (j.get("check_true") or {}).get("torque_share")
            out["decline"].append({"slot": sl["slot"], "name": sl.get("name"), "tried": 1, "how": "slot sheet",
                                   "tray": _slot_tray(sl), "first_share": sh, "best_share": sh, "reason": None})
        if j["targets"] is None:
            out["rejection"] = {"reason": f"no job of slot {sl.get('name', sl['slot'])} ({sl['job']}) passes the "
                                          "generator's checks on the nominal model and the true unit",
                                "slot": sl["slot"], "detail": j["rejections"][-1] if j["rejections"] else None}
            break
        out["slots"].append(sl)
        out["jobs"].append({"slot": sl["slot"], "targets": j["targets"], "record": j["record"],
                            "check": j["check"], "check_true": j["check_true"], "swivels": j.get("swivels"),
                            "rejections": j["rejections"]})
    return out


def generate_shift_duty(ctx):
    GEN = duty_generator()
    r = GEN.generate_shift(ctx["key"], ctx["label"], ctx["state"], ctx["session"], ctx["unit"], ctx["n_decline"],
                           cell=CELL)
    dec = []
    for sl in r["slots"]:
        if sl.get("label") == "decline":
            tu = sl.get("take_up") or {}
            dec.append({"slot": sl["slot"], "name": sl.get("name"), "tried": tu.get("tried"),
                        "how": None if tu.get("taken_draw") is None else f"draw {tu['taken_draw']}",
                        "tray": None if tu.get("taken_draw") is None else _slot_tray(sl) if "mass_kg" in sl else None,
                        "first_share": tu.get("first_share_true"), "best_share": tu.get("best_share_true"),
                        "reason": tu.get("reason")})
    rej = None if r["ok"] else {"reason": r["reason"], "slot": (r.get("rejection_detail") or {}).get("slot"),
                                "detail": r.get("rejection_detail")}
    return {"generator": f"duty generator (duty/generator.py generate_shift, version {r.get('generator_version')})",
            "standin": False, "slots": r["slots"] if r["ok"] else r["slots"], "jobs": r["jobs"], "decline": dec,
            "rejection": rej, "sheet_digest": r.get("sheet_digest"), "shift_digest": r.get("shift_digest"),
            "order": r.get("order"), "duty_cpu_s": r.get("cpu_s")}


def generate_shift(ctx, which=None):
    which = which or ("standin" if ctx.get("slots") is not None else default_generator())
    return generate_shift_duty(ctx) if which == "duty" else generate_shift_standin(ctx)


def capture_note(st, site):
    cn = SH.capture_note_check(st, site["F"]["true"])
    worst = {"axis": None, "angle": None}
    for sw, row in cn["per_swivel"].items():
        for side, r in row.items():
            a = max(abs(x) for x in r["p_mm"])
            g = max(r["angle_deg"])
            if worst["axis"] is None or a > worst["axis"]["mm"]:
                worst["axis"] = {"mm": a, "swivel": sw, "side": side, "axis": "xyz"[int(np.argmax(np.abs(r["p_mm"])))]}
            if worst["angle"] is None or g > worst["angle"]["deg"]:
                worst["angle"] = {"deg": g, "swivel": sw, "side": side,
                                  "axis": "xyz"[int(np.argmax(r["angle_deg"]))]}
    return {"ok": bool(cn["ok"]), "reason": cn["reason"], "worst_axis_mm": cn["worst_axis_mm"],
            "worst_angle_deg": cn["worst_angle_deg"], "limits": cn["limits"], "per_swivel": cn["per_swivel"],
            "worst": worst}


def site_of(reg, label, ambient_C, n_captures=N_SLOTS_MAX):
    return SH.draw_site(reg.key, label, n_captures, ambient_C=ambient_C)


def _cut_site(site, n):
    s = copy.deepcopy(site)
    s["seats"] = {k: v[:n] for k, v in s["seats"].items()}
    return s


def candidate(root, place, reg=None, shift_generator=None, slots=None):
    check_environment()
    reg = open_registry(root) if reg is None else reg
    require_fresh(reg, root)
    gen = _read(os.path.join(root, "generator.json"))
    order = gen["set"]["decline_count_order"]
    n_decline = order[min(int(place), len(order) - 1)]
    which = "standin" if slots is not None else gen.get("shift_generator", "standin")
    gfun = (lambda c: generate_shift(c, which)) if shift_generator is None else shift_generator
    ui = int(reg.log["next_unit_index"])
    label = f"cand{ui:02d}"
    rec = {"format": FORMAT + "/candidate", "label": label, "unit_index": ui, "place": int(place),
           "n_decline": n_decline, "registry_id": reg.registry_id, "generator_combined": gen["code"]["combined"],
           "loadavg_before": _loadavg(), "cpu_s": {}, "outcome": None, "step": None, "reason": None}
    c = time.process_time()
    try:
        st, ses = reg.draw(label, sanity=True, max_units=1)
    except UnitStale:
        raise
    except RegistryError:
        last = reg.log["steps"][-1] if reg.log["steps"] else {}
        if last.get("kind") != "rejected" or last.get("unit_index") != ui:
            raise
        rec["cpu_s"]["draw_and_sanity"] = time.process_time() - c
        rec.update(outcome="rejected", step=STEP_SANITY if last.get("step") == "sanity" else STEP_DRAW,
                   reason=last.get("reason"))
        return _finish(root, rec)
    rec["cpu_s"]["draw_and_sanity"] = time.process_time() - c
    rec["unit_digest"] = st.digest
    rec["session_digests"] = {s.session_id: s.digest for s in ses}
    site = site_of(reg, label, ses[0].ambient_C)
    c = time.process_time()
    hold = handover_hold(reg, label, site)
    rec["cpu_s"]["handover_hold"] = time.process_time() - c
    rec["handover_hold"] = hold
    if not hold["ok"]:
        reg.reject(label, STEP_HOLD, "; ".join(hold["reasons"]))
        rec.update(outcome="rejected", step=STEP_HOLD, reason="; ".join(hold["reasons"]))
        return _finish(root, rec)
    c = time.process_time()
    u = reg.build(label, world=SH.make_world(site, cell=CELL), policy="stage05", truth_profile=SH.trace_fields(CELL))
    ent = lambda n: urng.digest(reg.key, f"jobs/{label}/slot{n}", 0)
    ctx = {"key": reg.key, "label": label, "state": st, "session": ses[0], "site": site, "unit": u, "entropy": ent,
           "n_decline": n_decline, "slots": slots}
    try:
        sh = gfun(ctx)
    except UnitInvalid as exc:
        sh = {"generator": "?", "standin": None, "slots": [], "jobs": [], "decline": [],
              "rejection": {"reason": f"the plant cannot integrate physics step {exc.step} ({exc.kind})"}}
    rec["cpu_s"]["shift"] = time.process_time() - c
    n_slots = len(sh["slots"]) if sh["rejection"] is None else N_SLOTS_MAX
    rec["site"] = _cut_site(site, n_slots)
    rec["shift"] = {"generator": sh["generator"], "standin": sh.get("standin"), "n_decline": n_decline,
                    "slots": sh["slots"], "jobs": sh["jobs"], "decline": sh["decline"], "rejection": sh["rejection"],
                    **{k: sh[k] for k in ("sheet_digest", "shift_digest", "order") if k in sh}}
    c = time.process_time()
    cn = capture_note(st, site)
    rec["cpu_s"]["capture_note"] = time.process_time() - c
    rec["capture_note"] = cn
    if sh["rejection"] is not None:
        reason = sh["rejection"]["reason"]
        reg.reject(label, STEP_SHIFT, reason)
        rec.update(outcome="rejected", step=STEP_SHIFT, reason=reason)
    elif not cn["ok"]:
        reg.reject(label, STEP_CAPTURE, cn["reason"])
        rec.update(outcome="rejected", step=STEP_CAPTURE, reason=cn["reason"])
    else:
        rec.update(outcome="admitted")
    return _finish(root, rec)


def _finish(root, rec):
    rec["loadavg_after"] = _loadavg()
    rec["cpu_s"]["total"] = sum(v for k, v in rec["cpu_s"].items() if k != "total")
    body = _plain(rec)
    body["digest"] = digest_of({k: v for k, v in body.items() if k not in ("cpu_s", "loadavg_before", "loadavg_after")})
    d = os.path.join(root, "candidates")
    os.makedirs(d, exist_ok=True)
    _write(os.path.join(d, f"{rec['label']}.json"), body)
    _log(root, "candidate", label=rec["label"], unit_index=rec["unit_index"], outcome=rec["outcome"],
         step=rec["step"], reason=rec["reason"], digest=body["digest"])
    return body


def draw_set(root, wanted=N_UNITS, max_candidates=MAX_CANDIDATES, accept=None, shift_generator=None, slots=None):
    reg = open_registry(root)
    recs, acc = [], []
    while len(acc) < wanted and int(reg.log["next_unit_index"]) < max_candidates:
        r = candidate(root, len(acc), reg=reg, shift_generator=shift_generator, slots=slots)
        recs.append(r)
        if r["outcome"] == "admitted" and (accept is None or accept(r)):
            acc.append(r["label"])
    out = {"records": [{k: r[k] for k in ("label", "unit_index", "place", "n_decline", "outcome", "step", "reason",
                                          "digest")} for r in recs], "accepted": acc, "complete": len(acc) >= wanted}
    _log(root, "set", accepted=acc, complete=out["complete"], next_unit_index=int(reg.log["next_unit_index"]))
    return out


def load_candidate(root, label):
    r = _read(os.path.join(root, "candidates", f"{label}.json"))
    want = r["digest"]
    if digest_of({k: v for k, v in r.items() if k not in ("digest", "cpu_s", "loadavg_before",
                                                           "loadavg_after")}) != want:
        raise RegistryError(f"candidate {label}: its digest does not match its content")
    return r


def _law_value(law_raw, path):
    v = law_raw
    for p in path:
        v = v[p]
    return v


def _interval(v):
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return [float(v), float(v)]
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return [float(v[0]), float(v[1])]
    return None


ELEMENT_LAW_PATH = {
    ("C4", "J1-J4"): "J1-J4", ("C4", "J5-J7"): "J5-J7"}


def profile_rows(profile_path=PROFILE_PATH, law_path=None):
    law_raw = json.load(open(law_path or L.PARAMS_LAW_PATH, encoding="utf-8"))
    prof = json.load(open(profile_path, encoding="utf-8"))
    rows = []
    for e in prof["entries"]:
        if e.get("mode") != "draw" or "range" not in e:
            continue
        lp = e.get("law_path")
        lv = _interval(_law_value(law_raw, lp)) if lp else None
        if lv is None and lp:
            raw = _law_value(law_raw, lp)
            if isinstance(raw, dict) and e["element"] in raw:
                lv = _interval(raw[e["element"]])
        pv = _interval(e["range"])
        rows.append({"quantity": e["quantity"], "element": e["element"], "mode": e["mode"], "law": lv, "profile": pv,
                     "narrows": lv is not None and pv is not None and lv[0] <= pv[0] <= pv[1] <= lv[1]})
    for t in prof["task_quantities"]:
        r = t["range"]
        if isinstance(r, dict) and "m" in r:
            for u_ in ("m", "deg"):
                rows.append({"quantity": t["quantity"], "element": f"per axis ({u_})", "mode": t["mode"],
                             "law": None, "profile": _interval(r[u_]), "narrows": True})
        elif isinstance(r, dict):
            for k_, v in r.items():
                if isinstance(v, dict):
                    for c_, vv in v.items():
                        rows.append({"quantity": t["quantity"], "element": f"{k_}.{c_}", "mode": t["mode"],
                                     "law": None, "profile": _interval(vv), "narrows": True})
                else:
                    rows.append({"quantity": t["quantity"], "element": k_, "mode": t["mode"], "law": None,
                                 "profile": _interval(v), "narrows": True})
        else:
            rows.append({"quantity": t["quantity"], "element": t["element"], "mode": t["mode"], "law": None,
                         "profile": _interval(r), "narrows": True})
    return rows


def unit_values(st, ses):
    p, c = st.plant, st.cal
    v = {}
    for j in range(7):
        v[("P1", f"J{j + 1}")] = [p.frictionloss[j], p.frictionloss[8 + j]]
    v[("HS1.period", "14 arm joints")] = list(st.hs.period)
    arm = [i for i in range(16) if i not in SH.GRIPPER.values()]
    v[("P2", "J1-J2")] = [p.damping[a * 8 + j] for a in (0, 1) for j in (0, 1)]
    v[("P2", "J3-J4")] = [p.damping[a * 8 + j] for a in (0, 1) for j in (2, 3)]
    v[("P2", "J5-J7")] = [p.damping[a * 8 + j] for a in (0, 1) for j in (4, 5, 6)]
    v[("P6", "left, right gripper")] = [p.frictionloss[g] for g in SH.GRIPPER.values()]
    v[("C1", "14 arm joints")] = [c.delta[i] / L.DEG for i in arm]
    v[("C3", "16 modules")] = list(c.k)
    v[("C4", "J1-J4")] = [c.b[a * 8 + j] for a in (0, 1) for j in range(4)]
    v[("C4", "J5-J7")] = [c.b[a * 8 + j] for a in (0, 1) for j in (4, 5, 6)]
    v[("TH1", "16 modules")] = list(st.thermal.h)
    v[("S3.ambient", "the unit's site")] = [ses.ambient_C]
    v[("S3.start_offset", "16 modules (both nodes of a module alike)")] = [t - ses.ambient_C for pair in
                                                                          ses.start_temperatures_C for t in pair]
    pos, quat = p.base_poses[1]
    from openarm_unit.state import public_params
    yr = float(public_params()["conventions"]["base_link_y_m"]["value"]["right"])
    v[("PL1.translation", "right arm (per axis)")] = [pos[0], pos[1] - yr, pos[2]]
    v[("PL1.rotation", "right arm (per axis)")] = [a / L.DEG for a in L._xyz_of_quat(quat)]
    from openarm_unit import registers as rg
    pub = public_params()
    regs = c.registers
    sp = []
    for i, m in enumerate(pub["conventions"]["modules"]["value"]):
        ed = st.editions[m["class"]]
        cat = rg.catalogue_registers(ed, m["command_id"], m["reply_id"], regs.get(i, rg.CTRL_MODE),
                                     regs.get(i, rg.TIMEOUT), vmax=regs.get(i, rg.VMAX))
        sp += [regs.get(i, rid) / cat[rid] - 1.0 for rid in (rg.FLUX, rg.RS, rg.LS)]
    v[("registers_identification_spread", "16 modules x (Flux, Rs, Ls)")] = sp
    return v


def seat_angles_of_quat(q):
    w, x, y, z = (float(c) for c in q)
    rx = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    ry = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    rz = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return rx, ry, rz


def site_values(site, shift=None):
    v = {("pad_friction", "per hand"): list(site["pad_friction"].values()),
         ("F_pose_offset", "per axis (m)"): list(site["F"]["offset_m"]),
         ("F_pose_offset", "per axis (deg)"): list(site["F"]["offset_deg"]),
         ("S_pose_offset", "per axis (m)"): list(site["S"]["offset_m"]),
         ("S_pose_offset", "per axis (deg)"): list(site["S"]["offset_deg"])}
    sm, sd = [], []
    for side in SH.SIDES:
        for t, q in site["seats"][side]:
            sm += list(t)
            sd += [a / L.DEG for a in seat_angles_of_quat(tuple(q))]
    v[("seat_offset", "per axis (m)")] = sm
    v[("seat_offset", "per axis (deg)")] = sd
    if shift is not None:
        tq = ("tray_mass_and_centre",)
        for sl in shift.get("slots") or []:
            if "mass_kg" not in sl:
                continue
            cls = sl.get("class") or ("decline" if sl.get("label") == "decline" else None)
            if cls is not None:
                v.setdefault(tq + (f"classes_kg.{cls}",), []).append(sl["mass_kg"])
            if sl.get("label") == "decline":
                v.setdefault(tq + ("decline_centre_along_abs_m",), []).append(abs(sl["along_m"]))
            v.setdefault(tq + ("centre_along_m",), []).append(sl["along_m"])
            v.setdefault(tq + ("centre_fore_aft_m",), []).append(sl["fore_aft_m"])
            v.setdefault(tq + ("centre_below_m",), []).append(sl["below_m"])
    return v


NOT_PRINTED_ROWS = {
    ("P1.arm_factor", "left, right"): "the per-arm friction factor: the printed friction range of a joint must contain "
                                      "the level of each arm, checked on the levels (rows P1 J1 to J7, both arms)",
    ("HS1.period", "14 arm joints"): "the period of the hold-scatter profile: structure of the law",
    ("registers_identification_spread", "16 modules x (Flux, Rs, Ls)"): "the registers' identification spread: each "
                                                                        "unit's values are in its delivery record",
}
FRICTION_LEVEL_ROWS = tuple(("P1", f"J{j}") for j in range(1, 8))


def ranges_report(units, profile_path=PROFILE_PATH, printed=None, tol=1e-9):
    fails = []
    skipped = None if printed is None else []
    try:
        prof_obj = L.load_profile(profile_path, strict=True)
    except L.LawError as exc:
        prof_obj = None
        fails.append({"problem": "the law refuses the profile", "detail": str(exc)[:400]})
    rows = profile_rows(profile_path)
    draws = {}
    cu = {}
    for label, st, ses, site, shift in units:
        try:
            cu[label] = {"ok": True, "checked": L.check_unit(st, [ses], prof_obj)}
        except L.LawError as exc:
            cu[label] = {"ok": False, "error": str(exc)}
            fails.append({"unit": label, "problem": "law.check_unit", "detail": str(exc)[:400]})
        vals = unit_values(st, ses)
        if site is not None:
            vals.update(site_values(site, shift))
        for k, xs in vals.items():
            draws.setdefault(k, []).extend((label, float(x)) for x in xs)
    out_rows = []
    for r in rows:
        k = (r["quantity"], r["element"])
        row = dict(r)
        ds = draws.get(k, [])
        row["draw"] = [min(x for _, x in ds), max(x for _, x in ds)] if ds else None
        row["n_values"] = len(ds)
        if r["law"] is not None and not r["narrows"]:
            fails.append({"row": k, "problem": "the profile widens the law", "law": r["law"], "profile": r["profile"]})
        for lab, x in ds:
            if not (r["profile"][0] - tol <= x <= r["profile"][1] + tol):
                fails.append({"row": k, "unit": lab, "problem": "a draw outside the profile", "value": x,
                              "profile": r["profile"]})
            if r["law"] is not None and not (r["law"][0] - tol <= x <= r["law"][1] + tol):
                fails.append({"row": k, "unit": lab, "problem": "a draw outside the law", "value": x, "law": r["law"]})
        if printed is not None:
            pr = printed.get(k)
            row["printed"] = None if pr is None else [float(pr[0]), float(pr[1])]
            if k in NOT_PRINTED_ROWS:
                row["printed_check"] = "skipped: " + NOT_PRINTED_ROWS[k]
                skipped.append({"row": k, "reason": NOT_PRINTED_ROWS[k]})
            elif pr is None:
                fails.append({"row": k, "problem": "no printed range"})
            else:
                lo, hi = float(pr[0]), float(pr[1])
                prof = r["profile"]
                row["printed_check"] = "printed >= profile >= draw"
                if not (lo - tol <= prof[0] and prof[1] <= hi + tol):
                    fails.append({"row": k, "problem": "the printed range does not contain the profile's",
                                  "printed": [lo, hi], "profile": prof})
                for lab, x in ds:
                    if not (lo - tol <= x <= hi + tol):
                        fails.append({"row": k, "unit": lab, "problem": "a draw outside the printed range",
                                      "value": x, "printed": [lo, hi]})
                if k in FRICTION_LEVEL_ROWS:
                    row["printed_check"] += " (the friction levels of both arms)"
        out_rows.append(row)
    return {"ok": not fails, "rows": out_rows, "failures": fails, "check_unit": cu, "printed_skipped": skipped}


def registry_units(root):
    reg = open_registry(root)
    out = []
    d = os.path.join(root, "candidates")
    for n in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        r = load_candidate(root, n[:-5])
        if r.get("unit_digest") is None:
            continue
        lab = r["label"]
        u = reg.log["units"].get(lab) or next((v for k, v in reg.log["units"].items()
                                               if k.startswith(lab + "~rejected~")), None)
        from openarm_unit.state import SessionState, UnitState
        st = UnitState.load(os.path.join(root, "units", u["files"]["unit"]))
        ses = SessionState.load(os.path.join(root, "units", next(iter(u["files"]["sessions"].values()))))
        if st.digest != r["unit_digest"]:
            raise RegistryError(f"{lab}: the stored state is not the candidate's")
        out.append((lab, st, ses, r.get("site"), r.get("shift")))
    return out


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)

    def take(flag, default=None):
        if flag in a:
            k = a.index(flag)
            v = a[k + 1]
            del a[k:k + 2]
            return v
        return default
    ent = take("--entropy-hex")
    out = take("--out")
    printed_path = take("--printed")
    mx = int(take("--max-candidates", MAX_CANDIDATES))
    wanted = int(take("--wanted", N_UNITS))
    slots = take("--slots")
    cmd, root = a[0], os.path.abspath(a[1])
    t0, c0, la0 = time.perf_counter(), time.process_time(), _loadavg()
    if cmd == "create":
        reg = create(root, None if ent is None else bytes.fromhex(ent))
        res = {"registry_id": reg.registry_id}
    elif cmd == "draw":
        res = draw_set(root, wanted=wanted, max_candidates=mx, slots=None if slots is None else slots.split(","))
    elif cmd == "ranges":
        printed = None
        if printed_path:
            printed = {(e["quantity"], e["element"]): list(e["printed"]) for e in _read(os.path.abspath(printed_path))}
        rep = ranges_report(registry_units(root), printed=printed)
        res = {"ok": rep["ok"], "failures": rep["failures"], "rows": rep["rows"], "check_unit": rep["check_unit"],
               "printed_skipped": rep["printed_skipped"]}
    else:
        raise SystemExit(f"unknown command {cmd!r}")
    res["wall_s"] = time.perf_counter() - t0
    res["cpu_s"] = time.process_time() - c0
    res["loadavg"] = {"before": la0, "after": _loadavg()}
    res["environment"] = check_environment()
    txt = json.dumps(_plain(res), indent=1, sort_keys=True)
    if out:
        with open(out, "w") as f:
            f.write(txt)
    print(txt if len(txt) < 4000 else txt[:4000])


if __name__ == "__main__":
    main()
