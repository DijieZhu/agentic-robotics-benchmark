import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common as C

SH, QT, K = C.SH, C.QT, C.K
FORMAT = "stage05-frozen-set/1"
KEYS = ("e_z_mm", "tilt_deg", "rho_mm", "e_xy_mm", "d_m_mm", "P_r_N")
GRADER_FILES = ("common.py", "gtrace.py", "units.py", "events.py", "score.py", "frozen.py", "published.py", "verify.py")


class SetRefused(RuntimeError):
    ""


def captured_slots(run):
    out = []
    for i, sl in enumerate(run.slots):
        if sl["label"] != "carry" or i not in run.pres:
            continue
        p = run.pres[i]
        if p.get("missed_capture") or p.get("decline_cycle") is not None:
            continue
        arms = {e["fields"].get("arm") for e in run._events_of("grasp_closed", i)}
        if all(s in arms for s in SH.SIDES):
            out.append(i)
    return out


def rows_of_run(run):
    out = {k: [] for k in KEYS}
    for i in captured_slots(run):
        r = run.quantity_rows(i)
        for k in KEYS:
            out[k] += [float(x) for x in r[k]]
    return out


def make_decision(ref_rows_by_unit, mutant_rows_by_name_unit, floors=None):
    fl = C.floor_table() if floors is None else floors
    ref = {k: {u: list(v.get(k, [])) for u, v in ref_rows_by_unit.items()} for k in KEYS}
    mut = {}
    for k in KEYS:
        name, size = C.ZERO_END_MUTANT[k]
        rows = mutant_rows_by_name_unit.get(f"{name}/{size}", {})
        mut[k] = {u: list(v.get(k, [])) for u, v in rows.items()}
    d = QT.decide({k: fl[k]["floor"] for k in KEYS}, ref, mut)
    bad = C.full_end_mismatches(d)
    if bad:
        raise C.RuleFileMismatch("the full ends do not follow the rule file's factor: " + "; ".join(bad))
    for k in KEYS:
        d["quantities"][k]["mutant"] = "/".join(str(x) for x in C.ZERO_END_MUTANT[k])
    return C.plain(d)


def grader_code_digests():
    return {f: C.sha256_file(os.path.join(HERE, f)) for f in GRADER_FILES if os.path.isfile(os.path.join(HERE, f))}


def make_set(kind, label, registry, units, ref_rows_by_unit, mutant_rows_by_name_unit, anchor=None, audit=None,
             recorded=None, floors=None, by="grader/frozen.py"):
    if kind not in ("development", "of_record"):
        raise ValueError("kind: 'development' or 'of_record'")
    fl = C.floor_table() if floors is None else floors
    d = make_decision(ref_rows_by_unit, mutant_rows_by_name_unit, fl)
    rows = {"reference": {k: {u: len(v.get(k, [])) for u, v in ref_rows_by_unit.items()} for k in KEYS}}
    for m, byu in mutant_rows_by_name_unit.items():
        rows[m] = {k: {u: len(v.get(k, [])) for u, v in byu.items()} for k in KEYS}
    fs = {"format": FORMAT, "kind": kind, "label": label, "rule": C.check_rule_files(), "registry": registry,
          "units": units, "floors": fl, "rows": rows, "decision": d,
          "audit": audit or {"S_height_shown": True, "source": "not yet audited: taken as shown"},
          "anchor": anchor, "recorded": recorded or {},
          "made": {"by": by, "utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                   "code": grader_code_digests()}}
    return seal(fs)


def seal(fs):
    body = {k: v for k, v in fs.items() if k != "digest"}
    body = C.plain(body)
    body["digest"] = C.digest_of({k: v for k, v in body.items() if k != "digest"})
    return body


def set_anchor(fs, raw, per_unit, source):
    out = dict(fs)
    out["anchor"] = {"raw": float(raw), "per_unit": dict(per_unit), "source": source}
    return seal(out)


def write(path, fs):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(fs, f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def load(path, registry_root=None, require_kind=None, check_generator=True):
    with open(path, encoding="utf-8") as f:
        fs = json.load(f)
    if fs.get("format") != FORMAT:
        raise SetRefused(f"format {fs.get('format')!r} is not {FORMAT!r}")
    want = C.digest_of({k: v for k, v in fs.items() if k != "digest"})
    if fs.get("digest") != want:
        raise SetRefused("the set's digest does not match its content")
    if require_kind is not None and fs.get("kind") != require_kind:
        raise SetRefused(f"the set is of kind {fs.get('kind')!r}, not {require_kind!r}")
    pinned = dict(C.RULE_FILES)
    pinned[C.AMENDMENT_FILE[0]] = C.AMENDMENT_FILE[1]
    if fs.get("rule") != pinned:
        raise SetRefused("the set was placed under other rule files than the grader's")
    bad = C.full_end_mismatches(fs["decision"])
    if bad:
        raise SetRefused("the set's full ends do not follow the rule file's factor: " + "; ".join(bad))
    if not fs.get("anchor") or not fs["anchor"].get("raw"):
        raise SetRefused("the set has no anchor")
    if len(fs.get("units", [])) != K.N_UNITS:
        raise SetRefused(f"the set has {len(fs.get('units', []))} units, not {K.N_UNITS}")
    if registry_root is not None:
        check_registry(fs, registry_root, check_generator=check_generator)
    return fs


def check_registry(fs, root, check_generator=True):
    import draw as DR
    gen = DR._read(os.path.join(root, "generator.json"))
    reg = fs["registry"]
    if C.sha256_file(os.path.join(root, "generator.json")) != reg["generator_json_sha256"]:
        raise SetRefused("generator.json differs from the one the set names")
    if gen["code"]["combined"] != reg["generator_combined"]:
        raise SetRefused("the registry's generator digest differs from the set's")
    if check_generator:
        stale = DR.generator_stale(root)
        if stale:
            raise SetRefused("the generator's code differs from the set's generator digest: " + ", ".join(stale))
    for u in fs["units"]:
        rec = DR.load_candidate(root, u["label"])
        if rec["digest"] != u["candidate_digest"]:
            raise SetRefused(f"unit {u['label']}: its candidate record is not the set's")
        if rec.get("generator_combined") != reg["generator_combined"]:
            raise SetRefused(f"unit {u['label']}: drawn with another generator digest")
        if rec.get("outcome") != "admitted":
            raise SetRefused(f"unit {u['label']}: not an admitted candidate")
    return True


def unit_entry(rec):
    sh = rec["shift"]
    return {"label": rec["label"], "unit_index": rec["unit_index"], "unit_digest": rec["unit_digest"],
            "candidate_digest": rec["digest"], "shift_digest": sh.get("shift_digest"),
            "sheet_digest": sh.get("sheet_digest"), "n_slots": len(sh["slots"]), "n_decline": sh["n_decline"]}


def registry_entry(root):
    import draw as DR
    gen = DR._read(os.path.join(root, "generator.json"))
    reg = DR._read(os.path.join(root, "registry.json"))
    return {"root_name": os.path.basename(os.path.normpath(root)), "registry_id": gen["registry_id"],
            "generator_json_sha256": C.sha256_file(os.path.join(root, "generator.json")),
            "generator_combined": gen["code"]["combined"], "duty_code_digest": gen.get("duty_code_digest"),
            "registry_json_sha256": C.sha256_file(os.path.join(root, "registry.json")),
            "entropy_source": reg.get("entropy_source") if isinstance(reg, dict) else None}
