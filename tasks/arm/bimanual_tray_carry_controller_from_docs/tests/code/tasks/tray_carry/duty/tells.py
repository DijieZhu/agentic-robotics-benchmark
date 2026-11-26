import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TASK_DIR = os.path.dirname(HERE)
for _p in (os.path.join(TASK_DIR, "ref"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import shift as SH
import generator as GEN
import sheet as SHEET

PLACE_FIELDS = ("tray", "planned_presentation_s")
VALUE_SCAN_DECIMALS = 4
CONTRACT_FIELDS = ("format_version", "cycle_s", "move_s", "hold_s", "lifted_point_above_seated_m",
                   "first_presentation_s")
CLASSIFIER_EXCESS_MAX = 0.02


def _lap_slots(shift):
    return [(i, sl["label"]) for i, sl in enumerate(shift["slots"]) if sl["job"] == "lap"]


def _leaves(obj, path=""):
    if isinstance(obj, dict):
        for k in sorted(obj):
            yield from _leaves(obj[k], f"{path}.{k}" if path else k)
    elif isinstance(obj, list) and obj and all(isinstance(x, (int, float)) for x in obj):
        yield path, json.dumps(obj)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{path}[{i}]")
    else:
        yield path, json.dumps(obj)


def audit_shift(shift):
    fail = []
    sh = shift["sheet"]
    try:
        SHEET.validate_sheet(sh)
    except ValueError as ex:
        fail.append(f"format: {ex}")
    slots = shift["slots"]
    n = len(slots)
    lap = _lap_slots(shift)
    if not any(lab == "carry" for _, lab in lap):
        fail.append("no carried lap slot")
    for i, lab in lap:
        if i in (0, n - 1):
            fail.append(f"a {'decline tray' if lab == 'decline' else 'carried lap'} is {'first' if i == 0 else 'last'}")
    for i, sl in enumerate(slots):
        if sl["class"] == "decline" and sl["job"] != "lap":
            fail.append(f"slot {i}: a decline tray on a {sl['job']} job")
    ss = sh["slots"] if isinstance(sh, dict) and "slots" in sh else []
    compared = 0
    if len(ss) == n:
        car = [i for i, lab in lap if lab == "carry"]
        dec = [i for i, lab in lap if lab == "decline"]
        for d in dec:
            for c in car:
                a = {k: v for k, v in ss[d].items() if k not in PLACE_FIELDS}
                b = {k: v for k, v in ss[c].items() if k not in PLACE_FIELDS}
                ka, kb = set(a), set(b)
                if ka != kb:
                    fail.append(f"slots {d} and {c}: fields {sorted(ka ^ kb)} in one only")
                for k in sorted(ka & kb):
                    compared += 1
                    if SHEET.canonical(a[k]) != SHEET.canonical(b[k]):
                        fail.append(f"slots {d} (decline) and {c} (carried lap): field {k!r} differs")
    else:
        fail.append("the sheet's slots do not match the shift's")
    hidden = set()
    for sl in slots:
        for q in ("mass_kg", "along_m", "fore_aft_m", "below_m"):
            if sl.get(q) is not None:
                hidden.add(json.dumps(float(sl[q])))
                hidden.add(json.dumps(round(float(sl[q]), VALUE_SCAN_DECIMALS)))
    hidden.discard(json.dumps(0.0))
    for path, v in _leaves(sh):
        if path in CONTRACT_FIELDS:
            continue
        if v in hidden and not path.endswith(".p") and "targets" not in path:
            fail.append(f"{path}: a tray's value {v}")
        if v.strip('"') in ("decline", "carry", "band", "proof"):
            fail.append(f"{path}: a label {v}")
    return {"ok": not fail, "failures": fail, "lap_slots": lap, "compared_fields": compared}


def sample_sheets(n, seed=b"t05 duty tell audit", order_fn=None):
    order_fn = GEN.slot_order if order_fn is None else order_fn
    lap, _ = GEN.lap_list()
    out = []
    cache = {}
    for s in range(n):
        key = hashlib.sha256(seed + s.to_bytes(4, "big")).digest()
        nd = 1 + (s % 2)
        rl = GEN.roles(nd)
        order = order_fn(key, f"sample{s}", rl)
        jobs, labels = [], []
        for i, (role, kind, cls) in enumerate(order):
            if kind == "lap":
                jobs.append((kind, lap))
                labels.append((i, "decline" if cls == "decline" else "carry"))
                continue
            ent = GEN.job_entropy(key, f"sample{s}", role)
            ck = (kind, ent)
            if ck not in cache:
                tg, rec = SH.short_or_medium_targets(kind, ent, 0)
                a = 1
                while tg is None:
                    tg, rec = SH.short_or_medium_targets(kind, ent, a)
                    a += 1
                cache[ck] = [list(t) for t in tg]
            jobs.append((kind, cache[ck]))
        out.append((SHEET.build_sheet(jobs), labels, len(order)))
    return out


def _features(slot, n_slots):
    f = {k: v for k, v in _leaves(slot)}
    f["@place"] = json.dumps(slot["tray"])
    f["@place_from_end"] = json.dumps(n_slots - 1 - slot["tray"])
    f["@whole_slot"] = hashlib.sha256(SHEET.canonical(slot)).hexdigest()
    f["@whole_slot_without_place"] = hashlib.sha256(SHEET.canonical(
        {k: v for k, v in slot.items() if k not in PLACE_FIELDS})).hexdigest()
    return f


def classifier_audit(samples):
    rows = {0: [], 1: []}
    for s, (sheet, labels, n_slots) in enumerate(samples):
        for i, lab in labels:
            rows[(s // 2) % 2].append((_features(sheet["slots"][i], n_slots), lab))
    names = sorted(set().union(*[set(f) for f, _ in rows[0] + rows[1]]))
    tr, te = rows[0], rows[1]
    maj_tr = max(("decline", "carry"), key=lambda c: sum(1 for _, l in tr if l == c))
    n_te = len(te)
    maj_rate = max(sum(1 for _, l in te if l == c) for c in ("decline", "carry")) / n_te
    acc = {}
    for f in names:
        cnt = {}
        for fs, lab in tr:
            v = fs.get(f, "<absent>")
            c = cnt.setdefault(v, {"decline": 0, "carry": 0})
            c[lab] += 1
        pred = {v: max(c, key=lambda k: (c[k], k == maj_tr)) for v, c in cnt.items()}
        ok = sum(1 for fs, lab in te if pred.get(fs.get(f, "<absent>"), maj_tr) == lab)
        acc[f] = ok / n_te
    worst = max(acc.items(), key=lambda kv: kv[1])
    return {"ok": worst[1] - maj_rate <= CLASSIFIER_EXCESS_MAX, "majority_rate": maj_rate, "n_test": n_te,
            "n_train": len(tr), "worst": worst, "excess_max": CLASSIFIER_EXCESS_MAX,
            "features": {k: acc[k] for k in sorted(acc, key=lambda k: -acc[k])[:12]}}


def first_last_audit(samples):
    bad = []
    dist = {"decline": {}, "carry": {}}
    for s, (sheet, labels, n) in enumerate(samples):
        for i, lab in labels:
            if i in (0, n - 1):
                bad.append((s, i, lab))
            dist[lab][f"{i}/{n}"] = dist[lab].get(f"{i}/{n}", 0) + 1
    return {"ok": not bad, "first_or_last": bad[:10], "n_first_or_last": len(bad), "place_counts": dist}
