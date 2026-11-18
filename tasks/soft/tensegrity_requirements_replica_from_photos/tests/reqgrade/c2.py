"""C2 requirements_sheet: requirements.yaml items against the reference sheet. No value of any truth is written here.

    value, det = score(docs, gt, cfg)      the interface verify.py calls; det = {"diagnostic", "stats", "parts", "detail"}
    out = grade(docs, truth, settings)     the same on load.truth_from / load.settings output

An entry matches a reference item when the quantity is the same, the comparator is of the same class ("=" and "==";
"<=" and "<"; ">=" and ">") and the bound, in the reference's unit, earns credit under the reference's band (exact
for integers and tokens). A match none of whose source ids is of a class the reference accepts (PH-<nnn> is of class
PH, every other id is its own class) keeps settings c2.source_class_factor of its credit. G2 has already made each
quantity and comparator class unique, so a reference item has at most one candidate.
Recall: the tier-weighted mean of the reference items' credit. Precision, over every submitted item: a match counts
its credit over one; an item whose quantity is W0 (its value is written in a given file) is left out when its value
is right in the wide sense (bands.within_zero_point; the comparator is not looked at) and counts as wrong otherwise;
every other item (a graded quantity with a wrong value or comparator, a quantity no given material fixes) counts as
wrong. Score = harmonic mean of precision and recall (F1), 0 when either is 0.
"""
import statistics

from . import bands as B
from . import load as L
from . import schema_check as SC


def source_class(source_id):
    return source_id.split("-")[0] if source_id.startswith("PH-") else source_id


def _value(item, unit_to, truth_value, units):
    bound = item.get("bound")
    if isinstance(truth_value, str):
        return bound if isinstance(bound, str) else None
    return B.convert(bound, item.get("unit"), unit_to, units)


def grade(docs, truth, settings):
    req = docs.get("requirements") if isinstance(docs.get("requirements"), dict) else {}
    items = [it for it in (req.get("items") or []) if isinstance(it, dict)]
    t, units, tw = truth["c2"], truth["units"], settings["c2"]["tier_weights"]
    factor = float(settings["c2"]["source_class_factor"])
    by_key = {}
    for i, it in enumerate(items):
        cc = SC.COMPARATOR_CLASS.get(it.get("comparator")) if isinstance(it.get("comparator"), str) else None
        key = (it.get("quantity"), cc)
        if isinstance(key[0], str) and cc is not None and key not in by_key:
            by_key[key] = i
    credit = {}          # item position -> credit of its match
    recall_pairs, ref_detail = [], []
    for ref in t["reference"]:
        i = by_key.get((ref["quantity"], ref["comparator_class"]))
        s, source_ok = 0.0, False
        if i is not None:
            it = items[i]
            val = _value(it, ref["unit"], ref["value"], units)
            s = 0.0 if val is None else B.score(val, ref["value"], ref["band"])
            ids = [x for x in (it.get("source_ids") or []) if isinstance(x, str)]
            source_ok = any(source_class(x) in ref["accepted"] for x in ids)
            if s > 0 and not source_ok:
                s *= factor
            if s > 0:
                credit[i] = s
        recall_pairs.append((float(tw[ref["tier"]]), s))
        ref_detail.append({"id": ref["id"], "quantity": ref["quantity"], "matched": i is not None and s > 0,
                           "source_class_ok": source_ok, "score": s})
    recall = B.weighted_mean(recall_pairs)
    num, den, item_detail = 0.0, 0, []
    tally = {"matched": 0, "w0_right": 0, "w0_wrong": 0, "graded_wrong": 0, "unsourced": 0}
    for i, it in enumerate(items):
        q = it.get("quantity")
        cls = t["classes"].get(q, "unsourced") if isinstance(q, str) else "unsourced"
        if i in credit:
            outcome = "matched"
            num += credit[i]
            den += 1
        elif cls == "w0":
            w = t["w0"][q]
            val = _value(it, w["unit"], w["value"], units)
            right = val is not None and B.within_zero_point(val, w["value"], w["band"])
            outcome = "w0_right" if right else "w0_wrong"
            den += 0 if right else 1
        elif cls == "graded":
            outcome = "graded_wrong"
            den += 1
        else:
            outcome = "unsourced"
            den += 1
        tally[outcome] += 1
        item_detail.append({"position": i, "id": it.get("id") if isinstance(it.get("id"), str) else None,
                            "quantity": q if isinstance(q, str) else None, "class": cls, "outcome": outcome,
                            "credit": credit.get(i, 0.0)})
    precision = num / den if den > 0 else 0.0
    value = statistics.harmonic_mean([precision, recall]) if precision > 0 and recall > 0 else 0.0
    parts = {"precision": precision, "recall": recall}
    stats = {"c2_precision": precision, "c2_recall": recall, "c2_items": float(len(items))}
    stats.update({"c2_items_" + k: float(v) for k, v in sorted(tally.items())})
    diag = "precision %.4f, recall %.4f; %d items: %s" % (
        precision, recall, len(items), ", ".join("%s %d" % (k, v) for k, v in sorted(tally.items())))
    return {"score": float(value), "parts": parts, "stats": stats,
            "detail": {"reference": ref_detail, "items": item_detail}, "diagnostic": diag}


def score(docs, gt, cfg):
    out = grade(docs, L.truth_from(gt, cfg), L.settings(cfg))
    return out["score"], {k: out[k] for k in ("diagnostic", "stats", "parts", "detail")}
