"""C3 derived_quantities: requirements.yaml derived entries against the closed-form truths. No truth value here.

    value, det = score(docs, gt, cfg)      the interface verify.py calls; det = {"diagnostic", "stats", "parts", "detail"}
    out = grade(docs, truth, settings)     the same on load.truth_from / load.settings output

Every graded derived quantity: the entry with that quantity (G2 has made it unique), its value converted to the
truth's unit, scored by the band of its ledger row (exact for integers; relative otherwise, with the tighter band
of the encoder counts as the row says); a missing entry, a unit of another dimension or a non-numeric value earns
nothing. Score = the tier-weighted mean.
"""
from . import bands as B
from . import load as L


def grade(docs, truth, settings):
    req = docs.get("requirements") if isinstance(docs.get("requirements"), dict) else {}
    entries = {}
    for d in req.get("derived") or []:
        if isinstance(d, dict) and isinstance(d.get("quantity"), str) and d["quantity"] not in entries:
            entries[d["quantity"]] = d
    tw, units = settings["c3"]["tier_weights"], truth["units"]
    pairs, detail = [], []
    for ref in truth["c3"]["derived"]:
        d = entries.get(ref["quantity"])
        val = None if d is None else B.convert(d.get("value"), d.get("unit"), ref["unit"], units)
        s = 0.0 if val is None else B.score(val, ref["value"], ref["band"])
        err = None if val is None else B.error(val, ref["value"], ref["band"])
        pairs.append((float(tw[ref["tier"]]), s))
        detail.append({"id": ref["id"], "quantity": ref["quantity"], "present": d is not None, "error": err,
                       "score": s})
    value = B.weighted_mean(pairs)
    full = sum(1 for x in detail if x["score"] >= 1.0)
    stats = {"c3_full": float(full), "c3_graded": float(len(detail)),
             "c3_present": float(sum(1 for x in detail if x["present"]))}
    diag = "%d of %d derived quantities at full credit, %d present" % (full, len(detail), stats["c3_present"])
    return {"score": value, "parts": {}, "stats": stats, "detail": {"derived": detail}, "diagnostic": diag}


def score(docs, gt, cfg):
    out = grade(docs, L.truth_from(gt, cfg), L.settings(cfg))
    return out["score"], {k: out[k] for k in ("diagnostic", "stats", "parts", "detail")}
