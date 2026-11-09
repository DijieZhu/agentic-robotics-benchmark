"""C1 inventory: 2 TP_label / (n_true + n_submitted), per work area when the areas are balanced (twin/regions.py).

TP_label counts the matched pairs (untyped Hungarian on the capped IoU at s*, capped IoU >= 0.25) whose submitted
label equals the true label. n_submitted counts every item listed in objects.json, excluded ones included. With
balanced areas, C1 = the mean over the areas of 2 TP_label(area) / (n_true(area) + n_submitted(area)), the pairs
counted in the area of their true item and the submitted items in the area regions.allocate gives them.
"""
from .regions import mean_over_areas


def score(pairs, true_solids, agent_solids, n_true, n_sub, areas=None):
    tp = 0
    rows = []
    tp_area = {}
    for i, j, c in pairs:
        T, S = true_solids[i], agent_solids[j]
        ok = bool(S.label_ok and S.label == T.label)
        tp += int(ok)
        r = (areas or {}).get("of_true", {}).get(T.name)
        if ok and r is not None:
            tp_area[r] = tp_area.get(r, 0) + 1
        rows.append({"true": T.name, "true_label": T.label, "agent": S.name, "agent_label": S.label,
                     "capped_iou": c, "label_ok": ok, "area": r})
    den = n_true + n_sub
    plain = 2.0 * tp / den if den > 0 else 0.0
    info = {"tp_label": tp, "n_pairs": len(pairs), "n_true": n_true, "n_submitted": n_sub, "pairs": rows,
            "unbalanced": plain}
    if not (areas and areas.get("balanced")):
        return plain, info
    per = {}
    for r in areas["areas"]:
        nt, ns = areas["n_true"][r], areas["n_sub"][r]
        per[r] = {"tp_label": tp_area.get(r, 0), "n_true": nt, "n_submitted": ns,
                  "score": (2.0 * tp_area.get(r, 0) / (nt + ns)) if (nt + ns) > 0 else None}
    info["per_area"] = per
    return mean_over_areas({r: v["score"] for r, v in per.items()}), info
