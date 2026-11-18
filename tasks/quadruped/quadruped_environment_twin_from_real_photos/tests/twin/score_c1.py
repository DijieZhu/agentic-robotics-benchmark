"""C1 inventory: C1_zone = 2 TP_label / (N_true + n_submitted); C1 = the mean over the zones that have
a true or a submitted unit.

TP_label counts the matched pairs (F >= matching.f_min) whose submitted label is the true label or one of its
accepted alternatives (and is in the vocabulary) AND whose shape is recognisable: the shape
measure of C2's b_shape (F_registered, the F-score at matching.tau_m after the registration of b_pose, when
C2.shape_on is "F_registered"; the matching's F otherwise) reaches C2.shape_band[0]. Matching itself stays tolerant
(F at tau_match_m), so a rough box standing in for a unit is matched (it takes the true unit out of the unmatched
ones and counts in n_submitted) but is no true positive. A pair without a shape measure (the q stage failed) is no
true positive. N_true counts the true units that are not 'ignore'; n_submitted the submitted units the zone
allocation counts there (excluded units included; units matched to an 'ignore' entry and units outside the region
left out).
"""
from .zones import mean_over_zones


def shape_measure(ctx, true_id, pair):
    """The F that C2's b_shape uses for this pair, or None when the q stage gave none."""
    sec = ctx.cfg["C2"]
    if sec.get("shape_on", "F_registered") != "F_registered":
        return float(pair["F"])
    row = (ctx.q_detail or {}).get(true_id) or {}
    v = row.get("F_registered")
    return None if v is None else float(v)


def score(ctx):
    Z, M, truth = ctx.zones, ctx.matching, ctx.truth
    labels = {u["body"]: (u.get("label"), bool(u.get("label_ok"))) for u in ctx.twin.units}
    f_min = float(ctx.cfg["C2"]["shape_band"][0])
    tp = {z: 0 for z in Z.ids}
    rows = []
    n_label_only = 0
    for p in M.pairs:
        if p["ignore"]:
            continue
        tu = truth.unit[p["true"]]
        lab, ok = labels.get(p["agent"], (None, False))
        label_ok = bool(ok and lab in tu.labels())
        f_shape = shape_measure(ctx, tu.id, p)
        shape_ok = f_shape is not None and f_shape >= f_min
        good = label_ok and shape_ok
        n_label_only += int(label_ok and not shape_ok)
        z = Z.of_true[tu.id]
        tp[z] += int(good)
        rows.append({"true": tu.id, "agent": p["agent"], "true_label": tu.label, "agent_label": lab,
                     "label_ok": label_ok, "F": p["F"], "F_shape": f_shape, "shape_ok": shape_ok, "tp": good,
                     "zone": z})
    per = {}
    for z in Z.ids:
        nt, ns = Z.n_true[z], Z.n_sub[z]
        per[z] = {"tp_label": tp[z], "n_true": nt, "n_submitted": ns,
                  "score": (2.0 * tp[z] / (nt + ns)) if (nt + ns) > 0 else None}
    value = mean_over_zones({z: v["score"] for z, v in per.items()})
    out = {"value": value, "detail": {"per_zone": per, "pairs": rows, "tp_label": sum(tp.values()),
                                      "n_true": sum(Z.n_true.values()), "n_submitted": sum(Z.n_sub.values()),
                                      "shape_min": f_min, "n_label_right_shape_not": n_label_only}}
    if n_label_only:
        out["diagnostic"] = ("%d matched unit(s) with the right label are no true positive: their shape is not "
                             "recognisable (F_reg below %g)" % (n_label_only, f_min))
    return out
