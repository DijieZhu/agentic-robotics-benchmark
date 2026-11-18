"""Criterion C4 `bom_buildability` (weight: grader_config.json `weights`): the bill of materials judged as a kit, by
the number of corrections a partner would have to make to the order before the robot can be built from it.
v1.1 of the scoring; the matcher is that of v1.0 with the four v1.1 points of match.py.

After the matcher (match.py: out-of-scope and neutral lines leave, equal lines merge, one-to-one assignment, an
unassigned line recognised as a part that another line already holds is folded into that line):

  q       = truth lines recognised with a wrong quantity (match.py compares the order the quantities make:
            robots x (struts_per_robot x qty_per_strut + qty_per_robot_extra); the same order written per robot
            instead of per strut is the same quantity)
  U_t(c)  = truth lines of category c that no line is recognised as
  U_a(c)  = lines of category c that take part and are recognised as no truth line
  corrections = q + sum over the categories c of max(U_t(c), U_a(c))
            (in one category, putting a wrong line right is one correction; a missing or a surplus line is one)
  C4      = max(0, 1 - corrections / N), N = grader_config.json c4.kit.zero_at_corrections (missing: a task error,
            no default); C4 = 0 when the freeze grades no BOM line.

Reported, never scored: the per-line F_beta of v1.0 (`c4_f_beta_score`: tier-weighted recall over the line scores of
c4.line_scores, precision = recognised lines / lines taking part, b = c4.f_beta) and `c4_kit_complete` (every graded
truth line recognised with the right quantity). This module carries no truth value, price or quantity.
"""
from . import match as M

__all__ = ["score", "evaluate"]


def _zero_at(cfg):
    """N of the kit rule: grader_config.json c4.kit.zero_at_corrections, a positive number (else a task error)."""
    kit = cfg["c4"].get("kit")
    if not isinstance(kit, dict) or "zero_at_corrections" not in kit:
        raise RuntimeError("grader_config.json lacks c4.kit.zero_at_corrections (a task error)")
    n = kit["zero_at_corrections"]
    if isinstance(n, bool) or not isinstance(n, (int, float)) or not n > 0:
        raise RuntimeError("grader_config.json c4.kit.zero_at_corrections = %r is not a positive number" % (n,))
    return float(n)


def evaluate(bom, gt, cfg, matched=None):
    """C4 with its parts, from a match (computed when not given)."""
    n_zero = _zero_at(cfg)
    mt = matched if matched is not None else M.match_bom(bom, gt, cfg)
    truth, merged = mt["truth"], mt["merged"]
    right = float(cfg["c4"]["line_scores"]["right_quantity"])
    # the per-line F_beta of v1.0: reported only
    wsum = sum(t["weight"] for t in truth)
    rec = sum(truth[j]["weight"] * s for j, s in sorted(mt["line_score"].items())) / wsum if wsum > 0 else 0.0
    prec = len(mt["pairs"]) / len(merged) if merged else 0.0
    b2 = float(cfg["c4"]["f_beta"]) ** 2
    f_beta = (1.0 + b2) * prec * rec / (b2 * prec + rec) if (prec > 0 and rec > 0) else 0.0
    # the kit
    n_right = sum(1 for s in mt["line_score"].values() if s == right)
    n_wrong_qty = sum(1 for s in mt["line_score"].values() if s != right)
    u_truth, u_lines = {}, {}
    for j, t in enumerate(truth):
        if j not in mt["line_score"]:
            u_truth[t["category"]] = u_truth.get(t["category"], 0) + 1
    for i, g in enumerate(merged):
        if i not in mt["pairs"]:
            u_lines[g["category"]] = u_lines.get(g["category"], 0) + 1
    per_cat = {c: max(u_truth.get(c, 0), u_lines.get(c, 0)) for c in set(u_truth) | set(u_lines)}
    corrections = n_wrong_qty + sum(per_cat.values())
    value = max(0.0, 1.0 - corrections / n_zero) if truth else 0.0
    kit = 1.0 if (truth and n_right == len(truth)) else 0.0
    n_neu = sum(1 for _, why in mt["excluded"] if why.startswith("neutral"))
    n_oos = sum(1 for _, why in mt["excluded"] if why == "out_of_scope")
    n_fold = sum(1 for _, why in mt["excluded"] if why == "folded")
    n_in = sum(len(g["line_ids"]) for g in merged)
    stats = {"c4_recall": rec, "c4_precision": prec, "c4_f_beta_score": f_beta, "c4_truth_lines": float(len(truth)),
             "c4_recognised_lines": float(len(mt["pairs"])), "c4_right_quantity_lines": float(n_right),
             "c4_merged_lines": float(len(merged)), "c4_lines_merged_away": float(n_in - len(merged)),
             "c4_lines_folded": float(n_fold), "c4_neutral_lines": float(n_neu), "c4_out_of_scope_lines": float(n_oos),
             "c4_kit_complete": kit, "c4_corrections": float(corrections),
             "c4_wrong_quantity_lines": float(n_wrong_qty),
             "c4_unrecognised_truth_lines": float(sum(u_truth.values())),
             "c4_unrecognised_lines": float(sum(u_lines.values()))}
    where = ", ".join("%s %d" % (c, per_cat[c]) for c in sorted(per_cat, key=str) if per_cat[c])
    diag = ("%d correction(s) of the order (zero credit at %g): %d recognised line(s) with a wrong quantity, %d truth "
            "line(s) no line is recognised as, %d line(s) recognised as no part%s; per-line F_beta %.4f (recall "
            "%.4f, precision %.4f), reported only; %d of %d graded truth lines right; %d folded, %d neutral and %d "
            "out-of-scope line(s)" % (corrections, n_zero, n_wrong_qty, sum(u_truth.values()), sum(u_lines.values()),
                                      (" (by category: %s)" % where) if where else "", f_beta, rec, prec, n_right,
                                      len(truth), n_fold, n_neu, n_oos))
    return value, {"diagnostic": diag, "stats": stats, "match": mt}


def score(docs, gt, cfg):
    """(C4 in [0, 1], details) for the submission documents {"bom": ..., ...}."""
    value, det = evaluate(docs.get("bom") or {}, gt, cfg)
    det = dict(det)
    det.pop("match", None)
    return value, det
