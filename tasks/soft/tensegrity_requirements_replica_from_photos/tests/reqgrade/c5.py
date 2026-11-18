"""Criterion C5 `bom_procurement` (weight: grader_config.json `weights`, 0.30 in v1.1): the agent's order priced with the
authors' own cost function against the exact order. The algorithm and the band are those of v1.0, unchanged
in v1.1, which only feeds it the matcher's folded lines (match.py step 4).

Money is always the authors' function (cost.py, driven by gt/price_table.json) at the order size of
grader_config.json (`c4.robots` robots of `c4.struts_per_robot` struts). With the match of match.py:

  D = [ sum over the priced truth lines j of | cost_j(agent units of the line matched to j) - cost_j(exact units) |
        (a truth line nothing is matched to counts as bought 0)
      + sum over the merged lines that take part but are matched to no truth line, in a category that has priced
        truth lines, of cost_r*(that line's units), r* = the category's cheapest truth line in the exact order ]
      / sum over the priced truth lines of cost_j(exact units)
  C5 = 1 - clip((D - full_at) / (zero_at - full_at), 0, 1)

"Cheapest truth line of the category" is read as the truth line whose exact-order cost is the lowest (ties: the lower
workbook row), priced at the agent's quantity with that line's own pack size and tiers: the reading under which the
design arithmetic of the BOM table was made (`min(same) * q`). The priced lines are the
graded truth lines whose price-table row is in the table's `priced_scope`. No price or quantity is in this module.
"""
from . import cost as K
from . import match as M

__all__ = ["score", "evaluate"]


def evaluate(bom, gt, cfg, matched=None):
    c4 = cfg["c4"]
    c5 = cfg["c5"]
    robots, spr = int(c4["robots"]), int(c4["struts_per_robot"])
    book = K.PriceBook(gt["price_table"])
    mt = matched if matched is not None else M.match_bom(bom, gt, cfg)
    truth, merged = mt["truth"], mt["merged"]
    priced = [j for j, t in enumerate(truth) if book.priced(t["money_row"])]
    exact = {j: book.cost(truth[j]["money_row"], K.order_units(truth[j]["qty"][0], truth[j]["qty"][1], robots, spr))
             for j in priced}
    total = sum(exact[j] for j in priced)
    # self-check: the cost function reproduces the price table's own record of the authors' priced order
    recorded = gt["price_table"].get("priced_scope_total_%d_robots" % robots)
    own = book.order_total(robots, spr, scope_only=True)[1]
    if recorded is not None and abs(own - float(recorded)) > float(c5.get("priced_total_check_abs", 1e-6)):
        raise RuntimeError("C5 self-check: the cost function prices the authors' order at %.6f but the price table "
                           "records %.6f (a task error)" % (own, float(recorded)))
    by_truth = {j: i for i, j in mt["pairs"].items()}
    dev = 0.0
    for j in priced:
        i = by_truth.get(j)
        units = K.order_units(merged[i]["qty"][0], merged[i]["qty"][1], robots, spr) if i is not None else 0
        dev += abs(book.cost(truth[j]["money_row"], units) - exact[j])
    cheapest = {}
    for j in priced:
        cat = truth[j]["category"]
        key = (exact[j], int(truth[j]["money_row"]))
        if cat not in cheapest or key < cheapest[cat][0]:
            cheapest[cat] = (key, truth[j]["money_row"])
    over, n_over = 0.0, 0
    for i, g in enumerate(merged):
        if i in mt["pairs"] or g["category"] not in cheapest:
            continue
        over += book.cost(cheapest[g["category"]][1], K.order_units(g["qty"][0], g["qty"][1], robots, spr))
        n_over += 1
    D = (dev + over) / total if total > 0 else 1.0
    full, zero = float(c5["band"]["full_at"]), float(c5["band"]["zero_at"])
    value = 1.0 - min(max((D - full) / (zero - full), 0.0), 1.0)
    stats = {"c5_distance": D, "c5_matched_deviation_share": dev / total if total > 0 else 0.0,
             "c5_unmatched_lines_share": over / total if total > 0 else 0.0, "c5_unmatched_lines_priced": float(n_over)}
    diag = ("money distance D = %.4f of the exact order (recognised lines off by %.4f, %d unrecognised line(s) "
            "priced at %.4f); full credit at D <= %.3f, none at D >= %.3f"
            % (D, stats["c5_matched_deviation_share"], n_over, stats["c5_unmatched_lines_share"], full, zero))
    return value, {"diagnostic": diag, "stats": stats}


def score(docs, gt, cfg):
    """(C5 in [0, 1], details) for the submission documents {"bom": ..., ...}."""
    return evaluate(docs.get("bom") or {}, gt, cfg)
