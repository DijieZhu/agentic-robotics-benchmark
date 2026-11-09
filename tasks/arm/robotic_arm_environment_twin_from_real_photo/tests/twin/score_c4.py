"""C4 static physics: w_static x static share + w_support x support F1, from the standard settle of the twin.

Inputs come from part2.settle_twin (the twin's settle with its items matched to held true items held like the true
twin) and the truth files:

- static share = sum over the true items that rest (the true twin's items that are not held) of
  q_i x [the agent item paired with i is static], divided by max(number of resting true items, number of submitted
  items not paired with a held true item). Static = the twin_sim rule (displacement within static.pos_mm and rotation
  within static.rot_deg, maxima over the settle) with the displacement converted to true units (x s*). Submitted
  items include the excluded ones (they never rest: they are deleted before the physics).
- support F1: the required edges of the support truth (an edge a -> b: a carries b) against the twin's edges mapped
  into true names (items through the pairs, an unpaired item under its own name, so it matches nothing; furniture
  and drawer supporters onto the true surface they touch). Edges that involve a held true item and edges that are
  ambiguous in the truth are ignored on both sides. A required edge the twin reproduces counts min(q_a, q_b)
  (furniture and the drawer have q = 1); every required edge and every one of the twin's edges counts 1 in the
  denominators, so F1 = 2 x sum over reproduced edges of min(q_a, q_b) / (n_required + n_twin_edges); 1 when both
  sets are empty.

Work areas (twin/regions.py): with balanced areas both terms are taken per area and averaged over the areas: the
static share of an area over its resting true items and the submitted items the allocation puts there (minus those
paired with held true items of the area), the support F1 of an area over the edges whose supported item lies in it
(a true item's area, an unpaired agent item's allocated area). An area with nothing to score on either side is left
out of the mean.
"""
from .regions import mean_over_areas


def static_share(truth2, q_true, t2a, agent_static, n_sub, areas=None):
    """(share, info). q_true {true item: q}; t2a {true item: agent body}; agent_static {agent body: bool} (held and
    absent bodies missing); n_sub: items listed in objects.json; areas: part 1's work areas (None: not balanced)."""
    rows = {}
    num = 0.0
    for i in truth2.static_items:
        j = t2a.get(i)
        st = bool(agent_static.get(j, False)) if j is not None else False
        q = float(q_true.get(i, 0.0)) if j is not None else 0.0
        num += q * (1.0 if st else 0.0)
        rows[i] = {"agent": j, "q": q, "static": st}
    n_held_paired = sum(1 for t in truth2.held if t in t2a)
    den = max(len(truth2.static_items), int(n_sub) - n_held_paired)
    share = num / den if den > 0 else 0.0
    info = {"per_true": rows, "numerator": num, "denominator": den, "n_rest_true": len(truth2.static_items),
            "n_submitted": int(n_sub), "n_paired_with_held": n_held_paired, "unbalanced": share}
    if not (areas and areas.get("balanced")):
        return share, info
    of = areas["of_true"]
    per = {}
    for r in areas["areas"]:
        items = [i for i in truth2.static_items if of.get(i) == r]
        num_r = sum(rows[i]["q"] * (1.0 if rows[i]["static"] else 0.0) for i in items)
        nh_r = sum(1 for t in truth2.held if t in t2a and of.get(t) == r)
        den_r = max(len(items), int(areas["n_sub"].get(r, 0)) - nh_r)
        per[r] = {"numerator": num_r, "denominator": den_r, "n_rest_true": len(items), "n_paired_with_held": nh_r,
                  "score": (num_r / den_r) if den_r > 0 else None}
    info["per_area"] = per
    return mean_over_areas({r: v["score"] for r, v in per.items()}), info


def support_f1(truth2, q_true, twin_edges, areas=None):
    """(F1, info). twin_edges: the twin's mapped edge set {(supporter, supported)} with held and ambiguous edges
    already removed (part2.map_support_edges); areas: part 1's work areas (None: not balanced)."""
    req = set(truth2.required)

    def q_node(n):
        if n.startswith("furniture:") or n == truth2.drawer_node:
            return 1.0
        return float(q_true.get(n, 0.0))

    tw_set = set(twin_edges)
    tp = sorted(req & tw_set)
    tp_w = sum(min(q_node(a), q_node(b)) for a, b in tp)
    n_req, n_tw = len(req), len(tw_set)
    f1 = 1.0 if (n_req + n_tw) == 0 else 2.0 * tp_w / (n_req + n_tw)
    info = {"n_required": n_req, "n_twin_edges": n_tw, "tp": [list(e) for e in tp], "tp_weight": tp_w,
            "missed": [list(e) for e in sorted(req - tw_set)],
            "extra": [list(e) for e in sorted(tw_set - req)],
            "precision_w": (tp_w / n_tw) if n_tw else None, "recall_w": (tp_w / n_req) if n_req else None,
            "unbalanced": f1}
    if not (areas and areas.get("balanced")):
        return f1, info

    def area_of(node):
        if node.startswith("agent:"):
            return areas["of_agent"].get(node[len("agent:"):])
        return areas["of_true"].get(node)

    per = {}
    for r in areas["areas"]:
        req_r = [e for e in req if area_of(e[1]) == r]
        # an edge whose supported item has no area (never expected) counts against every area
        tw_r = [e for e in tw_set if area_of(e[1]) in (r, None)]
        tp_r = [e for e in tp if area_of(e[1]) == r]
        w_r = sum(min(q_node(a), q_node(b)) for a, b in tp_r)
        n = len(req_r) + len(tw_r)
        per[r] = {"n_required": len(req_r), "n_twin_edges": len(tw_r), "tp_weight": w_r,
                  "score": (2.0 * w_r / n) if n else None}
    info["per_area"] = per
    if all(v["score"] is None for v in per.values()):
        return 1.0, info
    return mean_over_areas({r: v["score"] for r, v in per.items()}), info


def score(truth2, q_true, t2a, agent_static, n_sub, twin_edges, cfg, areas=None):
    sc = cfg["static_physics"]
    share, i1 = static_share(truth2, q_true, t2a, agent_static, n_sub, areas)
    f1, i2 = support_f1(truth2, q_true, twin_edges, areas)
    c4 = float(sc["w_static"]) * share + float(sc["w_support"]) * f1
    return c4, {"static_share": share, "support_f1": f1, "static": i1, "support": i2}
