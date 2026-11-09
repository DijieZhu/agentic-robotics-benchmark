"""C5 interaction: w_lift x lift + w_drawer x drawer, from the probes of the twin (part2.probe_twin), outcomes in
true units, against the probe truth.

Lift. One probe per true item that rests (held true items are never lifted): the twin's probe lifts the agent item
paired with it. Outcome sets hold the other items that end up "carried" or "disturbed", by true name (an agent item
through its pair; an unpaired one under its own name, so it never matches); items paired with held true items and the
probe's ambiguous true items are left out on both sides; "intermediate" is in no set.
- A probe whose true set is not empty scores q_i x J, J = |items in both sets with the same outcome| / |items in
  either set| (0 when the true item is unpaired, or when the twin's probe went non-finite).
- A probe whose true set is empty charges false movers only: share = (the twin's items in its set) / (the twin's
  items that took part: not the lifted one, not paired with a held or an ambiguous true item); a non-finite probe
  has share 1; an unpaired true item gives no probe and share 0.
lift = max(0, mean over the non-empty probes of q_i x J - false_mover_penalty x mean over the empty probes of share)
(the first mean is 1 when there is no non-empty probe).

Drawer. The pull-out: J of the sets of items that ride along ("rides"), same naming and exclusions (1 when both sets
are empty, 0 when the twin's probe went non-finite), times band(angle between the twin's slide axis and the true one,
taken as lines in the true camera frame, axis_deg), times 1 if the gate's drawer travel rule holds (twin_gate
drawer_travel: a limited slide, qpos0 inside the range, travel 0.5-1.0 x the drawer's depth, 50 mm more pull-out),
else 0.

Work areas (twin/regions.py): with balanced areas, lift is computed per area over the probes of the area's true
items (the same rule: the area's mean of q_i x J minus the penalty times the area's mean false-mover share, floored at
0) and averaged over the areas; an area without probes is left out. The drawer term is the drawer's own.
"""
from .config import band_down
from .regions import mean_over_areas

MOVED = ("carried", "disturbed")


def _jaccard(true_set, twin_set):
    """Node-level Jaccard with equal classes: |{n in both, same class}| / |nodes in either| (1 when both empty)."""
    nodes = set(true_set) | set(twin_set)
    if not nodes:
        return 1.0, 0, 0
    inter = sum(1 for n in nodes if n in true_set and n in twin_set and true_set[n] == twin_set[n])
    return inter / len(nodes), inter, len(nodes)


def twin_set(outcomes, a2t, lifted, held_true, ambiguous, classes):
    """(set {node: class}, n_taking_part) of one twin probe. outcomes {agent body: class} of the probe (held bodies
    already removed)."""
    out, n = {}, 0
    for k, cls in sorted(outcomes.items()):
        if k == lifted:
            continue
        t = a2t.get(k)
        if t is not None and (t in held_true or t in ambiguous):
            continue
        n += 1
        if cls in classes:
            out[t if t is not None else "agent:" + k] = cls
    return out, n


def score(truth2, q_true, t2a, a2t, lifts, drawer, axis_deg, cfg, areas=None, drawer_travel_ok=True):
    """lifts {true item: {"finite": bool, "outcomes": {agent body: class}}} for the paired true items that rest;
    drawer {"finite", "outcomes"} or None; axis_deg: the slide-axis angle (None without a drawer)."""
    ic = cfg["interaction"]
    held = set(truth2.held)
    rows, nonempty, shares = {}, [], []
    for i in truth2.static_items:
        tp = truth2.lift[i]
        S_t = tp["set"]
        amb = tp["ambiguous"]
        j = t2a.get(i)
        q = float(q_true.get(i, 0.0)) if j is not None else 0.0
        r = lifts.get(i)
        row = {"agent": j, "q": q, "true_set": dict(S_t)}
        if S_t:
            if j is None or r is None:
                row.update(score=0.0, jaccard=0.0, why="the true item is not paired" if j is None else "not run")
            elif not r["finite"]:
                row.update(score=0.0, jaccard=0.0, why="the probe went non-finite")
            else:
                S_a, _n = twin_set(r["outcomes"], a2t, j, held, amb, MOVED)
                J, inter, uni = _jaccard(S_t, S_a)
                row.update(score=q * J, jaccard=J, twin_set=S_a, intersection=inter, union=uni)
            nonempty.append(row["score"])
        else:
            if j is None or r is None:
                row.update(share=0.0, why="the true item is not paired" if j is None else "not run")
            elif not r["finite"]:
                row.update(share=1.0, why="the probe went non-finite")
            else:
                S_a, n = twin_set(r["outcomes"], a2t, j, held, amb, MOVED)
                row.update(share=(len(S_a) / n) if n else 0.0, false_movers=sorted(S_a), taking_part=n)
            shares.append(row["share"])
        rows["lift:" + i] = row
    base = sum(nonempty) / len(nonempty) if nonempty else 1.0
    pen = (sum(shares) / len(shares)) if shares else 0.0
    lift = max(0.0, base - float(ic["false_mover_penalty"]) * pen)
    lift_unbalanced = lift
    per_area = None
    if areas and areas.get("balanced"):
        per_area = {}
        for r in areas["areas"]:
            keys = [i for i in truth2.static_items if areas["of_true"].get(i) == r]
            ne = [rows["lift:" + i]["score"] for i in keys if truth2.lift[i]["set"]]
            sh = [rows["lift:" + i]["share"] for i in keys if not truth2.lift[i]["set"]]
            if not ne and not sh:
                per_area[r] = {"score": None, "n_nonempty": 0, "n_empty": 0}
                continue
            b_r = sum(ne) / len(ne) if ne else 1.0
            p_r = sum(sh) / len(sh) if sh else 0.0
            per_area[r] = {"score": max(0.0, b_r - float(ic["false_mover_penalty"]) * p_r), "nonempty_mean": b_r,
                           "penalty": p_r, "n_nonempty": len(ne), "n_empty": len(sh)}
        if any(v["score"] is not None for v in per_area.values()):
            lift = mean_over_areas({r: v["score"] for r, v in per_area.items()})
    # the drawer pull-out
    dt = truth2.drawer
    R_t = {k: c for k, c in dt["set"].items() if c == "rides"}
    if drawer is None:
        Jd, dinfo = 0.0, {"why": "no drawer probe"}
    elif not drawer["finite"]:
        Jd, dinfo = 0.0, {"why": "the drawer probe went non-finite"}
    else:
        R_a, n = twin_set(drawer["outcomes"], a2t, None, held, dt["ambiguous"], ("rides",))
        Jd, inter, uni = _jaccard(R_t, R_a)
        dinfo = {"twin_set": R_a, "intersection": inter, "union": uni, "taking_part": n}
    band = band_down(axis_deg, *ic["axis_deg"]) if axis_deg is not None else 0.0
    dr = Jd * band * (1.0 if drawer_travel_ok else 0.0)
    dinfo.update(true_set=R_t, jaccard=Jd, axis_deg=axis_deg, axis_band=band, travel_ok=bool(drawer_travel_ok))
    c5 = float(ic["w_lift"]) * lift + float(ic["w_drawer"]) * dr
    return c5, {"lift": lift, "lift_unbalanced": lift_unbalanced, "lift_per_area": per_area,
                "lift_nonempty_mean": base, "lift_penalty": pen, "n_nonempty": len(nonempty),
                "n_empty": len(shares), "drawer": dr, "drawer_detail": dinfo, "probes": rows}
