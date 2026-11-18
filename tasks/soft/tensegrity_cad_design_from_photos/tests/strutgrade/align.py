"""Whole-strut alignment without a base part.

Rigid (no scale, no reflection): A maps the submission's world frame to the strut frame. Every fitted piece "wishes"
the strut were placed by A_ij = W_j T_fit_i (W_j: reference instance j of the piece's family in the strut frame); A
is the consensus of these wishes.

Contributors (one per fitted part and per recognised purchased instance, each of total weight 1 spread evenly over
its points, align.points_per_part fixed points):
  surface   an asymmetric design whose best shape score is at least align.shape_min (0.2): fixed surface points of
            the submitted part (submission frame); their targets W_j T_c p for instance j and candidate pose c
  axis      a symmetric design (end cap C4, end cap cover C_inf, winch approximately D_inf; families.json) and the
            rod (continuous, end for end): the two ends of the symmetry axis only, as the part carries them
            (inv(T_c) applied to the design's axis ends); targets the reference instance's axis ends; with an end-for-
            end symmetry the pairing of the two ends is free (the better one is taken)
  centroid  a part whose best shape score is below align.shape_min: its B-rep centroid only (no say in the turn)
  purchased an asymmetric purchased instance (the gearmotor): fixed points of the grader's copy of its type at the
            recognised pose (exact for an unchanged import); targets the same points at the reference placement
Cost of a contributor under (A, j, c): the mean over its points of the truncated robust loss
rho(r) = s^2 log(1 + min(r, cap)^2 / s^2), r = |A p - q|, s = align.robust_mm, cap = align.cap_mm. Beyond the cap a
point costs a constant (a badly placed piece does not pull the rest); inside it the loss grows like r^2 near zero and
only logarithmically past s, so a piece fitted with a bias of a millimetre or two (a convex hull reduced to its
centroid) moves the consensus of the well-fitted pieces by hundredths of a millimetre, not tenths (a plain
truncated sum of squares let the P01 hull pull the strut 0.115 mm). The A
step is the Kabsch solution with the matching reweighting, w s^2 / (s^2 + r^2) for every point within the cap
(iteratively reweighted least squares of the same loss).

Search: a start from every wish A_ij (every tied candidate of every part and instance of its family, every purchased
instance and reference placement of its type; duplicates within 1e-3 mm dropped). From each start, at most
align.rounds (5) rounds in a fixed order: (1) inside every family and every purchased type, the Hungarian assignment
of pieces to reference instances on the cost under the current A (each piece at its cheapest candidate); (2) every
part's candidate among its tied ones, cheapest under A; (3) A by weighted Kabsch on all points within cap; stop when
(1) and (2) do not change and A moved less than 1e-9 mm. The result of lowest total cost wins; a tie (within
align.tie_mm) goes to the one with more landed contributors (rms <= align.land_mm), then to the earlier start. So
which end of the rod is +z is decided by all asymmetric pieces together (the electronics enclosure halves and the two
motor modules), not by any single one.

Fewer than three fitted pieces cannot be aligned: gates.fit_gate zeroes the document before this runs.
"""
import time

import numpy as np

from .geom import apply, box_corners, even_subset, hungarian, inv, kabsch, pose_distance


def _part_contributor(ctx, sid, fit):
    """The alignment points of one fitted part for each of its tied candidates and the targets for each instance."""
    A_cfg = ctx.cfg["align"]
    ref = ctx.ref
    s = ctx.solid(sid)
    key = fit["family"]
    d = ref.designs[key]
    sym = d["symmetry"]
    cands = [fit["candidates"][i] for i in fit["tied"]] if fit["tied"] else fit["candidates"][:1]
    idx = list(fit["tied"]) if fit["tied"] else [0]
    best = fit["candidates"][0]["shape"] if fit["candidates"] else 0.0
    insts = ref.family_tags[key]
    if best < float(A_cfg["shape_min"]):
        mode = "centroid"
    elif sym["kind"] != "none":
        mode = "axis"
    else:
        mode = "surface"
    src, tgt = [], {}
    if mode == "surface":
        P = even_subset(s["geo"].query()[0], int(A_cfg["points_per_part"]))
        for c in cands:
            src.append(P)
        for j in insts:
            tgt[j] = [apply(ref.instances[j]["W"] @ c["T"], P) for c in cands]
    elif mode == "axis":
        ends = sym["axis_points"]
        for c in cands:
            src.append(apply(inv(c["T"]), ends))
        for j in insts:
            q = apply(ref.instances[j]["W"], ends)
            tgt[j] = [q for _c in cands]
    else:
        P = np.asarray(s["com"], float)[None, :]
        for c in cands:
            src.append(P)
        for j in insts:
            tgt[j] = [apply(ref.instances[j]["W"] @ c["T"], P) for c in cands]
    wishes = {j: [ref.instances[j]["W"] @ c["T"] for c in cands] for j in insts}
    return {"key": "part:%s" % s["tag"], "kind": "part", "sid": sid, "group": "family:" + key, "mode": mode,
            "src": src, "tgt": tgt, "cand_index": idx, "flip": bool(mode == "axis" and sym.get("flip")),
            "wishes": wishes, "slots": list(insts)}


def _purchased_contributor(ctx, k, it):
    ref = ctx.ref
    A_cfg = ctx.cfg["align"]
    typ = ref.types[it["type"]]
    sym = typ["symmetry"]
    pids = sorted(p for p, pl in ref.placements.items() if pl["type"] == it["type"])
    W = it["W"]
    if sym["continuous"]:
        mode = "axis"
        P_local = sym["axis_points"]
    else:
        mode = "purchased"
        anchor = typ["cons"][typ["anchor"]]
        P_local = even_subset(anchor["geo"].query()[0], int(A_cfg["points_per_part"])) if anchor["geo"] is not None \
            else np.asarray([anchor["com"]], float)
    src = [apply(W, P_local)]
    tgt = {p: [apply(ref.placements[p]["W"], P_local)] for p in pids}
    wishes = {p: [ref.placements[p]["W"] @ inv(W)] for p in pids}
    return {"key": "purchased:%d" % k, "kind": "purchased", "inst": k, "group": "type:" + it["type"], "mode": mode,
            "src": src, "tgt": tgt, "cand_index": [0], "flip": bool(mode == "axis" and sym["flip"]),
            "wishes": wishes, "slots": pids}


def _loss(r2, cap2, sig2):
    """Mean truncated robust loss of squared residuals r2 (module docstring)."""
    return float(np.mean(sig2 * np.log1p(np.minimum(r2, cap2) / sig2)))


def _cost(A, src, tgt, flip, cap2, sig2):
    """(robust cost, rms, per-point squared residuals, target used) of one contributor's points (weights equal,
    summing to 1)."""
    P = apply(A, src)
    r2 = np.sum((P - tgt) ** 2, axis=1)
    if flip and len(src) == 2:
        r2b = np.sum((P - tgt[::-1]) ** 2, axis=1)
        if _loss(r2b, cap2, sig2) < _loss(r2, cap2, sig2):
            r2, tgt = r2b, tgt[::-1]
    return _loss(r2, cap2, sig2), float(np.sqrt(np.mean(r2))), r2, tgt


def _round(contribs, A, cap2, sig2, cand_tie):
    """One assignment + candidate choice under A: {contributor index: (slot, candidate position)}."""
    groups = {}
    for ci, c in enumerate(contribs):
        groups.setdefault(c["group"], []).append(ci)
    choice = {}
    for g in sorted(groups):
        members = groups[g]
        slots = contribs[members[0]]["slots"]
        cost = np.zeros((len(members), len(slots)))
        best_c = {}
        for a, ci in enumerate(members):
            c = contribs[ci]
            for b, j in enumerate(slots):
                vals = [_cost(A, c["src"][k], c["tgt"][j][k], c["flip"], cap2, sig2)[0] for k in range(len(c["src"]))]
                # the earliest of the candidates as near as the nearest (within align.cand_tie_mm2): the fit's own order
                # decides between poses the consensus cannot tell apart (the turns of a symmetric part)
                lo = min(vals)
                kbest = next(k for k, v in enumerate(vals) if v <= lo + cand_tie)
                cost[a, b] = vals[kbest]
                best_c[(ci, j)] = kbest
        for a, b in hungarian(cost):
            ci = members[a]
            choice[ci] = (slots[b], best_c[(ci, slots[b])])
    return choice


def _solve(contribs, A0, cfg):
    A_cfg = cfg["align"]
    cap2 = float(A_cfg["cap_mm"]) ** 2
    cand_tie = float(A_cfg["cand_tie_mm2"])
    sig2 = float(A_cfg["robust_mm"]) ** 2
    A = np.array(A0, float)
    choice, rounds = None, 0
    probe = box_corners(np.array([-160.0, -40.0, -160.0]), np.array([160.0, 40.0, 160.0]))
    for rounds in range(1, int(A_cfg["rounds"]) + 1):
        new = _round(contribs, A, cap2, sig2, cand_tie)
        P, Q, W = [], [], []
        for ci in sorted(new):
            j, k = new[ci]
            c = contribs[ci]
            _cv, _rms, r2, tgt = _cost(A, c["src"][k], c["tgt"][j][k], c["flip"], cap2, sig2)
            inl = r2 <= cap2
            if inl.any():
                P.append(c["src"][k][inl])
                Q.append(tgt[inl])
                # inside the cap, a point far from its wish counts less (Cauchy weight, scale align.robust_mm): a
                # piece fitted with a bias of a millimetre or two does not drag the well-fitted rest
                W.append(sig2 / (sig2 + r2[inl]) / len(c["src"][k]))
        if P and sum(len(p) for p in P) >= 3:
            A_new = kabsch(np.vstack(P), np.vstack(Q), np.concatenate(W))
        else:
            A_new = A
        moved = pose_distance(A_new, A, probe)
        same = new == choice
        A, choice = A_new, new
        if same and moved < 1e-9:
            break
    final = _round(contribs, A, cap2, sig2, cand_tie)
    total, landed, per = 0.0, 0, {}
    for ci, c in enumerate(contribs):
        if ci in final:
            j, k = final[ci]
            cv, rms, r2, _t = _cost(A, c["src"][k], c["tgt"][j][k], c["flip"], cap2, sig2)
            sq = float(np.mean(np.minimum(r2, cap2)))
        else:
            j, k, cv, rms, sq = None, None, _loss(np.array([cap2]), cap2, sig2), float("inf"), cap2
        total += cv
        ok = rms <= float(A_cfg["land_mm"])
        landed += int(ok)
        per[ci] = {"slot": j, "cand_pos": k, "cost": cv, "sq": sq, "rms_mm": rms, "landed": bool(ok)}
    return {"A": A, "choice": final, "total": total, "landed": landed, "per": per, "rounds": rounds}


def align(ctx):
    """The whole-strut alignment (module docstring). Sets ctx.alignment:
    {"A" (submission -> strut), "assignment" {reference tag: sid}, "part_instance" {sid: reference tag},
     "chosen" {sid: candidate index}, "T_fit" {sid: chosen candidate's T}, "purchased_assignment" {placement id:
     instance index}, "residual_mm", "landed", "contributors", "rounds", "start", "n_starts", "per_piece" {key: {...}}}."""
    t0 = time.time()
    A_cfg = ctx.cfg["align"]
    contribs = []
    for sid in sorted(ctx.fits):
        if ctx.fits[sid]["candidates"]:
            contribs.append(_part_contributor(ctx, sid, ctx.fits[sid]))
    for k, it in enumerate(ctx.instances):
        contribs.append(_purchased_contributor(ctx, k, it))
    probe = box_corners(np.array([-160.0, -40.0, -160.0]), np.array([160.0, 40.0, 160.0]))
    starts, labels = [], []
    for c in contribs:
        for j in c["slots"]:
            for k, Wish in enumerate(c["wishes"][j]):
                if any(pose_distance(Wish, S, probe) <= 1e-3 for S in starts):
                    continue
                starts.append(Wish)
                labels.append("%s->%s#%d" % (c["key"], j, k))
    best = None
    tie = float(A_cfg["tie_mm"])
    for si, (S, lab) in enumerate(zip(starts, labels)):
        r = _solve(contribs, S, ctx.cfg)
        key = (r["total"], -r["landed"], si)
        if best is None:
            best = (key, r, lab)
            continue
        (bt, bl, _bi) = best[0]
        if r["total"] < bt - tie or (abs(r["total"] - bt) <= tie and -r["landed"] < bl):
            best = (key, r, lab)
    _key, r, lab = best
    out = {"A": r["A"], "assignment": {}, "part_instance": {}, "chosen": {}, "T_fit": {}, "purchased_assignment": {},
           "contributors": len(contribs), "landed": r["landed"], "rounds": r["rounds"], "start": lab,
           "n_starts": len(starts), "per_piece": {}}
    n = 0
    tot = 0.0
    for ci, c in enumerate(contribs):
        p = r["per"][ci]
        out["per_piece"][c["key"]] = {"mode": c["mode"], "instance": p["slot"], "rms_mm": p["rms_mm"],
                                      "landed": p["landed"]}
        if p["slot"] is None:
            continue
        tot += p["sq"]
        n += 1
        if c["kind"] == "part":
            ci_idx = c["cand_index"][p["cand_pos"]]
            out["assignment"][p["slot"]] = c["sid"]
            out["part_instance"][c["sid"]] = p["slot"]
            out["chosen"][c["sid"]] = ci_idx
            out["T_fit"][c["sid"]] = ctx.fits[c["sid"]]["candidates"][ci_idx]["T"]
        else:
            out["purchased_assignment"][p["slot"]] = c["inst"]
    # the parts that took no part in the alignment (none today: every fitted part contributes) keep their best pose
    for sid, f in ctx.fits.items():
        if sid not in out["chosen"] and f["candidates"]:
            out["chosen"][sid] = 0
            out["T_fit"][sid] = f["candidates"][0]["T"]
    out["residual_mm"] = float(np.sqrt(tot / max(1, n)))
    ctx.alignment = out
    ctx.stats["alignment_residual_mm"] = out["residual_mm"]
    ctx.stats["alignment_landed"] = float(out["landed"])
    ctx.stats["alignment_contributors"] = float(out["contributors"])
    ctx.stats["alignment_starts"] = float(out["n_starts"])
    ctx.stats["t_align_s"] = time.time() - t0
    return out
