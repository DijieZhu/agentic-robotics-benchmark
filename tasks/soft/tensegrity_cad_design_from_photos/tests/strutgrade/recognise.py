"""Recognition of purchased parts by their full shape, in the submission's own frame (the code of the UAV CAD
task's recognise / form_instances unchanged, with this task's type table: GEARMOTOR (Pololu 5192) and ROD
(McMaster-Carr 8974K19); the five boards join when they are in the payload).

Every untagged solid (not demoted by a local rule) is compared with every constituent class of the purchased types
(the grader's copies, read from the pinned vendor STEPs): volume within recognition.volume_rel (1 %), then a rigid
registration with proper rotations only: exact on the B-rep vertex set when the solid is an unchanged import (every
vertex within exact_tol_mm), else ICP on surface samples, accepted when the symmetric surface p95 is within
approx_p95_mm (absolute, relative to the size) of the representative's own sampling noise. Labels are not looked at.

A module instance is seeded by every solid that matches its type's anchor constituent (the module pose); the other
constituents join the instance whose predicted constituent centroid they meet within attach_mm, one to one.
Completeness = the share of the type's constituents found. Recognised solids left over are marked extra_cls.
"""
import numpy as np

from .geom import Placed, apply, hungarian, inv, mesh_solid, one_way, register_exact, register_samples


def item_error(ctx, what, exc):
    from . import item_error as _ie
    _ie(ctx, what, exc)


def submission_geo(ctx, s):
    """The tessellation of a submitted solid in its own (world) frame, cached on the record."""
    if "geo" not in s:
        from .detenv import seed_for
        s["geo"] = mesh_solid(s["shape"], ctx.cfg, ctx.tmp, seed_for(ctx.cfg, "submission", s["sid"]), volume=s.get("V"))
    return s["geo"]


def recognise(ctx):
    """Match every untagged solid to a constituent class. Sets s["match"] = {"cls", "T" (class representative local
    -> submission frame), "err", "mode" (exact | approx), "alt" [every exact solution]}."""
    R = ctx.cfg["recognition"]
    vol_rel = float(R["volume_rel"])
    tol = float(R["exact_tol_mm"])
    n_exact = n_approx = 0
    for s in ctx.solids:
        if s["tag"] is not None or s.get("demoted"):
            continue
        try:
            got = _recognise_one(ctx, s, R, vol_rel, tol)
        except Exception as exc:  # noqa: BLE001 - this solid stays unrecognised (an extra solid)
            item_error(ctx, "recognition of a solid of %s" % (s["label"] or s["component"]), exc)
            got = None
        if got is not None:
            s["match"] = got
            if got["mode"] == "exact":
                n_exact += 1
            else:
                n_approx += 1
    ctx.stats["purchased_solids_exact"] = float(n_exact)
    ctx.stats["purchased_solids_approx"] = float(n_approx)


def _recognise_one(ctx, s, R, vol_rel, tol):
    ref = ctx.ref
    cands = [cl for cl in ref.classes if abs(s["V"] / cl["rep"]["V"] - 1.0) <= vol_rel]
    cands.sort(key=lambda cl: (abs(s["V"] / cl["rep"]["V"] - 1.0), cl["id"]))
    for cl in cands:
        r = register_exact(cl["rep"], s, tol, all_solutions=True)
        if r:
            return {"cls": cl["id"], "T": r[0][0], "err": r[0][1], "mode": "exact", "alt": [g[0] for g in r]}
    disk = float(ctx.cfg["sampling"]["disk_mm"])
    for cl in cands[:int(R["approx_candidates"])]:
        rep = cl["rep"]
        if rep["geo"] is None:
            continue
        g = submission_geo(ctx, s)
        if g is None:
            continue
        T, _rms = register_samples(rep["geo"], g, ctx.cfg, src=rep, dst=s)
        a = Placed(rep["geo"], T)
        b = Placed(g, np.eye(4))
        d = np.r_[one_way(a.query()[0], b.dense(), disk), one_way(b.query()[0], a.dense(), disk)]
        p95 = float(np.quantile(d, 0.95)) if len(d) else np.inf
        if "base95" not in rep:      # the representative's own samples against themselves (sampling noise)
            pr = Placed(rep["geo"], np.eye(4))
            rep["base95"] = float(np.quantile(one_way(pr.query()[0], pr.dense(), disk), 0.95))
        lim = rep["base95"] + max(float(R["approx_p95_mm"][0]), float(R["approx_p95_mm"][1]) * rep["size"])
        if p95 <= lim:
            return {"cls": cl["id"], "T": T, "err": p95, "mode": "approx", "alt": [T]}
    return None


def form_instances(ctx):
    """Module instances: seeded by anchor matches, the other constituents attached one to one by predicted centroid.
    Returns [{"type", "W" (local -> submission frame), "slots": {k: sid}, "completeness", "mode", "alt_W"}]."""
    ref = ctx.ref
    attach = float(ctx.cfg["recognition"]["attach_mm"])
    by_cls = {}
    for s in ctx.solids:
        if "match" in s:
            by_cls.setdefault(s["match"]["cls"], []).append(s)
    claimed = set()
    inst = []

    def n_attachable(typ, W):
        n = 0
        for c in typ["cons"]:
            p = apply(W, c["com"])
            if any(np.linalg.norm(p - o["com"]) <= attach for o in by_cls.get(c["cls"], [])):
                n += 1
        return n

    for tname in sorted(ref.types):
        typ = ref.types[tname]
        a = typ["cons"][typ["anchor"]]
        cl = ref.classes[a["cls"]]
        M = dict((id(m[0]), m[1]) for m in cl["members"])[id(a)]
        for s in sorted(by_cls.get(a["cls"], []), key=lambda x: x["sid"]):
            if s["sid"] in claimed:
                continue
            Ws = [T @ inv(M) for T in s["match"].get("alt", [s["match"]["T"]])]
            if len(typ["cons"]) > 1 and len(Ws) > 1:
                Ws.sort(key=lambda W: -n_attachable(typ, W))
            inst.append({"type": tname, "W": Ws[0], "alt_W": Ws, "slots": {typ["anchor"]: s["sid"]},
                         "mode": s["match"]["mode"], "err": s["match"]["err"], "anchor_sid": s["sid"]})
            claimed.add(s["sid"])
    for cl in ref.classes:
        slots = []
        for i, it in enumerate(inst):
            for c, _M in cl["members"]:
                if c["type"] == it["type"] and c["k"] not in it["slots"]:
                    slots.append((i, c["k"], apply(it["W"], c["com"])))
        cands = [s for s in by_cls.get(cl["id"], []) if s["sid"] not in claimed]
        if not slots or not cands:
            continue
        cost = np.array([[float(np.linalg.norm(p - s["com"])) for s in cands] for (_i, _k, p) in slots])
        for r, c in hungarian(cost):
            if cost[r, c] <= attach:
                i, k, _p = slots[r]
                inst[i]["slots"][k] = cands[c]["sid"]
                claimed.add(cands[c]["sid"])
    for it in inst:
        it["completeness"] = len(it["slots"]) / float(len(ref.types[it["type"]]["cons"]))
    for s in ctx.solids:
        if "match" in s and s["sid"] not in claimed:
            s["extra_cls"] = s["match"]["cls"]
    ctx.stats["purchased_instances_found"] = float(len(inst))
    ctx.stats["purchased_solids_unattached"] = float(sum(1 for s in ctx.solids if "extra_cls" in s))
    return inst
