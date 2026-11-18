"""Criterion C, fit and winch clearance (weight 0.10):

  C = fit_clearance.w_purchased 0.4 * purchased + fit_clearance.w_made 0.3 * made + fit_clearance.w_clearance 0.3 * clearance

purchased  per reference placement (equal weights), the recognised instance assigned to it (the grader's copies of its
           constituents at the recognised poses):
             contact   over every pair (placement, made instance) of features.json supports.supports[*].pairs (the made
                       parts near it in the original, not only the ones it rests on): band(max(0, d - d_ref);
                       fit_clearance.contact_mm 0.1 / 2.0), d the smallest surface distance between the submitted instance and the
                       part assigned to that made instance (none: no credit), d_ref the same measure on the reference;
                       the pairs averaged
             intrusion band(excess / volume of the instance; fit_clearance.intrusion_rel 0.05 % / 5 %), excess = the volume it
                       shares with each made part beyond what the original shares (same boolean on the reference) and
                       fit_clearance.slack_mm3, plus all it shares with extra solids
           score = contact x intrusion; a missing placement scores 0.
made       per reference instance: band(excess / own volume; fit_clearance.intrusion_rel), excess = its common volume with every
           other made part beyond the original's own overlap of that pair (the exemption, e.g. end cap and cover 1.59
           mm3) and fit_clearance.slack_mm3, plus all it shares with extra solids; missing scores 0; averaged.
blame      an overlap between two pieces beyond the original's is charged to the piece further from the original: the
           lower quality (a made part: its chosen pose's shape score x its placement score; a purchased instance: its
           placement score) carries it alone, within fit_clearance.blame_tie (0.05) both do; a purchased-made overlap
           charged to the made part counts in the made part's excess. So a part modelled as its convex hull pays for
           what it cuts into, its well-made neighbours do not (the QC row "P01 as its convex hull", README.md 9).
           A support pair whose made part is missing is left out of the contact mean (the missing part scores 0 in its
           own terms).
clearance  per winch: a cylinder of the original flange radius and winch width (features.json supports.
           winch_sweep_reference) on the submitted winch's own axis (its pose A inv(T_fit) applied to the design's axis
           and extent middle), its common volume with everything except the winch itself and its end's gearmotor (made
           parts, other purchased instances, extra solids), beyond the original's own (the same boolean on the
           reference, 0.0006 mm3) and fit_clearance.slack_mm3: band(excess; fit_clearance.sweep_mm3 0.5 / 40); a missing winch scores 0.
All volumes and distances in the strut frame (A); the reference side is computed once per Reference and cached on it.

Booleans need a closed two-manifold. A tessellation that is not one (the base then falls back on the convex hull,
geom.mesh_solid) is repaired locally here before any volume is taken: the triangles on edges not shared by exactly two
triangles and their one-ring are removed and the holes closed with fans (geom.fill_holes); only when that does not give
a manifold is the hull used (statistic fit_hull_fallbacks). The vendor gearmotor STEP's tessellation has 5 open and 2
non-manifold edges among 228 297 (a sliver inside the gearbox); repaired, its volume is within 0.013 % of the B-rep's,
while its hull overlaps the original winch by 254.6 mm3.
"""
import copy
import math

import numpy as np

from .geom import (Placed, aabb_overlap, apply, band, common_volume, fill_holes, make_T, rotation_between,
                   surface_distances, to_manifold)
from .recognise import submission_geo

NAME = "fit"
REPAIR_NOTE = "fit_clearance: repaired locally for the booleans"


def repaired(ctx, geo):
    """geo itself when it has a manifold, else a shallow copy whose manifold is the local repair (module docstring),
    cached on the geo; None for no geometry."""
    if geo is None or geo.man is not None:
        return geo
    got = getattr(geo, "_fit_repaired", None)
    if got is not None:
        return got
    V, F = geo.V, geo.F
    man = None
    for _round in range(int(ctx.cfg["fit_clearance"]["repair_rounds"])):
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        _u, inv, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
        bad = np.any(cnt[inv.reshape(3, -1).T] != 2, axis=1)
        if not bad.any():
            man = to_manifold(V, F)
            break
        ring = np.isin(F, np.unique(F[bad])).any(axis=1)
        V, F, _n = fill_holes(V, F[~ring], int(ctx.cfg["mesh"]["fill_max_loop"]))
        man = to_manifold(V, F)
        if man is not None:
            break
    g2 = copy.copy(geo)
    if man is not None and abs(float(man.volume()) / max(geo.volume, 1e-9) - 1.0) <= float(ctx.cfg["mesh"]["volume_tol_rel"]):
        g2.man = man
        g2.notes = list(geo.notes) + [REPAIR_NOTE]
    geo._fit_repaired = g2
    return g2


def min_distance(group_a, group_b, disk, reach):
    """Smallest surface distance between two groups of Placed (query points within `reach` of the other's box only;
    the UAV CAD task's min_distance)."""
    best = np.inf
    for a in group_a:
        for b in group_b:
            if not aabb_overlap(a.box, b.box, reach):
                continue
            for x, y in ((a, b), (b, a)):
                P = x.query()[0]
                lo, hi = y.box[:3] - reach, y.box[3:] + reach
                sel = np.all((P >= lo) & (P <= hi), axis=1)
                if not sel.any():
                    continue
                D, N, tree = y.dense()
                if tree is None:
                    continue
                d = surface_distances(P[sel], D, N, tree, disk)
                best = min(best, float(d.min()))
    return best


def group_common(group_a, group_b):
    tot = 0.0
    for a in group_a:
        for b in group_b:
            if aabb_overlap(a.box, b.box):
                tot += common_volume(a.man, b.man)
    return tot


class Scene(object):
    """The pieces of one side (reference or submission) in the strut frame: made instances (tag -> Placed),
    purchased placements (placement id -> [Placed]), extra solids ([Placed]) and the winches' design -> strut poses
    (tag -> 4x4)."""

    def __init__(self, made, purchased, extras, winch_pose=None, quality=None):
        self.made, self.purchased, self.extras = made, purchased, extras
        self.winch_pose = winch_pose or {}
        self.quality = quality or {}


def _reference_scene(ctx):
    ref = ctx.ref
    made = {t: Placed(repaired(ctx, ref.designs[ref.family_of[t]]["geo"]), ref.instances[t]["W"]) for t in ref.tags}
    pur = {}
    for pid, pl in sorted(ref.placements.items()):
        pur[pid] = [Placed(repaired(ctx, c["geo"]), pl["W"]) for c in ref.types[pl["type"]]["cons"] if c["geo"] is not None]
    return Scene(made, pur, [], {t: ref.instances[t]["W"] for t in _winch_motor(ctx)})


def _submission_scene(ctx):
    al = ctx.alignment
    A = al["A"]
    made = {}
    for tag in ctx.ref.tags:
        sid = al["assignment"].get(tag)
        if sid is not None:
            g = repaired(ctx, submission_geo(ctx, ctx.solid(sid)))
            if g is not None:
                made[tag] = Placed(g, A)
    pur = {}
    for pid, k in sorted(al["purchased_assignment"].items()):
        it = ctx.instances[k]
        items = []
        for sid in sorted(it["slots"].values()):
            s = ctx.solid(sid)
            rep = ctx.ref.classes[s["match"]["cls"]]["rep"]
            if rep["geo"] is not None:
                items.append(Placed(repaired(ctx, rep["geo"]), A @ s["match"]["T"]))
        pur[pid] = items
    extras = []
    claimed = set()
    for k, it in enumerate(ctx.instances):
        if k not in al["purchased_assignment"].values():
            continue
        claimed.update(it["slots"].values())
    assigned = set(al["assignment"].values())
    for s in ctx.solids:
        if s["sid"] in claimed or s["sid"] in assigned:
            continue
        # extra solids, recognised purchased solids left over, and tagged parts that took no part (not fitted)
        if "match" in s:
            rep = ctx.ref.classes[s["match"]["cls"]]["rep"]
            if rep["geo"] is not None:
                extras.append(Placed(repaired(ctx, rep["geo"]), A @ s["match"]["T"]))
                continue
        g = repaired(ctx, submission_geo(ctx, s))
        if g is not None:
            extras.append(Placed(g, A))
    wp = {t: ctx.part_pose(al["assignment"][t]) for t in _winch_motor(ctx) if al["assignment"].get(t) is not None}
    return Scene(made, pur, extras, wp, quality(ctx))


def quality(ctx):
    """How far each assigned piece is from the original, in [0, 1]: a made part's chosen pose's shape score times its
    placement score (items_placement), a purchased instance's placement score (its shape is the vendor's)."""
    from . import items_placement as IP
    P = ctx.cfg["placement"]
    al = ctx.alignment
    _m, made = IP.made_scores(ctx, P)
    _p, pur = IP.purchased_scores(ctx, P)
    q = {}
    for tag, r in made.items():
        sid = al["assignment"].get(tag)
        if sid is not None:
            sh = float(ctx.fits[sid]["candidates"][al["chosen"][sid]]["shape"])
            q[tag] = sh * float(r["score"])
    for pid, r in pur.items():
        if r.get("present"):
            q[pid] = float(r["score"])
    return q


def _supports(ctx):
    """[(placement id, [(made instance, the reference's min distance)])] from features.json, the pieces mapped to placements by
    purchased.json source_piece."""
    by_piece = {pl["spec"].get("source_piece"): pid for pid, pl in ctx.ref.placements.items()}
    out = []
    for s in ctx.ref.features["supports"]["supports"]:
        pid = by_piece.get(s["piece"])
        if pid is None:
            continue
        out.append((pid, [(p["made_instance"], float(p["min_distance_mm"])) for p in s["pairs"]]))
    return out


def _sweep_cylinder(ctx, tag, X):
    """The winch's sweep cylinder (manifold) at the design -> strut pose X, with its box."""
    import manifold3d as m3
    Fc = ctx.cfg["fit_clearance"]
    sw = [w for w in ctx.ref.features["supports"]["winch_sweep_reference"] if w["winch"] == tag][0]
    d = ctx.ref.designs[ctx.ref.family_of[tag]]
    pr = [p for p in ctx.ref.features["functional"]["primitives"][ctx.ref.family_of[tag]] if p["kind"] == "radial_profile"][0]
    a = np.asarray(d["symmetry"]["axis_dir"], float)
    a = a / np.linalg.norm(a)
    mid = np.asarray(pr["extent_mid_mm"], float)
    L, R = float(sw["cylinder_length_mm"]), float(sw["cylinder_radius_mm"])
    cyl = m3.Manifold.cylinder(L, R, R, int(Fc["segments"]))
    T = X @ make_T(rotation_between([0.0, 0.0, 1.0], a), mid - a * 0.5 * L)
    man = cyl.transform(np.ascontiguousarray(T[:3, :], np.float32))
    lo, hi = np.array([-R, -R, 0.0]), np.array([R, R, L])
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
    w = apply(T, c)
    return man, np.concatenate([w.min(axis=0), w.max(axis=0)])


def _sweep_overlap(man, box, items):
    tot = 0.0
    for p in items:
        if aabb_overlap(box, p.box):
            tot += common_volume(man, p.man)
    return tot


def _winch_motor(ctx):
    return ctx.cfg["placement"]["joints"]["winch_motor"]


def _reference_cache(ctx):
    """Reference-side numbers (cached on the Reference): pair distances, purchased-made and made-made common volumes,
    the sweep cylinders' own overlaps."""
    ref = ctx.ref
    if getattr(ref, "_fit_cache", None) is not None:
        return ref._fit_cache
    Fc = ctx.cfg["fit_clearance"]
    disk = float(ctx.cfg["sampling"]["disk_mm"])
    reach = float(Fc["contact_mm"][1])
    sc = _reference_scene(ctx)
    out = {"d": {}, "pc": {}, "mc": {}, "sweep": {}, "d_features": {}}
    for pid, pairs in _supports(ctx):
        for tag, d_feat in pairs:
            out["d"][(pid, tag)] = min_distance(sc.purchased[pid], [sc.made[tag]], disk, reach)
            out["d_features"][(pid, tag)] = d_feat
    for pid in sc.purchased:
        for tag in ref.tags:
            out["pc"][(pid, tag)] = group_common(sc.purchased[pid], [sc.made[tag]])
    for i, a in enumerate(ref.tags):
        for b in ref.tags[i + 1:]:
            pa, pb = sc.made[a], sc.made[b]
            out["mc"][(a, b)] = common_volume(pa.man, pb.man) if aabb_overlap(pa.box, pb.box) else 0.0
    wm = _winch_motor(ctx)
    for tag in sorted(wm):
        man, box = _sweep_cylinder(ctx, tag, ref.instances[tag]["W"])
        items = [sc.made[t] for t in ref.tags if t != tag]
        for pid in sc.purchased:
            if pid != wm[tag]:
                items += sc.purchased[pid]
        out["sweep"][tag] = _sweep_overlap(man, box, items)
    ref._fit_cache = out
    return out


def _mc(cache, a, b):
    return cache["mc"].get((a, b), cache["mc"].get((b, a), 0.0))


def item(ctx):
    """Criterion C (module docstring)."""
    return score_scene(ctx, _submission_scene(ctx))


def score_scene(ctx, sc):
    """Criterion C of a scene (the submission's, or a development scene such as the face-offset honest model)."""
    Fc = ctx.cfg["fit_clearance"]
    ref = ctx.ref
    disk = float(ctx.cfg["sampling"]["disk_mm"])
    reach = float(Fc["contact_mm"][1])
    slack = float(Fc["slack_mm3"])
    rc = _reference_cache(ctx)
    pieces = list(sc.made.values()) + [p for v in sc.purchased.values() for p in v] + list(sc.extras)
    ctx.stats["fit_meshes_repaired"] = float(sum(1 for p in pieces if REPAIR_NOTE in p.geo.notes))
    ctx.stats["fit_hull_fallbacks"] = float(sum(1 for p in pieces if p.geo.man is None))
    per = {}
    diags = []
    tie = float(Fc["blame_tie"])
    q = sc.quality

    def blame(x, y):
        """(charge x, charge y) of an overlap between pieces x and y beyond the original's: the one that is further
        from the original (lower quality) carries it alone; within blame_tie both do."""
        qx, qy = q.get(x, 1.0), q.get(y, 1.0)
        if qx < qy - tie:
            return True, False
        if qy < qx - tie:
            return False, True
        return True, True

    made_excess = {t: 0.0 for t in sc.made}
    # purchased: contact over the support pairs x intrusion
    pur_scores = []
    pairs_of = dict(_supports(ctx))
    for pid in sorted(ref.placements):
        mine = sc.purchased.get(pid)
        if not mine:
            pur_scores.append(0.0)
            per[pid] = {"present": False, "score": 0.0}
            continue
        cs, pr = [], []
        for tag, _d in pairs_of.get(pid, []):
            other = sc.made.get(tag)
            if other is None:
                pr.append({"made": tag, "skipped": "part missing"})
                continue
            d = min_distance(mine, [other], disk, reach)
            dr = rc["d"][(pid, tag)]
            cs.append(band(max(0.0, d - (dr if math.isfinite(dr) else 0.0)), Fc["contact_mm"]))
            pr.append({"made": tag, "d_mm": d if math.isfinite(d) else None, "d_ref_mm": dr if math.isfinite(dr) else None})
        contact = float(np.mean(cs)) if cs else 1.0
        excess = 0.0
        for tag, p in sc.made.items():
            if not any(aabb_overlap(m.box, p.box) for m in mine):
                continue
            e = max(0.0, group_common(mine, [p]) - rc["pc"][(pid, tag)] - slack)
            if e > 0.0:
                cp, cm = blame(pid, tag)
                excess += e if cp else 0.0
                made_excess[tag] += e if cm else 0.0
        for ex in sc.extras:
            if any(aabb_overlap(m.box, ex.box) for m in mine):
                excess += max(0.0, group_common(mine, [ex]) - slack)
        vol = sum(m.volume for m in mine)
        s = contact * band(excess / max(vol, 1e-9), Fc["intrusion_rel"])
        pur_scores.append(s)
        per[pid] = {"present": True, "contact": contact, "intrusion_mm3": excess, "score": s, "pairs": pr,
                    "quality": q.get(pid, 1.0)}
        if s < 1.0:
            diags.append("Fit %s: contact %.2f, cuts %.2f mm3 into other pieces" % (pid, contact, excess))
    purchased = float(np.mean(pur_scores)) if pur_scores else 0.0
    # made parts: intrusion into each other, into purchased parts (when charged to the made part) and into extras
    tags = sorted(sc.made)
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            pa, pb = sc.made[a], sc.made[b]
            if not aabb_overlap(pa.box, pb.box):
                continue
            e = max(0.0, common_volume(pa.man, pb.man) - _mc(rc, a, b) - slack)
            if e > 0.0:
                ca, cb = blame(a, b)
                made_excess[a] += e if ca else 0.0
                made_excess[b] += e if cb else 0.0
    made_scores = []
    for tag in ref.tags:
        p = sc.made.get(tag)
        if p is None:
            made_scores.append(0.0)
            per[tag] = {"present": False, "score": 0.0}
            continue
        excess = made_excess[tag]
        for ex in sc.extras:
            if aabb_overlap(p.box, ex.box):
                excess += max(0.0, common_volume(p.man, ex.man) - slack)
        s = band(excess / max(p.volume, 1e-9), Fc["intrusion_rel"])
        made_scores.append(s)
        per[tag] = {"present": True, "intrusion_mm3": excess, "score": s, "quality": q.get(tag, 1.0)}
        if s < 1.0:
            diags.append("Fit %s: cuts %.2f mm3 into other pieces" % (tag, excess))
    made = float(np.mean(made_scores)) if made_scores else 0.0
    # winch sweep clearance
    wm = _winch_motor(ctx)
    cl = []
    for tag in sorted(wm):
        if tag not in sc.winch_pose:
            cl.append(0.0)
            per["sweep_" + tag] = {"present": False, "score": 0.0}
            continue
        man, box = _sweep_cylinder(ctx, tag, sc.winch_pose[tag])
        items = [p for t, p in sc.made.items() if t != tag]
        for pid, its in sc.purchased.items():
            if pid != wm[tag]:
                items += its
        items += sc.extras
        tot = _sweep_overlap(man, box, items)
        excess = max(0.0, tot - rc["sweep"][tag] - slack)
        s = band(excess, Fc["sweep_mm3"])
        cl.append(s)
        per["sweep_" + tag] = {"present": True, "overlap_mm3": tot, "reference_mm3": rc["sweep"][tag], "score": s}
        if s < 1.0:
            diags.append("Clearance of %s: its turning envelope overlaps %.2f mm3 of other pieces" % (tag, tot))
    clearance = float(np.mean(cl)) if cl else 0.0
    score = float(Fc["w_purchased"]) * purchased + float(Fc["w_made"]) * made + float(Fc["w_clearance"]) * clearance
    ctx.stats.update(fit_purchased=purchased, fit_made=made, fit_clearance=clearance)
    d_dev = [abs(rc["d"][k] - rc["d_features"][k]) for k in rc["d"] if math.isfinite(rc["d"][k])]
    ctx.stats["fit_reference_pair_distance_vs_features_max_mm"] = float(max(d_dev)) if d_dev else 0.0
    return {"score": score, "per_part": per, "diagnostics": diags[:6]}
