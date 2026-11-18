"""Scoring item H, holes (weight 0.20; every hole step counts once), and the feature
preference among a part's candidate poses.

Reference side: the hole steps of features.json (concave cylinder patches of the authors' mesh found by
primitives.find_cylinders; 64 per strut), each with axis, radius and two ends in its design's frame; graded = position
gradable in seen.json (all without it).

Submission side: the concave cylinders of the submitted part's own tessellation (geom.mesh_solid, the triangles the
other items use), found with the vendored primitives.py in three passes over the same triangles:
  1 smooth     primitives.find_cylinders with the reference's HOLE_RULE unchanged (analytic cylinders and B-spline bores
               tessellate at <= 0.1 rad, fine facets): the code that built the reference table
  2 trimmed    a smooth component that fails the cylinder test (a bore chained through a tangent fillet, a rounded
               or lead-in mouth into its neighbouring faces): its dominant local axes (the cross product of two
               neighbouring faces' normals is the axis on a cylinder; clustered, holes.axis_*), and for each axis the
               component's faces within holes.trim_perp_deg of perpendicular to it; each edge-connected piece of those
               is tested again
  3 facets     the faces not used yet, linked across edges whose dihedral is <= holes.facet_link_deg (a bore written
               as a prism of as few as holes.facet_min_facets flat facets, e.g. 12), tested whole or cut as in pass 2;
               a piece counts when it passes the same cylinder test, spans >= holes.facet_min_span_deg about its axis
               and has >= holes.facet_min_facets distinct facet planes
  merge        pieces of one wall (coaxial, same radius, extents touching: holes.merge_*) become one cylinder
Every piece is tested as in find_cylinders: >= HOLE_RULE min_facets faces, normals spanning a plane, vertices on a
circle about the normal-plane axis within max(res_abs_mm, res_rel x r), area-weighted normals towards the axis.

Matching, per Body, at a pose T (submission -> design): every graded reference step and every submitted cylinder
(moved by T) give
  angle    between the axes (deg, sign free)
  offset   sqrt(radial^2 + gap^2): radial = distance of the reference step's axis midpoint from the submitted axis
           line, gap = how far the two axial extents are apart along the submitted axis (0 when they overlap)
  dr       difference of the radii
a pair is admissible when offset <= holes.match_cap_factor x the zero point of the size class's offset band; the
one-to-one assignment (Hungarian) minimises the sum of min(offset / zero, clip) + min(dr / zero_r, clip) +
min(angle / zero_angle, clip). A matched step scores band(offset) x band(dr) x band(angle) (dr only where the step's
radius is gradable: its mouth is seen without an insert in it, seen.json); an unmatched step 0.
H = mean over all graded steps of the strut (all 13 Bodies); the steps of a missing or unfitted Body score 0.

pose_preference(ctx, solid, design_key, T) -> the mean of the part's step scores and flat-face scores (the
plane scores on items_planes' ranking subset) at T, or None when the design has no graded feature: the fit's
feature preference among candidate poses (fit.feature_tie).

Configuration: section "holes" of grader_config.json over HOLES_DEFAULTS.
"""
import math

import numpy as np

from . import primitives as pr
from .geom import apply, apply_dir, band, hungarian
from .items_shape import body_pose, graded_steps, section
from .recognise import submission_geo

HOLES_DEFAULTS = {
    "offset_mm": {"small": [0.12, 0.60], "large": [0.20, 1.00]},
    "radius_mm": [0.08, 0.35],
    "angle_deg": [2.0, 10.0],
    "match_cap_factor": 2.0,
    "cost_clip": 2.0,
    "trim_perp_deg": 2.0,
    "facet_link_deg": 46.0,
    "facet_min_facets": 4,
    "facet_min_span_deg": 90.0,
    "axis_min_dihedral_deg": 1.0,
    "axis_cluster_deg": 2.0,
    "axis_min_support": 4,
    "axis_max_candidates": 4,
    "merge_angle_deg": 1.0,
    "merge_axis_mm": 0.02,
    "merge_radius_mm": 0.02,
    "merge_gap_mm": 0.2,
}


# ============================================================================ submitted cylinders
def _patch(mesh, fi, rule):
    """The cylinder test of primitives.find_cylinders on one set of faces: the cylinder dict (same keys) or None."""
    if len(fi) < rule["min_facets"]:
        return None
    w = mesh.A[fi]
    if w.sum() <= 0:
        return None
    Nn = mesh.N[fi]
    vid = np.unique(mesh.F[fi])
    P = mesh.V[vid]
    cyl = pr.fit_cylinder(P, Nn, w)
    if cyl is None:
        return None
    ev = cyl["ev"]
    if ev[0] > rule["ev0_max"] or ev[1] < rule["ev1_min"]:
        return None
    r, res, d, centre = cyl["radius"], cyl["res"], cyl["dir"], cyl["centre"]
    if res.max() > max(rule["res_abs_mm"], rule["res_rel"] * r):
        return None
    rel = mesh.C[fi] - centre
    radial = rel - np.outer(rel @ d, d)
    concave = float(np.sum(w * np.einsum("ij,ij->i", Nn, radial))) < 0
    t = (P - centre) @ d
    t0, t1 = float(t.min()), float(t.max())
    point = centre + d * 0.5 * (t0 + t1)
    loc = {v: i for i, v in enumerate(vid.tolist())}
    faces_local = [[loc[v] for v in tri] for tri in mesh.F[fi].tolist()]
    span, step = pr.angular_span(P, point, d, faces_local, rule["span_bin_deg"])
    return {"concave": bool(concave), "radius": float(r), "dir": d, "point": point,
            "t_range": (t0 - 0.5 * (t0 + t1), t1 - 0.5 * (t0 + t1)), "length": t1 - t0,
            "max_residual": float(res.max()), "span_deg": span, "facet_step_deg": step, "faces": np.sort(fi),
            "area": float(w.sum())}


def _groups(lab, members):
    """Lists of face indices per label, for the faces `members` (label order)."""
    lab = lab[members]
    order = np.argsort(lab, kind="stable")
    ls = lab[order]
    if not len(ls):
        return []
    starts = np.r_[0, np.nonzero(np.diff(ls))[0] + 1]
    ends = np.r_[starts[1:], len(ls)]
    return [np.sort(members[order[s:e]]) for s, e in zip(starts, ends)]


def _sub_labels(mesh, keep_face, edge_ok):
    """Connected components of the faces keep_face over the adjacency edges edge_ok (labels for every face; faces
    outside keep_face get their own label)."""
    sel = edge_ok & keep_face[mesh.fa] & keep_face[mesh.fb]
    return pr.connected_labels(len(mesh.F), mesh.fa[sel], mesh.fb[sel])


def _facet_planes(N, tol_deg=1.0):
    """Number of distinct facet planes among unit normals N (greedy, in order, within tol_deg)."""
    reps = np.zeros((0, 3))
    c = math.cos(math.radians(tol_deg))
    for nv in N:
        if not len(reps) or float((reps @ nv).max()) < c:
            reps = np.vstack([reps, nv])
    return len(reps)


def _local_axes(mesh, edges, C):
    """Dominant axis directions of a set of adjacency edges: on a cylinder, the cross product of two neighbouring
    faces' normals is the axis. Directions (sign-free) of the edges whose dihedral is >= holes.axis_min_dihedral_deg,
    clustered greedily within holes.axis_cluster_deg; at most holes.axis_max_candidates directions with at least
    holes.axis_min_support edges each, the best supported first (deterministic)."""
    e = edges[mesh.dih[edges] >= float(C["axis_min_dihedral_deg"])]
    if len(e) < int(C["axis_min_support"]):
        return []
    u = np.cross(mesh.N[mesh.fa[e]], mesh.N[mesh.fb[e]])
    ln = np.linalg.norm(u, axis=1)
    u = u[ln > 1e-12] / ln[ln > 1e-12, None]
    cmin = math.cos(math.radians(float(C["axis_cluster_deg"])))
    rem = np.arange(len(u))
    out = []
    while len(rem) >= int(C["axis_min_support"]) and len(out) < int(C["axis_max_candidates"]):
        centres = rem[np.linspace(0, len(rem) - 1, min(len(rem), 400)).astype(np.int64)]
        G = np.abs(u[centres] @ u[rem].T) >= cmin
        cnt = G.sum(axis=1)
        k = int(np.argmax(cnt))
        if cnt[k] < int(C["axis_min_support"]):
            break
        mem = rem[G[k]]
        ref = u[centres[k]]
        sg = np.sign(u[mem] @ ref)
        ax = (u[mem] * np.where(sg == 0, 1.0, sg)[:, None]).sum(axis=0)
        out.append(ax / np.linalg.norm(ax))
        rem = rem[~G[k]]
    return out


def _merge(cyls, C):
    """Concave cylinders that are pieces of one wall (axes within holes.merge_angle_deg and holes.merge_axis_mm, radii
    within holes.merge_radius_mm, axial extents overlapping or within holes.merge_gap_mm) merged into one: the larger
    piece's axis, the area-weighted radius, the union of the extents, the summed span (at most 360)."""
    order = sorted(range(len(cyls)), key=lambda i: -cyls[i]["area"])
    used = [False] * len(cyls)
    out = []
    ca = math.cos(math.radians(float(C["merge_angle_deg"])))
    for i in order:
        if used[i]:
            continue
        used[i] = True
        base = dict(cyls[i])
        d, p = base["dir"], base["point"]
        lo, hi = base["t_range"]
        area, rsum, span = base["area"], base["radius"] * base["area"], base["span_deg"]
        for j in order:
            if used[j]:
                continue
            c = cyls[j]
            if abs(float(c["dir"] @ d)) < ca or abs(c["radius"] - base["radius"]) > float(C["merge_radius_mm"]):
                continue
            v = c["point"] - p
            if float(np.linalg.norm(v - d * float(v @ d))) > float(C["merge_axis_mm"]):
                continue
            tc = float(v @ d)
            half = 0.5 * c["length"]
            clo, chi = tc - half, tc + half
            if max(lo, clo) - min(hi, chi) > float(C["merge_gap_mm"]):
                continue
            used[j] = True
            lo, hi = min(lo, clo), max(hi, chi)
            area += c["area"]
            rsum += c["radius"] * c["area"]
            span += c["span_deg"]
            base["pass"] = base["pass"] + "+" + c["pass"]
        mid = 0.5 * (lo + hi)
        base["point"] = p + d * mid
        base["t_range"] = (lo - mid, hi - mid)
        base["length"] = hi - lo
        base["radius"] = rsum / area
        base["area"] = area
        base["span_deg"] = min(360.0, span)
        out.append(base)
    return out


def find_concave_cylinders(V, F, C):
    """Concave cylinders of a tessellation (module docstring, passes 1-3, then the merge of pieces): [{"radius", "dir",
    "point", "t_range", "length", "span_deg", "facet_step_deg", "max_residual", "area", "pass"}]."""
    rule = pr.HOLE_RULE
    mesh = pr.Mesh(V, F)
    n = len(mesh.F)
    used = np.zeros(n, bool)
    out = []
    sin_t = math.sin(math.radians(float(C["trim_perp_deg"])))

    def take(c, tag):
        used[c["faces"]] = True
        if c["concave"]:
            c = dict(c)
            c["pass"] = tag
            out.append(c)

    def accept(c, facets):
        if c is None:
            return False
        if not facets:
            return True
        if not c["concave"] or c["span_deg"] < float(C["facet_min_span_deg"]):
            return False
        return _facet_planes(mesh.N[c["faces"]]) >= int(C["facet_min_facets"])

    def trimmed(fi, edge_ok, facets, tag):
        """The faces of component fi perpendicular (within trim_perp_deg) to each of its dominant local axes, each
        connected piece tested as a cylinder."""
        inside = np.zeros(n, bool)
        inside[fi] = True
        edges = np.nonzero(edge_ok & inside[mesh.fa] & inside[mesh.fb])[0]
        for ax in _local_axes(mesh, edges, C):
            keep = inside & ~used
            keep[fi] &= np.abs(mesh.N[fi] @ ax) <= sin_t
            k = int(keep.sum())
            if k < rule["min_facets"]:
                continue
            for g in _groups(_sub_labels(mesh, keep, edge_ok), np.nonzero(keep)[0]):
                c = _patch(mesh, g, rule)
                if accept(c, facets):
                    take(c, tag)

    # pass 1: the reference's finder unchanged
    for c in pr.find_cylinders(mesh, rule):
        take(c, "smooth")
    # pass 2: the smooth components that failed, cut along their dominant local axes
    smooth = mesh.dih < rule["smooth_deg"]
    lab = mesh.components(rule["smooth_deg"], strict=True)
    for fi in _groups(lab, np.nonzero(~used)[0]):
        if len(fi) >= rule["min_facets"]:
            trimmed(fi, smooth, False, "trimmed")
    # pass 3: coarse facets (the faces not used yet, linked across dihedrals <= facet_link_deg), as one piece or cut
    link = mesh.dih <= float(C["facet_link_deg"])
    free = ~used
    lab3 = _sub_labels(mesh, free, link)
    for fi in _groups(lab3, np.nonzero(free)[0]):
        if len(fi) < rule["min_facets"] or used[fi].any():
            continue
        c = _patch(mesh, fi, rule)
        if accept(c, True):
            take(c, "facets")
        elif c is None:
            trimmed(fi, link, True, "facets")
    for c in out:
        c.pop("faces", None)
    return _merge(out, C)


def submitted_cylinders(ctx, s):
    """The concave cylinders of a submitted part in the submission frame (cached on ctx.cache)."""
    st = ctx.cache.setdefault("holes_cylinders", {})
    if s["sid"] not in st:
        g = submission_geo(ctx, s)
        st[s["sid"]] = [] if g is None else find_concave_cylinders(g.V, g.F, section(ctx, "holes", HOLES_DEFAULTS))
    return st[s["sid"]]


# ============================================================================ matching
def step_scores(ctx, s, key, T):
    """[(step id, score, detail)] of every graded step of design `key` for part s at pose T (module docstring)."""
    C = section(ctx, "holes", HOLES_DEFAULTS)
    steps = graded_steps(ctx, key)
    if not steps:
        return []
    T = np.asarray(T, float)
    ob = C["offset_mm"][ctx.ref.designs[key]["size_class"]]
    rb, ab = C["radius_mm"], C["angle_deg"]
    cap = float(C["match_cap_factor"]) * float(ob[1])
    clip = float(C["cost_clip"])
    cyl = submitted_cylinders(ctx, s) if s is not None else []
    if not cyl:
        return [(st["id"], 0.0, {"matched": False}) for st, _r in steps]
    sd = apply_dir(T, np.array([c["dir"] for c in cyl]))
    sd /= np.linalg.norm(sd, axis=1, keepdims=True)
    sp = apply(T, np.array([c["point"] for c in cyl]))
    s_lo = np.array([c["t_range"][0] for c in cyl])
    s_hi = np.array([c["t_range"][1] for c in cyl])
    s_r = np.array([c["radius"] for c in cyl])
    ns, nc = len(steps), len(cyl)
    off = np.zeros((ns, nc))
    ang = np.zeros((ns, nc))
    dr = np.zeros((ns, nc))
    for i, (st, _rg) in enumerate(steps):
        a, b = np.asarray(st["ends_mm"][0], float), np.asarray(st["ends_mm"][1], float)
        dref = np.asarray(st["axis_dir"], float)
        mid = 0.5 * (a + b)
        cosang = np.clip(np.abs(sd @ dref), 0.0, 1.0)
        ang[i] = np.degrees(np.arccos(cosang))
        v = mid - sp
        tm = np.einsum("ij,ij->i", v, sd)
        radial = np.linalg.norm(v - sd * tm[:, None], axis=1)
        ta = np.einsum("ij,ij->i", a - sp, sd)
        tb = np.einsum("ij,ij->i", b - sp, sd)
        lo, hi = np.minimum(ta, tb), np.maximum(ta, tb)
        gap = np.maximum(0.0, np.maximum(lo, s_lo) - np.minimum(hi, s_hi))
        off[i] = np.sqrt(radial ** 2 + gap ** 2)
        dr[i] = np.abs(s_r - float(st["radius_mm"]))
    cost = (np.minimum(off / float(ob[1]), clip) + np.minimum(dr / float(rb[1]), clip) +
            np.minimum(ang / float(ab[1]), clip))
    big = 1e6
    cost = np.where(off <= cap, cost, big)
    match = {i: j for i, j in hungarian(cost) if cost[i, j] < big}
    out = []
    for i, (st, rg) in enumerate(steps):
        j = match.get(i)
        if j is None:
            out.append((st["id"], 0.0, {"matched": False}))
            continue
        sc = band(off[i, j], ob) * (band(dr[i, j], rb) if rg else 1.0) * band(ang[i, j], ab)
        out.append((st["id"], float(sc), {"matched": True, "offset_mm": float(off[i, j]), "dr_mm": float(dr[i, j]),
                                          "angle_deg": float(ang[i, j]), "radius_graded": bool(rg),
                                          "pass": cyl[j]["pass"]}))
    return out


def pose_preference(ctx, s, key, T):
    """The fit's feature callback: mean score of the part's graded hole steps and flat faces at pose T (flat faces on
    the ranking subset), None when the design has none."""
    from .items_planes import plane_scores
    if submission_geo(ctx, s) is None:
        return 0.0
    h = [x[1] for x in step_scores(ctx, s, key, T)]
    p = [x[1] for x in plane_scores(ctx, s, key, T, full=False)]
    if not h and not p:
        return None
    return float(sum(h) + sum(p)) / float(len(h) + len(p))


def item(ctx):
    """Scoring item H (module docstring): {"score", "per_part", "diagnostics"}."""
    per, tot, n, diags = {}, 0.0, 0, []
    misses = []
    for tag in ctx.ref.tags:
        key = ctx.ref.family_of[tag]
        steps = graded_steps(ctx, key)
        n += len(steps)
        if not steps:
            per[tag] = {"score": None, "steps": 0, "design": key}
            continue
        s, T = body_pose(ctx, tag)
        if s is None or T is None or submission_geo(ctx, s) is None:
            per[tag] = {"score": 0.0, "steps": len(steps), "design": key,
                        "why": "missing" if s is None else "not fitted"}
            continue
        rows = step_scores(ctx, s, key, T)
        tot += sum(r[1] for r in rows)
        per[tag] = {"score": float(np.mean([r[1] for r in rows])), "steps": len(steps), "design": key,
                    "found_cylinders": len(submitted_cylinders(ctx, s)),
                    "per_step": {r[0]: dict(r[2], score=r[1]) for r in rows}}
        misses.extend((tag, r[0]) for r in rows if not r[2]["matched"])
    H = tot / n if n else 0.0
    if misses:
        diags.append("Holes: %d of %d graded hole steps have no matching bore (%s)." % (
            len(misses), n, ", ".join("%s %s" % m for m in misses[:6])))
    ctx.stats["holes_graded_steps"] = float(n)
    ctx.stats["holes_unmatched_steps"] = float(len(misses))
    return {"score": float(H), "per_part": per, "diagnostics": diags}


__all__ = ["HOLES_DEFAULTS", "find_concave_cylinders", "item", "pose_preference", "step_scores", "submitted_cylinders"]
