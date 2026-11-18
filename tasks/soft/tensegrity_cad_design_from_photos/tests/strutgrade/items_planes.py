"""Scoring item K, flat-face features (weight 0.15).

Reference side: the flat faces of features.json (rule (a) + (c) + (b) of features.json rules: 402 per strut), each a
set of triangles of its design with an outward normal; graded = gradable in seen.json (all without it). A face's
points are the design's dense samples (the shape item's samples: geom.Geo.dense(), fixed seed) that lie on its
triangles and are seen.

Submission side: nothing is recognised and nothing is paired. At a pose T (submission -> design), for every point p of
a graded face with outward normal n, the nearest dense sample of the submitted part whose outward normal is within
planes.normal_deg (30 deg) of n, searched within planes.search_mm (5 mm): the tangent-disk distance to it (the shape
item's distance, sampling.disk_mm); planes.missing_mm (5 mm) when there is none. The face's error e_f = the median of
its points' distances; its score = band(e_f; the design's size-class band). K = mean over all graded faces of the
strut; the faces of a missing or unfitted Body score 0.

What this reads: a missing rib -> its top's points find the floor below (e_f = rib height); a cavity
not cut -> its floor's points find the outer face above (e_f = depth); a window not opened -> its side walls' points
find no face of that orientation nearby; a face misplaced by d -> e_f = d; an honest model with every face off by d
-> band(d) on every face, the oracle 1.

The nearest compatible sample: the k = planes.k_nn nearest samples are tried in order of distance (the first
compatible one is the nearest compatible one); a point with none among them is searched again in a KD-tree of only
the compatible samples (built once per submitted part and face orientation).

plane_scores(ctx, s, key, T, full) gives [(face id, score, e_f)]; full=False keeps at most
planes.rank_points_per_plane evenly spaced points of each face (the candidate preference, items_holes).

Configuration: section "planes" of grader_config.json over PLANES_DEFAULTS.
"""
import math

import numpy as np

from .geom import apply, apply_dir, band, even_subset, inv
from .items_shape import body_pose, dense_with_faces, graded_planes, marks, outward_normals, section
from .recognise import submission_geo

PLANES_DEFAULTS = {
    "band_mm": {"small": [0.12, 0.60], "large": [0.20, 1.00]},
    "normal_deg": 30.0,
    "search_mm": 5.0,
    "missing_mm": 5.0,
    "k_nn": 16,
    "rank_points_per_plane": 48,
}


def plane_points(ctx, key):
    """{face id: indices of the design's dense samples on that face that are seen} (cached)."""
    st = ctx.cache.setdefault("planes_points", {})
    if key not in st:
        D, _N, _t, fid = dense_with_faces(ctx, key)
        seen = marks(ctx, key)["dense"]
        nf = len(ctx.ref.designs[key]["F"])
        face_plane = np.full(nf, -1, np.int64)
        planes = graded_planes(ctx, key)
        for k, p in enumerate(planes):
            face_plane[np.asarray(p["triangles"], np.int64)] = k
        lab = face_plane[fid]
        order = np.argsort(lab, kind="stable")
        out = {}
        ls = lab[order]
        for k, p in enumerate(planes):
            lo, hi = np.searchsorted(ls, k, "left"), np.searchsorted(ls, k, "right")
            idx = np.sort(order[lo:hi])
            out[p["id"]] = idx[seen[idx]]
        st[key] = out
    return st[key]


def _compatible_tree(ctx, s, n_sub, cosmax):
    """KD-tree of the submitted part's dense samples whose outward normal is within the cone about n_sub."""
    from scipy.spatial import cKDTree
    st = ctx.cache.setdefault("planes_trees", {})
    ck = (s["sid"], tuple(np.round(n_sub, 9)))
    if ck not in st:
        P, N, _t = outward_normals(submission_geo(ctx, s))
        sel = np.nonzero(N @ n_sub >= cosmax)[0]
        st[ck] = (sel, cKDTree(P[sel]) if len(sel) else None)
    return st[ck]


def face_distances(ctx, s, Q, n_sub, C):
    """Distance of each point Q (submission frame) to the nearest compatible submitted sample (module docstring)."""
    disk = float(ctx.cfg["sampling"]["disk_mm"])
    search, miss = float(C["search_mm"]), float(C["missing_mm"])
    cosmax = math.cos(math.radians(float(C["normal_deg"])))
    P, N, tree = outward_normals(submission_geo(ctx, s))
    out = np.full(len(Q), miss)
    if tree is None or len(Q) == 0:
        return out
    k = int(C["k_nn"])
    d, idx = tree.query(Q, k=k, distance_upper_bound=search)
    if k == 1:
        d, idx = d[:, None], idx[:, None]
    valid = np.isfinite(d)
    idc = np.where(valid, idx, 0)
    ok = valid & (N[idc] @ n_sub >= cosmax)
    has = ok.any(axis=1)
    first = np.argmax(ok, axis=1)
    rows = np.nonzero(has)[0]
    j = idc[rows, first[rows]]
    out[rows] = _disk(Q[rows], P[j], N[j], disk)
    # rows with no compatible sample among the k nearest (and not all k beyond the search radius)
    rest = np.nonzero(~has & valid[:, -1])[0]
    if len(rest):
        sel, t2 = _compatible_tree(ctx, s, n_sub, cosmax)
        if t2 is not None:
            d2, i2 = t2.query(Q[rest], distance_upper_bound=search)
            got = np.isfinite(d2)
            if got.any():
                jj = sel[i2[got]]
                out[rest[got]] = _disk(Q[rest[got]], P[jj], N[jj], disk)
    return np.minimum(out, miss)


def _disk(Q, D, N, disk):
    v = Q - D
    d2 = np.einsum("ij,ij->i", v, v)
    dn = np.abs(np.einsum("ij,ij->i", v, N))
    dl = np.sqrt(np.maximum(0.0, d2 - dn * dn))
    return np.sqrt(dn * dn + np.maximum(0.0, dl - disk) ** 2)


def plane_scores(ctx, s, key, T, full=True):
    """[(face id, score, e_f mm)] of every graded flat face of design `key` for part s at pose T."""
    C = section(ctx, "planes", PLANES_DEFAULTS)
    planes = graded_planes(ctx, key)
    if not planes:
        return []
    b = C["band_mm"][ctx.ref.designs[key]["size_class"]]
    D = dense_with_faces(ctx, key)[0]
    pts = plane_points(ctx, key)
    Tinv = inv(np.asarray(T, float))
    out = []
    nrank = int(C["rank_points_per_plane"])
    for p in planes:
        idx = pts[p["id"]]
        if not full:
            idx = even_subset(idx, nrank)
        if len(idx) == 0:
            out.append((p["id"], 0.0, float(C["missing_mm"])))
            continue
        n_sub = apply_dir(Tinv, np.asarray(p["normal"], float))
        n_sub = n_sub / np.linalg.norm(n_sub)
        dist = face_distances(ctx, s, apply(Tinv, D[idx]), n_sub, C)
        e = float(np.median(dist))
        out.append((p["id"], float(band(e, b)), e))
    return out


def item(ctx):
    """Scoring item K (module docstring): {"score", "per_part", "diagnostics"}."""
    per, tot, n, diags = {}, 0.0, 0, []
    worst = []
    for tag in ctx.ref.tags:
        key = ctx.ref.family_of[tag]
        planes = graded_planes(ctx, key)
        n += len(planes)
        if not planes:
            per[tag] = {"score": None, "planes": 0, "design": key}
            continue
        s, T = body_pose(ctx, tag)
        if s is None or T is None or submission_geo(ctx, s) is None:
            per[tag] = {"score": 0.0, "planes": len(planes), "design": key,
                        "why": "missing" if s is None else "not fitted"}
            continue
        rows = plane_scores(ctx, s, key, T, full=True)
        tot += sum(r[1] for r in rows)
        per[tag] = {"score": float(np.mean([r[1] for r in rows])), "planes": len(planes), "design": key,
                    "per_plane": {r[0]: {"score": r[1], "e_mm": r[2]} for r in rows}}
        worst.extend((r[1], tag, r[0], r[2]) for r in rows if r[1] < 1.0)
    K = tot / n if n else 0.0
    worst.sort()
    if worst:
        diags.append("Flat faces: %d of %d graded faces below full marks (worst: %s)." % (
            len(worst), n, ", ".join("%s %s %.2f mm" % (t, f, e) for _sc, t, f, e in worst[:5])))
    ctx.stats["planes_graded"] = float(n)
    ctx.stats["planes_below_full"] = float(len(worst))
    return {"score": float(K), "per_part": per, "diagnostics": diags}


__all__ = ["PLANES_DEFAULTS", "face_distances", "item", "plane_points", "plane_scores"]
