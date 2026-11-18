"""Scoring item S, shape (weight 0.25), and what the three surface items share: the reference features
(features.json), the "can be seen" marks (seen.json) and the per-Body weights.

Per Body (each of the 13 reference tags; a Body is scored against its family's design at the pose the whole-strut
alignment chose among its fit candidates, ctx.alignment["T_fit"]):

    s = w_surface * band(e; the design's size-class band) + w_iou * rising(IoU; iou)
    e = max(p90(reference -> submission), p90(submission -> reference))

  reference -> submission  the design's dense samples (geom.Geo.dense(): >= 20 points / mm2, the grader's fixed seed)
                           that are seen, each to the submitted surface (tangent-disk distance, sampling.disk_mm)
  submission -> reference  the submitted part's dense samples, each to the reference surface; only the samples whose
                           nearest reference sample is seen count
  IoU                      manifold3d exact booleans on the whole part (fit.pose_measures, cached per pose)
  distances are capped at shape.distance_cap_mm (3 mm, three times the widest zero point; the reported p90 too)

S = sum(w_b s_b) / sum(w_b) over the 13 reference tags; w_b = 1 + graded hole steps + graded flat faces of the Body's
design (features.json with seen.json; steps, not holes, are counted). A tag without a credited,
fitted part scores 0 and keeps its weight.

The fit callback pose_score(ctx, solid, design_key, T_fit) is the same score on a fixed subset (shape.n_rank points
each way, the query samples and their marks): it ranks the candidate poses (fit.py); the item itself is evaluated at
the chosen pose on all samples.

Without tests/reference/seen.json every sample, hole step and flat face counts as seen (seen.json holds the marks; the
layout read here is designs.<key>.samples.{dense,query}.{file,n,points_hash} with numpy.packbits(bitorder
"little") bitmaps in the grader's sampling order, hole_steps[].{id,position_gradable,radius_gradable},
planes[].{id,gradable}). A bitmap whose length or points hash does not match the grader's own samples is a broken
task (TaskError), never a zero.

Configuration: section "shape" of grader_config.json, overlaid on SHAPE_DEFAULTS (defaults_used lists what the file
did not give). Imports numpy and the package only (scipy and manifold3d through geom and fit).
"""
import os

import numpy as np

from . import TaskError
from .geom import apply, band, even_subset, inv, rising, sample_surface
from .recognise import submission_geo

SHAPE_DEFAULTS = {
    "surface_p90_mm": {"small": [0.12, 0.60], "large": [0.20, 1.00]},
    "iou": [0.40, 0.95],
    "w_surface": 0.7,
    "w_iou": 0.3,
    "quantile": 0.9,
    "n_rank": 6000,
    "distance_cap_mm": 3.0,
}


# ============================================================================ configuration
def section(ctx, name, defaults):
    """The item section `name` of the configuration overlaid on `defaults` (cached on ctx.cache)."""
    store = ctx.cache.setdefault("item_sections", {})
    if name not in store:
        have = ctx.cfg.get(name) or {}
        out = {k: have.get(k, v) for k, v in defaults.items()}
        store[name] = out
    return store[name]


def defaults_used(cfg):
    """'section.key' of every item default that grader_config.json (cfg) does not give (the freeze checks it empty)."""
    from .items_holes import HOLES_DEFAULTS
    from .items_planes import PLANES_DEFAULTS
    out = []
    for name, d in (("shape", SHAPE_DEFAULTS), ("holes", HOLES_DEFAULTS), ("planes", PLANES_DEFAULTS)):
        have = (cfg or {}).get(name) or {}
        out.extend("%s.%s" % (name, k) for k in d if k not in have)
    return out


# ============================================================================ reference features and marks
def _store(ctx):
    return ctx.cache.setdefault("surface_items", {})


def features_of(ctx, key):
    """features.json's table of design `key` (holes.steps, planes); a missing table is a broken task."""
    f = ctx.ref.features
    if f is None or key not in (f.get("families") or {}):
        raise TaskError("tests/reference/features.json has no table for design %s" % key)
    return f["families"][key]


def dense_with_faces(ctx, key):
    """(points, normals, KD-tree, face index) of design `key`'s dense samples: the very samples of Geo.dense() (same
    function, count and seed), with the triangle each point lies on."""
    st = _store(ctx).setdefault("dense_faces", {})
    if key not in st:
        d = ctx.ref.designs[key]
        D, N, tree = d["geo"].dense()
        P, _N, fid = sample_surface(d["V"], d["F"], len(D), d["geo"].seed)
        if len(P) != len(D) or not np.array_equal(P, D):
            raise TaskError("the dense samples of %s cannot be reproduced with their face indices" % key)
        st[key] = (D, N, tree, fid)
    return st[key]


def _bits(ctx, rec, n, what):
    path = os.path.join(ctx.ref.dir, rec["file"])
    if not os.path.isfile(path):
        raise TaskError("seen.json names a missing bitmap %s" % rec["file"])
    with open(path, "rb") as fh:
        raw = np.frombuffer(fh.read(), np.uint8)
    if int(rec.get("n", -1)) != n or len(raw) * 8 < n:
        raise TaskError("the %s bitmap holds %s marks, the grader has %d samples" % (what, rec.get("n"), n))
    return np.unpackbits(raw, bitorder="little")[:n].astype(bool)


def marks(ctx, key):
    """The marks of design `key`: {"dense": bool (n_dense,), "query": bool (n_query,), "steps": {id: (position graded,
    radius graded)}, "planes": {id: graded}, "source": "seen.json" | "all (no seen.json)"}."""
    st = _store(ctx).setdefault("marks", {})
    if key in st:
        return st[key]
    from .geom import points_hash
    feats = features_of(ctx, key)
    D = ctx.ref.designs[key]["geo"].dense()[0]
    Q = ctx.ref.designs[key]["geo"].query()[0]
    steps = {s["id"]: (True, True) for s in feats["holes"]["steps"]}
    planes = {p["id"]: True for p in feats["planes"]}
    seen = ctx.ref.seen
    if seen is None:
        out = {"dense": np.ones(len(D), bool), "query": np.ones(len(Q), bool), "steps": steps, "planes": planes,
               "source": "all (no seen.json)"}
        st[key] = out
        return out
    rec = (seen.get("designs") or {}).get(key)
    if rec is None:
        raise TaskError("seen.json has no marks for design %s" % key)
    out = {"source": "seen.json"}
    for kind, P in (("dense", D), ("query", Q)):
        r = rec["samples"][kind]
        if r.get("points_hash") and r["points_hash"] != points_hash(P):
            raise TaskError("seen.json's %s samples of %s are not the grader's (points hash)" % (kind, key))
        out[kind] = _bits(ctx, r, len(P), "%s %s" % (key, kind))
    for h in rec.get("hole_steps", []):
        if h["id"] in steps:
            steps[h["id"]] = (bool(h.get("position_gradable", True)), bool(h.get("radius_gradable", True)))
    for p in rec.get("planes", []):
        if p["id"] in planes:
            planes[p["id"]] = bool(p.get("gradable", True))
    out["steps"], out["planes"] = steps, planes
    st[key] = out
    return out


def graded_steps(ctx, key):
    """The hole steps of design `key` that are graded (position gradable), each with its radius flag."""
    m = marks(ctx, key)
    return [(s, m["steps"][s["id"]][1]) for s in features_of(ctx, key)["holes"]["steps"] if m["steps"][s["id"]][0]]


def graded_planes(ctx, key):
    m = marks(ctx, key)
    return [p for p in features_of(ctx, key)["planes"] if m["planes"][p["id"]]]


def body_weight(ctx, key):
    """w = 1 + graded hole steps + graded flat faces of design `key`."""
    return 1.0 + len(graded_steps(ctx, key)) + len(graded_planes(ctx, key))


def body_pose(ctx, tag):
    """(solid record, T_fit) of the credited, fitted part carrying `tag`, or (None, None)."""
    s = ctx.parts.get(tag)
    if s is None or ctx.alignment is None:
        return None, None
    T = ctx.alignment["T_fit"].get(s["sid"])
    f = ctx.fits.get(s["sid"])
    if T is None or f is None or not f["candidates"]:
        return s, None
    return s, np.asarray(T, float)


def outward_normals(g):
    """The submitted tessellation's dense samples (points, unit normals pointing out of the material, tree)."""
    P, N, tree = g.dense()
    if getattr(g, "_outward_sign", None) is None:
        tri = g.V[g.F]
        vol = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum()) / 6.0
        g._outward_sign = 1.0 if vol >= 0 else -1.0
    return P, (N if g._outward_sign > 0 else -N), tree


def disk_distances(Q, D, N, tree, disk_mm, cap_mm=np.inf):
    """(tangent-disk distance, nearest sample index) of points Q to the surface sampled by (D, N, tree): the formula
    of geom.surface_distances, also returning the index. A point with no sample within cap_mm gets cap_mm and index
    -1 (the KD search is bounded: far points are slow to search and every distance beyond a band's zero point scores
    the same)."""
    if tree is None or len(Q) == 0:
        return np.full(len(Q), cap_mm if np.isfinite(cap_mm) else np.inf), np.full(len(Q), -1, np.int64)
    d, idx = tree.query(Q, distance_upper_bound=cap_mm)
    far = ~np.isfinite(d)
    idx = np.where(far, -1, idx)
    j = np.where(far, 0, idx)
    v = Q - D[j]
    dd = np.where(far, 0.0, d)
    dn = np.abs(np.einsum("ij,ij->i", v, N[j]))
    dl = np.sqrt(np.maximum(0.0, dd * dd - dn * dn))
    out = np.sqrt(dn * dn + np.maximum(0.0, dl - disk_mm) ** 2)
    return np.where(far, cap_mm, np.minimum(out, cap_mm)), idx


# ============================================================================ the shape score
def measures(ctx, s, key, T, full=True):
    """{"p90_ref_to_sub_mm", "p90_sub_to_ref_mm", "iou", "n_ref", "n_sub"} of a part at pose T (submission ->
    design). full: every dense sample; else shape.n_rank of the query samples each way (the candidate ranking)."""
    from .fit import pose_measures
    C = section(ctx, "shape", SHAPE_DEFAULTS)
    q = float(C["quantile"])
    disk = float(ctx.cfg["sampling"]["disk_mm"])
    d = ctx.ref.designs[key]
    g = submission_geo(ctx, s)
    m = marks(ctx, key)
    Dr, Nr, tree_r = d["geo"].dense()
    if full:
        Pr = Dr[m["dense"]]
        Ps = g.dense()[0]
    else:
        n = int(C["n_rank"])
        Pr = even_subset(d["geo"].query()[0][m["query"]], n)
        Ps = even_subset(g.query()[0], n)
    cap = float(C["distance_cap_mm"])
    a = disk_distances(apply(inv(T), Pr), *g.dense(), disk, cap)[0] if len(Pr) else np.zeros(0)
    b, idx = disk_distances(apply(T, Ps), Dr, Nr, tree_r, disk, cap)
    # a submitted point with no reference sample within the cap counts (material far from the part is wrong
    # whatever is seen); the others count when their nearest reference sample is seen
    b = b[(idx < 0) | m["dense"][np.maximum(idx, 0)]] if len(b) else b
    return {"p90_ref_to_sub_mm": float(np.quantile(a, q)) if len(a) else 0.0,
            "p90_sub_to_ref_mm": float(np.quantile(b, q)) if len(b) else 0.0,
            "iou": float(pose_measures(ctx, s, key, T)[2]), "n_ref": int(len(a)), "n_sub": int(len(b))}


def score_of(ctx, key, ms):
    C = section(ctx, "shape", SHAPE_DEFAULTS)
    e = max(ms["p90_ref_to_sub_mm"], ms["p90_sub_to_ref_mm"])
    b = C["surface_p90_mm"][ctx.ref.designs[key]["size_class"]]
    return float(C["w_surface"]) * band(e, b) + float(C["w_iou"]) * rising(ms["iou"], C["iou"])


def pose_score(ctx, s, key, T):
    """The fit callback (fit.py: shape_score(ctx, solid, design_key, T_fit) -> [0, 1]): the shape score on the
    ranking subset."""
    if submission_geo(ctx, s) is None:
        return 0.0
    return score_of(ctx, key, measures(ctx, s, key, np.asarray(T, float), full=False))


def item(ctx):
    """Scoring item S (module docstring): {"score", "per_part", "diagnostics"}."""
    per, num, den, diags = {}, 0.0, 0.0, []
    for tag in ctx.ref.tags:
        key = ctx.ref.family_of[tag]
        w = body_weight(ctx, key)
        den += w
        s, T = body_pose(ctx, tag)
        if s is None or T is None or submission_geo(ctx, s) is None:
            per[tag] = {"score": 0.0, "weight": w, "design": key,
                        "why": "missing" if s is None else "not fitted"}
            continue
        ms = measures(ctx, s, key, T, full=True)
        sc = score_of(ctx, key, ms)
        num += w * sc
        rec = {"score": sc, "weight": w, "design": key, "chosen_candidate": ctx.alignment["chosen"].get(s["sid"]),
               "candidates": len(ctx.fits[s["sid"]]["candidates"])}
        rec.update(ms)
        per[tag] = rec
    S = num / den if den > 0 else 0.0
    low = sorted((v["score"], t) for t, v in per.items())[:3]
    for sc, t in low:
        if sc < 0.9:
            v = per[t]
            if "why" in v:
                diags.append("Shape: %s is %s and scores 0." % (t, v["why"]))
            else:
                diags.append("Shape: %s scores %.2f (surface p90 %.3f / %.3f mm, IoU %.3f)." % (
                    t, sc, v["p90_ref_to_sub_mm"], v["p90_sub_to_ref_mm"], v["iou"]))
    ctx.stats["shape_weight_total"] = float(den)
    return {"score": float(S), "per_part": per, "diagnostics": diags}


__all__ = ["SHAPE_DEFAULTS", "body_pose", "body_weight", "defaults_used", "dense_with_faces", "disk_distances",
           "features_of", "graded_planes", "graded_steps", "item", "marks", "measures", "outward_normals",
           "pose_score", "score_of", "section"]
