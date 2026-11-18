"""Per-part fit with candidate poses.

Every credited tagged part is registered rigidly onto its family's reference design by a global search that ignores
where the part sits in the document:

  starts   the 24 principal-axis starts (every axis permutation and proper sign choice of the part's exact B-rep
           inertia frame onto the design's mesh inertia frame, centroids matched); for a symmetric design (cyclic or
           continuous, families.json) also the part's own most distinct principal axis (fit.spin_axes "distinct"; "all"
           takes the three), in both senses, onto the design's symmetry axis, turned about that axis in
           fit.spin_step_deg (15 deg) steps
  coarse   trimmed point-to-plane ICP of fit.n_coarse submission samples (fit.coarse_iters) from every start, onto the
           design's query samples, keeping the fit.coarse_trim (0.9) nearest pairs
  refine   the distinct coarse poses whose trimmed RMS is within refine_rms_factor x best + refine_rms_margin_mm, at
           most fit.refine (fit.refine_symmetric for a symmetric design: one per spin start), ICP on fit.n_fine samples
           onto the design's dense samples (at most fit.fine_iters, stopping below fit.fine_tol) with the share
           fit.trim of the pairs; fit.trim is 1.0 in the frozen configuration: the refinement uses every pair (a
           trimmed refinement pulls an honest uniformly offset part 0.25-0.56 mm off); a part whose B-rep
           vertex set lands one to one within fit.exact_snap_mm on the design's vertices is snapped by Kabsch on them
  score    every distinct refined pose gets its shape score from the shape-score callback (the built-in
           minimal_shape_score unless the shape item registers its own) and its detail measure (the larger of the two
           one-way p99 surface distances: the small features that the shape score's p90 and IoU do not see, such as
           the phase of a winch's holes); the candidates are the poses whose shape score is within fit.window (0.05)
           of the best, ordered by (shape rounded to 1e-3 desc, detail, rms, start), at most fit.max_candidates
  prefer   with a feature-score callback (the part's own holes and flat faces), only the candidates within
           fit.feature_tie (0.01) of the best feature score stay tied; the whole-strut alignment chooses among the tied
           ones the pose nearest the consensus, the earliest of equally near ones (align.py)

T_fit maps the submission's world frame to the design file's frame. The seeds are fixed (detenv.seed_for); the starts,
sorts and tie-breaks are in a fixed order, so a document fits bit for bit the same twice.

Callback signatures (strutgrade.run_base passes them):
    shape_score(ctx, solid, design_key, T_fit) -> float in [0, 1]
    feature_score(ctx, solid, design_key, T_fit) -> float in [0, 1] or None (no opinion)
"""
import math
import time

import numpy as np

from .geom import (apply, band, box_corners, eigen, even_subset, icp, inv, iou, kabsch, make_T, one_way,
                   pose_distance, principal_starts, rising, rotation_about, rotation_between)
from .recognise import submission_geo


def pose_measures(ctx, s, key, T):
    """(p90, p99, IoU) of a pose: the larger of the two one-way quantiles of the tangent-disk surface distance
    (shape_minimal.n_query samples each way, no visibility marks) and the volume IoU; cached per pose."""
    cache = ctx.cache.setdefault("fit_measures", {})
    ck = (s["sid"], key, np.asarray(T, float).round(12).tobytes())
    if ck in cache:
        return cache[ck]
    S = ctx.cfg["shape_minimal"]
    d = ctx.ref.designs[key]
    g = submission_geo(ctx, s)
    disk = float(ctx.cfg["sampling"]["disk_mm"])
    n = int(S["n_query"])
    qs = even_subset(g.query()[0], n)
    qr = even_subset(d["geo"].query()[0], n)
    a = one_way(apply(T, qs), d["geo"].dense(), disk)
    b = one_way(apply(inv(T), qr), g.dense(), disk)
    out = (max(float(np.quantile(a, 0.9)), float(np.quantile(b, 0.9))),
           max(float(np.quantile(a, 0.99)), float(np.quantile(b, 0.99))),
           iou(g, T, d["geo"], np.eye(4)))
    cache[ck] = out
    return out


def minimal_shape_score(ctx, s, key, T):
    """The base's own shape score of a pose, used only to rank candidate poses while the shape item is not registered:
    w_surface band(p90 of pose_measures; the design's size-class band) + w_iou rising(IoU)."""
    S = ctx.cfg["shape_minimal"]
    d = ctx.ref.designs[key]
    if submission_geo(ctx, s) is None:
        return 0.0
    p90, _p99, v = pose_measures(ctx, s, key, T)
    b = S["surface_p90_mm"][d["size_class"]]
    return float(S["w_surface"]) * band(p90, b) + float(S["w_iou"]) * rising(v, S["iou"])


def _starts(s, d, F):
    """(T0 submission -> design, label) of every start (module docstring)."""
    out = [(T, "principal%02d" % i) for i, T in enumerate(principal_starts(s["com"], s["I"], d["com"], d["I"]))]
    sym = d["symmetry"]
    if sym["kind"] == "none":
        return out
    w, E = eigen(s["I"])
    ax, p = sym["axis_dir"], sym["axis_point"]
    # the design's centroid projected on its symmetry axis: the turns are about the axis line
    c_axis = p + ax * float((d["com"] - p) @ ax)
    n = int(round(360.0 / float(F["spin_step_deg"])))
    if F.get("spin_axes", "distinct") == "all":
        axes = [0, 1, 2]
    else:
        # the part's own principal axis whose moment stands furthest from the other two (its symmetry axis when it is
        # a body of revolution or of 3+-fold symmetry)
        gap = [abs(w[k] - 0.5 * (w[(k + 1) % 3] + w[(k + 2) % 3])) for k in range(3)]
        axes = [int(np.argmax(gap))]
    for k in axes:
        for sense in (1.0, -1.0):
            R0 = rotation_between(E[:, k], sense * ax)
            T0 = make_T(R0, c_axis - R0 @ s["com"])
            for j in range(n):
                out.append((rotation_about(c_axis, ax, math.radians(j * float(F["spin_step_deg"]))) @ T0,
                            "spin_a%d_%s_%03d" % (k, "p" if sense > 0 else "m", j)))
    return out


def _distinct(poses, pts, tol):
    """Indices of poses that differ from every earlier kept one by more than tol at the points pts."""
    keep = []
    for i, T in enumerate(poses):
        if all(pose_distance(T, poses[j], pts) > tol for j in keep):
            keep.append(i)
    return keep


def _snap(s, d, T, tol):
    """Kabsch on the vertex correspondences when the part's B-rep vertices land one to one on the design's vertices
    within tol under T; else None."""
    vs, vr = s.get("verts"), d["V"]
    if vs is None or len(vs) != len(vr) or len(vs) < 4:
        return None
    if "vtree" not in d:
        from scipy.spatial import cKDTree
        d["vtree"] = cKDTree(vr)
    dist, idx = d["vtree"].query(apply(T, vs))
    if float(dist.max()) > tol or len(np.unique(idx)) != len(idx):
        return None
    Ts = kabsch(vs, vr[idx])
    err = float(d["vtree"].query(apply(Ts, vs))[0].max())
    return Ts, err


def fit_part(ctx, s, key, shape_score=None, feature_score=None):
    """The fit of one credited part onto design `key` (module docstring). Returns the FitResult dict:
    {"sid", "tag", "family", "candidates": [{"T", "rms", "shape", "detail_p99_mm", "feature", "snapped", "snap_err",
     "start"}], "tied": [candidate indices still tied after the feature preference], "best": 0, "n_starts",
     "n_refined", "t_s"}."""
    t0 = time.time()
    F = ctx.cfg["fit"]
    shape_score = shape_score or minimal_shape_score
    d = ctx.ref.designs[key]
    g = submission_geo(ctx, s)
    if g is None:
        raise ValueError("the part cannot be tessellated")
    q = g.query()[0]
    coarse_pts = even_subset(q, int(F["n_coarse"]))
    fine_pts = even_subset(q, int(F["n_fine"]))
    D, N, tree = d["geo"].dense()
    Dc, Nc, treec = d["geo"].coarse()
    trim, bound = float(F["trim"]), float(F["bound_mm"])
    # the coarse stage keeps its own trim (robust to the far points of a wrong start); a configuration without the key
    # (the base's defaults) trims the coarse stage like the refinement
    coarse_trim = float(F.get("coarse_trim", F["trim"]))
    starts = _starts(s, d, F)
    coarse = []
    for i, (T0, lab) in enumerate(starts):
        Tk, rk = icp(coarse_pts, Dc, Nc, treec, T0, int(F["coarse_iters"]), coarse_trim, bound)
        coarse.append((rk, i, Tk, lab))
    coarse.sort(key=lambda x: (x[0], x[1]))
    corners = box_corners(*g.bounds)
    dedupe = float(F["dedupe_mm"])
    kept = _distinct([c[2] for c in coarse], corners, max(dedupe, float(F["coarse_dedupe_mm"])))
    # refine the distinct coarse poses whose residual is near the best one's (a wrong turn of an asymmetric part is
    # far worse; the near-equivalent poses of a symmetric or nearly symmetric part all stay)
    lim = float(coarse[kept[0]][0]) * float(F["refine_rms_factor"]) + float(F["refine_rms_margin_mm"]) if kept else 0.0
    kept = [j for j in kept if coarse[j][0] <= lim]
    n_ref = int(F["refine_symmetric"] if d["symmetry"]["kind"] != "none" else F["refine"])
    refined = []
    for j in kept[:n_ref]:
        rk, i, Tk, lab = coarse[j]
        Tf, rf = icp(fine_pts, D, N, tree, Tk, int(F["fine_iters"]), trim, bound, tol=float(F["fine_tol"]))
        snapped, snap_err = False, None
        got = _snap(s, d, Tf, float(F["exact_snap_mm"]))
        if got is not None:
            Tf, snap_err = got
            snapped = True
            rf = 0.0
        refined.append({"T": Tf, "rms": float(rf), "start": lab, "start_index": i, "snapped": snapped,
                        "snap_err": snap_err})
    refined.sort(key=lambda c: (c["rms"], c["start_index"]))
    keep = _distinct([c["T"] for c in refined], corners, dedupe)
    refined = [refined[i] for i in keep]
    for c in refined:
        c["shape"] = float(shape_score(ctx, s, key, c["T"]))
        c["detail_p99_mm"] = float(pose_measures(ctx, s, key, c["T"])[1])
    refined.sort(key=lambda c: (-round(c["shape"], 3), round(c["detail_p99_mm"], 6), c["rms"], c["start_index"]))
    best = max(c["shape"] for c in refined) if refined else 0.0
    cands = [c for c in refined if c["shape"] >= best - float(F["window"])][:int(F["max_candidates"])]
    for c in cands:
        c["feature"] = None
    tied = list(range(len(cands)))
    if feature_score is not None and cands:
        for c in cands:
            v = feature_score(ctx, s, key, c["T"])
            c["feature"] = None if v is None else float(v)
        scored = [i for i, c in enumerate(cands) if c["feature"] is not None]
        if scored:
            top = max(cands[i]["feature"] for i in scored)
            tied = [i for i in scored if cands[i]["feature"] >= top - float(F["feature_tie"])]
    return {"sid": s["sid"], "tag": s["tag"], "family": key, "candidates": cands, "tied": tied, "best": 0,
            "n_starts": len(starts), "n_refined": len(refined), "t_s": time.time() - t0}


def fit_all(ctx, shape_score=None, feature_score=None):
    """Fit every credited part (ctx.parts). A part whose fit fails is an item error: it has no fit and counts as not
    fittable (gates.fit_gate)."""
    from . import item_error
    ctx.fits = {}
    t0 = time.time()
    for tag in sorted(ctx.parts):
        s = ctx.parts[tag]
        key = ctx.ref.family_of[tag]
        try:
            ctx.fits[s["sid"]] = fit_part(ctx, s, key, shape_score, feature_score)
        except Exception as exc:  # noqa: BLE001 - this part is not fitted; it costs only its own items
            item_error(ctx, "the fit of %s" % tag, exc)
    ctx.stats["t_fit_s"] = time.time() - t0
    ctx.stats["parts_fitted"] = float(len(ctx.fits))
    ctx.stats["fit_candidates_total"] = float(sum(len(f["candidates"]) for f in ctx.fits.values()))
    ctx.stats["fit_snapped"] = float(sum(1 for f in ctx.fits.values() if f["candidates"] and f["candidates"][0]["snapped"]))
