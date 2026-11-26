"""The two terms of a flight and their aggregations (pure functions of arrays; every number and every choice from the
configuration; every registered aggregation is computed, the configuration names the one in force).

Depth term D (per pixel, then per frame, then per flight):
  b(r) = max(depth.band_full_abs, depth.band_full_rel x r); Z = depth.band_zero_factor
  c = clip((Z b - |q - r|) / ((Z - 1) b), 0, 1): 1 within b, 0 at Z b and beyond; a ray of the twin that hits
      nothing (q not finite) and every pixel of a zeroed frame score 0
  Dc_k = the mean of c over frame k's scored content pixels, for the frames with at least
      depth.min_class_pixels_per_frame of them; Dc = the mean over those frames (each frame weighs the same); Ds alike
  D_weighted = depth.weight_content x Dc + depth.weight_structure_times_content x Ds x Dc (the first version)
  D_product = Dc x Ds (rule S of the second version)
  D = the one score.depth_aggregation names ('weighted' or 'product'); both are reported
Hazard term H (per hidden move, then per class, then per flight):
  clear move: 1 when the twin's frame is clear along it over its length, else 0
  blocked move: 0 when the twin's frame is clear along it over the longest length; else with n = |k_twin - k*|, the
      difference of the stop indices in path samples, 1 when n <= hazard.stop_band_full / hazard.path_sample, 0 when
      n >= hazard.stop_band_zero / hazard.path_sample, linear between (the bands are whole numbers of samples, so this
      is the band of distances evaluated exactly)
  every move of a zeroed frame scores 0
  aC, aBC, aBS = the means per class; J = max(0, aC + aBC - 1); H_moves = J x (f + (1 - f) aBS),
      f = hazard.structure_floor (the first version's H)
Key poses (rule P of the second version): a key pose is a key frame with at least one scored move; it is lost when
  one of its scored moves scores 0 (the reading in force), lost_clean when one of them scores below 1 (the registered
  alternative); every move of a zeroed frame scores 0, so a zeroed key pose is lost under both.
  H_poses = 1 - lost / key poses; H_clean_poses = 1 - lost_clean / key poses
  H = the one score.hazard_aggregation names: 'moves' = H_moves; 'min_of_moves_and_poses' = min(H_moves, H_poses);
      'min_of_moves_and_clean_poses' = min(H_moves, H_clean_poses); all three are reported
Flight: sum = score.weight_hazard x H + score.weight_depth x D; product = H x D; the flight's score is the one
score.flight_aggregation names (both are reported).
"""
import numpy as np


def band_full(r, cfg):
    return np.maximum(cfg["depth.band_full_abs"], cfg["depth.band_full_rel"] * np.asarray(r, float))


def depth_credit(q, r, cfg):
    """Credit per pixel of twin ranges q against true ranges r (arrays of one shape)."""
    q = np.asarray(q, float)
    r = np.asarray(r, float)
    b = band_full(r, cfg)
    Z = float(cfg["depth.band_zero_factor"])
    ok = np.isfinite(q)
    d = np.abs(np.where(ok, q, 0.0) - r)
    c = np.clip((Z * b - d) / ((Z - 1.0) * b), 0.0, 1.0)
    return np.where(ok, c, 0.0)


def depth_term(frames, cfg):
    """frames: list of (content credits (n_c,), structure credits (n_s,)) per key frame, in key-frame order (the
    credits of a zeroed frame are all 0). Returns a dict of the term and its parts."""
    n_min = cfg["depth.min_class_pixels_per_frame"]
    dc = [float(np.mean(c)) for c, _ in frames if len(c) >= n_min]
    ds = [float(np.mean(s)) for _, s in frames if len(s) >= n_min]
    if not dc or not ds:
        raise ValueError("no key frame counts in the content or the structure class")
    Dc = float(np.mean(dc))
    Ds = float(np.mean(ds))
    forms = depth_aggregations(Dc, Ds, cfg)
    D = forms[cfg["score.depth_aggregation"]]
    return {"D": float(D), "Dc": Dc, "Ds": Ds, "D_weighted": forms["weighted"], "D_product": forms["product"],
            "content_frames": len(dc), "structure_frames": len(ds)}


def depth_aggregations(Dc, Ds, cfg):
    """Every registered form of the depth term: {"weighted": the first version's, "product": rule S's}."""
    weighted = cfg["depth.weight_content"] * Dc + cfg["depth.weight_structure_times_content"] * Ds * Dc
    return {"weighted": float(weighted), "product": float(Dc * Ds)}


def stop_band_samples(cfg):
    """(full, zero) of the stop band in path samples; both must be whole numbers of samples."""
    step = cfg["hazard.path_sample"]
    out = []
    for key in ("hazard.stop_band_full", "hazard.stop_band_zero"):
        n = cfg[key] / step
        if abs(n - round(n)) > 1e-9:
            raise ValueError("%s is not a whole number of path samples" % key)
        out.append(int(round(n)))
    if not 0 <= out[0] < out[1]:
        raise ValueError("the stop band's full end must lie below its zero end")
    return out[0], out[1]


def stop_credit(k_twin, k_star, band):
    """Credit of a blocked move: k_twin the twin's first colliding sample (-1: clear over the longest length), k_star
    the truth's; band = stop_band_samples."""
    if k_twin < 0:
        return 0.0
    n = abs(int(k_twin) - int(k_star))
    full, zero = band
    if n <= full:
        return 1.0
    if n >= zero:
        return 0.0
    return float(zero - n) / float(zero - full)


def hazard_term(per_class, cfg):
    """per_class: {"clear": [credits], "blocked_by_content": [...], "blocked_by_structure": [...]}."""
    a = {}
    for c in ("clear", "blocked_by_content", "blocked_by_structure"):
        if not per_class[c]:
            raise ValueError("no move of class %s" % c)
        a[c] = float(np.mean(per_class[c]))
    f = cfg["hazard.structure_floor"]
    J = max(0.0, a["clear"] + a["blocked_by_content"] - 1.0)
    H = J * (f + (1.0 - f) * a["blocked_by_structure"])
    return {"H": float(H), "J": float(J), "aC": a["clear"], "aBC": a["blocked_by_content"],
            "aBS": a["blocked_by_structure"]}


def pose_term(per_pose):
    """per_pose: [(key frame, [credits of its scored moves])] in key-frame order, one row per key frame with at least
    one scored move. Returns the shares of key poses kept under both readings and the per-pose record."""
    if not per_pose:
        raise ValueError("no key frame has a scored move")
    rows = []
    for frame, cr in per_pose:
        if not cr:
            raise ValueError("key pose %s has no scored move" % frame)
        low = float(min(cr))
        rows.append({"frame": frame, "moves": len(cr), "lowest_credit": low, "lost_zero": bool(low == 0.0),
                     "lost_clean": bool(low < 1.0)})
    n = len(rows)
    lost = sum(1 for r in rows if r["lost_zero"])
    lost_clean = sum(1 for r in rows if r["lost_clean"])
    return {"H_poses": 1.0 - float(lost) / float(n), "H_clean_poses": 1.0 - float(lost_clean) / float(n),
            "key_poses": n, "key_poses_kept": n - lost, "key_poses_kept_clean": n - lost_clean, "poses": rows}


def hazard_aggregations(H_moves, H_poses, H_clean_poses):
    """Every registered form of the hazard term (score.hazard_aggregation names the one in force)."""
    return {"moves": float(H_moves), "min_of_moves_and_poses": float(min(H_moves, H_poses)),
            "min_of_moves_and_clean_poses": float(min(H_moves, H_clean_poses))}


def aggregate(H, D, cfg):
    s = cfg["score.weight_hazard"] * H + cfg["score.weight_depth"] * D
    p = H * D
    forms = {"sum": float(s), "product": float(p)}
    return {"sum": forms["sum"], "product": forms["product"], "score": forms[cfg["score.flight_aggregation"]]}
