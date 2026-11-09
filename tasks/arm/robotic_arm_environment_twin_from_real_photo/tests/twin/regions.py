"""Work areas (regions) of the scene: the console top, the open drawer and the right shelf.

The site brief's job covers all three places, so with `regions.balanced` each of them counts the same in the item
criteria (C1, C2 here; C4 and C5 in part 2): a criterion is computed per area, over that area's true items, and the
areas' values are averaged.

- A true item belongs to the area its support chain ends in (`regions.of_item`, from the truth build; checked
  against the support truth in part 2).
- A submitted item belongs to the area of the true item it is paired with; an unpaired one (excluded items
  included) to the area of the nearest true item, by the distance between its centre (the solid's volume centroid,
  or the item body's centre of mass when it has no solid) and the true items' centroids in the true camera frame at
  s*. So the areas' submitted counts always add up to n_sub.
"""
import numpy as np


def config_regions(cfg):
    rc = cfg.get("regions") or {}
    return bool(rc.get("balanced", False)), dict(rc.get("of_item") or {}), dict(rc.get("of_supporter") or {})


def check_truth(cfg, true_names):
    """TruthError text (None when fine): every true item has an area when balancing is on."""
    on, of_item, _ = config_regions(cfg)
    if not on:
        return None
    missing = [n for n in true_names if n not in of_item]
    extra = [n for n in of_item if n not in set(true_names)]
    if missing or extra:
        return "the work-area map does not cover the true items (missing %s, unknown %s)" % (missing[:5], extra[:5])
    return None


def allocate(truth, tw, S, pairs, s_star, Ma, ta, cfg):
    """The areas of the submitted items. S: the agent solids (in tw.items order, ok items only), pairs: (i, j, c)."""
    on, of_item, _ = config_regions(cfg)
    true_names = [x.name for x in truth.solids]
    areas = sorted(set(of_item.values())) if on else []
    out = {"balanced": on, "areas": areas, "of_true": {n: of_item.get(n) for n in true_names} if on else {},
           "of_agent": {}, "how": {}, "n_true": {}, "n_sub": {}}
    if not on:
        return out
    paired = {S[j].name: true_names[i] for i, j, _c in pairs}
    sol_c = {x.name: float(s_star) * np.asarray(x.volume_centroid()[1], float) for x in S}
    C = np.asarray(truth.centroids, float)
    for it in tw.items:
        b = it["body"]
        if b in paired:
            out["of_agent"][b] = of_item[paired[b]]
            out["how"][b] = "pair"
            continue
        if b in sol_c:
            x = sol_c[b]
        else:
            x = float(s_star) * (np.asarray(Ma, float) @ np.asarray(tw.data.xipos[it["body_id"]], float) +
                                 np.asarray(ta, float))
        k = int(np.argmin(np.linalg.norm(C - x[None, :], axis=1))) if len(C) else 0
        out["of_agent"][b] = of_item[true_names[k]]
        out["how"][b] = "nearest"
    for r in areas:
        out["n_true"][r] = sum(1 for n in true_names if of_item[n] == r)
        out["n_sub"][r] = sum(1 for v in out["of_agent"].values() if v == r)
    return out


def mean_over_areas(values):
    """The mean of {area: value} (areas with a None value left out); 0 when none is left."""
    v = [float(x) for x in values.values() if x is not None]
    return sum(v) / len(v) if v else 0.0
