"""Grader part 1: G0 gate, alignment, C1 inventory, C2 pose and shape, C3 environment, C6 scale.

    truth = Truth(gt_twin_dir, depth_ref_png, depth_class_png, vocabulary)   # once per verify; raises TruthError
    res = score_part1(submission_dir, truth)                                   # JSON-ready dict

res["pairs"] lists the matched (true body, agent body, q) for the physics criteria; res["alignment"] holds R_pp
and s* for mapping the agent's world into the true camera frame. A submission that fails G0 scores 0 in every
criterion here (the settle part of G0 is load.g0_settle, run by the caller on load.Twin.physics_model(), the model
without the excluded items).
"""
import time

import numpy as np

from . import align, load, match, regions, score_c1, score_c2, score_c3, score_c6, solids
from .config import merged

FORMAT = "twin-score-part1/1"


class TruthError(RuntimeError):
    """The ground truth fails its own checks: a task error, never the agent's fault."""


class Truth:
    def __init__(self, gt_dir, depth_png, class_png, vocabulary=None, config=None):
        self.cfg = merged(config)
        t0 = time.time()
        self.vocabulary = load.vocabulary(vocabulary)   # read and validated once; TaskError if broken
        self.twin = load.load_twin(gt_dir, self.vocabulary)
        tw = self.twin
        if not tw.ok:
            raise TruthError("the ground-truth twin fails the gate: %s" % "; ".join(tw.reasons[:5]))
        bad = [it["body"] for it in tw.items if it["status"] != "ok"]
        if bad:
            why = {it["body"]: it for it in tw.report["items"] if it["body"] in bad}
            raise TruthError("ground-truth items excluded by the gate: %s" % "; ".join(
                "%s (%s)" % (b, "; ".join(why[b].get("reasons", [])[:2])) for b in bad[:5]))
        badlab = [it["body"] for it in tw.items if not it["label_ok"]]
        if badlab:
            raise TruthError("ground-truth labels outside the product vocabulary: %s" % ", ".join(badlab[:5]))
        self.M, self.t = align.frame_map(tw)
        sol = solids.item_solids(tw, self.M, self.t)
        self.solids = [sol[it["body"]] for it in tw.items]
        why = regions.check_truth(self.cfg, [x.name for x in self.solids])
        if why:
            raise TruthError(why)
        self.n_true = len(self.solids)
        self.true_set = match.TrueSet(self.solids, self.cfg)
        self.centroids = np.array([S.volume_centroid()[1] for S in self.solids])
        self.true_set.ensure_icp(self.cfg)
        self.surfaces = self.true_set.surfaces
        self.stiff = self.true_set.stiff
        self.symmetry = self.true_set.symmetry
        self.ref = score_c3.load_reference(depth_png, class_png)
        self.gt_cast = None
        self.seconds = time.time() - t0


def _zero(res, why):
    res["g0"]["pass"] = False
    res["g0"]["reasons"] = why
    for k in ("C1", "C2", "C3", "C6"):
        res[k] = {"score": 0.0}
    return res


def score_part1(sub_dir, truth, vocabulary=None, config=None):
    cfg = truth.cfg if config is None else merged(config)
    t0 = time.time()
    res = {"format": FORMAT, "g0": {}, "alignment": {}, "pairs": [], "timing_s": {}}
    tw = load.load_twin(sub_dir, truth.vocabulary if vocabulary is None else vocabulary)
    res["g0"] = {"pass": tw.ok, "reasons": tw.reasons, "warnings": tw.report.get("warnings", []),
                 "excluded": tw.report.get("excluded", []), "n_items": tw.n_items,
                 "items": [{"body": it["body"], "label": it.get("label"), "status": it["status"],
                            "reasons": it.get("reasons", [])} for it in tw.report.get("items", [])]}
    res["timing_s"]["gate"] = time.time() - t0
    if not tw.ok:
        return _zero(res, tw.reasons)
    # alignment into the true camera frame
    T = truth.twin
    if cfg["frame_alignment"] == "principal_ray_then_scale":
        R_pp, rinfo = align.principal_ray_rotation(T.K, tw.K, cfg["principal_ray_clamp_deg"])
    elif cfg["frame_alignment"] == "scale_only":
        R_pp, rinfo = np.eye(3), {"angle_deg": 0.0, "clamped": False, "mode": "scale_only"}
    else:
        raise ValueError("unknown frame_alignment %r" % cfg["frame_alignment"])
    R_arg = None if rinfo["angle_deg"] == 0.0 else R_pp
    Ma, ta = align.frame_map(tw, R_arg)
    sol = solids.item_solids(tw, Ma, ta)
    S = [sol[it["body"]] for it in tw.items if it["body"] in sol]
    t1 = time.time()
    m = match.Matcher(truth.true_set, S, cfg)
    ca = np.array([x.volume_centroid()[1] for x in S]) if S else np.zeros((0, 3))
    s_star, pairs, sinfo = align.fit_scale(truth.centroids, ca, lambda s: m.assign(s)[0], cfg)
    pairs, details = m.assign(s_star)
    res["timing_s"]["match"] = time.time() - t1
    res["alignment"] = {"mode": cfg["frame_alignment"], "R_pp": R_pp.tolist(), "R_pp_info": rinfo,
                        "s_star": float(s_star), "n_pairs": len(pairs), "fit": sinfo,
                        "grid_evaluations": m.evaluations, "icp_runs": m.icp_runs, "cap": m.cap_mode,
                        "agent_K": {"fx": float(tw.K[0, 0]), "fy": float(tw.K[1, 1]), "cx": float(tw.K[0, 2]),
                                    "cy": float(tw.K[1, 2])}}
    n_true, n_sub = truth.n_true, tw.n_items
    areas = regions.allocate(truth, tw, S, pairs, s_star, Ma, ta, cfg)
    res["areas"] = areas
    c1, i1 = score_c1.score(pairs, truth.solids, S, n_true, n_sub, areas)
    t2 = time.time()
    c2, i2 = score_c2.score(pairs, truth.solids, S, s_star, details, n_true, n_sub, cfg, truth.surfaces, truth.stiff,
                            areas)
    res["timing_s"]["c2"] = time.time() - t2
    t3 = time.time()
    try:
        c3, i3, gt_cast = score_c3.score(T, tw, R_arg, s_star, truth.ref, cfg, truth.gt_cast)
        truth.gt_cast = gt_cast
    except Exception as exc:  # noqa: BLE001 - a twin whose furniture cannot be ray-cast scores 0 here
        c3, i3 = 0.0, {"error": str(exc)[:300]}
    res["timing_s"]["c3"] = time.time() - t3
    c6, i6 = score_c6.score(s_star, len(pairs), cfg)
    if cfg.get("gating", {}).get("scene_by_inventory", False):
        # the stage counts as far as the inventory on it: furniture and scale without the items earn nothing
        i3 = dict(i3, ungated=c3, gate_inventory=c1)
        i6 = dict(i6, ungated=c6, gate_inventory=c1)
        c3, c6 = c3 * c1, c6 * c1
    res["C1"] = dict(score=c1, **i1)
    res["C2"] = dict(score=c2, **i2)
    res["C3"] = dict(score=c3, **i3)
    res["C6"] = dict(score=c6, **i6)
    per = i2["per_true"]
    res["pairs"] = [{"true": truth.solids[i].name, "agent": S[j].name, "q": per[truth.solids[i].name]["q"],
                     "capped_iou": c} for i, j, c in pairs]
    res["timing_s"]["total"] = time.time() - t0
    return res


def selftest(truth, oracle_result=None):
    """The truth's own checks (raise TruthError on failure): the gate passes with no exclusions (checked in Truth);
    the GT furniture ray cast reproduces the C3 reference; the oracle, if given, scores 1 exactly."""
    out = {"truth_items": truth.n_true}
    if oracle_result is not None:
        c3 = oracle_result["C3"]
        out["gt_agree_frac"] = c3.get("gt_agree_frac")
        if c3.get("gt_agree_frac", 0.0) < truth.cfg["environment"]["selftest_frac"]:
            raise TruthError("the GT furniture ray cast agrees with the C3 reference on only %.4f of the pixels" %
                             c3.get("gt_agree_frac", 0.0))
        for k in ("C1", "C2", "C3", "C6"):
            out[k] = oracle_result[k]["score"]
        out["s_star"] = oracle_result["alignment"]["s_star"]
    return out
