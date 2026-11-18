"""Grader part 1: the geometric core of one twin, and the truth's reproduction checks.

    core(ctx)            # after the alignment: unit solids (site frame, at s*), the matching, the zones
    q_stage(ctx)         # q per counted pair (score_c2.pair_q), needed by C2, C4 and C5
    check_truth(truth, ctx, cfg)   # the self-test's reproduction checks on the true twin's own run (TruthError)

The alignment itself is part2's "align" stage (it decides G0). Everything here works in the site frame on the agent's
geometry mapped by ctx.align.map, except the support columns of b_vert, which stay in the agent's own twin.
"""
import math

import numpy as np

from . import match, raycast, surfaces, zones
from .truth import TruthError


def core(ctx):
    ctx.solids = surfaces.unit_solids(ctx.twin, ctx.align.map, ctx.cfg)
    ctx.matching = match.match_units(ctx.truth, ctx.solids, ctx.cfg)
    ctx.zones = zones.allocate(ctx.truth, ctx.twin, ctx.solids, ctx.matching, ctx.align.map, ctx.cfg)
    return {"n_solids": len(ctx.solids), "n_pairs": len([p for p in ctx.matching.pairs if not p["ignore"]]),
            "n_removed": len(ctx.matching.removed_agents), "n_candidates": ctx.matching.n_candidates}


def q_stage(ctx):
    from . import score_c2
    ctx.q, ctx.q_detail = score_c2.pair_q(ctx)
    return {"n_q": len(ctx.q), "mean_q": float(np.mean(list(ctx.q.values()))) if ctx.q else 0.0}


# ---------------------------------------------------------------------------------------------------------------
# the truth's reproduction checks (part of the self-test)

def check_cameras(truth, cfg):
    st = cfg["selftest"]
    tw = truth.twin
    if set(tw.cameras) != set(truth.cameras):
        raise TruthError("the true twin's valid photo cameras %s differ from cameras_true.json %s" % (
            sorted(tw.cameras)[:8], sorted(truth.cameras)[:8]))
    worst = {"centre_m": 0.0, "axes_deg": 0.0, "intr_px": 0.0}
    for p, cam in sorted(tw.cameras.items()):
        tc = truth.cameras[p]
        dc = float(np.linalg.norm(cam["p"] - tc.C))
        da = math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(cam["R_cv"].T @ tc.T)) - 1.0) / 2.0))))
        di = float(np.max(np.abs(cam["K"] - tc.K)))
        if (cam["W"], cam["H"]) != (tc.W, tc.H):
            raise TruthError("photo %d: the true twin's camera is %d x %d, cameras_true says %d x %d" % (
                p, cam["W"], cam["H"], tc.W, tc.H))
        worst = {"centre_m": max(worst["centre_m"], dc), "axes_deg": max(worst["axes_deg"], da),
                 "intr_px": max(worst["intr_px"], di)}
    if (worst["centre_m"] > float(st["camera_tol_m"]) or worst["axes_deg"] > float(st["camera_tol_deg"])
            or worst["intr_px"] > float(st["intr_tol_px"])):
        raise TruthError("the true twin's cameras differ from cameras_true.json: %s" % worst)
    return worst


def check_zones(truth, ctx):
    bad = []
    for u in truth.counted_units():
        S = ctx.solids.get(u.id)
        if S is None:
            bad.append((u.id, "no solid"))
            continue
        z = truth.region.zone_of(S.footprint_centroid()[None])[0]
        if z != u.zone:
            bad.append((u.id, z, u.zone))
    if bad:
        raise TruthError("true units whose footprint centroid is not in their zone: %s" % bad[:5])
    return len(truth.counted_units())


def check_surfaces(truth, ctx, cfg):
    """The stored samples are the very samples surfaces.sample_union makes on the true twin."""
    tol = float(cfg["selftest"]["surface_tol_m"])
    worst = 0.0
    for u in truth.units:
        S = ctx.solids.get(u.id)
        if S is None:
            raise TruthError("true unit %s has no solid" % u.id)
        P, _N = S.samples()
        Q = truth.surfaces.P[u.k]
        if P.shape != Q.shape:
            raise TruthError("unit %s: %d stored samples, %d recomputed" % (u.id, len(Q), len(P)))
        worst = max(worst, float(np.max(np.abs(P - Q))) if len(P) else 0.0)
    if worst > tol:
        raise TruthError("the stored surface samples differ from the true twin's by %.2e m" % worst)
    return worst


def check_c3(truth, cfg):
    """The first hit of every stored c3 ray on the true twin's full collision scene is the stored point."""
    st = cfg["selftest"]
    c3 = truth.c3
    if not len(c3["origin"]):
        return {"n": 0, "agree": 1.0}
    dist, _own, _src, _n = raycast.cast_grouped(truth.full_scene(), c3["origin"], c3["direction"])
    P = np.asarray(c3["origin"], float) + np.where(dist >= 0, dist, np.nan)[:, None] * np.asarray(c3["direction"])
    err = np.linalg.norm(P - np.asarray(c3["point"], float), axis=1)
    agree = float(np.mean(np.isfinite(err) & (err <= float(st["c3_agree_m"]))))
    if agree < float(st["c3_agree_frac"]):
        raise TruthError("the c3 samples agree with the true twin's ray cast on only %.4f of the rays" % agree)
    return {"n": int(len(err)), "agree": agree, "max_err_m": float(np.nanmax(err)) if np.any(np.isfinite(err)) else None}
