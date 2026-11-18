"""C2 pose and shape: per matched pair q = b_pose x b_shape x b_vert; C2_zone = sum q / max(N_true,
n_submitted); C2 = the mean over the zones that have a true or a submitted unit.

    q, detail = pair_q(ctx)        # run by the core before any criterion (C4 and C5 weight by q)
    score(ctx)                     # the criterion

b_pose = band_down(e_pose, C2.pose_band_m). A restricted registration of the true unit's determined surface samples
  onto the agent unit's surface (site frame, at s*): a yaw psi about the vertical through the centre c of the true
  footprint box (the minimum-area rectangle of the true samples' xy) plus a 3-D translation t, point-to-plane on the
  agent samples' normals with nearest neighbours taken in [position, normal_weight_m x normal] space (orientation-
  aware: measured on a test site, plain nearest neighbours recovered only 0.8 of 2 deg of a trailer's yaw), trimmed
  (registration.trim), damped (max_step_deg / max_step_m per iteration, max_total_deg / max_total_m in all), from yaw
  0 and, for c2 / c4 units, also from the symmetric yaws; the start with the smallest trimmed mean 6-D distance wins
  (ties: the first). `free` units fit the translation only.
  G(x) = Rz(psi) (x - c) + c + t; e_pose = min over the symmetry group S of max over the four corners k of the true
  footprint box of |G(S(corner_k)) - corner_k| in xy (S = rotations about c by 180 deg for c2, 90 deg for c4; none:
  the identity only); free: |t_xy|. A corner further than C2.pose_corner_clip_m from c is taken on its ray from c at
  that distance (a 1.3 deg turn of a 6.7 m truck moves its far corners 8 cm, but the robot
  meets the unit where it passes it; the clip charges the yaw of every unit like that of a unit 2 x clip across).
b_shape = band_up(F_reg, C2.shape_band) (C2.shape_on "F_registered"): F_reg is the surface
  F-score at matching.tau_m (3 cm) after applying the registration of b_pose to the true samples, so it measures the
  shape alone (measured on a test site: the matching's F at 3 cm took a pallet box's q to 0 at a 6 cm shift
  while b_pose was 0.82). shape_on "F" (calibration only) uses the matching's F instead, which is taken at
  matching.tau_match_m (0.10 m).
b_vert = band_down(e_vert, C2.vert_band_m): support.SupportField.vertical_gap in the agent's own twin, times s* (a
  shape whose top is more than C2.vert.support_above_m above the unit's lowest point is no support), then along the
  support chain: a unit's b_vert is at most the chained b_vert of what it rests on (its
  supporters: support.CHAIN_TOL_M), and the chain must end in the environment or a gate, so a unit on a floating
  unit floats too (support.chain_b_vert; the supporters may be units that are not paired, which get their own
  vertical gap). detail: b_vert (chained), b_vert_own, support.supporters.
detail also carries F_registered when C2.report_registered_shape is set; C1 counts a pair as a true positive only
when its shape measure reaches C2.shape_band[0] (score_c1.shape_measure).
"""
import math

import numpy as np

from .config import band_down, band_up
from .zones import mean_over_zones

SYM_STEPS = {"none": [0.0], "c2": [0.0, math.pi], "c4": [0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi], "free": [0.0]}


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def true_box(truth, k):
    """The true unit's footprint box (geom2d.min_area_rect of all its stored samples' xy), cached on the truth."""
    from . import geom2d
    cache = truth.__dict__.setdefault("_c2_boxes", {})
    if k not in cache:
        cache[k] = geom2d.min_area_rect(truth.surfaces.P[k][:, :2])
    return cache[k]


def _subsample(P, cap):
    if len(P) <= cap:
        return P
    step = int(math.ceil(len(P) / float(cap)))
    return P[::step]


def register(Pt, Nt, Pa, Na, tree, c, sym, rc):
    """Restricted registration of the true samples (Pt, normals Nt) onto the agent surface (Pa, Na; `tree` a cKDTree
    of [Pa, w Na], w = registration.normal_weight_m): (psi, t, info). Correspondences are nearest neighbours in that
    6-D space, so a sample only pairs with agent surface of a similar orientation (a wheel top never pairs with the
    deck bottom 2 cm above it); the residual is point-to-plane on the agent normal."""
    trim = float(rc["trim"])
    w = float(rc["normal_weight_m"])
    step_r = math.radians(float(rc["max_step_deg"]))
    step_t = float(rc["max_step_m"])
    tot_r = math.radians(float(rc["max_total_deg"]))
    tot_t = float(rc["max_total_m"])
    c3 = np.array([c[0], c[1], 0.0])
    best = None
    rows = []
    for psi0 in SYM_STEPS[sym]:
        psi, t = psi0, np.zeros(3)
        it = 0
        for it in range(1, int(rc["iterations"]) + 1):
            Rz = _Rz(psi)
            X = (Pt - c3) @ Rz.T + c3 + t
            d, idx = tree.query(np.c_[X, w * (Nt @ Rz.T)], k=1, workers=1)
            keep = np.argsort(d, kind="stable")[:max(6, int(len(d) * trim))]
            n = Na[idx[keep]]
            r = np.einsum("ij,ij->i", n, X[keep] - Pa[idx[keep]])
            rel = X[keep] - c3 - t
            cols = [n[:, 0], n[:, 1], n[:, 2]]
            if sym != "free":
                cols = [n[:, 1] * rel[:, 0] - n[:, 0] * rel[:, 1]] + cols      # n . (e_z x rel)
            J = np.stack(cols, axis=1)
            x = np.linalg.lstsq(J, -r, rcond=1e-10)[0]
            if not np.all(np.isfinite(x)):
                break
            dpsi = float(x[0]) if sym != "free" else 0.0
            dt = x[1:4] if sym != "free" else x[0:3]
            dpsi = max(-step_r, min(step_r, dpsi))
            nt = float(np.linalg.norm(dt))
            if nt > step_t:
                dt = dt * (step_t / nt)
            if abs(psi + dpsi - psi0) > tot_r or float(np.linalg.norm(t + dt)) > tot_t:
                break
            psi += dpsi
            t = t + dt
            if abs(dpsi) < 1e-9 and float(np.linalg.norm(dt)) < 1e-8:
                break
        Rz = _Rz(psi)
        X = (Pt - c3) @ Rz.T + c3 + t
        d, _idx = tree.query(np.c_[X, w * (Nt @ Rz.T)], k=1, workers=1)
        dk = np.sort(d, kind="stable")[:max(6, int(len(d) * trim))]
        cost = float(np.mean(dk))
        rows.append({"start_deg": math.degrees(psi0), "psi_deg": math.degrees(psi), "t": [float(v) for v in t],
                     "cost_m": cost, "iterations": it})
        if best is None or cost < best[0]:
            best = (cost, psi, t)
    return best[1], best[2], {"starts": rows}


def e_pose(psi, t, box, sym, clip=None):
    """The largest horizontal displacement of the true footprint box's four corners under the registration (psi about
    the box centre, then t), minimised over the symmetry group; `clip` (C2.pose_corner_clip_m):
    a corner further than clip from the box's centre is taken on the same ray at the clip distance, so the yaw
    of a long unit is charged like that of a unit 2 x clip across. free: |t_xy| (no corner, no clip)."""
    if sym == "free":
        return float(np.linalg.norm(t[:2]))
    c = np.asarray(box["centre"], float)
    Q = np.asarray(box["corners"], float) - c
    if clip is not None:
        n = np.linalg.norm(Q, axis=1)
        far = n > float(clip)
        if np.any(far):
            Q = Q.copy()
            Q[far] = Q[far] * (float(clip) / n[far])[:, None]
    best = math.inf
    for k in SYM_STEPS[sym]:
        a = psi + k
        R2 = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        D = Q @ R2.T + t[:2] - Q
        best = min(best, float(np.max(np.linalg.norm(D, axis=1))))
    return best


def _registered_F(D_mask, tree_t_points, Pa, psi, t, c, tau):
    """F after applying the registration G to the true samples: the shape alone, the pose removed."""
    from scipy.spatial import cKDTree
    c3 = np.array([c[0], c[1], 0.0])
    X = (tree_t_points - c3) @ _Rz(psi).T + c3 + t
    tt = cKDTree(X)
    d_t, i_t = tt.query(Pa, k=1, workers=1)
    det = D_mask[i_t]
    prec = float(np.sum(det & (d_t <= tau))) / max(1, int(det.sum()))
    ta = cKDTree(Pa)
    d_a, _ = ta.query(X[D_mask], k=1, distance_upper_bound=tau, workers=1)
    rec = float(np.mean(np.isfinite(d_a)))
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _vert_own(vg, s_star, sec):
    """A unit's own b_vert (before the chain) from its vertical gap: band_down(e_vert x s*, vert_band_m)."""
    ev = float(vg["e_vert"]) * float(s_star) if math.isfinite(vg["e_vert"]) else math.inf
    return ev, band_down(ev, *[float(v) for v in sec["vert_band_m"]])


def chained_vert(ctx, gaps, sf):
    """b_vert along the support chain. gaps {agent body: vertical_gap result} of the paired
    units; the supporters they name (and theirs, recursively) get their own vertical gap too. Returns {body: (own
    b_vert, chained b_vert)} for every body visited."""
    from .support import chain_b_vert, is_chain_end
    sec = ctx.cfg["C2"]
    s = float(ctx.align.s)
    gaps = dict(gaps)
    todo = sorted({o for vg in gaps.values() for o in vg.get("supporters") or () if not is_chain_end(o)})
    while todo:
        b = todo.pop(0)
        if b in gaps:
            continue
        gaps[b] = sf.vertical_gap(b, ctx.cfg, s)
        for o in gaps[b].get("supporters") or ():
            if not is_chain_end(o) and o not in gaps and o not in todo:
                todo.append(o)
        todo.sort()
    own = {b: _vert_own(vg, s, sec)[1] for b, vg in gaps.items()}
    sups = {b: list(vg.get("supporters") or ()) for b, vg in gaps.items()}
    ch = chain_b_vert(own, sups)
    return {b: (own[b], ch[b]) for b in sorted(gaps)}


def pair_q(ctx):
    from scipy.spatial import cKDTree
    sec = ctx.cfg["C2"]
    rc = sec["registration"]
    truth, M = ctx.truth, ctx.matching
    tau = float(ctx.cfg["matching"]["tau_m"])
    sf = ctx.support()
    q, detail = {}, {}
    gaps = {p["agent"]: sf.vertical_gap(p["agent"], ctx.cfg, ctx.align.s) for p in M.pairs if not p["ignore"]}
    vert = chained_vert(ctx, gaps, sf)
    for p in M.pairs:
        if p["ignore"]:
            continue
        tu = truth.unit[p["true"]]
        k = tu.k
        S = ctx.solids[p["agent"]]
        Pa, Na = S.samples()
        tree = cKDTree(np.c_[Pa, float(rc["normal_weight_m"]) * Na])
        box = true_box(truth, k)
        D = truth.surfaces.D[k]
        Pt = _subsample(truth.surfaces.P[k][D], int(rc["max_samples"]))
        Nt = _subsample(truth.surfaces.N[k][D], int(rc["max_samples"]))
        psi, t, rinfo = register(Pt, Nt, Pa, Na, tree, box["centre"], tu.symmetry, rc)
        ep = e_pose(psi, t, box, tu.symmetry, float(sec["pose_corner_clip_m"]))
        b_pose = band_down(ep, *[float(v) for v in sec["pose_band_m"]])
        f_reg = None
        if sec.get("report_registered_shape") or sec["shape_on"] == "F_registered":
            f_reg = _registered_F(truth.surfaces.D[k], truth.surfaces.P[k], Pa, psi, t, box["centre"], tau)
        f_shape = f_reg if sec["shape_on"] == "F_registered" else p["F"]
        b_shape = band_up(f_shape, *[float(v) for v in sec["shape_band"]])
        vg = gaps[p["agent"]]
        ev, _own = _vert_own(vg, ctx.align.s, sec)
        b_vert_own, b_vert = vert[p["agent"]]
        qq = b_pose * b_shape * b_vert
        q[tu.id] = qq
        row = {"agent": p["agent"], "q": qq, "b_pose": b_pose, "b_shape": b_shape, "b_vert": b_vert,
               "b_vert_own": b_vert_own, "e_pose_m": ep, "e_vert_m": ev, "F": p["F"], "precision": p["precision"],
               "recall": p["recall"], "psi_deg": math.degrees(psi), "t_m": [float(v) for v in t],
               "symmetry": tu.symmetry, "registration": rinfo,
               "support": {kk: vg.get(kk) for kk in ("min_gap", "supporter", "supporters", "n_columns",
                                                     "n_supported", "column_m")}}
        if f_reg is not None:
            row["F_registered"] = f_reg
        row["shape_on"] = sec["shape_on"]
        detail[tu.id] = row
    return q, detail


def score(ctx):
    Z = ctx.zones
    sums = {z: 0.0 for z in Z.ids}
    for tid, qq in ctx.q.items():
        sums[Z.of_true[tid]] += float(qq)
    per = {}
    for z in Z.ids:
        nt, ns = Z.n_true[z], Z.n_sub[z]
        den = max(nt, ns)
        per[z] = {"sum_q": sums[z], "n_true": nt, "n_submitted": ns, "denominator": den,
                  "score": (sums[z] / den) if den > 0 else None}
    return {"value": mean_over_zones({z: v["score"] for z, v in per.items()}),
            "detail": {"per_zone": per, "per_true": ctx.q_detail}}
