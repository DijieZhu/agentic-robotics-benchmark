"""Matching submitted units to true units on the surface F-score.

    ts = TrueSurfaces(npz, truth_units, cfg, twin)    # the true samples (recomputed from the true twin) + KD-trees
    m = pair_measure(ts, k, solid, tree, cfg)         # {"F", "precision", "recall", "n_prec", "n_rec"} at tau_match_m
    M = match_units(truth, solids, cfg)               # Matching

Pair measure, after the alignment map (the agent's solid is in the site frame at s*), tau = matching.tau_match_m
(0.10 m: the matching says whether the unit is there; C2's shape band is the F-score at matching.tau_m = 3 cm after
the registration of b_pose, score_c2):
- precision = share of the agent unit's surface samples within tau of the nearest true sample, over the agent
  samples whose nearest true sample is "determined" (the others are left out);
- recall = share of the true unit's determined samples within tau of the nearest agent sample;
- F = 2 P R / (P + R) (0 when both are 0). An inflated solid loses precision, a shrunk or shifted one both.
Distances are to the nearest sample (scipy cKDTree, workers=1), not to the exact surface; with 1-2 cm spacing a
perfect copy has every distance far below tau.

Candidates: pairs whose bounding boxes overlap after a margin of pair_bbox_margin_m + tau; every other pair has F 0.
Matching: Hungarian (scipy linear_sum_assignment) maximising F over the entries with F >= f_min (the others set to 0
first, so a sub-threshold overlap never displaces a valid pair); labels play no part. Two passes over the same F table
(the counted units first): pass 1 over the COUNTED true units and every agent unit, pass 2
over the 'ignore' entries and the agent units that pass 1 left unpaired, so an 'ignore' entry (clutter lying on a
counted unit) cannot take the agent's model of that unit away from it. A submitted unit matched to an 'ignore' entry
is removed from all counts (neither credit nor charge). Where no agent unit has a pair (F >= f_min) with both
a counted unit and an 'ignore' entry, the table splits into the two passes and the result is the single assignment's.

The true samples: items_surface.npz stores no sample, only per unit the sample count, the
"determined" flags as packed bits and a sha256 of the sample coordinates rounded to 1e-6 m (plus per-unit moments);
TrueSurfaces recomputes the samples from the true twin with surfaces.sample_union (exactly what the oracle's solid
gives) and checks them against the stored digest: a mismatch is a TruthError (task error).
"""
import numpy as np

from .truth import TruthError


FORMAT_SURFACE = "site-items-surface/2"
DIGEST_ROUND_M = 1e-6
MOMENT_TOL_M = 1e-6


def sample_digest(P):
    """sha256 (hex) of sample coordinates rounded to DIGEST_ROUND_M."""
    import hashlib
    Q = np.round(np.asarray(P, float) / DIGEST_ROUND_M).astype(np.int64)
    return hashlib.sha256(np.ascontiguousarray(Q).tobytes()).hexdigest()


def sample_moments(P):
    """(7,) float64: the count-free fingerprint of a sample set: mean x, y, z and the root mean square distance from
    the mean along x, y, z, and the largest distance from the mean (all in m)."""
    P = np.asarray(P, float)
    if not len(P):
        return np.zeros(7)
    c = P.mean(axis=0)
    D = P - c
    return np.r_[c, np.sqrt(np.mean(D * D, axis=0)), float(np.max(np.linalg.norm(D, axis=1)))]


def true_unit_samples(twin, unit_id, cfg):
    """(P, N, spacing) of a true unit: surfaces.sample_union on its collision pieces at qpos0 (site frame), exactly
    what the grader computes for the oracle's solid."""
    from . import surfaces
    mc = cfg["matching"]
    u = twin.unit(unit_id)
    return surfaces.sample_union(twin.pieces(u["geoms"]), float(mc["spacing_m"]), int(mc["max_samples_per_unit"]),
                                 surfaces.unit_seed(unit_id, cfg))


class TrueSurfaces:
    """The true units' surface samples (recomputed from the true twin) with their stored "determined" flags."""

    def __init__(self, npz, units, cfg, twin):
        for k in ("format", "ids", "counts", "determined_bits", "digest", "moments", "spacing_m"):
            if k not in npz:
                raise TruthError("items_surface.npz lacks %r (format %s)" % (k, FORMAT_SURFACE))
        if str(npz["format"]) != FORMAT_SURFACE:
            raise TruthError("items_surface.npz has format %r, not %r" % (str(npz["format"]), FORMAT_SURFACE))
        ids = [str(v) for v in npz["ids"]]
        if ids != [u.id for u in units]:
            raise TruthError("items_surface.npz ids differ from items_truth.json (order included)")
        counts = np.asarray(npz["counts"], np.int64)
        if counts.shape != (len(ids),) or np.any(counts <= 0):
            raise TruthError("items_surface.npz counts are inconsistent")
        off = np.r_[0, np.cumsum(counts)]
        bits = np.unpackbits(np.asarray(npz["determined_bits"], np.uint8), count=int(off[-1])).astype(bool)
        if len(bits) != int(off[-1]):
            raise TruthError("items_surface.npz determined bits do not cover the samples")
        self.ids = ids
        self.offsets = off
        self.P, self.N, self.D = [], [], []
        self.spacing = np.asarray(npz["spacing_m"], float)
        self.notes = []
        digests = [str(v) for v in npz["digest"]]
        moments = np.asarray(npz["moments"], float).reshape(len(ids), 7)
        for k, uid in enumerate(ids):
            P, N, h = true_unit_samples(twin, uid, cfg)
            if len(P) != int(counts[k]):
                raise TruthError("unit %s: %d surface samples recomputed from the true twin, %d stored" % (
                    uid, len(P), int(counts[k])))
            if abs(float(h) - float(self.spacing[k])) > 1e-12:
                raise TruthError("unit %s: sample spacing %.6g recomputed, %.6g stored" % (uid, h, self.spacing[k]))
            if sample_digest(P) != digests[k]:
                dm = float(np.max(np.abs(sample_moments(P) - moments[k])))
                if dm > MOMENT_TOL_M:
                    raise TruthError("unit %s: the surface samples recomputed from the true twin are not the stored "
                                     "ones (digest differs, moments by %.3g m): rebuild items_surface.npz" % (uid, dm))
                self.notes.append("unit %s: the sample digest differs by rounding only (moments agree to %.2g m)"
                                  % (uid, dm))
            self.P.append(P)
            self.N.append(np.asarray(N, float))
            self.D.append(bits[off[k]:off[k + 1]])
        for k, i in enumerate(ids):
            if not self.D[k].any():
                raise TruthError("unit %s has no determined surface sample" % i)
        self._trees = {}
        self.boxes = np.array([np.r_[p.min(axis=0), p.max(axis=0)] for p in self.P])

    def tree(self, k):
        if k not in self._trees:
            from scipy.spatial import cKDTree
            self._trees[k] = cKDTree(self.P[k])
        return self._trees[k]

    def determined_points(self, k):
        return self.P[k][self.D[k]]


def pair_measure(ts, k, solid, tree_a, cfg, tau=None):
    """The pair measure of true unit k and an agent solid at tau (default matching.tau_match_m)."""
    tau = float(cfg["matching"]["tau_match_m"] if tau is None else tau)
    Pa, _Na = solid.samples()
    out = {"F": 0.0, "precision": 0.0, "recall": 0.0, "n_prec": 0, "n_rec": 0}
    if not len(Pa):
        return out
    tt = ts.tree(k)
    d_t, i_t = tt.query(Pa, k=1, distance_upper_bound=tau, workers=1)
    far = ~np.isfinite(d_t)
    if np.any(far):
        d2, i2 = tt.query(Pa[far], k=1, workers=1)
        d_t = d_t.copy()
        i_t = i_t.copy()
        d_t[far], i_t[far] = d2, i2
    det = ts.D[k][i_t]
    n_prec = int(det.sum())
    prec = float(np.sum(det & (d_t <= tau))) / n_prec if n_prec else 0.0
    Q = ts.determined_points(k)
    d_a, _ = tree_a.query(Q, k=1, distance_upper_bound=tau, workers=1)
    rec = float(np.mean(np.isfinite(d_a) & (d_a <= tau))) if len(Q) else 0.0
    F = 2.0 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    out.update({"F": F, "precision": prec, "recall": rec, "n_prec": n_prec, "n_rec": int(len(Q))})
    return out


class Matching:
    def __init__(self):
        self.pairs = []                 # [{"true", "agent", "F", "precision", "recall", "ignore"}], by true id order
        self.a2t, self.t2a = {}, {}     # counted pairs only (true entry not 'ignore')
        self.removed_agents = []        # matched to an 'ignore' truth entry
        self.unpaired_agents = []
        self.unpaired_true = []
        self.table = {}                 # "true|agent" -> F for every candidate with F >= table_min_f
        self.n_candidates = 0

    def summary(self):
        return {"pairs": self.pairs, "removed_agents": self.removed_agents, "unpaired_agents": self.unpaired_agents,
                "unpaired_true": self.unpaired_true, "n_candidates": self.n_candidates, "table": self.table}


def assign_counted_first(W, ignore, f_min):
    """The pairs (true index, agent index), sorted, of the counted-first matching on the F table W (true x agent): a Hungarian
    assignment maximising F over the entries with F >= f_min, first over the counted true rows (ignore[i] false) and
    every column, then over the 'ignore' rows and the columns the first pass left unpaired."""
    from scipy.optimize import linear_sum_assignment
    nt, na = W.shape
    chosen = []
    if not (nt and na):
        return chosen
    W2 = np.where(W >= f_min, W, 0.0)
    used = set()
    for rows in ([i for i in range(nt) if not ignore[i]], [i for i in range(nt) if ignore[i]]):
        cols = [j for j in range(na) if j not in used]
        if not rows or not cols:
            continue
        sub = W2[np.ix_(rows, cols)]
        ri, cj = linear_sum_assignment(-sub)
        for a, b in zip(ri, cj):
            if sub[a, b] >= f_min:
                chosen.append((int(rows[a]), int(cols[b])))
                used.add(cols[b])
    return sorted(chosen)


def match_units(truth, solids, cfg):
    """solids: {agent body: surfaces.Solid} (site frame). Returns a Matching (counted units first)."""
    from scipy.spatial import cKDTree
    mc = cfg["matching"]
    ts = truth.surfaces
    units = truth.units
    agents = sorted(solids)
    M = Matching()
    nt, na = len(units), len(agents)
    W = np.zeros((nt, na))
    detail = {}
    margin = float(mc["pair_bbox_margin_m"]) + float(mc["tau_match_m"])
    trees = {}
    for j, a in enumerate(agents):
        lo, hi = solids[a].bbox()
        for i in range(nt):
            b = ts.boxes[i]
            if np.any(np.minimum(hi, b[3:]) + margin < np.maximum(lo, b[:3])):
                continue
            if a not in trees:
                Pa = solids[a].samples()[0]
                trees[a] = cKDTree(Pa) if len(Pa) else None
            if trees[a] is None:
                continue
            M.n_candidates += 1
            r = pair_measure(ts, i, solids[a], trees[a], cfg)
            W[i, j] = r["F"]
            detail[(i, j)] = r
            if r["F"] >= float(mc["table_min_f"]):
                M.table["%s|%s" % (units[i].id, a)] = round(r["F"], 6)
    chosen = assign_counted_first(W, [u.ignore for u in units], float(mc["f_min"]))
    used_a = set()
    for i, j in chosen:
        u, a = units[i], agents[j]
        r = detail[(i, j)]
        M.pairs.append({"true": u.id, "agent": a, "F": r["F"], "precision": r["precision"], "recall": r["recall"],
                        "ignore": u.ignore})
        used_a.add(a)
        if u.ignore:
            M.removed_agents.append(a)
        else:
            M.a2t[a] = u.id
            M.t2a[u.id] = a
    M.unpaired_agents = [a for a in agents if a not in used_a]
    M.unpaired_true = [u.id for u in units if not u.ignore and u.id not in M.t2a]
    return M
