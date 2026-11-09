"""Matching of submitted items to true items on the capped 3-D IoU.

capped IoU = IoU(T, S) x cap, both solids on the same grid of cells in the true camera frame, S the agent solid at
scale s. Grid: 3 mm cells, 1 mm cells for true items thinner than 40 mm.

The oversize cap (match.cap = "extents_after_icp", the default) charges a solid that is larger than the truth along
its own axes, and nothing else:
  - the agent solid's own frame: its item body's axes, or the principal axes of its surface samples, whichever gives
    the smaller box around it (so a geom turned inside its body is measured along its own sides);
  - the agent solid is turned by its ICP rotation onto the truth (score_c2.icp_rotation: about the true centroid,
    only about the true shape's stiff axes, a uniform scale fitted alongside as a nuisance);
  - e_agent = the agent's extents along its own axes, e_true = the true solid's extents along the same axes after
    that rotation; cap = product over the three axes of min(1, e_true / e_agent).
Uniform inflation by f costs 1 / f^3, as in a volume ratio; a shape that differs from the truth but is not larger
along any of its axes (a box around a can) costs only its IoU; a rotation error is charged once (in IoU and in C2's
rotation term), never as inflation. match.cap = "volume_ratio" gives the earlier form min(1, |T| / |S|) (diagnostics).

The assignment is an untyped Hungarian (labels play no part) on the capped IoU with every entry below iou_min = 0.25
set to 0 first, so a sub-threshold overlap can never displace a valid pair; pairs below 0.25 are then dropped. Pairs
whose volumes alone bound the capped IoU below volume_ratio_skip, or whose boxes do not overlap, are 0 without a
grid; an agent solid too large for the grid caps (solids.TooLarge) overlaps nothing. The ICP rotation of every pair
that reaches iou_min is kept in the details: C2 uses the same rotation for b_rot.
"""
import numpy as np

from . import score_c2
from . import solids as SO


class TrueSet:
    """The true solids with their grids, ICP surfaces and stiff rotation axes (computed once per verify)."""

    def __init__(self, true_solids, cfg):
        mc = cfg["match"]
        self.T = list(true_solids)
        self.h, self.keys, self.vol, self.box, self.min_width = [], [], [], [], []
        for T in self.T:
            w = T.min_width()
            h = (float(mc["fine_grid_mm"]) if w < float(mc["thin_mm"]) / 1000.0 else float(mc["grid_mm"])) / 1000.0
            self.min_width.append(w)
            self.h.append(h)
            self.keys.append(T.keys(h, max_cells=int(mc["max_cells"])))
            self.vol.append(T.volume_centroid()[0])
            self.box.append(T.bbox())
        self.surfaces = None
        self.stiff = None
        self.symmetry = None

    def ensure_icp(self, cfg):
        """The true surfaces and stiff axes for the ICP (Truth fills them in; computed here for a bare TrueSet)."""
        if self.surfaces is None:
            self.surfaces = score_c2.true_surfaces(self.T, cfg)
        if self.stiff is None:
            self.stiff, self.symmetry = [], []
            for i, T in enumerate(self.T):
                B, info = score_c2.symmetry_axes(T, self.surfaces[i], self.keys[i], self.h[i], cfg)
                self.stiff.append(B)
                self.symmetry.append(dict(info, body=T.name, label=T.label))


def own_axes(S, P):
    """(axes (3, 3), which): the agent solid's own frame, the body axes or the principal axes of its surface samples P,
    whichever gives the smaller box (ties: the body axes)."""
    cands = [("body", np.asarray(S.frame, float))]
    if len(P) >= 3:
        X = P - P.mean(axis=0)
        _w, U = np.linalg.eigh(X.T @ X)
        cands.append(("principal", U))
    best = None
    for which, A in cands:
        vol = float(np.prod(S.extents(A)))
        if best is None or vol < best[0]:
            best = (vol, A, which)
    return best[1], best[2]


class Matcher:
    def __init__(self, true_set, agent_solids, cfg):
        """true_set: TrueSet (or a list of true Solids); agent_solids: list of Solid at scale 1 in the aligned agent
        camera frame."""
        ts = true_set if isinstance(true_set, TrueSet) else TrueSet(true_set, cfg)
        ts.ensure_icp(cfg)
        self.ts = ts
        self.T = ts.T
        self.S = list(agent_solids)
        self.cfg = cfg
        mc = cfg["match"]
        self.iou_min = float(mc["iou_min"])
        self.max_cells = int(mc["max_cells"])
        self.vr_skip = float(mc["volume_ratio_skip"])
        self.cap_mode = str(mc.get("cap", "extents_after_icp"))
        if self.cap_mode not in ("extents_after_icp", "volume_ratio"):
            raise ValueError("unknown match.cap %r" % self.cap_mode)
        self.h, self.tkeys, self.tvol, self.tbox = ts.h, ts.keys, ts.vol, ts.box
        self.svol = [S.volume_centroid()[0] for S in self.S]
        self._skeys = {}
        self._surf = {}
        self._icp = {}
        self.evaluations = 0
        self.icp_runs = 0

    def _agent_keys(self, j, s, h):
        k = (j, s, h)
        if self._skeys and next(iter(self._skeys))[1] != s:
            self._skeys = {}                      # keep the keys of one scale only (memory)
        if k not in self._skeys:
            try:
                self._skeys[k] = self.S[j].scaled(s).keys(h, max_cells=self.max_cells)
            except SO.TooLarge:
                self._skeys[k] = None
        return self._skeys[k]

    def _agent_surface(self, j, s):
        k = (j, s)
        if self._surf and next(iter(self._surf))[1] != s:
            self._surf = {}
        if k not in self._surf:
            pc = self.cfg["pose"]
            self._surf[k] = self.S[j].scaled(s).surface(int(pc["icp_samples"]), int(pc["seed"]))
        return self._surf[k]

    def icp(self, i, j, s):
        """ICP of agent j at scale s onto true i: dict(R, rot_deg, iterations, scale, cS, cT)."""
        k = (i, j, s)
        if k not in self._icp:
            T, S = self.T[i], self.S[j]
            cT = T.volume_centroid()[1]
            cS = s * S.volume_centroid()[1]
            P, _N = self._agent_surface(j, s)
            P = P + (cT - cS)
            dst, dst_n = self.ts.surfaces[i]
            R, ang, iters, sig = score_c2.icp_rotation(P, dst, dst_n, cT, self.cfg, self.ts.stiff[i])
            self._icp[k] = {"R": R, "rot_deg": float(ang), "iterations": int(iters), "scale": float(sig)}
            self.icp_runs += 1
        return self._icp[k]

    def oversize_cap(self, i, j, s, R):
        """(cap, info) of the extents-after-ICP form."""
        Ss = self.S[j].scaled(s)
        A, which = own_axes(Ss, self._agent_surface(j, s)[0])
        e_a = Ss.extents(A)
        e_t = self.T[i].extents(R @ A)
        ratio = np.where(e_a > 0, e_t / np.where(e_a > 0, e_a, 1.0), 1.0)
        cap = float(np.prod(np.minimum(1.0, ratio)))
        return cap, {"cap_axes": which, "extent_agent_mm": [1000.0 * float(v) for v in e_a],
                     "extent_true_mm": [1000.0 * float(v) for v in e_t]}

    def pair(self, i, j, s):
        """dict(iou, cap, capped, n_true, n_agent, n_inter, grid_mm, ICP fields) of true i and agent j at scale s, or
        None when the pair is ruled out without a grid."""
        vt, vs = self.tvol[i], self.svol[j] * s ** 3
        if vt <= 0 or vs <= 0:
            return None
        r = vs / vt
        if r < 1.0:
            bound = r
        else:
            bound = 1.0 / (r * r) if self.cap_mode == "volume_ratio" else 1.0 / r
        if bound < self.vr_skip:
            return None
        lo_t, hi_t = self.tbox[i]
        lo_s, hi_s = self.S[j].bbox()
        lo_s, hi_s = lo_s * s, hi_s * s
        if np.any(np.minimum(hi_t, hi_s) < np.maximum(lo_t, lo_s)):
            return None
        h = self.h[i]
        ks = self._agent_keys(j, s, h)
        if ks is None:
            return None
        kt = self.tkeys[i]
        self.evaluations += 1
        ni = int(np.intersect1d(kt, ks, assume_unique=True).size)
        nt, ns = int(kt.size), int(ks.size)
        un = nt + ns - ni
        iou = ni / un if un > 0 else 0.0
        out = {"iou": float(iou), "cap": None, "capped": 0.0, "n_true": nt, "n_agent": ns, "n_inter": ni,
               "grid_mm": 1000.0 * h}
        if iou < self.iou_min:
            return out                             # capped <= iou < iou_min: never a pair, no ICP needed
        icp = self.icp(i, j, s)
        out.update({"rot_deg": icp["rot_deg"], "icp_iterations": icp["iterations"], "icp_scale": icp["scale"]})
        if self.cap_mode == "volume_ratio":
            cap = min(1.0, nt / ns) if ns > 0 else 0.0
        else:
            cap, info = self.oversize_cap(i, j, s, icp["R"])
            out.update(info)
        out["cap"] = float(cap)
        out["capped"] = float(iou * cap)
        return out

    def matrix(self, s):
        W = np.zeros((len(self.T), len(self.S)))
        D = {}
        for i in range(len(self.T)):
            for j in range(len(self.S)):
                d = self.pair(i, j, s)
                if d is not None:
                    W[i, j] = d["capped"]
                    D[(i, j)] = d
        return W, D

    def assign(self, s):
        """[(i, j, capped)] of the Hungarian on the capped IoU at scale s (entries below iou_min set to 0 first),
        pairs >= iou_min, and the details."""
        from scipy.optimize import linear_sum_assignment
        if not self.T or not self.S:
            return [], {}
        W, D = self.matrix(s)
        W = np.where(W >= self.iou_min, W, 0.0)
        ri, cj = linear_sum_assignment(-W)
        pairs = [(int(i), int(j), float(W[i, j])) for i, j in zip(ri, cj) if W[i, j] >= self.iou_min]
        pairs.sort()
        return pairs, D
