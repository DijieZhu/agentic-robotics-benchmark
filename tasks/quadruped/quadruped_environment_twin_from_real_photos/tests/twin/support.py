"""What a unit rests on, in the twin's own frame and units (C2's b_vert).

    sf = SupportField(twin)                  # every collision shape of the twin at qpos0, by owner
    r = sf.vertical_gap("pallet_box_1", cfg) # {"e_vert": m (agent units), "min_gap": signed, "supporter", ...}

Columns. Over the unit's footprint box, a grid of vertical columns (C2.vert.column_m, coarsened to at most
C2.vert.max_columns columns). In each column the unit's own solid gives its lowest point z_low and the top of its
lowest segment; the support is the highest top of any OTHER shape whose interval in that column starts below the
middle of the unit's lowest segment AND whose top is at most C2.vert.support_above_m above the unit's lowest point
(a shape reaching higher shares the column from the side or swallows the unit: it is no support, so a neighbour that
overlaps a unit's edge by a centimetre does not count as a deep penetration): convex pieces (exact column intervals,
twin_gate.Piece.z_interval) of the environment, of the other units the gate kept and of the gates, and every plane
facing up (its half-space). gap = z_low - support: positive is a gap,
negative a penetration (the unit is sunk into its support). The unit's min_gap is the smallest gap over the columns
that have a support; e_vert = |min_gap| (a resting unit touches: 0). A unit with no support under any column floats
over the void or is sunk deeper than support_above_m: e_vert = inf. All of it in the twin's own frame (its z is its
up); the caller multiplies by s*, and support_above_m is divided by s* to keep it in site metres.

Supporters (the support chain of b_vert): the owners ("env", "gate:<body>" or a unit's body)
of the columns whose gap lies within CHAIN_TOL_M (site metres, / s* in the twin's frame) of the unit's min_gap, i.e.
what the unit rests on at its lowest contact (a unit standing both on the floor and on a neighbour at the same level
has both). chain_b_vert() turns the per-unit band values into the chained ones: a unit's b_vert is at most the best
chained b_vert of its supporters, the environment and the gates (bodies hanging from the static world) end a chain
with 1, and a chain that never reaches them (a stack on a unit that floats, a loop of units resting on each other)
gives 0 [the least fixed point of b(u) = min(own(u), max over supporters s of b(s)), i.e. the widest
path to the environment].
"""
import math

import numpy as np

from . import load

CHAIN_TOL_M = 0.005          # site metres: columns this close to the lowest contact name the unit's supporters


def is_chain_end(owner):
    """The environment and the gates end a support chain."""
    return owner == "env" or str(owner).startswith("gate:")


def chain_b_vert(own, supporters):
    """own {body: b_vert of the unit by itself}, supporters {body: [owner, ...]} -> {body: chained b_vert}.
    b(u) = min(own(u), max over the supporters s of b(s)), b(env) = b(gate) = 1, a supporter that is no listed unit
    (or that has no value) counts 0; computed as the least fixed point from b = 0 (a monotone iteration over the
    sorted bodies, at most len(own) + 1 sweeps), so a loop of units that never reaches the environment stays 0."""
    names = sorted(own)
    b = {u: 0.0 for u in names}
    for _sweep in range(len(names) + 1):
        changed = False
        for u in names:
            best = 0.0
            for s in supporters.get(u, ()):
                v = 1.0 if is_chain_end(s) else b.get(s, 0.0)
                if v > best:
                    best = v
            nv = min(float(own[u]), best)
            if nv > b[u]:
                b[u] = nv
                changed = True
        if not changed:
            break
    return b


class SupportField:
    def __init__(self, twin):
        import mujoco
        self.twin = twin
        m, d = twin.model, twin.data
        TG = load.payload()["twin_gate"]
        self.TG = TG
        self.shapes = []                              # (owner, piece)
        self.planes = []                              # (point, normal) of static planes facing up
        units_ok = {u["body"] for u in twin.units if u["status"] == "ok"}
        owner_of = {g: "env" for g in twin.static_geoms()}
        for u in twin.units:
            if u["body"] in units_ok:
                for g in u["geoms"]:
                    owner_of[g] = u["body"]
        for gt in twin.gates:
            for g in gt["geoms"]:
                owner_of[g] = "gate:" + gt["body"]
        for g in sorted(owner_of):
            t = int(m.geom_type[g])
            if t == int(mujoco.mjtGeom.mjGEOM_PLANE):
                n = np.asarray(d.geom_xmat[g], float).reshape(3, 3)[:, 2]
                if n[2] > 0.1:
                    self.planes.append((np.asarray(d.geom_xpos[g], float).copy(), n.copy()))
                continue
            pc = twin.pieces([g])
            if pc:
                self.shapes.append((owner_of[g], pc[0]))
        self.boxes = np.array([np.r_[p.bbox()[0], p.bbox()[1]] for _o, p in self.shapes]).reshape(-1, 6)

    def vertical_gap(self, body, cfg, s_star=1.0):
        vc = cfg["C2"]["vert"]
        above = float(vc["support_above_m"]) / float(s_star)
        mine = [p for o, p in self.shapes if o == body]
        if not mine:
            return {"e_vert": math.inf, "min_gap": None, "n_columns": 0, "n_supported": 0, "supporter": None,
                    "why": "the unit has no collision pieces"}
        lo = np.min([p.bbox()[0] for p in mine], axis=0)
        hi = np.max([p.bbox()[1] for p in mine], axis=0)
        h = float(vc["column_m"])
        nmax = int(vc["max_columns"])
        while math.ceil((hi[0] - lo[0]) / h) * math.ceil((hi[1] - lo[1]) / h) > nmax:
            h *= 1.25
        nx = max(1, int(math.ceil((hi[0] - lo[0]) / h)))
        ny = max(1, int(math.ceil((hi[1] - lo[1]) / h)))
        xs = lo[0] + (np.arange(nx) + 0.5) * (hi[0] - lo[0]) / nx
        ys = lo[1] + (np.arange(ny) + 0.5) * (hi[1] - lo[1]) / ny
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        X, Y = X.ravel(), Y.ravel()
        Z0, Z1 = self.TG.column_union(mine, X, Y)
        z_low, z_seg_top = Z0[:, 0], Z1[:, 0]
        under = np.isfinite(z_low)
        X, Y, z_low, z_seg_top = X[under], Y[under], z_low[under], z_seg_top[under]
        if not len(X):
            return {"e_vert": math.inf, "min_gap": None, "n_columns": 0, "n_supported": 0, "supporter": None,
                    "why": "the unit covers no column"}
        z_mid = 0.5 * (z_low + z_seg_top)
        sup = np.full(len(X), -np.inf)
        who = np.full(len(X), -1, int)
        owners = []
        sel = ((self.boxes[:, 3] >= X.min()) & (self.boxes[:, 0] <= X.max()) & (self.boxes[:, 4] >= Y.min()) &
               (self.boxes[:, 1] <= Y.max()) & (self.boxes[:, 2] <= float(z_mid.max())))
        for k in np.nonzero(sel)[0]:
            o, p = self.shapes[k]
            if o == body:
                continue
            a0, a1 = p.z_interval(X, Y)
            ok = np.isfinite(a0) & (a0 <= z_mid) & (a1 <= z_low + above) & (a1 > sup)
            if np.any(ok):
                owners.append(o)
                sup = np.where(ok, a1, sup)
                who = np.where(ok, len(owners) - 1, who)
        for p0, n in self.planes:
            zp = p0[2] - (n[0] * (X - p0[0]) + n[1] * (Y - p0[1])) / n[2]
            ok = (zp > sup) & (zp <= z_low + above)
            if np.any(ok):
                owners.append("env")
                sup = np.where(ok, zp, sup)
                who = np.where(ok, len(owners) - 1, who)
        has = np.isfinite(sup)
        if not np.any(has):
            return {"e_vert": math.inf, "min_gap": None, "n_columns": int(len(X)), "n_supported": 0,
                    "supporter": None, "column_m": h,
                    "why": "nothing under the unit (or it is sunk deeper than %.3f m)" % above}
        gap = np.where(has, z_low - sup, np.inf)
        k = int(np.argmin(gap))
        g = float(gap[k])
        near = has & (gap <= g + CHAIN_TOL_M / float(s_star))
        sups = sorted({owners[int(w)] for w in who[near] if w >= 0})
        return {"e_vert": abs(g), "min_gap": g, "n_columns": int(len(X)), "n_supported": int(np.sum(has)),
                "supporter": owners[who[k]] if who[k] >= 0 else None, "supporters": sups, "column_m": h,
                "at": [float(X[k]), float(Y[k]), float(z_low[k])]}
