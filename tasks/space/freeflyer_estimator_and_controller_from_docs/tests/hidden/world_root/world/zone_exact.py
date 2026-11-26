"""The exact zone test of the printed words (safety.keep_in, safety.body_radius_m): the body's bounding sphere, the
closed ball of radius r about the body origin, lies inside the union of the closed keep-in boxes of the zone file in
force and meets no keep-out box.






















"""
import numpy as np

FAR_M = 10.0          # m: the far plane beyond each end of the keep-in boxes' extent on each axis (the probe's)
THIN_M = 0.004        # m: gaps between keep-in boxes thinner than this (outside cells with inside cells on both sides along
                      #    their thin axis) are counted for the record
NEAR_M = 1.0          # m: the single-point path (point) is exact below this distance and reports NEAR_M above it
SLACK_M = 0.25        # m: the single-point path's local list of cells serves every point within this distance (max norm)


def box_dist(P, lo, hi):
    """Euclidean distances (N, M) from points P (N, 3) to the closed boxes [lo, hi] (M, 3)."""
    d = np.maximum(np.maximum(lo[None, :, :] - P[:, None, :], P[:, None, :] - hi[None, :, :]), 0.0)
    return np.sqrt((d * d).sum(-1))


class ExactZones:
    def __init__(self, zones):
        kin = [z for z in zones if z["type"] == "keepin"]
        kout = [z for z in zones if z["type"] == "keepout"]
        if not kin:
            raise ValueError("zero keep-in zones")
        self.other_types = sorted({str(z["type"]) for z in zones if z["type"] not in ("keepin", "keepout")})
        lo = np.array([np.minimum(z["min"], z["max"]) for z in kin], float)
        hi = np.array([np.maximum(z["min"], z["max"]) for z in kin], float)
        self.kin_lo, self.kin_hi = lo, hi
        br = []
        for a in range(3):
            v = np.unique(np.concatenate([lo[:, a], hi[:, a]]))
            br.append(np.concatenate([[v[0] - FAR_M], v, [v[-1] + FAR_M]]))
        self.br = br
        cx = [0.5 * (b[:-1] + b[1:]) for b in br]
        G = np.stack(np.meshgrid(*cx, indexing="ij"), -1)
        inside = np.zeros(G.shape[:3], bool)
        for i in range(len(lo)):
            inside |= np.all((G > lo[i]) & (G < hi[i]), axis=-1)
        self.inside = inside
        nb = np.zeros_like(inside)
        nb[1:, :, :] |= inside[:-1, :, :]
        nb[:-1, :, :] |= inside[1:, :, :]
        nb[:, 1:, :] |= inside[:, :-1, :]
        nb[:, :-1, :] |= inside[:, 1:, :]
        nb[:, :, 1:] |= inside[:, :, :-1]
        nb[:, :, :-1] |= inside[:, :, 1:]
        idx = np.argwhere(nb & ~inside)
        self.out_lo = np.stack([br[a][idx[:, a]] for a in range(3)], 1)
        self.out_hi = np.stack([br[a][idx[:, a] + 1] for a in range(3)], 1)
        # gaps between keep-in boxes: outside cells thinner than THIN_M along an axis with inside cells on both sides along it
        gaps = 0
        for a in range(3):
            w = np.diff(br[a])
            for k in range(1, len(w) - 1):
                if w[k] < THIN_M:
                    s0 = [slice(None)] * 3; s1 = list(s0); s2 = list(s0)
                    s0[a], s1[a], s2[a] = k - 1, k, k + 1
                    gaps += int(np.sum(inside[tuple(s0)] & ~inside[tuple(s1)] & inside[tuple(s2)]))
        self.thin_gap_cells = gaps
        self._near = None             # the single-point path's local list: (centre, out_lo, out_hi)
        if kout:
            self.ko_lo = np.array([np.minimum(z["min"], z["max"]) for z in kout], float)
            self.ko_hi = np.array([np.maximum(z["min"], z["max"]) for z in kout], float)
        else:
            self.ko_lo = np.zeros((0, 3))
            self.ko_hi = np.zeros((0, 3))

    def describe(self):
        return {"keep_in_boxes": int(len(self.kin_lo)), "keep_out_boxes": int(len(self.ko_lo)),
                "boundary_outside_cells": int(len(self.out_lo)), "thin_gap_cells_below_m": [self.thin_gap_cells, THIN_M],
                "other_zone_types_ignored": self.other_types}

    # ------------------------------------------------------------------ point tests (closed boxes)
    def in_union(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        r = np.zeros(len(P), bool)
        for i in range(len(self.kin_lo)):
            r |= np.all((P >= self.kin_lo[i]) & (P <= self.kin_hi[i]), axis=1)
        return r

    def in_keep_out(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        r = np.zeros(len(P), bool)
        for i in range(len(self.ko_lo)):
            r |= np.all((P >= self.ko_lo[i]) & (P <= self.ko_hi[i]), axis=1)
        return r

    # ------------------------------------------------------------------ distances
    def d_in(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        out = np.zeros(len(P))
        ins = self.in_union(P)
        if ins.any():
            Q = P[ins]
            res = np.empty(len(Q))
            for s in range(0, len(Q), 256):
                res[s:s + 256] = box_dist(Q[s:s + 256], self.out_lo, self.out_hi).min(1)
            out[ins] = res
        return out

    def d_out(self, P):
        P = np.atleast_2d(np.asarray(P, float))
        if not len(self.ko_lo):
            return np.full(len(P), np.inf)
        return box_dist(P, self.ko_lo, self.ko_hi).min(1)

    def clearance(self, P):
        """d(p) = min(d_in, d_out) for points P (N, 3)."""
        return np.minimum(self.d_in(P), self.d_out(P))

    def point(self, p):
        """(d_in, d_out) of one point, each exact below NEAR_M and NEAR_M above it (the rule's decisions need r < NEAR_M).
        The cells of O that can lie within NEAR_M of a point within SLACK_M (max norm) of a centre are those meeting the cube
        of half side NEAR_M + SLACK_M about the centre (a Euclidean distance is at least the max-norm one); the list is made
        again when the point leaves the centre's slack."""
        p = np.asarray(p, float).reshape(3)
        if not np.any(np.all((p >= self.kin_lo) & (p <= self.kin_hi), axis=1)):
            di = 0.0
        else:
            nr = self._near
            if nr is None or np.max(np.abs(p - nr[0])) > SLACK_M:
                h = NEAR_M + SLACK_M
                m = np.all(self.out_hi >= p - h, axis=1) & np.all(self.out_lo <= p + h, axis=1)
                nr = self._near = (p.copy(), self.out_lo[m], self.out_hi[m])
            if len(nr[1]):
                d = np.maximum(np.maximum(nr[1] - p, p - nr[2]), 0.0)
                di = min(float(np.sqrt((d * d).sum(1)).min()), NEAR_M)
            else:
                di = NEAR_M
        if len(self.ko_lo):
            d = np.maximum(np.maximum(self.ko_lo - p, p - self.ko_hi), 0.0)
            do = min(float(np.sqrt((d * d).sum(1)).min()), NEAR_M)
        else:
            do = NEAR_M
        return di, do

    # ------------------------------------------------------------------ the rule's answers
    def breach(self, p, r):
        """None when the closed ball B(p, r) lies in the keep-in union and meets no keep-out box except in a touching point;
        else 'keep_in' (the ball leaves the union) or 'keep_out' (it enters a keep-out box), the nearer of the two."""
        di, do = self.point(p)
        assert r < NEAR_M
        if di >= r and do >= r:
            return None
        return "keep_in" if di <= do else "keep_out"

    def contact(self, p):
        """None when the body origin lies in the closed keep-in union and outside every closed keep-out box; else 'outside'
        (beyond the keep-in volume, where the walls are) or 'structure' (in a keep-out box)."""
        p = np.asarray(p, float).reshape(3)
        if not np.any(np.all((p >= self.kin_lo) & (p <= self.kin_hi), axis=1)):
            return "outside"
        if len(self.ko_lo) and np.any(np.all((p >= self.ko_lo) & (p <= self.ko_hi), axis=1)):
            return "structure"
        return None
