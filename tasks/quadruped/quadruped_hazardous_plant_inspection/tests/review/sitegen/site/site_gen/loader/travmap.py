"""The run-time half of the traversability map: the 0.05 m grid, a stored map's arrays, the (cell, mode) graph and
the searches on it. The generator's site_gen.trav.TravMap adds the rasteriser on top of this class.

A cell is enterable in a mode when the robot's footprint can be centred there; costs are times (cell distance
over the cell's speed in the mode, plus a mode-switch time). Arrays are indexed [iy, ix].
"""

from __future__ import annotations

import math

import numpy as np

MODES = ("walk", "high_step", "crouch_walk")
NB8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
       (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def _signed_area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _min_width(P):
    """Minimum width of a convex polygon (rotating calipers over its edges)."""
    if len(P) < 3:
        return 0.0
    best = float("inf")
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        e = b - a
        L = float(np.hypot(e[0], e[1]))
        if L < 1e-12:
            continue
        n = np.array([-e[1], e[0]]) / L
        h = np.abs((P - a) @ n).max()
        best = min(best, float(h))
    return best


class Grid:
    def __init__(self, bbox, res):
        self.res = float(res)
        self.x0 = math.floor(bbox[0] / res) * res
        self.y0 = math.floor(bbox[1] / res) * res
        self.nx = int(math.ceil((bbox[2] - self.x0) / res))
        self.ny = int(math.ceil((bbox[3] - self.y0) / res))
        self.shape = (self.ny, self.nx)

    @classmethod
    def exact(cls, x0, y0, res, shape):
        g = cls.__new__(cls)
        g.res = float(res)
        g.x0, g.y0 = float(x0), float(y0)
        g.ny, g.nx = int(shape[0]), int(shape[1])
        g.shape = (g.ny, g.nx)
        return g

    def cell(self, x, y):
        return int(math.floor((y - self.y0) / self.res)), int(math.floor((x - self.x0) / self.res))

    def xy(self, iy, ix):
        return self.x0 + (ix + 0.5) * self.res, self.y0 + (iy + 0.5) * self.res

    def centers(self):
        xs = self.x0 + (np.arange(self.nx) + 0.5) * self.res
        ys = self.y0 + (np.arange(self.ny) + 0.5) * self.res
        return np.meshgrid(xs, ys)

    def fill_poly(self, mask, poly, value=1):
        """Rasterise a convex polygon by cell-centre inclusion: a cell is set when its centre lies inside (or on)
        the polygon. A polygon thinner than a cell (a 45 mm door leaf, a thin plate) is drawn as the cells under its
        centre line instead, so it stays a connected one-cell line."""
        P = np.asarray(poly, np.float64)
        if len(P) < 2:
            return
        res = self.res
        x0, y0 = P.min(0)
        x1, y1 = P.max(0)
        ix0 = max(0, int(math.ceil((x0 - self.x0) / res - 0.5 - 1e-9)))
        ix1 = min(self.nx - 1, int(math.floor((x1 - self.x0) / res - 0.5 + 1e-9)))
        iy0 = max(0, int(math.ceil((y0 - self.y0) / res - 0.5 - 1e-9)))
        iy1 = min(self.ny - 1, int(math.floor((y1 - self.y0) / res - 0.5 + 1e-9)))
        thin = len(P) < 3 or _min_width(P) < res
        if ix1 >= ix0 and iy1 >= iy0 and len(P) >= 3:
            xs = self.x0 + (np.arange(ix0, ix1 + 1) + 0.5) * res
            ys = self.y0 + (np.arange(iy0, iy1 + 1) + 0.5) * res
            X, Y = np.meshgrid(xs, ys)
            Q = P if _signed_area(P) >= 0 else P[::-1]
            ins = np.ones(X.shape, bool)
            for k in range(len(Q)):
                ax, ay = Q[k]
                bx, by = Q[(k + 1) % len(Q)]
                ins &= (bx - ax) * (Y - ay) - (by - ay) * (X - ax) >= -1e-9
            if ins.any():
                sub = mask[iy0:iy1 + 1, ix0:ix1 + 1]
                sub[ins] = value
        if thin:
            c = P.mean(0)
            Q0 = P - c
            w_, v_ = np.linalg.eigh(Q0.T @ Q0)
            u = v_[:, int(np.argmax(w_))]
            t = Q0 @ u
            n = max(2, int(math.ceil((t.max() - t.min()) / (res / 3.0))) + 1)
            S = c[None, :] + np.outer(np.linspace(t.min(), t.max(), n), u)
            jx = np.floor((S[:, 0] - self.x0) / res).astype(int)
            jy = np.floor((S[:, 1] - self.y0) / res).astype(int)
            ok = (jx >= 0) & (jx < self.nx) & (jy >= 0) & (jy < self.ny)
            mask[jy[ok], jx[ok]] = value

    def info(self):
        return {"origin": [self.x0, self.y0], "res": self.res, "shape": [self.ny, self.nx]}


class TravMap:
    """A stored traversability map (arrays from trav.npz) and the searches on it.

    params: r_in, r_circ (footprint thresholds, m), switch_time_s (mode change)."""

    def __init__(self, grid: Grid, params: dict):
        self.g = grid
        self.params = dict(params)
        self.th = {"r_in": float(params.get("r_in") or 0.1875), "r_circ": float(params.get("r_circ") or 0.325)}
        self.switch_time = float(params.get("switch_time_s", 3.0))

    # ------------------------------------------------------------------------------------------------ load / save
    @classmethod
    def from_arrays(cls, z: dict, params: dict):
        res = float(z["res"])
        x0, y0 = [float(v) for v in z["origin"]]
        g = Grid.exact(x0, y0, res, z["enter_walk"].shape)
        t = cls.__new__(cls)
        TravMap.__init__(t, g, params)
        t._set_arrays(z)
        return t

    def _set_arrays(self, z):
        self.enter = {m: np.asarray(z["enter_%s" % m]) for m in MODES}
        self.tight = {m: np.asarray(z["tight_%s" % m]) for m in MODES}
        self.speed = {m: np.asarray(z["speed_%s" % m]) for m in MODES}
        self.slow = np.asarray(z["slow"])
        self.hsreq = np.asarray(z["hsreq"])
        self.crreq = np.asarray(z["crreq"])
        self.inside = np.asarray(z["inside"])
        self.elem = np.asarray(z["elem"]).astype(np.int32)
        self.elem_ids = [str(s) for s in z["elem_ids"]]

    def arrays(self) -> dict:
        return {"origin": np.array([self.g.x0, self.g.y0]), "res": np.array(self.g.res),
                **{"enter_%s" % m: np.asarray(self.enter[m], bool) for m in MODES},
                **{"tight_%s" % m: np.asarray(self.tight[m], bool) for m in MODES},
                **{"speed_%s" % m: np.asarray(self.speed[m], np.float32) for m in MODES},
                "slow": np.asarray(self.slow, bool), "hsreq": np.asarray(self.hsreq, bool),
                "crreq": np.asarray(self.crreq, bool), "inside": np.asarray(self.inside, bool),
                "elem": np.asarray(self.elem).astype(np.int16), "elem_ids": np.array(self.elem_ids)}

    # ------------------------------------------------------------------------------------------------ masks
    def enter_blocked(self, extra_solid):
        """Enterable masks per mode when the cells of `extra_solid` (bool grid) are solid for every mode."""
        from scipy import ndimage
        res = self.g.res
        out = {}
        for m in MODES:
            s = self.solid[m] | extra_solid if hasattr(self, "solid") else None
            if s is None:
                d = ndimage.distance_transform_edt(~extra_solid)
                out[m] = self.enter[m] & ((d - 0.5) * res >= self.th["r_in"])
                continue
            d = ndimage.distance_transform_edt(~s)
            out[m] = (~s) & ((d - 0.5) * res >= self.th["r_in"]) & self.inside
        return out

    def reach_from(self, pad_node, modes=MODES, enter=None):
        G = self.build_graph(modes, enter=enter)
        return self.reach(G, pad_node)

    def cells(self, nodes_mask):
        N = self.g.nx * self.g.ny
        r = nodes_mask[:N] | nodes_mask[N:2 * N] | nodes_mask[2 * N:3 * N]
        return r.reshape(self.g.shape)

    def cells_mode(self, nodes_mask, mode):
        N = self.g.nx * self.g.ny
        k = MODES.index(mode)
        return nodes_mask[k * N:(k + 1) * N].reshape(self.g.shape)

    # ------------------------------------------------------------------------------------------------ graph
    def build_graph(self, modes=MODES, enter=None):
        """Sparse directed graph over (mode, cell) states; node = mode_index * N + iy * nx + ix."""
        from scipy.sparse import csr_matrix
        ny, nx = self.g.shape
        N = ny * nx
        rows, cols, vals = [], [], []
        res = self.g.res
        idx = np.arange(N).reshape(ny, nx)
        enter = enter if enter is not None else self.enter
        for mi, m in enumerate(MODES):
            if m not in modes:
                continue
            en = enter[m]
            inv = np.where(en, 1.0 / np.maximum(self.speed[m], 1e-6), 0.0)
            for dy, dx, dist in NB8:
                ys0, ys1 = max(0, -dy), ny - max(0, dy)
                xs0, xs1 = max(0, -dx), nx - max(0, dx)
                a = en[ys0:ys1, xs0:xs1]
                b = en[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
                ok = a & b
                if dx != 0 and dy != 0:
                    ok &= en[ys0 + dy:ys1 + dy, xs0:xs1] & en[ys0:ys1, xs0 + dx:xs1 + dx]
                ia = idx[ys0:ys1, xs0:xs1][ok]
                ib = idx[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx][ok]
                cost = 0.5 * dist * res * (inv[ys0:ys1, xs0:xs1][ok] + inv[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx][ok])
                rows.append(mi * N + ia)
                cols.append(mi * N + ib)
                vals.append(cost)
        ms = [i for i, m in enumerate(MODES) if m in modes]
        for i in ms:
            for j in ms:
                if i >= j:
                    continue
                both = (enter[MODES[i]] & enter[MODES[j]]).ravel()
                c = np.nonzero(both)[0]
                rows += [i * N + c, j * N + c]
                cols += [j * N + c, i * N + c]
                vals += [np.full(len(c), self.switch_time), np.full(len(c), self.switch_time)]
        r = np.concatenate(rows)
        c = np.concatenate(cols)
        v = np.concatenate(vals).astype(np.float64)
        return csr_matrix((v, (r, c)), shape=(len(MODES) * N, len(MODES) * N))

    def node(self, x, y, mode="walk"):
        iy, ix = self.g.cell(x, y)
        return MODES.index(mode) * self.g.nx * self.g.ny + iy * self.g.nx + ix

    def unnode(self, n):
        N = self.g.nx * self.g.ny
        mi, c = divmod(int(n), N)
        iy, ix = divmod(c, self.g.nx)
        return iy, ix, MODES[mi]

    def nearest_enterable(self, x, y, mode="walk", max_r=0.6):
        iy, ix = self.g.cell(x, y)
        en = self.enter[mode]
        r = int(max_r / self.g.res)
        best = None
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                yy, xx = iy + dy, ix + dx
                if 0 <= yy < self.g.ny and 0 <= xx < self.g.nx and en[yy, xx]:
                    d = dx * dx + dy * dy
                    if best is None or d < best[0]:
                        best = (d, yy, xx)
        if best is None:
            return None
        return best[1], best[2]

    def reach(self, G, src_node):
        from scipy.sparse.csgraph import breadth_first_order
        order = breadth_first_order(G, src_node, directed=True, return_predecessors=False)
        m = np.zeros(G.shape[0], bool)
        m[order] = True
        return m

    def dijkstra(self, G, sources):
        from scipy.sparse.csgraph import dijkstra
        return dijkstra(G, directed=True, indices=list(sources), return_predecessors=True)

    def path(self, pred_row, src, dst):
        out = []
        n = int(dst)
        guard = 0
        while n != src and n >= 0 and guard < 10_000_000:
            out.append(n)
            n = int(pred_row[n])
            guard += 1
        if n != src:
            return []
        out.append(src)
        return out[::-1]

    # ------------------------------------------------------------------------------------------------ queries
    def modes_at(self, x, y):
        iy, ix = self.g.cell(x, y)
        if not (0 <= iy < self.g.ny and 0 <= ix < self.g.nx):
            return []
        return [m for m in MODES if self.enter[m][iy, ix]]

    def required_mode_map(self):
        """Per cell the cheapest-capability mode that can enter it: 0 none, 1 walk, 2 walk (slow), 3 walk (tight),
        4 high_step only, 5 crouch_walk only."""
        out = np.zeros(self.g.shape, np.uint8)
        w = self.enter["walk"]
        out[self.enter["crouch_walk"] & ~w & ~self.enter["high_step"]] = 5
        out[self.enter["high_step"] & ~w] = 4
        out[w] = 1
        out[w & self.slow] = 2
        out[w & self.tight["walk"]] = 3
        return out
