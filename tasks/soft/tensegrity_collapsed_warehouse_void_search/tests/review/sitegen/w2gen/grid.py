"""Occupancy grid and shortest paths for the W2 predicates (no robot dynamics).

The robot is a disc of radius r on a 0.02 m grid: a cell is free when the exact Euclidean distance from its
centre to every solid footprint and to the site rectangle's edges is >= r. Paths are 8-connected shortest paths
(scipy.sparse.csgraph.dijkstra) between the cells nearest to the given points.
"""
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

SOLID_KINDS = ("rubble", "support", "crate", "plug", "wall")


def solid_footprints(site, extra=()):
    """(x0, x1, y0, y1) of every box that reaches the floor, plus covered openings too low for the robot."""
    out = []
    for b in site["boxes"]:
        if b["kind"] in SOLID_KINDS:
            a = b["aabb"]
            out.append((a[0], a[1], a[2], a[3]))
    need = site["envelope"]["values"]["H_ROLL"] + 0.02
    for row in ("r1", "r2"):
        for g in site[row]["openings"]:
            if g["covered"] and g["clear_h_min"] < need:
                out.append((g["x0"], g["x1"], g["y0"], g["y1"]))
    out += list(extra)
    return out


class Grid:
    def __init__(self, site, step=0.02):
        self.step = step
        self.lx, self.ly = site["size_m"]
        self.xs = np.arange(step / 2, self.lx, step)
        self.ys = np.arange(step / 2, self.ly, step)
        self.X, self.Y = np.meshgrid(self.xs, self.ys)

    def clearance(self, boxes):
        """Exact distance from each cell centre to the nearest footprint or site edge (0 inside a footprint)."""
        X, Y = self.X, self.Y
        d = np.minimum.reduce([X, self.lx - X, Y, self.ly - Y])
        for x0, x1, y0, y1 in boxes:
            dx = np.maximum(np.maximum(x0 - X, 0.0), X - x1)
            dy = np.maximum(np.maximum(y0 - Y, 0.0), Y - y1)
            d = np.minimum(d, np.hypot(dx, dy))
        return d

    def inside_any(self, boxes):
        m = np.zeros(self.X.shape, bool)
        for x0, x1, y0, y1 in boxes:
            m |= (self.X >= x0) & (self.X <= x1) & (self.Y >= y0) & (self.Y <= y1)
        return m

    def cell(self, p):
        i = int(np.clip(round((p[1] - self.ys[0]) / self.step), 0, len(self.ys) - 1))
        j = int(np.clip(round((p[0] - self.xs[0]) / self.step), 0, len(self.xs) - 1))
        return i, j

    def graph(self, ok):
        H, W = ok.shape
        idx = np.arange(H * W).reshape(H, W)
        rows, cols, wts = [], [], []
        s, s2 = self.step, self.step * math.sqrt(2.0)
        for di, dj, w in ((0, 1, s), (1, 0, s), (1, 1, s2), (1, -1, s2)):
            i0, i1 = 0, H - di
            j0, j1 = max(0, -dj), W - max(0, dj)
            a, b = ok[i0:i1, j0:j1], ok[i0 + di:i1 + di, j0 + dj:j1 + dj]
            ia, ib = idx[i0:i1, j0:j1], idx[i0 + di:i1 + di, j0 + dj:j1 + dj]
            both = a & b
            rows.append(ia[both])
            cols.append(ib[both])
            wts.append(np.full(int(both.sum()), w))
        r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        return csr_matrix((w, (r, c)), shape=(H * W, H * W))

    def legs(self, ok, points, order):
        """Shortest path for each consecutive pair in `order` (keys of `points`). Returns a list of dicts with
        a, b, length_m (None if no path) and path (list of (x, y))."""
        H, W = ok.shape
        G = self.graph(ok)
        cells = {k: self.cell(points[k]) for k in set(order)}
        srcs = list(dict.fromkeys(order[:-1]))
        usable = {k: bool(ok[cells[k]]) for k in cells}
        src_idx = [cells[k][0] * W + cells[k][1] for k in srcs]
        dist, pred = dijkstra(G, directed=False, indices=src_idx, return_predecessors=True)
        out = []
        for a, b in zip(order[:-1], order[1:]):
            row = srcs.index(a)
            t = cells[b][0] * W + cells[b][1]
            if not (usable[a] and usable[b]) or not np.isfinite(dist[row, t]):
                out.append(dict(a=str(a), b=str(b), length_m=None, path=None))
                continue
            path = [t]
            s_ = src_idx[row]
            while path[-1] != s_:
                path.append(int(pred[row, path[-1]]))
            path.reverse()
            pts = [(float(self.xs[q % W]), float(self.ys[q // W])) for q in path]
            out.append(dict(a=str(a), b=str(b), length_m=round(float(dist[row, t]), 3), path=pts))
        return out


def path_length(path):
    if path is None or len(path) < 2:
        return 0.0
    pa = np.asarray(path)
    return float(np.linalg.norm(np.diff(pa, axis=0), axis=1).sum())
