"""Hybrid A* of the privileged reference over the TRUE occupancy of a site.

State: planar centroid (x, y), heading th (the authors' SE(2): direction of mean(nodes 0,2,4) - mean(nodes 1,3,5)),
and the bottom face. Actions: the 11 published primitives; the SE(2) transform of each is the mean of the robot's
own primitive table for the start face (PRIM_TABLE: dx, dy in the start frame x = heading, y = heading + 90 deg,
dtheta, time) and the face after it (deterministic on flat ground). Cost: simulated time (s) plus a clearance
penalty; heuristic: geodesic grid distance to the goal / the fastest primitive speed, weighted (eps).

Occupancy: the exact clearance of every 0.02 m cell to the footprints of the solids (truth.SiteTruth). A move is
checked along the straight centroid segment at N_SAMPLES points:
  * hard: clearance >= r_hard everywhere (default max(W_SWEEP / 2, W_ROLL / 2 + 0.02));
  * soft: the prescribed inflation r_plan = W_SWEEP / 2 + 0.04; below it the move costs
    W_PEN * ((r_plan - c) / (r_plan - r_hard))^2 seconds, so the planner keeps r_plan wherever the site allows it and
    still finds a way from a pose that an execution error left inside the inflated band.
From a start pose closer than r_hard to a solid (an execution error), the hard limit starts at the start clearance
- 0.01 and recovers to r_hard with the distance from the start (0.25 m per m), so the plan can lead out of it.
"""
import heapq
import math

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

STEP = 0.02
KEY_XY = 0.03
KEY_TH = math.radians(7.5)
N_TH = int(round(2 * math.pi / KEY_TH))
W_PEN = 30.0
RECOVER = 0.25                 # m of required clearance regained per m of distance from a too-tight start
MIN_CLEAR = 0.10
N_SAMPLES = 16                 # samples per move (<= 0.019 m apart for the longest roll)
FRAC = np.linspace(0.0, 1.0, N_SAMPLES)


class Table:
    """PRIM_TABLE view: per start face string -> arrays of the primitives (names, dx, dy, dtheta_rad, t, next)."""

    def __init__(self, prim_table, names=None):
        self.names = list(names or sorted(prim_table))
        self.faces = sorted({f for n in self.names for f in prim_table[n]})
        self.arr = {}
        vmax = 0.0
        for f in self.faces:
            nm, dx, dy, dth, t, nf = [], [], [], [], [], []
            for n in self.names:
                c = prim_table[n].get(f)
                if c is None:
                    continue
                nm.append(n)
                dx.append(float(c["dx"]))
                dy.append(float(c["dy"]))
                dth.append(math.radians(float(c["dtheta_deg"])))
                t.append(float(c["t"]))
                nf.append(str(c["next_face"]))
            self.arr[f] = dict(names=nm, dx=np.array(dx), dy=np.array(dy), dth=np.array(dth), t=np.array(t), nf=nf)
            vmax = max(vmax, float(np.max(np.hypot(dx, dy) / np.array(t))))
        self.v_max = vmax


class Planner:
    def __init__(self, site_truth, table, r_plan, r_hard, block_ids=(), solid_classes=("T", "XN", "XL"), step=STEP):
        self.st = site_truth
        self.table = table
        self.r_plan, self.r_hard = float(r_plan), float(r_hard)
        self.step = step
        self.xs, self.ys, self.clr = site_truth.clearance_grid(step, block_ids, solid_classes)
        self.x0, self.y0 = float(self.xs[0]), float(self.ys[0])
        self.ny, self.nx = self.clr.shape
        self.block_ids, self.solid_classes = tuple(block_ids), tuple(solid_classes)
        self._hcache = {}

    # ---- grid lookups
    def cl(self, x, y):
        j = np.rint((np.asarray(x, float) - self.x0) / self.step).astype(int)
        i = np.rint((np.asarray(y, float) - self.y0) / self.step).astype(int)
        ok = (i >= 0) & (j >= 0) & (i < self.ny) & (j < self.nx)
        out = np.full(np.shape(i), -1.0)
        out[ok] = self.clr[i[ok], j[ok]]
        return out

    def _cell(self, x, y):
        j = int(np.clip(round((x - self.x0) / self.step), 0, self.nx - 1))
        i = int(np.clip(round((y - self.y0) / self.step), 0, self.ny - 1))
        return i, j

    def distance_field(self, goal):
        """Geodesic distance (m) from every cell with clearance >= r_hard to the goal cell (8-connected)."""
        key = (round(goal[0], 4), round(goal[1], 4))
        if key in self._hcache:
            return self._hcache[key]
        ok = self.clr >= self.r_hard
        H, Wd = ok.shape
        idx = np.arange(H * Wd).reshape(H, Wd)
        rows, cols, wts = [], [], []
        s, s2 = self.step, self.step * math.sqrt(2.0)
        for di, dj, w in ((0, 1, s), (1, 0, s), (1, 1, s2), (1, -1, s2)):
            i0, i1 = 0, H - di
            j0, j1 = max(0, -dj), Wd - max(0, dj)
            a, b = ok[i0:i1, j0:j1], ok[i0 + di:i1 + di, j0 + dj:j1 + dj]
            ia, ib = idx[i0:i1, j0:j1], idx[i0 + di:i1 + di, j0 + dj:j1 + dj]
            both = a & b
            rows.append(ia[both])
            cols.append(ib[both])
            wts.append(np.full(int(both.sum()), w))
        G = csr_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(H * Wd, H * Wd))
        gi, gj = self._cell(*goal)
        if not ok[gi, gj]:
            ii, jj = np.nonzero(ok)
            k = int(np.argmin((ii - gi) ** 2 + (jj - gj) ** 2))
            gi, gj = int(ii[k]), int(jj[k])
        dist = dijkstra(G, directed=False, indices=gi * Wd + gj).reshape(H, Wd)
        dist[~ok] = np.inf
        self._hcache[key] = dist
        return dist

    def h_dist(self, field, x, y):
        i, j = self._cell(x, y)
        d = field[i, j]
        if np.isfinite(d):
            return float(d)
        win = 12
        i0, i1, j0, j1 = max(0, i - win), min(self.ny, i + win + 1), max(0, j - win), min(self.nx, j + win + 1)
        sub = field[i0:i1, j0:j1]
        if not np.isfinite(sub).any():
            return float("inf")
        ii, jj = np.nonzero(np.isfinite(sub))
        dd = sub[ii, jj] + self.step * np.hypot(ii + i0 - i, jj + j0 - j)
        return float(dd.min())

    # ---- search
    def plan(self, pose, face, goal, tol, max_exp=60000, eps=1.6, banned=(), names=None):
        """Plan from pose (x, y, th) on `face` (string) to within tol of goal. Returns dict(found, path=[(name,
        from_node, to_node), ...], cost, expansions); a node = (x, y, th, face)."""
        field = self.distance_field(goal)
        tab = self.table
        vmax = tab.v_max
        gx, gy = float(goal[0]), float(goal[1])

        def h(x, y):
            return max(0.0, self.h_dist(field, x, y) - tol) / vmax

        def key(x, y, th, f):
            return (int(round(x / KEY_XY)), int(round(y / KEY_XY)), int(round((th % (2 * math.pi)) / KEY_TH)) % N_TH, f)

        x0, y0, th0 = float(pose[0]), float(pose[1]), float(pose[2])
        c_start = float(self.cl(np.array([x0]), np.array([y0]))[0])
        relax = c_start < self.r_hard + 0.01
        lim0 = max(MIN_CLEAR, c_start - 0.01)
        start = (x0, y0, th0, face)
        k0 = key(*start)
        best_g = {k0: 0.0}
        parent = {k0: None}
        node_of = {k0: start}
        openl = [(eps * h(x0, y0), 0.0, k0)]
        n_exp = 0
        banned = set(banned)
        allowed = set(names) if names else None
        span = max(1e-6, self.r_plan - self.r_hard)
        while openl and n_exp < max_exp:
            f_, g, k = heapq.heappop(openl)
            if g > best_g.get(k, float("inf")) + 1e-9:
                continue
            x, y, th, fc = node_of[k]
            n_exp += 1
            if math.hypot(x - gx, y - gy) <= tol:
                path = []
                kk = k
                while parent[kk] is not None:
                    pk, nm = parent[kk]
                    path.append((nm, node_of[pk], node_of[kk]))
                    kk = pk
                path.reverse()
                return dict(found=True, path=path, cost=g, expansions=n_exp)
            A = tab.arr.get(fc)
            if A is None:
                continue
            c, s = math.cos(th), math.sin(th)
            ex = x + c * A["dx"] - s * A["dy"]
            ey = y + s * A["dx"] + c * A["dy"]
            PX = x + FRAC[None, :] * (ex - x)[:, None]
            PY = y + FRAC[None, :] * (ey - y)[:, None]
            CC = self.cl(PX, PY)
            if relax:
                LIM = np.minimum(self.r_hard, lim0 + RECOVER * np.hypot(PX - x0, PY - y0))
                ok = np.all(CC >= LIM, axis=1)
            else:
                ok = CC.min(axis=1) >= self.r_hard
            cmin = CC.min(axis=1)
            q = np.clip((self.r_plan - cmin) / span, 0.0, 1.0)
            pen = W_PEN * q * q
            for i in np.nonzero(ok)[0]:
                nm = A["names"][i]
                if allowed is not None and nm not in allowed:
                    continue
                if (k[:3], nm) in banned:
                    continue
                g2 = g + float(A["t"][i]) + float(pen[i])
                nx_, ny_ = float(ex[i]), float(ey[i])
                th2 = th + float(A["dth"][i])
                k2 = key(nx_, ny_, th2, A["nf"][i])
                if g2 < best_g.get(k2, float("inf")) - 1e-9:
                    best_g[k2] = g2
                    parent[k2] = (k, nm)
                    node_of[k2] = (nx_, ny_, th2, A["nf"][i])
                    heapq.heappush(openl, (g2 + eps * h(nx_, ny_), g2, k2))
        return dict(found=False, path=[], cost=None, expansions=n_exp)

    def route_length(self, a, b):
        """Geodesic grid distance between two points (m) on this planner's admissible set."""
        field = self.distance_field(b)
        return self.h_dist(field, a[0], a[1])
