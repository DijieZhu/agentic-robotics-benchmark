#!/usr/bin/env python3
"""twin_slice: the robot-terms map of a site twin (the "slice" of C3 in twin_contract.md).

Where in the twin can the probe robot stand, where must it crouch, where is it blocked? The same code runs on your
twin (check_twin --slice, or this file's command line), on the hidden true twin and in the grader, which computes the
map of your twin at the true cells (mapped into your frame by the alignment) and in your twin's own units: the
robot's sizes below are absolute, so a twin built too large or too small gives the robot more or less room.

    shapes = twin_shapes(model, data, geoms)        # the collision geometry of a compiled twin at data's pose
    shapes = transform_shapes(shapes, R, t)         # rigidly into another frame: x -> R x + t
    grid = Grid(origin_xy, cell_m, nx, ny)          # cells (i, j) = [x0 + i h, x0 + (i+1) h] x [y0 + j h, ...]
    ext = column_extents(shapes, grid, z_ref, params, deltas=(0.0,))   # per cell: clearance and support
    cls = classify(ext["clear"][0], ext["support"][0], z_ref, grid.cell_m, params)   # 0 none 1 stand 2 crouch 3 blocked
    m = map_twin(model, data)                       # all of it for one compiled twin, the reference level from its own
                                                    # static geometry ("auto": its floor)
    render_png(path, cls, grid)                     # a picture with a legend

Definitions (every length in the twin's own metres, z up, gravity along -z):
- Cell (i, j) is the CLOSED square above; its reference level z_f = z_ref[i, j] is the walking surface there (the
  grader uses the true FLOOR, mapped into your frame: a dock top or a kerb is an obstacle standing on it); NaN means
  no reference: class 0.
- A shape MEETS the column of a cell between heights a and b when some point of it lies over the cell (anywhere in
  the square, not only above its centre) at a height in [a, b]. This is computed exactly, not sampled: for every
  collision shape the lowest and highest point over each cell's square are found from the shape's vertices, the
  crossings of its edges with the cell borders and its height over the cell corners (boxes, convex meshes as their
  convex hulls), or analytically (spheres, capsules, cylinders, ellipsoids, planes), so a 2 cm
  thin wall that misses every cell centre is still found in every cell it passes.
- clearance = height above z_f of the lowest point, over the cell, of the geometry that reaches above
  z_f + low_from_m (exactly low_from_m when it rises from below that level); inf when nothing reaches above it within
  clear_cap_m (values at or above the cap are reported as inf).
- support: some shape's highest point over the cell lies within support_tol_m of z_f.
- O_low = cells with clearance <= h_crouch_m or without support; O_high the same with h_stand_m.
- Configuration space: a cell is in dilated O when the disc of radius r_robot_m (the robot's half-width plus margin)
  centred on the cell's centre meets the square of a cell of O (cells without a reference count as O).
- class: blocked (3) in dilated O_low; else crouch (2) in dilated O_high; else stand (1).
The twin is taken as submitted (qpos0: units where you put them, gates at their photographed position); visual geoms
(contype = conaffinity = 0) are ignored.

numpy + mujoco (+ twin_gate.py next to this file for the convex pieces); Pillow only for the picture.
"""
import json
import math
import os
import sys

import numpy as np

FORMAT = "site-slice/1"
HERE = os.path.dirname(os.path.abspath(__file__))
CLASS_NAMES = ("none", "stand", "crouch", "blocked")
NONE, STAND, CROUCH, BLOCKED = 0, 1, 2, 3
# the probe robot's terms (the grader's configuration holds the same numbers; robot_card.md states them)
DEFAULT_PARAMS = {"cell_m": 0.05, "low_from_m": 0.06, "support_tol_m": 0.06, "h_crouch_m": 0.26, "h_stand_m": 0.33,
                  "r_robot_m": 0.17, "clear_cap_m": 1.0}
_EPS_AXIS = 1e-12          # an axis component below this is taken as 0 in the analytic sections
_EPS_PARALLEL = 1e-6       # a cylinder axis this close to a cutting plane is taken as parallel (error < 1e-6 hl)
_TOL = 1e-9                # m: node-in-triangle tolerance (a node on a shared edge counts for both triangles)
_BLOCK = 2_000_000         # candidate entries per vectorised block
_BOX_EDGES = np.array([[a, a ^ b] for a in range(8) for b in (1, 2, 4) if not a & b], np.int64)
_BOX_TRIS = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                      [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
_SWAP = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def _gate():
    try:
        import twin_gate
    except ImportError:  # pragma: no cover - twin_gate sits next to this file
        sys.path.insert(0, HERE)
        import twin_gate
    return twin_gate


def params_from(cfg=None):
    """The map's parameters: DEFAULT_PARAMS updated by cfg (e.g. the grader's C3.slice section; unknown keys are
    ignored)."""
    p = dict(DEFAULT_PARAMS)
    for k in DEFAULT_PARAMS:
        if cfg is not None and k in cfg:
            p[k] = float(cfg[k])
    return p


# ---------------------------------------------------------------------------------------------------------------
# the grid

class Grid:
    """nx x ny closed square cells of side cell_m; cell (i, j) = [x0 + i h, x0 + (i+1) h] x [y0 + j h, y0 + (j+1) h].
    Internally every shape is moved into the grid's local frame (x - x0, y - y0), where the cell borders are the lines
    x = i h and y = j h."""

    def __init__(self, origin_xy, cell_m, nx, ny):
        self.origin = np.asarray(origin_xy, float).reshape(2)
        self.cell_m = float(cell_m)
        self.nx, self.ny = int(nx), int(ny)
        if not (self.cell_m > 0 and self.nx > 0 and self.ny > 0):
            raise ValueError("a grid needs a positive cell size and at least one cell")

    @classmethod
    def covering(cls, lo_xy, hi_xy, cell_m, margin_m=0.0):
        """The grid on multiples of cell_m that covers [lo, hi] grown by margin_m."""
        h = float(cell_m)
        lo = np.floor((np.asarray(lo_xy, float) - margin_m) / h) * h
        hi = np.ceil((np.asarray(hi_xy, float) + margin_m) / h) * h
        n = np.maximum(1, np.round((hi - lo) / h).astype(int))
        return cls(lo, h, int(n[0]), int(n[1]))

    @property
    def shape(self):
        return (self.nx, self.ny)

    def centres(self):
        """Cell centres (X, Y), each (nx, ny), world frame."""
        h = self.cell_m
        x = self.origin[0] + (np.arange(self.nx) + 0.5) * h
        y = self.origin[1] + (np.arange(self.ny) + 0.5) * h
        return np.meshgrid(x, y, indexing="ij")

    def scaled(self, f):
        """The same cells in a frame whose lengths are f times these (origin and size multiplied by f)."""
        return Grid(self.origin * f, self.cell_m * f, self.nx, self.ny)

    def as_dict(self):
        return {"origin_xy": [float(v) for v in self.origin], "cell_m": self.cell_m, "nx": self.nx, "ny": self.ny}


# ---------------------------------------------------------------------------------------------------------------
# the shapes of a twin

class Shape:
    """One collision shape: kind 'convex' (data: a twin_gate.Piece: box, hull, sphere, capsule, cylinder,
    ellipsoid) or 'plane' (data: {'c', 'n'}: the half-space n . (x - c) <= 0, infinite as in MuJoCo's collisions;
    a twin has no height field: the gate refuses them). geom = the source geom id (-1 when made by hand)."""

    def __init__(self, kind, data, geom=-1):
        self.kind = kind
        self.data = data
        self.geom = int(geom)

    def bbox(self):
        if self.kind == "convex":
            return self.data.bbox()
        return np.full(3, -np.inf), np.full(3, np.inf)

    def transformed(self, R, t):
        R = np.asarray(R, float).reshape(3, 3)
        t = np.asarray(t, float).reshape(3)
        if self.kind == "convex":
            return Shape("convex", self.data.transformed(R, t), self.geom)
        return Shape("plane", {"c": R @ self.data["c"] + t, "n": R @ self.data["n"]}, self.geom)


def collision_geoms(model, exclude_bodies=()):
    """Every collision geom of the model ((contype, conaffinity) != (0, 0)) outside the subtrees of exclude_bodies."""
    TG = _gate()
    ex = set()
    for b in exclude_bodies:
        ex.update(TG._subtree(model, int(b)))
        ex.add(int(b))
    return [g for g in range(model.ngeom) if TG.is_collision_geom(model, g) and int(model.geom_bodyid[g]) not in ex]


def twin_shapes(model, data, geoms=None):
    """Shapes of the given collision geoms (default: all) at data's pose (after mj_forward), world frame, geom
    order. A mesh is its convex hull (what MuJoCo collides with)."""
    import mujoco
    TG = _gate()
    T = mujoco.mjtGeom
    out = []
    for g in sorted(collision_geoms(model) if geoms is None else (int(v) for v in geoms)):
        t = int(model.geom_type[g])
        if t == int(T.mjGEOM_PLANE):
            Rg = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
            out.append(Shape("plane", {"c": np.asarray(data.geom_xpos[g], float).copy(), "n": Rg[:, 2].copy()}, g))
        else:
            pc = TG.geom_piece(model, data, g)
            if pc is not None:
                out.append(Shape("convex", pc, g))
    return out


def transform_shapes(shapes, R, t):
    """Every shape under x -> R x + t (R a proper rotation; lengths unchanged)."""
    return [s.transformed(R, t) for s in shapes]


# ---------------------------------------------------------------------------------------------------------------
# exact z ranges of a shape over the cells

def _closed_bins(u):
    """For coordinates u in cell units: (lower, upper) index of the closed cells containing them (equal unless u is
    on a border, where it belongs to both neighbours)."""
    fl = np.floor(u)
    return (np.ceil(u) - 1).astype(np.int64), fl.astype(np.int64)


class _Block:
    """zmin / zmax of one shape over the cells [i0, i1) x [j0, j1) of a grid (local frame, cell h)."""

    def __init__(self, i0, i1, j0, j1):
        self.i0, self.i1, self.j0, self.j1 = i0, i1, j0, j1
        self.zmin = np.full((i1 - i0, j1 - j0), np.inf)
        self.zmax = np.full((i1 - i0, j1 - j0), -np.inf)

    def add(self, I, J, Z):
        """Point candidates: cell indices and heights (points of the shape over those cells)."""
        ok = (I >= self.i0) & (I < self.i1) & (J >= self.j0) & (J < self.j1) & np.isfinite(Z)
        if not np.any(ok):
            return
        ii, jj, zz = I[ok] - self.i0, J[ok] - self.j0, Z[ok]
        np.minimum.at(self.zmin, (ii, jj), zz)
        np.maximum.at(self.zmax, (ii, jj), zz)

    def add_points(self, P, h):
        """Points (n, 3) of the shape: each counts for every closed cell that contains it (1, 2 or 4 cells)."""
        if not len(P):
            return
        il, ih = _closed_bins(P[:, 0] / h)
        jl, jh = _closed_bins(P[:, 1] / h)
        for I in (il, ih):
            for J in (jl, jh):
                self.add(I, J, P[:, 2])

    def add_nodes(self, k0, l0, z0, z1):
        """Heights on the vertical lines through the nodes (k0 + a, l0 + b) (arrays (na, nb), NaN where the line misses
        the shape): each node counts for its four cells."""
        for zz, red in ((z0, np.minimum), (z1, np.maximum)):
            A = np.where(np.isnan(zz), np.inf if red is np.minimum else -np.inf, zz)
            na, nb = A.shape
            for da in (0, 1):
                for db in (0, 1):
                    # node (k0 + a, l0 + b) is corner (da, db) of cell (k0 + a - da, l0 + b - db)
                    ci0, cj0 = k0 - da, l0 - db
                    a0 = max(self.i0 - ci0, 0)
                    a1 = min(self.i1 - ci0, na)
                    b0 = max(self.j0 - cj0, 0)
                    b1 = min(self.j1 - cj0, nb)
                    if a0 >= a1 or b0 >= b1:
                        continue
                    tgt = (self.zmin if red is np.minimum else self.zmax)
                    sl = (slice(ci0 + a0 - self.i0, ci0 + a1 - self.i0), slice(cj0 + b0 - self.j0, cj0 + b1 - self.j0))
                    tgt[sl] = red(tgt[sl], A[a0:a1, b0:b1])


def _cell_range(lo, hi, h, n):
    """Closed cells [a, b) along one axis whose squares meet [lo, hi] (clipped to [0, n))."""
    a = max(0, int(math.ceil(lo / h)) - 1)
    b = min(n, int(math.floor(hi / h)) + 1)
    return a, b


def _mesh_into(blk, V, E, T, h):
    """Exact zmin / zmax of a closed triangle mesh (convex or not) over the cells: the extremes of z over the solid
    within a cell's column are attained at a vertex of the solid in the cell, where an edge crosses the cell border,
    or where a corner line of the cell pierces a face."""
    blk.add_points(V, h)
    # edge crossings with the lines x = k h (axis 0) and y = l h (axis 1)
    for ax in (0, 1):
        P, Q = V[E[:, 0]], V[E[:, 1]]
        up, uq = P[:, ax] / h, Q[:, ax] / h
        lo, hi = np.minimum(up, uq), np.maximum(up, uq)
        n_lines = (blk.i1 if ax == 0 else blk.j1)
        k0 = np.maximum(np.ceil(lo), (blk.i0 if ax == 0 else blk.j0)).astype(np.int64)
        k1 = np.minimum(np.floor(hi), n_lines).astype(np.int64)
        cnt = np.where(up != uq, np.maximum(k1 - k0 + 1, 0), 0)
        sel = np.nonzero(cnt)[0]
        if not len(sel):
            continue
        cnt_s = cnt[sel]
        csum = np.cumsum(cnt_s)
        s0 = 0
        while s0 < len(sel):
            base = csum[s0 - 1] if s0 else 0
            s1 = max(s0 + 1, int(np.searchsorted(csum, base + _BLOCK, side="right")))
            es = sel[s0:s1]
            c = cnt_s[s0:s1]
            rep = np.repeat(es, c)
            off = np.arange(int(c.sum()), dtype=np.int64) - np.repeat(np.cumsum(c) - c, c)
            k = k0[rep] + off
            Pr, Qr = P[rep], Q[rep]
            t = (k * h - Pr[:, ax]) / (Qr[:, ax] - Pr[:, ax])
            t = np.clip(t, 0.0, 1.0)
            X = Pr + t[:, None] * (Qr - Pr)
            other = 1 - ax
            ol, oh = _closed_bins(X[:, other] / h)
            for Kc in (k - 1, k):
                for O in (ol, oh):
                    if ax == 0:
                        blk.add(Kc, O, X[:, 2])
                    else:
                        blk.add(O, Kc, X[:, 2])
            s0 = s1
    # corner lines through the nodes: plane values of the faces whose projection contains the node
    lo_, hi_ = _raster_lattice(V, T, h, 0.0, blk.i0, blk.i1, blk.j0, blk.j1)
    blk.add_nodes(blk.i0, blk.j0, lo_, hi_)


def _raster_lattice(V, T, h, off, k0, k1, l0, l1):
    """For the lattice points (k h + off, l h + off), k in [k0, k1], l in [l0, l1]: the lowest and highest plane
    value of the triangles T of V whose (closed) projection contains the point: arrays (k1 - k0 + 1, l1 - l0 + 1),
    NaN where no triangle covers the point. Faces seen edge-on from above are skipped (their points on the line are
    on neighbouring faces' edges); a value is clipped to its triangle's height range (no extrapolation)."""
    out_lo = np.full((k1 - k0 + 1, l1 - l0 + 1), np.inf)
    out_hi = np.full_like(out_lo, -np.inf)
    A, B, C = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    cr = np.cross(B - A, C - A)
    keep = np.abs(cr[:, 2]) > 1e-15 * np.maximum(1.0, np.linalg.norm(cr, axis=1))
    if not np.any(keep):
        return np.full(out_lo.shape, np.nan), np.full(out_lo.shape, np.nan)
    A, B, C, cr = A[keep], B[keep], C[keep], cr[keep]
    P2 = np.stack([A[:, :2], B[:, :2], C[:, :2]], axis=1) - off              # (m, 3, 2), lattice at k h
    zlo = np.minimum(np.minimum(A[:, 2], B[:, 2]), C[:, 2])
    zhi = np.maximum(np.maximum(A[:, 2], B[:, 2]), C[:, 2])
    lo, hi = P2.min(axis=1), P2.max(axis=1)
    ka = np.maximum(np.ceil((lo[:, 0] - _TOL) / h), k0).astype(np.int64)
    kb = np.minimum(np.floor((hi[:, 0] + _TOL) / h), k1).astype(np.int64)
    la = np.maximum(np.ceil((lo[:, 1] - _TOL) / h), l0).astype(np.int64)
    lb = np.minimum(np.floor((hi[:, 1] + _TOL) / h), l1).astype(np.int64)
    nk = np.maximum(kb - ka + 1, 0)
    nl = np.maximum(lb - la + 1, 0)
    cnt = nk * nl
    sel = np.nonzero(cnt)[0]
    if not len(sel):
        return np.full(out_lo.shape, np.nan), np.full(out_lo.shape, np.nan)
    sgn = np.sign(cr[:, 2])
    En, Eo = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        e = P2[:, b] - P2[:, a]
        L = np.linalg.norm(e, axis=1)
        nrm = np.c_[-e[:, 1], e[:, 0]] * (sgn / np.where(L > 0, L, 1.0))[:, None]   # inward unit normal
        En.append(nrm)
        Eo.append(np.einsum("ij,ij->i", nrm, P2[:, a]))
    En = np.stack(En, axis=1)
    Eo = np.stack(Eo, axis=1)
    Ax, Ay = A[:, 0] - off, A[:, 1] - off
    cnt_s = cnt[sel]
    csum = np.cumsum(cnt_s)
    s0 = 0
    while s0 < len(sel):
        base = csum[s0 - 1] if s0 else 0
        s1 = max(s0 + 1, int(np.searchsorted(csum, base + _BLOCK, side="right")))
        fs = sel[s0:s1]
        c = cnt_s[s0:s1]
        rep = np.repeat(fs, c)
        o_ = np.arange(int(c.sum()), dtype=np.int64) - np.repeat(np.cumsum(c) - c, c)
        kk = ka[rep] + o_ // nl[rep]
        ll = la[rep] + o_ % nl[rep]
        x, y = kk * h, ll * h
        inside = np.ones(len(rep), bool)
        for e in range(3):
            inside &= En[rep, e, 0] * x + En[rep, e, 1] * y - Eo[rep, e] >= -_TOL
        if np.any(inside):
            f, x, y, kk, ll = rep[inside], x[inside], y[inside], kk[inside], ll[inside]
            z = A[f, 2] - (cr[f, 0] * (x - Ax[f]) + cr[f, 1] * (y - Ay[f])) / cr[f, 2]
            z = np.clip(z, zlo[f], zhi[f])
            np.minimum.at(out_lo, (kk - k0, ll - l0), z)
            np.maximum.at(out_hi, (kk - k0, ll - l0), z)
        s0 = s1
    return (np.where(np.isfinite(out_lo), out_lo, np.nan), np.where(np.isfinite(out_hi), out_hi, np.nan))


def _top_bottom(pc):
    """The highest and the lowest point of a smooth convex piece ((3,), (3,))."""
    c, R, p, k = pc.c, pc.R, pc.p, pc.kind
    ez = np.array([0.0, 0.0, 1.0])
    if k == "sphere":
        return c + p[0] * ez, c - p[0] * ez
    if k == "ellipsoid":
        Qi = R @ np.diag(p ** 2) @ R.T
        v = Qi[:, 2] / math.sqrt(Qi[2, 2])
        return c + v, c - v
    a = R[:, 2]
    s = float(np.sign(a[2]))
    if k == "capsule":
        return c + p[1] * s * a + p[0] * ez, c - p[1] * s * a - p[0] * ez
    w = ez - a[2] * a                                              # cylinder: rim point toward +z
    nw = float(np.linalg.norm(w))
    rim = p[0] * w / nw if nw > _EPS_AXIS else np.zeros(3)
    return c + p[1] * s * a + rim, c - p[1] * s * a - rim


def _section_x(pc, X):
    """Cross-sections of a smooth convex piece by the vertical planes x = X (array): the highest and lowest points
    (y_top, z_top, y_bot, z_bot), NaN where the plane misses the piece."""
    X = np.asarray(X, float)
    n = len(X)
    nan = np.full(n, np.nan)
    c, R, p, k = pc.c, pc.R, pc.p, pc.kind
    dx = X - c[0]
    if k == "sphere":
        w2 = p[0] ** 2 - dx * dx
        ok = w2 >= 0
        w = np.sqrt(np.where(ok, w2, 0.0))
        y = np.where(ok, c[1], np.nan)
        return y, np.where(ok, c[2] + w, np.nan), y, np.where(ok, c[2] - w, np.nan)
    if k == "ellipsoid":
        Q = R @ np.diag(1.0 / p ** 2) @ R.T
        Qi = R @ np.diag(p ** 2) @ R.T
        M = Q[1:, 1:]
        Minv = np.linalg.inv(M)
        w0 = -dx[:, None] * (Minv @ Q[1:, 0])[None, :]
        rho = 1.0 - dx * dx / Qi[0, 0]
        ok = rho >= 0
        g = Minv[:, 1]
        sc = np.sqrt(np.where(ok, rho, 0.0) / g[1])
        top = w0 + sc[:, None] * g[None, :]
        bot = w0 - sc[:, None] * g[None, :]
        return (np.where(ok, c[1] + top[:, 0], np.nan), np.where(ok, c[2] + top[:, 1], np.nan),
                np.where(ok, c[1] + bot[:, 0], np.nan), np.where(ok, c[2] + bot[:, 1], np.nan))
    a = R[:, 2]
    r, hl = float(p[0]), float(p[1])
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
    if k == "capsule":
        out = []
        for sign in (1.0, -1.0):                                 # +1: the top, -1: the bottom
            if abs(ax) > _EPS_AXIS:
                ta, tb = (dx - r) / ax, (dx + r) / ax
                tlo = np.maximum(np.minimum(ta, tb), -hl)
                thi = np.minimum(np.maximum(ta, tb), hl)
                ok = tlo <= thi
                nrm = math.hypot(ax, az)
                dstar = -sign * r * az * math.copysign(1.0, ax) / nrm if nrm > 0 else 0.0
                t = np.clip((dx - dstar) / ax, tlo, thi)
            else:
                ok = np.abs(dx) <= r
                t = np.full(n, sign * hl * float(np.sign(az)))
            delta = dx - t * ax
            w = np.sqrt(np.maximum(r * r - delta * delta, 0.0))
            y = c[1] + t * ay
            z = c[2] + t * az + sign * w
            out += [np.where(ok, y, np.nan), np.where(ok, z, np.nan)]
        return out[0], out[1], out[2], out[3]
    if k != "cylinder":
        raise ValueError(k)
    ap = np.array([ay, az])
    if abs(ax) <= _EPS_PARALLEL:
        # the axis lies in the plane: the section is a rectangle |alpha| <= hl along the axis, |beta| <= rho across
        rho2 = r * r - dx * dx
        ok = rho2 >= 0
        rho = np.sqrt(np.where(ok, rho2, 0.0))
        ah = ap / np.linalg.norm(ap)
        nh = np.array([-ah[1], ah[0]])
        cy, cz = [], []
        for sa in (-1.0, 1.0):
            for sb in (-1.0, 1.0):
                cy.append(sa * hl * ah[0] + sb * rho * nh[0])
                cz.append(sa * hl * ah[1] + sb * rho * nh[1])
        cy, cz = np.stack(cy, axis=1), np.stack(cz, axis=1)
        it, ib = np.argmax(cz, axis=1), np.argmin(cz, axis=1)
        rows = np.arange(n)
        return (np.where(ok, c[1] + cy[rows, it], np.nan), np.where(ok, c[2] + cz[rows, it], np.nan),
                np.where(ok, c[1] + cy[rows, ib], np.nan), np.where(ok, c[2] + cz[rows, ib], np.nan))
    # the axis crosses the plane: an ellipse u' M u <= r^2 around the axis point w0 = t0 a', cut by the strip
    # -hl - t0 <= a' . u <= hl - t0 between the end caps
    t0 = dx / ax
    w0 = t0[:, None] * ap[None, :]
    # M = I - a' a'^T has the eigenvalues ax^2 (along a') and 1 (across): its inverse without cancellation
    na = float(np.linalg.norm(ap))
    if na > 0:
        ah = ap / na
        nh_ = np.array([-ah[1], ah[0]])
        Minv = np.outer(nh_, nh_) + np.outer(ah, ah) / (ax * ax)
    else:
        Minv = np.eye(2)
    g = Minv[:, 1]
    u1 = r * g / math.sqrt(g[1])                                     # the ellipse's top, relative to w0
    L = [-hl - t0, hl - t0]
    na2 = float(ap @ ap)
    cands_t, cands_b = [], []
    in1 = (ap @ u1 >= L[0] - 1e-15) & (ap @ u1 <= L[1] + 1e-15)
    cands_t.append((np.where(in1, u1[0], np.nan), np.where(in1, u1[1], np.nan)))
    in2 = (-ap @ u1 >= L[0] - 1e-15) & (-ap @ u1 <= L[1] + 1e-15)
    cands_b.append((np.where(in2, -u1[0], np.nan), np.where(in2, -u1[1], np.nan)))
    if na2 > _EPS_AXIS:
        nh = np.array([-ap[1], ap[0]]) / math.sqrt(na2)
        st = 1.0 if nh[1] >= 0 else -1.0
        for Lk in L:
            tau2 = r * r - Lk * Lk * (ax * ax) / na2
            ok = tau2 >= 0
            tau = np.sqrt(np.where(ok, tau2, 0.0))
            base_y = Lk * ap[0] / na2
            base_z = Lk * ap[1] / na2
            for sgn, lst in ((st, cands_t), (-st, cands_b)):
                lst.append((np.where(ok, base_y + sgn * tau * nh[0], np.nan),
                            np.where(ok, base_z + sgn * tau * nh[1], np.nan)))
    out = []
    for lst, pick in ((cands_t, np.nanargmax), (cands_b, np.nanargmin)):
        Yc = np.stack([v[0] for v in lst], axis=1)
        Zc = np.stack([v[1] for v in lst], axis=1)
        anyok = np.any(np.isfinite(Zc), axis=1)
        idx = pick(np.where(anyok[:, None], Zc, 0.0), axis=1)
        rows = np.arange(n)
        out += [np.where(anyok, c[1] + w0[:, 0] + Yc[rows, idx], np.nan),
                np.where(anyok, c[2] + w0[:, 1] + Zc[rows, idx], np.nan)]
    return out[0], out[1], out[2], out[3]


def _smooth_into(blk, pc, h):
    """Exact zmin / zmax of a smooth convex piece over the cells: over a closed square, the highest point of a convex
    body is its top point if that lies over the square, else on one of the square's sides, where it is the top of the
    body's cross-section by that side's vertical plane if that lies within the side, else at a corner (the height
    along a side is concave). Likewise for the lowest point."""
    tp, bt = _top_bottom(pc)
    blk.add_points(np.vstack([tp, bt]), h)
    lo, hi = pc.bbox()
    for ax in (0, 1):
        q = pc if ax == 0 else pc.transformed(_SWAP, np.zeros(3))
        a0, a1 = (blk.i0, blk.i1) if ax == 0 else (blk.j0, blk.j1)
        ks = np.arange(max(a0, int(math.ceil(lo[ax] / h))), min(a1, int(math.floor(hi[ax] / h))) + 1)
        if not len(ks):
            continue
        yt, zt, yb, zb = _section_x(q, ks * h)
        for oy, oz in ((yt, zt), (yb, zb)):
            ok = np.isfinite(oz)
            if not np.any(ok):
                continue
            kk, oy, oz = ks[ok], oy[ok], oz[ok]
            ol, oh = _closed_bins(oy / h)
            for Kc in (kk - 1, kk):
                for O in (ol, oh):
                    if ax == 0:
                        blk.add(Kc, O, oz)
                    else:
                        blk.add(O, Kc, oz)
    k0, k1 = blk.i0, blk.i1 + 1
    l0, l1 = blk.j0, blk.j1 + 1
    K, L = np.meshgrid(np.arange(k0, k1), np.arange(l0, l1), indexing="ij")
    z0, z1 = pc.z_interval(K.ravel() * h, L.ravel() * h)
    blk.add_nodes(k0, l0, z0.reshape(K.shape), z1.reshape(K.shape))


def _sphere_block(blk, pc, h):
    c, r = pc.c, float(pc.p[0])
    I = np.arange(blk.i0, blk.i1)
    J = np.arange(blk.j0, blk.j1)
    ex = np.maximum(0.0, np.maximum(I * h - c[0], c[0] - (I + 1) * h))
    ey = np.maximum(0.0, np.maximum(J * h - c[1], c[1] - (J + 1) * h))
    d2 = ex[:, None] ** 2 + ey[None, :] ** 2
    ok = d2 <= r * r
    w = np.sqrt(np.where(ok, r * r - d2, 0.0))
    blk.zmin = np.where(ok, c[2] - w, np.inf)
    blk.zmax = np.where(ok, c[2] + w, -np.inf)


def _plane_block(blk, sh, h):
    c, n = sh.data["c"], sh.data["n"] / np.linalg.norm(sh.data["n"])
    d = float(n @ c)
    K = np.arange(blk.i0, blk.i1 + 1) * h
    L = np.arange(blk.j0, blk.j1 + 1) * h
    F = n[0] * K[:, None] + n[1] * L[None, :]                     # n_xy . node
    corners = [F[:-1, :-1], F[1:, :-1], F[:-1, 1:], F[1:, 1:]]
    if n[2] > 1e-12:                                                 # solid below z = (d - F) / n_z
        blk.zmax = np.max([(d - f) / n[2] for f in corners], axis=0)
        blk.zmin = np.full_like(blk.zmax, -np.inf)
    elif n[2] < -1e-12:                                              # solid above
        blk.zmin = np.min([(d - f) / n[2] for f in corners], axis=0)
        blk.zmax = np.full_like(blk.zmin, np.inf)
    else:
        inside = np.min(corners, axis=0) <= d
        blk.zmin = np.where(inside, -np.inf, np.inf)
        blk.zmax = np.where(inside, np.inf, -np.inf)


def shape_ranges(shape, grid):
    """(i0, j0, zmin, zmax) of one shape (world frame) over the grid's cells: zmin / zmax (ni, nj) the lowest / highest
    point of the shape over each closed cell of the block [i0, i0 + ni) x [j0, j0 + nj) (inf / -inf where it is not
    over the cell); None when the shape is over no cell."""
    h = grid.cell_m
    t = np.array([-grid.origin[0], -grid.origin[1], 0.0])
    if shape.kind == "plane":
        sh = shape.transformed(np.eye(3), t)
        blk = _Block(0, grid.nx, 0, grid.ny)
        _plane_block(blk, sh, h)
        return 0, 0, blk.zmin, blk.zmax
    sh = shape.transformed(np.eye(3), t)
    lo, hi = sh.bbox()
    i0, i1 = _cell_range(lo[0], hi[0], h, grid.nx)
    j0, j1 = _cell_range(lo[1], hi[1], h, grid.ny)
    if i0 >= i1 or j0 >= j1:
        return None
    blk = _Block(i0, i1, j0, j1)
    pc = sh.data
    if pc.kind == "box":
        _mesh_into(blk, pc.V, _BOX_EDGES, _BOX_TRIS, h)
    elif pc.kind == "hull":
        F = np.asarray(pc.F, np.int64)
        E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
        _mesh_into(blk, pc.V, E, F, h)
    elif pc.kind == "sphere":
        _sphere_block(blk, pc, h)
    else:
        _smooth_into(blk, pc, h)
    if not np.any(blk.zmax >= blk.zmin):
        return None
    return i0, j0, blk.zmin, blk.zmax


def column_extents(shapes, grid, z_ref, params=None, deltas=(0.0,)):
    """Per cell and per reference offset delta (the reference level z_ref + delta): the clearance and the support.
    Returns {"clear": (nd, nx, ny) float (inf: nothing within clear_cap_m), "support": (nd, nx, ny) bool,
    "deltas", "n_shapes_used"}. z_ref (nx, ny) in the shapes' frame; NaN = no reference."""
    p = params_from(params)
    lo_m, tol, cap = p["low_from_m"], p["support_tol_m"], p["clear_cap_m"]
    z_ref = np.asarray(z_ref, float).reshape(grid.nx, grid.ny)
    deltas = [float(d) for d in deltas]
    nd = len(deltas)
    clear = np.full((nd, grid.nx, grid.ny), np.inf)
    support = np.zeros((nd, grid.nx, grid.ny), bool)
    fin = np.isfinite(z_ref)
    used = 0
    if not np.any(fin):
        return {"clear": clear, "support": support, "deltas": deltas, "n_shapes_used": 0}
    zr_lo = float(np.min(z_ref[fin])) + min(deltas) - tol
    zr_hi = float(np.max(z_ref[fin])) + max(deltas) + max(cap, tol)
    for sh in shapes:
        blo, bhi = sh.bbox()
        if blo[2] > zr_hi or bhi[2] < zr_lo:
            continue                                   # entirely above the cap or below every support window
        r = shape_ranges(sh, grid)
        if r is None:
            continue
        used += 1
        i0, j0, zmin, zmax = r
        ni, nj = zmin.shape
        sl = (slice(i0, i0 + ni), slice(j0, j0 + nj))
        zr = z_ref[sl]
        valid = zmax >= zmin
        for k, d in enumerate(deltas):
            base = zr + d
            a = base + lo_m
            with np.errstate(invalid="ignore"):
                reach = valid & (zmax >= a)
                cand = np.where(reach, np.maximum(zmin, a) - base, np.inf)
                cand = np.where(cand < cap, cand, np.inf)
                sup = valid & (zmax >= base - tol) & (zmax <= base + tol)
            np.minimum(clear[k][sl], cand, out=clear[k][sl])
            support[k][sl] |= sup
    return {"clear": clear, "support": support, "deltas": deltas, "n_shapes_used": used}


# ---------------------------------------------------------------------------------------------------------------
# classes

def disc_offsets(cell_m, radius_m):
    """Cell offsets (di, dj) whose square meets the disc of radius_m around a cell's centre."""
    h, r = float(cell_m), float(radius_m)
    n = int(math.floor(r / h + 0.5)) + 1
    out = []
    for di in range(-n, n + 1):
        ex = max(0.0, abs(di) - 0.5) * h
        for dj in range(-n, n + 1):
            ey = max(0.0, abs(dj) - 0.5) * h
            if ex * ex + ey * ey <= r * r:
                out.append((di, dj))
    return out


def dilate(mask, cell_m, radius_m):
    """Configuration-space dilation of a cell mask (nx, ny): a cell is set when the disc of radius_m around its
    centre meets the square of a set cell (cells outside the grid count as unset)."""
    mask = np.asarray(mask, bool)
    out = mask.copy()
    nx, ny = mask.shape
    for di, dj in disc_offsets(cell_m, radius_m):
        if di == 0 and dj == 0:
            continue
        if abs(di) >= nx or abs(dj) >= ny:
            continue
        dst = (slice(max(di, 0), nx + min(di, 0)), slice(max(dj, 0), ny + min(dj, 0)))
        src = (slice(max(-di, 0), nx + min(-di, 0)), slice(max(-dj, 0), ny + min(-dj, 0)))
        out[dst] |= mask[src]
    return out


def obstacle_masks(clear, support, z_ref, params=None):
    """(O_low, O_high) before dilation; cells without a reference count as obstacles."""
    p = params_from(params)
    known = np.isfinite(z_ref)
    o_low = ~known | ~support | (clear <= p["h_crouch_m"])
    o_high = ~known | ~support | (clear <= p["h_stand_m"])
    return o_low, o_high


def classify(clear, support, z_ref, cell_m, params=None, radius_m=None):
    """The class map int8 (nx, ny): 0 none (no reference), 1 stand, 2 crouch, 3 blocked."""
    p = params_from(params)
    r = p["r_robot_m"] if radius_m is None else float(radius_m)
    o_low, o_high = obstacle_masks(clear, support, z_ref, p)
    dl = dilate(o_low, cell_m, r)
    dh = dilate(o_high, cell_m, r)
    cls = np.full(o_low.shape, STAND, np.int8)
    cls[dh] = CROUCH
    cls[dl] = BLOCKED
    cls[~np.isfinite(z_ref)] = NONE
    return cls


def boundary_cells(cls):
    """Cells with a class (>= 1) that have a 4-neighbour with another class (>= 1): both sides of every boundary."""
    cls = np.asarray(cls)
    b = np.zeros(cls.shape, bool)
    d0 = (cls[1:, :] != cls[:-1, :]) & (cls[1:, :] > 0) & (cls[:-1, :] > 0)
    b[1:, :] |= d0
    b[:-1, :] |= d0
    d1 = (cls[:, 1:] != cls[:, :-1]) & (cls[:, 1:] > 0) & (cls[:, :-1] > 0)
    b[:, 1:] |= d1
    b[:, :-1] |= d1
    return b


def near_boundary(cls, cell_m, dist_m):
    """Cells whose centre lies within dist_m of the centre of a boundary cell (boundary_cells)."""
    b = boundary_cells(cls)
    out = b.copy()
    if not np.any(b):
        return out
    nx, ny = b.shape
    n = int(math.floor(float(dist_m) / float(cell_m) + 1e-9))
    lim = (float(dist_m) / float(cell_m)) ** 2 + 1e-9
    for di in range(-n, n + 1):
        for dj in range(-n, n + 1):
            if (di == 0 and dj == 0) or di * di + dj * dj > lim or abs(di) >= nx or abs(dj) >= ny:
                continue
            dst = (slice(max(di, 0), nx + min(di, 0)), slice(max(dj, 0), ny + min(dj, 0)))
            src = (slice(max(-di, 0), nx + min(-di, 0)), slice(max(-dj, 0), ny + min(-dj, 0)))
            out[dst] |= b[src]
    return out


# ---------------------------------------------------------------------------------------------------------------
# a whole twin

def tops_at_centres(shapes, grid):
    """Per shape, its highest point over each cell CENTRE ((nx, ny), NaN where the vertical line misses it): a list
    in shape order (for the reference level)."""
    X, Y = grid.centres()
    out = []
    for sh in shapes:
        z = np.full(X.shape, np.nan)
        blo, bhi = sh.bbox()
        sel = (X >= blo[0]) & (X <= bhi[0]) & (Y >= blo[1]) & (Y <= bhi[1])
        if np.any(sel):
            if sh.kind == "convex":
                _z0, z1 = sh.data.z_interval(X[sel], Y[sel])
                z[sel] = z1
            else:
                c, n = sh.data["c"], sh.data["n"] / np.linalg.norm(sh.data["n"])
                if n[2] > 1e-12:
                    z[sel] = c[2] - (n[0] * (X[sel] - c[0]) + n[1] * (Y[sel] - c[1])) / n[2]
        out.append(z)
    return out


def auto_reference(static_shapes, grid, max_rise_m=0.5):
    """A reference level from a twin's own static geometry, for checking a twin without the truth: per cell the LOWEST
    top of a static shape over the cell centre (your floor where it runs under everything), NaN where there is none or
    where that top is more than max_rise_m above the median of these tops (the top of a wall or a dock that has no
    floor under it is no walking level). The grader uses the TRUE floor instead (a dock or a kerb on it is an
    obstacle)."""
    tops = tops_at_centres(static_shapes, grid)
    if not tops:
        return np.full(grid.shape, np.nan)
    low = np.full(grid.shape, np.inf)
    for z in tops:
        low = np.where(np.isfinite(z), np.minimum(low, z), low)
    fin = np.isfinite(low)
    if not np.any(fin):
        return np.full(grid.shape, np.nan)
    level = float(np.median(low[fin]))
    return np.where(fin & (low <= level + float(max_rise_m)), low, np.nan)


class SliceMap:
    """The map of one twin: classes (nx, ny) int8, the grid, the clearance and support at the nominal reference,
    the parameters, a few counts."""

    def __init__(self, grid, classes, clear, support, z_ref, params, info=None):
        self.grid = grid
        self.classes = classes
        self.clear = clear
        self.support = support
        self.z_ref = z_ref
        self.params = params
        self.info = info or {}

    def counts(self):
        return {CLASS_NAMES[k]: int(np.sum(self.classes == k)) for k in range(4)}

    def summary(self):
        a = self.grid.cell_m ** 2
        return {"format": FORMAT, "grid": self.grid.as_dict(), "params": self.params, "cells": self.counts(),
                "area_m2": {k: v * a for k, v in self.counts().items()}, **self.info}


def map_shapes(shapes, grid, z_ref, params=None):
    p = params_from(params)
    ext = column_extents(shapes, grid, z_ref, p, deltas=(0.0,))
    cls = classify(ext["clear"][0], ext["support"][0], z_ref, grid.cell_m, p)
    return SliceMap(grid, cls, ext["clear"][0], ext["support"][0], z_ref, p,
                    {"n_shapes": len(shapes), "n_shapes_used": ext["n_shapes_used"]})


def map_twin(model, data, geoms=None, bounds=None, params=None, reference="auto", static_geoms=None,
             max_rise_m=0.5, margin_m=0.5):
    """The robot-terms map of a compiled twin at data's pose (call mj_forward first). geoms: the collision geoms that
    count (default all); bounds: (x0, y0, x1, y1) of the map (default: the extent of the non-plane collision
    geometry, grown by margin_m); reference: "auto" (auto_reference on static_geoms, default the world-welded collision
    geoms), a number (one level everywhere) or an (nx, ny) array for the grid of these bounds."""
    p = params_from(params)
    shapes = twin_shapes(model, data, geoms)
    if bounds is None:
        boxes = [s.bbox() for s in shapes if s.kind != "plane"]
        if not boxes:
            raise ValueError("the twin has no bounded collision geometry: give the map's bounds")
        lo = np.min([b[0][:2] for b in boxes], axis=0)
        hi = np.max([b[1][:2] for b in boxes], axis=0)
        lo, hi = np.maximum(lo, -200.0), np.minimum(hi, 200.0)
    else:
        lo, hi = np.asarray(bounds[:2], float), np.asarray(bounds[2:], float)
    grid = Grid.covering(lo, hi, p["cell_m"], margin_m if bounds is None else 0.0)
    if isinstance(reference, str):
        if reference != "auto":
            raise ValueError("reference must be 'auto', a number or an array")
        if static_geoms is None:
            static_geoms = [g for g in (collision_geoms(model) if geoms is None else geoms)
                            if int(model.body_weldid[int(model.geom_bodyid[g])]) == 0]
        st = set(int(g) for g in static_geoms)
        z_ref = auto_reference([s for s in shapes if s.geom in st], grid, max_rise_m)
    elif np.ndim(reference) == 0:
        z_ref = np.full(grid.shape, float(reference))
    else:
        z_ref = np.asarray(reference, float).reshape(grid.shape)
    return map_shapes(shapes, grid, z_ref, p)


# ---------------------------------------------------------------------------------------------------------------
# picture

COLOURS = {NONE: (40, 40, 40), STAND: (178, 223, 138), CROUCH: (253, 174, 97), BLOCKED: (165, 0, 38)}


def render_png(path, classes, grid, title=None, dim=None, max_px=1600, notes=()):
    """Write the class map as a PNG with a legend: x to the right, y up; one cell = an integer number of pixels (at
    most max_px wide). dim (nx, ny) bool: cells drawn faded (e.g. not scored). Needs Pillow."""
    from PIL import Image, ImageDraw, ImageFont
    cls = np.asarray(classes)
    nx, ny = cls.shape
    rgb = np.zeros((nx, ny, 3), np.uint8)
    for k, col in COLOURS.items():
        rgb[cls == k] = col
    if dim is not None:
        d = np.asarray(dim, bool)
        rgb[d] = (0.35 * rgb[d] + 0.65 * 255).astype(np.uint8)
    img = np.transpose(rgb, (1, 0, 2))[::-1]                     # rows = y (top = +y), columns = x
    sc = max(1, min(8, int(max_px // max(nx, 1))))
    if sc > 1:
        img = np.repeat(np.repeat(img, sc, axis=0), sc, axis=1)
    H, W = img.shape[:2]
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # pragma: no cover - old Pillow
        font = ImageFont.load_default()
    legend_h = 70 + 16 * len(notes)
    canvas = Image.new("RGB", (max(W, 640), H + legend_h + (24 if title else 0)), (255, 255, 255))
    y0 = 24 if title else 0
    canvas.paste(Image.fromarray(img), (0, y0))
    dr = ImageDraw.Draw(canvas)
    if title:
        dr.text((6, 4), str(title)[:150], fill=(0, 0, 0), font=font)
    a = grid.cell_m ** 2
    x = 6
    yl = y0 + H + 8
    for k in (STAND, CROUCH, BLOCKED, NONE):
        dr.rectangle([x, yl, x + 16, yl + 14], fill=COLOURS[k], outline=(0, 0, 0))
        lab = "%s %.1f m2" % (CLASS_NAMES[k], float(np.sum(cls == k)) * a)
        dr.text((x + 22, yl), lab, fill=(0, 0, 0), font=font)
        x += 30 + int(dr.textlength(lab, font=font))
    # scale bar: 1 m
    bar = int(round(sc / grid.cell_m))
    yb = yl + 26
    dr.rectangle([6, yb, 6 + bar, yb + 5], fill=(0, 0, 0))
    dr.text((12 + bar, yb - 4), "1 m;  x right, y up;  lower-left corner (%.2f, %.2f) m;  cell %.3f m" % (
        grid.origin[0], grid.origin[1], grid.cell_m), fill=(0, 0, 0), font=font)
    for n_, line in enumerate(notes):
        dr.text((6, yb + 18 + 16 * n_), str(line)[:180], fill=(0, 0, 0), font=font)
    canvas.save(path, format="PNG")
    return path


# ---------------------------------------------------------------------------------------------------------------
# command line: the map of a submission folder

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="The robot-terms map (stand / crouch / blocked) of a site twin.")
    ap.add_argument("submission", nargs="?", default="/home/user/submission")
    ap.add_argument("--png", default=None, help="write the map as a PNG")
    ap.add_argument("--bounds", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=None)
    ap.add_argument("--floor-z", type=float, default=None,
                    help="one reference level everywhere (default: auto, from your static geometry)")
    ap.add_argument("--max-rise", type=float, default=0.5,
                    help="auto reference: a lowest static top more than this above your floor's median level is no "
                         "walking level (no reference there)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    import mujoco
    TG = _gate()
    res = TG.gate(a.submission)
    if res.model is None:
        print("the twin does not compile: %s" % "; ".join(res.report.get("reasons", [])[:3]))
        return 1
    m = TG.physics_model(res) if res.ok else res.model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    sm = map_twin(m, d, bounds=a.bounds, reference="auto" if a.floor_z is None else a.floor_z,
                  max_rise_m=a.max_rise)
    s = sm.summary()
    if a.json:
        print(json.dumps(s, indent=1))
    else:
        print("robot-terms map: %d x %d cells of %.3f m from (%.2f, %.2f)" % (
            sm.grid.nx, sm.grid.ny, sm.grid.cell_m, sm.grid.origin[0], sm.grid.origin[1]))
        for k, v in s["area_m2"].items():
            print("  %-8s %9.2f m2" % (k, v))
        print("  robot terms: crouch below %.2f m, stand below %.2f m, radius %.2f m, step-over %.2f m" % (
            sm.params["h_crouch_m"], sm.params["h_stand_m"], sm.params["r_robot_m"], sm.params["low_from_m"]))
    if a.png:
        render_png(a.png, sm.classes, sm.grid, title="robot-terms map of %s" % os.path.basename(
            os.path.abspath(a.submission)), notes=["reference level: %s; crouch below %.2f m, stand below %.2f m, "
                                                   "robot radius %.2f m, step-over %.2f m" % (
                "auto (the lowest top of your static geometry)" if a.floor_z is None else "%.3f m" % a.floor_z,
                sm.params["h_crouch_m"], sm.params["h_stand_m"], sm.params["r_robot_m"], sm.params["low_from_m"])])
        print("wrote %s" % a.png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
