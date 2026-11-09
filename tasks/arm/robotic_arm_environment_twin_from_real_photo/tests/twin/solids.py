"""Per-item collision solids in a camera frame.

A solid is the union of an item's collision pieces: a mesh geom is the convex hull of its vertices (MuJoCo's own
collision hull, read from the compiled model), a primitive (box, sphere, capsule, cylinder, ellipsoid) is itself,
with analytic inside tests and column intervals. The piece geometry is the public twin_gate code. Each solid also
carries its own frame: the item body's axes in the same coordinates (used by the oversize cap of the matching).

    S = item_solids(twin, M, t)       # {body: Solid}, world -> frame by x' = M x + t
    S.scaled(s), S.volume_centroid(), S.keys(h), S.surface(n, seed), S.min_width(), S.extents(axes)

Grid keys: cell (i, j, k) of spacing h has its centre at ((i + 0.5) h, (j + 0.5) h, (k + 0.5) h) in the frame; a
cell belongs to the solid when its centre does (z ranges are exact per column of cells). The columns are processed
in bands of rows (twin_gate.grid_union), each piece only on its own columns, so memory stays bounded whatever the
item's footprint. Work caps: a solid whose grid would exceed max_columns columns, max_piece_columns piece-columns or
max_cells cells raises TooLarge (the matcher then treats the pair as not overlapping); the volume integral coarsens
its grid instead.
"""
import math

import numpy as np

from . import load

KEY_OFF = 1 << 20
KEY_SPAN = 1 << 21

# work bounds of the grader's grids (the gate has its own, smaller ones)
VOLUME_H_M = 0.0005                  # column spacing of the volume / centroid integral of a multi-piece solid
VOLUME_MAX_COLUMNS = 4_000_000
VOLUME_MAX_PIECE_COLUMNS = 4_000_000
KEYS_MAX_COLUMNS = 8_000_000
KEYS_MAX_PIECE_COLUMNS = 16_000_000


class TooLarge(ValueError):
    pass


def _TG():
    return load.payload()["twin_gate"]


class Solid:
    def __init__(self, name, label, pieces, label_ok=True, frame=None):
        self.name = name
        self.label = label
        self.label_ok = label_ok
        self.pieces = list(pieces)
        self.frame = np.eye(3) if frame is None else np.asarray(frame, float).reshape(3, 3)
        self._vc = None

    def transformed(self, M, t, s=1.0):
        return Solid(self.name, self.label, [p.transformed(M, t, s) for p in self.pieces], self.label_ok,
                     np.asarray(M, float) @ self.frame)

    def scaled(self, s):
        if s == 1.0:
            return self
        return Solid(self.name, self.label, [p.scaled(s) for p in self.pieces], self.label_ok, self.frame)

    def bbox(self):
        lo = np.min([p.bbox()[0] for p in self.pieces], axis=0)
        hi = np.max([p.bbox()[1] for p in self.pieces], axis=0)
        return lo, hi

    def support(self, U):
        return np.max([p.support(U) for p in self.pieces], axis=0)

    def extents(self, axes):
        """Extent of the solid along each column of axes (3, k) (unit vectors)."""
        U = np.asarray(axes, float).T
        return self.support(U) + self.support(-U)

    def volume_centroid(self, h=VOLUME_H_M):
        """(volume m^3, centroid (3,)): exact for one piece, else the union integral on columns of spacing h (coarsened
        until the grid is within the work bounds)."""
        if self._vc is None:
            if len(self.pieces) == 1:
                p = self.pieces[0]
                self._vc = (float(p.volume()), np.asarray(p.centroid(), float))
            else:
                v, c, _ = _TG().union_integrals(self.pieces, h, VOLUME_MAX_COLUMNS, VOLUME_MAX_PIECE_COLUMNS)
                self._vc = (float(v), np.asarray(c, float))
        return self._vc

    def min_width(self):
        """Smallest extent over the piece face normals, the box axes and 512 spread directions."""
        U = [_TG().fibonacci_dirs(512)]
        for p in self.pieces:
            if p.A is not None:
                U.append(p.A)
            U.append(p.R.T)
        U = np.vstack(U)
        ext = self.support(U) + self.support(-U)
        return float(np.min(ext))

    def inside(self, P, tol=0.0):
        P = np.atleast_2d(P)
        out = np.zeros(len(P), bool)
        for p in self.pieces:
            out |= p.inside(P, tol)
        return out

    def keys(self, h, max_cells=20_000_000, max_columns=KEYS_MAX_COLUMNS, max_piece_columns=KEYS_MAX_PIECE_COLUMNS):
        """Sorted int64 keys of the grid cells (spacing h) whose centres lie inside the solid."""
        TG = _TG()
        i0, i1, j0, j1, work = TG.grid_extent(self.pieces, h)
        ni, nj = i1 - i0, j1 - j0
        if ni <= 0 or nj <= 0 or ni * nj > max_columns:
            raise TooLarge("%d x %d columns" % (ni, nj))
        if work > max_piece_columns:
            raise TooLarge("%d piece-columns" % work)
        out, total = [], 0
        J = np.arange(j0, j1)
        for ia, ib, Z0, Z1 in TG.grid_union(self.pieces, h, i0, i1, j0, j1):
            with np.errstate(invalid="ignore"):
                k0 = np.ceil(Z0 / h - 0.5)
                k1 = np.floor(Z1 / h - 0.5)
            valid = ~np.isnan(Z0) & (k1 >= k0)
            col = np.nonzero(valid)[0]                # the column of each valid segment (row-major order)
            if not len(col):
                continue
            k0 = k0[valid].astype(np.int64)
            cnt = (k1[valid].astype(np.int64) - k0 + 1)
            n = int(cnt.sum())
            total += n
            if total > max_cells:
                raise TooLarge("more than %d cells" % max_cells)
            Ic = ia + col // nj
            Jc = J[col % nj]
            base = ((Ic.astype(np.int64) + KEY_OFF) * KEY_SPAN + (Jc.astype(np.int64) + KEY_OFF)) * KEY_SPAN
            start = base + KEY_OFF + k0
            first = np.repeat(np.cumsum(cnt) - cnt, cnt)
            out.append(np.repeat(start, cnt) + (np.arange(n, dtype=np.int64) - first))
        if not out:
            return np.zeros(0, np.int64)
        keys = np.concatenate(out)
        # bands come in row order, columns in key order and the segments of a column in z order, so the keys are
        # sorted; two segments can only share a cell whose centre sits exactly on their common end
        dk = np.diff(keys)
        if np.any(dk < 0):
            return np.unique(keys)
        if np.any(dk == 0):
            keys = keys[np.r_[True, dk != 0]]
        return keys

    def surface(self, n, seed):
        """About n points on the boundary of the union, with outward normals (deterministic for a given seed)."""
        rng = np.random.default_rng(int(seed))
        areas = np.array([p.area() for p in self.pieces], float)
        tot = float(areas.sum())
        Ps, Ns = [], []
        for k, p in enumerate(self.pieces):
            nk = max(1, int(round(n * areas[k] / tot))) if tot > 0 else max(1, n // len(self.pieces))
            P, N = p.surface_samples(nk, rng)
            if len(self.pieces) > 1:
                keep = np.ones(len(P), bool)
                for q, other in enumerate(self.pieces):
                    if q != k:
                        keep &= ~other.inside(P, tol=-1e-7)
                P, N = P[keep], N[keep]
            Ps.append(P)
            Ns.append(N)
        return np.vstack(Ps), np.vstack(Ns)


def item_solids(twin, M, t, s=1.0, which="ok"):
    """{body: Solid} of the twin's items (status 'ok' only by default), mapped by x' = s (M x + t). The solid's frame
    is the item body's rotation mapped by M."""
    TG = _TG()
    out = {}
    M = np.asarray(M, float)
    for it in twin.items:
        if which == "ok" and it["status"] != "ok":
            continue
        pieces = []
        for g in it["geoms"]:
            pc = TG.geom_piece(twin.model, twin.data, g)
            if pc is not None:
                pieces.append(pc.transformed(M, t, s))
        if pieces:
            frame = M @ np.asarray(twin.data.xmat[it["body_id"]], float).reshape(3, 3)
            out[it["body"]] = Solid(it["body"], it["label"], pieces, it["label_ok"], frame)
    return out
