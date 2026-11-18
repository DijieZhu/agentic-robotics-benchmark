"""Unit solids and their outer-surface samples (replacing the arm twin's voxel solids).

A unit's solid is the union of its convex collision pieces (twin_gate.Piece: a mesh as its convex hull, a primitive as
itself), here in the SITE frame: the agent's pieces are mapped by the alignment's final similarity first, so every
length below is in site metres.

    S = unit_solid(twin, unit, simmap, cfg)     # surfaces.Solid
    P, N = S.samples()                          # outer-surface samples (n, 3) and outward normals (n, 3)
    S.footprint_centroid(cfg)                   # (2,) centroid of the occupied xy columns (zones)
    P, N, h = sample_union(pieces, spacing, cap, seed)

Sampling (the matching's F-score and C2 use it; the truth builder stores the true units' samples made by this very
function, so the oracle's samples are identical to the stored ones): every piece gets round(n x area_k / area)
points, n = total area / spacing^2 capped at `cap` (the spacing then grows to sqrt(area / cap)), drawn iid and
area-weighted by twin_gate.Piece.surface_samples from one numpy Generator seeded with unit_seed(name); a sample is kept
when the point 0.1 mm outside it along its normal lies in no other piece of the same unit (so faces where two pieces
touch or overlap are not outer surface).
"""
import math
import zlib

import numpy as np

from . import load

OUTSIDE_EPS_M = 1e-4


def _TG():
    return load.payload()["twin_gate"]


def unit_seed(name, cfg):
    """The sampling seed of a unit: crc32 of its name xor the configured seed (never Python's hash)."""
    return (zlib.crc32(str(name).encode("utf-8")) ^ int(cfg["matching"]["seed"])) & 0x7FFFFFFF


def sample_union(pieces, spacing, cap, seed):
    """(P (n, 3), N (n, 3), spacing used) of the outer surface of the union of the pieces."""
    pieces = list(pieces)
    if not pieces:
        return np.zeros((0, 3)), np.zeros((0, 3)), float(spacing)
    areas = np.array([float(p.area()) for p in pieces])
    tot = float(areas.sum())
    if tot <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), float(spacing)
    n_total = tot / (spacing * spacing)
    if n_total > cap:
        spacing = math.sqrt(tot / cap)
        n_total = float(cap)
    rng = np.random.default_rng(int(seed))
    Ps, Ns = [], []
    for k, p in enumerate(pieces):
        nk = max(1, int(round(n_total * areas[k] / tot)))
        P, N = p.surface_samples(nk, rng)
        if len(pieces) > 1:
            Q = P + OUTSIDE_EPS_M * N
            keep = np.ones(len(P), bool)
            for q, other in enumerate(pieces):
                if q != k:
                    keep &= ~other.inside(Q, tol=0.0)
            P, N = P[keep], N[keep]
        Ps.append(P)
        Ns.append(N)
    return np.vstack(Ps), np.vstack(Ns), float(spacing)


class Solid:
    """A unit's solid in the site frame: pieces, samples (cached), bbox, area, footprint."""

    def __init__(self, name, label, label_ok, pieces, cfg):
        self.name = name
        self.label = label
        self.label_ok = bool(label_ok)
        self.pieces = list(pieces)
        self.cfg = cfg
        self._samples = None
        self._fp = None
        self.spacing = float(cfg["matching"]["spacing_m"])

    def samples(self):
        if self._samples is None:
            mc = self.cfg["matching"]
            P, N, h = sample_union(self.pieces, float(mc["spacing_m"]), int(mc["max_samples_per_unit"]),
                                   unit_seed(self.name, self.cfg))
            self.spacing = h
            self._samples = (P, N)
        return self._samples

    def bbox(self):
        lo = np.min([p.bbox()[0] for p in self.pieces], axis=0)
        hi = np.max([p.bbox()[1] for p in self.pieces], axis=0)
        return lo, hi

    def area(self):
        return float(sum(p.area() for p in self.pieces))

    def footprint(self):
        """(cell h, occupied column centres (k, 2)) of the solid's vertical projection on a grid of
        zones.footprint_cell_m (coarsened until the box has at most zones.max_footprint_cells columns)."""
        if self._fp is None:
            TG = _TG()
            zc = self.cfg["zones"]
            h = float(zc["footprint_cell_m"])
            while True:
                i0, i1, j0, j1, _work = TG.grid_extent(self.pieces, h)
                if (i1 - i0) * (j1 - j0) <= int(zc["max_footprint_cells"]):
                    break
                h *= 2.0
            cols = []
            nj = j1 - j0
            for ia, ib, Z0, _Z1 in TG.grid_union(self.pieces, h, i0, i1, j0, j1):
                occ = np.any(~np.isnan(Z0), axis=1)
                idx = np.nonzero(occ)[0]
                I = ia + idx // nj
                J = j0 + idx % nj
                cols.append(np.c_[(I + 0.5) * h, (J + 0.5) * h])
            C = np.vstack(cols) if cols else np.zeros((0, 2))
            if not len(C):                        # a solid thinner than one column: its box centre
                lo, hi = self.bbox()
                C = (0.5 * (lo + hi))[None, :2]
            self._fp = (h, C)
        return self._fp

    def footprint_centroid(self):
        return self.footprint()[1].mean(axis=0)


def map_pieces(pieces, simmap):
    """The pieces under x' = s R x + t (twin_gate.Piece.transformed computes s (M x + t'))."""
    if simmap is None or simmap.is_identity():
        return list(pieces)
    t = np.asarray(simmap.t, float) / float(simmap.s)
    return [p.transformed(simmap.R, t, simmap.s) for p in pieces]


def unit_solid(twin, unit, simmap, cfg):
    """The Solid of one unit of `twin` (a dict of twin.units) in the site frame; None without collision pieces."""
    pieces = twin.pieces(unit["geoms"])
    if not pieces:
        return None
    return Solid(unit["body"], unit.get("label"), unit.get("label_ok"), map_pieces(pieces, simmap), cfg)


def unit_solids(twin, simmap, cfg, which="ok"):
    """{body: Solid} of the twin's units (status 'ok' by default; which="all" includes excluded ones)."""
    out = {}
    for u in twin.units:
        if u["body_id"] < 0 or (which == "ok" and u["status"] != "ok"):
            continue
        S = unit_solid(twin, u, simmap, cfg)
        if S is not None:
            out[u["body"]] = S
    return out
