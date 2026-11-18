"""Link coverage: line of sight (2-D, through radio-opaque walls / closed doors / shields) to the base station in the
airlock within its range, plus the area around the one working corridor repeater."""

from __future__ import annotations

import math

import numpy as np

from .trav import Grid, _hull


def opaque_raster(W, grid: Grid):
    m = np.zeros(grid.shape, np.uint8)
    for p in W.P.prims:
        if not p.radio or p.group not in (0, 3):
            continue
        c = p.corners()
        if c[:, 2].min() > 1.2:
            continue
        grid.fill_poly(m, _hull(c[:, :2]), 1)
    return m.astype(bool)


def coverage(grid: Grid, opaque, antennas, inside):
    """antennas: [(x, y, range)]. Returns a bool grid."""
    X, Y = grid.centers()
    cov = np.zeros(grid.shape, bool)
    step = grid.res * 0.7
    for ax, ay, rng in antennas:
        sel = ((X - ax) ** 2 + (Y - ay) ** 2 <= rng * rng) & inside
        iy, ix = np.nonzero(sel)
        if not len(iy):
            continue
        px, py = X[iy, ix], Y[iy, ix]
        d = np.hypot(px - ax, py - ay)
        K = int(math.ceil(rng / step)) + 1
        ok = np.ones(len(iy), bool)
        for k in range(1, K):
            t = np.minimum(1.0, (k * step) / np.maximum(d, 1e-9))
            active = (k * step) < d - grid.res * 0.6
            if not active.any():
                break
            sx = ax + t * (px - ax)
            sy = ay + t * (py - ay)
            jy = np.clip(((sy - grid.y0) / grid.res).astype(int), 0, grid.ny - 1)
            jx = np.clip(((sx - grid.x0) / grid.res).astype(int), 0, grid.nx - 1)
            hit = opaque[jy, jx] & active
            ok &= ~hit
        cov[iy[ok], ix[ok]] = True
    return cov
