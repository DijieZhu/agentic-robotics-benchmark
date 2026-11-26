"""cprim: small drawn parts as meshes (V, N, UV, F): boxes with broken edges, turned cylinders, sleeves, balls
(the cable task's harness_parts.py, the builders this task's scene uses). cscene.py places them."""
import math

import numpy as np

import cgeom as G


def box(lo, hi, b=0.0006):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return G.transform(G.bevel_box((hi - lo) / 2, b), np.eye(3), (lo + hi) / 2)


def sleeve(p0, p1, r_out, r_in, nseg=64, bevel=0.0):
    """a turned sleeve from p0 to p1: the outer wall (rims broken by `bevel`), the bore's wall and the two end faces"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    L = float(np.linalg.norm(ax))
    z = ax / L
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    b = min(bevel, 0.45 * (r_out - r_in), 0.45 * L)
    rings = [(0.0, r_out - b, -0.7), (b, r_out, 0.0), (L - b, r_out, 0.0), (L, r_out - b, 0.7)] if b > 0 else [(0.0, r_out, 0.0), (L, r_out, 0.0)]
    m = G.Mesh()
    for i in range(nseg):
        a0, a1 = 2 * math.pi * i / nseg, 2 * math.pi * (i + 1) / nseg
        d0, d1 = math.cos(a0) * x + math.sin(a0) * y, math.cos(a1) * x + math.sin(a1) * y
        for (h0, r0, t0), (h1, r1, t1) in zip(rings[:-1], rings[1:]):
            m.quad([p0 + z * h0 + r0 * d0, p0 + z * h0 + r0 * d1, p0 + z * h1 + r1 * d1, p0 + z * h1 + r1 * d0],
                   np.asarray([d0 + t0 * z, d1 + t0 * z, d1 + t1 * z, d0 + t1 * z]),
                   [(r_out * a0, h0), (r_out * a1, h0), (r_out * a1, h1), (r_out * a0, h1)])
        m.quad([p0 + r_in * d1, p0 + r_in * d0, p0 + z * L + r_in * d0, p0 + z * L + r_in * d1], np.asarray([-d1, -d0, -d0, -d1]),
               [(r_in * a1, 0.0), (r_in * a0, 0.0), (r_in * a0, L), (r_in * a1, L)])
        ro = r_out - b
        m.quad([p0 + z * L + r_in * d0, p0 + z * L + ro * d0, p0 + z * L + ro * d1, p0 + z * L + r_in * d1], z,
               [(r_in * math.cos(a0), r_in * math.sin(a0)), (ro * math.cos(a0), ro * math.sin(a0)),
                (ro * math.cos(a1), ro * math.sin(a1)), (r_in * math.cos(a1), r_in * math.sin(a1))])
        m.quad([p0 + r_in * d1, p0 + ro * d1, p0 + ro * d0, p0 + r_in * d0], -z,
               [(r_in * math.cos(a1), r_in * math.sin(a1)), (ro * math.cos(a1), ro * math.sin(a1)),
                (ro * math.cos(a0), ro * math.sin(a0)), (r_in * math.cos(a0), r_in * math.sin(a0))])
    return m.arrays()


def smooth_cyl(p0, p1, r, nseg=64, bevel=0.0, caps=True):
    """a cylinder with smooth side normals (G.cylinder's sides are faceted), its rims broken by `bevel`"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ax = p1 - p0
    L = float(np.linalg.norm(ax))
    z = ax / L
    x = np.cross(z, [0, 0, 1.0])
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    b = min(bevel, 0.45 * r, 0.45 * L)
    m = G.Mesh()
    rings = [(0.0, r - b, -0.7), (b, r, 0.0), (L - b, r, 0.0), (L, r - b, 0.7)] if b > 0 else [(0.0, r, 0.0), (L, r, 0.0)]
    for i in range(nseg):
        a0, a1 = 2 * math.pi * i / nseg, 2 * math.pi * (i + 1) / nseg
        d0, d1 = math.cos(a0) * x + math.sin(a0) * y, math.cos(a1) * x + math.sin(a1) * y
        for (h0, r0, t0), (h1, r1, t1) in zip(rings[:-1], rings[1:]):
            m.quad([p0 + z * h0 + r0 * d0, p0 + z * h0 + r0 * d1, p0 + z * h1 + r1 * d1, p0 + z * h1 + r1 * d0],
                   np.asarray([d0 + t0 * z, d1 + t0 * z, d1 + t1 * z, d0 + t1 * z]),
                   [(r * a0, h0), (r * a1, h0), (r * a1, h1), (r * a0, h1)])
        if caps:
            rc = r - b
            m.tri([p0 + z * L, p0 + z * L + rc * d0, p0 + z * L + rc * d1], z, [(0, 0), (rc * math.cos(a0), rc * math.sin(a0)), (rc * math.cos(a1), rc * math.sin(a1))])
            m.tri([p0, p0 + rc * d1, p0 + rc * d0], -z, [(0, 0), (rc * math.cos(a1), rc * math.sin(a1)), (rc * math.cos(a0), rc * math.sin(a0))])
    return m.arrays()


def ball(c, r, nlat=8, nlon=16):
    m = G.Mesh()
    c = np.asarray(c, float)

    def pt(i, j):
        th, ph = math.pi * i / nlat, 2 * math.pi * j / nlon
        n = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
        return c + r * n, n
    for i in range(nlat):
        for j in range(nlon):
            (a, na), (b, nb), (cc, nc), (dd, nd) = pt(i, j), pt(i + 1, j), pt(i + 1, j + 1), pt(i, j + 1)
            m.quad([a, b, cc, dd], np.asarray([na, nb, nc, nd]), [(0, 0)] * 4)
    return m.arrays()
