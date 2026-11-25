"""Harness cell: small drawn parts as meshes (V, N, UV, F) in the frame of the body they belong to: boxes with broken
edges, turned cylinders, tubes, balls, screw heads, and the Panda finger's metal and its soft pad cut from the hand's
own finger mesh. harness_scene.py places them.
"""
import math

import numpy as np

import harness_geom as G

PAD_Y_BACK, PAD_Y_FACE, PAD_Z, PAD_HX = 0.0080, -0.0005, (0.0362, 0.0539), 0.0088   # the gripper's tip block, finger frame
PAD_T = PAD_Y_BACK - PAD_Y_FACE


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


def tube(points, r, nseg=20):
    """a round tube along a polyline (smooth sides, a ball at every joint)"""
    ms = []
    P = [np.asarray(p, float) for p in points]
    for a, b in zip(P[:-1], P[1:]):
        ms.append(smooth_cyl(a, b, r, nseg=nseg, caps=True))
    for p in P[1:-1]:
        ms.append(ball(p, r))
    return G.merge(*ms)


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


def screw(c, n, r=0.0034, h=0.0009):
    """a countersunk screw head: a low disc at c, proud by h along n, with a slot"""
    c, n = np.asarray(c, float), np.asarray(n, float)
    head = smooth_cyl(c - 0.0002 * n, c + h * n, r, nseg=24, bevel=0.0004)
    return head


def _clip(poly, axis, val, keep_ge):
    out = []
    n = len(poly)
    for i in range(n):
        (p, pn), (q, qn) = poly[i], poly[(i + 1) % n]
        ip = (p[axis] >= val) == keep_ge
        iq = (q[axis] >= val) == keep_ge
        if ip:
            out.append((p, pn))
        if ip != iq:
            t = (val - p[axis]) / (q[axis] - p[axis])
            x = p + t * (q - p)
            x[axis] = val
            out.append((x, pn + t * (qn - pn)))
    return out


def finger_metal(V, N, sign, z_cut=0.0345):
    """a gripper finger's visual mesh (a triangle soup with normals, finger frame) without its tip block (everything
    with z > z_cut and sign * y < PAD_Y_BACK cut away, the cut closed): the soft pad stands in the block's place"""
    V, N = V.copy(), N.copy()
    V[:, 1] *= sign
    N[:, 1] *= sign
    tris, cut = [], []
    for i in range(0, len(V), 3):
        poly = [(V[i + k], N[i + k]) for k in range(3)]
        low = _clip(poly, 2, z_cut, False)
        up = _clip(poly, 2, z_cut, True)
        if len(low) >= 3:
            tris.append(low)
        if len(up) >= 3:
            kept = _clip(up, 1, PAD_Y_BACK, True)
            if len(kept) >= 3:
                tris.append(kept)
                cut += [p[0] for p in kept if abs(p[0][1] - PAD_Y_BACK) < 1e-9]
    m = G.Mesh()
    for poly in tris:
        for k in range(1, len(poly) - 1):
            m.tri([poly[0][0], poly[k][0], poly[k + 1][0]], np.asarray([poly[0][1], poly[k][1], poly[k + 1][1]]), [(0, 0)] * 3)
    if cut:
        cp = np.asarray(cut)[:, [0, 2]]
        lo, hi = cp.min(0), cp.max(0)
        q = [np.array([lo[0], PAD_Y_BACK, lo[1]]), np.array([hi[0], PAD_Y_BACK, lo[1]]), np.array([hi[0], PAD_Y_BACK, hi[1]]),
             np.array([lo[0], PAD_Y_BACK, hi[1]])]
        m.quad(q, [0, -1.0, 0], [(0, 0)] * 4)
    Vm, Nm, UVm, Fm = m.arrays()
    Vm[:, 1] *= sign
    Nm[:, 1] *= sign
    return G.uv_box((Vm, Nm, UVm, Fm))


def finger_pad(sign):
    """the soft pad in the finger's frame: its back on the shank's face, its face where the model's pad meets a part"""
    hz = (PAD_Z[1] - PAD_Z[0]) / 2
    V, N, UV, F = G.slab(G.rounded_rect(PAD_HX, hz, 0.0026, 6), 0.0, PAD_T, 0.0012, 0.0, top=True, bottom=False)
    R = np.array([[sign, 0, 0], [0, 0, -sign], [0, 1.0, 0]])
    return V @ R.T + np.array([0, sign * PAD_Y_BACK, 0.5 * (PAD_Z[0] + PAD_Z[1])]), N @ R.T, UV, (F if sign > 0 else F)
