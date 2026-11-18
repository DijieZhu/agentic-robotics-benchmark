"""Stable resting poses of a rigid part on a flat bench: the facets of its collision hull's convex hull that its centre
of mass projects into (quasi-static), without trimesh's topple graph (which needs networkx).

`stable_rests(V, com)` -> [{"R": rotation part frame -> bench frame (the facet's outward normal turned to -z),
"margin_m": distance of the centre of mass's projection to the facet's edges, "com_height_m", "solid_angle": the facet's
solid angle seen from the centre of mass / 4 pi (a proxy of how often the part lands on it), "normal"}], most likely
first. Rule from the assembly skill (pitfall 11): a bounding-box face is not a resting pose; use the COLLISION geometry's
hull with the body's centre of mass, and leave knife-edge rests (margin < 0.5 mm) out of a bench plan.
"""
import math

import numpy as np


def _facets(P):
    """(normal, offset, vertex indices of the facet polygon in order) of the convex hull of P, coplanar faces merged"""
    from scipy.spatial import ConvexHull
    H = ConvexHull(P)
    eqs = H.equations                                     # n . x + d <= 0 inside
    groups = []
    used = np.zeros(len(eqs), bool)
    for i in range(len(eqs)):
        if used[i]:
            continue
        same = np.where(~used & (np.abs(eqs[:, :3] @ eqs[i, :3] - 1.0) < 1e-6) & (np.abs(eqs[:, 3] - eqs[i, 3]) < 1e-7))[0]
        used[same] = True
        verts = np.unique(H.simplices[same].ravel())
        groups.append((eqs[i, :3].copy(), float(eqs[i, 3]), verts))
    return groups


def _rot_to_down(n):
    """a rotation taking the unit vector n to (0, 0, -1)"""
    n = n / np.linalg.norm(n)
    t = np.array([0.0, 0.0, -1.0])
    v = np.cross(n, t)
    c = float(n @ t)
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    # Rodrigues with (1 - c) / |v|^2 (= 1 / (1 + c) in exact arithmetic): stays a rotation for n near +z, where 1 + c
    # cancels to 0 (a facet whose outward normal points up)
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / float(v @ v))


def _poly_order(Q):
    c = Q.mean(0)
    a = np.arctan2(Q[:, 1] - c[1], Q[:, 0] - c[0])
    return Q[np.argsort(a)]


def _solid_angle(tri, o):
    a, b, c = tri[0] - o, tri[1] - o, tri[2] - o
    la, lb, lc = np.linalg.norm(a), np.linalg.norm(b), np.linalg.norm(c)
    num = abs(float(a @ np.cross(b, c)))
    den = la * lb * lc + float(a @ b) * lc + float(a @ c) * lb + float(b @ c) * la
    return 2.0 * math.atan2(num, den)


def stable_rests(V, com, min_margin=0.0):
    P = np.asarray(V, float)
    com = np.asarray(com, float)
    out = []
    for n, dd, idx in _facets(P):
        R = _rot_to_down(n)
        Q = (P[idx] - com) @ R.T                        # bench frame, the centre of mass at the origin, facet below at z = -h
        h = -float(Q[:, 2].mean())
        poly = _poly_order(Q[:, :2])
        # the centre of mass's projection (0, 0) inside the facet polygon, and its distance to the edges
        inside, margin = True, 1e9
        for i in range(len(poly)):
            p0, p1 = poly[i], poly[(i + 1) % len(poly)]
            e = p1 - p0
            cr = e[0] * (0 - p0[1]) - e[1] * (0 - p0[0])
            L = np.linalg.norm(e)
            if L < 1e-12:
                continue
            dist = cr / L
            if dist < 0:
                inside = False
            margin = min(margin, dist)
        if not inside or margin < min_margin:
            continue
        # the facet's solid angle from the centre of mass (fan of triangles of the ordered polygon, in 3-D)
        F3 = np.column_stack([poly, np.full(len(poly), -h)])
        sa = sum(_solid_angle(np.array([F3[0], F3[i], F3[i + 1]]), np.zeros(3)) for i in range(1, len(F3) - 1))
        out.append({"R": R, "normal": n, "margin_m": float(margin), "com_height_m": h, "solid_angle": sa / (4 * math.pi)})
    out.sort(key=lambda r: -r["solid_angle"])
    return out
