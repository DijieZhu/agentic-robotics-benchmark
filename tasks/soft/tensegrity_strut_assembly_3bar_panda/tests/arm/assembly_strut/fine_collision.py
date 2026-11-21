"""Collision pieces of the strut assembly kit, derived from the authors' triangles.

Everything here is a derived, degraded copy of the true geometry (convex pieces, carved slots); the cameras always see
the authors' triangles. All lengths in mm, in the module frame of the piece.

  coacd_hulls(V, F)                   CoACD (threshold 0.03, at most 12 hulls, seed 0) -> [hull vertices]
  hull_of(V)                          the convex hull of a point set -> (V, F)
  bore_prism(r, z0, z1)               the slot cutter: a 32-sided prism circumscribing radius r about the module's
                                      z axis (the bore axis of the motor module and the end cap), z0 <= z <= z1
  carve_mesh(V, F, cutters)           a closed piece's true mesh minus the union of convex cutters (manifold3d)
  trim_intruders(hulls, cutters)      every convex piece that still reaches into a cutter is cut by the one cutter face
                                      plane that keeps the most of it (the kept half-space lies outside the cutter, so
                                      the result never intrudes); remnants under 1 mm^3 or thinner than 1 mm dropped
  sweep_grow(H, d, length, margin)    a convex piece swept along d over length and grown by a cube of half-size margin
                                      (convex: the hull of the shifted vertex sets)
  min_width(H)                        the smallest width of a convex piece over its face normals
  true_intersection_volume(A, B)      the intersection volume of two closed meshes (manifold3d)

The grader ships the pieces as files (the hidden kit's meshes/collision/); nothing here runs at verify time.
"""
import math

import numpy as np

COACD = {"threshold": 0.03, "max_convex_hull": 12, "seed": 0}
PRISM_SIDES = 32


def _mf_from(V, F):
    import manifold3d as mf
    return mf.Manifold(mf.Mesh(vert_properties=np.ascontiguousarray(V, np.float32), tri_verts=np.ascontiguousarray(F, np.uint32)))


def manifold_of_points(V):
    import manifold3d as mf
    return mf.Manifold.hull_points(np.ascontiguousarray(V, np.float64))


def mesh_of(man):
    mm = man.to_mesh64() if hasattr(man, "to_mesh64") else man.to_mesh()
    return np.asarray(mm.vert_properties, float)[:, :3], np.asarray(mm.tri_verts, np.int64)


def hull_of(V):
    """(V, F) of the convex hull of a point set (scipy qhull), outward faces"""
    from scipy.spatial import ConvexHull
    P = np.asarray(V, float)
    h = ConvexHull(P)
    used = np.unique(h.simplices)
    remap = -np.ones(len(P), np.int64)
    remap[used] = np.arange(len(used))
    Vh = P[used]
    F = remap[h.simplices]
    c = Vh.mean(0)
    for i, f in enumerate(F):
        n = np.cross(Vh[f[1]] - Vh[f[0]], Vh[f[2]] - Vh[f[0]])
        if n @ (Vh[f[0]] - c) < 0:
            F[i] = f[[0, 2, 1]]
    return Vh, F


def hull_volume(V):
    from scipy.spatial import ConvexHull
    try:
        return float(ConvexHull(np.asarray(V, float)).volume)
    except Exception:
        return 0.0


def min_width(V):
    """smallest width over the hull's face normals (mm)"""
    from scipy.spatial import ConvexHull
    P = np.asarray(V, float)
    try:
        h = ConvexHull(P)
    except Exception:
        return 0.0
    N = h.equations[:, :3]
    pr = P @ N.T
    return float((pr.max(0) - pr.min(0)).min())


def coacd_hulls(V, F, threshold=None, max_hulls=None, seed=None, preprocess="auto"):
    import coacd
    coacd.set_log_level("error")
    parts = coacd.run_coacd(coacd.Mesh(np.asarray(V, float), np.asarray(F, int)),
                            threshold=COACD["threshold"] if threshold is None else threshold,
                            max_convex_hull=COACD["max_convex_hull"] if max_hulls is None else max_hulls,
                            preprocess_mode=preprocess, seed=COACD["seed"] if seed is None else seed)
    out = []
    for v, _f in parts:
        v = np.asarray(v, float)
        if len(v) >= 4 and hull_volume(v) > 0:
            out.append(hull_of(v)[0])
    return out


def bore_prism(r, z0, z1, sides=PRISM_SIDES):
    """vertices and half-spaces (a, b: a.x <= b inside) of the slot cutter about the z axis"""
    rc = r / math.cos(math.pi / sides)
    ang = (np.arange(sides) + 0.5) * 2 * math.pi / sides
    ring = np.stack([rc * np.cos(ang), rc * np.sin(ang)], 1)
    V = np.concatenate([np.c_[ring, np.full(sides, z0)], np.c_[ring, np.full(sides, z1)]])
    # side faces: the face normal through the middle of each edge (edges between ang[k] and ang[k+1]); the faces lie
    # at distance r from the axis (the polygon circumscribes the circle of radius r)
    pl = []
    for k in range(sides):
        mid = (ang[k] + ang[(k + 1) % sides] + (2 * math.pi if k == sides - 1 else 0.0)) / 2.0
        pl.append((np.array([math.cos(mid), math.sin(mid), 0.0]), float(r)))
    pl.append((np.array([0.0, 0.0, 1.0]), float(z1)))
    pl.append((np.array([0.0, 0.0, -1.0]), float(-z0)))
    return V, pl


def planes_of_hull(V):
    from scipy.spatial import ConvexHull
    h = ConvexHull(np.asarray(V, float))
    return [(e[:3].copy(), float(-e[3])) for e in h.equations]


def _union(mans):
    import manifold3d as mf
    if len(mans) == 1:
        return mans[0]
    return mf.Manifold.batch_boolean(mans, mf.OpType.Add)


def intersection_volume(A, B):
    """volume of the intersection of two manifolds"""
    return float((A ^ B).volume())


def carve_mesh(V, F, cutters):
    """the closed mesh (V, F) minus the union of the convex cutters [(vertices, planes)] -> (V, F) or None when the
    piece does not meet any cutter"""
    A = _mf_from(V, F)
    lo, hi = np.asarray(V).min(0), np.asarray(V).max(0)
    hit = []
    for Vc, _pl in cutters:
        if np.all(Vc.min(0) <= hi) and np.all(Vc.max(0) >= lo):
            C = manifold_of_points(Vc)
            if intersection_volume(A, C) > 1e-6:
                hit.append(C)
    if not hit:
        return None
    R = A - _union(hit)
    if R.is_empty():
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return mesh_of(R)


def trim_intruders(hulls, cutters, min_volume=1.0, min_thickness=1.0, tol_mm3=1e-4, max_rounds=8):
    """[(vertices, trimmed?)]: each convex piece that reaches into a cutter by more than tol_mm3 is replaced by its
    part on the outer side of the cutter's face plane that keeps the most volume (repeated over cutters, at most
    max_rounds times); trimmed pieces under min_volume or thinner than min_thickness are dropped. Returns the pieces
    and the number dropped."""
    cm = [(manifold_of_points(Vc), Vc.min(0), Vc.max(0), pl) for Vc, pl in cutters]
    out, dropped = [], 0
    for H in hulls:
        cur, trimmed = manifold_of_points(H), False
        for _r in range(max_rounds):
            Vh, _ = mesh_of(cur)
            lo, hi = Vh.min(0), Vh.max(0)
            bad = None
            for C, clo, chi, pl in cm:
                if np.all(clo <= hi) and np.all(chi >= lo) and intersection_volume(cur, C) > tol_mm3:
                    bad = pl
                    break
            if bad is None:
                break
            best, bv = None, -1.0
            for a, b in bad:
                a = np.asarray(a, float)
                k = cur.trim_by_plane(tuple(a.tolist()), float(b))
                v = 0.0 if k.is_empty() else k.volume()
                if v > bv:
                    best, bv = k, v
            cur, trimmed = best, True
            if cur is None or cur.is_empty():
                break
        if cur is None or cur.is_empty():
            dropped += 1
            continue
        Vh = np.unique(np.round(mesh_of(cur)[0], 9), axis=0)
        if trimmed and (len(Vh) < 4 or hull_volume(Vh) < min_volume or min_width(Vh) < min_thickness):
            dropped += 1
            continue
        out.append((Vh, trimmed))
    return out, dropped


def sweep_grow(H, d, length, margin):
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    P = np.concatenate([H, H + length * d])
    cube = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * margin
    Q = (P[:, None, :] + cube[None, :, :]).reshape(-1, 3)
    return hull_of(Q)[0]


def true_intersection_volume(VA, FA, VB, FB):
    """intersection volume (mm^3) of two closed triangle meshes; None when either is not a valid manifold"""
    try:
        A, B = _mf_from(VA, FA), _mf_from(VB, FB)
    except Exception:
        return None
    if A.is_empty() or B.is_empty():
        return None
    return intersection_volume(A, B)


def is_closed_manifold(V, F):
    try:
        m = _mf_from(V, F)
    except Exception:
        return False
    return (not m.is_empty()) and m.volume() > 0
