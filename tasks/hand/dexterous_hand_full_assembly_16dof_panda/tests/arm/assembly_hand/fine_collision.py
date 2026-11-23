"""Collision pieces of the hand assembly kit, cut from the closed true solids (the authors' release STEP as the audit
tessellated it). Adapted from the tensegrity cell's fine_collision.py (CoACD, convex hulls, carving by the partner's
pieces, trimming of intruders, sweeps), with the servo's slabs, chunking of the large shells, rotation sweeps and the
lateral growth of a slot added.

Everything here is a derived copy of the true geometry (convex pieces, carved and trimmed); the cameras always see the
authors' triangles. All lengths in mm.

  coacd_hulls(V, F)                   CoACD (threshold 0.03, at most 12 hulls, seed 0) -> [hull vertices]
  hull_of(V)                          the convex hull of a point set -> (V, F)
  slab_hulls(M, axis_T, cuts)         a closed solid cut by planes normal to one axis of a frame; one hull per slab
  chunks(M, nx, ny)                   a closed solid cut by a grid of planes in x and y -> [manifold]
  carve_mesh(V, F, cutters)           a closed piece minus the union of convex cutters (manifold3d, one at a time)
  trim_intruders(hulls, cutters)      every convex piece that still reaches into a cutter is cut by the one cutter face
                                      plane that keeps the most of it (the kept half-space lies outside the cutter, so
                                      the result never intrudes); remnants under 1 mm^3 or thinner than 1 mm dropped
  sweep_translate(H, d, length)       the convex hull of a convex piece and its copy moved along d (its exact sweep)
  sweep_rotate(H, p, a, a0, a1)       hulls of consecutive copies turned about the axis (p, a) from a0 to a1 (rad)
  grow_lateral(H, r, axes)            the piece grown by an octagon of radius r in the plane of two axes
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
    R = A
    v = R.volume()
    for C in hit:
        R2 = R - C
        v2 = 0.0 if R2.is_empty() else R2.volume()
        assert v2 <= v + 1e-6, "a subtraction grew the solid"
        R, v = R2, v2
        if R.is_empty():
            break
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


def slab_hulls(M, T, cuts, axis=2):
    """the closed manifold M (mm) cut by the planes x_axis = c (c in cuts, in the frame T: 4x4, the frame's axes in
    the world) into slabs; the convex hull of each slab's vertices -> [V]"""
    import manifold3d as mf
    n = np.asarray(T[:3, axis], float)
    o = np.asarray(T[:3, 3], float)
    out = []
    edges = [-1e9] + sorted(float(c) for c in cuts) + [1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        P = M
        if lo > -1e8:
            P = P.trim_by_plane(tuple(n.tolist()), float(n @ o + lo))
        if hi < 1e8:
            P = P.trim_by_plane(tuple((-n).tolist()), float(-(n @ o + hi)))
        if P.is_empty() or P.volume() < 1e-3:
            continue
        V, _F = mesh_of(P)
        out.append(hull_of(V)[0])
    return out


def chunks(M, nx, ny):
    """the closed manifold M cut by a regular grid of nx x ny cells over its box in x and y -> [(manifold, cell)]"""
    bb = M.bounding_box()
    lo, hi = np.array(bb[:3]), np.array(bb[3:])
    xs = np.linspace(lo[0], hi[0], nx + 1)
    ys = np.linspace(lo[1], hi[1], ny + 1)
    out = []
    for i in range(nx):
        for j in range(ny):
            P = M
            if i > 0:
                P = P.trim_by_plane((1.0, 0.0, 0.0), float(xs[i]))
            if i < nx - 1:
                P = P.trim_by_plane((-1.0, 0.0, 0.0), float(-xs[i + 1]))
            if j > 0:
                P = P.trim_by_plane((0.0, 1.0, 0.0), float(ys[j]))
            if j < ny - 1:
                P = P.trim_by_plane((0.0, -1.0, 0.0), float(-ys[j + 1]))
            if P.is_empty() or P.volume() < 1e-3:
                continue
            out.append((P, (i, j)))
    return out


def sweep_translate(H, d, length):
    d = np.asarray(d, float)
    d = d / np.linalg.norm(d)
    return hull_of(np.concatenate([H, H + length * d]))[0]


def rot_about(p, a, ang):
    a = np.asarray(a, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
    return R, np.asarray(p, float) - R @ np.asarray(p, float)


def sweep_rotate(H, p, a, a0, a1, step=math.radians(3.0), margin=0.05):
    """the sweep of a convex piece turned about the axis through p along a from a0 to a1 (rad): hulls of consecutive
    copies, grown by `margin` (covers the arc between two copies: r * step^2 / 8 = 0.05 mm at r = 140 mm)"""
    n = max(1, int(math.ceil(abs(a1 - a0) / step)))
    angs = np.linspace(a0, a1, n + 1)
    cube = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * margin
    out = []
    prev = None
    for t in angs:
        R, c = rot_about(p, a, t)
        Q = H @ R.T + c
        if prev is not None:
            P = np.concatenate([prev, Q])
            out.append(hull_of((P[:, None, :] + cube[None, :, :]).reshape(-1, 3))[0])
        prev = Q
    return out


def grow_lateral(H, r, axes=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), sides=8):
    """the convex piece grown by a regular polygon of circumradius r / cos(pi / sides) (so it contains the disc of
    radius r) in the plane of the two axes"""
    u, v = np.asarray(axes[0], float), np.asarray(axes[1], float)
    rc = r / math.cos(math.pi / sides)
    ang = (np.arange(sides) + 0.5) * 2 * math.pi / sides
    off = np.array([rc * (math.cos(t) * u + math.sin(t) * v) for t in ang])
    return hull_of((H[:, None, :] + off[None, :, :]).reshape(-1, 3))[0]


def hull_manifold(V):
    return manifold_of_points(V)
