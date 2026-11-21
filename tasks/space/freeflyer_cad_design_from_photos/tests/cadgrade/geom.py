"""B-rep reading (OCP), tessellation, seeded surface samples and the family's surface distance.

Tessellation (mesh_solid) follows the UAV grader: OCC's mesher on a fresh copy of the solid in its principal frame
(the mesher depends on the numbers of the geometry, so congruent solids tessellate alike wherever they sit), welded and
hole-filled, accepted when it is a closed manifold whose volume is within volume_tol_rel of the B-rep volume, else the
next pose or the retry settings, else the closest result (noted). The mesher runs single-threaded.

Samples (sample_triangles) are drawn with numpy's PCG64 from a seed: triangles by area (cumulative sums, searchsorted),
points by the reflection rule, so a sample set is a pure function of the triangles and the seed.

Distances (surface_distances, pooled_quantile): the nearest dense sample is found exactly, each search bounded by the
nearest point of a subset (cheaper, the same answer) and capped; the pooled quantile is exact whenever it depends only on
points within the cap, and otherwise is known to lie at or beyond the band's zero line (a score of 0), so a part far
from the original costs seconds, not the grade's time limit.
"""
import math
import os
import tempfile

import numpy as np

from .util import apply, axis_angle_R, inv, make_T


# ---------------------------------------------------------------------------------------------------- B-rep
def read_brep(path):
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    s = TopoDS_Shape()
    if not BRepTools.Read_s(s, str(path), BRep_Builder()):
        raise IOError("cannot read %s" % os.path.basename(str(path)))
    return s


def solids_of(shape):
    from OCP.TopAbs import TopAbs_SOLID
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    ex = TopExp_Explorer(shape, TopAbs_SOLID)
    out = []
    while ex.More():
        out.append(TopoDS.Solid_s(ex.Current()))
        ex.Next()
    return out


def brep_valid(shape):
    from OCP.BRepCheck import BRepCheck_Analyzer
    return bool(BRepCheck_Analyzer(shape).IsValid())


def valid_after_tolerance_update(shape):
    """A solid read back from BREP can fail BRepCheck on vertex tolerances that equal its curve deviation exactly;
    the tolerances are updated and the solid checked again (the update changes no geometry)."""
    from OCP.BRepLib import BRepLib
    try:
        BRepLib.UpdateTolerances_s(shape, False)
    except Exception:  # noqa: BLE001
        return False
    return brep_valid(shape)


def mass_props(shape):
    """(volume, centroid (3,), inertia tensor about the centroid (3, 3)) at density 1 (OCC GProp, exact on the B-rep)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, g)
    V = float(g.Mass())
    c = g.CentreOfMass()
    M = g.MatrixOfInertia()
    I = np.array([[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)], float)
    if V < 0:
        V, I = -V, -I
    return V, np.array([c.X(), c.Y(), c.Z()], float), I


def fresh_copy(shape):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepTools import BRepTools
    c = BRepBuilderAPI_Copy(shape, True, False).Shape()
    BRepTools.Clean_s(c)
    return c


def _moved_copy(shape, T):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    tr = gp_Trsf()
    tr.SetValues(*(float(v) for v in np.asarray(T, float)[:3, :].ravel()))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


# ---------------------------------------------------------------------------------------------------- meshes
def _stl_mesh(shape, defl, ang, tmpdir):
    import trimesh
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IMeshTools import IMeshTools_Parameters
    from OCP.StlAPI import StlAPI_Writer
    s = fresh_copy(shape)
    p = IMeshTools_Parameters()
    p.Deflection = float(defl)
    p.Angle = float(ang)
    p.Relative = False
    p.InParallel = False
    BRepMesh_IncrementalMesh(s, p)
    fd, path = tempfile.mkstemp(suffix=".stl", dir=tmpdir)
    os.close(fd)
    try:
        w = StlAPI_Writer()
        w.ASCIIMode = False
        if not w.Write(s, path):
            return None
        m = trimesh.load(path, file_type="stl", process=False, force="mesh")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if m is None or len(m.faces) == 0:
        return None
    return np.asarray(m.vertices, float), np.asarray(m.faces, np.int64)


def weld(V, F, tol):
    """Merge vertices closer than tol (KD-tree clusters), drop degenerate and repeated triangles."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    pairs = cKDTree(V).query_pairs(float(tol), output_type="ndarray")
    n = len(V)
    if len(pairs):
        g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        _nc, lab = connected_components(g, directed=False)
    else:
        lab = np.arange(n)
    _u, first, labc = np.unique(lab, return_index=True, return_inverse=True)
    labc = np.asarray(labc).reshape(-1)
    V2 = V[first]
    F2 = labc[F]
    keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[keep]
    srt = np.sort(F2, axis=1)
    _u2, idx = np.unique(srt, axis=0, return_index=True)
    return V2, F2[np.sort(idx)]


def fill_holes(V, F, max_loop):
    """Close boundary loops of at most max_loop edges with triangle fans (the OCC mesher leaves faces it cannot mesh and
    seam gaps on some B-reps). Adapted from the bimanual task's cad_tree.checks.fill_small_holes."""
    import trimesh
    mesh = trimesh.Trimesh(V, F, process=False)
    edges = mesh.edges_sorted
    bidx = trimesh.grouping.group_rows(edges, require_count=1)
    if len(bidx) == 0:
        return V, F, 0
    directed = mesh.edges[bidx]
    nxt = {}
    for a, b in directed:
        nxt.setdefault(int(a), []).append(int(b))
    used, loops = set(), []
    for a, b in directed:
        key = (int(a), int(b))
        if key in used:
            continue
        loop, cur, ok = [int(a)], int(b), True
        used.add(key)
        for _ in range(len(directed) + 1):
            if cur == loop[0]:
                break
            loop.append(cur)
            cands = [c for c in nxt.get(cur, []) if (cur, c) not in used]
            if not cands:
                ok = False
                break
            used.add((cur, cands[0]))
            cur = cands[0]
        if ok and cur == loop[0] and len(loop) >= 3:
            loops.append(loop)
    new = []
    for loop in loops:
        if len(loop) > max_loop:
            continue
        for i in range(1, len(loop) - 1):
            new.append([loop[0], loop[i + 1], loop[i]])
    if not new:
        return V, F, 0
    return V, np.vstack([F, np.asarray(new, np.int64)]), len(loops)


def manifold_volume(V, F):
    """The volume of the mesh as a manifold (manifold3d), or None when it is not a closed 2-manifold."""
    import manifold3d as m3
    man = m3.Manifold(m3.Mesh(vert_properties=np.ascontiguousarray(V, np.float32),
                              tri_verts=np.ascontiguousarray(F, np.uint32)))
    if str(man.status()) != "Error.NoError" or man.is_empty():
        return None
    return float(man.volume())


def _canonical_poses(c, I):
    """Rigid moves under which a solid is meshed, in order: its principal frame (centroid at the origin, inertia axes on
    x, y, z), the same turned 30 deg about z, and where it is."""
    _w, E = np.linalg.eigh(np.asarray(I, float))
    if np.linalg.det(E) < 0:
        E[:, 0] *= -1.0
    R = E.T
    R30 = axis_angle_R([0, 0, 1], math.radians(30.0)) @ R
    return [make_T(R, -R @ c), make_T(R30, -R30 @ c), np.eye(4)]


def mesh_solid(shape, cfg, tmpdir, volume, centroid, inertia):
    """(V, F, notes) of a solid in the document frame, or None when no setting tessellates it at all."""
    M = cfg["mesh"]
    tries = [(M["deflection_mm"], M["angle_rad"])] + [tuple(x) for x in M.get("retry", [])]
    vtol = float(M.get("volume_tol_rel", 0.02))
    best = None
    try:
        poses = _canonical_poses(centroid, inertia)
    except Exception:  # noqa: BLE001
        poses = [np.eye(4)]
    attempts = [(tries[0], k) for k in range(len(poses))] + [(t, 0) for t in tries[1:]]
    for (defl, ang), k in attempts:
        T = poses[k]
        try:
            src = shape if k == len(poses) - 1 else _moved_copy(shape, T)
            got = _stl_mesh(src, defl, ang, tmpdir)
        except Exception:  # noqa: BLE001
            continue
        if got is None:
            continue
        V, F = weld(got[0], got[1], M["weld_mm"])
        V, F, filled = fill_holes(V, F, int(M["fill_max_loop"]))
        if len(F) < 4:
            continue
        V = apply(inv(T), V)
        vol = manifold_volume(V, F)
        err = abs(vol / volume - 1.0) if (vol is not None and volume > 0) else np.inf
        notes = ["filled %d hole(s)" % filled] if filled else []
        if (defl, ang) != tries[0]:
            notes.append("re-meshed at %g mm / %g rad" % (defl, ang))
        if err <= vtol:
            return V, F, notes
        if best is None or err < best[0]:
            best = (err, V, F, notes)
    if best is None:
        return None
    err, V, F, notes = best
    if math.isfinite(err):
        notes = notes + ["the closest tessellation's volume is off by %.1f %%" % (100 * err)]
    else:
        notes = notes + ["the tessellation is not a closed manifold"]
    return V, F, notes


# ---------------------------------------------------------------------------------------------------- samples
def triangle_areas(T):
    T = np.asarray(T, float)
    cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    return 0.5 * np.sqrt(np.einsum("ij,ij->i", cr, cr))


def area_centroid(T):
    """Area-weighted centroid of a triangle set (its surface centroid) and its area."""
    T = np.asarray(T, float)
    if len(T) == 0:
        return np.zeros(3), 0.0
    a = triangle_areas(T)
    A = float(a.sum())
    if not A > 0:
        return T.reshape(-1, 3).mean(axis=0), 0.0
    c = (T[:, 0] + T[:, 1] + T[:, 2]) / 3.0
    return (c * a[:, None]).sum(axis=0) / A, A


def sample_count(area, per_mm2, n_min, n_max):
    """area x density, at least n_min, at most n_max (and whether the cap cut it)."""
    want = int(math.ceil(float(per_mm2) * float(area)))
    n = max(int(n_min), want)
    return min(n, int(n_max)), n > int(n_max)


def sample_triangles(T, n, seed):
    """n points on the triangles T (M, 3, 3) by area, with the unit normal of each point's triangle and its index."""
    T = np.asarray(T, float)
    n = int(n)
    if n <= 0 or len(T) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
    e1 = T[:, 1] - T[:, 0]
    e2 = T[:, 2] - T[:, 0]
    cr = np.cross(e1, e2)
    a2 = np.sqrt(np.einsum("ij,ij->i", cr, cr))
    cdf = np.cumsum(a2)
    if not cdf[-1] > 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
    rng = np.random.default_rng(int(seed) % (2 ** 63))
    r = rng.random(n) * cdf[-1]
    tri = np.minimum(np.searchsorted(cdf, r, side="right"), len(T) - 1).astype(np.int64)
    u = rng.random(n)
    v = rng.random(n)
    flip = (u + v) > 1.0
    u = np.where(flip, 1.0 - u, u)
    v = np.where(flip, 1.0 - v, v)
    P = (T[tri, 0] + u[:, None] * e1[tri]) + v[:, None] * e2[tri]
    N = cr[tri] / a2[tri][:, None]
    return P, N, tri


CHUNK = 4096


def nearest_within(tree, Q, cap, coarse_tree=None):
    """(distance, index) of the exact nearest point of `tree` within `cap` for every row of Q (inf and index 0 where
    there is none; cap None: no limit). coarse_tree is a KD-tree over a subset of tree's points: its nearest distance
    bounds the full set's from above, so each search starts pruned (queries sorted by that bound, one scalar bound per
    chunk) and the result is still the exact nearest point. An unbounded search far from a dense cloud visits many of
    its leaves; a capped one stops at once."""
    Q = np.asarray(Q, float)
    n = len(Q)
    d = np.full(n, np.inf)
    idx = np.zeros(n, np.int64)
    if n == 0:
        return d, idx
    ub = np.inf if cap is None else float(cap)
    if coarse_tree is None:
        dd, ii = tree.query(Q, distance_upper_bound=ub)
        ok = np.isfinite(dd)
        d[ok], idx[ok] = dd[ok], ii[ok]
        return d, idx
    ds, _ = coarse_tree.query(Q, distance_upper_bound=ub)
    near = np.nonzero(np.isfinite(ds))[0]
    order = near[np.argsort(ds[near], kind="stable")]
    for c in range(0, len(order), CHUNK):
        sel = order[c:c + CHUNK]
        b = float(ds[sel].max()) * (1.0 + 1e-9) + 1e-9
        dd, ii = tree.query(Q[sel], distance_upper_bound=b)
        if not np.all(np.isfinite(dd)):
            raise ValueError("a bounded nearest-point search found nothing within its bound")
        d[sel], idx[sel] = dd, ii
    rest = np.nonzero(~np.isfinite(ds))[0]
    if len(rest):
        dd, ii = tree.query(Q[rest], distance_upper_bound=ub)
        ok = np.isfinite(dd)
        d[rest[ok]], idx[rest[ok]] = dd[ok], ii[ok]
    return d, idx


def surface_distances(query, samples, normals, tree, disk_mm, coarse_tree=None, cap=None):
    """Distance of each query point to a surface given by dense samples with normals: the distance to the tangent disk
    (radius disk_mm) of the nearest sample (the family's statistic, UAV grade.surface_distances). With a cap, a point
    whose nearest sample is farther than cap gets inf: its distance is at least cap - disk_mm."""
    d, idx = nearest_within(tree, query, cap, coarse_tree)
    out = np.full(len(d), np.inf)
    ok = np.isfinite(d)
    v = query[ok] - samples[idx[ok]]
    dn = np.abs(np.einsum("ij,ij->i", v, normals[idx[ok]]))
    dl = np.sqrt(np.maximum(0.0, d[ok] * d[ok] - dn * dn))
    out[ok] = np.sqrt(dn * dn + np.maximum(0.0, dl - disk_mm) ** 2)
    return out


def pooled_quantile(parts, q, zero, disk_mm, caps):
    """The q-quantile (numpy's linear interpolation) of the pooled distances of `parts`, exact where it matters.

    parts: [(query points, samples, normals, tree, coarse_tree)]. Distances are computed with the caps in turn, the
    first cap for every point, each larger one only for the points still beyond the previous cap. A point beyond cap c
    is known to lie at least c - disk_mm away; with those lower bounds the quantile is exact as soon as both order
    statistics it interpolates lie below c - disk_mm, and it is known to be >= zero (a score of 0 on any band whose zero
    line is `zero`) as soon as its lower bound reaches zero. Returns (value, exact, distances of each part) where value
    is the quantile (exact) or its lower bound."""
    D = [np.full(len(p[0]), np.inf) for p in parts]
    todo = [np.arange(len(p[0])) for p in parts]
    levels = list(caps) + [None]
    for cap in levels:
        for k, (Q, S, N, tree, coarse) in enumerate(parts):
            sel = todo[k]
            if len(sel) and tree is not None:
                D[k][sel] = surface_distances(Q[sel], S, N, tree, disk_mm, coarse, cap)
            todo[k] = sel[~np.isfinite(D[k][sel])] if len(sel) else sel
        allv = np.concatenate(D) if D else np.zeros(0)
        if len(allv) == 0:
            return np.inf, False, D
        floor_ = (cap - disk_mm) if cap is not None else np.inf
        L = np.where(np.isfinite(allv), allv, floor_)
        val = float(np.quantile(L, q))
        h = q * (len(L) - 1)
        lo = int(np.floor(h))
        hi = min(lo + 1, len(L) - 1)
        part = np.partition(L, [lo, hi])
        top = part[hi] if h > lo else part[lo]
        if cap is None or top < floor_:
            return val, bool(np.all(np.isfinite(L))) or top < floor_, D
        if val >= zero:
            return val, False, D
    return val, True, D
