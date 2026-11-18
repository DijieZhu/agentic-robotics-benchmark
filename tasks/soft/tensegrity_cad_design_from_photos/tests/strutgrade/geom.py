"""Geometry of the strut grader: rigid-motion helpers, B-rep access (OCP), tessellation, manifold3d booleans, seeded
dense surface sampling with normals and the one-way tangent-disk surface distance.

Leaf module: it imports numpy, scipy, trimesh, manifold3d and OCP only (OCP and manifold3d lazily) and, from the
package, only TaskError. Every number comes from the resolved configuration (strutgrade.resolve_config); the defaults
are in strutgrade.BASE_DEFAULTS.

Sampling (the contract the visibility marks of tests/reference/seen.json are written against):

    sample_surface(V, F, n, seed) -> (points (n, 3), unit face normals (n, 3), face index (n,))

draws n points uniformly by area from the triangles (V[F]) with numpy's PCG64 generator seeded by `seed`: first n
uniforms pick the triangle through the cumulative area (searchsorted, side "right"), then an (n, 2) block of uniforms
gives the barycentric pair (reflected into the triangle when the pair sums above 1). n for a reference design =
n_samples(area, cfg["sampling"]) and the seed = seed_for(cfg, "reference", design key) (strutgrade.detenv). The
point set's identity is points_hash (sha256 of the points rounded to 1e-6 mm). Anyone reproducing the samples (the
photo QA) calls this function; the freeze compares the hashes.
"""
import hashlib
import itertools
import math
import os
import tempfile

import numpy as np


from . import TaskError  # noqa: E402,F401  (defined in the package, standard library only)


# ============================================================================ small helpers
def band(value, b):
    """1 when |value| <= b[0], 0 when |value| >= b[1], linear between; non-finite -> 0."""
    v = abs(float(value))
    if not math.isfinite(v):
        return 0.0
    full, zero = float(b[0]), float(b[1])
    if v <= full:
        return 1.0
    if v >= zero:
        return 0.0
    return (zero - v) / (zero - full)


def rising(value, b):
    """0 at value <= b[0], 1 at value >= b[1], linear between; non-finite -> 0."""
    v = float(value)
    if not math.isfinite(v):
        return 0.0
    lo, hi = float(b[0]), float(b[1])
    return float(min(1.0, max(0.0, (v - lo) / (hi - lo))))


def finite(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def apply(T, P):
    T = np.asarray(T, float)
    P = np.asarray(P, float)
    if P.ndim == 1:
        return T[:3, :3] @ P + T[:3, 3]
    return P @ T[:3, :3].T + T[:3, 3]


def apply_dir(T, D):
    T = np.asarray(T, float)
    D = np.asarray(D, float)
    return D @ T[:3, :3].T if D.ndim > 1 else T[:3, :3] @ D


def inv(T):
    T = np.asarray(T, float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def orthonormal(R):
    U, _s, Vt = np.linalg.svd(np.asarray(R, float)[:3, :3])
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def rigid(T):
    """T with its rotation re-orthonormalised."""
    T = np.asarray(T, float)
    return make_T(orthonormal(T[:3, :3]), T[:3, 3])


def rot_angle_deg(R):
    R = np.asarray(R, float)[:3, :3]
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return math.degrees(math.atan2(float(np.linalg.norm(v)), 0.5 * (float(np.trace(R)) - 1.0)))


def axis_angle_R(axis, ang):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def rotation_about(point, axis, ang):
    """Rigid 4x4 turning by ang (rad) about the line through point along axis."""
    R = axis_angle_R(axis, ang)
    p = np.asarray(point, float)
    return make_T(R, p - R @ p)


def rotation_between(a, b):
    """The proper rotation of least angle taking unit vector a onto unit vector b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        p = p - a * (p @ a)
        return axis_angle_R(p, math.pi)
    return axis_angle_R(v / s, math.atan2(s, c))


def kabsch(P, Q, w=None):
    """Proper rigid T minimising sum w |T(P) - Q|^2 over corresponding rows (no scale, no reflection)."""
    P = np.asarray(P, float)
    Q = np.asarray(Q, float)
    if w is None:
        w = np.ones(len(P))
    w = np.asarray(w, float)
    sw = float(w.sum())
    if sw <= 0:
        return np.eye(4)
    cp = (w[:, None] * P).sum(axis=0) / sw
    cq = (w[:, None] * Q).sum(axis=0) / sw
    H = ((P - cp) * w[:, None]).T @ (Q - cq)
    U, _s, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return make_T(R, cq - R @ cp)


def hungarian(cost):
    from scipy.optimize import linear_sum_assignment
    cost = np.asarray(cost, float)
    if cost.size == 0:
        return []
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def pose_distance(Ta, Tb, pts):
    """Largest displacement of the points pts between the poses Ta and Tb (mm)."""
    return float(np.max(np.linalg.norm(apply(Ta, pts) - apply(Tb, pts), axis=1))) if len(pts) else 0.0


def box_corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def points_hash(P):
    """Identity of a sample set: sha256 of the points rounded to 1e-6 mm (little-endian float64), with the count."""
    P = np.ascontiguousarray(np.round(np.asarray(P, float), 6), dtype="<f8")
    h = hashlib.sha256()
    h.update(str(P.shape).encode())
    h.update(P.tobytes())
    return h.hexdigest()


# ============================================================================ sampling
def triangle_areas(V, F):
    tri = np.asarray(V, float)[np.asarray(F, np.int64)]
    cr = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return 0.5 * np.linalg.norm(cr, axis=1), cr


def n_samples(area, S, kind="dense"):
    """Number of samples on a surface of the given area: kind "dense" (the reference and every distance target) or
    "query" (the points distances are measured from), from the sampling section of the configuration."""
    per = float(S["%s_per_mm2" % kind])
    lo, hi = int(S["%s_min" % kind]), int(S["%s_max" % kind])
    return int(min(hi, max(lo, int(math.ceil(per * float(area))))))


def sample_surface(V, F, n, seed):
    """n points uniformly by area on the triangles V[F], with their unit face normals and face indices (module
    docstring: the exact algorithm is part of the grader's contract)."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    n = int(n)
    if len(F) == 0 or n <= 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
    a, cr = triangle_areas(V, F)
    tot = float(a.sum())
    if not tot > 0:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0, np.int64)
    cdf = np.cumsum(a) / tot
    rng = np.random.Generator(np.random.PCG64(int(seed) % (2 ** 63)))
    u = rng.random(n)
    fid = np.minimum(np.searchsorted(cdf, u, side="right"), len(F) - 1)
    r = rng.random((n, 2))
    over = r.sum(axis=1) > 1.0
    r[over] = 1.0 - r[over]
    tri = V[F[fid]]
    P = tri[:, 0] + r[:, :1] * (tri[:, 1] - tri[:, 0]) + r[:, 1:2] * (tri[:, 2] - tri[:, 0])
    nrm = cr[fid]
    ln = np.linalg.norm(nrm, axis=1)
    N = nrm / np.where(ln > 0, ln, 1.0)[:, None]
    return P, N, fid


def even_subset(P, k):
    """k rows of P at evenly spaced indices (deterministic; P itself when it has no more than k rows)."""
    P = np.asarray(P)
    if len(P) <= k:
        return P
    return P[np.linspace(0, len(P) - 1, int(k)).astype(np.int64)]


# ============================================================================ B-rep (OCP)
def read_brep(path):
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    s = TopoDS_Shape()
    if not BRepTools.Read_s(s, str(path), BRep_Builder()):
        raise IOError("cannot read %s" % path)
    return s


def write_brep(shape, path):
    from OCP.BRepTools import BRepTools
    if not BRepTools.Write_s(shape, str(path)):
        raise IOError("cannot write %s" % path)


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


def faces_of(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    ex = TopExp_Explorer(shape, TopAbs_FACE)
    out = []
    while ex.More():
        out.append(TopoDS.Face_s(ex.Current()))
        ex.Next()
    return out


def brep_valid(shape):
    from OCP.BRepCheck import BRepCheck_Analyzer
    return bool(BRepCheck_Analyzer(shape).IsValid())


def heal_tolerances(shape):
    """BRepLib.UpdateTolerances on a solid read from BREP that fails BRepCheck; returns whether it is
    valid afterwards."""
    try:
        from OCP.BRepLib import BRepLib
        BRepLib.UpdateTolerances_s(shape, False)
    except Exception:  # noqa: BLE001
        return False
    return brep_valid(shape)


def mass_props(shape, eps=None):
    """(volume, centroid (3,), inertia tensor about the centroid (3, 3)) at density 1 (OCC GProp, exact on the B-rep).
    eps: adaptive integration (the purchased pins are written with eps 1e-7: the default integration differs by up to
    5e-5 relative on B-spline solids)."""
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    if eps is None:
        BRepGProp.VolumeProperties_s(shape, g)
    else:
        BRepGProp.VolumeProperties_s(shape, g, float(eps), True)
    V = float(g.Mass())
    c = g.CentreOfMass()
    M = g.MatrixOfInertia()
    I = np.array([[M.Value(i, j) for j in (1, 2, 3)] for i in (1, 2, 3)], float)
    if V < 0:
        V, I = -V, -I
    return V, np.array([c.X(), c.Y(), c.Z()], float), I


def unique_vertices(shape):
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp
    from OCP.TopoDS import TopoDS
    from OCP.TopTools import TopTools_IndexedMapOfShape
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_VERTEX, m)
    n = m.Extent()
    P = np.empty((n, 3), float)
    for i in range(1, n + 1):
        p = BRep_Tool.Pnt_s(TopoDS.Vertex_s(m.FindKey(i)))
        P[i - 1] = (p.X(), p.Y(), p.Z())
    return P


def n_faces(shape):
    from OCP.TopAbs import TopAbs_FACE
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedMapOfShape
    m = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, m)
    return int(m.Extent())


def fresh_copy(shape):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepTools import BRepTools
    c = BRepBuilderAPI_Copy(shape, True, False).Shape()
    BRepTools.Clean_s(c)
    return c


def moved_copy(shape, T):
    """The shape moved rigidly by T (re-orthonormalised), as a copy."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    tr = gp_Trsf()
    tr.SetValues(*(float(v) for v in rigid(T)[:3, :].ravel()))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


# ============================================================================ meshes
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
    V2 = V[first]
    F2 = labc[F]
    keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[keep]
    srt = np.sort(F2, axis=1)
    _u2, idx = np.unique(srt, axis=0, return_index=True)
    return V2, F2[np.sort(idx)]


def fill_holes(V, F, max_loop):
    """Close boundary loops of at most max_loop edges of a *tessellation of a submitted B-rep* with triangle fans (the
    OCC mesher leaves seam gaps on some imported B-reps; as in the UAV CAD task). Never used on a reference mesh."""
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


EXACT64_NOTE = "manifold: float64, bit-identical vertices merged only"
REPAIR_NOTE = "manifold: repaired locally (non-manifold edges)"


def to_manifold(V, F):
    import manifold3d as m3
    man = m3.Manifold(m3.Mesh(vert_properties=np.ascontiguousarray(V, np.float32),
                              tri_verts=np.ascontiguousarray(F, np.uint32)))
    return man if str(man.status()) == "Error.NoError" and not man.is_empty() else None


def exact_manifold64(V, F):
    """A float64 manifold (manifold3d Mesh64) of a triangle soup with only its bit-identical vertices merged: no weld
    tolerance, so nearly touching sheets (a closed slit's two walls) stay apart; None
    when that is no closed two-manifold."""
    import manifold3d as m3
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    if len(F) < 4:
        return None
    U, inv_ = np.unique(V, axis=0, return_inverse=True)
    F2 = np.asarray(inv_, np.int64).reshape(-1)[F]
    keep = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 0] != F2[:, 2])
    F2 = F2[keep]
    if len(F2) < 4:
        return None
    try:
        man = m3.Manifold(m3.Mesh64(vert_properties=np.ascontiguousarray(U, np.float64),
                                    tri_verts=np.ascontiguousarray(F2, np.uint64)))
    except Exception:  # noqa: BLE001
        return None
    return man if str(man.status()) == "Error.NoError" and not man.is_empty() else None


def repair_manifold(V, F, rounds, max_loop):
    """The local repair of a tessellation that is no closed two-manifold:
    the triangles on edges not shared by exactly two triangles and their one-ring are
    removed and the holes closed with fans (fill_holes), at most `rounds` times; the float32 manifold of the result,
    or None."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    man = None
    for _round in range(int(rounds)):
        e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
        _u, inv_, cnt = np.unique(e, axis=0, return_inverse=True, return_counts=True)
        bad = np.any(cnt[np.asarray(inv_).reshape(3, -1).T] != 2, axis=1)
        if not bad.any():
            man = to_manifold(V, F)
            break
        ring = np.isin(F, np.unique(F[bad])).any(axis=1)
        V, F, _n = fill_holes(V, F[~ring], int(max_loop))
        if len(F) < 4:
            return None
        man = to_manifold(V, F)
        if man is not None:
            break
    return man


def mesh_mass(V, F):
    """(volume, centroid, inertia about the centroid) of a closed, outward-oriented triangle mesh at density 1
    (exact polyhedral integrals; signed tetrahedra from the origin)."""
    V = np.asarray(V, float)
    tri = V[np.asarray(F, np.int64)]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    d = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    vol = float(d.sum())
    if abs(vol) < 1e-15:
        return 0.0, V.mean(axis=0) if len(V) else np.zeros(3), np.zeros((3, 3))
    cen = ((a + b + c) * d[:, None]).sum(axis=0) / (4.0 * vol)
    # second moments about the origin: for a tetrahedron (0, a, b, c), int x_i x_j = det/120 * (sum of the 10 terms)
    S = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            S[i, j] = float((d * 6.0 / 120.0 * (
                2 * (a[:, i] * a[:, j] + b[:, i] * b[:, j] + c[:, i] * c[:, j]) +
                a[:, i] * b[:, j] + a[:, j] * b[:, i] + a[:, i] * c[:, j] + a[:, j] * c[:, i] +
                b[:, i] * c[:, j] + b[:, j] * c[:, i])).sum())
    S_c = S - vol * np.outer(cen, cen)
    I = np.trace(S_c) * np.eye(3) - S_c
    if vol < 0:
        vol, I = -vol, -I
    return vol, cen, I


class Geo(object):
    """A solid's triangle surface in its own frame: vertices, faces, manifold (or None), area, volume, bounds and,
    computed once on demand, the dense samples (points, unit normals, KD-tree) at the configured density and query
    samples. `seed` fixes both sample sets (dense: seed, query: seed + 1)."""

    def __init__(self, V, F, man, notes, seed, cfg):
        self.V = np.asarray(V, float)
        self.F = np.asarray(F, np.int64)
        self.man, self.notes, self.seed, self.cfg = man, list(notes), int(seed), cfg
        a, _cr = triangle_areas(self.V, self.F)
        self.area = float(a.sum())
        self.volume = float(man.volume()) if man is not None else abs(mesh_mass(self.V, self.F)[0])
        self.bounds = np.array([self.V.min(axis=0), self.V.max(axis=0)]) if len(self.V) else np.zeros((2, 3))
        self._dense = self._query = None

    def dense(self):
        """(points, normals, cKDTree) at sampling.dense_per_mm2 (>= 20 points / mm2)."""
        if self._dense is None:
            from scipy.spatial import cKDTree
            n = n_samples(self.area, self.cfg["sampling"], "dense")
            P, N, _f = sample_surface(self.V, self.F, n, self.seed)
            self._dense = (P, N, cKDTree(P) if len(P) else None)
        return self._dense

    def query(self):
        """(points, normals) at sampling.query_per_mm2."""
        if self._query is None:
            n = n_samples(self.area, self.cfg["sampling"], "query")
            P, N, _f = sample_surface(self.V, self.F, n, self.seed + 1)
            self._query = (P, N)
        return self._query

    def coarse(self):
        """(points, normals, cKDTree) of the query samples: the cheap target of a coarse registration."""
        if getattr(self, "_coarse", None) is None:
            from scipy.spatial import cKDTree
            P, N = self.query()
            self._coarse = (P, N, cKDTree(P) if len(P) else None)
        return self._coarse


def geo_from_mesh(V, F, seed, cfg, notes=()):
    """Geo of a closed mesh taken as it is (the reference designs: no welding, no repair)."""
    V = np.asarray(V, float)
    F = np.asarray(F, np.int64)
    return Geo(V, F, to_manifold(V, F), notes, seed, cfg)


def _canonical_poses(shape):
    """Rigid moves under which a solid is meshed, in order: its principal frame, the same turned 30 deg about z, and
    where it is (OCC's mesher depends on the placement)."""
    _V, c, I = mass_props(shape)
    _w, E = np.linalg.eigh(I)
    if np.linalg.det(E) < 0:
        E[:, 0] *= -1.0
    R = E.T
    R30 = axis_angle_R([0, 0, 1], math.radians(30.0)) @ R
    return [make_T(R, -R @ c), make_T(R30, -R30 @ c), np.eye(4)]


def mesh_solid(shape, cfg, tmpdir, seed, volume=None):
    """Geo of a B-rep solid, in the solid's own frame: the OCC tessellation (deflection / angle from config, then the
    retry list), welded and hole-filled, tried in each canonical pose until it has a manifold whose volume is within
    volume_tol_rel of the B-rep volume; else the closest one (noted; a mesh that is no manifold has man None and the
    booleans use its convex hull). A faceted solid tessellates to its own triangles.

    The manifold of a tessellation is the first of: (1) the float64 manifold
    of the raw tessellation with only bit-identical vertices merged (exact_manifold64; mesh.exact64), (2) the float32
    manifold of the welded, hole-filled mesh, (3) the local repair of that mesh (repair_manifold, mesh.repair_rounds
    rounds) with the note REPAIR_NOTE. The surface samples are always those of the welded, hole-filled mesh."""
    M = cfg["mesh"]
    tries = [(M["deflection_mm"], M["angle_rad"])] + [tuple(x) for x in M.get("retry", [])]
    if volume is None:
        volume = mass_props(shape)[0]
    vtol = float(M.get("volume_tol_rel", 0.02))
    use64 = bool(M.get("exact64", True))
    rounds = int(M.get("repair_rounds", 3))

    def vol_err(man):
        if man is None:
            return np.inf
        return abs(float(man.volume()) / volume - 1.0) if volume > 0 else np.inf
    best = None
    tried = []
    try:
        poses = _canonical_poses(shape)
    except Exception:  # noqa: BLE001
        poses = [np.eye(4)]
    # placed as it is first: a faceted solid meshes to its own triangles wherever it is, and the identity move keeps
    # the vertex coordinates bit for bit
    order = [len(poses) - 1] + list(range(len(poses) - 1))
    attempts = [(tries[0], k) for k in order] + [(t, 0) for t in tries[1:]]
    for (defl, ang), k in attempts:
        T = poses[k]
        try:
            src = shape if k == len(poses) - 1 else moved_copy(shape, T)
            got = _stl_mesh(src, defl, ang, tmpdir)
        except Exception:  # noqa: BLE001
            continue
        if got is None:
            continue
        V, F = weld(got[0], got[1], M["weld_mm"])
        V, F, filled = fill_holes(V, F, int(M["fill_max_loop"]))
        if len(F) < 4:
            continue
        raw_V = got[0]
        if k != len(poses) - 1:
            V = apply(inv(T), V)
            raw_V = apply(inv(T), raw_V)
        notes = ["filled %d hole(s)" % filled] if filled else []
        if (defl, ang) != tries[0]:
            notes.append("re-meshed at %g mm / %g rad" % (defl, ang))
        man, err = None, np.inf
        if use64:
            man = exact_manifold64(raw_V, got[1])
            err = vol_err(man)
            if man is not None and err <= vtol:
                return Geo(V, F, man, notes + [EXACT64_NOTE], seed, cfg)
        man = to_manifold(V, F)
        err = vol_err(man)
        if man is not None and err <= vtol:
            return Geo(V, F, man, notes, seed, cfg)
        tried.append((V, F, notes))
        if best is None or err < best[0]:
            best = (err, V, F, man, notes)
    # no attempt gave a clean manifold: the local repair, attempt by attempt in the same order, before the hull
    if rounds > 0:
        for V, F, notes in tried:
            rman = repair_manifold(V, F, rounds, int(M["fill_max_loop"]))
            if rman is not None and vol_err(rman) <= vtol:
                return Geo(V, F, rman, notes + [REPAIR_NOTE], seed, cfg)
    if best is None:
        return None
    err, V, F, man, notes = best
    if man is not None and err > vtol:
        notes = notes + ["the closest tessellation's volume is off by %.1f %%" % (100 * err)]
        man = None if err > 0.5 else man
    if man is None:
        notes = notes + ["the tessellation is not a closed manifold of the solid's volume: booleans use its convex hull"]
    return Geo(V, F, man, notes, seed, cfg)


def placed_manifold(geo, T):
    """The geo's manifold moved by T (a rigid 4x4), or its convex hull when it has none."""
    import manifold3d as m3
    man = geo.man
    if man is None:
        try:
            man = m3.Manifold.hull_points(np.ascontiguousarray(geo.V, np.float32))
        except Exception:  # noqa: BLE001
            return None
    return man.transform(np.ascontiguousarray(np.asarray(T, float)[:3, :], np.float32))


def aabb(geo, T):
    w = apply(T, box_corners(*geo.bounds))
    return np.concatenate([w.min(axis=0), w.max(axis=0)])


def aabb_overlap(a, b, margin=0.0):
    return all(a[i] - margin <= b[i + 3] and b[i] - margin <= a[i + 3] for i in range(3))


def common_volume(ma, mb):
    """Volume of the intersection of two manifolds (0 for an empty or failed boolean)."""
    if ma is None or mb is None:
        return 0.0
    try:
        return max(0.0, float((ma ^ mb).volume()))
    except Exception:  # noqa: BLE001
        return 0.0


def iou(geo_a, T_a, geo_b, T_b):
    """Volume IoU of two solids at their poses (manifold3d exact booleans)."""
    if not aabb_overlap(aabb(geo_a, T_a), aabb(geo_b, T_b)):
        return 0.0
    ma, mb = placed_manifold(geo_a, T_a), placed_manifold(geo_b, T_b)
    if ma is None or mb is None:
        return 0.0
    inter = common_volume(ma, mb)
    va, vb = float(ma.volume()), float(mb.volume())
    return float(inter / max(va + vb - inter, 1e-9))


# ============================================================================ distances
def surface_distances(query, samples, normals, tree, disk_mm):
    """Distance of each query point to a surface given by dense samples with normals: the distance to the tangent disk
    (radius disk_mm) of the nearest sample (the bimanual task's cad_tree.checks, as in the UAV CAD task)."""
    if tree is None or len(query) == 0:
        return np.full(max(1, len(query)), np.inf)
    d, idx = tree.query(query)
    v = query - samples[idx]
    dn = np.abs(np.einsum("ij,ij->i", v, normals[idx]))
    dl = np.sqrt(np.maximum(0.0, d * d - dn * dn))
    return np.sqrt(dn * dn + np.maximum(0.0, dl - disk_mm) ** 2)


def one_way(P, dense, disk_mm):
    """One-way distances of the points P to the surface whose dense samples are dense = (points, normals, tree)."""
    D, N, tree = dense
    return surface_distances(np.asarray(P, float), D, N, tree, disk_mm)


class Placed(object):
    """A Geo at a pose T (its frame -> a target frame): transformed samples and manifold, computed on demand."""

    def __init__(self, geo, T):
        self.geo, self.T = geo, np.asarray(T, float)
        self.box = aabb(geo, T)
        self._man = self._dense = self._query = None
        self.volume = geo.volume

    @property
    def man(self):
        if self._man is None:
            self._man = placed_manifold(self.geo, self.T)
        return self._man

    def dense(self):
        if self._dense is None:
            from scipy.spatial import cKDTree
            P, N, _t = self.geo.dense()
            P2, N2 = apply(self.T, P), apply_dir(self.T, N)
            self._dense = (P2, N2, cKDTree(P2) if len(P2) else None)
        return self._dense

    def query(self):
        if self._query is None:
            P, N = self.geo.query()
            self._query = (apply(self.T, P), apply_dir(self.T, N))
        return self._query


# ============================================================================ registration
def eigen(I):
    w, E = np.linalg.eigh(np.asarray(I, float))
    return w, E


def principal_starts(com_s, I_s, com_d, I_d):
    """The 24 proper rotations taking the source's principal (inertia) axes onto the destination's, every axis
    permutation and sign with det +1, centroids matched: rigid 4x4 source -> destination."""
    _ws, Es = eigen(I_s)
    _wd, Ed = eigen(I_d)
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[:, i] = s * Ed[:, p]
            R = M @ Es.T
            if np.linalg.det(R) < 0:
                continue
            out.append(make_T(R, com_d - R @ com_s))
    return out


def icp(src, dst, dst_n, tree, T0, iters, trim, bound_mm=30.0, step_rad=0.15, tol=1e-10):
    """Trimmed point-to-plane ICP of src onto (dst, dst_n): returns (T, trimmed RMS). Deterministic (stable sorts; at
    most iters iterations, fewer once a step moves less than tol (rad and mm))."""
    T = np.array(T0, float)
    for _ in range(int(iters)):
        P = apply(T, src)
        d, idx = tree.query(P, distance_upper_bound=bound_mm)
        far = ~np.isfinite(d)
        d = np.where(far, bound_mm, d)
        idx = np.where(far, 0, idx)
        k = max(6, int(len(d) * trim))
        keep = np.argsort(d, kind="stable")[:k]
        p, q, n = P[keep], dst[idx[keep]], dst_n[idx[keep]]
        c = p.mean(axis=0)
        A = np.c_[np.cross(p - c, n), n]
        b = np.einsum("ij,ij->i", q - p, n)
        try:
            x, *_ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(x)):
            break
        w = float(np.linalg.norm(x[:3]))
        if w > step_rad:
            x[:3] *= step_rad / w
        R = axis_angle_R(x[:3], float(np.linalg.norm(x[:3]))) if np.linalg.norm(x[:3]) > 1e-15 else np.eye(3)
        dT = make_T(R, c - R @ c + x[3:])
        T = dT @ T
        if float(np.linalg.norm(x)) < tol:
            break
    T = rigid(T)
    d, _ = tree.query(apply(T, src), distance_upper_bound=bound_mm)
    d = np.where(np.isfinite(d), d, bound_mm)
    k = max(1, int(len(d) * trim))
    return T, float(math.sqrt(np.mean(np.sort(d, kind="stable")[:k] ** 2)))


def _axial_rotations(src_v, cs, Is, dst_v, cd, Id, rh_tol):
    """Rotations for a source whose inertia has two equal moments (a body of revolution or of 3+-fold symmetry): the
    distinct principal axis onto the destination's (both senses), then the turn about it that takes the source's most
    distinctive off-axis vertex onto each destination vertex of the same radius and height (as in the UAV CAD task)."""
    ws, Es = eigen(Is)
    wd, Ed = eigen(Id)
    out = []
    sc = max(abs(ws).max(), 1e-12)
    for k in (0, 2):
        others = [i for i in range(3) if i != k]
        if abs(ws[others[0]] - ws[others[1]]) > 1e-3 * sc and abs(wd[others[0]] - wd[others[1]]) > 1e-3 * sc:
            continue
        a_s, a_d = Es[:, k], Ed[:, k]
        S = src_v - cs
        hs = S @ a_s
        rs = np.linalg.norm(S - np.outer(hs, a_s), axis=1)
        if rs.max() < 1e-6:
            continue
        cand = np.nonzero(rs > 0.3 * rs.max())[0]
        q = np.round(np.c_[rs[cand], hs[cand]] / rh_tol).astype(np.int64)
        _u, inv_, cnt = np.unique(q, axis=0, return_inverse=True, return_counts=True)
        pick = cand[int(np.argmin(cnt[inv_.ravel()] * 1e6 - rs[cand]))]
        D = dst_v - cd
        for sense in (1.0, -1.0):
            R0 = rotation_between(a_s, sense * a_d)
            hd = D @ (sense * a_d)
            rd = np.linalg.norm(D - np.outer(D @ a_d, a_d), axis=1)
            hit = np.nonzero((np.abs(rd - rs[pick]) < rh_tol) & (np.abs(hd - hs[pick]) < rh_tol))[0]
            u = R0 @ S[pick]
            u = u - a_d * float(u @ a_d)
            for j in hit[:64]:
                w = D[j] - a_d * float(D[j] @ a_d)
                ang = math.atan2(float(np.cross(u, w) @ a_d), float(u @ w))
                out.append(axis_angle_R(a_d, ang) @ R0)
    return out


def _pca_rotations(Is, Id):
    _ws, Es = eigen(Is)
    _wd, Ed = eigen(Id)
    out = []
    for signs in itertools.product((1.0, -1.0), repeat=3):
        R = Ed @ np.diag(signs) @ Es.T
        if np.linalg.det(R) > 0:
            out.append(R)
    return out


def register_exact(src, dst, tol, all_solutions=False):
    """Exact registration of two copies of one vertex set: src, dst = {"verts", "com", "I", "size"} (B-rep vertices,
    centroid and inertia of the same integration). Proper rotations only. Returns (T src->dst, max vertex error) or
    None; with all_solutions, every distinct exact solution (a symmetric vertex set has several), best first."""
    from scipy.spatial import cKDTree
    Vs, Vd = src["verts"], dst["verts"]
    if len(Vs) != len(Vd) or len(Vs) == 0:
        return [] if all_solutions else None
    tree = dst.get("tree")
    if tree is None:
        tree = cKDTree(Vd)
        dst["tree"] = tree
    cs, cd = src["com"], dst["com"]
    size = max(float(src["size"]), 1e-6)
    seed_tol = max(10.0 * tol, 1e-4 * size)

    def check(R):
        T = make_T(R, cd - R @ cs)
        d, idx = tree.query(apply(T, Vs))
        return float(d.max()), T, idx

    good = []
    for gen in ("pca", "axial"):
        Rs = _pca_rotations(src["I"], dst["I"]) if gen == "pca" else _axial_rotations(
            Vs, cs, src["I"], Vd, cd, dst["I"], max(1e-3, 1e-4 * size))
        for R in Rs:
            e, T, idx = check(R)
            if e <= seed_tol and len(np.unique(idx)) == len(idx):
                T = kabsch(Vs, Vd[idx])
                d, _idx = tree.query(apply(T, Vs))
                e = float(d.max())
                if e <= tol and all(rot_angle_deg(T[:3, :3] @ g[0][:3, :3].T) > 1e-3 for g in good):
                    good.append((T, e))
        if good and not all_solutions:
            break
    good.sort(key=lambda g: g[1])
    if all_solutions:
        return good
    return good[0] if good else None


def register_samples(src_geo, dst_geo, cfg, src=None, dst=None, n_keep=6):
    """Rigid registration of two tessellated solids (T src->dst, trimmed RMS): coarse ICP on a subsample from the 24
    principal-axis starts (exact inertia when both are given, else of the sample clouds), the best n_keep refined on
    the query samples (register_samples of the UAV CAD task, for purchased solids that are not an unchanged import)."""
    R = cfg["recognition"]
    q = src_geo.query()[0]
    D, N, tree = dst_geo.dense()
    if len(q) < 10 or tree is None:
        return np.eye(4), np.inf
    if src is not None and dst is not None:
        starts = principal_starts(src["com"], src["I"], dst["com"], dst["I"])
    else:
        cs, cd = q.mean(axis=0), D.mean(axis=0)
        starts = principal_starts(cs, np.cov((q - cs).T), cd, np.cov((D - cd).T))
    sub = even_subset(q, int(R.get("approx_coarse_points", 1200)))
    coarse = []
    for i, T0 in enumerate(starts):
        Tk, rk = icp(sub, D, N, tree, T0, 12, float(R.get("approx_trim", 0.9)))
        coarse.append((rk, i, Tk))
    coarse.sort(key=lambda x: (x[0], x[1]))
    best = (np.inf, np.eye(4))
    fine = even_subset(q, int(R.get("approx_fine_points", 6000)))
    for _rk, _i, Tk in coarse[:n_keep]:
        Tf, rf = icp(fine, D, N, tree, Tk, 30, float(R.get("approx_trim", 0.9)))
        if rf < best[0]:
            best = (rf, Tf)
    return best[1], best[0]
