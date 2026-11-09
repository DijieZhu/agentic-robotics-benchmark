"""Grader of uav_cad_design_from_photos: the extracted FreeCAD document of the AGIPIX v2 quadrotor
against the original manufacturing CAD (reference/), in millimetres. Root, verify stage, agent gone; reads data only.

    python3 -B grade.py EXTRACT_DIR REF_DIR PURCHASED_ROOT OUT_JSON [--config grader_config.json]

EXTRACT_DIR is fcstd_extract.py's output (model.json + one world-frame BREP per solid). REF_DIR holds reference.json
(schema agipix-cad-reference/2), parts/Pnn.brep (the 51 manufactured instances, world frame of the original assembly)
and purchased_pins.json (the vendor STEP of every purchased type, pinned by sha256, and which STEP solid is which
constituent). PURCHASED_ROOT is where the vendor STEPs are read: the image's own /opt/task-assets/payload/purchased
(the separate verifier sandbox is fresh from the image; a file whose sha256 differs from its pin is a task error).
Every number is in grader_config.json. OUT_JSON: {"ok", "design_score", "criteria", "checks", "statistics", "parts",
"purchased", "rotors", "diagnostics"} (or {"ok": false, "error"} when the grader itself fails).

The rules (the instruction's grading table states them):

 1. What is graded: the solids inside the document's Assembly (its members, App::Part containers, folders and links);
    without an Assembly, the document's visible solids, with no joint credit. Only three things zero the document,
    with the reason: it cannot be read; it holds no solid; P01 is missing or unusable for the alignment (on two leaves,
    several solids or none valid; a lone valid P01 solid that is no Body still aligns, without credit for P01).
    Everything else is local (Submission): a tagged leaf that is not a PartDesign Body of exactly one valid closed
    solid, or a tag on more than one leaf (a Body reused through links), gets no credit and its solids count as extra
    solids; invalid solids, open shells and loose faces of untagged leaves are left out; each is reported (diagnostic
    and statistic). A grading item that fails (one part's shape or features, one purchased instance's pose or fit, one
    rotor, one solid's recognition) scores 0 and counts in grader_item_errors; a failure outside every item is the
    grader's own and fails the run (task_error).
 2. Alignment: rigid (no scale, no reflection). Point-to-plane ICP of the submitted P01 onto the reference P01 from the
    24 principal-axis starts; among the candidates whose residual lies within the tie band of the best, the one under
    which the whole assembly (every tagged part's centroid to the nearest reference instance of its family, every
    purchased module to the nearest reference placement of its type) matches best. A P01 with the reference's exact
    vertex set is snapped by Kabsch on its vertices. The residual is reported.
 3. Purchased recognition by full shape, in the submission's own frame: every untagged solid against every constituent
    of every purchased type (the grader's copies, read from the pinned vendor STEPs): volume within volume_rel, then a
    rigid registration (proper rotations only, so a mirror-image propeller is not its twin): exact on the B-rep vertex
    set when the solid is an unchanged import, else ICP on surface samples accepted when the symmetric surface p95 is
    small. A module instance is seeded by every solid that matches its type's anchor constituent (the module pose); the
    other constituents join the instance whose predicted constituent centroid they meet within attach_mm (one-to-one).
    Completeness = the share of the type's constituents found. Instances go to reference placements one-to-one per type
    (Hungarian on the module centroid).
 4. Manufactured assignment: the tag gives the family; inside a family the tagged parts go to the reference instances
    one-to-one by installed position (Hungarian on the centroid). Tags inside a family are interchangeable; a part of a
    mirror family is never credited as its mirror image.
 5. Criteria (weights in config), each in [0, 1]:
    shape        per reference instance: w_surface band(symmetric surface p90) + w_iou rising(IoU), at the installed
                 placement; mean over the 51 instances (a missing one scores 0).
    features     per reference hole (and its counterbore steps): the best one-to-one match among the part's concave
                 cylindrical faces, band(axis offset) x band(radius error) x band(axis angle); mean per part, then over
                 the instances that have holes.
    installation per reference purchased instance: completeness x band(position) x (1 - w_orient + w_orient x
                 band(orientation modulo the type's symmetry)); a propeller is measured in its motor's frame (on the right
                 motor axis; the hand is its type; the spin phase is ignored); type weights; then w_poses x that +
                 w_connected x the share of parts and modules (weighted) connected to the grounded Fixed-joint graph.
    mass         w x band(total mass), w x band(centre of mass), w x band(largest principal-inertia error); purchased
                 constituents at their type's density, unrecognised solids at the PRINT density; the unscored source
                 solids (cable, screws) are on neither side.
    fit          per reference purchased instance: mean over its reference supports of band(contact distance beyond the
                 reference's) x band(penetration volume into manufactured parts and extra solids / own volume); per
                 manufactured part band(penetration into the other parts and extra solids / own volume); overlaps the
                 reference has itself are exempt.
    clearance    per rotor: the reference swept disc moved with the submitted motor, band(its overlap with everything
                 else: parts, modules other than the motor and its propeller, unrecognised solids, the other discs,
                 beyond the reference's own overlap).
 6. Shape floor: rising(mean shape score) multiplies features, installation, mass, fit and clearance.
design_score = sum of weight x criterion.
"""
import hashlib
import itertools
import json
import math
import os
import sys
import tempfile
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CRITERIA = ["shape", "features", "installation", "mass", "fit", "clearance"]
TAG_PREFIX = "P"


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


def apply(T, P):
    T = np.asarray(T, float)
    P = np.asarray(P, float)
    if P.ndim == 1:
        return T[:3, :3] @ P + T[:3, 3]
    return P @ T[:3, :3].T + T[:3, 3]


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


def rot_angle_deg(R):
    R = np.asarray(R, float)[:3, :3]
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return math.degrees(math.atan2(float(np.linalg.norm(v)), 0.5 * (float(np.trace(R)) - 1.0)))


def axis_angle_R(axis, ang):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


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


def kabsch(P, Q):
    """Proper rigid T minimising |T(P) - Q| over corresponding rows."""
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _s, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    R = Vt.T @ D @ U.T
    return make_T(R, cq - R @ cp)


def hungarian(cost):
    from scipy.optimize import linear_sum_assignment
    cost = np.asarray(cost, float)
    if cost.size == 0:
        return []
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def finite(x, default=0.0):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return default
    return x if math.isfinite(x) else default


def limit_cpus(n):
    """Pin this process to n CPUs: OCC's mesher and manifold3d's TBB size their pools to the host (skill pitfalls 64)."""
    try:
        cpus = sorted(os.sched_getaffinity(0))[:max(1, int(n))]
        os.sched_setaffinity(0, cpus)
        return cpus
    except (AttributeError, OSError):
        return []


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ============================================================================ B-rep (OCP)
def read_brep(path):
    from OCP.BRep import BRep_Builder
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import TopoDS_Shape
    s = TopoDS_Shape()
    if not BRepTools.Read_s(s, str(path), BRep_Builder()):
        raise IOError("cannot read %s" % path)
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


def fresh_copy(shape):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepTools import BRepTools
    c = BRepBuilderAPI_Copy(shape, True, False).Shape()
    BRepTools.Clean_s(c)
    return c


def concave_axial_faces(shape):
    """Concave cylindrical and conical faces of a solid (the material outside the surface: holes, bores, countersinks):
    [{kind "cylinder" | "cone", point (on the axis), dir, radius (a cone: its largest radius on the face), span_rad}]."""
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.BRepTools import BRepTools
    from OCP.GeomAbs import GeomAbs_Cone, GeomAbs_Cylinder
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus
    out = []
    for f in faces_of(shape):
        try:
            ad = BRepAdaptor_Surface(f, True)
            t = ad.GetType()
            if t in (GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus):
                continue
            if t not in (GeomAbs_Cylinder, GeomAbs_Cone):
                c = _fitted_cylinder(f, ad)          # a B-spline (or other free-form) face that is a cylinder
                if c is not None:
                    out.append(c)
                continue
            g = ad.Cylinder() if t == GeomAbs_Cylinder else ad.Cone()
            ax = g.Axis()
            loc = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
            d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
            u0, u1, v0, v1 = BRepTools.UVBounds_s(f)
            pr = BRepLProp_SLProps(ad, 0.5 * (u0 + u1), 0.5 * (v0 + v1), 1, 1e-7)
            if not pr.IsNormalDefined():
                continue
            nv, pv = pr.Normal(), pr.Value()
            n = np.array([nv.X(), nv.Y(), nv.Z()])
            p = np.array([pv.X(), pv.Y(), pv.Z()])
            if f.Orientation() == TopAbs_REVERSED:
                n = -n
            radial = (p - loc) - d * float((p - loc) @ d)
            if float(np.linalg.norm(radial)) < 1e-9:
                continue
            if t == GeomAbs_Cylinder:
                r = float(g.Radius())
            else:
                semi = float(g.SemiAngle())
                r = float(max(g.RefRadius() + v0 * math.sin(semi), g.RefRadius() + v1 * math.sin(semi)))
            out.append({"kind": "cylinder" if t == GeomAbs_Cylinder else "cone", "point": p - radial, "dir": d,
                        "radius": r, "concave": bool(n @ radial < 0), "span_rad": float(u1 - u0)})
        except Exception:  # noqa: BLE001 - an odd face is not a feature
            continue
    return [c for c in out if c["concave"]]


def _fitted_cylinder(f, ad, n=7, tol_rel=0.03, tol_abs=0.02):
    """A free-form face that is a cylinder within max(tol_abs, tol_rel x radius) (a hole after a non-uniform scale, a
    lofted or approximated bore): the axis from the normals (they are perpendicular to it), the circle fitted in the
    plane across it. None if it is not one."""
    from OCP.BRepLProp import BRepLProp_SLProps
    from OCP.BRepTools import BRepTools
    from OCP.TopAbs import TopAbs_REVERSED
    u0, u1, v0, v1 = BRepTools.UVBounds_s(f)
    P, N = [], []
    for u in np.linspace(u0, u1, n):
        for v in np.linspace(v0, v1, n):
            pr = BRepLProp_SLProps(ad, float(u), float(v), 1, 1e-7)
            if not pr.IsNormalDefined():
                continue
            nv, pv = pr.Normal(), pr.Value()
            P.append((pv.X(), pv.Y(), pv.Z()))
            N.append((nv.X(), nv.Y(), nv.Z()))
    if len(P) < 12:
        return None
    P, N = np.asarray(P), np.asarray(N)
    if f.Orientation() == TopAbs_REVERSED:
        N = -N
    w, E = np.linalg.eigh(N.T @ N)
    if w[0] > 1e-4 * w[2] or w[1] < 0.05 * w[2]:
        return None                                  # the normals do not span a plane: not a cylinder
    d = E[:, 0]
    e1 = E[:, 2]
    e2 = np.cross(d, e1)
    x, y = P @ e1, P @ e2
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    cx, cy = sol[0], sol[1]
    r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0))
    if not r > 0:
        return None
    res = np.abs(np.hypot(x - cx, y - cy) - r)
    if float(res.max()) > max(tol_abs, tol_rel * r):
        return None
    centre = cx * e1 + cy * e2 + d * float(np.mean(P @ d))
    radial = (P - centre) - np.outer((P - centre) @ d, d)
    concave = float(np.mean(np.einsum("ij,ij->i", N, radial))) < 0
    ang = np.arctan2((P - centre) @ e2, (P - centre) @ e1)
    span = float(np.ptp(np.unwrap(np.sort(ang))))
    return {"kind": "cylinder", "point": centre, "dir": d, "radius": float(r), "concave": concave, "span_rad": span,
            "fitted": True}


def feature_items(h):
    """The graded segments of a reference feature: its steps from the opening inward (a cylinder or a cone each, with
    the radius at that segment; a cone's is its largest), else the opening radius alone."""
    p = np.asarray(h.get("point_mm", h.get("point")), float)
    d = np.asarray(h["dir"], float)
    d = d / np.linalg.norm(d)
    steps = h.get("steps") or []
    if not steps:
        return [(p, d, float(h.get("radius_mm", h.get("radius"))), "cylinder")]
    return [(p, d, float(st["radius_mm"]), st.get("kind", "cylinder")) for st in steps]


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
    p.InParallel = True
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
    """Close boundary loops of at most max_loop edges with triangle fans (the OCC mesher leaves faces it cannot mesh and
    seam gaps on some imported B-reps). Adapted from the bimanual task's cad_tree.checks.fill_small_holes."""
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


def to_manifold(V, F):
    import manifold3d as m3
    man = m3.Manifold(m3.Mesh(vert_properties=np.ascontiguousarray(V, np.float32),
                              tri_verts=np.ascontiguousarray(F, np.uint32)))
    return man if str(man.status()) == "Error.NoError" and not man.is_empty() else None


class Geo(object):
    """A solid's tessellation in its own frame: vertices, faces, manifold (or None), area, volume, bounds, and lazily the
    dense samples (with normals and a KD-tree) and the query samples used for surface distances."""

    def __init__(self, V, F, man, notes, seed):
        import trimesh
        self.V, self.F, self.man, self.notes, self.seed = V, F, man, notes, int(seed)
        self.tm = trimesh.Trimesh(V, F, process=False)
        self.area = float(self.tm.area)
        self.volume = float(man.volume()) if man is not None else abs(float(self.tm.volume))
        self.bounds = np.array([V.min(axis=0), V.max(axis=0)])
        self._dense = self._query = None

    def samples(self, n, seed):
        import trimesh
        if len(self.F) == 0 or self.area <= 0:
            return np.zeros((0, 3)), np.zeros((0, 3))
        pts, fid = trimesh.sample.sample_surface(self.tm, int(n), seed=int(seed) % (2 ** 32))
        return np.asarray(pts, float), np.asarray(self.tm.face_normals[fid], float)

    def dense(self, cfg):
        if self._dense is None:
            from scipy.spatial import cKDTree
            S = cfg["sampling"]
            n = int(min(S["dense_max"], max(S["dense_min"], S["dense_per_mm2"] * self.area)))
            P, N = self.samples(n, self.seed * 7 + 1)
            self._dense = (P, N, cKDTree(P) if len(P) else None)
        return self._dense

    def query(self, cfg):
        if self._query is None:
            S = cfg["sampling"]
            n = int(min(S["query_max"], max(S["query_min"], S["query_per_mm2"] * self.area)))
            self._query = self.samples(n, self.seed * 7 + 2)
        return self._query


def _canonical_poses(shape):
    """Rigid moves under which a solid is meshed, in order: its principal frame (centroid at the origin, inertia axes on
    x, y, z), the same turned 30 deg about z, and where it is. OCC's mesher depends on the numbers of the geometry: the
    original landing dampers mesh with faces missing at three of their four placements and cleanly in their principal
    frame, and a canonical frame makes the tessellation of congruent solids the same wherever they are placed."""
    _V, c, I = mass_props(shape)
    _w, E = np.linalg.eigh(I)
    if np.linalg.det(E) < 0:
        E[:, 0] *= -1.0
    R = E.T
    R30 = axis_angle_R([0, 0, 1], math.radians(30.0)) @ R
    return [make_T(R, -R @ c), make_T(R30, -R30 @ c), np.eye(4)]


def _moved_copy(shape, T):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    tr = gp_Trsf()
    tr.SetValues(*(float(v) for v in np.asarray(T, float)[:3, :].ravel()))
    return BRepBuilderAPI_Transform(shape, tr, True).Shape()


def mesh_solid(shape, cfg, tmpdir, seed, volume=None):
    """Geo of a solid, in the solid's own frame: the OCC tessellation (deflection / angle from config, then the retry
    list), welded and hole-filled, in each canonical pose in turn until it is a manifold whose volume is within
    volume_tol_rel of the B-rep volume; else the closest one (noted; a mesh that is no manifold uses its convex hull
    in booleans)."""
    M = cfg["mesh"]
    tries = [(M["deflection_mm"], M["angle_rad"])] + [tuple(x) for x in M.get("retry", [])]
    if volume is None:
        volume = mass_props(shape)[0]
    vtol = float(M.get("volume_tol_rel", 0.02))
    best = None
    try:
        poses = _canonical_poses(shape)
    except Exception:  # noqa: BLE001
        poses = [np.eye(4)]
    # the first setting in every pose, the retry settings in the principal frame only (a solid that no setting meshes
    # within the volume tolerance would otherwise cost every combination)
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
        man = to_manifold(V, F)
        vol = float(man.volume()) if man is not None else 0.0
        err = abs(vol / volume - 1.0) if volume > 0 else np.inf
        notes = ["filled %d hole(s)" % filled] if filled else []
        if (defl, ang) != tries[0]:
            notes.append("re-meshed at %g mm / %g rad" % (defl, ang))
        if man is not None and err <= vtol:
            return Geo(V, F, man, notes, seed)
        if best is None or err < best[0]:
            best = (err, V, F, man, notes)
    if best is None:
        return None
    err, V, F, man, notes = best
    if man is not None and err > vtol:
        notes = notes + ["the closest tessellation's volume is off by %.1f %%" % (100 * err)]
        man = None if err > 0.5 else man
    if man is None:
        notes = notes + ["the tessellation is not a closed manifold of the solid's volume: booleans use its convex hull"]
    return Geo(V, F, man, notes, seed)


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
    lo, hi = geo.bounds
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    w = apply(T, c)
    return np.concatenate([w.min(axis=0), w.max(axis=0)])


def aabb_overlap(a, b, margin=0.0):
    return all(a[i] - margin <= b[i + 3] and b[i] - margin <= a[i + 3] for i in range(3))


def common_volume(ma, mb):
    if ma is None or mb is None:
        return 0.0
    try:
        return max(0.0, float((ma ^ mb).volume()))
    except Exception:  # noqa: BLE001
        return 0.0


def surface_distances(query, samples, normals, tree, disk_mm):
    """Distance of each query point to a surface given by dense samples with normals: the distance to the tangent disk
    (radius disk_mm) of the nearest sample (from the bimanual task's cad_tree.checks)."""
    d, idx = tree.query(query)
    v = query - samples[idx]
    dn = np.abs(np.einsum("ij,ij->i", v, normals[idx]))
    dl = np.sqrt(np.maximum(0.0, d * d - dn * dn))
    return np.sqrt(dn * dn + np.maximum(0.0, dl - disk_mm) ** 2)


class Placed(object):
    """A Geo at a pose T (its frame -> the reference frame): transformed samples and manifold, computed on demand."""

    def __init__(self, geo, T, cfg):
        self.geo, self.T, self.cfg = geo, np.asarray(T, float), cfg
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
            P, N, _t = self.geo.dense(self.cfg)
            P2, N2 = apply(self.T, P), N @ self.T[:3, :3].T
            self._dense = (P2, N2, cKDTree(P2) if len(P2) else None)
        return self._dense

    def query(self):
        if self._query is None:
            P, N = self.geo.query(self.cfg)
            self._query = (apply(self.T, P), N @ self.T[:3, :3].T)
        return self._query


def one_way(a, b, disk, cap=None):
    """Distances of a's query samples to b's surface."""
    P = a.query()[0]
    D, N, tree = b.dense()
    if len(P) == 0 or tree is None:
        return np.full(max(1, len(P)), np.inf)
    return surface_distances(P, D, N, tree, disk)


def min_distance(group_a, group_b, disk, reach):
    """Smallest surface distance between two groups of Placed (query points within `reach` of the other's box only)."""
    best = np.inf
    for a in group_a:
        for b in group_b:
            if not aabb_overlap(a.box, b.box, reach):
                continue
            for x, y in ((a, b), (b, a)):
                P = x.query()[0]
                lo, hi = y.box[:3] - reach, y.box[3:] + reach
                sel = np.all((P >= lo) & (P <= hi), axis=1)
                if not sel.any():
                    continue
                D, N, tree = y.dense()
                if tree is None:
                    continue
                d = surface_distances(P[sel], D, N, tree, disk)
                best = min(best, float(d.min()))
    return best


def group_common(group_a, group_b):
    tot = 0.0
    for a in group_a:
        for b in group_b:
            if aabb_overlap(a.box, b.box):
                tot += common_volume(a.man, b.man)
    return tot


# ============================================================================ registration
def eigen(I):
    w, E = np.linalg.eigh(np.asarray(I, float))
    return w, E


def _pca_rotations(Is, Id):
    """Proper rotations taking the source's principal axes onto the destination's (inertia eigenvectors, sorted by
    eigenvalue; the 4 sign choices with det +1)."""
    _ws, Es = eigen(Is)
    _wd, Ed = eigen(Id)
    out = []
    for signs in itertools.product((1.0, -1.0), repeat=3):
        R = Ed @ np.diag(signs) @ Es.T
        if np.linalg.det(R) > 0:
            out.append(R)
    return out


def _axial_rotations(src_v, cs, Is, dst_v, cd, Id, rh_tol):
    """Rotations for a source whose inertia has two equal moments (a body of revolution or of 3+-fold symmetry): the
    distinct principal axis onto the destination's (both senses), then the turn about it that takes the source's most
    distinctive off-axis vertex onto each destination vertex of the same radius and height."""
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
        pick = cand[int(np.argmin(cnt[inv_] * 1e6 - rs[cand]))]
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


def register_exact(src, dst, tol, all_solutions=False):
    """Exact registration of two copies of one B-rep: src, dst = {"verts", "com", "I", "size"}. Proper rotations only.
    Returns (T src->dst, max vertex error) or None; with all_solutions, every distinct exact solution (a vertex set with
    a symmetry has several) as a list. The vertex sets must have the same size and map one-to-one."""
    from scipy.spatial import cKDTree
    Vs, Vd = src["verts"], dst["verts"]
    if len(Vs) != len(Vd) or len(Vs) == 0:
        return None
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
    if all_solutions:
        return sorted(good, key=lambda g: g[1])
    return min(good, key=lambda g: g[1]) if good else None


def _principal_frame(P):
    c = P.mean(axis=0)
    w, V = np.linalg.eigh(np.cov((P - c).T))
    return c, V[:, np.argsort(w)[::-1]]


def pca_starts(src, dst):
    """The 24 proper rotations mapping the source sample cloud's principal axes onto the destination's, centroids
    matched (from the bimanual task's cad_tree.checks)."""
    cs, Vs = _principal_frame(src)
    cd, Vd = _principal_frame(dst)
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                M[:, i] = s * Vd[:, p]
            R = M @ Vs.T
            if np.linalg.det(R) < 0:
                continue
            out.append(make_T(R, cd - R @ cs))
    return out


def icp(src, dst, dst_n, tree, T0, iters, trim, bound_mm=60.0):
    """Point-to-plane ICP of src onto (dst, dst_n), trimmed; returns (T, trimmed RMS)."""
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
        if w > 0.15:
            x[:3] *= 0.15 / w
        R = axis_angle_R(x[:3], float(np.linalg.norm(x[:3]))) if np.linalg.norm(x[:3]) > 1e-15 else np.eye(3)
        dT = make_T(R, c - R @ c + x[3:])
        T = dT @ T
        if float(np.linalg.norm(x)) < 1e-9:
            break
    d, _ = tree.query(apply(T, src), distance_upper_bound=bound_mm)
    d = np.where(np.isfinite(d), d, bound_mm)
    k = max(1, int(len(d) * trim))
    return make_T(orthonormal(T[:3, :3]), T[:3, 3]), float(math.sqrt(np.mean(np.sort(d)[:k] ** 2)))


def inertia_starts(src, dst, step_deg=15.0, rel=0.05):
    """Initial poses src -> dst from the exact (B-rep) inertia of both solids: the principal axes matched (the 4
    proper sign choices), and where two principal moments are near-equal (a body of revolution or of 3+-fold symmetry,
    whose in-plane axes inertia leaves arbitrary) each turned about the distinct axis in step_deg steps."""
    ws, Es = eigen(src["I"])
    wd, Ed = eigen(dst["I"])
    out = []
    for R in _pca_rotations(src["I"], dst["I"]):
        out.append(make_T(R, dst["com"] - R @ src["com"]))
    axes = []
    sc = max(abs(wd).max(), 1e-12)
    if abs(wd[1] - wd[0]) <= rel * sc:
        axes.append(Ed[:, 2])
    if abs(wd[2] - wd[1]) <= rel * sc:
        axes.append(Ed[:, 0])
    base = list(out)
    for ax in axes:
        for T0 in base:
            for k in range(1, int(round(360.0 / step_deg))):
                R = axis_angle_R(ax, math.radians(k * step_deg))
                out.append(make_T(R, dst["com"] - R @ dst["com"]) @ T0)
    return out


def register_samples(src_geo, dst_geo, cfg, src=None, dst=None, n_keep=12):
    """Rigid registration of two tessellated solids (T src->dst, trimmed RMS): coarse ICP on a subsample from the
    inertia starts when both solids' exact inertia is given (inertia_starts), else from the 24 sample-cloud principal-
    axis starts; the best n_keep refined on the full sample set."""
    A = cfg["alignment"]
    q = src_geo.query(cfg)[0]
    D, N, tree = dst_geo.dense(cfg)
    if len(q) < 10 or tree is None:
        return np.eye(4), np.inf
    starts = inertia_starts(src, dst) if (src is not None and dst is not None) else pca_starts(q, D)
    coarse = []
    sub = q[np.linspace(0, len(q) - 1, min(len(q), 1200)).astype(int)]
    for T0 in starts:
        Tk, rk = icp(sub, D, N, tree, T0, 12, A["trim"])
        coarse.append((rk, Tk))
    coarse.sort(key=lambda x: x[0])
    best = (np.inf, np.eye(4))
    for _rk, Tk in coarse[:n_keep]:
        Tf, rf = icp(q, D, N, tree, Tk, A["icp_iters"], A["trim"])
        if rf < best[0]:
            best = (rf, Tf)
    return best[1], best[0]


# ============================================================================ the reference
class TaskError(Exception):
    """The grader's own reference or the image's vendor files are unusable: a broken task, never a zero."""


class Reference(object):
    """reference.json, the manufactured parts (B-rep, tessellation, mass properties, features) and the purchased types
    (constituents read from the pinned vendor STEPs, their congruence classes, tessellations and mass properties)."""

    def __init__(self, ref_dir, purchased_root, cfg, tmpdir, log):
        self.dir, self.cfg, self.tmp, self.log = ref_dir, cfg, tmpdir, log
        with open(os.path.join(ref_dir, "reference.json")) as fh:
            self.spec = json.load(fh)
        if self.spec.get("schema") != "agipix-cad-reference/2":
            raise TaskError("reference.json schema %r" % self.spec.get("schema"))
        mats = self.spec["materials"]
        self.extra_density = float(mats[cfg["mass"]["extra_material"]]["density_g_mm3"])
        self.families = self.spec["families"]
        self.parts = {}
        seed = int(cfg["sampling"]["seed"])
        pending = self._start_purchased(purchased_root)
        for i, (tag, p) in enumerate(sorted(self.spec["parts"].items())):
            shape = read_brep(os.path.join(ref_dir, p["brep"]))
            sols = solids_of(shape)
            if len(sols) != 1:
                raise TaskError("reference part %s has %d solids" % (tag, len(sols)))
            V, c, I = mass_props(sols[0])
            geo = mesh_solid(sols[0], cfg, tmpdir, seed + 1000 + i, volume=V)
            if geo is None or geo.man is None:
                raise TaskError("reference part %s does not tessellate to a closed manifold" % tag)
            self.parts[tag] = {"tag": tag, "family": p["family"], "density": float(p["density_g_mm3"]), "shape": sols[0],
                               "V": V, "com": c, "I": I, "geo": geo, "placed": Placed(geo, np.eye(4), cfg),
                               "features": list(p.get("features", []))}
        self.tags = sorted(self.parts)
        self.family_of = {t: p["family"] for t, p in self.parts.items()}
        self.family_tags = {}
        for t, f in sorted(self.family_of.items()):
            self.family_tags.setdefault(f, []).append(t)
        self._load_purchased(pending)
        self.placements = {}
        for pl in self.spec["placements"]:
            self.placements[pl["id"]] = {"id": pl["id"], "type": pl["type"], "W": np.asarray(pl["local_to_world"], float)}
        self.rotors = self.spec.get("rotors", [])
        self.prop_types = sorted(set(r["prop_type"] for r in self.rotors))
        self.prop_motor = {r["prop"]: r["motor"] for r in self.rotors}
        self.supports = self.spec.get("supports", {})
        self.unscored = self.spec.get("unscored", [])

    # ------------------------------------------------------------------ purchased types
    def _start_purchased(self, root):
        """Check every vendor STEP against its pin and start reading them in worker processes (before any meshing in
        this process)."""
        with open(os.path.join(self.dir, "purchased_pins.json")) as fh:
            pins = json.load(fh)
        jobs = []
        for tname, t in sorted(self.spec["purchased"].items()):
            pin = pins["types"][tname]
            path = os.path.join(root, pin["step"])
            if not os.path.isfile(path):
                raise TaskError("the vendor file of %s is missing: %s" % (tname, path))
            digest = sha256_file(path)
            if digest != pin["sha256"]:
                raise TaskError("the vendor file %s does not match its pin (sha256 %s, pinned %s)" % (path, digest, pin["sha256"]))
            jobs.append((tname, path))
        return pins, StepReader(jobs, self.tmp, int(self.cfg.get("step_workers", 4)))

    def _load_purchased(self, pending):
        pins, reader = pending
        breps = reader.result()
        self.types = {}
        seed = int(self.cfg["sampling"]["seed"])
        n = 0
        for tname, t in sorted(self.spec["purchased"].items()):
            pin = pins["types"][tname]
            files = breps[tname]
            if len(files) != len(pin["solids"]):
                raise TaskError("%s: the vendor STEP has %d solids, the pin %d" % (tname, len(files), len(pin["solids"])))
            byid = {c["id"]: c for c in t["constituents"]}
            cons = [None] * len(t["constituents"])
            order = [c["id"] for c in t["constituents"]]
            for k, (f, ps) in enumerate(zip(files, pin["solids"])):
                s = solids_of(read_brep(f))[0]
                V, c, I = mass_props(s)
                if abs(V - ps["volume_mm3"]) > 1e-5 * max(1.0, abs(ps["volume_mm3"])) or \
                        np.linalg.norm(c - np.asarray(ps["centroid_mm"])) > 1e-3:
                    raise TaskError("%s solid %d does not match its pin (volume %.6f vs %.6f)" % (tname, k, V, ps["volume_mm3"]))
                cid = ps["constituent"]
                n += 1
                geo = mesh_solid(s, self.cfg, self.tmp, seed + 5000 + n, volume=V)
                verts = unique_vertices(s)
                ext = verts.max(axis=0) - verts.min(axis=0) if len(verts) else np.zeros(3)
                cons[order.index(cid)] = {"id": cid, "type": tname, "k": order.index(cid), "V": V, "com": c, "I": I,
                                          "verts": verts, "size": float(np.linalg.norm(ext)), "geo": geo,
                                          "role": byid[cid].get("source_role")}
            if any(c is None for c in cons):
                raise TaskError("%s: a constituent has no vendor solid" % tname)
            vol = sum(c["V"] for c in cons)
            cen = sum(c["com"] * c["V"] for c in cons) / vol
            anchor = [c["id"] for c in cons].index(t["anchor"])
            sym = t.get("symmetry") or {}
            self.types[tname] = {"name": tname, "pn": t.get("part_number"), "mass_g": float(t["mass_g"]),
                                 "density": float(t["density_g_mm3"]), "cons": cons, "anchor": anchor, "V": vol,
                                 "centroid": cen, "sym_axis": np.asarray(sym.get("axis_local", [0, 0, 1]), float),
                                 "symmetry": sym}
        self._classes()

    def _classes(self):
        """Congruence classes of all constituents (exact registration on their vertex sets): the representative is the
        first member; M = the member's pose relative to the representative (rep local -> member local)."""
        tol = float(self.cfg["recognition"]["exact_tol_mm"])
        self.classes = []
        for tname in sorted(self.types):
            for c in self.types[tname]["cons"]:
                placed = False
                for cl in self.classes:
                    rep = cl["rep"]
                    if len(rep["verts"]) != len(c["verts"]) or abs(rep["V"] - c["V"]) > 1e-6 * max(1.0, rep["V"]):
                        continue
                    got = register_exact(rep, c, tol)
                    if got is not None:
                        cl["members"].append((c, got[0]))
                        c["cls"] = cl["id"]
                        placed = True
                        break
                if not placed:
                    cl = {"id": len(self.classes), "rep": c, "members": [(c, np.eye(4))]}
                    c["cls"] = cl["id"]
                    self.classes.append(cl)
        for cl in self.classes:
            cl["types"] = sorted(set(m[0]["type"] for m in cl["members"]))

    # ------------------------------------------------------------------ reference geometry in the world frame
    def placement_items(self, pid):
        """Placed constituents of a reference placement (cached)."""
        if not hasattr(self, "_items"):
            self._items = {}
        if pid not in self._items:
            pl = self.placements[pid]
            typ = self.types[pl["type"]]
            self._items[pid] = [Placed(c["geo"], pl["W"], self.cfg) for c in typ["cons"] if c["geo"] is not None]
        return self._items[pid]

    def module_samples(self, tname):
        """(query points, dense points, dense normals, KD-tree, size) of a purchased type's whole module, local frame."""
        if not hasattr(self, "_modules"):
            self._modules = {}
        if tname not in self._modules:
            from scipy.spatial import cKDTree
            cons = [c for c in self.types[tname]["cons"] if c["geo"] is not None]
            q = np.vstack([c["geo"].query(self.cfg)[0] for c in cons])
            D = np.vstack([c["geo"].dense(self.cfg)[0] for c in cons])
            N = np.vstack([c["geo"].dense(self.cfg)[1] for c in cons])
            size = float(np.linalg.norm(D.max(axis=0) - D.min(axis=0))) if len(D) else 0.0
            tree = cKDTree(D)
            base = float(np.quantile(surface_distances(q, D, N, tree, self.cfg["sampling"]["disk_mm"]), 0.9)) if len(q) else 0.0
            self._modules[tname] = (q, D, N, tree, size, base)
        return self._modules[tname]


def _step_worker(job):
    tname, path, out_dir = job
    from OCP.BRepTools import BRepTools
    from OCP.IFSelect import IFSelect_RetDone
    from OCP.STEPControl import STEPControl_Reader
    r = STEPControl_Reader()
    if r.ReadFile(path) != IFSelect_RetDone:
        raise IOError("cannot read %s" % path)
    r.TransferRoots()
    shape = r.OneShape()
    out = []
    for k, s in enumerate(solids_of(shape)):
        p = os.path.join(out_dir, "%s_%03d.brep" % (tname, k))
        BRepTools.Write_s(s, p)
        out.append(p)
    return tname, out


class StepReader(object):
    """Read the vendor STEPs (the slow part of loading the reference) in spawned worker processes, started at once;
    each worker writes the STEP's solids, in the reader's order, as BREP files. result() = {type: [brep paths]}."""

    def __init__(self, jobs, tmpdir, workers):
        out_dir = os.path.join(tmpdir, "vendor")
        os.makedirs(out_dir, exist_ok=True)
        self.jobs = sorted([(t, p, out_dir) for t, p in jobs], key=lambda j: -os.path.getsize(j[1]))  # biggest first
        self.ex = self.futs = None
        if workers > 1 and len(self.jobs) > 1:
            import concurrent.futures as cf
            import multiprocessing as mp
            self.ex = cf.ProcessPoolExecutor(max_workers=min(workers, len(self.jobs)), mp_context=mp.get_context("spawn"))
            self.futs = [self.ex.submit(_step_worker, j) for j in self.jobs]

    def result(self):
        if self.ex is None:
            return dict(_step_worker(j) for j in self.jobs)
        try:
            return dict(f.result() for f in self.futs)
        finally:
            self.ex.shutdown()


# ============================================================================ the submission
class Submission(object):
    """The extracted document: every solid with its tag (if any), component, mass properties and vertices.

    Only an unreadable document, a document without any solid and a missing or unusable P01 stop the grade (ok False,
    with the reason). Everything else is local, with a diagnostic and a statistic:
      - a tagged leaf that is not a PartDesign Body of exactly one valid closed solid: the tag gets no credit (its
        instance is unmatched); its valid solids stay in the unit as untagged extra solids (mass, fit, clearance);
      - a tag on more than one leaf (a Body reused through several links included): none of them is matched; their
        solids are extra solids;
      - an untagged leaf with invalid solids, open shells or loose faces, or a solid that cannot be read: the bad
        pieces are left out (counted), its valid solids stay;
      - no Assembly in the document: its visible solids are graded (the extractor read them), without joint credit.
    A P01 that is a single valid closed solid but no Body is still used for the alignment (its instance gets no
    credit); a P01 on two leaves, with several solids or none valid is unusable."""

    def __init__(self, ex_dir, ref, cfg, tmpdir, diags):
        self.ok = False
        self.solids = []
        self.cfg, self.tmp = cfg, tmpdir
        self.connected = set()
        self.grounded = []
        self.p01 = None
        self.stats = st = {"tags_invalid": 0, "tags_duplicated": 0, "ignored_non_solid_leaves": 0,
                           "ignored_bad_solids": 0, "ignored_unreadable_leaves": 0, "no_assembly": 0}
        path = os.path.join(ex_dir, "model.json")
        if not os.path.isfile(path):
            diags.append("No readable FreeCAD document.")
            return
        try:
            with open(path) as fh:
                model = json.load(fh)
        except ValueError as exc:
            raise TaskError("the extraction's model.json is not JSON: %s" % exc)
        if not isinstance(model, dict) or not str(model.get("format", "")).startswith("uav-fcstd-extract/"):
            raise TaskError("the extraction's model.json has an unknown format")
        if model.get("extractor_failure"):
            raise TaskError("the FreeCAD reader failed: " + " | ".join(str(e) for e in model.get("errors", []))[-1500:])
        self.model = model
        if model.get("errors"):
            diags.extend("The document cannot be read: %s" % str(e)[:300] for e in model["errors"][:5])
            return
        if model.get("no_assembly"):
            st["no_assembly"] = 1
            diags.append("No Assembly in the document: its visible solids are graded, without joint credit.")
        G = cfg["gates"]
        ex_real = os.path.realpath(ex_dir)
        seen = set()
        tagged = {}          # tag -> [(leaf, comp name, solid records, single valid Body?)]
        untagged = []        # (leaf, comp name, solid record)

        def note(msg):
            if sum(1 for d in diags if d.startswith("Local:")) < 25:
                diags.append("Local: " + msg)

        for comp in model.get("components", []):
            for leaf in comp.get("leaves", []):
                label = (leaf.get("label") or "").strip()
                tag = label if label in ref.parts else None
                where = "%s (%s)" % (label or leaf.get("target"), leaf.get("type"))
                if leaf.get("error"):
                    st["ignored_unreadable_leaves"] += 1
                    note("the shape of %s cannot be read (%s): left out" % (where, str(leaf["error"])[:120]))
                    if tag:
                        tagged.setdefault(tag, []).append((leaf, comp["name"], [], False))
                    continue
                key = (leaf.get("target"), tuple(np.round(np.asarray(leaf.get("placement", np.eye(4)), float).ravel(), 6)))
                if key in seen:
                    # the same object reached twice at the same place (a folder and a part listing it): one physical solid
                    st["duplicate_leaves_ignored"] = st.get("duplicate_leaves_ignored", 0) + 1
                    continue
                seen.add(key)
                sols = leaf.get("solids", [])
                good = [x for x in sols if x.get("valid") and x.get("closed")]
                free = int(leaf.get("n_free_faces", 0) or 0)
                if tag is not None:
                    single = (leaf.get("type") == "PartDesign::Body" and len(sols) == 1 and len(good) == 1 and
                              bool(leaf.get("valid")) and free == 0)
                    tagged.setdefault(tag, []).append((leaf, comp["name"], good, single))
                    continue
                if not sols:
                    if free:
                        st["ignored_non_solid_leaves"] += 1
                        note("%s has open faces and no solid: left out" % where)
                    else:
                        st["ignored_non_solid_leaves"] += 1      # a sketch, a datum, a wire
                    continue
                if free or len(good) < len(sols) or not leaf.get("valid"):
                    st["ignored_bad_solids"] += (len(sols) - len(good)) + (1 if free else 0)
                    note("%s has %d invalid or open solid(s)%s: those are left out" % (
                        where, len(sols) - len(good), " and loose faces" if free else ""))
                for x in good:
                    untagged.append((leaf, comp["name"], x))

        def add(leaf, comp_name, x, tag, demoted=None):
            p = os.path.realpath(os.path.join(ex_dir, x["brep"]))
            if os.path.commonpath([ex_real, p]) != ex_real:
                raise TaskError("the extractor returned an escaping shape path")
            rec = {"sid": len(self.solids), "tag": tag, "label": (leaf.get("label") or "").strip(), "component": comp_name,
                   "leaf_type": leaf.get("type"), "file": p}
            if demoted:
                rec["demoted"] = demoted
            self.solids.append(rec)
            return rec

        p01_entries = tagged.get("P01", [])
        for tag in sorted(tagged):
            entries = tagged[tag]
            if len(entries) > 1:
                st["tags_duplicated"] += 1
                note("tag %s is on %d leaves: none of them is matched; their solids count as extra solids" % (tag, len(entries)))
                for leaf, cname, good, _single in entries:
                    for x in good:
                        add(leaf, cname, x, None, demoted="duplicate tag " + tag)
                continue
            leaf, cname, good, single = entries[0]
            if single:
                add(leaf, cname, good[0], tag)
                continue
            st["tags_invalid"] += 1
            n = len(leaf.get("solids", []))
            note("%s is a %s with %d solid(s), %d valid and closed: not a PartDesign Body of exactly one valid closed "
                 "solid, so the tag gets no credit; its valid solids count as extra solids" % (tag, leaf.get("type"), n, len(good)))
            for x in good:
                rec = add(leaf, cname, x, None, demoted="invalid tagged leaf " + tag)
                if tag == "P01" and n == 1 and len(good) == 1:
                    self.p01 = rec              # a lone valid P01 solid still serves the alignment
        for leaf, cname, x in untagged:
            add(leaf, cname, x, None)
        if len(self.solids) > int(G["max_solids"]):
            diags.append("Too many solids in the unit (%d)." % len(self.solids))
            self.solids = []
            return
        keep = []
        for s in self.solids:
            try:
                shape = read_brep(s["file"])
                sols = solids_of(shape)
                if len(sols) != 1:
                    raise ValueError("%d solids in one solid file" % len(sols))
                s["shape"] = sols[0]
                if G.get("recheck_validity", True) and not brep_valid(sols[0]):
                    raise ValueError("BRepCheck reports an invalid solid")
                V, c, I = mass_props(sols[0])
                if not V > float(G["min_volume_mm3"]):
                    raise ValueError("no volume")
                s["V"], s["com"], s["I"] = V, c, I
                if s["tag"] is None:
                    s["verts"] = unique_vertices(sols[0])
                    ext = s["verts"].max(axis=0) - s["verts"].min(axis=0) if len(s["verts"]) else np.zeros(3)
                    s["size"] = float(np.linalg.norm(ext))
            except Exception as exc:  # noqa: BLE001 - a solid that does not load is left out
                st["ignored_bad_solids"] += 1
                note("a solid of %s cannot be used (%s): left out" % (s["label"] or s["component"], str(exc)[:100]))
                if s["tag"] is not None:
                    st["tags_invalid"] += 1
                if s is self.p01:
                    self.p01 = None
                continue
            keep.append(s)
        for k, s in enumerate(keep):
            s["sid"] = k
        self.solids = keep
        if not self.solids:
            diags.append("The %s holds no solid." % ("document" if model.get("no_assembly") else "Assembly"))
            return
        tagged_p01 = [s for s in self.solids if s["tag"] == "P01"]
        if tagged_p01:
            self.p01 = tagged_p01[0]
        if self.p01 is None:
            if len(p01_entries) > 1:
                diags.append("P01 is on %d leaves: no unique base for the alignment." % len(p01_entries))
            elif p01_entries:
                diags.append("P01 is not a single valid closed solid: it cannot serve the alignment.")
            else:
                diags.append("The tagged base P01 is required for the alignment.")
            return
        self.connected = fixed_connected(model)
        self.grounded = [g for g in model.get("grounded", []) if g]
        self.ok = True

    def geo(self, s, seed):
        if "geo" not in s:
            s["geo"] = mesh_solid(s["shape"], self.cfg, self.tmp, seed, volume=s.get("V"))
        return s["geo"]


def fixed_connected(model):
    """Components reached from a grounded component through unsuppressed Fixed joints."""
    graph = {c["name"]: set() for c in model.get("components", [])}
    for j in model.get("joints", []):
        if j.get("suppressed") or j.get("error") or str(j.get("type", "")).lower() != "fixed":
            continue
        a, b = j.get("part1"), j.get("part2")
        if a in graph and b in graph and a != b:
            graph[a].add(b)
            graph[b].add(a)
    reached = set(g for g in model.get("grounded", []) if g in graph)
    todo = list(reached)
    while todo:
        for nxt in graph[todo.pop()] - reached:
            reached.add(nxt)
            todo.append(nxt)
    return reached


# ============================================================================ purchased recognition
def recognise(sub, ref, cfg, stats):
    """Match every untagged solid to a constituent class (module docstring, rule 3). Sets s["match"] =
    {"cls", "T" (class representative, local -> submission frame), "err", "mode"}."""
    R = cfg["recognition"]
    vol_rel = float(R["volume_rel"])
    tol = float(R["exact_tol_mm"])
    n_exact = n_approx = 0
    for s in sub.solids:
        if s["tag"] is not None or s.get("demoted"):
            continue
        try:
            got = _recognise_one(s, sub, ref, cfg, R, vol_rel, tol)
        except Exception as exc:  # noqa: BLE001 - this solid stays unrecognised (an extra solid)
            item_error(stats, sub.diags_ref, "recognition of a solid of %s" % (s["label"] or s["component"]), exc)
            got = None
        if got is not None:
            s["match"] = got
            if got["mode"] == "exact":
                n_exact += 1
            else:
                n_approx += 1
    stats["purchased_solids_exact"] = n_exact
    stats["purchased_solids_approx"] = n_approx


def item_error(stats, diags, what, exc):
    """A grading item that raised: it scores 0; counted (grader_item_errors) and reported."""
    stats["grader_item_errors"] = stats.get("grader_item_errors", 0) + 1
    if diags is not None and sum(1 for d in diags if d.startswith("Item error:")) < 20:
        diags.append("Item error: %s could not be evaluated (%s: %s); it scores 0" % (what, type(exc).__name__, str(exc)[:160]))


def _recognise_one(s, sub, ref, cfg, R, vol_rel, tol):
    """The constituent class one untagged solid matches (exact on its vertex set, else sampled), or None."""
    cands = [cl for cl in ref.classes if abs(s["V"] / cl["rep"]["V"] - 1.0) <= vol_rel]
    cands.sort(key=lambda cl: abs(s["V"] / cl["rep"]["V"] - 1.0))
    got = None
    for cl in cands:
        r = register_exact(cl["rep"], s, tol, all_solutions=True)
        if r:
            got = {"cls": cl["id"], "T": r[0][0], "err": r[0][1], "mode": "exact", "alt": [g[0] for g in r]}
            break
    if got is None and cands:
        for cl in cands[:int(R["approx_candidates"])]:
            rep = cl["rep"]
            if rep["geo"] is None:
                continue
            g = sub.geo(s, int(cfg["sampling"]["seed"]) + 300000 + s["sid"])
            if g is None:
                continue
            T, _rms = register_samples(rep["geo"], g, cfg, src=rep, dst=s)
            a = Placed(rep["geo"], T, cfg)
            b = Placed(g, np.eye(4), cfg)
            disk = cfg["sampling"]["disk_mm"]
            d = np.r_[one_way(a, b, disk), one_way(b, a, disk)]
            p95 = float(np.quantile(d, 0.95)) if len(d) else np.inf
            if "base95" not in rep:      # the representative's own samples against themselves (sampling noise)
                pr = Placed(rep["geo"], np.eye(4), cfg)
                rep["base95"] = float(np.quantile(one_way(pr, pr, disk), 0.95))
            lim = rep["base95"] + max(float(R["approx_p95_mm"][0]), float(R["approx_p95_mm"][1]) * rep["size"])
            if p95 <= lim:
                got = {"cls": cl["id"], "T": T, "err": p95, "mode": "approx", "alt": [T]}
                break
    return got


def form_instances(sub, ref, cfg, stats):
    """Module instances (rule 3): seeded by anchor matches, the other constituents attached one-to-one by predicted
    centroid. Returns [{"type", "W" (local -> submission frame), "slots": {k: sid}, "completeness"}]; matched solids
    left over are marked s["extra_cls"]."""
    attach = float(cfg["recognition"]["attach_mm"])
    by_cls = {}
    for s in sub.solids:
        if "match" in s:
            by_cls.setdefault(s["match"]["cls"], []).append(s)
    claimed = set()
    inst = []

    def n_attachable(typ, W):
        n = 0
        for c in typ["cons"]:
            p = apply(W, c["com"])
            if any(np.linalg.norm(p - o["com"]) <= attach for o in by_cls.get(c["cls"], [])):
                n += 1
        return n

    for tname in sorted(ref.types):
        typ = ref.types[tname]
        a = typ["cons"][typ["anchor"]]
        cl = ref.classes[a["cls"]]
        M = dict((id(m[0]), m[1]) for m in cl["members"])[id(a)]
        for s in by_cls.get(a["cls"], []):
            if s["sid"] in claimed:
                continue
            Ws = [T @ inv(M) for T in s["match"].get("alt", [s["match"]["T"]])]
            if len(typ["cons"]) > 1 and len(Ws) > 1:
                Ws.sort(key=lambda W: -n_attachable(typ, W))
            inst.append({"type": tname, "W": Ws[0], "slots": {typ["anchor"]: s["sid"]}})
            claimed.add(s["sid"])
    sol = {s["sid"]: s for s in sub.solids}
    for cl in ref.classes:
        slots = []
        for i, it in enumerate(inst):
            typ = ref.types[it["type"]]
            for c, _M in cl["members"]:
                if c["type"] == it["type"] and c["k"] not in it["slots"]:
                    slots.append((i, c["k"], apply(it["W"], c["com"])))
        cands = [s for s in by_cls.get(cl["id"], []) if s["sid"] not in claimed]
        if not slots or not cands:
            continue
        cost = np.array([[float(np.linalg.norm(p - s["com"])) for s in cands] for (_i, _k, p) in slots])
        for r, c in hungarian(cost):
            if cost[r, c] <= attach:
                i, k, _p = slots[r]
                inst[i]["slots"][k] = cands[c]["sid"]
                claimed.add(cands[c]["sid"])
    for it in inst:
        it["completeness"] = len(it["slots"]) / float(len(ref.types[it["type"]]["cons"]))
    for s in sub.solids:
        if "match" in s and s["sid"] not in claimed:
            s["extra_cls"] = s["match"]["cls"]
    stats["purchased_instances_found"] = len(inst)
    stats["purchased_solids_unattached"] = sum(1 for s in sub.solids if "extra_cls" in s)
    return inst


# ============================================================================ alignment
def align(sub, ref, inst, cfg, stats, diags):
    """Rigid alignment of the submission onto the reference from P01 (module docstring, rule 2): T submission ->
    reference."""
    A = cfg["alignment"]
    seed = int(cfg["sampling"]["seed"])
    p01 = sub.p01
    g = sub.geo(p01, seed + 11)
    if g is None:
        diags.append("P01 cannot be tessellated.")
        return None
    rp = ref.parts["P01"]
    D, N, tree = rp["geo"].dense(cfg)
    q = g.query(cfg)[0]
    src = q[np.linspace(0, len(q) - 1, min(len(q), int(A["n_fine"]))).astype(int)]
    sub_src = q[np.linspace(0, len(q) - 1, min(len(q), int(A["n_coarse"]))).astype(int)]
    coarse = []
    for T0 in pca_starts(q, D):
        Tk, rk = icp(sub_src, D, N, tree, T0, 15, A["trim"])
        coarse.append((rk, Tk))
    coarse.sort(key=lambda x: x[0])
    cands = []
    for _rk, Tk in coarse[:int(A["refine_candidates"])]:
        Tf, rf = icp(src, D, N, tree, Tk, A["icp_iters"], A["trim"])
        if all(rot_angle_deg(Tf[:3, :3] @ c[1][:3, :3].T) > 0.5 or np.linalg.norm(Tf[:3, 3] - c[1][:3, 3]) > 0.5 for c in cands):
            cands.append((rf, Tf))
    cands.sort(key=lambda x: x[0])
    best_r = cands[0][0]
    band_r = best_r + max(float(A["tie_abs_mm"]), float(A["tie_rel"]) * best_r)
    tied = [c for c in cands if c[0] <= band_r]
    stats["alignment_candidates_tied"] = len(tied)

    fam_cent = {}
    for t, p in ref.parts.items():
        fam_cent.setdefault(p["family"], []).append(p["com"])
    type_cent = {}
    for pid, pl in ref.placements.items():
        type_cent.setdefault(pl["type"], []).append(apply(pl["W"], ref.types[pl["type"]]["centroid"]))
    cap = float(A["assembly_cap_mm"])

    def assembly_score(T):
        d = []
        for s in sub.solids:
            if s["tag"] is None or s["tag"] == "P01":
                continue
            c = apply(T, s["com"])
            d.append(min(cap, min(float(np.linalg.norm(c - q)) for q in fam_cent[ref.family_of[s["tag"]]])))
        for it in inst:
            c = apply(T, apply(it["W"], ref.types[it["type"]]["centroid"]))
            d.append(min(cap, min(float(np.linalg.norm(c - q)) for q in type_cent.get(it["type"], [c + cap]))))
        return float(np.mean(d)) if d else 0.0

    scored = sorted(((assembly_score(T), r, i, T) for i, (r, T) in enumerate(tied)), key=lambda x: (round(x[0], 6), x[1], x[2]))
    sa, r, _i, T = scored[0]
    stats["alignment_residual_mm"] = r
    stats["alignment_assembly_mm"] = sa
    # exact snap: an unchanged reference P01 (its vertex set) aligns by Kabsch on the vertices
    from scipy.spatial import cKDTree
    vs = p01["verts"] if "verts" in p01 else unique_vertices(p01["shape"])
    vr = unique_vertices(rp["shape"])
    stats["alignment_snapped"] = 0.0
    if len(vs) == len(vr) and len(vs) >= 4:
        d, idx = cKDTree(vr).query(apply(T, vs))
        if float(d.max()) <= float(A["exact_snap_mm"]) and len(np.unique(idx)) == len(idx):
            T = kabsch(vs, vr[idx])
            stats["alignment_snapped"] = 1.0
            stats["alignment_residual_mm"] = float(cKDTree(vr).query(apply(T, vs))[0].max())
    return T


# ============================================================================ the grade
def grade(ex_dir, ref_dir, purchased_root, cfg, log=None, ref=None):
    """The grade of one extraction (module docstring). `ref`: an already loaded Reference (a development batch run
    grades many documents against one; the verify stage loads it once per document anyway)."""
    t0 = time.time()
    log = log or (lambda m: print("[grade] %s" % m, file=sys.stderr, flush=True))
    tmp = tempfile.mkdtemp(prefix="uav_grade_")
    res = {"ok": True, "design_score": 0.0, "criteria": {k: 0.0 for k in CRITERIA}, "checks": {"cad_valid": 0.0},
           "statistics": {}, "parts": {}, "purchased": {}, "rotors": {}, "diagnostics": []}
    stats, diags = res["statistics"], res["diagnostics"]
    stats["grader_item_errors"] = 0
    if ref is None:
        ref = Reference(ref_dir, purchased_root, cfg, tmp, log)
    stats["t_reference_s"] = time.time() - t0
    log("reference loaded (%.1f s): %d parts, %d purchased types, %d classes" % (
        stats["t_reference_s"], len(ref.parts), len(ref.types), len(ref.classes)))
    sub = Submission(ex_dir, ref, cfg, tmp, diags)
    sub.diags_ref = diags
    stats.update(sub.stats)
    if not sub.ok:
        return res
    res["checks"]["cad_valid"] = 1.0
    stats["n_solids"] = len(sub.solids)
    stats["n_tagged"] = sum(1 for s in sub.solids if s["tag"])
    t1 = time.time()
    recognise(sub, ref, cfg, stats)
    inst = form_instances(sub, ref, cfg, stats)
    stats["t_recognition_s"] = time.time() - t1
    t1 = time.time()
    try:
        A = align(sub, ref, inst, cfg, stats, diags)
    except Exception as exc:  # noqa: BLE001 - the submitted P01 defeats the alignment: unusable, the document scores 0
        item_error(stats, diags, "the alignment on P01", exc)
        A = None
    stats["t_alignment_s"] = time.time() - t1
    if A is None:
        diags.append("P01 is unusable for the alignment: the document scores 0.")
        res["checks"]["cad_valid"] = 0.0
        return res
    log("recognised %d module instances; alignment residual %.4f mm" % (len(inst), stats["alignment_residual_mm"]))
    t1 = time.time()
    G = Grading(sub, ref, inst, A, cfg, stats, diags, res)
    G.run()
    stats["t_criteria_s"] = time.time() - t1
    stats["t_total_s"] = time.time() - t0
    return res


class Grading(object):
    def __init__(self, sub, ref, inst, A, cfg, stats, diags, res):
        self.sub, self.ref, self.inst, self.A, self.cfg = sub, ref, inst, A, cfg
        self.stats, self.diags, self.res = stats, diags, res
        self.seed = int(cfg["sampling"]["seed"])
        self.sol = {s["sid"]: s for s in sub.solids}

    # ------------------------------------------------------------------ assignments
    def assign_parts(self):
        """Tagged parts to reference instances, one-to-one by installed position inside each family."""
        self.part_of = {}          # reference tag -> submission solid
        for fam, rtags in sorted(self.ref.family_tags.items()):
            agents = [s for s in self.sub.solids if s["tag"] in rtags]
            if not agents:
                continue
            cost = np.array([[float(np.linalg.norm(apply(self.A, s["com"]) - self.ref.parts[t]["com"])) for s in agents]
                             for t in rtags])
            for r, c in hungarian(cost):
                self.part_of[rtags[r]] = agents[c]
        self.stats["tags_reassigned"] = sum(1 for t, s in self.part_of.items() if s["tag"] != t)

    def assign_instances(self):
        """Module instances to reference placements, one-to-one per type by module centroid."""
        self.inst_of = {}
        for tname, typ in sorted(self.ref.types.items()):
            pids = sorted(p for p, pl in self.ref.placements.items() if pl["type"] == tname)
            mine = [it for it in self.inst if it["type"] == tname]
            if not pids or not mine:
                continue
            c = typ["centroid"]
            cost = np.array([[float(np.linalg.norm(apply(self.A @ it["W"], c) - apply(self.ref.placements[p]["W"], c)))
                              for it in mine] for p in pids])
            for r, cc in hungarian(cost):
                self.inst_of[pids[r]] = mine[cc]
        self.stats["purchased_instances_assigned"] = len(self.inst_of)

    # ------------------------------------------------------------------ geometry of the submission (reference frame)
    def part_placed(self, s):
        if "placed" not in s:
            try:
                g = self.sub.geo(s, self.seed + 20000 + s["sid"])
            except Exception as exc:  # noqa: BLE001 - a solid that cannot be tessellated takes no part in the checks
                item_error(self.stats, self.diags, "the tessellation of a solid of %s" % (s["label"] or s["component"]), exc)
                g = None
            s["placed"] = Placed(g, self.A, self.cfg) if g is not None else None
        return s["placed"]

    def slot_placed(self, sid):
        """A recognised solid as the grader's copy of its constituent class at the registered pose."""
        s = self.sol[sid]
        if "placed" not in s:
            cl = self.ref.classes[s["match"]["cls"]]
            s["placed"] = Placed(cl["rep"]["geo"], self.A @ s["match"]["T"], self.cfg) if cl["rep"]["geo"] is not None else None
        return s["placed"]

    def instance_items(self, it):
        return [p for p in (self.slot_placed(sid) for sid in sorted(it["slots"].values())) if p is not None]

    def extra_items(self):
        """Untagged solids in no module instance: recognised leftovers (the grader's copy) and unrecognised solids."""
        out = []
        for s in self.sub.solids:
            if s["tag"] is not None:
                continue
            if "extra_cls" in s:
                p = self.slot_placed(s["sid"])
            elif "match" not in s:
                p = self.part_placed(s)
            else:
                continue
            if p is not None:
                out.append((s, p))
        return out

    # ------------------------------------------------------------------ criteria
    def run(self):
        cfg, res = self.cfg, self.res
        self.assign_parts()
        self.assign_instances()
        t = time.time()
        self.shape()
        self.stats["t_shape_s"] = time.time() - t
        t = time.time()
        floor = self.floor
        crit = res["criteria"]
        crit["features"] = floor * self.features()
        self.stats["t_features_s"] = time.time() - t
        t = time.time()
        crit["installation"] = floor * self.installation()
        crit["mass"] = floor * self.mass()
        self.stats["t_install_mass_s"] = time.time() - t
        t = time.time()
        crit["fit"] = floor * self.fit()
        self.stats["t_fit_s"] = time.time() - t
        t = time.time()
        crit["clearance"] = floor * self.clearance()
        self.stats["t_clearance_s"] = time.time() - t
        for k in CRITERIA:
            crit[k] = finite(crit[k])
        res["design_score"] = float(sum(cfg["weights"][k] * crit[k] for k in CRITERIA))

    def shape(self):
        S = self.cfg["shape"]
        disk = self.cfg["sampling"]["disk_mm"]
        scores = {}
        for tag in self.ref.tags:
            rp = self.ref.parts[tag]
            s = self.part_of.get(tag)
            rec = {"family": rp["family"], "present": s is not None}
            if s is None:
                scores[tag] = 0.0
                rec.update(shape=0.0, iou=0.0)
                self.res["parts"][tag] = rec
                continue
            try:
                a = self.part_placed(s)
                b = rp["placed"]
                if a is None:
                    scores[tag] = 0.0
                    rec.update(shape=0.0, iou=0.0, note="the part cannot be tessellated")
                    self.res["parts"][tag] = rec
                    continue
                inter = common_volume(a.man, b.man) if aabb_overlap(a.box, b.box) else 0.0
                iou = inter / max(a.volume + b.volume - inter, 1e-9)
                d = np.r_[one_way(a, b, disk), one_way(b, a, disk)]
                p90 = float(np.quantile(d, 0.9)) if len(d) else np.inf
                sc = float(S["w_surface"]) * band(p90, S["surface_p90_mm"]) + float(S["w_iou"]) * rising(iou, S["iou"])
                rec.update(shape=sc, iou=float(iou), surface_p90_mm=finite(p90, 1e9), tag=s["tag"],
                           offset_mm=float(np.linalg.norm(apply(self.A, s["com"]) - rp["com"])))
                if a.geo.notes:
                    rec["mesh_notes"] = a.geo.notes
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the shape of %s" % tag, exc)
                sc = 0.0
                rec.update(shape=0.0, iou=0.0, error=str(exc)[:200])
            scores[tag] = sc
            self.res["parts"][tag] = rec
        self.shape_scores = scores
        self.res["criteria"]["shape"] = float(np.mean([scores[t] for t in self.ref.tags]))
        F = self.cfg["floor"]
        q = self.res["criteria"]["shape"]
        self.floor = rising(q, F["ramp"])
        self.stats["shape_floor"] = self.floor

    def features(self):
        Fc = self.cfg["features"]
        kinds = set(Fc.get("kinds", ["hole", "partial"]))
        per_part = []
        n_holes = n_found = 0
        for tag in self.ref.tags:
            holes = [h for h in self.ref.parts[tag]["features"] if h.get("kind", "hole") in kinds]
            if not holes:
                continue
            s = self.part_of.get(tag)
            if s is None:
                per_part.append(0.0)
                self.res["parts"][tag]["features"] = 0.0
                n_holes += len(holes)
                continue
            try:
                ps = self._features_of(tag, holes, s, Fc)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the features of %s" % tag, exc)
                ps = (0.0, len(holes), 0)
            per_part.append(ps[0])
            n_holes += ps[1]
            n_found += ps[2]
            self.res["parts"][tag]["features"] = ps[0]
        self.stats["holes_total"] = n_holes
        self.stats["holes_found"] = n_found
        return float(np.mean(per_part)) if per_part else 0.0

    def _features_of(self, tag, holes, s, Fc):
        """(score, holes, holes found) of one part's features."""
        if "axial" not in s:
            s["axial"] = concave_axial_faces(s["shape"])
        R = self.A[:3, :3]
        faces = [{"kind": c["kind"], "point": apply(self.A, c["point"]), "dir": R @ c["dir"], "radius": c["radius"]}
                 for c in s["axial"]]
        items = []
        for h in holes:
            for (p, d, r, kind) in feature_items(h):
                items.append((h["id"], p, d, r, kind))
        sc = np.zeros(len(items))
        if faces:
            M = np.zeros((len(items), len(faces)))
            for i, (_hid, p, d, r, kind) in enumerate(items):
                for j, c in enumerate(faces):
                    if c["kind"] != kind:
                        continue
                    ang = math.degrees(math.acos(min(1.0, abs(float(d @ c["dir"])))))
                    v = p - c["point"]
                    off = float(np.linalg.norm(v - c["dir"] * float(v @ c["dir"])))
                    M[i, j] = band(off, Fc["offset_mm"]) * band(c["radius"] - r, Fc["radius_mm"]) * band(ang, Fc["angle_deg"])
            for i, j in hungarian(-M):
                sc[i] = M[i, j]
        by_hole = {}
        for (hid, _p, _d, _r, _k), v in zip(items, sc):
            by_hole.setdefault(hid, []).append(v)
        hole_scores = [float(np.mean(v)) for v in by_hole.values()]
        return float(np.mean(hole_scores)), len(hole_scores), sum(1 for v in hole_scores if v > 0.5)

    def _orientation_error(self, Wa, Wr, typ, axis_only):
        """Orientation error of a module pose, modulo the module's own symmetry. A propeller (axis_only): the angle between
        its axes (the type's symmetry axis; the spin phase is free). Otherwise by geometry: the submitted module is moved
        so that its centroid sits on the reference one, then registered onto the reference module by ICP starting from
        where it is; the rotation that registration needs is the error (a symmetric equivalent of the reference pose
        needs none), unless the registration does not land on the surface (surface p90 above the tolerance: an error
        larger than the ICP basin), which reads as no orientation credit."""
        a = typ["sym_axis"] / max(np.linalg.norm(typ["sym_axis"]), 1e-12)
        if axis_only or str(typ["symmetry"].get("order")) == "continuous" or typ["symmetry"].get("continuous"):
            # a propeller (spin phase free) or a body of revolution: the angle between the axes
            return math.degrees(math.acos(min(1.0, max(-1.0, float((Wa[:3, :3] @ a) @ (Wr[:3, :3] @ a))))))
        I = self.cfg["installation"]
        q, D, N, tree, size, base = self.ref.module_samples(typ["name"])
        disk = self.cfg["sampling"]["disk_mm"]
        tol = base + max(float(I["orientation_fit_mm"][0]), float(I["orientation_fit_mm"][1]) * size)
        c = typ["centroid"]
        shift = make_T(np.eye(3), apply(Wr, c) - apply(Wa, c))
        P = apply(inv(Wr) @ shift @ Wa, q)
        # the module's own samples read p90 = base against its dense samples (curvature, sample spacing); a pose that
        # already lands on the reference module (a symmetric or near-symmetric equivalent) has no orientation error
        d0 = surface_distances(P, D, N, tree, disk)
        if len(d0) and float(np.quantile(d0, 0.9)) <= tol:
            return 0.0
        T, _rms = icp(P, D, N, tree, np.eye(4), self.cfg["alignment"]["icp_iters"], self.cfg["alignment"]["trim"])
        d = surface_distances(apply(T, P), D, N, tree, disk)
        p90 = float(np.quantile(d, 0.9)) if len(d) else np.inf
        if p90 > tol:
            return 180.0
        return rot_angle_deg(T[:3, :3])

    def installation(self):
        I = self.cfg["installation"]
        ref = self.ref
        wsum = wacc = wconn = 0.0
        for pid in sorted(ref.placements):
            pl = ref.placements[pid]
            typ = ref.types[pl["type"]]
            w = float(I["type_weights"].get(pl["type"], I.get("type_default_weight", 1.0)))
            wsum += w
            it = self.inst_of.get(pid)
            rec = {"type": pl["type"], "present": it is not None, "weight": w}
            if it is None:
                rec["score"] = 0.0
                self.res["purchased"][pid] = rec
                continue
            try:
                Wa = self.A @ it["W"]
                Wr = pl["W"]
                measured_in = "reference"
                if pl["type"] in ref.prop_types:
                    m = ref.prop_motor.get(pid)
                    im = self.inst_of.get(m)
                    if im is not None:
                        Wr = (self.A @ im["W"]) @ inv(ref.placements[m]["W"]) @ pl["W"]
                        measured_in = "motor " + m
                c = typ["centroid"]
                e_p = float(np.linalg.norm(apply(Wa, c) - apply(Wr, c)))
                e_r = self._orientation_error(Wa, Wr, typ, pl["type"] in ref.prop_types)
                pos = band(e_p, I["position_mm"])
                ori = band(e_r, I["orientation_deg"])
                wo = float(I["w_orientation"])
                sc = it["completeness"] * pos * (1.0 - wo + wo * ori)
                rec.update(position_error_mm=e_p, orientation_error_deg=e_r, measured_in=measured_in)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the installation of %s" % pid, exc)
                sc = 0.0
                rec["error"] = str(exc)[:200]
            vol_all = sum(self.sol[sid]["V"] for sid in it["slots"].values())
            vol_conn = sum(self.sol[sid]["V"] for sid in it["slots"].values() if self.sol[sid]["component"] in self.sub.connected)
            conn = vol_conn / vol_all if vol_all > 0 else 0.0
            wacc += w * sc
            wconn += w * conn
            rec.update(score=sc, completeness=it["completeness"], connected=conn)
            self.res["purchased"][pid] = rec
        poses = wacc / wsum if wsum > 0 else 0.0
        conn_p = wconn / wsum if wsum > 0 else 0.0
        conn_m = float(np.mean([1.0 if (self.part_of.get(t) is not None and self.part_of[t]["component"] in self.sub.connected)
                                else 0.0 for t in ref.tags]))
        conn = 0.5 * conn_m + 0.5 * conn_p
        self.stats["installation_poses"] = poses
        self.stats["connected_parts_fraction"] = conn_m
        self.stats["connected_modules_fraction"] = conn_p
        self.stats["p01_grounded"] = 1.0 if (self.part_of.get("P01") is not None and
                                             self.part_of["P01"]["component"] in self.sub.grounded) else 0.0
        return float(I["w_poses"]) * poses + float(I["w_connected"]) * conn

    # ------------------------------------------------------------------ mass properties
    @staticmethod
    def _combine(items):
        """items: (mass g, centroid, inertia per unit density about the centroid (g mm2 / density), density)."""
        M = sum(m for m, _c, _I, _r in items)
        if M <= 0:
            return 0.0, np.zeros(3), np.zeros(3)
        com = sum(m * c for m, c, _I, _r in items) / M
        It = np.zeros((3, 3))
        for m, c, I, rho in items:
            d = c - com
            It += rho * I + m * (float(d @ d) * np.eye(3) - np.outer(d, d))
        return M, com, np.sort(np.linalg.eigvalsh(It))

    def mass(self):
        Mc = self.cfg["mass"]
        ref = self.ref

        def moved(V, c, I, T, rho):
            R = T[:3, :3]
            return (V * rho, apply(T, c), R @ I @ R.T, rho)

        exp = []
        for t, p in ref.parts.items():
            exp.append(moved(p["V"], p["com"], p["I"], np.eye(4), p["density"]))
        for pid, pl in ref.placements.items():
            typ = ref.types[pl["type"]]
            for c in typ["cons"]:
                exp.append(moved(c["V"], c["com"], c["I"], pl["W"], typ["density"]))
        act = []
        for s in self.sub.solids:
            if s["tag"] is not None:
                rho = float(ref.parts[s["tag"]]["density"])
                act.append(moved(s["V"], s["com"], s["I"], self.A, rho))
            elif "match" in s:
                cl = ref.classes[s["match"]["cls"]]
                rep = cl["rep"]
                rho = ref.types[rep["type"]]["density"]
                for it in self.inst:
                    if s["sid"] in it["slots"].values():
                        rho = ref.types[it["type"]]["density"]
                        break
                act.append(moved(rep["V"], rep["com"], rep["I"], self.A @ s["match"]["T"], rho))
            else:
                act.append(moved(s["V"], s["com"], s["I"], self.A, ref.extra_density))
        am, ac, ai = self._combine(act)
        bm, bc, bi = self._combine(exp)
        e_m = abs(am / bm - 1.0) if bm > 0 else np.inf
        e_c = float(np.linalg.norm(ac - bc))
        e_i = float(np.max(np.abs(ai / np.maximum(bi, 1e-9) - 1.0)))
        w = Mc["weights"]
        self.stats.update(mass_g=am, mass_ref_g=bm, mass_error_rel=finite(e_m, 1e9), com_error_mm=e_c,
                          inertia_error_rel=finite(e_i, 1e9))
        return (float(w[0]) * band(e_m, Mc["mass_rel"]) + float(w[1]) * band(e_c, Mc["com_mm"]) +
                float(w[2]) * band(e_i, Mc["inertia_rel"]))

    # ------------------------------------------------------------------ fit
    def _ref_items(self, key):
        if key in self.ref.parts:
            return [self.ref.parts[key]["placed"]]
        if key in self.ref.placements:
            return self.ref.placement_items(key)
        return []

    def _agent_items(self, key):
        if key in self.ref.parts:
            s = self.part_of.get(key)
            p = self.part_placed(s) if s is not None else None
            return [p] if p is not None else []
        if key in self.ref.placements:
            it = self.inst_of.get(key)
            return self.instance_items(it) if it is not None else []
        return []

    def fit(self):
        """Fit (module docstring, rule 5): purchased instances against their supports and the made parts, made parts
        against each other; unrecognised extra solids (an unmatched tagged part included) are obstacles too, without
        exemption (the reference has none)."""
        Fc = self.cfg["fit"]
        I = self.cfg["installation"]
        ref = self.ref
        if not hasattr(ref, "_fit_cache"):
            ref._fit_cache = {}
        cache = ref._fit_cache
        extras = [p for _s, p in self.extra_items()]
        wsum = wacc = 0.0
        for pid in sorted(ref.placements):
            pl = ref.placements[pid]
            w = float(I["type_weights"].get(pl["type"], I.get("type_default_weight", 1.0)))
            wsum += w
            it = self.inst_of.get(pid)
            rec = self.res["purchased"].setdefault(pid, {"type": pl["type"]})
            if it is None:
                rec["fit"] = 0.0
                continue
            try:
                sc = self._fit_instance(pid, it, cache, extras, rec)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the fit of %s" % pid, exc)
                sc = 0.0
                rec["fit"] = 0.0
            wacc += w * sc
        purchased = wacc / wsum if wsum > 0 else 0.0
        man_scores = []
        placed = {t: self.part_placed(self.part_of[t]) for t in ref.tags if self.part_of.get(t) is not None}
        for tag in ref.tags:
            p = placed.get(tag)
            if p is None:
                man_scores.append(0.0)
                continue
            try:
                sc, excess = self._penetration_part(tag, p, placed, extras, cache)
                self.res["parts"][tag]["penetration_mm3"] = excess
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the penetration of %s" % tag, exc)
                sc = 0.0
            man_scores.append(sc)
        manufactured = float(np.mean(man_scores)) if man_scores else 0.0
        self.stats["fit_purchased"] = purchased
        self.stats["fit_manufactured"] = manufactured
        return float(Fc["w_purchased"]) * purchased + float(Fc["w_manufactured"]) * manufactured

    def _fit_instance(self, pid, it, cache, extras, rec):
        Fc = self.cfg["fit"]
        ref = self.ref
        disk = self.cfg["sampling"]["disk_mm"]
        reach = float(Fc["contact_mm"][1])
        slack = float(Fc["penetration_slack_mm3"])
        mine = self.instance_items(it)
        cs = []
        for sup in ref.supports.get(pid, []):
            if not self._ref_items(sup):
                continue                     # a support the grader does not model (not a part, not a placement)
            key = ("d", pid, sup)
            if key not in cache:
                cache[key] = min_distance(ref.placement_items(pid), self._ref_items(sup), disk, reach)
            d_ref = cache[key]
            other = self._agent_items(sup)
            d = min_distance(mine, other, disk, reach) if other else np.inf
            cs.append(band(max(0.0, d - (d_ref if math.isfinite(d_ref) else 0.0)), Fc["contact_mm"]))
        contact = float(np.mean(cs)) if cs else 1.0
        excess = 0.0
        for tag in ref.tags:
            s = self.part_of.get(tag)
            p = self.part_placed(s) if s is not None else None
            if p is None or not any(aabb_overlap(m.box, p.box) for m in mine):
                continue
            key = ("c", pid, tag)
            if key not in cache:
                cache[key] = group_common(ref.placement_items(pid), [ref.parts[tag]["placed"]])
            excess += max(0.0, group_common(mine, [p]) - cache[key] - slack)
        for e in extras:
            if any(aabb_overlap(m.box, e.box) for m in mine):
                excess += max(0.0, group_common(mine, [e]) - slack)
        vol = sum(m.volume for m in mine)
        sc = contact * band(excess / max(vol, 1e-9), Fc["penetration_rel"])
        rec.update(fit=sc, contact=contact, penetration_mm3=excess)
        return sc

    def _penetration_part(self, tag, p, placed, extras, cache):
        Fc = self.cfg["fit"]
        ref = self.ref
        slack = float(Fc["penetration_slack_mm3"])
        excess = 0.0
        for t2, p2 in placed.items():
            if t2 == tag or p2 is None or not aabb_overlap(p.box, p2.box):
                continue
            key = ("m",) + tuple(sorted((tag, t2)))
            if key not in cache:
                a, b = ref.parts[tag]["placed"], ref.parts[t2]["placed"]
                cache[key] = common_volume(a.man, b.man) if aabb_overlap(a.box, b.box) else 0.0
            excess += max(0.0, common_volume(p.man, p2.man) - cache[key] - slack)
        for e in extras:
            if aabb_overlap(p.box, e.box):
                excess += max(0.0, common_volume(p.man, e.man) - slack)
        return band(excess / max(p.volume, 1e-9), Fc["penetration_rel"]), excess

    # ------------------------------------------------------------------ rotor clearance
    def _disc(self, rotor):
        import manifold3d as m3
        C = self.cfg["clearance"]
        p = np.asarray(rotor["axis_point_mm"], float)
        d = np.asarray(rotor["axis_dir"], float)
        d = d / np.linalg.norm(d)
        a0, a1 = float(rotor["disc_axial_min_mm"]), float(rotor["disc_axial_max_mm"])
        cyl = m3.Manifold.cylinder(a1 - a0, float(rotor["disc_radius_mm"]), float(rotor["disc_radius_mm"]),
                                   int(C["segments"]))
        T = make_T(rotation_between([0, 0, 1], d), p + d * a0)
        return cyl, T

    def clearance(self):
        C = self.cfg["clearance"]
        ref = self.ref
        if not ref.rotors:
            return 0.0
        slack = float(C["slack_mm3"])

        def disc_placed(cyl, T):
            m = cyl.transform(np.ascontiguousarray(T[:3, :], np.float32))
            lo, hi = cyl.bounding_box()[:3], cyl.bounding_box()[3:]
            c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)
            w = apply(T, c)
            return m, np.concatenate([w.min(axis=0), w.max(axis=0)])

        def overlap(dm, dbox, items):
            tot = 0.0
            for p in items:
                if aabb_overlap(dbox, p.box):
                    tot += common_volume(dm, p.man)
            return tot

        if not hasattr(ref, "_disc_cache"):
            discs = {}
            for r in ref.rotors:
                cyl, T = self._disc(r)
                discs[r["motor"]] = (cyl, T)
            ref_common = {}
            for r in ref.rotors:
                dm, dbox = disc_placed(*discs[r["motor"]])
                items = [ref.parts[t]["placed"] for t in ref.tags]
                for pid in ref.placements:
                    if pid in (r["motor"], r["prop"]):
                        continue
                    items += ref.placement_items(pid)
                tot = overlap(dm, dbox, items)
                for r2 in ref.rotors:
                    if r2["motor"] != r["motor"]:
                        dm2, _b2 = disc_placed(*discs[r2["motor"]])
                        tot += common_volume(dm, dm2)
                ref_common[r["motor"]] = tot
            ref._disc_cache = (discs, ref_common)
        discs, ref_common = ref._disc_cache
        agent_discs = {}
        for r in ref.rotors:
            im = self.inst_of.get(r["motor"])
            if im is None:
                continue
            cyl, Tr = discs[r["motor"]]
            Ta = (self.A @ im["W"]) @ inv(ref.placements[r["motor"]]["W"]) @ Tr
            agent_discs[r["motor"]] = disc_placed(cyl, Ta)
        others_extra = [p for _s, p in self.extra_items()]
        parts = [self.part_placed(s) for s in self.part_of.values()]
        parts = [p for p in parts if p is not None]
        scores = []
        for r in ref.rotors:
            m = r["motor"]
            if m not in agent_discs:
                scores.append(0.0)
                self.res["rotors"][m] = {"score": 0.0, "present": False}
                continue
            try:
                sc = self._clearance_rotor(r, agent_discs, parts, others_extra, ref_common, overlap, slack, C)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the clearance of rotor %s" % m, exc)
                sc = 0.0
                self.res["rotors"][m] = {"score": 0.0, "error": str(exc)[:200]}
            scores.append(sc)
        return float(np.mean(scores))

    def _clearance_rotor(self, r, agent_discs, parts, others_extra, ref_common, overlap, slack, C):
        ref = self.ref
        m = r["motor"]
        dm, dbox = agent_discs[m]
        items = list(parts) + others_extra
        skip = set()
        for key in (m, r["prop"]):
            it = self.inst_of.get(key)
            if it is not None:
                skip.add(id(it))
        # whatever propeller sits on this motor is its rotor (a propeller of the wrong hand there is an installation
        # error, not a collision): propeller instances whose centroid is on the motor's axis within the disc's height
        im = self.inst_of[m]
        Wm = self.A @ im["W"]
        Td = (Wm @ inv(ref.placements[m]["W"]))
        ax_p = apply(Td, np.asarray(r["axis_point_mm"], float))
        ax_d = Td[:3, :3] @ (np.asarray(r["axis_dir"], float) / np.linalg.norm(r["axis_dir"]))
        for it in self.inst:
            if it["type"] not in ref.prop_types or id(it) in skip:
                continue
            c = apply(self.A @ it["W"], ref.types[it["type"]]["centroid"]) - ax_p
            h = float(c @ ax_d)
            if (np.linalg.norm(c - h * ax_d) <= float(C["own_prop_axis_mm"]) and
                    float(r["disc_axial_min_mm"]) - 5.0 <= h <= float(r["disc_axial_max_mm"]) + 5.0):
                skip.add(id(it))
        for it in self.inst:
            if id(it) in skip:
                continue
            items += self.instance_items(it)
        tot = overlap(dm, dbox, items)
        for m2, (dm2, _b2) in agent_discs.items():
            if m2 != m:
                tot += common_volume(dm, dm2)
        excess = max(0.0, tot - ref_common[m] - slack)
        sc = band(excess, C["excess_mm3"])
        self.res["rotors"][m] = {"score": sc, "overlap_mm3": tot, "reference_overlap_mm3": ref_common[m]}
        return sc


# ============================================================================ entry point (the verify stage's child)
def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("extraction")
    ap.add_argument("reference")
    ap.add_argument("purchased_root")
    ap.add_argument("output")
    ap.add_argument("--config", default=os.path.join(HERE, "grader_config.json"))
    a = ap.parse_args(argv[1:])
    with open(a.config) as fh:
        cfg = json.load(fh)
    limit_cpus(int(os.environ.get("ARB_GRADER_CPUS", cfg.get("grader_cpus", 8))))
    try:
        res = grade(a.extraction, a.reference, a.purchased_root, cfg)
    except TaskError as exc:  # the reference, the vendor files or the extraction: verify.py fails the run (task error)
        res = {"ok": False, "task_error": str(exc), "error": traceback.format_exc()[-3000:]}
    except Exception:  # noqa: BLE001 - outside every grading item: a grader fault, never a silent 0 (task error)
        res = {"ok": False, "task_error": "the grader failed outside a grading item", "error": traceback.format_exc()[-3000:]}
    with open(a.output, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    if res.get("ok"):
        print(json.dumps({k: res[k] for k in ("design_score", "criteria", "checks")}, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
