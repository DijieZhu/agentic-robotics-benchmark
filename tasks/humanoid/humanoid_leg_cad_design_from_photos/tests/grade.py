"""Grader of humanoid_leg_cad_design_from_photos: the extracted FreeCAD document of the leg against the leg as designed
(reference/), in millimetres. Root, verify stage, agent gone; reads data only. FreeCAD's assembly solver is never run:
every quantity comes from the placements as saved.

forked from tasks/ground/racecar_cad_design_from_photos/tests/grade.py (benchmark main 850dee3, sha256
7803e197bc86630058c6b410d169469d79d9eef55482c3f638c4d0f6ea8b87ba); the mechanism criterion is in mech.py (new). The
changes are numbered G-H1 to G-H16, each marked "G-H<n>" where it is made:
 G-H1  names, the schema humanoid-cad-reference/1, six criteria, the material keys;
 G-H2  a given type may be carried by a BREP file (the thigh shell), pinned by sha256 like the STEP files;
 G-H3  a no_volume constituent (the shell: its stored faces enclose no usable volume) takes no part in overlaps,
       penetration and inertia; its mass is a point mass at its type's mass centroid; the share of its surface samples
       that lies inside the made parts is reported, not scored;
 G-H4  the alignment: mode "shell" (the shell's exact registration alone) or "thigh_lsq" (a least-squares rigid fit of
       reference points of the thigh's exactly recognised given instances), with a fallback without the shell;
 G-H5  a sphere_like constituent (its three principal moments equal: a ball) left without a solid by the exact
       recognition is claimed among the unclaimed solids by volume and predicted centroid, one-to-one;
 G-H6  a type with more instances in the document than placements in the leg: copies are removed one at a time, each
       time the one whose removal lowers the type's installation score under the usual assignment most; among
       removals of equal cost, the copy at the placement nearest to a surplus copy; the removed copies are extra
       solids;
 G-H7  the hole radius cap; in a hole that holds a threaded insert, the segments the insert covers form one item scored
       on position and direction only;
 G-H8  the floor: three segments, each the weighted mean of band(surface p90) over its parts (weights by count, square
       root of the area or area); every item is multiplied by its segment's floor, features by the smaller of the
       part's own band and its segment's floor, mass by the whole leg's; the template's ramp on the mean shape is gone;
       the criteria before their floors and the design score under all three weightings are statistics;
 G-H9  the mechanism criterion (mech.py);
 G-H10 how a Revolute is read: its frame on the thigh's side, the worse of parallel ones, empty groups merged, a series
       shared by its count (mech.py);
 G-H11 a ball-centre type (the rod ends): position = the ball centre, orientation not graded, assigned by ball centre;
 G-H12 statistics: per joint offset and angle, the foot points, membership, per-chain transmission and closure errors,
       the segment floors, every criterion before its floor, the alignment and the per-segment fits;
 G-H13 bounded memory of G-H3's inside share: each tessellation tests the points in its box in chunks of points
       (inside_share), so that one containment call holds at most INSIDE_CANDIDATES_MAX ray-triangle candidates; each
       point's rays, candidates and hits are those of the single call over all points (trimesh decides every ray on its
       own), and the few points whose two rays disagree are cast again in a random direction, per call, as before;
 G-H14 the inside share is reproducible: the generator trimesh draws that random direction from is seeded per tessellation
       and chunk (INSIDE_RECAST_SEED) for each containment call, its state saved before and restored after the call, so
       that nothing else sees a different random stream;
 G-H15 the fallback alignment without the shell needs every other type of alignment.thigh_types (the two thigh
       actuators and the bearing), each recognised exactly and present once;
 G-H16 G-H6's removal loop scores, per removal, only the copies that the usual assignment of all remaining copies uses
       (removing an unused copy leaves that assignment and its score as they are); the assignment's cost matrix is
       computed with numpy (the same distances, bit for bit).
The template's own later changes (coincident vertices, hull union, connected from the base, left-over copies, empty
links) are kept.

    python3 -B grade.py EXTRACT_DIR REF_DIR PURCHASED_ROOT OUT_JSON [--config grader_config.json]

EXTRACT_DIR is fcstd_extract.py's output (model.json + one world-frame BREP per solid). REF_DIR holds reference.json
(schema humanoid-cad-reference/1), parts/Pnn.brep (the made parts, in the reference frame = the frame of the given shell
file) and purchased_pins.json (the file of every given type, pinned by sha256, and which solid is which constituent).
PURCHASED_ROOT is where those files are read: the image's own /opt/task-assets/payload (given/<PN>/<PN>.step and the
shell's BREP; a file whose sha256 differs from its pin is a task error). Every number is in grader_config.json. OUT_JSON:
{"ok", "design_score", "criteria", "checks", "statistics", "parts", "purchased", "mechanism", "diagnostics"} (or
{"ok": false, "error"} when the grader itself fails).

The rules (the instruction's grading section states them):

 1. What is graded: the solids inside the document's Assembly (its members, App::Part containers, folders and links);
    without an Assembly, the document's visible solids, with no joint credit. A solid is usable when it is closed and has
    a volume; there is no BRepCheck (its geometry is judged by its tessellation, mesh_solid). Only three things zero the
    document, with the reason: it cannot be read; it holds no solid; neither the shell nor the thigh's other given parts
    (G-H15) can serve the alignment. Everything else is local (Submission): a tagged leaf that is not a PartDesign Body of exactly
    one closed solid, or a tag on more than one leaf, gets no credit and its solids count as extra solids; open solids
    and loose faces of untagged leaves are left out; each is reported. A grading item that fails scores 0 and counts in
    grader_item_errors; a failure outside every item is the grader's own and fails the run (task_error).
 2. Alignment (G-H4), rigid (no scale, no reflection, no ICP), document frame -> reference frame. Mode "shell": the pose
    of the one shell instance, recognised exactly on its B-rep vertex set, taken onto its reference placement. Mode
    "thigh_lsq": a least-squares rigid fit (Kabsch) of reference points of the thigh's given instances that are
    recognised exactly, one per placement (the shell three points, each thigh actuator two points on its axis, the
    flanged bearing one). When the shell is absent, not recognised exactly or present more than once, either mode uses
    the fit without the shell, which needs every other thigh type (the two thigh actuators and the bearing, each exactly
    once, G-H15); neither -> the third whole-document zero. The mode used,
    the residual and, per segment, the residual of a rigid fit of that segment's exactly recognised given instances onto
    the reference are statistics.
 3. Given-part recognition by full shape, in the submission's own frame: every untagged solid against every constituent
    of every given type: volume within volume_rel, then a rigid registration (proper rotations only): exact on the
    B-rep vertex set when the solid is an unchanged import (vertices that coincide within mesh.weld_mm counted once), else
    ICP on surface samples accepted when the symmetric surface p95 is small. Instances are formed in two rounds: the base
    type (the shell) first, then every other type, each seeded by the solids matching its anchor constituent. A
    sphere_like constituent left without a solid is claimed (G-H5). Completeness = the share of the type's constituents
    found. Instances go to the reference placements one-to-one per type (Hungarian on the module centroid; a ball-centre
    type on its ball centre, G-H11); surplus copies are removed first (G-H6). Light constituents (small, never an
    anchor) are recognised and counted, but not tessellated.
 4. Made-part assignment: the tag gives the family; inside a family the tagged parts go to the reference instances
    one-to-one by installed position (Hungarian on the centroid). Tags inside a family are interchangeable.
 5. Criteria (weights in config), each in [0, 1]:
    shape        per made part: w_surface band(symmetric surface p90) + w_iou rising(IoU), at the installed placement;
                 mean over the made parts (a missing one scores 0). A part whose tessellation is no closed manifold of
                 its own volume enters the volume measures as its convex hull (both the intersection and the union).
    features     per reference hole (r <= r_max_mm) and its steps: the best one-to-one match among the part's concave
                 cylindrical and conical faces, band(axis offset) x band(radius error) x band(axis angle); the covered
                 segments of an insert hole are one item without the radius term, matched by a face whose radius lies
                 in the insert window (G-H7); mean per hole, per part, then over the parts that have holes.
    installation per reference given instance: completeness x band(position) x (1 - w_orient + w_orient x
                 band(orientation modulo the type's symmetry; an axis-only type the angle between its axes; a ball-centre
                 type not graded)); type weights (the base type 0 in mode "shell": it is the datum); then w_poses x that
                 + w_connected x the share of made parts and given instances connected through unsuppressed joints of
                 any type to the component that holds the shell, when that component is grounded.
    mechanism    mech.py: joints, foot points, membership, transmission, closure (G-H9, G-H10, G-H11).
    mass         w x band(total mass), w x band(centre of mass), w x band(largest principal-inertia error); made parts
                 at their material's density, given parts at their types' masses (a no_volume type as a point mass,
                 G-H3), unrecognised solids at the extra_material density.
    fit          per given instance: mean over its reference supports of band(contact distance beyond the reference's)
                 x band(penetration volume into made parts and extra solids / own volume); per made part band(penetration
                 into the other made parts, the given instances and extra solids / its own volume); overlaps the
                 reference has itself are exempt, pair by pair; a no_volume constituent overlaps nothing (G-H3).
 6. Floors (G-H8): per made part band(its surface p90; floor.p90_mm) (missing: 0); a segment's floor = the weighted mean
    over its parts (floor.weight); shape has none; features x min(own band, segment floor); every installation, fit and
    mechanism item x the floor of its segment(s); mass x the whole leg's weighted floor.
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
sys.path.insert(0, HERE)
import mech  # noqa: E402  (G-H9: the mechanism criterion, numpy only)

CRITERIA = ["shape", "features", "installation", "mechanism", "mass", "fit"]          # G-H1
TAG_PREFIX = "P"
WEIGHTINGS = ("count", "sqrt_area", "area")                                          # G-H8
THIGH_LSQ_BASE_WEIGHT = 1.0          # G-H4: the base type's installation weight when the configured mode is thigh_lsq
# G-H13: the most ray-triangle candidates one containment call of inside_share may hold (two rays per point, at most one
# candidate per ray and triangle; about 0.5 kB each in trimesh's broad and narrow phase). Bounds memory, not results.
INSIDE_CANDIDATES_MAX = 2000000
# G-H14: the seed of the re-cast directions of inside_share (with the tessellation's index and the chunk's index).
INSIDE_RECAST_SEED = 20261120


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
    """Pin this process to n CPUs: OCC's mesher and manifold3d's TBB size their pools to the whole host otherwise."""
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


def unique_vertices(shape, merge_mm=None):
    """The points of a shape's distinct B-rep vertices, (n, 3), in the order of OCC's vertex map. With merge_mm
    (coincident vertices: the grader passes mesh.weld_mm), vertices that coincide within merge_mm are kept once
    (merge_coincident); without it, every vertex is kept."""
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
    if merge_mm is not None:
        P = merge_coincident(P, merge_mm)
    return P


def merge_coincident(P, tol_mm):
    """Coincident vertices: the points with every group of points within tol_mm of one another (pairwise, joined
    transitively) kept once, as its first point, in the original order. Some vendor solids hold two vertices at one
    point; register_exact maps vertex sets one-to-one and so cannot register such a set, not even onto itself. The
    reference's copy and an unchanged import of the same STEP solid merge alike (the same points up to a rigid move)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree
    P = np.asarray(P, float)
    if len(P) < 2:
        return P
    pairs = cKDTree(P).query_pairs(float(tol_mm), output_type="ndarray")
    if not len(pairs):
        return P
    n = len(P)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _nc, lab = connected_components(g, directed=False)
    _u, first = np.unique(lab, return_index=True)
    return P[np.sort(first)]


def fresh_copy(shape):
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Copy
    from OCP.BRepTools import BRepTools
    c = BRepBuilderAPI_Copy(shape, True, False).Shape()
    BRepTools.Clean_s(c)
    return c


def concave_axial_faces(shape, tol_rel=0.03, tol_abs=0.02):
    """Concave cylindrical and conical faces of a solid (the material outside the surface: holes, bores, countersinks):
    [{kind "cylinder" | "cone", point (on the axis), dir, radius (a cone: its largest radius on the face), span_rad}].
    A free-form face counts as a cylinder when it fits one within max(tol_abs, tol_rel x radius) (the grader passes
    features.fitted_tol_rel / fitted_tol_abs)."""
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
                c = _fitted_cylinder(f, ad, tol_rel=tol_rel, tol_abs=tol_abs)  # a free-form face that is a cylinder
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


def mesh_surface(shape, cfg, tmpdir, seed):
    """G-H3: the Geo of a no_volume solid (the shell): one tessellation where it is, welded, without a manifold (its
    surface samples are used, never its volume). None when it does not tessellate."""
    M = cfg["mesh"]
    try:
        got = _stl_mesh(shape, M["deflection_mm"], M["angle_rad"], tmpdir)
    except Exception:  # noqa: BLE001
        return None
    if got is None:
        return None
    V, F = weld(got[0], got[1], M["weld_mm"])
    if len(F) < 4:
        return None
    geo = Geo(V, F, None, ["no_volume: surface samples only"], seed)
    geo.no_volume = True
    return geo


def exact_box(shape):
    """[xmin, ymin, zmin, xmax, ymax, zmax] of a shape's exact bounding box (OCC AddOptimal, no tolerance enlargement)."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    b = Bnd_Box()
    BRepBndLib.AddOptimal_s(shape, b, False, False)
    return list(b.Get())


def box_corners(lo, hi):
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], float)


def placed_manifold(geo, T):
    """The geo's manifold moved by T (a rigid 4x4), or its convex hull when it has none; None for a no_volume solid
    (G-H3: the shell takes no part in overlaps and penetration)."""
    import manifold3d as m3
    if getattr(geo, "no_volume", False):
        return None
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
    """The grader's own reference or the image's given files are unusable: a broken task, never a zero."""


MECH_KEYS = ("bodies", "body_order", "joints", "poses", "foot_points_mm", "foot_fit_instances", "foot_joints",
             "foot_points_segments", "membership", "chains", "bars", "pitch_bores", "roll_bores", "pitch_joint",
             "roll_joint", "ankle_poses_deg", "transmission_segments", "closure_segments")


class Reference(object):
    """reference.json, the made parts (B-rep, tessellation, mass properties, features, segment) and the given types
    (constituents read from the pinned files, their congruence classes, tessellations and mass properties); the base type
    (alignment.base_type, the shell) and its one reference placement (base_pid); the floor segments (G-H8) and the
    mechanism block (G-H9)."""

    def __init__(self, ref_dir, purchased_root, cfg, tmpdir, log):
        self.dir, self.cfg, self.tmp, self.log = ref_dir, cfg, tmpdir, log
        with open(os.path.join(ref_dir, "reference.json")) as fh:
            self.spec = json.load(fh)
        if self.spec.get("schema") != "humanoid-cad-reference/1":                       # G-H1
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
            dens = p.get("density_g_mm3")
            if dens is None:
                dens = mats[p["material"]]["density_g_mm3"]                              # G-H1: the material keys
            self.parts[tag] = {"tag": tag, "family": p["family"], "density": float(dens), "shape": sols[0],
                               "V": V, "com": c, "I": I, "geo": geo, "placed": Placed(geo, np.eye(4), cfg),
                               "features": list(p.get("features", [])), "segment": p.get("segment"),
                               "box": exact_box(sols[0])}
        self.tags = sorted(self.parts)
        self.family_of = {t: p["family"] for t, p in self.parts.items()}
        self.family_tags = {}
        for t, f in sorted(self.family_of.items()):
            self.family_tags.setdefault(f, []).append(t)
        self._load_purchased(pending)
        self.placements = {}
        for pl in self.spec["placements"]:
            if pl["type"] not in self.types:
                raise TaskError("placement %s of an unknown type %s" % (pl["id"], pl["type"]))
            W = np.asarray(pl["local_to_world"], float)
            W = make_T(orthonormal(W[:3, :3]), W[:3, 3])
            self.placements[pl["id"]] = {"id": pl["id"], "type": pl["type"], "W": W, "segment": pl.get("segment")}
        self.base_type = str(cfg["alignment"]["base_type"])
        bases = sorted(p for p, pl in self.placements.items() if pl["type"] == self.base_type)
        if self.base_type not in self.types or len(bases) != 1:
            raise TaskError("the reference needs exactly one placement of the base type %s (found %d)" % (
                self.base_type, len(bases)))
        self.base_pid = bases[0]
        self.supports = self.spec.get("supports", {})
        self.unscored = self.spec.get("unscored", [])
        self._load_segments()                                                            # G-H8
        self._load_alignment_points()                                                    # G-H4
        self._load_mechanism()                                                           # G-H9

    # ------------------------------------------------------------------ given types
    def _start_purchased(self, root):
        """Check every given file against its pin and start reading them in worker processes (before any meshing in
        this process). G-H2: a type may be carried by a BREP file."""
        with open(os.path.join(self.dir, "purchased_pins.json")) as fh:
            pins = json.load(fh)
        jobs = []
        for tname, t in sorted(self.spec["purchased"].items()):
            pin = pins["types"][tname]
            rel = pin.get("file", pin.get("step"))
            path = os.path.join(root, rel)
            if not os.path.isfile(path):
                raise TaskError("the given file of %s is missing: %s" % (tname, path))
            digest = sha256_file(path)
            if digest != pin["sha256"]:
                raise TaskError("the given file %s does not match its pin (sha256 %s, pinned %s)" % (path, digest, pin["sha256"]))
            jobs.append((tname, path))
        return pins, StepReader(jobs, self.tmp, int(self.cfg.get("step_workers", 4)))

    def _load_purchased(self, pending):
        pins, reader = pending
        breps = reader.result()
        self.types = {}
        seed = int(self.cfg["sampling"]["seed"])
        weld_mm = float(self.cfg["mesh"]["weld_mm"])
        n = 0
        for tname, t in sorted(self.spec["purchased"].items()):
            pin = pins["types"][tname]
            files = breps[tname]
            if len(files) != len(pin["solids"]):
                raise TaskError("%s: the given file has %d solids, the pin %d" % (tname, len(files), len(pin["solids"])))
            byid = {c["id"]: c for c in t["constituents"]}
            cons = [None] * len(t["constituents"])
            order = [c["id"] for c in t["constituents"]]
            no_volume = bool(t.get("no_volume"))                                       # G-H3
            for k, (f, ps) in enumerate(zip(files, pin["solids"])):
                s = solids_of(read_brep(f))[0]
                V, c, I = mass_props(s)
                if abs(V - ps["volume_mm3"]) > 1e-5 * max(1.0, abs(ps["volume_mm3"])) or \
                        np.linalg.norm(c - np.asarray(ps["centroid_mm"])) > 1e-3:
                    raise TaskError("%s solid %d does not match its pin (volume %.6f vs %.6f)" % (tname, k, V, ps["volume_mm3"]))
                cid = ps["constituent"]
                n += 1
                # light constituents (small, never the anchor) are recognised on their vertex sets and counted in
                # completeness and mass, but not tessellated; a no_volume one keeps its surface samples only (G-H3)
                light = bool(byid[cid].get("light"))
                if light:
                    geo = None
                elif no_volume:
                    geo = mesh_surface(s, self.cfg, self.tmp, seed + 5000 + n)
                else:
                    geo = mesh_solid(s, self.cfg, self.tmp, seed + 5000 + n, volume=V)
                verts = unique_vertices(s, weld_mm)                                    # coincident vertices once
                ext = verts.max(axis=0) - verts.min(axis=0) if len(verts) else np.zeros(3)
                cons[order.index(cid)] = {"id": cid, "type": tname, "k": order.index(cid), "V": V, "com": c, "I": I,
                                          "verts": verts, "size": float(np.linalg.norm(ext)), "geo": geo,
                                          "role": byid[cid].get("source_role"), "light": light,
                                          "sphere_like": bool(byid[cid].get("sphere_like")),            # G-H5
                                          "no_volume": no_volume}
            if any(c is None for c in cons):
                raise TaskError("%s: a constituent has no given solid" % tname)
            if no_volume and len(cons) != 1:
                raise TaskError("%s: a no_volume type must be one solid" % tname)
            vol = sum(c["V"] for c in cons)
            cen = sum(c["com"] * c["V"] for c in cons) / vol
            anchor = [c["id"] for c in cons].index(t["anchor"])
            if cons[anchor]["light"]:
                raise TaskError("%s: the anchor constituent %s is flagged light" % (tname, t["anchor"]))
            sym = t.get("symmetry") or {}
            ax = sym.get("axis_local")
            allv = np.vstack([c["verts"] for c in cons if len(c["verts"])])
            sk = t.get("skeleton") or {}
            mc = t.get("mass_centroid_local_mm")
            if no_volume and mc is None:
                raise TaskError("%s: a no_volume type needs mass_centroid_local_mm" % tname)
            self.types[tname] = {"name": tname, "pn": t.get("part_number"), "mass_g": float(t["mass_g"]),
                                 "density": float(t["mass_g"]) / vol, "cons": cons, "anchor": anchor, "V": vol,
                                 "centroid": cen, "sym_axis": np.asarray(ax if ax is not None else [0, 0, 1], float),
                                 "symmetry": sym, "axis_only": bool(t.get("axis_only") or sym.get("axis_only")),
                                 "ball_centre_only": bool(t.get("ball_centre_only")),            # G-H11
                                 "no_volume": no_volume,                                         # G-H3
                                 "mass_centroid": None if mc is None else np.asarray(mc, float),
                                 "skeleton": {k: np.asarray(v, float) for k, v in sk.items()},
                                 "box_local": box_corners(allv.min(axis=0), allv.max(axis=0)),
                                 "anchor_box": (cons[anchor]["verts"].min(axis=0), cons[anchor]["verts"].max(axis=0))}
            if self.types[tname]["ball_centre_only"] and "ball_centre_local" not in self.types[tname]["skeleton"]:
                raise TaskError("%s: a ball-centre type needs skeleton.ball_centre_local" % tname)
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
            cl["M"] = dict((id(m[0]), m[1]) for m in cl["members"])

    # ------------------------------------------------------------------ G-H8: floor segments
    def _load_segments(self):
        segs = self.spec.get("segments")
        if not isinstance(segs, dict) or not segs:
            raise TaskError("reference.json has no floor segments")
        self.segments = {}
        seen = {}
        for name in sorted(segs):
            tags = list(segs[name]["tags"])
            area = segs[name]["area_mesh_mm2"]
            for t in tags:
                if t not in self.parts:
                    raise TaskError("segment %s lists an unknown part %s" % (name, t))
                if t in seen:
                    raise TaskError("part %s is in two segments" % t)
                if not float(area[t]) > 0:
                    raise TaskError("part %s has no area" % t)
                seen[t] = name
            self.segments[name] = {"tags": tags, "area": {t: float(area[t]) for t in tags}}
        missing = sorted(t for t in self.parts if t not in seen)
        if missing:
            raise TaskError("parts in no segment: %s" % missing)
        for t, p in self.parts.items():
            if p["segment"] is not None and p["segment"] != seen[t]:
                raise TaskError("part %s: its segment %s differs from the segment table (%s)" % (t, p["segment"], seen[t]))
            p["segment"] = seen[t]
        for pid, pl in self.placements.items():
            if pl["segment"] not in self.segments:
                raise TaskError("placement %s has no valid segment (%r)" % (pid, pl["segment"]))

    def floor_weight(self, tag, kind):
        a = self.segments[self.parts[tag]["segment"]]["area"][tag]
        if kind == "count":
            return 1.0
        if kind == "sqrt_area":
            return math.sqrt(a)
        if kind == "area":
            return a
        raise TaskError("floor.weight %r" % kind)

    # ------------------------------------------------------------------ G-H4: least-squares points per type
    def _load_alignment_points(self):
        A = self.cfg["alignment"]
        h = float(A["lsq_axis_half_span_mm"])
        self.lsq_local = {}
        partial = bool((self.spec.get("mechanism") or {}).get("partial"))      # a test reference may lack a type
        for tname, npts in sorted(A["lsq_points"].items()):
            if tname not in self.types:
                if partial:
                    continue
                raise TaskError("alignment.lsq_points names an unknown type %s" % tname)
            typ = self.types[tname]
            self.lsq_local[tname] = self.type_points(typ, int(npts), h)
        for tname in A["thigh_types"]:
            pids = [p for p, pl in self.placements.items() if pl["type"] == tname]
            if partial and tname not in self.types:
                continue
            if tname not in self.lsq_local or len(pids) != 1:
                raise TaskError("thigh type %s needs one placement and its least-squares points" % tname)

    @staticmethod
    def type_points(typ, npts, h):
        """Reference points of a type in its local frame: 1 = the centroid (a ball-centre type: the ball centre); 2 = the
        centroid's foot on the type's axis (skeleton axis, else the symmetry axis through the centroid) +- h along it;
        3 = the centre of the anchor's vertex box and that point + h along local x and local y."""
        c = np.asarray(typ["centroid"], float)
        if npts == 1:
            return (typ["skeleton"]["ball_centre_local"] if typ["ball_centre_only"] else c)[None, :].copy()
        if npts == 2:
            sk = typ["skeleton"]
            if "axis_point_local" in sk:
                p0, d = sk["axis_point_local"], sk["axis_dir_local"]
            else:
                p0, d = c, typ["sym_axis"]
            d = np.asarray(d, float) / np.linalg.norm(d)
            f = p0 + d * float((c - p0) @ d)
            return np.array([f - h * d, f + h * d])
        if npts == 3:
            lo, hi = typ["anchor_box"]
            b = 0.5 * (np.asarray(lo) + np.asarray(hi))
            return np.array([b, b + [h, 0.0, 0.0], b + [0.0, h, 0.0]])
        raise TaskError("lsq_points must be 1, 2 or 3")

    def fit_point(self, pid, W):
        """The single reference point of a placement at pose W (type local -> reference): a ball-centre type its ball
        centre, any other its centroid."""
        typ = self.types[self.placements[pid]["type"]]
        return apply(W, typ["skeleton"]["ball_centre_local"] if typ["ball_centre_only"] else typ["centroid"])

    # ------------------------------------------------------------------ G-H9: the mechanism block
    def _load_mechanism(self):
        m = self.spec.get("mechanism")
        if not isinstance(m, dict):
            raise TaskError("reference.json has no mechanism block")
        lack = [k for k in MECH_KEYS if k not in m]
        if lack:
            raise TaskError("the mechanism block lacks %s" % ", ".join(lack))
        segs = set(self.segments)
        bodies = m["bodies"]
        where = {}
        for b in sorted(bodies):
            for t in bodies[b].get("tags", []):
                if t not in self.parts or t in where:
                    raise TaskError("body %s: tag %s unknown or in two bodies" % (b, t))
                where[t] = b
            for pid in bodies[b].get("given_instances", []):
                if pid not in self.placements or pid in where:
                    raise TaskError("body %s: instance %s unknown or in two bodies" % (b, pid))
                where[pid] = b
        children = set()
        names = []
        for j in m["joints"]:
            for k in ("name", "parent", "child", "point_mm", "dir", "floor_segments"):
                if k not in j:
                    raise TaskError("mechanism joint %s lacks %s" % (j.get("name"), k))
            if j["parent"] not in bodies or j["child"] not in bodies or j["child"] in children:
                raise TaskError("mechanism joint %s: bodies %s -> %s" % (j["name"], j["parent"], j["child"]))
            if names and j["parent"] not in children and any(j["parent"] == jj["child"] for jj in m["joints"]):
                raise TaskError("mechanism joints are not in tree order at %s" % j["name"])
            if not set(j["floor_segments"]) <= segs:
                raise TaskError("mechanism joint %s: unknown floor segments" % j["name"])
            children.add(j["child"])
            names.append(j["name"])
        if list(m["poses"]["joints"]) != names:
            raise TaskError("mechanism poses are not given in the joints' order")
        deg = np.asarray(m["poses"]["deg"], float)
        if deg.ndim != 2 or deg.shape[1] != len(names) or not np.all(np.isfinite(deg)):
            raise TaskError("mechanism poses malformed")
        for k in ("foot_points_segments", "transmission_segments", "closure_segments"):
            if not set(m[k]) <= segs:
                raise TaskError("mechanism %s: unknown segments" % k)
        mem = m["membership"]
        for t in mem.get("tags", []):
            if t not in where:
                raise TaskError("membership tag %s is in no body" % t)
        for pid in mem.get("given_instances", []):
            if pid not in where:
                raise TaskError("membership instance %s is in no body" % pid)
        self.member_body = where
        for ch, c in m["chains"].items():
            for k in ("actuator", "upper", "lower"):
                if c.get(k) not in self.placements:
                    raise TaskError("chain %s: %s %r is no placement" % (ch, k, c.get(k)))
        for nm, c in m["bars"].items():
            if c.get("tag") not in self.parts or c.get("upper") not in self.placements or c.get("lower") not in self.placements:
                raise TaskError("bar %s malformed" % nm)
        for pid in (list(m["pitch_bores"]) + list(m["roll_bores"]) if m["chains"] else []) + list(m["foot_fit_instances"]):
            if pid not in self.placements:
                raise TaskError("mechanism names an unknown placement %s" % pid)
        self.partial = bool(m.get("partial"))
        if not self.partial:
            if not (m["joints"] and m["chains"] and m["bars"] and mem.get("tags") and len(m["foot_fit_instances"]) >= 3):
                raise TaskError("the mechanism block is incomplete (only a test reference may be marked partial)")
        self.mech = m
        self.mech_names = names
        self.sk_ref = self.skeleton(lambda pid: self.placements[pid]["W"], self.placements)
        self.dphi_truth, self.bore_senses = {}, None
        if m["chains"]:
            try:
                self.dphi_truth, self.bore_senses = mech.reference_dphi(self.sk_ref, m)
            except (ValueError, KeyError) as exc:
                raise TaskError("the reference's ankle skeleton: %s" % exc)
            for ch, v in (m.get("dphi_truth_deg") or {}).items():
                if ch not in self.dphi_truth or float(np.max(np.abs(np.asarray(v, float) - self.dphi_truth[ch]))) > 1e-6:
                    raise TaskError("the stored crank angle changes of chain %s differ from the grader's" % ch)
        self.foot_ref_pts = {pid: self.fit_point(pid, self.placements[pid]["W"]) for pid in m["foot_fit_instances"]}
        self.member_corners = {}
        for t in mem.get("tags", []):
            bx = self.parts[t]["box"]
            self.member_corners[t] = box_corners(bx[:3], bx[3:])
        for pid in mem.get("given_instances", []):
            typ = self.types[self.placements[pid]["type"]]
            self.member_corners[pid] = apply(self.placements[pid]["W"], typ["box_local"])

    def skeleton(self, Wfun, pids):
        """Skeleton points in the reference frame of the placements `pids` (Wfun(pid) = type local -> reference):
        {"act": {pid: (axis point, directed axis)}, "ball": {pid: ball centre}, "bore": {pid: bore centre}}."""
        out = {"act": {}, "ball": {}, "bore": {}}
        for pid in sorted(pids):
            typ = self.types[self.placements[pid]["type"]]
            sk = typ["skeleton"]
            if not sk:
                continue
            W = np.asarray(Wfun(pid), float)
            if "axis_point_local" in sk:
                d = W[:3, :3] @ sk["axis_dir_local"]
                out["act"][pid] = (apply(W, sk["axis_point_local"]), d / np.linalg.norm(d))
            if "ball_centre_local" in sk:
                out["ball"][pid] = apply(W, sk["ball_centre_local"])
            if "bore_centre_local" in sk:
                out["bore"][pid] = apply(W, sk["bore_centre_local"])
        return out

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
        """(query points, dense points, dense normals, KD-tree, size) of a given type's whole module, local frame."""
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
    if path.lower().endswith(".brep"):                                                    # G-H2: a BREP-carried type
        shape = read_brep(path)
    else:
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

    Only an unreadable document and a document without any usable solid stop the grade here (ok False, with the
    reason); the third whole-document condition, neither the shell nor the thigh's other given parts usable for the
    alignment (G-H15),
    is decided once the given parts are recognised (align, G-H4). A solid is usable when it is closed and has a volume
    (no BRepCheck). Everything else is local, with a diagnostic and a statistic:
      - a tagged leaf that is not a PartDesign Body of exactly one closed solid: the tag gets no credit (its instance is
        unmatched); its closed solids stay in the unit as untagged extra solids (mass, fit);
      - a tag on more than one leaf (a Body reused through several links included): none of them is matched; their
        solids are extra solids;
      - an untagged leaf with open solids, open shells or loose faces, or a solid that cannot be read: the bad pieces
        are left out (counted), its closed solids stay;
      - no Assembly in the document: its visible solids are graded (the extractor read them), without joint credit."""

    def __init__(self, ex_dir, ref, cfg, tmpdir, diags):
        self.ok = False
        self.solids = []
        self.cfg, self.tmp = cfg, tmpdir
        self.connected = set()
        self.grounded = []
        self.stats = st = {"tags_invalid": 0, "tags_duplicated": 0, "ignored_non_solid_leaves": 0,
                           "ignored_bad_solids": 0, "ignored_unreadable_leaves": 0, "no_assembly": 0, "empty_links": 0}
        path = os.path.join(ex_dir, "model.json")
        if not os.path.isfile(path):
            diags.append("No readable FreeCAD document.")
            return
        try:
            with open(path) as fh:
                model = json.load(fh)
        except ValueError as exc:
            raise TaskError("the extraction's model.json is not JSON: %s" % exc)
        if not isinstance(model, dict) or not str(model.get("format", "")).startswith("humanoid-fcstd-extract/"):  # G-H1
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
        tagged = {}          # tag -> [(leaf, comp name, solid records, a Body of one closed solid?)]
        untagged = []        # (leaf, comp name, solid record)

        def note(msg):
            local_note(diags, msg)

        # empty links: components the extractor marked as links whose target has no readable shape (an object of a
        # document that is not in the submission); nothing of them is read, which costs what is missing; one summary
        empty = [c for c in model.get("components", []) if c.get("empty_link")]
        st["empty_links"] = len(empty)
        if empty:
            labels = [str(c.get("label") or c.get("name"))[:40] for c in empty]
            note("%d component(s) are links whose target has no readable shape (an object of a document that is not in "
                 "the submission folder?); nothing of them is read: %s%s" % (
                     len(empty), ", ".join(labels[:5]), ", ..." if len(labels) > 5 else ""))

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
                good = [x for x in sols if x.get("closed")]                       # closed, no BRepCheck
                free = int(leaf.get("n_free_faces", 0) or 0)
                if tag is not None:
                    single = (leaf.get("type") == "PartDesign::Body" and len(sols) == 1 and len(good) == 1 and
                              free == 0)
                    tagged.setdefault(tag, []).append((leaf, comp["name"], good, single))
                    continue
                if not sols:
                    if free:
                        st["ignored_non_solid_leaves"] += 1
                        note("%s has open faces and no solid: left out" % where)
                    else:
                        st["ignored_non_solid_leaves"] += 1      # a sketch, a datum, a wire
                    continue
                if free or len(good) < len(sols):
                    st["ignored_bad_solids"] += (len(sols) - len(good)) + (1 if free else 0)
                    note("%s has %d open solid(s)%s: those are left out" % (
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
            note("%s is a %s with %d solid(s), %d closed: not a PartDesign Body of exactly one closed solid, so the tag "
                 "gets no credit; its closed solids count as extra solids" % (tag, leaf.get("type"), n, len(good)))
            for x in good:
                add(leaf, cname, x, None, demoted="invalid tagged leaf " + tag)
        for leaf, cname, x in untagged:
            add(leaf, cname, x, None)
        if len(self.solids) > int(G["max_solids"]):
            diags.append("Too many solids in the unit (%d)." % len(self.solids))
            self.solids = []
            return
        keep = []
        merge_mm = float(cfg["mesh"]["weld_mm"])     # coincident vertices once (read here: a config without it is a task error)
        for s in self.solids:
            try:
                shape = read_brep(s["file"])
                sols = solids_of(shape)
                if len(sols) != 1:
                    raise ValueError("%d solids in one solid file" % len(sols))
                s["shape"] = sols[0]
                V, c, I = mass_props(sols[0])                                     # no BRepCheck re-check
                if not V > float(G["min_volume_mm3"]):
                    raise ValueError("no volume")
                s["V"], s["com"], s["I"] = V, c, I
                if s["tag"] is None:
                    s["verts"] = unique_vertices(sols[0], merge_mm)
                    ext = s["verts"].max(axis=0) - s["verts"].min(axis=0) if len(s["verts"]) else np.zeros(3)
                    s["size"] = float(np.linalg.norm(ext))
            except Exception as exc:  # noqa: BLE001 - a solid that does not load is left out
                st["ignored_bad_solids"] += 1
                note("a solid of %s cannot be used (%s): left out" % (s["label"] or s["component"], str(exc)[:100]))
                if s["tag"] is not None:
                    st["tags_invalid"] += 1
                continue
            keep.append(s)
        for k, s in enumerate(keep):
            s["sid"] = k
        self.solids = keep
        if not self.solids:
            diags.append("The document holds no solid." if model.get("no_assembly") else
                         "The Assembly holds no solid (objects outside the Assembly are not read).")
            return
        # no tagged base part: the shell is found among the recognised given parts (align, G-H4); connected from the
        # base: the connected components are found once the assigned shell is known (Grading.connect)
        self.grounded = [g for g in model.get("grounded", []) if g]
        self.ok = True

    def geo(self, s, seed):
        if "geo" not in s:
            s["geo"] = mesh_solid(s["shape"], self.cfg, self.tmp, seed, volume=s.get("V"))
        return s["geo"]


def local_note(diags, msg):
    """A local diagnostic ("Local: ..."), at most 25 of them."""
    if sum(1 for d in diags if d.startswith("Local:")) < 25:
        diags.append("Local: " + msg)


def joint_connected(model, start):
    """Components reached through unsuppressed joints of any type (the joint type is not graded; a revolute or a
    slider joint connects two components as a fixed one does) from `start`, the component that holds the assigned shell
    (connected from the base: the search starts there only, and only when that component is grounded; grounding another
    component connects nothing). No start: nothing is connected."""
    graph = {c["name"]: set() for c in model.get("components", [])}
    for j in model.get("joints", []):
        if j.get("suppressed") or j.get("error"):
            continue
        a, b = j.get("part1"), j.get("part2")
        if a in graph and b in graph and a != b:
            graph[a].add(b)
            graph[b].add(a)
    grounded = set(g for g in model.get("grounded", []) if g in graph)
    reached = set([start]) if start in grounded else set()
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
    """Module instances (rule 3), in two rounds: first the base type (alignment.base_type), then every other type. A
    round seeds an instance on every not yet claimed solid that matches the type's anchor constituent, then attaches the
    other constituents one-to-one by predicted centroid (within attach_mm) to the instances of that round. Then the
    sphere_like constituents left without a solid are claimed (G-H5, claim_sphere_like). Returns [{"type", "W" (local ->
    submission frame), "slots": {k: sid}, "completeness"}]; matched solids left over are marked s["extra_cls"]."""
    attach = float(cfg["recognition"]["attach_mm"])
    by_cls = {}
    for s in sub.solids:
        if "match" in s:
            by_cls.setdefault(s["match"]["cls"], []).append(s)
    claimed = set()

    def n_attachable(typ, W):
        n = 0
        for c in typ["cons"]:
            p = apply(W, c["com"])
            if any(np.linalg.norm(p - o["com"]) <= attach for o in by_cls.get(c["cls"], []) if o["sid"] not in claimed):
                n += 1
        return n

    def one_round(tnames):
        inst = []
        for tname in sorted(tnames):
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
        for cl in ref.classes:
            slots = []
            for i, it in enumerate(inst):
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
        return inst

    base = ref.base_type
    inst = one_round([base]) + one_round([t for t in ref.types if t != base])
    claim_sphere_like(sub, ref, inst, cfg, stats, claimed)                              # G-H5
    for it in inst:
        it["completeness"] = len(it["slots"]) / float(len(ref.types[it["type"]]["cons"]))
    for s in sub.solids:
        if "match" in s and s["sid"] not in claimed:
            s["extra_cls"] = s["match"]["cls"]
    stats["purchased_instances_found"] = len(inst)
    stats["purchased_solids_unattached"] = sum(1 for s in sub.solids if "extra_cls" in s)
    return inst


def claim_sphere_like(sub, ref, inst, cfg, stats, claimed):
    """G-H5: a sphere_like constituent (three equal principal moments: a ball, whose vertex set does not register) left
    without a solid is claimed among the solids no instance has claimed (untagged, not demoted): volume within
    recognition.sphere_like_volume_rel of the constituent's and centroid within recognition.attach_mm of its predicted
    one, one-to-one (Hungarian on the distance). A claimed solid is recognised as that constituent at its predicted
    pose."""
    R = cfg["recognition"]
    vrel, attach = float(R["sphere_like_volume_rel"]), float(R["attach_mm"])
    slots = []
    for i, it in enumerate(inst):
        typ = ref.types[it["type"]]
        for c in typ["cons"]:
            if c["sphere_like"] and c["k"] not in it["slots"]:
                slots.append((i, c, apply(it["W"], c["com"])))
    cands = [s for s in sub.solids if s["tag"] is None and not s.get("demoted") and s["sid"] not in claimed]
    n = 0
    if slots and cands:
        big = 1e12
        cost = np.full((len(slots), len(cands)), big)
        for a, (_i, c, p) in enumerate(slots):
            for b, s in enumerate(cands):
                if abs(s["V"] / c["V"] - 1.0) <= vrel:
                    d = float(np.linalg.norm(p - s["com"]))
                    if d <= attach:
                        cost[a, b] = d
        for a, b in hungarian(cost):
            if cost[a, b] >= big:
                continue
            i, c, _p = slots[a]
            s = cands[b]
            cl = ref.classes[c["cls"]]
            T = inst[i]["W"] @ cl["M"][id(c)]
            s["match"] = {"cls": c["cls"], "T": T, "err": float(cost[a, b]), "mode": "sphere_like", "alt": [T]}
            s.pop("extra_cls", None)
            inst[i]["slots"][c["k"]] = s["sid"]
            claimed.add(s["sid"])
            n += 1
    stats["purchased_solids_sphere_like"] = n


# ============================================================================ alignment (G-H4)
def align(sub, ref, inst, cfg, stats, diags):
    """Rigid alignment document -> reference (module docstring, rule 2), or None with the reason (the third
    whole-document zero). Mode "shell": T = W_ref(shell) @ inv(W_sub(shell)) from the one shell instance recognised
    exactly. Mode "thigh_lsq" (and the fallback when the shell cannot serve): a Kabsch fit of the least-squares points
    of the thigh's given instances (alignment.thigh_types), one recognised exactly per placement; without the shell it
    needs every other thigh type (G-H15: the two thigh actuators and the bearing), each exactly once. No ICP, no
    candidates."""
    A = cfg["alignment"]
    mode = str(A["mode"])
    if mode not in ("shell", "thigh_lsq"):
        raise TaskError("alignment.mode %r" % mode)
    sol = {s["sid"]: s for s in sub.solids}
    bt = ref.base_type

    def exact_instances(tname):
        typ = ref.types[tname]
        return [it for it in inst if it["type"] == tname and typ["anchor"] in it["slots"]
                and sol[it["slots"][typ["anchor"]]]["match"]["mode"] == "exact"]

    shells = [it for it in inst if it["type"] == bt and ref.types[bt]["anchor"] in it["slots"]]
    stats["base_instances_found"] = len(shells)
    shell_exact = exact_instances(bt)
    shell_ok = len(shells) == 1 and len(shell_exact) == 1
    if not shell_ok:
        if not shells:
            diags.append("The given thigh shell is not among the Assembly's solids (objects outside the Assembly are not "
                         "read): the alignment uses the thigh's other given parts.")
        elif len(shells) > 1:
            diags.append("The given thigh shell is in the document %d times: the alignment uses the thigh's other given "
                         "parts." % len(shells))
        else:
            diags.append("The given thigh shell is not its unchanged import (its vertex set does not register exactly): "
                         "the alignment uses the thigh's other given parts.")
    if mode == "shell" and shell_ok:
        it = shell_exact[0]
        m = sol[it["slots"][ref.types[bt]["anchor"]]]["match"]
        stats["alignment_mode"] = 1.0
        stats["alignment_residual_mm"] = float(m["err"])
        stats["base_completeness"] = float(it["completeness"]) if "completeness" in it else 1.0
        return ref.placements[ref.base_pid]["W"] @ inv(it["W"])
    P, Q, used = [], [], []
    for tname in A["thigh_types"]:
        if tname not in ref.lsq_local or (tname == bt and not shell_ok):
            continue
        its = exact_instances(tname)
        allits = [it for it in inst if it["type"] == tname]
        pid = [p for p, pl in ref.placements.items() if pl["type"] == tname][0]
        if len(its) != 1 or len(allits) != 1:
            continue
        loc = ref.lsq_local[tname]
        P.append(apply(its[0]["W"], loc))
        Q.append(apply(ref.placements[pid]["W"], loc))
        used.append(tname)
    need = [t for t in A["thigh_types"] if t != bt and t in ref.lsq_local]                    # G-H15: every one of them
    stats["alignment_lsq_types"] = float(len(used))
    if (bt not in used and not all(t in used for t in need)) or not P:
        diags.append("Neither the thigh shell nor every other given part of the thigh (the two thigh actuators and the "
                     "bearing, each imported unchanged and once) is in the Assembly: no alignment, the document scores "
                     "0.")
        return None
    P, Q = np.vstack(P), np.vstack(Q)
    if not mech.spread_ok(Q) or not mech.spread_ok(P):
        diags.append("The thigh's given parts do not fix a unique pose: no alignment, the document scores 0.")
        return None
    T = kabsch(P, Q)
    T = make_T(orthonormal(T[:3, :3]), T[:3, 3])
    res = float(np.sqrt(np.mean(np.sum((apply(T, P) - Q) ** 2, axis=1))))
    stats["alignment_mode"] = 2.0 if bt in used else 3.0
    stats["alignment_residual_mm"] = res
    return T


# ============================================================================ the grade
CONFIG_RULES = {("mechanism", "frame"): "parent_side", ("mechanism", "parallel_revolute"): "worst",
                ("mechanism", "series_revolute"): "best_divided_by_count", ("mechanism", "membership_evidence"):
                "adjacent_joints_matched", ("mechanism", "transmission"): "per_chain",
                ("recognition", "surplus"): "worst_subset_greedy", ("floor", "kind"): "segment_p90",
                ("floor", "features"): "min_own_segment"}


def check_config(cfg):
    """The descriptive keys of grader_config.json name the rules this code implements; a config that names another
    rule is a broken task, not a grade."""
    for (sec, key), want in CONFIG_RULES.items():
        if cfg.get(sec, {}).get(key) != want:
            raise TaskError("grader_config %s.%s = %r: this grader implements %r" % (sec, key, cfg.get(sec, {}).get(key), want))


def grade(ex_dir, ref_dir, purchased_root, cfg, log=None, ref=None):
    """The grade of one extraction (module docstring). `ref`: an already loaded Reference (a development batch run
    grades many documents against one; the verify stage loads it once per document anyway)."""
    t0 = time.time()
    log = log or (lambda m: print("[grade] %s" % m, file=sys.stderr, flush=True))
    tmp = tempfile.mkdtemp(prefix="leg_grade_")
    res = {"ok": True, "design_score": 0.0, "criteria": {k: 0.0 for k in CRITERIA}, "checks": {"cad_valid": 0.0},
           "statistics": {}, "parts": {}, "purchased": {}, "mechanism": {}, "diagnostics": []}
    stats, diags = res["statistics"], res["diagnostics"]
    stats["grader_item_errors"] = 0
    check_config(cfg)
    if ref is None:
        ref = Reference(ref_dir, purchased_root, cfg, tmp, log)
    if len(ref.mech["poses"]["deg"]) != int(cfg["mechanism"]["poses"]) and not ref.partial:
        raise TaskError("the reference has %d mechanism poses, the config %s" % (len(ref.mech["poses"]["deg"]), cfg["mechanism"]["poses"]))
    stats["t_reference_s"] = time.time() - t0
    log("reference loaded (%.1f s): %d parts, %d given types, %d classes" % (
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
        A = align(sub, ref, inst, cfg, stats, diags)                                 # G-H4
    except TaskError:
        raise
    except Exception as exc:  # noqa: BLE001 - the submitted thigh parts defeat the alignment: the document scores 0
        item_error(stats, diags, "the alignment", exc)
        A = None
    stats["t_alignment_s"] = time.time() - t1
    if A is None:
        res["checks"]["cad_valid"] = 0.0
        return res
    log("recognised %d given instances; alignment mode %d, residual %.2e mm" % (
        len(inst), int(stats.get("alignment_mode", 0)), stats["alignment_residual_mm"]))
    comps = set(c["name"] for c in sub.model.get("components", []))
    stats["grounded_components"] = len(set(g for g in sub.grounded if g in comps))
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
        self._common_ip = {}           # (given placement, tag) -> common volume as submitted
        self._score = {}               # (instance id, placement) -> (installation score, position, orientation)

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

    def type_weight(self, tname):
        I = self.cfg["installation"]
        if tname == self.ref.base_type and str(self.cfg["alignment"]["mode"]) == "thigh_lsq":
            return THIGH_LSQ_BASE_WEIGHT                                                # G-H4
        return float(I["type_weights"].get(tname, I.get("type_default_weight", 1.0)))

    def key_point(self, typ, W):
        """The point a type is assigned and positioned by: the ball centre of a ball-centre type (G-H11), else the
        centroid."""
        return apply(W, typ["skeleton"]["ball_centre_local"] if typ["ball_centre_only"] else typ["centroid"])

    def _hungarian(self, tname, pids, insts):
        # G-H16: the key points once per instance and placement, the distance matrix by numpy (the same numbers as the
        # norm of each pair's difference)
        typ = self.ref.types[tname]
        Ki = np.array([self._inst_key(typ, it) for it in insts], float).reshape(-1, 3)
        Kp = np.array([self.key_point(typ, self.ref.placements[p]["W"]) for p in pids], float).reshape(-1, 3)
        cost = np.linalg.norm(Ki[None, :, :] - Kp[:, None, :], axis=-1)
        return [(pids[r], insts[c]) for r, c in hungarian(cost)]

    def _inst_key(self, typ, it):
        """G-H16: an instance's key point in the reference frame (cached per instance)."""
        if not hasattr(self, "_keys"):
            self._keys = {}
        k = id(it)
        if k not in self._keys:
            self._keys[k] = self.key_point(typ, self.A @ it["W"])
        return self._keys[k]

    def inst_score(self, it, pid):
        """(score, position error, orientation error or None) of instance `it` at placement `pid` (cached)."""
        key = (id(it), pid)
        if key not in self._score:
            I = self.cfg["installation"]
            pl = self.ref.placements[pid]
            typ = self.ref.types[pl["type"]]
            Wa, Wr = self.A @ it["W"], pl["W"]
            e_p = float(np.linalg.norm(self.key_point(typ, Wa) - self.key_point(typ, Wr)))
            e_r = None
            ori = 1.0
            # G-H11: a ball-centre type has no orientation; a type of weight 0 (the datum) is not turned either
            if not typ["ball_centre_only"] and self.type_weight(pl["type"]) > 0:
                e_r = self._orientation_error(Wa, Wr, typ, typ["axis_only"])
                ori = band(e_r, I["orientation_deg"])
            wo = float(I["w_orientation"])
            sc = it["completeness"] * band(e_p, I["position_mm"]) * (1.0 - wo + wo * ori)
            self._score[key] = (sc, e_p, e_r)
        return self._score[key]

    def _type_score(self, tname, pids, insts):
        return sum(self.inst_score(it, p)[0] for p, it in self._hungarian(tname, pids, insts))

    SURPLUS_TIE = 1e-12    # G-H6: removals whose type scores differ by no more than this cost the same

    def assign_instances(self):
        """Given instances to reference placements, one-to-one per type (Hungarian on the key point). G-H6: a type with
        more instances than placements first loses copies, one at a time, each time the one whose removal lowers the
        type's installation score under that assignment most; among removals of equal cost, the copy at the placement
        nearest to a surplus copy (surplus_tie_break); the removed copies are extra solids."""
        self.inst_of = {}
        removed = ties = 0
        for tname, typ in sorted(self.ref.types.items()):
            pids = sorted(p for p, pl in self.ref.placements.items() if pl["type"] == tname)
            mine = [it for it in self.inst if it["type"] == tname]
            if not pids or not mine:
                continue
            keep = list(mine)
            while len(keep) > len(pids):
                # G-H16: removing a copy that the usual assignment of all of keep leaves without a placement leaves that
                # assignment as it is: its score is the assignment's own; only the copies it uses are scored again
                try:
                    cur = self._hungarian(tname, pids, keep)
                    used = set(id(it) for _p, it in cur)
                    cur_score = sum(self.inst_score(it, p)[0] for p, it in cur)
                except Exception:  # noqa: BLE001 - no current assignment: every copy is scored as before
                    used, cur_score = None, None
                scores = []
                for i in range(len(keep)):
                    if used is not None and id(keep[i]) not in used:
                        scores.append(cur_score)
                        continue
                    try:
                        sc = self._type_score(tname, pids, keep[:i] + keep[i + 1:])
                    except Exception as exc:  # noqa: BLE001 - the copy cannot be scored: it is the one removed
                        item_error(self.stats, self.diags, "a copy of %s" % tname, exc)
                        sc = -1.0
                    scores.append(sc)
                lo = min(scores)
                tied = [i for i, sc in enumerate(scores) if sc <= lo + self.SURPLUS_TIE]
                if len(tied) > 1:
                    ties += 1
                    keep.pop(self.surplus_tie_break(tname, pids, keep, tied))
                else:
                    keep.pop(tied[0])
                removed += 1
            for pid, it in self._hungarian(tname, pids, keep):
                self.inst_of[pid] = it
        self.stats["purchased_instances_assigned"] = len(self.inst_of)
        self.stats["purchased_instances_surplus_removed"] = removed
        self.stats["purchased_surplus_ties_broken"] = ties

    def surplus_tie_break(self, tname, pids, keep, tied):
        """G-H6, removals of equal cost: the index (into keep) of the tied copy at the placement nearest to a surplus
        copy. A surplus copy is a copy that the usual assignment of all of `keep` leaves without a placement; a tied
        copy is at the placement that assignment gives it (a tied copy that is itself a surplus copy: distance 0);
        distances between key points in the reference frame; equal distances: the first in `keep`."""
        typ = self.ref.types[tname]
        try:
            at = {id(it): pid for pid, it in self._hungarian(tname, pids, keep)}
        except Exception as exc:  # noqa: BLE001 - no assignment to measure from: the first tied copy goes
            item_error(self.stats, self.diags, "the surplus copies of %s" % tname, exc)
            return tied[0]
        spare = [self.key_point(typ, self.A @ it["W"]) for it in keep if id(it) not in at]
        if not spare:
            return tied[0]

        def dist(i):
            pid = at.get(id(keep[i]))
            if pid is None:
                return 0.0
            p = self.key_point(typ, self.ref.placements[pid]["W"])
            return min(float(np.linalg.norm(p - q)) for q in spare)
        return min(tied, key=lambda i: (dist(i), i))

    def connect(self):
        """connected from the base: the walk starts at the component holding the assigned shell, when it is grounded."""
        base = self.inst_of.get(self.ref.base_pid)
        btyp = self.ref.types[self.ref.base_type]
        start = None
        if base is not None and btyp["anchor"] in base["slots"]:
            start = self.sol[base["slots"][btyp["anchor"]]]["component"]
        self.sub.connected = joint_connected(self.sub.model, start) if start is not None else set()

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

    def left_over_sids(self):
        """left-over copies: the solids of the given instances that no reference placement received (G-H6's removed
        copies included)."""
        if not hasattr(self, "_left_over"):
            taken = set(id(it) for it in self.inst_of.values())
            left = [it for it in self.inst if id(it) not in taken]
            self._left_over = set(sid for it in left for sid in it["slots"].values())
            self.stats["purchased_instances_left_over"] = len(left)
            if left:
                kinds = sorted(set(it["type"] for it in left))
                local_note(self.diags, "%d given piece(s) in the document beyond the leg's count (%s): their solids "
                                       "count as extra solids" % (len(left), ", ".join(kinds[:8])))
        return self._left_over

    def extra_items(self):
        """Untagged solids in no given instance that a reference placement received: recognised leftovers and the
        solids of left-over instances (the grader's copy at the registered pose), and unrecognised solids."""
        out = []
        left = self.left_over_sids()
        for s in self.sub.solids:
            if s["tag"] is not None:
                continue
            if "extra_cls" in s or s["sid"] in left:
                p = self.slot_placed(s["sid"])
            elif "match" not in s:
                p = self.part_placed(s)
            else:
                continue
            if p is not None:
                out.append((s, p))
        return out

    # ------------------------------------------------------------------ the grade
    def run(self):
        cfg, res, st = self.cfg, self.res, self.stats
        F = cfg["floor"]
        if F.get("kind") != "segment_p90" or F.get("features") != "min_own_segment" or F.get("weight") not in WEIGHTINGS:
            raise TaskError("floor configuration %r" % F)
        self.assign_parts()
        self.assign_instances()
        self.connect()
        self.segment_fits()
        t = time.time()
        self.shape()
        st["t_shape_s"] = time.time() - t
        self.floors = {w: self.floor_values(w) for w in WEIGHTINGS}                   # G-H8
        items = {}
        t = time.time()
        items["features"] = self.features()
        st["t_features_s"] = time.time() - t
        t = time.time()
        items["installation"] = self.installation()
        items["mass"] = self.mass()
        st["t_install_mass_s"] = time.time() - t
        t = time.time()
        items["mechanism"] = self.mechanism()
        st["t_mechanism_s"] = time.time() - t
        t = time.time()
        items["fit"] = self.fit()
        st["t_fit_s"] = time.time() - t
        self.shell_inside()
        ones = {k: 1.0 for k in self.floors[F["weight"]]}
        raw = self.evaluate(items, ones)
        for k in CRITERIA:
            if k != "shape":
                st["raw_" + k] = finite(raw[k])                                          # G-H12
        for w in WEIGHTINGS:
            crit = self.evaluate(items, self.floors[w])
            st["reward_floor_" + w] = finite(sum(cfg["weights"][k] * finite(crit[k]) for k in CRITERIA))
            if w == F["weight"]:
                for k in CRITERIA:
                    res["criteria"][k] = finite(crit[k])
        for seg, v in self.floors[F["weight"]].items():
            st["floor_" + seg] = finite(v)
        res["design_score"] = float(sum(cfg["weights"][k] * res["criteria"][k] for k in CRITERIA))

    def evaluate(self, items, fl):
        """The six criteria from the item lists and the floors {segment: floor, "leg": the whole leg's} (G-H8)."""
        cfg = self.cfg
        out = {"shape": self.res["criteria"]["shape"]}
        fi = items["features"]
        out["features"] = float(np.mean([v * min(own, fl[seg]) for _t, v, own, seg in fi])) if fi else 0.0
        I = cfg["installation"]
        ins = items["installation"]
        wsum = sum(w for w, _s, _g in ins["poses"])
        poses = sum(w * s * fl[g] for w, s, g in ins["poses"]) / wsum if wsum > 0 else 0.0
        conn_m = float(np.mean([c * fl[g] for c, g in ins["conn_parts"]])) if ins["conn_parts"] else 0.0
        conn_p = sum(w * c * fl[g] for w, c, g in ins["conn_inst"]) / wsum if wsum > 0 else 0.0
        out["installation"] = float(I["w_poses"]) * poses + float(I["w_connected"]) * (0.5 * conn_m + 0.5 * conn_p)
        out["mechanism"] = mech.combine(items["mechanism"], fl, cfg["mechanism"]["w"])
        out["mass"] = items["mass"] * fl["leg"]
        Fc = cfg["fit"]
        fp = items["fit"]
        wsum = sum(w for w, _s, _g in fp["purchased"])
        pur = sum(w * s * fl[g] for w, s, g in fp["purchased"]) / wsum if wsum > 0 else 0.0
        man = float(np.mean([s * fl[g] for s, g in fp["made"]])) if fp["made"] else 0.0
        out["fit"] = float(Fc["w_purchased"]) * pur + float(Fc["w_manufactured"]) * man
        return out

    # ------------------------------------------------------------------ G-H8: the floors
    def floor_values(self, kind):
        """{segment: weighted mean of band(p90) over its parts, "leg": the same over every part}."""
        F = self.cfg["floor"]
        out = {}
        na = da = 0.0
        for seg in sorted(self.ref.segments):
            num = den = 0.0
            for t in self.ref.segments[seg]["tags"]:
                w = self.ref.floor_weight(t, kind)
                num += w * self.own_band(t)
                den += w
            out[seg] = num / den if den > 0 else 0.0
            na += num
            da += den
        out["leg"] = na / da if da > 0 else 0.0
        return out

    def own_band(self, tag):
        p90 = self.res["parts"].get(tag, {}).get("surface_p90_mm")
        if p90 is None or not self.res["parts"][tag].get("present"):
            return 0.0
        return band(p90, self.cfg["floor"]["p90_mm"])

    def segment_fits(self):
        """G-H4 / G-H12 statistics: per segment, the rigid least-squares fit of the segment's exactly recognised given
        instances (their least-squares points; one point for the other types) onto the reference: RMS residual, shift,
        turn. They tell the datum's error from the parts' errors; nothing is scored by them."""
        A = self.cfg["alignment"]
        h = float(A["lsq_axis_half_span_mm"])
        for seg in sorted(self.ref.segments):
            P, Q = [], []
            for pid in sorted(self.inst_of):
                pl = self.ref.placements[pid]
                if pl["segment"] != seg:
                    continue
                it = self.inst_of[pid]
                typ = self.ref.types[pl["type"]]
                if typ["anchor"] not in it["slots"] or self.sol[it["slots"][typ["anchor"]]]["match"]["mode"] != "exact":
                    continue
                loc = self.ref.lsq_local.get(pl["type"])
                if loc is None:
                    loc = Reference.type_points(typ, 1, h)
                P.append(apply(self.A @ it["W"], loc))
                Q.append(apply(pl["W"], loc))
            n = sum(len(x) for x in P)
            self.stats["segment_fit_points_" + seg] = float(n)
            if not P:
                continue
            P, Q = np.vstack(P), np.vstack(Q)
            if mech.spread_ok(Q):
                T = kabsch(P, Q)
                self.stats["segment_fit_turn_deg_" + seg] = rot_angle_deg(T[:3, :3])
            else:
                T = make_T(np.eye(3), Q.mean(axis=0) - P.mean(axis=0))
            self.stats["segment_fit_rms_mm_" + seg] = float(np.sqrt(np.mean(np.sum((apply(T, P) - Q) ** 2, axis=1))))
            self.stats["segment_fit_shift_mm_" + seg] = float(np.linalg.norm(apply(T, P.mean(axis=0)) - P.mean(axis=0)))

    # ------------------------------------------------------------------ criteria
    def shape(self):
        S = self.cfg["shape"]
        disk = self.cfg["sampling"]["disk_mm"]
        scores = {}
        hull = []                      # hull union: the tags of the parts graded on their convex hull
        self.stats["parts_on_convex_hull"] = hull
        for tag in self.ref.tags:
            rp = self.ref.parts[tag]
            s = self.part_of.get(tag)
            rec = {"family": rp["family"], "present": s is not None, "segment": rp["segment"]}
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
                # hull union: the union from the volumes of the manifolds actually intersected (a tessellation that is
                # no manifold of the solid's volume enters as its convex hull, placed_manifold), so IoU <= 1
                va = float(a.man.volume()) if (a.geo.man is None and a.man is not None) else a.volume
                vb = float(b.man.volume()) if (b.geo.man is None and b.man is not None) else b.volume
                iou = inter / max(va + vb - inter, 1e-9)
                if a.geo.man is None:
                    hull.append(s["tag"])
                    rec["convex_hull"] = True
                    local_note(self.diags, "%s: its surface cannot be meshed as a closed volume of its own size, so its "
                                           "convex hull stands for it in the volume measures (overlap and penetration)"
                               % s["tag"])
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
        hull.sort()
        self.res["criteria"]["shape"] = float(np.mean([scores[t] for t in self.ref.tags]))
        # G-H8: the template's ramp on the mean shape is gone; the floors come from each part's own p90 (floor_values)

    def features(self):
        """[(tag, raw features value, own band, segment)] of the parts with holes (G-H7, G-H8)."""
        Fc = self.cfg["features"]
        kinds = set(Fc.get("kinds", ["hole", "partial"]))
        # the fitted-cylinder tolerances, the radius cap and the insert rule come from the config (read here, outside
        # every item: a config without them is a broken task, not a zero)
        tols = (float(Fc["fitted_tol_rel"]), float(Fc["fitted_tol_abs"]))
        r_max = float(Fc["r_max_mm"])                                                    # G-H7
        ins = Fc["insert_holes"]
        if ins.get("score") != "offset_x_angle":
            raise TaskError("features.insert_holes.score %r" % ins.get("score"))
        cover = (float(ins["covered_if_radius_le_body_plus_mm"]), [float(x) for x in ins["match_radius_window"]])
        out = []
        n_holes = n_found = n_cov = 0
        for tag in self.ref.tags:
            # a hole the reference marks "scored": false is not read (the hidden second hole of an insert's seat)
            holes = [h for h in self.ref.parts[tag]["features"] if h.get("kind", "hole") in kinds and
                     h.get("scored", True) is not False and
                     max([float(h.get("radius_mm", h.get("radius", 0.0)))] +
                         [float(x["radius_mm"]) for x in (h.get("steps") or [])]) <= r_max]
            if not holes:
                continue
            seg = self.ref.parts[tag]["segment"]
            own = self.own_band(tag)
            s = self.part_of.get(tag)
            if s is None:
                out.append((tag, 0.0, own, seg))
                self.res["parts"][tag]["features"] = 0.0
                n_holes += len(holes)
                continue
            try:
                ps = self._features_of(tag, holes, s, Fc, tols, cover)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the features of %s" % tag, exc)
                ps = (0.0, len(holes), 0, 0)
            out.append((tag, ps[0], own, seg))
            n_holes += ps[1]
            n_found += ps[2]
            n_cov += ps[3]
            self.res["parts"][tag]["features"] = ps[0]
            self.res["parts"][tag]["features_floor"] = own
        self.stats["holes_total"] = n_holes
        self.stats["holes_found"] = n_found
        self.stats["insert_items"] = n_cov
        return out

    def _features_of(self, tag, holes, s, Fc, tols, cover):
        """(score, holes, holes found, insert items) of one part's features. G-H7: in a hole that holds a threaded
        insert (its feature names the insert's body radius rb), the steps of radius <= rb + covered_if... are one item,
        matched by a concave cylinder or cone of radius in [w0 x rb, rb + w1] and scored on offset and angle only."""
        if "axial" not in s:
            s["axial"] = concave_axial_faces(s["shape"], tol_rel=tols[0], tol_abs=tols[1])
        R = self.A[:3, :3]
        faces = [{"kind": c["kind"], "point": apply(self.A, c["point"]), "dir": R @ c["dir"], "radius": c["radius"]}
                 for c in s["axial"]]
        plus, win = cover
        items = []
        n_cov = 0
        for h in holes:
            steps = feature_items(h)
            rb = (h.get("insert") or {}).get("body_radius_mm")
            if rb is None:
                items += [(h["id"], p, d, r, kind, None) for (p, d, r, kind) in steps]
                continue
            rb = float(rb)
            covered = [x for x in steps if x[2] <= rb + plus]
            items += [(h["id"], p, d, r, kind, None) for (p, d, r, kind) in steps if r > rb + plus]
            if covered:
                items.append((h["id"], covered[0][0], covered[0][1], None, "covered", (win[0] * rb, rb + win[1])))
                n_cov += 1
        sc = np.zeros(len(items))
        if faces:
            M = np.zeros((len(items), len(faces)))
            for i, (_hid, p, d, r, kind, rw) in enumerate(items):
                for j, c in enumerate(faces):
                    if rw is None:
                        if c["kind"] != kind:
                            continue
                        rad = band(c["radius"] - r, Fc["radius_mm"])
                    else:
                        if not (rw[0] <= c["radius"] <= rw[1]):
                            continue
                        rad = 1.0
                    ang = math.degrees(math.acos(min(1.0, abs(float(d @ c["dir"])))))
                    v = p - c["point"]
                    off = float(np.linalg.norm(v - c["dir"] * float(v @ c["dir"])))
                    M[i, j] = band(off, Fc["offset_mm"]) * rad * band(ang, Fc["angle_deg"])
            for i, j in hungarian(-M):
                sc[i] = M[i, j]
        by_hole = {}
        for (hid, _p, _d, _r, _k, _w), v in zip(items, sc):
            by_hole.setdefault(hid, []).append(v)
        hole_scores = [float(np.mean(v)) for v in by_hole.values()]
        return float(np.mean(hole_scores)), len(hole_scores), sum(1 for v in hole_scores if v > 0.5), n_cov

    def _orientation_error(self, Wa, Wr, typ, axis_only):
        """Orientation error of a module pose, modulo the module's own symmetry. An axis_only type (its spin about the
        local axis is not graded) and a body of revolution: the angle between its axes (the type's symmetry axis); when
        the type is also flagged flip_symmetric (a half turn about a transverse axis maps it onto itself), the angle
        between the axis lines, either sense. Otherwise by geometry: the submitted module is moved so that its centroid
        sits on the reference one, then registered onto the reference module by ICP starting from where it is; the
        rotation that registration needs is the error (a symmetric equivalent of the reference pose needs none), unless
        the registration does not land on the surface (surface p90 above the tolerance: an error larger than the ICP
        basin), which reads as no orientation credit."""
        a = typ["sym_axis"] / max(np.linalg.norm(typ["sym_axis"]), 1e-12)
        if axis_only or str(typ["symmetry"].get("order")) == "continuous" or typ["symmetry"].get("continuous"):
            # a spin-free type or a body of revolution: the angle between the axes
            c = float((Wa[:3, :3] @ a) @ (Wr[:3, :3] @ a))
            if typ["symmetry"].get("flip_symmetric"):
                c = abs(c)        # either end up; the exact registration of such a solid has both solutions
            return math.degrees(math.acos(min(1.0, max(-1.0, c))))
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
        T, _rms = icp(P, D, N, tree, np.eye(4), int(self.cfg["alignment"].get("icp_iters", 40)),
                      float(self.cfg["alignment"].get("trim", 0.9)))
        d = surface_distances(apply(T, P), D, N, tree, disk)
        p90 = float(np.quantile(d, 0.9)) if len(d) else np.inf
        if p90 > tol:
            return 180.0
        return rot_angle_deg(T[:3, :3])

    def installation(self):
        """{"poses": [(weight, score, segment)], "conn_parts": [(connected, segment)], "conn_inst": [(weight, connected
        volume share, segment)]} (G-H8: each item times its segment's floor in evaluate)."""
        ref = self.ref
        poses, conn_i = [], []
        for pid in sorted(ref.placements):
            pl = ref.placements[pid]
            typ = ref.types[pl["type"]]
            w = self.type_weight(pl["type"])
            seg = pl["segment"]
            it = self.inst_of.get(pid)
            rec = {"type": pl["type"], "present": it is not None, "weight": w, "segment": seg}
            if it is None:
                rec["score"] = 0.0
                poses.append((w, 0.0, seg))
                conn_i.append((w, 0.0, seg))
                self.res["purchased"][pid] = rec
                continue
            try:
                sc, e_p, e_r = self.inst_score(it, pid)
                rec.update(position_error_mm=e_p, orientation_error_deg=e_r, axis_only=typ["axis_only"],
                           ball_centre_only=typ["ball_centre_only"])
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the installation of %s" % pid, exc)
                sc = 0.0
                rec["error"] = str(exc)[:200]
            vol_all = sum(self.sol[sid]["V"] for sid in it["slots"].values())
            vol_conn = sum(self.sol[sid]["V"] for sid in it["slots"].values() if self.sol[sid]["component"] in self.sub.connected)
            conn = vol_conn / vol_all if vol_all > 0 else 0.0
            poses.append((w, sc, seg))
            conn_i.append((w, conn, seg))
            rec.update(score=sc, completeness=it["completeness"], connected=conn)
            self.res["purchased"][pid] = rec
        conn_parts = [(1.0 if (self.part_of.get(t) is not None and self.part_of[t]["component"] in self.sub.connected)
                       else 0.0, ref.parts[t]["segment"]) for t in ref.tags]
        wsum = sum(w for w, _s, _g in poses)
        self.stats["installation_poses"] = sum(w * s for w, s, _g in poses) / wsum if wsum > 0 else 0.0
        self.stats["connected_parts_fraction"] = float(np.mean([c for c, _g in conn_parts])) if conn_parts else 0.0
        self.stats["connected_modules_fraction"] = sum(w * c for w, c, _g in conn_i) / wsum if wsum > 0 else 0.0
        base = self.inst_of.get(ref.base_pid)
        btyp = ref.types[ref.base_type]
        self.stats["base_grounded"] = 1.0 if (base is not None and btyp["anchor"] in base["slots"] and
                                              self.sol[base["slots"][btyp["anchor"]]]["component"] in self.sub.grounded) else 0.0
        return {"poses": poses, "conn_parts": conn_parts, "conn_inst": conn_i}

    # ------------------------------------------------------------------ G-H9: the mechanism
    def mechanism(self):
        """{sub-item: [(score, [segments])] or None} (mech.py); records in res["mechanism"]."""
        mc = self.cfg["mechanism"]
        ref, m = self.ref, self.ref.mech
        out = {k: None for k in mc["w"]}
        rec = self.res["mechanism"]
        holdings = {}
        for tag, s in self.part_of.items():
            holdings.setdefault(s["component"], {"tags": [], "instances": []})["tags"].append(tag)
        anchor_comp = {}
        for pid, it in self.inst_of.items():
            sid = it["slots"].get(ref.types[it["type"]]["anchor"])
            if sid is None:
                continue
            anchor_comp[pid] = self.sol[sid]["component"]
            holdings.setdefault(anchor_comp[pid], {"tags": [], "instances": []})["instances"].append(pid)
        try:
            comps = [c["name"] for c in self.sub.model.get("components", [])]
            G = mech.DocGraph(comps, self.sub.model.get("joints", []), holdings,
                              merge_empty=bool(mc["merge_empty_groups"]), max_revolute=int(mc["max_revolute"]))
            tag_group = {t: G.group_of(s["component"]) for t, s in self.part_of.items()}
            bg = mech.body_groups(G, m["bodies"], tag_group, {t: ref.parts[t]["V"] for t in ref.parts})
            look = mech.lookup(G, bg, m, self.A, mc)
            for k, v in G.stats.items():
                self.stats["mech_" + k] = float(v)
            self.stats["mech_frame_mismatches"] = float(sum(
                1 for e in G.edges if float(np.linalg.norm(e["f1"][0] - e["f2"][0])) > float(mc["frame_mismatch_note"][0])
                or mech.line_angle_deg(e["f1"][1], e["f2"][1]) > float(mc["frame_mismatch_note"][1])))
            for msg in G.notes[:5]:
                local_note(self.diags, msg)
            rec["groups"] = {b: g for b, g in bg.items()}
        except Exception as exc:  # noqa: BLE001 - the document's joints cannot be read: the joint items score 0
            item_error(self.stats, self.diags, "the document's joints", exc)
            G, bg, look = None, {}, {j["name"]: {"score": 0.0, "matched": None} for j in m["joints"]}
        if m["joints"]:
            out["joints"] = [(float(look[j["name"]]["score"]), list(j["floor_segments"])) for j in m["joints"]]
            rec["joints"] = {}
            for j in m["joints"]:
                r = look[j["name"]]
                mm = r.get("matched")
                rec["joints"][j["name"]] = {"score": float(r["score"]), "why": r.get("why"), "steps": r.get("n_steps"),
                                            "revolute": None if mm is None else mm["label"],
                                            "offset_mm": None if mm is None else mm["offset_mm"],
                                            "angle_deg": None if mm is None else mm["angle_deg"],
                                            "parallel": None if mm is None else mm.get("parallel")}
                self.stats["mech_joint_%s" % j["name"]] = float(r["score"])
                if mm is not None:
                    self.stats["mech_joint_%s_offset_mm" % j["name"]] = float(mm["offset_mm"])
                    self.stats["mech_joint_%s_angle_deg" % j["name"]] = float(mm["angle_deg"])
        sk = ref.skeleton(lambda pid: self.A @ self.inst_of[pid]["W"], list(self.inst_of))
        if len(m["foot_fit_instances"]) >= 3 and m["foot_joints"]:
            try:
                sub_pts = {pid: ref.fit_point(pid, self.A @ self.inst_of[pid]["W"]) for pid in m["foot_fit_instances"]
                           if pid in self.inst_of}
                s2a, r2a = mech.foot_points(look, m, ref.foot_ref_pts, sub_pts, mc)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the foot points", exc)
                s2a, r2a = 0.0, {}
            out["foot_points"] = [(s2a, list(m["foot_points_segments"]))]
            rec["foot_points"] = dict(r2a, score=s2a)
            for k in ("median_mm", "p90_mm", "fit_rms_mm", "first_fit_rms_mm", "n_fit"):
                if k in r2a:
                    self.stats["mech_foot_" + k] = float(r2a[k])
            if "dropped" in r2a:
                self.stats["mech_foot_fit_dropped"] = float(len(r2a["dropped"]))
        mem = m["membership"]
        if mem.get("tags") or mem.get("given_instances"):
            its = []
            for t in mem.get("tags", []):
                s = self.part_of.get(t)
                its.append({"id": t, "body": ref.member_body[t], "corners": ref.member_corners[t],
                            "group": (G.group_of(s["component"]) if (G is not None and s is not None) else None),
                            "segment": ref.parts[t]["segment"]})
            for pid in mem.get("given_instances", []):
                its.append({"id": pid, "body": ref.member_body[pid], "corners": ref.member_corners[pid],
                            "group": (G.group_of(anchor_comp[pid]) if (G is not None and pid in anchor_comp) else None),
                            "segment": ref.placements[pid]["segment"]})
            try:
                if G is None:
                    raise ValueError("the document's joints cannot be read")
                _s2b, r2b = mech.membership(G, bg, look, m, its, mc)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the membership", exc)
                r2b = [{"id": x["id"], "score": 0.0} for x in its]
            segof = {x["id"]: x["segment"] for x in its}
            out["membership"] = [(float(r["score"]), [segof[r["id"]]]) for r in r2b]
            rec["membership"] = r2b
            self.stats["mech_membership_mean"] = float(np.mean([r["score"] for r in r2b])) if r2b else 0.0
        if m["chains"]:
            try:
                _s3, r3 = mech.transmission(sk, ref.dphi_truth, ref.bore_senses, m, mc)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the transmission", exc)
                r3 = {ch: {"score": 0.0} for ch in m["chains"]}
            out["transmission"] = [(float(r3[ch]["score"]), list(m["transmission_segments"])) for ch in sorted(m["chains"])]
            rec["transmission"] = r3
            for ch, r in r3.items():
                if "p90_deg" in r:
                    self.stats["mech_chain_%s_p90_deg" % ch] = float(r["p90_deg"])
        if m["bars"]:
            try:
                bar_len = {c["tag"]: self.bar_length(c["tag"]) for c in m["bars"].values()}
                _s4, r4 = mech.closure(sk, bar_len, m, mc)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the closure", exc)
                r4 = {c["tag"]: {"score": 0.0} for c in m["bars"].values()}
            out["closure"] = [(float(r4[m["bars"][nm]["tag"]]["score"]), list(m["closure_segments"])) for nm in sorted(m["bars"])]
            rec["closure"] = r4
            for t, r in r4.items():
                if "error_mm" in r:
                    self.stats["mech_closure_%s_mm" % t] = float(r["error_mm"])
        return out

    def bar_length(self, tag):
        """The extent of a submitted made part along its own principal axis of least inertia (its long axis), from its
        tessellation; None when it is missing or has no tessellation."""
        s = self.part_of.get(tag)
        if s is None:
            return None
        p = self.part_placed(s)
        if p is None:
            return None
        _w, E = np.linalg.eigh(np.asarray(s["I"], float))
        proj = np.asarray(p.geo.V, float) @ E[:, 0]
        return float(proj.max() - proj.min())

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
        """The raw mass value (G-H8: times the whole leg's floor in evaluate). G-H3: a no_volume type is a point mass of
        its type mass at its mass centroid."""
        Mc = self.cfg["mass"]
        ref = self.ref

        def moved(V, c, I, T, rho):
            R = T[:3, :3]
            return (V * rho, apply(T, c), R @ I @ R.T, rho)

        def point(m, c):
            return (float(m), np.asarray(c, float), np.zeros((3, 3)), 1.0)

        exp = []
        for t, p in ref.parts.items():
            exp.append(moved(p["V"], p["com"], p["I"], np.eye(4), p["density"]))
        for pid, pl in ref.placements.items():
            typ = ref.types[pl["type"]]
            if typ["no_volume"]:
                exp.append(point(typ["mass_g"], apply(pl["W"], typ["mass_centroid"])))
                continue
            for c in typ["cons"]:
                exp.append(moved(c["V"], c["com"], c["I"], pl["W"], typ["density"]))
        act = []
        in_inst = {}
        for it in self.inst:
            for sid in it["slots"].values():
                in_inst[sid] = it["type"]
        for s in self.sub.solids:
            if s["tag"] is not None:
                rho = float(ref.parts[s["tag"]]["density"])
                act.append(moved(s["V"], s["com"], s["I"], self.A, rho))
            elif "match" in s:
                cl = ref.classes[s["match"]["cls"]]
                rep = cl["rep"]
                typ = ref.types[in_inst.get(s["sid"], rep["type"])]
                T = self.A @ s["match"]["T"]
                if typ["no_volume"]:
                    act.append(point(typ["mass_g"] * rep["V"] / typ["V"], apply(T, typ["mass_centroid"])))
                else:
                    act.append(moved(rep["V"], rep["com"], rep["I"], T, typ["density"]))
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
        """{"purchased": [(weight, score, segment)], "made": [(score, segment)]} (module docstring, rule 5; G-H8: each
        item times its segment's floor in evaluate): given instances against their supports and the made parts, made
        parts against each other; extra solids are obstacles too, without exemption. A no_volume constituent (the
        shell) overlaps nothing (G-H3)."""
        I = self.cfg["installation"]
        ref = self.ref
        if not hasattr(ref, "_fit_cache"):
            ref._fit_cache = {}
        cache = ref._fit_cache
        extras = [p for _s, p in self.extra_items()]
        pur = []
        for pid in sorted(ref.placements):
            pl = ref.placements[pid]
            w = self.type_weight(pl["type"])
            it = self.inst_of.get(pid)
            rec = self.res["purchased"].setdefault(pid, {"type": pl["type"]})
            if it is None:
                rec["fit"] = 0.0
                pur.append((w, 0.0, pl["segment"]))
                continue
            try:
                sc = self._fit_instance(pid, it, cache, extras, rec)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the fit of %s" % pid, exc)
                sc = 0.0
                rec["fit"] = 0.0
            pur.append((w, sc, pl["segment"]))
        made = []
        placed = {t: self.part_placed(self.part_of[t]) for t in ref.tags if self.part_of.get(t) is not None}
        for tag in ref.tags:
            p = placed.get(tag)
            seg = ref.parts[tag]["segment"]
            if p is None:
                made.append((0.0, seg))
                continue
            try:
                sc, excess = self._penetration_part(tag, p, placed, extras, cache)
                self.res["parts"][tag]["penetration_mm3"] = excess
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the penetration of %s" % tag, exc)
                sc = 0.0
            made.append((sc, seg))
        wsum = sum(w for w, _s, _g in pur)
        self.stats["fit_purchased"] = sum(w * s for w, s, _g in pur) / wsum if wsum > 0 else 0.0
        self.stats["fit_manufactured"] = float(np.mean([s for s, _g in made])) if made else 0.0
        return {"purchased": pur, "made": made}

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
            excess += max(0.0, self._instance_part_common(pid, mine, tag, p) - cache[key] - slack)
        for e in extras:
            if any(aabb_overlap(m.box, e.box) for m in mine):
                excess += max(0.0, group_common(mine, [e]) - slack)
        vol = sum(m.volume for m in mine)
        sc = contact * band(excess / max(vol, 1e-9), Fc["penetration_rel"])
        rec.update(fit=sc, contact=contact, penetration_mm3=excess)
        return sc

    def _instance_part_common(self, pid, mine, tag, p):
        """The volume common to a given instance's solids (`mine`, as submitted) and a made part (`p`, as submitted):
        the solids whose box meets the part's, through group_common. Memoised per grading, since the instance's fit and
        the part's penetration read the same pair."""
        key = (pid, tag)
        if key not in self._common_ip:
            self._common_ip[key] = group_common([m for m in mine if aabb_overlap(m.box, p.box)], [p])
        return self._common_ip[key]

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
        # the given instances too; the reference's own overlap of the same pair is exempt (the cache key of
        # _fit_instance); the denominator stays the made part's own volume
        for pid in sorted(self.inst_of):
            mine = self.instance_items(self.inst_of[pid])
            if not any(aabb_overlap(m.box, p.box) for m in mine):
                continue
            key = ("c", pid, tag)
            if key not in cache:
                cache[key] = group_common(ref.placement_items(pid), [ref.parts[tag]["placed"]])
            excess += max(0.0, self._instance_part_common(pid, mine, tag, p) - cache[key] - slack)
        for e in extras:
            if aabb_overlap(p.box, e.box):
                excess += max(0.0, common_volume(p.man, e.man) - slack)
        return band(excess / max(p.volume, 1e-9), Fc["penetration_rel"]), excess

    # ------------------------------------------------------------------ G-H3: the shell's samples inside made parts
    def shell_inside(self):
        """Statistics only: the share of the assigned shell's surface samples that lie inside a made part (the fit
        criterion cannot see a part pushed into the shell), next to the same share in the reference."""
        try:
            base = self.inst_of.get(self.ref.base_pid)
            if base is None:
                return
            items = [p for p in self.instance_items(base) if p.geo is not None]
            if not items:
                return
            pts = np.vstack([p.query()[0] for p in items])
            parts = [self.part_placed(self.part_of[t]) for t in self.ref.tags if self.part_of.get(t) is not None]
            self.stats["shell_samples_inside_made_share"] = inside_share(pts, [p for p in parts if p is not None])
            if not hasattr(self.ref, "_shell_inside_ref"):
                rp = [Placed(c["geo"], self.ref.placements[self.ref.base_pid]["W"], self.cfg)
                      for c in self.ref.types[self.ref.base_type]["cons"] if c["geo"] is not None]
                q = np.vstack([p.query()[0] for p in rp])
                self.ref._shell_inside_ref = inside_share(q, [self.ref.parts[t]["placed"] for t in self.ref.tags])
            self.stats["shell_samples_inside_made_share_reference"] = self.ref._shell_inside_ref
        except Exception as exc:  # noqa: BLE001 - a statistic only
            local_note(self.diags, "the shell sample statistic could not be computed (%s)" % str(exc)[:100])


def inside_share(points, placed, max_points=6000, seed=INSIDE_RECAST_SEED):
    """The share of `points` (at most max_points of them, evenly taken) inside any of the placed closed tessellations
    (G-H13: tested in chunks of points, at most INSIDE_CANDIDATES_MAX ray-triangle candidates per call; G-H14: the
    re-cast directions seeded per tessellation and chunk, the shared generator's state restored after each call)."""
    import trimesh
    gen = trimesh.util.random_generator()        # G-H14: the shared generator trimesh draws re-cast directions from
    P = np.asarray(points, float)
    if len(P) > max_points:
        P = P[np.linspace(0, len(P) - 1, max_points).astype(int)]
    inside = np.zeros(len(P), bool)
    for j, pl in enumerate(placed):
        if pl is None or pl.geo is None or pl.geo.man is None:
            continue
        lo, hi = pl.box[:3], pl.box[3:]
        sel = np.nonzero(np.all((P >= lo) & (P <= hi), axis=1) & ~inside)[0]
        if not len(sel):
            continue
        tm = trimesh.Trimesh(apply(pl.T, pl.geo.V), pl.geo.F, process=False)
        # G-H13: in chunks of points. A call casts two rays per point and keeps every (ray, triangle) pair whose boxes
        # meet, at most 2 x len(F) pairs per point; one call over all points held 25 GB for a document of about 5 500
        # drilled holes. Each ray's candidates, hits and parity are its own, so a point gets the single call's result;
        # a point whose two rays disagree is cast again in a random direction (trimesh's own rule, random before too).
        step = max(1, INSIDE_CANDIDATES_MAX // (2 * max(1, len(pl.geo.F))))
        for c, k in enumerate(range(0, len(sel), step)):
            part = sel[k:k + step]
            # G-H14: a fixed stream for this tessellation and chunk; the shared generator continues afterwards as if
            # this call had not drawn from it
            saved = gen.bit_generator.state
            gen.bit_generator.state = type(gen.bit_generator)(np.random.SeedSequence([int(seed), j, c])).state
            try:
                hit = tm.contains(P[part])
            finally:
                gen.bit_generator.state = saved
            inside[part[hit]] = True
        del tm
    return float(inside.mean()) if len(P) else 0.0



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
