"""Grader of dexterous_hand_cad_design_from_photos, rigid items: the extracted FreeCAD document of the four-digit hand
against the hand as modelled (reference/), in millimetres. Root, verify stage, agent gone; reads data only.

forked from tasks/ground/racecar_cad_design_from_photos/tests/grade.py (benchmark main 850dee3, sha256
7803e197bc86630058c6b410d169469d79d9eef55482c3f638c4d0f6ea8b87ba). Kept as they are: the helpers, the B-rep reading,
the tessellation (mesh_solid and its weld and hole fill), the surface distances, the exact and the sampled
registration, the reading of the given STEPs against their pins, the recognition of given parts by their full shape
(vertices that coincide within mesh.weld_mm counted once). The per-part fit follows the per-part free fit of
tasks/soft/tensegrity_cad_design_from_photos/tests/strutgrade/fit.py (sha256
08dfa78d0948d66e43c65d3ae4d296fab88276794c7995bc8bffe52a547eda83). The changes are named, each marked with its name in
square brackets where it is made: [names] names and formats (the extraction's tree nodes take the place of components);
[no base assembly] the given types are formed in one round and the alignment is on the palm ([alignment]); [reference]
the reference holds one B-rep per made kind with the placements of its tags, the reference meshes by the tier recorded
per kind (never a convex hull), the reference surface samples taken on the B-rep (samples.npz, with the marks of what
the photographs locate); [part fit] every made part is graded in its own best rigid fit onto its kind (an exact copy of
the kind's solid, recognised on its vertex set, takes the kind's own tessellation); [mesher tiers] a submitted part's
tessellation tries finer settings before its convex hull stands for it; [shape] the shape score with size-class bands
and square-root-of-area weights; [holes] holes in the part's own fit, a submitted step eligible by its arc and depth,
the steps of a submitted hole with more eligible steps than its reference hole read by structure, class weights; [extras]
a submitted step that matches no step of the true part counts against the part's hole score; [located] the reference's
located marks on hinges and installed places (an unlocated item is not graded);
[groups] rigid groups and palm groups; [alignment] the palm alignment; [assembly] the assembly criterion;
[combination] the combination with the shape floor, the mechanism criteria taken from the mechanism module (handtree.py)
when it is there.

    python3 -B grade.py EXTRACT_DIR REF_DIR PURCHASED_ROOT OUT_JSON [--config grader_config.json] [--rigid-only]

EXTRACT_DIR is fcstd_extract.py's output (model.json + one world-frame BREP per solid). REF_DIR holds reference.json
(schema hand-cad-reference/1), parts/<kind>.brep (the made kinds, each in its own frame) with their meshes where a
kind's tier needs a shipped one, samples.npz (the B-rep surface samples of every kind and their marks) and
purchased_pins.json (the STEP of every given type, pinned by sha256). PURCHASED_ROOT is where those STEPs are read (the
image's /opt/task-assets/payload; a file whose sha256 differs from its pin is a task error). Every number is in
grader_config.json. --rigid-only grades without the mechanism module (development: the kinematic, mass, fit and
interference criteria are then 0 and reported as not computed). OUT_JSON: {"ok", "design_score", "criteria",
"criteria_raw", "checks", "statistics", "parts", "purchased", "diagnostics"} (or {"ok": false, "task_error", "error"}).

The rules of the rigid items (the instruction's grading section states them):

 1. What is graded: the solids inside the document's Assembly (its members, App::Part containers, sub-assemblies,
    folders and links). A solid is usable when it is closed and has a volume; there is no validity check. The
    document scores 0 only when it cannot be read, holds no solid, or holds no palm group (rule 4). A tagged leaf that
    is not a PartDesign Body of exactly one closed solid, or a tag on more than one leaf, gets no credit and its solids
    count as extra solids; open solids and loose faces of untagged leaves are left out; each is reported. A grading
    item that fails scores 0 and counts in grader_item_errors; a failure outside every item fails the run.
 2. Given parts are recognised by their full shape, whatever their label: every untagged solid against every given
    type (volume within recognition.volume_rel, then a rigid registration: exact on the B-rep vertex set for an
    unchanged import, else sampled). Instances go to the reference placements one to one per type (Hungarian on
    position, in the aligned frame); copies left over are extra solids.
 3. Every tagged part is fitted onto its kind on its own, wherever it sits: an exact copy of the kind's solid by its
    vertex set; else the principal-axis starts of its exact inertia, refined by ICP, the pose chosen by the shape
    score. Shape, holes and the part's own placement are read in that fit.
 4. Rigid groups: the leaves joined by unsuppressed Fixed joints, and the members of a rigid container. A group is a
    palm group when the reference volume of the palm parts it holds is larger than that of the digits' structural
    parts it holds. Grounding joins nothing.
 5. The alignment: one rigid transform (no scale, no mirror) from the submission onto the true palm, decided among
    hypotheses seeded by every full-pose member (a board, the hub, a fitted palm part) and by every pair of palm
    servos (output centres, undirected axes); members are the recognised given instances of alignment.member_types in
    palm groups and the palm made parts; each hypothesis is scored by the Huber sum (alignment.huber_mm) of the
    members' residuals after the one-to-one assignment within types (a servo's residual is that of its undirected
    output axis line); the best is refined by iteratively re-weighted least squares over the members within
    alignment.inlier_mm. With no member within alignment.inlier_mm the best hypothesis stands, its residual is taken
    over every member and a diagnostic says so.
 6. Criteria of this file (weights in config), each in [0, 1]:
    shape     per reference tag: w_surface band(two-way p90 surface distance; the kind's size-class band) + w_iou
              rising(IoU), in the part's fit; only reference surface marked located enters the distance; a part whose
              tessellation is no closed manifold of its volume enters the IoU as its convex hull; mean weighted by the
              square root of the kind's surface area; a missing part 0.
    features  per graded hole step of the kind, matched one to one with a submitted step in the part's fit:
              band(axis offset at the step's mid-depth) x band(radius error) x band(axis angle); a submitted step is
              the union of coaxial concave cylinders (or cones) of one radius, eligible when its arc spans at least
              features.min_span_deg and its depth is at least max(features.min_depth_mm, features.depth_ratio x the
              reference step's depth); a submitted hole is a set of its steps whose axis lines lie within the zero
              ends of the offset and angle bands of each other; when a submitted hole has more steps eligible for a
              reference hole than that hole has steps and one of them lies within those zero ends of one of its
              steps, the steps read against it are chosen by structure, never by their score: the deepest first
              (depths within features.coaxial_merge.offset_mm are a tie), ties to the larger radius, then document
              order; the steps left over are not read against it and earn nothing (counted in hole_steps_unread);
              equal or fewer steps keep the one-to-one matching; a hole scores the mean of its steps, a part the
              class-weighted mean of its
              located holes; the criterion is the mean over the tags with holes, weighted like shape. Extras
              ([extras]): a submitted step that is a features.extras.kinds wall spanning at least
              features.extras.min_span_deg, at least features.extras.min_depth_mm deep, of a radius within
              features.extras.radius_mm, that is neither credited (assigned with a score above 0 to a located step) nor
              matched one to one (position and angle within the zero ends of the offset and angle bands, the radius
              free) to another step of the true part, located or not, is an extra; a step not read by the structural
              choice is an extra; every extra adds to the part's denominator the largest class weight per step of one
              of its located holes (hole_steps_extra); a step matched to a step of a hole that is not located is
              neither credited nor counted (hole_steps_at_unlocated). A face that
              is no exact cylinder spans the arc its own closure gives (a face closed around its axis is a full
              wall) or, if it is not closed, the arc a dense sampling of it covers; such faces of one hole merge
              within the fit's own tolerance.
    assembly  w_placement x placement + w_installation x installation + w_connection x connection: placement per
              tag: band(centroid position) x (1 - w + w band(orientation modulo the kind's own symmetries)), the fitted
              pose against the reference pose in the aligned frame, weighted like shape; installation per reference
              given placement: band(position of the type's origin) x (1 - w + w band(orientation)), an axis-only type
              by its axis (sense-free when flagged), else modulo the type's own symmetries; by type weights;
              ([located]) a tag or a given placement the reference marks place_located false is left out of its
              item's numerator and denominator (placement_items_not_graded, installation_items_not_graded) and is
              no member of the alignment;
              connection: the weighted share (made parts by shape weight, given instances by type weight, each half)
              of the largest set of rigid groups that unsuppressed Fixed, Revolute, Cylindrical and Ball joints connect
              and that contains a palm group.
 7. design_score = G x [w_shape S + F_s (w_kinematic K + w_features H + w_assembly A) + F_s (w_mass M' + w_fit Fi' +
    w_interference I')], F_s = rising(S; floors.lo, floors.hi); K, M', Fi', I' come from the mechanism module (with
    the digits' floors applied there); the criteria before F_s are in criteria_raw.
 8. The statistics: every verdict carries the same names (metric_template), a statistic that was not measured at its
    stated default; the parts on the convex-hull fallback are counted (parts_on_hull) and named in a diagnostic; a
    joint whose placement is not a finite number is not read, and counted (joints_nonfinite) and named.
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
CRITERIA = ["shape", "kinematic", "features", "assembly", "mass", "fit", "interference"]           # [combination]
MECHANISM_CRITERIA = ["kinematic", "mass", "fit", "interference"]
# the statistics of this file, each at the value it takes when it is not measured (every verdict carries them all)
GRADE_STATS = {
    "alignment_hypotheses": 0.0, "alignment_inliers": 0.0, "alignment_members": 0.0, "alignment_residual_mm": -1.0,
    "alignment_runner_up_ratio": 0.0, "assembly_connection": 0.0, "assembly_installation": 0.0,
    "assembly_placement": 0.0, "connected_sets_with_palm": 0.0, "empty_links": 0.0, "fixed_joints_read": 0.0,
    "grader_item_errors": 0.0, "hole_steps_at_unlocated": 0.0, "hole_steps_extra": 0.0, "hole_steps_found": 0.0,
    "hole_steps_total": 0.0, "hole_steps_unread": 0.0, "ignored_bad_solids": 0.0, "installation_items_not_graded": 0.0,
    "placement_items_not_graded": 0.0,
    "ignored_non_solid_leaves": 0.0, "ignored_unreadable_leaves": 0.0, "joints_nonfinite": 0.0,
    "mechanism_module": 0.0, "n_solids": 0.0, "n_tagged": 0.0, "nonfinite_leaves": 0.0, "palm_groups": 0.0,
    "palm_split": 0.0, "parts_exact_copies": 0.0, "parts_fitted": 0.0, "parts_on_hull": 0.0,
    "purchased_instances_assigned": 0.0, "purchased_instances_found": 0.0, "purchased_instances_left_over": 0.0,
    "purchased_solids_approx": 0.0, "purchased_solids_exact": 0.0, "purchased_solids_unattached": 0.0,
    "reference_samples_brep": 0.0, "shape_floor": 0.0, "tags_duplicated": 0.0, "tags_invalid": 0.0,
    "tags_reassigned": 0.0}


class _SpecFile(object):
    """reference.json read as data (the mechanism block and the ids), for the names of the statistics."""

    def __init__(self, ref_dir):
        self.dir = ref_dir
        with open(os.path.join(ref_dir, "reference.json")) as fh:
            self.spec = json.load(fh)
        self.parts = dict(self.spec.get("parts") or {})
        self.placements = {p["id"]: {"type": p["type"]} for p in self.spec.get("placements") or []}


def metric_template(ref):
    """{statistic name: its value when not measured} of every statistic grade.py and the mechanism module write; ref is
    a loaded Reference or a reference folder."""
    import hand_task
    import handtree
    out = dict(GRADE_STATS)
    out.update(handtree.metric_defaults(hand_task.load_spec(_SpecFile(ref) if isinstance(ref, str) else ref)))
    return out


def nonfinite_joints(model):
    """The labels of the joints whose own Placement1 or Placement2 holds a number that is not finite: such a joint is
    not read at all (Groups.usable: it joins nothing, connects nothing and is no hinge)."""
    bad = []
    for j in model.get("joints", []):
        if j.get("suppressed") or j.get("foreign") or j.get("error"):
            continue
        if not (j.get("placement1_finite", True) and j.get("placement2_finite", True)):
            bad.append(str(j.get("label") or j.get("name") or "?"))
    return bad
SCHEMA = "hand-cad-reference/1"                                                                     # [names]
EXTRACT_FORMAT = "hand-fcstd-extract/"


# ============================================================================ small helpers (as forked)
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
    span = _fitted_span(f, ad, centre, e1, e2, u0, u1, v0, v1)
    return {"kind": "cylinder", "point": centre, "dir": d, "radius": float(r), "concave": concave, "span_rad": span,
            "fitted": True}


def _face_closed(f):
    """True when the face closes on itself (an edge of its boundary is a seam of the face): a full wall."""
    from OCP.BRep import BRep_Tool
    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopoDS import TopoDS
    ex = TopExp_Explorer(f, TopAbs_EDGE)
    while ex.More():
        if BRep_Tool.IsClosed_s(TopoDS.Edge_s(ex.Current()), f):
            return True
        ex.Next()
    return False


def _fitted_span(f, ad, centre, e1, e2, u0, u1, v0, v1, n=37):
    """The arc (radians) a fitted wall covers about its axis: the full turn when the face closes on itself; else 2 pi
    less the largest angular gap of an n x n sampling of the face (a few parameter samples would read a closed wall
    at no more than 300 degrees)."""
    if _face_closed(f):
        return 2.0 * math.pi
    ang = []
    for u in np.linspace(u0, u1, n):
        for v in np.linspace(v0, v1, n):
            pv = ad.Value(float(u), float(v))
            q = np.array([pv.X(), pv.Y(), pv.Z()]) - centre
            ang.append(math.atan2(float(q @ e2), float(q @ e1)))
    a = np.sort(np.asarray(ang, float))
    if len(a) < 2:
        return 0.0
    gaps = np.diff(a)
    wrap = 2.0 * math.pi - float(a[-1] - a[0])
    return float(max(0.0, 2.0 * math.pi - max(float(gaps.max()) if len(gaps) else 0.0, wrap)))


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



def kabsch_w(P, Q, w):
    """Proper rigid T minimising sum w |T(P) - Q|^2 over corresponding rows ([alignment])."""
    P, Q = np.asarray(P, float), np.asarray(Q, float)
    w = np.asarray(w, float).reshape(-1)
    w = w / max(float(w.sum()), 1e-300)
    cp, cq = w @ P, w @ Q
    H = (P - cp).T @ ((Q - cq) * w[:, None])
    U, _s, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0])
    R = Vt.T @ D @ U.T
    return make_T(R, cq - R @ cp)


def surface_area(shape):
    from OCP.BRepGProp import BRepGProp
    from OCP.GProp import GProp_GProps
    g = GProp_GProps()
    BRepGProp.SurfaceProperties_s(shape, g)
    return float(g.Mass())


def outer_shell_solid(solid):
    """The solid bounded by its outer shell only (internal voids filled: a given type's collider)."""
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeSolid
    from OCP.BRepClass3d import BRepClass3d
    from OCP.TopAbs import TopAbs_SHELL
    from OCP.TopExp import TopExp_Explorer
    ex = TopExp_Explorer(solid, TopAbs_SHELL)
    n = 0
    while ex.More():
        n += 1
        ex.Next()
    if n <= 1:
        return solid
    s = BRepBuilderAPI_MakeSolid(BRepClass3d.OuterShell_s(solid)).Solid()
    if mass_props(s)[0] < 0:
        s.Reverse()
    return s


def pose_probe(T, P):
    return apply(T, P)


def box_probe(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def pose_distance(Ta, Tb, P):
    return float(np.max(np.linalg.norm(apply(Ta, P) - apply(Tb, P), axis=1))) if len(P) else 0.0


# ============================================================================ [mesher tiers] the mesher's tiers
def _tier_single(shape, defl, ang, weld_mm, fill, tmpdir):
    """One tessellation in the solid's principal frame (as mesh_solid's first pose): (V, F, manifold or None)."""
    T = _canonical_poses(shape)[0]
    got = _stl_mesh(_moved_copy(shape, T), defl, ang, tmpdir)
    if got is None:
        return None
    V, F = weld(got[0], got[1], weld_mm)
    V, F, _filled = fill_holes(V, F, int(fill))
    if len(F) < 4:
        return None
    V = apply(inv(T), V)
    return V, F, to_manifold(V, F)


def _face_soup(shape, defl, ang):
    """The OCC tessellation face by face in double precision (no STL round trip): (V, F)."""
    from OCP.BRep import BRep_Tool
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.IMeshTools import IMeshTools_Parameters
    from OCP.TopAbs import TopAbs_REVERSED
    from OCP.TopLoc import TopLoc_Location
    c = fresh_copy(shape)
    p = IMeshTools_Parameters()
    p.Deflection, p.Angle, p.Relative, p.InParallel = float(defl), float(ang), False, False
    BRepMesh_IncrementalMesh(c, p)
    V, F, off = [], [], 0
    for f in faces_of(c):
        loc = TopLoc_Location()
        tri = BRep_Tool.Triangulation_s(f, loc)
        if tri is None:
            continue
        T = loc.Transformation()
        for k in range(1, tri.NbNodes() + 1):
            q = tri.Node(k).Transformed(T)
            V.append((q.X(), q.Y(), q.Z()))
        rev = f.Orientation() == TopAbs_REVERSED
        for k in range(1, tri.NbTriangles() + 1):
            a, b, cc = tri.Triangle(k).Get()
            F.append((off + a - 1, off + cc - 1, off + b - 1) if rev else (off + a - 1, off + b - 1, off + cc - 1))
        off += tri.NbNodes()
    return np.array(V, float).reshape(-1, 3), np.array(F, np.int64).reshape(-1, 3)


def _tier_merge(shape, defl, ang):
    """Tier 2m: the face-by-face soup, its boundary nodes merged by manifold3d's own merge (only merges that keep the
    mesh manifold), centred on the centroid for float32 precision: (V, F, manifold or None)."""
    import manifold3d as m3
    V, F = _face_soup(shape, defl, ang)
    if len(F) < 4:
        return None
    c = mass_props(shape)[1]
    mesh = m3.Mesh(vert_properties=np.ascontiguousarray(V - c, np.float32), tri_verts=np.ascontiguousarray(F, np.uint32))
    mesh.merge()
    man = m3.Manifold(mesh)
    if str(man.status()) != "Error.NoError" or man.is_empty():
        return None
    out = man.to_mesh()
    V2 = np.asarray(out.vert_properties, float)[:, :3] + c
    F2 = np.asarray(out.tri_verts, np.int64)
    return V2, F2, man.translate(tuple(float(x) for x in c))


def mesh_with_tier(shape, tier, cfg, tmpdir, seed, volume, tol):
    """(Geo, error) of a solid by one named tier of the meshing rule: "1" the standard mesher (mesh_solid, its poses and
    retries), "2" finer settings with exact then welded merging, "2m" the face-by-face merge. The Geo is accepted when
    its manifold's volume is within tol of `volume`; else (None, error)."""
    M = cfg["mesh"]
    if tier == "1":
        c1 = dict(cfg)
        c1["mesh"] = dict(M, volume_tol_rel=tol)
        g = mesh_solid(shape, c1, tmpdir, seed, volume=volume)
        if g is not None and g.man is not None and abs(g.volume / volume - 1.0) <= tol:
            return g, None
        return None, "tier 1: %s" % ("no tessellation" if g is None else "%s, volume off by %.3f %%" % (
            "a manifold" if g.man is not None else "no manifold", 100 * abs(g.volume / volume - 1)))
    tiers = M["tiers"]
    errs = []
    if tier == "2":
        for defl, ang in tiers["2"]["settings"]:
            for wmm in tiers["2"]["merge_mm"]:
                got = _tier_single(shape, defl, ang, wmm, M["fill_max_loop"], tmpdir)
                if got is None or got[2] is None:
                    errs.append("%g/%g/%g: no manifold" % (defl, ang, wmm))
                    continue
                err = abs(float(got[2].volume()) / volume - 1.0)
                if err <= tol:
                    return Geo(got[0], got[1], got[2], ["tier 2 at %g mm, merge %g mm" % (defl, wmm)], seed), None
                errs.append("%g/%g/%g: volume off by %.3f %%" % (defl, ang, wmm, 100 * err))
        return None, "tier 2: " + "; ".join(errs)
    if tier == "2m":
        for defl, ang in tiers["2m"]["settings"]:
            got = _tier_merge(shape, defl, ang)
            if got is None:
                errs.append("%g/%g: no manifold" % (defl, ang))
                continue
            err = abs(float(got[2].volume()) / volume - 1.0)
            if err <= tol:
                return Geo(got[0], got[1], got[2], ["tier 2m at %g mm" % defl], seed), None
            errs.append("%g/%g: volume off by %.3f %%" % (defl, ang, 100 * err))
        return None, "tier 2m: " + "; ".join(errs)
    return None, "unknown tier %r" % tier


def mesh_submitted(shape, cfg, tmpdir, seed, volume):
    """[mesher tiers] a submitted solid's Geo: the tiers of mesh.submitted_tiers in turn, each within mesh.volume_tol_rel; when
    none gives a closed manifold of the solid's volume, the closest tessellation with no manifold (booleans then use
    its convex hull, as before)."""
    tol = float(cfg["mesh"]["volume_tol_rel"])
    for tier in cfg["mesh"]["submitted_tiers"]:
        try:
            g, _err = mesh_with_tier(shape, str(tier), cfg, tmpdir, seed, volume, tol)
        except Exception:  # noqa: BLE001 - the next tier
            g = None
        if g is not None:
            g.tier = str(tier)
            return g
    g = mesh_solid(shape, cfg, tmpdir, seed, volume=volume)
    if g is not None:
        g.man = None
        g.notes = list(g.notes) + ["no tier gives a closed manifold of the solid's volume: booleans use its convex hull"]
        g.tier = "hull"
    return g


def geo_moved(geo, T, seed):
    """A Geo moved rigidly by T (an exact copy of a reference kind takes the kind's own tessellation, [part fit])."""
    T = np.asarray(T, float)
    V = apply(T, geo.V)
    man = geo.man.transform(np.ascontiguousarray(T[:3, :], np.float32)) if geo.man is not None else None
    g = Geo(V, geo.F.copy(), man, list(geo.notes), seed)
    g.volume = geo.volume
    g.tier = getattr(geo, "tier", None)
    return g


def load_mesh_npz(path):
    with np.load(path) as z:
        return np.asarray(z["V"], float), np.asarray(z["F"], np.int64)


# ============================================================================ the reference ([reference])
class TaskError(Exception):
    """The grader's own reference or the image's given files are unusable: a broken task, never a zero."""


def located_mark(rec, key, what):
    """[located] A located mark of the reference: absent counts as located; a value that is no JSON boolean is a
    task error (the grader does not guess)."""
    v = rec.get(key, True)
    if not isinstance(v, bool):
        raise TaskError("%s: %s is %r, not true or false" % (what, key, v))
    return v


class Reference(object):
    """reference.json: the made kinds (B-rep in the kind's own frame, its tessellation by its tier, mass properties,
    area, own symmetries, B-rep surface samples with their located marks, hole steps), the 52 tags with their kind and
    reference pose, the given types (from the pinned STEPs) and their placements, the palm and structural lists."""

    def __init__(self, ref_dir, purchased_root, cfg, tmpdir, log):
        self.dir, self.cfg, self.tmp, self.log = ref_dir, cfg, tmpdir, log
        with open(os.path.join(ref_dir, "reference.json")) as fh:
            self.spec = json.load(fh)
        if self.spec.get("schema") != SCHEMA:
            raise TaskError("reference.json schema %r" % self.spec.get("schema"))
        seed = int(cfg["sampling"]["seed"])
        pending = self._start_purchased(purchased_root)
        samples = None
        sp = self.spec.get("samples")
        if sp:
            path = os.path.join(ref_dir, sp["file"])
            if sha256_file(path) != sp["sha256"]:
                raise TaskError("the reference samples do not match their pin")
            samples = np.load(path)
        self.samples_source = "brep" if samples is not None else "mesh"
        tol_ref = float(cfg["mesh"]["reference_volume_tol_rel"])
        self.kinds = {}
        for i, (kid, k) in enumerate(sorted(self.spec["kinds"].items())):
            shape = read_brep(os.path.join(ref_dir, k["brep"]))
            sols = solids_of(shape)
            if len(sols) != 1:
                raise TaskError("reference kind %s has %d solids" % (kid, len(sols)))
            s = sols[0]
            V, c, I = mass_props(s)
            tier = str(k["mesh"]["tier"])
            if k["mesh"].get("file"):
                Vm, Fm = load_mesh_npz(os.path.join(ref_dir, k["mesh"]["file"]))
                man = to_manifold(Vm, Fm)
                geo = Geo(Vm, Fm, man, ["tier %s (shipped mesh)" % tier], seed + 1000 + i) if man is not None else None
                err = None if geo is not None else "the shipped mesh is no manifold"
                if geo is not None and abs(geo.volume / V - 1.0) > tol_ref:
                    geo, err = None, "the shipped mesh's volume is off by %.3f %%" % (100 * abs(geo.volume / V - 1))
            else:
                geo, err = mesh_with_tier(s, tier, cfg, tmpdir, seed + 1000 + i, V, tol_ref)
            if geo is None:
                raise TaskError("reference kind %s does not mesh as a closed manifold by its tier %s (%s)" % (kid, tier, err))
            geo.tier = tier
            verts = unique_vertices(s, float(cfg["mesh"]["weld_mm"]))
            ext = verts.max(axis=0) - verts.min(axis=0) if len(verts) else np.zeros(3)
            rec = {"id": kid, "name": k.get("name"), "density": float(k["density_g_mm3"]), "shape": s, "V": V, "com": c,
                   "I": I, "geo": geo, "verts": verts, "size": float(np.linalg.norm(ext)), "area": surface_area(s),
                   "size_class": k["size_class"], "features": list(k.get("features", [])), "tier": tier}
            for h in rec["features"]:
                located_mark(h, "located", "kind %s hole %s" % (kid, h.get("id")))
            rec["weight"] = math.sqrt(rec["area"])                                                      # [shape]
            sym = register_exact(rec, rec, float(cfg["recognition"]["exact_tol_mm"]), all_solutions=True) or []
            rec["symmetries"] = [g[0] for g in sym] or [np.eye(4)]
            if samples is not None:
                P = np.asarray(samples[kid + "/points"], float)
                N = np.asarray(samples[kid + "/normals"], float)
                L = np.asarray(samples[kid + "/located"], bool) if (kid + "/located") in samples.files else np.ones(len(P), bool)
                if len(P) != len(N) or len(P) != len(L) or len(P) < 100:
                    raise TaskError("reference samples of %s are malformed" % kid)
            else:
                P, N, _t = geo.dense(cfg)
                L = np.ones(len(P), bool)
            from scipy.spatial import cKDTree
            rec["samples"] = (P, N, cKDTree(P), L)
            nq = int(min(cfg["sampling"]["query_max"], max(cfg["sampling"]["query_min"],
                                                            cfg["sampling"]["query_per_mm2"] * rec["area"])))
            idx = np.nonzero(L)[0]
            rec["query_idx"] = idx[np.linspace(0, len(idx) - 1, min(nq, len(idx))).astype(int)] if len(idx) else idx
            self.kinds[kid] = rec
        self.parts = {}
        for tag, p in sorted(self.spec["parts"].items()):
            if p["kind"] not in self.kinds:
                raise TaskError("tag %s has an unknown kind %s" % (tag, p["kind"]))
            self.parts[tag] = {"tag": tag, "kind": p["kind"], "W": np.asarray(p["local_to_world"], float),
                               "located": located_mark(p, "place_located", "tag %s" % tag)}
        self.tags = sorted(self.parts)
        self.kind_tags = {}
        for t in self.tags:
            self.kind_tags.setdefault(self.parts[t]["kind"], []).append(t)
        self._load_purchased(pending)
        self.placements = {}
        for pl in self.spec["placements"]:
            if pl["type"] not in self.types:
                raise TaskError("placement %s has an unknown type %s" % (pl["id"], pl["type"]))
            self.placements[pl["id"]] = {"id": pl["id"], "type": pl["type"], "W": np.asarray(pl["local_to_world"], float),
                                         "located": located_mark(pl, "place_located", "placement %s" % pl["id"])}
        palm = self.spec["palm"]
        self.palm_kinds = set(palm["kinds"])
        self.palm_types = set(palm["types"])
        self.palm_placements = set(palm["placements"])
        self.structural_kinds = set(self.spec["structural"]["kinds"])
        for k in list(self.palm_kinds) + list(self.structural_kinds):
            if k not in self.kinds:
                raise TaskError("palm / structural kind %s is not a kind" % k)
        for p in self.palm_placements:
            if p not in self.placements:
                raise TaskError("palm placement %s is not a placement" % p)

    # ------------------------------------------------------------------ given types
    def _start_purchased(self, root):
        with open(os.path.join(self.dir, "purchased_pins.json")) as fh:
            pins = json.load(fh)
        jobs = []
        for tname, t in sorted(self.spec["purchased"].items()):
            pin = pins["types"][tname]
            path = os.path.join(root, pin["step"])
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
        tol_g = float(self.cfg["mesh"]["given_volume_tol_rel"])
        tol = float(self.cfg["recognition"]["exact_tol_mm"])
        n = 0
        for tname, t in sorted(self.spec["purchased"].items()):
            pin = pins["types"][tname]
            files = breps[tname]
            if len(files) != len(pin["solids"]):
                raise TaskError("%s: the given STEP has %d solids, the pin %d" % (tname, len(files), len(pin["solids"])))
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
                coll = outer_shell_solid(s)                    # the collider: voids filled
                Vo = mass_props(coll)[0]
                tier = str(t.get("mesh", {}).get("tier", "1"))
                geo, err = mesh_with_tier(coll, tier, self.cfg, self.tmp, seed + 5000 + n, Vo, tol_g)
                if geo is None:
                    raise TaskError("given type %s does not mesh as a closed manifold by its tier %s (%s)" % (tname, tier, err))
                geo.tier = tier
                verts = unique_vertices(s, float(self.cfg["mesh"]["weld_mm"]))
                ext = verts.max(axis=0) - verts.min(axis=0) if len(verts) else np.zeros(3)
                cons[order.index(cid)] = {"id": cid, "type": tname, "k": order.index(cid), "V": V, "com": c, "I": I,
                                          "verts": verts, "size": float(np.linalg.norm(ext)), "geo": geo,
                                          "role": byid[cid].get("source_role"), "light": False}
            if any(c is None for c in cons):
                raise TaskError("%s: a constituent has no given solid" % tname)
            vol = sum(c["V"] for c in cons)
            cen = sum(c["com"] * c["V"] for c in cons) / vol
            anchor = [c["id"] for c in cons].index(t["anchor"])
            sym = t.get("symmetry") or {}
            own = register_exact(cons[anchor], cons[anchor], tol, all_solutions=True) or []
            self.types[tname] = {"name": tname, "pn": t.get("part_number"), "mass_g": float(t["mass_g"]),
                                 "density": float(t.get("density_g_mm3", 0.0)), "cons": cons, "anchor": anchor, "V": vol,
                                 "centroid": cen, "symmetry": sym,
                                 "axis_only": bool(sym.get("axis_only")), "sense_free": bool(sym.get("sense_free")),
                                 "sym_axis": np.asarray(sym.get("axis_local", [0, 0, 1]), float),
                                 "origin": np.asarray(t.get("origin_local", [0.0, 0.0, 0.0]), float),
                                 "output_axis": t.get("output_axis_local"),
                                 "own_symmetries": [g[0] for g in own] or [np.eye(4)]}
        self._classes()

    def _classes(self):
        """Congruence classes of all constituents (exact registration on their vertex sets), as forked."""
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
    """Read the given STEPs in spawned worker processes, started at once (as forked)."""

    def __init__(self, jobs, tmpdir, workers):
        out_dir = os.path.join(tmpdir, "given")
        os.makedirs(out_dir, exist_ok=True)
        self.jobs = sorted([(t, p, out_dir) for t, p in jobs], key=lambda j: -os.path.getsize(j[1]))
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


# ============================================================================ the submission ([names])
def local_note(diags, msg):
    """A local diagnostic ("Local: ..."), at most 25 of them."""
    if sum(1 for d in diags if d.startswith("Local:")) < 25:
        diags.append("Local: " + msg)


class Submission(object):
    """The extracted document: every solid with its tag (if any), its tree node, mass properties and vertices.
    Only an unreadable document and a document without any usable solid stop the grade here (rule 1)."""

    def __init__(self, ex_dir, ref, cfg, tmpdir, diags):
        self.ok = False
        self.solids = []
        self.cfg, self.tmp = cfg, tmpdir
        self.stats = st = {"tags_invalid": 0, "tags_duplicated": 0, "ignored_non_solid_leaves": 0,
                           "ignored_bad_solids": 0, "ignored_unreadable_leaves": 0, "empty_links": 0,
                           "nonfinite_leaves": 0}
        path = os.path.join(ex_dir, "model.json")
        if not os.path.isfile(path):
            diags.append("No readable FreeCAD document.")
            return
        try:
            with open(path) as fh:
                model = json.load(fh)
        except ValueError as exc:
            raise TaskError("the extraction's model.json is not JSON: %s" % exc)
        if not isinstance(model, dict) or not str(model.get("format", "")).startswith(EXTRACT_FORMAT):
            raise TaskError("the extraction's model.json has an unknown format")
        if model.get("extractor_failure"):
            raise TaskError("the FreeCAD reader failed: " + " | ".join(str(e) for e in model.get("errors", []))[-1500:])
        self.model = model
        if model.get("errors"):
            diags.extend("The document cannot be read: %s" % str(e)[:300] for e in model["errors"][:5])
            if any("no Assembly" in str(e) or "no top-level Assembly" in str(e) for e in model["errors"]):
                diags.append("Objects outside an Assembly are not graded.")
            return
        G = cfg["gates"]
        ex_real = os.path.realpath(ex_dir)
        tagged, untagged = {}, []

        def note(msg):
            local_note(diags, msg)

        self.leaves = {}
        for comp in model.get("components", []):
            for leaf in comp.get("leaves", []):
                self.leaves[leaf["node"]] = leaf
                label = (leaf.get("label") or "").strip()
                tag = label if label in ref.parts else None
                where = "%s (%s)" % (label or leaf.get("linked_object"), leaf.get("type"))
                if leaf.get("empty_link"):
                    st["empty_links"] += 1
                    note("%s is a link whose target has no readable shape (an object of a document that is not in the "
                         "submission folder?): nothing of it is read" % where)
                    if tag:
                        tagged.setdefault(tag, []).append((leaf, [], False))
                    continue
                if not leaf.get("placement_finite", True):
                    st["nonfinite_leaves"] += 1
                    note("%s has a placement that is not a finite number (%s): left out" % (
                        where, leaf.get("placement_nonfinite")))
                    if tag:
                        tagged.setdefault(tag, []).append((leaf, [], False))
                    continue
                if leaf.get("error"):
                    st["ignored_unreadable_leaves"] += 1
                    note("the shape of %s cannot be read (%s): left out" % (where, str(leaf["error"])[:120]))
                    if tag:
                        tagged.setdefault(tag, []).append((leaf, [], False))
                    continue
                sols = leaf.get("solids", [])
                good = [x for x in sols if x.get("closed")]
                free = int(leaf.get("n_free_faces", 0) or 0)
                if tag is not None:
                    single = (leaf.get("type") == "PartDesign::Body" and len(sols) == 1 and len(good) == 1 and free == 0)
                    tagged.setdefault(tag, []).append((leaf, good, single))
                    continue
                if not sols:
                    st["ignored_non_solid_leaves"] += 1
                    if free:
                        note("%s has open faces and no solid: left out" % where)
                    continue
                if free or len(good) < len(sols):
                    st["ignored_bad_solids"] += (len(sols) - len(good)) + (1 if free else 0)
                    note("%s has %d open solid(s)%s: those are left out" % (
                        where, len(sols) - len(good), " and loose faces" if free else ""))
                for x in good:
                    untagged.append((leaf, x))

        def add(leaf, x, tag, demoted=None):
            p = os.path.realpath(os.path.join(ex_dir, x["brep"]))
            if os.path.commonpath([ex_real, p]) != ex_real:
                raise TaskError("the extractor returned an escaping shape path")
            rec = {"sid": len(self.solids), "tag": tag, "label": (leaf.get("label") or "").strip(), "node": leaf["node"],
                   "component": leaf.get("top"), "leaf_type": leaf.get("type"), "file": p}
            if demoted:
                rec["demoted"] = demoted
            self.solids.append(rec)
            return rec

        for tag in sorted(tagged):
            entries = tagged[tag]
            if len(entries) > 1:
                st["tags_duplicated"] += 1
                note("tag %s is on %d leaves: none of them is matched; their solids count as extra solids" % (tag, len(entries)))
                for leaf, good, _single in entries:
                    for x in good:
                        add(leaf, x, None, demoted="duplicate tag " + tag)
                continue
            leaf, good, single = entries[0]
            if single:
                add(leaf, good[0], tag)
                continue
            st["tags_invalid"] += 1
            if leaf.get("empty_link") or not leaf.get("placement_finite", True) or leaf.get("error"):
                continue
            n = len(leaf.get("solids", []))
            note("%s is a %s with %d solid(s), %d closed: not a PartDesign Body of exactly one closed solid, so the tag "
                 "gets no credit; its closed solids count as extra solids" % (tag, leaf.get("type"), n, len(good)))
            for x in good:
                add(leaf, x, None, demoted="invalid tagged leaf " + tag)
        for leaf, x in untagged:
            add(leaf, x, None)
        if len(self.solids) > int(G["max_solids"]):
            diags.append("Too many solids in the Assembly (%d)." % len(self.solids))
            self.solids = []
            return
        keep = []
        merge_mm = float(cfg["mesh"]["weld_mm"])
        for s in self.solids:
            try:
                shape = read_brep(s["file"])
                sols = solids_of(shape)
                if len(sols) != 1:
                    raise ValueError("%d solids in one solid file" % len(sols))
                s["shape"] = sols[0]
                V, c, I = mass_props(sols[0])
                if not V > float(G["min_volume_mm3"]):
                    raise ValueError("no volume")
                s["V"], s["com"], s["I"] = V, c, I
                s["verts"] = unique_vertices(sols[0], merge_mm)
                ext = s["verts"].max(axis=0) - s["verts"].min(axis=0) if len(s["verts"]) else np.zeros(3)
                s["size"] = float(np.linalg.norm(ext))
            except Exception as exc:  # noqa: BLE001 - a solid that does not load is left out
                st["ignored_bad_solids"] += 1
                note("a solid of %s cannot be used (%s): left out" % (s["label"] or s["node"], str(exc)[:100]))
                if s["tag"] is not None:
                    st["tags_invalid"] += 1
                continue
            keep.append(s)
        for k, s in enumerate(keep):
            s["sid"] = k
        self.solids = keep
        if not self.solids:
            diags.append("The Assembly holds no solid (objects outside the Assembly are not read).")
            return
        self.ok = True

    def geo(self, s, seed):
        if "geo" not in s:
            s["geo"] = mesh_submitted(s["shape"], self.cfg, self.tmp, seed, s.get("V"))                     # [mesher tiers]
        return s["geo"]


def item_error(stats, diags, what, exc):
    """A grading item that raised: it scores 0; counted (grader_item_errors) and reported."""
    stats["grader_item_errors"] = stats.get("grader_item_errors", 0) + 1
    if diags is not None and sum(1 for d in diags if d.startswith("Item error:")) < 20:
        diags.append("Item error: %s could not be evaluated (%s: %s); it scores 0" % (what, type(exc).__name__, str(exc)[:160]))


# ============================================================================ given-part recognition (as forked)
def recognise(sub, ref, cfg, stats):
    """Match every untagged solid to a constituent class. Sets s["match"] = {"cls", "T" (class representative, local
    -> submission frame), "err", "mode", "alt"}."""
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
            item_error(stats, sub.diags_ref, "recognition of a solid of %s" % (s["label"] or s["node"]), exc)
            got = None
        if got is not None:
            s["match"] = got
            if got["mode"] == "exact":
                n_exact += 1
            else:
                n_approx += 1
    stats["purchased_solids_exact"] = n_exact
    stats["purchased_solids_approx"] = n_approx


def _recognise_one(s, sub, ref, cfg, R, vol_rel, tol):
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
            if "base95" not in rep:
                pr = Placed(rep["geo"], np.eye(4), cfg)
                rep["base95"] = float(np.quantile(one_way(pr, pr, disk), 0.95))
            lim = rep["base95"] + max(float(R["approx_p95_mm"][0]), float(R["approx_p95_mm"][1]) * rep["size"])
            if p95 <= lim:
                got = {"cls": cl["id"], "T": T, "err": p95, "mode": "approx", "alt": [T]}
                break
    return got


def form_instances(sub, ref, cfg, stats):
    """Instances of the given types (one round over every type, [no base assembly]): an instance is seeded on every
    solid that matches the type's anchor constituent; the other constituents attach one to one by predicted centroid
    (within attach_mm). Returns [{"type", "W" (local -> submission frame), "slots": {k: sid}, "completeness", "alt"}]."""
    attach = float(cfg["recognition"]["attach_mm"])
    by_cls = {}
    for s in sub.solids:
        if "match" in s:
            by_cls.setdefault(s["match"]["cls"], []).append(s)
    claimed = set()
    inst = []
    for tname in sorted(ref.types):
        typ = ref.types[tname]
        a = typ["cons"][typ["anchor"]]
        cl = ref.classes[a["cls"]]
        M = dict((id(m[0]), m[1]) for m in cl["members"])[id(a)]
        for s in by_cls.get(a["cls"], []):
            if s["sid"] in claimed:
                continue
            Ws = [T @ inv(M) for T in s["match"].get("alt", [s["match"]["T"]])]
            inst.append({"type": tname, "W": Ws[0], "alt": Ws, "slots": {typ["anchor"]: s["sid"]}})
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
    for it in inst:
        it["completeness"] = len(it["slots"]) / float(len(ref.types[it["type"]]["cons"]))
    for s in sub.solids:
        if "match" in s and s["sid"] not in claimed:
            s["extra_cls"] = s["match"]["cls"]
    stats["purchased_instances_found"] = len(inst)
    stats["purchased_solids_unattached"] = sum(1 for s in sub.solids if "extra_cls" in s)
    return inst


# ============================================================================ [part fit] the per-part fit
def principal_starts(cs, Is, cd, Id):
    """The 24 proper rotations mapping the source's inertia axes onto the destination's (every permutation and sign
    choice with det +1), centroids matched: T source -> destination."""
    _ws, Es = eigen(Is)
    _wd, Ed = eigen(Id)
    out = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            M = np.zeros((3, 3))
            for i, (p, sg) in enumerate(zip(perm, signs)):
                M[:, p] = sg * Ed[:, i]
            R = M @ Es.T
            if np.linalg.det(R) < 0:
                continue
            out.append(make_T(R, cd - R @ cs))
    return out


def even_subset(P, n):
    P = np.asarray(P)
    if len(P) <= n:
        return P
    return P[np.linspace(0, len(P) - 1, int(n)).astype(int)]


class PartFit(object):
    """The fit of one tagged solid onto its kind (rule 3): T maps the submission frame onto the kind's frame."""

    def __init__(self, G, s, kind):
        self.G, self.s, self.k = G, s, kind
        self.cfg = G.cfg

    def measures(self, T, nq=None):
        """(p90 of the two-way surface distance, IoU, n distances) at pose T; nq limits the samples each way."""
        cfg, k, g = self.cfg, self.k, self.s["geo"]
        disk = float(cfg["sampling"]["disk_mm"])
        P, N, tree, L = k["samples"]
        q_sub = g.query(cfg)[0]
        if nq:
            q_sub = even_subset(q_sub, nq)
        q_sub = apply(T, q_sub)
        d1, idx = tree.query(q_sub)
        keep = L[idx]
        d1 = surface_distances(q_sub[keep], P, N, tree, disk) if keep.any() else np.zeros(0)
        qi = k["query_idx"] if not nq else even_subset(k["query_idx"], nq)
        Ds, Ns, ts = g.dense(cfg)
        q_ref = apply(inv(T), P[qi])
        d2 = surface_distances(q_ref, Ds, Ns, ts, disk) if len(q_ref) and ts is not None else np.zeros(0)
        d = np.r_[d1, d2]
        p90 = float(np.quantile(d, 0.9)) if len(d) else np.inf
        iou = self.iou(T)
        return p90, iou, len(d)

    def iou(self, T):
        g, k = self.s["geo"], self.k
        a = placed_manifold(g, T)
        b = k["geo"].man
        if a is None or b is None:
            return 0.0
        inter = common_volume(a, b)
        va = float(a.volume()) if g.man is None else g.volume
        vb = k["geo"].volume
        return inter / max(va + vb - inter, 1e-9)

    def score(self, p90, iou):
        S = self.cfg["shape"]
        return (float(S["w_surface"]) * band(p90, S["surface_p90_mm"][self.k["size_class"]]) +
                float(S["w_iou"]) * rising(iou, S["iou"]))

    def run(self):
        cfg, s, k = self.cfg, self.s, self.k
        F = cfg["part_fit"]
        tol = float(cfg["recognition"]["exact_tol_mm"])
        seed = int(cfg["sampling"]["seed"]) + 20000 + s["sid"]
        # an exact copy of the kind's solid: its vertex set registers exactly and its volume is the kind's
        exact = None
        if len(s["verts"]) == len(k["verts"]) and abs(s["V"] / k["V"] - 1.0) <= float(F["exact_volume_rel"]):
            exact = register_exact(s, k, tol, all_solutions=True)
        if exact:
            T = exact[0][0]
            s["geo"] = geo_moved(k["geo"], inv(T), seed)
            s["geo"].notes = ["an exact copy of its kind: the kind's own tessellation (tier %s)" % k["tier"]]
            s["geo"].tier = "exact"
            p90, iou, n = self.measures(T)
            return {"T": T, "mode": "exact", "err": exact[0][1], "p90": p90, "iou": iou, "n": n,
                    "score": self.score(p90, iou), "n_candidates": len(exact), "start": "exact"}
        g = self.G.sub.geo(s, seed)
        if g is None:
            raise ValueError("the part cannot be tessellated")
        P, N, tree, _L = k["samples"]
        q = g.query(cfg)[0]
        coarse_pts = even_subset(q, int(F["n_coarse"]))
        fine_pts = even_subset(q, int(F["n_fine"]))
        starts = principal_starts(s["com"], s["I"], k["com"], k["I"])
        coarse = []
        for i, T0 in enumerate(starts):
            Tk, rk = icp(coarse_pts, P, N, tree, T0, int(F["coarse_iters"]), float(F["coarse_trim"]), float(F["bound_mm"]))
            coarse.append((rk, i, Tk))
        coarse.sort(key=lambda x: (x[0], x[1]))
        corners = box_probe(*g.bounds)
        kept = []
        for rk, i, Tk in coarse:
            if all(pose_distance(Tk, c[2], corners) > float(F["dedupe_mm"]) for c in kept):
                kept.append((rk, i, Tk))
            if len(kept) >= int(F["refine"]):
                break
        cands = []
        for rk, i, Tk in kept:
            Tf, rf = icp(fine_pts, P, N, tree, Tk, int(F["fine_iters"]), float(F["fine_trim"]), float(F["bound_mm"]))
            p90, iou, _n = self.measures(Tf, int(F["n_rank"]))
            cands.append({"T": Tf, "rms": rf, "start": i, "score": self.score(p90, iou)})
        cands.sort(key=lambda c: (-round(c["score"], 6), round(c["rms"], 9), c["start"]))
        best = cands[0]
        p90, iou, n = self.measures(best["T"])
        return {"T": best["T"], "mode": "fit", "err": best["rms"], "p90": p90, "iou": iou, "n": n,
                "score": self.score(p90, iou), "n_candidates": len(cands), "start": "principal%02d" % best["start"]}


# ============================================================================ [holes] hole steps
def axial_steps(shape, cfg):
    """The submitted hole steps of a solid: its concave cylindrical and conical faces (and free-form faces that fit a
    cylinder), coaxial faces of one radius joined into one step: [{kind, point (on the axis, the step's mid-depth),
    dir, radius, span_deg, depth}]."""
    Fc = cfg["features"]
    faces = concave_axial_faces(shape, tol_rel=float(Fc["fitted_tol_rel"]), tol_abs=float(Fc["fitted_tol_abs"]))
    out = []
    for c in faces:
        ext = _axial_extent(c)
        if ext is None:
            continue
        c["h0"], c["h1"] = ext
        out.append(c)
    M = Fc["coaxial_merge"]
    steps = []
    for c in out:
        d = c["dir"] / np.linalg.norm(c["dir"])
        placed = False
        for st in steps:
            # faces that are no exact cylinders (fitted) merge within the fit's own tolerance: the two halves of a
            # scaled wall fit circles a few hundredths of a millimetre apart
            tol_r, tol_o = float(M["radius_mm"]), float(M["offset_mm"])
            if c.get("fitted") or st.get("fitted"):
                ft = max(float(Fc["fitted_tol_abs"]), float(Fc["fitted_tol_rel"]) * min(st["radius"], c["radius"]))
                tol_r, tol_o = max(tol_r, ft), max(tol_o, ft)
            if st["kind"] != c["kind"] or abs(st["radius"] - c["radius"]) > tol_r:
                continue
            if math.degrees(math.acos(min(1.0, abs(float(d @ st["dir"]))))) > float(M["angle_deg"]):
                continue
            v = c["point"] - st["point0"]
            if float(np.linalg.norm(v - st["dir"] * float(v @ st["dir"]))) > tol_o:
                continue
            sg = 1.0 if float(d @ st["dir"]) >= 0 else -1.0
            h0 = float((c["point"] - st["point0"]) @ st["dir"])
            lo, hi = sorted((h0 + sg * c["h0"], h0 + sg * c["h1"]))
            if hi < st["lo"] - float(M["offset_mm"]) or lo > st["hi"] + float(M["offset_mm"]):
                continue
            st["lo"], st["hi"] = min(st["lo"], lo), max(st["hi"], hi)
            st["span"] += c["span_rad"]
            st["fitted"] = st.get("fitted") or bool(c.get("fitted"))
            placed = True
            break
        if not placed:
            steps.append({"kind": c["kind"], "point0": np.asarray(c["point"], float), "dir": d, "radius": c["radius"],
                          "lo": float(c["h0"]), "hi": float(c["h1"]), "span": float(c["span_rad"]),
                          "fitted": bool(c.get("fitted"))})
    res = []
    for st in steps:
        mid = st["point0"] + st["dir"] * 0.5 * (st["lo"] + st["hi"])
        res.append({"kind": st["kind"], "point": mid, "dir": st["dir"], "radius": float(st["radius"]),
                    "span_deg": float(min(360.0, math.degrees(st["span"]))), "depth": float(st["hi"] - st["lo"])})
    return res


def hole_groups(mine, Fc):
    """[holes] The submitted holes of a part: its steps whose arc spans at least features.min_span_deg, joined when
    their axis lines lie within the zero ends of the offset band (the distance of each one's mid-depth point from the
    other's line) and of the angle band of each other, transitively; a list of index lists in document order. Built
    from the submission alone."""
    idx = [j for j, c in enumerate(mine) if c["span_deg"] >= float(Fc["min_span_deg"])]
    off_max, ang_max = float(Fc["offset_mm"][1]), float(Fc["angle_deg"][1])
    root = {j: j for j in idx}

    def find(j):
        while root[j] != j:
            j = root[j]
        return j

    def dist(a, b):
        v = a["point"] - b["point"]
        db = b["dir"] / np.linalg.norm(b["dir"])
        return float(np.linalg.norm(v - db * float(v @ db)))
    for x in range(len(idx)):
        for y in range(x + 1, len(idx)):
            a, b = mine[idx[x]], mine[idx[y]]
            da, db = a["dir"] / np.linalg.norm(a["dir"]), b["dir"] / np.linalg.norm(b["dir"])
            if math.degrees(math.acos(min(1.0, abs(float(da @ db))))) >= ang_max:
                continue
            if max(dist(a, b), dist(b, a)) >= off_max:
                continue
            ra, rb = find(idx[x]), find(idx[y])
            if ra != rb:
                root[max(ra, rb)] = min(ra, rb)
    groups = {}
    for j in idx:
        groups.setdefault(find(j), []).append(j)
    return [groups[k] for k in sorted(groups)]


def read_by_structure(hole_of, mine, M, E, near, Fc):
    """[holes] The steps read when a submitted hole has more steps than the reference hole it is matched to. hole_of:
    the reference hole of each row of M (one row per reference step); M the scores, E the eligible pairs, near the
    eligible pairs within the offset and angle zero ends. A submitted hole is matched to a reference hole when one of
    its steps is near one of the hole's steps; when more of its steps are eligible for that hole than the hole has
    steps, as many as the hole has steps are kept by structure (the deepest first, depths within
    features.coaxial_merge.offset_mm a tie, ties to the larger radius, then document order) and the others' scores
    against the hole are set to 0 in M (not read). Returns the set of submitted steps not read against some hole."""
    tie = float(Fc["coaxial_merge"]["offset_mm"])
    rows = {}
    for i, h in enumerate(hole_of):
        rows.setdefault(h, []).append(i)
    unread = set()
    for g in hole_groups(mine, Fc):
        for h in sorted(rows, key=lambda h: rows[h][0]):
            rs = rows[h]
            if not near[np.ix_(rs, g)].any():
                continue
            el = [j for j in g if E[rs, j].any()]
            if len(el) <= len(rs):
                continue
            keep = []
            left = list(el)
            while len(keep) < len(rs):
                deep = max(mine[j]["depth"] for j in left)
                cand = [j for j in left if mine[j]["depth"] >= deep - tie]
                j = sorted(cand, key=lambda j: (-mine[j]["radius"], j))[0]
                keep.append(j)
                left.remove(j)
            for j in left:
                M[rs, j] = 0.0
                unread.add(j)
    return unread


def extra_steps(mine, items, cred_rows, credited, unread, unlocated, Fc):
    """[extras] The extra hole steps of one part (rule 6, features). mine: the submitted steps in the part's fit; items:
    the located reference steps (rows of the score matrix), cred_rows / credited: the rows and the submitted steps the
    assignment credited (score above 0); unread: the steps the structural choice did not read; unlocated: the part's
    holes that are not located. A step is a candidate when it is a features.extras.kinds wall spanning at least
    features.extras.min_span_deg, at least features.extras.min_depth_mm deep, of a radius within
    features.extras.radius_mm; the candidates neither credited nor unread are matched one to one to the true steps not
    credited (the located ones left over and every step of the unlocated holes) where their axis lines lie within the
    zero ends of the offset and angle bands (the radius free); a maximum matching, so the count does not depend on the
    order. Extras: the unread steps not credited elsewhere and the candidates left unmatched. Returns (number of extras,
    number of steps matched to a step of a hole that is not located)."""
    X = Fc["extras"]
    lo, hi = float(X["radius_mm"][0]), float(X["radius_mm"][1])
    kinds = set(X["kinds"])
    cand = [j for j, c in enumerate(mine) if c["kind"] in kinds and c["span_deg"] >= float(X["min_span_deg"])
            and c["depth"] >= float(X["min_depth_mm"]) - 1e-9 and lo <= c["radius"] <= hi]
    left = [j for j in cand if j not in credited and j not in unread]
    refs = [(True, it[1], it[2]) for i, it in enumerate(items) if i not in cred_rows]
    refs += [(False, np.asarray(st["point_mm"], float), np.asarray(st["dir"], float)) for h in unlocated for st in h["steps"]]
    matched, n_unloc = set(), 0
    if refs and left:
        N = np.zeros((len(refs), len(left)))
        off_max, ang_max = float(Fc["offset_mm"][1]), float(Fc["angle_deg"][1])
        for a, (_loc, p, d) in enumerate(refs):
            d = d / np.linalg.norm(d)
            for b, j in enumerate(left):
                c = mine[j]
                ang = math.degrees(math.acos(min(1.0, abs(float(d @ c["dir"])))))
                v = p - c["point"]
                off = float(np.linalg.norm(v - c["dir"] * float(v @ c["dir"])))
                N[a, b] = 1.0 if (off < off_max and ang < ang_max) else 0.0
        for a, b in hungarian(-N):
            if N[a, b] > 0:
                matched.add(left[b])
                n_unloc += 0 if refs[a][0] else 1
    extra = (set(unread) - set(credited)) | (set(left) - matched)
    return len(extra), n_unloc


def _axial_extent(c):
    """(lo, hi) of a face's points along its axis, relative to c["point"] (set by concave_axial_faces from the face's
    mid point); from the face's own sampled points."""
    P = c.get("_pts")
    if P is None:
        return None
    h = (P - c["point"]) @ (c["dir"] / np.linalg.norm(c["dir"]))
    return float(h.min()), float(h.max())


def concave_axial_faces(shape, tol_rel=0.03, tol_abs=0.02):
    """As forked, plus the points of each face sampled on a 7 x 7 grid of its parameter range ([holes] a step's depth)."""
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
            u0, u1, v0, v1 = BRepTools.UVBounds_s(f)
            pts = []
            for u in np.linspace(u0, u1, 7):
                for v in np.linspace(v0, v1, 7):
                    pv = ad.Value(float(u), float(v))
                    pts.append((pv.X(), pv.Y(), pv.Z()))
            pts = np.asarray(pts, float)
            if t not in (GeomAbs_Cylinder, GeomAbs_Cone):
                c = _fitted_cylinder(f, ad, tol_rel=tol_rel, tol_abs=tol_abs)
                if c is not None and c["concave"]:
                    c["_pts"] = pts
                    out.append(c)
                continue
            g = ad.Cylinder() if t == GeomAbs_Cylinder else ad.Cone()
            ax = g.Axis()
            loc = np.array([ax.Location().X(), ax.Location().Y(), ax.Location().Z()])
            d = np.array([ax.Direction().X(), ax.Direction().Y(), ax.Direction().Z()])
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
            if not bool(n @ radial < 0):
                continue
            out.append({"kind": "cylinder" if t == GeomAbs_Cylinder else "cone", "point": p - radial, "dir": d,
                        "radius": r, "concave": True, "span_rad": float(u1 - u0), "_pts": pts})
        except Exception:  # noqa: BLE001 - an odd face is not a feature
            continue
    return out


# ============================================================================ [groups] rigid groups and palm groups
class Groups(object):
    """Union-find over the extraction's leaf nodes: unsuppressed Fixed joints and rigid containers join (rule 4)."""

    def __init__(self, model):
        self.parent = {}
        self.leaf_nodes = sorted(set(lf["node"] for c in model.get("components", []) for lf in c.get("leaves", [])))
        for n in self.leaf_nodes:
            self.parent[n] = n
        self.containers = {c["node"]: c for c in model.get("containers", [])}
        self.notes = []
        for node, c in sorted(self.containers.items()):
            if c.get("unit"):
                under = self.leaves_under(node)
                for a in under[1:]:
                    self.union(under[0], a)
        self.fixed = 0
        for j in model.get("joints", []):
            if j.get("type") != "Fixed" or not self.usable(j):
                continue
            a, b = self.leaf_of(j.get("part1")), self.leaf_of(j.get("part2"))
            if a is not None and b is not None and a != b:
                self.union(a, b)
                self.fixed += 1

    @staticmethod
    def usable(j):
        return (not j.get("suppressed") and not j.get("error") and not j.get("foreign") and
                j.get("placement1_finite", True) and j.get("placement2_finite", True))

    def leaves_under(self, node):
        return [n for n in self.leaf_nodes if n == node or n.startswith(node + ".")]

    def leaf_of(self, node):
        """The leaf a joint reference attaches to: the node itself, or the one leaf of a container (a rigid container's
        leaves are one group, so its first leaf stands for it)."""
        if node is None:
            return None
        if node in self.parent:
            return node
        under = self.leaves_under(node)
        if not under:
            return None
        c = self.containers.get(node)
        if len(under) == 1 or (c is not None and c.get("unit")):
            return under[0]
        self.notes.append("a joint attaches to %s, a flexible container of %d leaves: not read" % (node, len(under)))
        return None

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            if rb < ra:
                ra, rb = rb, ra
            self.parent[rb] = ra

    def group(self, node):
        return self.find(node) if node in self.parent else None


# ============================================================================ the grade
def grade(ex_dir, ref_dir, purchased_root, cfg, log=None, ref=None, rigid_only=False):
    """The grade of one extraction (module docstring). `ref`: an already loaded Reference."""
    t0 = time.time()
    log = log or (lambda m: print("[grade] %s" % m, file=sys.stderr, flush=True))
    tmp = tempfile.mkdtemp(prefix="hand_grade_")
    res = {"ok": True, "design_score": 0.0, "criteria": {k: 0.0 for k in CRITERIA},
           "criteria_raw": {k: 0.0 for k in CRITERIA}, "checks": {"cad_valid": 0.0, "usable_palm": 0.0},
           "statistics": {}, "parts": {}, "purchased": {}, "diagnostics": []}
    stats, diags = res["statistics"], res["diagnostics"]
    stats["grader_item_errors"] = 0
    if ref is None:
        ref = Reference(ref_dir, purchased_root, cfg, tmp, log)
    stats["t_reference_s"] = time.time() - t0
    stats["reference_samples_brep"] = 1.0 if ref.samples_source == "brep" else 0.0
    log("reference loaded (%.1f s): %d kinds, %d tags, %d given types" % (
        stats["t_reference_s"], len(ref.kinds), len(ref.tags), len(ref.types)))
    try:
        template = metric_template(ref)
    except Exception:  # noqa: BLE001 - without a mechanism block only a rigid-only development grade goes on
        if not rigid_only:
            raise
        template = dict(GRADE_STATS)
    sub = Submission(ex_dir, ref, cfg, tmp, diags)
    sub.diags_ref = diags
    stats.update(sub.stats)
    if sub.ok:
        res["checks"]["cad_valid"] = 1.0
        stats["n_solids"] = len(sub.solids)
        stats["n_tagged"] = sum(1 for s in sub.solids if s["tag"])
        G = Grading(sub, ref, cfg, stats, diags, res, rigid_only)
        G.run()
    for k, v in template.items():
        stats.setdefault(k, v)
    stats["t_total_s"] = time.time() - t0
    return res


class Grading(object):
    def __init__(self, sub, ref, cfg, stats, diags, res, rigid_only=False):
        self.sub, self.ref, self.cfg = sub, ref, cfg
        self.stats, self.diags, self.res = stats, diags, res
        self.rigid_only = rigid_only
        self.seed = int(cfg["sampling"]["seed"])
        self.sol = {s["sid"]: s for s in sub.solids}

    # ------------------------------------------------------------------ the pipeline
    def run(self):
        st, res, cfg = self.stats, self.res, self.cfg
        t = time.time()
        recognise(self.sub, self.ref, cfg, st)
        self.inst = form_instances(self.sub, self.ref, cfg, st)
        st["t_recognition_s"] = time.time() - t
        t = time.time()
        self.fit_parts()
        st["t_part_fit_s"] = time.time() - t
        self.groups = Groups(self.sub.model)
        st["fixed_joints_read"] = self.groups.fixed
        bad = nonfinite_joints(self.sub.model)
        st["joints_nonfinite"] = float(len(bad))
        if bad:
            local_note(self.diags, "%d joint(s) with a placement that is not a finite number, not read (they join "
                                   "nothing, connect nothing and are no hinge): %s" % (len(bad), ", ".join(bad[:8])))
        for n in self.groups.notes[:5]:
            local_note(self.diags, n)
        self.palm_groups()
        if not self.palm:
            self.diags.append("No usable palm: no rigid group of the Assembly holds more palm parts than digit parts "
                              "(objects outside the Assembly are not read): nothing can be aligned, the document "
                              "scores 0.")
            res["checks"]["cad_valid"] = 0.0
            return
        res["checks"]["usable_palm"] = 1.0
        t = time.time()
        self.A = self.align()
        st["t_alignment_s"] = time.time() - t
        if self.A is None:
            self.diags.append("No usable palm: the palm groups hold no member that can be aligned; the document "
                              "scores 0.")
            res["checks"]["cad_valid"] = 0.0
            res["checks"]["usable_palm"] = 0.0
            return
        self.assign_parts()
        self.assign_instances()
        t = time.time()
        raw = res["criteria_raw"]
        raw["shape"] = self.shape()
        raw["features"] = self.features()
        raw["assembly"] = self.assembly()
        st["t_rigid_criteria_s"] = time.time() - t
        lo, hi = float(cfg["floors"]["lo"]), float(cfg["floors"]["hi"])
        self.floor = rising(raw["shape"], (lo, hi))
        st["shape_floor"] = self.floor
        mech = {k: 0.0 for k in MECHANISM_CRITERIA}
        st["mechanism_module"] = 0.0
        if not self.rigid_only:
            import handtree                                                                  # [combination] the mechanism module
            t = time.time()
            mech = handtree.mechanism(self)
            st["mechanism_module"] = 1.0
            st["t_mechanism_s"] = time.time() - t
        for k in MECHANISM_CRITERIA:
            raw[k] = finite(mech.get(k, 0.0))
        crit = res["criteria"]
        crit["shape"] = raw["shape"]
        for k in ("kinematic", "features", "assembly", "mass", "fit", "interference"):
            crit[k] = finite(self.floor * raw[k])
        for k in CRITERIA:
            raw[k] = finite(raw[k])
            crit[k] = finite(crit[k])
        res["design_score"] = float(sum(float(cfg["weights"][k]) * crit[k] for k in CRITERIA))

    # ------------------------------------------------------------------ [part fit] the per-part fits
    def fit_parts(self):
        self.fits = {}
        n_exact = 0
        for s in self.sub.solids:
            if s["tag"] is None:
                continue
            k = self.ref.kinds[self.ref.parts[s["tag"]]["kind"]]
            try:
                f = PartFit(self, s, k).run()
                self.fits[s["sid"]] = f
                n_exact += 1 if f["mode"] == "exact" else 0
            except Exception as exc:  # noqa: BLE001 - this part has no fit: it costs only its own items
                item_error(self.stats, self.diags, "the fit of %s" % s["tag"], exc)
        self.stats["parts_fitted"] = len(self.fits)
        self.stats["parts_exact_copies"] = n_exact

    def part_pose(self, s):
        """The fitted part's pose: kind frame -> submission frame (inv of the fit), or None."""
        f = self.fits.get(s["sid"])
        return inv(f["T"]) if f is not None else None

    # ------------------------------------------------------------------ [groups] palm groups
    def palm_groups(self):
        """Every rigid group whose palm parts' reference volume exceeds its digit structural parts' (rule 4)."""
        ref = self.ref
        vol = {}
        for s in self.sub.solids:
            g = self.groups.group(s["node"])
            if g is None:
                continue
            pv = sv = 0.0
            if s["tag"] is not None:
                kid = ref.parts[s["tag"]]["kind"]
                if kid in ref.palm_kinds:
                    pv = ref.kinds[kid]["V"]
                elif kid in ref.structural_kinds:
                    sv = ref.kinds[kid]["V"]
            vol.setdefault(g, [0.0, 0.0])
            vol[g][0] += pv
            vol[g][1] += sv
        for it in self.inst:
            if it["type"] in ref.palm_types:
                s = self.sol[it["slots"][ref.types[it["type"]]["anchor"]]]
                g = self.groups.group(s["node"])
                if g is not None:
                    vol.setdefault(g, [0.0, 0.0])
                    vol[g][0] += ref.types[it["type"]]["V"]
        self.palm = sorted(g for g, (p, sv) in vol.items() if p > sv)
        self.stats["palm_groups"] = len(self.palm)
        self.stats["palm_split"] = 1.0 if len(self.palm) > 1 else 0.0
        if len(self.palm) > 1:
            local_note(self.diags, "the palm is in %d rigid groups (palm_split): only their connection is lost" % len(self.palm))

    # ------------------------------------------------------------------ [alignment] the alignment
    def _members(self):
        """The alignment's members: (kind of member, key, submission pose local -> submission frame, list of candidate
        reference slots with their poses, probe points (local), undirected?, weight)."""
        ref, cfg = self.ref, self.cfg
        A = cfg["alignment"]
        h = float(A["probe_mm"])
        palm = set(self.palm)
        out = []
        types = set(A["member_types"])
        for i, it in enumerate(self.inst):
            if it["type"] not in types:
                continue
            typ = ref.types[it["type"]]
            s = self.sol[it["slots"][typ["anchor"]]]
            if self.groups.group(s["node"]) not in palm:
                continue
            slots = [(p, ref.placements[p]["W"]) for p in sorted(ref.palm_placements)
                     if ref.placements[p]["type"] == it["type"] and ref.placements[p]["located"]]           # [located]
            if not slots:
                continue
            w = float(cfg["assembly"]["type_weights"].get(it["type"], 1.0))
            if typ["output_axis"]:
                o = np.asarray(typ["output_axis"]["point"], float)
                a = np.asarray(typ["output_axis"]["dir"], float)
                a = a / np.linalg.norm(a)
                probe = np.array([o, o + h * a, o - h * a])
                out.append({"what": "axis", "key": "given:%d" % i, "type": it["type"], "W": it["W"], "slots": slots,
                            "probe": probe, "undirected": True, "w": w})
            else:
                c = typ["origin"]
                probe = np.array([c, c + [h, 0, 0], c + [0, h, 0], c + [0, 0, h]])
                out.append({"what": "pose", "key": "given:%d" % i, "type": it["type"], "W": it["W"], "slots": slots,
                            "probe": probe, "undirected": False, "w": w})
        wm = float(A["made_weight_per_sqrt_mm2"])
        for s in self.sub.solids:
            if s["tag"] is None or s["sid"] not in self.fits:
                continue
            kid = ref.parts[s["tag"]]["kind"]
            if kid not in ref.palm_kinds or self.groups.group(s["node"]) not in palm:
                continue
            k = ref.kinds[kid]
            P = self.part_pose(s)
            slots = [(t, ref.parts[t]["W"]) for t in ref.kind_tags[kid] if ref.parts[t]["located"]]     # [located]
            if not slots:
                continue
            c = k["com"]
            _w, E = eigen(k["I"])
            probe = np.array([c, c + h * E[:, 0], c + h * E[:, 1], c + h * E[:, 2]])
            out.append({"what": "pose", "key": "part:%s" % s["tag"], "type": "kind:" + kid, "W": P, "slots": slots,
                        "probe": probe, "undirected": False, "w": wm * k["weight"],
                        "symmetries": k["symmetries"]})
        return out

    @staticmethod
    def _residual(A, m, Wr):
        """RMS residual of member m against reference pose Wr under A; (r, target points). A member with an output
        axis (a servo) counts by its undirected axis line: the RMS distance of its probe points from the reference
        output axis line, the targets their feet on that line (where along the line and how it is turned about it are
        free: a servo turned end for end keeps its line, and its hinge's sense prices the turn)."""
        src = apply(A @ m["W"], m["probe"])
        if m["what"] == "axis":
            o = apply(Wr, m["probe"][:1])[0]
            a = Wr[:3, :3] @ (m["probe"][1] - m["probe"][0])
            a = a / np.linalg.norm(a)
            feet = o + np.outer((src - o) @ a, a)
            return float(np.sqrt(np.mean(np.sum((src - feet) ** 2, axis=1)))), feet
        tgt = apply(Wr, m["probe"])
        best = None
        variants = [tgt]
        if m["undirected"]:
            variants.append(tgt[[0, 2, 1]])
        for S in m.get("symmetries", [])[1:]:
            variants.append(apply(Wr @ S, m["probe"]))
        for v in variants:
            r = float(np.sqrt(np.mean(np.sum((src - v) ** 2, axis=1))))
            if best is None or r < best[0]:
                best = (r, v)
        return best[0], best[1]

    def _cost(self, A, members):
        """(Huber cost, assignment {member index: (slot id, r, target points)}) under A, one to one within types."""
        d = float(self.cfg["alignment"]["huber_mm"])
        by = {}
        for i, m in enumerate(members):
            by.setdefault(m["type"], []).append(i)
        total, assign = 0.0, {}
        for _t, idx in sorted(by.items()):
            slots = members[idx[0]]["slots"]
            C = np.zeros((len(idx), len(slots)))
            keep = {}
            for a, i in enumerate(idx):
                for b, (sid, Wr) in enumerate(slots):
                    r, tg = self._residual(A, members[i], Wr)
                    C[a, b] = r
                    keep[(a, b)] = tg
            for a, b in hungarian(C):
                r = C[a, b]
                hub = 0.5 * r * r if r <= d else d * (r - 0.5 * d)
                total += members[idx[a]]["w"] * hub
                assign[idx[a]] = (slots[b][0], r, keep[(a, b)])
        return total, assign

    def _hypotheses(self, members):
        hyps = []
        for m in members:
            if m["what"] != "pose":
                continue
            for sid, Wr in m["slots"]:
                for S in m.get("symmetries", [np.eye(4)])[:1]:
                    hyps.append(("%s->%s" % (m["key"], sid), Wr @ S @ inv(m["W"])))
        ax = [m for m in members if m["what"] == "axis"]
        h = float(self.cfg["alignment"]["probe_mm"])
        for a, b in itertools.combinations(range(len(ax)), 2):
            ma, mb = ax[a], ax[b]
            P = np.vstack([apply(ma["W"], ma["probe"][:2]), apply(mb["W"], mb["probe"][:2])])
            slots = ma["slots"]
            for (ia, Wa), (ib, Wb) in itertools.permutations(slots, 2):
                for sa, sb in itertools.product((1, 2), repeat=2):
                    Q = np.vstack([apply(Wa, ma["probe"][[0, sa]]), apply(Wb, mb["probe"][[0, sb]])])
                    if np.linalg.norm(P[0] - P[2]) < 1e-6:
                        continue
                    hyps.append(("%s,%s->%s,%s" % (ma["key"], mb["key"], ia, ib), kabsch(P, Q)))
        return hyps

    def align(self):
        """The palm alignment (rule 5). Returns A (submission -> reference) or None."""
        cfg, st = self.cfg, self.stats
        Ac = cfg["alignment"]
        members = self._members()
        st["alignment_members"] = len(members)
        if not members:
            return None
        hyps = self._hypotheses(members)
        st["alignment_hypotheses"] = len(hyps)
        scored = []
        for k, (lab, H) in enumerate(hyps):
            H = make_T(orthonormal(H[:3, :3]), H[:3, 3])
            c, _a = self._cost(H, members)
            scored.append((c, k, lab, H))
        scored.sort(key=lambda x: (x[0], x[1]))
        c0, _k, lab, A = scored[0]
        inl = float(Ac["inlier_mm"])
        d = float(Ac["huber_mm"])
        for _it in range(int(Ac["irls_iters"])):
            _c, assign = self._cost(A, members)
            P, Q, W = [], [], []
            for i, (sid, r, tg) in assign.items():
                if r > inl:
                    continue
                m = members[i]
                w = m["w"] * (1.0 if r <= d else d / r)
                P.append(apply(m["W"], m["probe"]))
                Q.append(tg)
                W.append(np.full(len(m["probe"]), w))
            if not P or sum(len(p) for p in P) < 3:
                break
            A2 = kabsch_w(np.vstack(P), np.vstack(Q), np.concatenate(W))
            probe = box_probe(np.full(3, -100.0), np.full(3, 100.0))
            moved = pose_distance(A2, A, probe)
            A = A2
            if moved < 1e-10:
                break
        cost, assign = self._cost(A, members)
        rs = [r for (_s, r, _t) in assign.values() if r <= inl]
        if rs:
            st["alignment_residual_mm"] = finite(np.sqrt(np.mean(np.square(rs))))
        else:
            # no member within inlier_mm: the best hypothesis stands; its residual is taken over every member
            allr = [r for (_s, r, _t) in assign.values()]
            st["alignment_residual_mm"] = finite(np.sqrt(np.mean(np.square(allr))) if allr else 0.0)
            local_note(self.diags, "the palm alignment has no member within %g mm of its reference placement: "
                                   "everything positional is read in the best hypothesis's frame (residual over all "
                                   "members %.1f mm)" % (inl, st["alignment_residual_mm"]))
        st["alignment_inliers"] = len(rs)
        st["alignment_seed"] = lab
        probe = box_probe(np.full(3, -100.0), np.full(3, 100.0))
        runner = None
        for c, _k, _lab, H in scored[1:]:
            if pose_distance(H, scored[0][3], probe) > float(Ac["distinct_mm"]):
                runner = c
                break
        st["alignment_runner_up_ratio"] = finite(c0 / runner, 0.0) if runner else 0.0
        self.alignment_members = [{"key": members[i]["key"], "slot": sid, "residual_mm": r}
                                  for i, (sid, r, _t) in sorted(assign.items())]
        return A

    # ------------------------------------------------------------------ assignments (in the aligned frame)
    def assign_parts(self):
        """Tagged parts to reference tags, one to one by installed position (fitted centroid) inside each kind."""
        self.part_of = {}
        for kid, rtags in sorted(self.ref.kind_tags.items()):
            k = self.ref.kinds[kid]
            agents = [s for s in self.sub.solids if s["tag"] in rtags and s["sid"] in self.fits]
            if not agents:
                continue
            cost = np.array([[float(np.linalg.norm(apply(self.A @ self.part_pose(s), k["com"]) -
                                                   apply(self.ref.parts[t]["W"], k["com"]))) for s in agents]
                             for t in rtags])
            for r, c in hungarian(cost):
                self.part_of[rtags[r]] = agents[c]
        self.stats["tags_reassigned"] = sum(1 for t, s in self.part_of.items() if s["tag"] != t)

    def assign_instances(self):
        """Given instances to reference placements, one to one per type by the position of the type's centroid."""
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
        taken = set(id(it) for it in self.inst_of.values())
        left = [it for it in self.inst if id(it) not in taken]
        self.left_over = left
        self.stats["purchased_instances_left_over"] = len(left)
        if left:
            local_note(self.diags, "%d given part(s) in the document beyond the hand's count (%s): their solids count "
                                   "as extra solids" % (len(left), ", ".join(sorted(set(it["type"] for it in left))[:8])))

    # ------------------------------------------------------------------ [shape] shape
    def shape(self):
        ref = self.ref
        num = den = 0.0
        hull = []
        self.stats["parts_on_convex_hull"] = hull
        for tag in ref.tags:
            k = ref.kinds[ref.parts[tag]["kind"]]
            w = k["weight"]
            den += w
            s = self.part_of.get(tag)
            rec = {"kind": k["id"], "present": s is not None, "weight": w}
            self.res["parts"][tag] = rec
            if s is None:
                rec.update(shape=0.0)
                continue
            f = self.fits[s["sid"]]
            sc = f["score"]
            rec.update(shape=sc, surface_p90_mm=finite(f["p90"], 1e9), iou=float(f["iou"]), fit=f["mode"],
                       fit_start=f["start"], tag=s["tag"], mesh_tier=getattr(s.get("geo"), "tier", None))
            if s.get("geo") is not None and s["geo"].man is None:
                hull.append(s["tag"])
                rec["convex_hull"] = True
                local_note(self.diags, "%s: its surface cannot be meshed as a closed volume of its own size, so its "
                                       "convex hull stands for it in the volume measures" % s["tag"])
            num += w * sc
        hull.sort()
        self.stats["parts_on_hull"] = float(len(hull))
        if hull:
            local_note(self.diags, "%d part(s) on the convex-hull fallback: %s" % (len(hull), ", ".join(hull[:20])))
        return num / den if den > 0 else 0.0

    # ------------------------------------------------------------------ [holes] features
    def features(self):
        ref, cfg = self.ref, self.cfg
        Fc = cfg["features"]
        num = den = 0.0
        n_steps = n_found = n_unread = n_extra = n_unloc = 0
        unread_parts, extra_parts = [], []
        for tag in ref.tags:
            k = ref.kinds[ref.parts[tag]["kind"]]
            holes = [h for h in k["features"] if h.get("located", True)]
            if not holes:
                continue
            w = k["weight"]
            den += w
            s = self.part_of.get(tag)
            if s is None:
                self.res["parts"][tag]["features"] = 0.0
                n_steps += sum(len(h["steps"]) for h in holes)
                continue
            try:
                sc, ns, nf, nu, nx, na = self._features_of(s, k, holes, Fc)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the features of %s" % tag, exc)
                sc, ns, nf, nu, nx, na = 0.0, sum(len(h["steps"]) for h in holes), 0, 0, 0, 0
            self.res["parts"][tag]["features"] = sc
            num += w * sc
            n_steps += ns
            n_found += nf
            n_unread += nu
            n_extra += nx
            n_unloc += na
            if nu:
                self.res["parts"][tag]["hole_steps_unread"] = nu
                unread_parts.append("%s (%d)" % (s["tag"], nu))
            if nx:
                self.res["parts"][tag]["hole_steps_extra"] = nx
                extra_parts.append("%s (%d)" % (s["tag"], nx))
        self.stats["hole_steps_total"] = n_steps
        self.stats["hole_steps_found"] = n_found
        self.stats["hole_steps_unread"] = float(n_unread)
        self.stats["hole_steps_extra"] = float(n_extra)
        self.stats["hole_steps_at_unlocated"] = float(n_unloc)
        if extra_parts:
            local_note(self.diags, "extra hole steps (matched to no step of the true part; each counts in its part's "
                                   "hole score): %s" % ", ".join(extra_parts[:20]))
        if unread_parts:
            local_note(self.diags, "hole steps not read (a hole with more eligible steps than the hole it matches: "
                                   "the deepest read, ties to the larger radius): %s" % ", ".join(unread_parts[:20]))
        return num / den if den > 0 else 0.0

    def _features_of(self, s, k, holes, Fc):
        if "steps" not in s:
            s["steps"] = axial_steps(s["shape"], self.cfg)
        T = self.fits[s["sid"]]["T"]
        R = T[:3, :3]
        mine = [{"kind": c["kind"], "point": apply(T, c["point"]), "dir": R @ c["dir"], "radius": c["radius"],
                 "span_deg": c["span_deg"], "depth": c["depth"]} for c in s["steps"]]
        items = []
        for h in holes:
            for stp in h["steps"]:
                items.append((h["id"], np.asarray(stp["point_mm"], float), np.asarray(stp["dir"], float),
                              float(stp["radius_mm"]), float(stp["depth_mm"]), stp.get("kind", "cylinder")))
        sc = np.zeros(len(items))
        unread = set()
        credited, cred_rows = set(), set()
        if mine and items:
            M = np.zeros((len(items), len(mine)))
            E = np.zeros((len(items), len(mine)), bool)        # eligible (kind, arc, depth)
            near = np.zeros((len(items), len(mine)), bool)     # eligible and within the offset and angle zero ends
            for i, (_hid, p, d, r, dep, kind) in enumerate(items):
                d = d / np.linalg.norm(d)
                need = max(float(Fc["min_depth_mm"]), float(Fc["depth_ratio"]) * dep)
                for j, c in enumerate(mine):
                    if c["kind"] != kind or c["span_deg"] < float(Fc["min_span_deg"]) or c["depth"] < need - 1e-9:
                        continue
                    E[i, j] = True
                    ang = math.degrees(math.acos(min(1.0, abs(float(d @ c["dir"])))))
                    v = p - c["point"]
                    off = float(np.linalg.norm(v - c["dir"] * float(v @ c["dir"])))
                    near[i, j] = off < float(Fc["offset_mm"][1]) and ang < float(Fc["angle_deg"][1])
                    M[i, j] = band(off, Fc["offset_mm"]) * band(c["radius"] - r, Fc["radius_mm"]) * band(ang, Fc["angle_deg"])
            unread = read_by_structure([it[0] for it in items], mine, M, E, near, Fc)
            for i, j in hungarian(-M):
                sc[i] = M[i, j]
                if M[i, j] > 0:
                    credited.add(j)
                    cred_rows.add(i)
        n_extra, n_unloc = extra_steps(mine, items, cred_rows, credited, unread,
                                       [h for h in ((k or {}).get("features") or []) if not h.get("located", True)], Fc)
        by_hole = {}
        for (hid, _p, _d, _r, _dp, _k), v in zip(items, sc):
            by_hole.setdefault(hid, []).append(v)
        cw = Fc["class_weights"]
        num = den = 0.0
        e = 0.0
        for h in holes:
            w = float(cw.get(h.get("class", "screw"), 1.0))
            num += w * float(np.mean(by_hole[h["id"]]))
            den += w
            e = max(e, w / len(h["steps"]))
        if n_extra:
            den += n_extra * e                                                          # [extras] an extra is not free
        return ((num / den if den > 0 else 0.0), len(items), int(np.sum(sc > 0.5)), len(unread), n_extra, n_unloc)

    # ------------------------------------------------------------------ [assembly] assembly
    def assembly(self):
        Ac = self.cfg["assembly"]
        pl = self.placement()
        ins = self.installation()
        con = self.connection()
        self.stats.update(assembly_placement=pl, assembly_installation=ins, assembly_connection=con)
        return float(Ac["w_placement"]) * pl + float(Ac["w_installation"]) * ins + float(Ac["w_connection"]) * con

    @staticmethod
    def _rot_error(Ra, Rb, syms):
        return min(rot_angle_deg(Rb[:3, :3].T @ Ra[:3, :3] @ S[:3, :3]) for S in syms)

    def placement(self):
        Ac = self.cfg["assembly"]
        ref = self.ref
        num = den = 0.0
        wo = float(Ac["placement_w_orientation"])
        n_out = 0
        for tag in ref.tags:
            k = ref.kinds[ref.parts[tag]["kind"]]
            if not ref.parts[tag]["located"]:                                   # [located] not graded
                n_out += 1
                self.res["parts"].setdefault(tag, {})["place_located"] = False
                continue
            den += k["weight"]
            s = self.part_of.get(tag)
            if s is None:
                continue
            P = self.A @ self.part_pose(s)
            Wr = ref.parts[tag]["W"]
            e_p = float(np.linalg.norm(apply(P, k["com"]) - apply(Wr, k["com"])))
            e_r = self._rot_error(P, Wr, k["symmetries"])
            sc = band(e_p, Ac["placement_position_mm"]) * (1.0 - wo + wo * band(e_r, Ac["placement_orientation_deg"]))
            self.res["parts"][tag].update(placement=sc, position_error_mm=e_p, orientation_error_deg=e_r)
            num += k["weight"] * sc
        self.stats["placement_items_not_graded"] = float(n_out)
        return num / den if den > 0 else 0.0

    def installation(self):
        Ac = self.cfg["assembly"]
        ref = self.ref
        num = den = 0.0
        wo = float(Ac["installation_w_orientation"])
        n_out = 0
        for pid in sorted(ref.placements):
            pl = ref.placements[pid]
            typ = ref.types[pl["type"]]
            w = float(Ac["type_weights"].get(pl["type"], 1.0))
            it = self.inst_of.get(pid)
            rec = {"type": pl["type"], "present": it is not None, "weight": w}
            self.res["purchased"][pid] = rec
            if not pl["located"]:                                               # [located] not graded
                n_out += 1
                rec["place_located"] = False
                continue
            den += w
            if it is None:
                rec["installation"] = 0.0
                continue
            try:
                Wr = pl["W"]
                best = None
                for Wa0 in it.get("alt", [it["W"]]):
                    Wa = self.A @ Wa0
                    e_p = float(np.linalg.norm(apply(Wa, typ["origin"]) - apply(Wr, typ["origin"])))
                    if typ["axis_only"]:
                        a = typ["sym_axis"] / np.linalg.norm(typ["sym_axis"])
                        c = float((Wa[:3, :3] @ a) @ (Wr[:3, :3] @ a))
                        if typ["sense_free"]:
                            c = abs(c)
                        e_r = math.degrees(math.acos(min(1.0, max(-1.0, c))))
                    else:
                        e_r = self._rot_error(Wa, Wr, typ["own_symmetries"])
                    sc = it["completeness"] * band(e_p, Ac["installation_position_mm"]) * (
                        1.0 - wo + wo * band(e_r, Ac["installation_orientation_deg"]))
                    if best is None or sc > best[0]:
                        best = (sc, e_p, e_r)
                sc, e_p, e_r = best
                rec.update(position_error_mm=e_p, orientation_error_deg=e_r)
            except Exception as exc:  # noqa: BLE001
                item_error(self.stats, self.diags, "the installation of %s" % pid, exc)
                sc = 0.0
            rec["installation"] = sc
            num += w * sc
        self.stats["installation_items_not_graded"] = float(n_out)
        return num / den if den > 0 else 0.0

    def connection(self):
        """The weighted share of the largest set of rigid groups that declared joints connect and that holds a palm
        group (made parts by shape weight, given instances by type weight, each half; grounding connects nothing)."""
        ref, Ac = self.ref, self.cfg["assembly"]
        types = set(Ac["connecting_joint_types"])
        adj = {}
        for j in self.sub.model.get("joints", []):
            if j.get("type") not in types or not Groups.usable(j):
                continue
            a = self.groups.group(self.groups.leaf_of(j.get("part1")) or "")
            b = self.groups.group(self.groups.leaf_of(j.get("part2")) or "")
            if a is None or b is None or a == b:
                continue
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
        comp_of = {}
        for g0 in sorted(set(self.groups.find(n) for n in self.groups.leaf_nodes)):
            if g0 in comp_of:
                continue
            todo, seen = [g0], {g0}
            while todo:
                for nx in adj.get(todo.pop(), ()):
                    if nx not in seen:
                        seen.add(nx)
                        todo.append(nx)
            for g in seen:
                comp_of[g] = g0
        wm_tot = sum(ref.kinds[ref.parts[t]["kind"]]["weight"] for t in ref.tags)
        wg_tot = sum(float(Ac["type_weights"].get(pl["type"], 1.0)) for pl in ref.placements.values())
        share = {}
        for t, s in self.part_of.items():
            c = comp_of.get(self.groups.group(s["node"]))
            if c is not None:
                share.setdefault(c, [0.0, 0.0])[0] += ref.kinds[ref.parts[t]["kind"]]["weight"]
        for pid, it in self.inst_of.items():
            s = self.sol[it["slots"][ref.types[it["type"]]["anchor"]]]
            c = comp_of.get(self.groups.group(s["node"]))
            if c is not None:
                share.setdefault(c, [0.0, 0.0])[1] += float(Ac["type_weights"].get(it["type"], 1.0))
        palm_comps = set(comp_of[g] for g in self.palm if g in comp_of)
        best = 0.0
        for c in palm_comps:
            m, g = share.get(c, [0.0, 0.0])
            v = 0.5 * (m / wm_tot if wm_tot > 0 else 0.0) + 0.5 * (g / wg_tot if wg_tot > 0 else 0.0)
            best = max(best, v)
        self.comp_of = comp_of
        self.stats["connected_sets_with_palm"] = len(palm_comps)
        return best


# ============================================================================ entry point (the verify stage's child)
def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("extraction")
    ap.add_argument("reference")
    ap.add_argument("purchased_root")
    ap.add_argument("output")
    ap.add_argument("--config", default=os.path.join(HERE, "grader_config.json"))
    ap.add_argument("--rigid-only", action="store_true")
    a = ap.parse_args(argv[1:])
    with open(a.config) as fh:
        cfg = json.load(fh)
    limit_cpus(int(os.environ.get("ARB_GRADER_CPUS", cfg.get("grader_cpus", 8))))
    try:
        res = grade(a.extraction, a.reference, a.purchased_root, cfg, rigid_only=a.rigid_only)
    except TaskError as exc:
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
