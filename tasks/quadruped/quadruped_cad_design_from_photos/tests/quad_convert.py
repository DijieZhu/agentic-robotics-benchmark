"""quad_convert: infer the agent's mechanism from its FreeCAD document (fcstd_extract.py's record) and write it in the
ground truth's own form (quad-cad-gt/1, gt_quad/CONTRACTS.md section 8), so that one loader reads both. GRADER_SPEC
section 3; a task-local fork of the arm task's fcstd_convert.py (serial chain) for a tree.

    res = convert(extract_dir, reference_dir, given_dir, out_dir, config=None)
    res = {"ok", "errors", "warnings", "stats", "details", "conversion", "legs_valid"}

Runtime: heavy documents (shape files totalling >= conversion.pool_min_total_mb) are tessellated and written by
`spawn` worker processes (conversion.workers, CPU affinity bound; conversion.part_timeout_s for all parts together): the
calling script must guard its entry point with `if __name__ == "__main__"` (verify.py does). use_brep_siblings() is
installed globally (cad_drawings_check.load_step_solids reads the BREP beside a STEP first), so the grader reads the
tier's parts exactly.

ok = False is `cad_valid` = 0 with the reasons in `errors` (no extraction record, the document does not open, not
exactly one top-level Assembly, no Grounded joint, the grounded group does not stand for the body, a closed loop, a joint
of another type); anything else only invalidates the legs or items it touches and is reported in the conversion block
(GRADER_SPEC 4: missing parts, cores or feet only invalidate the legs or items they belong to). Nothing raises for a
missing or broken document (a broken REFERENCE folder raises quad_task.SpecError: a task error).

The steps (GRADER_SPEC 3.1-3.8):
  parts      leaves labelled with a tag of the reference, present once, one valid watertight solid each, at a finite
             placement (else missing); cores and foot pads by shape class (the extractor), unusable when their pose is
             not the given geometry's or their placement is not finite;
  groups     union-find over Fixed joints, rigid units (App::Part; a rigid sub-assembly without inner joints) and all
             Grounded components; a revolute joint whose own Placement1 / Placement2 is not finite is broken and not
             read (its leg is invalid); one whose frame on one reference is not finite (that component's placement is
             not) is read from its other reference;
  tree       revolute joints are edges between groups; a breadth-first walk from the grounded group (sorted, so
             deterministic); an edge back to a reached group is a loop; groups not reached are floating;
  links      the body = the grounded group, when it carries more of the body's tags than of any other single link's
             (else cad_valid = 0); a body tag it lacks (missing, floating or fixed into a leg link) costs only that
             part's own items, a leg tag it carries only keeps that leg's link from matching (conversion.body lists
             both); a leg link matches the reached group whose tag set equals the link's exactly and whose tree parent
             is the group matched to the link's parent;
  cores      each revolute joint of the tree is driven by a core rigidly in its parent group whose +z lies within 5 deg of
             the joint axis: among those within 2 mm of the axis line the one whose output centre is nearest to the
             child group's parts, else the nearest line (inconsistent); one core per joint, assigned greedily over all
             (joint, core) pairs in that order (the coaxial front / hind HAA cores cannot both go to one joint);
  sense      positive = the driving core's +z; the saved pose is zero; limits = AngleMin / AngleMax relative to the
             saved angle (the signed angle of the two joint frames' x axes about the first's z), negated for a
             child-first joint and again for a frame whose z opposes the core's +z, then clipped to [-pi, pi]; a
             declared range a full turn wide is the full turn [-pi, pi] whatever the saved angle;
  legs       a leg is valid when none of its parts is missing, its three links are matched, each joint has a driving
             core and is consistent (axis within 1 deg and 0.5 mm of the core's output axis; the child's parts rest on
             the parent's or on the core: gap <= 0.2 mm) and its lower link carries exactly one foot pad;
  alignment  6 DoF, agent world -> robot frame: a Kabsch seed on the valid legs' HAA cores (output centre and the point
             50 mm along the output axis, matched by leg, that point on whichever side fits: a unit mounted the other way
             round does not turn the robot; the best-supported pair of non-collinear legs, ties broken by the body
             parts' fit), else on the body parts' centroids; refined by the restricted point-to-plane ICP of the agent's
             body parts onto the reference's (15 mm / 15 deg from the seed, fixed sample seeds);
  writer     links world-aligned at q = 0 in the aligned frame, a link's origin = its driving core's output centre
             projected onto the joint axis (the body: the robot origin); every valid part as STEP + BREP in the aligned
             robot frame (file frame) with T_link = the translation into its link (parts of groups that match no
             written link: link null, T_link identity); cores and foot pads as occurrences; joints depth-first (leg by
             leg: MuJoCo's qpos order); foot points = the foot pads' origins in the lower-leg frames.
"""
import itertools
import json
import math
import os

import numpy as np

try:  # verify stage: flat vendored copies
    import cad_drawings_check as C
    import cad_photo_align as A
except ImportError:  # the chain's shared package
    from arm.cad_drawings import checks as C
    from arm.cad_photo import align as A
import quad_task as Q

VERSION = "quad-convert/1"
TINY_RANGE_RAD = 1e-6          # a joint with an empty declared range is written as [-1e-6, 1e-6] (it cannot move)
FULL_TURN_TOL_RAD = 1e-6       # a declared range at least 2 pi - this wide is the full turn (FreeCAD stores degrees)


# ---------------------------------------------------------------------------------------------------- small helpers
def _m(a):
    return np.asarray(a, float).reshape(4, 4)


def _unit(v):
    v = np.asarray(v, float)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _apply(T, P):
    P = np.asarray(P, float)
    return P @ T[:3, :3].T + T[:3, 3]


def _rot_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(R)) - 1.0) / 2.0))))


def _axis_angle_deg(a, b):
    """Angle between two lines' directions (either sign), degrees."""
    c = abs(float(np.dot(_unit(a), _unit(b))))
    return math.degrees(math.acos(min(1.0, c)))


def _translation(v):
    T = np.eye(4)
    T[:3, 3] = v
    return T


def _f(x, nd=6):
    return float(round(float(x), nd))


def _mat_list(T):
    return [[float(v) for v in row] for row in np.asarray(T, float)]


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


class _UF(object):
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:                      # deterministic representative: the smaller id
            if rb < ra:
                ra, rb = rb, ra
            self.p[rb] = ra


def _num_or_nan(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _finite_m(a):
    """True for a 4 x 4 matrix of finite numbers."""
    try:
        m = np.asarray(a, float)
    except (TypeError, ValueError):
        return False
    return m.shape == (4, 4) and bool(np.all(np.isfinite(m)))


def _broken_why(j):
    """Why a revolute joint cannot be read (None when it can): its own Placement1 / Placement2 not finite (a broken
    joint: FreeCAD keeps a NaN there across a save, and its solver cannot use the joint), or neither of its two global
    frames finite (the placements of both components it references are not finite). A frame that is not finite because
    only ONE referenced component's placement is not finite is taken from the other reference (_groups)."""
    bad = [k for k in ("1", "2") if j.get("placement%s_finite" % k) is False]
    if bad:
        return "its Placement%s %s not finite (a broken joint)" % (" and Placement".join(bad), "are" if len(bad) > 1 else "is")
    if not _finite_m(j.get("jcs1")) and not _finite_m(j.get("jcs2")):
        return "neither of its frames is finite (the placements of the components it references are not finite)"
    return None


def _line_distance(p1, d1, p2, d2):
    n = np.cross(d1, d2)
    if np.linalg.norm(n) < 1e-9:
        v = p2 - p1
        return float(np.linalg.norm(v - np.dot(v, d1) * d1))
    return float(abs(np.dot(p2 - p1, n)) / np.linalg.norm(n))


def _signed_angle(x1, x2, z):
    return math.atan2(float(np.dot(np.cross(x1, x2), z)), float(np.dot(x1, x2)))


def kabsch(P, G):
    """Rigid T (4x4) minimising sum |T p - g|^2."""
    P, G = np.asarray(P, float), np.asarray(G, float)
    cp, cg = P.mean(axis=0), G.mean(axis=0)
    H = (P - cp).T @ (G - cg)
    U, _S, Vt = np.linalg.svd(H)
    d = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cg - R @ cp
    return T


def _collinear(P, rel=1e-3):
    P = np.asarray(P, float)
    if len(P) < 3:
        return True
    s = np.linalg.svd(P - P.mean(axis=0), compute_uv=False)
    return bool(s[0] <= 1e-9 or s[1] <= rel * s[0])


def _transform_solid(solid, T):
    """A rigid transform of a solid (the rotation re-orthonormalised: OCC rejects a numerically skewed one), applied as a
    location, not as a geometric copy (fix of 2026-10-28, grader QC on the real geometry): a copy's curves and surfaces
    differ from the original's by round-off, and a vertex whose tolerance OCC left exactly at its deviation from its
    edges (what healing on a STEP read leaves; B-spline-heavy parts carry many) then fails BRepCheck_Analyzer by about
    1e-13 mm when the part's BREP file is read back (BRepCheck_InvalidPointOnCurve). An oracle document built from STEP
    parts lost 3 to 7 of its 42 parts as invalid, which ones depending on the alignment's round-off (host and task image
    differ). A location leaves the geometry as it is."""
    import cadquery as cq
    from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCP.gp import gp_Trsf
    U, _s, Vt = np.linalg.svd(np.asarray(T, float)[:3, :3])
    Rm = U @ Vt
    t = np.asarray(T, float)[:3, 3]
    tr = gp_Trsf()
    tr.SetValues(*(float(v) for v in np.hstack([Rm, t[:, None]]).ravel()))
    return cq.Shape.cast(BRepBuilderAPI_Transform(solid.wrapped, tr, False).Shape())


def _brep_sibling(path):
    return os.path.splitext(path)[0] + ".brep"


def _write_step(solid, path):
    import cadquery as cq
    cq.exporters.export(cq.Workplane().add(solid), path, exportType="STEP")
    solid.exportBrep(_brep_sibling(path))


def refresh_tolerances(solids):
    """The solids of a BREP file, each one that fails BRepCheck_Analyzer with its tolerances recomputed
    (BRepLib::UpdateTolerances; geometry unchanged). Fix of 2026-10-28 (grader QC on the real geometry): a part
    transformed in FreeCAD (a Placement baked into the shape, Part.transformShape) can come out of the extractor's BREP a
    round-off past its vertex tolerances (BRepCheck_InvalidPointOnCurve, about 1e-13 mm) although FreeCAD and the STEP
    sibling call it valid; it was dropped as invalid (4 upper-leg shells of a uniformly scaled oracle). Used by this
    module's BREP-first reader and by quad_grade._load_solids."""
    from OCP.BRepCheck import BRepCheck_Analyzer
    from OCP.BRepLib import BRepLib
    for s in solids:
        if not BRepCheck_Analyzer(s.wrapped).IsValid():
            BRepLib.UpdateTolerances_s(s.wrapped, False)
    return solids


def install_tess_retry():
    """Make every loaded copy of the cad-drawings checks re-tessellate a solid whose default mesh (TESS_TOL / TESS_ANG,
    a relative deflection) is not watertight: once at an angle of 0.1 rad, then at a fifth of the deflection, each from a
    cleaned triangulation, before the part counts as not watertight (the first watertight mesh wins; a watertight default
    mesh is kept as it is). Fix of 2026-10-28 (grader QC on the real geometry): on valid B-spline faces (a non-uniform
    scale, a loft) the default edge discretisations of neighbouring faces need not meet, and 11 to 12 of the 42 parts of
    an oracle scaled per part by +-0.5 to 2 % were dropped (two legs invalid, 0.41 to 0.55 instead of 0.93 to 1.00).
    Called by use_brep_siblings and quad_grade._mesh_from_solid_file."""
    import sys
    for name in ("cad_drawings_check", "arm.cad_drawings.checks"):
        mod = sys.modules.get(name)
        if mod is None or getattr(mod.solid_to_trimesh, "_retry", False):
            continue
        base = mod.solid_to_trimesh

        def tess(solid, tol=mod.TESS_TOL, ang=mod.TESS_ANG, _base=base):
            from OCP.BRepTools import BRepTools
            mesh, notes = _base(solid, tol, ang)
            if mesh is not None and mesh.is_watertight:
                return mesh, notes
            for t2, a2 in ((tol, min(ang, 0.1)), (tol / 5.0, min(ang, 0.1))):
                BRepTools.Clean_s(solid.wrapped)
                m2, n2 = _base(solid, t2, a2)
                if m2 is not None and m2.is_watertight:
                    return m2, list(n2) + ["re-tessellated at %.3f / %.2f (the default mesh was not watertight)" % (t2, a2)]
            return mesh, notes
        tess._retry = True
        mod.solid_to_trimesh = tess


def use_brep_siblings():
    """Make every loaded copy of the cad-drawings checks read a part from the BREP file beside its STEP when there is one
    (the arm's fix: a STEP round trip can turn a valid solid invalid; the BREP file holds the shape exactly). The
    extractor writes both for the agent's leaves and _write_step for the converted parts; the reference parts have no
    BREP siblings and are read from STEP. Also installs the tessellation retry (install_tess_retry) and refreshes the
    tolerances of a BREP solid that fails BRepCheck (refresh_tolerances)."""
    import sys
    install_tess_retry()
    for name in ("cad_drawings_check", "arm.cad_drawings.checks"):
        mod = sys.modules.get(name)
        if mod is None or getattr(mod.load_step_solids, "_brep_first", False):
            continue
        step_reader = mod.load_step_solids

        def load(path, _step_reader=step_reader):
            brep = _brep_sibling(path)
            if not os.path.isfile(brep):
                return _step_reader(path)
            import cadquery as cq
            try:
                shape = cq.Shape.importBrep(brep)
            except Exception as exc:  # noqa: BLE001
                return [], "BREP reader failed: {}".format(str(exc)[:200])
            return refresh_tolerances(list(shape.Solids())), None
        load._brep_first = True
        mod.load_step_solids = load


def _samples(mesh, n, seed):
    pts, normals = A.sample_surface(mesh, int(n), int(seed))
    return pts, normals


# ---------------------------------------------------------------------------------------------------- worker pool
def _pool_run(fn, jobs, workers, budget_s, notes, min_total_mb=0.0):
    """{key: fn(args)} for jobs [(key, args)]: in spawned worker processes when there are several jobs and their files
    total at least `min_total_mb` (a spawned worker imports the kernel afresh; light documents run in-process); results
    are keyed, so the outcome does not depend on the completion order; a job not finished within `budget_s` (all jobs
    together) gets {"error": ...}; in-process when the pool cannot start. Spawn, not fork: verify.py runs its agent
    freezer in a thread, and a forked child can inherit a lock held by it."""
    results = {}

    def size(a):                              # the largest files first (load balance)
        try:
            return os.path.getsize(a if isinstance(a, str) else a[0])
        except (OSError, TypeError, IndexError):
            return 0
    jobs = sorted(jobs, key=lambda kv: (-size(kv[1]), str(kv[0])))
    try:
        workers = min(int(workers), len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        workers = min(int(workers), os.cpu_count() or 1)
    total_mb = sum(size(a) for _k, a in jobs) / 1e6
    if workers > 1 and len(jobs) > 1 and total_mb >= float(min_total_mb):
        try:
            import multiprocessing as mp
            import time as _t
            pool = mp.get_context("spawn").Pool(min(workers, len(jobs)))
            try:
                pending = [(k, pool.apply_async(fn, (a,))) for k, a in jobs]
                t_end = _t.time() + float(budget_s)
                for k, r in pending:
                    try:
                        results[k] = r.get(max(1.0, t_end - _t.time()))
                    except mp.TimeoutError:
                        results[k] = {"error": "not finished within the %.0f s budget" % float(budget_s)}
            finally:
                pool.terminate()
                pool.join()
        except Exception as exc:  # noqa: BLE001 - no pool: in-process below
            notes.append("worker pool unavailable (%s); in-process" % str(exc)[:120])
            results = {k: v for k, v in results.items() if isinstance(v, dict) and "error" not in v}
    for k, a in jobs:
        if k not in results:
            results[k] = fn(a)
    return results


def _body_job(path):
    """Worker: one part's C.Body (validity, watertight tessellation) as plain arrays (OCC shapes do not pickle)."""
    use_brep_siblings()
    try:
        sols, err = C.load_step_solids(path)
        if err or len(sols) != 1:
            return {"error": err or "%d solids on re-reading" % len(sols)}
        b = C.Body("part", sols[0])
        if not b.ok:
            return {"error": "; ".join(b.errors) or "not a valid watertight solid"}
        return {"vertices": np.asarray(b.mesh.vertices, float), "faces": np.asarray(b.mesh.faces, np.int64),
                "notes": list(b.notes), "solid_volume": float(b.solid_volume), "void_ratio": float(b.void_ratio)}
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}


def _mesh_job(path):
    """Worker: a reference STEP's tessellation (vertices, faces)."""
    use_brep_siblings()
    try:
        sols, err = C.load_step_solids(path)
        if err or len(sols) != 1:
            return {"error": err or "%d solids" % len(sols)}
        mesh, _n = C.solid_to_trimesh(sols[0])
        return {"vertices": np.asarray(mesh.vertices, float), "faces": np.asarray(mesh.faces, np.int64)}
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}


def _write_job(args):
    """Worker: read a part's shape file, move it rigidly, write it as STEP + BREP."""
    src, T, dst = args
    use_brep_siblings()
    try:
        sols, err = C.load_step_solids(src)
        if err or len(sols) != 1:
            return {"error": err or "%d solids" % len(sols)}
        _write_step(_transform_solid(sols[0], np.asarray(T, float)), dst)
        return {"ok": True}
    except Exception as exc:  # noqa: BLE001
        return {"error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}


def _body_from_job(node, rec):
    """A C.Body rebuilt from a worker's arrays (the same tessellation; no second meshing; no B-rep in this process)."""
    import trimesh
    b = C.Body.__new__(C.Body)
    b.name, b.solid, b.errors, b.notes = node, None, [], list(rec["notes"])
    b.mesh = trimesh.Trimesh(rec["vertices"], rec["faces"], process=False)
    b.manifold = C.trimesh_to_manifold(b.mesh)
    if str(b.manifold.status()) != "Error.NoError":
        b.errors.append("manifold construction failed: {}".format(b.manifold.status()))
        b.manifold = None
    b.volume = float(b.mesh.volume)
    b.centroid = np.asarray(b.mesh.center_mass, float)
    b.bounds = np.asarray(b.mesh.bounds, float)
    b.solid_volume, b.void_ratio = rec["solid_volume"], rec["void_ratio"]
    return b


# ---------------------------------------------------------------------------------------------------- the conversion
class _Ctx(object):
    """The state of one conversion (kept on one object so the steps stay small)."""

    def __init__(self, extract_dir, spec, given_dir, settings):
        self.dir = extract_dir
        self.spec = spec
        self.given = given_dir
        self.s = settings
        self.errors, self.warnings, self.notes = [], [], []
        self.stats, self.details = {}, {}
        self.bodies = {}          # leaf node -> C.Body (world frame; mesh and manifold, no B-rep)
        self.body_errors = {}
        self._trees = {}          # leaf node -> KD-tree of its surface (the tie-break distance)
        self._core_man = None

    def shape_path(self, node):
        leaf = self.leaf[node]
        rel = leaf.get("step") or leaf.get("brep")
        return os.path.join(self.dir, rel) if rel else None

    def pool(self, fn, jobs):
        return _pool_run(fn, jobs, int(self.s.get("workers", 8) or 1), float(self.s.get("part_timeout_s", 300.0)), self.notes,
                         float(self.s.get("pool_min_total_mb", 2.0)))

    def preload(self, nodes):
        """C.Body of every node (validity and tessellation in parallel worker processes)."""
        todo = [(n, self.shape_path(n)) for n in nodes if n not in self.bodies and n not in self.body_errors]
        todo = [(n, p) for n, p in todo if p]
        results = self.pool(_body_job, todo)
        for n, _p in todo:
            rec = results[n]
            if "error" in rec:
                self.body_errors[n] = rec["error"]
                continue
            b = _body_from_job(n, rec)
            if not b.ok:
                self.body_errors[n] = "; ".join(b.errors)
                continue
            self.bodies[n] = b

    def surface_distance(self, node, point):
        """Distance from a point to a leaf's surface (its tessellation's vertices and 4000 surface samples)."""
        if node not in self._trees:
            b = self.bodies.get(node)
            if b is None:
                self.preload([node])
                b = self.bodies.get(node)
            if b is None:
                self._trees[node] = None
            else:
                from scipy.spatial import cKDTree
                pts, _n = _samples(b.mesh, 4000, 1009)
                self._trees[node] = cKDTree(np.vstack([np.asarray(b.mesh.vertices, float), pts]))
        tree = self._trees[node]
        return float(tree.query(np.asarray(point, float))[0]) if tree is not None else float("inf")

    def body(self, node):
        if node in self.bodies:
            return self.bodies[node]
        if node in self.body_errors:
            return None
        if self.shape_path(node) is None:
            self.body_errors[node] = "no shape file"
            return None
        self.preload([node])
        return self.bodies.get(node)

    def core_manifold(self, T):
        if self._core_man is None:
            meta = Q.load_component(self.given, "core")
            sols, err = C.load_step_solids(meta["step_path"])
            if err or not sols:
                raise RuntimeError("the given core STEP does not load: %s" % err)
            man = None
            for s in sols:
                mesh, _n = C.solid_to_trimesh(s)
                m = C.trimesh_to_manifold(mesh)
                man = m if man is None else man + m
            self._core_man = man
        return C.manifold_transform(self._core_man, T)


def _read_model(ctx):
    path = os.path.join(ctx.dir, "model.json")
    if not os.path.isfile(path):
        ctx.errors.append("the document could not be read (no extraction record)")
        return None
    try:
        with open(path) as fh:
            model = json.load(fh)
    except (OSError, ValueError) as exc:
        ctx.errors.append("the extraction record is unreadable (%s)" % str(exc)[:200])
        return None
    if model.get("errors"):
        ctx.errors += ["document: " + str(e)[:400] for e in model["errors"]]
        return None
    n_top = int(model.get("n_top_assemblies", 1) or 0)
    if n_top != 1:
        ctx.errors.append("the document has %d top-level Assemblies with components (%s); one is expected" % (
            n_top, ", ".join(model.get("top_assemblies", []))[:200]))
    ctx.warnings += ["document: " + str(w)[:300] for w in model.get("warnings", [])]
    return model


def _collect(ctx, model):
    """Leaves by node; tags -> leaves; cores and feet; part validity."""
    spec = ctx.spec
    ctx.leaf, ctx.cores, ctx.feet = {}, {}, {}
    ctx.tag_of = {}                       # printed leaf node -> its tag (component-class leaves never carry one)
    by_tag, unknown, unusable = {}, [], []
    for comp in model.get("components", []):
        for leaf in comp.get("leaves", []):
            node = leaf["node"]
            ctx.leaf[node] = leaf
            if leaf.get("error"):
                ctx.warnings.append("component %s: %s" % (leaf.get("path", node), str(leaf["error"])[:200]))
            cls = leaf.get("component_class")
            if cls in ("core", "foot") and not leaf.get("error"):
                if leaf.get("placement_finite") is False:
                    unusable.append("%s %s (the placement of %s is not finite)" % (
                        cls, leaf.get("label") or node, leaf.get("placement_nonfinite") or "it"))
                    continue
                if not leaf.get("pose_ok"):
                    unusable.append("%s %s (%s)" % (cls, leaf.get("label") or node, (leaf.get("pose_check") or {}).get("note", "pose")))
                    continue
                (ctx.cores if cls == "core" else ctx.feet)[node] = leaf
                continue
            label = leaf.get("label") or ""
            if cls in ("core", "foot"):
                continue                       # an unreadable component-class leaf: neither a part nor a component
            if Q.TAG_RE.match(label):
                if label in spec.parts:
                    by_tag.setdefault(label, []).append(node)
                    ctx.tag_of[node] = label
                else:
                    unknown.append(label)
    ctx.by_tag = by_tag
    cand = [nodes[0] for t, nodes in sorted(by_tag.items()) if len(nodes) == 1 and not ctx.leaf[nodes[0]].get("error")
            and ctx.leaf[nodes[0]].get("placement_finite") is not False
            and ctx.leaf[nodes[0]].get("valid") and ctx.leaf[nodes[0]].get("n_solids") == 1]
    ctx.preload(cand)
    missing = {}
    ctx.part_node = {}
    for tag in spec.tags:
        nodes = by_tag.get(tag, [])
        if not nodes:
            missing[tag] = "no component labelled %s" % tag
            continue
        if len(nodes) > 1:
            missing[tag] = "label %s is used %d times" % (tag, len(nodes))
            continue
        leaf = ctx.leaf[nodes[0]]
        if leaf.get("error"):
            missing[tag] = "unreadable (%s)" % str(leaf["error"])[:120]
        elif leaf.get("placement_finite") is False:
            # FreeCAD keeps a NaN in an App::Link's Placement and Part.getShape then falls back silently to the
            # linked Body's own placement: the part's pose is undefined (input review fix 2026-10-28)
            missing[tag] = "its placement is not finite (%s)" % (leaf.get("placement_nonfinite") or "a NaN")
        elif not leaf.get("valid") or leaf.get("n_solids") != 1:
            missing[tag] = "not one valid solid (%s solids, valid %s)" % (leaf.get("n_solids"), leaf.get("valid"))
        elif ctx.body(nodes[0]) is None:
            missing[tag] = "not a valid watertight solid (%s)" % ctx.body_errors.get(nodes[0], "?")[:160]
        else:
            ctx.part_node[tag] = nodes[0]
    ctx.missing = missing
    if unknown:
        ctx.warnings.append("labels of the tag form that the photographed unit does not carry: %s" % ", ".join(sorted(set(unknown))))
    if unusable:
        ctx.warnings.append("components of a given shape class whose geometry is not the given one at their placement "
                            "(not used): %s" % "; ".join(unusable)[:400])
    ctx.unusable = unusable
    ctx.stats.update(n_parts_labelled=float(sum(1 for t in spec.tags if by_tag.get(t))), n_parts_valid=float(len(ctx.part_node)),
                     n_parts_missing=float(len(missing)), n_cores_found=float(len(ctx.cores)), n_feet_found=float(len(ctx.feet)),
                     n_components_unusable=float(len(unusable)))


def _groups(ctx, model):
    uf = _UF()
    for node in ctx.leaf:
        uf.find(node)
    containers = {c["node"]: c for c in model.get("containers", [])}
    for node, c in sorted(containers.items()):
        uf.find(node)
        # a rigid unit (an App::Part; a rigid sub-assembly without inner joints): one group. A rigid sub-assembly whose
        # inner joints the extractor read is grouped by those joints (GRADER_SPEC 2: read like top-level joints)
        if c.get("unit", c.get("rigid")):
            for ch in c.get("children", []):
                uf.union(node, ch)
    revolute, ground, fixed_n = [], [], 0
    ctx.broken = []                           # (joint record, why): revolute joints not read (their legs are invalid)
    for g in model.get("grounded", []):
        if g.get("foreign") or g.get("suppressed"):
            continue
        if g.get("node") is None:
            ctx.warnings.append("grounded joint %s: %s" % (g.get("label"), g.get("error") or "its object is not a component of the Assembly"))
            continue
        owner = g.get("owner")
        if owner in (None, "top"):
            ground.append(g["node"])
        else:                                 # grounded inside a sub-assembly: fixed to that sub-assembly
            uf.union(owner, g["node"])
    for j in model.get("joints", []):
        if j.get("foreign"):
            continue
        if j.get("suppressed"):
            ctx.notes.append("joint %s is suppressed (ignored)" % j.get("label"))
            continue
        jt = j.get("type")
        if jt not in ("Fixed", "Revolute"):
            ctx.errors.append("joint %s is of type %s (only Grounded, Fixed and Revolute joints are read)" % (j.get("label"), jt))
            continue
        if j.get("error"):
            ctx.warnings.append("joint %s could not be read: %s" % (j.get("label"), str(j["error"])[:200]))
            continue
        p1, p2 = j.get("part1"), j.get("part2")
        if p1 is None or p2 is None:
            ctx.warnings.append("joint %s: a reference reaches no component of the Assembly (%s)" % (
                j.get("label"), j.get("note1") or j.get("note2") or "unresolved"))
            continue
        if jt == "Fixed":
            uf.union(p1, p2)
            fixed_n += 1
            continue
        why = _broken_why(j)
        if why:
            ctx.broken.append((j, why))
            ctx.warnings.append("revolute joint %s is not read: %s" % (j.get("label"), why))
            continue
        f1, f2 = _finite_m(j.get("jcs1")), _finite_m(j.get("jcs2"))
        if not f1 or not f2:
            j = dict(j, frame_from=2 if not f1 else 1)
            ctx.warnings.append("revolute joint %s: its frame on %s is not finite (that component's placement is not "
                                "finite); its frame on %s is used and its saved angle is read as 0" % (
                                    j.get("label"), "the first reference" if not f1 else "the second reference",
                                    "the second" if not f1 else "the first"))
        revolute.append(j)
    if not ground:
        ctx.errors.append("no Grounded joint on a component of the Assembly (the body must be grounded)")
        return None
    for g in ground[1:]:
        uf.union(ground[0], g)
    ctx.uf = uf
    ctx.g0 = uf.find(ground[0])
    groups = {}
    for node in sorted(ctx.leaf):
        groups.setdefault(uf.find(node), []).append(node)
    if ctx.g0 not in groups:
        groups[ctx.g0] = []
    ctx.groups = groups
    ctx.revolute = sorted(revolute, key=lambda j: (str(j.get("label")), str(j.get("name"))))
    ctx.stats.update(n_revolute=float(len(revolute)), n_fixed=float(fixed_n), n_grounded=float(len(ground)),
                     n_groups=float(len(groups)), n_revolute_broken=float(len(ctx.broken)),
                     n_revolute_one_frame=float(sum(1 for j in revolute if j.get("frame_from"))))
    return groups


def _tree(ctx):
    uf, groups = ctx.uf, ctx.groups
    adj = {}
    loops = []
    for j in ctx.revolute:
        a, b = uf.find(j["part1"]), uf.find(j["part2"])
        if a == b:
            loops.append("revolute joint %s connects two components of one rigid group" % j.get("label"))
            continue
        adj.setdefault(a, []).append((b, j))
        adj.setdefault(b, []).append((a, j))
    parent = {ctx.g0: None}
    order = [ctx.g0]
    queue = [ctx.g0]
    seen_loop = set()
    while queue:
        g = queue.pop(0)
        came = parent[g][1]["name"] if parent[g] else None
        for h, j in sorted(adj.get(g, []), key=lambda e: (str(e[1].get("label")), str(e[1].get("name")))):
            if j["name"] == came:
                continue
            if h not in parent:
                parent[h] = (g, j)
                order.append(h)
                queue.append(h)
            elif j["name"] not in seen_loop and (parent[h] is None or parent[h][1]["name"] != j["name"]):
                seen_loop.add(j["name"])
                loops.append("revolute joint %s closes a loop" % j.get("label"))
    for msg in loops:
        ctx.errors.append(msg + " (the mechanism must be a tree)")
    floating = [g for g in sorted(groups) if g not in parent]
    ctx.parent, ctx.order = parent, order
    ctx.floating = floating
    if floating:
        tags = sorted(ctx.tag_of[n] for g in floating for n in groups[g] if n in ctx.tag_of)
        ctx.warnings.append("%d group(s) not connected to the grounded body by the joints (floating; their parts: %s)" % (
            len(floating), ", ".join(tags)[:300] or "untagged"))
    ctx.stats.update(n_groups_reached=float(len(order)), n_groups_floating=float(len(floating)), n_loops=float(len(loops)))
    ctx.loops = loops


def _group_tags(ctx, g):
    """The tags carried by a group's printed leaves (valid or not: a link is identified by its labels)."""
    return sorted(set(ctx.tag_of[n] for n in ctx.groups.get(g, []) if n in ctx.tag_of))


def _body_claim(ctx, g):
    """(the body's tags group g carries, the other link of which it carries the most tags, how many): the grounded
    group stands for the body when the first count is positive and exceeds the second (ties: the first link in order)."""
    spec = ctx.spec
    have = set(_group_tags(ctx, g))
    n_body = len(have & set(spec.link_tags[spec.root]))
    rival, n_rival = None, 0
    for l in spec.link_order[1:]:
        n = len(have & set(spec.link_tags[l]))
        if n > n_rival:
            rival, n_rival = l, n
    return n_body, rival, n_rival


def _where(ctx, tag):
    """Where a tag of the photographed unit is in the document (a diagnostic): missing, or the link / group holding it."""
    if tag in ctx.missing:
        return "%s (%s)" % (tag, ctx.missing[tag])
    g = ctx.uf.find(ctx.part_node[tag])
    if g in ctx.floating:
        return "%s (floating: no joint chain to the grounded body)" % tag
    l = ctx.group_link.get(g) if hasattr(ctx, "group_link") else None
    return "%s (in %s)" % (tag, l or "the group of %s" % ", ".join(_group_tags(ctx, g)))


def _match(ctx):
    """Links by tag set (GRADER_SPEC 3.4, read with section 4 and DESIGN D5: missing parts only invalidate the legs or
    items they belong to; review fix 2026-10-27). The body is the grounded group when that group carries more of the
    body's tags than of any other single link's: a body tag it lacks (missing, floating or fixed into a leg link) costs
    that part's own items (its shape and features where it is, the body's mass and centre of mass), and a leg tag it
    carries keeps that leg link from matching (that leg only is invalid). A grounded group that does not stand for the
    body is cad_valid = 0. A leg link matches the reached group (never the grounded one) whose tag set equals its own
    exactly and whose tree parent is the group matched to the link's parent."""
    spec = ctx.spec
    root = spec.root
    link_group, link_reason = {}, {}
    n_body, rival, n_rival = _body_claim(ctx, ctx.g0)
    have, body_tags = set(_group_tags(ctx, ctx.g0)), set(spec.link_tags[root])
    ctx.body_missing, ctx.body_foreign = [], {}
    if n_body > 0 and n_body > n_rival:
        link_group[root] = ctx.g0
        # a body tag it lacks, or carries as a missing part (labelled, but not one valid solid or not placed)
        ctx.body_missing = sorted((body_tags - have) | (body_tags & set(ctx.missing)))
        ctx.body_foreign = {t: spec.tag_link[t] for t in sorted(have - body_tags)}
    else:
        link_reason[root] = "it carries %d of the body's %d tags%s" % (
            n_body, len(body_tags), " and %d of %s's" % (n_rival, rival) if rival else "")
    by_set = {}
    for g in ctx.order:
        if g != ctx.g0:
            by_set.setdefault(tuple(_group_tags(ctx, g)), []).append(g)
    for link in spec.link_order[1:]:                  # parents before children
        want = tuple(spec.link_tags[link])
        cands = by_set.get(want, [])
        if len(cands) > 1:
            link_reason[link] = "%d groups carry its tags %s" % (len(cands), list(want))
            continue
        if not cands:
            best, best_n = None, 0
            for g in ctx.order:
                n = len(set(want) & set(_group_tags(ctx, g)))
                if n > best_n:
                    best, best_n = g, n
            in_body = sorted(set(want) & have) if root in link_group else []
            fixed = "; %s fixed to the body" % ", ".join(in_body) if in_body else ""
            if best is None:
                link_reason[link] = "no group of the tree carries its tags %s%s" % (list(want), fixed)
            elif best == ctx.g0 and in_body:
                link_reason[link] = "no group carries exactly its tags %s (%s fixed to the body)" % (list(want), ", ".join(in_body))
            else:
                link_reason[link] = "no group carries exactly its tags %s (the closest carries %s%s)" % (
                    list(want), _group_tags(ctx, best), fixed)
            continue
        g = cands[0]
        par = spec.links[link]["parent"]
        pg = ctx.parent[g][0] if ctx.parent.get(g) else None
        if par not in link_group or pg != link_group[par]:
            link_reason[link] = "its group hangs from %s, not from %s" % (
                _describe_group(ctx, pg) if pg is not None else "nothing", par)
            continue
        link_group[link] = g
    ctx.link_group = link_group
    ctx.group_link = {g: l for l, g in link_group.items()}
    ctx.link_reason = link_reason
    if root not in link_group:
        # the counts first: the verdict's diagnostic keeps only the first 150 characters (integration lane, 2026-10-27)
        ctx.errors.append("the grounded group is not the body (%s): its tags %s; the photographed unit's body carries "
                          "%s" % (link_reason[root], sorted(have), spec.link_tags[root]))
    elif ctx.body_missing or ctx.body_foreign:
        bits = []
        if ctx.body_missing:
            bits.append("lacks %s: only those parts' own items and the body's mass properties are lost" % ", ".join(
                _where(ctx, t) for t in ctx.body_missing))
        if ctx.body_foreign:
            bits.append("carries %s: those links do not match, only their legs are invalid" % ", ".join(
                "%s of %s" % kv for kv in sorted(ctx.body_foreign.items())))
        ctx.warnings.append(("the body (the grounded group) " + "; it ".join(bits))[:600])
    ctx.stats["n_links_matched"] = float(len(link_group))
    ctx.stats["n_body_tags_missing"] = float(len(ctx.body_missing))
    ctx.stats["n_body_tags_foreign"] = float(len(ctx.body_foreign))


def _describe_group(ctx, g):
    l = ctx.group_link.get(g) if hasattr(ctx, "group_link") else None
    if l:
        return l
    tags = _group_tags(ctx, g)
    return "the group of %s" % (", ".join(tags) if tags else "untagged parts")


def _joint_frame(ctx, j, parent_group):
    """(point, axis z, theta0, child_first) of a revolute joint as FreeCAD defines it (JCS1's origin and z). When one of
    its frames is not finite (the placement of the component it references is not), the other one stands for both:
    the axis is the same, the saved angle reads 0."""
    J1, J2 = _m(j["jcs1"]), _m(j["jcs2"])
    if j.get("frame_from") == 2:
        J1 = J2
    elif j.get("frame_from") == 1:
        J2 = J1
    p, z = J1[:3, 3].copy(), _unit(J1[:3, 2])
    theta0 = _signed_angle(_unit(J1[:3, 0]), _unit(J2[:3, 0]), z)
    child_first = ctx.uf.find(j["part1"]) != parent_group
    return p, z, theta0, child_first, J1, J2


def _child_distance(ctx, point, child_group):
    """Distance from a point to the child group's printed parts (on their tessellations: a tie-break between cores
    hundreds of millimetres apart)."""
    best = float("inf")
    for n in ctx.groups.get(child_group, []):
        if n in ctx.cores or n in ctx.feet or ctx.leaf[n].get("component_class") or ctx.shape_path(n) is None:
            continue
        best = min(best, ctx.surface_distance(n, point))
    return best


def _core_axis(ctx, node):
    meta = ctx.core_meta
    P = _m(ctx.cores[node]["pose"])
    c = _apply(P, np.asarray(meta["output"]["point"], float)[None, :])[0]
    d = _unit(P[:3, :3] @ np.asarray(meta["output"]["axis"], float))
    return P, c, d


def _assign_cores(ctx):
    """One core per tree joint: candidates in the parent group within the axis angle; on-line ones by the child's
    nearest output centre, else by the line distance; greedy over all pairs in that order."""
    s = ctx.s
    core_group = {n: ctx.uf.find(n) for n in ctx.cores}
    pairs, edges = [], []
    for h in ctx.order[1:]:
        g, j = ctx.parent[h]
        p, z, theta0, child_first, _J1, J2 = _joint_frame(ctx, j, g)
        edges.append((h, g, j, p, z, theta0, child_first, J2))
        for n in sorted(ctx.cores):
            if core_group[n] != g:
                continue
            P, c, d = _core_axis(ctx, n)
            ang = _axis_angle_deg(d, z)
            if ang > s["core_axis_deg"]:
                continue
            line = _line_distance(p, z, c, d)
            on = line <= s["core_line_mm"]
            near = _child_distance(ctx, c, h) if on else float("inf")
            key = (0, near, line, ang, n) if on else (1, line, ang, n)
            pairs.append((key, j["name"], n, {"axis_angle_deg": ang, "line_mm": line, "child_mm": near, "on_line": on}))
    pairs.sort(key=lambda r: (r[0], r[1]))
    chosen, used = {}, set()
    n_cand = {}
    for _key, jn, n, info in pairs:
        n_cand[jn] = n_cand.get(jn, 0) + 1
    for _key, jn, n, info in pairs:
        if jn in chosen or n in used:
            continue
        chosen[jn] = (n, info)
        used.add(n)
    ctx.edges = edges
    ctx.core_choice = chosen
    ctx.core_candidates = n_cand


def _limits(j, theta0, child_first, sign):
    """The joint's declared range about the driving core's +z, relative to the saved pose (the arm's rule). FreeCAD's
    angle (the signed angle of the two frames' x axes about the first frame's z) stays within [-180, 180] deg, so a
    declared interval at least a full turn wide bounds nothing: it is read as the full turn [-pi, pi] whatever the saved
    angle theta0 (shifting it by theta0 and clipping would drop |theta0| of it; review fix 2026-10-27). A narrower one
    is shifted and clipped to [-pi, pi] (the part beyond -pi or pi, reachable through the wrap, is not credited)."""
    notes = []
    bad = [k for k in ("min", "max") if j.get("enable_angle_" + k) and not math.isfinite(_num_or_nan(j.get("angle_%s_deg" % k)))]
    if bad:                              # an enabled limit that is not a number bounds nothing that can be read: empty
        notes.append("the enabled angle limit%s %s not finite: the declared range is read as empty" % (
            "s" if len(bad) > 1 else "", " and ".join("Angle" + k.capitalize() for k in bad) + (" are" if len(bad) > 1 else " is")))
        return -TINY_RANGE_RAD, TINY_RANGE_RAD, notes
    lo = math.radians(j.get("angle_min_deg", 0.0)) - theta0 if j.get("enable_angle_min") else -math.pi
    hi = math.radians(j.get("angle_max_deg", 0.0)) - theta0 if j.get("enable_angle_max") else math.pi
    if not j.get("enable_angle_min") or not j.get("enable_angle_max"):
        notes.append("angle limits not enabled (read as -pi / pi)")
    if child_first:                      # Reference1 is the child: FreeCAD's angle is the parent's relative to the child
        lo, hi = -hi, -lo
    if sign < 0:
        lo, hi = -hi, -lo
    if hi - lo >= 2.0 * math.pi - FULL_TURN_TOL_RAD:
        if lo < -math.pi - FULL_TURN_TOL_RAD or hi > math.pi + FULL_TURN_TOL_RAD:
            notes.append("a full turn declared (%.1f .. %.1f deg at a saved angle of %.1f deg): read as the full turn" % (
                j.get("angle_min_deg", 0.0), j.get("angle_max_deg", 0.0), math.degrees(theta0)))
        lo, hi = -math.pi, math.pi
    lo, hi = max(lo, -math.pi), min(hi, math.pi)
    if not lo < hi:
        notes.append("the declared range is empty (%.1f .. %.1f deg): the joint cannot move" % (math.degrees(lo), math.degrees(hi)))
        lo, hi = -TINY_RANGE_RAD, TINY_RANGE_RAD
    elif not lo <= 0.0 <= hi:
        notes.append("the saved pose lies outside the declared range")
    return lo, hi, notes


def _min_gap(ctx, child_nodes, others, core_T=None):
    """Smallest gap between the child's parts and the other parts (or the core body), mm (search radius clipped)."""
    lim = float(ctx.s["contact_search_mm"])
    best = lim
    cm = [ctx.body(n) for n in child_nodes]
    cm = [b for b in cm if b is not None]
    om = [ctx.body(n) for n in others]
    om = [b for b in om if b is not None]
    for a in cm:
        for b in om:
            if np.any(a.bounds[0] > b.bounds[1] + lim) or np.any(b.bounds[0] > a.bounds[1] + lim):
                continue
            best = min(best, float(a.manifold.min_gap(b.manifold, lim)))
        if core_T is not None:
            best = min(best, float(a.manifold.min_gap(ctx.core_manifold(core_T), lim)))
    return best if cm else float("inf")


def _tag_nodes(ctx, g):
    """The valid tagged parts of a group (the link's geometry as the tier carries it)."""
    return [ctx.part_node[t] for t in _group_tags(ctx, g) if t in ctx.part_node and ctx.uf.find(ctx.part_node[t]) == g]


def _joints(ctx):
    """Per tree joint: the driving core, the sense, the limits, the consistency."""
    s = ctx.s
    ctx.jinfo = {}
    for h, g, j, p, z, theta0, child_first, J2 in ctx.edges:
        # the two joint frames should share the axis at the saved pose (a solved document); a mismatch is reported
        jcs_gap = float(C.line_dist(p, z, J2[:3, 3]))
        jcs_z = _axis_angle_deg(z, J2[:3, 2])
        if jcs_gap > s["consistent_offset_mm"] or jcs_z > s["consistent_axis_deg"]:
            ctx.warnings.append("joint %s: its two frames disagree at the saved pose (%.2f mm off the axis, %.2f deg); the "
                                "first one's axis is used" % (j.get("label"), jcs_gap, jcs_z))
        rec = {"document_joint": j.get("label"), "name": j.get("name"), "parent_group": g, "child_group": h,
               "point": p, "z": z, "theta0": theta0, "child_first": child_first, "candidates": ctx.core_candidates.get(j["name"], 0),
               "jcs_gap_mm": jcs_gap, "jcs_z_deg": jcs_z}
        ch = ctx.core_choice.get(j["name"])
        if ch is None:
            rec.update(core=None, sign=1.0, axis=z, consistent=False,
                       reason="no core rigidly in the parent link with its +z within %.0f deg of the joint axis" % s["core_axis_deg"])
        else:
            n, info = ch
            P, c, d = _core_axis(ctx, n)
            sign = 1.0 if float(np.dot(d, z)) >= 0 else -1.0
            axis = sign * z
            origin = p + float(np.dot(c - p, axis)) * axis
            offset = float(C.line_dist(c, d, origin))
            ang = _axis_angle_deg(d, z)
            child_nodes = _tag_nodes(ctx, h)
            gap = _min_gap(ctx, child_nodes, _tag_nodes(ctx, g), core_T=P)
            ok = ang <= s["consistent_axis_deg"] and offset <= s["consistent_offset_mm"] and gap <= s["contact_mm"]
            reason = None
            if not ok:
                bits = []
                if ang > s["consistent_axis_deg"]:
                    bits.append("axis %.2f deg from the core's output axis" % ang)
                if offset > s["consistent_offset_mm"]:
                    bits.append("axis %.2f mm off the core's output axis" % offset)
                if gap > s["contact_mm"]:
                    bits.append("the child's parts are %s from the parent's and the core" % (
                        "%.2f mm" % gap if gap < s["contact_search_mm"] else "more than %.0f mm" % s["contact_search_mm"]))
                reason = "; ".join(bits)
            rec.update(core=n, core_P=P, core_c=c, core_d=d, sign=sign, axis=axis, origin=origin, offset_mm=offset,
                       axis_angle_deg=ang, line_mm=info["line_mm"], child_mm=info["child_mm"], on_line=info["on_line"],
                       gap_mm=gap, consistent=bool(ok), reason=reason)
        lo, hi, notes = _limits(j, theta0, child_first, rec["sign"])
        rec.update(lo=lo, hi=hi, limit_notes=notes)
        ctx.jinfo[h] = rec


def _broken_by_leg(ctx):
    """{leg: [reason]} for the revolute joints that were not read (ctx.broken), by the legs whose tags their two
    components' groups carry (a diagnostic: the leg is already invalid, its links below the joint float)."""
    spec, out = ctx.spec, {}
    for j, why in getattr(ctx, "broken", []):
        legs = set()
        for k in ("part1", "part2"):
            n = j.get(k)
            if n is None:
                continue
            for t in _group_tags(ctx, ctx.uf.find(n)):
                leg = Q.leg_of(spec.tag_link.get(t, ""))
                if leg:
                    legs.add(leg)
        for leg in sorted(legs):
            out.setdefault(leg, []).append("joint %s is not read (%s)" % (j.get("label"), why))
    return out


def _legs(ctx):
    """Per-leg validity (GRADER_SPEC 3.6). Besides the links, joints, cores and the foot pad, none of the leg's parts may
    be missing (3.1: a tag absent or used twice, not one valid solid, not a watertight solid, an undefined placement):
    the links are identified by their labels whether or not the parts are valid (so an invalid part does not hide its
    link), and a missing part then invalidates its leg as GRADER_SPEC 4 says, whichever way it is missing (input review
    fix 2026-10-28: a leg part with a stray second solid left its leg valid while the same part removed invalidated it)."""
    spec = ctx.spec
    legs = {}
    broken = _broken_by_leg(ctx)
    for leg in spec.legs:
        reasons = []
        links = Q.leg_links(leg)
        for l in links:
            miss = [t for t in spec.link_tags[l] if t in ctx.missing]
            if miss:
                reasons.append("%s: %s missing (%s)" % (l, ", ".join(miss), "; ".join(
                    "%s: %s" % (t, ctx.missing[t]) for t in miss)[:240]))
        reasons += broken.get(leg, [])
        for l in links:
            if l not in ctx.link_group:
                reasons.append("%s: %s" % (l, ctx.link_reason.get(l, "not matched")))
        for l in links:
            if l not in ctx.link_group:
                continue
            ji = ctx.jinfo.get(ctx.link_group[l])
            jn = spec.links[l]["joint"]
            if ji is None:
                reasons.append("%s: no revolute joint" % jn)
            elif ji["core"] is None:
                reasons.append("%s: %s" % (jn, ji["reason"]))
            elif not ji["consistent"]:
                reasons.append("%s: not consistent with its core (%s)" % (jn, ji["reason"]))
        lower = links[2]
        if lower in ctx.link_group:
            g = ctx.link_group[lower]
            nf = sum(1 for n in ctx.feet if ctx.uf.find(n) == g)
            if nf != 1:
                reasons.append("%s carries %d foot pads (one expected)" % (lower, nf))
        legs[leg] = {"valid": not reasons, "reasons": reasons}
    ctx.legs = legs
    ctx.stats["legs_valid"] = float(sum(1 for v in legs.values() if v["valid"]))


# ---------------------------------------------------------------------------------------------------- alignment
def _gt_body_samples(ctx):
    spec, s = ctx.spec, ctx.s
    tags = spec.link_tags[spec.root]
    paths = sorted(set(spec.parts[t]["path"] for t in tags))
    cache, todo = {}, []
    for path in paths:
        mesh = _reference_stl(spec, path)
        if mesh is not None:
            cache[path] = mesh
        else:
            todo.append((path, path))
    import trimesh
    for path, rec in ctx.pool(_mesh_job, todo).items():
        if "error" in rec:
            raise RuntimeError("reference part file %s does not load (%s)" % (os.path.basename(path), rec["error"]))
        cache[path] = trimesh.Trimesh(rec["vertices"], rec["faces"], process=False)
    pts, nrm, cents = [], [], {}
    for k, tag in enumerate(tags):
        mesh = cache[spec.parts[tag]["path"]].copy()
        mesh.apply_transform(spec.part_T_robot(tag))
        p, n = _samples(mesh, s["align_samples_gt"], s["align_seed"] + 2 * k + 1)
        pts.append(p)
        nrm.append(n)
        cents[tag] = np.asarray(mesh.center_mass, float)
    return np.vstack(pts), np.vstack(nrm), cents


def _reference_stl(spec, path):
    """A reference part's tessellation in its file frame when the folder ships it (meshes/<name>.stl beside parts/, the
    generator's 0.05 mm tessellation, as the grader reads it), else None (the STEP is tessellated by a worker)."""
    stl = os.path.join(spec.dir, "meshes", os.path.splitext(os.path.basename(path))[0] + ".stl")
    if os.path.isfile(stl):
        import trimesh
        mesh = trimesh.load(stl, force="mesh", process=True)
        if len(mesh.faces) > 0:
            return mesh
    return None


def _agent_body_samples(ctx):
    spec, s = ctx.spec, ctx.s
    pts, cents = [], {}
    for k, tag in enumerate(spec.link_tags[spec.root]):
        n = ctx.part_node.get(tag)
        if n is None or ctx.uf.find(n) != ctx.g0:
            continue
        b = ctx.body(n)
        p, _nn = _samples(b.mesh, s["align_samples_agent"], s["align_seed"] + 2 * k + 2)
        pts.append(p)
        cents[tag] = np.asarray(b.centroid, float)
    return (np.vstack(pts) if pts else np.zeros((0, 3))), cents


def _align(ctx):
    """agent world -> robot frame (4x4) and the statistics."""
    from scipy.spatial import cKDTree
    spec, s = ctx.spec, ctx.s
    st = {"seed": "none"}
    gpts, gnrm, gcent = _gt_body_samples(ctx)
    tree = cKDTree(gpts)
    apts, acent = _agent_body_samples(ctx)

    def body_rms(T):
        if len(apts) == 0:
            return float("inf")
        return A.trimmed_rms(T, apts, tree)

    # ---- seed 1: the valid legs' HAA cores
    meta = ctx.core_meta
    haa = {}
    for leg in spec.legs:
        if not ctx.legs[leg]["valid"]:
            continue
        ji = ctx.jinfo.get(ctx.link_group.get(Q.leg_links(leg)[0]))
        if ji is None or ji["core"] is None:
            continue
        a = np.vstack([ji["core_c"], ji["core_c"] + s["seed_axis_mm"] * ji["core_d"]])
        Tg = spec.core_T_robot(spec.core_of_joint[Q.leg_joints(leg)[0]])
        cg = _apply(Tg, np.asarray(meta["output"]["point"], float)[None, :])[0]
        dg = _unit(Tg[:3, :3] @ np.asarray(meta["output"]["axis"], float))
        haa[leg] = (a, np.vstack([cg, cg + s["seed_axis_mm"] * dg]))
    seed = None
    cands = {}
    legs_h = sorted(haa)

    def axis_pts(l, sg):
        """The leg's output centre and its axis point on side sg (+1: along the unit's +z, -1: against it)."""
        a = haa[l][0]
        return np.vstack([a[0], a[0] + sg * (a[1] - a[0])])

    def fit(T, l):
        """(residual, side) of leg l under T with its axis point on the side that fits: a unit mounted the other way
        round (its +z reversed: the joint's sense flips, which its own items price) must not turn the whole robot
        (review of round 2, 2026-10-29: two reversed units seeded a half-turn of the body)."""
        return min((float(np.max(np.linalg.norm(_apply(T, axis_pts(l, q)) - haa[l][1], axis=1))), q) for q in (1.0, -1.0))
    for l1, l2 in itertools.combinations(legs_h, 2):
        for s1, s2 in itertools.product((1.0, -1.0), repeat=2):
            Pa = np.vstack([axis_pts(l1, s1), axis_pts(l2, s2)])
            Pg = np.vstack([haa[l1][1], haa[l2][1]])
            if _collinear(Pa) or _collinear(Pg):
                continue
            T = kabsch(Pa, Pg)
            side = {}
            for l in legs_h:
                r, q = fit(T, l)
                if r <= s["haa_inlier_mm"]:
                    side[l] = q
            inl = tuple(sorted(side))
            if len(inl) < 2:
                continue
            Pa = np.vstack([axis_pts(l, side[l]) for l in inl])
            Pg = np.vstack([haa[l][1] for l in inl])
            if _collinear(Pa) or _collinear(Pg):
                continue
            Ti = kabsch(Pa, Pg)
            r = float(np.sqrt(np.mean(np.sum((_apply(Ti, Pa) - Pg) ** 2, axis=1))))
            if inl not in cands or body_rms(Ti) < body_rms(cands[inl][0]):    # the same legs both ways: the body decides
                cands[inl] = (Ti, r)
    if cands:
        m = max(len(k) for k in cands)
        best = sorted([(body_rms(T), k, T, r) for k, (T, r) in cands.items() if len(k) == m], key=lambda x: (x[0], x[1]))
        brms, inl, seed, rms = best[0]
        st.update(seed="haa", seed_legs=list(inl), seed_outlier_legs=[l for l in legs_h if l not in inl],
                  seed_rms_mm=rms, seed_candidates=len(best))
    # ---- seed 2: the body parts' centroids (tag-matched)
    if seed is None:
        tags = sorted(t for t in acent if t in gcent)
        Pa = np.array([acent[t] for t in tags]) if tags else np.zeros((0, 3))
        Pg = np.array([gcent[t] for t in tags]) if tags else np.zeros((0, 3))
        if len(tags) >= 3 and not _collinear(Pa) and not _collinear(Pg):
            T = kabsch(Pa, Pg)
            res = np.linalg.norm(_apply(T, Pa) - Pg, axis=1)
            keep = res <= max(5.0, 3.0 * float(np.median(res)))
            if keep.sum() >= 3 and not keep.all() and not _collinear(Pa[keep]) and not _collinear(Pg[keep]):
                T = kabsch(Pa[keep], Pg[keep])
                res = np.linalg.norm(_apply(T, Pa) - Pg, axis=1)
            seed = T
            st.update(seed="body_centroids", seed_parts=len(tags), seed_rms_mm=float(np.sqrt(np.mean(res[keep] ** 2))),
                      seed_outlier_parts=[t for t, k in zip(tags, keep) if not k])
        else:
            seed = np.eye(4)
            ctx.warnings.append("alignment: fewer than two valid non-collinear HAA joints and fewer than three body parts; "
                                "the document frame is used as is")
    # ---- refine: restricted ICP of the agent's body parts onto the reference's (in the seeded frame)
    T = seed
    st["body_rms_seed_mm"] = body_rms(seed) if len(apts) else None
    if len(apts) >= 6:
        src = _apply(seed, apts)
        old = (A.MAX_T_MM, A.MAX_R_DEG)
        try:
            A.MAX_T_MM, A.MAX_R_DEG = float(s["icp_max_mm"]), float(s["icp_max_deg"])
            Ti, rms = A.icp(src, gpts, gnrm, tree, np.eye(4), restrict=True)
        finally:
            A.MAX_T_MM, A.MAX_R_DEG = old
        piv = src.mean(axis=0)
        T = Ti @ seed
        st.update(body_rms_mm=float(rms), icp_rotation_deg=_rot_deg(Ti[:3, :3]),
                  icp_translation_mm=float(np.linalg.norm(_apply(Ti, piv[None, :])[0] - piv)), n_src=int(len(apts)),
                  n_dst=int(len(gpts)))
    else:
        st.update(body_rms_mm=st["body_rms_seed_mm"], icp_rotation_deg=0.0, icp_translation_mm=0.0, n_src=int(len(apts)),
                  n_dst=int(len(gpts)))
        ctx.warnings.append("alignment: too few body samples for the ICP refinement (seed used as is)")
    # per-leg HAA residual after the alignment (informative)
    st["haa_residual_mm"] = {l: fit(T, l)[0] for l in legs_h}
    st["T"] = _mat_list(T)
    ctx.T = T
    ctx.align = st
    for k in ("seed_rms_mm", "body_rms_seed_mm", "body_rms_mm", "icp_rotation_deg", "icp_translation_mm"):
        if isinstance(st.get(k), float):
            ctx.stats["align_" + k] = st[k]
    ctx.stats["align_seed_haa"] = 1.0 if st["seed"] == "haa" else 0.0


# ---------------------------------------------------------------------------------------------------- the writer
def _write(ctx, out_dir):
    spec, T = ctx.spec, ctx.T
    R = T[:3, :3]
    os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
    # link origins (aligned frame)
    origin = {spec.root: np.zeros(3)}
    written = [spec.root]
    joints_out = []
    jrecs = {}
    for link in spec.link_order[1:]:
        g = ctx.link_group.get(link)
        par = spec.links[link]["parent"]
        if g is None or par not in origin:
            continue
        ji = ctx.jinfo[g]
        p_al = _apply(T, ji["point"][None, :])[0]
        axis = _unit(R @ ji["axis"])
        if ji["core"] is not None:
            c_al = _apply(T, ji["core_c"][None, :])[0]
            o = p_al + float(np.dot(c_al - p_al, axis)) * axis
        else:
            o = p_al
            ctx.warnings.append("%s: no driving core; its origin is the joint frame's" % link)
        if not (np.all(np.isfinite(o)) and np.all(np.isfinite(axis))):
            # never write a non-finite frame: the grader would drop the link and every part in it (shape is not
            # leg-gated); the link and those below it are left out, their parts written unattached
            ctx.warnings.append("%s: its joint frame is not finite; the link is not written" % link)
            continue
        origin[link] = o
        written.append(link)
        jn = spec.links[link]["joint"]
        joints_out.append({"name": jn, "parent_link": par, "child_link": link, "axis": [float(v) for v in axis],
                           "point_mm": [float(v) for v in o],
                           "driven_by": ("core." + jn) if ji["core"] is not None else None,
                           "limits_rad": [float(ji["lo"]), float(ji["hi"])]})
        jrecs[jn] = ji
    group_written = {ctx.link_group[l]: l for l in written}
    # parts: every valid part, aligned robot frame in the file, T_link into its written link (written by the workers)
    parts_out, writes = [], []
    for tag in spec.tags:
        n = ctx.part_node.get(tag)
        if n is None:
            continue
        g = ctx.uf.find(n)
        link = group_written.get(g)
        rel = "parts/%s.step" % tag
        writes.append((tag, (ctx.shape_path(n), T, os.path.join(out_dir, rel))))
        parts_out.append({"tag": tag, "link": link, "reference_link": spec.tag_link[tag], "file": rel, "frame": "robot",
                          "instance": ctx.leaf[n].get("path"),
                          "T_link": _mat_list(_translation(-origin[link]) if link else np.eye(4))})
    failed = {tag: rec["error"] for tag, rec in ctx.pool(_write_job, writes).items() if "error" in rec}
    if failed:                                # a part that does not survive the move is not in the tier
        ctx.warnings.append("parts not written: %s" % "; ".join("%s (%s)" % kv for kv in sorted(failed.items()))[:300])
        parts_out = [p for p in parts_out if p["tag"] not in failed]
        for tag, why in failed.items():
            ctx.missing[tag] = "not written: " + why[:120]
    # cores and feet as occurrences
    cores_out, extra = [], 0
    by_core = {ji["core"]: jn for jn, ji in jrecs.items() if ji["core"] is not None}
    for n in sorted(ctx.cores):
        Tw = T @ _m(ctx.cores[n]["pose"])
        link = group_written.get(ctx.uf.find(n))
        if n in by_core and link is not None:
            cid = "core." + by_core[n]
        else:
            extra += 1
            cid = "core.extra_%d" % extra
        cores_out.append({"id": cid, "link": link, "T_link": _mat_list((_translation(-origin[link]) if link else np.eye(4)) @ Tw),
                          "component": ctx.cores[n].get("label") or n})
    feet_out, foot_points, extra = [], {}, 0
    feet_in = {}
    for n in sorted(ctx.feet):
        feet_in.setdefault(group_written.get(ctx.uf.find(n)), []).append(n)
    for n in sorted(ctx.feet):
        Tw = T @ _m(ctx.feet[n]["pose"])
        link = group_written.get(ctx.uf.find(n))
        leg = Q.leg_of(link) if link else None
        if link and leg and link == Q.leg_links(leg)[2] and len(feet_in[link]) == 1:
            fid = "foot." + leg
            foot_points[leg] = [float(v) for v in Tw[:3, 3] - origin[link]]
        else:
            extra += 1
            fid = "foot.extra_%d" % extra
        feet_out.append({"id": fid, "link": link, "T_link": _mat_list((_translation(-origin[link]) if link else np.eye(4)) @ Tw),
                         "component": ctx.feet[n].get("label") or n})
    links_out = [{"name": l, "parent": spec.links[l]["parent"], "joint": spec.links[l]["joint"],
                  "origin_mm": [float(v) for v in origin[l]]} for l in written]
    ctx.written = written
    return {"format": Q.FORMAT, "units": {"length": "mm", "angle": "rad"}, "root_link": spec.root, "links": links_out,
            "parts": parts_out, "cores": cores_out, "feet": feet_out, "joints": joints_out, "foot_points": foot_points}


def _conversion_block(ctx):
    spec = ctx.spec
    joints = {}
    for link in spec.link_order[1:]:
        g = ctx.link_group.get(link)
        jn = spec.links[link]["joint"]
        if g is None:
            joints[jn] = {"found": False}
            continue
        ji = ctx.jinfo[g]
        r = {"found": True, "document_joint": ji["document_joint"], "child_first": bool(ji["child_first"]),
             "theta0_deg": _f(math.degrees(ji["theta0"])), "candidates": int(ji["candidates"]),
             "frames_gap_mm": _f(ji["jcs_gap_mm"]), "frames_angle_deg": _f(ji["jcs_z_deg"]),
             "limits_rad": [_f(ji["lo"]), _f(ji["hi"])], "limit_notes": ji["limit_notes"]}
        if ji["core"] is None:
            r.update(core=None, consistent=False, reason=ji["reason"])
        else:
            r.update(core=ctx.cores[ji["core"]].get("label") or ji["core"], core_node=ji["core"], core_id="core." + jn,
                     axis_angle_deg=_f(ji["axis_angle_deg"]), offset_mm=_f(ji["offset_mm"]), line_mm=_f(ji["line_mm"]),
                     child_mm=_f(ji["child_mm"]) if math.isfinite(ji["child_mm"]) else None, on_line=bool(ji["on_line"]),
                     gap_mm=_f(ji["gap_mm"]) if math.isfinite(ji["gap_mm"]) else None, flipped=bool(ji["sign"] < 0),
                     consistent=bool(ji["consistent"]), reason=ji["reason"])
        joints[jn] = r
    links = {}
    for link in spec.link_order:
        g = ctx.link_group.get(link)
        links[link] = {"matched": g is not None, "written": link in getattr(ctx, "written", []),
                       "reason": ctx.link_reason.get(link)}
    unmatched = sorted(t for t, n in ctx.part_node.items() if ctx.uf.find(n) not in ctx.group_link)
    assigned = set(ji["core"] for ji in ctx.jinfo.values() if ji.get("core"))
    al = dict(ctx.align) if hasattr(ctx, "align") else {}
    legs = ctx.legs if hasattr(ctx, "legs") else {}
    body = {"matched": spec.root in ctx.link_group, "missing_tags": list(getattr(ctx, "body_missing", [])),
            "foreign_tags": dict(getattr(ctx, "body_foreign", {}))}
    return {"version": VERSION, "ok": not ctx.errors,
            "legs_valid": [l for l in spec.legs if legs.get(l, {}).get("valid")], "n_legs_valid": int(ctx.stats.get("legs_valid", 0)),
            "reasons": {l: list(v["reasons"]) for l, v in legs.items()},
            "cad_json_consistent": {jn: bool(r.get("consistent")) for jn, r in joints.items()},
            "body": body, "legs": legs, "links": links, "joints": joints,
            "parts": {"valid": len(ctx.part_node), "missing": ctx.missing, "in_unmatched_groups": unmatched},
            "cores": {"found": len(ctx.cores), "assigned": len(assigned), "unusable": ctx.unusable},
            "feet": {"found": len(ctx.feet)},
            "tree": {"groups": len(ctx.groups), "reached": len(ctx.order), "floating": len(ctx.floating), "loops": ctx.loops},
            "alignment": al, "warnings": ctx.warnings[:60], "notes": ctx.notes[:60], "errors": ctx.errors[:30]}


def _joint_stats(ctx):
    for link in ctx.spec.link_order[1:]:
        g = ctx.link_group.get(link)
        jn = ctx.spec.links[link]["joint"]
        ji = ctx.jinfo.get(g) if g is not None else None
        ok = ji is not None and ji.get("core") is not None
        ctx.stats["core_axis_angle_deg_" + jn] = float(ji["axis_angle_deg"]) if ok else -1.0
        ctx.stats["core_axis_offset_mm_" + jn] = float(ji["offset_mm"]) if ok else -1.0
        ctx.stats["contact_gap_mm_" + jn] = float(min(ji["gap_mm"], 1e3)) if ok else -1.0
        ctx.stats["cad_json_consistent_" + jn] = 1.0 if ok and ji["consistent"] else 0.0
    for leg in ctx.spec.legs:
        ctx.stats["leg_valid_" + leg] = 1.0 if ctx.legs[leg]["valid"] else 0.0


def convert(extract_dir, reference_dir, given_dir, out_dir, config=None):
    """See the module docstring. `config` = quad_task.load_config(...) (None: the defaults)."""
    use_brep_siblings()
    spec = Q.load_spec(reference_dir)
    ctx = _Ctx(extract_dir, spec, given_dir, Q.conversion_settings(config))
    res = {"ok": False, "errors": ctx.errors, "warnings": ctx.warnings, "stats": ctx.stats, "details": ctx.details,
           "conversion": None, "legs_valid": 0}
    try:
        ctx.core_meta = Q.load_component(given_dir, "core")
        model = _read_model(ctx)
        if model is None:
            return res
        ctx.stats["recompute_moved"] = float(len(model.get("recompute_moved", [])))
        ctx.stats["sub_assemblies"] = float(sum(1 for c in model.get("containers", []) if c.get("flagged", True)))
        import time as _time

        def step(name, fn, *args):
            t = _time.time()
            out = fn(*args)
            ctx.stats["time_%s_s" % name] = _time.time() - t
            return out
        step("parts", _collect, ctx, model)
        if step("groups", _groups, ctx, model) is None:
            return res
        step("tree", _tree, ctx)
        step("match", _match, ctx)
        step("cores", _assign_cores, ctx)
        step("joints", _joints, ctx)
        _legs(ctx)
        _joint_stats(ctx)
        res["legs_valid"] = int(ctx.stats["legs_valid"])
        if spec.root not in ctx.link_group:
            res["conversion"] = _conversion_block(ctx)
            return res
        step("align", _align, ctx)
        if ctx.errors:                       # fatal (cad_valid = 0); the analysis is kept for the record
            res["conversion"] = _conversion_block(ctx)
            return res
        tier = step("write", _write, ctx, out_dir)
        tier["conversion"] = _conversion_block(ctx)
        res["conversion"] = tier["conversion"]
        with open(os.path.join(out_dir, "assembly.json"), "w") as fh:
            json.dump(tier, fh, indent=1, default=_json_default)
        res["ok"] = True
        res["details"]["legs"] = ["%s %s%s" % (leg, "valid" if v["valid"] else "invalid", "" if v["valid"] else ": " + "; ".join(v["reasons"])[:300])
                                  for leg, v in ctx.legs.items()]
        if ctx.body_missing or ctx.body_foreign:
            res["details"]["body"] = ["the grounded body " + "; it ".join(
                ([("lacks " + ", ".join(ctx.body_missing))] if ctx.body_missing else []) +
                ([("carries " + ", ".join("%s (%s)" % kv for kv in sorted(ctx.body_foreign.items())))] if ctx.body_foreign else []))]
        al = ctx.align
        res["details"]["alignment"] = ["seed %s (%s), body parts' residual %s mm, ICP correction %.2f deg / %.2f mm" % (
            al.get("seed"), ", ".join(al.get("seed_legs") or []) or "%s parts" % al.get("seed_parts", 0),
            "%.2f" % al["body_rms_mm"] if isinstance(al.get("body_rms_mm"), float) else "n/a",
            al.get("icp_rotation_deg", 0.0), al.get("icp_translation_mm", 0.0))]
        res["details"]["warnings"] = [w[:200] for w in ctx.warnings[:8]]
        return res
    except Exception:  # noqa: BLE001 - a document that breaks the conversion earns 0 with the trace, never a crash
        import traceback
        ctx.errors.append("conversion failed: " + traceback.format_exc()[-1200:])
        res["ok"] = False
        return res
