"""quad_grade: the grader core of quadruped_cad_design_from_photos (GRADER_SPEC sections 4-7): gates, criteria, floors and
statistics of the agent's mechanism, as the converter wrote it, against the hidden robot. Both sides are quad-cad-gt/1
folders (gt_quad/CONTRACTS.md section 8) read by one loader.

    prepared = prepare(ref_dir, given_dir, config)          optional: load the reference and the components once
    result = grade(sub_dir, ref_dir, given_dir, config, rng, log=None, prepared=None)
    scores, worst = interference_scores(model, comp, leg, qs, config)   the collision rule, for the hidden box builder
    python -I quad_grade.py SUB REF GIVEN [--config C] [--seeds hi:lo] [--out result.json]   (developer CLI)

Other public names: load_config(stage_dir), Model(folder, role, cfg) (a quad-cad-gt/1 folder; fk(q), driver(comp, joint)),
Components(given_dir), gt_legs(reference model), make_cases(rng, legs, reference, cfg), Collider.

Arguments
    sub_dir    the converter's output: assembly.json (links, parts, cores, feet, joints, foot_points and the converter's
               `conversion` block) and parts/*.step with .brep siblings (the BREP is read first: a STEP round trip can turn
               a valid solid invalid). The folder is data; nothing in it is executed.
    ref_dir    tests/reference/: assembly.json, parts/*.step, meshes/*.stl (read instead of the STEP when present),
               features.json
    given_dir  tests/given/: core.json, core.step, core_keepout.step, core_collision.step, foot.json, foot.step,
               public_blocklist.json
    config     grader_config.json as a dict; a missing key takes DEFAULT_CONFIG's value
    rng        numpy Generator of the per-run hidden seed (verify.py: default_rng(SeedSequence([hi, lo])))

Result (a dict; a missing or broken submission gives an honest 0, never an exception)
    rewards    cad_valid and public_model_reuse (0 / 1), the six criteria after leg gating, floors and the two gates, and
               design_score = sum(weight x criterion) / sum(weight)
    criteria   {criterion: {raw, floored, final, weight}}: raw = the item mean with the leg gating, floored = with the floors
    items      {criterion: [{id, weight, score, leg, valid, floor, value}]}: value = the measured error
    stats      flat numeric statistics, the seconds spent per stage included
    details    {check or criterion: [diagnostic strings]}

Rules (GRADER_SPEC 4-7; every number is a knob of grader_config.json)
  frames      both trees are read in the robot frame at q = 0 (the converter aligned the agent's body with the hidden
              robot's); link frames are world-aligned at q = 0, so a link's pose at q = 0 is the translation to its origin.
              FK: T(child) = T(parent) . Trans(o_child - o_parent) . Rot(axis, q) about the joint point. A joint's output
              centre and axis = its driving core's frame origin and +z (core.json `output`).
  leg gating  a leg is valid when its three links hang from the right parents, its three joints exist between them, each
              with a driving core carried by the joint's parent link, and its lower link carries one foot pad with a foot
              point; plus the converter's verdict (conversion.legs_valid) when the folder carries one, else this grader's
              own consistency test of the leg's joints (axis within 1 deg and 0.5 mm of the core's output axis, the child
              within 0.2 mm of the parent or the core). The items of an invalid leg score 0.
  cad_valid   the folder loads (assembly.json, the root link = the reference's root, conversion.ok not false) and the
              converted tree compiles as a MuJoCo model whose forward kinematics agree with this chain (qpos set by joint
              name; 1 mm / 0.5 deg at 16 configurations of the declared box). Missing parts, cores or feet only invalidate
              the items and legs they belong to.
  reuse gate  no part file's sha256 (STEP or BREP) and no part's 0.01 mm vertex fingerprint (file and link frame) is in the
              blocklist; pose-invariant matches are statistics.
  kinematic   axis:<joint> (output centre distance), length:* (body HAA spans: the error of haa_span_x = the mean |error|
              of the left and the right side's spans, of haa_span_y the front and the hind end's, so opposite errors do
              not cancel; per leg hip_along / hip_off / thigh / knee_lat / shank / foot_lat), fk:<leg>:median / p90 (the
              foot point over 64 poses of the leg in the hidden box), range:<joint> (|declared ^ hidden| / |hidden|).
  feature     cutout:<n>:<kind>:<tag> (the arm's cut-out probe on the aligned part), extent:<tag>:<axis>, openings:<tag> (the
              centre plates' through-openings seen along z, Hungarian matched, diameter range from the hidden plate).
  shape       per part after the arm's per-part ICP from the document placement: surface:<tag> (symmetric p90 surface
              distance) and iou:<tag> (1 - IoU); not leg-gated.
  mass        mass:<link> (printed outer-shell volume x density + the datasheet masses of the cores and feet carried) and
              com:<link> (centre of mass, robot frame, q = 0).
  fit         fit:<joint> per driving core = keep-out x seats x mount x key (core.json probes; scored on the agent alone).
  interference  inter:<leg>: within-leg and leg-body link pairs (leg-leg pairs are not checked) over 64 poses of the leg in
              the hidden box that lie inside the declared limits, min over the pairs, mean over the poses, then the arm's
              declared-range share. A core enters the link sets as the given collision solid (core_collision.step: the
              unit without the output pulley's swept region, which turns with the driven link).
  floors      F_s = mean over the reference's parts of band(p90 surface distance; floors.shape_p90_mm) (or, when that key
              is absent, of clamp((IoU - a) / (b - a)) with floors.shape_iou); F_k(L) = band(fk median of L) (0 for a
              leg without FK: its chain is broken). kinematic and feature items x F_s; mass, fit and interference: leg
              items x F_s x F_k(L), body items x F_s x mean F_k; shape unfloored.
  seeds       the FK, interference and MJCF poses are scrambled Halton points drawn from `rng` in a fixed order before any
              submission data is read (the sample does not depend on the document); ICP and surface samples from
              sample_seed = rng.integers(2^31 - 1).
  run time    a heavy document's part files (more than parallel.min_total_mb) are loaded in spawned worker processes
              (the B-rep check of a faceted solid dominates); parts not loaded within parallel.load_timeout_s count as
              missing. Point-in-solid probes use exact booleans on small cubes (see _contains_manifold). Seconds per
              stage are statistics (time_*_s).
"""
import json
import math
import os
import tempfile
import time
import traceback
from xml.sax.saxutils import quoteattr

import numpy as np

if __name__ == "__main__":       # developer CLI under python -I: the vendored helpers sit beside this file
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cad_drawings_check as C
import cad_photo_align as A
import cad_photo_features as FT

GRADER_VERSION = "quad-grade/1.0.0"
FORMAT = "quad-cad-gt/1"
CHECKS = ["cad_valid", "public_model_reuse"]
CRITERIA = ["kinematic_fidelity", "feature_fidelity", "shape_fidelity", "mass_properties", "component_fit", "interference_free"]
LEG_LENGTHS = ["hip_along", "hip_off", "thigh", "knee_lat", "shank", "foot_lat"]
BODY_LENGTHS = ["haa_span_x", "haa_span_y"]
RANGE_TOL_RAD = 1e-6           # range coverage: a declared range short of the hidden one by less than this is full

DEFAULT_CONFIG = {
    "weights": {"kinematic_fidelity": 0.30, "feature_fidelity": 0.25, "shape_fidelity": 0.25, "mass_properties": 0.10,
                "component_fit": 0.05, "interference_free": 0.05},
    "gates": ["cad_valid", "public_model_reuse"],
    "subweights": {"kinematic": {"axes": 0.30, "lengths": 0.20, "fk": 0.40, "range": 0.10},
                   "fk": {"median": 0.40, "p90": 0.35},
                   "feature": {"cutouts": 0.40, "extents": 0.40, "openings": 0.20},
                   "shape": {"surface": 0.60, "iou": 0.40},
                   "mass": {"mass": 0.50, "com": 0.50}},
    "bands": {"axis_mm": [0.3, 3.0], "length_mm": [0.3, 3.0], "fk_median_mm": [0.5, 8.0], "fk_p90_mm": [1.0, 15.0],
              "cutout_offset_mm": [0.3, 3.0], "cutout_area_rel": [0.10, 0.50], "extent_mm": [0.3, 3.0],
              "opening_mm": [0.3, 3.0], "surface_mm": [0.3, 2.5], "one_minus_iou": [0.05, 0.40], "mass_rel": [0.02, 0.15],
              "com_mm": [0.5, 6.0]},
    "floors": {"shape_iou": [0.20, 0.50], "leg_fk_median_mm": [15.0, 60.0]},
    "material": {"density_kg_m3": 1050.0},
    "consistency": {"axis_angle_deg": 1.0, "axis_offset_mm": 0.5, "contact_mm": 0.2, "gap_search_mm": 5.0},
    "mjcf": {"configs": 16, "fk_mm": 1.0, "fk_deg": 0.5, "include_meshes": False},
    "fk": {"configs_per_leg": 64},
    "component_fit": {"keepout_ratio": 0.001, "ring_score": [0.25, 0.75], "seats_combine": "mean", "mount_combine": "mean",
                      "key_full_fraction": 0.8, "key_sweep_steps": 90, "key_depth_fraction": 1.0, "key_grid_mm": 1.0,
                      "key_inset_mm": 0.5, "mounted_min": 0.5},
    "interference": {"adjacent_ratio": 0.01, "nonadjacent_ratio": 0.001, "configs_per_leg": 64,
                     "declared_configs_per_leg": 16, "stream_multiplier": 24},
    "features": {"raster_mm": 0.1, "search_mm": 6.0, "present_fraction": 0.6},
    "openings": {"designs": ["body_center"], "d_search_mm": [1.5, 40.0], "d_margin_mm": 0.5},
    "alignment": {"n_src": 20000, "n_dst": 200000},
    "public_model_reuse": {"invariant_rel": 0.002},
    "reference": {"use_meshes": True},
    "parallel": {"workers": 8, "min_total_mb": 20.0, "load_timeout_s": 600.0},
}


# ============================================================================= small helpers
def _merge(base, over):
    out = json.loads(json.dumps(base))
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(stage_dir):
    """grader_config.json of the stage merged over DEFAULT_CONFIG."""
    with open(os.path.join(stage_dir, "grader_config.json")) as fh:
        return _merge(DEFAULT_CONFIG, json.load(fh))


def _band(v, b):
    return C.band(v, float(b[0]), float(b[1]))


def _trans(v):
    T = np.eye(4)
    T[:3, 3] = np.asarray(v, float)[:3]
    return T


def _apply(T, p):
    return np.asarray(p, float) @ T[:3, :3].T + T[:3, 3]


def _vec(x, n=3):
    a = np.asarray(x, float).reshape(-1)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError("expected %d finite numbers" % n)
    return a


def _unit(x):
    a = _vec(x)
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        raise ValueError("zero vector")
    return a / n


def _rigid(x):
    """A finite rigid 4 x 4 (rotation re-orthonormalised); raises ValueError otherwise."""
    M = np.asarray(x, float)
    if M.shape != (4, 4) or not np.all(np.isfinite(M)):
        raise ValueError("not a finite 4 x 4 matrix")
    if np.abs(M[3] - np.array([0.0, 0.0, 0.0, 1.0])).max() > 1e-9:
        raise ValueError("last row is not 0 0 0 1")
    R = M[:3, :3]
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-4 or np.linalg.det(R) < 0:
        raise ValueError("not a rigid transform")
    U, _s, Vt = np.linalg.svd(R)
    out = M.copy()
    out[:3, :3] = U @ Vt
    return out


def _finite(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _num(v, default=0.0):
    f = _finite(v)
    return default if f is None else f


def _angle_deg(a, b):
    return C.angle_between(a, b)


def _halton(n, d, rng):
    """n scrambled Halton points in [0, 1)^d drawn from the Generator rng (scipy.stats.qmc)."""
    from scipy.stats import qmc
    if n <= 0:
        return np.zeros((0, d))
    try:
        eng = qmc.Halton(d, scramble=True, rng=rng)
    except TypeError:                     # scipy < 1.15
        eng = qmc.Halton(d, scramble=True, seed=rng)
    return np.asarray(eng.random(int(n)), float)


def _ring_points(centre, axis, r, n):
    c = _vec(centre)
    a = _unit(axis)
    e = np.eye(3)[int(np.argmin(np.abs(a)))]
    u = np.cross(a, e)
    u /= np.linalg.norm(u)
    v = np.cross(a, u)
    th = 2.0 * math.pi * np.arange(int(n)) / float(n)
    return c[None, :] + float(r) * (np.cos(th)[:, None] * u[None, :] + np.sin(th)[:, None] * v[None, :])


def _in_polygon(pts, poly):
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), bool)
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cond & (x < xint)
    return inside


def _dist_to_polygon(pts, poly):
    d = np.full(len(pts), np.inf)
    n = len(poly)
    for i in range(n):
        a = np.asarray(poly[i], float)
        b = np.asarray(poly[(i + 1) % n], float)
        ab = b - a
        L2 = float(ab @ ab)
        t = np.clip(((pts - a) @ ab) / L2, 0.0, 1.0) if L2 > 0 else np.zeros(len(pts))
        d = np.minimum(d, np.linalg.norm(pts - (a[None, :] + t[:, None] * ab[None, :]), axis=1))
    return d


def _key_grid(outline, grid, inset):
    """Grid points (u, v) inside the socket outline, at least `inset` from its edges (symmetric about the outline's box)."""
    P = np.asarray(outline, float).reshape(-1, 2)
    lo, hi = P.min(axis=0), P.max(axis=0)
    axes = []
    for k in range(2):
        span = hi[k] - lo[k]
        n = int(math.floor(span / grid + 1e-9)) + 1
        start = lo[k] + 0.5 * (span - (n - 1) * grid)
        axes.append(start + grid * np.arange(n))
    U, V = np.meshgrid(axes[0], axes[1], indexing="ij")
    pts = np.c_[U.ravel(), V.ravel()]
    keep = _in_polygon(pts, P) & (_dist_to_polygon(pts, P) >= inset - 1e-9)
    return pts[keep]


def _combine(values, how):
    if not values:
        return 1.0
    if how == "min":
        return float(min(values))
    if how == "product":
        return float(np.prod(values))
    return float(np.mean(values))


def _aabb_overlap(a, b):
    return all(a[i] <= b[i + 3] and b[i] <= a[i + 3] for i in range(3))


def _outer_shell(mesh):
    """The arm's rule: enclosed voids (inner shells) dropped; the largest component by |volume|."""
    try:
        comps = mesh.split(only_watertight=False)
    except Exception:  # noqa: BLE001
        return mesh
    if len(comps) <= 1:
        return mesh
    return max(comps, key=lambda c: abs(float(c.volume)))


def _sanitize(x):
    if isinstance(x, dict):
        return {str(k): _sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_sanitize(v) for v in x]
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return _sanitize(x.tolist())
    return x


# ============================================================================= geometry
class Geom(object):
    """One watertight solid tessellation in a link frame (mesh, manifold, outer shell, mass data, bounds)."""

    def __init__(self, mesh_link, mesh_file=None):
        self.mesh = mesh_link
        self.mesh_file = mesh_file if mesh_file is not None else mesh_link
        self.manifold = C.trimesh_to_manifold(mesh_link)
        if str(self.manifold.status()) != "Error.NoError":
            raise ValueError("manifold construction failed: %s" % self.manifold.status())
        self.outer = _outer_shell(mesh_link)
        self.volume = float(abs(self.outer.volume))           # outer shell: enclosed voids count as material
        self.centroid = np.asarray(self.outer.center_mass, float)
        self.mesh_volume = float(mesh_link.volume)
        self.bounds = np.asarray(mesh_link.bounds, float)
        self.centre = 0.5 * (self.bounds[0] + self.bounds[1])
        self.radius = 0.5 * float(np.linalg.norm(self.bounds[1] - self.bounds[0]))

    def contains(self, pts):
        """Point-in-solid (link frame): exact booleans (_contains_manifold), ray parity on the tessellation as the
        fallback; both agree for every point farther than PROBE_CUBE_MM from the surface."""
        pts = np.asarray(pts, float).reshape(-1, 3)
        out = np.zeros(len(pts), bool)
        if len(pts) == 0:
            return out
        lo, hi = self.bounds[0] - 1e-6, self.bounds[1] + 1e-6
        m = np.all((pts >= lo) & (pts <= hi), axis=1)
        if m.any():
            r = None
            try:
                r = _contains_manifold(self.manifold, pts[m])
            except Exception:  # noqa: BLE001
                r = None
            out[m] = r if r is not None else np.asarray(self.mesh.contains(pts[m]), bool)
        return out


PROBE_CUBE_MM = 0.02


def _contains_manifold(man, pts, h=PROBE_CUBE_MM):
    """Point-in-solid by one exact boolean: a cube of side h around each point intersected with the solid; a point is
    inside when more than half of its cube is. About 100 times faster than trimesh's ray parity without embree (3.4 s
    per 1000 points on a 135k-triangle shell) and equal to it away from the surface. Points closer than 2 h to each other
    share one cube (the cubes must be disjoint)."""
    import manifold3d as m3
    from scipy.spatial import cKDTree
    pts = np.asarray(pts, float).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, bool)
    uniq, inv = np.unique(np.round(pts, 6), axis=0, return_inverse=True)
    rep = np.arange(len(uniq))                         # points closer than 2 h share one probe (their cluster's first)
    pairs = cKDTree(uniq).query_pairs(2.0 * h, output_type="ndarray") if len(uniq) > 1 else np.zeros((0, 2), int)
    if len(pairs):
        par = list(range(len(uniq)))

        def find(a):
            while par[a] != a:
                par[a] = par[par[a]]
                a = par[a]
            return a
        for a, b in pairs:
            ra, rb = find(int(a)), find(int(b))
            if ra != rb:
                par[max(ra, rb)] = min(ra, rb)
        rep = np.array([find(i) for i in range(len(uniq))])
    reps = np.unique(rep)
    centres = uniq[reps]
    tree = cKDTree(centres)
    cube = m3.Manifold.cube([h, h, h], True)
    probe = m3.Manifold.compose([cube.translate([float(x), float(y), float(z)]) for x, y, z in centres])
    inter = man ^ probe
    res = np.zeros(len(centres), bool)
    if not inter.is_empty():
        for c in inter.decompose():
            bb = np.asarray(c.bounding_box(), float)
            dist, i = tree.query(0.5 * (bb[:3] + bb[3:]))
            if dist < h and c.volume() > 0.5 * h ** 3:
                res[int(i)] = True
    idx = np.searchsorted(reps, rep)
    return res[idx][np.asarray(inv).reshape(-1)]


def _brep_sibling(path):
    return os.path.splitext(path)[0] + ".brep"


def _load_solids(path, prefer_brep=True):
    """Solids of a part file: the .brep sibling first (exact transport; a solid that fails BRepCheck by a round-off of its
    tolerances gets them recomputed, quad_convert.refresh_tolerances), else the STEP (the vendored reader)."""
    import cadquery as cq
    from quad_convert import refresh_tolerances
    if path.lower().endswith(".brep"):
        brep = path
    else:
        brep = _brep_sibling(path) if prefer_brep else None
    if brep and os.path.isfile(brep) and not os.path.islink(brep):
        try:
            return refresh_tolerances(list(cq.Shape.importBrep(brep).Solids())), None
        except Exception as exc:  # noqa: BLE001
            return [], "BREP reader failed: %s" % str(exc)[:200]
    return C.load_step_solids(path)


def _mesh_from_solid_file(path, prefer_brep=True):
    from quad_convert import install_tess_retry
    install_tess_retry()          # a valid solid whose default mesh is not watertight is re-tessellated finer
    solids, err = _load_solids(path, prefer_brep)
    if err:
        raise ValueError(err)
    if len(solids) != 1:
        raise ValueError("the file holds %d solids (one expected)" % len(solids))
    body = C.Body(os.path.basename(path), solids[0])
    if not body.ok:
        raise ValueError("; ".join(body.errors))
    return body.mesh, list(body.notes)


def _load_job(job):
    """Worker of Model._preload (a spawned process): (key, path, prefer_brep) -> (key, vertices, faces, notes, error)."""
    key, path, prefer_brep = job
    try:
        mesh, notes = _mesh_from_solid_file(path, prefer_brep)
        return key, np.asarray(mesh.vertices, float), np.asarray(mesh.faces, np.int64), notes, None
    except Exception as exc:  # noqa: BLE001
        return key, None, None, [], str(exc)[:200]


def _mesh_from_solids_file(path):
    """One watertight mesh of every solid of a STEP file (the given collision solid: disjoint solids allowed)."""
    import trimesh
    solids, err = C.load_step_solids(path)
    if err or not solids:
        raise ValueError("%s: %s" % (os.path.basename(path), err or "no solids"))
    meshes = []
    for k, s in enumerate(solids):
        body = C.Body("%s#%d" % (os.path.basename(path), k), s)
        if not body.ok:
            raise ValueError("%s solid %d: %s" % (os.path.basename(path), k, "; ".join(body.errors)))
        meshes.append(body.mesh)
    return meshes[0] if len(meshes) == 1 else trimesh.util.concatenate(meshes)


def _mesh_from_stl(path):
    import trimesh
    m = trimesh.load(path, force="mesh", process=True)
    m.update_faces(m.nondegenerate_faces())
    m.remove_unreferenced_vertices()
    notes = []
    if not m.is_watertight:
        m, closed, left = C.fill_small_holes(m)
        if closed:
            notes.append("closed %d micro-gap loop(s)" % closed)
        if left:
            raise ValueError("%d boundary loop(s) too large to close" % left)
    if not (m.is_watertight and m.is_winding_consistent and m.volume > 0):
        raise ValueError("the mesh is not a closed, consistently wound solid")
    return m, notes


# ============================================================================= components (given files)
class Components(object):
    """The bought components of tests/given/: the actuator core (geometry, keep-out, mass, probes) and the foot pad."""

    def __init__(self, given_dir):
        self.dir = given_dir
        with open(os.path.join(given_dir, "core.json")) as fh:
            core = json.load(fh)
        with open(os.path.join(given_dir, "foot.json")) as fh:
            foot = json.load(fh)
        self.core_json, self.foot_json = core, foot

        def fname(doc, key, default):
            """A file name of the record: top level (CONTRACTS 9) or under "files" (the datasheet form)."""
            v = doc.get(key) or (doc.get("files") or {}).get(key)
            if v:
                return v
            return default if (default and os.path.isfile(os.path.join(given_dir, default))) else None
        mesh, _n = _mesh_from_solid_file(os.path.join(given_dir, fname(core, "step", "core.step") or "core.step"), prefer_brep=False)
        self.core = Geom(mesh)
        ko = fname(core, "keepout_step", "core_keepout.step")
        if ko:
            mesh, _n = _mesh_from_solid_file(os.path.join(given_dir, ko), prefer_brep=False)
            self.keepout = Geom(mesh)
        else:
            self.keepout = self.core
        # the collision solid of the interference sets: the unit without the output pulley's swept region (the pulley
        # and the bearings' inner rings turn with the driven link); several solids allowed (the bearings' outer rings
        # are held only by the housing). Without one, the interference sets use core.step (fit, keep-out and the
        # consistency gap always do).
        col = fname(core, "collision_step", None)
        self.core_collision = Geom(_mesh_from_solids_file(os.path.join(given_dir, col))) if col else self.core
        self.core_volume = float(core.get("volume_mm3") or self.core.mesh_volume)
        self.core_mass = float(core["mass_kg"])
        self.core_com = _vec(core.get("com_mm", self.core.centroid.tolist()))
        I = core.get("inertia_kg_mm2")
        if I is not None:
            self.core_inertia = np.asarray(I, float).reshape(3, 3)
        else:
            e = self.core.bounds[1] - self.core.bounds[0]
            self.core_inertia = self.core_mass / 12.0 * np.diag([e[1] ** 2 + e[2] ** 2, e[0] ** 2 + e[2] ** 2, e[0] ** 2 + e[1] ** 2])
        out = core.get("output", {}) or {}
        self.out_point = _vec(out.get("point", [0.0, 0.0, 0.0]))
        self.out_axis = _unit(out.get("axis", [0.0, 0.0, 1.0]))
        self.probes = core.get("probes", {}) or {}
        mesh, _n = _mesh_from_solid_file(os.path.join(given_dir, fname(foot, "step", "foot.step") or "foot.step"), prefer_brep=False)
        self.foot = Geom(mesh)
        self.foot_mass = float(foot["mass_kg"])
        self.foot_com = _vec(foot.get("com_mm", self.foot.centroid.tolist()))
        if foot.get("inertia_kg_mm2") is not None:
            self.foot_inertia = np.asarray(foot["inertia_kg_mm2"], float).reshape(3, 3)
        else:
            ext = foot.get("extents_mm") or (self.foot.bounds[1] - self.foot.bounds[0]).tolist()
            e = np.asarray(ext, float)
            self.foot_inertia = self.foot_mass / 12.0 * np.diag([e[1] ** 2 + e[2] ** 2, e[0] ** 2 + e[2] ** 2, e[0] ** 2 + e[1] ** 2])
        self.blocklist = {"sha256": set(), "vertex_fingerprints": set(), "invariants": [], "present": False}
        p = os.path.join(given_dir, "public_blocklist.json")
        if os.path.isfile(p):
            with open(p) as fh:
                doc = json.load(fh)
            self.blocklist = {"sha256": set(doc.get("sha256", [])), "vertex_fingerprints": set(doc.get("vertex_fingerprints", [])),
                              "invariants": list(doc.get("invariants", [])), "present": True}


# ============================================================================= a quad-cad-gt/1 folder
class Model(object):
    """A quad-cad-gt/1 folder. `errors` (non-empty: the folder is unusable) and `notes` (entries skipped or missing)."""

    def __init__(self, folder, role, cfg, cache=None, strict=False):
        self.folder = os.path.abspath(folder)
        self.role = role
        self.strict = strict
        self.errors, self.notes = [], []
        self.links, self.joints, self.parts, self.cores, self.feet, self.foot_points = {}, {}, {}, {}, {}, {}
        self.joint_order, self.order, self.root, self.reached = [], [], None, set()
        self.conversion, self.spec = None, None
        self.cache = cache if cache is not None else {}
        self.use_meshes = bool(cfg["reference"].get("use_meshes", True)) and role == "reference"
        self.parallel = dict(cfg.get("parallel") or {})
        self._read()

    @property
    def ok(self):
        return not self.errors

    def _bad(self, msg):
        if self.strict:
            raise RuntimeError("%s folder %s: %s" % (self.role, self.folder, msg))
        self.notes.append(msg)

    def _list(self, spec, key):
        v = spec.get(key)
        if v is None:
            return []
        if not isinstance(v, list):
            self._bad("%s is not a list (ignored)" % key)
            return []
        return v

    # ---- reading
    def _read(self):
        path = os.path.join(self.folder, "assembly.json")
        if not os.path.isdir(self.folder):
            self.errors.append("no converted mechanism (the folder %s does not exist)" % self.folder)
            return
        if not os.path.isfile(path) or os.path.islink(path):
            self.errors.append("no assembly.json in the converted mechanism")
            return
        try:
            with open(path) as fh:
                spec = json.load(fh)
        except (OSError, ValueError) as exc:
            self.errors.append("assembly.json unreadable: %s" % str(exc)[:200])
            return
        if not isinstance(spec, dict):
            self.errors.append("assembly.json is not an object")
            return
        self.spec = spec
        if spec.get("format") not in (None, FORMAT):
            self.notes.append("format %r (expected %s)" % (spec.get("format"), FORMAT))
        conv = spec.get("conversion")
        self.conversion = conv if isinstance(conv, dict) else None
        for k, l in enumerate(self._list(spec, "links")):
            try:
                name = l["name"]
                if not isinstance(name, str) or not name:
                    raise ValueError("no name")
                parent = l.get("parent")
                if parent is not None and not isinstance(parent, str):
                    raise ValueError("parent is not a name")
                if name in self.links:
                    raise ValueError("link %s listed twice" % name)
                self.links[name] = {"parent": parent, "origin": _vec(l.get("origin_mm", [0.0, 0.0, 0.0])), "joint": None}
            except Exception as exc:  # noqa: BLE001
                self._bad("link entry %d skipped (%s)" % (k, str(exc)[:80]))
        root = spec.get("root_link")
        roots = [n for n, l in self.links.items() if l["parent"] is None]
        if isinstance(root, str) and root in self.links and self.links[root]["parent"] is None:
            self.root = root
        elif len(roots) == 1:
            self.root = roots[0]
        if self.root is None:
            self.errors.append("no root link (links without a parent: %s)" % roots[:4])
            return
        for k, j in enumerate(self._list(spec, "joints")):
            try:
                name = j["name"]
                if not isinstance(name, str) or not name or name in self.joints:
                    raise ValueError("bad or repeated name")
                lim = j.get("limits_rad")
                lo, hi = float(lim[0]), float(lim[1])
                if not (math.isfinite(lo) and math.isfinite(hi)):
                    raise ValueError("non-finite limits")
                child = j["child_link"]
                point = _vec(j["point_mm"]) if j.get("point_mm") is not None else None
                if point is None and isinstance(child, str) and child in self.links:
                    point = self.links[child]["origin"].copy()
                self.joints[name] = {"parent": j["parent_link"], "child": child, "axis": _unit(j["axis"]), "point": point,
                                     "lo": lo, "hi": hi, "driven_by": j.get("driven_by")}
                self.joint_order.append(name)
            except Exception as exc:  # noqa: BLE001
                self._bad("joint entry %d skipped (%s)" % (k, str(exc)[:80]))
        for k, c in enumerate(self._list(spec, "cores")):
            try:
                cid = c["id"]
                if not isinstance(cid, str) or cid in self.cores:
                    raise ValueError("bad or repeated id")
                self.cores[cid] = {"link": c["link"], "T": _rigid(c.get("T_link", np.eye(4)))}
            except Exception as exc:  # noqa: BLE001
                self._bad("core entry %d skipped (%s)" % (k, str(exc)[:80]))
        for k, f in enumerate(self._list(spec, "feet")):
            try:
                fid = f["id"]
                if not isinstance(fid, str) or fid in self.feet:
                    raise ValueError("bad or repeated id")
                self.feet[fid] = {"link": f["link"], "T": _rigid(f.get("T_link", np.eye(4)))}
            except Exception as exc:  # noqa: BLE001
                self._bad("foot entry %d skipped (%s)" % (k, str(exc)[:80]))
        fp = spec.get("foot_points") or {}
        if isinstance(fp, dict):
            for leg, v in fp.items():
                try:
                    self.foot_points[str(leg)] = _vec(v)
                except Exception:  # noqa: BLE001
                    self._bad("foot point of %s skipped" % leg)
        self._tree()
        self._preload(self._list(spec, "parts"))
        for k, p in enumerate(self._list(spec, "parts")):
            tag = p.get("tag") if isinstance(p, dict) else None
            if not isinstance(tag, str) or not tag:
                self._bad("part entry %d has no tag" % k)
                continue
            if tag in self.parts:
                self._bad("part tag %s listed twice (the first entry is used)" % tag)
                continue
            rec = {"link": p.get("link"), "design": p.get("design"), "instance": p.get("instance"), "file": p.get("file"),
                   "T": None, "geom": None, "error": None, "path": None, "unattached": False}
            self.parts[tag] = rec
            try:
                rec["T"] = _rigid(p.get("T_link", np.eye(4)))
                if rec["link"] is None and self.role != "reference":
                    # a valid part of a group that matches no link of the tree (the converter writes it with link null,
                    # T_link into the aligned robot frame): graded for its shape from that placement (GRADER_SPEC 5:
                    # shapes do not depend on the mechanism); no link carries it, so mass, fit, interference and the
                    # MJCF never see it (integration lane fix 2026-10-27)
                    rec["unattached"] = True
                    self.notes.append("part %s belongs to no link of the tree (its group matches none): graded for its "
                                      "shape from its robot-frame placement" % tag)
                elif not isinstance(rec["link"], str) or rec["link"] not in self.links:
                    raise ValueError("its link %r is not a link of the tree" % (rec["link"],))
                rec["geom"], rec["path"] = self._load_part(rec["file"], rec["T"])
            except Exception as exc:  # noqa: BLE001
                rec["error"] = str(exc)[:200]
                self._bad("part %s missing: %s" % (tag, rec["error"]))

    def _tree(self):
        by_child = {}
        for jn in self.joint_order:
            by_child.setdefault(self.joints[jn]["child"], []).append(jn)
        children = {}
        for name, l in self.links.items():
            if l["parent"] is not None:
                children.setdefault(l["parent"], []).append(name)
        order, seen, queue = [self.root], {self.root}, [self.root]
        while queue:
            cur = queue.pop(0)
            for ch in children.get(cur, []):
                js = by_child.get(ch, [])
                if len(js) != 1:
                    self._bad("link %s has %d joints leading to it (one expected)" % (ch, len(js)))
                    continue
                j = self.joints[js[0]]
                if j["parent"] != cur:
                    self._bad("link %s: its joint %s starts at %s, the link's parent is %s" % (ch, js[0], j["parent"], cur))
                    continue
                if ch in seen:
                    continue
                self.links[ch]["joint"] = js[0]
                if j["point"] is None:
                    j["point"] = self.links[ch]["origin"].copy()
                seen.add(ch)
                order.append(ch)
                queue.append(ch)
        self.order, self.reached = order, seen
        for name in self.links:
            if name not in seen:
                self._bad("link %s is not connected to the root %s" % (name, self.root))

    def _preload(self, entries):
        """Load the part files in spawned worker processes when the document is heavy: the B-rep check and the
        tessellation of a faceted solid (an imported mesh) cost about 12 s for 24k faces and 58 s for 135k faces on one
        core. Parts not loaded by the deadline count as missing (noted). Light documents load in-process."""
        par = self.parallel
        workers = int(par.get("workers", 1))
        try:
            workers = min(workers, len(os.sched_getaffinity(0)))
        except (AttributeError, OSError):
            pass
        jobs, total = [], 0
        for p in entries:
            if not isinstance(p, dict):
                continue
            try:
                path = self._safe_path(p.get("file"))
            except ValueError:
                continue
            key = (path, self.use_meshes)
            stem = os.path.splitext(os.path.basename(path))[0]
            if key in self.cache or (self.use_meshes and os.path.isfile(os.path.join(self.folder, "meshes", stem + ".stl"))):
                continue
            src = _brep_sibling(path) if (self.role != "reference" and os.path.isfile(_brep_sibling(path))) else path
            jobs.append((key, path, self.role != "reference"))
            total += os.path.getsize(src)
            self.cache[key] = None                                  # placeholder: one job per file
        for job in jobs:
            del self.cache[job[0]]
        if workers <= 1 or len(jobs) < 2 or total < float(par.get("min_total_mb", 20.0)) * 1e6:
            return
        import multiprocessing as mp
        import trimesh
        limit = float(par.get("load_timeout_s", 600.0))
        deadline = time.time() + limit
        done = set()
        pool = mp.get_context("spawn").Pool(min(workers, len(jobs)))
        try:
            it = pool.imap_unordered(_load_job, jobs)
            for _ in jobs:
                try:
                    key, V, F, notes, err = it.next(timeout=max(1.0, deadline - time.time()))
                except mp.TimeoutError:
                    break
                mesh = trimesh.Trimesh(V, F, process=False) if err is None else None
                self.cache[key] = (mesh, notes, err)
                done.add(key)
        finally:
            pool.terminate()
            pool.join()
        for key, _path, _pb in jobs:
            if key not in done:
                self.cache[key] = (None, [], "the part did not load within %.0f s" % limit)
        self.notes.append("%d part files (%.0f MB) loaded in %d worker processes" % (len(jobs), total / 1e6, min(workers, len(jobs))))

    def _safe_path(self, rel):
        if not isinstance(rel, str) or not rel:
            raise ValueError("no file")
        path = os.path.normpath(os.path.join(self.folder, rel))
        if not path.startswith(self.folder + os.sep):
            raise ValueError("file %s lies outside the folder" % rel)
        if not os.path.isfile(path) or os.path.islink(path):
            raise ValueError("file %s missing (or a symbolic link)" % rel)
        return path

    def _load_part(self, rel, T):
        path = self._safe_path(rel)
        key = (path, self.use_meshes)
        if key not in self.cache:
            try:
                stl = None
                if self.use_meshes:
                    stem = os.path.splitext(os.path.basename(path))[0]
                    cand = os.path.join(self.folder, "meshes", stem + ".stl")
                    if os.path.isfile(cand) and not os.path.islink(cand):
                        stl = cand
                if stl:
                    mesh, notes = _mesh_from_stl(stl)
                else:
                    mesh, notes = _mesh_from_solid_file(path, prefer_brep=(self.role != "reference"))
                self.cache[key] = (mesh, notes, None)
            except Exception as exc:  # noqa: BLE001
                self.cache[key] = (None, [], str(exc)[:200])
        mesh, notes, err = self.cache[key]
        if err:
            raise ValueError(err)
        m = mesh.copy()
        if np.abs(T - np.eye(4)).max() > 1e-12:
            m.apply_transform(T)
        return Geom(m, mesh), path

    # ---- queries
    def origin(self, link):
        return self.links[link]["origin"]

    def link_parts(self, link):
        return [(tag, p) for tag, p in sorted(self.parts.items()) if p["link"] == link and p["geom"] is not None]

    def link_cores(self, link):
        return [(cid, c) for cid, c in sorted(self.cores.items()) if c["link"] == link]

    def link_feet(self, link):
        return [(fid, f) for fid, f in sorted(self.feet.items()) if f["link"] == link]

    def fk(self, q=None):
        """World pose of every reached link at the joint angles `q` ({joint: rad}, others 0)."""
        q = q or {}
        T = {self.root: _trans(self.origin(self.root))}
        for name in self.order[1:]:
            L = self.links[name]
            j = self.joints[L["joint"]]
            piv = j["point"] - L["origin"]
            qi = float(q.get(L["joint"], 0.0))
            R = C.rot_axis(j["axis"], qi) if qi != 0.0 else np.eye(4)
            T[name] = T[L["parent"]] @ _trans(L["origin"] - self.origin(L["parent"])) @ _trans(piv) @ R @ _trans(-piv)
        return T

    def driver(self, comp, jn):
        """(output centre, output axis, core pose) of joint jn's driving core in the robot frame at q = 0, or None."""
        j = self.joints.get(jn)
        if j is None:
            return None
        c = self.cores.get(j.get("driven_by")) if isinstance(j.get("driven_by"), str) else None
        if c is None or c["link"] not in self.links:
            return None
        Tw = _trans(self.origin(c["link"])) @ c["T"]
        p = _apply(Tw, comp.out_point)
        z = Tw[:3, :3] @ comp.out_axis
        return p, z / np.linalg.norm(z), Tw


# ============================================================================= the reference's legs
def gt_legs(ref):
    """The reference's legs: [{name, joints [3], links [3]}] in joint order (each leg = a chain from a root child)."""
    legs = []
    for jn in ref.joint_order:
        j = ref.joints[jn]
        if j["parent"] != ref.root:
            continue
        joints, links, cur = [jn], [j["child"]], j["child"]
        while True:
            nxt = [k for k in ref.joint_order if ref.joints[k]["parent"] == cur]
            if len(nxt) != 1:
                if len(nxt) > 1:
                    raise RuntimeError("reference: the leg below %s branches" % cur)
                break
            joints.append(nxt[0])
            cur = ref.joints[nxt[0]]["child"]
            links.append(cur)
        name = jn.split("_")[0] if "_" in jn else links[0]
        legs.append({"name": name, "joints": joints, "links": links})
    if not legs:
        raise RuntimeError("reference: no leg below the root")
    return legs


# ============================================================================= prepared reference + components
def prepare(ref_dir, given_dir, config=None):
    cfg = _merge(DEFAULT_CONFIG, config or {})
    comp = Components(given_dir)
    ref = Model(ref_dir, "reference", cfg, strict=True)
    if not ref.ok:
        raise RuntimeError("reference folder %s does not load: %s" % (ref_dir, ref.errors[:3]))
    missing = [t for t, p in ref.parts.items() if p["geom"] is None]
    if missing:
        raise RuntimeError("reference parts do not load: %s" % missing[:5])
    legs = gt_legs(ref)
    for leg in legs:
        ok, why = _leg_structure(ref, ref, leg)
        if not ok:
            raise RuntimeError("reference leg %s: %s" % (leg["name"], "; ".join(why)))
        for jn in leg["joints"]:
            if not ref.joints[jn]["lo"] < ref.joints[jn]["hi"]:
                raise RuntimeError("reference joint %s: empty operating range" % jn)
    feats = []
    fp = os.path.join(ref_dir, "features.json")
    if os.path.isfile(fp):
        with open(fp) as fh:
            doc = json.load(fh)
        feats = doc.get("features", []) if isinstance(doc, dict) else list(doc)
    feats = [_feature_frame(f, k, ref) for k, f in enumerate(feats) if f.get("placed", True)]
    return {"ref": ref, "comp": comp, "legs": legs, "features": feats, "ref_dir": os.path.abspath(ref_dir),
            "given_dir": os.path.abspath(given_dir)}


def _feature_frame(f, k, ref):
    """A reference cut-out record checked for the vendored probe: its plane frame must be right-handed (e1 x e2 = normal:
    the probe slices the part in (e1, e2, normal) coordinates, and a mirrored frame turns the part inside out, which
    leaves both sections empty and scores every part as having the cut-out), so a left-handed record is turned into the
    same plane with e2 and the footprint's v negated; and the plane must cut the reference part."""
    if f.get("part") not in ref.parts:
        raise RuntimeError("reference feature %d on an unknown part %s" % (k + 1, f.get("part")))
    f = json.loads(json.dumps(f))
    c, n, e1, e2 = FT.plane_frame(f)
    if abs(float(n @ e1)) > 1e-3 or abs(float(n @ e2)) > 1e-3 or abs(float(e1 @ e2)) > 1e-3:
        raise RuntimeError("reference feature %d: normal, e1, e2 are not orthogonal" % (k + 1))
    if float(np.cross(e1, e2) @ n) < 0:
        f["e2"] = (-e2).tolist()
        f["footprint_mm"] = [[float(u), -float(v)] for u, v in f["footprint_mm"]]
        f["_frame_flipped"] = True
    if not FT._loops_in_plane(ref.parts[f["part"]]["geom"].mesh, FT.plane_frame(f)):
        raise RuntimeError("reference feature %d: its plane does not cut part %s" % (k + 1, f["part"]))
    return f


def make_cases(rng, legs, ref, cfg):
    """The per-run draws, in a fixed order that does not depend on the submission."""
    n_fk = int(cfg["fk"]["configs_per_leg"])
    ic = cfg["interference"]
    n_stream = int(ic["stream_multiplier"]) * int(ic["configs_per_leg"])
    fk = {leg["name"]: _halton(n_fk, len(leg["joints"]), rng) for leg in legs}
    inter = {leg["name"]: _halton(n_stream, len(leg["joints"]), rng) for leg in legs}
    mj = _halton(int(cfg["mjcf"]["configs"]), max(1, len(ref.joint_order)), rng)
    return {"fk": fk, "int": inter, "mjcf": mj, "sample_seed": int(rng.integers(2 ** 31 - 1))}


def _apply_module_knobs(cfg):
    """Sample sizes of the vendored modules that grader_config.json exposes (module attributes; files untouched)."""
    A.N_SRC = int(cfg["alignment"]["n_src"])
    A.N_DST = int(cfg["alignment"]["n_dst"])
    FT.PRESENT_FRAC = float(cfg["features"]["present_fraction"])


# ============================================================================= result skeleton and combination
def _empty_result(cfg):
    rewards = {k: 0.0 for k in CHECKS + CRITERIA}
    rewards["design_score"] = 0.0
    W = cfg["weights"]
    return {"rewards": rewards, "criteria": {c: {"raw": 0.0, "floored": 0.0, "final": 0.0, "weight": float(W[c])} for c in CRITERIA},
            "items": {c: [] for c in CRITERIA}, "stats": {"grader_version_" + GRADER_VERSION.split("/")[1].replace(".", "_"): 1.0},
            "details": {}}


def _item(iid, w, s, leg=None, value=None):
    s = _finite(s)                         # NaN would survive max(0, min(1, s)) as 1.0: a non-finite score is 0
    return {"id": iid, "weight": float(w), "score": 0.0 if s is None else float(max(0.0, min(1.0, s))), "leg": leg,
            "value": (None if value is None else _finite(value))}


def _criterion(items, valid, floor_of, gated=True):
    """(raw, floored) = sum(w s v [f]) / sum(w); math.fsum on both sides, so a perfect item list gives exactly 1."""
    sw = math.fsum(it["weight"] for it in items)
    raw, fl = [], []
    for it in items:
        v = 1.0 if (not gated or it["leg"] is None) else float(valid.get(it["leg"], 0.0))
        f = float(floor_of(it["leg"]))
        it["valid"], it["floor"] = v, f
        raw.append(it["weight"] * it["score"] * v)
        fl.append(it["weight"] * it["score"] * v * f)
    if sw <= 0:
        return 0.0, 0.0
    return min(1.0, math.fsum(raw) / sw), min(1.0, math.fsum(fl) / sw)


# ============================================================================= gates: structure, consistency, MJCF, reuse
def _leg_structure(sub, ref, leg):
    reasons = []
    for jn, ln in zip(leg["joints"], leg["links"]):
        gj = ref.joints[jn]
        aj = sub.joints.get(jn)
        if ln not in sub.reached:
            reasons.append("link %s is not in the tree" % ln)
        if aj is None:
            reasons.append("joint %s missing" % jn)
            continue
        if aj["parent"] != gj["parent"] or aj["child"] != gj["child"]:
            reasons.append("joint %s connects %s -> %s (the robot's: %s -> %s)" % (jn, aj["parent"], aj["child"], gj["parent"], gj["child"]))
        if ln in sub.links and sub.links[ln]["parent"] != ref.links[ln]["parent"]:
            reasons.append("link %s hangs from %s (the robot's: %s)" % (ln, sub.links[ln]["parent"], ref.links[ln]["parent"]))
        core = sub.cores.get(aj["driven_by"]) if isinstance(aj.get("driven_by"), str) else None
        if core is None:
            reasons.append("joint %s has no driving core" % jn)
        elif core["link"] != aj["parent"]:
            reasons.append("joint %s: its core sits in %s, not in the parent link %s" % (jn, core["link"], aj["parent"]))
    lower = leg["links"][-1]
    nf = len(sub.link_feet(lower))
    if nf != 1:
        reasons.append("the lower link %s carries %d foot pads (one expected)" % (lower, nf))
    if leg["name"] not in sub.foot_points:
        reasons.append("no foot point for %s" % leg["name"])
    return (not reasons), reasons


def _link_manifolds(model, link, T, comp, with_cores=False):
    out = []
    for _tag, p in model.link_parts(link):
        out.append((C.manifold_transform(p["geom"].manifold, T), None))
    if with_cores:
        for _cid, c in model.link_cores(link):
            out.append((C.manifold_transform(comp.core.manifold, T @ c["T"]), None))
    return [(m, m.bounding_box()) for m, _ in out]


def _min_gap(set_a, set_b, search):
    best = float(search)
    for ma, ba in set_a:
        for mb, bb in set_b:
            if not _aabb_overlap(tuple(np.asarray(ba[:3]) - search) + tuple(np.asarray(ba[3:]) + search), bb):
                continue
            best = min(best, float(ma.min_gap(mb, float(search))))
    return best


def _consistency(sub, comp, jn, cfg):
    """The converter's per-joint test (GRADER_SPEC 3.6) recomputed on the converted folder at q = 0."""
    cc = cfg["consistency"]
    j = sub.joints[jn]
    drv = sub.driver(comp, jn)
    out = {"ok": False, "angle_deg": float("nan"), "offset_mm": float("nan"), "gap_mm": float("nan"), "why": ""}
    if drv is None:
        out["why"] = "no driving core"
        return out
    p, z, Tw = drv
    core_link = sub.cores[j["driven_by"]]["link"]
    if j["point"] is None:
        out["why"] = "the joint has no point"
        return out
    ang = _angle_deg(j["axis"], z)
    ang = min(ang, 180.0 - ang)
    off = C.line_dist(p, z, j["point"])
    search = float(cc["gap_search_mm"])
    gap = float(search)
    if j["child"] in sub.links and j["parent"] in sub.links:
        child = _link_manifolds(sub, j["child"], _trans(sub.origin(j["child"])), comp)
        parent = _link_manifolds(sub, j["parent"], _trans(sub.origin(j["parent"])), comp)
        core = [(C.manifold_transform(comp.core.manifold, Tw), None)]
        core = [(m, m.bounding_box()) for m, _ in core]
        gap = min(_min_gap(child, parent, search), _min_gap(child, core, search))
    ok = (core_link == j["parent"] and ang <= cc["axis_angle_deg"] and off <= cc["axis_offset_mm"] and gap <= cc["contact_mm"])
    out.update(ok=bool(ok), angle_deg=float(ang), offset_mm=float(off), gap_mm=float(gap),
               why="" if ok else ("core not in the parent link" if core_link != j["parent"] else "axis %.2f deg, offset %.3f mm, gap %.3f mm" % (ang, off, gap)))
    return out


def _converter_flags(conv, legs):
    """{leg: bool} from conversion.legs_valid ({leg: bool | {valid}} or [valid leg names]); None when absent."""
    if not isinstance(conv, dict):
        return None
    lv = conv.get("legs_valid")
    if lv is None and isinstance(conv.get("legs"), dict):
        lv = conv["legs"]
    names = [l["name"] for l in legs]
    if isinstance(lv, dict):
        out = {}
        for n in names:
            v = lv.get(n, False)
            if isinstance(v, dict):
                v = v.get("valid", False)
            out[n] = bool(v)
        return out
    if isinstance(lv, (list, tuple)):
        return {n: (n in lv) for n in names}
    return None


def _mass_props(model, comp, density):
    """Per link: (mass kg, CoM in the robot frame at q = 0, mm), the published rule."""
    out = {}
    for link in model.links:
        o = model.origin(link)
        m, mc = 0.0, np.zeros(3)
        for _tag, p in model.link_parts(link):
            g = p["geom"]
            mi = g.volume * 1e-9 * density
            m += mi
            mc += mi * (o + g.centroid)
        for _cid, c in model.link_cores(link):
            m += comp.core_mass
            mc += comp.core_mass * (o + _apply(c["T"], comp.core_com))
        for _fid, f in model.link_feet(link):
            m += comp.foot_mass
            mc += comp.foot_mass * (o + _apply(f["T"], comp.foot_com))
        out[link] = (m, mc / m if m > 0 else None)
    return out


def _link_inertia(model, comp, link, density):
    """(mass kg, CoM m, inertia kg m^2 about the CoM) in the link frame, for the MJCF."""
    parts = []
    for _tag, p in model.link_parts(link):
        g = p["geom"]
        parts.append((g.volume * 1e-9 * density, g.centroid * 1e-3, np.asarray(g.outer.moment_inertia, float) * 1e-15 * density))
    for _cid, c in model.link_cores(link):
        R = c["T"][:3, :3]
        parts.append((comp.core_mass, _apply(c["T"], comp.core_com) * 1e-3, R @ comp.core_inertia @ R.T * 1e-6))
    for _fid, f in model.link_feet(link):
        R = f["T"][:3, :3]
        parts.append((comp.foot_mass, _apply(f["T"], comp.foot_com) * 1e-3, R @ comp.foot_inertia @ R.T * 1e-6))
    m = sum(x[0] for x in parts)
    if m <= 0:
        return 0.0, np.zeros(3), np.zeros((3, 3))
    c = sum(x[0] * x[1] for x in parts) / m
    I = np.zeros((3, 3))
    for mi, ci, Ii in parts:
        d = ci - c
        I += Ii + mi * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return m, c, 0.5 * (I + I.T)


def _mjcf_check(sub, comp, cases, cfg, ref):
    """The converted tree as a MuJoCo model: compiles without warnings, FK agrees with this chain (qpos by joint name)."""
    import shutil
    work = tempfile.mkdtemp(prefix="quad_mjcf_")
    try:
        return _mjcf_check_in(sub, comp, cases, cfg, ref, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _mjcf_check_in(sub, comp, cases, cfg, ref, work):
    import mujoco
    mc = cfg["mjcf"]
    density = float(cfg["material"]["density_kg_m3"])
    inertia = {l: _link_inertia(sub, comp, l, density) for l in sub.order}
    included, dropped = [sub.root], []
    for l in sub.order[1:]:
        if sub.links[l]["parent"] in included and inertia[l][0] > 0:
            included.append(l)
        else:
            dropped.append(l)
    children = {}
    for l in included[1:]:
        children.setdefault(sub.links[l]["parent"], []).append(l)
    lines = ['<mujoco model="quad_converted">', '  <compiler angle="radian" autolimits="true" balanceinertia="true" meshdir="meshes"/>',
             '  <option gravity="0 0 -9.81"/>']
    if mc.get("include_meshes"):
        os.makedirs(os.path.join(work, "meshes"), exist_ok=True)
        lines.append("  <asset>")
        for tag, p in sorted(sub.parts.items()):
            if p["geom"] is not None and p["link"] in included:
                p["geom"].mesh.export(os.path.join(work, "meshes", "%s.stl" % tag))
                lines.append('    <mesh name=%s file="%s.stl" scale="0.001 0.001 0.001"/>' % (quoteattr("m_" + tag), tag))
        lines.append("  </asset>")
    lines.append("  <worldbody>")

    def f9(v):
        return " ".join("%.9f" % float(x) for x in v)

    def emit(name, depth):
        ind = "  " * (depth + 2)
        L = sub.links[name]
        if name == sub.root:
            lines.append("%s<body name=%s pos=\"%s\">" % (ind, quoteattr(name), f9(L["origin"] * 1e-3)))
        else:
            j = sub.joints[L["joint"]]
            pos = (L["origin"] - sub.origin(L["parent"])) * 1e-3
            lines.append("%s<body name=%s pos=\"%s\">" % (ind, quoteattr(name), f9(pos)))
            rng = ' range="%.9f %.9f"' % (j["lo"], j["hi"]) if j["lo"] < j["hi"] else ' limited="false"'
            lines.append("%s  <joint name=%s type=\"hinge\" pos=\"%s\" axis=\"%s\"%s/>" % (
                ind, quoteattr(L["joint"]), f9((j["point"] - L["origin"]) * 1e-3), f9(j["axis"]), rng))
        m, c, I = inertia[name]
        if m > 0:
            lines.append("%s  <inertial pos=\"%s\" mass=\"%.9f\" fullinertia=\"%.12e %.12e %.12e %.12e %.12e %.12e\"/>" % (
                ind, f9(c), m, I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2]))
        if mc.get("include_meshes"):
            for tag, p in sub.link_parts(name):
                lines.append("%s  <geom type=\"mesh\" mesh=%s contype=\"0\" conaffinity=\"0\" mass=\"0\"/>" % (ind, quoteattr("m_" + tag)))
        for ch in children.get(name, []):
            emit(ch, depth + 1)
        lines.append("%s</body>" % ind)

    emit(sub.root, 0)
    lines += ["  </worldbody>", "</mujoco>"]
    xml = os.path.join(work, "quad.xml")
    with open(xml, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    warnings_seen = []
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    os.dup2(tmp.fileno(), 2)
    model = None
    err = None
    try:
        try:
            mujoco.set_mju_user_warning(lambda msg: warnings_seen.append(str(msg)))
        except Exception:  # noqa: BLE001
            pass
        try:
            model = mujoco.MjModel.from_xml_path(xml)
        except Exception as exc:  # noqa: BLE001
            err = str(exc)[:300]
        finally:
            try:
                mujoco.set_mju_user_warning(None)
            except Exception:  # noqa: BLE001
                pass
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        tmp.seek(0)
        captured = tmp.read().decode("utf-8", "replace")
        tmp.close()
    if "WARNING" in captured.upper():
        warnings_seen.append(captured.strip()[:300])
    info = {"ok": False, "dev_mm": float("nan"), "dev_deg": float("nan"), "bodies": len(included), "dropped": dropped}
    if model is None:
        info["why"] = "the MJCF does not compile: %s" % err
        return info
    if warnings_seen:
        info["why"] = "MuJoCo warnings: %s" % warnings_seen[:2]
        return info
    data = mujoco.MjData(model)
    jnames = [sub.links[l]["joint"] for l in included[1:]]
    col = {jn: k for k, jn in enumerate(ref.joint_order)}
    worst_p = worst_a = 0.0
    for u in np.asarray(cases["mjcf"]):
        q = {}
        data.qpos[:] = 0.0
        for k, jn in enumerate(jnames):
            j = sub.joints[jn]
            uk = float(u[col[jn] % len(u)]) if jn in col else float(u[k % len(u)])
            qi = j["lo"] + uk * (j["hi"] - j["lo"])
            q[jn] = qi
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
            data.qpos[model.jnt_qposadr[jid]] = qi
        mujoco.mj_kinematics(model, data)
        T = sub.fk(q)
        for name in included:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            worst_p = max(worst_p, float(np.linalg.norm(data.xpos[bid] * 1e3 - T[name][:3, 3])))
            Rm = data.xmat[bid].reshape(3, 3)
            ca = (np.trace(Rm.T @ T[name][:3, :3]) - 1.0) / 2.0
            worst_a = max(worst_a, math.degrees(math.acos(max(-1.0, min(1.0, ca)))))
    info.update(ok=bool(worst_p <= mc["fk_mm"] and worst_a <= mc["fk_deg"]), dev_mm=worst_p, dev_deg=worst_a,
                why="FK deviation %.4f mm / %.4f deg over %d configurations" % (worst_p, worst_a, len(cases["mjcf"])))
    return info


def _public_reuse(sub, comp, cfg):
    bl = comp.blocklist
    hits, det = [], []
    matches = similar = 0
    rel = float(cfg["public_model_reuse"]["invariant_rel"])
    for tag, p in sorted(sub.parts.items()):
        if p["geom"] is None:
            continue
        files = [p["path"]]
        if not p["path"].lower().endswith(".brep") and os.path.isfile(_brep_sibling(p["path"])):
            files.append(_brep_sibling(p["path"]))
        if any(C.sha256_file(f) in bl["sha256"] for f in files):
            hits.append(tag)
            det.append("%s: the part file's digest is a public model's" % tag)
            continue
        fps = {C.vertex_fingerprint(p["geom"].mesh_file), C.vertex_fingerprint(p["geom"].mesh)}
        if fps & bl["vertex_fingerprints"]:
            hits.append(tag)
            det.append("%s: the part's vertex fingerprint is a public model's" % tag)
            continue
        if bl["invariants"]:
            g = C.geometry_invariants(p["geom"].mesh)
            gm = any(C.invariants_match(g, r, rel) for r in bl["invariants"])
            gs = gm or any(C.invariants_match_similar(g, r, rel) for r in bl["invariants"])
            matches += 1 if gm else 0
            similar += 1 if gs else 0
            if gs:
                det.append("%s: pose-invariant geometry matches a public part%s (statistic)" % (tag, "" if gm else " up to scale"))
    return (0.0 if hits else 1.0), det, matches, similar


# ============================================================================= kinematic fidelity
def _leg_lengths(model, comp, leg, ref_axes=None):
    """The six per-leg lengths at q = 0 (robot frame), or None. ref_axes: the reference's joint axes (sign alignment)."""
    drv = [model.driver(comp, jn) for jn in leg["joints"]]
    if any(d is None for d in drv) or leg["name"] not in model.foot_points or leg["links"][-1] not in model.links:
        return None, None
    c = [d[0] for d in drv]
    a = [model.joints[jn]["axis"] for jn in leg["joints"]]
    if ref_axes is not None:
        a = [ai if float(ai @ gi) >= 0.0 else -ai for ai, gi in zip(a, ref_axes)]
    f = model.origin(leg["links"][-1]) + model.foot_points[leg["name"]]

    def split(d, ax):
        along = float(d @ ax)
        return along, float(np.linalg.norm(d - along * ax))
    hip_along, hip_off = split(c[1] - c[0], a[0])
    knee_lat, thigh = split(c[2] - c[1], a[1])
    foot_lat, shank = split(f - c[2], a[2])
    return {"hip_along": hip_along, "hip_off": hip_off, "thigh": thigh, "knee_lat": knee_lat, "shank": shank,
            "foot_lat": foot_lat}, a


def _foot_points(model, leg, qs):
    out = []
    lower = leg["links"][-1]
    fp = model.foot_points[leg["name"]]
    for q in qs:
        T = model.fk(dict(zip(leg["joints"], q)))
        out.append(_apply(T[lower], fp))
    return np.asarray(out)


def _kinematic(sub, ref, comp, legs, cases, cfg, struct_ok, S):
    B, W, Wf = cfg["bands"], cfg["subweights"]["kinematic"], cfg["subweights"]["fk"]
    leg_of_joint = {jn: leg["name"] for leg in legs for jn in leg["joints"]}
    items, det = [], []
    nj = len(ref.joint_order)
    for jn in ref.joint_order:
        g = ref.driver(comp, jn)
        a = sub.driver(comp, jn)
        e = float(np.linalg.norm(a[0] - g[0])) if a is not None else float("inf")
        if a is not None:
            ang = _angle_deg(a[1], g[1])
            S["axis_angle_deg_" + jn] = min(ang, 180.0 - ang)
            S["axis_sense_flipped_" + jn] = 1.0 if ang > 90.0 else 0.0
        S["axis_err_mm_" + jn] = e if math.isfinite(e) else -1.0
        items.append(_item("axis:" + jn, W["axes"] / nj, _band(e, B["axis_mm"]), leg_of_joint[jn], e))
    # lengths
    n_len = len(BODY_LENGTHS) + len(LEG_LENGTHS) * len(legs)
    gl, al = {}, {}
    for leg in legs:
        L = leg["name"]
        gl[L], g_axes = _leg_lengths(ref, comp, leg)
        al[L] = _leg_lengths(sub, comp, leg, g_axes)[0] if struct_ok[L] else None
    c_g = {leg["name"]: ref.driver(comp, leg["joints"][0])[0] for leg in legs}
    c_a = {}
    for leg in legs:
        d = sub.driver(comp, leg["joints"][0])
        if d is not None:
            c_a[leg["name"]] = d[0]
    names = [leg["name"] for leg in legs]
    # the body's HAA spans (between the HAA output centres, along the robot's x and y): two pairs each (x: the left and
    # the right side, y: the front and the hind end; the four spans of key_quantities.json). An item's error = the mean
    # |error| of its pairs that have both HAA output centres (none: inf), so the two sides' errors never cancel (a body
    # 2 mm longer on the left and 2 mm shorter on the right scored 1 when the signed errors were averaged; review fix
    # 2026-10-27). The signed error of each measured pair is a statistic (length_err_mm_<span>_<side>).
    spans = {}
    if len(names) == 4:
        front = set(sorted(names, key=lambda n: -c_g[n][0])[:2])
        left = set(sorted(names, key=lambda n: -c_g[n][1])[:2])
        quad = {(f, l): [n for n in names if (n in front) == f and (n in left) == l] for f in (True, False) for l in (True, False)}
        if all(len(v) == 1 for v in quad.values()):
            pick = {k: v[0] for k, v in quad.items()}
            for key, pairs, ax in (("haa_span_x", [("left", (True, True), (False, True)), ("right", (True, False), (False, False))], 0),
                                   ("haa_span_y", [("front", (True, True), (True, False)), ("hind", (False, True), (False, False))], 1)):
                errs = []
                for side, p1, p2 in pairs:
                    n1, n2 = pick[p1], pick[p2]
                    if n1 in c_a and n2 in c_a:
                        e = float((c_a[n1][ax] - c_a[n2][ax]) - (c_g[n1][ax] - c_g[n2][ax]))
                        errs.append(abs(e))
                        S["length_err_mm_%s_%s" % (key, side)] = e
                spans[key] = float(np.mean(errs)) if errs else float("inf")
    for key in BODY_LENGTHS:
        e = spans.get(key, float("inf"))
        S["length_err_mm_" + key] = e if math.isfinite(e) else -1.0
        items.append(_item("length:body:" + key, W["lengths"] / n_len, _band(e, B["length_mm"]), None, e))
    for leg in legs:
        L = leg["name"]
        for key in LEG_LENGTHS:
            e = (al[L][key] - gl[L][key]) if al[L] is not None else float("inf")
            S["length_err_mm_%s_%s" % (L, key)] = e if math.isfinite(e) else -1.0
            items.append(_item("length:%s:%s" % (L, key), W["lengths"] / n_len, _band(e, B["length_mm"]), L, e))
    # forward kinematics of the foot points
    sfk = float(Wf["median"] + Wf["p90"])
    fk_med = {}
    for leg in legs:
        L = leg["name"]
        lo = np.array([ref.joints[j]["lo"] for j in leg["joints"]])
        hi = np.array([ref.joints[j]["hi"] for j in leg["joints"]])
        qs = lo + np.asarray(cases["fk"][L]) * (hi - lo)
        med = p90 = float("inf")
        if struct_ok[L]:
            e = np.linalg.norm(_foot_points(sub, leg, qs) - _foot_points(ref, leg, qs), axis=1)
            med, p90 = float(np.median(e)), float(np.percentile(e, 90))
        fk_med[L] = med
        S["fk_median_mm_" + L] = med if math.isfinite(med) else -1.0
        S["fk_p90_mm_" + L] = p90 if math.isfinite(p90) else -1.0
        items.append(_item("fk:%s:median" % L, W["fk"] * Wf["median"] / sfk / len(legs), _band(med, B["fk_median_mm"]), L, med))
        items.append(_item("fk:%s:p90" % L, W["fk"] * Wf["p90"] / sfk / len(legs), _band(p90, B["fk_p90_mm"]), L, p90))
        det.append("%s foot point median %.2f mm, p90 %.2f mm" % (L, med, p90))
    # declared range coverage. Both ranges are compared on the converter's domain [-pi, pi] (it clips the declared
    # limits there; a full-turn hidden range is stored rounded, e.g. +-3.14159265359 > pi), and a shortfall below
    # RANGE_TOL_RAD is rounding (FreeCAD stores the limits in degrees), so the oracle's coverage is exactly 1
    # (integration lane fix 2026-10-27: ale validate requires the oracle's reward to be exactly 1.0).
    for jn in ref.joint_order:
        g = ref.joints[jn]
        a = sub.joints.get(jn)
        cov = 0.0
        g_lo, g_hi = max(g["lo"], -math.pi), min(g["hi"], math.pi)
        if a is not None:
            cov = max(0.0, min(a["hi"] + RANGE_TOL_RAD, g_hi) - max(a["lo"] - RANGE_TOL_RAD, g_lo)) / max(1e-9, g_hi - g_lo)
        S["range_cov_" + jn] = cov
        items.append(_item("range:" + jn, W["range"] / nj, cov, leg_of_joint[jn], 1.0 - cov))
    return items, fk_med, det


# ============================================================================= shape and feature fidelity (per-part alignment)
def _plate_holes(manifold, bounds, d_lo, d_hi):
    class _L(object):
        pass
    body = _L()
    body.manifold, body.bounds = manifold, bounds
    link = _L()
    link.bodies = [body]
    return C.base_bottom_holes(link, d_lo, d_hi)


def _parts(sub, ref, legs, feats, cfg, seed, S, log):
    from scipy.spatial import cKDTree
    B, Ws, Wf = cfg["bands"], cfg["subweights"]["shape"], cfg["subweights"]["feature"]
    leg_of_link = {l: leg["name"] for leg in legs for l in leg["links"]}
    tags = sorted(ref.parts)
    n = len(tags)
    ssw = float(Ws["surface"] + Ws["iou"])
    shape_items, ext_items, aligned, ious, sd_all, det = [], [], {}, {}, {}, []
    for k, tag in enumerate(tags):
        gp = ref.parts[tag]
        gl = gp["link"]
        L = leg_of_link.get(gl)
        ap = sub.parts.get(tag)
        iou = 0.0
        sd = float("inf")
        if ap is None or ap["geom"] is None:
            why = "not in the converted mechanism" if ap is None else ap["error"]
            det.append("%s missing (%s)" % (tag, why))
            S["part_missing_" + tag] = 1.0
        else:
            try:
                ga, gg = ap["geom"], gp["geom"]
                Ta = np.eye(4) if ap.get("unattached") else _trans(sub.origin(ap["link"]))   # unattached: robot frame
                Tg = _trans(ref.origin(gl))
                ma = ga.outer.copy()
                ma.apply_transform(Ta)
                mg = gg.mesh.copy()
                mg.apply_transform(Tg)
                al = A.align_part(ma, mg, seed + 101 * k)
                Tl = np.linalg.inv(Tg) @ al["T"] @ Ta                  # agent link frame -> GT link frame
                m_al = ga.outer.copy()
                m_al.apply_transform(Tl)
                aligned[tag] = m_al
                pa, _n = A.sample_surface(m_al, A.N_SRC, seed + 7 * k + 3)
                pg, _n = A.sample_surface(gg.mesh, A.N_SRC, seed + 7 * k + 4)
                dense_a, dense_an = A.sample_surface(m_al, A.N_DST, seed + 7 * k + 5)
                d1 = A.surface_distances(_apply(Tg, pa), al["gt_pts"], al["gt_normals"], al["gt_tree"])
                d2 = A.surface_distances(pg, dense_a, dense_an, cKDTree(dense_a))
                sd = float(np.percentile(np.concatenate([d1, d2]), 90))
                man_a = C.trimesh_to_manifold(m_al)
                vi = (man_a ^ gg.manifold).volume()
                vu = man_a.volume() + gg.manifold.volume() - vi
                iou = float(vi / vu) if vu > 0 else 0.0
                S["align_rms_mm_" + tag] = al["rms_mm"]
                S["align_offset_mm_" + tag] = al["offset_mm"]
                S["align_rot_deg_" + tag] = al["rotation_deg"]
                S["align_multistart_" + tag] = 1.0 if al["multistart"] else 0.0
                S["shape_surface_mean_mm_" + tag] = float(0.5 * (np.mean(np.sort(d1)[: max(1, int(0.95 * len(d1)))]) +
                                                                 np.mean(np.sort(d2)[: max(1, int(0.95 * len(d2)))])))
            except Exception as exc:  # noqa: BLE001 - a part the probes cannot handle scores 0 on its items
                det.append("%s: alignment failed (%s)" % (tag, str(exc)[:120]))
                aligned.pop(tag, None)
                iou, sd = 0.0, float("inf")
        ious[tag], sd_all[tag] = iou, sd
        S["shape_iou_" + tag] = iou
        S["shape_surface_mm_" + tag] = sd if math.isfinite(sd) else -1.0
        shape_items.append(_item("surface:" + tag, Ws["surface"] / (ssw * n), _band(sd, B["surface_mm"]), None, sd))
        shape_items.append(_item("iou:" + tag, Ws["iou"] / (ssw * n), _band(1.0 - iou, B["one_minus_iou"]), None, 1.0 - iou))
        eg = gp["geom"].bounds[1] - gp["geom"].bounds[0]
        if tag in aligned:
            ea = aligned[tag].bounds[1] - aligned[tag].bounds[0]
            err = np.abs(ea - eg)
        else:
            err = np.full(3, np.inf)
        S["extent_err_mm_" + tag] = float(err.max()) if np.all(np.isfinite(err)) else -1.0
        for i in range(3):
            ext_items.append(_item("extent:%s:%s" % (tag, "xyz"[i]), Wf["extents"] / (3 * n), _band(err[i], B["extent_mm"]), L, err[i]))
    # cut-outs
    fc = cfg["features"]
    cut_items = []
    for m, f in enumerate(feats):
        tag = f["part"]
        L = leg_of_link.get(ref.parts[tag]["link"])
        kind = str(f.get("kind", "F-cut")).replace("F-", "")
        iid = "cutout:%d:%s:%s" % (m + 1, kind, tag)
        s, val, msg = 0.0, None, "part missing"
        if tag in aligned:
            try:
                r = FT.cutout_item(aligned[tag], ref.parts[tag]["geom"].mesh, f, px=float(fc["raster_mm"]), search_mm=float(fc["search_mm"]))
                if r["present"] > 0:
                    s = 0.5 * (_band(r["offset_mm"], B["cutout_offset_mm"]) + _band(r["area_error"], B["cutout_area_rel"]))
                val = r["offset_mm"]
                msg = "present %.0f (overlap %.2f), offset %s mm, area error %s" % (r["present"], r["overlap"], r["offset_mm"], r["area_error"])
            except Exception as exc:  # noqa: BLE001
                msg = "probe failed (%s)" % str(exc)[:100]
        S["cutout_%d_%s_%s" % (m + 1, kind, tag)] = s
        cut_items.append(_item(iid, Wf["cutouts"] / len(feats), s, L, val))
        det.append("%s %s on %s: %s -> %.2f" % (f.get("kind"), m + 1, tag, msg, s))
    if not feats:
        cut_items.append(_item("cutout:none", Wf["cutouts"], 1.0))
    # openings of the centre plates
    oc = cfg["openings"]
    plates = [t for t in tags if ref.parts[t].get("design") in set(oc["designs"])]
    open_items = []
    for tag in plates:
        gg = ref.parts[tag]["geom"]
        holes_all = _plate_holes(gg.manifold, gg.bounds, float(oc["d_search_mm"][0]), float(oc["d_search_mm"][1]))
        if not holes_all:
            open_items.append(_item("openings:" + tag, Wf["openings"] / len(plates), 1.0, None, 0.0))
            det.append("openings %s: the plate has none" % tag)
            continue
        ds = [h[2] for h in holes_all]
        d_lo, d_hi = min(ds) - float(oc["d_margin_mm"]), max(ds) + float(oc["d_margin_mm"])
        hg = _plate_holes(gg.manifold, gg.bounds, d_lo, d_hi)
        err = float("inf")
        ha = []
        if tag in aligned:
            try:
                man = C.trimesh_to_manifold(aligned[tag])
                ha = _plate_holes(man, aligned[tag].bounds, d_lo, d_hi)
                err = FT.openings_error(ha, hg, B["opening_mm"][1])
                err = 0.0 if err is None else float(err)
            except Exception as exc:  # noqa: BLE001
                det.append("openings %s: probe failed (%s)" % (tag, str(exc)[:100]))
        S["openings_gt_" + tag], S["openings_agent_" + tag] = float(len(hg)), float(len(ha))
        S["openings_err_mm_" + tag] = err if math.isfinite(err) else -1.0
        open_items.append(_item("openings:" + tag, Wf["openings"] / len(plates), _band(err, B["opening_mm"]), None, err))
        det.append("openings %s: %d on the robot's plate (d %.1f-%.1f mm), %d found, mean matched error %s" % (
            tag, len(hg), d_lo, d_hi, len(ha), "%.2f mm" % err if math.isfinite(err) else "-"))
    if not plates:
        open_items.append(_item("openings:none", Wf["openings"], 1.0))
    return shape_items, cut_items + ext_items + open_items, ious, sd_all, det


# ============================================================================= mass properties
def _mass(sub, ref, comp, legs, cfg, S):
    B, Wm = cfg["bands"], cfg["subweights"]["mass"]
    rho = float(cfg["material"]["density_kg_m3"])
    leg_of_link = {l: leg["name"] for leg in legs for l in leg["links"]}
    ma, mg = _mass_props(sub, comp, rho), _mass_props(ref, comp, rho)
    links = list(ref.order)
    sw = float(Wm["mass"] + Wm["com"])
    items, det = [], []
    for ln in links:
        L = leg_of_link.get(ln)
        m_g, c_g = mg[ln]
        rel, dc = float("inf"), float("inf")
        if ln in ma and ma[ln][0] > 0:
            rel = (ma[ln][0] - m_g) / m_g
            dc = float(np.linalg.norm(ma[ln][1] - c_g))
        S["mass_rel_err_" + ln] = rel if math.isfinite(rel) else -1.0
        S["com_err_mm_" + ln] = dc if math.isfinite(dc) else -1.0
        S["mass_kg_" + ln] = ma[ln][0] if ln in ma else 0.0
        items.append(_item("mass:" + ln, Wm["mass"] / (sw * len(links)), _band(rel, B["mass_rel"]), L, rel))
        items.append(_item("com:" + ln, Wm["com"] / (sw * len(links)), _band(dc, B["com_mm"]), L, dc))
        det.append("%s %+.3f / %.1f mm" % (ln, rel, dc))
    S["total_mass_kg"] = float(sum(v[0] for v in ma.values()))
    S["total_mass_kg_reference"] = float(sum(v[0] for v in mg.values()))
    return items, det


# ============================================================================= component fit
def _ramp(frac, full, zero):
    """1 at frac >= full, 0 at frac <= zero, linear between (the arm's ring_score with its two thresholds)."""
    if full <= zero:
        return 1.0 if frac >= full else 0.0
    return max(0.0, min(1.0, (float(frac) - zero) / (full - zero)))


def _sections(model, link, T_to_core, z, normal, T_from_core):
    """A point test for 2-D points (u, v) of the core frame's plane z (normal along the core's z): inside the link's
    printed material. One planar section per part (manifold slice in the core frame), even-odd over its loops; when the
    plane is not perpendicular to the core's z the 3-D cube test is used."""
    if abs(float(normal[2])) < 0.999:
        def test3(pts2):
            pts = np.c_[np.asarray(pts2, float).reshape(-1, 2), np.full(len(pts2), z)]
            out = np.zeros(len(pts), bool)
            for _tag, p in model.link_parts(link):
                out |= p["geom"].contains(_apply(T_from_core, pts))
            return out
        return test3
    loops = []
    for _tag, p in model.link_parts(link):
        man = C.manifold_transform(p["geom"].manifold, T_to_core)
        bb = man.bounding_box()
        if not (bb[2] - 1e-9 <= z <= bb[5] + 1e-9):
            continue
        polys = [np.asarray(q, float) for q in man.slice(float(z)).to_polygons() if len(q) >= 3]
        if polys:
            loops.append(polys)

    def test(pts2):
        pts2 = np.asarray(pts2, float).reshape(-1, 2)
        out = np.zeros(len(pts2), bool)
        for polys in loops:
            m = np.zeros(len(pts2), bool)
            for poly in polys:
                m ^= _in_polygon(pts2, poly)
            out |= m
        return out
    return test


def _fit_probes(comp, fc):
    """The fit probes of core.json in the core frame, prepared once (CONTRACTS 9 plus the core lane's rules,
    gt_quad/core.py PROBES['rules'], when the record carries them):
      seats   per seat ring: n points (radius r about the axis through the centre) in the carrier's material,
              full at >= ring_score[1] inside;
      mount   per screw: the material points (the screw's `stator_clamp` points when given, else the ring of radius
              ring_r about the screw axis; full at min(ring_score[1], source_fraction), so a rule the source shells meet
              scores them 1) and the screw path (the probe point) empty;
      key     the output turns with the driven link, so the socket is swept about z over [0, symmetry_deg) and the best
              (face, angle) kept: the key points (the outline vertices when the record gives a probe outline
              (`outline_inset_mm`), else an interior grid inset by key_inset_mm) at `depth` x key_depth_fraction inside
              the socket in the driven link's material (full at >= key_full_fraction), the face ring (`face_ring`:
              radius r, `offset` beyond the face) in the driven link's material (full at >= ring_score[1]), and the
              key-screw axes (`holes`, `hole_offset` beyond the face) empty."""
    pr = comp.probes
    lo_r, hi_r = [float(v) for v in fc["ring_score"]]
    out = {"seats": [], "mount": [], "key": []}
    for r in pr.get("seat_rings", []):
        out["seats"].append((r.get("name", "seat"), _ring_points(r["centre"], r["axis"], r["r"], r.get("n", 16))))
    clamp = {c.get("name"): np.asarray(c["points"], float).reshape(-1, 3) for c in pr.get("stator_clamp", []) if c.get("points")}
    for mp in pr.get("mount", []):
        name = mp.get("name", "screw")
        if name in clamp:
            mat, full = clamp[name], hi_r
        else:
            mat = _ring_points(mp["point"], mp["axis"], mp["ring_r"], mp.get("n", 8))
            sf = _finite(mp.get("source_fraction"))
            full = min(hi_r, sf) if sf is not None and sf > 0 else hi_r
        out["mount"].append((name, mat, np.asarray([mp["point"]], float), full, full * lo_r / hi_r))
    key = pr.get("key") or {}
    if key.get("outline") and key.get("faces"):
        outline = np.asarray(key["outline"], float).reshape(-1, 2)
        if "outline_inset_mm" in key:
            kp2 = outline
        else:
            kp2 = _key_grid(outline, float(fc["key_grid_mm"]), float(fc["key_inset_mm"]))
        depth = float(key.get("depth", 2.0)) * float(fc["key_depth_fraction"])
        fr = key.get("face_ring")
        ring2 = np.zeros((0, 2))
        if fr:
            th = 2.0 * math.pi * np.arange(int(fr.get("n", 16))) / float(fr.get("n", 16))
            ring2 = float(fr["r"]) * np.c_[np.cos(th), np.sin(th)]
        holes2 = np.asarray(key.get("holes") or [], float).reshape(-1, 2)
        sym = float(key.get("symmetry_deg", 0.0) or 0.0)
        steps = max(1, int(fc["key_sweep_steps"])) if sym > 0 else 1
        angles = [sym * k / steps for k in range(steps)]
        for face in key["faces"]:
            nrm = _unit(face["normal"])
            z = float(face["z"])
            out["key"].append({"z": z, "normal": nrm, "kp2": kp2, "ring2": ring2, "holes2": holes2, "angles": angles,
                               "z_key": z - depth * float(nrm[2]), "z_ring": z + float((fr or {}).get("offset", 0.5)) * float(nrm[2]),
                               "z_hole": z + float(key.get("hole_offset", 1.0)) * float(nrm[2])})
    return out


def _fit(sub, comp, legs, cfg, struct_ok, S):
    fc = cfg["component_fit"]
    lo_r, hi_r = [float(v) for v in fc["ring_score"]]
    key_full = float(fc["key_full_fraction"])
    thr = float(fc["keepout_ratio"])
    probes = _fit_probes(comp, fc)
    leg_of_joint = {jn: leg["name"] for leg in legs for jn in leg["joints"]}
    joints = [jn for leg in legs for jn in leg["joints"]]
    items, det = [], []
    mounted = 0

    def contains(model, link, pts_link):
        out = np.zeros(len(pts_link), bool)
        for _tag, p in model.link_parts(link):
            out |= p["geom"].contains(pts_link)
        return out

    def keepout_vol(model, link, T):
        ko = C.manifold_transform(comp.keepout.manifold, T)
        bb = ko.bounding_box()
        vol = 0.0
        for _tag, p in model.link_parts(link):
            g = p["geom"]
            if _aabb_overlap(bb, tuple(g.bounds[0]) + tuple(g.bounds[1])):
                vol += (ko ^ g.manifold).volume()
        return vol

    for jn in joints:
        L = leg_of_joint[jn]
        j = sub.joints.get(jn)
        drv = sub.driver(comp, jn) if j is not None else None
        if j is None or drv is None or j["child"] not in sub.links:
            items.append(_item("fit:" + jn, 1.0 / len(joints), 0.0, L, None))
            S["fit_" + jn] = 0.0
            det.append("%s: no driving core" % jn)
            continue
        key_angle = float("nan")
        try:
            c = sub.cores[j["driven_by"]]
            carrier, driven = c["link"], j["child"]
            Tc = c["T"]
            To = _trans(sub.origin(carrier) - sub.origin(driven)) @ Tc        # core frame -> driven link frame (q = 0)
            v_c = keepout_vol(sub, carrier, Tc)
            v_d = keepout_vol(sub, driven, To)
            ko = min(C.score_ratio(v_c / comp.core_volume, thr), C.score_ratio(v_d / comp.core_volume, thr))
            # the carrier's probes in one query
            chunks = [pts for _n, pts in probes["seats"]]
            for _n, mat, axis_pt, _f, _z in probes["mount"]:
                chunks += [mat, axis_pt]
            inside = contains(sub, carrier, _apply(Tc, np.vstack(chunks))) if chunks else np.zeros(0, bool)
            k0, seat_s, mount_s = 0, [], []
            for _n, pts in probes["seats"]:
                seat_s.append(_ramp(float(inside[k0:k0 + len(pts)].mean()), hi_r, lo_r))
                k0 += len(pts)
            for _n, mat, axis_pt, full, zero in probes["mount"]:
                frac = float(inside[k0:k0 + len(mat)].mean())
                k0 += len(mat)
                empty = not bool(inside[k0])
                k0 += 1
                mount_s.append(_ramp(frac, full, zero) * (1.0 if empty else 0.0))
            # the driven link's key probes: every face and sweep angle against planar sections of the driven link
            keyv = 1.0
            if probes["key"]:
                best = (-1.0, None)
                Tinv = np.linalg.inv(To)                                   # driven link frame -> core frame
                for f in probes["key"]:
                    sec = {}
                    for name in ("z_key", "z_ring", "z_hole"):
                        sec[name] = _sections(sub, driven, Tinv, f[name], f["normal"], To)
                    for ang in f["angles"]:
                        c_, s_ = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                        R2 = np.array([[c_, -s_], [s_, c_]])
                        fk = float(sec["z_key"](f["kp2"] @ R2.T).mean()) if len(f["kp2"]) else 1.0
                        fr_ = float(sec["z_ring"](f["ring2"] @ R2.T).mean()) if len(f["ring2"]) else 1.0
                        holes_empty = not bool(sec["z_hole"](f["holes2"] @ R2.T).any()) if len(f["holes2"]) else True
                        sc = _ramp(fk, key_full, key_full - (hi_r - lo_r)) * _ramp(fr_, hi_r, lo_r) * (1.0 if holes_empty else 0.0)
                        if sc > best[0] + 1e-12:
                            best = (sc, ang)
                keyv, key_angle = best[0], best[1]
            seats = _combine(seat_s, fc["seats_combine"])
            mount = _combine(mount_s, fc["mount_combine"])
            fit = ko * seats * mount * keyv
        except Exception as exc:  # noqa: BLE001
            fit, ko, seats, mount, keyv, v_c, v_d = 0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), float("nan")
            det.append("%s: probe failed (%s)" % (jn, str(exc)[:100]))
        mounted += 1 if (fit >= float(fc["mounted_min"]) and struct_ok.get(L, False)) else 0
        S["fit_" + jn], S["fit_keepout_" + jn], S["fit_seats_" + jn] = fit, ko, seats
        S["fit_mount_" + jn], S["fit_key_" + jn] = mount, keyv
        S["fit_key_angle_deg_" + jn] = _num(key_angle, -1.0)
        S["fit_keepout_mm3_carrier_" + jn], S["fit_keepout_mm3_driven_" + jn] = _num(v_c, -1.0), _num(v_d, -1.0)
        items.append(_item("fit:" + jn, 1.0 / len(joints), fit, L, 1.0 - fit))
        det.append("%s: keep-out %.2f (%.1f / %.1f mm3), seats %.2f, mount %.2f, key %.2f (at %s deg) -> %.2f" % (
            jn, ko, _num(v_c, -1), _num(v_d, -1), seats, mount, keyv, "%.0f" % key_angle if math.isfinite(key_angle) else "-", fit))
    S["cores_mounted"] = float(mounted)
    return items, det


# ============================================================================= interference
class Collider(object):
    """The interference rule (the arm's check_interference, applied per leg): a link's collision items are its printed
    parts, its cores and its feet (foot.step); a core's item is the given collision solid (core.json `collision_step`:
    the unit without the output pulley's swept region, whose pulley and bearing inner rings turn with the driven link;
    core.step when the record names none); a pair of links scores score_ratio(intersection volume / the smaller
    link's item volume, threshold), the threshold `adjacent_ratio` for a joint's parent and child and
    `nonadjacent_ratio` otherwise; a pose scores the minimum over the pairs of its leg (the leg's links with each other
    and with the root link; other legs are not checked). Item pairs are tested by bounding sphere and box first."""

    def __init__(self, model, comp, cfg, adjacent=None):
        self.model, self.comp = model, comp
        ic = _merge(DEFAULT_CONFIG, cfg or {})["interference"]
        self.adj_thr, self.non_thr = float(ic["adjacent_ratio"]), float(ic["nonadjacent_ratio"])
        if adjacent is None:
            adjacent = {(j["parent"], j["child"]) for j in model.joints.values()}
        self.adjacent = set(adjacent) | {(b, a) for a, b in adjacent}
        self.sets = {}

    def items(self, link):
        if link not in self.sets:
            its = []
            for _tag, p in self.model.link_parts(link):
                g = p["geom"]
                its.append((g.manifold, g.mesh_volume, g.centre, g.radius))
            core, foot = getattr(self.comp, "core_collision", self.comp.core), self.comp.foot
            for _cid, c in self.model.link_cores(link):
                its.append((C.manifold_transform(core.manifold, c["T"]), core.mesh_volume, _apply(c["T"], core.centre), core.radius))
            for _fid, f in self.model.link_feet(link):
                its.append((C.manifold_transform(foot.manifold, f["T"]), foot.mesh_volume, _apply(f["T"], foot.centre), foot.radius))
            self.sets[link] = its
        return self.sets[link]

    def pairs(self, leg):
        group = [self.model.root] + list(leg["links"])
        return [(group[i], group[k], self.adj_thr if (group[i], group[k]) in self.adjacent else self.non_thr)
                for i in range(len(group)) for k in range(i + 1, len(group))]

    def score(self, leg, q, worst=None):
        """The pose score of leg at its joint vector q (the leg's joint order); worst: [score, text] updated in place."""
        T = self.model.fk(dict(zip(leg["joints"], q)))
        placed = {}

        def get(link, idx):
            key = (link, idx)
            if key not in placed:
                m = C.manifold_transform(self.items(link)[idx][0], T[link])
                placed[key] = (m, m.bounding_box())
            return placed[key]
        out = 1.0
        for a, b, thr in self.pairs(leg):
            ia_list, ib_list = self.items(a), self.items(b)
            vol = 0.0
            for ia, (_ma, _va, cen_a, rad_a) in enumerate(ia_list):
                ca = _apply(T[a], cen_a)
                for ib, (_mb, _vb, cen_b, rad_b) in enumerate(ib_list):
                    if float(np.linalg.norm(ca - _apply(T[b], cen_b))) > rad_a + rad_b:
                        continue
                    ma, bba = get(a, ia)
                    mb, bbb = get(b, ib)
                    if _aabb_overlap(bba, bbb):
                        vol += (ma ^ mb).volume()
            tot = min(sum(it[1] for it in ia_list), sum(it[1] for it in ib_list))
            ratio = vol / max(1e-9, tot)
            s = C.score_ratio(ratio, thr)
            if worst is not None and s < worst[0]:
                worst[0], worst[1] = s, "%s-%s %.1f mm3 (ratio %.1e) at q = %s" % (a, b, vol, ratio, np.round(np.asarray(q), 3).tolist())
            out = min(out, s)
            if out <= 0.0:
                break
        return out


def interference_scores(model, comp, leg, qs, cfg=None, adjacent=None):
    """Public form of the grader's interference rule for one leg: the pose scores at the joint vectors `qs` (n x k, the
    leg's joint order) on `model` (a Model, e.g. the reference) and the worst pair. The hidden operating box should be
    built with this function so the oracle scores 1 here. Returns (scores, worst_text)."""
    col = Collider(model, comp, cfg, adjacent)
    worst = [1.0, ""]
    return [col.score(leg, q, worst) for q in np.asarray(qs, float).reshape(-1, len(leg["joints"]))], worst[1]


def _interference(sub, ref, comp, legs, cases, cfg, struct_ok, S, log):
    ic = cfg["interference"]
    n_int, n_decl = int(ic["configs_per_leg"]), int(ic["declared_configs_per_leg"])
    col = Collider(sub, comp, cfg, adjacent={(j["parent"], j["child"]) for j in ref.joints.values()})
    items, det = [], []
    for leg in legs:
        L = leg["name"]
        if not struct_ok[L]:
            items.append(_item("inter:" + L, 1.0 / len(legs), 0.0, L, None))
            S["inter_" + L] = 0.0
            continue
        lo_g = np.array([ref.joints[j]["lo"] for j in leg["joints"]])
        hi_g = np.array([ref.joints[j]["hi"] for j in leg["joints"]])
        lo_d = np.array([sub.joints[j]["lo"] for j in leg["joints"]])
        hi_d = np.array([sub.joints[j]["hi"] for j in leg["joints"]])
        stream = np.asarray(cases["int"][L])
        op, skipped = [], 0
        for u in stream:                                   # poses of the hidden box inside the declared limits
            q = lo_g + u * (hi_g - lo_g)
            if np.any(q < lo_d - 1e-9) or np.any(q > hi_d + 1e-9):
                skipped += 1
                continue
            op.append(q)
            if len(op) >= n_int:
                break
        share = 1.0                                        # the declared box's share inside the hidden box
        for i in range(len(leg["joints"])):
            width = max(hi_d[i] - lo_d[i], 1e-9)
            share *= min(1.0, max(0.0, min(hi_d[i], hi_g[i]) - max(lo_d[i], lo_g[i])) / width)
        decl = []
        if not (np.all(lo_d >= lo_g - 1e-6) and np.all(hi_d <= hi_g + 1e-6)):
            for u in stream:                               # declared poses beyond the hidden box
                q = lo_d + u * (hi_d - lo_d)
                if np.all(q >= lo_g - 1e-6) and np.all(q <= hi_g + 1e-6):
                    continue
                decl.append(q)
                if len(decl) >= n_decl:
                    break
        worst = [1.0, ""]
        s_op = [col.score(leg, q, worst) for q in op]
        s_decl = [col.score(leg, q, worst) for q in decl]
        inter_op = float(np.mean(s_op)) if s_op else 0.0
        inter = inter_op
        if s_decl:
            inter = share * inter_op + (1.0 - share) * float(np.mean(s_decl))
        S["inter_" + L], S["inter_op_" + L] = inter, inter_op
        S["inter_decl_" + L] = float(np.mean(s_decl)) if s_decl else -1.0
        S["inside_share_" + L] = share
        S["operating_configs_" + L], S["configs_outside_declared_" + L] = float(len(op)), float(skipped)
        S["declared_range_configs_" + L] = float(len(decl))
        items.append(_item("inter:" + L, 1.0 / len(legs), inter, L, 1.0 - inter))
        det.append("%s: %d poses (%d clean), %d beyond the hidden box (share inside %.2f), worst %s -> %.3f" % (
            L, len(op), sum(1 for s in s_op if s >= 1.0), len(decl), share, worst[1] or "none", inter))
    return items, det


# ============================================================================= the grade
def grade(sub_dir, ref_dir, given_dir, config, rng, log=None, prepared=None):
    """Grade the converted mechanism in `sub_dir` against the reference (module docstring)."""
    t0 = time.time()
    log = log or (lambda m: None)
    cfg = _merge(DEFAULT_CONFIG, config or {})
    _apply_module_knobs(cfg)
    if prepared is None:
        prepared = prepare(ref_dir, given_dir, cfg)          # a broken reference / given folder raises: a task error
    ref, comp, legs, feats = prepared["ref"], prepared["comp"], prepared["legs"], prepared["features"]
    cases = make_cases(rng, legs, ref, cfg)
    res = _empty_result(cfg)
    res["stats"]["time_prepare_s"] = time.time() - t0
    try:
        _grade_submission(res, sub_dir, ref, comp, legs, feats, cases, cfg, log)
    except Exception:  # noqa: BLE001 - a submission that breaks the grader earns 0, never a crash
        tb = traceback.format_exc()
        log("grader exception on this submission:\n" + tb)
        keep = res["stats"]
        res = _empty_result(cfg)
        res["stats"].update({k: v for k, v in keep.items() if k.startswith("time_")})
        res["stats"]["grader_exception"] = 1.0
        res["details"]["cad_valid"] = ["grader exception on this submission: " + tb[-600:]]
    res["stats"]["time_grade_total_s"] = time.time() - t0
    res["stats"] = {k: float(v) for k, v in res["stats"].items() if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool)}
    res["items"] = _sanitize(res["items"])
    return res


def _grade_submission(res, sub_dir, ref, comp, legs, feats, cases, cfg, log):
    R, S, D, I, CR = res["rewards"], res["stats"], res["details"], res["items"], res["criteria"]
    S["grader_exception"] = 0.0
    t = time.time()
    sub = Model(sub_dir, "submission", cfg)
    S["time_load_s"] = time.time() - t
    S["parts_loaded"] = float(sum(1 for p in sub.parts.values() if p["geom"] is not None))
    S["parts_missing"] = float(sum(1 for tg in ref.parts if tg not in sub.parts or sub.parts[tg]["geom"] is None))
    S["parts_unattached"] = float(sum(1 for p in sub.parts.values() if p["geom"] is not None and p.get("unattached")))
    if sub.notes:
        D["notes"] = sub.notes[:30]
    if not sub.ok:
        D["cad_valid"] = sub.errors[:10]
        log("[check] cad_valid 0: " + " | ".join(sub.errors[:4]))
        return
    fatal = []
    if sub.root != ref.root:
        fatal.append("the root link is %s (the robot's body is %s)" % (sub.root, ref.root))
    conv = sub.conversion or {}
    if conv.get("ok") is False or conv.get("cad_valid") in (False, 0, 0.0):
        fatal.append("the converter rejected the mechanism: %s" % "; ".join(str(e) for e in (conv.get("errors") or []))[:300])
    if fatal:
        D["cad_valid"] = fatal
        log("[check] cad_valid 0: " + " | ".join(fatal))
        return
    # ---- legs: structure, consistency, the converter's verdict
    t = time.time()
    flags = _converter_flags(sub.conversion, legs)
    valid, struct_ok, leg_det = {}, {}, []
    cons = {}
    for leg in legs:
        L = leg["name"]
        ok, why = _leg_structure(sub, ref, leg)
        struct_ok[L] = ok
        c_ok = True
        for jn in leg["joints"]:
            if jn in sub.joints and sub.driver(comp, jn) is not None:
                cons[jn] = _consistency(sub, comp, jn, cfg)
            else:
                cons[jn] = {"ok": False, "angle_deg": float("nan"), "offset_mm": float("nan"), "gap_mm": float("nan"), "why": "joint or core missing"}
            c_ok = c_ok and cons[jn]["ok"]
            S["consistent_" + jn] = 1.0 if cons[jn]["ok"] else 0.0
            for k in ("angle_deg", "offset_mm", "gap_mm"):
                S["consistency_%s_%s" % (k, jn)] = _num(cons[jn][k], -1.0)
        if flags is not None:
            v = ok and flags.get(L, False)
            src = "the converter's legs_valid"
            if not flags.get(L, False):
                # the converter's reasons first: they name the cause (a missing part, a joint not read), the structure
                # test only its consequences, and the diagnostic keeps the first 200 characters
                rs = (sub.conversion.get("reasons") or {}).get(L) if isinstance(sub.conversion.get("reasons"), dict) else None
                why = ["the converter: %s" % ("; ".join(map(str, rs)) if isinstance(rs, list) else (rs or "invalid"))] + why
        else:
            v = ok and c_ok
            src = "the grader's consistency test"
            if not c_ok:
                why = why + ["%s: %s" % (jn, cons[jn]["why"]) for jn in leg["joints"] if not cons[jn]["ok"]]
        valid[L] = 1.0 if v else 0.0
        S["leg_valid_" + L] = valid[L]
        S["leg_structure_ok_" + L] = 1.0 if ok else 0.0
        leg_det.append("%s %s (structure %s; verdict from %s)%s" % (L, "valid" if v else "INVALID", "ok" if ok else "FAILED",
                                                                    src, "" if v else ": " + "; ".join(why)[:200]))
    S["legs_valid"] = float(sum(valid.values()))
    S["cad_json_consistent"] = 1.0 if all(c["ok"] for c in cons.values()) else 0.0
    if isinstance(conv.get("cad_json_consistent"), dict):
        for jn, v in conv["cad_json_consistent"].items():
            f = _finite(v) if not isinstance(v, bool) else float(v)
            if f is not None:
                S["converter_consistent_" + str(jn)] = f
    D["legs_valid"] = leg_det
    S["time_legs_s"] = time.time() - t
    # ---- cad_valid: the MuJoCo model of the converted tree
    t = time.time()
    try:
        mj = _mjcf_check(sub, comp, cases, cfg, ref)
    except Exception as exc:  # noqa: BLE001
        mj = {"ok": False, "why": "MJCF check failed: %s" % str(exc)[:200], "dev_mm": float("nan"), "dev_deg": float("nan"), "bodies": 0, "dropped": []}
    S["time_mjcf_s"] = time.time() - t
    S["mjcf_ok"] = 1.0 if mj["ok"] else 0.0
    S["mjcf_fk_dev_mm"], S["mjcf_fk_dev_deg"] = _num(mj["dev_mm"], -1.0), _num(mj["dev_deg"], -1.0)
    S["mjcf_bodies"] = float(mj["bodies"])
    D["cad_valid"] = ["%d of %d parts loaded%s; %d links in the tree; MJCF: %s" % (
        S["parts_loaded"], len(ref.parts), " (%d in no link)" % S["parts_unattached"] if S["parts_unattached"] else "",
        len(sub.reached), mj.get("why", ""))]
    if mj.get("dropped"):
        D["cad_valid"].append("links left out of the MJCF (no mass or no parent): %s" % ", ".join(mj["dropped"])[:200])
    cb = conv.get("body") if isinstance(conv.get("body"), dict) else {}
    if cb.get("missing_tags") or cb.get("foreign_tags"):
        ft = cb.get("foreign_tags") or {}
        foreign = ", ".join("%s (%s)" % (t, ft[t]) for t in sorted(ft)) if isinstance(ft, dict) else ", ".join(map(str, ft))
        missing = ", ".join(map(str, cb.get("missing_tags") or []))
        D["cad_valid"].append("the grounded body %s (only those parts' items and their legs are lost)" % "; it ".join(
            (["lacks " + missing] if missing else []) + (["carries " + foreign] if foreign else [])))
    R["cad_valid"] = 1.0 if mj["ok"] else 0.0
    # ---- public model reuse
    t = time.time()
    reuse, rdet, matches, similar = _public_reuse(sub, comp, cfg)
    R["public_model_reuse"] = reuse
    D["public_model_reuse"] = rdet or ["no part file or fingerprint matches the blocklist%s" % ("" if comp.blocklist["present"] else " (no blocklist given)")]
    S["source_geometry_matches"], S["source_geometry_similar"] = float(matches), float(similar)
    S["time_reuse_s"] = time.time() - t
    log("[check] cad_valid %.0f, public_model_reuse %.0f, legs valid %d" % (R["cad_valid"], reuse, S["legs_valid"]))
    if R["cad_valid"] <= 0:
        return
    seed = int(cases["sample_seed"])
    S["sample_seed"] = float(seed)
    # ---- criteria
    t = time.time()
    kin_items, fk_med, kdet = _kinematic(sub, ref, comp, legs, cases, cfg, struct_ok, S)
    S["time_kinematic_s"] = time.time() - t
    t = time.time()
    shape_items, feat_items, ious, sds, pdet = _parts(sub, ref, legs, feats, cfg, seed, S, log)
    S["time_parts_s"] = time.time() - t
    t = time.time()
    mass_items, mdet = _mass(sub, ref, comp, legs, cfg, S)
    S["time_mass_s"] = time.time() - t
    t = time.time()
    fit_items, fdet = _fit(sub, comp, legs, cfg, struct_ok, S)
    S["time_fit_s"] = time.time() - t
    t = time.time()
    int_items, idet = _interference(sub, ref, comp, legs, cases, cfg, struct_ok, S, log)
    S["time_interference_s"] = time.time() - t
    # ---- floors
    fl = cfg["floors"]
    if fl.get("shape_p90_mm"):
        # the shape floor on each part's p90 surface distance, band (full, zero); a missing part counts 0 (its p90 is inf).
        # PRECISION.md section 8: on thin shells an IoU floor cannot spare honest wall errors without sparing crude
        # geometry (convex hulls, boxes); the p90 distance separates them. The band (CALIBRATION.md, round 1) starts
        # where the surface item stops paying and is 0 at about a part's bounding box
        F_s = float(np.mean([_band(sds[tg], fl["shape_p90_mm"]) for tg in sorted(ref.parts)]))
    else:
        a, b = fl["shape_iou"]
        F_s = float(np.mean([min(1.0, max(0.0, (ious[tg] - a) / (b - a))) for tg in sorted(ref.parts)]))
    # GRADER_SPEC 6: F_k(L) = band(fk median of leg L) alone. An invalid leg's own items are already 0 (the validity
    # gate); its F_k only enters the body items' floor (mean F_k), and it is 0 there whenever the leg's chain is broken
    # (no FK: median inf). A structurally complete leg that fails the converter's consistency test keeps the floor its
    # foot points earn (it was x valid(L), against the spec; review fix 2026-10-27)
    F_k = {L: _band(fk_med[L], cfg["floors"]["leg_fk_median_mm"]) for L in fk_med}
    F_k_mean = float(np.mean(list(F_k.values())))
    S["floor_shape"], S["floor_fk_mean"] = F_s, F_k_mean
    for L, v in F_k.items():
        S["floor_fk_" + L] = v
    S["iou_mean"] = float(np.mean(list(ious.values())))
    fin = [v for v in sds.values() if math.isfinite(v)]
    S["surface_mm_mean"] = float(np.mean(fin)) if fin else -1.0
    gate = R["cad_valid"] * R["public_model_reuse"]
    lists = {"kinematic_fidelity": (kin_items, lambda L: F_s, True), "feature_fidelity": (feat_items, lambda L: F_s, True),
             "shape_fidelity": (shape_items, lambda L: 1.0, False),
             "mass_properties": (mass_items, lambda L: F_s * (F_k[L] if L is not None else F_k_mean), True),
             "component_fit": (fit_items, lambda L: F_s * (F_k[L] if L is not None else F_k_mean), True),
             "interference_free": (int_items, lambda L: F_s * (F_k[L] if L is not None else F_k_mean), True)}
    W = cfg["weights"]
    for c in CRITERIA:
        its, floor_of, gated = lists[c]
        raw, fl = _criterion(its, valid, floor_of, gated)
        final = float(max(0.0, min(1.0, fl * gate)))
        CR[c] = {"raw": raw, "floored": fl, "final": final, "weight": float(W[c])}
        R[c] = final
        I[c] = its
        S["raw_" + c], S["floored_" + c] = raw, fl
    R["design_score"] = min(1.0, math.fsum(float(W[c]) * R[c] for c in CRITERIA) / math.fsum(float(W[c]) for c in CRITERIA))
    S["design_score"] = R["design_score"]
    grp = {}
    for c in ("kinematic_fidelity", "feature_fidelity"):
        acc = {}
        for it in I[c]:
            g = it["id"].split(":")[0]
            x = acc.setdefault(g, [0.0, 0.0])
            x[0] += it["weight"] * it["score"] * it["valid"]
            x[1] += it["weight"]
        grp[c] = {g: (v[0] / v[1] if v[1] > 0 else 0.0) for g, v in acc.items()}
        for g, v in grp[c].items():
            S["%s_%s" % (c.split("_")[0], g)] = v
    D["kinematic_fidelity"] = ["axes %.3f, lengths %.3f, FK %.3f, range %.3f (gated, before F_s %.3f)" % (
        grp["kinematic_fidelity"].get("axis", 0), grp["kinematic_fidelity"].get("length", 0), grp["kinematic_fidelity"].get("fk", 0),
        grp["kinematic_fidelity"].get("range", 0), F_s)] + kdet
    D["feature_fidelity"] = ["cut-outs %.3f, extents %.3f, openings %.3f (gated, before F_s)" % (
        grp["feature_fidelity"].get("cutout", 0), grp["feature_fidelity"].get("extent", 0), grp["feature_fidelity"].get("openings", 0))] + \
        [d for d in pdet if d.startswith(("F-", "openings", "cut"))][:12]
    D["shape_fidelity"] = ["mean IoU %.3f, mean p90 surface distance %.2f mm; floors F_s %.3f, F_k %s" % (
        S["iou_mean"], S["surface_mm_mean"], F_s, ", ".join("%s %.2f" % (L, v) for L, v in F_k.items()))] + \
        [d for d in pdet if not d.startswith(("F-", "openings", "cut"))][:12]
    D["mass_properties"] = ["per link mass rel. error / CoM error: " + ", ".join(mdet)]
    D["component_fit"] = ["cores mounted (fit >= %.2f) %d of %d" % (cfg["component_fit"]["mounted_min"], S["cores_mounted"], len(fit_items))] + fdet
    D["interference_free"] = idet
    log("[check] design_score %.4f: %s" % (R["design_score"], ", ".join("%s %.3f" % (c, R[c]) for c in CRITERIA)))


# ============================================================================= developer CLI
def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(description="grade a converted quad-cad-gt/1 folder against a reference (dev use)")
    ap.add_argument("sub_dir")
    ap.add_argument("ref_dir")
    ap.add_argument("given_dir")
    ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "grader_config.json"))
    ap.add_argument("--seeds", default="2026:1027", help="hi:lo of the SeedSequence (ARB_SEEDS_PIN form)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config) as fh:
            cfg = json.load(fh)
    hi, lo = [int(x) for x in args.seeds.split(":")]
    rng = np.random.default_rng(np.random.SeedSequence([hi, lo]))
    import sys
    res = grade(args.sub_dir, args.ref_dir, args.given_dir, cfg, rng, log=lambda m: print(m, file=sys.stderr))
    print(json.dumps({"rewards": res["rewards"], "criteria": res["criteria"]}, indent=1))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(_sanitize(res), fh, indent=1)


if __name__ == "__main__":
    import sys
    _main(sys.argv[1:])
