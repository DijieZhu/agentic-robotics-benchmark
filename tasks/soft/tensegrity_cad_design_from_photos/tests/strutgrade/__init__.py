"""strutgrade: the grader of tensegrity_cad_design_from_photos (one strut: 13 printed Bodies of 8 designs, 2 gearmotors
and the rod, in a FreeCAD Assembly), against the authors' own mesh pieces in tests/reference/.

The base (this package's __init__, reference, geom, recognise, fit, align, gates, detenv) reads an extraction into a
GradeContext; the scoring items (items_*.py, added without changing the base files) read the
context and return one score each. grade.py (the child-process entry) runs:

    from strutgrade import detenv; detenv.early(cpus)          # before numpy (BLAS kernel, threads)
    import strutgrade
    cfg = strutgrade.resolve_config(json.load(open("grader_config.json")))
    ctx = strutgrade.GradeContext(extract_dir, ref_dir, purchased_root, cfg)
    strutgrade.run_base(ctx, shape_score=items_shape.pose_score, feature_score=items_holes.pose_preference)
    if ctx.gate is None:
        res = strutgrade.run_item(ctx, "shape", items_shape.item)       # ... one call per item

ITEM CONTRACT
-------------
An item is a function item(ctx) -> {"score": float, "per_part": dict, "diagnostics": [str]}:
  score        in [0, 1] (run_item clamps it and turns a non-finite value into 0)
  per_part     anything per piece, keyed by reference tag ("P01" .. "P13") or purchased placement id ("GM1", "GM2",
               "ROD1"); plain JSON values only
  diagnostics  short English sentences for the verdict text (run_item keeps the first item.max_diagnostics, each cut
               to 300 characters)
It may read everything on the context and may cache on ctx.cache[<its own name>]; it must not change the base's
results (ctx.parts, ctx.fits, ctx.alignment, ctx.instances, ctx.extras). An exception inside an item makes that item
score 0, counts in statistics["grader_item_errors"] and adds a diagnostic (run_item); an exception outside every
item is the grader's own failure (grade.py turns it into a task error). The empty item below is
the smallest valid item:

    def item_empty(ctx):
        return {"score": 0.0, "per_part": {}, "diagnostics": []}

WHAT THE CONTEXT HOLDS AFTER run_base
-------------------------------------
  cfg            the resolved configuration (BASE_DEFAULTS overlaid by grader_config.json)
  ref            strutgrade.reference.Reference: designs, instances, purchased types and placements, features, seen
  sub            strutgrade.gates.Submission (model.json, joints, grounded, no_assembly)
  solids         every usable solid of the document (see gates.Submission)
  parts          credited tag -> solid record (a tagged PartDesign Body of exactly one valid closed solid, its tag on
                 no other leaf)
  instances      recognised purchased instances (recognise.form_instances): type, W (local -> submission frame),
                 slots, completeness, mode (exact | approx)
  extras         the other solids: untagged and not recognised, or demoted by a local rule (obstacles for fit and
                 clearance, printed-part density for mass)
  fits           sid -> fit result (fit.fit_part): candidates [{T (submission -> design), rms, shape, feature,
                 snapped}], tied, n_starts
  alignment      align.align's result: A (submission -> strut), assignment {reference tag: sid}, part_instance,
                 chosen {sid: candidate index}, T_fit {sid: chosen T}, purchased_assignment {placement id: instance
                 index}, residual_mm, landed, per_piece
  gate           None, or {"reason": unreadable | no_solid | too_few_fitted, "message"}: the whole document scores 0
  stats          numbers for the verdict's metrics (finite floats); diagnostics: sentences
  tmp            a private temporary folder; cache: per-item scratch
Helpers: ctx.solid(sid); ctx.part_pose(sid) = design frame -> strut frame of a fitted part as placed (A inv(T_fit));
ctx.instance_pose(k) = local -> strut of purchased instance k (A W); ctx.placed_part(sid) = the part's tessellation in
the strut frame (geom.Placed); ctx.placed_design(tag) = the reference instance's design mesh in the strut frame.

CONFIGURATION
-------------
Every number of the base is in BASE_DEFAULTS below; grader_config.json overrides any of them section by section
(resolve_config). resolve_config records in cfg["_defaults_used"] which base keys the file did not give, so the
freeze can check that the frozen file names every number.
"""
import copy
import math
import shutil
import tempfile
import time



class TaskError(Exception):
    """The grader's own reference, the image's vendor files or the extraction are unusable: a broken task (verify
    fails the run), never a zero. (Defined here, standard library only, so that importing the package does not import
    numpy before detenv.early() has set the BLAS kernel and thread counts.)"""


BASE_DEFAULTS = {
    "detenv": {"grader_cpus": 4},
    "gates": {"max_solids": 2000, "min_volume_mm3": 1e-6, "recheck_validity": True, "min_fittable": 3},
    "mesh": {"deflection_mm": 0.01, "angle_rad": 0.1, "retry": [[0.005, 0.05], [0.02, 0.2], [0.05, 0.3]],
             "weld_mm": 1e-4, "volume_tol_rel": 0.02, "fill_max_loop": 4000},
    "sampling": {"seed": 20261117, "dense_per_mm2": 20.0, "dense_min": 20000, "dense_max": 500000,
                 "query_per_mm2": 2.0, "query_min": 4000, "query_max": 60000, "disk_mm": 0.15},
    "recognition": {"volume_rel": 0.01, "exact_tol_mm": 0.001, "approx_candidates": 2, "approx_p95_mm": [0.05, 0.004],
                    "attach_mm": 1.0, "step_workers": 2, "pin_volume_eps": 1e-7, "approx_coarse_points": 1200,
                    "approx_fine_points": 6000, "approx_trim": 0.9},
    "fit": {"spin_step_deg": 15.0, "n_coarse": 800, "coarse_iters": 10, "n_fine": 6000, "fine_iters": 40,
            "refine": 8, "trim": 0.9, "bound_mm": 30.0, "window": 0.05, "feature_tie": 0.01, "max_candidates": 8,
            "exact_snap_mm": 0.05, "dedupe_mm": 0.05, "coarse_dedupe_mm": 0.5, "spin_axes": "distinct",
            "refine_rms_factor": 2.0, "refine_rms_margin_mm": 0.3, "fine_tol": 1e-7, "refine_symmetric": 24},
    "shape_minimal": {"surface_p90_mm": {"small": [0.12, 0.60], "large": [0.20, 1.00]}, "iou": [0.40, 0.95],
                      "w_surface": 0.7, "w_iou": 0.3, "n_query": 6000},
    "align": {"points_per_part": 64, "cap_mm": 2.0, "land_mm": 1.0, "rounds": 5, "shape_min": 0.2, "tie_mm": 1e-6,
              "cand_tie_mm2": 1e-6, "robust_mm": 0.25},
    "items": {"max_diagnostics": 6},
}


def resolve_config(cfg):
    """BASE_DEFAULTS overlaid by cfg (one level of sections; a section's keys override the defaults one by one; other
    top-level keys of cfg pass through). cfg["_defaults_used"] lists "section.key" of every default that was used."""
    out = copy.deepcopy(cfg) if cfg else {}
    used = []
    for sec, vals in BASE_DEFAULTS.items():
        have = out.setdefault(sec, {})
        for k, v in vals.items():
            if k not in have:
                have[k] = copy.deepcopy(v)
                used.append("%s.%s" % (sec, k))
    out["_defaults_used"] = used
    return out


def item_error(ctx, what, exc):
    """A grading step that raised: it scores 0; counted (grader_item_errors) and reported."""
    ctx.stats["grader_item_errors"] = ctx.stats.get("grader_item_errors", 0.0) + 1.0
    if sum(1 for d in ctx.diagnostics if d.startswith("Item error:")) < 20:
        ctx.diagnostics.append("Item error: %s could not be evaluated (%s: %s); it scores 0" % (
            what, type(exc).__name__, str(exc)[:160]))


def item_empty(ctx):
    """The smallest valid item (the contract's example): no score, nothing per part, nothing to say."""
    return {"score": 0.0, "per_part": {}, "diagnostics": []}


def run_item(ctx, name, fn):
    """Run one scoring item under the contract: {"score", "per_part", "diagnostics", "error"}. A gated document scores
    0 without running the item."""
    if ctx.gate is not None:
        return {"score": 0.0, "per_part": {}, "diagnostics": [], "gated": ctx.gate["reason"]}
    t0 = time.time()
    try:
        r = fn(ctx)
        if not isinstance(r, dict) or "score" not in r:
            raise TypeError("item %s returned %r instead of a result dict" % (name, type(r).__name__))
        sc = float(r["score"])
        sc = min(1.0, max(0.0, sc)) if math.isfinite(sc) else 0.0
        diags = [str(d)[:300] for d in (r.get("diagnostics") or [])][:int(ctx.cfg["items"]["max_diagnostics"])]
        out = {"score": sc, "per_part": r.get("per_part") or {}, "diagnostics": diags}
    except Exception as exc:  # noqa: BLE001 - an item that fails scores 0
        item_error(ctx, "the item %s" % name, exc)
        out = {"score": 0.0, "per_part": {}, "diagnostics": [], "error": "%s: %s" % (type(exc).__name__, str(exc)[:200])}
    ctx.stats["t_item_%s_s" % name] = time.time() - t0
    return out


class GradeContext(object):
    """Configuration, reference, submitted solids, fit and alignment results, statistics and diagnostics of one
    grade (module docstring). Construct, then run_base(ctx)."""

    def __init__(self, extract_dir, ref_dir, purchased_root, cfg, log=None, ref=None, tmpdir=None):
        self.extract_dir, self.ref_dir, self.purchased_root = extract_dir, ref_dir, purchased_root
        self.cfg = cfg if "_defaults_used" in cfg else resolve_config(cfg)
        self.log = log or (lambda m: None)
        self.tmp = tmpdir or tempfile.mkdtemp(prefix="strut_grade_")
        self.ref = ref
        self.sub = None
        self.solids, self.parts, self.instances, self.extras = [], {}, [], []
        self.fits, self.alignment = {}, None
        self.gate = None
        self.stats = {"grader_item_errors": 0.0}
        self.diagnostics = []
        self.cache = {}
        self._by_sid = {}

    # ------------------------------------------------------------------ helpers for the items
    def solid(self, sid):
        return self._by_sid[sid]

    def part_pose(self, sid):
        """Design frame -> strut frame of a fitted part as the document places it: A inv(T_fit) (chosen candidate)."""
        from .geom import inv
        return self.alignment["A"] @ inv(self.alignment["T_fit"][sid])

    def instance_pose(self, k):
        """Local -> strut frame of recognised purchased instance k: A W."""
        return self.alignment["A"] @ self.instances[k]["W"]

    def placed_part(self, sid):
        """The submitted part's tessellation in the strut frame (geom.Placed at A), cached."""
        from .geom import Placed
        from .recognise import submission_geo
        key = ("placed_part", sid)
        if key not in self.cache:
            g = submission_geo(self, self.solid(sid))
            self.cache[key] = Placed(g, self.alignment["A"]) if g is not None else None
        return self.cache[key]

    def placed_design(self, tag):
        """The reference instance's design mesh in the strut frame (geom.Placed at its pose W), cached."""
        from .geom import Placed
        key = ("placed_design", tag)
        if key not in self.cache:
            fam = self.ref.family_of[tag]
            self.cache[key] = Placed(self.ref.designs[fam]["geo"], self.ref.instances[tag]["W"])
        return self.cache[key]

    def close(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


def run_base(ctx, shape_score=None, feature_score=None):
    """The base of the grade, in order: deterministic environment, reference, submission and local rules (gate 1 and
    2), purchased recognition, per-part fit with candidates, gate 3 (fewer than three fitted pieces), whole-strut
    alignment. Sets ctx.gate when the document scores 0. Returns ctx."""
    from . import align as _align
    from . import detenv, fit as _fit, gates, recognise
    from .reference import Reference
    t0 = time.time()
    ctx.stats.update(detenv.pin(ctx.cfg))
    if ctx.ref is None:
        ctx.ref = Reference(ctx.ref_dir, ctx.purchased_root, ctx.cfg, ctx.tmp, ctx.log)
    ctx.stats["t_reference_s"] = time.time() - t0
    t1 = time.time()
    ctx.sub = gates.Submission(ctx.extract_dir, ctx.ref.tags, ctx.cfg, ctx.diagnostics)
    for k, v in ctx.sub.stats.items():
        ctx.stats[k] = float(v)
    ctx.stats["t_submission_s"] = time.time() - t1
    if not ctx.sub.ok:
        ctx.gate = ctx.sub.gate
        ctx.stats["gate_" + ctx.gate["reason"]] = 1.0
        return ctx
    ctx.solids = ctx.sub.solids
    ctx._by_sid = {s["sid"]: s for s in ctx.solids}
    ctx.parts = {s["tag"]: s for s in ctx.solids if s["tag"] is not None}
    ctx.stats["n_solids"] = float(len(ctx.solids))
    ctx.stats["n_tagged"] = float(len(ctx.parts))
    t1 = time.time()
    recognise.recognise(ctx)
    ctx.instances = recognise.form_instances(ctx)
    claimed = set()
    for it in ctx.instances:
        claimed.update(it["slots"].values())
    ctx.extras = [s for s in ctx.solids if s["tag"] is None and s["sid"] not in claimed]
    ctx.stats["n_extra_solids"] = float(len(ctx.extras))
    ctx.stats["t_recognition_s"] = time.time() - t1
    _fit.fit_all(ctx, shape_score=shape_score, feature_score=feature_score)
    # a credited part whose fit failed is not fittable: it keeps its tag but takes part in nothing that needs the fit
    g3 = gates.fit_gate(_FitCount(ctx))
    if g3 is not None:
        ctx.gate = g3
        ctx.diagnostics.append(g3["message"])
        ctx.stats["gate_" + g3["reason"]] = 1.0
        return ctx
    _align.align(ctx)
    ctx.stats["t_base_s"] = time.time() - t0
    return ctx


class _FitCount(object):
    """What gates.fit_gate counts: fitted parts and recognised purchased instances."""

    def __init__(self, ctx):
        self.parts = {sid: f for sid, f in ctx.fits.items() if f["candidates"]}
        self.instances = ctx.instances
        self.cfg = ctx.cfg
        self.stats = ctx.stats


__all__ = ["BASE_DEFAULTS", "GradeContext", "TaskError", "item_empty", "item_error", "resolve_config", "run_base",
           "run_item"]
