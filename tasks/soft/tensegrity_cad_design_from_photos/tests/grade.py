"""Grader of tensegrity_cad_design_from_photos: the verify stage's child process (verify.py runs it with a time limit
and a CPU limit through lockdown_runtime.grade_child; a development run calls grade() directly).

    python -B grade.py EXTRACTION REFERENCE PURCHASED_ROOT OUTPUT_JSON [--config grader_config.json]

EXTRACTION is the folder fcstd_extract.py wrote (format strut-fcstd-extract/1); REFERENCE is tests/reference/;
PURCHASED_ROOT holds the vendor STEPs pinned in reference/purchased_pins.json. Every number comes from
grader_config.json (strutgrade.resolve_config overlays it on the base's defaults).

Order: the base (strutgrade.run_base: deterministic environment, reference, submission and local rules,
purchased recognition, per-part fit, the three whole-document zeros, whole-strut alignment), then the seven criteria,
each through strutgrade.run_item (an exception inside an item scores that item 0 and is counted in
statistics grader_item_errors), then

    design_score = w_shape S + phi * (w_holes H + w_planes K + w_functional G + w_placement P + w_fit C + w_mass M)
    phi = rising(S; aggregate.floor) (the shape floor; 0.30 -> 0.85 today)

verify.py multiplies design_score by public_model_reuse. A gated document (unreadable, no solid, fewer than three
fittable pieces) scores 0 on everything with cad_valid 0. The grader's own failures (reference or vendor files
unusable, an exception outside every item) are a task error: the output says ok false with task_error, never a 0.

Criteria and their modules (strutgrade/): shape items_shape, holes items_holes, planes items_planes;
functional items_functional, placement items_placement, fit items_fit, mass items_mass. While the configuration
allows placeholders (aggregate.allow_placeholders, a provisional configuration only), a missing shape, holes or planes module is replaced
by a placeholder: shape = the base's minimal shape score of each part's chosen pose, weighted by features.json
weights_upper_bound; holes and planes = that same value; statistics placeholder_<criterion> = 1 say so. A frozen
configuration does not allow placeholders: a missing module is then a task error.
"""
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
CRITERIA = ["shape", "holes", "planes", "functional", "placement", "fit", "mass"]
MODULES = {"shape": "items_shape", "holes": "items_holes", "planes": "items_planes", "functional": "items_functional",
           "placement": "items_placement", "fit": "items_fit", "mass": "items_mass"}
PLACEHOLDER_OK = ("shape", "holes", "planes")
MAX_DIAGNOSTICS = 40


def load_items(cfg):
    """{criterion: module} and the criteria that run on placeholders (module docstring)."""
    import importlib
    from strutgrade import TaskError
    mods, missing = {}, []
    for k in CRITERIA:
        name = "strutgrade." + MODULES[k]
        try:
            mods[k] = importlib.import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            missing.append(k)
    allow = bool(cfg.get("aggregate", {}).get("allow_placeholders", False))
    bad = [k for k in missing if not (allow and k in PLACEHOLDER_OK)]
    if bad:
        raise TaskError("grading modules missing: %s" % ", ".join(MODULES[k] for k in bad))
    return mods, missing


def _placeholder_shape(ctx):
    """Provisional shape: the base's minimal shape score of each assigned part's chosen pose, weighted by
    features.json weights_upper_bound (while items_shape is missing)."""
    W = ctx.ref.features["weights_upper_bound"]["per_body"]
    al = ctx.alignment
    num = den = 0.0
    per = {}
    for tag in ctx.ref.tags:
        w = float(W[tag]["w_upper_bound"])
        den += w
        sid = al["assignment"].get(tag)
        s = 0.0
        if sid is not None:
            s = float(ctx.fits[sid]["candidates"][al["chosen"][sid]]["shape"])
        per[tag] = {"score": s, "weight": w}
        num += w * s
    return {"score": num / den if den > 0 else 0.0, "per_part": per,
            "diagnostics": ["Shape: provisional placeholder (the base's minimal shape score)"]}


def _finite_stats(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        x = float(v)
        if math.isfinite(x):
            out[k] = x
    return out


def grade(ex_dir, ref_dir, purchased_root, cfg, log=None, ref=None, keep_context=False):
    """The grade of one extraction (module docstring). `ref`: an already loaded strutgrade Reference (a development
    batch grades many documents against one). Returns the result dict (and the context when keep_context)."""
    import strutgrade
    from strutgrade.geom import rising
    t0 = time.time()
    log = log or (lambda m: print("[grade] %s" % m, file=sys.stderr, flush=True))
    cfg = strutgrade.resolve_config(cfg)
    mods, placeholders = load_items(cfg)
    res = {"ok": True, "design_score": 0.0, "criteria": {k: 0.0 for k in CRITERIA}, "checks": {"cad_valid": 0.0},
           "shape_floor": 0.0, "statistics": {}, "parts": {}, "purchased": {}, "diagnostics": [],
           "placeholders": list(placeholders), "config_status": str(cfg.get("status", "unknown"))}
    ctx = strutgrade.GradeContext(ex_dir, ref_dir, purchased_root, cfg, log=log, ref=ref)
    try:
        shape_cb = getattr(mods.get("shape"), "pose_score", None)
        feat_cb = getattr(mods.get("holes"), "pose_preference", None)
        strutgrade.run_base(ctx, shape_score=shape_cb, feature_score=feat_cb)
        items = {}
        if ctx.gate is None:
            res["checks"]["cad_valid"] = 1.0
            for k in CRITERIA:
                if k in mods:
                    items[k] = strutgrade.run_item(ctx, k, mods[k].item)
                elif k == "shape":
                    items[k] = strutgrade.run_item(ctx, k, _placeholder_shape)
                else:
                    s = items["shape"]["score"]
                    items[k] = strutgrade.run_item(ctx, k, lambda c, s=s: {"score": s, "per_part": {}, "diagnostics": []})
            Ag = cfg["aggregate"]
            S = items["shape"]["score"]
            phi = rising(S, Ag["floor"])
            w = Ag["weights"]
            ds = float(w["shape"]) * S + phi * sum(float(w[k]) * items[k]["score"] for k in CRITERIA if k != "shape")
            res["criteria"] = {k: float(items[k]["score"]) for k in CRITERIA}
            res["shape_floor"] = float(phi)
            res["design_score"] = float(min(1.0, max(0.0, ds)))
            for k in CRITERIA:
                res["diagnostics"] += [str(d) for d in items[k]["diagnostics"]]
        else:
            ctx.stats["gate_" + ctx.gate["reason"]] = 1.0
        res["items"] = {k: {"per_part": v.get("per_part", {}), "error": v.get("error")} for k, v in items.items()}
        for tag in ctx.ref.tags:
            s = ctx.parts.get(tag)
            rec = {"present": s is not None, "fitted": bool(s is not None and s["sid"] in ctx.fits and
                                                            ctx.fits[s["sid"]]["candidates"])}
            if ctx.alignment:
                rec["instance"] = ctx.alignment["part_instance"].get(s["sid"]) if s is not None else None
            res["parts"][tag] = rec
        pa = (ctx.alignment or {}).get("purchased_assignment", {})
        for pid in sorted(ctx.ref.placements):
            res["purchased"][pid] = {"type": ctx.ref.placements[pid]["type"], "present": pid in pa}
        st = dict(ctx.stats)
        st["shape_floor"] = res["shape_floor"]
        st["design_score_unscaled"] = res["design_score"]
        for k in CRITERIA:
            st["criterion_" + k] = res["criteria"][k]
            st["placeholder_" + k] = 1.0 if k in placeholders else 0.0
        st["config_frozen"] = 1.0 if res["config_status"] == "frozen" else 0.0
        st["t_grade_s"] = time.time() - t0
        res["statistics"] = _finite_stats(st)
        res["diagnostics"] = [str(d)[:300] for d in (list(ctx.diagnostics) + res["diagnostics"])][:MAX_DIAGNOSTICS]
        if ctx.gate is not None:
            res["gate"] = dict(ctx.gate)
        if ctx.alignment:
            res["alignment_residual_mm"] = float(ctx.alignment["residual_mm"])
    finally:
        if not keep_context:
            ctx.close()
    log("design_score %.4f (%.1f s)%s" % (res["design_score"], time.time() - t0,
                                          " placeholders: " + ",".join(placeholders) if placeholders else ""))
    return (res, ctx) if keep_context else res


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
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    from strutgrade import detenv
    detenv.early(int(os.environ.get("ARB_GRADER_CPUS", cfg.get("detenv", {}).get("grader_cpus", 4))))
    import strutgrade
    try:
        res = grade(a.extraction, a.reference, a.purchased_root, cfg)
    except strutgrade.TaskError as exc:  # the reference, the vendor files or the extraction: a task error (verify.py)
        res = {"ok": False, "task_error": str(exc), "error": traceback.format_exc()[-3000:]}
    except Exception:  # noqa: BLE001 - outside every grading item: a grader fault, never a silent 0 (task error)
        res = {"ok": False, "task_error": "the grader failed outside a grading item", "error": traceback.format_exc()[-3000:]}
    with open(a.output, "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=float)
    if res.get("ok"):
        print(json.dumps({k: res[k] for k in ("design_score", "criteria", "checks", "shape_floor")}, sort_keys=True),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
