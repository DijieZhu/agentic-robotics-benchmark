"""Maintainers' check of the mission's colliders rule (scan.colliders_kept, pxr only, no Kit).

    /opt/venv/bin/python3 -I -s tests/review/check_colliders_rule.py <scene.usda> [--out DIR] [--sweep]

On the given scene (the reference's, say) it expects every robot to pass, then builds edited copies, each a layer over
the scene or a rewritten copy beside it, and expects the vehicle to fail the rule in each: one of its library
colliders switched off (`physics:collisionEnabled`), made a trigger (`PhysxTriggerAPI`), shrunk by a scale op, and
its arm built from a copy of the library model in the submission instead of the library's file. With --sweep it also
references every catalog model plainly under a robot prim, as an agent would, and expects each to pass (no false
positives). Nothing here is part of the grade; the task's grader never runs it.
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
import scan  # noqa: E402
from pxr import Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

CATALOG = json.load(open(os.path.join(TESTS, "grader_config.json")))["catalog"]
ROOTS = tuple(os.path.realpath(r) for r in scan.ASSET_ROOTS)


def kept(path, robot):
    st = Usd.Stage.Open(path)
    rp = st.GetPrimAtPath(f"/World/Fleet/{robot}")
    decl = {}
    for key in ("airframe_model", "arm_model", "base_model"):
        vals = sorted({str(p.GetAttribute(f"arb:{key}").Get()) for p in Usd.PrimRange(rp)
                       if p.GetAttribute(f"arb:{key}") and p.GetAttribute(f"arb:{key}").Get()})
        decl[key] = vals[0] if len(vals) == 1 else None
    return scan.colliders_kept(st, f"/World/Fleet/{robot}", ROOTS, decl, CATALOG)


def over(scene, out_dir, name):
    p = os.path.join(out_dir, name)
    lay = Sdf.Layer.CreateNew(p)
    lay.subLayerPaths.append(scene)
    lay.defaultPrim = "World"
    st = Usd.Stage.Open(lay)
    st.SetEditTarget(lay)
    return p, st


def first_collider(st, root):
    return next(p for p in Usd.PrimRange(st.GetPrimAtPath(root), Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.CollisionAPI) and not p.IsInstanceProxy())


def variants(scene, out_dir):
    out = []
    p, st = over(scene, out_dir, "col_off.usda")
    st.OverridePrim(first_collider(st, "/World/Fleet/UGV").GetPath()).CreateAttribute(
        "physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    st.GetRootLayer().Save()
    out.append(("a library collider switched off", p, "n_disabled"))
    p, st = over(scene, out_dir, "col_trigger.usda")
    st.OverridePrim(first_collider(st, "/World/Fleet/UGV").GetPath()).AddAppliedSchema("PhysxTriggerAPI")
    st.GetRootLayer().Save()
    out.append(("a library collider made a trigger", p, "n_disabled"))
    p, st = over(scene, out_dir, "col_shrunk.usda")
    x = UsdGeom.Xformable(st.OverridePrim(first_collider(st, "/World/Fleet/UGV").GetPath()))
    ops = list(x.GetOrderedXformOps())
    op = x.AddScaleOp(opSuffix="shrink")
    op.Set((0.01, 0.01, 0.01))
    x.SetXformOpOrder(ops + [op])
    st.GetRootLayer().Save()
    out.append(("a library collider shrunk", p, "n_shrunk"))
    # the arm from a copy of its library model, beside the scene, its colliders switched off
    src = Usd.Stage.Open(scene)
    arcs = [a for q in Usd.PrimRange(src.GetPrimAtPath("/World/Fleet/UGV")) for a in scan._asset_arcs(q, ROOTS)]
    arm = (src.GetPrimAtPath("/World/Fleet/UGV").GetAttribute("arb:arm_model").Get()
           or next((str(q.GetAttribute("arb:arm_model").Get()) for q in Usd.PrimRange(src.GetPrimAtPath("/World/Fleet/UGV"))
                    if q.GetAttribute("arb:arm_model") and q.GetAttribute("arb:arm_model").Get()), None))
    asset = next((a for a, _ in arcs if arm and CATALOG["arms"][arm]["asset"] in a), None)
    if asset:
        d = os.path.join(out_dir, "copy")
        shutil.rmtree(d, ignore_errors=True)
        shutil.copytree(os.path.dirname(asset), os.path.join(d, "lib"))
        text = open(scene).read()
        for layer_dir in (os.path.dirname(os.path.abspath(scene)),):
            text = text.replace("@./", "@" + layer_dir + "/")  # (the copy sits elsewhere: the scene's own relative paths kept)
        text = text.replace("@" + asset + "@", "@./lib/" + os.path.basename(asset) + "@")
        p = os.path.join(d, "scene.usda")
        open(p, "w").write(text)
        st = Usd.Stage.Open(p)
        st.SetEditTarget(st.GetRootLayer())
        for q in list(Usd.PrimRange(st.GetPrimAtPath("/World/Fleet/UGV"), Usd.TraverseInstanceProxies())):
            if q.HasAPI(UsdPhysics.CollisionAPI) and not q.IsInstanceProxy() and "lib" in str(q.GetPrimStack()[-1].layer.realPath):
                q.CreateAttribute("physics:collisionEnabled", Sdf.ValueTypeNames.Bool).Set(False)
        st.GetRootLayer().Save()
        out.append(("the arm built from a copy of its library model", p, "models_not_referenced"))
    return out


def sweep():
    files = [f for r in ROOTS for f in glob.glob(r + "/**/*.usd*", recursive=True)]
    bad = []
    for table, key in (("airframes", "airframe_model"), ("bases", "base_model"), ("arms", "arm_model")):
        for name, e in CATALOG[table].items():
            cands = sorted(f for f in files if e["asset"] in f and "/configuration/" not in f and "/Props/" not in f)
            if not cands:
                bad.append(f"{table}/{name}: no library file for {e['asset']}")
                continue
            st = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageMetersPerUnit(st, 1.0)
            st.DefinePrim("/World/Fleet/R", "Xform")
            st.DefinePrim("/World/Fleet/R/M", "Xform").GetReferences().AddReference(min(cands, key=lambda f: (len(f), f)))
            k = scan.colliders_kept(st, "/World/Fleet/R", ROOTS, {key: name}, CATALOG)
            if not k["ok"] or not k["checked"]:
                bad.append(f"{table}/{name}: {k}")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()
    out_dir = a.out or tempfile.mkdtemp(prefix="colliders_rule_")
    os.makedirs(out_dir, exist_ok=True)
    problems = []
    for robot in ("UAV_1", "UAV_2", "UAV_3", "UGV"):
        k = kept(a.scene, robot)
        print(f"scene {robot}: ok={k['ok']} checked={k['checked']}")
        if not k["ok"]:
            problems.append(f"the scene's {robot} fails the rule: {k}")
    for what, path, field in variants(a.scene, out_dir):
        k = kept(path, "UGV")
        flagged = (not k["ok"]) and bool(k[field])
        print(f"{what}: ok={k['ok']} {field}={k[field]}")
        if not flagged:
            problems.append(f"{what}: not caught ({k})")
    if a.sweep:
        bad = sweep()
        print(f"catalog sweep: {len(bad)} false positive(s)")
        problems += bad
    print("ALL OK" if not problems else "PROBLEMS:\n  " + "\n  ".join(problems))
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
