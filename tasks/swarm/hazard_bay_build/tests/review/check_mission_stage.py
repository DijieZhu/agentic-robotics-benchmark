"""Maintainers' check of the bay mission's stage builder (pxr only, no Kit): T1 of the 2026-11-03 hardening plan.

    /opt/venv/bin/python3 -I -s tests/review/check_mission_stage.py <scene.usda> [<scene.usda> ...] [--synthetic] [--out DIR]

For each scene it builds the mission stage (tests/mission_stage.py on tests/hidden/bay_true.usda) and checks: the four
robots active; three plant ropes, each a chain hooked to the scene's own hook body and ringed to the real bucket; no
active joint with an end outside its robot; one physics scene; no OmniGraph prim; no collision group of the scene's
that takes in a collider of the real bay. With --synthetic it also builds edited copies of the first scene and checks
that the builder copes: a hook joint outside `Rope`, a physics material outside the fleet bound to the vehicle's wheels,
a joint from the vehicle to the scene's floor, two collision groups that take in the whole stage (`includeRoot` on
the group, or on a collection the group includes) with the vehicle's arm filtered against them, and a collision group
inside an instanceable reference (the builder must edit it without failing); and, for the sensors' dress
(`mission_stage.dress`), a fleet whose robots' meshes are instanceable and a light under a robot (switched off).
Every stage is also dressed and checked: the physics of the stage before the dress and after it are the same (every
prim with a physics schema, its schemas, its physics attributes, its bound physics material); no physics schema and no
graph under `/World/Look` or `/World/Sensors`; one physics scene; no light under the robots; the plant's ropes bound
for rendering. The hall's asset must resolve: run it in the task image (`/opt/assets`).
Nothing here is part of the grade; the task's grader never runs it.
"""

import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.dirname(HERE)
sys.path.insert(0, TESTS)
import mission_stage  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade  # noqa: E402

FACTS = json.load(open(os.path.join(TESTS, "hidden", "bay_facts.json")))
BAY = os.path.join(TESTS, "hidden", "bay_true.usda")
LOOK = json.load(open(os.path.join(TESTS, "grader_config.json")))["sensors"]["look"]


def physics_print(path):
    """Everything of a stage the simulation reads: per prim with a physics schema (or a typed physics prim) its
    schemas, its physics attributes' values and its bound physics material."""
    st = Usd.Stage.Open(path)
    out = {}
    for p in st.Traverse(Usd.TraverseInstanceProxies()):
        names = sorted(mission_stage._physics_names(p))
        typed = p.GetTypeName() if mission_stage._physics_typed(p) else None
        if not names and not typed:
            continue
        attrs = {a.GetName(): str(a.Get()) for a in p.GetAttributes() if a.GetName().startswith(("physics:", "physx")) and a.HasAuthoredValue()}
        rels = {r.GetName(): [str(t) for t in r.GetTargets()] for r in p.GetRelationships() if r.GetName().startswith("physics:")}
        mat = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial("physics")[0]
        out[p.GetPath().pathString] = dict(active=p.IsActive(), names=names, typed=typed, attrs=attrs, rels=rels, material=str(mat.GetPath()) if mat else None)
    return out


def verdict(scene, out_dir, expect_pads=None, extra=None):
    out = os.path.join(out_dir, os.path.basename(os.path.dirname(scene)) + "_" + os.path.basename(scene).replace(".", "_") + "_mission.usda")
    info = mission_stage.build(scene, BAY, FACTS, out)
    chk = mission_stage.check(out, info)
    problems = []
    if not all(chk["robots"].values()):
        problems.append(f"robots not all active: {chk['robots']}")
    if chk["scenes"] != ["/World/PhysicsScene"]:
        problems.append(f"physics scenes: {chk['scenes']}")
    if chk["graphs"]:
        problems.append(f"{chk['graphs']} OmniGraph prims active")
    if chk["foreign_joints"]:
        problems.append(f"joints with an end outside their robot: {chk['foreign_joints'][:4]}")
    if chk["groups_reaching_real_bay"]:
        problems.append(f"collision groups that take in the real bay: {chk['groups_reaching_real_bay'][:4]}")
    for r, rope in chk["ropes"].items():
        want_hook = info["ropes"][r]["hook_body"]
        if rope["links"] != FACTS["rig"]["rope"]["links"]:
            problems.append(f"{r}: {rope['links']} links")
        if not rope["hook"] or rope["hook"][0] != want_hook:
            problems.append(f"{r}: hook joint {rope['hook']} (expected on {want_hook})")
        if not rope["ring"] or mission_stage.BUCKET not in rope["ring"]:
            problems.append(f"{r}: ring joint {rope['ring']}")
        if abs(info["ropes"][r]["miss"]) > 0.01:
            problems.append(f"{r}: the rope's last link misses its ring by {info['ropes'][r]['miss']} m")
    if expect_pads and info["pads"] != expect_pads:
        problems.append(f"pads {info['pads']} (expected {expect_pads})")
    for fn in extra or []:
        problems += fn(out, info)
    before = physics_print(out)  # the dress: nothing the simulation reads changes
    dressed = mission_stage.dress(out, info, LOOK)
    after = physics_print(out)
    if before != after:
        changed = sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))
        problems.append(f"the dress changed the physics of {len(changed)} prims: {changed[:4]}")
    chk = mission_stage.check(out, info)
    if chk["n_look_physics"]:
        problems.append(f"physics or graphs under the look or the sensors: {chk['look_physics'][:4]}")
    if not chk["one_physics_scene"]:
        problems.append(f"physics scenes after the dress: {chk['scenes']}")
    if chk["robot_lights_on"]:
        problems.append(f"lights on under the robots: {chk['robot_lights_on'][:4]}")
    if any(chk["rope_geoms_unbound"].values()):
        problems.append(f"rope links not bound for rendering: {chk['rope_geoms_unbound']}")
    # the hall resolved (the plant's hall authors no physics, so the dress has nothing of it to remove: count its prims)
    hall = Usd.Stage.Open(out).GetPrimAtPath(f"{mission_stage.LOOK}/Hall")
    n_hall = sum(1 for _ in Usd.PrimRange(hall, Usd.TraverseInstanceProxies())) if hall.IsValid() else 0
    if n_hall < 100:
        problems.append(f"the hall has {n_hall} prims (its asset did not resolve?)")
    print(json.dumps(dict(scene=scene, pads=info["pads"], shifts=info["shifts"], kept=info["kept_scopes"], dropped=info["dropped"],
                          ropes={r: {k: v for k, v in x.items() if k in ("ring", "chord", "lay", "miss", "filtered_links")} for r, x in info["ropes"].items()},
                          bodies_active=chk["bodies_active"], dress={k: v for k, v in dressed.items() if k != "robot_lights_off"},
                          robot_lights_off=len(dressed["robot_lights_off"]), problems=problems)))
    return not problems


def synthetic(scene, out_dir):
    """Three edited copies of `scene` (each a layer over it)."""
    cases = []
    # (a) the hook joint outside Rope
    p = os.path.join(out_dir, "syn_hook_outside.usda")
    lay = Sdf.Layer.CreateNew(p)
    lay.subLayerPaths.append(scene)
    lay.defaultPrim = "World"
    st = Usd.Stage.Open(lay)
    src = Usd.Stage.Open(scene)
    hook = None
    for j in src.Traverse():
        if j.IsA(UsdPhysics.Joint) and j.GetPath().pathString.startswith("/World/Fleet/UAV_1/Rope/"):
            b0 = [str(t) for t in UsdPhysics.Joint(j).GetBody0Rel().GetTargets()]
            if b0 and "/Rope/" not in b0[0]:
                hook = j
    assert hook is not None, "no hook joint found in UAV_1/Rope"
    st.GetPrimAtPath(hook.GetPath()).SetActive(False)
    nj = UsdPhysics.SphericalJoint.Define(st, "/World/Fleet/UAV_1/HookOutside")
    jj = UsdPhysics.Joint(hook)
    nj.CreateBody0Rel().SetTargets(jj.GetBody0Rel().GetTargets())
    nj.CreateBody1Rel().SetTargets(jj.GetBody1Rel().GetTargets())
    nj.CreateLocalPos0Attr(jj.GetLocalPos0Attr().Get())
    nj.CreateLocalPos1Attr(jj.GetLocalPos1Attr().Get())
    nj.CreateExcludeFromArticulationAttr(True)
    lay.Save()

    def no_outside(out, info):
        s2 = Usd.Stage.Open(out)
        pr = s2.GetPrimAtPath("/Agent/Fleet/UAV_1/HookOutside")
        return [] if not (pr and pr.IsValid() and pr.IsActive()) else ["the hook joint outside Rope is still active"]
    cases.append((p, [no_outside]))
    # (b) a physics material outside the fleet, bound to the vehicle's wheel colliders
    p = os.path.join(out_dir, "syn_material_outside.usda")
    lay = Sdf.Layer.CreateNew(p)
    lay.subLayerPaths.append(scene)
    lay.defaultPrim = "World"
    st = Usd.Stage.Open(lay)
    mat = UsdShade.Material.Define(st, "/World/PhysMats/Rubber")
    m = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    m.CreateStaticFrictionAttr(1.0)
    m.CreateDynamicFrictionAttr(0.9)
    wheels = []
    for q in Usd.PrimRange(st.GetPrimAtPath("/World/Fleet/UGV"), Usd.TraverseInstanceProxies()):
        if q.HasAPI(UsdPhysics.CollisionAPI) and "wheel" in q.GetPath().pathString and not q.IsInstanceProxy():
            UsdShade.MaterialBindingAPI.Apply(q).Bind(mat, UsdShade.Tokens.strongerThanDescendants, "physics")
            wheels.append(q.GetPath().pathString)
    lay.Save()

    def material_kept(out, info, wheels=tuple(wheels)):
        s2 = Usd.Stage.Open(out)
        probs = []
        mp = s2.GetPrimAtPath("/Agent/PhysMats/Rubber")
        if not (mp and mp.IsValid() and mp.IsActive()):
            probs.append("the robot's physics material outside the fleet is gone")
        for w in wheels[:1]:
            q = s2.GetPrimAtPath(mission_stage.mounted(w))
            b = UsdShade.MaterialBindingAPI(q).ComputeBoundMaterial("physics")[0] if q and q.IsValid() else None
            if not b or b.GetPath().pathString != "/Agent/PhysMats/Rubber":
                probs.append(f"{w}: bound to {b and b.GetPath()}")
        return probs
    cases.append((p, [material_kept]))
    # (c) a joint from the vehicle to the scene's floor
    p = os.path.join(out_dir, "syn_vehicle_tethered.usda")
    lay = Sdf.Layer.CreateNew(p)
    lay.subLayerPaths.append(scene)
    lay.defaultPrim = "World"
    st = Usd.Stage.Open(lay)
    tj = UsdPhysics.FixedJoint.Define(st, "/World/Fleet/UGV/Tether")
    tj.CreateBody0Rel().SetTargets([Sdf.Path("/World/Facility/Floor")])
    art = next(q for q in Usd.PrimRange(st.GetPrimAtPath("/World/Fleet/UGV")) if q.HasAPI(UsdPhysics.ArticulationRootAPI))
    tj.CreateBody1Rel().SetTargets([art.GetPath()])
    tj.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    lay.Save()

    def no_tether(out, info):
        s2 = Usd.Stage.Open(out)
        pr = s2.GetPrimAtPath("/Agent/Fleet/UGV/Tether")
        return [] if not (pr and pr.IsValid() and pr.IsActive()) else ["the vehicle's tether to the floor is still active"]
    cases.append((p, [no_tether]))
    # (d, e) collision groups that take in the whole stage: on the group itself, or through a collection it includes
    for name, nested in (("syn_group_root.usda", False), ("syn_group_root_nested.usda", True)):
        p = os.path.join(out_dir, name)
        lay = Sdf.Layer.CreateNew(p)
        lay.subLayerPaths.append(scene)
        lay.defaultPrim = "World"
        st = Usd.Stage.Open(lay)
        everything = UsdPhysics.CollisionGroup.Define(st, "/World/Fleet/Everything")
        if nested:
            holder = st.OverridePrim("/World/Fleet/UGV")
            Usd.CollectionAPI.Apply(holder, "everything").CreateIncludeRootAttr(True)
            everything.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/Fleet/UGV.collection:everything")
        else:
            everything.GetCollidersCollectionAPI().CreateIncludeRootAttr(True)
        arm = UsdPhysics.CollisionGroup.Define(st, "/World/Fleet/ArmGroup")
        arm.GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/World/Fleet/UGV")
        arm.CreateFilteredGroupsRel().AddTarget("/World/Fleet/Everything")
        lay.Save()

        def confined(out, info):
            s2 = Usd.Stage.Open(out)
            g = s2.GetPrimAtPath("/Agent/Fleet/Everything")
            probs = [] if info.get("collections_confined") else ["no collection was confined"]
            if not (g and g.IsValid() and g.IsActive()):
                probs.append("the group was switched off instead of confined")
            elif not Usd.CollectionAPI(g, "colliders").ComputeMembershipQuery().IsPathIncluded("/Agent/Fleet/UGV"):
                probs.append("the confined group lost the scene's own robots")
            return probs
        cases.append((p, [confined]))
    # (f) a collision group inside an instanceable reference
    lib = os.path.join(out_dir, "syn_groups_lib.usda")
    lst = Usd.Stage.CreateNew(lib)
    lst.SetDefaultPrim(lst.DefinePrim("/Groups", "Scope"))
    UsdPhysics.CollisionGroup.Define(lst, "/Groups/G").GetCollidersCollectionAPI().CreateIncludesRel().AddTarget("/Groups")
    lst.GetRootLayer().Save()
    p = os.path.join(out_dir, "syn_group_instanced.usda")
    lay = Sdf.Layer.CreateNew(p)
    lay.subLayerPaths.append(scene)
    lay.defaultPrim = "World"
    st = Usd.Stage.Open(lay)
    holder = st.DefinePrim("/World/Fleet/Groups", "Scope")
    holder.GetReferences().AddReference(lib)
    holder.SetInstanceable(True)
    lay.Save()
    cases.append((p, []))
    # (g) instanceable meshes in the robots, (h) a light under a robot (the dress switches it off)
    p = os.path.join(out_dir, "syn_instanced_lit.usda")
    lay = Sdf.Layer.CreateNew(p)
    lay.subLayerPaths.append(scene)
    lay.defaultPrim = "World"
    st = Usd.Stage.Open(lay)
    made = 0
    for q in list(Usd.PrimRange(st.GetPrimAtPath("/World/Fleet"))):
        if q.IsA(UsdGeom.Mesh) and not q.HasAPI(UsdPhysics.CollisionAPI) and q.GetParent().GetName() not in ("Rope",) and made < 12:
            holder = q.GetParent()
            if holder.HasAPI(UsdPhysics.RigidBodyAPI) or holder.IsInstanceable():
                continue
            holder.SetInstanceable(True)
            made += 1
    lamp = UsdLux.SphereLight.Define(st, "/World/Fleet/UGV/Lamp")
    lamp.CreateIntensityAttr(5000.0)
    lay.Save()

    def lamp_off(out, info, made=made):
        s2 = Usd.Stage.Open(out)
        probs = [] if made else ["no instanceable mesh holder could be made"]
        lp = s2.GetPrimAtPath("/Agent/Fleet/UGV/Lamp")
        if not lp.IsValid():
            probs.append("the robot's lamp is gone")
        return probs
    cases.append((p, [lamp_off]))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenes", nargs="+")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--expect-pads", default=None, help="JSON {robot: pad} for the first scene")
    a = ap.parse_args()
    out_dir = a.out or tempfile.mkdtemp(prefix="mission_stage_")
    os.makedirs(out_dir, exist_ok=True)
    ok = True
    for i, sc in enumerate(a.scenes):
        ok &= verdict(sc, out_dir, json.loads(a.expect_pads) if (a.expect_pads and i == 0) else None)
    if a.synthetic:
        for sc, extra in synthetic(a.scenes[0], out_dir):
            ok &= verdict(sc, out_dir, extra=extra)
    print("ALL OK" if ok else "PROBLEMS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
