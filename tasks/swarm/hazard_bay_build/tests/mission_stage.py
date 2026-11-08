"""The bay mission's stage: the real bay with the agent's four robots in it.

    info = mission_stage.build(agent_scene, bay_layer, facts, out_path)

The real bay (`hidden/bay_true.usda`, written once by the reference build with --bay-only) is the stage's sublayer:
`/World`, `/World/PhysicsScene`, `/World/Facility`. The agent's whole stage is mounted at `/Agent` by one reference to
its default prim (`/World` -> `/Agent`: every relationship among its own prims still holds, and none can reach the real
bay), and everything but its four robots is switched off:

- kept as composed: `/World/Fleet/UAV_1..3` and `UGV`, every body, joint, drive, collider, mass, material binding,
  camera and sensor prim under them, and the scopes outside the fleet that hold a material a robot binds or a
  collision group (their own physics prims switched off);
- switched off: the agent's facility, physics scenes, OmniGraph prims (anywhere), lights, hall and helpers, the other
  children of `/World/Fleet`, each flying robot's `Rope`, and every joint with an end outside its own robot (the hook,
  the ring, anything to the agent's bay or to another robot); `physics:simulationOwner` cleared, external forces off;
- confined: the agent's collision filtering reaches only its own stage (a collection that takes in the whole stage,
  `includeRoot`, takes in `/Agent` instead; a collision group that would still take in the real bay is off).

Each flying robot is then put on the real pad nearest to where the agent's scene puts it (the assignment with the
smallest total distance), the vehicle in the real parking bay, each by one translation of its root (headings and
heights as authored), its footprint centred as the layout check measures it (the robot's box without its `Rope`).
Each flying robot gets the plant's rope (`bay_rig`) from the agent's own hook anchor (the hook joint's body and local
position) to the bucket's ring facing its pad.

`dress(path, info, look)` then makes the stage what the robots' sensors see (control.md, The sensors), changing
nothing the simulation reads:

- the hall: the plant's hall (`sensors.look.hall`, the task's own `/opt/assets/arb/hall/bay_hall.usda`, visual only: it
  authors no physics schema) referenced at `/World/Look/Hall` at the identity (as the reference twin does), any physics
  schema under it removed in the mission layer (API schemas deleted, not switched off); its own 24 panel lights come
  with it, and the photographs' two (a dome light of 300, a distant light of 1800 at 4 degrees) are added under
  `/World/Look/Lights`;
- every light under the robots switched off (intensity 0; a light prim also made invisible): the real bay's lights alone
  light it;
- the plant's rope links bound to the real bay's black, with the material purpose `full` (render only);
- the pads' coplanar 1 mm strokes lifted apart by 0.1-0.3 mm (they have no colliders), so that they no longer z-fight;
- the gauge's needle given a transform op at the head of its op order (identity), which the probe turns with the
  pressure at each capture.

The physics of `/World/Facility` and of the robots is not touched (`check` proves it on the built stage).

pxr only, nothing executed; the grader's control probe builds it (as the probe account, into its own folder) before
Kit opens it.
"""

import itertools
import json
import math
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bay_rig  # noqa: E402
from bay_rig import MissionStageError  # noqa: E402,F401

UAVS, ROBOTS = ("UAV_1", "UAV_2", "UAV_3"), ("UAV_1", "UAV_2", "UAV_3", "UGV")
SRC, MOUNT = "/World", "/Agent"
FLEET_SRC, FLEET = "/World/Fleet", "/Agent/Fleet"
BUCKET = "/World/Facility/Bucket/Body"
ROPE_NAME = "BayRope"
PURPOSES = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]


def mounted(path):
    """An agent path (`/World/...`) as it appears in the mission stage (`/Agent/...`)."""
    p = str(path)
    return MOUNT + p[len(SRC):] if p == SRC or p.startswith(SRC + "/") else p


def _under(path, root):
    return path == root or path.startswith(root + "/")


def _joint_ends(j):
    jj = UsdPhysics.Joint(j)
    return [t.pathString for t in jj.GetBody0Rel().GetTargets()], [t.pathString for t in jj.GetBody1Rel().GetTargets()]


def _is_graph(p):
    return p.GetTypeName().startswith("OmniGraph")


def _is_physics(p):
    """A prim that would add something to the simulation (a body, a collider, a joint, a scene, a particle or
    deformable system, an attachment)."""
    if p.IsA(UsdPhysics.Scene) or p.IsA(UsdPhysics.Joint):
        return True
    if p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.ArticulationRootAPI):
        return True
    t = p.GetTypeName()
    if t.startswith("Physx") or (t.startswith("Physics") and t != "PhysicsCollisionGroup"):
        return True
    names = " ".join(p.GetAppliedSchemas())
    return any(k in names for k in ("Deformable", "Particle", "Attachment", "PhysxForceAPI"))


def survey(stage):
    """What the agent's stage holds, read as data: per robot its bodies, its rope's links and its hook; what has to
    be switched off; the scopes outside the fleet to keep."""
    out = dict(robots={}, drop=[], owners=[], forces=[], keep_scopes=[], groups=[], filtered=[])
    fleet = stage.GetPrimAtPath(FLEET_SRC)
    if not fleet or not fleet.IsValid():
        raise MissionStageError("the scene has no /World/Fleet")
    roots = {r: f"{FLEET_SRC}/{r}" for r in ROBOTS}
    for r, root in roots.items():
        if not stage.GetPrimAtPath(root).IsValid():
            raise MissionStageError(f"the scene has no {root}")
    joints = [p for p in stage.Traverse(Usd.TraverseInstanceProxies()) if p.IsA(UsdPhysics.Joint) and p.IsActive()]
    for r, root in roots.items():
        rope = f"{root}/Rope" if r in UAVS else None
        bodies, links = [], []
        for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            if p.HasAPI(UsdPhysics.RigidBodyAPI):
                (links if rope and _under(p.GetPath().pathString, rope) else bodies).append(p.GetPath().pathString)
        info = dict(root=root, bodies=bodies, links=links, hook=None)
        if not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) and not (rope and _under(p.GetPath().pathString, rope))
                   for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies())):
            raise MissionStageError(f"{root} has no articulation root outside its rope")
        if r in UAVS:
            linkset = set(links)
            for j in joints:  # the hook: the one joint with one end in the links and one in the robot (the scan's rule)
                b0, b1 = _joint_ends(j)
                ends = b0 + b1
                if not any(e in linkset for e in ends):
                    continue
                others = [e for e in ends if e not in linkset]
                robot_side = [e for e in others if _under(e, root) and e != BUCKET]
                if len(others) == 1 and robot_side:
                    side = "0" if robot_side[0] in b0 else "1"
                    lp = j.GetAttribute(f"physics:localPos{side}").Get()
                    info["hook"] = dict(joint=j.GetPath().pathString, body=robot_side[0],
                                        local=[float(v) for v in (lp if lp is not None else (0, 0, 0))])
                    break
            if info["hook"] is None:
                raise MissionStageError(f"{root} has no hook joint between a body of the robot and a link of its rope")
            if rope:
                out["drop"].append(rope)
        out["robots"][r] = info
    # joints: kept only inside one robot, between two of its own bodies (not its rope's links)
    for j in joints:
        jp = j.GetPath().pathString
        owner = next((r for r, root in roots.items() if _under(jp, root)), None)
        if owner is None:
            continue  # (outside the robots: switched off with its scope, or as a physics prim of a kept scope)
        root = roots[owner]
        rope = f"{root}/Rope" if owner in UAVS else None
        b0, b1 = _joint_ends(j)
        ok = bool(b0) and bool(b1) and all(_under(e, root) and not (rope and _under(e, rope)) for e in b0 + b1)
        if not ok and not (rope and _under(jp, rope)):
            out["drop"].append(jp)
    # scopes: the children of /World but the fleet, and the fleet's own children but the robots; kept only when they
    # hold a material a robot binds or a collision group (their physics prims and graphs switched off), else off
    def scope_of(path):
        if _under(path, FLEET_SRC):
            rest = path[len(FLEET_SRC) + 1:].split("/")[0] if path != FLEET_SRC else ""
            return None if not rest or rest in ROBOTS else f"{FLEET_SRC}/{rest}"
        if _under(path, SRC) and path != SRC:
            return SRC + "/" + path[len(SRC) + 1:].split("/")[0]
        return None
    scopes = set()
    for r, root in roots.items():
        for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
            path = p.GetPath().pathString
            if _is_graph(p) or p.IsA(UsdPhysics.Scene):
                out["drop"].append(path)
                continue
            own = p.GetRelationship("physics:simulationOwner")
            if own and own.GetTargets():
                out["owners"].append(path)
            if "PhysxForceAPI" in p.GetAppliedSchemas():
                out["forces"].append(path)
            if p.HasAPI(UsdPhysics.FilteredPairsAPI):
                out["filtered"].append(path)
            for rel in p.GetRelationships():
                if not rel.GetName().startswith("material:binding"):
                    continue
                for t in rel.GetTargets():
                    sc = scope_of(t.pathString)
                    if sc:
                        scopes.add(sc)
    for p in stage.Traverse(Usd.TraverseInstanceProxies()):
        if p.GetTypeName() == "PhysicsCollisionGroup":
            path = p.GetPath().pathString
            out["groups"].append(path)
            sc = scope_of(path)
            if sc:
                scopes.add(sc)
    out["keep_scopes"] = sorted(scopes)
    for c in fleet.GetChildren():
        if c.GetName() not in ROBOTS and c.GetPath().pathString not in scopes:
            out["drop"].append(c.GetPath().pathString)
    # inside kept scopes: every physics prim and graph off
    for top in out["keep_scopes"]:
        for p in Usd.PrimRange(stage.GetPrimAtPath(top), Usd.TraverseInstanceProxies()):
            if _is_graph(p) or _is_physics(p):
                out["drop"].append(p.GetPath().pathString)
    return out


def _editable(st, path):
    """The prim at `path`, made editable: an instance proxy cannot carry opinions, so its instance (and any instance
    above it) stops being instanced first."""
    p = st.GetPrimAtPath(path)
    for _ in range(16):
        if not p or not p.IsValid() or not p.IsInstanceProxy():
            return p
        a = p.GetParent()
        while a and a.IsValid() and not a.IsInstance():
            a = a.GetParent()
        if not a or not a.IsValid():
            return p
        a.SetInstanceable(False)
        p = st.GetPrimAtPath(path)
    return p


def set_active(st, path, active):
    p = _editable(st, path)
    if p and p.IsValid():
        p.SetActive(active)
        return True
    return False


def _box_except(st, path, skip):
    """The world box of a prim's children except those named in `skip` (probes.box_facts_except)."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), PURPOSES, useExtentsHint=False)
    lo, hi = None, None
    for c in st.GetPrimAtPath(path).GetChildren():
        if c.GetName() in skip:
            continue
        r = cache.ComputeWorldBound(c).ComputeAlignedRange()
        if r.IsEmpty():
            continue
        clo, chi = r.GetMin(), r.GetMax()
        lo = clo if lo is None else Gf.Vec3d(*(min(a, b) for a, b in zip(lo, clo)))
        hi = chi if hi is None else Gf.Vec3d(*(max(a, b) for a, b in zip(hi, chi)))
    return None if lo is None else ([float(v) for v in lo], [float(v) for v in hi])


def _collider_boxes(st, root):
    """The world boxes of a robot's colliders (every purpose, visible or not; its rope excluded)."""
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy,
                                                       UsdGeom.Tokens.guide], useExtentsHint=False, ignoreVisibility=True)
    out = []
    for p in Usd.PrimRange(st.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if p.GetName() == "Rope" and p.GetParent().GetPath().pathString == root:
            continue
        if p.HasAPI(UsdPhysics.CollisionAPI) and p.GetAttribute("physics:collisionEnabled").Get() is not False:
            r = cache.ComputeWorldBound(p).ComputeAlignedRange()
            if not r.IsEmpty():
                out.append(([float(v) for v in r.GetMin()], [float(v) for v in r.GetMax()]))
    return out


def _world(st, path):
    return UsdGeom.Xformable(st.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())


def nearest_pads(centres, pads):
    """The assignment of flying robots to pads with the smallest total distance (all permutations)."""
    names, best = sorted(pads), None
    robots = sorted(centres)
    for perm in itertools.permutations(names, len(robots)):
        d = sum(math.hypot(centres[r][0] - pads[p]["center"][0], centres[r][1] - pads[p]["center"][1]) for r, p in zip(robots, perm))
        if best is None or d < best[0]:
            best = (d, dict(zip(robots, perm)))
    return best[1]


def prepend_world_transform(st, path, world_row, tag="real_bay"):
    """Move a prim (and everything below it) by a world-space transform (a Gf row-vector matrix): one op at the head of
    its op order, the world move turned into its parent's frame; a reset of the transform stack is kept."""
    prim = st.GetPrimAtPath(path)
    if not prim.IsA(UsdGeom.Xformable):
        prim.SetTypeName("Xform")
    xf = UsdGeom.Xformable(prim)
    reset = xf.GetResetXformStack()
    parent = Gf.Matrix4d(1.0) if reset else UsdGeom.XformCache().GetParentToWorldTransform(prim)
    head = parent * world_row * parent.GetInverse()
    old = xf.GetOrderedXformOps()
    names = {o.GetOpName() for o in old}
    suffix = "arb_" + tag
    while f"xformOp:translate:{suffix}" in names or f"xformOp:transform:{suffix}" in names:
        suffix += "_"
    pure = all(abs(head[i][j] - (1.0 if i == j else 0.0)) < 1e-12 for i in range(3) for j in range(3))
    if pure:
        op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, suffix)
        op.Set(Gf.Vec3d(head.ExtractTranslation()))
    else:
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, suffix)
        op.Set(head)
    xf.SetXformOpOrder([op] + old, reset)


def build(agent_scene, bay_layer, facts, out_path):
    """Write the mission stage to `out_path`; returns what the grader needs to run it and record it."""
    src = Usd.Stage.Open(agent_scene)
    if src is None:
        raise MissionStageError("the scene does not open")
    sv = survey(src)
    if os.path.exists(out_path):
        os.remove(out_path)
    layer = Sdf.Layer.CreateNew(out_path)
    layer.subLayerPaths.append(bay_layer)
    layer.defaultPrim = "World"
    st = Usd.Stage.Open(layer)
    st.SetEditTarget(layer)
    UsdGeom.SetStageUpAxis(st, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(st, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(st, 1.0)
    agent = st.DefinePrim(MOUNT, "Xform")
    agent.GetReferences().AddReference(agent_scene)
    for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.ArticulationRootAPI, UsdPhysics.MassAPI):
        if agent.HasAPI(api):
            agent.RemoveAPI(api)
    keep = {mounted(s) for s in sv["keep_scopes"]} | {FLEET}
    for child in agent.GetChildren():
        if child.GetPath().pathString not in keep:
            child.SetActive(False)
    dropped = 0
    for p in sv["drop"]:
        dropped += bool(set_active(st, mounted(p), False))
    for p in sv["owners"]:
        q = _editable(st, mounted(p))
        if q and q.IsValid():
            q.GetRelationship("physics:simulationOwner").SetTargets([])
    for p in sv["forces"]:
        q = _editable(st, mounted(p))
        if q and q.IsValid():
            q.CreateAttribute("physxForce:forceEnabled", Sdf.ValueTypeNames.Bool).Set(False)
    for g in sv["groups"]:  # (a collision group that selects by expression could reach the real bay: off)
        q = st.GetPrimAtPath(mounted(g))
        if q and q.IsValid() and any(a.GetName().endswith("membershipExpression") and a.HasAuthoredValue() for a in q.GetAttributes()):
            set_active(st, mounted(g), False)
    cut = confine_collections(st)
    # place: each flying robot on its nearest real pad, the vehicle in the real parking bay (footprint centres)
    centres = {}
    for r in ROBOTS:
        root = f"{FLEET}/{r}"
        box = _box_except(st, root, ("Rope",))
        if box is None:
            t = _world(st, mounted(sv["robots"][r]["bodies"][0])).ExtractTranslation() if sv["robots"][r]["bodies"] else Gf.Vec3d(0, 0, 0)
            centres[r] = [float(t[0]), float(t[1])]
        else:
            centres[r] = [(box[0][0] + box[1][0]) / 2, (box[0][1] + box[1][1]) / 2]
    pads = nearest_pads({r: centres[r] for r in UAVS}, facts["pads"])
    targets = {r: facts["pads"][pads[r]]["center"] for r in UAVS}
    targets["UGV"] = facts["ugv_bay"]["center"]
    shifts = {}
    for r in ROBOTS:
        d = Gf.Vec3d(targets[r][0] - centres[r][0], targets[r][1] - centres[r][1], 0.0)
        prepend_world_transform(st, f"{FLEET}/{r}", Gf.Matrix4d(1.0).SetTranslate(d))
        shifts[r] = [round(float(v), 4) for v in d]
        _editable(st, f"{FLEET}/{r}").CreateAttribute("arb:disabled", Sdf.ValueTypeNames.Bool).Set(False)
    # the rig: the plant's ropes from the agent's hooks to the rings facing the pads
    rig, links_all, ropes = facts["rig"], [], {}
    bucket_w = _world(st, BUCKET)
    for r in UAVS:
        hk = sv["robots"][r]["hook"]
        body = mounted(hk["body"])
        hook_w = _world(st, body).Transform(Gf.Vec3d(*hk["local"]))
        ring = int(rig["ring_of_pad"][pads[r]])
        ring_local = rig["rings"][ring]
        ring_w = bucket_w.Transform(Gf.Vec3d(*ring_local))
        name = ROPE_NAME
        while st.GetPrimAtPath(f"{FLEET}/{r}/{name}").IsValid():
            name += "_"
        carriers = [mounted(b) for b in sv["robots"][r]["bodies"]]
        lay = bay_rig.lay_rope(st, f"{FLEET}/{r}/{name}", body, hk["local"], hook_w, BUCKET, ring_local, ring_w, carriers, rig["rope"],
                               carrier_boxes=_collider_boxes(st, f"{FLEET}/{r}"))
        links_all += lay["links"]
        ropes[r] = dict(root=f"{FLEET}/{r}/{name}", ring=ring, hook_body=body, hook_world=[round(float(v), 4) for v in hook_w],
                        chord=lay["chord"], miss=lay["miss"], lay=lay["lay"], filtered_links=lay["filtered"])
    bay_rig.filter_bucket(st, BUCKET, links_all)
    # the agent's collision filtering reaches only its own robots: never the plant's ropes
    rope_roots = [v["root"] for v in ropes.values()]
    for g in sv["groups"]:
        q = st.GetPrimAtPath(mounted(g))
        if q and q.IsValid() and q.IsActive():
            q = _editable(st, mounted(g))  # (a group inside an instance cannot take an opinion as it is)
            ex = Usd.CollectionAPI(q, "colliders").CreateExcludesRel()
            for rr in rope_roots:
                ex.AddTarget(rr)
    for p in sv["filtered"]:
        q = _editable(st, mounted(p))
        if q and q.IsValid():
            rel = UsdPhysics.FilteredPairsAPI(q).GetFilteredPairsRel()
            tg = [t for t in rel.GetTargets() if not any(_under(t.pathString, rr) for rr in rope_roots)]
            if len(tg) != len(rel.GetTargets()):
                rel.SetTargets(tg)
    cut += confine_groups(st)
    layer.Save()
    return dict(fleet_root=FLEET, rope_root={r: ropes[r]["root"] for r in UAVS}, pads=pads, centres=centres, shifts=shifts,
                ropes=ropes, kept_scopes=[mounted(s) for s in sv["keep_scopes"]], dropped=dropped,
                hooks={r: sv["robots"][r]["hook"] for r in UAVS}, collections_confined=cut)


LOOK, SENSORS = "/World/Look", "/World/Sensors"
NEEDLE_OP = "arb_gauge"


def _physics_names(p):
    """The physics API schemas applied to a prim (by name: UsdPhysics', PhysX's, deformables, particles)."""
    return [a for a in p.GetAppliedSchemas() if a.startswith(("Physics", "Physx")) or any(k in a for k in ("Deformable", "Particle", "Attachment"))]


def _physics_typed(p):
    t = p.GetTypeName()
    return t.startswith(("Physics", "Physx")) or _is_graph(p)


def strip_physics(st, root):
    """Every physics schema under `root` removed in the edit target, not switched off: each applied physics API schema
    deleted from the prim's list, each typed physics prim and graph deactivated (an instance holding one is made
    editable first). Returns (API schemas removed, prims deactivated)."""
    n_api = n_typed = 0
    for p in list(Usd.PrimRange(st.GetPrimAtPath(root), Usd.TraverseInstanceProxies())):
        if not p.IsValid():
            continue
        names, typed = _physics_names(p), _physics_typed(p)
        if not names and not typed:
            continue
        q = _editable(st, p.GetPath().pathString)
        if typed:
            q.SetActive(False)
            n_typed += 1
            continue
        for a in names:
            q.RemoveAppliedSchema(a)
            n_api += 1
    return n_api, n_typed


def dress(path, info, look):
    """Make the built mission stage at `path` what the robots' sensors see (the module's docstring); returns what it
    did. `look`: the grader's `sensors.look` (the hall's asset, the lights, the rope material, the pads' lift, the
    gauge)."""
    layer = Sdf.Layer.FindOrOpen(path)
    st = Usd.Stage.Open(layer)
    st.SetEditTarget(layer)
    out = {}
    # the hall, physics-free
    UsdGeom.Xform.Define(st, LOOK)
    if look.get("hall"):  # (a maintainer's bench may leave the hall out: review/run_mission.py --probe-set)
        hall = UsdGeom.Xform.Define(st, f"{LOOK}/Hall").GetPrim()
        hall.GetReferences().AddReference(look["hall"])
        out["hall_api_removed"], out["hall_prims_off"] = strip_physics(st, f"{LOOK}/Hall")
    # the photographs' two lights
    for L in look.get("lights") or []:
        lp = f"{LOOK}/Lights/{L['name']}"
        if L["type"] == "DomeLight":
            lt = UsdLux.DomeLight.Define(st, lp)
        else:
            lt = UsdLux.DistantLight.Define(st, lp)
            lt.CreateAngleAttr(float(L.get("angle", 0.53)))
            rot = Gf.Rotation(Gf.Vec3d(0, 0, 1), float(L.get("rotate_z_deg", 0.0))) * Gf.Rotation(Gf.Vec3d(1, 0, 0), float(L.get("rotate_x_deg", 0.0)))
            UsdGeom.Xformable(lt).AddOrientOp().Set(Gf.Quatf(rot.GetQuat()))
        lt.CreateIntensityAttr(float(L["intensity"]))
    # no light under the robots
    off = []
    for r in ROBOTS:
        root = st.GetPrimAtPath(f"{FLEET}/{r}")
        for p in list(Usd.PrimRange(root, Usd.TraverseInstanceProxies())) if root.IsValid() else []:
            if not p.HasAPI(UsdLux.LightAPI):
                continue
            q = _editable(st, p.GetPath().pathString)
            UsdLux.LightAPI(q).CreateIntensityAttr(0.0)
            if q.IsA(UsdLux.BoundableLightBase) or q.IsA(UsdLux.NonboundableLightBase):
                UsdGeom.Imageable(q).MakeInvisible()
            off.append(q.GetPath().pathString)
    out["robot_lights_off"] = off
    # the plant's ropes black, for rendering only
    mat = UsdShade.Material(st.GetPrimAtPath(look["rope_material"]))
    bound = 0
    for rr in (info.get("rope_root") or {}).values():
        for p in Usd.PrimRange(st.GetPrimAtPath(rr)):
            if p.IsA(UsdGeom.Gprim):
                UsdShade.MaterialBindingAPI.Apply(p).Bind(mat, UsdShade.Tokens.fallbackStrength, UsdShade.Tokens.full)
                bound += 1
    out["rope_geoms_bound"] = bound
    # the pads' strokes lifted apart (visual prims: no collider under the pads)
    lo, hi = [float(v) for v in look.get("pads_lift_m", (0.0001, 0.0003))]
    lifted = 0
    pads = st.GetPrimAtPath(look.get("pads", "/World/Facility/Pads"))
    for pad in (pads.GetChildren() if pads.IsValid() else []):
        strokes = [p for p in Usd.PrimRange(pad) if p.IsA(UsdGeom.Gprim) and not p.HasAPI(UsdPhysics.CollisionAPI)]
        for k, p in enumerate(strokes):
            lift = lo + (hi - lo) * (k / (len(strokes) - 1) if len(strokes) > 1 else 0.0)
            op = next((o for o in UsdGeom.Xformable(p).GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
            if op is None:
                op = UsdGeom.Xformable(p).AddTranslateOp(opSuffix="arb_lift")
                op.Set(Gf.Vec3d(0.0, 0.0, lift))
            else:
                v = op.Get()
                op.Set(type(v)(v[0], v[1], v[2] + lift))
            lifted += 1
    out["pad_strokes_lifted"] = lifted
    # the gauge's needle: a head op the probe turns
    needle = st.GetPrimAtPath(look["needle"])
    xf = UsdGeom.Xformable(needle)
    old_ops = xf.GetOrderedXformOps()
    hop = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, NEEDLE_OP)
    hop.Set(Gf.Matrix4d(1.0))
    xf.SetXformOpOrder([hop] + old_ops, xf.GetResetXformStack())
    m = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).RemoveScaleShear()
    z = m.TransformDir(Gf.Vec3d(0, 0, 1))
    rest = math.degrees(math.atan2(z[0], z[2]))  # (from vertical, clockwise seen from the bay: toward +x)
    dial = UsdGeom.BBoxCache(Usd.TimeCode.Default(), PURPOSES, useExtentsHint=False).ComputeWorldBound(st.GetPrimAtPath(look["dial"])).ComputeAlignedRange()
    centre = [(a + b) / 2 for a, b in zip(dial.GetMin(), dial.GetMax())]
    out["needle"] = dict(path=look["needle"], op=f"xformOp:transform:{NEEDLE_OP}", rest_deg=round(rest, 4), axis_point=[round(float(v), 6) for v in centre])
    if look.get("deinstance_robots"):
        n = 0
        for r in ROBOTS:
            for _ in range(16):
                inst = [p for p in Usd.PrimRange(st.GetPrimAtPath(f"{FLEET}/{r}"), Usd.TraverseInstanceProxies()) if p.IsInstance() and not p.IsInstanceProxy()]
                if not inst:
                    break
                for p in inst:
                    p.SetInstanceable(False)
                    n += 1
        out["robots_deinstanced"] = n
    layer.Save()
    return out


def confine_collections(st):
    """A collection on the agent's prims that takes in the whole stage (`includeRoot`; a collision group's own or one
    it includes) would take in the real bay too, and PhysX honours it: the agent's collision filtering would reach the
    real furniture. Each such collection takes in the agent's own stage (`/Agent`) instead. Returns what was changed."""
    out = []
    for p in list(Usd.PrimRange(st.GetPrimAtPath(MOUNT), Usd.TraverseInstanceProxies())):
        for col in Usd.CollectionAPI.GetAllCollections(p):
            if col.GetIncludeRootAttr().Get():
                path, name = p.GetPath().pathString, col.GetName()
                q = _editable(st, path)
                col = Usd.CollectionAPI(q, name)
                col.CreateIncludeRootAttr().Set(False)
                col.CreateIncludesRel().AddTarget(MOUNT)
                out.append(f"{path}.collection:{name}")
    return out


def confine_groups(st):
    """The last word on the agent's collision groups: one whose membership still takes in any collider of the real bay
    (`/World`) is switched off. Returns the groups switched off."""
    real = [p.GetPath().pathString for p in st.Traverse(Usd.TraverseInstanceProxies())
            if _under(p.GetPath().pathString, SRC) and p.HasAPI(UsdPhysics.CollisionAPI)] + [SRC]
    out = []
    for p in list(st.Traverse(Usd.TraverseInstanceProxies())):
        if p.GetTypeName() != "PhysicsCollisionGroup" or not _under(p.GetPath().pathString, MOUNT):
            continue
        q = Usd.CollectionAPI(p, "colliders").ComputeMembershipQuery()
        if any(q.IsPathIncluded(r) for r in real):
            set_active(st, p.GetPath().pathString, False)
            out.append(f"{p.GetPath().pathString} (off)")
    return out


def check(path, info=None):
    """The built stage as the control probe will see it (the maintainers' checks): which robots are active, the physics
    scenes, the graphs, every active joint with an end outside its own robot (the plant's ropes may end on the bucket),
    and each rope's links, hook and ring."""
    st = Usd.Stage.Open(path)
    out = dict(robots={}, scenes=[], graphs=0, foreign_joints=[], ropes={})
    roots = {r: f"{FLEET}/{r}" for r in ROBOTS}
    for r in ROBOTS:
        p = st.GetPrimAtPath(roots[r])
        out["robots"][r] = bool(p and p.IsValid() and p.IsActive())
    for p in st.Traverse(Usd.TraverseInstanceProxies()):
        if p.IsA(UsdPhysics.Scene):
            out["scenes"].append(p.GetPath().pathString)
        if _is_graph(p):
            out["graphs"] += 1
        if p.IsA(UsdPhysics.Joint) and p.IsActive() and _under(p.GetPath().pathString, MOUNT):
            jp = p.GetPath().pathString
            b0, b1 = _joint_ends(p)
            owner = next((r for r, root in roots.items() if _under(jp, root)), None)
            in_rope = owner is not None and f"/{ROPE_NAME}" in jp
            ok = owner is not None and b0 and b1 and all(_under(e, roots[owner]) or (in_rope and e == BUCKET) for e in b0 + b1)
            if not ok:
                out["foreign_joints"].append(dict(joint=jp, ends=b0 + b1))
    for r in UAVS:
        root = (info or {}).get("rope_root", {}).get(r) or f"{roots[r]}/{ROPE_NAME}"
        rp = st.GetPrimAtPath(root)
        links = [q.GetPath().pathString for q in Usd.PrimRange(rp) if q.HasAPI(UsdPhysics.RigidBodyAPI)] if rp.IsValid() else []
        hook = ring = None
        for q in (Usd.PrimRange(rp) if rp.IsValid() else []):
            if q.IsA(UsdPhysics.Joint):
                b0, b1 = _joint_ends(q)
                if BUCKET in b0 + b1:
                    ring = b0 + b1
                elif b0 and b0[0] not in links:
                    hook = b0 + b1
        out["ropes"][r] = dict(root=root, links=len(links), hook=hook, ring=ring)
    out["bodies_active"] = sum(1 for p in st.Traverse(Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.RigidBodyAPI))
    real = [p.GetPath().pathString for p in st.Traverse(Usd.TraverseInstanceProxies())
            if _under(p.GetPath().pathString, SRC) and p.HasAPI(UsdPhysics.CollisionAPI)]
    out["groups_reaching_real_bay"] = [p.GetPath().pathString for p in st.Traverse(Usd.TraverseInstanceProxies())
                                       if p.GetTypeName() == "PhysicsCollisionGroup" and _under(p.GetPath().pathString, MOUNT)
                                       and any(Usd.CollectionAPI(p, "colliders").ComputeMembershipQuery().IsPathIncluded(r) for r in real)]
    # the dress (when there is one): no physics and no graph under the look and the sensors, one physics scene, no
    # light under the robots, the plant's ropes bound for rendering
    look_phys = []
    for top in (LOOK, SENSORS):
        tp = st.GetPrimAtPath(top)
        for p in (Usd.PrimRange(tp, Usd.TraverseInstanceProxies()) if tp.IsValid() else []):
            if _physics_names(p) or _physics_typed(p):
                look_phys.append(p.GetPath().pathString)
    out["dressed"] = st.GetPrimAtPath(f"{LOOK}/Hall").IsValid()
    out["look_physics"] = look_phys[:20]
    out["n_look_physics"] = len(look_phys)
    out["one_physics_scene"] = len(out["scenes"]) == 1
    lit = []
    for r in ROBOTS:
        rp = st.GetPrimAtPath(f"{FLEET}/{r}")
        for p in (Usd.PrimRange(rp, Usd.TraverseInstanceProxies()) if rp.IsValid() else []):
            if p.HasAPI(UsdLux.LightAPI) and (UsdLux.LightAPI(p).GetIntensityAttr().Get() or 0.0) > 0.0:
                lit.append(p.GetPath().pathString)
    out["robot_lights_on"] = lit
    unbound = {}
    for r in UAVS:
        root = (info or {}).get("rope_root", {}).get(r) or f"{roots[r]}/{ROPE_NAME}"
        rp = st.GetPrimAtPath(root)
        geoms = [p for p in (Usd.PrimRange(rp) if rp.IsValid() else []) if p.IsA(UsdGeom.Gprim)]
        unbound[r] = sum(1 for p in geoms if not UsdShade.MaterialBindingAPI(p).GetDirectBindingRel(UsdShade.Tokens.full).GetTargets())
    out["rope_geoms_unbound"] = unbound
    return out


if __name__ == "__main__":  # python3 mission_stage.py <scene> <bay_true.usda> <bay_facts.json> <out.usda>
    info = build(sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4])
    print(json.dumps(info, indent=1))
    print(json.dumps(check(sys.argv[4], info), indent=1))
