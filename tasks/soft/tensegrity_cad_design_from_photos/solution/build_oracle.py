"""Build the oracle deliverable: the native FreeCAD document of the reference strut, made from the original parts.

Runs under FreeCAD 1.1.4's headless interpreter (the task image's /opt/freecad/usr/bin/freecadcmd), from solve.sh:

    freecadcmd -c "import sys; sys.argv = ['build_oracle.py', REF_DIR, PURCHASED_DIR, OUT_FCSTD(, MODE)]; exec(open(PATH).read())"

REF_DIR = solution/reference: oracle.json (tags, families and instance poses; the purchased instances with their vendor
STEP paths relative to PURCHASED_DIR, sha256 and local_to_strut; the joint table) and parts/<family>.ply (the original
manufactured parts as triangle meshes, byte-identical to the grader's tests/reference/parts). PURCHASED_DIR = the vendor
files staged in the agent home (~/purchased/<PN>/<PN>.step). Nothing of the robot is modelled here: every manufactured
part is its original mesh turned triangle for triangle into a faceted solid (Mesh.Mesh -> Part.Shape.makeShapeFromMesh,
tolerance 0.001 mm -> Part.Shell -> Part.Solid, reversed when its volume is negative; no coplanar faces merged, no
vertex moved), and every purchased part is the vendor's STEP imported unchanged. The document holds:
  - one PartDesign Body per tag, labelled with the tag; its BaseFeature is the faceted solid of the tag's family file in
    the file's own coordinates (identity placement) and the Body's placement is the instance pose;
  - every purchased instance at its local_to_strut: MODE "compound" (default) one Part::Feature holding the STEP's
    shape; "insert" FreeCAD's own STEP import (Import.insert) into an App::Part per instance; "link" one Part::Feature
    per type, outside the Assembly, linked once per instance;
  - one Assembly: an App::Link per component, a Grounded joint on the grounded tag (P01), a detached Fixed joint from it
    to every other component (frames at the strut origin; two subreference slots), and for each winch
    a Revolute joint without limits whose two frames are the motor's local frame (origin on the output-shaft axis in
    the gearbox front face, +z along the shaft).
When the GUI is up, the joints get FreeCAD's own view providers (headless there is no view object).
The Assembly is recomputed and solved, every component's placement compared with the one it was given (nothing may move
more than 1e-6 mm or 1e-9 rad), the document saved, reopened, recomputed, solved again and checked again. Every input
file is checked against the sha256 that oracle.json records. Prints ORACLE-BUILD <json>.
"""
import hashlib
import json
import math
import os
import sys
import time

import FreeCAD as App
import JointObject
import Mesh
import Part

FIXED = list(JointObject.JointTypes).index("Fixed")
REVOLUTE = list(JointObject.JointTypes).index("Revolute")
MOVE_TOL_MM = 1e-6
MOVE_TOL_RAD = 1e-9
VOLUME_TOL_REL = 1e-6


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def placement(m):
    return App.Placement(App.Matrix(*[float(v) for row in m for v in row]))


def deviation(links, intended):
    worst = (0.0, 0.0, None)
    for key, lk in links.items():
        p, q = lk.Placement, intended[key]
        d = (p.Base - q.Base).Length
        a = abs(q.Rotation.inverted().multiply(p.Rotation).Angle)
        a = min(a, abs(2.0 * math.pi - a))
        if (d, a) > worst[:2]:
            worst = (d, a, key)
    return worst


def faceted_solid(path):
    """The mesh file's triangles as one faceted solid (one planar face per triangle)."""
    me = Mesh.Mesh(path)
    topo = me.Topology
    sh = Part.Shape()
    sh.makeShapeFromMesh(topo, 0.001)
    shell = Part.Shell(sh.Faces)
    so = Part.Solid(shell)
    if so.Volume < 0:
        so.reverse()
    # the mesh's own enclosed volume in double precision (Mesh.Volume sums in single precision)
    pts, tris = topo
    vol6 = 0.0
    for i, j, k in tris:
        a, b, c = pts[i], pts[j], pts[k]
        vol6 += a.dot(b.cross(c))
    return so, {"triangles": int(me.CountFacets), "faces": len(so.Faces), "closed": bool(shell.isClosed()),
                "valid": bool(so.isValid()), "mesh_volume_mm3": abs(vol6) / 6.0, "solid_volume_mm3": float(so.Volume)}


def view_providers(joints, grounded):
    """FreeCAD's own joint view providers; only possible with the GUI up."""
    if not App.GuiUp:
        return False
    for j in joints:
        JointObject.ViewProviderJoint(j.ViewObject)
        if not hasattr(j.ViewObject.Proxy, "switch_JCS1"):          # not attached yet: no coin nodes
            j.ViewObject.Proxy.attach(j.ViewObject)
    JointObject.ViewProviderGroundedJoint(grounded.ViewObject)
    if not hasattr(grounded.ViewObject.Proxy, "app_obj"):
        grounded.ViewObject.Proxy.attach(grounded.ViewObject)
    return True


def build(ref_dir, purchased_dir, out_path, mode="compound"):
    t0 = time.time()
    problems = []
    with open(os.path.join(ref_dir, "oracle.json")) as fh:
        spec = json.load(fh)
    if mode not in ("compound", "insert", "link"):
        raise SystemExit("unknown mode %r" % mode)
    doc = App.newDocument("Strut")
    objs, solids, facet_rec = {}, {}, {}
    for tag in sorted(spec["parts"]):
        p = spec["parts"][tag]
        path = os.path.join(ref_dir, p["file"])
        if p["file"] not in solids:
            if sha256(path) != p["sha256"]:
                problems.append("mesh %s is not the recorded file" % p["file"])
            so, rec = faceted_solid(path)
            if not (rec["valid"] and rec["closed"]) or rec["faces"] != rec["triangles"] or \
                    abs(rec["solid_volume_mm3"] - rec["mesh_volume_mm3"]) > VOLUME_TOL_REL * rec["mesh_volume_mm3"]:
                problems.append("mesh %s did not become one closed valid solid, one face per triangle, of the "
                                "mesh's own volume" % p["file"])
            solids[p["file"]], facet_rec[p["family"]] = so, rec
        base = doc.addObject("Part::Feature", tag + "_base")
        base.Label = tag + " faceted base"
        base.Shape = solids[p["file"]].copy()
        body = doc.addObject("PartDesign::Body", tag)
        body.Label = tag
        body.BaseFeature = base
        body.Placement = placement(p["pose"])
        objs[tag] = body
    t_parts = time.time() - t0
    shapes = {}
    for pl in spec["placements"]:
        step = os.path.join(purchased_dir, pl["step"])
        if pl["type"] not in shapes and sha256(step) != pl["step_sha256"]:
            problems.append("vendor file %s is not the recorded file" % pl["step"])
        W = placement(pl["local_to_strut"])
        name = "Purchased_" + pl["id"].replace("-", "_")
        label = "%s %s" % (pl["pn"], pl["id"])
        if mode == "insert":
            import Import
            before = set(o.Name for o in doc.Objects)
            Import.insert(step, doc.Name)
            new = [o for o in doc.Objects if o.Name not in before]
            tops = [o for o in new if not any(q in new for q in o.InList)]
            part = doc.addObject("App::Part", name)
            part.Label = label
            for o in tops:
                part.addObject(o)
            part.Placement = W
            objs[pl["id"]] = part
            shapes.setdefault(pl["type"], None)
            continue
        if shapes.get(pl["type"]) is None:
            shapes[pl["type"]] = Part.read(step)
        src = shapes[pl["type"]]
        if mode == "link":
            key = "Vendor_" + pl["type"].replace("-", "_")
            if doc.getObject(key) is None:
                f = doc.addObject("Part::Feature", key)
                f.Label = "%s vendor model" % pl["pn"]
                f.Shape = src
            lk = doc.addObject("App::Link", name)
            lk.LinkedObject = doc.getObject(key)
            lk.Label = label
            lk.Placement = W.multiply(src.Placement)
            objs[pl["id"]] = lk
            continue
        f = doc.addObject("Part::Feature", name)
        f.Label = label
        f.Shape = src
        f.Placement = W.multiply(src.Placement)
        objs[pl["id"]] = f
    doc.recompute()
    t_purchased = time.time() - t0 - t_parts

    J = spec["joints"]
    asm = doc.addObject("Assembly::AssemblyObject", "Assembly")
    jg = asm.newObject("Assembly::JointGroup", "Joints")
    links, intended = {}, {}
    for key in sorted(objs):
        obj = objs[key]
        lk = asm.newObject("App::Link", "Link_" + key.replace("-", "_"))
        lk.LinkedObject = obj
        lk.Placement = obj.Placement
        lk.Label = key + " instance"
        links[key] = lk
        intended[key] = App.Placement(obj.Placement.toMatrix())
    ground = J["grounded"]
    g = jg.newObject("App::FeaturePython", "GroundedJoint")
    JointObject.GroundedJoint(g, links[ground])
    joints = []
    for key in J["fixed_from_grounded_to"]:
        a, b = links[ground], links[key]
        j = jg.newObject("App::FeaturePython", "Fixed_" + key.replace("-", "_"))
        JointObject.Joint(j, FIXED)
        j.Detach1 = j.Detach2 = True
        # FreeCAD 1.1's restore migration reads two subreference slots, even for detached frames
        j.Reference1 = (a, ["", ""])
        j.Reference2 = (b, ["", ""])
        world = App.Placement()
        j.Placement1 = a.Placement.inverse().multiply(world)
        j.Placement2 = b.Placement.inverse().multiply(world)
        joints.append(j)
    motor_pose = {pl["id"]: pl["local_to_strut"] for pl in spec["placements"]}
    n_rev = 0
    for r in J["revolute"]:
        m, w = links[r["motor"]], links[r["winch"]]
        j = jg.newObject("App::FeaturePython", "Revolute_" + r["winch"])
        JointObject.Joint(j, REVOLUTE)
        j.Detach1 = j.Detach2 = True
        j.Reference1 = (m, ["", ""])
        j.Reference2 = (w, ["", ""])
        frame = placement(motor_pose[r["motor"]])
        j.Placement1 = m.Placement.inverse().multiply(frame)
        j.Placement2 = w.Placement.inverse().multiply(frame)
        joints.append(j)
        n_rev += 1
    gui_vp = view_providers(joints, g)
    doc.recompute()
    d_recompute = deviation(links, intended)
    solve_rc = asm.solve()
    d_solve = deviation(links, intended)
    doc.recompute()
    d_saved = deviation(links, intended)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    t_built = time.time() - t0
    doc.saveAs(out_path)
    t_save = time.time() - t0 - t_built
    names = {k: v.Name for k, v in links.items()}
    App.closeDocument(doc.Name)

    t1 = time.time()
    doc2 = App.openDocument(out_path)
    doc2.recompute()
    asm2 = doc2.getObject("Assembly")
    links2 = {k: doc2.getObject(v) for k, v in names.items()}
    d_reopen = deviation(links2, intended)
    solve_rc2 = asm2.solve() if asm2 is not None else None
    d_resolve = deviation(links2, intended)
    bodies = [o for o in doc2.Objects if o.TypeId == "PartDesign::Body"]
    bad = [o.Label for o in bodies if not o.Shape.isValid() or len(o.Shape.Solids) != 1]
    kinds = {}
    for o in doc2.Objects:
        jt = getattr(o, "JointType", None)
        if jt is not None:
            kinds[str(jt)] = kinds.get(str(jt), 0) + 1
        elif hasattr(o, "ObjectToGround"):
            kinds["Grounded"] = kinds.get("Grounded", 0) + 1
    App.closeDocument(doc2.Name)
    t_reopen = time.time() - t1
    moved = [d for d in (d_recompute, d_solve, d_saved, d_reopen, d_resolve)
             if d[0] > MOVE_TOL_MM or d[1] > MOVE_TOL_RAD]
    want = {"Grounded": 1, "Fixed": len(J["fixed_from_grounded_to"]), "Revolute": len(J["revolute"])}
    ok = (not moved and solve_rc == 0 and solve_rc2 == 0 and not bad and asm2 is not None and not problems
          and len(bodies) == len(spec["parts"]) and kinds == want)
    return {"ok": ok, "out": out_path, "mode": mode, "bodies": len(bodies), "purchased_instances": len(spec["placements"]),
            "joints": kinds, "joints_wanted": want, "solve_rc": solve_rc, "solve_rc_after_reopen": solve_rc2,
            "moved_after_recompute": list(d_recompute), "moved_after_solve": list(d_solve),
            "moved_as_saved": list(d_saved), "moved_after_reopen": list(d_reopen),
            "moved_after_reopen_solve": list(d_resolve), "move_tolerance_mm_rad": [MOVE_TOL_MM, MOVE_TOL_RAD],
            "invalid_bodies": bad, "problems": problems, "faceted": facet_rec, "gui_view_providers": gui_vp,
            "bytes": os.path.getsize(out_path), "t_parts_s": t_parts, "t_purchased_s": t_purchased,
            "build_s": t_built, "save_s": t_save, "reopen_check_s": t_reopen, "total_s": time.time() - t0,
            "freecad": ".".join(App.Version()[:3])}


if __name__ == "__main__":
    rep = build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "compound")
    print("ORACLE-BUILD " + json.dumps(rep, sort_keys=True), flush=True)   # flushed: SystemExit may skip it
    if not rep["ok"]:
        raise SystemExit("the oracle document did not build cleanly")
