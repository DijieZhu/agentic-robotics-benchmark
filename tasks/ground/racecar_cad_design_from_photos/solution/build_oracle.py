"""Build the oracle deliverable: the native FreeCAD document of the reference race car (the car as modelled).

forked from tasks/uav/uav_cad_design_from_photos/solution/build_oracle.py (benchmark main b7b55ac, sha256
91728ad456c72e605d3ddcf52c5b17ba3757e8610f97e042ec128fc711c1dd07). Changes: the document is named Racecar; a Body is
bad when it is not exactly one closed solid (no validity check: the vendor solids of this car fail FreeCAD's BRepCheck
after a rigid move while being closed and usable); the report is printed with flush and also written to
OUT_FCSTD.report.json (under freecadcmd -c the last lines printed before a SystemExit can be lost), an exception of
the build included; mode "link" keeps its one feature per type in a dict (it looked the feature up by a name, and
FreeCAD rewrites names with dots, as in the spacer pin types SP-2.8-48.20: those links had no target); a component
without geometry fails the build (components_without_shape).

Runs under FreeCAD 1.1.4's headless interpreter (the task image's /opt/freecad/usr/bin/freecadcmd), from solve.sh:

    freecadcmd -c "import sys; sys.argv = ['build_oracle.py', REF_DIR, HOME_DIR, OUT_FCSTD(, MODE)]; exec(open(PATH).read())"

REF_DIR = solution/reference: oracle.json (the tags and families, the ground component, the given placements: type, part
number, STEP path relative to HOME_DIR, local_to_world) and parts/Pnn.brep (the made parts' solids in the car frame,
byte-identical to the grader's tests/reference/parts). HOME_DIR = the agent home, where setup.sh staged the payload
(base/base_assembly.step, purchased/<PN>/<PN>.step). The document holds:
  - one PartDesign Body per tag, labelled with the tag, its base feature the solid read from the exact BREP (a Body
    ignores its base feature's placement: a solid that carries a top-level location goes in as its raw geometry and the
    location becomes the Body's placement);
  - every given instance, its STEP imported unchanged and placed at the instance's local_to_world: MODE "compound"
    (default) one Part::Feature holding the STEP's solids; "insert" FreeCAD's own STEP import (Import.insert) into an
    App::Part per instance; "link" one Part::Feature per type, outside the Assembly, linked once per instance;
  - one Assembly: an App::Link per component, a Grounded joint on oracle.json's ground (the base assembly) and a Fixed
    joint from it to every other component (detached frames at the world origin; two subreference slots, as FreeCAD
    1.1's restore migration expects even for detached frames).
The Assembly is recomputed and solved, every component's placement compared with the one it was given (nothing may move),
the document saved, reopened and checked again. Prints ORACLE-BUILD <json> and writes the same report to
OUT_FCSTD.report.json.
"""
import json
import math
import os
import sys
import time
import traceback

import FreeCAD as App
import JointObject
import Part

FIXED = list(JointObject.JointTypes).index("Fixed")
MOVE_TOL_MM = 1e-7
MOVE_TOL_RAD = 1e-9


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


def build(ref_dir, purchased_dir, out_path, mode="compound"):
    t0 = time.time()
    with open(os.path.join(ref_dir, "oracle.json")) as fh:
        spec = json.load(fh)
    doc = App.newDocument("Racecar")
    objs = {}
    for tag in sorted(spec["parts"]):
        shape = Part.read(os.path.join(ref_dir, "parts", tag + ".brep"))
        loc = shape.Placement
        shape.Placement = App.Placement()
        base = doc.addObject("Part::Feature", tag + "_base")
        base.Label = tag + " original solid"
        base.Shape = shape
        body = doc.addObject("PartDesign::Body", tag)
        body.Label = tag
        body.BaseFeature = base
        body.Placement = loc
        objs[tag] = body
    shapes = {}
    vendors = {}      # mode "link": the one feature per type (FreeCAD rewrites object names with dots, e.g. SP-2.8-48.20)
    for pl in spec["placements"]:
        step = os.path.join(purchased_dir, pl["step"])
        W = placement(pl["local_to_world"])
        name = "Purchased_" + pl["id"].replace("-", "_")
        label = "%s %s" % (pl["pn"], pl["id"])
        if mode == "insert":
            import Import
            before = set(o.Name for o in doc.Objects)
            Import.insert(step, doc.Name)
            new = [o for o in doc.Objects if o.Name not in before]
            tops = [o for o in new if not any(p in new for p in o.InList)]
            part = doc.addObject("App::Part", name)
            part.Label = label
            for o in tops:
                part.addObject(o)
            part.Placement = W
            objs[pl["id"]] = part
            continue
        if pl["type"] not in shapes:
            shapes[pl["type"]] = Part.read(step)
        src = shapes[pl["type"]]
        if mode == "link":
            if pl["type"] not in vendors:
                f = doc.addObject("Part::Feature", "Vendor_" + pl["type"].replace("-", "_"))
                f.Label = "%s vendor model" % pl["pn"]
                f.Shape = src
                vendors[pl["type"]] = f
            lk = doc.addObject("App::Link", name)
            lk.LinkedObject = vendors[pl["type"]]
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

    asm = doc.addObject("Assembly::AssemblyObject", "Assembly")
    jg = asm.newObject("Assembly::JointGroup", "Joints")
    links, intended = {}, {}
    for key in sorted(objs):
        obj = objs[key]
        lk = asm.newObject("App::Link", "Link_" + key.replace("-", "_"))
        lk.LinkedObject = obj
        lk.Placement = obj.Placement
        lk.Label = obj.Label if key not in spec["parts"] else key + " instance"
        links[key] = lk
        intended[key] = App.Placement(obj.Placement.toMatrix())
    ground = spec.get("ground", "P01")
    g = jg.newObject("App::FeaturePython", "GroundedJoint")
    JointObject.GroundedJoint(g, links[ground])
    n_fixed = 0
    for key in sorted(links):
        if key == ground:
            continue
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
        n_fixed += 1
    doc.recompute()
    d_recompute = deviation(links, intended)
    solve_rc = asm.solve()
    d_solve = deviation(links, intended)
    doc.recompute()
    d_saved = deviation(links, intended)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    doc.saveAs(out_path)
    names = {k: v.Name for k, v in links.items()}
    App.closeDocument(doc.Name)
    t_save = time.time()
    doc2 = App.openDocument(out_path)
    doc2.recompute()
    asm2 = doc2.getObject("Assembly")
    links2 = {k: doc2.getObject(v) for k, v in names.items()}
    d_reopen = deviation(links2, intended)
    bodies = [o for o in doc2.Objects if o.TypeId == "PartDesign::Body"]
    # a Body is bad unless it is exactly one closed solid (no validity check: see the module docstring)
    bad = [o.Label for o in bodies if len(o.Shape.Solids) != 1 or not o.Shape.Solids[0].isClosed()]
    # every component must carry geometry (a link that lost its target would pass every other check)
    empty = sorted(k for k, lk in links2.items() if lk is None or Part.getShape(lk).isNull())
    App.closeDocument(doc2.Name)
    moved = [d for d in (d_recompute, d_solve, d_saved, d_reopen) if d[0] > MOVE_TOL_MM or d[1] > MOVE_TOL_RAD]
    return {"ok": not moved and solve_rc == 0 and not bad and not empty and asm2 is not None, "out": out_path,
            "mode": mode, "bodies": len(bodies), "purchased_instances": len(spec["placements"]), "n_fixed": n_fixed,
            "solve_rc": solve_rc, "moved_after_recompute": list(d_recompute), "moved_after_solve": list(d_solve),
            "moved_as_saved": list(d_saved), "moved_after_reopen": list(d_reopen), "invalid_bodies": bad,
            "components_without_shape": empty, "bytes": os.path.getsize(out_path), "build_s": t_save - t0,
            "total_s": time.time() - t0}


def write_report(rep, out_path):
    with open(out_path + ".report.json", "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)
    print("ORACLE-BUILD " + json.dumps(rep, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        rep = build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "compound")
    except Exception:  # noqa: BLE001 - reported in the file, then the build fails
        rep = {"ok": False, "out": sys.argv[3], "error": traceback.format_exc()[-3000:]}
    write_report(rep, sys.argv[3])
    if not rep["ok"]:
        raise SystemExit("the oracle document did not build cleanly")
