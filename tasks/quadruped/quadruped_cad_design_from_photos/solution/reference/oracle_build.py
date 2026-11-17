"""Build the oracle deliverable from a quad-cad-gt/1 folder: the FreeCAD document an engineer would leave (DESIGN.md
section 1; GRADER_SPEC.md section 8 "oracle"). Follows the arm task's oracle_build.py.

Runs under FreeCAD 1.1.4's headless interpreter (the task image's /opt/freecad/usr/bin/freecadcmd, or the host copy):

    freecadcmd -c "import sys; sys.argv = ['oracle_build.py', GT_DIR, CORE_STEP, FOOT_STEP, OUT_FCSTD]; exec(open(PATH).read())"

GT_DIR holds assembly.json and parts/*.step (each file in its design frame; a part's T_link maps it into its link frame;
instances of one geometry share a file, which is read once). The document holds
- one PartDesign Body per printed part, labelled with its tag: its BaseFeature is the part's STEP shape moved by T_link
  (baked into the geometry: a Body ignores its base feature's own Placement) and the Body sits at its link's origin
  (link frames are world-aligned at q = 0), so the Body is the part at configuration Z;
- the cores and foot pads imported from the given STEPs, one Part::Feature each, Placement = link origin x T_link;
- one Assembly: an App::Link component per Body, core and foot pad (Placement = the object's), a Grounded joint on the
  root link's first part, Fixed joints from each link's first part to its other parts, cores and feet (joint frame at
  the link origin), and one Revolute joint per GT joint between the parent's and the child's first parts (frame at
  point_mm with z = axis, both limits enabled, in degrees, relative to the saved pose).
"First part" = the first entry of assembly.json's parts list on that link. Nothing is solved: every joint's two frames
coincide in the saved pose. Prints ORACLE-BUILD {json}: counts, the document's size, the build and save times.
"""
import json
import math
import os
import sys
import time

import FreeCAD as App
import JointObject
import Part

JOINT_TYPES = list(JointObject.JointTypes)
FIXED, REVOLUTE = JOINT_TYPES.index("Fixed"), JOINT_TYPES.index("Revolute")


def fc_matrix(T):
    """FreeCAD matrix of a rigid 4 x 4 nested list. The rotation is re-orthonormalised (Gram-Schmidt on its columns: OCC
    refuses a rotation that is off by rounding); anything that is not a proper rotation within 1e-6 raises."""
    cols = [[float(T[i][k]) for i in range(3)] for k in range(3)]
    dev = max(abs(sum(a * b for a, b in zip(cols[i], cols[k])) - (1.0 if i == k else 0.0)) for i in range(3) for k in range(3))
    det = (cols[0][0] * (cols[1][1] * cols[2][2] - cols[1][2] * cols[2][1]) - cols[1][0] * (cols[0][1] * cols[2][2] - cols[0][2] * cols[2][1])
           + cols[2][0] * (cols[0][1] * cols[1][2] - cols[0][2] * cols[1][1]))
    if dev > 1e-6 or det <= 0 or any(abs(float(v)) > 1e-12 for v in T[3][:3]) or abs(float(T[3][3]) - 1.0) > 1e-12:
        raise ValueError("not a rigid transform (orthogonality error %.2e, det %.6f): %s" % (dev, det, T))

    def unit(v):
        n = math.sqrt(sum(x * x for x in v))
        return [x / n for x in v]

    x = unit(cols[0])
    y = unit([b - sum(p * q for p, q in zip(cols[1], x)) * a for a, b in zip(x, cols[1])])
    z = [x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]]
    R = [[x[i], y[i], z[i]] for i in range(3)]
    return App.Matrix(*(R[0] + [float(T[0][3])] + R[1] + [float(T[1][3])] + R[2] + [float(T[2][3])] + [0.0, 0.0, 0.0, 1.0]))


def frame_z(origin, z):
    """A placement at `origin` whose z axis is `z` (x chosen perpendicular, as the arm's oracle)."""
    z = App.Vector(*z).normalize()
    ref = App.Vector(1, 0, 0) if abs(z.x) < 0.9 else App.Vector(0, 1, 0)
    x = (ref - z * ref.dot(z)).normalize()
    y = z.cross(x)
    m = App.Matrix(x.x, y.x, z.x, origin[0], x.y, y.y, z.y, origin[1], x.z, y.z, z.z, origin[2], 0, 0, 0, 1)
    return App.Placement(m)


def safe_name(s):
    return "".join(c if c.isalnum() else "_" for c in s)


def save(doc, path):
    """saveAs without FreeCAD's backup copy of an existing file (a stray .FCBak beside the deliverable)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.isfile(path):
        os.remove(path)
    doc.saveAs(path)


def build(gt_dir, core_step, foot_step, out_path, doc_name="quadruped"):
    t0 = time.time()
    with open(os.path.join(gt_dir, "assembly.json")) as fh:
        gt = json.load(fh)
    origin = {l["name"]: App.Vector(*l["origin_mm"]) for l in gt["links"]}
    root = gt["root_link"]

    doc = App.newDocument(doc_name)
    shapes = {}

    def read(path):
        if path not in shapes:
            shapes[path] = Part.read(path)
        return shapes[path]

    objs = {}            # key -> (document object, link name, kind)
    for p in gt["parts"]:
        sh = read(os.path.join(gt_dir, p["file"])).copy()
        sh.transformShape(fc_matrix(p["T_link"]), True)            # into the link frame, identity placement
        base = doc.addObject("Part::Feature", p["tag"] + "_base")
        base.Shape = sh
        body = doc.addObject("PartDesign::Body", p["tag"])
        body.Label = p["tag"]
        body.BaseFeature = base
        body.Placement = App.Placement(origin[p["link"]], App.Rotation())
        objs[p["tag"]] = (body, p["link"], "part")
    for kind, recs, step, prefix in (("core", gt["cores"], core_step, "Core_"), ("foot", gt["feet"], foot_step, "Foot_")):
        src = read(step)
        for c in recs:
            name = prefix + safe_name(c["id"].split(".", 1)[-1])
            f = doc.addObject("Part::Feature", name)
            f.Label = name
            f.Shape = src.copy()
            f.Placement = App.Placement(origin[c["link"]], App.Rotation()).multiply(App.Placement(fc_matrix(c["T_link"])))
            objs[c["id"]] = (f, c["link"], kind)
    doc.recompute()
    t_bodies = time.time() - t0

    asm = doc.addObject("Assembly::AssemblyObject", "Assembly")
    jg = asm.newObject("Assembly::JointGroup", "Joints")
    comp = {}
    for key, (obj, _link, _kind) in objs.items():
        lk = asm.newObject("App::Link", obj.Name + "_cmp")
        lk.LinkedObject = obj
        lk.Placement = obj.Placement
        lk.Label = obj.Label
        comp[key] = lk

    def joint(name, type_index, a, b, world):
        j = jg.newObject("App::FeaturePython", name)
        JointObject.Joint(j, type_index)
        j.Detach1, j.Detach2 = True, True
        j.Reference1 = (a, [""])
        j.Reference2 = (b, [""])
        j.Placement1 = a.Placement.inverse().multiply(world)
        j.Placement2 = b.Placement.inverse().multiply(world)
        return j

    first = {}
    for p in gt["parts"]:
        first.setdefault(p["link"], p["tag"])
    g = jg.newObject("App::FeaturePython", "GroundedJoint")
    JointObject.GroundedJoint(g, comp[first[root]])
    n_fixed = 0
    for l in gt["links"]:
        name = l["name"]
        anchor = comp[first[name]]
        members = [p["tag"] for p in gt["parts"] if p["link"] == name and p["tag"] != first[name]]
        members += [c["id"] for c in gt["cores"] if c["link"] == name] + [f["id"] for f in gt["feet"] if f["link"] == name]
        for key in members:
            joint("Fixed_" + safe_name(key), FIXED, anchor, comp[key], App.Placement(origin[name], App.Rotation()))
            n_fixed += 1
    for jt in gt["joints"]:
        a, b = comp[first[jt["parent_link"]]], comp[first[jt["child_link"]]]
        j = joint("Revolute_" + jt["name"], REVOLUTE, a, b, frame_z(jt["point_mm"], jt["axis"]))
        j.EnableAngleMin, j.EnableAngleMax = True, True
        j.AngleMin = math.degrees(jt["limits_rad"][0])
        j.AngleMax = math.degrees(jt["limits_rad"][1])
    doc.recompute()
    t_build = time.time() - t0
    save(doc, out_path)
    t_save = time.time() - t0 - t_build
    out = {"document": out_path, "bytes": os.path.getsize(out_path), "bodies": len(gt["parts"]), "cores": len(gt["cores"]),
           "feet": len(gt["feet"]), "components": len(comp), "grounded": comp[first[root]].Name, "fixed": n_fixed,
           "revolute": len(gt["joints"]), "step_files_read": len(shapes), "seconds_bodies": round(t_bodies, 2),
           "seconds_build": round(t_build, 2), "seconds_save": round(t_save, 2), "freecad": ".".join(App.Version()[:3])}
    App.closeDocument(doc.Name)
    return out


if __name__ == "__main__" or "oracle_build" in sys.argv[0]:
    gt_dir, core_step, foot_step, out_path = sys.argv[1:5]
    print("ORACLE-BUILD", json.dumps(build(gt_dir, core_step, foot_step, out_path)))
