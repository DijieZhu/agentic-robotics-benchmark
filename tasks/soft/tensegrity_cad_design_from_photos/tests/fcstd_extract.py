"""Read the submitted FreeCAD document into plain data (the grader's only contact with FreeCAD).

Runs under FreeCAD 1.1.4's headless interpreter as an unprivileged account (verify.py starts it as `nobody` through
runuser and /usr/bin/env HOME=..., in a private work folder that holds only a copy of the document and of this file,
with a time limit); it imports FreeCAD's own modules and the standard library only:

    freecadcmd -c "import sys; sys.argv = ['fcstd_extract.py', FCSTD, OUT_DIR]; exec(open(PATH).read())"

Base: the extractor of uav_cad_design_from_photos (tests/fcstd_extract.py), with the root-object reading of
link sources from an earlier extractor of this task and the joint fields of the arm task's extractor (FreeCAD 1.1.4's
Revolute: EnableAngleMin / Max, AngleMin / Max; the global joint frames through UtilsAssembly.getJcsGlobalPlc, the
same reading as the arm task, checked on 1.1.4).

Before FreeCAD opens the document, the reader lists the Python proxy classes the document names (module and class of
every <Python> property in Document.xml; zipfile + xml.etree, nothing executed) into model.json["proxy_classes"], and
takes the work folder and the empty entry off sys.path, so a module could only come from FreeCAD's own installation.
Nothing in the document is executed beyond what FreeCAD runs to open and recompute it.

The document is opened and recomputed with the Assembly's solve-on-recompute turned off (the saved placements are the
assembled configuration), its top-level Assembly is found, and OUT_DIR/model.json (format "strut-fcstd-extract/1") is
written with, per solid, one BREP file (exact; the grader reads it: a STEP round trip can turn a valid solid invalid).
No per-leaf STEP is written (WRITE_STEP False: the grader never read it, and on
very large documents writing it took most of the extraction's time limit):

  components  the Assembly's members (App::Link, Body, feature, rigid App::Part / nested Assembly, the members of
              document folders) and their shape-bearing leaves. A leaf is a Body, a feature or a link to one; the members
              of an App::Part container (what FreeCAD's STEP import makes) are leaves of their own. Per leaf: label (of
              the linked object when the leaf is a link), link_label, type (TypeId of the linked object), object /
              target (document names), path, containers, body (the PartDesign Body when the target is one), visible,
              placement (global 4x4), valid (BRepCheck of the whole leaf shape), n_solids, n_free_faces / n_free_edges,
              volume and solids: one record per solid (brep, valid, closed, volume).
  no_assembly true when the document holds no Assembly: its visible top-level objects (doc.RootObjects, the members of
              their containers, and the visible objects referenced only by links to them) are read instead, no joints.
  joints      every joint object: name, label, type (JointType: Fixed, Revolute, ...), suppressed, assembly, part1 /
              part2 (the assembly components its two references attach to, through UtilsAssembly),
              jcs1 / jcs2 (global joint frames, 4x4; a Revolute turns about their common z axis), reference1 /
              reference2 (object name and subnames, for the diagnostics), detach1 / detach2, offset1 / offset2 (4x4;
              FreeCAD has already applied them to the joint placements), enable_angle_min / max and angle_min / max_deg,
              enable_length_min / max and length_min / max_mm.
  grounded    the components of the Grounded joints.

Errors are recorded in model.json, never raised: "errors" (the document does not open, too many shapes) and per-leaf
"error" describe the document; "extractor_failure" marks an exception of this reader itself. A model.json saying the
extraction did not finish is written first, so a run killed by the time limit leaves a reason.
"""
import json
import os
import sys
import time
import traceback
import xml.etree.ElementTree as ET
import zipfile

FORMAT = "strut-fcstd-extract/1"
MAX_DEPTH = 12                      # nesting of folders / parts / links followed (a cycle cannot recurse forever)
MAX_LEAVES = 4000                   # a document with more shape leaves than this is not read further
MAX_SOLIDS = 6000
MAX_PROXY_CLASSES = 200
WRITE_STEP = False                  # no per-leaf STEP beside the per-solid BREP (inspection only)
JOINT_CONTAINERS = ("Assembly::JointGroup", "Assembly::ViewGroup", "Assembly::BomGroup", "Assembly::SimulationGroup")
LINKS = ("App::Link", "App::LinkElement", "App::LinkPython")


def _sanitise_path():
    """Take the work folder and the empty entry off sys.path before FreeCAD restores any object."""
    cwd = os.path.realpath(os.getcwd())
    here = os.path.realpath(os.path.dirname(os.path.abspath(sys.argv[0]))) if sys.argv and sys.argv[0] else None
    keep = []
    for p in sys.path:
        rp = os.path.realpath(p) if p else ""
        if p in ("", ".") or rp == cwd or (here and rp == here):
            continue
        keep.append(p)
    sys.path[:] = keep
    sys.dont_write_bytecode = True


def proxy_classes(fcstd):
    """(module, class) of every Python proxy the document names, read from Document.xml without executing anything."""
    out = set()
    try:
        with zipfile.ZipFile(fcstd) as z:
            names = [n for n in z.namelist() if n == "Document.xml"]
            if not names:
                return [], "no Document.xml"
            with z.open(names[0]) as fh:
                for _ev, el in ET.iterparse(fh, events=("end",)):
                    if el.tag == "Python":
                        out.add((el.get("module") or "", el.get("class") or ""))
                        if len(out) >= MAX_PROXY_CLASSES:
                            break
                    el.clear()
    except Exception as exc:  # noqa: BLE001
        return sorted(out), "%s" % exc
    return sorted(out), None


_sanitise_path()
import FreeCAD as App  # noqa: E402
import Part  # noqa: E402


def matrix(placement):
    m = placement.toMatrix()
    return [[m.A11, m.A12, m.A13, m.A14], [m.A21, m.A22, m.A23, m.A24], [m.A31, m.A32, m.A33, m.A34], [0.0, 0.0, 0.0, 1.0]]


def linked(obj):
    return obj.getLinkedObject(True) if hasattr(obj, "getLinkedObject") else obj


def is_joint_container(obj):
    return obj.TypeId in JOINT_CONTAINERS


def is_joint(obj):
    return obj.TypeId == "App::FeaturePython" and (hasattr(obj, "JointType") or hasattr(obj, "ObjectToGround"))


def skipped(obj):
    return is_joint_container(obj) or is_joint(obj) or obj.TypeId.startswith("App::Origin")


def components_of(asm):
    """(folder prefix, object) of every Assembly component: the Assembly's members, document folders
    (App::DocumentObjectGroup) descended into; joints, joint containers and origins skipped; each object listed once, at
    the shallowest level it appears."""
    out, seen = [], set()

    def walk(group, prefix, depth):
        folders = []
        for child in group.Group:
            if skipped(child) or child.Name in seen:
                continue
            if child.TypeId == "App::DocumentObjectGroup":
                folders.append(child)
                continue
            seen.add(child.Name)
            out.append((prefix, child))
        for f in folders:
            if depth < MAX_DEPTH and f.Name not in seen:
                seen.add(f.Name)
                walk(f, prefix + f.Name + ".", depth + 1)
    walk(asm, "", 0)
    return out


def leaves(obj, prefix, containers, depth=0):
    """(subname path, leaf object, container chain) of every shape-bearing leaf under an assembly component: the
    component itself when it is a body / feature / link to one, else the shape-bearing members of a rigid sub-assembly
    (App::Part, a nested Assembly, a folder inside one), recursively."""
    target = linked(obj)
    if target.TypeId in ("App::Part", "Assembly::AssemblyObject", "App::DocumentObjectGroup"):
        out = []
        if depth < MAX_DEPTH:
            for child in target.Group:
                if not skipped(child):
                    out += leaves(child, prefix + child.Name + ".", containers + [target.Name], depth + 1)
        return out
    if hasattr(target, "Shape") and not target.Shape.isNull():
        return [(prefix, obj, containers)]
    return []


def moving_part(UtilsAssembly, asm, components, ref):
    """The assembly component a joint reference attaches to (FreeCAD 1.1's getMovingPart, else getComponentReference
    for a reference rooted at the Assembly with the component in its subname)."""
    part = UtilsAssembly.getMovingPart(ref)
    if part is not None and part.Name in components:
        return part
    if ref is None or len(ref) != 2:
        return part
    sub = ref[1][0] if ref[1] else ""
    comp, _rel = UtilsAssembly.getComponentReference(asm, ref[0], sub)
    return comp if comp is not None else part


def owning_assembly(obj):
    seen, level = set(), [obj]
    for _ in range(4):
        nxt = []
        for o in level:
            for p in o.InList:
                if p.TypeId == "Assembly::AssemblyObject":
                    return p.Name
                if p.Name not in seen:
                    seen.add(p.Name)
                    nxt.append(p)
        level = nxt
    return None


def free_geometry(shp):
    """(faces, edges) of the shape that belong to no solid (an open shell, a loose face or wire)."""
    sol = shp.Solids
    if not sol:
        n_faces = len(shp.Faces)
        n_edges = len(shp.Edges) - (len(Part.Compound(shp.Faces).Edges) if n_faces else 0)
        return n_faces, max(0, n_edges)
    in_faces = sum(len(s.Faces) for s in sol)
    n_faces = max(0, len(shp.Faces) - in_faces)
    in_edges = len(Part.Compound(sol).Edges)
    n_edges = max(0, len(shp.Edges) - in_edges)
    return n_faces, n_edges


def write_model(model, out_dir):
    with open(os.path.join(out_dir, "model.json"), "w") as fh:
        json.dump(model, fh, indent=1, sort_keys=True)


def extract(fcstd, out_dir):
    grp = App.ParamGet("User parameter:BaseApp/Preferences/Mod/Assembly")
    had = grp.GetBool("SolveOnRecompute", True)
    grp.SetBool("SolveOnRecompute", False)
    try:
        return _extract(fcstd, out_dir)
    finally:
        grp.SetBool("SolveOnRecompute", had)


def root_components(doc):
    """(prefix, object) of the shape-bearing objects at the top of a document without an Assembly: its visible root
    objects, then the visible shape-bearing objects referenced only by App::Link objects (the source of a link that
    reuses it sits in no container and is not a root object only because of those links), once each."""
    out = []
    for o in doc.RootObjects:
        if skipped(o) or not bool(getattr(o, "Visibility", True)):
            continue
        out.append(("", o))
    roots = set(o.Name for o in doc.RootObjects)
    for o in doc.Objects:
        if o.Name in roots or o.TypeId == "App::LinkElement" or skipped(o) or not bool(getattr(o, "Visibility", True)) or \
                not hasattr(o, "Shape"):
            continue
        parents = o.InList
        if parents and all(p.TypeId in LINKS for p in parents):
            out.append(("", o))
    return out


def _quantity(v):
    return float(getattr(v, "Value", v))


def joint_record(UtilsAssembly, asm, components, obj):
    rec = {"name": obj.Name, "label": obj.Label, "type": str(obj.JointType),
           "suppressed": bool(getattr(obj, "Suppressed", False)), "assembly": owning_assembly(obj)}
    try:
        for k in ("1", "2"):
            ref = getattr(obj, "Reference" + k)
            part = moving_part(UtilsAssembly, asm, components, ref)
            rec["part" + k] = part.Name if part is not None else None
            rec["jcs" + k] = matrix(UtilsAssembly.getJcsGlobalPlc(getattr(obj, "Placement" + k), ref))
            if ref is not None and len(ref) == 2:
                rec["reference" + k] = {"object": getattr(ref[0], "Name", None), "subnames": [str(x) for x in (ref[1] or [])][:4]}
            rec["detach" + k] = bool(getattr(obj, "Detach" + k, False))
            off = getattr(obj, "Offset" + k, None)
            if off is not None:
                rec["offset" + k] = matrix(off)
        for k in ("Min", "Max"):
            rec["enable_angle_" + k.lower()] = bool(getattr(obj, "EnableAngle" + k, False))
            rec["angle_" + k.lower() + "_deg"] = _quantity(getattr(obj, "Angle" + k, 0.0))
            rec["enable_length_" + k.lower()] = bool(getattr(obj, "EnableLength" + k, False))
            rec["length_" + k.lower() + "_mm"] = _quantity(getattr(obj, "Length" + k, 0.0))
    except Exception as exc:  # noqa: BLE001
        rec["error"] = "%s" % exc
    return rec


def _extract(fcstd, out_dir):
    t0 = time.time()
    model = {"format": FORMAT, "errors": [], "notes": [], "components": [], "joints": [], "grounded": []}
    os.makedirs(os.path.join(out_dir, "shapes"), exist_ok=True)
    classes, perr = proxy_classes(fcstd)
    model["proxy_classes"] = [list(c) for c in classes]
    if perr:
        model["notes"].append("Document.xml not listed: %s" % perr[:200])
    try:
        doc = App.openDocument(fcstd)
    except Exception as exc:  # noqa: BLE001
        model["errors"].append("the document does not open: %s" % exc)
        return model
    model["freecad"] = ".".join(App.Version()[:3])
    model["open_s"] = time.time() - t0
    try:
        doc.recompute()
    except Exception as exc:  # noqa: BLE001
        model["notes"].append("the document does not recompute (%s): read as saved" % exc)
    assemblies = [o for o in doc.Objects if o.TypeId == "Assembly::AssemblyObject"]
    model["n_assemblies"] = len(assemblies)
    asm = root = None
    root_sub = ""
    if assemblies:
        asm = assemblies[0]
        if len(assemblies) > 1:  # the top-level one: not contained in another assembly
            tops = [a for a in assemblies if not any(p[0].TypeId == "Assembly::AssemblyObject" for p in (a.Parents or []))]
            asm = tops[0] if tops else asm
            model["assemblies"] = [a.Name for a in assemblies]
        root = asm
        parents = asm.Parents or []
        if parents:
            root, root_sub = parents[0][0], parents[0][1]
        model["assembly"] = asm.Name
        comps = components_of(asm)
    else:
        model["no_assembly"] = True
        model["assembly"] = None
        model["notes"].append("no Assembly in the document: its visible top-level solids are read")
        comps = root_components(doc)

    import UtilsAssembly  # FreeCAD's Assembly helpers (headless-safe)

    n_leaves = n_solids = 0
    t_step = 0.0
    for prefix, comp in comps:
        entry = {"name": comp.Name, "label": comp.Label, "type": comp.TypeId, "leaves": []}
        if prefix:
            entry["folder"] = prefix[:-1]
        try:
            found = leaves(comp, "", [])
        except Exception as exc:  # noqa: BLE001
            entry["error"] = "the component's leaves cannot be listed: %s" % exc
            found = []
        for sub, leaf, containers in found:
            if asm is None and sub and not bool(getattr(leaf, "Visibility", True)):
                continue                      # a hidden member of a container outside any Assembly
            n_leaves += 1
            if n_leaves > MAX_LEAVES:
                model["errors"].append("more than %d shape leaves in the document" % MAX_LEAVES)
                return model
            target = linked(leaf)
            path = (root_sub + prefix + comp.Name + "." + sub) if asm is not None else (comp.Name + "." + sub)
            rec = {"label": (target.Label or "").strip(), "link_label": (leaf.Label or "").strip(), "type": target.TypeId,
                   "object": leaf.Name, "target": target.Name, "path": path, "containers": containers,
                   "visible": bool(getattr(leaf, "Visibility", True)), "solids": []}
            if target.TypeId == "PartDesign::Body":
                rec["body"] = target.Name
            try:
                if asm is not None:
                    shp = Part.getShape(root, path, needSubElement=False, transform=True).copy()
                else:
                    shp = Part.getShape(comp, sub, needSubElement=False, transform=True).copy()
                rec["valid"] = bool(shp.isValid())
                sols = shp.Solids
                rec["n_solids"] = len(sols)
                rec["n_free_faces"], rec["n_free_edges"] = free_geometry(shp)
                rec["volume"] = float(shp.Volume)
                rec["placement"] = matrix(shp.Placement)
                for k, s in enumerate(sols):
                    n_solids += 1
                    if n_solids > MAX_SOLIDS:
                        model["errors"].append("more than %d solids in the document" % MAX_SOLIDS)
                        return model
                    srec = {"brep": "shapes/%04d_%03d.brep" % (n_leaves, k), "valid": bool(s.isValid()),
                            "closed": bool(s.isClosed()), "volume": float(s.Volume)}
                    s.exportBrep(os.path.join(out_dir, srec["brep"]))
                    rec["solids"].append(srec)
                if WRITE_STEP and sols:
                    ts = time.time()
                    rec["step"] = "shapes/%04d.step" % n_leaves
                    try:
                        shp.exportStep(os.path.join(out_dir, rec["step"]))
                    except Exception as exc:  # noqa: BLE001 - the STEP is for inspection only
                        rec["step_error"] = "%s" % exc
                        rec.pop("step", None)
                    t_step += time.time() - ts
            except Exception as exc:  # noqa: BLE001
                rec["error"] = "%s" % exc
            entry["leaves"].append(rec)
        model["components"].append(entry)
    model["step_s"] = t_step

    if asm is None:
        model["n_leaves"] = n_leaves
        model["n_solids"] = n_solids
        model["elapsed_s"] = time.time() - t0
        return model
    components = set(c.Name for _p, c in comps)
    for obj in doc.Objects:
        if not is_joint(obj):
            continue
        if hasattr(obj, "ObjectToGround"):
            tgt = obj.ObjectToGround
            model["grounded"].append(tgt.Name if tgt is not None else None)
            continue
        model["joints"].append(joint_record(UtilsAssembly, asm, components, obj))
    model["n_leaves"] = n_leaves
    model["n_solids"] = n_solids
    model["elapsed_s"] = time.time() - t0
    return model


def main(argv):
    fcstd, out_dir = argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    write_model({"format": FORMAT, "errors": ["the extraction did not finish (time limit?)"]}, out_dir)
    try:
        model = extract(fcstd, out_dir)
    except Exception:  # noqa: BLE001 - a fault of this reader, not of the document: the grader fails the run with it
        model = {"format": FORMAT, "errors": ["extraction failed: " + traceback.format_exc()[-1500:]],
                 "extractor_failure": True}
    write_model(model, out_dir)
    print("STRUT-EXTRACT done", len(model.get("components", [])), "components", model.get("n_leaves", 0), "leaves",
          model.get("n_solids", 0), "solids", len(model.get("joints", [])), "joints", len(model.get("errors", [])), "errors")


main(sys.argv)
