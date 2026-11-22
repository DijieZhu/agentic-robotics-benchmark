"""Build the oracle deliverable: the native FreeCAD document of the reference leg (the leg as designed).

forked from tasks/ground/racecar_cad_design_from_photos/solution/build_oracle.py (benchmark main 850dee3, sha256
fc0007d73414b103bd317f92209a054145013f01b1390b1d569bf2c720d929da). Changes: the document is named Leg; the given files
are read from oracle.json's "file" (a STEP, or the shell's BREP); the joints follow oracle.json's rigid groups: a Fixed
joint from each group's first member to every other member, a Revolute joint between two groups at each revolute's axis
(its frames' z on the axis), a Ball joint at each ball's centre, the shell grounded; the assembly solver is never run
(the Assembly's solve-on-recompute is turned off while the document is built, as the extractor does): the placements as
built are the document; the report records the largest move of any component after recompute, save and reopen (none
may move), every Body being one closed solid and every component carrying geometry.

Runs under FreeCAD 1.1.4's headless interpreter (the task image's /opt/freecad/usr/bin/freecadcmd), from solve.sh:

    freecadcmd -c "import sys; sys.argv = ['build_oracle.py', REF_DIR, HOME_DIR, OUT_FCSTD(, MODE)]; exec(open(PATH).read())"

REF_DIR = solution/reference: oracle.json (tags and families, the ground, the given placements: type, part number, file
relative to HOME_DIR, local_to_world; the rigid groups; the revolutes and balls) and parts/Pnn.brep (the made parts'
solids in the reference frame, byte-identical to the grader's tests/reference/parts). HOME_DIR = the agent home, where
setup.sh staged the payload (given/<PN>/<PN>.step, given/thigh_shell/thigh_shell.brep). The document holds:
  - one PartDesign Body per tag, labelled with the tag, its base feature the solid read from the exact BREP (a Body
    ignores its base feature's placement: a solid that carries a top-level location goes in as its raw geometry and the
    location becomes the Body's placement);
  - every given instance, its file imported unchanged and placed at the instance's local_to_world: MODE "compound"
    (default) one Part::Feature holding the file's solids; "insert" FreeCAD's own import (Import.insert for a STEP,
    Part.insert for a BREP) into an App::Part per instance; "link" one Part::Feature per type, outside the Assembly,
    linked once per instance;
  - one Assembly: an App::Link per component, the Grounded joint and the group joints (detached frames; two
    subreference slots, as FreeCAD 1.1's restore migration expects even for detached frames).
Prints ORACLE-BUILD <json> and writes the same report to OUT_FCSTD.report.json.

The view (display only; nothing the grader reads): FreeCAD draws two damaged faces of the given shell file beyond their
true boundary (SHELL_FALSE_FACES, 0-based indices into Shape.Faces). The document hides them in the 3D view, with the
calls the shell's README gives (one appearance per face, those two fully transparent), and shows the Assembly and its
links in place of the source objects (the made parts' Bodies draw nothing in the GUI: see shown_hidden). With a GUI
(App.GuiUp) the calls run on the view objects and FreeCAD saves the view itself; headless there is no view object, so
the same view data are written into the saved file as its
GuiDocument.xml (visibility, the shell's per-face appearance, an isometric camera): FreeCAD's GUI reads them when it opens
the document; FreeCAD without a GUI, and so the grader's extractor, never reads them. Shapes and placements are not
touched; the report records what was stored ("view").
"""
import json
import math
import os
import struct
import sys
import time
import traceback
import zipfile
from xml.sax.saxutils import escape

import FreeCAD as App
import JointObject
import Part

JT = list(JointObject.JointTypes)
FIXED, REVOLUTE, BALL = JT.index("Fixed"), JT.index("Revolute"), JT.index("Ball")
MOVE_TOL_MM = 1e-7
MOVE_TOL_RAD = 1e-9
SHELL_TYPE = "SHELL"
SHELL_FACES = 312                 # faces of the given shell file
SHELL_FALSE_FACES = (171, 269)    # the two faces FreeCAD draws beyond their boundary (display only)
# FreeCAD 1.1's default shape appearance (ambient, diffuse, specular, emissive as packed RGBA; shininess; transparency)
DEFAULT_LOOK = (0x555555FF, 0xCCCCE6FF, 0x888888FF, 0x000000FF, 0.9, 0.0)
ISOMETRIC_Q = (0.424708, 0.17592, 0.339851, 0.820473)   # FreeCAD's isometric view orientation (x, y, z, w)


def placement(m):
    return App.Placement(App.Matrix(*[float(v) for row in m for v in row]))


def frame_z(point, direction):
    """A placement with its origin at point and its z along direction."""
    d = App.Vector(*[float(x) for x in direction])
    d.normalize()
    return App.Placement(App.Vector(*[float(x) for x in point]), App.Rotation(App.Vector(0, 0, 1), d))


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


def shell_feature(spec, objs):
    """The Part::Feature that holds the given shell (None in mode insert / link, or when its file is not the known one)."""
    ids = [pl["id"] for pl in spec["placements"] if pl["type"] == SHELL_TYPE]
    f = objs.get(ids[0]) if len(ids) == 1 else None
    if f is None or f.TypeId != "Part::Feature" or len(f.Shape.Faces) != SHELL_FACES:
        return None
    return f


def shown_hidden(doc, asm, links):
    """Shown: the Assembly, its members and links. Hidden: the Bodies, their base features and the given pieces'
    features. The links draw the given pieces in place. A Body built here has a base feature and no Tip, and FreeCAD's
    GUI draws nothing for it or for a link to it, while its base feature drawn on its own stands in the part's own frame,
    not where the part is: so the base features stay hidden too, and the made parts are not drawn when the document is
    opened (the grader reads the Bodies' shapes, which are unaffected)."""
    shown = set([asm.Name] + [o.Name for o in asm.Group] + [lk.Name for lk in links.values()])
    hidden = set(o.Name for o in doc.Objects if o.TypeId in ("PartDesign::Body", "Part::Feature") and o.Name not in shown)
    return shown, hidden


def view_in_gui(doc, asm, links, shell):
    """With a GUI: hide the shell's two false faces (the README's calls) and show the Assembly instead of the sources."""
    if shell is not None:
        view = shell.ViewObject
        look = [view.ShapeAppearance[0] for face in shell.Shape.Faces]
        for i in SHELL_FALSE_FACES:
            look[i].Transparency = 1.0
        view.ShapeAppearance = look
    shown, hidden = shown_hidden(doc, asm, links)
    for o in doc.Objects:
        if o.ViewObject is not None and (o.Name in shown or o.Name in hidden):
            o.ViewObject.Visibility = o.Name in shown
    return {"stored_by": "the GUI (view objects)", "shell_faces_hidden": list(SHELL_FALSE_FACES) if shell else []}


def isometric_camera(links):
    """An orthographic camera in FreeCAD's isometric orientation framing every component, as Coin's camera text."""
    bb = App.BoundBox()
    for lk in links.values():
        bb.add(Part.getShape(lk).BoundBox)
    c = bb.Center
    diag = max(bb.DiagonalLength, 1.0)
    rot = App.Rotation(*ISOMETRIC_Q)
    pos = c + rot.multVec(App.Vector(0, 0, 1)) * diag
    axis, ang = rot.Axis, rot.Angle
    return ("OrthographicCamera {\n  viewportMapping ADJUST_CAMERA\n  position %.6f %.6f %.6f\n  orientation %.6f %.6f %.6f  %.6f\n"
            "  aspectRatio 1\n  focalDistance %.6f\n  height %.6f\n\n}\n" % (pos.x, pos.y, pos.z, axis.x, axis.y, axis.z, ang,
                                                                         diag, 1.05 * diag))


def view_in_file(out_path, doc, asm, links, shell):
    """Headless: write the same view into the saved file as its GuiDocument.xml, which only FreeCAD's GUI reads."""
    with zipfile.ZipFile(out_path) as z:
        names = set(z.namelist())
    if "GuiDocument.xml" in names:
        return {"stored_by": None, "note": "the file already has a GuiDocument.xml: left as saved"}
    shown, hidden = shown_hidden(doc, asm, links)
    vis = '<Property name="Visibility" type="App::PropertyBool" status="1">\n<Bool value="%s"/>\n</Property>\n'
    entries, files = [], {}
    for name in sorted(shown | hidden):
        props = [vis % ("true" if name in shown else "false")]
        if shell is not None and name == shell.Name:
            fname = "ShapeAppearance_" + name
            data = struct.pack("<I", SHELL_FACES)
            for i in range(SHELL_FACES):
                look = DEFAULT_LOOK[:5] + ((1.0,) if i in SHELL_FALSE_FACES else DEFAULT_LOOK[5:])
                data += struct.pack("<IIIIff", *look)
            data += b"\0" * (12 * SHELL_FACES)        # per material: image, image path, uuid (empty strings)
            files[fname] = data
            props.insert(0, '<Property name="ShapeAppearance" type="App::PropertyMaterialList" status="1">\n'
                            '<MaterialList file="%s" version="3"/>\n</Property>\n' % fname)
        entries.append('<ViewProvider name="%s" expanded="0">\n<Properties Count="%d" TransientCount="0">\n%s'
                       '</Properties>\n</ViewProvider>\n' % (escape(name), len(props), "".join(props)))
    xml = ("<?xml version='1.0' encoding='utf-8'?>\n<Document SchemaVersion=\"1\" HasExpansion=\"1\">\n<Expand />\n"
           "<ViewProviderData Count=\"%d\">\n%s</ViewProviderData>\n<Camera settings=\"%s\"/>\n</Document>\n"
           % (len(entries), "".join(entries), escape(isometric_camera(links), {"\n": "&#10;", '"': "&quot;"})))
    with zipfile.ZipFile(out_path, "a", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("GuiDocument.xml", xml)
        for k, v in files.items():
            z.writestr(k, v)
    return {"stored_by": "GuiDocument.xml written into the saved file (headless build)", "view_providers": len(entries),
            "shown": len(shown), "hidden": len(hidden), "shell_faces_hidden": list(SHELL_FALSE_FACES) if shell else []}


def build(ref_dir, home_dir, out_path, mode="compound"):
    t0 = time.time()
    with open(os.path.join(ref_dir, "oracle.json")) as fh:
        spec = json.load(fh)
    doc = App.newDocument("Leg")
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
    vendors = {}      # mode "link": the one feature per type (FreeCAD rewrites object names with dots)
    for pl in spec["placements"]:
        path = os.path.join(home_dir, pl["file"])
        W = placement(pl["local_to_world"])
        name = "Given_" + pl["id"].replace("-", "_")
        label = "%s %s" % (pl["pn"] or pl["type"], pl["id"])
        if mode == "insert":
            before = set(o.Name for o in doc.Objects)
            if path.lower().endswith(".brep"):
                Part.insert(path, doc.Name)
            else:
                import Import
                Import.insert(path, doc.Name)
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
            shapes[pl["type"]] = Part.read(path)
        src = shapes[pl["type"]]
        if mode == "link":
            if pl["type"] not in vendors:
                f = doc.addObject("Part::Feature", "Type_" + pl["type"].replace("-", "_"))
                f.Label = "%s model" % (pl["pn"] or pl["type"])
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
    g = jg.newObject("App::FeaturePython", "GroundedJoint")
    JointObject.GroundedJoint(g, links[spec["ground"]])

    def joint(name, jtype, a, b, world):
        j = jg.newObject("App::FeaturePython", name)
        JointObject.Joint(j, jtype)
        j.Detach1 = j.Detach2 = True
        # FreeCAD 1.1's restore migration reads two subreference slots, even for detached frames
        j.Reference1 = (a, ["", ""])
        j.Reference2 = (b, ["", ""])
        j.Placement1 = a.Placement.inverse().multiply(world)
        j.Placement2 = b.Placement.inverse().multiply(world)
        return j

    n = {"fixed": 0, "revolute": 0, "ball": 0}
    first = {}
    for gname in sorted(spec["groups"]):
        members = spec["groups"][gname]
        first[gname] = members[0]
        for key in members[1:]:
            joint("Fixed_" + key.replace("-", "_"), FIXED, links[members[0]], links[key], App.Placement())
            n["fixed"] += 1
    for r in spec.get("revolutes", []):
        joint("Revolute_" + r["name"], REVOLUTE, links[first[r["a"]]], links[first[r["b"]]], frame_z(r["point_mm"], r["dir"]))
        n["revolute"] += 1
    for r in spec.get("balls", []):
        joint("Ball_" + r["name"], BALL, links[first[r["a"]]], links[first[r["b"]]], frame_z(r["point_mm"], [0, 0, 1]))
        n["ball"] += 1
    doc.recompute()
    d_recompute = deviation(links, intended)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    shell = shell_feature(spec, objs)
    gui = bool(App.GuiUp) and shell is not None and getattr(shell, "ViewObject", None) is not None
    view = view_in_gui(doc, asm, links, shell) if gui else None
    doc.saveAs(out_path)
    if not gui:
        view = view_in_file(out_path, doc, asm, links, shell)
    names = {k: v.Name for k, v in links.items()}
    App.closeDocument(doc.Name)
    t_save = time.time()
    doc2 = App.openDocument(out_path)
    doc2.recompute()
    asm2 = doc2.getObject("Assembly")
    links2 = {k: doc2.getObject(v) for k, v in names.items()}
    d_reopen = deviation(links2, intended)
    bodies = [o for o in doc2.Objects if o.TypeId == "PartDesign::Body"]
    # a Body is bad unless it is exactly one closed solid (no validity check)
    bad = [o.Label for o in bodies if len(o.Shape.Solids) != 1 or not o.Shape.Solids[0].isClosed()]
    # every component must carry geometry (a link that lost its target would pass every other check)
    empty = sorted(k for k, lk in links2.items() if lk is None or Part.getShape(lk).isNull())
    App.closeDocument(doc2.Name)
    moved = [d for d in (d_recompute, d_reopen) if d[0] > MOVE_TOL_MM or d[1] > MOVE_TOL_RAD]
    return {"ok": not moved and not bad and not empty and asm2 is not None, "out": out_path, "mode": mode,
            "bodies": len(bodies), "given_instances": len(spec["placements"]), "joints": n,
            "moved_after_recompute": list(d_recompute), "moved_after_reopen": list(d_reopen), "invalid_bodies": bad,
            "components_without_shape": empty, "bytes": os.path.getsize(out_path), "build_s": t_save - t0,
            "total_s": time.time() - t0, "view": view}


def write_report(rep, out_path):
    with open(out_path + ".report.json", "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)
    print("ORACLE-BUILD " + json.dumps(rep, sort_keys=True), flush=True)


if __name__ == "__main__":
    grp = App.ParamGet("User parameter:BaseApp/Preferences/Mod/Assembly")
    had = grp.GetBool("SolveOnRecompute", True)
    grp.SetBool("SolveOnRecompute", False)
    try:
        rep = build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else "compound")
    except Exception:  # noqa: BLE001 - reported in the file, then the build fails
        rep = {"ok": False, "out": sys.argv[3], "error": traceback.format_exc()[-3000:]}
    finally:
        grp.SetBool("SolveOnRecompute", had)
    write_report(rep, sys.argv[3])
    if not rep["ok"]:
        raise SystemExit("the oracle document did not build cleanly")
