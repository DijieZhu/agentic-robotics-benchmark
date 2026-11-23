"""Build the oracle deliverable: the native FreeCAD document of the reference hand (the hand as modelled).

forked from tasks/ground/racecar_cad_design_from_photos/solution/build_oracle.py (benchmark main 850dee3, sha256
fc0007d73414b103bd317f92209a054145013f01b1390b1d569bf2c720d929da), with the Revolute joints and limits of the quadruped
task's oracle builder (sha256 e0e025f6eb3228649f5e970d86ea62b0e252d9738bc5880fd363f99972ffc605). Changes: the document is Hand; the
plan is hand-oracle/1 (below), whose made parts are per-kind BREPs in the kind's own frame with each tag's pose; one
Fixed joint from every member of a link to that link's anchor part (not to the ground); one Revolute joint per hinge of
the hand between the anchors of its two links, with detached frames written explicitly at the hinge (z along its
positive sense) and both angle limits enabled on the independent hinges (the loops' pin hinges carry none); the
displacement line is 1e-6 mm; the joints are given as a list of plain records first (plan_joints), so that a QC
document can edit them before they are made. Nothing is posed by FreeCAD's solver: every placement is written, the
solver runs only to show that it keeps them.

Runs under FreeCAD 1.1.4's headless interpreter (the task image's /opt/freecad/usr/bin/freecadcmd), from solve.sh:

    freecadcmd -c "import sys; sys.argv = ['build_oracle.py', REF_DIR, HOME_DIR, OUT_FCSTD(, MODE)]; exec(open(PATH).read())"

REF_DIR = solution/reference: oracle.json, format hand-oracle/1, in one of two forms (normalize_plan):
  mapping form
    ground      the tag of the palm's largest part (the Grounded joint's component)
    parts       {tag: {brep (relative to REF_DIR: the kind's solid in its own frame), local_to_world 4x4}}
    placements  [{id, type, pn, step (relative to HOME_DIR), local_to_world 4x4}]
    links       {link: {anchor (a tag or a placement id), members [tags and placement ids]}}
    hinges      [{id, parent, child (links), centre_mm, axis (unit, the hinge's positive sense), independent,
                 limits_deg [lo, hi] (independent hinges) or null}]
  list form (the reference build's): ground; parts [{id, brep, kind, link, placement}]; given [{id, type, step,
    placement, link}]; anchors {link: anchor}; fixed [{link, anchor, member}]; hinges [{id, parent, child, point_mm,
    dir, independent, limits_rad, parent_anchor, child_anchor}].
HOME_DIR = the agent home, where setup.sh staged the payload (purchased/<PN>/<PN>.step). MODE "compound" (default):
one Part::Feature per given instance holding its STEP's solids; "insert": FreeCAD's own STEP import (Import.insert) into
an App::Part per instance. The document holds one PartDesign Body per tag (labelled with the tag; its base feature the
kind's exact BREP; a solid that carries a top-level location goes in as its raw geometry and the location becomes part
of the Body's placement), every given instance, one Assembly of App::Links with the Grounded joint, the Fixed joints and
the Revolutes. The Assembly is recomputed and solved, every component's placement compared with the one it was given
(nothing may move), the document saved, reopened and checked again. Prints ORACLE-BUILD <json> and writes the same
report to OUT_FCSTD.report.json.
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

JOINT_TYPES = list(JointObject.JointTypes)
MOVE_TOL_MM = 1e-6
MOVE_TOL_RAD = 1e-8


def placement(m):
    return App.Placement(App.Matrix(*[float(v) for row in m for v in row]))


def frame_z(origin, z):
    """A placement with its origin at `origin` and its z axis along `z` (x: the world axis least parallel to z, made
    orthogonal)."""
    z = App.Vector(*[float(v) for v in z])
    z.normalize()
    ref = App.Vector(1, 0, 0) if abs(z.x) < 0.9 else App.Vector(0, 1, 0)
    x = ref - z * ref.dot(z)
    x.normalize()
    y = z.cross(x)
    m = App.Matrix(x.x, y.x, z.x, float(origin[0]), x.y, y.y, z.y, float(origin[1]), x.z, y.z, z.z, float(origin[2]),
                   0, 0, 0, 1)
    return App.Placement(m)


def safe_name(s):
    return "".join(ch if ch.isalnum() else "_" for ch in str(s))


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


def normalize_plan(plan):
    """One form of the plan for the builder. The reference build may write hand-oracle/1 in its list form: parts
    [{id, brep, kind, link, placement}], given [{id, type, step, placement, link}], anchors {link: tag}, fixed
    [{link, anchor, member}], hinges [{id, parent, child, point_mm, dir, independent, limits_rad}]; it becomes the
    mapping form of the module docstring."""
    if plan.get("format", plan.get("schema")) != "hand-oracle/1":
        raise ValueError("oracle.json format %r" % plan.get("format", plan.get("schema")))
    if isinstance(plan.get("parts"), dict) and "links" in plan:
        return plan
    out = {"format": "hand-oracle/1", "ground": plan["ground"]}
    out["parts"] = {p["id"]: {"brep": p["brep"], "local_to_world": p["placement"]} for p in plan["parts"]}
    out["placements"] = [{"id": g["id"], "type": g["type"], "step": g["step"], "local_to_world": g["placement"],
                          "pn": g.get("pn") or os.path.splitext(os.path.basename(g["step"]))[0]} for g in plan["given"]]
    links = {name: {"anchor": a, "members": [a]} for name, a in plan["anchors"].items()}
    for f in plan.get("fixed", []):
        if f["anchor"] != links[f["link"]]["anchor"]:
            raise ValueError("fixed member %s: anchor %s is not its link's" % (f["member"], f["anchor"]))
        links[f["link"]]["members"].append(f["member"])
    out["links"] = links
    out["hinges"] = []
    for h in plan["hinges"]:
        lim = h.get("limits_rad")
        out["hinges"].append({"id": h["id"], "parent": h["parent"], "child": h["child"], "centre_mm": h["point_mm"],
                              "axis": h["dir"], "independent": bool(h.get("independent")),
                              "limits_deg": [math.degrees(float(x)) for x in lim] if (lim and h.get("independent")) else None})
        for side in ("parent", "child"):
            if h.get(side + "_anchor") and h[side + "_anchor"] != links[h[side]]["anchor"]:
                raise ValueError("hinge %s: its %s anchor is not the link's" % (h["id"], side))
    return out


def plan_joints(plan):
    """The oracle's joints as plain records: the Grounded joint, a Fixed joint from every member of a link to its
    anchor (frame at the member's own placement), a Revolute per hinge between the two anchors (frame at the hinge).
    world = the joint's frame in the world (both frames coincide at configuration Z)."""
    W = {t: p["local_to_world"] for t, p in plan["parts"].items()}
    W.update({pl["id"]: pl["local_to_world"] for pl in plan["placements"]})
    out = [{"name": "Grounded_" + safe_name(plan["ground"]), "type": "Grounded", "a": plan["ground"]}]
    for link in sorted(plan["links"]):
        L = plan["links"][link]
        for m in sorted(L["members"]):
            if m == L["anchor"]:
                continue
            out.append({"name": "Fixed_" + safe_name(m), "type": "Fixed", "a": L["anchor"], "b": m,
                        "world": placement(W[m])})
    for h in plan["hinges"]:
        rec = {"name": "Revolute_" + safe_name(h["id"]), "type": "Revolute", "a": plan["links"][h["parent"]]["anchor"],
               "b": plan["links"][h["child"]]["anchor"], "world": frame_z(h["centre_mm"], h["axis"]), "hinge": h["id"]}
        if h.get("independent") and h.get("limits_deg"):
            rec["limits_deg"] = [float(h["limits_deg"][0]), float(h["limits_deg"][1])]
        out.append(rec)
    return out


def make_objects(doc, plan, ref_dir, home_dir, mode, moves=None, drop=()):
    """The Bodies and the given instances; moves = {id: Placement applied in the world} (QC documents)."""
    moves = moves or {}
    objs = {}
    for tag in sorted(plan["parts"]):
        if tag in drop:
            continue
        p = plan["parts"][tag]
        shape = Part.read(os.path.join(ref_dir, p["brep"]))
        loc = shape.Placement
        shape.Placement = App.Placement()
        base = doc.addObject("Part::Feature", safe_name(tag) + "_base")
        base.Label = tag + " original solid"
        base.Shape = shape
        body = doc.addObject("PartDesign::Body", safe_name(tag))
        body.Label = tag
        body.BaseFeature = base
        body.Placement = moves.get(tag, App.Placement()).multiply(placement(p["local_to_world"]).multiply(loc))
        objs[tag] = body
    shapes = {}
    for pl in plan["placements"]:
        if pl["id"] in drop:
            continue
        step = os.path.join(home_dir, pl["step"])
        Wp = moves.get(pl["id"], App.Placement()).multiply(placement(pl["local_to_world"]))
        name = "Given_" + safe_name(pl["id"])
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
            part.Placement = Wp
            objs[pl["id"]] = part
            continue
        if pl["type"] not in shapes:
            shapes[pl["type"]] = Part.read(step)
        src = shapes[pl["type"]]
        f = doc.addObject("Part::Feature", name)
        f.Label = label
        f.Shape = src
        f.Placement = Wp.multiply(src.Placement)
        objs[pl["id"]] = f
    doc.recompute()
    return objs


def make_joint(jg, links, rec):
    """One joint of a record (detached frames; two subreference slots, as FreeCAD 1.1's restore migration expects)."""
    if rec["type"] == "Grounded":
        g = jg.newObject("App::FeaturePython", rec["name"])
        JointObject.GroundedJoint(g, links[rec["a"]])
        return g
    a, b = links[rec["a"]], links[rec["b"]]
    j = jg.newObject("App::FeaturePython", rec["name"])
    JointObject.Joint(j, JOINT_TYPES.index(rec["type"]))
    j.Detach1 = j.Detach2 = True
    j.Reference1 = (a, ["", ""])
    j.Reference2 = (b, ["", ""])
    w1 = rec["world"]
    w2 = rec.get("world2", w1)
    j.Placement1 = a.Placement.inverse().multiply(w1)
    j.Placement2 = b.Placement.inverse().multiply(w2)
    lim = rec.get("limits_deg")
    if lim is not None and rec["type"] in ("Revolute", "Cylindrical"):
        j.EnableAngleMin = j.EnableAngleMax = True
        j.AngleMin, j.AngleMax = float(lim[0]), float(lim[1])
    if rec.get("suppressed"):
        j.Suppressed = True
    return j


def build(ref_dir, home_dir, out_path, mode="compound", edit=None):
    """edit(plan, objs, joints) -> (joints, extra report) is a QC document's hook; the oracle passes none."""
    t0 = time.time()
    with open(os.path.join(ref_dir, "oracle.json")) as fh:
        plan = normalize_plan(json.load(fh))
    doc = App.newDocument("Hand")
    hook = edit or {}
    objs = make_objects(doc, plan, ref_dir, home_dir, mode, moves=hook.get("moves"), drop=set(hook.get("drop", [])))
    joints = plan_joints(plan)
    extra = {}
    if hook.get("joints"):
        joints, extra = hook["joints"](plan, objs, joints)
    asm = doc.addObject("Assembly::AssemblyObject", "Assembly")
    jg = asm.newObject("Assembly::JointGroup", "Joints")
    links, intended = {}, {}
    outside = set(hook.get("outside", []))
    for key in sorted(objs):
        if key in outside:
            continue
        obj = objs[key]
        lk = asm.newObject("App::Link", "Link_" + safe_name(key))
        lk.LinkedObject = obj
        lk.Placement = obj.Placement
        lk.Label = obj.Label if key not in plan["parts"] else key + " instance"
        links[key] = lk
        intended[key] = App.Placement(obj.Placement.toMatrix())
    n = {"Grounded": 0, "Fixed": 0, "Revolute": 0, "Cylindrical": 0, "Ball": 0}
    for rec in joints:
        if rec["a"] not in links or (rec["type"] != "Grounded" and rec["b"] not in links):
            continue
        make_joint(jg, links, rec)
        n[rec["type"]] = n.get(rec["type"], 0) + 1
    doc.recompute()
    d_recompute = deviation(links, intended)
    solve_rc = asm.solve()
    d_solve = deviation(links, intended)
    doc.recompute()
    d_saved = deviation(links, intended)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)
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
    # a Body is bad unless it is exactly one closed solid (no validity check: the given solids fail BRepCheck)
    bad = [o.Label for o in bodies if len(o.Shape.Solids) != 1 or not o.Shape.Solids[0].isClosed()]
    empty = sorted(k for k, lk in links2.items() if lk is None or Part.getShape(lk).isNull())
    App.closeDocument(doc2.Name)
    moved = [d for d in (d_recompute, d_solve, d_saved, d_reopen) if d[0] > MOVE_TOL_MM or d[1] > MOVE_TOL_RAD]
    rep = {"ok": not moved and solve_rc == 0 and not bad and not empty and asm2 is not None, "out": out_path,
           "mode": mode, "bodies": len(bodies), "purchased_instances": len(plan["placements"]), "joints": n,
           "solve_rc": solve_rc, "moved_after_recompute": list(d_recompute), "moved_after_solve": list(d_solve),
           "moved_as_saved": list(d_saved), "moved_after_reopen": list(d_reopen), "invalid_bodies": bad,
           "components_without_shape": empty, "bytes": os.path.getsize(out_path), "build_s": t_save - t0,
           "total_s": time.time() - t0, "freecad": ".".join(App.Version()[:3])}
    rep.update(extra)
    return rep


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
