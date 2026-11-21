"""Build the oracle deliverable: a native FreeCAD document of the cabin free-flyer made of the maker's own triangles.

Runs under FreeCAD 1.1.4's headless interpreter (the task image's /opt/freecad/usr/bin/freecadcmd, its own Python 3.11
with numpy), from solve.sh; FreeCAD's own modules, numpy and the standard library only:

    build_oracle.py REF_DIR OUT_FCSTD [FORM]

REF_DIR = solution/reference: oracle.json (the tags and their maker files, the pins of the 12 maker files, the skin
parameters, the patch selection, the expected counts and the joint table), placement.json (the maker description's
placement of every tag in configuration Z and its joints, derived from the description's text) and maker/ (byte copies
of the maker's meshes and description; three meshes stored as xz streams). The triangles are read by makerstd/ (the
task's own reader, a byte copy) and made into solids by refsolids.py (the reference rule: closed patches as they are,
every other orientable patch a 0.3 mm skin, +-0.15 mm along its vertex normals; one planar face per triangle; nothing
simplified, smoothed, re-meshed or added). Nothing of the robot is modelled here.

The document holds:
  - one Assembly member per tag, labelled with the tag, holding all of its solids, placed at the description's place
    (a placement, not a copy): FORM "A" (default) one Part::Feature holding the compound of the tag's solids, "B" one
    App::Part holding one Part::Feature per solid;
  - one Assembly: a Grounded joint on P01, a detached Fixed joint from P01 to each of P02..P06 (frames at the body
    origin), and the description's six Revolute joints, each from its parent tag (Reference1) to its child tag
    (Reference2), both detached frames at the description's joint frame (origin = the joint origin, z axis = the joint
    axis), no limits; two subreference slots ["", ""] (skill pitfall 75).
When the GUI is up the joints get FreeCAD's own view providers (skill pitfall 79). The document is recomputed and
solved, every member's placement compared with the one it was given (nothing may move more than 1e-6 mm or 1e-9 rad),
recomputed and checked as saved, saved, reopened and checked, solved again and checked again; on the reopened document
every member, solid, face, joint and joint axis is counted and compared with oracle.json and placement.json. Every input
file is checked against its pin first. Prints one flushed line ORACLE-BUILD <json>.

Each step is a function with the three geometry choices as parameters (member form, skin offsets, patch selection; the
defaults are the reference's, read from oracle.json), so a variant document can be put together from these functions:
load_spec, read_triangles, build_solids, build_members, build_assembly (add_grounded, add_fixed, add_revolute,
revolute_frames), save_and_check, build.
"""
import hashlib
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(globals().get("__file__") or sys.argv[0]))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402

try:  # FreeCAD's own modules (absent when the pure functions are tested outside FreeCAD)
    import FreeCAD as App  # noqa: E402
    import Part  # noqa: E402
except ImportError:  # pragma: no cover - outside FreeCAD
    App = Part = None

import refsolids  # noqa: E402
from makerstd import dae as std_dae  # noqa: E402
from makerstd import fk as std_fk  # noqa: E402
from makerstd import pinned as std_pinned  # noqa: E402
from makerstd import standard as std_standard  # noqa: E402

SCHEMA = "ffcad.oracle/1"
MOVE_TOL_MM = 1e-6
MOVE_TOL_RAD = 1e-9
AXIS_TOL_MM = 1e-6
AXIS_TOL_DEG = 1e-6
FORMS = ("A", "B")


class OracleError(RuntimeError):
    """An input does not agree with its pin or with oracle.json."""


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    with open(path, "rb") as fh:
        return sha256_bytes(fh.read())


# ------------------------------------------------------------------------------------------------ inputs
def load_spec(ref_dir):
    """oracle.json of REF_DIR, its schema checked, and placement.json checked against the sha256 it records."""
    with open(os.path.join(ref_dir, "oracle.json"), "rb") as fh:
        spec = json.loads(fh.read().decode("utf-8"))
    if spec.get("schema") != SCHEMA:
        raise OracleError("oracle.json is not a %s file" % SCHEMA)
    pl_path = os.path.join(ref_dir, spec["placement"]["path"])
    if sha256_file(pl_path) != spec["placement"]["sha256"]:
        raise OracleError("placement.json is not the recorded file")
    return spec


def check_pins(ref_dir, spec):
    """{logical file: sha256 of the maker's bytes}: every file of the pins read (xz decompressed) and checked."""
    maker = os.path.join(ref_dir, "maker")
    try:
        return std_pinned.check_all(maker, spec["pins"])
    except Exception as exc:  # MakerStdError: a missing or changed file
        raise OracleError("a maker file is not the pinned file: %s" % exc)


def read_triangles(ref_dir, spec):
    """{tag: {"file", "V" (n, 3) mm in the maker file's frame, "F" (m, 3) rows = the file's triangles, "T_mm" 4 x 4
    body frame <- file frame at configuration Z (rotation as written, translation mm)}} plus the loaded placement.
    Files are read once; a pair's two tags share the arrays (they are not modified downstream)."""
    maker = os.path.join(ref_dir, "maker")
    std = std_standard.load(maker, os.path.join(ref_dir, spec["placement"]["path"]), pins=spec["pins"],
                            check_placement=False)
    out = {}
    for tag in sorted(spec["tags"]):
        f = spec["tags"][tag]["maker_file"]
        if std.maker_file(tag) != f:
            raise OracleError("%s: placement.json names %s, oracle.json %s" % (tag, std.maker_file(tag), f))
        m = std.mesh(f)
        out[tag] = {"file": f, "V": m.vertices_mm(), "F": m.F, "T_mm": std.T_body_file_mm(tag),
                    "flat_list_sha256": m.digest()}
    return out, std.placement


# ------------------------------------------------------------------------------------------------ solids
def _selection(select, tag, file):
    """refsolids.make_patches' select argument for one tag's file."""
    if select is None or select == "all":
        return None
    if callable(select):
        return lambda rec: select(tag, rec)
    if isinstance(select, dict) and "triangle_indices" in select:
        per_file = select["triangle_indices"]
        if file not in per_file:
            raise OracleError("the patch selection lists no triangles of %s" % file)
        return {"triangle_indices": per_file[file]}
    raise OracleError("unknown patch selection %r" % (select,))


def build_solids(tri, offsets=None, select="all", tags=None, log=None):
    """{tag: [(solid, record)]} and {tag: patch summary}: refsolids.make_patches then refsolids.build_solid per patch,
    tag by tag (the two tags of a pair are built separately). offsets: the skin's (outer, inner) mm, default the
    reference's; select: "all", {"triangle_indices": {file: [...]}} or a callable(tag, patch record)."""
    offsets = tuple(refsolids.REFERENCE_OFFSETS_MM if offsets is None else offsets)
    solids, summaries = {}, {}
    for tag in (tags or sorted(tri)):
        t0 = time.perf_counter()
        d = tri[tag]
        pats, summ = refsolids.make_patches(d["V"], d["F"], offsets=offsets, select=_selection(select, tag, d["file"]))
        lst = []
        for k, p in enumerate(pats):
            so, rec = refsolids.build_solid(p["V"], p["F"])
            rec.update(tag=tag, index=k, patch=int(p["patch"]), kind=p["kind"],
                       source_triangles=int(len(p["source_triangles"])))
            rec["ok"] = bool(rec["valid"] and rec["closed"] and rec["volume_mm3"] > 0.0 and
                             rec["faces"] == rec["triangles"])
            lst.append((so, rec))
        solids[tag] = lst
        summ["t_s"] = time.perf_counter() - t0
        summaries[tag] = summ
        if log:
            log("solids %s: %d (%d ok), %d faces, %.1f s" % (tag, len(lst), sum(r["ok"] for _s, r in lst),
                                                            sum(r["faces"] for _s, r in lst), summ["t_s"]))
    return solids, summaries


# ------------------------------------------------------------------------------------------------ members
def placement_of(T):
    """App.Placement of a 4 x 4 nested list or array (translation in mm)."""
    return App.Placement(App.Matrix(*[float(v) for row in T for v in row]))


def build_members(doc, solids, tri, form="A"):
    """{tag: member} in doc: FORM "A" a Part::Feature (Name and Label = the tag) holding Part.Compound of the tag's
    solids, "B" an App::Part (Name and Label = the tag) holding one Part::Feature per solid; the member's Placement is
    the tag's description placement (the solids stay in the maker file's frame)."""
    if form not in FORMS:
        raise OracleError("unknown member form %r" % (form,))
    members = {}
    for tag in sorted(solids):
        shapes = [s for s, _r in solids[tag]]
        if form == "A":
            m = doc.addObject("Part::Feature", tag)
            m.Label = tag
            m.Shape = Part.Compound(shapes)
        else:
            m = doc.addObject("App::Part", tag)
            m.Label = tag
            feats = []
            for k, s in enumerate(shapes):
                f = doc.addObject("Part::Feature", "%s_%04d" % (tag, k))
                f.Label = "%s.%04d" % (tag, k)
                f.Shape = s
                feats.append(f)
            m.addObjects(feats)
        m.Placement = placement_of(tri[tag]["T_mm"])
        members[tag] = m
    return members


# ------------------------------------------------------------------------------------------------ joints
def _basis_z(a):
    """3 x 3 rotation (columns x, y, z) whose z column is the unit vector a."""
    z = std_fk.unit(a)
    x = [1.0, 0.0, 0.0] if abs(z[0]) < 0.9 else [0.0, 1.0, 0.0]
    d = x[0] * z[0] + x[1] * z[1] + x[2] * z[2]
    x = std_fk.unit([x[0] - d * z[0], x[1] - d * z[1], x[2] - d * z[2]])
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    return [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]


def revolute_frames(placement, spec):
    """{joint name: {"part1", "part2", "frame_mm" 4 x 4 in the body frame (mm), "point_mm", "dir"}}: the joint frame
    of the description in configuration Z (its parent link's frame times the joint origin, turned so that z is the
    joint axis), checked against the axis line placement.json records (1e-9 mm, 1e-12)."""
    joints = {j["name"]: j for j in placement["joints"]}
    out = {}
    for r in spec["joints"]["revolute"]:
        j = joints[r["joint"]]
        if j["type"]["text"] != "revolute" or list(j["tags"]) != [r["part1"], r["part2"]]:
            raise OracleError("joint %s is not the revolute joint %s -> %s" % (r["joint"], r["part1"], r["part2"]))
        Tp = placement["links_at_Z"][j["parent_link"]]
        Tj = std_fk.mul(Tp, std_fk.origin(j["origin_xyz"]["value"], j["origin_rpy"]["value"]))
        Q = _basis_z(j["axis"]["value"])
        R = [[sum(Tj[i][k] * Q[k][c] for k in range(3)) for c in range(3)] for i in range(3)]
        F = [[R[0][0], R[0][1], R[0][2], Tj[0][3] * 1000.0], [R[1][0], R[1][1], R[1][2], Tj[1][3] * 1000.0],
             [R[2][0], R[2][1], R[2][2], Tj[2][3] * 1000.0], [0.0, 0.0, 0.0, 1.0]]
        line = j["axis_line_at_Z"]
        pt = [F[0][3], F[1][3], F[2][3]]
        dr = [F[0][2], F[1][2], F[2][2]]
        if max(abs(a - b) for a, b in zip(pt, line["point_mm"])) > 1e-9 or \
                max(abs(a - b) for a, b in zip(dr, line["dir"])) > 1e-12:
            raise OracleError("joint %s: frame %r / %r is not placement.json's axis line" % (r["joint"], pt, dr))
        out[r["joint"]] = {"part1": r["part1"], "part2": r["part2"], "frame_mm": F, "point_mm": pt, "dir": dr}
    return out


def _joint_type(name):
    import JointObject
    return list(JointObject.JointTypes).index(name)


def add_grounded(jg, member):
    import JointObject
    g = jg.newObject("App::FeaturePython", "GroundedJoint")
    JointObject.GroundedJoint(g, member)
    return g


def add_fixed(jg, a, b, name, frame=None):
    """A detached Fixed joint a -> b, both frames at the given placement in the document (default: the origin)."""
    import JointObject
    j = jg.newObject("App::FeaturePython", name)
    JointObject.Joint(j, _joint_type("Fixed"))
    j.Detach1 = j.Detach2 = True
    # FreeCAD 1.1's restore migration reads two subreference slots, even for detached frames (pitfall 75)
    j.Reference1 = (a, ["", ""])
    j.Reference2 = (b, ["", ""])
    w = App.Placement() if frame is None else frame
    j.Placement1 = a.Placement.inverse().multiply(w)
    j.Placement2 = b.Placement.inverse().multiply(w)
    return j


def add_revolute(jg, a, b, name, frame):
    """A detached Revolute joint a -> b without limits, both frames at the given placement in the document."""
    import JointObject
    j = jg.newObject("App::FeaturePython", name)
    JointObject.Joint(j, _joint_type("Revolute"))
    j.Detach1 = j.Detach2 = True
    j.Reference1 = (a, ["", ""])
    j.Reference2 = (b, ["", ""])
    j.Placement1 = a.Placement.inverse().multiply(frame)
    j.Placement2 = b.Placement.inverse().multiply(frame)
    return j


def view_providers(joints, grounded):
    """FreeCAD's own joint view providers (skill pitfall 79); only possible with the GUI up."""
    if not App.GuiUp:
        return False
    import JointObject
    for j in joints:
        JointObject.ViewProviderJoint(j.ViewObject)
        if not hasattr(j.ViewObject.Proxy, "switch_JCS1"):          # not attached yet: no coin nodes
            j.ViewObject.Proxy.attach(j.ViewObject)
    JointObject.ViewProviderGroundedJoint(grounded.ViewObject)
    if not hasattr(grounded.ViewObject.Proxy, "app_obj"):
        grounded.ViewObject.Proxy.attach(grounded.ViewObject)
    return True


def build_assembly(doc, members, spec, frames):
    """(assembly, grounded joint, [fixed and revolute joints]): the members put in one Assembly (in tag order), the
    Grounded joint on oracle.json's grounded tag, its Fixed joints and the Revolute joints at their frames."""
    asm = doc.addObject("Assembly::AssemblyObject", "Assembly")
    for tag in sorted(members):
        asm.addObject(members[tag])
    jg = asm.newObject("Assembly::JointGroup", "Joints")
    J = spec["joints"]
    g = add_grounded(jg, members[J["grounded"]])
    joints = []
    for f in J["fixed"]:
        joints.append(add_fixed(jg, members[f["part1"]], members[f["part2"]], "Fixed_%s" % f["part2"]))
    for r in J["revolute"]:
        fr = frames[r["joint"]]
        joints.append(add_revolute(jg, members[fr["part1"]], members[fr["part2"]],
                                   "Revolute_%s_%s" % (fr["part1"], fr["part2"]), placement_of(fr["frame_mm"])))
    return asm, g, joints


# ------------------------------------------------------------------------------------------------ checks
def rotation_angle(r):
    """The angle of a rotation, accurate near zero: 2 atan2(|q_xyz|, |q_w|)."""
    x, y, z, w = r.Q
    return 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), abs(w))


def deviation(objs, intended):
    """[largest translation mm, largest rotation rad, the tag of the first, the tag of the second] over the
    members."""
    worst = [0.0, 0.0, None, None]
    for key in sorted(objs):
        p, q = objs[key].Placement, intended[key]
        d = (p.Base - q.Base).Length
        a = rotation_angle(q.Rotation.inverted().multiply(p.Rotation))
        if d > worst[0]:
            worst[0], worst[2] = d, key
        if a > worst[1]:
            worst[1], worst[3] = a, key
    return worst


def joint_kinds(doc):
    kinds, rev = {}, []
    for o in doc.Objects:
        jt = getattr(o, "JointType", None)
        if jt is not None:
            kinds[str(jt)] = kinds.get(str(jt), 0) + 1
            if str(jt) == "Revolute":
                rev.append(o)
        elif hasattr(o, "ObjectToGround"):
            kinds["Grounded"] = kinds.get("Grounded", 0) + 1
    return kinds, rev


def axis_errors(rev_joints, frames):
    """Per Revolute joint of the reopened document: its two members' Labels and, for both joint frames, the distance
    of the description's joint origin from the frame's z line (mm) and the angle between the z axes (deg, direction
    free); also the worst of each."""
    import UtilsAssembly
    by_pair = {tuple(sorted((f["part1"], f["part2"]))): f for f in frames.values()}
    out, worst_mm, worst_deg = [], 0.0, 0.0
    for j in rev_joints:
        rec = {"name": j.Name}
        parts = []
        for k in ("1", "2"):
            ref = getattr(j, "Reference" + k)
            part = UtilsAssembly.getMovingPart(ref)
            parts.append(part.Label if part is not None else None)
        rec["parts"] = parts
        f = by_pair.get(tuple(sorted(p or "" for p in parts)))
        if f is None:
            rec["error"] = "not a pair of the description"
            out.append(rec)
            worst_mm, worst_deg = math.inf, math.inf
            continue
        P = np.asarray(f["point_mm"], float)
        D = np.asarray(f["dir"], float)
        for k in ("1", "2"):
            g = UtilsAssembly.getJcsGlobalPlc(getattr(j, "Placement" + k), getattr(j, "Reference" + k))
            o = np.array([g.Base.x, g.Base.y, g.Base.z])
            zv = g.Rotation.multVec(App.Vector(0, 0, 1))
            z = np.array([zv.x, zv.y, zv.z])
            z = z / np.linalg.norm(z)
            v = P - o
            off = float(np.linalg.norm(v - np.dot(v, z) * z))
            c = min(1.0, abs(float(np.dot(z, D))))
            ang = math.degrees(math.atan2(float(np.linalg.norm(np.cross(z, D))), c))
            rec["offset_mm_" + k], rec["angle_deg_" + k] = off, ang
            worst_mm, worst_deg = max(worst_mm, off), max(worst_deg, ang)
        out.append(rec)
    return out, worst_mm, worst_deg


def save_and_check(doc, asm, members, intended, frames, out_path, spec, expect=None):
    """recompute -> check; solve -> check; recompute -> check (as saved); save, close, reopen, recompute -> check;
    solve -> check; then count the reopened document. Returns the report (no "ok" yet) and the problems found."""
    names = {k: v.Name for k, v in members.items()}
    asm_name = asm.Name
    rep, problems = {}, []
    doc.recompute()
    rep["moved_after_recompute"] = deviation(members, intended)
    rep["solve_rc"] = asm.solve()
    rep["moved_after_solve"] = deviation(members, intended)
    doc.recompute()
    rep["moved_as_saved"] = deviation(members, intended)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    t0 = time.perf_counter()
    doc.saveAs(out_path)
    rep["save_s"] = time.perf_counter() - t0
    App.closeDocument(doc.Name)
    t1 = time.perf_counter()
    doc2 = App.openDocument(out_path)
    doc2.recompute()
    asm2 = doc2.getObject(asm_name)
    m2 = {k: doc2.getObject(v) for k, v in names.items()}
    rep["moved_after_reopen"] = deviation(m2, intended)
    rep["solve_rc_after_reopen"] = asm2.solve() if asm2 is not None else None
    rep["moved_after_reopen_solve"] = deviation(m2, intended)
    # counts on the reopened document
    labels = sorted(o.Label for o in asm2.Group if o.Name in names.values()) if asm2 is not None else []
    rep["labels"] = labels
    per_tag = {}
    for tag, m in m2.items():
        sols = Part.getShape(m).Solids
        per_tag[tag] = {"label": m.Label, "solids": len(sols), "faces": sum(len(s.Faces) for s in sols),
                        "valid_closed": sum(1 for s in sols if s.isValid() and s.isClosed())}
    rep["per_tag"] = per_tag
    rep["solids"] = sum(v["solids"] for v in per_tag.values())
    rep["faces"] = sum(v["faces"] for v in per_tag.values())
    rep["valid_closed_solids"] = sum(v["valid_closed"] for v in per_tag.values())
    kinds, rev = joint_kinds(doc2)
    rep["joints"] = kinds
    rep["joints_wanted"] = dict(spec["joints"]["counts"])
    rep["revolute_axes"], rep["axis_offset_max_mm"], rep["axis_angle_max_deg"] = axis_errors(rev, frames)
    g = [o for o in doc2.Objects if hasattr(o, "ObjectToGround")]
    rep["grounded"] = sorted(o.ObjectToGround.Label for o in g if o.ObjectToGround is not None)
    App.closeDocument(doc2.Name)
    rep["reopen_check_s"] = time.perf_counter() - t1
    # the problems
    for key in ("moved_after_recompute", "moved_after_solve", "moved_as_saved", "moved_after_reopen",
                "moved_after_reopen_solve"):
        d = rep[key]
        if not (d[0] <= MOVE_TOL_MM and d[1] <= MOVE_TOL_RAD):
            problems.append("%s: %.3g mm / %.3g rad (%s)" % (key, d[0], d[1], d[2]))
    if rep["solve_rc"] != 0 or rep["solve_rc_after_reopen"] != 0:
        problems.append("solve returned %r / %r" % (rep["solve_rc"], rep["solve_rc_after_reopen"]))
    if labels != sorted(spec["tags"]):
        problems.append("the Assembly's members are labelled %s" % labels)
    if any(v["label"] != t for t, v in per_tag.items()):
        problems.append("a member's Label is not its tag")
    if rep["valid_closed_solids"] != rep["solids"]:
        problems.append("%d of %d solids are not valid and closed" % (rep["solids"] - rep["valid_closed_solids"],
                                                                       rep["solids"]))
    if kinds != rep["joints_wanted"]:
        problems.append("joints %s, wanted %s" % (kinds, rep["joints_wanted"]))
    if rep["grounded"] != [spec["joints"]["grounded"]]:
        problems.append("grounded %s" % rep["grounded"])
    if not (rep["axis_offset_max_mm"] <= AXIS_TOL_MM and rep["axis_angle_max_deg"] <= AXIS_TOL_DEG):
        problems.append("joint axes off by %.3g mm / %.3g deg" % (rep["axis_offset_max_mm"], rep["axis_angle_max_deg"]))
    if expect is not None:
        for tag, v in per_tag.items():
            e = expect[tag]
            if v["solids"] != e["solids"] or v["faces"] != e["faces"]:
                problems.append("%s: %d solids / %d faces, oracle.json %d / %d" % (tag, v["solids"], v["faces"],
                                                                                     e["solids"], e["faces"]))
    return rep, problems


# ------------------------------------------------------------------------------------------------ the whole build
def build(ref_dir, out_path, form=None, offsets=None, select=None, log=None):
    """The reference document (defaults from oracle.json) saved at out_path; returns the ORACLE-BUILD report."""
    t0 = time.time()
    log = log or (lambda s: print(s, flush=True))
    spec = load_spec(ref_dir)
    form = form or spec["member_form"]
    offsets = tuple(spec["skin"]["offsets_mm"] if offsets is None else offsets)
    select = spec["patch_selection"] if select is None else select
    pins_checked = check_pins(ref_dir, spec)
    t_pins = time.time() - t0
    tri, placement = read_triangles(ref_dir, spec)
    problems = []
    for tag, d in tri.items():
        if d["flat_list_sha256"] != spec["files"][d["file"]]["flat_list_sha256"]:
            problems.append("%s: the file's triangle list is not the recorded one" % tag)
    t_read = time.time() - t0 - t_pins
    frames = revolute_frames(placement, spec)
    t1 = time.time()
    solids, summaries = build_solids(tri, offsets=offsets, select=select, log=log)
    t_solids = time.time() - t1
    reference_rule = select == "all" and offsets == tuple(spec["skin"]["offsets_mm"])
    for tag, s in summaries.items():
        f = spec["files"][tri[tag]["file"]]
        if select == "all" and (s["collapsed_triangles"] != f["collapsed_by_weld"] or s["patches"] != f["patches"]):
            problems.append("%s: %d collapsed / %d patches, oracle.json %d / %d" % (
                tag, s["collapsed_triangles"], s["patches"], f["collapsed_by_weld"], f["patches"]))
    bad = [r for lst in solids.values() for _s, r in lst if not r["ok"]]
    if bad:
        problems.append("%d solids not valid, closed, positive and one face per triangle" % len(bad))
    t2 = time.time()
    doc = App.newDocument("Unit")
    members = build_members(doc, solids, tri, form=form)
    intended = {k: App.Placement(m.Placement.toMatrix()) for k, m in members.items()}
    asm, g, joints = build_assembly(doc, members, spec, frames)
    gui_vp = view_providers(joints, g)
    t_doc = time.time() - t2
    rep, probs = save_and_check(doc, asm, members, intended, frames, out_path, spec,
                                expect=spec["tags"] if reference_rule else None)
    problems += probs
    moves = [rep[k] for k in ("moved_after_recompute", "moved_after_solve", "moved_as_saved", "moved_after_reopen",
                              "moved_after_reopen_solve")]
    rep.update({
        "ok": not problems, "problems": problems, "out": out_path, "member_form": form, "offsets_mm": list(offsets),
        "patch_selection": select if isinstance(select, str) else "custom", "reference_rule": reference_rule,
        "members": len(members), "solids_built": sum(len(v) for v in solids.values()),
        "solids_built_ok": sum(r["ok"] for lst in solids.values() for _s, r in lst),
        "solids_reversed_by_build": sum(r["reversed"] for lst in solids.values() for _s, r in lst),
        "max_move_mm": max(d[0] for d in moves), "max_move_rad": max(d[1] for d in moves),
        "move_tolerance_mm_rad": [MOVE_TOL_MM, MOVE_TOL_RAD], "axis_tolerance_mm_deg": [AXIS_TOL_MM, AXIS_TOL_DEG],
        "gui_view_providers": gui_vp, "pins_checked": len(pins_checked), "bytes": os.path.getsize(out_path),
        "t_pins_s": t_pins, "t_read_s": t_read, "t_solids_s": t_solids, "t_doc_s": t_doc, "total_s": time.time() - t0,
        "freecad": ".".join(App.Version()[:3]), "python": sys.version.split()[0], "numpy": np.__version__,
        "numpy_path": os.path.dirname(np.__file__)})
    return rep


def main(argv):
    ref_dir, out_path = argv[1], argv[2]
    form = argv[3] if len(argv) > 3 and argv[3] else None
    try:
        rep = build(ref_dir, out_path, form=form)
    except Exception as exc:  # noqa: BLE001 - reported on the one line solve.sh reads
        import traceback
        rep = {"ok": False, "error": "%s: %s" % (type(exc).__name__, exc), "trace": traceback.format_exc()[-2000:]}
    print("ORACLE-BUILD " + json.dumps(rep, sort_keys=True, default=str), flush=True)   # flushed: SystemExit may skip it
    if not rep.get("ok"):
        raise SystemExit("the oracle document did not build cleanly")


if __name__ == "__main__":
    main(sys.argv)
