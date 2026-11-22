"""Read the submitted FreeCAD document into plain data (the grader's only contact with FreeCAD).

forked from tasks/quadruped/quadruped_cad_design_from_photos/tests/fcstd_extract.py (benchmark main 850dee3, sha256
f8a64c5ba2c9dac4e993bea10a5b42210dfddd16ea2f32471dc549aeea204d80: the Assembly tree, sub-assemblies flattened with
their inner joints, non-finite placements, the restore guard, the saved placements kept over the recompute) with the
output of tasks/ground/racecar_cad_design_from_photos/tests/fcstd_extract.py (sha256
51d991970562cc6e502942b20d6e2ec00a3011eda5aabb24c0d0b83c77eaa0d1: one exact BREP per solid in world coordinates, no
validity check, loose geometry, empty links). Changes, each marked "E<n>" where it is made: E1 the format name and
the final line; E2 no component classes in FreeCAD (the grader recognises the given parts by their shape), so no
STEP is written and nothing is collapsed; every solid of every leaf is exported as BREP (a STEP beside it can change a
solid's volume); E3 joints of every type are recorded: Grounded, Fixed, Revolute, Cylindrical (its angle limits and its
length limits), Ball, and any other type by its name; E4 the Assembly read is the one that holds the palm's largest
part (the leaf labelled with the palm tag), the others are reported; E5 a leaf whose link has no readable target is
marked empty_link and noted, not an error; E6 FreeCAD's solve-on-recompute is switched off while the document is read
(the saved placements are also restored after the recompute, as before); E7 no isValid() anywhere.

Runs under FreeCAD 1.1.4's headless interpreter as an unprivileged account (verify.py starts it with a time limit):

    freecadcmd -c "import sys; sys.argv = ['fcstd_extract.py', FCSTD, OUT_DIR]; exec(open(PATH).read())"

An optional third argument is a JSON object (or a file holding one) with "tag_pattern" and "palm_tag". OUT_DIR gets
model.json (format "hand-fcstd-extract/1") and shapes/<n>_<k>.brep, one per solid:

- components: the members of the Assembly; App::Part containers, nested Assemblies, AssemblyLinks and links to
  assemblies are flattened into their shape leaves ("containers" names the chain); folders (App::DocumentObjectGroup)
  and link arrays are transparent, as FreeCAD's Assembly treats their members as separate components.
- leaves: per shape-bearing leaf its node id (its subname path in the Assembly), labels (the linked object's and its
  own; "label" is the linked object's, or the leaf's own when several leaves share one linked label and the own label
  is a tag), type (TypeId of the linked object), body (when it is a PartDesign Body), placement (global 4x4),
  placement_finite, n_solids, n_free_faces / n_free_edges (geometry outside any solid), volume, and solids: per solid
  brep (world coordinates), closed (isClosed()), volume.
- containers: per flattened container its kind, its rigid flag, unit (its members form one rigid group: an App::Part,
  a rigid sub-assembly without inner joints), parent, children.
- joints: every joint of the Assembly and of its flattened sub-assemblies: type, suppressed, part1 / part2 (the tree
  nodes the references reach), jcs1 / jcs2 (global joint frames from FreeCAD's own UtilsAssembly), whether the
  joint's own placements are finite, the angle limits (enable flags and values, degrees) with the saved angle of a
  Revolute or Cylindrical joint (angle_saved_deg) and, for a Cylindrical joint, the length limits. joint_types_other lists the joints of a type the grader does not read.
- grounded: the Grounded joints (the node they ground).

Nothing in the document is executed beyond what FreeCAD runs to open and recompute it. Errors are recorded in
model.json, never raised ("errors": the document does not open, no Assembly, too many shapes; a leaf's "error"); a
model.json saying the extraction did not finish is written first; "extractor_failure" marks a fault of this reader.
"""
import json
import math
import os
import re
import sys
import time
import traceback

import FreeCAD as App
import Part

FORMAT = "hand-fcstd-extract/1"                                                    # E1
MAX_LEAVES = 4000
MAX_SOLIDS = 6000
JOINT_TYPES_READ = ("Fixed", "Revolute", "Cylindrical", "Ball")                  # E3
MOVED_TOL = 1e-6
TAG_PATTERN = r"^P(0[1-9]|[1-4][0-9]|5[0-2])$"
PALM_TAG = "P01"
TRANSPARENT = ("link_array",)    # containers whose members are separate components in FreeCAD's Assembly (not flagged)
SKIP_TYPES = ("Assembly::JointGroup", "Assembly::ViewGroup", "Assembly::BomGroup", "Assembly::SimulationGroup")
DATUM_PREFIXES = ("App::Origin", "App::Line", "App::Plane", "App::Point", "App::LocalCoordinateSystem",
                  "App::DatumElement")
PLC_KEYS = ("A11", "A12", "A13", "A14", "A21", "A22", "A23", "A24", "A31", "A32", "A33", "A34")


def matrix(placement):
    m = placement.toMatrix()
    return [[m.A11, m.A12, m.A13, m.A14], [m.A21, m.A22, m.A23, m.A24], [m.A31, m.A32, m.A33, m.A34], [0.0, 0.0, 0.0, 1.0]]


def placement_finite(plc):
    """True when every entry of a Placement's matrix is a finite number (None or no placement: True)."""
    if plc is None:
        return True
    try:
        m = plc.toMatrix()
        return all(math.isfinite(float(getattr(m, k))) for k in PLC_KEYS)
    except Exception:  # noqa: BLE001 - not a placement
        return True


def leaf_label(obj):
    target = obj.getLinkedObject(True) if hasattr(obj, "getLinkedObject") else obj
    return (target.Label or "").strip(), target


def is_joint_container(obj):
    return obj.TypeId in SKIP_TYPES or (
        obj.TypeId == "App::DocumentObjectGroup" and obj.Label.lower().startswith("joint")
        and all(is_joint(c) for c in getattr(obj, "Group", [])))


def is_joint(obj):
    return obj.TypeId == "App::FeaturePython" and (hasattr(obj, "JointType") or hasattr(obj, "ObjectToGround"))


def is_skipped(obj):
    return is_joint(obj) or is_joint_container(obj) or any(obj.TypeId.startswith(p) for p in DATUM_PREFIXES)


def is_link_group(obj):
    return obj.TypeId == "App::Link" and int(getattr(obj, "ElementCount", 0) or 0) > 0


def is_link(obj):
    return hasattr(obj, "LinkedObject")


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


# ---- the Assembly tree -------------------------------------------------------------------------------------------
def top_level_assemblies(doc):
    """Assemblies that are neither inside another Assembly nor the target of a link (a sub-assembly inserted by an
    AssemblyLink or an App::Link)."""
    asms = [o for o in doc.Objects if o.TypeId == "Assembly::AssemblyObject"]
    in_asm = ("Assembly::AssemblyObject", "Assembly::AssemblyLink")
    linked = set()
    for o in doc.Objects:
        lo = getattr(o, "LinkedObject", None)
        if lo is None or getattr(lo, "TypeId", "") != "Assembly::AssemblyObject":
            continue
        if o.TypeId == "Assembly::AssemblyLink" or any(p[0].TypeId in in_asm for p in (o.Parents or [])):
            linked.add(lo.Name)
    tops = []
    for a in asms:
        inside = any(p[0].TypeId in ("Assembly::AssemblyObject", "Assembly::AssemblyLink") for p in (a.Parents or []))
        if not inside and a.Name not in linked:
            tops.append(a)
    return asms, tops


def n_components(asm):
    return sum(1 for o in asm.Group if not is_skipped(o))


class Tree(object):
    """The flattened Assembly: leaves (shape-bearing objects), containers (sub-assemblies) and the object -> node map.
    A node id is the object's subname path in the Assembly without the trailing dot ("Link001", "Leg.P07")."""

    def __init__(self, asm):
        self.asm = asm
        self.direct = set(o.Name for o in asm.Group)
        self.leaves = []          # {node, path, obj, chain (enclosing container nodes), top (top-level component)}
        self.containers = {}      # node -> {kind, rigid, unit, parent, children, obj, top, path, chain}
        self.visited = []         # (obj, path) in pre-order: every object whose Placement the solver may change
        self.node_by_obj = {}     # object Name -> [node]
        self.notes = []

    def leaf_nodes(self):
        return set(lf["node"] for lf in self.leaves)

    def path_of(self, node):
        if node in self.containers:
            return self.containers[node]["path"]
        for lf in self.leaves:
            if lf["node"] == node:
                return lf["path"]
        return None

    def _register(self, obj, node, chain):
        self.node_by_obj.setdefault(obj.Name, []).append(node)
        if chain:
            self.containers[chain[-1]]["children"].append(node)

    def walk(self, obj, path, chain, top, depth=0):
        if depth > 32:
            self.notes.append("tree deeper than 32 levels at %s: not read further" % path)
            return
        if is_skipped(obj):
            return
        node = path[:-1]
        self.visited.append((obj, path))
        if obj.TypeId == "App::DocumentObjectGroup":                       # a folder: transparent
            for child in obj.Group:
                if child.Name in self.direct and not chain:
                    continue                                                # also a direct member of the Assembly
                self.walk(child, path + child.Name + ".", chain, top, depth + 1)
            return
        if is_link_group(obj):                                              # a link array: its elements are components
            self.containers[node] = {"kind": "link_array", "rigid": False, "unit": False,
                                     "parent": chain[-1] if chain else None,
                                     "children": [], "obj": obj, "top": top, "path": path, "chain": list(chain)}
            self._register(obj, node, chain)
            for el in list(getattr(obj, "ElementList", []) or []):
                self.walk(el, path + el.Name + ".", chain + [node], top, depth + 1)
            return
        target = obj.getLinkedObject(True) if hasattr(obj, "getLinkedObject") else obj
        kind = rigid = None
        if target.TypeId == "App::Part":
            kind, rigid = "part", True
        elif target.TypeId == "Assembly::AssemblyObject":
            kind, rigid = ("assembly", False) if target is obj else ("link_to_assembly", True)
        elif target.TypeId == "Assembly::AssemblyLink":
            kind, rigid = "assembly_link", bool(getattr(target, "Rigid", True))
        if kind is None:
            self.leaves.append({"node": node, "path": path, "obj": obj, "chain": list(chain), "top": top})
            self._register(obj, node, chain)
            return
        self.containers[node] = {"kind": kind, "rigid": rigid, "unit": bool(rigid), "parent": chain[-1] if chain else None,
                                 "children": [], "obj": obj, "top": top, "path": path, "chain": list(chain)}
        self._register(obj, node, chain)
        for child in list(getattr(target, "Group", []) or []):
            self.walk(child, path + child.Name + ".", chain + [node], top, depth + 1)


def build_tree(asm):
    tree = Tree(asm)
    for comp in asm.Group:
        if not is_skipped(comp):
            tree.walk(comp, comp.Name + ".", [], comp.Name)
    return tree


def holds_label(tree, label):
    """True when a leaf of the tree carries `label` (the linked object's label or its own)."""
    for lf in tree.leaves:
        try:
            own = (lf["obj"].Label or "").strip()
            linked, _t = leaf_label(lf["obj"])
        except Exception:  # noqa: BLE001
            continue
        if label in (own, linked):
            return True
    return False


def moving_part(UtilsAssembly, asm, components, ref):
    """The assembly component a joint reference attaches to (getMovingPart, else getComponentReference for a reference
    rooted at the assembly with the component in its subname: the form FreeCAD's own joint tools write)."""
    part = UtilsAssembly.getMovingPart(ref)
    if part is not None and part.Name in components:
        return part
    sub = ref[1][0] if ref is not None and len(ref) == 2 and ref[1] else ""
    comp, _rel = UtilsAssembly.getComponentReference(asm, ref[0], sub) if ref is not None and len(ref) == 2 else (None, "")
    return comp if comp is not None else part


def resolve_node(UtilsAssembly, asm, tree, ref):
    """(node id, note) of the tree node a joint reference attaches to (None when it reaches nothing of this Assembly)."""
    if ref is None or len(ref) != 2 or ref[0] is None:
        return None, "no reference"
    part = moving_part(UtilsAssembly, asm, tree.direct, ref)
    if part is None:
        return None, "the reference has no object"
    nodes = tree.node_by_obj.get(part.Name, [])
    if len(nodes) > 1:
        return None, "the reference's object %s appears %d times in the Assembly" % (part.Name, len(nodes))
    if not nodes:
        return None, "the reference's object %s is not a component of this Assembly" % part.Name
    return descend(tree, nodes[0], ref[1][0] if ref[1] else ""), None


def descend(tree, node, sub, leaves=None):
    """The deepest tree node below `node` that the subname `sub` names (its last element, a face or '', dropped)."""
    names = [n for n in str(sub).split(".")[:-1] if n]
    leaves = tree.leaf_nodes() if leaves is None else leaves
    for _ in range(32):
        if node not in tree.containers:
            break
        nxt = None
        for nm in names:
            cand = node + "." + nm
            if cand in tree.containers or cand in leaves:
                nxt = cand
                break
        if nxt is None:
            break
        node = nxt
    return node


def joint_owner(obj, asm, tree):
    """'top' for a joint of the Assembly, the container node of the sub-assembly holding it, or None (foreign)."""
    for p in getattr(obj, "InList", []) or []:
        if p is asm:
            return "top"
    for p in getattr(obj, "InList", []) or []:
        if p.TypeId in ("Assembly::AssemblyObject", "Assembly::AssemblyLink"):
            nodes = [n for n in tree.node_by_obj.get(p.Name, []) if n in tree.containers]
            if len(nodes) == 1:
                return nodes[0]
    return None


def linked_sources(tree):
    """{container node: (source Assembly, {source member name: node of this tree})} for the sub-assemblies whose inner
    joints live only in the assembly they link (a rigid AssemblyLink; an App::Link to an Assembly)."""
    out = {}
    leaves = tree.leaf_nodes()
    for node, c in sorted(tree.containers.items()):
        obj, cmap, src = c["obj"], {}, None
        if c["kind"] == "assembly_link" and c["rigid"]:
            src = getattr(obj, "LinkedObject", None)
            for ch in list(getattr(obj, "Group", []) or []):
                lo = getattr(ch, "LinkedObject", None)
                cand = node + "." + ch.Name
                if lo is not None and (cand in tree.containers or cand in leaves):
                    cmap.setdefault(lo.Name, cand)
        elif c["kind"] == "link_to_assembly":
            src = obj.getLinkedObject(True)
            for ch in list(getattr(src, "Group", []) or []):
                cand = node + "." + ch.Name
                if cand in tree.containers or cand in leaves:
                    cmap[ch.Name] = cand
        if (src is not None and getattr(src, "TypeId", "") == "Assembly::AssemblyObject" and src is not tree.asm
                and src.Name not in tree.node_by_obj):
            out[node] = (src, cmap)
    return out


def joints_of(doc, assembly):
    return [o for o in doc.Objects if is_joint(o) and any(p is assembly for p in (getattr(o, "InList", []) or []))]


def node_placement(root, root_sub, path):
    try:
        return root.getPlacementOf(root_sub + path)
    except Exception:  # noqa: BLE001 - older API: the shape's placement
        return Part.getShape(root, root_sub + path, needSubElement=False, transform=True).Placement


def read_limits(obj, rec):
    """E3: the angle limits of every joint that has them (Revolute, Cylindrical) and the length limits of a
    Cylindrical joint (recorded; the grader reads a Cylindrical joint as a hinge and ignores its slide)."""
    for k in ("Min", "Max"):
        rec["enable_angle_" + k.lower()] = bool(getattr(obj, "EnableAngle" + k, False))
        q = getattr(obj, "Angle" + k, 0.0)
        rec["angle_" + k.lower() + "_deg"] = float(getattr(q, "Value", q))
    if str(getattr(obj, "JointType", "")) in ("Revolute", "Cylindrical"):
        rec["angle_saved_deg"] = saved_angle(rec)
    if str(getattr(obj, "JointType", "")) == "Cylindrical":
        for k in ("Min", "Max"):
            rec["enable_length_" + k.lower()] = bool(getattr(obj, "EnableLength" + k, False))
            q = getattr(obj, "Length" + k, 0.0)
            rec["length_" + k.lower() + "_mm"] = float(getattr(q, "Value", q))


def saved_angle(rec):
    """E3: the angle of a Revolute or Cylindrical joint at the saved pose: from the x axis of jcs1 to the x axis of
    jcs2 about the z axis of jcs1, in degrees (None when a frame is missing or not finite). FreeCAD's solve does not
    enforce the angle limits, so the saved angle can lie outside the declared interval."""
    try:
        a, b = rec["jcs1"], rec["jcs2"]
        x1 = [a[i][0] for i in range(3)]
        z1 = [a[i][2] for i in range(3)]
        x2 = [b[i][0] for i in range(3)]
        c = [x1[1] * x2[2] - x1[2] * x2[1], x1[2] * x2[0] - x1[0] * x2[2], x1[0] * x2[1] - x1[1] * x2[0]]
        v = math.degrees(math.atan2(sum(ci * zi for ci, zi in zip(c, z1)), sum(p * q for p, q in zip(x1, x2))))
        return v if math.isfinite(v) else None
    except Exception:  # noqa: BLE001
        return None


def note_type(model, rec, label):
    if rec["type"] not in JOINT_TYPES_READ and not rec.get("foreign") and not rec.get("suppressed"):
        model["joint_types_other"].append({"name": rec["name"], "label": label, "type": rec["type"]})


def mapped_joints(doc, UtilsAssembly, tree, sources, root, root_sub, model):
    """The inner joints of the rigid sub-assemblies (linked_sources), read like top-level joints: each reference's moving
    part is mapped onto the container's member that links it, and its global frame is moved with that member."""
    handled = set()
    leaves = tree.leaf_nodes()
    for node, (src, cmap) in sorted(sources.items()):
        c = tree.containers[node]
        if c["kind"] == "assembly_link" and joints_of(doc, c["obj"]):
            c["unit"] = False
            continue
        direct = set(o.Name for o in getattr(src, "Group", []) or [])
        n_read = 0
        for obj in joints_of(doc, src):
            handled.add(obj.Name)
            label = "%s (in %s)" % (obj.Label, c["obj"].Label)
            jname = "%s:%s" % (node, obj.Name)
            supp = bool(getattr(obj, "Suppressed", False))
            if hasattr(obj, "ObjectToGround"):
                tgt = obj.ObjectToGround
                gnode = cmap.get(tgt.Name) if tgt is not None else None
                model["grounded"].append({"name": jname, "label": label, "owner": node, "suppressed": supp,
                                          "object": tgt.Name if tgt is not None else None, "node": gnode,
                                          "mapped_from": src.Name})
                n_read += 1 if gnode is not None else 0
                continue
            rec = {"name": jname, "label": label, "type": str(obj.JointType), "owner": node, "suppressed": supp,
                   "mapped_from": src.Name}
            try:
                for k in ("1", "2"):
                    ref = getattr(obj, "Reference" + k)
                    plc = getattr(obj, "Placement" + k)
                    part = moving_part(UtilsAssembly, src, direct, ref)
                    nd = cmap.get(part.Name) if part is not None else None
                    if nd is not None and c["kind"] == "link_to_assembly":
                        nd = descend(tree, nd, ref[1][0] if ref[1] else "", leaves)
                    rec["part" + k] = nd
                    if nd is None:
                        rec["note" + k] = "the reference's object %s is not a member of %s" % (
                            part.Name if part is not None else None, src.Label)
                    rec["placement%s_finite" % k] = placement_finite(plc)
                    jcs = UtilsAssembly.getJcsGlobalPlc(plc, ref)
                    if nd is not None:
                        to_part = "" if ref[0] is part else UtilsAssembly.truncateSubAtFirst(ref[1][0] if ref[1] else "", part.Name)
                        w_src = UtilsAssembly.getGlobalPlacement((ref[0], [to_part]), part)
                        jcs = node_placement(root, root_sub, tree.path_of(nd)).multiply(w_src.inverse()).multiply(jcs)
                    rec["jcs" + k] = matrix(jcs)
                read_limits(obj, rec)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = "%s" % exc
            if rec.get("part1") is None and rec.get("part2") is None:
                rec["foreign"] = True
            else:
                n_read += 1
            note_type(model, rec, label)
            model["joints"].append(rec)
        c["unit"] = n_read == 0
        c["joints_mapped"] = n_read
        if n_read:
            model["notes"].append("%d joint(s) of %s read inside %s (a rigid sub-assembly: its inner joints are read like "
                                  "top-level joints)" % (n_read, src.Label, c["obj"].Label))
    return handled


def guard_freecad_restore():
    """FreeCAD 1.1.4's JointObject.getContext walks InList[0] upwards without a cycle guard and loops forever inside
    App.openDocument for one kind of malformed joint reference: replace it by a cycle-safe copy before opening."""
    try:
        import JointObject
    except Exception:  # noqa: BLE001
        return False

    def getContext(obj):
        context, current, seen = [], obj, set()
        while current is not None and len(context) < 64:
            name = getattr(current, "Name", None)
            if name in seen:
                break
            seen.add(name)
            context.insert(0, current.Label if current.Label else current.Name)
            parents = getattr(current, "InList", [])
            current = parents[0] if parents else None
        return ".".join(context)
    JointObject.getContext = getContext
    return True


def extract(fcstd, out_dir, options):
    grp = App.ParamGet("User parameter:BaseApp/Preferences/Mod/Assembly")             # E6
    had = grp.GetBool("SolveOnRecompute", True)
    grp.SetBool("SolveOnRecompute", False)
    try:
        return _extract(fcstd, out_dir, options)
    finally:
        grp.SetBool("SolveOnRecompute", had)


def _extract(fcstd, out_dir, options):
    t0 = time.time()
    model = {"format": FORMAT, "errors": [], "warnings": [], "notes": [], "components": [], "containers": [],
             "joints": [], "grounded": [], "recompute_moved": [], "joint_types_other": [], "assemblies_ignored": []}
    os.makedirs(os.path.join(out_dir, "shapes"), exist_ok=True)
    tag_re = re.compile(options.get("tag_pattern") or TAG_PATTERN)
    palm_tag = str(options.get("palm_tag") or PALM_TAG)
    model["restore_guard"] = guard_freecad_restore()
    try:
        doc = App.openDocument(fcstd)
    except Exception as exc:  # noqa: BLE001
        model["errors"].append("the document does not open: %s" % exc)
        return model
    model["freecad"] = ".".join(App.Version()[:3])
    asms, tops = top_level_assemblies(doc)
    model["n_assemblies"] = len(asms)
    with_comps = [a for a in tops if n_components(a) > 0]
    model["n_top_assemblies"] = len(with_comps)
    model["top_assemblies"] = [a.Name for a in with_comps]
    if not asms:
        model["errors"].append("no Assembly in the document")
        return model
    if not with_comps:
        model["errors"].append("no top-level Assembly with components in the document")
        return model
    # E4: the Assembly that holds the palm's largest part; else the first of the largest (document order)
    trees = [(a, build_tree(a)) for a in with_comps]
    holding = [(a, t) for a, t in trees if holds_label(t, palm_tag)]
    pool = holding or trees
    asm, tree = max(pool, key=lambda at: n_components(at[0]))
    model["assembly_choice"] = "holds %s" % palm_tag if holding else "largest (no Assembly holds %s)" % palm_tag
    for a, _t in trees:
        if a is not asm:
            model["assemblies_ignored"].append({"name": a.Name, "label": a.Label, "components": n_components(a)})
    if model["assemblies_ignored"]:
        model["warnings"].append("%d other Assembly(ies) with components are not read: %s" % (
            len(model["assemblies_ignored"]), ", ".join(x["label"] for x in model["assemblies_ignored"])[:300]))
    root, root_sub = asm, ""
    parents = asm.Parents or []
    if parents:
        root, root_sub = parents[0][0], parents[0][1]
    model["assembly"] = asm.Name
    model["assembly_label"] = asm.Label

    import UtilsAssembly  # FreeCAD's Assembly helpers (headless-safe)

    # ---- the saved placements, the recompute, the saved placements restored (configuration Z is the saved pose)
    saved = []
    for obj, path in tree.visited:
        try:
            saved.append((obj, path, obj.Placement.copy()))
        except Exception:  # noqa: BLE001 - objects without a placement
            pass
    try:
        doc.recompute()
    except Exception as exc:  # noqa: BLE001
        model["warnings"].append("recompute raised: %s" % str(exc)[:200])
    for obj, path, plc in saved:
        try:
            now = obj.Placement
            a, b = plc.toMatrix(), now.toMatrix()
            diff = max(abs(getattr(a, k) - getattr(b, k)) for k in PLC_KEYS)
            if not diff <= MOVED_TOL:            # NaN-safe: a non-finite saved placement is put back as saved too
                if placement_finite(plc) and placement_finite(now):
                    model["recompute_moved"].append({"object": obj.Name, "path": root_sub + path,
                                                     "moved_mm": float((now.Base - plc.Base).Length),
                                                     "turned_deg": float(math.degrees(now.Rotation.multiply(plc.Rotation.inverted()).Angle))})
                obj.Placement = plc
        except Exception as exc:  # noqa: BLE001
            model["notes"].append("placement of %s not compared: %s" % (path, str(exc)[:120]))
    if model["recompute_moved"]:
        model["warnings"].append("the recompute moved %d component(s); the saved placements are used" % len(
            model["recompute_moved"]))
    model["notes"] += tree.notes

    def nonfinite_on_path(path):
        for name in [None] + [n for n in (root_sub + path).split(".") if n]:
            o = root if name is None else doc.getObject(name)
            if o is None:
                continue
            if not placement_finite(getattr(o, "Placement", None)):
                return o
            if o.TypeId == "App::Link" and getattr(o, "LinkTransform", False):
                lo = getattr(o, "LinkedObject", None)
                if lo is not None and not placement_finite(getattr(lo, "Placement", None)):
                    return lo
        return None

    # ---- leaves: shapes and the per-solid export (E2)
    by_top = {}
    leaf_recs = []
    n_leaves = n_solids = 0
    for lf in sorted(tree.leaves, key=lambda r: r["path"]):
        n_leaves += 1
        if n_leaves > MAX_LEAVES:
            model["errors"].append("more than %d shape leaves in the document" % MAX_LEAVES)
            return model
        obj = lf["obj"]
        linked_label, target = leaf_label(obj)
        rec = {"node": lf["node"], "path": root_sub + lf["path"], "object": obj.Name, "type": target.TypeId,
               "object_type": obj.TypeId, "own_label": (obj.Label or "").strip(), "linked_label": linked_label,
               "linked_object": target.Name, "containers": lf["chain"], "top": lf["top"], "solids": []}
        if target.TypeId == "PartDesign::Body":
            rec["body"] = target.Name
        bad = nonfinite_on_path(lf["path"])
        rec["placement_finite"] = bad is None
        if bad is not None:
            rec["placement_nonfinite"] = "%s (%s)" % ((bad.Label or "").strip(), bad.Name)
        try:
            shp = Part.getShape(root, root_sub + lf["path"], needSubElement=False, transform=True).copy()
            if shp.isNull():
                if is_link(obj) or is_link(target):                                   # E5
                    rec["empty_link"] = True
                    model["notes"].append("leaf %s: a link whose target has no readable shape (an object of a document "
                                          "that is not in the submission folder?); nothing of it is read" % rec["own_label"])
                    leaf_recs.append(rec)
                    by_top.setdefault(lf["top"], []).append(rec)
                    continue
                raise ValueError("the leaf has no shape")
            sols = shp.Solids
            rec["n_solids"] = len(sols)
            rec["n_free_faces"], rec["n_free_edges"] = free_geometry(shp)
            rec["volume"] = float(shp.Volume)
            rec["placement"] = matrix(shp.Placement)
            if bad is None:
                for k, s in enumerate(sols):
                    n_solids += 1
                    if n_solids > MAX_SOLIDS:
                        model["errors"].append("more than %d solids in the document" % MAX_SOLIDS)
                        return model
                    srec = {"brep": "shapes/%04d_%03d.brep" % (n_leaves, k), "closed": bool(s.isClosed()),
                            "volume": float(s.Volume)}
                    s.exportBrep(os.path.join(out_dir, srec["brep"]))
                    rec["solids"].append(srec)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = "%s" % exc
        leaf_recs.append(rec)
        by_top.setdefault(lf["top"], []).append(rec)

    # ---- labels: the linked object's; a shared linked label gives way to each leaf's own label when it is a tag
    count = {}
    for rec in leaf_recs:
        count[rec["linked_label"]] = count.get(rec["linked_label"], 0) + 1
    for rec in leaf_recs:
        if count[rec["linked_label"]] > 1 and tag_re.match(rec["own_label"]):
            rec["label"], rec["label_source"] = rec["own_label"], "own"
        else:
            rec["label"], rec["label_source"] = rec["linked_label"], "linked"

    for comp in asm.Group:
        if is_skipped(comp):
            continue
        sub_flags = [c for c in tree.containers.values() if c["top"] == comp.Name and c["kind"] not in TRANSPARENT]
        model["components"].append({"name": comp.Name, "label": comp.Label, "type": comp.TypeId,
                                    "flattened": bool(sub_flags), "leaves": by_top.get(comp.Name, [])})

    # ---- joints: the rigid sub-assemblies' inner joints (mapped), then every other one (E3: every type recorded)
    handled = mapped_joints(doc, UtilsAssembly, tree, linked_sources(tree), root, root_sub, model)
    for obj in doc.Objects:
        if not is_joint(obj) or obj.Name in handled:
            continue
        owner = joint_owner(obj, asm, tree)
        if hasattr(obj, "ObjectToGround"):
            tgt = obj.ObjectToGround
            rec = {"name": obj.Name, "label": obj.Label, "owner": owner,
                   "suppressed": bool(getattr(obj, "Suppressed", False)), "object": tgt.Name if tgt is not None else None}
            node = None
            if tgt is not None:
                nodes = tree.node_by_obj.get(tgt.Name, [])
                node = nodes[0] if len(nodes) == 1 else None
                if len(nodes) > 1:
                    rec["error"] = "the grounded object %s appears %d times in the Assembly" % (tgt.Name, len(nodes))
            rec["node"] = node
            if owner is None and node is None:
                rec["foreign"] = True
            model["grounded"].append(rec)
            continue
        rec = {"name": obj.Name, "label": obj.Label, "type": str(obj.JointType), "owner": owner,
               "suppressed": bool(getattr(obj, "Suppressed", False))}
        try:
            for k in ("1", "2"):
                ref = getattr(obj, "Reference" + k)
                node, why = resolve_node(UtilsAssembly, asm, tree, ref)
                rec["part" + k] = node
                if why:
                    rec["note" + k] = why
                rec["ref" + k] = {"object": ref[0].Name if ref is not None and len(ref) == 2 and ref[0] is not None else None,
                                  "sub": list(ref[1]) if ref is not None and len(ref) == 2 else []}
                rec["placement%s_finite" % k] = placement_finite(getattr(obj, "Placement" + k))
                rec["jcs" + k] = matrix(UtilsAssembly.getJcsGlobalPlc(getattr(obj, "Placement" + k), ref))
            read_limits(obj, rec)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = "%s" % exc
        if owner is None and rec.get("part1") is None and rec.get("part2") is None:
            rec["foreign"] = True
        note_type(model, rec, obj.Label)
        model["joints"].append(rec)

    # ---- containers (after the joints: a rigid sub-assembly whose inner joints were read is no rigid unit)
    for node in sorted(tree.containers):
        c = tree.containers[node]
        model["containers"].append({"node": node, "kind": c["kind"], "rigid": c["rigid"], "unit": bool(c["unit"]),
                                    "joints_mapped": int(c.get("joints_mapped", 0)), "parent": c["parent"],
                                    "flagged": c["kind"] not in TRANSPARENT,
                                    "children": sorted(c["children"]), "object": c["obj"].Name, "label": c["obj"].Label,
                                    "top": c["top"]})
    model["n_leaves"] = n_leaves
    model["n_solids"] = n_solids
    model["elapsed_s"] = time.time() - t0
    return model


def parse_options(argv):
    if len(argv) < 4 or not argv[3]:
        return {}
    text = argv[3]
    if not text.lstrip().startswith("{"):
        with open(text) as fh:
            text = fh.read()
    spec = json.loads(text)
    return spec if isinstance(spec, dict) else {}


def write_model(model, out_dir):
    def plain(o):                     # anything unexpected as text (never fail here)
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)
    with open(os.path.join(out_dir, "model.json"), "w") as fh:
        json.dump(model, fh, indent=1, sort_keys=True, default=plain)


def main(argv):
    fcstd, out_dir = argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    write_model({"format": FORMAT, "errors": ["the extraction did not finish (time limit?)"]}, out_dir)
    try:
        model = extract(fcstd, out_dir, parse_options(argv))
    except Exception:  # noqa: BLE001 - a fault of this reader, not of the document: the grader fails the run with it
        model = {"format": FORMAT, "errors": ["extraction failed: " + traceback.format_exc()[-1500:]],
                 "extractor_failure": True}
    write_model(model, out_dir)
    print("HAND-EXTRACT done", len(model.get("components", [])), "components", model.get("n_leaves", 0), "leaves",   # E1
          model.get("n_solids", 0), "solids", len(model.get("joints", [])), "joints", len(model.get("errors", [])), "errors")


main(sys.argv)
