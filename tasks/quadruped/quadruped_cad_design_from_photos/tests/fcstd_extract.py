"""Read the submitted FreeCAD document into plain data (the grader's only contact with FreeCAD).

Runs under FreeCAD 1.1.4's headless interpreter as an unprivileged account (verify.py starts it with a time limit):

    freecadcmd -c "import sys; sys.argv = ['fcstd_extract.py', FCSTD, CLASSES, OUT_DIR]; exec(open(PATH).read())"

CLASSES is a JSON list of [class, step] pairs (the bought components the grader recognises by shape, e.g.
[["core", "/w/core.step"], ["foot", "/w/foot.step"]]), the path of a JSON file holding that list, or a JSON object
{"classes": [...], "tag_pattern": "...", "export_components": false}. It opens the document, finds the top-level
Assembly and writes OUT_DIR/model.json plus a STEP and a BREP file per printed (unclassified) shape leaf, in world
coordinates (the document as saved). Nothing in the document is executed beyond what FreeCAD runs to open and
recompute it. Errors are recorded in model.json, never raised.

What it records (format fcstd-extract-quad/1, read by quad_convert.py):
- components: the members of the Assembly; App::Part containers, nested Assemblies, AssemblyLinks and links to
  assemblies are flattened into their shape leaves and flagged ("containers"); folders (App::DocumentObjectGroup) and
  link arrays are transparent, as FreeCAD's Assembly treats their members as separate components.
- leaves: one record per shape-bearing leaf with a node id (its subname path in the Assembly), its label, validity,
  solid count, volume, world placement and component class. A leaf is of class C when, after its Placement is reset,
  its volume is within 1 % and its sorted extents within 0.5 mm of C's STEP (also when its volume and principal moments
  match and its pose can be recovered exactly: a transform baked into the geometry). A class leaf's pose is checked:
  its placement-free geometry must be C's (centre of mass and inertia); a baked transform is recovered from the
  principal axes and verified on C's vertices; a mirrored or deformed copy is flagged pose_ok = false.
- labels: the linked object's label; when several leaves resolve to one linked label, each one's own label is used
  when it matches the tag pattern (one Body linked several times, each link labelled with its part's tag).
- non-finite placements (fix of 2026-10-28, input review): FreeCAD keeps a NaN in the Placement of an App::Link (and of
  a joint) across a save, and Part.getShape then silently falls back to the linked object's own placement (the Body's,
  often the origin). A leaf with a non-finite placement anywhere on its path (the Assembly, a container, the component;
  a LinkTransform link's target) is recorded with placement_finite = false and the object named; the converter treats it
  as missing. Each joint records whether its own Placement1 / Placement2 are finite.
- joints: every joint of the Assembly and of the flattened sub-assemblies (FreeCAD's own helpers resolve references
  and global joint frames); a joint whose references reach no component of this Assembly is foreign (another assembly)
  and ignored. Grounded joints inside a sub-assembly ground their part to that sub-assembly, not to the world. A
  reference rooted at a sub-assembly is resolved down its subname to the member it names (FreeCAD 1.1.4's joint tool
  roots a reference to a part of a nested assembly at that assembly: (Leg, ['P22_cmp.'])). A rigid AssemblyLink and an
  App::Link to an Assembly keep their inner joints only in the assembly they link (FreeCAD copies them into a flexible
  AssemblyLink's own joint group): those joints are read too, mapped onto the container's members (their global frames
  moved with the members), so a sub-assembly's inner joints are read like top-level joints whatever its Rigid flag
  (GRADER_SPEC 2; fix of 2026-10-28); a container whose members form one rigid unit (an App::Part, a rigid sub-assembly
  without inner joints) is recorded with unit = true.
- recompute: FreeCAD's Assembly solver runs on recompute when the document was saved with an unsolved assembly and can
  move parts to another branch; the placements of the Assembly's components are restored to the saved ones after the
  recompute (the saved pose is configuration Z) and every move is recorded.
- restore guard: FreeCAD 1.1.4 loops forever inside openDocument for a joint reference with one subname rooted at the
  Assembly (JointObject.getContext has no cycle guard); a cycle-safe copy is installed before opening.
- bought components are not exported (the grader places the given STEP at the recorded pose; "export_components"
  exports them too).
"""
import json
import math
import os
import re
import sys
import traceback

import FreeCAD as App
import Part

FORMAT = "fcstd-extract-quad/1"
CLASS_VOLUME_REL_TOL = 0.01
CLASS_EXTENT_TOL_MM = 0.5
POSE_COM_TOL_MM = 0.05
POSE_INERTIA_REL_TOL = 1e-3
RECOVER_EIG_REL_TOL = 0.03       # principal-moment prefilter of a baked copy (B-spline faces move them by ~0.5 %)
POSE_VERTEX_TOL_MM = 0.05        # a recovered pose: the vertex sets coincide within this, both ways
JOINT_TYPES_READ = ("Fixed", "Revolute")
MOVED_TOL = 1e-6
TAG_PATTERN = r"^P(0[1-9]|[1-3][0-9]|4[0-2])$"
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


def extents(shape):
    b = shape.BoundBox
    return sorted([b.XLength, b.YLength, b.ZLength])


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


# ---- mass properties and component poses (numpy ships with FreeCAD's interpreter) ----------------------------------
def _np():
    import numpy as np
    return np


def mass_props(shape):
    """(volume, centre of mass, inertia tensor about it in the shape's coordinates) over its solids, or None."""
    np = _np()
    sols = list(shape.Solids)
    if not sols:
        return None
    vols = [float(s.Volume) for s in sols]
    V = sum(vols)
    if V <= 0:
        return None
    cs = [np.array([s.CenterOfMass.x, s.CenterOfMass.y, s.CenterOfMass.z]) for s in sols]
    c = sum(v * ci for v, ci in zip(vols, cs)) / V
    I = np.zeros((3, 3))
    for s, v, ci in zip(sols, vols, cs):
        M = s.MatrixOfInertia
        Mi = np.array([[M.A11, M.A12, M.A13], [M.A21, M.A22, M.A23], [M.A31, M.A32, M.A33]])
        d = ci - c
        I += Mi + v * (float(d @ d) * np.eye(3) - np.outer(d, d))
    return V, c, I


def _eig(I):
    np = _np()
    w, V = np.linalg.eigh(I)
    return w, V


class ClassRef(object):
    """A bought component's reference geometry (its STEP in its own frame)."""

    def __init__(self, cls, step):
        self.cls, self.step = cls, step
        shape = Part.read(step)
        self.shape = shape
        self.volume = float(shape.Volume)
        self.extents = extents(shape)
        self.props = mass_props(shape)
        self.eig = _eig(self.props[2])[0] if self.props else None
        self.vertices = _vertices(shape)
        self.read_placement_identity = bool(shape.Placement.isIdentity())

    def record(self):
        return {"class": self.cls, "step": self.step, "volume": self.volume, "extents": self.extents,
                "n_solids": len(self.shape.Solids), "com": [float(v) for v in self.props[1]] if self.props else None,
                "principal_moments": [float(v) for v in self.eig] if self.eig is not None else None,
                "read_placement_identity": self.read_placement_identity}


def classify(local, refs):
    """(ClassRef, how) of the first class matching the placement-free leaf shape, else (None, None). `how` is
    "extents" (the arm's rule) or "inertia" (volume and principal moments only: a baked transform, accepted only if its
    pose is recovered)."""
    vol = float(local.Volume)
    ext = extents(local)
    for ref in refs:
        if ref.volume > 0 and abs(vol - ref.volume) <= CLASS_VOLUME_REL_TOL * ref.volume and all(
                abs(a - b) <= CLASS_EXTENT_TOL_MM for a, b in zip(ext, ref.extents)):
            return ref, "extents"
    props = None
    for ref in refs:
        if ref.volume <= 0 or ref.eig is None or abs(vol - ref.volume) > CLASS_VOLUME_REL_TOL * ref.volume:
            continue
        if props is None:
            props = mass_props(local)
            if props is None:
                return None, None
        w = _eig(props[2])[0]
        scale = max(abs(float(ref.eig[-1])), 1e-9)
        if all(abs(float(a) - float(b)) <= RECOVER_EIG_REL_TOL * scale for a, b in zip(w, ref.eig)):
            return ref, "inertia"
    return None, None


def _kabsch(P, Q):
    """Rotation R and translation t minimising sum |R p + t - q|^2."""
    np = _np()
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _S, Vt = np.linalg.svd(H)
    d = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cq - R @ cp


def _vertices(shape):
    np = _np()
    return np.array([[v.Point.x, v.Point.y, v.Point.z] for v in shape.Vertexes], float).reshape(-1, 3)


def recover_transform(local, props, ref):
    """The rigid transform M (4x4) with local = M . ref (a transform baked into the geometry, e.g. by
    transformGeometry, which also turns the faces into B-splines). Starts from the principal axes (the proper sign
    choices; turns about a degenerate axis), refined by a point-to-point ICP of the reference's vertices onto the leaf's,
    accepted when the two vertex sets coincide within POSE_VERTEX_TOL_MM both ways; None for a mirrored or deformed
    copy (no proper rotation maps the vertices)."""
    np = _np()
    from scipy.spatial import cKDTree
    if props is None or ref.props is None:
        return None, "no solids"
    VL, VR = _vertices(local), ref.vertices
    if len(VL) == 0 or len(VR) == 0:
        return None, "no vertices"
    wl, El = _eig(props[2])
    wr, Er = _eig(ref.props[2])
    scale = max(abs(float(wr[-1])), 1e-9)
    if any(abs(float(a) - float(b)) > RECOVER_EIG_REL_TOL * scale for a, b in zip(wl, wr)):
        return None, "principal moments differ"
    turns = [np.eye(3)]
    for i in range(2):                       # a (near) degenerate pair: try turns about the third axis
        if abs(float(wr[i + 1] - wr[i])) <= RECOVER_EIG_REL_TOL * scale:
            k = 2 if i == 0 else 0
            axis = np.eye(3)[k]
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            turns = [np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K for a in np.radians(np.arange(0.0, 360.0, 15.0))]
            break
    tree_l = cKDTree(VL)
    best = None
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            for sz in (1.0, -1.0):
                D = np.diag([sx, sy, sz])
                for Tn in turns:
                    R = El @ D @ Tn @ Er.T
                    if np.linalg.det(R) < 0:
                        continue
                    t = props[1] - R @ ref.props[1]
                    for _ in range(40):
                        d, idx = tree_l.query(VR @ R.T + t)
                        R2, t2 = _kabsch(VR, VL[idx])
                        done = np.allclose(R2, R, atol=1e-12) and np.allclose(t2, t, atol=1e-9)
                        R, t = R2, t2
                        if done:
                            break
                    d1, _i = tree_l.query(VR @ R.T + t)
                    d2, _i = cKDTree(VR @ R.T + t).query(VL)
                    err = float(max(d1.max(), d2.max()))
                    if best is None or err < best[0]:
                        best = (err, R, t)
    if best is None or best[0] > POSE_VERTEX_TOL_MM:
        return None, "no proper rotation maps the reference's vertices onto it (max %.3f mm): mirrored or deformed" % (
            best[0] if best else float("nan"))
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = best[1], best[2]
    return M, "recovered from its vertices (max vertex distance %.4f mm)" % best[0]


def component_pose(shp, local, ref):
    """(4x4 component frame in world, check record). The placement is the pose when the placement-free geometry is the
    reference's (centre of mass and inertia); otherwise the baked transform is recovered."""
    np = _np()
    P = np.array(matrix(shp.Placement))
    props = mass_props(local)
    rec = {"ok": False, "recovered": False}
    if props is not None and ref.props is not None:
        dc = float(np.linalg.norm(props[1] - ref.props[1]))
        dI = float(np.linalg.norm(props[2] - ref.props[2]) / max(np.linalg.norm(ref.props[2]), 1e-12))
        rec.update(com_mm=dc, inertia_rel=dI)
        if dc <= POSE_COM_TOL_MM and dI <= POSE_INERTIA_REL_TOL:
            rec["ok"] = True
            return P, rec
    M, why = recover_transform(local, props, ref)
    rec["note"] = why
    if M is None:
        return P, rec
    rec.update(ok=True, recovered=True)
    return P @ M, rec


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
    A node id is the object's subname path in the Assembly without the trailing dot ("P07_cmp", "Leg.P07")."""

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
        """The subname path (relative to the Assembly, trailing dot) of a leaf or container node, or None."""
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
            # nested directly: one unit of this Assembly whose inner joints move its members; through an App::Link: rigid
            kind, rigid = ("assembly", False) if target is obj else ("link_to_assembly", True)
        elif target.TypeId == "Assembly::AssemblyLink":
            kind, rigid = "assembly_link", bool(getattr(target, "Rigid", True))
        if kind is None:
            self.leaves.append({"node": node, "path": path, "obj": obj, "chain": list(chain), "top": top})
            self._register(obj, node, chain)
            return
        # unit: the members form one rigid group (an App::Part; a rigid sub-assembly until its inner joints are mapped)
        self.containers[node] = {"kind": kind, "rigid": rigid, "unit": bool(rigid), "parent": chain[-1] if chain else None,
                                 "children": [], "obj": obj, "top": top, "path": path, "chain": list(chain)}
        self._register(obj, node, chain)
        for child in list(getattr(target, "Group", []) or []):
            self.walk(child, path + child.Name + ".", chain + [node], top, depth + 1)

    def collapse(self, node):
        """Turn container `node` into one leaf (its whole shape is a bought component)."""
        c = self.containers[node]
        inside = [n for n in list(self.containers) if n.startswith(node + ".")]
        self.leaves = [lf for lf in self.leaves if not lf["node"].startswith(node + ".")]
        for n in inside:
            del self.containers[n]
        for name in list(self.node_by_obj):            # its members resolve to the collapsed leaf
            nodes = [node if n.startswith(node + ".") else n for n in self.node_by_obj[name]]
            self.node_by_obj[name] = sorted(set(nodes), key=nodes.index)
        del self.containers[node]
        self.leaves.append({"node": node, "path": c["path"], "obj": c["obj"], "chain": c["chain"], "top": c["top"],
                            "collapsed": c["kind"]})


def moving_part(UtilsAssembly, asm, components, ref):
    """The assembly component a joint reference attaches to. FreeCAD 1.1's getMovingPart returns the reference's object,
    which is the component only for references rooted at it ((P01, ["Face1"])); a reference rooted at the assembly with
    the component in its subname ((Assembly, ["P01.Face1"]), the form FreeCAD's own joint tools write) is resolved
    through getComponentReference. Its global joint frame is the same either way (getJcsGlobalPlc)."""
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
    """The deepest tree node below `node` that the subname `sub` names (its last element, a face or '', dropped). A
    reference rooted at a sub-assembly attaches to the member its subname names, whatever the container's kind: FreeCAD
    1.1.4's joint tool roots a reference to a part of a nested assembly or a rigid AssemblyLink at that container,
    (Leg, ['P22_cmp.']); stopping at the container node left the joint on a node with no geometry (input review of
    2026-10-28: a nested leg without an inner Grounded joint floated). For a unit container (an App::Part) the member
    and the container are one rigid group anyway."""
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
    joints live only in the assembly they link: a rigid AssemblyLink (its members are App::Links to the source's
    members; FreeCAD copies a flexible one's joints into its own joint group, which is read as it is) and an App::Link
    to an Assembly (the walk enters the source's own members)."""
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
        # not an assembly this tree also holds directly (its joints are then read there, owned by that container)
        if (src is not None and getattr(src, "TypeId", "") == "Assembly::AssemblyObject" and src is not tree.asm
                and src.Name not in tree.node_by_obj):
            out[node] = (src, cmap)
    return out


def joints_of(doc, assembly):
    """The joints (Grounded included) that belong to `assembly` (it is in their InList, as for the top Assembly)."""
    return [o for o in doc.Objects if is_joint(o) and any(p is assembly for p in (getattr(o, "InList", []) or []))]


def node_placement(root, root_sub, path):
    """World placement of a tree node (a member of the Assembly) as FreeCAD's global placements are expressed."""
    try:
        return root.getPlacementOf(root_sub + path)
    except Exception:  # noqa: BLE001 - older API: the shape's placement
        return Part.getShape(root, root_sub + path, needSubElement=False, transform=True).Placement


def read_limits(obj, rec):
    for k in ("Min", "Max"):
        rec["enable_angle_" + k.lower()] = bool(getattr(obj, "EnableAngle" + k, False))
        q = getattr(obj, "Angle" + k, 0.0)
        rec["angle_" + k.lower() + "_deg"] = float(getattr(q, "Value", q))


def mapped_joints(doc, UtilsAssembly, tree, sources, root, root_sub, model):
    """The inner joints of the rigid sub-assemblies (linked_sources), read like top-level joints: each reference's moving
    part (a member of the source assembly) is mapped onto the container's member that links it, and its global frame is
    moved with that member: jcs = W(member in this Assembly) . W(source member)^-1 . jcs(source). A Grounded joint in the
    source grounds its member to the container (as any grounded joint inside a sub-assembly). Returns the names of the
    source joints read (the generic loop skips them); a container whose source holds no joint stays one rigid unit."""
    handled = set()
    leaves = tree.leaf_nodes()
    for node, (src, cmap) in sorted(sources.items()):
        c = tree.containers[node]
        if c["kind"] == "assembly_link" and joints_of(doc, c["obj"]):
            c["unit"] = False                    # it holds joints of its own (read as they are): not mapped twice
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
                    if nd is not None and c["kind"] == "link_to_assembly":      # the same objects: follow the subname
                        nd = descend(tree, nd, ref[1][0] if ref[1] else "", leaves)
                    rec["part" + k] = nd
                    if nd is None:
                        rec["note" + k] = "the reference's object %s is not a member of %s" % (
                            part.Name if part is not None else None, src.Label)
                    rec["ref" + k] = {"object": ref[0].Name if ref is not None and len(ref) == 2 and ref[0] is not None else None,
                                      "sub": list(ref[1]) if ref is not None and len(ref) == 2 else []}
                    rec["placement%s_finite" % k] = placement_finite(plc)
                    jcs = UtilsAssembly.getJcsGlobalPlc(plc, ref)
                    if nd is not None:
                        # the moving part's global placement by the same rule (and from the same root) as the frame's
                        to_part = "" if ref[0] is part else UtilsAssembly.truncateSubAtFirst(ref[1][0] if ref[1] else "", part.Name)
                        w_src = UtilsAssembly.getGlobalPlacement((ref[0], [to_part]), part)
                        jcs = node_placement(root, root_sub, tree.path_of(nd)).multiply(w_src.inverse()).multiply(jcs)
                    rec["jcs" + k] = matrix(jcs)
                read_limits(obj, rec)
            except Exception as exc:  # noqa: BLE001
                rec["error"] = "%s" % exc
            if rec.get("part1") is None and rec.get("part2") is None:
                rec["foreign"] = True                # reaches no member of the container: as a joint of another assembly
            else:
                n_read += 1
                if rec["type"] not in JOINT_TYPES_READ and not supp:
                    model["joint_type_errors"].append("joint %s is of type %s (only Grounded, Fixed and Revolute joints "
                                                      "are read)" % (label, rec["type"]))
            model["joints"].append(rec)
        c["unit"] = n_read == 0
        c["joints_mapped"] = n_read
        if n_read:
            model["notes"].append("%d joint(s) of %s read inside %s (a rigid sub-assembly: its inner joints are read like "
                                  "top-level joints)" % (n_read, src.Label, c["obj"].Label))
    return handled


def guard_freecad_restore():
    """FreeCAD 1.1.4's JointObject.getContext walks InList[0] upwards without a cycle guard; on restore, a joint whose
    reference holds one subname (migrationScript4 raises IndexError) and is rooted at the Assembly (the Assembly is then
    in the joint's InList chain and the joint in the Assembly's) loops forever inside App.openDocument. The function
    only builds a warning text: replace it by a cycle-safe copy before any document is opened."""
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


def extract(fcstd, refs_spec, out_dir, options):
    model = {"format": FORMAT, "errors": [], "warnings": [], "notes": [], "components": [], "containers": [],
             "joints": [], "grounded": [], "classes": [], "recompute_moved": [], "joint_type_errors": []}
    os.makedirs(os.path.join(out_dir, "shapes"), exist_ok=True)
    tag_re = re.compile(options.get("tag_pattern") or TAG_PATTERN)
    refs = []
    for cls, step in refs_spec:
        try:
            ref = ClassRef(str(cls), str(step))
            refs.append(ref)
            model["classes"].append(ref.record())
        except Exception as exc:  # noqa: BLE001
            model["errors"].append("component class %s: its STEP %s cannot be read (%s)" % (cls, step, exc))
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
    asm = max(with_comps, key=lambda a: n_components(a))     # first of the largest (document order)
    root, root_sub = asm, ""
    parents = asm.Parents or []
    if parents:
        root, root_sub = parents[0][0], parents[0][1]
    model["assembly"] = asm.Name
    model["assembly_label"] = asm.Label

    import UtilsAssembly  # FreeCAD's Assembly helpers (headless-safe)

    # ---- the tree, the saved placements, the recompute (the solver may move parts), the saved placements restored
    tree = Tree(asm)
    for comp in asm.Group:
        if not is_skipped(comp):
            tree.walk(comp, comp.Name + ".", [], comp.Name)
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
        except Exception as exc:  # noqa: BLE001 - objects without a usable placement
            model["notes"].append("placement of %s not compared: %s" % (path, str(exc)[:120]))
    if model["recompute_moved"]:
        model["warnings"].append("the recompute's assembly solve moved %d component(s) (the document was saved with an "
                                 "unsolved assembly); the saved placements are used" % len(model["recompute_moved"]))
    model["notes"] += tree.notes

    def nonfinite_on_path(path):
        """The first object on a member's path (the root, the Assembly, its containers, the member; a LinkTransform
        link's target) whose Placement is not finite, or None."""
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

    # ---- leaves: shapes, classes, poses, exports
    export_components = bool(options.get("export_components", False))
    # a container whose whole shape is a bought component (an importer's App::Part around its solids) is one leaf
    for node in sorted(tree.containers, key=lambda n: (n.count("."), n)):
        if node not in tree.containers:
            continue
        try:
            shp = Part.getShape(root, root_sub + tree.containers[node]["path"], needSubElement=False, transform=True).copy()
            if shp.isNull():
                continue
            local = shp.copy()
            local.Placement = App.Placement()
            if classify(local, refs)[0] is not None:
                tree.collapse(node)
        except Exception:  # noqa: BLE001 - the container is then read through its members
            continue
    shape_id = 0
    by_top = {}
    leaf_recs = []
    for lf in sorted(tree.leaves, key=lambda r: r["path"]):
        obj = lf["obj"]
        linked_label, target = leaf_label(obj)
        rec = {"node": lf["node"], "path": root_sub + lf["path"], "object": obj.Name, "type": target.TypeId,
               "object_type": obj.TypeId, "own_label": (obj.Label or "").strip(), "linked_label": linked_label,
               "linked_object": target.Name, "containers": lf["chain"], "top": lf["top"]}
        if lf.get("collapsed"):
            rec["collapsed_container"] = lf["collapsed"]
        bad = nonfinite_on_path(lf["path"])
        rec["placement_finite"] = bad is None
        if bad is not None:                  # Part.getShape would place it by a fallback: not exported, not read
            rec["placement_nonfinite"] = "%s (%s)" % ((bad.Label or "").strip(), bad.Name)
        try:
            shp = Part.getShape(root, root_sub + lf["path"], needSubElement=False, transform=True).copy()
            if shp.isNull():
                raise ValueError("the leaf has no shape")
            rec["valid"] = bool(shp.isValid())
            rec["n_solids"] = len(shp.Solids)
            rec["volume"] = float(shp.Volume)
            rec["placement"] = matrix(shp.Placement)
            local = shp.copy()
            local.Placement = App.Placement()
            ref, how = classify(local, refs)
            rec["component_class"] = None
            if ref is not None:
                pose, check = component_pose(shp, local, ref)
                if how == "inertia" and not check.get("recovered"):
                    ref = None                       # same volume and moments, but not this component's geometry
                else:
                    rec["component_class"] = ref.cls
                    rec["class_match"] = how
                    rec["pose"] = [[float(v) for v in row] for row in pose]
                    rec["pose_ok"] = bool(check.get("ok"))
                    rec["pose_recovered"] = bool(check.get("recovered"))
                    rec["pose_check"] = {k: v for k, v in check.items() if k not in ("ok", "recovered")}
            if (ref is None or export_components) and bad is None:
                shape_id += 1
                rec["step"] = "shapes/%03d.step" % shape_id
                shp.exportStep(os.path.join(out_dir, rec["step"]))
                # the exact shape beside it: a STEP round trip can turn a valid solid invalid (wire nesting re-read)
                rec["brep"] = "shapes/%03d.brep" % shape_id
                shp.exportBrep(os.path.join(out_dir, rec["brep"]))
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

    # ---- joints: the rigid sub-assemblies' inner joints (mapped from the assemblies they link), then every other one
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
            elif owner is None:
                model["warnings"].append("grounded joint %s is not in a joint group of the Assembly; read because it "
                                         "grounds one of its components" % obj.Label)
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
        if owner is None and not rec.get("foreign"):
            model["warnings"].append("joint %s is not in a joint group of the Assembly; read because it attaches to its "
                                     "components" % obj.Label)
        if rec["type"] not in JOINT_TYPES_READ and not rec.get("foreign") and not rec["suppressed"]:
            model["joint_type_errors"].append("joint %s is of type %s (only Grounded, Fixed and Revolute joints are read)" % (
                obj.Label, rec["type"]))
        model["joints"].append(rec)

    # ---- containers (after the joints: a rigid sub-assembly whose inner joints were read is no rigid unit)
    for node in sorted(tree.containers):
        c = tree.containers[node]
        model["containers"].append({"node": node, "kind": c["kind"], "rigid": c["rigid"], "unit": bool(c["unit"]),
                                    "joints_mapped": int(c.get("joints_mapped", 0)), "parent": c["parent"],
                                    "flagged": c["kind"] not in TRANSPARENT,
                                    "children": sorted(c["children"]), "object": c["obj"].Name, "label": c["obj"].Label,
                                    "top": c["top"]})
    flagged = [(n, c) for n, c in sorted(tree.containers.items()) if c["kind"] not in TRANSPARENT]
    if flagged:
        model["warnings"].append("sub-assemblies flattened into their parts: %s" % ", ".join(
            "%s (%s, %s)" % (n, c["kind"], "rigid" if c["rigid"] else "flexible") for n, c in flagged)[:400])
    return model


def parse_classes(arg):
    """CLASSES: JSON text or a JSON file holding a list of [class, step] or {"classes": [...], ...}."""
    text = arg
    if not arg.lstrip().startswith(("[", "{")):
        with open(arg) as fh:
            text = fh.read()
    spec = json.loads(text)
    options = {}
    if isinstance(spec, dict):
        options = {k: v for k, v in spec.items() if k != "classes"}
        spec = spec.get("classes", [])
    pairs = [(str(c), str(s)) for c, s in spec]
    return pairs, options


def main(argv):
    fcstd, classes_arg, out_dir = argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    try:
        pairs, options = parse_classes(classes_arg)
        model = extract(fcstd, pairs, out_dir, options)
    except Exception:  # noqa: BLE001
        model = {"format": FORMAT, "errors": ["extraction failed: " + traceback.format_exc()[-1500:]]}
    def plain(o):                     # numpy scalars from the pose checks; anything else as text (never fail here)
        try:
            return float(o)
        except (TypeError, ValueError):
            return str(o)
    with open(os.path.join(out_dir, "model.json"), "w") as fh:
        json.dump(model, fh, indent=1, sort_keys=True, default=plain)
    print("FCSTD-EXTRACT done", len(model.get("components", [])), "components", len(model.get("joints", [])), "joints",
          len(model.get("errors", [])), "errors")


main(sys.argv)
