"""COLLADA 1.4 reader of the maker's appearance meshes: the file's data as written, nothing repaired.

What it returns for one file (class Mesh):
    unit_meter, unit_name, up_axis        from <asset>
    primitives                            one Primitive per <triangles> element reached from the scene, in order
    V_m (N, 3) float64                    the flat vertex list in the file's frame, metres: every primitive's whole
                                          position array with its node matrix (composed from the scene root) and
                                          the unit applied
    F (M, 3) int64                        the flat triangle list: each primitive's vertex indices plus the offset of
                                          its positions in V_m

Order (pinned by the tests; the scoring-sample tables index this list): the scene's top-level nodes in document
order, depth first; inside a node its children in document order (an <instance_geometry> contributes its geometry's
<triangles> elements in document order, a child <node> is walked where it stands). This is the order of an
independent COLLADA library walking the scene graph, and the reader's indices equal its indices exactly.

Arithmetic is fixed and elementwise (no BLAS): x' = ((m00 x + m01 y) + m02 z) + m03, then times the unit, then
times 1000.0 for millimetres, so results do not depend on the CPU.

Only <triangles> primitives are accepted (the pinned files have no other kind); anything else raises MakerStdError.
Triangles that repeat a vertex index are reported (repeated_index_triangles), never removed here.
"""
import hashlib
import math
import xml.etree.ElementTree as ET

import numpy as np

from . import pinned
from .errors import MakerStdError

NS = "{http://www.collada.org/2005/11/COLLADASchema}"
TRANSFORM_TAGS = ("matrix", "translate", "rotate", "scale")
UNSUPPORTED_TRANSFORMS = ("lookat", "skew")
UNSUPPORTED_PRIMITIVES = ("polylist", "polygons", "trifans", "tristrips", "lines", "linestrips")
SHADERS = ("phong", "blinn", "lambert", "constant")


def _tag(el):
    t = el.tag
    return t[len(NS):] if t.startswith(NS) else t


def _floats(text):
    return np.array(text.split(), dtype=np.float64) if text and text.strip() else np.zeros(0, dtype=np.float64)


def identity4():
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def mat4_mul(A, B):
    """4 x 4 product in Python floats, fixed summation order."""
    return [[((A[i][0] * B[0][j] + A[i][1] * B[1][j]) + A[i][2] * B[2][j]) + A[i][3] * B[3][j] for j in range(4)]
            for i in range(4)]


def _transform_matrix(el):
    kind = _tag(el)
    v = [float(x) for x in (el.text or "").split()]
    if kind == "matrix":
        if len(v) != 16:
            raise MakerStdError("a <matrix> needs 16 numbers, found %d" % len(v))
        return [v[0:4], v[4:8], v[8:12], v[12:16]]
    if kind == "translate":
        T = identity4()
        T[0][3], T[1][3], T[2][3] = v
        return T
    if kind == "scale":
        T = identity4()
        T[0][0], T[1][1], T[2][2] = v
        return T
    if kind == "rotate":
        x, y, z, deg = v
        n = math.sqrt(x * x + y * y + z * z)
        x, y, z = x / n, y / n, z / n
        a = math.radians(deg)
        c, s, t = math.cos(a), math.sin(a), 1.0 - math.cos(a)
        return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y, 0.0],
                [t * x * y + s * z, t * y * y + c, t * y * z - s * x, 0.0],
                [t * x * z - s * y, t * y * z + s * x, t * z * z + c, 0.0],
                [0.0, 0.0, 0.0, 1.0]]
    raise MakerStdError("unsupported transform <%s>" % kind)


def node_local_matrix(node):
    """The node's own transform: its transform elements composed in document order (M = T1 T2 ...)."""
    M = identity4()
    for ch in node:
        kind = _tag(ch)
        if kind in TRANSFORM_TAGS:
            M = mat4_mul(M, _transform_matrix(ch))
        elif kind in UNSUPPORTED_TRANSFORMS:
            raise MakerStdError("unsupported transform <%s> in node %r" % (kind, node.get("id")))
    return M


def apply_matrix(M, P):
    """Rows of P (n, 3) mapped by the 4 x 4 matrix M, elementwise in a fixed order."""
    P = np.asarray(P, dtype=np.float64)
    out = np.empty(P.shape, dtype=np.float64)
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    for i in range(3):
        out[:, i] = ((M[i][0] * x + M[i][1] * y) + M[i][2] * z) + M[i][3]
    return out


class Primitive:
    """One <triangles> element as reached from the scene."""

    def __init__(self, **kw):
        self.node_id = kw["node_id"]
        self.node_name = kw["node_name"]
        self.geometry_id = kw["geometry_id"]
        self.index_in_geometry = kw["index_in_geometry"]
        self.material_symbol = kw["material_symbol"]
        self.material = kw["material"]
        self.positions = kw["positions"]          # (n, 3) float64, the file's numbers
        self.node_matrix = kw["node_matrix"]      # 4 x 4 list, composed from the scene root
        self.vertex_index = kw["vertex_index"]    # (m, 3) int64 into positions
        self.normal_corners = kw["normal_corners"]  # (m, 3, 3) float64 as written, or None
        self.uv_corners = kw["uv_corners"]        # (m, 3, 2) float64 as written, or None
        self.first_vertex = kw["first_vertex"]
        self.first_triangle = kw["first_triangle"]

    @property
    def triangle_count(self):
        return int(self.vertex_index.shape[0])


class Mesh:
    """The flat triangle list of one maker file and the primitives it comes from."""

    def __init__(self, name, unit_meter, unit_name, up_axis, primitives):
        self.name = name
        self.unit_meter = unit_meter
        self.unit_name = unit_name
        self.up_axis = up_axis
        self.primitives = primitives
        vs, fs, tp = [], [], []
        for k, p in enumerate(primitives):
            vs.append(apply_matrix(p.node_matrix, p.positions) * unit_meter)
            fs.append(p.vertex_index + p.first_vertex)
            tp.append(np.full(p.triangle_count, k, dtype=np.int32))
        self.V_m = np.concatenate(vs) if vs else np.zeros((0, 3))
        self.F = np.concatenate(fs) if fs else np.zeros((0, 3), dtype=np.int64)
        self.tri_primitive = np.concatenate(tp) if tp else np.zeros(0, dtype=np.int32)

    @property
    def triangle_count(self):
        return int(self.F.shape[0])

    def vertices_mm(self):
        return self.V_m * 1000.0

    def triangles_mm(self):
        """(M, 3, 3): the corners V0, V1, V2 of every triangle, file frame, millimetres."""
        return self.vertices_mm()[self.F]

    def triangle_areas_mm2(self):
        T = self.triangles_mm()
        return 0.5 * np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)

    def repeated_index_triangles(self):
        """Indices (into F) of the triangles that name the same vertex index twice or three times."""
        F = self.F
        return np.nonzero((F[:, 0] == F[:, 1]) | (F[:, 1] == F[:, 2]) | (F[:, 0] == F[:, 2]))[0]

    def corner_normals(self):
        """(M, 3, 3) unit normals per triangle corner, the file's normals mapped by the inverse transpose of the
        node matrix (zero where the file has none)."""
        out = np.zeros((self.triangle_count, 3, 3))
        for p in self.primitives:
            if p.normal_corners is None:
                continue
            R = np.array([row[:3] for row in p.node_matrix[:3]], dtype=np.float64)
            n = p.normal_corners.reshape(-1, 3) @ np.linalg.inv(R)
            ln = np.linalg.norm(n, axis=1, keepdims=True)
            n = np.where(ln > 0, n / np.where(ln > 0, ln, 1.0), 0.0)
            out[p.first_triangle:p.first_triangle + p.triangle_count] = n.reshape(-1, 3, 3)
        return out

    def corner_uvs(self):
        """(M, 3, 2) texture coordinates per corner as written (zero where the file has none)."""
        out = np.zeros((self.triangle_count, 3, 2))
        for p in self.primitives:
            if p.uv_corners is not None:
                out[p.first_triangle:p.first_triangle + p.triangle_count] = p.uv_corners
        return out

    def materials(self):
        """[(first_triangle, triangle_count, material dict)] per primitive, in order."""
        return [(p.first_triangle, p.triangle_count, p.material) for p in self.primitives]

    def digest(self):
        """sha256 of the flat list (F as little-endian int64, then V_m as little-endian float64): pins its order."""
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.F, dtype="<i8").tobytes())
        h.update(np.ascontiguousarray(self.V_m, dtype="<f8").tobytes())
        return h.hexdigest()


def _sources(mesh_el):
    out = {}
    for s in mesh_el.findall(NS + "source"):
        fa = s.find(NS + "float_array")
        acc = s.find(NS + "technique_common/" + NS + "accessor")
        if fa is None or acc is None:
            continue
        arr = _floats(fa.text)
        stride = int(acc.get("stride", "1"))
        count = int(acc.get("count", str(len(arr) // stride)))
        offset = int(acc.get("offset", "0"))
        if offset != 0 or count * stride > len(arr):
            raise MakerStdError("source %r: accessor does not fit its array" % s.get("id"))
        out[s.get("id")] = arr[:count * stride].reshape(count, stride)
    return out


def _effects(root):
    images = {}
    for im in root.iter(NS + "image"):
        ini = im.find(NS + "init_from")
        images[im.get("id")] = (ini.text or "").strip() if ini is not None else None
    effects = {}
    for ef in root.iter(NS + "effect"):
        prof = ef.find(NS + "profile_COMMON")
        if prof is None:
            continue
        params = {}
        for npar in prof.findall(NS + "newparam"):
            surf = npar.find(NS + "surface/" + NS + "init_from")
            samp = npar.find(NS + "sampler2D/" + NS + "source")
            if surf is not None:
                params[npar.get("sid")] = ("surface", (surf.text or "").strip())
            if samp is not None:
                params[npar.get("sid")] = ("sampler", (samp.text or "").strip())
        tech = prof.find(NS + "technique")
        shader = None
        if tech is not None:
            for ch in tech:
                if _tag(ch) in SHADERS:
                    shader = ch
                    break
        d = {"effect_id": ef.get("id"), "shader": _tag(shader) if shader is not None else None}
        if shader is not None:
            for ch in shader:
                key = _tag(ch)
                col = ch.find(NS + "color")
                flt = ch.find(NS + "float")
                tex = ch.find(NS + "texture")
                if col is not None:
                    d[key] = [float(x) for x in (col.text or "").split()]
                elif flt is not None:
                    d[key] = float((flt.text or "0").strip())
                elif tex is not None:
                    img = None
                    s = params.get(tex.get("texture"))
                    if s and s[0] == "sampler":
                        surf = params.get(s[1])
                        if surf and surf[0] == "surface":
                            img = images.get(surf[1])
                    d[key + "_texture"] = img
                    d[key + "_texcoord"] = tex.get("texcoord")
        effects[ef.get("id")] = d
    materials = {}
    for m in root.iter(NS + "material"):
        ie = m.find(NS + "instance_effect")
        eid = (ie.get("url") or "").lstrip("#") if ie is not None else None
        d = dict(effects.get(eid, {"effect_id": eid, "shader": None}))
        d["material_id"] = m.get("id")
        d["material_name"] = m.get("name")
        materials[m.get("id")] = d
    return materials


def _geometries(root):
    geoms = {}
    for g in root.iter(NS + "geometry"):
        mesh = g.find(NS + "mesh")
        if mesh is None:
            continue
        srcs = _sources(mesh)
        vert = mesh.find(NS + "vertices")
        if vert is None:
            raise MakerStdError("geometry %r has no <vertices>" % g.get("id"))
        pos_id = None
        for inp in vert.findall(NS + "input"):
            if inp.get("semantic") == "POSITION":
                pos_id = (inp.get("source") or "").lstrip("#")
            else:
                raise MakerStdError("geometry %r: per-vertex input %r is not supported" % (g.get("id"), inp.get("semantic")))
        if pos_id not in srcs:
            raise MakerStdError("geometry %r: no position source" % g.get("id"))
        positions = srcs[pos_id]
        if positions.shape[1] != 3:
            raise MakerStdError("geometry %r: positions are not 3D" % g.get("id"))
        prims = []
        for el in mesh:
            kind = _tag(el)
            if kind in ("source", "vertices", "extra"):
                continue
            if kind in UNSUPPORTED_PRIMITIVES:
                raise MakerStdError("geometry %r: <%s> is not supported" % (g.get("id"), kind))
            if kind != "triangles":
                raise MakerStdError("geometry %r: unexpected element <%s>" % (g.get("id"), kind))
            inputs = el.findall(NS + "input")
            stride = max(int(i.get("offset")) for i in inputs) + 1
            count = int(el.get("count"))
            p_el = el.find(NS + "p")
            idx = np.array((p_el.text or "").split() if p_el is not None else [], dtype=np.int64)
            if idx.size != count * 3 * stride:
                raise MakerStdError("geometry %r: <p> holds %d indices, count %d needs %d"
                                    % (g.get("id"), idx.size, count, count * 3 * stride))
            idx = idx.reshape(count, 3, stride)
            vidx = normals = uvs = None
            uv_set = None
            for i in inputs:
                sem, off = i.get("semantic"), int(i.get("offset"))
                if sem == "VERTEX":
                    vidx = idx[:, :, off].copy()
                elif sem == "NORMAL":
                    normals = srcs[(i.get("source") or "").lstrip("#")][idx[:, :, off]]
                elif sem == "TEXCOORD":
                    st = int(i.get("set", "0"))
                    if uv_set is None or st < uv_set:
                        uv_set = st
                        uvs = srcs[(i.get("source") or "").lstrip("#")][idx[:, :, off]][:, :, :2]
            if vidx is None:
                raise MakerStdError("geometry %r: <triangles> without VERTEX input" % g.get("id"))
            if vidx.size and (vidx.min() < 0 or vidx.max() >= positions.shape[0]):
                raise MakerStdError("geometry %r: vertex index out of range" % g.get("id"))
            prims.append({"material_symbol": el.get("material"), "vertex_index": vidx, "normals": normals, "uvs": uvs})
        geoms[g.get("id")] = {"positions": positions, "prims": prims}
    return geoms


def parse(data, name=""):
    """Mesh of a COLLADA document given as bytes."""
    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        raise MakerStdError("%s: not well-formed XML: %s" % (name, exc))
    if _tag(root) != "COLLADA":
        raise MakerStdError("%s: not a COLLADA document" % name)
    asset = root.find(NS + "asset")
    unit_el = asset.find(NS + "unit") if asset is not None else None
    up_el = asset.find(NS + "up_axis") if asset is not None else None
    unit_meter = float(unit_el.get("meter", "1")) if unit_el is not None else 1.0
    unit_name = unit_el.get("name") if unit_el is not None else None
    up_axis = (up_el.text or "").strip() if up_el is not None else "Y_UP"
    materials = _effects(root)
    geoms = _geometries(root)
    scenes = {vs.get("id"): vs for vs in root.iter(NS + "visual_scene")}
    sc = root.find(NS + "scene/" + NS + "instance_visual_scene")
    if sc is not None:
        vs = scenes.get((sc.get("url") or "").lstrip("#"))
    else:
        vs = next(iter(scenes.values()), None)
    if vs is None:
        raise MakerStdError("%s: no visual scene" % name)
    prims = []
    state = {"v": 0, "t": 0}

    def walk(node, parent):
        M = mat4_mul(parent, node_local_matrix(node))
        for ch in node:
            kind = _tag(ch)
            if kind == "instance_geometry":
                gid = (ch.get("url") or "").lstrip("#")
                if gid not in geoms:
                    raise MakerStdError("%s: node %r instances a missing geometry %r" % (name, node.get("id"), gid))
                bind = {}
                for im in ch.iter(NS + "instance_material"):
                    bind[im.get("symbol")] = (im.get("target") or "").lstrip("#")
                g = geoms[gid]
                for k, pr in enumerate(g["prims"]):
                    sym = pr["material_symbol"]
                    mat = materials.get(bind.get(sym, sym))
                    p = Primitive(node_id=node.get("id"), node_name=node.get("name") or node.get("id"), geometry_id=gid,
                                  index_in_geometry=k, material_symbol=sym, material=mat, positions=g["positions"],
                                  node_matrix=M, vertex_index=pr["vertex_index"], normal_corners=pr["normals"],
                                  uv_corners=pr["uvs"], first_vertex=state["v"], first_triangle=state["t"])
                    state["v"] += g["positions"].shape[0]
                    state["t"] += p.triangle_count
                    prims.append(p)
            elif kind == "node":
                walk(ch, M)
            elif kind in ("instance_node", "instance_controller"):
                raise MakerStdError("%s: <%s> is not supported" % (name, kind))

    for n in vs.findall(NS + "node"):
        walk(n, identity4())
    return Mesh(name, unit_meter, unit_name, up_axis, prims)


def read(maker_root, maker_file, pins=None):
    """Mesh of the pinned maker file meshes/<maker_file> (pin checked, xz read transparently)."""
    logical = "meshes/" + maker_file
    return parse(pinned.read(maker_root, logical, pins), maker_file)


def read_path(path):
    """Mesh of an unpinned COLLADA file on disk (development use: the downloads, test copies)."""
    with open(path, "rb") as fh:
        return parse(fh.read(), path.replace("\\", "/").split("/")[-1])
