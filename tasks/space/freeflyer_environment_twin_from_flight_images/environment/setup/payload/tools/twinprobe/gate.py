"""The validity gate of one flight of a submission (rule Z2), and the submission's size check (part of rule Z1).

    res = check_flight(flight_dir, key_frames, cfg)     # res.ok, res.reasons, res.lines, res.model, res.poses
    res = gate_and_scene(flight_dir, key_frames, cfg)   # the same, plus res.scene (raycast) and res.solids
    why = submission_size(submission_dir, cfg)          # [] or the reasons the whole submission is too large

A flight's folder holds scene.xml, the mesh files it names and cameras.json. Nothing is compiled before every static
check has passed, and the simulator only sees bytes this module has read and checked:
  1. scene.xml is a regular file at the flight's root (no symbolic links anywhere on the way);
  2. scene.xml, statically: UTF-8, no DOCTYPE, ENTITY, namespaces or processing instructions; at most
     caps.xml_elements_per_flight XML elements, counted while the text is read and before anything else parses it;
     the root is <mujoco>; every element and attribute is on the allow list below (anything else fails the gate);
     at most caps.geoms_per_flight geoms, caps.bodies_per_flight bodies and caps.mesh_assets_per_flight mesh assets,
     counted in the text before the simulator sees it;
  3. every mesh file: a relative path without '..' inside the flight's folder, a regular file, .stl (binary) or
     .obj, a consistent header; at most caps.stl_faces_per_file triangles in one STL file (the simulator's reader
     refuses more; OBJ has no such limit);
  4. the caps of one flight: at most caps.mesh_faces_per_flight triangles, counted per mesh asset and summed over
     the mesh assets (an OBJ polygon of n corners counts n - 2; a file named by two assets counts twice); at most
     caps.mesh_megabytes_per_flight MB of mesh files; one MB is contract.megabyte_bytes bytes in every size cap;
  5. a copy of the scene is compiled from memory (mesh paths written out in full, compiled on one thread); a
     compile error fails the gate, a mesh whose vertices all lie in one plane with the message MESSAGES["flat_mesh"];
     the simulator's warnings are reported in the gate's lines and fail the gate only if gate.warnings_fail is true;
  6. the compiled scene: only geoms of the allowed types, no joint, camera, light, site or texture; the counts of
     geoms, bodies and mesh assets again within their caps; every number finite; every geom's extent within
     gate.coordinate_bound metres of the origin on each axis;
  7. cameras.json: a JSON object of format "flight-twin-cameras/1" with a "frames" object (posechain); a key frame
     without a valid pose is not a gate failure, it is that frame's zero (rule Z4).

Allowed MJCF. <mujoco model>; <compiler angle meshdir> (angle "degree", the default, or "radian": euler angles
and axis-angles are read in this unit); <asset> with <mesh name file scale inertia> and <material name rgba>;
<default class> (nested defaults allowed) holding <geom> defaults; <worldbody>; <body name childclass pos quat euler
axisangle xyaxes zaxis> without joints (bodies may nest); <geom> of type box, sphere, capsule, cylinder, ellipsoid
or mesh (a geom may name a material). Every geom counts as collision geometry, whatever its contype, conaffinity,
group, colour or material (raycast rebuilds every geom opaque and without material before casting and the report
says so).
Materials. A material that carries a colour only (rgba) is accepted and ignored: no geom keeps it in the ray scene.
A texture, a material's texture attributes or layers, and any attribute that loads a file other than a mesh's
file stay refused, with the message MESSAGES["texture"].

Errors. check_flight never raises for a bad submission; it raises KeyError when the configuration lacks a number
and lets MemoryError through: both are task errors, never a verdict on the twin.
"""
import io
import json
import os
import re
import stat as _stat
import struct
import sys
import tempfile
import warnings as _warnings
import xml.etree.ElementTree as ET

import numpy as np

from . import need
from . import posechain

GATE_VERSION = "flight-twin-gate/1"
MAX_REASONS = 200          # reasons and warnings kept in a report (a bound on its length, not a rule)
COUNT_CHUNK = 1 << 18      # characters fed to the element counter at a time (a reading step, not a rule)

# the messages the contract quotes for two compile traps
MESSAGES = {
    "stl_faces": "mesh file {file}: {n} triangles in one STL file; the simulator reads at most {cap} triangles from "
                 "an STL file, so the scene cannot be compiled (write a larger mesh as an OBJ file)",
    "flat_mesh": "mesh {mesh}: all its vertices lie in one plane, so it has no convex hull and the scene cannot be "
                 "compiled (give a surface patch a thickness, or use a thin box)",
    "texture": "textures are not allowed: a material may carry an rgba colour only, and it is ignored (every geom "
               "is cast opaque and without material)",
}

# ---------------------------------------------------------------------------------------------------------------
# the allow list

_POSE = {"pos", "quat", "axisangle", "xyaxes", "zaxis", "euler"}
GEOM_ATTRS = {"type", "size", "fromto", "mesh", "rgba", "material", "group", "contype", "conaffinity", "condim",
              "priority", "friction", "mass", "density", "shellinertia", "solmix", "solref", "solimp", "margin", "gap",
              "fitscale"} | _POSE
ELEMENTS = {
    "mujoco": ((None,), {"model"}),
    "compiler": (("mujoco",), {"angle", "meshdir"}),
    "asset": (("mujoco",), set()),
    "mesh": (("asset",), {"name", "file", "scale", "inertia"}),
    "material": (("asset",), {"name", "rgba"}),
    "default": (("mujoco", "default"), {"class"}),
    "worldbody": (("mujoco",), set()),
    "body": (("worldbody", "body"), {"name", "childclass"} | _POSE),
    "geom": (("worldbody", "body", "default"), {"name", "class"} | GEOM_ATTRS),
}
DEFAULT_CHILD_ATTRS = {"geom": GEOM_ATTRS}
GEOM_TYPES = ("box", "sphere", "capsule", "cylinder", "ellipsoid", "mesh")
FORBIDDEN_GEOM_TYPES = {
    "plane": "planes are not allowed (use a thin box for a wall or a deck)",
    "hfield": "height fields are not allowed",
    "sdf": "signed-distance geoms are not allowed",
}
# elements and attributes that bring a texture: refused with MESSAGES["texture"]
TEXTURE_TAGS = {"texture", "layer"}
TEXTURE_ATTRS = {"texture", "texrepeat", "texuniform"}
FORBIDDEN_WHY = {
    "joint": "no joints: the twin is static geometry", "freejoint": "no joints: the twin is static geometry",
    "camera": "no cameras: camera poses go into cameras.json", "light": "no lights",
    "site": "no sites", "inertial": "no inertial elements",
    "frame": "no frame elements (use bodies)", "include": "no includes: the twin is one scene.xml",
    "option": "no option element", "visual": "no visual element", "statistic": "no statistic element",
    "size": "no size element", "hfield": "no height fields", "keyframe": "no keyframes",
    "contact": "no contact pairs or excludes", "equality": "no equalities", "tendon": "no tendons",
    "actuator": "no actuators", "sensor": "no sensors", "custom": "no custom data",
    "extension": "no plugins", "plugin": "no plugins", "deformable": "no flex or skin", "flexcomp": "no flex",
    "composite": "no composites", "replicate": "no replicate", "attach": "no attached models",
}
_MATERIAL_ONLY_RGBA = "a material may carry an rgba colour only (it is ignored: every geom is cast without material)"
FORBIDDEN_ATTR_WHY = {
    "emission": _MATERIAL_ONLY_RGBA, "specular": _MATERIAL_ONLY_RGBA, "shininess": _MATERIAL_ONLY_RGBA,
    "reflectance": _MATERIAL_ONLY_RGBA, "metallic": _MATERIAL_ONLY_RGBA, "roughness": _MATERIAL_ONLY_RGBA,
    "vertex": "a mesh is read from a binary STL or an OBJ file", "face": "a mesh is read from a binary STL or an OBJ file",
    "normal": "a mesh is read from a binary STL or an OBJ file",
    "texcoord": "a mesh is read from a binary STL or an OBJ file",
    "maxhullvert": "a mesh collides as the hull of all its vertices",
    "content_type": "the file extension decides the format", "builtin": "built-in meshes are not allowed",
    "texturedir": "only meshdir may be set", "assetdir": "only meshdir may be set",
    "eulerseq": "euler angles are read in the default sequence",
}
COMPILER_ANGLES = ("degree", "radian")
MESH_INERTIA = ("convex", "exact", "legacy", "shell")
MESH_EXT = (".stl", ".obj")
PATH_RX = re.compile(r"^[A-Za-z0-9_.+\- ()/]+$")


def rules_from_config(cfg):
    megabyte = int(need(cfg, "contract.megabyte_bytes"))
    warnings_fail = need(cfg, "gate.warnings_fail")
    if not isinstance(warnings_fail, bool):
        raise ValueError("the configuration's gate.warnings_fail must be true or false, not %r" % (warnings_fail,))
    return {"geoms": int(need(cfg, "caps.geoms_per_flight")),
            "bodies": int(need(cfg, "caps.bodies_per_flight")),
            "mesh_assets": int(need(cfg, "caps.mesh_assets_per_flight")),
            "xml_elements": int(need(cfg, "caps.xml_elements_per_flight")),
            "mesh_faces": int(need(cfg, "caps.mesh_faces_per_flight")),
            "mesh_bytes": int(round(float(need(cfg, "caps.mesh_megabytes_per_flight")) * megabyte)),
            "stl_faces": int(need(cfg, "caps.stl_faces_per_file")),
            "coordinate_bound": float(need(cfg, "gate.coordinate_bound")),
            "files": int(need(cfg, "caps.files")),
            "bytes": int(round(float(need(cfg, "caps.megabytes")) * megabyte)),
            "megabyte": megabyte,
            "warnings_fail": warnings_fail}


class GateResult:
    def __init__(self):
        self.reasons = []        # why the flight fails the gate (empty: it passes)
        self.warnings = []
        self.lines = []          # the report, line by line, as printed
        self.report = {"gate": GATE_VERSION}
        self.model = None        # the compiled copy (mujoco.MjModel) when every check passed
        self.spec = None
        self.poses = {}          # key frame -> T_twin_wide (valid declared poses)
        self.frame_errors = {}   # key frame -> why it has no valid pose (rule Z4)
        self.scene = None
        self.solids = None

    @property
    def ok(self):
        return not self.reasons

    def fail(self, msg):
        if len(self.reasons) < MAX_REASONS:
            self.reasons.append(str(msg))

    def warn(self, msg):
        if len(self.warnings) < MAX_REASONS:
            self.warnings.append(str(msg))


# ---------------------------------------------------------------------------------------------------------------
# files and paths

def tree_stats(root):
    """(entries, bytes, irregular entries) of a folder, without following symbolic links."""
    entries, total, irregular = 0, 0, []
    for top, dirs, files in os.walk(root, followlinks=False):
        for nm in list(dirs):
            p = os.path.join(top, nm)
            entries += 1
            if os.path.islink(p):
                irregular.append(os.path.relpath(p, root) + " (symbolic link)")
                dirs.remove(nm)
        for nm in files:
            p = os.path.join(top, nm)
            entries += 1
            try:
                st = os.lstat(p)
            except OSError:
                irregular.append(os.path.relpath(p, root) + " (unreadable)")
                continue
            if _stat.S_ISLNK(st.st_mode):
                irregular.append(os.path.relpath(p, root) + " (symbolic link)")
            elif not _stat.S_ISREG(st.st_mode):
                irregular.append(os.path.relpath(p, root) + " (not a regular file)")
            else:
                total += int(st.st_size)
    return entries, total, irregular


def submission_size(root, cfg):
    """Reasons the whole submission folder breaks the size caps (files and folders, megabytes) or holds an
    irregular entry; [] if it does not."""
    rules = rules_from_config(cfg)
    entries, total, irregular = tree_stats(root)
    why = []
    if entries > rules["files"]:
        why.append("the submission holds %d files and folders (cap %d)" % (entries, rules["files"]))
    if total > rules["bytes"]:
        why.append("the submission holds %d bytes (cap %d)" % (total, rules["bytes"]))
    for x in irregular[:MAX_REASONS]:
        why.append("irregular entry: %s" % x)
    return why


def _regular_file(root, rel):
    """Absolute path of rel inside root if every component exists without symbolic links and the last is a regular
    file; else (None, why)."""
    parts = rel.split("/")
    cur = root
    for k, part in enumerate(parts):
        cur = os.path.join(cur, part)
        try:
            st = os.lstat(cur)
        except OSError:
            return None, "does not exist"
        if _stat.S_ISLNK(st.st_mode):
            return None, "goes through a symbolic link"
        if k < len(parts) - 1 and not _stat.S_ISDIR(st.st_mode):
            return None, "a path component is not a folder"
        if k == len(parts) - 1 and not _stat.S_ISREG(st.st_mode):
            return None, "is not a regular file"
    real = os.path.realpath(cur)
    rroot = os.path.realpath(root)
    if not (real == rroot or real.startswith(rroot + os.sep)):
        return None, "resolves outside the flight's folder"
    return cur, None


def _path_problem(v, is_dir=False):
    v = str(v)
    if is_dir and v.endswith("/"):
        v = v[:-1]
    if not v:
        return "is empty" if not is_dir else None
    if len(v) > 240:
        return "is too long"
    if any(ord(ch) < 32 for ch in v):
        return "contains control characters"
    if "\\" in v:
        return "contains a backslash (use '/')"
    if ":" in v:
        return "contains ':'"
    if v.startswith("/") or v.startswith("~"):
        return "is an absolute path"
    parts = v.split("/")
    if any(p == ".." for p in parts):
        return "contains a '..' component"
    if any(p == "" for p in parts):
        return "contains an empty path component"
    if not PATH_RX.match(v):
        return "contains characters other than letters, digits, space and _ . + - ( ) /"
    return None


def _join_rel(*parts):
    out = []
    for p in parts:
        if not p:
            continue
        for q in p.split("/"):
            if q in ("", "."):
                continue
            out.append(q)
    return "/".join(out)


def _read(path, limit, res, what):
    try:
        with open(path, "rb") as fh:
            data = fh.read(limit + 1)
    except OSError as exc:
        res.fail("%s cannot be read: %s" % (what, getattr(exc, "strerror", None) or type(exc).__name__))
        return None
    if len(data) > limit:
        res.fail("%s is larger than %d bytes" % (what, limit))
        return None
    return data


# ---------------------------------------------------------------------------------------------------------------
# scene.xml, statically

def count_elements(text, cap):
    """(n, over): the XML elements of text, counted by a streaming parse that stops as soon as more than cap are
    seen (the simulator has not seen the text yet); a text that does not parse is left to the full parse."""
    parser = ET.XMLPullParser(events=("start",))
    n = 0
    try:
        for i in range(0, len(text), COUNT_CHUNK):
            parser.feed(text[i:i + COUNT_CHUNK])
            n += sum(1 for _ in parser.read_events())
            if n > cap:
                return n, True
        parser.close()
        n += sum(1 for _ in parser.read_events())
    except ET.ParseError:
        pass
    return n, n > cap


def static_scene(raw, res, max_elements=None):
    """Parse and check scene.xml bytes against the allow list. Returns the ElementTree root or None. With
    max_elements (the gate passes caps.xml_elements_per_flight), a text of more XML elements fails before it is
    parsed into a tree."""
    if b"\x00" in raw:
        res.fail("scene.xml contains NUL bytes")
        return None
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        res.fail("scene.xml is not valid UTF-8")
        return None
    m = re.match(r"\s*<\?xml[^>]*encoding\s*=\s*['\"]([^'\"]+)['\"]", text)
    if m and m.group(1).strip().lower() not in ("utf-8", "utf8", "us-ascii", "ascii"):
        res.fail("scene.xml declares encoding %r (UTF-8 required)" % m.group(1))
        return None
    up = text.upper()
    if "<!DOCTYPE" in up or "<!ENTITY" in up:
        res.fail("DOCTYPE / ENTITY declarations are not allowed")
        return None
    if "XMLNS" in up:
        res.fail("XML namespaces (xmlns) are not allowed")
        return None
    body = text.lstrip()
    if body.startswith("<?xml"):
        body = body[body.find("?>") + 2:]
    if "<?" in body:
        res.fail("XML processing instructions are not allowed")
        return None
    if max_elements is not None:
        n, over = count_elements(text, int(max_elements))
        res.report["xml_elements"] = int(n)
        if over:
            res.fail("scene.xml holds more than %d XML elements (cap %d)" % (int(max_elements), int(max_elements)))
            return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        res.fail("scene.xml does not parse as XML: %s" % exc)
        return None
    if root.tag != "mujoco":
        res.fail("the root element is <%s>, not <mujoco>" % root.tag)
        return None
    stack = [(root, None, False)]
    while stack:
        el, parent, in_default = stack.pop()
        tag = el.tag
        if not isinstance(tag, str):
            res.fail("XML comments or processing instructions inside elements are not allowed")
            continue
        if tag in TEXTURE_TAGS:
            res.fail(MESSAGES["texture"])
            continue
        if tag not in ELEMENTS:
            why = FORBIDDEN_WHY.get(tag)
            res.fail("<%s> is not allowed%s" % (tag, (": " + why) if why else " (not on the allow list)"))
            continue
        parents, attrs = ELEMENTS[tag]
        if parent not in parents:
            res.fail("<%s> is not allowed inside <%s>" % (tag, parent))
            continue
        if in_default and tag != "default":
            attrs = DEFAULT_CHILD_ATTRS.get(tag, set())
        if (el.text or "").strip() or (el.tail or "").strip():
            res.fail("<%s> carries text content (attributes only)" % tag)
        for k, v in el.attrib.items():
            if k in TEXTURE_ATTRS:
                res.fail(MESSAGES["texture"])
                continue
            if k not in attrs:
                why = FORBIDDEN_ATTR_WHY.get(k)
                res.fail("<%s %s=...> is not allowed%s" % (tag, k, (": " + why) if why else
                                                            " (not on the attribute list of <%s>%s)" %
                                                            (tag, " inside <default>" if in_default else "")))
                continue
            _check_value(tag, k, v, res)
        if len(res.reasons) >= MAX_REASONS:
            return None
        for ch in reversed(list(el)):
            stack.append((ch, tag, in_default or tag == "default"))
    if len(root.findall("compiler")) > 1:
        res.fail("only one <compiler> element is allowed")
    return None if res.reasons else root


def _check_value(tag, k, v, res):
    v = str(v).strip()
    if tag == "geom" and k == "type":
        if v in FORBIDDEN_GEOM_TYPES:
            res.fail("<geom type=%r> is not allowed: %s" % (v, FORBIDDEN_GEOM_TYPES[v]))
        elif v not in GEOM_TYPES:
            res.fail("<geom type=%r> is not allowed (%s)" % (v, ", ".join(GEOM_TYPES)))
    elif tag == "compiler" and k == "angle" and v not in COMPILER_ANGLES:
        res.fail("<compiler angle=%r> must be \"degree\" or \"radian\"" % v)
    elif tag == "compiler" and k == "meshdir":
        why = _path_problem(v, is_dir=True)
        if why:
            res.fail("<compiler meshdir=%r> %s" % (v, why))
    elif tag == "mesh" and k == "inertia" and v not in MESH_INERTIA:
        res.fail("<mesh inertia=%r> is not allowed (%s)" % (v, ", ".join(MESH_INERTIA)))


# ---------------------------------------------------------------------------------------------------------------
# mesh files

def _stl_info(data):
    if len(data) < 84:
        return None, "is shorter than a binary STL header"
    n = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * n:
        return None, ("is not a consistent binary STL file (its header says %d triangles, its size is %d bytes; an "
                      "ASCII STL file is not read)" % (n, len(data)))
    if n == 0:
        return None, "holds no triangles"
    return {"faces": int(n)}, None


def _obj_info(data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    nv = nf = 0
    for line in io.StringIO(text):
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "v":
            nv += 1
        elif tok[0] == "f":
            nf += max(1, len(tok) - 3)
        elif tok[0] == "mtllib":
            for nm in tok[1:]:
                if "/" in nm or "\\" in nm or ":" in nm or nm in (".", "..") or nm.startswith("~"):
                    return None, "has mtllib %r (a material library must be a plain file name)" % nm
    if nf == 0 or nv == 0:
        return None, "holds no faces"
    return {"faces": int(nf)}, None


def check_meshes(flight_dir, tree, res, rules):
    """Check every mesh file the scene names. Returns {path relative to the flight's folder: bytes}; the triangles
    are counted per mesh asset (a file named by two assets counts twice: each becomes a mesh of the model)."""
    comp = tree.find("compiler")
    meshdir = comp.attrib.get("meshdir", "") if comp is not None else ""
    files, faces, total_bytes, total_faces = {}, {}, 0, 0
    for k, el in enumerate(tree.iter("mesh")):
        f = el.attrib.get("file")
        if f is None:
            res.fail("<mesh name=%r> has no file (a mesh is read from a binary STL or an OBJ file)"
                     % el.attrib.get("name", ""))
            continue
        why = _path_problem(f)
        if why:
            res.fail("<mesh file=%r> %s" % (f, why))
            continue
        rel = _join_rel(meshdir, f)
        ext = os.path.splitext(rel)[1].lower()
        if ext not in MESH_EXT:
            res.fail("<mesh file=%r>: mesh files must be .stl (binary) or .obj" % f)
            continue
        path, why = _regular_file(flight_dir, rel)
        if why:
            res.fail("<mesh file=%r> (%s) %s" % (f, rel, why))
            continue
        if rel not in files:
            data = _read(path, rules["mesh_bytes"], res, "mesh file %s" % rel)
            if data is None:
                continue
            info, why = (_stl_info if ext == ".stl" else _obj_info)(data)
            if why:
                res.fail("mesh file %s %s" % (rel, why))
                continue
            if ext == ".stl" and info["faces"] > rules["stl_faces"]:
                res.fail(MESSAGES["stl_faces"].format(file=rel, n=info["faces"], cap=rules["stl_faces"]))
                continue
            files[rel] = data
            total_bytes += len(data)
            faces[rel] = info["faces"]
        total_faces += faces[rel]
    if total_bytes > rules["mesh_bytes"]:
        res.fail("the mesh files hold %d bytes (cap %d)" % (total_bytes, rules["mesh_bytes"]))
    if total_faces > rules["mesh_faces"]:
        res.fail("the mesh assets hold %d triangles in all (cap %d)" % (total_faces, rules["mesh_faces"]))
    res.report["mesh_files"] = len(files)
    res.report["mesh_bytes"] = int(total_bytes)
    res.report["mesh_triangles"] = int(total_faces)
    return files


# ---------------------------------------------------------------------------------------------------------------
# the compile

class _Capture:
    """Collect the simulator's warnings: its warning callback, Python warnings and file descriptor 2."""

    def __init__(self):
        self.messages = []

    def __enter__(self):
        import mujoco
        self._mj = mujoco
        self._wctx = _warnings.catch_warnings(record=True)
        self._wrec = self._wctx.__enter__()
        _warnings.simplefilter("always")
        try:
            mujoco.set_mju_user_warning(lambda msg: self.messages.append(str(msg).strip()))
            self._cb = True
        except Exception:  # noqa: BLE001
            self._cb = False
        try:
            sys.stderr.flush()
        except Exception:  # noqa: BLE001
            pass
        self._saved = os.dup(2)
        self._tmp = tempfile.TemporaryFile()
        os.dup2(self._tmp.fileno(), 2)
        return self

    def __exit__(self, *exc):
        os.dup2(self._saved, 2)
        os.close(self._saved)
        self._tmp.seek(0)
        text = self._tmp.read().decode("utf-8", "replace").strip()
        self._tmp.close()
        if self._cb:
            try:
                self._mj.set_mju_user_warning(None)
            except Exception:  # noqa: BLE001
                pass
        for w in self._wrec:
            msg = str(w.message).strip()
            if msg and msg not in self.messages:
                self.messages.append(msg[:600])
        self._wctx.__exit__(None, None, None)
        if text and text not in self.messages:
            self.messages.append(text[:600])
        return False


def grader_copy(tree):
    """XML text of the compiled copy: mesh paths written out in full (relative to the flight's folder), the
    compiler's meshdir dropped."""
    import copy
    root = copy.deepcopy(tree)
    comp = root.find("compiler")
    meshdir = ""
    if comp is not None:
        meshdir = comp.attrib.pop("meshdir", "")
    for el in root.iter("mesh"):
        if "file" in el.attrib:
            el.attrib["file"] = _join_rel(meshdir, el.attrib["file"])
    return ET.tostring(root, encoding="unicode")


def _mesh_name_from_error(msg):
    m = re.search(r"mesh\s+'?([^\s']+)'?\s+has coplanar", msg)
    return m.group(1) if m else "?"


def compile_copy(xml_text, files, res):
    import mujoco
    spec = model = None
    with _Capture() as cap:
        try:
            spec = mujoco.MjSpec.from_string(xml_text, assets=dict(files))
            spec.compiler.usethread = False
            model = spec.compile()
        except MemoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - the simulator raises ValueError for parse and compile errors
            msg = str(exc).strip()[:600]
            if "coplanar" in msg:
                res.fail(MESSAGES["flat_mesh"].format(mesh=_mesh_name_from_error(msg)))
            else:
                res.fail("scene.xml does not compile: %s" % msg)
            model = None
    for msg in cap.messages:
        if msg:
            res.warn("simulator: %s" % msg)
    return spec, model


# ---------------------------------------------------------------------------------------------------------------
# the compiled scene

def _geom_extent(m, d, g):
    """(lo, hi) world axis-aligned extent of geom g."""
    import mujoco
    T = mujoco.mjtGeom
    t = int(m.geom_type[g])
    p = np.asarray(d.geom_xpos[g], float)
    R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
    s = np.asarray(m.geom_size[g], float)
    if t == int(T.mjGEOM_MESH):
        mid = int(m.geom_dataid[g])
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        W = np.asarray(m.mesh_vert[va:va + vn], float) @ R.T + p
        return W.min(axis=0), W.max(axis=0)
    if t == int(T.mjGEOM_BOX):
        h = np.abs(R) @ s[:3]
    elif t == int(T.mjGEOM_SPHERE):
        h = np.full(3, s[0])
    elif t == int(T.mjGEOM_CAPSULE):
        h = np.abs(R[:, 2]) * s[1] + s[0]
    elif t == int(T.mjGEOM_CYLINDER):
        h = np.abs(R[:, 2]) * s[1] + s[0] * np.sqrt(np.clip(1.0 - R[:, 2] ** 2, 0.0, 1.0))
    elif t == int(T.mjGEOM_ELLIPSOID):
        h = np.sqrt((R * s[:3][None, :]) ** 2 @ np.ones(3))
    else:
        h = np.full(3, np.inf)
    return p - h, p + h


def compiled_checks(model, res, rules):
    import mujoco
    m = model
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)                  # poses only: a collision pass could exhaust the simulator's stack
    counts = {"geoms": int(m.ngeom), "bodies": int(m.nbody - 1), "meshes": int(m.nmesh), "materials": int(m.nmat)}
    res.report["counts"] = counts
    for what, n in (("joints", m.njnt), ("cameras", m.ncam), ("lights", m.nlight), ("sites", m.nsite),
                    ("textures", m.ntex), ("height fields", m.nhfield)):
        if int(n):
            res.fail("the compiled scene has %d %s (none allowed)" % (int(n), what))
    for what, key, n in (("geoms", "geoms", counts["geoms"]), ("bodies", "bodies", counts["bodies"]),
                         ("mesh assets", "mesh_assets", counts["meshes"])):
        if n > rules[key]:
            res.fail("the scene has %d %s (cap %d)" % (n, what, rules[key]))
    allowed = {int(getattr(mujoco.mjtGeom, "mjGEOM_" + t.upper())) for t in GEOM_TYPES}
    bad_type = [g for g in range(m.ngeom) if int(m.geom_type[g]) not in allowed]
    if bad_type:
        res.fail("%d geoms have a type that is not allowed" % len(bad_type))
        return
    arrays = (d.geom_xpos, d.geom_xmat, m.geom_size, m.mesh_vert if m.nmesh else np.zeros((0, 3)))
    if not all(np.all(np.isfinite(np.asarray(a, float))) for a in arrays):
        res.fail("the compiled scene holds numbers that are not finite")
        return
    bound = rules["coordinate_bound"]
    far = []
    for g in range(m.ngeom):
        lo, hi = _geom_extent(m, d, g)
        if np.any(np.abs(lo) > bound) or np.any(np.abs(hi) > bound):
            far.append(g)
    if far:
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, far[0]) or ("geom %d" % far[0])
        res.fail("%d geoms reach beyond %g m of the origin on some axis (first: %s)" % (len(far), bound, nm))


# ---------------------------------------------------------------------------------------------------------------
# one flight

def check_flight(flight_dir, key_frames, cfg):
    """The validity gate of one flight's folder (scene.xml, its mesh files, cameras.json) for the given key frames."""
    res = GateResult()
    rules = rules_from_config(cfg)
    res.report["rules"] = rules
    path, why = _regular_file(flight_dir, "scene.xml")
    if why:
        res.fail("scene.xml %s (a regular file at the flight's root is required)" % why)
    tree = None
    if path:
        raw = _read(path, rules["bytes"], res, "scene.xml")
        if raw is not None:
            tree = static_scene(raw, res, max_elements=rules["xml_elements"])
    files = None
    if tree is not None:
        static = static_counts(tree)
        res.report["static_counts"] = static
        over = False
        for what, key in (("geoms", "geoms"), ("bodies", "bodies"), ("mesh assets", "mesh_assets")):
            if static[key] > rules[key]:
                res.fail("the scene has %d %s (cap %d)" % (static[key], what, rules[key]))
                over = True
        if not over:
            files = check_meshes(flight_dir, tree, res, rules)
    if tree is not None and not res.reasons:
        spec, model = compile_copy(grader_copy(tree), files, res)
        if model is not None and rules["warnings_fail"]:
            for w in list(res.warnings):
                res.fail("a simulator warning fails the gate under this configuration: %s" % w)
        if model is not None:
            compiled_checks(model, res, rules)
        if not res.reasons:
            res.spec, res.model = spec, model
    # cameras.json
    cpath, why = _regular_file(flight_dir, "cameras.json")
    if why:
        res.fail("cameras.json %s (a regular file at the flight's root is required)" % why)
    else:
        raw = _read(cpath, rules["bytes"], res, "cameras.json")
        if raw is not None:
            poses, errors, file_error = posechain.parse_cameras(raw, list(key_frames), cfg)
            if file_error:
                res.fail(file_error)
            else:
                res.poses, res.frame_errors = poses, errors
    res.report["key_frames_with_valid_pose"] = len(res.poses)
    res.report["key_frames_without_valid_pose"] = dict(sorted(res.frame_errors.items()))
    res.lines.append("validity gate (%s): %s" % (GATE_VERSION, "PASS" if res.ok else "FAIL"))
    for r in res.reasons:
        res.lines.append("  reason: %s" % r)
    for w in res.warnings:
        res.lines.append("  warning: %s" % w)
    if res.warnings and not rules["warnings_fail"]:
        res.lines.append("  %d simulator warning(s) reported; a warning does not fail the gate" % len(res.warnings))
    if res.ok:
        c = res.report.get("counts", {})
        res.lines.append("  %d geoms, %d bodies, %d mesh assets, %d mesh triangles"
                         % (c.get("geoms", 0), c.get("bodies", 0), c.get("meshes", 0), res.report.get("mesh_triangles", 0)))
        if c.get("materials", 0):
            res.lines.append("  %d material(s) with a colour only: accepted and ignored (every geom is cast without "
                             "material)" % c["materials"])
        res.lines.append("  key frames with a valid pose: %d of %d" % (len(res.poses), len(list(key_frames))))
        for fr, why in sorted(res.frame_errors.items()):
            res.lines.append("  %s: no valid pose (%s): rule Z4, this frame scores 0" % (fr, why))
    return res


def static_counts(tree):
    """Geoms (outside defaults), bodies and mesh assets of a checked scene tree, counted before any compile."""
    in_defaults = sum(1 for dflt in tree.iter("default") for _ in dflt.findall("geom"))
    return {"geoms": sum(1 for _ in tree.iter("geom")) - in_defaults,
            "bodies": sum(1 for _ in tree.iter("body")),
            "mesh_assets": sum(1 for _ in tree.iter("mesh"))}


def gate_and_scene(flight_dir, key_frames, cfg):
    """check_flight, then (when it passes) the ray scene and the solids of the compiled copy; the ray scene's
    rebuild is reported in res.lines."""
    from .insidesolid import Solids
    from .raycast import RayScene
    res = check_flight(flight_dir, key_frames, cfg)
    if res.ok:
        import mujoco
        d = mujoco.MjData(res.model)
        mujoco.mj_kinematics(res.model, d)
        res.scene = RayScene(res.model, d)
        res.solids = Solids(res.model, d, scene=res.scene)
        for ln in res.scene.report_lines():
            res.lines.append("  " + ln)
    return res


def report_json(res):
    """The gate's report as JSON text (no host paths: only names relative to the flight's folder)."""
    out = dict(res.report)
    out["pass"] = res.ok
    out["reasons"] = list(res.reasons)
    out["warnings"] = list(res.warnings)
    return json.dumps(out, indent=1, sort_keys=True, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x))
