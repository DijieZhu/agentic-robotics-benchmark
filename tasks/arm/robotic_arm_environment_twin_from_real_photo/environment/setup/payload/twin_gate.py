"""twin_gate: the validity gate (G0) of an environment-twin submission.

The same file runs in the check tool you are given and in the grader, so a twin that passes here passes G0 there. It
needs only the Python standard library, numpy and mujoco (plus twin_sim.py next to it for the settle check), and it
reads no ground truth.

    python3 twin_gate.py SUBMISSION_DIR [--vocab product_names.json] [--json] [--settle]

    from twin_gate import gate, physics_model, blowup_check
    res = gate("/home/user/submission")        # res.ok, res.report (JSON-ready dict), res.model (mujoco.MjModel)
    chk = blowup_check(physics_model(res), drawer=res.report["drawer"])    # G0's settle part: chk["fail"]

Order of the checks. Nothing in the submission is compiled before every static check has passed, and MuJoCo only
ever sees bytes this module has read and checked:
  1. files: scene.xml and objects.json are regular files at the submission root; the whole folder is within the
     entry and byte caps; a file that cannot be read fails the gate with a reason;
  2. scene.xml, statically: size, UTF-8, no DOCTYPE / ENTITY / namespaces, then the element whitelist with an
     attribute allowlist per element (anything not listed is rejected);
  3. every referenced file: a relative path without '..' that resolves inside the submission, a regular file (no
     symbolic links on the way), the right extension, within the size caps, and a consistent header (binary STL
     triangle count, MSH counts, PNG size; an OBJ's mtllib must be a plain file name);
  4. objects.json against the twin-objects/1 schema;
  5. the grader copy: textures and texture references removed, every mesh path written out in full, compiled from
     memory; every MuJoCo warning is captured (its warning callback, Python warnings with the filter forced to
     "always", and anything written to stderr), so the verdict never depends on the interpreter's warning state;
     any warning fails the gate;
  6. compiled checks: no equality, tendon, actuator, flex, plugin, mocap, contact pair/exclude, keyframe or sensor;
     the caps; finite numbers; photo_cam; the bodies named in objects.json; the drawer; the joints;
  7. per item: visual geoms massless, collision-geom count, density, friction, inertia and centre of mass. A failure
     here excludes that item only; it does not fail the twin.

Collision geometry. A geom collides when (contype, conaffinity) != (0, 0); every other geom is visual (it must be
massless and plays no part in the scoring). The standard simulation gives every colliding geom the standard contact
classes, so the delivered bits beyond that make no difference. A mesh geom collides as the convex hull of its vertices
(MuJoCo's own hull is used here), a primitive as itself. The geometry helpers in this file (Piece, geom_piece,
grid_union, union_integrals) are the ones the grader's scoring code uses as well.

Per-item plausibility (step 7). M is the item's mass, C its centre of mass, I its inertia about C (all bodies of the
item together), d_max the largest distance from C to the item's collision geometry:
  - density M / V within density_kg_m3, V the volume of the union of the collision geometry;
  - sliding friction within friction;
  - smallest principal inertia >= inertia_min_ratio x M x L^2 (L the largest extent);
  - largest principal inertia <= (1 + inertia_max_tolerance) x M x d_max^2, the bound every mass distribution
    inside the collision geometry obeys (so no <compiler boundinertia>-style inflation can freeze an item);
  - C inside the convex hull of the collision geometry (within com_tolerance_m).

Excluded items. An item that fails step 7 scores 0 in every criterion, and it is DELETED from the model before any
physics: the G0 settle check, the standard settle, the support truth and every probe run on physics_model(res), the
grader copy without the excluded items' bodies. An excluded item can therefore neither support nor disturb another
item. It still counts as a submitted item in the inventory denominators.

G0's settle check (blowup_check). The standard simulation runs for blowup_window_s from the photographed state (the
drawer held, nothing else). G0 fails only when the twin blows up:
  - a non-finite state, or MuJoCo reporting bad accelerations;
  - mechanical-energy gain: KE(t) > PE(0) - PE(t) + eps, eps = 0.5 x (total item mass) x v_tol^2 (the kinetic
    energy of every item moving at v_tol), v_tol = blowup_speed_tol_m_s;
  - an item whose centre of mass moves faster than the free-fall bound g t + v_tol.
Nothing falling, sliding or toppling from rest can break these bounds; the physics criteria charge such motion. What
breaks them is the release of a deep interpenetration: under the standard contact parameters an item pushed out of
a penetration leaves at about 0.08 m/s per mm of depth, so items sharing more than about 25 mm of space with the
furniture or with each other fail G0.

Errors. gate() never raises for a bad submission. It raises TaskError when the task's own inputs are broken (the
product vocabulary) and lets MemoryError through: both are task errors, never a verdict on the twin.
"""
import io
import json
import math
import os
import re
import stat as _stat
import struct
import sys
import tempfile
import warnings as _warnings
import xml.etree.ElementTree as ET

import numpy as np

GATE_VERSION = "twin-gate/1.1.0"
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------------------------------
# public rules (the contract states the same numbers)

DEFAULT_RULES = {
    "max_entries": 5000,                     # files and directories in the whole submission folder
    "max_bytes": 300 * 1024 * 1024,          # bytes in the whole submission folder
    "max_scene_bytes": 5 * 1024 * 1024,
    "max_objects_bytes": 1024 * 1024,
    "max_mesh_file_bytes": 50 * 1024 * 1024,
    "max_png_file_bytes": 64 * 1024 * 1024,
    "max_png_side_px": 4096,
    "max_referenced_files": 2000,
    "max_mesh_assets": 2000,
    "max_textures": 500,
    "max_xml_elements": 20000,
    "max_xml_depth": 64,
    "max_item_hull_vertices": 50000,         # summed over an item's collision meshes
    "volume_grid_max_columns": 40000,        # work bound of the gate's union-volume integral
    "max_bodies": 120,                       # bodies other than the world
    "max_geoms": 3000,
    "max_collision_geoms": 1500,
    "max_collision_geoms_per_item": 64,
    "max_collision_mesh_vertices": 20000,    # per mesh used by a collision geom
    "max_mesh_faces_total": 3000000,
    "max_mesh_vertices_total": 3000000,
    "items": [1, 60],
    "camera": "photo_cam",
    "resolution": [1920, 1080],
    "focal_px": [500.0, 5000.0],
    "density_kg_m3": [10.0, 3000.0],
    "friction": [0.1, 1.5],
    "inertia_min_ratio": 1e-4,               # smallest principal inertia >= ratio * mass * L^2 (L: largest extent)
    "inertia_max_tolerance": 0.01,           # largest principal inertia <= (1 + tol) * mass * d_max^2
    "com_tolerance_m": 1e-6,
    "max_item_extent_m": 2.0,
    "max_abs_coordinate_m": 100.0,
    "max_geom_size_m": 100.0,
    "volume_grid_max_m": 0.003,              # column spacing of the union-volume integral (finer for small items)
    "volume_grid_min_m": 0.0005,
    "volume_grid_max_piece_columns": 400000, # work bound: sum over pieces of the columns their boxes cover
    "blowup_window_s": 2.0,                  # G0's settle check runs this long
    "blowup_speed_tol_m_s": 2.0,             # v_tol of the free-fall and energy bounds
    # the drawer's travel, a plausibility rule like an item's (it never fails G0; a drawer that breaks it scores 0 in
    # C5's drawer term): a limited slide, the photographed opening (qpos0) inside the range, a total travel between
    # these fractions of the drawer's own depth along its slide, and at least min_out_mm more pull-out from qpos0
    "drawer_travel": {"min_depth_ratio": 0.5, "max_depth_ratio": 1.0, "min_out_mm": 50.0},
}

FORMAT_OBJECTS = "twin-objects/1"

# ---------------------------------------------------------------------------------------------------------------
# the element whitelist: allowed parents and allowed attributes per element. Anything else is rejected.

_POSE = {"pos", "quat", "axisangle", "xyaxes", "zaxis", "euler"}
_GEOM_DEF = {"type", "contype", "conaffinity", "condim", "group", "priority", "size", "material", "rgba", "friction",
             "mass", "density", "shellinertia", "solmix", "solref", "solimp", "margin", "gap", "fromto", "mesh",
             "fitscale"} | _POSE
_SITE_DEF = {"type", "group", "size", "material", "rgba", "fromto"} | _POSE
_CAMERA_DEF = {"mode", "fovy", "ipd", "resolution", "focal", "focalpixel", "principal", "principalpixel",
               "sensorsize", "projection"} | _POSE
_LIGHT_DEF = {"mode", "directional", "type", "castshadow", "active", "pos", "dir", "attenuation", "cutoff",
              "exponent", "ambient", "diffuse", "specular", "bulbradius", "intensity", "range", "softness"}
_MATERIAL_DEF = {"texture", "texrepeat", "texuniform", "emission", "specular", "shininess", "reflectance",
                 "metallic", "roughness", "rgba"}
_JOINT_DEF = {"type", "axis", "pos", "range", "limited", "group"}
_BODYLIKE = ("worldbody", "body", "frame")

ELEMENTS = {
    "mujoco": ((None,), {"model"}),
    "compiler": (("mujoco",), {"angle", "autolimits", "eulerseq", "meshdir", "texturedir", "assetdir",
                               "inertiafromgeom", "inertiagrouprange", "fitaabb", "usethread", "discardvisual"}),
    "option": (("mujoco",), {"timestep", "gravity", "integrator", "cone", "impratio", "solver", "iterations",
                             "ls_iterations", "tolerance", "ls_tolerance", "noslip_iterations", "noslip_tolerance",
                             "ccd_iterations", "ccd_tolerance", "jacobian", "sdf_iterations", "sdf_initpoints"}),
    "flag": (("option",), {"actuation", "autoreset", "clampctrl", "constraint", "contact", "damper", "diagexact",
                           "energy", "equality", "eulerdamp", "filterparent", "frictionloss", "fwdinv", "gravity",
                           "invdiscrete", "ipc", "island", "limit", "midphase", "multiccd", "nativeccd", "refsafe",
                           "sensor", "spring", "warmstart"}),
    "statistic": (("mujoco",), {"center", "extent"}),
    "visual": (("mujoco",), set()),
    "global": (("visual",), {"azimuth", "bvactive", "cameraid", "elevation", "ellipsoidinertia", "fovy", "glow",
                             "ipd", "linewidth", "offheight", "offwidth", "orthographic", "realtime"}),
    "quality": (("visual",), {"numquads", "numslices", "numstacks", "offsamples", "shadowsize"}),
    "headlight": (("visual",), {"active", "ambient", "diffuse", "specular"}),
    "map": (("visual",), {"actuatortendon", "alpha", "fogend", "fogstart", "force", "haze", "shadowclip",
                          "shadowscale", "stiffness", "stiffnessrot", "torque", "zfar", "znear"}),
    "scale": (("visual",), {"actuatorlength", "actuatorwidth", "camera", "com", "connect", "constraint",
                            "contactheight", "contactwidth", "forcewidth", "framelength", "framewidth", "frustum",
                            "jointlength", "jointwidth", "light", "selectpoint", "slidercrank"}),
    "rgba": (("visual",), {"actuator", "actuatornegative", "actuatorpositive", "bv", "bvactive", "camera", "com",
                           "connect", "constraint", "contactforce", "contactfriction", "contactgap", "contactpoint",
                           "contacttorque", "crankbroken", "fog", "force", "frustum", "haze", "inertia", "joint",
                           "light", "rangefinder", "selectpoint", "slidercrank"}),
    "default": (("mujoco", "default"), {"class"}),
    "asset": (("mujoco",), set()),
    "mesh": (("asset", "default"), {"name", "class", "file", "scale", "vertex", "normal", "texcoord", "face",
                                    "refpos", "refquat", "inertia", "smoothnormal"}),
    "texture": (("asset",), {"name", "type", "file", "builtin", "rgb1", "rgb2", "mark", "markrgb", "random",
                             "width", "height", "gridsize", "gridlayout", "hflip", "vflip", "nchannel",
                             "colorspace"}),
    "material": (("asset", "default"), {"name", "class"} | _MATERIAL_DEF),
    "layer": (("material",), {"role", "texture"}),
    "worldbody": (("mujoco",), set()),
    "body": (_BODYLIKE, {"name", "childclass"} | _POSE),
    "frame": (_BODYLIKE, {"name", "childclass"} | _POSE),
    "inertial": (("body",), {"mass", "diaginertia", "fullinertia"} | _POSE),
    "joint": (("body", "default"), {"name", "class"} | _JOINT_DEF),
    "freejoint": (("body",), {"name", "group"}),
    "geom": (_BODYLIKE + ("default",), {"name", "class"} | _GEOM_DEF),
    "site": (_BODYLIKE + ("default",), {"name", "class"} | _SITE_DEF),
    "camera": (_BODYLIKE + ("default",), {"name", "class", "target"} | _CAMERA_DEF),
    "light": (_BODYLIKE + ("default",), {"name", "class", "target"} | _LIGHT_DEF),
}
# inside <default> an element carries its default attributes only (no names, files or targets)
DEFAULT_CHILD_ATTRS = {
    "mesh": {"inertia", "scale"},
    "material": set(_MATERIAL_DEF),
    "joint": set(_JOINT_DEF),
    "geom": set(_GEOM_DEF),
    "site": set(_SITE_DEF),
    "camera": set(_CAMERA_DEF),
    "light": set(_LIGHT_DEF),
}
# elements that are named so that a rejection says why (everything not in ELEMENTS is rejected anyway)
FORBIDDEN_WHY = {
    "include": "no includes: the twin is one scene.xml",
    "size": "<size> is not allowed (memory and array sizes are the engine's)",
    "keyframe": "no keyframes: the twin is its qpos0 state", "key": "no keyframes",
    "contact": "no contact pairs or excludes", "pair": "no contact pairs", "exclude": "no contact excludes",
    "equality": "no equalities (welds, connects, joint couplings)", "weld": "no welds", "connect": "no connects",
    "tendon": "no tendons", "spatial": "no tendons", "fixed": "no tendons",
    "actuator": "no actuators", "sensor": "no sensors",
    "custom": "no custom data", "numeric": "no custom data", "text": "no custom data", "tuple": "no custom data",
    "extension": "no plugins", "plugin": "no plugins", "instance": "no plugins",
    "deformable": "no flex or skin", "flex": "no flex", "flexcomp": "no flex", "skin": "no skins",
    "composite": "no composites", "replicate": "no replicate", "attach": "no attached models",
    "model": "no attached models", "hfield": "no height fields", "lengthrange": "not allowed",
}
# attributes that are named so that a rejection says why
FORBIDDEN_ATTR_WHY = {
    "stiffness": "joint springs are not allowed", "springref": "joint springs are not allowed",
    "springdamper": "joint springs are not allowed", "damping": "joint damping is not allowed",
    "armature": "joint armature is not allowed", "frictionloss": "joint friction loss is not allowed",
    "ref": "joint ref is not allowed (qpos0 is the photographed state)",
    "margin": "joint margins are not allowed", "user": "user data is not allowed",
    "adhesion": "adhesion is not allowed", "surfacevel": "surface velocities are not allowed",
    "fluidshape": "fluid forces are not allowed", "fluidcoef": "fluid forces are not allowed",
    "density": "fluid density is not allowed", "viscosity": "fluid viscosity is not allowed",
    "wind": "wind is not allowed", "magnetic": "magnetic fields are not allowed",
    "o_margin": "contact overrides are not allowed", "o_solref": "contact overrides are not allowed",
    "o_solimp": "contact overrides are not allowed", "o_friction": "contact overrides are not allowed",
    "override": "contact overrides are not allowed", "sleep": "sleeping is not allowed",
    "sleep_tolerance": "sleeping is not allowed",
    "mocap": "mocap bodies are not allowed", "gravcomp": "gravity compensation is not allowed",
    "maxhullvert": "maxhullvert is not allowed (a mesh collides as the hull of all its vertices)",
    "content_type": "content_type is not allowed (the file extension decides the format)",
    "hfield": "height fields are not allowed", "strippath": "strippath is not allowed",
    "fusestatic": "fusestatic is not allowed", "alignfree": "alignfree is not allowed",
    "savecompiled": "the compiler may not write files", "saveinertial": "the compiler may not write files",
    "savecanonical": "the compiler may not write files", "settotalmass": "settotalmass is not allowed",
    "builtin": "builtin meshes are not allowed", "params": "builtin meshes are not allowed",
    "fileright": "cube-face texture files are not allowed (one file per texture)",
    "fileleft": "cube-face texture files are not allowed", "fileup": "cube-face texture files are not allowed",
    "filedown": "cube-face texture files are not allowed", "filefront": "cube-face texture files are not allowed",
    "fileback": "cube-face texture files are not allowed",
    "texture": "only materials and material layers may reference a texture",
    "output": "camera outputs are not allowed", "align": "freejoint align is not allowed",
    "boundinertia": "boundinertia is not allowed (item inertias come from the geometry or an explicit <inertial>)",
    "boundmass": "boundmass is not allowed (item masses come from the geometry or an explicit <inertial>)",
    "balanceinertia": "balanceinertia is not allowed (an explicit inertia must be physical as given)",
}
JOINT_TYPES = {"slide", "free"}
GEOM_TYPES = {"plane", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh"}
TEXTURE_TYPES = {"2d", "cube", "skybox"}
TEXTURE_BUILTINS = {"none", "gradient", "checker", "flat"}
MESH_INERTIA = {"convex", "exact", "legacy", "shell"}
MESH_EXT = (".stl", ".obj", ".msh")
TEXTURE_EXT = (".png",)
PATH_RX = re.compile(r"^[A-Za-z0-9_.+\- ()/]+$")
PNG_SIG = b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------------------------------------------
# result

class TaskError(RuntimeError):
    """The task's own inputs are broken (e.g. the product vocabulary): never a verdict on the submission."""


class GateResult:
    """ok: the twin passes G0; report: everything measured (JSON-ready); model: the compiled grader copy (textures
    removed) or None; spec: the MjSpec it came from; objects: the parsed objects.json."""

    def __init__(self, report, model=None, spec=None, objects=None):
        self.report = report
        self.model = model
        self.spec = spec
        self.objects = objects

    @property
    def ok(self):
        return bool(self.report.get("pass"))

    def item_bodies(self, status=None):
        out = []
        for e in self.report.get("items", []):
            if status is None or e.get("status") == status:
                out.append(e["body"])
        return out

    def physics_model(self):
        """The model every simulation runs: the grader copy with the excluded items deleted (see physics_model)."""
        return physics_model(self)


class _Collector:
    def __init__(self):
        self.reasons = []
        self.warnings = []

    def fail(self, msg):
        if len(self.reasons) < 200:
            self.reasons.append(str(msg))

    def warn(self, msg):
        if len(self.warnings) < 200:
            self.warnings.append(str(msg))


# ---------------------------------------------------------------------------------------------------------------
# 1. files

def tree_stats(root):
    """(entries, bytes, irregular entries) of the folder, without following symbolic links."""
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
            return None, "a path component is not a directory"
        if k == len(parts) - 1 and not _stat.S_ISREG(st.st_mode):
            return None, "is not a regular file"
    real = os.path.realpath(cur)
    rroot = os.path.realpath(root)
    if not (real == rroot or real.startswith(rroot + os.sep)):
        return None, "resolves outside the submission"
    return cur, None


def _path_problem(v, is_dir=False):
    """Why a file or directory attribute is not a plain relative path inside the submission (None if it is)."""
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
        return "contains ':' (no drive letters or URL schemes)"
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


def _os_why(exc):
    """An OSError as a short reason without host paths."""
    return getattr(exc, "strerror", None) or type(exc).__name__


def _read_bytes(path, limit, col, what):
    """At most `limit` bytes of a regular file, or None after a failure with a reason (unreadable, vanished)."""
    try:
        with open(path, "rb") as fh:
            return fh.read(limit)
    except OSError as exc:
        col.fail("%s cannot be read: %s" % (what, _os_why(exc)))
        return None


def _join_rel(*parts):
    """Join relative path parts with '/' and normalise '.' components (no '..' can occur after the checks)."""
    out = []
    for p in parts:
        if not p:
            continue
        for q in p.split("/"):
            if q in ("", "."):
                continue
            out.append(q)
    return "/".join(out)


# ---------------------------------------------------------------------------------------------------------------
# 2. scene.xml, statically

def _floats(text):
    try:
        return [float(v) for v in str(text).replace(",", " ").split()]
    except ValueError:
        return None


def static_scene(raw, col, rules):
    """Parse and check scene.xml bytes. Returns the ElementTree root or None."""
    if len(raw) > rules["max_scene_bytes"]:
        col.fail("scene.xml is %d bytes (cap %d)" % (len(raw), rules["max_scene_bytes"]))
        return None
    if b"\x00" in raw:
        col.fail("scene.xml contains NUL bytes")
        return None
    try:
        text = raw.decode("utf-8-sig")          # a leading byte-order mark is dropped
    except UnicodeDecodeError:
        col.fail("scene.xml is not valid UTF-8")
        return None
    m = re.match(r"\s*<\?xml[^>]*encoding\s*=\s*['\"]([^'\"]+)['\"]", text)
    if m and m.group(1).strip().lower() not in ("utf-8", "utf8", "us-ascii", "ascii"):
        col.fail("scene.xml declares encoding %r (UTF-8 required)" % m.group(1))
        return None
    up = text.upper()
    if "<!DOCTYPE" in up or "<!ENTITY" in up:
        col.fail("DOCTYPE / ENTITY declarations are not allowed")
        return None
    if "XMLNS" in up:
        col.fail("XML namespaces (xmlns) are not allowed")
        return None
    body = text.lstrip()
    if body.startswith("<?xml"):
        body = body[body.find("?>") + 2:]
    if "<?" in body:
        col.fail("XML processing instructions are not allowed")
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        col.fail("scene.xml does not parse as XML: %s" % exc)
        return None
    if root.tag != "mujoco":
        col.fail("the root element is <%s>, not <mujoco>" % root.tag)
        return None
    n_el = sum(1 for _ in root.iter())
    if n_el > rules["max_xml_elements"]:
        col.fail("scene.xml has %d elements (cap %d)" % (n_el, rules["max_xml_elements"]))
        return None
    _walk(root, col, rules)
    if col.reasons:
        return None
    return root


def _walk(root, col, rules):
    """Check every element against the whitelist (iteratively: deep nesting cannot exhaust the Python stack)."""
    stack = [(root, None, False, 0)]
    while stack:
        el, parent, in_default, depth = stack.pop()
        tag = el.tag
        if not isinstance(tag, str):
            col.fail("XML comments or processing instructions inside elements are not allowed")
            continue
        if depth > rules["max_xml_depth"]:
            col.fail("scene.xml nests elements more than %d deep" % rules["max_xml_depth"])
            return
        if tag not in ELEMENTS:
            why = FORBIDDEN_WHY.get(tag)
            col.fail("<%s> is not allowed%s" % (tag, (": " + why) if why else " (not in the element whitelist)"))
            continue
        parents, attrs = ELEMENTS[tag]
        if parent not in parents:
            col.fail("<%s> is not allowed inside <%s>" % (tag, parent))
            continue
        if in_default and tag != "default":
            attrs = DEFAULT_CHILD_ATTRS.get(tag, set())
        if (el.text or "").strip() or (el.tail or "").strip():
            col.fail("<%s> carries text content (attributes only)" % tag)
        for k, v in el.attrib.items():
            if k not in attrs:
                why = FORBIDDEN_ATTR_WHY.get(k)
                if k == "texture" and tag in ("material", "layer"):
                    why = None
                col.fail("<%s %s=...> is not allowed%s" % (tag, k, (": " + why) if why else
                                                           " (not in the attribute allowlist of <%s>%s)" %
                                                           (tag, " inside <default>" if in_default else "")))
                continue
            _check_value(tag, k, v, col, rules)
        if len(col.reasons) >= 200:
            return
        for ch in reversed(list(el)):
            stack.append((ch, tag, in_default or tag == "default", depth + 1))


def _check_value(tag, k, v, col, rules):
    v = str(v)
    if tag == "joint" and k == "type" and v not in JOINT_TYPES:
        col.fail("<joint type=%r> is not allowed (slide for the drawer, free for items)" % v)
    elif tag == "geom" and k == "type" and v not in GEOM_TYPES:
        col.fail("<geom type=%r> is not allowed (%s)" % (v, ", ".join(sorted(GEOM_TYPES))))
    elif tag == "texture" and k == "type" and v not in TEXTURE_TYPES:
        col.fail("<texture type=%r> is not allowed" % v)
    elif tag == "texture" and k == "builtin" and v not in TEXTURE_BUILTINS:
        col.fail("<texture builtin=%r> is not allowed" % v)
    elif tag == "texture" and k in ("width", "height"):
        f = _floats(v)
        if not f or not (0 <= f[0] <= rules["max_png_side_px"]):
            col.fail("<texture %s=%s> is outside [0, %d]" % (k, v, rules["max_png_side_px"]))
    elif tag == "mesh" and k == "inertia" and v not in MESH_INERTIA:
        col.fail("<mesh inertia=%r> is not allowed" % v)
    elif tag == "global" and k in ("offwidth", "offheight"):
        f = _floats(v)
        if not f or not (1 <= f[0] <= 8192):
            col.fail("<global %s=%s> is outside [1, 8192]" % (k, v))
    elif tag == "mesh" and k in ("vertex", "normal", "texcoord", "face"):
        n = len(v.split())
        if n > 3 * rules["max_collision_mesh_vertices"] * 20:
            col.fail("<mesh %s> holds %d numbers (too many for an inline mesh)" % (k, n))
    elif tag == "compiler" and k in ("meshdir", "texturedir", "assetdir"):
        why = _path_problem(v, is_dir=True)
        if why:
            col.fail("<compiler %s=%r> %s" % (k, v, why))


def count_static(root):
    n = {"body": 0, "geom": 0, "mesh": 0, "texture": 0, "compiler": 0, "camera": 0}
    for el in root.iter():
        if el.tag in n:
            n[el.tag] += 1
    return n


# ---------------------------------------------------------------------------------------------------------------
# 3. referenced files

def _stl_info(data):
    if len(data) < 84:
        return None, "is shorter than a binary STL header"
    n = struct.unpack_from("<I", data, 80)[0]
    if len(data) != 84 + 50 * n:
        return None, "is not a consistent binary STL (header says %d triangles, size %d bytes)" % (n, len(data))
    return {"faces": n, "vertices": 3 * n}, None


def _msh_info(data):
    if len(data) < 16:
        return None, "is shorter than an MSH header"
    nv, nn, nt, nf = struct.unpack_from("<4i", data, 0)
    if min(nv, nn, nt, nf) < 0:
        return None, "has negative counts in its MSH header"
    need = 16 + 12 * nv + 12 * nn + 8 * nt + 12 * nf
    if len(data) != need:
        return None, "is not a consistent MSH file (header says %d vertices, %d faces; size %d, expected %d)" % (
            nv, nf, len(data), need)
    return {"faces": nf, "vertices": nv}, None


def _obj_info(data):
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = data.decode("latin-1")
        except UnicodeDecodeError:  # pragma: no cover - latin-1 decodes everything
            return None, "is not a text OBJ file"
    nv = nf = 0
    for line in io.StringIO(text):
        tok = line.split()
        if not tok:
            continue
        key = tok[0]
        if key == "v":
            nv += 1
        elif key == "f":
            nf += max(1, len(tok) - 3)
        elif key == "mtllib":
            for nm in tok[1:]:
                if "/" in nm or "\\" in nm or ":" in nm or nm in (".", "..") or nm.startswith("~"):
                    return None, "has mtllib %r (a material library must be a plain file name)" % nm
    return {"faces": nf, "vertices": nv}, None


def _png_info(head):
    if len(head) < 33 or head[:8] != PNG_SIG or head[12:16] != b"IHDR":
        return None, "is not a PNG file"
    w, h = struct.unpack(">II", head[16:24])
    return {"width": int(w), "height": int(h)}, None


def check_files(root, tree, col, rules):
    """Check every file the scene references. Returns {canonical relative path: bytes} for the meshes, a list of
    (element, canonical path) for the textures and {canonical path: header counts} for the meshes."""
    comp = tree.find("compiler")
    if len(tree.findall("compiler")) > 1:
        col.fail("only one <compiler> element is allowed")
        return {}, [], {}
    a = comp.attrib if comp is not None else {}
    meshdir = a.get("meshdir", a.get("assetdir", ""))
    texdir = a.get("texturedir", a.get("assetdir", ""))
    meshes, textures, mesh_info = {}, [], {}
    totals = {"faces": 0, "vertices": 0, "files": 0}
    for el in tree.iter():
        if el.tag not in ("mesh", "texture") or "file" not in el.attrib:
            continue
        f = el.attrib["file"]
        why = _path_problem(f)
        if why:
            col.fail("<%s file=%r> %s" % (el.tag, f, why))
            continue
        rel = _join_rel(meshdir if el.tag == "mesh" else texdir, f)
        ext = os.path.splitext(rel)[1].lower()
        if el.tag == "mesh" and ext not in MESH_EXT:
            col.fail("<mesh file=%r>: mesh files must be .stl, .obj or .msh" % f)
            continue
        if el.tag == "texture" and ext not in TEXTURE_EXT:
            col.fail("<texture file=%r>: texture files must be .png" % f)
            continue
        path, why = _regular_file(root, rel)
        if why:
            col.fail("<%s file=%r> (%s) %s" % (el.tag, f, rel, why))
            continue
        try:
            size = os.lstat(path).st_size
        except OSError as exc:
            col.fail("%s %s cannot be read: %s" % (el.tag, rel, _os_why(exc)))
            continue
        if el.tag == "texture":
            if size > rules["max_png_file_bytes"]:
                col.fail("texture %s is %d bytes (cap %d)" % (rel, size, rules["max_png_file_bytes"]))
                continue
            head = _read_bytes(path, 33, col, "texture %s" % rel)
            if head is None:
                continue
            info, why = _png_info(head)
            if why:
                col.fail("texture %s %s" % (rel, why))
                continue
            side = rules["max_png_side_px"]
            if not (1 <= info["width"] <= side and 1 <= info["height"] <= side):
                col.fail("texture %s is %d x %d px (at most %d a side)" % (rel, info["width"], info["height"], side))
                continue
            textures.append((el, rel))
            totals["files"] += 1
            continue
        if rel in meshes:
            continue
        if size > rules["max_mesh_file_bytes"]:
            col.fail("mesh %s is %d bytes (cap %d)" % (rel, size, rules["max_mesh_file_bytes"]))
            continue
        data = _read_bytes(path, rules["max_mesh_file_bytes"] + 1, col, "mesh %s" % rel)
        if data is None:
            continue
        info, why = {".stl": _stl_info, ".msh": _msh_info, ".obj": _obj_info}[ext](data)
        if why:
            col.fail("mesh %s %s" % (rel, why))
            continue
        totals["faces"] += info["faces"]
        totals["vertices"] += info["vertices"]
        totals["files"] += 1
        meshes[rel] = data
        mesh_info[rel] = info
    if totals["faces"] > rules["max_mesh_faces_total"]:
        col.fail("the mesh files hold %d faces in total (cap %d)" % (totals["faces"], rules["max_mesh_faces_total"]))
    if totals["vertices"] > rules["max_mesh_vertices_total"]:
        col.fail("the mesh files hold %d vertices in total (cap %d)" % (totals["vertices"],
                                                                         rules["max_mesh_vertices_total"]))
    if totals["files"] > rules["max_referenced_files"]:
        col.fail("scene.xml references %d files (cap %d)" % (totals["files"], rules["max_referenced_files"]))
    return meshes, textures, mesh_info


# ---------------------------------------------------------------------------------------------------------------
# 4. objects.json

def _no_constants(x):
    raise ValueError("NaN / Infinity are not JSON")


def check_objects(raw, col, rules, vocabulary):
    """Parse and check objects.json. Returns (objects dict or None, per-item label flags)."""
    if len(raw) > rules["max_objects_bytes"]:
        col.fail("objects.json is %d bytes (cap %d)" % (len(raw), rules["max_objects_bytes"]))
        return None
    try:
        obj = json.loads(raw.decode("utf-8"), parse_constant=_no_constants)
    except MemoryError:
        raise
    except Exception as exc:  # noqa: BLE001 - UnicodeDecodeError, ValueError, RecursionError (deep nesting)
        col.fail("objects.json does not parse as JSON: %s" % str(exc)[:160])
        return None
    if not isinstance(obj, dict):
        col.fail('objects.json must be an object {"format", "items", "drawer"}')
        return None
    for k in obj:
        if k not in ("format", "items", "drawer"):
            col.warn("objects.json: key %r is ignored" % k)
    if obj.get("format") != FORMAT_OBJECTS:
        col.fail('objects.json "format" must be %r' % FORMAT_OBJECTS)
    items = obj.get("items")
    lo, hi = rules["items"]
    if not isinstance(items, list):
        col.fail('objects.json "items" must be a list of {"body", "label"}')
        return None
    if not lo <= len(items) <= hi:
        col.fail("objects.json lists %d items (%d to %d allowed)" % (len(items), lo, hi))
    seen = set()
    for k, it in enumerate(items):
        if not isinstance(it, dict) or not isinstance(it.get("body"), str) or not isinstance(it.get("label"), str):
            col.fail('objects.json items[%d] must be {"body": "<body name>", "label": "<product name>"}' % k)
            continue
        for kk in it:
            if kk not in ("body", "label"):
                col.warn("objects.json items[%d]: key %r is ignored" % (k, kk))
        if not it["body"]:
            col.fail("objects.json items[%d] has an empty body name" % k)
        if it["body"] in seen:
            col.fail("objects.json lists body %r twice" % it["body"])
        seen.add(it["body"])
    dr = obj.get("drawer")
    if not isinstance(dr, str) or not dr:
        col.fail('objects.json "drawer" must name the drawer body')
    elif dr in seen:
        col.fail("the drawer body %r is also listed as an item" % dr)
    return obj


_VOCAB_CACHE = {}


def load_vocabulary(vocabulary=None):
    """The product names: a list, a path to product_names.json, or None for the file next to this module. A file is
    read and validated once per process; a missing or malformed vocabulary raises TaskError (the task's inputs are
    broken, not the submission)."""
    if vocabulary is None:
        vocabulary = os.path.join(HERE, "product_names.json")
    if isinstance(vocabulary, (list, tuple, set)):
        names = [v for v in vocabulary]
        if not names or not all(isinstance(n, str) for n in names):
            raise TaskError("the product vocabulary must be a non-empty list of names")
        return list(names)
    key = os.path.abspath(str(vocabulary))
    if key in _VOCAB_CACHE:
        return list(_VOCAB_CACHE[key])
    try:
        with open(key, "rb") as f:
            d = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise TaskError("the product vocabulary %s cannot be read: %s" % (os.path.basename(key), str(exc)[:120]))
    names = d.get("names") if isinstance(d, dict) else d
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise TaskError("the product vocabulary %s must be {\"names\": [...]}" % os.path.basename(key))
    _VOCAB_CACHE[key] = tuple(names)
    return list(names)


# ---------------------------------------------------------------------------------------------------------------
# 5. the grader copy and the compile

def grader_copy(tree, meshes, rules):
    """XML text of the grader copy: <texture> elements and every texture reference removed, mesh paths written out
    in full (relative to the submission root) and the compiler's directory attributes dropped."""
    import copy
    root = copy.deepcopy(tree)
    comp = root.find("compiler")
    meshdir = ""
    if comp is not None:
        a = comp.attrib
        meshdir = a.get("meshdir", a.get("assetdir", ""))
        for k in ("meshdir", "texturedir", "assetdir"):
            a.pop(k, None)
    for parent in root.iter():
        for ch in list(parent):
            if ch.tag == "texture" or (ch.tag == "layer" and parent.tag == "material"):
                parent.remove(ch)
    for el in root.iter():
        if el.tag == "material":
            el.attrib.pop("texture", None)
        if el.tag == "mesh" and "file" in el.attrib:
            el.attrib["file"] = _join_rel(meshdir, el.attrib["file"])
    return ET.tostring(root, encoding="unicode")


_WARN_PREFIX = re.compile(r"^.*?\.py:\d+: \w*Warning: ", re.M)


def _clean_message(msg):
    """A warning text without the 'file.py:line: UserWarning: ' prefix Python adds when it prints a warning."""
    return _WARN_PREFIX.sub("", str(msg)).strip()


class _CaptureStderr:
    """Collect MuJoCo's warnings deterministically: its warning callback, every Python warning (recorded with the
    filter forced to "always", so neither -W ignore nor the once-per-location registry can hide one) and anything
    written to file descriptor 2."""

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
        recorded = [_clean_message(w.message) for w in self._wrec]
        self._wctx.__exit__(None, None, None)
        for msg in recorded:
            if msg and msg not in self.messages:
                self.messages.append(msg[:600])
        text = _clean_message(text)
        if text and not any(text == m or text in m or m in text for m in self.messages if m):
            self.messages.append(text[:600])
        return False


def parse_copy(xml_text, meshes):
    """(spec, warnings, error): the grader copy parsed (defaults applied; no mesh file is decoded yet)."""
    import mujoco
    spec = error = None
    with _CaptureStderr() as cap:
        try:
            spec = mujoco.MjSpec.from_string(xml_text, assets=dict(meshes))
        except MemoryError:
            raise
        except Exception as exc:  # noqa: BLE001
            error = _clean_message(exc)[:600]
    return spec, [m for m in cap.messages if m], error


def precompile_caps(spec, mesh_info, col, rules):
    """Collision meshes are checked against the vertex cap from their file headers before MuJoCo computes hulls."""
    import mujoco
    cap = rules["max_collision_mesh_vertices"]
    nverts = {}
    for ms in spec.meshes:
        f = getattr(ms, "file", "") or ""
        nverts[ms.name] = (mesh_info.get(f, {}).get("vertices", 0) if f else len(ms.uservert) // 3)
    for gs in spec.geoms:
        if int(gs.type) != int(mujoco.mjtGeom.mjGEOM_MESH) or not gs.meshname:
            continue
        if int(gs.contype) == 0 and int(gs.conaffinity) == 0:
            continue
        nv = nverts.get(gs.meshname, 0)
        # file vertex counts repeat shared vertices (an STL stores 3 per triangle), so the exact cap on the unique
        # vertices is applied after the compile; this bound only keeps MuJoCo's hull computation small
        if nv > 3 * cap:
            col.fail("collision mesh %s has %d vertices in its file (more than 3 x the cap of %d unique vertices)" % (
                gs.meshname, nv, cap))


def compile_copy(spec):
    """(model, warnings, error) of the parsed grader copy, compiled from memory."""
    model = error = None
    with _CaptureStderr() as cap:
        try:
            model = spec.compile()
        except MemoryError:
            raise
        except Exception as exc:  # noqa: BLE001 - MuJoCo raises ValueError for compile errors
            error = _clean_message(exc)[:600]
            model = None
    return model, [m for m in cap.messages if m], error


# ---------------------------------------------------------------------------------------------------------------
# geometry: convex pieces of collision geoms, their union (shared with the grader's scoring code)

def fibonacci_dirs(n):
    k = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * k / n)
    th = math.pi * (1.0 + 5 ** 0.5) * k
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def _quad_interval(A, B, C):
    """Elementwise interval of z with A z^2 + B z + C <= 0 (A >= 0): (z0, z1), NaN where empty, +-inf where the
    condition is unbounded."""
    A, B, C = (np.asarray(x, float) for x in np.broadcast_arrays(A, B, C))
    z0 = np.full(A.shape, np.nan)
    z1 = np.full(A.shape, np.nan)
    quad = A > 1e-14
    disc = B * B - 4.0 * A * C
    ok = quad & (disc >= 0)
    if np.any(ok):
        sq = np.sqrt(np.where(ok, disc, 0.0))
        q = -0.5 * (B + np.copysign(sq, B))
        qa = np.where(ok, q, 1.0)
        r1 = np.where(ok, qa / np.where(ok, A, 1.0), np.nan)
        safe = np.abs(qa) > 0
        r2 = np.where(ok & safe, C / np.where(safe, qa, 1.0), r1)
        z0 = np.where(ok, np.minimum(r1, r2), z0)
        z1 = np.where(ok, np.maximum(r1, r2), z1)
    lin = ~quad
    if np.any(lin):
        bpos = lin & (B > 0)
        bneg = lin & (B < 0)
        bzero = lin & (B == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            root = -C / np.where(B != 0, B, 1.0)
        z0 = np.where(bpos, -np.inf, z0)
        z1 = np.where(bpos, root, z1)
        z0 = np.where(bneg, root, z0)
        z1 = np.where(bneg, np.inf, z1)
        allz = bzero & (C <= 0)
        z0 = np.where(allz, -np.inf, z0)
        z1 = np.where(allz, np.inf, z1)
    return z0, z1


def _intersect(a0, a1, b0, b1):
    z0 = np.fmax(a0, b0)
    z1 = np.fmin(a1, b1)
    empty = np.isnan(a0) | np.isnan(b0) | (z0 > z1)
    return np.where(empty, np.nan, z0), np.where(empty, np.nan, z1)


class Piece:
    """One convex collision piece in some frame: a mesh hull (vertices V, faces F, planes A p <= b) or a primitive
    ('box' p = half sizes, 'sphere' p = [r], 'capsule' / 'cylinder' p = [r, half length], 'ellipsoid' p = radii)
    with centre c and axes R (columns)."""

    def __init__(self, kind, c, R, p=None, V=None, F=None, A=None, b=None, geom=-1):
        self.kind = kind
        self.c = np.asarray(c, float).reshape(3)
        self.R = np.asarray(R, float).reshape(3, 3)
        self.p = None if p is None else np.asarray(p, float).ravel()
        self.V = V
        self.F = F
        self.A = A
        self.b = b
        self.geom = int(geom)
        if kind == "box" and V is None:
            s = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * self.p
            self.V = s @ self.R.T + self.c
            A = np.vstack([self.R.T, -self.R.T])
            self.A = A
            self.b = A @ self.c + np.r_[self.p, self.p]

    # -- transforms
    def transformed(self, M, t, s=1.0):
        """The piece under p -> s (M p + t), M a rotation."""
        M = np.asarray(M, float)
        t = np.asarray(t, float)
        c = s * (M @ self.c + t)
        R = M @ self.R
        p = None if self.p is None else s * self.p
        V = None if self.V is None else s * (self.V @ M.T + t)
        A = b = None
        if self.A is not None:
            A = self.A @ M.T
            b = s * (self.b + A @ t)
        return Piece(self.kind, c, R, p=p, V=V, F=self.F, A=A, b=b, geom=self.geom)

    def scaled(self, s):
        return self.transformed(np.eye(3), np.zeros(3), s)

    # -- measures
    def support(self, U):
        """max over the piece of u . x for unit directions U (k, 3) (in row blocks: memory stays bounded)."""
        U = np.atleast_2d(np.asarray(U, float))
        if self.kind in ("hull", "box"):
            V = self.V
            blk = max(1, _BLOCK_ENTRIES // max(1, len(V)))
            if len(U) <= blk:
                return np.max(U @ V.T, axis=1)
            out = np.empty(len(U))
            for k in range(0, len(U), blk):
                out[k:k + blk] = np.max(U[k:k + blk] @ V.T, axis=1)
            return out
        base = U @ self.c
        if self.kind == "sphere":
            return base + self.p[0]
        a = self.R[:, 2]
        ua = U @ a
        if self.kind == "capsule":
            return base + self.p[0] + self.p[1] * np.abs(ua)
        if self.kind == "cylinder":
            return base + self.p[0] * np.sqrt(np.maximum(0.0, 1.0 - ua * ua)) + self.p[1] * np.abs(ua)
        if self.kind == "ellipsoid":
            return base + np.linalg.norm((U @ self.R) * self.p, axis=1)
        raise ValueError(self.kind)

    def bbox(self):
        E = np.vstack([np.eye(3), -np.eye(3)])
        h = self.support(E)
        return -h[3:], h[:3]

    def max_dist(self, C):
        """Largest distance from the point C to the piece (an upper bound for an off-centre ellipsoid)."""
        C = np.asarray(C, float).reshape(3)
        k, p = self.kind, self.p
        if k in ("hull", "box"):
            return float(np.sqrt(np.max(np.sum((self.V - C) ** 2, axis=1))))
        if k == "sphere":
            return float(np.linalg.norm(self.c - C) + p[0])
        if k == "ellipsoid":
            return float(np.linalg.norm(self.c - C) + np.max(p))
        a = self.R[:, 2]
        best = 0.0
        for sgn in (-1.0, 1.0):
            w = self.c + sgn * p[1] * a - C
            if k == "capsule":
                best = max(best, float(np.linalg.norm(w)) + p[0])
            else:                                            # cylinder: the farthest point of an end rim
                wp = float(w @ a)
                wr = float(np.linalg.norm(w - wp * a))
                best = max(best, math.sqrt(wp * wp + (wr + p[0]) ** 2))
        return best

    def volume(self):
        k, p = self.kind, self.p
        if k == "box":
            return float(8.0 * p[0] * p[1] * p[2])
        if k == "sphere":
            return float(4.0 / 3.0 * math.pi * p[0] ** 3)
        if k == "ellipsoid":
            return float(4.0 / 3.0 * math.pi * p[0] * p[1] * p[2])
        if k == "cylinder":
            return float(math.pi * p[0] ** 2 * 2.0 * p[1])
        if k == "capsule":
            return float(math.pi * p[0] ** 2 * 2.0 * p[1] + 4.0 / 3.0 * math.pi * p[0] ** 3)
        return _hull_volume_centroid(self.V, self.F)[0]

    def centroid(self):
        if self.kind == "hull":
            return _hull_volume_centroid(self.V, self.F)[1]
        return self.c.copy()

    def area(self):
        k, p = self.kind, self.p
        if k == "box":
            return float(8.0 * (p[0] * p[1] + p[1] * p[2] + p[0] * p[2]))
        if k == "sphere":
            return float(4.0 * math.pi * p[0] ** 2)
        if k == "cylinder":
            return float(2.0 * math.pi * p[0] * 2.0 * p[1] + 2.0 * math.pi * p[0] ** 2)
        if k == "capsule":
            return float(2.0 * math.pi * p[0] * 2.0 * p[1] + 4.0 * math.pi * p[0] ** 2)
        if k == "ellipsoid":
            a, b2, c2 = p
            q = 1.6075
            return float(4.0 * math.pi * (((a * b2) ** q + (a * c2) ** q + (b2 * c2) ** q) / 3.0) ** (1.0 / q))
        T = self.V[self.F]
        return float(0.5 * np.sum(np.linalg.norm(np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0]), axis=1)))

    def inside(self, P, tol=0.0):
        """Points (N, 3) inside the piece (within tol)."""
        P = np.atleast_2d(np.asarray(P, float))
        if self.kind in ("hull", "box"):
            out = np.ones(len(P), bool)
            for k0 in range(0, len(self.b), 256):
                out &= np.all(P @ self.A[k0:k0 + 256].T <= self.b[k0:k0 + 256] + tol, axis=1)
            return out
        q = (P - self.c) @ self.R
        r = self.p[0]
        if self.kind == "sphere":
            return np.einsum("ij,ij->i", q, q) <= (r + tol) ** 2
        if self.kind == "ellipsoid":
            return np.sum((q / (self.p + tol)) ** 2, axis=1) <= 1.0
        hl = self.p[1]
        if self.kind == "cylinder":
            return (np.abs(q[:, 2]) <= hl + tol) & (q[:, 0] ** 2 + q[:, 1] ** 2 <= (r + tol) ** 2)
        if self.kind == "capsule":
            z = np.clip(q[:, 2], -hl, hl)
            d2 = q[:, 0] ** 2 + q[:, 1] ** 2 + (q[:, 2] - z) ** 2
            return d2 <= (r + tol) ** 2
        raise ValueError(self.kind)

    def index_box(self, h):
        """Grid columns (i0, i1, j0, j1) (half-open, one column of padding) whose centres ((i + 0.5) h, (j + 0.5) h)
        can meet the piece."""
        lo, hi = self.bbox()
        return (int(math.floor(lo[0] / h - 0.5)), int(math.ceil(hi[0] / h - 0.5)) + 1,
                int(math.floor(lo[1] / h - 0.5)), int(math.ceil(hi[1] / h - 0.5)) + 1)

    def grid_interval(self, h, i0, i1, j0, j1):
        """z intervals inside the piece along the grid columns i in [i0, i1), j in [j0, j1) (centres at
        ((i + 0.5) h, (j + 0.5) h)): (z0, z1) of shape (i1 - i0, j1 - j0), NaN where empty. A hull with many faces
        is rasterised face by face (cost ~ columns + faces) instead of testing every plane on every column."""
        ni, nj = i1 - i0, j1 - j0
        if ni <= 0 or nj <= 0:
            return np.zeros((max(ni, 0), max(nj, 0))), np.zeros((max(ni, 0), max(nj, 0)))
        if self.kind == "hull" and self.F is not None and len(self.F) > _RASTER_MIN_FACES:
            rd = self._raster_data()
            ncand = _raster_candidates(rd, h, i0, i1, j0, j1)
            if 8 * ncand + len(self.F) < ni * nj * len(self.b):
                z0, z1 = _raster_hull(rd, h, i0, i1, j0, j1)
                return z0.reshape(ni, nj), z1.reshape(ni, nj)
        I, J = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
        z0, z1 = self.z_interval(((I.ravel() + 0.5) * h), ((J.ravel() + 0.5) * h))
        return z0.reshape(ni, nj), z1.reshape(ni, nj)

    def _raster_data(self):
        if getattr(self, "_rd", None) is None:
            self._rd = _raster_prepare(self.V, self.F)
        return self._rd

    def z_interval(self, X, Y):
        """For vertical lines through (X, Y) (N,), the z interval inside the piece: (z0, z1), NaN where empty."""
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        if self.kind in ("hull", "box"):
            return _planes_interval(self.A, self.b, X, Y)
        dx, dy = X - self.c[0], Y - self.c[1]
        if self.kind == "sphere":
            w2 = self.p[0] ** 2 - dx * dx - dy * dy
            ok = w2 >= 0
            w = np.sqrt(np.where(ok, w2, 0.0))
            return np.where(ok, self.c[2] - w, np.nan), np.where(ok, self.c[2] + w, np.nan)
        if self.kind == "ellipsoid":
            Q = self.R @ np.diag(1.0 / self.p ** 2) @ self.R.T
            A = Q[2, 2] * np.ones_like(dx)
            B = 2.0 * (Q[0, 2] * dx + Q[1, 2] * dy)
            C = Q[0, 0] * dx * dx + 2.0 * Q[0, 1] * dx * dy + Q[1, 1] * dy * dy - 1.0
            z0, z1 = _quad_interval(A, B, C)
            return z0 + self.c[2], z1 + self.c[2]
        a = self.R[:, 2]
        r, hl = self.p
        t0 = a[0] * dx + a[1] * dy
        A = (1.0 - a[2] ** 2) * np.ones_like(dx)
        B = -2.0 * t0 * a[2]
        C = dx * dx + dy * dy - t0 * t0 - r * r
        rz0, rz1 = _quad_interval(A, B, C)
        if abs(a[2]) > 1e-14:
            s0 = (-hl - t0) / a[2]
            s1 = (hl - t0) / a[2]
            sz0, sz1 = np.minimum(s0, s1), np.maximum(s0, s1)
        else:
            inside = np.abs(t0) <= hl
            sz0 = np.where(inside, -np.inf, np.nan)
            sz1 = np.where(inside, np.inf, np.nan)
        z0, z1 = _intersect(rz0, rz1, sz0, sz1)
        z0, z1 = z0 + self.c[2], z1 + self.c[2]          # offsets from the centre -> absolute z
        if self.kind == "capsule":
            for sgn in (-1.0, 1.0):
                e = self.c + sgn * hl * a
                ex, ey = X - e[0], Y - e[1]
                w2 = r * r - ex * ex - ey * ey
                ok = w2 >= 0
                w = np.sqrt(np.where(ok, w2, 0.0))
                e0 = np.where(ok, e[2] - w, np.nan)
                e1 = np.where(ok, e[2] + w, np.nan)
                z0 = np.fmin(z0, e0)
                z1 = np.fmax(z1, e1)
            return z0, z1
        if self.kind == "cylinder":
            return z0, z1
        raise ValueError(self.kind)

    def surface_samples(self, n, rng):
        """About n points on the surface with unit outward normals: (P (m, 3), N (m, 3))."""
        n = max(1, int(n))
        k = self.kind
        if k in ("hull", "box"):
            if k == "box":
                F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                              [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
            else:
                F = self.F
            T = self.V[F]
            cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
            ar = np.linalg.norm(cr, axis=1)
            keep = ar > 0
            T, cr, ar = T[keep], cr[keep], ar[keep]
            nrm = cr / ar[:, None]
            cen = self.V.mean(axis=0)
            flip = np.einsum("ij,ij->i", nrm, T[:, 0] - cen) < 0
            nrm[flip] *= -1.0
            idx = rng.choice(len(T), size=n, p=ar / ar.sum())
            u = rng.random(n)
            v = rng.random(n)
            su = np.sqrt(u)
            w0, w1, w2 = 1.0 - su, su * (1.0 - v), su * v
            P = w0[:, None] * T[idx, 0] + w1[:, None] * T[idx, 1] + w2[:, None] * T[idx, 2]
            return P, nrm[idx]
        if k in ("sphere", "ellipsoid"):
            u = rng.normal(size=(n, 3))
            u /= np.linalg.norm(u, axis=1)[:, None]
            pr = self.p if k == "ellipsoid" else np.repeat(self.p[0], 3)
            P = (u * pr) @ self.R.T + self.c
            N = (u / pr) @ self.R.T
            return P, N / np.linalg.norm(N, axis=1)[:, None]
        r, hl = self.p
        side = 2.0 * math.pi * r * 2.0 * hl
        ends = 2.0 * math.pi * r * r if k == "cylinder" else 4.0 * math.pi * r * r
        ns = int(round(n * side / (side + ends)))
        th = rng.random(ns) * 2.0 * math.pi
        t = (rng.random(ns) * 2.0 - 1.0) * hl
        loc = np.c_[r * np.cos(th), r * np.sin(th), t]
        nloc = np.c_[np.cos(th), np.sin(th), np.zeros(ns)]
        ne = n - ns
        if k == "cylinder":
            rr = r * np.sqrt(rng.random(ne))
            ph = rng.random(ne) * 2.0 * math.pi
            sg = np.where(rng.random(ne) < 0.5, -1.0, 1.0)
            le = np.c_[rr * np.cos(ph), rr * np.sin(ph), sg * hl]
            nle = np.c_[np.zeros(ne), np.zeros(ne), sg]
        else:
            u = rng.normal(size=(ne, 3))
            u /= np.linalg.norm(u, axis=1)[:, None]
            le = r * u + np.c_[np.zeros(ne), np.zeros(ne), np.sign(u[:, 2]) * hl]
            nle = u
        L = np.vstack([loc, le])
        NL = np.vstack([nloc, nle])
        return L @ self.R.T + self.c, NL @ self.R.T


def _planes_interval(A, b, X, Y, chunk=2_000_000):
    n = len(X)
    z0 = np.full(n, np.nan)
    z1 = np.full(n, np.nan)
    az = A[:, 2]
    up = az > 1e-12
    lo = az < -1e-12
    vt = ~(up | lo)
    # each plane a . p <= b bounds z along a column: z <= (b - ax x - ay y) / az (az > 0) or z >= ... (az < 0)
    Au, bu = A[up] / az[up, None], b[up] / az[up]
    Al, bl = A[lo] / az[lo, None], b[lo] / az[lo]
    Av, bv = A[vt], b[vt]
    step = max(1, chunk // max(1, len(b)))
    for s in range(0, n, step):
        x = X[s:s + step, None]
        y = Y[s:s + step, None]
        hi = np.min(bu[None, :] - x * Au[None, :, 0] - y * Au[None, :, 1], axis=1) if len(bu) else np.full(len(x), np.inf)
        lw = np.max(bl[None, :] - x * Al[None, :, 0] - y * Al[None, :, 1], axis=1) if len(bl) else np.full(len(x), -np.inf)
        bad = (np.any(bv[None, :] - x * Av[None, :, 0] - y * Av[None, :, 1] < 0, axis=1) if len(bv)
               else np.zeros(len(x), bool))
        empty = bad | (lw > hi)
        z0[s:s + step] = np.where(empty, np.nan, lw)
        z1[s:s + step] = np.where(empty, np.nan, hi)
    return z0, z1


def _hull_volume_centroid(V, F):
    T = V[F]
    o = V.mean(axis=0)
    a, b, c = T[:, 0] - o, T[:, 1] - o, T[:, 2] - o
    vol6 = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    vol = float(vol6.sum() / 6.0)
    if vol <= 0:
        return 0.0, o
    cen = o + (vol6[:, None] * (a + b + c) / 4.0).sum(axis=0) / vol6.sum()
    return vol, cen


def hull_from_graph(model, mesh_id):
    """(vertex ids, faces (local vertex ids into the mesh), ) of MuJoCo's convex hull of a mesh, or None."""
    adr = int(model.mesh_graphadr[mesh_id])
    if adr < 0:
        return None
    g = model.mesh_graph[adr:]
    nv, nf = int(g[0]), int(g[1])
    if nv < 4 or nf < 4:
        return None
    off = 2 + 2 * nv + (nv + 3 * nf)
    faces = np.asarray(g[off:off + 3 * nf], np.int64).reshape(nf, 3)
    return faces


def _planes_of(V, F):
    T = V[F]
    cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ar = np.linalg.norm(cr, axis=1)
    keep = ar > 1e-18
    n = cr[keep] / ar[keep][:, None]
    cen = V.mean(axis=0)
    flip = np.einsum("ij,ij->i", n, T[keep, 0] - cen) < 0
    n[flip] *= -1.0
    b = np.einsum("ij,ij->i", n, T[keep, 0])
    return n, b


def geom_piece(model, data, g):
    """The convex collision piece of geom g in world coordinates (after mj_forward), or None for a plane."""
    import mujoco
    t = int(model.geom_type[g])
    c = np.asarray(data.geom_xpos[g], float)
    R = np.asarray(data.geom_xmat[g], float).reshape(3, 3)
    s = np.asarray(model.geom_size[g], float)
    T = mujoco.mjtGeom
    if t == T.mjGEOM_BOX:
        return Piece("box", c, R, p=s[:3], geom=g)
    if t == T.mjGEOM_SPHERE:
        return Piece("sphere", c, R, p=s[:1], geom=g)
    if t == T.mjGEOM_CAPSULE:
        return Piece("capsule", c, R, p=s[:2], geom=g)
    if t == T.mjGEOM_CYLINDER:
        return Piece("cylinder", c, R, p=s[:2], geom=g)
    if t == T.mjGEOM_ELLIPSOID:
        return Piece("ellipsoid", c, R, p=s[:3], geom=g)
    if t == T.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        F = hull_from_graph(model, mid)
        if F is None:
            return None
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        Vl = np.asarray(model.mesh_vert[va:va + vn], float)
        used = np.unique(F)
        remap = np.full(vn, -1, np.int64)
        remap[used] = np.arange(len(used))
        V = Vl[used] @ R.T + c
        F2 = remap[F]
        A, b = _planes_of(V, F2)
        return Piece("hull", c, R, V=V, F=F2, A=A, b=b, geom=g)
    return None


_BLOCK_ENTRIES = 2_000_000          # matrix entries per block in the chunked support / union computations
_RASTER_MIN_FACES = 64              # hulls with more faces may be rasterised instead of tested plane by plane
_RASTER_TOL = 1e-9                  # m: a column on a shared edge counts for both faces


def _raster_prepare(V, F):
    """Per hull face: the projected triangle's inward edge lines (unit normals, offsets) and its plane, split into
    upper (outward normal up) and lower faces; vertical faces are dropped (they bound no column)."""
    T = V[F]
    cr = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    nrm = np.linalg.norm(cr, axis=1)
    cen = V.mean(axis=0)
    flip = np.einsum("ij,ij->i", cr, T[:, 0] - cen) < 0
    cr[flip] *= -1.0
    nz = np.where(nrm > 0, cr[:, 2] / np.where(nrm > 0, nrm, 1.0), 0.0)
    out = {}
    for name, sel in (("up", nz > 1e-12), ("lo", nz < -1e-12)):
        Ts, cs = T[sel], cr[sel]
        P = Ts[:, :, :2]
        area2 = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
        sgn = np.where(area2 >= 0, 1.0, -1.0)
        En, Eo, ok = [], [], np.abs(area2) > 0
        for a, b in ((0, 1), (1, 2), (2, 0)):
            e = P[:, b] - P[:, a]
            L = np.linalg.norm(e, axis=1)
            ok &= L > 0
            n = np.c_[-e[:, 1], e[:, 0]] * (sgn / np.where(L > 0, L, 1.0))[:, None]     # inward unit normal
            En.append(n)
            Eo.append(np.einsum("ij,ij->i", n, P[:, a]))
        En = np.stack(En, axis=1)[ok]
        Eo = np.stack(Eo, axis=1)[ok]
        P, Ts, cs = P[ok], Ts[ok], cs[ok]
        out[name] = {"lo": P.min(axis=1), "hi": P.max(axis=1), "En": En, "Eo": Eo, "p0": Ts[:, 0], "cr": cs}
    return out


def _raster_ranges(fd, h, i0, i1, j0, j1):
    t = _RASTER_TOL / h
    ia = np.maximum(np.ceil(fd["lo"][:, 0] / h - 0.5 - t), i0).astype(np.int64)
    ib = np.minimum(np.floor(fd["hi"][:, 0] / h - 0.5 + t), i1 - 1).astype(np.int64)
    ja = np.maximum(np.ceil(fd["lo"][:, 1] / h - 0.5 - t), j0).astype(np.int64)
    jb = np.minimum(np.floor(fd["hi"][:, 1] / h - 0.5 + t), j1 - 1).astype(np.int64)
    ni = np.maximum(ib - ia + 1, 0)
    nj = np.maximum(jb - ja + 1, 0)
    return ia, ja, ni, nj


def _raster_candidates(rd, h, i0, i1, j0, j1):
    n = 0
    for fd in rd.values():
        _ia, _ja, ni, nj = _raster_ranges(fd, h, i0, i1, j0, j1)
        n += int(np.sum(ni * nj))
    return n


def _raster_hull(rd, h, i0, i1, j0, j1):
    """(z0, z1) flat over the columns of [i0, i1) x [j0, j1): z1 the smallest plane height of the upper faces whose
    projection covers the column, z0 the largest of the lower faces (exact for a convex hull: every face plane lies
    above / below the whole hull), NaN where no face pair covers the column."""
    ncol = (i1 - i0) * (j1 - j0)
    nj_all = j1 - j0
    z = {"up": np.full(ncol, np.inf), "lo": np.full(ncol, -np.inf)}
    for name, fd in rd.items():
        ia, ja, ni, nj = _raster_ranges(fd, h, i0, i1, j0, j1)
        cnt = ni * nj
        faces = np.nonzero(cnt > 0)[0]
        if not len(faces):
            continue
        csum = np.cumsum(cnt[faces])
        k0 = 0
        while k0 < len(faces):
            base = csum[k0 - 1] if k0 else 0
            k1 = int(np.searchsorted(csum, base + _BLOCK_ENTRIES, side="right"))
            k1 = max(k1, k0 + 1)
            fb = faces[k0:k1]
            c = cnt[fb]
            tot = int(c.sum())
            rep = np.repeat(np.arange(len(fb)), c)
            off = np.arange(tot, dtype=np.int64) - np.repeat(np.cumsum(c) - c, c)
            f = fb[rep]
            njf = nj[f]
            ii = ia[f] + off // njf
            jj = ja[f] + off % njf
            x = (ii + 0.5) * h
            y = (jj + 0.5) * h
            En, Eo = fd["En"][f], fd["Eo"][f]
            inside = np.ones(tot, bool)
            for e in range(3):
                inside &= En[:, e, 0] * x + En[:, e, 1] * y - Eo[:, e] >= -_RASTER_TOL
            if np.any(inside):
                f, x, y = f[inside], x[inside], y[inside]
                cr, p0 = fd["cr"][f], fd["p0"][f]
                zz = p0[:, 2] - (cr[:, 0] * (x - p0[:, 0]) + cr[:, 1] * (y - p0[:, 1])) / cr[:, 2]
                flat = (ii[inside] - i0) * nj_all + (jj[inside] - j0)
                if name == "up":
                    np.minimum.at(z["up"], flat, zz)
                else:
                    np.maximum.at(z["lo"], flat, zz)
            k0 = k1
    z1, z0 = z["up"], z["lo"]
    ok = np.isfinite(z1) & np.isfinite(z0) & (z0 <= z1)
    return np.where(ok, z0, np.nan), np.where(ok, z1, np.nan)


def _merge_segments(S, E):
    """Disjoint z segments of the union along each row of S, E (n, P) (inf / -inf for unused slots): each segment
    contributes the part of its interval not covered by intervals that start earlier. (Z0, Z1), NaN unused."""
    n, P = S.shape
    if P == 1:
        ok = np.isfinite(S)
        return np.where(ok, S, np.nan), np.where(ok, E, np.nan)
    order = np.argsort(S, axis=1, kind="stable")
    S = np.take_along_axis(S, order, axis=1)
    E = np.take_along_axis(E, order, axis=1)
    M = np.maximum.accumulate(E, axis=1)
    prev = np.concatenate([np.full((n, 1), -np.inf), M[:, :-1]], axis=1)
    lo = np.maximum(S, prev)
    valid = np.isfinite(S) & (E > lo)
    return np.where(valid, lo, np.nan), np.where(valid, E, np.nan)


def grid_extent(pieces, h):
    """(i0, i1, j0, j1, piece columns): the grid columns of the pieces' joint bounding box (cells i in [i0, i1)
    with centres (i + 0.5) h) and the sum over pieces of the columns their own boxes cover (the work)."""
    lo = np.min([p.bbox()[0] for p in pieces], axis=0)
    hi = np.max([p.bbox()[1] for p in pieces], axis=0)
    i0, i1 = int(math.floor(lo[0] / h)), int(math.ceil(hi[0] / h))
    j0, j1 = int(math.floor(lo[1] / h)), int(math.ceil(hi[1] / h))
    work = 0
    for p in pieces:
        a0, a1, b0, b1 = p.index_box(h)
        work += max(0, min(a1, i1) - max(a0, i0)) * max(0, min(b1, j1) - max(b0, j0))
    return i0, i1, j0, j1, work


def grid_union(pieces, h, i0, i1, j0, j1, budget=_BLOCK_ENTRIES):
    """Yield (ia, ib, Z0, Z1) for bands of grid rows [ia, ib): Z0, Z1 of shape ((ib - ia) * (j1 - j0), P_band) hold
    the disjoint z segments of the union of the pieces along each column of the band (row-major, segments in z
    order, NaN for unused slots). Only the pieces whose boxes meet a band take part in it, and each piece is
    evaluated only on its own columns, so memory stays within `budget` entries per band."""
    nj = j1 - j0
    boxes = [p.index_box(h) for p in pieces]
    nb = max(1, int(budget // max(1, nj * len(pieces))))
    ia = i0
    while ia < i1:
        ib = min(i1, ia + nb)
        band = [k for k, bx in enumerate(boxes) if bx[0] < ib and bx[1] > ia and bx[2] < j1 and bx[3] > j0]
        n = (ib - ia) * nj
        S = np.full((n, max(1, len(band))), np.inf)
        E = np.full((n, max(1, len(band))), -np.inf)
        for col, k in enumerate(band):
            a0, a1, b0, b1 = boxes[k]
            ra, rb = max(ia, a0), min(ib, a1)
            ca, cb = max(j0, b0), min(j1, b1)
            if ra >= rb or ca >= cb:
                continue
            z0, z1 = pieces[k].grid_interval(h, ra, rb, ca, cb)
            ok = ~np.isnan(z0)
            if not np.any(ok):
                continue
            idx = ((np.arange(ra, rb) - ia)[:, None] * nj + (np.arange(ca, cb) - j0)[None, :])[ok]
            S[idx, col] = z0[ok]
            E[idx, col] = z1[ok]
        Z0, Z1 = _merge_segments(S, E)
        yield ia, ib, Z0, Z1
        ia = ib


def union_integrals(pieces, h, max_columns=4_000_000, max_piece_columns=None):
    """Volume and centroid of the union of convex pieces: columns along z every h in x and y (cell centres), the z
    extent of each column integrated exactly. h is doubled until the grid has at most max_columns columns and the
    pieces' own boxes at most max_piece_columns (default 4 x max_columns) columns in total, which bounds the work.
    Returns (volume, centroid, h used)."""
    mpc = 4 * max_columns if max_piece_columns is None else max_piece_columns
    while True:
        i0, i1, j0, j1, work = grid_extent(pieces, h)
        if (i1 - i0) * (j1 - j0) <= max_columns and work <= mpc:
            break
        h *= 2.0
    vol = mx = my = mz = 0.0
    Yj = (np.arange(j0, j1) + 0.5) * h
    for ia, ib, Z0, Z1 in grid_union(pieces, h, i0, i1, j0, j1):
        empty = np.isnan(Z0)
        L = np.where(empty, 0.0, Z1 - Z0).sum(axis=1).reshape(ib - ia, j1 - j0)
        mom = np.where(empty, 0.0, 0.5 * (Z1 * Z1 - Z0 * Z0)).sum()
        Xi = (np.arange(ia, ib) + 0.5) * h
        vol += float(L.sum())
        mx += float((L.sum(axis=1) * Xi).sum())
        my += float((L.sum(axis=0) * Yj).sum())
        mz += float(mom)
    if vol <= 0:
        return 0.0, np.full(3, np.nan), h
    return vol * h * h, np.array([mx / vol, my / vol, mz / vol]), h


def column_union(pieces, X, Y):
    """Disjoint z segments of the union of the pieces along the columns (X, Y) (N,): (Z0, Z1) of shape (N, P), NaN
    for unused slots (for small inputs; the grader's integrals use grid_union)."""
    n = len(X)
    S = np.full((n, len(pieces)), np.inf)
    E = np.full((n, len(pieces)), -np.inf)
    for k, pc in enumerate(pieces):
        blo, bhi = pc.bbox()
        sel = (X >= blo[0]) & (X <= bhi[0]) & (Y >= blo[1]) & (Y <= bhi[1])
        if not np.any(sel):
            continue
        z0, z1 = pc.z_interval(X[sel], Y[sel])
        ok = ~np.isnan(z0)
        idx = np.nonzero(sel)[0][ok]
        S[idx, k] = z0[ok]
        E[idx, k] = z1[ok]
    return _merge_segments(S, E)


def com_hull_margin(pieces, C, extra_dirs=None, tol=0.0):
    """max over test directions u of u . C - H(u), H the support function of the union of the pieces: > tol means C
    lies outside the convex hull of the collision geometry. Test directions: 512 spread directions, extra_dirs, and
    every piece's face normals (for its own faces a piece's support is its plane offset). Evaluated in blocks; a
    face-normal direction is dropped as soon as some piece reaches past C along it, so memory and time stay bounded.
    Returns (margin over the spread and extra directions, largest margin over the face normals or -inf)."""
    C = np.asarray(C, float).reshape(3)
    U0 = fibonacci_dirs(512) if extra_dirs is None else np.vstack([fibonacci_dirs(512), extra_dirs])
    H0 = np.max([p.support(U0) for p in pieces], axis=0)
    m0 = float(np.max(U0 @ C - H0))
    worst = -np.inf
    for k, p in enumerate(pieces):
        if p.A is None:
            continue
        for a in range(0, len(p.A), 4096):
            U = p.A[a:a + 4096]
            r = U @ C - p.b[a:a + 4096]               # the piece's own support along its face normals is b
            keep = r > tol
            for q, other in enumerate(pieces):
                if q == k or not np.any(keep):
                    continue
                idx = np.nonzero(keep)[0]
                r[idx] = np.minimum(r[idx], U[idx] @ C - other.support(U[idx]))
                keep[idx] = r[idx] > tol
            if np.any(keep):
                worst = max(worst, float(np.max(r[keep])))
    return m0, worst


# ---------------------------------------------------------------------------------------------------------------
# 6. and 7. compiled checks

def _subtree(model, root_body):
    out = []
    for b in range(model.nbody):
        k = b
        while k > 0 and k != root_body:
            k = int(model.body_parentid[k])
        if k == root_body and b != 0:
            out.append(b)
    return out


def _name(model, objtype, i):
    import mujoco
    nm = mujoco.mj_id2name(model, objtype, int(i))
    return nm if nm else "#%d" % int(i)


def compiled_checks(model, spec, objects, vocabulary, col, rules):
    """Twin-level checks on the compiled grader copy; returns (camera dict, item records, drawer record)."""
    import mujoco
    m = model
    zero = ["neq", "ntendon", "nu", "nflex", "nplugin", "nmocap", "nexclude", "npair", "nkey", "nsensor", "nhfield",
            "nskin", "nnumeric", "ntext", "ntuple"]
    for k in zero:
        if int(getattr(m, k)) != 0:
            col.fail("the compiled twin has %s = %d (must be 0)" % (k, int(getattr(m, k))))
    nb = int(m.nbody) - 1
    if nb > rules["max_bodies"]:
        col.fail("%d bodies (cap %d)" % (nb, rules["max_bodies"]))
    if int(m.ngeom) > rules["max_geoms"]:
        col.fail("%d geoms (cap %d)" % (int(m.ngeom), rules["max_geoms"]))
    colg = [g for g in range(m.ngeom) if is_collision_geom(m, g)]
    if len(colg) > rules["max_collision_geoms"]:
        col.fail("%d collision geoms (cap %d)" % (len(colg), rules["max_collision_geoms"]))
    if int(m.nmeshface) > rules["max_mesh_faces_total"]:
        col.fail("%d mesh faces in total (cap %d)" % (int(m.nmeshface), rules["max_mesh_faces_total"]))
    T = mujoco.mjtGeom
    ok_types = {int(T.mjGEOM_PLANE), int(T.mjGEOM_BOX), int(T.mjGEOM_SPHERE), int(T.mjGEOM_CAPSULE),
                int(T.mjGEOM_CYLINDER), int(T.mjGEOM_ELLIPSOID), int(T.mjGEOM_MESH)}
    for g in colg:
        if int(m.geom_type[g]) not in ok_types:
            col.fail("collision geom %s has an unsupported type" % _name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
        elif int(m.geom_type[g]) == int(T.mjGEOM_MESH):
            mid = int(m.geom_dataid[g])
            if int(m.mesh_vertnum[mid]) > rules["max_collision_mesh_vertices"]:
                col.fail("collision mesh %s has %d vertices (cap %d)" % (
                    _name(m, mujoco.mjtObj.mjOBJ_MESH, mid), int(m.mesh_vertnum[mid]),
                    rules["max_collision_mesh_vertices"]))
            elif int(m.mesh_graphadr[mid]) < 0:
                col.fail("collision mesh %s has no convex hull" % _name(m, mujoco.mjtObj.mjOBJ_MESH, mid))
    # finite, bounded numbers
    big = rules["max_abs_coordinate_m"]
    for k in ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "geom_size", "geom_pos",
              "geom_quat", "geom_friction", "mesh_vert", "cam_pos", "cam_quat", "jnt_pos", "jnt_axis", "qpos0",
              "light_pos", "site_pos"):
        a = np.asarray(getattr(m, k), float)
        if a.size and not np.all(np.isfinite(a)):
            col.fail("the compiled twin has non-finite values in %s" % k)
    if m.nbody > 1 and float(np.max(np.abs(m.body_pos))) > big:
        col.fail("a body lies more than %g m from its parent" % big)
    if m.ngeom and float(np.max(np.abs(m.geom_size))) > rules["max_geom_size_m"]:
        col.fail("a geom size exceeds %g m" % rules["max_geom_size_m"])
    # mechanisms the whitelist already closes (checked again on the compiled values)
    checks = [("jnt_stiffness", "joint stiffness"), ("jnt_stiffnesspoly", "joint stiffness"),
              ("dof_damping", "joint damping"), ("dof_dampingpoly", "joint damping"), ("dof_armature", "armature"),
              ("dof_frictionloss", "friction loss"), ("geom_adhesion", "adhesion"), ("geom_surfacevel", "surface velocity"),
              ("geom_fluid", "fluid forces"), ("body_gravcomp", "gravity compensation"), ("jnt_margin", "joint margin")]
    for k, label in checks:
        if hasattr(m, k):
            a = np.asarray(getattr(m, k), float)
            if a.size and float(np.max(np.abs(a))) != 0.0:
                col.fail("the compiled twin has non-zero %s (%s)" % (label, k))
    # camera
    cam = {"name": rules["camera"]}
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, rules["camera"])
    if cid < 0:
        col.fail("no camera named %s" % rules["camera"])
    else:
        if int(m.cam_mode[cid]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
            col.fail("%s must have mode fixed" % rules["camera"])
        if int(m.body_weldid[int(m.cam_bodyid[cid])]) != 0:
            col.fail("%s must sit on a static body (the world or a body welded to it)" % rules["camera"])
        W, H = (int(v) for v in m.cam_resolution[cid])
        if [W, H] != list(rules["resolution"]):
            col.fail("%s resolution is %d x %d (must be %d x %d)" % (rules["camera"], W, H, *rules["resolution"]))
        try:
            import twin_camera
        except ImportError:  # pragma: no cover - twin_camera sits next to this file
            sys.path.insert(0, HERE)
            import twin_camera
        try:
            K, W2, H2 = twin_camera.decode(m, cid)
            fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
            cam.update({"fx": fx, "fy": fy, "cx": cx, "cy": cy, "W": W2, "H": H2})
            lo, hi = rules["focal_px"]
            if not (lo <= fx <= hi and lo <= fy <= hi):
                col.fail("%s focal lengths fx %.2f, fy %.2f px are outside [%g, %g]" % (rules["camera"], fx, fy, lo, hi))
            if not (-0.5 <= cx <= W2 - 0.5 and -0.5 <= cy <= H2 - 0.5):
                col.fail("%s principal point (%.2f, %.2f) lies outside the image" % (rules["camera"], cx, cy))
        except Exception as exc:  # noqa: BLE001 - twin_camera.CameraError
            col.fail("%s intrinsics cannot be decoded: %s" % (rules["camera"], exc))
    # bodies, joints
    names = {_name(m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(m.nbody)}
    items = []
    listed = {}
    for it in (objects or {}).get("items", []) if isinstance((objects or {}).get("items"), list) else []:
        if isinstance(it, dict) and isinstance(it.get("body"), str):
            listed[it["body"]] = it.get("label")
    free_bodies = set()
    slide = []
    for j in range(m.njnt):
        t = int(m.jnt_type[j])
        b = int(m.jnt_bodyid[j])
        if t == int(mujoco.mjtJoint.mjJNT_FREE):
            if int(m.body_parentid[b]) != 0:
                col.fail("free joint on body %s, which is not a top-level body" % _name(m, mujoco.mjtObj.mjOBJ_BODY, b))
            free_bodies.add(b)
        elif t == int(mujoco.mjtJoint.mjJNT_SLIDE):
            slide.append(j)
        else:
            col.fail("joint %s is neither free nor slide" % _name(m, mujoco.mjtObj.mjOBJ_JOINT, j))
    for b in free_bodies:
        if int(m.body_jntnum[b]) != 1:
            col.fail("item body %s carries %d joints (only its free joint)" % (_name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                                                                               int(m.body_jntnum[b])))
        nm = _name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if nm not in listed:
            col.fail("free body %s is not listed in objects.json (every loose object is an item)" % nm)
    vocab = set(vocabulary)
    for nm, label in listed.items():
        b = names.get(nm)
        rec = {"body": nm, "label": label, "label_in_vocabulary": label in vocab, "status": "ok", "reasons": []}
        if b is None:
            col.fail("objects.json item %s is not a body of scene.xml" % nm)
            rec["status"] = "missing"
        elif b not in free_bodies:
            col.fail("objects.json item %s is not a top-level body with a free joint" % nm)
            rec["status"] = "missing"
        else:
            rec["body_id"] = b
        if not rec["label_in_vocabulary"]:
            rec["reasons"].append("label %r is not in the product vocabulary (it never counts as correct)" % label)
        items.append(rec)
    # the drawer
    drawer = {"body": (objects or {}).get("drawer")}
    db = names.get(drawer["body"]) if isinstance(drawer["body"], str) else None
    if db is None:
        col.fail("the drawer body %r named in objects.json is not a body of scene.xml" % drawer["body"])
    else:
        drawer["body_id"] = db
        if int(m.body_jntnum[db]) != 1 or int(m.jnt_type[int(m.body_jntadr[db])]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
            col.fail("the drawer body %s must carry exactly one slide joint" % drawer["body"])
        if int(m.body_parentid[db]) != 0 and int(m.body_weldid[int(m.body_parentid[db])]) != 0:
            col.fail("the drawer body %s must hang from a static body" % drawer["body"])
        for b in _subtree(m, db):
            if b != db and int(m.body_jntnum[b]) != 0:
                col.fail("body %s inside the drawer carries a joint" % _name(m, mujoco.mjtObj.mjOBJ_BODY, b))
    if len(slide) != 1:
        col.fail("the twin has %d slide joints (exactly one: the drawer's)" % len(slide))
    elif db is not None and int(m.jnt_bodyid[slide[0]]) != db:
        col.fail("the slide joint is not on the drawer body %s" % drawer["body"])
    elif db is not None:
        drawer["travel"] = drawer_travel(m, db, slide[0], rules)
        if not drawer["travel"]["ok"]:
            col.warn("drawer travel rule not met (C5's drawer term scores 0): " + "; ".join(drawer["travel"]["reasons"]))
    return cam, items, drawer


def drawer_corners(model, d, body):
    """World corners of the bounding boxes of the drawer's collision geoms (the drawer body's subtree), at d's state."""
    bodies = set(_subtree(model, body))
    P = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) not in bodies or not is_collision_geom(model, g):
            continue
        c, h = np.asarray(model.geom_aabb[g][:3], float), np.asarray(model.geom_aabb[g][3:], float)
        R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    P.append(np.asarray(d.geom_xpos[g], float) + R @ (c + h * np.array([sx, sy, sz])))
    return np.asarray(P).reshape(-1, 3)


def drawer_out_sign(model, d, body, joint, camera="photo_cam"):
    """+1 if increasing the slide moves the drawer toward the camera, else -1 (twin_sim.drawer_out_sign, the same
    rule): from the centre of the drawer's collision geometry, not the body origin; on a tie, the camera's +z."""
    import mujoco
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
    if cam < 0:
        return 1.0
    axis = np.asarray(d.xaxis[joint], float)
    X = drawer_corners(model, d, body)
    ctr = 0.5 * (X.min(axis=0) + X.max(axis=0)) if len(X) else np.asarray(d.xpos[body], float)
    v = float(axis @ (np.asarray(d.cam_xpos[cam], float) - ctr))
    if abs(v) < 1e-9:
        v = float(axis @ np.asarray(d.cam_xmat[cam], float).reshape(3, 3)[:, 2])
    return 1.0 if v > 0 else -1.0


def drawer_travel(model, body, joint, rules):
    """The drawer's travel against its own depth (rules["drawer_travel"]). Depth: the extent along the slide axis of the
    drawer's collision geoms (their bounding boxes, at qpos0). Out: the joint direction that moves the drawer toward
    photo_cam. Returns {"ok", "reasons", "limited", "range_mm", "qpos0_mm", "depth_mm", "travel_mm", "out_mm", ...}."""
    import mujoco
    m = model
    R = rules["drawer_travel"]
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    axis = np.asarray(d.xaxis[joint], float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    X = drawer_corners(m, d, body)
    depth = float(np.max(X @ axis) - np.min(X @ axis)) if len(X) else 0.0
    limited = bool(m.jnt_limited[joint])
    lo, hi = (float(v) for v in m.jnt_range[joint])
    q0 = float(m.qpos0[int(m.jnt_qposadr[joint])])
    sign = drawer_out_sign(m, d, body, joint, rules["camera"])
    out_m = (hi - q0) if sign > 0 else (q0 - lo)
    travel = hi - lo
    reasons = []
    eps = 1e-6
    if not limited:
        reasons.append("the drawer slide has no limits (give it a range, the real drawer's travel)")
    else:
        if not (lo - eps <= q0 <= hi + eps):
            reasons.append("the photographed opening (qpos0 %.1f mm) is outside the slide range [%.1f, %.1f] mm" % (
                1000 * q0, 1000 * lo, 1000 * hi))
        if travel < float(R["min_depth_ratio"]) * depth - eps:
            reasons.append("the travel %.1f mm is below %g x the drawer's depth (%.1f mm)" % (
                1000 * travel, float(R["min_depth_ratio"]), 1000 * depth))
        if travel > float(R["max_depth_ratio"]) * depth + eps:
            reasons.append("the travel %.1f mm is above %g x the drawer's depth (%.1f mm)" % (
                1000 * travel, float(R["max_depth_ratio"]), 1000 * depth))
        if out_m < float(R["min_out_mm"]) / 1000.0 - eps:
            reasons.append("the drawer comes out only %.1f mm beyond the photographed opening (at least %g mm)" % (
                1000 * out_m, float(R["min_out_mm"])))
    return {"ok": not reasons, "reasons": reasons, "limited": limited,
            "range_mm": [round(1000 * lo, 3), round(1000 * hi, 3)] if limited else None,
            "qpos0_mm": round(1000 * q0, 3), "depth_mm": round(1000 * depth, 3), "travel_mm": round(1000 * travel, 3)
            if limited else None, "out_mm": round(1000 * out_m, 3) if limited else None,
            "out_direction": "+axis" if sign > 0 else "-axis",
            "travel_over_depth": round(travel / depth, 4) if (limited and depth > 0) else None}


def is_collision_geom(model, g):
    """A geom collides when (contype, conaffinity) != (0, 0) (the same rule as the standard simulation's)."""
    return int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0


def item_checks(model, spec, rec, rules, data=None):
    """Per-item contract and plausibility checks; fills rec (status 'ok' or 'excluded' with reasons)."""
    import mujoco
    m = model
    b = rec["body_id"]
    d = data
    if d is None:
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
    bodies = _subtree(m, b)
    geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies]
    coll, vis, why = [], [], []
    sgeoms = {int(gs.id): gs for gs in spec.geoms} if spec is not None else {}
    for g in geoms:
        nm = _name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        if is_collision_geom(m, g):
            coll.append(g)
        else:
            vis.append(g)
            gs = sgeoms.get(g)
            if gs is not None:
                mass = float(gs.mass)
                dens = float(gs.density)
                massless = (mass == 0.0) or (math.isnan(mass) and dens == 0.0)
                if not massless:
                    why.append("visual geom %s must have density 0 (or mass 0)" % nm)
    rec["n_collision_geoms"] = len(coll)
    rec["n_visual_geoms"] = len(vis)
    hv = sum(int(m.mesh_vertnum[int(m.geom_dataid[g])]) for g in coll
             if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH))
    if hv > rules["max_item_hull_vertices"]:
        why.append("its collision meshes hold %d vertices (cap %d per item)" % (hv, rules["max_item_hull_vertices"]))
        rec["reasons"] = rec.get("reasons", []) + why
        rec["status"] = "excluded"
        rec.pop("body_id", None)
        return rec
    if not coll:
        why.append("no collision geom (a geom with contype or conaffinity non-zero)")
    if len(coll) > rules["max_collision_geoms_per_item"]:
        why.append("%d collision geoms (cap %d)" % (len(coll), rules["max_collision_geoms_per_item"]))
    pieces = []
    for g in coll:
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            why.append("a plane cannot be part of an item")
            continue
        pc = geom_piece(m, d, g)
        if pc is None:
            why.append("collision geom %s has no convex shape" % _name(m, mujoco.mjtObj.mjOBJ_GEOM, g))
            continue
        pieces.append(pc)
    if coll:
        fr = np.asarray(m.geom_friction[coll, 0], float)
        rec["friction"] = [float(fr.min()), float(fr.max())]
        lo, hi = rules["friction"]
        if fr.min() < lo or fr.max() > hi:
            why.append("sliding friction %.3g..%.3g is outside [%g, %g]" % (fr.min(), fr.max(), lo, hi))
    # composite mass, centre of mass and inertia of the item's bodies at qpos0
    M = float(sum(float(m.body_mass[k]) for k in bodies))
    rec["mass_kg"] = M
    if pieces and M > 0 and math.isfinite(M):
        C = sum(float(m.body_mass[k]) * np.asarray(d.xipos[k], float) for k in bodies) / M
        I = np.zeros((3, 3))
        for k in bodies:
            Rk = np.asarray(d.ximat[k], float).reshape(3, 3)
            r = np.asarray(d.xipos[k], float) - C
            I += Rk @ np.diag(np.asarray(m.body_inertia[k], float)) @ Rk.T + float(m.body_mass[k]) * (
                (r @ r) * np.eye(3) - np.outer(r, r))
        w, U = np.linalg.eigh(0.5 * (I + I.T))
        lo_b = np.min([p.bbox()[0] for p in pieces], axis=0)
        hi_b = np.max([p.bbox()[1] for p in pieces], axis=0)
        if float(np.max(hi_b - lo_b)) > rules["max_item_extent_m"]:
            why.append("the item is larger than %g m" % rules["max_item_extent_m"])
        else:
            dirs = np.vstack([U.T, -U.T])
            hs = np.max([p.support(dirs) for p in pieces], axis=0)
            ext = hs[:3] + hs[3:]
            Lmax = float(np.max(ext))
            Lmin = float(np.min(ext))
            rec["extent_m"] = [float(v) for v in ext]
            ratio = float(w[0] / (M * Lmax * Lmax)) if Lmax > 0 else 0.0
            rec["inertia_min_ratio"] = ratio
            if not ratio >= rules["inertia_min_ratio"]:
                why.append("smallest principal inertia %.3g is below %g x mass x L^2" % (w[0], rules["inertia_min_ratio"]))
            # upper bound: any mass distribution within distance d_max of its centre of mass has I <= M d_max^2
            dmax = max(p.max_dist(C) for p in pieces)
            rec["d_max_m"] = float(dmax)
            cap = M * dmax * dmax
            rmax = float(w[2] / cap) if cap > 0 else float("inf")
            rec["inertia_max_ratio"] = rmax
            if not rmax <= 1.0 + rules["inertia_max_tolerance"]:
                why.append("largest principal inertia %.3g is above mass x d_max^2 = %.3g (d_max %.1f mm, the farthest "
                           "point of the collision geometry from the centre of mass): no mass distribution inside the "
                           "geometry has it" % (w[2], cap, 1000.0 * dmax))
            m0, mf = com_hull_margin(pieces, C, dirs, rules["com_tolerance_m"])
            margin = max(m0, mf)
            rec["com_hull_margin_m"] = float(margin)
            if margin > rules["com_tolerance_m"]:
                why.append("the centre of mass lies %.1f mm outside the convex hull of the collision geometry" % (
                    1000.0 * margin))
            if len(pieces) == 1:
                vol = pieces[0].volume()
            else:
                h = min(rules["volume_grid_max_m"], max(rules["volume_grid_min_m"], Lmin / 12.0))
                area = float((hi_b[0] - lo_b[0]) * (hi_b[1] - lo_b[1]))
                h = max(h, math.sqrt(area / rules["volume_grid_max_columns"]))
                vol = union_integrals(pieces, h, rules["volume_grid_max_columns"],
                                      rules["volume_grid_max_piece_columns"])[0]
            rec["volume_m3"] = float(vol)
            dens = M / vol if vol > 0 else float("inf")
            rec["density_kg_m3"] = float(dens)
            lo, hi = rules["density_kg_m3"]
            if not lo <= dens <= hi:
                why.append("density %.4g kg/m^3 (mass / volume of the collision geometry) is outside [%g, %g]" % (
                    dens, lo, hi))
    elif pieces:
        why.append("mass %.4g kg is not positive" % M)
    rec["reasons"] = rec.get("reasons", []) + why
    rec["status"] = "excluded" if why else "ok"
    rec.pop("body_id", None)
    return rec


# ---------------------------------------------------------------------------------------------------------------
# the gate

def gate(sub_dir, vocabulary=None, rules=None):
    """Run every check on a submission folder. Returns a GateResult (never raises for a bad submission)."""
    R = dict(DEFAULT_RULES, **(rules or {}))
    col = _Collector()
    root = os.path.abspath(sub_dir)
    rep = {"format": "twin-gate-report/1", "gate_version": GATE_VERSION, "submission": "", "pass": False,
           "reasons": col.reasons, "warnings": col.warnings, "items": [], "excluded": [], "drawer": None,
           "drawer_travel": None,
           "camera": None, "stats": {}}
    vocab = load_vocabulary(vocabulary)               # TaskError if the task's vocabulary is broken
    res = GateResult(rep)
    if not os.path.isdir(root) or os.path.islink(root):
        col.fail("the submission is not a directory")
        return res
    files = {}
    for nm in ("scene.xml", "objects.json"):
        p, why = _regular_file(root, nm)
        if why:
            col.fail("%s %s (a regular file at the submission root is required)" % (nm, why))
        else:
            files[nm] = p
    entries, total, irregular = tree_stats(root)
    rep["stats"].update({"entries": entries, "bytes": total})
    if entries > R["max_entries"]:
        col.fail("the submission holds %d entries (cap %d)" % (entries, R["max_entries"]))
    if total > R["max_bytes"]:
        col.fail("the submission holds %d bytes (cap %d)" % (total, R["max_bytes"]))
    if irregular:
        col.warn("ignored (never read): %s" % ", ".join(irregular[:8]))
    if col.reasons:
        return res
    raw = _read_bytes(files["scene.xml"], R["max_scene_bytes"] + 1, col, "scene.xml")
    if raw is None:
        return res
    tree = static_scene(raw, col, R)
    if tree is None:
        return res
    n = count_static(tree)
    rep["stats"]["static"] = n
    if n["body"] > R["max_bodies"]:
        col.fail("scene.xml has %d bodies (cap %d)" % (n["body"], R["max_bodies"]))
    if n["geom"] > R["max_geoms"]:
        col.fail("scene.xml has %d geoms (cap %d)" % (n["geom"], R["max_geoms"]))
    if n["mesh"] > R["max_mesh_assets"]:
        col.fail("scene.xml has %d meshes (cap %d)" % (n["mesh"], R["max_mesh_assets"]))
    if n["texture"] > R["max_textures"]:
        col.fail("scene.xml has %d textures (cap %d)" % (n["texture"], R["max_textures"]))
    if col.reasons:
        return res
    meshes, _textures, mesh_info = check_files(root, tree, col, R)
    oraw = _read_bytes(files["objects.json"], R["max_objects_bytes"] + 1, col, "objects.json")
    if oraw is None:
        return res
    objects = check_objects(oraw, col, R, vocab)
    res.objects = objects
    if col.reasons:
        return res
    xml_text = grader_copy(tree, meshes, R)
    spec, warnings, error = parse_copy(xml_text, meshes)
    if not error and not warnings:
        precompile_caps(spec, mesh_info, col, R)
        if col.reasons:
            return res
        model, warnings, error = compile_copy(spec)
    else:
        model = None
    rep["stats"]["compile_warnings"] = len(warnings)
    if error:
        col.fail("scene.xml does not compile: %s" % error)
        return res
    if warnings:
        col.fail("compiling scene.xml gives %d warning(s): %s" % (len(warnings), " | ".join(w[:200] for w in warnings[:3])))
        return res
    res.model, res.spec = model, spec
    rep["stats"].update({"nbody": int(model.nbody) - 1, "ngeom": int(model.ngeom), "nmesh": int(model.nmesh),
                         "nmeshvert": int(model.nmeshvert), "nmeshface": int(model.nmeshface)})
    cam, items, drawer = compiled_checks(model, spec, objects, vocab, col, R)
    rep["camera"] = cam
    rep["drawer"] = drawer.get("body")
    rep["drawer_travel"] = drawer.get("travel")
    if col.reasons:
        rep["items"] = [{k: v for k, v in it.items() if k != "body_id"} for it in items]
        return res
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = []
    for rec in items:
        try:
            out.append(item_checks(model, spec, rec, R, data))
        except MemoryError:
            raise                                         # a task error (the grader's resources), never an exclusion
        except Exception as exc:  # noqa: BLE001 - a broken item must not crash the gate
            rec.pop("body_id", None)
            rec["status"] = "excluded"
            rec["reasons"] = rec.get("reasons", []) + ["the item could not be checked: %s" % str(exc)[:200]]
            out.append(rec)
    rep["items"] = out
    rep["excluded"] = [r["body"] for r in out if r["status"] == "excluded"]
    rep["pass"] = not col.reasons
    return res


# ---------------------------------------------------------------------------------------------------------------
# physics: the model the simulations run, and G0's settle check

def physics_model(res):
    """The compiled model every simulation of the twin runs (G0's settle check, the standard settle, the support truth,
    the probes): the gate's grader copy with the bodies of the excluded items deleted, so an excluded item can neither
    support nor disturb another item. A fresh model on every call (callers standardise it in place); None when the
    gate produced no model."""
    import copy as _copy
    if res.model is None or res.spec is None:
        return None
    gone = list(res.report.get("excluded") or [])
    if not gone:
        return _copy.deepcopy(res.model)
    sp = res.spec.copy()
    for nm in gone:
        b = sp.body(nm)
        if b is not None:
            sp.delete(b)
    with _CaptureStderr():
        return sp.compile()


def _twin_sim():
    try:
        import twin_sim
    except ImportError:  # pragma: no cover - twin_sim sits next to this file
        sys.path.insert(0, HERE)
        import twin_sim
    return twin_sim


def blowup_check(model, drawer=None, rules=None, sim=None):
    """G0's settle check on a physics model (physics_model(res)): the standard simulation (twin_sim.standardize) for
    blowup_window_s from qpos0 with the drawer held still and nothing else held. Fails only on a numerical blow-up:
    a non-finite state or MuJoCo's bad-acceleration warning; a gain of mechanical energy KE(t) + PE(t) - PE(0) above
    eps = 0.5 x (total item mass) x v_tol^2; or an item whose centre of mass moves faster than g t + v_tol (v_tol =
    blowup_speed_tol_m_s). Energies are MuJoCo's (gravity potential, kinetic energy of every dof; the held drawer
    contributes a constant), sampled at every step. Returns a JSON-ready dict with "fail" and "reasons"."""
    import copy as _copy
    import mujoco
    R = dict(DEFAULT_RULES, **(rules or {}))
    out = {"ran": False, "fail": False, "reasons": []}
    if model is None:
        out["reasons"].append("no model")
        out["fail"] = True
        return out
    TS = sim or _twin_sim()
    m = _copy.deepcopy(model)
    try:
        roles = TS.standardize(m, drawer=drawer)
    except Exception as exc:  # noqa: BLE001 - TwinSimError, a drawer that is not one
        out.update({"fail": True, "reasons": ["the standard simulation does not accept the twin: %s" % str(exc)[:200]]})
        return out
    m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qvel[:] = 0.0
    items = np.asarray(roles.items, int)
    mass = np.asarray(m.body_subtreemass[items], float) if len(items) else np.zeros(0)
    vtol = float(R["blowup_speed_tol_m_s"])
    g = float(np.linalg.norm(m.opt.gravity))
    eps = 0.5 * float(mass.sum()) * vtol * vtol + 1e-9
    dt = float(m.opt.timestep)
    n = int(round(float(R["blowup_window_s"]) / dt))
    dq = roles.drawer_qadr if roles.drawer_joint is not None else None
    dd = roles.drawer_dadr if roles.drawer_joint is not None else None
    q0 = float(d.qpos[dq]) if dq is not None else None
    names = [roles.name(b) for b in items]
    pe0 = None
    gmax, gmax_t = -np.inf, 0.0
    vmax, vmax_t, vmax_item = -np.inf, 0.0, None
    bad = int(mujoco.mjtWarning.mjWARN_BADQACC)
    fail_t = None
    for k in range(n + 1):
        if dq is not None:
            d.qpos[dq] = q0
            d.qvel[dd] = 0.0
        if k < n:
            mujoco.mj_step(m, d)          # its forward pass evaluates the state at time k dt (energy, velocities)
        else:
            mujoco.mj_forward(m, d)
        t = k * dt
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)) and np.all(np.isfinite(d.energy))):
            out["reasons"].append("the state becomes non-finite at t = %.3f s" % t)
            fail_t = t
            break
        if int(d.warning[bad].number) > 0:
            out["reasons"].append("MuJoCo reports bad accelerations (a numerical blow-up) at t = %.3f s" % t)
            fail_t = t
            break
        pe, ke = float(d.energy[0]), float(d.energy[1])
        if pe0 is None:
            pe0 = pe
        gain = ke + pe - pe0
        if gain > gmax:
            gmax, gmax_t = gain, t
        if len(items):
            mujoco.mj_subtreeVel(m, d)
            v = np.linalg.norm(d.subtree_linvel[items], axis=1)
            ex = v - (g * t + vtol)
            a = int(np.argmax(ex))
            if ex[a] > vmax:
                vmax, vmax_t, vmax_item = float(ex[a]), t, names[a]
            if ex[a] > 0:
                out["reasons"].append("item %s moves at %.2f m/s at t = %.3f s, above the free-fall bound g t + %.1f = "
                                      "%.2f m/s" % (names[a], float(v[a]), t, vtol, g * t + vtol))
                fail_t = t
                break
        if gain > eps:
            out["reasons"].append("the mechanical energy grows by %.4g J beyond the potential energy released at t = "
                                  "%.3f s (bound %.4g J = 0.5 x %.4g kg x (%.1f m/s)^2)" % (gain, t, eps, mass.sum(), vtol))
            fail_t = t
            break
    warn = {str(mujoco.mjtWarning(i)).split(".")[-1]: int(d.warning[i].number) for i in range(len(d.warning))
            if d.warning[i].number}
    out.update({"ran": True, "fail": bool(out["reasons"]), "window_s": n * dt, "fail_time_s": fail_t,
                "n_items": int(len(items)), "item_mass_kg": float(mass.sum()), "energy_bound_J": eps,
                "max_energy_gain_J": float(gmax) if np.isfinite(gmax) else None, "max_energy_gain_time_s": gmax_t,
                "max_speed_excess_m_s": float(vmax) if np.isfinite(vmax) else None, "max_speed_excess_time_s": vmax_t,
                "max_speed_excess_item": vmax_item, "speed_tol_m_s": vtol, "warnings": warn})
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="The validity gate (G0) of an environment-twin submission.")
    ap.add_argument("submission", nargs="?", default="/home/user/submission")
    ap.add_argument("--vocab", default=None, help="product_names.json (default: the file next to this script)")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument("--settle", action="store_true", help="also run G0's settle check (needs twin_sim.py)")
    a = ap.parse_args(argv)
    try:
        res = gate(a.submission, vocabulary=a.vocab)
    except TaskError as exc:
        print("task error (not your submission): %s" % exc)
        return 2
    rep = res.report
    if a.settle and res.ok:
        rep["settle_check"] = blowup_check(physics_model(res), drawer=rep.get("drawer"))
        if rep["settle_check"]["fail"]:
            rep["pass"] = False
            rep["reasons"].extend("settle check: " + r for r in rep["settle_check"]["reasons"])
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        print("G0 %s: %s" % ("PASS" if res.ok else "FAIL", a.submission))
        for r in rep["reasons"]:
            print("  - " + r)
        for w in rep["warnings"]:
            print("  (note) " + w)
        if rep.get("camera") and "fx" in rep["camera"]:
            c = rep["camera"]
            print("  photo_cam: fx %.2f fy %.2f cx %.2f cy %.2f (%d x %d)" % (c["fx"], c["fy"], c["cx"], c["cy"], c["W"], c["H"]))
        for it in rep["items"]:
            line = "  item %-24s %-18s %s" % (it["body"], it.get("label"), it["status"])
            if "density_kg_m3" in it:
                line += "  density %.1f kg/m^3" % it["density_kg_m3"]
            print(line)
            for r in it.get("reasons", []):
                print("      - " + r)
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
