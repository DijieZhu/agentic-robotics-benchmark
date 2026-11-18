"""twin_gate: the validity gate (G0) of a site-twin submission.

The same file runs in the check tool you are given and in the grader, so a twin that passes here passes G0 there. It
needs only the Python standard library, numpy and mujoco (plus twin_camera.py and twin_sim.py next to it), and it
reads no ground truth.

    python3 twin_gate.py SUBMISSION_DIR [--vocab class_names.json] [--json] [--settle]

    from twin_gate import gate, settle_check
    res = gate("/home/user/submission")        # res.ok, res.report (JSON-ready dict), res.model (mujoco.MjModel)
    chk = settle_check(res)                    # G0's settle part: {"fail", "reasons", "excluded", ...}

Order of the checks. Nothing in the submission is compiled before every static check has passed, and MuJoCo only
ever sees bytes this module has read and checked:
  1. files: scene.xml and objects.json are regular files at the submission root; the whole folder is within the
     entry and byte caps; a file that cannot be read fails the gate with a reason;
  2. scene.xml, statically: size, UTF-8, no DOCTYPE / ENTITY / namespaces, then the element whitelist with an
     attribute allowlist per element (anything not listed is rejected; there are no height fields: a floor is
     planes, boxes or convex meshes);
  3. every referenced file: a relative path without '..' that resolves inside the submission, a regular file (no
     symbolic links on the way), the right extension, within the size caps, and a consistent header (binary STL
     triangle count, MSH counts, PNG size; an OBJ's mtllib must be a plain file name);
  4. objects.json against the site-twin-objects/1 schema: {"format", "items": [{"body", "label"}],
     "gates": [{"body", "joint"}]};
  5. the grader copy: textures and texture references removed, every mesh path written out in full, compiled from
     memory; every MuJoCo warning is captured (its warning callback, Python warnings with the filter forced to
     "always", and anything written to stderr), so the verdict never depends on the interpreter's warning state;
     any warning fails the gate;
  6. compiled checks: no equality, tendon, actuator, flex, plugin, mocap, contact pair/exclude, keyframe, sensor or
     height field; the caps (the column depth among them: no vertical line may meet more than max_column_depth
     collision geoms); finite numbers; the photo cameras photo_NNN (at least min_photo_cameras valid ones); the
     bodies named in objects.json; the gates; the joints;
  7. per unit: visual geoms massless, collision-geom count, density, friction, inertia, centre of mass and extent. A
     failure here excludes that unit only; it does not fail the twin.

Units, gates, environment. A unit ("item" in objects.json) is a top-level body with exactly one free joint. A gate is
a listed body with exactly one slide joint, hanging from a static body. Everything else that collides is the
environment: geoms of the world body or of bodies without joints. The gate travel rule (gate_travel, measured with
twin_sim.gate_rule and the numbers of twin_sim_config.json "gate_rule") and the hull trap are notes only: they never
fail the twin.

Collision geometry. A geom collides when (contype, conaffinity) != (0, 0); every other geom is visual (it must be
massless and plays no part in the scoring). The standard simulation gives every colliding geom the standard contact
classes, so the delivered bits beyond that make no difference. A mesh geom collides as the convex hull of its vertices
(MuJoCo's own hull is used here), a primitive as itself. A room exported as ONE static mesh therefore collides as a
solid block: the gate notes every static collision geom whose collision shape contains a photo camera centre or, by
more than 1 cm, another body's centre of mass. The geometry helpers in this file (Piece, geom_piece, grid_union,
union_integrals) are the ones the grader's scoring code uses as well.

Column depth. Every simulation and every walk probe casts vertical rays through the twin (the robot looks for the
surface under it every 40 ms), and the cost of one cast grows with the number of collision geoms stacked along the
line. The gate therefore counts, on a grid of column_depth_cell_m squares over the twin, how many collision geoms'
world bounding boxes reach over each square (planes excepted) and fails the twin when a square is reached by more than
max_column_depth geoms. An ordinary twin (a floor, a unit of up to 64 pieces, three units stacked) stays far below.

Per-unit plausibility (step 7). M is the unit's mass, C its centre of mass, I its inertia about C (all bodies of the
unit together), d_max the largest distance from C to the unit's collision geometry:
  - density M / V within density_kg_m3, V the volume of the union of the collision geometry;
  - sliding friction within friction;
  - smallest principal inertia >= inertia_min_ratio x M x L^2 (L the largest extent);
  - largest principal inertia <= (1 + inertia_max_tolerance) x M x d_max^2, the bound every mass distribution
    inside the collision geometry obeys (so no <compiler boundinertia>-style inflation can freeze a unit);
  - C inside the convex hull of the collision geometry (within com_tolerance_m);
  - the largest extent of the collision geometry at most max_item_extent_m.

Excluded units. A unit that fails step 7 scores 0 in every criterion, and it is DELETED from the model before any
physics: the G0 settle check, the standard settle and every probe run on physics_model(res), the grader copy without
the excluded units' bodies. An excluded unit can therefore neither support nor disturb another unit. It still counts
as a submitted unit in the inventory denominators.

Photo cameras. Each photo you registered is a fixed camera named photo_NNN (NNN = the number of site_NNN.jpg) on a
static body, with the photo's resolution and your estimate of its intrinsics (twin_camera.py). A photo camera that
breaks a camera rule is reported invalid and left out of the alignment; the gate fails when fewer than
min_photo_cameras valid photo cameras remain. The report carries a sanity line per photo camera: the angle between
the image's up direction and world +z (near 0 for a level shot, near 90 deg for a portrait shot), and the share of
your twin's collision geometry in front of the camera and inside its image.

G0's settle check (settle_check, blowup_check). The standard simulation starts from the photographed state (the gates
held, nothing else) with the quasi-static relaxation of the standard settle (twin_sim.relax: twin_sim_config.json
settle.relax_s of steps, every velocity set to zero after each step, so a penetration is pushed out slowly instead of
launching the unit), then runs blowup_window_s of free dynamics. The dynamic phase blows up on:
  - a non-finite state, or MuJoCo reporting bad accelerations;
  - mechanical-energy gain: KE(t) > PE(0) - PE(t) + eps, eps = 0.5 x (total unit mass) x v_tol^2 (the kinetic
    energy of every unit moving at v_tol), v_tol = blowup_speed_tol_m_s;
  - a unit whose centre of mass moves faster than the free-fall bound g t + v_tol.
Nothing falling, sliding or toppling from rest can break these bounds; the physics criteria charge such motion. A unit
that still breaks the speed bound after the relaxation is EXCLUDED (it scores 0, like a unit that fails a
plausibility rule) and the check is repeated once without it. G0 fails only on a non-finite state (in either phase),
or when the twin still blows up without the excluded units. What the relaxation cannot rest is the release of a deep
interpenetration between units, or a unit trapped inside the environment.

Errors. gate() never raises for a bad submission. It raises TaskError when the task's own inputs are broken (the
class vocabulary) and lets MemoryError through: both are task errors, never a verdict on the twin.
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

GATE_VERSION = "site-twin-gate/0.2.1"
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------------------------------------------
# public rules (the contract states the same numbers)

DEFAULT_RULES = {
    "max_entries": 5000,                     # files and directories in the whole submission folder
    "max_bytes": 300 * 1024 * 1024,          # bytes in the whole submission folder
    "max_scene_bytes": 16 * 1024 * 1024,
    "max_objects_bytes": 1024 * 1024,
    "max_mesh_file_bytes": 50 * 1024 * 1024,
    "max_png_file_bytes": 64 * 1024 * 1024,
    "max_png_side_px": 4096,
    "max_referenced_files": 5000,
    "max_mesh_assets": 8000,
    "max_textures": 500,
    "max_xml_elements": 100000,
    "max_xml_depth": 64,
    "max_item_hull_vertices": 50000,         # summed over a unit's collision meshes (more: the unit is excluded)
    "max_hull_vertices_total": 400000,       # summed over every collision mesh geom of the twin (more: G0 fails)
    "volume_grid_max_columns": 40000,        # work bound of the gate's union-volume integral
    "max_bodies": 2000,                      # bodies other than the world
    "max_geoms": 20000,
    "max_collision_geoms": 6000,             # environment included
    "max_collision_geoms_per_item": 64,      # convex pieces per unit (more: the unit is excluded)
    "max_collision_mesh_vertices": 20000,    # per mesh used by a collision geom
    "max_mesh_faces_total": 3000000,
    "max_mesh_vertices_total": 3000000,
    "items": [0, 150],                       # units listed in objects.json
    "photo_prefix": "photo_",                # photo cameras are photo_NNN, NNN the number of site_NNN.jpg
    "photo_numbers": [1, 44],
    "min_photo_cameras": 12,                 # valid photo cameras needed
    "photo_resolution": [6208, 4135],        # every photo's size, as stored
    "focal_px": [1000.0, 20000.0],
    "density_kg_m3": [10.0, 3000.0],
    "friction": [0.2, 1.5],
    "inertia_min_ratio": 1e-4,               # smallest principal inertia >= ratio * mass * L^2 (L: largest extent)
    "inertia_max_tolerance": 0.01,           # largest principal inertia <= (1 + tol) * mass * d_max^2
    "com_tolerance_m": 1e-6,
    "max_item_extent_m": 8.0,
    "max_abs_coordinate_m": 200.0,
    "max_geom_size_m": 100.0,
    "volume_grid_max_m": 0.02,               # column spacing of the union-volume integral (finer for small units)
    "volume_grid_min_m": 0.001,
    "volume_grid_max_piece_columns": 400000, # work bound: sum over pieces of the columns their boxes cover
    "max_column_depth": 128,                 # collision geoms whose bounding boxes reach over one grid square
    "column_depth_cell_m": 0.25,             # the squares of that count
    "blowup_window_s": 2.0,                  # G0's settle check: the dynamic phase after the relaxation runs this long
    "blowup_speed_tol_m_s": 2.0,             # v_tol of the free-fall and energy bounds
    "hull_trap_depth_m": 0.01,               # a body's centre of mass this deep inside a static shape: the hull note
    "sanity_front_m": 0.05,                  # camera sanity line: a point is in front when its camera z exceeds this
    # the gate travel rule (C5's gate term; never fails G0) is twin_sim.gate_rule with twin_sim_config.json
    # "gate_rule" (one source of its numbers); here only the rail search that suggests the opening direction for the
    # note: static geoms lying within rail_band_m of the leaf's top edge height, within rail_lateral_m of its mid-plane
    "gate_travel": {"rail_band_m": 0.25, "rail_lateral_m": 0.25, "rail_tie_m": 0.001},
}

FORMAT_OBJECTS = "site-twin-objects/1"
PHOTO_RX = re.compile(r"^photo_(\d{3})$")

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
    "model": "no attached models", "lengthrange": "not allowed",
    "hfield": "no height fields: a floor is planes, boxes or convex meshes",
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
    "strippath": "strippath is not allowed",
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
    "hfield": "no height fields: a floor is planes, boxes or convex meshes",
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
    """The task's own inputs are broken (e.g. the class vocabulary): never a verdict on the submission."""


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
        for e in self.report.get("units", []):
            if status is None or e.get("status") == status:
                out.append(e["body"])
        return out

    def physics_model(self):
        """The model every simulation runs: the grader copy with the excluded units deleted (see physics_model)."""
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
        col.fail("<joint type=%r> is not allowed (slide for a gate, free for a unit)" % v)
    elif tag == "geom" and k == "type" and v == "hfield":
        col.fail("<geom type='hfield'> is not allowed: no height fields (a floor is planes, boxes or convex meshes)")
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
    """Parse and check objects.json: {"format": "site-twin-objects/1", "items": [{"body", "label"}],
    "gates": [{"body", "joint"}]}. Returns the objects dict or None."""
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
        col.fail('objects.json must be an object {"format", "items", "gates"}')
        return None
    for k in obj:
        if k not in ("format", "items", "gates"):
            col.warn("objects.json: key %r is ignored" % k)
    if obj.get("format") != FORMAT_OBJECTS:
        col.fail('objects.json "format" must be %r' % FORMAT_OBJECTS)
    items = obj.get("items")
    lo, hi = rules["items"]
    if not isinstance(items, list):
        col.fail('objects.json "items" must be a list of {"body", "label"} (one per unit)')
        return None
    if not lo <= len(items) <= hi:
        col.fail("objects.json lists %d units (%d to %d allowed)" % (len(items), lo, hi))
    seen = set()
    for k, it in enumerate(items):
        if not isinstance(it, dict) or not isinstance(it.get("body"), str) or not isinstance(it.get("label"), str):
            col.fail('objects.json items[%d] must be {"body": "<body name>", "label": "<class name>"}' % k)
            continue
        for kk in it:
            if kk not in ("body", "label"):
                col.warn("objects.json items[%d]: key %r is ignored" % (k, kk))
        if not it["body"]:
            col.fail("objects.json items[%d] has an empty body name" % k)
        if it["body"] in seen:
            col.fail("objects.json lists body %r twice" % it["body"])
        seen.add(it["body"])
    gates = obj.get("gates")
    if not isinstance(gates, list):
        col.fail('objects.json "gates" must be a list of {"body", "joint"} (an empty list when you model no gate)')
        return obj
    gseen = set()
    for k, g in enumerate(gates):
        if not isinstance(g, dict) or not isinstance(g.get("body"), str) or not isinstance(g.get("joint"), str):
            col.fail('objects.json gates[%d] must be {"body": "<gate body>", "joint": "<its slide joint>"}' % k)
            continue
        for kk in g:
            if kk not in ("body", "joint"):
                col.warn("objects.json gates[%d]: key %r is ignored" % (k, kk))
        if not g["body"] or not g["joint"]:
            col.fail("objects.json gates[%d] has an empty body or joint name" % k)
        if g["body"] in seen:
            col.fail("the gate body %r is also listed as a unit" % g["body"])
        if g["body"] in gseen:
            col.fail("objects.json lists gate body %r twice" % g["body"])
        gseen.add(g["body"])
    return obj


_VOCAB_CACHE = {}


def load_vocabulary(vocabulary=None):
    """The class names: a list, a path to class_names.json ({"names": [...]}), or None for the file next to this
    module. A file is read and validated once per process; a missing or malformed vocabulary raises TaskError (the
    task's inputs are broken, not the submission)."""
    if vocabulary is None:
        vocabulary = os.path.join(HERE, "class_names.json")
    if isinstance(vocabulary, (list, tuple, set)):
        names = sorted(vocabulary) if isinstance(vocabulary, set) else [v for v in vocabulary]
        if not names or not all(isinstance(n, str) for n in names):
            raise TaskError("the class vocabulary must be a non-empty list of names")
        return list(names)
    key = os.path.abspath(str(vocabulary))
    if key in _VOCAB_CACHE:
        return list(_VOCAB_CACHE[key])
    try:
        with open(key, "rb") as f:
            d = json.loads(f.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise TaskError("the class vocabulary %s cannot be read: %s" % (os.path.basename(key), str(exc)[:120]))
    names = d.get("names") if isinstance(d, dict) else d
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise TaskError("the class vocabulary %s must be {\"names\": [...]}" % os.path.basename(key))
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
    if spec is not None:
        # single-threaded compile, whatever the twin's <compiler usethread>: MuJoCo's threaded mesh compile starts a
        # thread per mesh up to the host's hardware threads (not the CPUs this process may use), and under the
        # grader's address-space limit that aborts the process on a many-core host (measured: 513 meshes, 256
        # hardware threads, 6 GB); the compiled model is the same, and every copy of this spec inherits the setting
        spec.compiler.usethread = False
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
    """The bodies of the subtree rooted at root_body (itself included; never the world), in id order. MuJoCo numbers
    a parent before its children, so one pass over the bodies after the root decides membership (linear time: the
    gate calls this for every unit and gate)."""
    root = int(root_body)
    if root <= 0 or root >= int(model.nbody):
        return [] if root != 0 else list(range(1, int(model.nbody)))
    par = model.body_parentid
    inside = {root}
    out = [root]
    for b in range(root + 1, int(model.nbody)):
        if int(par[b]) in inside:
            inside.add(b)
            out.append(b)
    return out


def _geoms_by_body(model):
    """{body id: [geom ids]} of a compiled model (one pass over the geoms)."""
    out = {}
    for g, b in enumerate(np.asarray(model.geom_bodyid, int)):
        out.setdefault(int(b), []).append(g)
    return out


def _name(model, objtype, i):
    import mujoco
    nm = mujoco.mj_id2name(model, objtype, int(i))
    return nm if nm else "#%d" % int(i)


def photo_number(name):
    """The NNN of a photo camera named photo_NNN (an int), or None when the name is not of that form."""
    m = PHOTO_RX.match(str(name or ""))
    return int(m.group(1)) if m else None


def _twin_camera():
    try:
        import twin_camera
    except ImportError:  # pragma: no cover - twin_camera sits next to this file
        sys.path.insert(0, HERE)
        import twin_camera
    return twin_camera


def column_depth(model, data, geoms, cell_m):
    """The largest number of the given collision geoms whose world bounding boxes reach over one square of a grid of
    cell_m squares, and where: {"max", "at" (x, y of that square's centre), "cell_m", "n_geoms"}. Planes are left
    out (a ray finds a plane analytically). Linear in the number of geoms plus the squares of the twin's extent: the
    boxes are added to a difference array and summed once. The count bounds the number of geoms any vertical line
    meets (a box may reach over a square that its shape misses: the count is an upper bound)."""
    import mujoco
    m, d = model, data
    boxes = []
    for g in geoms:
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        X = _box_corners(m, d, [g])
        boxes.append((float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 1].min()), float(X[:, 1].max())))
    out = {"max": 0, "at": None, "cell_m": float(cell_m), "n_geoms": len(boxes)}
    if not boxes:
        return out
    out["cell_m"] = None                             # set below (the squares actually used)
    B = np.asarray(boxes, float)
    x0, y0 = float(B[:, 0].min()), float(B[:, 2].min())
    span = max(float(B[:, 1].max()) - x0, float(B[:, 3].max()) - y0)
    h = max(float(cell_m), span / 4096.0)            # at most 4096 squares a side (coarser only for a huge twin)
    i0 = np.floor((B[:, 0] - x0) / h).astype(np.int64)
    i1 = np.floor((B[:, 1] - x0) / h).astype(np.int64)
    j0 = np.floor((B[:, 2] - y0) / h).astype(np.int64)
    j1 = np.floor((B[:, 3] - y0) / h).astype(np.int64)
    nx, ny = int(i1.max()) + 2, int(j1.max()) + 2
    D = np.zeros((nx, ny), np.int32)
    np.add.at(D, (i0, j0), 1)
    np.add.at(D, (i1 + 1, j0), -1)
    np.add.at(D, (i0, j1 + 1), -1)
    np.add.at(D, (i1 + 1, j1 + 1), 1)
    C = np.cumsum(np.cumsum(D, axis=0), axis=1)
    k = int(np.argmax(C))
    i, j = divmod(k, ny)
    out.update({"max": int(C[i, j]), "at": [round(x0 + (i + 0.5) * h, 3), round(y0 + (j + 0.5) * h, 3)],
                "cell_m": h})
    return out


def _hull_nvert(model, mesh_id):
    adr = int(model.mesh_graphadr[mesh_id])
    return int(model.mesh_graph[adr]) if adr >= 0 else int(model.mesh_vertnum[mesh_id])


def _check_camera(m, cid, rules):
    """(record, reasons) of photo camera cid (the parts that need no MjData)."""
    import mujoco
    TC = _twin_camera()
    rec = {"name": _name(m, mujoco.mjtObj.mjOBJ_CAMERA, cid), "camera_id": int(cid)}
    why = []
    if int(m.cam_mode[cid]) != int(mujoco.mjtCamLight.mjCAMLIGHT_FIXED):
        why.append("mode must be fixed")
    if int(m.body_weldid[int(m.cam_bodyid[cid])]) != 0:
        why.append("it must sit on a static body (the world or a body welded to it)")
    W, H = (int(v) for v in m.cam_resolution[cid])
    if [W, H] != list(rules["photo_resolution"]):
        why.append("resolution is %d x %d (the photos are %d x %d)" % (W, H, *rules["photo_resolution"]))
    try:
        K, W2, H2 = TC.decode(m, cid)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        rec.update({"fx": fx, "fy": fy, "cx": cx, "cy": cy, "W": W2, "H": H2})
        lo, hi = rules["focal_px"]
        if not (lo <= fx <= hi and lo <= fy <= hi):
            why.append("focal lengths fx %.2f, fy %.2f px are outside [%g, %g]" % (fx, fy, lo, hi))
        if not (-0.5 <= cx <= W2 - 0.5 and -0.5 <= cy <= H2 - 0.5):
            why.append("principal point (%.2f, %.2f) lies outside the image" % (cx, cy))
    except Exception as exc:  # noqa: BLE001 - twin_camera.CameraError
        why.append("intrinsics cannot be decoded: %s" % exc)
    return rec, why


def compiled_checks(model, spec, objects, vocabulary, col, rules):
    """Twin-level checks on the compiled grader copy; returns (camera records, unit records, gate records)."""
    import mujoco
    m = model
    zero = ["neq", "ntendon", "nu", "nflex", "nplugin", "nmocap", "nexclude", "npair", "nkey", "nsensor",
            "nskin", "nnumeric", "ntext", "ntuple", "nhfield"]
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
        col.fail("%d collision geoms (cap %d, the environment included)" % (len(colg), rules["max_collision_geoms"]))
    if int(m.nmeshface) > rules["max_mesh_faces_total"]:
        col.fail("%d mesh faces in total (cap %d)" % (int(m.nmeshface), rules["max_mesh_faces_total"]))
    T = mujoco.mjtGeom
    ok_types = {int(T.mjGEOM_PLANE), int(T.mjGEOM_BOX), int(T.mjGEOM_SPHERE), int(T.mjGEOM_CAPSULE),
                int(T.mjGEOM_CYLINDER), int(T.mjGEOM_ELLIPSOID), int(T.mjGEOM_MESH)}
    hull_total = 0
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
            else:
                hull_total += _hull_nvert(m, mid)
    if hull_total > rules["max_hull_vertices_total"]:
        col.fail("the collision meshes hold %d hull vertices in total (cap %d)" % (hull_total,
                                                                                  rules["max_hull_vertices_total"]))
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
    if m.ngeom and float(np.max(np.abs(m.geom_pos))) > big:
        col.fail("a geom lies more than %g m from its body" % big)
    if m.ngeom and float(np.max(np.abs(m.geom_size))) > rules["max_geom_size_m"]:
        col.fail("a geom size exceeds %g m" % rules["max_geom_size_m"])
    if int(m.nmeshvert) and float(np.max(np.abs(m.mesh_vert))) > rules["max_geom_size_m"]:
        col.fail("a mesh vertex lies more than %g m from its mesh's centre" % rules["max_geom_size_m"])
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
    # the photo cameras
    cams = []
    lo_n, hi_n = rules["photo_numbers"]
    for cid in range(m.ncam):
        nm = _name(m, mujoco.mjtObj.mjOBJ_CAMERA, cid)
        num = photo_number(nm)
        if num is None:
            if nm.startswith(rules["photo_prefix"]):
                col.fail("camera %s: photo cameras are named photo_NNN with three digits (photo_007 for site_007.jpg)"
                         % nm)
            else:
                col.warn("camera %s is not a photo camera (photo_NNN): ignored" % nm)
            continue
        if not lo_n <= num <= hi_n:
            col.fail("camera %s names no given photo (the photos are site_%03d .. site_%03d)" % (nm, lo_n, hi_n))
            continue
        rec, why = _check_camera(m, cid, rules)
        rec["photo"] = num
        rec["status"] = "invalid" if why else "ok"
        rec["reasons"] = why
        if why:
            col.warn("photo camera %s is invalid and left out of the alignment: %s" % (nm, "; ".join(why)))
        cams.append(rec)
    cams.sort(key=lambda r: r["photo"])
    n_valid = sum(1 for r in cams if r["status"] == "ok")
    if n_valid < rules["min_photo_cameras"]:
        col.fail("%d valid photo cameras (at least %d are needed; %d photo cameras in the twin)" % (
            n_valid, rules["min_photo_cameras"], len(cams)))
    # bodies, joints: the units
    names = {_name(m, mujoco.mjtObj.mjOBJ_BODY, b): b for b in range(m.nbody)}
    by_body = _geoms_by_body(m)
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
    jointed = set(int(v) for v in np.nonzero(np.asarray(m.body_jntnum) > 0)[0])
    for b in sorted(free_bodies):
        if int(m.body_jntnum[b]) != 1:
            col.fail("unit body %s carries %d joints (only its free joint)" % (_name(m, mujoco.mjtObj.mjOBJ_BODY, b),
                                                                               int(m.body_jntnum[b])))
        nm = _name(m, mujoco.mjtObj.mjOBJ_BODY, b)
        if nm not in listed:
            col.fail("free body %s is not listed in objects.json (every loose unit is an item)" % nm)
        for c in _subtree(m, b):
            if c != b and c in jointed:
                col.fail("body %s inside unit %s carries a joint" % (_name(m, mujoco.mjtObj.mjOBJ_BODY, c), nm))
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
            rec["reasons"].append("label %r is not in the class vocabulary (it never counts as correct)" % label)
        items.append(rec)
    # the gates
    gates = []
    gate_bodies = set()
    for g in (objects or {}).get("gates", []) if isinstance((objects or {}).get("gates"), list) else []:
        if not isinstance(g, dict) or not isinstance(g.get("body"), str) or not isinstance(g.get("joint"), str):
            continue
        rec = {"body": g["body"], "joint": g["joint"]}
        b = names.get(g["body"])
        if b is None or b == 0:
            col.fail("the gate body %r named in objects.json is not a body of scene.xml" % g["body"])
            gates.append(rec)
            continue
        gate_bodies.add(b)
        ja = int(m.body_jntadr[b])
        if int(m.body_jntnum[b]) != 1 or int(m.jnt_type[ja]) != int(mujoco.mjtJoint.mjJNT_SLIDE):
            col.fail("the gate body %s must carry exactly one slide joint" % g["body"])
        elif _name(m, mujoco.mjtObj.mjOBJ_JOINT, ja) != g["joint"]:
            col.fail("the slide joint of gate %s is %r, not %r as objects.json says" % (
                g["body"], _name(m, mujoco.mjtObj.mjOBJ_JOINT, ja), g["joint"]))
        else:
            rec["body_id"], rec["joint_id"] = b, ja
        if int(m.body_parentid[b]) != 0 and int(m.body_weldid[int(m.body_parentid[b])]) != 0:
            col.fail("the gate body %s must hang from a static body" % g["body"])
        sub = set(_subtree(m, b))
        for c in sorted(sub):
            if c != b and c in jointed:
                col.fail("body %s inside gate %s carries a joint" % (_name(m, mujoco.mjtObj.mjOBJ_BODY, c), g["body"]))
        gcol = [k for c in sorted(sub) for k in by_body.get(c, []) if is_collision_geom(m, k)]
        if not gcol:
            col.fail("the gate body %s has no collision geometry" % g["body"])
        for k in gcol:
            if int(m.geom_type[k]) == int(T.mjGEOM_PLANE):
                col.fail("gate %s: a plane cannot be part of a gate" % g["body"])
        gates.append(rec)
    for j in slide:
        if int(m.jnt_bodyid[j]) not in gate_bodies:
            col.fail("slide joint %s (body %s) is not the joint of a gate listed in objects.json" % (
                _name(m, mujoco.mjtObj.mjOBJ_JOINT, j), _name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j]))))
    return cams, items, gates


def _box_corners(model, d, geoms):
    """World corners of the bounding boxes of the given geoms, at d's state ((8 n, 3))."""
    P = []
    for g in geoms:
        c, h = np.asarray(model.geom_aabb[g][:3], float), np.asarray(model.geom_aabb[g][3:], float)
        R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    P.append(np.asarray(d.geom_xpos[g], float) + R @ (c + h * np.array([sx, sy, sz])))
    return np.asarray(P, float).reshape(-1, 3)


def static_geoms(model, exclude_bodies=()):
    """Collision geoms of the environment: bodies welded to the world, outside the given unit / gate bodies."""
    ex = set()
    for b in exclude_bodies:
        ex.update(_subtree(model, int(b)))
        ex.add(int(b))
    return [g for g in range(model.ngeom) if is_collision_geom(model, g) and int(model.geom_bodyid[g]) not in ex
            and int(model.body_weldid[int(model.geom_bodyid[g])]) == 0]


def gate_travel(model, d, body, joint, rules, static=None):
    """The gate travel rule (C5's gate term) measured on the twin at d's state, in the twin's own units: the rule is
    twin_sim.gate_rule with twin_sim_config.json "gate_rule" (the one place its numbers live), evaluated here for the
    opening direction the twin's own rail suggests (the grader takes the direction from the real gate instead).
    Width: the extent along the slide axis of the leaf's collision geometry (its geoms' bounding boxes). Suggested
    opening direction: the side along the axis on which the top rail continues beyond the leaf. Rail = the static
    collision geoms whose bounding boxes lie entirely within rail_band_m of the height of the leaf's top edge, come
    within rail_lateral_m of the leaf's vertical mid-plane and overlap the leaf along the axis; ext_plus / ext_minus =
    how far they reach beyond the leaf's +axis / -axis end; the side reaching farther by more than rail_tie_m opens,
    else the direction is undetermined (the note says so). Measured from the leaf's collision geometry, never from the
    body origin. "fails_either_way" lists the reasons that hold in both directions (no limits, the photographed
    position outside the range, the travel against the leaf's width), "fails_this_way" those of the open and closing
    lengths in the rail's direction (not judged without one: "direction_from_rail" false)."""
    import mujoco
    m = model
    R = rules["gate_travel"]
    axis = np.asarray(d.xaxis[joint], float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    sub = set(_subtree(m, body)) | {int(body)}
    leaf = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in sub and is_collision_geom(m, g)]
    X = _box_corners(m, d, leaf)
    reasons = []
    if not len(X):
        return {"ok": False, "reasons": ["the gate has no collision geometry"], "opening_sign": 0}
    a_lo, a_hi = float(np.min(X @ axis)), float(np.max(X @ axis))
    width = a_hi - a_lo
    ctr = 0.5 * (X.min(axis=0) + X.max(axis=0))
    up = np.array([0.0, 0.0, 1.0])
    lat = np.cross(up, axis)
    if float(np.linalg.norm(lat)) < 1e-9:
        lat = np.array([1.0, 0.0, 0.0])
    lat = lat / float(np.linalg.norm(lat))
    top = float(np.max(X[:, 2]))
    if static is None:
        static = static_geoms(m, [int(body)])
    ext_p, ext_m, cands = -np.inf, -np.inf, []
    for g in static:
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue                                      # no bounded box
        Y = _box_corners(m, d, [g])
        zlo, zhi = float(Y[:, 2].min()), float(Y[:, 2].max())
        if zlo < top - R["rail_band_m"] or zhi > top + R["rail_band_m"]:
            continue                                      # a top rail lies entirely around the leaf's top edge
        lv = (Y - ctr) @ lat
        if float(lv.min()) > R["rail_lateral_m"] or float(lv.max()) < -R["rail_lateral_m"]:
            continue
        av = Y @ axis
        if float(av.max()) < a_lo or float(av.min()) > a_hi:
            continue                                      # beside the leaf, not along it
        cands.append(int(g))
        ext_p = max(ext_p, float(av.max()) - a_hi)
        ext_m = max(ext_m, a_lo - float(av.min()))
    sign = 0
    if cands:
        if ext_p > ext_m + R["rail_tie_m"]:
            sign = 1
        elif ext_m > ext_p + R["rail_tie_m"]:
            sign = -1
    limited = bool(m.jnt_limited[joint])
    lo, hi = (float(v) for v in m.jnt_range[joint])
    q0 = float(m.qpos0[int(m.jnt_qposadr[joint])])
    travel = hi - lo
    meas = {"limited": limited, "range_m": [round(lo, 6), round(hi, 6)] if limited else None, "qpos0_m": round(q0, 6),
            "width_m": round(width, 6)}
    direction = None
    if not cands:
        direction = ("no rail found: no static geometry runs along the gate's top edge (model the rail it hangs "
                     "from, continuing on the side the gate opens to)")
    elif sign == 0:
        direction = "the rail reaches equally far beyond both ends of the leaf"
    if direction:
        reasons.append(direction)
    TS = _twin_sim()
    rr = TS.gate_rule(meas, sign if sign != 0 else 1)
    either, this_way = [], []
    for r in rr["reasons"]:
        if r.startswith(("the gate opens only", "the gate travels")):
            if sign != 0:                                 # without a direction, the open and closing lengths are
                this_way.append(r)                        # not judged
        else:
            either.append(r)
    reasons += either + this_way
    out = {"ok": not reasons, "reasons": reasons, "direction_from_rail": direction is None,
           "fails_either_way": either, "fails_this_way": this_way}
    out.update(meas)
    out.update({"travel_m": round(travel, 6) if limited else None,
                "travel_over_width": round(travel / width, 4) if (limited and width > 0) else None,
                "opening_sign": int(sign), "open_m": (round(float(rr["open_m"]), 6) if (limited and sign != 0
                                                                                        and rr["open_m"] is not None)
                                                      else None),
                "close_m": (round(float(rr["close_m"]), 6) if (limited and sign != 0 and rr.get("close_m") is not None)
                            else None),
                "leaf_centre": [round(float(v), 6) for v in ctr], "axis": [round(float(v), 9) for v in axis],
                "rail": {"geoms": [_name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in cands],
                         "ext_plus_m": round(ext_p, 6) if np.isfinite(ext_p) else None,
                         "ext_minus_m": round(ext_m, 6) if np.isfinite(ext_m) else None}})
    return out


def geom_points(model, d, geoms):
    """The sanity-line sample of the twin's geometry: per geom the 8 corners and the centre of its bounding box
    (planes excepted: their boxes are not bounded by what they show)."""
    import mujoco
    out = []
    for g in geoms:
        if int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        c = np.asarray(model.geom_aabb[g][:3], float)
        R = np.asarray(d.geom_xmat[g], float).reshape(3, 3)
        out.append(np.asarray(d.geom_xpos[g], float) + R @ c)
        out.extend(list(_box_corners(model, d, [g])))
    return np.asarray(out, float).reshape(-1, 3)


def camera_sanity(model, data, cam_id, K, W, H, points, rules=None):
    """The sanity line of one photo camera: the angle between the image's up direction (-y of the camera) and world
    +z (degrees: about 0 for a level landscape shot, 90 for a portrait shot), and the shares of `points` (the twin's
    geometry, geom_points) in front of the camera and inside its image."""
    R = dict(DEFAULT_RULES, **(rules or {}))
    TC = _twin_camera()
    Rcv, p = TC.camera_frame(model, data, cam_id)
    up = -Rcv[:, 1]
    ang = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
    out = {"up_angle_deg": round(ang, 3), "optical_axis_elevation_deg":
           round(math.degrees(math.asin(max(-1.0, min(1.0, float(Rcv[2, 2]))))), 3)}
    P = np.asarray(points, float).reshape(-1, 3)
    if not len(P):
        out.update({"in_front_share": 0.0, "in_view_share": 0.0, "n_points": 0})
        return out
    Xc = (P - p) @ Rcv
    z = Xc[:, 2]
    front = z > float(R["sanity_front_m"])
    with np.errstate(divide="ignore", invalid="ignore"):
        u = K[0, 0] * Xc[:, 0] / z + K[0, 2]
        v = K[1, 1] * Xc[:, 1] / z + K[1, 2]
    view = front & (u >= -0.5) & (u <= W - 0.5) & (v >= -0.5) & (v <= H - 0.5)
    out.update({"in_front_share": round(float(np.mean(front)), 4), "in_view_share": round(float(np.mean(view)), 4),
                "n_points": int(len(P))})
    return out


def hull_traps(model, d, static, points):
    """Notes for static collision geoms whose collision shape contains one of `points` ({label: (owner body, point
    (3,), depth)}) deeper than its depth: a mesh as its convex hull, a primitive as itself, a plane as its half-space.
    Returns a list of (geom name, label)."""
    import mujoco
    m = model
    out = []
    labels = sorted(points)
    P = np.asarray([points[k][1] for k in labels], float).reshape(-1, 3)
    owner = [points[k][0] for k in labels]
    depth = np.asarray([float(points[k][2]) if len(points[k]) > 2 else 1e-6 for k in labels], float)
    if not len(P):
        return out
    for g in static:
        t = int(m.geom_type[g])
        if t == int(mujoco.mjtGeom.mjGEOM_PLANE):
            n = np.asarray(d.geom_xmat[g], float).reshape(3, 3)[:, 2]
            inside = (P - np.asarray(d.geom_xpos[g], float)) @ n < -depth
        else:
            pc = geom_piece(m, d, g)
            if pc is None:
                continue
            lo, hi = pc.bbox()
            near = np.all((P >= lo) & (P <= hi), axis=1)
            inside = np.zeros(len(P), bool)
            for dep in sorted(set(depth[near].tolist())):
                sel = near & (depth == dep)
                inside[sel] = pc.inside(P[sel], tol=-dep)
        gb = int(m.geom_bodyid[g])
        for k in np.nonzero(inside)[0]:
            if owner[k] == gb:
                continue
            out.append((_name(m, mujoco.mjtObj.mjOBJ_GEOM, g), labels[k]))
    return out


def is_collision_geom(model, g):
    """A geom collides when (contype, conaffinity) != (0, 0) (the same rule as the standard simulation's)."""
    return int(model.geom_contype[g]) != 0 or int(model.geom_conaffinity[g]) != 0


def item_checks(model, spec, rec, rules, data=None, by_body=None, spec_geoms=None):
    """Per-unit contract and plausibility checks; fills rec (status 'ok' or 'excluded' with reasons). by_body
    ({body: [geoms]}, _geoms_by_body) and spec_geoms ({geom id: MjsGeom}) may be passed in to share them between
    the units (the gate does)."""
    import mujoco
    m = model
    b = rec["body_id"]
    d = data
    if d is None:
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
    bodies = _subtree(m, b)
    by_body = _geoms_by_body(m) if by_body is None else by_body
    geoms = sorted(g for c in bodies for g in by_body.get(c, []))
    coll, vis, why = [], [], []
    if spec_geoms is not None:
        sgeoms = spec_geoms
    else:
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
        why.append("its collision meshes hold %d vertices (cap %d per unit)" % (hv, rules["max_item_hull_vertices"]))
        rec["reasons"] = rec.get("reasons", []) + why
        rec["status"] = "excluded"
        rec.pop("body_id", None)
        return rec
    if not coll:
        why.append("no collision geom (a geom with contype or conaffinity non-zero)")
    if len(coll) > rules["max_collision_geoms_per_item"]:
        why.append("%d collision geoms (cap %d convex pieces per unit)" % (len(coll), rules["max_collision_geoms_per_item"]))
    pieces = []
    for g in coll:
        if int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            why.append("a plane cannot be part of a unit")
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
    # composite mass, centre of mass and inertia of the unit's bodies at qpos0
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
            why.append("the unit is larger than %g m (its collision geometry spans %.2f m)" % (
                rules["max_item_extent_m"], float(np.max(hi_b - lo_b))))
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
    rep = {"format": "site-twin-gate-report/1", "gate_version": GATE_VERSION, "submission": "", "pass": False,
           "reasons": col.reasons, "warnings": col.warnings, "units": [], "excluded": [], "gates": [],
           "cameras": [], "n_photo_cameras_valid": 0, "stats": {}}
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
    cams, items, gates = compiled_checks(model, spec, objects, vocab, col, R)
    rep["n_photo_cameras_valid"] = sum(1 for c in cams if c["status"] == "ok")
    if col.reasons:
        rep["cameras"] = cams
        rep["units"] = [{k: v for k, v in it.items() if k != "body_id"} for it in items]
        rep["gates"] = [{k: v for k, v in g.items() if k not in ("body_id", "joint_id")} for g in gates]
        return res
    import mujoco
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    # the column depth: the vertical ray casts of every simulation and walk probe stay bounded
    colg = [g for g in range(model.ngeom) if is_collision_geom(model, g)]
    cd = column_depth(model, data, colg, R["column_depth_cell_m"])
    rep["stats"]["column_depth"] = cd
    if cd["max"] > R["max_column_depth"]:
        col.fail("%d collision geoms reach over one %.2f m square at (%.2f, %.2f) (at most %d may stack along a "
                 "vertical line: merge thin layers or stacked pieces)" % (cd["max"], cd["cell_m"], cd["at"][0],
                                                                         cd["at"][1], R["max_column_depth"]))
        rep["cameras"] = cams
        rep["units"] = [{k: v for k, v in it.items() if k != "body_id"} for it in items]
        rep["gates"] = [{k: v for k, v in g.items() if k not in ("body_id", "joint_id")} for g in gates]
        return res
    # the gates' travel rule (notes only; the assessment takes the opening direction from the site, never from the
    # twin's rail: the rail only suggests the direction in which the note judges the open and closing lengths)
    unit_bodies = [it["body_id"] for it in items if "body_id" in it]
    gate_ids = [g["body_id"] for g in gates]
    static = static_geoms(model, unit_bodies + gate_ids)
    for g in gates:
        tr = g["travel"] = gate_travel(model, data, g["body_id"], g["joint_id"], R, static)
        if tr["ok"] or "fails_either_way" not in tr:
            if not tr["ok"]:
                col.warn("gate %s: travel rule not met in either direction (C5's gate term scores 0 for it): %s" % (
                    g["body"], "; ".join(tr["reasons"])))
            continue
        if tr["fails_either_way"]:
            col.warn("gate %s: travel rule not met in either direction (C5's gate term scores 0 for it): %s" % (
                g["body"], "; ".join(tr["fails_either_way"])))
        if not tr["direction_from_rail"]:
            col.warn("gate %s: the opening direction cannot be read from your twin's rail (%s); the assessment takes "
                     "it from the site, not from your rail: check the travel rule for both directions of the axis "
                     "(check_twin.py prints them)" % (g["body"], tr["reasons"][0]))
        elif tr["fails_this_way"]:
            col.warn("gate %s: travel rule not met if it opens toward %saxis, the side on which your twin's rail "
                     "continues (C5's gate term scores 0 for it if the real gate opens that way; the assessment takes "
                     "the direction from the site): %s" % (g["body"], "+" if tr["opening_sign"] > 0 else "-",
                                                         "; ".join(tr["fails_this_way"])))
    rep["gates"] = [{k: v for k, v in g.items() if k not in ("body_id", "joint_id")} for g in gates]
    # the photo cameras: pose and sanity line
    TC = _twin_camera()
    pts = geom_points(model, data, [g for g in range(model.ngeom) if is_collision_geom(model, g)])
    for c in cams:
        Rcv, p = TC.camera_frame(model, data, c["camera_id"])
        c["centre"] = [float(v) for v in p]
        c["R_cv"] = [[float(v) for v in row] for row in Rcv]
        if c["status"] == "ok":
            K = TC.K_matrix(c["fx"], c["fy"], c["cx"], c["cy"])
            c["sanity"] = camera_sanity(model, data, c["camera_id"], K, c["W"], c["H"], pts, R)
    rep["cameras"] = cams
    # the hull trap: a static collision shape that contains a camera centre, or deeper than hull_trap_depth_m another
    # body's centre of mass (a unit resting on a floor tile has its centre of mass above it, never inside)
    probe = {"camera " + c["name"]: (-1, np.asarray(c["centre"], float), 1e-6) for c in cams}
    with_coll = set(int(model.geom_bodyid[g]) for g in colg)
    unit_roots = set(int(it["body_id"]) for it in items if "body_id" in it)
    root_has_coll = set(int(model.body_rootid[b]) for b in with_coll)
    trap_depth = float(R["hull_trap_depth_m"])
    for b in range(1, model.nbody):
        root = int(model.body_rootid[b])
        if root in unit_roots:
            if b != root or root not in root_has_coll:
                continue                                  # a unit is probed once, by its whole centre of mass
            pt = np.asarray(data.subtree_com[b], float)
        elif b in with_coll:
            pt = np.asarray(data.xipos[b], float)
        else:
            continue                                      # a body without collision geometry is no evidence
        probe["body " + _name(model, mujoco.mjtObj.mjOBJ_BODY, b)] = (b, pt, trap_depth)
    traps = hull_traps(model, data, static, probe)
    rep["stats"]["hull_traps"] = len(traps)
    for gname, what in traps[:12]:
        col.warn("static geom %s collides as a solid convex shape that contains the %s: a room or a large mesh "
                 "exported as one geom collides as a solid block (split it into convex pieces)" % (gname, what))
    if len(traps) > 12:
        col.warn("%d more hull-trap notes" % (len(traps) - 12))
    out = []
    by_body = _geoms_by_body(model)
    spec_geoms = {int(gs.id): gs for gs in spec.geoms} if spec is not None else {}
    for rec in items:
        try:
            out.append(item_checks(model, spec, rec, R, data, by_body=by_body, spec_geoms=spec_geoms))
        except MemoryError:
            raise                                         # a task error (the grader's resources), never an exclusion
        except Exception as exc:  # noqa: BLE001 - a broken unit must not crash the gate
            rec.pop("body_id", None)
            rec["status"] = "excluded"
            rec["reasons"] = rec.get("reasons", []) + ["the unit could not be checked: %s" % str(exc)[:200]]
            out.append(rec)
    rep["units"] = out
    rep["excluded"] = [r["body"] for r in out if r["status"] == "excluded"]
    rep["pass"] = not col.reasons
    return res


# ---------------------------------------------------------------------------------------------------------------
# physics: the model the simulations run, and G0's settle check

def physics_model(res, also_exclude=()):
    """The compiled model every simulation of the twin runs (G0's settle check, the standard settle, the support truth,
    the probes): the gate's grader copy with the bodies of the excluded items deleted (and of also_exclude), so an
    excluded item can neither support nor disturb another item. A fresh model on every call (callers standardise it in
    place); None when the gate produced no model."""
    import copy as _copy
    if res.model is None or res.spec is None:
        return None
    gone = sorted(set(res.report.get("excluded") or []) | set(also_exclude or ()))
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


def blowup_check(model, gates=None, rules=None, sim=None):
    """One pass of G0's settle check on a physics model (physics_model(res)): the standard simulation
    (twin_sim.standardize), the gates held still and nothing else held: first the quasi-static relaxation of the
    standard settle (twin_sim.relax, twin_sim_config.json settle.relax_s; a non-finite state there fails), then
    blowup_window_s of free dynamics from the relaxed state, which blows up on a non-finite state or MuJoCo's
    bad-acceleration warning, a gain of mechanical energy KE(t) + PE(t) - PE(0) above eps = 0.5 x (total unit mass) x
    v_tol^2, or a unit whose centre of mass moves faster than g t + v_tol (v_tol = blowup_speed_tol_m_s, t from the
    start of the dynamic phase). Energies are MuJoCo's (gravity potential, kinetic energy of every dof; the held gates
    contribute a constant), sampled at every step. The pass stops at the first step that blows up; "speed_units" then
    lists every unit above the speed bound at that step and "non_finite" says whether the state went non-finite.
    Returns a JSON-ready dict with "fail" and "reasons" (settle_check decides G0 from one or two passes)."""
    import copy as _copy
    import mujoco
    R = dict(DEFAULT_RULES, **(rules or {}))
    out = {"ran": False, "fail": False, "reasons": [], "speed_units": [], "non_finite": False}
    if model is None:
        out["reasons"].append("no model")
        out["fail"] = True
        return out
    TS = sim or _twin_sim()
    m = _copy.deepcopy(model)
    try:
        roles = TS.standardize(m, gates=gates)
    except Exception as exc:  # noqa: BLE001 - TwinSimError, a gate that is not one
        out.update({"fail": True, "reasons": ["the standard simulation does not accept the twin: %s" % str(exc)[:200]]})
        return out
    m.opt.enableflags |= int(mujoco.mjtEnableBit.mjENBL_ENERGY)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qvel[:] = 0.0
    items = np.asarray(roles.items, int)
    names = [roles.name(b) for b in items]
    rx = TS.relax(m, roles, d, config=roles.config)
    out["relax_s"] = float(rx["steps"]) * float(m.opt.timestep)
    if not rx["finite"]:
        out.update({"ran": True, "fail": True, "non_finite": True, "fail_phase": "relax",
                    "reasons": ["the state becomes non-finite during the relaxation"]})
        return out
    mass = np.asarray(m.body_subtreemass[items], float) if len(items) else np.zeros(0)
    vtol = float(R["blowup_speed_tol_m_s"])
    g = float(np.linalg.norm(m.opt.gravity))
    eps = 0.5 * float(mass.sum()) * vtol * vtol + 1e-9
    dt = float(m.opt.timestep)
    n = int(round(float(R["blowup_window_s"]) / dt))
    held = [(roles.gate_qadr[b], roles.gate_dadr[b], float(d.qpos[roles.gate_qadr[b]])) for b in roles.gates]
    pe0 = None
    gmax, gmax_t = -np.inf, 0.0
    vmax, vmax_t, vmax_item = -np.inf, 0.0, None
    bad = int(mujoco.mjtWarning.mjWARN_BADQACC)
    nbad0 = int(d.warning[bad].number)                # warnings of the relaxation do not count
    fail_t = None
    for k in range(n + 1):
        for qa, da, q0 in held:
            d.qpos[qa] = q0
            d.qvel[da] = 0.0
        if k < n:
            mujoco.mj_step(m, d)          # its forward pass evaluates the state at time k dt (energy, velocities)
        else:
            mujoco.mj_forward(m, d)
        t = k * dt
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel)) and np.all(np.isfinite(d.energy))):
            out["reasons"].append("the state becomes non-finite at t = %.3f s of the dynamic phase" % t)
            out["non_finite"] = True
            fail_t = t
            break
        blown = False
        if len(items):
            mujoco.mj_subtreeVel(m, d)
            v = np.linalg.norm(d.subtree_linvel[items], axis=1)
            ex = v - (g * t + vtol)
            a = int(np.argmax(ex))
            if ex[a] > vmax:
                vmax, vmax_t, vmax_item = float(ex[a]), t, names[a]
            over = np.nonzero(ex > 0)[0]
            if len(over):
                out["speed_units"] = sorted(names[i] for i in over)
                out["speed_m_s"] = {names[i]: round(float(v[i]), 3) for i in over}
                out["reasons"].append("unit%s %s move%s faster than the free-fall bound g t + %.1f = %.2f m/s at t = "
                                      "%.3f s of the dynamic phase (%s: %.2f m/s; a deep interpenetration released)" % (
                                          "s" if len(over) > 1 else "", ", ".join(out["speed_units"][:6]),
                                          "" if len(over) > 1 else "s", vtol, g * t + vtol, t, names[a], float(v[a])))
                blown = True
        if int(d.warning[bad].number) > nbad0:
            out["reasons"].append("MuJoCo reports bad accelerations (a numerical blow-up) at t = %.3f s of the "
                                  "dynamic phase" % t)
            blown = True
        pe, ke = float(d.energy[0]), float(d.energy[1])
        if pe0 is None:
            pe0 = pe
        gain = ke + pe - pe0
        if gain > gmax:
            gmax, gmax_t = gain, t
        if gain > eps:
            out["reasons"].append("the mechanical energy grows by %.4g J beyond the potential energy released at t = "
                                  "%.3f s of the dynamic phase (bound %.4g J = 0.5 x %.4g kg x (%.1f m/s)^2)" % (
                                      gain, t, eps, mass.sum(), vtol))
            blown = True
        if blown:
            fail_t = t
            break
    warn = {str(mujoco.mjtWarning(i)).split(".")[-1]: int(d.warning[i].number) for i in range(len(d.warning))
            if d.warning[i].number}
    out.update({"ran": True, "fail": bool(out["reasons"]), "window_s": n * dt, "fail_time_s": fail_t,
                "fail_phase": "dynamic" if out["reasons"] else None,
                "n_items": int(len(items)), "item_mass_kg": float(mass.sum()), "energy_bound_J": eps,
                "max_energy_gain_J": float(gmax) if np.isfinite(gmax) else None, "max_energy_gain_time_s": gmax_t,
                "max_speed_excess_m_s": float(vmax) if np.isfinite(vmax) else None, "max_speed_excess_time_s": vmax_t,
                "max_speed_excess_item": vmax_item, "speed_tol_m_s": vtol, "warnings": warn})
    return out


SETTLE_KEYS = ("fail", "reasons", "max_energy_gain_J", "energy_bound_J", "max_speed_excess_m_s",
               "max_speed_excess_item", "fail_time_s", "fail_phase", "relax_s", "window_s")


def settle_check(res, rules=None, sim=None):
    """G0's settle check of a gate result (the relaxation first, then the dynamic window): a blowup_check pass on
    physics_model(res); when it blows up because units break the speed bound (after the relaxation), those units are
    EXCLUDED (like a unit that fails a plausibility rule: they score 0 and are deleted from every simulation) and the
    check is repeated once without them. G0 fails on a non-finite state, on a blow-up no unit is to blame for (energy
    or bad accelerations alone), or when the repeated pass still blows up. Returns {"ran", "fail", "reasons",
    "excluded" [unit bodies], "exclusion_reasons" {unit: text}, "passes" [pass dicts], and the deciding pass's
    SETTLE_KEYS}."""
    gates = [g["body"] for g in res.report.get("gates", [])]
    first = blowup_check(physics_model(res), gates=gates, rules=rules, sim=sim)
    out = {"ran": bool(first.get("ran")), "passes": [first], "excluded": [], "exclusion_reasons": {}}
    deciding = first
    if first["fail"] and first.get("speed_units") and not first.get("non_finite"):
        ex = list(first["speed_units"])
        for u in ex:
            out["exclusion_reasons"][u] = ("it moves at %.2f m/s at t = %.3f s of G0's settle check, above the free-"
                                           "fall bound, after the relaxation (a deep interpenetration released): "
                                           "excluded" % (float(first.get("speed_m_s", {}).get(u, float("nan"))),
                                                         float(first.get("fail_time_s") or 0.0)))
        second = blowup_check(physics_model(res, also_exclude=ex), gates=gates, rules=rules, sim=sim)
        out["passes"].append(second)
        out["excluded"] = ex
        deciding = second
        if second["fail"]:
            second = dict(second, reasons=["after excluding %s: %s" % (", ".join(ex[:6]), r)
                                           for r in second["reasons"]])
            deciding = second
    for k in SETTLE_KEYS:
        out[k] = deciding.get(k)
    out["fail"] = bool(deciding["fail"])
    out["reasons"] = list(deciding.get("reasons") or [])
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="The validity gate (G0) of a site-twin submission.")
    ap.add_argument("submission", nargs="?", default="/home/user/submission")
    ap.add_argument("--vocab", default=None, help="class_names.json (default: the file next to this script)")
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
        rep["settle_check"] = settle_check(res)
        if rep["settle_check"]["fail"]:
            rep["pass"] = False
            rep["reasons"].extend("settle check: " + r for r in rep["settle_check"]["reasons"])
        for u in rep["settle_check"]["excluded"]:
            rep["warnings"].append("unit %s: %s" % (u, rep["settle_check"]["exclusion_reasons"][u]))
    if a.json:
        print(json.dumps(rep, indent=1))
    else:
        print("G0 %s: %s" % ("PASS" if rep["pass"] else "FAIL", a.submission))
        for r in rep["reasons"]:
            print("  - " + r)
        for w in rep["warnings"]:
            print("  (note) " + w)
        print("  photo cameras: %d valid of %d" % (rep.get("n_photo_cameras_valid", 0), len(rep.get("cameras", []))))
        for c in rep.get("cameras", []):
            if c.get("status") == "ok" and c.get("sanity"):
                sn = c["sanity"]
                print("  %s fx %.1f fy %.1f cx %.1f cy %.1f  image-up vs +z %.1f deg  axis elevation %.1f deg  "
                      "geometry in front %.0f %% / in view %.0f %%" % (
                          c["name"], c["fx"], c["fy"], c["cx"], c["cy"], sn["up_angle_deg"],
                          sn["optical_axis_elevation_deg"], 100 * sn["in_front_share"], 100 * sn["in_view_share"]))
            else:
                print("  %s %s: %s" % (c.get("name"), c.get("status"), "; ".join(c.get("reasons", []))))
        cd = (rep.get("stats") or {}).get("column_depth")
        if cd:
            print("  column depth: at most %d collision geoms over one %.2f m square (cap %d)" % (
                cd["max"], cd["cell_m"] or 0.0, DEFAULT_RULES["max_column_depth"]))
        for g in rep.get("gates", []):
            tr = g.get("travel") or {}
            print("  gate %-20s joint %-16s travel rule %s" % (g["body"], g["joint"], "ok" if tr.get("ok") else
                                                               "NOT met (see the notes)"))
        for it in rep.get("units", []):
            line = "  unit %-24s %-18s %s" % (it["body"], it.get("label"), it["status"])
            if "density_kg_m3" in it:
                line += "  density %.1f kg/m^3" % it["density_kg_m3"]
            print(line)
            for r in it.get("reasons", []):
                print("      - " + r)
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
